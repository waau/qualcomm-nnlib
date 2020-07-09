
/*
 * Copyright (c) 2016-2020, The Linux Foundation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted (subject to the limitations in the
 * disclaimer below) provided that the following conditions are met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided
 *      with the distribution.
 *
 *    * Neither the name of The Linux Foundation nor the names of its
 *      contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *
 * NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
 * GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
 * HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 * IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
 * IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */
#include <nn_graph.h>
#include <string.h>
#include "hvx_inlines.h"
#include "quantize.h"

#if defined(__hexagon__)
#include "hexagon_types.h"
#endif

#define DIV_EPISILON 1e-6f
#define ALIGN_SIZE 4

#ifdef HEXAGON_V66
#define CONVERT_MAX_THREADS 4
#else
#define CONVERT_MAX_THREADS 2
#endif

/*
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * 
 * This contains an elementwise div op for 1-D denominators
 */

struct mapdata {
    const uint8_t *in_data;
    uint8_t *map_data;
    uint8_t *out_data;
    nn_sem_t donesem;
    int num_elements;
};

struct flipdata {
    const uint8_t *in_data;
    uint8_t *out_data;
    nn_sem_t donesem;
    int num_elements;
};

struct convert_state {
    int32_t n_batchinputs;	// actual number of batches
    float out_min;	// output range
    float out_level_recip;
    volatile int32_t input_next; // input to process next batch.
    uint32_t out_stride;
    const uint8_t *input_data;
    uint8_t *output_data;
    float *min_array;
    float *max_array;
    nn_sem_t done_sem;
};

static void map_values(struct nn_graph *nn, void *vtd)
{
    struct mapdata *td = vtd;
    uint8_t* in_data = (uint8_t*) td->in_data;
    uint8_t* out_data = td->out_data;
    unsigned char* map_data = td->map_data;

    const int num_loops = 1 + ((td->num_elements - 1) / 128); //ceiling

    for (int i=0; i<num_loops; i++) {
        HVX_Vector vin = *(HVX_Vector *) in_data;
        HVX_Vector *vout = (HVX_Vector *) out_data;
        // byte shuffle table
        HVX_Vector luta = *(HVX_Vector *) map_data;
        HVX_Vector lutb = *(HVX_Vector *) & map_data[128];
        HVX_Vector lut0 = Q6_Vb_vshuff_Vb(luta);
        HVX_Vector lut1 = Q6_Vb_vshuff_Vb(lutb);

        // look up value in table
        // only 32 bytes can be done at a time, so we need to do 8 lookups and OR the results

        *vout = q6op_Vb_vlut32_VbVbI(vin, lut0, 0);
        *vout = q6op_Vb_vlut32or_VbVbVbI(*vout, vin, lut0, 1);
        *vout = q6op_Vb_vlut32or_VbVbVbI(*vout, vin, lut0, 2);
        *vout = q6op_Vb_vlut32or_VbVbVbI(*vout, vin, lut0, 3);
        *vout = q6op_Vb_vlut32or_VbVbVbI(*vout, vin, lut1, 4);
        *vout = q6op_Vb_vlut32or_VbVbVbI(*vout, vin, lut1, 5);
        *vout = q6op_Vb_vlut32or_VbVbVbI(*vout, vin, lut1, 6);
        *vout = q6op_Vb_vlut32or_VbVbVbI(*vout, vin, lut1, 7);
        // move pointers to process next 128 bytes
        in_data += 128;
        out_data += 128;
    }

    nn_sem_post(&td->donesem);
}

static void flip_values(struct nn_graph *nn, void *vtd)
{
    struct flipdata *td = vtd;
    uint8_t* in_data = (uint8_t*) td->in_data;
    uint8_t* out_data = td->out_data;
    
    HVX_Vector max_value = Q6_V_vsplat_R(0xFFFFFFFF);

    const int num_loops = 1 + ((td->num_elements - 1) / 128); //ceiling

    for (int i=0; i<num_loops; i++) {
        HVX_Vector vin = q6op_V_vldu_A( (HVX_Vector *)in_data );
        HVX_Vector *voutp = (HVX_Vector*) out_data;

        HVX_Vector vout = Q6_Vub_vsub_VubVub_sat(max_value, vin);
        q6op_vstu_AV(voutp,vout);

        in_data += 128;
        out_data += 128;
    }

    nn_sem_post(&td->donesem);
}

static int div_depthwise_execute(struct nn_node *self, struct nn_graph *nn)
{
    logmsg(nn,2,"div depthwise execute. self=%p ",self);
    const struct tensor *a_tensor = self->inputs[0];
    const struct tensor *b_tensor = self->inputs[1];
    const struct tensor *a_min_tensor = self->inputs[2];
    const struct tensor *a_max_tensor = self->inputs[3];
    const struct tensor *b_min_tensor = self->inputs[4];
    const struct tensor *b_max_tensor = self->inputs[5];
    struct tensor *out_tensor = self->outputs[0];
    struct tensor *out_min_tensor = self->outputs[1];
    struct tensor *out_max_tensor = self->outputs[2];
    uint8_t *a_data = a_tensor->data;
    uint8_t *b_data = b_tensor->data;
    float a_min_float = tensor_get_float(a_min_tensor,0);
    float a_max_float = tensor_get_float(a_max_tensor,0);
    float a_step = (a_max_float - a_min_float) / 255.f;
    float b_min_float = tensor_get_float(b_min_tensor,0);
    float b_max_float = tensor_get_float(b_max_tensor,0);
    float b_step = (b_max_float - b_min_float) / 255.f;
    uint8_t *out_data = out_tensor->data;
    float *out_min = out_min_tensor->data;
    float *out_max = out_max_tensor->data;

    int elements = a_tensor->shape.batches * a_tensor->shape.height * a_tensor->shape.width * a_tensor->shape.depth;
    // the height and width of denominator tensor b should equal to 1
    int denominator_size = b_tensor->shape.batches * b_tensor->shape.depth;
    if(b_tensor->shape.batches != 1){
        if(a_tensor->shape.height != 1 || a_tensor->shape.width != 1){
            return errlog(nn,"only support elementwise div when both batch and depth not equal to 1");
        }
    }

    if(self->n_inputs > 6){
        *out_min = tensor_get_float(self->inputs[6],0);
        *out_max = tensor_get_float(self->inputs[7],0);
    }
    else{
        *out_min = 2147483648.0f;
        *out_max = -2147483648.0f;

        for(int d = 0; d < denominator_size; d++){
            float denominator = b_min_float + ((float)b_data[d]) * b_step;
            if(denominator == 0.f){
                logmsg(nn,0,"Problem - divided by zero, add Episilon");
                denominator = DIV_EPISILON;
            }
            if (denominator > 0){
                if(a_max_float/denominator > *out_max) *out_max = a_max_float/denominator;
                if(a_min_float/denominator < *out_min) *out_min = a_min_float/denominator;
            }
            else{
                if(a_min_float/denominator > *out_max) *out_max = a_min_float/denominator;
                if(a_max_float/denominator < *out_min) *out_min = a_max_float/denominator;
            }
        }
        if (*out_min == -0.f ){*out_min = 0;}
        if (*out_max == -0.f ){*out_max = 0;}
        if(*out_min < 0 && *out_max < 0) *out_max = 0;
        if(*out_min > 0 && *out_max > 0) *out_min = 0;
    }
    logmsg(nn,2,"div out min/max=%f, %f ",*out_min, *out_max);

    tensor_out_prepare_normal(out_min_tensor, 1, 1, 1, 1, NN_TYPE_FLOAT);
    tensor_out_prepare_normal(out_max_tensor, 1, 1, 1, 1, NN_TYPE_FLOAT);
    tensor_out_prepare_normal(out_tensor, a_tensor->shape.batches, a_tensor->shape.height, a_tensor->shape.width, a_tensor->shape.depth, NN_TYPE_QUINT8);

    // For each depth:
    //   1. Calculate a map from the depths' true range to quantized output
    //   2. Go through the values at that depth and map them to the output
    uint8_t map[256];
    for(int d = 0; d < denominator_size; d++){
        float denominator = b_min_float + ((float)b_data[d]) * b_step;
        if(denominator == 0.f){
            logmsg(nn,0,"Problem - divided by zero, add Episilon");
            denominator = DIV_EPISILON;
        }
        // Calculate map
        for(int i = 0; i <= 255 ; i++){
            float numerator = a_min_float + ((float)i) * a_step;
            float quotient = numerator / denominator;
            map[i] = quantize_uint8(quotient, *out_min, *out_max);
        }

        // Map the input to the output
        for(int i = d; i < elements; i+=denominator_size){
            out_data[i] = map[a_data[i]];
        }
    }

    return 0;
}

static void convert_work(
    struct nn_graph *nn,
    void *thrinfo)
{
    struct convert_state *thrdesc = (struct convert_state *)thrinfo;
    float *min_batch = thrdesc->min_array;
    float *max_batch = thrdesc->max_array;

    int32_t out_stride = thrdesc->out_stride;
    int32_t outer_count = 1;

    float out_min = thrdesc->out_min;
    float out_level_recip = thrdesc->out_level_recip;
    int32_t jobid = 0;
    while (jobid = __sync_fetch_and_add(&thrdesc->input_next, 1), jobid < thrdesc->n_batchinputs) {
        const uint8_t *in_data = thrdesc->input_data;
        uint8_t* out_data = thrdesc->output_data;
        in_data += jobid*out_stride;
        out_data += jobid*out_stride;
        uint32_t copylen = out_stride;

        l2fetch(in_data, copylen * sizeof(uint16_t), copylen * sizeof(uint16_t), outer_count);

        float in_min = min_batch[jobid];
        float in_max = max_batch[jobid];
        in_min = fminf(0.0f, in_min); // in_min <= 0.0f
        float in_level = flt_div_255(in_max-in_min);

        int32_t offset = max_i32(0, roundf_i32((in_min - out_min)/in_level));
        int32_t gaint = roundf_i32(out_level_recip*in_level* 32768.0f);
        int32_t gain = min_i32(32767, gaint);

        if( offset != 0 || gain < 0x7fc0) {    // scale the input into common range
            memconvert_hvx(
                out_data,
                in_data,
                copylen,
                offset,
                gain,
                out_stride,
                outer_count);
        }
        else {                                 // is unity gain (0->0, 255->255)
            vmemcpy_2d_general_asm(
                copylen,                       // bytes wide
                outer_count,                   // rows
                out_data,                      // destination address, any allowed
                out_stride,                    // row pitch of dest; any allowed
                in_data,                       // source address, any allowed
                copylen);                      // source stride, any
        }
    }

    // signal complete in thread.
    nn_sem_post(&thrdesc->done_sem);
}

// If denominator tensor shape is (batches, 1, 1, 1), we could do min/max
// scale division (like scalar divide) for each batch data and then convert
// batch data to unify min/max with hvx convert
static int div_batchwise_execute(struct nn_node *self, struct nn_graph *nn)
{
    logmsg(nn,2,"div batchwise execute. self=%p ",self);
    const struct tensor *input_tensor = self->inputs[0];
    uint8_t *input_data = input_tensor->data;
    float in_min = tensor_get_float(self->inputs[2],0);
    float in_max = tensor_get_float(self->inputs[3],0);

    const struct tensor *b_tensor = self->inputs[1];

    uint8_t *b_data = b_tensor->data;
    uint32_t batch_size = b_tensor->shape.batches;
    float b_min = tensor_get_float(self->inputs[4],0);
    float b_max = tensor_get_float(self->inputs[5],0);
    float b_step = (b_max - b_min) / 255.f;

    struct tensor *out_tensor = self->outputs[0];
    uint8_t *out_data = out_tensor->data;

    struct tensor *out_min_tensor = self->outputs[1];
    struct tensor *out_max_tensor = self->outputs[2];

    float *out_min = out_min_tensor->data;
    float *out_max = out_max_tensor->data;
    *out_min = 0.0f;
    *out_max = 0.0f;

    tensor_out_prepare_normal(out_min_tensor, 1, 1, 1, 1, NN_TYPE_FLOAT);
    tensor_out_prepare_normal(out_max_tensor, 1, 1, 1, 1, NN_TYPE_FLOAT);
    tensor_out_prepare_normal(out_tensor, input_tensor->shape.batches, input_tensor->shape.height,
                              input_tensor->shape.width, input_tensor->shape.depth, NN_TYPE_QUINT8);

    uint32_t batch_elements = input_tensor->shape.height*input_tensor->shape.width*input_tensor->shape.depth;
    uint8_t * div_numerator_ptr = input_data;
    uint8_t * convert_buffer = nn->scratch;
    size_t element_offset = (batch_elements*batch_size+ALIGN_SIZE-1)&~(ALIGN_SIZE-1);
    float * out_min_batch = (float *)((uint8_t *)convert_buffer + element_offset);
    float * out_max_batch = (float *)((uint8_t *)out_min_batch + batch_size*sizeof(float));

    for (uint32_t i=0; i<batch_size; i++) {
        float scalar = b_min + ((float)b_data[i]) * b_step;
        if(scalar == 0.f) {
            logmsg(nn,0,"Problem - divided by zero, add Episilon");
            scalar = DIV_EPISILON;
        }
        // if the quotient is positive then we simply divide the min and max
        // and copy the data as-is
        // otherwise, we divide the min and max, swap them and then flip
        // the data around 128 so that x becomes 255-x
        if(scalar > 0.f){
            out_min_batch[i] = in_min / scalar;
            out_max_batch[i] = in_max / scalar;
            // copy
            memcpy(convert_buffer, div_numerator_ptr, batch_elements);
        }
        else{
            out_min_batch[i] = in_max / scalar;
            out_max_batch[i] = in_min / scalar;

            int elements_128aligned = batch_elements / 128;
            elements_128aligned *= 128;

            struct flipdata td = {
                    .in_data = div_numerator_ptr,
                    .out_data = convert_buffer,
                    .num_elements = elements_128aligned
            };
            nn_sem_init(&td.donesem,0);
            nn_os_work_for_vector(nn,flip_values,&td);
            nn_sem_wait(&td.donesem);

            for(int i = elements_128aligned; i < batch_elements; i++) {
                convert_buffer[i] = 255 - div_numerator_ptr[i];
            }
        }
        // find out the min/max among batch min/max data
        *out_min = fminf(*out_min, out_min_batch[i]);
        *out_max = fmaxf(*out_max, out_max_batch[i]);
        div_numerator_ptr += batch_elements;
        convert_buffer += batch_elements;
    }

    float out_level_recip = 255.0f / (*out_max - *out_min);

    struct convert_state rundesc;
    // fire the threads
    rundesc.n_batchinputs = batch_size;
    rundesc.out_stride = batch_elements;
    rundesc.out_min = *out_min;
    rundesc.out_level_recip = out_level_recip;
    rundesc.input_next = 0;
    nn_sem_init(&rundesc.done_sem, 0);

    rundesc.input_data = (uint8_t *)nn->scratch;
    rundesc.min_array = (float *)out_min_batch;
    rundesc.max_array = (float *)out_max_batch;
    rundesc.output_data = out_data;
    // convert data to unify min/max
    int32_t num_actual_threads = min_i32(CONVERT_MAX_THREADS, batch_size);
    for (int32_t i = 0; i < num_actual_threads; i++) {
        nn_os_work_for_vector(nn, convert_work, &rundesc);
    }
    nn_sem_wait_n_times(&rundesc.done_sem, num_actual_threads);

    return 0;
}

static int div_scalar_execute(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"div execute. self=%p ",self);

	const struct tensor *input_tensor = self->inputs[0];
	uint8_t *input_data = input_tensor->data;
	float in_min = tensor_get_float(self->inputs[2],0);
	float in_max = tensor_get_float(self->inputs[3],0);
	
	const struct tensor *b_tensor = self->inputs[1];
	
	uint8_t *b_data = b_tensor->data;
	float b_min = tensor_get_float(self->inputs[4],0);
	float b_max = tensor_get_float(self->inputs[5],0);
	float b_step = (b_max - b_min) / 255.f;

	float scalar = b_min + ((float)b_data[0]) * b_step;
	if(scalar == 0.f) {
        logmsg(nn,0,"Problem - divided by zero, add Episilon");
        scalar = DIV_EPISILON;
	}
	
	struct tensor *out_tensor = self->outputs[0];
	uint8_t *out_data = out_tensor->data;
	
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];
	
	float *out_min = out_min_tensor->data;
	float *out_max = out_max_tensor->data;

	tensor_out_prepare_normal(out_min_tensor, 1, 1, 1, 1, NN_TYPE_FLOAT);
	tensor_out_prepare_normal(out_max_tensor, 1, 1, 1, 1, NN_TYPE_FLOAT);

	// if the quotient is positive then we simply divide the min and max
	// and copy the data as-is
	// otherwise, we divide the min and max, swap them and then flip
	// the data around 128 so that x becomes 255-x
	if(scalar > 0.f){
		out_min[0] = in_min / scalar;
		out_max[0] = in_max / scalar;

		tensor_copy(self->outputs[0],self->inputs[0]);
	}
	else{
		tensor_out_prepare_normal(out_tensor, input_tensor->shape.batches, input_tensor->shape.height, input_tensor->shape.width, input_tensor->shape.depth, NN_TYPE_QUINT8);
		
		out_min[0] = in_max / scalar;
		out_max[0] = in_min / scalar;

		int elements = input_tensor->shape.batches * input_tensor->shape.height * input_tensor->shape.width * input_tensor->shape.depth;
		int elements_128aligned = elements / 128;
		elements_128aligned *= 128;

		struct flipdata td = {
				.in_data = input_data,
				.out_data = out_data,
				.num_elements = elements_128aligned
		};
		nn_sem_init(&td.donesem,0);
		nn_os_work_for_vector(nn,flip_values,&td);
		nn_sem_wait(&td.donesem);

		for(int i = elements_128aligned; i < elements; i++) out_data[i] = 255 - input_data[i];
	}

	return 0;
}

static int div_scalar_static_minmax_execute(struct nn_node *self, struct nn_graph *nn)
{
    logmsg(nn,2,"div execute. self=%p ",self);

    const struct tensor *input_tensor = self->inputs[0];
    const struct tensor *b_tensor = self->inputs[1];

    struct tensor *out_tensor = self->outputs[0];
    struct tensor *out_min_tensor = self->outputs[1];
    struct tensor *out_max_tensor = self->outputs[2];

    uint8_t *input_data = input_tensor->data;
    float in_min = tensor_get_float(self->inputs[2],0);
    float in_max = tensor_get_float(self->inputs[3],0);
    float in_step = (in_max - in_min) / 255.f;

    uint8_t *b_data = b_tensor->data;
    float b_min = tensor_get_float(self->inputs[4],0);
    float b_max = tensor_get_float(self->inputs[5],0);
    float b_step = (b_max - b_min) / 255.f;

    float static_min = tensor_get_float(self->inputs[6],0);
    float static_max = tensor_get_float(self->inputs[7],0);

    uint8_t *out_data = out_tensor->data;
    float *out_min = out_min_tensor->data;
    float *out_max = out_max_tensor->data;

    float scalar = b_min + ((float)b_data[0]) * b_step;
    if(scalar == 0.f) return errlog(nn,"division by zero");

    tensor_out_prepare_normal(out_tensor, input_tensor->shape.batches, input_tensor->shape.height, input_tensor->shape.width, input_tensor->shape.depth, NN_TYPE_QUINT8);
    tensor_out_prepare_normal(out_min_tensor, 1, 1, 1, 1, NN_TYPE_FLOAT);
    tensor_out_prepare_normal(out_max_tensor, 1, 1, 1, 1, NN_TYPE_FLOAT);

    out_min[0] = static_min;
    out_max[0] = static_max;

    unsigned char div_lookup[256] __attribute__ ((aligned(128)));

    for(int i = 0; i < 256; i++){
        float float_value = (in_min + ((float)i) * in_step) / scalar;
        div_lookup[i] = quantize_uint8(float_value, static_min, static_max);
    }

    int elements = input_tensor->shape.batches * input_tensor->shape.height * input_tensor->shape.width * input_tensor->shape.depth;
    int elements_128aligned = elements / 128;
    elements_128aligned *= 128;

    struct mapdata td = {
            .in_data = input_data,
            .map_data = div_lookup,
            .out_data = out_data,
            .num_elements = elements_128aligned
    };
    nn_sem_init(&td.donesem,0);
    nn_os_work_for_vector(nn,map_values,&td);
    nn_sem_wait(&td.donesem);

    for(int i = elements_128aligned; i < elements; i++) out_data[i] = div_lookup[input_data[i]];

    return 0;
}

static int div_execute(struct nn_node *self, struct nn_graph *nn){

    if( self->inputs[1]->shape.batches == 1 &&
        self->inputs[1]->shape.depth == 1 ){
        if(self->n_inputs == 6){
            return div_scalar_execute(self, nn);
        } else {
            return div_scalar_static_minmax_execute(self, nn);
        }
    } else if (self->inputs[1]->shape.depth == 1) {
        if (self->inputs[0]->shape.batches != self->inputs[1]->shape.batches)
            return errlog(nn,"input batch size mismatch for broadcast");
        return div_batchwise_execute(self, nn);
    }
    else {
        return div_depthwise_execute(self, nn);
    }
}

static int div_check(struct nn_node *self, struct nn_graph *nn){
    if(self->n_inputs != 6 && self->n_inputs != 8)
        return errlog(nn,"must have 6 or 8 inputs");
    if(self->inputs[1]->shape.height != 1
       || self->inputs[1]->shape.width != 1){
        return errlog(nn,"op only supported for scalar and 1d tensors");
    }
    if(self->inputs[1]->shape.batches != 1){
        if (self->inputs[0]->shape.batches!=self->inputs[1]->shape.batches) {
            return errlog(nn,"div batch shape mismatch");
        }
        if (self->inputs[1]->shape.depth != 1) {
            if(self->inputs[0]->shape.height != 1
               || self->inputs[0]->shape.width != 1){
                return errlog(nn,"op only support elementwise div when both batch and depth not equal to 1");
            }
        }
        uint32_t data_maxsize = (self->inputs[0]->max_size+ALIGN_SIZE-1)&~(ALIGN_SIZE-1);
        data_maxsize += 2*self->inputs[1]->shape.batches*sizeof(float);
        if (nn_scratch_grow(nn,data_maxsize)) {
            return errlog(nn,"couldn't allocate buffer for div batchwise operation");
        }
    }
    logmsg(nn,2,"div node %p check OK",self);
    return 0;
}

struct nn_node_ops nn_ops_for_QuantizedDiv_8 = {
        .execute = div_execute,
        .check = div_check,
        .ctor = node_alloc_common,
        .dtor = node_free_common,
        .n_inputs = NN_IOCOUNT_RANGE(6,8),
        .n_outputs = NN_IOCOUNT(3),
};

