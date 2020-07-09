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


/*
 * Includes code used with permission of University of Freiburg.
 */
#include <nn_graph.h>
#include <quantize.h>
#include "hvx_inlines.h"
#include "hexagon_types.h"
#include <math.h>
#include <limits.h>

static inline void __attribute__((always_inline))
l2fetch_linear( uint8_t const * addr, unsigned len){
	unsigned misalign = (size_t)addr & 127;
	int vec_count = (len + misalign + 127)>>7;		// exact # of vectors needed
	l2fetch( addr-misalign, 128,128, vec_count);
}

#define VELEM(elem_size)  (sizeof(HVX_Vector) / elem_size)

#ifndef TRUE
#define TRUE 1
#endif
#ifndef FALSE
#define FALSE 0
#endif
#ifndef BOOL
#define BOOL int32_t 
#endif


#ifdef HEXAGON_V66
#define CORRELATION1D_MAX_THREADS 4
#else
#define CORRELATION1D_MAX_THREADS 2
#endif


typedef enum {
	LEFT  = 0,
	RIGHT = 1,
	FIX,
} ShiftType;


typedef struct
{	
	uint8_t thread_id;
	uint8_t num_threads;
	uint8_t *a_data;
	uint8_t *b_data;
	uint8_t *out;
	uint32_t a_batches;
	uint32_t a_height;
	uint32_t a_width;
	uint32_t a_depth;
	uint32_t b_batches;
	uint32_t b_height;
	uint32_t b_width;
	uint32_t b_depth;
	uint32_t out_batches;
	uint32_t out_height;
	uint32_t out_width;
	uint32_t out_depth;
	uint32_t a_batch_stride;
	uint32_t a_height_stride;
	uint32_t b_batch_stride;
	uint32_t b_height_stride;
	uint32_t out_batch_stride;
	uint32_t out_height_stride;

	int32_t displacement_left;
	int32_t shift;
	int32_t a_offset;
	int32_t b_offset;
	int32_t out_offset;
	int32_t in_depth_iterations;
	int32_t in_depth_residue;

	int32_t quantized_multiplier;
	int32_t shift_bit;
	ShiftType direction;

	volatile int32_t jobno;		// volatile int
	int32_t jobcount;
	int32_t inner_count;
	nn_sem_t done_sem;
} correlation1d_param;


//Common Function
int32_t QuantizeMultiplierLessThanOne(double double_multiplier,
	int32_t* quantized_multiplier,
	int32_t* right_shift) {
	if (double_multiplier == 0.) {
		*quantized_multiplier = 0;
		*right_shift = 0;
		return 0;
	}
	const double q = frexp(double_multiplier, (int*)right_shift);
	*right_shift *= -1;
	int64_t q_fixed = (int64_t)(round(q * (1ll << 31)));
	if (q_fixed == (1ll << 31)) {
		q_fixed /= 2;
		--*right_shift;
	}
	*quantized_multiplier = (int32_t)(q_fixed);
	return 0;
}


int32_t QuantizeMultiplierGreaterThanOne(double double_multiplier,
	int32_t* quantized_multiplier,
	int32_t* left_shift) {
	const double q = frexp(double_multiplier, (int*)left_shift);
	int64_t q_fixed = (int64_t)(round(q * (1ll << 31)));
	if (q_fixed == (1ll << 31)) {
		q_fixed /= 2;
		++*left_shift;
	}
	*quantized_multiplier = (int32_t)(q_fixed);
	return 0;
}


static void correlation1d_ref(uint8_t* a_data, uint8_t* b_data, uint8_t* out_data,
						int32_t a_width, int32_t a_height, int32_t a_depth, int32_t b_width,
						int32_t b_height, int32_t b_depth, int32_t out_batches, int32_t out_width,
						int32_t out_height, int32_t out_depth, int32_t a_offset, int32_t b_offset,
						int out_offset, int32_t quantized_multiplier, int32_t shift_bit, ShiftType direction)					
{
	for(int32_t b = 0; b < out_batches; b++) {
		for(int32_t h = 0; h < out_height; h++) {
			for(int32_t w = 0; w < out_width; w++) {
				const uint8_t* p_a = a_data + a_depth * (w + a_width * (h + a_height * (b)));
				const uint8_t* p_b = b_data + b_depth * (w + b_width * (h + b_height * (b)));
				for(int32_t d = 0; d < out_depth; d++) {
					int32_t sum = 0;
					const uint8_t* a_loc = p_a;
					const uint8_t* b_loc = p_b + d * b_depth;
					for(int32_t i = 0; i < a_depth; i++){
						sum += ((int32_t)a_loc[i] - a_offset) * ((int32_t)b_loc[i] - b_offset);
					}
					if(direction == LEFT) {
						sum = ((long long int)sum * quantized_multiplier) >> (31 - shift_bit);
					} else if(direction == RIGHT) {
						sum = ((long long int)sum * quantized_multiplier) >> (31 + shift_bit);
					}
					sum = sum / a_depth;
					sum += out_offset;
					sum = sum < 0 ? 0 : sum;
					sum = sum > 255 ? 255 : sum;

					out_data[d + (out_depth * (w + out_width * (h + out_height * (b))))] = (uint8_t)sum;
				}
			}
		}
	}
}


static void correlation1d_hvx(uint8_t* a_data, uint8_t* b_data, uint8_t* out_data,
						int32_t a_depth, int32_t b_depth, int32_t out_width, int32_t out_depth,
						int32_t a_offset, int32_t b_offset, int out_offset,
						int32_t in_depth_iterations, int32_t in_depth_residue,
						int32_t quantized_multiplier, int32_t shift_bit, ShiftType direction)
{
	union { HVX_Vector as_v_tmp; int32_t as_i32_tmp[32];} vector_scalars_tmp;
	//using mask to indicate the valid data
	HVX_VectorPred Qvalid_data_mask;

	HVX_Vector a_vec_offset = q6op_Vb_vsplat_R(a_offset);
	HVX_Vector b_vec_offset = q6op_Vb_vsplat_R(b_offset);
	HVX_Vector* p_a_vec = (HVX_Vector*) a_data;
	HVX_Vector* p_b_vec = (HVX_Vector*) b_data;

	for(int32_t w = 0; w < out_width; w++) {
		for(int32_t d = 0; d < out_depth; d++) {
			uint8_t* b_position = (uint8_t*)b_data + d * b_depth;
			int32_t result = 0;
			p_a_vec = (HVX_Vector*) a_data;
			p_b_vec = (HVX_Vector*) b_position;

			for(int32_t i = 0; i < in_depth_iterations; i++)
			{
				HVX_VectorPair a_sub = Q6_Wh_vsub_VubVub(q6op_V_vldu_A((HVX_Vector const*)p_a_vec++), a_vec_offset); //keep invalid sub result to zero
				HVX_VectorPair b_sub = Q6_Wh_vsub_VubVub(q6op_V_vldu_A((HVX_Vector const*)p_b_vec++), b_vec_offset);
				HVX_Vector lo_mul = Q6_Vw_vdmpy_VhVh_sat(Q6_V_lo_W(a_sub), Q6_V_lo_W(b_sub));
				HVX_Vector hi_mul = Q6_Vw_vdmpy_VhVh_sat(Q6_V_hi_W(a_sub), Q6_V_hi_W(b_sub));

				HVX_VectorPair dadd = Q6_W_vshuff_VVR(hi_mul,lo_mul, 4);
				lo_mul = Q6_Vw_vadd_VwVw(Q6_V_hi_W(dadd), Q6_V_lo_W(dadd));
				dadd = Q6_W_vshuff_VVR(lo_mul,lo_mul, 8);
				lo_mul = Q6_Vw_vadd_VwVw(Q6_V_hi_W(dadd), Q6_V_lo_W(dadd));
				dadd = Q6_W_vshuff_VVR(lo_mul,lo_mul, 16);
				lo_mul = Q6_Vw_vadd_VwVw(Q6_V_hi_W(dadd), Q6_V_lo_W(dadd));
				dadd = Q6_W_vshuff_VVR(lo_mul,lo_mul,32);
				lo_mul = Q6_Vw_vadd_VwVw(Q6_V_hi_W(dadd), Q6_V_lo_W(dadd));
				dadd = Q6_W_vshuff_VVR(lo_mul,lo_mul, 64);
				lo_mul = Q6_Vw_vadd_VwVw(Q6_V_hi_W(dadd), Q6_V_lo_W(dadd));
				vector_scalars_tmp.as_v_tmp = lo_mul;
				Q6_dcfetch_A(&vector_scalars_tmp);
				result +=vector_scalars_tmp.as_i32_tmp[0];
				result +=vector_scalars_tmp.as_i32_tmp[1];
			}

			if(in_depth_residue != 0)
			{
				Qvalid_data_mask = q6op_Q_vsetq2_R(in_depth_residue);
				HVX_VectorPair a_sub = Q6_Wh_vsub_VubVub(Q6_V_vmux_QVV(Qvalid_data_mask, q6op_V_vldu_A((HVX_Vector const*)p_a_vec++), a_vec_offset), a_vec_offset); //keep invalid sub result to zero
				HVX_VectorPair b_sub = Q6_Wh_vsub_VubVub(Q6_V_vmux_QVV(Qvalid_data_mask, q6op_V_vldu_A((HVX_Vector const*)p_b_vec++), a_vec_offset), b_vec_offset);
				HVX_Vector lo_mul = Q6_Vw_vdmpy_VhVh_sat(Q6_V_lo_W(a_sub), Q6_V_lo_W(b_sub));
				HVX_Vector hi_mul = Q6_Vw_vdmpy_VhVh_sat(Q6_V_hi_W(a_sub), Q6_V_hi_W(b_sub));

				HVX_VectorPair dadd = Q6_W_vshuff_VVR(hi_mul,lo_mul, 4);
				lo_mul = Q6_Vw_vadd_VwVw(Q6_V_hi_W(dadd), Q6_V_lo_W(dadd));
				dadd = Q6_W_vshuff_VVR(lo_mul,lo_mul, 8);
				lo_mul = Q6_Vw_vadd_VwVw(Q6_V_hi_W(dadd), Q6_V_lo_W(dadd));
				dadd = Q6_W_vshuff_VVR(lo_mul,lo_mul, 16);
				lo_mul = Q6_Vw_vadd_VwVw(Q6_V_hi_W(dadd), Q6_V_lo_W(dadd));
				dadd = Q6_W_vshuff_VVR(lo_mul,lo_mul,32);
				lo_mul = Q6_Vw_vadd_VwVw(Q6_V_hi_W(dadd), Q6_V_lo_W(dadd));
				dadd = Q6_W_vshuff_VVR(lo_mul,lo_mul, 64);
				lo_mul = Q6_Vw_vadd_VwVw(Q6_V_hi_W(dadd), Q6_V_lo_W(dadd));
				vector_scalars_tmp.as_v_tmp = lo_mul;
				Q6_dcfetch_A(&vector_scalars_tmp);
				result +=vector_scalars_tmp.as_i32_tmp[0];
				result +=vector_scalars_tmp.as_i32_tmp[1];
			}

			b_position = (uint8_t*)b_position + b_depth;

			if(direction == LEFT) {
				result = ((long long int)result * quantized_multiplier) >> (31 - shift_bit);
			} else if(direction == RIGHT) {
				result = ((long long int)result * quantized_multiplier) >> (31 + shift_bit);
			}

			result = result / (int32_t)a_depth;
			result += out_offset;
			result = result < 0 ? 0 :(result > 255 ? 255 : result);
			*out_data++ = result;
		}

		 a_data = a_data + a_depth;
		 b_data = b_data + b_depth;
	}
}


static void correlation1d_thread_work(struct nn_graph *nn, void *work_info) {
	correlation1d_param* work = work_info;	
	uint8_t *a_data = work->a_data;
	uint8_t *b_data = work->b_data;
	uint8_t  *out_data = work->out;

	uint32_t a_depth = work->a_depth;
	uint32_t b_depth = work->b_depth;
	uint32_t out_width = work->out_width;
	uint32_t out_depth = work->out_depth;

	int32_t a_offset = work->a_offset;
	int32_t b_offset = work->b_offset;
	int32_t out_offset = work->out_offset;
	int32_t in_depth_iterations = work->in_depth_iterations;
	int32_t in_depth_residue = work->in_depth_residue;

	ShiftType direction = work->direction;
	int32_t quantized_multiplier = work->quantized_multiplier;
	int32_t shift_bit = work->shift_bit;

	int jobno;
	batchslice_decode bsdecode;
	batchslice_decode_init(&bsdecode, work->inner_count);
	
	while(jobno = __sync_fetch_and_add(&work->jobno, 1), jobno < work->jobcount) {
		int height_idx = batchslice_decode_update(&bsdecode, jobno);
		int batch_idx = bsdecode.ibatch;
		a_data = work->a_data + batch_idx * work->a_batch_stride + height_idx * work->a_height_stride;
		b_data = work->b_data + batch_idx * work->b_batch_stride + height_idx * work->b_height_stride;
		out_data = work->out + batch_idx * work->out_batch_stride + height_idx * work->out_height_stride;
		l2fetch_linear(a_data, work->a_height_stride);
		l2fetch_linear(b_data, work->b_height_stride);

		correlation1d_hvx(a_data, b_data, out_data, a_depth, b_depth, out_width, out_depth, a_offset, b_offset, out_offset,
				in_depth_iterations, in_depth_residue, quantized_multiplier, shift_bit, direction);

	}
	nn_sem_post(&work->done_sem);
}


static int correlation1d_prepare(struct nn_node *self, struct nn_graph *nn, struct nn_node *predecessor, 
								 BOOL for_reference)
{
	logmsg(nn,2,"correlation1d node %p preparation",self);
	const struct tensor* a_tensor = self->inputs[0];
	const struct tensor* b_tensor = self->inputs[1];
	const struct tensor* min_a_tensor = self->inputs[2];
	const struct tensor* max_a_tensor = self->inputs[3];
	const struct tensor* min_b_tensor = self->inputs[4];
	const struct tensor* max_b_tensor = self->inputs[5];
	const struct tensor* displacement_tensor = self->inputs[6];
	const struct tensor* shift_tensor = self->inputs[7];
	const struct tensor* min_out_tensor = self->inputs[8];
	const struct tensor* max_out_tensor = self->inputs[9];
	struct tensor* out_tensor = self->outputs[0];
	struct tensor* out_min = self->outputs[1];
	struct tensor* out_max = self->outputs[2];

	uint8_t* a_data = (uint8_t*)a_tensor->data;
	uint8_t* b_data = (uint8_t*)b_tensor->data;
	uint8_t* out = (uint8_t*)out_tensor->data;
	int32_t displacement = tensor_get_int32(displacement_tensor, 0);
	int32_t shift = tensor_get_int32(shift_tensor, 0);
   
	float a_max_float = tensor_get_float(max_a_tensor, 0);
	float a_min_float = tensor_get_float(min_a_tensor, 0);
	float b_max_float = tensor_get_float(max_b_tensor, 0);
	float b_min_float = tensor_get_float(min_b_tensor, 0);
	float out_max_float = tensor_get_float(max_out_tensor, 0);
	float out_min_float = tensor_get_float(min_out_tensor, 0);

	int32_t a_offset = quantize_int(0.0f, a_min_float, a_max_float);
	int32_t b_offset = quantize_int(0.0f, b_min_float, b_max_float);
	int32_t out_offset = quantize_int(0.0f, out_min_float, out_max_float);

	if(tensor_set_single_float(out_min, out_min_float) != 0
		|| tensor_set_single_float(out_max, out_max_float) != 0){
		return errlog(nn, "min or max too small");
	}

	float scale_A = (a_max_float - a_min_float) / 255;					
	float scale_B = (b_max_float - b_min_float) / 255;					
	float scale_Out = (out_max_float - out_min_float) / 255;

	float scale = scale_A * scale_B / scale_Out;

	int32_t quantized_multiplier;
	int32_t shift_bit;
	ShiftType direction;

	if (scale > 1.0){
		direction = LEFT;
		QuantizeMultiplierGreaterThanOne((double)scale, &quantized_multiplier, &shift_bit);
	}
	else if(scale < 1.0){
		direction = RIGHT;
		QuantizeMultiplierLessThanOne((double)scale, &quantized_multiplier, &shift_bit);
	}
	else{
		direction = FIX;
	}

	int32_t a_batches = a_tensor->shape.batches;
	int32_t a_height = a_tensor->shape.height;
	int32_t a_width = a_tensor->shape.width;
	int32_t a_depth = a_tensor->shape.depth;

	int32_t b_batches = b_tensor->shape.batches;
	int32_t b_height = b_tensor->shape.height;
	int32_t b_width = b_tensor->shape.width;
	int32_t b_depth = b_tensor->shape.depth;

	int32_t out_batches = a_batches;
	int32_t out_height = a_height;
	int32_t out_width = a_width;
	int32_t out_depth = displacement * 2 + 1;

	if ( tensor_out_prepare_normal(out_tensor, out_batches, out_height, out_width, out_depth, NN_TYPE_UINT8)!=0 ){
		return errlog(nn,"output too small");
	}

	correlation1d_param* corre_param = (correlation1d_param*)self->opaque;

	corre_param->a_data = a_data;
	corre_param->b_data = b_data;
	corre_param->out = out;
	corre_param->a_batches = a_batches;
	corre_param->a_height = a_height;
	corre_param->a_width = a_width;
	corre_param->a_depth = a_depth;
	corre_param->b_batches = b_batches;
	corre_param->b_height = b_height;
	corre_param->b_width = b_width;
	corre_param->b_depth = b_depth;
	corre_param->out_batches = out_batches;
	corre_param->out_height = out_height;
	corre_param->out_width = out_width;
	corre_param->out_depth = out_depth;
	corre_param->shift = shift;
	corre_param->a_offset = a_offset;
	corre_param->b_offset = b_offset;
	corre_param->out_offset = out_offset;
	corre_param->quantized_multiplier = quantized_multiplier;
	corre_param->shift_bit = shift_bit;
	corre_param->direction = direction;

	if (for_reference == FALSE)
	{
		int32_t displacement_left = displacement + shift;
		int32_t VLEN_8byte = VELEM(sizeof(uint8_t));
		int32_t in_depth_iterations = a_depth / VLEN_8byte;
		int32_t in_depth_residue = a_depth % VLEN_8byte;

		corre_param->a_batch_stride = a_height * a_width * a_depth * sizeof(uint8_t);
		corre_param->a_height_stride = a_width * a_depth * sizeof(uint8_t);
		corre_param->b_batch_stride = b_height * b_width * b_depth * sizeof(uint8_t);
		corre_param->b_height_stride = b_width * b_depth * sizeof(uint8_t);
		corre_param->out_batch_stride = out_height * out_width * out_depth * sizeof(uint8_t);
		corre_param->out_height_stride = out_width * out_depth * sizeof(uint8_t);
		corre_param->displacement_left = displacement_left;
		corre_param->in_depth_iterations = in_depth_iterations;
		corre_param->in_depth_residue = in_depth_residue;
		corre_param->num_threads = CORRELATION1D_MAX_THREADS;
		corre_param->jobno = 0;
		corre_param->inner_count = out_height;
		corre_param->jobcount = out_height * out_batches;
	}
	return 0;
}


static int prepare_work_8(struct nn_node *self, struct nn_graph *nn, struct nn_node *predecessor) {
	return correlation1d_prepare(self, nn, predecessor, FALSE);
}


static int prepare_work_ref(struct nn_node *self, struct nn_graph *nn, struct nn_node *predecessor) {
	return correlation1d_prepare(self, nn, predecessor, TRUE);
}


static int correlation1d_execute_hvx(struct nn_node *self, struct nn_graph *nn)
{
	prepare_work_8(self, nn, NULL);

	correlation1d_param* param = (correlation1d_param*)self->opaque;
	nn_sem_init(&param->done_sem, 0);

	int num_threads = param->num_threads;
	for(int i = 0; i < num_threads; i++) {
		nn_os_work_for_vector(nn, correlation1d_thread_work, param);
	}

	nn_sem_wait_n_times(&param->done_sem, num_threads);

	logmsg(nn, 2, "correlation1d execute done!");
	return 0;
}


static int correlation1d_execute_ref(struct nn_node* self, struct nn_graph* nn) 
{
	prepare_work_ref(self, nn, NULL);
	correlation1d_param* param = (correlation1d_param*)self->opaque;

	correlation1d_ref(param->a_data, param->b_data, param->out,
				  param->a_width, param->a_height, param->a_depth, param->b_width,
				  param->b_height, param->b_depth, param->out_batches, param->out_width,
				  param->out_height, param->out_depth, param->a_offset, param->b_offset,
				  param->out_offset, param->quantized_multiplier, param->shift_bit, param->direction);

	logmsg(nn, 2, "correlation1d execute (ref) done! %dx%dx%dx%d",
		   param->out_batches, param->out_height, param->out_width, param->out_depth);
	return 0;
}


static int correlation1d_check(struct nn_node* self, struct nn_graph* nn) 
{
	int needed_buffer_size = sizeof(correlation1d_param);
	correlation1d_param* corre_param = nn_calloc(needed_buffer_size,1);
	if(corre_param ==NULL) {
		return errlog(nn, "can't allocate %d bytes for corre_param", needed_buffer_size);
	}
	self->opaque = corre_param;
	return 0;
}

struct nn_node_ops nn_ops_for_QuantizedCorrelation1d_8x8to8 = {
	.execute = correlation1d_execute_hvx,
	.check = correlation1d_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common_release_opaque,
	.n_inputs = NN_IOCOUNT(10),
	.n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedCorrelation1d_8x8to8_ref = {
	.execute = correlation1d_execute_ref,
	.check = correlation1d_check,
	.ctor = node_alloc_common,
	.dtor = node_free_common_release_opaque,
	.n_inputs = NN_IOCOUNT(10),
	.n_outputs = NN_IOCOUNT(3),
};
