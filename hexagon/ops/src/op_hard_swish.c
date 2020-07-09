 /* Copyright (c) 2016-2020, The Linux Foundation. All rights reserved.
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
 * This contains the implementation of HARD SWISH:
 * h-swish = x * max(0, min(6, (x + 3))) / 6
 */

#include <nn_graph.h>
#include <math.h>
#include <quantize.h>
#include "hvx_inlines.h"

#ifdef HEXAGON_V66
#define MAX_THREADS 4
#else
#define MAX_THREADS 2
#endif

// h-swish is a piece-wise function, for x in [-3, 3], it reaches global min at -1.5
#define HS_GLOBAL_MIN -1.5

struct hard_swish_info {
    int is_executed;
    float in_min_val;
    float in_max_val;
    float out_min_val;
    float out_max_val;
    uint8_t *pre_lut;
};

struct h_swish_run_state {
    uint8_t *in_data;
    uint8_t *lut; // Look Up Table
    uint8_t *out_data;
    nn_sem_t *done_sem;
    int num_vec;  // number of vectors to process
};

struct h_swish_d32_run_state {
    uint8_t *in_data;
    uint8_t *lut; // Look Up Table
    uint8_t *out_data;
    nn_sem_t *done_sem;
    int num_vec;  // number of vectors to process
    struct shape shape;// shape of operation
    struct tensor_addressing t_address_in;
    struct tensor_addressing t_address_out;
    volatile int next_job_index;
    int n_jobs;
};

static void h_swish_d32_run_thread( struct nn_graph * nn, void * rstpv);

static int h_swish_check_opt(struct nn_node *self, struct  nn_graph *nn) {
    struct hard_swish_info *hs_info;
    if (self->opaque == NULL){
        hs_info = nn_calloc(1, sizeof(struct hard_swish_info));
        if (hs_info == NULL) return errlog(nn, "calloc failed");
        //set is_executed == 0 to represent new execution
        if ((hs_info->pre_lut = nn_memalign(128,256))== NULL)
        {
            nn_free(hs_info);
            return errlog(nn, "can't allocate hs_info->pre_lut",-1);
        }
        hs_info->is_executed = 0;
        self->opaque = hs_info;
    }
    return 0;
};

static int h_swish_dtor(struct nn_node *self, struct  nn_graph *nn) {
    struct hard_swish_info *hs_info = self->opaque;
    if (hs_info)
    {   
        if (hs_info->pre_lut) nn_free(hs_info->pre_lut);
        nn_free(hs_info);
    }
    self->opaque = NULL;
    return node_free_common(self, nn);
};

static void lookup_hvx(struct nn_graph *nn, void *rst) {
    struct h_swish_run_state *rstp = (struct h_swish_run_state *)rst;
    uint8_t* in = rstp->in_data;
    uint8_t* out = rstp->out_data;
    HVX_Vector lut0,lut1,luta,lutb;
    lut0 = (*(HVX_Vector *) rstp->lut);
    lut1 = (*(HVX_Vector *) &rstp->lut[128]);
    luta = Q6_Vb_vshuff_Vb(lut0);
    lutb = Q6_Vb_vshuff_Vb(lut1);

    for (int i = 0; i < rstp->num_vec; i++) {
        HVX_Vector vin = *(HVX_Vector *) in;
        HVX_Vector *vout = (HVX_Vector *) out;
        *vout = hvx_table_lookup_u8( vin, luta, lutb);
        in += 128;
        out += 128;
    }

    nn_sem_post(rstp->done_sem);
}

static void build_table(struct hard_swish_info *hs_info, int zero_in, float in_step, float out_min, float out_max) {

    for (int i = 0; i < 256; i++) {
        float in_float =  (i-zero_in) * in_step;
        float out_float = in_float * fmaxf(fminf(6.f,in_float+3.f),0.f)/6.f;
        hs_info->pre_lut[i] = quantize_uint8(out_float,out_min,out_max);
    }
}

static int h_swish_execute_q8(struct nn_node *self, struct nn_graph *nn) {
    int is_flat = 0;
    if (self->node_type == OP_QuantizedHardSwish_8_ref || self->node_type == OP_QuantizedHardSwish_8) is_flat = 1;
    const struct tensor *input_tensor = self->inputs[0];
    const struct tensor *input_min_tensor = self->inputs[1];
    const struct tensor *input_max_tensor = self->inputs[2];
    struct tensor *out_tensor = self->outputs[0];
    struct tensor *out_min_tensor = self->outputs[1];
    struct tensor *out_max_tensor = self->outputs[2];
    float input_min = tensor_get_float(input_min_tensor,0);
    float input_max = tensor_get_float(input_max_tensor,0);
    struct hard_swish_info *hs_info = self->opaque;

    float output_min;
    float output_max;
    float in_step;
    float out_step;
    int input_size ;
    int zero_in = get_qu8_level_size_zero(input_min,input_max,&in_step);

    //Given the output min/max
    if (self->n_inputs == 5) {
        output_min = tensor_get_float(self->inputs[3], 0);
        output_max = tensor_get_float(self->inputs[4], 0);
    } else {
        float left = input_min * fmaxf(fminf(6.f,input_min+3.f),0.f)/6.f;
        float right = input_max * fmaxf(fminf(6.f,input_max+3.f),0.f)/6.f;
        //Three cases: depends on input minmax related to -1.5
        if (input_min >= HS_GLOBAL_MIN) { 
            output_min = left;
            output_max = right;
        } else if(input_max <= HS_GLOBAL_MIN) {
            output_min = right;
            output_max = left;
        } else {
            //x = -1.5 y = -0.375
            output_min = HS_GLOBAL_MIN * fmaxf(0.f,fminf(6.f,HS_GLOBAL_MIN+3.f))/6;
            output_max = fmaxf(left,right);
        }
    }

    get_qu8_level_size_zero (output_min,output_max,&out_step);
    if (hs_info != NULL && hs_info->is_executed == 1 
        && hs_info->in_min_val == input_min && hs_info->in_max_val == input_max 
        && hs_info->out_min_val == output_min && hs_info->out_max_val == output_max) {
            logmsg(nn,2,"Hard Swish get look-up table loan from previous");
    } else {
        build_table(hs_info, zero_in,in_step,output_min,output_max);
        hs_info->in_min_val = input_min;
        hs_info->in_max_val = input_max;
        hs_info->out_min_val = output_min;
        hs_info->out_max_val = output_max;
        hs_info->is_executed = 1;
    }

    nn_sem_t done_sem;
    nn_sem_init(&done_sem, 0);

    //input is quantized 8
    if (unlikely(is_flat==1)) {
        input_size = input_tensor->shape.batches * input_tensor->shape.height * input_tensor->shape.width * input_tensor->shape.depth;
        uint8_t *input_data = input_tensor->data;
        uint8_t *output_data = out_tensor->data;
        tensor_out_prepare_normal(out_tensor, input_tensor->shape.batches, input_tensor->shape.height, input_tensor->shape.width, input_tensor->shape.depth, NN_TYPE_QUINT8);
        tensor_out_prepare_normal(out_min_tensor, 1, 1, 1, 1, NN_TYPE_FLOAT);
        tensor_out_prepare_normal(out_max_tensor, 1, 1, 1, 1, NN_TYPE_FLOAT);
        int offset = 0;
        int total_vectors = input_size / 128;
        int nthreads = total_vectors < MAX_THREADS ? total_vectors: MAX_THREADS;
        struct h_swish_run_state rst[nthreads];
        for (int i = 0; i < nthreads; i++) {
            int thread_vectors = total_vectors / nthreads;
            if (i == nthreads - 1) thread_vectors += (total_vectors % nthreads);
            rst[i].in_data = input_data + offset;
            rst[i].lut = hs_info->pre_lut;
            rst[i].out_data = output_data + offset;
            rst[i].done_sem = &done_sem;
            rst[i].num_vec = thread_vectors;
            offset += thread_vectors * 128;
        }
        for (int i = 0; i < nthreads; i++) {
            nn_os_work_for_vector(nn, lookup_hvx, &rst[i]);
        }
        for(int i = offset; i < input_size; i++) {
            output_data[i] = hs_info->pre_lut[input_data[i]];
        }
        nn_sem_wait_n_times(&done_sem, nthreads);
        tensor_set_single_float(out_min_tensor, output_min);
        tensor_set_single_float(out_max_tensor, output_max);
    } else{
        struct h_swish_d32_run_state rstt;
        int n_jobs;
        rstt.t_address_in = tensor_addressing_d32(input_tensor);
        rstt.shape = input_tensor->shape;
        int b = input_tensor->shape.batches;
        int h = input_tensor->shape.height;
        int h_pad_before = input_tensor->format.height_pad[0];
        int h_pad_after = input_tensor->format.height_pad[1];
        int w = input_tensor->shape.width;
        int w_pad_before = input_tensor->format.width_pad[0];
        int w_pad_after = input_tensor->format.width_pad[1];
        int d = input_tensor->shape.depth;
        int d_pad_before = input_tensor->format.depth_pad[0];
        int nd32 = rstt.t_address_in.nd32;
        n_jobs = b*nd32;
        int d_pad_after = nd32*32 - (d_pad_before + d);
        if(tensor_out_prepare_padded_d32(out_tensor,b,
                h,h_pad_before,h_pad_after,
                w,w_pad_before,w_pad_after,
                d,d_pad_before,d_pad_after,NN_TYPE_QUINT8)!= 0){
            return errlog(nn,"out too small");
        }
        rstt.t_address_out = tensor_addressing_d32(out_tensor);
        rstt.n_jobs = n_jobs;
        int nthreads = n_jobs < MAX_THREADS ? n_jobs: MAX_THREADS;

        rstt.next_job_index = 0;
        rstt.lut = hs_info->pre_lut;
        void (*thread_run_fp)( struct nn_graph * nn, void * rstpv) = h_swish_d32_run_thread;
        for (int i=1;i<=nthreads;i++){
            rstt.done_sem = &done_sem;
            nn_os_work_for_vector(nn,thread_run_fp, &rstt);
        }
        tensor_set_single_float(out_min_tensor, output_min);
        tensor_set_single_float(out_max_tensor, output_max);
        nn_sem_wait_n_times(&done_sem, nthreads);
    }

    return 0;
}

static void h_swish_d32_run_thread(struct nn_graph * nn, void * rstpv) {
    struct h_swish_d32_run_state *rstp = (struct h_swish_d32_run_state *)rstpv;
    int job_index;
    int d = rstp->shape.depth;
    int d0 = rstp->t_address_in.d0;
    int nd32 = rstp->t_address_in.nd32;
    int pf_h = rstp->shape.height;
    int pf_w = rstp->shape.width*32;
    int p_h_stride = rstp->t_address_in.height_stride;
    HVX_Vector luta,lutb,lut0,lut1;
    luta = (*(HVX_Vector *) rstp->lut);
    lutb = (*(HVX_Vector *) &rstp->lut[128]);
    lut0 = Q6_Vb_vshuff_Vb(luta);
    lut1 = Q6_Vb_vshuff_Vb(lutb);

    batchslice_decode bsdecode;
    batchslice_decode_init( &bsdecode,nd32);
    int n_jobs = rstp->n_jobs;
    while( job_index = __sync_fetch_and_add(&rstp->next_job_index, 1), job_index < n_jobs ){
        int id32 = batchslice_decode_update(&bsdecode,job_index);
        int ib = bsdecode.ibatch;
        uint8_t const * in_ptr = rstp->t_address_in.data + ib*rstp->t_address_in.batch_stride + id32*rstp->t_address_in.d32_stride;
        l2pref( in_ptr, pf_h, pf_w, p_h_stride);
        uint8_t * out_ptr = rstp->t_address_out.data + ib*rstp->t_address_out.batch_stride + id32*rstp->t_address_out.d32_stride;
        int d_start = id32==0? d0: 0;
        int dn = min_i32(d0 + d - id32*32, 32)-d_start;
        int h = rstp->shape.height;
        int wid = rstp->shape.width;
        // offset for width padding
        int wpad = (int)(size_t)out_ptr & 127;
        wid += wpad >> 5;
        out_ptr -= wpad;
        in_ptr -= wpad;    
        int nvec_wide = (wid + 3)>>2; 

        int in_height_stride = rstp->t_address_in.height_stride;
        int out_height_stride = rstp->t_address_out.height_stride;

        // if all the active 'depth' elements are in the first half,
        // and h >= 2, do two rows at once, packing to one vector, for more efficiency.
        if( h >= 2 && (dn+d0) <=16){
            int nhpair = h>>1;
            for( int ih = 0; ih < nhpair; ih++){
                HVX_Vector const * vinp0 =  (HVX_Vector const *) in_ptr;
                HVX_Vector const * vinp1 =  (HVX_Vector const *)( in_ptr + in_height_stride);
                HVX_Vector * voutp0 =  (HVX_Vector  *) out_ptr;
                HVX_Vector * voutp1 =  (HVX_Vector  *)( out_ptr + out_height_stride);
                for(int iw = 0; iw <nvec_wide; iw++){
                    // shuffle vectors from two rows together, keeping 1st 16 from each
                    // depth slot.
                    HVX_Vector vin = Q6_V_lo_W( Q6_W_vshuff_VVR(vinp1[iw], vinp0[iw],16));
                    HVX_Vector vout = hvx_table_lookup_u8( vin, lut0, lut1);
                    voutp0[iw] = vout;  // first result
                    voutp1[iw] = Q6_V_vror_VR( vout, 16);   //second result
                }
                in_ptr += in_height_stride *2;
                out_ptr += out_height_stride *2;
            }
            h = h & 1;    // maybe one odd row left
        }

        for(int ih = 0; ih < h; ih++){
            HVX_Vector const * vinp =  (HVX_Vector const *)( in_ptr + in_height_stride * ih);
            HVX_Vector * voutp =  (HVX_Vector  *)( out_ptr + out_height_stride * ih);
            for(int iw = 0; iw <nvec_wide; iw++){
                HVX_Vector vin = vinp[iw];
                voutp[iw]= hvx_table_lookup_u8( vin, lut0, lut1);
            }
        }
    }
    nn_sem_post(rstp->done_sem);
}

struct nn_node_ops nn_ops_for_QuantizedHardSwish_8 = {
    .execute = h_swish_execute_q8,
    .check = h_swish_check_opt,
    .ctor = node_alloc_common,
    .dtor = h_swish_dtor,
    .n_inputs = NN_IOCOUNT_RANGE(3,5),
    .n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedHardSwish_8_ref = {
    .execute = h_swish_execute_q8,
    .check = h_swish_check_opt,
    .ctor = node_alloc_common,
    .dtor = h_swish_dtor,
    .n_inputs = NN_IOCOUNT_RANGE(3,5),
    .n_outputs = NN_IOCOUNT(3),
};
 struct nn_node_ops nn_ops_for_QuantizedHardSwish_8_d32 = {
     .execute = h_swish_execute_q8,
     .check = h_swish_check_opt,
     .ctor = node_alloc_common,
     .dtor = h_swish_dtor,
     .n_inputs = NN_IOCOUNT_RANGE(3,5),
     .n_outputs = NN_IOCOUNT(3),
     .flags = NN_NODE_FLAG_D32_INPUT | NN_NODE_FLAG_D32_OUTPUT,
 };
