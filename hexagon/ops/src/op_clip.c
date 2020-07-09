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
 * 
 * Now that that's out of the way, let's get to the good stuff.
 * This contains the code for QuantizedClip_i16
 */

#include <nn_graph.h>
#include <string.h>
#include <quantize.h>
#include <math.h>
#if defined(__hexagon__)
#include <hexagon_types.h>
#endif
#include "hvx_hexagon_protos.h"
#include <stdio.h>
#include "hvx_inlines.h"
#include "nn_hvx_debug_helper.h"
#include "nn_reduce_utils.h"
#define ELEMENTS_PER_HVX_16B 64U
#define VECTORS_PER_PREFETCH 4U


static int execute_clip_16b_ref(struct nn_node *self, struct nn_graph *nn){
    //Here we compute the following
    //Given a clip value x
    //We clip the input y
    //To be [-x, x]
    //But with the original quantization range
    const struct tensor *in_tensor = self->inputs[0];
    const struct tensor *in_min_tensor = self->inputs[1];
    const struct tensor *in_max_tensor = self->inputs[2];
    const struct tensor *clip_tensor = self->inputs[3];
    struct tensor *out_tensor = self->outputs[0];
    struct tensor *out_min_tensor = self->outputs[1];
    struct tensor *out_max_tensor = self->outputs[2];

    float in_min, in_max, clip_val;
    in_min = tensor_get_float(in_min_tensor, 0);
    in_max = tensor_get_float(in_max_tensor, 0);
    clip_val = tensor_get_float(clip_tensor, 0);
    float in_scale = get_qi16_level_size(in_min, in_max);
    uint32_t batches = in_tensor->shape.batches;
    uint32_t width = in_tensor->shape.width;
    uint32_t height = in_tensor->shape.height;
    uint32_t depth = in_tensor->shape.depth;
    uint32_t numElements = batches * width *height*depth;
    int16_t clipMin, clipMax;

    //Compute the quantized clip range as clipMax = clipVal / in_scale
    //And clipMin = -clipMax
    clipMax = saturate_i16(roundf_i32(clip_val/in_scale));
    clipMin = -clipMax;
    logmsg(nn,2, "OP_Clip_i16 id: %x with clip value %f has [%d,%d] quantized clip range", self->node_id, clip_val, clipMin, clipMax);

    if(0!=(tensor_out_prepare_normal(out_tensor, batches, height, width, depth, NN_TYPE_QINT16))){
        return errlog(nn, "OP_Clip_i16 id: %x output tensor too small for input", self->node_id);
    }
    //We clip within the original range
    tensor_set_single_float(out_min_tensor, in_min);
    tensor_set_single_float(out_max_tensor, in_max);
    //Process the data
    int16_t currVal;
    const int16_t *inData = in_tensor->data;
    int16_t *outData = out_tensor->data;
    for(uint32_t i = 0; i < numElements; ++i){
        currVal = *inData;
        //We have hvx intrinistics for min/max
        //But not for scalar
        currVal = (currVal <= clipMax)? currVal:clipMax;
        currVal = (currVal >= clipMin)? currVal:clipMin;
        *outData = currVal;
        inData++;
        outData++; 
    }
    return 0;
}

struct clip_16_runstate_shared{
    nn_sem_t worker_sem;
    volatile uint32_t curr_pos;
};

struct clip_16_runstate{
    struct clip_16_runstate_shared *shared;
    struct nn_node *self;
    const int16_t *in_data;
    int16_t * out_data;
    int16_t clipMax;
    int16_t clipMin;
    uint32_t num_vectors;
    uint32_t partial_vector_size;
    uint32_t thread_id;
};

static void clip_16_hvx(struct nn_graph *nn, void *rstate){
    struct clip_16_runstate *runstate = (struct clip_16_runstate *)rstate;
    HVX_Vector clipMinHvx = q6op_Vh_vsplat_R(runstate->clipMin);
    HVX_Vector clipMaxHvx = q6op_Vh_vsplat_R(runstate->clipMax);
    HVX_Vector input;
    uint32_t pos = __sync_fetch_and_add(&(runstate->shared->curr_pos),1U);
    const int16_t *in_data = runstate->in_data;
    int16_t *out_data;
    while(pos < runstate->num_vectors){
        l2pref(in_data+(pos*ELEMENTS_PER_HVX_16B),1U,ELEMENTS_PER_HVX_16B*VECTORS_PER_PREFETCH,1U);
        in_data = runstate->in_data
                + pos * ELEMENTS_PER_HVX_16B;
        out_data = runstate->out_data
                + pos * ELEMENTS_PER_HVX_16B;
        //In theory we shouldn't need to use unaligned load/store
        //But the perf impact is minimal and its safer
        //So we will
        input = q6op_V_vldu_A((HVX_Vector *)in_data);
        input = Q6_Vh_vmin_VhVh(input, clipMaxHvx);
        input = Q6_Vh_vmax_VhVh(input, clipMinHvx);
        q6op_vstu_AV((HVX_Vector *)out_data, input);
        pos = __sync_fetch_and_add(&(runstate->shared->curr_pos),1U);
    }
    //Only worker 0 will do the leftovers
    if(0 == runstate->thread_id){
        HVX_VectorPred leftoverMask;
        //Need to multiply # of leftovers by 2 since we are dealing with int16
        leftoverMask = Q6_Q_vsetq_R(runstate->partial_vector_size*2);
        in_data = runstate->in_data
                + runstate->num_vectors * ELEMENTS_PER_HVX_16B;
        out_data = runstate->out_data
                + runstate->num_vectors * ELEMENTS_PER_HVX_16B;
        input = q6op_V_vldu_A((HVX_Vector *)in_data);
        input = Q6_Vh_vmin_VhVh(input, clipMaxHvx);
        input = Q6_Vh_vmax_VhVh(input, clipMinHvx);
        q6op_vstcc_QAV(leftoverMask, (HVX_Vector *)out_data, input);
    }
    //Post to the semaphore
    nn_sem_post(&(runstate->shared->worker_sem));
    return;
}

static int execute_clip_16b(struct nn_node *self, struct nn_graph *nn){
    //Here we compute the following
    //Given a clip value x
    //We clip the input y
    //To be [-x, x]
    //But with the original quantization range
    const struct tensor *in_tensor = self->inputs[0];
    const struct tensor *in_min_tensor = self->inputs[1];
    const struct tensor *in_max_tensor = self->inputs[2];
    const struct tensor *clip_tensor = self->inputs[3];
    struct tensor *out_tensor = self->outputs[0];
    struct tensor *out_min_tensor = self->outputs[1];
    struct tensor *out_max_tensor = self->outputs[2];

    float in_min, in_max, clip_val;
    in_min = tensor_get_float(in_min_tensor, 0);
    in_max = tensor_get_float(in_max_tensor, 0);
    clip_val = tensor_get_float(clip_tensor, 0);
    float in_scale = get_qi16_level_size(in_min, in_max);
    uint32_t batches = in_tensor->shape.batches;
    uint32_t width = in_tensor->shape.width;
    uint32_t height = in_tensor->shape.height;
    uint32_t depth = in_tensor->shape.depth;
    uint32_t numElements = batches * width *height*depth;
    int16_t clipMin, clipMax;

    //Compute the quantized clip range as clipMax = clipVal / in_scale
    //And clipMin = -clipMax
    clipMax = saturate_i16(roundf_i32(clip_val/in_scale));
    clipMin = -clipMax;
    logmsg(nn,2, "OP_Clip_i16 id: %x with clip value %f has [%d,%d] quantized clip range", self->node_id, clip_val, clipMin, clipMax);

    if(0!=(tensor_out_prepare_normal(out_tensor, batches, height, width, depth, NN_TYPE_QINT16))){
        return errlog(nn, "OP_Clip_i16 id: %x output tensor too small for input", self->node_id);
    }
    //How many full vectors are there?
    uint32_t num_vectors = numElements / ELEMENTS_PER_HVX_16B;
    //Any elements leftover?
    uint32_t leftover_size = numElements % ELEMENTS_PER_HVX_16B;
    //Determine the number of vector threads to spin up
    uint32_t num_threads;
    num_threads = (num_vectors > nn->num_vector_threads)?nn->num_vector_threads:num_vectors;
    num_threads = (num_threads < 1U)?1U:num_threads;
    nn_scratch_reset(nn);
    if(0 != nn_scratch_grow(nn, sizeof(struct clip_16_runstate_shared))){
        errlog(nn, "OP_Clip_i16 id: %x could not allocate scratch space for shared worker data", self->node_id);
        return NN_EXECUTE_OUT_OF_SCRATCH_ERROR;
    }
    struct clip_16_runstate_shared *shared_runstate = nn_scratch_alloc(nn, sizeof(struct clip_16_runstate_shared));
    if(0 != nn_scratch_grow(nn, num_threads*sizeof(struct clip_16_runstate))){
        errlog(nn, "OP_Clip_i16 id: %x could not allocate scratch space for worker data", self->node_id);
        return NN_EXECUTE_OUT_OF_SCRATCH_ERROR;
    }
    struct clip_16_runstate *worker_runstates = nn_scratch_alloc(nn, num_threads*sizeof(struct clip_16_runstate));

    nn_sem_init(&(shared_runstate->worker_sem), 0);
    shared_runstate->curr_pos = 0U;
    //Initialize the runstates
    for(uint32_t i = 0; i < num_threads; ++i){
        worker_runstates[i].self = self;
        worker_runstates[i].in_data = in_tensor->data;
        worker_runstates[i].out_data = out_tensor->data;
        worker_runstates[i].clipMin = clipMin;
        worker_runstates[i].clipMax = clipMax;
        worker_runstates[i].num_vectors = num_vectors;
        worker_runstates[i].partial_vector_size = leftover_size;
        worker_runstates[i].thread_id = i;
        worker_runstates[i].shared = shared_runstate;
    }
    //Spin up vector threads
    for(uint32_t i=0; i < num_threads; ++i){
        nn_os_work_for_vector(nn, clip_16_hvx, &(worker_runstates[i]));
    }
    nn_sem_wait_n_times(&(shared_runstate->worker_sem), num_threads);
    //We clip within the original range
    tensor_set_single_float(out_min_tensor, in_min);
    tensor_set_single_float(out_max_tensor, in_max);
    return 0;
}

struct nn_node_ops nn_ops_for_QuantizedClip_i16 = {
    .execute = execute_clip_16b,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT(4),
    .n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedClip_i16_ref = {
    .execute = execute_clip_16b_ref,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT(4),
    .n_outputs = NN_IOCOUNT(3),
};