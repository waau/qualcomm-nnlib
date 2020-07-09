/*
 * Copyright (c) 2020, The Linux Foundation. All rights reserved.
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

#include "cast_utils.h"

static int quantized_cast_execute_ref(struct nn_node *self, struct nn_graph *nn) {
    const struct tensor *in_tensor = self->inputs[0];
    const struct tensor *in_min_tensor = self->inputs[1];
    const struct tensor *in_max_tensor = self->inputs[2];
    struct tensor *out_tensor = self->outputs[0];
    struct tensor *out_min_tensor = self->outputs[1];
    struct tensor *out_max_tensor = self->outputs[2];
    struct shape in_shape = in_tensor->shape;
    uint8_t *in_data = in_tensor->data;
    uint8_t *out_data = out_tensor->data;

    if (tensor_out_prepare_normal_fromshape( out_tensor, &in_tensor->shape, NN_TYPE_QUINT8 )!= 0) {
        return errlog(nn,"out too small");
    }
    if (tensor_out_prepare_normal(out_min_tensor,1,1,1,1,NN_TYPE_FLOAT )!= 0) {
        return errlog(nn,"out min too small");
    }
    if (tensor_out_prepare_normal(out_max_tensor,1,1,1,1,NN_TYPE_FLOAT )!= 0) {
        return errlog(nn,"out max too small");
    }
    
    for (int i = 0; i < in_shape.batches*in_shape.height*in_shape.width*in_shape.depth; i++){
        out_data[i] = in_data[i] + 0x80;
    }
    
    tensor_set_float(out_min_tensor, 0, tensor_get_float(in_min_tensor, 0));
    tensor_set_float(out_max_tensor, 0, tensor_get_float(in_max_tensor, 0));
    return 0;
}


static int quantized_cast_execute(struct nn_node *self, struct nn_graph *nn) {
    const struct tensor *in_tensor = self->inputs[0];
    const struct tensor *in_min_tensor = self->inputs[1];
    const struct tensor *in_max_tensor = self->inputs[2];
    struct tensor *out_tensor = self->outputs[0];
    struct tensor *out_min_tensor = self->outputs[1];
    struct tensor *out_max_tensor = self->outputs[2];
    
    
    if (tensor_out_prepare_normal_fromshape( out_tensor, &in_tensor->shape, NN_TYPE_QUINT8 )!= 0) return errlog(nn,"out too small");
    if (tensor_out_prepare_normal(out_min_tensor,1,1,1,1,NN_TYPE_FLOAT )!= 0) return errlog(nn,"out min too small");
    if (tensor_out_prepare_normal(out_max_tensor,1,1,1,1,NN_TYPE_FLOAT )!= 0) return errlog(nn,"out max too small");

    do_cast_operation(nn, in_tensor, out_tensor);
    
    tensor_set_float(out_min_tensor, 0, tensor_get_float(in_min_tensor, 0));
    tensor_set_float(out_max_tensor, 0, tensor_get_float(in_max_tensor, 0));
    return 0;
}


static void cast_hvx(struct nn_graph *nn, void *rst) {
    struct cast_run_state *rstp = (struct cast_run_state *)rst;

    HVX_Vector *in_data = (HVX_Vector *) rstp->in_data;
    HVX_Vector *out_data = (HVX_Vector *) rstp->out_data;
    HVX_Vector sign_flip_vector = Q6_V_vsplat_R(0x80808080); // Add 0x80 to every element

    for (int i = 0; i < rstp->num_vec; i++) {
        *out_data = Q6_V_vxor_VV(*in_data, sign_flip_vector);
        in_data++;
        out_data++;
    }

    // compute partial vector at the end of the tensor
    if (rstp->leftover) {
        HVX_VectorPred leftoverMask;
        leftoverMask = Q6_Q_vsetq_R(rstp->leftover);
        HVX_Vector partial_vec = Q6_V_vmux_QVV(leftoverMask, *in_data, Q6_V_vzero());
        q6op_vstcc_QAV(leftoverMask, (HVX_Vector *) out_data, Q6_V_vxor_VV(partial_vec, sign_flip_vector));
    }
    
    nn_sem_post(rstp->donesem);
}

int do_cast_operation(struct nn_graph *nn, const struct tensor *in_tensor, const struct tensor * out_tensor) {
    struct shape in_shape = in_tensor->shape;
    uint8_t *in_data = in_tensor->data;
    uint8_t *out_data = out_tensor->data;
    uint32_t input_size = in_shape.batches*in_shape.height*in_shape.width*in_shape.depth;

    int total_vectors = input_size / sizeof(HVX_Vector);
    int nthreads = total_vectors < MAX_THREADS ? total_vectors: MAX_THREADS;
    if (nthreads == 0 && (input_size % sizeof(HVX_Vector)) != 0) nthreads = 1;
    struct cast_run_state rst[nthreads];
    int pointer_offset = 0;
    nn_sem_t done_sem;
    nn_sem_init(&done_sem, 0);

    for (int i = 0; i < nthreads; i++) {
        int thread_vectors = total_vectors / nthreads;
        if (i == nthreads - 1) thread_vectors += (total_vectors % nthreads);
        rst[i].in_data = in_data + pointer_offset;
        rst[i].out_data = out_data + pointer_offset;
        rst[i].donesem = &done_sem;
        rst[i].num_vec = thread_vectors;
        rst[i].leftover = (i == nthreads-1)? input_size % sizeof(HVX_Vector) : 0;   // last work thread may compute partial vector
        pointer_offset += thread_vectors * sizeof(HVX_Vector);
    }

    for (int i = 0; i < nthreads; i++) {
        nn_os_work_for_vector(nn, cast_hvx, &rst[i]);
    }

    nn_sem_wait_n_times(&done_sem, nthreads);

    return 0;
}

struct nn_node_ops nn_ops_for_Quantized_CastUInt8ToInt8_ref = {
        .execute = quantized_cast_execute_ref,
        .check = NULL,
        .ctor = node_alloc_common,
        .dtor = node_free_common,
        .n_inputs = NN_IOCOUNT(3),
        .n_outputs = NN_IOCOUNT(3)
};

struct nn_node_ops nn_ops_for_Quantized_CastUInt8ToInt8 = {
        .execute = quantized_cast_execute,
        .check = NULL,
        .ctor = node_alloc_common,
        .dtor = node_free_common,
        .n_inputs = NN_IOCOUNT(3),
        .n_outputs = NN_IOCOUNT(3)
};

struct nn_node_ops nn_ops_for_Quantized_CastInt8ToUInt8_ref = {
        .execute = quantized_cast_execute_ref,
        .check = NULL,
        .ctor = node_alloc_common,
        .dtor = node_free_common,
        .n_inputs = NN_IOCOUNT(3),
        .n_outputs = NN_IOCOUNT(3)
};

struct nn_node_ops nn_ops_for_Quantized_CastInt8ToUInt8 = {
        .execute = quantized_cast_execute,
        .check = NULL,
        .ctor = node_alloc_common,
        .dtor = node_free_common,
        .n_inputs = NN_IOCOUNT(3),
        .n_outputs = NN_IOCOUNT(3),
        .flags = NN_NODE_FLAG_CLS_QUANTIZEDCAST
};