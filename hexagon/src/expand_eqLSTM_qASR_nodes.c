/*
 * Copyright (c) 2019-2020, The Linux Foundation. All rights reserved.
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
#include "nn_prepare.h"
#include "nn_graph.h"
#include "expand_nodes.h"
#include <quantize.h>
#include <nn_graph.h>
#include <nn_graph_types.h>
#include <data_utils.h>
#include "expand_eqLSTM_qASR_nodes.h"

#define MATMUL_INP_COUNT 11
#define MATMUL_OUT_COUNT 3
#define QAdd_16_INP_COUNT 8
#define LAYERNORM_INP_COUNT 12
#define ELE_WISE_16_OUT_COUNT 3
#define SIG_TANH_INP_OUT_COUNT 3
#define CONVERT_16_8_INP_COUNT 5
#define MATMUL_U8_INP_COUNT 6
#define MATMUL_U8_OUT_COUNT 3
#define CLIP_INP_COUNT 4
#define CLIP_OUT_COUNT 3


//Utility to find the producer of an input
//And get the shape
static inline struct shape get_shape_from_producer(struct nn_graph *nn, struct input input_to_find){
    struct nn_node *producer;
    producer = find_node(nn, input_to_find.src_id);
    return producer->outputs[input_to_find.output_idx]->shape;
}

//Utility to compute the shape of a matmul from the shape of its inputs
//Currently completely naive
static inline struct shape compute_matmul_output_shape(struct shape input, struct shape weights){
    //TODO: Check supported shape?
    struct shape matmul_out;
    matmul_out.batches = 1u;
    matmul_out.height = 1u;
    matmul_out.width = input.width*input.height*input.batches;
    matmul_out.depth = weights.depth;
    return matmul_out;
}

//Create Nodes for the Ops in a gate
//Ops in gate are:
//MatMul8x8p32to16 \
//                  \
//MatMul8x8p32to16 -> Addqi16 -> LayerNorm_i16 -> activation (either sigmoid or tanh)
static int create_eqLSTM_gate(struct nn_graph *nn, struct nn_node *eqLSTM_node, struct nn_node **eqLSTM_node_list, uint32_t *node_list_start_index, uint32_t layerNormConstNid,
            struct eqLSTM_gate_input_refs gate_refs, struct expand_eqLSTM_qASR_state *expand_state){
    struct nn_node *input_MatMul;
    struct nn_node *recurrent_MatMul;
    struct nn_node *MatMul_accumulate;
    struct nn_node *layerNorm;
    struct nn_node *activation;
    struct shape input_shape;
    struct shape input_weights_shape;
    struct shape input_matmul_shape;
    //Make input and output defs for i2i_MatMul
    struct input input_MatMul_refs[11];
    input_MatMul_refs[0] = eqLSTM_node->input_refs[0];
    input_MatMul_refs[1] = eqLSTM_node->input_refs[gate_refs.input_weights_id];
    input_MatMul_refs[2] = eqLSTM_node->input_refs[1];
    input_MatMul_refs[3] = eqLSTM_node->input_refs[2];
    input_MatMul_refs[4] = eqLSTM_node->input_refs[gate_refs.input_weights_min];
    input_MatMul_refs[5] = eqLSTM_node->input_refs[gate_refs.input_weights_max];
    input_MatMul_refs[6] = (struct input){expand_state->fake_bias_node_id, 0};
    input_MatMul_refs[7] = (struct input){expand_state->fake_bias_q_params_id, 0};
    input_MatMul_refs[8] = (struct input){expand_state->fake_bias_q_params_id, 0};
    input_MatMul_refs[9] = eqLSTM_node->input_refs[gate_refs.gate_matmul_min];
    input_MatMul_refs[10] = eqLSTM_node->input_refs[gate_refs.gate_matmul_max];

    //Determine the output shape of the MatMul
    //step one find producer and get shape of input
    input_shape = get_shape_from_producer(nn, input_MatMul_refs[0]);
    input_weights_shape = get_shape_from_producer(nn, input_MatMul_refs[1]);
    input_matmul_shape = compute_matmul_output_shape(input_shape, input_weights_shape);

    struct output input_MM_output_refs[3];
    make_outputdesc_from_shape(&input_MM_output_refs[0], &input_matmul_shape, tensor_type_size(NN_TYPE_QINT16), 0);
    input_MM_output_refs[1] = Output_ScalarFloat;
    input_MM_output_refs[2] = Output_ScalarFloat;

    input_MatMul = create_node(nn, 0, OP_QuantizedMatMul_8x8p32to16, NN_PAD_NA, MATMUL_INP_COUNT, MATMUL_OUT_COUNT, input_MatMul_refs, input_MM_output_refs);

    struct output input_requant_outputs[3];
    make_outputdesc_from_shape(&input_requant_outputs[0], &input_matmul_shape, tensor_type_size(NN_TYPE_QINT16), 0);
    input_requant_outputs[1] = Output_ScalarFloat;
    input_requant_outputs[2] = Output_ScalarFloat;

    struct input recurrent_MM_refs[11];
    recurrent_MM_refs[0] = eqLSTM_node->input_refs[3];
    recurrent_MM_refs[1] = eqLSTM_node->input_refs[gate_refs.recurrent_weights_id];
    recurrent_MM_refs[2] = eqLSTM_node->input_refs[4];
    recurrent_MM_refs[3] = eqLSTM_node->input_refs[5];
    recurrent_MM_refs[4] = eqLSTM_node->input_refs[gate_refs.recurrent_weights_min];
    recurrent_MM_refs[5] = eqLSTM_node->input_refs[gate_refs.recurrent_weights_max];
    recurrent_MM_refs[6] = (struct input){expand_state->fake_bias_node_id, 0};
    recurrent_MM_refs[7] = (struct input){expand_state->fake_bias_q_params_id, 0};
    recurrent_MM_refs[8] = (struct input){expand_state->fake_bias_q_params_id, 0};
    recurrent_MM_refs[9] = eqLSTM_node->input_refs[gate_refs.gate_matmul_min];
    recurrent_MM_refs[10] = eqLSTM_node->input_refs[gate_refs.gate_matmul_max];

    struct output recurrent_MM_output_refs[3];
    struct shape recurrent_shape;
    struct shape recurent_weights_shape;
    struct shape recurrent_matmul_shape;
    recurrent_shape = get_shape_from_producer(nn, recurrent_MM_refs[0]);
    recurent_weights_shape = get_shape_from_producer(nn, recurrent_MM_refs[1]);
    recurrent_matmul_shape = compute_matmul_output_shape(recurrent_shape, recurent_weights_shape);

    make_outputdesc_from_shape(&recurrent_MM_output_refs[0], &recurrent_matmul_shape, tensor_type_size(NN_TYPE_QINT16), 0);
    recurrent_MM_output_refs[1] = Output_ScalarFloat;
    recurrent_MM_output_refs[2] = Output_ScalarFloat;

    recurrent_MatMul = create_node(nn, 0, OP_QuantizedMatMul_8x8p32to16, NN_PAD_NA, MATMUL_INP_COUNT, MATMUL_OUT_COUNT, recurrent_MM_refs, recurrent_MM_output_refs);

    //input refs for matmul accumulate
    struct input MatMul_accumulate_inputs[8];
    //OP_QuantizedAdd_i16 has 8 inputs
    MatMul_accumulate_inputs[0] = (struct input){input_MatMul->node_id, 0};
    MatMul_accumulate_inputs[1] = (struct input){recurrent_MatMul->node_id, 0};
    MatMul_accumulate_inputs[2] = (struct input){input_MatMul->node_id, 1};
    MatMul_accumulate_inputs[3] = (struct input){input_MatMul->node_id, 2};
    //Life is nice when the inputs and the output all share a range
    MatMul_accumulate_inputs[4] = MatMul_accumulate_inputs[2];
    MatMul_accumulate_inputs[5] = MatMul_accumulate_inputs[3];
    MatMul_accumulate_inputs[6] = MatMul_accumulate_inputs[2]; 
    MatMul_accumulate_inputs[7] = MatMul_accumulate_inputs[3];
    //And the same output descriptions

    MatMul_accumulate = create_node(nn, 0, OP_QuantizedAdd_16, NN_PAD_NA, QAdd_16_INP_COUNT, MATMUL_OUT_COUNT, MatMul_accumulate_inputs, input_requant_outputs);

    //LayerNorm inputs
    struct input LayerNorm_inputs[12];
    LayerNorm_inputs[0] = (struct input){MatMul_accumulate->node_id, 0};
    LayerNorm_inputs[1] = (struct input){MatMul_accumulate->node_id, 1};
    LayerNorm_inputs[2] = (struct input){MatMul_accumulate->node_id, 2};
    LayerNorm_inputs[3] = eqLSTM_node->input_refs[gate_refs.gate_layernorm_coefficients];
    LayerNorm_inputs[4] = eqLSTM_node->input_refs[gate_refs.gate_layernorm_coefficients_min];
    LayerNorm_inputs[5] = eqLSTM_node->input_refs[gate_refs.gate_layernorm_coefficients_max];
    LayerNorm_inputs[6] = eqLSTM_node->input_refs[gate_refs.gate_layernorm_bias];
    LayerNorm_inputs[7] = eqLSTM_node->input_refs[gate_refs.gate_layernorm_bias_min];
    LayerNorm_inputs[8] = eqLSTM_node->input_refs[gate_refs.gate_layernorm_bias_max];
    LayerNorm_inputs[9] = eqLSTM_node->input_refs[gate_refs.gate_layernorm_min];
    LayerNorm_inputs[10] = eqLSTM_node->input_refs[gate_refs.gate_layernorm_max];
    LayerNorm_inputs[11] = (struct input){layerNormConstNid, 0};

    //LayerNorm outputs are still the same three outputs
    layerNorm = create_node(nn, 0, OP_QuantizedLayerNorm_i16, NN_PAD_NA, LAYERNORM_INP_COUNT, MATMUL_OUT_COUNT, LayerNorm_inputs, input_requant_outputs);

    struct input activation_inputs[3];
    activation_inputs[0] = (struct input){layerNorm->node_id, 0};
    activation_inputs[1] = (struct input){layerNorm->node_id, 1};
    activation_inputs[2] = (struct input){layerNorm->node_id, 2};
    if(!(gate_refs.apply_tanh)){
        activation = create_node(nn, 0, OP_QuantizedSigmoid_16, NN_PAD_NA, SIG_TANH_INP_OUT_COUNT, SIG_TANH_INP_OUT_COUNT, activation_inputs, input_requant_outputs);
    }
    else{
        activation = create_node(nn, 0, OP_QuantizedTanh_16, NN_PAD_NA, SIG_TANH_INP_OUT_COUNT, SIG_TANH_INP_OUT_COUNT, activation_inputs, input_requant_outputs);   
    }

    eqLSTM_node_list[(*node_list_start_index)++] = input_MatMul;
    eqLSTM_node_list[(*node_list_start_index)++] = recurrent_MatMul;
    eqLSTM_node_list[(*node_list_start_index)++] = MatMul_accumulate;
    eqLSTM_node_list[(*node_list_start_index)++] = layerNorm;
    eqLSTM_node_list[(*node_list_start_index)++] = activation;

    return 0;
}

static int expand_eqLSTM_qASR_node(struct nn_graph *nn, struct nn_node *eqLSTM_node, struct expand_eqLSTM_qASR_state *state){
    uint32_t eqLSTM_node_list_start_index = 0;
    uint32_t input_gate_sigmoid_index, forget_gate_sigmoid_index, cell_gate_tanh_index, output_gate_sigmoid_index;
    //Create input gate
    if((create_eqLSTM_gate(nn, eqLSTM_node, state->eqLSTM_node_list, &eqLSTM_node_list_start_index, state->layernorm_axis_const_node_id, state->input_gate_refs,
        state))!=0){
        errlog(nn, "Error expanding eqLSTM qASR OP id: %u, could not create input gate", eqLSTM_node->node_id);
        return NN_PREPARE_ERROR;
    }
    input_gate_sigmoid_index = eqLSTM_node_list_start_index-1;
    //Create forget gate
    if((create_eqLSTM_gate(nn, eqLSTM_node, state->eqLSTM_node_list, &eqLSTM_node_list_start_index, state->layernorm_axis_const_node_id, state->forget_gate_refs,
        state))!=0){
        errlog(nn, "Error expanding eqLSTM qASR OP id: %u, could not create forget gate", eqLSTM_node->node_id);
        return NN_PREPARE_ERROR;
    }
    forget_gate_sigmoid_index = eqLSTM_node_list_start_index-1;
    //Create cell gate
    if((create_eqLSTM_gate(nn, eqLSTM_node, state->eqLSTM_node_list, &eqLSTM_node_list_start_index, state->layernorm_axis_const_node_id, state->cell_gate_refs,
        state))!=0){
        errlog(nn, "Error expanding eqLSTM qASR OP id: %u, could not create cell gate", eqLSTM_node->node_id);
        return NN_PREPARE_ERROR;
    }
    cell_gate_tanh_index = eqLSTM_node_list_start_index-1;
    //Create output gate
    if((create_eqLSTM_gate(nn, eqLSTM_node, state->eqLSTM_node_list, &eqLSTM_node_list_start_index, state->layernorm_axis_const_node_id, state->output_gate_refs,
        state))!=0){
        errlog(nn, "Error expanding eqLSTM qASR OP id: %u, could not create output gate", eqLSTM_node->node_id);
        return NN_PREPARE_ERROR;
    }
    output_gate_sigmoid_index = eqLSTM_node_list_start_index-1;

    //New Cell State = forget_gate * old_cell + input_gate*cell_gate
    //So three nodes?
    //Save ids
    uint32_t inp_x_cell_index, forget_x_prev_cell_index, new_cell_add_index;
    struct input inp_x_cell_inputs[8];
    inp_x_cell_inputs[0] = (struct input){state->eqLSTM_node_list[input_gate_sigmoid_index]->node_id, 0};
    inp_x_cell_inputs[1] = (struct input){state->eqLSTM_node_list[cell_gate_tanh_index]->node_id, 0};
    inp_x_cell_inputs[2] = (struct input){state->eqLSTM_node_list[input_gate_sigmoid_index]->node_id, 1};
    inp_x_cell_inputs[3] = (struct input){state->eqLSTM_node_list[input_gate_sigmoid_index]->node_id, 2};
    inp_x_cell_inputs[4] = (struct input){state->eqLSTM_node_list[cell_gate_tanh_index]->node_id, 1};
    inp_x_cell_inputs[5] = (struct input){state->eqLSTM_node_list[cell_gate_tanh_index]->node_id, 2};

    inp_x_cell_inputs[6] = (struct input){state->eqLSTM_node_list[cell_gate_tanh_index]->node_id, 1};
    inp_x_cell_inputs[7] = (struct input){state->eqLSTM_node_list[cell_gate_tanh_index]->node_id, 2};
    struct nn_node *ele_wise_producer;
    ele_wise_producer = state->eqLSTM_node_list[cell_gate_tanh_index];
    struct output ele_wise_output_defs[3];
    struct shape ele_wise_output_shape;
    ele_wise_output_shape = ele_wise_producer->outputs[0]->shape;
    make_outputdesc_from_shape(&ele_wise_output_defs[0], &ele_wise_output_shape, tensor_type_size(NN_TYPE_QINT16), 0);
    ele_wise_output_defs[1] = Output_ScalarFloat;
    ele_wise_output_defs[2] = Output_ScalarFloat;

    struct input forget_x_prev_cell_inputs[8];
    forget_x_prev_cell_inputs[0] = (struct input){state->eqLSTM_node_list[forget_gate_sigmoid_index]->node_id, 0};
    forget_x_prev_cell_inputs[1] = eqLSTM_node->input_refs[64];
    forget_x_prev_cell_inputs[2] = (struct input){state->eqLSTM_node_list[forget_gate_sigmoid_index]->node_id, 1};
    forget_x_prev_cell_inputs[3] = (struct input){state->eqLSTM_node_list[forget_gate_sigmoid_index]->node_id, 2};
    forget_x_prev_cell_inputs[4] = eqLSTM_node->input_refs[65];
    forget_x_prev_cell_inputs[5] = eqLSTM_node->input_refs[66];
    forget_x_prev_cell_inputs[6] = eqLSTM_node->input_refs[65];
    forget_x_prev_cell_inputs[7] = eqLSTM_node->input_refs[66];
    inp_x_cell_index = eqLSTM_node_list_start_index;
    state->eqLSTM_node_list[eqLSTM_node_list_start_index++] = create_node(nn, 0, OP_QuantizedMul_16, NN_PAD_NA, QAdd_16_INP_COUNT, ELE_WISE_16_OUT_COUNT, inp_x_cell_inputs, ele_wise_output_defs);
    forget_x_prev_cell_index = eqLSTM_node_list_start_index;
    state->eqLSTM_node_list[eqLSTM_node_list_start_index++] = create_node(nn, 0, OP_QuantizedMul_16, NN_PAD_NA, QAdd_16_INP_COUNT, ELE_WISE_16_OUT_COUNT, forget_x_prev_cell_inputs, ele_wise_output_defs);

    struct input new_cell_add_inputs[8];
    new_cell_add_inputs[0] = (struct input){state->eqLSTM_node_list[inp_x_cell_index]->node_id, 0};
    new_cell_add_inputs[1] = (struct input){state->eqLSTM_node_list[forget_x_prev_cell_index]->node_id, 0};
    new_cell_add_inputs[2] = (struct input){state->eqLSTM_node_list[inp_x_cell_index]->node_id, 1};
    new_cell_add_inputs[3] = (struct input){state->eqLSTM_node_list[inp_x_cell_index]->node_id, 2};
    new_cell_add_inputs[4] = (struct input){state->eqLSTM_node_list[forget_x_prev_cell_index]->node_id, 1};
    new_cell_add_inputs[5] = (struct input){state->eqLSTM_node_list[forget_x_prev_cell_index]->node_id, 2};
    //Use output scale from forget_x_prev_cell
    new_cell_add_inputs[6] = eqLSTM_node->input_refs[65];
    new_cell_add_inputs[7] = eqLSTM_node->input_refs[66];

    new_cell_add_index = eqLSTM_node_list_start_index;
    state->eqLSTM_node_list[eqLSTM_node_list_start_index++] = create_node(nn, 0, OP_QuantizedAdd_16, NN_PAD_NA, QAdd_16_INP_COUNT, ELE_WISE_16_OUT_COUNT, new_cell_add_inputs, ele_wise_output_defs);
    //Add optional cell clipping here
    if(state->has_cell_clip){
        struct input cell_clip_inputs[4];
        cell_clip_inputs[0] = (struct input){state->eqLSTM_node_list[new_cell_add_index]->node_id, 0};
        cell_clip_inputs[1] = (struct input){state->eqLSTM_node_list[new_cell_add_index]->node_id, 1};
        cell_clip_inputs[2] = (struct input){state->eqLSTM_node_list[new_cell_add_index]->node_id, 2};
        cell_clip_inputs[3] = eqLSTM_node->input_refs[79];
        new_cell_add_index = eqLSTM_node_list_start_index;
        state->eqLSTM_node_list[eqLSTM_node_list_start_index++] = create_node(nn, 0, OP_QuantizedClip_i16, NN_PAD_NA, CLIP_INP_COUNT, CLIP_OUT_COUNT, cell_clip_inputs, ele_wise_output_defs);
    }

    //After computing new cell state we compute tanh(new_cell_state)
    struct input tanh_inputs[3];
    tanh_inputs[0] = (struct input){state->eqLSTM_node_list[new_cell_add_index]->node_id, 0};
    tanh_inputs[1] = (struct input){state->eqLSTM_node_list[new_cell_add_index]->node_id, 1};
    tanh_inputs[2] = (struct input){state->eqLSTM_node_list[new_cell_add_index]->node_id, 2};
    
    uint32_t cell_tanh_index;
    cell_tanh_index = eqLSTM_node_list_start_index;
    state->eqLSTM_node_list[eqLSTM_node_list_start_index++] = create_node(nn, 0, OP_QuantizedTanh_16, NN_PAD_NA, SIG_TANH_INP_OUT_COUNT, SIG_TANH_INP_OUT_COUNT, tanh_inputs, ele_wise_output_defs);

    //Node for computing output_gate x tanh(new_cell_state)
    struct input cell_tanh_x_output_inputs[8];
    cell_tanh_x_output_inputs[0] = (struct input){state->eqLSTM_node_list[cell_tanh_index]->node_id, 0};
    cell_tanh_x_output_inputs[1] = (struct input){state->eqLSTM_node_list[output_gate_sigmoid_index]->node_id, 0};
    cell_tanh_x_output_inputs[2] = (struct input){state->eqLSTM_node_list[cell_tanh_index]->node_id, 1};
    cell_tanh_x_output_inputs[3] = (struct input){state->eqLSTM_node_list[cell_tanh_index]->node_id, 2};
    cell_tanh_x_output_inputs[4] = (struct input){state->eqLSTM_node_list[output_gate_sigmoid_index]->node_id, 1};
    cell_tanh_x_output_inputs[5] = (struct input){state->eqLSTM_node_list[output_gate_sigmoid_index]->node_id, 2};
    //Both tanh_16 and sigmoid_16 have output ranges from -1.0 to 1.0
    //So either is fine for the output range I suppose
    cell_tanh_x_output_inputs[6] = (struct input){state->eqLSTM_node_list[output_gate_sigmoid_index]->node_id, 1};
    cell_tanh_x_output_inputs[7] = (struct input){state->eqLSTM_node_list[output_gate_sigmoid_index]->node_id, 2};

    uint32_t cell_tanh_x_output_index;
    cell_tanh_x_output_index = eqLSTM_node_list_start_index;
    state->eqLSTM_node_list[eqLSTM_node_list_start_index++] = create_node(nn, 0, OP_QuantizedMul_16, NN_PAD_NA, QAdd_16_INP_COUNT, ELE_WISE_16_OUT_COUNT, cell_tanh_x_output_inputs, ele_wise_output_defs);

    //Quantize down to uint8 before projection
    struct input quant_i16_to_u8_inputs[5];
    quant_i16_to_u8_inputs[0] = (struct input){state->eqLSTM_node_list[cell_tanh_x_output_index]->node_id, 0};
    quant_i16_to_u8_inputs[1] = (struct input){state->eqLSTM_node_list[cell_tanh_x_output_index]->node_id, 1};
    quant_i16_to_u8_inputs[2] = (struct input){state->eqLSTM_node_list[cell_tanh_x_output_index]->node_id, 2};
    //We are supposed to quantize into some input range for the projection layer.
    //I don't think it is necessary
    quant_i16_to_u8_inputs[3] = eqLSTM_node->input_refs[73];
    quant_i16_to_u8_inputs[4] = eqLSTM_node->input_refs[74];
    
    uint32_t quant_i16_to_u8_index;
    struct output ele_wise_output_u8_defs[3];
    make_outputdesc_from_shape(&ele_wise_output_u8_defs[0],&ele_wise_output_shape, tensor_type_size(NN_TYPE_QUINT8),0);
    ele_wise_output_u8_defs[1] = Output_ScalarFloat;
    ele_wise_output_u8_defs[2] = Output_ScalarFloat;

    quant_i16_to_u8_index = eqLSTM_node_list_start_index;
    state->eqLSTM_node_list[eqLSTM_node_list_start_index++] = create_node(nn, 0, OP_Convert_16_8, NN_PAD_NA, CONVERT_16_8_INP_COUNT, ELE_WISE_16_OUT_COUNT, quant_i16_to_u8_inputs, ele_wise_output_u8_defs);
    //Make projection matrix multiply
    struct shape projection_weights_shape;
    struct shape projection_output_shape;
    projection_weights_shape = get_shape_from_producer(nn, eqLSTM_node->input_refs[67]);
    projection_output_shape = compute_matmul_output_shape(ele_wise_output_shape, projection_weights_shape);
    struct input projection_mat_mul_inputs[6];
    
    projection_mat_mul_inputs[0] = (struct input){state->eqLSTM_node_list[quant_i16_to_u8_index]->node_id, 0};
    projection_mat_mul_inputs[1] = eqLSTM_node->input_refs[67];
    projection_mat_mul_inputs[2] = eqLSTM_node->input_refs[73];
    projection_mat_mul_inputs[3] = eqLSTM_node->input_refs[74];
    projection_mat_mul_inputs[4] = eqLSTM_node->input_refs[68];
    projection_mat_mul_inputs[5] = eqLSTM_node->input_refs[69];

    if(eqLSTM_node->n_inputs == 80){
        projection_mat_mul_inputs[0] = eqLSTM_node->input_refs[79];
        errlog(nn, "eqLSTM node_id: %u has debug golden projection input", eqLSTM_node->node_id);
    }
      
    struct output projection_mat_mul_outputs[3];
    make_outputdesc_from_shape(&projection_mat_mul_outputs[0], &projection_output_shape, tensor_type_size(NN_TYPE_INT32), 0);
    projection_mat_mul_outputs[1] = Output_ScalarFloat;
    projection_mat_mul_outputs[2] = Output_ScalarFloat;

    uint32_t projection_mat_mul_index;
    projection_mat_mul_index = eqLSTM_node_list_start_index;
    state->eqLSTM_node_list[eqLSTM_node_list_start_index++] = create_node(nn, 0, OP_QuantizedMatMul_8x8to32, NN_PAD_NA, MATMUL_U8_INP_COUNT, MATMUL_U8_OUT_COUNT, projection_mat_mul_inputs, projection_mat_mul_outputs);
    
    //Make the projection bias add
    struct input projection_bias_add_inputs[6];
    projection_bias_add_inputs[0] = (struct input){state->eqLSTM_node_list[projection_mat_mul_index]->node_id, 0};
    projection_bias_add_inputs[1] = eqLSTM_node->input_refs[70];
    projection_bias_add_inputs[2] = (struct input){state->eqLSTM_node_list[projection_mat_mul_index]->node_id, 1};
    projection_bias_add_inputs[3] = (struct input){state->eqLSTM_node_list[projection_mat_mul_index]->node_id, 2};
    projection_bias_add_inputs[4] = eqLSTM_node->input_refs[71];
    projection_bias_add_inputs[5] = eqLSTM_node->input_refs[72];
    //We will reuse the output defs from the Mat Mul

    uint32_t projection_bias_add_index;
    projection_bias_add_index = eqLSTM_node_list_start_index;
    state->eqLSTM_node_list[eqLSTM_node_list_start_index++] = create_node(nn, 0, OP_QuantizedBiasAdd_32p32to32, NN_PAD_NA, MATMUL_U8_INP_COUNT, MATMUL_U8_OUT_COUNT, projection_bias_add_inputs, projection_mat_mul_outputs);
    
    //Make the projection requantize_32to8
    struct input projection_requantize_inputs[5];
    projection_requantize_inputs[0] = (struct input){state->eqLSTM_node_list[projection_bias_add_index]->node_id, 0};
    projection_requantize_inputs[1] = (struct input){state->eqLSTM_node_list[projection_bias_add_index]->node_id, 1};
    projection_requantize_inputs[2] = (struct input){state->eqLSTM_node_list[projection_bias_add_index]->node_id, 2};
    projection_requantize_inputs[3] = eqLSTM_node->input_refs[75];
    projection_requantize_inputs[4] = eqLSTM_node->input_refs[76];
    
    struct output projection_requantize_outputs[3];
    make_outputdesc_from_shape(&projection_requantize_outputs[0], &projection_output_shape, tensor_type_size(NN_TYPE_QUINT8), 0);
    projection_requantize_outputs[1] = Output_ScalarFloat;
    projection_requantize_outputs[2] = Output_ScalarFloat;

    state->projection_requantize_index = eqLSTM_node_list_start_index;
    state->eqLSTM_node_list[eqLSTM_node_list_start_index++] = create_node(nn, 0, OP_Requantize_32to8, NN_PAD_NA, CONVERT_16_8_INP_COUNT, ELE_WISE_16_OUT_COUNT, projection_requantize_inputs, projection_requantize_outputs);
    state->new_cell_add_index = new_cell_add_index;
    return 0;
}

static inline void fill_input_gate_refs(struct nn_graph *nn, struct expand_eqLSTM_qASR_state *expand_state){
    expand_state->input_gate_refs.input_weights_id = 6U;
    expand_state->input_gate_refs.input_weights_min = 7U;
    expand_state->input_gate_refs.input_weights_max = 8U;
    expand_state->input_gate_refs.recurrent_weights_id = 18U;
    expand_state->input_gate_refs.recurrent_weights_min = 19U;
    expand_state->input_gate_refs.recurrent_weights_max = 20U;
    expand_state->input_gate_refs.gate_matmul_min = 30U;
    expand_state->input_gate_refs.gate_matmul_max = 31U;
    expand_state->input_gate_refs.gate_layernorm_coefficients = 38U;
    expand_state->input_gate_refs.gate_layernorm_coefficients_min = 39U;
    expand_state->input_gate_refs.gate_layernorm_coefficients_max = 40U;
    expand_state->input_gate_refs.gate_layernorm_bias = 52U;
    expand_state->input_gate_refs.gate_layernorm_bias_min = 53U;
    expand_state->input_gate_refs.gate_layernorm_bias_max = 54U;
    expand_state->input_gate_refs.gate_layernorm_min = 50U;
    expand_state->input_gate_refs.gate_layernorm_max = 51U;
    expand_state->input_gate_refs.apply_tanh = 0U;
}

static inline void fill_forget_gate_refs(struct nn_graph *nn, struct expand_eqLSTM_qASR_state *expand_state){
    expand_state->forget_gate_refs.input_weights_id = 9U;
    expand_state->forget_gate_refs.input_weights_min = 10U;
    expand_state->forget_gate_refs.input_weights_max = 11U;
    expand_state->forget_gate_refs.recurrent_weights_id = 21U;
    expand_state->forget_gate_refs.recurrent_weights_min = 22U;
    expand_state->forget_gate_refs.recurrent_weights_max = 23U;
    expand_state->forget_gate_refs.gate_matmul_min = 32U;
    expand_state->forget_gate_refs.gate_matmul_max = 33U;
    expand_state->forget_gate_refs.gate_layernorm_coefficients = 41U;
    expand_state->forget_gate_refs.gate_layernorm_coefficients_min = 42U;
    expand_state->forget_gate_refs.gate_layernorm_coefficients_max = 43U;
    expand_state->forget_gate_refs.gate_layernorm_bias = 55U;
    expand_state->forget_gate_refs.gate_layernorm_bias_min = 56U;
    expand_state->forget_gate_refs.gate_layernorm_bias_max = 57U;
    expand_state->forget_gate_refs.gate_layernorm_min = 50U;
    expand_state->forget_gate_refs.gate_layernorm_max = 51U;
    expand_state->forget_gate_refs.apply_tanh = 0U;
}

static inline void fill_cell_gate_refs(struct nn_graph *nn, struct expand_eqLSTM_qASR_state *expand_state){
    expand_state->cell_gate_refs.input_weights_id = 12U;
    expand_state->cell_gate_refs.input_weights_min = 13U;
    expand_state->cell_gate_refs.input_weights_max = 14U;
    expand_state->cell_gate_refs.recurrent_weights_id = 24U;
    expand_state->cell_gate_refs.recurrent_weights_min = 25U;
    expand_state->cell_gate_refs.recurrent_weights_max = 26U;
    expand_state->cell_gate_refs.gate_matmul_min = 34U;
    expand_state->cell_gate_refs.gate_matmul_max = 35U;
    expand_state->cell_gate_refs.gate_layernorm_coefficients = 44U;
    expand_state->cell_gate_refs.gate_layernorm_coefficients_min = 45U;
    expand_state->cell_gate_refs.gate_layernorm_coefficients_max = 46U;
    expand_state->cell_gate_refs.gate_layernorm_bias = 58U;
    expand_state->cell_gate_refs.gate_layernorm_bias_min = 59U;
    expand_state->cell_gate_refs.gate_layernorm_bias_max = 60U;
    expand_state->cell_gate_refs.gate_layernorm_min = 50U;
    expand_state->cell_gate_refs.gate_layernorm_max = 51U;
    expand_state->cell_gate_refs.apply_tanh = 1U;
}

static inline void fill_output_gate_refs(struct nn_graph *nn, struct expand_eqLSTM_qASR_state *expand_state){
    expand_state->output_gate_refs.input_weights_id = 15U;
    expand_state->output_gate_refs.input_weights_min = 16U;
    expand_state->output_gate_refs.input_weights_max = 17U;
    expand_state->output_gate_refs.recurrent_weights_id = 27U;
    expand_state->output_gate_refs.recurrent_weights_min = 28U;
    expand_state->output_gate_refs.recurrent_weights_max = 29U;
    expand_state->output_gate_refs.gate_matmul_min = 36U;
    expand_state->output_gate_refs.gate_matmul_max = 37U;
    expand_state->output_gate_refs.gate_layernorm_coefficients = 47U;
    expand_state->output_gate_refs.gate_layernorm_coefficients_min = 48U;
    expand_state->output_gate_refs.gate_layernorm_coefficients_max = 49U;
    expand_state->output_gate_refs.gate_layernorm_bias = 61U;
    expand_state->output_gate_refs.gate_layernorm_bias_min = 62U;
    expand_state->output_gate_refs.gate_layernorm_bias_max = 63U;
    expand_state->output_gate_refs.gate_layernorm_min = 50U;
    expand_state->output_gate_refs.gate_layernorm_max = 51U;
    expand_state->output_gate_refs.apply_tanh = 0U;
}

static int replace_eqLSTM_qASR_node(struct nn_graph *nn, struct nn_node *eqLSTM_node){
    struct expand_eqLSTM_qASR_state *expand_state;
    //How many nodes do we need?
    //20 nodes for the gates (5 per gate)
    //3 to compute new cell state
    //2 to compute output_gate x tanh(new_cell_state)
    //4 to compute projection (qint16_to_quint8, MatMul, BiasAdd, Requant)
    //Total 29
    uint32_t new_nodes_per_eqLSTM = 29U;
    //Because our interfaced changed now this is variable
    //Because of course it is....
    nn_scratch_reset(nn);
    if(0 != nn_scratch_grow(nn, sizeof(struct expand_eqLSTM_qASR_state) + sizeof(struct nn_node*)*new_nodes_per_eqLSTM)){
        errlog(nn, "Error expanding eqLSTM qASR OP id: %u, could not allocate scratch space", eqLSTM_node->node_id);
        return NN_PREPARE_OUT_OF_SCRATCH_ERROR;
    }
    expand_state = nn_scratch_alloc(nn, sizeof(struct expand_eqLSTM_qASR_state));
    expand_state->has_cell_clip = 0U;
    const struct tensor *clip_tensor = eqLSTM_node->inputs[79];
    if(0.0f != tensor_get_float(clip_tensor, 0)){
        expand_state->has_cell_clip=1U;
        new_nodes_per_eqLSTM = 30U;
    }
    expand_state->eqLSTM_node_list = nn_scratch_alloc(nn, sizeof(struct nn_node*)*new_nodes_per_eqLSTM);

    uint32_t layernorm_axis_const_nid = nn_graph_new_internal_node_id(nn);
    //For qASR LSTM we reduce upto axis 2 (IE width)
    int32_t reduction_axis = 2;
    if((do_prepend_const_node(nn, layernorm_axis_const_nid, 1, 1, 1,1, (const uint8_t *)(&reduction_axis), sizeof(int32_t))) != 0){
        errlog(nn, "Error expanding eqLSTM qASR OP id: %u, could not create layernorm axis const node", eqLSTM_node->node_id);
        nn_scratch_reset(nn);
        return NN_PREPARE_NEW_CONST_ERROR;
    }
    expand_state->fake_bias_node_id = nn_graph_new_internal_node_id(nn);
    //Need to know the depth for the bias.
    //Luckily the depth of the cell state is the same as the output depth of the gate matmuls
    //So we use that shape
    struct shape cell_state_shape = get_shape_from_producer(nn, eqLSTM_node->input_refs[64]);
    struct tensor const* bias_tensor = do_prepend_const_node_ptr(nn, expand_state->fake_bias_node_id, 1,1,1, cell_state_shape.depth,NULL, cell_state_shape.depth*sizeof(int32_t));
    if(bias_tensor == NULL){
        errlog(nn, "Error expanding eqLSTM qASR OP id: %x, could not create zero bias tensor", eqLSTM_node->node_id);
        nn_scratch_reset(nn);
        return NN_PREPARE_NEW_CONST_ERROR;
    }
    memset(bias_tensor->data, 0, cell_state_shape.depth*sizeof(int32_t));
    //Create min/max node for zero bias
    expand_state->fake_bias_q_params_id = nn_graph_new_internal_node_id(nn);
    const float zero = 0.0f;
    if((do_prepend_const_node(nn, expand_state->fake_bias_q_params_id, 1,1,1,1, (const uint8_t *)&zero, sizeof(float))) != 0){
        errlog(nn, "Error expanding eqLSTM qASR OP id: %x, could not create zero bias range tensor", eqLSTM_node->node_id);
        nn_scratch_reset(nn);
        return NN_PREPARE_NEW_CONST_ERROR;
    }
    expand_state->layernorm_axis_const_node_id = layernorm_axis_const_nid;
    fill_input_gate_refs(nn, expand_state);
    fill_forget_gate_refs(nn, expand_state);
    fill_cell_gate_refs(nn, expand_state);
    fill_output_gate_refs(nn, expand_state);
    if((expand_eqLSTM_qASR_node(nn, eqLSTM_node, expand_state))!=0){
        errlog(nn, "Error expanding eqLSTM qASR OP id: %u", eqLSTM_node->node_id);
        return NN_PREPARE_ERROR;
    }
    uint32_t projection_requantize_id = expand_state->eqLSTM_node_list[expand_state->projection_requantize_index]->node_id;
    //We need to replace references to LSTM outputs as follows:
    //ref to output 0 -> requant output 0
    //ref to output 1 -> requant output 1
    //ref to output 2 -> requant output 2
    //ref to output 3 -> handled later
    //ref to output 4 -> handled later
    //ref to output 5 -> handled later
    //ref to output 6 -> requant 0
    //ref to output 7 -> requant 1
    //ref to output 8 -> requant 2
    uint64_t output_ref_change_pattern = 0x321FFF321ULL;
    //Let's assume we don't know how many output_refs should be changed
    //Therefore error occurs when change_output_refs returns a value < 0
    if((change_output_refs(nn, NULL, eqLSTM_node->node_id, projection_requantize_id, output_ref_change_pattern)) < 0){
        errlog(nn, "Error expanding eqLSTM qASR OP id: %u, could not change output refs to projection requantize id: %u", eqLSTM_node->node_id, projection_requantize_id);
        nn_scratch_reset(nn);
        return NN_PREPARE_ERROR;
    }
    uint32_t new_cell_add_id = expand_state->eqLSTM_node_list[expand_state->new_cell_add_index]->node_id;
    
    //Since we already handled the output refs above
    //now we just need to replace references to the cell state output
    //ref to output 0 -> handled already
    //ref to output 1 -> handled already
    //ref to output 2 -> handled already
    //ref to output 3 -> new cell add 0
    //ref to output 4 -> new cell add 1
    //ref to output 5 -> new cell add 2
    //ref to output 6 -> handled already
    //ref to output 7 -> handled already
    //ref to output 8 -> handled already
    output_ref_change_pattern = 0xFFF321FFFULL;
    if((change_output_refs(nn, NULL, eqLSTM_node->node_id, new_cell_add_id, output_ref_change_pattern)) < 0){
        errlog(nn, "Error expanding eqLSTM qASR OP id: %u, could not change new cell refs to new cell add id: %u", eqLSTM_node->node_id, new_cell_add_id);
        nn_scratch_reset(nn);
        return NN_PREPARE_ERROR;
    }
    //Time to replace
    if((replace_node_with_sequence(nn, NULL, eqLSTM_node, expand_state->eqLSTM_node_list, new_nodes_per_eqLSTM)) < 0){
        errlog(nn, "Error expanding eqLSTM qASR OP id: %u, could not replace node with sequence", eqLSTM_node->node_id);
        nn_scratch_reset(nn);
        return NN_PREPARE_ERROR;
    }
    //We should be able to free the expand state and node list now
    nn_scratch_reset(nn);
    return 0;
}

int find_and_replace_eqLSTM_nodes(struct nn_graph *nn, struct nn_node **node_to_test_p){
    struct nn_node *node_to_test = *node_to_test_p;
    if(node_to_test->node_type == OP_EqLSTM_qASR){
        //Found one!
        return replace_eqLSTM_qASR_node(nn, node_to_test);
    }
    return 0;
}
