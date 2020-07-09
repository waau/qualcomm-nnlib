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

#ifndef EXPAND_EQLSTM_NODE_H
#define EXPAND_EQLSTM_NODE_H 1
#include "nn_prepare.h"
#include "expand_nodes.h"
#include <nn_graph.h>
#include <nn_graph_types.h>
#include <data_utils.h>

struct eqLSTM_gate_input_refs {
    uint32_t input_weights_id;
    uint32_t input_weights_min;
    uint32_t input_weights_max;
    uint32_t recurrent_weights_id;
    uint32_t recurrent_weights_min;
    uint32_t recurrent_weights_max;
    uint32_t gate_matmul_min;
    uint32_t gate_matmul_max;
    uint32_t gate_layernorm_coefficients;
    uint32_t gate_layernorm_coefficients_min;
    uint32_t gate_layernorm_coefficients_max;
    uint32_t gate_layernorm_bias;
    uint32_t gate_layernorm_bias_min;
    uint32_t gate_layernorm_bias_max;
    uint32_t gate_layernorm_min;
    uint32_t gate_layernorm_max;
    uint32_t apply_tanh;
};

struct expand_eqLSTM_qASR_state {
    struct nn_node **eqLSTM_node_list;
    uint32_t new_cell_add_index;
    uint32_t projection_requantize_index;
    uint32_t layernorm_axis_const_node_id;
    uint32_t fake_bias_node_id;
    uint32_t fake_bias_q_params_id;
    uint32_t has_cell_clip;
    struct eqLSTM_gate_input_refs input_gate_refs;
    struct eqLSTM_gate_input_refs forget_gate_refs;
    struct eqLSTM_gate_input_refs cell_gate_refs;
    struct eqLSTM_gate_input_refs output_gate_refs;
};




int find_and_replace_eqLSTM_nodes(struct nn_graph *nn, struct nn_node **node_to_test_p);
#endif
