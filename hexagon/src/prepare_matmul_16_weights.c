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

#include <prepare_matmul_16_weights.h>

int prepare_matmul_16_weights(struct nn_graph *nn, struct nn_node *matmul_node);

int find_and_prepare_matmul_16_weights(struct nn_graph *nn, struct nn_node **node_to_test_p){
    struct nn_node *node_to_test = *node_to_test_p;
    if(node_to_test->node_type == OP_QuantizedMatMul_8x8p32to16 && node_to_test->n_inputs != 12){
        return prepare_matmul_16_weights(nn, node_to_test);
    }
    return 0;
}

int prepare_matmul_16_weights(struct nn_graph *nn, struct nn_node *matmul_node){
    //Only const propogate if we would be using HVX
    //Get tensors from input refs
    struct nn_node *const_node_ptr;
    struct nn_node *weights_node_ptr;
    struct nn_node **weight_consumers_list;
    const_node_ptr = find_node_must_be_Const_from_ref(nn, &matmul_node->input_refs[1]);
    if(NULL == const_node_ptr){
        //Tensor not const abort
        logmsg(nn,2, "QuantizedMatMul_8x8p32to16 id: %x couldn't have weights prepared got non-const weights", matmul_node->node_id);
        return 0;
    }
    struct tensor const *b_tensor = const_node_ptr->outputs[0];
    weights_node_ptr = const_node_ptr;
    const_node_ptr = find_node_must_be_Const_from_ref(nn, &matmul_node->input_refs[4]);
    if(NULL == const_node_ptr){
        //Tensor not const abort
        logmsg(nn,2, "QuantizedMatMul_8x8p32to16 id: %x couldn't have weights prepared got non-const weights min", matmul_node->node_id);
        return 0;
    }
    struct tensor const *b_min_tensor = const_node_ptr->outputs[0];
    const_node_ptr = find_node_must_be_Const_from_ref(nn, &matmul_node->input_refs[5]);
    if(NULL == const_node_ptr){
        //Tensor not const abort
        logmsg(nn,2, "QuantizedMatMul_8x8p32to16 id: %x couldn't have weights prepared got non-const weights max", matmul_node->node_id);
        return 0;
    }
    struct tensor const *b_max_tensor = const_node_ptr->outputs[0];
    uint32_t num_weight_consumers = 0U;
    int res = count_all_consumers(nn, weights_node_ptr, OP_QuantizedMatMul_8x8p32to16, &num_weight_consumers);
    if(res != 0){
        return errlog(nn, "Error preparing QuantizedMatMul_8x8p32to16 weights count_all_consumers returned non-zero status: %d", res);
    }
    if(num_weight_consumers == 0U){
        return errlog(nn, "Error preparing QuantizedMatMul_8x8p32to16 weights count_all_consumers returned zero consumers for weight node");
    }
    if(num_weight_consumers > 1U){
        logmsg(nn, 2, "QuantizedMatMul_8x8p32to16 found %u nodes that share weights", num_weight_consumers);
        //Multiple consumers must be careful
        weight_consumers_list = nn_calloc(num_weight_consumers,sizeof(struct nn_node *));
        if(weight_consumers_list == NULL){
            return errlog(nn, "Error preparing QuantizedMatMul_8x8p32to16 weights could not allocate consumer list. Null allocation return");   
        }
        //Find all consumers
        res = find_all_consumers(nn, weights_node_ptr, OP_QuantizedMatMul_8x8p32to16, weight_consumers_list, num_weight_consumers);
        if(res!=0){
            nn_free(weight_consumers_list);
            return errlog(nn, "Error preparing QuantizedMatMul_8x8p32to16 weights find_all_consumers returned non-zero status: %d", res);
        }
    }
    int d_in = b_tensor->shape.width;
    int d_out = b_tensor->shape.depth;
    if( (d_in & 7) != 0 && (d_out & 63 ) != 0 ){
        //Wouldn't use hvx abort
        return 0;
    }
    int d_in_padded = (d_in + 31) & ~31;
    int d_out_padded = d_out;
    //Create new const nodes
    uint32_t repacked_weights_nid = nn_graph_new_internal_node_id(nn);
    uint32_t gemmsumb_nid = nn_graph_new_internal_node_id(nn);
    //uint32_t weights_prepared_nid = nn_graph_new_internal_node_id(nn);
    //uint8_t one = 1U;
    //struct tensor *weights_prepared_tensor = do_prepend_const_node_ptr(nn, weights_prepared_nid,1,1,1,1,&one, 1);
    const struct tensor *repacked_tensor = do_prepend_const_node_ptr(nn, repacked_weights_nid, 1,1,d_in_padded,d_out_padded, NULL,
        d_in_padded*d_out_padded);

    const struct tensor *gemmsumb_tensor = do_prepend_const_node_ptr(nn, gemmsumb_nid, 1,1,1,d_out_padded, NULL, sizeof(int32_t)*d_out_padded);
    // call a function which will reorder the weights and find the sums.
    struct repack_filter_parms rpfp;
    // find B input range
    float b_in_min = tensor_get_float( b_min_tensor, 0);
    float b_in_max = tensor_get_float( b_max_tensor, 0);

    //float b_in_step = flt_div_255(b_in_max-b_in_min);
    int b_in_zero = saturate_u8( roundf_i32( b_in_min *-255.0f/(b_in_max-b_in_min)));
    rpfp.filt_tensor = b_tensor;
    rpfp.zero_offset = b_in_zero;
    rpfp.signed_mode_sel = 0;
    rpfp.out_data = repacked_tensor->data;
    rpfp.gemsumb = gemmsumb_tensor->data;
    nn_sem_init( &rpfp.done_sem, 0);
    nn_os_vector_workers_acquire(nn);
    nn_os_work_for_vector( nn,repack_filter_for_d32, &rpfp);
    nn_sem_wait( &rpfp.done_sem);
    nn_os_vector_workers_release(nn);
    //Flush data cache
    supernode_cleaninv_weights(repacked_tensor->data, d_in_padded*d_out_padded);
    //If single consumer path
    uint32_t new_matmul_input_count = 12u;
    uint32_t new_matmul_output_count = 3u;
    if(num_weight_consumers == 1U){
        //Replace existing node with an equivalent node
        struct input new_matmul_inputs[12];
        for (int i = 0; i < 11; ++i){
            new_matmul_inputs[i] = matmul_node->input_refs[i];
        }
        new_matmul_inputs[1] = (struct input){repacked_weights_nid, 0};
        new_matmul_inputs[11] = (struct input){gemmsumb_nid, 0};   
        struct nn_node *new_matmul_node;
        new_matmul_node = create_node(nn, matmul_node->node_id, OP_QuantizedMatMul_8x8p32to16, NN_PAD_NA, new_matmul_input_count, new_matmul_output_count, new_matmul_inputs, matmul_node->output_defs);
        if(new_matmul_node == NULL){
            return errlog(nn, "error creating new QuantizedMatMul_8x8p32to16 node with prepared weights");
        }
        res = replace_nodes(nn, NULL, new_matmul_node, matmul_node);
        if(res!=0){
            return errlog(nn, "error replacing QuantizedMatMul_8x8p32to16 with prepared weights version. replace_nodes returned non-zero status: %d", res);
        }
    }
    else{
        //Iterate over the consumer_list
        struct input new_matmul_inputs[12];
        struct nn_node *new_matmul_node;
        struct nn_node *curr_matmul_node;
        int found_self = 0;
        for(uint32_t i = 0; i < num_weight_consumers; ++i){
            curr_matmul_node = weight_consumers_list[i];
            if(curr_matmul_node->node_id == matmul_node->node_id){
                //Kind of sanity test we should find the node that triggered all this
                //But also should only find it once
                ++found_self;
            }
            for(uint32_t j = 0; j < 11; ++j){
                new_matmul_inputs[j] = curr_matmul_node->input_refs[j];
            }
            new_matmul_inputs[1] = (struct input){repacked_weights_nid, 0};
            new_matmul_inputs[11] = (struct input){gemmsumb_nid, 0};
            new_matmul_node = create_node(nn, curr_matmul_node->node_id, OP_QuantizedMatMul_8x8p32to16, NN_PAD_NA, new_matmul_input_count, new_matmul_output_count, new_matmul_inputs, curr_matmul_node->output_defs);
            if(new_matmul_node == NULL){
                nn_free(weight_consumers_list);
                return errlog(nn, "error creating new QuantizedMatMul_8x8p32to16 node with prepared weights");
            }
            res = replace_nodes(nn, NULL, new_matmul_node, curr_matmul_node);
            if(res!=0){
                nn_free(weight_consumers_list);
                return errlog(nn, "error replacing QuantizedMatMul_8x8p32to16 with prepared weights version. replace_nodes returned non-zero status: %d", res);
            }
        }
        if(found_self > 1U || found_self == 0U){
            nn_free(weight_consumers_list);
            return errlog(nn, "error while replacing QuantizedMatMul_8x8p32to16 ops, found originator op: %x %u times. Should be only 1", matmul_node->node_id, found_self);
        }
    }
    //Victory!
    if(num_weight_consumers > 1U){
        nn_free(weight_consumers_list);
    }
    return 0;
}


