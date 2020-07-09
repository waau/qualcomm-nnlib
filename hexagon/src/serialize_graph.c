
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
 *
 */
#include "serialize_graph.h"
#include "nn_graph.h"

int serialize(struct nn_graph * nn, uint8_t * cache_buffer, uint32_t buf_len) {
    uint8_t * cache_pointer = (uint8_t *) (cache_buffer + sizeof(struct cache_head));
    struct cache_head * head = (struct cache_head *) cache_buffer;
    struct nn_node *node;
    int nn_version;
    uint32_t num_nodes = 0;
    uint32_t total_size = sizeof(struct cache_head);
    hexagon_nn_version(&nn_version);
    for (node = nn->head; node != NULL; node = node->next) {
        // Check that there's enough space in the buffer to serialize this node
        total_size += get_record_size(node);
        if (total_size > buf_len) {
            errlog(nn, "Error: Buffer not large enough to serialize graph");
            return NN_SERIALIZATION_BUFFER_SIZE_ERROR;
        }

        // UDOs have node_type=NN_OPS_MAX this works for serializing, but this isn't a safe way to check when deserializing
        // as the number of OPS in the op table can change and what used to be NN_OPS_MAX is now a hex-nn op
        if (node->node_type == NN_OPS_MAX) {
            errlog(nn, "Serializing UDOs is not supported");
            return NN_SERIALIZATION_UDO_ERROR;
        }
        
        num_nodes++;
        uint32_t alloc_size = 0;
        if (node->node_type == OP_Const) {
            alloc_size = node->outputs[0]->data_size;
        }
        struct record_head current_record = {
                .node_type = node->node_type,
                .node_id = node->node_id,
                .padding_type = node->padding,
                .n_inputs = node->n_inputs,
                .n_outputs = node->n_outputs,
                .alloc_size = alloc_size
        };
        memcpy(cache_pointer, &current_record, sizeof(struct record_head));
        cache_pointer += sizeof(struct record_head);
        for (int i = 0; i < node->n_inputs; i++) {
            memcpy(cache_pointer, &node->input_refs[i], sizeof(struct input));
            cache_pointer += sizeof(struct input);
        }
        for (int i = 0; i < node->n_outputs; i++) {
            memcpy(cache_pointer, &node->output_defs[i], sizeof(struct output));
            cache_pointer += sizeof(struct output);
        }
        if (alloc_size != 0){
            memcpy(cache_pointer, node->outputs[0]->data, alloc_size);
            cache_pointer += padded_data_size(alloc_size);
        }
    }
    *head = (struct cache_head) {
        .nn_version = nn_version,
        .signature = SIGNATURE,
        .cache_size = total_size,
        .num_records = num_nodes,
        .graph_state = nn->state
    };
    return NN_SERIALIZATION_SUCCESS;
}


int serialize_size(struct nn_graph *nn, uint32_t * buffer_size) {
    struct nn_node *node;
    uint32_t total_size = sizeof(struct cache_head);
    for (node = nn->head;  node != NULL; node = node->next) {
        total_size += get_record_size(node);
    }
    *buffer_size = total_size;
    return NN_SERIALIZATION_SUCCESS;
}


int deserialize(struct nn_graph * nn, uint32_t buf_len, const uint8_t * cache_buffer) {
    uint8_t *record_pointer = (uint8_t *) (cache_buffer + sizeof(struct cache_head));
    struct cache_head *head = (struct cache_head *) cache_buffer;
    uint32_t total_parsed = sizeof(struct cache_head);
    int current_nn_version;
    hexagon_nn_version(&current_nn_version);
    
    if (head->nn_version != current_nn_version) {
        errlog(nn, "Error: Cached Graph was cached using a different version of Hexagon-nn");
        return NN_SERIALIZATION_INVALID_CACHE;
    }
    if (head->signature != SIGNATURE) {
        errlog(nn, "Error: Cached Graph is invalid");
        return NN_SERIALIZATION_INVALID_CACHE;
    }
    if (head->graph_state != NN_GRAPH_PREPARED) {
        errlog(nn, "Error: Cached Graph must be prepared to deserialize");
        return NN_SERIALIZATION_GRAPH_NOT_PREPARED;
    }
    if (head->cache_size != buf_len) {
        errlog(nn, "Error: Size of Cached graph doesn't match Cache buffer size");
        return NN_SERIALIZATION_BUFFER_SIZE_ERROR;
    }
       
    
    for (int r = 0; r < head->num_records; r++) {
        uint32_t offset = 0;
        struct record_head record = *(struct record_head *) record_pointer;
        offset += sizeof(struct record_head);
        const struct input *inputs_ptr = (struct input *) (record_pointer + offset);
        offset += sizeof(struct input) * record.n_inputs;
        const struct output *outputs_ptr = (struct output *) (record_pointer + offset);
        offset += sizeof(struct output) * record.n_outputs;

        uint32_t record_size = offset + padded_data_size(record.alloc_size);

        if (total_parsed + record_size > buf_len) {
            errlog(nn, "Error: Buffer size doesn't match required space for serialized graph");
            return NN_SERIALIZATION_BUFFER_SIZE_ERROR;
        }
        
        uint8_t * data = record_pointer + offset;
        if (record.alloc_size == 0) data = NULL;

        if(record.node_type == OP_Const) {
            if(do_append_const_node(nn,
                    record.node_id,
                    outputs_ptr->max_sizes[0],
                    outputs_ptr->max_sizes[1],
                    outputs_ptr->max_sizes[2],
                    outputs_ptr->max_sizes[3],
                    data,
                    record.alloc_size)){
                errlog(nn, "Error: Failed to deserialize const node with id=%d", record.node_id);
                return NN_SERIALIZATION_ERROR;
            }
        } else {
            if (do_append_node(nn,
                    record.node_id,
                    record.node_type,
                    record.padding_type,
                    record.n_inputs,
                    record.n_outputs,
                    inputs_ptr,
                    outputs_ptr)) {
                errlog(nn, "Error: Failed to deserialize node with id=%d", record.node_id);
                return NN_SERIALIZATION_ERROR;
            }
        }
        record_pointer += record_size;
        total_parsed += record_size;
    }
    if (total_parsed != buf_len) {
        errlog(nn, "Error: Cache buffer is not the correct size for the cached graph");
        return NN_SERIALIZATION_BUFFER_SIZE_ERROR;
    }
    return NN_SERIALIZATION_SUCCESS;
}

uint32_t padded_data_size(uint32_t data_size) {
    uint32_t remainder = data_size % 4;
    return data_size + (remainder == 0 ? 0 : (4 - (remainder))); // 4 byte alignment
}


uint32_t get_record_size(struct nn_node *node){
    uint32_t record_size = sizeof(struct record_head) + sizeof(struct input)*node->n_inputs + sizeof(struct output)*node->n_outputs;
    if (node->node_type == OP_Const) {
        record_size += padded_data_size(node->outputs[0]->data_size);
    }
    return record_size;
}