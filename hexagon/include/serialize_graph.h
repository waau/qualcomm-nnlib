
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

#ifndef SERIALIZE_GRAPH_H_
#define SERIALIZE_GRAPH_H_ 1
#define SIGNATURE 0xac001516
#include <nn_graph.h>

enum hexagon_nn_serialize_results {
    NN_SERIALIZATION_SUCCESS = 0,  
    NN_SERIALIZATION_ERROR,
    NN_SERIALIZATION_BUFFER_SIZE_ERROR,
    NN_SERIALIZATION_UDO_ERROR,
    NN_SERIALIZATION_GRAPH_NOT_FOUND,
    NN_SERIALIZATION_GRAPH_NOT_PREPARED,
    NN_SERIALIZATION_INVALID_CACHE,
};

int serialize_size(struct nn_graph *, uint32_t *);
int serialize(struct nn_graph *, uint8_t *, uint32_t);
int deserialize(struct nn_graph *, uint32_t, const uint8_t *);
uint32_t padded_data_size(uint32_t data_size);
uint32_t get_record_size(struct nn_node *node);

struct record_head {
    uint32_t node_type;
    uint32_t node_id;
    uint32_t padding_type;
    uint32_t n_inputs;
    uint32_t n_outputs;
    uint32_t alloc_size;
};

struct cache_head {
    int nn_version;
    uint32_t signature;
    uint32_t cache_size;
    uint32_t num_records;
    uint32_t graph_state;
};

#endif