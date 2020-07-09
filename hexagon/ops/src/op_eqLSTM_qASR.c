
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
 * This contains the descriptor of the enhanced qLSTM
 * for ASR
 * It doesn't actually do anything but act as an interface
 * so that the node can be appended to a graph
 */

#include <nn_graph.h>
#include <string.h>
#include <quantize.h>
#include <math.h>
#include <hvx_hexagon_protos.h>
#if defined(__hexagon__)
#include <hexagon_types.h>
#endif
/*
Op inputs and outputs
Inputs:
    0: input (quint8 tensor)
    1: input min (float scalar)
    2: input max (float scalar)
    3: recurrent activation (quint8 tensor)
    4: recurrent min (float scalar)
    5: recurrent max (float scalar)
    6: input to input weights (quint8 tensor)
    7: i2i weight min (float scalar)
    8: i2i weight max (float scalar)
    9: input to forget weights (quint8 tensor) 
    10: i2f weight min (float scalar)
    11: i2f weight max (float scalar)
    12: input to cell weights (quint8 tensor) 
    13: i2c weight min (float scalar)
    14: i2c weight max (float scalar)
    15: input to output weights (quint8 tensor) 
    16: i2o weight min (float scalar)
    17: i2o weight max (float scalar)
    18: recurrent to input weights (quint8 tensor)
    19: r2i weight min (float scalar)
    20: r2i weight max (float scalar)
    21: recurrent to forget weights (quint8 tensor) 
    22: r2f weight min (float scalar)
    23: r2f weight max (float scalar)
    24: recurrent to cell weights (quint8 tensor) 
    25: r2c weight min (float scalar)
    26: r2c weight max (float scalar)
    27: recurrent to output weights (quint8 tensor) 
    28: r2o weight min (float scalar)
    29: r2o weight max (float scalar)
    30: input gate matmul min (float scalar)
    31: input gate matmul max (float scalar)
    32: forget gate matmul min (float scalar)
    33: forget gate matmul max (float scalar)
    34: cell gate matmul min (float scalar)
    35: cell gate matmul max (float scalar)
    36: output gate matmul min (float scalar)
    37: output gate matmul max (float scalar)
    38: input gate layer norm scale (qint16 tensor)
    39: input gate LN scale min (float scalar)
    40: input gate LN scale max (float scalar)
    41: forget gate layer norm scale (qint16 tensor)
    42: forget gate LN scale min (float scalar)
    43: forget gate LN scale max (float scalar)
    44: cell gate layer norm scale (qint16 tensor)
    45: cell gate LN scale min (float scalar)
    46: cell gate LN scale max (float scalar)
    47: output gate layer norm scale (qint16 tensor)
    48: output gate LN scale min (float scalar)
    49: output gate LN scale max (float scalar)
    50: LSTM LayerNorm output min (float scalar)
    51: LSTM LayerNorm output max (float scalar)
    52: input gate bias (qint32 tensor)
    53: input gate bias min (float scalar)
    54: input gate bias max (float scalar)
    55: forget gate bias (qint32 tensor)
    56: forget gate bias min (float scalar)
    57: forget gate bias max (float scalar)
    58: cell gate bias (qint32 tensor)
    59: cell gate bias min (float scalar)
    60: cell gate bias max (float scalar)
    61: output gate bias (qint32 tensor)
    62: output gate bias min (float scalar)
    63: output gate bias max (float scalar)
    64: input cell state (qint16 tensor)
    65: input cell state min (float scalar)
    66: input cell state max (float scalar)
    67: projection weights (quint8 tensor)
    68: projection weights min (float scalar)
    69: projection weights max (float scalar)
    70: projection bias (qint32 tensor)
    71: projection bias min (float scalar)
    72: projection bias max (float scalar)
    73: projection input min (float scalar)
    74: projection input max (float scalar)
    75: LSTM output min (float scalar) <- output scale
    76: LSTM output max (float scalar) <- output scale
    77: new cell state tanh input min (float scalar)
    78: new cell state tanh input max (float scalar)
    79: cell clipping value (float scalar)
    80: projection clipping value (float scalar) (MUST BE 0.0 currently)
Outputs:
    0: output state (quint8 tensor)
    1: output state min (float scalar) <- equal to input 75
    2: output state max (float scalar) <- equal to input 76
    3: output cell state (quint16 tensor)
    4: output cell state min (float scalar) <- equal to input 65
    5: output cell state max (float scalar) <- equal to input 66
    6: output (quint8 tensor) <- equal to output 0
    7: output min (float scalar) <- equal to output 1
    8: output max (float scalar) <- equal to output 2

*/

struct nn_node_ops nn_ops_for_EqLSTM_qASR = {
    .execute = NULL,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT(81),
    .n_outputs = NN_IOCOUNT(9),
    .flags = NN_NODE_FLAG_CLS_EQLSTM,
};