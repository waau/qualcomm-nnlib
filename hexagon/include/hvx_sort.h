#ifndef HVX_SORT_H
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
#include "hvx_inlines.h"
#include <stdint.h>
#include "hvx_inlines.h"


// NOTE: These are inline functions which are intended to be used in loops.
// in ALL cases the REVERSE parameter must be a compile time-constant when the
// function is expanded, otherwise the code will be *terrible* (some functions
// have REVERSE_A and REVERSE_B, which may be different; and this applies to that too).

// ===================================================================== //
// function: SORT_OVER_TWO
// description: takes in two HVX vectors (VEC0 and VEC1) and a direction REV,
//              compares the two vectors word-wise.
//              REV = 0, the smaller elements will be in t_res_low
//              REV = 1, the larger elements will be in t_res_low
// ===================================================================== //
#define SORT_OVER_TWO( VEC0, VEC1, REV )\
   ({ HVX_VectorPair t_res;\
    if( !(REV)){\
        t_res = Q6_W_vcombine_VV( Q6_Vw_vmax_VwVw(VEC0,VEC1),Q6_Vw_vmin_VwVw(VEC0,VEC1));\
    }else{\
        t_res = Q6_W_vcombine_VV( Q6_Vw_vmin_VwVw(VEC0,VEC1),Q6_Vw_vmax_VwVw(VEC0,VEC1));\
    }; t_res;})

// ===================================================================== //
// function: hvx_sort_bitonic_seq_32
// description: takes in a vector pair with up to 64 elements. The even
//              and odd lanes of this vector pair will hold the 32 elements
//              from batch_1 and batch_2, respectively. This function will
//              perform bitonic sort on the two batches.
// output: sorted elements from batch_1 in vpRes_low, sorted elements from
//         batch_2 in vpRes_high
// NOTE: the input elements (in the even and odd lanes) are expected to be
//       in bitonic order. It will not work properly otherwise.
// ===================================================================== //
static inline HVX_VectorPair __attribute__((always_inline,unused)) 
hvx_sort_bitonic_seq_32( HVX_VectorPair vpIn, int REVERSE )
{
    // this is a straight forward sort network on a bitonic sequence of length 32
    HVX_VectorPair vpRes = SORT_OVER_TWO(Q6_V_hi_W(vpIn), Q6_V_lo_W(vpIn), REVERSE);
    vpRes = Q6_W_vdeal_VVR(Q6_V_hi_W(vpRes), Q6_V_lo_W(vpRes), 4*16);
    vpRes = SORT_OVER_TWO(Q6_V_lo_W(vpRes), Q6_V_hi_W(vpRes), REVERSE);
    vpRes = Q6_W_vdeal_VVR(Q6_V_hi_W(vpRes), Q6_V_lo_W(vpRes), 4*8);
    vpRes = SORT_OVER_TWO(Q6_V_lo_W(vpRes), Q6_V_hi_W(vpRes), REVERSE);
    vpRes = Q6_W_vdeal_VVR(Q6_V_hi_W(vpRes), Q6_V_lo_W(vpRes), 4*4);
    vpRes = SORT_OVER_TWO(Q6_V_lo_W(vpRes), Q6_V_hi_W(vpRes), REVERSE);
    vpRes = Q6_W_vdeal_VVR(Q6_V_hi_W(vpRes), Q6_V_lo_W(vpRes), 4*2);
    vpRes = SORT_OVER_TWO(Q6_V_lo_W(vpRes), Q6_V_hi_W(vpRes), REVERSE);
    // now that all the comparisons are done, reorder the vector back into the correct format
    vpRes = Q6_W_vdeal_VVR(Q6_V_hi_W(vpRes), Q6_V_lo_W(vpRes), 4*2);
    vpRes = Q6_W_vdeal_VVR(Q6_V_hi_W(vpRes), Q6_V_lo_W(vpRes), 4*4);
    vpRes = Q6_W_vdeal_VVR(Q6_V_hi_W(vpRes), Q6_V_lo_W(vpRes), 4*8);
    vpRes = Q6_W_vdeal_VVR(Q6_V_hi_W(vpRes), Q6_V_lo_W(vpRes), 4*16);

    return vpRes;
}

// ===================================================================== //
// function: hvx_convert_to_bitonic_seq_32
// description: takes in a vector pair with up to 64 elements. The even
//              and odd lanes of this vector pair will hold the 32 elements
//              from batch_1 and batch_2, respectively. This function will
//              build a bitonic sequence for each of the two batches.
// output: bitonic sequence of elements in batch_1 in vpRes_low, 
//         bitonic_sequence of elements in batch_2 in vpRes_high.
// NOTES: a bitonic sequence x of length N is a sequence which has the 
//        following structure,
//        x_0 < x_1 < x_2 < ... < x_M > x_(M+1) > ... > x_(N-1) > x_N.
//        In other words, it consists of a monotonically increasing sequence
//        followed by a monotonically decreasing sequence. The reverse is
//        also a bitonic sequence (decreasing seq followed by increasing seq).
// ===================================================================== //
static inline HVX_VectorPair __attribute__((always_inline,unused))
hvx_convert_to_bitonic_seq_32( HVX_VectorPair vpIn, int REVERSE )
{
    vpIn = Q6_W_vdeal_VVR( Q6_V_hi_W( vpIn ), Q6_V_lo_W( vpIn ), 4);

    // PHASE ONE
    // perform a sort between two elements, i.e., (A0, A1), (A2, A3), (A4, A5), ..., (A30, A31)
    // this generates 8 bitonic sequences of length 4
    HVX_VectorPair res1 = SORT_OVER_TWO(Q6_V_lo_W( vpIn ), Q6_V_hi_W( vpIn ), REVERSE);

    // PHASE TWO
    // sort bitonic sequences of length 4 from PHASE ONE
    // this generates 2 bitonic sequences of length 8
    // let vX_Y denote element on vectorX at index Y. at this point, we have 
    // v1_0, v1_1, v1_2, v1_3, v1_4, v1_5, v1_6, v1_7, v1_8, ... v1_31
    // v2_0, v2_1, v2_2, v2_3, v2_4, v2_5, v2_6, v2_7, v2_8, ... v2_31
    // the element-wise comparisons are done between (v1_0, v2_1), (v1_1, v2_0), (v1_2, v2_3), (v1_3, v2_2), ...
    // thus, need to manipulate v2 to be v2_1, v2_0, v2_3, v2_2, v2_5, v2_4, ...
    HVX_VectorPair res2 = Q6_W_vdeal_VVR( Q6_V_hi_W(res1), Q6_V_hi_W(res1), 4*2);
    res2 = Q6_W_vshuff_VVR( Q6_V_lo_W(res2), Q6_V_hi_W(res2), 4*2); 
    // execute 8 sorting networks for size 4
    res2 = SORT_OVER_TWO(Q6_V_lo_W(res1), Q6_V_hi_W(res2), REVERSE);
    // manipulate the vectors to be able to perform the second part of size for sorting networks
    // NOTE: these manipulations are NOT reflected in the sorting network graph, this is a necessary
    //       step in HVX to align the vectors to perform correct word-wise operations
    res2 = Q6_W_vdeal_VVR(Q6_V_hi_W(res2), Q6_V_lo_W(res2), 4*2);
    res2 = SORT_OVER_TWO(Q6_V_lo_W(res2), Q6_V_hi_W(res2), REVERSE);

    // PHASE THREE
    // sort 4 bitonic sequences of length 8 to generate two bitonic sequences of length 16
    // there are 3 layers of SORT_OVER_TWOs
    HVX_VectorPair res3 = Q6_W_vdeal_VVR(Q6_V_hi_W(res2), Q6_V_hi_W(res2), 4*4);
    res3 = Q6_W_vshuff_VVR(Q6_V_lo_W(res3), Q6_V_hi_W(res3), 4*4);
    res3 = Q6_W_vshuff_VVR(Q6_V_hi_W(res3), Q6_V_hi_W(res3), 4*2);
    res3 = Q6_W_vshuff_VVR(Q6_V_lo_W(res3), Q6_V_hi_W(res3), 4*2); 
    res3 = SORT_OVER_TWO(Q6_V_lo_W(res2), Q6_V_hi_W(res3), REVERSE);
    res3 = Q6_W_vdeal_VVR(Q6_V_hi_W(res3), Q6_V_lo_W(res3), 4*2);
    res3 = SORT_OVER_TWO(Q6_V_lo_W(res3), Q6_V_hi_W(res3), REVERSE);
    res3 = Q6_W_vdeal_VVR(Q6_V_hi_W(res3), Q6_V_lo_W(res3), 4*4);
    res3 = SORT_OVER_TWO(Q6_V_lo_W(res3), Q6_V_hi_W(res3), REVERSE);

    // PHASE FOUR
    // execute 2 sorting networks of size 16 (in difference directions) to generate a bitonic sequence of length 32
    // this includes 4 levels of SORT_OVER_TWOs and the required manipulations for vectorization
    HVX_Vector res_lo = Q6_V_lo_W(res3);
    HVX_Vector res_hi = Q6_V_hi_W(res3);
    res_hi =  Q6_V_vdelta_VV(res_hi, q6op_Vb_vsplat_R( 128-4*2 ));
    res_hi =  Q6_V_vdelta_VV(res_hi, q6op_Vb_vsplat_R( 128-4*16 ));
    res3 = SORT_OVER_TWO( res_lo, res_hi, REVERSE);
    res3 = Q6_W_vdeal_VVR(Q6_V_hi_W(res3), Q6_V_lo_W(res3), 4*2); 
    res3 = SORT_OVER_TWO(Q6_V_lo_W(res3), Q6_V_hi_W(res3), REVERSE);
    res3 = Q6_W_vdeal_VVR(Q6_V_hi_W(res3), Q6_V_lo_W(res3), 4*4);
    res3 = SORT_OVER_TWO(Q6_V_lo_W(res3), Q6_V_hi_W(res3), REVERSE);
    res3 = Q6_W_vdeal_VVR(Q6_V_hi_W(res3), Q6_V_lo_W(res3), 4*8);
    res3 = SORT_OVER_TWO(Q6_V_lo_W(res3), Q6_V_hi_W(res3), REVERSE);

    // PHASE FIVE
    // reorder the result of the bitonic_builder into a bitonic sequence of length 32
    // after this, we will have a bitonic sequence
    res3 = Q6_W_vdeal_VVR(Q6_V_hi_W(res3), Q6_V_lo_W(res3), 4*16 );
    res3 = Q6_W_vcombine_VV(Q6_V_vdelta_VV(Q6_V_hi_W(res3), q6op_Vb_vsplat_R( 128-4*2 )), Q6_V_lo_W(res3));
    res3 = Q6_W_vdeal_VVR(Q6_V_hi_W(res3), Q6_V_lo_W(res3), 4*16 + 4*8 );  // ( 1), ( 2)
    res3 = Q6_W_vdeal_VVR(Q6_V_hi_W(res3), Q6_V_lo_W(res3), 4*16 + 4*2 );  // ( 3), ( 4)
    res3 = Q6_W_vdeal_VVR(Q6_V_hi_W(res3), Q6_V_lo_W(res3),  4*8 + 4*2 );  // ( 5), ( 6)
    res3 = Q6_W_vdeal_VVR(Q6_V_hi_W(res3), Q6_V_lo_W(res3), 4*16 + 4*4 );  // ( 7), ( 8)
    res3 = Q6_W_vdeal_VVR(Q6_V_hi_W(res3), Q6_V_lo_W(res3),  4*8 );  // ( 9)
    res3 = Q6_W_vdeal_VVR(Q6_V_hi_W(res3), Q6_V_lo_W(res3), 4*16 );  // (10)  
    return res3;
}

#undef SORT_OVER_TWO

#endif
