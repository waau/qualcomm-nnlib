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


#include "nn_graph.h"
#include "hvx_sort.h"
#include "bitonic_sort.h"

//
// given pointers to two arrays, each of length 'depth',
// find the largest 32 in each, and their indices.
// This is done as follows:
//
//  (1) init 'prev' arrays (two registers) to all 7FFFFFFF
//
//  (2) read next 32 from each of two batches. The values are converted to 32-bits as follows:
//     - invert, and <<23; LSB's are filled with the index (0..depth-1).
//      [by finding the smallest 32 of these, we will find the largest 32 values, and their indicies;
//       with ties going to the smallest index (first encountered). If this is the last group of 32,
//       and it's not 'full' (since depth%32 != 0), force any 'out of range' bytes to 0 before proceeding
//
//  (3)  put the 32 new values into bitonic order, with both batches interleaved in 2 registers.
//       (the 'prev' values are also in bitonic order from previous iteration)
//  (4)  finish sorting 'prev' into ascending order, and 'new' into descending order.
///      Then, find the elementwise 'min' across these two; which are the smallest 32 seen so far
//       and are bitonic order. This is the 'prev' value for next iteration. Go back to step (2) until done.
//  (5) when all are processed, finish sorting 'prev' into ascending order - and this then contains the result.
//

static inline void
find_best_32_of_u8_bitonic_sort(  uint8_t const * inp0, uint8_t const * inp1, int depth,
        uint8_t * bestk0,         // output pointers (bestk1 = NULL to skip)
        uint8_t * bestk1,
        int32_t * best_idx0,
        int32_t * best_idx1,
        int k )                                     // must be 1..32
{
    HVX_Vector CONST_MIN = Q6_V_vsplat_R(0x80FFFFFF);
    HVX_Vector vInA;        // vInA holds data for the first batch
    HVX_Vector vInB;        // vInB holds data for the second batch
    HVX_VectorPair vpIn;    // vpIn holds data from both batches and sends it to the bitonic network
    // vpPrev holds the top 32 elements (64 elemenets in total)
    HVX_VectorPair vpPrev = Q6_W_vcombine_VV( CONST_MIN, CONST_MIN);
    HVX_VectorPair vpCurr;  // vpCurr holds the current 32 elements from batch A and B (64 elements in total)

    HVX_Vector vIdxSeqAdd = Q6_V_vsplat_R(0x00000020); // add constant for adjusting the indices
    HVX_Vector vIdxSeq = *(HVX_Vector const*)const_Count32W;  // {0,1,2,3,4  .. 31} in words

    int nd32 = (depth+31)/32u;
    int idlast = nd32;
    // this is to check whether on the last load will load a full 32 elements.
    // if not, right masking must be done to avoid potentially random behavior
    if( (depth & 31) != 0 ){
        idlast = (depth/32u) & ~3;
    }
    for( int id32 = 0; id32 < nd32; id32 ++ ){
        if( (id32&3)== 0 ){
            // loads 32 elements every 4 iterations
            vInA = q6op_V_vldu_A( (HVX_Vector const*)inp0); inp0 += 128;
            vInB = q6op_V_vldu_A( (HVX_Vector const*)inp1); inp1 += 128;
            if( id32 == idlast){
                // right masking if the final load does not contain 32 elements
                HVX_Vector qmask = Q6_Q_vsetq_R(depth);
                vInA = q6op_V_vand_QV( qmask, vInA );
                vInB = q6op_V_vand_QV( qmask, vInB );
            }
        } // if( (id32&3) == 0 )

        // perform zero-extension on inputs
        // each element is a byte, we want to make it into 4 bytes, where the data is read as follows
        // -------------------------------------------------------------------------
        // | 1 signed bit | 8 bits to represent value | 23 bits to represent index |
        // -------------------------------------------------------------------------
        HVX_VectorPair vpZeros = Q6_W_vcombine_VV( Q6_V_vzero(), Q6_V_vzero() );
        vpIn = Q6_Wh_vunpackoor_WhVb( vpZeros, vInA );
        HVX_Vector vvpInA = Q6_V_lo_W(vpIn);
        vpIn = Q6_Wh_vunpackoor_WhVb( vpZeros, vvpInA );
        vvpInA = Q6_V_lo_W(vpIn);
        vpIn = Q6_Wh_vunpackoor_WhVb( vpZeros, vInB );
        HVX_Vector vvpInB = Q6_V_lo_W(vpIn);
        vpIn = Q6_Wh_vunpackoor_WhVb( vpZeros, vvpInB );
        vvpInB = Q6_V_lo_W(vpIn);
        // left shift the values word-wise
        vvpInA = Q6_Vuw_vlsr_VuwR( vvpInA, 1);
        vvpInB = Q6_Vuw_vlsr_VuwR( vvpInB, 1);

        // populate vpIn with 2 sets of 32 elements from input
        vpIn = Q6_W_vcombine_VV( vvpInB, vvpInA );
        // add index informations to the lower 23-bits of each word
        HVX_VectorPair vpIdxSeq = Q6_W_vcombine_VV( vIdxSeq, vIdxSeq);
        vpIn = Q6_Ww_vadd_WwWw(vpIn, vpIdxSeq);
        vvpInA = Q6_V_lo_W( vpIn );
        vvpInB = Q6_V_hi_W( vpIn );
        // shuffle the elements of vpIn such that vpIn contains the 32 elements in the following order
        // vpIn_lo = A0, A1, A2, ..., A30, A31
        // vpIn_hi = B0, B1, B2, ..., B30, B31
        vpIn = Q6_W_vshuff_VVR( vvpInB,  vvpInA, -4*32);

        // at this point the preprocessing is done, and the bitonic sort procedure starts
        // build the input into a bitonic sequence
        vpCurr = hvx_convert_to_bitonic_seq_32(vpIn, 0);
        // sort the bitonic sequence
        vpCurr = hvx_sort_bitonic_seq_32(vpCurr, 0);
        // vpCurr holds 32 elements in ascending order, while vpPrev holds 32 elements in descending order
        // do a compare to take the largest 32 elements of the combined 64 elements (of vpCurr and vpPrev)
        // this will result in a bitonic sequence with 32 elements
        // perform a sort on the resulting top 32 elements and make this the new vpPrev 
        vpPrev = Q6_W_vcombine_VV( Q6_Vw_vmax_VwVw( Q6_V_hi_W( vpCurr ), Q6_V_hi_W( vpPrev ) ),
                                   Q6_Vw_vmax_VwVw( Q6_V_lo_W( vpCurr ), Q6_V_lo_W( vpPrev )) );
        vpPrev = hvx_sort_bitonic_seq_32(vpPrev, 1);

        // move the indices up by 32, they will be used to index the next 32 elements
        vIdxSeq = Q6_Vw_vadd_VwVw( vIdxSeq, vIdxSeqAdd);
        // shift (using rotate) vIn by 32 elements (bytes) so that the next 32 elements will be processed
        // in the next iteration of the loop
        vInA = Q6_V_vror_VR(vInA, 32);
        vInB = Q6_V_vror_VR(vInB, 32);
    } // for( int id32 = 0; id32 < nd32; id32 ++ )

    // for the top 32 case, the elements in vpPrev are sorted as follows
    // consider the sorted elements A31 > A30 > A29 > A28 > ... > A02 > A01 > A00,
    // and B31 > B30 > B29 > B28 > ... > B02 > B01 > B00. They are now arranged as follows
    // vpPrev_low = A31, B31, A30, B30, A29, B29, ..., A17, B17, A16, B16
    // vpPrev_high = A15, B15, A14, B14, A13, B13, ..., A01, B01, A00, B00
    // need to separate (take out the duplicates) and get an HVX_VectorPair of
    // vpParse_low = A31, A30, A29, A28, ..., A02, A01, A00
    // vpParse_low = B31, B30, B29, B28, ..., B02, B01, B00
    HVX_VectorPair vpParse = Q6_W_vdeal_VVR( Q6_V_hi_W(vpPrev), Q6_V_lo_W(vpPrev), -4 );

    // separate the values from the indices
    // get values, leftshift the values by 1 to get rid of the signed bit and get 
    // values in words. unpack values to return to bytes 
    HVX_Vector vVals0 = Q6_Vh_vpacko_VwVw( Q6_V_hi_W(vpParse), Q6_V_lo_W(vpParse) );
    vVals0 = Q6_Vh_vasl_VhR( vVals0, 1);
    vVals0 = Q6_Vb_vpacko_VhVh( vVals0,  vVals0 );
    HVX_Vector vVals1 = Q6_Vh_vpacko_VwVw( Q6_V_lo_W(vpParse), Q6_V_hi_W(vpParse) );
    vVals1 = Q6_Vh_vasl_VhR( vVals1, 1);
    vVals1 = Q6_Vb_vpacko_VhVh( vVals1,  vVals1 );
    // get indices. This is done simply just masking out the values portion
    HVX_Vector vIdxs0 = Q6_V_vand_VV( Q6_V_lo_W(vpParse), Q6_V_vsplat_R(0x7FFFFF));
    HVX_Vector vIdxs1 = Q6_V_vand_VV( Q6_V_hi_W(vpParse), Q6_V_vsplat_R(0x7FFFFF));
    // write the results
    q6op_vstu_variable_ARV( bestk0, k, vVals0); 
    q6op_vstu_variable_ARV( best_idx0, k*4, vIdxs0);
    if(bestk1 != NULL){
        q6op_vstu_variable_ARV( bestk1, k, vVals1); 
        q6op_vstu_variable_ARV( best_idx1, k*4, vIdxs1);        
    }

}

// inp points to  u8[batches*indepth]
// results go to bestk[batches*k]
// and to best_idx[batches*k]
void
find_best_K_max32_of_u8_bitonic_sort(  uint8_t const * inp, int indepth,     // indepth must be >= k
        uint8_t * best_k, int32_t * best_idx, int k,        // k must be >= 1, <= 32
        int batches )                                       // batches must be >= 1;
{
    for( int ib = 0; ib < batches; ib += 2 ){
        uint8_t const * in0 = inp + ib*indepth;
        uint8_t const * in1 = in0 + indepth;
        uint8_t * best_k0 = best_k + ib*k;
        uint8_t * best_k1 = best_k0 + k;

        if( ib+1 == batches){
             in1 = inp;
             best_k1 = NULL;
        }
        find_best_32_of_u8_bitonic_sort( in0, in1, indepth, 
                                         best_k0, best_k1, 
                                         best_idx + ib*k, best_idx + (ib+1)*k,
                                         k );
    }
}
