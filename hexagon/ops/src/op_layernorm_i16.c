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
 * This contains implementations for layer normalization for qi16 tensors
 * 
 * Layer normalization is like instance normalization, but we don't average over all the images.
 * Find per-batch element mean and variance
 * norm = (in - mean) / sqrt(variance + variance_epsilon)
 * out = scale * norm + bias
 * 
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

//Some special defines because magic numbers are bad
#define HVX_VECTOR_SIZE 128U
#define HVX_PREFETCH_AMOUNT 512U
//Bias has double data size so we prefetch twice as much
#define HVX_PREFETCH_AMOUNT_BIAS 1024U
#define HVX_VECTOR_SIZE_SIGNED 128
#define ELEMENTS_PER_HVX_16B 64U
#define ELEMENTS_PER_HVX_32B 32U

struct reduction_stats{
    int16_t reduction_max;
    int16_t reduction_min;
    int16_t reduction_mean;
    int32_t reduction_sum;
    uint64_t reduction_sum_squares;
    float reduction_variance;
    float reduction_std;
    float reduction_inv_std;
    float reduction_final_scale;
    int32_t reduction_layernorm_scale;
    int32_t reduction_layernorm_shamt;
    uint32_t reduction_all_same;
    uint64_t sum_square;
    uint64_t scaled_sum_of_squares;
    uint64_t variance_quant;
};

struct layernorm_hvx_partial_stats
{
    HVX_Vector sum;
    HVX_VectorPair sqLowerBits;
    HVX_Vector_x2 sqUpperBits;
};

//Structure to hold values that will be shared between threads
struct layernorm_hvx_runstate_shared{
    nn_sem_t worker_sem;
    struct nn_node *self;
    const int16_t *in_data;
    
    int16_t *out_data;
    int16_t *scale_data;
    int32_t *bias_data;
    volatile uint32_t curr_pos_hvx;
    volatile uint32_t curr_pos;
    volatile uint32_t curr_reduction;
    volatile uint32_t vectors_left_in_curr_reduction;
    uint32_t vectors_in_reduction;
    uint32_t num_reductions;
    uint32_t reduction_size;
    uint32_t num_batches;
    uint32_t vectors_in_input;
    uint32_t leftovers_per_reduction;
    uint32_t bytes_per_batch;
    uint32_t num_threads;
    uint32_t bias_scale;
    int32_t bias_shamt;
    struct reduction_stats *statistics;
};

struct layernorm_hvx_worker_runstate{
    int worker_id;
    struct layernorm_hvx_partial_stats *partial_stats;
    struct layernorm_hvx_runstate_shared *shared_info;
};


//Function to reset the worker partial stats to zero
//It shouldn't really be necessary but we do it once
//and it helps garuntee that everything is zero when
//we expect it to be zero
static inline void reset_worker_stats(struct nn_graph *nn, struct layernorm_hvx_worker_runstate *worker_state, uint32_t num_reductions)
{
    for(uint32_t j = 0; j < num_reductions; ++j){
        worker_state->partial_stats[j].sum = Q6_V_vzero();
        worker_state->partial_stats[j].sqLowerBits = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());
        worker_state->partial_stats[j].sqUpperBits.val[0] = Q6_V_vzero();
        worker_state->partial_stats[j].sqUpperBits.val[1] = Q6_V_vzero();
    }
}

static inline void compute_stats_core(struct nn_graph *nn, struct layernorm_hvx_partial_stats *partial_stats,
                                        HVX_Vector input_vector, HVX_Vector one_hvx){
    HVX_VectorPair overflowGuard;
    HVX_VectorPred overflowDetect;
    //Unpack 16-bit input to 32-bit for sum
    HVX_VectorPair unpacked_input = Q6_Ww_vunpack_Vh(input_vector);
    //Accumulate input
    partial_stats->sum = Q6_Vw_vadd_VwVw(partial_stats->sum, Q6_V_lo_W(unpacked_input));
    //Add the rest
    partial_stats->sum = Q6_Vw_vadd_VwVw(partial_stats->sum, Q6_V_hi_W(unpacked_input));
    //Overflow is possible
    //Need to be careful
    //Setup the guard
    overflowGuard = partial_stats->sqLowerBits;
    //Square input and accumulate
    //We allow overflow and detect if it occurred
    partial_stats->sqLowerBits = Q6_Ww_vmpyacc_WwVhVh(partial_stats->sqLowerBits,
                                                       input_vector, input_vector);
 
    //We detect overflow by checking to see if the old value of the
    //sum of squares is larger than the new value
    //Recall that x^2 > 0 forall x
    //So this works
    //There is one wonky part
    //We have to treat everything as unsigned for this to work
    //First compare the lower parts of the vector pairs
    overflowDetect = Q6_Q_vcmp_gt_VuwVuw(Q6_V_lo_W(overflowGuard), Q6_V_lo_W(partial_stats->sqLowerBits));
    //If overflow occured accumulate one into upperbits
    partial_stats->sqUpperBits.val[0] = Q6_Vw_condacc_QVwVw(overflowDetect, 
                                                            partial_stats->sqUpperBits.val[0], one_hvx);
    //Repeat for upper part of vector pair
    overflowDetect = Q6_Q_vcmp_gt_VuwVuw(Q6_V_hi_W(overflowGuard), Q6_V_hi_W(partial_stats->sqLowerBits));
    //If overflow occured accumulate one into upperbits
    partial_stats->sqUpperBits.val[1] = Q6_Vw_condacc_QVwVw(overflowDetect, partial_stats->sqUpperBits.val[1], one_hvx);
}

/*
* Function that will be used by the worker threads to do the
* bulk of the computation of the sum and the sum of squares
* The basic idea is that all of the workers will blast along the data
* skipping over any incomplete vectors that might be at the end of a reduction
* This is done to limit the amount of conditional/scalar code so that
* we get the most performance from the HVX workers
* Note that this does mean that there is no garuntee which worker will compute what
* We tackle this by letting each worker keep a partial sum and sum of squares for each reduction
* Later on we will combine the partial statistics and compute the final scalar values
* The final thing that we need to discuss is the sum of squares
* Recall that the maximum absolute value of a int16 is 2^15
* So if we were to square a int16 with value -2^15 (int16 min)
* We end up with 2^30
* If we add another int16 min squared (IE 2^30)
* Our accumulator will be equal to 2^31
* Repeat this process 2 more times and we overflow our 32-bit accumulator
* Unfortunately HVX does not support double word instructions
* So we are stuck with 32-bit accumulators
* So what we do is after each accumulation we check to see if the accumulator overflowed
* if it did we add 1 to a seperate HVX vector which we will interperate as the upper
* 32-bits of a 64-bit accumulator
*/
static inline void compute_stats_hvx(struct nn_graph *nn, void *rstate){
    struct layernorm_hvx_worker_runstate *runstate = (struct layernorm_hvx_worker_runstate *)rstate;
    const int16_t *input_p;
    uint32_t pos_hvx = __sync_fetch_and_add(&(runstate->shared_info->curr_pos_hvx),1U);
    uint32_t curr_reduction = 0U;
    uint32_t curr_vector = 0U;
    uint32_t one_scalar = 1U;
    //Vector of 32-bit words with each 32-bit word = 1
    //Useful for the inevitable 64-bit overflow stuff below
    HVX_Vector one_hvx = Q6_V_vsplat_R(one_scalar);
    reset_worker_stats(nn, runstate, runstate->shared_info->num_reductions);
    while(pos_hvx < runstate->shared_info->vectors_in_input){
        //How to figure out which normalization group I am working with?
        //shared_info->vectors_in_input = num_reductions * vectors_per_reduction
        //Therefore reduction number = floor(pos_hvx / vectors_per_reduction)
        curr_reduction = pos_hvx / runstate->shared_info->vectors_in_reduction;
        //Likewise our current vector in the reduction is given by
        //curr_vector = pos_hvx % shared_info->vectors_in_reduction
        curr_vector = pos_hvx % runstate->shared_info->vectors_in_reduction;
        //There maybe less than one full vector at the end of a reduction
        //Handle that somewhere else to reduce the amount of conditional code here
        //But we do need to account for the "leftovers" when we determine where to grab the next vector from
        //If we were working with a int16 pointer then our pointer math is as follows:
        //pointer_location = int16_per_vector * curr_reduction * vectors_in_reduction 
        //                 + leftovers_per_reduction * curr_reduction
        //                 + curr_vector * int16_per_vector
        input_p = runstate->shared_info->in_data 
                + ELEMENTS_PER_HVX_16B * curr_reduction * runstate->shared_info->vectors_in_reduction
                + runstate->shared_info->leftovers_per_reduction * curr_reduction
                + curr_vector * ELEMENTS_PER_HVX_16B;
        //Be a nice person and prefetch the next vector for everyone
        l2pref(input_p + ELEMENTS_PER_HVX_16B, 1U, HVX_PREFETCH_AMOUNT, 1U);

        HVX_Vector input_vector = q6op_V_vldu_A((HVX_Vector *)input_p);        
        compute_stats_core(nn, &(runstate->partial_stats[curr_reduction]), input_vector, one_hvx);
        pos_hvx = __sync_fetch_and_add(&(runstate->shared_info->curr_pos_hvx),1);
    }
    //This thread has finished computing its partial statistics
    //Update the worker semaphore
    nn_sem_post(&(runstate->shared_info->worker_sem));
    return;
}

/*
* Function to compute the "leftovers" (ie anything less than a full hvx vector) 
* that may be present at the end of a reduction. For example a reduction of size
* 65 has one full vector and then a single integer
* This function pretty much repeats the calculations done in the main compute_stats function
* So all the previous logic about overflow still applies
*/
static inline void compute_stats_hvx_leftovers(struct nn_graph *nn, void *rstate){
    struct layernorm_hvx_worker_runstate *runstate = (struct layernorm_hvx_worker_runstate *)rstate;
    const int16_t *input_p;
    uint32_t curr_reduction = __sync_fetch_and_add(&(runstate->shared_info->curr_reduction),1);
    uint32_t one_scalar = 1U;
    //Vector of 32-bit words with each 32-bit word = 1
    //Useful for the inevitable 64-bit overflow stuff below
    HVX_Vector one_hvx = Q6_V_vsplat_R(one_scalar);
    HVX_VectorPred maskPred;
    maskPred = Q6_Q_vsetq_R(runstate->shared_info->leftovers_per_reduction*2);
    while(curr_reduction < runstate->shared_info->num_reductions){
        //The address of the leftovers we need to grab is given by:
        //pointer_location = int16_per_vector * curr_reduction * vectors_in_reduction
        input_p = runstate->shared_info->in_data
                + ELEMENTS_PER_HVX_16B * curr_reduction * runstate->shared_info->vectors_in_reduction;
        HVX_Vector input_vector = q6op_V_vldu_A((HVX_Vector *)input_p);     
        //Mask out values from the next reduction over
        input_vector = Q6_V_vmux_QVV(maskPred, input_vector, Q6_V_vzero());
        compute_stats_core(nn, &(runstate->partial_stats[curr_reduction]), input_vector, one_hvx);
        curr_reduction = __sync_fetch_and_add(&(runstate->shared_info->curr_reduction),1);
    }
    //This thread has finished computing its partial statistics
    //Update the worker semaphore
    nn_sem_post(&(runstate->shared_info->worker_sem));
    return;
}

//Function to combine the partial worker stats
//Will save the result in the first workers buffer
static inline void combine_worker_stats(struct nn_graph *nn, void* runstate_void){
    struct layernorm_hvx_worker_runstate *runstate = (struct layernorm_hvx_worker_runstate *)runstate_void;
    if (runstate->shared_info->num_threads <= 1){
        //Realistically num_threads should never be 0
        //Unless there were less than 64 elements in the input
        //But for now we will lump it in
        //If there was 1 or 0 threads then we can instantly return
        logmsg(nn,2, "Only one thread! Return immediately!");
        //Post to semaphore
        nn_sem_post(&(runstate->shared_info->worker_sem));
        return;
    }
    //Ok so there were some number of threads
    uint32_t one_scalar = 1U;
    HVX_Vector one_hvx = Q6_V_vsplat_R(one_scalar);
    HVX_VectorPred overflowDetect;
    HVX_VectorPair overflowGuard;
    //For each reduction get the partial stats from each worker
    //Combine the partial stats into the first workers buffers and 
    //move on
    struct layernorm_hvx_partial_stats *partial_stats = runstate->partial_stats;
    struct layernorm_hvx_partial_stats *current_stats;
    struct layernorm_hvx_partial_stats *combined_stats;
    uint32_t stats_offset = 0U;
    for (uint32_t curr_reduction = 0; curr_reduction < runstate->shared_info->num_reductions; ++curr_reduction){
        //We always combine into the first workers stats
        //In memory the partial stats are arranged like:
        //worker1_reduction1, worker1_reduction2 ... worker2_reduction1,... workerN_reductionN
        combined_stats = partial_stats + curr_reduction;
        for(uint32_t curr_worker = 1; curr_worker < runstate->shared_info->num_threads; ++curr_worker){
            //Start at one since combining the first workers stats
            //into its own is completely silly
            //To compute the offset into our big blob of stats do the following:
            //stats_offset = curr_worker * runstate->num_reductions
            //             + curr_reduction 
            stats_offset = curr_worker * runstate->shared_info->num_reductions
                         + curr_reduction;
            current_stats = partial_stats + stats_offset;
            combined_stats->sum = Q6_Vw_vadd_VwVw(combined_stats->sum, current_stats->sum);
            
            overflowGuard = combined_stats->sqLowerBits;
            combined_stats->sqLowerBits = Q6_Ww_vadd_WwWw(combined_stats->sqLowerBits, current_stats->sqLowerBits);
            overflowDetect = Q6_Q_vcmp_gt_VuwVuw(Q6_V_lo_W(overflowGuard), Q6_V_lo_W(combined_stats->sqLowerBits));
            combined_stats->sqUpperBits.val[0] = Q6_Vw_condacc_QVwVw(overflowDetect, combined_stats->sqUpperBits.val[0], one_hvx);
            
            overflowDetect = Q6_Q_vcmp_gt_VuwVuw(Q6_V_hi_W(overflowGuard), Q6_V_hi_W(combined_stats->sqLowerBits));
            combined_stats->sqUpperBits.val[1] = Q6_Vw_condacc_QVwVw(overflowDetect, combined_stats->sqUpperBits.val[1], one_hvx);

            //Add the upper bits from the current worker
            combined_stats->sqUpperBits.val[0] = Q6_Vw_vadd_VwVw(combined_stats->sqUpperBits.val[0], current_stats->sqUpperBits.val[0]);
            combined_stats->sqUpperBits.val[1] = Q6_Vw_vadd_VwVw(combined_stats->sqUpperBits.val[1], current_stats->sqUpperBits.val[1]);

        }
    }
    //Post to semaphore
    nn_sem_post(&(runstate->shared_info->worker_sem));
    return;
}

static inline uint64_t reduce_worker_sum_64b(struct nn_graph *nn, struct layernorm_hvx_runstate_shared *runstate, struct layernorm_hvx_partial_stats *partial_stats){
    //Heres a weird trick
    //We are going to pass the pointer to the exact reduction number we want to work with
    //Unfortunately for the sum of squares we need to worry about overflow
    //Stupid 16-bit
    HVX_VectorPred overflowDetect;
    HVX_Vector overflowGuard;
    uint32_t one_scalar = 1U;
    HVX_Vector one_hvx = Q6_V_vsplat_R(one_scalar);
    HVX_Vector sumCombined, oldLowerBits, overflowBuffer, oldOverflowBuffer;
    
    overflowBuffer = Q6_V_vzero();
    overflowGuard = Q6_Vw_vmax_VwVw(Q6_V_lo_W(partial_stats->sqLowerBits),Q6_V_hi_W(partial_stats->sqLowerBits));
    sumCombined = Q6_Vw_vadd_VwVw(Q6_V_lo_W(partial_stats->sqLowerBits),Q6_V_hi_W(partial_stats->sqLowerBits));
    
    //Detect overflow
    overflowDetect = Q6_Q_vcmp_gt_VuwVuw(overflowGuard, sumCombined);
    //We accumulate into a seperate overflow buffer
    overflowBuffer = Q6_Vw_condacc_QVwVw(overflowDetect, overflowBuffer, one_hvx);
    //The code below is more or less the same as
    //the existing compute_reduced_sum function
    //from nn_reduce_utils.h
    //Except that it handles overflow detection
    //So the rotation values 1,2,4,8,16 are based on
    //that implementation
    oldLowerBits = sumCombined;
    //Here we rotate the lower bits and begin the reduction sum
    sumCombined = Q6_V_vror_VR(sumCombined, 1 * sizeof(uint32_t));
    sumCombined = Q6_Vw_vadd_VwVw(sumCombined, oldLowerBits);
    //Detect overflow
    overflowDetect = Q6_Q_vcmp_gt_VuwVuw(oldLowerBits, sumCombined);
    oldOverflowBuffer = overflowBuffer;
    overflowBuffer = Q6_V_vror_VR(overflowBuffer, 1*sizeof(uint32_t));
    overflowBuffer = Q6_Vw_vadd_VwVw(overflowBuffer, oldOverflowBuffer);
    overflowBuffer = Q6_Vw_condacc_QVwVw(overflowDetect, overflowBuffer, one_hvx);

    //Rinse and repeat a couple more times
    oldLowerBits = sumCombined;
    sumCombined = Q6_V_vror_VR(sumCombined, 2 * sizeof(uint32_t));
    sumCombined = Q6_Vw_vadd_VwVw(sumCombined, oldLowerBits);
    //Detect overflow
    overflowDetect = Q6_Q_vcmp_gt_VuwVuw(oldLowerBits, sumCombined);
    oldOverflowBuffer = overflowBuffer;
    overflowBuffer = Q6_V_vror_VR(overflowBuffer, 2*sizeof(uint32_t));
    overflowBuffer = Q6_Vw_vadd_VwVw(overflowBuffer, oldOverflowBuffer);
    overflowBuffer = Q6_Vw_condacc_QVwVw(overflowDetect, overflowBuffer, one_hvx);

    oldLowerBits = sumCombined;
    sumCombined = Q6_V_vror_VR(sumCombined, 4 * sizeof(uint32_t));
    sumCombined = Q6_Vw_vadd_VwVw(sumCombined, oldLowerBits);
    //Detect overflow
    overflowDetect = Q6_Q_vcmp_gt_VuwVuw(oldLowerBits, sumCombined);
    oldOverflowBuffer = overflowBuffer;
    overflowBuffer = Q6_V_vror_VR(overflowBuffer, 4*sizeof(uint32_t));
    overflowBuffer = Q6_Vw_vadd_VwVw(overflowBuffer, oldOverflowBuffer);
    overflowBuffer = Q6_Vw_condacc_QVwVw(overflowDetect, overflowBuffer, one_hvx);

    oldLowerBits = sumCombined;
    sumCombined = Q6_V_vror_VR(sumCombined, 8 * sizeof(uint32_t));
    sumCombined = Q6_Vw_vadd_VwVw(sumCombined, oldLowerBits);
    //Detect overflow
    overflowDetect = Q6_Q_vcmp_gt_VuwVuw(oldLowerBits, sumCombined);
    oldOverflowBuffer = overflowBuffer;
    overflowBuffer = Q6_V_vror_VR(overflowBuffer, 8*sizeof(uint32_t));
    overflowBuffer = Q6_Vw_vadd_VwVw(overflowBuffer, oldOverflowBuffer);
    overflowBuffer = Q6_Vw_condacc_QVwVw(overflowDetect, overflowBuffer, one_hvx);

    oldLowerBits = sumCombined;
    sumCombined = Q6_V_vror_VR(sumCombined, 16 * sizeof(uint32_t));
    sumCombined = Q6_Vw_vadd_VwVw(sumCombined, oldLowerBits);
    //Detect overflow
    overflowDetect = Q6_Q_vcmp_gt_VuwVuw(oldLowerBits, sumCombined);
    oldOverflowBuffer = overflowBuffer;
    overflowBuffer = Q6_V_vror_VR(overflowBuffer, 16*sizeof(uint32_t));
    overflowBuffer = Q6_Vw_vadd_VwVw(overflowBuffer, oldOverflowBuffer);
    overflowBuffer = Q6_Vw_condacc_QVwVw(overflowDetect, overflowBuffer, one_hvx);

    //Now do the reduction for the upper bits
    //We don't worry about overflow here
    //Since an overflow in the upper bits would mean that we have actually managed to overflow a 64-bit integer
    //Which is just not supported (and highly illegal)
    HVX_Vector combinedUpper = Q6_Vw_vadd_VwVw(partial_stats->sqUpperBits.val[0],partial_stats->sqUpperBits.val[1]);
    
    HVX_Vector oldUpperBits;
    oldUpperBits = combinedUpper;
    combinedUpper = Q6_V_vror_VR(combinedUpper, 1*sizeof(uint32_t));
    combinedUpper = Q6_Vw_vadd_VwVw(combinedUpper, oldUpperBits);

    oldUpperBits = combinedUpper;
    combinedUpper = Q6_V_vror_VR(combinedUpper, 2*sizeof(uint32_t));
    combinedUpper = Q6_Vw_vadd_VwVw(combinedUpper, oldUpperBits);

    oldUpperBits = combinedUpper;
    combinedUpper = Q6_V_vror_VR(combinedUpper, 4*sizeof(uint32_t));
    combinedUpper = Q6_Vw_vadd_VwVw(combinedUpper, oldUpperBits);

    oldUpperBits = combinedUpper;
    combinedUpper = Q6_V_vror_VR(combinedUpper, 8*sizeof(uint32_t));
    combinedUpper = Q6_Vw_vadd_VwVw(combinedUpper, oldUpperBits);

    oldUpperBits = combinedUpper;
    combinedUpper = Q6_V_vror_VR(combinedUpper, 16*sizeof(uint32_t));
    combinedUpper = Q6_Vw_vadd_VwVw(combinedUpper, oldUpperBits);
    
    uint64_t upperBitsScalar =(uint64_t) ((*(uint32_t *)&combinedUpper)+(*(uint32_t *)&overflowBuffer));
    
    upperBitsScalar = upperBitsScalar << 32;
    uint64_t lowerBitsScalar = (uint64_t)((*(uint32_t *)&sumCombined));
    
    return upperBitsScalar + lowerBitsScalar;
}


static void reduce_stats_to_scalars(struct nn_graph *nn, void *runstate_void){
    struct layernorm_hvx_worker_runstate *runstate = (struct layernorm_hvx_worker_runstate *)runstate_void;

    for(uint32_t curr_reduction = 0; curr_reduction < runstate->shared_info->num_reductions; ++curr_reduction){
        HVX_Vector sumReduced = compute_reduced_sum(runstate->partial_stats[curr_reduction].sum);
        int32_t sumReducedScalar = *((int32_t *)&sumReduced);
        uint64_t sumOfSquaresScalar = reduce_worker_sum_64b(nn, runstate->shared_info, &(runstate->partial_stats[curr_reduction]));
        runstate->shared_info->statistics[curr_reduction].reduction_sum = sumReducedScalar;
        runstate->shared_info->statistics[curr_reduction].reduction_sum_squares = sumOfSquaresScalar;
    }
    //Post to semaphore
    nn_sem_post(&(runstate->shared_info->worker_sem));
    return;
}
/*
*The main supervisor function to compute 
*the basic statistics of the inputs
*Its basic job is to marshal data for 
*HVX workers and handle edge cases
*The threading logic is fairly simple
*Initially we spawn as many workers as we can
*Then we let them blast along the input and compute a series
*of partial statistics
*Once they finish with that we dispatch the workers to handle
*any leftovers at the end of a reduction
*So now we have a bunch of partial statistics in the following format:
*worker1_reduction1, worker1_reduction2 ... worker2_reduction1,... workerN_reductionN
*We then accumulate the partial statistics for all workers > 1 into the partial statistics
*of worker1
*Finally we perform a reduction sum for worker1's partial statisics
*And then we have the sum and sum of squares for a given reduction
*/
static inline int statistics_hvx_supervisor(struct nn_node *self, struct nn_graph *nn, struct layernorm_hvx_runstate_shared *runstate){
    //Now we decide the number of threads based on number of vectors in input
    uint32_t num_threads;
    num_threads = (runstate->vectors_in_input > nn->num_vector_threads)?nn->num_vector_threads:runstate->vectors_in_input;
    num_threads = (num_threads < 1U)?1U:num_threads;
    if(0 != (nn_scratch_grow(nn, num_threads*sizeof(struct layernorm_hvx_worker_runstate)))){
        errlog(nn, "OP_LayerNorm_i16 id: %x could not allocate scratch space for HVX worker metadata", self->node_id);
        return NN_EXECUTE_OUT_OF_SCRATCH_ERROR;
    }
    if(0 != (nn_scratch_grow(nn, num_threads*runstate->num_reductions*sizeof(struct layernorm_hvx_partial_stats)))){
        errlog(nn, "OP_LayerNorm_i16 id: %x could not allocate scratch space for HVX worker partial statistics", self->node_id);
        return NN_EXECUTE_OUT_OF_SCRATCH_ERROR;
    }
    //We have enough scratch for what we need
    //Allocate it
    struct layernorm_hvx_worker_runstate *worker_runstates = nn_scratch_alloc(nn, num_threads*sizeof(struct layernorm_hvx_worker_runstate));
    struct layernorm_hvx_partial_stats *worker_partial_stats = nn_scratch_alloc(nn, num_threads*runstate->num_reductions*sizeof(struct layernorm_hvx_partial_stats));
    
    //Assign the memory to the workers
    for(uint32_t i = 0; i < num_threads; ++i){
        worker_runstates[i].shared_info = runstate;
        worker_runstates[i].partial_stats = worker_partial_stats + i*(runstate->num_reductions);        
    }
    
    runstate->num_threads = num_threads;
    //Compute the statistics without worrying about leftovers
    for(uint32_t j = 0; j < num_threads; ++j){
        nn_os_work_for_vector(nn, compute_stats_hvx, &(worker_runstates[j]));
    }
    nn_sem_wait_n_times(&(runstate->worker_sem), num_threads);
    //Get the workers to handle the leftovers
    if (runstate->leftovers_per_reduction != 0){
        for(uint32_t j = 0; j < num_threads; ++j){
            nn_os_work_for_vector(nn, compute_stats_hvx_leftovers, &(worker_runstates[j]));
        }
        nn_sem_wait_n_times(&(runstate->worker_sem), num_threads);
    }
    //Combine the partial stats into worker1's stats
    struct layernorm_hvx_worker_runstate combine_runstate;
    combine_runstate.shared_info = runstate;
    combine_runstate.partial_stats = worker_partial_stats;
    nn_os_work_for_vector(nn, combine_worker_stats, &combine_runstate);
    nn_sem_wait_n_times(&(runstate->worker_sem), 1U);
    //Reduce to scalars
    nn_os_work_for_vector(nn, reduce_stats_to_scalars, &combine_runstate);
    nn_sem_wait_n_times(&(runstate->worker_sem),1U);
    return 0;  
}


//This function does the actual heavy lifting for the normalization
//It mostly exists to reduce duplicate code
static inline HVX_Vector layer_norm_core(struct nn_graph *nn, struct reduction_stats statistics,
                                        const int16_t *input_p, const int16_t *scale_p, const int32_t *bias_p,
                                        HVX_Vector bias_scale_hvx){
    uint32_t reduction_scale;
    int16_t reduction_mean;
    int32_t zero_var_mask;
    //Load the scalar data
    reduction_scale = statistics.reduction_layernorm_scale;
    reduction_mean = statistics.reduction_mean;
    zero_var_mask = statistics.reduction_all_same * HVX_VECTOR_SIZE_SIGNED;
    HVX_VectorPred zero_var_mask_hvx = Q6_Q_vsetq_R(zero_var_mask);
    //Splat to vectors
    HVX_Vector reduction_scale_hvx = Q6_V_vsplat_R(reduction_scale);
    HVX_Vector reduction_mean_hvx = q6op_Vh_vsplat_R(reduction_mean);
    //Grab input and scale
    HVX_Vector input_vector = q6op_V_vldu_A((HVX_Vector *)input_p);
    HVX_Vector scale_vector = q6op_V_vldu_A((HVX_Vector *)scale_p);
    //Grab bias with two loads
    HVX_Vector bias_hvx_1 = q6op_V_vldu_A((HVX_Vector *)bias_p);
    HVX_Vector bias_hvx_2 = q6op_V_vldu_A((HVX_Vector *)(bias_p+ELEMENTS_PER_HVX_32B));
    
    //Multiply input by scale
    HVX_VectorPair input_prime = Q6_Ww_vmpy_VhVh(input_vector, scale_vector);
    //Multiply mean by scale
    HVX_VectorPair mean_prime = Q6_Ww_vmpy_VhVh(reduction_mean_hvx, scale_vector);
    //Subtract mean from input
    HVX_VectorPair input_sub_mean = Q6_Ww_vsub_WwWw_sat(input_prime, mean_prime);
    //Need to reshuffle and interleave
    input_sub_mean = Q6_W_vshuff_VVR(Q6_V_hi_W(input_sub_mean), Q6_V_lo_W(input_sub_mean), -4);
    //Multiply by final scale
    HVX_Vector norm_input_1 = q6op_Vw_vmpy_VwVw_s1_sat(Q6_V_lo_W(input_sub_mean), reduction_scale_hvx);
    HVX_Vector norm_input_2 = q6op_Vw_vmpy_VwVw_s1_sat(Q6_V_hi_W(input_sub_mean), reduction_scale_hvx);

    //Multiply bias by bias scale
    bias_hvx_1 = q6op_Vw_vmpy_VwVw_s1_sat(bias_hvx_1, bias_scale_hvx);
    bias_hvx_2 = q6op_Vw_vmpy_VwVw_s1_sat(bias_hvx_2, bias_scale_hvx);
    //HANDLE SHORT CIRCUIT BIAS CASE
    //If the variance was zero then the result is given by just the bias
    norm_input_1 = Q6_V_vmux_QVV(zero_var_mask_hvx, Q6_V_vzero(), norm_input_1);
    norm_input_2 = Q6_V_vmux_QVV(zero_var_mask_hvx, Q6_V_vzero(), norm_input_2);
    //Add bias to mux'd input
    norm_input_1 = Q6_Vw_vadd_VwVw_sat(norm_input_1, bias_hvx_1);
    norm_input_2 = Q6_Vw_vadd_VwVw_sat(norm_input_2, bias_hvx_2);     
    //Saturate down to 16-bit and write the results
    HVX_Vector output = Q6_Vh_vpack_VwVw_sat(norm_input_2, norm_input_1);
    return output;
}

static void normalize_hvx_worker(struct nn_graph *nn, void *rstate){
    struct layernorm_hvx_runstate_shared *runstate = (struct layernorm_hvx_runstate_shared *)rstate;
    //Well each worker has to grab the input
    //And the coefficients and biases
    //Make some pointers
    const int16_t *input_p = runstate->in_data;
    const int16_t *scale_p = runstate->scale_data;
    const int32_t *bias_p = runstate->bias_data;
    int16_t *output_p = runstate->out_data;
    uint32_t curr_reduction = 0U;
    uint32_t curr_vector = 0U;
    uint32_t pos_hvx = __sync_fetch_and_add(&(runstate->curr_pos_hvx), 1U);
    uint32_t bias_scale;
    //int32_t reduction_shamt, bias_shamt;
    bias_scale = runstate->bias_scale;

    HVX_Vector bias_scale_hvx = Q6_V_vsplat_R(bias_scale);
    //HVX_Vector bias_shamt_hvx = Q6_V_vsplat_R(bias_shamt);
    while(pos_hvx < runstate->vectors_in_input){
        //How to figure out which normalization group I am working with?
        //shared_info->vectors_in_input = num_reductions * vectors_per_reduction
        //Therefore reduction number = floor(pos_hvx / vectors_per_reduction)
        curr_reduction = pos_hvx / runstate->vectors_in_reduction;
        //Likewise our current vector in the reduction is given by
        //curr_vector = pos_hvx % shared_info->vectors_in_reduction
        curr_vector = pos_hvx % runstate->vectors_in_reduction;
        //There maybe less than one full vector at the end of a reduction
        //Handle that somewhere else to reduce the amount of conditional code here
        //But we do need to account for the "leftovers" when we determine where to grab the next vector from
        //If we were working with a int16 pointer then our pointer math is as follows:
        //pointer_location = int16_per_vector * curr_reduction * vectors_in_reduction 
        //                 + leftovers_per_reduction * curr_reduction
        //                 + curr_vector * int16_per_vector
        input_p = runstate->in_data 
                + ELEMENTS_PER_HVX_16B * curr_reduction * runstate->vectors_in_reduction
                + runstate->leftovers_per_reduction * curr_reduction
                + curr_vector * ELEMENTS_PER_HVX_16B;
        output_p = runstate->out_data 
                + ELEMENTS_PER_HVX_16B * curr_reduction * runstate->vectors_in_reduction
                + runstate->leftovers_per_reduction * curr_reduction
                + curr_vector * ELEMENTS_PER_HVX_16B;
        //We expect that there will be one scale and bias for each element in the reduction
        //Thus we loop back to the start of the scales/biases when we start a new reduction
        scale_p = runstate->scale_data 
                + curr_vector * ELEMENTS_PER_HVX_16B;
        //Our biases are expected to be 32-bit
        //So we will advance the pointer by 64 elements
        //Ultimately though we are going to have to do 2 vector loads
        bias_p = runstate->bias_data
                + curr_vector * ELEMENTS_PER_HVX_16B;
        l2pref(input_p + ELEMENTS_PER_HVX_16B, 1U, HVX_PREFETCH_AMOUNT, 1U);
        l2pref(scale_p + ELEMENTS_PER_HVX_16B, 1U, HVX_PREFETCH_AMOUNT, 1U);
        l2pref(bias_p + ELEMENTS_PER_HVX_16B, 1U, HVX_PREFETCH_AMOUNT_BIAS, 1U);

        //Call the core function
        HVX_Vector output = layer_norm_core(nn, runstate->statistics[curr_reduction], 
                                input_p, scale_p, bias_p, bias_scale_hvx);
        //Unaligned vector store
        q6op_vstu_AV((HVX_Vector *)output_p, output);
        //Increment and repeat
        pos_hvx = __sync_fetch_and_add(&(runstate->curr_pos_hvx),1U);
    }
    nn_sem_post(&(runstate->worker_sem));
    return;
}

//HVX worker function to normalize input leftovers
static void normalize_hvx_leftovers_worker(struct nn_graph *nn, void *rstate){
    struct layernorm_hvx_runstate_shared *runstate = (struct layernorm_hvx_runstate_shared *)rstate;
    //Well each worker has to grab the input
    //And the coefficients and biases
    //Make some pointers
    const int16_t *input_p = runstate->in_data;
    const int16_t *scale_p = runstate->scale_data;
    const int32_t *bias_p = runstate->bias_data;
    int16_t *output_p = runstate->out_data;
    uint32_t curr_reduction = 0U;
    curr_reduction = __sync_fetch_and_add(&(runstate->curr_reduction), 1U);
    uint32_t bias_scale;
    bias_scale = runstate->bias_scale;

    uint32_t pointer_offset;
    HVX_Vector bias_scale_hvx = Q6_V_vsplat_R(bias_scale);
    HVX_VectorPred leftoverMask;
    //Need to multiply # of leftovers by 2 since we are dealing with int16
    leftoverMask = Q6_Q_vsetq_R(runstate->leftovers_per_reduction*2);
    while(curr_reduction < runstate->num_reductions){
        //Set up pointers
        pointer_offset = ELEMENTS_PER_HVX_16B * (curr_reduction+1U) * runstate->vectors_in_reduction;
        input_p = runstate->in_data + pointer_offset;
        output_p = runstate->out_data + pointer_offset;

        scale_p = runstate->scale_data + pointer_offset;

        bias_p = runstate->bias_data + pointer_offset;

        HVX_Vector output = layer_norm_core(nn, runstate->statistics[curr_reduction], 
                                input_p, scale_p, bias_p, bias_scale_hvx);
        //Conditional store
        q6op_vstcc_QAV(leftoverMask, (HVX_Vector *)output_p, output);
        curr_reduction = __sync_fetch_and_add(&(runstate->curr_reduction), 1U);
    }
    nn_sem_post(&(runstate->worker_sem));
    return;
}

//Main function to normalize the inputs
//Sets up the workers and sends them running
//Similar idea to the statistics supervisor
static inline int normalize_hvx_supervisor(struct nn_node *self, struct nn_graph *nn, struct layernorm_hvx_runstate_shared *runstate){
    //Again start with just one thread for sanity of debugging
    uint32_t num_threads;
    num_threads = (runstate->vectors_in_input > nn->num_vector_threads)?nn->num_vector_threads:runstate->vectors_in_input;
    num_threads = (num_threads < 1U)?1U:num_threads;
    //Reset position variables
    runstate->curr_pos_hvx = 0U;
    runstate->curr_reduction = 0U;
    //Now what?
    //For now I won't handle leftovers so just spin up the workers I guess?
    for(uint32_t i=0; i < num_threads; ++i){
        nn_os_work_for_vector(nn, normalize_hvx_worker, runstate);
    }
    nn_sem_wait_n_times(&(runstate->worker_sem), num_threads);
    if(runstate->leftovers_per_reduction != 0){
        for(uint32_t i=0; i < num_threads; ++i){
            nn_os_work_for_vector(nn, normalize_hvx_leftovers_worker, runstate);
        }
        nn_sem_wait_n_times(&(runstate->worker_sem), num_threads);
    }
    return 0;
}

static int execute_layernorm_16b(struct nn_node *self, struct nn_graph *nn){
    //Ok what is our plan here
    //In general we need to do the following:
    //1) Compute the per batch statistics (mean, variance, etc)
    //2) Normalize the elements
    //3) Multiply the elements by the coefficients
    //4) Add the bias
    const struct tensor *in_tensor = self->inputs[0];
    const struct tensor *in_min_tensor = self->inputs[1];
    const struct tensor *in_max_tensor = self->inputs[2];
    const struct tensor *scale_tensor = self->inputs[3];
    const struct tensor *scale_min_tensor = self->inputs[4];
    const struct tensor *scale_max_tensor = self->inputs[5];
    const struct tensor *bias_tensor = self->inputs[6];
    const struct tensor *bias_min_tensor = self->inputs[7];
    const struct tensor *bias_max_tensor = self->inputs[8];
    const struct tensor *given_out_min_tensor;
    const struct tensor *given_out_max_tensor;
    given_out_min_tensor = self->inputs[9];
    given_out_max_tensor = self->inputs[10];
    const struct tensor *axis_tensor = self->inputs[11];
    int32_t reduction_axis = tensor_get_int32(axis_tensor,0);
    if(reduction_axis > 2 || reduction_axis < 0){
        errlog(nn, "OP_LayerNorm_i16 id: %x got unexpected reduction axis: %d. Valid reduction axis range is [0, 2)", self->node_id, reduction_axis);
        return -1;
    }
    struct tensor *out_tensor = self->outputs[0];
    struct tensor *out_min_tensor = self->outputs[1];
    struct tensor *out_max_tensor = self->outputs[2];
    float in_min, in_max, scale_min, scale_max, bias_min, bias_max;
    uint32_t batches = in_tensor->shape.batches;
    uint32_t width = in_tensor->shape.width;
    uint32_t height = in_tensor->shape.height;
    uint32_t depth = in_tensor->shape.depth;
    in_min = tensor_get_float(in_min_tensor, 0);
    in_max = tensor_get_float(in_max_tensor, 0);
    scale_min = tensor_get_float(scale_min_tensor, 0);
    scale_max = tensor_get_float(scale_max_tensor, 0);
    bias_min = tensor_get_float(bias_min_tensor, 0);
    bias_max = tensor_get_float(bias_max_tensor, 0);
    float out_min = tensor_get_float(given_out_min_tensor, 0);
    float out_max = tensor_get_float(given_out_max_tensor, 0);
    float in_scale = get_qi16_level_size(in_min, in_max);
    float scale_scale = get_qi16_level_size(scale_min, scale_max);
    float bias_scale;
    get_qi32_level_size_zero(bias_min, bias_max, &bias_scale);
    adjust_minmax_for_zero_16b(&out_min, &out_max);
    float out_scale = get_qi16_level_size(out_min, out_max);
    //Compute the number of reduction elements r
    //Compute the size of each reduction reduction_size
    //At minimum r = batches
    uint32_t reductions = 1U;
    for(int32_t i = 0; i <= reduction_axis; ++i){
        reductions = reductions * in_tensor->shape.dimension[i];
    }
    logmsg(nn,2, "OP_LayerNorm_i16 id: %x with reduction axis: %d has %u reduction elements", self->node_id, reduction_axis, reductions);
    int32_t reduction_size = 1U;
    for(uint32_t i = 3; i > reduction_axis; --i){
        reduction_size = reduction_size * in_tensor->shape.dimension[i];
    }
    logmsg(nn,2, "OP_LayerNorm_i16 id: %x with reduction axis: %d has %u elements per reduction", self->node_id, reduction_axis, reduction_size);

    if(0!=(tensor_out_prepare_normal(out_tensor, batches, height, width, depth, NN_TYPE_QINT16))){
        return errlog(nn, "OP_LayerNorm_i16 id: %x output tensor too small for input", self->node_id);
    }
    tensor_set_single_float(out_min_tensor, out_min);
    tensor_set_single_float(out_max_tensor, out_max);
    //Step 1) Compute batch statistics
    //Allocate storage
    nn_scratch_reset(nn);
    if(0 != nn_scratch_grow(nn, reductions*sizeof(struct reduction_stats))){
        errlog(nn, "OP_LayerNorm_i16 could not allocate scratch space for statistic metadata");
        return NN_EXECUTE_OUT_OF_SCRATCH_ERROR;
    }
    struct reduction_stats *statistics = nn_scratch_alloc(nn, reductions*sizeof(struct reduction_stats));
    //Initialize values to zero explicitly
    for(uint32_t r=0; r < reductions; ++r){
        statistics[r].reduction_sum = 0;
        statistics[r].reduction_sum_squares = 0;
        statistics[r].reduction_mean = 0;
        statistics[r].reduction_sum = 0;
        statistics[r].reduction_sum_squares = 0ULL;
        statistics[r].reduction_variance = 0.0f;
        statistics[r].reduction_std = 0.0f;
        statistics[r].reduction_inv_std = 0.0f;
        statistics[r].reduction_final_scale = 0.0f;
        statistics[r].reduction_layernorm_scale = 0;
        statistics[r].reduction_layernorm_shamt = 0;
        statistics[r].reduction_all_same = 0U;
    }
    //uint32_t batch_size = height * width *depth;
    //Collect stats
    //Set up our HVX supervisor
    struct layernorm_hvx_runstate_shared *shared_info;
    if(0 != nn_scratch_grow(nn, sizeof(struct layernorm_hvx_runstate_shared))){
        errlog(nn, "OP_LayerNorm_i16 could not allocate scratch space for hvx runstate metadata");
        return NN_EXECUTE_OUT_OF_SCRATCH_ERROR;
    }
    shared_info = nn_scratch_alloc(nn, sizeof(struct layernorm_hvx_runstate_shared));
    //Assign/compute the required values
    shared_info->in_data = in_tensor->data;
    shared_info->curr_pos_hvx = 0U;
    shared_info->curr_reduction = 0U;
    shared_info->num_reductions = reductions;
    shared_info->statistics = statistics;
    shared_info->vectors_in_reduction = reduction_size / ELEMENTS_PER_HVX_16B;
    shared_info->vectors_in_input = shared_info->vectors_in_reduction * reductions;
    shared_info->leftovers_per_reduction = reduction_size % ELEMENTS_PER_HVX_16B;
    shared_info->scale_data = scale_tensor->data;
    shared_info->bias_data = bias_tensor->data;
    shared_info->out_data = out_tensor->data;
    nn_sem_init(&(shared_info->worker_sem), 0);
    //Invoke HVX
    int hvx_statistics_return = statistics_hvx_supervisor(self, nn, shared_info);
    if(0 != hvx_statistics_return){
        errlog(nn, "OP_LayerNorm_i16 id: %x failure %d occured during HVX computation of statistics!", self->node_id, hvx_statistics_return);
        return hvx_statistics_return;
    }
    float variance_scale = (in_scale*in_scale)/((float)reduction_size * (float)reduction_size);
    float bias_to_out = bias_scale / out_scale;
    int32_t bias_to_out_shamt = (bias_to_out <= 1.0f)?0:flt_getexp(bias_to_out);
    uint32_t bias_to_out_q = roundf_u32((float)(1U<<(31-bias_to_out_shamt))*bias_to_out);
    shared_info->bias_scale = bias_to_out_q;
    shared_info->bias_shamt = bias_to_out_shamt;
    //Safety step of resetting the worker semaphore
    nn_sem_init(&(shared_info->worker_sem), 0);
    for(uint32_t r =0; r < reductions; ++r){
        //Compute the floating point values that we need
        statistics[r].reduction_mean = saturate_i16(roundf_i32((float)(statistics[r].reduction_sum)/(float)reduction_size));
        statistics[r].sum_square = (uint64_t)(statistics[r].reduction_sum *statistics[r].reduction_sum);
        statistics[r].scaled_sum_of_squares = (uint64_t)(reduction_size * statistics[r].reduction_sum_squares);
        statistics[r].variance_quant = statistics[r].scaled_sum_of_squares - statistics[r].sum_square;
        //There is another way to hit the bad case which will occur here
        if(statistics[r].variance_quant == 0){
            //Bad case
            statistics[r].reduction_all_same = 1;
            statistics[r].reduction_mean = 0;
            statistics[r].reduction_layernorm_scale = 0;
            statistics[r].reduction_layernorm_shamt = 0;       
        }
        else{
            statistics[r].reduction_variance = statistics[r].variance_quant * variance_scale;
            statistics[r].reduction_std = sqrtf(statistics[r].reduction_variance);
            statistics[r].reduction_inv_std = 1.0f/statistics[r].reduction_std;
            statistics[r].reduction_final_scale = (in_scale * statistics[r].reduction_inv_std * scale_scale)/(out_scale);
            statistics[r].reduction_layernorm_shamt = (statistics[r].reduction_final_scale <= 1.0f)?0:flt_getexp(statistics[r].reduction_final_scale);
            statistics[r].reduction_layernorm_scale = roundf_i32((float)(1U<<(31-statistics[r].reduction_layernorm_shamt))*statistics[r].reduction_final_scale);
        }         
    }
    //Step 2 Renormalize
    //Way nicer than the ref code isn't it?
    int normalize_return = normalize_hvx_supervisor(self, nn, shared_info);
    nn_scratch_reset(nn);
    return normalize_return;
}

static int execute_layernorm_16b_ref(struct nn_node *self, struct nn_graph *nn){
    //Ok what is our plan here
    //In general we need to do the following:
    //1) Compute the per batch statistics (mean, variance, etc)
    //2) Normalize the elements
    //3) Multiply the elements by the coefficients
    //4) Add the bias
    const struct tensor *in_tensor = self->inputs[0];
    const struct tensor *in_min_tensor = self->inputs[1];
    const struct tensor *in_max_tensor = self->inputs[2];
    const struct tensor *scale_tensor = self->inputs[3];
    const struct tensor *scale_min_tensor = self->inputs[4];
    const struct tensor *scale_max_tensor = self->inputs[5];
    const struct tensor *bias_tensor = self->inputs[6];
    const struct tensor *bias_min_tensor = self->inputs[7];
    const struct tensor *bias_max_tensor = self->inputs[8];
    const struct tensor *given_out_min_tensor;
    const struct tensor *given_out_max_tensor;
    given_out_min_tensor = self->inputs[9];
    given_out_max_tensor = self->inputs[10];
    const struct tensor *axis_tensor = self->inputs[11];
    int32_t reduction_axis = tensor_get_int32(axis_tensor,0);
    if(reduction_axis > 2){
        errlog(nn, "OP_LayerNorm_i16 id: %x got unexpected reduction axis: %d, greater than 2. Reduction undefined beyond axis 2", self->node_id, reduction_axis);
        return -1;
    }
    struct tensor *out_tensor = self->outputs[0];
    struct tensor *out_min_tensor = self->outputs[1];
    struct tensor *out_max_tensor = self->outputs[2];
    float in_min, in_max, scale_min, scale_max, bias_min, bias_max;
    uint32_t batches = in_tensor->shape.batches;
    uint32_t width = in_tensor->shape.width;
    uint32_t height = in_tensor->shape.height;
    uint32_t depth = in_tensor->shape.depth;
    in_min = tensor_get_float(in_min_tensor, 0);
    in_max = tensor_get_float(in_max_tensor, 0);
    scale_min = tensor_get_float(scale_min_tensor, 0);
    scale_max = tensor_get_float(scale_max_tensor, 0);
    bias_min = tensor_get_float(bias_min_tensor, 0);
    bias_max = tensor_get_float(bias_max_tensor, 0);
    float out_min = tensor_get_float(given_out_min_tensor, 0);
    float out_max = tensor_get_float(given_out_max_tensor, 0);
    float in_scale = get_qi16_level_size(in_min, in_max);
    float scale_scale = get_qi16_level_size(scale_min, scale_max);
    float bias_scale;
    get_qi32_level_size_zero(bias_min, bias_max, &bias_scale);
    adjust_minmax_for_zero_16b(&out_min, &out_max);
    float out_scale = get_qi16_level_size(out_min, out_max);
    //Compute the number of reduction elements r
    //Compute the size of each reduction reduction_size
    //At minimum r = batches
    uint32_t reductions = 1U;
    for(int32_t i = 0; i <= reduction_axis; ++i){
        reductions = reductions * in_tensor->shape.dimension[i];
    }
    logmsg(nn,2, "OP_LayerNorm_i16 id: %x with reduction axis: %d has %u reduction elements", self->node_id, reduction_axis, reductions);
    int32_t reduction_size = 1U;
    for(uint32_t i = 3; i > reduction_axis; --i){
        reduction_size = reduction_size * in_tensor->shape.dimension[i];
    }
    logmsg(nn,2, "OP_LayerNorm_i16 id: %x with reduction axis: %d has %u elements per reduction", self->node_id, reduction_axis, reduction_size);

    if(0!=(tensor_out_prepare_normal(out_tensor, batches, height, width, depth, NN_TYPE_QINT16))){
        return errlog(nn, "OP_LayerNorm_i16 id: %x output tensor too small for input", self->node_id);
    }
    tensor_set_single_float(out_min_tensor, out_min);
    tensor_set_single_float(out_max_tensor, out_max);
    //Step 1) Compute batch statistics
    //Allocate storage
    nn_scratch_reset(nn);
    if(0 != nn_scratch_grow(nn, reductions*sizeof(struct reduction_stats))){
        errlog(nn, "OP_LayerNorm_i16 could not allocate scratch space for statistic metadata");
        return -1;
    }
    struct reduction_stats *statistics = nn_scratch_alloc(nn, reductions*sizeof(struct reduction_stats));
    //Initialize values to zero explicitly
    for(uint32_t r=0; r < reductions; ++r){
        statistics[r].reduction_sum = 0;
        statistics[r].reduction_sum_squares = 0;
        statistics[r].reduction_mean = 0;
        statistics[r].reduction_sum = 0;
        statistics[r].reduction_sum_squares = 0ULL;
        statistics[r].reduction_variance = 0.0f;
        statistics[r].reduction_std = 0.0f;
        statistics[r].reduction_inv_std = 0.0f;
        statistics[r].reduction_final_scale = 0.0f;
        statistics[r].reduction_layernorm_scale = 0;
        statistics[r].reduction_layernorm_shamt = 0;
        statistics[r].reduction_all_same = 0U;
    }
    memset(statistics, 0, reductions*sizeof(struct reduction_stats));
    //Collect stats
    const int16_t *in_data = in_tensor->data;
    float variance_scale = (in_scale*in_scale)/((float)reduction_size * (float)reduction_size);
    for(uint32_t r =0; r < reductions; ++r){
        statistics[r].reduction_max = INT16_MIN;
        statistics[r].reduction_min = INT16_MAX;

        for(uint32_t r_elements=0; r_elements < reduction_size; ++r_elements){
            statistics[r].reduction_sum += *in_data;
            statistics[r].reduction_sum_squares += (uint64_t)(*in_data * *in_data);
            if(*in_data < statistics[r].reduction_min){
                statistics[r].reduction_min = *in_data;
            }
            if(*in_data > statistics[r].reduction_max){
                statistics[r].reduction_max = *in_data;
            }
            ++in_data;

        }
        //Check for the bad case where all input values are the same
        if(statistics[r].reduction_min == statistics[r].reduction_max){
            statistics[r].reduction_all_same = 1;
        }
        else{
            //Compute the floating point values that we need
            statistics[r].reduction_mean = saturate_i16(roundf_i32((float)(statistics[r].reduction_sum)/(float)reduction_size));
            statistics[r].sum_square = (uint64_t)(statistics[r].reduction_sum *statistics[r].reduction_sum);
            statistics[r].scaled_sum_of_squares = (uint64_t)(reduction_size * statistics[r].reduction_sum_squares);
            statistics[r].variance_quant = statistics[r].scaled_sum_of_squares - statistics[r].sum_square;
            statistics[r].reduction_variance = statistics[r].variance_quant * variance_scale;
            statistics[r].reduction_std = sqrtf(statistics[r].reduction_variance);
            statistics[r].reduction_inv_std = 1.0f/statistics[r].reduction_std;
            statistics[r].reduction_final_scale = (in_scale * statistics[r].reduction_inv_std * scale_scale)/(out_scale);
            statistics[r].reduction_layernorm_shamt = (statistics[r].reduction_final_scale <= 1.0f)?0:flt_getexp(statistics[r].reduction_final_scale);
            statistics[r].reduction_layernorm_scale = roundf_u32((float)(1U<<(31-statistics[r].reduction_layernorm_shamt))*statistics[r].reduction_final_scale);
            
        }
    }
    //Step 2 Renormalize
    //We are going to be sneaky here and compute the full output all at once
    //It's ugly but efficient
    //But first we need to calculate so intermediate values
    float bias_to_out = bias_scale / out_scale;
    int32_t bias_to_out_shamt = (bias_to_out <= 1.0f)?0:flt_getexp(bias_to_out);
    uint32_t bias_to_out_q = roundf_u32((float)(1U<<(31-bias_to_out_shamt))*bias_to_out);
    //Set up some pointers
    in_data = in_tensor->data;
    const int16_t *scale_data = scale_tensor->data;
    const int32_t *bias_data = bias_tensor->data;
    int16_t *out_data = out_tensor->data;
    for(uint32_t r = 0; r < reductions; ++r){
        //Let's do some stuff
        //I believe it is more efficient to evaluate a conditional here
        if(statistics[r].reduction_all_same){
            //Ironically this isn't actually the bad case
            //I just call it that
            //Ok so if all the values are the same
            //Then by definition
            //norm = (in - mean) / sqrt(variance + variance_epsilon)
            //norm = (in - in) / sqrt(variance + variance_epsilion)
            //norm = 0 / sqrt(0 + variance_epsilion)
            //norm = 0
            //Therefore
            //out = scale * 0 + bias
            //out = bias
            //So all we do is convert the bias to the output scale
            for(uint32_t r_elements=0; r_elements < reduction_size; ++r_elements){
                int64_t bias_in_out_scale = (int64_t)(*bias_data)<<bias_to_out_shamt;
                bias_in_out_scale = bias_in_out_scale * (int64_t)bias_to_out_q;
                bias_in_out_scale = bias_in_out_scale >> 31;
                *out_data = saturate_i16(bias_in_out_scale);
                ++out_data;
                ++bias_data;
                //Unfortunately we also need to advance the input data pointer here
                ++in_data;
            }
        }
        else{
            //Well our not so bad case didn't happen
            //Oh well
            //Luckily by now we should be in good standing
            //Here's the plan
            //multiply input - reduction_mean by reduction_layernorm_scale
            //This is the normalized input using the output quantization scheme
            //With whatever shifting we need
            //multiply scale by scale to out q
            //Again shift as needed
            //multiply normalized input by output quantization
            //multiply this value by the out_scale_q value
            //Now we have the scaled normalized value in the output quantization scheme
            //Add bias using the scheme above
            //Saturate and write out
            //Rinse and repeat
            for(uint32_t r_elements=0; r_elements < reduction_size; ++r_elements){

                int32_t input_prime = (int32_t)(*in_data)-(int32_t)statistics[r].reduction_mean;
                int64_t scaled_input_prime = (int64_t)input_prime*(int64_t)(*scale_data);
                int64_t scaled_input_in_output = (scaled_input_prime<<statistics[r].reduction_layernorm_shamt)*(int64_t)statistics[r].reduction_layernorm_scale;
                int32_t normalized_input_32b = (int32_t)(scaled_input_in_output>>31);

                int64_t bias_in_out_scale = (int64_t)(*bias_data)<<bias_to_out_shamt;
                bias_in_out_scale = bias_in_out_scale * (int64_t)bias_to_out_q;
                bias_in_out_scale = bias_in_out_scale >> 31;

                int32_t bias_in_out_scale_32b = (int32_t)(bias_in_out_scale);

                int64_t result = normalized_input_32b + bias_in_out_scale_32b;

                *out_data = saturate_i16(result);
                //Advance the pointers
                ++in_data;
                ++out_data;
                ++bias_data;
                ++scale_data;
            }
        }
        //Reset scale and bias pointers
        scale_data = scale_tensor->data;
        bias_data = bias_tensor->data;
    }
    nn_scratch_reset(nn);
    return 0;
}

struct nn_node_ops nn_ops_for_QuantizedLayerNorm_i16_ref = {
    .execute = execute_layernorm_16b_ref,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT(12),
    .n_outputs = NN_IOCOUNT(3),
};

struct nn_node_ops nn_ops_for_QuantizedLayerNorm_i16 = {
    .execute = execute_layernorm_16b,
    .check = NULL,
    .ctor = node_alloc_common,
    .dtor = node_free_common,
    .n_inputs = NN_IOCOUNT(12),
    .n_outputs = NN_IOCOUNT(3),
};