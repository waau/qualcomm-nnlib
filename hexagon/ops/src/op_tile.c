
/*
 * Copyright (c) 2018-2020, The Linux Foundation. All rights reserved.
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
#include <nn_graph.h>
#include <string.h>
#include "hvx_inlines.h"
#if defined(__hexagon__)
#include <hexagon_types.h>
#endif


struct tile_info {
	int has_valid_vctl;
	HVX_Vector *vctl_ptr;
};

struct tile_run_state {
	struct tile_info *info;
	uint8_t *in_data;
	uint8_t *out_data;
	nn_sem_t done_sem;
	int batches;
	int height;
	int width;
	int depth;
	int b_multiple;
	int h_multiple;
	int w_multiple;
	int d_multiple;
};

//limitation: d_in must be power of 2(including 1), and (d_in*d_mul) < vector length
static void do_depth_tile_hvx(struct tile_run_state *rst, const uint8_t *in_data, uint8_t *out_data, const HVX_Vector *pvctl)
{
	const uint8_t *pin = in_data;
	uint8_t *pout = out_data;
	size_t size = rst->width * rst->depth * rst->w_multiple * rst->d_multiple;
	int d_mul = rst->d_multiple;
	while (pout < (out_data + size)) {
		//unaligned load vector, first 128/d_mul bytes are used in vdelta, the rest are useless
		HVX_Vector vin = q6op_V_vldu_A((HVX_Vector *)(pin));
		//use permutation command for tile along depth
		// vin:		[w0][0],[w0][1],...,[w0][depth-1],
		//			[w1][0],[w1][1],...,[w1][depth-1],
		//			X,X,......X (unused)
		// vout:	[w0][0],[w0][1],...,[w0][depth-1], [w0][0],[w0][1],...,[w0][depth-1], ...
		//			[w1][0],[w1][1],...,[w1][depth-1], [w1][0],[w1][1],...,[w1][depth-1], ...
		//			......
		HVX_Vector vout = Q6_V_vdelta_VV(vin, *pvctl);
		//unaligned store vector
		q6op_vstu_AV((HVX_Vector *)pout, vout);

		pin += 128/d_mul;
		pout += (128/d_mul) * d_mul;
	}
}

static void tile_worker_thread(struct nn_graph * nn, void * rstpv)
{
	logmsg(nn,2,"tile_worker_thread");
	struct tile_run_state *rst = (struct tile_run_state *)rstpv;
	uint8_t  *in_data       = (uint8_t *)rst->in_data;
	uint8_t  *out_data      = (uint8_t *)rst->out_data;

	int batches = rst->batches;
	int height = rst->height;
	int width = rst->width;
	int depth = rst->depth;

	int b_multiple = rst->b_multiple;
	int h_multiple = rst->h_multiple;
	int w_multiple = rst->w_multiple;
	int d_multiple = rst->d_multiple;

	int dwh = depth*width*height;
	int dw = depth*width;

	struct tile_info *info = rst->info;

	int use_hvx = 0;
	if ((depth & depth-1) == 0 && (depth*d_multiple) <= 128 && depth <= 8 && d_multiple > 1 ) {
		use_hvx = 1;
		if (info->has_valid_vctl == 0) {
			// design map and ctrl for vdelta only once
			//  one group contains d_multiple*depth data, d = { 0,1,2,...,depth-1}, duplicate d_multiple times,
			//  each command can handle 128/(d_multiple*depth) groups.
			HVX_Vector vmap;
			uint8_t *map = (uint8_t*)(&vmap);
			int n_groups = 128/(d_multiple*depth);
			for (int src=0, dst=0; src < (n_groups*depth); ) {
				for (int m=0; m < d_multiple; m++) {
					for (int d=0; d < depth; d++) {
						map[dst++] = src + d;
					}
				}
				src += depth; //next group
			}
			*(info->vctl_ptr) = design_for_delta(vmap, vmap, 0);
			info->has_valid_vctl = 1;
		}
	}

	int chunk_size_d = depth;
	int chunk_size_w = chunk_size_d * d_multiple*width;
	int chunk_size_h = chunk_size_w * w_multiple*height;
	int chunk_size_b = chunk_size_h * h_multiple*batches;

	int counter = 0;

	for(int b = 0; b < batches; b++){
		int c1 = b*dwh;
		uint8_t *copy_start_w = out_data+counter;
		for(int h = 0; h < height; h++){
			int c2 = h*dw + c1;
			uint8_t *copy_start_d = out_data+counter;
			// inner loop for tile along depth
			if (d_multiple > 1) {
				if (use_hvx) {
					// special case: data replication with vdelta
					do_depth_tile_hvx(rst, in_data+c2, out_data+counter, info->vctl_ptr);
				}
				else {
					// general case: copy src 2d square [width,depth] to dst for d_multiple times,
					// increase offset of dst square by depth each time.
					int offset_2d = counter;
					for(int d_mul = 0; d_mul < d_multiple; d_mul++){
						int src_width = depth;
						int src_height = width;
						int src_stride = depth;
						int dst_stride = depth * d_multiple;
						vmemcpy_2d_general_asm(src_width, src_height, out_data+offset_2d, dst_stride, in_data+c2, src_stride);
						offset_2d += depth;
					}
				}
			}
			else {
				// special case: if d_multiple is 1, the square in dst is also contiguous (stride==width),
				// we can use 1d vmemcpy instead of 2d for speed.
				vmemcpy_asm(out_data+counter, in_data+c2, depth*width);
			}
			counter += chunk_size_w;

			for(int w_mul = 1; w_mul < w_multiple; w_mul++){
				vmemcpy_asm(out_data+counter, copy_start_d, chunk_size_w );
				counter += chunk_size_w;
			}
		}
		for(int h_mul = 1; h_mul < h_multiple; h_mul++){
			vmemcpy_asm(out_data+counter, copy_start_w, chunk_size_h );
			counter += chunk_size_h;
		}
	}
	for(int b_mul = 1; b_mul < b_multiple; b_mul++){
		vmemcpy_asm( out_data+counter, out_data, chunk_size_b );
		counter += chunk_size_b;
	}

	nn_sem_post(&rst->done_sem);
}

/*
 *
 * Now that that's out of the way, let's get to the good stuff.
 *
 * This contains a tile node
 */


static int tile_execute(struct nn_node *self, struct nn_graph *nn)
{
	logmsg(nn,2,"tile execute. self=%p ",self);

	const struct tensor *in_tensor = self->inputs[0];
	const struct tensor *multiples_tensor = self->inputs[1];

	struct tensor *out_tensor = self->outputs[0];
	struct tensor *out_min_tensor = self->outputs[1];
	struct tensor *out_max_tensor = self->outputs[2];

	uint8_t *in_data = in_tensor->data;
	uint32_t *multiples = multiples_tensor->data;
	float in_min = tensor_get_float(self->inputs[2], 0);
	float in_max = tensor_get_float(self->inputs[3], 0);

	uint8_t *out_data = out_tensor->data;
	float *out_min = out_min_tensor->data;
	float *out_max = out_max_tensor->data;

	int batches = in_tensor->shape.batches;
	int height = in_tensor->shape.height;
	int width = in_tensor->shape.width;
	int depth = in_tensor->shape.depth;

    int multiple_size = multiples_tensor->shape.batches * multiples_tensor->shape.height * multiples_tensor->shape.width * multiples_tensor->shape.depth;

	int b_multiple = multiple_size > 3 ? multiples[0] : 1;
	int h_multiple = multiple_size > 2 ? multiples[multiple_size-3] : 1;
	int w_multiple = multiple_size > 1 ? multiples[multiple_size-2] : 1;
	int d_multiple = multiples[multiple_size-1];

	tensor_out_prepare_normal(out_min_tensor, 1, 1, 1, 1, NN_TYPE_FLOAT);
	tensor_out_prepare_normal(out_max_tensor, 1, 1, 1, 1, NN_TYPE_FLOAT);
	tensor_out_prepare_normal(out_tensor, batches*b_multiple, height*h_multiple, width*w_multiple, depth*d_multiple, NN_TYPE_QUINT8);

	*out_min = in_min;
	*out_max = in_max;

	//worker thread
	void (*thread_run_fp)( struct nn_graph * nn, void * rstpv) = tile_worker_thread;
	struct tile_run_state rstt;
	rstt.info = (struct tile_info*)self->opaque;
	rstt.in_data = in_data;
	rstt.out_data = out_data;
	rstt.batches = batches;
	rstt.height = height;
	rstt.width = width;
	rstt.depth = depth;
	rstt.b_multiple = b_multiple;
	rstt.h_multiple = h_multiple;
	rstt.w_multiple = w_multiple;
	rstt.d_multiple = d_multiple;
	nn_sem_init(&rstt.done_sem, 0);
	nn_os_work_for_vector(nn,thread_run_fp, &rstt);
	nn_sem_wait(&rstt.done_sem);

	return 0;
}

static int tile_check(struct nn_node *self, struct nn_graph *nn)
{
	if( self->opaque == NULL){
		struct tile_info *info = (struct tile_info*)nn_calloc( 1, sizeof(struct tile_info));
		if(info == NULL) return errlog(nn,"calloc failed");
		if (info->vctl_ptr == NULL) {
			if ((info->vctl_ptr = nn_memalign(128,128)) == NULL) {
				nn_free(info);
				return errlog(nn, "can't allocate vctl_ptr");
			}
		}
		self->opaque = (void*)info;
	}
	return 0;
};

static int tile_dtor(struct nn_node *self, struct nn_graph *nn)
{
	if( self->opaque != NULL){
		struct tile_info * info = (struct tile_info*)self->opaque;
		if( info->vctl_ptr != NULL) nn_free(info->vctl_ptr);
		nn_free(self->opaque);
		self->opaque = NULL;
	}
	return node_free_common(self, nn);
}

struct nn_node_ops nn_ops_for_QuantizedTile_8 = {
	.execute = tile_execute,
	.check = tile_check,
	.ctor = node_alloc_common,
	.dtor = tile_dtor,
	.n_inputs = NN_IOCOUNT(4),
	.n_outputs = NN_IOCOUNT(3),
};

