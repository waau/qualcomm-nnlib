    .text
    .file "vmemcpy_3d_short.S"
#if __HEXAGON_ARCH__ >= 62
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

// This is a 'short' vmemcpy_3d, only supports depth in range 1..256
// The operation is equivalent to the following:
//
//  for( size_t ih = 0; ih < ht; ih ++ )
//    for( size_t iw = 0; iw < wid; iw ++ )
//       memcpy(
//          (char*)dest + ih * dst_pitch_h + iw * dst_pitch_w,
//          (char const*)src + ih * src_pitch_h + iw * src_pitch_w,
//          dep);
// All access are by vector load/store; no 'spurious' vector
// accesses are made.
// There is an entirely separate loop for the depth= 129..256 case.
//
// vmemcpy_3d_short_general(
//    void                *dst,      // destination address, any allowed
//    const void          *src,      // source address, any allowed
//    size_t              dst_pitch_h, // h pitch of dest; any allowed
//    size_t              src_pitch_h, // h pitch of src; any allowed
//    size_t              dst_pitch_w, // h pitch of dest; any allowed
//    size_t              src_pitch_w, // h pitch of src; any allowed
//    int32_t             height       // rows; >=0
//    int32_t             width,       // cols; >=0
//    int32_t             depth        // bytes wide : 0 <= wid <= 128
//   )
#define out_ptr              r0
#define in_ptr               r1
#define out_stride_h         r2
#define in_stride_h          r3
#define out_stride           r4
#define in_stride            r5

#define in_ptr_out_ptr       r1:0
#define in_stride_h_out_stride_h r3:2
#define in_stride_out_stride r5:4

#define depth                r6
#define out_ptr_x            r7
#define offs_limit           r8
#define out_offs_end         r9
#define maskd_127            r11:10
#define out_offs             r12
#define in_offs              r13
#define in_offs_out_offs     r13:12
#define height               r14
#define width                r15
#define width_height         r15:14


#define in_ptr_nxt_out_ptr_nxt       r17:16


#define V_in0           v0
#define V_in1           v1
#define V_in2           v2
#define V_data0x        v3
#define V_data0         v4
#define V_data1         v5
#define V_aligned0      v6
#define V_aligned1      v7

    .globl vmemcpy_3d_short_general
    .balign 64
    .type vmemcpy_3d_short_general,@function
vmemcpy_3d_short_general:
    {
        maskd_127 = combine(#127,#127)
        width_height = memd(sp+#0)                  // read ht,wid
        memd(sp+#0) = r17:16                        // save regs on top of ht,wid
    } {
        depth = memw(sp+#8)
        p0 = cmp.gt( width,#0)
        p0 = cmp.gt( height,#0)
        if(!p0.new) jumpr:nt r31
    } {
        p3 = cmp.gt(depth,#128)           // need long loop?
        p0 = cmp.gt(depth,#0)
        p2 = cmp.gt(width,#1)
        if (!p2.new) jump:nt .L_w_h_swap
    }
.L_w_h_swap_done:
    {
        if(!p0) jump:nt .L_restore_return	// since depth <= 0
        if( p3) jump:nt .L_large_depth		// since depth > 128
        width = add(width,#-1)              // adjust to get loop count
        offs_limit = sub(#128,depth)
    } {  // below here, can't return w/o restoring r17:16
        in_ptr_nxt_out_ptr_nxt = vaddw( in_ptr_out_ptr, in_stride_h_out_stride_h)
        loop1( .L_h_loop, height )
    }
    .balign 32
.L_h_loop:
    {
        loop0( .L_w_loop, width )
        in_offs_out_offs = and( in_ptr_out_ptr,maskd_127)
        q0 = vsetq(out_ptr)
    } {
        if(!p2) jump:nt .L_w_is_one
        p0 = cmp.gt(in_offs,offs_limit)
        out_offs_end = add(out_offs,depth)
        V_in0 = vmem(in_ptr+#0)
    }

    .balign 32
.L_w_loop:
    {
        p1 = cmp.gt(out_offs,offs_limit)            //[2]
        q1 = vsetq2(out_offs_end)                   //[2]
        out_ptr_x = out_ptr                         //[2] (save out_ptr for store)
    } {
        V_data0 = valign(V_in1,V_in0,in_ptr)         //[2]
        in_ptr_out_ptr = vaddw(in_ptr_out_ptr, in_stride_out_stride) // bump ptrs
        if(p0) V_in1.cur = vmem(in_ptr+#1)          //[2]
    } {
        q2 = or(q0,!q1)                             //[2]
        in_offs_out_offs = and( in_ptr_out_ptr,maskd_127)   //[1]
        if(p1) jump:nt .L_store_two                 //[2]
    } {
        V_aligned0 = vlalign(V_data0,V_data0,out_ptr_x)   //[2]
        p0 = cmp.gt( in_offs,offs_limit)                  //[1]
        V_in0 = vmem(in_ptr+#0)                           //[1]
    } {
        q0 = vsetq(out_ptr)                               //[1]
        out_offs_end = add(out_offs,depth)                //[1]
        if (!q2) vmem(out_ptr_x+#0) = V_aligned0          //[2]
    }:endloop0

.L_w_is_one:
    {
        p1 = cmp.gt(out_offs,offs_limit)
        q1 = vsetq2(out_offs_end)
    }
.L_epilogue:
    {
        if(p0) V_in1.cur = vmem(in_ptr+#1)
        V_data0 = valign(V_in1,V_in0,in_ptr)
    } {
        q2 = or(q0,!q1)
        V_aligned0 = vlalign(V_data0,V_data0,out_ptr)
        if(p1) jump:nt .L_store_two_last
    } {
        if (!q2) vmem(out_ptr+#0) = V_aligned0
        in_ptr_nxt_out_ptr_nxt = vaddw( in_ptr_nxt_out_ptr_nxt, in_stride_h_out_stride_h)
        in_ptr_out_ptr = in_ptr_nxt_out_ptr_nxt
    }:endloop1
    {
        r17:16 = memd(sp+#0)
        jumpr r31
    }

   .balign 32
.L_store_two_last:
    {
        if (!q0) vmem(out_ptr+#0) = V_aligned0
    } {
        if (q1) vmem(out_ptr+#1) = V_aligned0
        in_ptr_nxt_out_ptr_nxt = vaddw( in_ptr_nxt_out_ptr_nxt, in_stride_h_out_stride_h)
        in_ptr_out_ptr = in_ptr_nxt_out_ptr_nxt
    }:endloop1
.L_restore_return:
    {
        r17:16 = memd(sp+#0)
        jumpr r31
    }


// loop ending for when two stores needed.
    .balign 32
.L_store_two:
    {
        V_aligned0 = vlalign(V_data0,V_data0,out_ptr_x)   //[2]
        p0 = cmp.gt( in_offs,offs_limit)                  //[1]
        V_in0 = vmem(in_ptr+#0)                           //[1]
    } {
        if (!q0) vmem(out_ptr_x+#0) = V_aligned0          //[2]
        out_offs_end = add(out_offs,depth)                //[1]
    } {
        if (q1) vmem(out_ptr_x+#1) = V_aligned0           //[2]
        q0 = vsetq(out_ptr)                               //[1]
    }:endloop0
    {
        p1 = cmp.gt(out_offs,offs_limit)
        q1 = vsetq2(out_offs_end)
        jump .L_epilogue
    }
    .balign 32
// if called with width = 1:
//     width = height; height =1; w_strides = h_strides;
//     restore p2 = width > 1
//     and carry on as usual.
.L_w_h_swap:
    {
        width_height = combine(height,#1)
        in_stride_out_stride = in_stride_h_out_stride_h
        p2 = cmp.gt(height,#1)
        jump .L_w_h_swap_done
    }

////////////////////////////////
/// code for depth >= 129, <=256 - a whole separate loop
.L_large_depth:
    {
        offs_limit = sub(#256,depth)
        in_ptr_nxt_out_ptr_nxt = vaddw( in_ptr_out_ptr, in_stride_h_out_stride_h)
        loop1( .L_2_h_loop, height )
    }
    .balign 32
.L_2_h_loop:
    {
        loop0( .L_2_w_loop, width )
        in_offs_out_offs = and( in_ptr_out_ptr,maskd_127)     //[1]
        V_in0 = vmem(in_ptr+#0)                               //[1]
    } {
        p0 = cmp.gt( in_offs,offs_limit)                      //[1]
        out_offs_end = add(out_offs,depth)                    //[1]
    } {
        V_in1.cur = vmem(in_ptr+#1)                           //[1]
        V_data0 = valign(V_in1,V_in0,in_ptr)                  //[1]
    } {
        if (p0) V_in2.cur = vmem(in_ptr+#2)                   //[1]
        V_data1 = valign(V_in2,V_in1,in_ptr)                  //[1]
    } {
        in_ptr_out_ptr = vaddw(in_ptr_out_ptr, in_stride_out_stride) // bump ptrs
        out_ptr_x = out_ptr                                   //[1]
        q0 = vsetq(out_ptr);                                  //[1]
        if(!p2) jump:nt .L_2_w_is_one
    }

    .balign 32
.L_2_w_loop:
    {
        V_aligned0 = vlalign(V_data0,V_data1,out_ptr_x)       //[2]
        p1 = cmp.gt(out_offs,offs_limit)                      //[2]
        in_offs_out_offs = and( in_ptr_out_ptr,maskd_127)     //[1]
        V_in0 = vmem(in_ptr+#0)                               //[1]
    } {
        if(!q0) vmem(out_ptr_x++#1) = V_aligned0              //[2]
        V_data0x = V_data0                                    //[2]
        q1 = vsetq2(out_offs_end)                             //[2]
        p0 = cmp.gt( in_offs,offs_limit)                      //[1]
    } {
        V_in1.cur = vmem(in_ptr+#1)                           //[1]
        V_data0 = valign(V_in1,V_in0,in_ptr)                  //[1]
    } {
        V_aligned1 = vlalign(V_data1,V_data0x,out_ptr_x)      //[2]
        out_offs_end = add(out_offs,depth)                    //[1]
    } {
        if (p1) vmem(out_ptr_x++#1) = V_aligned1              //[2]
        if (p1) V_aligned1 = V_aligned0                       //[2]
        if (p0) V_in2.cur = vmem(in_ptr+#2)                   //[1]
        V_data1 = valign(V_in2,V_in1,in_ptr)                  //[1]
    } {
        if(q1) vmem(out_ptr_x+#0) = V_aligned1                //[2]
        in_ptr_out_ptr = vaddw(in_ptr_out_ptr, in_stride_out_stride) // bump ptrs
        out_ptr_x = out_ptr                                   //[1]
        q0 = vsetq(out_ptr);                                  //[1]
    } :endloop0


.L_2_w_is_one:
    {
        V_aligned0 = vlalign(V_data0,V_data1,out_ptr_x)       //[2]
        p1 = cmp.gt(out_offs,offs_limit)                      //[2]
    } {
        if(!q0) vmem(out_ptr_x++#1) = V_aligned0              //[2]
        q1 = vsetq2(out_offs_end)                             //[2]
    } {
        V_aligned1 = vlalign(V_data1,V_data0,out_ptr_x)       //[2]
    } {
        if (p1) vmem(out_ptr_x++#1) = V_aligned1              //[2]
        if (p1) V_aligned1 = V_aligned0                       //[2]
    } {
        if(q1) vmem(out_ptr_x+#0) = V_aligned1                //[2]
        in_ptr_nxt_out_ptr_nxt = vaddw( in_ptr_nxt_out_ptr_nxt, in_stride_h_out_stride_h)
        in_ptr_out_ptr = in_ptr_nxt_out_ptr_nxt
    }:endloop1
    {
        r17:16 = memd(sp+#0)
        jumpr r31
    }




    .falign
.Lfunc_end:
    .size vmemcpy_2d_short_general, .Lfunc_end-vmemcpy_2d_short_general

#endif

