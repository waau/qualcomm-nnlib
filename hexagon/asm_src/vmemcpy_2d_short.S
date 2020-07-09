    .text
    .file "vmemcpy_2d_short.S"
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

// This is a 'short' vmemcpy2, only supports wid <=256.
// (there is a seperate loop for wid >= 129).
//
//
// vmemcpy_2d_short_general( wid,
//    int32_t             wid,       // bytes wide : 0 <= wid <= 128
//    int32_t             ht,        // rows; >=0
//    void                *dst,      // destination address, any allowed
//    size_t              dst_pitch, // row pitch of dest; any allowed
//    const void          *src,      // source address, any allowed
//    size_t              src_pitch);// row pitch of source; any allowed

#define wid                  r0
#define ht                   r1
#define out_ptr              r2
#define out_stride           r4  // need to swap r3,r4 from caller
#define in_ptr               r3
#define in_stride            r5
#define in_ptr_out_ptr       r3:2
#define in_stride_out_stride r5:4
#define out_ptr_x            r6
#define offs_limit           r8
#define out_offs_end         r9
#define maskd_127            r11:10
#define out_offs             r12
#define in_offs              r13
#define in_offs_out_offs     r13:12

#define V_in0           v0
#define V_in1           v1
#define V_in2           v2
#define V_data0x        v3
#define V_data0         v4
#define V_data1         v5
#define V_aligned0      v6
#define V_aligned1      v7

    .globl vmemcpy_2d_short_general
    .balign 64
    .type vmemcpy_2d_short_general,@function
vmemcpy_2d_short_general:
    {
        p0 = cmp.gt( wid,#0)
        p0 = cmp.gt( ht,#0)
        p1 = cmp.gt( wid,#128)
       if( p1.new )jump:nt .L_large_wid    // wid 129..256 cases
    } {
        if(!p0) jumpr:nt r31
        p3 = cmp.gt(ht,#1)
    }
    {
        r3=r4;
        r4=r3;
        ht = add(ht,#-1)
        maskd_127 = combine(#127,#127)
    } {
        offs_limit = sub(#128,wid)
        loop0( .L_loop, ht )
        in_offs_out_offs = and( in_ptr_out_ptr,maskd_127)
        q0 = vsetq(out_ptr)
    } {
        if(!p3) jump:nt .L_one_iter
        p0 = cmp.gt( in_offs,offs_limit)
        out_offs_end = add(out_offs,wid)
        V_in0 = vmem(in_ptr+#0)
    }

    .balign 32
.L_loop:
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
        V_aligned0 = vlalign(V_data0,V_data0,out_ptr_x) //[2]
        p0 = cmp.gt( in_offs,offs_limit)                  //[1]
        V_in0 = vmem(in_ptr+#0)                           //[1]
    } {
        q0 = vsetq(out_ptr)                               //[1]
        out_offs_end = add(out_offs,wid)                  //[1]
        if (!q2) vmem(out_ptr_x+#0) = V_aligned0          //[2]
    }:endloop0

.L_one_iter:
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
        jumpr r31
    }
   .balign 32
.L_store_two_last:
    {
        if (!q0) vmem(out_ptr+#0) = V_aligned0
    } {
        if (q1) vmem(out_ptr+#1) = V_aligned0
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
        if (!q0) vmem(out_ptr_x+#0) = V_aligned0         //[2]
        out_offs_end = add(out_offs,wid)                 //[1]
    } {
        if (q1) vmem(out_ptr_x+#1) = V_aligned0          //[2]
        q0 = vsetq(out_ptr)                              //[1]
    }:endloop0
    {
        p1 = cmp.gt(out_offs,offs_limit)
        q1 = vsetq2(out_offs_end)
        jump .L_epilogue
    }

//////////////////////////////////////////////////
//////////////////////////////////////////////////
// Code for width >= 129, <= 256
// For this range, regardless of alignment, the data always spans 2 or 3 vectors;
// since we never have to use two 'q' masks in one store, the loop can run
// without conditional branches.
//
//
/* Core operation sequence, to read from in_ptr and store to out_ptr:
        in_offs  = and(in_ptr,#127)
        out_offs  = and(out_ptr,#127)

        // loading
        p0 = cmp.gt( in_offs,offs_limit)   // do we need 3rd load ? offs + wid > 256
        V_in0 = vmem(in_ptr+#0)
        V_in1 = vmem(in_ptr+#1)
        if(p0) V_in2 = vmem(in_ptr+#2)
        V_data0 = valign(V_in1,V_in0,in_ptr)   // extract aligned data
        V_data1 = valign(V_in2,V_in1,in_ptr)   //

        p1 = cmp.gt(out_offs,offs_limit)       // need 3rd store?  offs + wid > 256
        // position data for store
        out_ptr_x = out_ptr
        out_offs_end = add(out_offs,wid)
        // rotate 'up' in 256 bytes, to position for output
        V_aligned0 = vlalign(V_data0,V_data1,out_ptr_x)
        V_aligned1 = vlalign(V_data1,V_data0,out_ptr_x)
        q0 = vsetq(out_ptr)
        q1 = vsetq2(out_offs_end)
        if(!q0) vmem(out_ptr_x++#1) = V_aligned0
        if(p1) {
            // need 'middle' store
            vmem(out_ptr_x++#1) = V_aligned1
            V_aligned1 = V_aligned0
        }
        if(q1) vmem(out_ptr_x+#0) = V_aligned1
***********/


    .balign 32
.L_large_wid:     {
        r3=r4;
        r4=r3;
        ht = add(ht,#-1)
        maskd_127 = combine(#127,#127)
    } {
        if(!p0) jumpr:nt r31     // ht is <=0
        offs_limit = sub(#256,wid)
        V_in0 = vmem(in_ptr+#0)                               //[1]
        in_offs_out_offs = and( in_ptr_out_ptr,maskd_127)     //[1]
    } {
        out_ptr_x = out_ptr                                   //[1]
        p0 = cmp.gt( in_offs,offs_limit)                      //[1]
        out_offs_end = add(out_offs,wid)                      //[1]
        q0 = vsetq(out_ptr)                                   //[1]
    } {
        p3 = cmp.gt(ht,#0)    // skip the loop?
        loop0( .L_2loop, ht )
        V_in1.cur = vmem(in_ptr+#1)                           //[1]
        V_data0 = valign(V_in1,V_in0,in_ptr)                  //[1]
    } {
        if(!p3) jump:nt .L_2one_iter
        if (p0) V_in2.cur = vmem(in_ptr+#2)                   //[1]
        V_data1 = valign(V_in2,V_in1,in_ptr)                  //[1]
        in_ptr_out_ptr = vaddw(in_ptr_out_ptr, in_stride_out_stride) // bump ptrs
    }

    .balign 32
// note, out_ptr_x is set at the end of stage 1, and is modified with ++#1 during
// stage 2; but the 7 lsbs (used in vlalign) are not affected by the ++#1
.L_2loop:
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
        out_offs_end = add(out_offs,wid)                      //[1]
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


.L_2one_iter:
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
        jumpr r31
    }

    .falign
.Lfunc_end:
    .size vmemcpy_2d_short_general, .Lfunc_end-vmemcpy_2d_short_general

#endif

