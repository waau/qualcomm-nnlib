# **ML on Hexagon Best Practice Guide**

# Introduction

## Purpose
This document is intended for users familiar with implementing graphs with Hexagon™-NN. It provides guidelines for maximizing graph execution performance.

For general guidance on how the Hexagon NN library works, see *Qualcomm® Hexagon™ Neural Network (NN) Library User Guide* (80-VB419-110) 

## Conventions
Function declarations, function names, type declarations, attributes, and code samples appear in a different font, for example, ```cp armcc armcpp```.

Code variables appear in angle brackets, for example, ```<number>```.

Commands to be entered appear in a different font, for example, ```copy a:*.* b:```.

Button and key names appear in bold font, for example, click **Save** or press **Enter**.

## Technical assistance
For assistance or clarification on information in this document, submit a case to Qualcomm Technologies, Inc. (QTI) at https://createpoint.qti.qualcomm.com/.

If you do not have access to the CDMATech Support website, register for access or send email to support.cdmatech@qti.qualcomm.com. 
 

# General Considerations
## Floating point

Floating point operations are in general not optimized.  Use the 8-bit quantized versions for best performance.

## Convolutions and Biases

Users should always add a bias op after a convolution, even if the biases are all zero.  Hexagon-NN can optimize this pattern, as described in Chapter 4.

## Tensor Depth
Dimension sizes in Hexagon-NN can make a significant difference.
In most cases, the optimal depth size is a multiple of 32 bytes. For depth sizes that are not a multiple of 32, the performance will be the same as if the depth size was the next multiple of 32. For example, depth size of 33 costs as much as depth size of 64.

Some optimizations have been done to handle depths <= 4 for convolutions as the first op of the network since a lot of networks take images as input, with depth of 3.

# Operation Optimization Levels
The following is a list of op configurations in Hexagon-NN with known relative performance. This is not an exhaustive list and is subject to change. The leveling guidelines are as follows:

- Level 1: Fully optimized
- Level 2: Mostly optimized, but may have room to improve
- Level 3: Configuration that may not be fully optimized
- Level 4: Known slow configuration

Ops not shown here may or may not perform well pending op configuration.  A future update to this document may include optimization status of more ops.


## Level 1 : Fully Optimized

| Op Name       | Configuration | Notes |
| ------------- |---------------| ------|
| QuantizedDepthwiseConv2d_8x8to32 </br> (followed by a bias add/requantize and optionally a relu) | kernel size is 3x3. </br> Stride width is 1,2, </br> input channels=output channels=groups, </br>depth_multiplier=1 | This is the most optimized version of depthwise convolution |
| QuantizedDepthwiseConv2d_8x8to32 </br> (followed by a bias add/requantize and optionally a relu) | kernel size is Hx3, Hx5, Hx7. </br> Stride width is 1,2, </br> input channels=output channels=groups, </br> depth_multiplier=1 | Not as hyper-optimized as 3x3. |
| QuantizedMatMul_8x8to32 </br> (followed by a bias add/requantize and optionally a relu) | input depth/output depth is a multiple of 16 </br> output depth is >= 32  |   |
| Quantized Add/Mul |    |    |
| Avg pool | </br> Cases where window size 3x3, stride 1 </br> Any case where output is b x 1 x 1 x d </br> Other cases it’s a 2 |     |
| Max pool | </br> Cases where window size 3x3, stride 1,2 </br> Cases where window size 2x2 stride 2 |    |
| Depth to Space | </br> stride_w must be either <=4, or exactly 8; </br> out depth must be a multiple of 32. </br> Special case: out_depth=16 with stride_w=2 |    |


## Level 2 : Mostly Optimized
| Op Name       | Configuration | Notes |
| ------------- |---------------| ------|
| QuantizedConv2d_8x8to32 </br> (followed by a bias add/requantize and optionally a relu) | input depth < 5, not at the start of a network |  |
| QuantizedConv2d_8x8to32 </br> (followed by a bias add/requantize and optionally a relu) | input depth/output depth not a multiple of 32 | In general, the further from the next multiple of 32, the worse the performance impact will be |
| QuantizedTransposeConv2d_8x8p32to8 </br> Deconvolution (SNPE) </br> (optionally followed by relu) | input depth/output depth not a multiple of 32 |  |
| QuantizedResizeBilinear_8 | x2 when depth is either >= 128 or a power of 2; </br> x4 when depth is a power of 2 less than 32; </br> x4 in width, when input width is 1 more than a multiple of 3, and w_in * depth < 128*3 </br> x2 in both directions |  |

## Level 3 : May Not Be Fully Optimized
| Op Name       | Configuration | Notes |
| ------------- |---------------| ------|
| QuantizedTransposeConv2d_8x8p32to8 </br> Deconvolution (SNPE) </br> (optionally followed by relu) | </br> stride > 4 </br> stride > 4 != 8, get 3. </br> Stride <=3 is 1 | Causes d32 converts (only the w stride is significant). </br> Related to DepthToSpace_d32 limitations |

## Level 4 : Known Slow Configuration
| Op Name       | Configuration | Notes |
| ------------- |---------------| ------|
| QuantizedResizeBilinear_8 | All configurations other than in Level 2 | 3 if depth is >= 32. |
| QuantizedConv2d_8x8to32 | Slices of weights do not fit into VTCM | Optimal performance is when 32 outchannels * filt_h * filt_w * input_depth fits in VTCM. |
| QuantizedConv2d_8x8to32 | L2 Cache miss when prefetch requests for activation data exceed L2 cache available size | Optimal performance: Width*depth <= 64K </br> and </br> activation data needed to get 2 output lines < 128K |


# Op Combinations
Hexagon-NN is able to execute certain combinations of ops faster than the sum of the execution times of the constituent ops if they were executed in isolation.  Hexagon-NN can fuse certain layer patterns and perform the equivalent math faster. These combinations are as follows  (list subject to change) :
- Op → Requantize→ QuantizedBiasAdd_8p8to32 → Requantize → Relu, where:
    - The first op is one of the following:
        - Shape_int32
        - QuantizedAdd_8p8to32
        - QuantizedConv2d_8x8to32
        - QuantizedConv2d_16x16to32
        - QuantizedDepthwiseConv2d_8x8to32
        - QuantizedMul_8x8to32
        - QuantizedMatMul_8x8to32
    - Requantize is one of the following:
        - QuantizeDownAndShrinkRange_32to8
        - QuantizeDownAndShrinkRange_32to16
        - Requantize_32to8, Requantize_32tou16
    - Relu is optional and one of the following:
        - QuantizedRelu_8
        - QuantizedReluX_8
        - QuantizedClamp_8
- Op → QuantizeDownAndShrinkRange_32to8 → QuantizedMul_8x8to32, where:
    - The first op is one of the following
        - QuantizedBatchNorm_8x8p8to8
        - QuantizedBatchNorm_8x8p32to8
        - Supernode_8x8p8to8
        - Supernode_8x8p32to8
        - DepthwiseSupernode_8x8p8to8
        - DepthwiseSupernode_8x8p32to8
    - The second parameter to QuantizedMul_8x8to32 is a scalar constant
