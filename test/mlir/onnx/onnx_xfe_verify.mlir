// RUN: onnx-mlir-opt %s -split-input-file -verify-diagnostics

//===----------------------------------------------------------------------===//
/// XFE MatMulBias - verify only checks dyn_cast
//===----------------------------------------------------------------------===//

// Corner: basic 2D matmul
func.func @matmul_2d(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>, %c: tensor<16xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEMatMulBias"(%a, %b, %c) : (tensor<4x8xf32>, tensor<8x16xf32>, tensor<16xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: batched 3D matmul
func.func @matmul_3d(%a: tensor<2x4x8xf32>, %b: tensor<2x8x16xf32>, %c: tensor<16xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEMatMulBias"(%a, %b, %c) : (tensor<2x4x8xf32>, tensor<2x8x16xf32>, tensor<16xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: fully dynamic dimensions
func.func @matmul_dynamic(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>, %c: tensor<?xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEMatMulBias"(%a, %b, %c) : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// XFE Conv - rank >= 3, X.rank == W.rank, unranked bypass
//===----------------------------------------------------------------------===//

// Corner: minimum valid rank = 3 (1D spatial conv)
func.func @conv_min_rank_3d(%x: tensor<2x128x16xf32>, %w: tensor<32x5x16xf32>, %b: tensor<32xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEConv"(%x, %w, %b) {strides = [1], pads = [0, 0], dilations = [1]} : (tensor<2x128x16xf32>, tensor<32x5x16xf32>, tensor<32xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: rank 4 (2D spatial) standard case
func.func @conv_rank4(%x: tensor<1x28x28x3xf32>, %w: tensor<64x3x3x3xf32>, %b: tensor<64xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEConv"(%x, %w, %b) {strides = [1, 1], pads = [1, 1, 1, 1], dilations = [1, 1]} : (tensor<1x28x28x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: rank 5 (3D spatial conv)
func.func @conv_rank5(%x: tensor<1x8x16x16x3xf32>, %w: tensor<64x3x3x3x3xf32>, %b: tensor<64xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEConv"(%x, %w, %b) {strides = [1, 1, 1], pads = [0, 0, 0, 0, 0, 0], dilations = [1, 1, 1]} : (tensor<1x8x16x16x3xf32>, tensor<64x3x3x3x3xf32>, tensor<64xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: bias is None (Optional operand)
func.func @conv_no_bias(%x: tensor<1x28x28x3xf32>, %w: tensor<64x3x3x3xf32>) -> tensor<*xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.XFEConv"(%x, %w, %none) {strides = [1, 1], pads = [0, 0, 0, 0], dilations = [1, 1]} : (tensor<1x28x28x3xf32>, tensor<64x3x3x3xf32>, none) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: dynamic spatial dims (hasShapeAndRank == true, but dims are ?)
func.func @conv_dynamic_spatial(%x: tensor<1x?x?x3xf32>, %w: tensor<64x3x3x3xf32>, %b: tensor<64xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEConv"(%x, %w, %b) {strides = [1, 1], pads = [0, 0, 0, 0], dilations = [1, 1]} : (tensor<1x?x?x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: unranked X → hasShapeAndRank returns false → skip all checks
func.func @conv_unranked_x(%x: tensor<*xf32>, %w: tensor<64x3x3x3xf32>, %b: tensor<64xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEConv"(%x, %w, %b) {} : (tensor<*xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: unranked W → hasShapeAndRank returns false → skip all checks
func.func @conv_unranked_w(%x: tensor<1x28x28x3xf32>, %w: tensor<*xf32>, %b: tensor<64xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEConv"(%x, %w, %b) {} : (tensor<1x28x28x3xf32>, tensor<*xf32>, tensor<64xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: grouped convolution (group > 1)
func.func @conv_grouped(%x: tensor<1x28x28x32xf32>, %w: tensor<64x3x3x16xf32>, %b: tensor<64xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEConv"(%x, %w, %b) {group = 2 : si64, strides = [1, 1], pads = [1, 1, 1, 1], dilations = [1, 1]} : (tensor<1x28x28x32xf32>, tensor<64x3x3x16xf32>, tensor<64xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// XFE ConvTranspose - rank >= 3, X.rank == W.rank, unranked bypass
//===----------------------------------------------------------------------===//

// Corner: minimum valid rank = 3 (1D spatial)
func.func @conv_transpose_min_rank_3d(%x: tensor<2x28x16xf32>, %w: tensor<32x4x16xf32>, %b: tensor<32xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEConvTranspose"(%x, %w, %b) {strides = [2], pads = [0, 0]} : (tensor<2x28x16xf32>, tensor<32x4x16xf32>, tensor<32xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: rank 4 with output_padding
func.func @conv_transpose_rank4(%x: tensor<1x28x28x3xf32>, %w: tensor<64x3x3x3xf32>, %b: tensor<64xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEConvTranspose"(%x, %w, %b) {strides = [2, 2], pads = [1, 1, 1, 1], output_padding = [1, 1]} : (tensor<1x28x28x3xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: unranked inputs bypass
func.func @conv_transpose_unranked(%x: tensor<*xf32>, %w: tensor<*xf32>, %b: tensor<64xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEConvTranspose"(%x, %w, %b) {} : (tensor<*xf32>, tensor<*xf32>, tensor<64xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: no bias
func.func @conv_transpose_no_bias(%x: tensor<1x28x28x3xf32>, %w: tensor<64x4x4x3xf32>) -> tensor<*xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.XFEConvTranspose"(%x, %w, %none) {strides = [2, 2], pads = [1, 1, 1, 1]} : (tensor<1x28x28x3xf32>, tensor<64x4x4x3xf32>, none) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// XFE AveragePool - rank >= 3, kernel_shape size >= numSpatialDims
//===----------------------------------------------------------------------===//

// Corner: minimum rank = 3 (1D spatial), kernel_shape has exactly 1 element
func.func @avgpool_min_rank_3d(%x: tensor<2x128x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEAveragePool"(%x) {kernel_shape = [3]} : (tensor<2x128x16xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: rank 4, kernel_shape exactly 2 elements (boundary match)
func.func @avgpool_rank4_exact_kernel(%x: tensor<1x28x28x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEAveragePool"(%x) {kernel_shape = [3, 3], strides = [1, 1], pads = [1, 1, 1, 1]} : (tensor<1x28x28x3xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: rank 5 (3D spatial), kernel_shape has 3 elements
func.func @avgpool_rank5(%x: tensor<1x8x8x8x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEAveragePool"(%x) {kernel_shape = [3, 3, 3]} : (tensor<1x8x8x8x16xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: kernel_shape has MORE elements than needed (>= check passes)
func.func @avgpool_kernel_extra_elements(%x: tensor<1x28x28x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEAveragePool"(%x) {kernel_shape = [3, 3, 3]} : (tensor<1x28x28x3xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: unranked → skip all checks
func.func @avgpool_unranked(%x: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEAveragePool"(%x) {} : (tensor<*xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: dynamic spatial dimensions (rank is known, dims are ?)
func.func @avgpool_dynamic_dims(%x: tensor<1x?x?x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEAveragePool"(%x) {kernel_shape = [3, 3]} : (tensor<1x?x?x3xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: ceil_mode enabled
func.func @avgpool_ceil_mode(%x: tensor<1x28x28x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEAveragePool"(%x) {kernel_shape = [3, 3], ceil_mode = 1 : si64} : (tensor<1x28x28x3xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// XFE MaxPool - rank >= 3, kernel_shape size >= numSpatialDims
//===----------------------------------------------------------------------===//

// Corner: minimum rank = 3
func.func @maxpool_min_rank_3d(%x: tensor<2x128x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEMaxPool"(%x) {kernel_shape = [3]} : (tensor<2x128x16xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: rank 4 with dilations
func.func @maxpool_rank4_dilations(%x: tensor<1x28x28x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEMaxPool"(%x) {kernel_shape = [3, 3], dilations = [2, 2], strides = [1, 1]} : (tensor<1x28x28x3xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: unranked → skip
func.func @maxpool_unranked(%x: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEMaxPool"(%x) {} : (tensor<*xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: kernel_shape extra elements (passes >= check)
func.func @maxpool_kernel_extra(%x: tensor<1x28x28x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEMaxPool"(%x) {kernel_shape = [3, 3, 3]} : (tensor<1x28x28x3xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// XFE GlobalAveragePool - rank >= 3
//===----------------------------------------------------------------------===//

// Corner: minimum rank = 3 (1D spatial)
func.func @global_avgpool_3d(%x: tensor<2x128x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEGlobalAveragePool"(%x) : (tensor<2x128x16xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: rank 4
func.func @global_avgpool_4d(%x: tensor<1x28x28x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEGlobalAveragePool"(%x) : (tensor<1x28x28x3xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: rank 5
func.func @global_avgpool_5d(%x: tensor<1x8x8x8x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEGlobalAveragePool"(%x) : (tensor<1x8x8x8x16xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: unranked → skip
func.func @global_avgpool_unranked(%x: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEGlobalAveragePool"(%x) : (tensor<*xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: fully dynamic dims (rank known)
func.func @global_avgpool_dynamic(%x: tensor<?x?x?x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEGlobalAveragePool"(%x) : (tensor<?x?x?x?xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// XFE GlobalMaxPool - rank >= 3
//===----------------------------------------------------------------------===//

// Corner: minimum rank = 3
func.func @global_maxpool_3d(%x: tensor<2x128x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEGlobalMaxPool"(%x) : (tensor<2x128x16xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: rank 4
func.func @global_maxpool_4d(%x: tensor<1x28x28x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEGlobalMaxPool"(%x) : (tensor<1x28x28x3xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: unranked → skip
func.func @global_maxpool_unranked(%x: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEGlobalMaxPool"(%x) : (tensor<*xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// XFE InstanceNormalization - rank >= 3
//===----------------------------------------------------------------------===//

// Corner: minimum rank = 3
func.func @instnorm_3d(%x: tensor<2x128x16xf32>, %s: tensor<16xf32>, %b: tensor<16xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEInstanceNormalization"(%x, %s, %b) {} : (tensor<2x128x16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: rank 4
func.func @instnorm_4d(%x: tensor<1x28x28x3xf32>, %s: tensor<3xf32>, %b: tensor<3xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEInstanceNormalization"(%x, %s, %b) {} : (tensor<1x28x28x3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: rank 5
func.func @instnorm_5d(%x: tensor<1x8x8x8x16xf32>, %s: tensor<16xf32>, %b: tensor<16xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEInstanceNormalization"(%x, %s, %b) {} : (tensor<1x8x8x8x16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: unranked → skip
func.func @instnorm_unranked(%x: tensor<*xf32>, %s: tensor<3xf32>, %b: tensor<3xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEInstanceNormalization"(%x, %s, %b) {} : (tensor<*xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: custom epsilon
func.func @instnorm_custom_eps(%x: tensor<1x28x28x3xf32>, %s: tensor<3xf32>, %b: tensor<3xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEInstanceNormalization"(%x, %s, %b) {epsilon = 1.0e-03 : f32} : (tensor<1x28x28x3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// XFE DepthToSpace - 4D, blocksize > 0, C % blocksize^2 == 0
//===----------------------------------------------------------------------===//

// Corner: blocksize = 1 → C % 1 == 0 always passes
func.func @d2s_blocksize_1(%x: tensor<1x4x4x7xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEDepthToSpace"(%x) {blocksize = 1 : si64} : (tensor<1x4x4x7xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: C exactly equals blocksize^2 (C=4, blocksize=2, 4%4=0)
func.func @d2s_c_equals_bsq(%x: tensor<1x4x4x4xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEDepthToSpace"(%x) {blocksize = 2 : si64} : (tensor<1x4x4x4xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: C is multiple of blocksize^2 (C=18, blocksize=3, 18%9=0)
func.func @d2s_c_multiple(%x: tensor<1x4x4x18xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEDepthToSpace"(%x) {blocksize = 3 : si64} : (tensor<1x4x4x18xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: large blocksize (blocksize=4, C=16, 16%16=0)
func.func @d2s_large_blocksize(%x: tensor<1x2x2x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEDepthToSpace"(%x) {blocksize = 4 : si64} : (tensor<1x2x2x16xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: dynamic C → divisibility check skipped (C == kDynamic)
func.func @d2s_dynamic_channels(%x: tensor<1x4x4x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEDepthToSpace"(%x) {blocksize = 2 : si64} : (tensor<1x4x4x?xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: dynamic spatial dims (rank known, H/W are ?)
func.func @d2s_dynamic_spatial(%x: tensor<1x?x?x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEDepthToSpace"(%x) {blocksize = 2 : si64} : (tensor<1x?x?x16xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: unranked → skip all checks
func.func @d2s_unranked(%x: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEDepthToSpace"(%x) {blocksize = 2 : si64} : (tensor<*xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: mode = "CRD" (non-default mode string)
func.func @d2s_crd_mode(%x: tensor<1x4x4x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEDepthToSpace"(%x) {blocksize = 2 : si64, mode = "CRD"} : (tensor<1x4x4x16xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: fully dynamic (batch + spatial + channels all ?)
func.func @d2s_fully_dynamic(%x: tensor<?x?x?x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFEDepthToSpace"(%x) {blocksize = 2 : si64} : (tensor<?x?x?x?xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// XFE SpaceToDepth - 4D, blocksize > 0, H%bs==0, W%bs==0
//===----------------------------------------------------------------------===//

// Corner: blocksize = 1 → any H,W always divisible
func.func @s2d_blocksize_1(%x: tensor<1x7x13x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFESpaceToDepth"(%x) {blocksize = 1 : si64} : (tensor<1x7x13x3xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: H=blocksize, W=blocksize (smallest divisible, H=2 W=2 bs=2)
func.func @s2d_h_w_equals_bs(%x: tensor<1x2x2x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFESpaceToDepth"(%x) {blocksize = 2 : si64} : (tensor<1x2x2x3xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: large blocksize (H=12 W=12 bs=4, 12%4=0)
func.func @s2d_large_blocksize(%x: tensor<1x12x12x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFESpaceToDepth"(%x) {blocksize = 4 : si64} : (tensor<1x12x12x3xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: H=blocksize*1 but W=blocksize*big (H=3 W=99 bs=3)
func.func @s2d_asymmetric(%x: tensor<1x3x99x8xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFESpaceToDepth"(%x) {blocksize = 3 : si64} : (tensor<1x3x99x8xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: dynamic H → H divisibility skipped
func.func @s2d_dynamic_h(%x: tensor<1x?x8x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFESpaceToDepth"(%x) {blocksize = 2 : si64} : (tensor<1x?x8x3xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: dynamic W → W divisibility skipped
func.func @s2d_dynamic_w(%x: tensor<1x8x?x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFESpaceToDepth"(%x) {blocksize = 2 : si64} : (tensor<1x8x?x3xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: both H and W dynamic → both checks skipped
func.func @s2d_dynamic_hw(%x: tensor<1x?x?x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFESpaceToDepth"(%x) {blocksize = 5 : si64} : (tensor<1x?x?x3xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: unranked → skip all
func.func @s2d_unranked(%x: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFESpaceToDepth"(%x) {blocksize = 2 : si64} : (tensor<*xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: fully dynamic 4D
func.func @s2d_fully_dynamic(%x: tensor<?x?x?x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.XFESpaceToDepth"(%x) {blocksize = 3 : si64} : (tensor<?x?x?x?xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// XFE Resize - scales/sizes matching rank, axes bypass, absent bypass
//===----------------------------------------------------------------------===//

// Corner: constant scales with correct length (4 elements for 4D)
func.func @resize_scales_correct(%x: tensor<1x4x4x3xf32>) -> tensor<*xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = "onnx.Constant"() {value = dense<[1.0, 2.0, 2.0, 1.0]> : tensor<4xf32>} : () -> tensor<4xf32>
  %0 = "onnx.XFEResize"(%x, %none, %scales, %none) {
    coordinate_transformation_mode = "half_pixel",
    mode = "nearest"
  } : (tensor<1x4x4x3xf32>, none, tensor<4xf32>, none) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: constant sizes with correct length (4 elements for 4D)
func.func @resize_sizes_correct(%x: tensor<1x4x4x3xf32>) -> tensor<*xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %sizes = "onnx.Constant"() {value = dense<[1, 8, 8, 3]> : tensor<4xi64>} : () -> tensor<4xi64>
  %0 = "onnx.XFEResize"(%x, %none, %none, %sizes) {
    coordinate_transformation_mode = "half_pixel",
    mode = "linear"
  } : (tensor<1x4x4x3xf32>, none, none, tensor<4xi64>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: non-constant (dynamic) scales → getDefiningOp returns null → skip
func.func @resize_dynamic_scales(%x: tensor<1x4x4x3xf32>, %scales: tensor<4xf32>) -> tensor<*xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.XFEResize"(%x, %none, %scales, %none) {
    coordinate_transformation_mode = "half_pixel",
    mode = "nearest"
  } : (tensor<1x4x4x3xf32>, none, tensor<4xf32>, none) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: non-constant (dynamic) sizes → getDefiningOp returns null → skip
func.func @resize_dynamic_sizes(%x: tensor<1x4x4x3xf32>, %sizes: tensor<4xi64>) -> tensor<*xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.XFEResize"(%x, %none, %none, %sizes) {
    coordinate_transformation_mode = "half_pixel",
    mode = "nearest"
  } : (tensor<1x4x4x3xf32>, none, none, tensor<4xi64>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: both scales and sizes absent → skip all length checks
func.func @resize_both_absent(%x: tensor<1x4x4x3xf32>) -> tensor<*xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.XFEResize"(%x, %none, %none, %none) {
    coordinate_transformation_mode = "half_pixel",
    mode = "nearest"
  } : (tensor<1x4x4x3xf32>, none, none, none) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: axes attribute present → entire verify skips extra checks
func.func @resize_with_axes(%x: tensor<1x4x4x3xf32>) -> tensor<*xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = "onnx.Constant"() {value = dense<[2.0, 2.0]> : tensor<2xf32>} : () -> tensor<2xf32>
  %0 = "onnx.XFEResize"(%x, %none, %scales, %none) {
    axes = [1, 2],
    coordinate_transformation_mode = "half_pixel",
    mode = "nearest"
  } : (tensor<1x4x4x3xf32>, none, tensor<2xf32>, none) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: unranked X → skip all
func.func @resize_unranked(%x: tensor<*xf32>) -> tensor<*xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = "onnx.Constant"() {value = dense<[1.0, 2.0, 2.0]> : tensor<3xf32>} : () -> tensor<3xf32>
  %0 = "onnx.XFEResize"(%x, %none, %scales, %none) {
    coordinate_transformation_mode = "half_pixel",
    mode = "nearest"
  } : (tensor<*xf32>, none, tensor<3xf32>, none) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: 3D resize with correct 3-element scales
func.func @resize_3d(%x: tensor<2x128x16xf32>) -> tensor<*xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = "onnx.Constant"() {value = dense<[1.0, 2.0, 1.0]> : tensor<3xf32>} : () -> tensor<3xf32>
  %0 = "onnx.XFEResize"(%x, %none, %scales, %none) {
    coordinate_transformation_mode = "asymmetric",
    mode = "nearest"
  } : (tensor<2x128x16xf32>, none, tensor<3xf32>, none) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: linear interpolation mode
func.func @resize_linear(%x: tensor<1x4x4x3xf32>) -> tensor<*xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = "onnx.Constant"() {value = dense<[1.0, 2.0, 2.0, 1.0]> : tensor<4xf32>} : () -> tensor<4xf32>
  %0 = "onnx.XFEResize"(%x, %none, %scales, %none) {
    coordinate_transformation_mode = "align_corners",
    mode = "linear"
  } : (tensor<1x4x4x3xf32>, none, tensor<4xf32>, none) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

// Corner: downsample (fractional scales)
func.func @resize_downsample(%x: tensor<1x32x32x64xf32>) -> tensor<*xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = "onnx.Constant"() {value = dense<[1.0, 0.5, 0.5, 1.0]> : tensor<4xf32>} : () -> tensor<4xf32>
  %0 = "onnx.XFEResize"(%x, %none, %scales, %none) {
    coordinate_transformation_mode = "asymmetric",
    mode = "nearest",
    nearest_mode = "floor"
  } : (tensor<1x32x32x64xf32>, none, tensor<4xf32>, none) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Channel-wise (per-axis) quantization axis verification tests for XFE ops.
//
// XFE ops use channel-last layout:
//   Data tensors (X / input): [N, spatial..., C]  => channel axis = rank - 1
//   Weight tensors (W):       [C_out, spatial..., C_in/group] => axis 0
//
// Tests cover:
//   POSITIVE: correct axis (should pass)
//   NEGATIVE: wrong axis (expected-error)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// POSITIVE: Conv with correct per-axis on both X (axis=3) and W (axis=0)
//===----------------------------------------------------------------------===//

func.func @conv_peraxis_correct(%x: tensor<1x4x4x3x!quant.uniform<i8:f32:3, {0.1, 0.2, 0.3}>>,
                                %w: tensor<4x3x3x3x!quant.uniform<i8:f32:0, {0.1, 0.2, 0.3, 0.4}>>,
                                %b: tensor<4xi32>) -> tensor<*x!quant.uniform<i8:f32, 0.1>> {
  %0 = "onnx.XFEConv"(%x, %w, %b) {strides = [1, 1], pads = [1, 1, 1, 1], dilations = [1, 1]} :
    (tensor<1x4x4x3x!quant.uniform<i8:f32:3, {0.1, 0.2, 0.3}>>,
     tensor<4x3x3x3x!quant.uniform<i8:f32:0, {0.1, 0.2, 0.3, 0.4}>>,
     tensor<4xi32>) -> tensor<*x!quant.uniform<i8:f32, 0.1>>
  onnx.Return %0 : tensor<*x!quant.uniform<i8:f32, 0.1>>
}

// -----

//===----------------------------------------------------------------------===//
// POSITIVE: Conv with per-tensor quantization (no axis to check)
//===----------------------------------------------------------------------===//

func.func @conv_pertensor_pass(%x: tensor<1x4x4x3x!quant.uniform<i8:f32, 0.1>>,
                               %w: tensor<4x3x3x3x!quant.uniform<i8:f32, 0.2>>,
                               %b: tensor<4xi32>) -> tensor<*x!quant.uniform<i8:f32, 0.1>> {
  %0 = "onnx.XFEConv"(%x, %w, %b) {strides = [1, 1], pads = [0, 0, 0, 0], dilations = [1, 1]} :
    (tensor<1x4x4x3x!quant.uniform<i8:f32, 0.1>>,
     tensor<4x3x3x3x!quant.uniform<i8:f32, 0.2>>,
     tensor<4xi32>) -> tensor<*x!quant.uniform<i8:f32, 0.1>>
  onnx.Return %0 : tensor<*x!quant.uniform<i8:f32, 0.1>>
}

// -----

//===----------------------------------------------------------------------===//
// POSITIVE: Conv with per-axis X only, W is per-tensor
//===----------------------------------------------------------------------===//

func.func @conv_peraxis_x_only(%x: tensor<1x4x4x3x!quant.uniform<i8:f32:3, {0.1, 0.2, 0.3}>>,
                               %w: tensor<4x3x3x3x!quant.uniform<i8:f32, 0.2>>,
                               %b: tensor<4xi32>) -> tensor<*x!quant.uniform<i8:f32, 0.1>> {
  %0 = "onnx.XFEConv"(%x, %w, %b) {strides = [1, 1], pads = [0, 0, 0, 0], dilations = [1, 1]} :
    (tensor<1x4x4x3x!quant.uniform<i8:f32:3, {0.1, 0.2, 0.3}>>,
     tensor<4x3x3x3x!quant.uniform<i8:f32, 0.2>>,
     tensor<4xi32>) -> tensor<*x!quant.uniform<i8:f32, 0.1>>
  onnx.Return %0 : tensor<*x!quant.uniform<i8:f32, 0.1>>
}

// -----

//===----------------------------------------------------------------------===//
// POSITIVE: Conv with per-axis W only (axis=0), X is per-tensor
//===----------------------------------------------------------------------===//

func.func @conv_peraxis_w_only(%x: tensor<1x4x4x3x!quant.uniform<i8:f32, 0.1>>,
                               %w: tensor<4x3x3x3x!quant.uniform<i8:f32:0, {0.1, 0.2, 0.3, 0.4}>>,
                               %b: tensor<4xi32>) -> tensor<*x!quant.uniform<i8:f32, 0.1>> {
  %0 = "onnx.XFEConv"(%x, %w, %b) {strides = [1, 1], pads = [0, 0, 0, 0], dilations = [1, 1]} :
    (tensor<1x4x4x3x!quant.uniform<i8:f32, 0.1>>,
     tensor<4x3x3x3x!quant.uniform<i8:f32:0, {0.1, 0.2, 0.3, 0.4}>>,
     tensor<4xi32>) -> tensor<*x!quant.uniform<i8:f32, 0.1>>
  onnx.Return %0 : tensor<*x!quant.uniform<i8:f32, 0.1>>
}

// -----

//===----------------------------------------------------------------------===//
// NEGATIVE: Conv with wrong per-axis on X (axis=1 instead of axis=3)
//===----------------------------------------------------------------------===//

func.func @conv_peraxis_x_wrong(%x: tensor<1x4x4x3x!quant.uniform<i8:f32:1, {0.1, 0.2, 0.3, 0.4}>>,
                                %w: tensor<4x3x3x3x!quant.uniform<i8:f32, 0.2>>,
                                %b: tensor<4xi32>) -> tensor<*x!quant.uniform<i8:f32, 0.1>> {
  // expected-error @+1 {{input X per-axis quantization axis is 1, but channel-last layout requires axis 3}}
  %0 = "onnx.XFEConv"(%x, %w, %b) {strides = [1, 1], pads = [0, 0, 0, 0], dilations = [1, 1]} :
    (tensor<1x4x4x3x!quant.uniform<i8:f32:1, {0.1, 0.2, 0.3, 0.4}>>,
     tensor<4x3x3x3x!quant.uniform<i8:f32, 0.2>>,
     tensor<4xi32>) -> tensor<*x!quant.uniform<i8:f32, 0.1>>
  onnx.Return %0 : tensor<*x!quant.uniform<i8:f32, 0.1>>
}

// -----

//===----------------------------------------------------------------------===//
// NEGATIVE: Conv with wrong per-axis on W (axis=1 instead of axis=0)
//===----------------------------------------------------------------------===//

func.func @conv_peraxis_w_wrong(%x: tensor<1x4x4x3x!quant.uniform<i8:f32, 0.1>>,
                                %w: tensor<4x3x3x3x!quant.uniform<i8:f32:1, {0.1, 0.2, 0.3}>>,
                                %b: tensor<4xi32>) -> tensor<*x!quant.uniform<i8:f32, 0.1>> {
  // expected-error @+1 {{weight W per-axis quantization axis is 1, but channel-last layout requires axis 0}}
  %0 = "onnx.XFEConv"(%x, %w, %b) {strides = [1, 1], pads = [0, 0, 0, 0], dilations = [1, 1]} :
    (tensor<1x4x4x3x!quant.uniform<i8:f32, 0.1>>,
     tensor<4x3x3x3x!quant.uniform<i8:f32:1, {0.1, 0.2, 0.3}>>,
     tensor<4xi32>) -> tensor<*x!quant.uniform<i8:f32, 0.1>>
  onnx.Return %0 : tensor<*x!quant.uniform<i8:f32, 0.1>>
}

// -----

//===----------------------------------------------------------------------===//
// POSITIVE: ConvTranspose with correct per-axis on X (axis=3) and W (axis=0)
//===----------------------------------------------------------------------===//

func.func @convtranspose_peraxis_correct(%x: tensor<1x4x4x3x!quant.uniform<i8:f32:3, {0.1, 0.2, 0.3}>>,
                                         %w: tensor<4x3x3x3x!quant.uniform<i8:f32:0, {0.1, 0.2, 0.3, 0.4}>>,
                                         %b: tensor<4xi32>) -> tensor<*x!quant.uniform<i8:f32, 0.1>> {
  %0 = "onnx.XFEConvTranspose"(%x, %w, %b) {strides = [2, 2], pads = [1, 1, 1, 1]} :
    (tensor<1x4x4x3x!quant.uniform<i8:f32:3, {0.1, 0.2, 0.3}>>,
     tensor<4x3x3x3x!quant.uniform<i8:f32:0, {0.1, 0.2, 0.3, 0.4}>>,
     tensor<4xi32>) -> tensor<*x!quant.uniform<i8:f32, 0.1>>
  onnx.Return %0 : tensor<*x!quant.uniform<i8:f32, 0.1>>
}

// -----

//===----------------------------------------------------------------------===//
// NEGATIVE: ConvTranspose with wrong per-axis on X (axis=0 instead of axis=3)
//===----------------------------------------------------------------------===//

func.func @convtranspose_peraxis_x_wrong(%x: tensor<1x4x4x3x!quant.uniform<i8:f32:0, {0.1}>>,
                                         %w: tensor<4x3x3x3x!quant.uniform<i8:f32, 0.2>>,
                                         %b: tensor<4xi32>) -> tensor<*x!quant.uniform<i8:f32, 0.1>> {
  // expected-error @+1 {{input X per-axis quantization axis is 0, but channel-last layout requires axis 3}}
  %0 = "onnx.XFEConvTranspose"(%x, %w, %b) {strides = [2, 2], pads = [1, 1, 1, 1]} :
    (tensor<1x4x4x3x!quant.uniform<i8:f32:0, {0.1}>>,
     tensor<4x3x3x3x!quant.uniform<i8:f32, 0.2>>,
     tensor<4xi32>) -> tensor<*x!quant.uniform<i8:f32, 0.1>>
  onnx.Return %0 : tensor<*x!quant.uniform<i8:f32, 0.1>>
}

// -----

//===----------------------------------------------------------------------===//
// NEGATIVE: ConvTranspose with wrong per-axis on W (axis=2 instead of axis=0)
//===----------------------------------------------------------------------===//

func.func @convtranspose_peraxis_w_wrong(%x: tensor<1x4x4x3x!quant.uniform<i8:f32, 0.1>>,
                                         %w: tensor<4x3x3x3x!quant.uniform<i8:f32:2, {0.1, 0.2, 0.3}>>,
                                         %b: tensor<4xi32>) -> tensor<*x!quant.uniform<i8:f32, 0.1>> {
  // expected-error @+1 {{weight W per-axis quantization axis is 2, but channel-last layout requires axis 0}}
  %0 = "onnx.XFEConvTranspose"(%x, %w, %b) {strides = [2, 2], pads = [1, 1, 1, 1]} :
    (tensor<1x4x4x3x!quant.uniform<i8:f32, 0.1>>,
     tensor<4x3x3x3x!quant.uniform<i8:f32:2, {0.1, 0.2, 0.3}>>,
     tensor<4xi32>) -> tensor<*x!quant.uniform<i8:f32, 0.1>>
  onnx.Return %0 : tensor<*x!quant.uniform<i8:f32, 0.1>>
}

// -----

//===----------------------------------------------------------------------===//
// POSITIVE: AveragePool with correct per-axis on X (axis=3)
//===----------------------------------------------------------------------===//

func.func @avgpool_peraxis_correct(%x: tensor<1x4x4x3x!quant.uniform<i8:f32:3, {0.1, 0.2, 0.3}>>) -> tensor<*x!quant.uniform<i8:f32, 0.1>> {
  %0 = "onnx.XFEAveragePool"(%x) {kernel_shape = [3, 3], strides = [1, 1], pads = [1, 1, 1, 1]} :
    (tensor<1x4x4x3x!quant.uniform<i8:f32:3, {0.1, 0.2, 0.3}>>) -> tensor<*x!quant.uniform<i8:f32, 0.1>>
  onnx.Return %0 : tensor<*x!quant.uniform<i8:f32, 0.1>>
}

// -----

//===----------------------------------------------------------------------===//
// NEGATIVE: AveragePool with wrong per-axis on X (axis=1 instead of axis=3)
//===----------------------------------------------------------------------===//

func.func @avgpool_peraxis_x_wrong(%x: tensor<1x4x4x3x!quant.uniform<i8:f32:1, {0.1, 0.2, 0.3, 0.4}>>) -> tensor<*x!quant.uniform<i8:f32, 0.1>> {
  // expected-error @+1 {{input X per-axis quantization axis is 1, but channel-last layout requires axis 3}}
  %0 = "onnx.XFEAveragePool"(%x) {kernel_shape = [3, 3], strides = [1, 1], pads = [1, 1, 1, 1]} :
    (tensor<1x4x4x3x!quant.uniform<i8:f32:1, {0.1, 0.2, 0.3, 0.4}>>) -> tensor<*x!quant.uniform<i8:f32, 0.1>>
  onnx.Return %0 : tensor<*x!quant.uniform<i8:f32, 0.1>>
}

// -----

//===----------------------------------------------------------------------===//
// POSITIVE: MaxPool with correct per-axis on X (axis=3)
//===----------------------------------------------------------------------===//

func.func @maxpool_peraxis_correct(%x: tensor<1x4x4x3x!quant.uniform<i8:f32:3, {0.1, 0.2, 0.3}>>) -> tensor<*x!quant.uniform<i8:f32, 0.1>> {
  %0 = "onnx.XFEMaxPool"(%x) {kernel_shape = [3, 3], strides = [1, 1]} :
    (tensor<1x4x4x3x!quant.uniform<i8:f32:3, {0.1, 0.2, 0.3}>>) -> tensor<*x!quant.uniform<i8:f32, 0.1>>
  onnx.Return %0 : tensor<*x!quant.uniform<i8:f32, 0.1>>
}

// -----

//===----------------------------------------------------------------------===//
// NEGATIVE: MaxPool with wrong per-axis on X (axis=2 instead of axis=3)
//===----------------------------------------------------------------------===//

func.func @maxpool_peraxis_x_wrong(%x: tensor<1x4x4x3x!quant.uniform<i8:f32:2, {0.1, 0.2, 0.3, 0.4}>>) -> tensor<*x!quant.uniform<i8:f32, 0.1>> {
  // expected-error @+1 {{input X per-axis quantization axis is 2, but channel-last layout requires axis 3}}
  %0 = "onnx.XFEMaxPool"(%x) {kernel_shape = [3, 3], strides = [1, 1]} :
    (tensor<1x4x4x3x!quant.uniform<i8:f32:2, {0.1, 0.2, 0.3, 0.4}>>) -> tensor<*x!quant.uniform<i8:f32, 0.1>>
  onnx.Return %0 : tensor<*x!quant.uniform<i8:f32, 0.1>>
}

// -----

//===----------------------------------------------------------------------===//
// POSITIVE: GlobalAveragePool with correct per-axis on X (axis=3)
//===----------------------------------------------------------------------===//

func.func @global_avgpool_peraxis_correct(%x: tensor<1x4x4x3x!quant.uniform<i8:f32:3, {0.1, 0.2, 0.3}>>) -> tensor<*x!quant.uniform<i8:f32, 0.1>> {
  %0 = "onnx.XFEGlobalAveragePool"(%x) :
    (tensor<1x4x4x3x!quant.uniform<i8:f32:3, {0.1, 0.2, 0.3}>>) -> tensor<*x!quant.uniform<i8:f32, 0.1>>
  onnx.Return %0 : tensor<*x!quant.uniform<i8:f32, 0.1>>
}

// -----

//===----------------------------------------------------------------------===//
// NEGATIVE: GlobalAveragePool with wrong per-axis on X (axis=0 instead of 3)
//===----------------------------------------------------------------------===//

func.func @global_avgpool_peraxis_x_wrong(%x: tensor<1x4x4x3x!quant.uniform<i8:f32:0, {0.1}>>) -> tensor<*x!quant.uniform<i8:f32, 0.1>> {
  // expected-error @+1 {{input X per-axis quantization axis is 0, but channel-last layout requires axis 3}}
  %0 = "onnx.XFEGlobalAveragePool"(%x) :
    (tensor<1x4x4x3x!quant.uniform<i8:f32:0, {0.1}>>) -> tensor<*x!quant.uniform<i8:f32, 0.1>>
  onnx.Return %0 : tensor<*x!quant.uniform<i8:f32, 0.1>>
}

// -----

//===----------------------------------------------------------------------===//
// POSITIVE: GlobalMaxPool with correct per-axis on X (axis=3)
//===----------------------------------------------------------------------===//

func.func @global_maxpool_peraxis_correct(%x: tensor<1x4x4x3x!quant.uniform<i8:f32:3, {0.1, 0.2, 0.3}>>) -> tensor<*x!quant.uniform<i8:f32, 0.1>> {
  %0 = "onnx.XFEGlobalMaxPool"(%x) :
    (tensor<1x4x4x3x!quant.uniform<i8:f32:3, {0.1, 0.2, 0.3}>>) -> tensor<*x!quant.uniform<i8:f32, 0.1>>
  onnx.Return %0 : tensor<*x!quant.uniform<i8:f32, 0.1>>
}

// -----

//===----------------------------------------------------------------------===//
// NEGATIVE: GlobalMaxPool with wrong per-axis on X (axis=1 instead of 3)
//===----------------------------------------------------------------------===//

func.func @global_maxpool_peraxis_x_wrong(%x: tensor<1x4x4x3x!quant.uniform<i8:f32:1, {0.1, 0.2, 0.3, 0.4}>>) -> tensor<*x!quant.uniform<i8:f32, 0.1>> {
  // expected-error @+1 {{input X per-axis quantization axis is 1, but channel-last layout requires axis 3}}
  %0 = "onnx.XFEGlobalMaxPool"(%x) :
    (tensor<1x4x4x3x!quant.uniform<i8:f32:1, {0.1, 0.2, 0.3, 0.4}>>) -> tensor<*x!quant.uniform<i8:f32, 0.1>>
  onnx.Return %0 : tensor<*x!quant.uniform<i8:f32, 0.1>>
}

// -----

//===----------------------------------------------------------------------===//
// POSITIVE: InstanceNormalization with correct per-axis on input (axis=3)
//===----------------------------------------------------------------------===//

func.func @instnorm_peraxis_correct(%x: tensor<1x4x4x3x!quant.uniform<i8:f32:3, {0.1, 0.2, 0.3}>>,
                                    %s: tensor<3xf32>, %b: tensor<3xf32>) -> tensor<*x!quant.uniform<i8:f32, 0.1>> {
  %0 = "onnx.XFEInstanceNormalization"(%x, %s, %b) {} :
    (tensor<1x4x4x3x!quant.uniform<i8:f32:3, {0.1, 0.2, 0.3}>>,
     tensor<3xf32>, tensor<3xf32>) -> tensor<*x!quant.uniform<i8:f32, 0.1>>
  onnx.Return %0 : tensor<*x!quant.uniform<i8:f32, 0.1>>
}

// -----

//===----------------------------------------------------------------------===//
// NEGATIVE: InstanceNormalization with wrong per-axis on input (axis=0)
//===----------------------------------------------------------------------===//

func.func @instnorm_peraxis_wrong(%x: tensor<1x4x4x3x!quant.uniform<i8:f32:0, {0.1}>>,
                                  %s: tensor<3xf32>, %b: tensor<3xf32>) -> tensor<*x!quant.uniform<i8:f32, 0.1>> {
  // expected-error @+1 {{input per-axis quantization axis is 0, but channel-last layout requires axis 3}}
  %0 = "onnx.XFEInstanceNormalization"(%x, %s, %b) {} :
    (tensor<1x4x4x3x!quant.uniform<i8:f32:0, {0.1}>>,
     tensor<3xf32>, tensor<3xf32>) -> tensor<*x!quant.uniform<i8:f32, 0.1>>
  onnx.Return %0 : tensor<*x!quant.uniform<i8:f32, 0.1>>
}

// -----

//===----------------------------------------------------------------------===//
// POSITIVE: DepthToSpace with correct per-axis on input (axis=3)
//===----------------------------------------------------------------------===//

func.func @d2s_peraxis_correct(%x: tensor<1x4x4x4x!quant.uniform<i8:f32:3, {0.1, 0.2, 0.3, 0.4}>>) -> tensor<*x!quant.uniform<i8:f32, 0.1>> {
  %0 = "onnx.XFEDepthToSpace"(%x) {blocksize = 2 : si64} :
    (tensor<1x4x4x4x!quant.uniform<i8:f32:3, {0.1, 0.2, 0.3, 0.4}>>) -> tensor<*x!quant.uniform<i8:f32, 0.1>>
  onnx.Return %0 : tensor<*x!quant.uniform<i8:f32, 0.1>>
}

// -----

//===----------------------------------------------------------------------===//
// NEGATIVE: DepthToSpace with wrong per-axis on input (axis=1 instead of 3)
//===----------------------------------------------------------------------===//

func.func @d2s_peraxis_wrong(%x: tensor<1x4x4x4x!quant.uniform<i8:f32:1, {0.1, 0.2, 0.3, 0.4}>>) -> tensor<*x!quant.uniform<i8:f32, 0.1>> {
  // expected-error @+1 {{input per-axis quantization axis is 1, but channel-last layout requires axis 3}}
  %0 = "onnx.XFEDepthToSpace"(%x) {blocksize = 2 : si64} :
    (tensor<1x4x4x4x!quant.uniform<i8:f32:1, {0.1, 0.2, 0.3, 0.4}>>) -> tensor<*x!quant.uniform<i8:f32, 0.1>>
  onnx.Return %0 : tensor<*x!quant.uniform<i8:f32, 0.1>>
}

// -----

//===----------------------------------------------------------------------===//
// POSITIVE: SpaceToDepth with correct per-axis on input (axis=3)
//===----------------------------------------------------------------------===//

func.func @s2d_peraxis_correct(%x: tensor<1x4x4x3x!quant.uniform<i8:f32:3, {0.1, 0.2, 0.3}>>) -> tensor<*x!quant.uniform<i8:f32, 0.1>> {
  %0 = "onnx.XFESpaceToDepth"(%x) {blocksize = 2 : si64} :
    (tensor<1x4x4x3x!quant.uniform<i8:f32:3, {0.1, 0.2, 0.3}>>) -> tensor<*x!quant.uniform<i8:f32, 0.1>>
  onnx.Return %0 : tensor<*x!quant.uniform<i8:f32, 0.1>>
}

// -----

//===----------------------------------------------------------------------===//
// NEGATIVE: SpaceToDepth with wrong per-axis on input (axis=2 instead of 3)
//===----------------------------------------------------------------------===//

func.func @s2d_peraxis_wrong(%x: tensor<1x4x4x3x!quant.uniform<i8:f32:2, {0.1, 0.2, 0.3, 0.4}>>) -> tensor<*x!quant.uniform<i8:f32, 0.1>> {
  // expected-error @+1 {{input per-axis quantization axis is 2, but channel-last layout requires axis 3}}
  %0 = "onnx.XFESpaceToDepth"(%x) {blocksize = 2 : si64} :
    (tensor<1x4x4x3x!quant.uniform<i8:f32:2, {0.1, 0.2, 0.3, 0.4}>>) -> tensor<*x!quant.uniform<i8:f32, 0.1>>
  onnx.Return %0 : tensor<*x!quant.uniform<i8:f32, 0.1>>
}

// -----

//===----------------------------------------------------------------------===//
// POSITIVE: Resize with correct per-axis on X (axis=3)
//===----------------------------------------------------------------------===//

func.func @resize_peraxis_correct(%x: tensor<1x4x4x3x!quant.uniform<i8:f32:3, {0.1, 0.2, 0.3}>>) -> tensor<*x!quant.uniform<i8:f32, 0.1>> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = "onnx.Constant"() {value = dense<[1.0, 2.0, 2.0, 1.0]> : tensor<4xf32>} : () -> tensor<4xf32>
  %0 = "onnx.XFEResize"(%x, %none, %scales, %none) {
    coordinate_transformation_mode = "half_pixel",
    mode = "nearest"
  } : (tensor<1x4x4x3x!quant.uniform<i8:f32:3, {0.1, 0.2, 0.3}>>, none, tensor<4xf32>, none) -> tensor<*x!quant.uniform<i8:f32, 0.1>>
  onnx.Return %0 : tensor<*x!quant.uniform<i8:f32, 0.1>>
}

// -----

//===----------------------------------------------------------------------===//
// NEGATIVE: Resize with wrong per-axis on X (axis=1 instead of 3)
//===----------------------------------------------------------------------===//

func.func @resize_peraxis_x_wrong(%x: tensor<1x4x4x3x!quant.uniform<i8:f32:1, {0.1, 0.2, 0.3, 0.4}>>) -> tensor<*x!quant.uniform<i8:f32, 0.1>> {
  %none = "onnx.NoValue"() {value} : () -> none
  %scales = "onnx.Constant"() {value = dense<[1.0, 2.0, 2.0, 1.0]> : tensor<4xf32>} : () -> tensor<4xf32>
  // expected-error @+1 {{input X per-axis quantization axis is 1, but channel-last layout requires axis 3}}
  %0 = "onnx.XFEResize"(%x, %none, %scales, %none) {
    coordinate_transformation_mode = "half_pixel",
    mode = "nearest"
  } : (tensor<1x4x4x3x!quant.uniform<i8:f32:1, {0.1, 0.2, 0.3, 0.4}>>, none, tensor<4xf32>, none) -> tensor<*x!quant.uniform<i8:f32, 0.1>>
  onnx.Return %0 : tensor<*x!quant.uniform<i8:f32, 0.1>>
}

// -----

//===----------------------------------------------------------------------===//
// POSITIVE: MatMulBias with correct per-axis on A (axis=1) and B (axis=1)
// A=[4,8], B=[8,3], C=[3] => A channel axis=1, B channel axis=1
//===----------------------------------------------------------------------===//

func.func @matmul_peraxis_correct(%a: tensor<4x8x!quant.uniform<i8:f32:1, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}>>,
                                  %b: tensor<8x3x!quant.uniform<i8:f32:1, {0.1, 0.2, 0.3}>>,
                                  %c: tensor<3xf32>) -> tensor<*x!quant.uniform<i8:f32, 0.1>> {
  %0 = "onnx.XFEMatMulBias"(%a, %b, %c) :
    (tensor<4x8x!quant.uniform<i8:f32:1, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}>>,
     tensor<8x3x!quant.uniform<i8:f32:1, {0.1, 0.2, 0.3}>>,
     tensor<3xf32>) -> tensor<*x!quant.uniform<i8:f32, 0.1>>
  onnx.Return %0 : tensor<*x!quant.uniform<i8:f32, 0.1>>
}

// -----

//===----------------------------------------------------------------------===//
// NEGATIVE: MatMulBias with wrong per-axis on A (axis=0 instead of axis=1)
//===----------------------------------------------------------------------===//

func.func @matmul_peraxis_a_wrong(%a: tensor<4x8x!quant.uniform<i8:f32:0, {0.1, 0.2, 0.3, 0.4}>>,
                                  %b: tensor<8x3xf32>,
                                  %c: tensor<3xf32>) -> tensor<*x!quant.uniform<i8:f32, 0.1>> {
  // expected-error @+1 {{input A per-axis quantization axis is 0, but channel-last layout requires axis 1}}
  %0 = "onnx.XFEMatMulBias"(%a, %b, %c) :
    (tensor<4x8x!quant.uniform<i8:f32:0, {0.1, 0.2, 0.3, 0.4}>>,
     tensor<8x3xf32>,
     tensor<3xf32>) -> tensor<*x!quant.uniform<i8:f32, 0.1>>
  onnx.Return %0 : tensor<*x!quant.uniform<i8:f32, 0.1>>
}

// -----

//===----------------------------------------------------------------------===//
// NEGATIVE: MatMulBias with wrong per-axis on B (axis=0 instead of axis=1)
//===----------------------------------------------------------------------===//

func.func @matmul_peraxis_b_wrong(%a: tensor<4x8xf32>,
                                  %b: tensor<8x3x!quant.uniform<i8:f32:0, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}>>,
                                  %c: tensor<3xf32>) -> tensor<*x!quant.uniform<i8:f32, 0.1>> {
  // expected-error @+1 {{input B per-axis quantization axis is 0, but channel-last layout requires axis 1}}
  %0 = "onnx.XFEMatMulBias"(%a, %b, %c) :
    (tensor<4x8xf32>,
     tensor<8x3x!quant.uniform<i8:f32:0, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}>>,
     tensor<3xf32>) -> tensor<*x!quant.uniform<i8:f32, 0.1>>
  onnx.Return %0 : tensor<*x!quant.uniform<i8:f32, 0.1>>
}

// -----

//===----------------------------------------------------------------------===//
// POSITIVE: Conv 3D (rank 5) with correct per-axis on X (axis=4)
// X=[N, D, H, W, C], W=[C_out, kD, kH, kW, C_in]
//===----------------------------------------------------------------------===//

func.func @conv3d_peraxis_correct(%x: tensor<1x4x4x4x3x!quant.uniform<i8:f32:4, {0.1, 0.2, 0.3}>>,
                                  %w: tensor<4x3x3x3x3x!quant.uniform<i8:f32:0, {0.1, 0.2, 0.3, 0.4}>>,
                                  %b: tensor<4xi32>) -> tensor<*x!quant.uniform<i8:f32, 0.1>> {
  %0 = "onnx.XFEConv"(%x, %w, %b) {strides = [1, 1, 1], pads = [0, 0, 0, 0, 0, 0], dilations = [1, 1, 1]} :
    (tensor<1x4x4x4x3x!quant.uniform<i8:f32:4, {0.1, 0.2, 0.3}>>,
     tensor<4x3x3x3x3x!quant.uniform<i8:f32:0, {0.1, 0.2, 0.3, 0.4}>>,
     tensor<4xi32>) -> tensor<*x!quant.uniform<i8:f32, 0.1>>
  onnx.Return %0 : tensor<*x!quant.uniform<i8:f32, 0.1>>
}

// -----

//===----------------------------------------------------------------------===//
// POSITIVE: Conv with unranked X bypasses quant axis check
//===----------------------------------------------------------------------===//

func.func @conv_unranked_bypass(%x: tensor<*x!quant.uniform<i8:f32:1, {0.1, 0.2}>>,
                                %w: tensor<4x3x3x3x!quant.uniform<i8:f32, 0.2>>,
                                %b: tensor<4xi32>) -> tensor<*x!quant.uniform<i8:f32, 0.1>> {
  %0 = "onnx.XFEConv"(%x, %w, %b) {} :
    (tensor<*x!quant.uniform<i8:f32:1, {0.1, 0.2}>>,
     tensor<4x3x3x3x!quant.uniform<i8:f32, 0.2>>,
     tensor<4xi32>) -> tensor<*x!quant.uniform<i8:f32, 0.1>>
  onnx.Return %0 : tensor<*x!quant.uniform<i8:f32, 0.1>>
}

// -----

//===----------------------------------------------------------------------===//
// POSITIVE: GlobalAveragePool with per-tensor quant (no axis check)
//===----------------------------------------------------------------------===//

func.func @global_avgpool_pertensor(%x: tensor<1x4x4x3x!quant.uniform<i8:f32, 0.1>>) -> tensor<*x!quant.uniform<i8:f32, 0.1>> {
  %0 = "onnx.XFEGlobalAveragePool"(%x) :
    (tensor<1x4x4x3x!quant.uniform<i8:f32, 0.1>>) -> tensor<*x!quant.uniform<i8:f32, 0.1>>
  onnx.Return %0 : tensor<*x!quant.uniform<i8:f32, 0.1>>
}

// -----

//===----------------------------------------------------------------------===//
// POSITIVE: Conv with u8 per-axis quantization (correct axis)
//===----------------------------------------------------------------------===//

func.func @conv_u8_peraxis_correct(%x: tensor<1x4x4x3x!quant.uniform<u8:f32:3, {0.1:128, 0.2:128, 0.3:128}>>,
                                   %w: tensor<4x3x3x3x!quant.uniform<i8:f32:0, {0.1, 0.2, 0.3, 0.4}>>,
                                   %b: tensor<4xi32>) -> tensor<*x!quant.uniform<i8:f32, 0.1>> {
  %0 = "onnx.XFEConv"(%x, %w, %b) {strides = [1, 1], pads = [0, 0, 0, 0], dilations = [1, 1]} :
    (tensor<1x4x4x3x!quant.uniform<u8:f32:3, {0.1:128, 0.2:128, 0.3:128}>>,
     tensor<4x3x3x3x!quant.uniform<i8:f32:0, {0.1, 0.2, 0.3, 0.4}>>,
     tensor<4xi32>) -> tensor<*x!quant.uniform<i8:f32, 0.1>>
  onnx.Return %0 : tensor<*x!quant.uniform<i8:f32, 0.1>>
}