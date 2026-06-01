// Tests for RemoveQDQForConsts: removes DequantizeLinear -> ReformattingOp ->
// QuantizeLinear wrappers around constant inputs when scale/zero_point match
// but tensor types differ (shape change), enabling subsequent constant folding.

// RUN: onnx-mlir-opt --split-input-file %s -constprop-onnx=enable-qdq | FileCheck %s

//===----------------------------------------------------------------------===//
// Positive cases: QDQ removed and reformatting op folded into constant.
//===----------------------------------------------------------------------===//

func.func @transpose_qdq_removal() -> tensor<2x1xui8> {
  %a = onnx.Constant dense<[[10, 20]]> : tensor<1x2xui8>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %dq = "onnx.DequantizeLinear"(%a, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<1x2xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x2xf32>
  %t = "onnx.Transpose"(%dq) {perm = [1, 0]} : (tensor<1x2xf32>) -> tensor<2x1xf32>
  %q = "onnx.QuantizeLinear"(%t, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<2x1xf32>, tensor<f32>, tensor<ui8>) -> tensor<2x1xui8>
  return %q : tensor<2x1xui8>
}

// CHECK-LABEL: @transpose_qdq_removal
// CHECK-NOT: onnx.DequantizeLinear
// CHECK-NOT: onnx.Transpose
// CHECK-NOT: onnx.QuantizeLinear
// CHECK: onnx.Constant dense<{{\[}}[10], [20]]> : tensor<2x1xui8>

// -----

func.func @reshape_qdq_removal() -> tensor<1x4xui8> {
  %a = onnx.Constant dense<[10, 20, 30, 40]> : tensor<4xui8>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %shape = onnx.Constant dense<[1, 4]> : tensor<2xi64>
  %dq = "onnx.DequantizeLinear"(%a, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<4xui8>, tensor<f32>, tensor<ui8>) -> tensor<4xf32>
  %r = "onnx.Reshape"(%dq, %shape) {allowzero = 0 : si64} : (tensor<4xf32>, tensor<2xi64>) -> tensor<1x4xf32>
  %q = "onnx.QuantizeLinear"(%r, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x4xui8>
  return %q : tensor<1x4xui8>
}

// CHECK-LABEL: @reshape_qdq_removal
// CHECK-NOT: onnx.DequantizeLinear
// CHECK-NOT: onnx.Reshape
// CHECK-NOT: onnx.QuantizeLinear
// CHECK: onnx.Constant dense<{{\[}}[10, 20, 30, 40]]> : tensor<1x4xui8>

// -----

func.func @slice_qdq_removal() -> tensor<2xui8> {
  %a = onnx.Constant dense<[10, 20, 30, 40]> : tensor<4xui8>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %starts = onnx.Constant dense<[1]> : tensor<1xi64>
  %ends = onnx.Constant dense<[3]> : tensor<1xi64>
  %axes = onnx.Constant dense<[0]> : tensor<1xi64>
  %steps = onnx.Constant dense<[1]> : tensor<1xi64>
  %dq = "onnx.DequantizeLinear"(%a, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<4xui8>, tensor<f32>, tensor<ui8>) -> tensor<4xf32>
  %s = "onnx.Slice"(%dq, %starts, %ends, %axes, %steps) : (tensor<4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xf32>
  %q = "onnx.QuantizeLinear"(%s, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<2xf32>, tensor<f32>, tensor<ui8>) -> tensor<2xui8>
  return %q : tensor<2xui8>
}

// CHECK-LABEL: @slice_qdq_removal
// CHECK-NOT: onnx.DequantizeLinear
// CHECK-NOT: onnx.Slice
// CHECK-NOT: onnx.QuantizeLinear
// CHECK: onnx.Constant dense<[20, 30]> : tensor<2xui8>

// -----

func.func @squeeze_qdq_removal() -> tensor<3xui8> {
  %a = onnx.Constant dense<[[10, 20, 30]]> : tensor<1x3xui8>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %axes = onnx.Constant dense<[0]> : tensor<1xi64>
  %dq = "onnx.DequantizeLinear"(%a, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<1x3xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x3xf32>
  %s = "onnx.Squeeze"(%dq, %axes) : (tensor<1x3xf32>, tensor<1xi64>) -> tensor<3xf32>
  %q = "onnx.QuantizeLinear"(%s, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<3xf32>, tensor<f32>, tensor<ui8>) -> tensor<3xui8>
  return %q : tensor<3xui8>
}

// CHECK-LABEL: @squeeze_qdq_removal
// CHECK-NOT: onnx.DequantizeLinear
// CHECK-NOT: onnx.Squeeze
// CHECK-NOT: onnx.QuantizeLinear
// CHECK: onnx.Constant dense<[10, 20, 30]> : tensor<3xui8>

// -----

func.func @unsqueeze_qdq_removal() -> tensor<1x3xui8> {
  %a = onnx.Constant dense<[10, 20, 30]> : tensor<3xui8>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %axes = onnx.Constant dense<[0]> : tensor<1xi64>
  %dq = "onnx.DequantizeLinear"(%a, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<3xui8>, tensor<f32>, tensor<ui8>) -> tensor<3xf32>
  %u = "onnx.Unsqueeze"(%dq, %axes) : (tensor<3xf32>, tensor<1xi64>) -> tensor<1x3xf32>
  %q = "onnx.QuantizeLinear"(%u, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x3xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x3xui8>
  return %q : tensor<1x3xui8>
}

// CHECK-LABEL: @unsqueeze_qdq_removal
// CHECK-NOT: onnx.DequantizeLinear
// CHECK-NOT: onnx.Unsqueeze
// CHECK-NOT: onnx.QuantizeLinear
// CHECK: onnx.Constant dense<{{\[}}[10, 20, 30]]> : tensor<1x3xui8>

// -----

func.func @gather_qdq_removal() -> tensor<2xui8> {
  %a = onnx.Constant dense<[10, 20, 30, 40]> : tensor<4xui8>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %indices = onnx.Constant dense<[1, 3]> : tensor<2xi64>
  %dq = "onnx.DequantizeLinear"(%a, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<4xui8>, tensor<f32>, tensor<ui8>) -> tensor<4xf32>
  %g = "onnx.Gather"(%dq, %indices) {axis = 0 : si64} : (tensor<4xf32>, tensor<2xi64>) -> tensor<2xf32>
  %q = "onnx.QuantizeLinear"(%g, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<2xf32>, tensor<f32>, tensor<ui8>) -> tensor<2xui8>
  return %q : tensor<2xui8>
}

// CHECK-LABEL: @gather_qdq_removal
// CHECK-NOT: onnx.DequantizeLinear
// CHECK-NOT: onnx.Gather
// CHECK-NOT: onnx.QuantizeLinear
// CHECK: onnx.Constant dense<[20, 40]> : tensor<2xui8>

// -----

// Chained: const -> DQ -> Slice -> Q -> DQ -> Slice -> Q -> DQ -> Transpose -> Q -> DQ -> output
// Each QDQ pair is iteratively removed by RemoveQDQForConst, then the reformatting op is folded
func.func @chained_qdq_removal() -> tensor<3x2xf32> {
  %a = onnx.Constant dense<[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]> : tensor<3x4xui8>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>

  // Stage 1: Slice rows [0:2] on axis 0 -> 2x4
  %dq1 = "onnx.DequantizeLinear"(%a, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<3x4xui8>, tensor<f32>, tensor<ui8>) -> tensor<3x4xf32>
  %starts1 = onnx.Constant dense<[0]> : tensor<1xi64>
  %ends1 = onnx.Constant dense<[2]> : tensor<1xi64>
  %axes1 = onnx.Constant dense<[0]> : tensor<1xi64>
  %steps1 = onnx.Constant dense<[1]> : tensor<1xi64>
  %slice1 = "onnx.Slice"(%dq1, %starts1, %ends1, %axes1, %steps1) : (tensor<3x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x4xf32>
  %q1 = "onnx.QuantizeLinear"(%slice1, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<2x4xf32>, tensor<f32>, tensor<ui8>) -> tensor<2x4xui8>

  // Stage 2: Slice cols [1:4] on axis 1 -> 2x3
  %dq2 = "onnx.DequantizeLinear"(%q1, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2x4xui8>, tensor<f32>, tensor<ui8>) -> tensor<2x4xf32>
  %starts2 = onnx.Constant dense<[1]> : tensor<1xi64>
  %ends2 = onnx.Constant dense<[4]> : tensor<1xi64>
  %axes2 = onnx.Constant dense<[1]> : tensor<1xi64>
  %steps2 = onnx.Constant dense<[1]> : tensor<1xi64>
  %slice2 = "onnx.Slice"(%dq2, %starts2, %ends2, %axes2, %steps2) : (tensor<2x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2x3xf32>
  %q2 = "onnx.QuantizeLinear"(%slice2, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<2x3xf32>, tensor<f32>, tensor<ui8>) -> tensor<2x3xui8>

  // Stage 3: Transpose perm=[1,0] -> 3x2
  %dq3 = "onnx.DequantizeLinear"(%q2, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2x3xui8>, tensor<f32>, tensor<ui8>) -> tensor<2x3xf32>
  %t = "onnx.Transpose"(%dq3) {perm = [1, 0]} : (tensor<2x3xf32>) -> tensor<3x2xf32>
  %q3 = "onnx.QuantizeLinear"(%t, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<3x2xf32>, tensor<f32>, tensor<ui8>) -> tensor<3x2xui8>

  // Final DQ -> float output
  %dq4 = "onnx.DequantizeLinear"(%q3, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<3x2xui8>, tensor<f32>, tensor<ui8>) -> tensor<3x2xf32>
  return %dq4 : tensor<3x2xf32>
}

// CHECK-LABEL: @chained_qdq_removal
// CHECK-NOT: onnx.Slice
// CHECK-NOT: onnx.Transpose
// CHECK-NOT: onnx.QuantizeLinear
// CHECK: onnx.Constant dense<{{\[}}[2, 6], [3, 7], [4, 8]]> : tensor<3x2xui8>
// CHECK: onnx.DequantizeLinear
// CHECK-SAME: (tensor<3x2xui8>, tensor<f32>, tensor<ui8>) -> tensor<3x2xf32>

// -----

//===----------------------------------------------------------------------===//
// Negative cases: QDQ should NOT be removed.
//===----------------------------------------------------------------------===//

// DQ and Q have different scales.
func.func @transpose_qdq_diff_scale() -> tensor<2x1xui8> {
  %a = onnx.Constant dense<[[10, 20]]> : tensor<1x2xui8>
  %dq_scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %q_scale = onnx.Constant dense<2.500000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %dq = "onnx.DequantizeLinear"(%a, %dq_scale, %zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<1x2xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x2xf32>
  %t = "onnx.Transpose"(%dq) {perm = [1, 0]} : (tensor<1x2xf32>) -> tensor<2x1xf32>
  %q = "onnx.QuantizeLinear"(%t, %q_scale, %zp) {axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<2x1xf32>, tensor<f32>, tensor<ui8>) -> tensor<2x1xui8>
  return %q : tensor<2x1xui8>
}

// BypassShapeOpThroughDQ folds the Transpose into the constant, but the
// mismatched Q/DQ pair must remain.
// CHECK-LABEL: @transpose_qdq_diff_scale
// CHECK: onnx.Constant dense<{{\[}}[10], [20]]> : tensor<2x1xui8>
// CHECK: onnx.DequantizeLinear
// CHECK-NOT: onnx.Transpose
// CHECK: onnx.QuantizeLinear

// -----

// DQ and Q have different zero points.
func.func @transpose_qdq_diff_zp() -> tensor<2x1xui8> {
  %a = onnx.Constant dense<[[10, 20]]> : tensor<1x2xui8>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %dq_zp = onnx.Constant dense<0> : tensor<ui8>
  %q_zp = onnx.Constant dense<10> : tensor<ui8>
  %dq = "onnx.DequantizeLinear"(%a, %scale, %dq_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<1x2xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x2xf32>
  %t = "onnx.Transpose"(%dq) {perm = [1, 0]} : (tensor<1x2xf32>) -> tensor<2x1xf32>
  %q = "onnx.QuantizeLinear"(%t, %scale, %q_zp) {axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<2x1xf32>, tensor<f32>, tensor<ui8>) -> tensor<2x1xui8>
  return %q : tensor<2x1xui8>
}

// BypassShapeOpThroughDQ folds the Transpose into the constant, but the
// mismatched Q/DQ pair must remain.
// CHECK-LABEL: @transpose_qdq_diff_zp
// CHECK: onnx.Constant dense<{{\[}}[10], [20]]> : tensor<2x1xui8>
// CHECK: onnx.DequantizeLinear
// CHECK-NOT: onnx.Transpose
// CHECK: onnx.QuantizeLinear

// -----

// Input to DQ is not a constant (function argument).
func.func @transpose_qdq_non_const(%arg0: tensor<1x2xui8>) -> tensor<2x1xui8> {
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %dq = "onnx.DequantizeLinear"(%arg0, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<1x2xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x2xf32>
  %t = "onnx.Transpose"(%dq) {perm = [1, 0]} : (tensor<1x2xf32>) -> tensor<2x1xf32>
  %q = "onnx.QuantizeLinear"(%t, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<2x1xf32>, tensor<f32>, tensor<ui8>) -> tensor<2x1xui8>
  return %q : tensor<2x1xui8>
}

// CHECK-LABEL: @transpose_qdq_non_const
// CHECK: onnx.DequantizeLinear
// CHECK: onnx.Transpose
// CHECK: onnx.QuantizeLinear

// -----

// Reformatting op result has multiple uses.
func.func @transpose_qdq_multi_use() -> (tensor<2x1xui8>, tensor<2x1xf32>) {
  %a = onnx.Constant dense<[[10, 20]]> : tensor<1x2xui8>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %dq = "onnx.DequantizeLinear"(%a, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<1x2xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x2xf32>
  %t = "onnx.Transpose"(%dq) {perm = [1, 0]} : (tensor<1x2xf32>) -> tensor<2x1xf32>
  %q = "onnx.QuantizeLinear"(%t, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<2x1xf32>, tensor<f32>, tensor<ui8>) -> tensor<2x1xui8>
  return %q, %t : tensor<2x1xui8>, tensor<2x1xf32>
}

// BypassShapeOpThroughDQ pushes the Transpose into the constant for both
// users; with matching Q/DQ scales the Q result collapses back to the const.
// CHECK-LABEL: @transpose_qdq_multi_use
// CHECK: onnx.Constant dense<{{\[}}[10], [20]]> : tensor<2x1xui8>
// CHECK: onnx.DequantizeLinear
// CHECK-NOT: onnx.Transpose
// CHECK-NOT: onnx.QuantizeLinear

// -----

func.func @transpose_qdq_diff_types() -> tensor<2x1xi8> {
  %a = onnx.Constant dense<[[10, 20]]> : tensor<1x2xui8>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %dq = "onnx.DequantizeLinear"(%a, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<1x2xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x2xf32>
  %t = "onnx.Transpose"(%dq) {perm = [1, 0]} : (tensor<1x2xf32>) -> tensor<2x1xf32>
  %q = "onnx.QuantizeLinear"(%t, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<2x1xf32>, tensor<f32>, tensor<ui8>) -> tensor<2x1xi8>
  return %q : tensor<2x1xi8>
}

// BypassShapeOpThroughDQ folds the Transpose into the constant, but the
// type-changing Q/DQ pair must remain.
// CHECK-LABEL: @transpose_qdq_diff_types
// CHECK: onnx.Constant dense<{{\[}}[10], [20]]> : tensor<2x1xui8>
// CHECK: onnx.DequantizeLinear
// CHECK-NOT: onnx.Transpose
// CHECK: onnx.QuantizeLinear

// -----

// No DQ on input (operand 0 is not from DequantizeLinear).
func.func @transpose_no_dq(%arg0: tensor<1x2xf32>) -> tensor<2x1xui8> {
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %t = "onnx.Transpose"(%arg0) {perm = [1, 0]} : (tensor<1x2xf32>) -> tensor<2x1xf32>
  %q = "onnx.QuantizeLinear"(%t, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<2x1xf32>, tensor<f32>, tensor<ui8>) -> tensor<2x1xui8>
  return %q : tensor<2x1xui8>
}

// CHECK-LABEL: @transpose_no_dq
// CHECK-NOT: onnx.DequantizeLinear
// CHECK: onnx.Transpose
// CHECK: onnx.QuantizeLinear

// -----

// No Q on output (result feeds into a non-QuantizeLinear consumer).
func.func @transpose_no_q(%arg0: tensor<2x1xf32>) -> tensor<2x1xf32> {
  %a = onnx.Constant dense<[[10, 20]]> : tensor<1x2xui8>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %dq = "onnx.DequantizeLinear"(%a, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<1x2xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x2xf32>
  %t = "onnx.Transpose"(%dq) {perm = [1, 0]} : (tensor<1x2xf32>) -> tensor<2x1xf32>
  return %t : tensor<2x1xf32>
}

// BypassShapeOpThroughDQ pushes the Transpose into the constant.
// CHECK-LABEL: @transpose_no_q
// CHECK: onnx.Constant dense<{{\[}}[10], [20]]> : tensor<2x1xui8>
// CHECK: onnx.DequantizeLinear
// CHECK-NOT: onnx.Transpose
// CHECK-NOT: onnx.QuantizeLinear
