// Tests for the new const-propagation patterns added on top of
// RemoveQDQForConst, all gated by --constprop-onnx=enable-qdq:
//   * BypassShapeOpThroughDQ<ONNXOp> -- swap a shape-only op past a DQ that
//     is fed by an integer constant so the *OfConst folders can fire.
//   * DropIdempotentQDQOnConst -- drop a Const -> DQ(s,z) -> Q(s,z,intT)
//     round trip back to the original constant.
//   * FoldRequantizeOnConst -- element-wise rewrite of
//     Const -> DQ(s1,z1) -> Q(s2,z2,intT2) into a single requantized
//     constant.

// RUN: onnx-mlir-opt --split-input-file %s -constprop-onnx=enable-qdq | FileCheck %s

//===----------------------------------------------------------------------===//
// BypassShapeOpThroughDQ: Transpose past a DQ-of-Const, with no matching Q
// on the output (the existing RemoveQDQForConst tests cover the Q/DQ-wrapped
// case; this one exercises the new "single-sided" bypass).
//===----------------------------------------------------------------------===//

func.func @transpose_bypass_dq_of_const() -> tensor<2x1xf32> {
  %a = onnx.Constant dense<[[10, 20]]> : tensor<1x2xui8>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %dq = "onnx.DequantizeLinear"(%a, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<1x2xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x2xf32>
  %t = "onnx.Transpose"(%dq) {perm = [1, 0]} : (tensor<1x2xf32>) -> tensor<2x1xf32>
  return %t : tensor<2x1xf32>
}

// The Transpose should be folded on the integer constant and a single DQ
// should remain feeding the return.
// CHECK-LABEL: @transpose_bypass_dq_of_const
// CHECK-NOT: onnx.Transpose
// CHECK: %[[C:.*]] = onnx.Constant dense<{{\[}}[10], [20]]> : tensor<2x1xui8>
// CHECK: %[[DQ:.*]] = "onnx.DequantizeLinear"(%[[C]],
// CHECK: return %[[DQ]] : tensor<2x1xf32>

// -----

//===----------------------------------------------------------------------===//
// BypassShapeOpThroughDQ: Reshape past a DQ-of-Const, again with no Q on the
// output side. Covers the variadic-operand path (shape operand is preserved).
//===----------------------------------------------------------------------===//

func.func @reshape_bypass_dq_of_const() -> tensor<1x4xf32> {
  %a = onnx.Constant dense<[10, 20, 30, 40]> : tensor<4xui8>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %shape = onnx.Constant dense<[1, 4]> : tensor<2xi64>
  %dq = "onnx.DequantizeLinear"(%a, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<4xui8>, tensor<f32>, tensor<ui8>) -> tensor<4xf32>
  %r = "onnx.Reshape"(%dq, %shape) {allowzero = 0 : si64} : (tensor<4xf32>, tensor<2xi64>) -> tensor<1x4xf32>
  return %r : tensor<1x4xf32>
}

// CHECK-LABEL: @reshape_bypass_dq_of_const
// CHECK-NOT: onnx.Reshape
// CHECK: %[[C:.*]] = onnx.Constant dense<{{\[}}[10, 20, 30, 40]]> : tensor<1x4xui8>
// CHECK: %[[DQ:.*]] = "onnx.DequantizeLinear"(%[[C]],
// CHECK: return %[[DQ]] : tensor<1x4xf32>

// -----

//===----------------------------------------------------------------------===//
// DropIdempotentQDQOnConst: Const -> DQ(s,z) -> Q(s,z, same intT) collapses
// to just the original integer constant.
//===----------------------------------------------------------------------===//

func.func @drop_idempotent_qdq_on_const() -> tensor<4xui8> {
  %a = onnx.Constant dense<[10, 20, 30, 40]> : tensor<4xui8>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %dq = "onnx.DequantizeLinear"(%a, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<4xui8>, tensor<f32>, tensor<ui8>) -> tensor<4xf32>
  %q = "onnx.QuantizeLinear"(%dq, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<4xf32>, tensor<f32>, tensor<ui8>) -> tensor<4xui8>
  return %q : tensor<4xui8>
}

// CHECK-LABEL: @drop_idempotent_qdq_on_const
// CHECK-NOT: onnx.DequantizeLinear
// CHECK-NOT: onnx.QuantizeLinear
// CHECK: %[[C:.*]] = onnx.Constant dense<[10, 20, 30, 40]> : tensor<4xui8>
// CHECK: return %[[C]] : tensor<4xui8>

// -----

//===----------------------------------------------------------------------===//
// FoldRequantizeOnConst: a genuine requantization (different output dtype,
// different scale, different zero point) is computed at compile time.
// Input  : ui8,  scale=0.5,  zp=0   -> floats: 10, 20, 30, 40
// Output : ui16, scale=1.0,  zp=5   -> q' = round(x * 0.5 / 1.0) + 5
//                                        = 10, 15, 20, 25
//===----------------------------------------------------------------------===//

func.func @fold_requantize_on_const() -> tensor<4xui16> {
  %a = onnx.Constant dense<[20, 30, 40, 50]> : tensor<4xui8>
  %s1 = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %z1 = onnx.Constant dense<0> : tensor<ui8>
  %s2 = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %z2 = onnx.Constant dense<5> : tensor<ui16>
  %dq = "onnx.DequantizeLinear"(%a, %s1, %z1) {axis = 0 : si64, block_size = 0 : si64} : (tensor<4xui8>, tensor<f32>, tensor<ui8>) -> tensor<4xf32>
  %q = "onnx.QuantizeLinear"(%dq, %s2, %z2) {axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<4xf32>, tensor<f32>, tensor<ui16>) -> tensor<4xui16>
  return %q : tensor<4xui16>
}

// CHECK-LABEL: @fold_requantize_on_const
// CHECK-NOT: onnx.DequantizeLinear
// CHECK-NOT: onnx.QuantizeLinear
// CHECK: %[[C:.*]] = onnx.Constant dense<[15, 20, 25, 30]> : tensor<4xui16>
// CHECK: return %[[C]] : tensor<4xui16>

// -----

//===----------------------------------------------------------------------===//
// BypassShapeOpThroughDQ: Slice past a DQ-of-Const (no Q sibling).
//===----------------------------------------------------------------------===//

func.func @slice_bypass_dq_of_const() -> tensor<2xf32> {
  %a = onnx.Constant dense<[10, 20, 30, 40]> : tensor<4xui8>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %starts = onnx.Constant dense<[1]> : tensor<1xi64>
  %ends = onnx.Constant dense<[3]> : tensor<1xi64>
  %axes = onnx.Constant dense<[0]> : tensor<1xi64>
  %steps = onnx.Constant dense<[1]> : tensor<1xi64>
  %dq = "onnx.DequantizeLinear"(%a, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<4xui8>, tensor<f32>, tensor<ui8>) -> tensor<4xf32>
  %s = "onnx.Slice"(%dq, %starts, %ends, %axes, %steps) : (tensor<4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xf32>
  return %s : tensor<2xf32>
}

// CHECK-LABEL: @slice_bypass_dq_of_const
// CHECK-NOT: onnx.Slice
// CHECK: %[[C:.*]] = onnx.Constant dense<[20, 30]> : tensor<2xui8>
// CHECK: %[[DQ:.*]] = "onnx.DequantizeLinear"(%[[C]],
// CHECK: return %[[DQ]] : tensor<2xf32>

// -----

//===----------------------------------------------------------------------===//
// BypassShapeOpThroughDQ: Squeeze past a DQ-of-Const.
//===----------------------------------------------------------------------===//

func.func @squeeze_bypass_dq_of_const() -> tensor<3xf32> {
  %a = onnx.Constant dense<[[10, 20, 30]]> : tensor<1x3xui8>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %axes = onnx.Constant dense<[0]> : tensor<1xi64>
  %dq = "onnx.DequantizeLinear"(%a, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<1x3xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x3xf32>
  %s = "onnx.Squeeze"(%dq, %axes) : (tensor<1x3xf32>, tensor<1xi64>) -> tensor<3xf32>
  return %s : tensor<3xf32>
}

// CHECK-LABEL: @squeeze_bypass_dq_of_const
// CHECK-NOT: onnx.Squeeze
// CHECK: %[[C:.*]] = onnx.Constant dense<[10, 20, 30]> : tensor<3xui8>
// CHECK: %[[DQ:.*]] = "onnx.DequantizeLinear"(%[[C]],
// CHECK: return %[[DQ]] : tensor<3xf32>

// -----

//===----------------------------------------------------------------------===//
// BypassShapeOpThroughDQ: Unsqueeze past a DQ-of-Const.
//===----------------------------------------------------------------------===//

func.func @unsqueeze_bypass_dq_of_const() -> tensor<1x3xf32> {
  %a = onnx.Constant dense<[10, 20, 30]> : tensor<3xui8>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %axes = onnx.Constant dense<[0]> : tensor<1xi64>
  %dq = "onnx.DequantizeLinear"(%a, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<3xui8>, tensor<f32>, tensor<ui8>) -> tensor<3xf32>
  %u = "onnx.Unsqueeze"(%dq, %axes) : (tensor<3xf32>, tensor<1xi64>) -> tensor<1x3xf32>
  return %u : tensor<1x3xf32>
}

// CHECK-LABEL: @unsqueeze_bypass_dq_of_const
// CHECK-NOT: onnx.Unsqueeze
// CHECK: %[[C:.*]] = onnx.Constant dense<{{\[}}[10, 20, 30]]> : tensor<1x3xui8>
// CHECK: %[[DQ:.*]] = "onnx.DequantizeLinear"(%[[C]],
// CHECK: return %[[DQ]] : tensor<1x3xf32>

// -----

//===----------------------------------------------------------------------===//
// Negative: BypassShapeOpThroughDQ must NOT fire for per-channel DQ
// (non-scalar scale / zero point) -- the rewrite explicitly bails on this
// to avoid silently dropping the per-axis quantization parameters.
//===----------------------------------------------------------------------===//

func.func @transpose_bypass_dq_per_channel_unchanged() -> tensor<2x1xf32> {
  %a = onnx.Constant dense<[[10, 20]]> : tensor<1x2xui8>
  %scale = onnx.Constant dense<[5.000000e-01, 2.500000e-01]> : tensor<2xf32>
  %zp = onnx.Constant dense<[0, 1]> : tensor<2xui8>
  %dq = "onnx.DequantizeLinear"(%a, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x2xui8>, tensor<2xf32>, tensor<2xui8>) -> tensor<1x2xf32>
  %t = "onnx.Transpose"(%dq) {perm = [1, 0]} : (tensor<1x2xf32>) -> tensor<2x1xf32>
  return %t : tensor<2x1xf32>
}

// CHECK-LABEL: @transpose_bypass_dq_per_channel_unchanged
// CHECK: "onnx.DequantizeLinear"
// CHECK: "onnx.Transpose"

// -----

//===----------------------------------------------------------------------===//
// FoldRequantizeOnConst: saturation -- values that fall outside the output
// integer range must be clamped to the output min/max, not wrapped.
// Input  : ui8  (raw 50, 200, 250),  scale=1.0, zp=0   -> f32 50, 200, 250
// Output : i8,  scale=1.0, zp=0
//   q' = round(x) saturated to [-128, 127]
//      = 50, 127, 127
//===----------------------------------------------------------------------===//

func.func @fold_requantize_saturates() -> tensor<3xi8> {
  %a = onnx.Constant dense<[50, 200, 250]> : tensor<3xui8>
  %s1 = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %z1 = onnx.Constant dense<0> : tensor<ui8>
  %s2 = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %z2 = onnx.Constant dense<0> : tensor<i8>
  %dq = "onnx.DequantizeLinear"(%a, %s1, %z1) {axis = 0 : si64, block_size = 0 : si64} : (tensor<3xui8>, tensor<f32>, tensor<ui8>) -> tensor<3xf32>
  %q = "onnx.QuantizeLinear"(%dq, %s2, %z2) {axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<3xf32>, tensor<f32>, tensor<i8>) -> tensor<3xi8>
  return %q : tensor<3xi8>
}

// CHECK-LABEL: @fold_requantize_saturates
// CHECK-NOT: onnx.DequantizeLinear
// CHECK-NOT: onnx.QuantizeLinear
// CHECK: %[[C:.*]] = onnx.Constant dense<[50, 127, 127]> : tensor<3xi8>
// CHECK: return %[[C]] : tensor<3xi8>

// -----

//===----------------------------------------------------------------------===//
// Negative: FoldRequantizeOnConst must NOT fire on per-channel scale/zp.
// (DropIdempotentQDQOnConst doesn't match either: scale != zp.) The IR must
// be left intact so a downstream per-channel-aware pass can handle it.
//===----------------------------------------------------------------------===//

func.func @fold_requantize_per_channel_unchanged() -> tensor<2xui16> {
  %a = onnx.Constant dense<[20, 30]> : tensor<2xui8>
  %s1 = onnx.Constant dense<[5.000000e-01, 2.500000e-01]> : tensor<2xf32>
  %z1 = onnx.Constant dense<[0, 1]> : tensor<2xui8>
  %s2 = onnx.Constant dense<[1.000000e+00, 5.000000e-01]> : tensor<2xf32>
  %z2 = onnx.Constant dense<[5, 7]> : tensor<2xui16>
  %dq = "onnx.DequantizeLinear"(%a, %s1, %z1) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<2xf32>, tensor<2xui8>) -> tensor<2xf32>
  %q = "onnx.QuantizeLinear"(%dq, %s2, %z2) {axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<2xf32>, tensor<2xf32>, tensor<2xui16>) -> tensor<2xui16>
  return %q : tensor<2xui16>
}

// CHECK-LABEL: @fold_requantize_per_channel_unchanged
// CHECK: "onnx.DequantizeLinear"
// CHECK: "onnx.QuantizeLinear"
