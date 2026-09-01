// RUN: onnx-mlir-opt --split-input-file %s -constprop-onnx=enable-dequant-const-fold | FileCheck %s
// RUN: onnx-mlir-opt --split-input-file %s -constprop-onnx | FileCheck %s --check-prefix=DISABLED

//===----------------------------------------------------------------------===//
// Positive cases.
//===----------------------------------------------------------------------===//

// Per-tensor, zero point = 0. (2-0)*0.5=1, (4-0)*0.5=2, (6-0)*0.5=3.
// The fold is gated behind enable-dequant-const-fold, so it is off by default.
func.func @fold_dq_per_tensor() -> tensor<3xf32> {
  %x = onnx.Constant dense<[2, 4, 6]> : tensor<3xui8>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %dq = "onnx.DequantizeLinear"(%x, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<3xui8>, tensor<f32>, tensor<ui8>) -> tensor<3xf32>
  return %dq : tensor<3xf32>
}

// CHECK-LABEL: @fold_dq_per_tensor
// CHECK-NOT: onnx.DequantizeLinear
// CHECK: onnx.Constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf32>

// DISABLED-LABEL: @fold_dq_per_tensor
// DISABLED: onnx.DequantizeLinear

// -----

// Per-tensor with a non-zero zero point. (18-10)*0.5=4, (10-10)*0.5=0.
func.func @fold_dq_with_zp() -> tensor<2xf32> {
  %x = onnx.Constant dense<[18, 10]> : tensor<2xui8>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<10> : tensor<ui8>
  %dq = "onnx.DequantizeLinear"(%x, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2xf32>
  return %dq : tensor<2xf32>
}

// CHECK-LABEL: @fold_dq_with_zp
// CHECK-NOT: onnx.DequantizeLinear
// CHECK: onnx.Constant dense<[4.000000e+00, 0.000000e+00]> : tensor<2xf32>

// -----

// Signed input with a negative value. (-3-(-1))*0.25 = -0.5, (5-(-1))*0.25 = 1.5.
func.func @fold_dq_signed() -> tensor<2xf32> {
  %x = onnx.Constant dense<[-3, 5]> : tensor<2xi8>
  %scale = onnx.Constant dense<2.500000e-01> : tensor<f32>
  %zp = onnx.Constant dense<-1> : tensor<i8>
  %dq = "onnx.DequantizeLinear"(%x, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<2xi8>, tensor<f32>, tensor<i8>) -> tensor<2xf32>
  return %dq : tensor<2xf32>
}

// CHECK-LABEL: @fold_dq_signed
// CHECK-NOT: onnx.DequantizeLinear
// CHECK: onnx.Constant dense<[-5.000000e-01, 1.500000e+00]> : tensor<2xf32>

// -----

// Per-axis along axis 0: row 0 uses scale 0.5, row 1 uses scale 0.25.
// row0: [(4-0)*0.5, (8-0)*0.5] = [2, 4]; row1: [(4-0)*0.25, (8-0)*0.25] = [1, 2].
func.func @fold_dq_per_axis() -> tensor<2x2xf32> {
  %x = onnx.Constant dense<[[4, 8], [4, 8]]> : tensor<2x2xui8>
  %scale = onnx.Constant dense<[5.000000e-01, 2.500000e-01]> : tensor<2xf32>
  %zp = onnx.Constant dense<[0, 0]> : tensor<2xui8>
  %dq = "onnx.DequantizeLinear"(%x, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2x2xui8>, tensor<2xf32>, tensor<2xui8>) -> tensor<2x2xf32>
  return %dq : tensor<2x2xf32>
}

// CHECK-LABEL: @fold_dq_per_axis
// CHECK-NOT: onnx.DequantizeLinear
// CHECK: onnx.Constant dense<{{\[}}[2.000000e+00, 4.000000e+00], [1.000000e+00, 2.000000e+00]]> : tensor<2x2xf32>

// -----

// No zero point operand (None). (4-0)*0.5 = 2.
func.func @fold_dq_no_zp() -> tensor<1xf32> {
  %x = onnx.Constant dense<[4]> : tensor<1xi8>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %none = "onnx.NoValue"() {value} : () -> none
  %dq = "onnx.DequantizeLinear"(%x, %scale, %none) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1xi8>, tensor<f32>, none) -> tensor<1xf32>
  return %dq : tensor<1xf32>
}

// CHECK-LABEL: @fold_dq_no_zp
// CHECK-NOT: onnx.DequantizeLinear
// CHECK: onnx.Constant dense<2.000000e+00> : tensor<1xf32>

// -----

// bf16 output: scale and result are bf16. (2-0)*0.5=1, (4-0)*0.5=2 (both exact in bf16).
func.func @fold_dq_bf16() -> tensor<2xbf16> {
  %x = onnx.Constant dense<[2, 4]> : tensor<2xui8>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<bf16>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %dq = "onnx.DequantizeLinear"(%x, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<2xui8>, tensor<bf16>, tensor<ui8>) -> tensor<2xbf16>
  return %dq : tensor<2xbf16>
}

// CHECK-LABEL: @fold_dq_bf16
// CHECK-NOT: onnx.DequantizeLinear
// CHECK: onnx.Constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xbf16>

// -----

// End-to-end: a MatMul whose operands both come
// from a DequantizeLinear-on-const folds all the way to a single Constant once
// the DQ folder turns each operand into a Constant.
// a = [[1,2],[3,4]] (scale 0.5 on [[2,4],[6,8]]), b = [[1,2],[3,4]] (scale 1.0).
// [[1,2],[3,4]] . [[1,2],[3,4]] = [[7,10],[15,22]].
func.func @fold_matmul_of_dq_consts() -> tensor<2x2xf32> {
  %a = onnx.Constant dense<[[2, 4], [6, 8]]> : tensor<2x2xui8>
  %a_scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %a_zp = onnx.Constant dense<0> : tensor<ui8>
  %a_dq = "onnx.DequantizeLinear"(%a, %a_scale, %a_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2x2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2x2xf32>

  %b = onnx.Constant dense<[[1, 2], [3, 4]]> : tensor<2x2xui8>
  %b_scale = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %b_zp = onnx.Constant dense<0> : tensor<ui8>
  %b_dq = "onnx.DequantizeLinear"(%b, %b_scale, %b_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2x2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2x2xf32>

  %mm = "onnx.MatMul"(%a_dq, %b_dq) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %mm : tensor<2x2xf32>
}

// CHECK-LABEL: @fold_matmul_of_dq_consts
// CHECK-NOT: onnx.DequantizeLinear
// CHECK-NOT: onnx.MatMul
// CHECK: onnx.Constant dense<{{\[}}[7.000000e+00, 1.000000e+01], [1.500000e+01, 2.200000e+01]]> : tensor<2x2xf32>

//===----------------------------------------------------------------------===//
// Negative cases.
//===----------------------------------------------------------------------===//

// -----

// Input is not a constant (function argument): DQ must remain.
func.func @no_fold_non_const(%arg0: tensor<3xui8>) -> tensor<3xf32> {
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %dq = "onnx.DequantizeLinear"(%arg0, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<3xui8>, tensor<f32>, tensor<ui8>) -> tensor<3xf32>
  return %dq : tensor<3xf32>
}

// CHECK-LABEL: @no_fold_non_const
// CHECK: onnx.DequantizeLinear

// -----

// Blocked quantization (block_size != 0): DQ must remain.
func.func @no_fold_blocked() -> tensor<4xf32> {
  %x = onnx.Constant dense<[1, 2, 3, 4]> : tensor<4xui8>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %dq = "onnx.DequantizeLinear"(%x, %scale, %zp) {axis = 0 : si64, block_size = 2 : si64} : (tensor<4xui8>, tensor<f32>, tensor<ui8>) -> tensor<4xf32>
  return %dq : tensor<4xf32>
}

// CHECK-LABEL: @no_fold_blocked
// CHECK: onnx.DequantizeLinear

// -----

// Scoping: a quantized weight feeding a real MatMul (with a non-constant
// activation) must NOT be dequantized -- it is not a constant island.
func.func @no_fold_weight_into_matmul(%act: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %w = onnx.Constant dense<[[1, 2], [3, 4]]> : tensor<2x2xui8>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %w_dq = "onnx.DequantizeLinear"(%w, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2x2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2x2xf32>
  %mm = "onnx.MatMul"(%act, %w_dq) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %mm : tensor<2x2xf32>
}

// CHECK-LABEL: @no_fold_weight_into_matmul
// CHECK: onnx.DequantizeLinear
// CHECK: onnx.MatMul

// -----

// Scoping: a quantized weight that is transposed before a real MatMul must NOT
// be dequantized -- the transitive consumer mixes in a non-constant activation.
func.func @no_fold_weight_via_transpose(%act: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %w = onnx.Constant dense<[[1, 2], [3, 4]]> : tensor<2x2xui8>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %w_dq = "onnx.DequantizeLinear"(%w, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2x2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2x2xf32>
  %wt = "onnx.Transpose"(%w_dq) {perm = [1, 0]} : (tensor<2x2xf32>) -> tensor<2x2xf32>
  %mm = "onnx.MatMul"(%act, %wt) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %mm : tensor<2x2xf32>
}

// CHECK-LABEL: @no_fold_weight_via_transpose
// CHECK: onnx.DequantizeLinear
// CHECK: onnx.Transpose

// -----

// A constant Add folds too: no op type is special-cased, so Add(const, const)
// inside an island collapses just like MatMul. [[1,2],[3,4]] + [[1,1],[1,1]].
func.func @fold_add_of_dq_consts() -> tensor<2x2xf32> {
  %a = onnx.Constant dense<[[2, 4], [6, 8]]> : tensor<2x2xui8>
  %a_scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %a_zp = onnx.Constant dense<0> : tensor<ui8>
  %a_dq = "onnx.DequantizeLinear"(%a, %a_scale, %a_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2x2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2x2xf32>

  %b = onnx.Constant dense<[[1, 1], [1, 1]]> : tensor<2x2xui8>
  %b_scale = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %b_zp = onnx.Constant dense<0> : tensor<ui8>
  %b_dq = "onnx.DequantizeLinear"(%b, %b_scale, %b_zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2x2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2x2xf32>

  %add = "onnx.Add"(%a_dq, %b_dq) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %add : tensor<2x2xf32>
}

// CHECK-LABEL: @fold_add_of_dq_consts
// CHECK-NOT: onnx.DequantizeLinear
// CHECK-NOT: onnx.Add
// CHECK: onnx.Constant dense<{{\[}}[2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00]]> : tensor<2x2xf32>

// -----

// A constant added to a non-constant activation must NOT be dequantized -- the
// operand check (not an op list) is what protects it.
func.func @no_fold_const_into_add(%act: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %b = onnx.Constant dense<[[1, 2], [3, 4]]> : tensor<2x2xui8>
  %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp = onnx.Constant dense<0> : tensor<ui8>
  %b_dq = "onnx.DequantizeLinear"(%b, %scale, %zp) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2x2xui8>, tensor<f32>, tensor<ui8>) -> tensor<2x2xf32>
  %add = "onnx.Add"(%act, %b_dq) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %add : tensor<2x2xf32>
}

// CHECK-LABEL: @no_fold_const_into_add
// CHECK: onnx.DequantizeLinear
// CHECK: onnx.Add
