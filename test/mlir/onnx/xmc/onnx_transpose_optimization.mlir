// RUN: onnx-mlir-opt --onnx-transpose-optimization %s | FileCheck %s

// ============================================================================
// COMPLETE ONNX TRANSPOSE FUSION TEST SUITE
// This test covers all 66+ transformation rules from vaip_pass_fuse_transpose
// ============================================================================

// ============================================================================
// SECTION 1: Basic Transpose Optimizations
// ============================================================================

// CHECK-LABEL: func @test_identity_transpose
func.func @test_identity_transpose(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf32> {
  // CHECK-NOT: onnx.Transpose
  // CHECK: return %arg0
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 1, 2, 3]} : (tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf32>
  return %0 : tensor<1x3x224x224xf32>
}

// -----
// CHECK-LABEL: func @test_consecutive_transpose_to_identity
func.func @test_consecutive_transpose_to_identity(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf32> {
  // NHWC [0,2,3,1] followed by inverse [0,3,1,2] = identity
  // Both transposes should be eliminated
  // CHECK-NOT: onnx.Transpose
  // CHECK: return %arg0
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.Transpose"(%0) {perm = [0, 3, 1, 2]} : (tensor<1x224x224x3xf32>) -> tensor<1x3x224x224xf32>
  return %1 : tensor<1x3x224x224xf32>
}

// -----
// CHECK-LABEL: func @test_consecutive_transpose_fused
func.func @test_consecutive_transpose_fused(%arg0: tensor<1x2x3x4xf32>) -> tensor<1x4x2x3xf32> {
  // CHECK: %[[RES:.*]] = "onnx.Transpose"(%arg0) {perm = [0, 3, 1, 2]}
  // CHECK: return %[[RES]]
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1, 3]} : (tensor<1x2x3x4xf32>) -> tensor<1x3x2x4xf32>
  %1 = "onnx.Transpose"(%0) {perm = [0, 3, 2, 1]} : (tensor<1x3x2x4xf32>) -> tensor<1x4x2x3xf32>
  return %1 : tensor<1x4x2x3xf32>
}

// -----
// CHECK-LABEL: func @test_three_consecutive_transposes
func.func @test_three_consecutive_transposes(%arg0: tensor<2x3x4x5xf32>) -> tensor<5x3x2x4xf32> {
  // CHECK: %[[T1:.*]] = "onnx.Transpose"(%arg0) {perm = [3, 1, 0, 2]}
  // CHECK: return %[[T1]]
  %0 = "onnx.Transpose"(%arg0) {perm = [1, 2, 3, 0]} : (tensor<2x3x4x5xf32>) -> tensor<3x4x5x2xf32>
  %1 = "onnx.Transpose"(%0) {perm = [2, 0, 3, 1]} : (tensor<3x4x5x2xf32>) -> tensor<5x3x2x4xf32>
  return %1 : tensor<5x3x2x4xf32>
}

// ============================================================================
// SECTION 2: SISO Unary Operations (21 ops)
// ============================================================================

// CHECK-LABEL: func @test_transpose_through_relu
func.func @test_transpose_through_relu(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32> {
  // CHECK: %[[RELU:.*]] = "onnx.Relu"(%arg0)
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[RELU]]) {perm = [0, 2, 3, 1]}
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.Relu"(%0) : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  return %1 : tensor<1x224x224x3xf32>
}

// -----
// CHECK-LABEL: func @test_transpose_through_sigmoid
func.func @test_transpose_through_sigmoid(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32> {
  // CHECK: %[[SIGMOID:.*]] = "onnx.Sigmoid"(%arg0)
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[SIGMOID]]) {perm = [0, 2, 3, 1]}
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.Sigmoid"(%0) : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  return %1 : tensor<1x224x224x3xf32>
}

// -----
// CHECK-LABEL: func @test_transpose_through_tanh
func.func @test_transpose_through_tanh(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32> {
  // CHECK: %[[TANH:.*]] = "onnx.Tanh"(%arg0)
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[TANH]]) {perm = [0, 2, 3, 1]}
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.Tanh"(%0) : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  return %1 : tensor<1x224x224x3xf32>
}

// -----
// CHECK-LABEL: func @test_multiple_siso_chain
func.func @test_multiple_siso_chain(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32> {
  // CHECK: %[[RELU:.*]] = "onnx.Relu"(%arg0)
  // CHECK: %[[SIGMOID:.*]] = "onnx.Sigmoid"(%[[RELU]])
  // CHECK: %[[TANH:.*]] = "onnx.Tanh"(%[[SIGMOID]])
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[TANH]]) {perm = [0, 2, 3, 1]}
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.Relu"(%0) : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  %2 = "onnx.Sigmoid"(%1) : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  %3 = "onnx.Tanh"(%2) : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  return %3 : tensor<1x224x224x3xf32>
}

// -----
// CHECK-LABEL: func @test_transpose_through_exp_log_sqrt
func.func @test_transpose_through_exp_log_sqrt(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32> {
  // CHECK: %[[EXP:.*]] = "onnx.Exp"(%arg0)
  // CHECK: %[[LOG:.*]] = "onnx.Log"(%[[EXP]])
  // CHECK: %[[SQRT:.*]] = "onnx.Sqrt"(%[[LOG]])
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[SQRT]]) {perm = [0, 2, 3, 1]}
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.Exp"(%0) : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  %2 = "onnx.Log"(%1) : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  %3 = "onnx.Sqrt"(%2) : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  return %3 : tensor<1x224x224x3xf32>
}

// ============================================================================
// SECTION 3: QDQ Operations (Quantization/Dequantization)
// ============================================================================

// CHECK-LABEL: func @test_transpose_through_quantize_linear
func.func @test_transpose_through_quantize_linear(%arg0: tensor<1x3x224x224xf32>, %arg1: tensor<f32>, %arg2: tensor<ui8>) -> tensor<1x224x224x3xui8> {
  // CHECK: %[[QUANT:.*]] = "onnx.QuantizeLinear"(%arg0, %arg1, %arg2)
  // CHECK-SAME: (tensor<1x3x224x224xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x3x224x224xui8>
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[QUANT]]) {perm = [0, 2, 3, 1]}
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.QuantizeLinear"(%0, %arg1, %arg2) : (tensor<1x224x224x3xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x224x224x3xui8>
  return %1 : tensor<1x224x224x3xui8>
}

// -----
// CHECK-LABEL: func @test_transpose_through_dequantize_linear
func.func @test_transpose_through_dequantize_linear(%arg0: tensor<1x3x224x224xui8>, %arg1: tensor<f32>, %arg2: tensor<ui8>) -> tensor<1x224x224x3xf32> {
  // CHECK: %[[DEQUANT:.*]] = "onnx.DequantizeLinear"(%arg0, %arg1, %arg2)
  // CHECK-SAME: (tensor<1x3x224x224xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x3x224x224xf32>
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[DEQUANT]]) {perm = [0, 2, 3, 1]}
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xui8>) -> tensor<1x224x224x3xui8>
  %1 = "onnx.DequantizeLinear"(%0, %arg1, %arg2) : (tensor<1x224x224x3xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x224x224x3xf32>
  return %1 : tensor<1x224x224x3xf32>
}

// -----
// CHECK-LABEL: func @test_qdq_chain
func.func @test_qdq_chain(%arg0: tensor<1x3x224x224xf32>, %arg1: tensor<f32>, %arg2: tensor<ui8>) -> tensor<1x224x224x3xf32> {
  // CHECK: %[[QUANT:.*]] = "onnx.QuantizeLinear"(%arg0
  // CHECK: %[[DEQUANT:.*]] = "onnx.DequantizeLinear"(%[[QUANT]]
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[DEQUANT]]) {perm = [0, 2, 3, 1]}
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.QuantizeLinear"(%0, %arg1, %arg2) : (tensor<1x224x224x3xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x224x224x3xui8>
  %2 = "onnx.DequantizeLinear"(%1, %arg1, %arg2) : (tensor<1x224x224x3xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x224x224x3xf32>
  return %2 : tensor<1x224x224x3xf32>
}


// ============================================================================
// SECTION 4: Binary Operations - Both Inputs Transposed
// ============================================================================

// CHECK-LABEL: func @test_binary_add_both_transposed
func.func @test_binary_add_both_transposed(%arg0: tensor<1x64x56x56xf32>, %arg1: tensor<1x64x56x56xf32>) -> tensor<1x56x56x64xf32> {
  // CHECK: %[[ADD:.*]] = "onnx.Add"(%arg0, %arg1)
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[ADD]]) {perm = [0, 2, 3, 1]}
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x64x56x56xf32>) -> tensor<1x56x56x64xf32>
  %1 = "onnx.Transpose"(%arg1) {perm = [0, 2, 3, 1]} : (tensor<1x64x56x56xf32>) -> tensor<1x56x56x64xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  return %2 : tensor<1x56x56x64xf32>
}

// -----
// CHECK-LABEL: func @test_binary_sub_mul_div
func.func @test_binary_sub_mul_div(%arg0: tensor<1x64x56x56xf32>, %arg1: tensor<1x64x56x56xf32>) -> tensor<1x56x56x64xf32> {
  // With cloning support: 2 input transposes fused backwards -> 1 output transpose
  // Net result: 1 transpose eliminated!
  // CHECK: %[[SUB:.*]] = "onnx.Sub"(%arg0, %arg1)
  // CHECK: %[[MUL:.*]] = "onnx.Mul"(%[[SUB]], %arg0)
  // CHECK: %[[DIV:.*]] = "onnx.Div"(%[[MUL]], %arg1)
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[DIV]]) {perm = [0, 2, 3, 1]}
  // CHECK: return %[[TRANS]]
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x64x56x56xf32>) -> tensor<1x56x56x64xf32>
  %1 = "onnx.Transpose"(%arg1) {perm = [0, 2, 3, 1]} : (tensor<1x64x56x56xf32>) -> tensor<1x56x56x64xf32>
  %2 = "onnx.Sub"(%0, %1) : (tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %3 = "onnx.Mul"(%2, %0) : (tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %4 = "onnx.Div"(%3, %1) : (tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  return %4 : tensor<1x56x56x64xf32>
}

// -----
// CHECK-LABEL: func @test_binary_add_with_relu
func.func @test_binary_add_with_relu(%arg0: tensor<1x64x56x56xf32>, %arg1: tensor<1x64x56x56xf32>) -> tensor<1x56x56x64xf32> {
  // 2 input transposes pushed down -> operations in original space -> 1 output transpose
  // CHECK: %[[ADD:.*]] = "onnx.Add"(%arg0, %arg1)
  // CHECK: %[[RELU:.*]] = "onnx.Relu"(%[[ADD]])
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[RELU]]) {perm = [0, 2, 3, 1]}
  // CHECK: return %[[TRANS]]
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x64x56x56xf32>) -> tensor<1x56x56x64xf32>
  %1 = "onnx.Transpose"(%arg1) {perm = [0, 2, 3, 1]} : (tensor<1x64x56x56xf32>) -> tensor<1x56x56x64xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %3 = "onnx.Relu"(%2) : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  return %3 : tensor<1x56x56x64xf32>
}

// -----
// CHECK-LABEL: func @test_binary_with_transpose_immune
func.func @test_binary_with_transpose_immune(%arg0: tensor<1x64x56x56xf32>, %arg1: tensor<1x1x1x1xf32>) -> tensor<1x56x56x64xf32> {
  // Transpose-immune constant (1x1x1x1) - no Reshape needed, shape doesn't change!
  // CHECK-NOT: onnx.Reshape
  // CHECK: %[[ADD:.*]] = "onnx.Add"(%arg0, %arg1)
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[ADD]]) {perm = [0, 2, 3, 1]}
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x64x56x56xf32>) -> tensor<1x56x56x64xf32>
  %1 = "onnx.Add"(%0, %arg1) : (tensor<1x56x56x64xf32>, tensor<1x1x1x1xf32>) -> tensor<1x56x56x64xf32>
  return %1 : tensor<1x56x56x64xf32>
}

// -----
// CHECK-LABEL: func @test_binary_add_with_constant
func.func @test_binary_add_with_constant(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x4x4x3xf32> {
  %const = "onnx.Constant"() {value = dense<[[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]]> : tensor<1x1x4x3xf32>} : () -> tensor<1x1x4x3xf32>
  // Constant is NOT transpose-immune (has non-1 dimensions), so should transpose the constant
  // CHECK: onnx.Constant
  // CHECK: %[[ADD:.*]] = "onnx.Add"(%arg0
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[ADD]]) {perm = [0, 2, 3, 1]}
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x4x4xf32>) -> tensor<1x4x4x3xf32>
  %1 = "onnx.Add"(%0, %const) : (tensor<1x4x4x3xf32>, tensor<1x1x4x3xf32>) -> tensor<1x4x4x3xf32>
  return %1 : tensor<1x4x4x3xf32>
}

// -----
// CHECK-LABEL: func @test_binary_mul_with_transpose_immune_constant
func.func @test_binary_mul_with_transpose_immune_constant(%arg0: tensor<2x3x4xf32>) -> tensor<4x3x2xf32> {
  %const = "onnx.Constant"() {value = dense<2.0> : tensor<1x1x1xf32>} : () -> tensor<1x1x1xf32>
  // Transpose-immune constant (1x1x1) - no Reshape needed, permuting [2,1,0] gives [1,1,1]!
  // CHECK: %[[CONST:.*]] = onnx.Constant dense<2.000000e+00>
  // CHECK-NOT: onnx.Reshape
  // CHECK: %[[MUL:.*]] = "onnx.Mul"(%arg0, %[[CONST]])
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[MUL]]) {perm = [2, 1, 0]}
  %0 = "onnx.Transpose"(%arg0) {perm = [2, 1, 0]} : (tensor<2x3x4xf32>) -> tensor<4x3x2xf32>
  %1 = "onnx.Mul"(%0, %const) : (tensor<4x3x2xf32>, tensor<1x1x1xf32>) -> tensor<4x3x2xf32>
  return %1 : tensor<4x3x2xf32>
}

// -----
// CHECK-LABEL: func @test_binary_sub_with_constant_lhs
func.func @test_binary_sub_with_constant_lhs(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x4x4x3xf32> {
  %const = "onnx.Constant"() {value = dense<[[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]]> : tensor<1x1x4x3xf32>} : () -> tensor<1x1x4x3xf32>
  // Constant on LHS
  // CHECK: onnx.Constant
  // CHECK: %[[SUB:.*]] = "onnx.Sub"(
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[SUB]]) {perm = [0, 2, 3, 1]}
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x4x4xf32>) -> tensor<1x4x4x3xf32>
  %1 = "onnx.Sub"(%const, %0) : (tensor<1x1x4x3xf32>, tensor<1x4x4x3xf32>) -> tensor<1x4x4x3xf32>
  return %1 : tensor<1x4x4x3xf32>
}

// -----
// CHECK-LABEL: func @test_binary_div_with_constant
func.func @test_binary_div_with_constant(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x4x4x3xf32> {
  %const = "onnx.Constant"() {value = dense<[[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]]> : tensor<1x1x4x3xf32>} : () -> tensor<1x1x4x3xf32>
  // CHECK: onnx.Constant
  // CHECK: %[[DIV:.*]] = "onnx.Div"(%arg0
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[DIV]]) {perm = [0, 2, 3, 1]}
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x4x4xf32>) -> tensor<1x4x4x3xf32>
  %1 = "onnx.Div"(%0, %const) : (tensor<1x4x4x3xf32>, tensor<1x1x4x3xf32>) -> tensor<1x4x4x3xf32>
  return %1 : tensor<1x4x4x3xf32>
}

// -----
// CHECK-LABEL: func @test_binary_pow_with_constant
func.func @test_binary_pow_with_constant(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x4x4x3xf32> {
  %const = "onnx.Constant"() {value = dense<[[[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]]]> : tensor<1x1x4x3xf32>} : () -> tensor<1x1x4x3xf32>
  // CHECK: onnx.Constant
  // CHECK: %[[POW:.*]] = "onnx.Pow"(%arg0
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[POW]]) {perm = [0, 2, 3, 1]}
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x4x4xf32>) -> tensor<1x4x4x3xf32>
  %1 = "onnx.Pow"(%0, %const) : (tensor<1x4x4x3xf32>, tensor<1x1x4x3xf32>) -> tensor<1x4x4x3xf32>
  return %1 : tensor<1x4x4x3xf32>
}

// ============================================================================
// SECTION 5: Reduction Operations (Transform Axes)
// ============================================================================

// CHECK-LABEL: func @test_transpose_through_reduce_mean
func.func @test_transpose_through_reduce_mean(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x224x224x1xf32> {
  %axes = "onnx.Constant"() {value = dense<[3]> : tensor<1xi64>} : () -> tensor<1xi64>
  // NHWC transformation: NCHW [1,3,224,224] -> NHWC [1,224,224,3] via perm [0,2,3,1]
  // Channel dimension C moves from position 1 to position 3
  // ReduceMean on axis=3 (channel dim in NHWC): [1,224,224,3] -> [1,224,224,1]
  // After pushing transpose: reduce axis=1 from original [1,3,224,224] -> [1,1,224,224]
  // Then apply transpose [0,2,3,1]: [1,1,224,224] -> [1,224,224,1]
  // CHECK: %[[AXES:.*]] = onnx.Constant dense<1> : tensor<1xi64>
  // CHECK: %[[REDUCE:.*]] = "onnx.ReduceMean"(%arg0, %[[AXES]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
  // CHECK-SAME: (tensor<1x3x224x224xf32>, tensor<1xi64>) -> tensor<1x1x224x224xf32>
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[REDUCE]]) {perm = [0, 2, 3, 1]}
  // CHECK-SAME: (tensor<1x1x224x224xf32>) -> tensor<1x224x224x1xf32>
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.ReduceMean"(%0, %axes) {keepdims = 1 : si64} : (tensor<1x224x224x3xf32>, tensor<1xi64>) -> tensor<1x224x224x1xf32>
  return %1 : tensor<1x224x224x1xf32>
}

// -----
// CHECK-LABEL: func @test_transpose_through_reduce_sum
func.func @test_transpose_through_reduce_sum(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x224x224x1xf32> {
  %axes = "onnx.Constant"() {value = dense<[3]> : tensor<1xi64>} : () -> tensor<1xi64>
  // NHWC transformation: NCHW [1,3,224,224] -> NHWC [1,224,224,3] via perm [0,2,3,1]
  // Channel dimension C moves from position 1 to position 3
  // ReduceSum on axis=3 (channel dim in NHWC): [1,224,224,3] -> [1,224,224,1]
  // CHECK: %[[AXES:.*]] = onnx.Constant dense<1> : tensor<1xi64>
  // CHECK: %[[REDUCE:.*]] = "onnx.ReduceSum"(%arg0, %[[AXES]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
  // CHECK-SAME: (tensor<1x3x224x224xf32>, tensor<1xi64>) -> tensor<1x1x224x224xf32>
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[REDUCE]]) {perm = [0, 2, 3, 1]}
  // CHECK-SAME: (tensor<1x1x224x224xf32>) -> tensor<1x224x224x1xf32>
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.ReduceSum"(%0, %axes) {keepdims = 1 : si64} : (tensor<1x224x224x3xf32>, tensor<1xi64>) -> tensor<1x224x224x1xf32>
  return %1 : tensor<1x224x224x1xf32>
}

// -----
// CHECK-LABEL: func @test_transpose_through_reduce_max_min
func.func @test_transpose_through_reduce_max_min(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x224x224x1xf32> {
  %axes = "onnx.Constant"() {value = dense<[3]> : tensor<1xi64>} : () -> tensor<1xi64>
  // NHWC transformation: reduce channel dimension twice
  // First ReduceMax on axis=3 (channel), then ReduceMin on axis=3
  // First reduce pushes transpose through, second one doesn't have transpose before it
  // CHECK: onnx.Constant
  // CHECK: onnx.ReduceMax
  // CHECK: onnx.ReduceMin
  // CHECK: onnx.Transpose
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.ReduceMax"(%0, %axes) {keepdims = 1 : si64} : (tensor<1x224x224x3xf32>, tensor<1xi64>) -> tensor<1x224x224x1xf32>
  %2 = "onnx.ReduceMin"(%1, %axes) {keepdims = 1 : si64} : (tensor<1x224x224x1xf32>, tensor<1xi64>) -> tensor<1x224x224x1xf32>
  return %2 : tensor<1x224x224x1xf32>
}

// ============================================================================
// SECTION 6: Specialized Operations
// ============================================================================

// CHECK-LABEL: func @test_concat_two_inputs
func.func @test_concat_two_inputs(%arg0: tensor<1x3x224x224xf32>, %arg1: tensor<1x3x224x224xf32>) -> tensor<1x224x224x6xf32> {
  // Input: Concat on axis=3 in transposed layout with perm [0,2,3,1]
  // Original axis 1 -> transposed position 3, so after pushing transpose, axis should be 1
  // CHECK: %[[CONCAT:.*]] = "onnx.Concat"(%arg0, %arg1) {axis = 1 : si64}
  // CHECK-SAME: (tensor<1x3x224x224xf32>, tensor<1x3x224x224xf32>) -> tensor<1x6x224x224xf32>
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[CONCAT]]) {perm = [0, 2, 3, 1]}
  // CHECK-SAME: (tensor<1x6x224x224xf32>) -> tensor<1x224x224x6xf32>
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.Transpose"(%arg1) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %2 = "onnx.Concat"(%0, %1) {axis = 3 : si64} : (tensor<1x224x224x3xf32>, tensor<1x224x224x3xf32>) -> tensor<1x224x224x6xf32>
  return %2 : tensor<1x224x224x6xf32>
}

// -----
// CHECK-LABEL: func @test_concat_three_inputs
func.func @test_concat_three_inputs(%arg0: tensor<1x64x56x56xf32>, %arg1: tensor<1x64x56x56xf32>, %arg2: tensor<1x64x56x56xf32>) -> tensor<1x56x56x192xf32> {
  // Input: Concat on axis=3 (channel dim in NHWC) with perm [0,2,3,1] (NCHW->NHWC)
  // Original channel axis=1 -> transposed position=3, so after pushing, axis should be 1
  // CHECK: %[[CONCAT:.*]] = "onnx.Concat"(%arg0, %arg1, %arg2) {axis = 1 : si64}
  // CHECK-SAME: (tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) -> tensor<1x192x56x56xf32>
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[CONCAT]]) {perm = [0, 2, 3, 1]}
  // CHECK-SAME: (tensor<1x192x56x56xf32>) -> tensor<1x56x56x192xf32>
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x64x56x56xf32>) -> tensor<1x56x56x64xf32>
  %1 = "onnx.Transpose"(%arg1) {perm = [0, 2, 3, 1]} : (tensor<1x64x56x56xf32>) -> tensor<1x56x56x64xf32>
  %2 = "onnx.Transpose"(%arg2) {perm = [0, 2, 3, 1]} : (tensor<1x64x56x56xf32>) -> tensor<1x56x56x64xf32>
  %3 = "onnx.Concat"(%0, %1, %2) {axis = 3 : si64} : (tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x192xf32>
  return %3 : tensor<1x56x56x192xf32>
}

// -----
// CHECK-LABEL: func @test_concat_different_sizes
func.func @test_concat_different_sizes(%arg0: tensor<1x3x224x224xf32>, %arg1: tensor<1x5x224x224xf32>) -> tensor<1x224x224x8xf32> {
  // Concatenating different channel sizes: 3 + 5 = 8
  // Input: axis=3 in NHWC (transposed), should become axis=1 in NCHW (original)
  // CHECK: %[[CONCAT:.*]] = "onnx.Concat"(%arg0, %arg1) {axis = 1 : si64}
  // CHECK-SAME: (tensor<1x3x224x224xf32>, tensor<1x5x224x224xf32>) -> tensor<1x8x224x224xf32>
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[CONCAT]]) {perm = [0, 2, 3, 1]}
  // CHECK-SAME: (tensor<1x8x224x224xf32>) -> tensor<1x224x224x8xf32>
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.Transpose"(%arg1) {perm = [0, 2, 3, 1]} : (tensor<1x5x224x224xf32>) -> tensor<1x224x224x5xf32>
  %2 = "onnx.Concat"(%0, %1) {axis = 3 : si64} : (tensor<1x224x224x3xf32>, tensor<1x224x224x5xf32>) -> tensor<1x224x224x8xf32>
  return %2 : tensor<1x224x224x8xf32>
}

// -----
// CHECK-LABEL: func @test_concat_negative_axis
func.func @test_concat_negative_axis(%arg0: tensor<1x3x224x224xf32>, %arg1: tensor<1x3x224x224xf32>) -> tensor<1x224x224x6xf32> {
  // Using negative axis: -1 means last dimension (axis=3 in 4D)
  // After normalization: -1 + 4 = 3, then transformed: perm[3] = 1
  // CHECK: %[[CONCAT:.*]] = "onnx.Concat"(%arg0, %arg1) {axis = 1 : si64}
  // CHECK-SAME: (tensor<1x3x224x224xf32>, tensor<1x3x224x224xf32>) -> tensor<1x6x224x224xf32>
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[CONCAT]]) {perm = [0, 2, 3, 1]}
  // CHECK-SAME: (tensor<1x6x224x224xf32>) -> tensor<1x224x224x6xf32>
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.Transpose"(%arg1) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %2 = "onnx.Concat"(%0, %1) {axis = -1 : si64} : (tensor<1x224x224x3xf32>, tensor<1x224x224x3xf32>) -> tensor<1x224x224x6xf32>
  return %2 : tensor<1x224x224x6xf32>
}

// -----
// CHECK-LABEL: func @test_concat_on_height_axis
func.func @test_concat_on_height_axis(%arg0: tensor<1x3x112x224xf32>, %arg1: tensor<1x3x112x224xf32>) -> tensor<1x224x224x3xf32> {
  // Concatenating on height dimension (axis=1 in NHWC after transpose)
  // With perm [0,2,3,1]: original axis 2 (H) -> transposed position 1
  // So axis=1 in transposed corresponds to axis=2 in original
  // CHECK: %[[CONCAT:.*]] = "onnx.Concat"(%arg0, %arg1) {axis = 2 : si64}
  // CHECK-SAME: (tensor<1x3x112x224xf32>, tensor<1x3x112x224xf32>) -> tensor<1x3x224x224xf32>
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[CONCAT]]) {perm = [0, 2, 3, 1]}
  // CHECK-SAME: (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x112x224xf32>) -> tensor<1x112x224x3xf32>
  %1 = "onnx.Transpose"(%arg1) {perm = [0, 2, 3, 1]} : (tensor<1x3x112x224xf32>) -> tensor<1x112x224x3xf32>
  %2 = "onnx.Concat"(%0, %1) {axis = 1 : si64} : (tensor<1x112x224x3xf32>, tensor<1x112x224x3xf32>) -> tensor<1x224x224x3xf32>
  return %2 : tensor<1x224x224x3xf32>
}

// -----
// CHECK-LABEL: func @test_concat_mixed_inputs_no_optimization
func.func @test_concat_mixed_inputs_no_optimization(%arg0: tensor<1x3x224x224xf32>, %arg1: tensor<1x224x224x3xf32>) -> tensor<1x224x224x6xf32> {
  // One input is transposed, one is not - should NOT optimize
  // CHECK: onnx.Transpose
  // CHECK: onnx.Concat
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.Concat"(%0, %arg1) {axis = 3 : si64} : (tensor<1x224x224x3xf32>, tensor<1x224x224x3xf32>) -> tensor<1x224x224x6xf32>
  return %1 : tensor<1x224x224x6xf32>
}

// -----
// CHECK-LABEL: func @test_concat_different_permutations_no_optimization
func.func @test_concat_different_permutations_no_optimization(%arg0: tensor<1x3x224x224xf32>, %arg1: tensor<1x224x3x224xf32>) -> tensor<1x224x224x6xf32> {
  // Different permutations - should NOT optimize
  // CHECK: onnx.Transpose
  // CHECK: onnx.Transpose
  // CHECK: onnx.Concat
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.Transpose"(%arg1) {perm = [0, 1, 3, 2]} : (tensor<1x224x3x224xf32>) -> tensor<1x224x224x3xf32>
  %2 = "onnx.Concat"(%0, %1) {axis = 3 : si64} : (tensor<1x224x224x3xf32>, tensor<1x224x224x3xf32>) -> tensor<1x224x224x6xf32>
  return %2 : tensor<1x224x224x6xf32>
}

// -----
// CHECK-LABEL: func @test_transpose_through_slice
func.func @test_transpose_through_slice(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x112x112x3xf32> {
  %starts = "onnx.Constant"() {value = dense<[0, 0, 0, 0]> : tensor<4xi64>} : () -> tensor<4xi64>
  %ends = "onnx.Constant"() {value = dense<[1, 112, 112, 3]> : tensor<4xi64>} : () -> tensor<4xi64>
  %axes = "onnx.Constant"() {value = dense<[0, 1, 2, 3]> : tensor<4xi64>} : () -> tensor<4xi64>
  %steps = "onnx.Constant"() {value = dense<[1, 1, 1, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
  // Input: Slice on transposed tensor [0,2,3,1]: axes=[0,1,2,3], starts=[0,0,0,0], ends=[1,112,112,3]
  // After pushing transpose with perm [0,2,3,1], inversePerm=[0,3,1,2]:
  //   - Strategy 1: Full-rank sequential axes
  //   - Axes stay unchanged: [0, 1, 2, 3]
  //   - starts/ends/steps reordered using inversePerm: [0,0,0,0] -> [0,0,0,0], [1,112,112,3] -> [1,3,112,112]
  // CHECK: onnx.Constant dense<1> : tensor<4xi64>
  // CHECK: onnx.Constant dense<[1, 3, 112, 112]> : tensor<4xi64>
  // CHECK: onnx.Constant dense<0> : tensor<4xi64>
  // CHECK: onnx.Constant dense<[0, 1, 2, 3]> : tensor<4xi64>
  // CHECK: "onnx.Slice"
  // CHECK: "onnx.Transpose"
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.Slice"(%0, %starts, %ends, %axes, %steps) : (tensor<1x224x224x3xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x112x112x3xf32>
  return %1 : tensor<1x112x112x3xf32>
}


// -----
// CHECK-LABEL: func @test_transpose_through_expand
func.func @test_transpose_through_expand(%arg0: tensor<1x3x1x1xf32>) -> tensor<1x224x224x3xf32> {
  %shape = "onnx.Constant"() {value = dense<[1, 224, 224, 3]> : tensor<4xi64>} : () -> tensor<4xi64>
  // Input: Expand with shape=[1, 224, 224, 3] for transposed tensor with perm [0,2,3,1]
  // After pushing transpose: shape=[1, 3, 224, 224] (dim 1->3, dim 2->1, dim 3->2)
  // CHECK: %[[SHAPE:.*]] = onnx.Constant dense<[1, 3, 224, 224]> : tensor<4xi64>
  // CHECK: %[[EXPAND:.*]] = "onnx.Expand"(%arg0, %[[SHAPE]])
  // CHECK-SAME: (tensor<1x3x1x1xf32>, tensor<4xi64>) -> tensor<1x3x224x224xf32>
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[EXPAND]]) {perm = [0, 2, 3, 1]}
  // CHECK-SAME: (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x1x1xf32>) -> tensor<1x1x1x3xf32>
  %1 = "onnx.Expand"(%0, %shape) : (tensor<1x1x1x3xf32>, tensor<4xi64>) -> tensor<1x224x224x3xf32>
  return %1 : tensor<1x224x224x3xf32>
}

// -----
// CHECK-LABEL: func @test_transpose_through_tile
func.func @test_transpose_through_tile(%arg0: tensor<1x3x56x56xf32>) -> tensor<1x112x112x3xf32> {
  %repeats = "onnx.Constant"() {value = dense<[1, 2, 2, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
  // Input: Tile with repeats=[1, 2, 2, 1] for transposed tensor with perm [0,2,3,1]
  // After pushing transpose: repeats=[1, 1, 2, 2] (dim 1->3, dim 2->1, dim 3->2)
  // CHECK: %[[REPEATS:.*]] = onnx.Constant dense<[1, 1, 2, 2]> : tensor<4xi64>
  // CHECK: %[[TILE:.*]] = "onnx.Tile"(%arg0, %[[REPEATS]])
  // CHECK-SAME: (tensor<1x3x56x56xf32>, tensor<4xi64>) -> tensor<1x3x112x112xf32>
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[TILE]]) {perm = [0, 2, 3, 1]}
  // CHECK-SAME: (tensor<1x3x112x112xf32>) -> tensor<1x112x112x3xf32>
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x56x56xf32>) -> tensor<1x56x56x3xf32>
  %1 = "onnx.Tile"(%0, %repeats) : (tensor<1x56x56x3xf32>, tensor<4xi64>) -> tensor<1x112x112x3xf32>
  return %1 : tensor<1x112x112x3xf32>
}

// -----
// CHECK-LABEL: func @test_transpose_through_argmax
func.func @test_transpose_through_argmax(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x224x224x1xi64> {
  // NHWC transformation: ArgMax on channel axis=3
  // Original axis 1 (C) -> transposed position 3, so after pushing transpose, axis should be 1
  // CHECK: %[[ARGMAX:.*]] = "onnx.ArgMax"(%arg0) {axis = 1 : si64, keepdims = 1 : si64, select_last_index = 0 : si64}
  // CHECK-SAME: (tensor<1x3x224x224xf32>) -> tensor<1x1x224x224xi64>
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[ARGMAX]]) {perm = [0, 2, 3, 1]}
  // CHECK-SAME: (tensor<1x1x224x224xi64>) -> tensor<1x224x224x1xi64>
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.ArgMax"(%0) {axis = 3 : si64, keepdims = 1 : si64, select_last_index = 0 : si64} : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x1xi64>
  return %1 : tensor<1x224x224x1xi64>
}

// -----
// CHECK-LABEL: func @test_transpose_through_squeeze
func.func @test_transpose_through_squeeze(%arg0: tensor<1x1x224x224xf32>) -> tensor<1x224x224xf32> {
  %axes = "onnx.Constant"() {value = dense<[3]> : tensor<1xi64>} : () -> tensor<1xi64>
  // NHWC transformation: [1,1,224,224] -> [1,224,224,1] via perm [0,2,3,1]
  // Squeeze on axis=3 removes last dimension: [1,224,224,1] -> [1,224,224]
  // After pushing transpose: axis 3 -> original axis 1
  // Squeeze on axis=1: [1,1,224,224] -> [1,224,224]
  // No transpose needed after squeeze - dimensions already in order!
  // CHECK: %[[AXES:.*]] = onnx.Constant dense<1> : tensor<1xi64>
  // CHECK: "onnx.Squeeze"(%arg0, %[[AXES]]) : (tensor<1x1x224x224xf32>, tensor<1xi64>) -> tensor<1x224x224xf32>
  // CHECK-NOT: onnx.Transpose
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x1x224x224xf32>) -> tensor<1x224x224x1xf32>
  %1 = "onnx.Squeeze"(%0, %axes) : (tensor<1x224x224x1xf32>, tensor<1xi64>) -> tensor<1x224x224xf32>
  return %1 : tensor<1x224x224xf32>
}

// -----
// CHECK-LABEL: func @test_transpose_through_pad
func.func @test_transpose_through_pad(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x230x230x3xf32> {
  %pads = "onnx.Constant"() {value = dense<[0, 3, 3, 0, 0, 3, 3, 0]> : tensor<8xi64>} : () -> tensor<8xi64>
  %const_val = "onnx.Constant"() {value = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
  %axes = "onnx.Constant"() {value = dense<[0, 1, 2, 3]> : tensor<4xi64>} : () -> tensor<4xi64>
  // Input pads for transposed tensor [0,2,3,1]: [0, 3, 3, 0, 0, 3, 3, 0]
  // After pushing transpose with perm [0,2,3,1], pads should transform to original layout: [0, 0, 3, 3, 0, 0, 3, 3]
  // CHECK: %[[PADS:.*]] = onnx.Constant dense<[0, 0, 3, 3, 0, 0, 3, 3]> : tensor<8xi64>
  // CHECK: %[[PAD:.*]] = "onnx.Pad"(%arg0, %[[PADS]]
  // CHECK-SAME: (tensor<1x3x224x224xf32>, tensor<8xi64>, tensor<1xf32>, tensor<4xi64>) -> tensor<1x3x230x230xf32>
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[PAD]]) {perm = [0, 2, 3, 1]}
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.Pad"(%0, %pads, %const_val, %axes) {mode = "constant"} : (tensor<1x224x224x3xf32>, tensor<8xi64>, tensor<1xf32>, tensor<4xi64>) -> tensor<1x230x230x3xf32>
  return %1 : tensor<1x230x230x3xf32>
}

// ============================================================================
// SECTION 7: Complex Real-World Patterns
// ============================================================================

// CHECK-LABEL: func @test_residual_connection
func.func @test_residual_connection(%arg0: tensor<1x64x56x56xf32>) -> tensor<1x56x56x64xf32> {
  // With cloning: 1 input transpose pushed down -> Relus in original space -> 1 output transpose
  // Net: Same number of transposes but operations are in original space
  // CHECK: %[[RELU1:.*]] = "onnx.Relu"(%arg0)
  // CHECK: %[[RELU2:.*]] = "onnx.Relu"(%[[RELU1]])
  // CHECK: %[[ADD:.*]] = "onnx.Add"(%arg0, %[[RELU2]])
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[ADD]]) {perm = [0, 2, 3, 1]}
  // CHECK: return %[[TRANS]]
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x64x56x56xf32>) -> tensor<1x56x56x64xf32>
  %1 = "onnx.Relu"(%0) : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %2 = "onnx.Relu"(%1) : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %3 = "onnx.Add"(%0, %2) : (tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  return %3 : tensor<1x56x56x64xf32>
}

// -----
// CHECK-LABEL: func @test_inception_style_concat
func.func @test_inception_style_concat(%arg0: tensor<1x64x56x56xf32>) -> tensor<1x56x56x192xf32> {
  // With cloning: 1 input transpose cloned 3x -> pushed through Relus -> Concat -> 1 output transpose
  // Net: 1 transpose input -> 1 transpose output (same count, but operations in original space)
  // CHECK: %[[RELU1:.*]] = "onnx.Relu"(%arg0)
  // CHECK: %[[RELU2:.*]] = "onnx.Relu"(%arg0)
  // CHECK: %[[RELU3:.*]] = "onnx.Relu"(%arg0)
  // CHECK: %[[CONCAT:.*]] = "onnx.Concat"(%[[RELU1]], %[[RELU2]], %[[RELU3]]) {axis = 1 : si64}
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[CONCAT]]) {perm = [0, 2, 3, 1]}
  // CHECK: return %[[TRANS]]
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x64x56x56xf32>) -> tensor<1x56x56x64xf32>
  %1 = "onnx.Relu"(%0) : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %2 = "onnx.Relu"(%0) : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %3 = "onnx.Relu"(%0) : (tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
  %4 = "onnx.Concat"(%1, %2, %3) {axis = 3 : si64} : (tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x192xf32>
  return %4 : tensor<1x56x56x192xf32>
}

// -----
// CHECK-LABEL: func @test_slice_concat_pattern
func.func @test_slice_concat_pattern(%arg0: tensor<1x6x224x224xf32>) -> tensor<1x224x224x6xf32> {
  %starts1 = "onnx.Constant"() {value = dense<[0, 0, 0, 0]> : tensor<4xi64>} : () -> tensor<4xi64>
  %ends1 = "onnx.Constant"() {value = dense<[1, 224, 224, 3]> : tensor<4xi64>} : () -> tensor<4xi64>
  %starts2 = "onnx.Constant"() {value = dense<[0, 0, 0, 3]> : tensor<4xi64>} : () -> tensor<4xi64>
  %ends2 = "onnx.Constant"() {value = dense<[1, 224, 224, 6]> : tensor<4xi64>} : () -> tensor<4xi64>
  %axes = "onnx.Constant"() {value = dense<[0, 1, 2, 3]> : tensor<4xi64>} : () -> tensor<4xi64>
  %steps = "onnx.Constant"() {value = dense<[1, 1, 1, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
  // With cloning: transpose cloned 2x -> pushed through Slices (axes transformed) -> Concat -> output transpose
  // The pattern is optimized: Slice axes adjusted, operations in original space
  // CHECK: onnx.Slice
  // CHECK: onnx.Slice
  // CHECK: onnx.Concat
  // CHECK: onnx.Transpose
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x6x224x224xf32>) -> tensor<1x224x224x6xf32>
  %1 = "onnx.Slice"(%0, %starts1, %ends1, %axes, %steps) : (tensor<1x224x224x6xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x224x224x3xf32>
  %2 = "onnx.Slice"(%0, %starts2, %ends2, %axes, %steps) : (tensor<1x224x224x6xf32>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, tensor<4xi64>) -> tensor<1x224x224x3xf32>
  %3 = "onnx.Concat"(%1, %2) {axis = 3 : si64} : (tensor<1x224x224x3xf32>, tensor<1x224x224x3xf32>) -> tensor<1x224x224x6xf32>
  return %3 : tensor<1x224x224x6xf32>
}

// ============================================================================
// SECTION 9: New Rules - Constant Folding, Reshape, Where, Min/Max
// ============================================================================

// -----
// NOTE: Constant folding tests removed - we rely on ONNX-MLIR's --constprop-onnx
// pass for Transpose(Constant) and Const→DequantizeLinear→Transpose patterns.
// This avoids code duplication and leverages well-tested infrastructure.

// -----
// Test: Reshape that changes rank - should NOT fuse
// CHECK-LABEL: func @test_reshape_rank_change_no_fuse
func.func @test_reshape_rank_change_no_fuse(%arg0: tensor<2x3x4xf32>) -> tensor<4x6xf32> {
  %shape = "onnx.Constant"() {value = dense<[4, 6]> : tensor<2xi64>} : () -> tensor<2xi64>
  // Rank-changing Reshape (3D->2D): Cannot fuse - not factorizable
  // CHECK: onnx.Reshape
  // CHECK: onnx.Transpose
  %0 = "onnx.Reshape"(%arg0, %shape) {allowzero = 0 : si64} : (tensor<2x3x4xf32>, tensor<2xi64>) -> tensor<6x4xf32>
  %1 = "onnx.Transpose"(%0) {perm = [1, 0]} : (tensor<6x4xf32>) -> tensor<4x6xf32>
  return %1 : tensor<4x6xf32>
}

// -----
// Test: Cross-dimension merge - NOW SUPPORTED!
// CHECK-LABEL: func @test_move_transpose_through_reshape_merge_dims
func.func @test_move_transpose_through_reshape_merge_dims(%arg0: tensor<2x4x8x16xf32>) -> tensor<2x128x4xf32> {
  // Original: NCHW [2,4,8,16] → Transpose[0,2,3,1] → NHWC [2,8,16,4]
  //           → Reshape[2,128,4] (merges transposed dims 1 and 2: 8*16=128)
  //
  // Algorithm trace:
  //   transposeOutputShape = [2, 8, 16, 4]
  //   reshapeOutputShape = [2, 128, 4]
  //   dimGroups: [2]→[2], [8,16]→[128,-1], [4]→[4]
  //   invPerm = [0, 3, 1, 2]
  //   Pre-transpose shape (in original dim order):
  //     origDim 0 (N=2) at transposed pos 0 → [2]
  //     origDim 1 (C=4) at transposed pos 3 → [4]
  //     origDim 2 (H=8) at transposed pos 1 → [128] (merged with W)
  //     origDim 3 (W=16) at transposed pos 2 → [-1] (skip, merged)
  //   Result: [2, 4, 128] (N, C, H*W)
  //   New perm: [0, 2, 1] to get (N, H*W, C) → [2, 128, 4]

  // CHECK: %[[SHAPE:.*]] = onnx.Constant dense<[2, 4, 128]>
  // CHECK: %[[RESHAPE:.*]] = "onnx.Reshape"(%arg0, %[[SHAPE]])
  // CHECK-SAME: (tensor<2x4x8x16xf32>, tensor<3xi64>) -> tensor<2x4x128xf32>
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[RESHAPE]]) {perm = [0, 2, 1]}
  // CHECK-SAME: (tensor<2x4x128xf32>) -> tensor<2x128x4xf32>
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<2x4x8x16xf32>) -> tensor<2x8x16x4xf32>
  %shape = "onnx.Constant"() {value = dense<[2, 128, 4]> : tensor<3xi64>} : () -> tensor<3xi64>
  %1 = "onnx.Reshape"(%0, %shape) {allowzero = 0 : si64} : (tensor<2x8x16x4xf32>, tensor<3xi64>) -> tensor<2x128x4xf32>
  return %1 : tensor<2x128x4xf32>
}

// -----
// Test: Split single dimension - supported!
// CHECK-LABEL: func @test_move_transpose_through_reshape_split_channel
func.func @test_move_transpose_through_reshape_split_channel(%arg0: tensor<1x64x32x32xf32>) -> tensor<1x32x32x8x8xf32> {
  // Original: [1,64,32,32] NCHW → Transpose[0,2,3,1] → [1,32,32,64] NHWC
  //           → Reshape[1,32,32,8,8] (splits channel dim: 64 → 8*8)
  //
  // Algorithm trace:
  //   transposeOutputShape = [1, 32, 32, 64]
  //   reshapeOutputShape = [1, 32, 32, 8, 8]
  //   dimGroups: [1]→[1], [32]→[32], [32]→[32], [64]→[8,8]
  //   invPerm = [0, 3, 1, 2] (inverse of [0, 2, 3, 1])
  //   Pre-transpose shape (in original dim order):
  //     origDim 0 (N=1) at transposed pos 0 → [1]
  //     origDim 1 (C=64) at transposed pos 3 → [8, 8]
  //     origDim 2 (H=32) at transposed pos 1 → [32]
  //     origDim 3 (W=32) at transposed pos 2 → [32]
  //   Result: [1, 8, 8, 32, 32]
  //   New perm: [0, 3, 4, 1, 2] to get [1, 32, 32, 8, 8]

  // CHECK: %[[SHAPE:.*]] = onnx.Constant dense<[1, 8, 8, 32, 32]>
  // CHECK: %[[RESHAPE:.*]] = "onnx.Reshape"(%arg0, %[[SHAPE]])
  // CHECK-SAME: (tensor<1x64x32x32xf32>, tensor<5xi64>) -> tensor<1x8x8x32x32xf32>
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[RESHAPE]]) {perm = [0, 3, 4, 1, 2]}
  // CHECK-SAME: (tensor<1x8x8x32x32xf32>) -> tensor<1x32x32x8x8xf32>
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x64x32x32xf32>) -> tensor<1x32x32x64xf32>
  %shape = "onnx.Constant"() {value = dense<[1, 32, 32, 8, 8]> : tensor<5xi64>} : () -> tensor<5xi64>
  %1 = "onnx.Reshape"(%0, %shape) {allowzero = 0 : si64} : (tensor<1x32x32x64xf32>, tensor<5xi64>) -> tensor<1x32x32x8x8xf32>
  return %1 : tensor<1x32x32x8x8xf32>
}

// -----
// Test: Unsafe fusion - dimension mixing
// CHECK-LABEL: func @test_no_move_transpose_through_reshape_mixing
func.func @test_no_move_transpose_through_reshape_mixing(%arg0: tensor<2x4x8x16xf32>) -> tensor<8x32x4xf32> {
  // Original: [2,4,8,16] → Transpose[0,2,3,1] → [2,8,16,4]
  //           → Reshape[8,32,4] (merges N*H: 2*8=16, but 16≠8!)
  // This mixes batch and spatial dimensions - NOT SAFE!
  // CHECK: onnx.Transpose
  // CHECK: onnx.Reshape
  // CHECK-NOT: onnx.Transpose
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<2x4x8x16xf32>) -> tensor<2x8x16x4xf32>
  %shape = "onnx.Constant"() {value = dense<[8, 32, 4]> : tensor<3xi64>} : () -> tensor<3xi64>
  %1 = "onnx.Reshape"(%0, %shape) {allowzero = 0 : si64} : (tensor<2x8x16x4xf32>, tensor<3xi64>) -> tensor<8x32x4xf32>
  return %1 : tensor<8x32x4xf32>
}

// -----
// Test: Trailing singleton - NOT supported (doesn't factor cleanly)
// CHECK-LABEL: func @test_no_move_transpose_through_reshape_add_singleton
func.func @test_no_move_transpose_through_reshape_add_singleton(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x224x224x3x1xf32> {
  // Original: [1,3,224,224] → Transpose[0,2,3,1] → [1,224,224,3]
  //           → Reshape[1,224,224,3,1] (adds trailing 1)
  // Trailing singleton doesn't map cleanly to any transposed dim - NOT supported
  // CHECK: onnx.Transpose
  // CHECK: onnx.Reshape
  // CHECK-NOT: onnx.Transpose
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %shape = "onnx.Constant"() {value = dense<[1, 224, 224, 3, 1]> : tensor<5xi64>} : () -> tensor<5xi64>
  %1 = "onnx.Reshape"(%0, %shape) {allowzero = 0 : si64} : (tensor<1x224x224x3xf32>, tensor<5xi64>) -> tensor<1x224x224x3x1xf32>
  return %1 : tensor<1x224x224x3x1xf32>
}

// -----
// Test: Cross-dimension split - NOT supported
// CHECK-LABEL: func @test_no_move_transpose_through_reshape_complex_split
func.func @test_no_move_transpose_through_reshape_complex_split(%arg0: tensor<1x6x32x32xf32>) -> tensor<1x16x16x2x2x6xf32> {
  // Original: [1,6,32,32] → Transpose[0,2,3,1] → [1,32,32,6]
  //           → Reshape[1,16,16,2,2,6]
  // This tries to split 32,32 into 16,16,2,2 which crosses dimension boundaries
  // NOT supported by simple factorization
  // CHECK: onnx.Transpose
  // CHECK: onnx.Reshape
  // CHECK-NOT: onnx.Transpose
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x6x32x32xf32>) -> tensor<1x32x32x6xf32>
  %shape = "onnx.Constant"() {value = dense<[1, 16, 16, 2, 2, 6]> : tensor<6xi64>} : () -> tensor<6xi64>
  %1 = "onnx.Reshape"(%0, %shape) {allowzero = 0 : si64} : (tensor<1x32x32x6xf32>, tensor<6xi64>) -> tensor<1x16x16x2x2x6xf32>
  return %1 : tensor<1x16x16x2x2x6xf32>
}

// -----
// Test: Dimension size 1 handling (bug fix verification)
// CHECK-LABEL: func @test_move_transpose_through_reshape_with_size_one_dims
func.func @test_move_transpose_through_reshape_with_size_one_dims(%arg0: tensor<1x64x1x32xf32>) -> tensor<1x1x32x8x8xf32> {
  // Original: [1,64,1,32] → Transpose[0,2,3,1] → [1,1,32,64]
  //           → Reshape[1,1,32,8,8] (splits channel dim: 64 → 8*8)
  // This tests the bug fix for dimension size 1 consumption
  // The algorithm should correctly consume dimensions of size 1

  // After fusion:
  // - Original dim 0 (N=1) at transposed pos 0 → [1]
  // - Original dim 1 (C=64) at transposed pos 3 → [8, 8]
  // - Original dim 2 (H=1) at transposed pos 1 → [1]
  // - Original dim 3 (W=32) at transposed pos 2 → [32]
  // Pre-transpose shape in original order: [1, 8, 8, 1, 32]
  // Then transpose with perm [0, 2, 3, 1] → [0, 3, 4, 1, 2]
  // Result: [1, 1, 32, 8, 8]

  // CHECK: %[[SHAPE:.*]] = onnx.Constant dense<[1, 8, 8, 1, 32]>
  // CHECK: %[[RESHAPE:.*]] = "onnx.Reshape"(%arg0, %[[SHAPE]])
  // CHECK-SAME: (tensor<1x64x1x32xf32>, tensor<5xi64>) -> tensor<1x8x8x1x32xf32>
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[RESHAPE]]) {perm = [0, 3, 4, 1, 2]}
  // CHECK-SAME: (tensor<1x8x8x1x32xf32>) -> tensor<1x1x32x8x8xf32>
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x64x1x32xf32>) -> tensor<1x1x32x64xf32>
  %shape = "onnx.Constant"() {value = dense<[1, 1, 32, 8, 8]> : tensor<5xi64>} : () -> tensor<5xi64>
  %1 = "onnx.Reshape"(%0, %shape) {allowzero = 0 : si64} : (tensor<1x1x32x64xf32>, tensor<5xi64>) -> tensor<1x1x32x8x8xf32>
  return %1 : tensor<1x1x32x8x8xf32>
}

// -----
// Test: Merge consecutive dimensions (H×W)
// CHECK-LABEL: func @test_move_transpose_through_reshape_merge_spatial
func.func @test_move_transpose_through_reshape_merge_spatial(%arg0: tensor<2x64x8x8xf32>) -> tensor<2x64x64xf32> {
  // Original: [2,64,8,8] NCHW → Transpose[0,2,3,1] → [2,8,8,64] NHWC
  //           → Reshape[2,64,64] (merges H and W: 8*8=64)
  //
  // Algorithm trace:
  //   transposeOutputShape = [2, 8, 8, 64]
  //   reshapeOutputShape = [2, 64, 64]
  //   Iteration 1: 2==2 identity → dimGroups[0] = [2]
  //   Iteration 2: 8<64 merge → accumulate 8*8=64 → dimGroups[1] = [64], dimGroups[2] = [-1]
  //   Iteration 4: 64==64 identity → dimGroups[3] = [64]
  //
  // Pre-transpose reshape:
  //   invPerm = [0, 3, 1, 2]
  //   origDim 0 (N) → transposed pos 0 → [2]
  //   origDim 1 (C) → transposed pos 3 → [64]
  //   origDim 2 (H) → transposed pos 1 → [64] (merged with W)
  //   origDim 3 (W) → transposed pos 2 → [-1] (skip)
  //   Pre-transpose shape: [2, 64, 64] (N, C, H*W)
  //
  // New permutation: [0, 2, 1] to get (N, H*W, C)

  // CHECK: %[[SHAPE:.*]] = onnx.Constant dense<[2, 64, 64]>
  // CHECK: %[[RESHAPE:.*]] = "onnx.Reshape"(%arg0, %[[SHAPE]])
  // CHECK-SAME: (tensor<2x64x8x8xf32>, tensor<3xi64>) -> tensor<2x64x64xf32>
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[RESHAPE]]) {perm = [0, 2, 1]}
  // CHECK-SAME: (tensor<2x64x64xf32>) -> tensor<2x64x64xf32>
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<2x64x8x8xf32>) -> tensor<2x8x8x64xf32>
  %shape = "onnx.Constant"() {value = dense<[2, 64, 64]> : tensor<3xi64>} : () -> tensor<3xi64>
  %1 = "onnx.Reshape"(%0, %shape) {allowzero = 0 : si64} : (tensor<2x8x8x64xf32>, tensor<3xi64>) -> tensor<2x64x64xf32>
  return %1 : tensor<2x64x64xf32>
}

// -----
// CHECK-LABEL: func @test_transpose_through_clip
func.func @test_transpose_through_clip(%arg0: tensor<1x3x224x224xf32>, %min: tensor<f32>, %max: tensor<f32>) -> tensor<1x224x224x3xf32> {
  // CHECK: %[[CLIP:.*]] = "onnx.Clip"(%arg0
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[CLIP]]) {perm = [0, 2, 3, 1]}
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.Clip"(%0, %min, %max) : (tensor<1x224x224x3xf32>, tensor<f32>, tensor<f32>) -> tensor<1x224x224x3xf32>
  return %1 : tensor<1x224x224x3xf32>
}

// -----
// CHECK-LABEL: func @test_transpose_through_hardsigmoid
func.func @test_transpose_through_hardsigmoid(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32> {
  // CHECK: %[[HS:.*]] = "onnx.HardSigmoid"(%arg0
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[HS]]) {perm = [0, 2, 3, 1]}
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.HardSigmoid"(%0) {alpha = 0.2 : f32, beta = 0.5 : f32} : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  return %1 : tensor<1x224x224x3xf32>
}

// -----
// CHECK-LABEL: func @test_where_both_transposed
func.func @test_where_both_transposed(%arg0: tensor<1x3x224x224xi1>, %arg1: tensor<1x3x224x224xf32>, %arg2: tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32> {
  // Pattern should push Where before transposes (condition also transposed)
  // CHECK: %[[WHERE:.*]] = "onnx.Where"(%arg0, %arg1, %arg2)
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[WHERE]]) {perm = [0, 2, 3, 1]}
  // CHECK: return %[[TRANS]]
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xi1>) -> tensor<1x224x224x3xi1>
  %1 = "onnx.Transpose"(%arg1) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %2 = "onnx.Transpose"(%arg2) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %3 = "onnx.Where"(%0, %1, %2) : (tensor<1x224x224x3xi1>, tensor<1x224x224x3xf32>, tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  return %3 : tensor<1x224x224x3xf32>
}

// -----
// CHECK-LABEL: func @test_where_with_transposed_condition
func.func @test_where_with_transposed_condition(%arg0: tensor<1x3x224x224xi1>, %arg1: tensor<1x3x224x224xf32>, %arg2: tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32> {
  // All three inputs transposed - should fuse
  // CHECK: %[[WHERE:.*]] = "onnx.Where"(%arg0, %arg1, %arg2)
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[WHERE]]) {perm = [0, 2, 3, 1]}
  // CHECK: return %[[TRANS]]
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xi1>) -> tensor<1x224x224x3xi1>
  %1 = "onnx.Transpose"(%arg1) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %2 = "onnx.Transpose"(%arg2) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %3 = "onnx.Where"(%0, %1, %2) : (tensor<1x224x224x3xi1>, tensor<1x224x224x3xf32>, tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  return %3 : tensor<1x224x224x3xf32>
}

// -----
// CHECK-LABEL: func @test_min_with_transposed_and_scalar
func.func @test_min_with_transposed_and_scalar(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32> {
  %const = "onnx.Constant"() {value = dense<6.0> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
  // Scalar constant (1x1x1x1) is transpose-immune - used directly without Reshape
  // CHECK: %[[CONST:.*]] = onnx.Constant dense<6.000000e+00>
  // CHECK: %[[MIN:.*]] = "onnx.Min"(%arg0, %[[CONST]])
  // CHECK-NOT: onnx.Reshape
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[MIN]]) {perm = [0, 2, 3, 1]}
  // CHECK: return %[[TRANS]]
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.Min"(%0, %const) : (tensor<1x224x224x3xf32>, tensor<1x1x1x1xf32>) -> tensor<1x224x224x3xf32>
  return %1 : tensor<1x224x224x3xf32>
}

// -----
// CHECK-LABEL: func @test_max_with_transposed_and_scalar
func.func @test_max_with_transposed_and_scalar(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32> {
  %const = "onnx.Constant"() {value = dense<0.0> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
  // Scalar constant (1x1x1x1) is transpose-immune - used directly without Reshape
  // CHECK: %[[CONST:.*]] = onnx.Constant dense<0.000000e+00>
  // CHECK: %[[MAX:.*]] = "onnx.Max"(%arg0, %[[CONST]])
  // CHECK-NOT: onnx.Reshape
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[MAX]]) {perm = [0, 2, 3, 1]}
  // CHECK: return %[[TRANS]]
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.Max"(%0, %const) : (tensor<1x224x224x3xf32>, tensor<1x1x1x1xf32>) -> tensor<1x224x224x3xf32>
  return %1 : tensor<1x224x224x3xf32>
}

// -----
// CHECK-LABEL: func @test_min_two_transposed_inputs
func.func @test_min_two_transposed_inputs(%arg0: tensor<1x3x224x224xf32>, %arg1: tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32> {
  // Both inputs transposed - should fuse
  // CHECK: %[[MIN:.*]] = "onnx.Min"(%arg0, %arg1)
  // CHECK-SAME: (tensor<1x3x224x224xf32>, tensor<1x3x224x224xf32>) -> tensor<1x3x224x224xf32>
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[MIN]]) {perm = [0, 2, 3, 1]}
  // CHECK: return %[[TRANS]]
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.Transpose"(%arg1) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %2 = "onnx.Min"(%0, %1) : (tensor<1x224x224x3xf32>, tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  return %2 : tensor<1x224x224x3xf32>
}

// -----
// CHECK-LABEL: func @test_max_three_transposed_inputs
func.func @test_max_three_transposed_inputs(%arg0: tensor<1x3x224x224xf32>, %arg1: tensor<1x3x224x224xf32>, %arg2: tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32> {
  // All three inputs transposed - should fuse
  // CHECK: %[[MAX:.*]] = "onnx.Max"(%arg0, %arg1, %arg2)
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[MAX]]) {perm = [0, 2, 3, 1]}
  // CHECK: return %[[TRANS]]
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.Transpose"(%arg1) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %2 = "onnx.Transpose"(%arg2) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %3 = "onnx.Max"(%0, %1, %2) : (tensor<1x224x224x3xf32>, tensor<1x224x224x3xf32>, tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  return %3 : tensor<1x224x224x3xf32>
}

// -----
// CHECK-LABEL: func @test_constant_folding_with_relu
func.func @test_constant_folding_with_relu() -> tensor<3x2xf32> {
  %const = "onnx.Constant"() {value = dense<[[-1.0, 2.0, 3.0], [4.0, -5.0, 6.0]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  // Transpose pushed through Relu (constant folding delegated to ONNX-MLIR --constprop-onnx)
  // CHECK: %[[CONST:.*]] = onnx.Constant dense<{{\[}}[-1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, -5.000000e+00, 6.000000e+00]]> : tensor<2x3xf32>
  // CHECK: %[[RELU:.*]] = "onnx.Relu"(%[[CONST]])
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[RELU]]) {perm = [1, 0]}
  // CHECK: return %[[TRANS]]
  %0 = "onnx.Transpose"(%const) {perm = [1, 0]} : (tensor<2x3xf32>) -> tensor<3x2xf32>
  %1 = "onnx.Relu"(%0) : (tensor<3x2xf32>) -> tensor<3x2xf32>
  return %1 : tensor<3x2xf32>
}

// -----
// CHECK-LABEL: func @test_clip_chain
func.func @test_clip_chain(%arg0: tensor<1x3x224x224xf32>, %min: tensor<f32>, %max: tensor<f32>) -> tensor<1x224x224x3xf32> {
  // CHECK: %[[RELU:.*]] = "onnx.Relu"(%arg0)
  // CHECK: %[[CLIP:.*]] = "onnx.Clip"(%[[RELU]]
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[CLIP]]) {perm = [0, 2, 3, 1]}
  // CHECK: return %[[TRANS]]
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.Relu"(%0) : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  %2 = "onnx.Clip"(%1, %min, %max) : (tensor<1x224x224x3xf32>, tensor<f32>, tensor<f32>) -> tensor<1x224x224x3xf32>
  return %2 : tensor<1x224x224x3xf32>
}

// -----
// CHECK-LABEL: func @test_hardsigmoid_chain
func.func @test_hardsigmoid_chain(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32> {
  // CHECK: %[[SIGMOID:.*]] = "onnx.Sigmoid"(%arg0)
  // CHECK: %[[HS:.*]] = "onnx.HardSigmoid"(%[[SIGMOID]]
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[HS]]) {perm = [0, 2, 3, 1]}
  // CHECK: return %[[TRANS]]
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.Sigmoid"(%0) : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  %2 = "onnx.HardSigmoid"(%1) {alpha = 0.2 : f32, beta = 0.5 : f32} : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  return %2 : tensor<1x224x224x3xf32>
}

// -----
// CHECK-LABEL: func @test_min_max_clamp_pattern
func.func @test_min_max_clamp_pattern(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32> {
  %min_const = "onnx.Constant"() {value = dense<0.0> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
  %max_const = "onnx.Constant"() {value = dense<6.0> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
  // Transpose pushed down - transpose-immune constants (1x1x1x1) used directly without Reshape
  // CHECK: %[[MAX:.*]] = "onnx.Max"(%arg0
  // CHECK-NOT: onnx.Reshape
  // CHECK: %[[MIN:.*]] = "onnx.Min"(%[[MAX]]
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[MIN]]) {perm = [0, 2, 3, 1]}
  // CHECK: return %[[TRANS]]
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.Max"(%0, %min_const) : (tensor<1x224x224x3xf32>, tensor<1x1x1x1xf32>) -> tensor<1x224x224x3xf32>
  %2 = "onnx.Min"(%1, %max_const) : (tensor<1x224x224x3xf32>, tensor<1x1x1x1xf32>) -> tensor<1x224x224x3xf32>
  return %2 : tensor<1x224x224x3xf32>
}

// -----
// CHECK-LABEL: func @test_where_add_pattern
func.func @test_where_add_pattern(%arg0: tensor<1x3x224x224xi1>, %arg1: tensor<1x3x224x224xf32>, %arg2: tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32> {
  // All inputs transposed including condition - should fuse
  // CHECK: %[[WHERE:.*]] = "onnx.Where"(%arg0, %arg1, %arg2)
  // CHECK: %[[RELU:.*]] = "onnx.Relu"(%[[WHERE]])
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[RELU]]) {perm = [0, 2, 3, 1]}
  // CHECK: return %[[TRANS]]
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xi1>) -> tensor<1x224x224x3xi1>
  %1 = "onnx.Transpose"(%arg1) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %2 = "onnx.Transpose"(%arg2) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %3 = "onnx.Where"(%0, %1, %2) : (tensor<1x224x224x3xi1>, tensor<1x224x224x3xf32>, tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  %4 = "onnx.Relu"(%3) : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  return %4 : tensor<1x224x224x3xf32>
}

// ============================================================================
// SECTION 10: Rank-Changing Operations (NEW - Generic Pattern Tests)
// ============================================================================

// -----
// CHECK-LABEL: func @test_squeeze_rank_change_single_axis
func.func @test_squeeze_rank_change_single_axis(%arg0: tensor<2x1x224x224xf32>) -> tensor<2x224x224xf32> {
  %axes = "onnx.Constant"() {value = dense<[3]> : tensor<1xi64>} : () -> tensor<1xi64>
  // NHWC transformation: [2,1,224,224] -> [2,224,224,1] via perm [0,2,3,1]
  // Squeeze on axis=3 (channel=1): [2,224,224,1] -> [2,224,224]
  // After pushing transpose: axis 3 -> original axis 1
  // Squeeze on axis=1: [2,1,224,224] -> [2,224,224]
  // No transpose needed! Dimensions already in order
  // CHECK: %[[AXES:.*]] = onnx.Constant dense<1> : tensor<1xi64>
  // CHECK: "onnx.Squeeze"(%arg0, %[[AXES]]) : (tensor<2x1x224x224xf32>, tensor<1xi64>) -> tensor<2x224x224xf32>
  // CHECK-NOT: onnx.Transpose
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<2x1x224x224xf32>) -> tensor<2x224x224x1xf32>
  %1 = "onnx.Squeeze"(%0, %axes) : (tensor<2x224x224x1xf32>, tensor<1xi64>) -> tensor<2x224x224xf32>
  return %1 : tensor<2x224x224xf32>
}

// -----
// CHECK-LABEL: func @test_squeeze_rank_change_multiple_axes
func.func @test_squeeze_rank_change_multiple_axes(%arg0: tensor<1x3x1x1xf32>) -> tensor<1x3xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64>} : () -> tensor<2xi64>
  // NHWC transformation: [1,3,1,1] -> [1,1,1,3] via perm [0,2,3,1]
  // Squeeze on axes [0,1]: [1,1,1,3] -> [1,3] (removes first two dimensions of size 1)
  // After pushing transpose: axes [0,1] in transposed space -> original axes [0,2]
  // Squeeze on axes [0,2]: [1,3,1,1] -> [3,1]
  // Final transpose: applying reduced perm [1,0] to swap remaining dims -> [1,3]
  // CHECK: %[[AXES:.*]] = onnx.Constant dense<[0, 2]> : tensor<2xi64>
  // CHECK: %[[SQUEEZE:.*]] = "onnx.Squeeze"(%arg0, %[[AXES]]) : (tensor<1x3x1x1xf32>, tensor<2xi64>) -> tensor<3x1xf32>
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[SQUEEZE]]) {perm = [1, 0]}
  // CHECK-SAME: (tensor<3x1xf32>) -> tensor<1x3xf32>
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x1x1xf32>) -> tensor<1x1x1x3xf32>
  %1 = "onnx.Squeeze"(%0, %axes) : (tensor<1x1x1x3xf32>, tensor<2xi64>) -> tensor<1x3xf32>
  return %1 : tensor<1x3xf32>
}

// -----
// CHECK-LABEL: func @test_argmax_rank_change_keepdims_zero
func.func @test_argmax_rank_change_keepdims_zero(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x224x224xi64> {
  // NHWC transformation: ArgMax with keepdims=0 reduces rank by 1
  // Transpose [0,2,3,1], then ArgMax on axis 3 (channel, original axis 1)
  // Reduced permutation: remove axis 3 from [0,2,3,1] -> identity [0,1,2]
  // No transpose needed after reduction!
  // CHECK: %[[ARGMAX:.*]] = "onnx.ArgMax"(%arg0) {axis = 1 : si64, keepdims = 0 : si64, select_last_index = 0 : si64}
  // CHECK-SAME: (tensor<1x3x224x224xf32>) -> tensor<1x224x224xi64>
  // CHECK-NOT: onnx.Transpose
  // CHECK: return %[[ARGMAX]]
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.ArgMax"(%0) {axis = 3 : si64, keepdims = 0 : si64, select_last_index = 0 : si64} : (tensor<1x224x224x3xf32>) -> tensor<1x224x224xi64>
  return %1 : tensor<1x224x224xi64>
}

// -----
// CHECK-LABEL: func @test_argmax_keepdims_one_no_rank_change
func.func @test_argmax_keepdims_one_no_rank_change(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x224x224x1xi64> {
  // NHWC transformation: ArgMax with keepdims=1 does NOT change rank - full permutation used
  // CHECK: %[[ARGMAX:.*]] = "onnx.ArgMax"(%arg0) {axis = 1 : si64, keepdims = 1 : si64, select_last_index = 0 : si64}
  // CHECK-SAME: (tensor<1x3x224x224xf32>) -> tensor<1x1x224x224xi64>
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[ARGMAX]]) {perm = [0, 2, 3, 1]}
  // CHECK-SAME: (tensor<1x1x224x224xi64>) -> tensor<1x224x224x1xi64>
  // CHECK: return %[[TRANS]]
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.ArgMax"(%0) {axis = 3 : si64, keepdims = 1 : si64, select_last_index = 0 : si64} : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x1xi64>
  return %1 : tensor<1x224x224x1xi64>
}

// -----
// CHECK-LABEL: func @test_reduce_mean_rank_change_keepdims_zero
func.func @test_reduce_mean_rank_change_keepdims_zero(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x224x224xf32> {
  %axes = "onnx.Constant"() {value = dense<[3]> : tensor<1xi64>} : () -> tensor<1xi64>
  // NHWC transformation: ReduceMean with keepdims=0 reduces rank by 1
  // Transpose [0,2,3,1], reduce on axis 3 (channel, original axis 1)
  // After reduction, shape becomes 3D, permutation [0,2,3,1] -> identity [0,1,2]
  // No transpose needed after reduction!
  // CHECK: %[[AXES:.*]] = onnx.Constant dense<1> : tensor<1xi64>
  // CHECK: "onnx.ReduceMean"(%arg0, %[[AXES]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
  // CHECK-SAME: (tensor<1x3x224x224xf32>, tensor<1xi64>) -> tensor<1x224x224xf32>
  // CHECK-NOT: onnx.Transpose
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.ReduceMean"(%0, %axes) {keepdims = 0 : si64} : (tensor<1x224x224x3xf32>, tensor<1xi64>) -> tensor<1x224x224xf32>
  return %1 : tensor<1x224x224xf32>
}

// -----
// CHECK-LABEL: func @test_reduce_sum_multi_axis_rank_change
func.func @test_reduce_sum_multi_axis_rank_change(%arg0: tensor<2x3x4x5xf32>) -> tensor<2x3xf32> {
  %axes = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  // ReduceSum with keepdims=0 on 2 axes reduces rank by 2
  // NHWC transformation: NCHW [2,3,4,5] -> NHWC [2,4,5,3] via perm [0,2,3,1]
  // Reduce on axes [1,2] in NHWC removes H,W dims: [2,4,5,3] -> [2,3]
  // After pushing transpose: reduce original axes [2,3] from [2,3,4,5] -> [2,3]
  // No final transpose needed since output dimensions are already in correct order!
  // CHECK: %[[AXES:.*]] = onnx.Constant dense<[2, 3]> : tensor<2xi64>
  // CHECK: "onnx.ReduceSum"(%arg0, %[[AXES]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}
  // CHECK-SAME: (tensor<2x3x4x5xf32>, tensor<2xi64>) -> tensor<2x3xf32>
  // CHECK-NOT: onnx.Transpose
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<2x3x4x5xf32>) -> tensor<2x4x5x3xf32>
  %1 = "onnx.ReduceSum"(%0, %axes) {keepdims = 0 : si64} : (tensor<2x4x5x3xf32>, tensor<2xi64>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}

// -----
// CHECK-LABEL: func @test_reduce_max_keepdims_one_no_rank_change
func.func @test_reduce_max_keepdims_one_no_rank_change(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x224x224x1xf32> {
  %axes = "onnx.Constant"() {value = dense<[3]> : tensor<1xi64>} : () -> tensor<1xi64>
  // NHWC transformation: ReduceMax with keepdims=1 does NOT change rank - full permutation used
  // Reduce on channel axis=3
  // CHECK: %[[AXES:.*]] = onnx.Constant dense<1> : tensor<1xi64>
  // CHECK: %[[REDUCE:.*]] = "onnx.ReduceMax"(%arg0, %[[AXES]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}
  // CHECK-SAME: (tensor<1x3x224x224xf32>, tensor<1xi64>) -> tensor<1x1x224x224xf32>
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[REDUCE]]) {perm = [0, 2, 3, 1]}
  // CHECK-SAME: (tensor<1x1x224x224xf32>) -> tensor<1x224x224x1xf32>
  // CHECK: return %[[TRANS]]
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.ReduceMax"(%0, %axes) {keepdims = 1 : si64} : (tensor<1x224x224x3xf32>, tensor<1xi64>) -> tensor<1x224x224x1xf32>
  return %1 : tensor<1x224x224x1xf32>
}


// -----
// CHECK-LABEL: func @test_concat_axis_transformation
func.func @test_concat_axis_transformation(%arg0: tensor<1x3x224x224xf32>, %arg1: tensor<1x5x224x224xf32>) -> tensor<1x224x224x8xf32> {
  // Concat on axis 3 after transpose [0,2,3,1] should become axis 1 before transpose
  // CHECK: %[[CONCAT:.*]] = "onnx.Concat"(%arg0, %arg1) {axis = 1 : si64}
  // CHECK-SAME: (tensor<1x3x224x224xf32>, tensor<1x5x224x224xf32>) -> tensor<1x8x224x224xf32>
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[CONCAT]]) {perm = [0, 2, 3, 1]}
  // CHECK-SAME: (tensor<1x8x224x224xf32>) -> tensor<1x224x224x8xf32>
  // CHECK: return %[[TRANS]]
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.Transpose"(%arg1) {perm = [0, 2, 3, 1]} : (tensor<1x5x224x224xf32>) -> tensor<1x224x224x5xf32>
  %2 = "onnx.Concat"(%0, %1) {axis = 3 : si64} : (tensor<1x224x224x3xf32>, tensor<1x224x224x5xf32>) -> tensor<1x224x224x8xf32>
  return %2 : tensor<1x224x224x8xf32>
}

// -----
// CHECK-LABEL: func @test_multiuse_transpose_consecutive_cancel
func.func @test_multiuse_transpose_consecutive_cancel(%arg0: tensor<1x40x64x64xf32>) -> (tensor<1x64x64x40xf32>, tensor<1x40x64x64xf32>) {
  // Transpose NCHW->NHWC, used by both consecutive transpose (cancel) and Relu
  // CHECK: %[[T1:.*]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]}
  // CHECK-SAME: (tensor<1x40x64x64xf32>) -> tensor<1x64x64x40xf32>
  // CHECK: %[[RELU:.*]] = "onnx.Relu"(%arg0)
  // CHECK: return %[[T1]], %[[RELU]]
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x40x64x64xf32>) -> tensor<1x64x64x40xf32>
  %1 = "onnx.Transpose"(%0) {perm = [0, 3, 1, 2]} : (tensor<1x64x64x40xf32>) -> tensor<1x40x64x64xf32>
  %2 = "onnx.Relu"(%1) : (tensor<1x40x64x64xf32>) -> tensor<1x40x64x64xf32>
  return %0, %2 : tensor<1x64x64x40xf32>, tensor<1x40x64x64xf32>
}

// -----
// CHECK-LABEL: func @test_multiuse_transpose_residual_pattern
func.func @test_multiuse_transpose_residual_pattern(%arg0: tensor<1x40x64x64x!quant.uniform<u8:f32, 6.250000e-02>>) -> (tensor<1x40x64x64x!quant.uniform<u8:f32, 1.250000e-01>>, tensor<1x64x64x40x!quant.uniform<u8:f32, 6.250000e-02>>) {
  // Residual pattern with multi-use: Relu pushed through, Add in original space
  // CHECK: %[[T1:.*]] = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]}
  // CHECK: %[[RELU:.*]] = "onnx.Relu"(%arg0)
  // CHECK: %[[ADD:.*]] = "onnx.Add"(%arg0, %[[RELU]])
  // CHECK: %[[T2:.*]] = "onnx.Transpose"(%[[ADD]]) {perm = [0, 2, 3, 1]}
  // CHECK: return %[[T2]], %[[T1]]
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x40x64x64x!quant.uniform<u8:f32, 6.250000e-02>>) -> tensor<1x64x64x40x!quant.uniform<u8:f32, 6.250000e-02>>
  %1 = "onnx.Transpose"(%0) {perm = [0, 3, 1, 2]} : (tensor<1x64x64x40x!quant.uniform<u8:f32, 6.250000e-02>>) -> tensor<1x40x64x64x!quant.uniform<u8:f32, 6.250000e-02>>
  %2 = "onnx.Relu"(%1) : (tensor<1x40x64x64x!quant.uniform<u8:f32, 6.250000e-02>>) -> tensor<1x40x64x64x!quant.uniform<u8:f32, 6.250000e-02>>
  %3 = "onnx.Transpose"(%2) {perm = [0, 2, 3, 1]} : (tensor<1x40x64x64x!quant.uniform<u8:f32, 6.250000e-02>>) -> tensor<1x64x64x40x!quant.uniform<u8:f32, 6.250000e-02>>
  %4 = "onnx.Add"(%0, %3) : (tensor<1x64x64x40x!quant.uniform<u8:f32, 6.250000e-02>>, tensor<1x64x64x40x!quant.uniform<u8:f32, 6.250000e-02>>) -> tensor<1x40x64x64x!quant.uniform<u8:f32, 1.250000e-01>>
  return %4, %0 : tensor<1x40x64x64x!quant.uniform<u8:f32, 1.250000e-01>>, tensor<1x64x64x40x!quant.uniform<u8:f32, 6.250000e-02>>
}

// -----
// CHECK-LABEL: func @test_multiuse_transpose_binary_ops
func.func @test_multiuse_transpose_binary_ops(%arg0: tensor<1x3x224x224xf32>, %arg1: tensor<1x3x224x224xf32>) -> (tensor<1x224x224x3xf32>, tensor<1x224x224x3xf32>) {
  // Transpose used by both Add and Mul operations
  // Pattern should clone and push through both
  // CHECK: %[[ADD:.*]] = "onnx.Add"(%arg0, %arg1)
  // CHECK: %[[T1:.*]] = "onnx.Transpose"(%[[ADD]]) {perm = [0, 2, 3, 1]}
  // CHECK: %[[MUL:.*]] = "onnx.Mul"(%arg0, %arg1)
  // CHECK: %[[T2:.*]] = "onnx.Transpose"(%[[MUL]]) {perm = [0, 2, 3, 1]}
  // CHECK: return %[[T1]], %[[T2]]
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.Transpose"(%arg1) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<1x224x224x3xf32>, tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  %3 = "onnx.Mul"(%0, %1) : (tensor<1x224x224x3xf32>, tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  return %2, %3 : tensor<1x224x224x3xf32>, tensor<1x224x224x3xf32>
}

// -----
// CHECK-LABEL: func @test_multiuse_transpose_triple_branch
func.func @test_multiuse_transpose_triple_branch(%arg0: tensor<1x3x224x224xf32>) -> (tensor<1x224x224x3xf32>, tensor<1x224x224x3xf32>, tensor<1x224x224x3xf32>) {
  // Transpose used by three different operations
  // Pattern should clone and push through all branches
  // CHECK: %[[RELU:.*]] = "onnx.Relu"(%arg0)
  // CHECK: %[[T1:.*]] = "onnx.Transpose"(%[[RELU]]) {perm = [0, 2, 3, 1]}
  // CHECK: %[[SIGMOID:.*]] = "onnx.Sigmoid"(%arg0)
  // CHECK: %[[T2:.*]] = "onnx.Transpose"(%[[SIGMOID]]) {perm = [0, 2, 3, 1]}
  // CHECK: %[[TANH:.*]] = "onnx.Tanh"(%arg0)
  // CHECK: %[[T3:.*]] = "onnx.Transpose"(%[[TANH]]) {perm = [0, 2, 3, 1]}
  // CHECK: return %[[T1]], %[[T2]], %[[T3]]
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.Relu"(%0) : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  %2 = "onnx.Sigmoid"(%0) : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  %3 = "onnx.Tanh"(%0) : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  return %1, %2, %3 : tensor<1x224x224x3xf32>, tensor<1x224x224x3xf32>, tensor<1x224x224x3xf32>
}

// -----
// CHECK-LABEL: func @test_multiuse_transpose_qdq_chain
func.func @test_multiuse_transpose_qdq_chain(%arg0: tensor<1x3x224x224xf32>, %scale: tensor<f32>, %zp: tensor<ui8>) -> (tensor<1x224x224x3x!quant.uniform<u8:f32, 1.562500e-02>>, tensor<1x224x224x3xf32>) {
  // Transpose used by both QuantizeLinear and Relu - pushed through both
  // CHECK: %[[Q:.*]] = "onnx.QuantizeLinear"(%arg0, %{{.*}}, %{{.*}})
  // CHECK: %[[T1:.*]] = "onnx.Transpose"(%[[Q]]) {perm = [0, 2, 3, 1]}
  // CHECK: %[[RELU:.*]] = "onnx.Relu"(%arg0)
  // CHECK: %[[T2:.*]] = "onnx.Transpose"(%[[RELU]]) {perm = [0, 2, 3, 1]}
  // CHECK: return %[[T1]], %[[T2]]
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.QuantizeLinear"(%0, %scale, %zp) : (tensor<1x224x224x3xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x224x224x3x!quant.uniform<u8:f32, 1.562500e-02>>
  %2 = "onnx.Relu"(%0) : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  return %1, %2 : tensor<1x224x224x3x!quant.uniform<u8:f32, 1.562500e-02>>, tensor<1x224x224x3xf32>
}

// -----
// CHECK-LABEL: func @test_multiuse_transpose_where_branches
func.func @test_multiuse_transpose_where_branches(%cond: tensor<1x3x224x224xi1>, %arg0: tensor<1x3x224x224xf32>, %arg1: tensor<1x3x224x224xf32>) -> (tensor<1x224x224x3xf32>, tensor<1x224x224x3xf32>) {
  // Where + Add pattern: operations pushed to original space, transposes after
  // CHECK: "onnx.Where"
  // CHECK: "onnx.Transpose"
  // CHECK: "onnx.Add"
  // CHECK: "onnx.Transpose"
  %cond_t = "onnx.Transpose"(%cond) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xi1>) -> tensor<1x224x224x3xi1>
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.Transpose"(%arg1) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %2 = "onnx.Where"(%cond_t, %0, %1) : (tensor<1x224x224x3xi1>, tensor<1x224x224x3xf32>, tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  %3 = "onnx.Add"(%0, %1) : (tensor<1x224x224x3xf32>, tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
  return %2, %3 : tensor<1x224x224x3xf32>, tensor<1x224x224x3xf32>
}

// -----
// CHECK-LABEL: func @test_multiuse_transpose_with_constant
func.func @test_multiuse_transpose_with_constant(%arg0: tensor<1x3x224x224xf32>) -> (tensor<1x224x224x3xf32>, tensor<1x224x224x3xf32>) {
  // Transpose used by operations with transpose-immune constant (1x1x1x1) - no Reshape needed!
  %const = "onnx.Constant"() {value = dense<2.0> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
  // CHECK: %[[CONST:.*]] = onnx.Constant dense<2.000000e+00>
  // CHECK-NOT: onnx.Reshape
  // CHECK: %[[MUL:.*]] = "onnx.Mul"(%arg0, %[[CONST]])
  // CHECK: %[[T1:.*]] = "onnx.Transpose"(%[[MUL]]) {perm = [0, 2, 3, 1]}
  // CHECK: %[[ADD:.*]] = "onnx.Add"(%arg0, %[[CONST]])
  // CHECK: %[[T2:.*]] = "onnx.Transpose"(%[[ADD]]) {perm = [0, 2, 3, 1]}
  // CHECK: return %[[T1]], %[[T2]]
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x224x224x3xf32>
  %1 = "onnx.Mul"(%0, %const) : (tensor<1x224x224x3xf32>, tensor<1x1x1x1xf32>) -> tensor<1x224x224x3xf32>
  %2 = "onnx.Add"(%0, %const) : (tensor<1x224x224x3xf32>, tensor<1x1x1x1xf32>) -> tensor<1x224x224x3xf32>
  return %1, %2 : tensor<1x224x224x3xf32>, tensor<1x224x224x3xf32>
}

// -----
// CHECK-LABEL: func @test_multiuse_xfeconv_residual_realistic
func.func @test_multiuse_xfeconv_residual_realistic(%arg0: tensor<1x64x64x40x!quant.uniform<u8:f32, 6.250000e-02>>) -> (tensor<1x40x64x64x!quant.uniform<u8:f32, 1.250000e-01>>, tensor<1x64x64x40x!quant.uniform<u8:f32, 6.250000e-02>>) {
  // Realistic pattern: Relu + Add in NHWC space, with NCHW->NHWC transpose for output
  // CHECK: %[[RELU:.*]] = "onnx.Relu"(%arg0)
  // CHECK: %[[ADD:.*]] = "onnx.Add"(%arg0, %[[RELU]])
  // CHECK: %[[T:.*]] = "onnx.Transpose"(%[[ADD]]) {perm = [0, 3, 1, 2]}
  // CHECK: return %[[T]], %arg0
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 3, 1, 2]} : (tensor<1x64x64x40x!quant.uniform<u8:f32, 6.250000e-02>>) -> tensor<1x40x64x64x!quant.uniform<u8:f32, 6.250000e-02>>
  %1 = "onnx.Transpose"(%0) {perm = [0, 2, 3, 1]} : (tensor<1x40x64x64x!quant.uniform<u8:f32, 6.250000e-02>>) -> tensor<1x64x64x40x!quant.uniform<u8:f32, 6.250000e-02>>
  %2 = "onnx.Relu"(%1) : (tensor<1x64x64x40x!quant.uniform<u8:f32, 6.250000e-02>>) -> tensor<1x64x64x40x!quant.uniform<u8:f32, 6.250000e-02>>
  %3 = "onnx.Transpose"(%2) {perm = [0, 3, 1, 2]} : (tensor<1x64x64x40x!quant.uniform<u8:f32, 6.250000e-02>>) -> tensor<1x40x64x64x!quant.uniform<u8:f32, 6.250000e-02>>
  %4 = "onnx.Add"(%0, %3) : (tensor<1x40x64x64x!quant.uniform<u8:f32, 6.250000e-02>>, tensor<1x40x64x64x!quant.uniform<u8:f32, 6.250000e-02>>) -> tensor<1x40x64x64x!quant.uniform<u8:f32, 1.250000e-01>>
  return %4, %1 : tensor<1x40x64x64x!quant.uniform<u8:f32, 1.250000e-01>>, tensor<1x64x64x40x!quant.uniform<u8:f32, 6.250000e-02>>
}

// -----
// ==========================================================================
// Push Transpose Through quant.scast Tests
// Rule: scast(transpose(x)) -> transpose(scast(x))
// ==========================================================================

// Test: Push transpose through scast (quantized -> storage)
// CHECK-LABEL: func @test_push_transpose_through_scast_to_storage
func.func @test_push_transpose_through_scast_to_storage(%arg0: tensor<1x3x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x4x4x3xi8> {
  // CHECK: quant.scast %arg0 : tensor<1x3x4x4x!quant.uniform<i8:f32, 5.000000e-02>> to tensor<1x3x4x4xi8>
  // CHECK: "onnx.Transpose"
  // CHECK-SAME: perm = [0, 2, 3, 1]
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x4x4x3x!quant.uniform<i8:f32, 0.05:0>>
  %1 = quant.scast %0 : tensor<1x4x4x3x!quant.uniform<i8:f32, 0.05:0>> to tensor<1x4x4x3xi8>
  return %1 : tensor<1x4x4x3xi8>
}

// -----

// Test: Push transpose through scast (storage -> quantized)
// CHECK-LABEL: func @test_push_transpose_through_scast_to_quant
func.func @test_push_transpose_through_scast_to_quant(%arg0: tensor<1x3x4x4xi8>) -> tensor<1x4x4x3x!quant.uniform<i8:f32, 0.05:0>> {
  // CHECK: quant.scast %arg0 : tensor<1x3x4x4xi8> to tensor<1x3x4x4x!quant.uniform<i8:f32, 5.000000e-02>>
  // CHECK: "onnx.Transpose"
  // CHECK-SAME: perm = [0, 2, 3, 1]
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x4x4xi8>) -> tensor<1x4x4x3xi8>
  %1 = quant.scast %0 : tensor<1x4x4x3xi8> to tensor<1x4x4x3x!quant.uniform<i8:f32, 0.05:0>>
  return %1 : tensor<1x4x4x3x!quant.uniform<i8:f32, 0.05:0>>
}

// -----

// Test: Push transpose through scast with signed quantized type
// CHECK-LABEL: func @test_push_transpose_through_scast_i8
func.func @test_push_transpose_through_scast_i8(%arg0: tensor<1x32x7x7x!quant.uniform<i8:f32, 0.00842:0>>) -> tensor<1x7x7x32xi8> {
  // CHECK: quant.scast %arg0 : tensor<1x32x7x7x!quant.uniform<i8:f32, 8.420000e-03>> to tensor<1x32x7x7xi8>
  // CHECK: "onnx.Transpose"
  // CHECK-SAME: perm = [0, 2, 3, 1]
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x32x7x7x!quant.uniform<i8:f32, 0.00842:0>>) -> tensor<1x7x7x32x!quant.uniform<i8:f32, 0.00842:0>>
  %1 = quant.scast %0 : tensor<1x7x7x32x!quant.uniform<i8:f32, 0.00842:0>> to tensor<1x7x7x32xi8>
  return %1 : tensor<1x7x7x32xi8>
}

// -----

// Test: scast feeding DequantizeLinear is a boundary - should NOT push transpose through
// CHECK-LABEL: func @test_no_push_transpose_through_boundary_scast
func.func @test_no_push_transpose_through_boundary_scast(
    %arg0: tensor<1x3x4x4x!quant.uniform<i8:f32, 0.05:0>>,
    %scale: tensor<f32>,
    %zp: tensor<i8>) -> tensor<1x4x4x3xf32> {
  // CHECK: "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]}
  // CHECK: quant.scast
  // CHECK: "onnx.DequantizeLinear"
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x3x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x4x4x3x!quant.uniform<i8:f32, 0.05:0>>
  %1 = quant.scast %0 : tensor<1x4x4x3x!quant.uniform<i8:f32, 0.05:0>> to tensor<1x4x4x3xi8>
  %2 = "onnx.DequantizeLinear"(%1, %scale, %zp) : (tensor<1x4x4x3xi8>, tensor<f32>, tensor<i8>) -> tensor<1x4x4x3xf32>
  return %2 : tensor<1x4x4x3xf32>
}

// -----

// Test: scast without transpose input should NOT be modified
// CHECK-LABEL: func @test_no_push_scast_without_transpose
func.func @test_no_push_scast_without_transpose(%arg0: tensor<1x3x4x4x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x3x4x4xi8> {
  // CHECK-NOT: onnx.Transpose
  // CHECK: quant.scast %arg0
  %0 = quant.scast %arg0 : tensor<1x3x4x4x!quant.uniform<i8:f32, 0.05:0>> to tensor<1x3x4x4xi8>
  return %0 : tensor<1x3x4x4xi8>
}

// -----

// ============================================================================
// SECTION: Push Transpose Through Softmax
// ============================================================================

// Test: Basic push transpose through softmax (NCHW -> NHWC, axis on channel dim)
// Transpose [0,2,3,1] moves C from position 1 to 3.
// Softmax on axis=3 (channel in NHWC) should become axis=1 (channel in NCHW)
// after pushing the transpose through.
// CHECK-LABEL: func @test_push_transpose_through_softmax_basic
func.func @test_push_transpose_through_softmax_basic(%arg0: tensor<1x16x4x8400xf32>) -> tensor<1x4x8400x16xf32> {
  // CHECK: %[[SOFTMAX:.*]] = "onnx.Softmax"(%arg0) {axis = 1 : si64}
  // CHECK-SAME: (tensor<1x16x4x8400xf32>) -> tensor<1x16x4x8400xf32>
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[SOFTMAX]]) {perm = [0, 2, 3, 1]}
  // CHECK-SAME: (tensor<1x16x4x8400xf32>) -> tensor<1x4x8400x16xf32>
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x16x4x8400xf32>) -> tensor<1x4x8400x16xf32>
  %1 = "onnx.Softmax"(%0) {axis = 3 : si64} : (tensor<1x4x8400x16xf32>) -> tensor<1x4x8400x16xf32>
  return %1 : tensor<1x4x8400x16xf32>
}

// -----

// Test: Push transpose through softmax on last axis (no axis change needed)
// Transpose [0,2,1] on 3D tensor. Softmax on axis=2 (last dim).
// perm[2]=1, so new axis should be 1.
// CHECK-LABEL: func @test_push_transpose_through_softmax_3d
func.func @test_push_transpose_through_softmax_3d(%arg0: tensor<1x8400x16xf32>) -> tensor<1x16x8400xf32> {
  // CHECK: %[[SOFTMAX:.*]] = "onnx.Softmax"(%arg0) {axis = 1 : si64}
  // CHECK-SAME: (tensor<1x8400x16xf32>) -> tensor<1x8400x16xf32>
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[SOFTMAX]]) {perm = [0, 2, 1]}
  // CHECK-SAME: (tensor<1x8400x16xf32>) -> tensor<1x16x8400xf32>
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1]} : (tensor<1x8400x16xf32>) -> tensor<1x16x8400xf32>
  %1 = "onnx.Softmax"(%0) {axis = 2 : si64} : (tensor<1x16x8400xf32>) -> tensor<1x16x8400xf32>
  return %1 : tensor<1x16x8400xf32>
}

// -----

// Test: Push transpose through softmax with negative axis
// Softmax axis=-1 on rank-4 tensor -> axis=3, perm[3]=1 -> new axis=1
// CHECK-LABEL: func @test_push_transpose_through_softmax_neg_axis
func.func @test_push_transpose_through_softmax_neg_axis(%arg0: tensor<2x16x8x8xf32>) -> tensor<2x8x8x16xf32> {
  // CHECK: %[[SOFTMAX:.*]] = "onnx.Softmax"(%arg0) {axis = 1 : si64}
  // CHECK-SAME: (tensor<2x16x8x8xf32>) -> tensor<2x16x8x8xf32>
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[SOFTMAX]]) {perm = [0, 2, 3, 1]}
  // CHECK-SAME: (tensor<2x16x8x8xf32>) -> tensor<2x8x8x16xf32>
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<2x16x8x8xf32>) -> tensor<2x8x8x16xf32>
  %1 = "onnx.Softmax"(%0) {axis = -1 : si64} : (tensor<2x8x8x16xf32>) -> tensor<2x8x8x16xf32>
  return %1 : tensor<2x8x8x16xf32>
}

// -----

// Test: Push transpose through softmax with quantized types
// CHECK-LABEL: func @test_push_transpose_through_softmax_quant
func.func @test_push_transpose_through_softmax_quant(%arg0: tensor<1x8400x4x16x!quant.uniform<u16:f32, 3.105E-4:32768>>) -> tensor<1x16x4x8400x!quant.uniform<u16:f32, 3.105E-4:32768>> {
  // CHECK: %[[SOFTMAX:.*]] = "onnx.Softmax"(%arg0) {axis = 3 : si64}
  // CHECK-SAME: (tensor<1x8400x4x16x!quant.uniform<u16:f32, 3.105000e-04:32768>>) -> tensor<1x8400x4x16x!quant.uniform<u16:f32, 3.105000e-04:32768>>
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[SOFTMAX]]) {perm = [0, 3, 2, 1]}
  // CHECK-SAME: (tensor<1x8400x4x16x!quant.uniform<u16:f32, 3.105000e-04:32768>>) -> tensor<1x16x4x8400x!quant.uniform<u16:f32, 3.105000e-04:32768>>
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 3, 2, 1]} : (tensor<1x8400x4x16x!quant.uniform<u16:f32, 3.105E-4:32768>>) -> tensor<1x16x4x8400x!quant.uniform<u16:f32, 3.105E-4:32768>>
  %1 = "onnx.Softmax"(%0) {axis = 1 : si64} : (tensor<1x16x4x8400x!quant.uniform<u16:f32, 3.105E-4:32768>>) -> tensor<1x16x4x8400x!quant.uniform<u16:f32, 3.105E-4:32768>>
  return %1 : tensor<1x16x4x8400x!quant.uniform<u16:f32, 3.105E-4:32768>>
}

// -----

// Test: Transpose + Softmax + Transpose should fuse into Softmax + single Transpose
// After pushing Transpose_A through Softmax, two consecutive transposes fuse.
// Transpose_A=[0,3,2,1] composed with Transpose_B=[0,2,3,1] = [0,2,1,3]
// CHECK-LABEL: func @test_softmax_transpose_fusion_pattern
func.func @test_softmax_transpose_fusion_pattern(%arg0: tensor<1x8400x4x16xf32>) -> tensor<1x4x8400x16xf32> {
  // CHECK: %[[SOFTMAX:.*]] = "onnx.Softmax"(%arg0) {axis = 3 : si64}
  // CHECK-SAME: (tensor<1x8400x4x16xf32>) -> tensor<1x8400x4x16xf32>
  // CHECK: %[[TRANS:.*]] = "onnx.Transpose"(%[[SOFTMAX]]) {perm = [0, 2, 1, 3]}
  // CHECK-SAME: (tensor<1x8400x4x16xf32>) -> tensor<1x4x8400x16xf32>
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 3, 2, 1]} : (tensor<1x8400x4x16xf32>) -> tensor<1x16x4x8400xf32>
  %1 = "onnx.Softmax"(%0) {axis = 1 : si64} : (tensor<1x16x4x8400xf32>) -> tensor<1x16x4x8400xf32>
  %2 = "onnx.Transpose"(%1) {perm = [0, 2, 3, 1]} : (tensor<1x16x4x8400xf32>) -> tensor<1x4x8400x16xf32>
  return %2 : tensor<1x4x8400x16xf32>
}

// -----
// ============================================================================
// Test: Per-axis quantized constant through transpose+binary fusion
// ============================================================================
// Transpose(x, [0,2,3,1]) * per-axis-quant-const pushes transpose through Mul.
// Const [1,1,1,4] is transpose-immune, so it gets Reshaped to [1,4,1,1].
// Per-axis dim 3 must remap to dim 1 via inverse perm [0,3,1,2].

// CHECK-LABEL: func @test_transpose_binary_per_axis_quant
func.func @test_transpose_binary_per_axis_quant(%arg0: tensor<1x4x8x8x!quant.uniform<i8:f32, 0.1:0>>) -> tensor<1x8x8x4x!quant.uniform<i8:f32, 0.1:0>> {
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 3, 1]} : (tensor<1x4x8x8x!quant.uniform<i8:f32, 0.1:0>>) -> tensor<1x8x8x4x!quant.uniform<i8:f32, 0.1:0>>
  %c = onnx.Constant {value = dense<1> : tensor<1x1x1x4xi8>} : tensor<1x1x1x4x!quant.uniform<i8:f32:3, {0.01, 0.02, 0.03, 0.04}>>
  %1 = "onnx.Mul"(%0, %c) : (tensor<1x8x8x4x!quant.uniform<i8:f32, 0.1:0>>, tensor<1x1x1x4x!quant.uniform<i8:f32:3, {0.01, 0.02, 0.03, 0.04}>>) -> tensor<1x8x8x4x!quant.uniform<i8:f32, 0.1:0>>
  return %1 : tensor<1x8x8x4x!quant.uniform<i8:f32, 0.1:0>>
}
// Reshaped to [1,4,1,1] with per-axis dim remapped from 3→1
// CHECK: onnx.Reshape{{.*}}tensor<1x4x1x1x!quant.uniform<i8:f32:1, {1.000000e-02,2.000000e-02,3.000000e-02,4.000000e-02}>>
// CHECK: onnx.Mul
// CHECK: onnx.Transpose
