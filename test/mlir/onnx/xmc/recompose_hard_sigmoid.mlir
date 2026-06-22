// RUN: onnx-mlir-opt --split-input-file --recompose-hard-sigmoid %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

// ============================================================================
// PASS CASES: canonical decomposed HardSigmoid
//   Mul(x, ~1/6) -> Add(., ~0.5) -> Clip(., 0, 1)
// is rewritten into `onnx.HardSigmoid(x) {alpha=0.2, beta=0.5}`.
// ============================================================================

// CHECK-LABEL: @recompose_canonical_pass
func.func @recompose_canonical_pass(%arg0: tensor<1x16x28x28xf32>) -> tensor<1x16x28x28xf32> {
  %alpha = onnx.Constant dense<0.166666672> : tensor<f32>
  %beta  = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %lo    = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %hi    = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %m = "onnx.Mul"(%arg0, %alpha) : (tensor<1x16x28x28xf32>, tensor<f32>) -> tensor<1x16x28x28xf32>
  %a = "onnx.Add"(%m, %beta)     : (tensor<1x16x28x28xf32>, tensor<f32>) -> tensor<1x16x28x28xf32>
  %c = "onnx.Clip"(%a, %lo, %hi) : (tensor<1x16x28x28xf32>, tensor<f32>, tensor<f32>) -> tensor<1x16x28x28xf32>
  return %c : tensor<1x16x28x28xf32>
}
// CHECK-NOT: "onnx.Mul"
// CHECK-NOT: "onnx.Add"
// CHECK-NOT: "onnx.Clip"
// CHECK: %[[HSIG:.*]] = "onnx.HardSigmoid"(%arg0)
// CHECK-SAME: alpha = 2.000000e-01 : f32
// CHECK-SAME: beta = 5.000000e-01 : f32

// -----

// Float drift in alpha/beta (exact constants seen in PyTorch QAT exports):
// still folded; alpha/beta are snapped to the ONNX HardSigmoid defaults.
// CHECK-LABEL: @recompose_fp_drift_pass
func.func @recompose_fp_drift_pass(%arg0: tensor<1x8x14x14xf32>) -> tensor<1x8x14x14xf32> {
  %alpha = onnx.Constant dense<0.166687012> : tensor<f32>
  %beta  = onnx.Constant dense<0.500061035> : tensor<f32>
  %lo    = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %hi    = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %m = "onnx.Mul"(%arg0, %alpha) : (tensor<1x8x14x14xf32>, tensor<f32>) -> tensor<1x8x14x14xf32>
  %a = "onnx.Add"(%m, %beta)     : (tensor<1x8x14x14xf32>, tensor<f32>) -> tensor<1x8x14x14xf32>
  %c = "onnx.Clip"(%a, %lo, %hi) : (tensor<1x8x14x14xf32>, tensor<f32>, tensor<f32>) -> tensor<1x8x14x14xf32>
  return %c : tensor<1x8x14x14xf32>
}
// CHECK: "onnx.HardSigmoid"(%arg0)
// CHECK-SAME: alpha = 2.000000e-01 : f32
// CHECK-SAME: beta = 5.000000e-01 : f32

// -----

// Operands of Mul commuted: Mul(alpha, x) is still recognised.
// CHECK-LABEL: @recompose_mul_commuted_pass
func.func @recompose_mul_commuted_pass(%arg0: tensor<1x4x7x7xf32>) -> tensor<1x4x7x7xf32> {
  %alpha = onnx.Constant dense<0.166666672> : tensor<f32>
  %beta  = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %lo    = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %hi    = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %m = "onnx.Mul"(%alpha, %arg0) : (tensor<f32>, tensor<1x4x7x7xf32>) -> tensor<1x4x7x7xf32>
  %a = "onnx.Add"(%m, %beta)     : (tensor<1x4x7x7xf32>, tensor<f32>) -> tensor<1x4x7x7xf32>
  %c = "onnx.Clip"(%a, %lo, %hi) : (tensor<1x4x7x7xf32>, tensor<f32>, tensor<f32>) -> tensor<1x4x7x7xf32>
  return %c : tensor<1x4x7x7xf32>
}
// CHECK: "onnx.HardSigmoid"(%arg0)
// CHECK-SAME: alpha = 2.000000e-01 : f32
// CHECK-SAME: beta = 5.000000e-01 : f32

// -----

// Operands of Add commuted: Add(beta, mul) is still recognised.
// CHECK-LABEL: @recompose_add_commuted_pass
func.func @recompose_add_commuted_pass(%arg0: tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32> {
  %alpha = onnx.Constant dense<0.166666672> : tensor<f32>
  %beta  = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %lo    = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %hi    = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %m = "onnx.Mul"(%arg0, %alpha) : (tensor<2x3x4x5xf32>, tensor<f32>) -> tensor<2x3x4x5xf32>
  %a = "onnx.Add"(%beta, %m)     : (tensor<f32>, tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
  %c = "onnx.Clip"(%a, %lo, %hi) : (tensor<2x3x4x5xf32>, tensor<f32>, tensor<f32>) -> tensor<2x3x4x5xf32>
  return %c : tensor<2x3x4x5xf32>
}
// CHECK: "onnx.HardSigmoid"(%arg0)
// CHECK-SAME: alpha = 2.000000e-01 : f32
// CHECK-SAME: beta = 5.000000e-01 : f32

// -----

// Chain sandwiched between DQ/Q (the actual shape seen in QAT models): the
// DQ feeds the Mul and the Q consumes the new HardSigmoid; both stay.
// CHECK-LABEL: @recompose_within_qdq_pass
func.func @recompose_within_qdq_pass(%arg0: tensor<1x72x1x1xi8>) -> tensor<1x72x1x1xi8> {
  %s_in  = onnx.Constant dense<3.125000e-02> : tensor<f32>
  %z_in  = onnx.Constant dense<0> : tensor<i8>
  %s_out = onnx.Constant dense<7.812500e-03> : tensor<f32>
  %z_out = onnx.Constant dense<0> : tensor<i8>
  %alpha = onnx.Constant dense<0.166666672> : tensor<f32>
  %beta  = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %lo    = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %hi    = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %dq = "onnx.DequantizeLinear"(%arg0, %s_in, %z_in) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x72x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x72x1x1xf32>
  %m  = "onnx.Mul"(%dq, %alpha) : (tensor<1x72x1x1xf32>, tensor<f32>) -> tensor<1x72x1x1xf32>
  %a  = "onnx.Add"(%m,  %beta)  : (tensor<1x72x1x1xf32>, tensor<f32>) -> tensor<1x72x1x1xf32>
  %c  = "onnx.Clip"(%a, %lo, %hi) : (tensor<1x72x1x1xf32>, tensor<f32>, tensor<f32>) -> tensor<1x72x1x1xf32>
  %q  = "onnx.QuantizeLinear"(%c, %s_out, %z_out) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x72x1x1xf32>, tensor<f32>, tensor<i8>) -> tensor<1x72x1x1xi8>
  return %q : tensor<1x72x1x1xi8>
}
// CHECK: %[[DQ:.*]] = "onnx.DequantizeLinear"
// CHECK: %[[HSIG:.*]] = "onnx.HardSigmoid"(%[[DQ]])
// CHECK-SAME: alpha = 2.000000e-01 : f32
// CHECK-SAME: beta = 5.000000e-01 : f32
// CHECK: "onnx.QuantizeLinear"(%[[HSIG]]
// CHECK-NOT: "onnx.Mul"
// CHECK-NOT: "onnx.Add"
// CHECK-NOT: "onnx.Clip"

// ============================================================================
// FAIL CASES: chains that must NOT be folded
// ============================================================================

// -----

// alpha is not ~= 1/6.
// CHECK-LABEL: @recompose_noncanonical_alpha_fail
func.func @recompose_noncanonical_alpha_fail(%arg0: tensor<1x8x14x14xf32>) -> tensor<1x8x14x14xf32> {
  %alpha = onnx.Constant dense<2.500000e-01> : tensor<f32>
  %beta  = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %lo    = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %hi    = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %m = "onnx.Mul"(%arg0, %alpha) : (tensor<1x8x14x14xf32>, tensor<f32>) -> tensor<1x8x14x14xf32>
  %a = "onnx.Add"(%m, %beta)     : (tensor<1x8x14x14xf32>, tensor<f32>) -> tensor<1x8x14x14xf32>
  %c = "onnx.Clip"(%a, %lo, %hi) : (tensor<1x8x14x14xf32>, tensor<f32>, tensor<f32>) -> tensor<1x8x14x14xf32>
  return %c : tensor<1x8x14x14xf32>
}
// CHECK: "onnx.Mul"
// CHECK: "onnx.Add"
// CHECK: "onnx.Clip"
// CHECK-NOT: "onnx.HardSigmoid"

// -----

// beta is not ~= 0.5.
// CHECK-LABEL: @recompose_noncanonical_beta_fail
func.func @recompose_noncanonical_beta_fail(%arg0: tensor<1x8x14x14xf32>) -> tensor<1x8x14x14xf32> {
  %alpha = onnx.Constant dense<0.166666672> : tensor<f32>
  %beta  = onnx.Constant dense<2.500000e-01> : tensor<f32>
  %lo    = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %hi    = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %m = "onnx.Mul"(%arg0, %alpha) : (tensor<1x8x14x14xf32>, tensor<f32>) -> tensor<1x8x14x14xf32>
  %a = "onnx.Add"(%m, %beta)     : (tensor<1x8x14x14xf32>, tensor<f32>) -> tensor<1x8x14x14xf32>
  %c = "onnx.Clip"(%a, %lo, %hi) : (tensor<1x8x14x14xf32>, tensor<f32>, tensor<f32>) -> tensor<1x8x14x14xf32>
  return %c : tensor<1x8x14x14xf32>
}
// CHECK: "onnx.Mul"
// CHECK: "onnx.Add"
// CHECK: "onnx.Clip"
// CHECK-NOT: "onnx.HardSigmoid"

// -----

// Clip bounds are not (0, 1).
// CHECK-LABEL: @recompose_wrong_clip_bounds_fail
func.func @recompose_wrong_clip_bounds_fail(%arg0: tensor<1x8x14x14xf32>) -> tensor<1x8x14x14xf32> {
  %alpha = onnx.Constant dense<0.166666672> : tensor<f32>
  %beta  = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %lo    = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %hi    = onnx.Constant dense<6.000000e+00> : tensor<f32>
  %m = "onnx.Mul"(%arg0, %alpha) : (tensor<1x8x14x14xf32>, tensor<f32>) -> tensor<1x8x14x14xf32>
  %a = "onnx.Add"(%m, %beta)     : (tensor<1x8x14x14xf32>, tensor<f32>) -> tensor<1x8x14x14xf32>
  %c = "onnx.Clip"(%a, %lo, %hi) : (tensor<1x8x14x14xf32>, tensor<f32>, tensor<f32>) -> tensor<1x8x14x14xf32>
  return %c : tensor<1x8x14x14xf32>
}
// CHECK: "onnx.Mul"
// CHECK: "onnx.Add"
// CHECK: "onnx.Clip"
// CHECK-NOT: "onnx.HardSigmoid"

// -----

// Mul has multiple uses: cannot safely fold.
// CHECK-LABEL: @recompose_mul_multiuse_fail
func.func @recompose_mul_multiuse_fail(%arg0: tensor<1x8x14x14xf32>) -> (tensor<1x8x14x14xf32>, tensor<1x8x14x14xf32>) {
  %alpha = onnx.Constant dense<0.166666672> : tensor<f32>
  %beta  = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %lo    = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %hi    = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %m = "onnx.Mul"(%arg0, %alpha) : (tensor<1x8x14x14xf32>, tensor<f32>) -> tensor<1x8x14x14xf32>
  %a = "onnx.Add"(%m, %beta)     : (tensor<1x8x14x14xf32>, tensor<f32>) -> tensor<1x8x14x14xf32>
  %c = "onnx.Clip"(%a, %lo, %hi) : (tensor<1x8x14x14xf32>, tensor<f32>, tensor<f32>) -> tensor<1x8x14x14xf32>
  return %c, %m : tensor<1x8x14x14xf32>, tensor<1x8x14x14xf32>
}
// CHECK: "onnx.Mul"
// CHECK: "onnx.Add"
// CHECK: "onnx.Clip"
// CHECK-NOT: "onnx.HardSigmoid"

// -----

// Add has multiple uses: cannot safely fold.
// CHECK-LABEL: @recompose_add_multiuse_fail
func.func @recompose_add_multiuse_fail(%arg0: tensor<1x8x14x14xf32>) -> (tensor<1x8x14x14xf32>, tensor<1x8x14x14xf32>) {
  %alpha = onnx.Constant dense<0.166666672> : tensor<f32>
  %beta  = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %lo    = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %hi    = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %m = "onnx.Mul"(%arg0, %alpha) : (tensor<1x8x14x14xf32>, tensor<f32>) -> tensor<1x8x14x14xf32>
  %a = "onnx.Add"(%m, %beta)     : (tensor<1x8x14x14xf32>, tensor<f32>) -> tensor<1x8x14x14xf32>
  %c = "onnx.Clip"(%a, %lo, %hi) : (tensor<1x8x14x14xf32>, tensor<f32>, tensor<f32>) -> tensor<1x8x14x14xf32>
  return %c, %a : tensor<1x8x14x14xf32>, tensor<1x8x14x14xf32>
}
// CHECK: "onnx.Mul"
// CHECK: "onnx.Add"
// CHECK: "onnx.Clip"
// CHECK-NOT: "onnx.HardSigmoid"

// -----

// Clip bounds are not constant scalars (operand-based): pattern bails.
// CHECK-LABEL: @recompose_nonconst_clip_bounds_fail
func.func @recompose_nonconst_clip_bounds_fail(%arg0: tensor<1x8x14x14xf32>, %lo: tensor<f32>) -> tensor<1x8x14x14xf32> {
  %alpha = onnx.Constant dense<0.166666672> : tensor<f32>
  %beta  = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %hi    = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %m = "onnx.Mul"(%arg0, %alpha) : (tensor<1x8x14x14xf32>, tensor<f32>) -> tensor<1x8x14x14xf32>
  %a = "onnx.Add"(%m, %beta)     : (tensor<1x8x14x14xf32>, tensor<f32>) -> tensor<1x8x14x14xf32>
  %c = "onnx.Clip"(%a, %lo, %hi) : (tensor<1x8x14x14xf32>, tensor<f32>, tensor<f32>) -> tensor<1x8x14x14xf32>
  return %c : tensor<1x8x14x14xf32>
}
// CHECK: "onnx.Mul"
// CHECK: "onnx.Add"
// CHECK: "onnx.Clip"
// CHECK-NOT: "onnx.HardSigmoid"
