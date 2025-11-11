// RUN: onnx-mlir-opt --dq-binary-q-opt-onnx-to-onnx %s -split-input-file | FileCheck %s


  func.func @test_fold_mul_case_b_safe(%arg0: tensor<10x1xf32>) -> tensor<10x1xf32> {
    %0 = onnx.Constant dense<0> : tensor<ui16>
    %1 = onnx.Constant dense<5.78499521E-6> : tensor<f32>
    %2 = onnx.Constant dense<0> : tensor<ui16>
    %3 = onnx.Constant dense<0.00152590231> : tensor<f32>
    %4 = onnx.Constant dense<65535> : tensor<ui16>
    %5 = onnx.Constant dense<10> : tensor<ui16>
    %6 = onnx.Constant dense<1.000000e-01> : tensor<f32>
    %7 = "onnx.DequantizeLinear"(%4, %3, %2) {axis = 1 : si64, block_size = 0 : si64} : (tensor<ui16>, tensor<f32>, tensor<ui16>) -> tensor<f32>
    %8 = "onnx.QuantizeLinear"(%arg0, %1, %0) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<10x1xf32>, tensor<f32>, tensor<ui16>) -> tensor<10x1xui16>
    %9 = "onnx.DequantizeLinear"(%8, %1, %0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<10x1xui16>, tensor<f32>, tensor<ui16>) -> tensor<10x1xf32>
    %10 = "onnx.Mul"(%9, %7) : (tensor<10x1xf32>, tensor<f32>) -> tensor<10x1xf32>
    %11 = "onnx.QuantizeLinear"(%10, %6, %5) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<10x1xf32>, tensor<f32>, tensor<ui16>) -> tensor<10x1xui16>
    %12 = "onnx.DequantizeLinear"(%11, %6, %5) {axis = 1 : si64, block_size = 0 : si64} : (tensor<10x1xui16>, tensor<f32>, tensor<ui16>) -> tensor<10x1xf32>
    return %12 : tensor<10x1xf32>
  }

// CHECK:        %[[ZP:.*]] = onnx.Constant dense<10> : tensor<ui16>
// CHECK:        %[[DQ_SCALE:.*]] = onnx.Constant dense<1.000000e-01> : tensor<f32>
// CHECK:        %[[NEW_SCALE:.*]] = onnx.Constant dense<9.99999931E-4> : tensor<f32>
// CHECK:        %[[Q:.*]] = "onnx.QuantizeLinear"(%arg0, %[[NEW_SCALE]], %[[ZP]])
// CHECK-SAME:     : (tensor<10x1xf32>, tensor<f32>, tensor<ui16>) -> tensor<10x1xui16>
// CHECK:        %[[DQ:.*]] = "onnx.DequantizeLinear"(%[[Q]], %[[DQ_SCALE]], %[[ZP]])
// CHECK-SAME:     : (tensor<10x1xui16>, tensor<f32>, tensor<ui16>) -> tensor<10x1xf32>
// CHECK:        return %[[DQ]]
// CHECK-NOT:    "onnx.Mul"

// ============================================================================
// ===== CASE A: lhs = DQ, rhs = Const  (fold into Q; update Q.y_zero_point) =====
// ============================================================================

func.func @caseA_lhsDQ_rhsConst_foldIntoQ(%arg0: tensor<1x4xf32>) -> tensor<1x4xi8> {
    %0 = onnx.Constant dense<5.000000e-01> : tensor<f32>
    %1 = onnx.Constant dense<0> : tensor<i8>
    %2 = "onnx.QuantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
    %3 = onnx.Constant dense<5.000000e-01> : tensor<f32>
    %4 = onnx.Constant dense<0> : tensor<i8>
    %5 = "onnx.DequantizeLinear"(%2, %3, %4) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xi8>, tensor<f32>, tensor<i8>) -> tensor<1x4xf32>
    %6 = onnx.Constant dense<1.000000e+01> : tensor<f32>
    %7 = "onnx.Add"(%5, %6) : (tensor<1x4xf32>, tensor<f32>) -> tensor<1x4xf32>
    %8 = onnx.Constant dense<1.000000e-01> : tensor<f32>
    %9 = onnx.Constant dense<0> : tensor<i8>
    %10 = "onnx.QuantizeLinear"(%7, %8, %9) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
    return %10 : tensor<1x4xi8>
  }

// CHECK-LABEL: func.func @caseA_lhsDQ_rhsConst_foldIntoQ
// CHECK: %[[S:.*]] = onnx.Constant dense<1.000000e-01> : tensor<f32>
// CHECK: %[[ZP:.*]] = onnx.Constant dense<99> : tensor<i8>
// CHECK: %[[Q:.*]] = "onnx.QuantizeLinear"(%arg0, %[[S]], %[[ZP]])
// CHECK-SAME: : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
// CHECK: return %[[Q]] : tensor<1x4xi8>

// ============================================================================
// ===== CASE A-REV: rhs = DQ, lhs = Const  (fold into Q; update Q.y_zero_point) =====
// ============================================================================

func.func @caseA_rev_rhsDQ_lhsConst_foldIntoQ(%arg0: tensor<1x4xf32>) -> tensor<1x4xi8> {
    %0 = onnx.Constant dense<5.000000e-01> : tensor<f32>
    %1 = onnx.Constant dense<0> : tensor<i8>
    %2 = "onnx.QuantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
    %3 = onnx.Constant dense<5.000000e-01> : tensor<f32>
    %4 = onnx.Constant dense<0> : tensor<i8>
    %5 = "onnx.DequantizeLinear"(%2, %3, %4) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xi8>, tensor<f32>, tensor<i8>) -> tensor<1x4xf32>
    %6 = onnx.Constant dense<1.000000e+01> : tensor<f32>
    %7 = "onnx.Add"(%6, %5) : (tensor<f32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    %8 = onnx.Constant dense<1.000000e-01> : tensor<f32>
    %9 = onnx.Constant dense<0> : tensor<i8>
    %10 = "onnx.QuantizeLinear"(%7, %8, %9) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
    return %10 : tensor<1x4xi8>
  }
 
// CHECK-LABEL: func.func @caseA_rev_rhsDQ_lhsConst_foldIntoQ
// CHECK: %[[S:.*]] = onnx.Constant dense<1.000000e-01> : tensor<f32>
// CHECK: %[[ZP:.*]] = onnx.Constant dense<99> : tensor<i8>
// CHECK: %[[Q:.*]] = "onnx.QuantizeLinear"(%arg0, %[[S]], %[[ZP]])
// CHECK-SAME: : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
// CHECK: return %[[Q]] : tensor<1x4xi8>

// ============================================================================
// ===== CASE B: both inputs are DQ; constant via dq1  (fold into Q) =====
// ============================================================================

func.func @caseB_bothDQ_constViaDQ1_foldIntoQ(%arg0: tensor<1x4xf32>) -> tensor<1x4xi8> {
    %0 = onnx.Constant dense<5.000000e-01> : tensor<f32>
    %1 = onnx.Constant dense<0> : tensor<i8>
    %2 = "onnx.QuantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
    %3 = "onnx.DequantizeLinear"(%2, %0, %1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xi8>, tensor<f32>, tensor<i8>) -> tensor<1x4xf32>
    %4 = onnx.Constant dense<10> : tensor<i8>
    %5 = onnx.Constant dense<5.000000e+00> : tensor<f32>
    %6 = onnx.Constant dense<0> : tensor<i8>
    %7 = "onnx.DequantizeLinear"(%4, %5, %6) {axis = 1 : si64, block_size = 0 : si64} : (tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<f32>
    %8 = "onnx.Add"(%3, %7) : (tensor<1x4xf32>, tensor<f32>) -> tensor<1x4xf32>
    %9 = onnx.Constant dense<5.000000e-01> : tensor<f32>
    %10 = onnx.Constant dense<0> : tensor<i8>
    %11 = "onnx.QuantizeLinear"(%8, %9, %10) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
    return %11 : tensor<1x4xi8>
  }
// CHECK-LABEL: func.func @caseB_bothDQ_constViaDQ1_foldIntoQ
// CHECK: %[[S:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CHECK: %[[ZP:.*]] = onnx.Constant dense<100> : tensor<i8>
// CHECK: %[[Q:.*]] = "onnx.QuantizeLinear"(%arg0, %[[S]], %[[ZP]])
// CHECK-SAME: : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
// CHECK: return %[[Q]] : tensor<1x4xi8>

// ============================================================================
// ===== CASE B with value-preserving link on constant side: Reshape(const_q) → DQ =====
// ============================================================================

  func.func @caseB_constViaReshape_foldIntoQ(%arg0: tensor<1x4xf32>) -> tensor<1x4xi8> {
    %0 = onnx.Constant dense<1.000000e+00> : tensor<f32>
    %1 = onnx.Constant dense<0> : tensor<i8>
    %2 = "onnx.QuantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
    %3 = "onnx.DequantizeLinear"(%2, %0, %1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xi8>, tensor<f32>, tensor<i8>) -> tensor<1x4xf32>
    %4 = onnx.Constant dense<25> : tensor<i8>
    %5 = onnx.Constant dense<> : tensor<0xi64>
    %6 = "onnx.Reshape"(%4, %5) {allowzero = 0 : si64} : (tensor<i8>, tensor<0xi64>) -> tensor<i8>
    %7 = onnx.Constant dense<4.000000e+00> : tensor<f32>
    %8 = onnx.Constant dense<0> : tensor<i8>
    %9 = "onnx.DequantizeLinear"(%6, %7, %8) {axis = 1 : si64, block_size = 0 : si64} : (tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<f32>
    %10 = "onnx.Add"(%3, %9) : (tensor<1x4xf32>, tensor<f32>) -> tensor<1x4xf32>
    %11 = onnx.Constant dense<1.000000e+00> : tensor<f32>
    %12 = onnx.Constant dense<0> : tensor<i8>
    %13 = "onnx.QuantizeLinear"(%10, %11, %12) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
    return %13 : tensor<1x4xi8>
  }

// CHECK-LABEL: func.func @caseB_constViaReshape_foldIntoQ
// CHECK:        %[[SCALE:.*]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK:        %[[ZP:.*]]    = onnx.Constant dense<100> : tensor<i8>
// CHECK:        %[[Q:.*]] = "onnx.QuantizeLinear"(%arg0, %[[SCALE]], %[[ZP]])
// CHECK-SAME:     : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
// CHECK:        return %[[Q]] : tensor<1x4xi8>
// CHECK-NOT:    onnx.Add
// CHECK-NOT:    onnx.DequantizeLinear
// CHECK-NOT:    onnx.Reshape

// ============================================================================
// ===== BRANCH-BEFORE: Q1 has another user  (fold into DQ; update DQ.x_zero_point) =====
// ============================================================================

  func.func @branchBefore_foldIntoDQ(%arg0: tensor<1x4xf32>) -> (tensor<1x4xf32>, tensor<1x4xi8>) {
    %0 = onnx.Constant dense<5.000000e-01> : tensor<f32>
    %1 = onnx.Constant dense<0> : tensor<i8>
    %2 = "onnx.QuantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
    %3 = "onnx.Abs"(%2) : (tensor<1x4xi8>) -> tensor<1x4xi8>
    %4 = onnx.Constant dense<1.000000e-01> : tensor<f32>
    %5 = onnx.Constant dense<0> : tensor<i8>
    %6 = "onnx.DequantizeLinear"(%2, %4, %5) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xi8>, tensor<f32>, tensor<i8>) -> tensor<1x4xf32>
    %7 = onnx.Constant dense<1.000000e+01> : tensor<f32>
    %8 = "onnx.Add"(%6, %7) : (tensor<1x4xf32>, tensor<f32>) -> tensor<1x4xf32>
    %9 = onnx.Constant dense<2.000000e-01> : tensor<f32>
    %10 = onnx.Constant dense<0> : tensor<i8>
    %11 = "onnx.QuantizeLinear"(%8, %9, %10) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
    %12 = "onnx.DequantizeLinear"(%11, %9, %10) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xi8>, tensor<f32>, tensor<i8>) -> tensor<1x4xf32>
    return %12, %3 : tensor<1x4xf32>, tensor<1x4xi8>
  }

// CHECK-LABEL: func.func @branchBefore_foldIntoDQ
// CHECK: %[[S_DQ:.*]] = onnx.Constant dense<1.000000e-01> : tensor<f32>
// CHECK: %[[S_Q:.*]]  = onnx.Constant dense<5.000000e-01> : tensor<f32>
// CHECK: %[[ZP:.*]]   = onnx.Constant dense<0> : tensor<i8>
// CHECK: %[[Q:.*]] = "onnx.QuantizeLinear"(%arg0, %[[S_Q]], %[[ZP]])
// CHECK-SAME: : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
// CHECK: %[[ABS:.*]] = "onnx.Abs"(%[[Q]])
// CHECK-SAME: : (tensor<1x4xi8>) -> tensor<1x4xi8>
// CHECK: %[[DQ:.*]] = "onnx.DequantizeLinear"(%[[Q]], %[[S_DQ]], %[[ZP]])
// CHECK-SAME: : (tensor<1x4xi8>, tensor<f32>, tensor<i8>) -> tensor<1x4xf32>
// CHECK: return %[[DQ]], %[[ABS]] : tensor<1x4xf32>, tensor<1x4xi8>


// ============================================================================
// k_value == 0 and (dst is DequantizeLinear) with a Div
// Expectation: DO NOT fold (would require scale_new = scale / k, div-by-zero)
// Reason: k_value = (const_q - zp) * scale_const = (7 - 7) * 0.5 = 0
// ============================================================================

func.func @guard_div_into_dq_k_zero(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  // Activation path: Q -> DQ
  %s_act = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp_act = onnx.Constant dense<0> : tensor<i8>
  %q_act = "onnx.QuantizeLinear"(%arg0, %s_act, %zp_act) : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
  %dq_act = "onnx.DequantizeLinear"(%q_act, %s_act, %zp_act) : (tensor<1x4xi8>, tensor<f32>, tensor<i8>) -> tensor<1x4xf32>

  // Constant path into DQ with k_value == 0
  // const_q = 7, zp = 7, scale = 0.5 => k = (7-7)*0.5 = 0
  %const_q = onnx.Constant dense<7> : tensor<i8>
  %s_c = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp_c = onnx.Constant dense<7> : tensor<i8>
  %dq_c = "onnx.DequantizeLinear"(%const_q, %s_c, %zp_c) : (tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<f32>

  // Binary op is Div. Destination for a fold here would be the upstream DQ (%dq_act).
  %div = "onnx.Div"(%dq_act, %dq_c) : (tensor<1x4xf32>, tensor<f32>) -> tensor<1x4xf32>

  return %div : tensor<1x4xf32>
}

  // CHECK-LABEL: @guard_div_into_dq_k_zero
  // CHECK: "onnx.Div"(
  // (No folding → Div must remain present.)

// ============================================================================
// k_value == 0 and (dst is QuantizeLinear) with a Mul
// ============================================================================

func.func @test_kval_0_dst_q_mul(%arg0: tensor<10x1xf32>) -> tensor<10x1xf32> {
  %0 = onnx.Constant dense<0> : tensor<ui16>
  %1 = onnx.Constant dense<5.78499521E-6> : tensor<f32>
  %2 = onnx.Constant dense<0> : tensor<ui16>
  %3 = onnx.Constant dense<0.00152590231> : tensor<f32>
  %4 = onnx.Constant dense<0> : tensor<ui16>  
  %5 = onnx.Constant dense<10> : tensor<ui16>
  %6 = onnx.Constant dense<1.000000e-01> : tensor<f32>
  %7 = "onnx.DequantizeLinear"(%4, %3, %2) {axis = 1 : si64, block_size = 0 : si64}
       : (tensor<ui16>, tensor<f32>, tensor<ui16>) -> tensor<f32>
  %8 = "onnx.QuantizeLinear"(%arg0, %1, %0) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64}
       : (tensor<10x1xf32>, tensor<f32>, tensor<ui16>) -> tensor<10x1xui16>
  %9  = "onnx.DequantizeLinear"(%8, %1, %0) {axis = 1 : si64, block_size = 0 : si64}
       : (tensor<10x1xui16>, tensor<f32>, tensor<ui16>) -> tensor<10x1xf32>
  %10 = "onnx.Mul"(%9, %7) : (tensor<10x1xf32>, tensor<f32>) -> tensor<10x1xf32>
  %11 = "onnx.QuantizeLinear"(%10, %6, %5) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64}
       : (tensor<10x1xf32>, tensor<f32>, tensor<ui16>) -> tensor<10x1xui16>
  %12 = "onnx.DequantizeLinear"(%11, %6, %5) {axis = 1 : si64, block_size = 0 : si64}
       : (tensor<10x1xui16>, tensor<f32>, tensor<ui16>) -> tensor<10x1xf32>

  return %12 : tensor<10x1xf32>
}

// CHECK-LABEL: func.func @test_kval_0_dst_q_mul(
// CHECK-SAME: %arg0: tensor<10x1xf32>) -> tensor<10x1xf32>
// CHECK: %[[ZP0:.*]] = onnx.Constant dense<0> : tensor<ui16>
// CHECK: %[[S_ACT:.*]] = onnx.Constant dense<5.78499521E-6> : tensor<f32>
// CHECK: %[[S_K:.*]] = onnx.Constant dense<0.00152590231> : tensor<f32>
// CHECK: %[[ZP_OUT:.*]] = onnx.Constant dense<10> : tensor<ui16>
// CHECK: %[[S_OUT:.*]] = onnx.Constant dense<1.000000e-01> : tensor<f32>
// CHECK: %[[DQK:.*]] = "onnx.DequantizeLinear"(%[[ZP0]], %[[S_K]], %[[ZP0]])
// CHECK-SAME: : (tensor<ui16>, tensor<f32>, tensor<ui16>) -> tensor<f32>
// CHECK: %[[QACT:.*]] = "onnx.QuantizeLinear"(%arg0, %[[S_ACT]], %[[ZP0]])
// CHECK-SAME: : (tensor<10x1xf32>, tensor<f32>, tensor<ui16>) -> tensor<10x1xui16>
// CHECK: %[[DQACT:.*]] = "onnx.DequantizeLinear"(%[[QACT]], %[[S_ACT]], %[[ZP0]])
// CHECK-SAME: : (tensor<10x1xui16>, tensor<f32>, tensor<ui16>) -> tensor<10x1xf32>
// CHECK: %[[MUL:.*]] = "onnx.Mul"(%[[DQACT]], %[[DQK]])
// CHECK-SAME: : (tensor<10x1xf32>, tensor<f32>) -> tensor<10x1xf32>
// CHECK: %[[QOUT:.*]] = "onnx.QuantizeLinear"(%[[MUL]], %[[S_OUT]], %[[ZP_OUT]])
// CHECK-SAME: : (tensor<10x1xf32>, tensor<f32>, tensor<ui16>) -> tensor<10x1xui16>
// CHECK: %[[DQOUT:.*]] = "onnx.DequantizeLinear"(%[[QOUT]], %[[S_OUT]], %[[ZP_OUT]])
// CHECK-SAME: : (tensor<10x1xui16>, tensor<f32>, tensor<ui16>) -> tensor<10x1xf32>
// CHECK: return %[[DQOUT]] : tensor<10x1xf32>

// ============================================================================
// k_value == 0 and (dst is QuantizeLinear) with a Div
// ============================================================================

func.func @test_kval_0_dst_q_div(%arg0: tensor<10x1xf32>) -> tensor<10x1xf32> {
  %0 = onnx.Constant dense<0> : tensor<ui16>
  %1 = onnx.Constant dense<5.78499521E-6> : tensor<f32>
  %2 = onnx.Constant dense<0> : tensor<ui16>
  %3 = onnx.Constant dense<0.00152590231> : tensor<f32>
  %4 = onnx.Constant dense<0> : tensor<ui16>  
  %5 = onnx.Constant dense<10> : tensor<ui16>
  %6 = onnx.Constant dense<1.000000e-01> : tensor<f32>
  %7 = "onnx.DequantizeLinear"(%4, %3, %2) {axis = 1 : si64, block_size = 0 : si64}
       : (tensor<ui16>, tensor<f32>, tensor<ui16>) -> tensor<f32>
  %8 = "onnx.QuantizeLinear"(%arg0, %1, %0) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64}
       : (tensor<10x1xf32>, tensor<f32>, tensor<ui16>) -> tensor<10x1xui16>
  %9  = "onnx.DequantizeLinear"(%8, %1, %0) {axis = 1 : si64, block_size = 0 : si64}
       : (tensor<10x1xui16>, tensor<f32>, tensor<ui16>) -> tensor<10x1xf32>
  %10 = "onnx.Div"(%9, %7) : (tensor<10x1xf32>, tensor<f32>) -> tensor<10x1xf32>
  %11 = "onnx.QuantizeLinear"(%10, %6, %5) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64}
       : (tensor<10x1xf32>, tensor<f32>, tensor<ui16>) -> tensor<10x1xui16>
  %12 = "onnx.DequantizeLinear"(%11, %6, %5) {axis = 1 : si64, block_size = 0 : si64}
       : (tensor<10x1xui16>, tensor<f32>, tensor<ui16>) -> tensor<10x1xf32>

  return %12 : tensor<10x1xf32>
}

// CHECK-LABEL: func.func @test_kval_0_dst_q_div(
// CHECK-SAME: %arg0: tensor<10x1xf32>) -> tensor<10x1xf32>
// CHECK: %[[ZP:.*]] = onnx.Constant dense<10> : tensor<ui16>
// CHECK: %[[S_DQ:.*]] = onnx.Constant dense<1.000000e-01> : tensor<f32>
// CHECK: %[[S_Q:.*]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK: %[[Q:.*]] = "onnx.QuantizeLinear"(%arg0, %[[S_Q]], %[[ZP]])
// CHECK-SAME: {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64}
// CHECK-SAME: : (tensor<10x1xf32>, tensor<f32>, tensor<ui16>) -> tensor<10x1xui16>
// CHECK: %[[DQ:.*]] = "onnx.DequantizeLinear"(%[[Q]], %[[S_DQ]], %[[ZP]])
// CHECK-SAME: {axis = 1 : si64, block_size = 0 : si64}
// CHECK-SAME: : (tensor<10x1xui16>, tensor<f32>, tensor<ui16>) -> tensor<10x1xf32>
// CHECK-NOT: "onnx.Div"
// CHECK: return %[[DQ]] : tensor<10x1xf32>

// ============================================================================
// Test A: Fold happened into DQ  →  chainStartQ = Quantize AFTER the BinOp
// Expect: the Q→DQ pair AFTER the BinOp is removed by Remove_Q_Plus_DQ.
// ============================================================================

func.func @cleanup_qdq_after_binop_folded_into_dq(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // Activation path: Q_act -> DQ_act
  %s_act  = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %zp_act = onnx.Constant dense<0>            : tensor<i8>
  %q_act  = "onnx.QuantizeLinear"(%arg0, %s_act, %zp_act)
            : (tensor<4xf32>, tensor<f32>, tensor<i8>) -> tensor<4xi8>
  %dq_act = "onnx.DequantizeLinear"(%q_act, %s_act, %zp_act)
            : (tensor<4xi8>, tensor<f32>, tensor<i8>) -> tensor<4xf32>
  %c_q  = onnx.Constant dense<10>             : tensor<i8>
  %c_s  = onnx.Constant dense<5.000000e+00>   : tensor<f32>
  %c_zp = onnx.Constant dense<0>              : tensor<i8>
  %dq_c = "onnx.DequantizeLinear"(%c_q, %c_s, %c_zp)
          : (tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<f32>
  %add = "onnx.Add"(%dq_act, %dq_c) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  %s_out  = onnx.Constant dense<1.000000e-01> : tensor<f32>
  %zp_out = onnx.Constant dense<0>            : tensor<i8>
  %q_out  = "onnx.QuantizeLinear"(%add, %s_out, %zp_out)
            : (tensor<4xf32>, tensor<f32>, tensor<i8>) -> tensor<4xi8>

  %dq_out = "onnx.DequantizeLinear"(%q_out, %s_out, %zp_out)
            : (tensor<4xi8>, tensor<f32>, tensor<i8>) -> tensor<4xf32>
  return %dq_out : tensor<4xf32>
}

// CHECK-LABEL: func.func @cleanup_qdq_after_binop_folded_into_dq(
// CHECK-SAME:                         %arg0: tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[S:.*]]      = onnx.Constant dense<1.000000e-01> : tensor<f32>
// CHECK: %[[ZP_DQ:.*]]  = onnx.Constant dense<0>            : tensor<i8>
// CHECK: %[[ZP_Q:.*]]   = onnx.Constant dense<-1>           : tensor<i8>
// CHECK: %[[Q:.*]] = "onnx.QuantizeLinear"(%arg0, %[[S]], %[[ZP_Q]])
// CHECK-SAME: {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64}
// CHECK-SAME: : (tensor<4xf32>, tensor<f32>, tensor<i8>) -> tensor<4xi8>
// CHECK: %[[DQ:.*]] = "onnx.DequantizeLinear"(%[[Q]], %[[S]], %[[ZP_DQ]])
// CHECK-SAME: {axis = 1 : si64, block_size = 0 : si64}
// CHECK-SAME: : (tensor<4xi8>, tensor<f32>, tensor<i8>) -> tensor<4xf32>
// CHECK-NOT: "onnx.Add"
// CHECK: return %[[DQ]] : tensor<4xf32>


// ============================================================================
// Test B: Fold happened into Q  →  chainStartQ = Quantize feeding DQ_act.x
// Expect: the UPSTREAM activation Q→DQ pair is removed by Remove_Q_Plus_DQ.
// ============================================================================

func.func @cleanup_qdq_activation_pair_folded_into_q(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // Activation path in fp, then (Q_act -> DQ_act) feeding the BinOp:
  %s_act  = onnx.Constant dense<2.500000e-01> : tensor<f32>
  %zp_act = onnx.Constant dense<0>            : tensor<i8>
  %q_act  = "onnx.QuantizeLinear"(%arg0, %s_act, %zp_act)
            : (tensor<4xf32>, tensor<f32>, tensor<i8>) -> tensor<4xi8>
  %dq_act = "onnx.DequantizeLinear"(%q_act, %s_act, %zp_act)
            : (tensor<4xi8>, tensor<f32>, tensor<i8>) -> tensor<4xf32>
  %c_q  = onnx.Constant dense<4>              : tensor<i8>
  %c_s  = onnx.Constant dense<1.000000e+00>   : tensor<f32>
  %c_zp = onnx.Constant dense<0>              : tensor<i8>
  %dq_c = "onnx.DequantizeLinear"(%c_q, %c_s, %c_zp)
          : (tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<f32>
  %mul = "onnx.Mul"(%dq_act, %dq_c) : (tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  %s_out  = onnx.Constant dense<1.250000e-01> : tensor<f32>
  %zp_out = onnx.Constant dense<0>            : tensor<i8>
  %q_out2 = "onnx.QuantizeLinear"(%mul, %s_out, %zp_out)
            : (tensor<4xf32>, tensor<f32>, tensor<i8>) -> tensor<4xi8>
  %dq_out2 = "onnx.DequantizeLinear"(%q_out2, %s_out, %zp_out)
             : (tensor<4xi8>, tensor<f32>, tensor<i8>) -> tensor<4xf32>

  return %dq_out2 : tensor<4xf32>
}

// CHECK-LABEL: func.func @cleanup_qdq_activation_pair_folded_into_q(
// CHECK-SAME: %arg0: tensor<4xf32>) -> tensor<4xf32>
// CHECK: %[[S_DQ:.*]] = onnx.Constant dense<1.250000e-01> : tensor<f32>
// CHECK: %[[ZP:.*]]   = onnx.Constant dense<0>            : tensor<i8>
// CHECK: %[[S_Q:.*]]  = onnx.Constant dense<3.125000e-02> : tensor<f32>
// CHECK: %[[Q:.*]] = "onnx.QuantizeLinear"(%arg0, %[[S_Q]], %[[ZP]])
// CHECK-SAME: {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64}
// CHECK-SAME: : (tensor<4xf32>, tensor<f32>, tensor<i8>) -> tensor<4xi8>
// CHECK: %[[DQ:.*]] = "onnx.DequantizeLinear"(%[[Q]], %[[S_DQ]], %[[ZP]])
// CHECK-SAME: {axis = 1 : si64, block_size = 0 : si64}
// CHECK-SAME: : (tensor<4xi8>, tensor<f32>, tensor<i8>) -> tensor<4xf32>
// CHECK-NOT: "onnx.Add"
// CHECK-NOT: "onnx.Mul"
// CHECK-NOT: "onnx.Div"
// CHECK-NOT: "onnx.Sub
// CHECK: return %[[DQ]] : tensor<4xf32>