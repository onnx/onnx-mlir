// RUN: onnx-mlir-opt %s -fold-quantized-binary | FileCheck %s
// RUN: onnx-mlir-opt %s -canonicalize -fold-quantized-binary | FileCheck --check-prefix=CANON %s

// from: @test_fold_mul_case_b_safe
func.func @dq_dq_mul_update_input(%arg0: tensor<10x1xf32>) -> tensor<10x1xf32> {
  %0 = onnx.Constant dense<0> : tensor<ui16>
  %1 = onnx.Constant dense<5.78499521E-6> : tensor<f32>
  %2 = onnx.Constant {value = dense<65535> : tensor<ui16>} : tensor<!quant.uniform<u16:f32, 0.0015259023057296872>>
  %3 = onnx.Constant dense<10> : tensor<ui16>
  %4 = onnx.Constant dense<1.000000e-01> : tensor<f32>
  %5 = "onnx.QuantizeLinear"(%arg0, %1, %0) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<10x1xf32>, tensor<f32>, tensor<ui16>) -> tensor<10x1xui16>
  %6 = quant.scast %5 : tensor<10x1xui16> to tensor<10x1x!quant.uniform<u16:f32, 5.7849952099786606E-6>>
  %7 = "onnx.Identity"(%6) : (tensor<10x1x!quant.uniform<u16:f32, 5.7849952099786606E-6>>) -> tensor<10x1x!quant.uniform<u16:f32, 5.7849952099786606E-6>>
  %8 = "onnx.Mul"(%7, %2) : (tensor<10x1x!quant.uniform<u16:f32, 5.7849952099786606E-6>>, tensor<!quant.uniform<u16:f32, 0.0015259023057296872>>) -> tensor<10x1x!quant.uniform<u16:f32, 0.10000000149011612:10>>
  %9 = "onnx.Identity"(%8) : (tensor<10x1x!quant.uniform<u16:f32, 0.10000000149011612:10>>) -> tensor<10x1x!quant.uniform<u16:f32, 0.10000000149011612:10>>
  %10 = quant.scast %9 : tensor<10x1x!quant.uniform<u16:f32, 0.10000000149011612:10>> to tensor<10x1xui16>
  %11 = "onnx.DequantizeLinear"(%10, %4, %3) {axis = 1 : si64, block_size = 0 : si64} : (tensor<10x1xui16>, tensor<f32>, tensor<ui16>) -> tensor<10x1xf32>
  return %11 : tensor<10x1xf32>
}

// CHECK-LABEL: @dq_dq_mul_update_input
// CHECK: onnx.Identity
// CHECK-NEXT: quant.scast
// CHECK-SAME: quant.uniform<u16:f32, 9.9999993108212947E-4:10>
// CHECK-NEXT: quant.scast
// CHECK-NEXT: onnx.Identity

// from: @caseB_bothDQ_constViaDQ1_foldIntoQ
func.func @dq_dq_add_update_input(%arg0: tensor<1x4xf32>) -> tensor<1x4xi8> {
  %0 = onnx.Constant {value = dense<10> : tensor<i8>} : tensor<!quant.uniform<i8:f32, 5.000000e+00>>
  %1 = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %2 = onnx.Constant dense<0> : tensor<i8>
  %3 = "onnx.QuantizeLinear"(%arg0, %1, %2) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
  %4 = quant.scast %3 : tensor<1x4xi8> to tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %5 = "onnx.Identity"(%4) : (tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %6 = "onnx.Add"(%5, %0) : (tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<!quant.uniform<i8:f32, 5.000000e+00>>) -> tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %7 = "onnx.Identity"(%6) : (tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %8 = quant.scast %7 : tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>> to tensor<1x4xi8>
  return %8 : tensor<1x4xi8>
}

// CHECK-LABEL: @dq_dq_add_update_input
// CHECK: onnx.Identity
// CHECK-NEXT: quant.scast
// CHECK-SAME: quant.uniform<i8:f32, 5.000000e-01:100>
// CHECK-NEXT: quant.scast
// CHECK-NEXT: onnx.Identity

// from: @caseA_lhsDQ_rhsConst_foldIntoQ
func.func @dq_const_add_update_input(%arg0: tensor<1x4xf32>) -> tensor<1x4xi8> {
  %0 = onnx.Constant dense<1.000000e+01> : tensor<f32>
  %1 = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %2 = onnx.Constant dense<0> : tensor<i8>
  %3 = "onnx.QuantizeLinear"(%arg0, %1, %2) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
  %4 = quant.scast %3 : tensor<1x4xi8> to tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %5 = "onnx.Identity"(%4) : (tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %6 = "onnx.Add"(%5, %0) : (tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<f32>) -> tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>
  %7 = "onnx.Identity"(%6) : (tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>) -> tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>
  %8 = quant.scast %7 : tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>> to tensor<1x4xi8>
  return %8 : tensor<1x4xi8>
}

// CHECK-LABEL: @dq_const_add_update_input
// CHECK: onnx.Identity
// CHECK-NEXT: quant.scast
// CHECK-SAME: tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612:100>>
// CHECK-NEXT: quant.scast
// CHECK-NEXT: onnx.Identity

// from: @caseA_rev_rhsDQ_lhsConst_foldIntoQ
func.func @const_dq_add_update_input(%arg0: tensor<1x4xf32>) -> tensor<1x4xi8> {
  %0 = onnx.Constant dense<1.000000e+01> : tensor<f32>
  %1 = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %2 = onnx.Constant dense<0> : tensor<i8>
  %3 = "onnx.QuantizeLinear"(%arg0, %1, %2) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
  %4 = quant.scast %3 : tensor<1x4xi8> to tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %5 = "onnx.Transpose"(%4) {perm = [1, 0]} : (tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<4x1x!quant.uniform<i8:f32, 5.000000e-01>>
  %6 = "onnx.Add"(%0, %5) : (tensor<f32>, tensor<4x1x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<4x1x!quant.uniform<i8:f32, 0.10000000149011612>>
  %7 = "onnx.Transpose"(%6) {perm = [1, 0]} : (tensor<4x1x!quant.uniform<i8:f32, 0.10000000149011612>>) -> tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>
  %8 = quant.scast %7 : tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>> to tensor<1x4xi8>
  return %8 : tensor<1x4xi8>
}

// CANON-LABEL: @const_dq_add_update_input
// CANON: onnx.Transpose
// CANON-NEXT: quant.scast
// CANON-SAME: quant.uniform<i8:f32, 0.10000000149011612:100>
// CANON-NEXT: quant.scast
// CANON-NEXT: onnx.Transpose

// skipped: @caseB_constViaReshape_foldIntoQ
// reason: When constprop is applied, Reshape will be folded.

// from: @branchBefore_foldIntoDQ
func.func @dq_const_add_multiuse_update_output(%arg0: tensor<1x4xf32>) -> (tensor<1x4xf32>, tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>) {
  %0 = onnx.Constant dense<2.000000e-01> : tensor<f32>
  %1 = onnx.Constant dense<1.000000e+01> : tensor<f32>
  %2 = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %3 = onnx.Constant dense<0> : tensor<i8>
  %4 = "onnx.QuantizeLinear"(%arg0, %2, %3) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
  %5 = quant.scast %4 : tensor<1x4xi8> to tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>
  %6 = "onnx.Identity"(%5) : (tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>) -> tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>
  %7 = "onnx.Add"(%6, %1) : (tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>, tensor<f32>) -> tensor<1x4x!quant.uniform<i8:f32, 0.20000000298023224>>
  %8 = "onnx.Identity"(%7) : (tensor<1x4x!quant.uniform<i8:f32, 0.20000000298023224>>) -> tensor<1x4x!quant.uniform<i8:f32, 0.20000000298023224>>
  %9 = quant.scast %8 : tensor<1x4x!quant.uniform<i8:f32, 0.20000000298023224>> to tensor<1x4xi8>
  %10 = "onnx.DequantizeLinear"(%9, %0, %3) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xi8>, tensor<f32>, tensor<i8>) -> tensor<1x4xf32>
  %11 = "onnx.Abs"(%6) : (tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>) -> tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>
  return %10, %11 : tensor<1x4xf32>, tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>
}

// CHECK-LABEL: @dq_const_add_multiuse_update_output
// CHECK: onnx.Identity
// CHECK-NEXT: quant.scast
// CHECK-NEXT: quant.scast
// CHECK-SAME: quant.uniform<i8:f32, 0.10000000149011612:-100>
// CHECK-NEXT: onnx.Identity

// from: @test_kval_0_dst_q_mul
func.func @div_by_zero_update_input(%arg0: tensor<10x1xf32>) -> tensor<10x1xf32> {
  %0 = onnx.Constant dense<0> : tensor<ui16>
  %1 = onnx.Constant dense<5.78499521E-6> : tensor<f32>
  %2 = onnx.Constant dense<10> : tensor<ui16>
  %3 = onnx.Constant dense<1.000000e-01> : tensor<f32>
  %4 = onnx.Constant {value = dense<0> : tensor<ui16>} : tensor<!quant.uniform<u16:f32, 0.0015259023057296872>>
  %5 = "onnx.QuantizeLinear"(%arg0, %1, %0) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<10x1xf32>, tensor<f32>, tensor<ui16>) -> tensor<10x1xui16>
  %6 = quant.scast %5 : tensor<10x1xui16> to tensor<10x1x!quant.uniform<u16:f32, 5.7849952099786606E-6>>
  %7 = "onnx.Identity"(%6) : (tensor<10x1x!quant.uniform<u16:f32, 5.7849952099786606E-6>>) -> tensor<10x1x!quant.uniform<u16:f32, 5.7849952099786606E-6>>
  %8 = "onnx.Mul"(%7, %4) : (tensor<10x1x!quant.uniform<u16:f32, 5.7849952099786606E-6>>, tensor<!quant.uniform<u16:f32, 0.0015259023057296872>>) -> tensor<10x1x!quant.uniform<u16:f32, 0.10000000149011612:10>>
  %9 = quant.scast %8 : tensor<10x1x!quant.uniform<u16:f32, 0.10000000149011612:10>> to tensor<10x1xui16>
  %10 = "onnx.DequantizeLinear"(%9, %3, %2) {axis = 1 : si64, block_size = 0 : si64} : (tensor<10x1xui16>, tensor<f32>, tensor<ui16>) -> tensor<10x1xf32>
  return %10 : tensor<10x1xf32>
}

// CHECK-LABEL: @div_by_zero_update_input
// CHECK: onnx.Mul

// from: @guard_div_into_dq_k_zero
func.func @div_by_zero_update_output(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %0 = onnx.Constant {value = dense<7> : tensor<i8>} : tensor<!quant.uniform<i8:f32, 5.000000e-01:7>>
  %1 = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %2 = onnx.Constant dense<0> : tensor<i8>
  %3 = "onnx.QuantizeLinear"(%arg0, %1, %2) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
  %4 = quant.scast %3 : tensor<1x4xi8> to tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %5 = "onnx.Div"(%4, %0) : (tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<!quant.uniform<i8:f32, 5.000000e-01:7>>) -> tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %6 = "onnx.Identity"(%5) : (tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %7 = quant.scast %6 : tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>> to tensor<1x4xi8>
  %8 = "onnx.DequantizeLinear"(%7, %1, %2) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xi8>, tensor<f32>, tensor<i8>) -> tensor<1x4xf32>
  return %8 : tensor<1x4xf32>
}

// CHECK-LABEL: @div_by_zero_update_output
// CHECK: onnx.Div

// from: @sub_weight_first_operand_no_fold
func.func @const_act_sub_negative(%arg0: tensor<1x4xi8>) -> tensor<1x4xi8> {
  %0 = onnx.Constant dense<1.000000e+01> : tensor<f32>
  %1 = quant.scast %arg0 : tensor<1x4xi8> to tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %2 = "onnx.Identity"(%1) : (tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %3 = "onnx.Sub"(%0, %2) : (tensor<f32>, tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>
  %4 = "onnx.Identity"(%3) : (tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>) -> tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>
  %5 = quant.scast %4 : tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>> to tensor<1x4xi8>
  return %5 : tensor<1x4xi8>
}

// CHECK-LABEL: @const_act_sub_negative
// CHECK: onnx.Sub


// from: @div_weight_first_operand_no_fold
func.func @const_act_div_negative(%arg0: tensor<1x4xi8>) -> tensor<1x4xi8> {
  %0 = onnx.Constant dense<2.000000e+00> : tensor<f32>
  %1 = quant.scast %arg0 : tensor<1x4xi8> to tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %2 = "onnx.Identity"(%1) : (tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %3 = "onnx.Div"(%0, %2) : (tensor<f32>, tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>
  %4 = "onnx.Identity"(%3) : (tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>) -> tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>
  %5 = quant.scast %4 : tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>> to tensor<1x4xi8>
  return %5 : tensor<1x4xi8>
}

// CHECK-LABEL: @const_act_div_negative
// CHECK : onnx.Div

// from: @zp_overflow_ui8
func.func @zp_overflow_ui8_negative(%arg0: tensor<1x4xui8>) -> tensor<1x4xui8> {
  %0 = onnx.Constant dense<1.000000e+03> : tensor<f32>
  %1 = quant.scast %arg0 : tensor<1x4xui8> to tensor<1x4x!quant.uniform<u8:f32, 1.000000e+00:255>>
  %2 = "onnx.Identity"(%1) : (tensor<1x4x!quant.uniform<u8:f32, 1.000000e+00:255>>) -> tensor<1x4x!quant.uniform<u8:f32, 1.000000e+00:255>>
  %3 = "onnx.Add"(%1, %0) : (tensor<1x4x!quant.uniform<u8:f32, 1.000000e+00:255>>, tensor<f32>) -> tensor<1x4x!quant.uniform<u8:f32, 5.000000e-01>>
  %4 = "onnx.Identity"(%3) : (tensor<1x4x!quant.uniform<u8:f32, 5.000000e-01>>) -> tensor<1x4x!quant.uniform<u8:f32, 5.000000e-01>>
  %5 = quant.scast %4 : tensor<1x4x!quant.uniform<u8:f32, 5.000000e-01>> to tensor<1x4xui8>
  return %5 : tensor<1x4xui8>
}

// CHECK-LABEL: @zp_overflow_ui8_negative
// CHECK: onnx.Add

// from: @zp_underflow_i8
func.func @zp_underflow_i8_negative(%arg0: tensor<1x4xi8>) -> tensor<1x4xi8> {
  %0 = onnx.Constant dense<5.000000e+02> : tensor<f32>
  %1 = quant.scast %arg0 : tensor<1x4xi8> to tensor<1x4x!quant.uniform<i8:f32, 1.000000e+00:-128>>
  %2 = "onnx.Identity"(%1) : (tensor<1x4x!quant.uniform<i8:f32, 1.000000e+00:-128>>) -> tensor<1x4x!quant.uniform<i8:f32, 1.000000e+00:-128>>
  %3 = "onnx.Sub"(%2, %0) : (tensor<1x4x!quant.uniform<i8:f32, 1.000000e+00:-128>>, tensor<f32>) -> tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %4 = "onnx.Identity"(%3) : (tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %5 = quant.scast %4 : tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>> to tensor<1x4xi8>
  return %5 : tensor<1x4xi8>
}

// CHECK-LABEL: @zp_underflow_i8_negative
// CHECK: onnx.Sub

// skipped: @mul_by_zero_fold_into_q
// reason: duplicate

// from: @zp_overflow_ui4
func.func @zp_overflow_ui4_negative(%arg0: tensor<1x4xui4>) -> tensor<1x4xui4> {
  %0 = onnx.Constant dense<1.000000e+02> : tensor<f32>
  %1 = quant.scast %arg0 : tensor<1x4xui4> to tensor<1x4x!quant.uniform<u4:f32, 1.000000e+00:15>>
  %2 = "onnx.Identity"(%1) : (tensor<1x4x!quant.uniform<u4:f32, 1.000000e+00:15>>) -> tensor<1x4x!quant.uniform<u4:f32, 1.000000e+00:15>>
  %3 = "onnx.Add"(%2, %0) : (tensor<1x4x!quant.uniform<u4:f32, 1.000000e+00:15>>, tensor<f32>) -> tensor<1x4x!quant.uniform<u4:f32, 0.10000000149011612>>
  %4 = "onnx.Identity"(%3) : (tensor<1x4x!quant.uniform<u4:f32, 0.10000000149011612>>) -> tensor<1x4x!quant.uniform<u4:f32, 0.10000000149011612>>
  %5 = quant.scast %4 : tensor<1x4x!quant.uniform<u4:f32, 0.10000000149011612>> to tensor<1x4xui4>
  return %5 : tensor<1x4xui4>
}

// CHECK-LABEL: @zp_overflow_ui4_negative
// CHECK: onnx.Add

// skipped: @qdq_weight_sub_first_operand
// skipped: @zp_overflow_i8_add, @zp_underflow_i8_sub
// skipped: @zp_underflow_ui16_sub, @zp_overflow_ui16_add
// skipped: @zp_overflow_i16_sub_into_dq
// skipped: @valid_add_no_overflow, @valid_sub_no_underflow
// reason: redundant

// skipped: @qdq_chain_add, @qdq_chain_mul, @qdq_chain_div_into_dq, @qdq_chain_sub
// reason: Not applicable, as Q-DQ chain will be removed in quant-types pass

// skipped: @div_by_zero_fold_into_dq
// reason: redundant

// Multi-element splat constant should still fold (getConstant accepts splats of any shape)
func.func @dq_splat_const_add_update_input(%arg0: tensor<1x4xf32>) -> tensor<1x4xi8> {
  %0 = onnx.Constant dense<1.000000e+01> : tensor<1x4xf32>
  %1 = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %2 = onnx.Constant dense<0> : tensor<i8>
  %3 = "onnx.QuantizeLinear"(%arg0, %1, %2) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
  %4 = quant.scast %3 : tensor<1x4xi8> to tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %5 = "onnx.Identity"(%4) : (tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %6 = "onnx.Add"(%5, %0) : (tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<1x4xf32>) -> tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>
  %7 = "onnx.Identity"(%6) : (tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>) -> tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>
  %8 = quant.scast %7 : tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>> to tensor<1x4xi8>
  return %8 : tensor<1x4xi8>
}

// CHECK-LABEL: @dq_splat_const_add_update_input
// CHECK: onnx.Identity
// CHECK-NEXT: quant.scast
// CHECK-SAME: tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612:100>>
// CHECK-NEXT: quant.scast
// CHECK-NEXT: onnx.Identity

// Non-splat constant should not fold
func.func @non_splat_const_add_negative(%arg0: tensor<1x4xi8>) -> tensor<1x4xi8> {
  %0 = onnx.Constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %1 = quant.scast %arg0 : tensor<1x4xi8> to tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %2 = "onnx.Identity"(%1) : (tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %3 = "onnx.Add"(%2, %0) : (tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<4xf32>) -> tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>
  %4 = "onnx.Identity"(%3) : (tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>) -> tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>
  %5 = quant.scast %4 : tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>> to tensor<1x4xi8>
  return %5 : tensor<1x4xi8>
}

// CHECK-LABEL: @non_splat_const_add_negative
// CHECK: onnx.Add

// Proper conversion to float type
func.func @dq_dq_mul_update_input_conversion(%arg0: tensor<1x1x1x3072xui16>) -> tensor<1x1x1x3072xui16> {
  %0 = onnx.Constant {value = dense<65535> : tensor<ui16>} : tensor<!quant.uniform<u16:f32, 3.0518043786287308E-4>>
  %1 = quant.scast %arg0 : tensor<1x1x1x3072xui16> to tensor<1x1x1x3072x!quant.uniform<u16:f32, 1.5259021893143654E-5:65535>>
  %2 = "onnx.Identity"(%1) : (tensor<1x1x1x3072x!quant.uniform<u16:f32, 1.5259021893143654E-5:65535>>) -> tensor<1x1x1x3072x!quant.uniform<u16:f32, 1.5259021893143654E-5:65535>>
  %3 = "onnx.Mul"(%2, %0) : (tensor<1x1x1x3072x!quant.uniform<u16:f32, 1.5259021893143654E-5:65535>>, tensor<!quant.uniform<u16:f32, 3.0518043786287308E-4>>) -> tensor<1x1x1x3072x!quant.uniform<u16:f32, 3.0518043786287308E-4:65535>>
  %4 = "onnx.Identity"(%3) : (tensor<1x1x1x3072x!quant.uniform<u16:f32, 3.0518043786287308E-4:65535>>) -> tensor<1x1x1x3072x!quant.uniform<u16:f32, 3.0518043786287308E-4:65535>>
  %5 = quant.scast %4 : tensor<1x1x1x3072x!quant.uniform<u16:f32, 3.0518043786287308E-4:65535>> to tensor<1x1x1x3072xui16>
  return %5 : tensor<1x1x1x3072xui16>
}

// CHECK-LABEL: @dq_dq_mul_update_input_conversion
// CHECK: onnx.Identity
// CHECK-SAME: ([[qType:tensor<.*>]])
// CHECK-SAME: -> [[qType]]
// CHECK-NEXT: quant.scast
// CHECK-SAME: tensor<1x1x1x3072x!quant.uniform<u16:f32, 1.5259021893143654E-5:65535>> to tensor<1x1x1x3072xui16>

// ResultNames propagation
func.func @ResultNames_update_input(%arg0: tensor<1x4xf32>) -> tensor<1x4xi8> {
  %0 = onnx.Constant {value = dense<10> : tensor<i8>} : tensor<!quant.uniform<i8:f32, 5.000000e+00>>
  %1 = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %2 = onnx.Constant dense<0> : tensor<i8>
  %3 = "onnx.QuantizeLinear"(%arg0, %1, %2) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
  %4 = quant.scast %3 : tensor<1x4xi8> to tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %5 = "onnx.Identity"(%4) : (tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %6 = "onnx.Add"(%5, %0) {ResultNames = ["add_Quant_output"]} : (tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<!quant.uniform<i8:f32, 5.000000e+00>>) -> tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %7 = "onnx.Identity"(%6) : (tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %8 = quant.scast %7 : tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>> to tensor<1x4xi8>
  return %8 : tensor<1x4xi8>
}

// CHECK-LABEL: @ResultNames_update_input
// CHECK: onnx.Identity
// CHECK-SAME: ResultNames = ["add_Quant_output"]
// CHECK-NEXT: quant.scast
// CHECK-SAME: quant.uniform<i8:f32, 5.000000e-01:100>
// CHECK-NEXT: quant.scast
// CHECK-NEXT: onnx.Identity

func.func @ResultNames_update_output(%arg0: tensor<1x4xf32>) -> (tensor<1x4xf32>, tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>) {
  %0 = onnx.Constant dense<2.000000e-01> : tensor<f32>
  %1 = onnx.Constant dense<1.000000e+01> : tensor<f32>
  %2 = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %3 = onnx.Constant dense<0> : tensor<i8>
  %4 = "onnx.QuantizeLinear"(%arg0, %2, %3) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
  %5 = quant.scast %4 : tensor<1x4xi8> to tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>
  %6 = "onnx.Identity"(%5) : (tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>) -> tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>
  %7 = "onnx.Add"(%6, %1) {ResultNames = ["add_Quant_output"]} : (tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>, tensor<f32>) -> tensor<1x4x!quant.uniform<i8:f32, 0.20000000298023224>>
  %8 = "onnx.Identity"(%7) : (tensor<1x4x!quant.uniform<i8:f32, 0.20000000298023224>>) -> tensor<1x4x!quant.uniform<i8:f32, 0.20000000298023224>>
  %9 = quant.scast %8 : tensor<1x4x!quant.uniform<i8:f32, 0.20000000298023224>> to tensor<1x4xi8>
  %10 = "onnx.DequantizeLinear"(%9, %0, %3) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xi8>, tensor<f32>, tensor<i8>) -> tensor<1x4xf32>
  %11 = "onnx.Abs"(%6) : (tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>) -> tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>
  return %10, %11 : tensor<1x4xf32>, tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>
}

// CHECK-LABEL: @ResultNames_update_output
// CHECK: onnx.Identity
// CHECK-NEXT: quant.scast
// CHECK-SAME: quant.uniform<i8:f32, 0.10000000149011612>
// CHECK-NEXT: quant.scast
// CHECK-NOT: ResultNames = ["add_Quant_output"]
// CHECK-SAME: quant.uniform<i8:f32, 0.10000000149011612:-100>
// CHECK-NEXT: onnx.Identity

// Binary followed by scast should be replaced with a single scast
func.func @dq_dq_mul_scast_update_input(%arg0: tensor<10x512x!quant.uniform<u16:f32, 1.152162531070644E-5:15586>>, %arg1: tensor<512x1x!quant.uniform<u16:f32, 1.4544223631673958E-5:16488>>) -> tensor<10x1xf32> {
  %0 = onnx.Constant dense<6.14215212E-4> : tensor<f32>
  %1 = onnx.Constant dense<922> : tensor<ui16>
  %2 = onnx.Constant {value = dense<65535> : tensor<ui16>} : tensor<!quant.uniform<u16:f32, 0.0015259023057296872>>
  %3 = "onnx.MatMul"(%arg0, %arg1) {ResultNames = ["MatMul_QuantizeLinear_Output"]} : (tensor<10x512x!quant.uniform<u16:f32, 1.152162531070644E-5:15586>>, tensor<512x1x!quant.uniform<u16:f32, 1.4544223631673958E-5:16488>>) -> tensor<10x1x!quant.uniform<u16:f32, 6.1421515056281351E-6:922>>
  %4 = "onnx.Mul"(%3, %2) {ResultNames = ["Mul_QuantizeLinear_Output"]} : (tensor<10x1x!quant.uniform<u16:f32, 6.1421515056281351E-6:922>>, tensor<!quant.uniform<u16:f32, 0.0015259023057296872>>) -> tensor<10x1x!quant.uniform<u16:f32, 6.1421521240845323E-4:922>>
  %5 = quant.scast %4 : tensor<10x1x!quant.uniform<u16:f32, 6.1421521240845323E-4:922>> to tensor<10x1xui16>
  %6 = "onnx.DequantizeLinear"(%5, %0, %1) {ResultNames = ["Graph_Output"], axis = 1 : si64, block_size = 0 : si64, dtype_frozen = 1 : i64} : (tensor<10x1xui16>, tensor<f32>, tensor<ui16>) -> tensor<10x1xf32>
  return %6 : tensor<10x1xf32>
}

// CHECK-LABEL: @dq_dq_mul_scast_update_input
// CHECK: onnx.MatMul
// CHECK-NEXT: quant.scast
// CHECK-SAME: quant.uniform<u16:f32, 6.1421515056281351E-6:922>
// CHECK-NEXT: onnx.DequantizeLinear
