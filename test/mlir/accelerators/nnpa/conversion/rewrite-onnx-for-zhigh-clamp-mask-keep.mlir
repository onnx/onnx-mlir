// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --rewrite-onnx-for-zhigh %s -split-input-file | FileCheck %s

// COM: Mask constant clamping in --rewrite-onnx-for-zhigh: cases where every
// COM: constant must stay unchanged, one per documented restriction. Nothing
// COM: changes, so there is nothing to ground; the clamped cases are in
// COM: rewrite-onnx-for-zhigh-clamp-mask.mlir.

// -----

// COM: The mask itself is also returned, so its constant is kept.
func.func @test_mask_escapes(%pad: tensor<1x4xi1> {onnx.name = "pad"}, %s: tensor<3x4xf32> {onnx.name = "s"}) -> (tensor<3x4xf32> {onnx.name = "y"}, tensor<1x4xf32> {onnx.name = "m"}) {
  %min = onnx.Constant dense<-3.40282347E+38> : tensor<f32>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %m = "onnx.Where"(%pad, %min, %zero) : (tensor<1x4xi1>, tensor<f32>, tensor<f32>) -> tensor<1x4xf32>
  %a = "onnx.Add"(%s, %m) : (tensor<3x4xf32>, tensor<1x4xf32>) -> tensor<3x4xf32>
  %y = "onnx.Softmax"(%a) {axis = -1 : si64} : (tensor<3x4xf32>) -> tensor<3x4xf32>
  return %y, %m : tensor<3x4xf32>, tensor<1x4xf32>


// CHECK-LABEL:  func.func @test_mask_escapes
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x4xi1> {onnx.name = "pad"}, [[PARAM_1_:%.+]]: tensor<3x4xf32> {onnx.name = "s"}) -> (tensor<3x4xf32> {onnx.name = "y"}, tensor<1x4xf32> {onnx.name = "m"}) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<-3.40282347E+38> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Where"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]]) : (tensor<1x4xi1>, tensor<f32>, tensor<f32>) -> tensor<1x4xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Add"([[PARAM_1_]], [[VAR_2_]]) : (tensor<3x4xf32>, tensor<1x4xf32>) -> tensor<3x4xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Softmax"([[VAR_3_]]) <{axis = -1 : si64}> : (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK:           return [[VAR_4_]], [[VAR_2_]] : tensor<3x4xf32>, tensor<1x4xf32>
// CHECK:         }

}

// -----

// COM: A mask scaled by Mul ((1 - mask) * -FLT_MAX): Mul is not followed.
func.func @test_mul_mask(%mask: tensor<1x4xf32> {onnx.name = "mask"}, %s: tensor<3x4xf32> {onnx.name = "s"}) -> (tensor<3x4xf32> {onnx.name = "y"}) {
  %min = onnx.Constant dense<-3.40282347E+38> : tensor<f32>
  %one = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %inv = "onnx.Sub"(%one, %mask) : (tensor<f32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  %m = "onnx.Mul"(%inv, %min) : (tensor<1x4xf32>, tensor<f32>) -> tensor<1x4xf32>
  %a = "onnx.Add"(%s, %m) : (tensor<3x4xf32>, tensor<1x4xf32>) -> tensor<3x4xf32>
  %y = "onnx.Softmax"(%a) {axis = -1 : si64} : (tensor<3x4xf32>) -> tensor<3x4xf32>
  return %y : tensor<3x4xf32>


// CHECK-LABEL:  func.func @test_mul_mask
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x4xf32> {onnx.name = "mask"}, [[PARAM_1_:%.+]]: tensor<3x4xf32> {onnx.name = "s"}) -> (tensor<3x4xf32> {onnx.name = "y"}) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<-3.40282347E+38> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Shape"([[PARAM_0_]]) <{start = 0 : si64}> : (tensor<1x4xf32>) -> tensor<2xi64>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Expand"([[VAR_1_]], [[VAR_2_]]) : (tensor<f32>, tensor<2xi64>) -> tensor<1x4xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Sub"([[VAR_3_]], [[PARAM_0_]]) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Shape"([[VAR_4_]]) <{start = 0 : si64}> : (tensor<1x4xf32>) -> tensor<2xi64>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Expand"([[VAR_0_]], [[VAR_5_]]) : (tensor<f32>, tensor<2xi64>) -> tensor<1x4xf32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Mul"([[VAR_4_]], [[VAR_6_]]) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
// CHECK:           [[VAR_8_:%.+]] = "onnx.Add"([[PARAM_1_]], [[VAR_7_]]) : (tensor<3x4xf32>, tensor<1x4xf32>) -> tensor<3x4xf32>
// CHECK:           [[VAR_9_:%.+]] = "onnx.Softmax"([[VAR_8_]]) <{axis = -1 : si64}> : (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK:           return [[VAR_9_]] : tensor<3x4xf32>
// CHECK:         }

}

// -----

// COM: The second operand of Sub is negated, so it is not followed.
func.func @test_sub_rhs(%pad: tensor<1x4xi1> {onnx.name = "pad"}, %s: tensor<3x4xf32> {onnx.name = "s"}) -> (tensor<3x4xf32> {onnx.name = "y"}) {
  %min = onnx.Constant dense<-3.40282347E+38> : tensor<f32>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %m = "onnx.Where"(%pad, %min, %zero) : (tensor<1x4xi1>, tensor<f32>, tensor<f32>) -> tensor<1x4xf32>
  %a = "onnx.Sub"(%s, %m) : (tensor<3x4xf32>, tensor<1x4xf32>) -> tensor<3x4xf32>
  %y = "onnx.Softmax"(%a) {axis = -1 : si64} : (tensor<3x4xf32>) -> tensor<3x4xf32>
  return %y : tensor<3x4xf32>


// CHECK-LABEL:  func.func @test_sub_rhs
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x4xi1> {onnx.name = "pad"}, [[PARAM_1_:%.+]]: tensor<3x4xf32> {onnx.name = "s"}) -> (tensor<3x4xf32> {onnx.name = "y"}) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<-3.40282347E+38> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Where"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]]) : (tensor<1x4xi1>, tensor<f32>, tensor<f32>) -> tensor<1x4xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Sub"([[PARAM_1_]], [[VAR_2_]]) : (tensor<3x4xf32>, tensor<1x4xf32>) -> tensor<3x4xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Softmax"([[VAR_3_]]) <{axis = -1 : si64}> : (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK:           return [[VAR_4_]] : tensor<3x4xf32>
// CHECK:         }

}

// -----

// COM: A Cast that changes the element type (f64 to f32) is not followed.
func.func @test_cast_type_change(%pad: tensor<1x4xi1> {onnx.name = "pad"}, %s: tensor<3x4xf32> {onnx.name = "s"}) -> (tensor<3x4xf32> {onnx.name = "y"}) {
  %min = onnx.Constant dense<-1.7976931348623157E+308> : tensor<f64>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f64>
  %m = "onnx.Where"(%pad, %min, %zero) : (tensor<1x4xi1>, tensor<f64>, tensor<f64>) -> tensor<1x4xf64>
  %c = "onnx.Cast"(%m) {saturate = 1 : si64, to = f32} : (tensor<1x4xf64>) -> tensor<1x4xf32>
  %a = "onnx.Add"(%s, %c) : (tensor<3x4xf32>, tensor<1x4xf32>) -> tensor<3x4xf32>
  %y = "onnx.Softmax"(%a) {axis = -1 : si64} : (tensor<3x4xf32>) -> tensor<3x4xf32>
  return %y : tensor<3x4xf32>


// CHECK-LABEL:  func.func @test_cast_type_change
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x4xi1> {onnx.name = "pad"}, [[PARAM_1_:%.+]]: tensor<3x4xf32> {onnx.name = "s"}) -> (tensor<3x4xf32> {onnx.name = "y"}) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<-1.7976931348623157E+308> : tensor<f64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Where"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]]) : (tensor<1x4xi1>, tensor<f64>, tensor<f64>) -> tensor<1x4xf64>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Cast"([[VAR_2_]]) <{saturate = 1 : si64, to = f32}> : (tensor<1x4xf64>) -> tensor<1x4xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Add"([[PARAM_1_]], [[VAR_3_]]) : (tensor<3x4xf32>, tensor<1x4xf32>) -> tensor<3x4xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Softmax"([[VAR_4_]]) <{axis = -1 : si64}> : (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK:           return [[VAR_5_]] : tensor<3x4xf32>
// CHECK:         }

}

// -----

// COM: A boolean attn_mask of onnx.Attention is not a sink (only float masks
// COM: are added to the scores); the unrelated -FLT_MAX constant is kept.
func.func @test_attention_bool_mask(%Q: tensor<1x1x2x2xf32> {onnx.name = "q"}, %K: tensor<1x1x3x2xf32> {onnx.name = "k"}, %V: tensor<1x1x3x2xf32> {onnx.name = "v"}, %pad: tensor<1x1x1x3xi1> {onnx.name = "pad"}) -> (tensor<1x1x2x2xf32> {onnx.name = "y"}, tensor<1x1x1x3xf32> {onnx.name = "m"}) {
  %none = "onnx.NoValue"() {value} : () -> none
  %min = onnx.Constant dense<-3.40282347E+38> : tensor<f32>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %m = "onnx.Where"(%pad, %min, %zero) : (tensor<1x1x1x3xi1>, tensor<f32>, tensor<f32>) -> tensor<1x1x1x3xf32>
  %Y, %pk, %pv, %qk = "onnx.Attention"(%Q, %K, %V, %pad, %none, %none, %none) {is_causal = 0 : si64, qk_matmul_output_mode = 0 : si64, softcap = 0.000000e+00 : f32} : (tensor<1x1x2x2xf32>, tensor<1x1x3x2xf32>, tensor<1x1x3x2xf32>, tensor<1x1x1x3xi1>, none, none, none) -> (tensor<1x1x2x2xf32>, none, none, none)
  return %Y, %m : tensor<1x1x2x2xf32>, tensor<1x1x1x3xf32>


// CHECK-LABEL:  func.func @test_attention_bool_mask
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x2x2xf32> {onnx.name = "q"}, [[PARAM_1_:%.+]]: tensor<1x1x3x2xf32> {onnx.name = "k"}, [[PARAM_2_:%.+]]: tensor<1x1x3x2xf32> {onnx.name = "v"}, [[PARAM_3_:%.+]]: tensor<1x1x1x3xi1> {onnx.name = "pad"}) -> (tensor<1x1x2x2xf32> {onnx.name = "y"}, tensor<1x1x1x3xf32> {onnx.name = "m"}) {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() <{value}> : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<-3.40282347E+38> : tensor<f32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Where"([[PARAM_3_]], [[VAR_1_]], [[VAR_2_]]) : (tensor<1x1x1x3xi1>, tensor<f32>, tensor<f32>) -> tensor<1x1x1x3xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) <{perm = [0, 1, 3, 2]}> : (tensor<1x1x3x2xf32>) -> tensor<1x1x2x3xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Shape"([[PARAM_0_]]) <{start = 0 : si64}> : (tensor<1x1x2x2xf32>) -> tensor<4xi64>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Slice"([[VAR_5_]], [[VAR_8_]], [[VAR_9_]], [[VAR_7_]], [[VAR_10_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_13_:%.+]] = "onnx.Concat"([[VAR_12_]], [[VAR_11_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<2xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_13_]]) <{allowzero = 0 : si64}> : (tensor<1x1x2x2xf32>, tensor<3xi64>) -> tensor<1x2x2xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.Shape"([[VAR_4_]]) <{start = 0 : si64}> : (tensor<1x1x2x3xf32>) -> tensor<4xi64>
// CHECK-DAG:       [[VAR_16_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_17_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_18_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_19_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_20_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.Slice"([[VAR_15_]], [[VAR_18_]], [[VAR_19_]], [[VAR_17_]], [[VAR_20_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_22_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_23_:%.+]] = "onnx.Concat"([[VAR_22_]], [[VAR_21_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<2xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_24_:%.+]] = "onnx.Reshape"([[VAR_4_]], [[VAR_23_]]) <{allowzero = 0 : si64}> : (tensor<1x1x2x3xf32>, tensor<3xi64>) -> tensor<1x2x3xf32>
// CHECK-DAG:       [[VAR_25_:%.+]] = "onnx.MatMul"([[VAR_14_]], [[VAR_24_]]) : (tensor<1x2x2xf32>, tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
// CHECK-DAG:       [[VAR_26_:%.+]] = "onnx.Shape"([[PARAM_0_]]) <{start = 0 : si64}> : (tensor<1x1x2x2xf32>) -> tensor<4xi64>
// CHECK-DAG:       [[VAR_27_:%.+]] = "onnx.Shape"([[VAR_4_]]) <{start = 0 : si64}> : (tensor<1x1x2x3xf32>) -> tensor<4xi64>
// CHECK-DAG:       [[VAR_28_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_29_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_30_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:       [[VAR_31_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_32_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:       [[VAR_33_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_34_:%.+]] = "onnx.Slice"([[VAR_27_]], [[VAR_32_]], [[VAR_33_]], [[VAR_28_]], [[VAR_29_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_35_:%.+]] = "onnx.Slice"([[VAR_26_]], [[VAR_28_]], [[VAR_30_]], [[VAR_28_]], [[VAR_29_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_36_:%.+]] = "onnx.Concat"([[VAR_35_]], [[VAR_34_]]) <{axis = 0 : si64}> : (tensor<3xi64>, tensor<1xi64>) -> tensor<4xi64>
// CHECK-DAG:       [[VAR_37_:%.+]] = "onnx.Reshape"([[VAR_25_]], [[VAR_36_]]) <{allowzero = 0 : si64}> : (tensor<1x2x3xf32>, tensor<4xi64>) -> tensor<1x1x2x3xf32>
// CHECK-DAG:       [[VAR_38_:%.+]] = onnx.Constant dense<0.707106769> : tensor<1xf32>
// CHECK:           [[VAR_39_:%.+]] = "onnx.Shape"([[VAR_37_]]) <{start = 0 : si64}> : (tensor<1x1x2x3xf32>) -> tensor<4xi64>
// CHECK:           [[VAR_40_:%.+]] = "onnx.Expand"([[VAR_38_]], [[VAR_39_]]) : (tensor<1xf32>, tensor<4xi64>) -> tensor<1x1x2x3xf32>
// CHECK-DAG:       [[VAR_41_:%.+]] = "onnx.Mul"([[VAR_37_]], [[VAR_40_]]) : (tensor<1x1x2x3xf32>, tensor<1x1x2x3xf32>) -> tensor<1x1x2x3xf32>
// CHECK-DAG:       [[VAR_42_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<1xf32>
// CHECK-DAG:       [[VAR_43_:%.+]] = onnx.Constant dense<-1.000000e+09> : tensor<1xf32>
// CHECK:           [[VAR_44_:%.+]] = "onnx.Where"([[PARAM_3_]], [[VAR_42_]], [[VAR_43_]]) : (tensor<1x1x1x3xi1>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x1x3xf32>
// CHECK:           [[VAR_45_:%.+]] = "onnx.Add"([[VAR_41_]], [[VAR_44_]]) : (tensor<1x1x2x3xf32>, tensor<1x1x1x3xf32>) -> tensor<1x1x2x3xf32>
// CHECK-DAG:       [[VAR_46_:%.+]] = "onnx.Shape"([[VAR_45_]]) <{start = 0 : si64}> : (tensor<1x1x2x3xf32>) -> tensor<4xi64>
// CHECK-DAG:       [[VAR_47_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_48_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_49_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_50_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_51_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_52_:%.+]] = "onnx.Slice"([[VAR_46_]], [[VAR_49_]], [[VAR_50_]], [[VAR_48_]], [[VAR_51_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_53_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_54_:%.+]] = "onnx.Concat"([[VAR_53_]], [[VAR_52_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<2xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_55_:%.+]] = "onnx.Reshape"([[VAR_45_]], [[VAR_54_]]) <{allowzero = 0 : si64}> : (tensor<1x1x2x3xf32>, tensor<3xi64>) -> tensor<1x2x3xf32>
// CHECK-DAG:       [[VAR_56_:%.+]] = "onnx.Softmax"([[VAR_55_]]) <{axis = -1 : si64}> : (tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
// CHECK-DAG:       [[VAR_57_:%.+]] = "onnx.Shape"([[VAR_45_]]) <{start = 0 : si64}> : (tensor<1x1x2x3xf32>) -> tensor<4xi64>
// CHECK:           [[VAR_58_:%.+]] = "onnx.Reshape"([[VAR_56_]], [[VAR_57_]]) <{allowzero = 0 : si64}> : (tensor<1x2x3xf32>, tensor<4xi64>) -> tensor<1x1x2x3xf32>
// CHECK-DAG:       [[VAR_59_:%.+]] = "onnx.Shape"([[VAR_58_]]) <{start = 0 : si64}> : (tensor<1x1x2x3xf32>) -> tensor<4xi64>
// CHECK-DAG:       [[VAR_60_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_61_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_62_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_63_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_64_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_65_:%.+]] = "onnx.Slice"([[VAR_59_]], [[VAR_62_]], [[VAR_63_]], [[VAR_61_]], [[VAR_64_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_66_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_67_:%.+]] = "onnx.Concat"([[VAR_66_]], [[VAR_65_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<2xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_68_:%.+]] = "onnx.Reshape"([[VAR_58_]], [[VAR_67_]]) <{allowzero = 0 : si64}> : (tensor<1x1x2x3xf32>, tensor<3xi64>) -> tensor<1x2x3xf32>
// CHECK-DAG:       [[VAR_69_:%.+]] = "onnx.Shape"([[PARAM_2_]]) <{start = 0 : si64}> : (tensor<1x1x3x2xf32>) -> tensor<4xi64>
// CHECK-DAG:       [[VAR_70_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_71_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_72_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_73_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_74_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_75_:%.+]] = "onnx.Slice"([[VAR_69_]], [[VAR_72_]], [[VAR_73_]], [[VAR_71_]], [[VAR_74_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_76_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_77_:%.+]] = "onnx.Concat"([[VAR_76_]], [[VAR_75_]]) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<2xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_78_:%.+]] = "onnx.Reshape"([[PARAM_2_]], [[VAR_77_]]) <{allowzero = 0 : si64}> : (tensor<1x1x3x2xf32>, tensor<3xi64>) -> tensor<1x3x2xf32>
// CHECK-DAG:       [[VAR_79_:%.+]] = "onnx.MatMul"([[VAR_68_]], [[VAR_78_]]) : (tensor<1x2x3xf32>, tensor<1x3x2xf32>) -> tensor<1x2x2xf32>
// CHECK-DAG:       [[VAR_80_:%.+]] = "onnx.Shape"([[VAR_58_]]) <{start = 0 : si64}> : (tensor<1x1x2x3xf32>) -> tensor<4xi64>
// CHECK-DAG:       [[VAR_81_:%.+]] = "onnx.Shape"([[PARAM_2_]]) <{start = 0 : si64}> : (tensor<1x1x3x2xf32>) -> tensor<4xi64>
// CHECK-DAG:       [[VAR_82_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_83_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_84_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:       [[VAR_85_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_86_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:       [[VAR_87_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_88_:%.+]] = "onnx.Slice"([[VAR_81_]], [[VAR_86_]], [[VAR_87_]], [[VAR_82_]], [[VAR_83_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_89_:%.+]] = "onnx.Slice"([[VAR_80_]], [[VAR_82_]], [[VAR_84_]], [[VAR_82_]], [[VAR_83_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_90_:%.+]] = "onnx.Concat"([[VAR_89_]], [[VAR_88_]]) <{axis = 0 : si64}> : (tensor<3xi64>, tensor<1xi64>) -> tensor<4xi64>
// CHECK-DAG:       [[VAR_91_:%.+]] = "onnx.Reshape"([[VAR_79_]], [[VAR_90_]]) <{allowzero = 0 : si64}> : (tensor<1x2x2xf32>, tensor<4xi64>) -> tensor<1x1x2x2xf32>
// CHECK-DAG:       [[VAR_92_:%.+]] = "onnx.NoValue"() : () -> none
// CHECK:           return [[VAR_91_]], [[VAR_3_]] : tensor<1x1x2x2xf32>, tensor<1x1x1x3xf32>
// CHECK:         }

}

// -----

// COM: A fill of ordinary magnitude (-1e4) is already safe and is left as is.
func.func @test_small_fill(%pad: tensor<1x4xi1> {onnx.name = "pad"}, %s: tensor<3x4xf32> {onnx.name = "s"}) -> (tensor<3x4xf32> {onnx.name = "y"}) {
  %min = onnx.Constant dense<-1.000000e+04> : tensor<f32>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %m = "onnx.Where"(%pad, %min, %zero) : (tensor<1x4xi1>, tensor<f32>, tensor<f32>) -> tensor<1x4xf32>
  %a = "onnx.Add"(%s, %m) : (tensor<3x4xf32>, tensor<1x4xf32>) -> tensor<3x4xf32>
  %y = "onnx.Softmax"(%a) {axis = -1 : si64} : (tensor<3x4xf32>) -> tensor<3x4xf32>
  return %y : tensor<3x4xf32>

// CHECK-LABEL:  func.func @test_small_fill
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x4xi1> {onnx.name = "pad"}, [[PARAM_1_:%.+]]: tensor<3x4xf32> {onnx.name = "s"}) -> (tensor<3x4xf32> {onnx.name = "y"}) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<-1.000000e+04> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Where"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]]) : (tensor<1x4xi1>, tensor<f32>, tensor<f32>) -> tensor<1x4xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Add"([[PARAM_1_]], [[VAR_2_]]) : (tensor<3x4xf32>, tensor<1x4xf32>) -> tensor<3x4xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Softmax"([[VAR_3_]]) <{axis = -1 : si64}> : (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK:           return [[VAR_4_]] : tensor<3x4xf32>
// CHECK:         }

}

