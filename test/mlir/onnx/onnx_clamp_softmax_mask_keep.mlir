// RUN: onnx-mlir-opt --clamp-softmax-mask %s -split-input-file | FileCheck %s

// COM: Cases where --clamp-softmax-mask must leave every constant unchanged,
// COM: one per documented restriction of the pass. Nothing changes, so there
// COM: is nothing to ground; the clamped cases are in
// COM: onnx_clamp_softmax_mask.mlir.

// -----

// COM: The -FLT_MAX constant is also used as a sentinel outside the mask
// COM: (Max with the scores, returned), so it is kept.
func.func @test_shared_sentinel(%pad: tensor<1x4xi1> {onnx.name = "pad"}, %s: tensor<3x4xf32> {onnx.name = "s"}) -> (tensor<3x4xf32> {onnx.name = "y"}, tensor<3x4xf32> {onnx.name = "r"}) {
  %min = onnx.Constant dense<-3.40282347E+38> : tensor<f32>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %m = "onnx.Where"(%pad, %min, %zero) : (tensor<1x4xi1>, tensor<f32>, tensor<f32>) -> tensor<1x4xf32>
  %a = "onnx.Add"(%s, %m) : (tensor<3x4xf32>, tensor<1x4xf32>) -> tensor<3x4xf32>
  %y = "onnx.Softmax"(%a) {axis = -1 : si64} : (tensor<3x4xf32>) -> tensor<3x4xf32>
  %r = "onnx.Max"(%s, %min) : (tensor<3x4xf32>, tensor<f32>) -> tensor<3x4xf32>
  return %y, %r : tensor<3x4xf32>, tensor<3x4xf32>

// CHECK-LABEL:  func.func @test_shared_sentinel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x4xi1> {onnx.name = "pad"}, [[PARAM_1_:%.+]]: tensor<3x4xf32> {onnx.name = "s"}) -> (tensor<3x4xf32> {onnx.name = "y"}, tensor<3x4xf32> {onnx.name = "r"}) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<-3.40282347E+38> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Where"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]]) : (tensor<1x4xi1>, tensor<f32>, tensor<f32>) -> tensor<1x4xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Add"([[PARAM_1_]], [[VAR_2_]]) : (tensor<3x4xf32>, tensor<1x4xf32>) -> tensor<3x4xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Softmax"([[VAR_3_]]) <{axis = -1 : si64}> : (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Max"([[PARAM_1_]], [[VAR_0_]]) : (tensor<3x4xf32>, tensor<f32>) -> tensor<3x4xf32>
// CHECK:           return [[VAR_4_]], [[VAR_5_]] : tensor<3x4xf32>, tensor<3x4xf32>
// CHECK:         }

}

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
// CHECK:           [[VAR_2_:%.+]] = "onnx.Sub"([[VAR_1_]], [[PARAM_0_]]) : (tensor<f32>, tensor<1x4xf32>) -> tensor<1x4xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Mul"([[VAR_2_]], [[VAR_0_]]) : (tensor<1x4xf32>, tensor<f32>) -> tensor<1x4xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Add"([[PARAM_1_]], [[VAR_3_]]) : (tensor<3x4xf32>, tensor<1x4xf32>) -> tensor<3x4xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Softmax"([[VAR_4_]]) <{axis = -1 : si64}> : (tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK:           return [[VAR_5_]] : tensor<3x4xf32>
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
// CHECK:           [[VAR_3_:%.+]] = "onnx.Where"([[PARAM_3_]], [[VAR_1_]], [[VAR_2_]]) : (tensor<1x1x1x3xi1>, tensor<f32>, tensor<f32>) -> tensor<1x1x1x3xf32>
// CHECK:           [[VAR_Y_:%.+]], [[VAR_present_key_:%.+]], [[VAR_present_value_:%.+]], [[VAR_qk_matmul_output_:%.+]] = "onnx.Attention"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[PARAM_3_]], [[VAR_0_]], [[VAR_0_]], [[VAR_0_]]) <{is_causal = 0 : si64, qk_matmul_output_mode = 0 : si64, softcap = 0.000000e+00 : f32}> : (tensor<1x1x2x2xf32>, tensor<1x1x3x2xf32>, tensor<1x1x3x2xf32>, tensor<1x1x1x3xi1>, none, none, none) -> (tensor<1x1x2x2xf32>, none, none, none)
// CHECK:           return [[VAR_Y_]], [[VAR_3_]] : tensor<1x1x2x2xf32>, tensor<1x1x1x3xf32>
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

