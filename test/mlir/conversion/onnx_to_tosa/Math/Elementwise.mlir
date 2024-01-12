// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s

func.func @test_relu(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Relu"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_relu
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.clamp"([[PARAM_0_]]) <{max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64}> : (tensor<10x10xf32>) -> tensor<10x10xf32>
// CHECK-NEXT:      return [[VAR_0_]] : tensor<10x10xf32>
// CHECK-NEXT:    }
}

// -----

func.func @test_relu_dynamic(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Relu"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  func @test_relu_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x10xf32>) -> tensor<?x10xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] =  "tosa.clamp"([[PARAM_0_]]) <{max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64}> : (tensor<?x10xf32>) -> tensor<?x10xf32>
// CHECK-NEXT:      return [[VAR_0_]] : tensor<?x10xf32>
// CHECK-NEXT:    }
}

// -----

func.func @test_neg(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Neg"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_neg
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.negate"([[PARAM_0_]]) : (tensor<10x10xf32>) -> tensor<10x10xf32>
}

// -----

func.func @test_floor(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Floor"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_floor
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.floor"([[PARAM_0_]]) : (tensor<10x10xf32>) -> tensor<10x10xf32>
}

// -----

func.func @test_add(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x1xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
  "func.return"(%0) : (tensor<13x21x1xf32>) -> ()
// CHECK-LABEL:  func @test_add
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xf32>, [[PARAM_1_:%.+]]: tensor<13x21x1xf32>) -> tensor<13x21x1xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.add"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<13x21x1xf32>, tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
}

// -----

func.func @test_add_broadcast(%arg0: tensor<13x21x1xf32>, %arg1: tensor<1xf32>) -> tensor<13x21x1xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<1xf32>) -> tensor<13x21x1xf32>
  "func.return"(%0) : (tensor<13x21x1xf32>) -> ()
// CHECK-LABEL:  func.func @test_add_broadcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xf32>, [[PARAM_1_:%.+]]: tensor<1xf32>) -> tensor<13x21x1xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.reshape"([[PARAM_1_]]) <{new_shape = array<i64: 1, 1, 1>}> : (tensor<1xf32>) -> tensor<1x1x1xf32>
// CHECK:           [[VAR_1_:%.+]] = "tosa.add"([[PARAM_0_]], [[VAR_0_]]) : (tensor<13x21x1xf32>, tensor<1x1x1xf32>) -> tensor<13x21x1xf32>
// CHECK:           return [[VAR_1_]] : tensor<13x21x1xf32>
}


// -----

func.func @test_sub(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x1xf32> {
  %0 = "onnx.Sub"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
  "func.return"(%0) : (tensor<13x21x1xf32>) -> ()
// CHECK-LABEL:  func @test_sub
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xf32>, [[PARAM_1_:%.+]]: tensor<13x21x1xf32>) -> tensor<13x21x1xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.sub"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<13x21x1xf32>, tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
}

// -----

func.func @test_sub_broadcast(%arg0: tensor<13x21x1xf32>, %arg1: tensor<1xf32>) -> tensor<13x21x1xf32> {
  %0 = "onnx.Sub"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<1xf32>) -> tensor<13x21x1xf32>
  "func.return"(%0) : (tensor<13x21x1xf32>) -> ()
// CHECK-LABEL:  func.func @test_sub_broadcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xf32>, [[PARAM_1_:%.+]]: tensor<1xf32>) -> tensor<13x21x1xf32> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.reshape"([[PARAM_1_]]) <{new_shape = array<i64: 1, 1, 1>}> : (tensor<1xf32>) -> tensor<1x1x1xf32>
// CHECK:           [[VAR_1_:%.+]] = "tosa.sub"([[PARAM_0_]], [[VAR_0_]]) : (tensor<13x21x1xf32>, tensor<1x1x1xf32>) -> tensor<13x21x1xf32>
// CHECK:           return [[VAR_1_]] : tensor<13x21x1xf32>
}

// -----

func.func @test_mul(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x1xf32> {
  %0 = "onnx.Mul"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
  "func.return"(%0) : (tensor<13x21x1xf32>) -> ()
// CHECK-LABEL:  func @test_mul
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xf32>, [[PARAM_1_:%.+]]: tensor<13x21x1xf32>) -> tensor<13x21x1xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.mul"([[PARAM_0_]], [[PARAM_1_]]) <{shift = 0 : i32}> : (tensor<13x21x1xf32>, tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
}

// -----

func.func @test_mul_rank_broadcast(%arg0: tensor<13x21x1xf32>, %arg1: tensor<21x1xf32>) -> tensor<13x21x1xf32> {
  %0 = "onnx.Mul"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<21x1xf32>) -> tensor<13x21x1xf32>
  "func.return"(%0) : (tensor<13x21x1xf32>) -> ()
// CHECK-LABEL:  func @test_mul_rank_broadcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xf32>, [[PARAM_1_:%.+]]: tensor<21x1xf32>) -> tensor<13x21x1xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.reshape"([[PARAM_1_]]) <{new_shape = array<i64: 1, 21, 1>}> : (tensor<21x1xf32>) -> tensor<1x21x1xf32>
// CHECK-NEXT:      [[VAR_1_:%.+]] = "tosa.mul"([[PARAM_0_]], [[VAR_0_]]) <{shift = 0 : i32}> : (tensor<13x21x1xf32>, tensor<1x21x1xf32>) -> tensor<13x21x1xf32>
}

// -----

func.func @test_mul_rank_broadcast2(%arg0: tensor<21x1xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x1xf32> {
  %0 = "onnx.Mul"(%arg0, %arg1) : (tensor<21x1xf32>, tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
  "func.return"(%0) : (tensor<13x21x1xf32>) -> ()
// CHECK-LABEL:  func @test_mul_rank_broadcast2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<21x1xf32>, [[PARAM_1_:%.+]]: tensor<13x21x1xf32>) -> tensor<13x21x1xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.reshape"([[PARAM_0_]]) <{new_shape = array<i64: 1, 21, 1>}> : (tensor<21x1xf32>) -> tensor<1x21x1xf32>
// CHECK-NEXT:      [[VAR_1_:%.+]] = "tosa.mul"([[VAR_0_]], [[PARAM_1_]]) <{shift = 0 : i32}> : (tensor<1x21x1xf32>, tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
}

// -----

func.func @test_div(%arg0: tensor<13x21x1xi32>, %arg1: tensor<13x21x1xi32>) -> tensor<13x21x1xi32> {
  %0 = "onnx.Div"(%arg0, %arg1) : (tensor<13x21x1xi32>, tensor<13x21x1xi32>) -> tensor<13x21x1xi32>
  "func.return"(%0) : (tensor<13x21x1xi32>) -> ()
// CHECK-LABEL:  func @test_div
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xi32>, [[PARAM_1_:%.+]]: tensor<13x21x1xi32>) -> tensor<13x21x1xi32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.div"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<13x21x1xi32>, tensor<13x21x1xi32>) -> tensor<13x21x1xi32>
}

// -----

func.func @test_div_broadcast(%arg0: tensor<13x21x1xi32>, %arg1: tensor<1xi32>) -> tensor<13x21x1xi32> {
  %0 = "onnx.Div"(%arg0, %arg1) : (tensor<13x21x1xi32>, tensor<1xi32>) -> tensor<13x21x1xi32>
  "func.return"(%0) : (tensor<13x21x1xi32>) -> ()
// CHECK-LABEL:  func @test_div_broadcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xi32>, [[PARAM_1_:%.+]]: tensor<1xi32>) -> tensor<13x21x1xi32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.reshape"([[PARAM_1_]]) <{new_shape = array<i64: 1, 1, 1>}> : (tensor<1xi32>) -> tensor<1x1x1xi32>
// CHECK-NEXT:      [[VAR_1_:%.+]] = "tosa.div"([[PARAM_0_]], [[VAR_0_]]) : (tensor<13x21x1xi32>, tensor<1x1x1xi32>) -> tensor<13x21x1xi32>
}

// -----

func.func @test_div_decomposed(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x1xf32> {
  %0 = "onnx.Div"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
  "func.return"(%0) : (tensor<13x21x1xf32>) -> ()
// CHECK-LABEL:  func @test_div_decomposed
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xf32>, [[PARAM_1_:%.+]]: tensor<13x21x1xf32>) -> tensor<13x21x1xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.reciprocal"([[PARAM_1_]]) : (tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
// CHECK-NEXT:      [[VAR_1_:%.+]] = "tosa.mul"([[PARAM_0_]], [[VAR_0_]]) <{shift = 0 : i32}> : (tensor<13x21x1xf32>, tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
}

// -----

func.func @test_leaky_relu(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "onnx.LeakyRelu"(%arg0) {alpha = 0.707330704  : f32} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
// CHECK-LABEL: test_leaky_relu
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x1xf32>}>
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() <{value = dense<0.707330704> : tensor<1x1x1xf32>}>
// CHECK-DAG: %[[VAR2:.*]] = "tosa.mul"(%arg0, %[[VAR1]]) <{shift = 0 : i32}>
// CHECK-DAG: %[[VAR3:.*]] = "tosa.greater_equal"(%arg0, %[[VAR0]])
// CHECK: %[[VAR6:.*]] = "tosa.select"(%[[VAR3]], %arg0, %[[VAR2]])
}

func.func @test_leaky_relu_bf16(%arg0: tensor<13x21x3xbf16>) -> tensor<13x21x3xbf16> {
  %0 = "onnx.LeakyRelu"(%arg0) {alpha = 0.707330704  : f32} : (tensor<13x21x3xbf16>) -> tensor<13x21x3xbf16>
  func.return %0 : tensor<13x21x3xbf16>
// CHECK-LABEL: test_leaky_relu_bf16
// CHECK-DAG: %[[VAR0:.*]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x1xbf16>}>
// CHECK-DAG: %[[VAR1:.*]] = "tosa.const"() <{value = dense<7.070310e-01> : tensor<1x1x1xbf16>}>
// CHECK-DAG: %[[VAR2:.*]] = "tosa.mul"(%arg0, %[[VAR1]]) <{shift = 0 : i32}>
// CHECK-DAG: %[[VAR3:.*]] = "tosa.greater_equal"(%arg0, %[[VAR0]])
// CHECK: %[[VAR6:.*]] = "tosa.select"(%[[VAR3]], %arg0, %[[VAR2]])
}

// -----

func.func @test_prelu(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
// CHECK-LABEL: test_prelu
// CHECK-DAG:       [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x1xf32>}>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.mul"(%arg0, %arg1) <{shift = 0 : i32}>
// CHECK:           [[VAR_2_:%.+]] = "tosa.greater_equal"(%arg0, [[VAR_0_]])
// CHECK:           [[VAR_3_:%.+]] = "tosa.select"([[VAR_2_]], %arg0, [[VAR_1_]])
}

func.func @test_prelu_bf16(%arg0: tensor<13x21x3xbf16>, %arg1: tensor<13x21x3xbf16>) -> tensor<13x21x3xbf16> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<13x21x3xbf16>, tensor<13x21x3xbf16>) -> tensor<13x21x3xbf16>
  func.return %0 : tensor<13x21x3xbf16>
// CHECK-LABEL: test_prelu_bf16
// CHECK-DAG:       [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x1xbf16>}> : () -> tensor<1x1x1xbf16>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.mul"(%arg0, %arg1) <{shift = 0 : i32}>
// CHECK:           [[VAR_2_:%.+]] = "tosa.greater_equal"(%arg0, [[VAR_0_]])
// CHECK:           [[VAR_3_:%.+]] = "tosa.select"([[VAR_2_]], %arg0, [[VAR_1_]])
}

// -----

func.func @test_selu_default_value(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "onnx.Selu"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
// CHECK-LABEL: test_selu_default_value
// CHECK-DAG:       [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<1.673260e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{value = dense<1.050700e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.exp"(%arg0)
// CHECK:           [[VAR_4_:%.+]] = "tosa.mul"([[VAR_3_]], [[VAR_0_]]) <{shift = 0 : i32}>
// CHECK-DAG:       [[VAR_5_:%.+]] = "tosa.sub"([[VAR_4_]], [[VAR_0_]])
// CHECK-DAG:       [[VAR_6_:%.+]] = "tosa.greater"(%arg0, [[VAR_2_]])
// CHECK:           [[VAR_7_:%.+]] = "tosa.select"([[VAR_6_]], %arg0, [[VAR_5_]])
// CHECK:           [[VAR_8_:%.+]] = "tosa.mul"([[VAR_7_]], [[VAR_1_]]) <{shift = 0 : i32}>
// CHECK:           return [[VAR_8_]] : tensor<13x21x3xf32>
}

func.func @test_selu(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "onnx.Selu"(%arg0) {alpha = 1.5 : f32, gamma = 2.0 : f32} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
// CHECK-LABEL: test_selu
// CHECK-DAG:       [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<1.500000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{value = dense<2.000000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.exp"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
// CHECK:           [[VAR_4_:%.+]] = "tosa.mul"([[VAR_3_]], [[VAR_0_]]) <{shift = 0 : i32}>
// CHECK-DAG:       [[VAR_5_:%.+]] = "tosa.sub"([[VAR_4_]], [[VAR_0_]])
// CHECK-DAG:       [[VAR_6_:%.+]] = "tosa.greater"(%arg0, [[VAR_2_]])
// CHECK:           [[VAR_7_:%.+]] = "tosa.select"([[VAR_6_]], %arg0, [[VAR_5_]])
// CHECK:           [[VAR_8_:%.+]] = "tosa.mul"([[VAR_7_]], [[VAR_1_]]) <{shift = 0 : i32}>
// CHECK:           return [[VAR_8_]] : tensor<13x21x3xf32>
}

// -----

func.func @test_softplus(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "onnx.Softplus"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
// CHECK-LABEL:  test_softplus
// CHECK-DAG:       [[VAR_0_:%.+]] = "tosa.exp"(%arg0)
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1x1xf32>}>
// CHECK:           [[VAR_2_:%.+]] = "tosa.add"([[VAR_0_]], [[VAR_1_]])
// CHECK:           [[VAR_3_:%.+]] = "tosa.log"([[VAR_2_]])
// CHECK:           return [[VAR_3_]] : tensor<13x21x3xf32>
}

// -----

func.func @test_thresholdedrelu(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "onnx.ThresholdedRelu"(%arg0) {alpha = 0.5 : f32} : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
// CHECK-LABEL: test_thresholdedrelu
// CHECK-DAG:       [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<5.000000e-01>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{value = dense<0.000000e+00>
// CHECK:           [[VAR_2_:%.+]] = "tosa.greater"(%arg0, [[VAR_0_]])
// CHECK:           [[VAR_3_:%.+]] = "tosa.select"([[VAR_2_]], %arg0, [[VAR_1_]])
// CHECK:           return [[VAR_3_]] : tensor<13x21x3xf32>
}



func.func @test_thresholdedrelu_default_value(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
  %0 = "onnx.ThresholdedRelu"(%arg0) : (tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
  func.return %0 : tensor<13x21x3xf32>
// CHECK-LABEL: test_thresholdedrelu_default_value
// CHECK-DAG:       [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
// CHECK:           [[VAR_2_:%.+]] = "tosa.greater"(%arg0, [[VAR_0_]])
// CHECK:           [[VAR_3_:%.+]] = "tosa.select"([[VAR_2_]], %arg0, [[VAR_1_]])
// CHECK:           return [[VAR_3_]] : tensor<13x21x3xf32>
}

// -----

func.func @test_sigmoid(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Sigmoid"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_sigmoid
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.sigmoid"([[PARAM_0_]]) : (tensor<10x10xf32>) -> tensor<10x10xf32>
// CHECK-NEXT:      return [[VAR_0_]] : tensor<10x10xf32>
// CHECK-NEXT:    }
}

// -----

func.func @test_ceil(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Ceil"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_ceil
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.ceil"([[PARAM_0_]]) : (tensor<10x10xf32>) -> tensor<10x10xf32>
// CHECK-NEXT:      return [[VAR_0_]] : tensor<10x10xf32>
// CHECK-NEXT:    }
}

// -----

func.func @test_exp(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Exp"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_exp
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.exp"([[PARAM_0_]]) : (tensor<10x10xf32>) -> tensor<10x10xf32>
// CHECK-NEXT:      return [[VAR_0_]] : tensor<10x10xf32>
// CHECK-NEXT:    }
}

// -----

func.func @test_log(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Log"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_log
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.log"([[PARAM_0_]]) : (tensor<10x10xf32>) -> tensor<10x10xf32>
// CHECK-NEXT:      return [[VAR_0_]] : tensor<10x10xf32>
// CHECK-NEXT:    }
}

// -----

func.func @test_reciprocal(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Reciprocal"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_reciprocal
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.reciprocal"([[PARAM_0_]]) : (tensor<10x10xf32>) -> tensor<10x10xf32>
// CHECK-NEXT:      return [[VAR_0_]] : tensor<10x10xf32>
// CHECK-NEXT:    }
}

// -----

func.func @test_tanh(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Tanh"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_tanh
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.tanh"([[PARAM_0_]]) : (tensor<10x10xf32>) -> tensor<10x10xf32>
// CHECK-NEXT:      return [[VAR_0_]] : tensor<10x10xf32>
// CHECK-NEXT:    }
}

// -----

func.func @test_clip(%arg0: tensor<3xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<3xi32> {
  %0 = "onnx.Clip"(%arg0, %arg1, %arg2) : (tensor<3xi32>, tensor<i32>, tensor<i32>) -> tensor<3xi32>
  return %0 : tensor<3xi32>
// CHECK-LABEL:  func @test_clip
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3xi32>, [[PARAM_1_:%.+]]: tensor<i32>, [[PARAM_2_:%.+]]: tensor<i32>) -> tensor<3xi32>
// CHECK-NEXT:     [[VAR_0_:%.+]] = "tosa.maximum"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<3xi32>, tensor<i32>) -> tensor<3xi32>
// CHECK-NEXT:     [[VAR_1_:%.+]] = "tosa.minimum"([[VAR_0_]], [[PARAM_2_]]) : (tensor<3xi32>, tensor<i32>) -> tensor<3xi32>
// CHECK-NEXT:     return [[VAR_1_]] : tensor<3xi32>
// CHECK-NEXT:   }
}

// -----

// Test when min is none
func.func @test_clip_default_min_f32(%arg0: tensor<3xf32>, %arg1: tensor<f32>) -> tensor<3xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Clip"(%arg0, %cst, %arg1) : (tensor<3xf32>, none, tensor<f32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
// CHECK-LABEL:  func @test_clip_default_min_f32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3xf32>, [[PARAM_1_:%.+]]: tensor<f32>) -> tensor<3xf32>
// CHECK-NEXT:     [[VAR_0_:%.+]] = "tosa.minimum"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
// CHECK-NEXT:     return [[VAR_0_]] : tensor<3xf32>
// CHECK-NEXT:   }
}

// -----

// Test when max is none
func.func @test_clip_default_max_bf16(%arg0: tensor<3xbf16>, %arg1: tensor<bf16>) -> tensor<3xbf16> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Clip"(%arg0, %arg1, %cst) : (tensor<3xbf16>, tensor<bf16>, none) -> tensor<3xbf16>
  return %0 : tensor<3xbf16>
// CHECK-LABEL:  func @test_clip_default_max_bf16
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3xbf16>, [[PARAM_1_:%.+]]: tensor<bf16>) -> tensor<3xbf16>
// CHECK-NEXT:     [[VAR_0_:%.+]] = "tosa.maximum"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<3xbf16>, tensor<bf16>) -> tensor<3xbf16>
// CHECK-NEXT:     return [[VAR_0_]] : tensor<3xbf16>
// CHECK-NEXT:   }
}

// -----

// Test when min and max are splat constants
func.func @test_clip_constant_minimum_maximum(%arg0: tensor<3xbf16>) -> tensor<3xbf16> {
  %cst1 = "onnx.Constant"() {value = dense<-2.0> : tensor<bf16>} : () -> tensor<bf16>
  %cst2 = "onnx.Constant"() {value = dense<[2.0]> : tensor<1xbf16>} : () -> tensor<1xbf16>
  %0 = "onnx.Clip"(%arg0, %cst1, %cst2) : (tensor<3xbf16>, tensor<bf16>, tensor<1xbf16>) -> tensor<3xbf16>
  return %0 : tensor<3xbf16>
// CHECK-LABEL:  func @test_clip_constant_minimum_maximum
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3xbf16>) -> tensor<3xbf16>
// CHECK-NEXT:     [[VAR_0_:%.+]] = "tosa.clamp"([[PARAM_0_]]) <{max_fp = 2.000000e+00 : f32, max_int = 2 : i64, min_fp = -2.000000e+00 : f32, min_int = -2 : i64}> : (tensor<3xbf16>) -> tensor<3xbf16>
// CHECK-NEXT:     return [[VAR_0_]] : tensor<3xbf16>
// CHECK-NEXT:   }
}

// -----

// Test when min and max are constants and min is non-splat.
func.func @test_clip_constant_minimum_maximum_non_splat(%arg0: tensor<3xi32>) -> tensor<3xi32> {
  %cst1 = "onnx.Constant"() {value = dense<[-1, 0, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
  %cst2 = "onnx.Constant"() {value = dense<[2]> : tensor<1xi32>} : () -> tensor<1xi32>
  %0 = "onnx.Clip"(%arg0, %cst1, %cst2) : (tensor<3xi32>, tensor<3xi32>, tensor<1xi32>) -> tensor<3xi32>
  return %0 : tensor<3xi32>
// CHECK-LABEL:  func @test_clip_constant_minimum_maximum_non_splat
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3xi32>) -> tensor<3xi32>
// CHECK-NEXT:     [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<[-1, 0, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
// CHECK-NEXT:     [[VAR_1_:%.+]] = "tosa.const"() <{value = dense<2> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK-NEXT:     [[VAR_2_:%.+]] = "tosa.maximum"([[PARAM_0_]], [[VAR_0_]]) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
// CHECK-NEXT:     [[VAR_3_:%.+]] = "tosa.minimum"([[VAR_2_]], [[VAR_1_]]) : (tensor<3xi32>, tensor<1xi32>) -> tensor<3xi32>
// CHECK-NEXT:     return [[VAR_3_]] : tensor<3xi32>
// CHECK-NEXT:   }
}

func.func @test_div_decomposed_broadcast(%arg0: tensor<13x21x1xf32>, %arg1: tensor<1xf32>) -> tensor<13x21x1xf32> {
  %0 = "onnx.Div"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<1xf32>) -> tensor<13x21x1xf32>
  "func.return"(%0) : (tensor<13x21x1xf32>) -> ()
// CHECK-LABEL:  func @test_div_decomposed_broadcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xf32>, [[PARAM_1_:%.+]]: tensor<1xf32>) -> tensor<13x21x1xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.reciprocal"([[PARAM_1_]]) : (tensor<1xf32>) -> tensor<1xf32>
// CHECK-NEXT:      [[VAR_1_:%.+]] = "tosa.reshape"([[VAR_0_]]) <{new_shape = array<i64: 1, 1, 1>}> : (tensor<1xf32>) -> tensor<1x1x1xf32>
// CHECK-NEXT:      [[VAR_2_:%.+]] = "tosa.mul"([[PARAM_0_]], [[VAR_1_]]) <{shift = 0 : i32}> : (tensor<13x21x1xf32>, tensor<1x1x1xf32>) -> tensor<13x21x1xf32>
}

// -----

func.func @test_pow(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x1xf32> {
  %0 = "onnx.Pow"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
  "func.return"(%0) : (tensor<13x21x1xf32>) -> ()
// CHECK-LABEL:  func @test_pow
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xf32>, [[PARAM_1_:%.+]]: tensor<13x21x1xf32>) -> tensor<13x21x1xf32> {
// CHECK-NEXT:      [[VAR_1_:%.+]] = "tosa.pow"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<13x21x1xf32>, tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
}

func.func @test_pow_broadcast(%arg0: tensor<13x21x1xf32>, %arg1: tensor<1xf32>) -> tensor<13x21x1xf32> {
  %0 = "onnx.Pow"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<1xf32>) -> tensor<13x21x1xf32>
  "func.return"(%0) : (tensor<13x21x1xf32>) -> ()
// CHECK-LABEL:  func @test_pow_broadcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xf32>, [[PARAM_1_:%.+]]: tensor<1xf32>) -> tensor<13x21x1xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.reshape"([[PARAM_1_]]) <{new_shape = array<i64: 1, 1, 1>}> : (tensor<1xf32>) -> tensor<1x1x1xf32>
// CHECK-NEXT:      [[VAR_1_:%.+]] = "tosa.pow"([[PARAM_0_]], [[VAR_0_]]) : (tensor<13x21x1xf32>, tensor<1x1x1xf32>) -> tensor<13x21x1xf32>
}

// -----

func.func @test_sqrt(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %0 = "onnx.Sqrt"(%arg0) : (tensor<3xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
// CHECK-LABEL:  func @test_sqrt
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3xf32>) -> tensor<3xf32>
// CHECK-NEXT:     [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK-NEXT:     [[VAR_1_:%.+]] = "tosa.pow"([[PARAM_0_]], [[VAR_0_]]) : (tensor<3xf32>, tensor<1xf32>) -> tensor<3xf32>
// CHECK-NEXT:     return [[VAR_1_]] : tensor<3xf32>
// CHECK-NEXT:   }
}

// -----

func.func @test_abs_i32(%arg0: tensor<3xi32>) -> tensor<3xi32> {
  %0 = "onnx.Abs"(%arg0) : (tensor<3xi32>) -> tensor<3xi32>
  return %0 : tensor<3xi32>
// CHECK-LABEL:  func @test_abs_i32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3xi32>) -> tensor<3xi32>
// CHECK-NEXT:     [[VAR_0_:%.+]] = "tosa.abs"([[PARAM_0_]]) : (tensor<3xi32>) -> tensor<3xi32>
// CHECK-NEXT:     return [[VAR_0_]] : tensor<3xi32>
// CHECK-NEXT:   }
}

func.func @test_abs_bf16(%arg0: tensor<3xbf16>) -> tensor<3xbf16> {
  %0 = "onnx.Abs"(%arg0) : (tensor<3xbf16>) -> tensor<3xbf16>
  return %0 : tensor<3xbf16>
// CHECK-LABEL:  func @test_abs_bf16
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3xbf16>) -> tensor<3xbf16>
// CHECK-NEXT:     [[VAR_0_:%.+]] = "tosa.abs"([[PARAM_0_]]) : (tensor<3xbf16>) -> tensor<3xbf16>
// CHECK-NEXT:     return [[VAR_0_]] : tensor<3xbf16>
// CHECK-NEXT:   }
}

// -----

func.func @test_erf_f32(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %0 = "onnx.Erf"(%arg0) : (tensor<3xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
// CHECK-LABEL:  func @test_erf_f32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3xf32>) -> tensor<3xf32>
// CHECK-NEXT:     [[VAR_0_:%.+]] = "tosa.erf"([[PARAM_0_]]) : (tensor<3xf32>) -> tensor<3xf32>
// CHECK-NEXT:     return [[VAR_0_]] : tensor<3xf32>
// CHECK-NEXT:   }
}

func.func @test_erf_bf16(%arg0: tensor<3xbf16>) -> tensor<3xbf16> {
  %0 = "onnx.Erf"(%arg0) : (tensor<3xbf16>) -> tensor<3xbf16>
  return %0 : tensor<3xbf16>
// CHECK-LABEL:  func @test_erf_bf16
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3xbf16>) -> tensor<3xbf16>
// CHECK-NEXT:     [[VAR_0_:%.+]] = "tosa.erf"([[PARAM_0_]]) : (tensor<3xbf16>) -> tensor<3xbf16>
// CHECK-NEXT:     return [[VAR_0_]] : tensor<3xbf16>
// CHECK-NEXT:   }
}

// -----

func.func @test_bitwise_not(%arg0 : tensor<10x10xi32>) -> tensor<10x10xi32> {
  %0 = "onnx.BitwiseNot"(%arg0) : (tensor<10x10xi32>) -> tensor<10x10xi32>
  "func.return"(%0) : (tensor<10x10xi32>) -> ()
// CHECK-LABEL:  func @test_bitwise_not
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xi32>) -> tensor<10x10xi32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.bitwise_not"([[PARAM_0_]]) : (tensor<10x10xi32>) -> tensor<10x10xi32>
// CHECK-NEXT:      return [[VAR_0_]] : tensor<10x10xi32>
// CHECK-NEXT:    }
}

// -----

func.func @test_not(%arg0 : tensor<10x10xi1>) -> tensor<10x10xi1> {
  %0 = "onnx.Not"(%arg0) : (tensor<10x10xi1>) -> tensor<10x10xi1>
  "func.return"(%0) : (tensor<10x10xi1>) -> ()
// CHECK-LABEL:  func @test_not
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xi1>) -> tensor<10x10xi1> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.logical_not"([[PARAM_0_]]) : (tensor<10x10xi1>) -> tensor<10x10xi1>
// CHECK-NEXT:      return [[VAR_0_]] : tensor<10x10xi1>
// CHECK-NEXT:    }
}

// -----

// Default values: alpha = 0.2, beta = 0.5
func.func @test_hardsigmoid_default_values_f32(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %0 = "onnx.HardSigmoid"(%arg0) : (tensor<3xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
// CHECK-LABEL:  func.func @test_hardsigmoid_default_values_f32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3xf32>) -> tensor<3xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<2.500000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{value = dense<2.000000e-01> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           [[VAR_2_:%.+]] = "tosa.add"([[PARAM_0_]], [[VAR_0_]]) : (tensor<3xf32>, tensor<1xf32>) -> tensor<3xf32>
// CHECK:           [[VAR_3_:%.+]] = "tosa.clamp"([[VAR_2_]]) <{max_fp = 5.000000e+00 : f32, max_int = 5 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64}> : (tensor<3xf32>) -> tensor<3xf32>
// CHECK:           [[VAR_4_:%.+]] = "tosa.mul"([[VAR_3_]], [[VAR_1_]]) <{shift = 0 : i32}> : (tensor<3xf32>, tensor<1xf32>) -> tensor<3xf32>
// CHECK:           return [[VAR_4_]] : tensor<3xf32>
}

func.func @test_hardsigmoid_default_values_f16(%arg0: tensor<3xf16>) -> tensor<3xf16> {
  %0 = "onnx.HardSigmoid"(%arg0) : (tensor<3xf16>) -> tensor<3xf16>
  return %0 : tensor<3xf16>
// CHECK-LABEL:  func @test_hardsigmoid_default_values_f16
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3xf16>) -> tensor<3xf16>
// CHECK-DAG:       [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<2.500000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{value = dense<1.999510e-01> : tensor<1xf16>}> : () -> tensor<1xf16>
// CHECK:           [[VAR_2_:%.+]] = "tosa.add"([[PARAM_0_]], [[VAR_0_]]) : (tensor<3xf16>, tensor<1xf16>) -> tensor<3xf16>
// CHECK:           [[VAR_3_:%.+]] = "tosa.clamp"([[VAR_2_]]) <{max_fp = 5.000000e+00 : f32, max_int = 5 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64}> : (tensor<3xf16>) -> tensor<3xf16>
// CHECK:           [[VAR_4_:%.+]] = "tosa.mul"([[VAR_3_]], [[VAR_1_]]) <{shift = 0 : i32}> : (tensor<3xf16>, tensor<1xf16>) -> tensor<3xf16>
// CHECK:           return [[VAR_4_]] : tensor<3xf16>
}

// alpha = 0.166666672, beta = 5.000000e-01
func.func @test_hardsigmoid_f32(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %0 = "onnx.HardSigmoid"(%arg0) {alpha = 0.166666672 : f32, beta = 5.000000e-01 : f32} : (tensor<3xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
// CHECK-LABEL:  func @test_hardsigmoid_f32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3xf32>) -> tensor<3xf32>
// CHECK-DAG:       [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<3.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{value = dense<0.166666672> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           [[VAR_2_:%.+]] = "tosa.add"([[PARAM_0_]], [[VAR_0_]]) : (tensor<3xf32>, tensor<1xf32>) -> tensor<3xf32>
// CHECK:           [[VAR_3_:%.+]] = "tosa.clamp"([[VAR_2_]]) <{max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64}> : (tensor<3xf32>) -> tensor<3xf32>
// CHECK:           [[VAR_4_:%.+]] = "tosa.mul"([[VAR_3_]], [[VAR_1_]]) <{shift = 0 : i32}> : (tensor<3xf32>, tensor<1xf32>) -> tensor<3xf32>
// CHECK:           return [[VAR_4_]] : tensor<3xf32>
}

func.func @test_hardsigmoid_f16(%arg0: tensor<3xf16>) -> tensor<3xf16> {
  %0 = "onnx.HardSigmoid"(%arg0) {alpha = 0.166666672 : f32, beta = 5.000000e-01 : f32} : (tensor<3xf16>) -> tensor<3xf16>
  return %0 : tensor<3xf16>
// CHECK-LABEL:  func @test_hardsigmoid_f16
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3xf16>) -> tensor<3xf16>
// CHECK-DAG:       [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<3.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{value = dense<1.666260e-01> : tensor<1xf16>}> : () -> tensor<1xf16>
// CHECK:           [[VAR_2_:%.+]] = "tosa.add"([[PARAM_0_]], [[VAR_0_]]) : (tensor<3xf16>, tensor<1xf16>) -> tensor<3xf16>
// CHECK:           [[VAR_3_:%.+]] = "tosa.clamp"([[VAR_2_]]) <{max_fp = 6.000000e+00 : f32, max_int = 6 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64}> : (tensor<3xf16>) -> tensor<3xf16>
// CHECK:           [[VAR_4_:%.+]] = "tosa.mul"([[VAR_3_]], [[VAR_1_]]) <{shift = 0 : i32}> : (tensor<3xf16>, tensor<1xf16>) -> tensor<3xf16>
// CHECK:           return [[VAR_4_]] : tensor<3xf16>
}

// -----

func.func @test_elu_f32(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %0 = "onnx.Elu"(%arg0) {alpha = 0.166666672 : f32} : (tensor<3xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
// CHECK-LABEL:  func.func @test_elu_f32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3xf32>) -> tensor<3xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{value = dense<0.166666672> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.exp"([[PARAM_0_]]) : (tensor<3xf32>) -> tensor<3xf32>
// CHECK:           [[VAR_4_:%.+]] = "tosa.sub"([[VAR_3_]], [[VAR_0_]]) : (tensor<3xf32>, tensor<1xf32>) -> tensor<3xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "tosa.mul"([[VAR_4_]], [[VAR_1_]]) <{shift = 0 : i32}> : (tensor<3xf32>, tensor<1xf32>) -> tensor<3xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "tosa.greater_equal"([[PARAM_0_]], [[VAR_2_]]) : (tensor<3xf32>, tensor<1xf32>) -> tensor<3xi1>
// CHECK:           [[VAR_7_:%.+]] = "tosa.select"([[VAR_6_]], [[PARAM_0_]], [[VAR_5_]]) : (tensor<3xi1>, tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
// CHECK:           return [[VAR_7_]]
}

func.func @test_elu_f16(%arg0: tensor<3xf16>) -> tensor<3xf16> {
  %0 = "onnx.Elu"(%arg0) {alpha = 0.166666672 : f32, beta = 5.000000e-01 : f32} : (tensor<3xf16>) -> tensor<3xf16>
  return %0 : tensor<3xf16>
// CHECK-LABEL:  func.func @test_elu_f16
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3xf16>) -> tensor<3xf16> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{value = dense<1.666260e-01> : tensor<1xf16>}> : () -> tensor<1xf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.exp"([[PARAM_0_]]) : (tensor<3xf16>) -> tensor<3xf16>
// CHECK:           [[VAR_4_:%.+]] = "tosa.sub"([[VAR_3_]], [[VAR_0_]]) : (tensor<3xf16>, tensor<1xf16>) -> tensor<3xf16>
// CHECK-DAG:       [[VAR_5_:%.+]] = "tosa.mul"([[VAR_4_]], [[VAR_1_]]) <{shift = 0 : i32}> : (tensor<3xf16>, tensor<1xf16>) -> tensor<3xf16>
// CHECK-DAG:       [[VAR_6_:%.+]] = "tosa.greater_equal"([[PARAM_0_]], [[VAR_2_]]) : (tensor<3xf16>, tensor<1xf16>) -> tensor<3xi1>
// CHECK:           [[VAR_7_:%.+]] = "tosa.select"([[VAR_6_]], [[PARAM_0_]], [[VAR_5_]]) : (tensor<3xi1>, tensor<3xf16>, tensor<3xf16>) -> tensor<3xf16>
// CHECK:           return [[VAR_7_]]
}

// -----

func.func @test_and(%arg0: tensor<13x21x1xi1>, %arg1: tensor<13x21x1xi1>) -> tensor<13x21x1xi1> {
  %0 = "onnx.And"(%arg0, %arg1) : (tensor<13x21x1xi1>, tensor<13x21x1xi1>) -> tensor<13x21x1xi1>
  "func.return"(%0) : (tensor<13x21x1xi1>) -> ()
// CHECK-LABEL:  func @test_and
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xi1>, [[PARAM_1_:%.+]]: tensor<13x21x1xi1>) -> tensor<13x21x1xi1> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.logical_and"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<13x21x1xi1>, tensor<13x21x1xi1>) -> tensor<13x21x1xi1>
}

// -----

func.func @test_and_broadcast(%arg0: tensor<13x21x1xi1>, %arg1: tensor<1xi1>) -> tensor<13x21x1xi1> {
  %0 = "onnx.And"(%arg0, %arg1) : (tensor<13x21x1xi1>, tensor<1xi1>) -> tensor<13x21x1xi1>
  "func.return"(%0) : (tensor<13x21x1xi1>) -> ()
// CHECK-LABEL:  func.func @test_and_broadcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xi1>, [[PARAM_1_:%.+]]: tensor<1xi1>) -> tensor<13x21x1xi1> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.reshape"([[PARAM_1_]]) <{new_shape = array<i64: 1, 1, 1>}> : (tensor<1xi1>) -> tensor<1x1x1xi1>
// CHECK:           [[VAR_1_:%.+]] = "tosa.logical_and"([[PARAM_0_]], [[VAR_0_]]) : (tensor<13x21x1xi1>, tensor<1x1x1xi1>) -> tensor<13x21x1xi1>
// CHECK:           return [[VAR_1_]] : tensor<13x21x1xi1>
}
// -----

func.func @test_bitwise_and(%arg0: tensor<13x21x1xi64>, %arg1: tensor<13x21x1xi64>) -> tensor<13x21x1xi64> {
  %0 = "onnx.BitwiseAnd"(%arg0, %arg1) : (tensor<13x21x1xi64>, tensor<13x21x1xi64>) -> tensor<13x21x1xi64>
  "func.return"(%0) : (tensor<13x21x1xi64>) -> ()
// CHECK-LABEL:  func @test_bitwise_and
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xi64>, [[PARAM_1_:%.+]]: tensor<13x21x1xi64>) -> tensor<13x21x1xi64> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.bitwise_and"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<13x21x1xi64>, tensor<13x21x1xi64>) -> tensor<13x21x1xi64>
}
// -----

func.func @test_bitwise_and_broadcast(%arg0: tensor<13x21x1xi64>, %arg1: tensor<1xi64>) -> tensor<13x21x1xi64> {
  %0 = "onnx.BitwiseAnd"(%arg0, %arg1) : (tensor<13x21x1xi64>, tensor<1xi64>) -> tensor<13x21x1xi64>
  "func.return"(%0) : (tensor<13x21x1xi64>) -> ()
// CHECK-LABEL:  func.func @test_bitwise_and_broadcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xi64>, [[PARAM_1_:%.+]]: tensor<1xi64>) -> tensor<13x21x1xi64> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.reshape"([[PARAM_1_]]) <{new_shape = array<i64: 1, 1, 1>}> : (tensor<1xi64>) -> tensor<1x1x1xi64>
// CHECK:           [[VAR_1_:%.+]] = "tosa.bitwise_and"([[PARAM_0_]], [[VAR_0_]]) : (tensor<13x21x1xi64>, tensor<1x1x1xi64>) -> tensor<13x21x1xi64>
// CHECK:           return [[VAR_1_]] : tensor<13x21x1xi64>
}
// -----

func.func @test_or(%arg0: tensor<13x21x1xi1>, %arg1: tensor<13x21x1xi1>) -> tensor<13x21x1xi1> {
  %0 = "onnx.Or"(%arg0, %arg1) : (tensor<13x21x1xi1>, tensor<13x21x1xi1>) -> tensor<13x21x1xi1>
  "func.return"(%0) : (tensor<13x21x1xi1>) -> ()
// CHECK-LABEL:  func @test_or
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xi1>, [[PARAM_1_:%.+]]: tensor<13x21x1xi1>) -> tensor<13x21x1xi1> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.logical_or"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<13x21x1xi1>, tensor<13x21x1xi1>) -> tensor<13x21x1xi1>
}
// -----

func.func @test_or_broadcast(%arg0: tensor<13x21x1xi1>, %arg1: tensor<1xi1>) -> tensor<13x21x1xi1> {
  %0 = "onnx.Or"(%arg0, %arg1) : (tensor<13x21x1xi1>, tensor<1xi1>) -> tensor<13x21x1xi1>
  "func.return"(%0) : (tensor<13x21x1xi1>) -> ()
// CHECK-LABEL:  func.func @test_or_broadcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xi1>, [[PARAM_1_:%.+]]: tensor<1xi1>) -> tensor<13x21x1xi1> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.reshape"([[PARAM_1_]]) <{new_shape = array<i64: 1, 1, 1>}> : (tensor<1xi1>) -> tensor<1x1x1xi1>
// CHECK:           [[VAR_1_:%.+]] = "tosa.logical_or"([[PARAM_0_]], [[VAR_0_]]) : (tensor<13x21x1xi1>, tensor<1x1x1xi1>) -> tensor<13x21x1xi1>
// CHECK:           return [[VAR_1_]] : tensor<13x21x1xi1>
}
// -----

func.func @test_bitwise_or(%arg0: tensor<13x21x1xi64>, %arg1: tensor<13x21x1xi64>) -> tensor<13x21x1xi64> {
  %0 = "onnx.BitwiseOr"(%arg0, %arg1) : (tensor<13x21x1xi64>, tensor<13x21x1xi64>) -> tensor<13x21x1xi64>
  "func.return"(%0) : (tensor<13x21x1xi64>) -> ()
// CHECK-LABEL:  func @test_bitwise_or
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xi64>, [[PARAM_1_:%.+]]: tensor<13x21x1xi64>) -> tensor<13x21x1xi64> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.bitwise_or"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<13x21x1xi64>, tensor<13x21x1xi64>) -> tensor<13x21x1xi64>
}
// -----

func.func @test_bitwise_or_broadcast(%arg0: tensor<13x21x1xi64>, %arg1: tensor<1xi64>) -> tensor<13x21x1xi64> {
  %0 = "onnx.BitwiseOr"(%arg0, %arg1) : (tensor<13x21x1xi64>, tensor<1xi64>) -> tensor<13x21x1xi64>
  "func.return"(%0) : (tensor<13x21x1xi64>) -> ()
// CHECK-LABEL:  func.func @test_bitwise_or_broadcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xi64>, [[PARAM_1_:%.+]]: tensor<1xi64>) -> tensor<13x21x1xi64> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.reshape"([[PARAM_1_]]) <{new_shape = array<i64: 1, 1, 1>}> : (tensor<1xi64>) -> tensor<1x1x1xi64>
// CHECK:           [[VAR_1_:%.+]] = "tosa.bitwise_or"([[PARAM_0_]], [[VAR_0_]]) : (tensor<13x21x1xi64>, tensor<1x1x1xi64>) -> tensor<13x21x1xi64>
// CHECK:           return [[VAR_1_]] : tensor<13x21x1xi64>
}

// -----

func.func @test_xor(%arg0: tensor<13x21x1xi1>, %arg1: tensor<13x21x1xi1>) -> tensor<13x21x1xi1> {
  %0 = "onnx.Xor"(%arg0, %arg1) : (tensor<13x21x1xi1>, tensor<13x21x1xi1>) -> tensor<13x21x1xi1>
  "func.return"(%0) : (tensor<13x21x1xi1>) -> ()
// CHECK-LABEL:  func @test_xor
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xi1>, [[PARAM_1_:%.+]]: tensor<13x21x1xi1>) -> tensor<13x21x1xi1> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.logical_xor"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<13x21x1xi1>, tensor<13x21x1xi1>) -> tensor<13x21x1xi1>
}

// -----

func.func @test_xor_broadcast(%arg0: tensor<13x21x1xi1>, %arg1: tensor<1xi1>) -> tensor<13x21x1xi1> {
  %0 = "onnx.Xor"(%arg0, %arg1) : (tensor<13x21x1xi1>, tensor<1xi1>) -> tensor<13x21x1xi1>
  "func.return"(%0) : (tensor<13x21x1xi1>) -> ()
// CHECK-LABEL:  func.func @test_xor_broadcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xi1>, [[PARAM_1_:%.+]]: tensor<1xi1>) -> tensor<13x21x1xi1> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.reshape"([[PARAM_1_]]) <{new_shape = array<i64: 1, 1, 1>}> : (tensor<1xi1>) -> tensor<1x1x1xi1>
// CHECK:           [[VAR_1_:%.+]] = "tosa.logical_xor"([[PARAM_0_]], [[VAR_0_]]) : (tensor<13x21x1xi1>, tensor<1x1x1xi1>) -> tensor<13x21x1xi1>
// CHECK:           return [[VAR_1_]] : tensor<13x21x1xi1>
}
// -----

func.func @test_bitwise_xor(%arg0: tensor<13x21x1xi64>, %arg1: tensor<13x21x1xi64>) -> tensor<13x21x1xi64> {
  %0 = "onnx.BitwiseXor"(%arg0, %arg1) : (tensor<13x21x1xi64>, tensor<13x21x1xi64>) -> tensor<13x21x1xi64>
  "func.return"(%0) : (tensor<13x21x1xi64>) -> ()
// CHECK-LABEL:  func @test_bitwise_xor
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xi64>, [[PARAM_1_:%.+]]: tensor<13x21x1xi64>) -> tensor<13x21x1xi64> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.bitwise_xor"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<13x21x1xi64>, tensor<13x21x1xi64>) -> tensor<13x21x1xi64>
}
// -----

func.func @test_bitwise_xor_broadcast(%arg0: tensor<13x21x1xi64>, %arg1: tensor<1xi64>) -> tensor<13x21x1xi64> {
  %0 = "onnx.BitwiseXor"(%arg0, %arg1) : (tensor<13x21x1xi64>, tensor<1xi64>) -> tensor<13x21x1xi64>
  "func.return"(%0) : (tensor<13x21x1xi64>) -> ()
// CHECK-LABEL:  func.func @test_bitwise_xor_broadcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xi64>, [[PARAM_1_:%.+]]: tensor<1xi64>) -> tensor<13x21x1xi64> {
// CHECK:           [[VAR_0_:%.+]] = "tosa.reshape"([[PARAM_1_]]) <{new_shape = array<i64: 1, 1, 1>}> : (tensor<1xi64>) -> tensor<1x1x1xi64>
// CHECK:           [[VAR_1_:%.+]] = "tosa.bitwise_xor"([[PARAM_0_]], [[VAR_0_]]) : (tensor<13x21x1xi64>, tensor<1x1x1xi64>) -> tensor<13x21x1xi64>
// CHECK:           return [[VAR_1_]] : tensor<13x21x1xi64>
}

// -----

func.func @test_min(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x1xf32> {
  %0 = "onnx.Min"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
  "func.return"(%0) : (tensor<13x21x1xf32>) -> ()
// CHECK-LABEL:  func @test_min
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xf32>, [[PARAM_1_:%.+]]: tensor<13x21x1xf32>) -> tensor<13x21x1xf32> {
// CHECK-NEXT:      [[VAR_1_:%.+]] = "tosa.minimum"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<13x21x1xf32>, tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
}

// -----

func.func @test_min_broadcast(%arg0: tensor<13x21x1xf32>, %arg1: tensor<1xf32>) -> tensor<13x21x1xf32> {
  %0 = "onnx.Min"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<1xf32>) -> tensor<13x21x1xf32>
  "func.return"(%0) : (tensor<13x21x1xf32>) -> ()
// CHECK-LABEL:  func @test_min_broadcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xf32>, [[PARAM_1_:%.+]]: tensor<1xf32>) -> tensor<13x21x1xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.reshape"([[PARAM_1_]]) <{new_shape = array<i64: 1, 1, 1>}> : (tensor<1xf32>) -> tensor<1x1x1xf32>
// CHECK-NEXT:      [[VAR_1_:%.+]] = "tosa.minimum"([[PARAM_0_]], [[VAR_0_]]) : (tensor<13x21x1xf32>, tensor<1x1x1xf32>) -> tensor<13x21x1xf32>
}
// -----

func.func @test_max(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x1xf32> {
  %0 = "onnx.Max"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
  "func.return"(%0) : (tensor<13x21x1xf32>) -> ()
// CHECK-LABEL:  func @test_max
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xf32>, [[PARAM_1_:%.+]]: tensor<13x21x1xf32>) -> tensor<13x21x1xf32> {
// CHECK-NEXT:      [[VAR_1_:%.+]] = "tosa.maximum"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<13x21x1xf32>, tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
}

// -----

func.func @test_max_broadcast(%arg0: tensor<13x21x1xf32>, %arg1: tensor<1xf32>) -> tensor<13x21x1xf32> {
  %0 = "onnx.Max"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<1xf32>) -> tensor<13x21x1xf32>
  "func.return"(%0) : (tensor<13x21x1xf32>) -> ()
// CHECK-LABEL:  func @test_max_broadcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xf32>, [[PARAM_1_:%.+]]: tensor<1xf32>) -> tensor<13x21x1xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.reshape"([[PARAM_1_]]) <{new_shape = array<i64: 1, 1, 1>}> : (tensor<1xf32>) -> tensor<1x1x1xf32>
// CHECK-NEXT:      [[VAR_1_:%.+]] = "tosa.maximum"([[PARAM_0_]], [[VAR_0_]]) : (tensor<13x21x1xf32>, tensor<1x1x1xf32>) -> tensor<13x21x1xf32>
}
