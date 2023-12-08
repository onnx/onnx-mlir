// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s

func.func @test_relu(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Relu"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_relu
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = tosa.clamp [[PARAM_0_]] {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<10x10xf32>) -> tensor<10x10xf32>
// CHECK-NEXT:      return [[VAR_0_]] : tensor<10x10xf32>
// CHECK-NEXT:    }
}

// -----

func.func @test_relu_dynamic(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Relu"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  func @test_relu_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x10xf32>) -> tensor<?x10xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] =  tosa.clamp [[PARAM_0_]] {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<?x10xf32>) -> tensor<?x10xf32>
// CHECK-NEXT:      return [[VAR_0_]] : tensor<?x10xf32>
// CHECK-NEXT:    }
}

// -----

func.func @test_neg(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Neg"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_neg
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = tosa.negate [[PARAM_0_]] : (tensor<10x10xf32>) -> tensor<10x10xf32>
}

// -----

func.func @test_floor(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Floor"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_floor
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = tosa.floor [[PARAM_0_]] : (tensor<10x10xf32>) -> tensor<10x10xf32>
}

// -----

func.func @test_add(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x1xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
  "func.return"(%0) : (tensor<13x21x1xf32>) -> ()
// CHECK-LABEL:  func @test_add
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xf32>, [[PARAM_1_:%.+]]: tensor<13x21x1xf32>) -> tensor<13x21x1xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = tosa.add [[PARAM_0_]], [[PARAM_1_]] : (tensor<13x21x1xf32>, tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
}

// -----

func.func @test_add_broadcast(%arg0: tensor<13x21x1xf32>, %arg1: tensor<1xf32>) -> tensor<13x21x1xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<1xf32>) -> tensor<13x21x1xf32>
  "func.return"(%0) : (tensor<13x21x1xf32>) -> ()
// CHECK-LABEL:  func.func @test_add_broadcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xf32>, [[PARAM_1_:%.+]]: tensor<1xf32>) -> tensor<13x21x1xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.reshape [[PARAM_1_]] {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
// CHECK:           [[VAR_1_:%.+]] = tosa.add [[PARAM_0_]], [[VAR_0_]] : (tensor<13x21x1xf32>, tensor<1x1x1xf32>) -> tensor<13x21x1xf32>
// CHECK:           return [[VAR_1_]] : tensor<13x21x1xf32>
}


// -----

func.func @test_sub(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x1xf32> {
  %0 = "onnx.Sub"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
  "func.return"(%0) : (tensor<13x21x1xf32>) -> ()
// CHECK-LABEL:  func @test_sub
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xf32>, [[PARAM_1_:%.+]]: tensor<13x21x1xf32>) -> tensor<13x21x1xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = tosa.sub [[PARAM_0_]], [[PARAM_1_]] : (tensor<13x21x1xf32>, tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
}

// -----

func.func @test_sub_broadcast(%arg0: tensor<13x21x1xf32>, %arg1: tensor<1xf32>) -> tensor<13x21x1xf32> {
  %0 = "onnx.Sub"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<1xf32>) -> tensor<13x21x1xf32>
  "func.return"(%0) : (tensor<13x21x1xf32>) -> ()
// CHECK-LABEL:  func.func @test_sub_broadcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xf32>, [[PARAM_1_:%.+]]: tensor<1xf32>) -> tensor<13x21x1xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.reshape [[PARAM_1_]] {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
// CHECK:           [[VAR_1_:%.+]] = tosa.sub [[PARAM_0_]], [[VAR_0_]] : (tensor<13x21x1xf32>, tensor<1x1x1xf32>) -> tensor<13x21x1xf32>
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

func.func @test_div(%arg0: tensor<13x21x1xi32>, %arg1: tensor<13x21x1xi32>) -> tensor<13x21x1xi32> {
  %0 = "onnx.Div"(%arg0, %arg1) : (tensor<13x21x1xi32>, tensor<13x21x1xi32>) -> tensor<13x21x1xi32>
  "func.return"(%0) : (tensor<13x21x1xi32>) -> ()
// CHECK-LABEL:  func @test_div
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xi32>, [[PARAM_1_:%.+]]: tensor<13x21x1xi32>) -> tensor<13x21x1xi32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = tosa.div [[PARAM_0_]], [[PARAM_1_]] : (tensor<13x21x1xi32>, tensor<13x21x1xi32>) -> tensor<13x21x1xi32>
}

// -----

func.func @test_div_broadcast(%arg0: tensor<13x21x1xi32>, %arg1: tensor<1xi32>) -> tensor<13x21x1xi32> {
  %0 = "onnx.Div"(%arg0, %arg1) : (tensor<13x21x1xi32>, tensor<1xi32>) -> tensor<13x21x1xi32>
  "func.return"(%0) : (tensor<13x21x1xi32>) -> ()
// CHECK-LABEL:  func @test_div_broadcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xi32>, [[PARAM_1_:%.+]]: tensor<1xi32>) -> tensor<13x21x1xi32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = tosa.reshape [[PARAM_1_]] {new_shape = array<i64: 1, 1, 1>} : (tensor<1xi32>) -> tensor<1x1x1xi32>
// CHECK-NEXT:      [[VAR_1_:%.+]] = tosa.div [[PARAM_0_]], [[VAR_0_]] : (tensor<13x21x1xi32>, tensor<1x1x1xi32>) -> tensor<13x21x1xi32>
}

// -----

func.func @test_div_decomposed(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x1xf32> {
  %0 = "onnx.Div"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
  "func.return"(%0) : (tensor<13x21x1xf32>) -> ()
// CHECK-LABEL:  func @test_div_decomposed
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xf32>, [[PARAM_1_:%.+]]: tensor<13x21x1xf32>) -> tensor<13x21x1xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = tosa.reciprocal [[PARAM_1_]] : (tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
// CHECK-NEXT:      [[VAR_1_:%.+]] = tosa.mul [[PARAM_0_]], [[VAR_0_]] {shift = 0 : i32} : (tensor<13x21x1xf32>, tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
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

// -----

func.func @test_sigmoid(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Sigmoid"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_sigmoid
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "tosa.sigmoid"([[PARAM_0_]]) : (tensor<10x10xf32>) -> tensor<10x10xf32>
// CHECK-NEXT:      return [[VAR_0_]] : tensor<10x10xf32>
// CHECK-NEXT:    }

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
// CHECK-NEXT:      [[VAR_0_:%.+]] = tosa.reciprocal [[PARAM_1_]] : (tensor<1xf32>) -> tensor<1xf32>
// CHECK-NEXT:      [[VAR_1_:%.+]] = tosa.reshape [[VAR_0_]] {new_shape = array<i64: 1, 1, 1>} : (tensor<1xf32>) -> tensor<1x1x1xf32>
// CHECK-NEXT:      [[VAR_2_:%.+]] = tosa.mul [[PARAM_0_]], [[VAR_1_]] {shift = 0 : i32} : (tensor<13x21x1xf32>, tensor<1x1x1xf32>) -> tensor<13x21x1xf32>
}
