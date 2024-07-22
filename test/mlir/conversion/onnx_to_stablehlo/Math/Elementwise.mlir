// RUN: onnx-mlir-opt --convert-onnx-to-stablehlo --canonicalize %s -split-input-file | FileCheck %s

func.func @test_add(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_add
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>, [[PARAM_1_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = shape.const_shape [10, 10] : tensor<2xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_0_]], [[VAR_0_]], dims = [0, 1] : (tensor<10x10xf32>, tensor<2xindex>) -> tensor<10x10xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<10x10xf32>, tensor<2xindex>) -> tensor<10x10xf32>
// CHECK:           [[VAR_3_:%.+]] = stablehlo.add [[VAR_1_]], [[VAR_2_]] : tensor<10x10xf32>
// CHECK:           return [[VAR_3_]] : tensor<10x10xf32>
// CHECK:         }

// -----

func.func @test_add_dynamic(%arg0 : tensor<?x10xf32>, %arg1 : tensor<?x10xf32>) -> tensor<?x10xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x10xf32>, tensor<?x10xf32>) -> tensor<?x10xf32>
  "func.return"(%0) : (tensor<?x10xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_add_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x10xf32>, [[PARAM_1_:%.+]]: tensor<?x10xf32>) -> tensor<?x10xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = shape.shape_of [[PARAM_0_]] : tensor<?x10xf32> -> tensor<2xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = shape.shape_of [[PARAM_1_]] : tensor<?x10xf32> -> tensor<2xindex>
// CHECK:           [[VAR_2_:%.+]] = shape.broadcast [[VAR_0_]], [[VAR_1_]] : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
// CHECK-DAG:       [[VAR_3_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_0_]], [[VAR_2_]], dims = [0, 1] : (tensor<?x10xf32>, tensor<2xindex>) -> tensor<?x10xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_1_]], [[VAR_2_]], dims = [0, 1] : (tensor<?x10xf32>, tensor<2xindex>) -> tensor<?x10xf32>
// CHECK:           [[VAR_5_:%.+]] = stablehlo.add [[VAR_3_]], [[VAR_4_]] : tensor<?x10xf32>
// CHECK:           return [[VAR_5_]] : tensor<?x10xf32>
// CHECK:         }

// -----

func.func @test_relu(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Relu"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_relu
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<10x10xf32>
// CHECK:           [[VAR_1_:%.+]] = stablehlo.maximum [[PARAM_0_]], [[VAR_0_]] : tensor<10x10xf32>
// CHECK:           return [[VAR_1_]] : tensor<10x10xf32>
// CHECK:         }

// -----

func.func @test_elu(%arg0 : tensor<20x40xf32>) -> tensor<20x40xf32> {
  %0 = "onnx.Elu"(%arg0) {alpha = 1.500000e+00 : f32} : (tensor<20x40xf32>) -> tensor<20x40xf32>
  "func.return"(%0) : (tensor<20x40xf32>) -> ()
// CHECK-LABEL:  func.func @test_elu
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<20x40xf32>) -> tensor<20x40xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<20x40xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.constant dense<1.500000e+00> : tensor<20x40xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = stablehlo.constant dense<1.000000e+00> : tensor<20x40xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = stablehlo.exponential [[PARAM_0_]] : tensor<20x40xf32>
// CHECK:           [[VAR_4_:%.+]] = stablehlo.subtract [[VAR_3_]], [[VAR_2_]] : tensor<20x40xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = stablehlo.multiply [[VAR_1_]], [[VAR_4_]] : tensor<20x40xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = stablehlo.compare  GE, [[PARAM_0_]], [[VAR_0_]] : (tensor<20x40xf32>, tensor<20x40xf32>) -> tensor<20x40xi1>
// CHECK:           [[VAR_7_:%.+]] = stablehlo.select [[VAR_6_]], [[PARAM_0_]], [[VAR_5_]] : tensor<20x40xi1>, tensor<20x40xf32>
// CHECK:           return [[VAR_7_]] : tensor<20x40xf32>
// CHECK:         }
}

// -----

func.func @test_relu_dynamic(%arg0 : tensor<?x10xf32>) -> tensor<?x10xf32> {
  %0 = "onnx.Relu"(%arg0) : (tensor<?x10xf32>) -> tensor<?x10xf32>
  "func.return"(%0) : (tensor<?x10xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_relu_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x10xf32>) -> tensor<?x10xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = shape.shape_of [[PARAM_0_]] : tensor<?x10xf32> -> tensor<2xindex>
// CHECK:           [[VAR_2_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_0_]], [[VAR_1_]], dims = [] : (tensor<f32>, tensor<2xindex>) -> tensor<?x10xf32>
// CHECK:           [[VAR_3_:%.+]] = stablehlo.maximum [[PARAM_0_]], [[VAR_2_]] : tensor<?x10xf32>
// CHECK:           return [[VAR_3_]] : tensor<?x10xf32>
// CHECK:         }

// -----

func.func @test_exp(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Exp"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_exp
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.exponential [[PARAM_0_]] : tensor<10x10xf32>
// CHECK:           return [[VAR_0_]] : tensor<10x10xf32>
// CHECK:         }

// -----

func.func @test_dynamic_exp(%arg0 : tensor<?x10xf32>) -> tensor<?x10xf32> {
  %0 = "onnx.Exp"(%arg0) : (tensor<?x10xf32>) -> tensor<?x10xf32>
  "func.return"(%0) : (tensor<?x10xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_dynamic_exp
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x10xf32>) -> tensor<?x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.exponential [[PARAM_0_]] : tensor<?x10xf32>
// CHECK:           return [[VAR_0_]] : tensor<?x10xf32>
// CHECK:         }

// -----

func.func @test_sigmoid(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Sigmoid"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_sigmoid
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.logistic [[PARAM_0_]] : tensor<10x10xf32>
// CHECK:           return [[VAR_0_]] : tensor<10x10xf32>
// CHECK:         }

// -----

func.func @test_dynamic_sigmoid(%arg0 : tensor<?x10xf32>) -> tensor<?x10xf32> {
  %0 = "onnx.Sigmoid"(%arg0) : (tensor<?x10xf32>) -> tensor<?x10xf32>
  "func.return"(%0) : (tensor<?x10xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_dynamic_sigmoid
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x10xf32>) -> tensor<?x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.logistic [[PARAM_0_]] : tensor<?x10xf32>
// CHECK:           return [[VAR_0_]] : tensor<?x10xf32>
// CHECK:         }

// -----

func.func @test_hard_sigmoid(%arg0 : tensor<20x40xf32>) -> tensor<20x40xf32> {
  %0 = "onnx.HardSigmoid"(%arg0) {alpha = 5.000000e-01 : f32, beta = 5.000000e-01 : f32} : (tensor<20x40xf32>) -> tensor<20x40xf32>
  "func.return"(%0) : (tensor<20x40xf32>) -> ()
// CHECK-LABEL:  func.func @test_hard_sigmoid
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<20x40xf32>) -> tensor<20x40xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = stablehlo.constant dense<5.000000e-01> : tensor<20x40xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<20x40xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = stablehlo.constant dense<1.000000e+00> : tensor<20x40xf32>
// CHECK:           [[VAR_3_:%.+]] = stablehlo.multiply [[PARAM_0_]], [[VAR_0_]] : tensor<20x40xf32>
// CHECK:           [[VAR_4_:%.+]] = stablehlo.add [[VAR_3_]], [[VAR_0_]] : tensor<20x40xf32>
// CHECK:           [[VAR_5_:%.+]] = stablehlo.clamp [[VAR_1_]], [[VAR_4_]], [[VAR_2_]] : tensor<20x40xf32>
// CHECK:           return [[VAR_5_]] : tensor<20x40xf32>
// CHECK:         }
}

// -----

func.func @test_abs(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Abs"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_abs
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.abs [[PARAM_0_]] : tensor<10x10xf32>
// CHECK:           return [[VAR_0_]] : tensor<10x10xf32>
// CHECK:         }

// -----

func.func @test_dyn_abs(%arg0 : tensor<?x10xf32>) -> tensor<?x10xf32> {
  %0 = "onnx.Abs"(%arg0) : (tensor<?x10xf32>) -> tensor<?x10xf32>
  "func.return"(%0) : (tensor<?x10xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_dyn_abs
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x10xf32>) -> tensor<?x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.abs [[PARAM_0_]] : tensor<?x10xf32>
// CHECK:           return [[VAR_0_]] : tensor<?x10xf32>
// CHECK:         }

// -----

func.func @test_and(%arg0 : tensor<10x10xi1>, %arg1 : tensor<10x10xi1>) -> tensor<10x10xi1> {
  %0 = "onnx.And"(%arg0, %arg1) : (tensor<10x10xi1>, tensor<10x10xi1>) -> tensor<10x10xi1>
  "func.return"(%0) : (tensor<10x10xi1>) -> ()
}

// CHECK-LABEL:  func.func @test_and
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xi1>, [[PARAM_1_:%.+]]: tensor<10x10xi1>) -> tensor<10x10xi1> {
// CHECK:           [[VAR_0_:%.+]] = shape.const_shape [10, 10] : tensor<2xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_0_]], [[VAR_0_]], dims = [0, 1] : (tensor<10x10xi1>, tensor<2xindex>) -> tensor<10x10xi1>
// CHECK-DAG:       [[VAR_2_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<10x10xi1>, tensor<2xindex>) -> tensor<10x10xi1>
// CHECK:           [[VAR_3_:%.+]] = stablehlo.and [[VAR_1_]], [[VAR_2_]] : tensor<10x10xi1>
// CHECK:           return [[VAR_3_]] : tensor<10x10xi1>
// CHECK:         }

// -----

func.func @test_dyn_and(%arg0 : tensor<?x10xi1>, %arg1 : tensor<?x10xi1>) -> tensor<?x10xi1> {
  %0 = "onnx.And"(%arg0, %arg1) : (tensor<?x10xi1>, tensor<?x10xi1>) -> tensor<?x10xi1>
  "func.return"(%0) : (tensor<?x10xi1>) -> ()
}

// CHECK-LABEL:  func.func @test_dyn_and
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x10xi1>, [[PARAM_1_:%.+]]: tensor<?x10xi1>) -> tensor<?x10xi1> {
// CHECK-DAG:       [[VAR_0_:%.+]] = shape.shape_of [[PARAM_0_]] : tensor<?x10xi1> -> tensor<2xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = shape.shape_of [[PARAM_1_]] : tensor<?x10xi1> -> tensor<2xindex>
// CHECK:           [[VAR_2_:%.+]] = shape.broadcast [[VAR_0_]], [[VAR_1_]] : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
// CHECK-DAG:       [[VAR_3_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_0_]], [[VAR_2_]], dims = [0, 1] : (tensor<?x10xi1>, tensor<2xindex>) -> tensor<?x10xi1>
// CHECK-DAG:       [[VAR_4_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_1_]], [[VAR_2_]], dims = [0, 1] : (tensor<?x10xi1>, tensor<2xindex>) -> tensor<?x10xi1>
// CHECK:           [[VAR_5_:%.+]] = stablehlo.and [[VAR_3_]], [[VAR_4_]] : tensor<?x10xi1>
// CHECK:           return [[VAR_5_]] : tensor<?x10xi1>
// CHECK:         }

// -----

func.func @cast_float(%arg0: tensor<2xf64>) -> tensor<2xf32> {
  %0 = "onnx.Cast"(%arg0) {to = f32} : (tensor<2xf64>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL:  func.func @cast_float
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2xf64>) -> tensor<2xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.convert [[PARAM_0_]] : (tensor<2xf64>) -> tensor<2xf32>
// CHECK:           return [[VAR_0_]] : tensor<2xf32>
// CHECK:         }

// -----

func.func @cast_int(%arg0: tensor<2xf32>) -> tensor<2xi32> {
  %0 = "onnx.Cast"(%arg0) {to = i32} : (tensor<2xf32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

// CHECK-LABEL:  func.func @cast_int
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2xf32>) -> tensor<2xi32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.convert [[PARAM_0_]] : (tensor<2xf32>) -> tensor<2xi32>
// CHECK:           return [[VAR_0_]] : tensor<2xi32>
// CHECK:         }

// -----

func.func @cast_dyn(%arg0: tensor<?x2xf64>) -> tensor<?x2xf32> {
  %0 = "onnx.Cast"(%arg0) {to = f32} : (tensor<?x2xf64>) -> tensor<?x2xf32>
  return %0 : tensor<?x2xf32>
}

// CHECK-LABEL:  func.func @cast_dyn
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x2xf64>) -> tensor<?x2xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.convert [[PARAM_0_]] : (tensor<?x2xf64>) -> tensor<?x2xf32>
// CHECK:           return [[VAR_0_]] : tensor<?x2xf32>
// CHECK:         }

// -----

func.func @test_ceil(%arg0 : tensor<?x10xf32>) -> tensor<?x10xf32> {
  %0 = "onnx.Ceil"(%arg0) : (tensor<?x10xf32>) -> tensor<?x10xf32>
  "func.return"(%0) : (tensor<?x10xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_ceil
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x10xf32>) -> tensor<?x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.ceil [[PARAM_0_]] : tensor<?x10xf32>
// CHECK:           return [[VAR_0_]] : tensor<?x10xf32>
// CHECK:         }

// -----

func.func @test_cos(%arg0 : tensor<?x10xf32>) -> tensor<?x10xf32> {
  %0 = "onnx.Cos"(%arg0) : (tensor<?x10xf32>) -> tensor<?x10xf32>
  "func.return"(%0) : (tensor<?x10xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_cos
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x10xf32>) -> tensor<?x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.cosine [[PARAM_0_]] : tensor<?x10xf32>
// CHECK:           return [[VAR_0_]] : tensor<?x10xf32>
// CHECK:         }

// -----

func.func @test_less(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<3x4x5xi1> {
  %0 = "onnx.Less"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xi1>
  return %0 : tensor<3x4x5xi1>
}

// CHECK-LABEL:  func.func @test_less
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x4x5xf32>, [[PARAM_1_:%.+]]: tensor<3x4x5xf32>) -> tensor<3x4x5xi1> {
// CHECK:           [[VAR_0_:%.+]] = shape.const_shape [3, 4, 5] : tensor<3xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_0_]], [[VAR_0_]], dims = [0, 1, 2] : (tensor<3x4x5xf32>, tensor<3xindex>) -> tensor<3x4x5xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_1_]], [[VAR_0_]], dims = [0, 1, 2] : (tensor<3x4x5xf32>, tensor<3xindex>) -> tensor<3x4x5xf32>
// CHECK:           [[VAR_3_:%.+]] = stablehlo.compare  LT, [[VAR_1_]], [[VAR_2_]] : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xi1>
// CHECK:           return [[VAR_3_]] : tensor<3x4x5xi1>
// CHECK:         }

// -----

func.func @test_binary_elementwise_op_template_unknown_dims(%arg0: tensor<?x4x5xf32>, %arg1: tensor<1x?x1xf32>) -> tensor<?x4x5xi1> {
  %0 = "onnx.Less"(%arg0, %arg1) : (tensor<?x4x5xf32>, tensor<1x?x1xf32>) -> tensor<?x4x5xi1>
  return %0 : tensor<?x4x5xi1>
}

// CHECK-LABEL:  func.func @test_binary_elementwise_op_template_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x4x5xf32>, [[PARAM_1_:%.+]]: tensor<1x?x1xf32>) -> tensor<?x4x5xi1> {
// CHECK-DAG:       [[VAR_0_:%.+]] = shape.shape_of [[PARAM_0_]] : tensor<?x4x5xf32> -> tensor<3xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = shape.shape_of [[PARAM_1_]] : tensor<1x?x1xf32> -> tensor<3xindex>
// CHECK:           [[VAR_2_:%.+]] = shape.broadcast [[VAR_0_]], [[VAR_1_]] : tensor<3xindex>, tensor<3xindex> -> tensor<3xindex>
// CHECK-DAG:       [[VAR_3_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_0_]], [[VAR_2_]], dims = [0, 1, 2] : (tensor<?x4x5xf32>, tensor<3xindex>) -> tensor<?x4x5xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_1_]], [[VAR_2_]], dims = [0, 1, 2] : (tensor<1x?x1xf32>, tensor<3xindex>) -> tensor<?x4x5xf32>
// CHECK:           [[VAR_5_:%.+]] = stablehlo.compare  LT, [[VAR_3_]], [[VAR_4_]] : (tensor<?x4x5xf32>, tensor<?x4x5xf32>) -> tensor<?x4x5xi1>
// CHECK:           return [[VAR_5_]] : tensor<?x4x5xi1>
// CHECK:         }

// -----

func.func @test_less_unknown_dims_2(%arg0: tensor<?x?x5xf32>, %arg1: tensor<?x4x5xf32>) -> tensor<?x4x5xi1> {
  %0 = "onnx.Less"(%arg0, %arg1) : (tensor<?x?x5xf32>, tensor<?x4x5xf32>) -> tensor<?x4x5xi1>
  return %0 : tensor<?x4x5xi1>
}

// CHECK-LABEL:  func.func @test_less_unknown_dims_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x5xf32>, [[PARAM_1_:%.+]]: tensor<?x4x5xf32>) -> tensor<?x4x5xi1> {
// CHECK-DAG:       [[VAR_0_:%.+]] = shape.shape_of [[PARAM_0_]] : tensor<?x?x5xf32> -> tensor<3xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = shape.shape_of [[PARAM_1_]] : tensor<?x4x5xf32> -> tensor<3xindex>
// CHECK:           [[VAR_2_:%.+]] = shape.broadcast [[VAR_0_]], [[VAR_1_]] : tensor<3xindex>, tensor<3xindex> -> tensor<3xindex>
// CHECK-DAG:       [[VAR_3_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_0_]], [[VAR_2_]], dims = [0, 1, 2] : (tensor<?x?x5xf32>, tensor<3xindex>) -> tensor<?x4x5xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_1_]], [[VAR_2_]], dims = [0, 1, 2] : (tensor<?x4x5xf32>, tensor<3xindex>) -> tensor<?x4x5xf32>
// CHECK:           [[VAR_5_:%.+]] = stablehlo.compare  LT, [[VAR_3_]], [[VAR_4_]] : (tensor<?x4x5xf32>, tensor<?x4x5xf32>) -> tensor<?x4x5xi1>
// CHECK:           return [[VAR_5_]] : tensor<?x4x5xi1>
// CHECK:         }

// -----

func.func @test_pow_verifier_1(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<f32>) -> tensor<1x2x3x4xf32> {
  %0 = "onnx.Pow"(%arg0, %arg1) : (tensor<1x2x3x4xf32>, tensor<f32>) -> tensor<1x2x3x4xf32>
  "func.return"(%0) : (tensor<1x2x3x4xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_pow_verifier_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x2x3x4xf32>, [[PARAM_1_:%.+]]: tensor<f32>) -> tensor<1x2x3x4xf32> {
// CHECK:           [[VAR_0_:%.+]] = shape.const_shape [1, 2, 3, 4] : tensor<4xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_0_]], [[VAR_0_]], dims = [0, 1, 2, 3] : (tensor<1x2x3x4xf32>, tensor<4xindex>) -> tensor<1x2x3x4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_1_]], [[VAR_0_]], dims = [] : (tensor<f32>, tensor<4xindex>) -> tensor<1x2x3x4xf32>
// CHECK:           [[VAR_3_:%.+]] = stablehlo.power [[VAR_1_]], [[VAR_2_]] : tensor<1x2x3x4xf32>
// CHECK:           return [[VAR_3_]] : tensor<1x2x3x4xf32>
// CHECK:         }

// -----

func.func @test_mul_unknown_dims(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x?xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Mul"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x?xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_mul_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>, [[PARAM_1_:%.+]]: tensor<10x?xf32>) -> tensor<10x10xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = shape.const_shape [10, 10] : tensor<2xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = shape.shape_of [[PARAM_1_]] : tensor<10x?xf32> -> tensor<2xindex>
// CHECK:           [[VAR_2_:%.+]] = shape.broadcast [[VAR_1_]], [[VAR_0_]] : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
// CHECK-DAG:       [[VAR_3_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_0_]], [[VAR_2_]], dims = [0, 1] : (tensor<10x10xf32>, tensor<2xindex>) -> tensor<10x10xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_1_]], [[VAR_2_]], dims = [0, 1] : (tensor<10x?xf32>, tensor<2xindex>) -> tensor<10x10xf32>
// CHECK:           [[VAR_5_:%.+]] = stablehlo.multiply [[VAR_3_]], [[VAR_4_]] : tensor<10x10xf32>
// CHECK:           return [[VAR_5_]] : tensor<10x10xf32>
// CHECK:         }

// -----

func.func @test_sqrt(%arg0 : tensor<?x10xf32>) -> tensor<?x10xf32> {
  %0 = "onnx.Sqrt"(%arg0) : (tensor<?x10xf32>) -> tensor<?x10xf32>
  "func.return"(%0) : (tensor<?x10xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_sqrt
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x10xf32>) -> tensor<?x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.sqrt [[PARAM_0_]] : tensor<?x10xf32>
// CHECK:           return [[VAR_0_]] : tensor<?x10xf32>
// CHECK:         }

// -----

func.func @test_log(%arg0 : tensor<?x10xf32>) -> tensor<?x10xf32> {
  %0 = "onnx.Log"(%arg0) : (tensor<?x10xf32>) -> tensor<?x10xf32>
  "func.return"(%0) : (tensor<?x10xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_log
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x10xf32>) -> tensor<?x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.log [[PARAM_0_]] : tensor<?x10xf32>
// CHECK:           return [[VAR_0_]] : tensor<?x10xf32>
// CHECK:         }

// -----

func.func @test_tanh(%arg0 : tensor<?x10xf32>) -> tensor<?x10xf32> {
  %0 = "onnx.Tanh"(%arg0) : (tensor<?x10xf32>) -> tensor<?x10xf32>
  "func.return"(%0) : (tensor<?x10xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_tanh
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x10xf32>) -> tensor<?x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.tanh [[PARAM_0_]] : tensor<?x10xf32>
// CHECK:           return [[VAR_0_]] : tensor<?x10xf32>
// CHECK:         }

// -----

func.func @test_max(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Max"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_max
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>, [[PARAM_1_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = shape.const_shape [10, 10] : tensor<2xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_0_]], [[VAR_0_]], dims = [0, 1] : (tensor<10x10xf32>, tensor<2xindex>) -> tensor<10x10xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<10x10xf32>, tensor<2xindex>) -> tensor<10x10xf32>
// CHECK:           [[VAR_3_:%.+]] = stablehlo.maximum [[VAR_1_]], [[VAR_2_]] : tensor<10x10xf32>
// CHECK:           return [[VAR_3_]] : tensor<10x10xf32>
// CHECK:         }

// -----

func.func @test_min(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Min"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func.func @test_min
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>, [[PARAM_1_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = shape.const_shape [10, 10] : tensor<2xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_0_]], [[VAR_0_]], dims = [0, 1] : (tensor<10x10xf32>, tensor<2xindex>) -> tensor<10x10xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_1_]], [[VAR_0_]], dims = [0, 1] : (tensor<10x10xf32>, tensor<2xindex>) -> tensor<10x10xf32>
// CHECK:           [[VAR_3_:%.+]] = stablehlo.minimum [[VAR_1_]], [[VAR_2_]] : tensor<10x10xf32>
// CHECK:           return [[VAR_3_]] : tensor<10x10xf32>
// CHECK:         }
}

// -----

func.func @test_leakyrelu_dynamic(%arg0 : tensor<?x10xf32>) -> tensor<?x10xf32> {
  %0 = "onnx.LeakyRelu"(%arg0) {alpha=0.5:f32} : (tensor<?x10xf32>) -> tensor<?x10xf32>
  "func.return"(%0) : (tensor<?x10xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_leakyrelu_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x10xf32>) -> tensor<?x10xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.constant dense<5.000000e-01> : tensor<f32>
// CHECK-DAG:       [[VAR_2_:%.+]] = shape.shape_of [[PARAM_0_]] : tensor<?x10xf32> -> tensor<2xindex>
// CHECK:           [[VAR_3_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_1_]], [[VAR_2_]], dims = [] : (tensor<f32>, tensor<2xindex>) -> tensor<?x10xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = stablehlo.multiply [[PARAM_0_]], [[VAR_3_]] : tensor<?x10xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = shape.shape_of [[PARAM_0_]] : tensor<?x10xf32> -> tensor<2xindex>
// CHECK:           [[VAR_6_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_0_]], [[VAR_5_]], dims = [] : (tensor<f32>, tensor<2xindex>) -> tensor<?x10xf32>
// CHECK:           [[VAR_7_:%.+]] = stablehlo.compare  GT, [[PARAM_0_]], [[VAR_6_]] : (tensor<?x10xf32>, tensor<?x10xf32>) -> tensor<?x10xi1>
// CHECK:           [[VAR_8_:%.+]] = stablehlo.select [[VAR_7_]], [[PARAM_0_]], [[VAR_4_]] : tensor<?x10xi1>, tensor<?x10xf32>
// CHECK:           return [[VAR_8_]] : tensor<?x10xf32>
// CHECK:         }

// -----

func.func @test_prelu_dynamic(%arg0 : tensor<?x10x12x12xf32>, %arg1: tensor<10x1x1xf32>) -> tensor<?x10x12x12xf32> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<?x10x12x12xf32>, tensor<10x1x1xf32>) -> tensor<?x10x12x12xf32>
  "func.return"(%0) : (tensor<?x10x12x12xf32>) -> ()
// CHECK-LABEL:  func.func @test_prelu_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x10x12x12xf32>, [[PARAM_1_:%.+]]: tensor<10x1x1xf32>) -> tensor<?x10x12x12xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = shape.const_shape [10, 1, 1] : tensor<3xindex>
// CHECK-DAG:       [[VAR_2_:%.+]] = shape.shape_of [[PARAM_0_]] : tensor<?x10x12x12xf32> -> tensor<4xindex>
// CHECK:           [[VAR_3_:%.+]] = shape.broadcast [[VAR_2_]], [[VAR_1_]] : tensor<4xindex>, tensor<3xindex> -> tensor<4xindex>
// CHECK-DAG:       [[VAR_4_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_0_]], [[VAR_3_]], dims = [0, 1, 2, 3] : (tensor<?x10x12x12xf32>, tensor<4xindex>) -> tensor<?x10x12x12xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_1_]], [[VAR_3_]], dims = [1, 2, 3] : (tensor<10x1x1xf32>, tensor<4xindex>) -> tensor<?x10x12x12xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = stablehlo.multiply [[VAR_4_]], [[VAR_5_]] : tensor<?x10x12x12xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = shape.shape_of [[VAR_4_]] : tensor<?x10x12x12xf32> -> tensor<4xindex>
// CHECK:           [[VAR_8_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[VAR_0_]], [[VAR_7_]], dims = [] : (tensor<f32>, tensor<4xindex>) -> tensor<?x10x12x12xf32>
// CHECK:           [[VAR_9_:%.+]] = stablehlo.compare  GT, [[VAR_4_]], [[VAR_8_]] : (tensor<?x10x12x12xf32>, tensor<?x10x12x12xf32>) -> tensor<?x10x12x12xi1>
// CHECK:           [[VAR_10_:%.+]] = stablehlo.select [[VAR_9_]], [[VAR_4_]], [[VAR_6_]] : tensor<?x10x12x12xi1>, tensor<?x10x12x12xf32>
// CHECK:           return [[VAR_10_]] : tensor<?x10x12x12xf32>
// CHECK:         }
}

// -----

func.func @test_neg(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Neg"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_neg
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.negate [[PARAM_0_]] : tensor<10x10xf32>
// CHECK:           return [[VAR_0_]] : tensor<10x10xf32>
// CHECK:         }

// -----

func.func @test_sin(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Sin"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_sin
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = stablehlo.sine [[PARAM_0_]] : tensor<10x10xf32>
// CHECK:           return [[VAR_0_]] : tensor<10x10xf32>
// CHECK:         }

func.func @test_where(%arg0 : tensor<16x24x36xi1>, %arg1 : tensor<16x24x36xi64>, %arg2 : tensor<16x24x36xi64>) -> tensor<16x24x36xi64> {
  %0 = "onnx.Where"(%arg0, %arg1, %arg2) : (tensor<16x24x36xi1>, tensor<16x24x36xi64>, tensor<16x24x36xi64>) -> tensor<16x24x36xi64>
  "func.return"(%0) : (tensor<16x24x36xi64>) -> ()
// CHECK-LABEL:  func.func @test_where
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<16x24x36xi1>, [[PARAM_1_:%.+]]: tensor<16x24x36xi64>, [[PARAM_2_:%.+]]: tensor<16x24x36xi64>) -> tensor<16x24x36xi64> {
// CHECK:           [[VAR_0_:%.+]] = shape.const_shape [16, 24, 36] : tensor<3xindex>
// CHECK-DAG:       [[VAR_1_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_0_]], [[VAR_0_]], dims = [0, 1, 2] : (tensor<16x24x36xi1>, tensor<3xindex>) -> tensor<16x24x36xi1>
// CHECK-DAG:       [[VAR_2_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_1_]], [[VAR_0_]], dims = [0, 1, 2] : (tensor<16x24x36xi64>, tensor<3xindex>) -> tensor<16x24x36xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = stablehlo.dynamic_broadcast_in_dim [[PARAM_2_]], [[VAR_0_]], dims = [0, 1, 2] : (tensor<16x24x36xi64>, tensor<3xindex>) -> tensor<16x24x36xi64>
// CHECK:           [[VAR_4_:%.+]] = stablehlo.select [[VAR_1_]], [[VAR_2_]], [[VAR_3_]] : tensor<16x24x36xi1>, tensor<16x24x36xi64>
// CHECK:           return [[VAR_4_]] : tensor<16x24x36xi64>
// CHECK:         }
}
