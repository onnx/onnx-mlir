// RUN: onnx-mlir-opt --convert-onnx-to-mhlo --canonicalize %s -split-input-file | FileCheck %s

func.func @test_add(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_add
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>, [[PARAM_1_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = mhlo.add [[PARAM_0_]], [[PARAM_1_]] : tensor<10x10xf32>
// CHECK-NEXT:      return [[VAR_0_]] : tensor<10x10xf32>
// CHECK-NEXT:    }
}

func.func @test_add_dynamic(%arg0 : tensor<?x10xf32>, %arg1 : tensor<?x10xf32>) -> tensor<?x10xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x10xf32>, tensor<?x10xf32>) -> tensor<?x10xf32>
  "func.return"(%0) : (tensor<?x10xf32>) -> ()
// CHECK-LABEL:  func @test_add_dynamic
// CHECK:         [[VAR_0_:%.+]] = mhlo.add [[PARAM_0_:%.+]], [[PARAM_1_:%.+]] : tensor<?x10xf32>
// CHECK-NEXT:     return [[VAR_0_]] : tensor<?x10xf32>
}

func.func @test_relu(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Relu"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_relu
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-NEXT:     [[VAR_0_:%.+]] = mhlo.constant dense<0.000000e+00> : tensor<10x10xf32>
// CHECK-NEXT:     [[VAR_1_:%.+]] = mhlo.maximum [[PARAM_0_]], [[VAR_0_]] : tensor<10x10xf32>
// CHECK-NEXT:     return [[VAR_1_]] : tensor<10x10xf32>
// CHECK-NEXT:   }
}

func.func @test_relu_dynamic(%arg0 : tensor<?x10xf32>) -> tensor<?x10xf32> {
  %0 = "onnx.Relu"(%arg0) : (tensor<?x10xf32>) -> tensor<?x10xf32>
  "func.return"(%0) : (tensor<?x10xf32>) -> ()
// CHECK-LABEL:  func @test_relu_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x10xf32>) -> tensor<?x10xf32> {
// CHECK-DAG:      [[VAR_0_:%.+]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:      [[VAR_1_:%.+]] = shape.shape_of [[PARAM_0_]] : tensor<?x10xf32> -> tensor<2xindex>
// CHECK-NEXT:     [[VAR_2_:%.+]] = "mhlo.dynamic_broadcast_in_dim"([[VAR_0_]], [[VAR_1_]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<2xindex>) -> tensor<?x10xf32>
// CHECK-NEXT:     [[VAR_3_:%.+]] = mhlo.maximum [[PARAM_0_]], [[VAR_2_]] : tensor<?x10xf32>
// CHECK-NEXT:     return [[VAR_3_]] : tensor<?x10xf32>
// CHECK-NEXT:   }
}

func.func @test_exp(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Exp"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_exp
// CHECK:         [[VAR_0_:%.+]] = mhlo.exponential [[PARAM_0_:%.+]] : tensor<10x10xf32>
// CHECK-NEXT:     return [[VAR_0_]] : tensor<10x10xf32>
}

func.func @test_dynamic_exp(%arg0 : tensor<?x10xf32>) -> tensor<?x10xf32> {
  %0 = "onnx.Exp"(%arg0) : (tensor<?x10xf32>) -> tensor<?x10xf32>
  "func.return"(%0) : (tensor<?x10xf32>) -> ()
// CHECK-LABEL:  func @test_dynamic_exp
// CHECK:         [[VAR_0_:%.+]] = mhlo.exponential [[PARAM_0_:%.+]] : tensor<?x10xf32>
// CHECK-NEXT:     return [[VAR_0_]] : tensor<?x10xf32>
}

func.func @test_sigmoid(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Sigmoid"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_sigmoid
// CHECK:         [[VAR_0_:%.+]] = mhlo.logistic [[PARAM_0_:%.+]] : tensor<10x10xf32>
}

func.func @test_dynamic_sigmoid(%arg0 : tensor<?x10xf32>) -> tensor<?x10xf32> {
  %0 = "onnx.Sigmoid"(%arg0) : (tensor<?x10xf32>) -> tensor<?x10xf32>
  "func.return"(%0) : (tensor<?x10xf32>) -> ()
// CHECK-LABEL:  func @test_dynamic_sigmoid
// CHECK:         [[VAR_0_:%.+]] = mhlo.logistic [[PARAM_0_:%.+]] : tensor<?x10xf32>
}

func.func @test_abs(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Abs"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
// CHECK-LABEL:  func @test_abs
// CHECK: %0 = mhlo.abs %arg0 : tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
}

func.func @test_dyn_abs(%arg0 : tensor<?x10xf32>) -> tensor<?x10xf32> {
  %0 = "onnx.Abs"(%arg0) : (tensor<?x10xf32>) -> tensor<?x10xf32>
// CHECK-LABEL:  func @test_dyn_abs
// CHECK: %0 = mhlo.abs %arg0 : tensor<?x10xf32>
  "func.return"(%0) : (tensor<?x10xf32>) -> ()
}

func.func @test_and(%arg0 : tensor<10x10xi1>, %arg1 : tensor<10x10xi1>) -> tensor<10x10xi1> {
  %0 = "onnx.And"(%arg0, %arg1) : (tensor<10x10xi1>, tensor<10x10xi1>) -> tensor<10x10xi1>
// CHECK-LABEL:  func @test_and
// CHECK: %0 = mhlo.and %arg0, %arg1 : tensor<10x10xi1>
  "func.return"(%0) : (tensor<10x10xi1>) -> ()
}

func.func @test_dyn_and(%arg0 : tensor<?x10xi1>, %arg1 : tensor<?x10xi1>) -> tensor<?x10xi1> {
  %0 = "onnx.And"(%arg0, %arg1) : (tensor<?x10xi1>, tensor<?x10xi1>) -> tensor<?x10xi1>
  "func.return"(%0) : (tensor<?x10xi1>) -> ()
// CHECK-LABEL:  func @test_dyn_and
// CHECK: %2 = shape.broadcast %0, %1 : tensor<2xindex>, tensor<2xindex> -> tensor<2xindex>
// CHECK-DAG:  [[VAR_3_:%.+]] = "mhlo.dynamic_broadcast_in_dim"(%arg0, %2) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x10xi1>, tensor<2xindex>) -> tensor<?x10xi1>
// CHECK-DAG:  [[VAR_4_:%.+]] = "mhlo.dynamic_broadcast_in_dim"(%arg1, %2) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x10xi1>, tensor<2xindex>) -> tensor<?x10xi1>
// CHECK-NEXT  %5 = mhlo.and %3, %4 : tensor<?x10xi1>
}

func.func @cast_float(%arg0: tensor<2xf64>) -> tensor<2xf32> {
  %0 = "onnx.Cast"(%arg0) {to = f32} : (tensor<2xf64>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
// CHECK-LABEL:  func @cast_float
// CHECK:         %0 = mhlo.convert(%arg0) : (tensor<2xf64>) -> tensor<2xf32>
}

func.func @cast_int(%arg0: tensor<2xf32>) -> tensor<2xi32> {
  %0 = "onnx.Cast"(%arg0) {to = i32} : (tensor<2xf32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
// CHECK-LABEL:  func @cast_int
// CHECK:         %0 = mhlo.convert(%arg0) : (tensor<2xf32>) -> tensor<2xi32>
}

func.func @cast_dyn(%arg0: tensor<?x2xf64>) -> tensor<?x2xf32> {
  %0 = "onnx.Cast"(%arg0) {to = f32} : (tensor<?x2xf64>) -> tensor<?x2xf32>
  return %0 : tensor<?x2xf32>
// CHECK-LABEL:  func @cast_dyn
// CHECK:         %0 = mhlo.convert(%arg0) : (tensor<?x2xf64>) -> tensor<?x2xf32>
}

func.func @test_ceil(%arg0 : tensor<?x10xf32>) -> tensor<?x10xf32> {
  %0 = "onnx.Ceil"(%arg0) : (tensor<?x10xf32>) -> tensor<?x10xf32>
  "func.return"(%0) : (tensor<?x10xf32>) -> ()
// CHECK-LABEL:  func @test_ceil
// CHECK:         %0 = mhlo.ceil %arg0 : tensor<?x10xf32>
}

func.func @test_cos(%arg0 : tensor<?x10xf32>) -> tensor<?x10xf32> {
  %0 = "onnx.Cos"(%arg0) : (tensor<?x10xf32>) -> tensor<?x10xf32>
  "func.return"(%0) : (tensor<?x10xf32>) -> ()
// CHECK-LABEL:  func @test_cos
// CHECK:         %0 = mhlo.cosine %arg0 : tensor<?x10xf32>
}

func.func @test_less(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<3x4x5xi1> {
  %0 = "onnx.Less"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xi1>
  return %0 : tensor<3x4x5xi1>
// CHECK-LABEL:  func @test_less
// CHECK:         %0 = "mhlo.compare"(%arg0, %arg1) {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction LT>} : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xi1>
}

func.func @test_binary_elementwise_op_template_unknown_dims(%arg0: tensor<?x4x5xf32>, %arg1: tensor<1x?x1xf32>) -> tensor<?x4x5xi1> {
  %0 = "onnx.Less"(%arg0, %arg1) : (tensor<?x4x5xf32>, tensor<1x?x1xf32>) -> tensor<?x4x5xi1>
  return %0 : tensor<?x4x5xi1>
// CHECK-LABEL:  func @test_binary_elementwise_op_template_unknown_dims
// CHECK: %3 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %2) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<?x4x5xf32>, tensor<3xindex>) -> tensor<?x4x5xf32>
// CHECK: %4 = "mhlo.dynamic_broadcast_in_dim"(%arg1, %2) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x?x1xf32>, tensor<3xindex>) -> tensor<?x4x5xf32>
// CHECK: %5 = "mhlo.compare"(%3, %4) {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction LT>} : (tensor<?x4x5xf32>, tensor<?x4x5xf32>) -> tensor<?x4x5xi1>

}

func.func @test_less_unknown_dims_2(%arg0: tensor<?x?x5xf32>, %arg1: tensor<?x4x5xf32>) -> tensor<?x4x5xi1> {
  %0 = "onnx.Less"(%arg0, %arg1) : (tensor<?x?x5xf32>, tensor<?x4x5xf32>) -> tensor<?x4x5xi1>
  return %0 : tensor<?x4x5xi1>
// CHECK-LABEL:  func @test_less_unknown_dims_2
// CHECK: %3 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %2) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<?x?x5xf32>, tensor<3xindex>) -> tensor<?x4x5xf32>
// CHECK: %4 = "mhlo.dynamic_broadcast_in_dim"(%arg1, %2) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<?x4x5xf32>, tensor<3xindex>) -> tensor<?x4x5xf32>
// CHECK: %5 = "mhlo.compare"(%3, %4) {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction LT>} : (tensor<?x4x5xf32>, tensor<?x4x5xf32>) -> tensor<?x4x5xi1>
}