// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_split(%arg0 : tensor<2x6xf32>) -> (tensor<2x2xf32>, tensor<2x4xf32>) {
  %split = "onnx.Constant"() {value = dense<[2, 4]> : tensor<2xi64>} : () -> tensor<2xi64>
  %0:2 = "onnx.Split"(%arg0, %split) {axis = 1 : si64} : (tensor<2x6xf32>, tensor<2xi64>) -> (tensor<2x2xf32>, tensor<2x4xf32>)
  "func.return"(%0#0, %0#1) : (tensor<2x2xf32>, tensor<2x4xf32>) -> ()
// CHECK-LABEL:  func @test_split
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x6xf32>) -> (tensor<2x2xf32>, tensor<2x4xf32>) {
// CHECK-DAG:       [[START0:%.+]] = tosa.const_shape  {values = dense<0> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[SIZE0:%.+]] = tosa.const_shape  {values = dense<2> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[SLICE0:%.+]] = tosa.slice [[PARAM_0_]], [[START0]], [[SIZE0]] : (tensor<2x6xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<2x2xf32>
// CHECK-DAG:       [[START1:%.+]] = tosa.const_shape  {values = dense<[0, 2]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[SIZE1:%.+]] = tosa.const_shape  {values = dense<[2, 4]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[SLICE1:%.+]] = tosa.slice [[PARAM_0_]], [[START1]], [[SIZE1]] : (tensor<2x6xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<2x4xf32>
// CHECK:           return [[SLICE0]], [[SLICE1]] : tensor<2x2xf32>, tensor<2x4xf32>
}

// -----

func.func @test_split_negative_axis(%arg0 : tensor<2x6xf32>) -> (tensor<2x2xf32>, tensor<2x4xf32>) {
  %split = "onnx.Constant"() {value = dense<[2, 4]> : tensor<2xi64>} : () -> tensor<2xi64>
  %0:2 = "onnx.Split"(%arg0, %split) {axis = -1 : si64} : (tensor<2x6xf32>, tensor<2xi64>) -> (tensor<2x2xf32>, tensor<2x4xf32>)
  "func.return"(%0#0, %0#1) : (tensor<2x2xf32>, tensor<2x4xf32>) -> ()
// CHECK-LABEL:  func @test_split_negative_axis
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x6xf32>) -> (tensor<2x2xf32>, tensor<2x4xf32>) {
// CHECK-DAG:       [[START0:%.+]] = tosa.const_shape  {values = dense<0> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[SIZE0:%.+]] = tosa.const_shape  {values = dense<2> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[SLICE0:%.+]] = tosa.slice [[PARAM_0_]], [[START0]], [[SIZE0]] : (tensor<2x6xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<2x2xf32>
// CHECK-DAG:       [[START1:%.+]] = tosa.const_shape  {values = dense<[0, 2]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-DAG:       [[SIZE1:%.+]] = tosa.const_shape  {values = dense<[2, 4]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[SLICE1:%.+]] = tosa.slice [[PARAM_0_]], [[START1]], [[SIZE1]] : (tensor<2x6xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<2x4xf32>
// CHECK:           return [[SLICE0]], [[SLICE1]] : tensor<2x2xf32>, tensor<2x4xf32>
}

// -----

func.func @test_split_num_outputs(%arg0 : tensor<2x6xf32>) -> (tensor<2x3xf32>, tensor<2x3xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %0:2 = "onnx.Split"(%arg0, %none) {axis = 1 : si64, num_outputs = 2 : si64} : (tensor<2x6xf32>, none) -> (tensor<2x3xf32>, tensor<2x3xf32>)
  "func.return"(%0#0, %0#1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> ()
// CHECK-LABEL:  func @test_split_num_outputs
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x6xf32>) -> (tensor<2x3xf32>, tensor<2x3xf32>) {
// CHECK:           [[SLICE0:%.+]] = tosa.slice [[PARAM_0_]], {{.*}} -> tensor<2x3xf32>
// CHECK:           [[SLICE1:%.+]] = tosa.slice [[PARAM_0_]], {{.*}} -> tensor<2x3xf32>
// CHECK:           return [[SLICE0]], [[SLICE1]] : tensor<2x3xf32>, tensor<2x3xf32>
}
