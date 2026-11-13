// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_reduce_max(%arg0: tensor<2x5x3xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
  %1 = "onnx.ReduceMax"(%arg0, %0) {keepdims = 0 : si64} : (tensor<2x5x3xf32>, tensor<1xi64>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
// CHECK-LABEL:  func.func @test_reduce_max
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x5x3xf32>) -> tensor<2x3xf32> {
// CHECK:           [[VAR_1_:%.+]] = tosa.reduce_max [[PARAM_0_]] {axis = 1 : i32} : (tensor<2x5x3xf32>) -> tensor<2x1x3xf32>
// CHECK:           [[SHAPE:%.+]] = tosa.const_shape  {values = dense<[2, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_3_:%.+]] = tosa.reshape [[VAR_1_]], [[SHAPE]] : (tensor<2x1x3xf32>, !tosa.shape<2>) -> tensor<2x3xf32>
// CHECK:           return [[VAR_3_]] : tensor<2x3xf32>
}

// -----

func.func @test_reduce_min_keepdims(%arg0: tensor<2x5x3xf32>) -> tensor<2x1x3xf32> {
  %0 = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
  %1 = "onnx.ReduceMin"(%arg0, %0) {keepdims = 1 : si64} : (tensor<2x5x3xf32>, tensor<1xi64>) -> tensor<2x1x3xf32>
  return %1 : tensor<2x1x3xf32>
// CHECK-LABEL:  func.func @test_reduce_min_keepdims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x5x3xf32>) -> tensor<2x1x3xf32> {
// CHECK:           [[VAR_1_:%.+]] = tosa.reduce_min [[PARAM_0_]] {axis = 1 : i32} : (tensor<2x5x3xf32>) -> tensor<2x1x3xf32>
// CHECK:           return [[VAR_1_]] : tensor<2x1x3xf32>
}

// -----

func.func @test_reduce_prod(%arg0: tensor<2x5x3xf32>) -> tensor<2xf32> {
  %0 = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.ReduceProd"(%arg0, %0) {keepdims = 0 : si64} : (tensor<2x5x3xf32>, tensor<2xi64>) -> tensor<2xf32>
  return %1 : tensor<2xf32>
// CHECK-LABEL:  func.func @test_reduce_prod
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x5x3xf32>) -> tensor<2xf32> {
// CHECK:           [[VAR_1_:%.+]] = tosa.reduce_product [[PARAM_0_]] {axis = 1 : i32} : (tensor<2x5x3xf32>) -> tensor<2x1x3xf32>
// CHECK:           [[VAR_2_:%.+]] = tosa.reduce_product [[VAR_1_]] {axis = 2 : i32} : (tensor<2x1x3xf32>) -> tensor<2x1x1xf32>
// CHECK:           [[SHAPE:%.+]] = tosa.const_shape  {values = dense<2> : tensor<1xindex>} : () -> !tosa.shape<1>
// CHECK:           [[VAR_4_:%.+]] = tosa.reshape [[VAR_2_]], [[SHAPE]] : (tensor<2x1x1xf32>, !tosa.shape<1>) -> tensor<2xf32>
// CHECK:           return [[VAR_4_]] : tensor<2xf32>
}

// -----

func.func @test_reduce_sum(%arg0: tensor<2x5x3xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
  %1 = "onnx.ReduceSum"(%arg0, %0) {keepdims = 0 : si64} : (tensor<2x5x3xf32>, tensor<1xi64>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
// CHECK-LABEL:  func.func @test_reduce_sum
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x5x3xf32>) -> tensor<2x3xf32> {
// CHECK:           [[VAR_1_:%.+]] = tosa.reduce_sum [[PARAM_0_]] {axis = 1 : i32} : (tensor<2x5x3xf32>) -> tensor<2x1x3xf32>
// CHECK:           [[SHAPE:%.+]] = tosa.const_shape  {values = dense<[2, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_3_:%.+]] = tosa.reshape [[VAR_1_]], [[SHAPE]] : (tensor<2x1x3xf32>, !tosa.shape<2>) -> tensor<2x3xf32>
// CHECK:           return [[VAR_3_]] : tensor<2x3xf32>
}

// -----

func.func @test_reduce_sum_noaxes(%arg0: tensor<2x5x3xf32>) -> tensor<f32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.ReduceSum"(%arg0, %none) {keepdims = 0 : si64} : (tensor<2x5x3xf32>, none) -> tensor<f32>
  return %1 : tensor<f32>
// CHECK-LABEL:  func.func @test_reduce_sum_noaxes
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x5x3xf32>) -> tensor<f32> {
// CHECK:           [[VAR_1_:%.+]] = tosa.reduce_sum [[PARAM_0_]] {axis = 0 : i32} : (tensor<2x5x3xf32>) -> tensor<1x5x3xf32>
// CHECK:           [[VAR_2_:%.+]] = tosa.reduce_sum [[VAR_1_]] {axis = 1 : i32} : (tensor<1x5x3xf32>) -> tensor<1x1x3xf32>
// CHECK:           [[VAR_3_:%.+]] = tosa.reduce_sum [[VAR_2_]] {axis = 2 : i32} : (tensor<1x1x3xf32>) -> tensor<1x1x1xf32>
// CHECK:           [[SHAPE:%.+]] = tosa.const_shape  {values = dense<> : tensor<0xindex>} : () -> !tosa.shape<0>
// CHECK:           [[VAR_5_:%.+]] = tosa.reshape [[VAR_3_]], [[SHAPE]] : (tensor<1x1x1xf32>, !tosa.shape<0>) -> tensor<f32>
// CHECK:           return [[VAR_5_]] : tensor<f32>
}

// -----

func.func @test_reduce_sum_noop(%arg0: tensor<2x5x3xf32>) -> tensor<2x5x3xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.ReduceSum"(%arg0, %none) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<2x5x3xf32>, none) -> tensor<2x5x3xf32>
  return %1 : tensor<2x5x3xf32>
// CHECK-LABEL:  func.func @test_reduce_sum_noop
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x5x3xf32>) -> tensor<2x5x3xf32> {
// CHECK-NOT:       tosa.reduce_sum
// CHECK:           return
}

// -----

func.func @test_reduce_max_int(%arg0: tensor<2x5x3xi32>) -> tensor<2x3xi32> {
  %0 = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
  %1 = "onnx.ReduceMax"(%arg0, %0) {keepdims = 0 : si64} : (tensor<2x5x3xi32>, tensor<1xi64>) -> tensor<2x3xi32>
  return %1 : tensor<2x3xi32>
// CHECK-LABEL:  func.func @test_reduce_max_int
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x5x3xi32>) -> tensor<2x3xi32> {
// CHECK:           [[VAR_1_:%.+]] = tosa.reduce_max [[PARAM_0_]] {axis = 1 : i32} : (tensor<2x5x3xi32>) -> tensor<2x1x3xi32>
// CHECK:           [[SHAPE:%.+]] = tosa.const_shape  {values = dense<[2, 3]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK:           [[VAR_3_:%.+]] = tosa.reshape [[VAR_1_]], [[SHAPE]] : (tensor<2x1x3xi32>, !tosa.shape<2>) -> tensor<2x3xi32>
// CHECK:           return [[VAR_3_]] : tensor<2x3xi32>
}
