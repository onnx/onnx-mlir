// RUN: onnx-mlir-opt --shape-inference --decompose-onnx=enable-reducel2-decompose=true %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --shape-inference --decompose-onnx=enable-reducel2-decompose=false %s -split-input-file | FileCheck %s --check-prefix=DISABLED-CHECK

func.func @test_reducel2_basic(%arg0: tensor<2x3x4xf32>, %arg1: tensor<1xi64>) -> tensor<?x?xf32>
{
    %0 = "onnx.ReduceL2"(%arg0, %arg1) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<?x?xf32>
    onnx.Return %0 : tensor<?x?xf32>
// CHECK-LABEL:  func.func @test_reducel2_basic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xf32>, [[PARAM_1_:%.+]]: tensor<1xi64>) -> tensor<?x?xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[PARAM_0_]]) : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.ReduceSum"([[VAR_0_]], [[PARAM_1_]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<?x?xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Sqrt"([[VAR_1_]]) : (tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<?x?xf32>
// CHECK:         }

// DISABLED-CHECK-LABEL:  func.func @test_reducel2_basic
// DISABLED-CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xf32>, [[PARAM_1_:%.+]]: tensor<1xi64>) -> tensor<?x?xf32> {
// DISABLED-CHECK:           [[VAR_0_:%.+]] = "onnx.ReduceL2"([[PARAM_0_]], [[PARAM_1_]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<?x?xf32>
// DISABLED-CHECK:           onnx.Return [[VAR_0_]] : tensor<?x?xf32>
// DISABLED-CHECK:         }
}

// -----

func.func @test_reducel2_keepdims(%arg0: tensor<2x3x4xf32>, %arg1: tensor<1xi64>) -> tensor<?x?x?xf32>
{
    %0 = "onnx.ReduceL2"(%arg0, %arg1) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<?x?x?xf32>
    onnx.Return %0 : tensor<?x?x?xf32>
// CHECK-LABEL:  func.func @test_reducel2_keepdims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xf32>, [[PARAM_1_:%.+]]: tensor<1xi64>) -> tensor<?x?x?xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[PARAM_0_]]) : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.ReduceSum"([[VAR_0_]], [[PARAM_1_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<?x?x?xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Sqrt"([[VAR_1_]]) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<?x?x?xf32>
// CHECK:         }

// DISABLED-CHECK-LABEL:  func.func @test_reducel2_keepdims
// DISABLED-CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xf32>, [[PARAM_1_:%.+]]: tensor<1xi64>) -> tensor<?x?x?xf32> {
// DISABLED-CHECK:           [[VAR_0_:%.+]] = "onnx.ReduceL2"([[PARAM_0_]], [[PARAM_1_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, tensor<1xi64>) -> tensor<?x?x?xf32>
// DISABLED-CHECK:           onnx.Return [[VAR_0_]] : tensor<?x?x?xf32>
// DISABLED-CHECK:         }
}

// -----

func.func @test_reducel2_noaxes(%arg0: tensor<2x3x4xf32>) -> tensor<1x1x1xf32>
{
    %none = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.ReduceL2"(%arg0, %none) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, none) -> tensor<1x1x1xf32>
    onnx.Return %0 : tensor<1x1x1xf32>
// CHECK-LABEL:  func.func @test_reducel2_noaxes
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xf32>) -> tensor<1x1x1xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_1_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[PARAM_0_]]) : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.ReduceSum"([[VAR_1_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, none) -> tensor<1x1x1xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Sqrt"([[VAR_2_]]) : (tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
// CHECK:           onnx.Return [[VAR_3_]] : tensor<1x1x1xf32>
// CHECK:         }

// DISABLED-CHECK-LABEL:  func.func @test_reducel2_noaxes
// DISABLED-CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xf32>) -> tensor<1x1x1xf32> {
// DISABLED-CHECK:           [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// DISABLED-CHECK:           [[VAR_1_:%.+]] = "onnx.ReduceL2"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3x4xf32>, none) -> tensor<1x1x1xf32>
// DISABLED-CHECK:           onnx.Return [[VAR_1_]] : tensor<1x1x1xf32>
// DISABLED-CHECK:         }
}
