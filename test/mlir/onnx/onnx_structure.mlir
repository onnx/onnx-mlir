// RUN: onnx-mlir-opt %s -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// CHECK-LABEL: @check_map1(%arg0: tuple<i64, f32>) -> tensor<*xf32> {
func @check_map1(%arg0: tuple<i64, f32>) -> tensor<*xf32> {
  %0 = "onnx.CastMap"(%arg0) {cast_to = "TO_FLOAT", map_form = "DENSE", max_map = 1 : i64} : (tuple<i64, f32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
  // CHECK-NEXT: %0 = "onnx.CastMap"(%arg0) {cast_to = "TO_FLOAT", map_form = "DENSE", max_map = 1 : i64} : (tuple<i64, f32>) -> tensor<*xf32>
}

// CHECK-LABEL: @check_string(%arg0: tensor<10x20x!onnx.String>) -> tensor<10x20x!onnx.String> {
func @check_string(%arg0: tensor<10x20x!onnx.String>) -> tensor<10x20x!onnx.String> {
  return %arg0 : tensor<10x20x!onnx.String>
  // CHECK-NEXT: return %arg0 : tensor<10x20x!onnx.String>
}

// CHECK-LABEL: @check_seq(%arg0: tensor<10x20xf32>, %arg1: tensor<5x20xf32>) -> tensor<*xf32> {
func @check_seq(%arg0: tensor<10x20xf32>, %arg1: tensor<5x20xf32>) -> tensor<*xf32> {
  %cst = "onnx.Constant"() {value = dense<[0]> : tensor<1xi32>} : () -> tensor<1xi32>
  %0 = "onnx.SequenceConstruct"(%arg0, %arg1) : (tensor<10x20xf32>, tensor<5x20xf32>) -> !onnx.Seq<tensor<10x20xf32>, tensor<5x20xf32>>
  %1 = "onnx.SequenceAt"(%0, %cst) : (!onnx.Seq<tensor<10x20xf32>, tensor<5x20xf32>>, tensor<1xi32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>
  // CHECK-NEXT: %0 = "onnx.Constant"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK-NEXT: %1 = "onnx.SequenceConstruct"(%arg0, %arg1) : (tensor<10x20xf32>, tensor<5x20xf32>) -> !onnx.Seq<tensor<10x20xf32>, tensor<5x20xf32>>
  // CHECK-NEXT: %2 = "onnx.SequenceAt"(%1, %0) : (!onnx.Seq<tensor<10x20xf32>, tensor<5x20xf32>>, tensor<1xi32>) -> tensor<*xf32>
}

