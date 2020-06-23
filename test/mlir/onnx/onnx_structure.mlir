// RUN: onnx-mlir-opt %s -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// CHECK-LABEL: @check_map1(%arg0: tuple<tensor<10xi64>, tensor<10xi64>>) -> tensor<*xi64> {
func @check_map1(%arg0: tuple<tensor<10xi64>, tensor<10xi64>>) -> tensor<*xi64> {
  %0 = "onnx.CastMap"(%arg0) {cast_to = "TO_FLOAT", map_form = "DENSE", max_map = 1 : i64} : (tuple<tensor<10xi64>, tensor<10xi64>>) -> tensor<*xi64>
  return %0 : tensor<*xi64>
  // CHECK-NEXT: %0 = "onnx.CastMap"(%arg0) {cast_to = "TO_FLOAT", map_form = "DENSE", max_map = 1 : i64} : (tuple<tensor<10xi64>, tensor<10xi64>>) -> tensor<*xi64>
}

// CHECK-LABEL: @check_string(%arg0: tensor<10x20x!onnx.String>) -> tensor<10x20x!onnx.String> {
func @check_string(%arg0: tensor<10x20x!onnx.String>) -> tensor<10x20x!onnx.String> {
  return %arg0 : tensor<10x20x!onnx.String>
  // CHECK-NEXT: return %arg0 : tensor<10x20x!onnx.String>
}

