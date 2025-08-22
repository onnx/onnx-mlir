// RUN: onnx-mlir-opt --canonicalize --qdq-around-op-opt-onnx-to-onnx %s -split-input-file | FileCheck %s

  func.func @gather_op() -> (tensor<2xui8> {onnx.name = "quantized"}) {
    %0 = onnx.Constant dense<[0, 1, 2, 3]> : tensor<4xui8>
    %1 = onnx.Constant dense<1.000000e-01> : tensor<1xf32>
    %2 = onnx.Constant dense<0> : tensor<1xui8>
    %3 = onnx.Constant dense<[0, 2]> : tensor<2xi64>
    %4 = "onnx.DequantizeLinear"(%0, %1, %2) {axis = 1 : si64, onnx_node_name = "onnx.DequantizeLinear_0"} : (tensor<4xui8>, tensor<1xf32>, tensor<1xui8>) -> tensor<4xf32>
    %5 = "onnx.Gather"(%4, %3) {axis = 0 : si64, onnx_node_name = "onnx.Gather_1"} : (tensor<4xf32>, tensor<2xi64>) -> tensor<2xf32>
    %6 = "onnx.QuantizeLinear"(%5, %1, %2) {axis = 1 : si64, onnx_node_name = "onnx.QuantizeLinear_2", saturate = 1 : si64} : (tensor<2xf32>, tensor<1xf32>, tensor<1xui8>) -> tensor<2xui8>
    return %6 : tensor<2xui8>
  }

// CHECK-LABEL:  func.func @gather_op
// CHECK-SAME:   () -> (tensor<2xui8> {onnx.name = "quantized"}) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[0, 1, 2, 3]> : tensor<4xui8>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[0, 2]> : tensor<2xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Gather"([[VAR_0_]], [[VAR_1_]]) {axis = 0 : si64, onnx_node_name = "onnx.Gather_1"} : (tensor<4xui8>, tensor<2xi64>) -> tensor<2xui8>
// CHECK:           return [[VAR_2_]] : tensor<2xui8>
// CHECK:         }