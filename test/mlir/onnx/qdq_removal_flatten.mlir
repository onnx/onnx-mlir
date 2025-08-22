// RUN: onnx-mlir-opt --canonicalize --qdq-around-op-opt-onnx-to-onnx %s -split-input-file | FileCheck %s

  func.func @flatten_op(%arg0: tensor<1x2x2xui8>) -> (tensor<1x4xui8>) {
    %0 = onnx.Constant dense<1.000000e-01> : tensor<f32>
    %1 = onnx.Constant dense<128> : tensor<ui8>
    %2 = "onnx.DequantizeLinear"(%arg0, %0, %1) {
      axis = 1 : si64,
      block_size = 0 : si64,
      onnx_node_name = "onnx.DequantizeLinear_0"} : (tensor<1x2x2xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x2x2xf32>
    %3 = "onnx.Flatten"(%2) {axis = 1 : si64, onnx_node_name = "onnx.Flatten_1"} : (tensor<1x2x2xf32>) -> tensor<1x4xf32>
    %4 = "onnx.QuantizeLinear"(%3, %0, %1) {
      axis = 1 : si64,
      block_size = 0 : si64,
      onnx_node_name = "onnx.QuantizeLinear_2",
      output_dtype = 0 : si64,
      saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x4xui8>
    return %4 : tensor<1x4xui8>
  }

// CHECK-LABEL:  func.func @flatten_op
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x2x2xui8>) -> tensor<1x4xui8> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Flatten"([[PARAM_0_]]) {axis = 1 : si64, onnx_node_name = "onnx.Flatten_1"} : (tensor<1x2x2xui8>) -> tensor<1x4xui8>
// CHECK:           return [[VAR_0_]] : tensor<1x4xui8>
// CHECK:         }