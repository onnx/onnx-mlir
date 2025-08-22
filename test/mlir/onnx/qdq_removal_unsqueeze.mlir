// RUN: onnx-mlir-opt --canonicalize --qdq-around-op-opt-onnx-to-onnx %s -split-input-file | FileCheck %s

 func.func @unqueeze_op(%arg0: tensor<1x3xi8>) -> (tensor<1x1x3xi8>) {
    %0 = onnx.Constant dense<1.000000e-01> : tensor<1xf32>
    %1 = onnx.Constant dense<-128> : tensor<1xi8>
    %2 = onnx.Constant dense<1> : tensor<1xi64>
    %3 = "onnx.DequantizeLinear"(%arg0, %0, %1) {
      axis = 1 : si64,
      block_size = 0 : si64,
      onnx_node_name = "onnx.DequantizeLinear_0"} : (tensor<1x3xi8>, tensor<1xf32>, tensor<1xi8>) -> tensor<1x3xf32>
    %4 = "onnx.Unsqueeze"(%3, %2) {onnx_node_name = "onnx.Unsqueeze_1"} : (tensor<1x3xf32>, tensor<1xi64>) -> tensor<1x1x3xf32>
    %5 = "onnx.QuantizeLinear"(%4, %0, %1) {
      axis = 1 : si64,
      block_size = 0 : si64,
      onnx_node_name = "onnx.QuantizeLinear_2",
      output_dtype = 0 : si64,
      saturate = 1 : si64} : (tensor<1x1x3xf32>, tensor<1xf32>, tensor<1xi8>) -> tensor<1x1x3xi8>
    return %5 : tensor<1x1x3xi8>
  }
  
// CHECK-LABEL:  func.func @unqueeze_op
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3xi8>) -> tensor<1x1x3xi8> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Unsqueeze"([[PARAM_0_]], [[VAR_0_]]) {onnx_node_name = "onnx.Unsqueeze_1"} : (tensor<1x3xi8>, tensor<1xi64>) -> tensor<1x1x3xi8>
// CHECK:           return [[VAR_1_]] : tensor<1x1x3xi8>
// CHECK:         }