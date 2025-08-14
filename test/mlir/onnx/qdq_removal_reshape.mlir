// RUN: onnx-mlir-opt --canonicalize --qdq-around-op-opt-onnx-to-onnx %s -split-input-file | FileCheck %s

  func.func @reshape_op(%arg0: tensor<1x4xui8> {onnx.name = "input_quant"} loc(unknown)) -> (tensor<2x2xui8> {onnx.name = "output_quant"}) {
    %0 = onnx.Constant dense<1.000000e-01> : tensor<f32>
    %1 = onnx.Constant dense<128> : tensor<ui8>
    %2 = onnx.Constant dense<2> : tensor<2xi64>
    %3 = onnx.Constant dense<1.000000e-01> : tensor<f32>
    %4 = onnx.Constant dense<128> : tensor<ui8>
    %5 = "onnx.DequantizeLinear"(%arg0, %0, %1) {
      axis = 1 : si64,
      block_size = 0 : si64,
      onnx_node_name = "onnx.DequantizeLinear_0"} : (tensor<1x4xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x4xf32>
    %6 = "onnx.Reshape"(%5, %2) {allowzero = 0 : si64, onnx_node_name = "onnx.Reshape_1"} : (tensor<1x4xf32>, tensor<2xi64>) -> tensor<2x2xf32>
    %7 = "onnx.QuantizeLinear"(%6, %3, %4) {
      axis = 1 : si64,
      block_size = 0 : si64,
      onnx_node_name = "onnx.QuantizeLinear_2",
      output_dtype = 0 : si64,
      saturate = 1 : si64} : (tensor<2x2xf32>, tensor<f32>, tensor<ui8>) -> tensor<2x2xui8>
    return %7 : tensor<2x2xui8>
  }

// CHECK-LABEL:  func.func @reshape_op
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x4xui8> {onnx.name = "input_quant"}) -> (tensor<2x2xui8> {onnx.name = "output_quant"}) {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<2> : tensor<2xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_0_]]) {allowzero = 0 : si64, onnx_node_name = "onnx.Reshape_1"} : (tensor<1x4xui8>, tensor<2xi64>) -> tensor<2x2xui8>
// CHECK:           return [[VAR_1_]] : tensor<2x2xui8>
// CHECK:         }