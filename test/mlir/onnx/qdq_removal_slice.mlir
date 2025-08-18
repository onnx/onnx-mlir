// RUN: onnx-mlir-opt --canonicalize --qdq-around-op-opt-onnx-to-onnx %s -split-input-file | FileCheck %s

func.func @slice_op(%arg0: tensor<1x4xui8> {onnx.name = "input_quant"})-> (tensor<1x2xui8> {onnx.name = "output_quant"}) {
    %0 = onnx.Constant dense<1.000000e-01> : tensor<f32>
    %1 = onnx.Constant dense<128> : tensor<ui8>
    %2 = onnx.Constant dense<1> : tensor<1xi64>
    %3 = onnx.Constant dense<3> : tensor<1xi64>
    %4 = onnx.Constant dense<1> : tensor<1xi64>
    %5 = onnx.Constant dense<1> : tensor<1xi64>
    %6 = onnx.Constant dense<1.000000e-01> : tensor<f32>
    %7 = onnx.Constant dense<128> : tensor<ui8>
    %8 = "onnx.DequantizeLinear"(%arg0, %0, %1) {
      axis = 1 : si64,
      block_size = 0 : si64,
      onnx_node_name = "onnx.DequantizeLinear_0"} : (tensor<1x4xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x4xf32>
    %9 = "onnx.Slice"(%8, %2, %3, %4, %5) {onnx_node_name = "onnx.Slice_1"} : (tensor<1x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x2xf32>
    %10 = "onnx.QuantizeLinear"(%9, %6, %7) {
      axis = 1 : si64,
      block_size = 0 : si64,
      onnx_node_name = "onnx.QuantizeLinear_2",
      output_dtype = 0 : si64,
      saturate = 1 : si64} : (tensor<1x2xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x2xui8>
    return %10 : tensor<1x2xui8>
  }

// CHECK-LABEL:  func.func @slice_op
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x4xui8> {onnx.name = "input_quant"}) -> (tensor<1x2xui8> {onnx.name = "output_quant"}) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Slice"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]], [[VAR_0_]], [[VAR_0_]]) {onnx_node_name = "onnx.Slice_1"} : (tensor<1x4xui8>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x2xui8>
// CHECK:           return [[VAR_2_]] : tensor<1x2xui8>
// CHECK:         }