// (c) Copyright 2022 - 2025 Advanced Micro Devices, Inc. All Rights reserved.

// RUN: onnx-mlir-opt %s --constprop-onnx --decompose-onnx=enable-split-to-slice --onnx-hybrid-transform --qdq-canonicalize=remove-qdq-around-ops | FileCheck %s

func.func @constprop() -> tensor<f32> {
  %0 = onnx.Constant {ResultNames = ["const0"]} dense<1.000000e+00> : tensor<f32>
  %1 = onnx.Constant {ResultNames = ["const0"]} dense<2.000000e+00> : tensor<f32>
  %2 = "onnx.Add"(%0, %1) {ResultNames = ["add0"]} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %2 : tensor<f32>
}

// CHECK-LABEL: @constprop()
// CHECK: onnx.Constant
// CHECK-SAME: ResultNames = ["add0"]
// CHECK-SAME: dense<3.000000e+00>

func.func @decompose(%arg0: tensor<8x4xf32>) -> (tensor<4x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>) {
  %0 = onnx.Constant dense<[4, 2, 2]> : tensor<3xi64>
  %1:3 = "onnx.Split"(%arg0, %0) {axis = 0 : si64, ResultNames = ["split_out0", "split_out1", "split_out2"]} : (tensor<8x4xf32>, tensor<3xi64>) -> (tensor<4x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>)
  return %1#0, %1#1, %1#2 : tensor<4x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>
}

// CHECK-LABEL: @decompose
// CHECK: onnx.Slice
// CHECK-SAME: ResultNames = ["split_out0"]
// CHECK-NEXT: onnx.Slice
// CHECK-SAME: ResultNames = ["split_out1"]
// CHECK-NEXT: onnx.Slice
// CHECK-SAME: ResultNames = ["split_out2"]

func.func @canonicalize(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = onnx.Constant {ResultNames = ["const0"]} dense<2.000000e+00> : tensor<f32>
  %1 = "onnx.Add"(%0, %arg0) {ResultNames = ["add0"]} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
}

// CHECK-LABEL: @canonicalize
// CHECK: "onnx.Add"(%arg0, %0)
// CHECK-SAME: ResultNames = ["add0"]

func.func @qdq_canonicalize(%arg0: tensor<1x128xf32>) -> tensor<1x1x128xf32> {
  %0 = onnx.Constant {ResultNames = ["scale"]} dense<1.000000e+00> : tensor<f32>
  %1 = onnx.Constant {ResultNames = ["zp"]} dense<128> : tensor<ui8>
  %2 = onnx.Constant {ResultNames = ["shape"]} dense<[1, 1, 128]> : tensor<3xi64>
  %3 = "onnx.QuantizeLinear"(%arg0, %0, %1) {ResultNames = ["q0"], axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x128xui8>
  %4 = "onnx.DequantizeLinear"(%3, %0, %1) {ResultNames = ["dq0"], axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x128xf32>
  %5 = "onnx.Reshape"(%4, %2) {ResultNames = ["reshape"], allowzero = 0 : si64} : (tensor<1x128xf32>, tensor<3xi64>) -> tensor<1x1x128xf32>
  %6 = "onnx.QuantizeLinear"(%5, %0, %1) {ResultNames = ["q1"], axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x1x128xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x1x128xui8>
  %7 = "onnx.DequantizeLinear"(%6, %0, %1) {ResultNames = ["dq1"], axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x128xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x1x128xf32>
  return %7 : tensor<1x1x128xf32>
}

// CHECK-LABEL: @qdq_canonicalize
// CHECK: onnx.QuantizeLinear
// CHECK-SAME: ResultNames = ["q0"]
// CHECK-NOT: onnx.DequantizeLinear
// CHECK: onnx.Reshape
// CHECK-SAME: ResultNames = ["q1"]
// CHECK-NOT: onnx.QuantizeLinear
// CHECK: onnx.DequantizeLinear
