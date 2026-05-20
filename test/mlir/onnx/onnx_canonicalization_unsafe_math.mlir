// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt --unsafe-math-optimizations=false --shape-inference --canonicalize="test-convergence=true" --shape-inference --cse %s -split-input-file -verify-diagnostics | FileCheck %s

// -----

// Negative: variant with none bias.
func.func @test_fuse_add_conv_qdq_null_bias_unsafe_math_off(%arg0 : tensor<1x1x4x4xf32>, %arg1 : tensor<8x1x3x3xf32>) -> tensor<1x8x4x4xf32> {
    %cst = "onnx.NoValue"() {value} : () -> none
    %scale = onnx.Constant dense<0.5> : tensor<f32>
    %zp = onnx.Constant dense<0> : tensor<i8>
    %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "SAME_UPPER", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], strides = [1, 1]} : (tensor<1x1x4x4xf32>, tensor<8x1x3x3xf32>, none) -> tensor<1x8x4x4xf32>
    %1 = "onnx.QuantizeLinear"(%0, %scale, %zp) : (tensor<1x8x4x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x8x4x4xi8>
    %2 = "onnx.DequantizeLinear"(%1, %scale, %zp) : (tensor<1x8x4x4xi8>, tensor<f32>, tensor<i8>) -> tensor<1x8x4x4xf32>
    %addend_q = onnx.Constant dense<48> : tensor<1x1x1x1xi8>
    %addend_scale = onnx.Constant dense<6.250000e-02> : tensor<f32>
    %addend_zp = onnx.Constant dense<0> : tensor<i8>
    %addend = "onnx.DequantizeLinear"(%addend_q, %addend_scale, %addend_zp) : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
    %3 = "onnx.Add"(%2, %addend) : (tensor<1x8x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x8x4x4xf32>
    onnx.Return %3 : tensor<1x8x4x4xf32>
}

// CHECK-LABEL:  func.func @test_fuse_add_conv_qdq_null_bias_unsafe_math_off
// CHECK:           [[CONV_:%.+]] = "onnx.Conv"({{.*}}, {{.*}}, {{.*}}) {{.*}}: (tensor<1x1x4x4xf32>, tensor<8x1x3x3xf32>, none) -> tensor<{{.*}}>
// CHECK:           [[Q_:%.+]] = "onnx.QuantizeLinear"([[CONV_]], {{.*}}, {{.*}})
// CHECK:           [[DQ_:%.+]] = "onnx.DequantizeLinear"([[Q_]], {{.*}}, {{.*}})
// CHECK:           [[ADD_:%.+]] = "onnx.Add"([[DQ_]], {{.*}})
// CHECK:           onnx.Return [[ADD_]] : tensor<1x8x4x4xf32>

// -----

// Negative: variant with zero bias.
func.func @test_fuse_add_conv_qdq_zero_bias_unsafe_math_off(%arg0 : tensor<1x3x4x4xf32>, %arg1 : tensor<3x3x1x1xf32>) -> tensor<1x3x4x4xf32> {
    %bq = onnx.Constant dense<0> : tensor<3xi8>
    %bias_scale = onnx.Constant dense<2.000000e+00> : tensor<f32>
    %bias_zp = onnx.Constant dense<0> : tensor<i8>
    %bias = "onnx.DequantizeLinear"(%bq, %bias_scale, %bias_zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<3xi8>, tensor<f32>, tensor<i8>) -> tensor<3xf32>
    %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
    %zp = onnx.Constant dense<0> : tensor<i8>
    %0 = "onnx.Conv"(%arg0, %arg1, %bias) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], strides = [1, 1]} : (tensor<1x3x4x4xf32>, tensor<3x3x1x1xf32>, tensor<3xf32>) -> tensor<1x3x4x4xf32>
    %1 = "onnx.QuantizeLinear"(%0, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x3x4x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x3x4x4xi8>
    %2 = "onnx.DequantizeLinear"(%1, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x3x4x4xi8>, tensor<f32>, tensor<i8>) -> tensor<1x3x4x4xf32>
    %addend_q = onnx.Constant dense<48> : tensor<1x1x1x1xi8>
    %addend_scale = onnx.Constant dense<6.250000e-02> : tensor<f32>
    %addend_zp = onnx.Constant dense<0> : tensor<i8>
    %addend = "onnx.DequantizeLinear"(%addend_q, %addend_scale, %addend_zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32>
    %3 = "onnx.Add"(%2, %addend) : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32>
    onnx.Return %3 : tensor<1x3x4x4xf32>
}

// CHECK-LABEL:  func.func @test_fuse_add_conv_qdq_zero_bias_unsafe_math_off
// CHECK:           [[CONV_:%.+]] = "onnx.Conv"({{.*}}, {{.*}}, {{.*}}) {{.*}}: (tensor<1x3x4x4xf32>, tensor<3x3x1x1xf32>, tensor<3xf32>) -> tensor<{{.*}}>
// CHECK:           [[Q_:%.+]] = "onnx.QuantizeLinear"([[CONV_]], {{.*}}, {{.*}})
// CHECK:           [[DQ_:%.+]] = "onnx.DequantizeLinear"([[Q_]], {{.*}}, {{.*}})
// CHECK:           [[ADD_:%.+]] = "onnx.Add"([[DQ_]], {{.*}})
// CHECK:           onnx.Return [[ADD_]] : tensor<1x3x4x4xf32>

