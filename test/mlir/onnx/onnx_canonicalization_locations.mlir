// Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt --shape-inference --canonicalize="test-convergence=true" --shape-inference --cse %s -split-input-file --mlir-print-debuginfo | FileCheck %s

// CHECK-LABEL:  func.func @layernorm_with_bias
func.func @layernorm_with_bias(%arg0: tensor<1x384x768xf32>, %arg1: tensor<768xf32>, %arg3: tensor<768xf32>) -> tensor<1x384x768xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %y, %mean, %stddev = "onnx.LayerNormalization"(%arg0, %arg1, %none) {axis = 2 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, none) -> (tensor<1x384x768xf32>, none, none) loc("LN")
  %ret = "onnx.Add"(%y, %arg3) : (tensor<1x384x768xf32>, tensor<768xf32>) -> tensor<1x384x768xf32> loc("Bias")
  return %ret : tensor<1x384x768xf32>
  // CHECK:           [[Y_:%.+]], [[Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"(%arg0, %arg1, %arg2) {axis = 2 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<1x384x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x384x768xf32>, none, none) loc([[LOC_FUSED:#.+]])
  // CHECK:           return [[Y_]] : tensor<1x384x768xf32>
  // CHECK-DAG:       [[LOC_LN:#.+]] = loc("LN")
  // CHECK-DAG:       [[LOC_BIAS:#.+]] = loc("Bias")
  // CHECK:           [[LOC_FUSED]] = loc(fused[[[LOC_LN]], [[LOC_BIAS]]]) 
}


// -----

func.func @test_reorder_relu_maxpool(%arg0: tensor<1x64x32x32xf32>) -> tensor<1x64x16x16xf32> {
  %0 = "onnx.Relu"(%arg0) : (tensor<1x64x32x32xf32>) -> tensor<1x64x32x32xf32> loc("Relu")
  %1 = "onnx.MaxPoolSingleOut"(%0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [2, 2], onnx_node_name = "onnx.MaxPoolSingleOut_1", storage_order = 0 : si64, strides = [2, 2]} : (tensor<1x64x32x32xf32>) -> tensor<1x64x16x16xf32> loc("MaxPool")
  return %1 : tensor<1x64x16x16xf32>

  // CHECK-LABEL: func @test_reorder_relu_maxpool
  // CHECK:           [[VAR_0_:%.+]] = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [2, 2], storage_order = 0 : si64, strides = [2, 2]} : (tensor<1x64x32x32xf32>) -> tensor<1x64x16x16xf32> loc([[LOC_MAX_POOL:#.+]])
  // CHECK:           [[VAR_1_:%.+]] = "onnx.Relu"([[VAR_0_]]) : (tensor<1x64x16x16xf32>) -> tensor<1x64x16x16xf32> loc([[LOC_RELU:#.+]])
  // CHECK-DAG:       [[LOC_MAX_POOL:#.+]] = loc("MaxPool")
  // CHECK-DAG:       [[LOC_RELU:#.+]] = loc("Relu")
}

// -----

func.func @test_recompose_concat(%arg0: tensor<1x3x4xf32>, %arg1: tensor<1x3x4xf32> ) -> tensor<1x12x4xf32> {
  %0 = "onnx.Concat"(%arg0, %arg1) {axis = 1 : si64} : (tensor<1x3x4xf32>, tensor<1x3x4xf32>) -> tensor<1x6x4xf32> loc("Concat1")
  %1 = "onnx.Concat"(%0, %arg0) {axis = 1 : si64} : (tensor<1x6x4xf32>, tensor<1x3x4xf32>) -> tensor<1x9x4xf32> loc("Concat2")
  %2 = "onnx.Concat"(%1, %arg1) {axis = 1 : si64} : (tensor<1x9x4xf32>, tensor<1x3x4xf32>) -> tensor<1x12x4xf32> loc("Concat3")
  return %2 : tensor<1x12x4xf32>

  // CHECK-LABEL: func @test_recompose_concat
  // CHECK: "onnx.Concat"
  // CHECK-SAME: loc([[LOC_FUSED:#.+]])
  // CHECK-DAG:       [[LOC_C1:#.+]] = loc("Concat1")
  // CHECK-DAG:       [[LOC_C2:#.+]] = loc("Concat2")
  // CHECK-DAG:       [[LOC_C3:#.+]] = loc("Concat3")
  // CHECK:           [[LOC_FUSED]] = loc(fused[[[LOC_C3]], [[LOC_C2]], [[LOC_C1]]]) 
}

// -----

func.func @consecutive_clips(%arg0: tensor<3x1024x1024xf32>) -> (tensor<3x1024x1024xf32> {onnx.name = "output"}) {
  %0 = onnx.Constant dense<-5.000000e-01> : tensor<f32>
  %1 = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %2 = onnx.Constant dense<-3.000000e-01> : tensor<f32>
  %3 = onnx.Constant dense<3.000000e-01> : tensor<f32>
  %4 = "onnx.Clip"(%arg0, %0, %1) : (tensor<3x1024x1024xf32>, tensor<f32>, tensor<f32>) -> tensor<3x1024x1024xf32> loc("Clip1")
  %5 = "onnx.Clip"(%4, %2, %3) : (tensor<3x1024x1024xf32>, tensor<f32>, tensor<f32>) -> tensor<3x1024x1024xf32> loc("Clip2")
  onnx.Return %5 : tensor<3x1024x1024xf32>

  // CHECK-LABEL: func.func @consecutive_clips
  // CHECK: onnx.Max  
  // CHECK-SAME: loc([[FUSED_LOC:#.+]])
  // CHECK: onnx.Min
  // CHECK-SAME: loc([[FUSED_LOC]])

  // CHECK: onnx.Clip
  // CHECK-SAME: loc([[FUSED_LOC]])

  // CHECK-DAG: [[LOC_CLIP1:#.+]] = loc("Clip1")
  // CHECK-DAG: [[LOC_CLIP2:#.+]] = loc("Clip2")
  // CHECK: [[FUSED_LOC]] = loc(fused[[[LOC_CLIP2]], [[LOC_CLIP1]]])
}

// -----

func.func @maxpool_k5_p1_s1_maxpool_k3_p1_s1_quant_int8(%arg0: tensor<1x3x224x224xi8> {onnx.name = "input"}) -> tensor<1x3x222x222xi8> {
  %0 = onnx.Constant dense<2.500000e-01> : tensor<f32> loc("scale")
  %1 = onnx.Constant dense<0> : tensor<i8> loc("zero_point")
  %2 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x3x224x224xi8>, tensor<f32>, tensor<i8>) -> tensor<1x3x224x224xf32> loc("DequantizeLinear_0")
  %3 = "onnx.MaxPoolSingleOut"(%2) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [5, 5], pads = [1, 1, 1, 1], storage_order = 0 : si64, strides = [1, 1]} : (tensor<1x3x224x224xf32>) -> tensor<1x3x222x222xf32> loc("MaxPool_upper")
  %4 = "onnx.QuantizeLinear"(%3, %0, %1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x3x222x222xf32>, tensor<f32>, tensor<i8>) -> tensor<1x3x222x222xi8> loc("QuantizeLinear")
  %5 = "onnx.DequantizeLinear"(%4, %0, %1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x3x222x222xi8>, tensor<f32>, tensor<i8>) -> tensor<1x3x222x222xf32> loc("DequantizeLinear_1")
  %6 = "onnx.MaxPoolSingleOut"(%5) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], storage_order = 0 : si64, strides = [1, 1]} : (tensor<1x3x222x222xf32>) -> tensor<1x3x222x222xf32> loc("MaxPool_lower")
  %7 = "onnx.QuantizeLinear"(%6, %0, %1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x3x222x222xf32>, tensor<f32>, tensor<i8>) -> tensor<1x3x222x222xi8> loc("QuantizeLinear_out")
  return %7 : tensor<1x3x222x222xi8>
}

// CHECK-LABEL: func.func @maxpool_k5_p1_s1_maxpool_k3_p1_s1_quant_int8
// CHECK: "onnx.MaxPoolSingleOut"
// CHECK-SAME: loc([[LOC_FUSED:#.+]])

// CHECK-DAG: [[LOC_MU:#.+]] = loc("MaxPool_upper")
// CHECK-DAG: [[LOC_QL:#.+]] = loc("QuantizeLinear")
// CHECK-DAG: [[LOC_DQ1:#.+]] = loc("DequantizeLinear_1")
// CHECK-DAG: [[LOC_ML:#.+]] = loc("MaxPool_lower")
// CHECK-DAG: [[LOC_FUSED]] = loc(fused[[[LOC_MU]], [[LOC_ML]], [[LOC_QL]], [[LOC_DQ1]]])

// -----

// Verify that `FuseAddConvQDQZeroBiasPattern` produces a `fused` location on
// the rewritten Conv
func.func @test_fuse_add_conv_qdq_zero_bias_loc(%arg0 : tensor<1x3x4x4xf32>, %arg1 : tensor<3x3x1x1xf32>) -> tensor<1x3x4x4xf32> {
    %bias = onnx.Constant dense<0.000000e+00> : tensor<3xf32> loc("BiasConst")
    %scale = onnx.Constant dense<5.000000e-01> : tensor<f32>
    %zp = onnx.Constant dense<0> : tensor<i8>
    %0 = "onnx.Conv"(%arg0, %arg1, %bias) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], strides = [1, 1]} : (tensor<1x3x4x4xf32>, tensor<3x3x1x1xf32>, tensor<3xf32>) -> tensor<1x3x4x4xf32> loc("MyConv")
    %1 = "onnx.QuantizeLinear"(%0, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x3x4x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x3x4x4xi8> loc("Q")
    %2 = "onnx.DequantizeLinear"(%1, %scale, %zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x3x4x4xi8>, tensor<f32>, tensor<i8>) -> tensor<1x3x4x4xf32> loc("DQ")
    %addend_q = onnx.Constant dense<48> : tensor<1x1x1x1xi8>
    %addend_scale = onnx.Constant dense<6.250000e-02> : tensor<f32>
    %addend_zp = onnx.Constant dense<0> : tensor<i8>
    %addend = "onnx.DequantizeLinear"(%addend_q, %addend_scale, %addend_zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x1xi8>, tensor<f32>, tensor<i8>) -> tensor<1x1x1x1xf32> loc("AddendDQ")
    %3 = "onnx.Add"(%2, %addend) : (tensor<1x3x4x4xf32>, tensor<1x1x1x1xf32>) -> tensor<1x3x4x4xf32> loc("MyAdd")
    onnx.Return %3 : tensor<1x3x4x4xf32>
}
// CHECK-LABEL: func.func @test_fuse_add_conv_qdq_zero_bias_loc
// CHECK:       "onnx.Conv"
// CHECK-SAME:  loc([[LOC_FUSED:#.+]])
// CHECK-DAG:   [[LOC_MY_ADD:#.+]] = loc("MyAdd")
// CHECK-DAG:   [[LOC_MY_CONV:#.+]] = loc("MyConv")
// CHECK-DAG:   [[LOC_Q:#.+]] = loc("Q")
// CHECK-DAG:   [[LOC_DQ:#.+]] = loc("DQ")
// CHECK-DAG:   [[LOC_ADDEND_DQ:#.+]] = loc("AddendDQ")
// CHECK-DAG:   [[LOC_FUSED]] = loc(fused[[[LOC_MY_ADD]], [[LOC_MY_CONV]], [[LOC_Q]], [[LOC_DQ]], [[LOC_ADDEND_DQ]]])