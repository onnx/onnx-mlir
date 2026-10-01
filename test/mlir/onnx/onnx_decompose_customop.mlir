// Modifications (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
// RUN: onnx-mlir-opt --decompose-onnx=enable-gqa-uint16-cache-slot-rewrite=true %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --decompose-onnx=enable-gqa-uint16-cache-slot-rewrite=false %s -split-input-file | FileCheck %s --check-prefix=SCATTER

// COM: Decompose CustomOp introduced by onnxruntime.

func.func @customop_fusedmatmul_onnxruntime(%arg0: tensor<3x5x7x9xf32>, %arg1:tensor<3x5x7x9xf32>) -> tensor<3x5x9x9xf32> {
    %0 = "onnx.Custom"(%arg0, %arg1) {alpha = 1.250000e-01 : f32, domain_name = "com.microsoft", function_name = "FusedMatMul", transA = 1 : si64, transB = 0 : si64} : (tensor<3x5x7x9xf32>, tensor<3x5x7x9xf32>) -> tensor<3x5x9x9xf32>
    onnx.Return %0: tensor<3x5x9x9xf32>

// CHECK-LABEL:  func.func @customop_fusedmatmul_onnxruntime
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x5x7x9xf32>, [[PARAM_1_:%.+]]: tensor<3x5x7x9xf32>) -> tensor<3x5x9x9xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Transpose"([[PARAM_0_]]) {perm = [0, 1, 3, 2]} : (tensor<3x5x7x9xf32>) -> tensor<3x5x9x7xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1.250000e-01> : tensor<1xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.MatMul"([[VAR_0_]], [[PARAM_1_]]) : (tensor<3x5x9x7xf32>, tensor<3x5x7x9xf32>) -> tensor<3x5x9x9xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Mul"([[VAR_2_]], [[VAR_1_]]) : (tensor<3x5x9x9xf32>, tensor<1xf32>) -> tensor<3x5x9x9xf32>
// CHECK:           onnx.Return [[VAR_3_]] : tensor<3x5x9x9xf32>
// CHECK:         }
}

// -----

func.func @customop_fusedmatmul_onnxruntime_no_transpose(%arg0: tensor<*xf32>, %arg1:tensor<*xf32>) -> tensor<*xf32> {
    %0 = "onnx.Custom"(%arg0, %arg1) {alpha = 1.250000e-01 : f32, domain_name = "com.microsoft", function_name = "FusedMatMul", transA = 0 : si64, transB = 0 : si64} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    onnx.Return %0: tensor<*xf32>

// CHECK-LABEL:  func.func @customop_fusedmatmul_onnxruntime_no_transpose
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<*xf32>, [[PARAM_1_:%.+]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.250000e-01> : tensor<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.MatMul"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Mul"([[VAR_1_]], [[VAR_0_]]) : (tensor<*xf32>, tensor<1xf32>) -> tensor<*xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<*xf32>
// CHECK:         }
}

// -----

func.func @customop_fusedmatmul_onnxruntime_transA(%arg0: tensor<*xf32>, %arg1:tensor<*xf32>) -> tensor<*xf32> {
    %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1, 3]} : (tensor<*xf32>) -> tensor<*xf32>
    %1 = "onnx.Custom"(%0, %arg1) {alpha = 1.250000e-01 : f32, domain_name = "com.microsoft", function_name = "FusedMatMul", transA = 1 : si64, transB = 0 : si64} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    onnx.Return %1: tensor<*xf32>

// CHECK-LABEL:  func.func @customop_fusedmatmul_onnxruntime_transA
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<*xf32>, [[PARAM_1_:%.+]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.250000e-01> : tensor<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Transpose"([[PARAM_0_]]) {perm = [0, 2, 1, 3]} : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Transpose"([[VAR_1_]]) {perm = [0, 1, 3, 2]} : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.MatMul"([[VAR_2_]], [[PARAM_1_]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Mul"([[VAR_3_]], [[VAR_0_]]) : (tensor<*xf32>, tensor<1xf32>) -> tensor<*xf32>
// CHECK:           onnx.Return [[VAR_4_]] : tensor<*xf32>
}

// -----

func.func @customop_fusedmatmul_onnxruntime_transB(%arg0: tensor<*xf32>, %arg1:tensor<*xf32>) -> tensor<*xf32> {
    %0 = "onnx.Transpose"(%arg1) {perm = [0, 2, 1, 3]} : (tensor<*xf32>) -> tensor<*xf32>
    %1 = "onnx.Custom"(%arg0, %0) {alpha = 1.250000e-01 : f32, domain_name = "com.microsoft", function_name = "FusedMatMul", transA = 0 : si64, transB = 1 : si64} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    onnx.Return %1: tensor<*xf32>

// CHECK-LABEL:  func.func @customop_fusedmatmul_onnxruntime_transB
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<*xf32>, [[PARAM_1_:%.+]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.250000e-01> : tensor<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [0, 2, 1, 3]} : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Transpose"([[VAR_1_]]) {perm = [0, 1, 3, 2]} : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.MatMul"([[PARAM_0_]], [[VAR_2_]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Mul"([[VAR_3_]], [[VAR_0_]]) : (tensor<*xf32>, tensor<1xf32>) -> tensor<*xf32>
// CHECK:           onnx.Return [[VAR_4_]] : tensor<*xf32>
}

// -----

// COM: Do not rewrite because the domain_name is not "com.microsoft"
func.func @customop_fusedmatmul_not_rewrite_domain(%arg0: tensor<*xf32>, %arg1:tensor<*xf32>) -> tensor<*xf32> {
    %0 = "onnx.Transpose"(%arg1) {perm = [0, 2, 1, 3]} : (tensor<*xf32>) -> tensor<*xf32>
    %1 = "onnx.Custom"(%arg0, %0) {alpha = 1.250000e-01 : f32, domain_name = "abc.xyz", function_name = "FusedMatMul", transA = 0 : si64, transB = 1 : si64} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    onnx.Return %1: tensor<*xf32>

// CHECK-LABEL:  func.func @customop_fusedmatmul_not_rewrite_domain
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<*xf32>, [[PARAM_1_:%.+]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [0, 2, 1, 3]} : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Custom"([[PARAM_0_]], [[VAR_0_]]) {alpha = 1.250000e-01 : f32, domain_name = "abc.xyz", function_name = "FusedMatMul", transA = 0 : si64, transB = 1 : si64} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
// CHECK:           onnx.Return [[VAR_1_]] : tensor<*xf32>
// CHECK:         }
}

// -----

// COM: Do not rewrite because A is transposed but its rank is unknown.
// COM: So, there is no information to generate a transpose op.
func.func @customop_fusedmatmul_not_rewrite_unranked_transpose(%arg0: tensor<*xf32>, %arg1:tensor<*xf32>) -> tensor<*xf32> {
    %1 = "onnx.Custom"(%arg0, %arg1) {alpha = 1.250000e-01 : f32, domain_name = "com.microsoft", function_name = "FusedMatMul", transA = 1 : si64, transB = 0 : si64} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    onnx.Return %1: tensor<*xf32>

// CHECK-LABEL:  func.func @customop_fusedmatmul_not_rewrite_unranked_transpose
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<*xf32>, [[PARAM_1_:%.+]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Custom"([[PARAM_0_]], [[PARAM_1_]]) {alpha = 1.250000e-01 : f32, domain_name = "com.microsoft", function_name = "FusedMatMul", transA = 1 : si64, transB = 0 : si64} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<*xf32>
// CHECK:         }
}

// -----

// COM: Do not rewrite because alpha is not given.
func.func @customop_fusedmatmul_not_rewrite_no_alpha(%arg0: tensor<*xf32>, %arg1:tensor<*xf32>) -> tensor<*xf32> {
    %1 = "onnx.Custom"(%arg0, %arg1) {domain_name = "com.microsoft", function_name = "FusedMatMul", transA = 0 : si64, transB = 0 : si64} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    onnx.Return %1: tensor<*xf32>

// CHECK-LABEL:  func.func @customop_fusedmatmul_not_rewrite_no_alpha
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<*xf32>, [[PARAM_1_:%.+]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Custom"([[PARAM_0_]], [[PARAM_1_]]) {domain_name = "com.microsoft", function_name = "FusedMatMul", transA = 0 : si64, transB = 0 : si64} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<*xf32>
// CHECK:         }
}

// -----


func.func @customop_quantize(%arg0: tensor<*xf32>, %arg1: tensor<f32>, %arg2: tensor<ui16>) -> tensor<*xui16> {
    %1 = "onnx.Custom"(%arg0, %arg1, %arg2) {domain_name = "com.microsoft", function_name = "QuantizeLinear"} : (tensor<*xf32>, tensor<f32>, tensor<ui16>) -> tensor<*xui16>
    onnx.Return %1: tensor<*xui16>

// CHECK-LABEL:  func.func @customop_quantize
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<*xf32>, [[PARAM_1_:%.+]]: tensor<f32>, [[PARAM_2_:%.+]]: tensor<ui16>) -> tensor<*xui16> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.QuantizeLinear"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<*xf32>, tensor<f32>, tensor<ui16>) -> tensor<*xui16>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<*xui16>
// CHECK:         }
}

// -----

func.func @customop_quantize_axis(%arg0: tensor<*xf32>, %arg1: tensor<5xf32>, %arg2: tensor<5xui16>) -> tensor<*xui16> {
    %1 = "onnx.Custom"(%arg0, %arg1, %arg2) {domain_name = "com.microsoft", function_name = "QuantizeLinear"} : (tensor<*xf32>, tensor<5xf32>, tensor<5xui16>) -> tensor<*xui16>
    onnx.Return %1: tensor<*xui16>

// CHECK-LABEL:  func.func @customop_quantize_axis
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<*xf32>, [[PARAM_1_:%.+]]: tensor<5xf32>, [[PARAM_2_:%.+]]: tensor<5xui16>) -> tensor<*xui16> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.QuantizeLinear"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<*xf32>, tensor<5xf32>, tensor<5xui16>) -> tensor<*xui16>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<*xui16>
// CHECK:         }
}

// -----


func.func @customop_dequantize(%arg0: tensor<*xui16>, %arg1: tensor<f32>, %arg2: tensor<ui16>) -> tensor<*xf32> {
    %1 = "onnx.Custom"(%arg0, %arg1, %arg2) {domain_name = "com.microsoft", function_name = "DequantizeLinear"} : (tensor<*xui16>, tensor<f32>, tensor<ui16>) -> tensor<*xf32>
    onnx.Return %1: tensor<*xf32>

// CHECK-LABEL:  func.func @customop_dequantize
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<*xui16>, [[PARAM_1_:%.+]]: tensor<f32>, [[PARAM_2_:%.+]]: tensor<ui16>) -> tensor<*xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.DequantizeLinear"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<*xui16>, tensor<f32>, tensor<ui16>) -> tensor<*xf32>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<*xf32>
// CHECK:         }
}

// -----

func.func @customop_dequantize_axis(%arg0: tensor<*xui16>, %arg1: tensor<5xf32>, %arg2: tensor<5xui16>) -> tensor<*xf32> {
    %1 = "onnx.Custom"(%arg0, %arg1, %arg2) {domain_name = "com.microsoft", function_name = "DequantizeLinear"} : (tensor<*xui16>, tensor<5xf32>, tensor<5xui16>) -> tensor<*xf32>
    onnx.Return %1: tensor<*xf32>

// CHECK-LABEL:  func.func @customop_dequantize_axis
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<*xui16>, [[PARAM_1_:%.+]]: tensor<5xf32>, [[PARAM_2_:%.+]]: tensor<5xui16>) -> tensor<*xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.DequantizeLinear"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) {axis = 1 : si64, block_size = 0 : si64} : (tensor<*xui16>, tensor<5xf32>, tensor<5xui16>) -> tensor<*xf32>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<*xf32>
// CHECK:         }
}

// -----

func.func @customop_bias_gelu(%arg0: tensor<*xf32>, %arg1: tensor<5xf32>) -> tensor<*xf32> {
    %1 = "onnx.Custom"(%arg0, %arg1) {domain_name = "com.microsoft", function_name = "BiasGelu"} : (tensor<*xf32>, tensor<5xf32>) -> tensor<*xf32>
    onnx.Return %1: tensor<*xf32>

// CHECK-LABEL:  func.func @customop_bias_gelu
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<*xf32>, [[PARAM_1_:%.+]]: tensor<5xf32>) -> tensor<*xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<*xf32>, tensor<5xf32>) -> tensor<*xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Gelu"([[VAR_0_]]) {approximate = "none"} : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:           onnx.Return [[VAR_1_]] : tensor<*xf32>
// CHECK:         }
}

// -----

func.func @fusedconv_relu_no_bias(%x: tensor<1x3x8x8xf32>, %w: tensor<4x3x3x3xf32>) -> tensor<1x4x8x8xf32> {
  %res = "onnx.Custom"(%x, %w) {function_name = "FusedConv", domain_name = "com.microsoft",
                                activation = "Relu", activation_params = [],
                                dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3],
                                pads = [1, 1, 1, 1], strides = [1, 1]} :
          (tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>) -> tensor<1x4x8x8xf32>
  onnx.Return %res : tensor<1x4x8x8xf32>
// CHECK-LABEL:  func.func @fusedconv_relu_no_bias
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x8x8xf32>, [[PARAM_1_:%.+]]: tensor<4x3x3x3xf32>) -> tensor<1x4x8x8xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_1_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[PARAM_1_]], [[VAR_0_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>, none) -> tensor<1x4x8x8xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Relu"([[VAR_1_]]) : (tensor<1x4x8x8xf32>) -> tensor<1x4x8x8xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<1x4x8x8xf32>
// CHECK:         }
}

// -----

func.func @fusedconv_relu_bias(%x: tensor<1x3x8x8xf32>, %w: tensor<4x3x3x3xf32>, %b: tensor<4xf32>) -> tensor<1x4x8x8xf32> {
  %res = "onnx.Custom"(%x, %w, %b) {function_name = "FusedConv", domain_name = "com.microsoft",
                                    activation = "Relu", activation_params = [],
                                    dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3],
                                    pads = [1, 1, 1, 1], strides = [1, 1]} :
          (tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>, tensor<4xf32>) -> tensor<1x4x8x8xf32>
  onnx.Return %res : tensor<1x4x8x8xf32>
// CHECK-LABEL:  func.func @fusedconv_relu_bias
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x8x8xf32>, [[PARAM_1_:%.+]]: tensor<4x3x3x3xf32>, [[PARAM_2_:%.+]]: tensor<4xf32>) -> tensor<1x4x8x8xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>, tensor<4xf32>) -> tensor<1x4x8x8xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Relu"([[VAR_0_]]) : (tensor<1x4x8x8xf32>) -> tensor<1x4x8x8xf32>
// CHECK:           onnx.Return [[VAR_1_]] : tensor<1x4x8x8xf32>
// CHECK:         }
}

// -----

func.func @fusedconv_tanh(%x: tensor<1x3x8x8xf32>, %w: tensor<4x3x3x3xf32>) -> tensor<1x4x8x8xf32> {
  %res = "onnx.Custom"(%x, %w) {function_name = "FusedConv", domain_name = "com.microsoft",
                                activation = "Tanh", activation_params = [],
                                dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3],
                                pads = [1, 1, 1, 1], strides = [1, 1]} :
          (tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>) -> tensor<1x4x8x8xf32>
  onnx.Return %res : tensor<1x4x8x8xf32>
// CHECK-LABEL:  func.func @fusedconv_tanh
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x8x8xf32>, [[PARAM_1_:%.+]]: tensor<4x3x3x3xf32>) -> tensor<1x4x8x8xf32> {
// CHECK:           [[VAR_1_:%.+]] = "onnx.Conv"
// CHECK:           [[VAR_2_:%.+]] = "onnx.Tanh"([[VAR_1_]]) : (tensor<1x4x8x8xf32>) -> tensor<1x4x8x8xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<1x4x8x8xf32>
// CHECK:         }
}

// -----
func.func @fusedconv_sigmoid(%x: tensor<1x3x8x8xf32>, %w: tensor<4x3x3x3xf32>) -> tensor<1x4x8x8xf32> {
  %res = "onnx.Custom"(%x, %w) {function_name = "FusedConv", domain_name = "com.microsoft",
                                activation = "Sigmoid", activation_params = [],
                                dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3],
                                pads = [1, 1, 1, 1], strides = [1, 1]} :
          (tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>) -> tensor<1x4x8x8xf32>
  onnx.Return %res : tensor<1x4x8x8xf32>
// CHECK-LABEL:  func.func @fusedconv_sigmoid
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x8x8xf32>, [[PARAM_1_:%.+]]: tensor<4x3x3x3xf32>) -> tensor<1x4x8x8xf32> {
// CHECK:           [[VAR_1_:%.+]] = "onnx.Conv"
// CHECK:           [[VAR_2_:%.+]] = "onnx.Sigmoid"([[VAR_1_]]) : (tensor<1x4x8x8xf32>) -> tensor<1x4x8x8xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<1x4x8x8xf32>
// CHECK:         }
}

// -----
func.func @fusedconv_leakyrelu(%x: tensor<1x3x8x8xf32>, %w: tensor<4x3x3x3xf32>) -> tensor<1x4x8x8xf32> {
  %res = "onnx.Custom"(%x, %w) {function_name = "FusedConv", domain_name = "com.microsoft",
                                activation = "LeakyRelu",
                                activation_params = [5.000000e-01 : f32],
                                dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3],
                                pads = [1, 1, 1, 1], strides = [1, 1]} :
          (tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>) -> tensor<1x4x8x8xf32>
  onnx.Return %res : tensor<1x4x8x8xf32>
// CHECK:         }
// CHECK-LABEL:  func.func @fusedconv_leakyrelu
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x8x8xf32>, [[PARAM_1_:%.+]]: tensor<4x3x3x3xf32>) -> tensor<1x4x8x8xf32> {
// CHECK:           [[VAR_1_:%.+]] = "onnx.Conv"
// CHECK:           [[VAR_2_:%.+]] = "onnx.LeakyRelu"([[VAR_1_]]) {alpha = 5.000000e-01 : f32} : (tensor<1x4x8x8xf32>) -> tensor<1x4x8x8xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<1x4x8x8xf32>
// CHECK:         }
}

// -----

func.func @fusedconv_clip(%x: tensor<1x3x8x8xf32>, %w: tensor<4x3x3x3xf32>) -> tensor<1x4x8x8xf32> {
  %res = "onnx.Custom"(%x, %w) {function_name = "FusedConv", domain_name = "com.microsoft",
                                activation = "Clip",
                                activation_params = [0.000000e+00 : f32, 1.000000e+00 : f32],
                                dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3],
                                pads = [1, 1, 1, 1], strides = [1, 1]} :
          (tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>) -> tensor<1x4x8x8xf32>
  onnx.Return %res : tensor<1x4x8x8xf32>
// CHECK-LABEL:  func.func @fusedconv_clip
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x8x8xf32>, [[PARAM_1_:%.+]]: tensor<4x3x3x3xf32>) -> tensor<1x4x8x8xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Conv"
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Cast"([[VAR_1_]]) {saturate = 1 : si64, to = f32} : (tensor<f32>) -> tensor<f32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Cast"([[VAR_0_]]) {saturate = 1 : si64, to = f32} : (tensor<f32>) -> tensor<f32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Clip"([[VAR_3_]], [[VAR_4_]], [[VAR_5_]]) : (tensor<1x4x8x8xf32>, tensor<f32>, tensor<f32>) -> tensor<1x4x8x8xf32>
// CHECK:           onnx.Return [[VAR_6_]] : tensor<1x4x8x8xf32>
// CHECK:         }
}

// -----
func.func @fusedconv_clip_not_f32(%x: tensor<1x3x8x8xf16>, %w: tensor<4x3x3x3xf16>) -> tensor<1x4x8x8xf16> {
  %res = "onnx.Custom"(%x, %w) {function_name = "FusedConv", domain_name = "com.microsoft",
                                activation = "Clip",
                                activation_params = [0.000000e+00 : f32, 1.000000e+00 : f32],
                                dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3],
                                pads = [1, 1, 1, 1], strides = [1, 1]} :
          (tensor<1x3x8x8xf16>, tensor<4x3x3x3xf16>) -> tensor<1x4x8x8xf16>
  onnx.Return %res : tensor<1x4x8x8xf16>
// CHECK-LABEL:  func.func @fusedconv_clip_not_f32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x8x8xf16>, [[PARAM_1_:%.+]]: tensor<4x3x3x3xf16>) -> tensor<1x4x8x8xf16> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Conv"
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Cast"([[VAR_1_]]) {saturate = 1 : si64, to = f16} : (tensor<f32>) -> tensor<f16>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Cast"([[VAR_0_]]) {saturate = 1 : si64, to = f16} : (tensor<f32>) -> tensor<f16>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Clip"([[VAR_3_]], [[VAR_4_]], [[VAR_5_]]) : (tensor<1x4x8x8xf16>, tensor<f16>, tensor<f16>) -> tensor<1x4x8x8xf16>
// CHECK:           onnx.Return [[VAR_6_]] : tensor<1x4x8x8xf16>
// CHECK:         }
}

// -----

func.func @fusedconv_hardsigmoid(%x: tensor<1x3x8x8xf32>, %w: tensor<4x3x3x3xf32>) -> tensor<1x4x8x8xf32> {
  %res = "onnx.Custom"(%x, %w) {function_name = "FusedConv", domain_name = "com.microsoft",
                                activation = "HardSigmoid",
                                activation_params = [2.000000e-01 : f32, 5.000000e-01 : f32],
                                dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3],
                                pads = [1, 1, 1, 1], strides = [1, 1]} :
          (tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>) -> tensor<1x4x8x8xf32>
  onnx.Return %res : tensor<1x4x8x8xf32>
// CHECK-LABEL:  func.func @fusedconv_hardsigmoid
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x8x8xf32>, [[PARAM_1_:%.+]]: tensor<4x3x3x3xf32>) -> tensor<1x4x8x8xf32> {
// CHECK:           [[VAR_1_:%.+]] = "onnx.Conv"
// CHECK:           [[VAR_2_:%.+]] = "onnx.HardSigmoid"([[VAR_1_]]) {alpha = 2.000000e-01 : f32, beta = 5.000000e-01 : f32} : (tensor<1x4x8x8xf32>) -> tensor<1x4x8x8xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<1x4x8x8xf32>
// CHECK:         }
}

// -----

func.func @fusedconv_unsupported_activation(%x: tensor<1x3x8x8xf32>, %w: tensor<4x3x3x3xf32>) -> tensor<1x4x8x8xf32> {
  %res = "onnx.Custom"(%x, %w) {function_name = "FusedConv", domain_name = "com.microsoft",
                                activation = "Softplus", activation_params = [],
                                dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3],
                                pads = [1, 1, 1, 1], strides = [1, 1]} :
          (tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>) -> tensor<1x4x8x8xf32>
  onnx.Return %res : tensor<1x4x8x8xf32>
// CHECK-LABEL:  func.func @fusedconv_unsupported_activation
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x8x8xf32>, [[PARAM_1_:%.+]]: tensor<4x3x3x3xf32>) -> tensor<1x4x8x8xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Custom"([[PARAM_0_]], [[PARAM_1_]]) {activation = "Softplus", activation_params = [], dilations = [1, 1], domain_name = "com.microsoft", function_name = "FusedConv", group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>) -> tensor<1x4x8x8xf32>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<1x4x8x8xf32>
// CHECK:         }
}


// -----
// Too many operands Z/Sum
func.func @fusedconv_too_many_operands(%x: tensor<1x3x8x8xf32>, %w: tensor<4x3x3x3xf32>, %b: tensor<4xf32>, %z: tensor<1x4x8x8xf32>) -> tensor<1x4x8x8xf32> {
  %res = "onnx.Custom"(%x, %w, %b, %z) {function_name = "FusedConv", domain_name = "com.microsoft",
                                        activation = "Relu", activation_params = [],
                                        dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3],
                                        pads = [1, 1, 1, 1], strides = [1, 1]} :
          (tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>, tensor<4xf32>, tensor<1x4x8x8xf32>) -> tensor<1x4x8x8xf32>
  onnx.Return %res : tensor<1x4x8x8xf32>
// CHECK-LABEL:  func.func @fusedconv_too_many_operands
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x8x8xf32>, [[PARAM_1_:%.+]]: tensor<4x3x3x3xf32>, [[PARAM_2_:%.+]]: tensor<4xf32>, [[PARAM_3_:%.+]]: tensor<1x4x8x8xf32>) -> tensor<1x4x8x8xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Custom"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[PARAM_3_]]) {activation = "Relu", activation_params = [], dilations = [1, 1], domain_name = "com.microsoft", function_name = "FusedConv", group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x3x8x8xf32>, tensor<4x3x3x3xf32>, tensor<4xf32>, tensor<1x4x8x8xf32>) -> tensor<1x4x8x8xf32>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<1x4x8x8xf32>
// CHECK:         }

}

// -----
// SkipLayerNormalization: 3 inputs, 1 output

func.func @skip_layernorm_basic(%input: tensor<2x4x8xf32>, %skip: tensor<2x4x8xf32>, %gamma: tensor<8xf32>) -> tensor<2x4x8xf32> {
  %r = "onnx.Custom"(%input, %skip, %gamma) {domain_name = "com.microsoft", function_name = "SkipLayerNormalization", epsilon = 1.000000e-05 : f32} : (tensor<2x4x8xf32>, tensor<2x4x8xf32>, tensor<8xf32>) -> tensor<2x4x8xf32>
  onnx.Return %r : tensor<2x4x8xf32>
// CHECK-LABEL:  func.func @skip_layernorm_basic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x8xf32>, [[PARAM_1_:%.+]]: tensor<2x4x8xf32>, [[PARAM_2_:%.+]]: tensor<8xf32>) -> tensor<2x4x8xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
// CHECK:           [[VAR_Y_:%.+]], [[VAR_Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[VAR_1_]], [[PARAM_2_]], [[VAR_0_]]) {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<2x4x8xf32>, tensor<8xf32>, none) -> (tensor<2x4x8xf32>, none, none)
// CHECK:           onnx.Return [[VAR_Y_]] : tensor<2x4x8xf32>
}




// -----
// SkipLayerNormalization: 4 inputs (beta), 1 output

func.func @skip_layernorm_beta_no_eps(%input: tensor<2x4x8xf32>, %skip: tensor<2x4x8xf32>, %gamma: tensor<8xf32>, %beta: tensor<8xf32>) -> tensor<2x4x8xf32> {
  %r = "onnx.Custom"(%input, %skip, %gamma, %beta) {domain_name = "com.microsoft", function_name = "SkipLayerNormalization"} : (tensor<2x4x8xf32>, tensor<2x4x8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<2x4x8xf32>
  onnx.Return %r : tensor<2x4x8xf32>
// CHECK-LABEL:  func.func @skip_layernorm_beta_no_eps
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x8xf32>, [[PARAM_1_:%.+]]: tensor<2x4x8xf32>, [[PARAM_2_:%.+]]: tensor<8xf32>, [[PARAM_3_:%.+]]: tensor<8xf32>) -> tensor<2x4x8xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
// CHECK:           [[VAR_Y_:%.+]], [[VAR_Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[VAR_0_]], [[PARAM_2_]], [[PARAM_3_]]) {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<2x4x8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<2x4x8xf32>, none, none)
// CHECK:           onnx.Return [[VAR_Y_]] : tensor<2x4x8xf32>
}


// -----
// SkipLayerNormalization: 5 inputs (beta + bias), 1 output

func.func @skip_layernorm_beta_bias(%input: tensor<2x4x8xf32>, %skip: tensor<2x4x8xf32>, %gamma: tensor<8xf32>, %beta: tensor<8xf32>, %bias: tensor<8xf32>) -> tensor<2x4x8xf32> {
  %r = "onnx.Custom"(%input, %skip, %gamma, %beta, %bias) {domain_name = "com.microsoft", function_name = "SkipLayerNormalization", epsilon = 1.000000e-05 : f32} : (tensor<2x4x8xf32>, tensor<2x4x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<2x4x8xf32>
  onnx.Return %r : tensor<2x4x8xf32>
// CHECK-LABEL:  func.func @skip_layernorm_beta_bias
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x8xf32>, [[PARAM_1_:%.+]]: tensor<2x4x8xf32>, [[PARAM_2_:%.+]]: tensor<8xf32>, [[PARAM_3_:%.+]]: tensor<8xf32>, [[PARAM_4_:%.+]]: tensor<8xf32>) -> tensor<2x4x8xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Add"([[VAR_0_]], [[PARAM_4_]]) : (tensor<2x4x8xf32>, tensor<8xf32>) -> tensor<2x4x8xf32>
// CHECK:           [[VAR_Y_:%.+]], [[VAR_Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[VAR_1_]], [[PARAM_2_]], [[PARAM_3_]]) {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<2x4x8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<2x4x8xf32>, none, none)
// CHECK:           onnx.Return [[VAR_Y_]] : tensor<2x4x8xf32>
}


// -----
// SkipLayerNormalization: 5 inputs, 2 outputs (output, mean)

func.func @skip_layernorm_two_outputs(%input: tensor<2x4x8xf32>, %skip: tensor<2x4x8xf32>, %gamma: tensor<8xf32>, %beta: tensor<8xf32>, %bias: tensor<8xf32>) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>) {
  %r0, %r1 = "onnx.Custom"(%input, %skip, %gamma, %beta, %bias) {domain_name = "com.microsoft", function_name = "SkipLayerNormalization", epsilon = 1.000000e-05 : f32} : (tensor<2x4x8xf32>, tensor<2x4x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>)
  onnx.Return %r0, %r1 : tensor<2x4x8xf32>, tensor<2x4x1xf32>
// CHECK-LABEL:  func.func @skip_layernorm_two_outputs
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x8xf32>, [[PARAM_1_:%.+]]: tensor<2x4x8xf32>, [[PARAM_2_:%.+]]: tensor<8xf32>, [[PARAM_3_:%.+]]: tensor<8xf32>, [[PARAM_4_:%.+]]: tensor<8xf32>) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>) {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Add"([[VAR_0_]], [[PARAM_4_]]) : (tensor<2x4x8xf32>, tensor<8xf32>) -> tensor<2x4x8xf32>
// CHECK:           [[VAR_Y_:%.+]], [[VAR_Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[VAR_1_]], [[PARAM_2_]], [[PARAM_3_]]) {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<2x4x8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>, none)
// CHECK:           onnx.Return [[VAR_Y_]], [[VAR_Mean_]] : tensor<2x4x8xf32>, tensor<2x4x1xf32>
}


// -----
// SkipLayerNormalization: 5 inputs, 3 outputs (output, mean, inv_std_var)

func.func @skip_layernorm_three_outputs(%input: tensor<2x4x8xf32>, %skip: tensor<2x4x8xf32>, %gamma: tensor<8xf32>, %beta: tensor<8xf32>, %bias: tensor<8xf32>) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>, tensor<2x4x1xf32>) {
  %r0, %r1, %r2 = "onnx.Custom"(%input, %skip, %gamma, %beta, %bias) {domain_name = "com.microsoft", function_name = "SkipLayerNormalization", epsilon = 1.000000e-05 : f32} : (tensor<2x4x8xf32>, tensor<2x4x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>, tensor<2x4x1xf32>)
  onnx.Return %r0, %r1, %r2 : tensor<2x4x8xf32>, tensor<2x4x1xf32>, tensor<2x4x1xf32>
// CHECK-LABEL:  func.func @skip_layernorm_three_outputs
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x8xf32>, [[PARAM_1_:%.+]]: tensor<2x4x8xf32>, [[PARAM_2_:%.+]]: tensor<8xf32>, [[PARAM_3_:%.+]]: tensor<8xf32>, [[PARAM_4_:%.+]]: tensor<8xf32>) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>, tensor<2x4x1xf32>) {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Add"([[VAR_0_]], [[PARAM_4_]]) : (tensor<2x4x8xf32>, tensor<8xf32>) -> tensor<2x4x8xf32>
// CHECK:           [[VAR_Y_:%.+]], [[VAR_Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[VAR_1_]], [[PARAM_2_]], [[PARAM_3_]]) {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<2x4x8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>, tensor<2x4x1xf32>)
// CHECK:           onnx.Return [[VAR_Y_]], [[VAR_Mean_]], [[VAR_InvStdDev_]] : tensor<2x4x8xf32>, tensor<2x4x1xf32>, tensor<2x4x1xf32>
}


// -----
// SkipLayerNormalization: 5 inputs, 4 outputs (output, mean, inv_std_var, sum)

func.func @skip_layernorm_four_outputs(%input: tensor<2x4x8xf32>, %skip: tensor<2x4x8xf32>, %gamma: tensor<8xf32>, %beta: tensor<8xf32>, %bias: tensor<8xf32>) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>, tensor<2x4x1xf32>, tensor<2x4x8xf32>) {
  %r0, %r1, %r2, %r3 = "onnx.Custom"(%input, %skip, %gamma, %beta, %bias) {domain_name = "com.microsoft", function_name = "SkipLayerNormalization", epsilon = 1.000000e-05 : f32} : (tensor<2x4x8xf32>, tensor<2x4x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>, tensor<2x4x1xf32>, tensor<2x4x8xf32>)
  onnx.Return %r0, %r1, %r2, %r3 : tensor<2x4x8xf32>, tensor<2x4x1xf32>, tensor<2x4x1xf32>, tensor<2x4x8xf32>
// CHECK-LABEL:  func.func @skip_layernorm_four_outputs
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x8xf32>, [[PARAM_1_:%.+]]: tensor<2x4x8xf32>, [[PARAM_2_:%.+]]: tensor<8xf32>, [[PARAM_3_:%.+]]: tensor<8xf32>, [[PARAM_4_:%.+]]: tensor<8xf32>) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>, tensor<2x4x1xf32>, tensor<2x4x8xf32>) {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Add"([[VAR_0_]], [[PARAM_4_]]) : (tensor<2x4x8xf32>, tensor<8xf32>) -> tensor<2x4x8xf32>
// CHECK:           [[VAR_Y_:%.+]], [[VAR_Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[VAR_1_]], [[PARAM_2_]], [[PARAM_3_]]) {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<2x4x8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>, tensor<2x4x1xf32>)
// CHECK:           onnx.Return [[VAR_Y_]], [[VAR_Mean_]], [[VAR_InvStdDev_]], [[VAR_1_]] : tensor<2x4x8xf32>, tensor<2x4x1xf32>, tensor<2x4x1xf32>, tensor<2x4x8xf32>
}


// -----
// SimplifiedLayerNormalization: 2 inputs (no bias), 1 output

func.func @simplified_layernorm_basic(%input: tensor<2x4x8xf32>, %scale: tensor<8xf32>) -> tensor<2x4x8xf32> {
  %r = "onnx.Custom"(%input, %scale) {domain_name = "", function_name = "SimplifiedLayerNormalization", epsilon = 1.000000e-05 : f32, axis = -1 : si64, stash_type = 1 : si64} : (tensor<2x4x8xf32>, tensor<8xf32>) -> tensor<2x4x8xf32>
  onnx.Return %r : tensor<2x4x8xf32>
// CHECK-LABEL:  func.func @simplified_layernorm_basic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x8xf32>, [[PARAM_1_:%.+]]: tensor<8xf32>) -> tensor<2x4x8xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_Y_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.RMSLayerNormalization"([[PARAM_0_]], [[PARAM_1_]], [[VAR_0_]]) {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<2x4x8xf32>, tensor<8xf32>, none) -> (tensor<2x4x8xf32>, none)
// CHECK:           onnx.Return [[VAR_Y_]] : tensor<2x4x8xf32>
}

// -----

func.func @simplified_layernorm_no_attrs(%input: tensor<2x4x8xf32>, %scale: tensor<8xf32>) -> tensor<2x4x8xf32> {
  %r = "onnx.Custom"(%input, %scale) {domain_name = "", function_name = "SimplifiedLayerNormalization"} : (tensor<2x4x8xf32>, tensor<8xf32>) -> tensor<2x4x8xf32>
  onnx.Return %r : tensor<2x4x8xf32>
// CHECK-LABEL:  func.func @simplified_layernorm_no_attrs
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x8xf32>, [[PARAM_1_:%.+]]: tensor<8xf32>) -> tensor<2x4x8xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_Y_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.RMSLayerNormalization"([[PARAM_0_]], [[PARAM_1_]], [[VAR_0_]]) {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<2x4x8xf32>, tensor<8xf32>, none) -> (tensor<2x4x8xf32>, none)
// CHECK:           onnx.Return [[VAR_Y_]] : tensor<2x4x8xf32>
}



// -----
// SimplifiedLayerNormalization: 3 inputs (with bias), 1 output

func.func @simplified_layernorm_bias(%input: tensor<2x4x8xf32>, %scale: tensor<8xf32>, %bias: tensor<8xf32>) -> tensor<2x4x8xf32> {
  %r = "onnx.Custom"(%input, %scale, %bias) {domain_name = "", function_name = "SimplifiedLayerNormalization", epsilon = 1.000000e-05 : f32, axis = -1 : si64, stash_type = 1 : si64} : (tensor<2x4x8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<2x4x8xf32>
  onnx.Return %r : tensor<2x4x8xf32>
// CHECK-LABEL:  func.func @simplified_layernorm_bias
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x8xf32>, [[PARAM_1_:%.+]]: tensor<8xf32>, [[PARAM_2_:%.+]]: tensor<8xf32>) -> tensor<2x4x8xf32> {
// CHECK:           [[VAR_Y_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.RMSLayerNormalization"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<2x4x8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<2x4x8xf32>, none)
// CHECK:           onnx.Return [[VAR_Y_]] : tensor<2x4x8xf32>
}


// -----
// SimplifiedLayerNormalization: 2 inputs, 2 outputs (output, mean) -> mean is unused

func.func @simplified_layernorm_two_outputs_mean_unused(%input: tensor<2x4x8xf32>, %scale: tensor<8xf32>) -> tensor<2x4x8xf32> {
  %r0, %r1 = "onnx.Custom"(%input, %scale) {domain_name = "", function_name = "SimplifiedLayerNormalization", epsilon = 1.000000e-05 : f32, axis = -1 : si64, stash_type = 1 : si64} : (tensor<2x4x8xf32>, tensor<8xf32>) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>)
  onnx.Return %r0 : tensor<2x4x8xf32>
// CHECK-LABEL:  func.func @simplified_layernorm_two_outputs_mean_unused
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x8xf32>, [[PARAM_1_:%.+]]: tensor<8xf32>) -> tensor<2x4x8xf32> {
// CHECK:           [[VAR_0_:%.+]]:2 = "onnx.Custom"([[PARAM_0_]], [[PARAM_1_]]) {axis = -1 : si64, domain_name = "", epsilon = 9.99999974E-6 : f32, function_name = "SimplifiedLayerNormalization", stash_type = 1 : si64} : (tensor<2x4x8xf32>, tensor<8xf32>) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>)
// CHECK:           onnx.Return [[VAR_0_]]#0 : tensor<2x4x8xf32>
}


// -----
// SimplifiedLayerNormalization: 2 inputs, 3 outputs (output, mean, inv_std_var) -> mean is unused

func.func @simplified_layernorm_three_outputs_mean_unused(%input: tensor<2x4x8xf32>, %scale: tensor<8xf32>) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>) {
  %r0, %r1, %r2 = "onnx.Custom"(%input, %scale) {domain_name = "", function_name = "SimplifiedLayerNormalization", epsilon = 1.000000e-05 : f32, axis = -1 : si64, stash_type = 1 : si64} : (tensor<2x4x8xf32>, tensor<8xf32>) -> (tensor<2x4x8xf32>, none, tensor<2x4x1xf32>)
  onnx.Return %r0, %r2 : tensor<2x4x8xf32>, tensor<2x4x1xf32>
// CHECK-LABEL:  func.func @simplified_layernorm_three_outputs_mean_unused
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x8xf32>, [[PARAM_1_:%.+]]: tensor<8xf32>) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>) {
// CHECK:           [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_Y_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.RMSLayerNormalization"([[PARAM_0_]], [[PARAM_1_]], [[VAR_0_]]) {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<2x4x8xf32>, tensor<8xf32>, none) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>)
// CHECK:           onnx.Return [[VAR_Y_]], [[VAR_InvStdDev_]] : tensor<2x4x8xf32>, tensor<2x4x1xf32>
}


// -----
// Negative: SimplifiedLayerNormalization: 2 inputs, 2 outputs (output, mean) -> mean is used

func.func @simplified_layernorm_two_outputs_mean_used(%input: tensor<2x4x8xf32>, %scale: tensor<8xf32>) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>) {
  %r0, %r1 = "onnx.Custom"(%input, %scale) {domain_name = "", function_name = "SimplifiedLayerNormalization", epsilon = 1.000000e-05 : f32, axis = -1 : si64, stash_type = 1 : si64} : (tensor<2x4x8xf32>, tensor<8xf32>) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>)
  onnx.Return %r0, %r1 : tensor<2x4x8xf32>, tensor<2x4x1xf32>
// CHECK-LABEL:  func.func @simplified_layernorm_two_outputs_mean_used
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x8xf32>, [[PARAM_1_:%.+]]: tensor<8xf32>) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>) {
// CHECK:           [[VAR_0_:%.+]]:2 = "onnx.Custom"([[PARAM_0_]], [[PARAM_1_]]) {axis = -1 : si64, domain_name = "", epsilon = 9.99999974E-6 : f32, function_name = "SimplifiedLayerNormalization", stash_type = 1 : si64} : (tensor<2x4x8xf32>, tensor<8xf32>) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>)
// CHECK:           onnx.Return [[VAR_0_]]#0, [[VAR_0_]]#1 : tensor<2x4x8xf32>, tensor<2x4x1xf32>
}

// -----
// SkipSimplifiedLayerNormalization: 3 inputs, 1 output

func.func @skip_simplified_layernorm_basic_no_attr(%input: tensor<2x4x8xf32>, %skip: tensor<2x4x8xf32>, %gamma: tensor<8xf32>) -> tensor<2x4x8xf32> {
  %r = "onnx.Custom"(%input, %skip, %gamma) {domain_name = "com.microsoft", function_name = "SkipSimplifiedLayerNormalization"} : (tensor<2x4x8xf32>, tensor<2x4x8xf32>, tensor<8xf32>) -> tensor<2x4x8xf32>
  onnx.Return %r : tensor<2x4x8xf32>
// CHECK-LABEL:  func.func @skip_simplified_layernorm_basic_no_attr
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x8xf32>, [[PARAM_1_:%.+]]: tensor<2x4x8xf32>, [[PARAM_2_:%.+]]: tensor<8xf32>) -> tensor<2x4x8xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
// CHECK:           [[VAR_Y_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.RMSLayerNormalization"([[VAR_1_]], [[PARAM_2_]], [[VAR_0_]]) {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<2x4x8xf32>, tensor<8xf32>, none) -> (tensor<2x4x8xf32>, none)
// CHECK:           onnx.Return [[VAR_Y_]] : tensor<2x4x8xf32>
}


// -----
// SkipSimplifiedLayerNormalization: 4 inputs (bias), 1 output

func.func @skip_simplified_layernorm_bias(%input: tensor<2x4x8xf32>, %skip: tensor<2x4x8xf32>, %gamma: tensor<8xf32>, %bias: tensor<8xf32>) -> tensor<2x4x8xf32> {
  %r = "onnx.Custom"(%input, %skip, %gamma, %bias) {domain_name = "com.microsoft", function_name = "SkipSimplifiedLayerNormalization", epsilon = 1.000000e-05 : f32} : (tensor<2x4x8xf32>, tensor<2x4x8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<2x4x8xf32>
  onnx.Return %r : tensor<2x4x8xf32>
// CHECK-LABEL:  func.func @skip_simplified_layernorm_bias
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x8xf32>, [[PARAM_1_:%.+]]: tensor<2x4x8xf32>, [[PARAM_2_:%.+]]: tensor<8xf32>, [[PARAM_3_:%.+]]: tensor<8xf32>) -> tensor<2x4x8xf32> {
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Add"([[VAR_1_]], [[PARAM_3_]]) : (tensor<2x4x8xf32>, tensor<8xf32>) -> tensor<2x4x8xf32>
// CHECK:           [[VAR_Y_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.RMSLayerNormalization"([[VAR_2_]], [[PARAM_2_]], [[VAR_0_]]) {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<2x4x8xf32>, tensor<8xf32>, none) -> (tensor<2x4x8xf32>, none)
// CHECK:           onnx.Return [[VAR_Y_]] : tensor<2x4x8xf32>
}



// -----
// SkipSimplifiedLayerNormalization: 4 inputs, 2 outputs (output, mean)

func.func @skip_simplified_layernorm_two_outputs(%input: tensor<2x4x8xf32>, %skip: tensor<2x4x8xf32>, %gamma: tensor<8xf32>, %bias: tensor<8xf32>) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>) {
  %r0, %r1 = "onnx.Custom"(%input, %skip, %gamma, %bias) {domain_name = "com.microsoft", function_name = "SkipSimplifiedLayerNormalization", epsilon = 1.000000e-05 : f32} : (tensor<2x4x8xf32>, tensor<2x4x8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>)
  onnx.Return %r0, %r1 : tensor<2x4x8xf32>, tensor<2x4x1xf32>
// CHECK-LABEL:  func.func @skip_simplified_layernorm_two_outputs
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x8xf32>, [[PARAM_1_:%.+]]: tensor<2x4x8xf32>, [[PARAM_2_:%.+]]: tensor<8xf32>, [[PARAM_3_:%.+]]: tensor<8xf32>) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>) {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Add"([[VAR_0_]], [[PARAM_3_]]) : (tensor<2x4x8xf32>, tensor<8xf32>) -> tensor<2x4x8xf32>
// CHECK:           [[VAR_2_:%.+]]:3 = "onnx.Custom"([[VAR_1_]], [[PARAM_2_]]) {axis = -1 : si64, domain_name = "", epsilon = 9.99999974E-6 : f32, function_name = "SimplifiedLayerNormalization", stash_type = 1 : si64} : (tensor<2x4x8xf32>, tensor<8xf32>) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>, none)
// CHECK:           onnx.Return [[VAR_2_]]#0, [[VAR_2_]]#1 : tensor<2x4x8xf32>, tensor<2x4x1xf32>
// CHECK:         }
}


// -----
// SkipSimplifiedLayerNormalization: 4 inputs, 3 outputs (output, mean, inv_std_var)

func.func @skip_simplified_layernorm_three_outputs(%input: tensor<2x4x8xf32>, %skip: tensor<2x4x8xf32>, %gamma: tensor<8xf32>, %bias: tensor<8xf32>) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>, tensor<2x4x1xf32>) {
  %r0, %r1, %r2 = "onnx.Custom"(%input, %skip, %gamma, %bias) {domain_name = "com.microsoft", function_name = "SkipSimplifiedLayerNormalization", epsilon = 1.000000e-05 : f32} : (tensor<2x4x8xf32>, tensor<2x4x8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>, tensor<2x4x1xf32>)
  onnx.Return %r0, %r1, %r2 : tensor<2x4x8xf32>, tensor<2x4x1xf32>, tensor<2x4x1xf32>
// CHECK-LABEL:  func.func @skip_simplified_layernorm_three_outputs
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x8xf32>, [[PARAM_1_:%.+]]: tensor<2x4x8xf32>, [[PARAM_2_:%.+]]: tensor<8xf32>, [[PARAM_3_:%.+]]: tensor<8xf32>) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>, tensor<2x4x1xf32>) {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Add"([[VAR_0_]], [[PARAM_3_]]) : (tensor<2x4x8xf32>, tensor<8xf32>) -> tensor<2x4x8xf32>
// CHECK:           [[VAR_2_:%.+]]:3 = "onnx.Custom"([[VAR_1_]], [[PARAM_2_]]) {axis = -1 : si64, domain_name = "", epsilon = 9.99999974E-6 : f32, function_name = "SimplifiedLayerNormalization", stash_type = 1 : si64} : (tensor<2x4x8xf32>, tensor<8xf32>) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>, tensor<2x4x1xf32>)
// CHECK:           onnx.Return [[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2 : tensor<2x4x8xf32>, tensor<2x4x1xf32>, tensor<2x4x1xf32>
}


// -----
// SkipSimplifiedLayerNormalization: 4 inputs, 4 outputs (output, mean, inv_std_var, sum)

func.func @skip_simplified_layernorm_four_outputs(%input: tensor<2x4x8xf32>, %skip: tensor<2x4x8xf32>, %gamma: tensor<8xf32>, %bias: tensor<8xf32>) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>, tensor<2x4x1xf32>, tensor<2x4x8xf32>) {
  %r0, %r1, %r2, %r3 = "onnx.Custom"(%input, %skip, %gamma, %bias) {domain_name = "com.microsoft", function_name = "SkipSimplifiedLayerNormalization", epsilon = 1.000000e-05 : f32} : (tensor<2x4x8xf32>, tensor<2x4x8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>, tensor<2x4x1xf32>, tensor<2x4x8xf32>)
  onnx.Return %r0, %r1, %r2, %r3 : tensor<2x4x8xf32>, tensor<2x4x1xf32>, tensor<2x4x1xf32>, tensor<2x4x8xf32>
// CHECK-LABEL:  func.func @skip_simplified_layernorm_four_outputs
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x8xf32>, [[PARAM_1_:%.+]]: tensor<2x4x8xf32>, [[PARAM_2_:%.+]]: tensor<8xf32>, [[PARAM_3_:%.+]]: tensor<8xf32>) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>, tensor<2x4x1xf32>, tensor<2x4x8xf32>) {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Add"([[VAR_0_]], [[PARAM_3_]]) : (tensor<2x4x8xf32>, tensor<8xf32>) -> tensor<2x4x8xf32>
// CHECK:           [[VAR_2_:%.+]]:3 = "onnx.Custom"([[VAR_1_]], [[PARAM_2_]]) {axis = -1 : si64, domain_name = "", epsilon = 9.99999974E-6 : f32, function_name = "SimplifiedLayerNormalization", stash_type = 1 : si64} : (tensor<2x4x8xf32>, tensor<8xf32>) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>, tensor<2x4x1xf32>)
// CHECK:           onnx.Return [[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_1_]] : tensor<2x4x8xf32>, tensor<2x4x1xf32>, tensor<2x4x1xf32>, tensor<2x4x8xf32>
// CHECK:         }
}

// -----
// SkipSimplifiedLayerNormalization: 4 inputs, 4 outputs (output, mean, inv_std_var, sum), mean unused

func.func @skip_simplified_layernorm_four_outputs_mean_unused(%input: tensor<2x4x8xf32>, %skip: tensor<2x4x8xf32>, %gamma: tensor<8xf32>, %bias: tensor<8xf32>) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>, tensor<2x4x8xf32>) {
  %r0, %r1, %r2, %r3 = "onnx.Custom"(%input, %skip, %gamma, %bias) {domain_name = "com.microsoft", function_name = "SkipSimplifiedLayerNormalization", epsilon = 1.000000e-05 : f32} : (tensor<2x4x8xf32>, tensor<2x4x8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<2x4x8xf32>, none, tensor<2x4x1xf32>, tensor<2x4x8xf32>)
  onnx.Return %r0, %r2, %r3 : tensor<2x4x8xf32>, tensor<2x4x1xf32>, tensor<2x4x8xf32>
// CHECK-LABEL:  func.func @skip_simplified_layernorm_four_outputs_mean_unused
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x8xf32>, [[PARAM_1_:%.+]]: tensor<2x4x8xf32>, [[PARAM_2_:%.+]]: tensor<8xf32>, [[PARAM_3_:%.+]]: tensor<8xf32>) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>, tensor<2x4x8xf32>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Add"([[VAR_1_]], [[PARAM_3_]]) : (tensor<2x4x8xf32>, tensor<8xf32>) -> tensor<2x4x8xf32>
// CHECK:           [[VAR_Y_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.RMSLayerNormalization"([[VAR_2_]], [[PARAM_2_]], [[VAR_0_]]) {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<2x4x8xf32>, tensor<8xf32>, none) -> (tensor<2x4x8xf32>, tensor<2x4x1xf32>)
// CHECK:           onnx.Return [[VAR_Y_]], [[VAR_InvStdDev_]], [[VAR_2_]] : tensor<2x4x8xf32>, tensor<2x4x1xf32>, tensor<2x4x8xf32>
}

// -----


func.func @minimal_gqa(%q: tensor<1x128x3072xf32>, %k: tensor<1x128x1536xf32>, %v: tensor<1x128x1536xf32>) -> tensor<1x128x3072xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %total_seqlen = "onnx.Constant"() {value = dense<256> : tensor<i32>} : () -> tensor<i32>
  %seqlens = "onnx.Constant"() {value = dense<255> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %out, %present_k, %present_v = "onnx.Custom"(%q, %k, %v, %none, %none, %seqlens, %total_seqlen) {
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    kv_num_heads = 16 : si64,
    num_heads = 32 : si64
  }: (tensor<1x128x3072xf32>, tensor<1x128x1536xf32>, tensor<1x128x1536xf32>, none, none, tensor<1x1xi32>, tensor<i32>) -> (tensor<1x128x3072xf32>, none, none)
  return %out : tensor<1x128x3072xf32>
}

// GQA decomposition checks: seqlens_k drives an explicit additive mask, so
// onnx.Attention must receive a mask operand and keep is_causal disabled.
// CHECK-LABEL: func.func @minimal_gqa
// CHECK-SAME:  (%[[Q:.*]]: tensor<1x128x3072xf32>, %[[K:.*]]: tensor<1x128x1536xf32>, %[[V:.*]]: tensor<1x128x1536xf32>) -> tensor<1x128x3072xf32>
// CHECK:       %[[NONE:.*]] = "onnx.NoValue"() {value} : () -> none
// CHECK:       %[[MASK:.*]] = "onnx.Where"({{.*}}) : (tensor<1x1x128x128xi1>, tensor<f32>, tensor<f32>) -> tensor<1x1x128x128xf32>
// CHECK:       %[[Y:.*]], %[[PK:.*]], %[[PV:.*]], %[[QK:.*]] = "onnx.Attention"(%[[Q]], %[[K]], %[[V]], %[[MASK]], %[[NONE]], %[[NONE]]) {is_causal = 0 : si64, kv_num_heads = 16 : si64, q_num_heads = 32 : si64, qk_matmul_output_mode = 0 : si64, softcap = 0.000000e+00 : f32}
// CHECK-SAME:      (tensor<1x128x3072xf32>, tensor<1x128x1536xf32>, tensor<1x128x1536xf32>, tensor<1x1x128x128xf32>, none, none) -> (tensor<1x128x3072xf32>, none, none, none)
// CHECK:       return %[[Y]] : tensor<1x128x3072xf32>

// -----

func.func @gqa_3d_inputs_present_kv(
  %q: tensor<1x128x3072xf32>,
  %k: tensor<1x128x1536xf32>,
  %v: tensor<1x128x1536xf32>,
  %past_k: tensor<1x16x256x96xf32>,
  %past_v: tensor<1x16x256x48xf32>
) -> (tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x48xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %total_seqlen = "onnx.Constant"() {value = dense<256> : tensor<i32>} : () -> tensor<i32>
  %seqlens = "onnx.Constant"() {value = dense<255> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %out, %present_k, %present_v = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %seqlens, %total_seqlen) {
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    kv_num_heads = 16 : si64,
    num_heads = 32 : si64
  }: (tensor<1x128x3072xf32>, tensor<1x128x1536xf32>, tensor<1x128x1536xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x48xf32>, tensor<1x1xi32>, tensor<i32>) -> (tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x48xf32>)
  return %out, %present_k, %present_v : tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x48xf32>
}

// CHECK-LABEL: func.func @gqa_3d_inputs_present_kv
// CHECK-SAME:  (%[[Q:.*]]: tensor<1x128x3072xf32>, %[[K:.*]]: tensor<1x128x1536xf32>, %[[V:.*]]: tensor<1x128x1536xf32>, %[[PAST_K:.*]]: tensor<1x16x256x96xf32>, %[[PAST_V:.*]]: tensor<1x16x256x48xf32>) -> (tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x48xf32>)
// CHECK:       %[[MASK:.*]] = "onnx.Where"({{.*}}) : (tensor<1x1x128x384xi1>, tensor<f32>, tensor<f32>) -> tensor<1x1x128x384xf32>
// CHECK:       %[[Y:.*]], %[[PK:.*]], %[[PV:.*]], %[[QK:.*]] = "onnx.Attention"(%[[Q]], %[[K]], %[[V]], %[[MASK]], %[[PAST_K]], %[[PAST_V]]) {is_causal = 0 : si64, kv_num_heads = 16 : si64, q_num_heads = 32 : si64, qk_matmul_output_mode = 0 : si64, softcap = 0.000000e+00 : f32}
// CHECK-SAME:      (tensor<1x128x3072xf32>, tensor<1x128x1536xf32>, tensor<1x128x1536xf32>, tensor<1x1x128x384xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x48xf32>) -> (tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x48xf32>, none)
// CHECK:       return %[[Y]], %[[PK]], %[[PV]] : tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x48xf32>

// -----

func.func @gqa_packed_inputs_3d(
  %qkv: tensor<1x128x6144xf32>,
  %past_k: tensor<1x16x256x96xf32>,
  %past_v: tensor<1x16x256x96xf32>
) -> (tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %total_seqlen = "onnx.Constant"() {value = dense<256> : tensor<i32>} : () -> tensor<i32>
  %seqlens = "onnx.Constant"() {value = dense<255> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %out, %present_k, %present_v = "onnx.Custom"(%qkv, %none, %none, %past_k, %past_v, %seqlens, %total_seqlen) {
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    kv_num_heads = 16 : si64,
    num_heads = 32 : si64
  }: (tensor<1x128x6144xf32>, none, none, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>, tensor<1x1xi32>, tensor<i32>) -> (tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>)
  return %out, %present_k, %present_v : tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>
}

// CHECK-LABEL: func.func @gqa_packed_inputs_3d
// CHECK-SAME:  (%[[QKV:.*]]: tensor<1x128x6144xf32>, %[[PAST_K:.*]]: tensor<1x16x256x96xf32>, %[[PAST_V:.*]]: tensor<1x16x256x96xf32>) -> (tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>)
// CHECK:       %[[SPLIT:.*]]:3 = "onnx.Split"(%[[QKV]], {{.*}}) {axis = 2 : si64} : (tensor<1x128x6144xf32>, tensor<3xi64>) -> (tensor<1x128x3072xf32>, tensor<1x128x1536xf32>, tensor<1x128x1536xf32>)
// CHECK:       %[[MASK:.*]] = "onnx.Where"({{.*}}) : (tensor<1x1x128x384xi1>, tensor<f32>, tensor<f32>) -> tensor<1x1x128x384xf32>
// CHECK:       %[[Y:.*]], %[[PK:.*]], %[[PV:.*]], %[[QK:.*]] = "onnx.Attention"(%[[SPLIT]]#0, %[[SPLIT]]#1, %[[SPLIT]]#2, %[[MASK]], %[[PAST_K]], %[[PAST_V]]) {is_causal = 0 : si64, kv_num_heads = 16 : si64, q_num_heads = 32 : si64, qk_matmul_output_mode = 0 : si64, softcap = 0.000000e+00 : f32}
// CHECK-SAME:      (tensor<1x128x3072xf32>, tensor<1x128x1536xf32>, tensor<1x128x1536xf32>, tensor<1x1x128x384xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>) -> (tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>, none)
// CHECK:       return %[[Y]], %[[PK]], %[[PV]] : tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>

// -----

func.func @gqa_packed_inputs_3d_rotary_embedding_no_position_ids(
  %qkv: tensor<1x128x6144xf32>,
  %past_k: tensor<1x16x256x96xf32>,
  %past_v: tensor<1x16x256x96xf32>,
  %cos_cache: tensor<4096x48xf32>,
  %sin_cache: tensor<4096x48xf32>
) -> (tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %total_seqlen = "onnx.Constant"() {value = dense<256> : tensor<i32>} : () -> tensor<i32>
  %seqlens = "onnx.Constant"() {value = dense<255> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %out, %present_k, %present_v = "onnx.Custom"(%qkv, %none, %none, %past_k, %past_v, %seqlens, %total_seqlen, %cos_cache, %sin_cache) {
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    kv_num_heads = 16 : si64,
    num_heads = 32 : si64,
    do_rotary = 1 : si64
  }: (tensor<1x128x6144xf32>, none, none, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>, tensor<1x1xi32>, tensor<i32>, tensor<4096x48xf32>, tensor<4096x48xf32>) -> (tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>)
  return %out, %present_k, %present_v : tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>
}

// CHECK-LABEL: func.func @gqa_packed_inputs_3d_rotary_embedding_no_position_ids
// CHECK-SAME:  (%[[QKV:.*]]: tensor<1x128x6144xf32>, %[[PAST_K:.*]]: tensor<1x16x256x96xf32>, %[[PAST_V:.*]]: tensor<1x16x256x96xf32>, %[[COS:.*]]: tensor<4096x48xf32>, %[[SIN:.*]]: tensor<4096x48xf32>) -> (tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>)
// CHECK:       %[[SPLIT:.*]]:3 = "onnx.Split"(%[[QKV]], {{.*}}) {axis = 2 : si64} : (tensor<1x128x6144xf32>, tensor<3xi64>) -> (tensor<1x128x3072xf32>, tensor<1x128x1536xf32>, tensor<1x128x1536xf32>)
// CHECK:       %[[POS:.*]] = "onnx.Add"({{.*}}) : (tensor<1x1xi64>, tensor<1x128xi64>) -> tensor<1x128xi64>
// CHECK:       %[[RQ:.*]] = "onnx.RotaryEmbedding"(%[[SPLIT]]#0, %[[COS]], %[[SIN]], %[[POS]]) {interleaved = 0 : si64, num_heads = 32 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x128x3072xf32>, tensor<4096x48xf32>, tensor<4096x48xf32>, tensor<1x128xi64>) -> tensor<1x128x3072xf32>
// CHECK:       %[[RK:.*]] = "onnx.RotaryEmbedding"(%[[SPLIT]]#1, %[[COS]], %[[SIN]], %[[POS]]) {interleaved = 0 : si64, num_heads = 16 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x128x1536xf32>, tensor<4096x48xf32>, tensor<4096x48xf32>, tensor<1x128xi64>) -> tensor<1x128x1536xf32>
// CHECK:       %[[MASK:.*]] = "onnx.Where"({{.*}}) : (tensor<1x1x128x384xi1>, tensor<f32>, tensor<f32>) -> tensor<1x1x128x384xf32>
// CHECK:       %[[Y:.*]], %[[PK:.*]], %[[PV:.*]], %[[QK:.*]] = "onnx.Attention"(%[[RQ]], %[[RK]], %[[SPLIT]]#2, %[[MASK]], %[[PAST_K]], %[[PAST_V]]) {is_causal = 0 : si64, kv_num_heads = 16 : si64, q_num_heads = 32 : si64, qk_matmul_output_mode = 0 : si64, softcap = 0.000000e+00 : f32}
// CHECK-SAME:      (tensor<1x128x3072xf32>, tensor<1x128x1536xf32>, tensor<1x128x1536xf32>, tensor<1x1x128x384xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>) -> (tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>, none)
// CHECK:       return %[[Y]], %[[PK]], %[[PV]] : tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>

// -----

func.func @gqa_rotary_embedding_with_position_ids(
  %q: tensor<1x128x3072xf32>,
  %k: tensor<1x128x1536xf32>,
  %v: tensor<1x128x1536xf32>,
  %past_k: tensor<1x16x256x96xf32>,
  %past_v: tensor<1x16x256x96xf32>,
  %cos_cache: tensor<4096x48xf32>,
  %sin_cache: tensor<4096x48xf32>,
  %pos_ids: tensor<1x128xi64>
) -> (tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>)
 {
  %total_seqlen = "onnx.Constant"() {value = dense<256> : tensor<i32>} : () -> tensor<i32>
  %seqlens = "onnx.Constant"() {value = dense<255> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %out, %present_k, %present_v = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %seqlens, %total_seqlen, %cos_cache, %sin_cache, %pos_ids) {
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    kv_num_heads = 16 : si64,
    num_heads = 32 : si64,
    do_rotary = 1 : si64,
    rotary_interleaved = 1 : si64
  } : (tensor<1x128x3072xf32>, tensor<1x128x1536xf32>, tensor<1x128x1536xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>, tensor<1x1xi32>, tensor<i32>, tensor<4096x48xf32>, tensor<4096x48xf32>, tensor<1x128xi64>)
    -> (tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>)
  return %out, %present_k, %present_v : tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>
}

// CHECK-LABEL: func.func @gqa_rotary_embedding_with_position_ids
// CHECK-SAME:  (%[[Q:.*]]: tensor<1x128x3072xf32>, %[[K:.*]]: tensor<1x128x1536xf32>, %[[V:.*]]: tensor<1x128x1536xf32>, %[[PAST_K:.*]]: tensor<1x16x256x96xf32>, %[[PAST_V:.*]]: tensor<1x16x256x96xf32>, %[[COS:.*]]: tensor<4096x48xf32>, %[[SIN:.*]]: tensor<4096x48xf32>, %[[POS:.*]]: tensor<1x128xi64>) -> (tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>)
// CHECK:       %[[RQ:.*]] = "onnx.RotaryEmbedding"(%[[Q]], %[[COS]], %[[SIN]], %[[POS]]) {interleaved = 1 : si64, num_heads = 32 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x128x3072xf32>, tensor<4096x48xf32>, tensor<4096x48xf32>, tensor<1x128xi64>) -> tensor<1x128x3072xf32>
// CHECK:       %[[RK:.*]] = "onnx.RotaryEmbedding"(%[[K]], %[[COS]], %[[SIN]], %[[POS]]) {interleaved = 1 : si64, num_heads = 16 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x128x1536xf32>, tensor<4096x48xf32>, tensor<4096x48xf32>, tensor<1x128xi64>) -> tensor<1x128x1536xf32>
// CHECK:       %[[MASK:.*]] = "onnx.Where"({{.*}}) : (tensor<1x1x128x384xi1>, tensor<f32>, tensor<f32>) -> tensor<1x1x128x384xf32>
// CHECK:       %[[Y:.*]], %[[PK:.*]], %[[PV:.*]], %[[QK:.*]] = "onnx.Attention"(%[[RQ]], %[[RK]], %[[V]], %[[MASK]], %[[PAST_K]], %[[PAST_V]]) {is_causal = 0 : si64, kv_num_heads = 16 : si64, q_num_heads = 32 : si64, qk_matmul_output_mode = 0 : si64, softcap = 0.000000e+00 : f32}
// CHECK-SAME:      (tensor<1x128x3072xf32>, tensor<1x128x1536xf32>, tensor<1x128x1536xf32>, tensor<1x1x128x384xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>) -> (tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>, none)
// CHECK:       return %[[Y]], %[[PK]], %[[PV]] : tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>

// -----

// GroupQueryAttention has no rotary_embedding_dim attribute, so a partial
// rotation is only visible in the narrower cos/sin caches: 36 instead of the
// 48 a fully rotated head_size of 96 would need. The emitted RotaryEmbedding
// must carry the recovered dimension, otherwise it claims to rotate the whole
// head.
func.func @gqa_rotary_embedding_partial_rotation(
  %q: tensor<1x128x3072xf32>,
  %k: tensor<1x128x1536xf32>,
  %v: tensor<1x128x1536xf32>,
  %past_k: tensor<1x16x256x96xf32>,
  %past_v: tensor<1x16x256x96xf32>,
  %cos_cache: tensor<4096x36xf32>,
  %sin_cache: tensor<4096x36xf32>,
  %pos_ids: tensor<1x128xi64>
) -> (tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>)
 {
  %total_seqlen = "onnx.Constant"() {value = dense<256> : tensor<i32>} : () -> tensor<i32>
  %seqlens = "onnx.Constant"() {value = dense<255> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %out, %present_k, %present_v = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %seqlens, %total_seqlen, %cos_cache, %sin_cache, %pos_ids) {
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    kv_num_heads = 16 : si64,
    num_heads = 32 : si64,
    do_rotary = 1 : si64
  } : (tensor<1x128x3072xf32>, tensor<1x128x1536xf32>, tensor<1x128x1536xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>, tensor<1x1xi32>, tensor<i32>, tensor<4096x36xf32>, tensor<4096x36xf32>, tensor<1x128xi64>)
    -> (tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>)
  return %out, %present_k, %present_v : tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>
}

// CHECK-LABEL: func.func @gqa_rotary_embedding_partial_rotation
// CHECK-SAME:  (%[[Q:.*]]: tensor<1x128x3072xf32>, %[[K:.*]]: tensor<1x128x1536xf32>, %[[V:.*]]: tensor<1x128x1536xf32>, %[[PAST_K:.*]]: tensor<1x16x256x96xf32>, %[[PAST_V:.*]]: tensor<1x16x256x96xf32>, %[[COS:.*]]: tensor<4096x36xf32>, %[[SIN:.*]]: tensor<4096x36xf32>, %[[POS:.*]]: tensor<1x128xi64>) -> (tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>)
// CHECK:       %[[RQ:.*]] = "onnx.RotaryEmbedding"(%[[Q]], %[[COS]], %[[SIN]], %[[POS]]) {interleaved = 0 : si64, num_heads = 32 : si64, rotary_embedding_dim = 72 : si64} : (tensor<1x128x3072xf32>, tensor<4096x36xf32>, tensor<4096x36xf32>, tensor<1x128xi64>) -> tensor<1x128x3072xf32>
// CHECK:       %[[RK:.*]] = "onnx.RotaryEmbedding"(%[[K]], %[[COS]], %[[SIN]], %[[POS]]) {interleaved = 0 : si64, num_heads = 16 : si64, rotary_embedding_dim = 72 : si64} : (tensor<1x128x1536xf32>, tensor<4096x36xf32>, tensor<4096x36xf32>, tensor<1x128xi64>) -> tensor<1x128x1536xf32>
// CHECK:       %[[Y:.*]], %[[PK:.*]], %[[PV:.*]], %[[QK:.*]] = "onnx.Attention"(%[[RQ]], %[[RK]], %[[V]], {{.*}}, %[[PAST_K]], %[[PAST_V]])
// CHECK:       return %[[Y]], %[[PK]], %[[PV]] : tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>

// -----

func.func @gqa_with_attention_bias_and_qk_output(
  %q: tensor<1x128x3072xf32>,
  %k: tensor<1x128x1536xf32>,
  %v: tensor<1x128x1536xf32>,
  %past_k: tensor<1x16x256x96xf32>,
  %past_v: tensor<1x16x256x96xf32>,
  %att_bias: tensor<1x1x128x384xf32>
) -> (tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>, tensor<1x32x128x256xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %total_seqlen = "onnx.Constant"() {value = dense<256> : tensor<i32>} : () -> tensor<i32>
  %seqlens = "onnx.Constant"() {value = dense<255> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %out, %present_k, %present_v, %qk_output = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %seqlens, %total_seqlen, %none, %none, %none, %att_bias) {
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    kv_num_heads = 16 : si64,
    num_heads = 32 : si64,
    qk_output = 1 : si64
  } : (tensor<1x128x3072xf32>, tensor<1x128x1536xf32>, tensor<1x128x1536xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>, tensor<1x1xi32>, tensor<i32>, none, none, none, tensor<1x1x128x384xf32>)
    -> (tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>, tensor<1x32x128x256xf32>)
  return %out, %present_k, %present_v, %qk_output : tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>, tensor<1x32x128x256xf32>
}

// CHECK-LABEL: func.func @gqa_with_attention_bias_and_qk_output
// CHECK-SAME:  (%[[Q:.*]]: tensor<1x128x3072xf32>, %[[K:.*]]: tensor<1x128x1536xf32>, %[[V:.*]]: tensor<1x128x1536xf32>, %[[PAST_K:.*]]: tensor<1x16x256x96xf32>, %[[PAST_V:.*]]: tensor<1x16x256x96xf32>, %[[BIAS:.*]]: tensor<1x1x128x384xf32>) -> (tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>, tensor<1x32x128x256xf32>)
// CHECK:       %[[MASK:.*]] = "onnx.Where"({{.*}}) : (tensor<1x1x128x384xi1>, tensor<f32>, tensor<f32>) -> tensor<1x1x128x384xf32>
// CHECK:       %[[BIAS_MASK:.*]] = "onnx.Add"(%[[BIAS]], %[[MASK]]) : (tensor<1x1x128x384xf32>, tensor<1x1x128x384xf32>) -> tensor<1x1x128x384xf32>
// CHECK:       %[[Y:.*]], %[[PK:.*]], %[[PV:.*]], %[[QK:.*]] = "onnx.Attention"(%[[Q]], %[[K]], %[[V]], %[[BIAS_MASK]], %[[PAST_K]], %[[PAST_V]]) {is_causal = 0 : si64, kv_num_heads = 16 : si64, q_num_heads = 32 : si64, qk_matmul_output_mode = 0 : si64, softcap = 0.000000e+00 : f32}
// CHECK-SAME:      (tensor<1x128x3072xf32>, tensor<1x128x1536xf32>, tensor<1x128x1536xf32>, tensor<1x1x128x384xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>) -> (tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>, tensor<1x32x128x256xf32>)
// CHECK:       return %[[Y]], %[[PK]], %[[PV]], %[[QK]] : tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>, tensor<1x32x128x256xf32>

// -----


func.func @gqa_with_attention_bias_per_head(
  %q: tensor<1x128x3072xf32>,
  %k: tensor<1x128x1536xf32>,
  %v: tensor<1x128x1536xf32>,
  %past_k: tensor<1x16x256x96xf32>,
  %past_v: tensor<1x16x256x96xf32>,
  %att_bias: tensor<1x32x128x384xf32>
) -> (tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>, tensor<1x32x128x256xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %total_seqlen = "onnx.Constant"() {value = dense<256> : tensor<i32>} : () -> tensor<i32>
  %seqlens = "onnx.Constant"() {value = dense<255> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %out, %present_k, %present_v, %qk_output = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %seqlens, %total_seqlen, %none, %none, %none, %att_bias) {
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    kv_num_heads = 16 : si64,
    num_heads = 32 : si64,
    qk_output = 1 : si64
  } : (tensor<1x128x3072xf32>, tensor<1x128x1536xf32>, tensor<1x128x1536xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>, tensor<1x1xi32>, tensor<i32>, none, none, none, tensor<1x32x128x384xf32>)
    -> (tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>, tensor<1x32x128x256xf32>)
  return %out, %present_k, %present_v, %qk_output : tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>, tensor<1x32x128x256xf32>
}

// CHECK-LABEL: func.func @gqa_with_attention_bias_per_head
// CHECK-SAME:  (%[[Q:.*]]: tensor<1x128x3072xf32>, %[[K:.*]]: tensor<1x128x1536xf32>, %[[V:.*]]: tensor<1x128x1536xf32>, %[[PAST_K:.*]]: tensor<1x16x256x96xf32>, %[[PAST_V:.*]]: tensor<1x16x256x96xf32>, %[[BIAS:.*]]: tensor<1x32x128x384xf32>) -> (tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>, tensor<1x32x128x256xf32>)
// CHECK:       %[[MASK:.*]] = "onnx.Where"({{.*}}) : (tensor<1x1x128x384xi1>, tensor<f32>, tensor<f32>) -> tensor<1x1x128x384xf32>
// CHECK:       %[[BIAS_MASK:.*]] = "onnx.Add"(%[[BIAS]], %[[MASK]]) : (tensor<1x32x128x384xf32>, tensor<1x1x128x384xf32>) -> tensor<1x32x128x384xf32>
// CHECK:       %[[Y:.*]], %[[PK:.*]], %[[PV:.*]], %[[QK:.*]] = "onnx.Attention"(%[[Q]], %[[K]], %[[V]], %[[BIAS_MASK]], %[[PAST_K]], %[[PAST_V]]) {is_causal = 0 : si64, kv_num_heads = 16 : si64, q_num_heads = 32 : si64, qk_matmul_output_mode = 0 : si64, softcap = 0.000000e+00 : f32}
// CHECK-SAME:      (tensor<1x128x3072xf32>, tensor<1x128x1536xf32>, tensor<1x128x1536xf32>, tensor<1x32x128x384xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>) -> (tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>, tensor<1x32x128x256xf32>)
// CHECK:       return %[[Y]], %[[PK]], %[[PV]], %[[QK]] : tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>, tensor<1x32x128x256xf32>

// -----

func.func @gqa_with_scale_softcap_and_qk_output_2(
  %q: tensor<1x128x3072xf32>,
  %k: tensor<1x128x1536xf32>,
  %v: tensor<1x128x1536xf32>,
  %past_k: tensor<1x16x256x96xf32>,
  %past_v: tensor<1x16x256x96xf32>
) -> (tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>, tensor<1x32x128x256xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %total_seqlen = "onnx.Constant"() {value = dense<256> : tensor<i32>} : () -> tensor<i32>
  %seqlens = "onnx.Constant"() {value = dense<255> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %out, %present_k, %present_v, %qk_output = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %seqlens, %total_seqlen) {
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    kv_num_heads = 16 : si64,
    num_heads = 32 : si64,
    qk_output = 2 : si64,
    scale = 2.0 : f32,
    softcap = 10.0 : f32
  } : (tensor<1x128x3072xf32>, tensor<1x128x1536xf32>, tensor<1x128x1536xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>, tensor<1x1xi32>, tensor<i32>)
    -> (tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>, tensor<1x32x128x256xf32>)
  return %out, %present_k, %present_v, %qk_output : tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>, tensor<1x32x128x256xf32>
}

// CHECK-LABEL: func.func @gqa_with_scale_softcap_and_qk_output_2
// CHECK-SAME:  (%[[Q:.*]]: tensor<1x128x3072xf32>, %[[K:.*]]: tensor<1x128x1536xf32>, %[[V:.*]]: tensor<1x128x1536xf32>, %[[PAST_K:.*]]: tensor<1x16x256x96xf32>, %[[PAST_V:.*]]: tensor<1x16x256x96xf32>) -> (tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>, tensor<1x32x128x256xf32>)
// CHECK:       %[[MASK:.*]] = "onnx.Where"({{.*}}) : (tensor<1x1x128x384xi1>, tensor<f32>, tensor<f32>) -> tensor<1x1x128x384xf32>
// CHECK:       %[[Y:.*]], %[[PK:.*]], %[[PV:.*]], %[[QK:.*]] = "onnx.Attention"(%[[Q]], %[[K]], %[[V]], %[[MASK]], %[[PAST_K]], %[[PAST_V]]) {is_causal = 0 : si64, kv_num_heads = 16 : si64, q_num_heads = 32 : si64, qk_matmul_output_mode = 3 : si64, scale = 2.000000e+00 : f32, softcap = 1.000000e+01 : f32}
// CHECK-SAME:      (tensor<1x128x3072xf32>, tensor<1x128x1536xf32>, tensor<1x128x1536xf32>, tensor<1x1x128x384xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>) -> (tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>, tensor<1x32x128x256xf32>)
// CHECK:       return %[[Y]], %[[PK]], %[[PV]], %[[QK]] : tensor<1x128x3072xf32>, tensor<1x16x384x96xf32>, tensor<1x16x384x96xf32>, tensor<1x32x128x256xf32>

// -----

func.func @gqa_batch4_do_rotary_no_position_ids(
  %q: tensor<4x128x3072xf32>,
  %k: tensor<4x128x1536xf32>,
  %v: tensor<4x128x1536xf32>,
  %past_k: tensor<4x16x256x96xf32>,
  %past_v: tensor<4x16x256x48xf32>,
  %cos_cache: tensor<4096x48xf32>,
  %sin_cache: tensor<4096x48xf32>
) -> (tensor<4x128x3072xf32>, tensor<4x16x384x96xf32>, tensor<4x16x384x48xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %total_seqlen = "onnx.Constant"() {value = dense<256> : tensor<i32>} : () -> tensor<i32>
  %seqlens = "onnx.Constant"() {value = dense<255> : tensor<4x1xi32>} : () -> tensor<4x1xi32>
  %out, %present_k, %present_v = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %seqlens, %total_seqlen, %cos_cache, %sin_cache) {
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    do_rotary = 1 : si64,
    kv_num_heads = 16 : si64,
    num_heads = 32 : si64
  }: (tensor<4x128x3072xf32>, tensor<4x128x1536xf32>, tensor<4x128x1536xf32>, tensor<4x16x256x96xf32>, tensor<4x16x256x48xf32>, tensor<4x1xi32>, tensor<i32>, tensor<4096x48xf32>, tensor<4096x48xf32>) -> (tensor<4x128x3072xf32>, tensor<4x16x384x96xf32>, tensor<4x16x384x48xf32>)
  return %out, %present_k, %present_v : tensor<4x128x3072xf32>, tensor<4x16x384x96xf32>, tensor<4x16x384x48xf32>
}

// CHECK-LABEL: func.func @gqa_batch4_do_rotary_no_position_ids
// CHECK-SAME:  (%[[Q:.*]]: tensor<4x128x3072xf32>, %[[K:.*]]: tensor<4x128x1536xf32>, %[[V:.*]]: tensor<4x128x1536xf32>, %[[PAST_K:.*]]: tensor<4x16x256x96xf32>, %[[PAST_V:.*]]: tensor<4x16x256x48xf32>, %[[COS:.*]]: tensor<4096x48xf32>, %[[SIN:.*]]: tensor<4096x48xf32>) -> (tensor<4x128x3072xf32>, tensor<4x16x384x96xf32>, tensor<4x16x384x48xf32>)
// CHECK:       %[[POS:.*]] = "onnx.Add"({{.*}}) : (tensor<4x1xi64>, tensor<1x128xi64>) -> tensor<4x128xi64>
// CHECK:       %[[RQ:.*]] = "onnx.RotaryEmbedding"(%[[Q]], %[[COS]], %[[SIN]], %[[POS]]) {interleaved = 0 : si64, num_heads = 32 : si64, rotary_embedding_dim = 0 : si64} : (tensor<4x128x3072xf32>, tensor<4096x48xf32>, tensor<4096x48xf32>, tensor<4x128xi64>) -> tensor<4x128x3072xf32>
// CHECK:       %[[RK:.*]] = "onnx.RotaryEmbedding"(%[[K]], %[[COS]], %[[SIN]], %[[POS]]) {interleaved = 0 : si64, num_heads = 16 : si64, rotary_embedding_dim = 0 : si64} : (tensor<4x128x1536xf32>, tensor<4096x48xf32>, tensor<4096x48xf32>, tensor<4x128xi64>) -> tensor<4x128x1536xf32>
// CHECK:       %[[MASK:.*]] = "onnx.Where"({{.*}}) : (tensor<4x1x128x384xi1>, tensor<f32>, tensor<f32>) -> tensor<4x1x128x384xf32>
// CHECK:       %[[Y:.*]], %[[PK:.*]], %[[PV:.*]], %[[QK:.*]] = "onnx.Attention"(%[[RQ]], %[[RK]], %[[V]], %[[MASK]], %[[PAST_K]], %[[PAST_V]]) {is_causal = 0 : si64, kv_num_heads = 16 : si64, q_num_heads = 32 : si64, qk_matmul_output_mode = 0 : si64, softcap = 0.000000e+00 : f32}
// CHECK-SAME:      (tensor<4x128x3072xf32>, tensor<4x128x1536xf32>, tensor<4x128x1536xf32>, tensor<4x1x128x384xf32>, tensor<4x16x256x96xf32>, tensor<4x16x256x48xf32>) -> (tensor<4x128x3072xf32>, tensor<4x16x384x96xf32>, tensor<4x16x384x48xf32>, none)
// CHECK:       return %[[Y]], %[[PK]], %[[PV]] : tensor<4x128x3072xf32>, tensor<4x16x384x96xf32>, tensor<4x16x384x48xf32>

// -----

// Test: do_rotary=1 with no position_ids and no past_key; GQA should not be decomposed.
func.func @gqa_rotary_no_position_ids_no_past_key(
  %q: tensor<1x128x3072xf32>,
  %k: tensor<1x128x1536xf32>,
  %v: tensor<1x128x1536xf32>,
  %cos_cache: tensor<4096x48xf32>,
  %sin_cache: tensor<4096x48xf32>
) -> (tensor<1x128x3072xf32>, tensor<?x16x?x96xf32>, tensor<?x16x?x48xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %total_seqlen = "onnx.Constant"() {value = dense<128> : tensor<i32>} : () -> tensor<i32>
  %seqlens = "onnx.Constant"() {value = dense<127> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %out, %present_k, %present_v = "onnx.Custom"(%q, %k, %v, %none, %none, %seqlens, %total_seqlen, %cos_cache, %sin_cache) {
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    do_rotary = 1 : si64,
    kv_num_heads = 16 : si64,
    num_heads = 32 : si64
  } : (tensor<1x128x3072xf32>, tensor<1x128x1536xf32>, tensor<1x128x1536xf32>, none, none, tensor<1x1xi32>, tensor<i32>, tensor<4096x48xf32>, tensor<4096x48xf32>) -> (tensor<1x128x3072xf32>, tensor<?x16x?x96xf32>, tensor<?x16x?x48xf32>)
  return %out, %present_k, %present_v : tensor<1x128x3072xf32>, tensor<?x16x?x96xf32>, tensor<?x16x?x48xf32>
}

// CHECK-LABEL: func.func @gqa_rotary_no_position_ids_no_past_key
// CHECK-SAME:  (%[[Q:.*]]: tensor<1x128x3072xf32>, %[[K:.*]]: tensor<1x128x1536xf32>, %[[V:.*]]: tensor<1x128x1536xf32>, %[[COS:.*]]: tensor<4096x48xf32>, %[[SIN:.*]]: tensor<4096x48xf32>) -> (tensor<1x128x3072xf32>, tensor<?x16x?x96xf32>, tensor<?x16x?x48xf32>)
// CHECK:       %[[NONE:.*]] = "onnx.NoValue"() {value} : () -> none
// CHECK:       %[[POS:.*]] = "onnx.Add"({{.*}}) : (tensor<1x1xi64>, tensor<1x128xi64>) -> tensor<1x128xi64>
// CHECK:       %[[RQ:.*]] = "onnx.RotaryEmbedding"(%[[Q]], %[[COS]], %[[SIN]], %[[POS]]) {interleaved = 0 : si64, num_heads = 32 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x128x3072xf32>, tensor<4096x48xf32>, tensor<4096x48xf32>, tensor<1x128xi64>) -> tensor<1x128x3072xf32>
// CHECK:       %[[RK:.*]] = "onnx.RotaryEmbedding"(%[[K]], %[[COS]], %[[SIN]], %[[POS]]) {interleaved = 0 : si64, num_heads = 16 : si64, rotary_embedding_dim = 0 : si64} : (tensor<1x128x1536xf32>, tensor<4096x48xf32>, tensor<4096x48xf32>, tensor<1x128xi64>) -> tensor<1x128x1536xf32>
// CHECK:       %[[MASK:.*]] = "onnx.Where"({{.*}}) : (tensor<1x1x128x128xi1>, tensor<f32>, tensor<f32>) -> tensor<1x1x128x128xf32>
// CHECK:       %[[Y:.*]], %[[PK_NONE:.*]], %[[PV_NONE:.*]], %[[QK:.*]] = "onnx.Attention"(%[[RQ]], %[[RK]], %[[V]], %[[MASK]], %[[NONE]], %[[NONE]]) {is_causal = 0 : si64, kv_num_heads = 16 : si64, q_num_heads = 32 : si64, qk_matmul_output_mode = 0 : si64, softcap = 0.000000e+00 : f32}
// CHECK-SAME:      (tensor<1x128x3072xf32>, tensor<1x128x1536xf32>, tensor<1x128x1536xf32>, tensor<1x1x128x128xf32>, none, none) -> (tensor<1x128x3072xf32>, none, none, none)
// CHECK:       return %[[Y]], {{.*}} : tensor<1x128x3072xf32>, tensor<?x16x?x96xf32>, tensor<?x16x?x48xf32>

// -----

// Test: do_rotary=1 with no position_ids and a dynamic past_key;  GQA should not be decomposed
func.func @gqa_rotary_no_position_ids_dynamic_past_key(
  %q: tensor<1x128x3072xf32>,
  %k: tensor<1x128x1536xf32>,
  %v: tensor<1x128x1536xf32>,
  %past_k: tensor<?x16x?x96xf32>,
  %past_v: tensor<?x16x?x48xf32>,
  %cos_cache: tensor<4096x48xf32>,
  %sin_cache: tensor<4096x48xf32>
) -> (tensor<1x128x3072xf32>, tensor<?x16x?x96xf32>, tensor<?x16x?x48xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %total_seqlen = "onnx.Constant"() {value = dense<128> : tensor<i32>} : () -> tensor<i32>
  %seqlens = "onnx.Constant"() {value = dense<127> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %out, %present_k, %present_v = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %seqlens, %total_seqlen, %cos_cache, %sin_cache) {
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    do_rotary = 1 : si64,
    kv_num_heads = 16 : si64,
    num_heads = 32 : si64
  } : (tensor<1x128x3072xf32>, tensor<1x128x1536xf32>, tensor<1x128x1536xf32>, tensor<?x16x?x96xf32>, tensor<?x16x?x48xf32>, tensor<1x1xi32>, tensor<i32>, tensor<4096x48xf32>, tensor<4096x48xf32>) -> (tensor<1x128x3072xf32>, tensor<?x16x?x96xf32>, tensor<?x16x?x48xf32>)
  return %out, %present_k, %present_v : tensor<1x128x3072xf32>, tensor<?x16x?x96xf32>, tensor<?x16x?x48xf32>
}

// CHECK-LABEL: func.func @gqa_rotary_no_position_ids_dynamic_past_key
// CHECK-SAME:  (%[[Q:.*]]: tensor<1x128x3072xf32>, %[[K:.*]]: tensor<1x128x1536xf32>, %[[V:.*]]: tensor<1x128x1536xf32>, %[[PAST_K:.*]]: tensor<?x16x?x96xf32>, %[[PAST_V:.*]]: tensor<?x16x?x48xf32>, %[[COS:.*]]: tensor<4096x48xf32>, %[[SIN:.*]]: tensor<4096x48xf32>) -> (tensor<1x128x3072xf32>, tensor<?x16x?x96xf32>, tensor<?x16x?x48xf32>)
// CHECK-NOT:   "onnx.Attention"
// CHECK:       %[[GQA:.*]]:3 = "onnx.Custom"(%[[Q]], %[[K]], %[[V]], %[[PAST_K]], %[[PAST_V]], {{.*}}, {{.*}}, %[[COS]], %[[SIN]]) {do_rotary = 1 : si64, domain_name = "com.microsoft", function_name = "GroupQueryAttention", kv_num_heads = 16 : si64, num_heads = 32 : si64}
// CHECK-SAME:      (tensor<1x128x3072xf32>, tensor<1x128x1536xf32>, tensor<1x128x1536xf32>, tensor<?x16x?x96xf32>, tensor<?x16x?x48xf32>, tensor<1x1xi32>, tensor<i32>, tensor<4096x48xf32>, tensor<4096x48xf32>) -> (tensor<1x128x3072xf32>, tensor<?x16x?x96xf32>, tensor<?x16x?x48xf32>)
// CHECK:       return %[[GQA]]#0, %[[GQA]]#1, %[[GQA]]#2 : tensor<1x128x3072xf32>, tensor<?x16x?x96xf32>, tensor<?x16x?x48xf32>

// -----

func.func @gqa_padded_prefill_mask(
  %q: tensor<2x4x3072xf32>,
  %k: tensor<2x4x1536xf32>,
  %v: tensor<2x4x1536xf32>
) -> (tensor<2x4x3072xf32>, tensor<2x16x4x96xf32>, tensor<2x16x4x96xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %total_seqlen = "onnx.Constant"() {value = dense<4> : tensor<i32>} : () -> tensor<i32>
  %seqlens = "onnx.Constant"() {value = dense<[[2], [3]]> : tensor<2x1xi32>} : () -> tensor<2x1xi32>
  %out, %present_k, %present_v = "onnx.Custom"(%q, %k, %v, %none, %none, %seqlens, %total_seqlen) {
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    kv_num_heads = 16 : si64,
    num_heads = 32 : si64
  } : (tensor<2x4x3072xf32>, tensor<2x4x1536xf32>, tensor<2x4x1536xf32>, none, none, tensor<2x1xi32>, tensor<i32>) -> (tensor<2x4x3072xf32>, tensor<2x16x4x96xf32>, tensor<2x16x4x96xf32>)
  return %out, %present_k, %present_v : tensor<2x4x3072xf32>, tensor<2x16x4x96xf32>, tensor<2x16x4x96xf32>
}

// CHECK-LABEL: func.func @gqa_padded_prefill_mask
// CHECK-SAME:  (%[[Q:.*]]: tensor<2x4x3072xf32>, %[[K:.*]]: tensor<2x4x1536xf32>, %[[V:.*]]: tensor<2x4x1536xf32>) -> (tensor<2x4x3072xf32>, tensor<2x16x4x96xf32>, tensor<2x16x4x96xf32>)
// CHECK:       %[[NONE:.*]] = "onnx.NoValue"() {value} : () -> none
// CHECK:       %[[MASK:.*]] = "onnx.Where"({{.*}}) : (tensor<2x1x4x4xi1>, tensor<f32>, tensor<f32>) -> tensor<2x1x4x4xf32>
// CHECK:       %[[Y:.*]], %[[PK_NONE:.*]], %[[PV_NONE:.*]], %[[QK:.*]] = "onnx.Attention"(%[[Q]], %[[K]], %[[V]], %[[MASK]], %[[NONE]], %[[NONE]]) {is_causal = 0 : si64, kv_num_heads = 16 : si64, q_num_heads = 32 : si64, qk_matmul_output_mode = 0 : si64, softcap = 0.000000e+00 : f32}
// CHECK-SAME:      (tensor<2x4x3072xf32>, tensor<2x4x1536xf32>, tensor<2x4x1536xf32>, tensor<2x1x4x4xf32>, none, none) -> (tensor<2x4x3072xf32>, none, none, none)
// CHECK:       return %[[Y]], {{.*}} : tensor<2x4x3072xf32>, tensor<2x16x4x96xf32>, tensor<2x16x4x96xf32>

// -----

func.func @gqa_preallocated_cache_slot_write(
  %q: tensor<1x1x3072xf32>,
  %k: tensor<1x1x1536xf32>,
  %v: tensor<1x1x1536xf32>,
  %past_k: tensor<1x16x512x96xf32>,
  %past_v: tensor<1x16x512x96xf32>
) -> (tensor<1x1x3072xf32>, tensor<1x16x512x96xf32>, tensor<1x16x512x96xf32>) {
  %total_seqlen = "onnx.Constant"() {value = dense<512> : tensor<i32>} : () -> tensor<i32>
  %seqlens = "onnx.Constant"() {value = dense<255> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %out, %present_k, %present_v = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %seqlens, %total_seqlen) {
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    kv_num_heads = 16 : si64,
    num_heads = 32 : si64
  } : (tensor<1x1x3072xf32>, tensor<1x1x1536xf32>, tensor<1x1x1536xf32>, tensor<1x16x512x96xf32>, tensor<1x16x512x96xf32>, tensor<1x1xi32>, tensor<i32>) -> (tensor<1x1x3072xf32>, tensor<1x16x512x96xf32>, tensor<1x16x512x96xf32>)
  return %out, %present_k, %present_v : tensor<1x1x3072xf32>, tensor<1x16x512x96xf32>, tensor<1x16x512x96xf32>
}

// CHECK-LABEL: func.func @gqa_preallocated_cache_slot_write
// CHECK-SAME:  (%[[Q:.*]]: tensor<1x1x3072xf32>, %[[K:.*]]: tensor<1x1x1536xf32>, %[[V:.*]]: tensor<1x1x1536xf32>, %[[PAST_K:.*]]: tensor<1x16x512x96xf32>, %[[PAST_V:.*]]: tensor<1x16x512x96xf32>) -> (tensor<1x1x3072xf32>, tensor<1x16x512x96xf32>, tensor<1x16x512x96xf32>)
// CHECK:       %[[POSITIONS:.*]] = onnx.Constant {{.*}} : tensor<1x1x512x1xui16>
// CHECK:       %[[NONE:.*]] = "onnx.NoValue"() {value} : () -> none
// CHECK:       %[[SEQLENS_UI16:.*]] = "onnx.Cast"({{.*}}) {saturate = 1 : si64, to = ui16} : (tensor<1x1xi32>) -> tensor<1x1xui16>
// CHECK:       %[[SEQLENS_4D:.*]] = "onnx.Reshape"(%[[SEQLENS_UI16]], {{.*}}) {allowzero = 0 : si64} : (tensor<1x1xui16>, tensor<4xi64>) -> tensor<1x1x1x1xui16>
// CHECK-NOT:   "onnx.Expand"
// CHECK:       %[[SELECTED_SLOT:.*]] = "onnx.Equal"(%[[SEQLENS_4D]], %[[POSITIONS]]) : (tensor<1x1x1x1xui16>, tensor<1x1x512x1xui16>) -> tensor<1x1x512x1xi1>
// CHECK:       %[[SLOT_SELECTOR:.*]] = "onnx.Cast"(%[[SELECTED_SLOT]]) {saturate = 1 : si64, to = f32} : (tensor<1x1x512x1xi1>) -> tensor<1x1x512x1xf32>
// CHECK:       %[[CURRENT_K:.*]] = "onnx.Transpose"({{.*}}) {perm = [0, 2, 1, 3]} : (tensor<1x1x16x96xf32>) -> tensor<1x16x1x96xf32>
// CHECK:       %[[K_DELTA:.*]] = "onnx.Sub"(%[[CURRENT_K]], %[[PAST_K]]) : (tensor<1x16x1x96xf32>, tensor<1x16x512x96xf32>) -> tensor<1x16x512x96xf32>
// CHECK:       %[[SELECTED_K_DELTA:.*]] = "onnx.Mul"(%[[K_DELTA]], %[[SLOT_SELECTOR]]) : (tensor<1x16x512x96xf32>, tensor<1x1x512x1xf32>) -> tensor<1x16x512x96xf32>
// CHECK:       %[[PRESENT_K:.*]] = "onnx.Add"(%[[PAST_K]], %[[SELECTED_K_DELTA]]) : (tensor<1x16x512x96xf32>, tensor<1x16x512x96xf32>) -> tensor<1x16x512x96xf32>
// CHECK:       %[[CURRENT_V:.*]] = "onnx.Transpose"({{.*}}) {perm = [0, 2, 1, 3]} : (tensor<1x1x16x96xf32>) -> tensor<1x16x1x96xf32>
// CHECK:       %[[V_DELTA:.*]] = "onnx.Sub"(%[[CURRENT_V]], %[[PAST_V]]) : (tensor<1x16x1x96xf32>, tensor<1x16x512x96xf32>) -> tensor<1x16x512x96xf32>
// CHECK:       %[[SELECTED_V_DELTA:.*]] = "onnx.Mul"(%[[V_DELTA]], %[[SLOT_SELECTOR]]) : (tensor<1x16x512x96xf32>, tensor<1x1x512x1xf32>) -> tensor<1x16x512x96xf32>
// CHECK:       %[[PRESENT_V:.*]] = "onnx.Add"(%[[PAST_V]], %[[SELECTED_V_DELTA]]) : (tensor<1x16x512x96xf32>, tensor<1x16x512x96xf32>) -> tensor<1x16x512x96xf32>
// CHECK-NOT:   "onnx.ScatterElements"
// CHECK:       %[[MASK:.*]] = "onnx.Where"({{.*}}) : (tensor<1x1x1x512xi1>, tensor<f32>, tensor<f32>) -> tensor<1x1x1x512xf32>
// CHECK:       %[[Y:.*]], %[[PK_NONE:.*]], %[[PV_NONE:.*]], %[[QK:.*]] = "onnx.Attention"(%[[Q]], %[[PRESENT_K]], %[[PRESENT_V]], %[[MASK]], %[[NONE]], %[[NONE]]) {is_causal = 0 : si64, kv_num_heads = 16 : si64, q_num_heads = 32 : si64, qk_matmul_output_mode = 0 : si64, softcap = 0.000000e+00 : f32}
// CHECK-SAME:      (tensor<1x1x3072xf32>, tensor<1x16x512x96xf32>, tensor<1x16x512x96xf32>, tensor<1x1x1x512xf32>, none, none) -> (tensor<1x1x3072xf32>, none, none, none)
// CHECK:       return %[[Y]], %[[PRESENT_K]], %[[PRESENT_V]] : tensor<1x1x3072xf32>, tensor<1x16x512x96xf32>, tensor<1x16x512x96xf32>

// SCATTER-LABEL: func.func @gqa_preallocated_cache_slot_write
// SCATTER-NOT:   ui16
// SCATTER:       %[[K_INDICES:.*]] = "onnx.Expand"({{.*}}) : (tensor<1x1x1x1xi64>, tensor<4xi64>) -> tensor<1x16x1x96xi64>
// SCATTER:       %[[PRESENT_K:.*]] = "onnx.ScatterElements"(%[[PAST_K:.*]], %[[K_INDICES]], {{.*}}) {axis = 2 : si64, reduction = "none"} : (tensor<1x16x512x96xf32>, tensor<1x16x1x96xi64>, tensor<1x16x1x96xf32>) -> tensor<1x16x512x96xf32>
// SCATTER:       %[[V_INDICES:.*]] = "onnx.Expand"({{.*}}) : (tensor<1x1x1x1xi64>, tensor<4xi64>) -> tensor<1x16x1x96xi64>
// SCATTER:       %[[PRESENT_V:.*]] = "onnx.ScatterElements"(%[[PAST_V:.*]], %[[V_INDICES]], {{.*}}) {axis = 2 : si64, reduction = "none"} : (tensor<1x16x512x96xf32>, tensor<1x16x1x96xi64>, tensor<1x16x1x96xf32>) -> tensor<1x16x512x96xf32>
// SCATTER:       return {{.*}}, %[[PRESENT_K]], %[[PRESENT_V]]

// -----

func.func @gqa_uint16_cache_slot_rewrite_falls_back_for_oversized_cache(
  %q: tensor<1x1x3072xf32>,
  %k: tensor<1x1x1536xf32>,
  %v: tensor<1x1x1536xf32>,
  %past_k: tensor<1x16x65537x96xf32>,
  %past_v: tensor<1x16x65537x96xf32>
) -> (tensor<1x1x3072xf32>, tensor<1x16x65537x96xf32>, tensor<1x16x65537x96xf32>) {
  %total_seqlen = "onnx.Constant"() {value = dense<65537> : tensor<i32>} : () -> tensor<i32>
  %seqlens = "onnx.Constant"() {value = dense<65536> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %out, %present_k, %present_v = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %seqlens, %total_seqlen) {
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    kv_num_heads = 16 : si64,
    num_heads = 32 : si64
  } : (tensor<1x1x3072xf32>, tensor<1x1x1536xf32>, tensor<1x1x1536xf32>, tensor<1x16x65537x96xf32>, tensor<1x16x65537x96xf32>, tensor<1x1xi32>, tensor<i32>) -> (tensor<1x1x3072xf32>, tensor<1x16x65537x96xf32>, tensor<1x16x65537x96xf32>)
  return %out, %present_k, %present_v : tensor<1x1x3072xf32>, tensor<1x16x65537x96xf32>, tensor<1x16x65537x96xf32>
}

// CHECK-LABEL: func.func @gqa_uint16_cache_slot_rewrite_falls_back_for_oversized_cache
// CHECK-NOT:   ui16
// CHECK:       "onnx.Expand"
// CHECK:       "onnx.ScatterElements"
// CHECK:       "onnx.Expand"
// CHECK:       "onnx.ScatterElements"
// CHECK:       "onnx.Attention"

// -----


func.func @gqa_preallocated_cache_slot_write_with_attention_bias(
  %q: tensor<1x1x3072xf32>,
  %k: tensor<1x1x1536xf32>,
  %v: tensor<1x1x1536xf32>,
  %past_k: tensor<1x16x512x96xf32>,
  %past_v: tensor<1x16x512x96xf32>,
  %att_bias: tensor<1x32x1x512xf32>
) -> (tensor<1x1x3072xf32>, tensor<1x16x512x96xf32>, tensor<1x16x512x96xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %total_seqlen = "onnx.Constant"() {value = dense<512> : tensor<i32>} : () -> tensor<i32>
  %seqlens = "onnx.Constant"() {value = dense<255> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %out, %present_k, %present_v = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %seqlens, %total_seqlen, %none, %none, %none, %att_bias) {
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    kv_num_heads = 16 : si64,
    num_heads = 32 : si64
  } : (tensor<1x1x3072xf32>, tensor<1x1x1536xf32>, tensor<1x1x1536xf32>, tensor<1x16x512x96xf32>, tensor<1x16x512x96xf32>, tensor<1x1xi32>, tensor<i32>, none, none, none, tensor<1x32x1x512xf32>) -> (tensor<1x1x3072xf32>, tensor<1x16x512x96xf32>, tensor<1x16x512x96xf32>)
  return %out, %present_k, %present_v : tensor<1x1x3072xf32>, tensor<1x16x512x96xf32>, tensor<1x16x512x96xf32>
}

// CHECK-LABEL: func.func @gqa_preallocated_cache_slot_write_with_attention_bias
// CHECK-SAME:  (%[[Q:.*]]: tensor<1x1x3072xf32>, %[[K:.*]]: tensor<1x1x1536xf32>, %[[V:.*]]: tensor<1x1x1536xf32>, %[[PAST_K:.*]]: tensor<1x16x512x96xf32>, %[[PAST_V:.*]]: tensor<1x16x512x96xf32>, %[[BIAS:.*]]: tensor<1x32x1x512xf32>) -> (tensor<1x1x3072xf32>, tensor<1x16x512x96xf32>, tensor<1x16x512x96xf32>)
// CHECK:       %[[POSITIONS:.*]] = onnx.Constant {{.*}} : tensor<1x1x512x1xui16>
// CHECK:       %[[NONE:.*]] = "onnx.NoValue"() {value} : () -> none
// CHECK:       %[[SEQLENS_UI16:.*]] = "onnx.Cast"({{.*}}) {saturate = 1 : si64, to = ui16} : (tensor<1x1xi32>) -> tensor<1x1xui16>
// CHECK:       %[[SEQLENS_4D:.*]] = "onnx.Reshape"(%[[SEQLENS_UI16]], {{.*}}) {allowzero = 0 : si64} : (tensor<1x1xui16>, tensor<4xi64>) -> tensor<1x1x1x1xui16>
// CHECK-NOT:   "onnx.Expand"
// CHECK:       %[[SELECTED_SLOT:.*]] = "onnx.Equal"(%[[SEQLENS_4D]], %[[POSITIONS]]) : (tensor<1x1x1x1xui16>, tensor<1x1x512x1xui16>) -> tensor<1x1x512x1xi1>
// CHECK:       %[[SLOT_SELECTOR:.*]] = "onnx.Cast"(%[[SELECTED_SLOT]]) {saturate = 1 : si64, to = f32} : (tensor<1x1x512x1xi1>) -> tensor<1x1x512x1xf32>
// CHECK:       %[[CURRENT_K:.*]] = "onnx.Transpose"({{.*}}) {perm = [0, 2, 1, 3]} : (tensor<1x1x16x96xf32>) -> tensor<1x16x1x96xf32>
// CHECK:       %[[K_DELTA:.*]] = "onnx.Sub"(%[[CURRENT_K]], %[[PAST_K]]) : (tensor<1x16x1x96xf32>, tensor<1x16x512x96xf32>) -> tensor<1x16x512x96xf32>
// CHECK:       %[[SELECTED_K_DELTA:.*]] = "onnx.Mul"(%[[K_DELTA]], %[[SLOT_SELECTOR]]) : (tensor<1x16x512x96xf32>, tensor<1x1x512x1xf32>) -> tensor<1x16x512x96xf32>
// CHECK:       %[[PRESENT_K:.*]] = "onnx.Add"(%[[PAST_K]], %[[SELECTED_K_DELTA]]) : (tensor<1x16x512x96xf32>, tensor<1x16x512x96xf32>) -> tensor<1x16x512x96xf32>
// CHECK:       %[[CURRENT_V:.*]] = "onnx.Transpose"({{.*}}) {perm = [0, 2, 1, 3]} : (tensor<1x1x16x96xf32>) -> tensor<1x16x1x96xf32>
// CHECK:       %[[V_DELTA:.*]] = "onnx.Sub"(%[[CURRENT_V]], %[[PAST_V]]) : (tensor<1x16x1x96xf32>, tensor<1x16x512x96xf32>) -> tensor<1x16x512x96xf32>
// CHECK:       %[[SELECTED_V_DELTA:.*]] = "onnx.Mul"(%[[V_DELTA]], %[[SLOT_SELECTOR]]) : (tensor<1x16x512x96xf32>, tensor<1x1x512x1xf32>) -> tensor<1x16x512x96xf32>
// CHECK:       %[[PRESENT_V:.*]] = "onnx.Add"(%[[PAST_V]], %[[SELECTED_V_DELTA]]) : (tensor<1x16x512x96xf32>, tensor<1x16x512x96xf32>) -> tensor<1x16x512x96xf32>
// CHECK-NOT:   "onnx.ScatterElements"
// CHECK:       %[[MASK:.*]] = "onnx.Where"({{.*}}) : (tensor<1x1x1x512xi1>, tensor<f32>, tensor<f32>) -> tensor<1x1x1x512xf32>
// CHECK:       %[[BIAS_MASK:.*]] = "onnx.Add"(%[[BIAS]], %[[MASK]]) : (tensor<1x32x1x512xf32>, tensor<1x1x1x512xf32>) -> tensor<1x32x1x512xf32>
// CHECK:       %[[Y:.*]], %[[PK_NONE:.*]], %[[PV_NONE:.*]], %[[QK:.*]] = "onnx.Attention"(%[[Q]], %[[PRESENT_K]], %[[PRESENT_V]], %[[BIAS_MASK]], %[[NONE]], %[[NONE]]) {is_causal = 0 : si64, kv_num_heads = 16 : si64, q_num_heads = 32 : si64, qk_matmul_output_mode = 0 : si64, softcap = 0.000000e+00 : f32}
// CHECK-SAME:      (tensor<1x1x3072xf32>, tensor<1x16x512x96xf32>, tensor<1x16x512x96xf32>, tensor<1x32x1x512xf32>, none, none) -> (tensor<1x1x3072xf32>, none, none, none)
// CHECK:       return %[[Y]], %[[PRESENT_K]], %[[PRESENT_V]] : tensor<1x1x3072xf32>, tensor<1x16x512x96xf32>, tensor<1x16x512x96xf32>

// -----


func.func @gqa_local_window_size_minus_one_decomposes(
  %q: tensor<1x1x3072xf32>,
  %k: tensor<1x1x1536xf32>,
  %v: tensor<1x1x1536xf32>,
  %past_k: tensor<1x16x256x96xf32>,
  %past_v: tensor<1x16x256x96xf32>
) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>) {
  %total_seqlen = "onnx.Constant"() {value = dense<257> : tensor<i32>} : () -> tensor<i32>
  %seqlens = "onnx.Constant"() {value = dense<256> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %out, %present_k, %present_v = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %seqlens, %total_seqlen) {
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    kv_num_heads = 16 : si64,
    local_window_size = -1 : si64,
    num_heads = 32 : si64
  } : (tensor<1x1x3072xf32>, tensor<1x1x1536xf32>, tensor<1x1x1536xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>, tensor<1x1xi32>, tensor<i32>) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>)
  return %out, %present_k, %present_v : tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>
}

// CHECK-LABEL: func.func @gqa_local_window_size_minus_one_decomposes
// CHECK-SAME:  (%[[Q:.*]]: tensor<1x1x3072xf32>, %[[K:.*]]: tensor<1x1x1536xf32>, %[[V:.*]]: tensor<1x1x1536xf32>, %[[PAST_K:.*]]: tensor<1x16x256x96xf32>, %[[PAST_V:.*]]: tensor<1x16x256x96xf32>) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>)
// CHECK:       %[[Y:.*]], %[[PK:.*]], %[[PV:.*]], %[[QK:.*]] = "onnx.Attention"(%[[Q]], %[[K]], %[[V]], {{.*}}, %[[PAST_K]], %[[PAST_V]])
// CHECK:       return %[[Y]], %[[PK]], %[[PV]] : tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>

// -----

// attention span is 257 here, so 256 is rejected
func.func @gqa_local_window_size_rejected(
  %q: tensor<1x1x3072xf32>,
  %k: tensor<1x1x1536xf32>,
  %v: tensor<1x1x1536xf32>,
  %past_k: tensor<1x16x256x96xf32>,
  %past_v: tensor<1x16x256x96xf32>,
  %seqlens: tensor<1x1xi32>,
  %total_seqlen: tensor<i32>
) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>) {
  %out, %present_k, %present_v = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %seqlens, %total_seqlen) {
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    kv_num_heads = 16 : si64,
    local_window_size = 256 : si64,
    num_heads = 32 : si64
  } : (tensor<1x1x3072xf32>, tensor<1x1x1536xf32>, tensor<1x1x1536xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>, tensor<1x1xi32>, tensor<i32>) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>)
  return %out, %present_k, %present_v : tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>
}

// CHECK-LABEL: func.func @gqa_local_window_size_rejected
// CHECK-SAME:  (%[[Q:.*]]: tensor<1x1x3072xf32>, %[[K:.*]]: tensor<1x1x1536xf32>, %[[V:.*]]: tensor<1x1x1536xf32>, %[[PAST_K:.*]]: tensor<1x16x256x96xf32>, %[[PAST_V:.*]]: tensor<1x16x256x96xf32>, %[[SEQLENS:.*]]: tensor<1x1xi32>, %[[TOTAL_SEQLEN:.*]]: tensor<i32>) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>)
// CHECK-NOT:   "onnx.Attention"
// CHECK:       %[[GQA:.*]]:3 = "onnx.Custom"
// CHECK-SAME:      local_window_size = 256 : si64
// CHECK:       return %[[GQA]]#0, %[[GQA]]#1, %[[GQA]]#2 : tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>

// -----

// A window wider than the 257-key attention span can never hide a key, so this
// must decompose exactly like local_window_size = -1 above.
func.func @gqa_local_window_size_unreachable_decomposes(
  %q: tensor<1x1x3072xf32>,
  %k: tensor<1x1x1536xf32>,
  %v: tensor<1x1x1536xf32>,
  %past_k: tensor<1x16x256x96xf32>,
  %past_v: tensor<1x16x256x96xf32>
) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>) {
  %total_seqlen = "onnx.Constant"() {value = dense<257> : tensor<i32>} : () -> tensor<i32>
  %seqlens = "onnx.Constant"() {value = dense<256> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %out, %present_k, %present_v = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %seqlens, %total_seqlen) {
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    kv_num_heads = 16 : si64,
    local_window_size = 262144 : si64,
    num_heads = 32 : si64
  } : (tensor<1x1x3072xf32>, tensor<1x1x1536xf32>, tensor<1x1x1536xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>, tensor<1x1xi32>, tensor<i32>) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>)
  return %out, %present_k, %present_v : tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>
}

// CHECK-LABEL: func.func @gqa_local_window_size_unreachable_decomposes
// CHECK-SAME:  (%[[Q:.*]]: tensor<1x1x3072xf32>, %[[K:.*]]: tensor<1x1x1536xf32>, %[[V:.*]]: tensor<1x1x1536xf32>, %[[PAST_K:.*]]: tensor<1x16x256x96xf32>, %[[PAST_V:.*]]: tensor<1x16x256x96xf32>) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>)
// CHECK:       %[[Y:.*]], %[[PK:.*]], %[[PV:.*]], %[[QK:.*]] = "onnx.Attention"(%[[Q]], %[[K]], %[[V]], {{.*}}, %[[PAST_K]], %[[PAST_V]])
// CHECK:       return %[[Y]], %[[PK]], %[[PV]] : tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>

// -----

// A window of exactly the attention span still hides nothing: the furthest
// query is at position 256 and the earliest key is 0, so 0 > 256 - 257.
func.func @gqa_local_window_size_equal_to_span_decomposes(
  %q: tensor<1x1x3072xf32>,
  %k: tensor<1x1x1536xf32>,
  %v: tensor<1x1x1536xf32>,
  %past_k: tensor<1x16x256x96xf32>,
  %past_v: tensor<1x16x256x96xf32>
) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>) {
  %total_seqlen = "onnx.Constant"() {value = dense<257> : tensor<i32>} : () -> tensor<i32>
  %seqlens = "onnx.Constant"() {value = dense<256> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %out, %present_k, %present_v = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %seqlens, %total_seqlen) {
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    kv_num_heads = 16 : si64,
    local_window_size = 257 : si64,
    num_heads = 32 : si64
  } : (tensor<1x1x3072xf32>, tensor<1x1x1536xf32>, tensor<1x1x1536xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>, tensor<1x1xi32>, tensor<i32>) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>)
  return %out, %present_k, %present_v : tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>
}

// CHECK-LABEL: func.func @gqa_local_window_size_equal_to_span_decomposes
// CHECK:       %[[Y:.*]], %[[PK:.*]], %[[PV:.*]], %[[QK:.*]] = "onnx.Attention"
// CHECK:       return %[[Y]], %[[PK]], %[[PV]] : tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>

// -----

// A zero-length window would hide even the current key, leaving an all -inf
// softmax row, so it is rejected regardless of the attention span.
func.func @gqa_local_window_size_zero_rejected(
  %q: tensor<1x1x3072xf32>,
  %k: tensor<1x1x1536xf32>,
  %v: tensor<1x1x1536xf32>,
  %past_k: tensor<1x16x256x96xf32>,
  %past_v: tensor<1x16x256x96xf32>
) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>) {
  %total_seqlen = "onnx.Constant"() {value = dense<257> : tensor<i32>} : () -> tensor<i32>
  %seqlens = "onnx.Constant"() {value = dense<256> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %out, %present_k, %present_v = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %seqlens, %total_seqlen) {
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    kv_num_heads = 16 : si64,
    local_window_size = 0 : si64,
    num_heads = 32 : si64
  } : (tensor<1x1x3072xf32>, tensor<1x1x1536xf32>, tensor<1x1x1536xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>, tensor<1x1xi32>, tensor<i32>) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>)
  return %out, %present_k, %present_v : tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>
}

// CHECK-LABEL: func.func @gqa_local_window_size_zero_rejected
// CHECK-NOT:   "onnx.Attention"
// CHECK:       %[[GQA:.*]]:3 = "onnx.Custom"
// CHECK-SAME:      local_window_size = 0 : si64
// CHECK:       return %[[GQA]]#0, %[[GQA]]#1, %[[GQA]]#2 : tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>

// -----

func.func @gqa_causal_and_linear_cache_defaults_decompose(
  %q: tensor<1x1x3072xf32>,
  %k: tensor<1x1x1536xf32>,
  %v: tensor<1x1x1536xf32>,
  %past_k: tensor<1x16x256x96xf32>,
  %past_v: tensor<1x16x256x96xf32>
) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>) {
  %total_seqlen = "onnx.Constant"() {value = dense<257> : tensor<i32>} : () -> tensor<i32>
  %seqlens = "onnx.Constant"() {value = dense<256> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %out, %present_k, %present_v = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %seqlens, %total_seqlen) {
    causal = 1 : si64,
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    kv_num_heads = 16 : si64,
    num_heads = 32 : si64,
    sliding_window_cache = 0 : si64
  } : (tensor<1x1x3072xf32>, tensor<1x1x1536xf32>, tensor<1x1x1536xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>, tensor<1x1xi32>, tensor<i32>) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>)
  return %out, %present_k, %present_v : tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>
}

// CHECK-LABEL: func.func @gqa_causal_and_linear_cache_defaults_decompose
// CHECK-NOT:   "onnx.Custom"
// CHECK:       %[[Y:.*]], %[[PK:.*]], %[[PV:.*]], %[[QK:.*]] = "onnx.Attention"
// CHECK:       return %[[Y]], %[[PK]], %[[PV]] : tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>

// -----

func.func @gqa_non_causal_rejected(
  %q: tensor<1x1x3072xf32>,
  %k: tensor<1x1x1536xf32>,
  %v: tensor<1x1x1536xf32>,
  %past_k: tensor<1x16x256x96xf32>,
  %past_v: tensor<1x16x256x96xf32>
) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>) {
  %total_seqlen = "onnx.Constant"() {value = dense<257> : tensor<i32>} : () -> tensor<i32>
  %seqlens = "onnx.Constant"() {value = dense<256> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %out, %present_k, %present_v = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %seqlens, %total_seqlen) {
    causal = 0 : si64,
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    kv_num_heads = 16 : si64,
    num_heads = 32 : si64
  } : (tensor<1x1x3072xf32>, tensor<1x1x1536xf32>, tensor<1x1x1536xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>, tensor<1x1xi32>, tensor<i32>) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>)
  return %out, %present_k, %present_v : tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>
}

// CHECK-LABEL: func.func @gqa_non_causal_rejected
// CHECK-NOT:   "onnx.Attention"
// CHECK:       %[[GQA:.*]]:3 = "onnx.Custom"
// CHECK-SAME:      causal = 0 : si64
// CHECK:       return %[[GQA]]#0, %[[GQA]]#1, %[[GQA]]#2 : tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>

// -----

func.func @gqa_sliding_window_cache_rejected(
  %q: tensor<1x1x3072xf32>,
  %k: tensor<1x1x1536xf32>,
  %v: tensor<1x1x1536xf32>,
  %past_k: tensor<1x16x256x96xf32>,
  %past_v: tensor<1x16x256x96xf32>
) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>) {
  %total_seqlen = "onnx.Constant"() {value = dense<257> : tensor<i32>} : () -> tensor<i32>
  %seqlens = "onnx.Constant"() {value = dense<256> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %out, %present_k, %present_v = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %seqlens, %total_seqlen) {
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    kv_num_heads = 16 : si64,
    num_heads = 32 : si64,
    sliding_window_cache = 1 : si64
  } : (tensor<1x1x3072xf32>, tensor<1x1x1536xf32>, tensor<1x1x1536xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>, tensor<1x1xi32>, tensor<i32>) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>)
  return %out, %present_k, %present_v : tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>
}

// CHECK-LABEL: func.func @gqa_sliding_window_cache_rejected
// CHECK-NOT:   "onnx.Attention"
// CHECK:       %[[GQA:.*]]:3 = "onnx.Custom"
// CHECK-SAME:      sliding_window_cache = 1 : si64
// CHECK:       return %[[GQA]]#0, %[[GQA]]#1, %[[GQA]]#2 : tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>

// -----

// preallocated mode has attention span of only 256, so this is accepted
// compare with gqa_local_window_size_rejected
// test also with runtime seqlens + total_seqlen
func.func @gqa_local_window_size_unreachable_preallocated_decode(
  %q: tensor<1x1x3072xf32>,
  %k: tensor<1x1x1536xf32>,
  %v: tensor<1x1x1536xf32>,
  %past_k: tensor<1x16x256x96xf32>,
  %past_v: tensor<1x16x256x96xf32>,
  %seqlens: tensor<1x1xi32>,
  %total_seqlen: tensor<i32>
) -> (tensor<1x1x3072xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>) {
  %out, %present_k, %present_v = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %seqlens, %total_seqlen) {
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    kv_num_heads = 16 : si64,
    local_window_size = 256 : si64,
    num_heads = 32 : si64
  } : (tensor<1x1x3072xf32>, tensor<1x1x1536xf32>, tensor<1x1x1536xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>, tensor<1x1xi32>, tensor<i32>) -> (tensor<1x1x3072xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>)
  return %out, %present_k, %present_v : tensor<1x1x3072xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>
}

// CHECK-LABEL: func.func @gqa_local_window_size_unreachable_preallocated_decode
// CHECK-SAME:  (%[[Q:.*]]: tensor<1x1x3072xf32>, %[[K:.*]]: tensor<1x1x1536xf32>, %[[V:.*]]: tensor<1x1x1536xf32>, %[[PAST_K:.*]]: tensor<1x16x256x96xf32>, %[[PAST_V:.*]]: tensor<1x16x256x96xf32>, %[[SEQLENS:.*]]: tensor<1x1xi32>, %[[TOTAL_SEQLEN:.*]]: tensor<i32>)
// CHECK:       "onnx.Cast"(%[[SEQLENS]]) {saturate = 1 : si64, to = i64} : (tensor<1x1xi32>) -> tensor<1x1xi64>
// CHECK:       %[[VALID_LEN:.*]] = "onnx.Add"({{.*}}) : (tensor<1x1x1x1xi64>, tensor<1xi64>) -> tensor<1x1x1x1xi64>
// CHECK:       %[[KEY_VALID:.*]] = "onnx.Less"({{.*}}, %[[VALID_LEN]]) : (tensor<1x1x1x256xi64>, tensor<1x1x1x1xi64>) -> tensor<1x1x1x256xi1>
// CHECK:       %[[MASK:.*]] = "onnx.Where"({{.*}}) : (tensor<1x1x1x256xi1>, tensor<f32>, tensor<f32>) -> tensor<1x1x1x256xf32>
// CHECK:       %[[Y:.*]], %[[PK_NONE:.*]], %[[PV_NONE:.*]], %[[QK:.*]] = "onnx.Attention"({{.*}}, %[[MASK]], {{.*}})
// CHECK:       return %[[Y]], {{.*}} : tensor<1x1x3072xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>

// -----

func.func @gqa_single_output_not_decomposed(
  %q: tensor<1x1x3072xf32>,
  %k: tensor<1x1x1536xf32>,
  %v: tensor<1x1x1536xf32>,
  %past_k: tensor<1x16x256x96xf32>,
  %past_v: tensor<1x16x256x96xf32>
) -> tensor<1x1x3072xf32> {
  %total_seqlen = "onnx.Constant"() {value = dense<257> : tensor<i32>} : () -> tensor<i32>
  %seqlens = "onnx.Constant"() {value = dense<256> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %out = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %seqlens, %total_seqlen) {
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    kv_num_heads = 16 : si64,
    num_heads = 32 : si64
  } : (tensor<1x1x3072xf32>, tensor<1x1x1536xf32>, tensor<1x1x1536xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>, tensor<1x1xi32>, tensor<i32>) -> tensor<1x1x3072xf32>
  return %out : tensor<1x1x3072xf32>
}

// CHECK-LABEL: func.func @gqa_single_output_not_decomposed
// CHECK-SAME:  (%[[Q:.*]]: tensor<1x1x3072xf32>, %[[K:.*]]: tensor<1x1x1536xf32>, %[[V:.*]]: tensor<1x1x1536xf32>, %[[PAST_K:.*]]: tensor<1x16x256x96xf32>, %[[PAST_V:.*]]: tensor<1x16x256x96xf32>) -> tensor<1x1x3072xf32>
// CHECK-NOT:   "onnx.Attention"
// CHECK:       %[[GQA:.*]] = "onnx.Custom"
// CHECK-SAME:      function_name = "GroupQueryAttention"
// CHECK:       return %[[GQA]] : tensor<1x1x3072xf32>

// -----

func.func @gqa_quantized_cache_rejected(
  %q: tensor<1x1x3072xf32>,
  %k: tensor<1x1x1536xf32>,
  %v: tensor<1x1x1536xf32>,
  %past_k: tensor<1x16x256x96xf32>,
  %past_v: tensor<1x16x256x96xf32>
) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>) {
  %total_seqlen = "onnx.Constant"() {value = dense<257> : tensor<i32>} : () -> tensor<i32>
  %seqlens = "onnx.Constant"() {value = dense<256> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %out, %present_k, %present_v = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %seqlens, %total_seqlen) {
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    kv_cache_bit_width = 8 : si64,
    kv_num_heads = 16 : si64,
    num_heads = 32 : si64
  } : (tensor<1x1x3072xf32>, tensor<1x1x1536xf32>, tensor<1x1x1536xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>, tensor<1x1xi32>, tensor<i32>) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>)
  return %out, %present_k, %present_v : tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>
}

// CHECK-LABEL: func.func @gqa_quantized_cache_rejected
// CHECK-SAME:  (%[[Q:.*]]: tensor<1x1x3072xf32>, %[[K:.*]]: tensor<1x1x1536xf32>, %[[V:.*]]: tensor<1x1x1536xf32>, %[[PAST_K:.*]]: tensor<1x16x256x96xf32>, %[[PAST_V:.*]]: tensor<1x16x256x96xf32>) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>)
// CHECK-NOT:   "onnx.Attention"
// CHECK:       %[[GQA:.*]]:3 = "onnx.Custom"(%[[Q]], %[[K]], %[[V]], %[[PAST_K]], %[[PAST_V]], {{.*}}, {{.*}}) {domain_name = "com.microsoft", function_name = "GroupQueryAttention", kv_cache_bit_width = 8 : si64, kv_num_heads = 16 : si64, num_heads = 32 : si64}
// CHECK-SAME:      (tensor<1x1x3072xf32>, tensor<1x1x1536xf32>, tensor<1x1x1536xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>, tensor<1x1xi32>, tensor<i32>) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>)
// CHECK:       return %[[GQA]]#0, %[[GQA]]#1, %[[GQA]]#2 : tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>

// -----

func.func @gqa_quant_type_none_decomposes(
  %q: tensor<1x1x3072xf32>,
  %k: tensor<1x1x1536xf32>,
  %v: tensor<1x1x1536xf32>,
  %past_k: tensor<1x16x256x96xf32>,
  %past_v: tensor<1x16x256x96xf32>
) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>) {
  %total_seqlen = "onnx.Constant"() {value = dense<257> : tensor<i32>} : () -> tensor<i32>
  %seqlens = "onnx.Constant"() {value = dense<256> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %out, %present_k, %present_v = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %seqlens, %total_seqlen) {
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    k_quant_type = "NONE",
    kv_num_heads = 16 : si64,
    num_heads = 32 : si64,
    v_quant_type = "NONE"
  } : (tensor<1x1x3072xf32>, tensor<1x1x1536xf32>, tensor<1x1x1536xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>, tensor<1x1xi32>, tensor<i32>) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>)
  return %out, %present_k, %present_v : tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>
}

// CHECK-LABEL: func.func @gqa_quant_type_none_decomposes
// CHECK-SAME:  (%[[Q:.*]]: tensor<1x1x3072xf32>, %[[K:.*]]: tensor<1x1x1536xf32>, %[[V:.*]]: tensor<1x1x1536xf32>, %[[PAST_K:.*]]: tensor<1x16x256x96xf32>, %[[PAST_V:.*]]: tensor<1x16x256x96xf32>) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>)
// CHECK:       %[[MASK:.*]] = "onnx.Where"({{.*}}) : (tensor<1x1x1x257xi1>, tensor<f32>, tensor<f32>) -> tensor<1x1x1x257xf32>
// CHECK:       %[[Y:.*]], %[[PK:.*]], %[[PV:.*]], %[[QK:.*]] = "onnx.Attention"(%[[Q]], %[[K]], %[[V]], %[[MASK]], %[[PAST_K]], %[[PAST_V]]) {is_causal = 0 : si64, kv_num_heads = 16 : si64, q_num_heads = 32 : si64, qk_matmul_output_mode = 0 : si64, softcap = 0.000000e+00 : f32}
// CHECK-SAME:      (tensor<1x1x3072xf32>, tensor<1x1x1536xf32>, tensor<1x1x1536xf32>, tensor<1x1x1x257xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>, none)
// CHECK:       return %[[Y]], %[[PK]], %[[PV]] : tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>

// -----

func.func @gqa_quantized_cache_scale_input_rejected(
  %q: tensor<1x1x3072xf32>,
  %k: tensor<1x1x1536xf32>,
  %v: tensor<1x1x1536xf32>,
  %past_k: tensor<1x16x256x96xf32>,
  %past_v: tensor<1x16x256x96xf32>,
  %k_scale: tensor<f32>,
  %v_scale: tensor<f32>
) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %total_seqlen = "onnx.Constant"() {value = dense<257> : tensor<i32>} : () -> tensor<i32>
  %seqlens = "onnx.Constant"() {value = dense<256> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %out, %present_k, %present_v = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %seqlens, %total_seqlen, %none, %none, %none, %none, %none, %k_scale, %v_scale) {
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    kv_num_heads = 16 : si64,
    num_heads = 32 : si64
  } : (tensor<1x1x3072xf32>, tensor<1x1x1536xf32>, tensor<1x1x1536xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>, tensor<1x1xi32>, tensor<i32>, none, none, none, none, none, tensor<f32>, tensor<f32>) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>)
  return %out, %present_k, %present_v : tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>
}

// CHECK-LABEL: func.func @gqa_quantized_cache_scale_input_rejected
// CHECK-SAME:  (%[[Q:.*]]: tensor<1x1x3072xf32>, %[[K:.*]]: tensor<1x1x1536xf32>, %[[V:.*]]: tensor<1x1x1536xf32>, %[[PAST_K:.*]]: tensor<1x16x256x96xf32>, %[[PAST_V:.*]]: tensor<1x16x256x96xf32>, %[[K_SCALE:.*]]: tensor<f32>, %[[V_SCALE:.*]]: tensor<f32>) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>)
// CHECK-NOT:   "onnx.Attention"
// CHECK:       %[[GQA:.*]]:3 = "onnx.Custom"(%[[Q]], %[[K]], %[[V]], %[[PAST_K]], %[[PAST_V]], {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[K_SCALE]], %[[V_SCALE]]) {domain_name = "com.microsoft", function_name = "GroupQueryAttention", kv_num_heads = 16 : si64, num_heads = 32 : si64}
// CHECK-SAME:      (tensor<1x1x3072xf32>, tensor<1x1x1536xf32>, tensor<1x1x1536xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>, tensor<1x1xi32>, tensor<i32>, none, none, none, none, none, tensor<f32>, tensor<f32>) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>)
// CHECK:       return %[[GQA]]#0, %[[GQA]]#1, %[[GQA]]#2 : tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>

// -----

func.func @gqa_quantized_cache_type_attr_rejected(
  %q: tensor<1x1x3072xf32>,
  %k: tensor<1x1x1536xf32>,
  %v: tensor<1x1x1536xf32>,
  %past_k: tensor<1x16x256x96xf32>,
  %past_v: tensor<1x16x256x96xf32>
) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>) {
  %total_seqlen = "onnx.Constant"() {value = dense<257> : tensor<i32>} : () -> tensor<i32>
  %seqlens = "onnx.Constant"() {value = dense<256> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %out, %present_k, %present_v = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %seqlens, %total_seqlen) {
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    k_quant_type = "PER_TENSOR",
    kv_num_heads = 16 : si64,
    num_heads = 32 : si64
  } : (tensor<1x1x3072xf32>, tensor<1x1x1536xf32>, tensor<1x1x1536xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>, tensor<1x1xi32>, tensor<i32>) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>)
  return %out, %present_k, %present_v : tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>
}

// CHECK-LABEL: func.func @gqa_quantized_cache_type_attr_rejected
// CHECK-SAME:  (%[[Q:.*]]: tensor<1x1x3072xf32>, %[[K:.*]]: tensor<1x1x1536xf32>, %[[V:.*]]: tensor<1x1x1536xf32>, %[[PAST_K:.*]]: tensor<1x16x256x96xf32>, %[[PAST_V:.*]]: tensor<1x16x256x96xf32>) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>)
// CHECK-NOT:   "onnx.Attention"
// CHECK:       %[[GQA:.*]]:3 = "onnx.Custom"(%[[Q]], %[[K]], %[[V]], %[[PAST_K]], %[[PAST_V]], {{.*}}, {{.*}}) {domain_name = "com.microsoft", function_name = "GroupQueryAttention", k_quant_type = "PER_TENSOR", kv_num_heads = 16 : si64, num_heads = 32 : si64}
// CHECK-SAME:      (tensor<1x1x3072xf32>, tensor<1x1x1536xf32>, tensor<1x1x1536xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>, tensor<1x1xi32>, tensor<i32>) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>)
// CHECK:       return %[[GQA]]#0, %[[GQA]]#1, %[[GQA]]#2 : tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>

// -----

func.func @gqa_qk_norm_rejected(
  %q: tensor<1x1x3072xf32>,
  %k: tensor<1x1x1536xf32>,
  %v: tensor<1x1x1536xf32>,
  %past_k: tensor<1x16x256x96xf32>,
  %past_v: tensor<1x16x256x96xf32>,
  %q_norm_weight: tensor<96xf32>,
  %k_norm_weight: tensor<96xf32>
) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>) {
  %none = "onnx.NoValue"() {value} : () -> none
  %total_seqlen = "onnx.Constant"() {value = dense<257> : tensor<i32>} : () -> tensor<i32>
  %seqlens = "onnx.Constant"() {value = dense<256> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  %out, %present_k, %present_v = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %seqlens, %total_seqlen, %none, %none, %none, %none, %none, %none, %none, %q_norm_weight, %k_norm_weight) {
    domain_name = "com.microsoft",
    function_name = "GroupQueryAttention",
    kv_num_heads = 16 : si64,
    num_heads = 32 : si64
  } : (tensor<1x1x3072xf32>, tensor<1x1x1536xf32>, tensor<1x1x1536xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>, tensor<1x1xi32>, tensor<i32>, none, none, none, none, none, none, none, tensor<96xf32>, tensor<96xf32>) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>)
  return %out, %present_k, %present_v : tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>
}

// CHECK-LABEL: func.func @gqa_qk_norm_rejected
// CHECK-SAME:  (%[[Q:.*]]: tensor<1x1x3072xf32>, %[[K:.*]]: tensor<1x1x1536xf32>, %[[V:.*]]: tensor<1x1x1536xf32>, %[[PAST_K:.*]]: tensor<1x16x256x96xf32>, %[[PAST_V:.*]]: tensor<1x16x256x96xf32>, %[[Q_NORM:.*]]: tensor<96xf32>, %[[K_NORM:.*]]: tensor<96xf32>) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>)
// CHECK-NOT:   "onnx.Attention"
// CHECK:       %[[GQA:.*]]:3 = "onnx.Custom"(%[[Q]], %[[K]], %[[V]], %[[PAST_K]], %[[PAST_V]], {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[Q_NORM]], %[[K_NORM]]) {domain_name = "com.microsoft", function_name = "GroupQueryAttention", kv_num_heads = 16 : si64, num_heads = 32 : si64}
// CHECK-SAME:      (tensor<1x1x3072xf32>, tensor<1x1x1536xf32>, tensor<1x1x1536xf32>, tensor<1x16x256x96xf32>, tensor<1x16x256x96xf32>, tensor<1x1xi32>, tensor<i32>, none, none, none, none, none, none, none, tensor<96xf32>, tensor<96xf32>) -> (tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>)
// CHECK:       return %[[GQA]]#0, %[[GQA]]#1, %[[GQA]]#2 : tensor<1x1x3072xf32>, tensor<1x16x257x96xf32>, tensor<1x16x257x96xf32>

// -----

func.func @rotary_embedding_4d_interleaved_rotdim_16(%data: tensor<1x32x128x96xf32>, %pos_ids: tensor<1x128xi64>, %cos_cache: tensor<4096x8xf32>, %sin_cache: tensor<4096x8xf32>) -> tensor<1x32x128x96xf32> {
  %0 = "onnx.Custom"(%data, %pos_ids, %cos_cache, %sin_cache) {
    domain_name = "com.microsoft",
    function_name = "RotaryEmbedding",
    interleaved = 1 : si64,
    rotary_embedding_dim = 16 : si64
  }: (tensor<1x32x128x96xf32>, tensor<1x128xi64>, tensor<4096x8xf32>, tensor<4096x8xf32>) -> tensor<1x32x128x96xf32>
  return %0 : tensor<1x32x128x96xf32>
}

// CHECK-LABEL:   func.func @rotary_embedding_4d_interleaved_rotdim_16(
// CHECK-SAME:                                                     %[[VAL_0:.*]]: tensor<1x32x128x96xf32>,
// CHECK-SAME:                                                     %[[VAL_1:.*]]: tensor<1x128xi64>,
// CHECK-SAME:                                                     %[[VAL_2:.*]]: tensor<4096x8xf32>,
// CHECK-SAME:                                                     %[[VAL_3:.*]]: tensor<4096x8xf32>) -> tensor<1x32x128x96xf32> {
// CHECK:           %[[VAL_4:.*]] = "onnx.RotaryEmbedding"(%[[VAL_0]], %[[VAL_2]], %[[VAL_3]], %[[VAL_1]])
// CHECK-SAME:          {interleaved = 1 : si64, rotary_embedding_dim = 16 : si64}
// CHECK-SAME:          : (tensor<1x32x128x96xf32>, tensor<4096x8xf32>, tensor<4096x8xf32>, tensor<1x128xi64>) -> tensor<1x32x128x96xf32>
// CHECK:           return %[[VAL_4]] : tensor<1x32x128x96xf32>
// CHECK:         }

// -----

func.func @test_rotary_embedding_3d(%data: tensor<1x128x3072xf32>, %pos_ids: tensor<1x128xi64>, %cos_cache: tensor<4096x48xf32>, %sin_cache: tensor<4096x48xf32>) -> tensor<1x128x3072xf32> {
  %0 = "onnx.Custom"(%data, %pos_ids, %cos_cache, %sin_cache) {
    domain_name = "com.microsoft",
    function_name = "RotaryEmbedding",
    num_heads = 32: si64
  } : (tensor<1x128x3072xf32>, tensor<1x128xi64>, tensor<4096x48xf32>, tensor<4096x48xf32>) -> tensor<1x128x3072xf32>
  return %0 : tensor<1x128x3072xf32>
}

// CHECK-LABEL:   func.func @test_rotary_embedding_3d(
// CHECK-SAME:                                        %[[VAL_0:.*]]: tensor<1x128x3072xf32>,
// CHECK-SAME:                                        %[[VAL_1:.*]]: tensor<1x128xi64>,
// CHECK-SAME:                                        %[[VAL_2:.*]]: tensor<4096x48xf32>,
// CHECK-SAME:                                        %[[VAL_3:.*]]: tensor<4096x48xf32>) -> tensor<1x128x3072xf32> {
// CHECK:           %[[VAL_4:.*]] = "onnx.RotaryEmbedding"(%[[VAL_0]], %[[VAL_2]], %[[VAL_3]], %[[VAL_1]])
// CHECK-SAME:          {interleaved = 0 : si64, num_heads = 32 : si64, rotary_embedding_dim = 0 : si64}
// CHECK-SAME:          : (tensor<1x128x3072xf32>, tensor<4096x48xf32>, tensor<4096x48xf32>, tensor<1x128xi64>) -> tensor<1x128x3072xf32>
// CHECK:           return %[[VAL_4]] : tensor<1x128x3072xf32>
// CHECK:         }
