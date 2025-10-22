// RUN: onnx-mlir-opt --decompose-onnx %s -split-input-file | FileCheck %s

// COM: Decompose CustomOp introduced by onnxruntime.

func.func @customop_fusedmatmul_onnxruntime(%arg0: tensor<3x5x7x9xf32>, %arg1:tensor<3x5x7x9xf32>) -> tensor<3x5x9x9xf32> {
    %0 = "onnx.Custom"(%arg0, %arg1) {alpha = 1.250000e-01 : f32, domain_name = "com.microsoft", function_name = "FusedMatMul", transA = 1 : si64, transB = 0 : si64} : (tensor<3x5x7x9xf32>, tensor<3x5x7x9xf32>) -> tensor<3x5x9x9xf32>
    onnx.Return %0: tensor<3x5x9x9xf32>

// CHECK-LABEL:  func.func @customop_fusedmatmul_onnxruntime
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x5x7x9xf32>, [[PARAM_1_:%.+]]: tensor<3x5x7x9xf32>) -> tensor<3x5x9x9xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Transpose"([[PARAM_0_]]) {perm = [0, 1, 3, 2]} : (tensor<3x5x7x9xf32>) -> tensor<3x5x9x7xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1.250000e-01> : tensor<1xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.MatMul"([[VAR_0_]], [[PARAM_1_]]) : (tensor<3x5x9x7xf32>, tensor<3x5x7x9xf32>) -> tensor<3x5x9x9xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Mul"([[VAR_1_]], [[VAR_2_]]) : (tensor<1xf32>, tensor<3x5x9x9xf32>) -> tensor<3x5x9x9xf32>
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
// CHECK:           [[VAR_2_:%.+]] = "onnx.Mul"([[VAR_0_]], [[VAR_1_]]) : (tensor<1xf32>, tensor<*xf32>) -> tensor<*xf32>
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
// CHECK:           [[VAR_4_:%.+]] = "onnx.Mul"([[VAR_0_]], [[VAR_3_]]) : (tensor<1xf32>, tensor<*xf32>) -> tensor<*xf32>
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
// CHECK:           [[VAR_4_:%.+]] = "onnx.Mul"([[VAR_0_]], [[VAR_3_]]) : (tensor<1xf32>, tensor<*xf32>) -> tensor<*xf32>
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

func.func @skip_layernorm_beta(%input: tensor<2x4x8xf32>, %skip: tensor<2x4x8xf32>, %gamma: tensor<8xf32>, %beta: tensor<8xf32>) -> tensor<2x4x8xf32> {
  %r = "onnx.Custom"(%input, %skip, %gamma, %beta) {domain_name = "com.microsoft", function_name = "SkipLayerNormalization", epsilon = 1.000000e-05 : f32} : (tensor<2x4x8xf32>, tensor<2x4x8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<2x4x8xf32>
  onnx.Return %r : tensor<2x4x8xf32>
// CHECK-LABEL:  func.func @skip_layernorm_beta
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

