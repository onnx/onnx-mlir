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