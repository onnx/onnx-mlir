// RUN: onnx-mlir-opt --maccel=NNPA --shape-inference --rewrite-onnx-for-zhigh %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --maccel=NNPA --rewrite-onnx-for-zhigh --shape-inference --canonicalize --constprop-onnx --shape-inference %s --split-input-file | FileCheck --check-prefix=MATMUL %s

func.func @test_batchnorm_epsilon(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<3xf32>, %arg2: tensor<3xf32>, %arg3: tensor<3xf32>, %arg4: tensor<3xf32>) -> tensor<2x3x4x5xf32> {
  %0 = "onnx.BatchNormalizationInferenceMode"(%arg0, %arg1, %arg2, %arg3, %arg4) {epsilon = 0.00999999977 : f32} : (tensor<2x3x4x5xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<2x3x4x5xf32>
  return %0 : tensor<2x3x4x5xf32>

// CHECK-LABEL:  func @test_batchnorm_epsilon
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4x5xf32>, [[PARAM_1_:%.+]]: tensor<3xf32>, [[PARAM_2_:%.+]]: tensor<3xf32>, [[PARAM_3_:%.+]]: tensor<3xf32>, [[PARAM_4_:%.+]]: tensor<3xf32>) -> tensor<2x3x4x5xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Constant"() {value = dense<0.00999999977> : tensor<1xf32>} : () -> tensor<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Shape"([[PARAM_4_]]) : (tensor<3xf32>) -> tensor<1xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Expand"([[VAR_0_]], [[VAR_1_]]) : (tensor<1xf32>, tensor<1xi64>) -> tensor<3xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Add"([[PARAM_4_]], [[VAR_2_]]) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Sqrt"([[VAR_3_]]) : (tensor<3xf32>) -> tensor<3xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Div"([[PARAM_1_]], [[VAR_4_]]) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Mul"([[PARAM_3_]], [[VAR_5_]]) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Sub"([[PARAM_2_]], [[VAR_6_]]) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_9_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "NHWC"} : (tensor<2x3x4x5xf32>) -> tensor<2x4x5x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK-DAG:       [[VAR_10_:%.+]] = "zhigh.Stick"([[VAR_5_]]) {layout = "1D"} : (tensor<3xf32>) -> tensor<3xf32, #zhigh.encoding<{dataLayout = "1D"}>>
// CHECK-DAG:       [[VAR_11_:%.+]] = "zhigh.Stick"([[VAR_7_]]) {layout = "1D"} : (tensor<3xf32>) -> tensor<3xf32, #zhigh.encoding<{dataLayout = "1D"}>>
// CHECK:           [[VAR_12_:%.+]] = "zhigh.BatchNorm"([[VAR_9_]], [[VAR_10_]], [[VAR_11_]]) : (tensor<2x4x5x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<3xf32, #zhigh.encoding<{dataLayout = "1D"}>>, tensor<3xf32, #zhigh.encoding<{dataLayout = "1D"}>>) -> tensor<2x4x5x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_13_:%.+]] = "zhigh.Unstick"([[VAR_12_]]) : (tensor<2x4x5x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<2x3x4x5xf32>
// CHECK:           return [[VAR_13_]] : tensor<2x3x4x5xf32>
// CHECK:         }
}

// -----

func.func @test_batchnorm_5d_not_lowered(%arg0: tensor<2x3x4x5x6xf32>, %arg1: tensor<3xf32>, %arg2: tensor<3xf32>, %arg3: tensor<3xf32>, %arg4: tensor<3xf32>) -> tensor<2x3x4x5x6xf32> {
  %0 = "onnx.BatchNormalizationInferenceMode"(%arg0, %arg1, %arg2, %arg3, %arg4) {epsilon = 0.00999999977 : f32} : (tensor<2x3x4x5x6xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<2x3x4x5x6xf32>
  return %0 : tensor<2x3x4x5x6xf32>

// CHECK-LABEL: test_batchnorm_5d_not_lowered
// CHECK: "onnx.BatchNormalizationInferenceMode"
}

// -----

func.func @test_add_expand_constant_lhs(%arg0: tensor<128x256xf32>) -> (tensor<128x256xf32>) {
  %cst = "onnx.Constant"() {value = dense<[1.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  %0 = "onnx.Add"(%cst, %arg0) : (tensor<1xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>

// CHECK-LABEL:  func.func @test_add_expand_constant_lhs
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Shape"([[PARAM_0_]]) : (tensor<128x256xf32>) -> tensor<2xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Expand"([[VAR_0_]], [[VAR_1_]]) : (tensor<1xf32>, tensor<2xi64>) -> tensor<128x256xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Add"([[PARAM_0_]], [[VAR_2_]]) : (tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
// CHECK:           return [[VAR_3_]] : tensor<128x256xf32>
// CHECK:         }
}

// -----

func.func @test_add_expand_constant_rhs(%arg0: tensor<128x256xf32>) -> (tensor<128x256xf32>) {
  %cst = "onnx.Constant"() {value = dense<[1.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  %0 = "onnx.Add"(%arg0, %cst) : (tensor<128x256xf32>, tensor<1xf32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>

// CHECK-LABEL:  func.func @test_add_expand_constant_rhs
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Shape"([[PARAM_0_]]) : (tensor<128x256xf32>) -> tensor<2xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Expand"([[VAR_0_]], [[VAR_1_]]) : (tensor<1xf32>, tensor<2xi64>) -> tensor<128x256xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Add"([[PARAM_0_]], [[VAR_2_]]) : (tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
// CHECK:           return [[VAR_3_]] : tensor<128x256xf32>
// CHECK:         }
}

// -----

func.func @test_add_expand_constant_scalar(%arg0: tensor<128x256xf32>) -> (tensor<128x256xf32>) {
  %cst = "onnx.Constant"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
  %0 = "onnx.Add"(%arg0, %cst) : (tensor<128x256xf32>, tensor<f32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>

// CHECK-LABEL:  func.func @test_add_expand_constant_scalar
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Shape"([[PARAM_0_]]) : (tensor<128x256xf32>) -> tensor<2xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Expand"([[VAR_0_]], [[VAR_1_]]) : (tensor<f32>, tensor<2xi64>) -> tensor<128x256xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Add"([[PARAM_0_]], [[VAR_2_]]) : (tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
// CHECK:           return [[VAR_3_]] : tensor<128x256xf32>
// CHECK:         }
}

// -----

func.func @test_add_block_arg(%arg0: tensor<128x256xf32>, %arg1: tensor<1xf32>) -> (tensor<128x256xf32>) {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<128x256xf32>, tensor<1xf32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>

// CHECK-LABEL:  func.func @test_add_block_arg
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x256xf32>, [[PARAM_1_:%.+]]: tensor<1xf32>) -> tensor<128x256xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<128x256xf32>, tensor<1xf32>) -> tensor<128x256xf32>
// CHECK:           return [[VAR_0_]] : tensor<128x256xf32>
// CHECK:         }
}

// -----

func.func @test_add_dynamic_dims(%arg0: tensor<128x?xf32>) -> (tensor<128x2xf32>) {
  %cst = "onnx.Constant"() {value = dense<1.0> : tensor<2xf32>} : () -> tensor<2xf32>
  %0 = "onnx.Add"(%arg0, %cst) : (tensor<128x?xf32>, tensor<2xf32>) -> tensor<128x2xf32>
  return %0 : tensor<128x2xf32>

// CHECK-LABEL:  func.func @test_add_dynamic_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x?xf32>) -> tensor<128x2xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Add"({{.*}}, {{.*}}) : (tensor<128x?xf32>, tensor<2xf32>) -> tensor<128x2xf32>
// CHECK:           return [[VAR_0_]] : tensor<128x2xf32>
// CHECK:         }
}

// -----

func.func @test_div_expand_constant_lhs(%arg0: tensor<128x256xf32>) -> (tensor<128x256xf32>) {
  %cst = "onnx.Constant"() {value = dense<[1.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  %0 = "onnx.Div"(%cst, %arg0) : (tensor<1xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>

// CHECK-LABEL:  func.func @test_div_expand_constant_lhs
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Shape"([[PARAM_0_]]) : (tensor<128x256xf32>) -> tensor<2xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Expand"([[VAR_0_]], [[VAR_1_]]) : (tensor<1xf32>, tensor<2xi64>) -> tensor<128x256xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Div"([[VAR_2_]], [[PARAM_0_]]) : (tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
// CHECK:           return [[VAR_3_]] : tensor<128x256xf32>
// CHECK:         }
}

// -----

func.func @test_div_expand_constant_rhs(%arg0: tensor<128x256xf32>) -> (tensor<128x256xf32>) {
  %cst = "onnx.Constant"() {value = dense<[1.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  %0 = "onnx.Div"(%arg0, %cst) : (tensor<128x256xf32>, tensor<1xf32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>

// CHECK-LABEL:  func.func @test_div_expand_constant_rhs
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Shape"([[PARAM_0_]]) : (tensor<128x256xf32>) -> tensor<2xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Expand"([[VAR_0_]], [[VAR_1_]]) : (tensor<1xf32>, tensor<2xi64>) -> tensor<128x256xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Div"([[PARAM_0_]], [[VAR_2_]]) : (tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
// CHECK:           return [[VAR_3_]] : tensor<128x256xf32>
// CHECK:         }
}

// -----

func.func @test_div_expand_constant_scalar(%arg0: tensor<128x256xf32>) -> (tensor<128x256xf32>) {
  %cst = "onnx.Constant"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
  %0 = "onnx.Div"(%arg0, %cst) : (tensor<128x256xf32>, tensor<f32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>

// CHECK-LABEL:  func.func @test_div_expand_constant_scalar
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Shape"([[PARAM_0_]]) : (tensor<128x256xf32>) -> tensor<2xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Expand"([[VAR_0_]], [[VAR_1_]]) : (tensor<f32>, tensor<2xi64>) -> tensor<128x256xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Div"([[PARAM_0_]], [[VAR_2_]]) : (tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
// CHECK:           return [[VAR_3_]] : tensor<128x256xf32>
// CHECK:         }
}

// -----

func.func @test_div_block_arg(%arg0: tensor<128x256xf32>, %arg1: tensor<1xf32>) -> (tensor<128x256xf32>) {
  %0 = "onnx.Div"(%arg0, %arg1) : (tensor<128x256xf32>, tensor<1xf32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>

// CHECK-LABEL:  func.func @test_div_block_arg
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x256xf32>, [[PARAM_1_:%.+]]: tensor<1xf32>) -> tensor<128x256xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Div"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<128x256xf32>, tensor<1xf32>) -> tensor<128x256xf32>
// CHECK:           return [[VAR_0_]] : tensor<128x256xf32>
// CHECK:         }
}

// -----

func.func @test_div_dynamic_dims(%arg0: tensor<128x?xf32>) -> (tensor<128x2xf32>) {
  %cst = "onnx.Constant"() {value = dense<1.0> : tensor<2xf32>} : () -> tensor<2xf32>
  %0 = "onnx.Div"(%arg0, %cst) : (tensor<128x?xf32>, tensor<2xf32>) -> tensor<128x2xf32>
  return %0 : tensor<128x2xf32>

// CHECK-LABEL:  func.func @test_div_dynamic_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x?xf32>) -> tensor<128x2xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Div"({{.*}}, {{.*}}) : (tensor<128x?xf32>, tensor<2xf32>) -> tensor<128x2xf32>
// CHECK:           return [[VAR_0_]] : tensor<128x2xf32>
// CHECK:         }
}

// -----

func.func @test_mul_expand_constant_lhs(%arg0: tensor<128x256xf32>) -> (tensor<128x256xf32>) {
  %cst = "onnx.Constant"() {value = dense<[1.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  %0 = "onnx.Mul"(%cst, %arg0) : (tensor<1xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>

// CHECK-LABEL:  func.func @test_mul_expand_constant_lhs
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Shape"([[PARAM_0_]]) : (tensor<128x256xf32>) -> tensor<2xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Expand"([[VAR_0_]], [[VAR_1_]]) : (tensor<1xf32>, tensor<2xi64>) -> tensor<128x256xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[VAR_2_]]) : (tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
// CHECK:           return [[VAR_3_]] : tensor<128x256xf32>
// CHECK:         }
}

// -----

func.func @test_mul_expand_constant_rhs(%arg0: tensor<128x256xf32>) -> (tensor<128x256xf32>) {
  %cst = "onnx.Constant"() {value = dense<[1.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  %0 = "onnx.Mul"(%arg0, %cst) : (tensor<128x256xf32>, tensor<1xf32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>

// CHECK-LABEL:  func.func @test_mul_expand_constant_rhs
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Shape"([[PARAM_0_]]) : (tensor<128x256xf32>) -> tensor<2xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Expand"([[VAR_0_]], [[VAR_1_]]) : (tensor<1xf32>, tensor<2xi64>) -> tensor<128x256xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[VAR_2_]]) : (tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
// CHECK:           return [[VAR_3_]] : tensor<128x256xf32>
// CHECK:         }
}

// -----

func.func @test_mul_expand_constant_scalar(%arg0: tensor<128x256xf32>) -> (tensor<128x256xf32>) {
  %cst = "onnx.Constant"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
  %0 = "onnx.Mul"(%arg0, %cst) : (tensor<128x256xf32>, tensor<f32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>

// CHECK-LABEL:  func.func @test_mul_expand_constant_scalar
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Shape"([[PARAM_0_]]) : (tensor<128x256xf32>) -> tensor<2xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Expand"([[VAR_0_]], [[VAR_1_]]) : (tensor<f32>, tensor<2xi64>) -> tensor<128x256xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[VAR_2_]]) : (tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
// CHECK:           return [[VAR_3_]] : tensor<128x256xf32>
// CHECK:         }
}

// -----

func.func @test_mul_block_arg(%arg0: tensor<128x256xf32>, %arg1: tensor<1xf32>) -> (tensor<128x256xf32>) {
  %0 = "onnx.Mul"(%arg0, %arg1) : (tensor<128x256xf32>, tensor<1xf32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>

// CHECK-LABEL:  func.func @test_mul_block_arg
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x256xf32>, [[PARAM_1_:%.+]]: tensor<1xf32>) -> tensor<128x256xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<128x256xf32>, tensor<1xf32>) -> tensor<128x256xf32>
// CHECK:           return [[VAR_0_]] : tensor<128x256xf32>
// CHECK:         }
}

// -----

func.func @test_mul_dynamic_dims(%arg0: tensor<128x?xf32>) -> (tensor<128x2xf32>) {
  %cst = "onnx.Constant"() {value = dense<1.0> : tensor<2xf32>} : () -> tensor<2xf32>
  %0 = "onnx.Mul"(%arg0, %cst) : (tensor<128x?xf32>, tensor<2xf32>) -> tensor<128x2xf32>
  return %0 : tensor<128x2xf32>

// CHECK-LABEL:  func.func @test_mul_dynamic_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x?xf32>) -> tensor<128x2xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Mul"({{.*}}, {{.*}}) : (tensor<128x?xf32>, tensor<2xf32>) -> tensor<128x2xf32>
// CHECK:           return [[VAR_0_]] : tensor<128x2xf32>
// CHECK:         }
}

// -----

func.func @test_sub_expand_constant_lhs(%arg0: tensor<128x256xf32>) -> (tensor<128x256xf32>) {
  %cst = "onnx.Constant"() {value = dense<[1.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  %0 = "onnx.Sub"(%cst, %arg0) : (tensor<1xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>

// CHECK-LABEL:  func.func @test_sub_expand_constant_lhs
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Shape"([[PARAM_0_]]) : (tensor<128x256xf32>) -> tensor<2xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Expand"([[VAR_0_]], [[VAR_1_]]) : (tensor<1xf32>, tensor<2xi64>) -> tensor<128x256xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Sub"([[VAR_2_]], [[PARAM_0_]]) : (tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
// CHECK:           return [[VAR_3_]] : tensor<128x256xf32>
// CHECK:         }
}

// -----

func.func @test_sub_expand_constant_rhs(%arg0: tensor<128x256xf32>) -> (tensor<128x256xf32>) {
  %cst = "onnx.Constant"() {value = dense<[1.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  %0 = "onnx.Sub"(%arg0, %cst) : (tensor<128x256xf32>, tensor<1xf32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>

// CHECK-LABEL:  func.func @test_sub_expand_constant_rhs
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Shape"([[PARAM_0_]]) : (tensor<128x256xf32>) -> tensor<2xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Expand"([[VAR_0_]], [[VAR_1_]]) : (tensor<1xf32>, tensor<2xi64>) -> tensor<128x256xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Sub"([[PARAM_0_]], [[VAR_2_]]) : (tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
// CHECK:           return [[VAR_3_]] : tensor<128x256xf32>
// CHECK:         }
}

// -----

func.func @test_sub_expand_constant_scalar(%arg0: tensor<128x256xf32>) -> (tensor<128x256xf32>) {
  %cst = "onnx.Constant"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
  %0 = "onnx.Sub"(%arg0, %cst) : (tensor<128x256xf32>, tensor<f32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>

// CHECK-LABEL:  func.func @test_sub_expand_constant_scalar
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Shape"([[PARAM_0_]]) : (tensor<128x256xf32>) -> tensor<2xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Expand"([[VAR_0_]], [[VAR_1_]]) : (tensor<f32>, tensor<2xi64>) -> tensor<128x256xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Sub"([[PARAM_0_]], [[VAR_2_]]) : (tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
// CHECK:           return [[VAR_3_]] : tensor<128x256xf32>
// CHECK:         }
}

// -----

func.func @test_sub_block_arg(%arg0: tensor<128x256xf32>, %arg1: tensor<1xf32>) -> (tensor<128x256xf32>) {
  %0 = "onnx.Sub"(%arg0, %arg1) : (tensor<128x256xf32>, tensor<1xf32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>

// CHECK-LABEL:  func.func @test_sub_block_arg
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x256xf32>, [[PARAM_1_:%.+]]: tensor<1xf32>) -> tensor<128x256xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Sub"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<128x256xf32>, tensor<1xf32>) -> tensor<128x256xf32>
// CHECK:           return [[VAR_0_]] : tensor<128x256xf32>
// CHECK:         }
}

// -----

func.func @test_sub_dynamic_dims(%arg0: tensor<128x?xf32>) -> (tensor<128x2xf32>) {
  %cst = "onnx.Constant"() {value = dense<1.0> : tensor<2xf32>} : () -> tensor<2xf32>
  %0 = "onnx.Sub"(%arg0, %cst) : (tensor<128x?xf32>, tensor<2xf32>) -> tensor<128x2xf32>
  return %0 : tensor<128x2xf32>

// CHECK-LABEL:  func.func @test_sub_dynamic_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x?xf32>) -> tensor<128x2xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Sub"({{.*}}, {{.*}}) : (tensor<128x?xf32>, tensor<2xf32>) -> tensor<128x2xf32>
// CHECK:           return [[VAR_0_]] : tensor<128x2xf32>
// CHECK:         }
}

// -----

func.func @test_matmul(%arg0: tensor<4x12x256x256xf32>, %arg1: tensor<4x12x256x64xf32>) -> (tensor<4x12x256x64xf32>) {
    %0= "onnx.MatMul"(%arg0, %arg1) : (tensor<4x12x256x256xf32>, tensor<4x12x256x64xf32>) -> tensor<4x12x256x64xf32>
    return %0 : tensor<4x12x256x64xf32>

// MATMUL-LABEL:  func.func @test_matmul
// MATMUL-SAME:   ([[PARAM_0_:%.+]]: tensor<4x12x256x256xf32>, [[PARAM_1_:%.+]]: tensor<4x12x256x64xf32>) -> tensor<4x12x256x64xf32> {
// MATMUL:           [[VAR_0_:%.+]] = "onnx.Constant"() {value = dense<[-1, 256, 256]> : tensor<3xi64>} : () -> tensor<3xi64>
// MATMUL-DAG:       [[VAR_1_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<4x12x256x256xf32>, tensor<3xi64>) -> tensor<48x256x256xf32>
// MATMUL-DAG:       [[VAR_2_:%.+]] = "onnx.Constant"() {value = dense<[-1, 256, 64]> : tensor<3xi64>} : () -> tensor<3xi64>
// MATMUL:           [[VAR_3_:%.+]] = "onnx.Reshape"([[PARAM_1_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<4x12x256x64xf32>, tensor<3xi64>) -> tensor<48x256x64xf32>
// MATMUL-DAG:       [[VAR_4_:%.+]] = "onnx.MatMul"([[VAR_1_]], [[VAR_3_]]) : (tensor<48x256x256xf32>, tensor<48x256x64xf32>) -> tensor<48x256x64xf32>
// MATMUL-DAG:       [[VAR_5_:%.+]] = "onnx.Constant"() {value = dense<[4, 12, 256, 64]> : tensor<4xi64>} : () -> tensor<4xi64>
// MATMUL:           [[VAR_6_:%.+]] = "onnx.Reshape"([[VAR_4_]], [[VAR_5_]]) {allowzero = 0 : si64} : (tensor<48x256x64xf32>, tensor<4xi64>) -> tensor<4x12x256x64xf32>
// MATMUL:           return [[VAR_6_]] : tensor<4x12x256x64xf32>
// MATMUL:         }
}

// -----

func.func @test_matmul_broadcast_1(%arg0: tensor<4x12x256x256xf32>, %arg1: tensor<256x64xf32>) -> (tensor<4x12x256x64xf32>) {
    %0= "onnx.MatMul"(%arg0, %arg1) : (tensor<4x12x256x256xf32>, tensor<256x64xf32>) -> tensor<4x12x256x64xf32>
    return %0 : tensor<4x12x256x64xf32>

// MATMUL-LABEL:  func.func @test_matmul_broadcast_1
// MATMUL-SAME:   ([[PARAM_0_:%.+]]: tensor<4x12x256x256xf32>, [[PARAM_1_:%.+]]: tensor<256x64xf32>) -> tensor<4x12x256x64xf32> {
// MATMUL:           [[VAR_0_:%.+]] = "onnx.Constant"() {value = dense<[-1, 256, 256]> : tensor<3xi64>} : () -> tensor<3xi64>
// MATMUL:           [[VAR_1_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<4x12x256x256xf32>, tensor<3xi64>) -> tensor<48x256x256xf32>
// MATMUL-DAG:       [[VAR_2_:%.+]] = "onnx.MatMul"([[VAR_1_]], [[PARAM_1_]]) : (tensor<48x256x256xf32>, tensor<256x64xf32>) -> tensor<48x256x64xf32>
// MATMUL-DAG:       [[VAR_3_:%.+]] = "onnx.Constant"() {value = dense<[4, 12, 256, 64]> : tensor<4xi64>} : () -> tensor<4xi64>
// MATMUL:           [[VAR_4_:%.+]] = "onnx.Reshape"([[VAR_2_]], [[VAR_3_]]) {allowzero = 0 : si64} : (tensor<48x256x64xf32>, tensor<4xi64>) -> tensor<4x12x256x64xf32>
// MATMUL:           return [[VAR_4_]] : tensor<4x12x256x64xf32>
// MATMUL:         }
}

// -----

func.func @test_matmul_broadcast_2(%arg0: tensor<256x256xf32>, %arg1: tensor<4x12x256x64xf32>) -> (tensor<4x12x256x64xf32>) {
    %0= "onnx.MatMul"(%arg0, %arg1) : (tensor<256x256xf32>, tensor<4x12x256x64xf32>) -> tensor<4x12x256x64xf32>
    return %0 : tensor<4x12x256x64xf32>

// MATMUL-LABEL:  func.func @test_matmul_broadcast_2
// MATMUL-SAME:   ([[PARAM_0_:%.+]]: tensor<256x256xf32>, [[PARAM_1_:%.+]]: tensor<4x12x256x64xf32>) -> tensor<4x12x256x64xf32> {
// MATMUL:           [[VAR_0_:%.+]] = "onnx.Constant"() {value = dense<[-1, 256, 64]> : tensor<3xi64>} : () -> tensor<3xi64>
// MATMUL:           [[VAR_1_:%.+]] = "onnx.Reshape"([[PARAM_1_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<4x12x256x64xf32>, tensor<3xi64>) -> tensor<48x256x64xf32>
// MATMUL-DAG:       [[VAR_2_:%.+]] = "onnx.MatMul"([[PARAM_0_]], [[VAR_1_]]) : (tensor<256x256xf32>, tensor<48x256x64xf32>) -> tensor<48x256x64xf32>
// MATMUL-DAG:       [[VAR_3_:%.+]] = "onnx.Constant"() {value = dense<[4, 12, 256, 64]> : tensor<4xi64>} : () -> tensor<4xi64>
// MATMUL:           [[VAR_4_:%.+]] = "onnx.Reshape"([[VAR_2_]], [[VAR_3_]]) {allowzero = 0 : si64} : (tensor<48x256x64xf32>, tensor<4xi64>) -> tensor<4x12x256x64xf32>
// MATMUL:           return [[VAR_4_]] : tensor<4x12x256x64xf32>
// MATMUL:         }
}

// -----

func.func @test_matmul_broadcast_dyn_dims(%arg0: tensor<256x?xf32>, %arg1: tensor<4x12x?x?xf32>) -> (tensor<4x12x256x?xf32>) {
    %0= "onnx.MatMul"(%arg0, %arg1) : (tensor<256x?xf32>, tensor<4x12x?x?xf32>) -> tensor<4x12x256x?xf32>
    return %0 : tensor<4x12x256x?xf32>

// MATMUL-LABEL:  func.func @test_matmul_broadcast_dyn_dims
// MATMUL-SAME:   ([[PARAM_0_:%.+]]: tensor<256x?xf32>, [[PARAM_1_:%.+]]: tensor<4x12x?x?xf32>) -> tensor<?x?x?x?xf32> {
// MATMUL-DAG:       [[VAR_0_:%.+]] = "onnx.Shape"([[PARAM_1_]]) : (tensor<4x12x?x?xf32>) -> tensor<4xi64>
// MATMUL-DAG:       [[VAR_1_:%.+]] = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
// MATMUL-DAG:       [[VAR_2_:%.+]] = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
// MATMUL-DAG:       [[VAR_3_:%.+]] = "onnx.Constant"() {value = dense<-1> : tensor<1xi64>} : () -> tensor<1xi64>
// MATMUL-DAG:       [[VAR_4_:%.+]] = "onnx.Constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
// MATMUL-DAG:       [[VAR_5_:%.+]] = "onnx.Constant"() {value = dense<4> : tensor<1xi64>} : () -> tensor<1xi64>
// MATMUL:           [[VAR_6_:%.+]] = "onnx.Slice"([[VAR_0_]], [[VAR_4_]], [[VAR_5_]], [[VAR_1_]], [[VAR_2_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// MATMUL:           [[VAR_7_:%.+]] = "onnx.Concat"([[VAR_3_]], [[VAR_6_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<2xi64>) -> tensor<3xi64>
// MATMUL:           [[VAR_8_:%.+]] = "onnx.Reshape"([[PARAM_1_]], [[VAR_7_]]) {allowzero = 0 : si64} : (tensor<4x12x?x?xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
// MATMUL-DAG:       [[VAR_9_:%.+]] = "onnx.MatMul"([[PARAM_0_]], [[VAR_8_]]) : (tensor<256x?xf32>, tensor<?x?x?xf32>) -> tensor<?x256x?xf32>
// MATMUL-DAG:       [[VAR_10_:%.+]] = "onnx.Shape"([[PARAM_0_]]) : (tensor<256x?xf32>) -> tensor<2xi64>
// MATMUL-DAG:       [[VAR_11_:%.+]] = "onnx.Shape"([[PARAM_1_]]) : (tensor<4x12x?x?xf32>) -> tensor<4xi64>
// MATMUL-DAG:       [[VAR_12_:%.+]] = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
// MATMUL-DAG:       [[VAR_13_:%.+]] = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
// MATMUL-DAG:       [[VAR_14_:%.+]] = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
// MATMUL-DAG:       [[VAR_15_:%.+]] = "onnx.Constant"() {value = dense<4> : tensor<1xi64>} : () -> tensor<1xi64>
// MATMUL-DAG:       [[VAR_16_:%.+]] = "onnx.Constant"() {value = dense<3> : tensor<1xi64>} : () -> tensor<1xi64>
// MATMUL-DAG:       [[VAR_17_:%.+]] = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
// MATMUL-DAG:       [[VAR_18_:%.+]] = "onnx.Constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
// MATMUL-NOT: separator of consecutive DAGs
// MATMUL-DAG:       [[VAR_19_:%.+]] = "onnx.Slice"([[VAR_11_]], [[VAR_12_]], [[VAR_18_]], [[VAR_12_]], [[VAR_13_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// MATMUL-DAG:       [[VAR_20_:%.+]] = "onnx.Slice"([[VAR_10_]], [[VAR_17_]], [[VAR_14_]], [[VAR_12_]], [[VAR_13_]]) : (tensor<2xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// MATMUL-DAG:       [[VAR_21_:%.+]] = "onnx.Slice"([[VAR_11_]], [[VAR_16_]], [[VAR_15_]], [[VAR_12_]], [[VAR_13_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// MATMUL:           [[VAR_22_:%.+]] = "onnx.Concat"([[VAR_19_]], [[VAR_20_]], [[VAR_21_]]) {axis = 0 : si64} : (tensor<2xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
// MATMUL:           [[VAR_23_:%.+]] = "onnx.Reshape"([[VAR_9_]], [[VAR_22_]]) {allowzero = 0 : si64} : (tensor<?x256x?xf32>, tensor<4xi64>) -> tensor<?x?x?x?xf32>
// MATMUL:           return [[VAR_23_]] : tensor<?x?x?x?xf32>
// MATMUL:         }
}
