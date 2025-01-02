// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --shape-inference --rewrite-onnx-for-zhigh %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --rewrite-onnx-for-zhigh --shape-inference --canonicalize --constprop-onnx  --shape-inference %s --split-input-file | FileCheck --check-prefix=CONSTPROP %s

func.func @test_batchnorm_epsilon(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<3xf32>, %arg2: tensor<3xf32>, %arg3: tensor<3xf32>, %arg4: tensor<3xf32>) -> tensor<2x3x4x5xf32> {
  %0 = "onnx.BatchNormalizationInferenceMode"(%arg0, %arg1, %arg2, %arg3, %arg4) {epsilon = 0.00999999977 : f32} : (tensor<2x3x4x5xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<2x3x4x5xf32>
  return %0 : tensor<2x3x4x5xf32>

// CHECK-LABEL:  func @test_batchnorm_epsilon
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4x5xf32>, [[PARAM_1_:%.+]]: tensor<3xf32>, [[PARAM_2_:%.+]]: tensor<3xf32>, [[PARAM_3_:%.+]]: tensor<3xf32>, [[PARAM_4_:%.+]]: tensor<3xf32>) -> tensor<2x3x4x5xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<0.00999999977> : tensor<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Shape"([[PARAM_4_]]) {start = 0 : si64} : (tensor<3xf32>) -> tensor<1xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Expand"([[VAR_0_]], [[VAR_1_]]) : (tensor<1xf32>, tensor<1xi64>) -> tensor<3xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Add"([[PARAM_4_]], [[VAR_2_]]) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Sqrt"([[VAR_3_]]) : (tensor<3xf32>) -> tensor<3xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Div"([[PARAM_1_]], [[VAR_4_]]) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Mul"([[PARAM_3_]], [[VAR_5_]]) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Sub"([[PARAM_2_]], [[VAR_6_]]) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_9_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "NHWC"} : (tensor<2x3x4x5xf32>) -> tensor<2x4x5x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK-DAG:       [[VAR_10_:%.+]] = "zhigh.Stick"([[VAR_5_]]) {layout = "1D"} : (tensor<3xf32>) -> tensor<3xf16, #zhigh.layout<{dataLayout = "1D"}>>
// CHECK-DAG:       [[VAR_11_:%.+]] = "zhigh.Stick"([[VAR_7_]]) {layout = "1D"} : (tensor<3xf32>) -> tensor<3xf16, #zhigh.layout<{dataLayout = "1D"}>>
// CHECK:           [[VAR_12_:%.+]] = "zhigh.BatchNorm"([[VAR_9_]], [[VAR_10_]], [[VAR_11_]]) : (tensor<2x4x5x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<3xf16, #zhigh.layout<{dataLayout = "1D"}>>, tensor<3xf16, #zhigh.layout<{dataLayout = "1D"}>>) -> tensor<2x4x5x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_13_:%.+]] = "zhigh.Unstick"([[VAR_12_]]) : (tensor<2x4x5x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<2x3x4x5xf32>
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
  %cst = onnx.Constant dense<[1.0]> : tensor<1xf32>
  %0 = "onnx.Add"(%cst, %arg0) : (tensor<1xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>

// CHECK-LABEL:  func.func @test_add_expand_constant_lhs
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Shape"([[PARAM_0_]]) {start = 0 : si64} : (tensor<128x256xf32>) -> tensor<2xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Expand"([[VAR_0_]], [[VAR_1_]]) : (tensor<1xf32>, tensor<2xi64>) -> tensor<128x256xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Add"([[PARAM_0_]], [[VAR_2_]]) : (tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
// CHECK:           return [[VAR_3_]] : tensor<128x256xf32>
// CHECK:         }
}

// -----

func.func @test_add_expand_constant_rhs(%arg0: tensor<128x256xf32>) -> (tensor<128x256xf32>) {
  %cst = onnx.Constant dense<[1.0]> : tensor<1xf32>
  %0 = "onnx.Add"(%arg0, %cst) : (tensor<128x256xf32>, tensor<1xf32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>

// CHECK-LABEL:  func.func @test_add_expand_constant_rhs
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Shape"([[PARAM_0_]]) {start = 0 : si64} : (tensor<128x256xf32>) -> tensor<2xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Expand"([[VAR_0_]], [[VAR_1_]]) : (tensor<1xf32>, tensor<2xi64>) -> tensor<128x256xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Add"([[PARAM_0_]], [[VAR_2_]]) : (tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
// CHECK:           return [[VAR_3_]] : tensor<128x256xf32>
// CHECK:         }
}

// -----

func.func @test_add_expand_constant_scalar(%arg0: tensor<128x256xf32>) -> (tensor<128x256xf32>) {
  %cst = onnx.Constant dense<1.0> : tensor<f32>
  %0 = "onnx.Add"(%arg0, %cst) : (tensor<128x256xf32>, tensor<f32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>

// CHECK-LABEL:  func.func @test_add_expand_constant_scalar
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Shape"([[PARAM_0_]]) {start = 0 : si64} : (tensor<128x256xf32>) -> tensor<2xi64>
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
  %cst = onnx.Constant dense<1.0> : tensor<2xf32>
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
  %cst = onnx.Constant dense<[1.0]> : tensor<1xf32>
  %0 = "onnx.Div"(%cst, %arg0) : (tensor<1xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>

// CHECK-LABEL:  func.func @test_div_expand_constant_lhs
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Shape"([[PARAM_0_]]) {start = 0 : si64} : (tensor<128x256xf32>) -> tensor<2xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Expand"([[VAR_0_]], [[VAR_1_]]) : (tensor<1xf32>, tensor<2xi64>) -> tensor<128x256xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Div"([[VAR_2_]], [[PARAM_0_]]) : (tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
// CHECK:           return [[VAR_3_]] : tensor<128x256xf32>
// CHECK:         }
}

// -----

func.func @test_div_expand_constant_rhs(%arg0: tensor<128x256xf32>) -> (tensor<128x256xf32>) {
  %cst = onnx.Constant dense<[1.0]> : tensor<1xf32>
  %0 = "onnx.Div"(%arg0, %cst) : (tensor<128x256xf32>, tensor<1xf32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>

// CHECK-LABEL:  func.func @test_div_expand_constant_rhs
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Shape"([[PARAM_0_]]) {start = 0 : si64} : (tensor<128x256xf32>) -> tensor<2xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Expand"([[VAR_0_]], [[VAR_1_]]) : (tensor<1xf32>, tensor<2xi64>) -> tensor<128x256xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Div"([[PARAM_0_]], [[VAR_2_]]) : (tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
// CHECK:           return [[VAR_3_]] : tensor<128x256xf32>
// CHECK:         }
}

// -----

func.func @test_div_expand_constant_scalar(%arg0: tensor<128x256xf32>) -> (tensor<128x256xf32>) {
  %cst = onnx.Constant dense<1.0> : tensor<f32>
  %0 = "onnx.Div"(%arg0, %cst) : (tensor<128x256xf32>, tensor<f32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>

// CHECK-LABEL:  func.func @test_div_expand_constant_scalar
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Shape"([[PARAM_0_]]) {start = 0 : si64} : (tensor<128x256xf32>) -> tensor<2xi64>
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
  %cst = onnx.Constant dense<1.0> : tensor<2xf32>
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
  %cst = onnx.Constant dense<[1.0]> : tensor<1xf32>
  %0 = "onnx.Mul"(%cst, %arg0) : (tensor<1xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>

// CHECK-LABEL:  func.func @test_mul_expand_constant_lhs
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Shape"([[PARAM_0_]]) {start = 0 : si64} : (tensor<128x256xf32>) -> tensor<2xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Expand"([[VAR_0_]], [[VAR_1_]]) : (tensor<1xf32>, tensor<2xi64>) -> tensor<128x256xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[VAR_2_]]) : (tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
// CHECK:           return [[VAR_3_]] : tensor<128x256xf32>
// CHECK:         }
}

// -----

func.func @test_mul_expand_constant_rhs(%arg0: tensor<128x256xf32>) -> (tensor<128x256xf32>) {
  %cst = onnx.Constant dense<[1.0]> : tensor<1xf32>
  %0 = "onnx.Mul"(%arg0, %cst) : (tensor<128x256xf32>, tensor<1xf32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>

// CHECK-LABEL:  func.func @test_mul_expand_constant_rhs
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Shape"([[PARAM_0_]]) {start = 0 : si64} : (tensor<128x256xf32>) -> tensor<2xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Expand"([[VAR_0_]], [[VAR_1_]]) : (tensor<1xf32>, tensor<2xi64>) -> tensor<128x256xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[VAR_2_]]) : (tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
// CHECK:           return [[VAR_3_]] : tensor<128x256xf32>
// CHECK:         }
}

// -----

func.func @test_mul_expand_constant_scalar(%arg0: tensor<128x256xf32>) -> (tensor<128x256xf32>) {
  %cst = onnx.Constant dense<1.0> : tensor<f32>
  %0 = "onnx.Mul"(%arg0, %cst) : (tensor<128x256xf32>, tensor<f32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>

// CHECK-LABEL:  func.func @test_mul_expand_constant_scalar
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Shape"([[PARAM_0_]]) {start = 0 : si64} : (tensor<128x256xf32>) -> tensor<2xi64>
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
  %cst = onnx.Constant dense<1.0> : tensor<2xf32>
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
  %cst = onnx.Constant dense<[1.0]> : tensor<1xf32>
  %0 = "onnx.Sub"(%cst, %arg0) : (tensor<1xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>

// CHECK-LABEL:  func.func @test_sub_expand_constant_lhs
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Shape"([[PARAM_0_]]) {start = 0 : si64} : (tensor<128x256xf32>) -> tensor<2xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Expand"([[VAR_0_]], [[VAR_1_]]) : (tensor<1xf32>, tensor<2xi64>) -> tensor<128x256xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Sub"([[VAR_2_]], [[PARAM_0_]]) : (tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
// CHECK:           return [[VAR_3_]] : tensor<128x256xf32>
// CHECK:         }
}

// -----

func.func @test_sub_expand_constant_rhs(%arg0: tensor<128x256xf32>) -> (tensor<128x256xf32>) {
  %cst = onnx.Constant dense<[1.0]> : tensor<1xf32>
  %0 = "onnx.Sub"(%arg0, %cst) : (tensor<128x256xf32>, tensor<1xf32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>

// CHECK-LABEL:  func.func @test_sub_expand_constant_rhs
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Shape"([[PARAM_0_]]) {start = 0 : si64} : (tensor<128x256xf32>) -> tensor<2xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Expand"([[VAR_0_]], [[VAR_1_]]) : (tensor<1xf32>, tensor<2xi64>) -> tensor<128x256xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Sub"([[PARAM_0_]], [[VAR_2_]]) : (tensor<128x256xf32>, tensor<128x256xf32>) -> tensor<128x256xf32>
// CHECK:           return [[VAR_3_]] : tensor<128x256xf32>
// CHECK:         }
}

// -----

func.func @test_sub_expand_constant_scalar(%arg0: tensor<128x256xf32>) -> (tensor<128x256xf32>) {
  %cst = onnx.Constant dense<1.0> : tensor<f32>
  %0 = "onnx.Sub"(%arg0, %cst) : (tensor<128x256xf32>, tensor<f32>) -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>

// CHECK-LABEL:  func.func @test_sub_expand_constant_scalar
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x256xf32>) -> tensor<128x256xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Shape"([[PARAM_0_]]) {start = 0 : si64} : (tensor<128x256xf32>) -> tensor<2xi64>
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
  %cst = onnx.Constant dense<1.0> : tensor<2xf32>
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

// CONSTPROP-LABEL:  func.func @test_matmul
// CONSTPROP-SAME:   ([[PARAM_0_:%.+]]: tensor<4x12x256x256xf32>, [[PARAM_1_:%.+]]: tensor<4x12x256x64xf32>) -> tensor<4x12x256x64xf32> {
// CONSTPROP-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[-1, 256, 256]> : tensor<3xi64>
// CONSTPROP-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[-1, 256, 64]> : tensor<3xi64>
// CONSTPROP-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[4, 12, 256, 64]> : tensor<4xi64>
// CONSTPROP-NOT: separator of consecutive DAGs
// CONSTPROP-DAG:       [[VAR_3_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<4x12x256x256xf32>, tensor<3xi64>) -> tensor<48x256x256xf32>
// CONSTPROP-DAG:       [[VAR_4_:%.+]] = "onnx.Reshape"([[PARAM_1_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<4x12x256x64xf32>, tensor<3xi64>) -> tensor<48x256x64xf32>
// CONSTPROP:           [[VAR_5_:%.+]] = "onnx.MatMul"([[VAR_3_]], [[VAR_4_]]) : (tensor<48x256x256xf32>, tensor<48x256x64xf32>) -> tensor<48x256x64xf32>
// CONSTPROP:           [[VAR_6_:%.+]] = "onnx.Reshape"([[VAR_5_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<48x256x64xf32>, tensor<4xi64>) -> tensor<4x12x256x64xf32>
// CONSTPROP:           return [[VAR_6_]] : tensor<4x12x256x64xf32>
}

// -----

func.func @test_matmul_broadcast_1(%arg0: tensor<4x12x256x256xf32>, %arg1: tensor<256x64xf32>) -> (tensor<4x12x256x64xf32>) {
    %0= "onnx.MatMul"(%arg0, %arg1) : (tensor<4x12x256x256xf32>, tensor<256x64xf32>) -> tensor<4x12x256x64xf32>
    return %0 : tensor<4x12x256x64xf32>

// CONSTPROP-LABEL:  func.func @test_matmul_broadcast_1
// CONSTPROP-SAME:   ([[PARAM_0_:%.+]]: tensor<4x12x256x256xf32>, [[PARAM_1_:%.+]]: tensor<256x64xf32>) -> tensor<4x12x256x64xf32> {
// CONSTPROP-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[-1, 256, 256]> : tensor<3xi64>
// CONSTPROP-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[4, 12, 256, 64]> : tensor<4xi64>
// CONSTPROP:           [[VAR_2_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<4x12x256x256xf32>, tensor<3xi64>) -> tensor<48x256x256xf32>
// CONSTPROP:           [[VAR_3_:%.+]] = "onnx.MatMul"([[VAR_2_]], [[PARAM_1_]]) : (tensor<48x256x256xf32>, tensor<256x64xf32>) -> tensor<48x256x64xf32>
// CONSTPROP:           [[VAR_4_:%.+]] = "onnx.Reshape"([[VAR_3_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<48x256x64xf32>, tensor<4xi64>) -> tensor<4x12x256x64xf32>
// CONSTPROP:           return [[VAR_4_]] : tensor<4x12x256x64xf32>
}

// -----

func.func @test_matmul_broadcast_2(%arg0: tensor<256x256xf32>, %arg1: tensor<4x12x256x64xf32>) -> (tensor<4x12x256x64xf32>) {
    %0= "onnx.MatMul"(%arg0, %arg1) : (tensor<256x256xf32>, tensor<4x12x256x64xf32>) -> tensor<4x12x256x64xf32>
    return %0 : tensor<4x12x256x64xf32>

// CONSTPROP-LABEL:  func.func @test_matmul_broadcast_2
// CONSTPROP-SAME:   ([[PARAM_0_:%.+]]: tensor<256x256xf32>, [[PARAM_1_:%.+]]: tensor<4x12x256x64xf32>) -> tensor<4x12x256x64xf32> {
// CONSTPROP-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[4, 12, 256, 64]> : tensor<4xi64>
// CONSTPROP:           [[VAR_0_:%.+]] = onnx.Constant dense<[-1, 256, 64]> : tensor<3xi64>
// CONSTPROP:           [[VAR_1_:%.+]] = "onnx.Reshape"([[PARAM_1_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<4x12x256x64xf32>, tensor<3xi64>) -> tensor<48x256x64xf32>
// CONSTPROP-DAG:       [[VAR_2_:%.+]] = "onnx.MatMul"([[PARAM_0_]], [[VAR_1_]]) : (tensor<256x256xf32>, tensor<48x256x64xf32>) -> tensor<48x256x64xf32>
// CONSTPROP:           [[VAR_4_:%.+]] = "onnx.Reshape"([[VAR_2_]], [[VAR_3_]]) {allowzero = 0 : si64} : (tensor<48x256x64xf32>, tensor<4xi64>) -> tensor<4x12x256x64xf32>
// CONSTPROP:           return [[VAR_4_]] : tensor<4x12x256x64xf32>
// CONSTPROP:         }
}

// -----

func.func @test_matmul_broadcast_dyn_dims(%arg0: tensor<256x?xf32>, %arg1: tensor<4x12x?x?xf32>) -> (tensor<4x12x256x?xf32>) {
    %0= "onnx.MatMul"(%arg0, %arg1) : (tensor<256x?xf32>, tensor<4x12x?x?xf32>) -> tensor<4x12x256x?xf32>
    return %0 : tensor<4x12x256x?xf32>
// CONSTPROP-LABEL:  func.func @test_matmul_broadcast_dyn_dims
// CONSTPROP-SAME:   ([[PARAM_0_:%.+]]: tensor<256x?xf32>, [[PARAM_1_:%.+]]: tensor<4x12x?x?xf32>) -> tensor<4x12x256x?xf32> {
// CONSTPROP-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CONSTPROP-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CONSTPROP-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CONSTPROP-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CONSTPROP-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CONSTPROP-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CONSTPROP-DAG:       [[VAR_6_:%.+]] = "onnx.Shape"([[PARAM_1_]]) {start = 0 : si64} : (tensor<4x12x?x?xf32>) -> tensor<4xi64>
// CONSTPROP:           [[VAR_7_:%.+]] = "onnx.Slice"([[VAR_6_]], [[VAR_3_]], [[VAR_2_]], [[VAR_4_]], [[VAR_1_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CONSTPROP:           [[VAR_8_:%.+]] = "onnx.Concat"([[VAR_5_]], [[VAR_7_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<2xi64>) -> tensor<3xi64>
// CONSTPROP:           [[VAR_9_:%.+]] = "onnx.Reshape"([[PARAM_1_]], [[VAR_8_]]) {allowzero = 0 : si64} : (tensor<4x12x?x?xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
// CONSTPROP-DAG:       [[VAR_10_:%.+]] = "onnx.MatMul"([[PARAM_0_]], [[VAR_9_]]) : (tensor<256x?xf32>, tensor<?x?x?xf32>) -> tensor<?x256x?xf32>
// CONSTPROP-DAG:       [[VAR_11_:%.+]] = "onnx.Shape"([[PARAM_0_]]) {start = 0 : si64} : (tensor<256x?xf32>) -> tensor<2xi64>
// CONSTPROP-DAG:       [[VAR_12_:%.+]] = "onnx.Shape"([[PARAM_1_]]) {start = 0 : si64} : (tensor<4x12x?x?xf32>) -> tensor<4xi64>
// CONSTPROP-NOT: separator of consecutive DAGs
// CONSTPROP-DAG:       [[VAR_13_:%.+]] = "onnx.Slice"([[VAR_12_]], [[VAR_4_]], [[VAR_3_]], [[VAR_4_]], [[VAR_1_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CONSTPROP-DAG:       [[VAR_14_:%.+]] = "onnx.Slice"([[VAR_11_]], [[VAR_4_]], [[VAR_1_]], [[VAR_4_]], [[VAR_1_]]) : (tensor<2xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CONSTPROP-DAG:       [[VAR_15_:%.+]] = "onnx.Slice"([[VAR_12_]], [[VAR_0_]], [[VAR_2_]], [[VAR_4_]], [[VAR_1_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CONSTPROP:           [[VAR_16_:%.+]] = "onnx.Concat"([[VAR_13_]], [[VAR_14_]], [[VAR_15_]]) {axis = 0 : si64} : (tensor<2xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
// CONSTPROP:           [[VAR_17_:%.+]] = "onnx.Reshape"([[VAR_10_]], [[VAR_16_]]) {allowzero = 0 : si64} : (tensor<?x256x?xf32>, tensor<4xi64>) -> tensor<4x12x256x?xf32>
// CONSTPROP:           return [[VAR_17_]] : tensor<4x12x256x?xf32>

}

// -----

// Not rewrite because we cannot determine whether there is broadcasting or not.
func.func @test_matmul_unknown_batch_dim_no_rewriting(%arg0: tensor<?x?x256x256xf32>, %arg1: tensor<?x?x256x64xf32>) -> (tensor<?x?x256x64xf32>) {
    %0= "onnx.MatMul"(%arg0, %arg1) : (tensor<?x?x256x256xf32>, tensor<?x?x256x64xf32>) -> tensor<?x?x256x64xf32>
    return %0 : tensor<?x?x256x64xf32>
// CHECK-LABEL: test_matmul_unknown_batch_dim_no_rewriting
// CHECK: %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x?x256x256xf32>, tensor<?x?x256x64xf32>) -> tensor<?x?x256x64xf32>
}

// -----

// Rewrite this matmul because we know that there is no broadcasting.
func.func @test_matmul_unknown_batch_dim(%arg0: tensor<?x?x256x256xf32>) -> (tensor<?x?x256x256xf32>) {
    %0= "onnx.MatMul"(%arg0, %arg0) : (tensor<?x?x256x256xf32>, tensor<?x?x256x256xf32>) -> tensor<?x?x256x256xf32>
    return %0 : tensor<?x?x256x256xf32>

// CHECK-LABEL:  func.func @test_matmul_unknown_batch_dim
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x256x256xf32>) -> tensor<?x?x256x256xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Shape"([[PARAM_0_]]) {start = 0 : si64} : (tensor<?x?x256x256xf32>) -> tensor<4xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Slice"([[VAR_0_]], [[VAR_3_]], [[VAR_4_]], [[VAR_2_]], [[VAR_5_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Concat"([[VAR_1_]], [[VAR_6_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<2xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_7_]]) {allowzero = 0 : si64} : (tensor<?x?x256x256xf32>, tensor<3xi64>) -> tensor<?x256x256xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Shape"([[PARAM_0_]]) {start = 0 : si64} : (tensor<?x?x256x256xf32>) -> tensor<4xi64>
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<-1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_15_:%.+]] = "onnx.Slice"([[VAR_9_]], [[VAR_12_]], [[VAR_13_]], [[VAR_11_]], [[VAR_14_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK:           [[VAR_16_:%.+]] = "onnx.Concat"([[VAR_10_]], [[VAR_15_]]) {axis = 0 : si64} : (tensor<1xi64>, tensor<2xi64>) -> tensor<3xi64>
// CHECK:           [[VAR_17_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_16_]]) {allowzero = 0 : si64} : (tensor<?x?x256x256xf32>, tensor<3xi64>) -> tensor<?x256x256xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.MatMul"([[VAR_8_]], [[VAR_17_]]) : (tensor<?x256x256xf32>, tensor<?x256x256xf32>) -> tensor<?x256x256xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.Shape"([[PARAM_0_]]) {start = 0 : si64} : (tensor<?x?x256x256xf32>) -> tensor<4xi64>
// CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.Shape"([[PARAM_0_]]) {start = 0 : si64} : (tensor<?x?x256x256xf32>) -> tensor<4xi64>
// CHECK-DAG:       [[VAR_21_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_22_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_23_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-DAG:       [[VAR_24_:%.+]] = onnx.Constant dense<4> : tensor<1xi64>
// CHECK-DAG:       [[VAR_25_:%.+]] = onnx.Constant dense<3> : tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_26_:%.+]] = "onnx.Slice"([[VAR_19_]], [[VAR_21_]], [[VAR_23_]], [[VAR_21_]], [[VAR_22_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_27_:%.+]] = "onnx.Slice"([[VAR_20_]], [[VAR_25_]], [[VAR_24_]], [[VAR_21_]], [[VAR_22_]]) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK:           [[VAR_28_:%.+]] = "onnx.Concat"([[VAR_26_]], [[VAR_27_]]) {axis = 0 : si64} : (tensor<3xi64>, tensor<1xi64>) -> tensor<4xi64>
// CHECK:           [[VAR_29_:%.+]] = "onnx.Reshape"([[VAR_18_]], [[VAR_28_]]) {allowzero = 0 : si64} : (tensor<?x256x256xf32>, tensor<4xi64>) -> tensor<?x?x256x256xf32>
// CHECK:           return [[VAR_29_]] : tensor<?x?x256x256xf32>
// CHECK:         }
}

// -----

// Split MatMul because a dimension exceeds NNPAGetMaxForDim = 32768.
func.func @test_matmul_splitting_A(%arg0: tensor<?x50257x768xf32>, %arg1: tensor<768x1024xf32>) -> (tensor<?x50257x1024xf32>) {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x50257x768xf32>, tensor<768x1024xf32>) -> tensor<?x50257x1024xf32>
  return %0 : tensor<?x50257x1024xf32>

// mlir2FileCheck.py -a '["A","B"]'
// CHECK-LABEL:  func.func @test_matmul_splitting_A
// CHECK-SAME:   ([[A_:%.+]]: tensor<?x50257x768xf32>, [[B_:%.+]]: tensor<768x1024xf32>) -> tensor<?x50257x1024xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<[32768, 17489]> : tensor<2xi64>
// CHECK:           [[VAR_1_:%.+]]:2 = "onnx.Split"([[A_]], [[VAR_0_]]) {axis = 1 : si64} : (tensor<?x50257x768xf32>, tensor<2xi64>) -> (tensor<?x32768x768xf32>, tensor<?x17489x768xf32>)
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.MatMul"([[VAR_1_]]#0, [[B_]]) : (tensor<?x32768x768xf32>, tensor<768x1024xf32>) -> tensor<?x32768x1024xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.MatMul"([[VAR_1_]]#1, [[B_]]) : (tensor<?x17489x768xf32>, tensor<768x1024xf32>) -> tensor<?x17489x1024xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Concat"([[VAR_2_]], [[VAR_3_]]) {axis = 1 : si64} : (tensor<?x32768x1024xf32>, tensor<?x17489x1024xf32>) -> tensor<?x50257x1024xf32>
// CHECK:           return [[VAR_4_]] : tensor<?x50257x1024xf32>
// CHECK:         }
}

// Split MatMul because a dimension exceeds NNPAGetMaxForDim = 32768.
func.func @test_matmul_splitting_B(%arg0: tensor<?x?x768xf32>, %arg1: tensor<768x50257xf32>) -> (tensor<?x?x50257xf32>) {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x?x768xf32>, tensor<768x50257xf32>) -> tensor<?x?x50257xf32>
  return %0 : tensor<?x?x50257xf32>

// mlir2FileCheck.py -a '["A","B"]'
// CHECK-LABEL:  func.func @test_matmul_splitting_B
// CHECK-SAME:   ([[A_:%.+]]: tensor<?x?x768xf32>, [[B_:%.+]]: tensor<768x50257xf32>) -> tensor<?x?x50257xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<[32768, 17489]> : tensor<2xi64>
// CHECK:           [[VAR_1_:%.+]]:2 = "onnx.Split"([[B_]], [[VAR_0_]]) {axis = 1 : si64} : (tensor<768x50257xf32>, tensor<2xi64>) -> (tensor<768x32768xf32>, tensor<768x17489xf32>)
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.MatMul"([[A_]], [[VAR_1_]]#0) : (tensor<?x?x768xf32>, tensor<768x32768xf32>) -> tensor<?x?x32768xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.MatMul"([[A_]], [[VAR_1_]]#1) : (tensor<?x?x768xf32>, tensor<768x17489xf32>) -> tensor<?x?x17489xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Concat"([[VAR_2_]], [[VAR_3_]]) {axis = 2 : si64} : (tensor<?x?x32768xf32>, tensor<?x?x17489xf32>) -> tensor<?x?x50257xf32>
// CHECK:           return [[VAR_4_]] : tensor<?x?x50257xf32>
// CHECK:         }
}

// -----

// Split MatMul because a dimension exceeds NNPAGetMaxForDim = 32768.
func.func @test_matmul_splitting_A_B(%arg0: tensor<?x50257x768xf32>, %arg1: tensor<768x50258xf32>) -> (tensor<?x50257x50258xf32>) {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x50257x768xf32>, tensor<768x50258xf32>) -> tensor<?x50257x50258xf32>
  return %0 : tensor<?x50257x50258xf32>

// mlir2FileCheck.py -a '["A","B"]'
// CHECK-LABEL:  func.func @test_matmul_splitting_A_B
// CHECK-SAME:   ([[A_:%.+]]: tensor<?x50257x768xf32>, [[B_:%.+]]: tensor<768x50258xf32>) -> tensor<?x50257x50258xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<[32768, 17489]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_1_:%.+]]:2 = "onnx.Split"([[A_]], [[VAR_0_]]) {axis = 1 : si64} : (tensor<?x50257x768xf32>, tensor<2xi64>) -> (tensor<?x32768x768xf32>, tensor<?x17489x768xf32>)
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[32768, 17490]> : tensor<2xi64>
// CHECK:           [[VAR_3_:%.+]]:2 = "onnx.Split"([[B_]], [[VAR_2_]]) {axis = 1 : si64} : (tensor<768x50258xf32>, tensor<2xi64>) -> (tensor<768x32768xf32>, tensor<768x17490xf32>)
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.MatMul"([[VAR_1_]]#0, [[VAR_3_]]#0) : (tensor<?x32768x768xf32>, tensor<768x32768xf32>) -> tensor<?x32768x32768xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.MatMul"([[VAR_1_]]#0, [[VAR_3_]]#1) : (tensor<?x32768x768xf32>, tensor<768x17490xf32>) -> tensor<?x32768x17490xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Concat"([[VAR_4_]], [[VAR_5_]]) {axis = 2 : si64} : (tensor<?x32768x32768xf32>, tensor<?x32768x17490xf32>) -> tensor<?x32768x50258xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.MatMul"([[VAR_1_]]#1, [[VAR_3_]]#0) : (tensor<?x17489x768xf32>, tensor<768x32768xf32>) -> tensor<?x17489x32768xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.MatMul"([[VAR_1_]]#1, [[VAR_3_]]#1) : (tensor<?x17489x768xf32>, tensor<768x17490xf32>) -> tensor<?x17489x17490xf32>
// CHECK:           [[VAR_9_:%.+]] = "onnx.Concat"([[VAR_7_]], [[VAR_8_]]) {axis = 2 : si64} : (tensor<?x17489x32768xf32>, tensor<?x17489x17490xf32>) -> tensor<?x17489x50258xf32>
// CHECK:           [[VAR_10_:%.+]] = "onnx.Concat"([[VAR_6_]], [[VAR_9_]]) {axis = 1 : si64} : (tensor<?x32768x50258xf32>, tensor<?x17489x50258xf32>) -> tensor<?x50257x50258xf32>
// CHECK:           return [[VAR_10_]] : tensor<?x50257x50258xf32>
// CHECK:         }
}

// -----

// COM: Rewrite N-D Softmax into 2-D softmax when axis is the last dim.
// COM: Disable this rule for now since we see NNAP Softmax produces NaNs for very small values that are out of range of DLFLoat16.

func.func @softmax_nd_to_2d(%arg0: tensor<4x12x256x256xf32>) -> (tensor<4x12x256x256xf32>) {
    %0 = "onnx.Softmax"(%arg0) {axis = 3 : si64} : (tensor<4x12x256x256xf32>) -> tensor<4x12x256x256xf32>
    return %0: tensor<4x12x256x256xf32>

// CONSTPROP-LABEL:  func.func @softmax_nd_to_2d
// CONSTPROP-SAME:   ([[PARAM_0_:%.+]]: tensor<4x12x256x256xf32>) -> tensor<4x12x256x256xf32> {
// CONSTPROP-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[4, 12, 256, 256]> : tensor<4xi64>
// CONSTPROP:           [[VAR_0_:%.+]] = onnx.Constant dense<[-1, 256, 256]> : tensor<3xi64>
// CONSTPROP:           [[VAR_1_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<4x12x256x256xf32>, tensor<3xi64>) -> tensor<48x256x256xf32>
// CONSTPROP-DAG:       [[VAR_2_:%.+]] = "onnx.Softmax"([[VAR_1_]]) {axis = -1 : si64} : (tensor<48x256x256xf32>) -> tensor<48x256x256xf32>
// CONSTPROP:           [[VAR_4_:%.+]] = "onnx.Reshape"([[VAR_2_]], [[VAR_3_]]) {allowzero = 0 : si64} : (tensor<48x256x256xf32>, tensor<4xi64>) -> tensor<4x12x256x256xf32>
// CONSTPROP:           return [[VAR_4_]] : tensor<4x12x256x256xf32>
// CONSTPROP:         }
}

// -----

func.func @test_onnx_conv2d_notset_with_pads(%arg0: tensor<5x3x32x32xf32>, %arg1 : tensor<1024x3x2x2xf32>) -> tensor<5x1024x33x33xf32> {
    %bias = "onnx.NoValue"() {value} : () -> none
    %1 = "onnx.Conv"(%arg0, %arg1, %bias) {auto_pad = "NOTSET", kernel_shape = [2, 2], pads = [0, 0, 2, 2]} : (tensor<5x3x32x32xf32>, tensor<1024x3x2x2xf32>, none) -> tensor<5x1024x33x33xf32>
    return %1 : tensor<5x1024x33x33xf32>
  // CHECK-LABEL: test_onnx_conv2d_notset_with_pads
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x3x32x32xf32>, [[PARAM_1_:%.+]]: tensor<1024x3x2x2xf32>) -> tensor<5x1024x33x33xf32> {
  // CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
  // CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[0, 0, 0, 0, 0, 0, 2, 2]> : tensor<8xi64>
  // CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.NoValue"() {value} : () -> none
  // CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Pad"([[PARAM_0_]], [[VAR_1_]], [[VAR_2_]], [[VAR_3_]]) {mode = "constant"} : (tensor<5x3x32x32xf32>, tensor<8xi64>, tensor<f32>, none) -> tensor<5x3x34x34xf32>
  // CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Conv"([[VAR_4_]], [[PARAM_1_]], [[VAR_0_]]) {auto_pad = "VALID", group = 1 : si64, kernel_shape = [2, 2], pads = [0, 0, 0, 0]} : (tensor<5x3x34x34xf32>, tensor<1024x3x2x2xf32>, none) -> tensor<5x1024x33x33xf32>
  // CHECK:           return [[VAR_5_]] : tensor<5x1024x33x33xf32>
  // CHECK:         }
}

// -----

func.func @test_onnx_conv2d_with_bias_and_different_pads(%arg0: tensor<1x3x224x224xf32>, %arg1 : tensor<64x3x7x7xf32>, %arg2 : tensor<64xf32>) -> tensor<1x64x112x112xf32> {
    %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {kernel_shape = [7, 7], onnx_node_name = "", pads = [3, 3, 3, 3], strides = [2, 2]} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
    return %0 : tensor<1x64x112x112xf32>
  // CHECK-LABEL: test_onnx_conv2d_with_bias_and_different_pads
  // CHECK-SAME:   ([[PARAM_0_]]: tensor<1x3x224x224xf32>, [[PARAM_1_]]: tensor<64x3x7x7xf32>, [[PARAM_2_]]: tensor<64xf32>) -> tensor<1x64x112x112xf32> {
  // CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[0, 0, 3, 3, 0, 0, 3, 3]> : tensor<8xi64>
  // CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.NoValue"() {value} : () -> none
  // CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Pad"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]], [[VAR_2_]]) {mode = "constant"} : (tensor<1x3x224x224xf32>, tensor<8xi64>, tensor<f32>, none) -> tensor<1x3x230x230xf32>
  // CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Conv"([[VAR_3_]], [[PARAM_1_]], [[PARAM_2_]]) {auto_pad = "VALID", group = 1 : si64, kernel_shape = [7, 7], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x3x230x230xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
  // CHECK:           return [[VAR_4_]] : tensor<1x64x112x112xf32>
  // CHECK:         }
}

// -----

func.func @test_onnx_conv2d_not_insert_onnxpad_when_not_necessary(%arg0: tensor<1x3x223x223xf32>, %arg1 : tensor<64x3x7x7xf32>, %arg2 : tensor<64xf32>) -> tensor<1x64x112x112xf32> {
    %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [7, 7], pads = [3, 3, 3, 3], strides = [2, 2]} : (tensor<1x3x223x223xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
    return %0 : tensor<1x64x112x112xf32>
  // CHECK-LABEL: test_onnx_conv2d_not_insert_onnxpad_when_not_necessary
  // CHECK-NOT: "onnx.Pad"
}

// -----

func.func @test_onnx_conv2d_not_insert_onnxpad_if_cannot_get_pads(%arg0: tensor<1x3x?x?xf32>, %arg1 : tensor<64x3x7x7xf32>, %arg2 : tensor<64xf32>) -> tensor<1x64x?x?xf32> {
    %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [7, 7], pads = [3, 3, 3, 3], strides = [2, 2]} : (tensor<1x3x?x?xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>) -> tensor<1x64x?x?xf32>
    return %0 : tensor<1x64x?x?xf32>
  // CHECK-LABEL: test_onnx_conv2d_not_insert_onnxpad_if_cannot_get_pads
  // CHECK-NOT: "onnx.Pad"
}

// -----

func.func @test_onnx_conv2d_not_insert_onnxpad_if_auto_pad_is_valid(%arg0: tensor<1x3x224x224xf32>, %arg1 : tensor<64x3x7x7xf32>, %arg2 : tensor<64xf32>) -> tensor<1x64x112x112xf32> {
    %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {auto_pad = "VALID", kernel_shape = [7, 7], onnx_node_name = "", strides = [2, 2]} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
    return %0 : tensor<1x64x112x112xf32>
  // CHECK-LABEL: test_onnx_conv2d_not_insert_onnxpad_if_auto_pad_is_valid
  // CHECK-NOT: "onnx.Pad"
}

// -----

func.func @test_replace_add_zero_expand(%arg0: tensor<2x4xf32>, %arg1: tensor<?xi64>) -> tensor<2x4xf32> {
  %0 = onnx.Constant dense<1> : tensor<1xi64>
  %1 = "onnx.Dim"(%arg1) {axis = 0 : si64} : (tensor<?xi64>) -> tensor<1xi64>
  %2 = "onnx.Concat"(%0, %1) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  %3 = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %4 = "onnx.Expand"(%3, %2) : (tensor<f32>, tensor<2xi64>) -> tensor<1x?xf32>
  %5 = "onnx.Add"(%arg0, %4) : (tensor<2x4xf32>, tensor<1x?xf32>) -> tensor<2x4xf32>
  return %5 : tensor<2x4xf32>

// CHECK-LABEL:  func.func @test_replace_add_zero_expand
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4xf32>, [[PARAM_1_:%.+]]: tensor<?xi64>) -> tensor<2x4xf32> {
// CHECK-NOT:       onnx.Add
// CHECK:           return [[PARAM_0_]] : tensor<2x4xf32>
// CHECK:         }
}

// -----

func.func @test_replace_sub_zero_expand(%arg0: tensor<2x4xf32>, %arg1: tensor<?xi64>) -> tensor<2x4xf32> {
  %0 = onnx.Constant dense<1> : tensor<1xi64>
  %1 = "onnx.Dim"(%arg1) {axis = 0 : si64} : (tensor<?xi64>) -> tensor<1xi64>
  %2 = "onnx.Concat"(%0, %1) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  %3 = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %4 = "onnx.Expand"(%3, %2) : (tensor<f32>, tensor<2xi64>) -> tensor<1x?xf32>
  %5 = "onnx.Sub"(%arg0, %4) : (tensor<2x4xf32>, tensor<1x?xf32>) -> tensor<2x4xf32>
  return %5 : tensor<2x4xf32>

// CHECK-LABEL:  func.func @test_replace_sub_zero_expand
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4xf32>, [[PARAM_1_:%.+]]: tensor<?xi64>) -> tensor<2x4xf32> {
// CHECK-NOT:       onnx.Sub
// CHECK:           return [[PARAM_0_]] : tensor<2x4xf32>
// CHECK:         }
}
