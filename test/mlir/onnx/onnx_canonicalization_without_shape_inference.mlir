// RUN: onnx-mlir-opt --canonicalize="test-convergence=true" %s -split-input-file | FileCheck %s

// FIXME: This tests have issues when running shape-inference previous to canonicalize.

// CHECK-LABEL: @test_gemm_add_fusion_rank3(%{{.*}}: tensor<128x128x256xf32>, %{{.*}}: tensor<128x128x256xf32>, %{{.*}}: tensor<256xf32>) -> tensor<*xf32> {
func.func @test_gemm_add_fusion_rank3(%arg0: tensor<128x128x256xf32>, %arg1: tensor<128x128x256xf32>, %arg2: tensor<256xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Gemm"(%arg0, %arg1, %cst) : (tensor<128x128x256xf32>, tensor<128x128x256xf32>, none) -> tensor<*xf32>
  %1 = "onnx.Add"(%0, %arg2) : (tensor<*xf32>, tensor<256xf32>) -> tensor<*xf32>
  onnx.Return %1 : tensor<*xf32>

  // CHECK-NEXT: [[GEMM:%.+]] = "onnx.Gemm"(%{{.*}}, %{{.*}}, %{{.*}}) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<128x128x256xf32>, tensor<128x128x256xf32>, tensor<256xf32>) -> tensor<*xf32>
  // onnx.Return [[GEMM]] : tensor<*xf32>
}

// -----

//CHECK-LABEL: @test_gemm_add_fusion(%{{.*}}: tensor<128x128xf32>, %{{.*}}: tensor<128x128xf32>, %{{.*}}: tensor<128xf32>) -> tensor<*xf32> {
func.func @test_gemm_add_fusion(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Gemm"(%arg0, %arg1, %cst) : (tensor<128x128xf32>, tensor<128x128xf32>, none) -> tensor<*xf32>
  %1 = "onnx.Add"(%0, %arg2) : (tensor<*xf32>, tensor<128xf32>) -> tensor<*xf32>
  onnx.Return %1 : tensor<*xf32>

  // CHECK-NEXT: [[GEMM:%.+]] = "onnx.Gemm"(%{{.*}}, %{{.*}}, %{{.*}}) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<*xf32>
  // onnx.Return [[GEMM]] : tensor<*xf32>
}

// -----

//CHECK-LABEL: @test_gemm_add_fusion_beta_zero(%{{.*}}: tensor<128x128xf32>, %{{.*}}: tensor<128x128xf32>, %{{.*}}: tensor<128xf32>) -> tensor<*xf32> {
func.func @test_gemm_add_fusion_beta_zero(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Gemm"(%arg0, %arg1, %cst) {beta = 0.0 : f32}: (tensor<128x128xf32>, tensor<128x128xf32>, none) -> tensor<*xf32>
  %1 = "onnx.Add"(%0, %arg2) : (tensor<*xf32>, tensor<128xf32>) -> tensor<*xf32>
  onnx.Return %1 : tensor<*xf32>

  // CHECK-NEXT: [[GEMM:%.+]] = "onnx.Gemm"(%{{.*}}, %{{.*}}, %{{.*}}) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128xf32>) -> tensor<*xf32>
  // onnx.Return [[GEMM]] : tensor<*xf32>
}

// -----

// Check deriving a new maximum trip count from the break condition of the loop.
// In this test, the new maximum trip count is a constant.
func.func @test_loop_derive_max_trip_count(%arg0: tensor<?x30xf32>) -> tensor<?x?x30xf32> {
  %0 = onnx.Constant dense<9223372036854775807> : tensor<i64>
  %1 = onnx.Constant dense<true> : tensor<i1>
  %2 = onnx.Constant dense<0> : tensor<i32>
  %3 = onnx.Constant dense<30> : tensor<i32>
  %4:4 = "onnx.Loop"(%0, %1, %2, %3, %arg0) ({
  ^bb0(%arg1: tensor<i64>, %arg2: tensor<i1>, %arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<?x30xf32>):
    %5 = onnx.Constant dense<4> : tensor<i32>
    %6 = "onnx.Add"(%arg3, %5) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %7 = "onnx.Relu"(%arg5) : (tensor<?x30xf32>) -> tensor<?x30xf32>
    %8 = "onnx.Less"(%6, %arg4) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    onnx.Yield %8, %6, %arg4, %7 : tensor<i1>, tensor<i32>, tensor<i32>, tensor<?x30xf32>
  }) : (tensor<i64>, tensor<i1>, tensor<i32>, tensor<i32>, tensor<?x30xf32>) -> (tensor<i32>, tensor<i32>, tensor<?x30xf32>, tensor<?x?x30xf32>)
  onnx.Return %4#3 : tensor<?x?x30xf32>
// CHECK-LABEL:  func.func @test_loop_derive_max_trip_count
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x30xf32>) -> tensor<?x?x30xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<8> : tensor<i64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<4> : tensor<i32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<true> : tensor<i1>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<0> : tensor<i32>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<30> : tensor<i32>
// CHECK:           [[VAR_5_:%.+]]:4 = "onnx.Loop"([[VAR_0_]], [[VAR_2_]], [[VAR_3_]], [[VAR_4_]], [[PARAM_0_]]) ({
// CHECK:           ^bb0([[arg1_:%.+]]: tensor<i64>, [[arg2_:%.+]]: tensor<i1>, [[arg3_:%.+]]: tensor<i32>, [[arg4_:%.+]]: tensor<i32>, [[arg5_:%.+]]: tensor<?x30xf32>):
// CHECK-DAG:         [[VAR_6_:%.+]] = "onnx.Add"([[arg3_]], [[VAR_1_]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK-DAG:         [[VAR_7_:%.+]] = "onnx.Relu"([[arg5_]]) : (tensor<?x30xf32>) -> tensor<?x30xf32>
// CHECK:             onnx.Yield [[arg2_]], [[VAR_6_]], [[arg4_]], [[VAR_7_]] : tensor<i1>, tensor<i32>, tensor<i32>, tensor<?x30xf32>
// CHECK:           }) : (tensor<i64>, tensor<i1>, tensor<i32>, tensor<i32>, tensor<?x30xf32>) -> (tensor<i32>, tensor<i32>, tensor<?x30xf32>, tensor<?x?x30xf32>)
// CHECK:           onnx.Return [[VAR_5_]]#3 : tensor<?x?x30xf32>

}

// -----

// Check deriving a new maximum trip count from the break condition of the loop.
// In this test, the new maximum trip count is not a constant.
func.func @test_loop_derive_max_trip_count_non_constant_ub(%arg0: tensor<?x30xf32>, %arg1: tensor<i32>) -> tensor<?x?x30xf32> {
  %0 = onnx.Constant dense<9223372036854775807> : tensor<i64>
  %1 = onnx.Constant dense<true> : tensor<i1>
  %2 = onnx.Constant dense<0> : tensor<i32>
  %3:4 = "onnx.Loop"(%0, %1, %2, %arg1, %arg0) ({
  ^bb0(%arg2: tensor<i64>, %arg3: tensor<i1>, %arg4: tensor<i32>, %arg5: tensor<i32>, %arg6: tensor<?x30xf32>):
    %4 = onnx.Constant dense<1> : tensor<i32>
    %5 = "onnx.Add"(%arg4, %4) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %6 = "onnx.Relu"(%arg6) : (tensor<?x30xf32>) -> tensor<?x30xf32>
    %7 = "onnx.Less"(%5, %arg5) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    onnx.Yield %7, %5, %arg5, %6 : tensor<i1>, tensor<i32>, tensor<i32>, tensor<?x30xf32>
  }) : (tensor<i64>, tensor<i1>, tensor<i32>, tensor<i32>, tensor<?x30xf32>) -> (tensor<i32>, tensor<i32>, tensor<?x30xf32>, tensor<?x?x30xf32>)
  onnx.Return %3#3 : tensor<?x?x30xf32>
// CHECK-LABEL:  func @test_loop_derive_max_trip_count_non_constant_ub
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x30xf32>, [[PARAM_1_:%.+]]: tensor<i32>) -> tensor<?x?x30xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<i32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<9223372036854775807> : tensor<i64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<true> : tensor<i1>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<0> : tensor<i32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Cast"([[PARAM_1_]]) {saturate = 1 : si64, to = i64} : (tensor<i32>) -> tensor<i64>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Cast"([[VAR_3_]]) {saturate = 1 : si64, to = i64} : (tensor<i32>) -> tensor<i64>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Sub"([[VAR_4_]], [[VAR_5_]]) : (tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Cast"([[VAR_6_]]) {saturate = 1 : si64, to = f32} : (tensor<i64>) -> tensor<f32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Cast"([[VAR_0_]]) {saturate = 1 : si64, to = f32} : (tensor<i32>) -> tensor<f32>
// CHECK:           [[VAR_9_:%.+]] = "onnx.Div"([[VAR_7_]], [[VAR_8_]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:           [[VAR_10_:%.+]] = "onnx.Ceil"([[VAR_9_]]) : (tensor<f32>) -> tensor<f32>
// CHECK:           [[VAR_11_:%.+]] = "onnx.Cast"([[VAR_10_]]) {saturate = 1 : si64, to = i64} : (tensor<f32>) -> tensor<i64>
// CHECK:           [[VAR_12_:%.+]] = "onnx.Min"([[VAR_1_]], [[VAR_1_]]1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK:           [[VAR_13_:%.+]]:4 = "onnx.Loop"([[VAR_12_]], [[VAR_2_]], [[VAR_3_]], [[PARAM_1_]], [[PARAM_0_]]) ({
// CHECK:           ^bb0([[arg2_:%.+]]: tensor<i64>, [[arg3_:%.+]]: tensor<i1>, [[arg4_:%.+]]: tensor<i32>, [[arg5_:%.+]]: tensor<i32>, [[arg6_:%.+]]: tensor<?x30xf32>):
// CHECK-DAG:         [[VAR_14_:%.+]] = "onnx.Add"([[arg4_]], [[VAR_0_]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK-DAG:         [[VAR_15_:%.+]] = "onnx.Relu"([[arg6_]]) : (tensor<?x30xf32>) -> tensor<?x30xf32>
// CHECK:             onnx.Yield [[arg3_]], [[VAR_14_]], [[arg5_]], [[VAR_15_]] : tensor<i1>, tensor<i32>, tensor<i32>, tensor<?x30xf32>
// CHECK:           }) : (tensor<i64>, tensor<i1>, tensor<i32>, tensor<i32>, tensor<?x30xf32>) -> (tensor<i32>, tensor<i32>, tensor<?x30xf32>, tensor<?x?x30xf32>)
// CHECK:           onnx.Return [[VAR_13_]]#3 : tensor<?x?x30xf32>

}

// -----

func.func @test_instancenorm_dynamic_1(%arg0: tensor<*xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> tensor<*xf32> {
  %0 = "onnx.InstanceNormalization"(%arg0, %arg1, %arg2) {epsilon = 0.00999999977 : f32} : (tensor<*xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
// CHECK-LABEL:  func.func @test_instancenorm_dynamic_1
// CHECK:           onnx.InstanceNormalization
}

// -----

func.func @test_instancenorm_dynamic_2(%arg0: tensor<2x3x4x5x6xf32>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "onnx.InstanceNormalization"(%arg0, %arg1, %arg2) {epsilon = 0.00999999977 : f32} : (tensor<2x3x4x5x6xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
// CHECK-LABEL:  func.func @test_instancenorm_dynamic_2
// CHECK:           onnx.InstanceNormalization
}
// -----

func.func @test_instancenorm_dynamic_3(%arg0: tensor<2x3x4x5x6xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?x?x?x?x?xf32> {
  %0 = "onnx.InstanceNormalization"(%arg0, %arg1, %arg2) {epsilon = 0.00999999977 : f32} : (tensor<2x3x4x5x6xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<?x?x?x?x?xf32>
  onnx.Return %0 : tensor<?x?x?x?x?xf32>
// CHECK-LABEL:  func.func @test_instancenorm_dynamic_3
// CHECK:           onnx.InstanceNormalization
}

// -----

func.func @test_instancenorm_dynamic_4(%arg0: tensor<?x?x?x?x?xf32>, %arg1: tensor<3xf32>, %arg2: tensor<3xf32>) -> tensor<2x3x4x5x6xf32> {
  %0 = "onnx.InstanceNormalization"(%arg0, %arg1, %arg2) {epsilon = 0.00999999977 : f32} : (tensor<?x?x?x?x?xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<2x3x4x5x6xf32>
  onnx.Return %0 : tensor<2x3x4x5x6xf32>
// CHECK-LABEL:  func.func @test_instancenorm_dynamic_4
// CHECK:           onnx.InstanceNormalization
}

// -----

func.func @test_batchnorm_f16_dynamic(%arg0: tensor<100x3x?x?xf16>) -> tensor<*xf16> {
    %0 = "onnx.Constant"() {value = dense<[1.0, 2.0, 3.0]> : tensor<3xf16>} : () -> tensor<3xf16>
    %1 = "onnx.Constant"() {value = dense<[2.0, 3.0, 4.0]> : tensor<3xf16>} : () -> tensor<3xf16>
    %2 = "onnx.Constant"() {value = dense<[3.0, 4.0, 5.0]> : tensor<3xf16>} : () -> tensor<3xf16>
    %3 = "onnx.Constant"() {value = dense<[4.0, 5.0, 6.0]> : tensor<3xf16>} : () -> tensor<3xf16>
    %4, %5, %6 = "onnx.BatchNormalization"(%arg0, %0, %1, %2, %3) {epsilon = 1.00000007E-5 : f32, momentum = 1.00000007E-3 : f32} : (tensor<100x3x?x?xf16>, tensor<3xf16>, tensor<3xf16>, tensor<3xf16>, tensor<3xf16>) -> (tensor<*xf16>, tensor<*xf16>, tensor<*xf16>)
    return %4 : tensor<*xf16>
// CHECK-LABEL:  func.func @test_batchnorm_f16_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<100x3x?x?xf16>) -> tensor<*xf16> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 2]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<3xf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[3.000000e+00, 4.000000e+00, 5.000000e+00]> : tensor<3xf16>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<[4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<3xf16>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<1.001360e-05> : tensor<1xf16>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Add"([[VAR_4_]], [[VAR_5_]]) : (tensor<3xf16>, tensor<1xf16>) -> tensor<3xf16>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Sqrt"([[VAR_6_]]) : (tensor<3xf16>) -> tensor<*xf16>
// CHECK:           [[VAR_8_:%.+]] = "onnx.Div"([[VAR_1_]], [[VAR_7_]]) : (tensor<3xf16>, tensor<*xf16>) -> tensor<*xf16>
// CHECK:           [[VAR_9_:%.+]] = "onnx.Unsqueeze"([[VAR_8_]], [[VAR_0_]]) : (tensor<*xf16>, tensor<2xi64>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[VAR_9_]]) : (tensor<100x3x?x?xf16>, tensor<*xf16>) -> tensor<*xf16>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Mul"([[VAR_8_]], [[VAR_3_]]) : (tensor<*xf16>, tensor<3xf16>) -> tensor<*xf16>
// CHECK:           [[VAR_12_:%.+]] = "onnx.Sub"([[VAR_2_]], [[VAR_11_]]) : (tensor<3xf16>, tensor<*xf16>) -> tensor<*xf16>
// CHECK:           [[VAR_13_:%.+]] = "onnx.Unsqueeze"([[VAR_12_]], [[VAR_0_]]) : (tensor<*xf16>, tensor<2xi64>) -> tensor<*xf16>
// CHECK:           [[VAR_14_:%.+]] = "onnx.Add"([[VAR_10_]], [[VAR_13_]]) : (tensor<*xf16>, tensor<*xf16>) -> tensor<*xf16>
// CHECK:           return [[VAR_14_]] : tensor<*xf16>
}

// -----

func.func @test_batchnorm_bf16_dynamic(%arg0: tensor<100x3x?x?xbf16>) -> tensor<*xbf16> {
    %0 = "onnx.Constant"() {value = dense<[1.0, 2.0, 3.0]> : tensor<3xbf16>} : () -> tensor<3xbf16>
    %1 = "onnx.Constant"() {value = dense<[2.0, 3.0, 4.0]> : tensor<3xbf16>} : () -> tensor<3xbf16>
    %2 = "onnx.Constant"() {value = dense<[3.0, 4.0, 5.0]> : tensor<3xbf16>} : () -> tensor<3xbf16>
    %3 = "onnx.Constant"() {value = dense<[4.0, 5.0, 6.0]> : tensor<3xbf16>} : () -> tensor<3xbf16>
    %4, %5, %6 = "onnx.BatchNormalization"(%arg0, %0, %1, %2, %3) {epsilon = 1.00000007E-5 : f32, momentum = 1.00000007E-3 : f32} : (tensor<100x3x?x?xbf16>, tensor<3xbf16>, tensor<3xbf16>, tensor<3xbf16>, tensor<3xbf16>) -> (tensor<*xbf16>, tensor<*xbf16>, tensor<*xbf16>)
    return %4 : tensor<*xbf16>
// CHECK-LABEL:  func.func @test_batchnorm_bf16_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<100x3x?x?xbf16>) -> tensor<*xbf16> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 2]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xbf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<3xbf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[3.000000e+00, 4.000000e+00, 5.000000e+00]> : tensor<3xbf16>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<[4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<3xbf16>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<1.001360e-05> : tensor<1xbf16>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Add"([[VAR_4_]], [[VAR_5_]]) : (tensor<3xbf16>, tensor<1xbf16>) -> tensor<3xbf16>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Sqrt"([[VAR_6_]]) : (tensor<3xbf16>) -> tensor<*xbf16>
// CHECK:           [[VAR_8_:%.+]] = "onnx.Div"([[VAR_1_]], [[VAR_7_]]) : (tensor<3xbf16>, tensor<*xbf16>) -> tensor<*xbf16>
// CHECK:           [[VAR_9_:%.+]] = "onnx.Unsqueeze"([[VAR_8_]], [[VAR_0_]]) : (tensor<*xbf16>, tensor<2xi64>) -> tensor<*xbf16>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[VAR_9_]]) : (tensor<100x3x?x?xbf16>, tensor<*xbf16>) -> tensor<*xbf16>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Mul"([[VAR_8_]], [[VAR_3_]]) : (tensor<*xbf16>, tensor<3xbf16>) -> tensor<*xbf16>
// CHECK:           [[VAR_12_:%.+]] = "onnx.Sub"([[VAR_2_]], [[VAR_11_]]) : (tensor<3xbf16>, tensor<*xbf16>) -> tensor<*xbf16>
// CHECK:           [[VAR_13_:%.+]] = "onnx.Unsqueeze"([[VAR_12_]], [[VAR_0_]]) : (tensor<*xbf16>, tensor<2xi64>) -> tensor<*xbf16>
// CHECK:           [[VAR_14_:%.+]] = "onnx.Add"([[VAR_10_]], [[VAR_13_]]) : (tensor<*xbf16>, tensor<*xbf16>) -> tensor<*xbf16>
// CHECK:           return [[VAR_14_]] : tensor<*xbf16>
}

// -----

func.func @test_batchnormv9_f16_dynamic(%arg0: tensor<100x3x?x?xf16>) -> (tensor<*xf16>, tensor<*xf16>, tensor<*xf16>) {
    %0 = "onnx.Constant"() {value = dense<[1.0, 2.0, 3.0]> : tensor<3xf16>} : () -> tensor<3xf16>
    %1 = "onnx.Constant"() {value = dense<[2.0, 3.0, 4.0]> : tensor<3xf16>} : () -> tensor<3xf16>
    %2 = "onnx.Constant"() {value = dense<[3.0, 4.0, 5.0]> : tensor<3xf16>} : () -> tensor<3xf16>
    %3 = "onnx.Constant"() {value = dense<[4.0, 5.0, 6.0]> : tensor<3xf16>} : () -> tensor<3xf16>
    %4, %5, %6, %7, %8 = "onnx.BatchNormalizationV9"(%arg0, %0, %1, %2, %3) {epsilon = 1.00000007E-5 : f32, momentum = 1.00000007E-3 : f32} : (tensor<100x3x?x?xf16>, tensor<3xf16>, tensor<3xf16>, tensor<3xf16>, tensor<3xf16>) -> (tensor<*xf16>, tensor<*xf16>, tensor<*xf16>,tensor<*xf16>, tensor<*xf16>)
    return %4, %5, %6 : tensor<*xf16>, tensor<*xf16>, tensor<*xf16>
}

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_batchnormv9_f16_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<100x3x?x?xf16>) -> (tensor<*xf16>, tensor<*xf16>, tensor<*xf16>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf16>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<3xf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[3.000000e+00, 4.000000e+00, 5.000000e+00]> : tensor<3xf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<3xf16>
// CHECK:           [[Y_:%.+]], [[VAR_running_mean_:%.+]], [[VAR_running_var_:%.+]] = "onnx.BatchNormalization"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]], [[VAR_2_]], [[VAR_3_]]) {epsilon = 1.00000007E-5 : f32, momentum = 1.000000e-03 : f32, training_mode = 0 : si64} : (tensor<100x3x?x?xf16>, tensor<3xf16>, tensor<3xf16>, tensor<3xf16>, tensor<3xf16>) -> (tensor<*xf16>, tensor<*xf16>, tensor<*xf16>)
// CHECK:           return [[Y_]], [[VAR_running_mean_]], [[VAR_running_var_]] : tensor<*xf16>, tensor<*xf16>, tensor<*xf16>
// CHECK:         }

// -----
func.func @test_split_relu_movement_missing_shape(%arg0: tensor<1x8x2xf32>) -> (tensor<1x2x2xf32>, tensor<*xf32>, tensor<1x3x2xf32>) {
  %cst = onnx.Constant dense<[2, 3, 3]> : tensor<3xi64>
  %0:3 = "onnx.Split"(%arg0, %cst) {axis = 1 : si64} : (tensor<1x8x2xf32>, tensor<3xi64>) -> (tensor<1x2x2xf32>, tensor<1x3x2xf32>, tensor<1x3x2xf32>)
  %1 = "onnx.Relu"(%0#0) {onnx_node_name = "onnx.Relu_1"} : (tensor<1x2x2xf32>) -> tensor<1x2x2xf32>
  %2 = "onnx.Relu"(%0#1) {onnx_node_name = "onnx.Relu_2"} : (tensor<1x3x2xf32>) -> tensor<*xf32>
  %3 = "onnx.Relu"(%0#2) {onnx_node_name = "onnx.Relu_3"} : (tensor<1x3x2xf32>) -> tensor<1x3x2xf32>
  onnx.Return %1, %2, %3 : tensor<1x2x2xf32>, tensor<*xf32>, tensor<1x3x2xf32>
}

// CHECK-LABEL:  func.func @test_split_relu_movement_missing_shape
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x8x2xf32>) -> (tensor<1x2x2xf32>, tensor<*xf32>, tensor<1x3x2xf32>) {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<[2, 3, 3]> : tensor<3xi64>
// CHECK:           [[VAR_1_:%.+]]:3 = "onnx.Split"([[PARAM_0_]], [[VAR_0_]]) {axis = 1 : si64} : (tensor<1x8x2xf32>, tensor<3xi64>) -> (tensor<1x2x2xf32>, tensor<1x3x2xf32>, tensor<1x3x2xf32>)
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Relu"([[VAR_1_]]#0) {onnx_node_name = "onnx.Relu_1"} : (tensor<1x2x2xf32>) -> tensor<1x2x2xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Relu"([[VAR_1_]]#1) {onnx_node_name = "onnx.Relu_2"} : (tensor<1x3x2xf32>) -> tensor<*xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Relu"([[VAR_1_]]#2) {onnx_node_name = "onnx.Relu_3"} : (tensor<1x3x2xf32>) -> tensor<1x3x2xf32>
// CHECK:           onnx.Return [[VAR_2_]], [[VAR_3_]], [[VAR_4_]] : tensor<1x2x2xf32>, tensor<*xf32>, tensor<1x3x2xf32>
// CHECK:         }