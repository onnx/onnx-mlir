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