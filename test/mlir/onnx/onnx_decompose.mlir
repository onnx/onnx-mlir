// RUN: onnx-mlir-opt --decompose-onnx %s -split-input-file | FileCheck %s

// -----

func.func @test_grid_sample_v16_bicubic(%arg0: tensor<2x1x4x4xf32>, %arg1: tensor<2x6x6x2xf32>) -> tensor<*xf32> {
  %0 = "onnx.GridSampleV16"(%arg0, %arg1) {align_corners = 1 : si64, mode = "bicubic", padding_mode = "zeros"} : (tensor<2x1x4x4xf32>, tensor<2x6x6x2xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
// CHECK-LABEL:  func.func @test_grid_sample_v16_bicubic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x1x4x4xf32>, [[PARAM_1_:%.+]]: tensor<2x6x6x2xf32>) -> tensor<*xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.GridSample"([[PARAM_0_]], [[PARAM_1_]]) {align_corners = 1 : si64, mode = "cubic", padding_mode = "zeros"} : (tensor<2x1x4x4xf32>, tensor<2x6x6x2xf32>) -> tensor<*xf32>
// CHECK:           return [[VAR_0_]] : tensor<*xf32>
// CHECK:         }
}

// -----

func.func @test_grid_sample_v16_bilinear(%arg0: tensor<2x1x4x4xf32>, %arg1: tensor<2x6x6x2xf32>) -> tensor<*xf32> {
  %0 = "onnx.GridSampleV16"(%arg0, %arg1) {align_corners = 1 : si64, mode = "bilinear", padding_mode = "zeros"} : (tensor<2x1x4x4xf32>, tensor<2x6x6x2xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
// CHECK-LABEL:  func.func @test_grid_sample_v16_bilinear
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x1x4x4xf32>, [[PARAM_1_:%.+]]: tensor<2x6x6x2xf32>) -> tensor<*xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.GridSample"([[PARAM_0_]], [[PARAM_1_]]) {align_corners = 1 : si64, mode = "linear", padding_mode = "zeros"} : (tensor<2x1x4x4xf32>, tensor<2x6x6x2xf32>) -> tensor<*xf32>
// CHECK:           return [[VAR_0_]] : tensor<*xf32>
// CHECK:         }
}

// -----

func.func @test_grid_sample_v16_nearest(%arg0: tensor<2x1x4x4xf32>, %arg1: tensor<2x6x6x2xf32>) -> tensor<*xf32> {
  %0 = "onnx.GridSampleV16"(%arg0, %arg1) {align_corners = 1 : si64, mode = "nearest", padding_mode = "zeros"} : (tensor<2x1x4x4xf32>, tensor<2x6x6x2xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
// CHECK-LABEL:  func.func @test_grid_sample_v16_nearest
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x1x4x4xf32>, [[PARAM_1_:%.+]]: tensor<2x6x6x2xf32>) -> tensor<*xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.GridSample"([[PARAM_0_]], [[PARAM_1_]]) {align_corners = 1 : si64, mode = "nearest", padding_mode = "zeros"} : (tensor<2x1x4x4xf32>, tensor<2x6x6x2xf32>) -> tensor<*xf32>
// CHECK:           return [[VAR_0_]] : tensor<*xf32>
// CHECK:         }
}

// -----

func.func @test_dft(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?xi64>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 ="onnx.DFTV17"(%arg0, %arg1) : (tensor<?x?x?xf32>, tensor<?xi64>)-> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_dft
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?xf32>, [[PARAM_1_:%.+]]: tensor<?xi64>) -> tensor<*xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<i64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.DFT"([[PARAM_0_]], [[PARAM_1_]], [[VAR_0_]]) {inverse = 0 : si64, onesided = 0 : si64} : (tensor<?x?x?xf32>, tensor<?xi64>, tensor<i64>) -> tensor<*xf32>
// CHECK:           onnx.Return [[VAR_1_]] : tensor<*xf32>
// CHECK:         }
}

// -----

func.func @test_dft_one_sided(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?xi64>) -> tensor<*xf32> {
  %0 ="onnx.DFTV17"(%arg0, %arg1) {onesided = 1 : si64}  : (tensor<?x?x?xf32>, tensor<?xi64>)-> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_dft_one_sided
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?xf32>, [[PARAM_1_:%.+]]: tensor<?xi64>) -> tensor<*xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<i64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.DFT"([[PARAM_0_]], [[PARAM_1_]], [[VAR_0_]]) {inverse = 0 : si64, onesided = 1 : si64} : (tensor<?x?x?xf32>, tensor<?xi64>, tensor<i64>) -> tensor<*xf32>
// CHECK:           onnx.Return [[VAR_1_]] : tensor<*xf32>
// CHECK:         }
}

// -----

func.func @test_dft_inverse(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?xi64>) -> tensor<*xf32> {
  %0 ="onnx.DFTV17"(%arg0, %arg1) {inverse = 1 : si64}  : (tensor<?x?x?xf32>, tensor<?xi64>)-> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_dft_inverse
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?xf32>, [[PARAM_1_:%.+]]: tensor<?xi64>) -> tensor<*xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<i64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.DFT"([[PARAM_0_]], [[PARAM_1_]], [[VAR_0_]]) {inverse = 1 : si64, onesided = 0 : si64} : (tensor<?x?x?xf32>, tensor<?xi64>, tensor<i64>) -> tensor<*xf32>
// CHECK:           onnx.Return [[VAR_1_]] : tensor<*xf32>
// CHECK:         }
}

// -----

// CHECK-LABEL: @test_reducel1(%{{.*}}: tensor<?x?x?xf32>, %{{.*}}: tensor<?xi64>) -> tensor<*xf32>
func.func @test_reducel1(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?xi64>) -> tensor<*xf32> {
  %0 ="onnx.ReduceL1"(%arg0, %arg1) {keepdims = 0 : si64} : (tensor<?x?x?xf32>, tensor<?xi64>)-> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-NEXT: [[ABS:%.+]] =  "onnx.Abs"(%arg0) : (tensor<?x?x?xf32>) -> tensor<*xf32>
  // CHECK-NEXT: %{{[0-9]+}} = "onnx.ReduceSum"([[ABS]], %arg1) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<*xf32>, tensor<?xi64>) -> tensor<*xf32>
}

// -----

// CHECK-LABEL: @test_reducel2(%{{.*}}: tensor<?x?x?xf32>, %{{.*}}: tensor<?xi64>) -> tensor<*xf32>
func.func @test_reducel2(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?xi64>) -> tensor<*xf32> {
  %0 ="onnx.ReduceL2"(%arg0, %arg1) {keepdims = 0 : si64} : (tensor<?x?x?xf32>, tensor<?xi64>)-> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-NEXT: [[MUL:%.+]] =  "onnx.Mul"(%arg0, %arg0) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<*xf32>
  // CHECK-NEXT: [[REDUCE_SUM:%.+]] = "onnx.ReduceSum"([[MUL]], %arg1) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<*xf32>, tensor<?xi64>) -> tensor<*xf32>
  // CHECK-NEXT: [[SQRT:%.+]] =  "onnx.Sqrt"([[REDUCE_SUM]]) : (tensor<*xf32>) -> tensor<*xf32>
}

// -----

// CHECK-LABEL: @test_reducelogsum(%{{.*}}: tensor<?x?x?xf32>, %{{.*}}: tensor<?xi64>) -> tensor<*xf32>
func.func @test_reducelogsum(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?xi64>) -> tensor<*xf32> {
  %0 ="onnx.ReduceLogSum"(%arg0, %arg1) {keepdims = 0 : si64} : (tensor<?x?x?xf32>, tensor<?xi64>)-> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()
  // CHECK-NEXT: [[REDUCE_SUM:%.+]] = "onnx.ReduceSum"(%arg0, %arg1) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<?x?x?xf32>, tensor<?xi64>) -> tensor<*xf32>
  // CHECK-NEXT: [[LOG:%.+]] =  "onnx.Log"([[REDUCE_SUM]]) : (tensor<*xf32>) -> tensor<*xf32>
}

// -----

// CHECK-LABEL: @test_reducelogsumexp(%{{.*}}: tensor<?x?x?xf32>, %{{.*}}: tensor<?xi64>) -> tensor<*xf32>
func.func @test_reducelogsumexp(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?xi64>) -> tensor<*xf32> {
  %0 ="onnx.ReduceLogSumExp"(%arg0, %arg1) {keepdims = 0 : si64} : (tensor<?x?x?xf32>, tensor<?xi64>)-> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-NEXT: [[REDUCE_MAX:%.+]] = "onnx.ReduceMax"(%arg0, %arg1) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<?x?x?xf32>, tensor<?xi64>) -> tensor<*xf32>
  // CHECK-NEXT: [[SUB:%.+]] = "onnx.Sub"(%arg0, [[REDUCE_MAX]]) : (tensor<?x?x?xf32>, tensor<*xf32>) -> tensor<*xf32>
  // CHECK-NEXT: [[EXP:%.+]] = "onnx.Exp"([[SUB]]) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK-NEXT: [[REDUCE_SUM:%.+]] = "onnx.ReduceSum"([[EXP]], %arg1) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<*xf32>, tensor<?xi64>) -> tensor<*xf32>
  // CHECK-NEXT: [[LOG:%.+]] = "onnx.Log"([[REDUCE_SUM]]) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK-NEXT: [[SQUEEZE:%.+]] = "onnx.Squeeze"([[REDUCE_MAX]], %arg1) : (tensor<*xf32>, tensor<?xi64>) -> tensor<*xf32>
  // CHECK-NEXT: [[RES:%.+]] = "onnx.Add"([[LOG]], [[SQUEEZE]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  // CHECK-NEXT: onnx.Return [[RES]] : tensor<*xf32>
}

// -----

// CHECK-LABEL: @test_reducelogsumexp_keepdims(%{{.*}}: tensor<?x?x?xf32>, %{{.*}}: tensor<?xi64>) -> tensor<*xf32>
func.func @test_reducelogsumexp_keepdims(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?xi64>) -> tensor<*xf32> {
  %0 ="onnx.ReduceLogSumExp"(%arg0, %arg1) {keepdims = 1 : si64} : (tensor<?x?x?xf32>, tensor<?xi64>)-> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-NEXT: [[REDUCE_MAX:%.+]] = "onnx.ReduceMax"(%arg0, %arg1) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<?x?x?xf32>, tensor<?xi64>) -> tensor<*xf32>
  // CHECK-NEXT: [[SUB:%.+]] = "onnx.Sub"(%arg0, [[REDUCE_MAX]]) : (tensor<?x?x?xf32>, tensor<*xf32>) -> tensor<*xf32>
  // CHECK-NEXT: [[EXP:%.+]] = "onnx.Exp"([[SUB]]) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK-NEXT: [[REDUCE_SUM:%.+]] = "onnx.ReduceSum"([[EXP]], %arg1) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<*xf32>, tensor<?xi64>) -> tensor<*xf32>
  // CHECK-NEXT: [[LOG:%.+]] = "onnx.Log"([[REDUCE_SUM]]) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK-NEXT: [[RES:%.+]] = "onnx.Add"([[LOG]], [[REDUCE_MAX]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  // CHECK-NEXT: onnx.Return [[RES]] : tensor<*xf32>
}

// -----

// CHECK-LABEL: @test_reducesumsquare(%{{.*}}: tensor<?x?x?xf32>, %{{.*}}: tensor<?xi64>) -> tensor<*xf32>
func.func @test_reducesumsquare(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?xi64>) -> tensor<*xf32> {
  %0 ="onnx.ReduceSumSquare"(%arg0, %arg1) {keepdims = 0 : si64} : (tensor<?x?x?xf32>, tensor<?xi64>)-> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-NEXT: [[SQUARE:%.+]] =  "onnx.Mul"(%arg0, %arg0) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<*xf32>
  // CHECK-NEXT: %{{[0-9]+}} = "onnx.ReduceSum"([[SQUARE]], %arg1) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<*xf32>, tensor<?xi64>) -> tensor<*xf32>
}

// -----

// null
// CHECK-LABEL: func @test_scaler_null_float(%{{.*}}: tensor<3xf32>) -> tensor<3xf32> {
func.func @test_scaler_null_float(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %0 = "onnx.Scaler"(%arg0) : (tensor<3xf32>) -> tensor<3xf32>
  onnx.Return %0 : tensor<3xf32>

  // CHECK-NEXT: onnx.Return %arg0 : tensor<3xf32>
}

// -----

// null not float
// CHECK-LABEL: func @test_scaler_null(%{{.*}}: tensor<3xi32>) -> tensor<3xf32> {
func.func @test_scaler_null(%arg0: tensor<3xi32>) -> tensor<3xf32> {
  %0 = "onnx.Scaler"(%arg0) : (tensor<3xi32>) -> tensor<3xf32>
  onnx.Return %0 : tensor<3xf32>

  // CHECK-NEXT: %0 = "onnx.Cast"(%arg0) {saturate = 1 : si64, to = f32} : (tensor<3xi32>) -> tensor<3xf32>
  // CHECK-NEXT: onnx.Return %0 : tensor<3xf32>
}

// -----

// scaler no offset
// CHECK-LABEL: func @test_scaler_no_offset(%{{.*}}: tensor<3xf32>) -> tensor<3xf32> {
func.func @test_scaler_no_offset(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %0 = "onnx.Scaler"(%arg0) {scale = [3.125000e-02 : f32, 0.0909090936 : f32, 0.0333333351 : f32]} : (tensor<3xf32>) -> tensor<3xf32>
  onnx.Return %0 : tensor<3xf32>

  // CHECK-NEXT: %0 = onnx.Constant dense<[3.125000e-02, 0.0909090936, 0.0333333351]> : tensor<3xf32>
  // CHECK-NEXT: %1 = "onnx.Mul"(%arg0, %0) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  // CHECK-NEXT: onnx.Return %1 : tensor<3xf32>
}

// -----

// scaler no offset, int input
// CHECK-LABEL:  func.func @test_scaler_no_offset2
func.func @test_scaler_no_offset2(%arg0: tensor<3xi32>) -> tensor<3xf32> {
  %0 = "onnx.Scaler"(%arg0) {scale = [3.125000e-02 : f32, 0.0909090936 : f32, 0.0333333351 : f32]} : (tensor<3xi32>) -> tensor<3xf32>
  onnx.Return %0 : tensor<3xf32>
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3xi32>) -> tensor<3xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[3.125000e-02, 0.0909090936, 0.0333333351]> : tensor<3xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Cast"([[PARAM_0_]]) {saturate = 1 : si64, to = f32} : (tensor<3xi32>) -> tensor<*xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Mul"([[VAR_1_]], [[VAR_0_]]) : (tensor<*xf32>, tensor<3xf32>) -> tensor<3xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<3xf32>
}

// -----

// scaler no scale
// CHECK-LABEL: func @test_scaler_no_scale(%{{.*}}: tensor<3xf32>) -> tensor<3xf32> {
func.func @test_scaler_no_scale(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %0 = "onnx.Scaler"(%arg0) {offset = [1986.99939 : f32, 0.99999988 : f32, 0.999999701 : f32]} : (tensor<3xf32>) -> tensor<3xf32>
  onnx.Return %0 : tensor<3xf32>

  // CHECK-NEXT: %0 = onnx.Constant dense<[1986.99939, 0.99999988, 0.999999701]> : tensor<3xf32>
  // CHECK-NEXT: %1 = "onnx.Sub"(%arg0, %0) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  // CHECK-NEXT: onnx.Return %1 : tensor<3xf32>
}

// -----

// scaler no scale, int input
// CHECK-LABEL: func @test_scaler_no_scale2
func.func @test_scaler_no_scale2(%arg0: tensor<3xi32>) -> tensor<3xf32> {
  %0 = "onnx.Scaler"(%arg0) {offset = [1986.99939 : f32, 0.99999988 : f32, 0.999999701 : f32]} : (tensor<3xi32>) -> tensor<3xf32>
  onnx.Return %0 : tensor<3xf32>

// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3xi32>) -> tensor<3xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1986.99939, 0.99999988, 0.999999701]> : tensor<3xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Cast"([[PARAM_0_]]) {saturate = 1 : si64, to = f32} : (tensor<3xi32>) -> tensor<*xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Sub"([[VAR_1_]], [[VAR_0_]]) : (tensor<*xf32>, tensor<3xf32>) -> tensor<3xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<3xf32>
}

// -----

// normal scaler
// CHECK-LABEL: func @test_scaler_normal
func.func @test_scaler_normal(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %0 = "onnx.Scaler"(%arg0) {offset = [1986.99939 : f32, 0.99999988 : f32, 0.999999701 : f32], scale = [3.125000e-02 : f32, 0.0909090936 : f32, 0.0333333351 : f32]} : (tensor<3xf32>) -> tensor<3xf32>
  onnx.Return %0 : tensor<3xf32>

// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3xf32>) -> tensor<3xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[3.125000e-02, 0.0909090936, 0.0333333351]> : tensor<3xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[1986.99939, 0.99999988, 0.999999701]> : tensor<3xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Sub"([[PARAM_0_]], [[VAR_1_]]) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Mul"([[VAR_2_]], [[VAR_0_]]) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
// CHECK:           onnx.Return [[VAR_3_]] : tensor<3xf32>
}

// -----

// normal scaler, int input
// CHECK-LABEL: func @test_scaler_normal2
func.func @test_scaler_normal2(%arg0: tensor<3xi32>) -> tensor<3xf32> {
  %0 = "onnx.Scaler"(%arg0) {offset = [1986.99939 : f32, 0.99999988 : f32, 0.999999701 : f32], scale = [3.125000e-02 : f32, 0.0909090936 : f32, 0.0333333351 : f32]} : (tensor<3xi32>) -> tensor<3xf32>
  onnx.Return %0 : tensor<3xf32>

// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3xi32>) -> tensor<3xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[3.125000e-02, 0.0909090936, 0.0333333351]> : tensor<3xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[1986.99939, 0.99999988, 0.999999701]> : tensor<3xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Cast"([[PARAM_0_]]) {saturate = 1 : si64, to = f32} : (tensor<3xi32>) -> tensor<*xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Sub"([[VAR_2_]], [[VAR_1_]]) : (tensor<*xf32>, tensor<3xf32>) -> tensor<*xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Mul"([[VAR_3_]], [[VAR_0_]]) : (tensor<*xf32>, tensor<3xf32>) -> tensor<3xf32>
// CHECK:           onnx.Return [[VAR_4_]] : tensor<3xf32>
}

// -----

// normal scaler with constant offset and scale
// CHECK-LABEL: func @test_scaler_constant
func.func @test_scaler_constant(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %0 = "onnx.Scaler"(%arg0) {offset = [1986.99939 : f32], scale = [3.125000e-02 : f32]} : (tensor<3xf32>) -> tensor<3xf32>
  onnx.Return %0 : tensor<3xf32>

// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3xf32>) -> tensor<3xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<3.125000e-02> : tensor<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1986.99939> : tensor<1xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Sub"([[PARAM_0_]], [[VAR_1_]]) : (tensor<3xf32>, tensor<1xf32>) -> tensor<3xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Mul"([[VAR_2_]], [[VAR_0_]]) : (tensor<3xf32>, tensor<1xf32>) -> tensor<3xf32>
// CHECK:           onnx.Return [[VAR_3_]] : tensor<3xf32>
}

// -----

// Rewrite LogSoftmax using Log and Softmax.
func.func @test_logsoftmax(%arg0 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.LogSoftmax"(%arg0) {axis=1: si64} : (tensor<10x10xf32>) -> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_logsoftmax
  // CHECK: [[SOFTMAX:%.+]] = "onnx.Softmax"(%arg0) {axis = 1 : si64} : (tensor<10x10xf32>) -> tensor<*xf32>
  // CHECK: [[RES:%.+]] = "onnx.Log"([[SOFTMAX]]) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK: onnx.Return [[RES]] : tensor<*xf32>
}

// -----

func.func @test_upsample(%arg0: tensor<1x1x2x2xf32>, %arg1: tensor<4xf32>) -> tensor<1x1x4x6xf32> {
  %0 = "onnx.Upsample"(%arg0, %arg1) {mode = "nearest"} : (tensor<1x1x2x2xf32>, tensor<4xf32>) -> tensor<1x1x4x6xf32>
  onnx.Return %0 : tensor<1x1x4x6xf32>
// CHECK-LABEL:  func.func @test_upsample
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x2x2xf32>, [[PARAM_1_:%.+]]: tensor<4xf32>) -> tensor<1x1x4x6xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_1_:%.+]] = "onnx.Resize"([[PARAM_0_]], [[VAR_0_]], [[PARAM_1_]], [[VAR_0_]]) {antialias = 0 : si64, coordinate_transformation_mode = "asymmetric", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "nearest", nearest_mode = "round_prefer_floor"} : (tensor<1x1x2x2xf32>, none, tensor<4xf32>, none) -> tensor<1x1x4x6xf32>
// CHECK:           onnx.Return [[VAR_1_]] : tensor<1x1x4x6xf32>
}

// -----

func.func @test_upsamplev7(%arg0: tensor<1x1x2x2xf32>) -> tensor<1x1x4x6xf32> {
  %0 = "onnx.UpsampleV7"(%arg0) {mode = "nearest", scales = [0.1 : f32, 0.2 : f32, 0.3 : f32, 0.4 : f32]} : (tensor<1x1x2x2xf32>) -> tensor<1x1x4x6xf32>
  onnx.Return %0 : tensor<1x1x4x6xf32>
// CHECK-LABEL:  func.func @test_upsamplev7
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x2x2xf32>) -> tensor<1x1x4x6xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[1.000000e-01, 2.000000e-01, 3.000000e-01, 4.000000e-01]> : tensor<4xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Resize"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]], [[VAR_0_]]) {antialias = 0 : si64, coordinate_transformation_mode = "asymmetric", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "nearest", nearest_mode = "round_prefer_floor"} : (tensor<1x1x2x2xf32>, none, tensor<4xf32>, none) -> tensor<1x1x4x6xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<1x1x4x6xf32>
}

// -----

func.func @test_padv2(%arg0: tensor<1x3x224x224xf32>) -> tensor<*xf32> {
    %0 = "onnx.PadV2"(%arg0) {mode = "reflect", pads = [0, 0, 4, 4, 0, 0, 4, 4]} : (tensor<1x3x224x224xf32>) -> tensor<*xf32>
    onnx.Return %0 : tensor<*xf32>
    // CHECK-LABEL: test_padv2
    // CHECK: [[PAD:%.+]] = onnx.Constant dense<[0, 0, 4, 4, 0, 0, 4, 4]> : tensor<8xi64>
    // CHECK: [[CONSTANT_VALUE:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<1xf32>
    // CHECK: [[NONE:%.+]] = "onnx.NoValue"() {value} : () -> none
    // CHECK: [[RES:%.+]] = "onnx.Pad"(%arg0, [[PAD]], [[CONSTANT_VALUE]], [[NONE]]) {mode = "reflect"} : (tensor<1x3x224x224xf32>, tensor<8xi64>, tensor<1xf32>, none) -> tensor<*xf32>
    // CHECK: onnx.Return [[RES]] : tensor<*xf32>
}

// -----

func.func @test_resizev10(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<4xf32>) -> tensor<*xf32> {
  %0 = "onnx.ResizeV10"(%arg0, %arg1) {mode = "nearest"} : (tensor<1x2x3x4xf32>, tensor<4xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
// CHECK-LABEL:  func.func @test_resizev10
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x2x3x4xf32>, [[PARAM_1_:%.+]]: tensor<4xf32>) -> tensor<*xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_1_:%.+]] = "onnx.Resize"([[PARAM_0_]], [[VAR_0_]], [[PARAM_1_]], [[VAR_0_]]) {antialias = 0 : si64, coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "nearest", nearest_mode = "round_prefer_floor"} : (tensor<1x2x3x4xf32>, none, tensor<4xf32>, none) -> tensor<*xf32>
// CHECK:           onnx.Return [[VAR_1_]] : tensor<*xf32>
// CHECK:         }
}

// -----

func.func @test_resizev11(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>, %arg3: tensor<*xi64>) -> tensor<*xf32> {
  %0 = "onnx.ResizeV11"(%arg0, %arg1, %arg2, %arg3) {coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, mode = "nearest", nearest_mode = "floor"} : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xi64>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
  // CHECK-LABEL:  func @test_resizev11
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<*xf32>, [[PARAM_1_:%.+]]: tensor<*xf32>, [[PARAM_2_:%.+]]: tensor<*xf32>, [[PARAM_3_:%.+]]: tensor<*xi64>) -> tensor<*xf32> {
  // CHECK:           [[VAR_1_:%.+]] = "onnx.Resize"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[PARAM_3_]]) {antialias = 0 : si64, coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "nearest", nearest_mode = "floor"} : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xi64>) -> tensor<*xf32>
  // CHECK:           onnx.Return [[VAR_1_]] : tensor<*xf32>
}

// -----

func.func @test_resizev13(%arg0: tensor<*xf32>, %arg1: tensor<*xi64>) -> tensor<*xf32> {
  %0 = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.ResizeV13"(%arg0, %0, %0, %arg1) {coordinate_transformation_mode = "half_pixel", mode = "nearest", nearest_mode = "floor"} : (tensor<*xf32>, none, none, tensor<*xi64>) -> tensor<*xf32>
  onnx.Return %1 : tensor<*xf32>
  // CHECK-LABEL:  func @test_resizev13
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<*xf32>, [[PARAM_1_:%.+]]: tensor<*xi64>) -> tensor<*xf32> {
  // CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"
  // CHECK:           [[VAR_1_:%.+]] = "onnx.Resize"([[PARAM_0_]], [[VAR_0_]], [[VAR_0_]], [[PARAM_1_]])  {antialias = 0 : si64, coordinate_transformation_mode = "half_pixel", cubic_coeff_a = -7.500000e-01 : f32, exclude_outside = 0 : si64, extrapolation_value = 0.000000e+00 : f32, keep_aspect_ratio_policy = "stretch", mode = "nearest", nearest_mode = "floor"} : (tensor<*xf32>, none, none, tensor<*xi64>) -> tensor<*xf32>
}

// -----

func.func @test_seqence_construct_1(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> !onnx.Seq<tensor<*xf32>> {
  %0 = "onnx.SequenceConstruct"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> !onnx.Seq<tensor<*xf32>>
  onnx.Return %0 : !onnx.Seq<tensor<*xf32>>

  // CHECK-LABEL:  func @test_seqence_construct_1
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<*xf32>, [[PARAM_1_:%.+]]: tensor<*xf32>) -> !onnx.Seq<tensor<*xf32>> {
  // CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.SequenceEmpty"() {dtype = 1 : si64} : () -> !onnx.Seq<tensor<*xf32>>
  // CHECK-DAG:       [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
  // CHECK:           [[VAR_1_:%.+]] = "onnx.SequenceInsert"([[VAR_0_]], [[PARAM_0_]], [[VAR_cst_]]) : (!onnx.Seq<tensor<*xf32>>, tensor<*xf32>, none) -> !onnx.Seq<tensor<*xf32>>
  // CHECK:           [[VAR_2_:%.+]] = "onnx.SequenceInsert"([[VAR_1_]], [[PARAM_1_]], [[VAR_cst_]]) : (!onnx.Seq<tensor<*xf32>>, tensor<*xf32>, none) -> !onnx.Seq<tensor<*xf32>>
  // CHECK:           onnx.Return [[VAR_2_]] : !onnx.Seq<tensor<*xf32>>
}

// -----

func.func @test_seqence_construct_2(%arg0: tensor<*xi16>) -> !onnx.Seq<tensor<*xi16>> {
  %0 = "onnx.SequenceConstruct"(%arg0) : (tensor<*xi16>) -> !onnx.Seq<tensor<*xi16>>
  onnx.Return %0 : !onnx.Seq<tensor<*xi16>>

  // CHECK-LABEL:  func @test_seqence_construct_2
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<*xi16>) -> !onnx.Seq<tensor<*xi16>> {
  // CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.SequenceEmpty"() {dtype = 5 : si64} : () -> !onnx.Seq<tensor<*xi16>>
  // CHECK-DAG:       [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
  // CHECK:           [[VAR_1_:%.+]] = "onnx.SequenceInsert"([[VAR_0_]], [[PARAM_0_]], [[VAR_cst_]]) : (!onnx.Seq<tensor<*xi16>>, tensor<*xi16>, none) -> !onnx.Seq<tensor<*xi16>>
  // CHECK:           onnx.Return [[VAR_1_]] : !onnx.Seq<tensor<*xi16>>
}

// -----

func.func @test_clipv6(%arg0 : tensor<*xf32>) -> tensor<*xf32> {
  %0 = "onnx.ClipV6"(%arg0) {max = 6.000000e+00 : f32, min = 0.000000e+00 : f32} : (tensor<*xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

// CHECK-LABEL:  func.func @test_clipv6
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<6.000000e+00> : tensor<f32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Clip"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]]) : (tensor<*xf32>, tensor<f32>, tensor<f32>) -> tensor<*xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<*xf32>
}

// -----

func.func @test_splitV11(%arg0 : tensor<*xf32>) -> tensor<*xf32> {
  %0 = "onnx.SplitV11"(%arg0) {axis = 1 : si64, split = [1]} : (tensor<*xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

// CHECK-LABEL:  func.func @test_splitV11
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Split"([[PARAM_0_]], [[VAR_0_]]) {axis = 1 : si64} : (tensor<*xf32>, tensor<1xi64>) -> tensor<*xf32>
// CHECK:           onnx.Return [[VAR_1_]] : tensor<*xf32>
}

// -----

func.func @test_splitV11_no_split(%arg0 : tensor<*xf32>) -> tensor<*xf32> {
  %0 = "onnx.SplitV11"(%arg0) {axis = 1 : si64} : (tensor<*xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

// CHECK-LABEL:  func.func @test_splitV11_no_split
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_1_:%.+]] = "onnx.Split"([[PARAM_0_]], [[VAR_0_]]) {axis = 1 : si64} : (tensor<*xf32>, none) -> tensor<*xf32>
// CHECK:           onnx.Return [[VAR_1_]] : tensor<*xf32>
}

// -----

func.func @test_splitV13(%arg0 : tensor<*xf32>) -> tensor<*xf32> {
  %0 = onnx.Constant dense<1> : tensor<1xi64>
  %1 = "onnx.SplitV13"(%arg0, %0) {axis = 1 : si64} : (tensor<*xf32>, tensor<1xi64>) -> tensor<*xf32>
  onnx.Return %1 : tensor<*xf32>

// CHECK-LABEL:  func.func @test_splitV13
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Split"([[PARAM_0_]], [[VAR_0_]]) {axis = 1 : si64} : (tensor<*xf32>, tensor<1xi64>) -> tensor<*xf32>
// CHECK:           onnx.Return [[VAR_1_]] : tensor<*xf32>
}

// -----

func.func @test_squeezeV11(%arg0 : tensor<*xf32>) -> tensor<*xf32> {
  %0 = "onnx.SqueezeV11"(%arg0) {axes = [1]} : (tensor<*xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

// CHECK-LABEL:  func.func @test_squeezeV11
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Squeeze"([[PARAM_0_]], [[VAR_0_]]) : (tensor<*xf32>, tensor<1xi64>) -> tensor<*xf32>
// CHECK:           onnx.Return [[VAR_1_]] : tensor<*xf32>
}

// -----

func.func @test_squeezeV11_no_axes(%arg0 : tensor<*xf32>) -> tensor<*xf32> {
  %0 = "onnx.SqueezeV11"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

// CHECK-LABEL:  func.func @test_squeezeV11_no_axes
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_1_:%.+]] = "onnx.Squeeze"([[PARAM_0_]], [[VAR_0_]]) : (tensor<*xf32>, none) -> tensor<*xf32>
// CHECK:           onnx.Return [[VAR_1_]] : tensor<*xf32>
}

// -----

func.func @test_unsqueezeV11(%arg0 : tensor<*xf32>) -> tensor<*xf32> {
  %0 = "onnx.UnsqueezeV11"(%arg0) {axes = [1]} : (tensor<*xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

// CHECK-LABEL:  func.func @test_unsqueezeV11
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Unsqueeze"([[PARAM_0_]], [[VAR_0_]]) : (tensor<*xf32>, tensor<1xi64>) -> tensor<*xf32>
// CHECK:           onnx.Return [[VAR_1_]] : tensor<*xf32>
}

// -----

func.func @test_padV13(%arg0 : tensor<*xi64>, %arg1 : tensor<2xi64>) -> tensor<*xi64> {
  %0 = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.PadV13"(%arg0, %arg1, %0) : (tensor<*xi64>, tensor<2xi64>, none) -> tensor<*xi64>
  onnx.Return %1 : tensor<*xi64>
// CHECK-LABEL:  func.func @test_padV13
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<*xi64>, [[PARAM_1_:%.+]]: tensor<2xi64>) -> tensor<*xi64> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_1_:%.+]] = "onnx.Pad"([[PARAM_0_]], [[PARAM_1_]], [[VAR_0_]], [[VAR_0_]]) {mode = "constant"} : (tensor<*xi64>, tensor<2xi64>, none, none) -> tensor<*xi64>
// CHECK:           onnx.Return [[VAR_1_]] : tensor<*xi64>
}

// -----

func.func @test_scatter(%arg0: tensor<64x25600xf32>, %arg1: tensor<64x100xi64>, %arg2: tensor<64x100xf32>) -> tensor<*xf32> {
  %0 = "onnx.Scatter"(%arg0, %arg1, %arg2) {axis = 1 : si64} : (tensor<64x25600xf32>, tensor<64x100xi64>, tensor<64x100xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL:  func @test_scatter
  // CHECK-SAME:   ([[PARAM_0:%.+]]: tensor<64x25600xf32>, [[PARAM_1:%.+]]: tensor<64x100xi64>, [[PARAM_2:%.+]]: tensor<64x100xf32>) -> tensor<*xf32> {
  // CHECK-NEXT:      [[RES:%.+]] = "onnx.ScatterElements"(%arg0, %arg1, %arg2) {axis = 1 : si64, reduction = "none"} : (tensor<64x25600xf32>, tensor<64x100xi64>, tensor<64x100xf32>) -> tensor<*xf32>
  // CHECK-NEXT:      onnx.Return [[RES]] : tensor<*xf32>
}

// -----

func.func @concat_fuse_0(%arg0: tensor<?x20xf32>, %arg1: tensor<?x30xf32>) -> (tensor<2xi64>, tensor<50x?xf32>)
{
    %1 = "onnx.Concat"(%arg0, %arg1) {axis = 1 : si64} : (tensor<?x20xf32>, tensor<?x30xf32>) -> tensor<?x50xf32>
    %2 = "onnx.Transpose"(%1) {perm = [1, 0]} : (tensor<?x50xf32>) -> tensor<50x?xf32>
    %3 = "onnx.Shape"(%1) : (tensor<?x50xf32>) -> tensor<2xi64>
    onnx.Return %3, %2 : tensor<2xi64>, tensor<50x?xf32>
// CHECK-LABEL:  func.func @concat_fuse_0
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x20xf32>, [[PARAM_1_:%.+]]: tensor<?x30xf32>) -> (tensor<2xi64>, tensor<50x?xf32>) {
// CHECK:           [[shape_:%.+]], [[VAR_transposed_:%.+]] = "onnx.ConcatShapeTranspose"([[PARAM_0_]], [[PARAM_1_]]) {axis = 1 : si64, perm = [1, 0], start = 0 : si64} : (tensor<?x20xf32>, tensor<?x30xf32>) -> (tensor<2xi64>, tensor<50x?xf32>)
// CHECK:           onnx.Return [[shape_]], [[VAR_transposed_]] : tensor<2xi64>, tensor<50x?xf32>
// CHECK:         }
}

// -----

func.func @test_concatfuse_1(%arg0: tensor<?x20xf32>, %arg1: tensor<?x30xf32>) -> (tensor<?x50xf32>, tensor<2xi64>, tensor<50x?xf32>)
{
    %1 = "onnx.Concat"(%arg0, %arg1) {axis = 1 : si64} : (tensor<?x20xf32>, tensor<?x30xf32>) -> tensor<?x50xf32>
    %2 = "onnx.Transpose"(%1) {perm = [1, 0]} : (tensor<?x50xf32>) -> tensor<50x?xf32>
    %3 = "onnx.Shape"(%1) : (tensor<?x50xf32>) -> tensor<2xi64>
    %4 = "onnx.Sin"(%1) : (tensor<?x50xf32>) -> tensor<?x50xf32>
    onnx.Return %4, %3, %2 : tensor<?x50xf32>, tensor<2xi64>, tensor<50x?xf32>
// CHECK-LABEL:  func.func @test_concatfuse_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x20xf32>, [[PARAM_1_:%.+]]: tensor<?x30xf32>) -> (tensor<?x50xf32>, tensor<2xi64>, tensor<50x?xf32>) {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Concat"([[PARAM_0_]], [[PARAM_1_]]) {axis = 1 : si64} : (tensor<?x20xf32>, tensor<?x30xf32>) -> tensor<?x50xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Transpose"([[VAR_0_]]) {perm = [1, 0]} : (tensor<?x50xf32>) -> tensor<50x?xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Shape"([[VAR_0_]]) {start = 0 : si64} : (tensor<?x50xf32>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Sin"([[VAR_0_]]) : (tensor<?x50xf32>) -> tensor<?x50xf32>
// CHECK:           onnx.Return [[VAR_3_]], [[VAR_2_]], [[VAR_1_]] : tensor<?x50xf32>, tensor<2xi64>, tensor<50x?xf32>
}

// -----

func.func @test_concatfuse_2(%arg0: tensor<?x20xf32>, %arg1: tensor<?x30xf32>) -> (tensor<2xi64>, tensor<?x50xf32>)
{
    %1 = "onnx.Concat"(%arg0, %arg1) {axis = 1 : si64} : (tensor<?x20xf32>, tensor<?x30xf32>) -> tensor<?x50xf32>
    %3 = "onnx.Shape"(%1) : (tensor<?x50xf32>) -> tensor<2xi64>
    %4 = "onnx.Sin"(%1) : (tensor<?x50xf32>) -> tensor<?x50xf32>
    onnx.Return %3, %4 : tensor<2xi64>, tensor<?x50xf32>
// CHECK-LABEL:  func.func @test_concatfuse_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x20xf32>, [[PARAM_1_:%.+]]: tensor<?x30xf32>) -> (tensor<2xi64>, tensor<?x50xf32>) {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Concat"([[PARAM_0_]], [[PARAM_1_]]) {axis = 1 : si64} : (tensor<?x20xf32>, tensor<?x30xf32>) -> tensor<?x50xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Shape"([[VAR_0_]]) {start = 0 : si64} : (tensor<?x50xf32>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Sin"([[VAR_0_]]) : (tensor<?x50xf32>) -> tensor<?x50xf32>
// CHECK:           onnx.Return [[VAR_1_]], [[VAR_2_]] : tensor<2xi64>, tensor<?x50xf32>
// CHECK:         }
}

// -----

func.func @test_constantofshape_1(%arg0: tensor<?xi64>) -> tensor<*xi32> {
  %0 = onnx.ConstantOfShape(%arg0) {value = dense<1> : tensor<1xi32>} : (tensor<?xi64>) -> tensor<*xi32>
  return %0 : tensor<*xi32>

// CHECK-LABEL:  func.func @test_constantofshape_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?xi64>) -> tensor<*xi32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<i32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Expand"([[VAR_0_]], [[PARAM_0_]]) : (tensor<i32>, tensor<?xi64>) -> tensor<*xi32>
// CHECK:           return [[VAR_1_]] : tensor<*xi32>
// CHECK:         }
}

// -----

func.func @test_constantofshape_2(%arg0: tensor<?xi64>) -> tensor<*xi32> {
  %0 = onnx.ConstantOfShape(%arg0) : (tensor<?xi64>) -> tensor<*xi32>
  return %0 : tensor<*xi32>

// CHECK-LABEL:  func.func @test_constantofshape_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?xi64>) -> tensor<*xi32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Expand"([[VAR_0_]], [[PARAM_0_]]) : (tensor<f32>, tensor<?xi64>) -> tensor<*xi32>
// CHECK:           return [[VAR_1_]] : tensor<*xi32>
// CHECK:         }
}

// -----

func.func @test_hardswish_f32(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "onnx.HardSwish"(%arg0) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
// CHECK-LABEL:  func @test_hardswish_f32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK:           [[VAR_0_:%.+]] = "onnx.HardSigmoid"([[PARAM_0_]]) {alpha = 0.166666672 : f32, beta = 5.000000e-01 : f32} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Mul"([[VAR_0_]], [[PARAM_0_]]) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK:           return [[VAR_1_]] : tensor<?x?x?xf32>
}

// -----

func.func @test_constant_1() -> tensor<i64> {
  %0 = onnx.Constant {value_int = 1 : si64} : tensor<i64>
  onnx.Return %0 : tensor<i64>
// CHECK-LABEL:       func @test_constant_1
// CHECK:           [[VAR_0:%.+]] = onnx.Constant dense<1> : tensor<i64>
// CHECK:           onnx.Return [[VAR_0]] : tensor<i64>
}


// -----

func.func @test_constant_2() -> tensor<f32> {
  %0 = onnx.Constant {value_float = 2.0 : f32 } : tensor<f32>
  onnx.Return %0 : tensor<f32>
// CHECK-LABEL:     func @test_constant_2
// CHECK: [[VAR_0:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<f32>
// CHECK: onnx.Return [[VAR_0]] : tensor<f32>
}

// -----

func.func @test_constant_3() -> tensor<3xi64> {
  %0 = onnx.Constant {value_ints = [1, 2, 3] } : tensor<3xi64>
  onnx.Return %0 : tensor<3xi64>
// CHECK-LABEL:       func @test_constant_3
// CHECK-SAME:     () -> tensor<3xi64> {
// CHECK:           [[VAR_0:%.+]] = onnx.Constant dense<[1, 2, 3]> : tensor<3xi64>
// CHECK:           onnx.Return [[VAR_0]] : tensor<3xi64>
}

// -----

func.func @test_castlike(%arg0 : tensor<*xf32>, %arg1 : tensor<*xf16>) -> tensor<*xf16> {
  %0 = "onnx.CastLike"(%arg0, %arg1) {saturate = 1 : si64} : (tensor<*xf32>, tensor<*xf16>) -> tensor<*xf16> 
  "onnx.Return"(%0) : (tensor<*xf16>) -> ()

  // CHECK-LABEL: test_castlike
  // CHECK: [[RES:%.+]] = "onnx.Cast"(%arg0) {saturate = 1 : si64, to = f16} : (tensor<*xf32>) -> tensor<*xf16> 
  // CHECK: onnx.Return [[RES]] : tensor<*xf16>
}

// -----

func.func @test_castlike_i4(%arg0 : tensor<*xf32>, %arg1 : tensor<*xi4>) -> tensor<*xi4> {
  %0 = "onnx.CastLike"(%arg0, %arg1) {saturate = 1 : si64} : (tensor<*xf32>, tensor<*xi4>) -> tensor<*xi4> 
  "onnx.Return"(%0) : (tensor<*xi4>) -> ()

  // CHECK-LABEL: test_castlike_i4
  // CHECK: [[RES:%.+]] = "onnx.Cast"(%arg0) {saturate = 1 : si64, to = i4} : (tensor<*xf32>) -> tensor<*xi4> 
  // CHECK: onnx.Return [[RES]] : tensor<*xi4>
}

// -----

func.func @test_castlike_ui4(%arg0 : tensor<*xf32>, %arg1 : tensor<*xui4>) -> tensor<*xui4> {
  %0 = "onnx.CastLike"(%arg0, %arg1) {saturate = 1 : si64} : (tensor<*xf32>, tensor<*xui4>) -> tensor<*xui4> 
  "onnx.Return"(%0) : (tensor<*xui4>) -> ()

  // CHECK-LABEL: test_castlike_ui4
  // CHECK: [[RES:%.+]] = "onnx.Cast"(%arg0) {saturate = 1 : si64, to = ui4} : (tensor<*xf32>) -> tensor<*xui4> 
  // CHECK: onnx.Return [[RES]] : tensor<*xui4>
}


// -----

func.func @test_sum(%arg0: tensor<128x10xf32>, %arg1: tensor<64x128x10xf32>, %arg2: tensor<10xf32>, %arg3: tensor<64x1x1xf32>) -> tensor<64x128x10xf32> {
  %0 = "onnx.Sum"(%arg0, %arg1, %arg2, %arg3) : (tensor<128x10xf32>, tensor<64x128x10xf32>, tensor<10xf32>, tensor<64x1x1xf32>) -> tensor<64x128x10xf32>
  onnx.Return %0 : tensor<64x128x10xf32> 
  // CHECK-LABEL:       func @test_sum
  // CHECK-SAME:     (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}}, %[[ARG3:.*]]: {{.*}})
  // CHECK-NEXT:      %[[SUM0:.*]] = "onnx.Add"(%[[ARG0]], %[[ARG1]])
  // CHECK-NEXT:      %[[SUM1:.*]] = "onnx.Add"(%[[SUM0]], %[[ARG2]])
  // CHECK-NEXT:      %[[SUM2:.*]] = "onnx.Add"(%[[SUM1]], %[[ARG3]])
  // CHECK-NEXT:      onnx.Return %[[SUM2]]
}

// -----

func.func @test_sum_to_unranked(%arg0: tensor<128x10xf32>, %arg1: tensor<64x128x10xf32>, %arg2: tensor<10xf32>, %arg3: tensor<64x1x1xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sum"(%arg0, %arg1, %arg2, %arg3) : (tensor<128x10xf32>, tensor<64x128x10xf32>, tensor<10xf32>, tensor<64x1x1xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32> 
  // CHECK-LABEL:       func @test_sum
  // CHECK-SAME:     (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}}, %[[ARG3:.*]]: {{.*}})
  // CHECK-NEXT:      %[[SUM0:.*]] = "onnx.Add"(%[[ARG0]], %[[ARG1]])
  // CHECK-NEXT:      %[[SUM1:.*]] = "onnx.Add"(%[[SUM0]], %[[ARG2]])
  // CHECK-NEXT:      %[[SUM2:.*]] = "onnx.Add"(%[[SUM1]], %[[ARG3]])
  // CHECK-NEXT:      %[[CAST:.*]] = "onnx.Cast"(%[[SUM2]]) {saturate = 1 : si64, to = f32} : (tensor<64x128x10xf32>) -> tensor<*xf32>
  // CHECK-NEXT:      onnx.Return %[[CAST]]
}

// -----

func.func @test_sum_single_input(%arg0: tensor<64x128x10xf32>) -> tensor<64x128x10xf32> {
  %0 = "onnx.Sum"(%arg0) : (tensor<64x128x10xf32>) -> tensor<64x128x10xf32>
  onnx.Return %0 : tensor<64x128x10xf32> 
  // CHECK-LABEL:       func @test_sum_single_input
  // CHECK-SAME:     (%[[ARG0:.*]]: {{.*}})
  // CHECK-NEXT:      onnx.Return %[[ARG0]]
}

// -----

func.func @test_sum_single_input_to_unranked(%arg0: tensor<64x128x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sum"(%arg0) : (tensor<64x128x10xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32> 
  // CHECK-LABEL:       func @test_sum_single_input_to_unranked
  // CHECK-SAME:     (%[[ARG0:.*]]: {{.*}})
  // CHECK-NEXT:      %[[CAST:.*]] = "onnx.Cast"(%[[ARG0]]) {saturate = 1 : si64, to = f32} : (tensor<64x128x10xf32>) -> tensor<*xf32>
  // CHECK-NEXT:      onnx.Return %[[CAST]]
}
// -----

func.func @test_pad_slice_only_slice() -> tensor<3x1xf32> {
  %data = onnx.Constant dense<[[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]]> : tensor<3x2xf32>
  %pads = onnx.Constant dense<[0, -1, 0, 0]> : tensor<4xi64>
  %non = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.Pad"(%data, %pads, %non, %non) { mode = "constant" } : (tensor<3x2xf32>, tensor<4xi64>, none, none) -> tensor<3x1xf32>
  onnx.Return %1 : tensor<3x1xf32>
}
// CHECK-LABEL:  func.func @test_pad_slice_only_slice
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<{{.}}[1.000000e+00, 1.200000e+00], [2.300000e+00, 3.400000e+00], [4.500000e+00, 5.700000e+00]{{.}}> : tensor<3x2xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[3, 2]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_4_:%.+]] = "onnx.Slice"([[VAR_0_]], [[VAR_1_]], [[VAR_2_]], [[VAR_3_]], [[VAR_3_]]) : (tensor<3x2xf32>, tensor<2xi64>, tensor<2xi64>, none, none) -> tensor<3x1xf32>
// CHECK:           onnx.Return [[VAR_4_]] : tensor<3x1xf32>

// -----

func.func @test_pad_slice() -> tensor<4x1xf32> {
  %data = onnx.Constant dense<[[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]]> : tensor<3x2xf32>
  %pads = onnx.Constant dense<[0, -1, 1, 0]> : tensor<4xi64>
  %non = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.Pad"(%data, %pads, %non, %non) { mode = "constant" } : (tensor<3x2xf32>, tensor<4xi64>, none, none) -> tensor<4x1xf32>
  onnx.Return %1 : tensor<4x1xf32>
}
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[0, 0, 1, 0]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<{{.}}[1.000000e+00, 1.200000e+00], [2.300000e+00, 3.400000e+00], [4.500000e+00, 5.700000e+00]{{.}}> : tensor<3x2xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<[3, 2]> : tensor<2xi64>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Slice"([[VAR_1_]], [[VAR_3_]], [[VAR_4_]], [[VAR_2_]], [[VAR_2_]]) : (tensor<3x2xf32>, tensor<2xi64>, tensor<2xi64>, none, none) -> tensor<3x1xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Pad"([[VAR_5_]], [[VAR_0_]], [[VAR_2_]], [[VAR_2_]]) {mode = "constant"} : (tensor<3x1xf32>, tensor<4xi64>, none, none) -> tensor<4x1xf32>
// CHECK:           onnx.Return [[VAR_6_]] : tensor<4x1xf32>

// -----

func.func @test_pad_slice_dynamic(%data : tensor<*xf32>) -> tensor<*xf32> {
  // Just checks that we do not crash
  %pads = onnx.Constant dense<[0, -1, 1, 0]> : tensor<4xi64>
  %non = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.Pad"(%data, %pads, %non, %non) { mode = "constant" } : (tensor<*xf32>, tensor<4xi64>, none, none) -> tensor<*xf32>
  onnx.Return %1 : tensor<*xf32>
}
// CHECK-LABEL:  func.func @test_pad_slice_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<*xf32>)
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[0, -1, 1, 0]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_2_:%.+]] = "onnx.Pad"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]], [[VAR_1_]]) {mode = "constant"} : (tensor<*xf32>, tensor<4xi64>, none, none) -> tensor<*xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<*xf32>

// -----
func.func @test_scatter_nd_single_split_begin(%data : tensor<1x6x10x12xf32>, %updates : tensor<1x1x10x12xf32> ) -> tensor<1x6x10x12xf32> {
  %indices = onnx.Constant dense<[[[[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5], [0, 0, 6], [0, 0, 7], [0, 0, 8], [0, 0, 9]]]]> : tensor<1x1x10x3xi64>
  %0 = "onnx.ScatterND"(%data, %indices, %updates) {reduction = "none"} : (tensor<1x6x10x12xf32>, tensor<1x1x10x3xi64>, tensor<1x1x10x12xf32>) -> tensor<1x6x10x12xf32>
  onnx.Return %0 : tensor<1x6x10x12xf32>
}
// CHECK-LABEL:  func.func @test_scatter_nd_single_split_begin
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x6x10x12xf32>, [[PARAM_1_:%.+]]: tensor<1x1x10x12xf32>) -> tensor<1x6x10x12xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<[0, 1, 5]> : tensor<3xi64>
// CHECK:           [[VAR_1_:%.+]]:3 = "onnx.Split"([[PARAM_0_]], [[VAR_0_]]) {axis = 1 : si64} : (tensor<1x6x10x12xf32>, tensor<3xi64>) -> (tensor<1x0x10x12xf32>, tensor<1x1x10x12xf32>, tensor<1x5x10x12xf32>)
// CHECK:           [[VAR_2_:%.+]] = "onnx.Concat"([[VAR_1_]]#0, [[PARAM_1_]], [[VAR_1_]]#2) {axis = 1 : si64} : (tensor<1x0x10x12xf32>, tensor<1x1x10x12xf32>, tensor<1x5x10x12xf32>) -> tensor<1x6x10x12xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<1x6x10x12xf32>
// CHECK:         }

// -----
func.func @test_scatter_nd_single_split_end(%data : tensor<1x6x10x12xf32>, %updates : tensor<1x1x10x12xf32> ) -> tensor<1x6x10x12xf32> {
  %indices = onnx.Constant dense<[[[[0, 5, 0], [0, 5, 1], [0, 5, 2], [0, 5, 3], [0, 5, 4], [0, 5, 5], [0, 5, 6], [0, 5, 7], [0, 5, 8], [0, 5, 9]]]]> : tensor<1x1x10x3xi64>
  %0 = "onnx.ScatterND"(%data, %indices, %updates) {reduction = "none"} : (tensor<1x6x10x12xf32>, tensor<1x1x10x3xi64>, tensor<1x1x10x12xf32>) -> tensor<1x6x10x12xf32>
  onnx.Return %0 : tensor<1x6x10x12xf32>
}
// CHECK-LABEL:  func.func @test_scatter_nd_single_split_end
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x6x10x12xf32>, [[PARAM_1_:%.+]]: tensor<1x1x10x12xf32>) -> tensor<1x6x10x12xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<[5, 1, 0]> : tensor<3xi64>
// CHECK:           [[VAR_1_:%.+]]:3 = "onnx.Split"([[PARAM_0_]], [[VAR_0_]]) {axis = 1 : si64} : (tensor<1x6x10x12xf32>, tensor<3xi64>) -> (tensor<1x5x10x12xf32>, tensor<1x1x10x12xf32>, tensor<1x0x10x12xf32>)
// CHECK:           [[VAR_2_:%.+]] = "onnx.Concat"([[VAR_1_]]#0, [[PARAM_1_]], [[VAR_1_]]#2) {axis = 1 : si64} : (tensor<1x5x10x12xf32>, tensor<1x1x10x12xf32>, tensor<1x0x10x12xf32>) -> tensor<1x6x10x12xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<1x6x10x12xf32>
// CHECK:         }

// -----
func.func @test_scatter_nd_double_split(%data : tensor<1x6x10x12xf32>, %updates : tensor<1x1x10x12xf32> ) -> tensor<1x6x10x12xf32> {
  %indices = onnx.Constant dense<[[[[0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5], [0, 1, 6], [0, 1, 7], [0, 1, 8], [0, 1, 9]]]]> : tensor<1x1x10x3xi64>
  %0 = "onnx.ScatterND"(%data, %indices, %updates) {reduction = "none"} : (tensor<1x6x10x12xf32>, tensor<1x1x10x3xi64>, tensor<1x1x10x12xf32>) -> tensor<1x6x10x12xf32>
  onnx.Return %0 : tensor<1x6x10x12xf32>
}
// CHECK-LABEL:  func.func @test_scatter_nd_double_split
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x6x10x12xf32>, [[PARAM_1_:%.+]]: tensor<1x1x10x12xf32>) -> tensor<1x6x10x12xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<[1, 1, 4]> : tensor<3xi64>
// CHECK:           [[VAR_1_:%.+]]:3 = "onnx.Split"([[PARAM_0_]], [[VAR_0_]]) {axis = 1 : si64} : (tensor<1x6x10x12xf32>, tensor<3xi64>) -> (tensor<1x1x10x12xf32>, tensor<1x1x10x12xf32>, tensor<1x4x10x12xf32>)
// CHECK:           [[VAR_2_:%.+]] = "onnx.Concat"([[VAR_1_]]#0, [[PARAM_1_]], [[VAR_1_]]#2) {axis = 1 : si64} : (tensor<1x1x10x12xf32>, tensor<1x1x10x12xf32>, tensor<1x4x10x12xf32>) -> tensor<1x6x10x12xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<1x6x10x12xf32>
// CHECK:         }

// -----
func.func @test_scatter_nd_reduction(%data : tensor<1x6x10x12xf32>, %updates : tensor<1x1x10x12xf32> ) -> tensor<1x6x10x12xf32> {
  %indices = onnx.Constant dense<[[[[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5], [0, 0, 6], [0, 0, 7], [0, 0, 8], [0, 0, 9]]]]> : tensor<1x1x10x3xi64>
  %0 = "onnx.ScatterND"(%data, %indices, %updates) {reduction = "add"} : (tensor<1x6x10x12xf32>, tensor<1x1x10x3xi64>, tensor<1x1x10x12xf32>) -> tensor<1x6x10x12xf32>
  onnx.Return %0 : tensor<1x6x10x12xf32>
}
// CHECK-LABEL:  func.func @test_scatter_nd_reduction
// CHECK:        onnx.ScatterND

// -----
func.func @test_scatter_nd_not_const(%data : tensor<1x6x10x12xf32>, %updates : tensor<1x1x10x12xf32>, %indices : tensor<1x1x10x3xi64>  ) -> tensor<1x6x10x12xf32> {
  %0 = "onnx.ScatterND"(%data, %indices, %updates) {reduction = "none"} : (tensor<1x6x10x12xf32>, tensor<1x1x10x3xi64>, tensor<1x1x10x12xf32>) -> tensor<1x6x10x12xf32>
  onnx.Return %0 : tensor<1x6x10x12xf32>
}
// CHECK-LABEL:  func.func @test_scatter_nd_not_const
// CHECK:        onnx.ScatterND

// -----
func.func @test_scatter_nd_dynamic(%data : tensor<*xf32>, %updates : tensor<1x1x10x12xf32> ) -> tensor<*xf32> {
  %indices = onnx.Constant dense<[[[[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5], [0, 0, 6], [0, 0, 7], [0, 0, 8], [0, 0, 9]]]]> : tensor<1x1x10x3xi64>
  %0 = "onnx.ScatterND"(%data, %indices, %updates) {reduction = "none"} : (tensor<*xf32>, tensor<1x1x10x3xi64>, tensor<1x1x10x12xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
}
// CHECK-LABEL:  func.func @test_scatter_nd_dynamic
// CHECK:        onnx.ScatterND

// -----
func.func @test_scatter_nd_multi_dim_differ(%data : tensor<2x6x10x12xf32>, %updates : tensor<1x1x10x12xf32> ) -> tensor<2x6x10x12xf32> {
  %indices = onnx.Constant dense<[[[[0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5], [0, 1, 6], [0, 1, 7], [0, 1, 8], [0, 1, 9]]]]> : tensor<1x1x10x3xi64>
  %0 = "onnx.ScatterND"(%data, %indices, %updates) {reduction = "none"} : (tensor<2x6x10x12xf32>, tensor<1x1x10x3xi64>, tensor<1x1x10x12xf32>) -> tensor<2x6x10x12xf32>
  onnx.Return %0 : tensor<2x6x10x12xf32>
}
// CHECK-LABEL:  func.func @test_scatter_nd_multi_dim_differ
// CHECK:        onnx.ScatterND

// -----
func.func @test_scatter_nd_multi_dim_differ_multi_shift(%data : tensor<2x6x10x12xf32>, %updates : tensor<1x1x10x12xf32> ) -> tensor<2x6x10x12xf32> {
  %indices = onnx.Constant dense<[[[[1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 1, 3], [1, 1, 4], [1, 1, 5], [1, 1, 6], [1, 1, 7], [1, 1, 8], [1, 1, 9]]]]> : tensor<1x1x10x3xi64>
  %0 = "onnx.ScatterND"(%data, %indices, %updates) {reduction = "none"} : (tensor<2x6x10x12xf32>, tensor<1x1x10x3xi64>, tensor<1x1x10x12xf32>) -> tensor<2x6x10x12xf32>
  onnx.Return %0 : tensor<2x6x10x12xf32>
}
// CHECK-LABEL:  func.func @test_scatter_nd_multi_dim_differ_multi_shift
// CHECK:        onnx.ScatterND

// -----
func.func @test_scatter_nd_negative_shift(%data : tensor<1x6x10x12xf32>, %updates : tensor<1x1x10x12xf32> ) -> tensor<1x6x10x12xf32> {
  %indices = onnx.Constant dense<[[[[ 0, -1, 0], [ 0, -1, 1], [ 0, -1, 2], [ 0, -1, 3], [ 0, -1, 4], [ 0, -1, 5], [ 0, -1, 6], [ 0, -1, 7], [ 0, -1, 8], [ 0, -1, 9]]]]> : tensor<1x1x10x3xi64>
  %0 = "onnx.ScatterND"(%data, %indices, %updates) {reduction = "none"} : (tensor<1x6x10x12xf32>, tensor<1x1x10x3xi64>, tensor<1x1x10x12xf32>) -> tensor<1x6x10x12xf32>
  onnx.Return %0 : tensor<1x6x10x12xf32>
}
// CHECK-LABEL:  func.func @test_scatter_nd_negative_shift
// CHECK:        onnx.ScatterND

// -----
func.func @test_scatter_nd_full_overwrite(%data : tensor<1x6x10x12xf32>, %updates : tensor<1x6x10x12xf32> ) -> tensor<1x6x10x12xf32> {
  %indices = onnx.Constant dense<[[[[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5], [0, 0, 6], [0, 0, 7], [0, 0, 8], [0, 0, 9]], [[0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5], [0, 1, 6], [0, 1, 7], [0, 1, 8], [0, 1, 9]], [[0, 2, 0], [0, 2, 1], [0, 2, 2], [0, 2, 3], [0, 2, 4], [0, 2, 5], [0, 2, 6], [0, 2, 7], [0, 2, 8], [0, 2, 9]], [[0, 3, 0], [0, 3, 1], [0, 3, 2], [0, 3, 3], [0, 3, 4], [0, 3, 5], [0, 3, 6], [0, 3, 7], [0, 3, 8], [0, 3, 9]], [[0, 4, 0], [0, 4, 1], [0, 4, 2], [0, 4, 3], [0, 4, 4], [0, 4, 5], [0, 4, 6], [0, 4, 7], [0, 4, 8], [0, 4, 9]], [[0, 5, 0], [0, 5, 1], [0, 5, 2], [0, 5, 3], [0, 5, 4], [0, 5, 5], [0, 5, 6], [0, 5, 7], [0, 5, 8], [0, 5, 9]]]]> : tensor<1x6x10x3xi64>
  %0 = "onnx.ScatterND"(%data, %indices, %updates) {reduction = "none"} : (tensor<1x6x10x12xf32>, tensor<1x6x10x3xi64>, tensor<1x6x10x12xf32>) -> tensor<1x6x10x12xf32>
  onnx.Return %0 : tensor<1x6x10x12xf32>
}
// CHECK-LABEL:  func.func @test_scatter_nd_full_overwrite
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x6x10x12xf32>, [[PARAM_1_:%.+]]: tensor<1x6x10x12xf32>) -> tensor<1x6x10x12xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<[0, 12, 0]> : tensor<3xi64>
// CHECK:           [[VAR_1_:%.+]]:3 = "onnx.Split"([[PARAM_0_]], [[VAR_0_]]) {axis = 3 : si64} : (tensor<1x6x10x12xf32>, tensor<3xi64>) -> (tensor<1x6x10x0xf32>, tensor<1x6x10x12xf32>, tensor<1x6x10x0xf32>)
// CHECK:           [[VAR_2_:%.+]] = "onnx.Concat"([[VAR_1_]]#0, [[PARAM_1_]], [[VAR_1_]]#2) {axis = 3 : si64} : (tensor<1x6x10x0xf32>, tensor<1x6x10x12xf32>, tensor<1x6x10x0xf32>) -> tensor<1x6x10x12xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<1x6x10x12xf32>
// CHECK:         }

// -----
func.func @test_scatter_nd_multi_dim(%data : tensor<1x4x6xf32>, %updates : tensor<1x4x2xf32> ) -> tensor<1x4x6xf32> {
  %indices = onnx.Constant dense<[[[[0, 0, 0], [0, 0, 1]], [[0, 1, 0], [0, 1, 1]], [[0, 2, 0], [0, 2, 1]], [[0, 3, 0], [0, 3, 1]]]]> : tensor<1x4x2x3xi64>
  %0 = "onnx.ScatterND"(%data, %indices, %updates) {reduction = "none"} : (tensor<1x4x6xf32>, tensor<1x4x2x3xi64>, tensor<1x4x2xf32>) -> tensor<1x4x6xf32>
  onnx.Return %0 : tensor<1x4x6xf32>
}
// CHECK-LABEL:  func.func @test_scatter_nd_multi_dim
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x4x6xf32>, [[PARAM_1_:%.+]]: tensor<1x4x2xf32>) -> tensor<1x4x6xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<[0, 2, 4]> : tensor<3xi64>
// CHECK:           [[VAR_1_:%.+]]:3 = "onnx.Split"([[PARAM_0_]], [[VAR_0_]]) {axis = 2 : si64} : (tensor<1x4x6xf32>, tensor<3xi64>) -> (tensor<1x4x0xf32>, tensor<1x4x2xf32>, tensor<1x4x4xf32>)
// CHECK:           [[VAR_2_:%.+]] = "onnx.Concat"([[VAR_1_]]#0, [[PARAM_1_]], [[VAR_1_]]#2) {axis = 2 : si64} : (tensor<1x4x0xf32>, tensor<1x4x2xf32>, tensor<1x4x4xf32>) -> tensor<1x4x6xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<1x4x6xf32>
// CHECK:         }

// -----
func.func @test_scatter_nd_multi_dim_shift(%data : tensor<1x4x6xf32>, %updates : tensor<1x4x2xf32> ) -> tensor<1x4x6xf32> {
  %indices = onnx.Constant dense<[[[[0, 0, 1], [0, 0, 2]], [[0, 1, 1], [0, 1, 2]], [[0, 2, 1], [0, 2, 2]], [[0, 3, 1], [0, 3, 2]]]]> : tensor<1x4x2x3xi64>
  %0 = "onnx.ScatterND"(%data, %indices, %updates) {reduction = "none"} : (tensor<1x4x6xf32>, tensor<1x4x2x3xi64>, tensor<1x4x2xf32>) -> tensor<1x4x6xf32>
  onnx.Return %0 : tensor<1x4x6xf32>
}
// CHECK-LABEL:  func.func @test_scatter_nd_multi_dim_shift
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x4x6xf32>, [[PARAM_1_:%.+]]: tensor<1x4x2xf32>) -> tensor<1x4x6xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<[1, 2, 3]> : tensor<3xi64>
// CHECK:           [[VAR_1_:%.+]]:3 = "onnx.Split"([[PARAM_0_]], [[VAR_0_]]) {axis = 2 : si64} : (tensor<1x4x6xf32>, tensor<3xi64>) -> (tensor<1x4x1xf32>, tensor<1x4x2xf32>, tensor<1x4x3xf32>)
// CHECK:           [[VAR_2_:%.+]] = "onnx.Concat"([[VAR_1_]]#0, [[PARAM_1_]], [[VAR_1_]]#2) {axis = 2 : si64} : (tensor<1x4x1xf32>, tensor<1x4x2xf32>, tensor<1x4x3xf32>) -> tensor<1x4x6xf32>
// CHECK:           onnx.Return [[VAR_2_]] : tensor<1x4x6xf32>
// CHECK:         }

// -----
func.func @test_scatter_nd_multi_dim_not_in_order(%data : tensor<1x4x6xf32>, %updates : tensor<1x4x2xf32> ) -> tensor<1x4x6xf32> {
  %indices = onnx.Constant dense<[[[[0, 0, 0], [0, 1, 1]], [[0, 1, 0], [0, 0, 1]], [[0, 2, 0], [0, 2, 1]], [[0, 3, 0], [0, 3, 1]]]]> : tensor<1x4x2x3xi64>
  %0 = "onnx.ScatterND"(%data, %indices, %updates) {reduction = "none"} : (tensor<1x4x6xf32>, tensor<1x4x2x3xi64>, tensor<1x4x2xf32>) -> tensor<1x4x6xf32>
  onnx.Return %0 : tensor<1x4x6xf32>
}
// CHECK-LABEL:  func.func @test_scatter_nd_multi_dim_not_in_order
// CHECK:        onnx.ScatterND

// -----
func.func @test_scatter_nd_single_not_in_order(%data : tensor<1x6x10x12xf32>, %updates : tensor<1x1x10x12xf32> ) -> tensor<1x6x10x12xf32> {
  %indices = onnx.Constant dense<[[[[0, 0, 0], [0, 0, 2], [0, 0, 1], [0, 0, 3], [0, 0, 4], [0, 0, 5], [0, 0, 6], [0, 0, 7], [0, 0, 8], [0, 0, 9]]]]> : tensor<1x1x10x3xi64>
  %0 = "onnx.ScatterND"(%data, %indices, %updates) {reduction = "none"} : (tensor<1x6x10x12xf32>, tensor<1x1x10x3xi64>, tensor<1x1x10x12xf32>) -> tensor<1x6x10x12xf32>
  onnx.Return %0 : tensor<1x6x10x12xf32>
}
// CHECK-LABEL:  func.func @test_scatter_nd_single_not_in_order
// CHECK:        onnx.ScatterND
// -----

func.func @sce_mean(%arg0: tensor<64x10xf32>, %arg1: tensor<64xi64>) -> tensor<f32> {
    %0 = "onnx.NoValue"() {value, weight} : () -> none
    %output, %log_prob = "onnx.SoftmaxCrossEntropyLoss"(%arg0, %arg1, %0) {reduction = "mean"} : (tensor<64x10xf32>, tensor<64xi64>, none) -> (tensor<f32>, none)
    onnx.Return %output : tensor<f32>
  // CHECK-LABEL:  func @sce_mean
  // CHECK-SAME:     (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}})
  // CHECK-DAG:      %[[NONE:.*]] = "onnx.NoValue"() {value}
  // CHECK-DAG:      %[[NUM_CLASSES:.*]] = onnx.Constant dense<10> : tensor<1xi64>
  // CHECK-DAG:      %[[ONE_HOT_VALS:.*]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  // CHECK-DAG:      %[[REDUCE_AXIS:.*]] = onnx.Constant dense<1> : tensor<1xi64>
  // CHECK:          %[[ONE_HOT_LABELS:.*]] = "onnx.OneHot"(%[[ARG1]], %[[NUM_CLASSES]], %[[ONE_HOT_VALS]]) {axis = 1 : si64} : ({{.*}}) -> tensor<64x10xi64>
  // CHECK-NEXT:     %[[ONE_HOT_LABELS_F:.*]] = "onnx.Cast"(%[[ONE_HOT_LABELS]]) {saturate = 1 : si64, to = f32}
  // CHECK-NEXT:     %[[SOFTMAX:.*]] = "onnx.Softmax"(%[[ARG0]]) {axis = 1 : si64}
  // CHECK-NEXT:     %[[LOG_SOFTMAX:.*]] = "onnx.Log"(%[[SOFTMAX]])
  // CHECK-NEXT:     %[[PROD:.*]] = "onnx.Mul"(%[[LOG_SOFTMAX]], %[[ONE_HOT_LABELS_F]])
  // CHECK-NEXT:     %[[SUM:.*]] = "onnx.ReduceSum"(%[[PROD]], %[[REDUCE_AXIS]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : ({{.*}}) -> tensor<64x1xf32>
  // CHECK-NEXT:     %[[MEAN:.*]] = "onnx.ReduceMean"(%[[SUM]], %[[NONE]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : ({{.*}}) -> tensor<f32>
  // CHECK-NEXT:     %[[LOSS:.*]] = "onnx.Neg"(%[[MEAN]])
  // CHECK-NEXT:     onnx.Return %[[LOSS]] : tensor<f32>
}

// -----

func.func @sce_mean_return_log_prob(%arg0: tensor<64x10xf32>, %arg1: tensor<64xi64>) -> (tensor<f32>, tensor<64x10xf32>) {
    %0 = "onnx.NoValue"() {value, weight} : () -> none
    %output, %log_prob = "onnx.SoftmaxCrossEntropyLoss"(%arg0, %arg1, %0) {reduction = "mean"} : (tensor<64x10xf32>, tensor<64xi64>, none) -> (tensor<f32>, tensor<64x10xf32>)
    onnx.Return %output, %log_prob : tensor<f32>, tensor<64x10xf32>
  // CHECK-LABEL:  func @sce_mean_return_log_prob
  // CHECK-SAME:     (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}})
  // CHECK-DAG:      %[[NONE:.*]] = "onnx.NoValue"() {value}
  // CHECK-DAG:      %[[NUM_CLASSES:.*]] = onnx.Constant dense<10> : tensor<1xi64>
  // CHECK-DAG:      %[[ONE_HOT_VALS:.*]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  // CHECK-DAG:      %[[REDUCE_AXIS:.*]] = onnx.Constant dense<1> : tensor<1xi64>
  // CHECK:          %[[ONE_HOT_LABELS:.*]] = "onnx.OneHot"(%[[ARG1]], %[[NUM_CLASSES]], %[[ONE_HOT_VALS]]) {axis = 1 : si64} : ({{.*}}) -> tensor<64x10xi64>
  // CHECK-NEXT:     %[[ONE_HOT_LABELS_F:.*]] = "onnx.Cast"(%[[ONE_HOT_LABELS]]) {saturate = 1 : si64, to = f32}
  // CHECK-NEXT:     %[[SOFTMAX:.*]] = "onnx.Softmax"(%[[ARG0]]) {axis = 1 : si64}
  // CHECK-NEXT:     %[[LOG_SOFTMAX:.*]] = "onnx.Log"(%[[SOFTMAX]])
  // CHECK-NEXT:     %[[PROD:.*]] = "onnx.Mul"(%[[LOG_SOFTMAX]], %[[ONE_HOT_LABELS_F]])
  // CHECK-NEXT:     %[[SUM:.*]] = "onnx.ReduceSum"(%[[PROD]], %[[REDUCE_AXIS]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : ({{.*}}) -> tensor<64x1xf32>
  // CHECK-NEXT:     %[[MEAN:.*]] = "onnx.ReduceMean"(%[[SUM]], %[[NONE]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : ({{.*}}) -> tensor<f32>
  // CHECK-NEXT:     %[[LOSS:.*]] = "onnx.Neg"(%[[MEAN]])
  // CHECK-NEXT:     onnx.Return %[[LOSS]], %[[LOG_SOFTMAX]] : tensor<f32>, tensor<64x10xf32>
}

// -----

func.func @sce_mean_with_weight_NCD1D2(%arg0: tensor<64x10x2x3xf32>, %arg1: tensor<64x2x3xi64>, %arg2: tensor<10xf32>) -> tensor<f32> {
    %output, %log_prob = "onnx.SoftmaxCrossEntropyLoss"(%arg0, %arg1, %arg2) {reduction = "mean"} : (tensor<64x10x2x3xf32>, tensor<64x2x3xi64>, tensor<10xf32>) -> (tensor<f32>, none)
    onnx.Return %output : tensor<f32>
  // CHECK-LABEL:  func @sce_mean_with_weight_NCD1D2
  // CHECK-SAME:     (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}})
  // CHECK-DAG:      %[[NONE:.*]] = "onnx.NoValue"() {value}
  // CHECK-DAG:      %[[NUM_CLASSES:.*]] = onnx.Constant dense<10> : tensor<1xi64>
  // CHECK-DAG:      %[[ONE_HOT_VALS:.*]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  // CHECK-DAG:      %[[UNSQUEEZE_AXES:.*]] = onnx.Constant dense<[0, 2, 3]> : tensor<3xi64>
  // CHECK-DAG:      %[[REDUCE_AXIS:.*]] = onnx.Constant dense<1> : tensor<1xi64>
  // CHECK-DAG:      %[[COLLAPSED_SHAPE:.*]] = onnx.Constant dense<[384, 10]> : tensor<2xi64>
  // CHECK-DAG:      %[[EXPANDED_WEIGHT_SHAPE:.*]] = onnx.Constant dense<[10, 1]> : tensor<2xi64>
  // CHECK-DAG:      %[[W_SHAPE:.*]] = onnx.Constant dense<[64, 2, 3]> : tensor<3xi64>
  // CHECK:          %[[ONE_HOT_LABELS:.*]] = "onnx.OneHot"(%[[ARG1]], %[[NUM_CLASSES]], %[[ONE_HOT_VALS]]) {axis = 1 : si64} : ({{.*}}) -> tensor<64x10x2x3xi64>
  // CHECK-NEXT:     %[[ONE_HOT_LABELS_F:.*]] = "onnx.Cast"(%[[ONE_HOT_LABELS]]) {saturate = 1 : si64, to = f32}
  // CHECK-NEXT:     %[[SOFTMAX:.*]] = "onnx.Softmax"(%[[ARG0]]) {axis = 1 : si64}
  // CHECK-NEXT:     %[[LOG_SOFTMAX:.*]] = "onnx.Log"(%[[SOFTMAX]])
  // CHECK-NEXT:     %[[PROD:.*]] = "onnx.Mul"(%[[LOG_SOFTMAX]], %[[ONE_HOT_LABELS_F]])
  // CHECK-NEXT:     %[[UNSQUEEZE_WEIGHT:.*]] = "onnx.Unsqueeze"(%[[ARG2]], %[[UNSQUEEZE_AXES]]) : ({{.*}}) -> tensor<1x10x1x1xf32>
  // CHECK-NEXT:     %[[WEIGHT_PROD:.*]] = "onnx.Mul"(%[[PROD]], %[[UNSQUEEZE_WEIGHT]])
  // CHECK-NEXT:     %[[SUM:.*]] = "onnx.ReduceSum"(%[[WEIGHT_PROD]], %[[REDUCE_AXIS]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : ({{.*}}) -> tensor<64x1x2x3xf32>
  // CHECK-NEXT:     %[[SUML:.*]] = "onnx.ReduceSum"(%[[SUM]], %[[NONE]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : ({{.*}}) -> tensor<f32>
  
  // This block is an `onnx.EinSum` expanded by a different pattern rewrite
  // CHECK-NEXT:     %[[TRANSPOSE_ONE_HOT:.*]] = "onnx.Transpose"(%[[ONE_HOT_LABELS_F]]) {perm = [0, 2, 3, 1]} : ({{.*}}) -> tensor<64x2x3x10xf32>
  // CHECK-NEXT:     %[[COLLAPSED_ONE_SHOT:.*]] = "onnx.Reshape"(%[[TRANSPOSE_ONE_HOT]], %[[COLLAPSED_SHAPE]]) {allowzero = 0 : si64} : ({{.*}}) -> tensor<384x10xf32>
  // CHECK-NEXT:     %[[EXPANDED_WEIGHT:.*]] = "onnx.Reshape"(%[[ARG2]], %[[EXPANDED_WEIGHT_SHAPE]]) {allowzero = 0 : si64} : ({{.*}}) -> tensor<10x1xf32>
  // CHECK-NEXT:     %[[MATMUL:.*]] = "onnx.MatMul"(%[[COLLAPSED_ONE_SHOT]], %[[EXPANDED_WEIGHT]]) : ({{.*}}) -> tensor<384x1xf32>
  // CHECK-NEXT:     %[[W:.*]] = "onnx.Reshape"(%[[MATMUL]], %[[W_SHAPE]]) {allowzero = 0 : si64} : ({{.*}}) -> tensor<64x2x3xf32>  
  
  // CHECK-NEXT:     %[[SUMW:.*]] = "onnx.ReduceSum"(%[[W]], %[[NONE]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : ({{.*}}) -> tensor<f32>
  // CHECK-NEXT:     %[[MEAN:.*]] = "onnx.Div"(%[[SUML]], %[[SUMW]])
  // CHECK-NEXT:     %[[LOSS:.*]] = "onnx.Neg"(%[[MEAN]])
  // CHECK-NEXT:     onnx.Return %[[LOSS]] : tensor<f32>
}

// -----

func.func @sce_mean_with_weight_NCD1D2_dynamic_num_classes(%arg0: tensor<64x?x2x3xf32>, %arg1: tensor<64x2x3xi64>, %arg2: tensor<?xf32>) -> tensor<f32> {
    %output, %log_prob = "onnx.SoftmaxCrossEntropyLoss"(%arg0, %arg1, %arg2) {reduction = "mean"} : (tensor<64x?x2x3xf32>, tensor<64x2x3xi64>, tensor<?xf32>) -> (tensor<f32>, none)
    onnx.Return %output : tensor<f32>
  // CHECK-LABEL:  func @sce_mean_with_weight_NCD1D2
  // CHECK-SAME:     (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}})
  // CHECK-DAG:      %[[NONE:.*]] = "onnx.NoValue"() {value}
  // CHECK-DAG:      %[[NUM_CLASSES:.*]] = "onnx.Dim"(%[[ARG0]]) {axis = 1 : si64} : (tensor<64x?x2x3xf32>) -> tensor<1xi64>
  // CHECK-DAG:      %[[ONE_HOT_VALS:.*]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  // CHECK-DAG:      %[[UNSQUEEZE_AXES:.*]] = onnx.Constant dense<[0, 2, 3]> : tensor<3xi64>
  // CHECK-DAG:     %[[REDUCE_AXIS:.*]] = onnx.Constant dense<1> : tensor<1xi64>
  // CHECK:          %[[ONE_HOT_LABELS:.*]] = "onnx.OneHot"(%[[ARG1]], %[[NUM_CLASSES]], %[[ONE_HOT_VALS]]) {axis = 1 : si64} : ({{.*}}) -> tensor<64x?x2x3xi64>
  // CHECK-NEXT:     %[[ONE_HOT_LABELS_F:.*]] = "onnx.Cast"(%[[ONE_HOT_LABELS]]) {saturate = 1 : si64, to = f32}
  // CHECK-NEXT:     %[[SOFTMAX:.*]] = "onnx.Softmax"(%[[ARG0]]) {axis = 1 : si64}
  // CHECK-NEXT:     %[[LOG_SOFTMAX:.*]] = "onnx.Log"(%[[SOFTMAX]])
  // CHECK-NEXT:     %[[PROD:.*]] = "onnx.Mul"(%[[LOG_SOFTMAX]], %[[ONE_HOT_LABELS_F]])
  // CHECK-NEXT:     %[[UNSQUEEZE_WEIGHT:.*]] = "onnx.Unsqueeze"(%[[ARG2]], %[[UNSQUEEZE_AXES]]) : ({{.*}}) -> tensor<1x?x1x1xf32>
  // CHECK-NEXT:     %[[WEIGHT_PROD:.*]] = "onnx.Mul"(%[[PROD]], %[[UNSQUEEZE_WEIGHT]])
  // CHECK-NEXT:     %[[SUM:.*]] = "onnx.ReduceSum"(%[[WEIGHT_PROD]], %[[REDUCE_AXIS]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : ({{.*}}) -> tensor<64x1x2x3xf32>
  // CHECK-NEXT:     %[[SUML:.*]] = "onnx.ReduceSum"(%[[SUM]], %[[NONE]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : ({{.*}}) -> tensor<f32>
  // CHECK-NEXT:     %[[S_WEIGHTS:.*]] = "onnx.Einsum"(%[[ONE_HOT_LABELS_F]], %[[ARG2]]) {equation = "ij...,j->i..."}
  // CHECK-NEXT:     %[[SUMW:.*]] = "onnx.ReduceSum"(%[[S_WEIGHTS]], %[[NONE]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : ({{.*}}) -> tensor<f32>
  // CHECK-NEXT:     %[[MEAN:.*]] = "onnx.Div"(%[[SUML]], %[[SUMW]])
  // CHECK-NEXT:     %[[LOSS:.*]] = "onnx.Neg"(%[[MEAN]])
  // CHECK-NEXT:     onnx.Return %[[LOSS]] : tensor<f32>
}

// -----

func.func @sce_sum(%arg0: tensor<64x10xf32>, %arg1: tensor<64xi64>) -> tensor<f32> {
    %0 = "onnx.NoValue"() {value, weight} : () -> none
    %output, %log_prob = "onnx.SoftmaxCrossEntropyLoss"(%arg0, %arg1, %0) {reduction = "sum"} : (tensor<64x10xf32>, tensor<64xi64>, none) -> (tensor<f32>, none)
    onnx.Return %output : tensor<f32>
  // CHECK-LABEL:  func @sce_sum
  // CHECK-SAME:     (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}})
  // CHECK-DAG:      %[[NONE:.*]] = "onnx.NoValue"() {value}
  // CHECK-DAG:      %[[NUM_CLASSES:.*]] = onnx.Constant dense<10> : tensor<1xi64>
  // CHECK-DAG:      %[[ONE_HOT_VALS:.*]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  // CHECK-DAG:     %[[REDUCE_AXIS:.*]] = onnx.Constant dense<1> : tensor<1xi64>
  // CHECK:          %[[ONE_HOT_LABELS:.*]] = "onnx.OneHot"(%[[ARG1]], %[[NUM_CLASSES]], %[[ONE_HOT_VALS]]) {axis = 1 : si64} : ({{.*}}) -> tensor<64x10xi64>
  // CHECK-NEXT:     %[[ONE_HOT_LABELS_F:.*]] = "onnx.Cast"(%[[ONE_HOT_LABELS]]) {saturate = 1 : si64, to = f32}
  // CHECK-NEXT:     %[[SOFTMAX:.*]] = "onnx.Softmax"(%[[ARG0]]) {axis = 1 : si64}
  // CHECK-NEXT:     %[[LOG_SOFTMAX:.*]] = "onnx.Log"(%[[SOFTMAX]])
  // CHECK-NEXT:     %[[PROD:.*]] = "onnx.Mul"(%[[LOG_SOFTMAX]], %[[ONE_HOT_LABELS_F]])
  // CHECK-NEXT:     %[[SUM:.*]] = "onnx.ReduceSum"(%[[PROD]], %[[REDUCE_AXIS]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : ({{.*}}) -> tensor<64x1xf32>
  // CHECK-NEXT:     %[[MEAN:.*]] = "onnx.ReduceSum"(%[[SUM]], %[[NONE]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : ({{.*}}) -> tensor<f32>
  // CHECK-NEXT:     %[[LOSS:.*]] = "onnx.Neg"(%[[MEAN]])
  // CHECK-NEXT:     onnx.Return %[[LOSS]] : tensor<f32>
}

// -----

func.func @sce_sum_with_weight(%arg0: tensor<64x10xf32>, %arg1: tensor<64xi64>, %arg2: tensor<10xf32>) -> tensor<f32> {
    %output, %log_prob = "onnx.SoftmaxCrossEntropyLoss"(%arg0, %arg1, %arg2) {reduction = "sum"} : (tensor<64x10xf32>, tensor<64xi64>, tensor<10xf32>) -> (tensor<f32>, none)
    onnx.Return %output : tensor<f32>
  // CHECK-LABEL:  func @sce_sum_with_weight
  // CHECK-SAME:     (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}})
  // CHECK-DAG:      %[[NONE:.*]] = "onnx.NoValue"() {value}
  // CHECK-DAG:      %[[NUM_CLASSES:.*]] = onnx.Constant dense<10> : tensor<1xi64>
  // CHECK-DAG:      %[[ONE_HOT_VALS:.*]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  // CHECK-DAG:      %[[UNSQUEEZE_AXES:.*]] = onnx.Constant dense<0> : tensor<1xi64>
  // CHECK-DAG:      %[[REDUCE_AXIS:.*]] = onnx.Constant dense<1> : tensor<1xi64>
  // CHECK:          %[[ONE_HOT_LABELS:.*]] = "onnx.OneHot"(%[[ARG1]], %[[NUM_CLASSES]], %[[ONE_HOT_VALS]]) {axis = 1 : si64} : ({{.*}}) -> tensor<64x10xi64>
  // CHECK-NEXT:     %[[ONE_HOT_LABELS_F:.*]] = "onnx.Cast"(%[[ONE_HOT_LABELS]]) {saturate = 1 : si64, to = f32}
  // CHECK-NEXT:     %[[SOFTMAX:.*]] = "onnx.Softmax"(%[[ARG0]]) {axis = 1 : si64}
  // CHECK-NEXT:     %[[LOG_SOFTMAX:.*]] = "onnx.Log"(%[[SOFTMAX]])
  // CHECK-NEXT:     %[[PROD:.*]] = "onnx.Mul"(%[[LOG_SOFTMAX]], %[[ONE_HOT_LABELS_F]])
  // CHECK-NEXT:     %[[UNSQUEEZE_WEIGHT:.*]] = "onnx.Unsqueeze"(%[[ARG2]], %[[UNSQUEEZE_AXES]]) : ({{.*}}) -> tensor<1x10xf32>
  // CHECK-NEXT:     %[[WEIGHT_PROD:.*]] = "onnx.Mul"(%[[PROD]], %[[UNSQUEEZE_WEIGHT]])
  // CHECK-NEXT:     %[[SUM:.*]] = "onnx.ReduceSum"(%[[WEIGHT_PROD]], %[[REDUCE_AXIS]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : ({{.*}}) -> tensor<64x1xf32>
  // CHECK-NEXT:     %[[MEAN:.*]] = "onnx.ReduceSum"(%[[SUM]], %[[NONE]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : ({{.*}}) -> tensor<f32>
  // CHECK-NEXT:     %[[LOSS:.*]] = "onnx.Neg"(%[[MEAN]])
  // CHECK-NEXT:     onnx.Return %[[LOSS]] : tensor<f32>
}

// -----

func.func @sce_none(%arg0: tensor<64x10x2x3xf32>, %arg1: tensor<64x2x3xi64>) -> tensor<64x2x3xf32> {
    %0 = "onnx.NoValue"() {value, weight} : () -> none
    %output, %log_prob = "onnx.SoftmaxCrossEntropyLoss"(%arg0, %arg1, %0) {reduction = "none"} : (tensor<64x10x2x3xf32>, tensor<64x2x3xi64>, none) -> (tensor<64x2x3xf32>, none)
    onnx.Return %output : tensor<64x2x3xf32>
  // CHECK-LABEL:  func @sce_none
  // CHECK-SAME:     (%[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}})
  // CHECK-DAG:      %[[NUM_CLASSES:.*]] = onnx.Constant dense<10> : tensor<1xi64>
  // CHECK-DAG:      %[[ONE_HOT_VALS:.*]] = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  // CHECK-DAG:     %[[REDUCE_AXIS:.*]] = onnx.Constant dense<1> : tensor<1xi64>
  // CHECK:          %[[ONE_HOT_LABELS:.*]] = "onnx.OneHot"(%[[ARG1]], %[[NUM_CLASSES]], %[[ONE_HOT_VALS]]) {axis = 1 : si64} : ({{.*}}) -> tensor<64x10x2x3xi64>
  // CHECK-NEXT:     %[[ONE_HOT_LABELS_F:.*]] = "onnx.Cast"(%[[ONE_HOT_LABELS]]) {saturate = 1 : si64, to = f32}
  // CHECK-NEXT:     %[[SOFTMAX:.*]] = "onnx.Softmax"(%[[ARG0]]) {axis = 1 : si64}
  // CHECK-NEXT:     %[[LOG_SOFTMAX:.*]] = "onnx.Log"(%[[SOFTMAX]])
  // CHECK-NEXT:     %[[PROD:.*]] = "onnx.Mul"(%[[LOG_SOFTMAX]], %[[ONE_HOT_LABELS_F]])
  // CHECK-NEXT:     %[[SUM:.*]] = "onnx.ReduceSum"(%[[PROD]], %[[REDUCE_AXIS]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : ({{.*}}) -> tensor<64x1x2x3xf32>
  // CHECK-NEXT:     %[[SQUEEZE:.*]] = "onnx.Squeeze"(%[[SUM]], %[[REDUCE_AXIS]]) : ({{.*}}) -> tensor<64x2x3xf32>
  // CHECK-NEXT:     %[[LOSS:.*]] = "onnx.Neg"(%[[SQUEEZE]])
  // CHECK-NEXT:     onnx.Return %[[LOSS]] : tensor<64x2x3xf32>
}

// -----

func.func @test_instancenorm(%arg0: tensor<2x3x4x5x6xf32>, %arg1: tensor<3xf32>, %arg2: tensor<3xf32>) -> tensor<2x3x4x5x6xf32> {
  %0 = "onnx.InstanceNormalization"(%arg0, %arg1, %arg2) {epsilon = 0.00999999977 : f32} : (tensor<2x3x4x5x6xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<2x3x4x5x6xf32>
  onnx.Return %0 : tensor<2x3x4x5x6xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_instancenorm
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4x5x6xf32>, [[PARAM_1_:%.+]]: tensor<3xf32>, [[PARAM_2_:%.+]]: tensor<3xf32>) -> tensor<2x3x4x5x6xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<[1, 2, 3]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Unsqueeze"([[PARAM_1_]], [[VAR_0_]]) : (tensor<3xf32>, tensor<3xi64>) -> tensor<3x1x1x1xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Unsqueeze"([[PARAM_2_]], [[VAR_0_]]) : (tensor<3xf32>, tensor<3xi64>) -> tensor<3x1x1x1xf32>
// CHECK:           [[Y_:%.+]], [[Mean_:%.+]], [[VAR_InvStdDev_:%.+]] = "onnx.LayerNormalization"([[PARAM_0_]], [[VAR_1_]], [[VAR_2_]]) {axis = 2 : si64, epsilon = 0.00999999977 : f32, stash_type = 1 : si64} : (tensor<2x3x4x5x6xf32>, tensor<3x1x1x1xf32>, tensor<3x1x1x1xf32>) -> (tensor<2x3x4x5x6xf32>, none, none)
// CHECK:           onnx.Return [[Y_]] : tensor<2x3x4x5x6xf32>
// CHECK:         }
}
