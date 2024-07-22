// RUN: onnx-mlir-opt --decompose-onnx %s -split-input-file | FileCheck %s

func.func @test_einsum_matmul(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x4x5xf32>) -> tensor<2x3x5xf32> {
  %0 = "onnx.Einsum"(%arg0, %arg1) {equation = "...ij,...jk"} : (tensor<2x3x4xf32>, tensor<2x4x5xf32>) -> tensor<2x3x5xf32>
  onnx.Return %0 : tensor<2x3x5xf32>
// CHECK-LABEL:  func @test_einsum_matmul
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xf32>, [[PARAM_1_:%.+]]: tensor<2x4x5xf32>) -> tensor<2x3x5xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "onnx.MatMul"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<2x3x4xf32>, tensor<2x4x5xf32>) -> tensor<2x3x5xf32>
// CHECK-NEXT:      onnx.Return [[VAR_0_]] : tensor<2x3x5xf32>
}

// -----

// "...ij,...jk" is not implemented with MatMul over the reduction axis j
// because j has dim 1 in the first argument and dim 4 in the second argument
// (like numpy.matmul, MatMul doesn't broadcast the reduction axis),
// instead we first Squeeze the j axis in the first argument and
// ReduceSum the j axis in the second argument, and then Mul the results
func.func @test_einsum_matmul_broadcast(%arg0: tensor<2x3x1xf32>, %arg1: tensor<1x4x5xf32>) -> tensor<2x3x5xf32> {
  %0 = "onnx.Einsum"(%arg0, %arg1) {equation = "...ij,...jk"} : (tensor<2x3x1xf32>, tensor<1x4x5xf32>) -> tensor<2x3x5xf32>
  onnx.Return %0 : tensor<2x3x5xf32>
// CHECK-LABEL:  func.func @test_einsum_matmul_broadcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x1xf32>, [[PARAM_1_:%.+]]: tensor<1x4x5xf32>) -> tensor<2x3x5xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<2> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Squeeze"([[PARAM_0_]], [[VAR_1_]]) : (tensor<2x3x1xf32>, tensor<1xi64>) -> tensor<2x3xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.ReduceSum"([[PARAM_1_]], [[VAR_0_]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x4x5xf32>, tensor<1xi64>) -> tensor<1x5xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Transpose"([[VAR_2_]]) {perm = [1, 0]} : (tensor<2x3xf32>) -> tensor<3x2xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Unsqueeze"([[VAR_4_]], [[VAR_1_]]) : (tensor<3x2xf32>, tensor<1xi64>) -> tensor<3x2x1xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Mul"([[VAR_5_]], [[VAR_3_]]) : (tensor<3x2x1xf32>, tensor<1x5xf32>) -> tensor<3x2x5xf32>
// CHECK:           [[VAR_7_:%.+]] = "onnx.Transpose"([[VAR_6_]]) {perm = [1, 0, 2]} : (tensor<3x2x5xf32>) -> tensor<2x3x5xf32>
// CHECK:           onnx.Return [[VAR_7_]] : tensor<2x3x5xf32>
}

// -----

func.func @test_einsum_transpose(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %0 = "onnx.Einsum"(%arg0) {equation = "ji"} : (tensor<2x3xf32>) -> tensor<3x2xf32>
  onnx.Return %0 : tensor<3x2xf32>

  // CHECK-LABEL:  func @test_einsum_transpose
  // CHECK-SAME:   ([[PARAM_0:%.+]]: tensor<2x3xf32>) -> tensor<3x2xf32> {
  // CHECK-NEXT:      [[RES:%.+]] = "onnx.Transpose"(%arg0) {perm = [1, 0]} : (tensor<2x3xf32>) -> tensor<3x2xf32>
  // CHECK-NEXT:      onnx.Return [[RES]] : tensor<3x2xf32>
}

// -----

func.func @test_einsum_transpose_last_first(%arg0: tensor<0x1x2xf32>) -> tensor<2x0x1xf32> {
  %0 = "onnx.Einsum"(%arg0) {equation = "...i->i..."} : (tensor<0x1x2xf32>) -> tensor<2x0x1xf32>
  onnx.Return %0 : tensor<2x0x1xf32>
// CHECK-LABEL:  func @test_einsum_transpose_last_first
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<0x1x2xf32>) -> tensor<2x0x1xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<> : tensor<2x0x1xf32>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<2x0x1xf32>
}

// -----

func.func @test_einsum_sum(%arg0: tensor<2x3xf32>) -> tensor<2xf32> {
  %0 = "onnx.Einsum"(%arg0) {equation = "ij->i"} : (tensor<2x3xf32>) -> tensor<2xf32>
  onnx.Return %0 : tensor<2xf32>
// CHECK-LABEL:  func @test_einsum_sum
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3xf32>) -> tensor<2xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.ReduceSum"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<2xf32>
// CHECK:           onnx.Return [[VAR_1_]] : tensor<2xf32>
}

// -----

func.func @test_einsum_mul3_broadcast(%arg0: tensor<1x3xf32>, %arg1: tensor<1x1xf32>, %arg2: tensor<2x1xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.Einsum"(%arg0, %arg1, %arg2) {equation = "...,...,..."} : (tensor<1x3xf32>, tensor<1x1xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>
  onnx.Return %0 : tensor<2x3xf32>
// CHECK-LABEL:  func @test_einsum_mul3_broadcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3xf32>, [[PARAM_1_:%.+]]: tensor<1x1xf32>, [[PARAM_2_:%.+]]: tensor<2x1xf32>) -> tensor<2x3xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<1x3xf32>, tensor<1x1xf32>) -> tensor<1x3xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Mul"([[VAR_0_]], [[PARAM_2_]]) : (tensor<1x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>
// CHECK:           onnx.Return [[VAR_1_]] : tensor<2x3xf32>
}

// -----

func.func @test_einsum_diagonal(%arg0: tensor<3x3xf32>) -> tensor<3xf32> {
  %0 = "onnx.Einsum"(%arg0) {equation = "ii->i"} : (tensor<3x3xf32>) -> tensor<3xf32>
  onnx.Return %0 : tensor<3xf32>

// CHECK-LABEL:  func.func @test_einsum_diagonal
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x3xf32>) -> tensor<3xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<{{\[\[true, false, false\], \[false, true, false\], \[false, false, true\]\]}}> : tensor<3x3xi1>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Where"([[VAR_1_]], [[PARAM_0_]], [[VAR_2_]]) : (tensor<3x3xi1>, tensor<3x3xf32>, tensor<f32>) -> tensor<3x3xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.ReduceSum"([[VAR_3_]], [[VAR_0_]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<3x3xf32>, tensor<1xi64>) -> tensor<3xf32>
// CHECK:           onnx.Return [[VAR_4_]] : tensor<3xf32>
}

// -----

func.func @test_einsum_trace(%arg0: tensor<3x3xf32>) -> tensor<f32> {
  %0 = "onnx.Einsum"(%arg0) {equation = "ii"} : (tensor<3x3xf32>) -> tensor<f32>
  onnx.Return %0 : tensor<f32>
// CHECK-LABEL:  func.func @test_einsum_trace
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x3xf32>) -> tensor<f32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<{{\[\[true, false, false\], \[false, true, false\], \[false, false, true\]\]}}> : tensor<3x3xi1>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Where"([[VAR_2_]], [[PARAM_0_]], [[VAR_3_]]) : (tensor<3x3xi1>, tensor<3x3xf32>, tensor<f32>) -> tensor<3x3xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.ReduceSum"([[VAR_4_]], [[VAR_1_]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<3x3xf32>, tensor<1xi64>) -> tensor<3xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.ReduceSum"([[VAR_5_]], [[VAR_0_]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<3xf32>, tensor<1xi64>) -> tensor<f32>
// CHECK:           onnx.Return [[VAR_6_]] : tensor<f32>
}

// -----

func.func @test_einsum_ibh_hnd(%arg0: tensor<128x1x1024xf16>, %arg1: tensor<1024x16x64xf16>) -> tensor<128x1x16x64xf16> {
  %0 = "onnx.Einsum"(%arg0, %arg1) {equation = "ibh,hnd->ibnd"} : (tensor<128x1x1024xf16>, tensor<1024x16x64xf16>) -> tensor<128x1x16x64xf16>
  onnx.Return %0 : tensor<128x1x16x64xf16>
// CHECK-LABEL:  func.func @test_einsum_ibh_hnd
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<128x1x1024xf16>, [[PARAM_1_:%.+]]: tensor<1024x16x64xf16>) -> tensor<128x1x16x64xf16> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[128, 1, 16, 64]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1024> : tensor<2xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[128, 1024]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Reshape"([[PARAM_0_]], [[VAR_2_]]) {allowzero = 0 : si64} : (tensor<128x1x1024xf16>, tensor<2xi64>) -> tensor<128x1024xf16>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Reshape"([[PARAM_1_]], [[VAR_1_]]) {allowzero = 0 : si64} : (tensor<1024x16x64xf16>, tensor<2xi64>) -> tensor<1024x1024xf16>
// CHECK:           [[VAR_5_:%.+]] = "onnx.MatMul"([[VAR_3_]], [[VAR_4_]]) : (tensor<128x1024xf16>, tensor<1024x1024xf16>) -> tensor<128x1024xf16>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Reshape"([[VAR_5_]], [[VAR_0_]]) {allowzero = 0 : si64} : (tensor<128x1024xf16>, tensor<4xi64>) -> tensor<128x1x16x64xf16>
// CHECK:           onnx.Return [[VAR_6_]] : tensor<128x1x16x64xf16>
}

// -----

// unsupported element type prevents Einsum decomposition
func.func @test_einsum_matmul(%arg0: tensor<2x3x4xi16>, %arg1: tensor<2x4x5xi16>) -> tensor<2x3x5xi16> {
  %0 = "onnx.Einsum"(%arg0, %arg1) {equation = "...ij,...jk"} : (tensor<2x3x4xi16>, tensor<2x4x5xi16>) -> tensor<2x3x5xi16>
  onnx.Return %0 : tensor<2x3x5xi16>
// CHECK-LABEL:  func.func @test_einsum_matmul
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xi16>, [[PARAM_1_:%.+]]: tensor<2x4x5xi16>) -> tensor<2x3x5xi16> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Einsum"([[PARAM_0_]], [[PARAM_1_]]) {equation = "...ij,...jk"} : (tensor<2x3x4xi16>, tensor<2x4x5xi16>) -> tensor<2x3x5xi16>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<2x3x5xi16>
// CHECK:         }
}

// -----

// unknown shapes prevent Einsum decomposition
func.func @test_einsum_qmark(%arg0: tensor<3x?xf32>) -> tensor<3xf32> {
  %0 = "onnx.Einsum"(%arg0) {equation = "ii->i"} : (tensor<3x?xf32>) -> tensor<3xf32>
  onnx.Return %0 : tensor<3xf32>
// CHECK-LABEL:  func.func @test_einsum_qmark
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x?xf32>) -> tensor<3xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Einsum"([[PARAM_0_]]) {equation = "ii->i"} : (tensor<3x?xf32>) -> tensor<3xf32>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<3xf32>
// CHECK:         }
}

// -----

// unknown shapes prevent Einsum decomposition
func.func @test_einsum_qmark1(%arg0: tensor<1x?xf32>) -> tensor<?xf32> {
  %0 = "onnx.Einsum"(%arg0) {equation = "ii->i"} : (tensor<1x?xf32>) -> tensor<?xf32>
  onnx.Return %0 : tensor<?xf32>
// CHECK-LABEL:  func.func @test_einsum_qmark1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x?xf32>) -> tensor<?xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Einsum"([[PARAM_0_]]) {equation = "ii->i"} : (tensor<1x?xf32>) -> tensor<?xf32>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<?xf32>
// CHECK:         }
}

// -----

// unknown result shape prevent Einsum decomposition
func.func @test_einsum_result_qmark(%arg0: tensor<1x1xf32>) -> tensor<?xf32> {
  %0 = "onnx.Einsum"(%arg0) {equation = "ii->i"} : (tensor<1x1xf32>) -> tensor<?xf32>
  onnx.Return %0 : tensor<?xf32>
// CHECK-LABEL:  func.func @test_einsum_result_qmark
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1xf32>) -> tensor<?xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Einsum"([[PARAM_0_]]) {equation = "ii->i"} : (tensor<1x1xf32>) -> tensor<?xf32>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<?xf32>
// CHECK:         }
}
