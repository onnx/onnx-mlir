// RUN: onnx-mlir-opt --decompose-onnx %s -split-input-file | FileCheck %s

func.func @test_einsum_matmul(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x4x5xf32>) -> tensor<2x3x5xf32> {
  %0 = "onnx.Einsum"(%arg0, %arg1) {equation = "...ij,...jk"} : (tensor<2x3x4xf32>, tensor<2x4x5xf32>) -> tensor<2x3x5xf32>
  return %0 : tensor<2x3x5xf32>
// CHECK-LABEL:  func @test_einsum_matmul
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xf32>, [[PARAM_1_:%.+]]: tensor<2x4x5xf32>) -> tensor<2x3x5xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "onnx.MatMul"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<2x3x4xf32>, tensor<2x4x5xf32>) -> tensor<2x3x5xf32>
// CHECK-NEXT:      return [[VAR_0_]] : tensor<2x3x5xf32>
}

// "...ij,...jk" is not implemented with MatMul over the reduction axis j
// because j has dim 1 in the first argument and dim 4 in the second argument
// (like numpy.matmul, MatMul doesn't broadcast the reduction axis),
// instead we first Squeeze the j axis in the first argument and
// ReduceSum the j axis in the second argument, and then Mul the results
func.func @test_einsum_matmul_broadcast(%arg0: tensor<2x3x1xf32>, %arg1: tensor<1x4x5xf32>) -> tensor<2x3x5xf32> {
  %0 = "onnx.Einsum"(%arg0, %arg1) {equation = "...ij,...jk"} : (tensor<2x3x1xf32>, tensor<1x4x5xf32>) -> tensor<2x3x5xf32>
  return %0 : tensor<2x3x5xf32>
// CHECK-LABEL:  func @test_einsum_matmul_broadcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x1xf32>, [[PARAM_1_:%.+]]: tensor<1x4x5xf32>) -> tensor<2x3x5xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = "onnx.Constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Squeeze"([[PARAM_0_]], [[VAR_0_]]) : (tensor<2x3x1xf32>, tensor<1xi64>) -> tensor<2x3xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.ReduceSum"([[PARAM_1_]], [[VAR_2_]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x4x5xf32>, tensor<1xi64>) -> tensor<1x5xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Transpose"([[VAR_1_]]) {perm = [1, 0]} : (tensor<2x3xf32>) -> tensor<3x2xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
// CHECK-NEXT:      [[VAR_6_:%.+]] = "onnx.Unsqueeze"([[VAR_4_]], [[VAR_5_]]) : (tensor<3x2xf32>, tensor<1xi64>) -> tensor<3x2x1xf32>
// CHECK-NEXT:      [[VAR_7_:%.+]] = "onnx.Mul"([[VAR_6_]], [[VAR_3_]]) : (tensor<3x2x1xf32>, tensor<1x5xf32>) -> tensor<3x2x5xf32>
// CHECK-NEXT:      [[VAR_8_:%.+]] = "onnx.Transpose"([[VAR_7_]]) {perm = [1, 0, 2]} : (tensor<3x2x5xf32>) -> tensor<2x3x5xf32>
// CHECK-NEXT:      return [[VAR_8_]] : tensor<2x3x5xf32>
}

func.func @test_einsum_transpose(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %0 = "onnx.Einsum"(%arg0) {equation = "ji"} : (tensor<2x3xf32>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>

  // CHECK-LABEL:  func @test_einsum_transpose
  // CHECK-SAME:   ([[PARAM_0:%.+]]: tensor<2x3xf32>) -> tensor<3x2xf32> {
  // CHECK-NEXT:      [[RES:%.+]] = "onnx.Transpose"(%arg0) {perm = [1, 0]} : (tensor<2x3xf32>) -> tensor<3x2xf32>
  // CHECK-NEXT:      return [[RES]] : tensor<3x2xf32>
}

func.func @test_einsum_transpose_last_first(%arg0: tensor<0x1x2xf32>) -> tensor<2x0x1xf32> {
  %0 = "onnx.Einsum"(%arg0) {equation = "...i->i..."} : (tensor<0x1x2xf32>) -> tensor<2x0x1xf32>
  return %0 : tensor<2x0x1xf32>
// CHECK-LABEL:  func @test_einsum_transpose_last_first
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<0x1x2xf32>) -> tensor<2x0x1xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Constant"() {value = dense<> : tensor<2x0x1xf32>} : () -> tensor<2x0x1xf32>
// CHECK:           return [[VAR_0_]] : tensor<2x0x1xf32>
}

func.func @test_einsum_sum(%arg0: tensor<2x3xf32>) -> tensor<2xf32> {
  %0 = "onnx.Einsum"(%arg0) {equation = "ij->i"} : (tensor<2x3xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
// CHECK-LABEL:  func @test_einsum_sum
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3xf32>) -> tensor<2xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.ReduceSum"([[PARAM_0_]], [[VAR_0_]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<2xf32>
// CHECK:           return [[VAR_1_]] : tensor<2xf32>
}

func.func @test_einsum_mul3_broadcast(%arg0: tensor<1x3xf32>, %arg1: tensor<1x1xf32>, %arg2: tensor<2x1xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.Einsum"(%arg0, %arg1, %arg2) {equation = "...,...,..."} : (tensor<1x3xf32>, tensor<1x1xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
// CHECK-LABEL:  func @test_einsum_mul3_broadcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3xf32>, [[PARAM_1_:%.+]]: tensor<1x1xf32>, [[PARAM_2_:%.+]]: tensor<2x1xf32>) -> tensor<2x3xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Mul"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<1x3xf32>, tensor<1x1xf32>) -> tensor<1x3xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Mul"([[VAR_0_]], [[PARAM_2_]]) : (tensor<1x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>
// CHECK:           return [[VAR_1_]] : tensor<2x3xf32>
}

func.func @test_einsum_diagonal(%arg0: tensor<3x3xf32>) -> tensor<3xf32> {
  %0 = "onnx.Einsum"(%arg0) {equation = "ii->i"} : (tensor<3x3xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>

  // CHECK-LABEL:  func @test_einsum_diagonal
  // CHECK-SAME:   ([[PARAM_0:%.+]]: tensor<3x3xf32>) -> tensor<3xf32> {
  // CHECK-NEXT:      [[MASK:%.+]] = "onnx.Constant"() {value = dense<{{\[\[true, false, false\], \[false, true, false\], \[false, false, true\]\]}}> : tensor<3x3xi1>} : () -> tensor<3x3xi1>
  // CHECK-NEXT:      [[ZERO:%.+]] = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK-NEXT:      [[WHER:%.+]] = "onnx.Where"([[MASK]], [[PARAM_0]], [[ZERO]]) : (tensor<3x3xi1>, tensor<3x3xf32>, tensor<f32>) -> tensor<3x3xf32>
  // CHECK-NEXT:      [[AXES:%.+]] = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
  // CHECK-NEXT:      [[RSUM:%.+]] = "onnx.ReduceSum"([[WHER]], [[AXES]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<3x3xf32>, tensor<1xi64>) -> tensor<3xf32>
  // CHECK-NEXT:      return [[RSUM]] : tensor<3xf32>
}

func.func @test_einsum_trace(%arg0: tensor<3x3xf32>) -> tensor<f32> {
  %0 = "onnx.Einsum"(%arg0) {equation = "ii"} : (tensor<3x3xf32>) -> tensor<f32>
  return %0 : tensor<f32>
// CHECK-LABEL:  func @test_einsum_trace
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x3xf32>) -> tensor<f32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.Constant"() {value = dense<{{\[\[true, false, false\], \[false, true, false\], \[false, false, true\]\]}}> : tensor<3x3xi1>} : () -> tensor<3x3xi1>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Where"([[VAR_0_]], [[PARAM_0_]], [[VAR_1_]]) : (tensor<3x3xi1>, tensor<3x3xf32>, tensor<f32>) -> tensor<3x3xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Constant"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.ReduceSum"([[VAR_2_]], [[VAR_3_]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<3x3xf32>, tensor<1xi64>) -> tensor<3xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
// CHECK:           [[VAR_6_:%.+]] = "onnx.ReduceSum"([[VAR_4_]], [[VAR_5_]]) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<3xf32>, tensor<1xi64>) -> tensor<f32>
// CHECK:           return [[VAR_6_]] : tensor<f32>
}
