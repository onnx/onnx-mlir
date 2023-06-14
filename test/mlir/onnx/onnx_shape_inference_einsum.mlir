// RUN: onnx-mlir-opt --shape-inference %s -split-input-file | FileCheck %s

// -----

func.func @test_einsum_matmul(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x4x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.Einsum"(%arg0, %arg1) {equation = "...ij,...jk"} : (tensor<2x3x4xf32>, tensor<2x4x5xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: func @test_einsum_matmul
  // CHECK: [[RES:%.+]] = "onnx.Einsum"(%arg0, %arg1) {equation = "...ij,...jk"} : (tensor<2x3x4xf32>, tensor<2x4x5xf32>) -> tensor<2x3x5xf32>
  // CHECK: onnx.Return [[RES]] : tensor<2x3x5xf32>
}

// -----

func.func @test_einsum_matmul_broadcast(%arg0: tensor<2x3x1xf32>, %arg1: tensor<1x4x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.Einsum"(%arg0, %arg1) {equation = "...ij,...jk"} : (tensor<2x3x1xf32>, tensor<1x4x5xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: func @test_einsum_matmul_broadcast
  // CHECK: [[RES:%.+]] = "onnx.Einsum"(%arg0, %arg1) {equation = "...ij,...jk"} : (tensor<2x3x1xf32>, tensor<1x4x5xf32>) -> tensor<2x3x5xf32>
  // CHECK: onnx.Return [[RES]] : tensor<2x3x5xf32>
}

// -----

func.func @test_einsum_transpose(%arg0: tensor<2x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.Einsum"(%arg0) {equation = "ji"} : (tensor<2x3xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: func @test_einsum_transpose
  // CHECK: [[RES:%.+]] = "onnx.Einsum"(%arg0) {equation = "ji"} : (tensor<2x3xf32>) -> tensor<3x2xf32>
  // CHECK: onnx.Return [[RES]] : tensor<3x2xf32>
}

// -----

func.func @test_einsum_transpose_last_first(%arg0: tensor<0x1x2xf32>) -> tensor<*xf32> {
  %0 = "onnx.Einsum"(%arg0) {equation = "...i->i..."} : (tensor<0x1x2xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: func @test_einsum_transpose_last_first
  // CHECK: [[RES:%.+]] = "onnx.Einsum"(%arg0) {equation = "...i->i..."} : (tensor<0x1x2xf32>) -> tensor<2x0x1xf32>
  // CHECK: onnx.Return [[RES]] : tensor<2x0x1xf32>
}

// -----

func.func @test_einsum_sum(%arg0: tensor<2x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.Einsum"(%arg0) {equation = "ij->i"} : (tensor<2x3xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: func @test_einsum_sum
  // CHECK: [[RES:%.+]] = "onnx.Einsum"(%arg0) {equation = "ij->i"} : (tensor<2x3xf32>) -> tensor<2xf32>
  // CHECK: onnx.Return [[RES]] : tensor<2xf32>
}

// -----

func.func @test_einsum_mul3_broadcast(%arg0: tensor<1x3xf32>, %arg1: tensor<1x1xf32>, %arg2: tensor<2x1xf32>) -> tensor<*xf32> {
  %0 = "onnx.Einsum"(%arg0, %arg1, %arg2) {equation = "...,...,..."} : (tensor<1x3xf32>, tensor<1x1xf32>, tensor<2x1xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: func @test_einsum_mul3_broadcast
  // CHECK: [[RES:%.+]] = "onnx.Einsum"(%arg0, %arg1, %arg2) {equation = "...,...,..."} : (tensor<1x3xf32>, tensor<1x1xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>
  // CHECK: onnx.Return [[RES]] : tensor<2x3xf32>
}

// -----

func.func @test_einsum_diagonal(%arg0: tensor<3x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.Einsum"(%arg0) {equation = "ii->i"} : (tensor<3x3xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: func @test_einsum_diagonal
  // CHECK: [[RES:%.+]] = "onnx.Einsum"(%arg0) {equation = "ii->i"} : (tensor<3x3xf32>) -> tensor<3xf32>
  // CHECK: onnx.Return [[RES]] : tensor<3xf32>
}

// -----

func.func @test_einsum_trace(%arg0: tensor<3x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.Einsum"(%arg0) {equation = "ii"} : (tensor<3x3xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: func @test_einsum_trace
  // CHECK: [[RES:%.+]] = "onnx.Einsum"(%arg0) {equation = "ii"} : (tensor<3x3xf32>) -> tensor<f32>
  // CHECK: onnx.Return [[RES]] : tensor<f32>
}

// -----

func.func @test_einsum_qmark(%arg0: tensor<3x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.Einsum"(%arg0) {equation = "ii->i"} : (tensor<3x?xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: func @test_einsum_qmark
  // CHECK: [[RES:%.+]] = "onnx.Einsum"(%arg0) {equation = "ii->i"} : (tensor<3x?xf32>) -> tensor<3xf32>
  // CHECK: onnx.Return [[RES]] : tensor<3xf32>
}

// -----

func.func @test_einsum_qmark1(%arg0: tensor<1x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.Einsum"(%arg0) {equation = "ii->i"} : (tensor<1x?xf32>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>

  // CHECK-LABEL: func @test_einsum_qmark1
  // CHECK: [[RES:%.+]] = "onnx.Einsum"(%arg0) {equation = "ii->i"} : (tensor<1x?xf32>) -> tensor<?xf32>
  // CHECK: onnx.Return [[RES]] : tensor<?xf32>
}
