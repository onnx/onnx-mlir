// RUN: onnx-mlir-opt --convert-onnx-to-linalg='linalg-ops=onnx.Transpose' %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --convert-onnx-to-linalg='linalg-ops=NONE' %s -split-input-file | FileCheck %s --check-prefix=NONE
// RUN: onnx-mlir-opt --convert-onnx-to-linalg %s -split-input-file | FileCheck %s --check-prefix=DEFAULT

// -----

func.func @transpose_explicit_perm(%arg0: tensor<2x3x4xf32>)
    -> tensor<2x4x3xf32> {
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1]} :
      (tensor<2x3x4xf32>) -> tensor<2x4x3xf32>
  return %0 : tensor<2x4x3xf32>

  // CHECK-LABEL: func.func @transpose_explicit_perm
  // CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<2x4x3xf32>
  // CHECK: [[RESULT:%.+]] = linalg.transpose ins(%arg0 : tensor<2x3x4xf32>) outs([[EMPTY]] : tensor<2x4x3xf32>) permutation = [0, 2, 1]
  // CHECK: return [[RESULT]] : tensor<2x4x3xf32>

  // NONE-LABEL: func.func @transpose_explicit_perm
  // NONE-NOT: linalg.transpose
  // NONE: "onnx.Transpose"(%arg0) {{.*}}perm = [0, 2, 1]

  // DEFAULT-LABEL: func.func @transpose_explicit_perm
  // DEFAULT: [[EMPTY:%.+]] = tensor.empty() : tensor<2x4x3xf32>
  // DEFAULT: [[RESULT:%.+]] = linalg.transpose ins(%arg0 : tensor<2x3x4xf32>) outs([[EMPTY]] : tensor<2x4x3xf32>) permutation = [0, 2, 1]
  // DEFAULT: return [[RESULT]] : tensor<2x4x3xf32>
}

// -----

func.func @transpose_default_perm(%arg0: tensor<2x3x4xi64>)
    -> tensor<4x3x2xi64> {
  %0 = "onnx.Transpose"(%arg0) :
      (tensor<2x3x4xi64>) -> tensor<4x3x2xi64>
  return %0 : tensor<4x3x2xi64>

  // CHECK-LABEL: func.func @transpose_default_perm
  // CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<4x3x2xi64>
  // CHECK: [[RESULT:%.+]] = linalg.transpose ins(%arg0 : tensor<2x3x4xi64>) outs([[EMPTY]] : tensor<4x3x2xi64>) permutation = [2, 1, 0]
  // CHECK: return [[RESULT]] : tensor<4x3x2xi64>

  // NONE-LABEL: func.func @transpose_default_perm
  // NONE-NOT: linalg.transpose
  // NONE: "onnx.Transpose"(%arg0)
}

// -----

func.func @transpose_default_perm_dynamic(%arg0: tensor<?x3x?xf32>)
    -> tensor<?x3x?xf32> {
  %0 = "onnx.Transpose"(%arg0) :
      (tensor<?x3x?xf32>) -> tensor<?x3x?xf32>
  return %0 : tensor<?x3x?xf32>

  // CHECK-LABEL: func.func @transpose_default_perm_dynamic
  // CHECK-DAG: [[DIM2:%.+]] = tensor.dim %arg0, %c2 : tensor<?x3x?xf32>
  // CHECK-DAG: [[DIM0:%.+]] = tensor.dim %arg0, %c0 : tensor<?x3x?xf32>
  // CHECK: [[EMPTY:%.+]] = tensor.empty([[DIM2]], [[DIM0]]) : tensor<?x3x?xf32>
  // CHECK: [[RESULT:%.+]] = linalg.transpose ins(%arg0 : tensor<?x3x?xf32>) outs([[EMPTY]] : tensor<?x3x?xf32>) permutation = [2, 1, 0]
  // CHECK: return [[RESULT]] : tensor<?x3x?xf32>
}

// -----

func.func @transpose_negative_perm(%arg0: tensor<2x3x4xf32>)
    -> tensor<4x2x3xf32> {
  %0 = "onnx.Transpose"(%arg0) {perm = [-1, 0, 1]} :
      (tensor<2x3x4xf32>) -> tensor<4x2x3xf32>
  return %0 : tensor<4x2x3xf32>

  // CHECK-LABEL: func.func @transpose_negative_perm
  // CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<4x2x3xf32>
  // CHECK: [[RESULT:%.+]] = linalg.transpose ins(%arg0 : tensor<2x3x4xf32>) outs([[EMPTY]] : tensor<4x2x3xf32>) permutation = [2, 0, 1]
  // CHECK: return [[RESULT]] : tensor<4x2x3xf32>
}

// -----

func.func @transpose_dynamic(%arg0: tensor<?x3x?xf32>)
    -> tensor<?x?x3xf32> {
  %0 = "onnx.Transpose"(%arg0) {perm = [2, 0, 1]} :
      (tensor<?x3x?xf32>) -> tensor<?x?x3xf32>
  return %0 : tensor<?x?x3xf32>

  // CHECK-LABEL: func.func @transpose_dynamic
  // CHECK-DAG: [[DIM2:%.+]] = tensor.dim %arg0, %c2 : tensor<?x3x?xf32>
  // CHECK-DAG: [[DIM0:%.+]] = tensor.dim %arg0, %c0 : tensor<?x3x?xf32>
  // CHECK: [[EMPTY:%.+]] = tensor.empty([[DIM2]], [[DIM0]]) : tensor<?x?x3xf32>
  // CHECK: [[RESULT:%.+]] = linalg.transpose ins(%arg0 : tensor<?x3x?xf32>) outs([[EMPTY]] : tensor<?x?x3xf32>) permutation = [2, 0, 1]
  // CHECK: return [[RESULT]] : tensor<?x?x3xf32>
}

// -----

func.func @transpose_rank_zero(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "onnx.Transpose"(%arg0) : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>

  // CHECK-LABEL: func.func @transpose_rank_zero
  // An empty permutation is an identity and folds away immediately.
  // CHECK-NOT: onnx.Transpose
  // CHECK-NOT: linalg.transpose
  // CHECK: return %arg0 : tensor<f32>
}

// -----

func.func @transpose_zero_extent(%arg0: tensor<2x0x3xf32>)
    -> tensor<3x2x0xf32> {
  %0 = "onnx.Transpose"(%arg0) {perm = [2, 0, 1]} :
      (tensor<2x0x3xf32>) -> tensor<3x2x0xf32>
  return %0 : tensor<3x2x0xf32>

  // CHECK-LABEL: func.func @transpose_zero_extent
  // CHECK: [[EMPTY:%.+]] = tensor.empty() : tensor<3x2x0xf32>
  // CHECK: [[RESULT:%.+]] = linalg.transpose ins(%arg0 : tensor<2x0x3xf32>) outs([[EMPTY]] : tensor<3x2x0xf32>) permutation = [2, 0, 1]
  // CHECK: return [[RESULT]] : tensor<3x2x0xf32>
}

// -----

func.func @transpose_unranked_not_lowered(%arg0: tensor<*xf32>)
    -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

  // CHECK-LABEL: func.func @transpose_unranked_not_lowered
  // CHECK-NOT: linalg.transpose
  // CHECK: "onnx.Transpose"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
}
