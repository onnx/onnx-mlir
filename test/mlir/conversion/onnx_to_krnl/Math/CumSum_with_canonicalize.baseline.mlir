// Baseline for CumSum_with_canonicalize.mlir (file-mode GroundLitTest.py).
// Each function computes CumSum along axis 0 without the CumSum lowering:
// y = T * x, where T is the n x n matrix of ones cut by Trilu. Inclusive is the
// lower triangle (k = 0), exclusive the strict lower triangle (k = -1), and
// exclusive reverse the strict upper triangle (upper = 1, k = 1).

// -----

// CHECK-LABEL: test_cumsum_axis_dim_1
func.func @test_cumsum_axis_dim_1(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %s = "onnx.Shape"(%arg0) {start = 0 : si64, end = 1 : si64} : (tensor<1x4xf32>) -> tensor<1xi64>
  %nn = "onnx.Concat"(%s, %s) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  %ones = "onnx.ConstantOfShape"(%nn) {value = dense<1.0> : tensor<1xf32>} : (tensor<2xi64>) -> tensor<?x?xf32>
  %k = onnx.Constant dense<0> : tensor<i64>
  %tri = "onnx.Trilu"(%ones, %k) {upper = 0 : si64} : (tensor<?x?xf32>, tensor<i64>) -> tensor<?x?xf32>
  %0 = "onnx.MatMul"(%tri, %arg0) : (tensor<?x?xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: test_cumsum_axis_dim_1_exclusive
func.func @test_cumsum_axis_dim_1_exclusive(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %s = "onnx.Shape"(%arg0) {start = 0 : si64, end = 1 : si64} : (tensor<1x4xf32>) -> tensor<1xi64>
  %nn = "onnx.Concat"(%s, %s) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  %ones = "onnx.ConstantOfShape"(%nn) {value = dense<1.0> : tensor<1xf32>} : (tensor<2xi64>) -> tensor<?x?xf32>
  %k = onnx.Constant dense<-1> : tensor<i64>
  %tri = "onnx.Trilu"(%ones, %k) {upper = 0 : si64} : (tensor<?x?xf32>, tensor<i64>) -> tensor<?x?xf32>
  %0 = "onnx.MatMul"(%tri, %arg0) : (tensor<?x?xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: test_cumsum_axis_dim_1_exclusive_reverse
func.func @test_cumsum_axis_dim_1_exclusive_reverse(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %s = "onnx.Shape"(%arg0) {start = 0 : si64, end = 1 : si64} : (tensor<1x4xf32>) -> tensor<1xi64>
  %nn = "onnx.Concat"(%s, %s) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  %ones = "onnx.ConstantOfShape"(%nn) {value = dense<1.0> : tensor<1xf32>} : (tensor<2xi64>) -> tensor<?x?xf32>
  %k = onnx.Constant dense<1> : tensor<i64>
  %tri = "onnx.Trilu"(%ones, %k) {upper = 1 : si64} : (tensor<?x?xf32>, tensor<i64>) -> tensor<?x?xf32>
  %0 = "onnx.MatMul"(%tri, %arg0) : (tensor<?x?xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----

// CHECK-LABEL: test_cumsum_axis_dim_0
func.func @test_cumsum_axis_dim_0(%arg0: tensor<0x4xf32>) -> tensor<0x4xf32> {
  %s = "onnx.Shape"(%arg0) {start = 0 : si64, end = 1 : si64} : (tensor<0x4xf32>) -> tensor<1xi64>
  %nn = "onnx.Concat"(%s, %s) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  %ones = "onnx.ConstantOfShape"(%nn) {value = dense<1.0> : tensor<1xf32>} : (tensor<2xi64>) -> tensor<?x?xf32>
  %k = onnx.Constant dense<0> : tensor<i64>
  %tri = "onnx.Trilu"(%ones, %k) {upper = 0 : si64} : (tensor<?x?xf32>, tensor<i64>) -> tensor<?x?xf32>
  %0 = "onnx.MatMul"(%tri, %arg0) : (tensor<?x?xf32>, tensor<0x4xf32>) -> tensor<0x4xf32>
  return %0 : tensor<0x4xf32>
}

// -----

// CHECK-LABEL: test_cumsum_dynamic_axis_dim
func.func @test_cumsum_dynamic_axis_dim(%arg0: tensor<?x4xf32>) -> tensor<?x4xf32> {
  %s = "onnx.Shape"(%arg0) {start = 0 : si64, end = 1 : si64} : (tensor<?x4xf32>) -> tensor<1xi64>
  %nn = "onnx.Concat"(%s, %s) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  %ones = "onnx.ConstantOfShape"(%nn) {value = dense<1.0> : tensor<1xf32>} : (tensor<2xi64>) -> tensor<?x?xf32>
  %k = onnx.Constant dense<0> : tensor<i64>
  %tri = "onnx.Trilu"(%ones, %k) {upper = 0 : si64} : (tensor<?x?xf32>, tensor<i64>) -> tensor<?x?xf32>
  %0 = "onnx.MatMul"(%tri, %arg0) : (tensor<?x?xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
  return %0 : tensor<?x4xf32>
}

// -----

// CHECK-LABEL: test_cumsum_dynamic_axis_dim_exclusive
func.func @test_cumsum_dynamic_axis_dim_exclusive(%arg0: tensor<?x4xf32>) -> tensor<?x4xf32> {
  %s = "onnx.Shape"(%arg0) {start = 0 : si64, end = 1 : si64} : (tensor<?x4xf32>) -> tensor<1xi64>
  %nn = "onnx.Concat"(%s, %s) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  %ones = "onnx.ConstantOfShape"(%nn) {value = dense<1.0> : tensor<1xf32>} : (tensor<2xi64>) -> tensor<?x?xf32>
  %k = onnx.Constant dense<-1> : tensor<i64>
  %tri = "onnx.Trilu"(%ones, %k) {upper = 0 : si64} : (tensor<?x?xf32>, tensor<i64>) -> tensor<?x?xf32>
  %0 = "onnx.MatMul"(%tri, %arg0) : (tensor<?x?xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
  return %0 : tensor<?x4xf32>
}

