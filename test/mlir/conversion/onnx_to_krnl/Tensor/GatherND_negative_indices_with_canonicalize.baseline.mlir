// Baseline for GatherND_negative_indices_with_canonicalize.mlir: each function
// normalizes the indices with ONNX ops (index < 0 ? index + dim : index) before
// a GatherND, so the GatherND itself only sees non-negative indices.

func.func @test_gather_nd_negative_indices(%arg0 : tensor<3x4xf32>, %arg1 : tensor<6x2xi64>) -> tensor<6xf32> {
  %zero = onnx.Constant dense<0> : tensor<i64>
  %dims = onnx.Constant dense<[3, 4]> : tensor<2xi64>
  %neg = "onnx.Less"(%arg1, %zero) : (tensor<6x2xi64>, tensor<i64>) -> tensor<6x2xi1>
  %wrapped = "onnx.Add"(%arg1, %dims) : (tensor<6x2xi64>, tensor<2xi64>) -> tensor<6x2xi64>
  %idx = "onnx.Where"(%neg, %wrapped, %arg1) : (tensor<6x2xi1>, tensor<6x2xi64>, tensor<6x2xi64>) -> tensor<6x2xi64>
  %0 = "onnx.GatherND"(%arg0, %idx) {batch_dims = 0 : si64} : (tensor<3x4xf32>, tensor<6x2xi64>) -> tensor<6xf32>
  "func.return"(%0) : (tensor<6xf32>) -> ()
// CHECK-LABEL: func.func @test_gather_nd_negative_indices
}

// -----

func.func @test_gather_nd_negative_indices_batch_dims(%arg0 : tensor<2x3xf32>, %arg1 : tensor<2x6x1xi64>) -> tensor<2x6xf32> {
  %zero = onnx.Constant dense<0> : tensor<i64>
  %dims = onnx.Constant dense<[3]> : tensor<1xi64>
  %neg = "onnx.Less"(%arg1, %zero) : (tensor<2x6x1xi64>, tensor<i64>) -> tensor<2x6x1xi1>
  %wrapped = "onnx.Add"(%arg1, %dims) : (tensor<2x6x1xi64>, tensor<1xi64>) -> tensor<2x6x1xi64>
  %idx = "onnx.Where"(%neg, %wrapped, %arg1) : (tensor<2x6x1xi1>, tensor<2x6x1xi64>, tensor<2x6x1xi64>) -> tensor<2x6x1xi64>
  %0 = "onnx.GatherND"(%arg0, %idx) {batch_dims = 1 : si64} : (tensor<2x3xf32>, tensor<2x6x1xi64>) -> tensor<2x6xf32>
  "func.return"(%0) : (tensor<2x6xf32>) -> ()
// CHECK-LABEL: func.func @test_gather_nd_negative_indices_batch_dims
}

// -----

func.func @test_gather_nd_negative_indices_dynamic(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x2xi64>) -> tensor<?xf32> {
  %zero = onnx.Constant dense<0> : tensor<i64>
  %dims = "onnx.Shape"(%arg0) {start = 0 : si64} : (tensor<?x?xf32>) -> tensor<2xi64>
  %neg = "onnx.Less"(%arg1, %zero) : (tensor<?x2xi64>, tensor<i64>) -> tensor<?x2xi1>
  %wrapped = "onnx.Add"(%arg1, %dims) : (tensor<?x2xi64>, tensor<2xi64>) -> tensor<?x2xi64>
  %idx = "onnx.Where"(%neg, %wrapped, %arg1) : (tensor<?x2xi1>, tensor<?x2xi64>, tensor<?x2xi64>) -> tensor<?x2xi64>
  %0 = "onnx.GatherND"(%arg0, %idx) {batch_dims = 0 : si64} : (tensor<?x?xf32>, tensor<?x2xi64>) -> tensor<?xf32>
  "func.return"(%0) : (tensor<?xf32>) -> ()
// CHECK-LABEL: func.func @test_gather_nd_negative_indices_dynamic
}

// -----

func.func @test_gather_nd_negative_indices_mixed(%arg0 : tensor<3x?xf32>, %arg1 : tensor<6x2xi64>) -> tensor<6xf32> {
  %zero = onnx.Constant dense<0> : tensor<i64>
  %dims = "onnx.Shape"(%arg0) {start = 0 : si64} : (tensor<3x?xf32>) -> tensor<2xi64>
  %neg = "onnx.Less"(%arg1, %zero) : (tensor<6x2xi64>, tensor<i64>) -> tensor<6x2xi1>
  %wrapped = "onnx.Add"(%arg1, %dims) : (tensor<6x2xi64>, tensor<2xi64>) -> tensor<6x2xi64>
  %idx = "onnx.Where"(%neg, %wrapped, %arg1) : (tensor<6x2xi1>, tensor<6x2xi64>, tensor<6x2xi64>) -> tensor<6x2xi64>
  %0 = "onnx.GatherND"(%arg0, %idx) {batch_dims = 0 : si64} : (tensor<3x?xf32>, tensor<6x2xi64>) -> tensor<6xf32>
  "func.return"(%0) : (tensor<6xf32>) -> ()
// CHECK-LABEL: func.func @test_gather_nd_negative_indices_mixed
}

// -----

// The same gather with the equivalent negative constant indices.
func.func @test_gather_nd_nonnegative_constant_indices(%arg0 : tensor<3x4xf32>) -> tensor<2xf32> {
  %0 = onnx.Constant dense<[[-1, -1], [-3, -3]]> : tensor<2x2xi64>
  %1 = "onnx.GatherND"(%arg0, %0) {batch_dims = 0 : si64} : (tensor<3x4xf32>, tensor<2x2xi64>) -> tensor<2xf32>
  "func.return"(%1) : (tensor<2xf32>) -> ()
// CHECK-LABEL: func.func @test_gather_nd_nonnegative_constant_indices
}
