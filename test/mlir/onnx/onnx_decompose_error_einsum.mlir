// RUN: onnx-mlir-opt --decompose-onnx %s -split-input-file -verify-diagnostics

func.func @test_einsum_matmul(%arg0: tensor<2x3x4xi16>, %arg1: tensor<2x4x5xi16>) -> tensor<2x3x5xi16> {
  // expected-error @+2 {{unsupported element type prevents Einsum decomposition}}
  // expected-error @+1 {{failed to legalize operation 'onnx.Einsum'}}
  %0 = "onnx.Einsum"(%arg0, %arg1) {equation = "...ij,...jk"} : (tensor<2x3x4xi16>, tensor<2x4x5xi16>) -> tensor<2x3x5xi16>
  return %0 : tensor<2x3x5xi16>
}

// -----

func.func @test_einsum_qmark(%arg0: tensor<3x?xf32>) -> tensor<3xf32> {
  // expected-error @+2 {{unknown shapes prevent Einsum decomposition}}
  // expected-error @+1 {{failed to legalize operation 'onnx.Einsum'}}
  %0 = "onnx.Einsum"(%arg0) {equation = "ii->i"} : (tensor<3x?xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// -----

func.func @test_einsum_qmark1(%arg0: tensor<1x?xf32>) -> tensor<?xf32> {
  // expected-error @+2 {{unknown shapes prevent Einsum decomposition}}
  // expected-error @+1 {{failed to legalize operation 'onnx.Einsum'}}
  %0 = "onnx.Einsum"(%arg0) {equation = "ii->i"} : (tensor<1x?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
