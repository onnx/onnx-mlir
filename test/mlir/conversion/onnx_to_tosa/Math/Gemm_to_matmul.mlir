//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func.func @test_gemm_to_matmul(%arg0: tensor<3x5xf32>, %arg1: tensor<5x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32> attributes {input_names = ["a", "b", "c"], output_names = ["y"]} {
    %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32} : (tensor<3x5xf32>, tensor<5x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
return %0 : tensor<3x4xf32>
  }
  
  func.func @test_transA(%arg0: tensor<6x3xf32>, %arg1: tensor<6x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32> attributes {input_names = ["a", "b", "c"], output_names = ["y"]} {
    %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {transA = 1 : si64} : (tensor<6x3xf32>, tensor<6x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
    return %0 : tensor<3x4xf32>
  }
  
  func.func @test_transB(%arg0: tensor<3x6xf32>, %arg1: tensor<4x6xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32> attributes {input_names = ["a", "b", "c"], output_names = ["y"]} {
    %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.184 : f32, transB = 1 : si64} : (tensor<3x6xf32>, tensor<4x6xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
    return %0 : tensor<3x4xf32>
  }

  func.func @test_alpha(%arg0: tensor<3x6xf32>, %arg1: tensor<4x6xf32>, %arg2: tensor<3x6xf32>) -> tensor<3x6xf32> attributes {input_names = ["a", "b", "c"], output_names = ["y"]} {
    %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.618 : f32} : (tensor<3x6xf32>, tensor<4x6xf32>, tensor<3x6xf32>) -> tensor<3x6xf32>
    return %0 : tensor<3x6xf32>
  }
  
  func.func @test_beta(%arg0: tensor<3x6xf32>, %arg1: tensor<4x6xf32>, %arg2: tensor<3x6xf32>) -> tensor<3x6xf32> attributes {input_names = ["a", "b", "c"], output_names = ["y"]} {
    %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {beta = 1.349 : f32} : (tensor<3x6xf32>, tensor<4x6xf32>, tensor<3x6xf32>) -> tensor<3x6xf32>
    return %0 : tensor<3x6xf32>
  }

  func.func @test_mixed(%arg0: tensor<3x6xf32>, %arg1: tensor<4x6xf32>, %arg2: tensor<6x4xf32>) -> tensor<6x4xf32> attributes {input_names = ["a", "b", "c"], output_names = ["y"]} {
    %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.402 : f32, beta = 1.998 : f32, transA = 1 : si64, transB = 1 : si64} : (tensor<3x6xf32>, tensor<4x6xf32>, tensor<6x4xf32>) -> tensor<6x4xf32>
    return %0 : tensor<6x4xf32>
  }

}
