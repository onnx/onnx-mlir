// RUN: onnx-mlir --march=z16 --maccel=NNPA --printIR --EmitMLIR %s 2>&1 | FileCheck %s

module {

// Conv with dynamic shape - runs on CPU (dynamic shapes not supported)
func.func @test_conv_cpu(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<64x3x3x3xf32>) -> tensor<?x?x?x?xf32> {
  %none = "onnx.NoValue"() <{value}> : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %none) <{auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]}> : (tensor<?x?x?x?xf32>, tensor<64x3x3x3xf32>, none) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
}
// CHECK: [Warning] There are 1 onnx.Conv operations that run on CPU.

// Gemm with dimension exceeding limit (32769 > 32768) - runs on CPU
func.func @test_gemm_cpu(%arg0: tensor<32769x20xf32>, %arg1: tensor<20x30xf32>, %arg2: tensor<30xf32>) -> tensor<32769x30xf32> {
  %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 1.0 : f32, transA = 0 : si64, transB = 0 : si64} : (tensor<32769x20xf32>, tensor<20x30xf32>, tensor<30xf32>) -> tensor<32769x30xf32>
  return %0 : tensor<32769x30xf32>
}
// CHECK: [Warning] There are 1 onnx.Gemm operations that run on CPU.

// GRU with dimension exceeding limit (32769 > 32768) - runs on CPU
func.func @test_gru_cpu(%arg0: tensor<7x32769x204xf32>, %arg1: tensor<1x600x204xf32>, %arg2: tensor<1x600x200xf32>, %arg3: tensor<1x1200xf32>) -> tensor<7x1x32769x200xf32> {
  %none = "onnx.NoValue"() <{value}> : () -> none
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %none, %none) {activations = ["Sigmoid", "Tanh"], direction = "forward", hidden_size = 200 : si64, linear_before_reset = 1 : si64} : (tensor<7x32769x204xf32>, tensor<1x600x204xf32>, tensor<1x600x200xf32>, tensor<1x1200xf32>, none, none) -> (tensor<7x1x32769x200xf32>, tensor<1x32769x200xf32>)
  return %Y : tensor<7x1x32769x200xf32>
}
// CHECK: [Warning] There are 1 onnx.GRU operations that run on CPU.

// LSTM with dimension exceeding limit (32769 > 32768) - runs on CPU
func.func @test_lstm_cpu(%arg0: tensor<7x32769x204xf32>, %arg1: tensor<1x800x204xf32>, %arg2: tensor<1x800x200xf32>, %arg3: tensor<1x1600xf32>) -> tensor<7x1x32769x200xf32> {
  %none = "onnx.NoValue"() <{value}> : () -> none
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %none, %none, %none, %none) {activations = ["Sigmoid", "Tanh", "Tanh"], direction = "forward", hidden_size = 200 : si64} : (tensor<7x32769x204xf32>, tensor<1x800x204xf32>, tensor<1x800x200xf32>, tensor<1x1600xf32>, none, none, none, none) -> (tensor<7x1x32769x200xf32>, tensor<1x32769x200xf32>, tensor<1x32769x200xf32>)
  return %Y : tensor<7x1x32769x200xf32>
}
// CHECK: [Warning] There are 1 onnx.LSTM operations that run on CPU.

// MatMul with 4D broadcasting - runs on CPU (not supported by NNPA)
func.func @test_matmul_cpu(%arg0: tensor<2x1x10x20xf32>, %arg1: tensor<2x3x20x30xf32>) -> tensor<2x3x10x30xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<2x1x10x20xf32>, tensor<2x3x20x30xf32>) -> tensor<2x3x10x30xf32>
  return %0 : tensor<2x3x10x30xf32>
}
// CHECK: [Warning] There are 1 onnx.MatMul operations that run on CPU.

// MatMulInteger with dimension exceeding limit (32769 > 32768) - runs on CPU
func.func @test_matmulinteger_cpu(%arg0: tensor<32769x768xui8>, %arg1: tensor<768x768xi8>, %arg2: tensor<ui8>, %arg3: tensor<i8>) -> tensor<32769x768xi32> {
  %0 = "onnx.MatMulInteger"(%arg0, %arg1, %arg2, %arg3) : (tensor<32769x768xui8>, tensor<768x768xi8>, tensor<ui8>, tensor<i8>) -> tensor<32769x768xi32>
  return %0 : tensor<32769x768xi32>
}
// CHECK: [Warning] There are 1 onnx.MatMulInteger operations that run on CPU.

// QLinearMatMul with dimension exceeding limit (32769 > 32768) - runs on CPU
func.func @test_qlinearmatmul_cpu(%arg0: tensor<32769x4xi8>, %arg1: tensor<f32>, %arg2: tensor<i8>, %arg3: tensor<4x3xi8>, %arg4: tensor<f32>, %arg5: tensor<i8>, %arg6: tensor<f32>, %arg7: tensor<i8>) -> tensor<32769x3xi8> {
  %0 = "onnx.QLinearMatMul"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7) : (tensor<32769x4xi8>, tensor<f32>, tensor<i8>, tensor<4x3xi8>, tensor<f32>, tensor<i8>, tensor<f32>, tensor<i8>) -> tensor<32769x3xi8>
  return %0 : tensor<32769x3xi8>
}
// CHECK: [Warning] There are 1 onnx.QLinearMatMul operations that run on CPU.

// Softmax with dimension exceeding limit (32769 > 32768) - runs on CPU
func.func @test_softmax_cpu(%arg0: tensor<32769x10xf32>) -> tensor<32769x10xf32> {
  %0 = "onnx.Softmax"(%arg0) {axis = 1 : si64} : (tensor<32769x10xf32>) -> tensor<32769x10xf32>
  return %0 : tensor<32769x10xf32>
}
// CHECK: [Warning] There are 1 onnx.Softmax operations that run on CPU.

}