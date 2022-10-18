// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_reshape(%arg0: tensor<32x512x1x1xf32>) -> tensor<32x512xf32> {
  %0 = "onnx.Constant"() {value = dense<[32,512]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Reshape"(%arg0, %0) : (tensor<32x512x1x1xf32>, tensor<2xi64>) -> tensor<32x512xf32>
  return %1 : tensor<32x512xf32>
  //CHECK:  %[[VAR0:.*]] = "tosa.reshape"(%arg0) {new_shape = [32, 512]} : (tensor<32x512x1x1xf32>) -> tensor<32x512xf32>
}
