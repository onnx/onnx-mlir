// RUN: onnx-mlir-opt --shape-inference --canonicalize %s -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
/// Test the folder called in canonicalize pass.
/// Shape-inference is invoked to ensure the type is correct and
/// constant may be introduced.
//===----------------------------------------------------------------------===//

// -----

func.func @test_squeeze() -> tensor<*xf32> {
  %0 = onnx.Constant dense<[[[4.0]], [[16.0]]]> : tensor<2x1x1xf32>
  %1 = onnx.Constant dense<[1, 2]> : tensor<2xi64>
  %2 = "onnx.Squeeze"(%0, %1) : (tensor<2x1x1xf32>, tensor<2xi64>) -> tensor<*xf32>
  "onnx.Return"(%2) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  func.func @test_squeeze
// CHECK-SAME:   () -> tensor<2xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<[4.000000e+00, 1.600000e+01]> : tensor<2xf32>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<2xf32>
}

// -----

func.func @test_squeezev11() -> tensor<*xf32> {
  %0 = onnx.Constant dense<[[[4.0]], [[16.0]]]> : tensor<2x1x1xf32>
  %1 = "onnx.SqueezeV11"(%0) {axes = [1, 2]} : (tensor<2x1x1xf32>) -> tensor<*xf32>
  "onnx.Return"(%1) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  func.func @test_squeezev11
// CHECK-SAME:   () -> tensor<2xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<[4.000000e+00, 1.600000e+01]> : tensor<2xf32>
// CHECK:           onnx.Return [[VAR_0_]] : tensor<2xf32>
}
