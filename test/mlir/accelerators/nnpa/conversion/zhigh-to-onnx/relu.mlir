// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --convert-zhigh-to-onnx %s -split-input-file | FileCheck %s

func.func @test_log() -> tensor<10x10xf32> {
  %cst0 = onnx.Constant dense<1.0> : tensor<10x10xf32>
  %0 = "zhigh.Stick"(%cst0) {layout = "2D"} : (tensor<10x10xf32>) -> tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
  %1 = "zhigh.Relu"(%0) : (tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>
  %2 = "zhigh.Unstick"(%1) : (tensor<10x10xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<10x10xf32>
  return %2 : tensor<10x10xf32>

// CHECK-LABEL:  func.func @test_log
// CHECK-SAME:   () -> tensor<10x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<10x10xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Relu"([[VAR_0_]]) : (tensor<10x10xf32>) -> tensor<10x10xf32>
// CHECK:           return [[VAR_1_]] : tensor<10x10xf32>
// CHECK:         }
}
