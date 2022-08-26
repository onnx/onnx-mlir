// RUN: onnx-mlir-opt --elide-constants %s -split-input-file | FileCheck %s

// CHECK-LABEL: func @test_elide_constant(%arg0: tensor<1xf32>) -> tensor<1x10xf32>
func.func @test_elide_constant(%arg0: tensor<1xf32>) -> tensor<1x10xf32> {
  %0 = "onnx.Constant"() {value = dense<[[-0.0448560268, 0.00779166119, 0.0681008175, 0.0299937408, -0.126409635, 0.14021875, -0.0552849025, -0.0493838154, 0.0843220502, -0.0545404144]]> : tensor<1x10xf32>} : () -> tensor<1x10xf32>
  "func.return"(%0) : (tensor<1x10xf32>) -> ()

  // CHECK: %0 = "onnx.Constant"() : () -> tensor<1x10xf32>
  // CHECK: return %0 : tensor<1x10xf32>
}
