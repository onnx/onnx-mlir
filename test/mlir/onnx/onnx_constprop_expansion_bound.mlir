// RUN: onnx-mlir-opt --constprop-onnx --onnx-const-prop-expansion-bound=2 %s -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// Constant propagate ONNXAddOp only if expansion bound satisfied
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @test_add_propagates() -> tensor<2x5xf32>
func.func @test_add_propagates() -> tensor<2x5xf32> {
  %0 = "onnx.Constant"() {value = dense<1.0> : tensor<2x1xf32>} : () -> tensor<2x1xf32>
  %1 = "onnx.Constant"() {value = dense<2.0> : tensor<1x5xf32>} : () -> tensor<1x5xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<2x1xf32> , tensor<1x5xf32>) -> tensor<2x5xf32>
  onnx.Return %2 : tensor<2x5xf32>
  // CHECK: onnx.Constant {{.*}} : tensor<2x5xf32>
}

// CHECK-LABEL: @test_add_doesnt_propagate() -> tensor<5x5xf32>
func.func @test_add_doesnt_propagate() -> tensor<5x5xf32> {
  %0 = "onnx.Constant"() {value = dense<1.0> : tensor<5x1xf32>} : () -> tensor<5x1xf32>
  %1 = "onnx.Constant"() {value = dense<2.0> : tensor<1x5xf32>} : () -> tensor<1x5xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<5x1xf32> , tensor<1x5xf32>) -> tensor<5x5xf32>
  onnx.Return %2 : tensor<5x5xf32>
  // CHECK: "onnx.Add"(%0, %1) : (tensor<5x1xf32>, tensor<1x5xf32>) -> tensor<5x5xf32>
}
