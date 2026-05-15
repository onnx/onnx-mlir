// RUN: onnx-mlir-opt --decompose-onnx %s -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @main_graph
func.func @main_graph(%arg0: tensor<1x3x32x32xf32>, %arg1: tensor<64x3x1x1xf32>) -> tensor<?x?x?x?xf32> {
  %0 = "onnx.NoValue"() <{value}> : () -> none
  // CHECK-NOT: onnx.Conv
  // CHECK: onnx.Reshape
  // CHECK: onnx.MatMul
  // CHECK: onnx.Reshape
  %1 = "onnx.Conv"(%arg0, %arg1, %0) <{auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]}> : (tensor<1x3x32x32xf32>, tensor<64x3x1x1xf32>, none) -> tensor<?x?x?x?xf32>
  return %1 : tensor<?x?x?x?xf32>
}