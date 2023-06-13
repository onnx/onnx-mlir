// RUN: onnx-mlir-opt --conv-opt-onnx %s -split-input-file | FileCheck %s

func.func @conv_to_matmul(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<1x1x1x1xf32>) -> tensor<?x1x?x?xf32> {
  %0 = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.Conv"(%arg0, %arg1, %0) {auto_pad = "VALID", dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<?x?x?x?xf32>, tensor<1x1x1x1xf32>, none) -> tensor<?x1x?x?xf32>
  onnx.Return %1 : tensor<?x1x?x?xf32>
}

// CHECK-LABEL: @conv_to_matmul
//       CHECK: "onnx.MatMul"
// CHECK-SAME:   -> tensor<?x1x?xf32>