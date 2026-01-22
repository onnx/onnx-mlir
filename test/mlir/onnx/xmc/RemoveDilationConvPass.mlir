// RUN: onnx-mlir-opt --split-input-file --remove-dilation-conv %s | FileCheck %s


// CHECK-LABEL: conv_dilation_1
func.func @conv_dilation_1(%arg0: tensor<1x3x16x16xf32>) -> tensor<1x8x12x12xf32> {
    %0 = onnx.Constant dense<1.0> : tensor<8x3x3x3xf32>
    %1 = "onnx.NoValue"() {onnx_node_name = "onnx.NoValue_0", value} : () -> none
    %2 = "onnx.Conv"(%arg0, %0, %1) {auto_pad = "NOTSET", dilations = [2, 2], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "onnx.Conv_1", pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x16x16xf32>, tensor<8x3x3x3xf32>, none) -> tensor<1x8x12x12xf32>
    return %2 : tensor<1x8x12x12xf32>
  }

// CHECK: onnx.Conv
// CHECK-SAME: dilations = [1, 1]
