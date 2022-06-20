//RUN: not onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t 2>&1 | FileCheck -v %s
//
//auto_pad values other than NOTSET are not currently supported, so check we fail. When
//support is added, this can be changed to check the padding is correctly set to zero.
//
//CHECK: failed to legalize operation 'onnx.Conv'

module {
  func @main_graph(%arg0: tensor<20x16x50x40xf32>) -> tensor<20x13x48x38xf32> attributes {input_names = ["0"], output_names = ["2"]} {
    %0 = "onnx.Constant"() {value = dense<0.0> : tensor<13x16x3x3xf32>} : () -> tensor<13x16x3x3xf32>
    %1 = "onnx.NoValue"() {value} : () -> none
    %2 = "onnx.Conv"(%arg0, %0, %1) {dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], auto_pad = "VALID", strides = [1, 1]} : (tensor<20x16x50x40xf32>, tensor<13x16x3x3xf32>, none) -> tensor<20x13x48x38xf32>
    return %2 : tensor<20x13x48x38xf32>
  }
}
