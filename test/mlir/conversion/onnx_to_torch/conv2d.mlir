//RUN: onnx-mlir --EmitONNXIR %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s

module {
  func @main_graph(%arg0: tensor<20x16x50x40xf32>) -> tensor<20x13x48x38xf32> attributes {input_names = ["0"], output_names = ["2"]} {
    %0 = "onnx.Constant"() {value = dense<0.0> : tensor<13x16x3x3xf32>} : () -> tensor<13x16x3x3xf32>
    %1 = "onnx.NoValue"() {value} : () -> none
    //CHECK: [[STRIDE:%.]] = torch.prim.ListConstruct %int1{{_*[0-9]*}}, %int1{{_*[0-9]*}} :
    //CHECK: [[DILATION:%.]] = torch.prim.ListConstruct %int1{{_*[0-9]*}}, %int1{{_*[0-9]*}} :
    //CHECK: [[PAD:%.]] = torch.prim.ListConstruct %int0{{_*[0-9]*}}, %int0{{_*[0-9]*}} :
    //CHECK: torch.aten.conv2d %arg0{{_*[0-9]*}}, %{{[^,]*}}, %none, [[STRIDE]], [[PAD]], [[DILATION]], %int1{{_*[0-9]*}} : !torch.vtensor<[20,16,50,40],f32>, !torch.vtensor<[13,16,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[20,13,48,38],f32>
    %2 = "onnx.Conv"(%arg0, %0, %1) {dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<20x16x50x40xf32>, tensor<13x16x3x3xf32>, none) -> tensor<20x13x48x38xf32>
    return %2 : tensor<20x13x48x38xf32>
  }
}
