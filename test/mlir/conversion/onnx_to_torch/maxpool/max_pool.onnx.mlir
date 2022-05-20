//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s

module {
  func @main_graph(%arg0: tensor<20x16x50x40xf32>) -> tensor<20x16x48x38xf32> attributes {input_names = ["0"], output_names = ["2"]} {
    //CHECK: [[STRIDE:%.]] = torch.prim.ListConstruct %int1{{_*[0-9]*}}, %int1{{_*[0-9]*}} :
    //CHECK: [[DILATION:%.]] = torch.prim.ListConstruct %int1{{_*[0-9]*}}, %int1{{_*[0-9]*}} :
    //CHECK: [[PAD:%.]] = torch.prim.ListConstruct %int0{{_*[0-9]*}}, %int0{{_*[0-9]*}} :
    //CHECK: torch.aten.max_pool2d %arg0, %{{[^,]*}}, [[STRIDE]], [[PAD]], [[DILATION]], %false : !torch.vtensor<[20,16,50,40],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[20,16,48,38],f32>
    %2 = "onnx.MaxPoolSingleOut"(%arg0) {dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<20x16x50x40xf32>) -> tensor<20x16x48x38xf32>
    return %2 : tensor<20x16x48x38xf32>
  }
}
