//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func @main_graph(%arg0: tensor<1x5x10x10xf32>) -> tensor<1x5x4x9xf32> attributes {input_names = ["0"], output_names = ["2"]} {
//CHECK: [[KERNEL:%.]] = torch.prim.ListConstruct %int2{{_*[0-9]*}}, %int1{{_*[0-9]*}} :
//CHECK: [[STRIDE:%.]] = torch.prim.ListConstruct %int1, %int1 :
//CHECK: [[DILATION:%.]] = torch.prim.ListConstruct %int0, %int0{{_*[0-9]*}} :
//CHECK: [[PAD:%.]] = torch.prim.ListConstruct %int3, %int2{{_*[0-9]*}} :
//CHECK: torch.aten.max_pool2d %arg0, [[PAD]], [[KERNEL]], [[DILATION]], [[STRIDE]], %false : !torch.vtensor<[1,5,10,10],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,5,4,9],f32>
    %0 = "onnx.MaxPoolSingleOut"(%arg0) {kernel_shape = [3, 2], onnx_node_name = "MaxPool_0", pads = [0, 0, 0, 0], strides = [2, 1]} : (tensor<1x5x10x10xf32>) -> tensor<1x5x4x9xf32>
    return %0 : tensor<1x5x4x9xf32>
  }
}
