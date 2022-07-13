//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func @main_graph(%arg0: tensor<1x5x10x10xf32>) -> tensor<1x9x3x7xf32> attributes {input_names = ["input.1"], output_names = ["9"]} {
    %0 = "onnx.Constant"() {value = dense<"0xDEADBEEF"> : tensor<9x5x3x4xf32>} : () -> tensor<9x5x3x4xf32>
    %1 = "onnx.Constant"() {value = dense<[-0.0848599597, -0.11561133, 0.0635902881, 0.0752899796, 0.0742146298, -0.0947614163, -4.916150e-03, -0.0397511758, -0.0190015137]> : tensor<9xf32>} : () -> tensor<9xf32>
//CHECK-DAG: [[STRIDE:%.]] = torch.prim.ListConstruct %int3{{_*[0-9]*}}, %int1{{_*[0-9]*}} :
//CHECK-DAG: [[DILATION:%.]] = torch.prim.ListConstruct %int1, %int1{{_*[0-9]*}} :
//CHECK-DAG: [[PAD:%.]] = torch.prim.ListConstruct %int0, %int0{{_*[0-9]*}} :
//CHECK: torch.aten.conv2d %arg0, %{{[^,]*}}, %{{[^,]*}}, [[STRIDE]], [[PAD]], [[DILATION]], %int1{{_*[0-9]*}} : !torch.vtensor<[1,5,10,10],f32>, !torch.vtensor<[9,5,3,4],f32>, !torch.vtensor<[9],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,9,3,7],f32>
    %2 = "onnx.Conv"(%arg0, %0, %1) {dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 4], onnx_node_name = "Conv_0", pads = [0, 0, 0, 0], strides = [3, 1]} : (tensor<1x5x10x10xf32>, tensor<9x5x3x4xf32>, tensor<9xf32>) -> tensor<1x9x3x7xf32>
    return %2 : tensor<1x9x3x7xf32>
  }
}
