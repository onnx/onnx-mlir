//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func @main_graph(%arg0: tensor<1x3x8x8xf32>) -> tensor<1x8x8x8xf32> attributes {input_names = ["input.1"], output_names = ["9"]} {
    %0 = "onnx.Constant"() {value = dense<[[[[-0.397202909]], [[-0.36456573]], [[0.554807603]]], [[[0.128473148]], [[-0.0792445242]], [[0.0802431181]]], [[[0.43963322]], [[-0.535440087]], [[-0.211081728]]], [[[0.205662608]], [[-0.0686611608]], [[0.00833704136]]], [[[0.115851991]], [[0.450539321]], [[0.0498450212]]], [[[0.0374790728]], [[0.0478444695]], [[0.0221564472]]], [[[0.135107309]], [[0.0581301674]], [[0.0429676324]]], [[[-0.0938993394]], [[0.55964911]], [[-0.168949708]]]]> : tensor<8x3x1x1xf32>} : () -> tensor<8x3x1x1xf32>
    %1 = "onnx.Constant"() {value = dense<[-0.212501466, 0.139968663, -0.53322506, -0.286446363, 0.449779421, -0.505617738, 0.116871089, -0.292490959]> : tensor<8xf32>} : () -> tensor<8xf32>
//CHECK: [[STRIDE:%.]] = torch.prim.ListConstruct %int1{{_*[0-9]*}}, %int1{{_*[0-9]*}} :
//CHECK: [[DILATION:%.]] = torch.prim.ListConstruct %int1, %int1{{_*[0-9]*}} :
//CHECK: [[PAD:%.]] = torch.prim.ListConstruct %int0, %int0{{_*[0-9]*}} :
//CHECK: torch.aten.conv2d %arg0, %{{[^,]*}}, %{{[^,]*}}, [[STRIDE]], [[PAD]], [[DILATION]], %int1{{_*[0-9]*}} : !torch.vtensor<[1,3,8,8],f32>, !torch.vtensor<[8,3,1,1],f32>, !torch.vtensor<[8],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,8,8,8],f32>
    %2 = "onnx.Conv"(%arg0, %0, %1) {dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], onnx_node_name = "Conv_0", pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x8x8xf32>, tensor<8x3x1x1xf32>, tensor<8xf32>) -> tensor<1x8x8x8xf32>
    return %2 : tensor<1x8x8x8xf32>
  }
}
