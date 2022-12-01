// RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func.func @main_graph(%arg0: tensor<2x5x9x11xf32>) -> tensor<2x5x1x1xf32> attributes {input_names = ["input"], output_names = ["1"]} {
// CHECK: %[[AXIS:.*]] = torch.prim.ListConstruct %int2, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[MEAN:.*]] = torch.aten.mean.dim %arg0, %[[AXIS]], %true, %none {layer_name = "Reduce_0"} : !torch.vtensor<[2,5,9,11],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[2,5,1,1],f32>
    %0 = "onnx.ReduceMean"(%arg0) {axes = [2, 3], onnx_node_name = "Reduce_0"} : (tensor<2x5x9x11xf32>) -> tensor<2x5x1x1xf32>
    return %0 : tensor<2x5x1x1xf32>
  }
}
