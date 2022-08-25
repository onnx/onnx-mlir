// RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func @main_graph(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x3x1x1xf32> attributes {input_names = ["input"], output_names = ["1"]} {
// `onnx.GlobalAveragePool` gets canonicalized to `onnx.ReduceMean`
// CHECK: %[[AXIS:.*]] = torch.prim.ListConstruct %int2, %int3 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: %[[MEAN:.*]] = torch.aten.mean.dim %arg0, %[[AXIS]], %true, %none : !torch.vtensor<[1,3,4,4],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,3,1,1],f32>
    %0 = "onnx.GlobalAveragePool"(%arg0) : (tensor<1x3x4x4xf32>) -> tensor<1x3x1x1xf32>
    return %0 : tensor<1x3x1x1xf32>
  }
}
