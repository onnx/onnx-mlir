//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func @main_graph(%arg0: tensor<?x2x1x3x1x1xf32>) -> tensor<?x2x1x3x1xf32> attributes {input_names = ["input"], output_names = ["output"]} {
//CHECK: %[[DIM:.*]] = torch.constant.int 4
    %0 = "onnx.SqueezeV11"(%arg0) {axes = [4], onnx_node_name = "Squeeze_0"} : (tensor<?x2x1x3x1x1xf32>) -> tensor<?x2x1x3x1xf32>
//CHECK: torch.aten.squeeze.dim %arg0, %[[DIM]] : !torch.vtensor<[?,2,1,3,1,1],f32>, !torch.int -> !torch.vtensor<[?,2,1,3,1],f32>
    return %0 : tensor<?x2x1x3x1xf32>
  }
}
