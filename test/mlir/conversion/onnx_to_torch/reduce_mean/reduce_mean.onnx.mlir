//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func @main_graph(%arg0: tensor<2x5x9x11xf32>) -> tensor<2x5x1x1xf32> attributes {input_names = ["input"], output_names = ["1"]} {
//CHECK: %[[CONST:.*]] = torch.constant.int 0
    %0 = "onnx.ReduceMean"(%arg0) {axes = [2, 3]} : (tensor<2x5x9x11xf32>) -> tensor<2x5x1x1xf32>
//CHECK: torch.aten.mean %arg0, %[[CONST]] : !torch.vtensor<[2,5,9,11],f32>, !torch.int -> !torch.vtensor<[2,5,1,1],f32>
    return %0 : tensor<2x5x1x1xf32>
  }
}
