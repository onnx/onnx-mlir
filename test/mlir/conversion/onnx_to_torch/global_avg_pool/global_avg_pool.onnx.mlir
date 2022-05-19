//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func @main_graph(%arg0: tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32> attributes {input_names = ["x"], output_names = ["y"]} {
//CHECK: %[[DIM:.*]] = torch.constant.int 0
//CHECK: torch.aten.mean %arg0, %[[DIM]] : !torch.vtensor<[1,3,5,5],f32>, !torch.int -> !torch.vtensor<[1,3,1,1],f32>
    %0 = "onnx.ReduceMean"(%arg0) {axes = [2, 3]} : (tensor<1x3x5x5xf32>) -> tensor<1x3x1x1xf32>
    return %0 : tensor<1x3x1x1xf32>
  }
}
