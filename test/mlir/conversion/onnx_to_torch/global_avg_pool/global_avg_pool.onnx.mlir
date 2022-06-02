//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func @main_graph(%arg0: tensor<1x3x4x4xf32>) -> tensor<1x3x1x1xf32> attributes {input_names = ["input"], output_names = ["1"]} {
//CHECK: %[[CONST:.*]] = torch.constant.int 0
// this gets canonicalized to an onnx.ReduceMean
    %0 = "onnx.GlobalAveragePool"(%arg0) : (tensor<1x3x4x4xf32>) -> tensor<1x3x1x1xf32>
//CHECK: torch.aten.mean %arg0, %[[CONST]] : !torch.vtensor<[1,3,4,4],f32>, !torch.int -> !torch.vtensor<[1,3,1,1],f32>
    return %0 : tensor<1x3x1x1xf32>
  }
}
