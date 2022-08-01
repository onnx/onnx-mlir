//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
// XFAIL: *
// We expect this to fail as the optional bias is not yet implemented in the upstream TorchToLinAlg pass
// We will always assume C or the bias is given. Our testing relies on the lowering from torch to host code.
// Note, below is what a conversion would look like for linear with an optional bias tensor.
module attributes {}  {
  func @main_graph(%arg0: tensor<3x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<3x4xf32> attributes {input_names = ["a", "b"], output_names = ["y"]} {
    %none = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Gemm"(%arg0, %arg1, %none) {transB = 1 : si64} : (tensor<3x5xf32>, tensor<4x5xf32>, none) -> tensor<3x4xf32>
//CHECK: %[[NONE:.*]] torch.constant.none
//CHECK: %[[OPTIONAL:.*]] = torch.derefine %none : !torch.none to !torch.optional<tensor>
//CHECK: %[[RES1:.]] = torch.aten.linear %arg0, %arg1, %[[OPTIONAL]] : !torch.vtensor<[3,5],f32>, !torch.vtensor<[4,5],f32>, !torch.optional<tensor> -> !torch.vtensor<[3,4],f32>>       
//CHECK: return %[[RES1]] : tensor<3x4xf32>
return %0 : tensor<3x4xf32>
  }
}