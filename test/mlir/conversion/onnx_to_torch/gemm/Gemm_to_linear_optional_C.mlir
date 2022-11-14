// RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s

// because the optional bias is not yet implemented in the upstream TorchToLinAlg pass we will instead
// create a bias with values of 0. Our testing relies on the lowering from torch to host code and this
// will allow us to run the lowering passes.
module attributes {}  {
  func.func @main_graph(%arg0: tensor<1x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<1x4xf32> attributes {input_names = ["a", "b"], output_names = ["y"]} {
    %none = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Gemm"(%arg0, %arg1, %none) {transB = 1 : si64, onnx_node_name = "Gemm_0"} : (tensor<1x5xf32>, tensor<4x5xf32>, none) -> tensor<1x4xf32>
//CHECK: %[[CONSTANT:.*]] = torch.vtensor.literal(dense<0.000000e+00> : tensor<4xf32>) : !torch.vtensor<[4],f32>
//CHECK: %[[RES:.*]] = torch.aten.linear %arg0, %arg1, %[[CONSTANT]] {layer_name = "Gemm_0"} : !torch.vtensor<[1,5],f32>, !torch.vtensor<[4,5],f32>, !torch.vtensor<[4],f32> -> !torch.vtensor<[1,4],f32>
//CHECK: return %[[RES]] :  !torch.vtensor<[1,4],f32>
return %0 : tensor<1x4xf32>
  }
}