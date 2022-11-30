//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func.func @main_graph(%arg0: tensor<4x4xf32>) -> tensor<4x1xi64> attributes {input_names = ["0"], output_names = ["1"]} {
//CHECK-DAG: %[[DIM:.*]] = torch.constant.int 1
//CHECK-DAG: %[[TRUE:.*]] = torch.constant.bool true
//CHECK: torch.aten.argmax %arg0, %[[DIM]], %[[TRUE]] {layer_name = "ArgMax_0"} : !torch.vtensor<[4,4],f32>, !torch.int, !torch.bool -> !torch.vtensor<[4,1],si64>
    %0 = "onnx.ArgMax"(%arg0) {axis = 1 : si64, keepdims = 1 : si64, onnx_node_name = "ArgMax_0"} : (tensor<4x4xf32>) -> tensor<4x1xi64>
    return %0 : tensor<4x1xi64>
  }
}
