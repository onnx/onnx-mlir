//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func.func @main_graph(%arg0: tensor<4x5xf32>) -> tensor<5xi64> attributes {input_names = ["input"], output_names = ["output"]} {
//CHECK-DAG: %[[DIM:.*]] = torch.constant.int 0
//CHECK-DAG: %[[FALSE:.*]] = torch.constant.bool false
//CHECK: torch.aten.argmax %arg0, %[[DIM]], %[[FALSE]] {layer_name = "ArgMax_0"} : !torch.vtensor<[4,5],f32>, !torch.int, !torch.bool -> !torch.vtensor<[5],si64>
    %0 = "onnx.ArgMax"(%arg0) {axis = 0 : si64, keepdims = 0 : si64, onnx_node_name = "ArgMax_0", select_last_index = 0 : si64} : (tensor<4x5xf32>) -> tensor<5xi64>
    return %0 : tensor<5xi64>
  }
}
