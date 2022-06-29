//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func @main_graph(%arg0: tensor<3x2x4x6xf32>) -> tensor<3x2x4x1xi64> attributes {input_names = ["0"], output_names = ["1"]} {
//CHECK-DAG: %[[DIM:.*]] = torch.constant.int 3
//CHECK-DAG: %[[TRUE:.*]] = torch.constant.bool true
//CHECK: torch.aten.argmax %arg0, %[[DIM]], %[[TRUE]] : !torch.vtensor<[3,2,4,6],f32>, !torch.int, !torch.bool -> !torch.vtensor<[3,2,4,1],si64>
    %0 = "onnx.ArgMax"(%arg0) {axis = 3 : si64, keepdims = 1 : si64, onnx_node_name = "ArgMax_0"} : (tensor<3x2x4x6xf32>) -> tensor<3x2x4x1xi64>
    return %0 : tensor<3x2x4x1xi64>
  }
}
