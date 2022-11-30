// RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func.func @main_graph(%arg0: tensor<1x17x18x18xf32>) -> tensor<1x17x36x18xf32> attributes {input_names = ["input"], output_names = ["output"]} {
// CHECK-DAG: [[DIM:%.*]] = torch.constant.int 2
// CHECK-DAG: [[LIST:%.*]] = torch.prim.ListConstruct %arg0, %arg0 : (!torch.vtensor<[1,17,18,18],f32>, !torch.vtensor<[1,17,18,18],f32>) -> !torch.list<vtensor>
// CHECK: torch.aten.cat [[LIST]], [[DIM]] {layer_name = "Concat_1"} : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[1,17,36,18],f32>
%1 = "onnx.Concat"(%arg0, %arg0) {axis = 2 : si64, onnx_node_name = "Concat_1"} : (tensor<1x17x18x18xf32>, tensor<1x17x18x18xf32>) -> tensor<1x17x36x18xf32>
    return %1 : tensor<1x17x36x18xf32>
  }
}
