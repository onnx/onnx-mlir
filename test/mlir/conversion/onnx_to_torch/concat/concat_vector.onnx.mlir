//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func.func @main_graph(%arg0: tensor<4xf32>, %arg1: tensor<2xf32>, %arg2: tensor<3xf32>, %arg3: tensor<5xf32>, %arg4: tensor<1xf32>) -> tensor<15xf32> attributes {input_names = ["0", "1", "2", "3", "4"], output_names = ["5"]} {
//CHECK-DAG: %[[DIM:.*]] = torch.constant.int 0
//CHECK-DAG: torch.prim.ListConstruct %arg0, %arg1, %arg2, %arg3, %arg4 : (!torch.vtensor<[4],f32>, !torch.vtensor<[2],f32>, !torch.vtensor<[3],f32>, !torch.vtensor<[5],f32>, !torch.vtensor<[1],f32>) -> !torch.list<vtensor>
    %0 = "onnx.Concat"(%arg0, %arg1, %arg2, %arg3, %arg4) {axis = 0 : si64, onnx_node_name = "Concat_0"} : (tensor<4xf32>, tensor<2xf32>, tensor<3xf32>, tensor<5xf32>, tensor<1xf32>) -> tensor<15xf32>
//CHECK: torch.aten.cat %{{[^,]*}}, %[[DIM]] {layer_name = "Concat_0"} : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[15],f32>
    return %0 : tensor<15xf32>
  }
}
