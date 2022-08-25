//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func @main_graph(%arg0: tensor<2x3x5x7xf32>) -> tensor<6x35xf32> attributes {input_names = ["0"], output_names = ["1"]} {
//CHECK-DAG: %[[DIM:.*]] = torch.constant.int 0
//CHECK-DAG: %[[DIM1:.*]] = torch.constant.int 1    
//CHECK-DAG: [[FLAT1:%.]] = torch.aten.flatten.using_ints %arg0, %[[DIM]], %[[DIM1]] : !torch.vtensor<[2,3,5,7],f32>, !torch.int, !torch.int -> !torch.vtensor<[6,5,7],f32>
//CHECK-DAG: [[FLAT2:%.]] = torch.aten.flatten.using_ints [[FLAT1]], %[[DIM1]], %{{[^,]*}} : !torch.vtensor<[6,5,7],f32>, !torch.int, !torch.int -> !torch.vtensor<[6,35],f32> 
    %0 = "onnx.Flatten"(%arg0) {axis = 2 : si64, onnx_node_name = "Flatten_0"} : (tensor<2x3x5x7xf32>) -> tensor<6x35xf32>
    return %0 : tensor<6x35xf32>
  }
}
