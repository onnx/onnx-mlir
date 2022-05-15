//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func @main_graph(%arg0: tensor<2x3xf32>) -> tensor<4x3xf32> attributes {input_names = ["0"], output_names = ["1"]} {
//CHECK: %int[[DIM:.]] = torch.constant.int 0
//CHECK: [[INPUT:%.]] = torch.prim.ListConstruct %arg0, %arg0 : (!torch.vtensor<[2,3],f32>, !torch.vtensor<[2,3],f32>) -> !torch.list<vtensor<[2,3],f32>>
    %0 = "onnx.Concat"(%arg0, %arg0) {axis = 0 : si64, onnx_node_name = "Concat_0"} : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<4x3xf32>
//CHECK: torch.aten.cat [[INPUT:%.]], %int[[DIM:.]] : !torch.list<vtensor<[2,3],f32>>, !torch.int -> !torch.vtensor<[4,3],f32>    
return %0 : tensor<4x3xf32>
  }
  }
