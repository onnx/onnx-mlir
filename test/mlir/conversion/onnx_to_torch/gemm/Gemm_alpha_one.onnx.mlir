//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func.func @main_graph(%arg0: tensor<3x5xf32>, %arg1: tensor<5x4xf32>, %arg2: tensor<1x4xf32>) -> tensor<3x4xf32> attributes {input_names = ["a", "b", "c"], output_names = ["y"]} {
//CHECK-DAG: %[[CONST:.*]] = torch.constant.int 1
    %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, onnx_node_name = "Gemm_0"} : (tensor<3x5xf32>, tensor<5x4xf32>, tensor<1x4xf32>) -> tensor<3x4xf32>
//CHECK: [[RES2:%.]] = torch.aten.mm %arg0, %arg1 {layer_name = "Gemm_0"} : !torch.vtensor<[3,5],f32>, !torch.vtensor<[5,4],f32> -> !torch.vtensor<[3,4],f32>
//CHECK: [[RES3:%.]] = torch.aten.add.Tensor [[RES2]], %arg2, %[[CONST]] {layer_name = "Gemm_0"} : !torch.vtensor<[3,4],f32>, !torch.vtensor<[1,4],f32>, !torch.int -> !torch.vtensor<[3,4],f32>   
return %0 : tensor<3x4xf32>
  }
}
