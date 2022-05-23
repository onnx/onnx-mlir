//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func @main_graph(%arg0: tensor<3x6xf32>, %arg1: tensor<6x4xf32>, %arg2: tensor<3x4xf32>) -> tensor<3x4xf32> attributes {input_names = ["a", "b", "c"], output_names = ["y"]} {
//CHECK: %[[CONST:.*]] = torch.constant.int 1
    %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) : (tensor<3x6xf32>, tensor<6x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
//CHECK: [[RES:%.]] = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[3,6],f32>, !torch.vtensor<[6,4],f32> -> !torch.vtensor<[3,4],f32> 
//CHECK: torch.aten.add.Tensor [[RES]], %arg2, %[[CONST]] : !torch.vtensor<[3,4],f32>, !torch.vtensor<[3,4],f32>, !torch.int -> !torch.vtensor<[3,4],f32>   
return %0 : tensor<3x4xf32>
  }
}
