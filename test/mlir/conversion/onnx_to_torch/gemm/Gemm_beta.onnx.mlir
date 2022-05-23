//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func @main_graph(%arg0: tensor<2x7xf32>, %arg1: tensor<7x4xf32>, %arg2: tensor<1x4xf32>) -> tensor<2x4xf32> attributes {input_names = ["a", "b", "c"], output_names = ["y"]} {
//CHECK: %[[CONST:.*]] = torch.constant.int 1
//CHECK: [[BETA:%[^ ]*]] = torch.constant.float 5.000000e-01
    %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {beta = 5.000000e-01 : f32} : (tensor<2x7xf32>, tensor<7x4xf32>, tensor<1x4xf32>) -> tensor<2x4xf32>
//CHECK: [[RES1:%.]] = torch.aten.mul.Scalar %arg2, [[BETA]] : !torch.vtensor<[1,4],f32>, !torch.float -> !torch.vtensor<[1,4],f32>
//CHECK: [[RES2:%.]] = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[2,7],f32>, !torch.vtensor<[7,4],f32> -> !torch.vtensor<[2,4],f32>
//CHECK: [[RES3:%.]] = torch.aten.add.Tensor [[RES2]], [[RES1]], %[[CONST]] : !torch.vtensor<[2,4],f32>, !torch.vtensor<[1,4],f32>, !torch.int -> !torch.vtensor<[2,4],f32>
return %0 : tensor<2x4xf32>
  }
}
