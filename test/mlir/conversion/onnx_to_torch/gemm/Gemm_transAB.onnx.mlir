//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func @main_graph(%arg0: tensor<4x3xf32>, %arg1: tensor<5x4xf32>, %arg2: tensor<1x5xf32>) -> tensor<3x5xf32> attributes {input_names = ["a", "b", "c"], output_names = ["y"]} {
//CHECK: %[[CONST:.*]] = torch.constant.int 1
    %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {transA = 1 : si64, transB = 1 : si64} : (tensor<4x3xf32>, tensor<5x4xf32>, tensor<1x5xf32>) -> tensor<3x5xf32>
//CHECK: [[RES1:%.]] = torch.aten.t %arg0 : !torch.vtensor<[4,3],f32> -> !torch.vtensor<[3,4],f32>
//CHECK: [[RES2:%.]] = torch.aten.t %arg1 : !torch.vtensor<[5,4],f32> -> !torch.vtensor<[4,5],f32>
//CHECK: [[RES3:%.]] = torch.aten.bmm [[RES1]], [[RES2]] : !torch.vtensor<[3,4],f32>, !torch.vtensor<[4,5],f32> -> !torch.vtensor<[3,5],f32>
//CHECK: torch.aten.add.Tensor [[RES3]], %arg2, %[[CONST]] : !torch.vtensor<[3,5],f32>, !torch.vtensor<[1,5],f32>, !torch.int -> !torch.vtensor<[3,5],f32>
    return %0 : tensor<3x5xf32>
  }
}

