//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func @main_graph(%arg0: tensor<3x5xf32>, %arg1: tensor<5x4xf32>, %arg2: tensor<1x4xf32>) -> tensor<3x4xf32> attributes {input_names = ["a", "b", "c"], output_names = ["y"]} {
//CHECK: %[[TRANSB:.*]] = torch.constant.int 1
//CHECK: [[ALPHA:%[^ ]*]] = torch.constant.float 5.000000e-01
//CHECK: torch.aten.mul.Scalar %arg0, [[ALPHA]] : !torch.vtensor<[3,5],f32>, !torch.float -> !torch.vtensor<[3,4],f32>
    %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 5.000000e-01 : f32} : (tensor<3x5xf32>, tensor<5x4xf32>, tensor<1x4xf32>) -> tensor<3x4xf32>
//CHECK: torch.aten.bmm %{{[^,]*}}, %arg1 : !torch.vtensor<[3,4],f32>, !torch.vtensor<[5,4],f32> -> !torch.vtensor<[3,4],f32>
//CHECK: torch.aten.add.Tensor %{{[^,]*}}, %arg1, %[[TRANSB]] : !torch.vtensor<[3,4],f32>, !torch.vtensor<[5,4],f32>, !torch.int -> !torch.vtensor<[3,4],f32>
    return %0 : tensor<3x4xf32>
  }
}
