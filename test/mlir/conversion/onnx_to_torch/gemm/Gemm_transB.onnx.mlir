//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func @main_graph(%arg0: tensor<3x6xf32>, %arg1: tensor<4x6xf32>, %arg2: tensor<1x4xf32>) -> tensor<3x4xf32> attributes {input_names = ["a", "b", "c"], output_names = ["y"]} {
//CHECK: %[[AVAL:.*]] = torch.constant.int 0
//CHECK: %[[BVAL:.*]] = torch.constant.int 1
//CHECK: torch.aten.transpose.int %arg1, %[[AVAL]], %[[BVAL]] : !torch.vtensor<[4,6],f32>, !torch.int, !torch.int -> !torch.vtensor<[3,4],f32>
    %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {transB = 1 : si64} : (tensor<3x6xf32>, tensor<4x6xf32>, tensor<1x4xf32>) -> tensor<3x4xf32>
//CHECK: torch.aten.bmm %arg0, %{{[^,]*}} : !torch.vtensor<[3,6],f32>, !torch.vtensor<[3,4],f32> -> !torch.vtensor<[3,4],f32>
    return %0 : tensor<3x4xf32>
  }
}
