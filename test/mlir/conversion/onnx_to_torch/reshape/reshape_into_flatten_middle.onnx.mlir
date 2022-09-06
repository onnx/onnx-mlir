
//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module  {
  func.func @main_graph(%arg0: tensor<2x3x4x5x6xf32>) -> tensor<2x60x6xf32> {
    //CHECK-DAG: %[[SDIM:.*]] = torch.constant.int 1
    //CHECK-DAG: %[[EDIM:.*]] = torch.constant.int 3
    %0 = "onnx.Constant"() {value = dense<[2,60,6]> : tensor<3xi64>} : () -> tensor<3xi64>
    %1 = "onnx.Reshape"(%arg0, %0) : (tensor<2x3x4x5x6xf32>, tensor<3xi64>) -> tensor<2x60x6xf32>
    //CHECK: torch.aten.flatten.using_ints %arg0, %[[SDIM]], %[[EDIM]] : !torch.vtensor<[2,3,4,5,6],f32>, !torch.int, !torch.int -> !torch.vtensor<[2,60,6],f32>
    return %1 : tensor<2x60x6xf32>
  }
}