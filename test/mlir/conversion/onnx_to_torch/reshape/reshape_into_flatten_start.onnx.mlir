//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module  {
  func.func @main_graph(%arg0: tensor<1x1x1x6x128xf32>) -> tensor<6x128xf32> {
    //CHECK-DAG: %[[SDIM:.*]] = torch.constant.int 0
    //CHECK-DAG: %[[EDIM:.*]] = torch.constant.int 3
    %0 = "onnx.Constant"() {value = dense<[6, 128]> : tensor<2xi64>} : () -> tensor<2xi64>
    %1 = "onnx.Reshape"(%arg0, %0) {onnx_node_name = "Reshape_0"} : (tensor<1x1x1x6x128xf32>, tensor<2xi64>) -> tensor<6x128xf32>
    //CHECK: torch.aten.flatten.using_ints %arg0, %[[SDIM]], %[[EDIM]] {layer_name = "Reshape_0"} : !torch.vtensor<[1,1,1,6,128],f32>, !torch.int, !torch.int -> !torch.vtensor<[6,128],f32>
    return %1 : tensor<6x128xf32>
  }
}