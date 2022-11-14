//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module  {
  func.func @main_graph(%arg0: tensor<32x512x1x1xf32>) -> tensor<32x512xf32> {
    //CHECK-DAG: %[[SDIM:.*]] = torch.constant.int 1
    //CHECK-DAG: %[[EDIM:.*]] = torch.constant.int 3
    %0 = "onnx.Constant"() {value = dense<[32,512]> : tensor<2xi64>} : () -> tensor<2xi64>
    %1 = "onnx.Reshape"(%arg0, %0) {onnx_node_name = "Reshape_0"} : (tensor<32x512x1x1xf32>, tensor<2xi64>) -> tensor<32x512xf32>
    //CHECK: torch.aten.flatten.using_ints %arg0, %[[SDIM]], %[[EDIM]] {layer_name = "Reshape_0"} : !torch.vtensor<[32,512,1,1],f32>, !torch.int, !torch.int -> !torch.vtensor<[32,512],f32>
    return %1 : tensor<32x512xf32>
  }
}