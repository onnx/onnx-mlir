//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @main_graph(%arg0: tensor<2x3x4x5xf32>) -> tensor<1x120xf32> attributes {input_names = ["0"], output_names = ["1"]} {
  //CHECK-DAG: %[[START:.*]] = torch.constant.int 1
  //CHECK-DAG: %[[END:.*]] = torch.constant.int 3
  //CHECK: torch.aten.flatten.using_ints %arg0, %[[START]],  %[[END]] :
    %0 = "onnx.Flatten"(%arg0) {axis = 0 : si64, onnx_node_name = "Flatten_0"} : (tensor<2x3x4x5xf32>) -> tensor<1x120xf32>
    return %0 : tensor<1x120xf32>
  }
}
