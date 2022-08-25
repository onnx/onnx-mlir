//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func @main_graph(%arg0: tensor<4x1x4x3x1x4x1xf32>) -> tensor<4x4x3x4xf32> attributes {input_names = ["input"], output_names = ["output"]} {
//CHECK-DAG: %[[CONST1:.*]] = torch.constant.int 1
//CHECK-DAG: %[[CONST2:.*]] = torch.constant.int 3
//CHECK-DAG: %[[CONST3:.*]] = torch.constant.int 4
//CHECK: [[RES1:%.]] = torch.aten.squeeze.dim %arg0, %[[CONST1]] : !torch.vtensor<[4,1,4,3,1,4,1],f32>, !torch.int -> !torch.vtensor<[4,4,3,1,4,1],f32>
//CHECK: [[RES2:%.]] = torch.aten.squeeze.dim [[RES1]], %[[CONST2]] : !torch.vtensor<[4,4,3,1,4,1],f32>, !torch.int -> !torch.vtensor<[4,4,3,4,1],f32>
//CHECK: [[RES3:%.]] = torch.aten.squeeze.dim [[RES2]], %[[CONST3]] : !torch.vtensor<[4,4,3,4,1],f32>, !torch.int -> !torch.vtensor<[4,4,3,4],f32>
    %0 = "onnx.SqueezeV11"(%arg0) {axes = [1, 6, 4], onnx_node_name = "Squeeze_0"} : (tensor<4x1x4x3x1x4x1xf32>) -> tensor<4x4x3x4xf32>
    return %0 : tensor<4x4x3x4xf32>
  }
}
