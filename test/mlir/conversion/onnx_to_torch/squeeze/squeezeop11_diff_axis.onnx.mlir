//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func @main_graph(%arg0: tensor<1x2x2x1x2xf32>) -> tensor<2x2x1x2xf32> attributes {input_names = ["input"], output_names = ["output"]} {
  //CHECK: %[[DIM:.*]] = torch.constant.int 0 
  //CHECK: torch.aten.squeeze.dim %arg0, %[[DIM]] : !torch.vtensor<[1,2,2,1,2],f32>, !torch.int -> !torch.vtensor<[2,2,1,2],f32>  
 %0 = "onnx.SqueezeV11"(%arg0) {axes = [0], onnx_node_name = "Squeeze_0"} : (tensor<1x2x2x1x2xf32>) -> tensor<2x2x1x2xf32>
    return %0 : tensor<2x2x1x2xf32>
  }
}
