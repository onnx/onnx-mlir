//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module {
  func @main_graph(%arg0: tensor<1x1x160x160xf32>) -> tensor<1x25600xf32> attributes {input_names = ["0"], output_names = ["1"]} {
    %0 = "onnx.Flatten"(%arg0) {axis = 1 : si64, onnx_node_name = "Flatten_0"} : (tensor<1x1x160x160xf32>) -> tensor<1x25600xf32>
    return %0 : tensor<1x25600xf32>
    //CHECK: %[[I0:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[1,1,160,160],f32> -> tensor<1x1x160x160xf32>
    //CHECK: %[[I1:.*]] = builtin.unrealized_conversion_cast %[[I0]] : tensor<1x1x160x160xf32> to memref<1x1x160x160xf32>
    //CHECK: %[[START:.*]] = torch.constant.int 1
    //CHECK: %[[END:.*]] = torch.constant.int -1
    //CHECK: %[[FLAT:.*]] = torch.aten.flatten.using_ints %arg0, %[[START]], %[[END]] : !torch.vtensor<[1,1,160,160],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,25600],f32>
    //CHECK: %[[O1:.*]] = torch.tensor_static_info_cast %[[FLAT]] : !torch.vtensor<[1,25600],f32> to !torch.vtensor<[1,25600],f32>
    //CHECK: return %[[O1]] : !torch.vtensor<[1,25600],f32>
  }
}
