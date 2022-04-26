//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module {
  func @main_graph(%arg0: tensor<1x1x160x160xf32>) -> tensor<1x25600xf32> attributes {input_names = ["0"], output_names = ["1"]} {
//CHECK: torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[1,1,160,160],f32> -> tensor<1x1x160x160xf32>
//CHECK: builtin.unrealized_conversion_cast %0 : tensor<1x1x160x160xf32> to memref<1x1x160x160xf32>
//CHECK: [[START:%.]] = torch.constant.int 0
//CHECK: [[END:%[^ ]*]] = torch.constant.int 0   
%0 = "onnx.Flatten"(%arg0) {axis = 1 : si64, onnx_node_name = "Flatten_0"} : (tensor<1x1x160x160xf32>) -> tensor<1x25600xf32>
//CHECK: torch.aten.flatten.using_ints %arg0, [[STRAT:%.]], [[END:%[^ ]*]] : !torch.vtensor<[1,1,160,160],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,25600],f32>
//CHECK: [[START:%.]] = torch.constant.int 1
//CHECK: [[END:%[^ ]*]] = torch.constant.int 4
//CHECK: torch.aten.flatten.using_ints %2, [[START:%.]], [[END:%[^ ]*]] : !torch.vtensor<[1,25600],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,25600],f32>
//CHECK: torch.tensor_static_info_cast %3 : !torch.vtensor<[1,25600],f32> to !torch.vtensor<[1,25600],f32>
//CHECK: builtin.unrealized_conversion_cast %4 : !torch.vtensor<[1,25600],f32> to tensor<1x25600xf32>
    return %0 : tensor<1x25600xf32>
  }
}
