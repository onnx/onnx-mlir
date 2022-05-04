//RUN: onnx-mlir --EmitONNXIR %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module {
  func @main_graph(%arg0: tensor<1x20x48x48xf32>) -> tensor<1x2366xf32> attributes {input_names = ["input"], output_names = ["4"]} {
    %0 = "onnx.Constant"() {value = dense<"0xDEADBEEF"> : tensor<14x20x5x5xf32>} : () -> tensor<14x20x5x5xf32>
    %1 = "onnx.Constant"() {value = dense<"0xDEADBEEF"> : tensor<14xf32>} : () -> tensor<14xf32>
    %2 = "onnx.Conv"(%arg0, %0, %1) {dilations = [1, 1], group = 1 : si64, kernel_shape = [5, 5], onnx_node_name = "Conv_0", pads = [4, 4, 4, 4], strides = [4, 4]} : (tensor<1x20x48x48xf32>, tensor<14x20x5x5xf32>, tensor<14xf32>) -> tensor<1x14x13x13xf32>
    %3 = "onnx.Flatten"(%2) {axis = 1 : si64, onnx_node_name = "Flatten_1"} : (tensor<1x14x13x13xf32>) -> tensor<1x2366xf32>
    return %3 : tensor<1x2366xf32>
    //CHECK: %[[START:.*]] = torch.constant.int 1
    //CHECK: %[[END:.*]] = torch.constant.int 3
    //CHECK: %[[FLATTEN1:.*]] = torch.aten.flatten.using_ints %8, [[START]], [[END]] : !torch.vtensor<[1,14,13,13],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2366],f32>
    //CHECK: %[[FLATTEN2:.*]] = torch.aten.flatten.using_ints %[[FLATTEN1]], %[[START]], %[[END]] : !torch.vtensor<[1,2366],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,2366],f32>
    //CHECK: %[[O1:.*]] = torch.tensor_static_info_cast %[[FLATTEN2]] : !torch.vtensor<[1,2366],f32> to !torch.vtensor<[1,2366],f32>
    //CHECK: return %[[O1]] : tensor<1x2366xf32>
  }
}
