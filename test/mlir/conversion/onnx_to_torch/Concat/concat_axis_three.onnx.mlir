//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func @main_graph(%arg0: tensor<1x4x12x12xf32>) -> tensor<1x12x12x24xf32> attributes {input_names = ["input"], output_names = ["4"]} {
    %0 = "onnx.Constant"() {value = dense<"0xDEADBEEF"> : tensor<12x4x3x3xf32>} : () -> tensor<12x4x3x3xf32>
    %1 = "onnx.Constant"() {value = dense<[-0.0787997097, 0.0934591516, -0.106071398, 0.105203271, 0.0158753395, -0.0936289653, -0.15529643, 0.165667564, 0.0695708394, -0.0372162871, 0.166498899, -0.0204498172]> : tensor<12xf32>} : () -> tensor<12xf32>
    %2 = "onnx.Conv"(%arg0, %0, %1) {dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "Conv_0", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x4x12x12xf32>, tensor<12x4x3x3xf32>, tensor<12xf32>) -> tensor<1x12x12x12xf32>
//CHECK: %int[[DIM:[^ ]*]] = torch.constant.int 3
//CHECK: [[INPUT:%.]] = torch.prim.ListConstruct %{{[^,]*}}, %{{[^,]*}} : (!torch.vtensor<[1,12,12,12],f32>, !torch.vtensor<[1,12,12,12],f32>) -> !torch.list<vtensor<[1,12,12,12],f32>>
    %3 = "onnx.Concat"(%2, %2) {axis = 3 : si64, onnx_node_name = "Concat_1"} : (tensor<1x12x12x12xf32>, tensor<1x12x12x12xf32>) -> tensor<1x12x12x24xf32>
//CHECK: torch.aten.cat [[INPUT:%.]],  %int[[DIM:[^ ]*]] : !torch.list<vtensor<[1,12,12,12],f32>>, !torch.int -> !torch.vtensor<[1,12,12,24],f32>
    return %3 : tensor<1x12x12x24xf32>
  }
}
