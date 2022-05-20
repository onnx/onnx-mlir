//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func @main_graph(%arg0: tensor<1x4x12x12xf32>) -> tensor<1x12x12x24xf32> attributes {input_names = ["input"], output_names = ["4"]} {
    %0 = "onnx.Constant"() {value = dense<"0xDEADBEEF"> : tensor<12x4x3x3xf32>} : () -> tensor<12x4x3x3xf32>
    %1 = "onnx.Constant"() {value = dense<0.0> : tensor<12xf32>} : () -> tensor<12xf32>
//CHECK: [[STRIDE:%.]] = torch.prim.ListConstruct %int1{{_*[0-9]*}}, %int1{{_*[0-9]*}} :
//CHECK: [[DILATION:%.]] = torch.prim.ListConstruct %int1{{_*[0-9]*}}, %int1{{_*[0-9]*}} :
//CHECK: [[PAD:%.]] = torch.prim.ListConstruct %int1, %int1{{_*[0-9]*}} :
//CHECK: [[RES1:%.]] = torch.aten.conv2d %arg0, %{{[^,]*}}, %{{[^,]*}}, [[STRIDE]], [[PAD]], [[DILATION]], %int1{{_*[0-9]*}} : !torch.vtensor<[1,4,12,12],f32>, !torch.vtensor<[12,4,3,3],f32>, !torch.vtensor<[12],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,12,12,12],f32>
    %2 = "onnx.Conv"(%arg0, %0, %1) {dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "Conv_0", pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x4x12x12xf32>, tensor<12x4x3x3xf32>, tensor<12xf32>) -> tensor<1x12x12x12xf32>
//CHECK: %[[DIM:int3_8]] = torch.constant.int 3
//CHECK: [[RES2:%.]] = torch.prim.ListConstruct [[RES1]], [[RES1]] : (!torch.vtensor<[1,12,12,12],f32>, !torch.vtensor<[1,12,12,12],f32>) -> !torch.list<vtensor>
    %3 = "onnx.Concat"(%2, %2) {axis = 3 : si64, onnx_node_name = "Concat_1"} : (tensor<1x12x12x12xf32>, tensor<1x12x12x12xf32>) -> tensor<1x12x12x24xf32>
//CHECK: torch.aten.cat [[RES2]], %[[DIM]] : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[1,12,12,24],f32>
    return %3 : tensor<1x12x12x24xf32>
  }
}
