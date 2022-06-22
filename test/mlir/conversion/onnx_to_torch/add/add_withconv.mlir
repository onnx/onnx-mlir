// RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module {
  func @main_graph(%arg0: tensor<2x3x7x5xf32>) -> tensor<2x4x5x3xf32> attributes {input_names = ["input"], output_names = ["5"]} {
    %0 = "onnx.Constant"() {value = dense<"0xDEADBEEF"> : tensor<4x3x3x3xf32>} : () -> tensor<4x3x3x3xf32>
    %1 = "onnx.Constant"() {value = dense<[-0.130966514, -0.14502807, -0.00976897869, 0.126693457]> : tensor<4xf32>} : () -> tensor<4xf32>
// CHECK: [[CONST1:%.*]] = torch.constant.int 1
// CHECK: [[CONST1_0:%.*]] = torch.constant.int 1
// CHECK: [[CONST0:%.*]] = torch.constant.int 0
// CHECK-DAG: [[STRIDE:%.]] = torch.prim.ListConstruct [[CONST1_0]], [[CONST1_0]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG: [[PAD:%.]] = torch.prim.ListConstruct [[CONST0]], [[CONST0]] : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG: [[RES:%.]] = torch.aten.conv2d %arg0, %{{[^,]*}}, %{{[^,]*}}, [[STRIDE]], [[PAD]], [[STRIDE]], %{{[^,]*}} : !torch.vtensor<[2,3,7,5],f32>, !torch.vtensor<[4,3,3,3],f32>, !torch.vtensor<[4],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[2,4,5,3],f32>
    %2 = "onnx.Conv"(%arg0, %0, %1) {dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "Conv_0", pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<2x3x7x5xf32>, tensor<4x3x3x3xf32>, tensor<4xf32>) -> tensor<2x4x5x3xf32>
    %3 = "onnx.Constant"() {onnx_node_name = "Constant_1", value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
// CHECK: torch.aten.add.Tensor [[RES]], %{{[^,]*}}, [[CONST1]] : !torch.vtensor<[2,4,5,3],f32>, !torch.vtensor<[],f32>, !torch.int -> !torch.vtensor<[2,4,5,3],f32>
    %4 = "onnx.Add"(%2, %3) {onnx_node_name = "Add_2"} : (tensor<2x4x5x3xf32>, tensor<f32>) -> tensor<2x4x5x3xf32>
    return %4 : tensor<2x4x5x3xf32>
  }
}
