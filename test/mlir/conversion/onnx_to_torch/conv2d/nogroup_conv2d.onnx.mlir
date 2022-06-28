//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s

module {
  func @main_graph(%arg0: tensor<20x16x50x40xf32>) -> tensor<20x13x48x38xf32> attributes {input_names = ["0"], output_names = ["2"]} {
    %0 = "onnx.Constant"() {value = dense<0.0> : tensor<13x16x3x3xf32>} : () -> tensor<13x16x3x3xf32>
    %1 = "onnx.NoValue"() {value} : () -> none
//CHECK-DAG: %[[DIM:.*]] = torch.constant.int 0
//CHECK-DAG: %[[DIM1:.*]] = torch.constant.int 1
//CHECK-DAG: [[STRIDE:%.]] = torch.prim.ListConstruct %[[DIM1]], %[[DIM1]] : (!torch.int, !torch.int) -> !torch.list<int>
//CHECK-DAG: [[PAD:%.]] = torch.prim.ListConstruct %[[DIM]], %[[DIM]] : (!torch.int, !torch.int) -> !torch.list<int>
//CHECK-DAG: %[[NONE:.*]] = torch.constant.none
//CHECK: torch.aten.conv2d %arg0, %{{[^,]*}}, %none, [[STRIDE]], [[PAD]], [[STRIDE]], %[[DIM1]] : !torch.vtensor<[20,16,50,40],f32>, !torch.vtensor<[13,16,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[20,13,48,38],f32>
    %2 = "onnx.Conv"(%arg0, %0, %1) {dilations = [1, 1], kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<20x16x50x40xf32>, tensor<13x16x3x3xf32>, none) -> tensor<20x13x48x38xf32>
    return %2 : tensor<20x13x48x38xf32>
  }
}
