// RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func @main_graph(%arg0: tensor<1x5x10x10xf32>) -> tensor<1x5x5x10xf32> attributes {input_names = ["0"], output_names = ["2"]} {
// CHECK-DAG: [[STRIDE:%.]] = torch.prim.ListConstruct %int2, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG: [[PAD:%.]] = torch.prim.ListConstruct %int0, %int1, %int0, %int1 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG: [[DILATION:%.]] = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK-DAG: [[KERNEL:%.]] = torch.prim.ListConstruct %int3, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: [[PAD_ND_OP:%.]] = torch.aten.constant_pad_nd %arg0, [[PAD]], %int0 : !torch.vtensor<[1,5,10,10],f32>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,5,11,11],f32>
// CHECK-DAG: [[PAD_ZERO:%.]] = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
// CHECK: [[MAXPOOL_OP:%.]] = torch.aten.max_pool2d [[PAD_ND_OP]], [[KERNEL]], [[STRIDE]], [[PAD_ZERO]], [[DILATION]], %false : !torch.vtensor<[1,5,11,11],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,5,5,10],f32>
    %0 = "onnx.MaxPoolSingleOut"(%arg0) {kernel_shape = [3, 2], onnx_node_name = "MaxPool_0", pads = [0, 0, 1, 1], strides = [2, 1]} : (tensor<1x5x10x10xf32>) -> tensor<1x5x5x10xf32>
    return %0 : tensor<1x5x5x10xf32>
  }
}
