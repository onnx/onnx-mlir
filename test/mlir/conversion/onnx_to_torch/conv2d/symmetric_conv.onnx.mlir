//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func @main_graph(%arg0: tensor<1x3x8x8xf32>) -> tensor<1x8x8x8xf32> attributes {input_names = ["input.1"], output_names = ["9"]} {
    %0 = "onnx.Constant"() {value = dense<0.0> : tensor<8x3x1x1xf32>} : () -> tensor<8x3x1x1xf32>
    %1 = "onnx.Constant"() {value = dense<0.0> : tensor<8xf32>} : () -> tensor<8xf32>
//CHECK-DAG: %[[DIM:.*]] = torch.constant.int 0
//CHECK-DAG: %[[DIM1:.*]] = torch.constant.int 1
//CHECK-DAG: [[STRIDE:%.]] = torch.prim.ListConstruct %[[DIM1]], %[[DIM1]] : (!torch.int, !torch.int) -> !torch.list<int>
//CHECK-DAG: [[PAD:%.]] = torch.prim.ListConstruct %[[DIM]], %[[DIM]] : (!torch.int, !torch.int) -> !torch.list<int>
//CHECK: torch.aten.conv2d %arg0, %{{[^,]*}}, %{{[^,]*}}, [[STRIDE]], [[PAD]], [[STRIDE]], %[[DIM1]] : !torch.vtensor<[1,3,8,8],f32>, !torch.vtensor<[8,3,1,1],f32>, !torch.vtensor<[8],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,8,8,8],f32>    
   %2 = "onnx.Conv"(%arg0, %0, %1) {dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], onnx_node_name = "Conv_0", pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x8x8xf32>, tensor<8x3x1x1xf32>, tensor<8xf32>) -> tensor<1x8x8x8xf32>
    return %2 : tensor<1x8x8x8xf32>
  }
}

