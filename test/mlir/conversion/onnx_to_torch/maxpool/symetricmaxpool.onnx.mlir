//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @main_graph(%arg0: tensor<1x3x8x8xf32>) -> tensor<1x3x3x3xf32> attributes {input_names = ["0"], output_names = ["3"]} {
//CHECK: [[STRIDE:%.]] = torch.prim.ListConstruct %int1{{_*[0-9]*}}, %int1{{_*[0-9]*}} :
//CHECK: [[DILATION:%.]] = torch.prim.ListConstruct %int1, %int1 :
//CHECK: [[PAD:%.]] = torch.prim.ListConstruct %int0, %int0{{_*[0-9]*}} :
//CHECK: [[BIAS:%.]] = torch.prim.ListConstruct %int2, %int2{{_*[0-9]*}} :
//CHECK: torch.aten.max_pool2d %arg0, [[BIAS]], [[STRIDE]], [[PAD]], [[DILATION]], %false : !torch.vtensor<[1,3,8,8],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,3,7,7],f32>    
%0 = "onnx.MaxPoolSingleOut"(%arg0) {kernel_shape = [2, 2], onnx_node_name = "MaxPool_0", pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x8x8xf32>) -> tensor<1x3x7x7xf32>
    %1 = "onnx.MaxPoolSingleOut"(%0) {kernel_shape = [3, 3], onnx_node_name = "MaxPool_1", pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x3x7x7xf32>) -> tensor<1x3x4x4xf32>
    %2 = "onnx.MaxPoolSingleOut"(%1) {kernel_shape = [4, 4], onnx_node_name = "MaxPool_2", pads = [2, 2, 2, 2], strides = [2, 2]} : (tensor<1x3x4x4xf32>) -> tensor<1x3x3x3xf32>
    return %2 : tensor<1x3x3x3xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [1 , 3 , 8 , 8] , \22name\22 : \220\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [1 , 3 , 3 , 3] , \22name\22 : \223\22 }\0A\0A]\00"} : () -> ()
}
