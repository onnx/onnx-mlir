//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func @main_graph(%arg0: tensor<1x3x8x8xf32>) -> tensor<1x3x7x7xf32> attributes {input_names = ["0"], output_names = ["3"]} {
 //CHECK: torch.aten.max_pool2d %arg0, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}} : !torch.vtensor<[1,3,8,8],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,3,7,7],f32>    
%0 = "onnx.MaxPoolSingleOut"(%arg0) {kernel_shape = [2, 2], onnx_node_name = "MaxPool_0", pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x8x8xf32>) -> tensor<1x3x7x7xf32>
    return %0 : tensor<1x3x7x7xf32>
  }
}
