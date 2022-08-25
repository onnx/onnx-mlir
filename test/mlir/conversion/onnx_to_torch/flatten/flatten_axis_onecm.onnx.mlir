//RUN: onnx-mlir --EmitONNXIR --run-torch-pass %s -o=%t >/dev/null && cat %t.onnx.mlir | FileCheck -v %s
module attributes {}  {
  func @main_graph(%arg0: tensor<1x4x15x15xf32>) -> tensor<1x5xf32> attributes {input_names = ["input.1"], output_names = ["10"]} {
    %0 = "onnx.Constant"() {value = dense<"0xDEADBEEF"> : tensor<6x4x7x7xf32>} : () -> tensor<6x4x7x7xf32>
    %1 = "onnx.Constant"() {value = dense<"0xDEADBEEF"> : tensor<6xf32>} : () -> tensor<6xf32>
//CHECK: torch.aten.conv2d %arg0, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}} : !torch.vtensor<[1,4,15,15],f32>, !torch.vtensor<[6,4,7,7],f32>, !torch.vtensor<[6],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,6,5,5],f32>   
    %2 = "onnx.Conv"(%arg0, %0, %1) {dilations = [1, 1], group = 1 : si64, kernel_shape = [7, 7], onnx_node_name = "Conv_0", pads = [2, 2, 2, 2], strides = [3, 3]} : (tensor<1x4x15x15xf32>, tensor<6x4x7x7xf32>, tensor<6xf32>) -> tensor<1x6x5x5xf32>
    %3 = "onnx.Constant"() {value = dense<"0xDEADBEEF"> : tensor<5x6x5x5xf32>} : () -> tensor<5x6x5x5xf32>
    %4 = "onnx.Constant"() {value = dense<0.0> : tensor<5xf32>} : () -> tensor<5xf32>
//CHECK: torch.aten.conv2d %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}} : !torch.vtensor<[1,6,5,5],f32>, !torch.vtensor<[5,6,5,5],f32>, !torch.vtensor<[5],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.int -> !torch.vtensor<[1,5,2,2],f32>  
    %5 = "onnx.Conv"(%2, %3, %4) {dilations = [1, 1], group = 1 : si64, kernel_shape = [5, 5], onnx_node_name = "Conv_1", pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x6x5x5xf32>, tensor<5x6x5x5xf32>, tensor<5xf32>) -> tensor<1x5x2x2xf32>
//CHECK: torch.aten.max_pool2d %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}} : !torch.vtensor<[1,5,2,2],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,5,1,1],f32> 
    %6 = "onnx.MaxPoolSingleOut"(%5) {kernel_shape = [2, 2], onnx_node_name = "MaxPool_2", pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x5x2x2xf32>) -> tensor<1x5x1x1xf32>
//CHECK: torch.aten.flatten.using_ints %{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}} : !torch.vtensor<[1,5,1,1],f32>, !torch.int, !torch.int -> !torch.vtensor<[1,5],f32> 
    %7 = "onnx.Flatten"(%6) {axis = 1 : si64, onnx_node_name = "Flatten_3"} : (tensor<1x5x1x1xf32>) -> tensor<1x5xf32>
    return %7 : tensor<1x5xf32>
  }
}
