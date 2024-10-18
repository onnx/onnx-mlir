// RUN: onnx-mlir-opt  --shape-inference --canonicalize="test-convergence=true" --shape-inference --cse %s | FileCheck %s

func.func @main_graph(%arg0: tensor<1x180x320x3xf32> ) -> (tensor<1x16x90x160xf32> {onnx.name = "r3o"}) {
  %0 = onnx.Constant dense<0.1> : tensor<3x1x1xf32>
  %2 = onnx.Constant dense<0.1> : tensor<16x3x3x3xf32>
  %3 = onnx.Constant dense<0.1> : tensor<16xf32>
  %4 = onnx.Constant dense<0.1> : tensor<f32>
  %5 = onnx.Constant dense<0.1> : tensor<f32>
  %6 = onnx.Constant dense<0.1> : tensor<f32>
  %7 = onnx.Constant dense<0.1> : tensor<3x1x1xf32>
  %8 = "onnx.Transpose"(%arg0) {} : (tensor<1x180x320x3xf32>) -> tensor<1x3x180x320xf32>
  %9 = "onnx.Add"(%8, %0) {} : (tensor<1x3x180x320xf32>, tensor<3x1x1xf32>) -> tensor<1x3x180x320xf32>
  %10 = "onnx.Div"(%9, %7) {} : (tensor<1x3x180x320xf32>, tensor<3x1x1xf32>) -> tensor<1x3x180x320xf32>
  %11 = "onnx.Conv"(%10, %2, %3) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "Conv_9", pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x3x180x320xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<1x16x90x160xf32>
  %12 = "onnx.Add"(%11, %4) {onnx_node_name = "Add_11"} : (tensor<1x16x90x160xf32>, tensor<f32>) -> tensor<1x16x90x160xf32>
  return  %12 : tensor<1x16x90x160xf32>
}
"onnx.EntryPoint"() {func = @main_graph} : () -> ()

//CHECK:  %{{[0-9]+}} = "onnx.Conv"(%{{.*}}, %{{.*}}, %{{.*}}) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "Conv_9", pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x3x180x320xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<1x16x90x160xf32>
//CHECK-NEXT:  %{{[0-9]+}} = "onnx.Add"(%{{.*}}, %{{.*}}) {onnx_node_name = "Add_11"} : (tensor<1x16x90x160xf32>, tensor<f32>) -> tensor<1x16x90x160xf32>
