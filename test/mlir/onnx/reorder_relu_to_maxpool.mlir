// RUN: onnx-mlir  --useOnnxModelTypes=false --EmitONNXIR --printIR %s | FileCheck %s

func.func @test_reorder_relu_maxpool(%arg0: tensor<1x64x32x32xf32>) -> tensor<1x64x16x16xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "onnx.Relu_0"} : (tensor<1x64x32x32xf32>) -> tensor<1x64x32x32xf32>
  %1 = "onnx.MaxPoolSingleOut"(%0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [2, 2], onnx_node_name = "onnx.MaxPoolSingleOut_1", storage_order = 0 : si64, strides = [2, 2]} : (tensor<1x64x32x32xf32>) -> tensor<1x64x16x16xf32>
  return %1 : tensor<1x64x16x16xf32>

  // CHECK-LABEL: func @test_reorder_relu_maxpool
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x64x32x32xf32>) -> tensor<1x64x16x16xf32> {
  // CHECK:      [[VAR_0_:%.+]] = "onnx.MaxPoolSingleOut"([[PARAM_0_]]) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [2, 2], onnx_node_name = "onnx.MaxPoolSingleOut_0", storage_order = 0 : si64, strides = [2, 2]} : (tensor<1x64x32x32xf32>) -> tensor<1x64x16x16xf32>
  // CHECK:      [[VAR_1_:%.+]] = "onnx.Relu"([[VAR_0_]]) {onnx_node_name = "onnx.Relu_1"} : (tensor<1x64x16x16xf32>) -> tensor<1x64x16x16xf32>
  // CHECK-NEXT:     return [[VAR_1_]] : tensor<1x64x16x16xf32>

}

func.func @test_reorder_relu_maxpool_conv(%arg0: tensor<1x3x32x32xf32>) -> tensor<1x16x15x15xf32> {
  %0 = onnx.Constant dense<0.00999999977> : tensor<16x3x3x3xf32>
  %1 = onnx.Constant dense<[-0.549453557, -0.827535748, -0.358648896, 0.968641698, -0.0196946431, 0.269008577, -0.445898831, 0.947227954, 0.384573817, 1.60240877, -0.970565319, 0.224884078, -1.80497575, 1.07463968, -0.368380129, -1.6080451]> : tensor<16xf32>
  %2 = "onnx.Conv"(%arg0, %0, %1) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "onnx.Conv_0", pads = [0, 0, 0, 0]} : (tensor<1x3x32x32xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<1x16x30x30xf32>
  %3 = "onnx.Relu"(%2) {onnx_node_name = "onnx.Relu_1"} : (tensor<1x16x30x30xf32>) -> tensor<1x16x30x30xf32>
  %4 = "onnx.MaxPoolSingleOut"(%3) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [2, 2], onnx_node_name = "onnx.MaxPoolSingleOut_2", storage_order = 0 : si64, strides = [2, 2]} : (tensor<1x16x30x30xf32>) -> tensor<1x16x15x15xf32>
  return %4 : tensor<1x16x15x15xf32>

  // CHECK-LABEL: func @test_reorder_relu_maxpool_conv
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x32x32xf32>) -> tensor<1x16x15x15xf32> {
  // CHECK:      [[VAR_0_:%.+]] = onnx.Constant dense<{{.*}}> : tensor<16x3x3x3xf32>
  // CHECK:      [[VAR_1_:%.+]] = onnx.Constant dense<{{.*}}> : tensor<16xf32>
  // CHECK:      [[CONV_OUT_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]])
  // CHECK-SAME:     (tensor<1x3x32x32xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<1x16x30x30xf32>
  // CHECK: [[VAR_2_:%.+]] = "onnx.MaxPoolSingleOut"([[CONV_OUT_]])  {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [2, 2], onnx_node_name = "onnx.Conv_0_onnx.MaxPoolSingleOut_2", storage_order = 0 : si64, strides = [2, 2]} : (tensor<1x16x30x30xf32>) -> tensor<1x16x15x15xf32>
  // CHECK: [[VAR_3_:%.+]] = "onnx.Relu"([[VAR_2_]]) {onnx_node_name = "onnx.Relu_3"} : (tensor<1x16x15x15xf32>) -> tensor<1x16x15x15xf32>
  // CHECK-NEXT:     return [[VAR_3_]] : tensor<1x16x15x15xf32>
}