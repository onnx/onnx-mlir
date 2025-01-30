// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --shape-inference --convert-onnx-to-zhigh %s -split-input-file | FileCheck %s

func.func @maxpool_should_lower_to_zhigh_padtype_valid(%arg0: tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", dilations = [1, 1], kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @maxpool_should_lower_to_zhigh_padtype_valid
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x32x32xf32>) -> tensor<1x3x31x31xf32> {
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "NHWC"} : (tensor<1x3x32x32xf32>) -> tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.MaxPool2D"([[VAR_1_]]) {kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1]} : (tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<*xf16>) -> tensor<1x3x31x31xf32>
// CHECK:           return [[VAR_3_]] : tensor<1x3x31x31xf32>
// CHECK:         }
}

// -----

func.func @maxpool_should_lower_to_zhigh_padtype_same(%arg0: tensor<1x1x5x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", dilations = [1, 1], kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x1x5x5xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @maxpool_should_lower_to_zhigh_padtype_same
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x5x5xf32>) -> tensor<1x1x3x3xf32> {
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "NHWC"} : (tensor<1x1x5x5xf32>) -> tensor<1x5x5x1xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.MaxPool2D"([[VAR_1_]]) {kernel_shape = [3, 3], padding_type = "SAME_PADDING", strides = [2, 2]} : (tensor<1x5x5x1xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<*xf16>) -> tensor<1x1x3x3xf32>
// CHECK:           return [[VAR_3_]] : tensor<1x1x3x3xf32>
// CHECK:         }
}

// -----

func.func @maxpool_should_lower_to_zhigh_same_upper(%arg0: tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "SAME_UPPER", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @maxpool_should_lower_to_zhigh_same_upper
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x32x32xf32>) -> tensor<1x3x32x32xf32> {
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "NHWC"} : (tensor<1x3x32x32xf32>) -> tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.MaxPool2D"([[VAR_1_]]) {kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1]} : (tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<*xf16>) -> tensor<1x3x32x32xf32>
// CHECK:           return [[VAR_3_]] : tensor<1x3x32x32xf32>
// CHECK:         }
}

// -----

func.func @averagepool_should_lower_to_zhigh_padtype_valid(%arg0: tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", dilations = [1, 1], kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @averagepool_should_lower_to_zhigh_padtype_valid
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x32x32xf32>) -> tensor<1x3x31x31xf32> {
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "NHWC"} : (tensor<1x3x32x32xf32>) -> tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.AvgPool2D"([[VAR_1_]]) {kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1]} : (tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<*xf16>) -> tensor<1x3x31x31xf32>
// CHECK:           return [[VAR_3_]] : tensor<1x3x31x31xf32>
// CHECK:         }
}

// -----

func.func @averagepool_should_lower_to_zhigh_padtype_same(%arg0: tensor<1x1x5x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", dilations = [1, 1], kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x1x5x5xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @averagepool_should_lower_to_zhigh_padtype_same
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x5x5xf32>) -> tensor<1x1x3x3xf32> {
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "NHWC"} : (tensor<1x1x5x5xf32>) -> tensor<1x5x5x1xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.AvgPool2D"([[VAR_1_]]) {kernel_shape = [3, 3], padding_type = "SAME_PADDING", strides = [2, 2]} : (tensor<1x5x5x1xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<*xf16>) -> tensor<1x1x3x3xf32>
// CHECK:           return [[VAR_3_]] : tensor<1x1x3x3xf32>
// CHECK:         }
}

// -----

func.func @averagepool_should_lower_to_zhigh_same_upper(%arg0: tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "SAME_UPPER", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @averagepool_should_lower_to_zhigh_same_upper
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x32x32xf32>) -> tensor<1x3x32x32xf32> {
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "NHWC"} : (tensor<1x3x32x32xf32>) -> tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.AvgPool2D"([[VAR_1_]]) {kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1]} : (tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<*xf16>) -> tensor<1x3x32x32xf32>
// CHECK:           return [[VAR_3_]] : tensor<1x3x32x32xf32>
// CHECK:         }
}

// -----
// COM: Pooling in zDNN only support 4D input (MaxPool1D and MaxPool3D not suported)
// CHECK-LABEL:  func @test_pool_not_lowered_pool1d
func.func @test_pool_not_lowered_pool1d(%arg0: tensor<1x3x32xf32>) -> (tensor<1x3x31xf32>, tensor<1x3x31xf32>) {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {kernel_shape = [2]} : (tensor<1x3x32xf32>) -> tensor<1x3x31xf32>
  // CHECK: "onnx.MaxPoolSingleOut"

  %1 = "onnx.AveragePool"(%arg0) {kernel_shape = [2]} : (tensor<1x3x32xf32>) -> tensor<1x3x31xf32>
  // CHECK: "onnx.AveragePool"

  return %0, %1 : tensor<1x3x31xf32>, tensor<1x3x31xf32>
}

// -----
// COM: Pooling in zDNN only support 4D input (MaxPool1D and MaxPool3D not suported)
// CHECK-LABEL:  func @test_pool_not_lowered_pool3d
func.func @test_pool_not_lowered_pool3d(%arg0: tensor<1x3x32x32x32xf32>) -> (tensor<1x3x31x31x31xf32>, tensor<1x3x31x31x31xf32>) {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {kernel_shape = [2, 2, 2]} : (tensor<1x3x32x32x32xf32>) -> tensor<1x3x31x31x31xf32>
  // CHECK: "onnx.MaxPoolSingleOut"

  %1 = "onnx.AveragePool"(%arg0) {kernel_shape = [2, 2, 2]} : (tensor<1x3x32x32x32xf32>) -> tensor<1x3x31x31x31xf32>
  // CHECK: "onnx.AveragePool"

  return %0, %1 : tensor<1x3x31x31x31xf32>, tensor<1x3x31x31x31xf32>
}

// -----

// CHECK-LABEL:  func @test_pool_not_lowered_ceil
func.func @test_pool_not_lowered_ceil(%arg0: tensor<1x1x4x4xf32>) -> (tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>) {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", ceil_mode = 1 : si64, dilations = [1, 1], kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x1x4x4xf32>) -> tensor<1x1x2x2xf32>
  // CHECK: "onnx.MaxPoolSingleOut"

  %1 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x1x4x4xf32>) -> tensor<1x1x2x2xf32>
  // CHECK: "onnx.AveragePool"

  return %0, %1 : tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>
}

// -----

// CHECK-LABEL:  func @test_pool_2d_not_lowered_non_same_valid_pads
func.func @test_pool_2d_not_lowered_non_same_valid_pads(%arg0: tensor<1x3x28x28xf32>) -> (tensor<1x3x30x30xf32>, tensor<1x3x30x30xf32>) {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", kernel_shape = [3, 3], pads = [2, 2, 2, 2], strides = [1, 1]} : (tensor<1x3x28x28xf32>) -> tensor<1x3x30x30xf32>
  // CHECK: "onnx.MaxPoolSingleOut"

  %1 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", kernel_shape = [3, 3], pads = [2, 2, 2, 2], strides = [1, 1]} : (tensor<1x3x28x28xf32>) -> tensor<1x3x30x30xf32>
  // CHECK: "onnx.AveragePool"
  return %0, %1 : tensor<1x3x30x30xf32>, tensor<1x3x30x30xf32>
}

// -----

// CHECK-LABEL:  func @test_pool_2d_not_lowered_kernel_greater_than_64
func.func @test_pool_2d_not_lowered_kernel_greater_than_64(%arg0: tensor<1x3x65x65xf32>) -> (tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32>){
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", dilations = [1, 1], kernel_shape = [65, 65], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x65x65xf32>) -> tensor<1x3x1x1xf32>
  // CHECK: "onnx.MaxPoolSingleOut"

  %1 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", dilations = [1, 1], kernel_shape = [65, 65], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x65x65xf32>) -> tensor<1x3x1x1xf32>
  // CHECK: "onnx.AveragePool"

  return %0, %1 : tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32>
}

// -----

// CHECK-LABEL:  func @test_pool_not_lowered_not_same_padding
func.func @test_pool_not_lowered_not_same_padding(%arg0: tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", dilations = [1, 1], kernel_shape = [2, 2], pads = [0, 0, 2, 2], strides = [1, 1]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
  // CHECK: "onnx.MaxPoolSingleOut" 
}

// -----

// CHECK-LABEL:  func @test_maxpool_not_lowered_non_default_dilations
func.func @test_maxpool_not_lowered_non_default_dilations(%arg0: tensor<1x1x4x4xf32>) -> tensor<1x1x2x2xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", dilations = [2, 2], kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x4x4xf32>) -> tensor<1x1x2x2xf32>
  return %0 : tensor<1x1x2x2xf32>
  // CHECK: "onnx.MaxPoolSingleOut"
}

// -----

// CHECK-LABEL: func @test_averagepool_2d_not_lowered_count_include_pad
func.func @test_averagepool_2d_not_lowered_count_include_pad(%arg0: tensor<1x3x28x28xf32>) -> tensor<1x3x30x30xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", count_include_pad = 1 : si64, kernel_shape = [3, 3], pads = [2, 2, 2, 2], strides = [1, 1]} : (tensor<1x3x28x28xf32>) -> tensor<1x3x30x30xf32>
  return %0 : tensor<1x3x30x30xf32>
  // CHECK: "onnx.AveragePool"
}

// -----

func.func @test_onnx_maxpool2d_computed_valid_dyn(%arg0: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", dilations = [1, 1], kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>

// CHECK-LABEL:  func @test_onnx_maxpool2d_computed_valid_dyn
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "NHWC"} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.MaxPool2D"([[VAR_1_]]) {kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1]} : (tensor<?x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<*xf16>) -> tensor<?x?x?x?xf32>
// CHECK:           return [[VAR_3_]] : tensor<?x?x?x?xf32>
// CHECK:         }
}

// -----

func.func @test_maxpool_2d_same_upper_dyn(%arg0: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "SAME_UPPER", kernel_shape = [2, 2]} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>

// CHECK-LABEL:  func @test_maxpool_2d_same_upper_dyn
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "NHWC"} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.MaxPool2D"([[VAR_1_]]) {kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1]} : (tensor<?x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<*xf16>) -> tensor<?x?x?x?xf32>
// CHECK:           return [[VAR_3_]] : tensor<?x?x?x?xf32>
// CHECK:         }
}

// -----

// COM: Can not lower because padding type can not be computed when input and output tensors has dynamic dimensions. (lowered in static dimension case)

// CHECK-LABEL:  test_maxpool_2d_padtype_not_computed_dyn
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
func.func @test_maxpool_2d_padtype_not_computed_dyn(%arg0: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", dilations = [1, 1], kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>

  // CHECK:           [[VAR_0_:%.+]] = "onnx.MaxPoolSingleOut"([[PARAM_0_]]) {auto_pad = "NOTSET", ceil_mode = 0 : si64, dilations = [1, 1], kernel_shape = [3, 3], pads = [1, 1, 1, 1], storage_order = 0 : si64, strides = [2, 2]} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  // CHECK:           return [[VAR_0_]] : tensor<?x?x?x?xf32>
}

// -----

func.func @test_averagepool_2d_computed_valid_dyn(%arg0: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>
// CHECK-LABEL:  func @test_averagepool_2d_computed_valid_dyn
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "NHWC"} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.AvgPool2D"([[VAR_1_]]) {kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1]} : (tensor<?x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<*xf16>) -> tensor<?x?x?x?xf32>
// CHECK:           return [[VAR_3_]] : tensor<?x?x?x?xf32>
// CHECK:         }
}

// -----

func.func @test_averagepool_2d_same_upper_dyn(%arg0: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "SAME_UPPER", kernel_shape = [2, 2]} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>

// CHECK-LABEL:  func @test_averagepool_2d_same_upper_dyn
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
// CHECK:           [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "NHWC"} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.AvgPool2D"([[VAR_1_]]) {kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1]} : (tensor<?x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Unstick"([[VAR_2_]]) : (tensor<*xf16>) -> tensor<?x?x?x?xf32>
// CHECK:           return [[VAR_3_]] : tensor<?x?x?x?xf32>
// CHECK:         }
}

// -----

// COM: Can not lower because padding type can not be computed when input and output tensors has dynamic dimensions. (lowered in static dimension case)

// CHECK-LABEL:  test_averagepool_2d_padtype_not_computed_dyn
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
func.func @test_averagepool_2d_padtype_not_computed_dyn(%arg0: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", dilations = [1, 1], kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  return %0 : tensor<?x?x?x?xf32>

  // CHECK:           [[VAR_0_:%.+]] = "onnx.AveragePool"([[PARAM_0_]]) {auto_pad = "NOTSET", ceil_mode = 0 : si64, count_include_pad = 0 : si64, dilations = [1, 1], kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
  // CHECK:           return [[VAR_0_]] : tensor<?x?x?x?xf32>
}

// -----

/// COM: Test for zdnn limitation.
/// COM: Not lowered when dimensin size exceeds DLCPP_MAXIMUM_DIMENSION_INDEX_SIZE in `third_party/zdnn-lib/zdnn_limit.h`
/// COM: DLCPP_MAXIMUM_DIMENSION_INDEX_SIZE depends on zAIU HW. Please check the value if these tests fails.

func.func @test_exceed_limit_maxpool(%arg0: tensor<32769x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", dilations = [1, 1], kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<32769x3x32x32xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @test_exceed_limit_maxpool
// CHECK:        "onnx.MaxPoolSingleOut"
}
