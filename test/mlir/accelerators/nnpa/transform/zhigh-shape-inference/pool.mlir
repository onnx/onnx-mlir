// RUN: onnx-mlir-opt --shape-inference %s -split-input-file | FileCheck %s

func @maxpool_valid_padding(%arg0: tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<*xf32> {
  %0 = "zhigh.MaxPool2D"(%arg0) {kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @maxpool_valid_padding
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x31x31x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.MaxPool2D"([[PARAM_0_]]) {act_func = "ACT_NONE", kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1]} : (tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x31x31x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           return [[VAR_0_]] : tensor<1x31x31x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:         }
}

// -----

func @maxpool_same_padding(%arg0: tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<*xf32> {
  %0 = "zhigh.MaxPool2D"(%arg0) {kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @maxpool_same_padding
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.MaxPool2D"([[PARAM_0_]]) {act_func = "ACT_NONE", kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1]} : (tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           return [[VAR_0_]] : tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:         }
}

// -----

func @maxpool_valid_padding_unknown_dims(%arg0: tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<*xf32> {
  %0 = "zhigh.MaxPool2D"(%arg0) {kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @maxpool_valid_padding_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.MaxPool2D"([[PARAM_0_]]) {act_func = "ACT_NONE", kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1]} : (tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           return [[VAR_0_]] : tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:         }
}

// -----

func @maxpool_same_padding_unknown_dims(%arg0: tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<*xf32> {
  %0 = "zhigh.MaxPool2D"(%arg0) {kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @maxpool_same_padding_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.MaxPool2D"([[PARAM_0_]]) {act_func = "ACT_NONE", kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1]} : (tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           return [[VAR_0_]] : tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:         }
}

// -----

func @maxpool_same_padding_no_bias_unknown_dims(%arg0: tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "zhigh.MaxPool2D"(%arg0) {kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @maxpool_same_padding_no_bias_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.MaxPool2D"([[PARAM_0_]]) {act_func = "ACT_NONE", kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1]} : (tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           return [[VAR_0_]] : tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:         }
}

// -----

func @avgpool_valid_padding(%arg0: tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<*xf32> {
  %0 = "zhigh.AvgPool2D"(%arg0) {kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @avgpool_valid_padding
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x31x31x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.AvgPool2D"([[PARAM_0_]]) {act_func = "ACT_NONE", kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1]} : (tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x31x31x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           return [[VAR_0_]] : tensor<1x31x31x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:         }
}

// -----

func @avgpool_same_padding(%arg0: tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<*xf32> {
  %0 = "zhigh.AvgPool2D"(%arg0) {kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @avgpool_same_padding
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.AvgPool2D"([[PARAM_0_]]) {act_func = "ACT_NONE", kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1]} : (tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           return [[VAR_0_]] : tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:         }
}

// -----

func @avgpool_valid_padding_unknown_dims(%arg0: tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<*xf32> {
  %0 = "zhigh.AvgPool2D"(%arg0) {kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @avgpool_valid_padding_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.AvgPool2D"([[PARAM_0_]]) {act_func = "ACT_NONE", kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1]} : (tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           return [[VAR_0_]] : tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:         }
}

// -----

func @avgpool_same_padding_unknown_dims(%arg0: tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<*xf32> {
  %0 = "zhigh.AvgPool2D"(%arg0) {kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @avgpool_same_padding_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.AvgPool2D"([[PARAM_0_]]) {act_func = "ACT_NONE", kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1]} : (tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           return [[VAR_0_]] : tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:         }
}

// -----

func @avgpool_same_padding_no_bias_unknown_dims(%arg0: tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "zhigh.AvgPool2D"(%arg0) {kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @avgpool_same_padding_no_bias_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.AvgPool2D"([[PARAM_0_]]) {act_func = "ACT_NONE", kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1]} : (tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>) -> tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           return [[VAR_0_]] : tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:         }
}
