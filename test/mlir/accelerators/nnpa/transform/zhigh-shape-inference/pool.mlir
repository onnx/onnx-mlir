// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --shape-inference %s -split-input-file | FileCheck %s

func.func @maxpool_valid_padding(%arg0: tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16> {
  %0 = "zhigh.MaxPool2D"(%arg0) {kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-LABEL:  func @maxpool_valid_padding
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x31x31x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.MaxPool2D"([[PARAM_0_]]) {act_func = "ACT_NONE", kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1]} : (tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x31x31x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           return [[VAR_0_]] : tensor<1x31x31x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:         }
}

// -----

func.func @maxpool_same_padding(%arg0: tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16> {
  %0 = "zhigh.MaxPool2D"(%arg0) {kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-LABEL:  func @maxpool_same_padding
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.MaxPool2D"([[PARAM_0_]]) {act_func = "ACT_NONE", kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1]} : (tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           return [[VAR_0_]] : tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:         }
}

// -----

func.func @maxpool_valid_padding_unknown_dims(%arg0: tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16> {
  %0 = "zhigh.MaxPool2D"(%arg0) {kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-LABEL:  func @maxpool_valid_padding_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.MaxPool2D"([[PARAM_0_]]) {act_func = "ACT_NONE", kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1]} : (tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           return [[VAR_0_]] : tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:         }
}

// -----

func.func @maxpool_same_padding_unknown_dims(%arg0: tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16> {
  %0 = "zhigh.MaxPool2D"(%arg0) {kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-LABEL:  func @maxpool_same_padding_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.MaxPool2D"([[PARAM_0_]]) {act_func = "ACT_NONE", kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1]} : (tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           return [[VAR_0_]] : tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:         }
}

// -----

func.func @maxpool_same_padding_no_bias_unknown_dims(%arg0: tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16> {
  %0 = "zhigh.MaxPool2D"(%arg0) {kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-LABEL:  func @maxpool_same_padding_no_bias_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.MaxPool2D"([[PARAM_0_]]) {act_func = "ACT_NONE", kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1]} : (tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           return [[VAR_0_]] : tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:         }
}

// -----

func.func @avgpool_valid_padding(%arg0: tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16> {
  %0 = "zhigh.AvgPool2D"(%arg0) {kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-LABEL:  func @avgpool_valid_padding
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x31x31x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.AvgPool2D"([[PARAM_0_]]) {act_func = "ACT_NONE", kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1]} : (tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x31x31x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           return [[VAR_0_]] : tensor<1x31x31x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:         }
}

// -----

func.func @avgpool_same_padding(%arg0: tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16> {
  %0 = "zhigh.AvgPool2D"(%arg0) {kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-LABEL:  func @avgpool_same_padding
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.AvgPool2D"([[PARAM_0_]]) {act_func = "ACT_NONE", kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1]} : (tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           return [[VAR_0_]] : tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:         }
}

// -----

func.func @avgpool_valid_padding_unknown_dims(%arg0: tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16> {
  %0 = "zhigh.AvgPool2D"(%arg0) {kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-LABEL:  func @avgpool_valid_padding_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.AvgPool2D"([[PARAM_0_]]) {act_func = "ACT_NONE", kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1]} : (tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           return [[VAR_0_]] : tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:         }
}

// -----

func.func @avgpool_same_padding_unknown_dims(%arg0: tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16> {
  %0 = "zhigh.AvgPool2D"(%arg0) {kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-LABEL:  func @avgpool_same_padding_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.AvgPool2D"([[PARAM_0_]]) {act_func = "ACT_NONE", kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1]} : (tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           return [[VAR_0_]] : tensor<1x?x?x?xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:         }
}

// -----

func.func @avgpool_same_padding_no_bias_unknown_dims(%arg0: tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16> {
  %0 = "zhigh.AvgPool2D"(%arg0) {kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

// CHECK-LABEL:  func @avgpool_same_padding_no_bias_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "zhigh.AvgPool2D"([[PARAM_0_]]) {act_func = "ACT_NONE", kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1]} : (tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           return [[VAR_0_]] : tensor<1x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:         }
}
