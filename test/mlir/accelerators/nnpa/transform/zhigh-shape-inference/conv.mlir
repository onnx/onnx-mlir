// RUN: onnx-mlir-opt --maccel=NNPA --shape-inference %s -split-input-file | FileCheck %s 

func.func @conv_valid_padding(%arg0: tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, %arg1: tensor<2x2x3x1xf32, #zhigh.encoding<{dataLayout = "HWCK"}>>, %arg2: tensor<1xf32, #zhigh.encoding<{dataLayout = "1D"}>>) -> tensor<*xf32> {
  %0 = "zhigh.Conv2D"(%arg0, %arg1, %arg2) {kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<2x2x3x1xf32, #zhigh.encoding<{dataLayout = "HWCK"}>>, tensor<1xf32, #zhigh.encoding<{dataLayout = "1D"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @conv_valid_padding
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, [[PARAM_1_:%.+]]: tensor<2x2x3x1xf32, #zhigh.encoding<{dataLayout = "HWCK"}>>, [[PARAM_2_:%.+]]: tensor<1xf32, #zhigh.encoding<{dataLayout = "1D"}>>) -> tensor<1x31x31x1xf32, #zhigh.encoding<{dataLayout = "NHWC"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Conv2D"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) {act_func = "ACT_NONE", kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1]} : (tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<2x2x3x1xf32, #zhigh.encoding<{dataLayout = "HWCK"}>>, tensor<1xf32, #zhigh.encoding<{dataLayout = "1D"}>>) -> tensor<1x31x31x1xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           return [[VAR_0_]] : tensor<1x31x31x1xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:         }
}

// -----

func.func @conv_same_padding(%arg0: tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, %arg1: tensor<2x2x3x1xf32, #zhigh.encoding<{dataLayout = "HWCK"}>>, %arg2: tensor<1xf32, #zhigh.encoding<{dataLayout = "1D"}>>) -> tensor<*xf32> {
  %0 = "zhigh.Conv2D"(%arg0, %arg1, %arg2) {kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<2x2x3x1xf32, #zhigh.encoding<{dataLayout = "HWCK"}>>, tensor<1xf32, #zhigh.encoding<{dataLayout = "1D"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @conv_same_padding
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, [[PARAM_1_:%.+]]: tensor<2x2x3x1xf32, #zhigh.encoding<{dataLayout = "HWCK"}>>, [[PARAM_2_:%.+]]: tensor<1xf32, #zhigh.encoding<{dataLayout = "1D"}>>) -> tensor<1x32x32x1xf32, #zhigh.encoding<{dataLayout = "NHWC"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Conv2D"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) {act_func = "ACT_NONE", kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1]} : (tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<2x2x3x1xf32, #zhigh.encoding<{dataLayout = "HWCK"}>>, tensor<1xf32, #zhigh.encoding<{dataLayout = "1D"}>>) -> tensor<1x32x32x1xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           return [[VAR_0_]] : tensor<1x32x32x1xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:         }
}

// -----

func.func @conv_valid_padding_unknown_dims(%arg0: tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, %arg1: tensor<2x2x?x1xf32, #zhigh.encoding<{dataLayout = "HWCK"}>>, %arg2: tensor<?xf32, #zhigh.encoding<{dataLayout = "1D"}>>) -> tensor<*xf32> {
  %0 = "zhigh.Conv2D"(%arg0, %arg1, %arg2) {kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<2x2x?x1xf32, #zhigh.encoding<{dataLayout = "HWCK"}>>, tensor<?xf32, #zhigh.encoding<{dataLayout = "1D"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @conv_valid_padding_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, [[PARAM_1_:%.+]]: tensor<2x2x?x1xf32, #zhigh.encoding<{dataLayout = "HWCK"}>>, [[PARAM_2_:%.+]]: tensor<?xf32, #zhigh.encoding<{dataLayout = "1D"}>>) -> tensor<1x?x?x1xf32, #zhigh.encoding<{dataLayout = "NHWC"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Conv2D"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) {act_func = "ACT_NONE", kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1]} : (tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<2x2x?x1xf32, #zhigh.encoding<{dataLayout = "HWCK"}>>, tensor<?xf32, #zhigh.encoding<{dataLayout = "1D"}>>) -> tensor<1x?x?x1xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           return [[VAR_0_]] : tensor<1x?x?x1xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:         }
}

// -----

func.func @conv_same_padding_unknown_dims(%arg0: tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, %arg1: tensor<2x2x?x1xf32, #zhigh.encoding<{dataLayout = "HWCK"}>>, %arg2: tensor<?xf32, #zhigh.encoding<{dataLayout = "1D"}>>) -> tensor<*xf32> {
  %0 = "zhigh.Conv2D"(%arg0, %arg1, %arg2) {kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<2x2x?x1xf32, #zhigh.encoding<{dataLayout = "HWCK"}>>, tensor<?xf32, #zhigh.encoding<{dataLayout = "1D"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @conv_same_padding_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, [[PARAM_1_:%.+]]: tensor<2x2x?x1xf32, #zhigh.encoding<{dataLayout = "HWCK"}>>, [[PARAM_2_:%.+]]: tensor<?xf32, #zhigh.encoding<{dataLayout = "1D"}>>) -> tensor<1x?x?x1xf32, #zhigh.encoding<{dataLayout = "NHWC"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Conv2D"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) {act_func = "ACT_NONE", kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1]} : (tensor<1x?x?x?xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<2x2x?x1xf32, #zhigh.encoding<{dataLayout = "HWCK"}>>, tensor<?xf32, #zhigh.encoding<{dataLayout = "1D"}>>) -> tensor<1x?x?x1xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           return [[VAR_0_]] : tensor<1x?x?x1xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:         }
}

// -----

func.func @conv_same_padding_no_bias_unknown_dims(%arg0: tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, %arg1: tensor<2x2x3x1xf32, #zhigh.encoding<{dataLayout = "HWCK"}>>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "zhigh.Conv2D"(%arg0, %arg1, %cst) {kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1], act_func = "ACT_NONE"} : (tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<2x2x3x1xf32, #zhigh.encoding<{dataLayout = "HWCK"}>>, none) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @conv_same_padding_no_bias_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, [[PARAM_1_:%.+]]: tensor<2x2x3x1xf32, #zhigh.encoding<{dataLayout = "HWCK"}>>) -> tensor<1x32x32x1xf32, #zhigh.encoding<{dataLayout = "NHWC"}>> {
// CHECK:           [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Conv2D"([[PARAM_0_]], [[PARAM_1_]], [[VAR_cst_]]) {act_func = "ACT_NONE", kernel_shape = [2, 2], padding_type = "SAME_PADDING", strides = [1, 1]} : (tensor<1x32x32x3xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>, tensor<2x2x3x1xf32, #zhigh.encoding<{dataLayout = "HWCK"}>>, none) -> tensor<1x32x32x1xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:           return [[VAR_0_]] : tensor<1x32x32x1xf32, #zhigh.encoding<{dataLayout = "NHWC"}>>
// CHECK:         }
}
