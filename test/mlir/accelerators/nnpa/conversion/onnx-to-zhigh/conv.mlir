// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --shape-inference --convert-onnx-to-zhigh %s -split-input-file | FileCheck %s

func.func @test_onnx_conv2d(%arg0: tensor<5x3x32x32xf32>, %arg1 : tensor<2x3x2x2xf32>, %arg2: tensor<2xf32>) -> tensor<*xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {kernel_shape = [2, 2]} : (tensor<5x3x32x32xf32>, tensor<2x3x2x2xf32>, tensor<2xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @test_onnx_conv2d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x3x32x32xf32>, [[PARAM_1_:%.+]]: tensor<2x3x2x2xf32>, [[PARAM_2_:%.+]]: tensor<2xf32>) -> tensor<5x2x31x31xf32> {
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "NHWC"} : (tensor<5x3x32x32xf32>) -> tensor<5x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 3, 1, 0]} : (tensor<2x3x2x2xf32>) -> tensor<2x2x3x2xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Stick"([[VAR_2_]]) {layout = "HWCK"} : (tensor<2x2x3x2xf32>) -> tensor<2x2x3x2xf16, #zhigh.layout<{dataLayout = "HWCK"}>>
// CHECK-DAG:       [[VAR_4_:%.+]] = "zhigh.Stick"([[PARAM_2_]]) {layout = "1D"} : (tensor<2xf32>) -> tensor<2xf16, #zhigh.layout<{dataLayout = "1D"}>>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Conv2D"([[VAR_1_]], [[VAR_3_]], [[VAR_4_]]) {act_func = "ACT_NONE", kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1]} : (tensor<5x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<2x2x3x2xf16, #zhigh.layout<{dataLayout = "HWCK"}>>, tensor<2xf16, #zhigh.layout<{dataLayout = "1D"}>>) -> tensor<*xf16>
// CHECK:           [[VAR_6_:%.+]] = "zhigh.Unstick"([[VAR_5_]]) : (tensor<*xf16>) -> tensor<5x2x31x31xf32>
// CHECK:           return [[VAR_6_]] : tensor<5x2x31x31xf32>
// CHECK:         }
}

// -----

func.func @test_onnx_conv2d_nobias(%arg0: tensor<5x3x32x32xf32>, %arg1 : tensor<2x3x2x2xf32>) -> tensor<*xf32> {
  %bias = "onnx.NoValue"() {value} : () -> none  
  %0 = "onnx.Conv"(%arg0, %arg1, %bias) {kernel_shape = [2, 2]} : (tensor<5x3x32x32xf32>, tensor<2x3x2x2xf32>, none) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @test_onnx_conv2d_nobias
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x3x32x32xf32>, [[PARAM_1_:%.+]]: tensor<2x3x2x2xf32>) -> tensor<5x2x31x31xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "NHWC"} : (tensor<5x3x32x32xf32>) -> tensor<5x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 3, 1, 0]} : (tensor<2x3x2x2xf32>) -> tensor<2x2x3x2xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Stick"([[VAR_2_]]) {layout = "HWCK"} : (tensor<2x2x3x2xf32>) -> tensor<2x2x3x2xf16, #zhigh.layout<{dataLayout = "HWCK"}>>
// CHECK-DAG:       [[VAR_4_:%.+]] = "zhigh.Stick"([[VAR_cst_]]) {layout = "1D"} : (none) -> none
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Conv2D"([[VAR_1_]], [[VAR_3_]], [[VAR_4_]]) {act_func = "ACT_NONE", kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1]} : (tensor<5x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<2x2x3x2xf16, #zhigh.layout<{dataLayout = "HWCK"}>>, none) -> tensor<*xf16>
// CHECK:           [[VAR_6_:%.+]] = "zhigh.Unstick"([[VAR_5_]]) : (tensor<*xf16>) -> tensor<5x2x31x31xf32>
// CHECK:           return [[VAR_6_]] : tensor<5x2x31x31xf32>
// CHECK:         }
}

// -----

func.func @test_onnx_conv2d_no_bias_unknown_bias_dims(%arg0: tensor<5x3x32x32xf32>, %arg1 : tensor<?x3x2x2xf32>) -> tensor<*xf32> {
  %bias = "onnx.NoValue"() {value} : () -> none  
  %0 = "onnx.Conv"(%arg0, %arg1, %bias) {kernel_shape = [2, 2]} : (tensor<5x3x32x32xf32>, tensor<?x3x2x2xf32>, none) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @test_onnx_conv2d_no_bias_unknown_bias_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x3x32x32xf32>, [[PARAM_1_:%.+]]: tensor<?x3x2x2xf32>) -> tensor<5x?x31x31xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "NHWC"} : (tensor<5x3x32x32xf32>) -> tensor<5x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 3, 1, 0]} : (tensor<?x3x2x2xf32>) -> tensor<2x2x3x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Stick"([[VAR_2_]]) {layout = "HWCK"} : (tensor<2x2x3x?xf32>) -> tensor<2x2x3x?xf16, #zhigh.layout<{dataLayout = "HWCK"}>>
// CHECK-DAG:       [[VAR_4_:%.+]] = "zhigh.Stick"([[VAR_cst_]]) {layout = "1D"} : (none) -> none
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Conv2D"([[VAR_1_]], [[VAR_3_]], [[VAR_4_]]) {act_func = "ACT_NONE", kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1]} : (tensor<5x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<2x2x3x?xf16, #zhigh.layout<{dataLayout = "HWCK"}>>, none) -> tensor<*xf16>
// CHECK:           [[VAR_6_:%.+]] = "zhigh.Unstick"([[VAR_5_]]) : (tensor<*xf16>) -> tensor<5x?x31x31xf32>
// CHECK:           return [[VAR_6_]] : tensor<5x?x31x31xf32>
// CHECK:         }
}

// -----

func.func @test_onnx_conv2d_kernel_64(%arg0: tensor<5x3x1024x1024xf32>, %arg1 : tensor<2x3x64x64xf32>, %arg2: tensor<2xf32>) -> tensor<*xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {kernel_shape = [64, 64]} : (tensor<5x3x1024x1024xf32>, tensor<2x3x64x64xf32>, tensor<2xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

  // CHECK-LABEL: test_onnx_conv2d_kernel_64
  // CHECK: "zhigh.Conv2D"
  // CHECK-NOT: "onnx.Conv"
}

// -----

func.func @test_onnx_conv2d_stride_13(%arg0: tensor<5x3x1024x1024xf32>, %arg1 : tensor<2x3x64x64xf32>, %arg2: tensor<2xf32>) -> tensor<*xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {kernel_shape = [64, 64], strides = [13, 13]} : (tensor<5x3x1024x1024xf32>, tensor<2x3x64x64xf32>, tensor<2xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

  // CHECK-LABEL: test_onnx_conv2d_stride_13
  // CHECK: "zhigh.Conv2D"
  // CHECK-NOT: "onnx.Conv"
}

// -----

func.func @test_onnx_conv2d_valid_padding_H_equal_KW(%arg0: tensor<?x1280x1x1xf32>, %arg1: tensor<448x1280x1x1xf32>, %arg2: tensor<448xf32>) -> tensor<*xf32> {
    %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {dilations = [1, 1], group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<?x1280x1x1xf32>, tensor<448x1280x1x1xf32>, tensor<448xf32>) -> tensor<*xf32>
    return %0 : tensor<*xf32>
  // CHECK-LABEL: test_onnx_conv2d_valid_padding_H_equal_KW
  // CHECK: "zhigh.Conv2D"
  // CHECK-NOT: "onnx.Conv"
}

// -----

func.func @test_fuse_onnx_relu_conv2d(%arg0: tensor<5x3x32x32xf32>, %arg1 : tensor<2x3x2x2xf32>, %arg2: tensor<2xf32>) -> tensor<*xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {kernel_shape = [2, 2]} : (tensor<5x3x32x32xf32>, tensor<2x3x2x2xf32>, tensor<2xf32>) -> tensor<*xf32>
  %1 = "onnx.Relu"(%0) : (tensor<*xf32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>

// CHECK-LABEL:  func @test_fuse_onnx_relu_conv2d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x3x32x32xf32>, [[PARAM_1_:%.+]]: tensor<2x3x2x2xf32>, [[PARAM_2_:%.+]]: tensor<2xf32>) -> tensor<5x2x31x31xf32> {
// CHECK-DAG:       [[VAR_1_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "NHWC"} : (tensor<5x3x32x32xf32>) -> tensor<5x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Transpose"([[PARAM_1_]]) {perm = [2, 3, 1, 0]} : (tensor<2x3x2x2xf32>) -> tensor<2x2x3x2xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = "zhigh.Stick"([[VAR_2_]]) {layout = "HWCK"} : (tensor<2x2x3x2xf32>) -> tensor<2x2x3x2xf16, #zhigh.layout<{dataLayout = "HWCK"}>>
// CHECK-DAG:       [[VAR_4_:%.+]] = "zhigh.Stick"([[PARAM_2_]]) {layout = "1D"} : (tensor<2xf32>) -> tensor<2xf16, #zhigh.layout<{dataLayout = "1D"}>>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Conv2D"([[VAR_1_]], [[VAR_3_]], [[VAR_4_]]) {act_func = "ACT_RELU", kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [1, 1]} : (tensor<5x32x32x3xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<2x2x3x2xf16, #zhigh.layout<{dataLayout = "HWCK"}>>, tensor<2xf16, #zhigh.layout<{dataLayout = "1D"}>>) -> tensor<5x31x31x2xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_6_:%.+]] = "zhigh.Unstick"([[VAR_5_]]) : (tensor<5x31x31x2xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<5x2x31x31xf32>
// CHECK:           return [[VAR_6_]] : tensor<5x2x31x31xf32>
// CHECK:         }
}

// -----

func.func @test_onnx_conv2d_not_lower_unknown_height_weight_dims(%arg0: tensor<5x3x?x?xf32>, %arg1 : tensor<2x3x2x2xf32>, %arg2: tensor<2xf32>) -> tensor<*xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {kernel_shape = [2, 2]} : (tensor<5x3x?x?xf32>, tensor<2x3x2x2xf32>, tensor<2xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
  // CHECK-LABEL: test_onnx_conv2d_not_lower_unknown_height_weight_dims
  // CHECK: "onnx.Conv"
}

// -----

func.func @test_onnx_conv2d_not_lower_not2D(%arg0: tensor<5x3x32x32x32xf32>, %arg1 : tensor<2x3x2x2x2xf32>, %arg2: tensor<2xf32>) -> tensor<*xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {kernel_shape = [2, 2, 2]} : (tensor<5x3x32x32x32xf32>, tensor<2x3x2x2x2xf32>, tensor<2xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
  // CHECK-LABEL: test_onnx_conv2d_not_lower_not2D
  // CHECK: "onnx.Conv"
}

// -----

func.func @test_onnx_conv2d_not_lower_stride_greater_than_13(%arg0: tensor<5x3x32x32xf32>, %arg1 : tensor<2x3x2x2xf32>, %arg2: tensor<2xf32>) -> tensor<*xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {kernel_shape = [2, 2], strides = [14, 14]} : (tensor<5x3x32x32xf32>, tensor<2x3x2x2xf32>, tensor<2xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
  // CHECK-LABEL: test_onnx_conv2d_not_lower_stride_greater_than_13
  // CHECK: "onnx.Conv"
}

// -----

func.func @test_onnx_conv2d_not_lower_kernel_height_greater_than_64_valid_padding(%arg0: tensor<5x3x1024x1024xf32>, %arg1 : tensor<2x3x65x65xf32>, %arg2: tensor<2xf32>) -> tensor<*xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {kernel_shape = [65, 65]} : (tensor<5x3x1024x1024xf32>, tensor<2x3x65x65xf32>, tensor<2xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
  // CHECK-LABEL: test_onnx_conv2d_not_lower_kernel_height_greater_than_64
  // CHECK: "onnx.Conv"
}

// -----

func.func @test_onnx_conv2d_not_lower_kernel_height_greater_than_64_same_padding(%arg0: tensor<5x3x1024x1024xf32>, %arg1 : tensor<2x3x128x128xf32>, %arg2: tensor<2xf32>) -> tensor<*xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {auto_pad="SAME_UPPER", kernel_shape = [128, 128]} : (tensor<5x3x1024x1024xf32>, tensor<2x3x128x128xf32>, tensor<2xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
  // CHECK-LABEL: test_onnx_conv2d_not_lower_kernel_height_greater_than_64_same_padding
  // CHECK: "onnx.Conv"
}

// -----

func.func @test_onnx_conv2d_not_lower_kernel_weight_greater_than_64_valid_padding(%arg0: tensor<5x3x1024x1024xf32>, %arg1 : tensor<2x3x65x65xf32>, %arg2: tensor<2xf32>) -> tensor<*xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {kernel_shape = [65, 65]} : (tensor<5x3x1024x1024xf32>, tensor<2x3x65x65xf32>, tensor<2xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
  // CHECK-LABEL: test_onnx_conv2d_not_lower_kernel_weight_greater_than_64
  // CHECK: "onnx.Conv"
}

// -----

func.func @test_onnx_conv2d_not_lower_kernel_weight_greater_than_64_same_padding(%arg0: tensor<5x3x1024x1024xf32>, %arg1 : tensor<2x3x128x128xf32>, %arg2: tensor<2xf32>) -> tensor<*xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {auto_pad="SAME_UPPER", kernel_shape = [128, 128]} : (tensor<5x3x1024x1024xf32>, tensor<2x3x128x128xf32>, tensor<2xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
  // CHECK-LABEL: test_onnx_conv2d_not_lower_kernel_weight_greater_than_64_same_padding
  // CHECK: "onnx.Conv"
}

// -----

func.func @test_onnx_conv2d_not_lower_group(%arg0: tensor<5x3x1024x1024xf32>, %arg1 : tensor<3x1x2x2xf32>, %arg2: tensor<3xf32>) -> tensor<*xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {group = 3 : si64, kernel_shape = [2, 2]} : (tensor<5x3x1024x1024xf32>, tensor<3x1x2x2xf32>, tensor<3xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
  // CHECK-LABEL: test_onnx_conv2d_not_lower_group
  // CHECK: "onnx.Conv"
}

// -----

func.func @test_onnx_conv2d_not_lowered_not_same_padding(%arg0: tensor<5x3x32x32xf32>, %arg1 : tensor<?x3x2x2xf32>) -> tensor<*xf32> {
  %bias = "onnx.NoValue"() {value} : () -> none  
  %0 = "onnx.Conv"(%arg0, %arg1, %bias) {auto_pad = "NOTSET", pads = [0, 0, 2, 2], kernel_shape = [2, 2]} : (tensor<5x3x32x32xf32>, tensor<?x3x2x2xf32>, none) -> tensor<*xf32>
  return %0 : tensor<*xf32>
  // CHECK-LABEL: test_onnx_conv2d_not_lowered_not_same_padding
  // CHECK: "onnx.Conv"
}

// -----

func.func @test_onnx_conv2d_not_lowered_zdnn_do_padding_differently(%arg0: tensor<1x3x224x224xf32>, %arg1 : tensor<64x3x7x7xf32>, %arg2 : tensor<64xf32>) -> tensor<*xf32> {
    %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {kernel_shape = [7, 7], onnx_node_name = "", pads = [3, 3, 3, 3], strides = [2, 2]} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>) -> tensor<*xf32>
    return %0 : tensor<*xf32>
  // CHECK-LABEL: test_onnx_conv2d_not_lowered_zdnn_do_padding_differently
  // CHECK: "onnx.Conv"
}

// -----

/// COM: Test for zdnn limitation.
/// COM: Not lowered when dimensin size exceeds DLCPP_MAXIMUM_DIMENSION_INDEX_SIZE in `third_party/zdnn-lib/zdnn_limit.h`
/// COM: DLCPP_MAXIMUM_DIMENSION_INDEX_SIZE depends on zAIU HW. Please check the value if these tests fails.

func.func @test_exceed_limit_conv2d(%arg0: tensor<32769x3x32x32xf32>, %arg1 : tensor<32769x3x2x2xf32>, %arg2: tensor<32769xf32>) -> tensor<*xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {kernel_shape = [2, 2]} : (tensor<32769x3x32x32xf32>, tensor<32769x3x2x2xf32>, tensor<32769xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @test_exceed_limit_conv2d
// CHECK:        "onnx.Conv"
}
