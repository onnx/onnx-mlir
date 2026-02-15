// RUN: onnx-mlir-opt --split-input-file --remove-dilation-conv %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Positive Tests: Should expand dilation
//===----------------------------------------------------------------------===//

// Test 1: Basic dilation=2 with 3x3 kernel -> expanded to 5x5
// CHECK-LABEL: conv_dilation_2_kernel_3x3
func.func @conv_dilation_2_kernel_3x3(%arg0: tensor<1x3x16x16x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x8x12x12x!quant.uniform<i8:f32, 0.1:0>> {
    %0 = "onnx.Constant"() {value = dense<1> : tensor<8x3x3x3xi8>} : () -> tensor<8x3x3x3x!quant.uniform<i8:f32, 0.02:0>>
    %1 = "onnx.NoValue"() {value} : () -> none
    %2 = "onnx.Conv"(%arg0, %0, %1) {auto_pad = "NOTSET", dilations = [2, 2], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x16x16x!quant.uniform<i8:f32, 0.05:0>>, tensor<8x3x3x3x!quant.uniform<i8:f32, 0.02:0>>, none) -> tensor<1x8x12x12x!quant.uniform<i8:f32, 0.1:0>>
    return %2 : tensor<1x8x12x12x!quant.uniform<i8:f32, 0.1:0>>
}
// CHECK: onnx.Conv
// CHECK-SAME: dilations = [1, 1]
// CHECK-SAME: kernel_shape = [5, 5]

// -----

// Test 2: Dilation=3 with 3x3 kernel -> expanded to 7x7
// CHECK-LABEL: conv_dilation_3_kernel_3x3
func.func @conv_dilation_3_kernel_3x3(%arg0: tensor<1x3x20x20x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x8x14x14x!quant.uniform<i8:f32, 0.1:0>> {
    %0 = "onnx.Constant"() {value = dense<1> : tensor<8x3x3x3xi8>} : () -> tensor<8x3x3x3x!quant.uniform<i8:f32, 0.02:0>>
    %1 = "onnx.NoValue"() {value} : () -> none
    %2 = "onnx.Conv"(%arg0, %0, %1) {auto_pad = "NOTSET", dilations = [3, 3], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x20x20x!quant.uniform<i8:f32, 0.05:0>>, tensor<8x3x3x3x!quant.uniform<i8:f32, 0.02:0>>, none) -> tensor<1x8x14x14x!quant.uniform<i8:f32, 0.1:0>>
    return %2 : tensor<1x8x14x14x!quant.uniform<i8:f32, 0.1:0>>
}
// CHECK: onnx.Conv
// CHECK-SAME: dilations = [1, 1]
// CHECK-SAME: kernel_shape = [7, 7]

// -----

// Test 3: Dilation=4 with 3x3 kernel -> expanded to 9x9
// CHECK-LABEL: conv_dilation_4_kernel_3x3
func.func @conv_dilation_4_kernel_3x3(%arg0: tensor<1x3x24x24x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x8x16x16x!quant.uniform<i8:f32, 0.1:0>> {
    %0 = "onnx.Constant"() {value = dense<1> : tensor<8x3x3x3xi8>} : () -> tensor<8x3x3x3x!quant.uniform<i8:f32, 0.02:0>>
    %1 = "onnx.NoValue"() {value} : () -> none
    %2 = "onnx.Conv"(%arg0, %0, %1) {auto_pad = "NOTSET", dilations = [4, 4], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x24x24x!quant.uniform<i8:f32, 0.05:0>>, tensor<8x3x3x3x!quant.uniform<i8:f32, 0.02:0>>, none) -> tensor<1x8x16x16x!quant.uniform<i8:f32, 0.1:0>>
    return %2 : tensor<1x8x16x16x!quant.uniform<i8:f32, 0.1:0>>
}
// CHECK: onnx.Conv
// CHECK-SAME: dilations = [1, 1]
// CHECK-SAME: kernel_shape = [9, 9]

// -----

// Test 4: Dilation=2 with more output channels
// CHECK-LABEL: conv_dilation_2_many_channels
func.func @conv_dilation_2_many_channels(%arg0: tensor<1x16x32x32x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x64x28x28x!quant.uniform<i8:f32, 0.1:0>> {
    %0 = "onnx.Constant"() {value = dense<1> : tensor<64x16x3x3xi8>} : () -> tensor<64x16x3x3x!quant.uniform<i8:f32, 0.02:0>>
    %1 = "onnx.NoValue"() {value} : () -> none
    %2 = "onnx.Conv"(%arg0, %0, %1) {auto_pad = "NOTSET", dilations = [2, 2], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x16x32x32x!quant.uniform<i8:f32, 0.05:0>>, tensor<64x16x3x3x!quant.uniform<i8:f32, 0.02:0>>, none) -> tensor<1x64x28x28x!quant.uniform<i8:f32, 0.1:0>>
    return %2 : tensor<1x64x28x28x!quant.uniform<i8:f32, 0.1:0>>
}
// CHECK: onnx.Conv
// CHECK-SAME: dilations = [1, 1]
// CHECK-SAME: kernel_shape = [5, 5]

// -----

// Test 5: Dilation with unsigned quantized type (u8)
// CHECK-LABEL: conv_dilation_2_u8
func.func @conv_dilation_2_u8(%arg0: tensor<1x3x16x16x!quant.uniform<u8:f32, 0.05:128>>) -> tensor<1x8x12x12x!quant.uniform<u8:f32, 0.1:128>> {
    %0 = "onnx.Constant"() {value = dense<-128> : tensor<8x3x3x3xi8>} : () -> tensor<8x3x3x3x!quant.uniform<u8:f32, 0.02:128>>
    %1 = "onnx.NoValue"() {value} : () -> none
    %2 = "onnx.Conv"(%arg0, %0, %1) {auto_pad = "NOTSET", dilations = [2, 2], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x16x16x!quant.uniform<u8:f32, 0.05:128>>, tensor<8x3x3x3x!quant.uniform<u8:f32, 0.02:128>>, none) -> tensor<1x8x12x12x!quant.uniform<u8:f32, 0.1:128>>
    return %2 : tensor<1x8x12x12x!quant.uniform<u8:f32, 0.1:128>>
}
// CHECK: onnx.Conv
// CHECK-SAME: dilations = [1, 1]
// CHECK-SAME: kernel_shape = [5, 5]

// -----

// Test 6: Dilation with strides
// CHECK-LABEL: conv_dilation_2_with_strides
func.func @conv_dilation_2_with_strides(%arg0: tensor<1x3x32x32x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x8x14x14x!quant.uniform<i8:f32, 0.1:0>> {
    %0 = "onnx.Constant"() {value = dense<1> : tensor<8x3x3x3xi8>} : () -> tensor<8x3x3x3x!quant.uniform<i8:f32, 0.02:0>>
    %1 = "onnx.NoValue"() {value} : () -> none
    %2 = "onnx.Conv"(%arg0, %0, %1) {auto_pad = "NOTSET", dilations = [2, 2], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x3x32x32x!quant.uniform<i8:f32, 0.05:0>>, tensor<8x3x3x3x!quant.uniform<i8:f32, 0.02:0>>, none) -> tensor<1x8x14x14x!quant.uniform<i8:f32, 0.1:0>>
    return %2 : tensor<1x8x14x14x!quant.uniform<i8:f32, 0.1:0>>
}
// CHECK: onnx.Conv
// CHECK-SAME: dilations = [1, 1]
// CHECK-SAME: strides = [2, 2]

// -----

//===----------------------------------------------------------------------===//
// Negative Tests: Should NOT expand dilation
//===----------------------------------------------------------------------===//

// Test 7: Dilation=1 (no dilation) should not be modified
// CHECK-LABEL: conv_no_dilation
func.func @conv_no_dilation(%arg0: tensor<1x3x16x16x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x8x14x14x!quant.uniform<i8:f32, 0.1:0>> {
    %0 = "onnx.Constant"() {value = dense<1> : tensor<8x3x3x3xi8>} : () -> tensor<8x3x3x3x!quant.uniform<i8:f32, 0.02:0>>
    %1 = "onnx.NoValue"() {value} : () -> none
    %2 = "onnx.Conv"(%arg0, %0, %1) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x16x16x!quant.uniform<i8:f32, 0.05:0>>, tensor<8x3x3x3x!quant.uniform<i8:f32, 0.02:0>>, none) -> tensor<1x8x14x14x!quant.uniform<i8:f32, 0.1:0>>
    return %2 : tensor<1x8x14x14x!quant.uniform<i8:f32, 0.1:0>>
}
// CHECK: onnx.Conv
// CHECK-SAME: dilations = [1, 1]
// CHECK-SAME: kernel_shape = [3, 3]

// -----

// Test 8: Dilation=5 (exceeds max of 4) should not be modified
// CHECK-LABEL: conv_dilation_too_large
func.func @conv_dilation_too_large(%arg0: tensor<1x3x32x32x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x8x24x24x!quant.uniform<i8:f32, 0.1:0>> {
    %0 = "onnx.Constant"() {value = dense<1> : tensor<8x3x3x3xi8>} : () -> tensor<8x3x3x3x!quant.uniform<i8:f32, 0.02:0>>
    %1 = "onnx.NoValue"() {value} : () -> none
    %2 = "onnx.Conv"(%arg0, %0, %1) {auto_pad = "NOTSET", dilations = [5, 5], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x32x32x!quant.uniform<i8:f32, 0.05:0>>, tensor<8x3x3x3x!quant.uniform<i8:f32, 0.02:0>>, none) -> tensor<1x8x24x24x!quant.uniform<i8:f32, 0.1:0>>
    return %2 : tensor<1x8x24x24x!quant.uniform<i8:f32, 0.1:0>>
}
// CHECK: onnx.Conv
// CHECK-SAME: dilations = [5, 5]
// CHECK-SAME: kernel_shape = [3, 3]

// -----

// Test 9: Non-uniform dilation [2, 3] should not be modified
// CHECK-LABEL: conv_non_uniform_dilation
func.func @conv_non_uniform_dilation(%arg0: tensor<1x3x16x16x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x8x12x10x!quant.uniform<i8:f32, 0.1:0>> {
    %0 = "onnx.Constant"() {value = dense<1> : tensor<8x3x3x3xi8>} : () -> tensor<8x3x3x3x!quant.uniform<i8:f32, 0.02:0>>
    %1 = "onnx.NoValue"() {value} : () -> none
    %2 = "onnx.Conv"(%arg0, %0, %1) {auto_pad = "NOTSET", dilations = [2, 3], group = 1 : si64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x16x16x!quant.uniform<i8:f32, 0.05:0>>, tensor<8x3x3x3x!quant.uniform<i8:f32, 0.02:0>>, none) -> tensor<1x8x12x10x!quant.uniform<i8:f32, 0.1:0>>
    return %2 : tensor<1x8x12x10x!quant.uniform<i8:f32, 0.1:0>>
}
// CHECK: onnx.Conv
// CHECK-SAME: dilations = [2, 3]
// CHECK-SAME: kernel_shape = [3, 3]

// -----

// Test 10: Non-square kernel [3, 5] should not be modified (pass requires kH==kW)
// CHECK-LABEL: conv_non_square_kernel
func.func @conv_non_square_kernel(%arg0: tensor<1x3x16x16x!quant.uniform<i8:f32, 0.05:0>>) -> tensor<1x8x12x8x!quant.uniform<i8:f32, 0.1:0>> {
    %0 = "onnx.Constant"() {value = dense<1> : tensor<8x3x3x5xi8>} : () -> tensor<8x3x3x5x!quant.uniform<i8:f32, 0.02:0>>
    %1 = "onnx.NoValue"() {value} : () -> none
    %2 = "onnx.Conv"(%arg0, %0, %1) {auto_pad = "NOTSET", dilations = [2, 2], group = 1 : si64, kernel_shape = [3, 5], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x3x16x16x!quant.uniform<i8:f32, 0.05:0>>, tensor<8x3x3x5x!quant.uniform<i8:f32, 0.02:0>>, none) -> tensor<1x8x12x8x!quant.uniform<i8:f32, 0.1:0>>
    return %2 : tensor<1x8x12x8x!quant.uniform<i8:f32, 0.1:0>>
}
// CHECK: onnx.Conv
// CHECK-SAME: dilations = [2, 2]
// CHECK-SAME: kernel_shape = [3, 5]
