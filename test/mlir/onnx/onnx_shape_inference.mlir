// RUN: onnf-opt --shape-inference %s -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
/// Test the default behavior of transpose when no information for the
/// permutation of the axes is provided and when a permutation is provided.
//===----------------------------------------------------------------------===//

func @test_default_transpose(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) : (tensor<5x5x1x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_default_transpose
  // CHECK: [[RES:%.+]] = "onnx.Transpose"(%arg0) : (tensor<5x5x1x32xf32>) -> tensor<32x1x5x5xf32>
  // CHECK: return [[RES]] : tensor<32x1x5x5xf32>
}

/// Test shape inference for transposition when perm attribute is specified.

func @test_transpose(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) {perm = [2, 0, 3, 1]} : (tensor<5x5x1x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_transpose
  // CHECK: [[RES_ATTR:%.+]] = "onnx.Transpose"(%arg0) {perm = [2, 0, 3, 1]} : (tensor<5x5x1x32xf32>) -> tensor<1x5x32x5xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x32x5xf32>
}

//===----------------------------------------------------------------------===//
/// Test the shape inferencing scheme for the matmul operation.
//===----------------------------------------------------------------------===//

/// MatMul: 1-D x 1-D

func @test_matmul_1(%arg0 : tensor<32xf32>, %arg1 : tensor<32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<32xf32>, tensor<32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_matmul_1
  // CHECK: [[RES1:%.+]] = "onnx.MatMul"(%arg0, %arg1) : (tensor<32xf32>, tensor<32xf32>) -> tensor<1xf32>
  // CHECK: return [[RES1]] : tensor<1xf32>
}

/// MatMul: K-D x 2-D (K > 2)

func @test_matmul_2(%arg0 : tensor<16x?x64x42xf32>, %arg1 : tensor<42x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<16x?x64x42xf32>, tensor<42x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_matmul_2
  // CHECK: [[RES2:%.+]] = "onnx.MatMul"(%arg0, %arg1) : (tensor<16x?x64x42xf32>, tensor<42x32xf32>) -> tensor<16x?x64x32xf32>
  // CHECK: return [[RES2]] : tensor<16x?x64x32xf32>
}

/// MatMul: 2-D x K-D (K > 2)

func @test_matmul_3(%arg0 : tensor<64x42xf32>, %arg1 : tensor<16x?x42x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<64x42xf32>, tensor<16x?x42x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_matmul_3
  // CHECK: [[RES3:%.+]] = "onnx.MatMul"(%arg0, %arg1) : (tensor<64x42xf32>, tensor<16x?x42x32xf32>) -> tensor<16x?x64x32xf32>
  // CHECK: return [[RES3]] : tensor<16x?x64x32xf32>
}

/// MatMul: 2-D x K-D (K > 2)

func @test_matmul_4(%arg0 : tensor<64x42xf32>, %arg1 : tensor<?x?x?x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<64x42xf32>, tensor<?x?x?x?xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_matmul_4
  // CHECK: [[RES4:%.+]] = "onnx.MatMul"(%arg0, %arg1) : (tensor<64x42xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x64x?xf32>
  // CHECK: return [[RES4]] : tensor<?x?x64x?xf32>
}

/// MatMul: K1-D x K2-D (K1 > 2, K2 > 2)

func @test_matmul_5(%arg0 : tensor<16x?x?x42xf32>, %arg1 : tensor<32x?x64x42x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<16x?x?x42xf32>, tensor<32x?x64x42x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_matmul_5
  // CHECK: [[RES5:%.+]] = "onnx.MatMul"(%arg0, %arg1) : (tensor<16x?x?x42xf32>, tensor<32x?x64x42x32xf32>) -> tensor<32x16x64x?x32xf32>
  // CHECK: return [[RES5]] : tensor<32x16x64x?x32xf32>
}

/// MatMul: 1-D x 2-D

func @test_matmul_6(%arg0 : tensor<32xf32>, %arg1 : tensor<32x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<32xf32>, tensor<32x64xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_matmul_6
  // CHECK: [[RES6:%.+]] = "onnx.MatMul"(%arg0, %arg1) : (tensor<32xf32>, tensor<32x64xf32>) -> tensor<64xf32>
  // CHECK: return [[RES6]] : tensor<64xf32>
}

/// MatMul: 2-D x 1-D

func @test_matmul_7(%arg0 : tensor<32x64xf32>, %arg1 : tensor<64xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<32x64xf32>, tensor<64xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_matmul_7
  // CHECK: [[RES7:%.+]] = "onnx.MatMul"(%arg0, %arg1) : (tensor<32x64xf32>, tensor<64xf32>) -> tensor<32xf32>
  // CHECK: return [[RES7]] : tensor<32xf32>
}

/// MatMul: 2-D x 2-D

func @test_matmul_8(%arg0 : tensor<32x64xf32>, %arg1 : tensor<64x128xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<32x64xf32>, tensor<64x128xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_matmul_8
  // CHECK: [[RES8:%.+]] = "onnx.MatMul"(%arg0, %arg1) : (tensor<32x64xf32>, tensor<64x128xf32>) -> tensor<32x128xf32>
  // CHECK: return [[RES8]] : tensor<32x128xf32>
}

/// MatMul: 1-D x N-D

func @test_matmul_9(%arg0 : tensor<42xf32>, %arg1 : tensor<?x42x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<42xf32>, tensor<?x42x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_matmul_9
  // CHECK: [[RES1:%.+]] = "onnx.MatMul"(%arg0, %arg1) : (tensor<42xf32>, tensor<?x42x32xf32>) -> tensor<?x32xf32>
  // CHECK: return [[RES1]] : tensor<?x32xf32>
}

/// MatMul: N-D x 1-D

func @test_matmul_10(%arg0 : tensor<?x42x32xf32>, %arg1 : tensor<32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x42x32xf32>, tensor<32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_matmul_10
  // CHECK: [[RES1:%.+]] = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x42x32xf32>, tensor<32xf32>) -> tensor<?x42xf32>
  // CHECK: return [[RES1]] : tensor<?x42xf32>
}

//===----------------------------------------------------------------------===//
/// Test shape inference for ConvNoBias operation and all its attributes.
//===----------------------------------------------------------------------===//

/// Default and required attributes for 1-D convolution.

func @test_conv_no_bias_0(%arg0 : tensor<1x2x32xf32>, %arg1 : tensor<5x2x6xf32>) -> tensor<*xf32> {
  %0 = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "NOTSET", group = 1 : i64} : (tensor<1x2x32xf32>, tensor<5x2x6xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_0
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "NOTSET", group = 1 : i64} : (tensor<1x2x32xf32>, tensor<5x2x6xf32>) -> tensor<1x5x27xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x27xf32>
}

/// Default and required attributes.

func @test_conv_no_bias_1(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>) -> tensor<*xf32> {
  %0 = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "NOTSET", group = 1 : i64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_1
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "NOTSET", group = 1 : i64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>) -> tensor<1x5x27x58xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x27x58xf32>
}

/// kernel_shape attribute.

func @test_conv_no_bias_2(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>) -> tensor<*xf32> {
  %0 = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "NOTSET", group = 1 : i64, kernel_shape = [8, 9]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_2
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "NOTSET", group = 1 : i64, kernel_shape = [8, 9]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>) -> tensor<1x5x25x56xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x25x56xf32>
}

/// pads attribute.
/// Use pads to make output size equal to input size by adding K - 1 to the result.

func @test_conv_no_bias_3(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "NOTSET", group = 1 : i64, pads = [2, 4, 3, 5]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_3
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "NOTSET", group = 1 : i64, pads = [2, 4, 3, 5]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x10xf32>) -> tensor<1x5x32x64xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x32x64xf32>
}

/// auto_pad set to SAME_UPPER and SAME_LOWER.

func @test_conv_no_bias_4(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "SAME_UPPER", group = 1 : i64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_4
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "NOTSET", group = 1 : i64, pads = [2, 4, 3, 5]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x10xf32>) -> tensor<1x5x32x64xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x32x64xf32>
}

func @test_conv_no_bias_5(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "SAME_LOWER", group = 1 : i64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_5
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "NOTSET", group = 1 : i64, pads = [3, 5, 2, 4]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x10xf32>) -> tensor<1x5x32x64xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x32x64xf32>
}

/// auto_pad set to VALID.

func @test_conv_no_bias_6(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "VALID", group = 1 : i64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_6
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "VALID", group = 1 : i64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x10xf32>) -> tensor<1x5x27x55xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x27x55xf32>
}

/// With strides attribute.

func @test_conv_no_bias_7(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>) -> tensor<*xf32> {
  %0 = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "NOTSET", group = 1 : i64, strides = [2, 3]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_7
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "NOTSET", group = 1 : i64, strides = [2, 3]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>) -> tensor<1x5x14x20xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x14x20xf32>
}

/// auto_pad set to SAME_UPPER with strides attribute.
/// The auto_pad will pas as if stride is equal to 1.

func @test_conv_no_bias_8(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>) -> tensor<*xf32> {
  %0 = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "SAME_UPPER", group = 1 : i64, strides = [2, 3]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_8
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "NOTSET", group = 1 : i64, pads = [18, 66, 18, 66], strides = [2, 3]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>) -> tensor<1x5x32x64xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x32x64xf32>
}

/// dilations attribute.

func @test_conv_no_bias_9(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>) -> tensor<*xf32> {
  %0 = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "NOTSET", group = 1 : i64, dilations = [2, 3]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_9
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "NOTSET", dilations = [2, 3], group = 1 : i64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>) -> tensor<1x5x20x42xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x20x42xf32>
}

/// dilations attribute with stride.

func @test_conv_no_bias_10(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>) -> tensor<*xf32> {
  %0 = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "NOTSET", group = 1 : i64, dilations = [2, 3], strides = [2, 2]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_10
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "NOTSET", dilations = [2, 3], group = 1 : i64, strides = [2, 2]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>) -> tensor<1x5x10x21xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x10x21xf32>
}

/// dilations attribute with auto_pad set to SAME_UPPER.

func @test_conv_no_bias_11(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>) -> tensor<*xf32> {
  %0 = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "SAME_UPPER", group = 1 : i64, dilations = [2, 3]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_11
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "NOTSET", dilations = [2, 3], group = 1 : i64, pads = [6, 11, 6, 11]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>) -> tensor<1x5x32x64xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x32x64xf32>
}

/// Test PadConstantValuePad

func @test_PadConstantValuePad_1(%arg0 : tensor<16x13xf32>) -> tensor<*xf32> {
  %0 = "onnx.PadConstantValuePad"(%arg0) {constant_value = 0.000000e+00 : f32, mode = "constant", pads = [0, 2, 0, 0]} : (tensor<16x13xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_PadConstantValuePad_1
  // CHECK: [[RES:%.+]] = "onnx.PadConstantValuePad"(%arg0) {constant_value = 0.000000e+00 : f32, mode = "constant", pads = [0, 2, 0, 0]} : (tensor<16x13xf32>) -> tensor<18x13xf32>
  // CHECK: return [[RES]] : tensor<18x13xf32>
}

/// Test PadConstantPad

func @test_PadConstantPad_1(%arg0 : tensor<16x13xf32>, %arg1 : tensor<*xf32>) -> tensor<*xf32> {
  %0 = "onnx.PadConstantPad"(%arg0, %arg1) {mode = "constant", pads = [0, 2, 3, 1]} : (tensor<16x13xf32>, tensor<*xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_PadConstantPad_1
  // CHECK: [[RES:%.+]] = "onnx.PadConstantPad"(%arg0, %arg1) {mode = "constant", pads = [0, 2, 3, 1]} : (tensor<16x13xf32>, tensor<*xf32>) -> tensor<18x17xf32>
  // CHECK: return [[RES]] : tensor<18x17xf32>
}

//===----------------------------------------------------------------------===//
/// Test the reshape op inference when constants are present.
//===----------------------------------------------------------------------===//

func @test_reshape_1(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {sparse_value = [], value = [5, 5, 16, 2] } : () -> tensor<4xi32>
  %1 = "onnx.Reshape"(%arg0, %0) : (tensor<5x5x1x32xf32>, tensor<4xi32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reshape_1
  // CHECK: [[RES:%.+]] = "onnx.Reshape"(%arg0, %0) : (tensor<5x5x1x32xf32>, tensor<4xi32>) -> tensor<5x5x16x2xf32>
  // CHECK: return [[RES]] : tensor<5x5x16x2xf32>
}

func @test_reshape_2(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {sparse_value = [], value = [-1, 16, 2] } : () -> tensor<3xi32>
  %1 = "onnx.Reshape"(%arg0, %0) : (tensor<5x5x1x32xf32>, tensor<3xi32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reshape_2
  // CHECK: [[RES:%.+]] = "onnx.Reshape"(%arg0, %0) : (tensor<5x5x1x32xf32>, tensor<3xi32>) -> tensor<25x16x2xf32>
  // CHECK: return [[RES]] : tensor<25x16x2xf32>
}
