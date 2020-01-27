// RUN: onnf-opt --shape-inference %s -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
/// Test the default behavior of transpose when no information for the
/// permutation of the axes is provided and when a permutation is provided.
//===----------------------------------------------------------------------===//

func @test_default_transpose(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) : (tensor<5x5x1x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL: test_default_transpose
// CHECK: [[RES:%.+]] = "onnx.Transpose"(%arg0) : (tensor<5x5x1x32xf32>) -> tensor<32x1x5x5xf32>
// CHECK: return [[RES]] : tensor<32x1x5x5xf32>

/// Test shape inference for transposition when perm attribute is specified.

func @test_transpose(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) {perm = [2, 0, 3, 1]} : (tensor<5x5x1x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL: test_transpose
// CHECK: [[RES_ATTR:%.+]] = "onnx.Transpose"(%arg0) {perm = [2, 0, 3, 1]} : (tensor<5x5x1x32xf32>) -> tensor<1x5x32x5xf32>
// CHECK: return [[RES_ATTR]] : tensor<1x5x32x5xf32>

//===----------------------------------------------------------------------===//
/// Test shape inference for ConvNoBias operation and all its attributes.
//===----------------------------------------------------------------------===//

/// Default and required attributes.

func @test_conv_no_bias_1(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>) -> tensor<*xf32> {
  %0 = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "NOTSET", group = 1 : i64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL: test_conv_no_bias_1
// CHECK: [[RES_ATTR:%.+]] = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "NOTSET", group = 1 : i64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>) -> tensor<1x5x27x58xf32>
// CHECK: return [[RES_ATTR]] : tensor<1x5x27x58xf32>

/// kernel_shape attribute.

func @test_conv_no_bias_2(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>) -> tensor<*xf32> {
  %0 = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "NOTSET", group = 1 : i64, kernel_shape = [8, 9]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL: test_conv_no_bias_2
// CHECK: [[RES_ATTR:%.+]] = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "NOTSET", group = 1 : i64, kernel_shape = [8, 9]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>) -> tensor<1x5x25x56xf32>
// CHECK: return [[RES_ATTR]] : tensor<1x5x25x56xf32>

/// pads attribute.
/// Use pads to make output size equal to input size by adding K - 1 to the result.

func @test_conv_no_bias_3(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "NOTSET", group = 1 : i64, pads = [2, 4, 3, 5]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL: test_conv_no_bias_3
// CHECK: [[RES_ATTR:%.+]] = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "NOTSET", group = 1 : i64, pads = [2, 4, 3, 5]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x10xf32>) -> tensor<1x5x32x64xf32>
// CHECK: return [[RES_ATTR]] : tensor<1x5x32x64xf32>

/// auto_pad set to SAME_UPPER and SAME_LOWER.

func @test_conv_no_bias_4(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "SAME_UPPER", group = 1 : i64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL: test_conv_no_bias_4
// CHECK: [[RES_ATTR:%.+]] = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "SAME_UPPER", group = 1 : i64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x10xf32>) -> tensor<1x5x32x64xf32>
// CHECK: return [[RES_ATTR]] : tensor<1x5x32x64xf32>

func @test_conv_no_bias_5(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "SAME_LOWER", group = 1 : i64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL: test_conv_no_bias_5
// CHECK: [[RES_ATTR:%.+]] = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "SAME_LOWER", group = 1 : i64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x10xf32>) -> tensor<1x5x32x64xf32>
// CHECK: return [[RES_ATTR]] : tensor<1x5x32x64xf32>

/// auto_pad set to VALID.

func @test_conv_no_bias_6(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "VALID", group = 1 : i64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL: test_conv_no_bias_6
// CHECK: [[RES_ATTR:%.+]] = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "VALID", group = 1 : i64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x10xf32>) -> tensor<1x5x27x55xf32>
// CHECK: return [[RES_ATTR]] : tensor<1x5x27x55xf32>

/// With strides attribute.

func @test_conv_no_bias_7(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>) -> tensor<*xf32> {
  %0 = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "NOTSET", group = 1 : i64, strides = [2, 3]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL: test_conv_no_bias_7
// CHECK: [[RES_ATTR:%.+]] = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "NOTSET", group = 1 : i64, strides = [2, 3]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>) -> tensor<1x5x14x20xf32>
// CHECK: return [[RES_ATTR]] : tensor<1x5x14x20xf32>

/// auto_pad set to SAME_UPPER with strides attribute.
/// The auto_pad will pas as if stride is equal to 1.

func @test_conv_no_bias_8(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>) -> tensor<*xf32> {
  %0 = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "SAME_UPPER", group = 1 : i64, strides = [2, 3]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL: test_conv_no_bias_8
// CHECK: [[RES_ATTR:%.+]] = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "SAME_UPPER", group = 1 : i64, strides = [2, 3]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>) -> tensor<1x5x16x22xf32>
// CHECK: return [[RES_ATTR]] : tensor<1x5x16x22xf32>

/// dilations attribute.

func @test_conv_no_bias_9(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>) -> tensor<*xf32> {
  %0 = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "NOTSET", group = 1 : i64, dilations = [2, 3]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL: test_conv_no_bias_9
// CHECK: [[RES_ATTR:%.+]] = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "NOTSET", dilations = [2, 3], group = 1 : i64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>) -> tensor<1x5x20x42xf32>
// CHECK: return [[RES_ATTR]] : tensor<1x5x20x42xf32>

/// dilations attribute with stride.

func @test_conv_no_bias_10(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>) -> tensor<*xf32> {
  %0 = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "NOTSET", group = 1 : i64, dilations = [2, 3], strides = [2, 2]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL: test_conv_no_bias_10
// CHECK: [[RES_ATTR:%.+]] = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "NOTSET", dilations = [2, 3], group = 1 : i64, strides = [2, 2]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>) -> tensor<1x5x10x21xf32>
// CHECK: return [[RES_ATTR]] : tensor<1x5x10x21xf32>

/// dilations attribute with auto_pad set to SAME_UPPER.

func @test_conv_no_bias_11(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>) -> tensor<*xf32> {
  %0 = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "SAME_UPPER", group = 1 : i64, dilations = [2, 3]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL: test_conv_no_bias_11
// CHECK: [[RES_ATTR:%.+]] = "onnx.ConvNoBias"(%arg0, %arg1) {auto_pad = "SAME_UPPER", dilations = [2, 3], group = 1 : i64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>) -> tensor<1x5x32x64xf32>
// CHECK: return [[RES_ATTR]] : tensor<1x5x32x64xf32>
