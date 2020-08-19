// RUN: onnx-mlir-opt --shape-inference %s -split-input-file | FileCheck %s

// -----

//===----------------------------------------------------------------------===//
/// Test the default behavior of transpose when no information for the
/// permutation of the axes is provided and when a permutation is provided.
//===----------------------------------------------------------------------===//

func @test_default_transpose(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) : (tensor<5x5x1x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_default_transpose
  // CHECK: [[RES:%.+]] = "onnx.Transpose"(%arg0) {perm = [3, 2, 1, 0]} : (tensor<5x5x1x32xf32>) -> tensor<32x1x5x5xf32>
  // CHECK: return [[RES]] : tensor<32x1x5x5xf32>
}

// -----

/// Test shape inference for transposition when perm attribute is specified.

func @test_transpose(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) {perm = [2, 0, 3, 1]} : (tensor<5x5x1x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_transpose
  // CHECK: [[RES_ATTR:%.+]] = "onnx.Transpose"(%arg0) {perm = [2, 0, 3, 1]} : (tensor<5x5x1x32xf32>) -> tensor<1x5x32x5xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x32x5xf32>
}

// -----

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

// -----

/// MatMul: K-D x 2-D (K > 2)

func @test_matmul_2(%arg0 : tensor<16x?x64x42xf32>, %arg1 : tensor<42x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<16x?x64x42xf32>, tensor<42x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_matmul_2
  // CHECK: [[RES2:%.+]] = "onnx.MatMul"(%arg0, %arg1) : (tensor<16x?x64x42xf32>, tensor<42x32xf32>) -> tensor<16x?x64x32xf32>
  // CHECK: return [[RES2]] : tensor<16x?x64x32xf32>
}

// -----

/// MatMul: 2-D x K-D (K > 2)

func @test_matmul_3(%arg0 : tensor<64x42xf32>, %arg1 : tensor<16x?x42x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<64x42xf32>, tensor<16x?x42x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_matmul_3
  // CHECK: [[RES3:%.+]] = "onnx.MatMul"(%arg0, %arg1) : (tensor<64x42xf32>, tensor<16x?x42x32xf32>) -> tensor<16x?x64x32xf32>
  // CHECK: return [[RES3]] : tensor<16x?x64x32xf32>
}

// -----

/// MatMul: 2-D x K-D (K > 2)

func @test_matmul_4(%arg0 : tensor<64x42xf32>, %arg1 : tensor<?x?x?x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<64x42xf32>, tensor<?x?x?x?xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_matmul_4
  // CHECK: [[RES4:%.+]] = "onnx.MatMul"(%arg0, %arg1) : (tensor<64x42xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x64x?xf32>
  // CHECK: return [[RES4]] : tensor<?x?x64x?xf32>
}

// -----

/// MatMul: K1-D x K2-D (K1 > 2, K2 > 2)

func @test_matmul_5(%arg0 : tensor<16x?x?x42xf32>, %arg1 : tensor<32x?x64x42x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<16x?x?x42xf32>, tensor<32x?x64x42x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_matmul_5
  // CHECK: [[RES5:%.+]] = "onnx.MatMul"(%arg0, %arg1) : (tensor<16x?x?x42xf32>, tensor<32x?x64x42x32xf32>) -> tensor<32x16x64x?x32xf32>
  // CHECK: return [[RES5]] : tensor<32x16x64x?x32xf32>
}

// -----

/// MatMul: 1-D x 2-D

func @test_matmul_6(%arg0 : tensor<32xf32>, %arg1 : tensor<32x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<32xf32>, tensor<32x64xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_matmul_6
  // CHECK: [[RES6:%.+]] = "onnx.MatMul"(%arg0, %arg1) : (tensor<32xf32>, tensor<32x64xf32>) -> tensor<64xf32>
  // CHECK: return [[RES6]] : tensor<64xf32>
}

// -----

/// MatMul: 2-D x 1-D

func @test_matmul_7(%arg0 : tensor<32x64xf32>, %arg1 : tensor<64xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<32x64xf32>, tensor<64xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_matmul_7
  // CHECK: [[RES7:%.+]] = "onnx.MatMul"(%arg0, %arg1) : (tensor<32x64xf32>, tensor<64xf32>) -> tensor<32xf32>
  // CHECK: return [[RES7]] : tensor<32xf32>
}

// -----

/// MatMul: 2-D x 2-D

func @test_matmul_8(%arg0 : tensor<32x64xf32>, %arg1 : tensor<64x128xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<32x64xf32>, tensor<64x128xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_matmul_8
  // CHECK: [[RES8:%.+]] = "onnx.MatMul"(%arg0, %arg1) : (tensor<32x64xf32>, tensor<64x128xf32>) -> tensor<32x128xf32>
  // CHECK: return [[RES8]] : tensor<32x128xf32>
}

// -----

/// MatMul: 1-D x N-D

func @test_matmul_9(%arg0 : tensor<42xf32>, %arg1 : tensor<?x42x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<42xf32>, tensor<?x42x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_matmul_9
  // CHECK: [[RES1:%.+]] = "onnx.MatMul"(%arg0, %arg1) : (tensor<42xf32>, tensor<?x42x32xf32>) -> tensor<?x32xf32>
  // CHECK: return [[RES1]] : tensor<?x32xf32>
}

// -----

/// MatMul: N-D x 1-D

func @test_matmul_10(%arg0 : tensor<?x42x32xf32>, %arg1 : tensor<32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x42x32xf32>, tensor<32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_matmul_10
  // CHECK: [[RES1:%.+]] = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x42x32xf32>, tensor<32xf32>) -> tensor<?x42xf32>
  // CHECK: return [[RES1]] : tensor<?x42xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// Test shape inference for Conv (first with no bias) operation and all its attributes.
//===----------------------------------------------------------------------===//

/// Default and required attributes for 1-D convolution.

func @test_conv_no_bias_0(%arg0 : tensor<1x2x32xf32>, %arg1 : tensor<5x2x6xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", group = 1 : i64} : (tensor<1x2x32xf32>, tensor<5x2x6xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_0
  // CHECK: [[RES_ATTR:%.+]] = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", dilations = [1], group = 1 : i64, kernel_shape = [6], pads = [0, 0], strides = [1]} : (tensor<1x2x32xf32>, tensor<5x2x6xf32>, none) -> tensor<1x5x27xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x27xf32>
}

// -----

/// Default and required attributes.

func @test_conv_no_bias_1(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", group = 1 : i64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_1
  // CHECK: [[RES_ATTR:%.+]] = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : i64, kernel_shape = [6, 7], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, none) -> tensor<1x5x27x58xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x27x58xf32>
}

// -----

/// kernel_shape attribute.

func @test_conv_no_bias_2(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", group = 1 : i64, kernel_shape = [8, 9]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_2
  // CHECK: [[RES_ATTR:%.+]] = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : i64, kernel_shape = [8, 9], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, none) -> tensor<1x5x25x56xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x25x56xf32>
}

// -----

/// pads attribute.
/// Use pads to make output size equal to input size by adding K - 1 to the result.

func @test_conv_no_bias_3(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x10xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", group = 1 : i64, pads = [2, 4, 3, 5]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x10xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_3
  // CHECK: [[RES_ATTR:%.+]] = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : i64, kernel_shape = [6, 10], pads = [2, 4, 3, 5], strides = [1, 1]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x10xf32>, none) -> tensor<1x5x32x64xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x32x64xf32>
}

// -----

/// auto_pad set to SAME_UPPER and SAME_LOWER.

func @test_conv_no_bias_4(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x10xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "SAME_UPPER", group = 1 : i64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x10xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_4
  // CHECK: [[RES_ATTR:%.+]] = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : i64, kernel_shape = [6, 10], pads = [2, 4, 3, 5], strides = [1, 1]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x10xf32>, none) -> tensor<1x5x32x64xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x32x64xf32>
}

// -----

func @test_conv_no_bias_5(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x10xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "SAME_LOWER", group = 1 : i64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x10xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_5
  // CHECK: [[RES_ATTR:%.+]] = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : i64, kernel_shape = [6, 10], pads = [3, 5, 2, 4], strides = [1, 1]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x10xf32>, none) -> tensor<1x5x32x64xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x32x64xf32>
}

// -----

/// auto_pad set to VALID.

func @test_conv_no_bias_6(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x10xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "VALID", group = 1 : i64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x10xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_6
  // CHECK: [[RES_ATTR:%.+]] = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : i64, kernel_shape = [6, 10], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x10xf32>, none) -> tensor<1x5x27x55xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x27x55xf32>
}

// -----

/// With strides attribute.

func @test_conv_no_bias_7(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", group = 1 : i64, strides = [2, 3]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_7
  // CHECK: [[RES_ATTR:%.+]] = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : i64, kernel_shape = [6, 7], pads = [0, 0, 0, 0], strides = [2, 3]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, none) -> tensor<1x5x14x20xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x14x20xf32>
}

// -----

/// auto_pad set to SAME_UPPER with strides attribute.
/// The auto_pad will pas as if stride is equal to 1.

func @test_conv_no_bias_8(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "SAME_UPPER", group = 1 : i64, strides = [2, 3]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_8
  // CHECK: [[RES_ATTR:%.+]] = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : i64, kernel_shape = [6, 7], pads = [2, 3, 2, 3], strides = [2, 3]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, none) -> tensor<1x5x16x22xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x16x22xf32>
}

// -----

/// dilations attribute.

func @test_conv_no_bias_9(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", group = 1 : i64, dilations = [2, 3]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_9
  // CHECK: [[RES_ATTR:%.+]] = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", dilations = [2, 3], group = 1 : i64, kernel_shape = [6, 7], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, none) -> tensor<1x5x22x46xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x22x46xf32>
}

// -----

/// dilations attribute with stride.

func @test_conv_no_bias_10(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", group = 1 : i64, dilations = [2, 3], strides = [2, 2]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_10
  // CHECK: [[RES_ATTR:%.+]] = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", dilations = [2, 3], group = 1 : i64, kernel_shape = [6, 7], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, none) -> tensor<1x5x11x23xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x11x23xf32>
}

// -----

/// dilations attribute with auto_pad set to SAME_UPPER.

func @test_conv_no_bias_11(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "SAME_UPPER", group = 1 : i64, dilations = [2, 3]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_11
  // CHECK: [[RES_ATTR:%.+]] = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", dilations = [2, 3], group = 1 : i64, kernel_shape = [6, 7], pads = [5, 9, 5, 9], strides = [1, 1]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, none) -> tensor<1x5x32x64xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x32x64xf32>
}
 
// -----

// Test convolution with bias input.

func @test_conv_12(%arg0 : tensor<1x2x32xf32>, %arg1 : tensor<5x2x6xf32>, %arg2 : tensor<5xf32>) -> tensor<*xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {auto_pad = "NOTSET", group = 1 : i64} : (tensor<1x2x32xf32>, tensor<5x2x6xf32>, tensor<5xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_12
  // CHECK: [[RES_ATTR:%.+]] = "onnx.Conv"(%arg0, %arg1, %arg2) {auto_pad = "NOTSET", dilations = [1], group = 1 : i64, kernel_shape = [6], pads = [0, 0], strides = [1]} : (tensor<1x2x32xf32>, tensor<5x2x6xf32>, tensor<5xf32>) -> tensor<1x5x27xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x27xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// Test shape inference for ConvTranspose.
//===----------------------------------------------------------------------===//

func @test_conv_transpose_1(%arg0 : tensor<1x64x36x48xf32>, %arg1 : tensor<64x1x2x2xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %0 = "onnx.ConvTranspose"(%arg0, %arg1, %cst) {dilations = [1, 1], kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x64x36x48xf32>, tensor<64x1x2x2xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_transpose_1
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvTranspose"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : i64, kernel_shape = [2, 2], output_shape = [1, 1, 72, 96], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x64x36x48xf32>, tensor<64x1x2x2xf32>, none) -> tensor<1x1x72x96xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x1x72x96xf32>
}

func @test_conv_transpose_2(%arg0 : tensor<1x64x36x48xf32>, %arg1 : tensor<64x1x2x2xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %0 = "onnx.ConvTranspose"(%arg0, %arg1, %cst) {dilations = [1, 1], group = 64 : i64, kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x64x36x48xf32>, tensor<64x1x2x2xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_transpose_2
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvTranspose"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", dilations = [1, 1], group = 64 : i64, kernel_shape = [2, 2], output_shape = [1, 64, 72, 96], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x64x36x48xf32>, tensor<64x1x2x2xf32>, none) -> tensor<1x64x72x96xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x64x72x96xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// Test shape inference for PadConstantValuePad.
//===----------------------------------------------------------------------===//

/// Test Pad_1
func @test_Pad_1(%arg0 : tensor<16x13xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %0 = "onnx.Pad"(%arg0, %cst, %cst) {constant_value = dense<0.000000e+00> : tensor<1xf32>, mode = "constant", pads = [0, 2, 2, 4]} : (tensor<16x13xf32>, none, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_Pad_1
  // CHECK-NEXT: [[NONE:%.+]] = constant unit
  // CHECK: [[RES:%.+]] = "onnx.Pad"(%arg0, [[NONE]], [[NONE]]) {constant_value = dense<0.000000e+00> : tensor<1xf32>, mode = "constant", pads = [0, 2, 2, 4]} : (tensor<16x13xf32>, none, none) -> tensor<18x19xf32>
  // CHECK: return [[RES]] : tensor<18x19xf32>
}

/// Test Pad_2
func @test_Pad_2(%arg0 : tensor<16x13xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %0 = "onnx.Pad"(%arg0, %cst, %cst) {mode = "edge", pads = [0, 2, 2, 4]} : (tensor<16x13xf32>, none, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_Pad_2
  // CHECK-NEXT: [[NONE:%.+]] = constant unit
  // CHECK: [[RES:%.+]] = "onnx.Pad"(%arg0, [[NONE]], [[NONE]]) {mode = "edge", pads = [0, 2, 2, 4]} : (tensor<16x13xf32>, none, none) -> tensor<18x19xf32>
  // CHECK: return [[RES]] : tensor<18x19xf32>
}

/// Test PadConstantValuePad_1
func @test_PadConstantValuePad_1(%arg0 : tensor<16x13xf32>) -> tensor<*xf32> {
  %0 = "onnx.PadConstantValuePad"(%arg0) {constant_value = 0.000000e+00 : f32, mode = "constant", pads = [0, 0, 2, 0]} : (tensor<16x13xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_PadConstantValuePad_1
  // CHECK: [[RES:%.+]] = "onnx.PadConstantValuePad"(%arg0) {constant_value = 0.000000e+00 : f32, mode = "constant", pads = [0, 0, 2, 0]} : (tensor<16x13xf32>) -> tensor<18x13xf32>
  // CHECK: return [[RES]] : tensor<18x13xf32>
}

// -----

/// Test PadConstantPad_1
func @test_PadConstantPad_1(%arg0 : tensor<16x13xf32>, %arg1 : tensor<*xf32>) -> tensor<*xf32> {
  %0 = "onnx.PadConstantPad"(%arg0, %arg1) {mode = "constant", pads = [0, 3, 2, 1]} : (tensor<16x13xf32>, tensor<*xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK-LABEL: test_PadConstantPad_1
  // CHECK: [[RES:%.+]] = "onnx.PadConstantPad"(%arg0, %arg1) {mode = "constant", pads = [0, 3, 2, 1]} : (tensor<16x13xf32>, tensor<*xf32>) -> tensor<18x17xf32>
  // CHECK: return [[RES]] : tensor<18x17xf32>
}

// -----

/// Test PadConstantPad_2
func @test_PadConstantPad_2(%arg0 : tensor<16x?xf32>, %arg1 : tensor<*xf32>) -> tensor<*xf32> {
  %0 = "onnx.PadConstantPad"(%arg0, %arg1) {mode = "constant", pads = [0, 3, 2, 1]} : (tensor<16x?xf32>, tensor<*xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_PadConstantPad_2
  // CHECK: [[RES:%.+]] = "onnx.PadConstantPad"(%arg0, %arg1) {mode = "constant", pads = [0, 3, 2, 1]} : (tensor<16x?xf32>, tensor<*xf32>) -> tensor<18x?xf32>
  // CHECK: return [[RES]] : tensor<18x?xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// Test for constant op.
//===----------------------------------------------------------------------===//

/// Test ConstantOp shape inference for 1-D dense tensor.
func @test_constant_dense_1d_value() -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[0.0, 1.0, 2.0]> : tensor<3xf32>} : () -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_constant_dense_1d_value
  // CHECK: [[RES:%.+]] = "onnx.Constant"() {value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<3xf32>} : () -> tensor<3xf32>
  // CHECK: return [[RES]] : tensor<3xf32>
}

// -----

/// Test ConstantOp shape inference for 2-D dense tensor.
func @test_constant_dense_2d_value() -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[[0.0, 0.0], [1.0, 1.1], [2.0, 2.1]]> : tensor<3x2xf32>} : () -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_constant_dense_2d_value
  // CHECK: [[RES:%.+]] = "onnx.Constant"() {value = dense<{{\[}}[0.000000e+00, 0.000000e+00], [1.000000e+00, 1.100000e+00], [2.000000e+00, 2.100000e+00{{\]}}]> : tensor<3x2xf32>} : () -> tensor<3x2xf32>
  // CHECK: return [[RES]] : tensor<3x2xf32>
}

// -----

/// Test ConstantOp shape inference for 1-D sparse tensor.
func @test_constant_sparse_1d_value() -> tensor<*xf32> {
  %0 = "onnx.Constant"() {sparse_value = sparse<[[0]], [1.0]> : tensor<3xf32>} : () -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_constant_sparse_1d_value
  // CHECK: [[RES:%.+]] = "onnx.Constant"() {sparse_value = sparse<0, 1.000000e+00> : tensor<3xf32>} : () -> tensor<3xf32>
  // CHECK: return [[RES]] : tensor<3xf32>
}

// -----

/// Test ConstantOp shape inference for 2-D sparse tensor.
func @test_constant_sparse_2d_value() -> tensor<*xf32> {
  %0 = "onnx.Constant"() {sparse_value = sparse<[[0, 1]], [2.0]> : tensor<3x2xf32>} : () -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_constant_sparse_2d_value
  // CHECK: [[RES:%.+]] = "onnx.Constant"() {sparse_value = sparse<{{\[}}[0, 1{{\]}}], 2.000000e+00> : tensor<3x2xf32>} : () -> tensor<3x2xf32>
  // CHECK: return [[RES]] : tensor<3x2xf32>
}

// -----

/// Test the default behavior of Average Pool with no padding (pad are set but shoud be ignored)
func @test_default_averagepool(%arg0 : tensor<5x5x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "VALID", ceil_mode = 0, kernel_shape = [3,3], pads = [1, 1, 1, 1] } : (tensor<5x5x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_default_averagepool
  // CHECK: [[RES:%.+]] = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : i64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32>
  // CHECK: return [[RES]] : tensor<5x5x30x30xf32>
}

// -----

/// Test the default behavior of Average Pool with no padding (pad are not set, default to zero)
func @test_default_averagepool_defpad(%arg0 : tensor<5x5x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0, kernel_shape = [3,3]} : (tensor<5x5x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_default_averagepool_defpad
  // CHECK: [[RES:%.+]] = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : i64, kernel_shape = [3, 3], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32>
  // CHECK: return [[RES]] : tensor<5x5x30x30xf32>
}

// -----

/// Test the default behavior of Average Pool with uniform padding
func @test_default_averagepool_pad(%arg0 : tensor<5x5x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0, kernel_shape = [3,3], pads = [1, 1, 1, 1] } : (tensor<5x5x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_default_averagepool_pad
  // CHECK: [[RES:%.+]] = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : i64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xf32>
  // CHECK: return [[RES]] : tensor<5x5x32x32xf32>
}

// -----

/// Test the default behavior of Average Pool with non uniform padding
func @test_default_averagepool_pad_nonunif(%arg0 : tensor<5x5x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0, kernel_shape = [5,3], pads = [2, 1, 1, 0] } : (tensor<5x5x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_default_averagepool_pad_nonunif
  // CHECK: [[RES:%.+]] = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : i64, kernel_shape = [5, 3], pads = [2, 1, 1, 0], strides = [1, 1]} : (tensor<5x5x32x32xf32>) -> tensor<5x5x31x31xf32>
  // CHECK: return [[RES]] : tensor<5x5x31x31xf32>
}

// -----

/// Test the default behavior of Average Pool with non uniform padding
func @test_default_averagepool_strides(%arg0 : tensor<5x5x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0, kernel_shape = [3,3], pads = [1, 1, 1, 1], strides = [2, 2] } : (tensor<5x5x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_default_averagepool_strides
  // CHECK: [[RES:%.+]] = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : i64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<5x5x32x32xf32>) -> tensor<5x5x16x16xf32>
  // CHECK: return [[RES]] : tensor<5x5x16x16xf32>
}

// -----

/// Test the default behavior of Average Pool with non uniform padding
func @test_default_averagepool_strides_nonunifpad(%arg0 : tensor<5x5x30x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0, kernel_shape = [2,2], pads = [1, 0, 0, 0], strides = [2, 2] } : (tensor<5x5x30x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_default_averagepool_strides_nonunifpad
  // CHECK: [[RES:%.+]] = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : i64, kernel_shape = [2, 2], pads = [1, 0, 0, 0], strides = [2, 2]} : (tensor<5x5x30x32xf32>) -> tensor<5x5x15x16xf32>
  // CHECK: return [[RES]] : tensor<5x5x15x16xf32>
}

// -----

/// Test the default behavior of Average Pool with non uniform padding
func @test_default_averagepool_strides_nonunifpad_ceil(%arg0 : tensor<5x5x30x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 1, kernel_shape = [2,2], pads = [1, 0, 0, 0], strides = [2, 2] } : (tensor<5x5x30x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_default_averagepool_strides_nonunifpad_ceil
  // CHECK: [[RES:%.+]] = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 1 : i64, kernel_shape = [2, 2], pads = [1, 0, 0, 0], strides = [2, 2]} : (tensor<5x5x30x32xf32>) -> tensor<5x5x16x16xf32>
  // CHECK: return [[RES]] : tensor<5x5x16x16xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// Test the reshape op inference when constants are present.
//===----------------------------------------------------------------------===//

func @test_reshape_dynamic(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<4xi64>) -> tensor<*xf32> {
  %0 = "onnx.Reshape"(%arg0, %arg1) : (tensor<5x5x1x32xf32>, tensor<4xi64>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reshape_dynamic
  // CHECK: [[RES:%.+]] = "onnx.Reshape"(%arg0, %arg1) : (tensor<5x5x1x32xf32>, tensor<4xi64>) -> tensor<?x?x?x?xf32>
  // CHECK: return [[RES]] : tensor<?x?x?x?xf32>
}

// -----

func @test_reshape_1(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[5, 5, 16, 2]> : tensor<4xi64> } : () -> tensor<4xi64>
  %1 = "onnx.Reshape"(%arg0, %0) : (tensor<5x5x1x32xf32>, tensor<4xi64>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reshape_1
  // CHECK: [[RES:%.+]] = "onnx.Reshape"(%arg0, %0) : (tensor<5x5x1x32xf32>, tensor<4xi64>) -> tensor<5x5x16x2xf32>
  // CHECK: return [[RES]] : tensor<5x5x16x2xf32>
}

// -----

func @test_reshape_2(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[-1, 16, 2]> : tensor<3xi64> } : () -> tensor<3xi64>
  %1 = "onnx.Reshape"(%arg0, %0) : (tensor<5x5x1x32xf32>, tensor<3xi64>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reshape_2
  // CHECK: [[RES:%.+]] = "onnx.Reshape"(%arg0, %0) : (tensor<5x5x1x32xf32>, tensor<3xi64>) -> tensor<25x16x2xf32>
  // CHECK: return [[RES]] : tensor<25x16x2xf32>
}

// -----

func @test_reshape_3(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[-1, 0, 2]> : tensor<3xi64> } : () -> tensor<3xi64>
  %1 = "onnx.Reshape"(%arg0, %0) : (tensor<5x5x1x32xf32>, tensor<3xi64>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reshape_3
  // CHECK: [[RES:%.+]] = "onnx.Reshape"(%arg0, %0) : (tensor<5x5x1x32xf32>, tensor<3xi64>) -> tensor<80x5x2xf32>
  // CHECK: return [[RES]] : tensor<80x5x2xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// Test the flatten op inference.
//===----------------------------------------------------------------------===//

func @test_flatten_1(%arg0 : tensor<5x2x3x4xf32>) -> tensor<*xf32> {
  %1 = "onnx.Flatten"(%arg0) {axis = 1 : i64} : (tensor<5x2x3x4xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_flatten_1
  // CHECK: [[RES:%.+]] = "onnx.Flatten"(%arg0) {axis = 1 : i64} : (tensor<5x2x3x4xf32>) -> tensor<5x24xf32>
  // CHECK: return [[RES]] : tensor<5x24xf32>
}

//===----------------------------------------------------------------------===//
/// Test the reshape op inference when concat are present.
//===----------------------------------------------------------------------===//

func @test_concat_1(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<5x5x3x32xf32>, %arg2 : tensor<5x5x5x32xf32>) -> tensor<*xf32> {
  %1 = "onnx.Concat"(%arg0, %arg1, %arg2) { axis = 2 } : (tensor<5x5x1x32xf32>, tensor<5x5x3x32xf32>, tensor<5x5x5x32xf32>)  -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_concat_1
  // CHECK: [[RES:%.+]] = "onnx.Concat"(%arg0, %arg1, %arg2) {axis = 2 : i64} : (tensor<5x5x1x32xf32>, tensor<5x5x3x32xf32>, tensor<5x5x5x32xf32>) -> tensor<5x5x9x32xf32>
  // CHECK: return [[RES]] : tensor<5x5x9x32xf32>
}

// -----

func @test_concat_2(%arg0 : tensor<5x1x32xf32>, %arg1 : tensor<5x3x32xf32>, %arg2 : tensor<5x5x32xf32>) -> tensor<*xf32> {
  %1 = "onnx.Concat"(%arg0, %arg1, %arg2) { axis = 1 } : (tensor<5x1x32xf32>, tensor<5x3x32xf32>, tensor<5x5x32xf32>)  -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_concat_2
  // CHECK: [[RES:%.+]] = "onnx.Concat"(%arg0, %arg1, %arg2) {axis = 1 : i64} : (tensor<5x1x32xf32>, tensor<5x3x32xf32>, tensor<5x5x32xf32>) -> tensor<5x9x32xf32>
  // CHECK: return [[RES]] : tensor<5x9x32xf32>
}

// -----

func @test_concat_3(%arg0 : tensor<5x1x32xf32>, %arg1 : tensor<5x3x32xf32>, %arg2 : tensor<5x5x32xf32>) -> tensor<*xf32> {
  %1 = "onnx.Concat"(%arg0, %arg1, %arg2) { axis = -2 } : (tensor<5x1x32xf32>, tensor<5x3x32xf32>, tensor<5x5x32xf32>)  -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_concat_3
  // CHECK: [[RES:%.+]] = "onnx.Concat"(%arg0, %arg1, %arg2) {axis = 1 : i64} : (tensor<5x1x32xf32>, tensor<5x3x32xf32>, tensor<5x5x32xf32>) -> tensor<5x9x32xf32>
  // CHECK: return [[RES]] : tensor<5x9x32xf32>
}

// -----

func @test_rnn_all_results(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : i64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (tensor<*xf32>, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_rnn_all_results
  // CHECK: %{{.*}}, [[RES:%.+]] = "onnx.RNN"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : i64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (tensor<4x1x3x3xf32>, tensor<1x3x3xf32>)
  // CHECK: return [[RES]] : tensor<1x3x3xf32>
}

// -----

func @test_rnn_no_results(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> () {
  %cst = constant unit
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : i64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (none, none)
  return

  // CHECK-LABEL: test_rnn_no_results
  // CHECK: %{{.*}}, [[RES:%.+]] = "onnx.RNN"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : i64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (none, none)
  // CHECK: return
}

// -----

func @test_rnn_missing_first_result(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : i64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_rnn_missing_first_result
  // CHECK: %{{.*}}, [[RES:%.+]] = "onnx.RNN"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : i64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (none, tensor<1x3x3xf32>)
  // CHECK: return [[RES]] : tensor<1x3x3xf32>
}

// -----

func @test_rnn_missing_trailing_result(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> () {
  %cst = constant unit
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : i64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (tensor<*xf32>, none)
  return

  // CHECK-LABEL: test_rnn_missing_trailing_result
  // CHECK: %{{.*}}, [[RES:%.+]] = "onnx.RNN"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : i64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (tensor<4x1x3x3xf32>, none)
  // CHECK: return
}

// -----

func @test_rnn_all_results_no_hidden_size(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %cst, %cst, %cst) : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (tensor<*xf32>, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_rnn_all_results_no_hidden_size
  // CHECK: %{{.*}}, [[RES:%.+]] = "onnx.RNN"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : i64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (tensor<4x1x3x3xf32>, tensor<1x3x3xf32>)
  // CHECK: return [[RES]] : tensor<1x3x3xf32>
}

// -----

func @test_rnn_all_results_unknown_dims(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %cst, %cst, %cst) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, none, none, none) -> (tensor<*xf32>, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_rnn_all_results_unknown_dims
  // CHECK: %{{.*}}, [[RES:%.+]] = "onnx.RNN"(%arg0, %arg1, %arg2, %cst, %cst, %cst) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, none, none, none) -> (tensor<?x1x?x?xf32>, tensor<1x?x?xf32>)
  // CHECK: return [[RES]] : tensor<1x?x?xf32>
}

// -----

func @test_gru_all_results(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : i64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (tensor<*xf32>, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_gru_all_results
  // CHECK: %{{.*}}, [[RES:%.+]] = "onnx.GRU"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : i64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (tensor<4x1x3x3xf32>, tensor<1x3x3xf32>)
  // CHECK: return [[RES]] : tensor<1x3x3xf32>
}

// -----

func @test_gru_no_results(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> () {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : i64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (none, none)
  return

  // CHECK-LABEL: test_gru_no_results
  // CHECK: %{{.*}}, [[RES:%.+]] = "onnx.GRU"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : i64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (none, none)
  // CHECK: return
}

// -----

func @test_gru_missing_first_result(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : i64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_gru_missing_first_result
  // CHECK: %{{.*}}, [[RES:%.+]] = "onnx.GRU"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : i64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (none, tensor<1x3x3xf32>)
  // CHECK: return [[RES]] : tensor<1x3x3xf32>
}

// -----

func @test_gru_missing_trailing_result(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> () {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : i64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (tensor<*xf32>, none)
  return

  // CHECK-LABEL: test_gru_missing_trailing_result
  // CHECK: %{{.*}}, [[RES:%.+]] = "onnx.GRU"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : i64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (tensor<4x1x3x3xf32>, none)
  // CHECK: return
}

// -----

func @test_gru_all_results_no_hidden_size(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %cst, %cst, %cst) : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (tensor<*xf32>, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_gru_all_results_no_hidden_size
  // CHECK: %{{.*}}, [[RES:%.+]] = "onnx.GRU"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : i64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (tensor<4x1x3x3xf32>, tensor<1x3x3xf32>)
  // CHECK: return [[RES]] : tensor<1x3x3xf32>
}

// -----

func @test_gru_all_results_unknown_dims(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %cst, %cst, %cst) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, none, none, none) -> (tensor<*xf32>, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_gru_all_results_unknown_dims
  // CHECK: %{{.*}}, [[RES:%.+]] = "onnx.GRU"(%arg0, %arg1, %arg2, %cst, %cst, %cst) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, none, none, none) -> (tensor<?x1x?x?xf32>, tensor<1x?x?xf32>)
  // CHECK: return [[RES]] : tensor<1x?x?xf32>
}

// -----

func @test_lstm_all_results(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : i64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_lstm_all_results
  // CHECK: %{{.*}}, [[RES:%.+]], %{{.*}} = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : i64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (tensor<4x1x3x3xf32>, tensor<1x3x3xf32>, tensor<1x3x3xf32>)
  // CHECK: return [[RES]] : tensor<1x3x3xf32>
}

// -----

func @test_lstm_no_results(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> () {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : i64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (none, none, none)
  return

  // CHECK-LABEL: test_lstm_no_results
  // CHECK: %{{.*}}, [[RES:%.+]], %{{.*}} = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : i64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (none, none, none)
  // CHECK: return
}

// -----

func @test_lstm_missing_first_result(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : i64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (none, tensor<*xf32>, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_lstm_missing_first_result
  // CHECK: %{{.*}}, [[RES:%.+]], %{{.*}} = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : i64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (none, tensor<1x3x3xf32>, tensor<1x3x3xf32>)
  // CHECK: return [[RES]] : tensor<1x3x3xf32>
}

// -----

func @test_lstm_missing_trailing_result(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : i64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (tensor<*xf32>, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_lstm_missing_trailing_result
  // CHECK: %{{.*}}, [[RES:%.+]], %{{.*}} = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : i64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (tensor<4x1x3x3xf32>, tensor<1x3x3xf32>, none)
  // CHECK: return [[RES]] : tensor<1x3x3xf32>
}

// -----

func @test_lstm_all_results_no_hidden_size(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_lstm_all_results_no_hidden_size
  // CHECK: %{{.*}}, [[RES:%.+]], %{{.*}} = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : i64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (tensor<4x1x3x3xf32>, tensor<1x3x3xf32>, tensor<1x3x3xf32>)
  // CHECK: return [[RES]] : tensor<1x3x3xf32>
}

// -----

func @test_lstm_all_results_unknown_dims(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, none, none, none, none, none) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_lstm_all_results_unknown_dims
  // CHECK: %{{.*}}, [[RES:%.+]], %{{.*}} = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, none, none, none, none, none) -> (tensor<?x1x?x?xf32>, tensor<1x?x?xf32>, tensor<1x?x?xf32>)
  // CHECK: return [[RES]] : tensor<1x?x?xf32>
}

// -----

func @test_split_1(%arg0 : tensor<16x32x64xf32>) -> tensor<*xf32> {
  %0, %1 = "onnx.Split"(%arg0) { axis = 1 } : (tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_split_1
  // CHECK: [[RES:%.+]]:2 = "onnx.Split"(%arg0) {axis = 1 : i64, split = [16, 16]} : (tensor<16x32x64xf32>) -> (tensor<16x16x64xf32>, tensor<16x16x64xf32>)
  // CHECK: return [[RES]]#0 : tensor<16x16x64xf32>
}

// -----

func @test_split_2(%arg0 : tensor<16x32x64xf32>) -> tensor<*xf32> {
  %0, %1 = "onnx.Split"(%arg0) { axis = -2 } : (tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_split_2
  // CHECK: [[RES:%.+]]:2 = "onnx.Split"(%arg0) {axis = 1 : i64, split = [16, 16]} : (tensor<16x32x64xf32>) -> (tensor<16x16x64xf32>, tensor<16x16x64xf32>)
  // CHECK: return [[RES]]#0 : tensor<16x16x64xf32>
}

// -----

func @test_split_3(%arg0 : tensor<16x32x64xf32>) -> tensor<*xf32> {
  %0, %1 = "onnx.Split"(%arg0) { axis = 1, split = [2, 30]} : (tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_split_3
  // CHECK: [[RES:%.+]]:2 = "onnx.Split"(%arg0) {axis = 1 : i64, split = [2, 30]} : (tensor<16x32x64xf32>) -> (tensor<16x2x64xf32>, tensor<16x30x64xf32>)
  // CHECK: return [[RES]]#0 : tensor<16x2x64xf32>
}

// -----

func @test_squeeze(%arg0 : tensor<16x1x32x1x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.Squeeze"(%arg0) { axes = [1]} : (tensor<16x1x32x1x64xf32>) -> (tensor<*xf32>)
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_squeeze
  // CHECK: [[RES:%.+]] = "onnx.Squeeze"(%arg0) {axes = [1]} : (tensor<16x1x32x1x64xf32>) -> tensor<16x32x1x64xf32>
  // CHECK: return [[RES]] : tensor<16x32x1x64xf32>
}

// -----

func @test_squeeze_negative_axis(%arg0 : tensor<16x1x32x1x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.Squeeze"(%arg0) { axes = [-2]} : (tensor<16x1x32x1x64xf32>) -> (tensor<*xf32>)
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_squeeze_negative_axis
  // CHECK: [[RES:%.+]] = "onnx.Squeeze"(%arg0) {axes = [3]} : (tensor<16x1x32x1x64xf32>) -> tensor<16x1x32x64xf32>
  // CHECK: return [[RES]] : tensor<16x1x32x64xf32>
}

// -----

func @test_squeeze_mix(%arg0 : tensor<16x1x32x1x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.Squeeze"(%arg0) { axes = [1, -2]} : (tensor<16x1x32x1x64xf32>) -> (tensor<*xf32>)
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_squeeze_mix
  // CHECK: [[RES:%.+]] = "onnx.Squeeze"(%arg0) {axes = [1, 3]} : (tensor<16x1x32x1x64xf32>) -> tensor<16x32x64xf32>
  // CHECK: return [[RES]] : tensor<16x32x64xf32>
}

//===----------------------------------------------------------------------===//
/// Test the cast op inference.
//===----------------------------------------------------------------------===//

func @test_cast_1(%arg0 : tensor<2x3x4xf32>) -> tensor<*xf32> {
  %1 = "onnx.Cast"(%arg0) {to = 1} : (tensor<2x3x4xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_cast_1
  // CHECK: [[RES:%.+]] = "onnx.Cast"(%arg0) {to = 1 : i64} : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  // CHECK: return [[RES]] : tensor<2x3x4xf32>
}

func @test_cast_2(%arg0 : tensor<2x3x4xf32>) -> tensor<*xui8> {
  %1 = "onnx.Cast"(%arg0) {to = 2} : (tensor<2x3x4xf32>) -> tensor<*xui8>
  "std.return"(%1) : (tensor<*xui8>) -> ()

  // CHECK-LABEL: test_cast_2
  // CHECK: [[RES:%.+]] = "onnx.Cast"(%arg0) {to = 2 : i64} : (tensor<2x3x4xf32>) -> tensor<2x3x4xui8>
  // CHECK: return [[RES]] : tensor<2x3x4xui8>
}

func @test_cast_3(%arg0 : tensor<2x3x4xf32>) -> tensor<*xi8> {
  %1 = "onnx.Cast"(%arg0) {to = 3} : (tensor<2x3x4xf32>) -> tensor<*xi8>
  "std.return"(%1) : (tensor<*xi8>) -> ()

  // CHECK-LABEL: test_cast_3
  // CHECK: [[RES:%.+]] = "onnx.Cast"(%arg0) {to = 3 : i64} : (tensor<2x3x4xf32>) -> tensor<2x3x4xi8>
  // CHECK: return [[RES]] : tensor<2x3x4xi8>
}

func @test_cast_10(%arg0 : tensor<2x3x4xf32>) -> tensor<*xf16> {
  %1 = "onnx.Cast"(%arg0) {to = 10} : (tensor<2x3x4xf32>) -> tensor<*xf16>
  "std.return"(%1) : (tensor<*xf16>) -> ()

  // CHECK-LABEL: test_cast_10
  // CHECK: [[RES:%.+]] = "onnx.Cast"(%arg0) {to = 10 : i64} : (tensor<2x3x4xf32>) -> tensor<2x3x4xf16>
  // CHECK: return [[RES]] : tensor<2x3x4xf16>
}

//===----------------------------------------------------------------------===//
/// Test the quantization op inferences.
//===----------------------------------------------------------------------===//

// TOFIX
// This test case is commented out because the #1 output should be tensor<f32>
// but tensor<i8> is generated
func @test_dyn_quantize_linear_1(%arg0 : tensor<5x2x3x4xf32>) -> tensor<*xui8> {
 %1:3 = "onnx.DynamicQuantizeLinear"(%arg0) {} : (tensor<5x2x3x4xf32>) -> (tensor<*xui8>, tensor<*xf32>, tensor<*xui8>)
 "std.return"(%1#0) {} : (tensor<*xui8>) -> ()

 // CHECK-LABEL: test_dyn_quantize_linear_1
 // CHECK: [[RES:%.+]], {{.*}}, {{.*}} = "onnx.DynamicQuantizeLinear"(%arg0) : (tensor<5x2x3x4xf32>) -> (tensor<5x2x3x4xui8>, tensor<f32>, tensor<ui8>)
 // CHECK: return [[RES]] : tensor<5x2x3x4xui8>
}

func @test_quantize_linear_1(%arg0 : tensor<5x2x3x4xf32>, %arg1 : tensor<f32>, %arg2 : tensor<i8>) -> tensor<*xi8> {
  %1 = "onnx.QuantizeLinear"(%arg0, %arg1, %arg2) {} : (tensor<5x2x3x4xf32>, tensor<f32>, tensor<i8>) -> tensor<*xi8>
  "std.return"(%1) {} : (tensor<*xi8>) -> ()

  // CHECK-LABEL: test_quantize_linear_1
  // CHECK: [[RES:%.+]] = "onnx.QuantizeLinear"(%arg0, %arg1, %arg2) : (tensor<5x2x3x4xf32>, tensor<f32>, tensor<i8>) -> tensor<5x2x3x4xi8>
  // CHECK: return [[RES]] : tensor<5x2x3x4xi8>
}

func @test_dequantize_linear_1(%arg0 : tensor<5x2x3x4xi8>, %arg1 : tensor<f32>, %arg2 : tensor<i8>) -> tensor<*xf32> {
  %1 = "onnx.DequantizeLinear"(%arg0, %arg1, %arg2) {} : (tensor<5x2x3x4xi8>, tensor<f32>, tensor<i8>) -> tensor<*xf32>
  "std.return"(%1) {} : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_dequantize_linear_1
  // CHECK: [[RES:%.+]] = "onnx.DequantizeLinear"(%arg0, %arg1, %arg2) : (tensor<5x2x3x4xi8>, tensor<f32>, tensor<i8>) -> tensor<5x2x3x4xf32>
  // CHECK: return [[RES]] : tensor<5x2x3x4xf32>
}

//===----------------------------------------------------------------------===//
/// Test shape inference for ConvInteger operation and all its attributes.
//===----------------------------------------------------------------------===//

/// Default and required attributes for 1-D convolution.

func @test_convinteger_0(%arg0 : tensor<1x2x32xi8>, %arg1 : tensor<5x2x6xi8>, %arg2 : tensor<i8>, %arg3 : tensor<i8>) -> tensor<*xi32> {
  %0 = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", group = 1 : i64} : (tensor<1x2x32xi8>, tensor<5x2x6xi8>, tensor<i8>, tensor<i8>) -> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_convinteger_0
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", dilations = [1], group = 1 : i64, kernel_shape = [6], pads = [0, 0], strides = [1]} : (tensor<1x2x32xi8>, tensor<5x2x6xi8>, tensor<i8>, tensor<i8>) -> tensor<1x5x27xi32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x27xi32>
}

/// Default and required attributes.

func @test_convinteger_1(%arg0 : tensor<1x2x32x64xi8>, %arg1 : tensor<5x2x6x7xi8>, %arg2 : tensor<i8>, %arg3 : tensor<i8>) -> tensor<*xi32> {
  %0 = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", group = 1 : i64} : (tensor<1x2x32x64xi8>, tensor<5x2x6x7xi8>, tensor<i8>, tensor<i8>) -> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_convinteger_1
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : i64, kernel_shape = [6, 7], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x7xi8>, tensor<i8>, tensor<i8>) -> tensor<1x5x27x58xi32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x27x58xi32>
}

/// kernel_shape attribute.

func @test_convinteger_2(%arg0 : tensor<1x2x32x64xi8>, %arg1 : tensor<5x2x6x7xi8>, %arg2 : tensor<i8>, %arg3 : tensor<i8>) -> tensor<*xi32> {
  %0 = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", group = 1 : i64, kernel_shape = [8, 9]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x7xi8>, tensor<i8>, tensor<i8>) -> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_convinteger_2
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : i64, kernel_shape = [8, 9], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x7xi8>, tensor<i8>, tensor<i8>) -> tensor<1x5x25x56xi32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x25x56xi32>
}

/// pads attribute.
/// Use pads to make output size equal to input size by adding K - 1 to the result.

func @test_convinteger_3(%arg0 : tensor<1x2x32x64xi8>, %arg1 : tensor<5x2x6x10xi8>, %arg2 : tensor<i8>, %arg3 : tensor<i8>) -> tensor<*xi32> {
  %0 = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", group = 1 : i64, pads = [2, 4, 3, 5]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x10xi8>, tensor<i8>, tensor<i8>) -> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_convinteger_3
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : i64, kernel_shape = [6, 10], pads = [2, 4, 3, 5], strides = [1, 1]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x10xi8>, tensor<i8>, tensor<i8>) -> tensor<1x5x32x64xi32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x32x64xi32>
}

/// auto_pad set to SAME_UPPER and SAME_LOWER.

func @test_convinteger_4(%arg0 : tensor<1x2x32x64xi8>, %arg1 : tensor<5x2x6x10xi8>, %arg2 : tensor<i8>, %arg3 : tensor<i8>) -> tensor<*xi32> {
  %0 = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "SAME_UPPER", group = 1 : i64} : (tensor<1x2x32x64xi8>, tensor<5x2x6x10xi8>, tensor<i8>, tensor<i8>) -> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_convinteger_4
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : i64, kernel_shape = [6, 10], pads = [2, 4, 3, 5], strides = [1, 1]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x10xi8>, tensor<i8>, tensor<i8>) -> tensor<1x5x32x64xi32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x32x64xi32>
}

func @test_convinteger_5(%arg0 : tensor<1x2x32x64xi8>, %arg1 : tensor<5x2x6x10xi8>, %arg2 : tensor<i8>, %arg3 : tensor<i8>) -> tensor<*xi32> {
  %0 = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "SAME_LOWER", group = 1 : i64} : (tensor<1x2x32x64xi8>, tensor<5x2x6x10xi8>, tensor<i8>, tensor<i8>) -> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_convinteger_5
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : i64, kernel_shape = [6, 10], pads = [3, 5, 2, 4], strides = [1, 1]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x10xi8>, tensor<i8>, tensor<i8>) -> tensor<1x5x32x64xi32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x32x64xi32>
}

/// auto_pad set to VALID.

func @test_convinteger_6(%arg0 : tensor<1x2x32x64xi8>, %arg1 : tensor<5x2x6x10xi8>, %arg2 : tensor<i8>, %arg3 : tensor<i8>) -> tensor<*xi32> {
  %0 = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "VALID", group = 1 : i64} : (tensor<1x2x32x64xi8>, tensor<5x2x6x10xi8>, tensor<i8>, tensor<i8>) -> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_convinteger_6
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : i64, kernel_shape = [6, 10], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x10xi8>, tensor<i8>, tensor<i8>) -> tensor<1x5x27x55xi32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x27x55xi32>
}

/// With strides attribute.

func @test_convinteger_7(%arg0 : tensor<1x2x32x64xi8>, %arg1 : tensor<5x2x6x7xi8>, %arg2 : tensor<i8>, %arg3 : tensor<i8>) -> tensor<*xi32> {
  %0 = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", group = 1 : i64, strides = [2, 3]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x7xi8>, tensor<i8>, tensor<i8>) -> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_convinteger_7
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : i64, kernel_shape = [6, 7], pads = [0, 0, 0, 0], strides = [2, 3]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x7xi8>, tensor<i8>, tensor<i8>) -> tensor<1x5x14x20xi32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x14x20xi32>
}

/// auto_pad set to SAME_UPPER with strides attribute.
/// The auto_pad will pas as if stride is equal to 1.

func @test_convinteger_8(%arg0 : tensor<1x2x32x64xi8>, %arg1 : tensor<5x2x6x7xi8>, %arg2 : tensor<i8>, %arg3 : tensor<i8>) -> tensor<*xi32> {
  %0 = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "SAME_UPPER", group = 1 : i64, strides = [2, 3]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x7xi8>, tensor<i8>, tensor<i8>) -> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_convinteger_8
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : i64, kernel_shape = [6, 7], pads = [2, 3, 2, 3], strides = [2, 3]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x7xi8>, tensor<i8>, tensor<i8>) -> tensor<1x5x16x22xi32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x16x22xi32>
}

/// dilations attribute.

func @test_convinteger_9(%arg0 : tensor<1x2x32x64xi8>, %arg1 : tensor<5x2x6x7xi8>, %arg2 : tensor<i8>, %arg3 : tensor<i8>) -> tensor<*xi32> {
  %0 = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", group = 1 : i64, dilations = [2, 3]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x7xi8>, tensor<i8>, tensor<i8>) -> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_convinteger_9
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", dilations = [2, 3], group = 1 : i64, kernel_shape = [6, 7], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x7xi8>, tensor<i8>, tensor<i8>) -> tensor<1x5x22x46xi32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x22x46xi32>
}

/// dilations attribute with stride.

func @test_convinteger_10(%arg0 : tensor<1x2x32x64xi8>, %arg1 : tensor<5x2x6x7xi8>, %arg2 : tensor<i8>, %arg3 : tensor<i8>) -> tensor<*xi32> {
  %0 = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", group = 1 : i64, dilations = [2, 3], strides = [2, 2]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x7xi8>, tensor<i8>, tensor<i8>) -> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_convinteger_10
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", dilations = [2, 3], group = 1 : i64, kernel_shape = [6, 7], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x7xi8>, tensor<i8>, tensor<i8>) -> tensor<1x5x11x23xi32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x11x23xi32>
}

/// dilations attribute with auto_pad set to SAME_UPPER.

func @test_convinteger_11(%arg0 : tensor<1x2x32x64xi8>, %arg1 : tensor<5x2x6x7xi8>, %arg2 : tensor<i8>, %arg3 : tensor<i8>) -> tensor<*xi32> {
  %0 = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "SAME_UPPER", group = 1 : i64, dilations = [2, 3]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x7xi8>, tensor<i8>, tensor<i8>) -> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_convinteger_11
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", dilations = [2, 3], group = 1 : i64, kernel_shape = [6, 7], pads = [5, 9, 5, 9], strides = [1, 1]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x7xi8>, tensor<i8>, tensor<i8>) -> tensor<1x5x32x64xi32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x32x64xi32>
}

// -----

func @test_shape(%arg0: tensor<?x3x2xf32>) -> tensor<*xi64> {
  %0 = "onnx.Shape"(%arg0) : (tensor<?x3x2xf32>) -> tensor<*xi64>
  return %0 : tensor<*xi64>

  // CHECK-LABEL: test_shape
  // CHECK: [[RES:%.+]] = "onnx.Shape"(%arg0) : (tensor<?x3x2xf32>) -> tensor<3xi64>
  // CHECK: return [[RES]] : tensor<3xi64>
}

// -----

func @test_tile_dynamic(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<4xi64>) -> tensor<*xf32> {
  %0 = "onnx.Tile"(%arg0, %arg1) : (tensor<5x5x1x32xf32>, tensor<4xi64>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_tile_dynamic
  // CHECK: [[RES:%.+]] = "onnx.Tile"(%arg0, %arg1) : (tensor<5x5x1x32xf32>, tensor<4xi64>) -> tensor<?x?x?x?xf32>
  // CHECK: return [[RES]] : tensor<?x?x?x?xf32>
}

// -----

func @test_tile_constant(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[5, 5, 16, 2]> : tensor<4xi64> } : () -> tensor<4xi64>
  %1 = "onnx.Tile"(%arg0, %0) : (tensor<5x5x1x32xf32>, tensor<4xi64>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_tile_constant
  // CHECK: [[RES:%.+]] = "onnx.Tile"(%arg0, %0) : (tensor<5x5x1x32xf32>, tensor<4xi64>) -> tensor<25x25x16x64xf32>
  // CHECK: return [[RES]] : tensor<25x25x16x64xf32>
}

// -----

func @test_gather_axis0(%arg0 : tensor<3x3xf32>, %arg1 : tensor<1x2xi64>) -> tensor<*xf32> {
  %0 = "onnx.Gather"(%arg0, %arg1) {axis = 0} : (tensor<3x3xf32>, tensor<1x2xi64>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_gather_axis0
  // CHECK: [[RES:%.+]] = "onnx.Gather"(%arg0, %arg1) {axis = 0 : i64} : (tensor<3x3xf32>, tensor<1x2xi64>) -> tensor<1x2x3xf32>
  // CHECK: return [[RES]] : tensor<1x2x3xf32>
}

// -----

func @test_gather_axis1(%arg0 : tensor<3x3xf32>, %arg1 : tensor<1x2xi64>) -> tensor<*xf32> {
  %0 = "onnx.Gather"(%arg0, %arg1) {axis = 1} : (tensor<3x3xf32>, tensor<1x2xi64>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_gather_axis1
  // CHECK: [[RES:%.+]] = "onnx.Gather"(%arg0, %arg1) {axis = 1 : i64} : (tensor<3x3xf32>, tensor<1x2xi64>) -> tensor<3x1x2xf32>
  // CHECK: return [[RES]] : tensor<3x1x2xf32>
}

// -----

func @test_gather_negative_axis(%arg0 : tensor<3x3xf32>, %arg1 : tensor<1x2xi64>) -> tensor<*xf32> {
  %0 = "onnx.Gather"(%arg0, %arg1) {axis = -1} : (tensor<3x3xf32>, tensor<1x2xi64>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_gather_negative_axis
  // CHECK: [[RES:%.+]] = "onnx.Gather"(%arg0, %arg1) {axis = 1 : i64} : (tensor<3x3xf32>, tensor<1x2xi64>) -> tensor<3x1x2xf32>
  // CHECK: return [[RES]] : tensor<3x1x2xf32>
}

// -----

func @test_constant_of_shape_empty_tensor(%arg0 : tensor<0xi64>) -> tensor<*xf32> {
  %0 = "onnx.ConstantOfShape"(%arg0) : (tensor<0xi64>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_constant_of_shape_empty_tensor
  // CHECK: [[RES:%.+]] = "onnx.ConstantOfShape"(%arg0) {value = dense<0.000000e+00> : tensor<1xf32>} : (tensor<0xi64>) -> tensor<f32>
  // CHECK: return [[RES]] : tensor<f32>
}

// -----

func @test_constant_of_shape(%arg0 : tensor<3xi64>) -> tensor<*xf32> {
  %0 = "onnx.ConstantOfShape"(%arg0) {value = dense<[1.0]> : tensor<1xf32>} : (tensor<3xi64>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_constant_of_shape
  // CHECK: [[RES:%.+]] = "onnx.ConstantOfShape"(%arg0) {value = dense<1.000000e+00> : tensor<1xf32>} : (tensor<3xi64>) -> tensor<?x?x?xf32>
  // CHECK: return [[RES]] : tensor<?x?x?xf32>
}

// -----

func @test_constant_of_shape_constant() -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[3, 4, 5]> : tensor<3xi64> } : () -> tensor<3xi64>
  %1 = "onnx.ConstantOfShape"(%0) {value = dense<[1.0]> : tensor<1xf32>} : (tensor<3xi64>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_constant_of_shape_constant
  // CHECK: [[CONSTANT:%.+]] = "onnx.Constant"() {value = dense<[3, 4, 5]> : tensor<3xi64>} : () -> tensor<3xi64>
  // CHECK: [[RES:%.+]] = "onnx.ConstantOfShape"([[CONSTANT]]) {value = dense<1.000000e+00> : tensor<1xf32>} : (tensor<3xi64>) -> tensor<3x4x5xf32>
  // CHECK: return [[RES]] : tensor<3x4x5xf32>
}

// -----

func @test_slice(%arg0 : tensor<2x4xf32>, %arg1: tensor<2xi64>, %arg2: tensor<2xi64>, %arg3: tensor<2xi64>, %arg4: tensor<2xi64>) -> tensor<*xf32> {
  %1 = "onnx.Slice"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_slice
  // CHECK: [[RES:%.+]] = "onnx.Slice"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<?x?xf32>
  // CHECK: return [[RES:%.+]] : tensor<?x?xf32>
}

// -----

func @test_slice_constant_default_axes(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = constant unit
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, none, tensor<2xi64>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_slice_constant_default_axes
  // CHECK: [[AXES:%.+]] = constant unit
  // CHECK: [[STARTS:%.+]] = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64>} : () -> tensor<2xi64> 
  // CHECK: [[ENDS:%.+]] = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  // CHECK: [[STEPS:%.+]] = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  // CHECK: [[RES:%.+]] = "onnx.Slice"(%arg0, [[STARTS]], [[ENDS]], [[AXES]], [[STEPS]]) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, none, tensor<2xi64>) -> tensor<1x2xf32>
  // CHECK: return [[RES]] : tensor<1x2xf32>
}

// -----

func @test_slice_constant_default_steps(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = constant unit
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, none) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_slice_constant_default_steps
  // CHECK: [[AXES:%.+]] = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64>} : () -> tensor<2xi64>
  // CHECK: [[STARTS:%.+]] = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64>} : () -> tensor<2xi64> 
  // CHECK: [[ENDS:%.+]] = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  // CHECK: [[STEPS:%.+]] = constant unit
  // CHECK: [[RES:%.+]] = "onnx.Slice"(%arg0, [[STARTS]], [[ENDS]], [[AXES]], [[STEPS]]) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, none) -> tensor<1x3xf32>
  // CHECK: return [[RES]] : tensor<1x3xf32>
}

// -----

func @test_slice_all_constant(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_slice_all_constant
  // CHECK: [[AXES:%.+]] = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64>} : () -> tensor<2xi64>
  // CHECK: [[STARTS:%.+]] = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64>} : () -> tensor<2xi64> 
  // CHECK: [[ENDS:%.+]] = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  // CHECK: [[STEPS:%.+]] = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  // CHECK: [[RES:%.+]] = "onnx.Slice"(%arg0, [[STARTS]], [[ENDS]], [[AXES]], [[STEPS]]) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x2xf32>
  // CHECK: return [[RES]] : tensor<1x2xf32>
}

// -----

func @test_slice_all_constant_negative(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, -1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[2, -1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_slice_all_constant_negative
  // CHECK: [[AXES:%.+]] = "onnx.Constant"() {value = dense<[0, -1]> : tensor<2xi64>} : () -> tensor<2xi64>
  // CHECK: [[STARTS:%.+]] = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64>} : () -> tensor<2xi64> 
  // CHECK: [[ENDS:%.+]] = "onnx.Constant"() {value = dense<[2, -1]> : tensor<2xi64>} : () -> tensor<2xi64>
  // CHECK: [[STEPS:%.+]] = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  // CHECK: [[RES:%.+]] = "onnx.Slice"(%arg0, [[STARTS]], [[ENDS]], [[AXES]], [[STEPS]]) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x2xf32>
  // CHECK: return [[RES]] : tensor<1x2xf32>
}

// -----

func @test_slice_all_constant_end_outofbound(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[5, 3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_slice_all_constant_end_outofbound
  // CHECK: [[AXES:%.+]] = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64>} : () -> tensor<2xi64>
  // CHECK: [[STARTS:%.+]] = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64>} : () -> tensor<2xi64> 
  // CHECK: [[ENDS:%.+]] = "onnx.Constant"() {value = dense<[5, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  // CHECK: [[STEPS:%.+]] = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  // CHECK: [[RES:%.+]] = "onnx.Slice"(%arg0, [[STARTS]], [[ENDS]], [[AXES]], [[STEPS]]) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x2xf32>
  // CHECK: return [[RES]] : tensor<1x2xf32>
}

// -----

func @test_slice_all_constant_negative_steps(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, -2]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_slice_all_constant_negative_steps
  // CHECK: [[AXES:%.+]] = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64>} : () -> tensor<2xi64>
  // CHECK: [[STARTS:%.+]] = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64>} : () -> tensor<2xi64> 
  // CHECK: [[ENDS:%.+]] = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  // CHECK: [[STEPS:%.+]] = "onnx.Constant"() {value = dense<[1, -2]> : tensor<2xi64>} : () -> tensor<2xi64>
  // CHECK: [[RES:%.+]] = "onnx.Slice"(%arg0, [[STARTS]], [[ENDS]], [[AXES]], [[STEPS]]) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x2xf32>
  // CHECK: return [[RES]] : tensor<1x2xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// Test the shape inferencing for the scaler operation.
//===----------------------------------------------------------------------===//
func @test_scaler_no_scale_int(%arg0: tensor<3xi32>) -> tensor<*xf32> {
  %0 = "onnx.Scaler"(%arg0) {offset = [1986.99939 : f32, 0.99999988 : f32, 0.999999701 : f32]} : (tensor<3xi32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_scaler_no_scale_int
  // CHECK: [[RES_ATTR:%.+]] = "onnx.Scaler"(%arg0) {offset = [1986.99939 : f32, 0.99999988 : f32, 0.999999701 : f32]} : (tensor<3xi32>) -> tensor<3xf32>
  // CHECK: return [[RES_ATTR]] : tensor<3xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// Test shape inference for Pow.
//===----------------------------------------------------------------------===//

func @test_pow(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<f32>) -> tensor<*xf32> {
  %0 = "onnx.Pow"(%arg0, %arg1) : (tensor<1x2x3x4xf32>, tensor<f32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_pow
  // CHECK: [[RES:%.+]] = "onnx.Pow"(%arg0, %arg1) : (tensor<1x2x3x4xf32>, tensor<f32>) -> tensor<1x2x3x4xf32>
  // CHECK: return [[RES]] : tensor<1x2x3x4xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// Test shape inference for Erf.
//===----------------------------------------------------------------------===//

func @test_erf(%arg0: tensor<1x2x3x4xf32>) -> tensor<*xf32> {
  %0 = "onnx.Erf"(%arg0) : (tensor<1x2x3x4xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_erf
  // CHECK: [[RES:%.+]] = "onnx.Erf"(%arg0) : (tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
  // CHECK: return [[RES]] : tensor<1x2x3x4xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// Test shape inference for Expand.
//===----------------------------------------------------------------------===//

func @test_expand_with_constant(%arg0 : tensor<2x1x6x1xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[7, 1, 5]> : tensor<3xi64> } : () -> tensor<3xi64>
  %1 = "onnx.Expand"(%arg0, %0) : (tensor<2x1x6x1xf32>, tensor<3xi64>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_expand_with_constant
  // CHECK: [[RES:%.+]] = "onnx.Expand"(%arg0, %0) : (tensor<2x1x6x1xf32>, tensor<3xi64>) -> tensor<2x7x6x5xf32>
  // CHECK: return [[RES]] : tensor<2x7x6x5xf32>
}

// -----

func @test_expand_with_shape(%arg0 : tensor<2x1x6x1xf32>, %arg1: tensor<6x2xf32>) -> tensor<*xf32> {
  %0 = "onnx.Shape"(%arg1) : (tensor<6x2xf32>) -> tensor<*xi64>
  %1 = "onnx.Expand"(%arg0, %0) : (tensor<2x1x6x1xf32>, tensor<*xi64>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_expand_with_shape
  // CHECK: [[SHAPE:%.+]] = "onnx.Shape"(%arg1) : (tensor<6x2xf32>) -> tensor<2xi64>
  // CHECK: [[RES:%.+]] = "onnx.Expand"(%arg0, [[SHAPE]]) : (tensor<2x1x6x1xf32>, tensor<2xi64>) -> tensor<2x1x6x2xf32>
  // CHECK: return [[RES]] : tensor<2x1x6x2xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// Test shape inference for ReduceMean.
//===----------------------------------------------------------------------===//

func @test_reduce_mean_1(%arg0: tensor<1x2x3x4xf32>) -> tensor<*xf32> {
  %0 = "onnx.ReduceMean"(%arg0) {axes = [-1], keepdims = 1 : i64} : (tensor<1x2x3x4xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reduce_mean_1
  // CHECK: [[RES:%.+]] = "onnx.ReduceMean"(%arg0) {axes = [-1], keepdims = 1 : i64} : (tensor<1x2x3x4xf32>) -> tensor<1x2x3x1xf32>
  // CHECK: return [[RES]] : tensor<1x2x3x1xf32>
}

// -----

func @test_reduce_mean_2(%arg0: tensor<1x2x3x4xf32>) -> tensor<*xf32> {
  %0 = "onnx.ReduceMean"(%arg0) {axes = [2], keepdims = 1 : i64} : (tensor<1x2x3x4xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reduce_mean_2
  // CHECK: [[RES:%.+]] = "onnx.ReduceMean"(%arg0) {axes = [2], keepdims = 1 : i64} : (tensor<1x2x3x4xf32>) -> tensor<1x2x1x4xf32>
  // CHECK: return [[RES]] : tensor<1x2x1x4xf32>
}

// -----

func @test_reduce_mean_3(%arg0: tensor<1x2x3x4xf32>) -> tensor<*xf32> {
  %0 = "onnx.ReduceMean"(%arg0) {axes = [-1], keepdims = 0 : i64} : (tensor<1x2x3x4xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reduce_mean_3
  // CHECK: [[RES:%.+]] = "onnx.ReduceMean"(%arg0) {axes = [-1], keepdims = 0 : i64} : (tensor<1x2x3x4xf32>) -> tensor<1x2x3xf32>
  // CHECK: return [[RES]] : tensor<1x2x3xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// Test shape inference for Dropout.
//===----------------------------------------------------------------------===//

func @test_dropout(%arg0: tensor<1x2x3x4xf32>) -> (tensor<*xf32>, tensor<*xi1>) {
  %output, %mask = "onnx.Dropout"(%arg0) {ratio =  1.000000e-01 : f32} : (tensor<1x2x3x4xf32>) -> (tensor<*xf32>, tensor<*xi1>)
  "std.return"(%output, %mask) : (tensor<*xf32>, tensor<*xi1>) -> ()

  // CHECK-LABEL: test_dropout
  // CHECK: [[RES:%.+]], [[MASK:%.+]] = "onnx.Dropout"(%arg0) {ratio =  1.000000e-01 : f32} : (tensor<1x2x3x4xf32>) -> (tensor<1x2x3x4xf32>, tensor<1x2x3x4xi1>)
  // CHECK: return [[RES]], [[MASK]] : tensor<1x2x3x4xf32>, tensor<1x2x3x4xi1>
}

// -----

//===----------------------------------------------------------------------===//
/// Test shape inference for OneHotEncoder.
//===----------------------------------------------------------------------===//

func @test_onehotencoder_string1 (%arg0: tensor<20x1x!onnx.String>) -> tensor<*xf32> {
  %0 = "onnx.OneHotEncoder"(%arg0) {cats_strings = ["female", "male"], zeros = 1 : i64} : (tensor<20x1x!onnx.String>) -> tensor<*xf32>  
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_onehotencoder_string1
  // CHECK: [[RES:%.+]] = "onnx.OneHotEncoder"(%arg0) {cats_strings = ["female", "male"], zeros = 1 : i64} : (tensor<20x1x!onnx.String>) -> tensor<20x1x2xf32>
  // CHECK: return [[RES]] : tensor<20x1x2xf32>
}

// -----

func @test_onehotencoder_string2 (%arg0: tensor<20x2x!onnx.String>) -> tensor<*xf32> {
  %0 = "onnx.OneHotEncoder"(%arg0) {cats_strings = ["female", "male"], zeros = 1 : i64} : (tensor<20x2x!onnx.String>) -> tensor<*xf32>  
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_onehotencoder_string2
  // CHECK: [[RES:%.+]] = "onnx.OneHotEncoder"(%arg0) {cats_strings = ["female", "male"], zeros = 1 : i64} : (tensor<20x2x!onnx.String>) -> tensor<20x2x2xf32>
  // CHECK: return [[RES]] : tensor<20x2x2xf32>
}

// -----

func @test_onehotencoder_float1(%arg0: tensor<20x1xf32>) -> tensor<*xf32> {
  %0 = "onnx.OneHotEncoder"(%arg0) {cats_strings = ["female", "male"], cats_int64s = [1, 2, 4], zeros = 1 : i64} : (tensor<20x1xf32>) -> tensor<*xf32>  
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_onehotencoder_float1
  // CHECK: [[RES:%.+]] = "onnx.OneHotEncoder"(%arg0) {cats_int64s = [1, 2, 4], cats_strings = ["female", "male"], zeros = 1 : i64} : (tensor<20x1xf32>) -> tensor<20x1x3xf32>
  // CHECK: return [[RES]] : tensor<20x1x3xf32>
}

// -----

func @test_onehotencoder_float2(%arg0: tensor<20x2x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.OneHotEncoder"(%arg0) {cats_strings = ["female", "male"], cats_int64s = [1, 2, 4], zeros = 1 : i64} : (tensor<20x2x3xf32>) -> tensor<*xf32>  
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_onehotencoder_float2
  // CHECK: [[RES:%.+]] = "onnx.OneHotEncoder"(%arg0) {cats_int64s = [1, 2, 4], cats_strings = ["female", "male"], zeros = 1 : i64} : (tensor<20x2x3xf32>) -> tensor<20x2x3x3xf32>
  // CHECK: return [[RES]] : tensor<20x2x3x3xf32>
}