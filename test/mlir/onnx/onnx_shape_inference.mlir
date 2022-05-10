// RUN: onnx-mlir-opt --shape-inference %s -split-input-file | FileCheck %s

// -----

//===----------------------------------------------------------------------===//
/// Test the default behavior of argmax when no information for the
/// permutation of the axes is provided and when a permutation is provided.
//===----------------------------------------------------------------------===//

func @test_default_argmax(%arg0 : tensor<2x3x4xf32>) -> tensor<*xi64> {
  %0 = "onnx.ArgMax"(%arg0) : (tensor<2x3x4xf32>) -> tensor<*xi64>
  "std.return"(%0) : (tensor<*xi64>) -> ()

  // CHECK-LABEL: test_default_argmax
  // CHECK: [[RES:%.+]] = "onnx.ArgMax"(%arg0) : (tensor<2x3x4xf32>) -> tensor<1x3x4xi64>
  // CHECK: return [[RES]] : tensor<1x3x4xi64>
}

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

//===----------------------------------------------------------------------===//
/// Test shape inference for Clip.
//===----------------------------------------------------------------------===//

func @test_clip(%arg0 : tensor<1x32x112x112xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Clip"(%arg0, %cst, %cst) {max = 6.000000e+00 : f32, min = 0.000000e+00 : f32} : (tensor<1x32x112x112xf32>, none, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_clip
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none
  // CHECK-NEXT: [[RES:%.+]] = "onnx.Clip"(%arg0, [[CST]], [[CST]]) {max = 6.000000e+00 : f32, min = 0.000000e+00 : f32} : (tensor<1x32x112x112xf32>, none, none) -> tensor<1x32x112x112xf32>
  // CHECK: return [[RES]] : tensor<1x32x112x112xf32>
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
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<1x2x32xf32>, tensor<5x2x6xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_0
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none
  // CHECK-NEXT: [[RES_ATTR:%.+]] = "onnx.Conv"(%arg0, %arg1, [[CST]]) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<1x2x32xf32>, tensor<5x2x6xf32>, none) -> tensor<1x5x27xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x27xf32>
}

// -----

/// Default and required attributes.

func @test_conv_no_bias_1(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_1
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none  
  // CHECK-NEXT: [[RES_ATTR:%.+]] = "onnx.Conv"(%arg0, %arg1, [[CST]]) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, none) -> tensor<1x5x27x58xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x27x58xf32>
}

// -----

/// kernel_shape attribute.

func @test_conv_no_bias_2(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x8x9xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [8, 9]} : (tensor<1x2x32x64xf32>, tensor<5x2x8x9xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_2
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none
  // CHECK-NEXT: [[RES_ATTR:%.+]] = "onnx.Conv"(%arg0, %arg1, [[CST]]) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [8, 9]} : (tensor<1x2x32x64xf32>, tensor<5x2x8x9xf32>, none) -> tensor<1x5x25x56xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x25x56xf32>
}

// -----

/// pads attribute.
/// Use pads to make output size equal to input size by adding K - 1 to the result.

func @test_conv_no_bias_3(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x10xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", group = 1 : si64, pads = [2, 4, 3, 5]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x10xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_3
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none  
  // CHECK-NEXT: [[RES_ATTR:%.+]] = "onnx.Conv"(%arg0, %arg1, [[CST]]) {auto_pad = "NOTSET", group = 1 : si64, pads = [2, 4, 3, 5]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x10xf32>, none) -> tensor<1x5x32x64xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x32x64xf32>
}

// -----

/// auto_pad set to SAME_UPPER and SAME_LOWER.

func @test_conv_no_bias_4(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x10xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "SAME_UPPER", group = 1 : si64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x10xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_4
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none  
  // CHECK-NEXT: [[RES_ATTR:%.+]] = "onnx.Conv"(%arg0, %arg1, [[CST]]) {auto_pad = "SAME_UPPER", group = 1 : si64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x10xf32>, none) -> tensor<1x5x32x64xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x32x64xf32>
}

// -----

func @test_conv_no_bias_5(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x10xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "SAME_LOWER", group = 1 : si64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x10xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_5
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none  
  // CHECK-NEXT: [[RES_ATTR:%.+]] = "onnx.Conv"(%arg0, %arg1, [[CST]]) {auto_pad = "SAME_LOWER", group = 1 : si64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x10xf32>, none) -> tensor<1x5x32x64xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x32x64xf32>
}

// -----

/// auto_pad set to VALID.

func @test_conv_no_bias_6(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x10xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "VALID", group = 1 : si64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x10xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_6
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none  
  // CHECK-NEXT: [[RES_ATTR:%.+]] = "onnx.Conv"(%arg0, %arg1, [[CST]]) {auto_pad = "VALID", group = 1 : si64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x10xf32>, none) -> tensor<1x5x27x55xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x27x55xf32>
}

// -----

/// With strides attribute.

func @test_conv_no_bias_7(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", group = 1 : si64, strides = [2, 3]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_7
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none  
  // CHECK-NEXT: [[RES_ATTR:%.+]] = "onnx.Conv"(%arg0, %arg1, [[CST]]) {auto_pad = "NOTSET", group = 1 : si64, strides = [2, 3]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, none) -> tensor<1x5x14x20xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x14x20xf32>
}

// -----

/// auto_pad set to SAME_UPPER with strides attribute.
/// The auto_pad will pas as if stride is equal to 1.

func @test_conv_no_bias_8(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "SAME_UPPER", group = 1 : si64, strides = [2, 3]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_8
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none  
  // CHECK-NEXT: [[RES_ATTR:%.+]] = "onnx.Conv"(%arg0, %arg1, [[CST]]) {auto_pad = "SAME_UPPER", group = 1 : si64, strides = [2, 3]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, none) -> tensor<1x5x16x22xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x16x22xf32>
}

// -----

/// dilations attribute.

func @test_conv_no_bias_9(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", group = 1 : si64, dilations = [2, 3]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_9
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none  
  // CHECK-NEXT: [[RES_ATTR:%.+]] = "onnx.Conv"(%arg0, %arg1, [[CST]]) {auto_pad = "NOTSET", dilations = [2, 3], group = 1 : si64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, none) -> tensor<1x5x22x46xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x22x46xf32>
}

// -----

/// dilations attribute with stride.

func @test_conv_no_bias_10(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", group = 1 : si64, dilations = [2, 3], strides = [2, 2]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_10
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none  
  // CHECK-NEXT: [[RES_ATTR:%.+]] = "onnx.Conv"(%arg0, %arg1, [[CST]]) {auto_pad = "NOTSET", dilations = [2, 3], group = 1 : si64, strides = [2, 2]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, none) -> tensor<1x5x11x23xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x11x23xf32>
}

// -----

/// dilations attribute with auto_pad set to SAME_UPPER.

func @test_conv_no_bias_11(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "SAME_UPPER", group = 1 : si64, dilations = [2, 3]} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_11
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none  
  // CHECK-NEXT: [[RES_ATTR:%.+]] = "onnx.Conv"(%arg0, %arg1, [[CST]]) {auto_pad = "SAME_UPPER", dilations = [2, 3], group = 1 : si64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, none) -> tensor<1x5x32x64xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x32x64xf32>
}

// -----

// Test convolution with bias input.

func @test_conv_12(%arg0 : tensor<1x2x32xf32>, %arg1 : tensor<5x2x6xf32>, %arg2 : tensor<5xf32>) -> tensor<*xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<1x2x32xf32>, tensor<5x2x6xf32>, tensor<5xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_12
  // CHECK: [[RES_ATTR:%.+]] = "onnx.Conv"(%arg0, %arg1, %arg2) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<1x2x32xf32>, tensor<5x2x6xf32>, tensor<5xf32>) -> tensor<1x5x27xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x27xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// Test shape inference for ConvTranspose.
//===----------------------------------------------------------------------===//

func @test_conv_transpose_1(%arg0 : tensor<1x64x36x48xf32>, %arg1 : tensor<64x1x2x2xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ConvTranspose"(%arg0, %arg1, %cst) {dilations = [1, 1], kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x64x36x48xf32>, tensor<64x1x2x2xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_transpose_1
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none  
  // CHECK-NEXT: [[RES_ATTR:%.+]] = "onnx.ConvTranspose"(%arg0, %arg1, [[CST]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [2, 2], output_shape = [1, 1, 72, 96], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x64x36x48xf32>, tensor<64x1x2x2xf32>, none) -> tensor<1x1x72x96xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x1x72x96xf32>
}

func @test_conv_transpose_2(%arg0 : tensor<1x64x36x48xf32>, %arg1 : tensor<64x1x2x2xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ConvTranspose"(%arg0, %arg1, %cst) {dilations = [1, 1], group = 64 : si64, kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x64x36x48xf32>, tensor<64x1x2x2xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_transpose_2
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none  
  // CHECK-NEXT: [[RES_ATTR:%.+]] = "onnx.ConvTranspose"(%arg0, %arg1, [[CST]]) {auto_pad = "NOTSET", dilations = [1, 1], group = 64 : si64, kernel_shape = [2, 2], output_shape = [1, 64, 72, 96], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x64x36x48xf32>, tensor<64x1x2x2xf32>, none) -> tensor<1x64x72x96xf32>
  // CHECK: return [[RES_ATTR]] : tensor<1x64x72x96xf32>
}

// -----
//===----------------------------------------------------------------------===//

/// Test Pad_1
func @test_Pad_1(%arg0 : tensor<16x13xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[0, 2, 2, 4]> : tensor<4xi64> } : () -> tensor<4xi64>
  %1 = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<1xf32> } : () -> tensor<1xf32>
  %2 = "onnx.Pad"(%arg0, %0, %1) {mode = "constant"} : (tensor<16x13xf32>, tensor<4xi64>, tensor<1xf32>) -> tensor<*xf32>
  "std.return"(%2) : (tensor<*xf32>) -> ()
  // CHECK-LABEL: test_Pad_1
  // CHECK-SAME:     ([[VAR_arg0:%.+]]: tensor<16x13xf32>) -> tensor<18x19xf32> {
  // CHECK: [[VAR_0:%.+]] = "onnx.Constant"() {value = dense<[0, 2, 2, 4]> : tensor<4xi64>} : () -> tensor<4xi64>
  // CHECK: [[VAR_1:%.+]] = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
  // CHECK: [[VAR_2:%.+]] = "onnx.Pad"([[VAR_arg0]], [[VAR_0]], [[VAR_1]]) {mode = "constant"} : (tensor<16x13xf32>, tensor<4xi64>, tensor<1xf32>) -> tensor<18x19xf32>
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
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "VALID", ceil_mode = 0 : si64, kernel_shape = [3,3] } : (tensor<5x5x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_default_averagepool
  // CHECK: [[RES:%.+]] = "onnx.AveragePool"(%arg0) {auto_pad = "VALID", ceil_mode = 0 : si64, kernel_shape = [3, 3]} : (tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32>
  // CHECK: return [[RES]] : tensor<5x5x30x30xf32>
}

// -----

/// Test the default behavior of Average Pool with no padding (pad are not set, default to zero)
func @test_default_averagepool_defpad(%arg0 : tensor<5x5x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [3,3]} : (tensor<5x5x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_default_averagepool_defpad
  // CHECK: [[RES:%.+]] = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [3, 3]} : (tensor<5x5x32x32xf32>) -> tensor<5x5x30x30xf32>
  // CHECK: return [[RES]] : tensor<5x5x30x30xf32>
}

// -----

/// Test the default behavior of Average Pool with uniform padding
func @test_default_averagepool_pad(%arg0 : tensor<5x5x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [3,3], pads = [1, 1, 1, 1] } : (tensor<5x5x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_default_averagepool_pad
  // CHECK: [[RES:%.+]] = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1]} : (tensor<5x5x32x32xf32>) -> tensor<5x5x32x32xf32>
  // CHECK: return [[RES]] : tensor<5x5x32x32xf32>
}

// -----

/// Test the default behavior of Average Pool with non uniform padding
func @test_default_averagepool_pad_nonunif(%arg0 : tensor<5x5x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [5,3], pads = [2, 1, 1, 0] } : (tensor<5x5x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_default_averagepool_pad_nonunif
  // CHECK: [[RES:%.+]] = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [5, 3], pads = [2, 1, 1, 0]} : (tensor<5x5x32x32xf32>) -> tensor<5x5x31x31xf32>
  // CHECK: return [[RES]] : tensor<5x5x31x31xf32>
}

// -----

/// Test the default behavior of Average Pool with non uniform padding
func @test_default_averagepool_strides(%arg0 : tensor<5x5x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [3,3], pads = [1, 1, 1, 1], strides = [2, 2] } : (tensor<5x5x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_default_averagepool_strides
  // CHECK: [[RES:%.+]] = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<5x5x32x32xf32>) -> tensor<5x5x16x16xf32>
  // CHECK: return [[RES]] : tensor<5x5x16x16xf32>
}

// -----

/// Test the default behavior of Average Pool with non uniform padding
func @test_default_averagepool_strides_nonunifpad(%arg0 : tensor<5x5x30x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [2,2], pads = [1, 0, 0, 0], strides = [2, 2] } : (tensor<5x5x30x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_default_averagepool_strides_nonunifpad
  // CHECK: [[RES:%.+]] = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [2, 2], pads = [1, 0, 0, 0], strides = [2, 2]} : (tensor<5x5x30x32xf32>) -> tensor<5x5x15x16xf32>
  // CHECK: return [[RES]] : tensor<5x5x15x16xf32>
}

// -----

/// Test the default behavior of Average Pool with non uniform padding
func @test_default_averagepool_strides_nonunifpad_ceil(%arg0 : tensor<5x5x30x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 1 : si64, kernel_shape = [2,2], pads = [1, 0, 0, 0], strides = [2, 2] } : (tensor<5x5x30x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_default_averagepool_strides_nonunifpad_ceil
  // CHECK: [[RES:%.+]] = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", ceil_mode = 1 : si64, kernel_shape = [2, 2], pads = [1, 0, 0, 0], strides = [2, 2]} : (tensor<5x5x30x32xf32>) -> tensor<5x5x16x16xf32>
  // CHECK: return [[RES]] : tensor<5x5x16x16xf32>
}

// -----

func @test_global_averagepool(%arg0 : tensor<5x5x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.GlobalAveragePool"(%arg0) : (tensor<5x5x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_global_averagepool
  // CHECK: [[RES:%.+]] = "onnx.GlobalAveragePool"(%arg0) : (tensor<5x5x32x32xf32>) -> tensor<5x5x1x1xf32>
  // CHECK: return [[RES]] : tensor<5x5x1x1xf32>
}

// -----

func @test_global_lppool(%arg0 : tensor<5x5x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.GlobalLpPool"(%arg0) : (tensor<5x5x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_global_lppool
  // CHECK: [[RES:%.+]] = "onnx.GlobalLpPool"(%arg0) : (tensor<5x5x32x32xf32>) -> tensor<5x5x1x1xf32>
  // CHECK: return [[RES]] : tensor<5x5x1x1xf32>
}

// -----

func @test_global_maxpool(%arg0 : tensor<5x5x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.GlobalMaxPool"(%arg0) : (tensor<5x5x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_global_maxpool
  // CHECK: [[RES:%.+]] = "onnx.GlobalMaxPool"(%arg0) : (tensor<5x5x32x32xf32>) -> tensor<5x5x1x1xf32>
  // CHECK: return [[RES]] : tensor<5x5x1x1xf32>
}

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
  %1 = "onnx.Flatten"(%arg0) {axis = 1 : si64} : (tensor<5x2x3x4xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_flatten_1
  // CHECK: [[RES:%.+]] = "onnx.Flatten"(%arg0) {axis = 1 : si64} : (tensor<5x2x3x4xf32>) -> tensor<5x24xf32>
  // CHECK: return [[RES]] : tensor<5x24xf32>
}

// -----

// Test when axis is 0
func @test_flatten_2(%arg0 : tensor<2x3x4xf32>) -> tensor<*xf32> {
  %1 = "onnx.Flatten"(%arg0) {axis = 0 : si64} : (tensor<2x3x4xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()
  // CHECK-LABEL: test_flatten_2
  // CHECK: [[RES:%.+]] = "onnx.Flatten"(%arg0) {axis = 0 : si64} : (tensor<2x3x4xf32>) -> tensor<1x24xf32>
  // CHECK: return [[RES]] : tensor<1x24xf32>
}

// -----

// Test when axis is negative
func @test_flatten_3(%arg0 : tensor<2x3x4xf32>) -> tensor<*xf32> {
  %1 = "onnx.Flatten"(%arg0) {axis = -1 : si64} : (tensor<2x3x4xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()
  // CHECK-LABEL: test_flatten_3
  // CHECK: [[RES:%.+]] = "onnx.Flatten"(%arg0) {axis = -1 : si64} : (tensor<2x3x4xf32>) -> tensor<6x4xf32>
  // CHECK: return [[RES]] : tensor<6x4xf32>
}

// -----

// Test when input is not static shape
func @test_flatten_4(%arg0 : tensor<2x4x5x?xf32>) -> tensor<*xf32> {
  %1 = "onnx.Flatten"(%arg0) {axis = 2 : si64} : (tensor<2x4x5x?xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()
  // CHECK-LABEL: test_flatten_4
  // CHECK: [[RES:%.+]] = "onnx.Flatten"(%arg0) {axis = 2 : si64} : (tensor<2x4x5x?xf32>) -> tensor<8x?xf32>
  // CHECK: return [[RES]] : tensor<8x?xf32>
}


//===----------------------------------------------------------------------===//
/// Test the reshape op inference when concat are present.
//===----------------------------------------------------------------------===//

func @test_concat_1(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<5x5x3x32xf32>, %arg2 : tensor<5x5x5x32xf32>) -> tensor<*xf32> {
  %1 = "onnx.Concat"(%arg0, %arg1, %arg2) { axis = 2 : si64} : (tensor<5x5x1x32xf32>, tensor<5x5x3x32xf32>, tensor<5x5x5x32xf32>)  -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_concat_1
  // CHECK: [[RES:%.+]] = "onnx.Concat"(%arg0, %arg1, %arg2) {axis = 2 : si64} : (tensor<5x5x1x32xf32>, tensor<5x5x3x32xf32>, tensor<5x5x5x32xf32>) -> tensor<5x5x9x32xf32>
  // CHECK: return [[RES]] : tensor<5x5x9x32xf32>
}

// -----

func @test_concat_2(%arg0 : tensor<5x1x32xf32>, %arg1 : tensor<5x3x32xf32>, %arg2 : tensor<5x5x32xf32>) -> tensor<*xf32> {
  %1 = "onnx.Concat"(%arg0, %arg1, %arg2) { axis = 1 : si64} : (tensor<5x1x32xf32>, tensor<5x3x32xf32>, tensor<5x5x32xf32>)  -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_concat_2
  // CHECK: [[RES:%.+]] = "onnx.Concat"(%arg0, %arg1, %arg2) {axis = 1 : si64} : (tensor<5x1x32xf32>, tensor<5x3x32xf32>, tensor<5x5x32xf32>) -> tensor<5x9x32xf32>
  // CHECK: return [[RES]] : tensor<5x9x32xf32>
}

// -----

func @test_concat_3(%arg0 : tensor<5x1x32xf32>, %arg1 : tensor<5x3x32xf32>, %arg2 : tensor<5x5x32xf32>) -> tensor<*xf32> {
  %1 = "onnx.Concat"(%arg0, %arg1, %arg2) { axis = -2 : si64} : (tensor<5x1x32xf32>, tensor<5x3x32xf32>, tensor<5x5x32xf32>)  -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_concat_3
  // CHECK: [[RES:%.+]] = "onnx.Concat"(%arg0, %arg1, %arg2) {axis = 1 : si64} : (tensor<5x1x32xf32>, tensor<5x3x32xf32>, tensor<5x5x32xf32>) -> tensor<5x9x32xf32>
  // CHECK: return [[RES]] : tensor<5x9x32xf32>
}

// -----

func @test_rnn_all_results(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (tensor<*xf32>, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_rnn_all_results
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none  
  // CHECK-NEXT: %{{.*}}, [[RES:%.+]] = "onnx.RNN"(%arg0, %arg1, %arg2, [[CST]], [[CST]], [[CST]]) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (tensor<4x1x3x3xf32>, tensor<1x3x3xf32>)
  // CHECK: return [[RES]] : tensor<1x3x3xf32>
}

// -----

func @test_rnn_no_results(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> () {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (none, none)
  return

  // CHECK-LABEL: test_rnn_no_results
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none
  // CHECK-NEXT: %{{.*}}, [[RES:%.+]] = "onnx.RNN"(%arg0, %arg1, %arg2, [[CST]], [[CST]], [[CST]]) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (none, none)
  // CHECK: return
}

// -----

func @test_rnn_missing_first_result(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_rnn_missing_first_result
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none
  // CHECK-NEXT: %{{.*}}, [[RES:%.+]] = "onnx.RNN"(%arg0, %arg1, %arg2, [[CST]], [[CST]], [[CST]]) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (none, tensor<1x3x3xf32>)
  // CHECK: return [[RES]] : tensor<1x3x3xf32>
}

// -----

func @test_rnn_missing_trailing_result(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> () {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (tensor<*xf32>, none)
  return

  // CHECK-LABEL: test_rnn_missing_trailing_result
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none  
  // CHECK-NEXT: %{{.*}}, [[RES:%.+]] = "onnx.RNN"(%arg0, %arg1, %arg2, [[CST]], [[CST]], [[CST]]) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (tensor<4x1x3x3xf32>, none)
  // CHECK: return
}

// -----

func @test_rnn_all_results_no_hidden_size(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %cst, %cst, %cst) : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (tensor<*xf32>, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_rnn_all_results_no_hidden_size
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none
  // CHECK-NEXT: %{{.*}}, [[RES:%.+]] = "onnx.RNN"(%arg0, %arg1, %arg2, [[CST]], [[CST]], [[CST]]) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (tensor<4x1x3x3xf32>, tensor<1x3x3xf32>)
  // CHECK: return [[RES]] : tensor<1x3x3xf32>
}

// -----

func @test_rnn_all_results_unknown_dims(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %cst, %cst, %cst) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, none, none, none) -> (tensor<*xf32>, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_rnn_all_results_unknown_dims
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none  
  // CHECK-NEXT: %{{.*}}, [[RES:%.+]] = "onnx.RNN"(%arg0, %arg1, %arg2, [[CST]], [[CST]], [[CST]]) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, none, none, none) -> (tensor<?x1x?x?xf32>, tensor<1x?x?xf32>)
  // CHECK: return [[RES]] : tensor<1x?x?xf32>
}

// -----

func @test_gru_all_results(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (tensor<*xf32>, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_gru_all_results
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none  
  // CHECK-NEXT: %{{.*}}, [[RES:%.+]] = "onnx.GRU"(%arg0, %arg1, %arg2, [[CST]], [[CST]], [[CST]]) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (tensor<4x1x3x3xf32>, tensor<1x3x3xf32>)
  // CHECK: return [[RES]] : tensor<1x3x3xf32>
}

// -----

func @test_gru_no_results(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> () {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (none, none)
  return

  // CHECK-LABEL: test_gru_no_results
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none  
  // CHECK-NEXT: %{{.*}}, [[RES:%.+]] = "onnx.GRU"(%arg0, %arg1, %arg2, [[CST]], [[CST]], [[CST]]) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (none, none)
  // CHECK: return
}

// -----

func @test_gru_missing_first_result(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_gru_missing_first_result
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none  
  // CHECK-NEXT: %{{.*}}, [[RES:%.+]] = "onnx.GRU"(%arg0, %arg1, %arg2, [[CST]], [[CST]], [[CST]]) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (none, tensor<1x3x3xf32>)
  // CHECK: return [[RES]] : tensor<1x3x3xf32>
}

// -----

func @test_gru_missing_trailing_result(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> () {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (tensor<*xf32>, none)
  return

  // CHECK-LABEL: test_gru_missing_trailing_result
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none  
  // CHECK-NEXT: %{{.*}}, [[RES:%.+]] = "onnx.GRU"(%arg0, %arg1, %arg2, [[CST]], [[CST]], [[CST]]) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (tensor<4x1x3x3xf32>, none)
  // CHECK: return
}

// -----

func @test_gru_all_results_no_hidden_size(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %cst, %cst, %cst) : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (tensor<*xf32>, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_gru_all_results_no_hidden_size
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none  
  // CHECK-NEXT: %{{.*}}, [[RES:%.+]] = "onnx.GRU"(%arg0, %arg1, %arg2, [[CST]], [[CST]], [[CST]]) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none) -> (tensor<4x1x3x3xf32>, tensor<1x3x3xf32>)
  // CHECK: return [[RES]] : tensor<1x3x3xf32>
}

// -----

func @test_gru_all_results_unknown_dims(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %cst, %cst, %cst) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, none, none, none) -> (tensor<*xf32>, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_gru_all_results_unknown_dims
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none  
  // CHECK-NEXT: %{{.*}}, [[RES:%.+]] = "onnx.GRU"(%arg0, %arg1, %arg2, [[CST]], [[CST]], [[CST]]) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, none, none, none) -> (tensor<?x1x?x?xf32>, tensor<1x?x?xf32>)
  // CHECK: return [[RES]] : tensor<1x?x?xf32>
}

// -----

func @test_lstm_all_results(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_lstm_all_results
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none  
  // CHECK-NEXT: %{{.*}}, [[RES:%.+]], %{{.*}} = "onnx.LSTM"(%arg0, %arg1, %arg2, [[CST]], [[CST]], [[CST]], [[CST]], [[CST]]) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (tensor<4x1x3x3xf32>, tensor<1x3x3xf32>, tensor<1x3x3xf32>)
  // CHECK: return [[RES]] : tensor<1x3x3xf32>
}

// -----

func @test_lstm_no_results(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> () {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (none, none, none)
  return

  // CHECK-LABEL: test_lstm_no_results
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none  
  // CHECK-NEXT: %{{.*}}, [[RES:%.+]], %{{.*}} = "onnx.LSTM"(%arg0, %arg1, %arg2, [[CST]], [[CST]], [[CST]], [[CST]], [[CST]]) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (none, none, none)
  // CHECK: return
}

// -----

func @test_lstm_missing_first_result(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (none, tensor<*xf32>, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_lstm_missing_first_result
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none  
  // CHECK-NEXT: %{{.*}}, [[RES:%.+]], %{{.*}} = "onnx.LSTM"(%arg0, %arg1, %arg2, [[CST]], [[CST]], [[CST]], [[CST]], [[CST]]) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (none, tensor<1x3x3xf32>, tensor<1x3x3xf32>)
  // CHECK: return [[RES]] : tensor<1x3x3xf32>
}

// -----

func @test_lstm_missing_trailing_result(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (tensor<*xf32>, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_lstm_missing_trailing_result
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none  
  // CHECK-NEXT: %{{.*}}, [[RES:%.+]], %{{.*}} = "onnx.LSTM"(%arg0, %arg1, %arg2, [[CST]], [[CST]], [[CST]], [[CST]], [[CST]]) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (tensor<4x1x3x3xf32>, tensor<1x3x3xf32>, none)
  // CHECK: return [[RES]] : tensor<1x3x3xf32>
}

// -----

func @test_lstm_all_results_no_hidden_size(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_lstm_all_results_no_hidden_size
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none  
  // CHECK-NEXT: %{{.*}}, [[RES:%.+]], %{{.*}} = "onnx.LSTM"(%arg0, %arg1, %arg2, [[CST]], [[CST]], [[CST]], [[CST]], [[CST]]) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (tensor<4x1x3x3xf32>, tensor<1x3x3xf32>, tensor<1x3x3xf32>)
  // CHECK: return [[RES]] : tensor<1x3x3xf32>
}

// -----

func @test_lstm_all_results_unknown_dims(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, none, none, none, none, none) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_lstm_all_results_unknown_dims
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none  
  // CHECK-NEXT: %{{.*}}, [[RES:%.+]], %{{.*}} = "onnx.LSTM"(%arg0, %arg1, %arg2, [[CST]], [[CST]], [[CST]], [[CST]], [[CST]]) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>, none, none, none, none, none) -> (tensor<?x1x?x?xf32>, tensor<1x?x?xf32>, tensor<1x?x?xf32>)
  // CHECK: return [[RES]] : tensor<1x?x?xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// Test the behavior of the SpaceToDepth operator.
//===----------------------------------------------------------------------===//

func @test_space_to_depth(%arg0 : tensor<1x16x32x64xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.SpaceToDepth"(%arg0) {blocksize = 4 : si64} : (tensor<1x16x32x64xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_space_to_depth
  // CHECK: [[RES:%.+]] = "onnx.SpaceToDepth"(%arg0) {blocksize = 4 : si64} : (tensor<1x16x32x64xf32>) -> tensor<1x256x8x16xf32>
  // CHECK: return [[RES]] : tensor<1x256x8x16xf32>
}

// -----

func @test_split_1(%arg0 : tensor<16x32x64xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0, %1 = "onnx.Split"(%arg0, %cst) { axis = 1 : si64} : (tensor<16x32x64xf32>, none) -> (tensor<*xf32>, tensor<*xf32>)
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_split_1
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none  
  // CHECK-NEXT: [[RES:%.+]]:2 = "onnx.Split"(%arg0, [[CST]]) {axis = 1 : si64} : (tensor<16x32x64xf32>, none) -> (tensor<16x16x64xf32>, tensor<16x16x64xf32>)
  // CHECK: return [[RES]]#0 : tensor<16x16x64xf32>
}

// -----

func @test_split_2(%arg0 : tensor<16x32x64xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0, %1 = "onnx.Split"(%arg0, %cst) { axis = -2 : si64} : (tensor<16x32x64xf32>, none) -> (tensor<*xf32>, tensor<*xf32>)
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_split_2
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none  
  // CHECK-NEXT: [[RES:%.+]]:2 = "onnx.Split"(%arg0, [[CST]]) {axis = 1 : si64} : (tensor<16x32x64xf32>, none) -> (tensor<16x16x64xf32>, tensor<16x16x64xf32>)
  // CHECK: return [[RES]]#0 : tensor<16x16x64xf32>
}

// -----

func @test_split_3(%arg0 : tensor<16x32x64xf32>) -> tensor<*xf32> {
  %split = "onnx.Constant"() {value = dense<[2, 30]> : tensor<2xi64>} : () -> tensor<2xi64>
  %0, %1 = "onnx.Split"(%arg0, %split) {axis = 1 : si64} : (tensor<16x32x64xf32>, tensor<2xi64>) -> (tensor<*xf32>, tensor<*xf32>)
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_split_3
  // CHECK: [[RES:%.+]]:2 = "onnx.Split"(%arg0, %0) {axis = 1 : si64} : (tensor<16x32x64xf32>, tensor<2xi64>) -> (tensor<16x2x64xf32>, tensor<16x30x64xf32>)
  // CHECK: return [[RES]]#0 : tensor<16x2x64xf32>
}

// -----

func @test_split_4(%arg0 : tensor<16x?x64xf32>) -> tensor<*xf32> {
  %split = "onnx.Constant"() {value = dense<[2, 30]> : tensor<2xi64>} : () -> tensor<2xi64>
  %0, %1 = "onnx.Split"(%arg0, %split) {axis = 1 : si64} : (tensor<16x?x64xf32>, tensor<2xi64>) -> (tensor<*xf32>, tensor<*xf32>)
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_split_4
  // CHECK: [[RES:%.+]]:2 = "onnx.Split"(%arg0, %0) {axis = 1 : si64} : (tensor<16x?x64xf32>, tensor<2xi64>) -> (tensor<16x2x64xf32>, tensor<16x30x64xf32>)
  // CHECK: return [[RES]]#0 : tensor<16x2x64xf32>
}

// -----

func @test_split_5(%arg0 : tensor<16x?x64xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0, %1 = "onnx.Split"(%arg0, %cst) {axis = 1 : si64} : (tensor<16x?x64xf32>, none) -> (tensor<*xf32>, tensor<*xf32>)
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_split_5
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none  
  // CHECK-NEXT: [[RES:%.+]]:2 = "onnx.Split"(%arg0, [[CST]]) {axis = 1 : si64} : (tensor<16x?x64xf32>, none) -> (tensor<16x?x64xf32>, tensor<16x?x64xf32>)
  // CHECK: return [[RES]]#0 : tensor<16x?x64xf32>
}

// -----

func @test_splitv11_1(%arg0 : tensor<16x32x64xf32>) -> tensor<*xf32> {
  %0, %1 = "onnx.SplitV11"(%arg0) { axis = 1 : si64} : (tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_splitv11_1
  // CHECK: [[RES:%.+]]:2 = "onnx.SplitV11"(%arg0) {axis = 1 : si64} : (tensor<16x32x64xf32>) -> (tensor<16x16x64xf32>, tensor<16x16x64xf32>)
  // CHECK: return [[RES]]#0 : tensor<16x16x64xf32>
}

// -----

func @test_splitv11_2(%arg0 : tensor<16x32x64xf32>) -> tensor<*xf32> {
  %0, %1 = "onnx.SplitV11"(%arg0) { axis = -2 : si64} : (tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_splitv11_2
  // CHECK: [[RES:%.+]]:2 = "onnx.SplitV11"(%arg0) {axis = 1 : si64} : (tensor<16x32x64xf32>) -> (tensor<16x16x64xf32>, tensor<16x16x64xf32>)
  // CHECK: return [[RES]]#0 : tensor<16x16x64xf32>
}

// -----

func @test_splitv11_3(%arg0 : tensor<16x32x64xf32>) -> tensor<*xf32> {
  %0, %1 = "onnx.SplitV11"(%arg0) {axis = 1 : si64, split = [2, 30]} : (tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_splitv11_3
  // CHECK: [[RES:%.+]]:2 = "onnx.SplitV11"(%arg0) {axis = 1 : si64, split = [2, 30]} : (tensor<16x32x64xf32>) -> (tensor<16x2x64xf32>, tensor<16x30x64xf32>)
  // CHECK: return [[RES]]#0 : tensor<16x2x64xf32>
}

// -----

func @test_splitv11_4(%arg0 : tensor<16x?x64xf32>) -> tensor<*xf32> {
  %0, %1 = "onnx.SplitV11"(%arg0) {axis = 1 : si64, split = [2, 30]} : (tensor<16x?x64xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_splitv11_4
  // CHECK: [[RES:%.+]]:2 = "onnx.SplitV11"(%arg0) {axis = 1 : si64, split = [2, 30]} : (tensor<16x?x64xf32>) -> (tensor<16x2x64xf32>, tensor<16x30x64xf32>)
  // CHECK: return [[RES]]#0 : tensor<16x2x64xf32>
}

// -----

func @test_splitv11_5(%arg0 : tensor<16x?x64xf32>) -> tensor<*xf32> {
  %0, %1 = "onnx.SplitV11"(%arg0) {axis = 1 : si64} : (tensor<16x?x64xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_splitv11_5
  // CHECK: [[RES:%.+]]:2 = "onnx.SplitV11"(%arg0) {axis = 1 : si64} : (tensor<16x?x64xf32>) -> (tensor<16x?x64xf32>, tensor<16x?x64xf32>)
  // CHECK: return [[RES]]#0 : tensor<16x?x64xf32>
}

// -----

func @test_squeeze(%arg0 : tensor<16x1x32x1x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
  %1 = "onnx.Squeeze"(%arg0, %0) : (tensor<16x1x32x1x64xf32>, tensor<1xi64>) -> (tensor<*xf32>)
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_squeeze
  // CHECK: [[RES:%.+]] = "onnx.Squeeze"(%arg0, %0) : (tensor<16x1x32x1x64xf32>, tensor<1xi64>) -> tensor<16x32x1x64xf32>
  // CHECK: return [[RES]] : tensor<16x32x1x64xf32>
}

// -----

func @test_squeezev11(%arg0 : tensor<16x1x32x1x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.SqueezeV11"(%arg0) { axes = [1]} : (tensor<16x1x32x1x64xf32>) -> (tensor<*xf32>)
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_squeezev11
  // CHECK: [[RES:%.+]] = "onnx.SqueezeV11"(%arg0) {axes = [1]} : (tensor<16x1x32x1x64xf32>) -> tensor<16x32x1x64xf32>
  // CHECK: return [[RES]] : tensor<16x32x1x64xf32>
}

// -----

func @test_squeeze_negative_axis(%arg0 : tensor<16x1x32x1x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[-2]> : tensor<1xi64>} : () -> tensor<1xi64>
  %1 = "onnx.Squeeze"(%arg0, %0) : (tensor<16x1x32x1x64xf32>, tensor<1xi64>) -> (tensor<*xf32>)
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_squeeze_negative_axis
  // CHECK: [[CSTPOS:%.+]] = "onnx.Constant"() {value = dense<3> : tensor<1xi64>} : () -> tensor<1xi64>
  // CHECK: [[RES:%.+]] = "onnx.Squeeze"(%arg0, [[CSTPOS]]) : (tensor<16x1x32x1x64xf32>, tensor<1xi64>) -> tensor<16x1x32x64xf32>
  // CHECK: return [[RES]] : tensor<16x1x32x64xf32>
}

// -----

func @test_squeezev11_negative_axis(%arg0 : tensor<16x1x32x1x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.SqueezeV11"(%arg0) { axes = [-2]} : (tensor<16x1x32x1x64xf32>) -> (tensor<*xf32>)
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_squeezev11_negative_axis
  // CHECK: [[RES:%.+]] = "onnx.SqueezeV11"(%arg0) {axes = [3]} : (tensor<16x1x32x1x64xf32>) -> tensor<16x1x32x64xf32>
  // CHECK: return [[RES]] : tensor<16x1x32x64xf32>
}

// -----

func @test_squeeze_mix(%arg0 : tensor<16x1x32x1x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[1, -2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Squeeze"(%arg0, %0) : (tensor<16x1x32x1x64xf32>, tensor<2xi64>) -> (tensor<*xf32>)
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_squeeze_mix
  // CHECK: [[CSTPOS:%.+]] = "onnx.Constant"() {value = dense<[1, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  // CHECK: [[RES:%.+]] = "onnx.Squeeze"(%arg0, [[CSTPOS]]) : (tensor<16x1x32x1x64xf32>, tensor<2xi64>) -> tensor<16x32x64xf32>
  // CHECK: return [[RES]] : tensor<16x32x64xf32>
}

// -----

func @test_squeezev11_mix(%arg0 : tensor<16x1x32x1x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.SqueezeV11"(%arg0) { axes = [1, -2]} : (tensor<16x1x32x1x64xf32>) -> (tensor<*xf32>)
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_squeezev11_mix
  // CHECK: [[RES:%.+]] = "onnx.SqueezeV11"(%arg0) {axes = [1, 3]} : (tensor<16x1x32x1x64xf32>) -> tensor<16x32x64xf32>
  // CHECK: return [[RES]] : tensor<16x32x64xf32>
}

// -----

func private @test_squeeze_empty_axes(%arg0 : tensor<16x1x32x1x64xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.Squeeze"(%arg0, %cst) : (tensor<16x1x32x1x64xf32>, none) -> (tensor<*xf32>)
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_squeeze_empty_axes
  // CHECK: [[CONST:%.+]] = "onnx.Constant"() {value = dense<[1, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  // CHECK: [[RES:%.+]] = "onnx.Squeeze"(%arg0, [[CONST]]) : (tensor<16x1x32x1x64xf32>, tensor<2xi64>) -> tensor<16x32x64xf32>
  // CHECK: return [[RES]] : tensor<16x32x64xf32>
}

// -----

func private @test_squeezev11_empty_axes(%arg0 : tensor<16x1x32x1x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.SqueezeV11"(%arg0) : (tensor<16x1x32x1x64xf32>) -> (tensor<*xf32>)
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_squeezev11_empty_axes
  // CHECK: [[RES:%.+]] = "onnx.SqueezeV11"(%arg0) {axes = [1, 3]} : (tensor<16x1x32x1x64xf32>) -> tensor<16x32x64xf32>
  // CHECK: return [[RES]] : tensor<16x32x64xf32>
}

// -----

func @test_unsqueeze(%arg0 : tensor<16x32x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64>} : () -> tensor<1xi64>
  %1 = "onnx.Unsqueeze"(%arg0, %0) : (tensor<16x32x64xf32>, tensor<1xi64>) -> (tensor<*xf32>)
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_unsqueeze
  // CHECK: [[RES:%.+]] = "onnx.Unsqueeze"(%arg0, %0) : (tensor<16x32x64xf32>, tensor<1xi64>) -> tensor<16x1x32x64xf32>
  // CHECK: return [[RES]] : tensor<16x1x32x64xf32>
}

// -----

func private @unsqueeze_of_const(%arg0: tensor<32x64xi64>) -> tensor<*xi64> {
   %1 = "onnx.Constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
   %2 = "onnx.Unsqueeze"(%arg0, %1) : (tensor<32x64xi64>, tensor<i64>) -> tensor<*xi64>
  "std.return"(%2) : (tensor<*xi64>) -> ()

// mlir2FileCheck.py -a'["data"]'
// CHECK-LABEL:  func private @unsqueeze_of_const
// CHECK-SAME:   ([[DATA_:%.+]]: tensor<32x64xi64>) -> tensor<1x32x64xi64> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Constant"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Unsqueeze"([[DATA_]], [[VAR_0_]]) : (tensor<32x64xi64>, tensor<i64>) -> tensor<1x32x64xi64>
// CHECK:           return [[VAR_1_]] : tensor<1x32x64xi64>
// CHECK:         }
}

// -----

func @test_unsqueezev11(%arg0 : tensor<16x32x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.UnsqueezeV11"(%arg0) { axes = [1]} : (tensor<16x32x64xf32>) -> (tensor<*xf32>)
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_unsqueezev11
  // CHECK: [[RES:%.+]] = "onnx.UnsqueezeV11"(%arg0) {axes = [1]} : (tensor<16x32x64xf32>) -> tensor<16x1x32x64xf32>
  // CHECK: return [[RES]] : tensor<16x1x32x64xf32>
}

// -----

func @test_unsqueeze_negative_axis(%arg0 : tensor<16x32x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[-2]> : tensor<1xi64>} : () -> tensor<1xi64>
  %1 = "onnx.Unsqueeze"(%arg0, %0) : (tensor<16x32x64xf32>, tensor<1xi64>) -> (tensor<*xf32>)
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_unsqueeze_negative_axis
  // CHECK: [[CSTPOS:%.+]] = "onnx.Constant"() {value = dense<2> : tensor<1xi64>} : () -> tensor<1xi64>
  // CHECK: [[RES:%.+]] = "onnx.Unsqueeze"(%arg0, [[CSTPOS]]) : (tensor<16x32x64xf32>, tensor<1xi64>) -> tensor<16x32x1x64xf32>
  // CHECK: return [[RES]] : tensor<16x32x1x64xf32>
}

// -----

func @test_unsqueezev11_negative_axis(%arg0 : tensor<16x32x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.UnsqueezeV11"(%arg0) { axes = [-2]} : (tensor<16x32x64xf32>) -> (tensor<*xf32>)
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_unsqueezev11_negative_axis
  // CHECK: [[RES:%.+]] = "onnx.UnsqueezeV11"(%arg0) {axes = [2]} : (tensor<16x32x64xf32>) -> tensor<16x32x1x64xf32>
  // CHECK: return [[RES]] : tensor<16x32x1x64xf32>
}

// -----

func @test_unsqueeze_mix(%arg0 : tensor<16x32x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[1, -2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Unsqueeze"(%arg0, %0) : (tensor<16x32x64xf32>, tensor<2xi64>) -> (tensor<*xf32>)
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_unsqueeze_mix
  // CHECK: [[CSTPOS:%.+]] = "onnx.Constant"() {value = dense<[1, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  // CHECK: [[RES:%.+]] = "onnx.Unsqueeze"(%arg0, [[CSTPOS]]) : (tensor<16x32x64xf32>, tensor<2xi64>) -> tensor<16x1x32x1x64xf32>
  // CHECK: return [[RES]] : tensor<16x1x32x1x64xf32>
}

// -----

func @test_unsqueezev11_mix(%arg0 : tensor<16x32x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.UnsqueezeV11"(%arg0) { axes = [1, -2]} : (tensor<16x32x64xf32>) -> (tensor<*xf32>)
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_unsqueezev11_mix
  // CHECK: [[RES:%.+]] = "onnx.UnsqueezeV11"(%arg0) {axes = [1, 3]} : (tensor<16x32x64xf32>) -> tensor<16x1x32x1x64xf32>
  // CHECK: return [[RES]] : tensor<16x1x32x1x64xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// Test the cast op inference.
//===----------------------------------------------------------------------===//

func @test_cast_1(%arg0 : tensor<2x3x4xf32>) -> tensor<*xf32> {
  %1 = "onnx.Cast"(%arg0) {to = f32} : (tensor<2x3x4xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_cast_1
  // CHECK: [[RES:%.+]] = "onnx.Cast"(%arg0) {to = f32} : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  // CHECK: return [[RES]] : tensor<2x3x4xf32>
}

func @test_cast_2(%arg0 : tensor<2x3x4xf32>) -> tensor<*xui8> {
  %1 = "onnx.Cast"(%arg0) {to = ui8} : (tensor<2x3x4xf32>) -> tensor<*xui8>
  "std.return"(%1) : (tensor<*xui8>) -> ()

  // CHECK-LABEL: test_cast_2
  // CHECK: [[RES:%.+]] = "onnx.Cast"(%arg0) {to = ui8} : (tensor<2x3x4xf32>) -> tensor<2x3x4xui8>
  // CHECK: return [[RES]] : tensor<2x3x4xui8>
}

func @test_cast_3(%arg0 : tensor<2x3x4xf32>) -> tensor<*xi8> {
  %1 = "onnx.Cast"(%arg0) {to = i8} : (tensor<2x3x4xf32>) -> tensor<*xi8>
  "std.return"(%1) : (tensor<*xi8>) -> ()

  // CHECK-LABEL: test_cast_3
  // CHECK: [[RES:%.+]] = "onnx.Cast"(%arg0) {to = i8} : (tensor<2x3x4xf32>) -> tensor<2x3x4xi8>
  // CHECK: return [[RES]] : tensor<2x3x4xi8>
}

func @test_cast_10(%arg0 : tensor<2x3x4xf32>) -> tensor<*xf16> {
  %1 = "onnx.Cast"(%arg0) {to = f16} : (tensor<2x3x4xf32>) -> tensor<*xf16>
  "std.return"(%1) : (tensor<*xf16>) -> ()

  // CHECK-LABEL: test_cast_10
  // CHECK: [[RES:%.+]] = "onnx.Cast"(%arg0) {to = f16} : (tensor<2x3x4xf32>) -> tensor<2x3x4xf16>
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
  %0 = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<1x2x32xi8>, tensor<5x2x6xi8>, tensor<i8>, tensor<i8>) -> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_convinteger_0
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", dilations = [1], group = 1 : si64, kernel_shape = [6], pads = [0, 0], strides = [1]} : (tensor<1x2x32xi8>, tensor<5x2x6xi8>, tensor<i8>, tensor<i8>) -> tensor<1x5x27xi32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x27xi32>
}

/// Default and required attributes.

func @test_convinteger_1(%arg0 : tensor<1x2x32x64xi8>, %arg1 : tensor<5x2x6x7xi8>, %arg2 : tensor<i8>, %arg3 : tensor<i8>) -> tensor<*xi32> {
  %0 = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<1x2x32x64xi8>, tensor<5x2x6x7xi8>, tensor<i8>, tensor<i8>) -> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_convinteger_1
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [6, 7], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x7xi8>, tensor<i8>, tensor<i8>) -> tensor<1x5x27x58xi32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x27x58xi32>
}

/// kernel_shape attribute.

func @test_convinteger_2(%arg0 : tensor<1x2x32x64xi8>, %arg1 : tensor<5x2x6x7xi8>, %arg2 : tensor<i8>, %arg3 : tensor<i8>) -> tensor<*xi32> {
  %0 = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [8, 9]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x7xi8>, tensor<i8>, tensor<i8>) -> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_convinteger_2
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [8, 9], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x7xi8>, tensor<i8>, tensor<i8>) -> tensor<1x5x25x56xi32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x25x56xi32>
}

/// pads attribute.
/// Use pads to make output size equal to input size by adding K - 1 to the result.

func @test_convinteger_3(%arg0 : tensor<1x2x32x64xi8>, %arg1 : tensor<5x2x6x10xi8>, %arg2 : tensor<i8>, %arg3 : tensor<i8>) -> tensor<*xi32> {
  %0 = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", group = 1 : si64, pads = [2, 4, 3, 5]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x10xi8>, tensor<i8>, tensor<i8>) -> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_convinteger_3
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [6, 10], pads = [2, 4, 3, 5], strides = [1, 1]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x10xi8>, tensor<i8>, tensor<i8>) -> tensor<1x5x32x64xi32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x32x64xi32>
}

/// auto_pad set to SAME_UPPER and SAME_LOWER.

func @test_convinteger_4(%arg0 : tensor<1x2x32x64xi8>, %arg1 : tensor<5x2x6x10xi8>, %arg2 : tensor<i8>, %arg3 : tensor<i8>) -> tensor<*xi32> {
  %0 = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "SAME_UPPER", group = 1 : si64} : (tensor<1x2x32x64xi8>, tensor<5x2x6x10xi8>, tensor<i8>, tensor<i8>) -> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_convinteger_4
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [6, 10], pads = [2, 4, 3, 5], strides = [1, 1]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x10xi8>, tensor<i8>, tensor<i8>) -> tensor<1x5x32x64xi32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x32x64xi32>
}

func @test_convinteger_5(%arg0 : tensor<1x2x32x64xi8>, %arg1 : tensor<5x2x6x10xi8>, %arg2 : tensor<i8>, %arg3 : tensor<i8>) -> tensor<*xi32> {
  %0 = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "SAME_LOWER", group = 1 : si64} : (tensor<1x2x32x64xi8>, tensor<5x2x6x10xi8>, tensor<i8>, tensor<i8>) -> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_convinteger_5
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [6, 10], pads = [3, 5, 2, 4], strides = [1, 1]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x10xi8>, tensor<i8>, tensor<i8>) -> tensor<1x5x32x64xi32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x32x64xi32>
}

/// auto_pad set to VALID.

func @test_convinteger_6(%arg0 : tensor<1x2x32x64xi8>, %arg1 : tensor<5x2x6x10xi8>, %arg2 : tensor<i8>, %arg3 : tensor<i8>) -> tensor<*xi32> {
  %0 = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "VALID", group = 1 : si64} : (tensor<1x2x32x64xi8>, tensor<5x2x6x10xi8>, tensor<i8>, tensor<i8>) -> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_convinteger_6
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [6, 10], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x10xi8>, tensor<i8>, tensor<i8>) -> tensor<1x5x27x55xi32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x27x55xi32>
}

/// With strides attribute.

func @test_convinteger_7(%arg0 : tensor<1x2x32x64xi8>, %arg1 : tensor<5x2x6x7xi8>, %arg2 : tensor<i8>, %arg3 : tensor<i8>) -> tensor<*xi32> {
  %0 = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", group = 1 : si64, strides = [2, 3]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x7xi8>, tensor<i8>, tensor<i8>) -> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_convinteger_7
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [6, 7], pads = [0, 0, 0, 0], strides = [2, 3]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x7xi8>, tensor<i8>, tensor<i8>) -> tensor<1x5x14x20xi32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x14x20xi32>
}

/// auto_pad set to SAME_UPPER with strides attribute.
/// The auto_pad will pas as if stride is equal to 1.

func @test_convinteger_8(%arg0 : tensor<1x2x32x64xi8>, %arg1 : tensor<5x2x6x7xi8>, %arg2 : tensor<i8>, %arg3 : tensor<i8>) -> tensor<*xi32> {
  %0 = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "SAME_UPPER", group = 1 : si64, strides = [2, 3]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x7xi8>, tensor<i8>, tensor<i8>) -> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_convinteger_8
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [6, 7], pads = [2, 3, 2, 3], strides = [2, 3]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x7xi8>, tensor<i8>, tensor<i8>) -> tensor<1x5x16x22xi32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x16x22xi32>
}

/// dilations attribute.

func @test_convinteger_9(%arg0 : tensor<1x2x32x64xi8>, %arg1 : tensor<5x2x6x7xi8>, %arg2 : tensor<i8>, %arg3 : tensor<i8>) -> tensor<*xi32> {
  %0 = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", group = 1 : si64, dilations = [2, 3]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x7xi8>, tensor<i8>, tensor<i8>) -> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_convinteger_9
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", dilations = [2, 3], group = 1 : si64, kernel_shape = [6, 7], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x7xi8>, tensor<i8>, tensor<i8>) -> tensor<1x5x22x46xi32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x22x46xi32>
}

/// dilations attribute with stride.

func @test_convinteger_10(%arg0 : tensor<1x2x32x64xi8>, %arg1 : tensor<5x2x6x7xi8>, %arg2 : tensor<i8>, %arg3 : tensor<i8>) -> tensor<*xi32> {
  %0 = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", group = 1 : si64, dilations = [2, 3], strides = [2, 2]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x7xi8>, tensor<i8>, tensor<i8>) -> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_convinteger_10
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", dilations = [2, 3], group = 1 : si64, kernel_shape = [6, 7], pads = [0, 0, 0, 0], strides = [2, 2]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x7xi8>, tensor<i8>, tensor<i8>) -> tensor<1x5x11x23xi32>
  // CHECK: return [[RES_ATTR]] : tensor<1x5x11x23xi32>
}

/// dilations attribute with auto_pad set to SAME_UPPER.

func @test_convinteger_11(%arg0 : tensor<1x2x32x64xi8>, %arg1 : tensor<5x2x6x7xi8>, %arg2 : tensor<i8>, %arg3 : tensor<i8>) -> tensor<*xi32> {
  %0 = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "SAME_UPPER", group = 1 : si64, dilations = [2, 3]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x7xi8>, tensor<i8>, tensor<i8>) -> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_convinteger_11
  // CHECK: [[RES_ATTR:%.+]] = "onnx.ConvInteger"(%arg0, %arg1, %arg2, %arg3) {auto_pad = "NOTSET", dilations = [2, 3], group = 1 : si64, kernel_shape = [6, 7], pads = [5, 9, 5, 9], strides = [1, 1]} : (tensor<1x2x32x64xi8>, tensor<5x2x6x7xi8>, tensor<i8>, tensor<i8>) -> tensor<1x5x32x64xi32>
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
  %0 = "onnx.Gather"(%arg0, %arg1) {axis = 0 : si64} : (tensor<3x3xf32>, tensor<1x2xi64>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_gather_axis0
  // CHECK: [[RES:%.+]] = "onnx.Gather"(%arg0, %arg1) {axis = 0 : si64} : (tensor<3x3xf32>, tensor<1x2xi64>) -> tensor<1x2x3xf32>
  // CHECK: return [[RES]] : tensor<1x2x3xf32>
}

// -----

func @test_gather_axis1(%arg0 : tensor<3x3xf32>, %arg1 : tensor<1x2xi64>) -> tensor<*xf32> {
  %0 = "onnx.Gather"(%arg0, %arg1) {axis = 1 : si64} : (tensor<3x3xf32>, tensor<1x2xi64>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_gather_axis1
  // CHECK: [[RES:%.+]] = "onnx.Gather"(%arg0, %arg1) {axis = 1 : si64} : (tensor<3x3xf32>, tensor<1x2xi64>) -> tensor<3x1x2xf32>
  // CHECK: return [[RES]] : tensor<3x1x2xf32>
}

// -----

func @test_gather_negative_axis(%arg0 : tensor<3x3xf32>, %arg1 : tensor<1x2xi64>) -> tensor<*xf32> {
  %0 = "onnx.Gather"(%arg0, %arg1) {axis = -1 : si64} : (tensor<3x3xf32>, tensor<1x2xi64>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_gather_negative_axis
  // CHECK: [[RES:%.+]] = "onnx.Gather"(%arg0, %arg1) {axis = -1 : si64} : (tensor<3x3xf32>, tensor<1x2xi64>) -> tensor<3x1x2xf32>
  // CHECK: return [[RES]] : tensor<3x1x2xf32>
}


// -----

func @test_gather_nd_1(%arg0 : tensor<2x2xf32>, %arg1 : tensor<2x2xi64>) -> tensor<*xf32> {
  %0 = "onnx.GatherND"(%arg0, %arg1) {batch_dims = 0 : si64} : (tensor<2x2xf32>, tensor<2x2xi64>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_gather_nd_1
  // CHECK: [[RES:%.+]] = "onnx.GatherND"(%arg0, %arg1) {batch_dims = 0 : si64} : (tensor<2x2xf32>, tensor<2x2xi64>) -> tensor<2xf32>
  // CHECK: return [[RES]] : tensor<2xf32>
}

// -----

func @test_gather_nd_2(%arg0 : tensor<2x2xf32>, %arg1 : tensor<2x1xi64>) -> tensor<*xf32> {
  %0 = "onnx.GatherND"(%arg0, %arg1) {batch_dims = 0 : si64} : (tensor<2x2xf32>, tensor<2x1xi64>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_gather_nd_2
  // CHECK: [[RES:%.+]] = "onnx.GatherND"(%arg0, %arg1) {batch_dims = 0 : si64} : (tensor<2x2xf32>, tensor<2x1xi64>) -> tensor<2x2xf32>
  // CHECK: return [[RES]] : tensor<2x2xf32>
}

// -----

func @test_gather_nd_3(%arg0 : tensor<2x2x2xf32>, %arg1 : tensor<2x2xi64>) -> tensor<*xf32> {
  %0 = "onnx.GatherND"(%arg0, %arg1) {batch_dims = 0 : si64} : (tensor<2x2x2xf32>, tensor<2x2xi64>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_gather_nd_3
  // CHECK: [[RES:%.+]] = "onnx.GatherND"(%arg0, %arg1) {batch_dims = 0 : si64} : (tensor<2x2x2xf32>, tensor<2x2xi64>) -> tensor<2x2xf32>
  // CHECK: return [[RES]] : tensor<2x2xf32>
}

// -----

func @test_gather_nd_4(%arg0 : tensor<2x2x2xf32>, %arg1 : tensor<2x1x2xi64>) -> tensor<*xf32> {
  %0 = "onnx.GatherND"(%arg0, %arg1) {batch_dims = 0 : si64} : (tensor<2x2x2xf32>, tensor<2x1x2xi64>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_gather_nd_4
  // CHECK: [[RES:%.+]] = "onnx.GatherND"(%arg0, %arg1) {batch_dims = 0 : si64} : (tensor<2x2x2xf32>, tensor<2x1x2xi64>) -> tensor<2x1x2xf32>
  // CHECK: return [[RES]] : tensor<2x1x2xf32>
}

// -----

func @test_gather_nd_5(%arg0 : tensor<2x2x2xf32>, %arg1 : tensor<2x1xi64>) -> tensor<*xf32> {
  %0 = "onnx.GatherND"(%arg0, %arg1) {batch_dims = 1 : si64} : (tensor<2x2x2xf32>, tensor<2x1xi64>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_gather_nd_5
  // CHECK: [[RES:%.+]] = "onnx.GatherND"(%arg0, %arg1) {batch_dims = 1 : si64} : (tensor<2x2x2xf32>, tensor<2x1xi64>) -> tensor<2x2xf32>
  // CHECK: return [[RES]] : tensor<2x2xf32>
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

//===----------------------------------------------------------------------===//
/// Test the behavior of the DepthToSpace operator.
//===----------------------------------------------------------------------===//

func @test_depth_to_space(%arg0 : tensor<1x256x8x16xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.DepthToSpace"(%arg0) {blocksize = 4 : si64} : (tensor<1x256x8x16xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_depth_to_space
  // CHECK: [[RES:%.+]] = "onnx.DepthToSpace"(%arg0) {blocksize = 4 : si64} : (tensor<1x256x8x16xf32>) -> tensor<1x16x32x64xf32>
  // CHECK: return [[RES]] : tensor<1x16x32x64xf32>
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
  %0 = "onnx.ReduceMean"(%arg0) {axes = [-1], keepdims = 1 : si64} : (tensor<1x2x3x4xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reduce_mean_1
  // CHECK: [[RES:%.+]] = "onnx.ReduceMean"(%arg0) {axes = [-1], keepdims = 1 : si64} : (tensor<1x2x3x4xf32>) -> tensor<1x2x3x1xf32>
  // CHECK: return [[RES]] : tensor<1x2x3x1xf32>
}

// -----

func @test_reduce_mean_2(%arg0: tensor<1x2x3x4xf32>) -> tensor<*xf32> {
  %0 = "onnx.ReduceMean"(%arg0) {axes = [2], keepdims = 1 : si64} : (tensor<1x2x3x4xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reduce_mean_2
  // CHECK: [[RES:%.+]] = "onnx.ReduceMean"(%arg0) {axes = [2], keepdims = 1 : si64} : (tensor<1x2x3x4xf32>) -> tensor<1x2x1x4xf32>
  // CHECK: return [[RES]] : tensor<1x2x1x4xf32>
}

// -----

func @test_reduce_mean_3(%arg0: tensor<1x2x3x4xf32>) -> tensor<*xf32> {
  %0 = "onnx.ReduceMean"(%arg0) {axes = [-1], keepdims = 0 : si64} : (tensor<1x2x3x4xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reduce_mean_3
  // CHECK: [[RES:%.+]] = "onnx.ReduceMean"(%arg0) {axes = [-1], keepdims = 0 : si64} : (tensor<1x2x3x4xf32>) -> tensor<1x2x3xf32>
  // CHECK: return [[RES]] : tensor<1x2x3xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// Test shape inference for ReduceSum.
//===----------------------------------------------------------------------===//

func @test_reduce_sum_1(%arg0: tensor<1x2x3x4xf32>) -> tensor<*xf32> {
  %cst = "onnx.Constant"() {value = dense<[-1]> : tensor<1xi64> } : () -> tensor<1xi64>
  %0 = "onnx.ReduceSum"(%arg0, %cst) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x2x3x4xf32>, tensor<1xi64>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reduce_sum_1
  // CHECK-NEXT [[CST:%.+]] = "onnx.Constant"() {value = dense<-1> : tensor<1xi64>} : () -> tensor<1xi64>
  // CHECK-NEXT [[RES:%.+]] = "onnx.ReduceSum"(%arg0, [[CST]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x2x3x4xf32>, tensor<1xi64>) -> tensor<1x2x3x1xf32>
  // CHECK-NEXT return [[RES]] : tensor<1x2x3x1xf32>
}

// -----

func @test_reduce_sum_2(%arg0: tensor<1x2x3x4xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.ReduceSum"(%arg0, %cst) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x2x3x4xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reduce_sum_2
  // CHECK-NEXT [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none
  // CHECK-NEXT [[RES:%.+]] = "onnx.ReduceSum"(%arg0, [[CST]]) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x2x3x4xf32>, none) -> tensor<1x1x1x1xf32>
  // CHECK-NEXT return [[RES]] : tensor<1x1x1x1xf32>
}

// -----

func @test_reduce_sum_3(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<2xi64>) -> tensor<*xf32> {
  %0 = "onnx.ReduceSum"(%arg0, %arg1) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x2x3x4xf32>, tensor<2xi64>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reduce_sum_3
  // CHECK-NEXT [[RES:%.+]] = "onnx.ReduceSum"(%arg0, %arg1) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x2x3x4xf32>, tensor<2xi64>) -> tensor<?x?x?x?xf32>
  // CHECK-NEXT return [[RES]] : tensor<?x?x?x?xf32>
}

// -----

func @test_reduce_sum_4(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<2xi64>) -> tensor<*xf32> {
  %0 = "onnx.ReduceSum"(%arg0, %arg1) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x2x3x4xf32>, tensor<2xi64>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reduce_sum_4
  // CHECK-NEXT [[RES:%.+]] = "onnx.ReduceSum"(%arg0, %arg1) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x2x3x4xf32>, tensor<2xi64>) -> tensor<?x?xf32>
  // CHECK-NEXT return [[RES]] : tensor<*xf32>
}

// -----

func @test_reduce_sum_5(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<?xi64>) -> tensor<*xf32> {
  %0 = "onnx.ReduceSum"(%arg0, %arg1) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x2x3x4xf32>, tensor<?xi64>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reduce_sum_5
  // CHECK-NEXT [[RES:%.+]] = "onnx.ReduceSum"(%arg0, %arg1) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x2x3x4xf32>, tensor<?xi64>) -> tensor<*xf32>
  // CHECK-NEXT return [[RES]] : tensor<*xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// Test shape inference for Dropout.
//===----------------------------------------------------------------------===//

func @test_dropout(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xi1>) -> (tensor<*xf32>, tensor<*xi1>) {
  %output, %mask = "onnx.Dropout"(%arg0, %arg1, %arg2) {ratio =  1.000000e-01 : f32} : (tensor<1x2x3x4xf32>, tensor<1xf32>, tensor<1xi1>) -> (tensor<*xf32>, tensor<*xi1>)
  "std.return"(%output, %mask) : (tensor<*xf32>, tensor<*xi1>) -> ()

  // CHECK-LABEL: test_dropout
  // CHECK: [[RES:%.+]], [[MASK:%.+]] = "onnx.Dropout"(%arg0, %arg1, %arg2) {ratio =  1.000000e-01 : f32} : (tensor<1x2x3x4xf32>, tensor<1xf32>, tensor<1xi1>) -> (tensor<1x2x3x4xf32>, tensor<1x2x3x4xi1>)
  // CHECK: return [[RES]], [[MASK]] : tensor<1x2x3x4xf32>, tensor<1x2x3x4xi1>
}

// -----

//===----------------------------------------------------------------------===//
/// Test shape inference for OneHotEncoder.
//===----------------------------------------------------------------------===//

func @test_onehotencoder_string1 (%arg0: tensor<20x1x!onnx.String>) -> tensor<*xf32> {
  %0 = "onnx.OneHotEncoder"(%arg0) {cats_strings = ["female", "male"], zeros = 1 : si64} : (tensor<20x1x!onnx.String>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_onehotencoder_string1
  // CHECK: [[RES:%.+]] = "onnx.OneHotEncoder"(%arg0) {cats_strings = ["female", "male"], zeros = 1 : si64} : (tensor<20x1x!onnx.String>) -> tensor<20x1x2xf32>
  // CHECK: return [[RES]] : tensor<20x1x2xf32>
}

// -----

func @test_onehotencoder_string2 (%arg0: tensor<20x2x!onnx.String>) -> tensor<*xf32> {
  %0 = "onnx.OneHotEncoder"(%arg0) {cats_strings = ["female", "male"], zeros = 1 : si64} : (tensor<20x2x!onnx.String>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_onehotencoder_string2
  // CHECK: [[RES:%.+]] = "onnx.OneHotEncoder"(%arg0) {cats_strings = ["female", "male"], zeros = 1 : si64} : (tensor<20x2x!onnx.String>) -> tensor<20x2x2xf32>
  // CHECK: return [[RES]] : tensor<20x2x2xf32>
}

// -----

func @test_onehotencoder_float1(%arg0: tensor<20x1xf32>) -> tensor<*xf32> {
  %0 = "onnx.OneHotEncoder"(%arg0) {cats_strings = ["female", "male"], cats_int64s = [1, 2, 4], zeros = 1 : si64} : (tensor<20x1xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_onehotencoder_float1
  // CHECK: [[RES:%.+]] = "onnx.OneHotEncoder"(%arg0) {cats_int64s = [1, 2, 4], cats_strings = ["female", "male"], zeros = 1 : si64} : (tensor<20x1xf32>) -> tensor<20x1x3xf32>
  // CHECK: return [[RES]] : tensor<20x1x3xf32>
}

// -----

func @test_onehotencoder_float2(%arg0: tensor<20x2x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.OneHotEncoder"(%arg0) {cats_strings = ["female", "male"], cats_int64s = [1, 2, 4], zeros = 1 : si64} : (tensor<20x2x3xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_onehotencoder_float2
  // CHECK: [[RES:%.+]] = "onnx.OneHotEncoder"(%arg0) {cats_int64s = [1, 2, 4], cats_strings = ["female", "male"], zeros = 1 : si64} : (tensor<20x2x3xf32>) -> tensor<20x2x3x3xf32>
  // CHECK: return [[RES]] : tensor<20x2x3x3xf32>
}

// -----

func @test_size(%arg0: tensor<*xf32>) -> tensor<*xi64> {
  %0 = "onnx.Size"(%arg0) : (tensor<*xf32>) -> tensor<*xi64>
  "std.return"(%0) : (tensor<*xi64>) -> ()

  // CHECK-LABEL: test_size
  // CHECK: [[RES:%.+]] = "onnx.Size"(%arg0) : (tensor<*xf32>) -> tensor<i64>
  // CHECK: return [[RES]] : tensor<i64>
}

// -----

func @test_less(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<*xi1> {
  %0 = "onnx.Less"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<*xi1>
  return %0 : tensor<*xi1>

  // CHECK-LABEL: test_less
  // CHECK: {{.*}} = "onnx.Less"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xi1>
}

// -----

func @test_less_broadcast(%arg0: tensor<3x4x5xf32>, %arg1: tensor<5xf32>) -> tensor<*xi1> {
  %0 = "onnx.Less"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<5xf32>) -> tensor<*xi1>
  return %0 : tensor<*xi1>

  // CHECK-LABEL: test_less_broadcast
  // CHECK: {{.*}} = "onnx.Less"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<5xf32>) -> tensor<3x4x5xi1>
}

// -----

func @test_less_unknown_dims_1(%arg0: tensor<3x4x5xf32>, %arg1: tensor<?x4x5xf32>) -> tensor<*xi1> {
  %0 = "onnx.Less"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<?x4x5xf32>) -> tensor<*xi1>
  return %0 : tensor<*xi1>

  // CHECK-LABEL: test_less_unknown_dims_1
  // CHECK: {{.*}} = "onnx.Less"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<?x4x5xf32>) -> tensor<3x4x5xi1>
}

// -----

func @test_less_unknown_dims_2(%arg0: tensor<?x?x5xf32>, %arg1: tensor<?x4x5xf32>) -> tensor<*xi1> {
  %0 = "onnx.Less"(%arg0, %arg1) : (tensor<?x?x5xf32>, tensor<?x4x5xf32>) -> tensor<*xi1>
  return %0 : tensor<*xi1>

  // CHECK-LABEL: test_less_unknown_dims_2
  // CHECK: {{.*}} = "onnx.Less"(%arg0, %arg1) : (tensor<?x?x5xf32>, tensor<?x4x5xf32>) -> tensor<?x4x5xi1>
}

// -----

func @test_clip2(%arg0: tensor<3xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<3xf32> attributes {input_names = ["x", "min", "max"], output_names = ["y"]} {
  %0 = "onnx.Clip"(%arg0, %arg1, %arg2) : (tensor<3xf32>, tensor<f32>, tensor<f32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>

// CHECK-LABEL:  func @test_clip2
// CHECK-SAME:   ([[INPUT_:%.+]]: tensor<3xf32>, [[MIN_:%.+]]: tensor<f32>, [[MAX_:%.+]]: tensor<f32>) -> tensor<3xf32> attributes {input_names = ["x", "min", "max"], output_names = ["y"]} {
// CHECK:           [[RES_:%.+]] = "onnx.Clip"([[INPUT_]], [[MIN_]], [[MAX_]]) : (tensor<3xf32>, tensor<f32>, tensor<f32>) -> tensor<3xf32>
// CHECK:           return [[RES_]] : tensor<3xf32>
// CHECK:         }
  }

// -----

// COM: Check PRelu without broadcasting.
func @test_prelu(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

  // CHECK-LABEL: func @test_prelu
  // CHECK: {{.*}} = "onnx.PRelu"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xf32>
  // CHECK: return {{.*}} : tensor<3x4x5xf32>
}

// -----

// COM: Check PRelu with unidirectional broadcasting.
func @test_prelu_broadcast(%arg0: tensor<3x4x5xf32>, %arg1: tensor<5xf32>) -> tensor<*xf32> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<5xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

  // CHECK-LABEL: func @test_prelu_broadcast
  // CHECK: {{.*}} = "onnx.PRelu"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<5xf32>) -> tensor<3x4x5xf32>
  // CHECK: return {{.*}} : tensor<3x4x5xf32>
}

// -----

// COM: Check PRelu with unidirectional broadcasting and unknown dimensions.
// COM: Because of unidirectional broadcasting, always get constant dimensions from X even thought their values are 1.
func @test_prelu_broadcast_unknown_dims(%arg0: tensor<3x1x5xf32>, %arg1: tensor<3x?x1xf32>) -> tensor<*xf32> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<3x1x5xf32>, tensor<3x?x1xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
  // CHECK-LABEL: func @test_prelu_broadcast_unknown_dims
  // CHECK: {{.*}} = "onnx.PRelu"(%arg0, %arg1) : (tensor<3x1x5xf32>, tensor<3x?x1xf32>) -> tensor<3x1x5xf32>
  // CHECK: return {{.*}} : tensor<3x1x5xf32>
}

// -----

// COM: Check PRelu with unidirectional broadcasting and unknown dimensions.
// COM: If X's dimensions are unknown, get dimensions from slope whenever they are non-zero constants.
func @test_prelu_broadcast_unknown_dims1(%arg0: tensor<?x1x?xf32>, %arg1: tensor<?x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<?x1x?xf32>, tensor<?x5xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
  // CHECK-LABEL: func @test_prelu_broadcast_unknown_dims1
  // CHECK: {{.*}} = "onnx.PRelu"(%arg0, %arg1) : (tensor<?x1x?xf32>, tensor<?x5xf32>) -> tensor<?x1x5xf32>
  // CHECK: return {{.*}} : tensor<?x1x5xf32>
}

//===----------------------------------------------------------------------===//
/// Test shape inference for LoopOp.
//===----------------------------------------------------------------------===//

// -----

func @test_loop_simple_no_scan_main_graph(%arg0: tensor<i64>, %arg1: tensor<i1>, %arg2: tensor<1xi64>) -> tensor<*xi64> {
  %0 = "onnx.Loop"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<*xi64>, %arg4: tensor<*xi1>, %arg5: tensor<*xi64>):
    %1 = "onnx.Identity"(%arg4) : (tensor<*xi1>) -> tensor<*xi1>
    %2 = "onnx.Add"(%arg5, %arg3) : (tensor<*xi64>, tensor<*xi64>) -> tensor<*xi64>
    onnx.Return %1, %2 : tensor<*xi1>, tensor<*xi64>
  }) : (tensor<i64>, tensor<i1>, tensor<1xi64>) -> tensor<*xi64>
  return %0 : tensor<*xi64>
// CHECK-LABEL:   func @test_loop_simple_no_scan_main_graph
// CHECK-SAME:     ([[TRIP_COUNT:%.+]]: tensor<i64>, [[COND:%.+]]: tensor<i1>, [[Y_INIT:%.+]]: tensor<1xi64>) -> tensor<1xi64> {
// CHECK:           [[Y_FINAL:%.+]] = "onnx.Loop"([[TRIP_COUNT]], [[COND]], [[Y_INIT]]) ({
// CHECK:           ^bb0([[I:%.+]]: tensor<i64>, [[BODY_COND:%.+]]: tensor<i1>, [[Y_PREV:%.+]]: tensor<1xi64>):
// CHECK:             [[NEXT_COND:%.+]] = "onnx.Identity"([[BODY_COND]]) : (tensor<i1>) -> tensor<i1>
// CHECK:             [[Y_CURR:%.+]] = "onnx.Add"([[Y_PREV]], [[I]]) : (tensor<1xi64>, tensor<i64>) -> tensor<1xi64>
// CHECK:             onnx.Return [[NEXT_COND]], [[Y_CURR]] : tensor<i1>, tensor<1xi64>
// CHECK:           }) : (tensor<i64>, tensor<i1>, tensor<1xi64>) -> tensor<1xi64>
// CHECK:           return [[Y_FINAL]] : tensor<1xi64>
// CHECK:         }
}


func @test_loop_simple_one_scan_main_graph(%arg0: tensor<i64>, %arg1: tensor<i1>, %arg2: tensor<1xi64>) ->(tensor<*xi64>, tensor<*xi64>) { %0:2 = "onnx.Loop"(%arg0, %arg1, %arg2) ({
  ^bb0(%body_arg0: tensor<*xi64>, %body_arg1: tensor<*xi1>, %body_arg2: tensor<*xi64>):
  %body_0 = "onnx.Identity"(%body_arg1) : (tensor<*xi1>) -> tensor<*xi1>
  %body_1 = "onnx.Add"(%body_arg2, %body_arg0) : (tensor<*xi64>, tensor<*xi64>) -> tensor<*xi64>
  %body_2 = "onnx.Identity"(%body_1) : (tensor<*xi64>) -> tensor<*xi64>
  onnx.Return %body_0, %body_1, %body_2 : tensor<*xi1>, tensor<*xi64>, tensor<*xi64>
  }) : (tensor<i64>, tensor<i1>, tensor<1xi64>) -> (tensor<*xi64>, tensor<*xi64>)
  return %0#0, %0#1 : tensor<*xi64>, tensor<*xi64>
  // CHECK-LABEL:       func @test_loop_simple_one_scan_main_graph
  // CHECK-SAME:     ([[TRIP_COUNT:%.+]]: tensor<i64>, [[COND:%.+]]: tensor<i1>, [[Y_INIT:%.+]]: tensor<1xi64>) -> (tensor<1xi64>, tensor<?x1xi64>) {
  // CHECK:           [[LOOP_OUT:%.+]]:2 = "onnx.Loop"([[TRIP_COUNT]], [[COND]], [[Y_INIT]]) ({
  // CHECK:           ^bb0([[I:%.+]]: tensor<i64>, [[BODY_COND:%.+]]: tensor<i1>, [[Y_PREV:%.+]]: tensor<1xi64>):
  // CHECK:             [[COND_NEXT:%.+]] = "onnx.Identity"([[BODY_COND]]) : (tensor<i1>) -> tensor<i1>
  // CHECK:             [[Y_CURR:%.+]] = "onnx.Add"([[Y_PREV]], [[I]]) : (tensor<1xi64>, tensor<i64>) -> tensor<1xi64>
  // CHECK:             [[Y_CURR_SCAN:%.+]] = "onnx.Identity"([[Y_CURR]]) : (tensor<1xi64>) -> tensor<1xi64>
  // CHECK:             onnx.Return [[COND_NEXT]], [[Y_CURR]], [[Y_CURR_SCAN]] : tensor<i1>, tensor<1xi64>, tensor<1xi64>
  // CHECK:           }) : (tensor<i64>, tensor<i1>, tensor<1xi64>) -> (tensor<1xi64>, tensor<?x1xi64>)
  // CHECK:           return [[LOOP_OUT]]#0, [[LOOP_OUT]]#1 : tensor<1xi64>, tensor<?x1xi64>
  // CHECK:         }
}

func @test_loop_multi_scan_main_graph(%arg0: tensor<i64>, %arg1: tensor<i1>, %arg2: tensor<1xi64>, %arg3: tensor<1xf32>) -> (tensor<*xi64>, tensor<*xf32>, tensor<*xi64>, tensor<*xf32>) {
  %0:4 = "onnx.Loop"(%arg0, %arg1, %arg2, %arg3) ({
  ^bb0(%body_arg0: tensor<*xi64>, %body_arg1: tensor<*xi1>, %body_arg2: tensor<*xi64>, %body_arg3: tensor<*xf32>):
  %body_0 = "onnx.Identity"(%body_arg1) : (tensor<*xi1>) -> tensor<*xi1>
  %body_1 = "onnx.Add"(%body_arg2, %body_arg0) : (tensor<*xi64>, tensor<*xi64>) -> tensor<*xi64>
  %body_2 = "onnx.Identity"(%body_1) : (tensor<*xi64>) -> tensor<*xi64>
  %body_3 = "onnx.Add"(%body_arg3, %body_arg3) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %body_4 = "onnx.Identity"(%body_3) : (tensor<*xf32>) -> tensor<*xf32>
  onnx.Return %body_0, %body_1, %body_3, %body_2, %body_4 : tensor<*xi1>, tensor<*xi64>, tensor<*xf32>, tensor<*xi64>, tensor<*xf32>
}) : (tensor<i64>, tensor<i1>, tensor<1xi64>, tensor<1xf32>) -> (tensor<*xi64>, tensor<*xf32>, tensor<*xi64>, tensor<*xf32>)
  return %0#0, %0#1, %0#2, %0#3 : tensor<*xi64>, tensor<*xf32>, tensor<*xi64>, tensor<*xf32>
  // CHECK-LABEL:       func @test_loop_multi_scan_main_graph
  // CHECK-SAME:     ([[TRIP_COUNT:%.+]]: tensor<i64>, [[COND:%.+]]: tensor<i1>, [[Y_INIT:%.+]]: tensor<1xi64>, [[Z_INIT:%.+]]: tensor<1xf32>) -> (tensor<1xi64>, tensor<1xf32>, tensor<?x1xi64>, tensor<?x1xf32>) {
  // CHECK:           [[LOOP_OUT:%.+]]:4 = "onnx.Loop"([[TRIP_COUNT]], [[COND]], [[Y_INIT]], [[Z_INIT]]) ({
  // CHECK:           ^bb0([[I:%.+]]: tensor<i64>, [[BODY_COND:%.+]]: tensor<i1>, [[Y_PREV:%.+]]: tensor<1xi64>, [[Z_PREV:%.+]]: tensor<1xf32>):
  // CHECK:             [[COND_NEXT:%.+]] = "onnx.Identity"([[BODY_COND]]) : (tensor<i1>) -> tensor<i1>
  // CHECK:             [[Y_CURR:%.+]] = "onnx.Add"([[Y_PREV]], [[I:%.+]]) : (tensor<1xi64>, tensor<i64>) -> tensor<1xi64>
  // CHECK:             [[Y_CURR_SCAN:%.+]] = "onnx.Identity"([[Y_CURR]]) : (tensor<1xi64>) -> tensor<1xi64>
  // CHECK:             [[Z_CURR:%.+]] = "onnx.Add"([[Z_PREV]], [[Z_PREV]]) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  // CHECK:             [[Z_CURR_SCAN:%.+]] = "onnx.Identity"([[Z_CURR]]) : (tensor<1xf32>) -> tensor<1xf32>
  // CHECK:             onnx.Return [[COND_NEXT]], [[Y_CURR]], [[Z_CURR]], [[Y_CURR_SCAN]], [[Z_CURR_SCAN]] : tensor<i1>, tensor<1xi64>, tensor<1xf32>, tensor<1xi64>, tensor<1xf32>
  // CHECK:           }) : (tensor<i64>, tensor<i1>, tensor<1xi64>, tensor<1xf32>) -> (tensor<1xi64>, tensor<1xf32>, tensor<?x1xi64>, tensor<?x1xf32>)
  // CHECK:           return [[LOOP_OUT]]#0, [[LOOP_OUT]]#1, [[LOOP_OUT]]#2, [[LOOP_OUT]]#3 : tensor<1xi64>, tensor<1xf32>, tensor<?x1xi64>, tensor<?x1xf32>
  // CHECK:         }
}

func @test_scan_simple_main_graph(%arg0: tensor<2xf32>, %arg1: tensor<3x2xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %0:2 = "onnx.Scan"(%arg0, %arg1) ( {
  ^bb0(%arg2: tensor<*xf32>, %arg3: tensor<*xf32>):  // no predecessors
    %1 = "onnx.Add"(%arg2, %arg3) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    onnx.Return %1, %1 : tensor<*xf32>, tensor<*xf32>
  }) {num_scan_inputs = 1 : si64} : (tensor<2xf32>, tensor<3x2xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  return %0#0, %0#1 : tensor<*xf32>, tensor<*xf32>
// CHECK-LABEL:       func @test_scan_simple_main_graph
// CHECK-SAME:     ([[SUM_INIT:%.+]]: tensor<2xf32>, [[TO_SUM:%.+]]: tensor<3x2xf32>) -> (tensor<2xf32>, tensor<3x2xf32>) {
// CHECK:           [[SCAN_OUT:%.+]]:2 = "onnx.Scan"([[SUM_INIT]], [[TO_SUM]]) (
// CHECK:           ^bb0([[SUM_PREV:%.+]]: tensor<2xf32>, [[SUM_CURR:%.+]]: tensor<2xf32>):
// CHECK:             [[ADD:%.+]] = "onnx.Add"([[SUM_PREV]], [[SUM_CURR]]) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
// CHECK:             onnx.Return [[ADD]], [[ADD]] : tensor<2xf32>, tensor<2xf32>
// CHECK:           }) {num_scan_inputs = 1 : si64} : (tensor<2xf32>, tensor<3x2xf32>) -> (tensor<2xf32>, tensor<3x2xf32>)
// CHECK:           return [[SCAN_OUT]]#0, [[SCAN_OUT]]#1 : tensor<2xf32>, tensor<3x2xf32>
// CHECK:         }
// CHECK:       }
}

// -----

func @test_range(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<*xf32> {
  %0 = "onnx.Range"(%arg0, %arg1, %arg2) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

  // CHECK-LABEL: test_range
  // CHECK: {{.*}} = "onnx.Range"(%arg0, %arg1, %arg2) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<?xf32>
}

// -----

func @test_range_float_constant() -> tensor<*xf32> {
  %start = "onnx.Constant"() {value = dense<[1.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  %limit = "onnx.Constant"() {value = dense<[10.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  %delta = "onnx.Constant"() {value = dense<[1.0]> : tensor<1xf32>} : () -> tensor<1xf32>
  %0 = "onnx.Range"(%start, %limit, %delta) : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

  // CHECK-LABEL: test_range_float_constant
  // CHECK: {{.*}} = "onnx.Range"(%0, %1, %2) : (tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<9xf32>
}

// -----

func @test_range_int_constant() -> tensor<*xi32> {
  %start = "onnx.Constant"() {value = dense<[1]> : tensor<1xi32>} : () -> tensor<1xi32>
  %limit = "onnx.Constant"() {value = dense<[10]> : tensor<1xi32>} : () -> tensor<1xi32>
  %delta = "onnx.Constant"() {value = dense<[1]> : tensor<1xi32>} : () -> tensor<1xi32>
  %0 = "onnx.Range"(%start, %limit, %delta) : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<*xi32>
  return %0 : tensor<*xi32>

  // CHECK-LABEL: test_range_int_constant
  // CHECK: {{.*}} = "onnx.Range"(%0, %1, %2) : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<9xi32>
}

//===----------------------------------------------------------------------===//

// -----

// Test Resize

func @test_resize1(%arg0 : tensor<3x4x5x6xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Constant"() {value = dense<[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]> : tensor<8xf32>} : () -> tensor<8xf32>
  %1 = "onnx.Constant"() {value = dense<[1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]> : tensor<4xf32>} : () -> tensor<4xf32>
  %2 = "onnx.Resize"(%arg0, %0, %1, %cst) {coordinate_transformation_mode = "asymmetric", mode = "nearest", nearest_mode = "floor", onnx_node_name = "Resize1"} : (tensor<3x4x5x6xf32>, tensor<8xf32>, tensor<4xf32>, none) -> tensor<*xf32>
  "std.return"(%2) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_resize1
  // CHECK-SAME: ([[ARG:%.+]]: tensor<3x4x5x6xf32>) -> tensor<3x4x10x12xf32> {
  // CHECK: [[CST:%.+]] = "onnx.NoValue"() {value} : () -> none
  // CHECK: [[R0:%.+]] = "onnx.Constant"() {value = dense<[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]> : tensor<8xf32>} : () -> tensor<8xf32>
  // CHECK: [[R1:%.+]] = "onnx.Constant"() {value = dense<[1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]> : tensor<4xf32>} : () -> tensor<4xf32>
  // CHECK: [[R2:%.+]] = "onnx.Resize"([[ARG]], [[R0]], [[R1]], [[CST]]) {coordinate_transformation_mode = "asymmetric", mode = "nearest", nearest_mode = "floor", onnx_node_name = "Resize1"} : (tensor<3x4x5x6xf32>, tensor<8xf32>, tensor<4xf32>, none) -> tensor<3x4x10x12xf32>
}

// -----

  func @test_reversesequence_1(%arg0: tensor<10x30xf32>, %arg1: tensor<30xi64>) -> tensor<*xf32> {
    %0 = "onnx.ReverseSequence"(%arg0, %arg1) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<10x30xf32>, tensor<30xi64>) -> tensor<*xf32>
    return %0 : tensor<*xf32>
// CHECK-LABEL:  @test_reversesequence_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x30xf32>, [[PARAM_1_:%.+]]: tensor<30xi64>) -> tensor<10x30xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.ReverseSequence"([[PARAM_0_]], [[PARAM_1_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<10x30xf32>, tensor<30xi64>) -> tensor<10x30xf32>
// CHECK:           return [[VAR_0_]] : tensor<10x30xf32>
  }


  func @test_reversesequence_2(%arg0: tensor<10x?xf32>, %arg1: tensor<10xi64>) -> tensor<*xf32> {
    %0 = "onnx.ReverseSequence"(%arg0, %arg1) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<10x?xf32>, tensor<10xi64>) -> tensor<*xf32>
    return %0 : tensor<*xf32>
// CHECK-LABEL:  @test_reversesequence_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x?xf32>, [[PARAM_1_:%.+]]: tensor<10xi64>) -> tensor<10x?xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.ReverseSequence"([[PARAM_0_]], [[PARAM_1_]]) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<10x?xf32>, tensor<10xi64>) -> tensor<10x?xf32>
// CHECK:           return [[VAR_0_]] : tensor<10x?xf32>
  }

// COM: Output's shape should be the same as input's shape.
func @test_cumsum(%arg0: tensor<2x3xf64>, %arg1: tensor<i32>) -> tensor<*xf64> {
  %0 = "onnx.CumSum"(%arg0, %arg1) : (tensor<2x3xf64>, tensor<i32>) -> tensor<*xf64>
  return %0 : tensor<*xf64>
  // CHECK-LABEL: test_cumsum
  // CHECK: "onnx.CumSum"(%arg0, %arg1) : (tensor<2x3xf64>, tensor<i32>) -> tensor<2x3xf64>
}

//===----------------------------------------------------------------------===//

// -----

// Test OneHot

func @test_onehot(%arg0: tensor<2x2xi64>, %arg1: tensor<2xf32>) -> tensor<*xf32> {
  %depth = "onnx.Constant"() {value = dense<10> : tensor<i64>} : () -> tensor<i64>
  %0 = "onnx.OneHot"(%arg0, %depth, %arg1) : (tensor<2x2xi64>, tensor<i64>, tensor<2xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

  // CHECK-LABEL: test_onehot
  // CHECK: [[R0:%.+]] = "onnx.Constant"() {value = dense<10> : tensor<i64>} : () -> tensor<i64>
  // CHECK: {{.*}} = "onnx.OneHot"(%arg0, [[R0]], %arg1) : (tensor<2x2xi64>, tensor<i64>, tensor<2xf32>) -> tensor<2x2x10xf32>
}

// -----

func @test_onehot_axis(%arg0: tensor<2x2xi64>, %arg1: tensor<2xf32>) -> tensor<*xf32> {
  %depth = "onnx.Constant"() {value = dense<10.0> : tensor<f32>} : () -> tensor<f32>
  %0 = "onnx.OneHot"(%arg0, %depth, %arg1) {axis = 1 : si64} : (tensor<2x2xi64>, tensor<f32>, tensor<2xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

  // CHECK-LABEL: test_onehot_axis
  // CHECK: [[R0:%.+]] = "onnx.Constant"() {value = dense<1.000000e+01> : tensor<f32>} : () -> tensor<f32>
  // CHECK: {{.*}} = "onnx.OneHot"(%arg0, [[R0]], %arg1) {axis = 1 : si64} : (tensor<2x2xi64>, tensor<f32>, tensor<2xf32>) -> tensor<2x10x2xf32>
}

// -----

func @test_onehot_depth(%arg0: tensor<2x2xi64>, %arg1: tensor<i64>, %arg2: tensor<2xf32>) -> tensor<*xf32> {
  %0 = "onnx.OneHot"(%arg0, %arg1, %arg2) : (tensor<2x2xi64>, tensor<i64>, tensor<2xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

  // CHECK-LABEL: test_onehot_depth
  // CHECK: {{.*}} = "onnx.OneHot"(%arg0, %arg1, %arg2)  : (tensor<2x2xi64>, tensor<i64>, tensor<2xf32>) -> tensor<2x2x?xf32>
}

// -----

func @test_onehot_dynamic(%arg0: tensor<?x2xi64>, %arg1: tensor<i64>, %arg2: tensor<2xf32>) -> tensor<*xf32> {
  %0 = "onnx.OneHot"(%arg0, %arg1, %arg2) {axis = 0 : si64} : (tensor<?x2xi64>, tensor<i64>, tensor<2xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

  // CHECK-LABEL: test_onehot_dynamic
  // CHECK: {{.*}} = "onnx.OneHot"(%arg0, %arg1, %arg2)  {axis = 0 : si64} : (tensor<?x2xi64>, tensor<i64>, tensor<2xf32>) -> tensor<?x?x2xf32>
}

//===----------------------------------------------------------------------===//

// -----

// Test RandomNormal static

func @test_random_normal_static_f16() -> tensor<*xf32> {
  %0 = "onnx.RandomNormal"() {shape = [3, 4, 5], dtype = 0 : si64, mean = 0.0 :f32, scale = 1.0 : f32, seed = 2.0 : f32} : () -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_random_normal_static_f16
  // CHECK: [[R0:%.+]] = "onnx.RandomNormal"() {dtype = 0 : si64, mean = 0.000000e+00 : f32, scale = 1.000000e+00 : f32, seed = 2.000000e+00 : f32, shape = [3, 4, 5]} : () -> tensor<3x4x5xf16>
}

func @test_random_normal_static_f32() -> tensor<*xf32> {
  %0 = "onnx.RandomNormal"() {shape = [3, 4, 5], dtype = 1 : si64, mean = 0.0 :f32, scale = 1.0 : f32, seed = 2.0 : f32} : () -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_random_normal_static_f32
  // CHECK: [[R0:%.+]] = "onnx.RandomNormal"() {dtype = 1 : si64, mean = 0.000000e+00 : f32, scale = 1.000000e+00 : f32, seed = 2.000000e+00 : f32, shape = [3, 4, 5]} : () -> tensor<3x4x5xf32>
}

func @test_random_normal_static_f64() -> tensor<*xf32> {
  %0 = "onnx.RandomNormal"() {shape = [3, 4, 5], dtype = 2 : si64, mean = 0.0 :f32, scale = 1.0 : f32, seed = 2.0 : f32} : () -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_random_normal_static_f64
  // CHECK: [[R0:%.+]] = "onnx.RandomNormal"() {dtype = 2 : si64, mean = 0.000000e+00 : f32, scale = 1.000000e+00 : f32, seed = 2.000000e+00 : f32, shape = [3, 4, 5]} : () -> tensor<3x4x5xf64>
}

//===----------------------------------------------------------------------===//

// -----

// Test RandomNormalLike static

func @test_random_normal_like_static_f16(%arg0: tensor<1x1x28x28xf32>) -> tensor<*xf32> {
  %0 = "onnx.RandomNormalLike"(%arg0) {dtype = 0 : si64, mean = 0.0 :f32, scale = 1.0 : f32, seed = 2.0 : f32} : (tensor<1x1x28x28xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_random_normal_like_static_f16
  // CHECK: [[R0:%.+]] = "onnx.RandomNormalLike"(%arg0) {dtype = 0 : si64, mean = 0.000000e+00 : f32, scale = 1.000000e+00 : f32, seed = 2.000000e+00 : f32} : (tensor<1x1x28x28xf32>) -> tensor<1x1x28x28xf16>
}

func @test_random_normal_like_static_f32(%arg0: tensor<1x1x28x28xf32>) -> tensor<*xf32> {
  %0 = "onnx.RandomNormalLike"(%arg0) {dtype = 1 : si64, mean = 0.0 :f32, scale = 1.0 : f32, seed = 2.0 : f32} : (tensor<1x1x28x28xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_random_normal_like_static_f32
  // CHECK: [[R0:%.+]] = "onnx.RandomNormalLike"(%arg0) {dtype = 1 : si64, mean = 0.000000e+00 : f32, scale = 1.000000e+00 : f32, seed = 2.000000e+00 : f32} : (tensor<1x1x28x28xf32>) -> tensor<1x1x28x28xf32>
}

func @test_random_normal_like_static_f64(%arg0: tensor<1x1x28x28xf32>) -> tensor<*xf32> {
  %0 = "onnx.RandomNormalLike"(%arg0) {dtype = 2 : si64, mean = 0.0 :f32, scale = 1.0 : f32, seed = 2.0 : f32} : (tensor<1x1x28x28xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_random_normal_like_static_f64
  // CHECK: [[R0:%.+]] = "onnx.RandomNormalLike"(%arg0) {dtype = 2 : si64, mean = 0.000000e+00 : f32, scale = 1.000000e+00 : f32, seed = 2.000000e+00 : f32} : (tensor<1x1x28x28xf32>) -> tensor<1x1x28x28xf64>
}

// -----

// Test RandomNormalLike dynamic

func @test_random_normal_like_dynamic_f16(%arg0: tensor<1x?x28x28xf32>) -> tensor<*xf32> {
  %0 = "onnx.RandomNormalLike"(%arg0) {dtype = 0 : si64, mean = 0.0 :f32, scale = 1.0 : f32, seed = 2.0 : f32} : (tensor<1x?x28x28xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_random_normal_like_dynamic_f16
  // CHECK: [[R0:%.+]] = "onnx.RandomNormalLike"(%arg0) {dtype = 0 : si64, mean = 0.000000e+00 : f32, scale = 1.000000e+00 : f32, seed = 2.000000e+00 : f32} : (tensor<1x?x28x28xf32>) -> tensor<1x?x28x28xf16>
}

func @test_random_normal_like_dynamic_f32(%arg0: tensor<1x1x?x28xf32>) -> tensor<*xf32> {
  %0 = "onnx.RandomNormalLike"(%arg0) {dtype = 1 : si64, mean = 0.0 :f32, scale = 1.0 : f32, seed = 2.0 : f32} : (tensor<1x1x?x28xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_random_normal_like_dynamic_f32
  // CHECK: [[R0:%.+]] = "onnx.RandomNormalLike"(%arg0) {dtype = 1 : si64, mean = 0.000000e+00 : f32, scale = 1.000000e+00 : f32, seed = 2.000000e+00 : f32} : (tensor<1x1x?x28xf32>) -> tensor<1x1x?x28xf32>
}

func @test_random_normal_like_dynamic_f64(%arg0: tensor<1x1x28x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.RandomNormalLike"(%arg0) {dtype = 2 : si64, mean = 0.0 :f32, scale = 1.0 : f32, seed = 2.0 : f32} : (tensor<1x1x28x?xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_random_normal_like_dynamic_f64
  // CHECK: [[R0:%.+]] = "onnx.RandomNormalLike"(%arg0) {dtype = 2 : si64, mean = 0.000000e+00 : f32, scale = 1.000000e+00 : f32, seed = 2.000000e+00 : f32} : (tensor<1x1x28x?xf32>) -> tensor<1x1x28x?xf64>
}

// -----

// Test RandomNormalLike missing dtype

func @test_random_normal_like_type_default1(%arg0: tensor<1x1x28x28xf64>) -> tensor<*xf32> {
  %0 = "onnx.RandomNormalLike"(%arg0) {mean = 0.0 :f32, scale = 1.0 : f32, seed = 2.0 : f32} : (tensor<1x1x28x28xf64>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_random_normal_like_type_default1
  // CHECK: [[R0:%.+]] = "onnx.RandomNormalLike"(%arg0) {mean = 0.000000e+00 : f32, scale = 1.000000e+00 : f32, seed = 2.000000e+00 : f32} : (tensor<1x1x28x28xf64>) -> tensor<1x1x28x28xf64>
}

// -----

//===----------------------------------------------------------------------===//
// Test NonMaxSuppression

func @test_nonmaxsuppression(%arg0: tensor<1x6x4xf32>, %arg1: tensor<1x1x6xf32>, %arg2: tensor<1xi64>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<*xi64> {
    %0 = "onnx.NonMaxSuppression"(%arg0, %arg1, %arg2, %arg3, %arg4) {center_point_box = 1 : si64} : (tensor<1x6x4xf32>, tensor<1x1x6xf32>, tensor<1xi64>, tensor<1xf32>, tensor<1xf32>) -> tensor<*xi64>
    return %0 : tensor<*xi64>
    // CHECK-LABEL: test_nonmaxsuppression
    // CHECK: [[RES:%.+]] = "onnx.NonMaxSuppression"(%arg0, %arg1, %arg2, %arg3, %arg4) {center_point_box = 1 : si64} : (tensor<1x6x4xf32>, tensor<1x1x6xf32>, tensor<1xi64>, tensor<1xf32>, tensor<1xf32>) -> tensor<?x3xi64>
    // CHECK: return [[RES]] : tensor<?x3xi64>
}

// -----

//===----------------------------------------------------------------------===//
// Test compress

func @compress_axis0(%arg0: tensor<3x2xf32>, %arg1: tensor<3xi1>) -> tensor<?x?xf32> {
  %0 = "onnx.Compress"(%arg0, %arg1) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<3xi1>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>

// mlir2FileCheck.py -a'["input", "condition"]'
// CHECK-LABEL:  func @compress_axis0
// CHECK-SAME:   ([[INPUT_:%.+]]: tensor<3x2xf32>, [[CONDITION_:%.+]]: tensor<3xi1>) -> tensor<?x2xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Compress"([[INPUT_]], [[CONDITION_]]) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<3xi1>) -> tensor<?x2xf32>
// CHECK:           return [[VAR_0_]] : tensor<?x2xf32>
// CHECK:         }
}

// -----

func @compress_axis1(%arg0: tensor<3x2xf32>, %arg1: tensor<3xi1>) -> tensor<?x?xf32> {
    %0 = "onnx.Compress"(%arg0, %arg1) {axis = 1 : si64} : (tensor<3x2xf32>, tensor<3xi1>) -> tensor<?x?xf32>
    return %0 : tensor<?x?xf32>
// mlir2FileCheck.py -a'["input", "condition"]'
// CHECK-LABEL:  func @compress_axis1
// CHECK-SAME:   ([[INPUT_:%.+]]: tensor<3x2xf32>, [[CONDITION_:%.+]]: tensor<3xi1>) -> tensor<3x?xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Compress"([[INPUT_]], [[CONDITION_]]) {axis = 1 : si64} : (tensor<3x2xf32>, tensor<3xi1>) -> tensor<3x?xf32>
// CHECK:           return [[VAR_0_]] : tensor<3x?xf32>
// CHECK:         }
}

// -----

func @compress_no_axis(%arg0: tensor<3x2xf32>, %arg1: tensor<3xi1>) -> tensor<?x?xf32> {
    %0 = "onnx.Compress"(%arg0, %arg1) : (tensor<3x2xf32>, tensor<3xi1>) -> tensor<?x?xf32>
    return %0 : tensor<?x?xf32>

// mlir2FileCheck.py -a'["input", "condition"]'
// CHECK-LABEL:  func @compress_no_axis
// CHECK-SAME:   ([[INPUT_:%.+]]: tensor<3x2xf32>, [[CONDITION_:%.+]]: tensor<3xi1>) -> tensor<?xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Compress"([[INPUT_]], [[CONDITION_]]) : (tensor<3x2xf32>, tensor<3xi1>) -> tensor<?xf32>
// CHECK:           return [[VAR_0_]] : tensor<?xf32>
// CHECK:         }
}

// -----

func @hardmax(%arg0: tensor<3x4x5xf32>) -> tensor<*xf32>{
  %0 = "onnx.Hardmax"(%arg0) {axis = 1 : si64} : (tensor<3x4x5xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
  // CHECK-LABEL: hardmax
  // CHECK: [[RES:%.+]] = "onnx.Hardmax"(%arg0) {axis = 1 : si64} : (tensor<3x4x5xf32>) -> tensor<3x4x5xf32>
  // CHECK: return [[RES]] : tensor<3x4x5xf32>
}

// -----

func @topk_default_axis_minus_one(%X: tensor<3x4x5xf32>, %K: tensor<i64>) -> tensor<*xf32> {
  %value, %indices = "onnx.TopK"(%X, %K) : (tensor<3x4x5xf32>, tensor<i64>) -> (tensor<*xf32>, tensor<*xi64>)
  return %value : tensor<*xf32>
  // CHECK-LABEL: topk_default_axis_minus_one
  // CHECK: {{.*}} = "onnx.TopK"({{.*}}, {{.*}}) : (tensor<3x4x5xf32>, tensor<i64>) -> (tensor<3x4x?xf32>, tensor<3x4x?xi64>)
}

// -----

func @topk_default_axis_one(%X: tensor<3x4x5xf32>, %K: tensor<i64>) -> tensor<*xf32> {
  %value, %indices = "onnx.TopK"(%X, %K) {axis = 1 : si64} : (tensor<3x4x5xf32>, tensor<i64>) -> (tensor<*xf32>, tensor<*xi64>)
  return %value : tensor<*xf32>
  // CHECK-LABEL: topk_default_axis_one
  // CHECK: {{.*}} = "onnx.TopK"({{.*}}, {{.*}}) {axis = 1 : si64} : (tensor<3x4x5xf32>, tensor<i64>) -> (tensor<3x?x5xf32>, tensor<3x?x5xi64>)
}

// -----

func @topk_constant_k(%X: tensor<3x4x5xf32>) -> tensor<*xf32> {
  %K = "onnx.Constant"() {value = dense<2> : tensor<i64>} : () -> tensor<i64>
  %value, %indices = "onnx.TopK"(%X, %K) {axis = 1 : si64} : (tensor<3x4x5xf32>, tensor<i64>) -> (tensor<*xf32>, tensor<*xi64>)
  return %value : tensor<*xf32>
  // CHECK-LABEL: topk_constant_k
  // CHECK: {{.*}} = "onnx.TopK"({{.*}}, {{.*}}) {axis = 1 : si64} : (tensor<3x4x5xf32>, tensor<i64>) -> (tensor<3x2x5xf32>, tensor<3x2x5xi64>)
}

// -----

//===----------------------------------------------------------------------===//
/// Test shape inference for CategoryMapper.
//===----------------------------------------------------------------------===//

func @test_category_mapper_string (%arg0: tensor<20x1x!onnx.String>) -> tensor<*xi64> {
  %0 = "onnx.CategoryMapper"(%arg0) {cats_int64s = [1, 2, 3], cats_strings = ["cat", "dog", "human"], default_int64 = 0 : si64} : (tensor<20x1x!onnx.String>) -> tensor<*xi64>
  "std.return"(%0) : (tensor<*xi64>) -> ()

  // CHECK-LABEL: test_category_mapper_string
  // CHECK: [[RES:%.+]] = "onnx.CategoryMapper"(%arg0) {cats_int64s = [1, 2, 3], cats_strings = ["cat", "dog", "human"], default_int64 = 0 : si64} : (tensor<20x1x!onnx.String>) -> tensor<20x1xi64>
  // CHECK: return [[RES]] : tensor<20x1xi64>
}

// -----

func @test_category_mapper_int64 (%arg0: tensor<20x1xi64>) -> tensor<*x!onnx.String> {
  %0 = "onnx.CategoryMapper"(%arg0) {cats_int64s = [1, 2, 3], cats_strings = ["cat", "dog", "human"], default_string = "unclassified" : !onnx.String} : (tensor<20x1xi64>) -> tensor<*x!onnx.String>
  "std.return"(%0) : (tensor<*x!onnx.String>) -> ()

  // CHECK-LABEL: test_category_mapper_int64
  // CHECK: [[RES:%.+]] = "onnx.CategoryMapper"(%arg0) {cats_int64s = [1, 2, 3], cats_strings = ["cat", "dog", "human"], default_string = "unclassified" : !onnx.String} : (tensor<20x1xi64>) -> tensor<20x1x!onnx.String>
  // CHECK: return [[RES]] : tensor<20x1x!onnx.String>
}

// -----

//===----------------------------------------------------------------------===//
/// Test shape inference for ScatterND.
//===----------------------------------------------------------------------===//
func @test_scatternd_int32(%arg0: tensor<8xi32>) -> tensor<*xi32> {
  %1 = "onnx.Constant"() {value = dense<[[4], [3], [1], [7]]> : tensor<4x1xi64> } : () -> tensor<4x1xi64>
  %2 = "onnx.Constant"() {value = dense<[9, 10, 11, 12]> : tensor<4xi32> } : () -> tensor<4xi32>
  %3 = "onnx.ScatterND"(%arg0, %1, %2) : (tensor<8xi32>, tensor<4x1xi64>, tensor<4xi32>) ->  tensor<*xi32>
  "std.return"(%3) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_scatternd_int32
  // CHECK: [[R1:%.+]] = "onnx.Constant"() {{.*}}
  // CHECK-NEXT: [[R2:%.+]] = "onnx.Constant"() {{.*}}
  // CHECK-NEXT: [[RES:%.+]] = "onnx.ScatterND"(%arg0, [[R1]], [[R2]]) : (tensor<8xi32>, tensor<4x1xi64>, tensor<4xi32>) -> tensor<8xi32>
  // CHECK-NEXT: return [[RES]] : tensor<8xi32>
}

// -----

func @test_scatternd_float32(%arg0: tensor<4x4x4xf32>) -> tensor<*xf32> {
  %1 = "onnx.Constant"() {value = dense<[[0], [2]]> : tensor<2x1xi64> } : () -> tensor<2x1xi64>
  %2 = "onnx.Constant"() {value = dense<[[[5., 5., 5., 5.], [6., 6., 6., 6.], [7., 7., 7., 7.], [8., 8., 8., 8.]],
                                         [[1., 1., 1., 1.], [2., 2., 2., 2.], [3., 3., 3., 3.], [4., 4., 4., 4.]]]> : tensor<2x4x4xf32> } : () -> tensor<2x4x4xf32>
  %3 = "onnx.ScatterND"(%arg0, %1, %2) : (tensor<4x4x4xf32>, tensor<2x1xi64>, tensor<2x4x4xf32>) ->  tensor<*xf32>
  "std.return"(%3) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_scatternd_float32
  // CHECK: [[R1:%.+]] = "onnx.Constant"() {{.*}}
  // CHECK-NEXT: [[R2:%.+]] = "onnx.Constant"() {{.*}}
  // CHECK-NEXT: [[RES:%.+]] = "onnx.ScatterND"(%arg0, [[R1]], [[R2]]) : (tensor<4x4x4xf32>, tensor<2x1xi64>, tensor<2x4x4xf32>) -> tensor<4x4x4xf32>
  // CHECK-NEXT: return [[RES]] : tensor<4x4x4xf32>
}

// -----
func @test_seqence_length(%arg0 : !onnx.Seq<tensor<*xf32>>) -> tensor<*xi64> {
  %0 = "onnx.SequenceLength"(%arg0) : (!onnx.Seq<tensor<*xf32>>) -> tensor<*xi64>
  return %0 : tensor<*xi64>
// mlir2FileCheck.py
// CHECK-LABEL:  func @test_seqence_length
// CHECK-SAME:   ([[PARAM_0_:%.+]]: !onnx.Seq<tensor<*xf32>>) -> tensor<i64> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.SequenceLength"([[PARAM_0_]]) : (!onnx.Seq<tensor<*xf32>>) -> tensor<i64>
// CHECK:           return [[VAR_0_]] : tensor<i64>
}

// -----
func @test_seqence_1(%arg0: tensor<2x4xf32>, %arg1: tensor<2x6xf32>) -> !onnx.Seq<tensor<*xf32>> {
  %0 = "onnx.SequenceEmpty"() : () -> !onnx.Seq<tensor<*xf32>>
  %cst = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.SequenceInsert"(%0, %arg0, %cst) : (!onnx.Seq<tensor<*xf32>>, tensor<2x4xf32>, none) -> !onnx.Seq<tensor<*xf32>>
  %2 = "onnx.SequenceInsert"(%1, %arg1, %cst) : (!onnx.Seq<tensor<*xf32>>, tensor<2x6xf32>, none) -> !onnx.Seq<tensor<*xf32>>
  return %2 : !onnx.Seq<tensor<*xf32>>
// CHECK-LABEL:  func @test_seqence_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4xf32>, [[PARAM_1_:%.+]]: tensor<2x6xf32>) -> !onnx.Seq<tensor<2x?xf32>> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.SequenceEmpty"() : () -> !onnx.Seq<tensor<*xf32>>
// CHECK-DAG:       [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_1_:%.+]] = "onnx.SequenceInsert"([[VAR_0_]], [[PARAM_0_]], [[VAR_cst_]]) : (!onnx.Seq<tensor<*xf32>>, tensor<2x4xf32>, none) -> !onnx.Seq<tensor<2x4xf32>>
// CHECK:           [[VAR_2_:%.+]] = "onnx.SequenceInsert"([[VAR_1_]], [[PARAM_1_]], [[VAR_cst_]]) : (!onnx.Seq<tensor<2x4xf32>>, tensor<2x6xf32>, none) -> !onnx.Seq<tensor<2x?xf32>>
// CHECK:           return [[VAR_2_]] : !onnx.Seq<tensor<2x?xf32>>
}

// -----

func @test_seqence_2(%arg0: tensor<2x4xf32>, %arg1: tensor<3x6xf32>) -> !onnx.Seq<tensor<*xf32>> {
  %0 = "onnx.SequenceEmpty"() : () -> !onnx.Seq<tensor<*xf32>>
  %cst = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.SequenceInsert"(%0, %arg0, %cst) : (!onnx.Seq<tensor<*xf32>>, tensor<2x4xf32>, none) -> !onnx.Seq<tensor<*xf32>>
  %2 = "onnx.SequenceInsert"(%1, %arg1, %cst) : (!onnx.Seq<tensor<*xf32>>, tensor<3x6xf32>, none) -> !onnx.Seq<tensor<*xf32>>
  return %2 : !onnx.Seq<tensor<*xf32>>
// CHECK-LABEL:  func @test_seqence_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4xf32>, [[PARAM_1_:%.+]]: tensor<3x6xf32>) -> !onnx.Seq<tensor<?x?xf32>> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.SequenceEmpty"() : () -> !onnx.Seq<tensor<*xf32>>
// CHECK-DAG:       [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_1_:%.+]] = "onnx.SequenceInsert"([[VAR_0_]], [[PARAM_0_]], [[VAR_cst_]]) : (!onnx.Seq<tensor<*xf32>>, tensor<2x4xf32>, none) -> !onnx.Seq<tensor<2x4xf32>>
// CHECK:           [[VAR_2_:%.+]] = "onnx.SequenceInsert"([[VAR_1_]], [[PARAM_1_]], [[VAR_cst_]]) : (!onnx.Seq<tensor<2x4xf32>>, tensor<3x6xf32>, none) -> !onnx.Seq<tensor<?x?xf32>>
// CHECK:           return [[VAR_2_]] : !onnx.Seq<tensor<?x?xf32>>
}

// -----

func @test_seqence_3(%arg0: tensor<2x4x8xf32>, %arg1: tensor<3x6xf32>) -> !onnx.Seq<tensor<*xf32>> {
  %0 = "onnx.SequenceEmpty"() : () -> !onnx.Seq<tensor<*xf32>>
  %cst = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.SequenceInsert"(%0, %arg0, %cst) : (!onnx.Seq<tensor<*xf32>>, tensor<2x4x8xf32>, none) -> !onnx.Seq<tensor<*xf32>>
  %2 = "onnx.SequenceInsert"(%1, %arg1, %cst) : (!onnx.Seq<tensor<*xf32>>, tensor<3x6xf32>, none) -> !onnx.Seq<tensor<*xf32>>
  return %2 : !onnx.Seq<tensor<*xf32>>
// CHECK-LABEL:  func @test_seqence_3
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x4x8xf32>, [[PARAM_1_:%.+]]: tensor<3x6xf32>) -> !onnx.Seq<tensor<*xf32>> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.SequenceEmpty"() : () -> !onnx.Seq<tensor<*xf32>>
// CHECK-DAG:       [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK:           [[VAR_1_:%.+]] = "onnx.SequenceInsert"([[VAR_0_]], [[PARAM_0_]], [[VAR_cst_]]) : (!onnx.Seq<tensor<*xf32>>, tensor<2x4x8xf32>, none) -> !onnx.Seq<tensor<2x4x8xf32>>
// CHECK:           [[VAR_2_:%.+]] = "onnx.SequenceInsert"([[VAR_1_]], [[PARAM_1_]], [[VAR_cst_]]) : (!onnx.Seq<tensor<2x4x8xf32>>, tensor<3x6xf32>, none) -> !onnx.Seq<tensor<*xf32>>
// CHECK:           return [[VAR_2_]] : !onnx.Seq<tensor<*xf32>>
}

// -----

//===----------------------------------------------------------------------===//
/// Test shape inference for RoiAlign.
//===----------------------------------------------------------------------===//
func @test_roialign(%arg0: tensor<1x2x4x8xf32>, %arg1: tensor<100x4xf32>, %arg2: tensor<100xi64>) -> tensor<*xf32> {
  %0 = "onnx.RoiAlign"(%arg0, %arg1, %arg2) {output_height = 7 : si64, output_width = 7 : si64, sampling_ratio = 2 : si64, spatial_scale = 1.000000e+00 : f32} : (tensor<1x2x4x8xf32>, tensor<100x4xf32>, tensor<100xi64>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
  
  // CHECK-LABEL: func @test_roialign
  // CHECK: [[RES:%.+]] = "onnx.RoiAlign"(%arg0, %arg1, %arg2) {output_height = 7 : si64, output_width = 7 : si64, sampling_ratio = 2 : si64, spatial_scale = 1.000000e+00 : f32} : (tensor<1x2x4x8xf32>, tensor<100x4xf32>, tensor<100xi64>) -> tensor<100x2x7x7xf32>
  // CHECK: return [[RES]] : tensor<100x2x7x7xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// Test shape inference for ScatterElements.
//===----------------------------------------------------------------------===//
func @test_scatterelements(%arg0: tensor<64x25600xf32>, %arg1: tensor<64x100xi64>, %arg2: tensor<64x100xf32>) -> tensor<*xf32> {
  %0 = "onnx.ScatterElements"(%arg0, %arg1, %arg2) {axis = 1 : si64} : (tensor<64x25600xf32>, tensor<64x100xi64>, tensor<64x100xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
  
  // CHECK-LABEL: func @test_scatterelements
  // CHECK: [[RES:%.+]] = "onnx.ScatterElements"(%arg0, %arg1, %arg2) {axis = 1 : si64} : (tensor<64x25600xf32>, tensor<64x100xi64>, tensor<64x100xf32>) -> tensor<64x25600xf32>
  // CHECK: return [[RES]] : tensor<64x25600xf32>
}
