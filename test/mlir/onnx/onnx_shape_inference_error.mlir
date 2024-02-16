// RUN: onnx-mlir-opt --shape-inference %s -split-input-file -verify-diagnostics

//===----------------------------------------------------------------------===//
/// Unsupported configurations for ONNXConvOp.
//===----------------------------------------------------------------------===//

// -----

// Error found in the valid tests, so fails before reaching shape inference.
func.func @unsupport_conv_bad_kernel_shape_attr(%arg0 : tensor<1x2x32x32xf32>, %arg1 : tensor<5x2x7x7xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{Bad kernel_shape value: must be strictly positive}}
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [-1, 7]} : (tensor<1x2x32x32xf32>, tensor<5x2x7x7xf32>, none) -> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()
}

// -----

// Error found in the valid tests, so fails before reaching shape inference.
func.func @unsupport_conv_bad_kernel_shape(%arg0 : tensor<1x2x32x32xf32>, %arg1 : tensor<5x2x0x7xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{Bad spatial filter size: cannot be zero}}
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<1x2x32x32xf32>, tensor<5x2x0x7xf32>, none) -> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()
}

// -----

//===----------------------------------------------------------------------===//
/// Unsupported configurations for ONNXMaxPoolOp.
//===----------------------------------------------------------------------===//

func.func @unsupport_maxpool_column_storage(%arg0 : tensor<5x5x32x32xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{Column major storage order not implemented yet}}
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "VALID", ceil_mode = 0 : si64, kernel_shape = [3, 3], storage_order = 1 : si64} : (tensor<5x5x32x32xf32>) -> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()
}

// -----

//===----------------------------------------------------------------------===//
/// Unsupported configurations for ONNXPowOp.
//===----------------------------------------------------------------------===//

func.func @test_reshape_2D_shape(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<1x2xi64>) -> tensor<*xf32> {
  // expected-error @+1 {{Shape tensor must have rank one}}
  %0 = "onnx.Reshape"(%arg0, %arg1) : (tensor<5x5x1x32xf32>, tensor<1x2xi64>) -> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()
}

// -----

//===----------------------------------------------------------------------===//
/// Errors with RNNOps. Take ONNXLSTMOp as an example.
//===----------------------------------------------------------------------===//

func.func @test_lstm_not_3D_input(%arg0: tensor<4x3xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  // expected-error @+3 {{The first input tensor must have rank 3}}
  // expected-error @+2 {{Failed to scan parameters successfully}}
  // expected-error @+1 {{shape inference failed}}
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  onnx.Return %Y_h : tensor<*xf32>
}

// -----

func.func @test_lstm_not_3D_weight(%arg0: tensor<4x3x2xf32>, %arg1: tensor<12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  // expected-error @+3 {{The second input tensor must have rank 3}}
  // expected-error @+2 {{Failed to scan parameters successfully}}
  // expected-error @+1 {{shape inference failed}}
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  onnx.Return %Y_h : tensor<*xf32>
}

// -----

func.func @test_lstm_not_3D_recurrent(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<12x3xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  // expected-error @+3 {{The third input tensor must have rank 3}}
  // expected-error @+2 {{Failed to scan parameters successfully}}
  // expected-error @+1 {{shape inference failed}}
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<12x3xf32>, none, none, none, none, none) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  onnx.Return %Y_h : tensor<*xf32>
}

// -----

func.func @test_lstm_wrong_direction(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  // expected-error @+3 {{direction attribute must be one of the strings: forward, reverse, and bidirectional}}
  // expected-error @+2 {{Failed to scan parameters successfully}}
  // expected-error @+1 {{shape inference failed}}
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : si64, direction="forwadr"} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  onnx.Return %Y_h : tensor<*xf32>
}

// -----

//===----------------------------------------------------------------------===//
/// Unsupported configurations for ONNXCategoryMapperOp.
//===----------------------------------------------------------------------===//

func.func @test_category_mapper_diff_size_attrs (%arg0: tensor<20x1xi64>) -> tensor<*x!onnx.String> {
  // expected-error @+1 {{cats_int64 and cats_strings should have the same size}}      
  %0 = "onnx.CategoryMapper"(%arg0) {cats_int64s = [1, 2], cats_strings = ["dog"]} : (tensor<20x1xi64>) -> tensor<*x!onnx.String>
  "onnx.Return"(%0) : (tensor<*x!onnx.String>) -> ()
}

// -----

func.func @test_category_mapper_diff_size_attrs (%arg0: tensor<20x1xi32>) -> tensor<*x!onnx.String> {
  // expected-error @+1 {{'onnx.CategoryMapper' op operand #0 must be tensor of string type values or tensor of 64-bit signless integer values, but got 'tensor<20x1xi32>'}}      
  %0 = "onnx.CategoryMapper"(%arg0) {cats_int64s = [1], cats_strings = ["cat"]} : (tensor<20x1xi32>) -> tensor<*x!onnx.String>
  "onnx.Return"(%0) : (tensor<*x!onnx.String>) -> ()
}

// -----
