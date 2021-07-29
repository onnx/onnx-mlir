// RUN: onnx-mlir-opt --shape-inference %s -split-input-file -verify-diagnostics

//===----------------------------------------------------------------------===//
/// Unsupported configurations for ONNXConvOp.
//===----------------------------------------------------------------------===//

// -----

func @unsupport_conv_same_upper_dynamic_X(%arg0 : tensor<1x2x?xf32>, %arg1 : tensor<5x2x6xf32>) -> tensor<*xf32> {
  %cst = constant unit
  // expected-error @+3 {{Conv Pads defined as SAME_UPPER or SAME_LOWER requires compile time X sizes}}
  // expected-error @+2 {{Failed to scan Conv parameters successfully}}
  // expected-error @+1 {{shape inference failed}}
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "SAME_UPPER", group = 1 : si64} : (tensor<1x2x?xf32>, tensor<5x2x6xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
}

// -----

func @unsupport_conv_same_lower_dynamic_X(%arg0 : tensor<1x2x?xf32>, %arg1 : tensor<5x2x6xf32>) -> tensor<*xf32> {
  %cst = constant unit
  // expected-error @+3 {{Conv Pads defined as SAME_UPPER or SAME_LOWER requires compile time X sizes}}
  // expected-error @+2 {{Failed to scan Conv parameters successfully}}
  // expected-error @+1 {{shape inference failed}}
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "SAME_LOWER", group = 1 : si64} : (tensor<1x2x?xf32>, tensor<5x2x6xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
}

// -----

//===----------------------------------------------------------------------===//
/// Unsupported configurations for ONNXPowOp.
//===----------------------------------------------------------------------===//

func @test_pow_diff_types(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<i32>) -> tensor<*xf32> {
  // expected-error @+2 {{do not support Pow with different input type yet}}
  // expected-error @+1 {{shape inference failed}}
  %0 = "onnx.Pow"(%arg0, %arg1) : (tensor<1x2x3x4xf32>, tensor<i32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
}

// -----

func @test_pow_int_power(%arg0: tensor<1x2x3x4xi32>, %arg1: tensor<i32>) -> tensor<*xi32> {
  // expected-error @+2 {{do not support integer power yet}}
  // expected-error @+1 {{shape inference failed}}
  %0 = "onnx.Pow"(%arg0, %arg1) : (tensor<1x2x3x4xi32>, tensor<i32>) -> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()
}

// -----

//===----------------------------------------------------------------------===//
/// Unsupported configurations for ONNXReshapeOp.
//===----------------------------------------------------------------------===//

func @test_reshape_unranked_shape(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<*xi64>) -> tensor<*xf32> {
  // expected-error @+2 {{Shape tensor not ranked}}
  // expected-error @+1 {{shape inference failed}}
  %0 = "onnx.Reshape"(%arg0, %arg1) : (tensor<5x5x1x32xf32>, tensor<*xi64>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
}

// -----

func @test_reshape_2D_shape(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<1x2xi64>) -> tensor<*xf32> {
  // expected-error @+2 {{Shape tensor must have rank one}}
  // expected-error @+1 {{shape inference failed}}
  %0 = "onnx.Reshape"(%arg0, %arg1) : (tensor<5x5x1x32xf32>, tensor<1x2xi64>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
}

// -----

func @test_reshape_1D_constant_shape(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<?xi64>) -> tensor<*xf32> {
  // expected-error @+2 {{Shape tensor must have constant shape}}
  // expected-error @+1 {{shape inference failed}}
  %0 = "onnx.Reshape"(%arg0, %arg1) : (tensor<5x5x1x32xf32>, tensor<?xi64>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
}

// -----

//===----------------------------------------------------------------------===//
/// Unsupported configurations for ONNXReduceSumOp.
//===----------------------------------------------------------------------===//

// COM: ReduceSum in OpSet 13.
func @test_reduce_sum_dynamic_axes(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<2xi64>) -> tensor<*xf32> {
  // expected-error @+2 {{ReduceSum: unknown axes}}
  // expected-error @+1 {{shape inference failed}}
  %0 = "onnx.ReduceSum"(%arg0, %arg1) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x2x3x4xf32>, tensor<2xi64>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
}

// -----

//===----------------------------------------------------------------------===//
/// Errors with RNNOps. Take ONNXLSTMOp as an example.
//===----------------------------------------------------------------------===//

func @test_lstm_not_3D_input(%arg0: tensor<4x3xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  // expected-error @+2 {{The first input tensor must have rank 3}}
  // expected-error @+1 {{shape inference failed}}
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  return %Y_h : tensor<*xf32>
}

// -----

func @test_lstm_not_3D_weight(%arg0: tensor<4x3x2xf32>, %arg1: tensor<12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  // expected-error @+2 {{The second input tensor must have rank 3}}
  // expected-error @+1 {{shape inference failed}}
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  return %Y_h : tensor<*xf32>
}

// -----

func @test_lstm_not_3D_recurrent(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<12x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  // expected-error @+2 {{The third input tensor must have rank 3}}
  // expected-error @+1 {{shape inference failed}}
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<12x3xf32>, none, none, none, none, none) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  return %Y_h : tensor<*xf32>
}

// -----

func @test_lstm_wrong_direction(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  // expected-error @+2 {{direction attribute must be one of the strings: forward, reverse, and bidirectional}}
  // expected-error @+1 {{shape inference failed}}
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : si64, direction="forwadr"} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  return %Y_h : tensor<*xf32>
}
