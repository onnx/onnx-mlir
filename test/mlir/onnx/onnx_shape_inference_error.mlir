// RUN: onnx-mlir-opt --shape-inference %s -split-input-file -verify-diagnostics

//===----------------------------------------------------------------------===//
/// Unsupported configurations for ONNXConvOp.
//===----------------------------------------------------------------------===//

// -----

// Error found in the valid tests, so fails before reaching shape inference.
func @unsupport_conv_bad_kernel_shape_attr(%arg0 : tensor<1x2x32x32xf32>, %arg1 : tensor<5x2x7x7xf32>) -> tensor<*xf32> {
  %cst = constant unit
  // expected-error @+1 {{Bad kernel_shape value: must be strictly positive}}
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [-1, 7]} : (tensor<1x2x32x32xf32>, tensor<5x2x7x7xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
}

// -----

// Error found in the valid tests, so fails before reaching shape inference.
func @unsupport_conv_bad_kernel_shape(%arg0 : tensor<1x2x32x32xf32>, %arg1 : tensor<5x2x0x7xf32>) -> tensor<*xf32> {
  %cst = constant unit
  // expected-error @+1 {{Bad spatial filter size: cannot be zero}}
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<1x2x32x32xf32>, tensor<5x2x0x7xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
}

// -----

//===----------------------------------------------------------------------===//
/// Unsupported configurations for ONNXMaxPoolOp.
//===----------------------------------------------------------------------===//

func @unsupport_maxpool_column_storage(%arg0 : tensor<5x5x32x32xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{Column major storage order not implemented yet}}
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "VALID", ceil_mode = 0 : si64, kernel_shape = [3, 3], storage_order = 1 : si64} : (tensor<5x5x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
}

// -----

//===----------------------------------------------------------------------===//
/// Unsupported configurations for ONNXPowOp.
//===----------------------------------------------------------------------===//

func @unsupport_pow_diff_types(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<i32>) -> tensor<*xf32> {
  // expected-error @+2 {{Pow with different input type not implemented yet}}
  // expected-error @+1 {{shape inference failed}}
  %0 = "onnx.Pow"(%arg0, %arg1) : (tensor<1x2x3x4xf32>, tensor<i32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
}

// -----

func @unsupport_pow_int_power(%arg0: tensor<1x2x3x4xi32>, %arg1: tensor<i32>) -> tensor<*xi32> {
  // expected-error @+2 {{Integer power not implemented yet}}
  // expected-error @+1 {{shape inference failed}}
  %0 = "onnx.Pow"(%arg0, %arg1) : (tensor<1x2x3x4xi32>, tensor<i32>) -> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()
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

// -----

//===----------------------------------------------------------------------===//
/// Unsupported configurations for ONNXPadOp.
//===----------------------------------------------------------------------===//

func @unsupport_pad_unknown_pad_values(%arg0 : tensor<16x13xf32>, %arg1 : tensor<4xi64>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<1xf32> } : () -> tensor<1xf32>
  // expected-error @+2 {{Pad: unknown pads}}
  // expected-error @+1 {{shape inference failed}}
  %1 = "onnx.Pad"(%arg0, %arg1, %0) {mode = "constant"} : (tensor<16x13xf32>, tensor<4xi64>, tensor<1xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()
}

// -----

//===----------------------------------------------------------------------===//
/// Unsupported configurations for ONNXResizeOp.
//===----------------------------------------------------------------------===//

func @unsupport_resize_linear_mode(%arg0 : tensor<3x4x5x6xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %0 = "onnx.Constant"() {value = dense<[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]> : tensor<8xf32>} : () -> tensor<8xf32>
  %1 = "onnx.Constant"() {value = dense<[1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]> : tensor<4xf32>} : () -> tensor<4xf32>
  // expected-error @+2 {{these modes() or coordinate_transformation_mode() not implemented yet}}
  // expected-error @+1 {{shape inference failed}}
  %2 = "onnx.Resize"(%arg0, %0, %1, %cst) {coordinate_transformation_mode = "asymmetric", mode = "linear", nearest_mode = "floor", onnx_node_name = "Resize1"} : (tensor<3x4x5x6xf32>, tensor<8xf32>, tensor<4xf32>, none) -> tensor<*xf32>
  "std.return"(%2) : (tensor<*xf32>) -> ()
}

// -----

func @unsupport_resize_cubic_mode(%arg0 : tensor<3x4x5x6xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %0 = "onnx.Constant"() {value = dense<[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]> : tensor<8xf32>} : () -> tensor<8xf32>
  %1 = "onnx.Constant"() {value = dense<[1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00]> : tensor<4xf32>} : () -> tensor<4xf32>
  // expected-error @+2 {{these modes() or coordinate_transformation_mode() not implemented yet}}
  // expected-error @+1 {{shape inference failed}}
  %2 = "onnx.Resize"(%arg0, %0, %1, %cst) {coordinate_transformation_mode = "asymmetric", mode = "cubic", nearest_mode = "floor", onnx_node_name = "Resize1"} : (tensor<3x4x5x6xf32>, tensor<8xf32>, tensor<4xf32>, none) -> tensor<*xf32>
  "std.return"(%2) : (tensor<*xf32>) -> ()
}

