// RUN: onnx-mlir-opt --onnx-pre-krnl-verify %s -split-input-file -verify-diagnostics

// -----
 
//===----------------------------------------------------------------------===//
/// Errors with ONNXReshapeOp.
//===----------------------------------------------------------------------===//

func.func @test_reshape_unranked_shape(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<*xi64>) -> tensor<*xf32> {
  // expected-error @+1 {{not ranked}}
  %0 = "onnx.Reshape"(%arg0, %arg1) : (tensor<5x5x1x32xf32>, tensor<*xi64>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
}

// -----

//===----------------------------------------------------------------------===//
/// Errors with ONNXAddOp.
//===----------------------------------------------------------------------===//

func.func @test_add_unranked_shape(%arg0 : tensor<32xf32>, %arg1 : tensor<*xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{not ranked}}
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<32xf32>, tensor<*xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
}

// -----

//===----------------------------------------------------------------------===//
/// Errors with ONNXPadOp: axes still not known after constant propagation.
//===----------------------------------------------------------------------===//

func.func @test_pad_axes_not_constant(%arg0 : tensor<2x3x4xf32>, %arg1 : tensor<1xi64>) -> tensor<?x?x?xf32> {
  %pads = onnx.Constant dense<[1, 2, 3, 4]> : tensor<4xi64>
  %a1 = onnx.Constant dense<[1]> : tensor<1xi64>
  %axes = "onnx.Concat"(%a1, %arg1) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{Pad axes must be compile-time constants after constant propagation; axes[1] is not}}
  %0 = "onnx.Pad"(%arg0, %pads, %none, %axes) {mode = "constant"} : (tensor<2x3x4xf32>, tensor<4xi64>, none, tensor<2xi64>) -> tensor<?x?x?xf32>
  "func.return"(%0) : (tensor<?x?x?xf32>) -> ()
}

// -----

func.func @test_pad_axes_dynamic_size(%arg0 : tensor<2x3x4xf32>, %arg1 : tensor<?xi64>, %arg2 : tensor<?xi64>) -> tensor<?x?x?xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  // expected-error @+1 {{Pad axes must have a static number of elements}}
  %0 = "onnx.Pad"(%arg0, %arg1, %none, %arg2) {mode = "constant"} : (tensor<2x3x4xf32>, tensor<?xi64>, none, tensor<?xi64>) -> tensor<?x?x?xf32>
  "func.return"(%0) : (tensor<?x?x?xf32>) -> ()
}
