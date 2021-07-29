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
