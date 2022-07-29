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
