// RUN: onnx-mlir-opt %s -split-input-file -verify-diagnostics

//===----------------------------------------------------------------------===//
/// Verification tests for XCOMPILER Operations  
/// Domain: com.amd.xcompiler
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// XCOMPILER FusedEltwise Tests (Quantized Element-wise Operations)
//===----------------------------------------------------------------------===//

// Test: clip_min should fail when type is not CLIP
func.func @test_clip_attr_invalid(%arg0: tensor<4xui8>, %arg1: tensor<4xui8>) -> tensor<4xui8> {
  // expected-error @+1 {{'clip_min' is only valid when type is CLIP}}
  %0 = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {type = "ADD", nonlinear = "NONE", clip_min = 0 : si64} : (tensor<4xui8>, tensor<4xui8>) -> tensor<4xui8>
  onnx.Return %0 : tensor<4xui8>
}

// ----- 

// Test: leakyrelu_alpha should fail when nonlinear is not LEAKYRELU  
func.func @test_leakyrelu_attr_invalid(%arg0: tensor<4xui8>, %arg1: tensor<4xui8>) -> tensor<4xui8> {
  // expected-error @+1 {{'leakyrelu_alpha' is only valid when nonlinear is LEAKYRELU}}
  %0 = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {type = "ADD", nonlinear = "RELU", leakyrelu_alpha = 0.1 : f32} : (tensor<4xui8>, tensor<4xui8>) -> tensor<4xui8>
  onnx.Return %0 : tensor<4xui8>
}

// ----- 

// Test: nonlinear_in_scales should fail when nonlinear is NONE
func.func @test_nonlinear_scales_invalid(%arg0: tensor<4xui8>, %arg1: tensor<4xui8>) -> tensor<4xui8>  {
  // expected-error @+1 {{'nonlinear_in_scales' is only valid when nonlinear is not NONE}}
  %0 = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {type = "ADD", nonlinear = "NONE", nonlinear_in_scales = 1.0 : f32} : (tensor<4xui8>, tensor<4xui8>) -> tensor<4xui8>
  onnx.Return %0 : tensor<4xui8>
}

// ----- 

// Test: Valid op (should pass)
func.func @test_valid_clip(%arg0: tensor<4xui8>, %arg1: tensor<4xui8>) -> tensor<4xui8> {
  %0 = "onnx.XCOMPILERFusedEltwise"(%arg0, %arg1) {type = "CLIP", nonlinear = "NONE", clip_min = 0 : si64, clip_max = 255 : si64} : (tensor<4xui8>, tensor<4xui8>) -> tensor<4xui8>
  onnx.Return %0 : tensor<4xui8>
}
