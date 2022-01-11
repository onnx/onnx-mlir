// RUN: onnx-mlir-opt %s -split-input-file -verify-diagnostics

// -----

func @mod(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{fmod must be 1 when the input type is floating point}}
  %0 = "onnx.Mod"(%arg0, %arg1) {fmod = 0 : si64} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

func @test_depth_to_space_default(%arg0 : tensor<1x256x8x16xf32>) -> tensor<1x16x32x64xf32> {
  %cst = constant unit
  // expected-error @+1 {{The input tensor depth must be divisible by the (blocksize * blocksize)}}  
  %0 = "onnx.DepthToSpace"(%arg0) {blocksize = 7 : si64} : (tensor<1x256x8x16xf32>) -> tensor<1x16x32x64xf32>
  "std.return"(%0) : (tensor<1x16x32x64xf32>) -> ()
}
