// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --enable-memory-pool --bundle-memory-pools --convert-krnl-to-affine --convert-krnl-to-llvm -verify-diagnostics %s

// Verify that dynamic shapes can be lowered all the way to the LLVM representation (no compiler crashes)

func @test_krnl_getref(%arg0: tensor<?x128xf32>) -> tensor<?x128xf32> {
  %0 = "onnx.Add"(%arg0, %arg0) : (tensor<?x128xf32>, tensor<?x128xf32>) -> tensor<?x128xf32>
  return %0 : tensor<?x128xf32>
}