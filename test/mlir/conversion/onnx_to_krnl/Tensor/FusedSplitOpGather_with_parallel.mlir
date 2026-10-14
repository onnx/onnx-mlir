// RUN: onnx-mlir-opt --march=arm64 --shape-inference --fusion-op-transform --convert-onnx-to-krnl="enable-simd=true enable-parallel=true" %s -split-input-file | FileCheck %s

// Tests that onnx.Fused(kind="simd-split-op-gather")'s outer (non-split-axis)
// loop, see src/Conversion/ONNXToKrnl/Tensor/FusedSplitOpGather.cpp, is
// tagged for parallel execution with --enable-parallel, using the same
// tryCreateKrnlParallel(..., firstInclusiveDim=0, lastExclusiveDim=min(rank,2))
// pattern used elsewhere (e.g. ZHighToZLow.cpp's concat-expand-stick fused
// outer loop) to parallelize across up to the first two outer dims.

func.func @rotate_half_parallel(%arg0: tensor<1x4x8x128xf32>) -> tensor<1x4x8x128xf32> {
  %c0 = onnx.Constant dense<0> : tensor<1xi64>
  %c64 = onnx.Constant dense<64> : tensor<1xi64>
  %c3 = onnx.Constant dense<3> : tensor<1xi64>
  %c1 = onnx.Constant dense<1> : tensor<1xi64>
  %cmax = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %c0, %c64, %c3, %c1) : (tensor<1x4x8x128xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x4x8x64xf32>
  %1 = "onnx.Slice"(%arg0, %c64, %cmax, %c3, %c1) : (tensor<1x4x8x128xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x4x8x64xf32>
  %2 = "onnx.Neg"(%1) : (tensor<1x4x8x64xf32>) -> tensor<1x4x8x64xf32>
  %3 = "onnx.Concat"(%2, %0) {axis = 3 : si64} : (tensor<1x4x8x64xf32>, tensor<1x4x8x64xf32>) -> tensor<1x4x8x128xf32>
  return %3 : tensor<1x4x8x128xf32>

// CHECK-LABEL:  func.func @rotate_half_parallel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x4x8x128xf32>) -> memref<1x4x8x128xf32> {
// CHECK:           [[ALLOC_:%.+]] = memref.alloc() {{.*}} : memref<1x4x8x128xf32>
// CHECK:           [[LOOP_:%.+]]:3 = krnl.define_loops 3
// The outer 3-dim loop (the two innermost, split-axis-adjacent loops of that
// nest are elided) is tagged parallel on one of its first two dims.
// CHECK:           krnl.parallel([[LOOP_]]#1) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_]]#0, [[LOOP_]]#1, [[LOOP_]]#2)
// CHECK:           return [[ALLOC_]]
}
