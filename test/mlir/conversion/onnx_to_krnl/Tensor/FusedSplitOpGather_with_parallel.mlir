// RUN: onnx-mlir-opt --march=arm64 --shape-inference --fusion-op-transform --convert-onnx-to-krnl="enable-simd=true enable-parallel=true" %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --march=arm64 --shape-inference --fusion-op-transform --convert-onnx-to-krnl="enable-simd=true enable-parallel=true enable-collapse=false" %s -split-input-file | FileCheck %s --check-prefix=NOCOLLAPSE

// Tests that onnx.Fused(kind="simd-split-op-gather")'s outer (non-split-axis)
// loop, see src/Conversion/ONNXToKrnl/Tensor/FusedSplitOpGather.cpp, is
// tagged for parallel execution with --enable-parallel, using the same
// tryCreateKrnlParallel(..., firstInclusiveDim=0, lastExclusiveDim=min(rank,2))
// pattern used elsewhere (e.g. ZHighToZLow.cpp's concat-expand-stick fused
// outer loop) to parallelize across up to the first two outer dims, which may
// be collapsed into one region. NOCOLLAPSE is the same input with collapse
// off: the single-level region is still created, so turning collapse on is
// additive.

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

// With the literal size-1 batch, there is nothing to fuse: same single region
// with collapse off.
// NOCOLLAPSE-LABEL:  func.func @rotate_half_parallel
// NOCOLLAPSE-NOT:      krnl.collapse
// NOCOLLAPSE:          krnl.parallel([[LOOP_OFF_:%.+]]#1) : !krnl.loop
}

// -----

// Dynamic batch, as in RoPE's rotate-half on [B, heads, S, head_dim]: B is 1
// at inference, so a region on the batch level alone runs on one thread.
// The top two outer levels (B and heads) are fused into one region instead.
// GROUND-THIS: --shape-info=0:2x12x5x32
func.func @rotate_half_parallel_dyn_batch(%arg0: tensor<?x12x?x32xf32>) -> tensor<?x12x?x32xf32> {
  %c0 = onnx.Constant dense<0> : tensor<1xi64>
  %c16 = onnx.Constant dense<16> : tensor<1xi64>
  %c3 = onnx.Constant dense<3> : tensor<1xi64>
  %c1 = onnx.Constant dense<1> : tensor<1xi64>
  %cmax = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
  %0 = "onnx.Slice"(%arg0, %c0, %c16, %c3, %c1) : (tensor<?x12x?x32xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x12x?x16xf32>
  %1 = "onnx.Slice"(%arg0, %c16, %cmax, %c3, %c1) : (tensor<?x12x?x32xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?x12x?x16xf32>
  %2 = "onnx.Neg"(%1) : (tensor<?x12x?x16xf32>) -> tensor<?x12x?x16xf32>
  %3 = "onnx.Concat"(%2, %0) {axis = 3 : si64} : (tensor<?x12x?x16xf32>, tensor<?x12x?x16xf32>) -> tensor<?x12x?x32xf32>
  return %3 : tensor<?x12x?x32xf32>

// CHECK-LABEL:  func.func @rotate_half_parallel_dyn_batch
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x12x?x32xf32>) -> memref<?x12x?x32xf32> {
// CHECK:           [[ALLOC_:%.+]] = memref.alloc({{.*}}) {{.*}} : memref<?x12x?x32xf32>
// CHECK:           [[LOOP_:%.+]]:3 = krnl.define_loops 3
// CHECK:           [[FUSED_:%.+]] = krnl.collapse([[LOOP_]]#0, [[LOOP_]]#1) : (!krnl.loop, !krnl.loop) -> !krnl.loop
// CHECK:           krnl.parallel([[FUSED_]]) : !krnl.loop
// CHECK:           krnl.iterate([[FUSED_]], [[LOOP_]]#2) with ([[LOOP_]]#0 -> {{.*}}, [[LOOP_]]#1 -> {{.*}} = 0 to 12, [[LOOP_]]#2 -> {{.*}})
// CHECK:           return [[ALLOC_]]

// NOCOLLAPSE-LABEL:  func.func @rotate_half_parallel_dyn_batch
// NOCOLLAPSE-NOT:      krnl.collapse
// NOCOLLAPSE:          krnl.parallel([[LOOP_OFF_:%.+]]#0) : !krnl.loop
// NOCOLLAPSE:          krnl.iterate([[LOOP_OFF_]]#0, [[LOOP_OFF_]]#1, [[LOOP_OFF_]]#2)
}
