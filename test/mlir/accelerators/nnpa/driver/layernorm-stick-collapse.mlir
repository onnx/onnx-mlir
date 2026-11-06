// RUN: onnx-mlir --march=z17 --maccel=NNPA --EmitZLowIR --parallel --printIR %s | FileCheck %s
// RUN: onnx-mlir --march=z17 --maccel=NNPA --EmitZLowIR --parallel --disable-collapse --printIR %s | FileCheck --check-prefix=NOCOLLAPSE %s

// Parallel region placement for the LayerNorm lowered directly to a stickified
// zTensor (FuzedStickUnstickGenericLayerNormaOpLowering). Its outer loops are
// [batch, seq / 4]: seq is blocked by 4 rows, reduced together. The pattern
// reads the global --parallel / --disable-collapse driver flags through the
// NNPA accelerator, so it can only be exercised through the onnx-mlir driver,
// not onnx-mlir-opt.
//
// No GROUND directives: an NNPA model cannot run on a non-z host, so these are
// compile-time assertions only.
//
// With a dynamic batch, the region used to go on the batch loop alone, which
// runs on one thread at batch 1 while the blocked seq loop holds all the work.
// The batch loop is now collapsed with the tile loop of the blocked seq loop,
// so the region spans batch * ceil(seq / 4) iterations whichever is large.

func.func @layernorm_dyn_batch(%arg0: tensor<?x?x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>) -> tensor<?x?x768xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %Y, %Mean, %InvStdDev = "onnx.LayerNormalization"(%arg0, %arg1, %arg2) {axis = 2 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<?x?x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<?x?x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none, none)
  return %Y : tensor<?x?x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// CHECK-LABEL: func.func @layernorm_dyn_batch
// CHECK:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:       [[BLOCK_TILE_:%.+]], [[BLOCK_IN_:%.+]] = krnl.block [[LOOP_0_]]#1 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:       krnl.permute([[LOOP_0_]]#0, [[BLOCK_TILE_]], [[BLOCK_IN_]]) [0, 1, 2] : !krnl.loop, !krnl.loop, !krnl.loop
// CHECK:       [[COLL_:%.+]] = krnl.collapse([[LOOP_0_]]#0, [[BLOCK_TILE_]]) : (!krnl.loop, !krnl.loop) -> !krnl.loop
// CHECK:       krnl.parallel([[COLL_]]) : !krnl.loop
// CHECK:       krnl.iterate([[COLL_]]) with

// Flag off: the region sits on the dynamic batch loop alone.
// NOCOLLAPSE-LABEL: func.func @layernorm_dyn_batch
// NOCOLLAPSE-NOT:   krnl.collapse
// NOCOLLAPSE:       krnl.parallel([[LOOP_0_:%.+]]#0) : !krnl.loop
// NOCOLLAPSE:       return
}

// Batch 1: the batch loop has a single iteration and is never worth a region,
// so the tile loop of the blocked seq loop is parallelized alone, with or
// without collapse.
func.func @layernorm_batch_1(%arg0: tensor<1x?x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>) -> tensor<1x?x768xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %Y, %Mean, %InvStdDev = "onnx.LayerNormalization"(%arg0, %arg1, %arg2) {axis = 2 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<1x?x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<1x?x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none, none)
  return %Y : tensor<1x?x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// CHECK-LABEL: func.func @layernorm_batch_1
// CHECK-NOT:   krnl.collapse
// CHECK:       [[BLOCK_TILE_:%.+]], [[BLOCK_IN_:%.+]] = krnl.block {{.*}} 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK-NOT:   krnl.collapse
// CHECK:       krnl.parallel([[BLOCK_TILE_]]) : !krnl.loop

// NOCOLLAPSE-LABEL: func.func @layernorm_batch_1
// NOCOLLAPSE:       [[BLOCK_TILE_:%.+]], [[BLOCK_IN_:%.+]] = krnl.block {{.*}} 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// NOCOLLAPSE:       krnl.parallel([[BLOCK_TILE_]]) : !krnl.loop
}

// A static batch large enough on its own keeps the region on the batch loop:
// STEP 1 of the cost model prefers a single level with a literal trip count.
func.func @layernorm_batch_8(%arg0: tensor<8x?x768xf32>, %arg1: tensor<768xf32>, %arg2: tensor<768xf32>) -> tensor<8x?x768xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %Y, %Mean, %InvStdDev = "onnx.LayerNormalization"(%arg0, %arg1, %arg2) {axis = 2 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<8x?x768xf32>, tensor<768xf32>, tensor<768xf32>) -> (tensor<8x?x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none, none)
  return %Y : tensor<8x?x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// CHECK-LABEL: func.func @layernorm_batch_8
// CHECK-NOT:   krnl.collapse
// CHECK:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-NOT:   krnl.collapse
// CHECK:       krnl.parallel([[LOOP_0_]]#0) : !krnl.loop
}

// RMS LayerNorm shares the lowering, and collapses the same way.
func.func @rms_layernorm_dyn_batch(%arg0: tensor<?x?x768xf32>, %arg1: tensor<768xf32>) -> tensor<?x?x768xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %none = "onnx.NoValue"() {value} : () -> none
  %Y, %InvStdDev = "onnx.RMSLayerNormalization"(%arg0, %arg1, %none) {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<?x?x768xf32>, tensor<768xf32>, none) -> (tensor<?x?x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none)
  return %Y : tensor<?x?x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// CHECK-LABEL: func.func @rms_layernorm_dyn_batch
// CHECK:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:       [[BLOCK_TILE_:%.+]], [[BLOCK_IN_:%.+]] = krnl.block [[LOOP_0_]]#1 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:       [[COLL_:%.+]] = krnl.collapse([[LOOP_0_]]#0, [[BLOCK_TILE_]]) : (!krnl.loop, !krnl.loop) -> !krnl.loop
// CHECK:       krnl.parallel([[COLL_]]) : !krnl.loop

// NOCOLLAPSE-LABEL: func.func @rms_layernorm_dyn_batch
// NOCOLLAPSE-NOT:   krnl.collapse
// NOCOLLAPSE:       krnl.parallel([[LOOP_0_:%.+]]#0) : !krnl.loop
// NOCOLLAPSE:       return
}
