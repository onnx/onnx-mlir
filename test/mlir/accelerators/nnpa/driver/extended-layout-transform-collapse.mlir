// RUN: onnx-mlir --march=z17 --maccel=NNPA --EmitZLowIR --parallel --printIR %s | FileCheck %s
// RUN: onnx-mlir --march=z17 --maccel=NNPA --EmitZLowIR --parallel --disable-collapse --printIR %s | FileCheck --check-prefix=NOCOLLAPSE %s

// Parallel region placement for zhigh.ExtendedLayoutTransform, the split +
// transpose (+ merge) + dlf16/f32 conversion used around attention. Its loop
// nest is the split source space with the innermost level tiled by sticks. The
// pattern reads the global --parallel / --disable-collapse driver flags through
// the NNPA accelerator, so it can only be exercised through the onnx-mlir
// driver, not onnx-mlir-opt.
//
// No GROUND directives: an NNPA model cannot run on a non-z host, so these are
// compile-time assertions only.
//
// Both windows stop at the top two levels: the group is at most [batch, seq],
// and the 12-head level stays a sequential loop inside each fused iteration,
// so recovering indices costs one divide by the dynamic seq rather than
// several per stick.

// BERT Q/K/V head split, [batch, seq, 768] -> [batch, 12, seq, 64]. The nest is
// (batch, seq, 12, 1 stick). With a dynamic batch, the region used to sit on
// batch alone, which runs on one thread at batch 1; collapse fuses batch x seq.
func.func @split_heads_dyn_batch(%arg0: tensor<?x?x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<?x12x?x64xf32> {
  %0 = "zhigh.ExtendedLayoutTransform"(%arg0) {dlf16_to_f32 = true, reshape_merge_axis = -1 : si64, reshape_split_axis = 2 : si64, reshape_split_factor = 64 : si64, transpose_pattern = [0, 2, 1, 3]} : (tensor<?x?x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<?x12x?x64xf32>
  return %0 : tensor<?x12x?x64xf32>

// CHECK-LABEL: func.func @split_heads_dyn_batch
// CHECK:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:       [[COLL_:%.+]] = krnl.collapse([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> !krnl.loop
// CHECK:       krnl.parallel([[COLL_]]) : !krnl.loop
// CHECK:       krnl.iterate([[COLL_]], [[LOOP_0_]]#2, [[LOOP_0_]]#3) with

// Flag off: the region sits on the dynamic batch loop alone.
// NOCOLLAPSE-LABEL: func.func @split_heads_dyn_batch
// NOCOLLAPSE-NOT:   krnl.collapse
// NOCOLLAPSE:       krnl.parallel([[LOOP_0_:%.+]]#0) : !krnl.loop
// NOCOLLAPSE:       return
}

// Batch 1: the batch loop is never worth a region, so the dynamic seq loop is
// parallelized alone, with or without collapse.
func.func @split_heads_batch_1(%arg0: tensor<1x?x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<1x12x?x64xf32> {
  %0 = "zhigh.ExtendedLayoutTransform"(%arg0) {dlf16_to_f32 = true, reshape_merge_axis = -1 : si64, reshape_split_axis = 2 : si64, reshape_split_factor = 64 : si64, transpose_pattern = [0, 2, 1, 3]} : (tensor<1x?x768xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<1x12x?x64xf32>
  return %0 : tensor<1x12x?x64xf32>

// CHECK-LABEL: func.func @split_heads_batch_1
// CHECK-NOT:   krnl.collapse
// CHECK:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK-NOT:   krnl.collapse
// CHECK:       krnl.parallel([[LOOP_0_]]#1) : !krnl.loop

// NOCOLLAPSE-LABEL: func.func @split_heads_batch_1
// NOCOLLAPSE:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// NOCOLLAPSE:       krnl.parallel([[LOOP_0_]]#1) : !krnl.loop
}
