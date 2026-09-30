// RUN: onnx-mlir --march=z16 --maccel=NNPA --EmitZLowIR --parallel --printIR %s | FileCheck %s
// RUN: onnx-mlir --march=z16 --maccel=NNPA --EmitZLowIR --parallel --disable-collapse --printIR %s | FileCheck --check-prefix=NOCOLLAPSE %s

// Collapse coverage for the zhigh.concat-expand-stick fused op, the sibling of
// concat-expand-stick-parallel.mlir. ZHighToZLowFusedConcatExpandStickLowering
// reads the global --parallel / --disable-collapse driver flags through the
// NNPA accelerator, so, like its sibling, it can only be exercised through the
// onnx-mlir driver, not onnx-mlir-opt.
//
// No GROUND directives: an NNPA model cannot run on a non-z host, so this is a
// compile-time assertion only.
//
// The outer loop over the concat's non-innermost dims shared by both inputs is
// here [0, concatAxis) = [0, 2), sizes 2 and 2. Neither level alone meets the
// minimum trip count for a parallel region (4), so with collapse off no region
// is created at all; with collapse on, the two levels fuse into one region of 4
// iterations.

func.func @concat_expand_stick_collapse(%arg0: tensor<2x2x3x64xf32>, %arg1: tensor<2x2x5x64xf32>) -> tensor<12x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %axes  = onnx.Constant dense<2>               : tensor<1xi64>
  %shexp = onnx.Constant dense<[2, 2, 3, 8, 64]> : tensor<5xi64>
  %shre  = onnx.Constant dense<[12, 8, 64]>      : tensor<3xi64>
  %cat  = "onnx.Concat"(%arg0, %arg1) <{axis = 2 : si64}>
            : (tensor<2x2x3x64xf32>, tensor<2x2x5x64xf32>) -> tensor<2x2x8x64xf32>
  %unsq = "onnx.Unsqueeze"(%cat, %axes)
            : (tensor<2x2x8x64xf32>, tensor<1xi64>) -> tensor<2x2x1x8x64xf32>
  %dlf  = "zhigh.F32ToDLF16"(%unsq)
            : (tensor<2x2x1x8x64xf32>) -> tensor<2x2x1x8x64xf16>
  %exp  = "onnx.Expand"(%dlf, %shexp)
            : (tensor<2x2x1x8x64xf16>, tensor<5xi64>) -> tensor<2x2x3x8x64xf16>
  %resh = "onnx.Reshape"(%exp, %shre) <{allowzero = 0 : si64}>
            : (tensor<2x2x3x8x64xf16>, tensor<3xi64>) -> tensor<12x8x64xf16>
  %out  = "onnx.LayoutTransform"(%resh) {target_layout = #zhigh.layout<{dataLayout = "3DS"}>}
            : (tensor<12x8x64xf16>) -> tensor<12x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %out : tensor<12x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// CHECK-LABEL: func.func @concat_expand_stick_collapse
// CHECK:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:       [[COLL_:%.+]] = krnl.collapse([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> !krnl.loop
// CHECK:       krnl.parallel([[COLL_]]) : !krnl.loop
// CHECK:       krnl.iterate([[COLL_]]) with ([[LOOP_0_]]#0 -> {{.*}} = 0 to 2, [[LOOP_0_]]#1 -> {{.*}} = 0 to 2)

// NOCOLLAPSE-LABEL: func.func @concat_expand_stick_collapse
// NOCOLLAPSE-NOT:   krnl.collapse
// NOCOLLAPSE-NOT:   krnl.parallel
// NOCOLLAPSE:       return
}
