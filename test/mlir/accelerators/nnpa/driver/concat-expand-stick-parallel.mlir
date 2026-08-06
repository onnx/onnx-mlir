// RUN: onnx-mlir --march=z16 --maccel=NNPA --EmitZLowIR --printIR %s | FileCheck --check-prefix=PAR_OFF %s
// RUN: onnx-mlir --march=z16 --maccel=NNPA --EmitZLowIR --parallel --printIR %s | FileCheck --check-prefix=PAR_ON %s

// Lowering of the zhigh.concat-expand-stick fused op (the GQA/MQA
// "repeat KV heads after cache-concat" idiom; see
// test/mlir/accelerators/nnpa/conversion/onnx-to-krnl/concat-expand-stick.mlir
// for the FusedOp-level tests) with the --parallel option.
//
// ZHighToZLowFusedConcatExpandStickLowering (ZHighToZLow.cpp) reads its own
// enableParallel flag from the NNPA accelerator, which -- unlike the CPU
// generic path's --convert-onnx-to-krnl=enable-parallel pass option -- is
// wired to the global --parallel driver flag instead. That flag belongs to
// a command-line category that onnx-mlir-opt strips (see
// removeUnrelatedOptions in onnx-mlir-opt.cpp), so it can only be exercised
// through the onnx-mlir driver, not onnx-mlir-opt.
//
// With --parallel on, the outer loop over the concat's non-innermost dims
// shared by both concat inputs (here dims [0, concatAxis) = [0, 2), sizes
// 2 and 4) gets a single krnl.parallel marking the first dim whose static
// trip count meets tryCreateKrnlParallel's minimum-iteration threshold (4):
// dim 0 (trip count 2) is skipped, dim 1 (trip count 4) is picked.

func.func @concat_expand_stick_parallel(%arg0: tensor<2x4x3x64xf32>, %arg1: tensor<2x4x5x64xf32>) -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %axes  = onnx.Constant dense<2>               : tensor<1xi64>
  %shexp = onnx.Constant dense<[2, 4, 3, 8, 64]> : tensor<5xi64>
  %shre  = onnx.Constant dense<[24, 8, 64]>      : tensor<3xi64>
  %cat  = "onnx.Concat"(%arg0, %arg1) <{axis = 2 : si64}>
            : (tensor<2x4x3x64xf32>, tensor<2x4x5x64xf32>) -> tensor<2x4x8x64xf32>
  %unsq = "onnx.Unsqueeze"(%cat, %axes)
            : (tensor<2x4x8x64xf32>, tensor<1xi64>) -> tensor<2x4x1x8x64xf32>
  %dlf  = "zhigh.F32ToDLF16"(%unsq)
            : (tensor<2x4x1x8x64xf32>) -> tensor<2x4x1x8x64xf16>
  %exp  = "onnx.Expand"(%dlf, %shexp)
            : (tensor<2x4x1x8x64xf16>, tensor<5xi64>) -> tensor<2x4x3x8x64xf16>
  %resh = "onnx.Reshape"(%exp, %shre) <{allowzero = 0 : si64}>
            : (tensor<2x4x3x8x64xf16>, tensor<3xi64>) -> tensor<24x8x64xf16>
  %out  = "onnx.LayoutTransform"(%resh) {target_layout = #zhigh.layout<{dataLayout = "3DS"}>}
            : (tensor<24x8x64xf16>) -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %out : tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// PAR_OFF-LABEL: func.func @concat_expand_stick_parallel
// PAR_OFF-NOT: krnl.parallel

// PAR_ON-LABEL: func.func @concat_expand_stick_parallel
// PAR_ON:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// PAR_ON:       krnl.parallel([[LOOP_0_]]#1) : !krnl.loop
// PAR_ON:       krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> {{.*}} = 0 to 2, [[LOOP_0_]]#1 -> {{.*}} = 0 to 4)
}
