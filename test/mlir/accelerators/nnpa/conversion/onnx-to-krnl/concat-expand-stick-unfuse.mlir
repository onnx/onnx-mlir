// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// The fallback path referenced at the top of concat-expand-stick.mlir: when
// the FusedOp body no longer matches its stored attrs, verifyAndRetrieveAttrs
// fails and FusedOpKindLowering<ConcatExpandStickFusionHelper> falls back to
// FusionOpKindHelper::unFuse(), which inlines the six chain ops (Concat,
// Unsqueeze, F32ToDLF16, Expand, Reshape, LayoutTransform) right before the
// FusedOp and lets each one lower on its own via the generic ONNX-to-Krnl
// patterns, instead of the dedicated tiled loop nest.
//
// This is otherwise the same IR as @concat_expand_stick_basic in
// concat-expand-stick.mlir, except reshapeCollapsedCount is tampered with
// (2 instead of the correct 3): ops[4]'s rank delta is inRank(5) - outRank(3)
// = 2, which requires reshapeCollapsedCount - 1 == 2, i.e. 3 -- so verify()
// rejects the reshape step and the whole fusedOp falls back to inline
// lowering, even though every other stored attr is untouched.

func.func @concat_expand_stick_unfuse(%arg0: tensor<2x4x3x64xf32>, %arg1: tensor<2x4x5x64xf32>) -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %0 = "onnx.Fused"(%arg0, %arg1) <{kind = "zhigh.concat-expand-stick"}> ({
  ^bb0(%arg2: tensor<2x4x3x64xf32>, %arg3: tensor<2x4x5x64xf32>):
    %1 = onnx.Constant dense<[24, 8, 64]> : tensor<3xi64>
    %2 = onnx.Constant dense<[2, 4, 3, 8, 64]> : tensor<5xi64>
    %3 = onnx.Constant dense<2> : tensor<1xi64>
    %4 = "onnx.Concat"(%arg2, %arg3) <{axis = 2 : si64}> : (tensor<2x4x3x64xf32>, tensor<2x4x5x64xf32>) -> tensor<2x4x8x64xf32>
    %5 = "onnx.Unsqueeze"(%4, %3) : (tensor<2x4x8x64xf32>, tensor<1xi64>) -> tensor<2x4x1x8x64xf32>
    %6 = "zhigh.F32ToDLF16"(%5) : (tensor<2x4x1x8x64xf32>) -> tensor<2x4x1x8x64xf16>
    %7 = "onnx.Expand"(%6, %2) : (tensor<2x4x1x8x64xf16>, tensor<5xi64>) -> tensor<2x4x3x8x64xf16>
    %8 = "onnx.Reshape"(%7, %1) <{allowzero = 0 : si64}> : (tensor<2x4x3x8x64xf16>, tensor<3xi64>) -> tensor<24x8x64xf16>
    %9 = "onnx.LayoutTransform"(%8) <{target_layout = #zhigh.layout<{dataLayout = "3DS"}>}> : (tensor<24x8x64xf16>) -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    onnx.Yield %9 : tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  }) {concatAxis = 2 : i64, expansionN = 3 : i64, finalLayout = "3DS", noSaturation = false, reshapeCollapsedCount = 2 : i64, reshapeFirstCollapsedDim = 0 : i64, unsqueezedPosition = 2 : i64, yieldConcatResult = false} : (tensor<2x4x3x64xf32>, tensor<2x4x5x64xf32>) -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %0 : tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 3)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-LABEL:  func.func @concat_expand_stick_unfuse
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4x3x64xf32>, [[PARAM_1_:%.+]]: memref<2x4x5x64xf32>) -> memref<24x8x64xf16, #map> {
// CHECK-DAG:       [[CST_60_:%.+]] = arith.constant 60 : index
// CHECK-DAG:       [[CST_52_:%.+]] = arith.constant 52 : index
// CHECK-DAG:       [[CST_44_:%.+]] = arith.constant 44 : index
// CHECK-DAG:       [[CST_36_:%.+]] = arith.constant 36 : index
// CHECK-DAG:       [[CST_28_:%.+]] = arith.constant 28 : index
// CHECK-DAG:       [[CST_20_:%.+]] = arith.constant 20 : index
// CHECK-DAG:       [[CST_12_:%.+]] = arith.constant 12 : index
// CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : i64
// CHECK-DAG:       [[CST_56_:%.+]] = arith.constant 56 : index
// CHECK-DAG:       [[CST_48_:%.+]] = arith.constant 48 : index
// CHECK-DAG:       [[CST_40_:%.+]] = arith.constant 40 : index
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[CST_24_:%.+]] = arith.constant 24 : index
// CHECK-DAG:       [[CST_16_:%.+]] = arith.constant 16 : index
// CHECK-DAG:       [[CST_4096_:%.+]] = arith.constant 4096 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x4x8x64xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 3, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 64){
// CHECK:             [[VAR_5_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_5_]]#0, [[VAR_5_]]#1, [[VAR_5_]]#2, [[VAR_5_]]#3] : memref<2x4x3x64xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_5_]]#0, [[VAR_5_]]#1, [[VAR_5_]]#2, [[VAR_5_]]#3] : memref<2x4x8x64xf32>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2, [[LOOP_1_]]#3) with ([[LOOP_1_]]#0 -> [[I_4_:%.+]] = 0 to 2, [[LOOP_1_]]#1 -> [[I_5_:%.+]] = 0 to 4, [[LOOP_1_]]#2 -> [[I_6_:%.+]] = 0 to 5, [[LOOP_1_]]#3 -> [[I_7_:%.+]] = 0 to 64){
// CHECK:             [[VAR_5_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2, [[LOOP_1_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = affine.apply [[MAP_1_]]([[VAR_5_1_]]#2)
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_5_1_]]#0, [[VAR_5_1_]]#1, [[VAR_5_1_]]#2, [[VAR_5_1_]]#3] : memref<2x4x5x64xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_]]{{.}}[[VAR_5_1_]]#0, [[VAR_5_1_]]#1, [[LOAD_PARAM_0_MEM_1_]], [[VAR_5_1_]]#3] : memref<2x4x8x64xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 4, 1, 8, 64], strides: [2048, 512, 512, 64, 1] : memref<2x4x8x64xf32> to memref<2x4x1x8x64xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4x1x8x64xf16>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_4096_]], [[RES_2_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[VAR_reinterpret_cast_]]([[RES_2_]]) : (memref<2x4x1x8x64xf32>, memref<1xindex>) -> memref<4096xf32>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_4096_]], [[RES_3_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_3_:%.+]] = memref.reshape [[RES_1_]]([[RES_3_]]) : (memref<2x4x1x8x64xf16>, memref<1xindex>) -> memref<4096xf16>
// CHECK-DAG:       [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_2_]] 64 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_2_]] -> [[I_8_:%.+]] = 0 to 4096){
// CHECK:             [[VAR_5_2_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = arith.addi [[VAR_5_2_]], [[CST_4_]] : index
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_1_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_5_2_]]{{.}} : memref<4096xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[LOAD_PARAM_0_MEM_1_]]{{.}} : memref<4096xf32>, vector<4xf32>
// CHECK:             [[VAR_9_:%.+]] = "zlow.vec_f32_to_dlf16"([[LOAD_PARAM_1_MEM_1_]], [[LOAD_VAR_reshape_MEM_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_9_]], [[VAR_reshape_3_]]{{.}}[[VAR_5_2_]]{{.}} : memref<4096xf16>, vector<8xf16>
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.addi [[VAR_5_2_]], [[CST_8_]] : index
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.addi [[VAR_5_2_]], [[CST_12_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_1_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_10_]]{{.}} : memref<4096xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_2_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_11_]]{{.}} : memref<4096xf32>, vector<4xf32>
// CHECK:             [[VAR_14_:%.+]] = "zlow.vec_f32_to_dlf16"([[LOAD_VAR_reshape_MEM_1_]], [[LOAD_VAR_reshape_MEM_2_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_14_]], [[VAR_reshape_3_]]{{.}}[[VAR_10_]]{{.}} : memref<4096xf16>, vector<8xf16>
// CHECK-DAG:         [[VAR_15_:%.+]] = arith.addi [[VAR_5_2_]], [[CST_16_]] : index
// CHECK-DAG:         [[VAR_16_:%.+]] = arith.addi [[VAR_5_2_]], [[CST_20_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_3_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_15_]]{{.}} : memref<4096xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_4_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_16_]]{{.}} : memref<4096xf32>, vector<4xf32>
// CHECK:             [[VAR_19_:%.+]] = "zlow.vec_f32_to_dlf16"([[LOAD_VAR_reshape_MEM_3_]], [[LOAD_VAR_reshape_MEM_4_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_19_]], [[VAR_reshape_3_]]{{.}}[[VAR_15_]]{{.}} : memref<4096xf16>, vector<8xf16>
// CHECK-DAG:         [[VAR_20_:%.+]] = arith.addi [[VAR_5_2_]], [[CST_24_]] : index
// CHECK-DAG:         [[VAR_21_:%.+]] = arith.addi [[VAR_5_2_]], [[CST_28_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_5_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_20_]]{{.}} : memref<4096xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_6_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_21_]]{{.}} : memref<4096xf32>, vector<4xf32>
// CHECK:             [[VAR_24_:%.+]] = "zlow.vec_f32_to_dlf16"([[LOAD_VAR_reshape_MEM_5_]], [[LOAD_VAR_reshape_MEM_6_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_24_]], [[VAR_reshape_3_]]{{.}}[[VAR_20_]]{{.}} : memref<4096xf16>, vector<8xf16>
// CHECK-DAG:         [[VAR_25_:%.+]] = arith.addi [[VAR_5_2_]], [[CST_32_]] : index
// CHECK-DAG:         [[VAR_26_:%.+]] = arith.addi [[VAR_5_2_]], [[CST_36_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_7_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_25_]]{{.}} : memref<4096xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_8_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_26_]]{{.}} : memref<4096xf32>, vector<4xf32>
// CHECK:             [[VAR_29_:%.+]] = "zlow.vec_f32_to_dlf16"([[LOAD_VAR_reshape_MEM_7_]], [[LOAD_VAR_reshape_MEM_8_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_29_]], [[VAR_reshape_3_]]{{.}}[[VAR_25_]]{{.}} : memref<4096xf16>, vector<8xf16>
// CHECK-DAG:         [[VAR_30_:%.+]] = arith.addi [[VAR_5_2_]], [[CST_40_]] : index
// CHECK-DAG:         [[VAR_31_:%.+]] = arith.addi [[VAR_5_2_]], [[CST_44_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_9_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_30_]]{{.}} : memref<4096xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_10_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_31_]]{{.}} : memref<4096xf32>, vector<4xf32>
// CHECK:             [[VAR_34_:%.+]] = "zlow.vec_f32_to_dlf16"([[LOAD_VAR_reshape_MEM_9_]], [[LOAD_VAR_reshape_MEM_10_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_34_]], [[VAR_reshape_3_]]{{.}}[[VAR_30_]]{{.}} : memref<4096xf16>, vector<8xf16>
// CHECK-DAG:         [[VAR_35_:%.+]] = arith.addi [[VAR_5_2_]], [[CST_48_]] : index
// CHECK-DAG:         [[VAR_36_:%.+]] = arith.addi [[VAR_5_2_]], [[CST_52_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_11_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_35_]]{{.}} : memref<4096xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_12_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_36_]]{{.}} : memref<4096xf32>, vector<4xf32>
// CHECK:             [[VAR_39_:%.+]] = "zlow.vec_f32_to_dlf16"([[LOAD_VAR_reshape_MEM_11_]], [[LOAD_VAR_reshape_MEM_12_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_39_]], [[VAR_reshape_3_]]{{.}}[[VAR_35_]]{{.}} : memref<4096xf16>, vector<8xf16>
// CHECK-DAG:         [[VAR_40_:%.+]] = arith.addi [[VAR_5_2_]], [[CST_56_]] : index
// CHECK-DAG:         [[VAR_41_:%.+]] = arith.addi [[VAR_5_2_]], [[CST_60_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_13_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_40_]]{{.}} : memref<4096xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_14_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_41_]]{{.}} : memref<4096xf32>, vector<4xf32>
// CHECK:             [[VAR_44_:%.+]] = "zlow.vec_f32_to_dlf16"([[LOAD_VAR_reshape_MEM_13_]], [[LOAD_VAR_reshape_MEM_14_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_44_]], [[VAR_reshape_3_]]{{.}}[[VAR_40_]]{{.}} : memref<4096xf16>, vector<8xf16>
// CHECK:           }
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<2x4x3x8x64xf16>
// CHECK-DAG:       [[LOOP_3_:%.+]]:5 = krnl.define_loops 5
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1, [[LOOP_3_]]#2, [[LOOP_3_]]#3, [[LOOP_3_]]#4) with ([[LOOP_3_]]#0 -> [[I_9_:%.+]] = 0 to 2, [[LOOP_3_]]#1 -> [[I_10_:%.+]] = 0 to 4, [[LOOP_3_]]#2 -> [[I_11_:%.+]] = 0 to 3, [[LOOP_3_]]#3 -> [[I_12_:%.+]] = 0 to 8, [[LOOP_3_]]#4 -> [[I_13_:%.+]] = 0 to 64){
// CHECK:             [[VAR_5_3_:%.+]]:5 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1, [[LOOP_3_]]#2, [[LOOP_3_]]#3, [[LOOP_3_]]#4) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_5_3_]]#0, [[VAR_5_3_]]#1, [[CST_0_]], [[VAR_5_3_]]#3, [[VAR_5_3_]]#4] : memref<2x4x1x8x64xf16>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_1_1_]], [[RES_4_]]{{.}}[[VAR_5_3_]]#0, [[VAR_5_3_]]#1, [[VAR_5_3_]]#2, [[VAR_5_3_]]#3, [[VAR_5_3_]]#4] : memref<2x4x3x8x64xf16>
// CHECK:           }
// CHECK-DAG:       [[VAR_reinterpret_cast_5_:%.+]] = memref.reinterpret_cast [[RES_4_]] to offset: [0], sizes: [24, 8, 64], strides: [512, 64, 1] : memref<2x4x3x8x64xf16> to memref<24x8x64xf16>
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<24x8x64xf16, #map>
// CHECK-DAG:       [[LOOP_4_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1, [[LOOP_4_]]#2) with ([[LOOP_4_]]#0 -> [[I_14_:%.+]] = 0 to 24, [[LOOP_4_]]#1 -> [[I_15_:%.+]] = 0 to 8, [[LOOP_4_]]#2 -> [[I_16_:%.+]] = 0 to 1){
// CHECK:             [[VAR_5_4_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1, [[LOOP_4_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_1_:%.+]] = affine.apply [[MAP_2_]]([[VAR_5_4_]]#2)
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.get_linear_offset_index [[RES_5_]] at {{.}}[[VAR_5_4_]]#0, [[VAR_5_4_]]#1, [[LOAD_PARAM_0_MEM_1_1_]]{{.}} : memref<24x8x64xf16, #map>
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_15_:%.+]] = krnl.get_linear_offset_index [[VAR_reinterpret_cast_5_]] at {{.}}[[VAR_5_4_]]#0, [[VAR_5_4_]]#1, [[LOAD_PARAM_0_MEM_1_1_]]{{.}} : memref<24x8x64xf16>
// CHECK:             "krnl.memcpy"([[RES_5_]], [[VAR_reinterpret_cast_5_]], [[CST_64_]], [[LOAD_PARAM_1_MEM_1_]], [[LOAD_VAR_reshape_MEM_15_]]) : (memref<24x8x64xf16, #map>, memref<24x8x64xf16>, i64, index, index) -> ()
// CHECK:           }
// CHECK:           return [[RES_5_]] : memref<24x8x64xf16, #map>
// CHECK:         }
}
