// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Dedicated lowering for the zhigh.concat-expand-stick FusedOp kind. See
// concat-expand-stick-unfuse.mlir for a case that still falls back to the
// generic inline lowering (a body that no longer matches the stored attrs).
//
// Computed as an outer loop over dims [0, concatAxis) shared by both concat
// inputs, containing two back-to-back tiled inner loop nests over dims
// [concatAxis, rank) -- one per concat input -- each converting its own
// operand to dlf16 once and fanning it out to the N (=expansionN) stickified
// output locations it broadcasts to. The second operand's inner loop shifts
// its axis-concatAxis coordinate by the first operand's extent there
// (affine_map<(d0) -> (d0 + 3)> below) before computing the output offset.
//
// When yieldConcatResult is true (the concat result also has uses outside
// the chain), a second, independent result is produced: the concat's own
// (plain, unstickified) result. It is folded into the same tiled loop nest
// as the stickified outputs -- each tile's read is shared, and the plain
// F32 halves are stored to the concat buffer right alongside the DLF16
// conversion, so no separate copy loop is needed. See
// @concat_expand_stick_multi_output below.

// -----

func.func @concat_expand_stick_basic(%arg0: tensor<2x4x3x64xf32>, %arg1: tensor<2x4x5x64xf32>) -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
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
  }) {concatAxis = 2 : i64, expansionN = 3 : i64, finalLayout = "3DS", mulScalar = 1.000000e+00 : f32, noSaturation = false, reshapeCollapsedCount = 3 : i64, reshapeFirstCollapsedDim = 0 : i64, unsqueezedPosition = 2 : i64, yieldConcatResult = false} : (tensor<2x4x3x64xf32>, tensor<2x4x5x64xf32>) -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %0 : tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d0 * 12 + d1 * 3)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 floordiv 64)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0, d1) -> (d0 * 12 + d1 * 3 + 1)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0, d1) -> (d0 * 12 + d1 * 3 + 2)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64)>
// CHECK-DAG:   [[MAP_7_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64 + 8)>
// CHECK-DAG:   [[MAP_8_:#.+]] = affine_map<(d0) -> (d0 + 8)>
// CHECK-DAG:   [[MAP_9_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64 + 16)>
// CHECK-DAG:   [[MAP_10_:#.+]] = affine_map<(d0) -> (d0 + 16)>
// CHECK-DAG:   [[MAP_11_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64 + 24)>
// CHECK-DAG:   [[MAP_12_:#.+]] = affine_map<(d0) -> (d0 + 24)>
// CHECK-DAG:   [[MAP_13_:#.+]] = affine_map<(d0) -> (d0 + 3)>
// CHECK-LABEL:  func.func @concat_expand_stick_basic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4x3x64xf32>, [[PARAM_1_:%.+]]: memref<2x4x5x64xf32>) -> memref<24x8x64xf16, #map> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<-8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<24x8x64xf16, #map>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<24x8x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<24x8x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[VAR_reinterpret_cast_2_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<24x8x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK-DAG:         [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 1){
// CHECK:               [[VAR_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[VAR_5_:%.+]] = affine.apply [[MAP_1_]]([[VAR_4_]]#1)
// CHECK-DAG:           [[VAR_6_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#0, [[VAR_1_]]#1)
// CHECK:               [[VAR_7_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_6_]], [[VAR_4_]]#0, [[VAR_5_]]{{.}} : memref<24x8x64xf16, #map>
// CHECK-DAG:           [[VAR_8_:%.+]] = affine.apply [[MAP_3_]]([[VAR_7_]])
// CHECK-DAG:           [[VAR_9_:%.+]] = affine.apply [[MAP_4_]]([[VAR_1_]]#0, [[VAR_1_]]#1)
// CHECK:               [[VAR_10_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_9_]], [[VAR_4_]]#0, [[VAR_5_]]{{.}} : memref<24x8x64xf16, #map>
// CHECK-DAG:           [[VAR_11_:%.+]] = affine.apply [[MAP_3_]]([[VAR_10_]])
// CHECK-DAG:           [[VAR_12_:%.+]] = affine.apply [[MAP_5_]]([[VAR_1_]]#0, [[VAR_1_]]#1)
// CHECK:               [[VAR_13_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_12_]], [[VAR_4_]]#0, [[VAR_5_]]{{.}} : memref<24x8x64xf16, #map>
// CHECK-DAG:           [[VAR_14_:%.+]] = affine.apply [[MAP_3_]]([[VAR_13_]])
// CHECK-DAG:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:               [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_2_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:               krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_2_]] -> [[I_4_:%.+]] = 0 to 64){
// CHECK:                 [[VAR_16_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:                 [[VAR_17_:%.+]] = affine.apply [[MAP_6_]]([[VAR_16_]], [[VAR_4_]]#1)
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_17_]]{{.}} : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_19_:%.+]] = arith.addi [[VAR_17_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_1_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_19_]]{{.}} : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_21_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_22_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_23_:%.+]] = arith.maxnumf [[VAR_21_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_24_:%.+]] = arith.maxnumf [[VAR_22_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_25_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_23_]], [[VAR_24_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                 vector.store [[VAR_25_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_8_]], [[VAR_16_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store [[VAR_25_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_11_]], [[VAR_16_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store [[VAR_25_]], [[VAR_reinterpret_cast_2_]]{{.}}[[VAR_14_]], [[VAR_16_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_26_:%.+]] = affine.apply [[MAP_7_]]([[VAR_16_]], [[VAR_4_]]#1)
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_2_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_26_]]{{.}} : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_28_:%.+]] = arith.addi [[VAR_26_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_3_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_28_]]{{.}} : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_30_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_2_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_31_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_3_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_32_:%.+]] = arith.maxnumf [[VAR_30_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_33_:%.+]] = arith.maxnumf [[VAR_31_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_34_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_32_]], [[VAR_33_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:             [[VAR_35_:%.+]] = affine.apply [[MAP_8_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_34_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_8_]], [[VAR_35_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_36_:%.+]] = affine.apply [[MAP_8_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_34_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_11_]], [[VAR_36_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_37_:%.+]] = affine.apply [[MAP_8_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_34_]], [[VAR_reinterpret_cast_2_]]{{.}}[[VAR_14_]], [[VAR_37_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_38_:%.+]] = affine.apply [[MAP_9_]]([[VAR_16_]], [[VAR_4_]]#1)
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_4_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_38_]]{{.}} : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_40_:%.+]] = arith.addi [[VAR_38_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_5_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_40_]]{{.}} : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_42_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_4_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_43_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_5_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_44_:%.+]] = arith.maxnumf [[VAR_42_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_45_:%.+]] = arith.maxnumf [[VAR_43_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_46_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_44_]], [[VAR_45_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:             [[VAR_47_:%.+]] = affine.apply [[MAP_10_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_46_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_8_]], [[VAR_47_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_48_:%.+]] = affine.apply [[MAP_10_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_46_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_11_]], [[VAR_48_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_49_:%.+]] = affine.apply [[MAP_10_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_46_]], [[VAR_reinterpret_cast_2_]]{{.}}[[VAR_14_]], [[VAR_49_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_50_:%.+]] = affine.apply [[MAP_11_]]([[VAR_16_]], [[VAR_4_]]#1)
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_6_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_50_]]{{.}} : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_52_:%.+]] = arith.addi [[VAR_50_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_7_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_52_]]{{.}} : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_54_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_6_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_55_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_7_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_56_:%.+]] = arith.maxnumf [[VAR_54_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_57_:%.+]] = arith.maxnumf [[VAR_55_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_58_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_56_]], [[VAR_57_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:             [[VAR_59_:%.+]] = affine.apply [[MAP_12_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_58_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_8_]], [[VAR_59_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_60_:%.+]] = affine.apply [[MAP_12_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_58_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_11_]], [[VAR_60_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_61_:%.+]] = affine.apply [[MAP_12_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_58_]], [[VAR_reinterpret_cast_2_]]{{.}}[[VAR_14_]], [[VAR_61_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               }
// CHECK:             }
// CHECK:             [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to 5, [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to 1){
// CHECK:               [[VAR_4_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[VAR_5_1_:%.+]] = affine.apply [[MAP_1_]]([[VAR_4_1_]]#1)
// CHECK-DAG:           [[VAR_6_1_:%.+]] = affine.apply [[MAP_13_]]([[VAR_4_1_]]#0)
// CHECK-DAG:           [[VAR_7_1_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#0, [[VAR_1_]]#1)
// CHECK:               [[VAR_8_1_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_7_1_]], [[VAR_6_1_]], [[VAR_5_1_]]{{.}} : memref<24x8x64xf16, #map>
// CHECK-DAG:           [[VAR_9_1_:%.+]] = affine.apply [[MAP_3_]]([[VAR_8_1_]])
// CHECK-DAG:           [[VAR_10_1_:%.+]] = affine.apply [[MAP_4_]]([[VAR_1_]]#0, [[VAR_1_]]#1)
// CHECK:               [[VAR_11_1_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_10_1_]], [[VAR_6_1_]], [[VAR_5_1_]]{{.}} : memref<24x8x64xf16, #map>
// CHECK-DAG:           [[VAR_12_1_:%.+]] = affine.apply [[MAP_3_]]([[VAR_11_1_]])
// CHECK-DAG:           [[VAR_13_1_:%.+]] = affine.apply [[MAP_5_]]([[VAR_1_]]#0, [[VAR_1_]]#1)
// CHECK:               [[VAR_14_1_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_13_1_]], [[VAR_6_1_]], [[VAR_5_1_]]{{.}} : memref<24x8x64xf16, #map>
// CHECK-DAG:           [[LOOP_2_:%.+]] = affine.apply [[MAP_3_]]([[VAR_14_1_]])
// CHECK-DAG:           [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:               [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_4_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:               krnl.iterate([[BLOCK_TILE__1_]]) with ([[LOOP_4_]] -> [[I_7_:%.+]] = 0 to 64){
// CHECK:                 [[VAR_17_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__1_]]) : (!krnl.loop) -> index
// CHECK:                 [[LOAD_PARAM_0_MEM_8_:%.+]] = affine.apply [[MAP_6_]]([[VAR_17_1_]], [[VAR_4_1_]]#1)
// CHECK-DAG:             [[VAR_19_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_1_]]#0, [[LOAD_PARAM_0_MEM_8_]]{{.}} : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_1_:%.+]] = arith.addi [[LOAD_PARAM_0_MEM_8_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_21_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_1_]]#0, [[LOAD_PARAM_0_MEM_1_]]{{.}} : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_22_1_:%.+]] = arith.minnumf [[VAR_19_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_23_1_:%.+]] = arith.minnumf [[VAR_21_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_24_1_:%.+]] = arith.maxnumf [[VAR_22_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_25_1_:%.+]] = arith.maxnumf [[VAR_23_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_26_1_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_24_1_]], [[VAR_25_1_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                 vector.store [[VAR_26_1_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_9_1_]], [[VAR_17_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store [[VAR_26_1_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_1_]], [[VAR_17_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store [[VAR_26_1_]], [[VAR_reinterpret_cast_2_]]{{.}}[[LOOP_2_]], [[VAR_17_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[LOAD_PARAM_0_MEM_2_:%.+]] = affine.apply [[MAP_7_]]([[VAR_17_1_]], [[VAR_4_1_]]#1)
// CHECK-DAG:             [[VAR_28_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_1_]]#0, [[LOAD_PARAM_0_MEM_2_]]{{.}} : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_3_:%.+]] = arith.addi [[LOAD_PARAM_0_MEM_2_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_30_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_1_]]#0, [[LOAD_PARAM_0_MEM_3_]]{{.}} : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_31_1_:%.+]] = arith.minnumf [[VAR_28_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_32_1_:%.+]] = arith.minnumf [[VAR_30_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_33_1_:%.+]] = arith.maxnumf [[VAR_31_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_34_1_:%.+]] = arith.maxnumf [[VAR_32_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_35_1_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_33_1_]], [[VAR_34_1_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:             [[VAR_36_1_:%.+]] = affine.apply [[MAP_8_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_35_1_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_9_1_]], [[VAR_36_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_37_1_:%.+]] = affine.apply [[MAP_8_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_35_1_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_1_]], [[VAR_37_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_38_1_:%.+]] = affine.apply [[MAP_8_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_35_1_]], [[VAR_reinterpret_cast_2_]]{{.}}[[LOOP_2_]], [[VAR_38_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[LOAD_PARAM_0_MEM_4_:%.+]] = affine.apply [[MAP_9_]]([[VAR_17_1_]], [[VAR_4_1_]]#1)
// CHECK-DAG:             [[VAR_40_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_1_]]#0, [[LOAD_PARAM_0_MEM_4_]]{{.}} : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_5_:%.+]] = arith.addi [[LOAD_PARAM_0_MEM_4_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_42_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_1_]]#0, [[LOAD_PARAM_0_MEM_5_]]{{.}} : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_43_1_:%.+]] = arith.minnumf [[VAR_40_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_44_1_:%.+]] = arith.minnumf [[VAR_42_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_45_1_:%.+]] = arith.maxnumf [[VAR_43_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_46_1_:%.+]] = arith.maxnumf [[VAR_44_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_47_1_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_45_1_]], [[VAR_46_1_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:             [[VAR_48_1_:%.+]] = affine.apply [[MAP_10_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_47_1_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_9_1_]], [[VAR_48_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_49_1_:%.+]] = affine.apply [[MAP_10_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_47_1_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_1_]], [[VAR_49_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_50_1_:%.+]] = affine.apply [[MAP_10_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_47_1_]], [[VAR_reinterpret_cast_2_]]{{.}}[[LOOP_2_]], [[VAR_50_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[LOAD_PARAM_0_MEM_6_:%.+]] = affine.apply [[MAP_11_]]([[VAR_17_1_]], [[VAR_4_1_]]#1)
// CHECK-DAG:             [[VAR_52_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_1_]]#0, [[LOAD_PARAM_0_MEM_6_]]{{.}} : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_7_:%.+]] = arith.addi [[LOAD_PARAM_0_MEM_6_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_54_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_1_]]#0, [[LOAD_PARAM_0_MEM_7_]]{{.}} : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_55_1_:%.+]] = arith.minnumf [[VAR_52_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_56_1_:%.+]] = arith.minnumf [[VAR_54_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_57_1_:%.+]] = arith.maxnumf [[VAR_55_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_58_1_:%.+]] = arith.maxnumf [[VAR_56_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_59_1_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_57_1_]], [[VAR_58_1_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:             [[VAR_60_1_:%.+]] = affine.apply [[MAP_12_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_59_1_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_9_1_]], [[VAR_60_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_61_1_:%.+]] = affine.apply [[MAP_12_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_59_1_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_1_]], [[VAR_61_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_62_:%.+]] = affine.apply [[MAP_12_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_59_1_]], [[VAR_reinterpret_cast_2_]]{{.}}[[LOOP_2_]], [[VAR_62_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<24x8x64xf16, #map>
// CHECK:         }
}

// -----

// Same pattern, but the source chain's F32ToDLF16 has no_saturation = true:
// the lowering must skip the min/max saturation clamp and go straight from
// the loaded F32 values to the dlf16 conversion.

func.func @concat_expand_stick_no_saturation(%arg0: tensor<2x4x3x64xf32>, %arg1: tensor<2x4x5x64xf32>) -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
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
  }) {concatAxis = 2 : i64, expansionN = 3 : i64, finalLayout = "3DS", mulScalar = 1.000000e+00 : f32, noSaturation = true, reshapeCollapsedCount = 3 : i64, reshapeFirstCollapsedDim = 0 : i64, unsqueezedPosition = 2 : i64, yieldConcatResult = false} : (tensor<2x4x3x64xf32>, tensor<2x4x5x64xf32>) -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %0 : tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d0 * 12 + d1 * 3)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 floordiv 64)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0, d1) -> (d0 * 12 + d1 * 3 + 1)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0, d1) -> (d0 * 12 + d1 * 3 + 2)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64)>
// CHECK-DAG:   [[MAP_7_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64 + 8)>
// CHECK-DAG:   [[MAP_8_:#.+]] = affine_map<(d0) -> (d0 + 8)>
// CHECK-DAG:   [[MAP_9_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64 + 16)>
// CHECK-DAG:   [[MAP_10_:#.+]] = affine_map<(d0) -> (d0 + 16)>
// CHECK-DAG:   [[MAP_11_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64 + 24)>
// CHECK-DAG:   [[MAP_12_:#.+]] = affine_map<(d0) -> (d0 + 24)>
// CHECK-DAG:   [[MAP_13_:#.+]] = affine_map<(d0) -> (d0 + 3)>
// CHECK-LABEL:  func.func @concat_expand_stick_no_saturation
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4x3x64xf32>, [[PARAM_1_:%.+]]: memref<2x4x5x64xf32>) -> memref<24x8x64xf16, #map> {
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<24x8x64xf16, #map>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<24x8x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<24x8x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<24x8x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK-DAG:         [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 1){
// CHECK:               [[VAR_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[VAR_5_:%.+]] = affine.apply [[MAP_1_]]([[VAR_4_]]#1)
// CHECK-DAG:           [[VAR_6_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#0, [[VAR_1_]]#1)
// CHECK:               [[VAR_7_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_6_]], [[VAR_4_]]#0, [[VAR_5_]]{{.}} : memref<24x8x64xf16, #map>
// CHECK-DAG:           [[VAR_8_:%.+]] = affine.apply [[MAP_3_]]([[VAR_7_]])
// CHECK-DAG:           [[VAR_9_:%.+]] = affine.apply [[MAP_4_]]([[VAR_1_]]#0, [[VAR_1_]]#1)
// CHECK:               [[VAR_10_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_9_]], [[VAR_4_]]#0, [[VAR_5_]]{{.}} : memref<24x8x64xf16, #map>
// CHECK-DAG:           [[VAR_11_:%.+]] = affine.apply [[MAP_3_]]([[VAR_10_]])
// CHECK-DAG:           [[VAR_12_:%.+]] = affine.apply [[MAP_5_]]([[VAR_1_]]#0, [[VAR_1_]]#1)
// CHECK:               [[VAR_13_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_12_]], [[VAR_4_]]#0, [[VAR_5_]]{{.}} : memref<24x8x64xf16, #map>
// CHECK-DAG:           [[VAR_14_:%.+]] = affine.apply [[MAP_3_]]([[VAR_13_]])
// CHECK-DAG:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:               [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_2_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:               krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_2_]] -> [[I_4_:%.+]] = 0 to 64){
// CHECK:                 [[VAR_16_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:                 [[VAR_17_:%.+]] = affine.apply [[MAP_6_]]([[VAR_16_]], [[VAR_4_]]#1)
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_17_]]{{.}} : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_19_:%.+]] = arith.addi [[VAR_17_]], [[CST_4_]] : index
// CHECK:                 [[LOAD_PARAM_0_MEM_1_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_19_]]{{.}} : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK:                 [[VAR_21_:%.+]] = "zlow.vec_f32_to_dlf16"([[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_0_MEM_1_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                 vector.store [[VAR_21_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_8_]], [[VAR_16_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store [[VAR_21_]], [[VAR_reinterpret_cast_0_]]{{.}}[[VAR_11_]], [[VAR_16_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store [[VAR_21_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_14_]], [[VAR_16_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_22_:%.+]] = affine.apply [[MAP_7_]]([[VAR_16_]], [[VAR_4_]]#1)
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_2_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_22_]]{{.}} : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_24_:%.+]] = arith.addi [[VAR_22_]], [[CST_4_]] : index
// CHECK:                 [[LOAD_PARAM_0_MEM_3_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_24_]]{{.}} : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_26_:%.+]] = "zlow.vec_f32_to_dlf16"([[LOAD_PARAM_0_MEM_2_]], [[LOAD_PARAM_0_MEM_3_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:             [[VAR_27_:%.+]] = affine.apply [[MAP_8_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_26_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_8_]], [[VAR_27_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_28_:%.+]] = affine.apply [[MAP_8_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_26_]], [[VAR_reinterpret_cast_0_]]{{.}}[[VAR_11_]], [[VAR_28_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_29_:%.+]] = affine.apply [[MAP_8_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_26_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_14_]], [[VAR_29_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_30_:%.+]] = affine.apply [[MAP_9_]]([[VAR_16_]], [[VAR_4_]]#1)
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_4_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_30_]]{{.}} : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_32_:%.+]] = arith.addi [[VAR_30_]], [[CST_4_]] : index
// CHECK:                 [[LOAD_PARAM_0_MEM_5_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_32_]]{{.}} : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_34_:%.+]] = "zlow.vec_f32_to_dlf16"([[LOAD_PARAM_0_MEM_4_]], [[LOAD_PARAM_0_MEM_5_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:             [[VAR_35_:%.+]] = affine.apply [[MAP_10_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_34_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_8_]], [[VAR_35_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_36_:%.+]] = affine.apply [[MAP_10_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_34_]], [[VAR_reinterpret_cast_0_]]{{.}}[[VAR_11_]], [[VAR_36_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_37_:%.+]] = affine.apply [[MAP_10_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_34_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_14_]], [[VAR_37_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_38_:%.+]] = affine.apply [[MAP_11_]]([[VAR_16_]], [[VAR_4_]]#1)
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_6_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_38_]]{{.}} : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_40_:%.+]] = arith.addi [[VAR_38_]], [[CST_4_]] : index
// CHECK:                 [[LOAD_PARAM_0_MEM_7_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_40_]]{{.}} : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_42_:%.+]] = "zlow.vec_f32_to_dlf16"([[LOAD_PARAM_0_MEM_6_]], [[LOAD_PARAM_0_MEM_7_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:             [[VAR_43_:%.+]] = affine.apply [[MAP_12_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_42_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_8_]], [[VAR_43_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_44_:%.+]] = affine.apply [[MAP_12_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_42_]], [[VAR_reinterpret_cast_0_]]{{.}}[[VAR_11_]], [[VAR_44_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_45_:%.+]] = affine.apply [[MAP_12_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_42_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_14_]], [[VAR_45_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               }
// CHECK:             }
// CHECK:             [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to 5, [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to 1){
// CHECK:               [[VAR_4_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[VAR_5_1_:%.+]] = affine.apply [[MAP_1_]]([[VAR_4_1_]]#1)
// CHECK-DAG:           [[VAR_6_1_:%.+]] = affine.apply [[MAP_13_]]([[VAR_4_1_]]#0)
// CHECK-DAG:           [[VAR_7_1_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#0, [[VAR_1_]]#1)
// CHECK:               [[VAR_8_1_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_7_1_]], [[VAR_6_1_]], [[VAR_5_1_]]{{.}} : memref<24x8x64xf16, #map>
// CHECK-DAG:           [[VAR_9_1_:%.+]] = affine.apply [[MAP_3_]]([[VAR_8_1_]])
// CHECK-DAG:           [[VAR_10_1_:%.+]] = affine.apply [[MAP_4_]]([[VAR_1_]]#0, [[VAR_1_]]#1)
// CHECK:               [[VAR_11_1_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_10_1_]], [[VAR_6_1_]], [[VAR_5_1_]]{{.}} : memref<24x8x64xf16, #map>
// CHECK-DAG:           [[VAR_12_1_:%.+]] = affine.apply [[MAP_3_]]([[VAR_11_1_]])
// CHECK-DAG:           [[VAR_13_1_:%.+]] = affine.apply [[MAP_5_]]([[VAR_1_]]#0, [[VAR_1_]]#1)
// CHECK:               [[VAR_14_1_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_13_1_]], [[VAR_6_1_]], [[VAR_5_1_]]{{.}} : memref<24x8x64xf16, #map>
// CHECK-DAG:           [[LOOP_2_:%.+]] = affine.apply [[MAP_3_]]([[VAR_14_1_]])
// CHECK-DAG:           [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:               [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_4_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:               krnl.iterate([[BLOCK_TILE__1_]]) with ([[LOOP_4_]] -> [[I_7_:%.+]] = 0 to 64){
// CHECK:                 [[VAR_17_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__1_]]) : (!krnl.loop) -> index
// CHECK:                 [[LOAD_PARAM_0_MEM_8_:%.+]] = affine.apply [[MAP_6_]]([[VAR_17_1_]], [[VAR_4_1_]]#1)
// CHECK-DAG:             [[VAR_19_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_1_]]#0, [[LOAD_PARAM_0_MEM_8_]]{{.}} : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_1_:%.+]] = arith.addi [[LOAD_PARAM_0_MEM_8_]], [[CST_4_]] : index
// CHECK:                 [[VAR_21_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_1_]]#0, [[LOAD_PARAM_0_MEM_1_]]{{.}} : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK:                 [[VAR_22_1_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_19_1_]], [[VAR_21_1_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                 vector.store [[VAR_22_1_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_9_1_]], [[VAR_17_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store [[VAR_22_1_]], [[VAR_reinterpret_cast_0_]]{{.}}[[VAR_12_1_]], [[VAR_17_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store [[VAR_22_1_]], [[VAR_reinterpret_cast_1_]]{{.}}[[LOOP_2_]], [[VAR_17_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[LOAD_PARAM_0_MEM_2_:%.+]] = affine.apply [[MAP_7_]]([[VAR_17_1_]], [[VAR_4_1_]]#1)
// CHECK-DAG:             [[VAR_24_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_1_]]#0, [[LOAD_PARAM_0_MEM_2_]]{{.}} : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_3_:%.+]] = arith.addi [[LOAD_PARAM_0_MEM_2_]], [[CST_4_]] : index
// CHECK:                 [[VAR_26_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_1_]]#0, [[LOAD_PARAM_0_MEM_3_]]{{.}} : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_27_1_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_24_1_]], [[VAR_26_1_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:             [[VAR_28_1_:%.+]] = affine.apply [[MAP_8_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_27_1_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_9_1_]], [[VAR_28_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_29_1_:%.+]] = affine.apply [[MAP_8_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_27_1_]], [[VAR_reinterpret_cast_0_]]{{.}}[[VAR_12_1_]], [[VAR_29_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_30_1_:%.+]] = affine.apply [[MAP_8_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_27_1_]], [[VAR_reinterpret_cast_1_]]{{.}}[[LOOP_2_]], [[VAR_30_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[LOAD_PARAM_0_MEM_4_:%.+]] = affine.apply [[MAP_9_]]([[VAR_17_1_]], [[VAR_4_1_]]#1)
// CHECK-DAG:             [[VAR_32_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_1_]]#0, [[LOAD_PARAM_0_MEM_4_]]{{.}} : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_5_:%.+]] = arith.addi [[LOAD_PARAM_0_MEM_4_]], [[CST_4_]] : index
// CHECK:                 [[VAR_34_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_1_]]#0, [[LOAD_PARAM_0_MEM_5_]]{{.}} : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_35_1_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_32_1_]], [[VAR_34_1_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:             [[VAR_36_1_:%.+]] = affine.apply [[MAP_10_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_35_1_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_9_1_]], [[VAR_36_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_37_1_:%.+]] = affine.apply [[MAP_10_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_35_1_]], [[VAR_reinterpret_cast_0_]]{{.}}[[VAR_12_1_]], [[VAR_37_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_38_1_:%.+]] = affine.apply [[MAP_10_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_35_1_]], [[VAR_reinterpret_cast_1_]]{{.}}[[LOOP_2_]], [[VAR_38_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[LOAD_PARAM_0_MEM_6_:%.+]] = affine.apply [[MAP_11_]]([[VAR_17_1_]], [[VAR_4_1_]]#1)
// CHECK-DAG:             [[VAR_40_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_1_]]#0, [[LOAD_PARAM_0_MEM_6_]]{{.}} : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_7_:%.+]] = arith.addi [[LOAD_PARAM_0_MEM_6_]], [[CST_4_]] : index
// CHECK:                 [[VAR_42_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_1_]]#0, [[LOAD_PARAM_0_MEM_7_]]{{.}} : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_43_1_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_40_1_]], [[VAR_42_1_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:             [[VAR_44_1_:%.+]] = affine.apply [[MAP_12_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_43_1_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_9_1_]], [[VAR_44_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_45_1_:%.+]] = affine.apply [[MAP_12_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_43_1_]], [[VAR_reinterpret_cast_0_]]{{.}}[[VAR_12_1_]], [[VAR_45_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_46_:%.+]] = affine.apply [[MAP_12_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_43_1_]], [[VAR_reinterpret_cast_1_]]{{.}}[[LOOP_2_]], [[VAR_46_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<24x8x64xf16, #map>
// CHECK:         }
}

// -----

// yieldConcatResult = true: the concat result also has uses outside the
// chain (e.g. a KV-cache "present" passthrough), so the fused op has two
// results. The primary (stickified) result is computed exactly as above;
// the second result -- the concat's own plain result -- is a separate,
// independent materialization via two ordinary copy loops (one per
// operand), matching what ONNXConcatOpLowering would produce standalone.

func.func @concat_expand_stick_multi_output(%arg0: tensor<2x4x3x64xf32>, %arg1: tensor<2x4x5x64xf32>) -> (tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x4x8x64xf32>) {
  %0:2 = "onnx.Fused"(%arg0, %arg1) <{kind = "zhigh.concat-expand-stick"}> ({
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
    onnx.Yield %9, %4 : tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x4x8x64xf32>
  }) {concatAxis = 2 : i64, expansionN = 3 : i64, finalLayout = "3DS", mulScalar = 1.000000e+00 : f32, noSaturation = false, reshapeCollapsedCount = 3 : i64, reshapeFirstCollapsedDim = 0 : i64, unsqueezedPosition = 2 : i64, yieldConcatResult = true} : (tensor<2x4x3x64xf32>, tensor<2x4x5x64xf32>) -> (tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x4x8x64xf32>)
  return %0#0, %0#1 : tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<2x4x8x64xf32>
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d0 * 12 + d1 * 3)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 floordiv 64)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0, d1) -> (d0 * 12 + d1 * 3 + 1)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0, d1) -> (d0 * 12 + d1 * 3 + 2)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64)>
// CHECK-DAG:   [[MAP_7_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64 + 8)>
// CHECK-DAG:   [[MAP_8_:#.+]] = affine_map<(d0) -> (d0 + 8)>
// CHECK-DAG:   [[MAP_9_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64 + 16)>
// CHECK-DAG:   [[MAP_10_:#.+]] = affine_map<(d0) -> (d0 + 16)>
// CHECK-DAG:   [[MAP_11_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64 + 24)>
// CHECK-DAG:   [[MAP_12_:#.+]] = affine_map<(d0) -> (d0 + 24)>
// CHECK-DAG:   [[MAP_13_:#.+]] = affine_map<(d0) -> (d0 + 3)>
// CHECK-LABEL:  func.func @concat_expand_stick_multi_output
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4x3x64xf32>, [[PARAM_1_:%.+]]: memref<2x4x5x64xf32>) -> (memref<24x8x64xf16, #map>, memref<2x4x8x64xf32>) {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<-8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x4x8x64xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<24x8x64xf16, #map>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[RES_1_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<24x8x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[VAR_reinterpret_cast_2_:%.+]] = memref.reinterpret_cast [[RES_1_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<24x8x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[VAR_reinterpret_cast_3_:%.+]] = memref.reinterpret_cast [[RES_1_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<24x8x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK-DAG:         [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 1){
// CHECK:               [[VAR_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[VAR_5_:%.+]] = affine.apply [[MAP_1_]]([[VAR_4_]]#1)
// CHECK-DAG:           [[VAR_6_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#0, [[VAR_1_]]#1)
// CHECK:               [[VAR_7_:%.+]] = krnl.get_linear_offset_index [[RES_1_]] at {{.}}[[VAR_6_]], [[VAR_4_]]#0, [[VAR_5_]]{{.}} : memref<24x8x64xf16, #map>
// CHECK-DAG:           [[VAR_8_:%.+]] = affine.apply [[MAP_3_]]([[VAR_7_]])
// CHECK-DAG:           [[VAR_9_:%.+]] = affine.apply [[MAP_4_]]([[VAR_1_]]#0, [[VAR_1_]]#1)
// CHECK:               [[VAR_10_:%.+]] = krnl.get_linear_offset_index [[RES_1_]] at {{.}}[[VAR_9_]], [[VAR_4_]]#0, [[VAR_5_]]{{.}} : memref<24x8x64xf16, #map>
// CHECK-DAG:           [[VAR_11_:%.+]] = affine.apply [[MAP_3_]]([[VAR_10_]])
// CHECK-DAG:           [[VAR_12_:%.+]] = affine.apply [[MAP_5_]]([[VAR_1_]]#0, [[VAR_1_]]#1)
// CHECK:               [[VAR_13_:%.+]] = krnl.get_linear_offset_index [[RES_1_]] at {{.}}[[VAR_12_]], [[VAR_4_]]#0, [[VAR_5_]]{{.}} : memref<24x8x64xf16, #map>
// CHECK-DAG:           [[VAR_14_:%.+]] = affine.apply [[MAP_3_]]([[VAR_13_]])
// CHECK-DAG:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:               [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_2_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:               krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_2_]] -> [[I_4_:%.+]] = 0 to 64){
// CHECK:                 [[VAR_16_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:                 [[VAR_17_:%.+]] = affine.apply [[MAP_6_]]([[VAR_16_]], [[VAR_4_]]#1)
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_17_]]{{.}} : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_19_:%.+]] = arith.addi [[VAR_17_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_1_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_19_]]{{.}} : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_21_:%.+]] = affine.apply [[MAP_6_]]([[VAR_16_]], [[VAR_4_]]#1)
// CHECK:                 vector.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_21_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK:                 [[VAR_22_:%.+]] = arith.addi [[VAR_21_]], [[CST_4_]] : index
// CHECK:                 vector.store [[LOAD_PARAM_0_MEM_1_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_22_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_23_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_24_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_25_:%.+]] = arith.maxnumf [[VAR_23_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_26_:%.+]] = arith.maxnumf [[VAR_24_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_27_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_25_]], [[VAR_26_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                 vector.store [[VAR_27_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_8_]], [[VAR_16_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store [[VAR_27_]], [[VAR_reinterpret_cast_2_]]{{.}}[[VAR_11_]], [[VAR_16_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store [[VAR_27_]], [[VAR_reinterpret_cast_3_]]{{.}}[[VAR_14_]], [[VAR_16_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_28_:%.+]] = affine.apply [[MAP_7_]]([[VAR_16_]], [[VAR_4_]]#1)
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_2_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_28_]]{{.}} : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_30_:%.+]] = arith.addi [[VAR_28_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_3_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_30_]]{{.}} : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_32_:%.+]] = affine.apply [[MAP_7_]]([[VAR_16_]], [[VAR_4_]]#1)
// CHECK:                 vector.store [[LOAD_PARAM_0_MEM_2_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_32_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK:                 [[VAR_33_:%.+]] = arith.addi [[VAR_32_]], [[CST_4_]] : index
// CHECK:                 vector.store [[LOAD_PARAM_0_MEM_3_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_33_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_34_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_2_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_35_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_3_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_36_:%.+]] = arith.maxnumf [[VAR_34_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_37_:%.+]] = arith.maxnumf [[VAR_35_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_38_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_36_]], [[VAR_37_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:             [[VAR_39_:%.+]] = affine.apply [[MAP_8_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_38_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_8_]], [[VAR_39_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_40_:%.+]] = affine.apply [[MAP_8_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_38_]], [[VAR_reinterpret_cast_2_]]{{.}}[[VAR_11_]], [[VAR_40_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_41_:%.+]] = affine.apply [[MAP_8_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_38_]], [[VAR_reinterpret_cast_3_]]{{.}}[[VAR_14_]], [[VAR_41_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_42_:%.+]] = affine.apply [[MAP_9_]]([[VAR_16_]], [[VAR_4_]]#1)
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_4_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_42_]]{{.}} : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_44_:%.+]] = arith.addi [[VAR_42_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_5_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_44_]]{{.}} : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_46_:%.+]] = affine.apply [[MAP_9_]]([[VAR_16_]], [[VAR_4_]]#1)
// CHECK:                 vector.store [[LOAD_PARAM_0_MEM_4_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_46_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK:                 [[VAR_47_:%.+]] = arith.addi [[VAR_46_]], [[CST_4_]] : index
// CHECK:                 vector.store [[LOAD_PARAM_0_MEM_5_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_47_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_48_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_4_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_49_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_5_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_50_:%.+]] = arith.maxnumf [[VAR_48_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_51_:%.+]] = arith.maxnumf [[VAR_49_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_52_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_50_]], [[VAR_51_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:             [[VAR_53_:%.+]] = affine.apply [[MAP_10_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_52_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_8_]], [[VAR_53_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_54_:%.+]] = affine.apply [[MAP_10_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_52_]], [[VAR_reinterpret_cast_2_]]{{.}}[[VAR_11_]], [[VAR_54_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_55_:%.+]] = affine.apply [[MAP_10_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_52_]], [[VAR_reinterpret_cast_3_]]{{.}}[[VAR_14_]], [[VAR_55_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_56_:%.+]] = affine.apply [[MAP_11_]]([[VAR_16_]], [[VAR_4_]]#1)
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_6_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_56_]]{{.}} : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_58_:%.+]] = arith.addi [[VAR_56_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_7_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_58_]]{{.}} : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_60_:%.+]] = affine.apply [[MAP_11_]]([[VAR_16_]], [[VAR_4_]]#1)
// CHECK:                 vector.store [[LOAD_PARAM_0_MEM_6_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_60_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK:                 [[VAR_61_:%.+]] = arith.addi [[VAR_60_]], [[CST_4_]] : index
// CHECK:                 vector.store [[LOAD_PARAM_0_MEM_7_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_61_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_62_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_6_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_63_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_7_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_64_:%.+]] = arith.maxnumf [[VAR_62_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_65_:%.+]] = arith.maxnumf [[VAR_63_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_66_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_64_]], [[VAR_65_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:             [[VAR_67_:%.+]] = affine.apply [[MAP_12_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_66_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_8_]], [[VAR_67_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_68_:%.+]] = affine.apply [[MAP_12_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_66_]], [[VAR_reinterpret_cast_2_]]{{.}}[[VAR_11_]], [[VAR_68_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_69_:%.+]] = affine.apply [[MAP_12_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_66_]], [[VAR_reinterpret_cast_3_]]{{.}}[[VAR_14_]], [[VAR_69_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               }
// CHECK:             }
// CHECK:             [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to 5, [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to 1){
// CHECK:               [[VAR_4_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[VAR_5_1_:%.+]] = affine.apply [[MAP_1_]]([[VAR_4_1_]]#1)
// CHECK-DAG:           [[VAR_6_1_:%.+]] = affine.apply [[MAP_13_]]([[VAR_4_1_]]#0)
// CHECK-DAG:           [[VAR_7_1_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#0, [[VAR_1_]]#1)
// CHECK:               [[VAR_8_1_:%.+]] = krnl.get_linear_offset_index [[RES_1_]] at {{.}}[[VAR_7_1_]], [[VAR_6_1_]], [[VAR_5_1_]]{{.}} : memref<24x8x64xf16, #map>
// CHECK-DAG:           [[VAR_9_1_:%.+]] = affine.apply [[MAP_3_]]([[VAR_8_1_]])
// CHECK-DAG:           [[VAR_10_1_:%.+]] = affine.apply [[MAP_4_]]([[VAR_1_]]#0, [[VAR_1_]]#1)
// CHECK:               [[VAR_11_1_:%.+]] = krnl.get_linear_offset_index [[RES_1_]] at {{.}}[[VAR_10_1_]], [[VAR_6_1_]], [[VAR_5_1_]]{{.}} : memref<24x8x64xf16, #map>
// CHECK-DAG:           [[VAR_12_1_:%.+]] = affine.apply [[MAP_3_]]([[VAR_11_1_]])
// CHECK-DAG:           [[VAR_13_1_:%.+]] = affine.apply [[MAP_5_]]([[VAR_1_]]#0, [[VAR_1_]]#1)
// CHECK:               [[VAR_14_1_:%.+]] = krnl.get_linear_offset_index [[RES_1_]] at {{.}}[[VAR_13_1_]], [[VAR_6_1_]], [[VAR_5_1_]]{{.}} : memref<24x8x64xf16, #map>
// CHECK-DAG:           [[LOOP_2_:%.+]] = affine.apply [[MAP_3_]]([[VAR_14_1_]])
// CHECK-DAG:           [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:               [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_4_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:               krnl.iterate([[BLOCK_TILE__1_]]) with ([[LOOP_4_]] -> [[I_7_:%.+]] = 0 to 64){
// CHECK:                 [[VAR_17_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__1_]]) : (!krnl.loop) -> index
// CHECK:                 [[LOAD_PARAM_0_MEM_8_:%.+]] = affine.apply [[MAP_6_]]([[VAR_17_1_]], [[VAR_4_1_]]#1)
// CHECK-DAG:             [[VAR_19_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_1_]]#0, [[LOAD_PARAM_0_MEM_8_]]{{.}} : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_1_:%.+]] = arith.addi [[LOAD_PARAM_0_MEM_8_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_21_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_1_]]#0, [[LOAD_PARAM_0_MEM_1_]]{{.}} : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_22_1_:%.+]] = affine.apply [[MAP_6_]]([[VAR_17_1_]], [[VAR_4_1_]]#1)
// CHECK:                 vector.store [[VAR_19_1_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_6_1_]], [[VAR_22_1_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK:                 [[VAR_23_1_:%.+]] = arith.addi [[VAR_22_1_]], [[CST_4_]] : index
// CHECK:                 vector.store [[VAR_21_1_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_6_1_]], [[VAR_23_1_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_24_1_:%.+]] = arith.minnumf [[VAR_19_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_25_1_:%.+]] = arith.minnumf [[VAR_21_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_26_1_:%.+]] = arith.maxnumf [[VAR_24_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_27_1_:%.+]] = arith.maxnumf [[VAR_25_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_28_1_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_26_1_]], [[VAR_27_1_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                 vector.store [[VAR_28_1_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_9_1_]], [[VAR_17_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store [[VAR_28_1_]], [[VAR_reinterpret_cast_2_]]{{.}}[[VAR_12_1_]], [[VAR_17_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store [[VAR_28_1_]], [[VAR_reinterpret_cast_3_]]{{.}}[[LOOP_2_]], [[VAR_17_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[LOAD_PARAM_0_MEM_2_:%.+]] = affine.apply [[MAP_7_]]([[VAR_17_1_]], [[VAR_4_1_]]#1)
// CHECK-DAG:             [[VAR_30_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_1_]]#0, [[LOAD_PARAM_0_MEM_2_]]{{.}} : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_3_:%.+]] = arith.addi [[LOAD_PARAM_0_MEM_2_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_32_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_1_]]#0, [[LOAD_PARAM_0_MEM_3_]]{{.}} : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_33_1_:%.+]] = affine.apply [[MAP_7_]]([[VAR_17_1_]], [[VAR_4_1_]]#1)
// CHECK:                 vector.store [[VAR_30_1_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_6_1_]], [[VAR_33_1_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK:                 [[VAR_34_1_:%.+]] = arith.addi [[VAR_33_1_]], [[CST_4_]] : index
// CHECK:                 vector.store [[VAR_32_1_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_6_1_]], [[VAR_34_1_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_35_1_:%.+]] = arith.minnumf [[VAR_30_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_36_1_:%.+]] = arith.minnumf [[VAR_32_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_37_1_:%.+]] = arith.maxnumf [[VAR_35_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_38_1_:%.+]] = arith.maxnumf [[VAR_36_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_39_1_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_37_1_]], [[VAR_38_1_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:             [[VAR_40_1_:%.+]] = affine.apply [[MAP_8_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_39_1_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_9_1_]], [[VAR_40_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_41_1_:%.+]] = affine.apply [[MAP_8_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_39_1_]], [[VAR_reinterpret_cast_2_]]{{.}}[[VAR_12_1_]], [[VAR_41_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_42_1_:%.+]] = affine.apply [[MAP_8_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_39_1_]], [[VAR_reinterpret_cast_3_]]{{.}}[[LOOP_2_]], [[VAR_42_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[LOAD_PARAM_0_MEM_4_:%.+]] = affine.apply [[MAP_9_]]([[VAR_17_1_]], [[VAR_4_1_]]#1)
// CHECK-DAG:             [[VAR_44_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_1_]]#0, [[LOAD_PARAM_0_MEM_4_]]{{.}} : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_5_:%.+]] = arith.addi [[LOAD_PARAM_0_MEM_4_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_46_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_1_]]#0, [[LOAD_PARAM_0_MEM_5_]]{{.}} : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_47_1_:%.+]] = affine.apply [[MAP_9_]]([[VAR_17_1_]], [[VAR_4_1_]]#1)
// CHECK:                 vector.store [[VAR_44_1_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_6_1_]], [[VAR_47_1_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK:                 [[VAR_48_1_:%.+]] = arith.addi [[VAR_47_1_]], [[CST_4_]] : index
// CHECK:                 vector.store [[VAR_46_1_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_6_1_]], [[VAR_48_1_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_49_1_:%.+]] = arith.minnumf [[VAR_44_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_50_1_:%.+]] = arith.minnumf [[VAR_46_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_51_1_:%.+]] = arith.maxnumf [[VAR_49_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_52_1_:%.+]] = arith.maxnumf [[VAR_50_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_53_1_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_51_1_]], [[VAR_52_1_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:             [[VAR_54_1_:%.+]] = affine.apply [[MAP_10_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_53_1_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_9_1_]], [[VAR_54_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_55_1_:%.+]] = affine.apply [[MAP_10_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_53_1_]], [[VAR_reinterpret_cast_2_]]{{.}}[[VAR_12_1_]], [[VAR_55_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_56_1_:%.+]] = affine.apply [[MAP_10_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_53_1_]], [[VAR_reinterpret_cast_3_]]{{.}}[[LOOP_2_]], [[VAR_56_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[LOAD_PARAM_0_MEM_6_:%.+]] = affine.apply [[MAP_11_]]([[VAR_17_1_]], [[VAR_4_1_]]#1)
// CHECK-DAG:             [[VAR_58_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_1_]]#0, [[LOAD_PARAM_0_MEM_6_]]{{.}} : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_7_:%.+]] = arith.addi [[LOAD_PARAM_0_MEM_6_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_60_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_1_]]#0, [[LOAD_PARAM_0_MEM_7_]]{{.}} : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_61_1_:%.+]] = affine.apply [[MAP_11_]]([[VAR_17_1_]], [[VAR_4_1_]]#1)
// CHECK:                 vector.store [[VAR_58_1_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_6_1_]], [[VAR_61_1_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK:                 [[VAR_62_1_:%.+]] = arith.addi [[VAR_61_1_]], [[CST_4_]] : index
// CHECK:                 vector.store [[VAR_60_1_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_6_1_]], [[VAR_62_1_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_63_1_:%.+]] = arith.minnumf [[VAR_58_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_64_1_:%.+]] = arith.minnumf [[VAR_60_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_65_1_:%.+]] = arith.maxnumf [[VAR_63_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_66_1_:%.+]] = arith.maxnumf [[VAR_64_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_67_1_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_65_1_]], [[VAR_66_1_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:             [[VAR_68_1_:%.+]] = affine.apply [[MAP_12_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_67_1_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_9_1_]], [[VAR_68_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_69_1_:%.+]] = affine.apply [[MAP_12_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_67_1_]], [[VAR_reinterpret_cast_2_]]{{.}}[[VAR_12_1_]], [[VAR_69_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_70_:%.+]] = affine.apply [[MAP_12_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_67_1_]], [[VAR_reinterpret_cast_3_]]{{.}}[[LOOP_2_]], [[VAR_70_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_1_]], [[RES_]] : memref<24x8x64xf16, #map>, memref<2x4x8x64xf32>
// CHECK:         }

}

// -----

// finalLayout = "4D" (the third option in the {3D, 3DS, 4D} allow-list,
// alongside the 3DS cases above): unlike 3D/3DS, LAYOUT_4D keeps all four
// axes live in the stick's own affine map (compare [[MAP_0_]] below, whose
// third component is the real dim1 [[MAP_0_]]'s "d1", to the 3DS cases
// above, whose corresponding component is a literal 0) -- so this also
// exercises a genuinely different stick-map shape, not just a different
// attribute value. Reached with unsqueezedPosition = 1 (rather than 2 in the
// 3DS cases): the outer loop over dims [0, concatAxis) still covers both
// dims 0 and 1 either way, but here the Reshape collapses only dims [0, 1]
// (reshapeCollapsedCount = 2) instead of [0, 2], leaving 4 result dims
// instead of 3.

func.func @concat_expand_stick_layout_4d(%arg0: tensor<2x4x3x64xf32>, %arg1: tensor<2x4x5x64xf32>) -> tensor<6x4x8x64xf16, #zhigh.layout<{dataLayout = "4D"}>> {
  %0 = "onnx.Fused"(%arg0, %arg1) <{kind = "zhigh.concat-expand-stick"}> ({
  ^bb0(%arg2: tensor<2x4x3x64xf32>, %arg3: tensor<2x4x5x64xf32>):
    %1 = onnx.Constant dense<[6, 4, 8, 64]> : tensor<4xi64>
    %2 = onnx.Constant dense<[2, 3, 4, 8, 64]> : tensor<5xi64>
    %3 = onnx.Constant dense<1> : tensor<1xi64>
    %4 = "onnx.Concat"(%arg2, %arg3) <{axis = 2 : si64}> : (tensor<2x4x3x64xf32>, tensor<2x4x5x64xf32>) -> tensor<2x4x8x64xf32>
    %5 = "onnx.Unsqueeze"(%4, %3) : (tensor<2x4x8x64xf32>, tensor<1xi64>) -> tensor<2x1x4x8x64xf32>
    %6 = "zhigh.F32ToDLF16"(%5) : (tensor<2x1x4x8x64xf32>) -> tensor<2x1x4x8x64xf16>
    %7 = "onnx.Expand"(%6, %2) : (tensor<2x1x4x8x64xf16>, tensor<5xi64>) -> tensor<2x3x4x8x64xf16>
    %8 = "onnx.Reshape"(%7, %1) <{allowzero = 0 : si64}> : (tensor<2x3x4x8x64xf16>, tensor<4xi64>) -> tensor<6x4x8x64xf16>
    %9 = "onnx.LayoutTransform"(%8) <{target_layout = #zhigh.layout<{dataLayout = "4D"}>}> : (tensor<6x4x8x64xf16>) -> tensor<6x4x8x64xf16, #zhigh.layout<{dataLayout = "4D"}>>
    onnx.Yield %9 : tensor<6x4x8x64xf16, #zhigh.layout<{dataLayout = "4D"}>>
  }) {concatAxis = 2 : i64, expansionN = 3 : i64, finalLayout = "4D", mulScalar = 1.000000e+00 : f32, noSaturation = false, reshapeCollapsedCount = 2 : i64, reshapeFirstCollapsedDim = 0 : i64, unsqueezedPosition = 1 : i64, yieldConcatResult = false} : (tensor<2x4x3x64xf32>, tensor<2x4x5x64xf32>) -> tensor<6x4x8x64xf16, #zhigh.layout<{dataLayout = "4D"}>>
  return %0 : tensor<6x4x8x64xf16, #zhigh.layout<{dataLayout = "4D"}>>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 * 3)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 floordiv 64)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 * 3 + 1)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0) -> (d0 * 3 + 2)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64)>
// CHECK-DAG:   [[MAP_7_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64 + 8)>
// CHECK-DAG:   [[MAP_8_:#.+]] = affine_map<(d0) -> (d0 + 8)>
// CHECK-DAG:   [[MAP_9_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64 + 16)>
// CHECK-DAG:   [[MAP_10_:#.+]] = affine_map<(d0) -> (d0 + 16)>
// CHECK-DAG:   [[MAP_11_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64 + 24)>
// CHECK-DAG:   [[MAP_12_:#.+]] = affine_map<(d0) -> (d0 + 24)>
// CHECK-DAG:   [[MAP_13_:#.+]] = affine_map<(d0) -> (d0 + 3)>
// CHECK-LABEL:  func.func @concat_expand_stick_layout_4d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4x3x64xf32>, [[PARAM_1_:%.+]]: memref<2x4x5x64xf32>) -> memref<6x4x8x64xf16, #map> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<-8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<6x4x8x64xf16, #map>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<6x4x8x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<6x4x8x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[VAR_reinterpret_cast_2_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<6x4x8x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK-DAG:         [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 1){
// CHECK:               [[VAR_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[VAR_5_:%.+]] = affine.apply [[MAP_1_]]([[VAR_4_]]#1)
// CHECK-DAG:           [[VAR_6_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#0)
// CHECK:               [[VAR_7_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_6_]], [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_5_]]{{.}} : memref<6x4x8x64xf16, #map>
// CHECK-DAG:           [[VAR_8_:%.+]] = affine.apply [[MAP_3_]]([[VAR_7_]])
// CHECK-DAG:           [[VAR_9_:%.+]] = affine.apply [[MAP_4_]]([[VAR_1_]]#0)
// CHECK:               [[VAR_10_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_9_]], [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_5_]]{{.}} : memref<6x4x8x64xf16, #map>
// CHECK-DAG:           [[VAR_11_:%.+]] = affine.apply [[MAP_3_]]([[VAR_10_]])
// CHECK-DAG:           [[VAR_12_:%.+]] = affine.apply [[MAP_5_]]([[VAR_1_]]#0)
// CHECK:               [[VAR_13_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_12_]], [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_5_]]{{.}} : memref<6x4x8x64xf16, #map>
// CHECK-DAG:           [[VAR_14_:%.+]] = affine.apply [[MAP_3_]]([[VAR_13_]])
// CHECK-DAG:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:               [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_2_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:               krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_2_]] -> [[I_4_:%.+]] = 0 to 64){
// CHECK:                 [[VAR_16_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:                 [[VAR_17_:%.+]] = affine.apply [[MAP_6_]]([[VAR_16_]], [[VAR_4_]]#1)
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_17_]]{{.}} : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_19_:%.+]] = arith.addi [[VAR_17_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_1_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_19_]]{{.}} : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_21_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_22_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_23_:%.+]] = arith.maxnumf [[VAR_21_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_24_:%.+]] = arith.maxnumf [[VAR_22_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_25_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_23_]], [[VAR_24_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                 vector.store [[VAR_25_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_8_]], [[VAR_16_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store [[VAR_25_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_11_]], [[VAR_16_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store [[VAR_25_]], [[VAR_reinterpret_cast_2_]]{{.}}[[VAR_14_]], [[VAR_16_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_26_:%.+]] = affine.apply [[MAP_7_]]([[VAR_16_]], [[VAR_4_]]#1)
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_2_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_26_]]{{.}} : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_28_:%.+]] = arith.addi [[VAR_26_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_3_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_28_]]{{.}} : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_30_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_2_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_31_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_3_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_32_:%.+]] = arith.maxnumf [[VAR_30_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_33_:%.+]] = arith.maxnumf [[VAR_31_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_34_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_32_]], [[VAR_33_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:             [[VAR_35_:%.+]] = affine.apply [[MAP_8_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_34_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_8_]], [[VAR_35_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_36_:%.+]] = affine.apply [[MAP_8_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_34_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_11_]], [[VAR_36_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_37_:%.+]] = affine.apply [[MAP_8_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_34_]], [[VAR_reinterpret_cast_2_]]{{.}}[[VAR_14_]], [[VAR_37_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_38_:%.+]] = affine.apply [[MAP_9_]]([[VAR_16_]], [[VAR_4_]]#1)
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_4_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_38_]]{{.}} : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_40_:%.+]] = arith.addi [[VAR_38_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_5_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_40_]]{{.}} : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_42_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_4_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_43_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_5_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_44_:%.+]] = arith.maxnumf [[VAR_42_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_45_:%.+]] = arith.maxnumf [[VAR_43_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_46_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_44_]], [[VAR_45_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:             [[VAR_47_:%.+]] = affine.apply [[MAP_10_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_46_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_8_]], [[VAR_47_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_48_:%.+]] = affine.apply [[MAP_10_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_46_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_11_]], [[VAR_48_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_49_:%.+]] = affine.apply [[MAP_10_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_46_]], [[VAR_reinterpret_cast_2_]]{{.}}[[VAR_14_]], [[VAR_49_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_50_:%.+]] = affine.apply [[MAP_11_]]([[VAR_16_]], [[VAR_4_]]#1)
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_6_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_50_]]{{.}} : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_52_:%.+]] = arith.addi [[VAR_50_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_7_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]]#0, [[VAR_52_]]{{.}} : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_54_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_6_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_55_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_7_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_56_:%.+]] = arith.maxnumf [[VAR_54_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_57_:%.+]] = arith.maxnumf [[VAR_55_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_58_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_56_]], [[VAR_57_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:             [[VAR_59_:%.+]] = affine.apply [[MAP_12_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_58_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_8_]], [[VAR_59_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_60_:%.+]] = affine.apply [[MAP_12_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_58_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_11_]], [[VAR_60_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_61_:%.+]] = affine.apply [[MAP_12_]]([[VAR_16_]])
// CHECK:                 vector.store [[VAR_58_]], [[VAR_reinterpret_cast_2_]]{{.}}[[VAR_14_]], [[VAR_61_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               }
// CHECK:             }
// CHECK:             [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to 5, [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to 1){
// CHECK:               [[VAR_4_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[VAR_5_1_:%.+]] = affine.apply [[MAP_1_]]([[VAR_4_1_]]#1)
// CHECK-DAG:           [[VAR_6_1_:%.+]] = affine.apply [[MAP_13_]]([[VAR_4_1_]]#0)
// CHECK-DAG:           [[VAR_7_1_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#0)
// CHECK:               [[VAR_8_1_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_7_1_]], [[VAR_1_]]#1, [[VAR_6_1_]], [[VAR_5_1_]]{{.}} : memref<6x4x8x64xf16, #map>
// CHECK-DAG:           [[VAR_9_1_:%.+]] = affine.apply [[MAP_3_]]([[VAR_8_1_]])
// CHECK-DAG:           [[VAR_10_1_:%.+]] = affine.apply [[MAP_4_]]([[VAR_1_]]#0)
// CHECK:               [[VAR_11_1_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_10_1_]], [[VAR_1_]]#1, [[VAR_6_1_]], [[VAR_5_1_]]{{.}} : memref<6x4x8x64xf16, #map>
// CHECK-DAG:           [[VAR_12_1_:%.+]] = affine.apply [[MAP_3_]]([[VAR_11_1_]])
// CHECK-DAG:           [[VAR_13_1_:%.+]] = affine.apply [[MAP_5_]]([[VAR_1_]]#0)
// CHECK:               [[VAR_14_1_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_13_1_]], [[VAR_1_]]#1, [[VAR_6_1_]], [[VAR_5_1_]]{{.}} : memref<6x4x8x64xf16, #map>
// CHECK-DAG:           [[LOOP_2_:%.+]] = affine.apply [[MAP_3_]]([[VAR_14_1_]])
// CHECK-DAG:           [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:               [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_4_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:               krnl.iterate([[BLOCK_TILE__1_]]) with ([[LOOP_4_]] -> [[I_7_:%.+]] = 0 to 64){
// CHECK:                 [[VAR_17_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__1_]]) : (!krnl.loop) -> index
// CHECK:                 [[LOAD_PARAM_0_MEM_8_:%.+]] = affine.apply [[MAP_6_]]([[VAR_17_1_]], [[VAR_4_1_]]#1)
// CHECK-DAG:             [[VAR_19_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_1_]]#0, [[LOAD_PARAM_0_MEM_8_]]{{.}} : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_1_:%.+]] = arith.addi [[LOAD_PARAM_0_MEM_8_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_21_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_1_]]#0, [[LOAD_PARAM_0_MEM_1_]]{{.}} : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_22_1_:%.+]] = arith.minnumf [[VAR_19_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_23_1_:%.+]] = arith.minnumf [[VAR_21_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_24_1_:%.+]] = arith.maxnumf [[VAR_22_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_25_1_:%.+]] = arith.maxnumf [[VAR_23_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_26_1_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_24_1_]], [[VAR_25_1_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                 vector.store [[VAR_26_1_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_9_1_]], [[VAR_17_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store [[VAR_26_1_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_1_]], [[VAR_17_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store [[VAR_26_1_]], [[VAR_reinterpret_cast_2_]]{{.}}[[LOOP_2_]], [[VAR_17_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[LOAD_PARAM_0_MEM_2_:%.+]] = affine.apply [[MAP_7_]]([[VAR_17_1_]], [[VAR_4_1_]]#1)
// CHECK-DAG:             [[VAR_28_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_1_]]#0, [[LOAD_PARAM_0_MEM_2_]]{{.}} : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_3_:%.+]] = arith.addi [[LOAD_PARAM_0_MEM_2_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_30_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_1_]]#0, [[LOAD_PARAM_0_MEM_3_]]{{.}} : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_31_1_:%.+]] = arith.minnumf [[VAR_28_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_32_1_:%.+]] = arith.minnumf [[VAR_30_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_33_1_:%.+]] = arith.maxnumf [[VAR_31_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_34_1_:%.+]] = arith.maxnumf [[VAR_32_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_35_1_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_33_1_]], [[VAR_34_1_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:             [[VAR_36_1_:%.+]] = affine.apply [[MAP_8_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_35_1_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_9_1_]], [[VAR_36_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_37_1_:%.+]] = affine.apply [[MAP_8_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_35_1_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_1_]], [[VAR_37_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_38_1_:%.+]] = affine.apply [[MAP_8_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_35_1_]], [[VAR_reinterpret_cast_2_]]{{.}}[[LOOP_2_]], [[VAR_38_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[LOAD_PARAM_0_MEM_4_:%.+]] = affine.apply [[MAP_9_]]([[VAR_17_1_]], [[VAR_4_1_]]#1)
// CHECK-DAG:             [[VAR_40_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_1_]]#0, [[LOAD_PARAM_0_MEM_4_]]{{.}} : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_5_:%.+]] = arith.addi [[LOAD_PARAM_0_MEM_4_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_42_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_1_]]#0, [[LOAD_PARAM_0_MEM_5_]]{{.}} : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_43_1_:%.+]] = arith.minnumf [[VAR_40_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_44_1_:%.+]] = arith.minnumf [[VAR_42_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_45_1_:%.+]] = arith.maxnumf [[VAR_43_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_46_1_:%.+]] = arith.maxnumf [[VAR_44_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_47_1_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_45_1_]], [[VAR_46_1_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:             [[VAR_48_1_:%.+]] = affine.apply [[MAP_10_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_47_1_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_9_1_]], [[VAR_48_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_49_1_:%.+]] = affine.apply [[MAP_10_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_47_1_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_1_]], [[VAR_49_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_50_1_:%.+]] = affine.apply [[MAP_10_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_47_1_]], [[VAR_reinterpret_cast_2_]]{{.}}[[LOOP_2_]], [[VAR_50_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[LOAD_PARAM_0_MEM_6_:%.+]] = affine.apply [[MAP_11_]]([[VAR_17_1_]], [[VAR_4_1_]]#1)
// CHECK-DAG:             [[VAR_52_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_1_]]#0, [[LOAD_PARAM_0_MEM_6_]]{{.}} : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_7_:%.+]] = arith.addi [[LOAD_PARAM_0_MEM_6_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_54_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_1_]]#0, [[LOAD_PARAM_0_MEM_7_]]{{.}} : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_55_1_:%.+]] = arith.minnumf [[VAR_52_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_56_1_:%.+]] = arith.minnumf [[VAR_54_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_57_1_:%.+]] = arith.maxnumf [[VAR_55_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_58_1_:%.+]] = arith.maxnumf [[VAR_56_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_59_1_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_57_1_]], [[VAR_58_1_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:             [[VAR_60_1_:%.+]] = affine.apply [[MAP_12_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_59_1_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_9_1_]], [[VAR_60_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_61_1_:%.+]] = affine.apply [[MAP_12_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_59_1_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_12_1_]], [[VAR_61_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_62_:%.+]] = affine.apply [[MAP_12_]]([[VAR_17_1_]])
// CHECK:                 vector.store [[VAR_59_1_]], [[VAR_reinterpret_cast_2_]]{{.}}[[LOOP_2_]], [[VAR_62_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<6x4x8x64xf16, #map>
// CHECK:         }
}

