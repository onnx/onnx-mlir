// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Dedicated lowering for zhigh.concat-expand-stick with dynamic shapes: the
// batch dim (axis 0) and the concat axis itself (axis 2, i.e. the KV-cache
// sequence-length dim -- the realistic dynamic case for this pattern) are
// both dynamic; only the head-count dim (axis 1, static 4) and the
// stick-aligned innermost dim (axis 3, static 64, required by fusion
// detection) stay static.
//
// This specifically exercises two scope-crossing fixes in the lowering: the
// outer loop's own induction variables (outerIndices) and the axis-A shift
// amount added to operand 2's coordinate (axisAShift, here a genuine
// memref.dim-derived value since the concat axis is dynamic) both have to
// be re-homed into the current (nested) IndexExprScope before being used in
// arithmetic there -- Dim-kind index exprs, unlike literals, cannot cross a
// scope boundary as-is. Look for the axis-A shift showing up as an affine
// symbol (affine_map<(d0)[s0] -> (d0 + s0)> below) rather than a literal
// offset, confirming the dynamic value flows through correctly.

func.func @concat_expand_stick_dynamic(%arg0: tensor<?x4x?x64xf32>, %arg1: tensor<?x4x?x64xf32>) -> tensor<?x?x64xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %0 = "onnx.Fused"(%arg0, %arg1) <{kind = "zhigh.concat-expand-stick"}> ({
  ^bb0(%arg2: tensor<?x4x?x64xf32>, %arg3: tensor<?x4x?x64xf32>):
    %1 = onnx.Constant dense<[-1, -1, 64]> : tensor<3xi64>
    %2 = onnx.Constant dense<[-1, 4, 8, -1, 64]> : tensor<5xi64>
    %3 = onnx.Constant dense<2> : tensor<1xi64>
    %4 = "onnx.Concat"(%arg2, %arg3) <{axis = 2 : si64}> : (tensor<?x4x?x64xf32>, tensor<?x4x?x64xf32>) -> tensor<?x4x?x64xf32>
    %5 = "onnx.Unsqueeze"(%4, %3) : (tensor<?x4x?x64xf32>, tensor<1xi64>) -> tensor<?x4x1x?x64xf32>
    %6 = "zhigh.F32ToDLF16"(%5) : (tensor<?x4x1x?x64xf32>) -> tensor<?x4x1x?x64xf16>
    %7 = "onnx.Expand"(%6, %2) : (tensor<?x4x1x?x64xf16>, tensor<5xi64>) -> tensor<?x4x8x?x64xf16>
    %8 = "onnx.Reshape"(%7, %1) <{allowzero = 0 : si64}> : (tensor<?x4x8x?x64xf16>, tensor<3xi64>) -> tensor<?x?x64xf16>
    %9 = "onnx.LayoutTransform"(%8) <{target_layout = #zhigh.layout<{dataLayout = "3DS"}>}> : (tensor<?x?x64xf16>) -> tensor<?x?x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    onnx.Yield %9 : tensor<?x?x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  }) {concatAxis = 2 : i64, expansionN = 8 : i64, finalLayout = "3DS", noSaturation = false, reshapeCollapsedCount = 3 : i64, reshapeFirstCollapsedDim = 0 : i64, unsqueezedPosition = 2 : i64, yieldConcatResult = false} : (tensor<?x4x?x64xf32>, tensor<?x4x?x64xf32>) -> tensor<?x?x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %0 : tensor<?x?x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0] -> (s0 * 32)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0, d1) -> (d0 * 32 + d1 * 8)>
// CHECK-DAG:   [[MAP_7_:#.+]] = affine_map<(d0) -> (d0 floordiv 64)>
// CHECK-DAG:   [[MAP_8_:#.+]] = affine_map<(d0, d1) -> (d0 * 32 + d1 * 8 + 1)>
// CHECK-DAG:   [[MAP_9_:#.+]] = affine_map<(d0, d1) -> (d0 * 32 + d1 * 8 + 2)>
// CHECK-DAG:   [[MAP_10_:#.+]] = affine_map<(d0, d1) -> (d0 * 32 + d1 * 8 + 3)>
// CHECK-DAG:   [[MAP_11_:#.+]] = affine_map<(d0, d1) -> (d0 * 32 + d1 * 8 + 4)>
// CHECK-DAG:   [[MAP_12_:#.+]] = affine_map<(d0, d1) -> (d0 * 32 + d1 * 8 + 5)>
// CHECK-DAG:   [[MAP_13_:#.+]] = affine_map<(d0, d1) -> (d0 * 32 + d1 * 8 + 6)>
// CHECK-DAG:   [[MAP_14_:#.+]] = affine_map<(d0, d1) -> (d0 * 32 + d1 * 8 + 7)>
// CHECK-DAG:   [[MAP_15_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64)>
// CHECK-DAG:   [[MAP_16_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64 + 8)>
// CHECK-DAG:   [[MAP_17_:#.+]] = affine_map<(d0) -> (d0 + 8)>
// CHECK-DAG:   [[MAP_18_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64 + 16)>
// CHECK-DAG:   [[MAP_19_:#.+]] = affine_map<(d0) -> (d0 + 16)>
// CHECK-DAG:   [[MAP_20_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64 + 24)>
// CHECK-DAG:   [[MAP_21_:#.+]] = affine_map<(d0) -> (d0 + 24)>
// CHECK-DAG:   [[MAP_22_:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG:   [[MAP_23_:#.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-LABEL:  func.func @concat_expand_stick_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x4x?x64xf32>, [[PARAM_1_:%.+]]: memref<?x4x?x64xf32>) -> memref<?x?x64xf16, #map> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<-8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x4x?x64xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<?x4x?x64xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_1_]], [[CST_2_]] : memref<?x4x?x64xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_1_]], [[VAR_dim_2_]]{{.}}
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.apply [[MAP_2_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_1_]], [[VAR_0_]]) {{.*}}: memref<?x?x64xf16, #map>
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<?x?x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[VAR_reinterpret_cast_3_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<?x?x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[VAR_reinterpret_cast_4_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<?x?x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[VAR_reinterpret_cast_5_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<?x?x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[VAR_reinterpret_cast_6_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<?x?x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[VAR_reinterpret_cast_7_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<?x?x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[VAR_reinterpret_cast_8_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<?x?x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[VAR_reinterpret_cast_9_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<?x?x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_3_]]([[VAR_dim_1_]], [[VAR_dim_2_]], [[VAR_dim_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK-DAG:         [[VAR_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to [[MAP_4_]]([[VAR_dim_1_]]), [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 1){
// CHECK:               [[VAR_6_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[VAR_7_:%.+]] = affine.apply [[MAP_5_]]([[VAR_6_]]#1)
// CHECK-DAG:           [[VAR_8_:%.+]] = affine.apply [[MAP_6_]]([[VAR_3_]]#0, [[VAR_3_]]#1)
// CHECK:               [[VAR_9_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_8_]], [[VAR_6_]]#0, [[VAR_7_]]{{.}} : memref<?x?x64xf16, #map>
// CHECK-DAG:           [[VAR_10_:%.+]] = affine.apply [[MAP_7_]]([[VAR_9_]])
// CHECK-DAG:           [[VAR_11_:%.+]] = affine.apply [[MAP_8_]]([[VAR_3_]]#0, [[VAR_3_]]#1)
// CHECK:               [[VAR_12_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_11_]], [[VAR_6_]]#0, [[VAR_7_]]{{.}} : memref<?x?x64xf16, #map>
// CHECK-DAG:           [[VAR_13_:%.+]] = affine.apply [[MAP_7_]]([[VAR_12_]])
// CHECK-DAG:           [[VAR_14_:%.+]] = affine.apply [[MAP_9_]]([[VAR_3_]]#0, [[VAR_3_]]#1)
// CHECK:               [[VAR_15_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_14_]], [[VAR_6_]]#0, [[VAR_7_]]{{.}} : memref<?x?x64xf16, #map>
// CHECK-DAG:           [[VAR_16_:%.+]] = affine.apply [[MAP_7_]]([[VAR_15_]])
// CHECK-DAG:           [[VAR_17_:%.+]] = affine.apply [[MAP_10_]]([[VAR_3_]]#0, [[VAR_3_]]#1)
// CHECK:               [[VAR_18_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_17_]], [[VAR_6_]]#0, [[VAR_7_]]{{.}} : memref<?x?x64xf16, #map>
// CHECK-DAG:           [[VAR_19_:%.+]] = affine.apply [[MAP_7_]]([[VAR_18_]])
// CHECK-DAG:           [[VAR_20_:%.+]] = affine.apply [[MAP_11_]]([[VAR_3_]]#0, [[VAR_3_]]#1)
// CHECK:               [[VAR_21_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_20_]], [[VAR_6_]]#0, [[VAR_7_]]{{.}} : memref<?x?x64xf16, #map>
// CHECK-DAG:           [[VAR_22_:%.+]] = affine.apply [[MAP_7_]]([[VAR_21_]])
// CHECK-DAG:           [[VAR_23_:%.+]] = affine.apply [[MAP_12_]]([[VAR_3_]]#0, [[VAR_3_]]#1)
// CHECK:               [[VAR_24_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_23_]], [[VAR_6_]]#0, [[VAR_7_]]{{.}} : memref<?x?x64xf16, #map>
// CHECK-DAG:           [[VAR_25_:%.+]] = affine.apply [[MAP_7_]]([[VAR_24_]])
// CHECK-DAG:           [[VAR_26_:%.+]] = affine.apply [[MAP_13_]]([[VAR_3_]]#0, [[VAR_3_]]#1)
// CHECK:               [[VAR_27_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_26_]], [[VAR_6_]]#0, [[VAR_7_]]{{.}} : memref<?x?x64xf16, #map>
// CHECK-DAG:           [[VAR_28_:%.+]] = affine.apply [[MAP_7_]]([[VAR_27_]])
// CHECK-DAG:           [[VAR_29_:%.+]] = affine.apply [[MAP_14_]]([[VAR_3_]]#0, [[VAR_3_]]#1)
// CHECK:               [[VAR_30_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_29_]], [[VAR_6_]]#0, [[VAR_7_]]{{.}} : memref<?x?x64xf16, #map>
// CHECK-DAG:           [[VAR_31_:%.+]] = affine.apply [[MAP_7_]]([[VAR_30_]])
// CHECK-DAG:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:               [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_2_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:               krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_2_]] -> [[I_4_:%.+]] = 0 to 64){
// CHECK:                 [[VAR_33_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:                 [[VAR_34_:%.+]] = affine.apply [[MAP_15_]]([[VAR_33_]], [[VAR_6_]]#1)
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_6_]]#0, [[VAR_34_]]{{.}} : memref<?x4x?x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_36_:%.+]] = arith.addi [[VAR_34_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_1_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_6_]]#0, [[VAR_36_]]{{.}} : memref<?x4x?x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_38_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_39_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_40_:%.+]] = arith.maxnumf [[VAR_38_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_41_:%.+]] = arith.maxnumf [[VAR_39_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_42_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_40_]], [[VAR_41_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                 vector.store [[VAR_42_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_10_]], [[VAR_33_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store [[VAR_42_]], [[VAR_reinterpret_cast_3_]]{{.}}[[VAR_13_]], [[VAR_33_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store [[VAR_42_]], [[VAR_reinterpret_cast_4_]]{{.}}[[VAR_16_]], [[VAR_33_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store [[VAR_42_]], [[VAR_reinterpret_cast_5_]]{{.}}[[VAR_19_]], [[VAR_33_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store [[VAR_42_]], [[VAR_reinterpret_cast_6_]]{{.}}[[VAR_22_]], [[VAR_33_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store [[VAR_42_]], [[VAR_reinterpret_cast_7_]]{{.}}[[VAR_25_]], [[VAR_33_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store [[VAR_42_]], [[VAR_reinterpret_cast_8_]]{{.}}[[VAR_28_]], [[VAR_33_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store [[VAR_42_]], [[VAR_reinterpret_cast_9_]]{{.}}[[VAR_31_]], [[VAR_33_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_43_:%.+]] = affine.apply [[MAP_16_]]([[VAR_33_]], [[VAR_6_]]#1)
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_2_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_6_]]#0, [[VAR_43_]]{{.}} : memref<?x4x?x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_45_:%.+]] = arith.addi [[VAR_43_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_3_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_6_]]#0, [[VAR_45_]]{{.}} : memref<?x4x?x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_47_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_2_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_48_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_3_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_49_:%.+]] = arith.maxnumf [[VAR_47_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_50_:%.+]] = arith.maxnumf [[VAR_48_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_51_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_49_]], [[VAR_50_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:             [[VAR_52_:%.+]] = affine.apply [[MAP_17_]]([[VAR_33_]])
// CHECK:                 vector.store [[VAR_51_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_10_]], [[VAR_52_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_53_:%.+]] = affine.apply [[MAP_17_]]([[VAR_33_]])
// CHECK:                 vector.store [[VAR_51_]], [[VAR_reinterpret_cast_3_]]{{.}}[[VAR_13_]], [[VAR_53_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_54_:%.+]] = affine.apply [[MAP_17_]]([[VAR_33_]])
// CHECK:                 vector.store [[VAR_51_]], [[VAR_reinterpret_cast_4_]]{{.}}[[VAR_16_]], [[VAR_54_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_55_:%.+]] = affine.apply [[MAP_17_]]([[VAR_33_]])
// CHECK:                 vector.store [[VAR_51_]], [[VAR_reinterpret_cast_5_]]{{.}}[[VAR_19_]], [[VAR_55_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_56_:%.+]] = affine.apply [[MAP_17_]]([[VAR_33_]])
// CHECK:                 vector.store [[VAR_51_]], [[VAR_reinterpret_cast_6_]]{{.}}[[VAR_22_]], [[VAR_56_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_57_:%.+]] = affine.apply [[MAP_17_]]([[VAR_33_]])
// CHECK:                 vector.store [[VAR_51_]], [[VAR_reinterpret_cast_7_]]{{.}}[[VAR_25_]], [[VAR_57_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_58_:%.+]] = affine.apply [[MAP_17_]]([[VAR_33_]])
// CHECK:                 vector.store [[VAR_51_]], [[VAR_reinterpret_cast_8_]]{{.}}[[VAR_28_]], [[VAR_58_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_59_:%.+]] = affine.apply [[MAP_17_]]([[VAR_33_]])
// CHECK:                 vector.store [[VAR_51_]], [[VAR_reinterpret_cast_9_]]{{.}}[[VAR_31_]], [[VAR_59_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_60_:%.+]] = affine.apply [[MAP_18_]]([[VAR_33_]], [[VAR_6_]]#1)
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_4_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_6_]]#0, [[VAR_60_]]{{.}} : memref<?x4x?x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_62_:%.+]] = arith.addi [[VAR_60_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_5_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_6_]]#0, [[VAR_62_]]{{.}} : memref<?x4x?x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_64_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_4_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_65_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_5_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_66_:%.+]] = arith.maxnumf [[VAR_64_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_67_:%.+]] = arith.maxnumf [[VAR_65_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_68_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_66_]], [[VAR_67_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:             [[VAR_69_:%.+]] = affine.apply [[MAP_19_]]([[VAR_33_]])
// CHECK:                 vector.store [[VAR_68_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_10_]], [[VAR_69_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_70_:%.+]] = affine.apply [[MAP_19_]]([[VAR_33_]])
// CHECK:                 vector.store [[VAR_68_]], [[VAR_reinterpret_cast_3_]]{{.}}[[VAR_13_]], [[VAR_70_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_71_:%.+]] = affine.apply [[MAP_19_]]([[VAR_33_]])
// CHECK:                 vector.store [[VAR_68_]], [[VAR_reinterpret_cast_4_]]{{.}}[[VAR_16_]], [[VAR_71_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_72_:%.+]] = affine.apply [[MAP_19_]]([[VAR_33_]])
// CHECK:                 vector.store [[VAR_68_]], [[VAR_reinterpret_cast_5_]]{{.}}[[VAR_19_]], [[VAR_72_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_73_:%.+]] = affine.apply [[MAP_19_]]([[VAR_33_]])
// CHECK:                 vector.store [[VAR_68_]], [[VAR_reinterpret_cast_6_]]{{.}}[[VAR_22_]], [[VAR_73_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_74_:%.+]] = affine.apply [[MAP_19_]]([[VAR_33_]])
// CHECK:                 vector.store [[VAR_68_]], [[VAR_reinterpret_cast_7_]]{{.}}[[VAR_25_]], [[VAR_74_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_75_:%.+]] = affine.apply [[MAP_19_]]([[VAR_33_]])
// CHECK:                 vector.store [[VAR_68_]], [[VAR_reinterpret_cast_8_]]{{.}}[[VAR_28_]], [[VAR_75_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_76_:%.+]] = affine.apply [[MAP_19_]]([[VAR_33_]])
// CHECK:                 vector.store [[VAR_68_]], [[VAR_reinterpret_cast_9_]]{{.}}[[VAR_31_]], [[VAR_76_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_77_:%.+]] = affine.apply [[MAP_20_]]([[VAR_33_]], [[VAR_6_]]#1)
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_6_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_6_]]#0, [[VAR_77_]]{{.}} : memref<?x4x?x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_79_:%.+]] = arith.addi [[VAR_77_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_7_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_6_]]#0, [[VAR_79_]]{{.}} : memref<?x4x?x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_81_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_6_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_82_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_7_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_83_:%.+]] = arith.maxnumf [[VAR_81_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_84_:%.+]] = arith.maxnumf [[VAR_82_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_85_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_83_]], [[VAR_84_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:             [[VAR_86_:%.+]] = affine.apply [[MAP_21_]]([[VAR_33_]])
// CHECK:                 vector.store [[VAR_85_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_10_]], [[VAR_86_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_87_:%.+]] = affine.apply [[MAP_21_]]([[VAR_33_]])
// CHECK:                 vector.store [[VAR_85_]], [[VAR_reinterpret_cast_3_]]{{.}}[[VAR_13_]], [[VAR_87_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_88_:%.+]] = affine.apply [[MAP_21_]]([[VAR_33_]])
// CHECK:                 vector.store [[VAR_85_]], [[VAR_reinterpret_cast_4_]]{{.}}[[VAR_16_]], [[VAR_88_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_89_:%.+]] = affine.apply [[MAP_21_]]([[VAR_33_]])
// CHECK:                 vector.store [[VAR_85_]], [[VAR_reinterpret_cast_5_]]{{.}}[[VAR_19_]], [[VAR_89_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_90_:%.+]] = affine.apply [[MAP_21_]]([[VAR_33_]])
// CHECK:                 vector.store [[VAR_85_]], [[VAR_reinterpret_cast_6_]]{{.}}[[VAR_22_]], [[VAR_90_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_91_:%.+]] = affine.apply [[MAP_21_]]([[VAR_33_]])
// CHECK:                 vector.store [[VAR_85_]], [[VAR_reinterpret_cast_7_]]{{.}}[[VAR_25_]], [[VAR_91_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_92_:%.+]] = affine.apply [[MAP_21_]]([[VAR_33_]])
// CHECK:                 vector.store [[VAR_85_]], [[VAR_reinterpret_cast_8_]]{{.}}[[VAR_28_]], [[VAR_92_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_93_:%.+]] = affine.apply [[MAP_21_]]([[VAR_33_]])
// CHECK:                 vector.store [[VAR_85_]], [[VAR_reinterpret_cast_9_]]{{.}}[[VAR_31_]], [[VAR_93_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               }
// CHECK:             }
// CHECK:             [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to [[MAP_22_]]([[VAR_dim_1_]], [[VAR_dim_2_]]), [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to 1){
// CHECK:               [[VAR_6_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[VAR_7_1_:%.+]] = affine.apply [[MAP_5_]]([[VAR_6_1_]]#1)
// CHECK-DAG:           [[VAR_8_1_:%.+]] = affine.apply [[MAP_23_]]([[VAR_6_1_]]#0){{.}}[[VAR_dim_1_]]{{.}}
// CHECK-DAG:           [[VAR_9_1_:%.+]] = affine.apply [[MAP_6_]]([[VAR_3_]]#0, [[VAR_3_]]#1)
// CHECK:               [[VAR_10_1_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_9_1_]], [[VAR_8_1_]], [[VAR_7_1_]]{{.}} : memref<?x?x64xf16, #map>
// CHECK-DAG:           [[VAR_11_1_:%.+]] = affine.apply [[MAP_7_]]([[VAR_10_1_]])
// CHECK-DAG:           [[VAR_12_1_:%.+]] = affine.apply [[MAP_8_]]([[VAR_3_]]#0, [[VAR_3_]]#1)
// CHECK:               [[VAR_13_1_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_12_1_]], [[VAR_8_1_]], [[VAR_7_1_]]{{.}} : memref<?x?x64xf16, #map>
// CHECK-DAG:           [[VAR_14_1_:%.+]] = affine.apply [[MAP_7_]]([[VAR_13_1_]])
// CHECK-DAG:           [[VAR_15_1_:%.+]] = affine.apply [[MAP_9_]]([[VAR_3_]]#0, [[VAR_3_]]#1)
// CHECK:               [[VAR_16_1_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_15_1_]], [[VAR_8_1_]], [[VAR_7_1_]]{{.}} : memref<?x?x64xf16, #map>
// CHECK-DAG:           [[VAR_17_1_:%.+]] = affine.apply [[MAP_7_]]([[VAR_16_1_]])
// CHECK-DAG:           [[VAR_18_1_:%.+]] = affine.apply [[MAP_10_]]([[VAR_3_]]#0, [[VAR_3_]]#1)
// CHECK:               [[VAR_19_1_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_18_1_]], [[VAR_8_1_]], [[VAR_7_1_]]{{.}} : memref<?x?x64xf16, #map>
// CHECK-DAG:           [[VAR_20_1_:%.+]] = affine.apply [[MAP_7_]]([[VAR_19_1_]])
// CHECK-DAG:           [[VAR_21_1_:%.+]] = affine.apply [[MAP_11_]]([[VAR_3_]]#0, [[VAR_3_]]#1)
// CHECK:               [[VAR_22_1_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_21_1_]], [[VAR_8_1_]], [[VAR_7_1_]]{{.}} : memref<?x?x64xf16, #map>
// CHECK-DAG:           [[VAR_23_1_:%.+]] = affine.apply [[MAP_7_]]([[VAR_22_1_]])
// CHECK-DAG:           [[VAR_24_1_:%.+]] = affine.apply [[MAP_12_]]([[VAR_3_]]#0, [[VAR_3_]]#1)
// CHECK:               [[VAR_25_1_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_24_1_]], [[VAR_8_1_]], [[VAR_7_1_]]{{.}} : memref<?x?x64xf16, #map>
// CHECK-DAG:           [[VAR_26_1_:%.+]] = affine.apply [[MAP_7_]]([[VAR_25_1_]])
// CHECK-DAG:           [[VAR_27_1_:%.+]] = affine.apply [[MAP_13_]]([[VAR_3_]]#0, [[VAR_3_]]#1)
// CHECK:               [[VAR_28_1_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_27_1_]], [[VAR_8_1_]], [[VAR_7_1_]]{{.}} : memref<?x?x64xf16, #map>
// CHECK-DAG:           [[VAR_29_1_:%.+]] = affine.apply [[MAP_7_]]([[VAR_28_1_]])
// CHECK-DAG:           [[VAR_30_1_:%.+]] = affine.apply [[MAP_14_]]([[VAR_3_]]#0, [[VAR_3_]]#1)
// CHECK:               [[VAR_31_1_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_30_1_]], [[VAR_8_1_]], [[VAR_7_1_]]{{.}} : memref<?x?x64xf16, #map>
// CHECK-DAG:           [[LOOP_2_:%.+]] = affine.apply [[MAP_7_]]([[VAR_31_1_]])
// CHECK-DAG:           [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:               [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_4_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:               krnl.iterate([[BLOCK_TILE__1_]]) with ([[LOOP_4_]] -> [[I_7_:%.+]] = 0 to 64){
// CHECK:                 [[VAR_34_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__1_]]) : (!krnl.loop) -> index
// CHECK:                 [[LOAD_PARAM_0_MEM_8_:%.+]] = affine.apply [[MAP_15_]]([[VAR_34_1_]], [[VAR_6_1_]]#1)
// CHECK-DAG:             [[VAR_36_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_6_1_]]#0, [[LOAD_PARAM_0_MEM_8_]]{{.}} : memref<?x4x?x64xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_1_:%.+]] = arith.addi [[LOAD_PARAM_0_MEM_8_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_38_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_6_1_]]#0, [[LOAD_PARAM_0_MEM_1_]]{{.}} : memref<?x4x?x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_39_1_:%.+]] = arith.minnumf [[VAR_36_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_40_1_:%.+]] = arith.minnumf [[VAR_38_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_41_1_:%.+]] = arith.maxnumf [[VAR_39_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_42_1_:%.+]] = arith.maxnumf [[VAR_40_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_43_1_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_41_1_]], [[VAR_42_1_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                 vector.store [[VAR_43_1_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_1_]], [[VAR_34_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store [[VAR_43_1_]], [[VAR_reinterpret_cast_3_]]{{.}}[[VAR_14_1_]], [[VAR_34_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store [[VAR_43_1_]], [[VAR_reinterpret_cast_4_]]{{.}}[[VAR_17_1_]], [[VAR_34_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store [[VAR_43_1_]], [[VAR_reinterpret_cast_5_]]{{.}}[[VAR_20_1_]], [[VAR_34_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store [[VAR_43_1_]], [[VAR_reinterpret_cast_6_]]{{.}}[[VAR_23_1_]], [[VAR_34_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store [[VAR_43_1_]], [[VAR_reinterpret_cast_7_]]{{.}}[[VAR_26_1_]], [[VAR_34_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store [[VAR_43_1_]], [[VAR_reinterpret_cast_8_]]{{.}}[[VAR_29_1_]], [[VAR_34_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store [[VAR_43_1_]], [[VAR_reinterpret_cast_9_]]{{.}}[[LOOP_2_]], [[VAR_34_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[LOAD_PARAM_0_MEM_2_:%.+]] = affine.apply [[MAP_16_]]([[VAR_34_1_]], [[VAR_6_1_]]#1)
// CHECK-DAG:             [[VAR_45_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_6_1_]]#0, [[LOAD_PARAM_0_MEM_2_]]{{.}} : memref<?x4x?x64xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_3_:%.+]] = arith.addi [[LOAD_PARAM_0_MEM_2_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_47_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_6_1_]]#0, [[LOAD_PARAM_0_MEM_3_]]{{.}} : memref<?x4x?x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_48_1_:%.+]] = arith.minnumf [[VAR_45_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_49_1_:%.+]] = arith.minnumf [[VAR_47_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_50_1_:%.+]] = arith.maxnumf [[VAR_48_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_51_1_:%.+]] = arith.maxnumf [[VAR_49_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_52_1_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_50_1_]], [[VAR_51_1_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:             [[VAR_53_1_:%.+]] = affine.apply [[MAP_17_]]([[VAR_34_1_]])
// CHECK:                 vector.store [[VAR_52_1_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_1_]], [[VAR_53_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_54_1_:%.+]] = affine.apply [[MAP_17_]]([[VAR_34_1_]])
// CHECK:                 vector.store [[VAR_52_1_]], [[VAR_reinterpret_cast_3_]]{{.}}[[VAR_14_1_]], [[VAR_54_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_55_1_:%.+]] = affine.apply [[MAP_17_]]([[VAR_34_1_]])
// CHECK:                 vector.store [[VAR_52_1_]], [[VAR_reinterpret_cast_4_]]{{.}}[[VAR_17_1_]], [[VAR_55_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_56_1_:%.+]] = affine.apply [[MAP_17_]]([[VAR_34_1_]])
// CHECK:                 vector.store [[VAR_52_1_]], [[VAR_reinterpret_cast_5_]]{{.}}[[VAR_20_1_]], [[VAR_56_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_57_1_:%.+]] = affine.apply [[MAP_17_]]([[VAR_34_1_]])
// CHECK:                 vector.store [[VAR_52_1_]], [[VAR_reinterpret_cast_6_]]{{.}}[[VAR_23_1_]], [[VAR_57_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_58_1_:%.+]] = affine.apply [[MAP_17_]]([[VAR_34_1_]])
// CHECK:                 vector.store [[VAR_52_1_]], [[VAR_reinterpret_cast_7_]]{{.}}[[VAR_26_1_]], [[VAR_58_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_59_1_:%.+]] = affine.apply [[MAP_17_]]([[VAR_34_1_]])
// CHECK:                 vector.store [[VAR_52_1_]], [[VAR_reinterpret_cast_8_]]{{.}}[[VAR_29_1_]], [[VAR_59_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_60_1_:%.+]] = affine.apply [[MAP_17_]]([[VAR_34_1_]])
// CHECK:                 vector.store [[VAR_52_1_]], [[VAR_reinterpret_cast_9_]]{{.}}[[LOOP_2_]], [[VAR_60_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[LOAD_PARAM_0_MEM_4_:%.+]] = affine.apply [[MAP_18_]]([[VAR_34_1_]], [[VAR_6_1_]]#1)
// CHECK-DAG:             [[VAR_62_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_6_1_]]#0, [[LOAD_PARAM_0_MEM_4_]]{{.}} : memref<?x4x?x64xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_5_:%.+]] = arith.addi [[LOAD_PARAM_0_MEM_4_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_64_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_6_1_]]#0, [[LOAD_PARAM_0_MEM_5_]]{{.}} : memref<?x4x?x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_65_1_:%.+]] = arith.minnumf [[VAR_62_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_66_1_:%.+]] = arith.minnumf [[VAR_64_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_67_1_:%.+]] = arith.maxnumf [[VAR_65_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_68_1_:%.+]] = arith.maxnumf [[VAR_66_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_69_1_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_67_1_]], [[VAR_68_1_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:             [[VAR_70_1_:%.+]] = affine.apply [[MAP_19_]]([[VAR_34_1_]])
// CHECK:                 vector.store [[VAR_69_1_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_1_]], [[VAR_70_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_71_1_:%.+]] = affine.apply [[MAP_19_]]([[VAR_34_1_]])
// CHECK:                 vector.store [[VAR_69_1_]], [[VAR_reinterpret_cast_3_]]{{.}}[[VAR_14_1_]], [[VAR_71_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_72_1_:%.+]] = affine.apply [[MAP_19_]]([[VAR_34_1_]])
// CHECK:                 vector.store [[VAR_69_1_]], [[VAR_reinterpret_cast_4_]]{{.}}[[VAR_17_1_]], [[VAR_72_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_73_1_:%.+]] = affine.apply [[MAP_19_]]([[VAR_34_1_]])
// CHECK:                 vector.store [[VAR_69_1_]], [[VAR_reinterpret_cast_5_]]{{.}}[[VAR_20_1_]], [[VAR_73_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_74_1_:%.+]] = affine.apply [[MAP_19_]]([[VAR_34_1_]])
// CHECK:                 vector.store [[VAR_69_1_]], [[VAR_reinterpret_cast_6_]]{{.}}[[VAR_23_1_]], [[VAR_74_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_75_1_:%.+]] = affine.apply [[MAP_19_]]([[VAR_34_1_]])
// CHECK:                 vector.store [[VAR_69_1_]], [[VAR_reinterpret_cast_7_]]{{.}}[[VAR_26_1_]], [[VAR_75_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_76_1_:%.+]] = affine.apply [[MAP_19_]]([[VAR_34_1_]])
// CHECK:                 vector.store [[VAR_69_1_]], [[VAR_reinterpret_cast_8_]]{{.}}[[VAR_29_1_]], [[VAR_76_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_77_1_:%.+]] = affine.apply [[MAP_19_]]([[VAR_34_1_]])
// CHECK:                 vector.store [[VAR_69_1_]], [[VAR_reinterpret_cast_9_]]{{.}}[[LOOP_2_]], [[VAR_77_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[LOAD_PARAM_0_MEM_6_:%.+]] = affine.apply [[MAP_20_]]([[VAR_34_1_]], [[VAR_6_1_]]#1)
// CHECK-DAG:             [[VAR_79_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_6_1_]]#0, [[LOAD_PARAM_0_MEM_6_]]{{.}} : memref<?x4x?x64xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_7_:%.+]] = arith.addi [[LOAD_PARAM_0_MEM_6_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_81_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_6_1_]]#0, [[LOAD_PARAM_0_MEM_7_]]{{.}} : memref<?x4x?x64xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_82_1_:%.+]] = arith.minnumf [[VAR_79_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_83_1_:%.+]] = arith.minnumf [[VAR_81_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_84_1_:%.+]] = arith.maxnumf [[VAR_82_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_85_1_:%.+]] = arith.maxnumf [[VAR_83_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_86_1_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_84_1_]], [[VAR_85_1_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:             [[VAR_87_1_:%.+]] = affine.apply [[MAP_21_]]([[VAR_34_1_]])
// CHECK:                 vector.store [[VAR_86_1_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_11_1_]], [[VAR_87_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_88_1_:%.+]] = affine.apply [[MAP_21_]]([[VAR_34_1_]])
// CHECK:                 vector.store [[VAR_86_1_]], [[VAR_reinterpret_cast_3_]]{{.}}[[VAR_14_1_]], [[VAR_88_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_89_1_:%.+]] = affine.apply [[MAP_21_]]([[VAR_34_1_]])
// CHECK:                 vector.store [[VAR_86_1_]], [[VAR_reinterpret_cast_4_]]{{.}}[[VAR_17_1_]], [[VAR_89_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_90_1_:%.+]] = affine.apply [[MAP_21_]]([[VAR_34_1_]])
// CHECK:                 vector.store [[VAR_86_1_]], [[VAR_reinterpret_cast_5_]]{{.}}[[VAR_20_1_]], [[VAR_90_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_91_1_:%.+]] = affine.apply [[MAP_21_]]([[VAR_34_1_]])
// CHECK:                 vector.store [[VAR_86_1_]], [[VAR_reinterpret_cast_6_]]{{.}}[[VAR_23_1_]], [[VAR_91_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_92_1_:%.+]] = affine.apply [[MAP_21_]]([[VAR_34_1_]])
// CHECK:                 vector.store [[VAR_86_1_]], [[VAR_reinterpret_cast_7_]]{{.}}[[VAR_26_1_]], [[VAR_92_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_93_1_:%.+]] = affine.apply [[MAP_21_]]([[VAR_34_1_]])
// CHECK:                 vector.store [[VAR_86_1_]], [[VAR_reinterpret_cast_8_]]{{.}}[[VAR_29_1_]], [[VAR_93_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_94_:%.+]] = affine.apply [[MAP_21_]]([[VAR_34_1_]])
// CHECK:                 vector.store [[VAR_86_1_]], [[VAR_reinterpret_cast_9_]]{{.}}[[LOOP_2_]], [[VAR_94_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x64xf16, #map>
// CHECK:         }
}
