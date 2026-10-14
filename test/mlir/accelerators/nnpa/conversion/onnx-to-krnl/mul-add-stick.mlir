// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// -----

// Model pattern (granite-embedding-97m-multilingual-r2 rotary embedding):
// (x1 * cos + x2 * sin) * k, dynamic (B, S), 12 heads, D = 32 (half stick),
// Reshape (B, 12, S, 32) -> (B * 12, S, 32), 3DS. The cos/sin tables
// broadcast over B and the heads. The Reshape shape input is unused.

func.func @test_mul_add_stick_rotary(%arg0: tensor<?x12x?x32xf32>, %arg1: tensor<1x1x?x32xf32>, %arg2: tensor<?x12x?x32xf32>, %arg3: tensor<1x1x?x32xf32>, %arg4: tensor<3xi64>) -> tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %0 = "onnx.Fused"(%arg0, %arg1, %arg2, %arg3, %arg4) <{kind = "zhigh.mul-add-stick"}> ({
  ^bb0(%a0: tensor<?x12x?x32xf32>, %a1: tensor<1x1x?x32xf32>, %b0: tensor<?x12x?x32xf32>, %b1: tensor<1x1x?x32xf32>, %shape: tensor<3xi64>):
    %k = onnx.Constant dense<0.420448214> : tensor<1xf32>
    %1 = "onnx.Mul"(%a0, %a1) : (tensor<?x12x?x32xf32>, tensor<1x1x?x32xf32>) -> tensor<?x12x?x32xf32>
    %2 = "onnx.Mul"(%b0, %b1) : (tensor<?x12x?x32xf32>, tensor<1x1x?x32xf32>) -> tensor<?x12x?x32xf32>
    %3 = "onnx.Add"(%1, %2) : (tensor<?x12x?x32xf32>, tensor<?x12x?x32xf32>) -> tensor<?x12x?x32xf32>
    %4 = "onnx.Mul"(%3, %k) : (tensor<?x12x?x32xf32>, tensor<1xf32>) -> tensor<?x12x?x32xf32>
    %5 = "onnx.Reshape"(%4, %shape) <{allowzero = 0 : si64}> : (tensor<?x12x?x32xf32>, tensor<3xi64>) -> tensor<?x?x32xf32>
    %6 = "zhigh.Stick"(%5) <{layout = "3DS"}> : (tensor<?x?x32xf32>) -> tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    onnx.Yield %6 : tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  }) {isSub = false, mulScalar = 0.420448214 : f32, reshapeCollapsedCount = 2 : i64, reshapeFirstCollapsedDim = 0 : i64, stickFormat = "3DS"} : (tensor<?x12x?x32xf32>, tensor<1x1x?x32xf32>, tensor<?x12x?x32xf32>, tensor<1x1x?x32xf32>, tensor<3xi64>) -> tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %0 : tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 * 12)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 * 32)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0, d1) -> (d0 * 12 + d1)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0) -> (d0 floordiv 64)>
// CHECK-DAG:   [[MAP_7_:#.+]] = affine_map<(d0) -> (d0 * 32 + 8)>
// CHECK-DAG:   [[MAP_8_:#.+]] = affine_map<(d0) -> (d0 * 32 + 16)>
// CHECK-DAG:   [[MAP_9_:#.+]] = affine_map<(d0) -> (d0 * 32 + 24)>
// CHECK-LABEL:  func.func @test_mul_add_stick_rotary
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x12x?x32xf32>, [[PARAM_1_:%.+]]: memref<1x1x?x32xf32>, [[PARAM_2_:%.+]]: memref<?x12x?x32xf32>, [[PARAM_3_:%.+]]: memref<1x1x?x32xf32>, [[PARAM_4_:%.+]]: memref<3xi64>) -> memref<?x?x32xf16, #map> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<-8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[VAR_cst_1_:%.+]] = arith.constant dense<0.420448214> : vector<4xf32>
// CHECK-DAG:       [[CST_24_:%.+]] = arith.constant 24 : index
// CHECK-DAG:       [[CST_16_:%.+]] = arith.constant 16 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x12x?x32xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<?x12x?x32xf32>
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_0_]], [[VAR_dim_2_]]) alignment = 4096 : memref<?x?x32xf16, #map>
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<?x?x32xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_2_]]([[VAR_dim_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 12, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 1, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to [[MAP_3_]]([[VAR_dim_]], [[VAR_dim_2_]])){
// CHECK:             [[VAR_2_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_4_]]([[VAR_2_]]#2)
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.apply [[MAP_5_]]([[VAR_2_]]#0, [[VAR_2_]]#1)
// CHECK:             [[VAR_5_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_4_]], [[VAR_2_]]#3, [[VAR_3_]]{{.}} : memref<?x?x32xf16, #map>
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply [[MAP_6_]]([[VAR_5_]])
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#3, [[VAR_3_]]{{.}} : memref<?x12x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.addi [[VAR_3_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#3, [[VAR_8_]]{{.}} : memref<?x12x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_0_]], [[CST_0_]], [[VAR_2_]]#3, [[VAR_3_]]{{.}} : memref<1x1x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.addi [[VAR_3_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_0_]], [[CST_0_]], [[VAR_2_]]#3, [[VAR_11_]]{{.}} : memref<1x1x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#3, [[VAR_3_]]{{.}} : memref<?x12x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.addi [[VAR_3_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_1_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#3, [[VAR_14_]]{{.}} : memref<?x12x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[CST_0_]], [[CST_0_]], [[VAR_2_]]#3, [[VAR_3_]]{{.}} : memref<1x1x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_17_:%.+]] = arith.addi [[VAR_3_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_1_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[CST_0_]], [[CST_0_]], [[VAR_2_]]#3, [[VAR_17_]]{{.}} : memref<1x1x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_19_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_20_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_]], [[LOAD_PARAM_3_MEM_]] : vector<4xf32>
// CHECK:             [[VAR_21_:%.+]] = arith.addf [[VAR_19_]], [[VAR_20_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_22_:%.+]] = arith.mulf [[VAR_21_]], [[VAR_cst_1_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_23_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_1_]], [[LOAD_PARAM_1_MEM_1_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_24_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_1_]], [[LOAD_PARAM_3_MEM_1_]] : vector<4xf32>
// CHECK:             [[VAR_25_:%.+]] = arith.addf [[VAR_23_]], [[VAR_24_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_26_:%.+]] = arith.mulf [[VAR_25_]], [[VAR_cst_1_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_27_:%.+]] = arith.minnumf [[VAR_22_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_28_:%.+]] = arith.minnumf [[VAR_26_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_29_:%.+]] = arith.maxnumf [[VAR_27_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_30_:%.+]] = arith.maxnumf [[VAR_28_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_31_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_29_]], [[VAR_30_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_31_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_6_]], [[CST_0_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_32_:%.+]] = affine.apply [[MAP_7_]]([[VAR_2_]]#2)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_2_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#3, [[VAR_32_]]{{.}} : memref<?x12x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_34_:%.+]] = arith.addi [[VAR_32_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_3_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#3, [[VAR_34_]]{{.}} : memref<?x12x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_36_:%.+]] = affine.apply [[MAP_7_]]([[VAR_2_]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_2_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_0_]], [[CST_0_]], [[VAR_2_]]#3, [[VAR_36_]]{{.}} : memref<1x1x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_38_:%.+]] = arith.addi [[VAR_36_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_3_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_0_]], [[CST_0_]], [[VAR_2_]]#3, [[VAR_38_]]{{.}} : memref<1x1x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_40_:%.+]] = affine.apply [[MAP_7_]]([[VAR_2_]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_2_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#3, [[VAR_40_]]{{.}} : memref<?x12x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_42_:%.+]] = arith.addi [[VAR_40_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_3_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#3, [[VAR_42_]]{{.}} : memref<?x12x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_44_:%.+]] = affine.apply [[MAP_7_]]([[VAR_2_]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_2_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[CST_0_]], [[CST_0_]], [[VAR_2_]]#3, [[VAR_44_]]{{.}} : memref<1x1x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_46_:%.+]] = arith.addi [[VAR_44_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_3_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[CST_0_]], [[CST_0_]], [[VAR_2_]]#3, [[VAR_46_]]{{.}} : memref<1x1x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_48_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_2_]], [[LOAD_PARAM_1_MEM_2_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_49_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_2_]], [[LOAD_PARAM_3_MEM_2_]] : vector<4xf32>
// CHECK:             [[VAR_50_:%.+]] = arith.addf [[VAR_48_]], [[VAR_49_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_51_:%.+]] = arith.mulf [[VAR_50_]], [[VAR_cst_1_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_52_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_3_]], [[LOAD_PARAM_1_MEM_3_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_53_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_3_]], [[LOAD_PARAM_3_MEM_3_]] : vector<4xf32>
// CHECK:             [[VAR_54_:%.+]] = arith.addf [[VAR_52_]], [[VAR_53_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_55_:%.+]] = arith.mulf [[VAR_54_]], [[VAR_cst_1_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_56_:%.+]] = arith.minnumf [[VAR_51_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_57_:%.+]] = arith.minnumf [[VAR_55_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_58_:%.+]] = arith.maxnumf [[VAR_56_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_59_:%.+]] = arith.maxnumf [[VAR_57_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_60_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_58_]], [[VAR_59_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_60_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_6_]], [[CST_8_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_61_:%.+]] = affine.apply [[MAP_8_]]([[VAR_2_]]#2)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_4_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#3, [[VAR_61_]]{{.}} : memref<?x12x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_63_:%.+]] = arith.addi [[VAR_61_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_5_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#3, [[VAR_63_]]{{.}} : memref<?x12x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_65_:%.+]] = affine.apply [[MAP_8_]]([[VAR_2_]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_4_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_0_]], [[CST_0_]], [[VAR_2_]]#3, [[VAR_65_]]{{.}} : memref<1x1x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_67_:%.+]] = arith.addi [[VAR_65_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_5_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_0_]], [[CST_0_]], [[VAR_2_]]#3, [[VAR_67_]]{{.}} : memref<1x1x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_69_:%.+]] = affine.apply [[MAP_8_]]([[VAR_2_]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_4_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#3, [[VAR_69_]]{{.}} : memref<?x12x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_71_:%.+]] = arith.addi [[VAR_69_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_5_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#3, [[VAR_71_]]{{.}} : memref<?x12x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_73_:%.+]] = affine.apply [[MAP_8_]]([[VAR_2_]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_4_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[CST_0_]], [[CST_0_]], [[VAR_2_]]#3, [[VAR_73_]]{{.}} : memref<1x1x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_75_:%.+]] = arith.addi [[VAR_73_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_5_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[CST_0_]], [[CST_0_]], [[VAR_2_]]#3, [[VAR_75_]]{{.}} : memref<1x1x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_77_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_4_]], [[LOAD_PARAM_1_MEM_4_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_78_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_4_]], [[LOAD_PARAM_3_MEM_4_]] : vector<4xf32>
// CHECK:             [[VAR_79_:%.+]] = arith.addf [[VAR_77_]], [[VAR_78_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_80_:%.+]] = arith.mulf [[VAR_79_]], [[VAR_cst_1_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_81_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_5_]], [[LOAD_PARAM_1_MEM_5_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_82_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_5_]], [[LOAD_PARAM_3_MEM_5_]] : vector<4xf32>
// CHECK:             [[VAR_83_:%.+]] = arith.addf [[VAR_81_]], [[VAR_82_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_84_:%.+]] = arith.mulf [[VAR_83_]], [[VAR_cst_1_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_85_:%.+]] = arith.minnumf [[VAR_80_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_86_:%.+]] = arith.minnumf [[VAR_84_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_87_:%.+]] = arith.maxnumf [[VAR_85_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_88_:%.+]] = arith.maxnumf [[VAR_86_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_89_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_87_]], [[VAR_88_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_89_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_6_]], [[CST_16_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_90_:%.+]] = affine.apply [[MAP_9_]]([[VAR_2_]]#2)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_6_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#3, [[VAR_90_]]{{.}} : memref<?x12x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_92_:%.+]] = arith.addi [[VAR_90_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_7_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#3, [[VAR_92_]]{{.}} : memref<?x12x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_94_:%.+]] = affine.apply [[MAP_9_]]([[VAR_2_]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_6_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_0_]], [[CST_0_]], [[VAR_2_]]#3, [[VAR_94_]]{{.}} : memref<1x1x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_96_:%.+]] = arith.addi [[VAR_94_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_7_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_0_]], [[CST_0_]], [[VAR_2_]]#3, [[VAR_96_]]{{.}} : memref<1x1x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_98_:%.+]] = affine.apply [[MAP_9_]]([[VAR_2_]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_6_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#3, [[VAR_98_]]{{.}} : memref<?x12x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_100_:%.+]] = arith.addi [[VAR_98_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_7_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#3, [[VAR_100_]]{{.}} : memref<?x12x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_102_:%.+]] = affine.apply [[MAP_9_]]([[VAR_2_]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_6_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[CST_0_]], [[CST_0_]], [[VAR_2_]]#3, [[VAR_102_]]{{.}} : memref<1x1x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_104_:%.+]] = arith.addi [[VAR_102_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_7_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[CST_0_]], [[CST_0_]], [[VAR_2_]]#3, [[VAR_104_]]{{.}} : memref<1x1x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_106_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_6_]], [[LOAD_PARAM_1_MEM_6_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_107_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_6_]], [[LOAD_PARAM_3_MEM_6_]] : vector<4xf32>
// CHECK:             [[VAR_108_:%.+]] = arith.addf [[VAR_106_]], [[VAR_107_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_109_:%.+]] = arith.mulf [[VAR_108_]], [[VAR_cst_1_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_110_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_7_]], [[LOAD_PARAM_1_MEM_7_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_111_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_7_]], [[LOAD_PARAM_3_MEM_7_]] : vector<4xf32>
// CHECK:             [[VAR_112_:%.+]] = arith.addf [[VAR_110_]], [[VAR_111_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_113_:%.+]] = arith.mulf [[VAR_112_]], [[VAR_cst_1_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_114_:%.+]] = arith.minnumf [[VAR_109_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_115_:%.+]] = arith.minnumf [[VAR_113_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_116_:%.+]] = arith.maxnumf [[VAR_114_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_117_:%.+]] = arith.maxnumf [[VAR_115_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_118_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_116_]], [[VAR_117_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_118_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_6_]], [[CST_24_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x32xf16, #map>
// CHECK:         }
}

// -----

// Sub join, no scalar Mul (no multiply emitted), table first in MulA, rank 2
// tables, D = 64 (one whole stick per row), 3D layout.

func.func @test_mul_sub_stick_no_scalar_d64(%arg0: tensor<8x64xf32>, %arg1: tensor<2x4x8x64xf32>, %arg2: tensor<2x4x8x64xf32>, %arg3: tensor<8x64xf32>) -> tensor<8x8x64xf16, #zhigh.layout<{dataLayout = "3D"}>> {
  %0 = "onnx.Fused"(%arg0, %arg1, %arg2, %arg3) <{kind = "zhigh.mul-add-stick"}> ({
  ^bb0(%a0: tensor<8x64xf32>, %a1: tensor<2x4x8x64xf32>, %b0: tensor<2x4x8x64xf32>, %b1: tensor<8x64xf32>):
    %shape = onnx.Constant dense<[8, 8, 64]> : tensor<3xi64>
    %1 = "onnx.Mul"(%a0, %a1) : (tensor<8x64xf32>, tensor<2x4x8x64xf32>) -> tensor<2x4x8x64xf32>
    %2 = "onnx.Mul"(%b0, %b1) : (tensor<2x4x8x64xf32>, tensor<8x64xf32>) -> tensor<2x4x8x64xf32>
    %3 = "onnx.Sub"(%1, %2) : (tensor<2x4x8x64xf32>, tensor<2x4x8x64xf32>) -> tensor<2x4x8x64xf32>
    %4 = "onnx.Reshape"(%3, %shape) <{allowzero = 0 : si64}> : (tensor<2x4x8x64xf32>, tensor<3xi64>) -> tensor<8x8x64xf32>
    %5 = "zhigh.Stick"(%4) <{layout = "3D"}> : (tensor<8x8x64xf32>) -> tensor<8x8x64xf16, #zhigh.layout<{dataLayout = "3D"}>>
    onnx.Yield %5 : tensor<8x8x64xf16, #zhigh.layout<{dataLayout = "3D"}>>
  }) {isSub = true, mulScalar = 1.000000e+00 : f32, reshapeCollapsedCount = 2 : i64, reshapeFirstCollapsedDim = 0 : i64, stickFormat = "3D"} : (tensor<8x64xf32>, tensor<2x4x8x64xf32>, tensor<2x4x8x64xf32>, tensor<8x64xf32>) -> tensor<8x8x64xf16, #zhigh.layout<{dataLayout = "3D"}>>
  return %0 : tensor<8x8x64xf16, #zhigh.layout<{dataLayout = "3D"}>>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (0, d2 floordiv 64, d0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d0 * 4 + d1)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 floordiv 64)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 * 64 + 8)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0) -> (d0 * 64 + 16)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0) -> (d0 * 64 + 24)>
// CHECK-DAG:   [[MAP_7_:#.+]] = affine_map<(d0) -> (d0 * 64 + 32)>
// CHECK-DAG:   [[MAP_8_:#.+]] = affine_map<(d0) -> (d0 * 64 + 40)>
// CHECK-DAG:   [[MAP_9_:#.+]] = affine_map<(d0) -> (d0 * 64 + 48)>
// CHECK-DAG:   [[MAP_10_:#.+]] = affine_map<(d0) -> (d0 * 64 + 56)>
// CHECK-LABEL:  func.func @test_mul_sub_stick_no_scalar_d64
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<8x64xf32>, [[PARAM_1_:%.+]]: memref<2x4x8x64xf32>, [[PARAM_2_:%.+]]: memref<2x4x8x64xf32>, [[PARAM_3_:%.+]]: memref<8x64xf32>) -> memref<8x8x64xf16, #map> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<-8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[CST_56_:%.+]] = arith.constant 56 : index
// CHECK-DAG:       [[CST_48_:%.+]] = arith.constant 48 : index
// CHECK-DAG:       [[CST_40_:%.+]] = arith.constant 40 : index
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[CST_24_:%.+]] = arith.constant 24 : index
// CHECK-DAG:       [[CST_16_:%.+]] = arith.constant 16 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 4096 : memref<8x8x64xf16, #map>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<8x8x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 1, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 8){
// CHECK:             [[VAR_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]]#2)
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#0, [[VAR_1_]]#1)
// CHECK:             [[VAR_4_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_3_]], [[VAR_1_]]#3, [[VAR_2_]]{{.}} : memref<8x8x64xf16, #map>
// CHECK-DAG:         [[VAR_5_:%.+]] = affine.apply [[MAP_3_]]([[VAR_4_]])
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#3, [[VAR_2_]]{{.}} : memref<8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.addi [[VAR_2_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#3, [[VAR_7_]]{{.}} : memref<8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#3, [[VAR_2_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.addi [[VAR_2_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#3, [[VAR_10_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#3, [[VAR_2_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.addi [[VAR_2_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_1_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#3, [[VAR_13_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[VAR_1_]]#3, [[VAR_2_]]{{.}} : memref<8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_16_:%.+]] = arith.addi [[VAR_2_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_1_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[VAR_1_]]#3, [[VAR_16_]]{{.}} : memref<8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_18_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_19_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_]], [[LOAD_PARAM_3_MEM_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_20_:%.+]] = arith.subf [[VAR_18_]], [[VAR_19_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_21_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_1_]], [[LOAD_PARAM_1_MEM_1_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_22_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_1_]], [[LOAD_PARAM_3_MEM_1_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_23_:%.+]] = arith.subf [[VAR_21_]], [[VAR_22_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_24_:%.+]] = arith.minnumf [[VAR_20_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_25_:%.+]] = arith.minnumf [[VAR_23_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_26_:%.+]] = arith.maxnumf [[VAR_24_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_27_:%.+]] = arith.maxnumf [[VAR_25_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_28_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_26_]], [[VAR_27_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_28_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_5_]], [[CST_0_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_29_:%.+]] = affine.apply [[MAP_4_]]([[VAR_1_]]#2)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_2_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#3, [[VAR_29_]]{{.}} : memref<8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_31_:%.+]] = arith.addi [[VAR_29_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_3_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#3, [[VAR_31_]]{{.}} : memref<8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_33_:%.+]] = affine.apply [[MAP_4_]]([[VAR_1_]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_2_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#3, [[VAR_33_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_35_:%.+]] = arith.addi [[VAR_33_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_3_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#3, [[VAR_35_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_37_:%.+]] = affine.apply [[MAP_4_]]([[VAR_1_]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_2_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#3, [[VAR_37_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_39_:%.+]] = arith.addi [[VAR_37_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_3_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#3, [[VAR_39_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_41_:%.+]] = affine.apply [[MAP_4_]]([[VAR_1_]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_2_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[VAR_1_]]#3, [[VAR_41_]]{{.}} : memref<8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_43_:%.+]] = arith.addi [[VAR_41_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_3_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[VAR_1_]]#3, [[VAR_43_]]{{.}} : memref<8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_45_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_2_]], [[LOAD_PARAM_1_MEM_2_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_46_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_2_]], [[LOAD_PARAM_3_MEM_2_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_47_:%.+]] = arith.subf [[VAR_45_]], [[VAR_46_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_48_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_3_]], [[LOAD_PARAM_1_MEM_3_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_49_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_3_]], [[LOAD_PARAM_3_MEM_3_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_50_:%.+]] = arith.subf [[VAR_48_]], [[VAR_49_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_51_:%.+]] = arith.minnumf [[VAR_47_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_52_:%.+]] = arith.minnumf [[VAR_50_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_53_:%.+]] = arith.maxnumf [[VAR_51_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_54_:%.+]] = arith.maxnumf [[VAR_52_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_55_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_53_]], [[VAR_54_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_55_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_5_]], [[CST_8_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_56_:%.+]] = affine.apply [[MAP_5_]]([[VAR_1_]]#2)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_4_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#3, [[VAR_56_]]{{.}} : memref<8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_58_:%.+]] = arith.addi [[VAR_56_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_5_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#3, [[VAR_58_]]{{.}} : memref<8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_60_:%.+]] = affine.apply [[MAP_5_]]([[VAR_1_]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_4_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#3, [[VAR_60_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_62_:%.+]] = arith.addi [[VAR_60_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_5_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#3, [[VAR_62_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_64_:%.+]] = affine.apply [[MAP_5_]]([[VAR_1_]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_4_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#3, [[VAR_64_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_66_:%.+]] = arith.addi [[VAR_64_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_5_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#3, [[VAR_66_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_68_:%.+]] = affine.apply [[MAP_5_]]([[VAR_1_]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_4_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[VAR_1_]]#3, [[VAR_68_]]{{.}} : memref<8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_70_:%.+]] = arith.addi [[VAR_68_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_5_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[VAR_1_]]#3, [[VAR_70_]]{{.}} : memref<8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_72_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_4_]], [[LOAD_PARAM_1_MEM_4_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_73_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_4_]], [[LOAD_PARAM_3_MEM_4_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_74_:%.+]] = arith.subf [[VAR_72_]], [[VAR_73_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_75_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_5_]], [[LOAD_PARAM_1_MEM_5_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_76_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_5_]], [[LOAD_PARAM_3_MEM_5_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_77_:%.+]] = arith.subf [[VAR_75_]], [[VAR_76_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_78_:%.+]] = arith.minnumf [[VAR_74_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_79_:%.+]] = arith.minnumf [[VAR_77_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_80_:%.+]] = arith.maxnumf [[VAR_78_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_81_:%.+]] = arith.maxnumf [[VAR_79_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_82_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_80_]], [[VAR_81_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_82_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_5_]], [[CST_16_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_83_:%.+]] = affine.apply [[MAP_6_]]([[VAR_1_]]#2)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_6_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#3, [[VAR_83_]]{{.}} : memref<8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_85_:%.+]] = arith.addi [[VAR_83_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_7_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#3, [[VAR_85_]]{{.}} : memref<8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_87_:%.+]] = affine.apply [[MAP_6_]]([[VAR_1_]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_6_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#3, [[VAR_87_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_89_:%.+]] = arith.addi [[VAR_87_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_7_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#3, [[VAR_89_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_91_:%.+]] = affine.apply [[MAP_6_]]([[VAR_1_]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_6_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#3, [[VAR_91_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_93_:%.+]] = arith.addi [[VAR_91_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_7_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#3, [[VAR_93_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_95_:%.+]] = affine.apply [[MAP_6_]]([[VAR_1_]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_6_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[VAR_1_]]#3, [[VAR_95_]]{{.}} : memref<8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_97_:%.+]] = arith.addi [[VAR_95_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_7_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[VAR_1_]]#3, [[VAR_97_]]{{.}} : memref<8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_99_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_6_]], [[LOAD_PARAM_1_MEM_6_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_100_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_6_]], [[LOAD_PARAM_3_MEM_6_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_101_:%.+]] = arith.subf [[VAR_99_]], [[VAR_100_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_102_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_7_]], [[LOAD_PARAM_1_MEM_7_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_103_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_7_]], [[LOAD_PARAM_3_MEM_7_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_104_:%.+]] = arith.subf [[VAR_102_]], [[VAR_103_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_105_:%.+]] = arith.minnumf [[VAR_101_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_106_:%.+]] = arith.minnumf [[VAR_104_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_107_:%.+]] = arith.maxnumf [[VAR_105_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_108_:%.+]] = arith.maxnumf [[VAR_106_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_109_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_107_]], [[VAR_108_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_109_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_5_]], [[CST_24_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_110_:%.+]] = affine.apply [[MAP_7_]]([[VAR_1_]]#2)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_8_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#3, [[VAR_110_]]{{.}} : memref<8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_112_:%.+]] = arith.addi [[VAR_110_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_9_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#3, [[VAR_112_]]{{.}} : memref<8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_114_:%.+]] = affine.apply [[MAP_7_]]([[VAR_1_]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_8_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#3, [[VAR_114_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_116_:%.+]] = arith.addi [[VAR_114_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_9_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#3, [[VAR_116_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_118_:%.+]] = affine.apply [[MAP_7_]]([[VAR_1_]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_8_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#3, [[VAR_118_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_120_:%.+]] = arith.addi [[VAR_118_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_9_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#3, [[VAR_120_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_122_:%.+]] = affine.apply [[MAP_7_]]([[VAR_1_]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_8_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[VAR_1_]]#3, [[VAR_122_]]{{.}} : memref<8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_124_:%.+]] = arith.addi [[VAR_122_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_9_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[VAR_1_]]#3, [[VAR_124_]]{{.}} : memref<8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_126_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_8_]], [[LOAD_PARAM_1_MEM_8_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_127_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_8_]], [[LOAD_PARAM_3_MEM_8_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_128_:%.+]] = arith.subf [[VAR_126_]], [[VAR_127_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_129_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_9_]], [[LOAD_PARAM_1_MEM_9_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_130_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_9_]], [[LOAD_PARAM_3_MEM_9_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_131_:%.+]] = arith.subf [[VAR_129_]], [[VAR_130_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_132_:%.+]] = arith.minnumf [[VAR_128_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_133_:%.+]] = arith.minnumf [[VAR_131_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_134_:%.+]] = arith.maxnumf [[VAR_132_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_135_:%.+]] = arith.maxnumf [[VAR_133_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_136_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_134_]], [[VAR_135_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_136_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_5_]], [[CST_32_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_137_:%.+]] = affine.apply [[MAP_8_]]([[VAR_1_]]#2)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_10_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#3, [[VAR_137_]]{{.}} : memref<8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_139_:%.+]] = arith.addi [[VAR_137_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_11_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#3, [[VAR_139_]]{{.}} : memref<8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_141_:%.+]] = affine.apply [[MAP_8_]]([[VAR_1_]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_10_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#3, [[VAR_141_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_143_:%.+]] = arith.addi [[VAR_141_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_11_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#3, [[VAR_143_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_145_:%.+]] = affine.apply [[MAP_8_]]([[VAR_1_]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_10_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#3, [[VAR_145_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_147_:%.+]] = arith.addi [[VAR_145_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_11_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#3, [[VAR_147_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_149_:%.+]] = affine.apply [[MAP_8_]]([[VAR_1_]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_10_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[VAR_1_]]#3, [[VAR_149_]]{{.}} : memref<8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_151_:%.+]] = arith.addi [[VAR_149_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_11_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[VAR_1_]]#3, [[VAR_151_]]{{.}} : memref<8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_153_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_10_]], [[LOAD_PARAM_1_MEM_10_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_154_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_10_]], [[LOAD_PARAM_3_MEM_10_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_155_:%.+]] = arith.subf [[VAR_153_]], [[VAR_154_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_156_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_11_]], [[LOAD_PARAM_1_MEM_11_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_157_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_11_]], [[LOAD_PARAM_3_MEM_11_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_158_:%.+]] = arith.subf [[VAR_156_]], [[VAR_157_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_159_:%.+]] = arith.minnumf [[VAR_155_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_160_:%.+]] = arith.minnumf [[VAR_158_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_161_:%.+]] = arith.maxnumf [[VAR_159_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_162_:%.+]] = arith.maxnumf [[VAR_160_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_163_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_161_]], [[VAR_162_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_163_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_5_]], [[CST_40_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_164_:%.+]] = affine.apply [[MAP_9_]]([[VAR_1_]]#2)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_12_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#3, [[VAR_164_]]{{.}} : memref<8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_166_:%.+]] = arith.addi [[VAR_164_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_13_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#3, [[VAR_166_]]{{.}} : memref<8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_168_:%.+]] = affine.apply [[MAP_9_]]([[VAR_1_]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_12_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#3, [[VAR_168_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_170_:%.+]] = arith.addi [[VAR_168_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_13_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#3, [[VAR_170_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_172_:%.+]] = affine.apply [[MAP_9_]]([[VAR_1_]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_12_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#3, [[VAR_172_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_174_:%.+]] = arith.addi [[VAR_172_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_13_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#3, [[VAR_174_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_176_:%.+]] = affine.apply [[MAP_9_]]([[VAR_1_]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_12_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[VAR_1_]]#3, [[VAR_176_]]{{.}} : memref<8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_178_:%.+]] = arith.addi [[VAR_176_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_13_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[VAR_1_]]#3, [[VAR_178_]]{{.}} : memref<8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_180_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_12_]], [[LOAD_PARAM_1_MEM_12_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_181_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_12_]], [[LOAD_PARAM_3_MEM_12_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_182_:%.+]] = arith.subf [[VAR_180_]], [[VAR_181_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_183_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_13_]], [[LOAD_PARAM_1_MEM_13_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_184_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_13_]], [[LOAD_PARAM_3_MEM_13_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_185_:%.+]] = arith.subf [[VAR_183_]], [[VAR_184_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_186_:%.+]] = arith.minnumf [[VAR_182_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_187_:%.+]] = arith.minnumf [[VAR_185_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_188_:%.+]] = arith.maxnumf [[VAR_186_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_189_:%.+]] = arith.maxnumf [[VAR_187_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_190_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_188_]], [[VAR_189_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_190_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_5_]], [[CST_48_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_191_:%.+]] = affine.apply [[MAP_10_]]([[VAR_1_]]#2)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_14_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#3, [[VAR_191_]]{{.}} : memref<8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_193_:%.+]] = arith.addi [[VAR_191_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_15_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#3, [[VAR_193_]]{{.}} : memref<8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_195_:%.+]] = affine.apply [[MAP_10_]]([[VAR_1_]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_14_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#3, [[VAR_195_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_197_:%.+]] = arith.addi [[VAR_195_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_15_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#3, [[VAR_197_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_199_:%.+]] = affine.apply [[MAP_10_]]([[VAR_1_]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_14_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#3, [[VAR_199_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_201_:%.+]] = arith.addi [[VAR_199_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_15_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#3, [[VAR_201_]]{{.}} : memref<2x4x8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_203_:%.+]] = affine.apply [[MAP_10_]]([[VAR_1_]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_14_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[VAR_1_]]#3, [[VAR_203_]]{{.}} : memref<8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_205_:%.+]] = arith.addi [[VAR_203_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_15_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[VAR_1_]]#3, [[VAR_205_]]{{.}} : memref<8x64xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_207_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_14_]], [[LOAD_PARAM_1_MEM_14_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_208_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_14_]], [[LOAD_PARAM_3_MEM_14_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_209_:%.+]] = arith.subf [[VAR_207_]], [[VAR_208_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_210_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_15_]], [[LOAD_PARAM_1_MEM_15_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_211_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_15_]], [[LOAD_PARAM_3_MEM_15_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_212_:%.+]] = arith.subf [[VAR_210_]], [[VAR_211_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_213_:%.+]] = arith.minnumf [[VAR_209_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_214_:%.+]] = arith.minnumf [[VAR_212_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_215_:%.+]] = arith.maxnumf [[VAR_213_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_216_:%.+]] = arith.maxnumf [[VAR_214_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_217_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_215_]], [[VAR_216_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_217_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_5_]], [[CST_56_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<8x8x64xf16, #map>
// CHECK:         }
}

// -----

// Rank 3 join with a no-op Reshape, D = 128 (two sticks per row), 3DS.

func.func @test_mul_add_stick_noop_reshape_d128(%arg0: tensor<6x8x128xf32>, %arg1: tensor<1x8x128xf32>, %arg2: tensor<6x8x128xf32>, %arg3: tensor<1x8x128xf32>) -> tensor<6x8x128xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %0 = "onnx.Fused"(%arg0, %arg1, %arg2, %arg3) <{kind = "zhigh.mul-add-stick"}> ({
  ^bb0(%a0: tensor<6x8x128xf32>, %a1: tensor<1x8x128xf32>, %b0: tensor<6x8x128xf32>, %b1: tensor<1x8x128xf32>):
    %shape = onnx.Constant dense<[6, 8, 128]> : tensor<3xi64>
    %k = onnx.Constant dense<2.0> : tensor<f32>
    %1 = "onnx.Mul"(%a0, %a1) : (tensor<6x8x128xf32>, tensor<1x8x128xf32>) -> tensor<6x8x128xf32>
    %2 = "onnx.Mul"(%b0, %b1) : (tensor<6x8x128xf32>, tensor<1x8x128xf32>) -> tensor<6x8x128xf32>
    %3 = "onnx.Add"(%1, %2) : (tensor<6x8x128xf32>, tensor<6x8x128xf32>) -> tensor<6x8x128xf32>
    %4 = "onnx.Mul"(%k, %3) : (tensor<f32>, tensor<6x8x128xf32>) -> tensor<6x8x128xf32>
    %5 = "onnx.Reshape"(%4, %shape) <{allowzero = 0 : si64}> : (tensor<6x8x128xf32>, tensor<3xi64>) -> tensor<6x8x128xf32>
    %6 = "zhigh.Stick"(%5) <{layout = "3DS"}> : (tensor<6x8x128xf32>) -> tensor<6x8x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    onnx.Yield %6 : tensor<6x8x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  }) {isSub = false, mulScalar = 2.000000e+00 : f32, reshapeCollapsedCount = 0 : i64, reshapeFirstCollapsedDim = -1 : i64, stickFormat = "3DS"} : (tensor<6x8x128xf32>, tensor<1x8x128xf32>, tensor<6x8x128xf32>, tensor<1x8x128xf32>) -> tensor<6x8x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %0 : tensor<6x8x128xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 floordiv 64)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 * 64 + 8)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 * 64 + 16)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0) -> (d0 * 64 + 24)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0) -> (d0 * 64 + 32)>
// CHECK-DAG:   [[MAP_7_:#.+]] = affine_map<(d0) -> (d0 * 64 + 40)>
// CHECK-DAG:   [[MAP_8_:#.+]] = affine_map<(d0) -> (d0 * 64 + 48)>
// CHECK-DAG:   [[MAP_9_:#.+]] = affine_map<(d0) -> (d0 * 64 + 56)>
// CHECK-LABEL:  func.func @test_mul_add_stick_noop_reshape_d128
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<6x8x128xf32>, [[PARAM_1_:%.+]]: memref<1x8x128xf32>, [[PARAM_2_:%.+]]: memref<6x8x128xf32>, [[PARAM_3_:%.+]]: memref<1x8x128xf32>) -> memref<6x8x128xf16, #map> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<-8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[VAR_cst_1_:%.+]] = arith.constant dense<2.000000e+00> : vector<4xf32>
// CHECK-DAG:       [[CST_56_:%.+]] = arith.constant 56 : index
// CHECK-DAG:       [[CST_48_:%.+]] = arith.constant 48 : index
// CHECK-DAG:       [[CST_40_:%.+]] = arith.constant 40 : index
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[CST_24_:%.+]] = arith.constant 24 : index
// CHECK-DAG:       [[CST_16_:%.+]] = arith.constant 16 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 4096 : memref<6x8x128xf16, #map>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<6x8x128xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 6, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 8){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[VAR_2_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]]#1)
// CHECK:             [[VAR_3_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_2_]]{{.}} : memref<6x8x128xf16, #map>
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.apply [[MAP_2_]]([[VAR_3_]])
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_2_]]{{.}} : memref<6x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.addi [[VAR_2_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_6_]]{{.}} : memref<6x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_0_]], [[VAR_1_]]#2, [[VAR_2_]]{{.}} : memref<1x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.addi [[VAR_2_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_0_]], [[VAR_1_]]#2, [[VAR_9_]]{{.}} : memref<1x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_2_]]{{.}} : memref<6x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.addi [[VAR_2_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_1_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_12_]]{{.}} : memref<6x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[CST_0_]], [[VAR_1_]]#2, [[VAR_2_]]{{.}} : memref<1x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_15_:%.+]] = arith.addi [[VAR_2_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_1_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[CST_0_]], [[VAR_1_]]#2, [[VAR_15_]]{{.}} : memref<1x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_17_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_18_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_]], [[LOAD_PARAM_3_MEM_]] : vector<4xf32>
// CHECK:             [[VAR_19_:%.+]] = arith.addf [[VAR_17_]], [[VAR_18_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_20_:%.+]] = arith.mulf [[VAR_19_]], [[VAR_cst_1_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_21_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_1_]], [[LOAD_PARAM_1_MEM_1_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_22_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_1_]], [[LOAD_PARAM_3_MEM_1_]] : vector<4xf32>
// CHECK:             [[VAR_23_:%.+]] = arith.addf [[VAR_21_]], [[VAR_22_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_24_:%.+]] = arith.mulf [[VAR_23_]], [[VAR_cst_1_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_25_:%.+]] = arith.minnumf [[VAR_20_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_26_:%.+]] = arith.minnumf [[VAR_24_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_27_:%.+]] = arith.maxnumf [[VAR_25_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_28_:%.+]] = arith.maxnumf [[VAR_26_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_29_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_27_]], [[VAR_28_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_29_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[CST_0_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_30_:%.+]] = affine.apply [[MAP_3_]]([[VAR_1_]]#1)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_2_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_30_]]{{.}} : memref<6x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_32_:%.+]] = arith.addi [[VAR_30_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_3_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_32_]]{{.}} : memref<6x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_34_:%.+]] = affine.apply [[MAP_3_]]([[VAR_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_2_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_0_]], [[VAR_1_]]#2, [[VAR_34_]]{{.}} : memref<1x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_36_:%.+]] = arith.addi [[VAR_34_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_3_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_0_]], [[VAR_1_]]#2, [[VAR_36_]]{{.}} : memref<1x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_38_:%.+]] = affine.apply [[MAP_3_]]([[VAR_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_2_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_38_]]{{.}} : memref<6x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_40_:%.+]] = arith.addi [[VAR_38_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_3_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_40_]]{{.}} : memref<6x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_42_:%.+]] = affine.apply [[MAP_3_]]([[VAR_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_2_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[CST_0_]], [[VAR_1_]]#2, [[VAR_42_]]{{.}} : memref<1x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_44_:%.+]] = arith.addi [[VAR_42_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_3_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[CST_0_]], [[VAR_1_]]#2, [[VAR_44_]]{{.}} : memref<1x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_46_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_2_]], [[LOAD_PARAM_1_MEM_2_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_47_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_2_]], [[LOAD_PARAM_3_MEM_2_]] : vector<4xf32>
// CHECK:             [[VAR_48_:%.+]] = arith.addf [[VAR_46_]], [[VAR_47_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_49_:%.+]] = arith.mulf [[VAR_48_]], [[VAR_cst_1_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_50_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_3_]], [[LOAD_PARAM_1_MEM_3_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_51_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_3_]], [[LOAD_PARAM_3_MEM_3_]] : vector<4xf32>
// CHECK:             [[VAR_52_:%.+]] = arith.addf [[VAR_50_]], [[VAR_51_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_53_:%.+]] = arith.mulf [[VAR_52_]], [[VAR_cst_1_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_54_:%.+]] = arith.minnumf [[VAR_49_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_55_:%.+]] = arith.minnumf [[VAR_53_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_56_:%.+]] = arith.maxnumf [[VAR_54_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_57_:%.+]] = arith.maxnumf [[VAR_55_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_58_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_56_]], [[VAR_57_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_58_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[CST_8_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_59_:%.+]] = affine.apply [[MAP_4_]]([[VAR_1_]]#1)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_4_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_59_]]{{.}} : memref<6x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_61_:%.+]] = arith.addi [[VAR_59_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_5_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_61_]]{{.}} : memref<6x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_63_:%.+]] = affine.apply [[MAP_4_]]([[VAR_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_4_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_0_]], [[VAR_1_]]#2, [[VAR_63_]]{{.}} : memref<1x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_65_:%.+]] = arith.addi [[VAR_63_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_5_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_0_]], [[VAR_1_]]#2, [[VAR_65_]]{{.}} : memref<1x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_67_:%.+]] = affine.apply [[MAP_4_]]([[VAR_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_4_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_67_]]{{.}} : memref<6x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_69_:%.+]] = arith.addi [[VAR_67_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_5_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_69_]]{{.}} : memref<6x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_71_:%.+]] = affine.apply [[MAP_4_]]([[VAR_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_4_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[CST_0_]], [[VAR_1_]]#2, [[VAR_71_]]{{.}} : memref<1x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_73_:%.+]] = arith.addi [[VAR_71_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_5_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[CST_0_]], [[VAR_1_]]#2, [[VAR_73_]]{{.}} : memref<1x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_75_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_4_]], [[LOAD_PARAM_1_MEM_4_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_76_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_4_]], [[LOAD_PARAM_3_MEM_4_]] : vector<4xf32>
// CHECK:             [[VAR_77_:%.+]] = arith.addf [[VAR_75_]], [[VAR_76_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_78_:%.+]] = arith.mulf [[VAR_77_]], [[VAR_cst_1_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_79_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_5_]], [[LOAD_PARAM_1_MEM_5_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_80_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_5_]], [[LOAD_PARAM_3_MEM_5_]] : vector<4xf32>
// CHECK:             [[VAR_81_:%.+]] = arith.addf [[VAR_79_]], [[VAR_80_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_82_:%.+]] = arith.mulf [[VAR_81_]], [[VAR_cst_1_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_83_:%.+]] = arith.minnumf [[VAR_78_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_84_:%.+]] = arith.minnumf [[VAR_82_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_85_:%.+]] = arith.maxnumf [[VAR_83_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_86_:%.+]] = arith.maxnumf [[VAR_84_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_87_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_85_]], [[VAR_86_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_87_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[CST_16_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_88_:%.+]] = affine.apply [[MAP_5_]]([[VAR_1_]]#1)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_6_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_88_]]{{.}} : memref<6x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_90_:%.+]] = arith.addi [[VAR_88_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_7_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_90_]]{{.}} : memref<6x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_92_:%.+]] = affine.apply [[MAP_5_]]([[VAR_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_6_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_0_]], [[VAR_1_]]#2, [[VAR_92_]]{{.}} : memref<1x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_94_:%.+]] = arith.addi [[VAR_92_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_7_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_0_]], [[VAR_1_]]#2, [[VAR_94_]]{{.}} : memref<1x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_96_:%.+]] = affine.apply [[MAP_5_]]([[VAR_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_6_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_96_]]{{.}} : memref<6x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_98_:%.+]] = arith.addi [[VAR_96_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_7_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_98_]]{{.}} : memref<6x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_100_:%.+]] = affine.apply [[MAP_5_]]([[VAR_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_6_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[CST_0_]], [[VAR_1_]]#2, [[VAR_100_]]{{.}} : memref<1x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_102_:%.+]] = arith.addi [[VAR_100_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_7_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[CST_0_]], [[VAR_1_]]#2, [[VAR_102_]]{{.}} : memref<1x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_104_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_6_]], [[LOAD_PARAM_1_MEM_6_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_105_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_6_]], [[LOAD_PARAM_3_MEM_6_]] : vector<4xf32>
// CHECK:             [[VAR_106_:%.+]] = arith.addf [[VAR_104_]], [[VAR_105_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_107_:%.+]] = arith.mulf [[VAR_106_]], [[VAR_cst_1_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_108_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_7_]], [[LOAD_PARAM_1_MEM_7_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_109_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_7_]], [[LOAD_PARAM_3_MEM_7_]] : vector<4xf32>
// CHECK:             [[VAR_110_:%.+]] = arith.addf [[VAR_108_]], [[VAR_109_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_111_:%.+]] = arith.mulf [[VAR_110_]], [[VAR_cst_1_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_112_:%.+]] = arith.minnumf [[VAR_107_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_113_:%.+]] = arith.minnumf [[VAR_111_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_114_:%.+]] = arith.maxnumf [[VAR_112_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_115_:%.+]] = arith.maxnumf [[VAR_113_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_116_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_114_]], [[VAR_115_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_116_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[CST_24_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_117_:%.+]] = affine.apply [[MAP_6_]]([[VAR_1_]]#1)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_8_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_117_]]{{.}} : memref<6x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_119_:%.+]] = arith.addi [[VAR_117_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_9_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_119_]]{{.}} : memref<6x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_121_:%.+]] = affine.apply [[MAP_6_]]([[VAR_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_8_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_0_]], [[VAR_1_]]#2, [[VAR_121_]]{{.}} : memref<1x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_123_:%.+]] = arith.addi [[VAR_121_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_9_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_0_]], [[VAR_1_]]#2, [[VAR_123_]]{{.}} : memref<1x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_125_:%.+]] = affine.apply [[MAP_6_]]([[VAR_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_8_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_125_]]{{.}} : memref<6x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_127_:%.+]] = arith.addi [[VAR_125_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_9_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_127_]]{{.}} : memref<6x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_129_:%.+]] = affine.apply [[MAP_6_]]([[VAR_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_8_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[CST_0_]], [[VAR_1_]]#2, [[VAR_129_]]{{.}} : memref<1x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_131_:%.+]] = arith.addi [[VAR_129_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_9_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[CST_0_]], [[VAR_1_]]#2, [[VAR_131_]]{{.}} : memref<1x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_133_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_8_]], [[LOAD_PARAM_1_MEM_8_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_134_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_8_]], [[LOAD_PARAM_3_MEM_8_]] : vector<4xf32>
// CHECK:             [[VAR_135_:%.+]] = arith.addf [[VAR_133_]], [[VAR_134_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_136_:%.+]] = arith.mulf [[VAR_135_]], [[VAR_cst_1_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_137_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_9_]], [[LOAD_PARAM_1_MEM_9_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_138_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_9_]], [[LOAD_PARAM_3_MEM_9_]] : vector<4xf32>
// CHECK:             [[VAR_139_:%.+]] = arith.addf [[VAR_137_]], [[VAR_138_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_140_:%.+]] = arith.mulf [[VAR_139_]], [[VAR_cst_1_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_141_:%.+]] = arith.minnumf [[VAR_136_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_142_:%.+]] = arith.minnumf [[VAR_140_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_143_:%.+]] = arith.maxnumf [[VAR_141_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_144_:%.+]] = arith.maxnumf [[VAR_142_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_145_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_143_]], [[VAR_144_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_145_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[CST_32_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_146_:%.+]] = affine.apply [[MAP_7_]]([[VAR_1_]]#1)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_10_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_146_]]{{.}} : memref<6x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_148_:%.+]] = arith.addi [[VAR_146_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_11_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_148_]]{{.}} : memref<6x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_150_:%.+]] = affine.apply [[MAP_7_]]([[VAR_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_10_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_0_]], [[VAR_1_]]#2, [[VAR_150_]]{{.}} : memref<1x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_152_:%.+]] = arith.addi [[VAR_150_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_11_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_0_]], [[VAR_1_]]#2, [[VAR_152_]]{{.}} : memref<1x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_154_:%.+]] = affine.apply [[MAP_7_]]([[VAR_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_10_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_154_]]{{.}} : memref<6x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_156_:%.+]] = arith.addi [[VAR_154_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_11_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_156_]]{{.}} : memref<6x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_158_:%.+]] = affine.apply [[MAP_7_]]([[VAR_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_10_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[CST_0_]], [[VAR_1_]]#2, [[VAR_158_]]{{.}} : memref<1x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_160_:%.+]] = arith.addi [[VAR_158_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_11_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[CST_0_]], [[VAR_1_]]#2, [[VAR_160_]]{{.}} : memref<1x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_162_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_10_]], [[LOAD_PARAM_1_MEM_10_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_163_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_10_]], [[LOAD_PARAM_3_MEM_10_]] : vector<4xf32>
// CHECK:             [[VAR_164_:%.+]] = arith.addf [[VAR_162_]], [[VAR_163_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_165_:%.+]] = arith.mulf [[VAR_164_]], [[VAR_cst_1_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_166_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_11_]], [[LOAD_PARAM_1_MEM_11_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_167_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_11_]], [[LOAD_PARAM_3_MEM_11_]] : vector<4xf32>
// CHECK:             [[VAR_168_:%.+]] = arith.addf [[VAR_166_]], [[VAR_167_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_169_:%.+]] = arith.mulf [[VAR_168_]], [[VAR_cst_1_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_170_:%.+]] = arith.minnumf [[VAR_165_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_171_:%.+]] = arith.minnumf [[VAR_169_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_172_:%.+]] = arith.maxnumf [[VAR_170_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_173_:%.+]] = arith.maxnumf [[VAR_171_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_174_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_172_]], [[VAR_173_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_174_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[CST_40_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_175_:%.+]] = affine.apply [[MAP_8_]]([[VAR_1_]]#1)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_12_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_175_]]{{.}} : memref<6x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_177_:%.+]] = arith.addi [[VAR_175_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_13_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_177_]]{{.}} : memref<6x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_179_:%.+]] = affine.apply [[MAP_8_]]([[VAR_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_12_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_0_]], [[VAR_1_]]#2, [[VAR_179_]]{{.}} : memref<1x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_181_:%.+]] = arith.addi [[VAR_179_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_13_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_0_]], [[VAR_1_]]#2, [[VAR_181_]]{{.}} : memref<1x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_183_:%.+]] = affine.apply [[MAP_8_]]([[VAR_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_12_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_183_]]{{.}} : memref<6x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_185_:%.+]] = arith.addi [[VAR_183_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_13_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_185_]]{{.}} : memref<6x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_187_:%.+]] = affine.apply [[MAP_8_]]([[VAR_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_12_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[CST_0_]], [[VAR_1_]]#2, [[VAR_187_]]{{.}} : memref<1x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_189_:%.+]] = arith.addi [[VAR_187_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_13_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[CST_0_]], [[VAR_1_]]#2, [[VAR_189_]]{{.}} : memref<1x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_191_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_12_]], [[LOAD_PARAM_1_MEM_12_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_192_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_12_]], [[LOAD_PARAM_3_MEM_12_]] : vector<4xf32>
// CHECK:             [[VAR_193_:%.+]] = arith.addf [[VAR_191_]], [[VAR_192_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_194_:%.+]] = arith.mulf [[VAR_193_]], [[VAR_cst_1_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_195_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_13_]], [[LOAD_PARAM_1_MEM_13_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_196_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_13_]], [[LOAD_PARAM_3_MEM_13_]] : vector<4xf32>
// CHECK:             [[VAR_197_:%.+]] = arith.addf [[VAR_195_]], [[VAR_196_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_198_:%.+]] = arith.mulf [[VAR_197_]], [[VAR_cst_1_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_199_:%.+]] = arith.minnumf [[VAR_194_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_200_:%.+]] = arith.minnumf [[VAR_198_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_201_:%.+]] = arith.maxnumf [[VAR_199_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_202_:%.+]] = arith.maxnumf [[VAR_200_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_203_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_201_]], [[VAR_202_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_203_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[CST_48_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_204_:%.+]] = affine.apply [[MAP_9_]]([[VAR_1_]]#1)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_14_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_204_]]{{.}} : memref<6x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_206_:%.+]] = arith.addi [[VAR_204_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_15_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_206_]]{{.}} : memref<6x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_208_:%.+]] = affine.apply [[MAP_9_]]([[VAR_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_14_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_0_]], [[VAR_1_]]#2, [[VAR_208_]]{{.}} : memref<1x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_210_:%.+]] = arith.addi [[VAR_208_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_15_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[CST_0_]], [[VAR_1_]]#2, [[VAR_210_]]{{.}} : memref<1x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_212_:%.+]] = affine.apply [[MAP_9_]]([[VAR_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_14_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_212_]]{{.}} : memref<6x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_214_:%.+]] = arith.addi [[VAR_212_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_15_:%.+]] = vector.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_214_]]{{.}} : memref<6x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_216_:%.+]] = affine.apply [[MAP_9_]]([[VAR_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_14_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[CST_0_]], [[VAR_1_]]#2, [[VAR_216_]]{{.}} : memref<1x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_218_:%.+]] = arith.addi [[VAR_216_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_15_:%.+]] = vector.load [[PARAM_3_]]{{.}}[[CST_0_]], [[VAR_1_]]#2, [[VAR_218_]]{{.}} : memref<1x8x128xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_220_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_14_]], [[LOAD_PARAM_1_MEM_14_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_221_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_14_]], [[LOAD_PARAM_3_MEM_14_]] : vector<4xf32>
// CHECK:             [[VAR_222_:%.+]] = arith.addf [[VAR_220_]], [[VAR_221_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_223_:%.+]] = arith.mulf [[VAR_222_]], [[VAR_cst_1_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_224_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_15_]], [[LOAD_PARAM_1_MEM_15_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_225_:%.+]] = arith.mulf [[LOAD_PARAM_2_MEM_15_]], [[LOAD_PARAM_3_MEM_15_]] : vector<4xf32>
// CHECK:             [[VAR_226_:%.+]] = arith.addf [[VAR_224_]], [[VAR_225_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_227_:%.+]] = arith.mulf [[VAR_226_]], [[VAR_cst_1_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_228_:%.+]] = arith.minnumf [[VAR_223_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_229_:%.+]] = arith.minnumf [[VAR_227_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_230_:%.+]] = arith.maxnumf [[VAR_228_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_231_:%.+]] = arith.maxnumf [[VAR_229_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:             [[VAR_232_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_230_]], [[VAR_231_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_232_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[CST_56_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<6x8x128xf16, #map>
// CHECK:         }
}
