// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// -----

  func.func @test_expand_mul_reshape_stick(%arg0: tensor<3x4x64xf32>, %arg1: tensor<24x4x64xf32>) -> tensor<24x4x64xf32> {
    %0 = "onnx.Fused"(%arg0) <{kind = "zhigh.expand-mul-stick"}> ({
    ^bb0(%arg2: tensor<3x4x64xf32>):
      %4 = onnx.Constant dense<[24, 4, 64]> : tensor<3xi64>
      %5 = onnx.Constant dense<2.000000e+00> : tensor<f32>
      %6 = onnx.Constant dense<[3, 8, 4, 64]> : tensor<4xi64>
      %7 = onnx.Constant dense<1> : tensor<1xi64>
      %8 = "onnx.Unsqueeze"(%arg2, %7) : (tensor<3x4x64xf32>, tensor<1xi64>) -> tensor<3x1x4x64xf32>
      %9 = "onnx.Expand"(%8, %6) : (tensor<3x1x4x64xf32>, tensor<4xi64>) -> tensor<3x8x4x64xf32>
      %10 = "onnx.Mul"(%9, %5) : (tensor<3x8x4x64xf32>, tensor<f32>) -> tensor<3x8x4x64xf32>
      %11 = "onnx.Reshape"(%10, %4) <{allowzero = 0 : si64}> : (tensor<3x8x4x64xf32>, tensor<3xi64>) -> tensor<24x4x64xf32>
      %12 = "zhigh.Stick"(%11) <{layout = "3DS"}> : (tensor<24x4x64xf32>) -> tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
      onnx.Yield %12 : tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    }) {expansionN = 8 : i64, mulScalar = 2.000000e+00 : f32, reshapeCollapsedCount = 2 : i64, reshapeFirstCollapsedDim = 0 : i64, stickFormat = "3DS", unsqueezedPosition = 1 : i64} : (tensor<3x4x64xf32>) -> tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    %1 = "zhigh.Stick"(%arg1) <{layout = "3DS"}> : (tensor<24x4x64xf32>) -> tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    %2 = "zhigh.Add"(%0, %1) : (tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    %3 = "zhigh.Unstick"(%2) : (tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<24x4x64xf32>
    return %3 : tensor<24x4x64xf32>
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 * 8)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 floordiv 64)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 * 8 + 1)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0) -> (d0 * 8 + 2)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0) -> (d0 * 8 + 3)>
// CHECK-DAG:   [[MAP_7_:#.+]] = affine_map<(d0) -> (d0 * 8 + 4)>
// CHECK-DAG:   [[MAP_8_:#.+]] = affine_map<(d0) -> (d0 * 8 + 5)>
// CHECK-DAG:   [[MAP_9_:#.+]] = affine_map<(d0) -> (d0 * 8 + 6)>
// CHECK-DAG:   [[MAP_10_:#.+]] = affine_map<(d0) -> (d0 * 8 + 7)>
// CHECK-DAG:   [[MAP_11_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64)>
// CHECK-DAG:   [[MAP_12_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64 + 8)>
// CHECK-DAG:   [[MAP_13_:#.+]] = affine_map<(d0) -> (d0 + 8)>
// CHECK-DAG:   [[MAP_14_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64 + 16)>
// CHECK-DAG:   [[MAP_15_:#.+]] = affine_map<(d0) -> (d0 + 16)>
// CHECK-DAG:   [[MAP_16_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64 + 24)>
// CHECK-DAG:   [[MAP_17_:#.+]] = affine_map<(d0) -> (d0 + 24)>
// CHECK-LABEL:  func.func @test_expand_mul_reshape_stick
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x4x64xf32>, [[PARAM_1_:%.+]]: memref<24x4x64xf32>) -> memref<24x4x64xf32> {
// CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : i64
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : i64
// CHECK-DAG:       [[CST_24_:%.+]] = arith.constant 24 : i64
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<-8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[VAR_cst_1_:%.+]] = arith.constant dense<2.000000e+00> : vector<4xf32>
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_4_1_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<24x4x64xf16, #map>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<24x4x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[VAR_reinterpret_cast_2_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<24x4x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[VAR_reinterpret_cast_3_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<24x4x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[VAR_reinterpret_cast_4_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<24x4x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[VAR_reinterpret_cast_5_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<24x4x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[VAR_reinterpret_cast_6_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<24x4x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[VAR_reinterpret_cast_7_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<24x4x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[VAR_reinterpret_cast_8_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<24x4x64xf16, #map> to memref<2x64xf16>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 1){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]]#2)
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#0)
// CHECK:             [[VAR_4_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_3_]], [[VAR_1_]]#1, [[VAR_2_]]{{.}} : memref<24x4x64xf16, #map>
// CHECK-DAG:         [[VAR_5_:%.+]] = affine.apply [[MAP_3_]]([[VAR_4_]])
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply [[MAP_4_]]([[VAR_1_]]#0)
// CHECK:             [[VAR_7_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_6_]], [[VAR_1_]]#1, [[VAR_2_]]{{.}} : memref<24x4x64xf16, #map>
// CHECK-DAG:         [[VAR_8_:%.+]] = affine.apply [[MAP_3_]]([[VAR_7_]])
// CHECK-DAG:         [[VAR_9_:%.+]] = affine.apply [[MAP_5_]]([[VAR_1_]]#0)
// CHECK:             [[VAR_10_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_9_]], [[VAR_1_]]#1, [[VAR_2_]]{{.}} : memref<24x4x64xf16, #map>
// CHECK-DAG:         [[VAR_11_:%.+]] = affine.apply [[MAP_3_]]([[VAR_10_]])
// CHECK-DAG:         [[VAR_12_:%.+]] = affine.apply [[MAP_6_]]([[VAR_1_]]#0)
// CHECK:             [[VAR_13_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_12_]], [[VAR_1_]]#1, [[VAR_2_]]{{.}} : memref<24x4x64xf16, #map>
// CHECK-DAG:         [[VAR_14_:%.+]] = affine.apply [[MAP_3_]]([[VAR_13_]])
// CHECK-DAG:         [[VAR_15_:%.+]] = affine.apply [[MAP_7_]]([[VAR_1_]]#0)
// CHECK:             [[VAR_16_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_15_]], [[VAR_1_]]#1, [[VAR_2_]]{{.}} : memref<24x4x64xf16, #map>
// CHECK-DAG:         [[VAR_17_:%.+]] = affine.apply [[MAP_3_]]([[VAR_16_]])
// CHECK-DAG:         [[VAR_18_:%.+]] = affine.apply [[MAP_8_]]([[VAR_1_]]#0)
// CHECK:             [[VAR_19_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_18_]], [[VAR_1_]]#1, [[VAR_2_]]{{.}} : memref<24x4x64xf16, #map>
// CHECK-DAG:         [[VAR_20_:%.+]] = affine.apply [[MAP_3_]]([[VAR_19_]])
// CHECK-DAG:         [[VAR_21_:%.+]] = affine.apply [[MAP_9_]]([[VAR_1_]]#0)
// CHECK:             [[VAR_22_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_21_]], [[VAR_1_]]#1, [[VAR_2_]]{{.}} : memref<24x4x64xf16, #map>
// CHECK-DAG:         [[VAR_23_:%.+]] = affine.apply [[MAP_3_]]([[VAR_22_]])
// CHECK-DAG:         [[VAR_24_:%.+]] = affine.apply [[MAP_10_]]([[VAR_1_]]#0)
// CHECK:             [[VAR_25_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_24_]], [[VAR_1_]]#1, [[VAR_2_]]{{.}} : memref<24x4x64xf16, #map>
// CHECK-DAG:         [[VAR_26_:%.+]] = affine.apply [[MAP_3_]]([[VAR_25_]])
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_1_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_1_]] -> [[I_3_:%.+]] = 0 to 64){
// CHECK:               [[VAR_28_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:               [[VAR_29_:%.+]] = affine.apply [[MAP_11_]]([[VAR_28_]], [[VAR_1_]]#2)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_29_]]{{.}} : memref<3x4x64xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_31_:%.+]] = arith.addi [[VAR_29_]], [[CST_4_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_31_]]{{.}} : memref<3x4x64xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_33_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_]], [[VAR_cst_1_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_34_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_1_]], [[VAR_cst_1_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_35_:%.+]] = arith.minnumf [[VAR_33_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_36_:%.+]] = arith.minnumf [[VAR_34_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_37_:%.+]] = arith.maxnumf [[VAR_35_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:               [[VAR_38_:%.+]] = arith.maxnumf [[VAR_36_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:               [[VAR_39_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_37_]], [[VAR_38_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:               vector.store [[VAR_39_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_5_]], [[VAR_28_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               vector.store [[VAR_39_]], [[VAR_reinterpret_cast_2_]]{{.}}[[VAR_8_]], [[VAR_28_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               vector.store [[VAR_39_]], [[VAR_reinterpret_cast_3_]]{{.}}[[VAR_11_]], [[VAR_28_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               vector.store [[VAR_39_]], [[VAR_reinterpret_cast_4_]]{{.}}[[VAR_14_]], [[VAR_28_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               vector.store [[VAR_39_]], [[VAR_reinterpret_cast_5_]]{{.}}[[VAR_17_]], [[VAR_28_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               vector.store [[VAR_39_]], [[VAR_reinterpret_cast_6_]]{{.}}[[VAR_20_]], [[VAR_28_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               vector.store [[VAR_39_]], [[VAR_reinterpret_cast_7_]]{{.}}[[VAR_23_]], [[VAR_28_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               vector.store [[VAR_39_]], [[VAR_reinterpret_cast_8_]]{{.}}[[VAR_26_]], [[VAR_28_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_40_:%.+]] = affine.apply [[MAP_12_]]([[VAR_28_]], [[VAR_1_]]#2)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_2_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_40_]]{{.}} : memref<3x4x64xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_42_:%.+]] = arith.addi [[VAR_40_]], [[CST_4_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_3_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_42_]]{{.}} : memref<3x4x64xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_44_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_2_]], [[VAR_cst_1_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_45_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_3_]], [[VAR_cst_1_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_46_:%.+]] = arith.minnumf [[VAR_44_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_47_:%.+]] = arith.minnumf [[VAR_45_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_48_:%.+]] = arith.maxnumf [[VAR_46_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:               [[VAR_49_:%.+]] = arith.maxnumf [[VAR_47_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_50_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_48_]], [[VAR_49_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:           [[VAR_51_:%.+]] = affine.apply [[MAP_13_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_50_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_5_]], [[VAR_51_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_52_:%.+]] = affine.apply [[MAP_13_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_50_]], [[VAR_reinterpret_cast_2_]]{{.}}[[VAR_8_]], [[VAR_52_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_53_:%.+]] = affine.apply [[MAP_13_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_50_]], [[VAR_reinterpret_cast_3_]]{{.}}[[VAR_11_]], [[VAR_53_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_54_:%.+]] = affine.apply [[MAP_13_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_50_]], [[VAR_reinterpret_cast_4_]]{{.}}[[VAR_14_]], [[VAR_54_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_55_:%.+]] = affine.apply [[MAP_13_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_50_]], [[VAR_reinterpret_cast_5_]]{{.}}[[VAR_17_]], [[VAR_55_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_56_:%.+]] = affine.apply [[MAP_13_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_50_]], [[VAR_reinterpret_cast_6_]]{{.}}[[VAR_20_]], [[VAR_56_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_57_:%.+]] = affine.apply [[MAP_13_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_50_]], [[VAR_reinterpret_cast_7_]]{{.}}[[VAR_23_]], [[VAR_57_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_58_:%.+]] = affine.apply [[MAP_13_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_50_]], [[VAR_reinterpret_cast_8_]]{{.}}[[VAR_26_]], [[VAR_58_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_59_:%.+]] = affine.apply [[MAP_14_]]([[VAR_28_]], [[VAR_1_]]#2)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_4_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_59_]]{{.}} : memref<3x4x64xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_61_:%.+]] = arith.addi [[VAR_59_]], [[CST_4_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_5_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_61_]]{{.}} : memref<3x4x64xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_63_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_4_]], [[VAR_cst_1_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_64_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_5_]], [[VAR_cst_1_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_65_:%.+]] = arith.minnumf [[VAR_63_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_66_:%.+]] = arith.minnumf [[VAR_64_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_67_:%.+]] = arith.maxnumf [[VAR_65_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:               [[VAR_68_:%.+]] = arith.maxnumf [[VAR_66_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_69_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_67_]], [[VAR_68_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:           [[VAR_70_:%.+]] = affine.apply [[MAP_15_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_69_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_5_]], [[VAR_70_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_71_:%.+]] = affine.apply [[MAP_15_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_69_]], [[VAR_reinterpret_cast_2_]]{{.}}[[VAR_8_]], [[VAR_71_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_72_:%.+]] = affine.apply [[MAP_15_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_69_]], [[VAR_reinterpret_cast_3_]]{{.}}[[VAR_11_]], [[VAR_72_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_73_:%.+]] = affine.apply [[MAP_15_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_69_]], [[VAR_reinterpret_cast_4_]]{{.}}[[VAR_14_]], [[VAR_73_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_74_:%.+]] = affine.apply [[MAP_15_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_69_]], [[VAR_reinterpret_cast_5_]]{{.}}[[VAR_17_]], [[VAR_74_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_75_:%.+]] = affine.apply [[MAP_15_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_69_]], [[VAR_reinterpret_cast_6_]]{{.}}[[VAR_20_]], [[VAR_75_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_76_:%.+]] = affine.apply [[MAP_15_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_69_]], [[VAR_reinterpret_cast_7_]]{{.}}[[VAR_23_]], [[VAR_76_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_77_:%.+]] = affine.apply [[MAP_15_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_69_]], [[VAR_reinterpret_cast_8_]]{{.}}[[VAR_26_]], [[VAR_77_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_78_:%.+]] = affine.apply [[MAP_16_]]([[VAR_28_]], [[VAR_1_]]#2)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_6_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_78_]]{{.}} : memref<3x4x64xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_80_:%.+]] = arith.addi [[VAR_78_]], [[CST_4_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_7_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_80_]]{{.}} : memref<3x4x64xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_82_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_6_]], [[VAR_cst_1_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_83_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_7_]], [[VAR_cst_1_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_84_:%.+]] = arith.minnumf [[VAR_82_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_85_:%.+]] = arith.minnumf [[VAR_83_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_86_:%.+]] = arith.maxnumf [[VAR_84_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:               [[VAR_87_:%.+]] = arith.maxnumf [[VAR_85_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_88_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_86_]], [[VAR_87_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:           [[VAR_89_:%.+]] = affine.apply [[MAP_17_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_88_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_5_]], [[VAR_89_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_90_:%.+]] = affine.apply [[MAP_17_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_88_]], [[VAR_reinterpret_cast_2_]]{{.}}[[VAR_8_]], [[VAR_90_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_91_:%.+]] = affine.apply [[MAP_17_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_88_]], [[VAR_reinterpret_cast_3_]]{{.}}[[VAR_11_]], [[VAR_91_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_92_:%.+]] = affine.apply [[MAP_17_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_88_]], [[VAR_reinterpret_cast_4_]]{{.}}[[VAR_14_]], [[VAR_92_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_93_:%.+]] = affine.apply [[MAP_17_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_88_]], [[VAR_reinterpret_cast_5_]]{{.}}[[VAR_17_]], [[VAR_93_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_94_:%.+]] = affine.apply [[MAP_17_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_88_]], [[VAR_reinterpret_cast_6_]]{{.}}[[VAR_20_]], [[VAR_94_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_95_:%.+]] = affine.apply [[MAP_17_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_88_]], [[VAR_reinterpret_cast_7_]]{{.}}[[VAR_23_]], [[VAR_95_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_96_:%.+]] = affine.apply [[MAP_17_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_88_]], [[VAR_reinterpret_cast_8_]]{{.}}[[VAR_26_]], [[VAR_96_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             }
// CHECK:           }
// CHECK:           [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<24x4x64xf16, #map>
// CHECK:           "zlow.stick"([[PARAM_1_]], [[RES_1_]]) <{layout = "3DS"}> : (memref<24x4x64xf32>, memref<24x4x64xf16, #map>) -> ()
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<24x4x64xf16, #map>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<3xi64>
// CHECK:           krnl.store [[CST_24_]], [[RES_3_]]{{.}}[[CST_0_]]{{.}} : memref<3xi64>
// CHECK:           krnl.store [[CST_4_]], [[RES_3_]]{{.}}[[CST_1_]]{{.}} : memref<3xi64>
// CHECK:           krnl.store [[CST_64_]], [[RES_3_]]{{.}}[[CST_2_]]{{.}} : memref<3xi64>
// CHECK:           "zlow.add"([[RES_]], [[RES_1_]], [[RES_3_]], [[RES_2_]]) <{layout = "3DS"}> : (memref<24x4x64xf16, #map>, memref<24x4x64xf16, #map>, memref<3xi64>, memref<24x4x64xf16, #map>) -> ()
// CHECK:           [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<24x4x64xf32>
// CHECK:           "zlow.unstick"([[RES_2_]], [[RES_4_]]) <{layout = "3DS"}> : (memref<24x4x64xf16, #map>, memref<24x4x64xf32>) -> ()
// CHECK:           return [[RES_4_]] : memref<24x4x64xf32>
// CHECK:         }

}

// -----

// Same pattern, but the source chain has no Mul op (mulScalar = 1.0, the
// neutral default): the lowering must skip the multiply and go straight
// from the loaded F32 values to the dlf16 conversion.

  func.func @test_expand_reshape_stick_no_mul(%arg0: tensor<3x4x64xf32>, %arg1: tensor<24x4x64xf32>) -> tensor<24x4x64xf32> {
    %0 = "onnx.Fused"(%arg0) <{kind = "zhigh.expand-mul-stick"}> ({
    ^bb0(%arg2: tensor<3x4x64xf32>):
      %4 = onnx.Constant dense<[24, 4, 64]> : tensor<3xi64>
      %6 = onnx.Constant dense<[3, 8, 4, 64]> : tensor<4xi64>
      %7 = onnx.Constant dense<1> : tensor<1xi64>
      %8 = "onnx.Unsqueeze"(%arg2, %7) : (tensor<3x4x64xf32>, tensor<1xi64>) -> tensor<3x1x4x64xf32>
      %9 = "onnx.Expand"(%8, %6) : (tensor<3x1x4x64xf32>, tensor<4xi64>) -> tensor<3x8x4x64xf32>
      %11 = "onnx.Reshape"(%9, %4) <{allowzero = 0 : si64}> : (tensor<3x8x4x64xf32>, tensor<3xi64>) -> tensor<24x4x64xf32>
      %12 = "zhigh.Stick"(%11) <{layout = "3DS"}> : (tensor<24x4x64xf32>) -> tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
      onnx.Yield %12 : tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    }) {expansionN = 8 : i64, mulScalar = 1.000000e+00 : f32, reshapeCollapsedCount = 2 : i64, reshapeFirstCollapsedDim = 0 : i64, stickFormat = "3DS", unsqueezedPosition = 1 : i64} : (tensor<3x4x64xf32>) -> tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    %1 = "zhigh.Stick"(%arg1) <{layout = "3DS"}> : (tensor<24x4x64xf32>) -> tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    %2 = "zhigh.Add"(%0, %1) : (tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    %3 = "zhigh.Unstick"(%2) : (tensor<24x4x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<24x4x64xf32>
    return %3 : tensor<24x4x64xf32>
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 * 8)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 floordiv 64)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 * 8 + 1)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0) -> (d0 * 8 + 2)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0) -> (d0 * 8 + 3)>
// CHECK-DAG:   [[MAP_7_:#.+]] = affine_map<(d0) -> (d0 * 8 + 4)>
// CHECK-DAG:   [[MAP_8_:#.+]] = affine_map<(d0) -> (d0 * 8 + 5)>
// CHECK-DAG:   [[MAP_9_:#.+]] = affine_map<(d0) -> (d0 * 8 + 6)>
// CHECK-DAG:   [[MAP_10_:#.+]] = affine_map<(d0) -> (d0 * 8 + 7)>
// CHECK-DAG:   [[MAP_11_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64)>
// CHECK-DAG:   [[MAP_12_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64 + 8)>
// CHECK-DAG:   [[MAP_13_:#.+]] = affine_map<(d0) -> (d0 + 8)>
// CHECK-DAG:   [[MAP_14_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64 + 16)>
// CHECK-DAG:   [[MAP_15_:#.+]] = affine_map<(d0) -> (d0 + 16)>
// CHECK-DAG:   [[MAP_16_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64 + 24)>
// CHECK-DAG:   [[MAP_17_:#.+]] = affine_map<(d0) -> (d0 + 24)>
// CHECK-LABEL:  func.func @test_expand_reshape_stick_no_mul
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x4x64xf32>, [[PARAM_1_:%.+]]: memref<24x4x64xf32>) -> memref<24x4x64xf32> {
// CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : i64
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : i64
// CHECK-DAG:       [[CST_24_:%.+]] = arith.constant 24 : i64
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<-8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_4_1_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<24x4x64xf16, #map>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<24x4x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<24x4x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[VAR_reinterpret_cast_2_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<24x4x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[VAR_reinterpret_cast_3_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<24x4x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[VAR_reinterpret_cast_4_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<24x4x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[VAR_reinterpret_cast_5_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<24x4x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[VAR_reinterpret_cast_6_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<24x4x64xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[VAR_reinterpret_cast_7_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<24x4x64xf16, #map> to memref<2x64xf16>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 1){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]]#2)
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#0)
// CHECK:             [[VAR_4_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_3_]], [[VAR_1_]]#1, [[VAR_2_]]{{.}} : memref<24x4x64xf16, #map>
// CHECK-DAG:         [[VAR_5_:%.+]] = affine.apply [[MAP_3_]]([[VAR_4_]])
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply [[MAP_4_]]([[VAR_1_]]#0)
// CHECK:             [[VAR_7_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_6_]], [[VAR_1_]]#1, [[VAR_2_]]{{.}} : memref<24x4x64xf16, #map>
// CHECK-DAG:         [[VAR_8_:%.+]] = affine.apply [[MAP_3_]]([[VAR_7_]])
// CHECK-DAG:         [[VAR_9_:%.+]] = affine.apply [[MAP_5_]]([[VAR_1_]]#0)
// CHECK:             [[VAR_10_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_9_]], [[VAR_1_]]#1, [[VAR_2_]]{{.}} : memref<24x4x64xf16, #map>
// CHECK-DAG:         [[VAR_11_:%.+]] = affine.apply [[MAP_3_]]([[VAR_10_]])
// CHECK-DAG:         [[VAR_12_:%.+]] = affine.apply [[MAP_6_]]([[VAR_1_]]#0)
// CHECK:             [[VAR_13_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_12_]], [[VAR_1_]]#1, [[VAR_2_]]{{.}} : memref<24x4x64xf16, #map>
// CHECK-DAG:         [[VAR_14_:%.+]] = affine.apply [[MAP_3_]]([[VAR_13_]])
// CHECK-DAG:         [[VAR_15_:%.+]] = affine.apply [[MAP_7_]]([[VAR_1_]]#0)
// CHECK:             [[VAR_16_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_15_]], [[VAR_1_]]#1, [[VAR_2_]]{{.}} : memref<24x4x64xf16, #map>
// CHECK-DAG:         [[VAR_17_:%.+]] = affine.apply [[MAP_3_]]([[VAR_16_]])
// CHECK-DAG:         [[VAR_18_:%.+]] = affine.apply [[MAP_8_]]([[VAR_1_]]#0)
// CHECK:             [[VAR_19_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_18_]], [[VAR_1_]]#1, [[VAR_2_]]{{.}} : memref<24x4x64xf16, #map>
// CHECK-DAG:         [[VAR_20_:%.+]] = affine.apply [[MAP_3_]]([[VAR_19_]])
// CHECK-DAG:         [[VAR_21_:%.+]] = affine.apply [[MAP_9_]]([[VAR_1_]]#0)
// CHECK:             [[VAR_22_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_21_]], [[VAR_1_]]#1, [[VAR_2_]]{{.}} : memref<24x4x64xf16, #map>
// CHECK-DAG:         [[VAR_23_:%.+]] = affine.apply [[MAP_3_]]([[VAR_22_]])
// CHECK-DAG:         [[VAR_24_:%.+]] = affine.apply [[MAP_10_]]([[VAR_1_]]#0)
// CHECK:             [[VAR_25_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_24_]], [[VAR_1_]]#1, [[VAR_2_]]{{.}} : memref<24x4x64xf16, #map>
// CHECK-DAG:         [[VAR_26_:%.+]] = affine.apply [[MAP_3_]]([[VAR_25_]])
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_1_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_1_]] -> [[I_3_:%.+]] = 0 to 64){
// CHECK:               [[VAR_28_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:               [[VAR_29_:%.+]] = affine.apply [[MAP_11_]]([[VAR_28_]], [[VAR_1_]]#2)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_29_]]{{.}} : memref<3x4x64xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_31_:%.+]] = arith.addi [[VAR_29_]], [[CST_4_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_31_]]{{.}} : memref<3x4x64xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_33_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_34_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_35_:%.+]] = arith.maxnumf [[VAR_33_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:               [[VAR_36_:%.+]] = arith.maxnumf [[VAR_34_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:               [[VAR_37_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_35_]], [[VAR_36_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:               vector.store [[VAR_37_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_5_]], [[VAR_28_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               vector.store [[VAR_37_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_8_]], [[VAR_28_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               vector.store [[VAR_37_]], [[VAR_reinterpret_cast_2_]]{{.}}[[VAR_11_]], [[VAR_28_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               vector.store [[VAR_37_]], [[VAR_reinterpret_cast_3_]]{{.}}[[VAR_14_]], [[VAR_28_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               vector.store [[VAR_37_]], [[VAR_reinterpret_cast_4_]]{{.}}[[VAR_17_]], [[VAR_28_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               vector.store [[VAR_37_]], [[VAR_reinterpret_cast_5_]]{{.}}[[VAR_20_]], [[VAR_28_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               vector.store [[VAR_37_]], [[VAR_reinterpret_cast_6_]]{{.}}[[VAR_23_]], [[VAR_28_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               vector.store [[VAR_37_]], [[VAR_reinterpret_cast_7_]]{{.}}[[VAR_26_]], [[VAR_28_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_38_:%.+]] = affine.apply [[MAP_12_]]([[VAR_28_]], [[VAR_1_]]#2)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_2_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_38_]]{{.}} : memref<3x4x64xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_40_:%.+]] = arith.addi [[VAR_38_]], [[CST_4_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_3_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_40_]]{{.}} : memref<3x4x64xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_42_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_2_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_43_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_3_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_44_:%.+]] = arith.maxnumf [[VAR_42_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:               [[VAR_45_:%.+]] = arith.maxnumf [[VAR_43_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_46_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_44_]], [[VAR_45_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:           [[VAR_47_:%.+]] = affine.apply [[MAP_13_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_46_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_5_]], [[VAR_47_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_48_:%.+]] = affine.apply [[MAP_13_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_46_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_8_]], [[VAR_48_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_49_:%.+]] = affine.apply [[MAP_13_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_46_]], [[VAR_reinterpret_cast_2_]]{{.}}[[VAR_11_]], [[VAR_49_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_50_:%.+]] = affine.apply [[MAP_13_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_46_]], [[VAR_reinterpret_cast_3_]]{{.}}[[VAR_14_]], [[VAR_50_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_51_:%.+]] = affine.apply [[MAP_13_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_46_]], [[VAR_reinterpret_cast_4_]]{{.}}[[VAR_17_]], [[VAR_51_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_52_:%.+]] = affine.apply [[MAP_13_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_46_]], [[VAR_reinterpret_cast_5_]]{{.}}[[VAR_20_]], [[VAR_52_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_53_:%.+]] = affine.apply [[MAP_13_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_46_]], [[VAR_reinterpret_cast_6_]]{{.}}[[VAR_23_]], [[VAR_53_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_54_:%.+]] = affine.apply [[MAP_13_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_46_]], [[VAR_reinterpret_cast_7_]]{{.}}[[VAR_26_]], [[VAR_54_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_55_:%.+]] = affine.apply [[MAP_14_]]([[VAR_28_]], [[VAR_1_]]#2)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_4_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_55_]]{{.}} : memref<3x4x64xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_57_:%.+]] = arith.addi [[VAR_55_]], [[CST_4_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_5_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_57_]]{{.}} : memref<3x4x64xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_59_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_4_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_60_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_5_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_61_:%.+]] = arith.maxnumf [[VAR_59_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:               [[VAR_62_:%.+]] = arith.maxnumf [[VAR_60_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_63_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_61_]], [[VAR_62_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:           [[VAR_64_:%.+]] = affine.apply [[MAP_15_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_63_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_5_]], [[VAR_64_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_65_:%.+]] = affine.apply [[MAP_15_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_63_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_8_]], [[VAR_65_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_66_:%.+]] = affine.apply [[MAP_15_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_63_]], [[VAR_reinterpret_cast_2_]]{{.}}[[VAR_11_]], [[VAR_66_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_67_:%.+]] = affine.apply [[MAP_15_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_63_]], [[VAR_reinterpret_cast_3_]]{{.}}[[VAR_14_]], [[VAR_67_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_68_:%.+]] = affine.apply [[MAP_15_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_63_]], [[VAR_reinterpret_cast_4_]]{{.}}[[VAR_17_]], [[VAR_68_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_69_:%.+]] = affine.apply [[MAP_15_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_63_]], [[VAR_reinterpret_cast_5_]]{{.}}[[VAR_20_]], [[VAR_69_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_70_:%.+]] = affine.apply [[MAP_15_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_63_]], [[VAR_reinterpret_cast_6_]]{{.}}[[VAR_23_]], [[VAR_70_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_71_:%.+]] = affine.apply [[MAP_15_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_63_]], [[VAR_reinterpret_cast_7_]]{{.}}[[VAR_26_]], [[VAR_71_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_72_:%.+]] = affine.apply [[MAP_16_]]([[VAR_28_]], [[VAR_1_]]#2)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_6_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_72_]]{{.}} : memref<3x4x64xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_74_:%.+]] = arith.addi [[VAR_72_]], [[CST_4_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_7_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_74_]]{{.}} : memref<3x4x64xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_76_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_6_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_77_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_7_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_78_:%.+]] = arith.maxnumf [[VAR_76_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:               [[VAR_79_:%.+]] = arith.maxnumf [[VAR_77_]], [[VAR_cst_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_80_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_78_]], [[VAR_79_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK-DAG:           [[VAR_81_:%.+]] = affine.apply [[MAP_17_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_80_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_5_]], [[VAR_81_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_82_:%.+]] = affine.apply [[MAP_17_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_80_]], [[VAR_reinterpret_cast_1_]]{{.}}[[VAR_8_]], [[VAR_82_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_83_:%.+]] = affine.apply [[MAP_17_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_80_]], [[VAR_reinterpret_cast_2_]]{{.}}[[VAR_11_]], [[VAR_83_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_84_:%.+]] = affine.apply [[MAP_17_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_80_]], [[VAR_reinterpret_cast_3_]]{{.}}[[VAR_14_]], [[VAR_84_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_85_:%.+]] = affine.apply [[MAP_17_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_80_]], [[VAR_reinterpret_cast_4_]]{{.}}[[VAR_17_]], [[VAR_85_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_86_:%.+]] = affine.apply [[MAP_17_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_80_]], [[VAR_reinterpret_cast_5_]]{{.}}[[VAR_20_]], [[VAR_86_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_87_:%.+]] = affine.apply [[MAP_17_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_80_]], [[VAR_reinterpret_cast_6_]]{{.}}[[VAR_23_]], [[VAR_87_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_88_:%.+]] = affine.apply [[MAP_17_]]([[VAR_28_]])
// CHECK:               vector.store [[VAR_80_]], [[VAR_reinterpret_cast_7_]]{{.}}[[VAR_26_]], [[VAR_88_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             }
// CHECK:           }
// CHECK:           [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<24x4x64xf16, #map>
// CHECK:           "zlow.stick"([[PARAM_1_]], [[RES_1_]]) <{layout = "3DS"}> : (memref<24x4x64xf32>, memref<24x4x64xf16, #map>) -> ()
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<24x4x64xf16, #map>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<3xi64>
// CHECK:           krnl.store [[CST_24_]], [[RES_3_]]{{.}}[[CST_0_]]{{.}} : memref<3xi64>
// CHECK:           krnl.store [[CST_4_]], [[RES_3_]]{{.}}[[CST_1_]]{{.}} : memref<3xi64>
// CHECK:           krnl.store [[CST_64_]], [[RES_3_]]{{.}}[[CST_2_]]{{.}} : memref<3xi64>
// CHECK:           "zlow.add"([[RES_]], [[RES_1_]], [[RES_3_]], [[RES_2_]]) <{layout = "3DS"}> : (memref<24x4x64xf16, #map>, memref<24x4x64xf16, #map>, memref<3xi64>, memref<24x4x64xf16, #map>) -> ()
// CHECK:           [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<24x4x64xf32>
// CHECK:           "zlow.unstick"([[RES_2_]], [[RES_4_]]) <{layout = "3DS"}> : (memref<24x4x64xf16, #map>, memref<24x4x64xf32>) -> ()
// CHECK:           return [[RES_4_]] : memref<24x4x64xf32>
// CHECK:         }

}

