// RUN: onnx-mlir-opt --march=z17 --maccel=NNPA --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// -----

// Test doing layer norm computation directly on zTensor.

func.func @layer_norm_after(%arg0: tensor<256x256xf32>, %arg1: tensor<256xf32>) -> tensor<256x256xf32> attributes {input_names = ["X", "S"], output_names = ["LN"]} {
  %0 = "onnx.NoValue"() {onnx_node_name = "onnx.NoValue_0", value} : () -> none
  %Y, %Mean, %InvStdDev = "onnx.LayerNormalization"(%arg0, %arg1, %0) {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, onnx_node_name = "onnx.LayerNormalization_1", stash_type = 1 : si64} : (tensor<256x256xf32>, tensor<256xf32>, none) -> (tensor<256x256xf16, #zhigh.layout<{dataLayout = "2D"}>>, none, none)
  %1 = "zhigh.MatMul"(%Y, %Y, %0) {transposeA = 0 : si64, transposeB = 0 : si64} : (tensor<256x256xf16, #zhigh.layout<{dataLayout = "2D"}>>, tensor<256x256xf16, #zhigh.layout<{dataLayout = "2D"}>>, none) -> tensor<256x256xf16, #zhigh.layout<{dataLayout = "2D"}>>
  %2 = "zhigh.Unstick"(%1) : (tensor<256x256xf16, #zhigh.layout<{dataLayout = "2D"}>>) -> tensor<256x256xf32>
  return %2 : tensor<256x256xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (0, d1 floordiv 64, 0, d0 floordiv 32, d0 mod 32, d1 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 + 2)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 + 3)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0) -> (d0 floordiv 64)>
// CHECK-LABEL:  func.func @layer_norm_after
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<256x256xf32>, [[PARAM_1_:%.+]]: memref<256xf32>) -> memref<256x256xf32> attributes {input_names = ["X", "S"], output_names = ["LN"]} {
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : i64
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<-8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<8.57315738E+9> : vector<4xf32>
// CHECK-DAG:       [[CST_2_dot_560000_:%.+]] = arith.constant 2.560000e+02 : f32
// CHECK-DAG:       [[VAR_cst_2_:%.+]] = arith.constant dense<0.000000e+00> : vector<8xf32>
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_9_dot_99999974_:%.+]] = arith.constant 9.99999974E-6 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<256x256xf16, #map>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<256x256xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.permute([[BLOCK_TILE__0_]], [[BLOCK_IN__0_]]) [0, 1] : !krnl.loop, !krnl.loop
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<4x8xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<4x8xf32>
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 256){
// CHECK:             [[VAR_2_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:             vector.store [[VAR_cst_2_]], [[RES_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x8xf32>, vector<8xf32>
// CHECK:             [[VAR_subview_:%.+]] = memref.subview [[RES_1_]][0, 0] [1, 8] [1, 1] : memref<4x8xf32> to memref<1x8xf32, strided<[8, 1]>>
// CHECK:             vector.store [[VAR_cst_2_]], [[RES_2_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x8xf32>, vector<8xf32>
// CHECK:             [[VAR_subview_10_:%.+]] = memref.subview [[RES_2_]][0, 0] [1, 8] [1, 1] : memref<4x8xf32> to memref<1x8xf32, strided<[8, 1]>>
// CHECK:             vector.store [[VAR_cst_2_]], [[RES_1_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x8xf32>, vector<8xf32>
// CHECK:             [[VAR_subview_11_:%.+]] = memref.subview [[RES_1_]][1, 0] [1, 8] [1, 1] : memref<4x8xf32> to memref<1x8xf32, strided<[8, 1], offset: 8>>
// CHECK:             vector.store [[VAR_cst_2_]], [[RES_2_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x8xf32>, vector<8xf32>
// CHECK:             [[VAR_subview_12_:%.+]] = memref.subview [[RES_2_]][1, 0] [1, 8] [1, 1] : memref<4x8xf32> to memref<1x8xf32, strided<[8, 1], offset: 8>>
// CHECK:             vector.store [[VAR_cst_2_]], [[RES_1_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x8xf32>, vector<8xf32>
// CHECK:             [[VAR_subview_13_:%.+]] = memref.subview [[RES_1_]][2, 0] [1, 8] [1, 1] : memref<4x8xf32> to memref<1x8xf32, strided<[8, 1], offset: 16>>
// CHECK:             vector.store [[VAR_cst_2_]], [[RES_2_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x8xf32>, vector<8xf32>
// CHECK:             [[VAR_subview_14_:%.+]] = memref.subview [[RES_2_]][2, 0] [1, 8] [1, 1] : memref<4x8xf32> to memref<1x8xf32, strided<[8, 1], offset: 16>>
// CHECK:             vector.store [[VAR_cst_2_]], [[RES_1_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x8xf32>, vector<8xf32>
// CHECK:             [[VAR_subview_15_:%.+]] = memref.subview [[RES_1_]][3, 0] [1, 8] [1, 1] : memref<4x8xf32> to memref<1x8xf32, strided<[8, 1], offset: 24>>
// CHECK:             vector.store [[VAR_cst_2_]], [[RES_2_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x8xf32>, vector<8xf32>
// CHECK:             [[VAR_subview_16_:%.+]] = memref.subview [[RES_2_]][3, 0] [1, 8] [1, 1] : memref<4x8xf32> to memref<1x8xf32, strided<[8, 1], offset: 24>>
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 256 step 64 {
// CHECK-DAG:           [[VAR_47_:%.+]] = affine.apply [[MAP_1_]]([[VAR_2_]])
// CHECK-DAG:           [[VAR_48_:%.+]] = affine.apply [[MAP_2_]]([[VAR_2_]])
// CHECK-DAG:           [[VAR_49_:%.+]] = affine.apply [[MAP_3_]]([[VAR_2_]])
// CHECK:               affine.for [[I_2_:%.+]] = 0 to 64 step 8 {
// CHECK:                 [[VAR_50_:%.+]] = affine.apply [[MAP_4_]]([[I_1_]], [[I_2_]])
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_2_]], [[VAR_50_]]{{.}} : memref<256x256xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_52_:%.+]] = arith.addi [[VAR_50_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_1_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_2_]], [[VAR_52_]]{{.}} : memref<256x256xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_VAR_subview_MEM_:%.+]] = vector.load [[VAR_subview_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<1x8xf32, strided<[8, 1]>>, vector<4xf32>
// CHECK-DAG:             [[LOAD_VAR_subview_MEM_1_:%.+]] = vector.load [[VAR_subview_]]{{.}}[[CST_0_]], [[CST_4_]]{{.}} : memref<1x8xf32, strided<[8, 1]>>, vector<4xf32>
// CHECK-DAG:             [[LOAD_VAR_subview_10_MEM_:%.+]] = vector.load [[VAR_subview_10_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<1x8xf32, strided<[8, 1]>>, vector<4xf32>
// CHECK-DAG:             [[LOAD_VAR_subview_10_MEM_1_:%.+]] = vector.load [[VAR_subview_10_]]{{.}}[[CST_0_]], [[CST_4_]]{{.}} : memref<1x8xf32, strided<[8, 1]>>, vector<4xf32>
// CHECK-DAG:             [[VAR_58_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_0_MEM_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_59_:%.+]] = arith.addf [[LOAD_VAR_subview_MEM_]], [[LOAD_PARAM_0_MEM_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_60_:%.+]] = arith.addf [[LOAD_VAR_subview_10_MEM_]], [[VAR_58_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_61_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_62_:%.+]] = arith.addf [[LOAD_VAR_subview_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : vector<4xf32>
// CHECK:                 [[VAR_63_:%.+]] = arith.addf [[LOAD_VAR_subview_10_MEM_1_]], [[VAR_61_]] : vector<4xf32>
// CHECK:                 vector.store [[VAR_59_]], [[VAR_subview_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<1x8xf32, strided<[8, 1]>>, vector<4xf32>
// CHECK:                 vector.store [[VAR_62_]], [[VAR_subview_]]{{.}}[[CST_0_]], [[CST_4_]]{{.}} : memref<1x8xf32, strided<[8, 1]>>, vector<4xf32>
// CHECK:                 vector.store [[VAR_60_]], [[VAR_subview_10_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<1x8xf32, strided<[8, 1]>>, vector<4xf32>
// CHECK:                 vector.store [[VAR_63_]], [[VAR_subview_10_]]{{.}}[[CST_0_]], [[CST_4_]]{{.}} : memref<1x8xf32, strided<[8, 1]>>, vector<4xf32>
// CHECK:                 [[VAR_64_:%.+]] = affine.apply [[MAP_4_]]([[I_1_]], [[I_2_]])
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_2_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_47_]], [[VAR_64_]]{{.}} : memref<256x256xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_66_:%.+]] = arith.addi [[VAR_64_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_3_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_47_]], [[VAR_66_]]{{.}} : memref<256x256xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_VAR_subview_11_MEM_:%.+]] = vector.load [[VAR_subview_11_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<1x8xf32, strided<[8, 1], offset: 8>>, vector<4xf32>
// CHECK-DAG:             [[LOAD_VAR_subview_11_MEM_1_:%.+]] = vector.load [[VAR_subview_11_]]{{.}}[[CST_0_]], [[CST_4_]]{{.}} : memref<1x8xf32, strided<[8, 1], offset: 8>>, vector<4xf32>
// CHECK-DAG:             [[LOAD_VAR_subview_12_MEM_:%.+]] = vector.load [[VAR_subview_12_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<1x8xf32, strided<[8, 1], offset: 8>>, vector<4xf32>
// CHECK-DAG:             [[LOAD_VAR_subview_12_MEM_1_:%.+]] = vector.load [[VAR_subview_12_]]{{.}}[[CST_0_]], [[CST_4_]]{{.}} : memref<1x8xf32, strided<[8, 1], offset: 8>>, vector<4xf32>
// CHECK-DAG:             [[VAR_72_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_2_]], [[LOAD_PARAM_0_MEM_2_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_73_:%.+]] = arith.addf [[LOAD_VAR_subview_11_MEM_]], [[LOAD_PARAM_0_MEM_2_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_74_:%.+]] = arith.addf [[LOAD_VAR_subview_12_MEM_]], [[VAR_72_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_75_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_3_]], [[LOAD_PARAM_0_MEM_3_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_76_:%.+]] = arith.addf [[LOAD_VAR_subview_11_MEM_1_]], [[LOAD_PARAM_0_MEM_3_]] : vector<4xf32>
// CHECK:                 [[VAR_77_:%.+]] = arith.addf [[LOAD_VAR_subview_12_MEM_1_]], [[VAR_75_]] : vector<4xf32>
// CHECK:                 vector.store [[VAR_73_]], [[VAR_subview_11_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<1x8xf32, strided<[8, 1], offset: 8>>, vector<4xf32>
// CHECK:                 vector.store [[VAR_76_]], [[VAR_subview_11_]]{{.}}[[CST_0_]], [[CST_4_]]{{.}} : memref<1x8xf32, strided<[8, 1], offset: 8>>, vector<4xf32>
// CHECK:                 vector.store [[VAR_74_]], [[VAR_subview_12_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<1x8xf32, strided<[8, 1], offset: 8>>, vector<4xf32>
// CHECK:                 vector.store [[VAR_77_]], [[VAR_subview_12_]]{{.}}[[CST_0_]], [[CST_4_]]{{.}} : memref<1x8xf32, strided<[8, 1], offset: 8>>, vector<4xf32>
// CHECK:                 [[VAR_78_:%.+]] = affine.apply [[MAP_4_]]([[I_1_]], [[I_2_]])
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_4_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_48_]], [[VAR_78_]]{{.}} : memref<256x256xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_80_:%.+]] = arith.addi [[VAR_78_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_5_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_48_]], [[VAR_80_]]{{.}} : memref<256x256xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_VAR_subview_13_MEM_:%.+]] = vector.load [[VAR_subview_13_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<1x8xf32, strided<[8, 1], offset: 16>>, vector<4xf32>
// CHECK-DAG:             [[LOAD_VAR_subview_13_MEM_1_:%.+]] = vector.load [[VAR_subview_13_]]{{.}}[[CST_0_]], [[CST_4_]]{{.}} : memref<1x8xf32, strided<[8, 1], offset: 16>>, vector<4xf32>
// CHECK-DAG:             [[LOAD_VAR_subview_14_MEM_:%.+]] = vector.load [[VAR_subview_14_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<1x8xf32, strided<[8, 1], offset: 16>>, vector<4xf32>
// CHECK-DAG:             [[LOAD_VAR_subview_14_MEM_1_:%.+]] = vector.load [[VAR_subview_14_]]{{.}}[[CST_0_]], [[CST_4_]]{{.}} : memref<1x8xf32, strided<[8, 1], offset: 16>>, vector<4xf32>
// CHECK-DAG:             [[VAR_86_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_4_]], [[LOAD_PARAM_0_MEM_4_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_87_:%.+]] = arith.addf [[LOAD_VAR_subview_13_MEM_]], [[LOAD_PARAM_0_MEM_4_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_88_:%.+]] = arith.addf [[LOAD_VAR_subview_14_MEM_]], [[VAR_86_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_89_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_5_]], [[LOAD_PARAM_0_MEM_5_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_90_:%.+]] = arith.addf [[LOAD_VAR_subview_13_MEM_1_]], [[LOAD_PARAM_0_MEM_5_]] : vector<4xf32>
// CHECK:                 [[VAR_91_:%.+]] = arith.addf [[LOAD_VAR_subview_14_MEM_1_]], [[VAR_89_]] : vector<4xf32>
// CHECK:                 vector.store [[VAR_87_]], [[VAR_subview_13_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<1x8xf32, strided<[8, 1], offset: 16>>, vector<4xf32>
// CHECK:                 vector.store [[VAR_90_]], [[VAR_subview_13_]]{{.}}[[CST_0_]], [[CST_4_]]{{.}} : memref<1x8xf32, strided<[8, 1], offset: 16>>, vector<4xf32>
// CHECK:                 vector.store [[VAR_88_]], [[VAR_subview_14_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<1x8xf32, strided<[8, 1], offset: 16>>, vector<4xf32>
// CHECK:                 vector.store [[VAR_91_]], [[VAR_subview_14_]]{{.}}[[CST_0_]], [[CST_4_]]{{.}} : memref<1x8xf32, strided<[8, 1], offset: 16>>, vector<4xf32>
// CHECK:                 [[VAR_92_:%.+]] = affine.apply [[MAP_4_]]([[I_1_]], [[I_2_]])
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_6_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_49_]], [[VAR_92_]]{{.}} : memref<256x256xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_94_:%.+]] = arith.addi [[VAR_92_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_7_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_49_]], [[VAR_94_]]{{.}} : memref<256x256xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_VAR_subview_15_MEM_:%.+]] = vector.load [[VAR_subview_15_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<1x8xf32, strided<[8, 1], offset: 24>>, vector<4xf32>
// CHECK-DAG:             [[LOAD_VAR_subview_15_MEM_1_:%.+]] = vector.load [[VAR_subview_15_]]{{.}}[[CST_0_]], [[CST_4_]]{{.}} : memref<1x8xf32, strided<[8, 1], offset: 24>>, vector<4xf32>
// CHECK-DAG:             [[LOAD_VAR_subview_16_MEM_:%.+]] = vector.load [[VAR_subview_16_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<1x8xf32, strided<[8, 1], offset: 24>>, vector<4xf32>
// CHECK-DAG:             [[LOAD_VAR_subview_16_MEM_1_:%.+]] = vector.load [[VAR_subview_16_]]{{.}}[[CST_0_]], [[CST_4_]]{{.}} : memref<1x8xf32, strided<[8, 1], offset: 24>>, vector<4xf32>
// CHECK-DAG:             [[VAR_100_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_6_]], [[LOAD_PARAM_0_MEM_6_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_101_:%.+]] = arith.addf [[LOAD_VAR_subview_15_MEM_]], [[LOAD_PARAM_0_MEM_6_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_102_:%.+]] = arith.addf [[LOAD_VAR_subview_16_MEM_]], [[VAR_100_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_103_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_7_]], [[LOAD_PARAM_0_MEM_7_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_104_:%.+]] = arith.addf [[LOAD_VAR_subview_15_MEM_1_]], [[LOAD_PARAM_0_MEM_7_]] : vector<4xf32>
// CHECK:                 [[VAR_105_:%.+]] = arith.addf [[LOAD_VAR_subview_16_MEM_1_]], [[VAR_103_]] : vector<4xf32>
// CHECK:                 vector.store [[VAR_101_]], [[VAR_subview_15_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<1x8xf32, strided<[8, 1], offset: 24>>, vector<4xf32>
// CHECK:                 vector.store [[VAR_104_]], [[VAR_subview_15_]]{{.}}[[CST_0_]], [[CST_4_]]{{.}} : memref<1x8xf32, strided<[8, 1], offset: 24>>, vector<4xf32>
// CHECK:                 vector.store [[VAR_102_]], [[VAR_subview_16_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<1x8xf32, strided<[8, 1], offset: 24>>, vector<4xf32>
// CHECK:                 vector.store [[VAR_105_]], [[VAR_subview_16_]]{{.}}[[CST_0_]], [[CST_4_]]{{.}} : memref<1x8xf32, strided<[8, 1], offset: 24>>, vector<4xf32>
// CHECK:               }
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_1_MEM_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x8xf32>, vector<8xf32>
// CHECK-DAG:         [[LOAD_RES_2_MEM_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x8xf32>, vector<8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_5_:%.+]] = vector.reduction <add>, [[LOAD_RES_1_MEM_]] : vector<8xf32> into f32
// CHECK-DAG:         [[VAR_6_:%.+]] = vector.reduction <add>, [[LOAD_RES_2_MEM_]] : vector<8xf32> into f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.divf [[VAR_5_]], [[CST_2_dot_560000_]] : f32
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.divf [[VAR_6_]], [[CST_2_dot_560000_]] : f32
// CHECK:             [[VAR_9_:%.+]] = arith.mulf [[VAR_7_]], [[VAR_7_]] : f32
// CHECK:             [[VAR_10_:%.+]] = arith.subf [[VAR_8_]], [[VAR_9_]] : f32
// CHECK:             [[VAR_11_:%.+]] = arith.addf [[VAR_10_]], [[CST_9_dot_99999974_]] : f32
// CHECK:             [[VAR_12_:%.+]] = math.sqrt [[VAR_11_]] : f32
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_12_]] : f32
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x8xf32>, vector<8xf32>
// CHECK-DAG:         [[LOAD_RES_2_MEM_1_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x8xf32>, vector<8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_16_:%.+]] = vector.reduction <add>, [[LOAD_RES_1_MEM_1_]] : vector<8xf32> into f32
// CHECK-DAG:         [[VAR_17_:%.+]] = vector.reduction <add>, [[LOAD_RES_2_MEM_1_]] : vector<8xf32> into f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_18_:%.+]] = arith.divf [[VAR_16_]], [[CST_2_dot_560000_]] : f32
// CHECK-DAG:         [[VAR_19_:%.+]] = arith.divf [[VAR_17_]], [[CST_2_dot_560000_]] : f32
// CHECK:             [[VAR_20_:%.+]] = arith.mulf [[VAR_18_]], [[VAR_18_]] : f32
// CHECK:             [[VAR_21_:%.+]] = arith.subf [[VAR_19_]], [[VAR_20_]] : f32
// CHECK:             [[VAR_22_:%.+]] = arith.addf [[VAR_21_]], [[CST_9_dot_99999974_]] : f32
// CHECK:             [[VAR_23_:%.+]] = math.sqrt [[VAR_22_]] : f32
// CHECK-DAG:         [[VAR_24_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_23_]] : f32
// CHECK-DAG:         [[LOAD_RES_1_MEM_2_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x8xf32>, vector<8xf32>
// CHECK-DAG:         [[LOAD_RES_2_MEM_2_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x8xf32>, vector<8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_27_:%.+]] = vector.reduction <add>, [[LOAD_RES_1_MEM_2_]] : vector<8xf32> into f32
// CHECK-DAG:         [[VAR_28_:%.+]] = vector.reduction <add>, [[LOAD_RES_2_MEM_2_]] : vector<8xf32> into f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_29_:%.+]] = arith.divf [[VAR_27_]], [[CST_2_dot_560000_]] : f32
// CHECK-DAG:         [[VAR_30_:%.+]] = arith.divf [[VAR_28_]], [[CST_2_dot_560000_]] : f32
// CHECK:             [[VAR_31_:%.+]] = arith.mulf [[VAR_29_]], [[VAR_29_]] : f32
// CHECK:             [[VAR_32_:%.+]] = arith.subf [[VAR_30_]], [[VAR_31_]] : f32
// CHECK:             [[VAR_33_:%.+]] = arith.addf [[VAR_32_]], [[CST_9_dot_99999974_]] : f32
// CHECK:             [[VAR_34_:%.+]] = math.sqrt [[VAR_33_]] : f32
// CHECK-DAG:         [[VAR_35_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_34_]] : f32
// CHECK-DAG:         [[LOAD_RES_1_MEM_3_:%.+]] = vector.load [[RES_1_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x8xf32>, vector<8xf32>
// CHECK-DAG:         [[LOAD_RES_2_MEM_3_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x8xf32>, vector<8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_38_:%.+]] = vector.reduction <add>, [[LOAD_RES_1_MEM_3_]] : vector<8xf32> into f32
// CHECK-DAG:         [[VAR_39_:%.+]] = vector.reduction <add>, [[LOAD_RES_2_MEM_3_]] : vector<8xf32> into f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_40_:%.+]] = arith.divf [[VAR_38_]], [[CST_2_dot_560000_]] : f32
// CHECK-DAG:         [[VAR_41_:%.+]] = arith.divf [[VAR_39_]], [[CST_2_dot_560000_]] : f32
// CHECK:             [[VAR_42_:%.+]] = arith.mulf [[VAR_40_]], [[VAR_40_]] : f32
// CHECK:             [[VAR_43_:%.+]] = arith.subf [[VAR_41_]], [[VAR_42_]] : f32
// CHECK:             [[VAR_44_:%.+]] = arith.addf [[VAR_43_]], [[CST_9_dot_99999974_]] : f32
// CHECK:             [[VAR_45_:%.+]] = math.sqrt [[VAR_44_]] : f32
// CHECK:             [[VAR_46_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_45_]] : f32
// CHECK:             affine.for [[I_3_:%.+]] = 0 to 256 step 64 {
// CHECK:               [[VAR_47_1_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_2_]], [[I_3_]]{{.}} : memref<256x256xf16, #map>
// CHECK-DAG:           [[VAR_48_1_:%.+]] = affine.apply [[MAP_5_]]([[VAR_47_1_]])
// CHECK-DAG:           [[VAR_49_1_:%.+]] = affine.apply [[MAP_1_]]([[VAR_2_]])
// CHECK:               [[VAR_50_1_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_49_1_]], [[I_3_]]{{.}} : memref<256x256xf16, #map>
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_8_:%.+]] = affine.apply [[MAP_5_]]([[VAR_50_1_]])
// CHECK-DAG:           [[VAR_52_1_:%.+]] = affine.apply [[MAP_2_]]([[VAR_2_]])
// CHECK:               [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_52_1_]], [[I_3_]]{{.}} : memref<256x256xf16, #map>
// CHECK-DAG:           [[LOAD_VAR_subview_MEM_2_:%.+]] = affine.apply [[MAP_5_]]([[LOAD_PARAM_0_MEM_1_]])
// CHECK-DAG:           [[LOAD_VAR_subview_MEM_1_:%.+]] = affine.apply [[MAP_3_]]([[VAR_2_]])
// CHECK:               [[LOAD_VAR_subview_10_MEM_2_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[LOAD_VAR_subview_MEM_1_]], [[I_3_]]{{.}} : memref<256x256xf16, #map>
// CHECK:               [[LOAD_VAR_subview_10_MEM_1_:%.+]] = affine.apply [[MAP_5_]]([[LOAD_VAR_subview_10_MEM_2_]])
// CHECK:               affine.for [[I_4_:%.+]] = 0 to 64 step 8 {
// CHECK:                 [[VAR_58_1_:%.+]] = affine.apply [[MAP_4_]]([[I_3_]], [[I_4_]])
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_9_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_2_]], [[VAR_58_1_]]{{.}} : memref<256x256xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_60_1_:%.+]] = arith.addi [[VAR_58_1_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_10_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_2_]], [[VAR_60_1_]]{{.}} : memref<256x256xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_62_1_:%.+]] = affine.apply [[MAP_4_]]([[I_3_]], [[I_4_]])
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_63_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_62_1_]]{{.}} : memref<256xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_64_1_:%.+]] = arith.addi [[VAR_62_1_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_2_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_64_1_]]{{.}} : memref<256xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_66_1_:%.+]] = vector.broadcast [[VAR_7_]] : f32 to vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_3_:%.+]] = arith.subf [[LOAD_PARAM_0_MEM_9_]], [[VAR_66_1_]] : vector<4xf32>
// CHECK-DAG:             [[LOAD_VAR_subview_11_MEM_2_:%.+]] = vector.broadcast [[VAR_13_]] : f32 to vector<4xf32>
// CHECK:                 [[LOAD_VAR_subview_11_MEM_1_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_3_]], [[LOAD_VAR_subview_11_MEM_2_]] : vector<4xf32>
// CHECK-DAG:             [[LOAD_VAR_subview_12_MEM_2_:%.+]] = arith.mulf [[LOAD_VAR_subview_11_MEM_1_]], [[VAR_63_1_]] : vector<4xf32>
// CHECK-DAG:             [[LOAD_VAR_subview_12_MEM_1_:%.+]] = vector.broadcast [[VAR_7_]] : f32 to vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_72_1_:%.+]] = arith.subf [[LOAD_PARAM_0_MEM_10_]], [[LOAD_VAR_subview_12_MEM_1_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_73_1_:%.+]] = vector.broadcast [[VAR_13_]] : f32 to vector<4xf32>
// CHECK:                 [[VAR_74_1_:%.+]] = arith.mulf [[VAR_72_1_]], [[VAR_73_1_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_75_1_:%.+]] = arith.mulf [[VAR_74_1_]], [[LOAD_PARAM_0_MEM_2_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_76_1_:%.+]] = arith.minnumf [[LOAD_VAR_subview_12_MEM_2_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_77_1_:%.+]] = arith.minnumf [[VAR_75_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_78_1_:%.+]] = arith.maxnumf [[VAR_76_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[LOAD_PARAM_0_MEM_4_:%.+]] = arith.maxnumf [[VAR_77_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_80_1_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_78_1_]], [[LOAD_PARAM_0_MEM_4_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                 vector.store [[VAR_80_1_]], [[VAR_reinterpret_cast_]]{{.}}[[VAR_48_1_]], [[I_4_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[LOAD_PARAM_0_MEM_5_:%.+]] = affine.apply [[MAP_4_]]([[I_3_]], [[I_4_]])
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_11_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_49_1_]], [[LOAD_PARAM_0_MEM_5_]]{{.}} : memref<256x256xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_VAR_subview_13_MEM_1_:%.+]] = arith.addi [[LOAD_PARAM_0_MEM_5_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_12_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_49_1_]], [[LOAD_VAR_subview_13_MEM_1_]]{{.}} : memref<256x256xf32>, vector<4xf32>
// CHECK-DAG:             [[LOAD_VAR_subview_14_MEM_1_:%.+]] = affine.apply [[MAP_4_]]([[I_3_]], [[I_4_]])
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_86_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[LOAD_VAR_subview_14_MEM_1_]]{{.}} : memref<256xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_87_1_:%.+]] = arith.addi [[LOAD_VAR_subview_14_MEM_1_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_88_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_87_1_]]{{.}} : memref<256xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_89_1_:%.+]] = vector.broadcast [[VAR_18_]] : f32 to vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_90_1_:%.+]] = arith.subf [[LOAD_PARAM_0_MEM_11_]], [[VAR_89_1_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_91_1_:%.+]] = vector.broadcast [[VAR_24_]] : f32 to vector<4xf32>
// CHECK:                 [[VAR_92_1_:%.+]] = arith.mulf [[VAR_90_1_]], [[VAR_91_1_]] : vector<4xf32>
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_6_:%.+]] = arith.mulf [[VAR_92_1_]], [[VAR_86_1_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_94_1_:%.+]] = vector.broadcast [[VAR_18_]] : f32 to vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_7_:%.+]] = arith.subf [[LOAD_PARAM_0_MEM_12_]], [[VAR_94_1_]] : vector<4xf32>
// CHECK-DAG:             [[LOAD_VAR_subview_15_MEM_2_:%.+]] = vector.broadcast [[VAR_24_]] : f32 to vector<4xf32>
// CHECK:                 [[LOAD_VAR_subview_15_MEM_1_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_7_]], [[LOAD_VAR_subview_15_MEM_2_]] : vector<4xf32>
// CHECK-DAG:             [[LOAD_VAR_subview_16_MEM_2_:%.+]] = arith.mulf [[LOAD_VAR_subview_15_MEM_1_]], [[VAR_88_1_]] : vector<4xf32>
// CHECK-DAG:             [[LOAD_VAR_subview_16_MEM_1_:%.+]] = arith.minnumf [[LOAD_PARAM_0_MEM_6_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_100_1_:%.+]] = arith.minnumf [[LOAD_VAR_subview_16_MEM_2_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_101_1_:%.+]] = arith.maxnumf [[LOAD_VAR_subview_16_MEM_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_102_1_:%.+]] = arith.maxnumf [[VAR_100_1_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_103_1_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_101_1_]], [[VAR_102_1_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                 vector.store [[VAR_103_1_]], [[VAR_reinterpret_cast_]]{{.}}[[LOAD_PARAM_0_MEM_8_]], [[I_4_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_104_1_:%.+]] = affine.apply [[MAP_4_]]([[I_3_]], [[I_4_]])
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_13_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_52_1_]], [[VAR_104_1_]]{{.}} : memref<256x256xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_106_:%.+]] = arith.addi [[VAR_104_1_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_14_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_52_1_]], [[VAR_106_]]{{.}} : memref<256x256xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_108_:%.+]] = affine.apply [[MAP_4_]]([[I_3_]], [[I_4_]])
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_PARAM_1_MEM_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_108_]]{{.}} : memref<256xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_110_:%.+]] = arith.addi [[VAR_108_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_PARAM_1_MEM_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_110_]]{{.}} : memref<256xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_112_:%.+]] = vector.broadcast [[VAR_29_]] : f32 to vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_113_:%.+]] = arith.subf [[LOAD_PARAM_0_MEM_13_]], [[VAR_112_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_114_:%.+]] = vector.broadcast [[VAR_35_]] : f32 to vector<4xf32>
// CHECK:                 [[VAR_115_:%.+]] = arith.mulf [[VAR_113_]], [[VAR_114_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_116_:%.+]] = arith.mulf [[VAR_115_]], [[LOAD_PARAM_1_MEM_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_117_:%.+]] = vector.broadcast [[VAR_29_]] : f32 to vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_118_:%.+]] = arith.subf [[LOAD_PARAM_0_MEM_14_]], [[VAR_117_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_119_:%.+]] = vector.broadcast [[VAR_35_]] : f32 to vector<4xf32>
// CHECK:                 [[VAR_120_:%.+]] = arith.mulf [[VAR_118_]], [[VAR_119_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_121_:%.+]] = arith.mulf [[VAR_120_]], [[LOAD_PARAM_1_MEM_1_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_122_:%.+]] = arith.minnumf [[VAR_116_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_123_:%.+]] = arith.minnumf [[VAR_121_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_124_:%.+]] = arith.maxnumf [[VAR_122_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_125_:%.+]] = arith.maxnumf [[VAR_123_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_126_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_124_]], [[VAR_125_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                 vector.store [[VAR_126_]], [[VAR_reinterpret_cast_]]{{.}}[[LOAD_VAR_subview_MEM_2_]], [[I_4_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:                 [[VAR_127_:%.+]] = affine.apply [[MAP_4_]]([[I_3_]], [[I_4_]])
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_15_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[LOAD_VAR_subview_MEM_1_]], [[VAR_127_]]{{.}} : memref<256x256xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_129_:%.+]] = arith.addi [[VAR_127_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_16_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[LOAD_VAR_subview_MEM_1_]], [[VAR_129_]]{{.}} : memref<256x256xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_131_:%.+]] = affine.apply [[MAP_4_]]([[I_3_]], [[I_4_]])
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_PARAM_1_MEM_2_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_131_]]{{.}} : memref<256xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_133_:%.+]] = arith.addi [[VAR_131_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_PARAM_1_MEM_3_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_133_]]{{.}} : memref<256xf32>, vector<4xf32>
// CHECK-DAG:             [[VAR_135_:%.+]] = vector.broadcast [[VAR_40_]] : f32 to vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_136_:%.+]] = arith.subf [[LOAD_PARAM_0_MEM_15_]], [[VAR_135_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_137_:%.+]] = vector.broadcast [[VAR_46_]] : f32 to vector<4xf32>
// CHECK:                 [[VAR_138_:%.+]] = arith.mulf [[VAR_136_]], [[VAR_137_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_139_:%.+]] = arith.mulf [[VAR_138_]], [[LOAD_PARAM_1_MEM_2_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_140_:%.+]] = vector.broadcast [[VAR_40_]] : f32 to vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_141_:%.+]] = arith.subf [[LOAD_PARAM_0_MEM_16_]], [[VAR_140_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_142_:%.+]] = vector.broadcast [[VAR_46_]] : f32 to vector<4xf32>
// CHECK:                 [[VAR_143_:%.+]] = arith.mulf [[VAR_141_]], [[VAR_142_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_144_:%.+]] = arith.mulf [[VAR_143_]], [[LOAD_PARAM_1_MEM_3_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_145_:%.+]] = arith.minnumf [[VAR_139_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_146_:%.+]] = arith.minnumf [[VAR_144_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK-DAG:             [[VAR_147_:%.+]] = arith.maxnumf [[VAR_145_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_148_:%.+]] = arith.maxnumf [[VAR_146_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:                 [[VAR_149_:%.+]] = "zlow.vec_f32_to_dlf16"([[VAR_147_]], [[VAR_148_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                 vector.store [[VAR_149_]], [[VAR_reinterpret_cast_]]{{.}}[[LOAD_VAR_subview_10_MEM_1_]], [[I_4_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<256x256xf16, #map>
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<3xi64>
// CHECK:         }
}

