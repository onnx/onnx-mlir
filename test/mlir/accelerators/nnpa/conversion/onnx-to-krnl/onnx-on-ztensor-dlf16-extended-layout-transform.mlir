// RUN: onnx-mlir-opt --march=z17 --maccel=NNPA --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// -----

// Test doing layer norm computation directly on zTensor.

func.func @extended_layout_transform(%arg0: tensor<3x?x512xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<3x8x?x64xf32> {
  %4 = "zhigh.ExtendedLayoutTransform"(%arg0) {dlf16_to_f32 = true, reshape_merge_axis = -1 : si64, reshape_split_axis = 2 : si64, reshape_split_factor = 64 : si64, transpose_pattern = [0, 2, 1, 3]} : (tensor<3x?x512xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<3x8x?x64xf32>
  return %4 : tensor<3x8x?x64xf32>
// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d0 * 64 + d1 * 64)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 floordiv 64)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0) -> (d0 + 8)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64 + 8)>
// CHECK-DAG:   [[MAP_7_:#.+]] = affine_map<(d0) -> (d0 + 16)>
// CHECK-DAG:   [[MAP_8_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64 + 16)>
// CHECK-DAG:   [[MAP_9_:#.+]] = affine_map<(d0) -> (d0 + 24)>
// CHECK-DAG:   [[MAP_10_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64 + 24)>
// CHECK-LABEL:  func.func @extended_layout_transform
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x?x512xf16, #map>) -> memref<3x8x?x64xf32> {
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<3x?x512xf16, #map>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<3x8x?x64xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<3x?x512xf16, #map> to memref<2x64xf16>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]]), [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 8, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 1){
// CHECK:             [[VAR_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[VAR_2_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#3, [[VAR_1_]]#2)
// CHECK:             [[VAR_3_:%.+]] = krnl.get_linear_offset_index [[PARAM_0_]] at {{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_2_]]{{.}} : memref<3x?x512xf16, #map>
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.apply [[MAP_3_]]([[VAR_3_]])
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_1_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_1_]] -> [[I_4_:%.+]] = 0 to 64){
// CHECK:               [[VAR_6_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reinterpret_cast_MEM_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[VAR_6_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_output1_:%.+]], [[VAR_output2_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:               [[VAR_8_:%.+]] = affine.apply [[MAP_4_]]([[VAR_6_]], [[VAR_1_]]#3)
// CHECK:               vector.store [[VAR_output1_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_1_]]#1, [[VAR_8_]]{{.}} : memref<3x8x?x64xf32>, vector<4xf32>
// CHECK:               [[VAR_9_:%.+]] = arith.addi [[VAR_8_]], [[CST_4_]] : index
// CHECK:               vector.store [[VAR_output2_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_1_]]#1, [[VAR_9_]]{{.}} : memref<3x8x?x64xf32>, vector<4xf32>
// CHECK:               [[VAR_10_:%.+]] = affine.apply [[MAP_5_]]([[VAR_6_]])
// CHECK:               [[LOAD_VAR_reinterpret_cast_MEM_1_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[VAR_10_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_output1_0_:%.+]], [[VAR_output2_1_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_1_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:               [[VAR_12_:%.+]] = affine.apply [[MAP_6_]]([[VAR_6_]], [[VAR_1_]]#3)
// CHECK:               vector.store [[VAR_output1_0_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_1_]]#1, [[VAR_1_]]2] : memref<3x8x?x64xf32>, vector<4xf32>
// CHECK:               [[VAR_13_:%.+]] = arith.addi [[VAR_12_]], [[CST_4_]] : index
// CHECK:               vector.store [[VAR_output2_1_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_1_]]#1, [[VAR_1_]]3] : memref<3x8x?x64xf32>, vector<4xf32>
// CHECK:               [[VAR_14_:%.+]] = affine.apply [[MAP_7_]]([[VAR_6_]])
// CHECK:               [[LOAD_VAR_reinterpret_cast_MEM_2_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[VAR_14_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_output1_2_:%.+]], [[VAR_output2_3_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_2_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:               [[VAR_16_:%.+]] = affine.apply [[MAP_8_]]([[VAR_6_]], [[VAR_1_]]#3)
// CHECK:               vector.store [[VAR_output1_2_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_1_]]#1, [[VAR_1_]]6] : memref<3x8x?x64xf32>, vector<4xf32>
// CHECK:               [[VAR_17_:%.+]] = arith.addi [[VAR_16_]], [[CST_4_]] : index
// CHECK:               vector.store [[VAR_output2_3_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_1_]]#1, [[VAR_1_]]7] : memref<3x8x?x64xf32>, vector<4xf32>
// CHECK:               [[VAR_18_:%.+]] = affine.apply [[MAP_9_]]([[VAR_6_]])
// CHECK:               [[LOAD_VAR_reinterpret_cast_MEM_3_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[VAR_18_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:               [[VAR_output1_4_:%.+]], [[VAR_output2_5_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_3_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:               [[VAR_20_:%.+]] = affine.apply [[MAP_10_]]([[VAR_6_]], [[VAR_1_]]#3)
// CHECK:               vector.store [[VAR_output1_4_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_1_]]#1, [[VAR_20_]]{{.}} : memref<3x8x?x64xf32>, vector<4xf32>
// CHECK:               [[VAR_21_:%.+]] = arith.addi [[VAR_20_]], [[CST_4_]] : index
// CHECK:               vector.store [[VAR_output2_5_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_1_]]#1, [[VAR_21_]]{{.}} : memref<3x8x?x64xf32>, vector<4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x8x?x64xf32>
// CHECK:         }
}