// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// -----

// Model pattern: dynamic (A, S), C = 1152 = 3 x 12 x 32, transpose
// [0, 3, 2, 1, 4]. D = 32, so each stick holds two heads.

func.func @test_unstick_split_heads_qkv(%arg0: tensor<?x?x1152xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %arg1: tensor<5xi64>) -> (tensor<?x12x1x?x32xf32>, tensor<?x12x1x?x32xf32>, tensor<?x12x1x?x32xf32>) {
  %0:3 = "onnx.Fused"(%arg0, %arg1) <{kind = "zhigh.unstick-split-heads"}> ({
  ^bb0(%arg2: tensor<?x?x1152xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %arg3: tensor<5xi64>):
    %1 = onnx.Constant dense<1> : tensor<3xi64>
    %2 = "zhigh.Unstick"(%arg2) : (tensor<?x?x1152xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<?x?x1152xf32>
    %3 = "onnx.Reshape"(%2, %arg3) <{allowzero = 0 : si64}> : (tensor<?x?x1152xf32>, tensor<5xi64>) -> tensor<?x?x3x12x32xf32>
    %4 = "onnx.Transpose"(%3) <{perm = [0, 3, 2, 1, 4]}> : (tensor<?x?x3x12x32xf32>) -> tensor<?x12x3x?x32xf32>
    %5:3 = "onnx.Split"(%4, %1) <{axis = 2 : si64}> : (tensor<?x12x3x?x32xf32>, tensor<3xi64>) -> (tensor<?x12x1x?x32xf32>, tensor<?x12x1x?x32xf32>, tensor<?x12x1x?x32xf32>)
    onnx.Yield %5#0, %5#1, %5#2 : tensor<?x12x1x?x32xf32>, tensor<?x12x1x?x32xf32>, tensor<?x12x1x?x32xf32>
  }) {headDim = 32 : i64, numHeads = 12 : i64, numSplits = 3 : i64, outputModes = ["f32", "f32", "f32"], splitAxis = 2 : i64, transposePattern = [0, 3, 2, 1, 4]} : (tensor<?x?x1152xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<5xi64>) -> (tensor<?x12x1x?x32xf32>, tensor<?x12x1x?x32xf32>, tensor<?x12x1x?x32xf32>)
  return %0#0, %0#1, %0#2 : tensor<?x12x1x?x32xf32>, tensor<?x12x1x?x32xf32>, tensor<?x12x1x?x32xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 floordiv 64)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0) -> (d0 * 2 + 1)>
// CHECK-DAG:   [[MAP_7_:#.+]] = affine_map<(d0) -> (d0 * 64 + 384)>
// CHECK-DAG:   [[MAP_8_:#.+]] = affine_map<(d0) -> (d0 * 64 + 768)>
// CHECK-LABEL:  func.func @test_unstick_split_heads_qkv
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x1152xf16, #map>, [[PARAM_1_:%.+]]: memref<5xi64>) -> (memref<?x12x1x?x32xf32>, memref<?x12x1x?x32xf32>, memref<?x12x1x?x32xf32>) {
// CHECK-DAG:       [[CST_28_:%.+]] = arith.constant 28 : index
// CHECK-DAG:       [[CST_20_:%.+]] = arith.constant 20 : index
// CHECK-DAG:       [[CST_12_:%.+]] = arith.constant 12 : index
// CHECK-DAG:       [[CST_56_:%.+]] = arith.constant 56 : index
// CHECK-DAG:       [[CST_48_:%.+]] = arith.constant 48 : index
// CHECK-DAG:       [[CST_40_:%.+]] = arith.constant 40 : index
// CHECK-DAG:       [[CST_24_:%.+]] = arith.constant 24 : index
// CHECK-DAG:       [[CST_16_:%.+]] = arith.constant 16 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x1152xf16, #map>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x1152xf16, #map>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_0_]]) alignment = 16 : memref<?x12x1x?x32xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_0_]]) alignment = 16 : memref<?x12x1x?x32xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_0_]]) alignment = 16 : memref<?x12x1x?x32xf32>
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<?x?x1152xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 6, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to [[MAP_2_]]([[VAR_dim_]], [[VAR_dim_0_]])){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[VAR_2_:%.+]] = affine.apply [[MAP_3_]]([[VAR_1_]]#1)
// CHECK:             [[VAR_3_:%.+]] = krnl.get_linear_offset_index [[PARAM_0_]] at {{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_2_]]{{.}} : memref<?x?x1152xf16, #map>
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.apply [[MAP_4_]]([[VAR_3_]])
// CHECK-DAG:         [[VAR_5_:%.+]] = affine.apply [[MAP_5_]]([[VAR_1_]]#1)
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[CST_0_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_:%.+]], [[VAR_output2_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_5_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_0_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_5_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_4_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_1_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[CST_8_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_3_:%.+]], [[VAR_output2_4_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_1_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_3_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_5_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_8_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_4_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_5_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_12_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_2_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[CST_16_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_5_:%.+]], [[VAR_output2_6_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_2_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_5_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_5_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_16_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_6_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_5_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_20_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_3_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[CST_24_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_7_:%.+]], [[VAR_output2_8_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_3_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_7_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_5_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_24_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_8_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_5_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_28_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_10_:%.+]] = affine.apply [[MAP_6_]]([[VAR_1_]]#1)
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_MEM_4_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[CST_32_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_9_:%.+]], [[VAR_output2_10_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_4_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_9_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_10_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_0_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_10_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_10_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_4_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_5_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[CST_40_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_11_:%.+]], [[VAR_output2_12_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_5_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_11_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_10_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_8_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_12_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_10_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_12_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_6_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[CST_48_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_13_:%.+]], [[VAR_output2_14_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_6_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_13_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_10_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_16_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_14_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_10_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_20_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_7_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[CST_56_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_15_:%.+]], [[VAR_output2_16_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_7_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_15_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_10_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_24_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_16_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_10_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_28_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             [[VAR_15_:%.+]] = affine.apply [[MAP_7_]]([[VAR_1_]]#1)
// CHECK:             [[VAR_16_:%.+]] = krnl.get_linear_offset_index [[PARAM_0_]] at {{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_15_]]{{.}} : memref<?x?x1152xf16, #map>
// CHECK-DAG:         [[VAR_17_:%.+]] = affine.apply [[MAP_4_]]([[VAR_16_]])
// CHECK-DAG:         [[VAR_18_:%.+]] = affine.apply [[MAP_5_]]([[VAR_1_]]#1)
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_8_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_17_]], [[CST_0_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_17_:%.+]], [[VAR_output2_18_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_8_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_17_]], [[RES_1_]]{{.}}[[VAR_1_]]#0, [[VAR_18_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_0_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_18_]], [[RES_1_]]{{.}}[[VAR_1_]]#0, [[VAR_18_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_4_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_9_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_17_]], [[CST_8_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_19_:%.+]], [[VAR_output2_20_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_9_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_19_]], [[RES_1_]]{{.}}[[VAR_1_]]#0, [[VAR_18_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_8_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_20_]], [[RES_1_]]{{.}}[[VAR_1_]]#0, [[VAR_18_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_12_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_10_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_17_]], [[CST_16_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_21_:%.+]], [[VAR_output2_22_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_10_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_21_]], [[RES_1_]]{{.}}[[VAR_1_]]#0, [[VAR_18_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_16_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_22_]], [[RES_1_]]{{.}}[[VAR_1_]]#0, [[VAR_18_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_20_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_11_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_17_]], [[CST_24_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_23_:%.+]], [[VAR_output2_24_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_11_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_23_]], [[RES_1_]]{{.}}[[VAR_1_]]#0, [[VAR_18_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_24_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_24_]], [[RES_1_]]{{.}}[[VAR_1_]]#0, [[VAR_18_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_28_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_23_:%.+]] = affine.apply [[MAP_6_]]([[VAR_1_]]#1)
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_MEM_12_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_17_]], [[CST_32_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_25_:%.+]], [[VAR_output2_26_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_12_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_25_]], [[RES_1_]]{{.}}[[VAR_1_]]#0, [[VAR_23_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_0_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_26_]], [[RES_1_]]{{.}}[[VAR_1_]]#0, [[VAR_23_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_4_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_13_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_17_]], [[CST_40_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_27_:%.+]], [[VAR_output2_28_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_13_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_27_]], [[RES_1_]]{{.}}[[VAR_1_]]#0, [[VAR_23_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_8_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_28_]], [[RES_1_]]{{.}}[[VAR_1_]]#0, [[VAR_23_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_12_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_14_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_17_]], [[CST_48_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_29_:%.+]], [[VAR_output2_30_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_14_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_29_]], [[RES_1_]]{{.}}[[VAR_1_]]#0, [[VAR_23_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_16_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_30_]], [[RES_1_]]{{.}}[[VAR_1_]]#0, [[VAR_23_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_20_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_15_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_17_]], [[CST_56_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_31_:%.+]], [[VAR_output2_32_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_15_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_31_]], [[RES_1_]]{{.}}[[VAR_1_]]#0, [[VAR_23_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_24_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_32_]], [[RES_1_]]{{.}}[[VAR_1_]]#0, [[VAR_23_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_28_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             [[VAR_28_:%.+]] = affine.apply [[MAP_8_]]([[VAR_1_]]#1)
// CHECK:             [[VAR_29_:%.+]] = krnl.get_linear_offset_index [[PARAM_0_]] at {{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_28_]]{{.}} : memref<?x?x1152xf16, #map>
// CHECK-DAG:         [[VAR_30_:%.+]] = affine.apply [[MAP_4_]]([[VAR_29_]])
// CHECK-DAG:         [[VAR_31_:%.+]] = affine.apply [[MAP_5_]]([[VAR_1_]]#1)
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_16_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_30_]], [[CST_0_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_33_:%.+]], [[VAR_output2_34_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_16_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_33_]], [[RES_2_]]{{.}}[[VAR_1_]]#0, [[VAR_31_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_0_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_34_]], [[RES_2_]]{{.}}[[VAR_1_]]#0, [[VAR_31_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_4_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_17_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_30_]], [[CST_8_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_35_:%.+]], [[VAR_output2_36_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_17_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_35_]], [[RES_2_]]{{.}}[[VAR_1_]]#0, [[VAR_31_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_8_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_36_]], [[RES_2_]]{{.}}[[VAR_1_]]#0, [[VAR_31_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_12_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_18_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_30_]], [[CST_16_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_37_:%.+]], [[VAR_output2_38_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_18_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_37_]], [[RES_2_]]{{.}}[[VAR_1_]]#0, [[VAR_31_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_16_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_38_]], [[RES_2_]]{{.}}[[VAR_1_]]#0, [[VAR_31_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_20_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_19_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_30_]], [[CST_24_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_39_:%.+]], [[VAR_output2_40_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_19_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_39_]], [[RES_2_]]{{.}}[[VAR_1_]]#0, [[VAR_31_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_24_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_40_]], [[RES_2_]]{{.}}[[VAR_1_]]#0, [[VAR_31_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_28_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_36_:%.+]] = affine.apply [[MAP_6_]]([[VAR_1_]]#1)
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_MEM_20_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_30_]], [[CST_32_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_41_:%.+]], [[VAR_output2_42_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_20_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_41_]], [[RES_2_]]{{.}}[[VAR_1_]]#0, [[VAR_36_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_0_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_42_]], [[RES_2_]]{{.}}[[VAR_1_]]#0, [[VAR_36_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_4_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_21_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_30_]], [[CST_40_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_43_:%.+]], [[VAR_output2_44_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_21_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_43_]], [[RES_2_]]{{.}}[[VAR_1_]]#0, [[VAR_36_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_8_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_44_]], [[RES_2_]]{{.}}[[VAR_1_]]#0, [[VAR_36_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_12_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_22_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_30_]], [[CST_48_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_45_:%.+]], [[VAR_output2_46_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_22_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_45_]], [[RES_2_]]{{.}}[[VAR_1_]]#0, [[VAR_36_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_16_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_46_]], [[RES_2_]]{{.}}[[VAR_1_]]#0, [[VAR_36_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_20_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_23_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_30_]], [[CST_56_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_47_:%.+]], [[VAR_output2_48_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_23_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_47_]], [[RES_2_]]{{.}}[[VAR_1_]]#0, [[VAR_36_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_24_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_48_]], [[RES_2_]]{{.}}[[VAR_1_]]#0, [[VAR_36_]], [[CST_0_]], [[VAR_1_]]#2, [[CST_28_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_1_]], [[RES_2_]] : memref<?x12x1x?x32xf32>, memref<?x12x1x?x32xf32>, memref<?x12x1x?x32xf32>
// CHECK:         }

}

// -----

// No transpose, static shapes, D = 64: each stick is part of a single head.

func.func @test_unstick_split_heads_d64_no_transpose(%arg0: tensor<2x8x384xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> (tensor<2x8x1x2x64xf32>, tensor<2x8x1x2x64xf32>, tensor<2x8x1x2x64xf32>) {
  %0:3 = "onnx.Fused"(%arg0) <{kind = "zhigh.unstick-split-heads"}> ({
  ^bb0(%arg1: tensor<2x8x384xf16, #zhigh.layout<{dataLayout = "3D"}>>):
    %1 = onnx.Constant dense<1> : tensor<3xi64>
    %2 = onnx.Constant dense<[2, 8, 3, 2, 64]> : tensor<5xi64>
    %3 = "zhigh.Unstick"(%arg1) : (tensor<2x8x384xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<2x8x384xf32>
    %4 = "onnx.Reshape"(%3, %2) <{allowzero = 0 : si64}> : (tensor<2x8x384xf32>, tensor<5xi64>) -> tensor<2x8x3x2x64xf32>
    %5:3 = "onnx.Split"(%4, %1) <{axis = 2 : si64}> : (tensor<2x8x3x2x64xf32>, tensor<3xi64>) -> (tensor<2x8x1x2x64xf32>, tensor<2x8x1x2x64xf32>, tensor<2x8x1x2x64xf32>)
    onnx.Yield %5#0, %5#1, %5#2 : tensor<2x8x1x2x64xf32>, tensor<2x8x1x2x64xf32>, tensor<2x8x1x2x64xf32>
  }) {headDim = 64 : i64, numHeads = 2 : i64, numSplits = 3 : i64, outputModes = ["f32", "f32", "f32"], splitAxis = 2 : i64} : (tensor<2x8x384xf16, #zhigh.layout<{dataLayout = "3D"}>>) -> (tensor<2x8x1x2x64xf32>, tensor<2x8x1x2x64xf32>, tensor<2x8x1x2x64xf32>)
  return %0#0, %0#1, %0#2 : tensor<2x8x1x2x64xf32>, tensor<2x8x1x2x64xf32>, tensor<2x8x1x2x64xf32>
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (0, d2 floordiv 64, d0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 floordiv 64)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 * 64 + 128)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 * 64 + 256)>
// CHECK-LABEL:  func.func @test_unstick_split_heads_d64_no_transpose
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x8x384xf16, #map>) -> (memref<2x8x1x2x64xf32>, memref<2x8x1x2x64xf32>, memref<2x8x1x2x64xf32>) {
// CHECK-DAG:       [[CST_60_:%.+]] = arith.constant 60 : index
// CHECK-DAG:       [[CST_52_:%.+]] = arith.constant 52 : index
// CHECK-DAG:       [[CST_44_:%.+]] = arith.constant 44 : index
// CHECK-DAG:       [[CST_36_:%.+]] = arith.constant 36 : index
// CHECK-DAG:       [[CST_28_:%.+]] = arith.constant 28 : index
// CHECK-DAG:       [[CST_20_:%.+]] = arith.constant 20 : index
// CHECK-DAG:       [[CST_12_:%.+]] = arith.constant 12 : index
// CHECK-DAG:       [[CST_56_:%.+]] = arith.constant 56 : index
// CHECK-DAG:       [[CST_48_:%.+]] = arith.constant 48 : index
// CHECK-DAG:       [[CST_40_:%.+]] = arith.constant 40 : index
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[CST_24_:%.+]] = arith.constant 24 : index
// CHECK-DAG:       [[CST_16_:%.+]] = arith.constant 16 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 16 : memref<2x8x1x2x64xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() alignment = 16 : memref<2x8x1x2x64xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() alignment = 16 : memref<2x8x1x2x64xf32>
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<2x8x384xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 8){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[VAR_2_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]]#1)
// CHECK:             [[VAR_3_:%.+]] = krnl.get_linear_offset_index [[PARAM_0_]] at {{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_2_]]{{.}} : memref<2x8x384xf16, #map>
// CHECK:             [[VAR_4_:%.+]] = affine.apply [[MAP_2_]]([[VAR_3_]])
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[CST_0_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_:%.+]], [[VAR_output2_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_0_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_4_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_1_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[CST_8_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_2_:%.+]], [[VAR_output2_3_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_1_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_2_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_8_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_3_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_12_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_2_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[CST_16_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_4_:%.+]], [[VAR_output2_5_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_2_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_4_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_16_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_5_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_20_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_3_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[CST_24_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_6_:%.+]], [[VAR_output2_7_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_3_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_6_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_24_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_7_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_28_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_4_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[CST_32_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_8_:%.+]], [[VAR_output2_9_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_4_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_8_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_32_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_9_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_36_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_5_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[CST_40_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_10_:%.+]], [[VAR_output2_11_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_5_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_10_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_40_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_11_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_44_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_6_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[CST_48_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_12_:%.+]], [[VAR_output2_13_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_6_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_12_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_48_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_13_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_52_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_7_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_4_]], [[CST_56_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_14_:%.+]], [[VAR_output2_15_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_7_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_14_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_56_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_15_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_60_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             [[VAR_13_:%.+]] = affine.apply [[MAP_3_]]([[VAR_1_]]#1)
// CHECK:             [[VAR_14_:%.+]] = krnl.get_linear_offset_index [[PARAM_0_]] at {{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_13_]]{{.}} : memref<2x8x384xf16, #map>
// CHECK:             [[VAR_15_:%.+]] = affine.apply [[MAP_2_]]([[VAR_14_]])
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_8_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_15_]], [[CST_0_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_16_:%.+]], [[VAR_output2_17_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_8_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_16_]], [[RES_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_0_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_17_]], [[RES_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_4_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_9_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_15_]], [[CST_8_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_18_:%.+]], [[VAR_output2_19_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_9_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_18_]], [[RES_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_8_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_19_]], [[RES_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_12_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_10_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_15_]], [[CST_16_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_20_:%.+]], [[VAR_output2_21_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_10_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_20_]], [[RES_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_16_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_21_]], [[RES_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_20_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_11_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_15_]], [[CST_24_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_22_:%.+]], [[VAR_output2_23_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_11_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_22_]], [[RES_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_24_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_23_]], [[RES_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_28_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_12_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_15_]], [[CST_32_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_24_:%.+]], [[VAR_output2_25_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_12_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_24_]], [[RES_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_32_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_25_]], [[RES_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_36_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_13_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_15_]], [[CST_40_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_26_:%.+]], [[VAR_output2_27_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_13_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_26_]], [[RES_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_40_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_27_]], [[RES_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_44_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_14_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_15_]], [[CST_48_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_28_:%.+]], [[VAR_output2_29_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_14_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_28_]], [[RES_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_48_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_29_]], [[RES_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_52_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_15_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_15_]], [[CST_56_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_30_:%.+]], [[VAR_output2_31_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_15_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_30_]], [[RES_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_56_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_31_]], [[RES_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_60_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             [[VAR_24_:%.+]] = affine.apply [[MAP_4_]]([[VAR_1_]]#1)
// CHECK:             [[VAR_25_:%.+]] = krnl.get_linear_offset_index [[PARAM_0_]] at {{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_24_]]{{.}} : memref<2x8x384xf16, #map>
// CHECK:             [[VAR_26_:%.+]] = affine.apply [[MAP_2_]]([[VAR_25_]])
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_16_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_26_]], [[CST_0_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_32_:%.+]], [[VAR_output2_33_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_16_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_32_]], [[RES_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_0_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_33_]], [[RES_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_4_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_17_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_26_]], [[CST_8_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_34_:%.+]], [[VAR_output2_35_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_17_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_34_]], [[RES_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_8_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_35_]], [[RES_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_12_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_18_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_26_]], [[CST_16_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_36_:%.+]], [[VAR_output2_37_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_18_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_36_]], [[RES_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_16_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_37_]], [[RES_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_20_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_19_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_26_]], [[CST_24_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_38_:%.+]], [[VAR_output2_39_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_19_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_38_]], [[RES_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_24_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_39_]], [[RES_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_28_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_20_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_26_]], [[CST_32_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_40_:%.+]], [[VAR_output2_41_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_20_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_40_]], [[RES_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_32_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_41_]], [[RES_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_36_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_21_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_26_]], [[CST_40_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_42_:%.+]], [[VAR_output2_43_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_21_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_42_]], [[RES_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_40_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_43_]], [[RES_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_44_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_22_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_26_]], [[CST_48_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_44_:%.+]], [[VAR_output2_45_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_22_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_44_]], [[RES_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_48_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_45_]], [[RES_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_52_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_23_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_26_]], [[CST_56_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_46_:%.+]], [[VAR_output2_47_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_23_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_46_]], [[RES_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_56_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_47_]], [[RES_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[CST_0_]], [[VAR_1_]]#1, [[CST_60_]]{{.}} : memref<2x8x1x2x64xf32>, vector<4xf32>
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_1_]], [[RES_2_]] : memref<2x8x1x2x64xf32>, memref<2x8x1x2x64xf32>, memref<2x8x1x2x64xf32>
// CHECK:         }

}


// -----

// Model pattern with V folded in (Squeeze -> Reshape (A * H, S, D) -> 3DS
// Stick): V is copied as dlf16, one half stick per head, no conversion.

func.func @test_unstick_split_heads_qkv_stick_v(%arg0: tensor<?x?x1152xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %arg1: tensor<1xi64>, %arg2: tensor<1xi64>) -> (tensor<?x12x1x?x32xf32>, tensor<?x12x1x?x32xf32>, tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>) {
  %0:3 = "onnx.Fused"(%arg0, %arg1, %arg2) <{kind = "zhigh.unstick-split-heads"}> ({
  ^bb0(%arg3: tensor<?x?x1152xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %arg4: tensor<1xi64>, %arg5: tensor<1xi64>):
    %1 = onnx.Constant dense<-1> : tensor<1xi64>
    %2 = onnx.Constant dense<3> : tensor<1xi64>
    %3 = onnx.Constant dense<12> : tensor<1xi64>
    %4 = onnx.Constant dense<32> : tensor<1xi64>
    %5 = onnx.Constant dense<1> : tensor<3xi64>
    %6 = onnx.Constant dense<2> : tensor<1xi64>
    %7 = "onnx.Concat"(%arg4, %1, %2, %3, %4) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<5xi64>
    %8 = "zhigh.Unstick"(%arg3) : (tensor<?x?x1152xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<?x?x1152xf32>
    %9 = "onnx.Reshape"(%8, %7) <{allowzero = 0 : si64}> : (tensor<?x?x1152xf32>, tensor<5xi64>) -> tensor<?x?x3x12x32xf32>
    %10 = "onnx.Transpose"(%9) <{perm = [0, 3, 2, 1, 4]}> : (tensor<?x?x3x12x32xf32>) -> tensor<?x12x3x?x32xf32>
    %11:3 = "onnx.Split"(%10, %5) <{axis = 2 : si64}> : (tensor<?x12x3x?x32xf32>, tensor<3xi64>) -> (tensor<?x12x1x?x32xf32>, tensor<?x12x1x?x32xf32>, tensor<?x12x1x?x32xf32>)
    %12 = "onnx.Squeeze"(%11#2, %6) : (tensor<?x12x1x?x32xf32>, tensor<1xi64>) -> tensor<?x12x?x32xf32>
    %13 = "onnx.Concat"(%1, %arg5, %4) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
    %14 = "onnx.Reshape"(%12, %13) <{allowzero = 0 : si64}> : (tensor<?x12x?x32xf32>, tensor<3xi64>) -> tensor<?x?x32xf32>
    %15 = "zhigh.Stick"(%14) <{layout = "3DS"}> : (tensor<?x?x32xf32>) -> tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    onnx.Yield %11#0, %11#1, %15 : tensor<?x12x1x?x32xf32>, tensor<?x12x1x?x32xf32>, tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  }) {headDim = 32 : i64, numHeads = 12 : i64, numSplits = 3 : i64, outputModes = ["f32", "f32", "stick-3DS"], splitAxis = 2 : i64, transposePattern = [0, 3, 2, 1, 4]} : (tensor<?x?x1152xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1xi64>, tensor<1xi64>) -> (tensor<?x12x1x?x32xf32>, tensor<?x12x1x?x32xf32>, tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>)
  return %0#0, %0#1, %0#2 : tensor<?x12x1x?x32xf32>, tensor<?x12x1x?x32xf32>, tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 * 12)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0) -> (d0 floordiv 64)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-DAG:   [[MAP_7_:#.+]] = affine_map<(d0) -> (d0 * 2 + 1)>
// CHECK-DAG:   [[MAP_8_:#.+]] = affine_map<(d0) -> (d0 * 64 + 384)>
// CHECK-DAG:   [[MAP_9_:#.+]] = affine_map<(d0) -> (d0 * 64 + 768)>
// CHECK-DAG:   [[MAP_10_:#.+]] = affine_map<(d0, d1) -> (d0 * 2 + d1 * 12)>
// CHECK-DAG:   [[MAP_11_:#.+]] = affine_map<(d0) -> (d0 * 64 + 800)>
// CHECK-DAG:   [[MAP_12_:#.+]] = affine_map<(d0, d1) -> (d0 * 2 + d1 * 12 + 1)>
// CHECK-LABEL:  func.func @test_unstick_split_heads_qkv_stick_v
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x1152xf16, #map>, [[PARAM_1_:%.+]]: memref<1xi64>, [[PARAM_2_:%.+]]: memref<1xi64>) -> (memref<?x12x1x?x32xf32>, memref<?x12x1x?x32xf32>, memref<?x?x32xf16, #map>) {
// CHECK-DAG:       [[CST_28_:%.+]] = arith.constant 28 : index
// CHECK-DAG:       [[CST_20_:%.+]] = arith.constant 20 : index
// CHECK-DAG:       [[CST_12_:%.+]] = arith.constant 12 : index
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : i64
// CHECK-DAG:       [[CST_56_:%.+]] = arith.constant 56 : index
// CHECK-DAG:       [[CST_48_:%.+]] = arith.constant 48 : index
// CHECK-DAG:       [[CST_40_:%.+]] = arith.constant 40 : index
// CHECK-DAG:       [[CST_24_:%.+]] = arith.constant 24 : index
// CHECK-DAG:       [[CST_16_:%.+]] = arith.constant 16 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_32_1_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x1152xf16, #map>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x1152xf16, #map>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_0_]]) alignment = 16 : memref<?x12x1x?x32xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_0_]]) alignment = 16 : memref<?x12x1x?x32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc([[VAR_0_]], [[VAR_dim_0_]]) alignment = 4096 : memref<?x?x32xf16, #map>
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<?x?x1152xf16, #map> to memref<2x64xf16>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_2_]]([[VAR_dim_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 6, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to [[MAP_3_]]([[VAR_dim_]], [[VAR_dim_0_]])){
// CHECK:             [[VAR_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[VAR_3_:%.+]] = affine.apply [[MAP_4_]]([[VAR_2_]]#1)
// CHECK:             [[VAR_4_:%.+]] = krnl.get_linear_offset_index [[PARAM_0_]] at {{.}}[[VAR_2_]]#0, [[VAR_2_]]#2, [[VAR_3_]]{{.}} : memref<?x?x1152xf16, #map>
// CHECK-DAG:         [[VAR_5_:%.+]] = affine.apply [[MAP_5_]]([[VAR_4_]])
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply [[MAP_6_]]([[VAR_2_]]#1)
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_5_]], [[CST_0_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_:%.+]], [[VAR_output2_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_6_]], [[CST_0_]], [[VAR_2_]]#2, [[CST_0_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_6_]], [[CST_0_]], [[VAR_2_]]#2, [[CST_4_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_1_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_5_]], [[CST_8_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_3_:%.+]], [[VAR_output2_4_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_1_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_3_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_6_]], [[CST_0_]], [[VAR_2_]]#2, [[CST_8_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_4_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_6_]], [[CST_0_]], [[VAR_2_]]#2, [[CST_12_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_2_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_5_]], [[CST_16_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_5_:%.+]], [[VAR_output2_6_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_2_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_5_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_6_]], [[CST_0_]], [[VAR_2_]]#2, [[CST_16_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_6_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_6_]], [[CST_0_]], [[VAR_2_]]#2, [[CST_20_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_3_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_5_]], [[CST_24_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_7_:%.+]], [[VAR_output2_8_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_3_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_7_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_6_]], [[CST_0_]], [[VAR_2_]]#2, [[CST_24_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_8_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_6_]], [[CST_0_]], [[VAR_2_]]#2, [[CST_28_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_11_:%.+]] = affine.apply [[MAP_7_]]([[VAR_2_]]#1)
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_MEM_4_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_5_]], [[CST_32_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_9_:%.+]], [[VAR_output2_10_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_4_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_9_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_11_]], [[CST_0_]], [[VAR_2_]]#2, [[CST_0_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_10_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_11_]], [[CST_0_]], [[VAR_2_]]#2, [[CST_4_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_5_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_5_]], [[CST_40_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_11_:%.+]], [[VAR_output2_12_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_5_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_11_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_11_]], [[CST_0_]], [[VAR_2_]]#2, [[CST_8_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_12_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_11_]], [[CST_0_]], [[VAR_2_]]#2, [[CST_12_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_6_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_5_]], [[CST_48_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_13_:%.+]], [[VAR_output2_14_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_6_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_13_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_11_]], [[CST_0_]], [[VAR_2_]]#2, [[CST_16_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_14_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_11_]], [[CST_0_]], [[VAR_2_]]#2, [[CST_20_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_7_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_5_]], [[CST_56_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_15_:%.+]], [[VAR_output2_16_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_7_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_15_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_11_]], [[CST_0_]], [[VAR_2_]]#2, [[CST_24_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_16_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_11_]], [[CST_0_]], [[VAR_2_]]#2, [[CST_28_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             [[VAR_16_:%.+]] = affine.apply [[MAP_8_]]([[VAR_2_]]#1)
// CHECK:             [[VAR_17_:%.+]] = krnl.get_linear_offset_index [[PARAM_0_]] at {{.}}[[VAR_2_]]#0, [[VAR_2_]]#2, [[VAR_16_]]{{.}} : memref<?x?x1152xf16, #map>
// CHECK-DAG:         [[VAR_18_:%.+]] = affine.apply [[MAP_5_]]([[VAR_17_]])
// CHECK-DAG:         [[VAR_19_:%.+]] = affine.apply [[MAP_6_]]([[VAR_2_]]#1)
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_8_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_18_]], [[CST_0_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_17_:%.+]], [[VAR_output2_18_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_8_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_17_]], [[RES_1_]]{{.}}[[VAR_2_]]#0, [[VAR_19_]], [[CST_0_]], [[VAR_2_]]#2, [[CST_0_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_18_]], [[RES_1_]]{{.}}[[VAR_2_]]#0, [[VAR_19_]], [[CST_0_]], [[VAR_2_]]#2, [[CST_4_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_9_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_18_]], [[CST_8_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_19_:%.+]], [[VAR_output2_20_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_9_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_19_]], [[RES_1_]]{{.}}[[VAR_2_]]#0, [[VAR_19_]], [[CST_0_]], [[VAR_2_]]#2, [[CST_8_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_20_]], [[RES_1_]]{{.}}[[VAR_2_]]#0, [[VAR_19_]], [[CST_0_]], [[VAR_2_]]#2, [[CST_12_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_10_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_18_]], [[CST_16_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_21_:%.+]], [[VAR_output2_22_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_10_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_21_]], [[RES_1_]]{{.}}[[VAR_2_]]#0, [[VAR_19_]], [[CST_0_]], [[VAR_2_]]#2, [[CST_16_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_22_]], [[RES_1_]]{{.}}[[VAR_2_]]#0, [[VAR_19_]], [[CST_0_]], [[VAR_2_]]#2, [[CST_20_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_11_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_18_]], [[CST_24_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_23_:%.+]], [[VAR_output2_24_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_11_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_23_]], [[RES_1_]]{{.}}[[VAR_2_]]#0, [[VAR_19_]], [[CST_0_]], [[VAR_2_]]#2, [[CST_24_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_24_]], [[RES_1_]]{{.}}[[VAR_2_]]#0, [[VAR_19_]], [[CST_0_]], [[VAR_2_]]#2, [[CST_28_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_24_:%.+]] = affine.apply [[MAP_7_]]([[VAR_2_]]#1)
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_MEM_12_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_18_]], [[CST_32_1_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_25_:%.+]], [[VAR_output2_26_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_12_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_25_]], [[RES_1_]]{{.}}[[VAR_2_]]#0, [[VAR_24_]], [[CST_0_]], [[VAR_2_]]#2, [[CST_0_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_26_]], [[RES_1_]]{{.}}[[VAR_2_]]#0, [[VAR_24_]], [[CST_0_]], [[VAR_2_]]#2, [[CST_4_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_13_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_18_]], [[CST_40_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_27_:%.+]], [[VAR_output2_28_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_13_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_27_]], [[RES_1_]]{{.}}[[VAR_2_]]#0, [[VAR_24_]], [[CST_0_]], [[VAR_2_]]#2, [[CST_8_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_28_]], [[RES_1_]]{{.}}[[VAR_2_]]#0, [[VAR_24_]], [[CST_0_]], [[VAR_2_]]#2, [[CST_12_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_14_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_18_]], [[CST_48_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_29_:%.+]], [[VAR_output2_30_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_14_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_29_]], [[RES_1_]]{{.}}[[VAR_2_]]#0, [[VAR_24_]], [[CST_0_]], [[VAR_2_]]#2, [[CST_16_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_30_]], [[RES_1_]]{{.}}[[VAR_2_]]#0, [[VAR_24_]], [[CST_0_]], [[VAR_2_]]#2, [[CST_20_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_15_:%.+]] = vector.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_18_]], [[CST_56_]]{{.}} : memref<2x64xf16>, vector<8xf16>
// CHECK:             [[VAR_output1_31_:%.+]], [[VAR_output2_32_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reinterpret_cast_MEM_15_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[VAR_output1_31_]], [[RES_1_]]{{.}}[[VAR_2_]]#0, [[VAR_24_]], [[CST_0_]], [[VAR_2_]]#2, [[CST_24_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_32_]], [[RES_1_]]{{.}}[[VAR_2_]]#0, [[VAR_24_]], [[CST_0_]], [[VAR_2_]]#2, [[CST_28_]]{{.}} : memref<?x12x1x?x32xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_29_:%.+]] = affine.apply [[MAP_9_]]([[VAR_2_]]#1)
// CHECK-DAG:         [[VAR_30_:%.+]] = affine.apply [[MAP_10_]]([[VAR_2_]]#1, [[VAR_2_]]#0)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_31_:%.+]] = krnl.get_linear_offset_index [[PARAM_0_]] at {{.}}[[VAR_2_]]#0, [[VAR_2_]]#2, [[VAR_29_]]{{.}} : memref<?x?x1152xf16, #map>
// CHECK-DAG:         [[VAR_32_:%.+]] = krnl.get_linear_offset_index [[RES_2_]] at {{.}}[[VAR_30_]], [[VAR_2_]]#2, [[CST_0_]]{{.}} : memref<?x?x32xf16, #map>
// CHECK:             "krnl.memcpy"([[RES_2_]], [[PARAM_0_]], [[CST_32_]], [[VAR_32_]], [[VAR_31_]]) : (memref<?x?x32xf16, #map>, memref<?x?x1152xf16, #map>, i64, index, index) -> ()
// CHECK-DAG:         [[VAR_33_:%.+]] = affine.apply [[MAP_11_]]([[VAR_2_]]#1)
// CHECK-DAG:         [[VAR_34_:%.+]] = affine.apply [[MAP_12_]]([[VAR_2_]]#1, [[VAR_2_]]#0)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_35_:%.+]] = krnl.get_linear_offset_index [[PARAM_0_]] at {{.}}[[VAR_2_]]#0, [[VAR_2_]]#2, [[VAR_33_]]{{.}} : memref<?x?x1152xf16, #map>
// CHECK-DAG:         [[VAR_36_:%.+]] = krnl.get_linear_offset_index [[RES_2_]] at {{.}}[[VAR_34_]], [[VAR_2_]]#2, [[CST_0_]]{{.}} : memref<?x?x32xf16, #map>
// CHECK:             "krnl.memcpy"([[RES_2_]], [[PARAM_0_]], [[CST_32_]], [[VAR_36_]], [[VAR_35_]]) : (memref<?x?x32xf16, #map>, memref<?x?x1152xf16, #map>, i64, index, index) -> ()
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_1_]], [[RES_2_]] : memref<?x12x1x?x32xf32>, memref<?x12x1x?x32xf32>, memref<?x?x32xf16, #map>
// CHECK:         }

}

// -----

// Static shapes, D = 64, all three outputs stickified: each input stick is
// copied whole into one output stick.

func.func @test_unstick_split_heads_d64_all_stick(%arg0: tensor<2x8x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> (tensor<4x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>) {
  %0:3 = "onnx.Fused"(%arg0) <{kind = "zhigh.unstick-split-heads"}> ({
  ^bb0(%arg1: tensor<2x8x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>):
    %1 = onnx.Constant dense<1> : tensor<3xi64>
    %2 = onnx.Constant dense<[2, 8, 3, 2, 64]> : tensor<5xi64>
    %3 = onnx.Constant dense<2> : tensor<1xi64>
    %4 = onnx.Constant dense<[4, 8, 64]> : tensor<3xi64>
    %5 = "zhigh.Unstick"(%arg1) : (tensor<2x8x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<2x8x384xf32>
    %6 = "onnx.Reshape"(%5, %2) <{allowzero = 0 : si64}> : (tensor<2x8x384xf32>, tensor<5xi64>) -> tensor<2x8x3x2x64xf32>
    %7 = "onnx.Transpose"(%6) <{perm = [0, 3, 2, 1, 4]}> : (tensor<2x8x3x2x64xf32>) -> tensor<2x2x3x8x64xf32>
    %8:3 = "onnx.Split"(%7, %1) <{axis = 2 : si64}> : (tensor<2x2x3x8x64xf32>, tensor<3xi64>) -> (tensor<2x2x1x8x64xf32>, tensor<2x2x1x8x64xf32>, tensor<2x2x1x8x64xf32>)
    %9 = "onnx.Squeeze"(%8#0, %3) : (tensor<2x2x1x8x64xf32>, tensor<1xi64>) -> tensor<2x2x8x64xf32>
    %10 = "onnx.Reshape"(%9, %4) <{allowzero = 0 : si64}> : (tensor<2x2x8x64xf32>, tensor<3xi64>) -> tensor<4x8x64xf32>
    %11 = "zhigh.Stick"(%10) <{layout = "3DS"}> : (tensor<4x8x64xf32>) -> tensor<4x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    %12 = "onnx.Squeeze"(%8#1, %3) : (tensor<2x2x1x8x64xf32>, tensor<1xi64>) -> tensor<2x2x8x64xf32>
    %13 = "onnx.Reshape"(%12, %4) <{allowzero = 0 : si64}> : (tensor<2x2x8x64xf32>, tensor<3xi64>) -> tensor<4x8x64xf32>
    %14 = "zhigh.Stick"(%13) <{layout = "3DS"}> : (tensor<4x8x64xf32>) -> tensor<4x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    %15 = "onnx.Squeeze"(%8#2, %3) : (tensor<2x2x1x8x64xf32>, tensor<1xi64>) -> tensor<2x2x8x64xf32>
    %16 = "onnx.Reshape"(%15, %4) <{allowzero = 0 : si64}> : (tensor<2x2x8x64xf32>, tensor<3xi64>) -> tensor<4x8x64xf32>
    %17 = "zhigh.Stick"(%16) <{layout = "3DS"}> : (tensor<4x8x64xf32>) -> tensor<4x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    onnx.Yield %11, %14, %17 : tensor<4x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  }) {headDim = 64 : i64, numHeads = 2 : i64, numSplits = 3 : i64, outputModes = ["stick-3DS", "stick-3DS", "stick-3DS"], splitAxis = 2 : i64, transposePattern = [0, 3, 2, 1, 4]} : (tensor<2x8x384xf16, #zhigh.layout<{dataLayout = "3DS"}>>) -> (tensor<4x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>)
  return %0#0, %0#1, %0#2 : tensor<4x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<4x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 2)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 * 64 + 128)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 * 64 + 256)>
// CHECK-LABEL:  func.func @test_unstick_split_heads_d64_all_stick
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x8x384xf16, #map>) -> (memref<4x8x64xf16, #map>, memref<4x8x64xf16, #map>, memref<4x8x64xf16, #map>) {
// CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : i64
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 4096 : memref<4x8x64xf16, #map>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() alignment = 4096 : memref<4x8x64xf16, #map>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() alignment = 4096 : memref<4x8x64xf16, #map>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 8){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]]#1)
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#1, [[VAR_1_]]#0)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_4_:%.+]] = krnl.get_linear_offset_index [[PARAM_0_]] at {{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_2_]]{{.}} : memref<2x8x384xf16, #map>
// CHECK-DAG:         [[VAR_5_:%.+]] = krnl.get_linear_offset_index [[RES_]] at {{.}}[[VAR_3_]], [[VAR_1_]]#2, [[CST_0_]]{{.}} : memref<4x8x64xf16, #map>
// CHECK:             "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[CST_64_]], [[VAR_5_]], [[VAR_4_]]) : (memref<4x8x64xf16, #map>, memref<2x8x384xf16, #map>, i64, index, index) -> ()
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply [[MAP_3_]]([[VAR_1_]]#1)
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#1, [[VAR_1_]]#0)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_:%.+]] = krnl.get_linear_offset_index [[PARAM_0_]] at {{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_6_]]{{.}} : memref<2x8x384xf16, #map>
// CHECK-DAG:         [[VAR_9_:%.+]] = krnl.get_linear_offset_index [[RES_1_]] at {{.}}[[VAR_7_]], [[VAR_1_]]#2, [[CST_0_]]{{.}} : memref<4x8x64xf16, #map>
// CHECK:             "krnl.memcpy"([[RES_1_]], [[PARAM_0_]], [[CST_64_]], [[VAR_9_]], [[VAR_8_]]) : (memref<4x8x64xf16, #map>, memref<2x8x384xf16, #map>, i64, index, index) -> ()
// CHECK-DAG:         [[VAR_10_:%.+]] = affine.apply [[MAP_4_]]([[VAR_1_]]#1)
// CHECK-DAG:         [[VAR_11_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#1, [[VAR_1_]]#0)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_12_:%.+]] = krnl.get_linear_offset_index [[PARAM_0_]] at {{.}}[[VAR_1_]]#0, [[VAR_1_]]#2, [[VAR_10_]]{{.}} : memref<2x8x384xf16, #map>
// CHECK-DAG:         [[VAR_13_:%.+]] = krnl.get_linear_offset_index [[RES_2_]] at {{.}}[[VAR_11_]], [[VAR_1_]]#2, [[CST_0_]]{{.}} : memref<4x8x64xf16, #map>
// CHECK:             "krnl.memcpy"([[RES_2_]], [[PARAM_0_]], [[CST_64_]], [[VAR_13_]], [[VAR_12_]]) : (memref<4x8x64xf16, #map>, memref<2x8x384xf16, #map>, i64, index, index) -> ()
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_1_]], [[RES_2_]] : memref<4x8x64xf16, #map>, memref<4x8x64xf16, #map>, memref<4x8x64xf16, #map>
// CHECK:         }

}

