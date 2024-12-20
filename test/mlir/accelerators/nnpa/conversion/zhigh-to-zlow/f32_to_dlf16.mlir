// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// -----


func.func @test_f32_to_dlf16(%arg0: tensor<1x3x5x?xf32>) -> tensor<1x3x5x?xf16> {
  %0 = "zhigh.F32ToDLF16"(%arg0) : (tensor<1x3x5x?xf32>) -> tensor<1x3x5x?xf16>
  return %0 : tensor<1x3x5x?xf16>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 30 + 128)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 * 15)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1] -> (s0 * 15)>
// CHECK-LABEL:  func.func @test_f32_to_dlf16
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x3x5x?xf32>) -> memref<1x3x5x?xf16> {
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
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_3_]] : memref<1x3x5x?xf32>
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?xi8>
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}{{.}}[[VAR_dim_]]{{.}} : memref<?xi8> to memref<1x3x5x?xf16>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_3_]] : memref<1x3x5x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_0_]]{{.}}
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_1_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<1x3x5x?xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_2_]], [[RES_2_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_3_:%.+]] = memref.reshape [[VAR_view_]]([[RES_2_]]) : (memref<1x3x5x?xf16>, memref<1xindex>) -> memref<?xf16>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 64 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_2_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_0]){
// CHECK:             [[VAR_4_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.addi [[VAR_4_]], [[CST_4_]] : index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<4xf32>
// CHECK:             [[LOAD_VAR_reshape_MEM_1_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_5_]]{{.}} : memref<?xf32>, vector<4xf32>
// CHECK:             [[VAR_8_:%.+]] = "zlow.vec_f32_to_dlf16"([[LOAD_VAR_reshape_MEM_]], [[LOAD_VAR_reshape_MEM_1_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_8_]], [[VAR_reshape_3_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf16>, vector<8xf16>
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.addi [[VAR_4_]], [[CST_8_]] : index
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.addi [[VAR_4_]], [[CST_12_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_2_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_9_]]{{.}} : memref<?xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_3_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_10_]]{{.}} : memref<?xf32>, vector<4xf32>
// CHECK:             [[VAR_13_:%.+]] = "zlow.vec_f32_to_dlf16"([[LOAD_VAR_reshape_MEM_2_]], [[LOAD_VAR_reshape_MEM_3_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_13_]], [[VAR_reshape_3_]]{{.}}[[VAR_9_]]{{.}} : memref<?xf16>, vector<8xf16>
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.addi [[VAR_4_]], [[CST_16_]] : index
// CHECK-DAG:         [[VAR_15_:%.+]] = arith.addi [[VAR_4_]], [[CST_20_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_4_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_14_]]{{.}} : memref<?xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_5_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_15_]]{{.}} : memref<?xf32>, vector<4xf32>
// CHECK:             [[VAR_18_:%.+]] = "zlow.vec_f32_to_dlf16"([[LOAD_VAR_reshape_MEM_4_]], [[LOAD_VAR_reshape_MEM_5_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_18_]], [[VAR_reshape_3_]]{{.}}[[VAR_14_]]{{.}} : memref<?xf16>, vector<8xf16>
// CHECK-DAG:         [[VAR_19_:%.+]] = arith.addi [[VAR_4_]], [[CST_24_]] : index
// CHECK-DAG:         [[VAR_20_:%.+]] = arith.addi [[VAR_4_]], [[CST_28_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_6_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_19_]]{{.}} : memref<?xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_7_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_20_]]{{.}} : memref<?xf32>, vector<4xf32>
// CHECK:             [[VAR_23_:%.+]] = "zlow.vec_f32_to_dlf16"([[LOAD_VAR_reshape_MEM_6_]], [[LOAD_VAR_reshape_MEM_7_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_23_]], [[VAR_reshape_3_]]{{.}}[[VAR_19_]]{{.}} : memref<?xf16>, vector<8xf16>
// CHECK-DAG:         [[VAR_24_:%.+]] = arith.addi [[VAR_4_]], [[CST_32_]] : index
// CHECK-DAG:         [[VAR_25_:%.+]] = arith.addi [[VAR_4_]], [[CST_36_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_8_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_24_]]{{.}} : memref<?xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_9_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_25_]]{{.}} : memref<?xf32>, vector<4xf32>
// CHECK:             [[VAR_28_:%.+]] = "zlow.vec_f32_to_dlf16"([[LOAD_VAR_reshape_MEM_8_]], [[LOAD_VAR_reshape_MEM_9_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_28_]], [[VAR_reshape_3_]]{{.}}[[VAR_24_]]{{.}} : memref<?xf16>, vector<8xf16>
// CHECK-DAG:         [[VAR_29_:%.+]] = arith.addi [[VAR_4_]], [[CST_40_]] : index
// CHECK-DAG:         [[VAR_30_:%.+]] = arith.addi [[VAR_4_]], [[CST_44_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_10_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_29_]]{{.}} : memref<?xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_11_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_30_]]{{.}} : memref<?xf32>, vector<4xf32>
// CHECK:             [[VAR_33_:%.+]] = "zlow.vec_f32_to_dlf16"([[LOAD_VAR_reshape_MEM_10_]], [[LOAD_VAR_reshape_MEM_11_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_33_]], [[VAR_reshape_3_]]{{.}}[[VAR_29_]]{{.}} : memref<?xf16>, vector<8xf16>
// CHECK-DAG:         [[VAR_34_:%.+]] = arith.addi [[VAR_4_]], [[CST_48_]] : index
// CHECK-DAG:         [[VAR_35_:%.+]] = arith.addi [[VAR_4_]], [[CST_52_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_12_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_34_]]{{.}} : memref<?xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_13_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_35_]]{{.}} : memref<?xf32>, vector<4xf32>
// CHECK:             [[VAR_38_:%.+]] = "zlow.vec_f32_to_dlf16"([[LOAD_VAR_reshape_MEM_12_]], [[LOAD_VAR_reshape_MEM_13_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_38_]], [[VAR_reshape_3_]]{{.}}[[VAR_34_]]{{.}} : memref<?xf16>, vector<8xf16>
// CHECK-DAG:         [[VAR_39_:%.+]] = arith.addi [[VAR_4_]], [[CST_56_]] : index
// CHECK-DAG:         [[VAR_40_:%.+]] = arith.addi [[VAR_4_]], [[CST_60_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_14_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_39_]]{{.}} : memref<?xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_15_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_40_]]{{.}} : memref<?xf32>, vector<4xf32>
// CHECK:             [[VAR_43_:%.+]] = "zlow.vec_f32_to_dlf16"([[LOAD_VAR_reshape_MEM_14_]], [[LOAD_VAR_reshape_MEM_15_]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:             vector.store [[VAR_43_]], [[VAR_reshape_3_]]{{.}}[[VAR_39_]]{{.}} : memref<?xf16>, vector<8xf16>
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<1x3x5x?xf16>
// CHECK:         }
}

