// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// -----


func.func @test_dlf16_to_f32(%arg0: tensor<1x3x5x?xf16>) -> tensor<1x3x5x?xf32> {
  %0 = "zhigh.DLF16ToF32"(%arg0) : (tensor<1x3x5x?xf16>) -> tensor<1x3x5x?xf32>
  return %0 : tensor<1x3x5x?xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 60 + 256)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 * 15)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1] -> (s0 * 15)>
// CHECK-LABEL:  func.func @test_dlf16_to_f32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x3x5x?xf16>) -> memref<1x3x5x?xf32> {
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
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_3_]] : memref<1x3x5x?xf16>
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?xi8>
// CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}{{.}}[[VAR_dim_]]{{.}} : memref<?xi8> to memref<1x3x5x?xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_3_]] : memref<1x3x5x?xf16>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_0_]]{{.}}
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_1_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<1x3x5x?xf16>, memref<1xindex>) -> memref<?xf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_2_]], [[RES_2_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_3_:%.+]] = memref.reshape [[VAR_view_]]([[RES_2_]]) : (memref<1x3x5x?xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 64 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_2_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_0]){
// CHECK:             [[VAR_4_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.addi [[VAR_4_]], [[CST_4_]] : index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf16>, vector<8xf16>
// CHECK:             [[output1_:%.+]], [[VAR_output2_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reshape_MEM_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[output1_]], [[VAR_reshape_3_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_]], [[VAR_reshape_3_]]{{.}}[[VAR_5_]]{{.}} : memref<?xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.addi [[VAR_4_]], [[CST_8_]] : index
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.addi [[VAR_4_]], [[CST_12_]] : index
// CHECK:             [[LOAD_VAR_reshape_MEM_1_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_7_]]{{.}} : memref<?xf16>, vector<8xf16>
// CHECK:             [[output1_4_:%.+]], [[VAR_output2_5_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reshape_MEM_1_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[output1_4_]], [[VAR_reshape_3_]]{{.}}[[VAR_7_]]{{.}} : memref<?xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_5_]], [[VAR_reshape_3_]]{{.}}[[VAR_8_]]{{.}} : memref<?xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.addi [[VAR_4_]], [[CST_16_]] : index
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.addi [[VAR_4_]], [[CST_20_]] : index
// CHECK:             [[LOAD_VAR_reshape_MEM_2_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_10_]]{{.}} : memref<?xf16>, vector<8xf16>
// CHECK:             [[output1_6_:%.+]], [[VAR_output2_7_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reshape_MEM_2_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[output1_6_]], [[VAR_reshape_3_]]{{.}}[[VAR_10_]]{{.}} : memref<?xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_7_]], [[VAR_reshape_3_]]{{.}}[[VAR_11_]]{{.}} : memref<?xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.addi [[VAR_4_]], [[CST_24_]] : index
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.addi [[VAR_4_]], [[CST_28_]] : index
// CHECK:             [[LOAD_VAR_reshape_MEM_3_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_13_]]{{.}} : memref<?xf16>, vector<8xf16>
// CHECK:             [[output1_8_:%.+]], [[VAR_output2_9_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reshape_MEM_3_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[output1_8_]], [[VAR_reshape_3_]]{{.}}[[VAR_13_]]{{.}} : memref<?xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_9_]], [[VAR_reshape_3_]]{{.}}[[VAR_14_]]{{.}} : memref<?xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_16_:%.+]] = arith.addi [[VAR_4_]], [[CST_32_]] : index
// CHECK-DAG:         [[VAR_17_:%.+]] = arith.addi [[VAR_4_]], [[CST_36_]] : index
// CHECK:             [[LOAD_VAR_reshape_MEM_4_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_16_]]{{.}} : memref<?xf16>, vector<8xf16>
// CHECK:             [[output1_10_:%.+]], [[VAR_output2_11_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reshape_MEM_4_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[output1_10_]], [[VAR_reshape_3_]]{{.}}[[VAR_16_]]{{.}} : memref<?xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_11_]], [[VAR_reshape_3_]]{{.}}[[VAR_17_]]{{.}} : memref<?xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_19_:%.+]] = arith.addi [[VAR_4_]], [[CST_40_]] : index
// CHECK-DAG:         [[VAR_20_:%.+]] = arith.addi [[VAR_4_]], [[CST_44_]] : index
// CHECK:             [[LOAD_VAR_reshape_MEM_5_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_19_]]{{.}} : memref<?xf16>, vector<8xf16>
// CHECK:             [[output1_12_:%.+]], [[VAR_output2_13_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reshape_MEM_5_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[output1_12_]], [[VAR_reshape_3_]]{{.}}[[VAR_19_]]{{.}} : memref<?xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_13_]], [[VAR_reshape_3_]]{{.}}[[VAR_20_]]{{.}} : memref<?xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_22_:%.+]] = arith.addi [[VAR_4_]], [[CST_48_]] : index
// CHECK-DAG:         [[VAR_23_:%.+]] = arith.addi [[VAR_4_]], [[CST_52_]] : index
// CHECK:             [[LOAD_VAR_reshape_MEM_6_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_22_]]{{.}} : memref<?xf16>, vector<8xf16>
// CHECK:             [[output1_14_:%.+]], [[VAR_output2_15_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reshape_MEM_6_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[output1_14_]], [[VAR_reshape_3_]]{{.}}[[VAR_22_]]{{.}} : memref<?xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_15_]], [[VAR_reshape_3_]]{{.}}[[VAR_23_]]{{.}} : memref<?xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_25_:%.+]] = arith.addi [[VAR_4_]], [[CST_56_]] : index
// CHECK-DAG:         [[VAR_26_:%.+]] = arith.addi [[VAR_4_]], [[CST_60_]] : index
// CHECK:             [[LOAD_VAR_reshape_MEM_7_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_25_]]{{.}} : memref<?xf16>, vector<8xf16>
// CHECK:             [[output1_16_:%.+]], [[VAR_output2_17_:%.+]] = "zlow.vec_dlf16_to_f32"([[LOAD_VAR_reshape_MEM_7_]]) : (vector<8xf16>) -> (vector<4xf32>, vector<4xf32>)
// CHECK:             vector.store [[output1_16_]], [[VAR_reshape_3_]]{{.}}[[VAR_25_]]{{.}} : memref<?xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_output2_17_]], [[VAR_reshape_3_]]{{.}}[[VAR_26_]]{{.}} : memref<?xf32>, vector<4xf32>
// CHECK:           }
// CHECK:           return [[VAR_view_]] : memref<1x3x5x?xf32>
// CHECK:         }
}

