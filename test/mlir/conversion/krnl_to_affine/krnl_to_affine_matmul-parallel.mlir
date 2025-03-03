
// RUN: onnx-mlir-opt -O3 --march=z16 --convert-krnl-to-affine=parallel-enabled --canonicalize %s -split-input-file | FileCheck %s

// -----


func.func @krnl_matmul_par_perfect_blocks(%arg0: memref<1024x1024xf32> {onnx.name = "x"}) -> (memref<1024x1024xf32> {onnx.name = "y"}) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c1024 = arith.constant 1024 : index
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<1024x1024xf32>
  krnl.memset %alloc, %cst : memref<1024x1024xf32>
  %0:3 = krnl.define_loops 3
  %loop_block, %loop_local = krnl.block %0#0 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  %loop_block_0, %loop_local_1 = krnl.block %0#1 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  %loop_block_2, %loop_local_3 = krnl.block %0#2 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  krnl.permute(%loop_block, %loop_local, %loop_block_0, %loop_local_1, %loop_block_2, %loop_local_3) [0, 3, 1, 4, 2, 5] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
  krnl.parallel(%loop_block) : !krnl.loop
  krnl.iterate(%loop_block, %loop_block_0, %loop_block_2) with (%0#0 -> %arg1 = %c0 to %c1024, %0#1 -> %arg2 = %c0 to %c1024, %0#2 -> %arg3 = %c0 to %c1024){
    %1:3 = krnl.get_induction_var_value(%loop_block, %loop_block_0, %loop_block_2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
    krnl.matmul %arg0[%c0, %c0], %arg0[%c0, %c0], %alloc[%c0, %c0], (%loop_local, %loop_local_1, %loop_local_3), (%1#0, %1#1, %1#2), (%c1024, %c1024, %c1024) {aTileSize = [], bTileSize = [], cTileSize = [], computeTileSize = [4, 8, 8]} : memref<1024x1024xf32>, memref<1024x1024xf32>, memref<1024x1024xf32>, (!krnl.loop, !krnl.loop, !krnl.loop)
  }
  return %alloc : memref<1024x1024xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 2)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 + 3)>
// CHECK-LABEL:  func.func @krnl_matmul_par_perfect_blocks
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1024x1024xf32> {onnx.name = "x"}) -> (memref<1024x1024xf32> {onnx.name = "y"}) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1024x1024xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 1024 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 1024 {
// CHECK:               affine.store [[CST_0_dot_000000_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<1024x1024xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           affine.parallel ([[I_0_]]) = (0) to (1024) step (4) {
// CHECK:             affine.for [[I_2_:%.+]] = 0 to 1024 step 8 {
// CHECK:               affine.for [[I_3_:%.+]] = 0 to 1024 step 8 {
// CHECK:                 [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<4xvector<8xf32>>
// skip this check        affine.if [[set_]]() {
// CHECK:                   [[LOAD_RES_MEM_:%.+]] = vector.load [[RES_]]{{.}}[[I_0_]], [[I_2_]]{{.}} : memref<1024x1024xf32>, vector<8xf32>
// CHECK:                   affine.store [[LOAD_RES_MEM_]], [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK:                   [[VAR_1_:%.+]] = arith.addi [[I_0_]], [[CST_1_]] : index
// CHECK:                   [[LOAD_RES_MEM_1_:%.+]] = vector.load [[RES_]]{{.}}[[VAR_1_]], [[I_2_]]{{.}} : memref<1024x1024xf32>, vector<8xf32>
// CHECK:                   affine.store [[LOAD_RES_MEM_1_]], [[RES_1_]][1] : memref<4xvector<8xf32>>
// CHECK:                   [[VAR_3_:%.+]] = arith.addi [[I_0_]], [[CST_2_]] : index
// CHECK:                   [[LOAD_RES_MEM_2_:%.+]] = vector.load [[RES_]]{{.}}[[VAR_3_]], [[I_2_]]{{.}} : memref<1024x1024xf32>, vector<8xf32>
// CHECK:                   affine.store [[LOAD_RES_MEM_2_]], [[RES_1_]][2] : memref<4xvector<8xf32>>
// CHECK:                   [[VAR_5_:%.+]] = arith.addi [[I_0_]], [[CST_3_]] : index
// CHECK:                   [[LOAD_RES_MEM_3_:%.+]] = vector.load [[RES_]]{{.}}[[VAR_5_]], [[I_2_]]{{.}} : memref<1024x1024xf32>, vector<8xf32>
// CHECK:                   affine.store [[LOAD_RES_MEM_3_]], [[RES_1_]][3] : memref<4xvector<8xf32>>
// CHECK:                   affine.for [[I_4_:%.+]] = 0 to 8 step 4 {
// CHECK:                     [[VAR_14_:%.+]] = arith.addi [[I_4_]], [[I_3_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[I_0_]], [[VAR_14_]]{{.}} : memref<1024x1024xf32>
// CHECK-DAG:                 [[VAR_16_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_17_:%.+]] = arith.addi [[I_4_]], [[I_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_0_MEM_1_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_17_]], [[I_2_]]{{.}} : memref<1024x1024xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_:%.+]] = affine.load [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_20_:%.+]] = vector.fma [[VAR_16_]], [[LOAD_PARAM_0_MEM_1_]], [[LOAD_RES_1_MEM_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_20_]], [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_21_:%.+]] = affine.apply [[MAP_0_]]([[I_4_]])
// CHECK:                     [[VAR_22_:%.+]] = arith.addi [[VAR_21_]], [[I_3_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_2_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[I_0_]], [[VAR_22_]]{{.}} : memref<1024x1024xf32>
// CHECK-DAG:                 [[VAR_24_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_2_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_25_:%.+]] = arith.addi [[VAR_21_]], [[I_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_0_MEM_3_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_25_]], [[I_2_]]{{.}} : memref<1024x1024xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_1_:%.+]] = affine.load [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_28_:%.+]] = vector.fma [[VAR_24_]], [[LOAD_PARAM_0_MEM_3_]], [[LOAD_RES_1_MEM_1_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_28_]], [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_29_:%.+]] = affine.apply [[MAP_1_]]([[I_4_]])
// CHECK:                     [[VAR_30_:%.+]] = arith.addi [[VAR_29_]], [[I_3_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_4_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[I_0_]], [[VAR_30_]]{{.}} : memref<1024x1024xf32>
// CHECK-DAG:                 [[VAR_32_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_4_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_33_:%.+]] = arith.addi [[VAR_29_]], [[I_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_0_MEM_5_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_33_]], [[I_2_]]{{.}} : memref<1024x1024xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_2_:%.+]] = affine.load [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_36_:%.+]] = vector.fma [[VAR_32_]], [[LOAD_PARAM_0_MEM_5_]], [[LOAD_RES_1_MEM_2_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_36_]], [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_37_:%.+]] = affine.apply [[MAP_2_]]([[I_4_]])
// CHECK:                     [[VAR_38_:%.+]] = arith.addi [[VAR_37_]], [[I_3_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_6_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[I_0_]], [[VAR_38_]]{{.}} : memref<1024x1024xf32>
// CHECK-DAG:                 [[VAR_40_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_6_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_41_:%.+]] = arith.addi [[VAR_37_]], [[I_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_0_MEM_7_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_41_]], [[I_2_]]{{.}} : memref<1024x1024xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_3_:%.+]] = affine.load [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_44_:%.+]] = vector.fma [[VAR_40_]], [[LOAD_PARAM_0_MEM_7_]], [[LOAD_RES_1_MEM_3_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_44_]], [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_45_:%.+]] = arith.addi [[I_0_]], [[CST_1_]] : index
// CHECK-DAG:                 [[VAR_46_:%.+]] = arith.addi [[I_4_]], [[I_3_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_8_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_45_]], [[VAR_46_]]{{.}} : memref<1024x1024xf32>
// CHECK-DAG:                 [[VAR_48_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_8_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_49_:%.+]] = arith.addi [[I_4_]], [[I_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_0_MEM_9_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_49_]], [[I_2_]]{{.}} : memref<1024x1024xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_4_:%.+]] = affine.load [[RES_1_]][1] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_52_:%.+]] = vector.fma [[VAR_48_]], [[LOAD_PARAM_0_MEM_9_]], [[LOAD_RES_1_MEM_4_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_52_]], [[RES_1_]][1] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_53_:%.+]] = affine.apply [[MAP_0_]]([[I_4_]])
// CHECK-DAG:                 [[VAR_54_:%.+]] = arith.addi [[I_0_]], [[CST_1_]] : index
// CHECK:                     [[VAR_55_:%.+]] = arith.addi [[VAR_53_]], [[I_3_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_10_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_54_]], [[VAR_55_]]{{.}} : memref<1024x1024xf32>
// CHECK-DAG:                 [[VAR_57_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_10_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_58_:%.+]] = arith.addi [[VAR_53_]], [[I_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_0_MEM_11_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_58_]], [[I_2_]]{{.}} : memref<1024x1024xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_5_:%.+]] = affine.load [[RES_1_]][1] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_61_:%.+]] = vector.fma [[VAR_57_]], [[LOAD_PARAM_0_MEM_11_]], [[LOAD_RES_1_MEM_5_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_61_]], [[RES_1_]][1] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_62_:%.+]] = affine.apply [[MAP_1_]]([[I_4_]])
// CHECK-DAG:                 [[VAR_63_:%.+]] = arith.addi [[I_0_]], [[CST_1_]] : index
// CHECK:                     [[VAR_64_:%.+]] = arith.addi [[VAR_62_]], [[I_3_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_12_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_63_]], [[VAR_64_]]{{.}} : memref<1024x1024xf32>
// CHECK-DAG:                 [[VAR_66_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_12_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_67_:%.+]] = arith.addi [[VAR_62_]], [[I_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_0_MEM_13_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_67_]], [[I_2_]]{{.}} : memref<1024x1024xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_6_:%.+]] = affine.load [[RES_1_]][1] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_70_:%.+]] = vector.fma [[VAR_66_]], [[LOAD_PARAM_0_MEM_13_]], [[LOAD_RES_1_MEM_6_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_70_]], [[RES_1_]][1] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_71_:%.+]] = affine.apply [[MAP_2_]]([[I_4_]])
// CHECK-DAG:                 [[VAR_72_:%.+]] = arith.addi [[I_0_]], [[CST_1_]] : index
// CHECK:                     [[VAR_73_:%.+]] = arith.addi [[VAR_71_]], [[I_3_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_14_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_72_]], [[VAR_73_]]{{.}} : memref<1024x1024xf32>
// CHECK-DAG:                 [[VAR_75_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_14_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_76_:%.+]] = arith.addi [[VAR_71_]], [[I_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_0_MEM_15_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_76_]], [[I_2_]]{{.}} : memref<1024x1024xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_7_:%.+]] = affine.load [[RES_1_]][1] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_79_:%.+]] = vector.fma [[VAR_75_]], [[LOAD_PARAM_0_MEM_15_]], [[LOAD_RES_1_MEM_7_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_79_]], [[RES_1_]][1] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_80_:%.+]] = arith.addi [[I_0_]], [[CST_2_]] : index
// CHECK-DAG:                 [[VAR_81_:%.+]] = arith.addi [[I_4_]], [[I_3_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_16_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_80_]], [[VAR_81_]]{{.}} : memref<1024x1024xf32>
// CHECK-DAG:                 [[VAR_83_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_16_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_84_:%.+]] = arith.addi [[I_4_]], [[I_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_0_MEM_17_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_84_]], [[I_2_]]{{.}} : memref<1024x1024xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_8_:%.+]] = affine.load [[RES_1_]][2] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_87_:%.+]] = vector.fma [[VAR_83_]], [[LOAD_PARAM_0_MEM_17_]], [[LOAD_RES_1_MEM_8_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_87_]], [[RES_1_]][2] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_88_:%.+]] = affine.apply [[MAP_0_]]([[I_4_]])
// CHECK-DAG:                 [[VAR_89_:%.+]] = arith.addi [[I_0_]], [[CST_2_]] : index
// CHECK:                     [[VAR_90_:%.+]] = arith.addi [[VAR_88_]], [[I_3_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_18_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_89_]], [[VAR_90_]]{{.}} : memref<1024x1024xf32>
// CHECK-DAG:                 [[VAR_92_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_18_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_93_:%.+]] = arith.addi [[VAR_88_]], [[I_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_0_MEM_19_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_93_]], [[I_2_]]{{.}} : memref<1024x1024xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_9_:%.+]] = affine.load [[RES_1_]][2] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_96_:%.+]] = vector.fma [[VAR_92_]], [[LOAD_PARAM_0_MEM_19_]], [[LOAD_RES_1_MEM_9_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_96_]], [[RES_1_]][2] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_97_:%.+]] = affine.apply [[MAP_1_]]([[I_4_]])
// CHECK-DAG:                 [[VAR_98_:%.+]] = arith.addi [[I_0_]], [[CST_2_]] : index
// CHECK:                     [[VAR_99_:%.+]] = arith.addi [[VAR_97_]], [[I_3_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_20_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_98_]], [[VAR_99_]]{{.}} : memref<1024x1024xf32>
// CHECK-DAG:                 [[VAR_101_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_20_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_102_:%.+]] = arith.addi [[VAR_97_]], [[I_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_0_MEM_21_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_102_]], [[I_2_]]{{.}} : memref<1024x1024xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_10_:%.+]] = affine.load [[RES_1_]][2] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_105_:%.+]] = vector.fma [[VAR_101_]], [[LOAD_PARAM_0_MEM_21_]], [[LOAD_RES_1_MEM_10_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_105_]], [[RES_1_]][2] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_106_:%.+]] = affine.apply [[MAP_2_]]([[I_4_]])
// CHECK-DAG:                 [[VAR_107_:%.+]] = arith.addi [[I_0_]], [[CST_2_]] : index
// CHECK:                     [[VAR_108_:%.+]] = arith.addi [[VAR_106_]], [[I_3_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_22_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_107_]], [[VAR_108_]]{{.}} : memref<1024x1024xf32>
// CHECK-DAG:                 [[VAR_110_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_22_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_111_:%.+]] = arith.addi [[VAR_106_]], [[I_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_0_MEM_23_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_111_]], [[I_2_]]{{.}} : memref<1024x1024xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_11_:%.+]] = affine.load [[RES_1_]][2] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_114_:%.+]] = vector.fma [[VAR_110_]], [[LOAD_PARAM_0_MEM_23_]], [[LOAD_RES_1_MEM_11_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_114_]], [[RES_1_]][2] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_115_:%.+]] = arith.addi [[I_0_]], [[CST_3_]] : index
// CHECK-DAG:                 [[VAR_116_:%.+]] = arith.addi [[I_4_]], [[I_3_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_24_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_115_]], [[VAR_116_]]{{.}} : memref<1024x1024xf32>
// CHECK-DAG:                 [[VAR_118_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_24_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_119_:%.+]] = arith.addi [[I_4_]], [[I_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_0_MEM_25_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_119_]], [[I_2_]]{{.}} : memref<1024x1024xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_12_:%.+]] = affine.load [[RES_1_]][3] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_122_:%.+]] = vector.fma [[VAR_118_]], [[LOAD_PARAM_0_MEM_25_]], [[LOAD_RES_1_MEM_12_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_122_]], [[RES_1_]][3] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_123_:%.+]] = affine.apply [[MAP_0_]]([[I_4_]])
// CHECK-DAG:                 [[VAR_124_:%.+]] = arith.addi [[I_0_]], [[CST_3_]] : index
// CHECK:                     [[VAR_125_:%.+]] = arith.addi [[VAR_123_]], [[I_3_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_26_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_124_]], [[VAR_125_]]{{.}} : memref<1024x1024xf32>
// CHECK-DAG:                 [[VAR_127_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_26_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_128_:%.+]] = arith.addi [[VAR_123_]], [[I_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_0_MEM_27_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_128_]], [[I_2_]]{{.}} : memref<1024x1024xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_13_:%.+]] = affine.load [[RES_1_]][3] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_131_:%.+]] = vector.fma [[VAR_127_]], [[LOAD_PARAM_0_MEM_27_]], [[LOAD_RES_1_MEM_13_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_131_]], [[RES_1_]][3] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_132_:%.+]] = affine.apply [[MAP_1_]]([[I_4_]])
// CHECK-DAG:                 [[VAR_133_:%.+]] = arith.addi [[I_0_]], [[CST_3_]] : index
// CHECK:                     [[VAR_134_:%.+]] = arith.addi [[VAR_132_]], [[I_3_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_28_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_133_]], [[VAR_134_]]{{.}} : memref<1024x1024xf32>
// CHECK-DAG:                 [[VAR_136_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_28_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_137_:%.+]] = arith.addi [[VAR_132_]], [[I_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_0_MEM_29_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_137_]], [[I_2_]]{{.}} : memref<1024x1024xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_14_:%.+]] = affine.load [[RES_1_]][3] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_140_:%.+]] = vector.fma [[VAR_136_]], [[LOAD_PARAM_0_MEM_29_]], [[LOAD_RES_1_MEM_14_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_140_]], [[RES_1_]][3] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_141_:%.+]] = affine.apply [[MAP_2_]]([[I_4_]])
// CHECK-DAG:                 [[VAR_142_:%.+]] = arith.addi [[I_0_]], [[CST_3_]] : index
// CHECK:                     [[VAR_143_:%.+]] = arith.addi [[VAR_141_]], [[I_3_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_30_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_142_]], [[VAR_143_]]{{.}} : memref<1024x1024xf32>
// CHECK-DAG:                 [[VAR_145_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_30_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_146_:%.+]] = arith.addi [[VAR_141_]], [[I_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_0_MEM_31_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_146_]], [[I_2_]]{{.}} : memref<1024x1024xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_15_:%.+]] = affine.load [[RES_1_]][3] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_149_:%.+]] = vector.fma [[VAR_145_]], [[LOAD_PARAM_0_MEM_31_]], [[LOAD_RES_1_MEM_15_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_149_]], [[RES_1_]][3] : memref<4xvector<8xf32>>
// CHECK:                   }
// CHECK:                   [[LOAD_RES_1_MEM_16_:%.+]] = affine.load [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK:                   vector.store [[LOAD_RES_1_MEM_16_]], [[RES_]]{{.}}[[I_0_]], [[I_2_]]{{.}} : memref<1024x1024xf32>, vector<8xf32>
// CHECK-DAG:               [[LOAD_RES_1_MEM_17_:%.+]] = affine.load [[RES_1_]][1] : memref<4xvector<8xf32>>
// CHECK-DAG:               [[VAR_9_:%.+]] = arith.addi [[I_0_]], [[CST_1_]] : index
// CHECK:                   vector.store [[LOAD_RES_1_MEM_17_]], [[RES_]]{{.}}[[VAR_9_]], [[I_2_]]{{.}} : memref<1024x1024xf32>, vector<8xf32>
// CHECK-DAG:               [[LOAD_RES_1_MEM_18_:%.+]] = affine.load [[RES_1_]][2] : memref<4xvector<8xf32>>
// CHECK-DAG:               [[VAR_11_:%.+]] = arith.addi [[I_0_]], [[CST_2_]] : index
// CHECK:                   vector.store [[LOAD_RES_1_MEM_18_]], [[RES_]]{{.}}[[VAR_11_]], [[I_2_]]{{.}} : memref<1024x1024xf32>, vector<8xf32>
// CHECK-DAG:               [[LOAD_RES_1_MEM_19_:%.+]] = affine.load [[RES_1_]][3] : memref<4xvector<8xf32>>
// CHECK-DAG:               [[VAR_13_:%.+]] = arith.addi [[I_0_]], [[CST_3_]] : index
// CHECK:                   vector.store [[LOAD_RES_1_MEM_19_]], [[RES_]]{{.}}[[VAR_13_]], [[I_2_]]{{.}} : memref<1024x1024xf32>, vector<8xf32>
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1024x1024xf32>
// CHECK:         }
}

// -----


func.func @krnl_matmul_parallel_partial_blocks(%arg0: memref<127x255xf32> {onnx.name = "x"}, %arg1: memref<255x63xf32> {onnx.name = "y"}) -> (memref<127x63xf32> {onnx.name = "y"}) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c127 = arith.constant 127 : index
  %c255 = arith.constant 255 : index
  %c63 = arith.constant 63 : index
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<127x63xf32>
  krnl.memset %alloc, %cst : memref<127x63xf32>
  %0:3 = krnl.define_loops 3
  %loop_block, %loop_local = krnl.block %0#0 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  %loop_block_0, %loop_local_1 = krnl.block %0#1 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  %loop_block_2, %loop_local_3 = krnl.block %0#2 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  krnl.permute(%loop_block, %loop_local, %loop_block_0, %loop_local_1, %loop_block_2, %loop_local_3) [0, 3, 1, 4, 2, 5] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
  krnl.parallel(%loop_block) : !krnl.loop
  krnl.iterate(%loop_block, %loop_block_0, %loop_block_2) with (%0#0 -> %arg2 = %c0 to %c127, %0#1 -> %arg3 = %c0 to %c63, %0#2 -> %arg4 = %c0 to %c255){
    %1:3 = krnl.get_induction_var_value(%loop_block, %loop_block_0, %loop_block_2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
    krnl.matmul %arg0[%c0, %c0], %arg1[%c0, %c0], %alloc[%c0, %c0], (%loop_local, %loop_local_1, %loop_local_3), (%1#0, %1#1, %1#2), (%c127, %c63, %c255) {aTileSize = [], bTileSize = [], cTileSize = [], computeTileSize = [4, 8, 8]} : memref<127x255xf32>, memref<255x63xf32>, memref<127x63xf32>, (!krnl.loop, !krnl.loop, !krnl.loop)
  }
  return %alloc : memref<127x63xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 2)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 + 3)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (-d0 + 127, 4)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (-d0 + 255, 8)>
// CHECK-LABEL:  func.func @krnl_matmul_parallel_partial_blocks
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<127x255xf32> {onnx.name = "x"}, [[PARAM_1_:%.+]]: memref<255x63xf32> {onnx.name = "y"}) -> (memref<127x63xf32> {onnx.name = "y"}) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<127x63xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 127 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 63 {
// CHECK:               affine.store [[CST_0_dot_000000_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<127x63xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           affine.parallel ([[I_0_]]) = (0) to (127) step (4) {
// CHECK:             affine.for [[I_2_:%.+]] = 0 to 63 step 8 {
// CHECK:               affine.for [[I_3_:%.+]] = 0 to 255 step 8 {
// CHECK-DAG:             [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<4xvector<8xf32>>
// CHECK-DAG:             [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xf32>
// skip this check        affine.if [[set_]]([[I_0_]], [[I_2_]], [[I_3_]]) {
// CHECK:                   [[LOAD_RES_MEM_:%.+]] = vector.load [[RES_]]{{.}}[[I_0_]], [[I_2_]]{{.}} : memref<127x63xf32>, vector<8xf32>
// CHECK:                   affine.store [[LOAD_RES_MEM_]], [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK:                   [[VAR_1_:%.+]] = arith.addi [[I_0_]], [[CST_1_]] : index
// CHECK:                   [[LOAD_RES_MEM_1_:%.+]] = vector.load [[RES_]]{{.}}[[VAR_1_]], [[I_2_]]{{.}} : memref<127x63xf32>, vector<8xf32>
// CHECK:                   affine.store [[LOAD_RES_MEM_1_]], [[RES_1_]][1] : memref<4xvector<8xf32>>
// CHECK:                   [[VAR_3_:%.+]] = arith.addi [[I_0_]], [[CST_2_]] : index
// CHECK:                   [[LOAD_RES_MEM_2_:%.+]] = vector.load [[RES_]]{{.}}[[VAR_3_]], [[I_2_]]{{.}} : memref<127x63xf32>, vector<8xf32>
// CHECK:                   affine.store [[LOAD_RES_MEM_2_]], [[RES_1_]][2] : memref<4xvector<8xf32>>
// CHECK:                   [[VAR_5_:%.+]] = arith.addi [[I_0_]], [[CST_3_]] : index
// CHECK:                   [[LOAD_RES_MEM_3_:%.+]] = vector.load [[RES_]]{{.}}[[VAR_5_]], [[I_2_]]{{.}} : memref<127x63xf32>, vector<8xf32>
// CHECK:                   affine.store [[LOAD_RES_MEM_3_]], [[RES_1_]][3] : memref<4xvector<8xf32>>
// CHECK:                   affine.for [[I_4_:%.+]] = 0 to 8 step 4 {
// CHECK:                     [[VAR_14_:%.+]] = arith.addi [[I_4_]], [[I_3_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[I_0_]], [[VAR_14_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                 [[VAR_16_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_17_:%.+]] = arith.addi [[I_4_]], [[I_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_1_MEM_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_17_]], [[I_2_]]{{.}} : memref<255x63xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_:%.+]] = affine.load [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_20_:%.+]] = vector.fma [[VAR_16_]], [[LOAD_PARAM_1_MEM_]], [[LOAD_RES_1_MEM_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_20_]], [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_21_:%.+]] = affine.apply [[MAP_0_]]([[I_4_]])
// CHECK:                     [[VAR_22_:%.+]] = arith.addi [[VAR_21_]], [[I_3_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_1_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[I_0_]], [[VAR_22_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                 [[VAR_24_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_1_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_25_:%.+]] = arith.addi [[VAR_21_]], [[I_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_1_MEM_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_25_]], [[I_2_]]{{.}} : memref<255x63xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_1_:%.+]] = affine.load [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_28_:%.+]] = vector.fma [[VAR_24_]], [[LOAD_PARAM_1_MEM_1_]], [[LOAD_RES_1_MEM_1_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_28_]], [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_29_:%.+]] = affine.apply [[MAP_1_]]([[I_4_]])
// CHECK:                     [[VAR_30_:%.+]] = arith.addi [[VAR_29_]], [[I_3_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_2_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[I_0_]], [[VAR_30_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                 [[VAR_32_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_2_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_33_:%.+]] = arith.addi [[VAR_29_]], [[I_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_1_MEM_2_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_33_]], [[I_2_]]{{.}} : memref<255x63xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_2_:%.+]] = affine.load [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_36_:%.+]] = vector.fma [[VAR_32_]], [[LOAD_PARAM_1_MEM_2_]], [[LOAD_RES_1_MEM_2_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_36_]], [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_37_:%.+]] = affine.apply [[MAP_2_]]([[I_4_]])
// CHECK:                     [[VAR_38_:%.+]] = arith.addi [[VAR_37_]], [[I_3_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_3_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[I_0_]], [[VAR_38_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                 [[VAR_40_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_3_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_41_:%.+]] = arith.addi [[VAR_37_]], [[I_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_1_MEM_3_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_41_]], [[I_2_]]{{.}} : memref<255x63xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_3_:%.+]] = affine.load [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_44_:%.+]] = vector.fma [[VAR_40_]], [[LOAD_PARAM_1_MEM_3_]], [[LOAD_RES_1_MEM_3_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_44_]], [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_45_:%.+]] = arith.addi [[I_0_]], [[CST_1_]] : index
// CHECK-DAG:                 [[VAR_46_:%.+]] = arith.addi [[I_4_]], [[I_3_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_4_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_45_]], [[VAR_46_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                 [[VAR_48_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_4_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_49_:%.+]] = arith.addi [[I_4_]], [[I_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_1_MEM_4_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_49_]], [[I_2_]]{{.}} : memref<255x63xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_4_:%.+]] = affine.load [[RES_1_]][1] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_52_:%.+]] = vector.fma [[VAR_48_]], [[LOAD_PARAM_1_MEM_4_]], [[LOAD_RES_1_MEM_4_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_52_]], [[RES_1_]][1] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_53_:%.+]] = affine.apply [[MAP_0_]]([[I_4_]])
// CHECK-DAG:                 [[VAR_54_:%.+]] = arith.addi [[I_0_]], [[CST_1_]] : index
// CHECK:                     [[VAR_55_:%.+]] = arith.addi [[VAR_53_]], [[I_3_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_5_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_54_]], [[VAR_55_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                 [[VAR_57_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_5_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_58_:%.+]] = arith.addi [[VAR_53_]], [[I_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_1_MEM_5_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_58_]], [[I_2_]]{{.}} : memref<255x63xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_5_:%.+]] = affine.load [[RES_1_]][1] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_61_:%.+]] = vector.fma [[VAR_57_]], [[LOAD_PARAM_1_MEM_5_]], [[LOAD_RES_1_MEM_5_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_61_]], [[RES_1_]][1] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_62_:%.+]] = affine.apply [[MAP_1_]]([[I_4_]])
// CHECK-DAG:                 [[VAR_63_:%.+]] = arith.addi [[I_0_]], [[CST_1_]] : index
// CHECK:                     [[VAR_64_:%.+]] = arith.addi [[VAR_62_]], [[I_3_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_6_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_63_]], [[VAR_64_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                 [[VAR_66_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_6_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_67_:%.+]] = arith.addi [[VAR_62_]], [[I_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_1_MEM_6_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_67_]], [[I_2_]]{{.}} : memref<255x63xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_6_:%.+]] = affine.load [[RES_1_]][1] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_70_:%.+]] = vector.fma [[VAR_66_]], [[LOAD_PARAM_1_MEM_6_]], [[LOAD_RES_1_MEM_6_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_70_]], [[RES_1_]][1] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_71_:%.+]] = affine.apply [[MAP_2_]]([[I_4_]])
// CHECK-DAG:                 [[VAR_72_:%.+]] = arith.addi [[I_0_]], [[CST_1_]] : index
// CHECK:                     [[VAR_73_:%.+]] = arith.addi [[VAR_71_]], [[I_3_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_7_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_72_]], [[VAR_73_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                 [[VAR_75_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_7_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_76_:%.+]] = arith.addi [[VAR_71_]], [[I_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_1_MEM_7_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_76_]], [[I_2_]]{{.}} : memref<255x63xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_7_:%.+]] = affine.load [[RES_1_]][1] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_79_:%.+]] = vector.fma [[VAR_75_]], [[LOAD_PARAM_1_MEM_7_]], [[LOAD_RES_1_MEM_7_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_79_]], [[RES_1_]][1] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_80_:%.+]] = arith.addi [[I_0_]], [[CST_2_]] : index
// CHECK-DAG:                 [[VAR_81_:%.+]] = arith.addi [[I_4_]], [[I_3_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_8_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_80_]], [[VAR_81_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                 [[VAR_83_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_8_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_84_:%.+]] = arith.addi [[I_4_]], [[I_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_1_MEM_8_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_84_]], [[I_2_]]{{.}} : memref<255x63xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_8_:%.+]] = affine.load [[RES_1_]][2] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_87_:%.+]] = vector.fma [[VAR_83_]], [[LOAD_PARAM_1_MEM_8_]], [[LOAD_RES_1_MEM_8_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_87_]], [[RES_1_]][2] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_88_:%.+]] = affine.apply [[MAP_0_]]([[I_4_]])
// CHECK-DAG:                 [[VAR_89_:%.+]] = arith.addi [[I_0_]], [[CST_2_]] : index
// CHECK:                     [[VAR_90_:%.+]] = arith.addi [[VAR_88_]], [[I_3_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_9_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_89_]], [[VAR_90_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                 [[VAR_92_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_9_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_93_:%.+]] = arith.addi [[VAR_88_]], [[I_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_1_MEM_9_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_93_]], [[I_2_]]{{.}} : memref<255x63xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_9_:%.+]] = affine.load [[RES_1_]][2] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_96_:%.+]] = vector.fma [[VAR_92_]], [[LOAD_PARAM_1_MEM_9_]], [[LOAD_RES_1_MEM_9_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_96_]], [[RES_1_]][2] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_97_:%.+]] = affine.apply [[MAP_1_]]([[I_4_]])
// CHECK-DAG:                 [[VAR_98_:%.+]] = arith.addi [[I_0_]], [[CST_2_]] : index
// CHECK:                     [[VAR_99_:%.+]] = arith.addi [[VAR_97_]], [[I_3_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_10_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_98_]], [[VAR_99_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                 [[VAR_101_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_10_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_102_:%.+]] = arith.addi [[VAR_97_]], [[I_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_1_MEM_10_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_102_]], [[I_2_]]{{.}} : memref<255x63xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_10_:%.+]] = affine.load [[RES_1_]][2] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_105_:%.+]] = vector.fma [[VAR_101_]], [[LOAD_PARAM_1_MEM_10_]], [[LOAD_RES_1_MEM_10_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_105_]], [[RES_1_]][2] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_106_:%.+]] = affine.apply [[MAP_2_]]([[I_4_]])
// CHECK-DAG:                 [[VAR_107_:%.+]] = arith.addi [[I_0_]], [[CST_2_]] : index
// CHECK:                     [[VAR_108_:%.+]] = arith.addi [[VAR_106_]], [[I_3_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_11_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_107_]], [[VAR_108_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                 [[VAR_110_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_11_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_111_:%.+]] = arith.addi [[VAR_106_]], [[I_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_1_MEM_11_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_111_]], [[I_2_]]{{.}} : memref<255x63xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_11_:%.+]] = affine.load [[RES_1_]][2] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_114_:%.+]] = vector.fma [[VAR_110_]], [[LOAD_PARAM_1_MEM_11_]], [[LOAD_RES_1_MEM_11_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_114_]], [[RES_1_]][2] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_115_:%.+]] = arith.addi [[I_0_]], [[CST_3_]] : index
// CHECK-DAG:                 [[VAR_116_:%.+]] = arith.addi [[I_4_]], [[I_3_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_12_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_115_]], [[VAR_116_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                 [[VAR_118_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_12_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_119_:%.+]] = arith.addi [[I_4_]], [[I_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_1_MEM_12_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_119_]], [[I_2_]]{{.}} : memref<255x63xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_12_:%.+]] = affine.load [[RES_1_]][3] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_122_:%.+]] = vector.fma [[VAR_118_]], [[LOAD_PARAM_1_MEM_12_]], [[LOAD_RES_1_MEM_12_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_122_]], [[RES_1_]][3] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_123_:%.+]] = affine.apply [[MAP_0_]]([[I_4_]])
// CHECK-DAG:                 [[VAR_124_:%.+]] = arith.addi [[I_0_]], [[CST_3_]] : index
// CHECK:                     [[VAR_125_:%.+]] = arith.addi [[VAR_123_]], [[I_3_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_13_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_124_]], [[VAR_125_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                 [[VAR_127_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_13_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_128_:%.+]] = arith.addi [[VAR_123_]], [[I_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_1_MEM_13_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_128_]], [[I_2_]]{{.}} : memref<255x63xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_13_:%.+]] = affine.load [[RES_1_]][3] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_131_:%.+]] = vector.fma [[VAR_127_]], [[LOAD_PARAM_1_MEM_13_]], [[LOAD_RES_1_MEM_13_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_131_]], [[RES_1_]][3] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_132_:%.+]] = affine.apply [[MAP_1_]]([[I_4_]])
// CHECK-DAG:                 [[VAR_133_:%.+]] = arith.addi [[I_0_]], [[CST_3_]] : index
// CHECK:                     [[VAR_134_:%.+]] = arith.addi [[VAR_132_]], [[I_3_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_14_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_133_]], [[VAR_134_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                 [[VAR_136_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_14_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_137_:%.+]] = arith.addi [[VAR_132_]], [[I_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_1_MEM_14_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_137_]], [[I_2_]]{{.}} : memref<255x63xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_14_:%.+]] = affine.load [[RES_1_]][3] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_140_:%.+]] = vector.fma [[VAR_136_]], [[LOAD_PARAM_1_MEM_14_]], [[LOAD_RES_1_MEM_14_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_140_]], [[RES_1_]][3] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_141_:%.+]] = affine.apply [[MAP_2_]]([[I_4_]])
// CHECK-DAG:                 [[VAR_142_:%.+]] = arith.addi [[I_0_]], [[CST_3_]] : index
// CHECK:                     [[VAR_143_:%.+]] = arith.addi [[VAR_141_]], [[I_3_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_15_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_142_]], [[VAR_143_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                 [[VAR_145_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_15_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_146_:%.+]] = arith.addi [[VAR_141_]], [[I_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_1_MEM_15_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_146_]], [[I_2_]]{{.}} : memref<255x63xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_15_:%.+]] = affine.load [[RES_1_]][3] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_149_:%.+]] = vector.fma [[VAR_145_]], [[LOAD_PARAM_1_MEM_15_]], [[LOAD_RES_1_MEM_15_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_149_]], [[RES_1_]][3] : memref<4xvector<8xf32>>
// CHECK:                   }
// CHECK:                   [[LOAD_RES_1_MEM_16_:%.+]] = affine.load [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK:                   vector.store [[LOAD_RES_1_MEM_16_]], [[RES_]]{{.}}[[I_0_]], [[I_2_]]{{.}} : memref<127x63xf32>, vector<8xf32>
// CHECK-DAG:               [[LOAD_RES_1_MEM_17_:%.+]] = affine.load [[RES_1_]][1] : memref<4xvector<8xf32>>
// CHECK-DAG:               [[VAR_9_:%.+]] = arith.addi [[I_0_]], [[CST_1_]] : index
// CHECK:                   vector.store [[LOAD_RES_1_MEM_17_]], [[RES_]]{{.}}[[VAR_9_]], [[I_2_]]{{.}} : memref<127x63xf32>, vector<8xf32>
// CHECK-DAG:               [[LOAD_RES_1_MEM_18_:%.+]] = affine.load [[RES_1_]][2] : memref<4xvector<8xf32>>
// CHECK-DAG:               [[VAR_11_:%.+]] = arith.addi [[I_0_]], [[CST_2_]] : index
// CHECK:                   vector.store [[LOAD_RES_1_MEM_18_]], [[RES_]]{{.}}[[VAR_11_]], [[I_2_]]{{.}} : memref<127x63xf32>, vector<8xf32>
// CHECK-DAG:               [[LOAD_RES_1_MEM_19_:%.+]] = affine.load [[RES_1_]][3] : memref<4xvector<8xf32>>
// CHECK-DAG:               [[VAR_13_:%.+]] = arith.addi [[I_0_]], [[CST_3_]] : index
// CHECK:                   vector.store [[LOAD_RES_1_MEM_19_]], [[RES_]]{{.}}[[VAR_13_]], [[I_2_]]{{.}} : memref<127x63xf32>, vector<8xf32>
// CHECK:                 } else {
// skip this check          affine.if [[set1_]]([[I_2_]]) {
// CHECK:                     affine.for [[I_5_:%.+]] = 0 to min [[MAP_3_]]([[I_0_]]) {
// CHECK:                       [[LOAD_RES_MEM_4_:%.+]] = arith.addi [[I_5_]], [[I_0_]] : index
// CHECK:                       [[LOAD_RES_MEM_5_:%.+]] = vector.load [[RES_]]{{.}}[[LOAD_RES_MEM_4_]], [[I_2_]]{{.}} : memref<127x63xf32>, vector<8xf32>
// CHECK:                       affine.store [[LOAD_RES_MEM_5_]], [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK:                       affine.for [[I_6_:%.+]] = 0 to min [[MAP_4_]]([[I_3_]]) {
// CHECK-DAG:                     [[LOAD_RES_MEM_2_:%.+]] = arith.addi [[I_5_]], [[I_0_]] : index
// CHECK-DAG:                     [[VAR_5_1_:%.+]] = arith.addi [[I_6_]], [[I_3_]] : index
// CHECK:                         [[LOAD_PARAM_0_MEM_16_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[LOAD_RES_MEM_2_]], [[VAR_5_1_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                     [[LOAD_RES_1_MEM_16_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_16_]] : f32 to vector<8xf32>
// CHECK-DAG:                     [[LOAD_RES_1_MEM_17_:%.+]] = arith.addi [[I_6_]], [[I_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                     [[LOAD_PARAM_1_MEM_16_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[LOAD_RES_1_MEM_17_]], [[I_2_]]{{.}} : memref<255x63xf32>, vector<8xf32>
// CHECK-DAG:                     [[LOAD_RES_1_MEM_20_:%.+]] = affine.load [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK:                         [[VAR_11_1_:%.+]] = vector.fma [[LOAD_RES_1_MEM_16_]], [[LOAD_PARAM_1_MEM_16_]], [[LOAD_RES_1_MEM_20_]] : vector<8xf32>
// CHECK:                         affine.store [[VAR_11_1_]], [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK:                       }
// CHECK-DAG:                   [[LOAD_RES_1_MEM_21_:%.+]] = affine.load [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK-DAG:                   [[VAR_3_1_:%.+]] = arith.addi [[I_5_]], [[I_0_]] : index
// CHECK:                       vector.store [[LOAD_RES_1_MEM_21_]], [[RES_]]{{.}}[[VAR_3_1_]], [[I_2_]]{{.}} : memref<127x63xf32>, vector<8xf32>
// CHECK:                     }
// CHECK:                   } else {
// CHECK:                     affine.for [[I_7_:%.+]] = 0 to min [[MAP_3_]]([[I_0_]]) {
// CHECK:                       affine.for [[I_8_:%.+]] = 0 to 7 {
// CHECK-DAG:                     [[LOAD_RES_MEM_4_:%.+]] = arith.addi [[I_7_]], [[I_0_]] : index
// CHECK-DAG:                     [[VAR_1_1_:%.+]] = arith.addi [[I_8_]], [[I_2_]] : index
// CHECK:                         [[LOAD_RES_MEM_6_:%.+]] = memref.load [[RES_]]{{.}}[[LOAD_RES_MEM_4_]], [[VAR_1_1_]]{{.}} : memref<127x63xf32>
// CHECK:                         affine.store [[LOAD_RES_MEM_6_]], [[RES_2_]][0] : memref<1xf32>
// CHECK:                         affine.for [[I_9_:%.+]] = 0 to min [[MAP_4_]]([[I_3_]]) {
// CHECK-DAG:                       [[LOAD_PARAM_0_MEM_16_:%.+]] = arith.addi [[I_7_]], [[I_0_]] : index
// CHECK-DAG:                       [[LOAD_RES_1_MEM_16_1_:%.+]] = arith.addi [[I_9_]], [[I_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                       [[LOAD_PARAM_0_MEM_17_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[LOAD_PARAM_0_MEM_16_]], [[LOAD_RES_1_MEM_16_1_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                       [[VAR_9_1_:%.+]] = arith.addi [[I_9_]], [[I_3_]] : index
// CHECK-DAG:                       [[LOAD_RES_1_MEM_20_:%.+]] = arith.addi [[I_8_]], [[I_2_]] : index
// CHECK:                           [[LOAD_PARAM_1_MEM_17_:%.+]] = memref.load [[PARAM_1_]]{{.}}[[VAR_9_1_]], [[LOAD_RES_1_MEM_20_]]{{.}} : memref<255x63xf32>
// CHECK-DAG:                       [[LOAD_RES_1_MEM_19_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_17_]], [[LOAD_PARAM_1_MEM_17_]] : f32
// CHECK-DAG:                       [[VAR_13_1_:%.+]] = affine.load [[RES_2_]][0] : memref<1xf32>
// CHECK:                           [[VAR_14_1_:%.+]] = arith.addf [[LOAD_RES_1_MEM_19_]], [[VAR_13_1_]] : f32
// CHECK:                           affine.store [[VAR_14_1_]], [[RES_2_]][0] : memref<1xf32>
// CHECK:                         }
// CHECK-DAG:                     [[VAR_3_1_:%.+]] = affine.load [[RES_2_]][0] : memref<1xf32>
// CHECK-DAG:                     [[LOAD_RES_MEM_2_1_:%.+]] = arith.addi [[I_7_]], [[I_0_]] : index
// CHECK-DAG:                     [[VAR_5_2_:%.+]] = arith.addi [[I_8_]], [[I_2_]] : index
// CHECK:                         memref.store [[VAR_3_1_]], [[RES_]]{{.}}[[LOAD_RES_MEM_2_1_]], [[VAR_5_2_]]{{.}} : memref<127x63xf32>
// CHECK:                       }
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<127x63xf32>
// CHECK:         }
}

