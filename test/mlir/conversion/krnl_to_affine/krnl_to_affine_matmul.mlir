
// RUN: onnx-mlir-opt -O3 --march=z16 --convert-krnl-to-affine --canonicalize %s -split-input-file | FileCheck %s

// -----

// matmul with perfect blocks, sequential

func.func @krnl_matmul_seq_perfect_blocks(%arg0: memref<128x256xf32>, %arg1: memref<256x512xf32>) -> (memref<128x512xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c512 = arith.constant 512 : index
  %alloc = memref.alloc() {alignment = 128 : i64} : memref<128x512xf32>
  krnl.memset %alloc, %cst : memref<128x512xf32>
  %0:3 = krnl.define_loops 3
  %loop_block, %loop_local = krnl.block %0#0 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  %loop_block_0, %loop_local_1 = krnl.block %loop_local 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  %loop_block_2, %loop_local_3 = krnl.block %0#1 64 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  %loop_block_4, %loop_local_5 = krnl.block %loop_local_3 16 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  %loop_block_6, %loop_local_7 = krnl.block %0#2 256 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  krnl.permute(%loop_block_2, %loop_block_4, %loop_local_5, %loop_block_6, %loop_local_7, %loop_block, %loop_block_0, %loop_local_1) [0, 3, 5, 1, 6, 2, 4, 7] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
  krnl.iterate(%loop_block_2, %loop_block_6) with (%0#1 -> %arg2 = 0 to 512, %0#2 -> %arg3 = 0 to 256, %0#0 -> %arg4 = 0 to 128){
    %1:2 = krnl.get_induction_var_value(%loop_block_2, %loop_block_6) : (!krnl.loop, !krnl.loop) -> (index, index)
    %alloc_8 = memref.alloc() {alignment = 128 : i64} : memref<32x256xf32>
    %alloc_9 = memref.alloc() {alignment = 128 : i64} : memref<256x64xf32>
    krnl.copy_to_tile_buffer %alloc_9, %arg1[%1#1, %1#0], %cst {padToNext = [], tileSize = []} : memref<256x64xf32>, memref<256x512xf32>
    krnl.iterate(%loop_block) with (){
      %2 = krnl.get_induction_var_value(%loop_block) : (!krnl.loop) -> index
      krnl.copy_to_tile_buffer %alloc_8, %arg0[%2, %1#1], %cst {padToNext = [], tileSize = []} : memref<32x256xf32>, memref<128x256xf32>
      krnl.iterate(%loop_block_4, %loop_block_0) with (){
        %3:2 = krnl.get_induction_var_value(%loop_block_4, %loop_block_0) : (!krnl.loop, !krnl.loop) -> (index, index)
        krnl.matmul %alloc_8[%2, %1#1], %alloc_9[%1#1, %1#0], %alloc[%c0, %c0], (%loop_local_1, %loop_local_5, %loop_local_7), (%3#1, %3#0, %1#1), (%c128, %c512, %c256) {aTileSize = [], bTileSize = [], cTileSize = [], computeTileSize = [4, 16, 256]} : memref<32x256xf32>, memref<256x64xf32>, memref<128x512xf32>, (!krnl.loop, !krnl.loop, !krnl.loop)
      }
    }
  }
  return %alloc : memref<128x512xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 64)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 + 32)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0, d1) -> (d0 - d1)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0) -> (d0 + 2)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0) -> (d0 + 3)>
// CHECK-LABEL:  func.func @krnl_matmul_seq_perfect_blocks
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<128x256xf32>, [[PARAM_1_:%.+]]: memref<256x512xf32>) -> memref<128x512xf32> attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<128x512xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 128 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 512 {
// CHECK:               affine.store [[CST_0_dot_000000_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<128x512xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           affine.for [[I_2_:%.+]] = 0 to 512 step 64 {
// CHECK:             affine.for [[I_3_:%.+]] = 0 to 256 step 256 {
// CHECK-DAG:           [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<32x256xf32>
// CHECK-DAG:           [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<256x64xf32>
// CHECK:               affine.for [[I_4_:%.+]] = 0 to 256 {
// CHECK:                 affine.for [[I_5_:%.+]] = 0 to 64 {
// CHECK:                   [[LOAD_PARAM_1_MEM_:%.+]] = affine.load [[PARAM_1_]]{{.}}[[I_4_]] + [[I_3_]], [[I_5_]] + [[I_2_]]{{.}} : memref<256x512xf32>
// CHECK:                   affine.store [[LOAD_PARAM_1_MEM_]], [[RES_2_]]{{.}}[[I_4_]], [[I_5_]]{{.}} : memref<256x64xf32>
// CHECK:                 }
// CHECK:               }
// CHECK:               affine.for [[I_6_:%.+]] = 0 to 128 step 32 {
// CHECK:                 affine.for [[I_7_:%.+]] = 0 to 32 {
// CHECK:                   affine.for [[I_8_:%.+]] = 0 to 256 {
// CHECK:                     [[LOAD_PARAM_1_MEM_1_:%.+]] = affine.load [[PARAM_0_]]{{.}}[[I_7_]] + [[I_6_]], [[I_8_]] + [[I_3_]]{{.}} : memref<128x256xf32>
// CHECK:                     affine.store [[LOAD_PARAM_1_MEM_1_]], [[RES_1_]]{{.}}[[I_7_]], [[I_8_]]{{.}} : memref<32x256xf32>
// CHECK:                   }
// CHECK:                 }
// CHECK:                 affine.for [[I_9_:%.+]] = [[MAP_0_]]([[I_2_]]) to [[MAP_1_]]([[I_2_]]) step 16 {
// CHECK:                   affine.for [[I_10_:%.+]] = [[MAP_0_]]([[I_6_]]) to [[MAP_2_]]([[I_6_]]) step 4 {
// CHECK-DAG:                 [[LOAD_PARAM_1_MEM_1_:%.+]] = affine.apply [[MAP_3_]]([[I_10_]], [[I_6_]])
// CHECK-DAG:                 [[VAR_1_:%.+]] = affine.apply [[MAP_3_]]([[I_9_]], [[I_2_]])
// CHECK-DAG:                 [[RES_3_:%.+]] = memref.alloca() {{.*}}: memref<4xvector<16xf32>>
// skip this one                affine.if [[SET_1_]]() {
// CHECK:                       [[LOAD_RES_MEM_:%.+]] = vector.load [[RES_]]{{.}}[[I_10_]], [[I_9_]]{{.}} : memref<128x512xf32>, vector<16xf32>
// CHECK:                       affine.store [[LOAD_RES_MEM_]], [[RES_3_]][0] : memref<4xvector<16xf32>>
// CHECK:                       [[VAR_3_:%.+]] = arith.addi [[I_10_]], [[CST_1_]] : index
// CHECK:                       [[LOAD_RES_MEM_1_:%.+]] = vector.load [[RES_]]{{.}}[[VAR_3_]], [[I_9_]]{{.}} : memref<128x512xf32>, vector<16xf32>
// CHECK:                       affine.store [[LOAD_RES_MEM_1_]], [[RES_3_]][1] : memref<4xvector<16xf32>>
// CHECK:                       [[VAR_5_:%.+]] = arith.addi [[I_10_]], [[CST_2_]] : index
// CHECK:                       [[LOAD_RES_MEM_2_:%.+]] = vector.load [[RES_]]{{.}}[[VAR_5_]], [[I_9_]]{{.}} : memref<128x512xf32>, vector<16xf32>
// CHECK:                       affine.store [[LOAD_RES_MEM_2_]], [[RES_3_]][2] : memref<4xvector<16xf32>>
// CHECK:                       [[VAR_7_:%.+]] = arith.addi [[I_10_]], [[CST_3_]] : index
// CHECK:                       [[LOAD_RES_MEM_3_:%.+]] = vector.load [[RES_]]{{.}}[[VAR_7_]], [[I_9_]]{{.}} : memref<128x512xf32>, vector<16xf32>
// CHECK:                       affine.store [[LOAD_RES_MEM_3_]], [[RES_3_]][3] : memref<4xvector<16xf32>>
// CHECK:                       affine.for [[I_11_:%.+]] = 0 to 256 step 4 {
// CHECK:                         [[LOAD_RES_1_MEM_:%.+]] = memref.load [[RES_1_]]{{.}}[[LOAD_PARAM_1_MEM_1_]], [[I_11_]]{{.}} : memref<32x256xf32>
// CHECK-DAG:                     [[VAR_17_:%.+]] = vector.broadcast [[LOAD_RES_1_MEM_]] : f32 to vector<16xf32>
// CHECK-DAG:                     [[LOAD_RES_2_MEM_:%.+]] = vector.load [[RES_2_]]{{.}}[[I_11_]], [[VAR_1_]]{{.}} : memref<256x64xf32>, vector<16xf32>
// CHECK-DAG:                     [[LOAD_RES_3_MEM_:%.+]] = affine.load [[RES_3_]][0] : memref<4xvector<16xf32>>
// CHECK:                         [[VAR_20_:%.+]] = vector.fma [[VAR_17_]], [[LOAD_RES_2_MEM_]], [[LOAD_RES_3_MEM_]] : vector<16xf32>
// CHECK:                         affine.store [[VAR_20_]], [[RES_3_]][0] : memref<4xvector<16xf32>>
// CHECK:                         [[VAR_21_:%.+]] = affine.apply [[MAP_4_]]([[I_11_]])
// CHECK:                         [[LOAD_RES_1_MEM_1_:%.+]] = memref.load [[RES_1_]]{{.}}[[LOAD_PARAM_1_MEM_1_]], [[VAR_21_]]{{.}} : memref<32x256xf32>
// CHECK-DAG:                     [[VAR_23_:%.+]] = vector.broadcast [[LOAD_RES_1_MEM_1_]] : f32 to vector<16xf32>
// CHECK-DAG:                     [[LOAD_RES_2_MEM_1_:%.+]] = vector.load [[RES_2_]]{{.}}[[VAR_21_]], [[VAR_1_]]{{.}} : memref<256x64xf32>, vector<16xf32>
// CHECK-DAG:                     [[LOAD_RES_3_MEM_1_:%.+]] = affine.load [[RES_3_]][0] : memref<4xvector<16xf32>>
// CHECK:                         [[VAR_26_:%.+]] = vector.fma [[VAR_23_]], [[LOAD_RES_2_MEM_1_]], [[LOAD_RES_3_MEM_1_]] : vector<16xf32>
// CHECK:                         affine.store [[VAR_26_]], [[RES_3_]][0] : memref<4xvector<16xf32>>
// CHECK:                         [[VAR_27_:%.+]] = affine.apply [[MAP_5_]]([[I_11_]])
// CHECK:                         [[LOAD_RES_1_MEM_2_:%.+]] = memref.load [[RES_1_]]{{.}}[[LOAD_PARAM_1_MEM_1_]], [[VAR_27_]]{{.}} : memref<32x256xf32>
// CHECK-DAG:                     [[VAR_29_:%.+]] = vector.broadcast [[LOAD_RES_1_MEM_2_]] : f32 to vector<16xf32>
// CHECK-DAG:                     [[LOAD_RES_2_MEM_2_:%.+]] = vector.load [[RES_2_]]{{.}}[[VAR_27_]], [[VAR_1_]]{{.}} : memref<256x64xf32>, vector<16xf32>
// CHECK-DAG:                     [[LOAD_RES_3_MEM_2_:%.+]] = affine.load [[RES_3_]][0] : memref<4xvector<16xf32>>
// CHECK:                         [[VAR_32_:%.+]] = vector.fma [[VAR_29_]], [[LOAD_RES_2_MEM_2_]], [[LOAD_RES_3_MEM_2_]] : vector<16xf32>
// CHECK:                         affine.store [[VAR_32_]], [[RES_3_]][0] : memref<4xvector<16xf32>>
// CHECK:                         [[VAR_33_:%.+]] = affine.apply [[MAP_6_]]([[I_11_]])
// CHECK:                         [[LOAD_RES_1_MEM_3_:%.+]] = memref.load [[RES_1_]]{{.}}[[LOAD_PARAM_1_MEM_1_]], [[VAR_33_]]{{.}} : memref<32x256xf32>
// CHECK-DAG:                     [[VAR_35_:%.+]] = vector.broadcast [[LOAD_RES_1_MEM_3_]] : f32 to vector<16xf32>
// CHECK-DAG:                     [[LOAD_RES_2_MEM_3_:%.+]] = vector.load [[RES_2_]]{{.}}[[VAR_33_]], [[VAR_1_]]{{.}} : memref<256x64xf32>, vector<16xf32>
// CHECK-DAG:                     [[LOAD_RES_3_MEM_3_:%.+]] = affine.load [[RES_3_]][0] : memref<4xvector<16xf32>>
// CHECK:                         [[VAR_38_:%.+]] = vector.fma [[VAR_35_]], [[LOAD_RES_2_MEM_3_]], [[LOAD_RES_3_MEM_3_]] : vector<16xf32>
// CHECK:                         affine.store [[VAR_38_]], [[RES_3_]][0] : memref<4xvector<16xf32>>
// CHECK:                         [[VAR_39_:%.+]] = arith.addi [[LOAD_PARAM_1_MEM_1_]], [[CST_1_]] : index
// CHECK:                         [[LOAD_RES_1_MEM_4_:%.+]] = memref.load [[RES_1_]]{{.}}[[VAR_39_]], [[I_11_]]{{.}} : memref<32x256xf32>
// CHECK-DAG:                     [[VAR_41_:%.+]] = vector.broadcast [[LOAD_RES_1_MEM_4_]] : f32 to vector<16xf32>
// CHECK-DAG:                     [[LOAD_RES_2_MEM_4_:%.+]] = vector.load [[RES_2_]]{{.}}[[I_11_]], [[VAR_1_]]{{.}} : memref<256x64xf32>, vector<16xf32>
// CHECK-DAG:                     [[LOAD_RES_3_MEM_4_:%.+]] = affine.load [[RES_3_]][1] : memref<4xvector<16xf32>>
// CHECK:                         [[VAR_44_:%.+]] = vector.fma [[VAR_41_]], [[LOAD_RES_2_MEM_4_]], [[LOAD_RES_3_MEM_4_]] : vector<16xf32>
// CHECK:                         affine.store [[VAR_44_]], [[RES_3_]][1] : memref<4xvector<16xf32>>
// CHECK-DAG:                     [[VAR_45_:%.+]] = affine.apply [[MAP_4_]]([[I_11_]])
// CHECK-DAG:                     [[VAR_46_:%.+]] = arith.addi [[LOAD_PARAM_1_MEM_1_]], [[CST_1_]] : index
// CHECK:                         [[LOAD_RES_1_MEM_5_:%.+]] = memref.load [[RES_1_]]{{.}}[[VAR_46_]], [[VAR_45_]]{{.}} : memref<32x256xf32>
// CHECK-DAG:                     [[VAR_48_:%.+]] = vector.broadcast [[LOAD_RES_1_MEM_5_]] : f32 to vector<16xf32>
// CHECK-DAG:                     [[LOAD_RES_2_MEM_5_:%.+]] = vector.load [[RES_2_]]{{.}}[[VAR_45_]], [[VAR_1_]]{{.}} : memref<256x64xf32>, vector<16xf32>
// CHECK-DAG:                     [[LOAD_RES_3_MEM_5_:%.+]] = affine.load [[RES_3_]][1] : memref<4xvector<16xf32>>
// CHECK:                         [[VAR_51_:%.+]] = vector.fma [[VAR_48_]], [[LOAD_RES_2_MEM_5_]], [[LOAD_RES_3_MEM_5_]] : vector<16xf32>
// CHECK:                         affine.store [[VAR_51_]], [[RES_3_]][1] : memref<4xvector<16xf32>>
// CHECK-DAG:                     [[VAR_52_:%.+]] = affine.apply [[MAP_5_]]([[I_11_]])
// CHECK-DAG:                     [[VAR_53_:%.+]] = arith.addi [[LOAD_PARAM_1_MEM_1_]], [[CST_1_]] : index
// CHECK:                         [[LOAD_RES_1_MEM_6_:%.+]] = memref.load [[RES_1_]]{{.}}[[VAR_53_]], [[VAR_52_]]{{.}} : memref<32x256xf32>
// CHECK-DAG:                     [[VAR_55_:%.+]] = vector.broadcast [[LOAD_RES_1_MEM_6_]] : f32 to vector<16xf32>
// CHECK-DAG:                     [[LOAD_RES_2_MEM_6_:%.+]] = vector.load [[RES_2_]]{{.}}[[VAR_52_]], [[VAR_1_]]{{.}} : memref<256x64xf32>, vector<16xf32>
// CHECK-DAG:                     [[LOAD_RES_3_MEM_6_:%.+]] = affine.load [[RES_3_]][1] : memref<4xvector<16xf32>>
// CHECK:                         [[VAR_58_:%.+]] = vector.fma [[VAR_55_]], [[LOAD_RES_2_MEM_6_]], [[LOAD_RES_3_MEM_6_]] : vector<16xf32>
// CHECK:                         affine.store [[VAR_58_]], [[RES_3_]][1] : memref<4xvector<16xf32>>
// CHECK-DAG:                     [[VAR_59_:%.+]] = affine.apply [[MAP_6_]]([[I_11_]])
// CHECK-DAG:                     [[VAR_60_:%.+]] = arith.addi [[LOAD_PARAM_1_MEM_1_]], [[CST_1_]] : index
// CHECK:                         [[LOAD_RES_1_MEM_7_:%.+]] = memref.load [[RES_1_]]{{.}}[[VAR_60_]], [[VAR_59_]]{{.}} : memref<32x256xf32>
// CHECK-DAG:                     [[VAR_62_:%.+]] = vector.broadcast [[LOAD_RES_1_MEM_7_]] : f32 to vector<16xf32>
// CHECK-DAG:                     [[LOAD_RES_2_MEM_7_:%.+]] = vector.load [[RES_2_]]{{.}}[[VAR_59_]], [[VAR_1_]]{{.}} : memref<256x64xf32>, vector<16xf32>
// CHECK-DAG:                     [[LOAD_RES_3_MEM_7_:%.+]] = affine.load [[RES_3_]][1] : memref<4xvector<16xf32>>
// CHECK:                         [[VAR_65_:%.+]] = vector.fma [[VAR_62_]], [[LOAD_RES_2_MEM_7_]], [[LOAD_RES_3_MEM_7_]] : vector<16xf32>
// CHECK:                         affine.store [[VAR_65_]], [[RES_3_]][1] : memref<4xvector<16xf32>>
// CHECK:                         [[VAR_66_:%.+]] = arith.addi [[LOAD_PARAM_1_MEM_1_]], [[CST_2_]] : index
// CHECK:                         [[LOAD_RES_1_MEM_8_:%.+]] = memref.load [[RES_1_]]{{.}}[[VAR_66_]], [[I_11_]]{{.}} : memref<32x256xf32>
// CHECK-DAG:                     [[VAR_68_:%.+]] = vector.broadcast [[LOAD_RES_1_MEM_8_]] : f32 to vector<16xf32>
// CHECK-DAG:                     [[LOAD_RES_2_MEM_8_:%.+]] = vector.load [[RES_2_]]{{.}}[[I_11_]], [[VAR_1_]]{{.}} : memref<256x64xf32>, vector<16xf32>
// CHECK-DAG:                     [[LOAD_RES_3_MEM_8_:%.+]] = affine.load [[RES_3_]][2] : memref<4xvector<16xf32>>
// CHECK:                         [[VAR_71_:%.+]] = vector.fma [[VAR_68_]], [[LOAD_RES_2_MEM_8_]], [[LOAD_RES_3_MEM_8_]] : vector<16xf32>
// CHECK:                         affine.store [[VAR_71_]], [[RES_3_]][2] : memref<4xvector<16xf32>>
// CHECK-DAG:                     [[VAR_72_:%.+]] = affine.apply [[MAP_4_]]([[I_11_]])
// CHECK-DAG:                     [[VAR_73_:%.+]] = arith.addi [[LOAD_PARAM_1_MEM_1_]], [[CST_2_]] : index
// CHECK:                         [[LOAD_RES_1_MEM_9_:%.+]] = memref.load [[RES_1_]]{{.}}[[VAR_73_]], [[VAR_72_]]{{.}} : memref<32x256xf32>
// CHECK-DAG:                     [[VAR_75_:%.+]] = vector.broadcast [[LOAD_RES_1_MEM_9_]] : f32 to vector<16xf32>
// CHECK-DAG:                     [[LOAD_RES_2_MEM_9_:%.+]] = vector.load [[RES_2_]]{{.}}[[VAR_72_]], [[VAR_1_]]{{.}} : memref<256x64xf32>, vector<16xf32>
// CHECK-DAG:                     [[LOAD_RES_3_MEM_9_:%.+]] = affine.load [[RES_3_]][2] : memref<4xvector<16xf32>>
// CHECK:                         [[VAR_78_:%.+]] = vector.fma [[VAR_75_]], [[LOAD_RES_2_MEM_9_]], [[LOAD_RES_3_MEM_9_]] : vector<16xf32>
// CHECK:                         affine.store [[VAR_78_]], [[RES_3_]][2] : memref<4xvector<16xf32>>
// CHECK-DAG:                     [[VAR_79_:%.+]] = affine.apply [[MAP_5_]]([[I_11_]])
// CHECK-DAG:                     [[VAR_80_:%.+]] = arith.addi [[LOAD_PARAM_1_MEM_1_]], [[CST_2_]] : index
// CHECK:                         [[LOAD_RES_1_MEM_10_:%.+]] = memref.load [[RES_1_]]{{.}}[[VAR_80_]], [[VAR_79_]]{{.}} : memref<32x256xf32>
// CHECK-DAG:                     [[VAR_82_:%.+]] = vector.broadcast [[LOAD_RES_1_MEM_10_]] : f32 to vector<16xf32>
// CHECK-DAG:                     [[LOAD_RES_2_MEM_10_:%.+]] = vector.load [[RES_2_]]{{.}}[[VAR_79_]], [[VAR_1_]]{{.}} : memref<256x64xf32>, vector<16xf32>
// CHECK-DAG:                     [[LOAD_RES_3_MEM_10_:%.+]] = affine.load [[RES_3_]][2] : memref<4xvector<16xf32>>
// CHECK:                         [[VAR_85_:%.+]] = vector.fma [[VAR_82_]], [[LOAD_RES_2_MEM_10_]], [[LOAD_RES_3_MEM_10_]] : vector<16xf32>
// CHECK:                         affine.store [[VAR_85_]], [[RES_3_]][2] : memref<4xvector<16xf32>>
// CHECK-DAG:                     [[VAR_86_:%.+]] = affine.apply [[MAP_6_]]([[I_11_]])
// CHECK-DAG:                     [[VAR_87_:%.+]] = arith.addi [[LOAD_PARAM_1_MEM_1_]], [[CST_2_]] : index
// CHECK:                         [[LOAD_RES_1_MEM_11_:%.+]] = memref.load [[RES_1_]]{{.}}[[VAR_87_]], [[VAR_86_]]{{.}} : memref<32x256xf32>
// CHECK-DAG:                     [[VAR_89_:%.+]] = vector.broadcast [[LOAD_RES_1_MEM_11_]] : f32 to vector<16xf32>
// CHECK-DAG:                     [[LOAD_RES_2_MEM_11_:%.+]] = vector.load [[RES_2_]]{{.}}[[VAR_86_]], [[VAR_1_]]{{.}} : memref<256x64xf32>, vector<16xf32>
// CHECK-DAG:                     [[LOAD_RES_3_MEM_11_:%.+]] = affine.load [[RES_3_]][2] : memref<4xvector<16xf32>>
// CHECK:                         [[VAR_92_:%.+]] = vector.fma [[VAR_89_]], [[LOAD_RES_2_MEM_11_]], [[LOAD_RES_3_MEM_11_]] : vector<16xf32>
// CHECK:                         affine.store [[VAR_92_]], [[RES_3_]][2] : memref<4xvector<16xf32>>
// CHECK:                         [[VAR_93_:%.+]] = arith.addi [[LOAD_PARAM_1_MEM_1_]], [[CST_3_]] : index
// CHECK:                         [[LOAD_RES_1_MEM_12_:%.+]] = memref.load [[RES_1_]]{{.}}[[VAR_93_]], [[I_11_]]{{.}} : memref<32x256xf32>
// CHECK-DAG:                     [[VAR_95_:%.+]] = vector.broadcast [[LOAD_RES_1_MEM_12_]] : f32 to vector<16xf32>
// CHECK-DAG:                     [[LOAD_RES_2_MEM_12_:%.+]] = vector.load [[RES_2_]]{{.}}[[I_11_]], [[VAR_1_]]{{.}} : memref<256x64xf32>, vector<16xf32>
// CHECK-DAG:                     [[LOAD_RES_3_MEM_12_:%.+]] = affine.load [[RES_3_]][3] : memref<4xvector<16xf32>>
// CHECK:                         [[VAR_98_:%.+]] = vector.fma [[VAR_95_]], [[LOAD_RES_2_MEM_12_]], [[LOAD_RES_3_MEM_12_]] : vector<16xf32>
// CHECK:                         affine.store [[VAR_98_]], [[RES_3_]][3] : memref<4xvector<16xf32>>
// CHECK-DAG:                     [[VAR_99_:%.+]] = affine.apply [[MAP_4_]]([[I_11_]])
// CHECK-DAG:                     [[VAR_100_:%.+]] = arith.addi [[LOAD_PARAM_1_MEM_1_]], [[CST_3_]] : index
// CHECK:                         [[LOAD_RES_1_MEM_13_:%.+]] = memref.load [[RES_1_]]{{.}}[[VAR_100_]], [[VAR_99_]]{{.}} : memref<32x256xf32>
// CHECK-DAG:                     [[VAR_102_:%.+]] = vector.broadcast [[LOAD_RES_1_MEM_13_]] : f32 to vector<16xf32>
// CHECK-DAG:                     [[LOAD_RES_2_MEM_13_:%.+]] = vector.load [[RES_2_]]{{.}}[[VAR_99_]], [[VAR_1_]]{{.}} : memref<256x64xf32>, vector<16xf32>
// CHECK-DAG:                     [[LOAD_RES_3_MEM_13_:%.+]] = affine.load [[RES_3_]][3] : memref<4xvector<16xf32>>
// CHECK:                         [[VAR_105_:%.+]] = vector.fma [[VAR_102_]], [[LOAD_RES_2_MEM_13_]], [[LOAD_RES_3_MEM_13_]] : vector<16xf32>
// CHECK:                         affine.store [[VAR_105_]], [[RES_3_]][3] : memref<4xvector<16xf32>>
// CHECK-DAG:                     [[VAR_106_:%.+]] = affine.apply [[MAP_5_]]([[I_11_]])
// CHECK-DAG:                     [[VAR_107_:%.+]] = arith.addi [[LOAD_PARAM_1_MEM_1_]], [[CST_3_]] : index
// CHECK:                         [[LOAD_RES_1_MEM_14_:%.+]] = memref.load [[RES_1_]]{{.}}[[VAR_107_]], [[VAR_106_]]{{.}} : memref<32x256xf32>
// CHECK-DAG:                     [[VAR_109_:%.+]] = vector.broadcast [[LOAD_RES_1_MEM_14_]] : f32 to vector<16xf32>
// CHECK-DAG:                     [[LOAD_RES_2_MEM_14_:%.+]] = vector.load [[RES_2_]]{{.}}[[VAR_106_]], [[VAR_1_]]{{.}} : memref<256x64xf32>, vector<16xf32>
// CHECK-DAG:                     [[LOAD_RES_3_MEM_14_:%.+]] = affine.load [[RES_3_]][3] : memref<4xvector<16xf32>>
// CHECK:                         [[VAR_112_:%.+]] = vector.fma [[VAR_109_]], [[LOAD_RES_2_MEM_14_]], [[LOAD_RES_3_MEM_14_]] : vector<16xf32>
// CHECK:                         affine.store [[VAR_112_]], [[RES_3_]][3] : memref<4xvector<16xf32>>
// CHECK-DAG:                     [[VAR_113_:%.+]] = affine.apply [[MAP_6_]]([[I_11_]])
// CHECK-DAG:                     [[VAR_114_:%.+]] = arith.addi [[LOAD_PARAM_1_MEM_1_]], [[CST_3_]] : index
// CHECK:                         [[LOAD_RES_1_MEM_15_:%.+]] = memref.load [[RES_1_]]{{.}}[[VAR_114_]], [[VAR_113_]]{{.}} : memref<32x256xf32>
// CHECK-DAG:                     [[VAR_116_:%.+]] = vector.broadcast [[LOAD_RES_1_MEM_15_]] : f32 to vector<16xf32>
// CHECK-DAG:                     [[LOAD_RES_2_MEM_15_:%.+]] = vector.load [[RES_2_]]{{.}}[[VAR_113_]], [[VAR_1_]]{{.}} : memref<256x64xf32>, vector<16xf32>
// CHECK-DAG:                     [[LOAD_RES_3_MEM_15_:%.+]] = affine.load [[RES_3_]][3] : memref<4xvector<16xf32>>
// CHECK:                         [[VAR_119_:%.+]] = vector.fma [[VAR_116_]], [[LOAD_RES_2_MEM_15_]], [[LOAD_RES_3_MEM_15_]] : vector<16xf32>
// CHECK:                         affine.store [[VAR_119_]], [[RES_3_]][3] : memref<4xvector<16xf32>>
// CHECK:                       }
// CHECK:                       [[LOAD_RES_3_MEM_16_:%.+]] = affine.load [[RES_3_]][0] : memref<4xvector<16xf32>>
// CHECK:                       vector.store [[LOAD_RES_3_MEM_16_]], [[RES_]]{{.}}[[I_10_]], [[I_9_]]{{.}} : memref<128x512xf32>, vector<16xf32>
// CHECK-DAG:                   [[LOAD_RES_3_MEM_17_:%.+]] = affine.load [[RES_3_]][1] : memref<4xvector<16xf32>>
// CHECK-DAG:                   [[VAR_11_:%.+]] = arith.addi [[I_10_]], [[CST_1_]] : index
// CHECK:                       vector.store [[LOAD_RES_3_MEM_17_]], [[RES_]]{{.}}[[VAR_11_]], [[I_9_]]{{.}} : memref<128x512xf32>, vector<16xf32>
// CHECK-DAG:                   [[LOAD_RES_3_MEM_18_:%.+]] = affine.load [[RES_3_]][2] : memref<4xvector<16xf32>>
// CHECK-DAG:                   [[VAR_13_:%.+]] = arith.addi [[I_10_]], [[CST_2_]] : index
// CHECK:                       vector.store [[LOAD_RES_3_MEM_18_]], [[RES_]]{{.}}[[VAR_13_]], [[I_9_]]{{.}} : memref<128x512xf32>, vector<16xf32>
// CHECK-DAG:                   [[LOAD_RES_3_MEM_19_:%.+]] = affine.load [[RES_3_]][3] : memref<4xvector<16xf32>>
// CHECK-DAG:                   [[VAR_15_:%.+]] = arith.addi [[I_10_]], [[CST_3_]] : index
// CHECK:                       vector.store [[LOAD_RES_3_MEM_19_]], [[RES_]]{{.}}[[VAR_15_]], [[I_9_]]{{.}} : memref<128x512xf32>, vector<16xf32>
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<128x512xf32>
// CHECK:         }
}

// -----


func.func @krnl_matmul_seq_partial_blocks(%arg0: memref<127x255xf32> {onnx.name = "x"}, %arg1: memref<255x63xf32> {onnx.name = "y"}) -> (memref<127x63xf32> {onnx.name = "y"}) {
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
// CHECK-LABEL:  func.func @krnl_matmul_seq_partial_blocks
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
// CHECK:           affine.for [[I_2_:%.+]] = 0 to 127 step 4 {
// CHECK:             affine.for [[I_3_:%.+]] = 0 to 63 step 8 {
// CHECK:               affine.for [[I_4_:%.+]] = 0 to 255 step 8 {
// CHECK-DAG:             [[RES_1_:%.+]] = memref.alloca() {{.*}}: memref<4xvector<8xf32>>
// CHECK-DAG:             [[RES_2_:%.+]] = memref.alloca() {{.*}}: memref<1xf32>
// skip this one            affine.if [[set_]]([[I_2_]], [[I_3_]], [[I_4_]]) {
// CHECK:                   [[LOAD_RES_MEM_:%.+]] = vector.load [[RES_]]{{.}}[[I_2_]], [[I_3_]]{{.}} : memref<127x63xf32>, vector<8xf32>
// CHECK:                   affine.store [[LOAD_RES_MEM_]], [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK:                   [[VAR_1_:%.+]] = arith.addi [[I_2_]], [[CST_1_]] : index
// CHECK:                   [[LOAD_RES_MEM_1_:%.+]] = vector.load [[RES_]]{{.}}[[VAR_1_]], [[I_3_]]{{.}} : memref<127x63xf32>, vector<8xf32>
// CHECK:                   affine.store [[LOAD_RES_MEM_1_]], [[RES_1_]][1] : memref<4xvector<8xf32>>
// CHECK:                   [[VAR_3_:%.+]] = arith.addi [[I_2_]], [[CST_2_]] : index
// CHECK:                   [[LOAD_RES_MEM_2_:%.+]] = vector.load [[RES_]]{{.}}[[VAR_3_]], [[I_3_]]{{.}} : memref<127x63xf32>, vector<8xf32>
// CHECK:                   affine.store [[LOAD_RES_MEM_2_]], [[RES_1_]][2] : memref<4xvector<8xf32>>
// CHECK:                   [[VAR_5_:%.+]] = arith.addi [[I_2_]], [[CST_3_]] : index
// CHECK:                   [[LOAD_RES_MEM_3_:%.+]] = vector.load [[RES_]]{{.}}[[VAR_5_]], [[I_3_]]{{.}} : memref<127x63xf32>, vector<8xf32>
// CHECK:                   affine.store [[LOAD_RES_MEM_3_]], [[RES_1_]][3] : memref<4xvector<8xf32>>
// CHECK:                   affine.for [[I_5_:%.+]] = 0 to 8 step 4 {
// CHECK:                     [[VAR_14_:%.+]] = arith.addi [[I_5_]], [[I_4_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_14_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                 [[VAR_16_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_17_:%.+]] = arith.addi [[I_5_]], [[I_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_1_MEM_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_17_]], [[I_3_]]{{.}} : memref<255x63xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_:%.+]] = affine.load [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_20_:%.+]] = vector.fma [[VAR_16_]], [[LOAD_PARAM_1_MEM_]], [[LOAD_RES_1_MEM_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_20_]], [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_21_:%.+]] = affine.apply [[MAP_0_]]([[I_5_]])
// CHECK:                     [[VAR_22_:%.+]] = arith.addi [[VAR_21_]], [[I_4_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_1_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_22_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                 [[VAR_24_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_1_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_25_:%.+]] = arith.addi [[VAR_21_]], [[I_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_1_MEM_1_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_25_]], [[I_3_]]{{.}} : memref<255x63xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_1_:%.+]] = affine.load [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_28_:%.+]] = vector.fma [[VAR_24_]], [[LOAD_PARAM_1_MEM_1_]], [[LOAD_RES_1_MEM_1_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_28_]], [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_29_:%.+]] = affine.apply [[MAP_1_]]([[I_5_]])
// CHECK:                     [[VAR_30_:%.+]] = arith.addi [[VAR_29_]], [[I_4_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_2_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_30_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                 [[VAR_32_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_2_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_33_:%.+]] = arith.addi [[VAR_29_]], [[I_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_1_MEM_2_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_33_]], [[I_3_]]{{.}} : memref<255x63xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_2_:%.+]] = affine.load [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_36_:%.+]] = vector.fma [[VAR_32_]], [[LOAD_PARAM_1_MEM_2_]], [[LOAD_RES_1_MEM_2_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_36_]], [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_37_:%.+]] = affine.apply [[MAP_2_]]([[I_5_]])
// CHECK:                     [[VAR_38_:%.+]] = arith.addi [[VAR_37_]], [[I_4_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_3_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_38_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                 [[VAR_40_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_3_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_41_:%.+]] = arith.addi [[VAR_37_]], [[I_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_1_MEM_3_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_41_]], [[I_3_]]{{.}} : memref<255x63xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_3_:%.+]] = affine.load [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_44_:%.+]] = vector.fma [[VAR_40_]], [[LOAD_PARAM_1_MEM_3_]], [[LOAD_RES_1_MEM_3_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_44_]], [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_45_:%.+]] = arith.addi [[I_2_]], [[CST_1_]] : index
// CHECK-DAG:                 [[VAR_46_:%.+]] = arith.addi [[I_5_]], [[I_4_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_4_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_45_]], [[VAR_46_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                 [[VAR_48_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_4_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_49_:%.+]] = arith.addi [[I_5_]], [[I_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_1_MEM_4_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_49_]], [[I_3_]]{{.}} : memref<255x63xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_4_:%.+]] = affine.load [[RES_1_]][1] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_52_:%.+]] = vector.fma [[VAR_48_]], [[LOAD_PARAM_1_MEM_4_]], [[LOAD_RES_1_MEM_4_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_52_]], [[RES_1_]][1] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_53_:%.+]] = affine.apply [[MAP_0_]]([[I_5_]])
// CHECK-DAG:                 [[VAR_54_:%.+]] = arith.addi [[I_2_]], [[CST_1_]] : index
// CHECK:                     [[VAR_55_:%.+]] = arith.addi [[VAR_53_]], [[I_4_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_5_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_54_]], [[VAR_55_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                 [[VAR_57_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_5_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_58_:%.+]] = arith.addi [[VAR_53_]], [[I_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_1_MEM_5_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_58_]], [[I_3_]]{{.}} : memref<255x63xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_5_:%.+]] = affine.load [[RES_1_]][1] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_61_:%.+]] = vector.fma [[VAR_57_]], [[LOAD_PARAM_1_MEM_5_]], [[LOAD_RES_1_MEM_5_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_61_]], [[RES_1_]][1] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_62_:%.+]] = affine.apply [[MAP_1_]]([[I_5_]])
// CHECK-DAG:                 [[VAR_63_:%.+]] = arith.addi [[I_2_]], [[CST_1_]] : index
// CHECK:                     [[VAR_64_:%.+]] = arith.addi [[VAR_62_]], [[I_4_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_6_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_63_]], [[VAR_64_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                 [[VAR_66_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_6_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_67_:%.+]] = arith.addi [[VAR_62_]], [[I_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_1_MEM_6_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_67_]], [[I_3_]]{{.}} : memref<255x63xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_6_:%.+]] = affine.load [[RES_1_]][1] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_70_:%.+]] = vector.fma [[VAR_66_]], [[LOAD_PARAM_1_MEM_6_]], [[LOAD_RES_1_MEM_6_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_70_]], [[RES_1_]][1] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_71_:%.+]] = affine.apply [[MAP_2_]]([[I_5_]])
// CHECK-DAG:                 [[VAR_72_:%.+]] = arith.addi [[I_2_]], [[CST_1_]] : index
// CHECK:                     [[VAR_73_:%.+]] = arith.addi [[VAR_71_]], [[I_4_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_7_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_72_]], [[VAR_73_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                 [[VAR_75_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_7_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_76_:%.+]] = arith.addi [[VAR_71_]], [[I_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_1_MEM_7_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_76_]], [[I_3_]]{{.}} : memref<255x63xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_7_:%.+]] = affine.load [[RES_1_]][1] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_79_:%.+]] = vector.fma [[VAR_75_]], [[LOAD_PARAM_1_MEM_7_]], [[LOAD_RES_1_MEM_7_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_79_]], [[RES_1_]][1] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_80_:%.+]] = arith.addi [[I_2_]], [[CST_2_]] : index
// CHECK-DAG:                 [[VAR_81_:%.+]] = arith.addi [[I_5_]], [[I_4_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_8_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_80_]], [[VAR_81_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                 [[VAR_83_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_8_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_84_:%.+]] = arith.addi [[I_5_]], [[I_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_1_MEM_8_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_84_]], [[I_3_]]{{.}} : memref<255x63xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_8_:%.+]] = affine.load [[RES_1_]][2] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_87_:%.+]] = vector.fma [[VAR_83_]], [[LOAD_PARAM_1_MEM_8_]], [[LOAD_RES_1_MEM_8_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_87_]], [[RES_1_]][2] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_88_:%.+]] = affine.apply [[MAP_0_]]([[I_5_]])
// CHECK-DAG:                 [[VAR_89_:%.+]] = arith.addi [[I_2_]], [[CST_2_]] : index
// CHECK:                     [[VAR_90_:%.+]] = arith.addi [[VAR_88_]], [[I_4_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_9_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_89_]], [[VAR_90_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                 [[VAR_92_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_9_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_93_:%.+]] = arith.addi [[VAR_88_]], [[I_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_1_MEM_9_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_93_]], [[I_3_]]{{.}} : memref<255x63xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_9_:%.+]] = affine.load [[RES_1_]][2] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_96_:%.+]] = vector.fma [[VAR_92_]], [[LOAD_PARAM_1_MEM_9_]], [[LOAD_RES_1_MEM_9_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_96_]], [[RES_1_]][2] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_97_:%.+]] = affine.apply [[MAP_1_]]([[I_5_]])
// CHECK-DAG:                 [[VAR_98_:%.+]] = arith.addi [[I_2_]], [[CST_2_]] : index
// CHECK:                     [[VAR_99_:%.+]] = arith.addi [[VAR_97_]], [[I_4_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_10_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_98_]], [[VAR_99_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                 [[VAR_101_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_10_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_102_:%.+]] = arith.addi [[VAR_97_]], [[I_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_1_MEM_10_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_102_]], [[I_3_]]{{.}} : memref<255x63xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_10_:%.+]] = affine.load [[RES_1_]][2] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_105_:%.+]] = vector.fma [[VAR_101_]], [[LOAD_PARAM_1_MEM_10_]], [[LOAD_RES_1_MEM_10_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_105_]], [[RES_1_]][2] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_106_:%.+]] = affine.apply [[MAP_2_]]([[I_5_]])
// CHECK-DAG:                 [[VAR_107_:%.+]] = arith.addi [[I_2_]], [[CST_2_]] : index
// CHECK:                     [[VAR_108_:%.+]] = arith.addi [[VAR_106_]], [[I_4_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_11_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_107_]], [[VAR_108_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                 [[VAR_110_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_11_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_111_:%.+]] = arith.addi [[VAR_106_]], [[I_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_1_MEM_11_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_111_]], [[I_3_]]{{.}} : memref<255x63xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_11_:%.+]] = affine.load [[RES_1_]][2] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_114_:%.+]] = vector.fma [[VAR_110_]], [[LOAD_PARAM_1_MEM_11_]], [[LOAD_RES_1_MEM_11_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_114_]], [[RES_1_]][2] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_115_:%.+]] = arith.addi [[I_2_]], [[CST_3_]] : index
// CHECK-DAG:                 [[VAR_116_:%.+]] = arith.addi [[I_5_]], [[I_4_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_12_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_115_]], [[VAR_116_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                 [[VAR_118_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_12_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_119_:%.+]] = arith.addi [[I_5_]], [[I_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_1_MEM_12_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_119_]], [[I_3_]]{{.}} : memref<255x63xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_12_:%.+]] = affine.load [[RES_1_]][3] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_122_:%.+]] = vector.fma [[VAR_118_]], [[LOAD_PARAM_1_MEM_12_]], [[LOAD_RES_1_MEM_12_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_122_]], [[RES_1_]][3] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_123_:%.+]] = affine.apply [[MAP_0_]]([[I_5_]])
// CHECK-DAG:                 [[VAR_124_:%.+]] = arith.addi [[I_2_]], [[CST_3_]] : index
// CHECK:                     [[VAR_125_:%.+]] = arith.addi [[VAR_123_]], [[I_4_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_13_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_124_]], [[VAR_125_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                 [[VAR_127_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_13_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_128_:%.+]] = arith.addi [[VAR_123_]], [[I_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_1_MEM_13_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_128_]], [[I_3_]]{{.}} : memref<255x63xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_13_:%.+]] = affine.load [[RES_1_]][3] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_131_:%.+]] = vector.fma [[VAR_127_]], [[LOAD_PARAM_1_MEM_13_]], [[LOAD_RES_1_MEM_13_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_131_]], [[RES_1_]][3] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_132_:%.+]] = affine.apply [[MAP_1_]]([[I_5_]])
// CHECK-DAG:                 [[VAR_133_:%.+]] = arith.addi [[I_2_]], [[CST_3_]] : index
// CHECK:                     [[VAR_134_:%.+]] = arith.addi [[VAR_132_]], [[I_4_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_14_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_133_]], [[VAR_134_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                 [[VAR_136_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_14_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_137_:%.+]] = arith.addi [[VAR_132_]], [[I_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_1_MEM_14_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_137_]], [[I_3_]]{{.}} : memref<255x63xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_14_:%.+]] = affine.load [[RES_1_]][3] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_140_:%.+]] = vector.fma [[VAR_136_]], [[LOAD_PARAM_1_MEM_14_]], [[LOAD_RES_1_MEM_14_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_140_]], [[RES_1_]][3] : memref<4xvector<8xf32>>
// CHECK-DAG:                 [[VAR_141_:%.+]] = affine.apply [[MAP_2_]]([[I_5_]])
// CHECK-DAG:                 [[VAR_142_:%.+]] = arith.addi [[I_2_]], [[CST_3_]] : index
// CHECK:                     [[VAR_143_:%.+]] = arith.addi [[VAR_141_]], [[I_4_]] : index
// CHECK:                     [[LOAD_PARAM_0_MEM_15_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_142_]], [[VAR_143_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                 [[VAR_145_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_15_]] : f32 to vector<8xf32>
// CHECK-DAG:                 [[VAR_146_:%.+]] = arith.addi [[VAR_141_]], [[I_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_PARAM_1_MEM_15_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[VAR_146_]], [[I_3_]]{{.}} : memref<255x63xf32>, vector<8xf32>
// CHECK-DAG:                 [[LOAD_RES_1_MEM_15_:%.+]] = affine.load [[RES_1_]][3] : memref<4xvector<8xf32>>
// CHECK:                     [[VAR_149_:%.+]] = vector.fma [[VAR_145_]], [[LOAD_PARAM_1_MEM_15_]], [[LOAD_RES_1_MEM_15_]] : vector<8xf32>
// CHECK:                     affine.store [[VAR_149_]], [[RES_1_]][3] : memref<4xvector<8xf32>>
// CHECK:                   }
// CHECK:                   [[LOAD_RES_1_MEM_16_:%.+]] = affine.load [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK:                   vector.store [[LOAD_RES_1_MEM_16_]], [[RES_]]{{.}}[[I_2_]], [[I_3_]]{{.}} : memref<127x63xf32>, vector<8xf32>
// CHECK-DAG:               [[LOAD_RES_1_MEM_17_:%.+]] = affine.load [[RES_1_]][1] : memref<4xvector<8xf32>>
// CHECK-DAG:               [[VAR_9_:%.+]] = arith.addi [[I_2_]], [[CST_1_]] : index
// CHECK:                   vector.store [[LOAD_RES_1_MEM_17_]], [[RES_]]{{.}}[[VAR_9_]], [[I_3_]]{{.}} : memref<127x63xf32>, vector<8xf32>
// CHECK-DAG:               [[LOAD_RES_1_MEM_18_:%.+]] = affine.load [[RES_1_]][2] : memref<4xvector<8xf32>>
// CHECK-DAG:               [[VAR_11_:%.+]] = arith.addi [[I_2_]], [[CST_2_]] : index
// CHECK:                   vector.store [[LOAD_RES_1_MEM_18_]], [[RES_]]{{.}}[[VAR_11_]], [[I_3_]]{{.}} : memref<127x63xf32>, vector<8xf32>
// CHECK-DAG:               [[LOAD_RES_1_MEM_19_:%.+]] = affine.load [[RES_1_]][3] : memref<4xvector<8xf32>>
// CHECK-DAG:               [[VAR_13_:%.+]] = arith.addi [[I_2_]], [[CST_3_]] : index
// CHECK:                   vector.store [[LOAD_RES_1_MEM_19_]], [[RES_]]{{.}}[[VAR_13_]], [[I_3_]]{{.}} : memref<127x63xf32>, vector<8xf32>
// CHECK:                 } else {
// skip this one            affine.if [[set1_]]([[I_3_]]) {
// CHECK:                     affine.for [[I_6_:%.+]] = 0 to min [[MAP_3_]]([[I_2_]]) {
// CHECK:                       [[LOAD_RES_MEM_4_:%.+]] = arith.addi [[I_6_]], [[I_2_]] : index
// CHECK:                       [[LOAD_RES_MEM_5_:%.+]] = vector.load [[RES_]]{{.}}[[LOAD_RES_MEM_4_]], [[I_3_]]{{.}} : memref<127x63xf32>, vector<8xf32>
// CHECK:                       affine.store [[LOAD_RES_MEM_5_]], [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK:                       affine.for [[I_7_:%.+]] = 0 to min [[MAP_4_]]([[I_4_]]) {
// CHECK-DAG:                     [[LOAD_RES_MEM_2_:%.+]] = arith.addi [[I_6_]], [[I_2_]] : index
// CHECK-DAG:                     [[VAR_5_1_:%.+]] = arith.addi [[I_7_]], [[I_4_]] : index
// CHECK:                         [[LOAD_PARAM_0_MEM_16_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[LOAD_RES_MEM_2_]], [[VAR_5_1_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                     [[LOAD_RES_1_MEM_16_:%.+]] = vector.broadcast [[LOAD_PARAM_0_MEM_16_]] : f32 to vector<8xf32>
// CHECK-DAG:                     [[LOAD_RES_1_MEM_17_:%.+]] = arith.addi [[I_7_]], [[I_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                     [[LOAD_PARAM_1_MEM_16_:%.+]] = vector.load [[PARAM_1_]]{{.}}[[LOAD_RES_1_MEM_17_]], [[I_3_]]{{.}} : memref<255x63xf32>, vector<8xf32>
// CHECK-DAG:                     [[LOAD_RES_1_MEM_20_:%.+]] = affine.load [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK:                         [[VAR_11_1_:%.+]] = vector.fma [[LOAD_RES_1_MEM_16_]], [[LOAD_PARAM_1_MEM_16_]], [[LOAD_RES_1_MEM_20_]] : vector<8xf32>
// CHECK:                         affine.store [[VAR_11_1_]], [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK:                       }
// CHECK-DAG:                   [[LOAD_RES_1_MEM_21_:%.+]] = affine.load [[RES_1_]][0] : memref<4xvector<8xf32>>
// CHECK-DAG:                   [[VAR_3_1_:%.+]] = arith.addi [[I_6_]], [[I_2_]] : index
// CHECK:                       vector.store [[LOAD_RES_1_MEM_21_]], [[RES_]]{{.}}[[VAR_3_1_]], [[I_3_]]{{.}} : memref<127x63xf32>, vector<8xf32>
// CHECK:                     }
// CHECK:                   } else {
// CHECK:                     affine.for [[I_8_:%.+]] = 0 to min [[MAP_3_]]([[I_2_]]) {
// CHECK:                       affine.for [[I_9_:%.+]] = 0 to 7 {
// CHECK-DAG:                     [[LOAD_RES_MEM_4_:%.+]] = arith.addi [[I_8_]], [[I_2_]] : index
// CHECK-DAG:                     [[VAR_1_1_:%.+]] = arith.addi [[I_9_]], [[I_3_]] : index
// CHECK:                         [[LOAD_RES_MEM_6_:%.+]] = memref.load [[RES_]]{{.}}[[LOAD_RES_MEM_4_]], [[VAR_1_1_]]{{.}} : memref<127x63xf32>
// CHECK:                         affine.store [[LOAD_RES_MEM_6_]], [[RES_2_]][0] : memref<1xf32>
// CHECK:                         affine.for [[I_10_:%.+]] = 0 to min [[MAP_4_]]([[I_4_]]) {
// CHECK-DAG:                       [[LOAD_PARAM_0_MEM_16_:%.+]] = arith.addi [[I_8_]], [[I_2_]] : index
// CHECK-DAG:                       [[LOAD_RES_1_MEM_16_1_:%.+]] = arith.addi [[I_10_]], [[I_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                       [[LOAD_PARAM_0_MEM_17_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[LOAD_PARAM_0_MEM_16_]], [[LOAD_RES_1_MEM_16_1_]]{{.}} : memref<127x255xf32>
// CHECK-DAG:                       [[VAR_9_1_:%.+]] = arith.addi [[I_10_]], [[I_4_]] : index
// CHECK-DAG:                       [[LOAD_RES_1_MEM_20_:%.+]] = arith.addi [[I_9_]], [[I_3_]] : index
// CHECK:                           [[LOAD_PARAM_1_MEM_17_:%.+]] = memref.load [[PARAM_1_]]{{.}}[[VAR_9_1_]], [[LOAD_RES_1_MEM_20_]]{{.}} : memref<255x63xf32>
// CHECK-DAG:                       [[LOAD_RES_1_MEM_19_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_17_]], [[LOAD_PARAM_1_MEM_17_]] : f32
// CHECK-DAG:                       [[VAR_13_1_:%.+]] = affine.load [[RES_2_]][0] : memref<1xf32>
// CHECK:                           [[VAR_14_1_:%.+]] = arith.addf [[LOAD_RES_1_MEM_19_]], [[VAR_13_1_]] : f32
// CHECK:                           affine.store [[VAR_14_1_]], [[RES_2_]][0] : memref<1xf32>
// CHECK:                         }
// CHECK-DAG:                     [[VAR_3_1_:%.+]] = affine.load [[RES_2_]][0] : memref<1xf32>
// CHECK-DAG:                     [[LOAD_RES_MEM_2_1_:%.+]] = arith.addi [[I_8_]], [[I_2_]] : index
// CHECK-DAG:                     [[VAR_5_2_:%.+]] = arith.addi [[I_9_]], [[I_3_]] : index
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

