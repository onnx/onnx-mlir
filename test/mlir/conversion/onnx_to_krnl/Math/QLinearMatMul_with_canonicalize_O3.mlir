// RUN: onnx-mlir-opt -O3 --mtriple=s390x-ibm-loz --march=z16 --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// -----


func.func @qlinearmatmul_i8_f32(%arg0: tensor<16x32xi8>, %arg1: tensor<1xf32>, %arg2: tensor<1xi8>, %arg3: tensor<32x64xi8>, %arg4: tensor<1xf32>, %arg5: tensor<1xi8>, %arg6: tensor<1xf32>, %arg7: tensor<1xi8>) -> (tensor<16x64xi8>) {
    %0 = "onnx.QLinearMatMul"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7) : (tensor<16x32xi8>, tensor<1xf32>, tensor<1xi8>, tensor<32x64xi8>, tensor<1xf32>, tensor<1xi8>, tensor<1xf32>, tensor<1xi8>) -> tensor<16x64xi8>
    return %0 : tensor<16x64xi8>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @qlinearmatmul_i8_f32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<16x32xi8>, [[PARAM_1_:%.+]]: memref<1xf32>, [[PARAM_2_:%.+]]: memref<1xi8>, [[PARAM_3_:%.+]]: memref<32x64xi8>, [[PARAM_4_:%.+]]: memref<1xf32>, [[PARAM_5_:%.+]]: memref<1xi8>, [[PARAM_6_:%.+]]: memref<1xf32>, [[PARAM_7_:%.+]]: memref<1xi8>) -> memref<16x64xi8> {
// CHECK-DAG:       [[CST_1024_:%.+]] = arith.constant 1024 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_2048_:%.+]] = arith.constant 2048 : index
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_512_:%.+]] = arith.constant 512 : index
// CHECK-DAG:       [[CST_16_:%.+]] = arith.constant 16 : index
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<16x32xi32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_512_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<16x32xi8>, memref<1xindex>) -> memref<512xi8>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_512_]], [[RES_2_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_2_:%.+]] = memref.reshape [[RES_]]([[RES_]]_1) : (memref<16x32xi32>, memref<1xindex>) -> memref<512xi32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 512){
// CHECK:               [[VAR_14_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_14_]]{{.}} : memref<512xi8>, vector<32xi8>
// CHECK:               [[VAR_16_:%.+]] = arith.extsi [[LOAD_VAR_reshape_MEM_]] : vector<32xi8> to vector<32xi32>
// CHECK:               vector.store [[VAR_16_]], [[VAR_reshape_2_]]{{.}}[[VAR_14_]]{{.}} : memref<512xi32>, vector<32xi32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1xi32>
// CHECK-DAG:       [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi8>
// CHECK:           [[VAR_1_:%.+]] = arith.extsi [[LOAD_PARAM_2_MEM_]] : i8 to i32
// CHECK:           krnl.store [[VAR_1_]], [[RES_3_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<16x32xi32>
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_512_]], [[RES_5_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_6_:%.+]] = memref.reshape [[RES_]]([[RES_]]_5) : (memref<16x32xi32>, memref<1xindex>) -> memref<512xi32>
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_512_]], [[RES_6_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_8_:%.+]] = memref.reshape [[RES_4_]]([[RES_6_]]) : (memref<16x32xi32>, memref<1xindex>) -> memref<512xi32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_1_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 512){
// CHECK:               [[VAR_14_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_1_:%.+]] = vector.load [[VAR_reshape_6_]]{{.}}[[VAR_14_1_]]{{.}} : memref<512xi32>, vector<32xi32>
// CHECK-DAG:           [[VAR_16_1_:%.+]] = krnl.load [[RES_3_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:               [[VAR_17_:%.+]] = vector.broadcast [[VAR_16_1_]] : i32 to vector<32xi32>
// CHECK:               [[VAR_18_:%.+]] = arith.subi [[LOAD_VAR_reshape_MEM_1_]], [[VAR_17_]] : vector<32xi32>
// CHECK:               vector.store [[VAR_18_]], [[VAR_reshape_8_]]{{.}}[[VAR_14_1_]]{{.}} : memref<512xi32>, vector<32xi32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi32>
// CHECK-DAG:       [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_8_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_11_:%.+]] = memref.reshape [[PARAM_3_]]([[RES_8_]]) : (memref<32x64xi8>, memref<1xindex>) -> memref<2048xi8>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_9_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_13_:%.+]] = memref.reshape [[RES_7_]]([[RES_9_]]) : (memref<32x64xi32>, memref<1xindex>) -> memref<2048xi32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__2_:%.+]], [[BLOCK_IN__2_:%.+]] = krnl.block [[LOOP_2_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 2048){
// CHECK:               [[VAR_14_2_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__2_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_1_:%.+]] = vector.load [[VAR_reshape_11_]]{{.}}[[VAR_14_2_]]{{.}} : memref<2048xi8>, vector<32xi8>
// CHECK:               [[VAR_16_2_:%.+]] = arith.extsi [[LOAD_VAR_reshape_MEM_1_]] : vector<32xi8> to vector<32xi32>
// CHECK:               vector.store [[VAR_16_2_]], [[VAR_reshape_13_]]{{.}}[[VAR_14_2_]]{{.}} : memref<2048xi32>, vector<32xi32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_10_:%.+]] = memref.alloc() {{.*}}: memref<1xi32>
// CHECK-DAG:       [[LOAD_PARAM_5_MEM_:%.+]] = krnl.load [[PARAM_5_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi8>
// CHECK:           [[VAR_3_:%.+]] = arith.extsi [[LOAD_PARAM_5_MEM_]] : i8 to i32
// CHECK:           krnl.store [[VAR_3_]], [[RES_10_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK-DAG:       [[RES_11_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi32>
// CHECK-DAG:       [[RES_12_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_12_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_17_:%.+]] = memref.reshape [[RES_7_]]([[RES_12_]]) : (memref<32x64xi32>, memref<1xindex>) -> memref<2048xi32>
// CHECK-DAG:       [[RES_13_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_13_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_19_:%.+]] = memref.reshape [[RES_11_]]([[RES_13_]]) : (memref<32x64xi32>, memref<1xindex>) -> memref<2048xi32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__3_:%.+]], [[BLOCK_IN__3_:%.+]] = krnl.block [[LOOP_3_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__3_]]) with ([[LOOP_3_]] -> [[I_3_:%.+]] = 0 to 2048){
// CHECK:               [[VAR_14_3_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__3_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_1_1_:%.+]] = vector.load [[VAR_reshape_17_]]{{.}}[[VAR_14_3_]]{{.}} : memref<2048xi32>, vector<32xi32>
// CHECK-DAG:           [[VAR_16_2_:%.+]] = krnl.load [[RES_10_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:               [[VAR_17_1_:%.+]] = vector.broadcast [[VAR_16_2_]] : i32 to vector<32xi32>
// CHECK:               [[VAR_18_1_:%.+]] = arith.subi [[LOAD_VAR_reshape_MEM_1_1_]], [[VAR_17_1_]] : vector<32xi32>
// CHECK:               vector.store [[VAR_18_1_]], [[VAR_reshape_19_]]{{.}}[[VAR_14_3_]]{{.}} : memref<2048xi32>, vector<32xi32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_14_:%.+]] = memref.alloc() {{.*}}: memref<1xi32>
// CHECK-DAG:       [[LOAD_PARAM_7_MEM_:%.+]] = krnl.load [[PARAM_7_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi8>
// CHECK:           [[VAR_5_:%.+]] = arith.extsi [[LOAD_PARAM_7_MEM_]] : i8 to i32
// CHECK:           krnl.store [[VAR_5_]], [[RES_14_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:           [[RES_15_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi32>
// CHECK:           krnl.memset [[RES_15_]], [[CST_0_]] : memref<16x64xi32>
// CHECK:           [[LOOP_4_:%.+]]:3 = krnl.define_loops 3
// CHECK:           [[BLOCK_TILE__4_:%.+]], [[BLOCK_IN__4_:%.+]] = krnl.block [[LOOP_4_]]#0 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__5_:%.+]], [[BLOCK_IN__5_:%.+]] = krnl.block [[LOOP_4_]]#1 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__6_:%.+]], [[BLOCK_IN__6_:%.+]] = krnl.block [[LOOP_4_]]#2 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.permute([[BLOCK_TILE__4_]], [[BLOCK_IN__4_]], [[BLOCK_TILE__4_]]_22, [[BLOCK_IN__4_]]_23, [[BLOCK_TILE__4_]]_24, [[BLOCK_IN__4_]]_25) [0, 3, 1, 4, 2, 5] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
// CHECK:           krnl.iterate([[BLOCK_TILE__4_]], [[BLOCK_TILE__4_]]_22, [[BLOCK_TILE__4_]]_24) with ([[LOOP_4_]]#0 -> [[I_4_:%.+]] = [[CST_0_1_]] to [[CST_16_]], [[LOOP_4_]]#1 -> [[I_5_:%.+]] = [[CST_0_1_]] to [[CST_64_]], [[LOOP_4_]]#2 -> [[I_6_:%.+]] = [[CST_0_1_]] to [[CST_32_]]){
// CHECK:             [[LOOP_3_:%.+]]:3 = krnl.get_induction_var_value([[BLOCK_TILE__4_]], [[BLOCK_TILE__4_]]_22, [[BLOCK_TILE__4_]]_24) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.matmul [[RES_4_]]{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, [[RES_11_]]{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, [[RES_15_]]{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, ([[BLOCK_IN__4_]], [[BLOCK_IN__4_]]_23, [[BLOCK_IN__4_]]_25), ([[LOOP_3_]]#0, [[LOOP_3_]]#1, [[LOOP_3_]]#2), ([[CST_16_]], [[CST_64_]], [[CST_32_]]) {aTileSize = [], bTileSize = [], cTileSize = [], computeTileSize = [4, 8, 8]} : memref<16x32xi32>, memref<32x64xi32>, memref<16x64xi32>, (!krnl.loop, !krnl.loop, !krnl.loop)
// CHECK:           }
// CHECK-DAG:       [[RES_16_:%.+]] = memref.alloc() {{.*}}: memref<16x64xf32>
// CHECK-DAG:       [[RES_17_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_17_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_28_:%.+]] = memref.reshape [[RES_15_]]([[RES_17_]]) : (memref<16x64xi32>, memref<1xindex>) -> memref<1024xi32>
// CHECK-DAG:       [[RES_18_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_18_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_30_:%.+]] = memref.reshape [[RES_16_]]([[RES_18_]]) : (memref<16x64xf32>, memref<1xindex>) -> memref<1024xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__7_:%.+]], [[BLOCK_IN__7_:%.+]] = krnl.block [[LOOP_5_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__7_]]) with ([[LOOP_5_]] -> [[I_7_:%.+]] = 0 to 1024){
// CHECK:               [[VAR_14_4_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__7_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_1_1_:%.+]] = vector.load [[VAR_reshape_28_]]{{.}}[[VAR_14_4_]]{{.}} : memref<1024xi32>, vector<32xi32>
// CHECK:               [[VAR_16_3_:%.+]] = arith.sitofp [[LOAD_VAR_reshape_MEM_1_1_]] : vector<32xi32> to vector<32xf32>
// CHECK:               vector.store [[VAR_16_3_]], [[VAR_reshape_30_]]{{.}}[[VAR_14_4_]]{{.}} : memref<1024xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_19_:%.+]] = memref.alloc() {{.*}}: memref<1xf32>
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK:           [[VAR_9_:%.+]] = arith.mulf [[LOAD_PARAM_1_MEM_]], [[LOAD_PARAM_4_MEM_]] : f32
// CHECK:           krnl.store [[VAR_9_]], [[RES_19_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[RES_20_:%.+]] = memref.alloc() {{.*}}: memref<1xf32>
// CHECK-DAG:       [[LOAD_RES_19_MEM_:%.+]] = krnl.load [[RES_19_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_PARAM_6_MEM_:%.+]] = krnl.load [[PARAM_6_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK:           [[VAR_12_:%.+]] = arith.divf [[LOAD_RES_19_MEM_]], [[LOAD_PARAM_6_MEM_]] : f32
// CHECK:           krnl.store [[VAR_12_]], [[RES_20_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[RES_21_:%.+]] = memref.alloc() {{.*}}: memref<16x64xf32>
// CHECK-DAG:       [[RES_22_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_22_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_35_:%.+]] = memref.reshape [[RES_16_]]([[RES_22_]]) : (memref<16x64xf32>, memref<1xindex>) -> memref<1024xf32>
// CHECK-DAG:       [[RES_23_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_23_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_37_:%.+]] = memref.reshape [[RES_21_]]([[RES_23_]]) : (memref<16x64xf32>, memref<1xindex>) -> memref<1024xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_6_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__8_:%.+]], [[BLOCK_IN__8_:%.+]] = krnl.block [[LOOP_6_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__8_]]) with ([[LOOP_6_]] -> [[I_8_:%.+]] = 0 to 1024){
// CHECK:               [[VAR_14_5_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__8_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_1_1_1_:%.+]] = vector.load [[VAR_reshape_35_]]{{.}}[[VAR_14_5_]]{{.}} : memref<1024xf32>, vector<32xf32>
// CHECK-DAG:           [[VAR_16_3_:%.+]] = krnl.load [[RES_20_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK:               [[VAR_17_2_:%.+]] = vector.broadcast [[VAR_16_3_]] : f32 to vector<32xf32>
// CHECK:               [[VAR_18_2_:%.+]] = arith.mulf [[LOAD_VAR_reshape_MEM_1_1_1_]], [[VAR_17_2_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_18_2_]], [[VAR_reshape_37_]]{{.}}[[VAR_14_5_]]{{.}} : memref<1024xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_24_:%.+]] = memref.alloc() {{.*}}: memref<16x64xf32>
// CHECK-DAG:       [[RES_25_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_25_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_40_:%.+]] = memref.reshape [[RES_21_]]([[RES_25_]]) : (memref<16x64xf32>, memref<1xindex>) -> memref<1024xf32>
// CHECK-DAG:       [[RES_26_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_26_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_42_:%.+]] = memref.reshape [[RES_24_]]([[RES_26_]]) : (memref<16x64xf32>, memref<1xindex>) -> memref<1024xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_7_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__9_:%.+]], [[BLOCK_IN__9_:%.+]] = krnl.block [[LOOP_7_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__9_]]) with ([[LOOP_7_]] -> [[I_9_:%.+]] = 0 to 1024){
// CHECK:               [[VAR_14_6_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__9_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_1_1_1_:%.+]] = vector.load [[VAR_reshape_40_]]{{.}}[[VAR_14_6_]]{{.}} : memref<1024xf32>, vector<32xf32>
// CHECK:               [[VAR_16_4_:%.+]] = vector.shape_cast [[LOAD_VAR_reshape_MEM_1_1_1_]] : vector<32xf32> to vector<8x4xf32>
// CHECK:               [[VAR_17_3_:%.+]] = vector.extract [[VAR_16_4_]][0] : vector<4xf32> from vector<8x4xf32>
// CHECK-DAG:           [[VAR_18_3_:%.+]] = "krnl.round_even"([[VAR_17_3_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:           [[VAR_19_:%.+]] = vector.extract [[VAR_16_4_]][1] : vector<4xf32> from vector<8x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_20_:%.+]] = "krnl.round_even"([[VAR_19_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:           [[VAR_21_:%.+]] = vector.extract [[VAR_16_4_]][2] : vector<4xf32> from vector<8x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_22_:%.+]] = "krnl.round_even"([[VAR_21_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:           [[VAR_23_:%.+]] = vector.extract [[VAR_16_4_]][3] : vector<4xf32> from vector<8x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_24_:%.+]] = "krnl.round_even"([[VAR_23_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:           [[VAR_25_:%.+]] = vector.extract [[VAR_16_4_]][4] : vector<4xf32> from vector<8x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_26_:%.+]] = "krnl.round_even"([[VAR_25_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:           [[VAR_27_:%.+]] = vector.extract [[VAR_16_4_]][5] : vector<4xf32> from vector<8x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_28_:%.+]] = "krnl.round_even"([[VAR_27_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:           [[VAR_29_:%.+]] = vector.extract [[VAR_16_4_]][6] : vector<4xf32> from vector<8x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_30_:%.+]] = "krnl.round_even"([[VAR_29_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:           [[VAR_31_:%.+]] = vector.extract [[VAR_16_4_]][7] : vector<4xf32> from vector<8x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_32_:%.+]] = "krnl.round_even"([[VAR_31_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:           [[VAR_33_:%.+]]:4 = vector.to_elements [[VAR_18_3_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_34_:%.+]]:4 = vector.to_elements [[VAR_20_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_35_:%.+]]:4 = vector.to_elements [[VAR_22_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_36_:%.+]]:4 = vector.to_elements [[VAR_24_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_37_:%.+]]:4 = vector.to_elements [[VAR_26_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_38_:%.+]]:4 = vector.to_elements [[VAR_28_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_39_:%.+]]:4 = vector.to_elements [[VAR_30_]] : vector<4xf32>
// CHECK:               [[VAR_40_:%.+]]:4 = vector.to_elements [[VAR_32_]] : vector<4xf32>
// CHECK:               [[VAR_41_:%.+]] = vector.from_elements [[VAR_33_]]#0, [[VAR_33_]]#1, [[VAR_33_]]#2, [[VAR_33_]]#3, [[VAR_34_]]#0, [[VAR_34_]]#1, [[VAR_34_]]#2, [[VAR_34_]]#3, [[VAR_35_]]#0, [[VAR_35_]]#1, [[VAR_35_]]#2, [[VAR_35_]]#3, [[VAR_36_]]#0, [[VAR_36_]]#1, [[VAR_36_]]#2, [[VAR_36_]]#3, [[VAR_37_]]#0, [[VAR_37_]]#1, [[VAR_37_]]#2, [[VAR_37_]]#3, [[VAR_38_]]#0, [[VAR_38_]]#1, [[VAR_38_]]#2, [[VAR_38_]]#3, [[VAR_39_]]#0, [[VAR_39_]]#1, [[VAR_39_]]#2, [[VAR_39_]]#3, [[VAR_40_]]#0, [[VAR_40_]]#1, [[VAR_40_]]#2, [[VAR_40_]]#3 : vector<8x4xf32>
// CHECK:               [[VAR_42_:%.+]] = vector.shape_cast [[VAR_41_]] : vector<8x4xf32> to vector<32xf32>
// CHECK:               vector.store [[VAR_42_]], [[VAR_reshape_42_]]{{.}}[[VAR_14_6_]]{{.}} : memref<1024xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_27_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi32>
// CHECK-DAG:       [[RES_28_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_28_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_45_:%.+]] = memref.reshape [[RES_24_]]([[RES_28_]]) : (memref<16x64xf32>, memref<1xindex>) -> memref<1024xf32>
// CHECK-DAG:       [[RES_29_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_29_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_47_:%.+]] = memref.reshape [[RES_27_]]([[RES_29_]]) : (memref<16x64xi32>, memref<1xindex>) -> memref<1024xi32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_8_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__10_:%.+]], [[BLOCK_IN__10_:%.+]] = krnl.block [[LOOP_8_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__10_]]) with ([[LOOP_8_]] -> [[I_10_:%.+]] = 0 to 1024){
// CHECK:               [[VAR_14_7_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__10_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_1_1_1_1_:%.+]] = vector.load [[VAR_reshape_45_]]{{.}}[[VAR_14_7_]]{{.}} : memref<1024xf32>, vector<32xf32>
// CHECK:               [[VAR_16_5_:%.+]] = arith.fptosi [[LOAD_VAR_reshape_MEM_1_1_1_1_]] : vector<32xf32> to vector<32xi32>
// CHECK:               vector.store [[VAR_16_5_]], [[VAR_reshape_47_]]{{.}}[[VAR_14_7_]]{{.}} : memref<1024xi32>, vector<32xi32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_30_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi32>
// CHECK-DAG:       [[RES_31_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_31_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_50_:%.+]] = memref.reshape [[RES_27_]]([[RES_31_]]) : (memref<16x64xi32>, memref<1xindex>) -> memref<1024xi32>
// CHECK-DAG:       [[RES_32_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_32_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_52_:%.+]] = memref.reshape [[RES_30_]]([[RES_32_]]) : (memref<16x64xi32>, memref<1xindex>) -> memref<1024xi32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_9_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__11_:%.+]], [[BLOCK_IN__11_:%.+]] = krnl.block [[LOOP_9_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__11_]]) with ([[LOOP_9_]] -> [[I_11_:%.+]] = 0 to 1024){
// CHECK:               [[VAR_14_8_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__11_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_1_1_1_1_:%.+]] = vector.load [[VAR_reshape_50_]]{{.}}[[VAR_14_8_]]{{.}} : memref<1024xi32>, vector<32xi32>
// CHECK-DAG:           [[VAR_16_5_:%.+]] = krnl.load [[RES_14_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:               [[VAR_17_4_:%.+]] = vector.broadcast [[VAR_16_5_]] : i32 to vector<32xi32>
// CHECK:               [[VAR_18_4_:%.+]] = arith.addi [[LOAD_VAR_reshape_MEM_1_1_1_1_]], [[VAR_17_4_]] : vector<32xi32>
// CHECK:               vector.store [[VAR_18_4_]], [[VAR_reshape_52_]]{{.}}[[VAR_14_8_]]{{.}} : memref<1024xi32>, vector<32xi32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_33_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi8>
// CHECK-DAG:       [[RES_34_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_34_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_55_:%.+]] = memref.reshape [[RES_30_]]([[RES_34_]]) : (memref<16x64xi32>, memref<1xindex>) -> memref<1024xi32>
// CHECK-DAG:       [[RES_35_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_35_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_57_:%.+]] = memref.reshape [[RES_33_]]([[RES_35_]]) : (memref<16x64xi8>, memref<1xindex>) -> memref<1024xi8>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_10_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__12_:%.+]], [[BLOCK_IN__12_:%.+]] = krnl.block [[LOOP_10_]] 128 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__12_]]) with ([[LOOP_10_]] -> [[I_12_:%.+]] = 0 to 1024){
// CHECK:               [[VAR_14_9_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__12_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_1_1_1_1_1_:%.+]] = vector.load [[VAR_reshape_55_]]{{.}}[[VAR_14_9_]]{{.}} : memref<1024xi32>, vector<128xi32>
// CHECK:               [[VAR_16_6_:%.+]] = arith.trunci [[LOAD_VAR_reshape_MEM_1_1_1_1_1_]] : vector<128xi32> to vector<128xi8>
// CHECK:               vector.store [[VAR_16_6_]], [[VAR_reshape_57_]]{{.}}[[VAR_14_9_]]{{.}} : memref<1024xi8>, vector<128xi8>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_33_]] : memref<16x64xi8>
// CHECK:         }
}

// -----


func.func @qlinearmatmul_ui8_f32(%arg0: tensor<16x32xui8>, %arg1: tensor<1xf32>, %arg2: tensor<1xui8>, %arg3: tensor<32x64xui8>, %arg4: tensor<1xf32>, %arg5: tensor<1xui8>, %arg6: tensor<1xf32>, %arg7: tensor<1xui8>) -> (tensor<16x64xui8>) {
    %0 = "onnx.QLinearMatMul"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7) : (tensor<16x32xui8>, tensor<1xf32>, tensor<1xui8>, tensor<32x64xui8>, tensor<1xf32>, tensor<1xui8>, tensor<1xf32>, tensor<1xui8>) -> tensor<16x64xui8>
    return %0 : tensor<16x64xui8>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @qlinearmatmul_ui8_f32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<16x32xui8>, [[PARAM_1_:%.+]]: memref<1xf32>, [[PARAM_2_:%.+]]: memref<1xui8>, [[PARAM_3_:%.+]]: memref<32x64xui8>, [[PARAM_4_:%.+]]: memref<1xf32>, [[PARAM_5_:%.+]]: memref<1xui8>, [[PARAM_6_:%.+]]: memref<1xf32>, [[PARAM_7_:%.+]]: memref<1xui8>) -> memref<16x64xui8> {
// CHECK-DAG:       [[CST_1024_:%.+]] = arith.constant 1024 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_2048_:%.+]] = arith.constant 2048 : index
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_512_:%.+]] = arith.constant 512 : index
// CHECK-DAG:       [[CST_16_:%.+]] = arith.constant 16 : index
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [], value = dense<128> : tensor<i32>} : () -> memref<i32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [], value = dense<128> : tensor<i16>} : () -> memref<i16>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<16x32xi16>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_512_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<16x32xui8>, memref<1xindex>) -> memref<512xui8>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_512_]], [[RES_2_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_2_:%.+]] = memref.reshape [[RES_]]([[RES_]]_1) : (memref<16x32xi16>, memref<1xindex>) -> memref<512xi16>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 64 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 512){
// CHECK:               [[VAR_44_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_44_]]{{.}} : memref<512xui8>, vector<64xui8>
// CHECK:               [[VAR_46_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_VAR_reshape_MEM_]] : vector<64xui8> to vector<64xi8>
// CHECK:               [[VAR_47_:%.+]] = arith.extui [[VAR_46_]] : vector<64xi8> to vector<64xi16>
// CHECK:               vector.store [[VAR_47_]], [[VAR_reshape_2_]]{{.}}[[VAR_44_]]{{.}} : memref<512xi16>, vector<64xi16>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<16x32xi16>
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_512_]], [[RES_4_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_5_:%.+]] = memref.reshape [[RES_]]([[RES_]]_4) : (memref<16x32xi16>, memref<1xindex>) -> memref<512xi16>
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_512_]], [[RES_5_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_7_:%.+]] = memref.reshape [[RES_3_]]([[RES_5_]]) : (memref<16x32xi16>, memref<1xindex>) -> memref<512xi16>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_1_]] 64 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 512){
// CHECK:               [[VAR_44_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_1_:%.+]] = vector.load [[VAR_reshape_5_]]{{.}}[[VAR_44_1_]]{{.}} : memref<512xi16>, vector<64xi16>
// CHECK-DAG:           [[VAR_46_1_:%.+]] = krnl.load [[VAR_1_]][] : memref<i16>
// CHECK:               [[VAR_47_1_:%.+]] = vector.broadcast [[VAR_46_1_]] : i16 to vector<64xi16>
// CHECK:               [[VAR_48_:%.+]] = arith.subi [[LOAD_VAR_reshape_MEM_1_]], [[VAR_47_1_]] : vector<64xi16>
// CHECK:               vector.store [[VAR_48_]], [[VAR_reshape_7_]]{{.}}[[VAR_44_1_]]{{.}} : memref<512xi16>, vector<64xi16>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<16x32xi8>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_512_]], [[RES_7_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_10_:%.+]] = memref.reshape [[RES_3_]]([[RES_7_]]) : (memref<16x32xi16>, memref<1xindex>) -> memref<512xi16>
// CHECK-DAG:       [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_512_]], [[RES_8_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_12_:%.+]] = memref.reshape [[RES_6_]]([[RES_8_]]) : (memref<16x32xi8>, memref<1xindex>) -> memref<512xi8>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__2_:%.+]], [[BLOCK_IN__2_:%.+]] = krnl.block [[LOOP_2_]] 128 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 512){
// CHECK:               [[VAR_44_2_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__2_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_1_:%.+]] = vector.load [[VAR_reshape_10_]]{{.}}[[VAR_44_2_]]{{.}} : memref<512xi16>, vector<128xi16>
// CHECK:               [[VAR_46_2_:%.+]] = arith.trunci [[LOAD_VAR_reshape_MEM_1_]] : vector<128xi16> to vector<128xi8>
// CHECK:               vector.store [[VAR_46_2_]], [[VAR_reshape_12_]]{{.}}[[VAR_44_2_]]{{.}} : memref<512xi8>, vector<128xi8>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc() {{.*}}: memref<16x32xi32>
// CHECK-DAG:       [[RES_10_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_512_]], [[RES_10_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_15_:%.+]] = memref.reshape [[RES_6_]]([[RES_10_]]) : (memref<16x32xi8>, memref<1xindex>) -> memref<512xi8>
// CHECK-DAG:       [[RES_11_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_512_]], [[RES_11_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_17_:%.+]] = memref.reshape [[RES_9_]]([[RES_11_]]) : (memref<16x32xi32>, memref<1xindex>) -> memref<512xi32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__3_:%.+]], [[BLOCK_IN__3_:%.+]] = krnl.block [[LOOP_3_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__3_]]) with ([[LOOP_3_]] -> [[I_3_:%.+]] = 0 to 512){
// CHECK:               [[VAR_44_3_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__3_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_1_1_:%.+]] = vector.load [[VAR_reshape_15_]]{{.}}[[VAR_44_3_]]{{.}} : memref<512xi8>, vector<32xi8>
// CHECK:               [[VAR_46_3_:%.+]] = arith.extsi [[LOAD_VAR_reshape_MEM_1_1_]] : vector<32xi8> to vector<32xi32>
// CHECK:               vector.store [[VAR_46_3_]], [[VAR_reshape_17_]]{{.}}[[VAR_44_3_]]{{.}} : memref<512xi32>, vector<32xi32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[VAR_2_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [], value = dense<128> : tensor<i16>} : () -> memref<i16>
// CHECK-DAG:       [[RES_12_:%.+]] = memref.alloc() {{.*}}: memref<1xi16>
// CHECK-DAG:       [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xui8>
// CHECK:           [[VAR_4_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_2_MEM_]] : ui8 to i8
// CHECK:           [[VAR_5_:%.+]] = arith.extui [[VAR_4_]] : i8 to i16
// CHECK:           krnl.store [[VAR_5_]], [[RES_12_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK-DAG:       [[RES_13_:%.+]] = memref.alloc() {{.*}}: memref<1xi16>
// CHECK-DAG:       [[LOAD_RES_12_MEM_:%.+]] = krnl.load [[RES_12_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK-DAG:       [[LOAD_VAR_2_MEM_:%.+]] = krnl.load [[VAR_2_]][] : memref<i16>
// CHECK:           [[VAR_8_:%.+]] = arith.subi [[LOAD_RES_12_MEM_]], [[LOAD_VAR_2_MEM_]] : i16
// CHECK:           krnl.store [[VAR_8_]], [[RES_13_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK-DAG:       [[RES_14_:%.+]] = memref.alloc() {{.*}}: memref<1xi8>
// CHECK-DAG:       [[LOAD_RES_13_MEM_:%.+]] = krnl.load [[RES_13_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK:           [[VAR_10_:%.+]] = arith.trunci [[LOAD_RES_13_MEM_]] : i16 to i8
// CHECK:           krnl.store [[VAR_10_]], [[RES_14_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi8>
// CHECK-DAG:       [[RES_15_:%.+]] = memref.alloc() {{.*}}: memref<1xi32>
// CHECK-DAG:       [[LOAD_RES_14_MEM_:%.+]] = krnl.load [[RES_14_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi8>
// CHECK:           [[VAR_12_:%.+]] = arith.extsi [[LOAD_RES_14_MEM_]] : i8 to i32
// CHECK:           krnl.store [[VAR_12_]], [[RES_15_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK-DAG:       [[RES_16_:%.+]] = memref.alloc() {{.*}}: memref<16x32xi32>
// CHECK-DAG:       [[RES_17_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_512_]], [[RES_17_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_24_:%.+]] = memref.reshape [[RES_9_]]([[RES_17_]]) : (memref<16x32xi32>, memref<1xindex>) -> memref<512xi32>
// CHECK-DAG:       [[RES_18_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_512_]], [[RES_18_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_26_:%.+]] = memref.reshape [[RES_16_]]([[RES_18_]]) : (memref<16x32xi32>, memref<1xindex>) -> memref<512xi32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__4_:%.+]], [[BLOCK_IN__4_:%.+]] = krnl.block [[LOOP_4_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__4_]]) with ([[LOOP_4_]] -> [[I_4_:%.+]] = 0 to 512){
// CHECK:               [[VAR_44_4_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__4_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_1_1_:%.+]] = vector.load [[VAR_reshape_24_]]{{.}}[[VAR_44_4_]]{{.}} : memref<512xi32>, vector<32xi32>
// CHECK-DAG:           [[VAR_46_3_:%.+]] = krnl.load [[RES_15_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:               [[VAR_47_2_:%.+]] = vector.broadcast [[VAR_46_3_]] : i32 to vector<32xi32>
// CHECK:               [[VAR_48_1_:%.+]] = arith.subi [[LOAD_VAR_reshape_MEM_1_1_]], [[VAR_47_2_]] : vector<32xi32>
// CHECK:               vector.store [[VAR_48_1_]], [[VAR_reshape_26_]]{{.}}[[VAR_44_4_]]{{.}} : memref<512xi32>, vector<32xi32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[VAR_13_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [], value = dense<128> : tensor<i16>} : () -> memref<i16>
// CHECK-DAG:       [[RES_19_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi16>
// CHECK-DAG:       [[RES_20_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_20_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_29_:%.+]] = memref.reshape [[PARAM_3_]]([[RES_20_]]) : (memref<32x64xui8>, memref<1xindex>) -> memref<2048xui8>
// CHECK-DAG:       [[RES_21_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_21_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_31_:%.+]] = memref.reshape [[RES_19_]]([[RES_21_]]) : (memref<32x64xi16>, memref<1xindex>) -> memref<2048xi16>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__5_:%.+]], [[BLOCK_IN__5_:%.+]] = krnl.block [[LOOP_5_]] 64 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__5_]]) with ([[LOOP_5_]] -> [[I_5_:%.+]] = 0 to 2048){
// CHECK:               [[VAR_44_5_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__5_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_1_1_1_:%.+]] = vector.load [[VAR_reshape_29_]]{{.}}[[VAR_44_5_]]{{.}} : memref<2048xui8>, vector<64xui8>
// CHECK:               [[VAR_46_4_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_VAR_reshape_MEM_1_1_1_]] : vector<64xui8> to vector<64xi8>
// CHECK:               [[VAR_47_3_:%.+]] = arith.extui [[VAR_46_4_]] : vector<64xi8> to vector<64xi16>
// CHECK:               vector.store [[VAR_47_3_]], [[VAR_reshape_31_]]{{.}}[[VAR_44_5_]]{{.}} : memref<2048xi16>, vector<64xi16>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_22_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi16>
// CHECK-DAG:       [[RES_23_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_23_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_34_:%.+]] = memref.reshape [[RES_19_]]([[RES_23_]]) : (memref<32x64xi16>, memref<1xindex>) -> memref<2048xi16>
// CHECK-DAG:       [[RES_24_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_24_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_36_:%.+]] = memref.reshape [[RES_22_]]([[RES_24_]]) : (memref<32x64xi16>, memref<1xindex>) -> memref<2048xi16>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_6_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__6_:%.+]], [[BLOCK_IN__6_:%.+]] = krnl.block [[LOOP_6_]] 64 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__6_]]) with ([[LOOP_6_]] -> [[I_6_:%.+]] = 0 to 2048){
// CHECK:               [[VAR_44_6_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__6_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_1_1_1_:%.+]] = vector.load [[VAR_reshape_34_]]{{.}}[[VAR_44_6_]]{{.}} : memref<2048xi16>, vector<64xi16>
// CHECK-DAG:           [[VAR_46_4_:%.+]] = krnl.load [[VAR_13_]][] : memref<i16>
// CHECK:               [[VAR_47_4_:%.+]] = vector.broadcast [[VAR_46_4_]] : i16 to vector<64xi16>
// CHECK:               [[VAR_48_2_:%.+]] = arith.subi [[LOAD_VAR_reshape_MEM_1_1_1_]], [[VAR_47_4_]] : vector<64xi16>
// CHECK:               vector.store [[VAR_48_2_]], [[VAR_reshape_36_]]{{.}}[[VAR_44_6_]]{{.}} : memref<2048xi16>, vector<64xi16>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_25_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi8>
// CHECK-DAG:       [[RES_26_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_26_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_39_:%.+]] = memref.reshape [[RES_22_]]([[RES_26_]]) : (memref<32x64xi16>, memref<1xindex>) -> memref<2048xi16>
// CHECK-DAG:       [[RES_27_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_27_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_41_:%.+]] = memref.reshape [[RES_25_]]([[RES_27_]]) : (memref<32x64xi8>, memref<1xindex>) -> memref<2048xi8>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_7_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__7_:%.+]], [[BLOCK_IN__7_:%.+]] = krnl.block [[LOOP_7_]] 128 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__7_]]) with ([[LOOP_7_]] -> [[I_7_:%.+]] = 0 to 2048){
// CHECK:               [[VAR_44_7_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__7_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_1_1_1_1_:%.+]] = vector.load [[VAR_reshape_39_]]{{.}}[[VAR_44_7_]]{{.}} : memref<2048xi16>, vector<128xi16>
// CHECK:               [[VAR_46_5_:%.+]] = arith.trunci [[LOAD_VAR_reshape_MEM_1_1_1_1_]] : vector<128xi16> to vector<128xi8>
// CHECK:               vector.store [[VAR_46_5_]], [[VAR_reshape_41_]]{{.}}[[VAR_44_7_]]{{.}} : memref<2048xi8>, vector<128xi8>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_28_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi32>
// CHECK-DAG:       [[RES_29_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_29_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_44_:%.+]] = memref.reshape [[RES_25_]]([[RES_29_]]) : (memref<32x64xi8>, memref<1xindex>) -> memref<2048xi8>
// CHECK-DAG:       [[RES_30_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_30_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_46_:%.+]] = memref.reshape [[RES_28_]]([[RES_30_]]) : (memref<32x64xi32>, memref<1xindex>) -> memref<2048xi32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_8_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__8_:%.+]], [[BLOCK_IN__8_:%.+]] = krnl.block [[LOOP_8_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__8_]]) with ([[LOOP_8_]] -> [[I_8_:%.+]] = 0 to 2048){
// CHECK:               [[VAR_44_8_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__8_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_1_1_1_1_:%.+]] = vector.load [[VAR_reshape_44_]]{{.}}[[VAR_44_8_]]{{.}} : memref<2048xi8>, vector<32xi8>
// CHECK:               [[VAR_46_6_:%.+]] = arith.extsi [[LOAD_VAR_reshape_MEM_1_1_1_1_]] : vector<32xi8> to vector<32xi32>
// CHECK:               vector.store [[VAR_46_6_]], [[VAR_reshape_46_]]{{.}}[[VAR_44_8_]]{{.}} : memref<2048xi32>, vector<32xi32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[VAR_14_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [], value = dense<128> : tensor<i16>} : () -> memref<i16>
// CHECK-DAG:       [[RES_31_:%.+]] = memref.alloc() {{.*}}: memref<1xi16>
// CHECK-DAG:       [[LOAD_PARAM_5_MEM_:%.+]] = krnl.load [[PARAM_5_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xui8>
// CHECK:           [[VAR_16_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_5_MEM_]] : ui8 to i8
// CHECK:           [[VAR_17_:%.+]] = arith.extui [[VAR_16_]] : i8 to i16
// CHECK:           krnl.store [[VAR_17_]], [[RES_31_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK-DAG:       [[RES_32_:%.+]] = memref.alloc() {{.*}}: memref<1xi16>
// CHECK-DAG:       [[LOAD_RES_31_MEM_:%.+]] = krnl.load [[RES_31_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK-DAG:       [[LOAD_VAR_14_MEM_:%.+]] = krnl.load [[VAR_14_]][] : memref<i16>
// CHECK:           [[VAR_20_:%.+]] = arith.subi [[LOAD_RES_31_MEM_]], [[LOAD_VAR_14_MEM_]] : i16
// CHECK:           krnl.store [[VAR_20_]], [[RES_32_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK-DAG:       [[RES_33_:%.+]] = memref.alloc() {{.*}}: memref<1xi8>
// CHECK-DAG:       [[LOAD_RES_32_MEM_:%.+]] = krnl.load [[RES_32_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK:           [[VAR_22_:%.+]] = arith.trunci [[LOAD_RES_32_MEM_]] : i16 to i8
// CHECK:           krnl.store [[VAR_22_]], [[RES_33_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi8>
// CHECK-DAG:       [[RES_34_:%.+]] = memref.alloc() {{.*}}: memref<1xi32>
// CHECK-DAG:       [[LOAD_RES_33_MEM_:%.+]] = krnl.load [[RES_33_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi8>
// CHECK:           [[VAR_24_:%.+]] = arith.extsi [[LOAD_RES_33_MEM_]] : i8 to i32
// CHECK:           krnl.store [[VAR_24_]], [[RES_34_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK-DAG:       [[RES_35_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi32>
// CHECK-DAG:       [[RES_36_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_36_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_53_:%.+]] = memref.reshape [[RES_28_]]([[RES_36_]]) : (memref<32x64xi32>, memref<1xindex>) -> memref<2048xi32>
// CHECK-DAG:       [[RES_37_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_37_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_55_:%.+]] = memref.reshape [[RES_35_]]([[RES_37_]]) : (memref<32x64xi32>, memref<1xindex>) -> memref<2048xi32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_9_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__9_:%.+]], [[BLOCK_IN__9_:%.+]] = krnl.block [[LOOP_9_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__9_]]) with ([[LOOP_9_]] -> [[I_9_:%.+]] = 0 to 2048){
// CHECK:               [[VAR_44_9_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__9_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_1_1_1_1_1_:%.+]] = vector.load [[VAR_reshape_53_]]{{.}}[[VAR_44_9_]]{{.}} : memref<2048xi32>, vector<32xi32>
// CHECK-DAG:           [[VAR_46_6_:%.+]] = krnl.load [[RES_34_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:               [[VAR_47_5_:%.+]] = vector.broadcast [[VAR_46_6_]] : i32 to vector<32xi32>
// CHECK:               [[VAR_48_3_:%.+]] = arith.subi [[LOAD_VAR_reshape_MEM_1_1_1_1_1_]], [[VAR_47_5_]] : vector<32xi32>
// CHECK:               vector.store [[VAR_48_3_]], [[VAR_reshape_55_]]{{.}}[[VAR_44_9_]]{{.}} : memref<2048xi32>, vector<32xi32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[VAR_25_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [], value = dense<128> : tensor<i16>} : () -> memref<i16>
// CHECK-DAG:       [[RES_38_:%.+]] = memref.alloc() {{.*}}: memref<1xi16>
// CHECK-DAG:       [[LOAD_PARAM_7_MEM_:%.+]] = krnl.load [[PARAM_7_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xui8>
// CHECK:           [[VAR_27_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_7_MEM_]] : ui8 to i8
// CHECK:           [[VAR_28_:%.+]] = arith.extui [[VAR_27_]] : i8 to i16
// CHECK:           krnl.store [[VAR_28_]], [[RES_38_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK-DAG:       [[RES_39_:%.+]] = memref.alloc() {{.*}}: memref<1xi16>
// CHECK-DAG:       [[LOAD_RES_38_MEM_:%.+]] = krnl.load [[RES_38_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK-DAG:       [[LOAD_VAR_25_MEM_:%.+]] = krnl.load [[VAR_25_]][] : memref<i16>
// CHECK:           [[VAR_31_:%.+]] = arith.subi [[LOAD_RES_38_MEM_]], [[LOAD_VAR_25_MEM_]] : i16
// CHECK:           krnl.store [[VAR_31_]], [[RES_39_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK-DAG:       [[RES_40_:%.+]] = memref.alloc() {{.*}}: memref<1xi8>
// CHECK-DAG:       [[LOAD_RES_39_MEM_:%.+]] = krnl.load [[RES_39_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK:           [[VAR_33_:%.+]] = arith.trunci [[LOAD_RES_39_MEM_]] : i16 to i8
// CHECK:           krnl.store [[VAR_33_]], [[RES_40_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi8>
// CHECK-DAG:       [[RES_41_:%.+]] = memref.alloc() {{.*}}: memref<1xi32>
// CHECK-DAG:       [[LOAD_RES_40_MEM_:%.+]] = krnl.load [[RES_40_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi8>
// CHECK:           [[VAR_35_:%.+]] = arith.extsi [[LOAD_RES_40_MEM_]] : i8 to i32
// CHECK:           krnl.store [[VAR_35_]], [[RES_41_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:           [[RES_42_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi32>
// CHECK:           krnl.memset [[RES_42_]], [[CST_0_]] : memref<16x64xi32>
// CHECK:           [[LOOP_10_:%.+]]:3 = krnl.define_loops 3
// CHECK:           [[BLOCK_TILE__10_:%.+]], [[BLOCK_IN__10_:%.+]] = krnl.block [[LOOP_10_]]#0 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__11_:%.+]], [[BLOCK_IN__11_:%.+]] = krnl.block [[LOOP_10_]]#1 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__12_:%.+]], [[BLOCK_IN__12_:%.+]] = krnl.block [[LOOP_10_]]#2 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.permute([[BLOCK_TILE__10_]], [[BLOCK_IN__10_]], [[BLOCK_TILE__10_]]_61, [[BLOCK_IN__10_]]_62, [[BLOCK_TILE__10_]]_63, [[BLOCK_IN__10_]]_64) [0, 3, 1, 4, 2, 5] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
// CHECK:           krnl.iterate([[BLOCK_TILE__10_]], [[BLOCK_TILE__10_]]_61, [[BLOCK_TILE__10_]]_63) with ([[LOOP_10_]]#0 -> [[I_10_:%.+]] = [[CST_0_1_]] to [[CST_16_]], [[LOOP_10_]]#1 -> [[I_11_:%.+]] = [[CST_0_1_]] to [[CST_64_]], [[LOOP_10_]]#2 -> [[I_12_:%.+]] = [[CST_0_1_]] to [[CST_32_]]){
// CHECK:             [[LOOP_9_:%.+]]:3 = krnl.get_induction_var_value([[BLOCK_TILE__10_]], [[BLOCK_TILE__10_]]_61, [[BLOCK_TILE__10_]]_63) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.matmul [[RES_16_]]{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, [[RES_35_]]{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, [[RES_42_]]{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, ([[BLOCK_IN__10_]], [[BLOCK_IN__10_]]_62, [[BLOCK_IN__10_]]_64), ([[LOOP_9_]]#0, [[LOOP_9_]]#1, [[LOOP_9_]]#2), ([[CST_16_]], [[CST_64_]], [[CST_32_]]) {aTileSize = [], bTileSize = [], cTileSize = [], computeTileSize = [4, 8, 8]} : memref<16x32xi32>, memref<32x64xi32>, memref<16x64xi32>, (!krnl.loop, !krnl.loop, !krnl.loop)
// CHECK:           }
// CHECK-DAG:       [[RES_43_:%.+]] = memref.alloc() {{.*}}: memref<16x64xf32>
// CHECK-DAG:       [[RES_44_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_44_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_67_:%.+]] = memref.reshape [[RES_42_]]([[RES_44_]]) : (memref<16x64xi32>, memref<1xindex>) -> memref<1024xi32>
// CHECK-DAG:       [[RES_45_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_45_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_69_:%.+]] = memref.reshape [[RES_43_]]([[RES_45_]]) : (memref<16x64xf32>, memref<1xindex>) -> memref<1024xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_11_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__13_:%.+]], [[BLOCK_IN__13_:%.+]] = krnl.block [[LOOP_11_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__13_]]) with ([[LOOP_11_]] -> [[I_13_:%.+]] = 0 to 1024){
// CHECK:               [[VAR_44_10_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__13_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_1_1_1_1_1_:%.+]] = vector.load [[VAR_reshape_67_]]{{.}}[[VAR_44_10_]]{{.}} : memref<1024xi32>, vector<32xi32>
// CHECK:               [[VAR_46_7_:%.+]] = arith.sitofp [[LOAD_VAR_reshape_MEM_1_1_1_1_1_]] : vector<32xi32> to vector<32xf32>
// CHECK:               vector.store [[VAR_46_7_]], [[VAR_reshape_69_]]{{.}}[[VAR_44_10_]]{{.}} : memref<1024xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_46_:%.+]] = memref.alloc() {{.*}}: memref<1xf32>
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK:           [[VAR_39_:%.+]] = arith.mulf [[LOAD_PARAM_1_MEM_]], [[LOAD_PARAM_4_MEM_]] : f32
// CHECK:           krnl.store [[VAR_39_]], [[RES_46_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[RES_47_:%.+]] = memref.alloc() {{.*}}: memref<1xf32>
// CHECK-DAG:       [[LOAD_RES_46_MEM_:%.+]] = krnl.load [[RES_46_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_PARAM_6_MEM_:%.+]] = krnl.load [[PARAM_6_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK:           [[VAR_42_:%.+]] = arith.divf [[LOAD_RES_46_MEM_]], [[LOAD_PARAM_6_MEM_]] : f32
// CHECK:           krnl.store [[VAR_42_]], [[RES_47_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[RES_48_:%.+]] = memref.alloc() {{.*}}: memref<16x64xf32>
// CHECK-DAG:       [[RES_49_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_49_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_74_:%.+]] = memref.reshape [[RES_43_]]([[RES_49_]]) : (memref<16x64xf32>, memref<1xindex>) -> memref<1024xf32>
// CHECK-DAG:       [[RES_50_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_50_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_76_:%.+]] = memref.reshape [[RES_48_]]([[RES_50_]]) : (memref<16x64xf32>, memref<1xindex>) -> memref<1024xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_12_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__14_:%.+]], [[BLOCK_IN__14_:%.+]] = krnl.block [[LOOP_12_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__14_]]) with ([[LOOP_12_]] -> [[I_14_:%.+]] = 0 to 1024){
// CHECK:               [[VAR_44_11_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__14_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_1_1_1_1_1_1_:%.+]] = vector.load [[VAR_reshape_74_]]{{.}}[[VAR_44_11_]]{{.}} : memref<1024xf32>, vector<32xf32>
// CHECK-DAG:           [[VAR_46_7_:%.+]] = krnl.load [[RES_47_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK:               [[VAR_47_6_:%.+]] = vector.broadcast [[VAR_46_7_]] : f32 to vector<32xf32>
// CHECK:               [[VAR_48_4_:%.+]] = arith.mulf [[LOAD_VAR_reshape_MEM_1_1_1_1_1_1_]], [[VAR_47_6_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_48_4_]], [[VAR_reshape_76_]]{{.}}[[VAR_44_11_]]{{.}} : memref<1024xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_51_:%.+]] = memref.alloc() {{.*}}: memref<16x64xf32>
// CHECK-DAG:       [[RES_52_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_52_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_79_:%.+]] = memref.reshape [[RES_48_]]([[RES_52_]]) : (memref<16x64xf32>, memref<1xindex>) -> memref<1024xf32>
// CHECK-DAG:       [[RES_53_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_53_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_81_:%.+]] = memref.reshape [[RES_51_]]([[RES_53_]]) : (memref<16x64xf32>, memref<1xindex>) -> memref<1024xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_13_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__15_:%.+]], [[BLOCK_IN__15_:%.+]] = krnl.block [[LOOP_13_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__15_]]) with ([[LOOP_13_]] -> [[I_15_:%.+]] = 0 to 1024){
// CHECK:               [[VAR_44_12_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__15_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_1_1_1_1_1_1_:%.+]] = vector.load [[VAR_reshape_79_]]{{.}}[[VAR_44_12_]]{{.}} : memref<1024xf32>, vector<32xf32>
// CHECK:               [[VAR_46_8_:%.+]] = vector.shape_cast [[LOAD_VAR_reshape_MEM_1_1_1_1_1_1_]] : vector<32xf32> to vector<8x4xf32>
// CHECK:               [[VAR_47_7_:%.+]] = vector.extract [[VAR_46_8_]][0] : vector<4xf32> from vector<8x4xf32>
// CHECK-DAG:           [[VAR_48_5_:%.+]] = "krnl.round_even"([[VAR_47_7_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:           [[VAR_49_:%.+]] = vector.extract [[VAR_46_8_]][1] : vector<4xf32> from vector<8x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_50_:%.+]] = "krnl.round_even"([[VAR_49_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:           [[VAR_51_:%.+]] = vector.extract [[VAR_46_8_]][2] : vector<4xf32> from vector<8x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_52_:%.+]] = "krnl.round_even"([[VAR_51_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:           [[VAR_53_:%.+]] = vector.extract [[VAR_46_8_]][3] : vector<4xf32> from vector<8x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_54_:%.+]] = "krnl.round_even"([[VAR_53_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:           [[VAR_55_:%.+]] = vector.extract [[VAR_46_8_]][4] : vector<4xf32> from vector<8x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_56_:%.+]] = "krnl.round_even"([[VAR_55_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:           [[VAR_57_:%.+]] = vector.extract [[VAR_46_8_]][5] : vector<4xf32> from vector<8x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_58_:%.+]] = "krnl.round_even"([[VAR_57_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:           [[VAR_59_:%.+]] = vector.extract [[VAR_46_8_]][6] : vector<4xf32> from vector<8x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_60_:%.+]] = "krnl.round_even"([[VAR_59_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:           [[VAR_61_:%.+]] = vector.extract [[VAR_46_8_]][7] : vector<4xf32> from vector<8x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_62_:%.+]] = "krnl.round_even"([[VAR_61_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:           [[VAR_63_:%.+]]:4 = vector.to_elements [[VAR_48_5_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_64_:%.+]]:4 = vector.to_elements [[VAR_50_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_65_:%.+]]:4 = vector.to_elements [[VAR_52_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_66_:%.+]]:4 = vector.to_elements [[VAR_54_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_67_:%.+]]:4 = vector.to_elements [[VAR_56_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_68_:%.+]]:4 = vector.to_elements [[VAR_58_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_69_:%.+]]:4 = vector.to_elements [[VAR_60_]] : vector<4xf32>
// CHECK:               [[VAR_70_:%.+]]:4 = vector.to_elements [[VAR_62_]] : vector<4xf32>
// CHECK:               [[VAR_71_:%.+]] = vector.from_elements [[VAR_63_]]#0, [[VAR_63_]]#1, [[VAR_63_]]#2, [[VAR_63_]]#3, [[VAR_64_]]#0, [[VAR_64_]]#1, [[VAR_64_]]#2, [[VAR_64_]]#3, [[VAR_65_]]#0, [[VAR_65_]]#1, [[VAR_65_]]#2, [[VAR_65_]]#3, [[VAR_66_]]#0, [[VAR_66_]]#1, [[VAR_66_]]#2, [[VAR_66_]]#3, [[VAR_67_]]#0, [[VAR_67_]]#1, [[VAR_67_]]#2, [[VAR_67_]]#3, [[VAR_68_]]#0, [[VAR_68_]]#1, [[VAR_68_]]#2, [[VAR_68_]]#3, [[VAR_69_]]#0, [[VAR_69_]]#1, [[VAR_69_]]#2, [[VAR_69_]]#3, [[VAR_70_]]#0, [[VAR_70_]]#1, [[VAR_70_]]#2, [[VAR_70_]]#3 : vector<8x4xf32>
// CHECK:               [[VAR_72_:%.+]] = vector.shape_cast [[VAR_71_]] : vector<8x4xf32> to vector<32xf32>
// CHECK:               vector.store [[VAR_72_]], [[VAR_reshape_81_]]{{.}}[[VAR_44_12_]]{{.}} : memref<1024xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_54_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi32>
// CHECK-DAG:       [[RES_55_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_55_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_84_:%.+]] = memref.reshape [[RES_51_]]([[RES_55_]]) : (memref<16x64xf32>, memref<1xindex>) -> memref<1024xf32>
// CHECK-DAG:       [[RES_56_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_56_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_86_:%.+]] = memref.reshape [[RES_54_]]([[RES_56_]]) : (memref<16x64xi32>, memref<1xindex>) -> memref<1024xi32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_14_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__16_:%.+]], [[BLOCK_IN__16_:%.+]] = krnl.block [[LOOP_14_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__16_]]) with ([[LOOP_14_]] -> [[I_16_:%.+]] = 0 to 1024){
// CHECK:               [[VAR_44_13_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__16_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_1_1_1_1_1_1_1_:%.+]] = vector.load [[VAR_reshape_84_]]{{.}}[[VAR_44_13_]]{{.}} : memref<1024xf32>, vector<32xf32>
// CHECK:               [[VAR_46_9_:%.+]] = arith.fptosi [[LOAD_VAR_reshape_MEM_1_1_1_1_1_1_1_]] : vector<32xf32> to vector<32xi32>
// CHECK:               vector.store [[VAR_46_9_]], [[VAR_reshape_86_]]{{.}}[[VAR_44_13_]]{{.}} : memref<1024xi32>, vector<32xi32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_57_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi32>
// CHECK-DAG:       [[RES_58_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_58_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_89_:%.+]] = memref.reshape [[RES_54_]]([[RES_58_]]) : (memref<16x64xi32>, memref<1xindex>) -> memref<1024xi32>
// CHECK-DAG:       [[RES_59_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_59_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_91_:%.+]] = memref.reshape [[RES_57_]]([[RES_59_]]) : (memref<16x64xi32>, memref<1xindex>) -> memref<1024xi32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_15_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__17_:%.+]], [[BLOCK_IN__17_:%.+]] = krnl.block [[LOOP_15_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__17_]]) with ([[LOOP_15_]] -> [[I_17_:%.+]] = 0 to 1024){
// CHECK:               [[VAR_44_14_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__17_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_1_1_1_1_1_1_1_:%.+]] = vector.load [[VAR_reshape_89_]]{{.}}[[VAR_44_14_]]{{.}} : memref<1024xi32>, vector<32xi32>
// CHECK-DAG:           [[VAR_46_9_:%.+]] = krnl.load [[RES_41_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:               [[VAR_47_8_:%.+]] = vector.broadcast [[VAR_46_9_]] : i32 to vector<32xi32>
// CHECK:               [[VAR_48_6_:%.+]] = arith.addi [[LOAD_VAR_reshape_MEM_1_1_1_1_1_1_1_]], [[VAR_47_8_]] : vector<32xi32>
// CHECK:               vector.store [[VAR_48_6_]], [[VAR_reshape_91_]]{{.}}[[VAR_44_14_]]{{.}} : memref<1024xi32>, vector<32xi32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_60_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi32>
// CHECK-DAG:       [[RES_61_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_61_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_94_:%.+]] = memref.reshape [[RES_57_]]([[RES_61_]]) : (memref<16x64xi32>, memref<1xindex>) -> memref<1024xi32>
// CHECK-DAG:       [[RES_62_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_62_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_96_:%.+]] = memref.reshape [[RES_60_]]([[RES_62_]]) : (memref<16x64xi32>, memref<1xindex>) -> memref<1024xi32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_16_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__18_:%.+]], [[BLOCK_IN__18_:%.+]] = krnl.block [[LOOP_16_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__18_]]) with ([[LOOP_16_]] -> [[I_18_:%.+]] = 0 to 1024){
// CHECK:               [[VAR_44_15_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__18_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_1_1_1_1_1_1_1_1_:%.+]] = vector.load [[VAR_reshape_94_]]{{.}}[[VAR_44_15_]]{{.}} : memref<1024xi32>, vector<32xi32>
// CHECK-DAG:           [[VAR_46_9_1_:%.+]] = krnl.load [[VAR_0_]][] : memref<i32>
// CHECK:               [[VAR_47_9_:%.+]] = vector.broadcast [[VAR_46_9_1_]] : i32 to vector<32xi32>
// CHECK:               [[VAR_48_7_:%.+]] = arith.addi [[LOAD_VAR_reshape_MEM_1_1_1_1_1_1_1_1_]], [[VAR_47_9_]] : vector<32xi32>
// CHECK:               vector.store [[VAR_48_7_]], [[VAR_reshape_96_]]{{.}}[[VAR_44_15_]]{{.}} : memref<1024xi32>, vector<32xi32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_63_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi8>
// CHECK-DAG:       [[RES_64_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_64_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_99_:%.+]] = memref.reshape [[RES_60_]]([[RES_64_]]) : (memref<16x64xi32>, memref<1xindex>) -> memref<1024xi32>
// CHECK-DAG:       [[RES_65_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_65_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_101_:%.+]] = memref.reshape [[RES_63_]]([[RES_65_]]) : (memref<16x64xi8>, memref<1xindex>) -> memref<1024xi8>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_17_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__19_:%.+]], [[BLOCK_IN__19_:%.+]] = krnl.block [[LOOP_17_]] 128 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__19_]]) with ([[LOOP_17_]] -> [[I_19_:%.+]] = 0 to 1024){
// CHECK:               [[VAR_44_16_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__19_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_1_1_1_1_1_1_1_1_:%.+]] = vector.load [[VAR_reshape_99_]]{{.}}[[VAR_44_16_]]{{.}} : memref<1024xi32>, vector<128xi32>
// CHECK:               [[VAR_46_10_:%.+]] = arith.trunci [[LOAD_VAR_reshape_MEM_1_1_1_1_1_1_1_1_]] : vector<128xi32> to vector<128xi8>
// CHECK:               vector.store [[VAR_46_10_]], [[VAR_reshape_101_]]{{.}}[[VAR_44_16_]]{{.}} : memref<1024xi8>, vector<128xi8>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_66_:%.+]] = memref.alloc() {{.*}}: memref<16x64xui8>
// CHECK-DAG:       [[RES_67_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_67_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_104_:%.+]] = memref.reshape [[RES_63_]]([[RES_67_]]) : (memref<16x64xi8>, memref<1xindex>) -> memref<1024xi8>
// CHECK-DAG:       [[RES_68_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_68_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_106_:%.+]] = memref.reshape [[RES_66_]]([[RES_68_]]) : (memref<16x64xui8>, memref<1xindex>) -> memref<1024xui8>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_18_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__20_:%.+]], [[BLOCK_IN__20_:%.+]] = krnl.block [[LOOP_18_]] 128 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__20_]]) with ([[LOOP_18_]] -> [[I_20_:%.+]] = 0 to 1024){
// CHECK:               [[VAR_44_17_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__20_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_reshape_MEM_1_1_1_1_1_1_1_1_1_:%.+]] = vector.load [[VAR_reshape_104_]]{{.}}[[VAR_44_17_]]{{.}} : memref<1024xi8>, vector<128xi8>
// CHECK:               [[VAR_46_11_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_VAR_reshape_MEM_1_1_1_1_1_1_1_1_1_]] : vector<128xi8> to vector<128xui8>
// CHECK:               vector.store [[VAR_46_11_]], [[VAR_reshape_106_]]{{.}}[[VAR_44_17_]]{{.}} : memref<1024xui8>, vector<128xui8>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_66_]] : memref<16x64xui8>
// CHECK:         }
}
