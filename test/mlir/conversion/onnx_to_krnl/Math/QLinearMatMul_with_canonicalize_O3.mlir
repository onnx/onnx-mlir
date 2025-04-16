// RUN: onnx-mlir-opt -O3 --mtriple=s390x-ibm-loz --march=z16 --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

func.func @qlinearmatmul_i8_f32(%arg0: tensor<16x32xi8>, %arg1: tensor<1xf32>, %arg2: tensor<1xi8>, %arg3: tensor<32x64xi8>, %arg4: tensor<1xf32>, %arg5: tensor<1xi8>, %arg6: tensor<1xf32>, %arg7: tensor<1xi8>) -> (tensor<16x64xi8>) {
    %0 = "onnx.QLinearMatMul"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7) : (tensor<16x32xi8>, tensor<1xf32>, tensor<1xi8>, tensor<32x64xi8>, tensor<1xf32>, tensor<1xi8>, tensor<1xf32>, tensor<1xi8>) -> tensor<16x64xi8>
    return %0 : tensor<16x64xi8>

// CHECK-LABEL:  func.func @qlinearmatmul_i8_f32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<16x32xi8>, [[PARAM_1_:%.+]]: memref<1xf32>, [[PARAM_2_:%.+]]: memref<1xi8>, [[PARAM_3_:%.+]]: memref<32x64xi8>, [[PARAM_4_:%.+]]: memref<1xf32>, [[PARAM_5_:%.+]]: memref<1xi8>, [[PARAM_6_:%.+]]: memref<1xf32>, [[PARAM_7_:%.+]]: memref<1xi8>) -> memref<16x64xi8> {
// CHECK-DAG:       [[CST_1024_:%.+]] = arith.constant 1024 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_2048_:%.+]] = arith.constant 2048 : index
// CHECK-DAG:       [[CST_512_:%.+]] = arith.constant 512 : index
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_16_:%.+]] = arith.constant 16 : index
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<16x32xi32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 16, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 32){
// CHECK:             [[VAR_18_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_18_]]#0, [[VAR_18_]]#1] : memref<16x32xi8>
// CHECK:             [[VAR_20_:%.+]] = arith.extsi [[LOAD_PARAM_0_MEM_]] : i8 to i32
// CHECK:             krnl.store [[VAR_20_]], [[RES_]]{{.}}[[VAR_18_]]#0, [[VAR_18_]]#1] : memref<16x32xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xi32>
// CHECK-DAG:       [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi8>
// CHECK:           [[VAR_2_:%.+]] = arith.extsi [[LOAD_PARAM_2_MEM_]] : i8 to i32
// CHECK:           krnl.store [[VAR_2_]], [[RES_1_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<16x32xi32>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_512_]], [[RES_3_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[RES_]]([[RES_]]_2) : (memref<16x32xi32>, memref<1xindex>) -> memref<512xi32>
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_512_]], [[RES_4_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_4_:%.+]] = memref.reshape [[RES_2_]]([[RES_4_]]) : (memref<16x32xi32>, memref<1xindex>) -> memref<512xi32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_1_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 512){
// CHECK:               [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[VAR_20_1_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[LOAD_PARAM_0_MEM_1_]]{{.}} : memref<512xi32>, vector<32xi32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:               [[VAR_22_:%.+]] = vector.splat [[LOAD_RES_1_MEM_]] : vector<32xi32>
// CHECK:               [[VAR_23_:%.+]] = arith.subi [[VAR_20_1_]], [[VAR_22_]] : vector<32xi32>
// CHECK:               vector.store [[VAR_23_]], [[VAR_reshape_4_]]{{.}}[[LOAD_PARAM_0_MEM_1_]]{{.}} : memref<512xi32>, vector<32xi32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi32>
// CHECK-DAG:       [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 32, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 64){
// CHECK:             [[VAR_18_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_3_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1] : memref<32x64xi8>
// CHECK:             [[VAR_20_2_:%.+]] = arith.extsi [[LOAD_PARAM_0_MEM_1_]] : i8 to i32
// CHECK:             krnl.store [[VAR_20_2_]], [[RES_5_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1] : memref<32x64xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<1xi32>
// CHECK-DAG:       [[LOAD_PARAM_5_MEM_:%.+]] = krnl.load [[PARAM_5_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi8>
// CHECK:           [[VAR_5_:%.+]] = arith.extsi [[LOAD_PARAM_5_MEM_]] : i8 to i32
// CHECK:           krnl.store [[VAR_5_]], [[RES_6_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi32>
// CHECK-DAG:       [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_8_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_9_:%.+]] = memref.reshape [[RES_5_]]([[RES_8_]]) : (memref<32x64xi32>, memref<1xindex>) -> memref<2048xi32>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_9_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_11_:%.+]] = memref.reshape [[RES_7_]]([[RES_9_]]) : (memref<32x64xi32>, memref<1xindex>) -> memref<2048xi32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_3_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__1_]]) with ([[LOOP_3_]] -> [[I_5_:%.+]] = 0 to 2048){
// CHECK:               [[LOAD_PARAM_0_MEM_1_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[VAR_20_2_:%.+]] = vector.load [[VAR_reshape_9_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]{{.}} : memref<2048xi32>, vector<32xi32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_6_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:               [[VAR_22_1_:%.+]] = vector.splat [[LOAD_RES_1_MEM_1_]] : vector<32xi32>
// CHECK:               [[VAR_23_1_:%.+]] = arith.subi [[VAR_20_2_]], [[VAR_22_1_]] : vector<32xi32>
// CHECK:               vector.store [[VAR_23_1_]], [[VAR_reshape_11_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]{{.}} : memref<2048xi32>, vector<32xi32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_10_:%.+]] = memref.alloc() {{.*}}: memref<1xi32>
// CHECK-DAG:       [[LOAD_PARAM_7_MEM_:%.+]] = krnl.load [[PARAM_7_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi8>
// CHECK:           [[VAR_7_:%.+]] = arith.extsi [[LOAD_PARAM_7_MEM_]] : i8 to i32
// CHECK:           krnl.store [[VAR_7_]], [[RES_10_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:           [[RES_11_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi32>
// CHECK:           krnl.memset [[RES_11_]], [[CST_0_]] : memref<16x64xi32>
// CHECK:           [[LOOP_4_:%.+]]:3 = krnl.define_loops 3
// CHECK:           [[BLOCK_TILE__2_:%.+]], [[BLOCK_IN__2_:%.+]] = krnl.block [[LOOP_4_]]#0 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__3_:%.+]], [[BLOCK_IN__3_:%.+]] = krnl.block [[LOOP_4_]]#1 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__4_:%.+]], [[BLOCK_IN__4_:%.+]] = krnl.block [[LOOP_4_]]#2 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.permute([[BLOCK_TILE__2_]], [[BLOCK_IN__2_]], [[BLOCK_TILE__2_]]_14, [[BLOCK_IN__2_]]_15, [[BLOCK_TILE__2_]]_16, [[BLOCK_IN__2_]]_17) [0, 3, 1, 4, 2, 5] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
// CHECK:           krnl.iterate([[BLOCK_TILE__2_]], [[BLOCK_TILE__2_]]_14, [[BLOCK_TILE__2_]]_16) with ([[LOOP_4_]]#0 -> [[I_6_:%.+]] = [[CST_0_1_]] to [[CST_16_]], [[LOOP_4_]]#1 -> [[I_7_:%.+]] = [[CST_0_1_]] to [[CST_64_]], [[LOOP_4_]]#2 -> [[I_8_:%.+]] = [[CST_0_1_]] to [[CST_32_]]){
// CHECK:             [[VAR_18_2_:%.+]]:3 = krnl.get_induction_var_value([[BLOCK_TILE__2_]], [[BLOCK_TILE__2_]]_14, [[BLOCK_TILE__2_]]_16) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.matmul [[RES_2_]]{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, [[RES_7_]]{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, [[RES_2_]]3{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, ([[BLOCK_IN__2_]], [[BLOCK_IN__2_]]_15, [[BLOCK_IN__2_]]_17), ([[VAR_18_2_]]#0, [[VAR_18_2_]]#1, [[VAR_18_2_]]#2), ([[CST_16_]], [[CST_64_]], [[CST_32_]]) {aTileSize = [], bTileSize = [], cTileSize = [], computeTileSize = [4, 8, 8]} : memref<16x32xi32>, memref<32x64xi32>, memref<16x64xi32>, (!krnl.loop, !krnl.loop, !krnl.loop)
// CHECK:           }
// CHECK-DAG:       [[RES_12_:%.+]] = memref.alloc() {{.*}}: memref<16x64xf32>
// CHECK-DAG:       [[LOOP_5_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_5_]]#0, [[LOOP_5_]]#1) with ([[LOOP_5_]]#0 -> [[I_9_:%.+]] = 0 to 16, [[LOOP_5_]]#1 -> [[I_10_:%.+]] = 0 to 64){
// CHECK:             [[VAR_18_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_5_]]#0, [[LOOP_5_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_1_:%.+]] = krnl.load [[RES_11_]]{{.}}[[VAR_18_3_]]#0, [[VAR_18_3_]]#1] : memref<16x64xi32>
// CHECK:             [[VAR_20_3_:%.+]] = arith.sitofp [[LOAD_PARAM_0_MEM_1_1_]] : i32 to f32
// CHECK:             krnl.store [[VAR_20_3_]], [[RES_12_]]{{.}}[[VAR_18_3_]]#0, [[VAR_18_3_]]#1] : memref<16x64xf32>
// CHECK:           }
// CHECK-DAG:       [[RES_13_:%.+]] = memref.alloc() {{.*}}: memref<1xf32>
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK:           [[VAR_12_:%.+]] = arith.mulf [[LOAD_PARAM_1_MEM_]], [[LOAD_PARAM_4_MEM_]] : f32
// CHECK:           krnl.store [[VAR_12_]], [[RES_13_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[RES_14_:%.+]] = memref.alloc() {{.*}}: memref<1xf32>
// CHECK-DAG:       [[LOAD_RES_13_MEM_:%.+]] = krnl.load [[RES_13_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_PARAM_6_MEM_:%.+]] = krnl.load [[PARAM_6_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK:           [[VAR_15_:%.+]] = arith.divf [[LOAD_RES_13_MEM_]], [[LOAD_PARAM_6_MEM_]] : f32
// CHECK:           krnl.store [[VAR_15_]], [[RES_14_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[RES_15_:%.+]] = memref.alloc() {{.*}}: memref<16x64xf32>
// CHECK-DAG:       [[RES_16_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_16_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_23_:%.+]] = memref.reshape [[RES_12_]]([[RES_16_]]) : (memref<16x64xf32>, memref<1xindex>) -> memref<1024xf32>
// CHECK-DAG:       [[RES_17_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_17_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_25_:%.+]] = memref.reshape [[RES_15_]]([[RES_17_]]) : (memref<16x64xf32>, memref<1xindex>) -> memref<1024xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_6_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__5_:%.+]], [[BLOCK_IN__5_:%.+]] = krnl.block [[LOOP_6_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__5_]]) with ([[LOOP_6_]] -> [[I_11_:%.+]] = 0 to 1024){
// CHECK:               [[LOAD_PARAM_0_MEM_1_1_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__5_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[VAR_20_3_:%.+]] = vector.load [[VAR_reshape_23_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]{{.}} : memref<1024xf32>, vector<32xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_14_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK:               [[VAR_22_2_:%.+]] = vector.splat [[LOAD_RES_1_MEM_1_]] : vector<32xf32>
// CHECK:               [[VAR_23_2_:%.+]] = arith.mulf [[VAR_20_3_]], [[VAR_22_2_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_23_2_]], [[VAR_reshape_25_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]{{.}} : memref<1024xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_18_:%.+]] = memref.alloc() {{.*}}: memref<16x64xf32>
// CHECK-DAG:       [[RES_19_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_19_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_28_:%.+]] = memref.reshape [[RES_15_]]([[RES_19_]]) : (memref<16x64xf32>, memref<1xindex>) -> memref<1024xf32>
// CHECK-DAG:       [[RES_20_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_20_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_30_:%.+]] = memref.reshape [[RES_18_]]([[RES_20_]]) : (memref<16x64xf32>, memref<1xindex>) -> memref<1024xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_7_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__6_:%.+]], [[BLOCK_IN__6_:%.+]] = krnl.block [[LOOP_7_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__6_]]) with ([[LOOP_7_]] -> [[I_12_:%.+]] = 0 to 1024){
// CHECK:               [[LOAD_PARAM_0_MEM_1_1_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__6_]]) : (!krnl.loop) -> index
// CHECK:               [[VAR_20_3_1_:%.+]] = vector.load [[VAR_reshape_28_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]{{.}} : memref<1024xf32>, vector<32xf32>
// CHECK:               [[LOAD_RES_1_MEM_1_1_:%.+]] = vector.shape_cast [[VAR_20_3_1_]] : vector<32xf32> to vector<8x4xf32>
// CHECK:               [[VAR_22_3_:%.+]] = vector.extract [[LOAD_RES_1_MEM_1_1_]][0] : vector<4xf32> from vector<8x4xf32>
// CHECK:               [[VAR_23_3_:%.+]] = "krnl.round_even"([[VAR_22_3_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:           [[VAR_24_:%.+]] = vector.insert [[VAR_23_3_]], [[LOAD_RES_1_MEM_1_1_]] [0] : vector<4xf32> into vector<8x4xf32>
// CHECK-DAG:           [[VAR_25_:%.+]] = vector.extract [[LOAD_RES_1_MEM_1_1_]][1] : vector<4xf32> from vector<8x4xf32>
// CHECK:               [[VAR_26_:%.+]] = "krnl.round_even"([[VAR_25_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:           [[VAR_27_:%.+]] = vector.insert [[VAR_26_]], [[VAR_24_]] [1] : vector<4xf32> into vector<8x4xf32>
// CHECK-DAG:           [[VAR_28_:%.+]] = vector.extract [[LOAD_RES_1_MEM_1_1_]][2] : vector<4xf32> from vector<8x4xf32>
// CHECK:               [[VAR_29_:%.+]] = "krnl.round_even"([[VAR_28_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:           [[VAR_30_:%.+]] = vector.insert [[VAR_29_]], [[VAR_27_]] [2] : vector<4xf32> into vector<8x4xf32>
// CHECK-DAG:           [[VAR_31_:%.+]] = vector.extract [[LOAD_RES_1_MEM_1_1_]][3] : vector<4xf32> from vector<8x4xf32>
// CHECK:               [[VAR_32_:%.+]] = "krnl.round_even"([[VAR_31_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:           [[VAR_33_:%.+]] = vector.insert [[VAR_32_]], [[VAR_30_]] [3] : vector<4xf32> into vector<8x4xf32>
// CHECK-DAG:           [[VAR_34_:%.+]] = vector.extract [[LOAD_RES_1_MEM_1_1_]][4] : vector<4xf32> from vector<8x4xf32>
// CHECK:               [[VAR_35_:%.+]] = "krnl.round_even"([[VAR_34_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:           [[VAR_36_:%.+]] = vector.insert [[VAR_35_]], [[VAR_33_]] [4] : vector<4xf32> into vector<8x4xf32>
// CHECK-DAG:           [[VAR_37_:%.+]] = vector.extract [[LOAD_RES_1_MEM_1_1_]][5] : vector<4xf32> from vector<8x4xf32>
// CHECK:               [[VAR_38_:%.+]] = "krnl.round_even"([[VAR_37_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:           [[VAR_39_:%.+]] = vector.insert [[VAR_38_]], [[VAR_36_]] [5] : vector<4xf32> into vector<8x4xf32>
// CHECK-DAG:           [[VAR_40_:%.+]] = vector.extract [[LOAD_RES_1_MEM_1_1_]][6] : vector<4xf32> from vector<8x4xf32>
// CHECK:               [[VAR_41_:%.+]] = "krnl.round_even"([[VAR_40_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:           [[VAR_42_:%.+]] = vector.insert [[VAR_41_]], [[VAR_39_]] [6] : vector<4xf32> into vector<8x4xf32>
// CHECK-DAG:           [[VAR_43_:%.+]] = vector.extract [[LOAD_RES_1_MEM_1_1_]][7] : vector<4xf32> from vector<8x4xf32>
// CHECK:               [[VAR_44_:%.+]] = "krnl.round_even"([[VAR_43_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK:               [[VAR_45_:%.+]] = vector.insert [[VAR_44_]], [[VAR_42_]] [7] : vector<4xf32> into vector<8x4xf32>
// CHECK:               [[VAR_46_:%.+]] = vector.shape_cast [[VAR_45_]] : vector<8x4xf32> to vector<32xf32>
// CHECK:               vector.store [[VAR_46_]], [[VAR_reshape_30_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]{{.}} : memref<1024xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_21_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi32>
// CHECK-DAG:       [[LOOP_8_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_8_]]#0, [[LOOP_8_]]#1) with ([[LOOP_8_]]#0 -> [[I_13_:%.+]] = 0 to 16, [[LOOP_8_]]#1 -> [[I_14_:%.+]] = 0 to 64){
// CHECK:             [[VAR_18_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_8_]]#0, [[LOOP_8_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_1_1_1_:%.+]] = krnl.load [[RES_18_]]{{.}}[[VAR_18_4_]]#0, [[VAR_18_4_]]#1] : memref<16x64xf32>
// CHECK:             [[VAR_20_4_:%.+]] = arith.fptosi [[LOAD_PARAM_0_MEM_1_1_1_1_]] : f32 to i32
// CHECK:             krnl.store [[VAR_20_4_]], [[RES_21_]]{{.}}[[VAR_18_4_]]#0, [[VAR_18_4_]]#1] : memref<16x64xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_22_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi32>
// CHECK-DAG:       [[RES_23_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_23_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_34_:%.+]] = memref.reshape [[RES_21_]]([[RES_23_]]) : (memref<16x64xi32>, memref<1xindex>) -> memref<1024xi32>
// CHECK-DAG:       [[RES_24_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_24_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_36_:%.+]] = memref.reshape [[RES_22_]]([[RES_24_]]) : (memref<16x64xi32>, memref<1xindex>) -> memref<1024xi32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_9_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__7_:%.+]], [[BLOCK_IN__7_:%.+]] = krnl.block [[LOOP_9_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__7_]]) with ([[LOOP_9_]] -> [[I_15_:%.+]] = 0 to 1024){
// CHECK:               [[LOAD_PARAM_0_MEM_1_1_1_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__7_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[VAR_20_4_:%.+]] = vector.load [[VAR_reshape_34_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_1_]]{{.}} : memref<1024xi32>, vector<32xi32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_1_1_:%.+]] = krnl.load [[RES_10_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:               [[VAR_22_4_:%.+]] = vector.splat [[LOAD_RES_1_MEM_1_1_]] : vector<32xi32>
// CHECK:               [[VAR_23_4_:%.+]] = arith.addi [[VAR_20_4_]], [[VAR_22_4_]] : vector<32xi32>
// CHECK:               vector.store [[VAR_23_4_]], [[VAR_reshape_36_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_1_]]{{.}} : memref<1024xi32>, vector<32xi32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_25_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi8>
// CHECK-DAG:       [[LOOP_10_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_10_]]#0, [[LOOP_10_]]#1) with ([[LOOP_10_]]#0 -> [[I_16_:%.+]] = 0 to 16, [[LOOP_10_]]#1 -> [[I_17_:%.+]] = 0 to 64){
// CHECK:             [[VAR_18_5_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_10_]]#0, [[LOOP_10_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_1_1_1_1_:%.+]] = krnl.load [[RES_22_]]{{.}}[[VAR_18_5_]]#0, [[VAR_18_5_]]#1] : memref<16x64xi32>
// CHECK:             [[VAR_20_5_:%.+]] = arith.trunci [[LOAD_PARAM_0_MEM_1_1_1_1_1_]] : i32 to i8
// CHECK:             krnl.store [[VAR_20_5_]], [[RES_25_]]{{.}}[[VAR_18_5_]]#0, [[VAR_18_5_]]#1] : memref<16x64xi8>
// CHECK:           }
// CHECK:           return [[RES_25_]] : memref<16x64xi8>
// CHECK:         }
}

//-----

func.func @qlinearmatmul_ui8_f32(%arg0: tensor<16x32xui8>, %arg1: tensor<1xf32>, %arg2: tensor<1xui8>, %arg3: tensor<32x64xui8>, %arg4: tensor<1xf32>, %arg5: tensor<1xui8>, %arg6: tensor<1xf32>, %arg7: tensor<1xui8>) -> (tensor<16x64xui8>) {
    %0 = "onnx.QLinearMatMul"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7) : (tensor<16x32xui8>, tensor<1xf32>, tensor<1xui8>, tensor<32x64xui8>, tensor<1xf32>, tensor<1xui8>, tensor<1xf32>, tensor<1xui8>) -> tensor<16x64xui8>
    return %0 : tensor<16x64xui8>

// CHECK-LABEL:  func.func @qlinearmatmul_ui8_f32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<16x32xui8>, [[PARAM_1_:%.+]]: memref<1xf32>, [[PARAM_2_:%.+]]: memref<1xui8>, [[PARAM_3_:%.+]]: memref<32x64xui8>, [[PARAM_4_:%.+]]: memref<1xf32>, [[PARAM_5_:%.+]]: memref<1xui8>, [[PARAM_6_:%.+]]: memref<1xf32>, [[PARAM_7_:%.+]]: memref<1xui8>) -> memref<16x64xui8> {
// CHECK-DAG:       [[CST_1024_:%.+]] = arith.constant 1024 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_2048_:%.+]] = arith.constant 2048 : index
// CHECK-DAG:       [[CST_512_:%.+]] = arith.constant 512 : index
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_16_:%.+]] = arith.constant 16 : index
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [], value = dense<128> : tensor<i32>} : () -> memref<i32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [], value = dense<128> : tensor<i16>} : () -> memref<i16>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<16x32xi32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 16, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 32){
// CHECK:             [[VAR_48_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_48_]]#0, [[VAR_48_]]#1] : memref<16x32xui8>
// CHECK:             [[VAR_50_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_]] : ui8 to i8
// CHECK-DAG:         [[VAR_51_:%.+]] = arith.extui [[VAR_50_]] : i8 to i16
// CHECK-DAG:         [[LOAD_VAR_1_MEM_:%.+]] = krnl.load [[VAR_1_]][] : memref<i16>
// CHECK:             [[VAR_53_:%.+]] = arith.subi [[VAR_51_]], [[LOAD_VAR_1_MEM_]] : i16
// CHECK:             [[VAR_54_:%.+]] = arith.trunci [[VAR_53_]] : i16 to i8
// CHECK:             [[VAR_55_:%.+]] = arith.extsi [[VAR_54_]] : i8 to i32
// CHECK:             krnl.store [[VAR_55_]], [[RES_]]{{.}}[[VAR_48_]]#0, [[VAR_48_]]#1] : memref<16x32xi32>
// CHECK:           }
// CHECK-DAG:       [[VAR_3_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [], value = dense<128> : tensor<i16>} : () -> memref<i16>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xi16>
// CHECK-DAG:       [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xui8>
// CHECK:           [[VAR_5_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_2_MEM_]] : ui8 to i8
// CHECK:           [[VAR_6_:%.+]] = arith.extui [[VAR_5_]] : i8 to i16
// CHECK:           krnl.store [[VAR_6_]], [[RES_1_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xi16>
// CHECK-DAG:       [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK-DAG:       [[LOAD_VAR_3_MEM_:%.+]] = krnl.load [[VAR_3_]][] : memref<i16>
// CHECK:           [[VAR_9_:%.+]] = arith.subi [[LOAD_RES_1_MEM_]], [[LOAD_VAR_3_MEM_]] : i16
// CHECK:           krnl.store [[VAR_9_]], [[RES_2_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1xi8>
// CHECK-DAG:       [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK:           [[VAR_11_:%.+]] = arith.trunci [[LOAD_RES_2_MEM_]] : i16 to i8
// CHECK:           krnl.store [[VAR_11_]], [[RES_3_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi8>
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<1xi32>
// CHECK-DAG:       [[LOAD_RES_3_MEM_:%.+]] = krnl.load [[RES_3_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi8>
// CHECK:           [[VAR_13_:%.+]] = arith.extsi [[LOAD_RES_3_MEM_]] : i8 to i32
// CHECK:           krnl.store [[VAR_13_]], [[RES_4_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<16x32xi32>
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_512_]], [[RES_6_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[RES_]]([[RES_]]_5) : (memref<16x32xi32>, memref<1xindex>) -> memref<512xi32>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_512_]], [[RES_7_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_7_:%.+]] = memref.reshape [[RES_5_]]([[RES_7_]]) : (memref<16x32xi32>, memref<1xindex>) -> memref<512xi32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_1_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 512){
// CHECK:               [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[VAR_50_1_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[LOAD_PARAM_0_MEM_1_]]{{.}} : memref<512xi32>, vector<32xi32>
// CHECK-DAG:           [[VAR_51_1_:%.+]] = krnl.load [[RES_4_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:               [[LOAD_VAR_1_MEM_1_:%.+]] = vector.splat [[VAR_51_1_]] : vector<32xi32>
// CHECK:               [[VAR_53_1_:%.+]] = arith.subi [[VAR_50_1_]], [[LOAD_VAR_1_MEM_1_]] : vector<32xi32>
// CHECK:               vector.store [[VAR_53_1_]], [[VAR_reshape_7_]]{{.}}[[LOAD_PARAM_0_MEM_1_]]{{.}} : memref<512xi32>, vector<32xi32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[VAR_14_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [], value = dense<128> : tensor<i16>} : () -> memref<i16>
// CHECK-DAG:       [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi32>
// CHECK-DAG:       [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 32, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 64){
// CHECK:             [[VAR_48_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_3_]]{{.}}[[VAR_48_1_]]#0, [[VAR_48_1_]]#1] : memref<32x64xui8>
// CHECK:             [[VAR_50_2_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_1_]] : ui8 to i8
// CHECK-DAG:         [[VAR_51_2_:%.+]] = arith.extui [[VAR_50_2_]] : i8 to i16
// CHECK-DAG:         [[LOAD_VAR_1_MEM_1_:%.+]] = krnl.load [[VAR_14_]][] : memref<i16>
// CHECK:             [[VAR_53_2_:%.+]] = arith.subi [[VAR_51_2_]], [[LOAD_VAR_1_MEM_1_]] : i16
// CHECK:             [[VAR_54_1_:%.+]] = arith.trunci [[VAR_53_2_]] : i16 to i8
// CHECK:             [[VAR_55_1_:%.+]] = arith.extsi [[VAR_54_1_]] : i8 to i32
// CHECK:             krnl.store [[VAR_55_1_]], [[RES_8_]]{{.}}[[VAR_48_1_]]#0, [[VAR_48_1_]]#1] : memref<32x64xi32>
// CHECK:           }
// CHECK-DAG:       [[VAR_16_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [], value = dense<128> : tensor<i16>} : () -> memref<i16>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc() {{.*}}: memref<1xi16>
// CHECK-DAG:       [[LOAD_PARAM_5_MEM_:%.+]] = krnl.load [[PARAM_5_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xui8>
// CHECK:           [[VAR_18_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_5_MEM_]] : ui8 to i8
// CHECK:           [[VAR_19_:%.+]] = arith.extui [[VAR_18_]] : i8 to i16
// CHECK:           krnl.store [[VAR_19_]], [[RES_9_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK-DAG:       [[RES_10_:%.+]] = memref.alloc() {{.*}}: memref<1xi16>
// CHECK-DAG:       [[LOAD_RES_9_MEM_:%.+]] = krnl.load [[RES_9_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK-DAG:       [[LOAD_VAR_16_MEM_:%.+]] = krnl.load [[VAR_16_]][] : memref<i16>
// CHECK:           [[VAR_22_:%.+]] = arith.subi [[LOAD_RES_9_MEM_]], [[LOAD_VAR_16_MEM_]] : i16
// CHECK:           krnl.store [[VAR_22_]], [[RES_10_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK-DAG:       [[RES_11_:%.+]] = memref.alloc() {{.*}}: memref<1xi8>
// CHECK-DAG:       [[LOAD_RES_10_MEM_:%.+]] = krnl.load [[RES_10_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK:           [[VAR_24_:%.+]] = arith.trunci [[LOAD_RES_10_MEM_]] : i16 to i8
// CHECK:           krnl.store [[VAR_24_]], [[RES_11_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi8>
// CHECK-DAG:       [[RES_12_:%.+]] = memref.alloc() {{.*}}: memref<1xi32>
// CHECK-DAG:       [[LOAD_RES_11_MEM_:%.+]] = krnl.load [[RES_11_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi8>
// CHECK:           [[VAR_26_:%.+]] = arith.extsi [[LOAD_RES_11_MEM_]] : i8 to i32
// CHECK:           krnl.store [[VAR_26_]], [[RES_12_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK-DAG:       [[RES_13_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi32>
// CHECK-DAG:       [[RES_14_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_14_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_15_:%.+]] = memref.reshape [[RES_8_]]([[RES_14_]]) : (memref<32x64xi32>, memref<1xindex>) -> memref<2048xi32>
// CHECK-DAG:       [[RES_15_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_2048_]], [[RES_15_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_17_:%.+]] = memref.reshape [[RES_13_]]([[RES_15_]]) : (memref<32x64xi32>, memref<1xindex>) -> memref<2048xi32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_3_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__1_]]) with ([[LOOP_3_]] -> [[I_5_:%.+]] = 0 to 2048){
// CHECK:               [[LOAD_PARAM_0_MEM_1_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[VAR_50_2_:%.+]] = vector.load [[VAR_reshape_15_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]{{.}} : memref<2048xi32>, vector<32xi32>
// CHECK-DAG:           [[VAR_51_2_:%.+]] = krnl.load [[RES_12_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:               [[LOAD_VAR_1_MEM_1_1_:%.+]] = vector.splat [[VAR_51_2_]] : vector<32xi32>
// CHECK:               [[VAR_53_3_:%.+]] = arith.subi [[VAR_50_2_]], [[LOAD_VAR_1_MEM_1_1_]] : vector<32xi32>
// CHECK:               vector.store [[VAR_53_3_]], [[VAR_reshape_17_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]{{.}} : memref<2048xi32>, vector<32xi32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[VAR_27_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [], value = dense<128> : tensor<i16>} : () -> memref<i16>
// CHECK-DAG:       [[RES_16_:%.+]] = memref.alloc() {{.*}}: memref<1xi16>
// CHECK-DAG:       [[LOAD_PARAM_7_MEM_:%.+]] = krnl.load [[PARAM_7_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xui8>
// CHECK:           [[VAR_29_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_7_MEM_]] : ui8 to i8
// CHECK:           [[VAR_30_:%.+]] = arith.extui [[VAR_29_]] : i8 to i16
// CHECK:           krnl.store [[VAR_30_]], [[RES_16_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK-DAG:       [[RES_17_:%.+]] = memref.alloc() {{.*}}: memref<1xi16>
// CHECK-DAG:       [[LOAD_RES_16_MEM_:%.+]] = krnl.load [[RES_16_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK-DAG:       [[LOAD_VAR_27_MEM_:%.+]] = krnl.load [[VAR_27_]][] : memref<i16>
// CHECK:           [[VAR_33_:%.+]] = arith.subi [[LOAD_RES_16_MEM_]], [[LOAD_VAR_27_MEM_]] : i16
// CHECK:           krnl.store [[VAR_33_]], [[RES_17_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK-DAG:       [[RES_18_:%.+]] = memref.alloc() {{.*}}: memref<1xi8>
// CHECK-DAG:       [[LOAD_RES_17_MEM_:%.+]] = krnl.load [[RES_17_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi16>
// CHECK:           [[VAR_35_:%.+]] = arith.trunci [[LOAD_RES_17_MEM_]] : i16 to i8
// CHECK:           krnl.store [[VAR_35_]], [[RES_18_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi8>
// CHECK-DAG:       [[RES_19_:%.+]] = memref.alloc() {{.*}}: memref<1xi32>
// CHECK-DAG:       [[LOAD_RES_18_MEM_:%.+]] = krnl.load [[RES_18_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi8>
// CHECK:           [[VAR_37_:%.+]] = arith.extsi [[LOAD_RES_18_MEM_]] : i8 to i32
// CHECK:           krnl.store [[VAR_37_]], [[RES_19_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:           [[RES_20_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi32>
// CHECK:           krnl.memset [[RES_20_]], [[CST_0_]] : memref<16x64xi32>
// CHECK:           [[LOOP_4_:%.+]]:3 = krnl.define_loops 3
// CHECK:           [[BLOCK_TILE__2_:%.+]], [[BLOCK_IN__2_:%.+]] = krnl.block [[LOOP_4_]]#0 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__3_:%.+]], [[BLOCK_IN__3_:%.+]] = krnl.block [[LOOP_4_]]#1 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__4_:%.+]], [[BLOCK_IN__4_:%.+]] = krnl.block [[LOOP_4_]]#2 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.permute([[BLOCK_TILE__2_]], [[BLOCK_IN__2_]], [[BLOCK_TILE__2_]]_23, [[BLOCK_IN__2_]]_24, [[BLOCK_TILE__2_]]_25, [[BLOCK_IN__2_]]_26) [0, 3, 1, 4, 2, 5] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
// CHECK:           krnl.iterate([[BLOCK_TILE__2_]], [[BLOCK_TILE__2_]]_23, [[BLOCK_TILE__2_]]_25) with ([[LOOP_4_]]#0 -> [[I_6_:%.+]] = [[CST_0_1_]] to [[CST_16_]], [[LOOP_4_]]#1 -> [[I_7_:%.+]] = [[CST_0_1_]] to [[CST_64_]], [[LOOP_4_]]#2 -> [[I_8_:%.+]] = [[CST_0_1_]] to [[CST_32_]]){
// CHECK:             [[VAR_48_2_:%.+]]:3 = krnl.get_induction_var_value([[BLOCK_TILE__2_]], [[BLOCK_TILE__2_]]_23, [[BLOCK_TILE__2_]]_25) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.matmul [[RES_5_]]{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, [[RES_13_]]{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, [[RES_20_]]{{.}}[[CST_0_1_]], [[CST_0_1_]]{{.}}, ([[BLOCK_IN__2_]], [[BLOCK_IN__2_]]_24, [[BLOCK_IN__2_]]_26), ([[VAR_48_2_]]#0, [[VAR_48_2_]]#1, [[VAR_48_2_]]#2), ([[CST_16_]], [[CST_64_]], [[CST_32_]]) {aTileSize = [], bTileSize = [], cTileSize = [], computeTileSize = [4, 8, 8]} : memref<16x32xi32>, memref<32x64xi32>, memref<16x64xi32>, (!krnl.loop, !krnl.loop, !krnl.loop)
// CHECK:           }
// CHECK-DAG:       [[RES_21_:%.+]] = memref.alloc() {{.*}}: memref<16x64xf32>
// CHECK-DAG:       [[LOOP_5_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_5_]]#0, [[LOOP_5_]]#1) with ([[LOOP_5_]]#0 -> [[I_9_:%.+]] = 0 to 16, [[LOOP_5_]]#1 -> [[I_10_:%.+]] = 0 to 64){
// CHECK:             [[VAR_48_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_5_]]#0, [[LOOP_5_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_1_:%.+]] = krnl.load [[RES_20_]]{{.}}[[VAR_48_3_]]#0, [[VAR_48_3_]]#1] : memref<16x64xi32>
// CHECK:             [[VAR_50_3_:%.+]] = arith.sitofp [[LOAD_PARAM_0_MEM_1_1_]] : i32 to f32
// CHECK:             krnl.store [[VAR_50_3_]], [[RES_21_]]{{.}}[[VAR_48_3_]]#0, [[VAR_48_3_]]#1] : memref<16x64xf32>
// CHECK:           }
// CHECK-DAG:       [[RES_22_:%.+]] = memref.alloc() {{.*}}: memref<1xf32>
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK:           [[VAR_42_:%.+]] = arith.mulf [[LOAD_PARAM_1_MEM_]], [[LOAD_PARAM_4_MEM_]] : f32
// CHECK:           krnl.store [[VAR_42_]], [[RES_22_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[RES_23_:%.+]] = memref.alloc() {{.*}}: memref<1xf32>
// CHECK-DAG:       [[LOAD_RES_22_MEM_:%.+]] = krnl.load [[RES_22_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_PARAM_6_MEM_:%.+]] = krnl.load [[PARAM_6_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK:           [[VAR_45_:%.+]] = arith.divf [[LOAD_RES_22_MEM_]], [[LOAD_PARAM_6_MEM_]] : f32
// CHECK:           krnl.store [[VAR_45_]], [[RES_23_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[RES_24_:%.+]] = memref.alloc() {{.*}}: memref<16x64xf32>
// CHECK-DAG:       [[RES_25_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_25_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_32_:%.+]] = memref.reshape [[RES_21_]]([[RES_25_]]) : (memref<16x64xf32>, memref<1xindex>) -> memref<1024xf32>
// CHECK-DAG:       [[RES_26_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_26_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_34_:%.+]] = memref.reshape [[RES_24_]]([[RES_26_]]) : (memref<16x64xf32>, memref<1xindex>) -> memref<1024xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_6_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__5_:%.+]], [[BLOCK_IN__5_:%.+]] = krnl.block [[LOOP_6_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__5_]]) with ([[LOOP_6_]] -> [[I_11_:%.+]] = 0 to 1024){
// CHECK:               [[LOAD_PARAM_0_MEM_1_1_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__5_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[VAR_50_3_:%.+]] = vector.load [[VAR_reshape_32_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]{{.}} : memref<1024xf32>, vector<32xf32>
// CHECK-DAG:           [[VAR_51_2_1_:%.+]] = krnl.load [[RES_23_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK:               [[LOAD_VAR_1_MEM_1_1_:%.+]] = vector.splat [[VAR_51_2_1_]] : vector<32xf32>
// CHECK:               [[VAR_53_4_:%.+]] = arith.mulf [[VAR_50_3_]], [[LOAD_VAR_1_MEM_1_1_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_53_4_]], [[VAR_reshape_34_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]{{.}} : memref<1024xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_27_:%.+]] = memref.alloc() {{.*}}: memref<16x64xf32>
// CHECK-DAG:       [[RES_28_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_28_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_37_:%.+]] = memref.reshape [[RES_24_]]([[RES_28_]]) : (memref<16x64xf32>, memref<1xindex>) -> memref<1024xf32>
// CHECK-DAG:       [[RES_29_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_29_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_39_:%.+]] = memref.reshape [[RES_27_]]([[RES_29_]]) : (memref<16x64xf32>, memref<1xindex>) -> memref<1024xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_7_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__6_:%.+]], [[BLOCK_IN__6_:%.+]] = krnl.block [[LOOP_7_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__6_]]) with ([[LOOP_7_]] -> [[I_12_:%.+]] = 0 to 1024){
// CHECK:               [[LOAD_PARAM_0_MEM_1_1_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__6_]]) : (!krnl.loop) -> index
// CHECK:               [[VAR_50_3_1_:%.+]] = vector.load [[VAR_reshape_37_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]{{.}} : memref<1024xf32>, vector<32xf32>
// CHECK:               [[VAR_51_3_:%.+]] = vector.shape_cast [[VAR_50_3_1_]] : vector<32xf32> to vector<8x4xf32>
// CHECK:               [[LOAD_VAR_1_MEM_1_1_1_:%.+]] = vector.extract [[VAR_51_3_]][0] : vector<4xf32> from vector<8x4xf32>
// CHECK:               [[VAR_53_5_:%.+]] = "krnl.round_even"([[LOAD_VAR_1_MEM_1_1_1_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:           [[VAR_54_2_:%.+]] = vector.insert [[VAR_53_5_]], [[VAR_51_3_]] [0] : vector<4xf32> into vector<8x4xf32>
// CHECK-DAG:           [[VAR_55_2_:%.+]] = vector.extract [[VAR_51_3_]][1] : vector<4xf32> from vector<8x4xf32>
// CHECK:               [[VAR_56_:%.+]] = "krnl.round_even"([[VAR_55_2_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:           [[VAR_57_:%.+]] = vector.insert [[VAR_56_]], [[VAR_54_2_]] [1] : vector<4xf32> into vector<8x4xf32>
// CHECK-DAG:           [[VAR_58_:%.+]] = vector.extract [[VAR_51_3_]][2] : vector<4xf32> from vector<8x4xf32>
// CHECK:               [[VAR_59_:%.+]] = "krnl.round_even"([[VAR_58_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:           [[VAR_60_:%.+]] = vector.insert [[VAR_59_]], [[VAR_57_]] [2] : vector<4xf32> into vector<8x4xf32>
// CHECK-DAG:           [[VAR_61_:%.+]] = vector.extract [[VAR_51_3_]][3] : vector<4xf32> from vector<8x4xf32>
// CHECK:               [[VAR_62_:%.+]] = "krnl.round_even"([[VAR_61_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:           [[VAR_63_:%.+]] = vector.insert [[VAR_62_]], [[VAR_60_]] [3] : vector<4xf32> into vector<8x4xf32>
// CHECK-DAG:           [[VAR_64_:%.+]] = vector.extract [[VAR_51_3_]][4] : vector<4xf32> from vector<8x4xf32>
// CHECK:               [[VAR_65_:%.+]] = "krnl.round_even"([[VAR_64_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:           [[VAR_66_:%.+]] = vector.insert [[VAR_65_]], [[VAR_63_]] [4] : vector<4xf32> into vector<8x4xf32>
// CHECK-DAG:           [[VAR_67_:%.+]] = vector.extract [[VAR_51_3_]][5] : vector<4xf32> from vector<8x4xf32>
// CHECK:               [[VAR_68_:%.+]] = "krnl.round_even"([[VAR_67_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:           [[VAR_69_:%.+]] = vector.insert [[VAR_68_]], [[VAR_66_]] [5] : vector<4xf32> into vector<8x4xf32>
// CHECK-DAG:           [[VAR_70_:%.+]] = vector.extract [[VAR_51_3_]][6] : vector<4xf32> from vector<8x4xf32>
// CHECK:               [[VAR_71_:%.+]] = "krnl.round_even"([[VAR_70_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:           [[VAR_72_:%.+]] = vector.insert [[VAR_71_]], [[VAR_69_]] [6] : vector<4xf32> into vector<8x4xf32>
// CHECK-DAG:           [[VAR_73_:%.+]] = vector.extract [[VAR_51_3_]][7] : vector<4xf32> from vector<8x4xf32>
// CHECK:               [[VAR_74_:%.+]] = "krnl.round_even"([[VAR_73_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK:               [[VAR_75_:%.+]] = vector.insert [[VAR_74_]], [[VAR_72_]] [7] : vector<4xf32> into vector<8x4xf32>
// CHECK:               [[VAR_76_:%.+]] = vector.shape_cast [[VAR_75_]] : vector<8x4xf32> to vector<32xf32>
// CHECK:               vector.store [[VAR_76_]], [[VAR_reshape_39_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]{{.}} : memref<1024xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_30_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi32>
// CHECK-DAG:       [[LOOP_8_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_8_]]#0, [[LOOP_8_]]#1) with ([[LOOP_8_]]#0 -> [[I_13_:%.+]] = 0 to 16, [[LOOP_8_]]#1 -> [[I_14_:%.+]] = 0 to 64){
// CHECK:             [[VAR_48_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_8_]]#0, [[LOOP_8_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_1_1_1_:%.+]] = krnl.load [[RES_27_]]{{.}}[[VAR_48_4_]]#0, [[VAR_48_4_]]#1] : memref<16x64xf32>
// CHECK:             [[VAR_50_4_:%.+]] = arith.fptosi [[LOAD_PARAM_0_MEM_1_1_1_1_]] : f32 to i32
// CHECK:             krnl.store [[VAR_50_4_]], [[RES_30_]]{{.}}[[VAR_48_4_]]#0, [[VAR_48_4_]]#1] : memref<16x64xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_31_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi32>
// CHECK-DAG:       [[RES_32_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_32_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_43_:%.+]] = memref.reshape [[RES_30_]]([[RES_32_]]) : (memref<16x64xi32>, memref<1xindex>) -> memref<1024xi32>
// CHECK-DAG:       [[RES_33_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_33_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_45_:%.+]] = memref.reshape [[RES_31_]]([[RES_33_]]) : (memref<16x64xi32>, memref<1xindex>) -> memref<1024xi32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_9_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__7_:%.+]], [[BLOCK_IN__7_:%.+]] = krnl.block [[LOOP_9_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__7_]]) with ([[LOOP_9_]] -> [[I_15_:%.+]] = 0 to 1024){
// CHECK:               [[LOAD_PARAM_0_MEM_1_1_1_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__7_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[VAR_50_4_:%.+]] = vector.load [[VAR_reshape_43_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_1_]]{{.}} : memref<1024xi32>, vector<32xi32>
// CHECK-DAG:           [[VAR_51_3_:%.+]] = krnl.load [[RES_19_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:               [[LOAD_VAR_1_MEM_1_1_1_:%.+]] = vector.splat [[VAR_51_3_]] : vector<32xi32>
// CHECK:               [[VAR_53_6_:%.+]] = arith.addi [[VAR_50_4_]], [[LOAD_VAR_1_MEM_1_1_1_]] : vector<32xi32>
// CHECK:               vector.store [[VAR_53_6_]], [[VAR_reshape_45_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_1_]]{{.}} : memref<1024xi32>, vector<32xi32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_34_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi32>
// CHECK-DAG:       [[RES_35_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_35_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_48_:%.+]] = memref.reshape [[RES_31_]]([[RES_35_]]) : (memref<16x64xi32>, memref<1xindex>) -> memref<1024xi32>
// CHECK-DAG:       [[RES_36_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_36_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_50_:%.+]] = memref.reshape [[RES_34_]]([[RES_36_]]) : (memref<16x64xi32>, memref<1xindex>) -> memref<1024xi32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_10_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__8_:%.+]], [[BLOCK_IN__8_:%.+]] = krnl.block [[LOOP_10_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__8_]]) with ([[LOOP_10_]] -> [[I_16_:%.+]] = 0 to 1024){
// CHECK:               [[LOAD_PARAM_0_MEM_1_1_1_1_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__8_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[VAR_50_4_1_:%.+]] = vector.load [[VAR_reshape_48_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_1_1_]]{{.}} : memref<1024xi32>, vector<32xi32>
// CHECK-DAG:           [[VAR_51_3_1_:%.+]] = krnl.load [[VAR_0_]][] : memref<i32>
// CHECK:               [[LOAD_VAR_1_MEM_1_1_1_1_:%.+]] = vector.splat [[VAR_51_3_1_]] : vector<32xi32>
// CHECK:               [[VAR_53_7_:%.+]] = arith.addi [[VAR_50_4_1_]], [[LOAD_VAR_1_MEM_1_1_1_1_]] : vector<32xi32>
// CHECK:               vector.store [[VAR_53_7_]], [[VAR_reshape_50_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_1_1_]]{{.}} : memref<1024xi32>, vector<32xi32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_37_:%.+]] = memref.alloc() {{.*}}: memref<16x64xui8>
// CHECK-DAG:       [[LOOP_11_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_11_]]#0, [[LOOP_11_]]#1) with ([[LOOP_11_]]#0 -> [[I_17_:%.+]] = 0 to 16, [[LOOP_11_]]#1 -> [[I_18_:%.+]] = 0 to 64){
// CHECK:             [[VAR_48_5_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_11_]]#0, [[LOOP_11_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_1_1_1_1_:%.+]] = krnl.load [[RES_34_]]{{.}}[[VAR_48_5_]]#0, [[VAR_48_5_]]#1] : memref<16x64xi32>
// CHECK:             [[VAR_50_5_:%.+]] = arith.trunci [[LOAD_PARAM_0_MEM_1_1_1_1_1_]] : i32 to i8
// CHECK:             [[VAR_51_4_:%.+]] = builtin.unrealized_conversion_cast [[VAR_50_5_]] : i8 to ui8
// CHECK:             krnl.store [[VAR_51_4_]], [[RES_37_]]{{.}}[[VAR_48_5_]]#0, [[VAR_48_5_]]#1] : memref<16x64xui8>
// CHECK:           }
// CHECK:           return [[RES_37_]] : memref<16x64xui8>
// CHECK:         }
}

