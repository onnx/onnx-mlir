// RUN: onnx-mlir-opt -O3 --march=x86-64 --shape-inference --convert-onnx-to-krnl=enable-parallel --canonicalize %s -split-input-file | FileCheck %s

// With enable-parallel, a krnl.parallel should be created, which takes a loop (to be parallelized)
// as input. The krnl.parallel should be the last operator before krnl.iterate, since the lowering
// needs to interpret krnl.block, krnl.permute, krnl.unroll first.
// -----

// Test parallelization of GEMM, disabled because too small

func.func @test_gemm_parallel(%arg0 : tensor<5x10xf32>, %arg1 : tensor<5x10xf32>, %arg2: tensor<10xf32>) -> tensor<*xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 5.0 : f32, transA = 1 : si64, transB = 0 : si64} : (tensor<5x10xf32>, tensor<5x10xf32>, tensor<10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_gemm_parallel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<5x10xf32>, [[PARAM_1_:%.+]]: memref<5x10xf32>, [[PARAM_2_:%.+]]: memref<10xf32>) -> memref<10x10xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_5_dot_000000_:%.+]] = arith.constant 5.000000e+00 : f32
// CHECK-DAG:       [[CST_10_:%.+]] = arith.constant 10 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<10x10xf32>
// CHECK:           krnl.memset [[RES_]], [[CST_0_dot_000000_]] : memref<10x10xf32>
// CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]]#0 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[BLOCK_IN__0_]] 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__2_:%.+]], [[BLOCK_IN__2_:%.+]] = krnl.block [[LOOP_0_]]#1 64 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__3_:%.+]], [[BLOCK_IN__3_:%.+]] = krnl.block [[BLOCK_IN__2_]] 16 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__4_:%.+]], [[BLOCK_IN__4_:%.+]] = krnl.block [[LOOP_0_]]#2 256 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.permute([[BLOCK_TILE__2_]], [[BLOCK_TILE__3_]], [[BLOCK_IN__3_]], [[BLOCK_TILE__4_]], [[BLOCK_IN__4_]], [[BLOCK_TILE__0_]], [[BLOCK_TILE__0_]]_1, [[BLOCK_IN__1_]]) [0, 3, 5, 1, 6, 2, 4, 7] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
// CHECK:           krnl.iterate([[BLOCK_TILE__2_]], [[BLOCK_TILE__4_]]) with ([[LOOP_0_]]#1 -> [[I_0_:%.+]] = 0 to 10, [[LOOP_0_]]#2 -> [[I_1_:%.+]] = 0 to 5, [[LOOP_0_]]#0 -> [[I_2_:%.+]] = 0 to 10){
// CHECK-DAG:         [[VAR_2_:%.+]]:2 = krnl.get_induction_var_value([[BLOCK_TILE__2_]], [[BLOCK_TILE__4_]]) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<32x256xf32>
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<256x64xf32>
// CHECK:             krnl.copy_to_tile_buffer [[RES_2_]], [[PARAM_1_]]{{.}}[[VAR_2_]]#1, [[VAR_2_]]#0], [[CST_0_dot_000000_]] {padToNext = [], tileSize = []} : memref<256x64xf32>, memref<5x10xf32>
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with (){
// CHECK:               [[VAR_3_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:               krnl.copy_to_tile_buffer [[RES_1_]], [[PARAM_0_]]{{.}}[[VAR_2_]]#1, [[VAR_3_]]{{.}}, [[CST_0_dot_000000_]] {padToNext = [], tileSize = [], transpose = true} : memref<32x256xf32>, memref<5x10xf32>
// CHECK:               krnl.iterate([[BLOCK_TILE__3_]], [[BLOCK_TILE__1_]]) with (){
// CHECK:                 [[VAR_4_:%.+]]:2 = krnl.get_induction_var_value([[BLOCK_TILE__3_]], [[BLOCK_TILE__1_]]) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:                 krnl.matmul [[RES_1_]]{{.}}[[VAR_3_]], [[VAR_2_]]#1], [[RES_2_]]{{.}}[[VAR_2_]]#1, [[VAR_2_]]#0], [[RES_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}}, ([[BLOCK_IN__1_]], [[BLOCK_IN__3_]], [[BLOCK_IN__4_]]), ([[VAR_4_]]#1, [[VAR_4_]]#0, [[VAR_2_]]#1), ([[CST_10_]], [[CST_10_]], [[CST_5_]]) {aTileSize = [], bTileSize = [], cTileSize = [], computeTileSize = [4, 16, 256], simdize = false} : memref<32x256xf32>, memref<256x64xf32>, memref<10x10xf32>, (!krnl.loop, !krnl.loop, !krnl.loop)
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to 10, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 10){
// CHECK:             [[VAR_2_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_3_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1] : memref<10x10xf32>
// CHECK-DAG:         [[VAR_4_1_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_2_1_]]#1] : memref<10xf32>
// CHECK:             [[VAR_5_:%.+]] = arith.mulf [[VAR_4_1_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_6_:%.+]] = arith.addf [[VAR_3_1_]], [[VAR_5_]] : f32
// CHECK:             krnl.store [[VAR_6_]], [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1] : memref<10x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<10x10xf32>
// CHECK:         }
}

// -----

// Test parallelization of GEMM, disabled because too small

func.func @test_gemm_parallel_success(%arg0 : tensor<1024x1024xf32>, %arg1 : tensor<1024x1024xf32>, %arg2: tensor<1024xf32>) -> tensor<*xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 5.0 : f32, transA = 1 : si64, transB = 0 : si64} : (tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_gemm_parallel_success
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1024x1024xf32>, [[PARAM_1_:%.+]]: memref<1024x1024xf32>, [[PARAM_2_:%.+]]: memref<1024xf32>) -> memref<1024x1024xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_5_dot_000000_:%.+]] = arith.constant 5.000000e+00 : f32
// CHECK-DAG:       [[CST_1024_:%.+]] = arith.constant 1024 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1024x1024xf32>
// CHECK:           krnl.memset [[RES_]], [[CST_0_dot_000000_]] : memref<1024x1024xf32>
// CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]]#0 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[BLOCK_IN__0_]] 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__2_:%.+]], [[BLOCK_IN__2_:%.+]] = krnl.block [[LOOP_0_]]#1 64 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__3_:%.+]], [[BLOCK_IN__3_:%.+]] = krnl.block [[BLOCK_IN__2_]] 16 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__4_:%.+]], [[BLOCK_IN__4_:%.+]] = krnl.block [[LOOP_0_]]#2 256 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.permute([[BLOCK_TILE__2_]], [[BLOCK_TILE__3_]], [[BLOCK_IN__3_]], [[BLOCK_TILE__4_]], [[BLOCK_IN__4_]], [[BLOCK_TILE__0_]], [[BLOCK_TILE__0_]]_1, [[BLOCK_IN__1_]]) [0, 3, 5, 1, 6, 2, 4, 7] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
// CHECK:           krnl.parallel([[BLOCK_TILE__2_]]) : !krnl.loop
// CHECK:           krnl.iterate([[BLOCK_TILE__2_]], [[BLOCK_TILE__4_]]) with ([[LOOP_0_]]#1 -> [[I_0_:%.+]] = 0 to 1024, [[LOOP_0_]]#2 -> [[I_1_:%.+]] = 0 to 1024, [[LOOP_0_]]#0 -> [[I_2_:%.+]] = 0 to 1024){
// CHECK-DAG:         [[VAR_2_:%.+]]:2 = krnl.get_induction_var_value([[BLOCK_TILE__2_]], [[BLOCK_TILE__4_]]) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<32x256xf32>
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<256x64xf32>
// CHECK:             krnl.copy_to_tile_buffer [[RES_2_]], [[PARAM_1_]]{{.}}[[VAR_2_]]#1, [[VAR_2_]]#0], [[CST_0_dot_000000_]] {padToNext = [], tileSize = []} : memref<256x64xf32>, memref<1024x1024xf32>
// CHECK:             krnl.iterate([[BLOCK_TILE__0_]]) with (){
// CHECK:               [[VAR_3_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:               krnl.copy_to_tile_buffer [[RES_1_]], [[PARAM_0_]]{{.}}[[VAR_2_]]#1, [[VAR_3_]]{{.}}, [[CST_0_dot_000000_]] {padToNext = [], tileSize = [], transpose = true} : memref<32x256xf32>, memref<1024x1024xf32>
// CHECK:               krnl.iterate([[BLOCK_TILE__3_]], [[BLOCK_TILE__1_]]) with (){
// CHECK:                 [[VAR_4_:%.+]]:2 = krnl.get_induction_var_value([[BLOCK_TILE__3_]], [[BLOCK_TILE__1_]]) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:                 krnl.matmul [[RES_1_]]{{.}}[[VAR_3_]], [[VAR_2_]]#1], [[RES_2_]]{{.}}[[VAR_2_]]#1, [[VAR_2_]]#0], [[RES_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}}, ([[BLOCK_IN__1_]], [[BLOCK_IN__3_]], [[BLOCK_IN__4_]]), ([[VAR_4_]]#1, [[VAR_4_]]#0, [[VAR_2_]]#1), ([[CST_1024_]], [[CST_1024_]], [[CST_1024_]]) {aTileSize = [], bTileSize = [], cTileSize = [], computeTileSize = [4, 16, 256]} : memref<32x256xf32>, memref<256x64xf32>, memref<1024x1024xf32>, (!krnl.loop, !krnl.loop, !krnl.loop)
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.parallel([[LOOP_1_]]#0) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to 1024, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 1024){
// CHECK:             [[VAR_2_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_3_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1] : memref<1024x1024xf32>
// CHECK-DAG:         [[VAR_4_1_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_2_1_]]#1] : memref<1024xf32>
// CHECK:             [[VAR_5_:%.+]] = arith.mulf [[VAR_4_1_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_6_:%.+]] = arith.addf [[VAR_3_1_]], [[VAR_5_]] : f32
// CHECK:             krnl.store [[VAR_6_]], [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1] : memref<1024x1024xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1024x1024xf32>
// CHECK:         }
}

