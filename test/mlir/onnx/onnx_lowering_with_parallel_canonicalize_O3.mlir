// RUN: onnx-mlir-opt -O3 --march=x86-64 --shape-inference --convert-onnx-to-krnl=enable-parallel --canonicalize %s -split-input-file | FileCheck %s

// With enable-parallel, a krnl.parallel should be created, which takes a loop (to be parallelized) 
// as input. The krnl.parallel should be the last operator before krnl.iterate, since the lowering
// needs to interpret krnl.block, krnl.permute, krnl.unroll first.

// -----

// Test parallelization of GEMM
func.func @test_gemm_parallel(%arg0 : tensor<5x10xf32>, %arg1 : tensor<5x10xf32>, %arg2: tensor<10xf32>) -> tensor<*xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 5.0 : f32, transA = 1 : si64, transB = 0 : si64} : (tensor<5x10xf32>, tensor<5x10xf32>, tensor<10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // mlir2FileCheck.py
  // CHECK-LABEL:  func.func @test_gemm
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<5x10xf32>, [[PARAM_1_:%.+]]: memref<5x10xf32>, [[PARAM_2_:%.+]]: memref<10xf32>) -> memref<10x10xf32> {
  // CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
  // CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG:       [[CST_5_dot_000000_:%.+]] = arith.constant 5.000000e+00 : f32
  // CHECK-DAG:       [[CST_10_:%.+]] = arith.constant 10 : index
  // CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
  // CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<10x10xf32>
  // CHECK:           krnl.memset [[RES_]], [[CST_0_dot_000000_]] : memref<10x10xf32>
  // CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<32x256xf32>
  // CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<256x64xf32>
  // CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
  // CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]]#0 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  // CHECK:           [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[BLOCK_IN__0_]] 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  // CHECK:           [[BLOCK_TILE__2_:%.+]], [[BLOCK_IN__2_:%.+]] = krnl.block [[LOOP_0_]]#1 64 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  // CHECK:           [[BLOCK_TILE__3_:%.+]], [[BLOCK_IN__3_:%.+]] = krnl.block [[BLOCK_IN__2_]] 16 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  // CHECK:           [[BLOCK_TILE__4_:%.+]], [[BLOCK_IN__4_:%.+]] = krnl.block [[LOOP_0_]]#2 256 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  // CHECK:           krnl.permute([[BLOCK_TILE__2_]], [[BLOCK_TILE__3_]], [[BLOCK_IN__3_]], [[BLOCK_TILE__4_]], [[BLOCK_IN__4_]], [[BLOCK_TILE__0_]], [[BLOCK_TILE__0_]]_3, [[BLOCK_IN__1_]]) [0, 3, 5, 1, 6, 2, 4, 7] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
  // CHECK:           krnl.parallel [[BLOCK_TILE__2_]] : !krnl.loop
  // CHECK:           krnl.iterate([[BLOCK_TILE__2_]], [[BLOCK_TILE__4_]]) with ([[LOOP_0_]]#1 -> [[I_0_:%.+]] = 0 to 10, [[LOOP_0_]]#2 -> [[I_1_:%.+]] = 0 to 5, [[LOOP_0_]]#0 -> [[I_2_:%.+]] = 0 to 10){
  // CHECK:             [[VAR_2_:%.+]]:2 = krnl.get_induction_var_value([[BLOCK_TILE__2_]], [[BLOCK_TILE__4_]]) : (!krnl.loop, !krnl.loop) -> (index, index)
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
  // CHECK:           krnl.parallel [[LOOP_1_]]#0 : !krnl.loop
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

// Test parallelization of Matmul
func.func @test_matmul_parallel(%arg0 : tensor<4x8xf32>, %arg1 : tensor<8x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // mlir2FileCheck.py
  // CHECK-LABEL:  func.func @test_matmul_parallel
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4x8xf32>, [[PARAM_1_:%.+]]: memref<8x16xf32>) -> memref<4x16xf32> {
  // CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
  // CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
  // CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
  // CHECK-DAG:       [[CST_16_:%.+]] = arith.constant 16 : index
  // CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4x16xf32>
  // CHECK:           krnl.memset [[RES_]], [[CST_0_dot_000000_]] : memref<4x16xf32>
  // CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
  // CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]]#0 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  // CHECK:           [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_0_]]#1 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  // CHECK:           [[BLOCK_TILE__2_:%.+]], [[BLOCK_IN__2_:%.+]] = krnl.block [[LOOP_0_]]#2 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  // CHECK:           krnl.permute([[BLOCK_TILE__0_]], [[BLOCK_IN__0_]], [[BLOCK_TILE__0_]]_0, [[BLOCK_IN__0_]]_1, [[BLOCK_TILE__0_]]_2, [[BLOCK_IN__0_]]_3) [0, 3, 1, 4, 2, 5] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
  // CHECK:           krnl.parallel [[BLOCK_TILE__0_]] : !krnl.loop
  // CHECK:           krnl.iterate([[BLOCK_TILE__0_]], [[BLOCK_TILE__0_]]_0, [[BLOCK_TILE__0_]]_2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = [[CST_0_]] to [[CST_4_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = [[CST_0_]] to [[CST_16_]], [[LOOP_0_]]#2 -> [[I_2_:%.+]] = [[CST_0_]] to [[CST_8_]]){
  // CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[BLOCK_TILE__0_]], [[BLOCK_TILE__0_]]_0, [[BLOCK_TILE__0_]]_2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
  // CHECK:             krnl.matmul [[PARAM_0_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}}, [[PARAM_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}}, [[RES_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}}, ([[BLOCK_IN__0_]], [[BLOCK_IN__0_]]_1, [[BLOCK_IN__0_]]_3), ([[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2), ([[CST_4_]], [[CST_16_]], [[CST_8_]]) {aTileSize = [], bTileSize = [], cTileSize = [], computeTileSize = [4, 8, 8]} : memref<4x8xf32>, memref<8x16xf32>, memref<4x16xf32>, (!krnl.loop, !krnl.loop, !krnl.loop)
  // CHECK:           }
  // CHECK:           return [[RES_]] : memref<4x16xf32>
  // CHECK:         }
}
// -----

// Test parallelization of Relu
func.func @test_relu_parallel(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Relu"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // mlir2FileCheck.py
  // CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 40 + 128)>
  // CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 * 10)>
  // CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1] -> (s0 * 10)>
  // CHECK-LABEL:  func.func @test_relu_parallel
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
  // CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<0.000000e+00> : vector<32xf32>
  // CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
  // CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
  // CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
  // CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?xi8>
  // CHECK-DAG:       [[VAR_view_:%.+]] = memref.view [[RES_]]{{.}}[[CST_0_]]{{.}}{{.}}[[VAR_dim_]]{{.}} : memref<?xi8> to memref<?x10xf32>
  // CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
  // CHECK-NOT: separator of consecutive DAGs
  // CHECK-DAG:       [[VAR_1_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_0_]]{{.}}
  // CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
  // CHECK:           affine.store [[VAR_1_]], [[RES_1_]][0] : memref<1xindex>
  // CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
  // CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_]]{{.}}
  // CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
  // CHECK:           affine.store [[VAR_2_]], [[RES_2_]][0] : memref<1xindex>
  // CHECK-DAG:       [[VAR_reshape_3_:%.+]] = memref.reshape [[VAR_view_]]([[RES_2_]]) : (memref<?x10xf32>, memref<1xindex>) -> memref<?xf32>
  // CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
  // CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  // CHECK:           krnl.parallel [[BLOCK_TILE__0_]] : !krnl.loop
  // CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_2_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_0]){
  // CHECK:             [[VAR_4_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
  // CHECK:             [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
  // CHECK:             [[VAR_6_:%.+]] = arith.cmpf oge, [[LOAD_VAR_reshape_MEM_]], [[VAR_cst_]] : vector<32xf32>
  // CHECK:             [[VAR_7_:%.+]] = arith.select [[VAR_6_]], [[LOAD_VAR_reshape_MEM_]], [[VAR_cst_]] : vector<32xi1>, vector<32xf32>
  // CHECK:             vector.store [[VAR_7_]], [[VAR_reshape_3_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>, vector<32xf32>
  // CHECK:           }
  // CHECK:           return [[VAR_view_]] : memref<?x10xf32>
  // CHECK:         }
}

// -----

// Test parallelization of Transpose
func.func @test_transpose_block_1_last_dim_parallel(%arg0: tensor<?x256x12x64xf32>) -> tensor<?x12x256x64xf32> {
    %1 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1, 3] } : (tensor<?x256x12x64xf32>) -> tensor<?x12x256x64xf32>
    return %1 : tensor<?x12x256x64xf32>

    // mlir2FileCheck.py
    // CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
    // CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2) -> (d0 * 196608 + d1 * 768 + d2 * 64)>
    // CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1, d2) -> (d0 * 196608 + d1 * 64 + d2 * 16384)>
    // CHECK-LABEL:  func.func @test_transpose_block_1_last_dim
    // CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x256x12x64xf32>) -> memref<?x12x256x64xf32> {
    // CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : i64
    // CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
    // CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x256x12x64xf32>
    // CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x12x256x64xf32>
    // CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x256x12x64xf32>
    // CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
    // CHECK:           krnl.parallel [[LOOP_0_]]#0 : !krnl.loop
    // CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 256, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 12){
    // CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
    // CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2)
    // CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2)
    // CHECK:             "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[CST_64_]], [[VAR_3_]], [[VAR_2_]]) : (memref<?x12x256x64xf32>, memref<?x256x12x64xf32>, i64, index, index) -> ()
    // CHECK:           }
    // CHECK:           return [[RES_]] : memref<?x12x256x64xf32>
    // CHECK:         }
}

// -----

// Test parallelization of Softmax
func.func @test_softmax_v13_parallel(%arg0 : tensor<10x20x30xf32>) -> tensor<*xf32> {
  %0 = "onnx.Softmax"(%arg0) {axis=1: si64} : (tensor<10x20x30xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  // mlir2FileCheck.py
  // CHECK-LABEL:  func.func @test_softmax_v13_parallel
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10x20x30xf32>) -> memref<10x20x30xf32> {
  // CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0xFF800000 : f32
  // CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<10x20x30xf32>
  // CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() : memref<f32>
  // CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() : memref<f32>
  // CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
  // CHECK:           krnl.parallel [[LOOP_0_]]#0 : !krnl.loop
  // CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 10, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 30){
  // CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
  // CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_1_]][] : memref<f32>
  // CHECK:             krnl.store [[CST_0_]], [[RES_2_]][] : memref<f32>
  // CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
  // CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 20){
  // CHECK-DAG:           [[VAR_7_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
  // CHECK-DAG:           [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]][] : memref<f32>
  // CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_7_]], [[VAR_1_]]#1] : memref<10x20x30xf32>
  // CHECK:               [[VAR_10_:%.+]] = arith.cmpf ogt, [[LOAD_RES_2_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
  // CHECK:               [[VAR_11_:%.+]] = arith.select [[VAR_10_]], [[LOAD_RES_2_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
  // CHECK:               krnl.store [[VAR_11_]], [[RES_2_]][] : memref<f32>
  // CHECK:             }
  // CHECK-DAG:         [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]][] : memref<f32>
  // CHECK-DAG:         [[LOOP_2_:%.+]] = krnl.define_loops 1
  // CHECK:             krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_3_:%.+]] = 0 to 20){
  // CHECK-DAG:           [[VAR_7_1_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
  // CHECK-DAG:           [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
  // CHECK:               [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_7_1_]], [[VAR_1_]]#1] : memref<10x20x30xf32>
  // CHECK:               [[VAR_10_1_:%.+]] = arith.subf [[LOAD_PARAM_0_MEM_1_]], [[LOAD_RES_2_MEM_1_]] : f32
  // CHECK:               [[VAR_11_1_:%.+]] = math.exp [[VAR_10_1_]] : f32
  // CHECK:               [[VAR_12_:%.+]] = arith.addf [[LOAD_RES_2_MEM_2_]], [[VAR_11_1_]] : f32
  // CHECK:               krnl.store [[VAR_12_]], [[RES_1_]][] : memref<f32>
  // CHECK:               krnl.store [[VAR_11_1_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_7_1_]], [[VAR_1_]]#1] : memref<10x20x30xf32>
  // CHECK:             }
  // CHECK-DAG:         [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
  // CHECK-DAG:         [[LOOP_3_:%.+]] = krnl.define_loops 1
  // CHECK:             krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[I_4_:%.+]] = 0 to 20){
  // CHECK:               [[VAR_7_2_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
  // CHECK:               [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_7_2_]], [[VAR_1_]]#1] : memref<10x20x30xf32>
  // CHECK:               [[LOAD_PARAM_0_MEM_1_:%.+]] = arith.divf [[LOAD_RES_2_MEM_2_]], [[LOAD_RES_1_MEM_]] : f32
  // CHECK:               krnl.store [[LOAD_PARAM_0_MEM_1_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_7_2_]], [[VAR_1_]]#1] : memref<10x20x30xf32>
  // CHECK:             }
  // CHECK:           }
  // CHECK:           return [[RES_]] : memref<10x20x30xf32>
  // CHECK:         }
}

// -----

// Test parallelization of Concat
func.func @test_concat_parallel(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<5x5x3x32xf32>, %arg2 : tensor<5x5x5x32xf32>) -> tensor<5x5x9x32xf32> {
  %1 = "onnx.Concat"(%arg0, %arg1, %arg2) { axis = 2 : si64} : (tensor<5x5x1x32xf32>, tensor<5x5x3x32xf32>, tensor<5x5x5x32xf32>)  -> tensor<5x5x9x32xf32>
  "func.return"(%1) : (tensor<5x5x9x32xf32>) -> ()
  // mlir2FileCheck.py
  // CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 1)>
  // CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 4)>
  // CHECK-LABEL:  func.func @test_concat_parallel
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<5x5x1x32xf32>, [[PARAM_1_:%.+]]: memref<5x5x3x32xf32>, [[PARAM_2_:%.+]]: memref<5x5x5x32xf32>) -> memref<5x5x9x32xf32> {
  // CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<5x5x9x32xf32>
  // CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
  // CHECK:           krnl.parallel [[LOOP_0_]]#0 : !krnl.loop
  // CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 5, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 5, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 1, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 32){
  // CHECK:             [[VAR_3_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
  // CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_3_]]#2, [[VAR_3_]]#3] : memref<5x5x1x32xf32>
  // CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_3_]]#2, [[VAR_3_]]#3] : memref<5x5x9x32xf32>
  // CHECK:           }
  // CHECK:           [[LOOP_1_:%.+]]:4 = krnl.define_loops 4
  // CHECK:           krnl.parallel [[LOOP_1_]]#0 : !krnl.loop
  // CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2, [[LOOP_1_]]#3) with ([[LOOP_1_]]#0 -> [[I_4_:%.+]] = 0 to 5, [[LOOP_1_]]#1 -> [[I_5_:%.+]] = 0 to 5, [[LOOP_1_]]#2 -> [[I_6_:%.+]] = 0 to 3, [[LOOP_1_]]#3 -> [[I_7_:%.+]] = 0 to 32){
  // CHECK:             [[VAR_3_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2, [[LOOP_1_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
  // CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = affine.apply [[MAP_0_]]([[VAR_3_1_]]#2)
  // CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_3_1_]]#0, [[VAR_3_1_]]#1, [[VAR_3_1_]]#2, [[VAR_3_1_]]#3] : memref<5x5x3x32xf32>
  // CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_]]{{.}}[[VAR_3_1_]]#0, [[VAR_3_1_]]#1, [[LOAD_PARAM_0_MEM_1_]], [[VAR_3_1_]]#3] : memref<5x5x9x32xf32>
  // CHECK:           }
  // CHECK:           [[LOOP_2_:%.+]]:4 = krnl.define_loops 4
  // CHECK:           krnl.parallel [[LOOP_2_]]#0 : !krnl.loop
  // CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2, [[LOOP_2_]]#3) with ([[LOOP_2_]]#0 -> [[I_8_:%.+]] = 0 to 5, [[LOOP_2_]]#1 -> [[I_9_:%.+]] = 0 to 5, [[LOOP_2_]]#2 -> [[I_10_:%.+]] = 0 to 5, [[LOOP_2_]]#3 -> [[I_11_:%.+]] = 0 to 32){
  // CHECK:             [[VAR_3_2_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2, [[LOOP_2_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
  // CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = affine.apply [[MAP_1_]]([[VAR_3_2_]]#2)
  // CHECK-DAG:         [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_3_2_]]#0, [[VAR_3_2_]]#1, [[VAR_3_2_]]#2, [[VAR_3_2_]]#3] : memref<5x5x5x32xf32>
  // CHECK:             krnl.store [[LOAD_PARAM_1_MEM_1_]], [[RES_]]{{.}}[[VAR_3_2_]]#0, [[VAR_3_2_]]#1, [[LOAD_PARAM_0_MEM_1_]], [[VAR_3_2_]]#3] : memref<5x5x9x32xf32>
  // CHECK:           }
  // CHECK:           return [[RES_]] : memref<5x5x9x32xf32>
  // CHECK:         }
}

// -----

// Test parallelization of ReduceMean
func.func private @test_reducemean_v13_f32(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceMeanV13"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  // mlir2FileCheck.py
  // CHECK-LABEL:  func.func private @test_reducemean_v13_f32
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2x2xf32>) -> memref<3x2xf32> {
  // CHECK-DAG:       [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
  // CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x2xf32>
  // CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
  // CHECK:           krnl.parallel [[LOOP_0_]]#0 : !krnl.loop
  // CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
  // CHECK:             [[VAR_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
  // CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1] : memref<3x2xf32>
  // CHECK:           }
  // CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
  // CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_1_]]#2 -> [[I_4_:%.+]] = 0 to 2){
  // CHECK:             [[VAR_3_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
  // CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_1_]]#0, [[VAR_3_1_]]#1, [[VAR_3_1_]]#2] : memref<3x2x2xf32>
  // CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_3_1_]]#0, [[VAR_3_1_]]#2] : memref<3x2xf32>
  // CHECK:             [[VAR_6_:%.+]] = arith.addf [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
  // CHECK:             krnl.store [[VAR_6_]], [[RES_]]{{.}}[[VAR_3_1_]]#0, [[VAR_3_1_]]#2] : memref<3x2xf32>
  // CHECK:           }
  // CHECK:           [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
  // CHECK:           krnl.parallel [[LOOP_2_]]#0 : !krnl.loop
  // CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_5_:%.+]] = 0 to 3, [[LOOP_2_]]#1 -> [[I_6_:%.+]] = 0 to 2){
  // CHECK:             [[VAR_3_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
  // CHECK:             [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_3_2_]]#0, [[VAR_3_2_]]#1] : memref<3x2xf32>
  // CHECK:             [[LOAD_RES_MEM_2_:%.+]] = arith.divf [[LOAD_RES_MEM_1_]], [[CST_2_dot_000000_]] : f32
  // CHECK:             krnl.store [[LOAD_RES_MEM_2_]], [[RES_]]{{.}}[[VAR_3_2_]]#0, [[VAR_3_2_]]#1] : memref<3x2xf32>
  // CHECK:           }
  // CHECK:           return [[RES_]] : memref<3x2xf32>
  // CHECK:         }
}