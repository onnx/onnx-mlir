// RUN: onnx-mlir-opt -O3 --march=x86-64 --shape-inference --convert-onnx-to-krnl=enable-parallel --canonicalize %s -split-input-file | FileCheck %s

// -----

// With enable-parallel, a krnl.parallel should be created, which takes a loop (to be parallelized)
// as input. The krnl.parallel should be the last operator before krnl.iterate, since the lowering
// needs to interpret krnl.block, krnl.permute, krnl.unroll first.

// Test parallelization of Matmul too small for parallel
func.func @test_matmul_parallel_too_small(%arg0 : tensor<4x8xf32>, %arg1 : tensor<8x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // mlir2FileCheck.py
  // CHECK-LABEL:  func.func @test_matmul_parallel_too_small
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
  // CHECK:           krnl.iterate([[BLOCK_TILE__0_]], [[BLOCK_TILE__0_]]_0, [[BLOCK_TILE__0_]]_2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = [[CST_0_]] to [[CST_4_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = [[CST_0_]] to [[CST_16_]], [[LOOP_0_]]#2 -> [[I_2_:%.+]] = [[CST_0_]] to [[CST_8_]]){
  // CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[BLOCK_TILE__0_]], [[BLOCK_TILE__0_]]_0, [[BLOCK_TILE__0_]]_2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
  // CHECK:             krnl.matmul [[PARAM_0_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}}, [[PARAM_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}}, [[RES_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}}, ([[BLOCK_IN__0_]], [[BLOCK_IN__0_]]_1, [[BLOCK_IN__0_]]_3), ([[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2), ([[CST_4_]], [[CST_16_]], [[CST_8_]]) {aTileSize = [], bTileSize = [], cTileSize = [], computeTileSize = [4, 8, 8]} : memref<4x8xf32>, memref<8x16xf32>, memref<4x16xf32>, (!krnl.loop, !krnl.loop, !krnl.loop)
  // CHECK:           }
  // CHECK:           return [[RES_]] : memref<4x16xf32>
  // CHECK:         }
}

// -----

// Test parallelization of Matmul too small for parallel

func.func @test_matmul_parallel_successful(%arg0 : tensor<1024x512xf32>, %arg1 : tensor<512x2048xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1024x512xf32>, tensor<512x2048xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_matmul_parallel_successful
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1024x512xf32>, [[PARAM_1_:%.+]]: memref<512x2048xf32>) -> memref<1024x2048xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_1024_:%.+]] = arith.constant 1024 : index
// CHECK-DAG:       [[CST_512_:%.+]] = arith.constant 512 : index
// CHECK-DAG:       [[CST_2048_:%.+]] = arith.constant 2048 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1024x2048xf32>
// CHECK:           krnl.memset [[RES_]], [[CST_0_dot_000000_]] : memref<1024x2048xf32>
// CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]]#0 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_0_]]#1 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           [[BLOCK_TILE__2_:%.+]], [[BLOCK_IN__2_:%.+]] = krnl.block [[LOOP_0_]]#2 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.permute([[BLOCK_TILE__0_]], [[BLOCK_IN__0_]], [[BLOCK_TILE__0_]]_0, [[BLOCK_IN__0_]]_1, [[BLOCK_TILE__0_]]_2, [[BLOCK_IN__0_]]_3) [0, 3, 1, 4, 2, 5] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
// CHECK:           krnl.parallel([[BLOCK_TILE__0_]]) : !krnl.loop
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]], [[BLOCK_TILE__0_]]_0, [[BLOCK_TILE__0_]]_2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = [[CST_0_]] to [[CST_1024_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = [[CST_0_]] to [[CST_2048_]], [[LOOP_0_]]#2 -> [[I_2_:%.+]] = [[CST_0_]] to [[CST_512_]]){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[BLOCK_TILE__0_]], [[BLOCK_TILE__0_]]_0, [[BLOCK_TILE__0_]]_2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.matmul [[PARAM_0_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}}, [[PARAM_1_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}}, [[RES_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}}, ([[BLOCK_IN__0_]], [[BLOCK_IN__0_]]_1, [[BLOCK_IN__0_]]_3), ([[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2), ([[CST_1024_]], [[CST_2048_]], [[CST_512_]]) {aTileSize = [], bTileSize = [], cTileSize = [], computeTileSize = [4, 8, 8]} : memref<1024x512xf32>, memref<512x2048xf32>, memref<1024x2048xf32>, (!krnl.loop, !krnl.loop, !krnl.loop)
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1024x2048xf32>
// CHECK:         }
}

