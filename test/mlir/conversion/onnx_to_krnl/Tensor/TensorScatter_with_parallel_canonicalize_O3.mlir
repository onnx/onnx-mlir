// RUN: onnx-mlir-opt -O3 --march=z16 --convert-onnx-to-krnl=enable-parallel --canonicalize %s -split-input-file | FileCheck %s

// Parallel coverage for TensorScatter, whose selection window is [0, 2) like
// most sites: the batch dim (0) and the dim right after it.

// Static shapes: batch (trip count 2) is below the parallel threshold, so the
// search moves to the next dim (trip count 4).
func.func @test_parallel_tensor_scatter(%arg0: tensor<2x4x8x16xf32>, %arg1: tensor<2x4x3x16xf32>, %arg2: tensor<2xi64>) -> tensor<2x4x8x16xf32> {
  %0 = "onnx.TensorScatter"(%arg0, %arg1, %arg2) {axis = 2 : si64, mode = "linear"} : (tensor<2x4x8x16xf32>, tensor<2x4x3x16xf32>, tensor<2xi64>) -> tensor<2x4x8x16xf32>
  return %0 : tensor<2x4x8x16xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_parallel_tensor_scatter
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4x8x16xf32>, [[PARAM_1_:%.+]]: memref<2x4x3x16xf32>, [[PARAM_2_:%.+]]: memref<2xi64>) -> memref<2x4x8x16xf32> {
// CHECK:           [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.parallel([[LOOP_0_]]#1) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 3, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 16){
// CHECK:             [[VAR_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0] : memref<2xi64>
// CHECK:             [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PARAM_2_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.addi [[VAR_3_]], [[VAR_1_]]#2 : index
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]#3] : memref<2x4x3x16xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]], [[VAR_1_]]#3] : memref<2x4x8x16xf32>
// CHECK:           }
// CHECK:           return [[PARAM_0_]] : memref<2x4x8x16xf32>
// CHECK:         }
}

// -----

// A dynamic batch dim is assumed wide enough by definition, so the search
// stops at level 0, even with 'mode' = "circular".
func.func @test_parallel_tensor_scatter_dyn_batch(%arg0: tensor<?x4x8x16xf32>, %arg1: tensor<?x4x3x16xf32>, %arg2: tensor<?xi64>) -> tensor<?x4x8x16xf32> {
  %0 = "onnx.TensorScatter"(%arg0, %arg1, %arg2) {axis = 2 : si64, mode = "circular"} : (tensor<?x4x8x16xf32>, tensor<?x4x3x16xf32>, tensor<?xi64>) -> tensor<?x4x8x16xf32>
  return %0 : tensor<?x4x8x16xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @test_parallel_tensor_scatter_dyn_batch
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x4x8x16xf32>, [[PARAM_1_:%.+]]: memref<?x4x3x16xf32>, [[PARAM_2_:%.+]]: memref<?xi64>) -> memref<?x4x8x16xf32> {
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x4x3x16xf32>
// CHECK:           krnl.parallel([[LOOP_0_]]#0) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 3, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 16){
// CHECK:             [[VAR_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0] : memref<?xi64>
// CHECK:             [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PARAM_2_MEM_]] : i64 to index
// CHECK:             [[VAR_4_:%.+]] = arith.addi [[VAR_3_]], [[VAR_1_]]#2 : index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.remsi [[VAR_4_]], [[CST_8_]] : index
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]#3] : memref<?x4x3x16xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_5_]], [[VAR_1_]]#3] : memref<?x4x8x16xf32>
// CHECK:           }
// CHECK:           return [[PARAM_0_]] : memref<?x4x8x16xf32>
// CHECK:         }
}
