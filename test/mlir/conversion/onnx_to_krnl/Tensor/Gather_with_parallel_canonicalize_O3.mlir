// RUN: onnx-mlir-opt -O3 --march=z16 --convert-onnx-to-krnl=enable-parallel --canonicalize %s -split-input-file | FileCheck %s

// Parallel coverage for Gather, whose selection window is [0, outputRank) rather
// than the [0, 2) most sites use. These cases pin down which level the search
// lands on, including one where it looks past two trip-count-1 levels to find
// enough work -- a depth only this wider window can reach.

// A dynamic leading dimension is assumed wide enough by definition, so the
// search stops at level 0.
func.func @test_parallel_gather_dyn_outer(%arg0: tensor<?x64xf32>, %arg1: tensor<?x8xi64>) -> tensor<?x8x64xf32> {
  %0 = "onnx.Gather"(%arg0, %arg1) {axis = 0 : si64} : (tensor<?x64xf32>, tensor<?x8xi64>) -> tensor<?x8x64xf32>
  return %0 : tensor<?x8x64xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @test_parallel_gather_dyn_outer
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x64xf32>, [[PARAM_1_:%.+]]: memref<?x8xi64>) -> memref<?x8x64xf32> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x8xi64>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x8x64xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x64xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.parallel([[LOOP_0_]]#0) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 8, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 64){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x8xi64>
// CHECK:             [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpi slt, [[VAR_3_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.addi [[VAR_3_]], [[VAR_dim_0_]] : index
// CHECK:             [[VAR_6_:%.+]] = arith.select [[VAR_4_]], [[VAR_5_]], [[VAR_3_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[VAR_1_]]#2] : memref<?x64xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<?x8x64xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x8x64xf32>
// CHECK:         }

}

// -----

// Levels 0 and 1 have a trip count of 1, below the 4-iteration floor, so the
// search continues to level 2.
func.func @test_parallel_gather_narrow_prefix(%arg0: tensor<?x64xf32>, %arg1: tensor<1x1x112xi64>) -> tensor<1x1x112x64xf32> {
  %0 = "onnx.Gather"(%arg0, %arg1) {axis = 0 : si64} : (tensor<?x64xf32>, tensor<1x1x112xi64>) -> tensor<1x1x112x64xf32>
  return %0 : tensor<1x1x112x64xf32>

// CHECK-LABEL:  func.func @test_parallel_gather_narrow_prefix
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x64xf32>, [[PARAM_1_:%.+]]: memref<1x1x112xi64>) -> memref<1x1x112x64xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x1x112x64xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x64xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.parallel([[LOOP_0_]]#2) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 112, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 64){
// CHECK:             [[VAR_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<1x1x112xi64>
// CHECK:             [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpi slt, [[VAR_3_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.addi [[VAR_3_]], [[VAR_dim_]] : index
// CHECK:             [[VAR_6_:%.+]] = arith.select [[VAR_4_]], [[VAR_5_]], [[VAR_3_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[VAR_1_]]#3] : memref<?x64xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]#3] : memref<1x1x112x64xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x1x112x64xf32>
// CHECK:         }

}

// -----

// Every level is too narrow, so no parallel region is created at all.
func.func @test_parallel_gather_all_narrow(%arg0: tensor<?x2xf32>, %arg1: tensor<1x2xi64>) -> tensor<1x2x2xf32> {
  %0 = "onnx.Gather"(%arg0, %arg1) {axis = 0 : si64} : (tensor<?x2xf32>, tensor<1x2xi64>) -> tensor<1x2x2xf32>
  return %0 : tensor<1x2x2xf32>
// CHECK-LABEL:  func.func @test_parallel_gather_all_narrow
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x2xf32>, [[PARAM_1_:%.+]]: memref<1x2xi64>) -> memref<1x2x2xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2x2xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<1x2xi64>
// CHECK:             [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpi slt, [[VAR_3_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.addi [[VAR_3_]], [[VAR_dim_]] : index
// CHECK:             [[VAR_6_:%.+]] = arith.select [[VAR_4_]], [[VAR_5_]], [[VAR_3_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[VAR_1_]]#2] : memref<?x2xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<1x2x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x2x2xf32>
// CHECK:         }

}

// -----

// A rank-0 output leaves no level to parallelize, so the window is empty and no
// region is created. Gather asks for [0, outputRank), which is [0, 0) here, and
// the plan holds no loop refs at all -- the case that must answer "no
// parallelism" rather than fail on an empty optimized-loop list.
func.func @test_parallel_gather_rank0_out(%arg0: tensor<4xf32>) -> tensor<f32> {
  %i = onnx.Constant dense<2> : tensor<i64>
  %0 = "onnx.Gather"(%arg0, %i) {axis = 0 : si64} : (tensor<4xf32>, tensor<i64>) -> tensor<f32>
  return %0 : tensor<f32>

// CHECK-LABEL:  func.func @test_parallel_gather_rank0_out
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4xf32>) -> memref<f32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [], value = dense<2> : tensor<i64>}> : () -> memref<i64>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.define_loops 0
// CHECK-NOT:       krnl.parallel
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]][] : memref<i64>
// CHECK:             [[VAR_2_:%.+]] = arith.index_cast [[LOAD_VAR_0_MEM_]] : i64 to index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]]{{.}} : memref<4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][] : memref<f32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<f32>
// CHECK:         }

}

