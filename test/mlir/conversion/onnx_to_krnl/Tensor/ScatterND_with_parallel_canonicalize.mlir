// RUN: onnx-mlir-opt --convert-onnx-to-krnl=enable-parallel --canonicalize %s -split-input-file | FileCheck %s

// Parallel coverage for ScatterND. With reduction "none" the spec requires
// indices to have no duplicate entries, so the whole updates loop nest is
// order-independent and its window is [0, updatesRank). With a reduction,
// duplicate indices are allowed and the nest stays sequential.

// Static shapes, q = 3, k = 2: the outermost level (trip count 2) is below the
// parallel threshold, so the search moves to the next level (trip count 5).
func.func @test_parallel_scatter_nd_static(%arg0: tensor<8x16x32xf32> {onnx.name = "data"}, %arg1: tensor<2x5x32xf32> {onnx.name = "updates"}) -> (tensor<8x16x32xf32> {onnx.name = "output"}) {
  %idx = onnx.Constant dense<[[[1, 2], [3, 4], [0, 0], [7, 15], [5, 9]], [[2, 2], [6, 1], [4, 3], [1, 0], [7, 0]]]> : tensor<2x5x2xi64>
  %0 = "onnx.ScatterND"(%arg0, %idx, %arg1) : (tensor<8x16x32xf32>, tensor<2x5x2xi64>, tensor<2x5x32xf32>) -> tensor<8x16x32xf32>
  return %0 : tensor<8x16x32xf32>

// CHECK-LABEL:  func.func @test_parallel_scatter_nd_static
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<8x16x32xf32> {onnx.name = "data"}, [[PARAM_1_:%.+]]: memref<2x5x32xf32> {onnx.name = "updates"}) -> (memref<8x16x32xf32> {onnx.name = "output"}) {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_4096_:%.+]] = arith.constant 4096 : i64
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [2, 5, 2], value = dense<{{.}}{{.}}[1, 2], [3, 4], [0, 0], [7, 15], [5, 9]{{.}}, {{.}}[2, 2], [6, 1], [4, 3], [1, 0], [7, 0]{{.}}{{.}}> : tensor<2x5x2xi64>}> : () -> memref<2x5x2xi64>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 16 : memref<8x16x32xf32>
// CHECK:           "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[CST_4096_]], [[CST_0_]], [[CST_0_]]) : (memref<8x16x32xf32>, memref<8x16x32xf32>, i64, index, index) -> ()
// CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.parallel([[LOOP_0_]]#1) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 5, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 32){
// CHECK:             [[VAR_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[CST_0_]]{{.}} : memref<2x5x2xi64>
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.index_cast [[LOAD_VAR_0_MEM_]] : i64 to index
// CHECK-DAG:         [[LOAD_VAR_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[CST_1_]]{{.}} : memref<2x5x2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.index_cast [[LOAD_VAR_0_MEM_1_]] : i64 to index
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2] : memref<2x5x32xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_]]{{.}}[[VAR_4_]], [[VAR_6_]], [[VAR_2_]]#2] : memref<8x16x32xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<8x16x32xf32>
// CHECK:         }

}

// -----

// Mixed static and dynamic shapes, k = 1: each index picks a 16x? slice.
func.func @test_parallel_scatter_nd_mixed(%arg0: tensor<8x16x?xf32> {onnx.name = "data"}, %arg1: tensor<4x16x?xf32> {onnx.name = "updates"}) -> (tensor<8x16x?xf32> {onnx.name = "output"}) {
  %idx = onnx.Constant dense<[[5], [0], [2], [7]]> : tensor<4x1xi64>
  %0 = "onnx.ScatterND"(%arg0, %idx, %arg1) : (tensor<8x16x?xf32>, tensor<4x1xi64>, tensor<4x16x?xf32>) -> tensor<8x16x?xf32>
  return %0 : tensor<8x16x?xf32>


// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 128)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> ((d0 * 128) ceildiv 262144)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 * 262144)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0)[s0] -> (d0 * -262144 + s0 * 128, 262144)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-LABEL:  func.func @test_parallel_scatter_nd_mixed
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<8x16x?xf32> {onnx.name = "data"}, [[PARAM_1_:%.+]]: memref<4x16x?xf32> {onnx.name = "updates"}) -> (memref<8x16x?xf32> {onnx.name = "output"}) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_2097152_:%.+]] = arith.constant 2097152 : index
// CHECK-DAG:       [[CST_128_:%.+]] = arith.constant 128 : i64
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [4, 1], value = dense<{{.}}[5], [0], [2], [7]{{.}}> : tensor<4x1xi64>}> : () -> memref<4x1xi64>
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<8x16x?xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) alignment = 16 : memref<8x16x?xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<8x16x?xf32>
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_dim_0_]] : index to i64
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.muli [[VAR_1_]], [[CST_128_]] : i64
// CHECK-DAG:       [[VAR_3_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK:           [[VAR_4_:%.+]] = arith.cmpi sge, [[VAR_3_]], [[CST_2097152_]] : index
// CHECK:           scf.if [[VAR_4_]] {
// CHECK:             [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.parallel([[LOOP_0_]]) : !krnl.loop
// CHECK:             krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]])){
// CHECK:               [[VAR_7_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[VAR_8_:%.+]] = affine.apply [[MAP_2_]]([[VAR_7_]])
// CHECK-DAG:           [[VAR_9_:%.+]] = affine.min [[MAP_3_]]([[VAR_7_]]){{.}}[[VAR_dim_]]{{.}}
// CHECK:               [[VAR_10_:%.+]] = arith.index_cast [[VAR_9_]] : index to i64
// CHECK:               "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[VAR_10_]], [[VAR_8_]], [[VAR_8_]]) : (memref<8x16x?xf32>, memref<8x16x?xf32>, i64, index, index) -> ()
// CHECK:             }
// CHECK:           } else {
// CHECK:             "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[VAR_2_]], [[CST_0_]], [[CST_0_]]) : (memref<8x16x?xf32>, memref<8x16x?xf32>, i64, index, index) -> ()
// CHECK:           }
// CHECK-DAG:       [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_1_]], [[CST_2_]] : memref<4x16x?xf32>
// CHECK:           krnl.parallel([[LOOP_1_]]#0) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_1_]]#1 -> [[I_2_:%.+]] = 0 to 16, [[LOOP_1_]]#2 -> [[I_3_:%.+]] = 0 to [[MAP_4_]]([[VAR_dim_]], [[VAR_dim_1_]])){
// CHECK:             [[LOOP_0_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[VAR_7_1_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[LOOP_0_]]#0, [[CST_0_]]{{.}} : memref<4x1xi64>
// CHECK-DAG:         [[VAR_8_1_:%.+]] = arith.index_cast [[VAR_7_1_]] : i64 to index
// CHECK-DAG:         [[VAR_9_1_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2] : memref<4x16x?xf32>
// CHECK:             krnl.store [[VAR_9_1_]], [[RES_]]{{.}}[[VAR_8_1_]], [[LOOP_0_]]#1, [[LOOP_0_]]#2] : memref<8x16x?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<8x16x?xf32>
// CHECK:         }

}

// -----

// All slice dims dynamic, including a degenerate 1.
func.func @test_parallel_scatter_nd_dynamic(%arg0: tensor<?x?x?xf32> {onnx.name = "data"}, %arg1: tensor<4x?x?xf32> {onnx.name = "updates"}) -> (tensor<?x?x?xf32> {onnx.name = "output"}) {
  %idx = onnx.Constant dense<[[5], [0], [2], [3]]> : tensor<4x1xi64>
  %0 = "onnx.ScatterND"(%arg0, %idx, %arg1) : (tensor<?x?x?xf32>, tensor<4x1xi64>, tensor<4x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>


// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 * 262144)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0)[s0] -> (d0 * -262144 + s0, 262144)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-LABEL:  func.func @test_parallel_scatter_nd_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x?xf32> {onnx.name = "data"}, [[PARAM_1_:%.+]]: memref<4x?x?xf32> {onnx.name = "updates"}) -> (memref<?x?x?xf32> {onnx.name = "output"}) {
// CHECK-DAG:       [[CST_2097152_:%.+]] = arith.constant 2097152 : index
// CHECK-DAG:       [[CST_262144_:%.+]] = arith.constant 262144 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [4, 1], value = dense<{{.}}[5], [0], [2], [3]{{.}}> : tensor<4x1xi64>}> : () -> memref<4x1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<?x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_0_]], [[VAR_dim_1_]]) alignment = 16 : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[VAR_dim_2_]] : index to i64
// CHECK-DAG:       [[VAR_dim_3_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x?xf32>
// CHECK:           [[VAR_2_:%.+]] = arith.index_cast [[VAR_dim_3_]] : index to i64
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.muli [[VAR_1_]], [[VAR_2_]] : i64
// CHECK-DAG:       [[VAR_dim_4_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<?x?x?xf32>
// CHECK:           [[VAR_4_:%.+]] = arith.index_cast [[VAR_dim_4_]] : index to i64
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.muli [[VAR_3_]], [[VAR_4_]] : i64
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.muli [[VAR_dim_]], [[VAR_dim_0_]] : index
// CHECK:           [[VAR_7_:%.+]] = arith.muli [[VAR_6_]], [[VAR_dim_1_]] : index
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.ceildivsi [[VAR_7_]], [[CST_262144_]] : index
// CHECK-DAG:       [[VAR_9_:%.+]] = arith.cmpi sge, [[VAR_7_]], [[CST_2097152_]] : index
// CHECK:           scf.if [[VAR_9_]] {
// CHECK:             [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.parallel([[LOOP_0_]]) : !krnl.loop
// CHECK:             krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[VAR_8_]]){
// CHECK:               [[VAR_12_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[VAR_13_:%.+]] = affine.apply [[MAP_0_]]([[VAR_12_]])
// CHECK-DAG:           [[VAR_14_:%.+]] = affine.min [[MAP_1_]]([[VAR_12_]]){{.}}[[VAR_7_]]{{.}}
// CHECK:               [[VAR_15_:%.+]] = arith.index_cast [[VAR_14_]] : index to i64
// CHECK:               "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[VAR_15_]], [[VAR_13_]], [[VAR_13_]]) : (memref<?x?x?xf32>, memref<?x?x?xf32>, i64, index, index) -> ()
// CHECK:             }
// CHECK:           } else {
// CHECK:             "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[VAR_5_]], [[CST_0_]], [[CST_0_]]) : (memref<?x?x?xf32>, memref<?x?x?xf32>, i64, index, index) -> ()
// CHECK:           }
// CHECK-DAG:       [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK-DAG:       [[VAR_dim_5_:%.+]] = memref.dim [[PARAM_1_]], [[CST_1_]] : memref<4x?x?xf32>
// CHECK-DAG:       [[VAR_dim_6_:%.+]] = memref.dim [[PARAM_1_]], [[CST_2_]] : memref<4x?x?xf32>
// CHECK:           krnl.parallel([[LOOP_1_]]#0) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_1_]]#1 -> [[I_2_:%.+]] = 0 to [[MAP_2_]]([[VAR_dim_5_]]), [[LOOP_1_]]#2 -> [[I_3_:%.+]] = 0 to [[MAP_3_]]([[VAR_dim_5_]], [[VAR_dim_6_]])){
// CHECK:             [[LOOP_0_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[VAR_12_1_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[LOOP_0_]]#0, [[CST_0_]]{{.}} : memref<4x1xi64>
// CHECK-DAG:         [[VAR_13_1_:%.+]] = arith.index_cast [[VAR_12_1_]] : i64 to index
// CHECK-DAG:         [[VAR_14_1_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2] : memref<4x?x?xf32>
// CHECK:             krnl.store [[VAR_14_1_]], [[RES_]]{{.}}[[VAR_13_1_]], [[LOOP_0_]]#1, [[LOOP_0_]]#2] : memref<?x?x?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x?xf32>
// CHECK:         }

}

// -----

// Negative indices count from the back and are normalized inside the parallel
// body: -1 is row 9 and -10 is row 0.
func.func @test_parallel_scatter_nd_negative(%arg0: tensor<10x24xf32> {onnx.name = "data"}, %arg1: tensor<6x24xf32> {onnx.name = "updates"}) -> (tensor<10x24xf32> {onnx.name = "output"}) {
  %idx = onnx.Constant dense<[[-1], [3], [-10], [5], [-3], [1]]> : tensor<6x1xi64>
  %0 = "onnx.ScatterND"(%arg0, %idx, %arg1) : (tensor<10x24xf32>, tensor<6x1xi64>, tensor<6x24xf32>) -> tensor<10x24xf32>
  return %0 : tensor<10x24xf32>

// CHECK-LABEL:  func.func @test_parallel_scatter_nd_negative
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10x24xf32> {onnx.name = "data"}, [[PARAM_1_:%.+]]: memref<6x24xf32> {onnx.name = "updates"}) -> (memref<10x24xf32> {onnx.name = "output"}) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_240_:%.+]] = arith.constant 240 : i64
// CHECK-DAG:       [[CST_10_:%.+]] = arith.constant 10 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [6, 1], value = dense<{{.}}[-1], [3], [-10], [5], [-3], [1]{{.}}> : tensor<6x1xi64>}> : () -> memref<6x1xi64>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 16 : memref<10x24xf32>
// CHECK:           "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[CST_240_]], [[CST_0_]], [[CST_0_]]) : (memref<10x24xf32>, memref<10x24xf32>, i64, index, index) -> ()
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.parallel([[LOOP_0_]]#0) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 6, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 24){
// CHECK:             [[VAR_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[VAR_2_]]#0, [[CST_0_]]{{.}} : memref<6x1xi64>
// CHECK:             [[VAR_4_:%.+]] = arith.index_cast [[LOAD_VAR_0_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.cmpi slt, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.addi [[VAR_4_]], [[CST_10_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.select [[VAR_5_]], [[VAR_6_]], [[VAR_4_]] : index
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1] : memref<6x24xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_]]{{.}}[[VAR_7_]], [[VAR_2_]]#1] : memref<10x24xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<10x24xf32>
// CHECK:         }

}

// -----

// q = 1: a single index tuple, so the nest is only the slice levels.
func.func @test_parallel_scatter_nd_single_tuple(%arg0: tensor<4x5x16x24xf32> {onnx.name = "data"}, %arg1: tensor<16x24xf32> {onnx.name = "updates"}) -> (tensor<4x5x16x24xf32> {onnx.name = "output"}) {
  %idx = onnx.Constant dense<[2, 3]> : tensor<2xi64>
  %0 = "onnx.ScatterND"(%arg0, %idx, %arg1) : (tensor<4x5x16x24xf32>, tensor<2xi64>, tensor<16x24xf32>) -> tensor<4x5x16x24xf32>
  return %0 : tensor<4x5x16x24xf32>

// CHECK-LABEL:  func.func @test_parallel_scatter_nd_single_tuple
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4x5x16x24xf32> {onnx.name = "data"}, [[PARAM_1_:%.+]]: memref<16x24xf32> {onnx.name = "updates"}) -> (memref<4x5x16x24xf32> {onnx.name = "output"}) {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_7680_:%.+]] = arith.constant 7680 : i64
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [2], value = dense<[2, 3]> : tensor<2xi64>}> : () -> memref<2xi64>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 16 : memref<4x5x16x24xf32>
// CHECK:           "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[CST_7680_]], [[CST_0_]], [[CST_0_]]) : (memref<4x5x16x24xf32>, memref<4x5x16x24xf32>, i64, index, index) -> ()
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.parallel([[LOOP_0_]]#0) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 16, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 24){
// CHECK-DAG:         [[VAR_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[CST_0_]]{{.}} : memref<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.index_cast [[LOAD_VAR_0_MEM_]] : i64 to index
// CHECK-DAG:         [[LOAD_VAR_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[CST_1_]]{{.}} : memref<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.index_cast [[LOAD_VAR_0_MEM_1_]] : i64 to index
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1] : memref<16x24xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_]]{{.}}[[VAR_4_]], [[VAR_6_]], [[VAR_2_]]#0, [[VAR_2_]]#1] : memref<4x5x16x24xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<4x5x16x24xf32>
// CHECK:         }

}

// -----

// reduction "add" with duplicate indices (row 1 twice): no krnl.parallel, since
// two iterations of the outer level read-modify-write the same output row.
func.func @test_no_parallel_scatter_nd_add(%arg0: tensor<8x32xf32> {onnx.name = "data"}, %arg1: tensor<6x32xf32> {onnx.name = "updates"}) -> (tensor<8x32xf32> {onnx.name = "output"}) {
  %idx = onnx.Constant dense<[[1], [3], [1], [0], [6], [3]]> : tensor<6x1xi64>
  %0 = "onnx.ScatterND"(%arg0, %idx, %arg1) {reduction = "add"} : (tensor<8x32xf32>, tensor<6x1xi64>, tensor<6x32xf32>) -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
// CHECK-LABEL:  func.func @test_no_parallel_scatter_nd_add
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<8x32xf32> {onnx.name = "data"}, [[PARAM_1_:%.+]]: memref<6x32xf32> {onnx.name = "updates"}) -> (memref<8x32xf32> {onnx.name = "output"}) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : i64
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [6, 1], value = dense<{{.}}[1], [3], [1], [0], [6], [3]{{.}}> : tensor<6x1xi64>}> : () -> memref<6x1xi64>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 16 : memref<8x32xf32>
// CHECK:           "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[CST_256_]], [[CST_0_]], [[CST_0_]]) : (memref<8x32xf32>, memref<8x32xf32>, i64, index, index) -> ()
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 6, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 32){
// CHECK:             [[VAR_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[VAR_2_]]#0, [[CST_0_]]{{.}} : memref<6x1xi64>
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.index_cast [[LOAD_VAR_0_MEM_]] : i64 to index
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1] : memref<6x32xf32>
// CHECK:             [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_4_]], [[VAR_2_]]#1] : memref<8x32xf32>
// CHECK:             [[VAR_7_:%.+]] = arith.addf [[LOAD_RES_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             krnl.store [[VAR_7_]], [[RES_]]{{.}}[[VAR_4_]], [[VAR_2_]]#1] : memref<8x32xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<8x32xf32>
// CHECK:         }

}


// -----

// A 16.0 MiB copy of `data`, above the parallel threshold: it is split into
// 1 MiB chunks copied in parallel. 4100 x 1024 elements is not a multiple of
// the chunk, so the last chunk is partial.
func.func @test_parallel_scatter_nd_copy_static(%arg0: tensor<4100x1024xf32> {onnx.name = "data"}, %arg1: tensor<2x1024xf32> {onnx.name = "updates"}) -> (tensor<4100x1024xf32> {onnx.name = "output"}) {
  %idx = onnx.Constant dense<[[3], [4097]]> : tensor<2x1xi64>
  %0 = "onnx.ScatterND"(%arg0, %idx, %arg1) : (tensor<4100x1024xf32>, tensor<2x1xi64>, tensor<2x1024xf32>) -> tensor<4100x1024xf32>
  return %0 : tensor<4100x1024xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 * 262144)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * -262144 + 4198400, 262144)>
// CHECK-LABEL:  func.func @test_parallel_scatter_nd_copy_static
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4100x1024xf32> {onnx.name = "data"}, [[PARAM_1_:%.+]]: memref<2x1024xf32> {onnx.name = "updates"}) -> (memref<4100x1024xf32> {onnx.name = "output"}) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [2, 1], value = dense<{{.}}[3], [4097]{{.}}> : tensor<2x1xi64>}> : () -> memref<2x1xi64>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 16 : memref<4100x1024xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.parallel([[LOOP_0_]]) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 17){
// CHECK:             [[VAR_3_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.apply [[MAP_0_]]([[VAR_3_]])
// CHECK-DAG:         [[VAR_5_:%.+]] = affine.min [[MAP_1_]]([[VAR_3_]])
// CHECK:             [[VAR_6_:%.+]] = arith.index_cast [[VAR_5_]] : index to i64
// CHECK:             "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[VAR_6_]], [[VAR_4_]], [[VAR_4_]]) : (memref<4100x1024xf32>, memref<4100x1024xf32>, i64, index, index) -> ()
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.parallel([[LOOP_1_]]#1) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_1_]]#1 -> [[I_2_:%.+]] = 0 to 1024){
// CHECK:             [[VAR_3_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[VAR_4_1_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[VAR_3_1_]]#0, [[CST_0_]]{{.}} : memref<2x1xi64>
// CHECK-DAG:         [[VAR_5_1_:%.+]] = arith.index_cast [[VAR_4_1_]] : i64 to index
// CHECK-DAG:         [[VAR_6_1_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_3_1_]]#0, [[VAR_3_1_]]#1] : memref<2x1024xf32>
// CHECK:             krnl.store [[VAR_6_1_]], [[RES_]]{{.}}[[VAR_5_1_]], [[VAR_3_1_]]#1] : memref<4100x1024xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<4100x1024xf32>
// CHECK:         }

}

// -----

// A copy of dynamic size: the parallel or serial copy is chosen at runtime.
func.func @test_parallel_scatter_nd_copy_dynamic(%arg0: tensor<?x1024xf32> {onnx.name = "data"}, %arg1: tensor<2x1024xf32> {onnx.name = "updates"}) -> (tensor<?x1024xf32> {onnx.name = "output"}) {
  %idx = onnx.Constant dense<[[0], [2]]> : tensor<2x1xi64>
  %0 = "onnx.ScatterND"(%arg0, %idx, %arg1) : (tensor<?x1024xf32>, tensor<2x1xi64>, tensor<2x1024xf32>) -> tensor<?x1024xf32>
  return %0 : tensor<?x1024xf32>
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 1024)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> ((d0 * 1024) ceildiv 262144)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 * 262144)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0)[s0] -> (d0 * -262144 + s0 * 1024, 262144)>
// CHECK-LABEL:  func.func @test_parallel_scatter_nd_copy_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x1024xf32> {onnx.name = "data"}, [[PARAM_1_:%.+]]: memref<2x1024xf32> {onnx.name = "updates"}) -> (memref<?x1024xf32> {onnx.name = "output"}) {
// CHECK-DAG:       [[CST_2097152_:%.+]] = arith.constant 2097152 : index
// CHECK-DAG:       [[CST_1024_:%.+]] = arith.constant 1024 : i64
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [2, 1], value = dense<{{.}}[0], [2]{{.}}> : tensor<2x1xi64>}> : () -> memref<2x1xi64>
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x1024xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) alignment = 16 : memref<?x1024xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x1024xf32>
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_dim_0_]] : index to i64
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.muli [[VAR_1_]], [[CST_1024_]] : i64
// CHECK-DAG:       [[VAR_3_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK:           [[VAR_4_:%.+]] = arith.cmpi sge, [[VAR_3_]], [[CST_2097152_]] : index
// CHECK:           scf.if [[VAR_4_]] {
// CHECK:             [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.parallel([[LOOP_0_]]) : !krnl.loop
// CHECK:             krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]])){
// CHECK:               [[VAR_7_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[VAR_8_:%.+]] = affine.apply [[MAP_2_]]([[VAR_7_]])
// CHECK-DAG:           [[VAR_9_:%.+]] = affine.min [[MAP_3_]]([[VAR_7_]]){{.}}[[VAR_dim_]]{{.}}
// CHECK:               [[VAR_10_:%.+]] = arith.index_cast [[VAR_9_]] : index to i64
// CHECK:               "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[VAR_10_]], [[VAR_8_]], [[VAR_8_]]) : (memref<?x1024xf32>, memref<?x1024xf32>, i64, index, index) -> ()
// CHECK:             }
// CHECK:           } else {
// CHECK:             "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[VAR_2_]], [[CST_0_]], [[CST_0_]]) : (memref<?x1024xf32>, memref<?x1024xf32>, i64, index, index) -> ()
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.parallel([[LOOP_1_]]#1) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_1_]]#1 -> [[I_2_:%.+]] = 0 to 1024){
// CHECK:             [[LOOP_0_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[VAR_7_1_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[LOOP_0_]]#0, [[CST_0_]]{{.}} : memref<2x1xi64>
// CHECK-DAG:         [[VAR_8_1_:%.+]] = arith.index_cast [[VAR_7_1_]] : i64 to index
// CHECK-DAG:         [[VAR_9_1_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[LOOP_0_]]#0, [[LOOP_0_]]#1] : memref<2x1024xf32>
// CHECK:             krnl.store [[VAR_9_1_]], [[RES_]]{{.}}[[VAR_8_1_]], [[LOOP_0_]]#1] : memref<?x1024xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x1024xf32>
// CHECK:         }

}

