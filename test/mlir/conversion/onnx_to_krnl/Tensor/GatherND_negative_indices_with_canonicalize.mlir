// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// GROUND-ALL: --lower-bound=int64:-3,float32:-100 --upper-bound=int64:2,float32:100 --rtol=0 --atol=0

// GatherND indices may be negative, counting back from the end of the data
// dimension they address, so the lowering adds that dimension to a negative
// index. The baseline file computes the same results by normalizing the
// indices with ONNX ops before the GatherND.

func.func @test_gather_nd_negative_indices(%arg0 : tensor<3x4xf32>, %arg1 : tensor<6x2xi64>) -> tensor<6xf32> {
  %0 = "onnx.GatherND"(%arg0, %arg1) {batch_dims = 0 : si64} : (tensor<3x4xf32>, tensor<6x2xi64>) -> tensor<6xf32>
  "func.return"(%0) : (tensor<6xf32>) -> ()

// CHECK-LABEL:  func.func @test_gather_nd_negative_indices
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x4xf32>, [[PARAM_1_:%.+]]: memref<6x2xi64>) -> memref<6xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: [0], sizes: [1, 6, 2], strides: [12, 2, 1] : memref<6x2xi64> to memref<1x6x2xi64>
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [1, 3, 4], strides: [12, 4, 1] : memref<3x4xf32> to memref<1x3x4xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<6xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_]], [[RES_1_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 6){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_:%.+]] = krnl.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[CST_0_]]{{.}} : memref<1x6x2xi64>
// CHECK:             [[VAR_3_:%.+]] = arith.index_cast [[LOAD_VAR_reinterpret_cast_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.addi [[VAR_3_]], [[CST_3_]] : index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.cmpi slt, [[VAR_3_]], [[CST_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.select [[VAR_5_]], [[VAR_4_]], [[VAR_3_]] : index
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_MEM_1_:%.+]] = krnl.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[CST_1_]]{{.}} : memref<1x6x2xi64>
// CHECK:             [[VAR_8_:%.+]] = arith.index_cast [[LOAD_VAR_reinterpret_cast_MEM_1_]] : i64 to index
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.addi [[VAR_8_]], [[CST_4_]] : index
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.cmpi slt, [[VAR_8_]], [[CST_0_]] : index
// CHECK:             [[VAR_11_:%.+]] = arith.select [[VAR_10_]], [[VAR_9_]], [[VAR_8_]] : index
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_0_MEM_:%.+]] = krnl.load [[VAR_reinterpret_cast_0_]]{{.}}[[VAR_1_]]#0, [[VAR_6_]], [[VAR_11_]]{{.}} : memref<1x3x4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:             krnl.store [[LOAD_VAR_reinterpret_cast_0_MEM_]], [[RES_]]{{.}}[[LOAD_RES_1_MEM_]]{{.}} : memref<6xf32>
// CHECK:             [[VAR_14_:%.+]] = arith.addi [[LOAD_RES_1_MEM_]], [[CST_1_]] : index
// CHECK:             krnl.store [[VAR_14_]], [[RES_1_]][] : memref<index>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<6xf32>
// CHECK:         }

}

// -----

// With batch_dims = 1, the index addresses data dimension 1.
func.func @test_gather_nd_negative_indices_batch_dims(%arg0 : tensor<2x3xf32>, %arg1 : tensor<2x6x1xi64>) -> tensor<2x6xf32> {
  %0 = "onnx.GatherND"(%arg0, %arg1) {batch_dims = 1 : si64} : (tensor<2x3xf32>, tensor<2x6x1xi64>) -> tensor<2x6xf32>
  "func.return"(%0) : (tensor<2x6xf32>) -> ()

// CHECK-LABEL:  func.func @test_gather_nd_negative_indices_batch_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x3xf32>, [[PARAM_1_:%.+]]: memref<2x6x1xi64>) -> memref<2x6xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<12xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_]], [[RES_1_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 6){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[CST_0_]]{{.}} : memref<2x6x1xi64>
// CHECK:             [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.addi [[VAR_3_]], [[CST_3_]] : index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.cmpi slt, [[VAR_3_]], [[CST_0_]] : index
// CHECK:             [[VAR_6_:%.+]] = arith.select [[VAR_5_]], [[VAR_4_]], [[VAR_3_]] : index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_6_]]{{.}} : memref<2x3xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[LOAD_RES_1_MEM_]]{{.}} : memref<12xf32>
// CHECK:             [[VAR_9_:%.+]] = arith.addi [[LOAD_RES_1_MEM_]], [[CST_1_]] : index
// CHECK:             krnl.store [[VAR_9_]], [[RES_1_]][] : memref<index>
// CHECK:           }
// CHECK:           [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 6], strides: [6, 1] : memref<12xf32> to memref<2x6xf32>
// CHECK:           return [[VAR_reinterpret_cast_]] : memref<2x6xf32>
// CHECK:         }

}

// -----

// GROUND-THIS: --shape-info=0:3x5,1:8x2
func.func @test_gather_nd_negative_indices_dynamic(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x2xi64>) -> tensor<?xf32> {
  %0 = "onnx.GatherND"(%arg0, %arg1) {batch_dims = 0 : si64} : (tensor<?x?xf32>, tensor<?x2xi64>) -> tensor<?xf32>
  "func.return"(%0) : (tensor<?xf32>) -> ()

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 2)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @test_gather_nd_negative_indices_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?xf32>, [[PARAM_1_:%.+]]: memref<?x2xi64>) -> memref<?xf32> {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x2xi64>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x2xi64>
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_2_]]{{.}}
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: [0], sizes: [1, [[VAR_dim_2_]], 2], strides: {{.}}[[VAR_0_]], 2, 1] : memref<?x2xi64> to memref<1x?x2xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.muli [[VAR_dim_1_]], [[VAR_dim_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_3_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [1, [[VAR_dim_0_]], [[VAR_dim_1_]]{{.}}, strides: {{.}}[[VAR_1_]], [[VAR_dim_1_]], 1] : memref<?x?xf32> to memref<1x?x?xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) : memref<?xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_]], [[RES_1_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_2_]])){
// CHECK:             [[VAR_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_:%.+]] = krnl.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[CST_0_]]{{.}} : memref<1x?x2xi64>
// CHECK:             [[VAR_5_:%.+]] = arith.index_cast [[LOAD_VAR_reinterpret_cast_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.addi [[VAR_5_]], [[VAR_dim_0_]] : index
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.cmpi slt, [[VAR_5_]], [[CST_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[VAR_6_]], [[VAR_5_]] : index
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_MEM_1_:%.+]] = krnl.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[CST_1_]]{{.}} : memref<1x?x2xi64>
// CHECK:             [[VAR_10_:%.+]] = arith.index_cast [[LOAD_VAR_reinterpret_cast_MEM_1_]] : i64 to index
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.addi [[VAR_10_]], [[VAR_dim_1_]] : index
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.cmpi slt, [[VAR_10_]], [[CST_0_]] : index
// CHECK:             [[VAR_13_:%.+]] = arith.select [[VAR_12_]], [[VAR_11_]], [[VAR_10_]] : index
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_3_MEM_:%.+]] = krnl.load [[VAR_reinterpret_cast_3_]]{{.}}[[VAR_3_]]#0, [[VAR_8_]], [[VAR_13_]]{{.}} : memref<1x?x?xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:             krnl.store [[LOAD_VAR_reinterpret_cast_3_MEM_]], [[RES_]]{{.}}[[LOAD_RES_1_MEM_]]{{.}} : memref<?xf32>
// CHECK:             [[VAR_16_:%.+]] = arith.addi [[LOAD_RES_1_MEM_]], [[CST_1_]] : index
// CHECK:             krnl.store [[VAR_16_]], [[RES_1_]][] : memref<index>
// CHECK:           }
// CHECK:           [[VAR_reinterpret_cast_4_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: {{.}}[[VAR_dim_]]{{.}}, strides: [1] : memref<?xf32> to memref<?xf32>
// CHECK:           return [[VAR_reinterpret_cast_4_]] : memref<?xf32>
// CHECK:         }

}

// -----

// GROUND-THIS: --shape-info=0:3x5
func.func @test_gather_nd_negative_indices_mixed(%arg0 : tensor<3x?xf32>, %arg1 : tensor<6x2xi64>) -> tensor<6xf32> {
  %0 = "onnx.GatherND"(%arg0, %arg1) {batch_dims = 0 : si64} : (tensor<3x?xf32>, tensor<6x2xi64>) -> tensor<6xf32>
  "func.return"(%0) : (tensor<6xf32>) -> ()

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 3)>
// CHECK-LABEL:  func.func @test_gather_nd_negative_indices_mixed
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x?xf32>, [[PARAM_1_:%.+]]: memref<6x2xi64>) -> memref<6xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<3x?xf32>
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: [0], sizes: [1, 6, 2], strides: [12, 2, 1] : memref<6x2xi64> to memref<1x6x2xi64>
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [1, 3, [[VAR_dim_]]{{.}}, strides: {{.}}[[VAR_0_]], [[VAR_dim_]], 1] : memref<3x?xf32> to memref<1x3x?xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<6xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_]], [[RES_1_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 6){
// CHECK:             [[VAR_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_:%.+]] = krnl.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[CST_0_]]{{.}} : memref<1x6x2xi64>
// CHECK:             [[VAR_4_:%.+]] = arith.index_cast [[LOAD_VAR_reinterpret_cast_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.addi [[VAR_4_]], [[CST_3_]] : index
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.cmpi slt, [[VAR_4_]], [[CST_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.select [[VAR_6_]], [[VAR_5_]], [[VAR_4_]] : index
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_MEM_1_:%.+]] = krnl.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[CST_1_]]{{.}} : memref<1x6x2xi64>
// CHECK:             [[VAR_9_:%.+]] = arith.index_cast [[LOAD_VAR_reinterpret_cast_MEM_1_]] : i64 to index
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.addi [[VAR_9_]], [[VAR_dim_]] : index
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.cmpi slt, [[VAR_9_]], [[CST_0_]] : index
// CHECK:             [[VAR_12_:%.+]] = arith.select [[VAR_11_]], [[VAR_10_]], [[VAR_9_]] : index
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_0_MEM_:%.+]] = krnl.load [[VAR_reinterpret_cast_0_]]{{.}}[[VAR_2_]]#0, [[VAR_7_]], [[VAR_12_]]{{.}} : memref<1x3x?xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:             krnl.store [[LOAD_VAR_reinterpret_cast_0_MEM_]], [[RES_]]{{.}}[[LOAD_RES_1_MEM_]]{{.}} : memref<6xf32>
// CHECK:             [[VAR_15_:%.+]] = arith.addi [[LOAD_RES_1_MEM_]], [[CST_1_]] : index
// CHECK:             krnl.store [[VAR_15_]], [[RES_1_]][] : memref<index>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<6xf32>
// CHECK:         }

}

// -----

// Constant non-negative indices need no normalization.
func.func @test_gather_nd_nonnegative_constant_indices(%arg0 : tensor<3x4xf32>) -> tensor<2xf32> {
  %0 = onnx.Constant dense<[[2, 3], [0, 1]]> : tensor<2x2xi64>
  %1 = "onnx.GatherND"(%arg0, %0) {batch_dims = 0 : si64} : (tensor<3x4xf32>, tensor<2x2xi64>) -> tensor<2xf32>
  "func.return"(%1) : (tensor<2xf32>) -> ()
// CHECK-LABEL:  func.func @test_gather_nd_nonnegative_constant_indices
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x4xf32>) -> memref<2xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [2, 2], value = dense<{{.}}[2, 3], [0, 1]{{.}}> : tensor<2x2xi64>}> : () -> memref<2x2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[VAR_0_]] to offset: [0], sizes: [1, 2, 2], strides: [4, 2, 1] : memref<2x2xi64> to memref<1x2x2xi64>
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [1, 3, 4], strides: [12, 4, 1] : memref<3x4xf32> to memref<1x3x4xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<2xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_]], [[RES_1_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[VAR_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_:%.+]] = krnl.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[CST_0_]]{{.}} : memref<1x2x2xi64>
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.index_cast [[LOAD_VAR_reinterpret_cast_MEM_]] : i64 to index
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_MEM_1_:%.+]] = krnl.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[CST_1_]]{{.}} : memref<1x2x2xi64>
// CHECK:             [[VAR_6_:%.+]] = arith.index_cast [[LOAD_VAR_reinterpret_cast_MEM_1_]] : i64 to index
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_0_MEM_:%.+]] = krnl.load [[VAR_reinterpret_cast_0_]]{{.}}[[VAR_2_]]#0, [[VAR_4_]], [[VAR_6_]]{{.}} : memref<1x3x4xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:             krnl.store [[LOAD_VAR_reinterpret_cast_0_MEM_]], [[RES_]]{{.}}[[LOAD_RES_1_MEM_]]{{.}} : memref<2xf32>
// CHECK:             [[VAR_9_:%.+]] = arith.addi [[LOAD_RES_1_MEM_]], [[CST_1_]] : index
// CHECK:             krnl.store [[VAR_9_]], [[RES_1_]][] : memref<index>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2xf32>
// CHECK:         }

}
