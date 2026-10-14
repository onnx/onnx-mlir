// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

func.func @test_scatter_nd1(%arg0: tensor<4x4x4xf32>, %arg1: tensor<2x1xi64>, %arg2: tensor<2x4x4xf32>) -> tensor<4x4x4xf32> {
  %0 = "onnx.ScatterND"(%arg0, %arg1, %arg2) : (tensor<4x4x4xf32>, tensor<2x1xi64>, tensor<2x4x4xf32>) -> tensor<4x4x4xf32>
  return %0 : tensor<4x4x4xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_scatter_nd1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4x4x4xf32>, [[PARAM_1_:%.+]]: memref<2x1xi64>, [[PARAM_2_:%.+]]: memref<2x4x4xf32>) -> memref<4x4x4xf32> {
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_4_1_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_4_2_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4x4x4xf32>
// CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : i64
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[CST_64_]], [[CST_0_]], [[CST_0_]]) : (memref<4x4x4xf32>, memref<4x4x4xf32>, i64, index, index) -> ()
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_4_3_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_4_4_:%.+]] = arith.constant 4 : index
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 4){
// CHECK-DAG:         [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[CST_0_2_]]{{.}} : memref<2x1xi64>
// CHECK-DAG:         [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:         [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpi slt, [[VAR_3_]], [[CST_0_3_]] : index
// CHECK-DAG:         [[CST_4_5_:%.+]] = arith.constant 4 : index
// CHECK:             [[VAR_5_:%.+]] = arith.addi [[VAR_3_]], [[CST_4_5_]] : index
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.select [[VAR_4_]], [[VAR_5_]], [[VAR_3_]] : index
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<2x4x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_2_MEM_]], [[RES_]]{{.}}[[VAR_6_]], [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<4x4x4xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<4x4x4xf32>
// CHECK:         }
}

// -----

// COM: Test ScatterND with dynamic shape
func.func @test_scatter_nd_with_dynamic_indices(%arg0: tensor<2x1xi64>, %arg1: tensor<?x2xi64>, %arg2: tensor<2xi64>) -> tensor<2x1xi64> {
  %0 = "onnx.ScatterND"(%arg0, %arg1, %arg2) {reduction = "none"} : (tensor<2x1xi64>, tensor<?x2xi64>, tensor<2xi64>) -> tensor<2x1xi64>
  return %0 : tensor<2x1xi64>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_scatter_nd_with_dynamic_indices
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x1xi64>, [[PARAM_1_:%.+]]: memref<?x2xi64>, [[PARAM_2_:%.+]]: memref<2xi64>) -> memref<2x1xi64> {
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x1xi64>
// CHECK-DAG:       [[CST_2_1_:%.+]] = arith.constant 2 : i64
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[CST_2_1_]], [[CST_0_]], [[CST_0_]]) : (memref<2x1xi64>, memref<2x1xi64>, i64, index, index) -> ()
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_2_2_:%.+]] = arith.constant 2 : index
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 2){
// CHECK-DAG:         [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]], [[CST_0_2_]]{{.}} : memref<?x2xi64>
// CHECK-DAG:         [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:         [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpi slt, [[VAR_3_]], [[CST_0_3_]] : index
// CHECK-DAG:         [[CST_2_3_:%.+]] = arith.constant 2 : index
// CHECK:             [[VAR_5_:%.+]] = arith.addi [[VAR_3_]], [[CST_2_3_]] : index
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.select [[VAR_4_]], [[VAR_5_]], [[VAR_3_]] : index
// CHECK-DAG:         [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK:             [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]], [[CST_1_1_]]{{.}} : memref<?x2xi64>
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_1_]] : i64 to index
// CHECK-DAG:         [[CST_0_4_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.cmpi slt, [[VAR_8_]], [[CST_0_4_]] : index
// CHECK-DAG:         [[CST_1_2_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_10_:%.+]] = arith.addi [[VAR_8_]], [[CST_1_2_]] : index
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.select [[VAR_9_]], [[VAR_10_]], [[VAR_8_]] : index
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_1_]]{{.}} : memref<2xi64>
// CHECK:             krnl.store [[LOAD_PARAM_2_MEM_]], [[RES_]]{{.}}[[VAR_6_]], [[VAR_11_]]{{.}} : memref<2x1xi64>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x1xi64>
// CHECK:         }
}

// -----

// COM: Test ScatterND with data dim >= 2d and indices dim >= data dim
func.func @test_scatter_nd_with_multiple_remaining_indices(%arg0: tensor<?x?x?x?xi64>, %arg1: tensor<?x?x?x?x4xi64>, %arg2: tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64> {
  %0 = "onnx.ScatterND"(%arg0, %arg1, %arg2) {reduction = "none"} : (tensor<?x?x?x?xi64>, tensor<?x?x?x?x4xi64>, tensor<?x?x?x?xi64>) -> tensor<?x?x?x?xi64>
  return %0 : tensor<?x?x?x?xi64>
// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK-LABEL:  func.func @test_scatter_nd_with_multiple_remaining_indices
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x?x?xi64>, [[PARAM_1_:%.+]]: memref<?x?x?x?x4xi64>, [[PARAM_2_:%.+]]: memref<?x?x?x?xi64>) -> memref<?x?x?x?xi64> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x?x?xi64>
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x?x?xi64>
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<?x?x?x?xi64>
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK:           [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_3_]] : memref<?x?x?x?xi64>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_0_]], [[VAR_dim_1_]], [[VAR_dim_2_]]) {{.*}}: memref<?x?x?x?xi64>
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_4_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_1_]] : memref<?x?x?x?xi64>
// CHECK:           [[VAR_0_:%.+]] = arith.index_cast [[VAR_dim_4_]] : index to i64
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.muli [[CST_1_1_]], [[VAR_0_]] : i64
// CHECK-DAG:       [[CST_1_2_:%.+]] = arith.constant 1 : index
// CHECK:           [[VAR_dim_6_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_2_]] : memref<?x?x?x?xi64>
// CHECK:           [[VAR_2_:%.+]] = arith.index_cast [[VAR_dim_6_]] : index to i64
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.muli [[VAR_1_]], [[VAR_2_]] : i64
// CHECK-DAG:       [[CST_2_1_:%.+]] = arith.constant 2 : index
// CHECK:           [[VAR_dim_8_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_1_]] : memref<?x?x?x?xi64>
// CHECK:           [[VAR_4_:%.+]] = arith.index_cast [[VAR_dim_8_]] : index to i64
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.muli [[VAR_3_]], [[VAR_4_]] : i64
// CHECK-DAG:       [[CST_3_1_:%.+]] = arith.constant 3 : index
// CHECK:           [[VAR_dim_10_:%.+]] = memref.dim [[PARAM_0_]], [[CST_3_1_]] : memref<?x?x?x?xi64>
// CHECK:           [[VAR_6_:%.+]] = arith.index_cast [[VAR_dim_10_]] : index to i64
// CHECK-DAG:       [[VAR_7_:%.+]] = arith.muli [[VAR_5_]], [[VAR_6_]] : i64
// CHECK-DAG:       [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK:           "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[VAR_7_]], [[CST_0_2_]], [[CST_0_2_]]) : (memref<?x?x?x?xi64>, memref<?x?x?x?xi64>, i64, index, index) -> ()
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK-DAG:       [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_4_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_14_:%.+]] = memref.dim [[PARAM_2_]], [[CST_0_4_]] : memref<?x?x?x?xi64>
// CHECK-DAG:       [[CST_1_3_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_16_:%.+]] = memref.dim [[PARAM_2_]], [[CST_1_3_]] : memref<?x?x?x?xi64>
// CHECK-DAG:       [[CST_2_2_:%.+]] = arith.constant 2 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_18_:%.+]] = memref.dim [[PARAM_2_]], [[CST_2_2_]] : memref<?x?x?x?xi64>
// CHECK-DAG:       [[CST_3_2_:%.+]] = arith.constant 3 : index
// CHECK:           [[VAR_dim_20_:%.+]] = memref.dim [[PARAM_2_]], [[CST_3_2_]] : memref<?x?x?x?xi64>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_14_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_14_]], [[VAR_dim_16_]]), [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to [[MAP_2_]]([[VAR_dim_14_]], [[VAR_dim_16_]], [[VAR_dim_18_]]), [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to [[MAP_3_]]([[VAR_dim_14_]], [[VAR_dim_16_]], [[VAR_dim_18_]], [[VAR_dim_20_]])){
// CHECK-DAG:         [[VAR_9_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[CST_0_5_:%.+]] = arith.constant 0 : index
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_9_]]#0, [[VAR_9_]]#1, [[VAR_9_]]#2, [[VAR_9_]]#3, [[CST_0_5_]]{{.}} : memref<?x?x?x?x4xi64>
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:         [[CST_0_6_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.cmpi slt, [[VAR_11_]], [[CST_0_6_]] : index
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.addi [[VAR_11_]], [[VAR_dim_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.select [[VAR_12_]], [[VAR_13_]], [[VAR_11_]] : index
// CHECK-DAG:         [[CST_1_4_:%.+]] = arith.constant 1 : index
// CHECK:             [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_9_]]#0, [[VAR_9_]]#1, [[VAR_9_]]#2, [[VAR_9_]]#3, [[CST_1_4_]]{{.}} : memref<?x?x?x?x4xi64>
// CHECK-DAG:         [[VAR_16_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_1_]] : i64 to index
// CHECK-DAG:         [[CST_0_7_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_17_:%.+]] = arith.cmpi slt, [[VAR_16_]], [[CST_0_7_]] : index
// CHECK-DAG:         [[VAR_18_:%.+]] = arith.addi [[VAR_16_]], [[VAR_dim_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_19_:%.+]] = arith.select [[VAR_17_]], [[VAR_18_]], [[VAR_16_]] : index
// CHECK-DAG:         [[CST_2_3_:%.+]] = arith.constant 2 : index
// CHECK:             [[LOAD_PARAM_1_MEM_2_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_9_]]#0, [[VAR_9_]]#1, [[VAR_9_]]#2, [[VAR_9_]]#3, [[CST_2_3_]]{{.}} : memref<?x?x?x?x4xi64>
// CHECK-DAG:         [[VAR_21_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_2_]] : i64 to index
// CHECK-DAG:         [[CST_0_8_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_22_:%.+]] = arith.cmpi slt, [[VAR_21_]], [[CST_0_8_]] : index
// CHECK-DAG:         [[VAR_23_:%.+]] = arith.addi [[VAR_21_]], [[VAR_dim_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_24_:%.+]] = arith.select [[VAR_22_]], [[VAR_23_]], [[VAR_21_]] : index
// CHECK-DAG:         [[CST_3_3_:%.+]] = arith.constant 3 : index
// CHECK:             [[LOAD_PARAM_1_MEM_3_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_9_]]#0, [[VAR_9_]]#1, [[VAR_9_]]#2, [[VAR_9_]]#3, [[CST_3_3_]]{{.}} : memref<?x?x?x?x4xi64>
// CHECK-DAG:         [[VAR_26_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_3_]] : i64 to index
// CHECK-DAG:         [[CST_0_9_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_27_:%.+]] = arith.cmpi slt, [[VAR_26_]], [[CST_0_9_]] : index
// CHECK-DAG:         [[VAR_28_:%.+]] = arith.addi [[VAR_26_]], [[VAR_dim_2_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_29_:%.+]] = arith.select [[VAR_27_]], [[VAR_28_]], [[VAR_26_]] : index
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_9_]]#0, [[VAR_9_]]#1, [[VAR_9_]]#2, [[VAR_9_]]#3] : memref<?x?x?x?xi64>
// CHECK:             krnl.store [[LOAD_PARAM_2_MEM_]], [[RES_]]{{.}}[[VAR_14_]], [[VAR_19_]], [[VAR_24_]], [[VAR_29_]]{{.}} : memref<?x?x?x?xi64>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x?x?xi64>
// CHECK:         }
}

// -----

// COM: Test ScatterND with add reduction
func.func @test_scatter_nd_add(%arg0: tensor<4x4x4xf32>, %arg1: tensor<2x1xi64>, %arg2: tensor<2x4x4xf32>) -> tensor<4x4x4xf32> {
  %0 = "onnx.ScatterND"(%arg0, %arg1, %arg2) {reduction = "add"} : (tensor<4x4x4xf32>, tensor<2x1xi64>, tensor<2x4x4xf32>) -> tensor<4x4x4xf32>
  return %0 : tensor<4x4x4xf32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_scatter_nd_add
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4x4x4xf32>, [[PARAM_1_:%.+]]: memref<2x1xi64>, [[PARAM_2_:%.+]]: memref<2x4x4xf32>) -> memref<4x4x4xf32> {
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_4_1_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_4_2_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4x4x4xf32>
// CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : i64
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[CST_64_]], [[CST_0_]], [[CST_0_]]) : (memref<4x4x4xf32>, memref<4x4x4xf32>, i64, index, index) -> ()
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_4_3_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_4_4_:%.+]] = arith.constant 4 : index
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 4){
// CHECK-DAG:         [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[CST_0_2_]]{{.}} : memref<2x1xi64>
// CHECK-DAG:         [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:         [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpi slt, [[VAR_3_]], [[CST_0_3_]] : index
// CHECK-DAG:         [[CST_4_5_:%.+]] = arith.constant 4 : index
// CHECK:             [[VAR_5_:%.+]] = arith.addi [[VAR_3_]], [[CST_4_5_]] : index
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.select [[VAR_4_]], [[VAR_5_]], [[VAR_3_]] : index
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<2x4x4xf32>
// CHECK:             [[CURRENT_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_6_]], [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<4x4x4xf32>
// CHECK:             [[RESULT_:%.+]] = arith.addf [[CURRENT_]], [[LOAD_PARAM_2_MEM_]] : f32
// CHECK:             krnl.store [[RESULT_]], [[RES_]]{{.}}[[VAR_6_]], [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<4x4x4xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<4x4x4xf32>
// CHECK:         }
}
