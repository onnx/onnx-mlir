// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

// COM: Test GatherND with indices_shape[-1] == rank(data) - batch_dims
func.func @test_gather_nd_1(%arg0 : tensor<2x2xf32>, %arg1 : tensor<2x2xi64>) -> tensor<2xf32> {
  %0 = "onnx.GatherND"(%arg0, %arg1) {batch_dims = 0 : si64} : (tensor<2x2xf32>, tensor<2x2xi64>) -> tensor<2xf32>
  "func.return"(%0) : (tensor<2xf32>) -> ()
// CHECK-LABEL:  func.func @test_gather_nd_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x2xf32>, [[PARAM_1_:%.+]]: memref<2x2xi64>) -> memref<2xf32> {
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_2_1_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_2_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_2_3_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_2_4_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_2_5_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_2_6_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_2_7_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_8_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: [0], sizes: [1, 2, 2], strides: [4, 2, 1] : memref<2x2xi64> to memref<1x2x2xi64>
// CHECK-DAG:       [[CST_1_2_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_4_1_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_reinterpret_cast_11_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [1, 2, 2], strides: [4, 2, 1] : memref<2x2xf32> to memref<1x2x2xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<2xf32>
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_3_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_]], [[RES_1_]][] : memref<index>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK-DAG:         [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_:%.+]] = krnl.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[CST_0_2_]]{{.}} : memref<1x2x2xi64>
// CHECK-DAG:         [[VAR_3_:%.+]] = arith.index_cast [[LOAD_VAR_reinterpret_cast_MEM_]] : i64 to index
// CHECK-DAG:         [[CST_1_4_:%.+]] = arith.constant 1 : index
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_1_:%.+]] = krnl.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[CST_1_4_]]{{.}} : memref<1x2x2xi64>
// CHECK:             [[VAR_5_:%.+]] = arith.index_cast [[LOAD_VAR_reinterpret_cast_MEM_1_]] : i64 to index
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_11_MEM_:%.+]] = krnl.load [[VAR_reinterpret_cast_11_]]{{.}}[[VAR_1_]]#0, [[VAR_3_]], [[VAR_5_]]{{.}} : memref<1x2x2xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:             krnl.store [[LOAD_VAR_reinterpret_cast_11_MEM_]], [[RES_]]{{.}}[[LOAD_RES_1_MEM_]]{{.}} : memref<2xf32>
// CHECK:             [[VAR_8_:%.+]] = arith.addi [[LOAD_RES_1_MEM_]], [[CST_1_3_]] : index
// CHECK:             krnl.store [[VAR_8_]], [[RES_1_]][] : memref<index>
// CHECK:           }
// CHECK-DAG:       [[CST_1_5_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_reinterpret_cast_15_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2], strides: [1] : memref<2xf32> to memref<2xf32>
// CHECK:           return [[VAR_reinterpret_cast_15_]] : memref<2xf32>
// CHECK:         }
}

// -----

// COM: Test GatherND with indices_shape[-1] < rank(data) - batch_dims
func.func @test_gather_nd_2(%arg0 : tensor<2x2x2xf32>, %arg1 : tensor<2x1x2xi64>) -> tensor<2x1x2xf32> {
  %0 = "onnx.GatherND"(%arg0, %arg1) {batch_dims = 0 : si64} : (tensor<2x2x2xf32>, tensor<2x1x2xi64>) -> tensor<2x1x2xf32>
  "func.return"(%0) : (tensor<2x1x2xf32>) -> ()
// CHECK-LABEL:  func.func @test_gather_nd_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x2x2xf32>, [[PARAM_1_:%.+]]: memref<2x1x2xi64>) -> memref<2x1x2xf32> {
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_2_1_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_2_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_2_3_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_4_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_2_5_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_2_6_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_2_7_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_2_8_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_9_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_2_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_10_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_3_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: [0], sizes: [1, 2, 2], strides: [4, 2, 1] : memref<2x1x2xi64> to memref<1x2x2xi64>
// CHECK-DAG:       [[CST_1_4_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_4_1_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[VAR_reinterpret_cast_15_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [1, 2, 2, 2], strides: [8, 4, 2, 1] : memref<2x2x2xf32> to memref<1x2x2x2xf32>
// CHECK-DAG:       [[CST_4_2_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<4xf32>
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_5_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_]], [[RES_1_]][] : memref<index>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK-DAG:         [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_:%.+]] = krnl.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[CST_0_2_]]{{.}} : memref<1x2x2xi64>
// CHECK-DAG:         [[VAR_3_:%.+]] = arith.index_cast [[LOAD_VAR_reinterpret_cast_MEM_]] : i64 to index
// CHECK-DAG:         [[CST_1_6_:%.+]] = arith.constant 1 : index
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_1_:%.+]] = krnl.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[CST_1_6_]]{{.}} : memref<1x2x2xi64>
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.index_cast [[LOAD_VAR_reinterpret_cast_MEM_1_]] : i64 to index
// CHECK-DAG:         [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = [[CST_0_3_]] to [[CST_2_7_]]){
// CHECK:               [[VAR_7_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_reinterpret_cast_15_MEM_:%.+]] = krnl.load [[VAR_reinterpret_cast_15_]]{{.}}[[VAR_1_]]#0, [[VAR_3_]], [[VAR_5_]], [[VAR_7_]]{{.}} : memref<1x2x2x2xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:               krnl.store [[LOAD_VAR_reinterpret_cast_15_MEM_]], [[RES_]]{{.}}[[LOAD_RES_1_MEM_]]{{.}} : memref<4xf32>
// CHECK:               [[VAR_10_:%.+]] = arith.addi [[LOAD_RES_1_MEM_]], [[CST_1_5_]] : index
// CHECK:               krnl.store [[VAR_10_]], [[RES_1_]][] : memref<index>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[CST_1_7_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_reinterpret_cast_20_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 1, 2], strides: [2, 2, 1] : memref<4xf32> to memref<2x1x2xf32>
// CHECK:           return [[VAR_reinterpret_cast_20_]] : memref<2x1x2xf32>
// CHECK:         }
}

// -----

// COM: Test GatherND with dynamic shape
func.func @test_gather_nd_with_dynamic_shape_int(%arg0 : tensor<2x2xi32>, %arg1 : tensor<?x2xi64>) -> tensor<?xi32> {
  %0 = "onnx.GatherND"(%arg0, %arg1) {batch_dims = 0 : si64} : (tensor<2x2xi32>, tensor<?x2xi64>) -> tensor<?xi32>
  "func.return"(%0) : (tensor<?xi32>) -> ()
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @test_gather_nd_with_dynamic_shape_int
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x2xi32>, [[PARAM_1_:%.+]]: memref<?x2xi64>) -> memref<?xi32> {
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_2_1_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x2xi64>
// CHECK-DAG:       [[CST_2_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_2_3_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_2_4_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_5_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_1_]] : memref<?x2xi64>
// CHECK-DAG:       [[CST_2_5_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_6_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]]([[VAR_dim_5_]])
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: [0], sizes: [1, [[VAR_dim_5_]], 2], strides: {{.}}[[VAR_0_]], 2, 1] : memref<?x2xi64> to memref<1x?x2xi64>
// CHECK-DAG:       [[CST_1_2_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_reinterpret_cast_10_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [1, 2, 2], strides: [4, 2, 1] : memref<2x2xi32> to memref<1x2x2xi32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) : memref<?xi32>
// CHECK-DAG:       [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_3_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_2_]], [[RES_1_]][] : memref<index>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_5_]])){
// CHECK-DAG:         [[VAR_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[CST_0_4_:%.+]] = arith.constant 0 : index
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_:%.+]] = krnl.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[CST_0_4_]]{{.}} : memref<1x?x2xi64>
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.index_cast [[LOAD_VAR_reinterpret_cast_MEM_]] : i64 to index
// CHECK-DAG:         [[CST_1_4_:%.+]] = arith.constant 1 : index
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_1_:%.+]] = krnl.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[CST_1_4_]]{{.}} : memref<1x?x2xi64>
// CHECK:             [[VAR_6_:%.+]] = arith.index_cast [[LOAD_VAR_reinterpret_cast_MEM_1_]] : i64 to index
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_10_MEM_:%.+]] = krnl.load [[VAR_reinterpret_cast_10_]]{{.}}[[VAR_2_]]#0, [[VAR_4_]], [[VAR_6_]]{{.}} : memref<1x2x2xi32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:             krnl.store [[LOAD_VAR_reinterpret_cast_10_MEM_]], [[RES_]]{{.}}[[LOAD_RES_1_MEM_]]{{.}} : memref<?xi32>
// CHECK:             [[VAR_9_:%.+]] = arith.addi [[LOAD_RES_1_MEM_]], [[CST_1_3_]] : index
// CHECK:             krnl.store [[VAR_9_]], [[RES_1_]][] : memref<index>
// CHECK:           }
// CHECK-DAG:       [[CST_1_5_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_reinterpret_cast_15_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: {{.}}[[VAR_dim_]]{{.}}, strides: [1] : memref<?xi32> to memref<?xi32>
// CHECK:           return [[VAR_reinterpret_cast_15_]] : memref<?xi32>
// CHECK:         }
}

