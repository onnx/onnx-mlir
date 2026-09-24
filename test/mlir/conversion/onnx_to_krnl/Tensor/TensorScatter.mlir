// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

func.func @test_tensor_scatter_linear(%arg0: tensor<2x4x8x16xf32>, %arg1: tensor<2x4x3x16xf32>, %arg2: tensor<2xi64>) -> tensor<2x4x8x16xf32> {
  %0 = "onnx.TensorScatter"(%arg0, %arg1, %arg2) {axis = 2 : si64, mode = "linear"} : (tensor<2x4x8x16xf32>, tensor<2x4x3x16xf32>, tensor<2xi64>) -> tensor<2x4x8x16xf32>
  return %0 : tensor<2x4x8x16xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_tensor_scatter_linear
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4x8x16xf32>, [[PARAM_1_:%.+]]: memref<2x4x3x16xf32>, [[PARAM_2_:%.+]]: memref<2xi64>) -> memref<2x4x8x16xf32> {
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_16_:%.+]] = arith.constant 16 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x4x8x16xf32>
// CHECK-DAG:       [[CST_1024_:%.+]] = arith.constant 1024 : i64
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[CST_1024_]], [[CST_0_]], [[CST_0_]]) : (memref<2x4x8x16xf32>, memref<2x4x8x16xf32>, i64, index, index) -> ()
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_2_1_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_4_1_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_16_1_:%.+]] = arith.constant 16 : index
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 3, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 16){
// CHECK:             [[VAR_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0] : memref<2xi64>
// CHECK:             [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PARAM_2_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.addi [[VAR_3_]], [[VAR_1_]]#2 : index
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]#3] : memref<2x4x3x16xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]], [[VAR_1_]]#3] : memref<2x4x8x16xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x4x8x16xf32>
// CHECK:         }
}

// -----

// COM: Test TensorScatter with 'mode' = "circular": the write position wraps
// around modulo the cache's 'axis' dimension.
func.func @test_tensor_scatter_circular(%arg0: tensor<2x4x8x16xf32>, %arg1: tensor<2x4x3x16xf32>, %arg2: tensor<2xi64>) -> tensor<2x4x8x16xf32> {
  %0 = "onnx.TensorScatter"(%arg0, %arg1, %arg2) {axis = 2 : si64, mode = "circular"} : (tensor<2x4x8x16xf32>, tensor<2x4x3x16xf32>, tensor<2xi64>) -> tensor<2x4x8x16xf32>
  return %0 : tensor<2x4x8x16xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_tensor_scatter_circular
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4x8x16xf32>, [[PARAM_1_:%.+]]: memref<2x4x3x16xf32>, [[PARAM_2_:%.+]]: memref<2xi64>) -> memref<2x4x8x16xf32> {
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_16_:%.+]] = arith.constant 16 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x4x8x16xf32>
// CHECK-DAG:       [[CST_1024_:%.+]] = arith.constant 1024 : i64
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[CST_1024_]], [[CST_0_]], [[CST_0_]]) : (memref<2x4x8x16xf32>, memref<2x4x8x16xf32>, i64, index, index) -> ()
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_2_1_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_4_1_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_16_1_:%.+]] = arith.constant 16 : index
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 3, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 16){
// CHECK:             [[VAR_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0] : memref<2xi64>
// CHECK:             [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PARAM_2_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.addi [[VAR_3_]], [[VAR_1_]]#2 : index
// CHECK-DAG:         [[CST_8_1_:%.+]] = arith.constant 8 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.remsi [[VAR_4_]], [[CST_8_1_]] : index
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]#3] : memref<2x4x3x16xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_5_]], [[VAR_1_]]#3] : memref<2x4x8x16xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x4x8x16xf32>
// CHECK:         }
}

// -----

// COM: Test TensorScatter without 'write_indices': the write offset defaults
// to zero for every batch element, so 'update' is scattered starting at
// index 0 along 'axis'.
func.func @test_tensor_scatter_no_write_indices(%arg0: tensor<2x4x8x16xf32>, %arg1: tensor<2x4x3x16xf32>) -> tensor<2x4x8x16xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.TensorScatter"(%arg0, %arg1, %none) {axis = 2 : si64} : (tensor<2x4x8x16xf32>, tensor<2x4x3x16xf32>, none) -> tensor<2x4x8x16xf32>
  return %0 : tensor<2x4x8x16xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_tensor_scatter_no_write_indices
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4x8x16xf32>, [[PARAM_1_:%.+]]: memref<2x4x3x16xf32>) -> memref<2x4x8x16xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() <{value}> : () -> none
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_16_:%.+]] = arith.constant 16 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x4x8x16xf32>
// CHECK-DAG:       [[CST_1024_:%.+]] = arith.constant 1024 : i64
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[CST_1024_]], [[CST_0_]], [[CST_0_]]) : (memref<2x4x8x16xf32>, memref<2x4x8x16xf32>, i64, index, index) -> ()
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_2_1_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_4_1_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_16_1_:%.+]] = arith.constant 16 : index
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 3, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 16){
// CHECK:             [[VAR_2_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_2_]]#3] : memref<2x4x3x16xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_2_]]#3] : memref<2x4x8x16xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x4x8x16xf32>
// CHECK:         }
}

// -----

// COM: Test TensorScatter with a negative 'axis', which must be normalized to
// a positive dimension index (here, -2 normalizes to 2 for a rank-4 tensor,
// same as the "linear" test case above).
func.func @test_tensor_scatter_negative_axis(%arg0: tensor<2x4x8x16xf32>, %arg1: tensor<2x4x3x16xf32>, %arg2: tensor<2xi64>) -> tensor<2x4x8x16xf32> {
  %0 = "onnx.TensorScatter"(%arg0, %arg1, %arg2) {axis = -2 : si64, mode = "linear"} : (tensor<2x4x8x16xf32>, tensor<2x4x3x16xf32>, tensor<2xi64>) -> tensor<2x4x8x16xf32>
  return %0 : tensor<2x4x8x16xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_tensor_scatter_negative_axis
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4x8x16xf32>, [[PARAM_1_:%.+]]: memref<2x4x3x16xf32>, [[PARAM_2_:%.+]]: memref<2xi64>) -> memref<2x4x8x16xf32> {
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_16_:%.+]] = arith.constant 16 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x4x8x16xf32>
// CHECK-DAG:       [[CST_1024_:%.+]] = arith.constant 1024 : i64
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[CST_1024_]], [[CST_0_]], [[CST_0_]]) : (memref<2x4x8x16xf32>, memref<2x4x8x16xf32>, i64, index, index) -> ()
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_2_1_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_4_1_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_16_1_:%.+]] = arith.constant 16 : index
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 3, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 16){
// CHECK:             [[VAR_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0] : memref<2xi64>
// CHECK:             [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PARAM_2_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.addi [[VAR_3_]], [[VAR_1_]]#2 : index
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]#3] : memref<2x4x3x16xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_4_]], [[VAR_1_]]#3] : memref<2x4x8x16xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x4x8x16xf32>
// CHECK:         }
}
