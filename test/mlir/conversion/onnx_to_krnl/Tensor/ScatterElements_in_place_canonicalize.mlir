// RUN: onnx-mlir-opt --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Scattering into the buffer of `data` instead of copying it. The cases
// rejecting in place are shared with ScatterND, whose in-place test covers
// them all; this file keeps one.

// `data` is a Relu result used only here: the updates are scattered into its
// buffer and there is no copy.
func.func @test_scatter_elements_in_place(%arg0: tensor<16x8xf32> {onnx.name = "data"}, %arg1: tensor<4x8xf32> {onnx.name = "updates"}) -> (tensor<16x8xf32> {onnx.name = "output"}) {
  %d = "onnx.Relu"(%arg0) : (tensor<16x8xf32>) -> tensor<16x8xf32>
  %idx = onnx.Constant dense<[[11, 3, 12, 7, 5, 11, 1, 1], [10, 13, 6, 15, 2, 10, 7, 12], [3, 6, 4, 0, 14, 15, 8, 7], [5, 5, 7, 10, 13, 14, 9, 8]]> : tensor<4x8xi64>
  %0 = "onnx.ScatterElements"(%d, %idx, %arg1) {axis = 0 : si64} : (tensor<16x8xf32>, tensor<4x8xi64>, tensor<4x8xf32>) -> tensor<16x8xf32>
  return %0 : tensor<16x8xf32>

// CHECK-LABEL:  func.func @test_scatter_elements_in_place
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<16x8xf32> {onnx.name = "data"}, [[PARAM_1_:%.+]]: memref<4x8xf32> {onnx.name = "updates"}) -> (memref<16x8xf32> {onnx.name = "output"}) {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 16 : memref<16x8xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 16, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 8){
// CHECK:             [[VAR_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1] : memref<16x8xf32>
// CHECK:             [[VAR_5_:%.+]] = arith.maxnumf [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK:             krnl.store [[VAR_5_]], [[RES_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1] : memref<16x8xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [4, 8], value = dense<{{.}}[11, 3, 12, 7, 5, 11, 1, 1], [10, 13, 6, 15, 2, 10, 7, 12], [3, 6, 4, 0, 14, 15, 8, 7], [5, 5, 7, 10, 13, 14, 9, 8]{{.}}> : tensor<4x8xi64>}> : () -> memref<4x8xi64>
// CHECK-DAG:       [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 4, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 8){
// CHECK:             [[VAR_3_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_3_1_]]#0, [[VAR_3_1_]]#1] : memref<4x8xf32>
// CHECK-DAG:         [[VAR_5_1_:%.+]] = krnl.load [[VAR_1_]]{{.}}[[VAR_3_1_]]#0, [[VAR_3_1_]]#1] : memref<4x8xi64>
// CHECK:             [[VAR_6_:%.+]] = arith.index_cast [[VAR_5_1_]] : i64 to index
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_1_]], [[RES_]]{{.}}[[VAR_6_]], [[VAR_3_1_]]#1] : memref<16x8xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<16x8xf32>
// CHECK:         }

}

// -----

// The same with a reduction and duplicate indices.
func.func @test_scatter_elements_in_place_add(%arg0: tensor<16x8xf32> {onnx.name = "data"}, %arg1: tensor<4x8xf32> {onnx.name = "updates"}) -> (tensor<16x8xf32> {onnx.name = "output"}) {
  %d = "onnx.Relu"(%arg0) : (tensor<16x8xf32>) -> tensor<16x8xf32>
  %idx = onnx.Constant dense<[[0, 10, 4, 5, 13, 10, 12, 0], [0, 10, 4, 5, 13, 10, 12, 0], [14, 4, 5, 0, 12, 5, 8, 10], [0, 1, 12, 11, 13, 10, 6, 8]]> : tensor<4x8xi64>
  %0 = "onnx.ScatterElements"(%d, %idx, %arg1) {axis = 0 : si64, reduction = "add"} : (tensor<16x8xf32>, tensor<4x8xi64>, tensor<4x8xf32>) -> tensor<16x8xf32>
  return %0 : tensor<16x8xf32>

// CHECK-LABEL:  func.func @test_scatter_elements_in_place_add
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<16x8xf32> {onnx.name = "data"}, [[PARAM_1_:%.+]]: memref<4x8xf32> {onnx.name = "updates"}) -> (memref<16x8xf32> {onnx.name = "output"}) {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 16 : memref<16x8xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 16, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 8){
// CHECK:             [[VAR_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1] : memref<16x8xf32>
// CHECK:             [[VAR_5_:%.+]] = arith.maxnumf [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK:             krnl.store [[VAR_5_]], [[RES_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1] : memref<16x8xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [4, 8], value = dense<{{.}}[0, 10, 4, 5, 13, 10, 12, 0], [0, 10, 4, 5, 13, 10, 12, 0], [14, 4, 5, 0, 12, 5, 8, 10], [0, 1, 12, 11, 13, 10, 6, 8]{{.}}> : tensor<4x8xi64>}> : () -> memref<4x8xi64>
// CHECK-DAG:       [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 4, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 8){
// CHECK:             [[VAR_3_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_3_1_]]#0, [[VAR_3_1_]]#1] : memref<4x8xf32>
// CHECK-DAG:         [[VAR_5_1_:%.+]] = krnl.load [[VAR_1_]]{{.}}[[VAR_3_1_]]#0, [[VAR_3_1_]]#1] : memref<4x8xi64>
// CHECK:             [[VAR_6_:%.+]] = arith.index_cast [[VAR_5_1_]] : i64 to index
// CHECK:             [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_6_]], [[VAR_3_1_]]#1] : memref<16x8xf32>
// CHECK:             [[VAR_8_:%.+]] = arith.addf [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:             krnl.store [[VAR_8_]], [[RES_]]{{.}}[[VAR_6_]], [[VAR_3_1_]]#1] : memref<16x8xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<16x8xf32>
// CHECK:         }

}

// -----

// `data` is a function argument, owned by the caller: copy.
func.func @test_scatter_elements_no_in_place_arg(%arg0: tensor<16x8xf32> {onnx.name = "data"}, %arg1: tensor<4x8xf32> {onnx.name = "updates"}) -> (tensor<16x8xf32> {onnx.name = "output"}) {
  %idx = onnx.Constant dense<[[10, 15, 15, 1, 2, 13, 4, 3], [2, 4, 14, 2, 9, 11, 1, 9], [9, 13, 13, 6, 6, 0, 6, 13], [8, 1, 12, 12, 4, 1, 15, 6]]> : tensor<4x8xi64>
  %0 = "onnx.ScatterElements"(%arg0, %idx, %arg1) {axis = 0 : si64} : (tensor<16x8xf32>, tensor<4x8xi64>, tensor<4x8xf32>) -> tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
// CHECK-LABEL:  func.func @test_scatter_elements_no_in_place_arg
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<16x8xf32> {onnx.name = "data"}, [[PARAM_1_:%.+]]: memref<4x8xf32> {onnx.name = "updates"}) -> (memref<16x8xf32> {onnx.name = "output"}) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_128_:%.+]] = arith.constant 128 : i64
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [4, 8], value = dense<{{.}}[10, 15, 15, 1, 2, 13, 4, 3], [2, 4, 14, 2, 9, 11, 1, 9], [9, 13, 13, 6, 6, 0, 6, 13], [8, 1, 12, 12, 4, 1, 15, 6]{{.}}> : tensor<4x8xi64>}> : () -> memref<4x8xi64>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 16 : memref<16x8xf32>
// CHECK:           "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[CST_128_]], [[CST_0_]], [[CST_0_]]) : (memref<16x8xf32>, memref<16x8xf32>, i64, index, index) -> ()
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 4, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 8){
// CHECK:             [[VAR_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1] : memref<4x8xf32>
// CHECK-DAG:         [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1] : memref<4x8xi64>
// CHECK:             [[VAR_5_:%.+]] = arith.index_cast [[LOAD_VAR_0_MEM_]] : i64 to index
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_]]{{.}}[[VAR_5_]], [[VAR_2_]]#1] : memref<16x8xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<16x8xf32>
// CHECK:         }

}

