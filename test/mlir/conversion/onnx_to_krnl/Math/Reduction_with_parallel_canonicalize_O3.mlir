// RUN: onnx-mlir-opt -O3 --march=x86-64 --shape-inference --convert-onnx-to-krnl=enable-parallel --canonicalize %s -split-input-file | FileCheck %s

// With enable-parallel, a krnl.parallel should be created, which takes a loop (to be parallelized) 
// as input. The krnl.parallel should be the last operator before krnl.iterate, since the lowering
// needs to interpret krnl.block, krnl.permute, krnl.unroll first.

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
