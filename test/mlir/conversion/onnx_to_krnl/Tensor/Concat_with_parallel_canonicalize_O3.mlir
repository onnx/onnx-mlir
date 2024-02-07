// RUN: onnx-mlir-opt -O3 --march=x86-64 --shape-inference --convert-onnx-to-krnl=enable-parallel --canonicalize %s -split-input-file | FileCheck %s

// With enable-parallel, a krnl.parallel should be created, which takes a loop (to be parallelized) 
// as input. The krnl.parallel should be the last operator before krnl.iterate, since the lowering
// needs to interpret krnl.block, krnl.permute, krnl.unroll first.

// -----

// Test parallelization of Concat
func.func @test_concat_parallel(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<5x5x3x32xf32>, %arg2 : tensor<5x5x5x32xf32>) -> tensor<5x5x9x32xf32> {
  %1 = "onnx.Concat"(%arg0, %arg1, %arg2) { axis = 2 : si64} : (tensor<5x5x1x32xf32>, tensor<5x5x3x32xf32>, tensor<5x5x5x32xf32>)  -> tensor<5x5x9x32xf32>
  "func.return"(%1) : (tensor<5x5x9x32xf32>) -> ()
  // mlir2FileCheck.py
  // CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 1)>
  // CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 4)>
  // CHECK-LABEL:  func.func @test_concat_parallel
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<5x5x1x32xf32>, [[PARAM_1_:%.+]]: memref<5x5x3x32xf32>, [[PARAM_2_:%.+]]: memref<5x5x5x32xf32>) -> memref<5x5x9x32xf32> {
  // CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<5x5x9x32xf32>
  // CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
  // CHECK:           krnl.parallel([[LOOP_0_]]#0) : !krnl.loop
  // CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 5, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 5, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 1, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 32){
  // CHECK:             [[VAR_3_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
  // CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_3_]]#2, [[VAR_3_]]#3] : memref<5x5x1x32xf32>
  // CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_3_]]#2, [[VAR_3_]]#3] : memref<5x5x9x32xf32>
  // CHECK:           }
  // CHECK:           [[LOOP_1_:%.+]]:4 = krnl.define_loops 4
  // CHECK:           krnl.parallel([[LOOP_1_]]#0) : !krnl.loop
  // CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2, [[LOOP_1_]]#3) with ([[LOOP_1_]]#0 -> [[I_4_:%.+]] = 0 to 5, [[LOOP_1_]]#1 -> [[I_5_:%.+]] = 0 to 5, [[LOOP_1_]]#2 -> [[I_6_:%.+]] = 0 to 3, [[LOOP_1_]]#3 -> [[I_7_:%.+]] = 0 to 32){
  // CHECK:             [[VAR_3_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2, [[LOOP_1_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
  // CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = affine.apply [[MAP_0_]]([[VAR_3_1_]]#2)
  // CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_3_1_]]#0, [[VAR_3_1_]]#1, [[VAR_3_1_]]#2, [[VAR_3_1_]]#3] : memref<5x5x3x32xf32>
  // CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_]]{{.}}[[VAR_3_1_]]#0, [[VAR_3_1_]]#1, [[LOAD_PARAM_0_MEM_1_]], [[VAR_3_1_]]#3] : memref<5x5x9x32xf32>
  // CHECK:           }
  // CHECK:           [[LOOP_2_:%.+]]:4 = krnl.define_loops 4
  // CHECK:           krnl.parallel([[LOOP_2_]]#0) : !krnl.loop
  // CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2, [[LOOP_2_]]#3) with ([[LOOP_2_]]#0 -> [[I_8_:%.+]] = 0 to 5, [[LOOP_2_]]#1 -> [[I_9_:%.+]] = 0 to 5, [[LOOP_2_]]#2 -> [[I_10_:%.+]] = 0 to 5, [[LOOP_2_]]#3 -> [[I_11_:%.+]] = 0 to 32){
  // CHECK:             [[VAR_3_2_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2, [[LOOP_2_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
  // CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = affine.apply [[MAP_1_]]([[VAR_3_2_]]#2)
  // CHECK-DAG:         [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_3_2_]]#0, [[VAR_3_2_]]#1, [[VAR_3_2_]]#2, [[VAR_3_2_]]#3] : memref<5x5x5x32xf32>
  // CHECK:             krnl.store [[LOAD_PARAM_1_MEM_1_]], [[RES_]]{{.}}[[VAR_3_2_]]#0, [[VAR_3_2_]]#1, [[LOAD_PARAM_0_MEM_1_]], [[VAR_3_2_]]#3] : memref<5x5x9x32xf32>
  // CHECK:           }
  // CHECK:           return [[RES_]] : memref<5x5x9x32xf32>
  // CHECK:         }
}
