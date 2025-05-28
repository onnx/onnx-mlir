// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

func.func private @test_lpNormalization(%arg0 : tensor<10x20xf32>) -> tensor<*xf32> {
  %0 = "onnx.LpNormalization"(%arg0) {axis = 1 : si64, p = 1 : si64} : (tensor<10x20xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_lpNormalization
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10x20xf32>) -> memref<10x20xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<10x20xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 10, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 20){
// CHECK:             [[VAR_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1] : memref<10x20xf32>
// CHECK:             [[VAR_5_:%.+]] = math.absf [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_5_]], [[RES_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1] : memref<10x20xf32>
// CHECK:           }
// CHECK:           [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<10x1xf32>
// CHECK:           krnl.memset [[RES_1_]], [[CST_0_dot_000000_]] : memref<10x1xf32>
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 10, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 20){
// CHECK:             [[VAR_3_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_3_1_]]#0, [[VAR_3_1_]]#1] : memref<10x20xf32>
// CHECK-DAG:         [[VAR_5_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_3_1_]]#0, [[CST_0_]]{{.}} : memref<10x1xf32>
// CHECK:             [[VAR_6_:%.+]] = arith.addf [[VAR_5_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:             krnl.store [[VAR_6_]], [[RES_1_]]{{.}}[[VAR_3_1_]]#0, [[CST_0_]]{{.}} : memref<10x1xf32>
// CHECK:           }
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<10x20xf32>
// CHECK-DAG:       [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_4_:%.+]] = 0 to 10, [[LOOP_2_]]#1 -> [[I_5_:%.+]] = 0 to 20){
// CHECK:             [[VAR_3_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_2_]]#0, [[VAR_3_2_]]#1] : memref<10x20xf32>
// CHECK-DAG:         [[VAR_5_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_3_2_]]#0, [[CST_0_]]{{.}} : memref<10x1xf32>
// CHECK:             [[VAR_6_1_:%.+]] = arith.divf [[LOAD_PARAM_0_MEM_2_]], [[VAR_5_1_]] : f32
// CHECK:             krnl.store [[VAR_6_1_]], [[RES_2_]]{{.}}[[VAR_3_2_]]#0, [[VAR_3_2_]]#1] : memref<10x20xf32>
// CHECK:           }
// CHECK:           return [[PARAM_0_]] : memref<10x20xf32>
// CHECK:         }
}