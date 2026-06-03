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
// CHECK:             [[VAR_8_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_8_]]#0, [[VAR_8_]]#1] : memref<10x20xf32>
// CHECK:             [[VAR_10_:%.+]] = math.absf [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_10_]], [[RES_]]{{.}}[[VAR_8_]]#0, [[VAR_8_]]#1] : memref<10x20xf32>
// CHECK:           }
// CHECK:           [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<10x1xf32>
// CHECK:           krnl.memset [[RES_1_]], [[CST_0_dot_000000_]] : memref<10x1xf32>
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 10, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 20){
// CHECK:             [[VAR_8_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_8_1_]]#0, [[VAR_8_1_]]#1] : memref<10x20xf32>
// CHECK-DAG:         [[VAR_10_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_8_1_]]#0, [[CST_0_]]{{.}} : memref<10x1xf32>
// CHECK:             [[VAR_11_:%.+]] = arith.addf [[VAR_10_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:             krnl.store [[VAR_11_]], [[RES_1_]]{{.}}[[VAR_8_1_]]#0, [[CST_0_]]{{.}} : memref<10x1xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_2_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [], value = dense<0.000000e+00> : tensor<f32>}> : () -> memref<f32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [], value = dense<1.000000e+00> : tensor<f32>}> : () -> memref<f32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<10x1xi1>
// CHECK-DAG:       [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_4_:%.+]] = 0 to 10, [[LOOP_2_]]#1 -> [[I_5_:%.+]] = 0 to 1){
// CHECK:             [[VAR_8_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_8_2_]]#0, [[CST_0_]]{{.}} : memref<10x1xf32>
// CHECK-DAG:         [[VAR_10_1_:%.+]] = krnl.load [[VAR_2_]][] : memref<f32>
// CHECK:             [[VAR_11_1_:%.+]] = arith.cmpf oeq, [[LOAD_PARAM_0_MEM_1_]], [[VAR_10_1_]] : f32
// CHECK:             krnl.store [[VAR_11_1_]], [[RES_2_]]{{.}}[[VAR_8_2_]]#0, [[VAR_8_2_]]#1] : memref<10x1xi1>
// CHECK:           }
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<10x20xf32>
// CHECK-DAG:       [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_6_:%.+]] = 0 to 10, [[LOOP_3_]]#1 -> [[I_7_:%.+]] = 0 to 20){
// CHECK:             [[VAR_8_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_8_3_]]#0, [[CST_0_]]{{.}} : memref<10x1xi1>
// CHECK-DAG:         [[VAR_10_1_1_:%.+]] = krnl.load [[VAR_2_]][] : memref<f32>
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_8_3_]]#0, [[VAR_8_3_]]#1] : memref<10x20xf32>
// CHECK:             [[VAR_12_:%.+]] = arith.select [[LOAD_PARAM_0_MEM_1_1_]], [[VAR_10_1_1_]], [[LOAD_PARAM_0_MEM_2_]] : f32
// CHECK:             krnl.store [[VAR_12_]], [[RES_3_]]{{.}}[[VAR_8_3_]]#0, [[VAR_8_3_]]#1] : memref<10x20xf32>
// CHECK:           }
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<10x1xf32>
// CHECK-DAG:       [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_8_:%.+]] = 0 to 10, [[LOOP_4_]]#1 -> [[I_9_:%.+]] = 0 to 1){
// CHECK:             [[VAR_8_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_8_4_]]#0, [[CST_0_]]{{.}} : memref<10x1xi1>
// CHECK-DAG:         [[VAR_10_1_1_:%.+]] = krnl.load [[VAR_3_]][] : memref<f32>
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_8_4_]]#0, [[CST_0_]]{{.}} : memref<10x1xf32>
// CHECK:             [[VAR_12_1_:%.+]] = arith.select [[LOAD_PARAM_0_MEM_1_1_]], [[VAR_10_1_1_]], [[LOAD_PARAM_0_MEM_2_]] : f32
// CHECK:             krnl.store [[VAR_12_1_]], [[RES_4_]]{{.}}[[VAR_8_4_]]#0, [[VAR_8_4_]]#1] : memref<10x1xf32>
// CHECK:           }
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<10x20xf32>
// CHECK-DAG:       [[LOOP_5_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_5_]]#0, [[LOOP_5_]]#1) with ([[LOOP_5_]]#0 -> [[I_10_:%.+]] = 0 to 10, [[LOOP_5_]]#1 -> [[I_11_:%.+]] = 0 to 20){
// CHECK:             [[VAR_8_5_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_5_]]#0, [[LOOP_5_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_1_1_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_8_5_]]#0, [[VAR_8_5_]]#1] : memref<10x20xf32>
// CHECK-DAG:         [[VAR_10_1_1_1_:%.+]] = krnl.load [[RES_4_]]{{.}}[[VAR_8_5_]]#0, [[CST_0_]]{{.}} : memref<10x1xf32>
// CHECK:             [[VAR_11_2_:%.+]] = arith.divf [[LOAD_PARAM_0_MEM_1_1_1_]], [[VAR_10_1_1_1_]] : f32
// CHECK:             krnl.store [[VAR_11_2_]], [[RES_5_]]{{.}}[[VAR_8_5_]]#0, [[VAR_8_5_]]#1] : memref<10x20xf32>
// CHECK:           }
// CHECK:           return [[RES_5_]] : memref<10x20xf32>
// CHECK:         }

}

