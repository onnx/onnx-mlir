// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

// -----

// COM: Lower Softmax opset 11.

func.func private @test_softmax_v11(%arg0 : tensor<10x20x30xf32>) -> tensor<*xf32> {
  %0 = "onnx.SoftmaxV11"(%arg0) {axis=1: si64} : (tensor<10x20x30xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_softmax_v11
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10x20x30xf32>) -> memref<10x20x30xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<10x20x30xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 10){
// CHECK-DAG:         [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_3_:%.+]] = krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[VAR_arg2_:%.+]] = 0 to 20, [[LOOP_1_]]#1 -> [[VAR_arg3_:%.+]] = 0 to 30) iter_args([[VAR_arg4_:%.+]] = [[CST_0_]]) -> (f32){
// CHECK-DAG:           [[VAR_7_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]], [[VAR_7_]]#0, [[VAR_7_]]#1] : memref<10x20x30xf32>
// CHECK:               [[VAR_9_:%.+]] = arith.maxnumf [[VAR_arg4_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:               krnl.yield [[VAR_9_]] : f32
// CHECK:             }
// CHECK:             [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:         [[VAR_5_:%.+]] = krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[VAR_arg2_1_:%.+]] = 0 to 20, [[LOOP_2_]]#1 -> [[VAR_arg3_1_:%.+]] = 0 to 30) iter_args([[VAR_arg4_1_:%.+]] = [[CST_0_dot_000000_]]) -> (f32){
// CHECK-DAG:           [[VAR_7_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]], [[VAR_7_1_]]#0, [[VAR_7_1_]]#1] : memref<10x20x30xf32>
// CHECK:               [[VAR_9_1_:%.+]] = arith.subf [[LOAD_PARAM_0_MEM_1_]], [[VAR_3_]] : f32
// CHECK:               [[VAR_10_:%.+]] = math.exp [[VAR_9_1_]] : f32
// CHECK:               [[VAR_11_:%.+]] = arith.addf [[VAR_arg4_1_]], [[VAR_10_]] : f32
// CHECK:               krnl.store [[VAR_10_]], [[RES_]]{{.}}[[VAR_1_]], [[VAR_7_1_]]#0, [[VAR_7_1_]]#1] : memref<10x20x30xf32>
// CHECK:               krnl.yield [[VAR_11_]] : f32
// CHECK:             }
// CHECK:             [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_1_:%.+]] = 0 to 20, [[LOOP_3_]]#1 -> [[I_2_:%.+]] = 0 to 30){
// CHECK:               [[VAR_7_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_1_]], [[VAR_7_2_]]#0, [[VAR_7_2_]]#1] : memref<10x20x30xf32>
// CHECK:               [[VAR_9_2_:%.+]] = arith.divf [[LOAD_PARAM_0_MEM_1_]], [[VAR_5_]] : f32
// CHECK:               krnl.store [[VAR_9_2_]], [[RES_]]{{.}}[[VAR_1_]], [[VAR_7_2_]]#0, [[VAR_7_2_]]#1] : memref<10x20x30xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<10x20x30xf32>
// CHECK:         }
}

// -----

// COM: Lower Softmax opset 13.

func.func private @test_softmax_v13(%arg0 : tensor<10x20x30xf32>) -> tensor<*xf32> {
  %0 = "onnx.Softmax"(%arg0) {axis=1: si64} : (tensor<10x20x30xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_softmax_v13
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10x20x30xf32>) -> memref<10x20x30xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<10x20x30xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 10, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 30){
// CHECK-DAG:         [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_3_:%.+]] = krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[VAR_arg3_:%.+]] = 0 to 20) iter_args([[VAR_arg4_:%.+]] = [[CST_0_]]) -> (f32){
// CHECK-DAG:           [[VAR_7_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_7_]], [[VAR_1_]]#1] : memref<10x20x30xf32>
// CHECK:               [[VAR_9_:%.+]] = arith.maxnumf [[VAR_arg4_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:               krnl.yield [[VAR_9_]] : f32
// CHECK:             }
// CHECK:             [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK-DAG:         [[VAR_5_:%.+]] = krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[VAR_arg3_1_:%.+]] = 0 to 20) iter_args([[VAR_arg4_1_:%.+]] = [[CST_0_dot_000000_]]) -> (f32){
// CHECK-DAG:           [[VAR_7_1_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_7_1_]], [[VAR_1_]]#1] : memref<10x20x30xf32>
// CHECK:               [[VAR_9_1_:%.+]] = arith.subf [[LOAD_PARAM_0_MEM_1_]], [[VAR_3_]] : f32
// CHECK:               [[VAR_10_:%.+]] = math.exp [[VAR_9_1_]] : f32
// CHECK:               [[VAR_11_:%.+]] = arith.addf [[VAR_arg4_1_]], [[VAR_10_]] : f32
// CHECK:               krnl.store [[VAR_10_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_7_1_]], [[VAR_1_]]#1] : memref<10x20x30xf32>
// CHECK:               krnl.yield [[VAR_11_]] : f32
// CHECK:             }
// CHECK:             [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[I_2_:%.+]] = 0 to 20){
// CHECK:               [[VAR_7_2_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_7_2_]], [[VAR_1_]]#1] : memref<10x20x30xf32>
// CHECK:               [[VAR_9_2_:%.+]] = arith.divf [[LOAD_PARAM_0_MEM_1_]], [[VAR_5_]] : f32
// CHECK:               krnl.store [[VAR_9_2_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_7_2_]], [[VAR_1_]]#1] : memref<10x20x30xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<10x20x30xf32>
// CHECK:         }
}

