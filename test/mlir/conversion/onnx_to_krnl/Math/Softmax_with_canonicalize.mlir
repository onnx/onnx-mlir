// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

// COM: Lower Softmax opset 11.
func.func private @test_softmax_v11(%arg0 : tensor<10x20x30xf32>) -> tensor<*xf32> {
  %0 = "onnx.SoftmaxV11"(%arg0) {axis=1: si64} : (tensor<10x20x30xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK:         func private @test_softmax_v11([[arg0_:%.+]]: memref<10x20x30xf32>) -> memref<10x20x30xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_2_:%.+]] = memref.alloc() {{.*}}: memref<10x20x30xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.alloc() {{.*}}: memref<f32>
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.alloc() {{.*}}: memref<f32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 10){
// CHECK:             [[VAR_4_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[VAR_1_]][] : memref<f32>
// CHECK:             krnl.store [[CST_0_]], [[VAR_0_]][] : memref<f32>
// CHECK:             [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_1_:%.+]] = 0 to 20, [[LOOP_1_]]#1 -> [[I_2_:%.+]] = 0 to 30){
// CHECK-DAG:           [[VAR_10_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]][] : memref<f32>
// CHECK:               [[LOAD_arg0_MEM_:%.+]] = krnl.load [[arg0_]]{{.}}[[VAR_4_]], [[VAR_10_]]#0, [[VAR_10_]]#1] : memref<10x20x30xf32>
// CHECK:               [[VAR_13_:%.+]] = arith.cmpf ogt, [[LOAD_VAR_0_MEM_]], [[LOAD_arg0_MEM_]] : f32
// CHECK:               [[VAR_14_:%.+]] = arith.select [[VAR_13_]], [[LOAD_VAR_0_MEM_]], [[LOAD_arg0_MEM_]] : f32
// CHECK:               krnl.store [[VAR_14_]], [[VAR_0_]][] : memref<f32>
// CHECK:             }
// CHECK-DAG:         [[LOAD_VAR_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]][] : memref<f32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 20, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 30){
// CHECK-DAG:           [[VAR_10_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_VAR_0_MEM_2_:%.+]] = krnl.load [[VAR_1_]][] : memref<f32>
// CHECK:               [[LOAD_arg0_MEM_1_:%.+]] = krnl.load [[arg0_]]{{.}}[[VAR_4_]], [[VAR_10_1_]]#0, [[VAR_10_1_]]#1] : memref<10x20x30xf32>
// CHECK:               [[VAR_13_1_:%.+]] = arith.subf [[LOAD_arg0_MEM_1_]], [[LOAD_VAR_0_MEM_1_]] : f32
// CHECK:               [[VAR_14_1_:%.+]] = math.exp [[VAR_13_1_]] : f32
// CHECK:               [[VAR_15_:%.+]] = arith.addf [[LOAD_VAR_0_MEM_2_]], [[VAR_14_1_]] : f32
// CHECK:               krnl.store [[VAR_15_]], [[VAR_1_]][] : memref<f32>
// CHECK:               krnl.store [[VAR_14_1_]], [[VAR_2_]]{{.}}[[VAR_4_]], [[VAR_10_1_]]#0, [[VAR_10_1_]]#1] : memref<10x20x30xf32>
// CHECK:             }
// CHECK-DAG:         [[LOAD_VAR_1_MEM_:%.+]] = krnl.load [[VAR_1_]][] : memref<f32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to 20, [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to 30){
// CHECK:               [[VAR_10_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_VAR_0_MEM_2_:%.+]] = krnl.load [[VAR_2_]]{{.}}[[VAR_4_]], [[VAR_10_2_]]#0, [[VAR_10_2_]]#1] : memref<10x20x30xf32>
// CHECK:               [[LOAD_arg0_MEM_1_:%.+]] = arith.divf [[LOAD_VAR_0_MEM_2_]], [[LOAD_VAR_1_MEM_]] : f32
// CHECK:               krnl.store [[LOAD_arg0_MEM_1_]], [[VAR_2_]]{{.}}[[VAR_4_]], [[VAR_10_2_]]#0, [[VAR_10_2_]]#1] : memref<10x20x30xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_2_]] : memref<10x20x30xf32>
// CHECK:         }
}

// -----

// COM: Lower Softmax opset 13.

func.func private @test_softmax_v13(%arg0 : tensor<10x20x30xf32>) -> tensor<*xf32> {
  %0 = "onnx.Softmax"(%arg0) {axis=1: si64} : (tensor<10x20x30xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK:         func private @test_softmax_v13([[arg0_:%.+]]: memref<10x20x30xf32>) -> memref<10x20x30xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_2_:%.+]] = memref.alloc() {{.*}}: memref<10x20x30xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.alloc() {{.*}}: memref<f32>
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.alloc() {{.*}}: memref<f32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 10, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 30){
// CHECK:             [[VAR_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[VAR_1_]][] : memref<f32>
// CHECK:             krnl.store [[CST_0_]], [[VAR_0_]][] : memref<f32>
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 20){
// CHECK-DAG:           [[VAR_10_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]][] : memref<f32>
// CHECK:               [[LOAD_arg0_MEM_:%.+]] = krnl.load [[arg0_]]{{.}}[[VAR_4_]]#0, [[VAR_10_]], [[VAR_4_]]#1] : memref<10x20x30xf32>
// CHECK:               [[VAR_13_:%.+]] = arith.cmpf ogt, [[LOAD_VAR_0_MEM_]], [[LOAD_arg0_MEM_]] : f32
// CHECK:               [[VAR_14_:%.+]] = arith.select [[VAR_13_]], [[LOAD_VAR_0_MEM_]], [[LOAD_arg0_MEM_]] : f32
// CHECK:               krnl.store [[VAR_14_]], [[VAR_0_]][] : memref<f32>
// CHECK:             }
// CHECK-DAG:         [[LOAD_VAR_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]][] : memref<f32>
// CHECK-DAG:         [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_3_:%.+]] = 0 to 20){
// CHECK-DAG:           [[VAR_10_1_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_0_MEM_2_:%.+]] = krnl.load [[VAR_1_]][] : memref<f32>
// CHECK:               [[LOAD_arg0_MEM_1_:%.+]] = krnl.load [[arg0_]]{{.}}[[VAR_4_]]#0, [[VAR_10_1_]], [[VAR_4_]]#1] : memref<10x20x30xf32>
// CHECK:               [[VAR_13_1_:%.+]] = arith.subf [[LOAD_arg0_MEM_1_]], [[LOAD_VAR_0_MEM_1_]] : f32
// CHECK:               [[VAR_14_1_:%.+]] = math.exp [[VAR_13_1_]] : f32
// CHECK:               [[VAR_15_:%.+]] = arith.addf [[LOAD_VAR_0_MEM_2_]], [[VAR_14_1_]] : f32
// CHECK:               krnl.store [[VAR_15_]], [[VAR_1_]][] : memref<f32>
// CHECK:               krnl.store [[VAR_14_1_]], [[VAR_2_]]{{.}}[[VAR_4_]]#0, [[VAR_10_1_]], [[VAR_4_]]#1] : memref<10x20x30xf32>
// CHECK:             }
// CHECK-DAG:         [[LOAD_VAR_1_MEM_:%.+]] = krnl.load [[VAR_1_]][] : memref<f32>
// CHECK-DAG:         [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[I_4_:%.+]] = 0 to 20){
// CHECK:               [[VAR_10_2_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_0_MEM_2_:%.+]] = krnl.load [[VAR_2_]]{{.}}[[VAR_4_]]#0, [[VAR_10_2_]], [[VAR_4_]]#1] : memref<10x20x30xf32>
// CHECK:               [[LOAD_arg0_MEM_1_:%.+]] = arith.divf [[LOAD_VAR_0_MEM_2_]], [[LOAD_VAR_1_MEM_]] : f32
// CHECK:               krnl.store [[LOAD_arg0_MEM_1_]], [[VAR_2_]]{{.}}[[VAR_4_]]#0, [[VAR_10_2_]], [[VAR_4_]]#1] : memref<10x20x30xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_2_]] : memref<10x20x30xf32>
// CHECK:         }
}
