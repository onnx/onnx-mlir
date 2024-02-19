// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

func.func @instance_norm(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<3xf32>, %arg2: tensor<3xf32>) -> tensor<2x3x4x5xf32> {
  %0 = "onnx.InstanceNormalization"(%arg0, %arg1, %arg2) {epsilon = 0.00999999977 : f32} : (tensor<2x3x4x5xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<2x3x4x5xf32>
  return %0 : tensor<2x3x4x5xf32>

// mlir2FileCheck.py -a'["input", "scale", "bias"]'
// CHECK-LABEL:  func @instance_norm
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<2x3x4x5xf32>, [[SCALE_:%.+]]: memref<3xf32>, [[BIAS_:%.+]]: memref<3xf32>) -> memref<2x3x4x5xf32> {
// CHECK-DAG:       [[CST_20_:%.+]] = arith.constant 2.000000e+01 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_dot_00999999977_:%.+]] = arith.constant 0.00999999977 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3x4x5xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3){
// CHECK-DAG:         [[VAR_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_1_]][] : memref<f32>
// CHECK:             [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_2_:%.+]] = 0 to 4, [[LOOP_2_]]#1 -> [[I_3_:%.+]] = 0 to 5){
// CHECK-DAG:           [[VAR_17_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:               [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_17_]]#0, [[VAR_17_]]#1] : memref<2x3x4x5xf32>
// CHECK:               [[VAR_20_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[LOAD_INPUT_MEM_]] : f32
// CHECK:               krnl.store [[VAR_20_]], [[RES_1_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:             [[VAR_8_:%.+]] = arith.divf [[LOAD_RES_1_MEM_1_]], [[CST_20_]] : f32
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_1_]][] : memref<f32>
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_4_:%.+]] = 0 to 4, [[LOOP_1_]]#1 -> [[I_5_:%.+]] = 0 to 5){
// CHECK-DAG:           [[VAR_17_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:               [[LOAD_INPUT_MEM_1_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_17_1_]]#0, [[VAR_17_1_]]#1] : memref<2x3x4x5xf32>
// CHECK:               [[VAR_20_1_:%.+]] = arith.subf [[LOAD_INPUT_MEM_1_]], [[VAR_8_]] : f32
// CHECK:               [[VAR_21_:%.+]] = arith.mulf [[VAR_20_1_]], [[VAR_20_1_]] : f32
// CHECK:               [[VAR_22_:%.+]] = arith.addf [[LOAD_RES_1_MEM_2_]], [[VAR_21_]] : f32
// CHECK:               krnl.store [[VAR_22_]], [[RES_1_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_1_MEM_3_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:             [[VAR_10_:%.+]] = arith.divf [[LOAD_RES_1_MEM_3_]], [[CST_20_]] : f32
// CHECK:             [[VAR_11_:%.+]] = arith.addf [[VAR_10_]], [[CST_0_dot_00999999977_]] : f32
// CHECK-DAG:         [[VAR_12_:%.+]] = math.sqrt [[VAR_11_]] : f32
// CHECK-DAG:         [[LOAD_SCALE_MEM_:%.+]] = krnl.load [[SCALE_]]{{.}}[[VAR_3_]]#1] : memref<3xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.divf [[LOAD_SCALE_MEM_]], [[VAR_12_]] : f32
// CHECK-DAG:         [[LOAD_BIAS_MEM_:%.+]] = krnl.load [[BIAS_]]{{.}}[[VAR_3_]]#1] : memref<3xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_6_:%.+]] = 0 to 4, [[LOOP_3_]]#1 -> [[I_7_:%.+]] = 0 to 5){
// CHECK:               [[VAR_17_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_INPUT_MEM_2_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_17_2_]]#0, [[VAR_17_2_]]#1] : memref<2x3x4x5xf32>
// CHECK:               [[LOAD_INPUT_MEM_1_:%.+]] = arith.subf [[LOAD_INPUT_MEM_2_]], [[VAR_8_]] : f32
// CHECK:               [[VAR_20_2_:%.+]] = arith.mulf [[VAR_14_]], [[LOAD_INPUT_MEM_1_]] : f32
// CHECK:               [[VAR_21_1_:%.+]] = arith.addf [[VAR_20_2_]], [[LOAD_BIAS_MEM_]] : f32
// CHECK:               krnl.store [[VAR_21_1_]], [[RES_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_17_2_]]#0, [[VAR_17_2_]]#1] : memref<2x3x4x5xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3x4x5xf32>
// CHECK:         }
}
