// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl='check-rnn-ops-lowering' %s -split-input-file | FileCheck %s

func private @test_gru_forward_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x12x3xf32>, %arg2: tensor<1x12x4xf32>, %arg3: tensor<1x24xf32>, %arg4: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64} : (tensor<7x2x3xf32>, tensor<1x12x3xf32>, tensor<1x12x4xf32>, tensor<1x24xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:  func private @test_gru_forward_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x12x3xf32>, [[PARAM_2_:%.+]]: memref<1x12x4xf32>, [[PARAM_3_:%.+]]: memref<1x24xf32>, [[PARAM_4_:%.+]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.alloc() : memref<2x4xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.alloc() : memref<1x2x4xf32>
// CHECK-DAG:       [[VAR_cst_:%.+]] = constant unit
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = constant 1 : index
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4) {
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[VAR_0_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Squeeze"([[PARAM_1_]]) {axes = [0]} : (memref<1x12x3xf32>) -> memref<12x3xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Squeeze"([[PARAM_2_]]) {axes = [0]} : (memref<1x12x4xf32>) -> memref<12x4xf32>
// CHECK:           [[VAR_5_:%.+]]:3 = "onnx.Split"([[VAR_3_]]) {axis = 0 : si64} : (memref<12x3xf32>) -> (memref<4x3xf32>, memref<4x3xf32>, memref<4x3xf32>)
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Transpose"([[VAR_5_]]#0) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Transpose"([[VAR_5_]]#1) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Transpose"([[VAR_5_]]#2) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_9_:%.+]]:3 = "onnx.Split"([[VAR_4_]]) {axis = 0 : si64} : (memref<12x4xf32>) -> (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Transpose"([[VAR_9_]]#0) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Transpose"([[VAR_9_]]#1) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Transpose"([[VAR_9_]]#2) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Squeeze"([[PARAM_3_]]) {axes = [0]} : (memref<1x24xf32>) -> memref<24xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_14_:%.+]]:6 = "onnx.Split"([[VAR_13_]]) {axis = 0 : si64} : (memref<24xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7) {
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_1_:%.+]] = memref.alloc() : memref<2x4xf32>
// CHECK-DAG:         [[VAR_17_:%.+]] = memref.alloc() : memref<2x4xf32>
// CHECK-DAG:         [[VAR_18_:%.+]] = memref.alloc() : memref<2x3xf32>
// CHECK-DAG:         [[CST_0_1_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_2_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_3_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_1_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_3_:%.+]] = constant 3 : index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_2_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_2_]] to [[CST_3_]]) {
// CHECK:               [[VAR_28_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_28_]]#0, [[VAR_28_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[VAR_18_]]{{.}}[[VAR_28_]]#0, [[VAR_28_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_20_:%.+]] = "onnx.MatMul"([[VAR_18_]], [[VAR_6_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_21_:%.+]] = "onnx.MatMul"([[VAR_0_]], [[VAR_10_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_22_:%.+]] = "onnx.MatMul"([[VAR_18_]], [[VAR_7_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_23_:%.+]] = "onnx.MatMul"([[VAR_0_]], [[VAR_11_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_24_:%.+]] = "onnx.MatMul"([[VAR_18_]], [[VAR_8_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[CST_1_dot_000000_:%.+]] = constant 1.000000e+00 : f32
// CHECK-DAG:         [[CST_0_4_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_5_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_1_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_2_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_4_]] to [[CST_2_1_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_4_]] to [[CST_4_]]) {
// CHECK:               [[VAR_28_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[VAR_28_1_]]#0, [[VAR_28_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_22_MEM_:%.+]] = krnl.load [[VAR_22_]]{{.}}[[VAR_28_1_]]#0, [[VAR_28_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_23_MEM_:%.+]] = krnl.load [[VAR_23_]]{{.}}[[VAR_28_1_]]#0, [[VAR_28_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_32_:%.+]] = addf [[LOAD_VAR_22_MEM_]], [[LOAD_VAR_23_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_14_MEM_:%.+]] = krnl.load [[VAR_14_]]#1{{.}}[[VAR_28_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_14_MEM_1_:%.+]] = krnl.load [[VAR_14_]]#4{{.}}[[VAR_28_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_35_:%.+]] = addf [[VAR_32_]], [[LOAD_VAR_14_MEM_]] : f32
// CHECK-DAG:           [[VAR_36_:%.+]] = addf [[VAR_35_]], [[LOAD_VAR_14_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_37_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_36_]], [[VAR_37_]][] : memref<f32>
// CHECK:               [[VAR_38_:%.+]] = "onnx.Sigmoid"([[VAR_37_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_VAR_38_MEM_:%.+]] = krnl.load [[VAR_38_]][] : memref<f32>
// CHECK:               krnl.store [[LOAD_VAR_38_MEM_]], [[VAR_17_]]{{.}}[[VAR_28_1_]]#0, [[VAR_28_1_]]#1] : memref<2x4xf32>
// CHECK:               [[VAR_40_:%.+]] = mulf [[LOAD_VAR_38_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               krnl.store [[VAR_40_]], [[LOAD_PARAM_4_MEM_1_]]{{.}}[[VAR_28_1_]]#0, [[VAR_28_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_26_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_4_MEM_1_]], [[VAR_12_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_7_:%.+]] = [[CST_0_4_]] to [[CST_2_1_]], [[LOOP_4_]]#1 -> [[I_8_:%.+]] = [[CST_0_4_]] to [[CST_4_]]) {
// CHECK:               [[VAR_28_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[VAR_28_2_]]#0, [[VAR_28_2_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_22_MEM_1_:%.+]] = krnl.load [[VAR_20_]]{{.}}[[VAR_28_2_]]#0, [[VAR_28_2_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_23_MEM_1_:%.+]] = krnl.load [[VAR_21_]]{{.}}[[VAR_28_2_]]#0, [[VAR_28_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_32_1_:%.+]] = addf [[LOAD_VAR_22_MEM_1_]], [[LOAD_VAR_23_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_14_MEM_2_:%.+]] = krnl.load [[VAR_14_]]#0{{.}}[[VAR_28_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_14_MEM_3_:%.+]] = krnl.load [[VAR_14_]]#3{{.}}[[VAR_28_2_]]#1] : memref<4xf32>
// CHECK:               [[VAR_35_1_:%.+]] = addf [[VAR_32_1_]], [[LOAD_VAR_14_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_36_1_:%.+]] = addf [[VAR_35_1_]], [[LOAD_VAR_14_MEM_3_]] : f32
// CHECK-DAG:           [[VAR_37_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_36_1_]], [[VAR_37_1_]][] : memref<f32>
// CHECK:               [[VAR_38_1_:%.+]] = "onnx.Sigmoid"([[VAR_37_1_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_38_MEM_1_:%.+]] = krnl.load [[VAR_38_1_]][] : memref<f32>
// CHECK-DAG:           [[VAR_40_1_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_28_2_]]#0, [[VAR_28_2_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_26_MEM_:%.+]] = krnl.load [[VAR_26_]]{{.}}[[VAR_28_2_]]#0, [[VAR_28_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_42_:%.+]] = addf [[VAR_40_1_]], [[LOAD_VAR_26_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_14_MEM_4_:%.+]] = krnl.load [[VAR_14_]]#2{{.}}[[VAR_28_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_14_MEM_5_:%.+]] = krnl.load [[VAR_14_]]#5{{.}}[[VAR_28_2_]]#1] : memref<4xf32>
// CHECK:               [[VAR_45_:%.+]] = addf [[VAR_42_]], [[LOAD_VAR_14_MEM_4_]] : f32
// CHECK-DAG:           [[VAR_46_:%.+]] = addf [[VAR_45_]], [[LOAD_VAR_14_MEM_5_]] : f32
// CHECK-DAG:           [[VAR_47_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_46_]], [[VAR_47_]][] : memref<f32>
// CHECK:               [[VAR_48_:%.+]] = "onnx.Tanh"([[VAR_47_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_48_MEM_:%.+]] = krnl.load [[VAR_48_]][] : memref<f32>
// CHECK-DAG:           [[VAR_50_:%.+]] = subf [[CST_1_dot_000000_]], [[LOAD_VAR_38_MEM_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_51_:%.+]] = mulf [[VAR_50_]], [[LOAD_VAR_48_MEM_]] : f32
// CHECK-DAG:           [[VAR_52_:%.+]] = mulf [[LOAD_VAR_38_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_53_:%.+]] = addf [[VAR_51_]], [[VAR_52_]] : f32
// CHECK:               krnl.store [[VAR_53_]], [[VAR_0_]]{{.}}[[VAR_28_2_]]#0, [[VAR_28_2_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc [[VAR_17_]] : memref<2x4xf32>
// CHECK:             memref.dealloc [[LOAD_PARAM_4_MEM_1_]] : memref<2x4xf32>
// CHECK:             memref.dealloc [[VAR_18_]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           [[CST_32_:%.+]] = constant 32 : i64
// CHECK:           "krnl.memcpy"([[VAR_1_]], [[VAR_0_]], [[CST_32_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
// CHECK:           memref.dealloc [[VAR_0_]] : memref<2x4xf32>
// CHECK:           return [[VAR_1_]] : memref<1x2x4xf32>
// CHECK:         }
}

// -----

func private @test_gru_forward_mode_linear_before_reset(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x12x3xf32>, %arg2: tensor<1x12x4xf32>, %arg3: tensor<1x24xf32>, %arg4: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64, linear_before_reset = 1 : si64} : (tensor<7x2x3xf32>, tensor<1x12x3xf32>, tensor<1x12x4xf32>, tensor<1x24xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>
  
// CHECK-LABEL:  func private @test_gru_forward_mode_linear_before_reset
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x12x3xf32>, [[PARAM_2_:%.+]]: memref<1x12x4xf32>, [[PARAM_3_:%.+]]: memref<1x24xf32>, [[PARAM_4_:%.+]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.alloc() : memref<2x4xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.alloc() : memref<1x2x4xf32>
// CHECK-DAG:       [[VAR_cst_:%.+]] = constant unit
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = constant 1 : index
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4) {
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[VAR_0_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Squeeze"([[PARAM_1_]]) {axes = [0]} : (memref<1x12x3xf32>) -> memref<12x3xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Squeeze"([[PARAM_2_]]) {axes = [0]} : (memref<1x12x4xf32>) -> memref<12x4xf32>
// CHECK:           [[VAR_5_:%.+]]:3 = "onnx.Split"([[VAR_3_]]) {axis = 0 : si64} : (memref<12x3xf32>) -> (memref<4x3xf32>, memref<4x3xf32>, memref<4x3xf32>)
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Transpose"([[VAR_5_]]#0) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Transpose"([[VAR_5_]]#1) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Transpose"([[VAR_5_]]#2) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_9_:%.+]]:3 = "onnx.Split"([[VAR_4_]]) {axis = 0 : si64} : (memref<12x4xf32>) -> (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Transpose"([[VAR_9_]]#0) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Transpose"([[VAR_9_]]#1) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Transpose"([[VAR_9_]]#2) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Squeeze"([[PARAM_3_]]) {axes = [0]} : (memref<1x24xf32>) -> memref<24xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_14_:%.+]]:6 = "onnx.Split"([[VAR_13_]]) {axis = 0 : si64} : (memref<24xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7) {
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_1_:%.+]] = memref.alloc() : memref<2x3xf32>
// CHECK-DAG:         [[CST_0_1_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_2_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_3_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_1_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_3_:%.+]] = constant 3 : index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_2_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_2_]] to [[CST_3_]]) {
// CHECK:               [[VAR_25_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_25_]]#0, [[VAR_25_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_4_MEM_1_]]{{.}}[[VAR_25_]]#0, [[VAR_25_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_18_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_4_MEM_1_]], [[VAR_6_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_19_:%.+]] = "onnx.MatMul"([[VAR_0_]], [[VAR_10_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_20_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_4_MEM_1_]], [[VAR_7_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_21_:%.+]] = "onnx.MatMul"([[VAR_0_]], [[VAR_11_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_22_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_4_MEM_1_]], [[VAR_8_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[CST_1_dot_000000_:%.+]] = constant 1.000000e+00 : f32
// CHECK-DAG:         [[CST_0_4_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_5_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_1_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_2_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:         [[VAR_23_:%.+]] = "onnx.MatMul"([[VAR_0_]], [[VAR_12_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_4_]] to [[CST_2_1_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_4_]] to [[CST_4_]]) {
// CHECK:               [[VAR_25_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[VAR_25_1_]]#0, [[VAR_25_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_18_MEM_:%.+]] = krnl.load [[VAR_18_]]{{.}}[[VAR_25_1_]]#0, [[VAR_25_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_19_MEM_:%.+]] = krnl.load [[VAR_19_]]{{.}}[[VAR_25_1_]]#0, [[VAR_25_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_29_:%.+]] = addf [[LOAD_VAR_18_MEM_]], [[LOAD_VAR_19_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_14_MEM_:%.+]] = krnl.load [[VAR_14_]]#0{{.}}[[VAR_25_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_14_MEM_1_:%.+]] = krnl.load [[VAR_14_]]#3{{.}}[[VAR_25_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_32_:%.+]] = addf [[VAR_29_]], [[LOAD_VAR_14_MEM_]] : f32
// CHECK-DAG:           [[VAR_33_:%.+]] = addf [[VAR_32_]], [[LOAD_VAR_14_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_34_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_33_]], [[VAR_34_]][] : memref<f32>
// CHECK:               [[VAR_35_:%.+]] = "onnx.Sigmoid"([[VAR_34_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_35_MEM_:%.+]] = krnl.load [[VAR_35_]][] : memref<f32>
// CHECK-DAG:           [[LOAD_VAR_20_MEM_:%.+]] = krnl.load [[VAR_20_]]{{.}}[[VAR_25_1_]]#0, [[VAR_25_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_21_MEM_:%.+]] = krnl.load [[VAR_21_]]{{.}}[[VAR_25_1_]]#0, [[VAR_25_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_39_:%.+]] = addf [[LOAD_VAR_20_MEM_]], [[LOAD_VAR_21_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_14_MEM_2_:%.+]] = krnl.load [[VAR_14_]]#1{{.}}[[VAR_25_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_14_MEM_3_:%.+]] = krnl.load [[VAR_14_]]#4{{.}}[[VAR_25_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_42_:%.+]] = addf [[VAR_39_]], [[LOAD_VAR_14_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_43_:%.+]] = addf [[VAR_42_]], [[LOAD_VAR_14_MEM_3_]] : f32
// CHECK-DAG:           [[VAR_44_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_43_]], [[VAR_44_]][] : memref<f32>
// CHECK:               [[VAR_45_:%.+]] = "onnx.Sigmoid"([[VAR_44_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_45_MEM_:%.+]] = krnl.load [[VAR_45_]][] : memref<f32>
// CHECK-DAG:           [[LOAD_VAR_22_MEM_:%.+]] = krnl.load [[VAR_22_]]{{.}}[[VAR_25_1_]]#0, [[VAR_25_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_23_MEM_:%.+]] = krnl.load [[VAR_23_]]{{.}}[[VAR_25_1_]]#0, [[VAR_25_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_14_MEM_4_:%.+]] = krnl.load [[VAR_14_]]#5{{.}}[[VAR_25_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_50_:%.+]] = addf [[LOAD_VAR_23_MEM_]], [[LOAD_VAR_14_MEM_4_]] : f32
// CHECK:               [[VAR_51_:%.+]] = mulf [[LOAD_VAR_45_MEM_]], [[VAR_50_]] : f32
// CHECK-DAG:           [[VAR_52_:%.+]] = addf [[LOAD_VAR_22_MEM_]], [[VAR_51_]] : f32
// CHECK-DAG:           [[LOAD_VAR_14_MEM_5_:%.+]] = krnl.load [[VAR_14_]]#2{{.}}[[VAR_25_1_]]#1] : memref<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_54_:%.+]] = addf [[VAR_52_]], [[LOAD_VAR_14_MEM_5_]] : f32
// CHECK-DAG:           [[VAR_55_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_54_]], [[VAR_55_]][] : memref<f32>
// CHECK:               [[VAR_56_:%.+]] = "onnx.Tanh"([[VAR_55_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_56_MEM_:%.+]] = krnl.load [[VAR_56_]][] : memref<f32>
// CHECK-DAG:           [[VAR_58_:%.+]] = subf [[CST_1_dot_000000_]], [[LOAD_VAR_35_MEM_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_59_:%.+]] = mulf [[VAR_58_]], [[LOAD_VAR_56_MEM_]] : f32
// CHECK-DAG:           [[VAR_60_:%.+]] = mulf [[LOAD_VAR_35_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_61_:%.+]] = addf [[VAR_59_]], [[VAR_60_]] : f32
// CHECK:               krnl.store [[VAR_61_]], [[VAR_0_]]{{.}}[[VAR_25_1_]]#0, [[VAR_25_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc [[LOAD_PARAM_4_MEM_1_]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           [[CST_32_:%.+]] = constant 32 : i64
// CHECK:           "krnl.memcpy"([[VAR_1_]], [[VAR_0_]], [[CST_32_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
// CHECK:           memref.dealloc [[VAR_0_]] : memref<2x4xf32>
// CHECK:           return [[VAR_1_]] : memref<1x2x4xf32>
// CHECK:         }
}

// -----

func private @test_gru_forward_mode_constant_weight_and_bias(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %w = "onnx.Constant"() {value = dense<[[[1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]]> : tensor<1x12x3xf32>} : () -> tensor<1x12x3xf32> 
  %r = "onnx.Constant"() {value = dense<[[[2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.]]]> : tensor<1x12x4xf32>} : () -> tensor<1x12x4xf32> 
  %b = "onnx.Constant"() {value = dense<[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.]]> : tensor<1x24xf32>} : () -> tensor<1x24xf32> 

  %Y, %Y_h = "onnx.GRU"(%arg0, %w, %r, %b, %cst, %arg1) {hidden_size = 4 : si64} : (tensor<7x2x3xf32>, tensor<1x12x3xf32>, tensor<1x12x4xf32>, tensor<1x24xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>
  
// CHECK-LABEL:  func private @test_gru_forward_mode_constant_weight_and_bias
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.alloc() : memref<2x4xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.alloc() : memref<1x2x4xf32>
// CHECK-DAG:       [[VAR_cst_:%.+]] = constant unit
// CHECK-DAG:       [[VAR_2_:%.+]] = "krnl.global"() {name = "constant_0", shape = [1, 12, 3], value = dense<1.000000e+00> : tensor<1x12x3xf32>} : () -> memref<1x12x3xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "krnl.global"() {name = "constant_1", shape = [1, 12, 4], value = dense<2.000000e+00> : tensor<1x12x4xf32>} : () -> memref<1x12x4xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "krnl.global"() {name = "constant_2", shape = [1, 24], value = dense<{{.}}[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01]{{.}}> : tensor<1x24xf32>} : () -> memref<1x24xf32>
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = constant 1 : index
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4) {
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[VAR_0_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_6_:%.+]] = "krnl.global"() {name = "constant_3", shape = [12, 3], value = dense<1.000000e+00> : tensor<12x3xf32>} : () -> memref<12x3xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "krnl.global"() {name = "constant_4", shape = [12, 4], value = dense<2.000000e+00> : tensor<12x4xf32>} : () -> memref<12x4xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "krnl.global"() {name = "constant_5", shape = [4, 3], value = dense<1.000000e+00> : tensor<4x3xf32>} : () -> memref<4x3xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "krnl.global"() {name = "constant_6", shape = [4, 3], value = dense<1.000000e+00> : tensor<4x3xf32>} : () -> memref<4x3xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "krnl.global"() {name = "constant_7", shape = [4, 3], value = dense<1.000000e+00> : tensor<4x3xf32>} : () -> memref<4x3xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "krnl.global"() {name = "constant_8", shape = [3, 4], value = dense<1.000000e+00> : tensor<3x4xf32>} : () -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "krnl.global"() {name = "constant_9", shape = [3, 4], value = dense<1.000000e+00> : tensor<3x4xf32>} : () -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "krnl.global"() {name = "constant_10", shape = [3, 4], value = dense<1.000000e+00> : tensor<3x4xf32>} : () -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = "krnl.global"() {name = "constant_11", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = "krnl.global"() {name = "constant_12", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "krnl.global"() {name = "constant_13", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = "krnl.global"() {name = "constant_14", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "krnl.global"() {name = "constant_15", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "krnl.global"() {name = "constant_16", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = "krnl.global"() {name = "constant_17", shape = [24], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01]> : tensor<24xf32>} : () -> memref<24xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "krnl.global"() {name = "constant_18", shape = [4], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = "krnl.global"() {name = "constant_19", shape = [4], value = dense<[5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_23_:%.+]] = "krnl.global"() {name = "constant_20", shape = [4], value = dense<[9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_24_:%.+]] = "krnl.global"() {name = "constant_21", shape = [4], value = dense<[1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_25_:%.+]] = "krnl.global"() {name = "constant_22", shape = [4], value = dense<[1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_26_:%.+]] = "krnl.global"() {name = "constant_23", shape = [4], value = dense<[2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7) {
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_1_:%.+]] = memref.alloc() : memref<2x4xf32>
// CHECK-DAG:         [[VAR_29_:%.+]] = memref.alloc() : memref<2x4xf32>
// CHECK-DAG:         [[VAR_30_:%.+]] = memref.alloc() : memref<2x3xf32>
// CHECK-DAG:         [[CST_0_1_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_2_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_3_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_1_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_3_:%.+]] = constant 3 : index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_2_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_2_]] to [[CST_3_]]) {
// CHECK:               [[VAR_40_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_40_]]#0, [[VAR_40_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[VAR_30_]]{{.}}[[VAR_40_]]#0, [[VAR_40_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_32_:%.+]] = "onnx.MatMul"([[VAR_30_]], [[VAR_11_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_33_:%.+]] = "onnx.MatMul"([[VAR_0_]], [[VAR_17_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_34_:%.+]] = "onnx.MatMul"([[VAR_30_]], [[VAR_12_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_35_:%.+]] = "onnx.MatMul"([[VAR_0_]], [[VAR_18_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_36_:%.+]] = "onnx.MatMul"([[VAR_30_]], [[VAR_13_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[CST_1_dot_000000_:%.+]] = constant 1.000000e+00 : f32
// CHECK-DAG:         [[CST_0_4_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_5_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_1_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_2_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_4_]] to [[CST_2_1_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_4_]] to [[CST_4_]]) {
// CHECK:               [[VAR_40_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[VAR_40_1_]]#0, [[VAR_40_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_34_MEM_:%.+]] = krnl.load [[VAR_34_]]{{.}}[[VAR_40_1_]]#0, [[VAR_40_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_35_MEM_:%.+]] = krnl.load [[VAR_35_]]{{.}}[[VAR_40_1_]]#0, [[VAR_40_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_44_:%.+]] = addf [[LOAD_VAR_34_MEM_]], [[LOAD_VAR_35_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_22_MEM_:%.+]] = krnl.load [[VAR_22_]]{{.}}[[VAR_40_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_25_MEM_:%.+]] = krnl.load [[VAR_25_]]{{.}}[[VAR_40_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_47_:%.+]] = addf [[VAR_44_]], [[LOAD_VAR_22_MEM_]] : f32
// CHECK-DAG:           [[VAR_48_:%.+]] = addf [[VAR_47_]], [[LOAD_VAR_25_MEM_]] : f32
// CHECK-DAG:           [[VAR_49_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_48_]], [[VAR_49_]][] : memref<f32>
// CHECK:               [[VAR_50_:%.+]] = "onnx.Sigmoid"([[VAR_49_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_VAR_50_MEM_:%.+]] = krnl.load [[VAR_50_]][] : memref<f32>
// CHECK:               krnl.store [[LOAD_VAR_50_MEM_]], [[VAR_29_]]{{.}}[[VAR_40_1_]]#0, [[VAR_40_1_]]#1] : memref<2x4xf32>
// CHECK:               [[VAR_52_:%.+]] = mulf [[LOAD_VAR_50_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               krnl.store [[VAR_52_]], [[LOAD_PARAM_1_MEM_1_]]{{.}}[[VAR_40_1_]]#0, [[VAR_40_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_38_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_1_MEM_1_]], [[VAR_19_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_7_:%.+]] = [[CST_0_4_]] to [[CST_2_1_]], [[LOOP_4_]]#1 -> [[I_8_:%.+]] = [[CST_0_4_]] to [[CST_4_]]) {
// CHECK:               [[VAR_40_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[VAR_40_2_]]#0, [[VAR_40_2_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_34_MEM_1_:%.+]] = krnl.load [[VAR_32_]]{{.}}[[VAR_40_2_]]#0, [[VAR_40_2_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_35_MEM_1_:%.+]] = krnl.load [[VAR_33_]]{{.}}[[VAR_40_2_]]#0, [[VAR_40_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_44_1_:%.+]] = addf [[LOAD_VAR_34_MEM_1_]], [[LOAD_VAR_35_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_22_MEM_1_:%.+]] = krnl.load [[VAR_21_]]{{.}}[[VAR_40_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_25_MEM_1_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_40_2_]]#1] : memref<4xf32>
// CHECK:               [[VAR_47_1_:%.+]] = addf [[VAR_44_1_]], [[LOAD_VAR_22_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_48_1_:%.+]] = addf [[VAR_47_1_]], [[LOAD_VAR_25_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_49_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_48_1_]], [[VAR_49_1_]][] : memref<f32>
// CHECK:               [[VAR_50_1_:%.+]] = "onnx.Sigmoid"([[VAR_49_1_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_50_MEM_1_:%.+]] = krnl.load [[VAR_50_1_]][] : memref<f32>
// CHECK-DAG:           [[VAR_52_1_:%.+]] = krnl.load [[VAR_36_]]{{.}}[[VAR_40_2_]]#0, [[VAR_40_2_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_38_MEM_:%.+]] = krnl.load [[VAR_38_]]{{.}}[[VAR_40_2_]]#0, [[VAR_40_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_54_:%.+]] = addf [[VAR_52_1_]], [[LOAD_VAR_38_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_23_MEM_:%.+]] = krnl.load [[VAR_23_]]{{.}}[[VAR_40_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_26_MEM_:%.+]] = krnl.load [[VAR_26_]]{{.}}[[VAR_40_2_]]#1] : memref<4xf32>
// CHECK:               [[VAR_57_:%.+]] = addf [[VAR_54_]], [[LOAD_VAR_23_MEM_]] : f32
// CHECK-DAG:           [[VAR_58_:%.+]] = addf [[VAR_57_]], [[LOAD_VAR_26_MEM_]] : f32
// CHECK-DAG:           [[VAR_59_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_58_]], [[VAR_59_]][] : memref<f32>
// CHECK:               [[VAR_60_:%.+]] = "onnx.Tanh"([[VAR_59_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_60_MEM_:%.+]] = krnl.load [[VAR_60_]][] : memref<f32>
// CHECK-DAG:           [[VAR_62_:%.+]] = subf [[CST_1_dot_000000_]], [[LOAD_VAR_50_MEM_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_63_:%.+]] = mulf [[VAR_62_]], [[LOAD_VAR_60_MEM_]] : f32
// CHECK-DAG:           [[VAR_64_:%.+]] = mulf [[LOAD_VAR_50_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_65_:%.+]] = addf [[VAR_63_]], [[VAR_64_]] : f32
// CHECK:               krnl.store [[VAR_65_]], [[VAR_0_]]{{.}}[[VAR_40_2_]]#0, [[VAR_40_2_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc [[VAR_29_]] : memref<2x4xf32>
// CHECK:             memref.dealloc [[LOAD_PARAM_1_MEM_1_]] : memref<2x4xf32>
// CHECK:             memref.dealloc [[VAR_30_]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           [[CST_32_:%.+]] = constant 32 : i64
// CHECK:           "krnl.memcpy"([[VAR_1_]], [[VAR_0_]], [[CST_32_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
// CHECK:           memref.dealloc [[VAR_0_]] : memref<2x4xf32>
// CHECK:           return [[VAR_1_]] : memref<1x2x4xf32>
// CHECK:         }
}

// -----

func private @test_gru_reverse_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x12x3xf32>, %arg2: tensor<1x12x4xf32>, %arg3: tensor<1x24xf32>, %arg4: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64, direction = "reverse"} : (tensor<7x2x3xf32>, tensor<1x12x3xf32>, tensor<1x12x4xf32>, tensor<1x24xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>
  
// CHECK-LABEL:  func private @test_gru_reverse_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x12x3xf32>, [[PARAM_2_:%.+]]: memref<1x12x4xf32>, [[PARAM_3_:%.+]]: memref<1x24xf32>, [[PARAM_4_:%.+]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.alloc() : memref<2x4xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.alloc() : memref<1x2x4xf32>
// CHECK-DAG:       [[VAR_cst_:%.+]] = constant unit
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = constant 1 : index
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4) {
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[VAR_0_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Squeeze"([[PARAM_1_]]) {axes = [0]} : (memref<1x12x3xf32>) -> memref<12x3xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Squeeze"([[PARAM_2_]]) {axes = [0]} : (memref<1x12x4xf32>) -> memref<12x4xf32>
// CHECK:           [[VAR_5_:%.+]]:3 = "onnx.Split"([[VAR_3_]]) {axis = 0 : si64} : (memref<12x3xf32>) -> (memref<4x3xf32>, memref<4x3xf32>, memref<4x3xf32>)
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Transpose"([[VAR_5_]]#0) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Transpose"([[VAR_5_]]#1) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Transpose"([[VAR_5_]]#2) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_9_:%.+]]:3 = "onnx.Split"([[VAR_4_]]) {axis = 0 : si64} : (memref<12x4xf32>) -> (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Transpose"([[VAR_9_]]#0) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Transpose"([[VAR_9_]]#1) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Transpose"([[VAR_9_]]#2) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Squeeze"([[PARAM_3_]]) {axes = [0]} : (memref<1x24xf32>) -> memref<24xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_14_:%.+]]:6 = "onnx.Split"([[VAR_13_]]) {axis = 0 : si64} : (memref<24xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7) {
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_1_:%.+]] = memref.alloc() : memref<2x4xf32>
// CHECK-DAG:         [[VAR_17_:%.+]] = memref.alloc() : memref<2x4xf32>
// CHECK-DAG:         [[VAR_18_:%.+]] = memref.alloc() : memref<2x3xf32>
// CHECK-DAG:         [[CST_0_1_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_7_:%.+]] = constant 7 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_19_:%.+]] = affine.apply #map([[I_2_]]){{.}}[[CST_7_]]{{.}}
// CHECK-DAG:         [[CST_0_2_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_3_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_1_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_3_:%.+]] = constant 3 : index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_2_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_2_]] to [[CST_3_]]) {
// CHECK:               [[VAR_29_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_19_]], [[VAR_29_]]#0, [[VAR_29_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[VAR_18_]]{{.}}[[VAR_29_]]#0, [[VAR_29_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_21_:%.+]] = "onnx.MatMul"([[VAR_18_]], [[VAR_6_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_22_:%.+]] = "onnx.MatMul"([[VAR_0_]], [[VAR_10_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_23_:%.+]] = "onnx.MatMul"([[VAR_18_]], [[VAR_7_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_24_:%.+]] = "onnx.MatMul"([[VAR_0_]], [[VAR_11_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_25_:%.+]] = "onnx.MatMul"([[VAR_18_]], [[VAR_8_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[CST_1_dot_000000_:%.+]] = constant 1.000000e+00 : f32
// CHECK-DAG:         [[CST_0_4_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_5_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_1_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_2_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_4_]] to [[CST_2_1_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_4_]] to [[CST_4_]]) {
// CHECK:               [[VAR_29_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[VAR_29_1_]]#0, [[VAR_29_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_23_MEM_:%.+]] = krnl.load [[VAR_23_]]{{.}}[[VAR_29_1_]]#0, [[VAR_29_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_24_MEM_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_29_1_]]#0, [[VAR_29_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_33_:%.+]] = addf [[LOAD_VAR_23_MEM_]], [[LOAD_VAR_24_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_14_MEM_:%.+]] = krnl.load [[VAR_14_]]#1{{.}}[[VAR_29_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_14_MEM_1_:%.+]] = krnl.load [[VAR_14_]]#4{{.}}[[VAR_29_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_36_:%.+]] = addf [[VAR_33_]], [[LOAD_VAR_14_MEM_]] : f32
// CHECK-DAG:           [[VAR_37_:%.+]] = addf [[VAR_36_]], [[LOAD_VAR_14_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_38_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_37_]], [[VAR_38_]][] : memref<f32>
// CHECK:               [[VAR_39_:%.+]] = "onnx.Sigmoid"([[VAR_38_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_VAR_39_MEM_:%.+]] = krnl.load [[VAR_39_]][] : memref<f32>
// CHECK:               krnl.store [[LOAD_VAR_39_MEM_]], [[VAR_17_]]{{.}}[[VAR_29_1_]]#0, [[VAR_29_1_]]#1] : memref<2x4xf32>
// CHECK:               [[VAR_41_:%.+]] = mulf [[LOAD_VAR_39_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               krnl.store [[VAR_41_]], [[LOAD_PARAM_4_MEM_1_]]{{.}}[[VAR_29_1_]]#0, [[VAR_29_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_27_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_4_MEM_1_]], [[VAR_12_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_7_:%.+]] = [[CST_0_4_]] to [[CST_2_1_]], [[LOOP_4_]]#1 -> [[I_8_:%.+]] = [[CST_0_4_]] to [[CST_4_]]) {
// CHECK:               [[VAR_29_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[VAR_29_2_]]#0, [[VAR_29_2_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_23_MEM_1_:%.+]] = krnl.load [[VAR_21_]]{{.}}[[VAR_29_2_]]#0, [[VAR_29_2_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_24_MEM_1_:%.+]] = krnl.load [[VAR_22_]]{{.}}[[VAR_29_2_]]#0, [[VAR_29_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_33_1_:%.+]] = addf [[LOAD_VAR_23_MEM_1_]], [[LOAD_VAR_24_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_14_MEM_2_:%.+]] = krnl.load [[VAR_14_]]#0{{.}}[[VAR_29_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_14_MEM_3_:%.+]] = krnl.load [[VAR_14_]]#3{{.}}[[VAR_29_2_]]#1] : memref<4xf32>
// CHECK:               [[VAR_36_1_:%.+]] = addf [[VAR_33_1_]], [[LOAD_VAR_14_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_37_1_:%.+]] = addf [[VAR_36_1_]], [[LOAD_VAR_14_MEM_3_]] : f32
// CHECK-DAG:           [[VAR_38_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_37_1_]], [[VAR_38_1_]][] : memref<f32>
// CHECK:               [[VAR_39_1_:%.+]] = "onnx.Sigmoid"([[VAR_38_1_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_39_MEM_1_:%.+]] = krnl.load [[VAR_39_1_]][] : memref<f32>
// CHECK-DAG:           [[VAR_41_1_:%.+]] = krnl.load [[VAR_25_]]{{.}}[[VAR_29_2_]]#0, [[VAR_29_2_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_27_MEM_:%.+]] = krnl.load [[VAR_27_]]{{.}}[[VAR_29_2_]]#0, [[VAR_29_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_43_:%.+]] = addf [[VAR_41_1_]], [[LOAD_VAR_27_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_14_MEM_4_:%.+]] = krnl.load [[VAR_14_]]#2{{.}}[[VAR_29_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_14_MEM_5_:%.+]] = krnl.load [[VAR_14_]]#5{{.}}[[VAR_29_2_]]#1] : memref<4xf32>
// CHECK:               [[VAR_46_:%.+]] = addf [[VAR_43_]], [[LOAD_VAR_14_MEM_4_]] : f32
// CHECK-DAG:           [[VAR_47_:%.+]] = addf [[VAR_46_]], [[LOAD_VAR_14_MEM_5_]] : f32
// CHECK-DAG:           [[VAR_48_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_47_]], [[VAR_48_]][] : memref<f32>
// CHECK:               [[VAR_49_:%.+]] = "onnx.Tanh"([[VAR_48_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_49_MEM_:%.+]] = krnl.load [[VAR_49_]][] : memref<f32>
// CHECK-DAG:           [[VAR_51_:%.+]] = subf [[CST_1_dot_000000_]], [[LOAD_VAR_39_MEM_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_52_:%.+]] = mulf [[VAR_51_]], [[LOAD_VAR_49_MEM_]] : f32
// CHECK-DAG:           [[VAR_53_:%.+]] = mulf [[LOAD_VAR_39_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_54_:%.+]] = addf [[VAR_52_]], [[VAR_53_]] : f32
// CHECK:               krnl.store [[VAR_54_]], [[VAR_0_]]{{.}}[[VAR_29_2_]]#0, [[VAR_29_2_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc [[VAR_17_]] : memref<2x4xf32>
// CHECK:             memref.dealloc [[LOAD_PARAM_4_MEM_1_]] : memref<2x4xf32>
// CHECK:             memref.dealloc [[VAR_18_]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           [[CST_32_:%.+]] = constant 32 : i64
// CHECK:           "krnl.memcpy"([[VAR_1_]], [[VAR_0_]], [[CST_32_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
// CHECK:           memref.dealloc [[VAR_0_]] : memref<2x4xf32>
// CHECK:           return [[VAR_1_]] : memref<1x2x4xf32>
// CHECK:         }
}

// -----

func private @test_gru_bidirectional_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<2x12x3xf32>, %arg2: tensor<2x12x4xf32>, %arg3: tensor<2x24xf32>, %arg4: tensor<2x2x4xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64, direction = "bidirectional"} : (tensor<7x2x3xf32>, tensor<2x12x3xf32>, tensor<2x12x4xf32>, tensor<2x24xf32>, none, tensor<2x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>
  
// CHECK-LABEL:  func private @test_gru_bidirectional_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<2x12x3xf32>, [[PARAM_2_:%.+]]: memref<2x12x4xf32>, [[PARAM_3_:%.+]]: memref<2x24xf32>, [[PARAM_4_:%.+]]: memref<2x2x4xf32>) -> memref<2x2x4xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.alloc() : memref<2x4xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.alloc() : memref<2x4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = memref.alloc() : memref<2x2x4xf32>
// CHECK-DAG:       [[VAR_cst_:%.+]] = constant unit
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = constant 1 : index
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4) {
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_]], [[I_0_]], [[I_1_]]{{.}} : memref<2x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[VAR_1_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:             [[LOAD_PARAM_4_MEM_1_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_1_]], [[I_0_]], [[I_1_]]{{.}} : memref<2x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_1_]], [[VAR_0_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:           }
// CHECK:           [[VAR_4_:%.+]]:2 = "onnx.Split"([[PARAM_1_]]) {axis = 0 : si64} : (memref<2x12x3xf32>) -> (memref<1x12x3xf32>, memref<1x12x3xf32>)
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Squeeze"([[VAR_4_]]#0) {axes = [0]} : (memref<1x12x3xf32>) -> memref<12x3xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Squeeze"([[VAR_4_]]#1) {axes = [0]} : (memref<1x12x3xf32>) -> memref<12x3xf32>
// CHECK-DAG:       [[VAR_7_:%.+]]:2 = "onnx.Split"([[PARAM_2_]]) {axis = 0 : si64} : (memref<2x12x4xf32>) -> (memref<1x12x4xf32>, memref<1x12x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Squeeze"([[VAR_7_]]#0) {axes = [0]} : (memref<1x12x4xf32>) -> memref<12x4xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Squeeze"([[VAR_7_]]#1) {axes = [0]} : (memref<1x12x4xf32>) -> memref<12x4xf32>
// CHECK-DAG:       [[VAR_10_:%.+]]:3 = "onnx.Split"([[VAR_5_]]) {axis = 0 : si64} : (memref<12x3xf32>) -> (memref<4x3xf32>, memref<4x3xf32>, memref<4x3xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Transpose"([[VAR_10_]]#0) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Transpose"([[VAR_10_]]#1) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Transpose"([[VAR_10_]]#2) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_14_:%.+]]:3 = "onnx.Split"([[VAR_8_]]) {axis = 0 : si64} : (memref<12x4xf32>) -> (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.Transpose"([[VAR_14_]]#0) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Transpose"([[VAR_14_]]#1) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Transpose"([[VAR_14_]]#2) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_18_:%.+]]:3 = "onnx.Split"([[VAR_6_]]) {axis = 0 : si64} : (memref<12x3xf32>) -> (memref<4x3xf32>, memref<4x3xf32>, memref<4x3xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.Transpose"([[VAR_18_]]#0) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.Transpose"([[VAR_18_]]#1) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.Transpose"([[VAR_18_]]#2) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_22_:%.+]]:3 = "onnx.Split"([[VAR_9_]]) {axis = 0 : si64} : (memref<12x4xf32>) -> (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_23_:%.+]] = "onnx.Transpose"([[VAR_22_]]#0) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_24_:%.+]] = "onnx.Transpose"([[VAR_22_]]#1) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_25_:%.+]] = "onnx.Transpose"([[VAR_22_]]#2) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_26_:%.+]]:2 = "onnx.Split"([[PARAM_3_]]) {axis = 0 : si64} : (memref<2x24xf32>) -> (memref<1x24xf32>, memref<1x24xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_27_:%.+]] = "onnx.Squeeze"([[VAR_26_]]#0) {axes = [0]} : (memref<1x24xf32>) -> memref<24xf32>
// CHECK-DAG:       [[VAR_28_:%.+]] = "onnx.Squeeze"([[VAR_26_]]#1) {axes = [0]} : (memref<1x24xf32>) -> memref<24xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_29_:%.+]]:6 = "onnx.Split"([[VAR_27_]]) {axis = 0 : si64} : (memref<24xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[VAR_30_:%.+]]:6 = "onnx.Split"([[VAR_28_]]) {axis = 0 : si64} : (memref<24xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7) {
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_2_:%.+]] = memref.alloc() : memref<2x4xf32>
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_1_:%.+]] = memref.alloc() : memref<2x4xf32>
// CHECK-DAG:         [[VAR_36_:%.+]] = memref.alloc() : memref<2x3xf32>
// CHECK-DAG:         [[CST_0_1_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_2_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_3_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_1_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_3_:%.+]] = constant 3 : index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_2_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_2_]] to [[CST_3_]]) {
// CHECK:               [[VAR_46_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_46_]]#0, [[VAR_46_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[VAR_36_]]{{.}}[[VAR_46_]]#0, [[VAR_46_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_38_:%.+]] = "onnx.MatMul"([[VAR_36_]], [[VAR_11_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_39_:%.+]] = "onnx.MatMul"([[VAR_1_]], [[VAR_1_]]5) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_40_:%.+]] = "onnx.MatMul"([[VAR_36_]], [[VAR_12_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_41_:%.+]] = "onnx.MatMul"([[VAR_1_]], [[VAR_1_]]6) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_42_:%.+]] = "onnx.MatMul"([[VAR_36_]], [[VAR_13_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[CST_1_dot_000000_:%.+]] = constant 1.000000e+00 : f32
// CHECK-DAG:         [[CST_0_4_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_5_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_1_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_2_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_4_]] to [[CST_2_1_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_4_]] to [[CST_4_]]) {
// CHECK:               [[VAR_46_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_1_]]{{.}}[[VAR_46_1_]]#0, [[VAR_46_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_40_MEM_:%.+]] = krnl.load [[VAR_40_]]{{.}}[[VAR_46_1_]]#0, [[VAR_46_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_41_MEM_:%.+]] = krnl.load [[VAR_41_]]{{.}}[[VAR_46_1_]]#0, [[VAR_46_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_50_:%.+]] = addf [[LOAD_VAR_40_MEM_]], [[LOAD_VAR_41_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_29_MEM_:%.+]] = krnl.load [[VAR_29_]]#1{{.}}[[VAR_46_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_29_MEM_1_:%.+]] = krnl.load [[VAR_29_]]#4{{.}}[[VAR_46_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_53_:%.+]] = addf [[VAR_50_]], [[LOAD_VAR_29_MEM_]] : f32
// CHECK-DAG:           [[VAR_54_:%.+]] = addf [[VAR_53_]], [[LOAD_VAR_29_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_55_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_54_]], [[VAR_55_]][] : memref<f32>
// CHECK:               [[VAR_56_:%.+]] = "onnx.Sigmoid"([[VAR_55_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_VAR_56_MEM_:%.+]] = krnl.load [[VAR_56_]][] : memref<f32>
// CHECK:               krnl.store [[LOAD_VAR_56_MEM_]], [[LOAD_PARAM_4_MEM_1_]]{{.}}[[VAR_46_1_]]#0, [[VAR_46_1_]]#1] : memref<2x4xf32>
// CHECK:               [[VAR_58_:%.+]] = mulf [[LOAD_VAR_56_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               krnl.store [[VAR_58_]], [[LOAD_PARAM_4_MEM_2_]]{{.}}[[VAR_46_1_]]#0, [[VAR_46_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_44_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_4_MEM_2_]], [[VAR_17_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_7_:%.+]] = [[CST_0_4_]] to [[CST_2_1_]], [[LOOP_4_]]#1 -> [[I_8_:%.+]] = [[CST_0_4_]] to [[CST_4_]]) {
// CHECK:               [[VAR_46_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_1_]]{{.}}[[VAR_46_2_]]#0, [[VAR_46_2_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_40_MEM_1_:%.+]] = krnl.load [[VAR_38_]]{{.}}[[VAR_46_2_]]#0, [[VAR_46_2_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_41_MEM_1_:%.+]] = krnl.load [[VAR_39_]]{{.}}[[VAR_46_2_]]#0, [[VAR_46_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_50_1_:%.+]] = addf [[LOAD_VAR_40_MEM_1_]], [[LOAD_VAR_41_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_29_MEM_2_:%.+]] = krnl.load [[VAR_29_]]#0{{.}}[[VAR_46_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_29_MEM_3_:%.+]] = krnl.load [[VAR_29_]]#3{{.}}[[VAR_46_2_]]#1] : memref<4xf32>
// CHECK:               [[VAR_53_1_:%.+]] = addf [[VAR_50_1_]], [[LOAD_VAR_29_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_54_1_:%.+]] = addf [[VAR_53_1_]], [[LOAD_VAR_29_MEM_3_]] : f32
// CHECK-DAG:           [[VAR_55_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_54_1_]], [[VAR_55_1_]][] : memref<f32>
// CHECK:               [[VAR_56_1_:%.+]] = "onnx.Sigmoid"([[VAR_55_1_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_56_MEM_1_:%.+]] = krnl.load [[VAR_56_1_]][] : memref<f32>
// CHECK-DAG:           [[VAR_58_1_:%.+]] = krnl.load [[VAR_42_]]{{.}}[[VAR_46_2_]]#0, [[VAR_46_2_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_44_MEM_:%.+]] = krnl.load [[VAR_44_]]{{.}}[[VAR_46_2_]]#0, [[VAR_46_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_60_:%.+]] = addf [[VAR_58_1_]], [[LOAD_VAR_44_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_29_MEM_4_:%.+]] = krnl.load [[VAR_29_]]#2{{.}}[[VAR_46_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_29_MEM_5_:%.+]] = krnl.load [[VAR_29_]]#5{{.}}[[VAR_46_2_]]#1] : memref<4xf32>
// CHECK:               [[VAR_63_:%.+]] = addf [[VAR_60_]], [[LOAD_VAR_29_MEM_4_]] : f32
// CHECK-DAG:           [[VAR_64_:%.+]] = addf [[VAR_63_]], [[LOAD_VAR_29_MEM_5_]] : f32
// CHECK-DAG:           [[VAR_65_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_64_]], [[VAR_65_]][] : memref<f32>
// CHECK:               [[VAR_66_:%.+]] = "onnx.Tanh"([[VAR_65_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_66_MEM_:%.+]] = krnl.load [[VAR_66_]][] : memref<f32>
// CHECK-DAG:           [[VAR_68_:%.+]] = subf [[CST_1_dot_000000_]], [[LOAD_VAR_56_MEM_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_69_:%.+]] = mulf [[VAR_68_]], [[LOAD_VAR_66_MEM_]] : f32
// CHECK-DAG:           [[VAR_70_:%.+]] = mulf [[LOAD_VAR_56_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_71_:%.+]] = addf [[VAR_69_]], [[VAR_70_]] : f32
// CHECK:               krnl.store [[VAR_71_]], [[VAR_1_]]{{.}}[[VAR_46_2_]]#0, [[VAR_46_2_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc [[LOAD_PARAM_4_MEM_1_]] : memref<2x4xf32>
// CHECK:             memref.dealloc [[LOAD_PARAM_4_MEM_2_]] : memref<2x4xf32>
// CHECK:             memref.dealloc [[VAR_36_]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_5_]]) with ([[LOOP_5_]] -> [[I_9_:%.+]] = 0 to 7) {
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_2_:%.+]] = memref.alloc() : memref<2x4xf32>
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_1_1_:%.+]] = memref.alloc() : memref<2x4xf32>
// CHECK-DAG:         [[VAR_36_1_:%.+]] = memref.alloc() : memref<2x3xf32>
// CHECK-DAG:         [[CST_1_3_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_7_:%.+]] = constant 7 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOOP_2_:%.+]] = affine.apply #map([[I_9_]]){{.}}[[CST_7_]]{{.}}
// CHECK-DAG:         [[CST_0_6_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_7_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_2_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_4_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_3_1_:%.+]] = constant 3 : index
// CHECK-DAG:         [[LOOP_6_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_6_]]#0, [[LOOP_6_]]#1) with ([[LOOP_6_]]#0 -> [[I_10_:%.+]] = [[CST_0_6_]] to [[CST_2_2_]], [[LOOP_6_]]#1 -> [[I_11_:%.+]] = [[CST_0_6_]] to [[CST_3_1_]]) {
// CHECK:               [[LOAD_PARAM_0_MEM_1_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_6_]]#0, [[LOOP_6_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[LOOP_2_]], [[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_2_]], [[VAR_36_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_39_1_:%.+]] = "onnx.MatMul"([[VAR_36_1_]], [[VAR_19_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_40_1_:%.+]] = "onnx.MatMul"([[VAR_0_]], [[VAR_23_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_41_1_:%.+]] = "onnx.MatMul"([[VAR_36_1_]], [[VAR_20_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_42_1_:%.+]] = "onnx.MatMul"([[VAR_0_]], [[VAR_24_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]] = "onnx.MatMul"([[VAR_36_1_]], [[VAR_21_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[CST_1_dot_000000_1_:%.+]] = constant 1.000000e+00 : f32
// CHECK-DAG:         [[CST_0_8_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_9_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_3_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_5_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_4_1_:%.+]] = constant 4 : index
// CHECK-DAG:         [[LOOP_7_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_7_]]#0, [[LOOP_7_]]#1) with ([[LOOP_7_]]#0 -> [[I_12_:%.+]] = [[CST_0_8_]] to [[CST_2_3_]], [[LOOP_7_]]#1 -> [[I_13_:%.+]] = [[CST_0_8_]] to [[CST_4_1_]]) {
// CHECK:               [[LOAD_PARAM_0_MEM_1_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_7_]]#0, [[LOOP_7_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_41_MEM_1_:%.+]] = krnl.load [[VAR_41_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[VAR_50_1_:%.+]] = krnl.load [[VAR_42_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_29_MEM_2_:%.+]] = addf [[LOAD_VAR_41_MEM_1_]], [[VAR_50_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_29_MEM_3_:%.+]] = krnl.load [[VAR_30_]]#1{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_53_1_:%.+]] = krnl.load [[VAR_30_]]#4{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_54_2_:%.+]] = addf [[LOAD_VAR_29_MEM_2_]], [[LOAD_VAR_29_MEM_3_]] : f32
// CHECK-DAG:           [[VAR_55_2_:%.+]] = addf [[VAR_54_2_]], [[VAR_53_1_]] : f32
// CHECK-DAG:           [[VAR_56_2_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_55_2_]], [[VAR_56_2_]][] : memref<f32>
// CHECK:               [[LOAD_VAR_56_MEM_1_:%.+]] = "onnx.Sigmoid"([[VAR_56_2_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[VAR_58_1_:%.+]] = krnl.load [[LOAD_VAR_56_MEM_1_]][] : memref<f32>
// CHECK:               krnl.store [[VAR_58_1_]], [[LOAD_PARAM_4_MEM_1_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK:               [[LOAD_VAR_44_MEM_1_:%.+]] = mulf [[VAR_58_1_]], [[LOAD_PARAM_0_MEM_2_]] : f32
// CHECK:               krnl.store [[LOAD_VAR_44_MEM_1_]], [[LOAD_PARAM_4_MEM_2_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK-DAG:         [[LOOP_4_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_4_MEM_2_]], [[VAR_25_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[LOOP_8_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_8_]]#0, [[LOOP_8_]]#1) with ([[LOOP_8_]]#0 -> [[I_14_:%.+]] = [[CST_0_8_]] to [[CST_2_3_]], [[LOOP_8_]]#1 -> [[I_15_:%.+]] = [[CST_0_8_]] to [[CST_4_1_]]) {
// CHECK:               [[LOAD_PARAM_0_MEM_1_1_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_8_]]#0, [[LOOP_8_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_2_1_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_41_MEM_1_1_:%.+]] = krnl.load [[VAR_39_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[VAR_50_1_1_:%.+]] = krnl.load [[VAR_40_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_29_MEM_2_1_:%.+]] = addf [[LOAD_VAR_41_MEM_1_1_]], [[VAR_50_1_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_29_MEM_3_1_:%.+]] = krnl.load [[VAR_30_]]#0{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_53_1_1_:%.+]] = krnl.load [[VAR_30_]]#3{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_54_3_:%.+]] = addf [[LOAD_VAR_29_MEM_2_1_]], [[LOAD_VAR_29_MEM_3_1_]] : f32
// CHECK-DAG:           [[VAR_55_3_:%.+]] = addf [[VAR_54_3_]], [[VAR_53_1_1_]] : f32
// CHECK-DAG:           [[VAR_56_3_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_55_3_]], [[VAR_56_3_]][] : memref<f32>
// CHECK:               [[LOAD_VAR_56_MEM_1_1_:%.+]] = "onnx.Sigmoid"([[VAR_56_3_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[VAR_58_1_1_:%.+]] = krnl.load [[LOAD_VAR_56_MEM_1_1_]][] : memref<f32>
// CHECK-DAG:           [[LOAD_VAR_44_MEM_1_:%.+]] = krnl.load [[LOOP_3_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[VAR_60_1_:%.+]] = krnl.load [[LOOP_4_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_29_MEM_4_:%.+]] = addf [[LOAD_VAR_44_MEM_1_]], [[VAR_60_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_29_MEM_5_:%.+]] = krnl.load [[VAR_30_]]#2{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_63_1_:%.+]] = krnl.load [[VAR_30_]]#5{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_64_1_:%.+]] = addf [[LOAD_VAR_29_MEM_4_]], [[LOAD_VAR_29_MEM_5_]] : f32
// CHECK-DAG:           [[VAR_65_1_:%.+]] = addf [[VAR_64_1_]], [[VAR_63_1_]] : f32
// CHECK-DAG:           [[VAR_66_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_65_1_]], [[VAR_66_1_]][] : memref<f32>
// CHECK:               [[LOAD_VAR_66_MEM_1_:%.+]] = "onnx.Tanh"([[VAR_66_1_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[VAR_68_1_:%.+]] = krnl.load [[LOAD_VAR_66_MEM_1_]][] : memref<f32>
// CHECK-DAG:           [[VAR_69_1_:%.+]] = subf [[CST_1_dot_000000_1_]], [[VAR_58_1_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_70_1_:%.+]] = mulf [[VAR_69_1_]], [[VAR_68_1_]] : f32
// CHECK-DAG:           [[VAR_71_1_:%.+]] = mulf [[VAR_58_1_1_]], [[LOAD_PARAM_0_MEM_2_1_]] : f32
// CHECK:               [[VAR_72_:%.+]] = addf [[VAR_70_1_]], [[VAR_71_1_]] : f32
// CHECK:               krnl.store [[VAR_72_]], [[VAR_0_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc [[LOAD_PARAM_4_MEM_1_1_]] : memref<2x4xf32>
// CHECK:             memref.dealloc [[LOAD_PARAM_4_MEM_2_]] : memref<2x4xf32>
// CHECK:             memref.dealloc [[VAR_36_1_]] : memref<2x3xf32>
// CHECK:           }
// CHECK-DAG:       [[CST_0_10_:%.+]] = constant 0 : index
// CHECK-DAG:       [[CST_1_6_:%.+]] = constant 1 : index
// CHECK-DAG:       [[CST_0_11_:%.+]] = constant 0 : index
// CHECK-DAG:       [[CST_2_4_:%.+]] = constant 2 : index
// CHECK-DAG:       [[CST_1_7_:%.+]] = constant 1 : index
// CHECK-DAG:       [[CST_4_2_:%.+]] = constant 4 : index
// CHECK-DAG:       [[LOOP_9_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_9_]]#0, [[LOOP_9_]]#1) with ([[LOOP_9_]]#0 -> [[I_16_:%.+]] = [[CST_0_10_]] to [[CST_2_4_]], [[LOOP_9_]]#1 -> [[I_17_:%.+]] = [[CST_0_10_]] to [[CST_4_2_]]) {
// CHECK:             [[LOAD_PARAM_4_MEM_2_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_9_]]#0, [[LOOP_9_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_4_MEM_1_1_:%.+]] = krnl.load [[VAR_1_]]{{.}}[[LOAD_PARAM_4_MEM_2_1_]]#0, [[LOAD_PARAM_4_MEM_2_1_]]#1] : memref<2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_1_1_]], [[VAR_2_]]{{.}}[[CST_0_10_]], [[LOAD_PARAM_4_MEM_2_1_]]#0, [[LOAD_PARAM_4_MEM_2_1_]]#1] : memref<2x2x4xf32>
// CHECK:             [[VAR_36_1_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[LOAD_PARAM_4_MEM_2_1_]]#0, [[LOAD_PARAM_4_MEM_2_1_]]#1] : memref<2x4xf32>
// CHECK:             krnl.store [[VAR_36_1_]], [[VAR_2_]]{{.}}[[CST_1_6_]], [[LOAD_PARAM_4_MEM_2_1_]]#0, [[LOAD_PARAM_4_MEM_2_1_]]#1] : memref<2x2x4xf32>
// CHECK:           }
// CHECK:           memref.dealloc [[VAR_1_]] : memref<2x4xf32>
// CHECK:           memref.dealloc [[VAR_0_]] : memref<2x4xf32>
// CHECK:           return [[VAR_2_]] : memref<2x2x4xf32>
// CHECK:         }
}

// -----

func private @test_gru_unknown_dims(%arg0: tensor<?x?x?xf32>, %arg1: tensor<1x12x?xf32>, %arg2: tensor<1x12x4xf32>, %arg3: tensor<1x24xf32>, %arg4: tensor<1x?x4xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64} : (tensor<?x?x?xf32>, tensor<1x12x?xf32>, tensor<1x12x4xf32>, tensor<1x24xf32>, none, tensor<1x?x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>
  
// CHECK-LABEL:  func private @test_gru_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x?xf32>, [[PARAM_1_:%.+]]: memref<1x12x?xf32>, [[PARAM_2_:%.+]]: memref<1x12x4xf32>, [[PARAM_3_:%.+]]: memref<1x24xf32>, [[PARAM_4_:%.+]]: memref<1x?x4xf32>) -> memref<1x?x4xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = constant unit
// CHECK-DAG:       [[CST_1_:%.+]] = constant 1 : index
// CHECK:           [[VAR_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.alloc([[VAR_0_]]) : memref<1x?x4xf32>
// CHECK-DAG:       [[CST_1_1_:%.+]] = constant 1 : index
// CHECK:           [[VAR_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = memref.alloc([[VAR_2_]]) : memref<?x4xf32>
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK-DAG:       [[CST_1_2_:%.+]] = constant 1 : index
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[CST_0_1_:%.+]] = constant 0 : index
// CHECK:           [[VAR_5_:%.+]] = memref.dim [[VAR_3_]], [[CST_0_1_]] : memref<?x4xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[VAR_5_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4) {
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x?x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[VAR_3_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<?x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Squeeze"([[PARAM_1_]]) {axes = [0]} : (memref<1x12x?xf32>) -> memref<12x?xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Squeeze"([[PARAM_2_]]) {axes = [0]} : (memref<1x12x4xf32>) -> memref<12x4xf32>
// CHECK:           [[VAR_8_:%.+]]:3 = "onnx.Split"([[VAR_6_]]) {axis = 0 : si64} : (memref<12x?xf32>) -> (memref<4x?xf32>, memref<4x?xf32>, memref<4x?xf32>)
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Transpose"([[VAR_8_]]#0) {perm = [1, 0]} : (memref<4x?xf32>) -> memref<?x4xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Transpose"([[VAR_8_]]#1) {perm = [1, 0]} : (memref<4x?xf32>) -> memref<?x4xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Transpose"([[VAR_8_]]#2) {perm = [1, 0]} : (memref<4x?xf32>) -> memref<?x4xf32>
// CHECK-DAG:       [[VAR_12_:%.+]]:3 = "onnx.Split"([[VAR_7_]]) {axis = 0 : si64} : (memref<12x4xf32>) -> (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Transpose"([[VAR_12_]]#0) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.Transpose"([[VAR_12_]]#1) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.Transpose"([[VAR_12_]]#2) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Squeeze"([[PARAM_3_]]) {axes = [0]} : (memref<1x24xf32>) -> memref<24xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_17_:%.+]]:6 = "onnx.Split"([[VAR_16_]]) {axis = 0 : si64} : (memref<24xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK-DAG:       [[CST_0_2_:%.+]] = constant 0 : index
// CHECK:           [[VAR_19_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_2_]] : memref<?x?x?xf32>
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to [[VAR_19_]]) {
// CHECK-DAG:         [[CST_0_3_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_1_3_:%.+]] = constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_3_]] : memref<?x?x?xf32>
// CHECK-DAG:         [[CST_2_:%.+]] = constant 2 : index
// CHECK:             [[VAR_24_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<?x?x?xf32>
// CHECK-DAG:         [[VAR_25_:%.+]] = memref.alloc([[LOAD_PARAM_4_MEM_1_]], [[VAR_24_]]) : memref<?x?xf32>
// CHECK-DAG:         [[CST_0_4_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_5_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_1_4_:%.+]] = constant 1 : index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_4_]] to [[LOAD_PARAM_4_MEM_1_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_4_]] to [[VAR_24_]]) {
// CHECK:               [[VAR_38_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_38_]]#0, [[VAR_38_]]#1] : memref<?x?x?xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[VAR_25_]]{{.}}[[VAR_38_]]#0, [[VAR_38_]]#1] : memref<?x?xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_27_:%.+]] = "onnx.MatMul"([[VAR_25_]], [[VAR_9_]]) : (memref<?x?xf32>, memref<?x4xf32>) -> memref<?x4xf32>
// CHECK-DAG:         [[VAR_28_:%.+]] = "onnx.MatMul"([[VAR_3_]], [[VAR_13_]]) : (memref<?x4xf32>, memref<4x4xf32>) -> memref<?x4xf32>
// CHECK-DAG:         [[VAR_29_:%.+]] = "onnx.MatMul"([[VAR_25_]], [[VAR_10_]]) : (memref<?x?xf32>, memref<?x4xf32>) -> memref<?x4xf32>
// CHECK-DAG:         [[VAR_30_:%.+]] = "onnx.MatMul"([[VAR_3_]], [[VAR_14_]]) : (memref<?x4xf32>, memref<4x4xf32>) -> memref<?x4xf32>
// CHECK-DAG:         [[VAR_31_:%.+]] = "onnx.MatMul"([[VAR_25_]], [[VAR_11_]]) : (memref<?x?xf32>, memref<?x4xf32>) -> memref<?x4xf32>
// CHECK-DAG:         [[CST_1_dot_000000_:%.+]] = constant 1.000000e+00 : f32
// CHECK-DAG:         [[CST_0_6_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_7_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_1_5_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:         [[CST_0_8_:%.+]] = constant 0 : index
// CHECK:             [[VAR_32_:%.+]] = memref.dim [[VAR_3_]], [[CST_0_8_]] : memref<?x4xf32>
// CHECK-DAG:         [[VAR_33_:%.+]] = memref.alloc([[VAR_32_]]) : memref<?x4xf32>
// CHECK-DAG:         [[VAR_34_:%.+]] = memref.alloc([[VAR_32_]]) : memref<?x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_6_]] to [[VAR_2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_6_]] to [[CST_4_]]) {
// CHECK:               [[VAR_38_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_3_]]{{.}}[[VAR_3_]]8#0, [[VAR_3_]]8#1] : memref<?x4xf32>
// CHECK-DAG:           [[LOAD_VAR_29_MEM_:%.+]] = krnl.load [[VAR_29_]]{{.}}[[VAR_38_1_]]#0, [[VAR_38_1_]]#1] : memref<?x4xf32>
// CHECK-DAG:           [[LOAD_VAR_30_MEM_:%.+]] = krnl.load [[VAR_30_]]{{.}}[[VAR_38_1_]]#0, [[VAR_38_1_]]#1] : memref<?x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_42_:%.+]] = addf [[LOAD_VAR_29_MEM_]], [[LOAD_VAR_30_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_17_MEM_:%.+]] = krnl.load [[VAR_17_]]#1{{.}}[[VAR_38_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_17_MEM_1_:%.+]] = krnl.load [[VAR_17_]]#4{{.}}[[VAR_38_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_45_:%.+]] = addf [[VAR_42_]], [[LOAD_VAR_17_MEM_]] : f32
// CHECK-DAG:           [[VAR_46_:%.+]] = addf [[VAR_45_]], [[LOAD_VAR_17_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_47_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_46_]], [[VAR_47_]][] : memref<f32>
// CHECK:               [[VAR_48_:%.+]] = "onnx.Sigmoid"([[VAR_47_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_VAR_48_MEM_:%.+]] = krnl.load [[VAR_48_]][] : memref<f32>
// CHECK:               krnl.store [[LOAD_VAR_48_MEM_]], [[VAR_33_]]{{.}}[[VAR_38_1_]]#0, [[VAR_38_1_]]#1] : memref<?x4xf32>
// CHECK:               [[VAR_50_:%.+]] = mulf [[LOAD_VAR_48_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               krnl.store [[VAR_50_]], [[VAR_34_]]{{.}}[[VAR_38_1_]]#0, [[VAR_38_1_]]#1] : memref<?x4xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_36_:%.+]] = "onnx.MatMul"([[VAR_34_]], [[VAR_15_]]) : (memref<?x4xf32>, memref<4x4xf32>) -> memref<?x4xf32>
// CHECK-DAG:         [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_7_:%.+]] = [[CST_0_6_]] to [[VAR_2_]], [[LOOP_4_]]#1 -> [[I_8_:%.+]] = [[CST_0_6_]] to [[CST_4_]]) {
// CHECK:               [[VAR_38_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_3_]]{{.}}[[VAR_3_]]8#0, [[VAR_3_]]8#1] : memref<?x4xf32>
// CHECK-DAG:           [[LOAD_VAR_29_MEM_1_:%.+]] = krnl.load [[VAR_27_]]{{.}}[[VAR_38_2_]]#0, [[VAR_38_2_]]#1] : memref<?x4xf32>
// CHECK-DAG:           [[LOAD_VAR_30_MEM_1_:%.+]] = krnl.load [[VAR_28_]]{{.}}[[VAR_38_2_]]#0, [[VAR_38_2_]]#1] : memref<?x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_42_1_:%.+]] = addf [[LOAD_VAR_29_MEM_1_]], [[LOAD_VAR_30_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_17_MEM_2_:%.+]] = krnl.load [[VAR_17_]]#0{{.}}[[VAR_38_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_17_MEM_3_:%.+]] = krnl.load [[VAR_17_]]#3{{.}}[[VAR_38_2_]]#1] : memref<4xf32>
// CHECK:               [[VAR_45_1_:%.+]] = addf [[VAR_42_1_]], [[LOAD_VAR_17_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_46_1_:%.+]] = addf [[VAR_45_1_]], [[LOAD_VAR_17_MEM_3_]] : f32
// CHECK-DAG:           [[VAR_47_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_46_1_]], [[VAR_47_1_]][] : memref<f32>
// CHECK:               [[VAR_48_1_:%.+]] = "onnx.Sigmoid"([[VAR_47_1_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_48_MEM_1_:%.+]] = krnl.load [[VAR_48_1_]][] : memref<f32>
// CHECK-DAG:           [[VAR_50_1_:%.+]] = krnl.load [[VAR_31_]]{{.}}[[VAR_38_2_]]#0, [[VAR_38_2_]]#1] : memref<?x4xf32>
// CHECK-DAG:           [[LOAD_VAR_36_MEM_:%.+]] = krnl.load [[VAR_36_]]{{.}}[[VAR_38_2_]]#0, [[VAR_38_2_]]#1] : memref<?x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_52_:%.+]] = addf [[VAR_50_1_]], [[LOAD_VAR_36_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_17_MEM_4_:%.+]] = krnl.load [[VAR_17_]]#2{{.}}[[VAR_38_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_17_MEM_5_:%.+]] = krnl.load [[VAR_17_]]#5{{.}}[[VAR_38_2_]]#1] : memref<4xf32>
// CHECK:               [[VAR_55_:%.+]] = addf [[VAR_52_]], [[LOAD_VAR_17_MEM_4_]] : f32
// CHECK-DAG:           [[VAR_56_:%.+]] = addf [[VAR_55_]], [[LOAD_VAR_17_MEM_5_]] : f32
// CHECK-DAG:           [[VAR_57_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_56_]], [[VAR_57_]][] : memref<f32>
// CHECK:               [[VAR_58_:%.+]] = "onnx.Tanh"([[VAR_57_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_58_MEM_:%.+]] = krnl.load [[VAR_58_]][] : memref<f32>
// CHECK-DAG:           [[VAR_60_:%.+]] = subf [[CST_1_dot_000000_]], [[LOAD_VAR_48_MEM_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_61_:%.+]] = mulf [[VAR_60_]], [[LOAD_VAR_58_MEM_]] : f32
// CHECK-DAG:           [[VAR_62_:%.+]] = mulf [[LOAD_VAR_48_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_63_:%.+]] = addf [[VAR_61_]], [[VAR_62_]] : f32
// CHECK:               krnl.store [[VAR_63_]], [[VAR_3_]]{{.}}[[VAR_3_]]8#0, [[VAR_3_]]8#1] : memref<?x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc [[VAR_33_]] : memref<?x4xf32>
// CHECK:             memref.dealloc [[VAR_34_]] : memref<?x4xf32>
// CHECK:             memref.dealloc [[VAR_25_]] : memref<?x?xf32>
// CHECK:           }
// CHECK-DAG:       [[CST_16_:%.+]] = constant 16 : i64
// CHECK-DAG:       [[CST_0_9_:%.+]] = constant 0 : index
// CHECK:           [[VAR_20_:%.+]] = memref.dim [[VAR_3_]], [[CST_0_9_]] : memref<?x4xf32>
// CHECK:           [[VAR_21_:%.+]] = index_cast [[VAR_20_]] : index to i64
// CHECK:           [[VAR_22_:%.+]] = muli [[CST_16_]], [[VAR_21_]] : i64
// CHECK:           "krnl.memcpy"([[VAR_1_]], [[VAR_3_]], [[VAR_22_]]) : (memref<1x?x4xf32>, memref<?x4xf32>, i64) -> ()
// CHECK:           memref.dealloc [[VAR_3_]] : memref<?x4xf32>
// CHECK:           return [[VAR_1_]] : memref<1x?x4xf32>
// CHECK:         }
}
