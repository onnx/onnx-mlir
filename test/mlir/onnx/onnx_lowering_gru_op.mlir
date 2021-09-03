// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl='check-rnn-ops-lowering' %s -split-input-file | FileCheck %s

func private @test_gru_forward_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x12x3xf32>, %arg2: tensor<1x12x4xf32>, %arg3: tensor<1x24xf32>, %arg4: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64} : (tensor<7x2x3xf32>, tensor<1x12x3xf32>, tensor<1x12x4xf32>, tensor<1x24xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:  func private @test_gru_forward_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x12x3xf32>, [[PARAM_2_:%.+]]: memref<1x12x4xf32>, [[PARAM_3_:%.+]]: memref<1x24xf32>, [[PARAM_4_:%.+]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.alloc() {{.*}}: memref<1x2x4xf32>
// CHECK-DAG:       [[VAR_cst_:%.+]] = constant unit
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = constant 1 : index
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4) {
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[VAR_0_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.SqueezeV11"([[PARAM_1_]]) {axes = [0]} : (memref<1x12x3xf32>) -> memref<12x3xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.SqueezeV11"([[PARAM_2_]]) {axes = [0]} : (memref<1x12x4xf32>) -> memref<12x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Transpose"([[VAR_3_]]) {perm = [1, 0]} : (memref<12x3xf32>) -> memref<3x12xf32>
// CHECK-DAG:       [[VAR_6_:%.+]]:3 = "onnx.Split"([[VAR_4_]]) {axis = 0 : si64} : (memref<12x4xf32>) -> (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Transpose"([[VAR_6_]]#0) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Transpose"([[VAR_6_]]#1) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Transpose"([[VAR_6_]]#2) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.SqueezeV11"([[PARAM_3_]]) {axes = [0]} : (memref<1x24xf32>) -> memref<24xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]]:6 = "onnx.Split"([[VAR_10_]]) {axis = 0 : si64} : (memref<24xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7) {
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[VAR_14_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[VAR_15_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[CST_0_1_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_2_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_3_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_1_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_3_:%.+]] = constant 3 : index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_2_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_2_]] to [[CST_3_]]) {
// CHECK:               [[VAR_23_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_23_]]#0, [[VAR_23_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[VAR_15_]]{{.}}[[VAR_23_]]#0, [[VAR_23_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_17_:%.+]] = "onnx.MatMul"([[VAR_15_]], [[VAR_5_]]) : (memref<2x3xf32>, memref<3x12xf32>) -> memref<2x12xf32>
// CHECK-DAG:         [[CST_1_dot_000000_:%.+]] = constant 1.000000e+00 : f32
// CHECK-DAG:         [[CST_0_4_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_5_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_1_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_2_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:         [[VAR_18_:%.+]] = "onnx.MatMul"([[VAR_0_]], [[VAR_7_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_19_:%.+]] = "onnx.MatMul"([[VAR_0_]], [[VAR_8_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_4_]] to [[CST_2_1_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_4_]] to [[CST_4_]]) {
// CHECK-DAG:           [[VAR_23_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[CST_4_1_:%.+]] = constant 4 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[VAR_23_1_]]#0, [[VAR_23_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[VAR_25_:%.+]] = affine.apply #map0(){{.}}[[VAR_23_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_17_MEM_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_23_1_]]#0, [[VAR_25_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_19_MEM_:%.+]] = krnl.load [[VAR_19_]]{{.}}[[VAR_23_1_]]#0, [[VAR_23_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_28_:%.+]] = addf [[LOAD_VAR_17_MEM_]], [[LOAD_VAR_19_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_:%.+]] = krnl.load [[VAR_11_]]#1{{.}}[[VAR_23_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_11_MEM_1_:%.+]] = krnl.load [[VAR_11_]]#4{{.}}[[VAR_23_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_31_:%.+]] = addf [[VAR_28_]], [[LOAD_VAR_11_MEM_]] : f32
// CHECK-DAG:           [[VAR_32_:%.+]] = addf [[VAR_31_]], [[LOAD_VAR_11_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_33_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_32_]], [[VAR_33_]][] : memref<f32>
// CHECK:               [[VAR_34_:%.+]] = "onnx.Sigmoid"([[VAR_33_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_VAR_34_MEM_:%.+]] = krnl.load [[VAR_34_]][] : memref<f32>
// CHECK:               krnl.store [[LOAD_VAR_34_MEM_]], [[VAR_14_]]{{.}}[[VAR_23_1_]]#0, [[VAR_23_1_]]#1] : memref<2x4xf32>
// CHECK:               [[VAR_36_:%.+]] = mulf [[LOAD_VAR_34_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               krnl.store [[VAR_36_]], [[LOAD_PARAM_4_MEM_1_]]{{.}}[[VAR_23_1_]]#0, [[VAR_23_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_21_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_4_MEM_1_]], [[VAR_9_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_7_:%.+]] = [[CST_0_4_]] to [[CST_2_1_]], [[LOOP_4_]]#1 -> [[I_8_:%.+]] = [[CST_0_4_]] to [[CST_4_]]) {
// CHECK-DAG:           [[VAR_23_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[CST_4_2_:%.+]] = constant 4 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_17_MEM_1_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1] : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_17_MEM_2_:%.+]] = krnl.load [[VAR_18_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_19_MEM_1_:%.+]] = addf [[LOAD_VAR_17_MEM_1_]], [[LOAD_VAR_17_MEM_2_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_2_:%.+]] = krnl.load [[VAR_11_]]#0{{.}}[[VAR_23_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_11_MEM_3_:%.+]] = krnl.load [[VAR_11_]]#3{{.}}[[VAR_23_2_]]#1] : memref<4xf32>
// CHECK:               [[LOAD_VAR_11_MEM_1_:%.+]] = addf [[LOAD_VAR_19_MEM_1_]], [[LOAD_VAR_11_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_31_1_:%.+]] = addf [[LOAD_VAR_11_MEM_1_]], [[LOAD_VAR_11_MEM_3_]] : f32
// CHECK-DAG:           [[VAR_32_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_31_1_]], [[VAR_32_1_]][] : memref<f32>
// CHECK:               [[VAR_33_1_:%.+]] = "onnx.Sigmoid"([[VAR_32_1_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[VAR_34_1_:%.+]] = krnl.load [[VAR_33_1_]][] : memref<f32>
// CHECK-DAG:           [[CST_2_2_:%.+]] = constant 2 : index
// CHECK-DAG:           [[CST_8_:%.+]] = constant 8 : index
// CHECK-DAG:           [[LOAD_VAR_34_MEM_1_:%.+]] = affine.apply #map1(){{.}}[[VAR_23_2_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_17_MEM_3_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_23_2_]]#0, [[LOAD_VAR_34_MEM_1_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_21_MEM_:%.+]] = krnl.load [[VAR_21_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_38_:%.+]] = addf [[LOAD_VAR_17_MEM_3_]], [[LOAD_VAR_21_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_4_:%.+]] = krnl.load [[VAR_11_]]#2{{.}}[[VAR_23_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_11_MEM_5_:%.+]] = krnl.load [[VAR_11_]]#5{{.}}[[VAR_23_2_]]#1] : memref<4xf32>
// CHECK:               [[VAR_41_:%.+]] = addf [[VAR_38_]], [[LOAD_VAR_11_MEM_4_]] : f32
// CHECK-DAG:           [[VAR_42_:%.+]] = addf [[VAR_41_]], [[LOAD_VAR_11_MEM_5_]] : f32
// CHECK-DAG:           [[VAR_43_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_42_]], [[VAR_43_]][] : memref<f32>
// CHECK:               [[VAR_44_:%.+]] = "onnx.Tanh"([[VAR_43_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_44_MEM_:%.+]] = krnl.load [[VAR_44_]][] : memref<f32>
// CHECK-DAG:           [[VAR_46_:%.+]] = subf [[CST_1_dot_000000_]], [[VAR_34_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_47_:%.+]] = mulf [[VAR_46_]], [[LOAD_VAR_44_MEM_]] : f32
// CHECK-DAG:           [[VAR_48_:%.+]] = mulf [[VAR_34_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_49_:%.+]] = addf [[VAR_47_]], [[VAR_48_]] : f32
// CHECK:               krnl.store [[VAR_49_]], [[VAR_0_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc [[VAR_14_]] : memref<2x4xf32>
// CHECK:             memref.dealloc [[LOAD_PARAM_4_MEM_1_]] : memref<2x4xf32>
// CHECK:             memref.dealloc [[VAR_15_]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           [[CST_32_:%.+]] = constant 32 : i64
// CHECK:           "krnl.memcpy"([[VAR_1_]], [[VAR_0_]], [[CST_32_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
// CHECK:           memref.dealloc [[VAR_0_]] : memref<2x4xf32>
// CHECK:           return [[VAR_1_]] : memref<1x2x4xf32>
}

// -----

func private @test_gru_forward_mode_linear_before_reset(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x12x3xf32>, %arg2: tensor<1x12x4xf32>, %arg3: tensor<1x24xf32>, %arg4: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64, linear_before_reset = 1 : si64} : (tensor<7x2x3xf32>, tensor<1x12x3xf32>, tensor<1x12x4xf32>, tensor<1x24xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>
  
// CHECK-LABEL:  func private @test_gru_forward_mode_linear_before_reset
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x12x3xf32>, [[PARAM_2_:%.+]]: memref<1x12x4xf32>, [[PARAM_3_:%.+]]: memref<1x24xf32>, [[PARAM_4_:%.+]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.alloc() {{.*}}: memref<1x2x4xf32>
// CHECK-DAG:       [[VAR_cst_:%.+]] = constant unit
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = constant 1 : index
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4) {
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[VAR_0_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.SqueezeV11"([[PARAM_1_]]) {axes = [0]} : (memref<1x12x3xf32>) -> memref<12x3xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.SqueezeV11"([[PARAM_2_]]) {axes = [0]} : (memref<1x12x4xf32>) -> memref<12x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Transpose"([[VAR_3_]]) {perm = [1, 0]} : (memref<12x3xf32>) -> memref<3x12xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Transpose"([[VAR_4_]]) {perm = [1, 0]} : (memref<12x4xf32>) -> memref<4x12xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.SqueezeV11"([[PARAM_3_]]) {axes = [0]} : (memref<1x24xf32>) -> memref<24xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]]:6 = "onnx.Split"([[VAR_7_]]) {axis = 0 : si64} : (memref<24xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7) {
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_1_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[CST_0_1_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_2_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_3_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_1_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_3_:%.+]] = constant 3 : index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_2_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_2_]] to [[CST_3_]]) {
// CHECK:               [[VAR_15_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_15_]]#0, [[VAR_15_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_4_MEM_1_]]{{.}}[[VAR_15_]]#0, [[VAR_15_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_12_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_4_MEM_1_]], [[VAR_5_]]) : (memref<2x3xf32>, memref<3x12xf32>) -> memref<2x12xf32>
// CHECK-DAG:         [[CST_1_dot_000000_:%.+]] = constant 1.000000e+00 : f32
// CHECK-DAG:         [[CST_0_4_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_5_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_1_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_2_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:         [[VAR_13_:%.+]] = "onnx.MatMul"([[VAR_0_]], [[VAR_6_]]) : (memref<2x4xf32>, memref<4x12xf32>) -> memref<2x12xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_4_]] to [[CST_2_1_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_4_]] to [[CST_4_]]) {
// CHECK-DAG:           [[VAR_15_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[CST_4_1_:%.+]] = constant 4 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[VAR_15_1_]]#0, [[VAR_15_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_12_MEM_:%.+]] = krnl.load [[VAR_12_]]{{.}}[[VAR_15_1_]]#0, [[VAR_15_1_]]#1] : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_13_MEM_:%.+]] = krnl.load [[VAR_13_]]{{.}}[[VAR_15_1_]]#0, [[VAR_15_1_]]#1] : memref<2x12xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_19_:%.+]] = addf [[LOAD_VAR_12_MEM_]], [[LOAD_VAR_13_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_8_MEM_:%.+]] = krnl.load [[VAR_8_]]#0{{.}}[[VAR_15_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_8_MEM_1_:%.+]] = krnl.load [[VAR_8_]]#3{{.}}[[VAR_15_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_22_:%.+]] = addf [[VAR_19_]], [[LOAD_VAR_8_MEM_]] : f32
// CHECK-DAG:           [[VAR_23_:%.+]] = addf [[VAR_22_]], [[LOAD_VAR_8_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_24_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_23_]], [[VAR_24_]][] : memref<f32>
// CHECK:               [[VAR_25_:%.+]] = "onnx.Sigmoid"([[VAR_24_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_25_MEM_:%.+]] = krnl.load [[VAR_25_]][] : memref<f32>
// CHECK-DAG:           [[VAR_27_:%.+]] = affine.apply #map0(){{.}}[[VAR_15_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_12_MEM_1_:%.+]] = krnl.load [[VAR_12_]]{{.}}[[VAR_15_1_]]#0, [[VAR_27_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[VAR_29_:%.+]] = affine.apply #map0(){{.}}[[VAR_15_1_]]#1]
// CHECK:               [[LOAD_VAR_13_MEM_1_:%.+]] = krnl.load [[VAR_13_]]{{.}}[[VAR_15_1_]]#0, [[VAR_29_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[VAR_31_:%.+]] = addf [[LOAD_VAR_12_MEM_1_]], [[LOAD_VAR_13_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_8_MEM_2_:%.+]] = krnl.load [[VAR_8_]]#1{{.}}[[VAR_15_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_8_MEM_3_:%.+]] = krnl.load [[VAR_8_]]#4{{.}}[[VAR_15_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_34_:%.+]] = addf [[VAR_31_]], [[LOAD_VAR_8_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_35_:%.+]] = addf [[VAR_34_]], [[LOAD_VAR_8_MEM_3_]] : f32
// CHECK-DAG:           [[VAR_36_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_35_]], [[VAR_36_]][] : memref<f32>
// CHECK:               [[VAR_37_:%.+]] = "onnx.Sigmoid"([[VAR_36_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_37_MEM_:%.+]] = krnl.load [[VAR_37_]][] : memref<f32>
// CHECK-DAG:           [[CST_2_2_:%.+]] = constant 2 : index
// CHECK-DAG:           [[CST_8_:%.+]] = constant 8 : index
// CHECK-DAG:           [[VAR_39_:%.+]] = affine.apply #map1(){{.}}[[VAR_15_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_12_MEM_2_:%.+]] = krnl.load [[VAR_12_]]{{.}}[[VAR_15_1_]]#0, [[VAR_39_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[CST_2_3_:%.+]] = constant 2 : index
// CHECK-DAG:           [[CST_8_1_:%.+]] = constant 8 : index
// CHECK-DAG:           [[VAR_41_:%.+]] = affine.apply #map1(){{.}}[[VAR_15_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_13_MEM_2_:%.+]] = krnl.load [[VAR_13_]]{{.}}[[VAR_15_1_]]#0, [[VAR_41_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_8_MEM_4_:%.+]] = krnl.load [[VAR_8_]]#5{{.}}[[VAR_15_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_44_:%.+]] = addf [[LOAD_VAR_13_MEM_2_]], [[LOAD_VAR_8_MEM_4_]] : f32
// CHECK:               [[VAR_45_:%.+]] = mulf [[LOAD_VAR_37_MEM_]], [[VAR_44_]] : f32
// CHECK-DAG:           [[VAR_46_:%.+]] = addf [[LOAD_VAR_12_MEM_2_]], [[VAR_45_]] : f32
// CHECK-DAG:           [[LOAD_VAR_8_MEM_5_:%.+]] = krnl.load [[VAR_8_]]#2{{.}}[[VAR_15_1_]]#1] : memref<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_48_:%.+]] = addf [[VAR_46_]], [[LOAD_VAR_8_MEM_5_]] : f32
// CHECK-DAG:           [[VAR_49_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_48_]], [[VAR_49_]][] : memref<f32>
// CHECK:               [[VAR_50_:%.+]] = "onnx.Tanh"([[VAR_49_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_50_MEM_:%.+]] = krnl.load [[VAR_50_]][] : memref<f32>
// CHECK-DAG:           [[VAR_52_:%.+]] = subf [[CST_1_dot_000000_]], [[LOAD_VAR_25_MEM_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_53_:%.+]] = mulf [[VAR_52_]], [[LOAD_VAR_50_MEM_]] : f32
// CHECK-DAG:           [[VAR_54_:%.+]] = mulf [[LOAD_VAR_25_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_55_:%.+]] = addf [[VAR_53_]], [[VAR_54_]] : f32
// CHECK:               krnl.store [[VAR_55_]], [[VAR_0_]]{{.}}[[VAR_15_1_]]#0, [[VAR_15_1_]]#1] : memref<2x4xf32>
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
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.alloc() {{.*}}: memref<1x2x4xf32>
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
// CHECK-DAG:       [[VAR_8_:%.+]] = "krnl.global"() {name = "constant_5", shape = [3, 12], value = dense<1.000000e+00> : tensor<3x12xf32>} : () -> memref<3x12xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "krnl.global"() {name = "constant_6", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "krnl.global"() {name = "constant_7", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "krnl.global"() {name = "constant_8", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "krnl.global"() {name = "constant_9", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "krnl.global"() {name = "constant_10", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = "krnl.global"() {name = "constant_11", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = "krnl.global"() {name = "constant_12", shape = [24], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01]> : tensor<24xf32>} : () -> memref<24xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "krnl.global"() {name = "constant_13", shape = [4], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = "krnl.global"() {name = "constant_14", shape = [4], value = dense<[5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "krnl.global"() {name = "constant_15", shape = [4], value = dense<[9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "krnl.global"() {name = "constant_16", shape = [4], value = dense<[1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = "krnl.global"() {name = "constant_17", shape = [4], value = dense<[1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "krnl.global"() {name = "constant_18", shape = [4], value = dense<[2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7) {
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[VAR_24_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[VAR_25_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[CST_0_1_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_2_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_3_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_1_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_3_:%.+]] = constant 3 : index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_2_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_2_]] to [[CST_3_]]) {
// CHECK:               [[VAR_33_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_33_]]#0, [[VAR_33_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[VAR_25_]]{{.}}[[VAR_33_]]#0, [[VAR_33_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_27_:%.+]] = "onnx.MatMul"([[VAR_25_]], [[VAR_8_]]) : (memref<2x3xf32>, memref<3x12xf32>) -> memref<2x12xf32>
// CHECK-DAG:         [[CST_1_dot_000000_:%.+]] = constant 1.000000e+00 : f32
// CHECK-DAG:         [[CST_0_4_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_5_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_1_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_2_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:         [[VAR_28_:%.+]] = "onnx.MatMul"([[VAR_0_]], [[VAR_12_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_29_:%.+]] = "onnx.MatMul"([[VAR_0_]], [[VAR_13_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_4_]] to [[CST_2_1_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_4_]] to [[CST_4_]]) {
// CHECK-DAG:           [[VAR_33_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[CST_4_1_:%.+]] = constant 4 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[VAR_33_1_]]#0, [[VAR_33_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[VAR_35_:%.+]] = affine.apply #map0(){{.}}[[VAR_33_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_27_MEM_:%.+]] = krnl.load [[VAR_27_]]{{.}}[[VAR_33_1_]]#0, [[VAR_35_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_29_MEM_:%.+]] = krnl.load [[VAR_29_]]{{.}}[[VAR_33_1_]]#0, [[VAR_33_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_38_:%.+]] = addf [[LOAD_VAR_27_MEM_]], [[LOAD_VAR_29_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_17_MEM_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_33_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_20_MEM_:%.+]] = krnl.load [[VAR_20_]]{{.}}[[VAR_33_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_41_:%.+]] = addf [[VAR_38_]], [[LOAD_VAR_17_MEM_]] : f32
// CHECK-DAG:           [[VAR_42_:%.+]] = addf [[VAR_41_]], [[LOAD_VAR_20_MEM_]] : f32
// CHECK-DAG:           [[VAR_43_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_42_]], [[VAR_43_]][] : memref<f32>
// CHECK:               [[VAR_44_:%.+]] = "onnx.Sigmoid"([[VAR_43_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_VAR_44_MEM_:%.+]] = krnl.load [[VAR_44_]][] : memref<f32>
// CHECK:               krnl.store [[LOAD_VAR_44_MEM_]], [[VAR_24_]]{{.}}[[VAR_33_1_]]#0, [[VAR_33_1_]]#1] : memref<2x4xf32>
// CHECK:               [[VAR_46_:%.+]] = mulf [[LOAD_VAR_44_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               krnl.store [[VAR_46_]], [[LOAD_PARAM_1_MEM_1_]]{{.}}[[VAR_33_1_]]#0, [[VAR_33_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_31_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_1_MEM_1_]], [[VAR_14_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_7_:%.+]] = [[CST_0_4_]] to [[CST_2_1_]], [[LOOP_4_]]#1 -> [[I_8_:%.+]] = [[CST_0_4_]] to [[CST_4_]]) {
// CHECK-DAG:           [[VAR_33_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[CST_4_2_:%.+]] = constant 4 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[VAR_33_2_]]#0, [[VAR_33_2_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_27_MEM_1_:%.+]] = krnl.load [[VAR_27_]]{{.}}[[VAR_33_2_]]#0, [[VAR_33_2_]]#1] : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_27_MEM_2_:%.+]] = krnl.load [[VAR_28_]]{{.}}[[VAR_33_2_]]#0, [[VAR_33_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_29_MEM_1_:%.+]] = addf [[LOAD_VAR_27_MEM_1_]], [[LOAD_VAR_27_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_38_1_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_33_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_17_MEM_1_:%.+]] = krnl.load [[VAR_19_]]{{.}}[[VAR_33_2_]]#1] : memref<4xf32>
// CHECK:               [[LOAD_VAR_20_MEM_1_:%.+]] = addf [[LOAD_VAR_29_MEM_1_]], [[VAR_38_1_]] : f32
// CHECK-DAG:           [[VAR_41_1_:%.+]] = addf [[LOAD_VAR_20_MEM_1_]], [[LOAD_VAR_17_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_42_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_41_1_]], [[VAR_42_1_]][] : memref<f32>
// CHECK:               [[VAR_43_1_:%.+]] = "onnx.Sigmoid"([[VAR_42_1_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[VAR_44_1_:%.+]] = krnl.load [[VAR_43_1_]][] : memref<f32>
// CHECK-DAG:           [[CST_2_2_:%.+]] = constant 2 : index
// CHECK-DAG:           [[CST_8_:%.+]] = constant 8 : index
// CHECK-DAG:           [[LOAD_VAR_44_MEM_1_:%.+]] = affine.apply #map1(){{.}}[[VAR_33_2_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_27_MEM_3_:%.+]] = krnl.load [[VAR_27_]]{{.}}[[VAR_33_2_]]#0, [[LOAD_VAR_44_MEM_1_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_31_MEM_:%.+]] = krnl.load [[VAR_31_]]{{.}}[[VAR_33_2_]]#0, [[VAR_33_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_48_:%.+]] = addf [[LOAD_VAR_27_MEM_3_]], [[LOAD_VAR_31_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_18_MEM_:%.+]] = krnl.load [[VAR_18_]]{{.}}[[VAR_33_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_21_MEM_:%.+]] = krnl.load [[VAR_21_]]{{.}}[[VAR_33_2_]]#1] : memref<4xf32>
// CHECK:               [[VAR_51_:%.+]] = addf [[VAR_48_]], [[LOAD_VAR_18_MEM_]] : f32
// CHECK-DAG:           [[VAR_52_:%.+]] = addf [[VAR_51_]], [[LOAD_VAR_21_MEM_]] : f32
// CHECK-DAG:           [[VAR_53_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_52_]], [[VAR_53_]][] : memref<f32>
// CHECK:               [[VAR_54_:%.+]] = "onnx.Tanh"([[VAR_53_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_54_MEM_:%.+]] = krnl.load [[VAR_54_]][] : memref<f32>
// CHECK-DAG:           [[VAR_56_:%.+]] = subf [[CST_1_dot_000000_]], [[VAR_44_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_57_:%.+]] = mulf [[VAR_56_]], [[LOAD_VAR_54_MEM_]] : f32
// CHECK-DAG:           [[VAR_58_:%.+]] = mulf [[VAR_44_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_59_:%.+]] = addf [[VAR_57_]], [[VAR_58_]] : f32
// CHECK:               krnl.store [[VAR_59_]], [[VAR_0_]]{{.}}[[VAR_33_2_]]#0, [[VAR_33_2_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc [[VAR_24_]] : memref<2x4xf32>
// CHECK:             memref.dealloc [[LOAD_PARAM_1_MEM_1_]] : memref<2x4xf32>
// CHECK:             memref.dealloc [[VAR_25_]] : memref<2x3xf32>
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
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.alloc() {{.*}}: memref<1x2x4xf32>
// CHECK-DAG:       [[VAR_cst_:%.+]] = constant unit
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = constant 1 : index
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4) {
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[VAR_0_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.SqueezeV11"([[PARAM_1_]]) {axes = [0]} : (memref<1x12x3xf32>) -> memref<12x3xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.SqueezeV11"([[PARAM_2_]]) {axes = [0]} : (memref<1x12x4xf32>) -> memref<12x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Transpose"([[VAR_3_]]) {perm = [1, 0]} : (memref<12x3xf32>) -> memref<3x12xf32>
// CHECK-DAG:       [[VAR_6_:%.+]]:3 = "onnx.Split"([[VAR_4_]]) {axis = 0 : si64} : (memref<12x4xf32>) -> (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Transpose"([[VAR_6_]]#0) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Transpose"([[VAR_6_]]#1) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Transpose"([[VAR_6_]]#2) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.SqueezeV11"([[PARAM_3_]]) {axes = [0]} : (memref<1x24xf32>) -> memref<24xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]]:6 = "onnx.Split"([[VAR_10_]]) {axis = 0 : si64} : (memref<24xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7) {
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[VAR_14_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[VAR_15_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[CST_0_1_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_7_:%.+]] = constant 7 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_16_:%.+]] = affine.apply #map0([[I_2_]]){{.}}[[CST_7_]]{{.}}
// CHECK-DAG:         [[CST_0_2_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_3_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_1_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_3_:%.+]] = constant 3 : index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_2_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_2_]] to [[CST_3_]]) {
// CHECK:               [[VAR_24_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_16_]], [[VAR_24_]]#0, [[VAR_24_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[VAR_15_]]{{.}}[[VAR_24_]]#0, [[VAR_24_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_18_:%.+]] = "onnx.MatMul"([[VAR_15_]], [[VAR_5_]]) : (memref<2x3xf32>, memref<3x12xf32>) -> memref<2x12xf32>
// CHECK-DAG:         [[CST_1_dot_000000_:%.+]] = constant 1.000000e+00 : f32
// CHECK-DAG:         [[CST_0_4_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_5_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_1_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_2_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:         [[VAR_19_:%.+]] = "onnx.MatMul"([[VAR_0_]], [[VAR_7_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_20_:%.+]] = "onnx.MatMul"([[VAR_0_]], [[VAR_8_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_4_]] to [[CST_2_1_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_4_]] to [[CST_4_]]) {
// CHECK-DAG:           [[VAR_24_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[CST_4_1_:%.+]] = constant 4 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[VAR_24_1_]]#0, [[VAR_24_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[VAR_26_:%.+]] = affine.apply #map1(){{.}}[[VAR_24_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_18_MEM_:%.+]] = krnl.load [[VAR_18_]]{{.}}[[VAR_24_1_]]#0, [[VAR_26_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_20_MEM_:%.+]] = krnl.load [[VAR_20_]]{{.}}[[VAR_24_1_]]#0, [[VAR_24_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_29_:%.+]] = addf [[LOAD_VAR_18_MEM_]], [[LOAD_VAR_20_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_:%.+]] = krnl.load [[VAR_11_]]#1{{.}}[[VAR_24_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_11_MEM_1_:%.+]] = krnl.load [[VAR_11_]]#4{{.}}[[VAR_24_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_32_:%.+]] = addf [[VAR_29_]], [[LOAD_VAR_11_MEM_]] : f32
// CHECK-DAG:           [[VAR_33_:%.+]] = addf [[VAR_32_]], [[LOAD_VAR_11_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_34_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_33_]], [[VAR_34_]][] : memref<f32>
// CHECK:               [[VAR_35_:%.+]] = "onnx.Sigmoid"([[VAR_34_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_VAR_35_MEM_:%.+]] = krnl.load [[VAR_35_]][] : memref<f32>
// CHECK:               krnl.store [[LOAD_VAR_35_MEM_]], [[VAR_14_]]{{.}}[[VAR_24_1_]]#0, [[VAR_24_1_]]#1] : memref<2x4xf32>
// CHECK:               [[VAR_37_:%.+]] = mulf [[LOAD_VAR_35_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               krnl.store [[VAR_37_]], [[LOAD_PARAM_4_MEM_1_]]{{.}}[[VAR_24_1_]]#0, [[VAR_24_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_22_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_4_MEM_1_]], [[VAR_9_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_7_:%.+]] = [[CST_0_4_]] to [[CST_2_1_]], [[LOOP_4_]]#1 -> [[I_8_:%.+]] = [[CST_0_4_]] to [[CST_4_]]) {
// CHECK-DAG:           [[VAR_24_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[CST_4_2_:%.+]] = constant 4 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[VAR_24_2_]]#0, [[VAR_24_2_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_18_MEM_1_:%.+]] = krnl.load [[VAR_18_]]{{.}}[[VAR_24_2_]]#0, [[VAR_24_2_]]#1] : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_18_MEM_2_:%.+]] = krnl.load [[VAR_19_]]{{.}}[[VAR_24_2_]]#0, [[VAR_24_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_20_MEM_1_:%.+]] = addf [[LOAD_VAR_18_MEM_1_]], [[LOAD_VAR_18_MEM_2_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_2_:%.+]] = krnl.load [[VAR_11_]]#0{{.}}[[VAR_24_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_11_MEM_3_:%.+]] = krnl.load [[VAR_11_]]#3{{.}}[[VAR_24_2_]]#1] : memref<4xf32>
// CHECK:               [[LOAD_VAR_11_MEM_1_:%.+]] = addf [[LOAD_VAR_20_MEM_1_]], [[LOAD_VAR_11_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_32_1_:%.+]] = addf [[LOAD_VAR_11_MEM_1_]], [[LOAD_VAR_11_MEM_3_]] : f32
// CHECK-DAG:           [[VAR_33_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_32_1_]], [[VAR_33_1_]][] : memref<f32>
// CHECK:               [[VAR_34_1_:%.+]] = "onnx.Sigmoid"([[VAR_33_1_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[VAR_35_1_:%.+]] = krnl.load [[VAR_34_1_]][] : memref<f32>
// CHECK-DAG:           [[CST_2_2_:%.+]] = constant 2 : index
// CHECK-DAG:           [[CST_8_:%.+]] = constant 8 : index
// CHECK-DAG:           [[LOAD_VAR_35_MEM_1_:%.+]] = affine.apply #map2(){{.}}[[VAR_24_2_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_18_MEM_3_:%.+]] = krnl.load [[VAR_18_]]{{.}}[[VAR_24_2_]]#0, [[LOAD_VAR_35_MEM_1_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_22_MEM_:%.+]] = krnl.load [[VAR_22_]]{{.}}[[VAR_24_2_]]#0, [[VAR_24_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_39_:%.+]] = addf [[LOAD_VAR_18_MEM_3_]], [[LOAD_VAR_22_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_4_:%.+]] = krnl.load [[VAR_11_]]#2{{.}}[[VAR_24_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_11_MEM_5_:%.+]] = krnl.load [[VAR_11_]]#5{{.}}[[VAR_24_2_]]#1] : memref<4xf32>
// CHECK:               [[VAR_42_:%.+]] = addf [[VAR_39_]], [[LOAD_VAR_11_MEM_4_]] : f32
// CHECK-DAG:           [[VAR_43_:%.+]] = addf [[VAR_42_]], [[LOAD_VAR_11_MEM_5_]] : f32
// CHECK-DAG:           [[VAR_44_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_43_]], [[VAR_44_]][] : memref<f32>
// CHECK:               [[VAR_45_:%.+]] = "onnx.Tanh"([[VAR_44_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_45_MEM_:%.+]] = krnl.load [[VAR_45_]][] : memref<f32>
// CHECK-DAG:           [[VAR_47_:%.+]] = subf [[CST_1_dot_000000_]], [[VAR_35_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_48_:%.+]] = mulf [[VAR_47_]], [[LOAD_VAR_45_MEM_]] : f32
// CHECK-DAG:           [[VAR_49_:%.+]] = mulf [[VAR_35_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_50_:%.+]] = addf [[VAR_48_]], [[VAR_49_]] : f32
// CHECK:               krnl.store [[VAR_50_]], [[VAR_0_]]{{.}}[[VAR_24_2_]]#0, [[VAR_24_2_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc [[VAR_14_]] : memref<2x4xf32>
// CHECK:             memref.dealloc [[LOAD_PARAM_4_MEM_1_]] : memref<2x4xf32>
// CHECK:             memref.dealloc [[VAR_15_]] : memref<2x3xf32>
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
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = memref.alloc() {{.*}}: memref<2x2x4xf32>
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
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.SqueezeV11"([[VAR_4_]]#0) {axes = [0]} : (memref<1x12x3xf32>) -> memref<12x3xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.SqueezeV11"([[VAR_4_]]#1) {axes = [0]} : (memref<1x12x3xf32>) -> memref<12x3xf32>
// CHECK-DAG:       [[VAR_7_:%.+]]:2 = "onnx.Split"([[PARAM_2_]]) {axis = 0 : si64} : (memref<2x12x4xf32>) -> (memref<1x12x4xf32>, memref<1x12x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.SqueezeV11"([[VAR_7_]]#0) {axes = [0]} : (memref<1x12x4xf32>) -> memref<12x4xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.SqueezeV11"([[VAR_7_]]#1) {axes = [0]} : (memref<1x12x4xf32>) -> memref<12x4xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Transpose"([[VAR_5_]]) {perm = [1, 0]} : (memref<12x3xf32>) -> memref<3x12xf32>
// CHECK:           [[VAR_11_:%.+]]:3 = "onnx.Split"([[VAR_8_]]) {axis = 0 : si64} : (memref<12x4xf32>) -> (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>)
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Transpose"([[VAR_11_]]#0) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Transpose"([[VAR_11_]]#1) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.Transpose"([[VAR_11_]]#2) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.Transpose"([[VAR_6_]]) {perm = [1, 0]} : (memref<12x3xf32>) -> memref<3x12xf32>
// CHECK-DAG:       [[VAR_16_:%.+]]:3 = "onnx.Split"([[VAR_9_]]) {axis = 0 : si64} : (memref<12x4xf32>) -> (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Transpose"([[VAR_16_]]#0) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Transpose"([[VAR_16_]]#1) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.Transpose"([[VAR_16_]]#2) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_20_:%.+]]:2 = "onnx.Split"([[PARAM_3_]]) {axis = 0 : si64} : (memref<2x24xf32>) -> (memref<1x24xf32>, memref<1x24xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.SqueezeV11"([[VAR_20_]]#0) {axes = [0]} : (memref<1x24xf32>) -> memref<24xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = "onnx.SqueezeV11"([[VAR_20_]]#1) {axes = [0]} : (memref<1x24xf32>) -> memref<24xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_23_:%.+]]:6 = "onnx.Split"([[VAR_21_]]) {axis = 0 : si64} : (memref<24xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[VAR_24_:%.+]]:6 = "onnx.Split"([[VAR_22_]]) {axis = 0 : si64} : (memref<24xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7) {
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_2_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[VAR_30_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[CST_0_1_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_2_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_3_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_1_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_3_:%.+]] = constant 3 : index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_2_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_2_]] to [[CST_3_]]) {
// CHECK:               [[VAR_38_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_38_]]#0, [[VAR_38_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[VAR_30_]]{{.}}[[VAR_38_]]#0, [[VAR_38_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_32_:%.+]] = "onnx.MatMul"([[VAR_30_]], [[VAR_10_]]) : (memref<2x3xf32>, memref<3x12xf32>) -> memref<2x12xf32>
// CHECK-DAG:         [[CST_1_dot_000000_:%.+]] = constant 1.000000e+00 : f32
// CHECK-DAG:         [[CST_0_4_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_5_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_1_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_2_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:         [[VAR_33_:%.+]] = "onnx.MatMul"([[VAR_1_]], [[VAR_1_]]2) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_34_:%.+]] = "onnx.MatMul"([[VAR_1_]], [[VAR_1_]]3) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_4_]] to [[CST_2_1_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_4_]] to [[CST_4_]]) {
// CHECK-DAG:           [[VAR_38_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[CST_4_1_:%.+]] = constant 4 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_1_]]{{.}}[[VAR_38_1_]]#0, [[VAR_38_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[VAR_40_:%.+]] = affine.apply #map0(){{.}}[[VAR_38_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_32_MEM_:%.+]] = krnl.load [[VAR_32_]]{{.}}[[VAR_38_1_]]#0, [[VAR_40_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_34_MEM_:%.+]] = krnl.load [[VAR_34_]]{{.}}[[VAR_38_1_]]#0, [[VAR_38_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_43_:%.+]] = addf [[LOAD_VAR_32_MEM_]], [[LOAD_VAR_34_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_23_MEM_:%.+]] = krnl.load [[VAR_23_]]#1{{.}}[[VAR_38_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_23_MEM_1_:%.+]] = krnl.load [[VAR_23_]]#4{{.}}[[VAR_38_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_46_:%.+]] = addf [[VAR_43_]], [[LOAD_VAR_23_MEM_]] : f32
// CHECK-DAG:           [[VAR_47_:%.+]] = addf [[VAR_46_]], [[LOAD_VAR_23_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_48_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_47_]], [[VAR_48_]][] : memref<f32>
// CHECK:               [[VAR_49_:%.+]] = "onnx.Sigmoid"([[VAR_48_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_VAR_49_MEM_:%.+]] = krnl.load [[VAR_49_]][] : memref<f32>
// CHECK:               krnl.store [[LOAD_VAR_49_MEM_]], [[LOAD_PARAM_4_MEM_1_]]{{.}}[[VAR_38_1_]]#0, [[VAR_38_1_]]#1] : memref<2x4xf32>
// CHECK:               [[VAR_51_:%.+]] = mulf [[LOAD_VAR_49_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               krnl.store [[VAR_51_]], [[LOAD_PARAM_4_MEM_2_]]{{.}}[[VAR_38_1_]]#0, [[VAR_38_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_36_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_4_MEM_2_]], [[VAR_14_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_7_:%.+]] = [[CST_0_4_]] to [[CST_2_1_]], [[LOOP_4_]]#1 -> [[I_8_:%.+]] = [[CST_0_4_]] to [[CST_4_]]) {
// CHECK-DAG:           [[VAR_38_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[CST_4_2_:%.+]] = constant 4 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_1_]]{{.}}[[VAR_38_2_]]#0, [[VAR_38_2_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_32_MEM_1_:%.+]] = krnl.load [[VAR_32_]]{{.}}[[VAR_38_2_]]#0, [[VAR_38_2_]]#1] : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_32_MEM_2_:%.+]] = krnl.load [[VAR_33_]]{{.}}[[VAR_38_2_]]#0, [[VAR_38_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_34_MEM_1_:%.+]] = addf [[LOAD_VAR_32_MEM_1_]], [[LOAD_VAR_32_MEM_2_]] : f32
// CHECK-DAG:           [[LOAD_VAR_23_MEM_2_:%.+]] = krnl.load [[VAR_23_]]#0{{.}}[[VAR_38_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_23_MEM_3_:%.+]] = krnl.load [[VAR_23_]]#3{{.}}[[VAR_38_2_]]#1] : memref<4xf32>
// CHECK:               [[LOAD_VAR_23_MEM_1_:%.+]] = addf [[LOAD_VAR_34_MEM_1_]], [[LOAD_VAR_23_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_46_1_:%.+]] = addf [[LOAD_VAR_23_MEM_1_]], [[LOAD_VAR_23_MEM_3_]] : f32
// CHECK-DAG:           [[VAR_47_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_46_1_]], [[VAR_47_1_]][] : memref<f32>
// CHECK:               [[VAR_48_1_:%.+]] = "onnx.Sigmoid"([[VAR_47_1_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[VAR_49_1_:%.+]] = krnl.load [[VAR_48_1_]][] : memref<f32>
// CHECK-DAG:           [[CST_2_2_:%.+]] = constant 2 : index
// CHECK-DAG:           [[CST_8_:%.+]] = constant 8 : index
// CHECK-DAG:           [[LOAD_VAR_49_MEM_1_:%.+]] = affine.apply #map1(){{.}}[[VAR_38_2_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_32_MEM_3_:%.+]] = krnl.load [[VAR_32_]]{{.}}[[VAR_38_2_]]#0, [[LOAD_VAR_49_MEM_1_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_36_MEM_:%.+]] = krnl.load [[VAR_36_]]{{.}}[[VAR_38_2_]]#0, [[VAR_38_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_53_:%.+]] = addf [[LOAD_VAR_32_MEM_3_]], [[LOAD_VAR_36_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_23_MEM_4_:%.+]] = krnl.load [[VAR_23_]]#2{{.}}[[VAR_38_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_23_MEM_5_:%.+]] = krnl.load [[VAR_23_]]#5{{.}}[[VAR_38_2_]]#1] : memref<4xf32>
// CHECK:               [[VAR_56_:%.+]] = addf [[VAR_53_]], [[LOAD_VAR_23_MEM_4_]] : f32
// CHECK-DAG:           [[VAR_57_:%.+]] = addf [[VAR_56_]], [[LOAD_VAR_23_MEM_5_]] : f32
// CHECK-DAG:           [[VAR_58_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_57_]], [[VAR_58_]][] : memref<f32>
// CHECK:               [[VAR_59_:%.+]] = "onnx.Tanh"([[VAR_58_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_59_MEM_:%.+]] = krnl.load [[VAR_59_]][] : memref<f32>
// CHECK-DAG:           [[VAR_61_:%.+]] = subf [[CST_1_dot_000000_]], [[VAR_49_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_62_:%.+]] = mulf [[VAR_61_]], [[LOAD_VAR_59_MEM_]] : f32
// CHECK-DAG:           [[VAR_63_:%.+]] = mulf [[VAR_49_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_64_:%.+]] = addf [[VAR_62_]], [[VAR_63_]] : f32
// CHECK:               krnl.store [[VAR_64_]], [[VAR_1_]]{{.}}[[VAR_38_2_]]#0, [[VAR_38_2_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc [[LOAD_PARAM_4_MEM_1_]] : memref<2x4xf32>
// CHECK:             memref.dealloc [[LOAD_PARAM_4_MEM_2_]] : memref<2x4xf32>
// CHECK:             memref.dealloc [[VAR_30_]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_5_]]) with ([[LOOP_5_]] -> [[I_9_:%.+]] = 0 to 7) {
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_2_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_1_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[VAR_30_1_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[CST_1_3_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_7_:%.+]] = constant 7 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOOP_2_:%.+]] = affine.apply #map2([[I_9_]]){{.}}[[CST_7_]]{{.}}
// CHECK-DAG:         [[CST_0_6_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_7_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_3_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_4_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_3_1_:%.+]] = constant 3 : index
// CHECK-DAG:         [[LOOP_6_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_6_]]#0, [[LOOP_6_]]#1) with ([[LOOP_6_]]#0 -> [[I_10_:%.+]] = [[CST_0_6_]] to [[CST_2_3_]], [[LOOP_6_]]#1 -> [[I_11_:%.+]] = [[CST_0_6_]] to [[CST_3_1_]]) {
// CHECK:               [[LOAD_PARAM_0_MEM_1_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_6_]]#0, [[LOOP_6_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[LOOP_2_]], [[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_2_]], [[VAR_30_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_33_1_:%.+]] = "onnx.MatMul"([[VAR_30_1_]], [[VAR_15_]]) : (memref<2x3xf32>, memref<3x12xf32>) -> memref<2x12xf32>
// CHECK-DAG:         [[CST_1_dot_000000_1_:%.+]] = constant 1.000000e+00 : f32
// CHECK-DAG:         [[CST_0_8_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_9_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_4_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_5_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_4_3_:%.+]] = constant 4 : index
// CHECK-DAG:         [[VAR_34_1_:%.+]] = "onnx.MatMul"([[VAR_0_]], [[VAR_17_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]] = "onnx.MatMul"([[VAR_0_]], [[VAR_18_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[LOOP_7_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_7_]]#0, [[LOOP_7_]]#1) with ([[LOOP_7_]]#0 -> [[I_12_:%.+]] = [[CST_0_8_]] to [[CST_2_4_]], [[LOOP_7_]]#1 -> [[I_13_:%.+]] = [[CST_0_8_]] to [[CST_4_3_]]) {
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_7_]]#0, [[LOOP_7_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[CST_4_4_:%.+]] = constant 4 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_32_MEM_2_:%.+]] = affine.apply #map0(){{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_34_MEM_1_:%.+]] = krnl.load [[VAR_33_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_VAR_32_MEM_2_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_23_MEM_2_:%.+]] = krnl.load [[LOOP_3_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_23_MEM_3_:%.+]] = addf [[LOAD_VAR_34_MEM_1_]], [[LOAD_VAR_23_MEM_2_]] : f32
// CHECK-DAG:           [[LOAD_VAR_23_MEM_1_1_:%.+]] = krnl.load [[VAR_24_]]#1{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_46_1_:%.+]] = krnl.load [[VAR_24_]]#4{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_47_2_:%.+]] = addf [[LOAD_VAR_23_MEM_3_]], [[LOAD_VAR_23_MEM_1_1_]] : f32
// CHECK-DAG:           [[VAR_48_2_:%.+]] = addf [[VAR_47_2_]], [[VAR_46_1_]] : f32
// CHECK-DAG:           [[VAR_49_2_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_48_2_]], [[VAR_49_2_]][] : memref<f32>
// CHECK:               [[LOAD_VAR_49_MEM_1_:%.+]] = "onnx.Sigmoid"([[VAR_49_2_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_VAR_32_MEM_3_:%.+]] = krnl.load [[LOAD_VAR_49_MEM_1_]][] : memref<f32>
// CHECK:               krnl.store [[LOAD_VAR_32_MEM_3_]], [[LOAD_PARAM_4_MEM_1_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK:               [[LOAD_VAR_36_MEM_1_:%.+]] = mulf [[LOAD_VAR_32_MEM_3_]], [[LOAD_PARAM_0_MEM_2_]] : f32
// CHECK:               krnl.store [[LOAD_VAR_36_MEM_1_]], [[LOAD_PARAM_4_MEM_2_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK-DAG:         [[LOOP_4_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_4_MEM_2_]], [[VAR_19_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[LOOP_8_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_8_]]#0, [[LOOP_8_]]#1) with ([[LOOP_8_]]#0 -> [[I_14_:%.+]] = [[CST_0_8_]] to [[CST_2_4_]], [[LOOP_8_]]#1 -> [[I_15_:%.+]] = [[CST_0_8_]] to [[CST_4_3_]]) {
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_1_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_8_]]#0, [[LOOP_8_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[CST_4_5_:%.+]] = constant 4 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_2_1_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_32_MEM_2_1_:%.+]] = krnl.load [[VAR_33_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_34_MEM_1_1_:%.+]] = krnl.load [[VAR_34_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_43_1_:%.+]] = addf [[LOAD_VAR_32_MEM_2_1_]], [[LOAD_VAR_34_MEM_1_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_23_MEM_3_1_:%.+]] = krnl.load [[VAR_24_]]#0{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_23_MEM_1_1_:%.+]] = krnl.load [[VAR_24_]]#3{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_46_2_:%.+]] = addf [[VAR_43_1_]], [[LOAD_VAR_23_MEM_3_1_]] : f32
// CHECK-DAG:           [[VAR_47_3_:%.+]] = addf [[VAR_46_2_]], [[LOAD_VAR_23_MEM_1_1_]] : f32
// CHECK-DAG:           [[VAR_48_3_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_47_3_]], [[VAR_48_3_]][] : memref<f32>
// CHECK:               [[VAR_49_3_:%.+]] = "onnx.Sigmoid"([[VAR_48_3_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_49_MEM_1_1_:%.+]] = krnl.load [[VAR_49_3_]][] : memref<f32>
// CHECK-DAG:           [[CST_2_5_:%.+]] = constant 2 : index
// CHECK-DAG:           [[CST_8_1_:%.+]] = constant 8 : index
// CHECK-DAG:           [[VAR_51_1_:%.+]] = affine.apply #map1(){{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_36_MEM_1_:%.+]] = krnl.load [[VAR_33_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#0, [[VAR_51_1_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[VAR_53_1_:%.+]] = krnl.load [[LOOP_4_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_23_MEM_4_:%.+]] = addf [[LOAD_VAR_36_MEM_1_]], [[VAR_53_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_23_MEM_5_:%.+]] = krnl.load [[VAR_24_]]#2{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_56_1_:%.+]] = krnl.load [[VAR_24_]]#5{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_57_1_:%.+]] = addf [[LOAD_VAR_23_MEM_4_]], [[LOAD_VAR_23_MEM_5_]] : f32
// CHECK-DAG:           [[VAR_58_1_:%.+]] = addf [[VAR_57_1_]], [[VAR_56_1_]] : f32
// CHECK-DAG:           [[VAR_59_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_58_1_]], [[VAR_59_1_]][] : memref<f32>
// CHECK:               [[LOAD_VAR_59_MEM_1_:%.+]] = "onnx.Tanh"([[VAR_59_1_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[VAR_61_1_:%.+]] = krnl.load [[LOAD_VAR_59_MEM_1_]][] : memref<f32>
// CHECK-DAG:           [[VAR_62_1_:%.+]] = subf [[CST_1_dot_000000_1_]], [[LOAD_VAR_49_MEM_1_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_63_1_:%.+]] = mulf [[VAR_62_1_]], [[VAR_61_1_]] : f32
// CHECK-DAG:           [[VAR_64_1_:%.+]] = mulf [[LOAD_VAR_49_MEM_1_1_]], [[LOAD_PARAM_0_MEM_2_1_]] : f32
// CHECK:               [[VAR_65_:%.+]] = addf [[VAR_63_1_]], [[VAR_64_1_]] : f32
// CHECK:               krnl.store [[VAR_65_]], [[VAR_0_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc [[LOAD_PARAM_4_MEM_1_1_]] : memref<2x4xf32>
// CHECK:             memref.dealloc [[LOAD_PARAM_4_MEM_2_]] : memref<2x4xf32>
// CHECK:             memref.dealloc [[VAR_30_1_]] : memref<2x3xf32>
// CHECK:           }
// CHECK-DAG:       [[CST_0_10_:%.+]] = constant 0 : index
// CHECK-DAG:       [[CST_1_6_:%.+]] = constant 1 : index
// CHECK-DAG:       [[CST_0_11_:%.+]] = constant 0 : index
// CHECK-DAG:       [[CST_2_6_:%.+]] = constant 2 : index
// CHECK-DAG:       [[CST_1_7_:%.+]] = constant 1 : index
// CHECK-DAG:       [[CST_4_6_:%.+]] = constant 4 : index
// CHECK-DAG:       [[LOOP_9_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_9_]]#0, [[LOOP_9_]]#1) with ([[LOOP_9_]]#0 -> [[I_16_:%.+]] = [[CST_0_10_]] to [[CST_2_6_]], [[LOOP_9_]]#1 -> [[I_17_:%.+]] = [[CST_0_10_]] to [[CST_4_6_]]) {
// CHECK:             [[LOAD_PARAM_4_MEM_2_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_9_]]#0, [[LOOP_9_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_4_MEM_1_1_:%.+]] = krnl.load [[VAR_1_]]{{.}}[[LOAD_PARAM_4_MEM_2_1_]]#0, [[LOAD_PARAM_4_MEM_2_1_]]#1] : memref<2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_1_1_]], [[VAR_2_]]{{.}}[[CST_0_10_]], [[VAR_2_]]8#0, [[VAR_2_]]8#1] : memref<2x2x4xf32>
// CHECK:             [[VAR_30_1_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[LOAD_PARAM_4_MEM_2_1_]]#0, [[LOAD_PARAM_4_MEM_2_1_]]#1] : memref<2x4xf32>
// CHECK:             krnl.store [[VAR_30_1_]], [[VAR_2_]]{{.}}[[CST_1_6_]], [[VAR_2_]]8#0, [[VAR_2_]]8#1] : memref<2x2x4xf32>
// CHECK:           }
// CHECK:           memref.dealloc [[VAR_1_]] : memref<2x4xf32>
// CHECK:           memref.dealloc [[VAR_0_]] : memref<2x4xf32>
// CHECK:           return [[VAR_2_]] : memref<2x2x4xf32>
// CHECK:         }
}
