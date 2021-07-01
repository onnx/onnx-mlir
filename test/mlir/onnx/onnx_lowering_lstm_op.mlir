// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl='check-rnn-ops-lowering' %s -split-input-file | FileCheck %s

func private @test_lstm_forward_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x16x3xf32>, %arg2: tensor<1x16x4xf32>, %arg3: tensor<1x32xf32>, %arg4: tensor<1x2x4xf32>, %arg5: tensor<1x2x4xf32>, %arg6: tensor<1x12xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4, %arg5, %arg6) {hidden_size = 4 : si64} : (tensor<7x2x3xf32>, tensor<1x16x3xf32>, tensor<1x16x4xf32>, tensor<1x32xf32>, none, tensor<1x2x4xf32>, tensor<1x2x4xf32>, tensor<1x12xf32>) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>
// CHECK-LABEL:  func private @test_lstm_forward_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x16x3xf32>, [[PARAM_2_:%.+]]: memref<1x16x4xf32>, [[PARAM_3_:%.+]]: memref<1x32xf32>, [[PARAM_4_:%.+]]: memref<1x2x4xf32>, [[PARAM_5_:%.+]]: memref<1x2x4xf32>, [[PARAM_6_:%.+]]: memref<1x12xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.alloc() : memref<2x4xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.alloc() : memref<2x4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = memref.alloc() : memref<1x2x4xf32>
// CHECK-DAG:       [[VAR_cst_:%.+]] = constant unit
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = constant 1 : index
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4) {
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[VAR_1_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:             [[LOAD_PARAM_5_MEM_:%.+]] = krnl.load [[PARAM_5_]]{{.}}[[CST_0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_5_MEM_]], [[VAR_0_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Squeeze"([[PARAM_1_]]) {axes = [0]} : (memref<1x16x3xf32>) -> memref<16x3xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Squeeze"([[PARAM_2_]]) {axes = [0]} : (memref<1x16x4xf32>) -> memref<16x4xf32>
// CHECK:           [[VAR_6_:%.+]]:4 = "onnx.Split"([[VAR_4_]]) {axis = 0 : si64} : (memref<16x3xf32>) -> (memref<4x3xf32>, memref<4x3xf32>, memref<4x3xf32>, memref<4x3xf32>)
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Transpose"([[VAR_6_]]#0) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Transpose"([[VAR_6_]]#1) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Transpose"([[VAR_6_]]#2) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Transpose"([[VAR_6_]]#3) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_11_:%.+]]:4 = "onnx.Split"([[VAR_5_]]) {axis = 0 : si64} : (memref<16x4xf32>) -> (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Transpose"([[VAR_11_]]#0) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Transpose"([[VAR_11_]]#1) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.Transpose"([[VAR_11_]]#2) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.Transpose"([[VAR_11_]]#3) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Squeeze"([[PARAM_3_]]) {axes = [0]} : (memref<1x32xf32>) -> memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_17_:%.+]]:8 = "onnx.Split"([[VAR_16_]]) {axis = 0 : si64} : (memref<32xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Squeeze"([[PARAM_6_]]) {axes = [0]} : (memref<1x12xf32>) -> memref<12xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_19_:%.+]]:3 = "onnx.Split"([[VAR_18_]]) {axis = 0 : si64} : (memref<12xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7) {
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_1_:%.+]] = memref.alloc() : memref<2x3xf32>
// CHECK-DAG:         [[CST_0_1_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_3_:%.+]] = constant 3 : index
// CHECK-DAG:         [[CST_0_2_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_3_:%.+]] = constant 0 : index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_2_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_3_]] to [[CST_3_]]) {
// CHECK:               [[VAR_32_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_32_]]#0, [[VAR_32_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_4_MEM_1_]]{{.}}[[VAR_32_]]#0, [[VAR_32_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_23_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_4_MEM_1_]], [[VAR_7_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_24_:%.+]] = "onnx.MatMul"([[VAR_1_]], [[VAR_1_]]2) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_25_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_4_MEM_1_]], [[VAR_9_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_26_:%.+]] = "onnx.MatMul"([[VAR_1_]], [[VAR_1_]]4) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_27_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_4_MEM_1_]], [[VAR_10_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_28_:%.+]] = "onnx.MatMul"([[VAR_1_]], [[VAR_1_]]5) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_29_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_4_MEM_1_]], [[VAR_8_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_30_:%.+]] = "onnx.MatMul"([[VAR_1_]], [[VAR_1_]]3) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[CST_0_4_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_5_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_1_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_1_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_4_]] to [[CST_2_1_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_4_]] to [[CST_4_]]) {
// CHECK:               [[VAR_32_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[VAR_32_1_]]#0, [[VAR_32_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_23_MEM_:%.+]] = krnl.load [[VAR_23_]]{{.}}[[VAR_32_1_]]#0, [[VAR_32_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_24_MEM_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_32_1_]]#0, [[VAR_32_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_36_:%.+]] = addf [[LOAD_VAR_23_MEM_]], [[LOAD_VAR_24_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_17_MEM_:%.+]] = krnl.load [[VAR_17_]]#0{{.}}[[VAR_32_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_17_MEM_1_:%.+]] = krnl.load [[VAR_17_]]#4{{.}}[[VAR_32_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_39_:%.+]] = addf [[VAR_36_]], [[LOAD_VAR_17_MEM_]] : f32
// CHECK-DAG:           [[VAR_40_:%.+]] = addf [[VAR_39_]], [[LOAD_VAR_17_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_19_MEM_:%.+]] = krnl.load [[VAR_19_]]#0{{.}}[[VAR_32_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_42_:%.+]] = mulf [[LOAD_VAR_19_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_43_:%.+]] = addf [[VAR_40_]], [[VAR_42_]] : f32
// CHECK-DAG:           [[VAR_44_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_43_]], [[VAR_44_]][] : memref<f32>
// CHECK:               [[VAR_45_:%.+]] = "onnx.Sigmoid"([[VAR_44_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_45_MEM_:%.+]] = krnl.load [[VAR_45_]][] : memref<f32>
// CHECK-DAG:           [[LOAD_VAR_25_MEM_:%.+]] = krnl.load [[VAR_25_]]{{.}}[[VAR_32_1_]]#0, [[VAR_32_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_26_MEM_:%.+]] = krnl.load [[VAR_26_]]{{.}}[[VAR_32_1_]]#0, [[VAR_32_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_49_:%.+]] = addf [[LOAD_VAR_25_MEM_]], [[LOAD_VAR_26_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_17_MEM_2_:%.+]] = krnl.load [[VAR_17_]]#2{{.}}[[VAR_32_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_17_MEM_3_:%.+]] = krnl.load [[VAR_17_]]#6{{.}}[[VAR_32_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_52_:%.+]] = addf [[VAR_49_]], [[LOAD_VAR_17_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_53_:%.+]] = addf [[VAR_52_]], [[LOAD_VAR_17_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_19_MEM_1_:%.+]] = krnl.load [[VAR_19_]]#2{{.}}[[VAR_32_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_55_:%.+]] = mulf [[LOAD_VAR_19_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_56_:%.+]] = addf [[VAR_53_]], [[VAR_55_]] : f32
// CHECK-DAG:           [[VAR_57_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_56_]], [[VAR_57_]][] : memref<f32>
// CHECK:               [[VAR_58_:%.+]] = "onnx.Sigmoid"([[VAR_57_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_58_MEM_:%.+]] = krnl.load [[VAR_58_]][] : memref<f32>
// CHECK-DAG:           [[LOAD_VAR_27_MEM_:%.+]] = krnl.load [[VAR_27_]]{{.}}[[VAR_32_1_]]#0, [[VAR_32_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_28_MEM_:%.+]] = krnl.load [[VAR_28_]]{{.}}[[VAR_32_1_]]#0, [[VAR_32_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_62_:%.+]] = addf [[LOAD_VAR_27_MEM_]], [[LOAD_VAR_28_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_17_MEM_4_:%.+]] = krnl.load [[VAR_17_]]#3{{.}}[[VAR_32_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_17_MEM_5_:%.+]] = krnl.load [[VAR_17_]]#7{{.}}[[VAR_32_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_65_:%.+]] = addf [[VAR_62_]], [[LOAD_VAR_17_MEM_4_]] : f32
// CHECK-DAG:           [[VAR_66_:%.+]] = addf [[VAR_65_]], [[LOAD_VAR_17_MEM_5_]] : f32
// CHECK-DAG:           [[VAR_67_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_66_]], [[VAR_67_]][] : memref<f32>
// CHECK:               [[VAR_68_:%.+]] = "onnx.Tanh"([[VAR_67_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_68_MEM_:%.+]] = krnl.load [[VAR_68_]][] : memref<f32>
// CHECK-DAG:           [[VAR_70_:%.+]] = mulf [[LOAD_VAR_58_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_71_:%.+]] = mulf [[LOAD_VAR_45_MEM_]], [[LOAD_VAR_68_MEM_]] : f32
// CHECK-DAG:           [[VAR_72_:%.+]] = addf [[VAR_70_]], [[VAR_71_]] : f32
// CHECK-DAG:           [[LOAD_VAR_29_MEM_:%.+]] = krnl.load [[VAR_29_]]{{.}}[[VAR_32_1_]]#0, [[VAR_32_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_30_MEM_:%.+]] = krnl.load [[VAR_30_]]{{.}}[[VAR_32_1_]]#0, [[VAR_32_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_75_:%.+]] = addf [[LOAD_VAR_29_MEM_]], [[LOAD_VAR_30_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_17_MEM_6_:%.+]] = krnl.load [[VAR_17_]]#1{{.}}[[VAR_32_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_17_MEM_7_:%.+]] = krnl.load [[VAR_17_]]#5{{.}}[[VAR_32_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_78_:%.+]] = addf [[VAR_75_]], [[LOAD_VAR_17_MEM_6_]] : f32
// CHECK-DAG:           [[VAR_79_:%.+]] = addf [[VAR_78_]], [[LOAD_VAR_17_MEM_7_]] : f32
// CHECK-DAG:           [[LOAD_VAR_19_MEM_2_:%.+]] = krnl.load [[VAR_19_]]#1{{.}}[[VAR_32_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_81_:%.+]] = mulf [[LOAD_VAR_19_MEM_2_]], [[VAR_72_]] : f32
// CHECK-DAG:           [[VAR_82_:%.+]] = addf [[VAR_79_]], [[VAR_81_]] : f32
// CHECK-DAG:           [[VAR_83_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_82_]], [[VAR_83_]][] : memref<f32>
// CHECK:               [[VAR_84_:%.+]] = "onnx.Sigmoid"([[VAR_83_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_84_MEM_:%.+]] = krnl.load [[VAR_84_]][] : memref<f32>
// CHECK-DAG:           [[VAR_86_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_72_]], [[VAR_86_]][] : memref<f32>
// CHECK:               [[VAR_87_:%.+]] = "onnx.Tanh"([[VAR_86_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_VAR_87_MEM_:%.+]] = krnl.load [[VAR_87_]][] : memref<f32>
// CHECK:               [[VAR_89_:%.+]] = mulf [[LOAD_VAR_84_MEM_]], [[LOAD_VAR_87_MEM_]] : f32
// CHECK:               krnl.store [[VAR_72_]], [[VAR_0_]]{{.}}[[VAR_32_1_]]#0, [[VAR_32_1_]]#1] : memref<2x4xf32>
// CHECK:               krnl.store [[VAR_89_]], [[VAR_1_]]{{.}}[[VAR_32_1_]]#0, [[VAR_32_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc [[LOAD_PARAM_4_MEM_1_]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           [[CST_32_:%.+]] = constant 32 : i64
// CHECK:           "krnl.memcpy"([[VAR_2_]], [[VAR_1_]], [[CST_32_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
// CHECK:           memref.dealloc [[VAR_1_]] : memref<2x4xf32>
// CHECK:           memref.dealloc [[VAR_0_]] : memref<2x4xf32>
// CHECK:           return [[VAR_2_]] : memref<1x2x4xf32>
// CHECK:         }
}// -----

func private @test_lstm_forward_mode_constant_weight_and_bias(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x2x4xf32>, %arg2: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %w = "onnx.Constant"() {value = dense<[[[1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]]> : tensor<1x16x3xf32>} : () -> tensor<1x16x3xf32> 
  %r = "onnx.Constant"() {value = dense<[[[2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.]]]> : tensor<1x16x4xf32>} : () -> tensor<1x16x4xf32> 
  %b = "onnx.Constant"() {value = dense<[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32.]]> : tensor<1x32xf32>} : () -> tensor<1x32xf32> 
  %p = "onnx.Constant"() {value = dense<[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]]> : tensor<1x12xf32>} : () -> tensor<1x12xf32> 

  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %w, %r, %b, %cst, %arg1, %arg2, %p) {hidden_size = 4 : si64} : (tensor<7x2x3xf32>, tensor<1x16x3xf32>, tensor<1x16x4xf32>, tensor<1x32xf32>, none, tensor<1x2x4xf32>, tensor<1x2x4xf32>, tensor<1x12xf32>) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>
// CHECK-LABEL:  func private @test_lstm_forward_mode_constant_weight_and_bias
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x2x4xf32>, [[PARAM_2_:%.+]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.alloc() : memref<2x4xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.alloc() : memref<2x4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = memref.alloc() : memref<1x2x4xf32>
// CHECK-DAG:       [[VAR_cst_:%.+]] = constant unit
// CHECK-DAG:       [[VAR_3_:%.+]] = "krnl.global"() {name = "constant_0", shape = [1, 16, 3], value = dense<1.000000e+00> : tensor<1x16x3xf32>} : () -> memref<1x16x3xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "krnl.global"() {name = "constant_1", shape = [1, 16, 4], value = dense<2.000000e+00> : tensor<1x16x4xf32>} : () -> memref<1x16x4xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "krnl.global"() {name = "constant_2", shape = [1, 32], value = dense<{{.}}[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01]{{.}}> : tensor<1x32xf32>} : () -> memref<1x32xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "krnl.global"() {name = "constant_3", shape = [1, 12], value = dense<{{.}}[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01]{{.}}> : tensor<1x12xf32>} : () -> memref<1x12xf32>
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = constant 1 : index
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4) {
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[VAR_1_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:             [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_2_MEM_]], [[VAR_0_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_8_:%.+]] = "krnl.global"() {name = "constant_4", shape = [16, 3], value = dense<1.000000e+00> : tensor<16x3xf32>} : () -> memref<16x3xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "krnl.global"() {name = "constant_5", shape = [16, 4], value = dense<2.000000e+00> : tensor<16x4xf32>} : () -> memref<16x4xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "krnl.global"() {name = "constant_6", shape = [4, 3], value = dense<1.000000e+00> : tensor<4x3xf32>} : () -> memref<4x3xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "krnl.global"() {name = "constant_7", shape = [4, 3], value = dense<1.000000e+00> : tensor<4x3xf32>} : () -> memref<4x3xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "krnl.global"() {name = "constant_8", shape = [4, 3], value = dense<1.000000e+00> : tensor<4x3xf32>} : () -> memref<4x3xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "krnl.global"() {name = "constant_9", shape = [4, 3], value = dense<1.000000e+00> : tensor<4x3xf32>} : () -> memref<4x3xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = "krnl.global"() {name = "constant_10", shape = [3, 4], value = dense<1.000000e+00> : tensor<3x4xf32>} : () -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = "krnl.global"() {name = "constant_11", shape = [3, 4], value = dense<1.000000e+00> : tensor<3x4xf32>} : () -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "krnl.global"() {name = "constant_12", shape = [3, 4], value = dense<1.000000e+00> : tensor<3x4xf32>} : () -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = "krnl.global"() {name = "constant_13", shape = [3, 4], value = dense<1.000000e+00> : tensor<3x4xf32>} : () -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "krnl.global"() {name = "constant_14", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "krnl.global"() {name = "constant_15", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = "krnl.global"() {name = "constant_16", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "krnl.global"() {name = "constant_17", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = "krnl.global"() {name = "constant_18", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_23_:%.+]] = "krnl.global"() {name = "constant_19", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_24_:%.+]] = "krnl.global"() {name = "constant_20", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_25_:%.+]] = "krnl.global"() {name = "constant_21", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_26_:%.+]] = "krnl.global"() {name = "constant_22", shape = [32], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01]> : tensor<32xf32>} : () -> memref<32xf32>
// CHECK-DAG:       [[VAR_27_:%.+]] = "krnl.global"() {name = "constant_23", shape = [4], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_28_:%.+]] = "krnl.global"() {name = "constant_24", shape = [4], value = dense<[5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_29_:%.+]] = "krnl.global"() {name = "constant_25", shape = [4], value = dense<[9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_30_:%.+]] = "krnl.global"() {name = "constant_26", shape = [4], value = dense<[1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_31_:%.+]] = "krnl.global"() {name = "constant_27", shape = [4], value = dense<[1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_32_:%.+]] = "krnl.global"() {name = "constant_28", shape = [4], value = dense<[2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_33_:%.+]] = "krnl.global"() {name = "constant_29", shape = [4], value = dense<[2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_34_:%.+]] = "krnl.global"() {name = "constant_30", shape = [4], value = dense<[2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_35_:%.+]] = "krnl.global"() {name = "constant_31", shape = [12], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01]> : tensor<12xf32>} : () -> memref<12xf32>
// CHECK-DAG:       [[VAR_36_:%.+]] = "krnl.global"() {name = "constant_32", shape = [4], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_37_:%.+]] = "krnl.global"() {name = "constant_33", shape = [4], value = dense<[5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_38_:%.+]] = "krnl.global"() {name = "constant_34", shape = [4], value = dense<[9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7) {
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_1_:%.+]] = memref.alloc() : memref<2x3xf32>
// CHECK-DAG:         [[CST_0_1_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_3_:%.+]] = constant 3 : index
// CHECK-DAG:         [[CST_0_2_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_3_:%.+]] = constant 0 : index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_2_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_3_]] to [[CST_3_]]) {
// CHECK:               [[VAR_51_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_51_]]#0, [[VAR_51_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_1_]]{{.}}[[VAR_51_]]#0, [[VAR_51_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_42_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_1_MEM_1_]], [[VAR_14_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_43_:%.+]] = "onnx.MatMul"([[VAR_1_]], [[VAR_22_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_44_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_1_MEM_1_]], [[VAR_16_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_45_:%.+]] = "onnx.MatMul"([[VAR_1_]], [[VAR_24_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_46_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_1_MEM_1_]], [[VAR_17_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_47_:%.+]] = "onnx.MatMul"([[VAR_1_]], [[VAR_25_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_48_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_1_MEM_1_]], [[VAR_15_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_49_:%.+]] = "onnx.MatMul"([[VAR_1_]], [[VAR_23_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[CST_0_4_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_5_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_1_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_1_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_4_]] to [[CST_2_1_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_4_]] to [[CST_4_]]) {
// CHECK:               [[VAR_51_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[VAR_51_1_]]#0, [[VAR_51_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_42_MEM_:%.+]] = krnl.load [[VAR_42_]]{{.}}[[VAR_51_1_]]#0, [[VAR_51_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_43_MEM_:%.+]] = krnl.load [[VAR_43_]]{{.}}[[VAR_51_1_]]#0, [[VAR_51_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_55_:%.+]] = addf [[LOAD_VAR_42_MEM_]], [[LOAD_VAR_43_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_27_MEM_:%.+]] = krnl.load [[VAR_27_]]{{.}}[[VAR_51_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_31_MEM_:%.+]] = krnl.load [[VAR_31_]]{{.}}[[VAR_51_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_58_:%.+]] = addf [[VAR_55_]], [[LOAD_VAR_27_MEM_]] : f32
// CHECK-DAG:           [[VAR_59_:%.+]] = addf [[VAR_58_]], [[LOAD_VAR_31_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_36_MEM_:%.+]] = krnl.load [[VAR_36_]]{{.}}[[VAR_51_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_61_:%.+]] = mulf [[LOAD_VAR_36_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_62_:%.+]] = addf [[VAR_59_]], [[VAR_61_]] : f32
// CHECK-DAG:           [[VAR_63_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_62_]], [[VAR_63_]][] : memref<f32>
// CHECK:               [[VAR_64_:%.+]] = "onnx.Sigmoid"([[VAR_63_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_64_MEM_:%.+]] = krnl.load [[VAR_64_]][] : memref<f32>
// CHECK-DAG:           [[LOAD_VAR_44_MEM_:%.+]] = krnl.load [[VAR_44_]]{{.}}[[VAR_51_1_]]#0, [[VAR_51_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_45_MEM_:%.+]] = krnl.load [[VAR_45_]]{{.}}[[VAR_51_1_]]#0, [[VAR_51_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_68_:%.+]] = addf [[LOAD_VAR_44_MEM_]], [[LOAD_VAR_45_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_29_MEM_:%.+]] = krnl.load [[VAR_29_]]{{.}}[[VAR_51_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_33_MEM_:%.+]] = krnl.load [[VAR_33_]]{{.}}[[VAR_51_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_71_:%.+]] = addf [[VAR_68_]], [[LOAD_VAR_29_MEM_]] : f32
// CHECK-DAG:           [[VAR_72_:%.+]] = addf [[VAR_71_]], [[LOAD_VAR_33_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_38_MEM_:%.+]] = krnl.load [[VAR_38_]]{{.}}[[VAR_51_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_74_:%.+]] = mulf [[LOAD_VAR_38_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_75_:%.+]] = addf [[VAR_72_]], [[VAR_74_]] : f32
// CHECK-DAG:           [[VAR_76_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_75_]], [[VAR_76_]][] : memref<f32>
// CHECK:               [[VAR_77_:%.+]] = "onnx.Sigmoid"([[VAR_76_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_77_MEM_:%.+]] = krnl.load [[VAR_77_]][] : memref<f32>
// CHECK-DAG:           [[LOAD_VAR_46_MEM_:%.+]] = krnl.load [[VAR_46_]]{{.}}[[VAR_51_1_]]#0, [[VAR_51_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_47_MEM_:%.+]] = krnl.load [[VAR_47_]]{{.}}[[VAR_51_1_]]#0, [[VAR_51_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_81_:%.+]] = addf [[LOAD_VAR_46_MEM_]], [[LOAD_VAR_47_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_30_MEM_:%.+]] = krnl.load [[VAR_30_]]{{.}}[[VAR_51_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_34_MEM_:%.+]] = krnl.load [[VAR_34_]]{{.}}[[VAR_51_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_84_:%.+]] = addf [[VAR_81_]], [[LOAD_VAR_30_MEM_]] : f32
// CHECK-DAG:           [[VAR_85_:%.+]] = addf [[VAR_84_]], [[LOAD_VAR_34_MEM_]] : f32
// CHECK-DAG:           [[VAR_86_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_85_]], [[VAR_86_]][] : memref<f32>
// CHECK:               [[VAR_87_:%.+]] = "onnx.Tanh"([[VAR_86_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_87_MEM_:%.+]] = krnl.load [[VAR_87_]][] : memref<f32>
// CHECK-DAG:           [[VAR_89_:%.+]] = mulf [[LOAD_VAR_77_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_90_:%.+]] = mulf [[LOAD_VAR_64_MEM_]], [[LOAD_VAR_87_MEM_]] : f32
// CHECK-DAG:           [[VAR_91_:%.+]] = addf [[VAR_89_]], [[VAR_90_]] : f32
// CHECK-DAG:           [[LOAD_VAR_48_MEM_:%.+]] = krnl.load [[VAR_48_]]{{.}}[[VAR_51_1_]]#0, [[VAR_51_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_49_MEM_:%.+]] = krnl.load [[VAR_49_]]{{.}}[[VAR_51_1_]]#0, [[VAR_51_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_94_:%.+]] = addf [[LOAD_VAR_48_MEM_]], [[LOAD_VAR_49_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_28_MEM_:%.+]] = krnl.load [[VAR_28_]]{{.}}[[VAR_51_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_32_MEM_:%.+]] = krnl.load [[VAR_32_]]{{.}}[[VAR_51_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_97_:%.+]] = addf [[VAR_94_]], [[LOAD_VAR_28_MEM_]] : f32
// CHECK-DAG:           [[VAR_98_:%.+]] = addf [[VAR_97_]], [[LOAD_VAR_32_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_37_MEM_:%.+]] = krnl.load [[VAR_37_]]{{.}}[[VAR_51_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_100_:%.+]] = mulf [[LOAD_VAR_37_MEM_]], [[VAR_91_]] : f32
// CHECK-DAG:           [[VAR_101_:%.+]] = addf [[VAR_98_]], [[VAR_100_]] : f32
// CHECK-DAG:           [[VAR_102_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_101_]], [[VAR_102_]][] : memref<f32>
// CHECK:               [[VAR_103_:%.+]] = "onnx.Sigmoid"([[VAR_102_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_103_MEM_:%.+]] = krnl.load [[VAR_103_]][] : memref<f32>
// CHECK-DAG:           [[VAR_105_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_91_]], [[VAR_105_]][] : memref<f32>
// CHECK:               [[VAR_106_:%.+]] = "onnx.Tanh"([[VAR_105_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_VAR_106_MEM_:%.+]] = krnl.load [[VAR_106_]][] : memref<f32>
// CHECK:               [[VAR_108_:%.+]] = mulf [[LOAD_VAR_103_MEM_]], [[LOAD_VAR_106_MEM_]] : f32
// CHECK:               krnl.store [[VAR_91_]], [[VAR_0_]]{{.}}[[VAR_51_1_]]#0, [[VAR_51_1_]]#1] : memref<2x4xf32>
// CHECK:               krnl.store [[VAR_108_]], [[VAR_1_]]{{.}}[[VAR_51_1_]]#0, [[VAR_51_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc [[LOAD_PARAM_1_MEM_1_]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           [[CST_32_:%.+]] = constant 32 : i64
// CHECK:           "krnl.memcpy"([[VAR_2_]], [[VAR_1_]], [[CST_32_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
// CHECK:           memref.dealloc [[VAR_1_]] : memref<2x4xf32>
// CHECK:           memref.dealloc [[VAR_0_]] : memref<2x4xf32>
// CHECK:           return [[VAR_2_]] : memref<1x2x4xf32>
// CHECK:         }
}// -----

func private @test_lstm_reverse_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x16x3xf32>, %arg2: tensor<1x16x4xf32>, %arg3: tensor<1x32xf32>, %arg4: tensor<1x2x4xf32>, %arg5: tensor<1x2x4xf32>, %arg6: tensor<1x12xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4, %arg5, %arg6) {hidden_size = 4 : si64, direction = "reverse"} : (tensor<7x2x3xf32>, tensor<1x16x3xf32>, tensor<1x16x4xf32>, tensor<1x32xf32>, none, tensor<1x2x4xf32>, tensor<1x2x4xf32>, tensor<1x12xf32>) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>
  // CHECK-DAG: #map = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
// CHECK-LABEL:  func private @test_lstm_reverse_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x16x3xf32>, [[PARAM_2_:%.+]]: memref<1x16x4xf32>, [[PARAM_3_:%.+]]: memref<1x32xf32>, [[PARAM_4_:%.+]]: memref<1x2x4xf32>, [[PARAM_5_:%.+]]: memref<1x2x4xf32>, [[PARAM_6_:%.+]]: memref<1x12xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.alloc() : memref<2x4xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.alloc() : memref<2x4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = memref.alloc() : memref<1x2x4xf32>
// CHECK-DAG:       [[VAR_cst_:%.+]] = constant unit
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = constant 1 : index
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4) {
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[VAR_1_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:             [[LOAD_PARAM_5_MEM_:%.+]] = krnl.load [[PARAM_5_]]{{.}}[[CST_0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_5_MEM_]], [[VAR_0_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Squeeze"([[PARAM_1_]]) {axes = [0]} : (memref<1x16x3xf32>) -> memref<16x3xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Squeeze"([[PARAM_2_]]) {axes = [0]} : (memref<1x16x4xf32>) -> memref<16x4xf32>
// CHECK:           [[VAR_6_:%.+]]:4 = "onnx.Split"([[VAR_4_]]) {axis = 0 : si64} : (memref<16x3xf32>) -> (memref<4x3xf32>, memref<4x3xf32>, memref<4x3xf32>, memref<4x3xf32>)
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Transpose"([[VAR_6_]]#0) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Transpose"([[VAR_6_]]#1) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Transpose"([[VAR_6_]]#2) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Transpose"([[VAR_6_]]#3) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_11_:%.+]]:4 = "onnx.Split"([[VAR_5_]]) {axis = 0 : si64} : (memref<16x4xf32>) -> (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Transpose"([[VAR_11_]]#0) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Transpose"([[VAR_11_]]#1) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.Transpose"([[VAR_11_]]#2) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.Transpose"([[VAR_11_]]#3) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Squeeze"([[PARAM_3_]]) {axes = [0]} : (memref<1x32xf32>) -> memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_17_:%.+]]:8 = "onnx.Split"([[VAR_16_]]) {axis = 0 : si64} : (memref<32xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Squeeze"([[PARAM_6_]]) {axes = [0]} : (memref<1x12xf32>) -> memref<12xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_19_:%.+]]:3 = "onnx.Split"([[VAR_18_]]) {axis = 0 : si64} : (memref<12xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7) {
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_1_:%.+]] = memref.alloc() : memref<2x3xf32>
// CHECK-DAG:         [[CST_0_1_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_7_:%.+]] = constant 7 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_5_MEM_1_:%.+]] = affine.apply #map([[I_2_]]){{.}}[[CST_7_]]{{.}}
// CHECK-DAG:         [[CST_2_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_3_:%.+]] = constant 3 : index
// CHECK-DAG:         [[CST_0_2_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_3_:%.+]] = constant 0 : index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_2_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_3_]] to [[CST_3_]]) {
// CHECK:               [[VAR_33_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[LOAD_PARAM_5_MEM_1_]], [[VAR_33_]]#0, [[VAR_33_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_4_MEM_1_]]{{.}}[[VAR_33_]]#0, [[VAR_33_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_24_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_4_MEM_1_]], [[VAR_7_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_25_:%.+]] = "onnx.MatMul"([[VAR_1_]], [[VAR_1_]]2) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_26_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_4_MEM_1_]], [[VAR_9_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_27_:%.+]] = "onnx.MatMul"([[VAR_1_]], [[VAR_1_]]4) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_28_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_4_MEM_1_]], [[VAR_10_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_29_:%.+]] = "onnx.MatMul"([[VAR_1_]], [[VAR_1_]]5) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_30_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_4_MEM_1_]], [[VAR_8_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_31_:%.+]] = "onnx.MatMul"([[VAR_1_]], [[VAR_1_]]3) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[CST_0_4_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_5_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_1_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_1_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_4_]] to [[CST_2_1_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_4_]] to [[CST_4_]]) {
// CHECK:               [[VAR_33_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[VAR_33_1_]]#0, [[VAR_33_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_24_MEM_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_33_1_]]#0, [[VAR_33_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_25_MEM_:%.+]] = krnl.load [[VAR_25_]]{{.}}[[VAR_33_1_]]#0, [[VAR_33_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_37_:%.+]] = addf [[LOAD_VAR_24_MEM_]], [[LOAD_VAR_25_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_17_MEM_:%.+]] = krnl.load [[VAR_17_]]#0{{.}}[[VAR_33_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_17_MEM_1_:%.+]] = krnl.load [[VAR_17_]]#4{{.}}[[VAR_33_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_40_:%.+]] = addf [[VAR_37_]], [[LOAD_VAR_17_MEM_]] : f32
// CHECK-DAG:           [[VAR_41_:%.+]] = addf [[VAR_40_]], [[LOAD_VAR_17_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_19_MEM_:%.+]] = krnl.load [[VAR_19_]]#0{{.}}[[VAR_33_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_43_:%.+]] = mulf [[LOAD_VAR_19_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_44_:%.+]] = addf [[VAR_41_]], [[VAR_43_]] : f32
// CHECK-DAG:           [[VAR_45_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_44_]], [[VAR_45_]][] : memref<f32>
// CHECK:               [[VAR_46_:%.+]] = "onnx.Sigmoid"([[VAR_45_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_46_MEM_:%.+]] = krnl.load [[VAR_46_]][] : memref<f32>
// CHECK-DAG:           [[LOAD_VAR_26_MEM_:%.+]] = krnl.load [[VAR_26_]]{{.}}[[VAR_33_1_]]#0, [[VAR_33_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_27_MEM_:%.+]] = krnl.load [[VAR_27_]]{{.}}[[VAR_33_1_]]#0, [[VAR_33_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_50_:%.+]] = addf [[LOAD_VAR_26_MEM_]], [[LOAD_VAR_27_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_17_MEM_2_:%.+]] = krnl.load [[VAR_17_]]#2{{.}}[[VAR_33_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_17_MEM_3_:%.+]] = krnl.load [[VAR_17_]]#6{{.}}[[VAR_33_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_53_:%.+]] = addf [[VAR_50_]], [[LOAD_VAR_17_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_54_:%.+]] = addf [[VAR_53_]], [[LOAD_VAR_17_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_19_MEM_1_:%.+]] = krnl.load [[VAR_19_]]#2{{.}}[[VAR_33_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_56_:%.+]] = mulf [[LOAD_VAR_19_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_57_:%.+]] = addf [[VAR_54_]], [[VAR_56_]] : f32
// CHECK-DAG:           [[VAR_58_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_57_]], [[VAR_58_]][] : memref<f32>
// CHECK:               [[VAR_59_:%.+]] = "onnx.Sigmoid"([[VAR_58_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_59_MEM_:%.+]] = krnl.load [[VAR_59_]][] : memref<f32>
// CHECK-DAG:           [[LOAD_VAR_28_MEM_:%.+]] = krnl.load [[VAR_28_]]{{.}}[[VAR_33_1_]]#0, [[VAR_33_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_29_MEM_:%.+]] = krnl.load [[VAR_29_]]{{.}}[[VAR_33_1_]]#0, [[VAR_33_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_63_:%.+]] = addf [[LOAD_VAR_28_MEM_]], [[LOAD_VAR_29_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_17_MEM_4_:%.+]] = krnl.load [[VAR_17_]]#3{{.}}[[VAR_33_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_17_MEM_5_:%.+]] = krnl.load [[VAR_17_]]#7{{.}}[[VAR_33_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_66_:%.+]] = addf [[VAR_63_]], [[LOAD_VAR_17_MEM_4_]] : f32
// CHECK-DAG:           [[VAR_67_:%.+]] = addf [[VAR_66_]], [[LOAD_VAR_17_MEM_5_]] : f32
// CHECK-DAG:           [[VAR_68_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_67_]], [[VAR_68_]][] : memref<f32>
// CHECK:               [[VAR_69_:%.+]] = "onnx.Tanh"([[VAR_68_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_69_MEM_:%.+]] = krnl.load [[VAR_69_]][] : memref<f32>
// CHECK-DAG:           [[VAR_71_:%.+]] = mulf [[LOAD_VAR_59_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_72_:%.+]] = mulf [[LOAD_VAR_46_MEM_]], [[LOAD_VAR_69_MEM_]] : f32
// CHECK-DAG:           [[VAR_73_:%.+]] = addf [[VAR_71_]], [[VAR_72_]] : f32
// CHECK-DAG:           [[LOAD_VAR_30_MEM_:%.+]] = krnl.load [[VAR_30_]]{{.}}[[VAR_33_1_]]#0, [[VAR_33_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_31_MEM_:%.+]] = krnl.load [[VAR_31_]]{{.}}[[VAR_33_1_]]#0, [[VAR_33_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_76_:%.+]] = addf [[LOAD_VAR_30_MEM_]], [[LOAD_VAR_31_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_17_MEM_6_:%.+]] = krnl.load [[VAR_17_]]#1{{.}}[[VAR_33_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_17_MEM_7_:%.+]] = krnl.load [[VAR_17_]]#5{{.}}[[VAR_33_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_79_:%.+]] = addf [[VAR_76_]], [[LOAD_VAR_17_MEM_6_]] : f32
// CHECK-DAG:           [[VAR_80_:%.+]] = addf [[VAR_79_]], [[LOAD_VAR_17_MEM_7_]] : f32
// CHECK-DAG:           [[LOAD_VAR_19_MEM_2_:%.+]] = krnl.load [[VAR_19_]]#1{{.}}[[VAR_33_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_82_:%.+]] = mulf [[LOAD_VAR_19_MEM_2_]], [[VAR_73_]] : f32
// CHECK-DAG:           [[VAR_83_:%.+]] = addf [[VAR_80_]], [[VAR_82_]] : f32
// CHECK-DAG:           [[VAR_84_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_83_]], [[VAR_84_]][] : memref<f32>
// CHECK:               [[VAR_85_:%.+]] = "onnx.Sigmoid"([[VAR_84_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_85_MEM_:%.+]] = krnl.load [[VAR_85_]][] : memref<f32>
// CHECK-DAG:           [[VAR_87_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_73_]], [[VAR_87_]][] : memref<f32>
// CHECK:               [[VAR_88_:%.+]] = "onnx.Tanh"([[VAR_87_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_VAR_88_MEM_:%.+]] = krnl.load [[VAR_88_]][] : memref<f32>
// CHECK:               [[VAR_90_:%.+]] = mulf [[LOAD_VAR_85_MEM_]], [[LOAD_VAR_88_MEM_]] : f32
// CHECK:               krnl.store [[VAR_73_]], [[VAR_0_]]{{.}}[[VAR_33_1_]]#0, [[VAR_33_1_]]#1] : memref<2x4xf32>
// CHECK:               krnl.store [[VAR_90_]], [[VAR_1_]]{{.}}[[VAR_33_1_]]#0, [[VAR_33_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc [[LOAD_PARAM_4_MEM_1_]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           [[CST_32_:%.+]] = constant 32 : i64
// CHECK:           "krnl.memcpy"([[VAR_2_]], [[VAR_1_]], [[CST_32_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
// CHECK:           memref.dealloc [[VAR_1_]] : memref<2x4xf32>
// CHECK:           memref.dealloc [[VAR_0_]] : memref<2x4xf32>
// CHECK:           return [[VAR_2_]] : memref<1x2x4xf32>
// CHECK:         }
}// -----

func private @test_lstm_bidirectional_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<2x16x3xf32>, %arg2: tensor<2x16x4xf32>, %arg3: tensor<2x32xf32>, %arg4: tensor<2x2x4xf32>, %arg5: tensor<2x2x4xf32>, %arg6: tensor<2x12xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4, %arg5, %arg6) {hidden_size = 4 : si64, direction = "bidirectional"} : (tensor<7x2x3xf32>, tensor<2x16x3xf32>, tensor<2x16x4xf32>, tensor<2x32xf32>, none, tensor<2x2x4xf32>, tensor<2x2x4xf32>, tensor<2x12xf32>) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>
// CHECK-DAG: #map = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
// CHECK-LABEL:  func private @test_lstm_bidirectional_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<2x16x3xf32>, [[PARAM_2_:%.+]]: memref<2x16x4xf32>, [[PARAM_3_:%.+]]: memref<2x32xf32>, [[PARAM_4_:%.+]]: memref<2x2x4xf32>, [[PARAM_5_:%.+]]: memref<2x2x4xf32>, [[PARAM_6_:%.+]]: memref<2x12xf32>) -> memref<2x2x4xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.alloc() : memref<2x4xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.alloc() : memref<2x4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = memref.alloc() : memref<2x4xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = memref.alloc() : memref<2x4xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = memref.alloc() : memref<2x2x4xf32>
// CHECK-DAG:       [[VAR_cst_:%.+]] = constant unit
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = constant 1 : index
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4) {
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_]], [[I_0_]], [[I_1_]]{{.}} : memref<2x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[VAR_3_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:             [[LOAD_PARAM_5_MEM_:%.+]] = krnl.load [[PARAM_5_]]{{.}}[[CST_0_]], [[I_0_]], [[I_1_]]{{.}} : memref<2x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_5_MEM_]], [[VAR_2_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:             [[LOAD_PARAM_4_MEM_1_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_1_]], [[I_0_]], [[I_1_]]{{.}} : memref<2x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_1_]], [[VAR_1_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:             [[LOAD_PARAM_5_MEM_1_:%.+]] = krnl.load [[PARAM_5_]]{{.}}[[CST_1_]], [[I_0_]], [[I_1_]]{{.}} : memref<2x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_5_MEM_1_]], [[VAR_0_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:           }
// CHECK:           [[VAR_6_:%.+]]:2 = "onnx.Split"([[PARAM_1_]]) {axis = 0 : si64} : (memref<2x16x3xf32>) -> (memref<1x16x3xf32>, memref<1x16x3xf32>)
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Squeeze"([[VAR_6_]]#0) {axes = [0]} : (memref<1x16x3xf32>) -> memref<16x3xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Squeeze"([[VAR_6_]]#1) {axes = [0]} : (memref<1x16x3xf32>) -> memref<16x3xf32>
// CHECK-DAG:       [[VAR_9_:%.+]]:2 = "onnx.Split"([[PARAM_2_]]) {axis = 0 : si64} : (memref<2x16x4xf32>) -> (memref<1x16x4xf32>, memref<1x16x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Squeeze"([[VAR_9_]]#0) {axes = [0]} : (memref<1x16x4xf32>) -> memref<16x4xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Squeeze"([[VAR_9_]]#1) {axes = [0]} : (memref<1x16x4xf32>) -> memref<16x4xf32>
// CHECK-DAG:       [[VAR_12_:%.+]]:4 = "onnx.Split"([[VAR_7_]]) {axis = 0 : si64} : (memref<16x3xf32>) -> (memref<4x3xf32>, memref<4x3xf32>, memref<4x3xf32>, memref<4x3xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Transpose"([[VAR_12_]]#0) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.Transpose"([[VAR_12_]]#1) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.Transpose"([[VAR_12_]]#2) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Transpose"([[VAR_12_]]#3) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_17_:%.+]]:4 = "onnx.Split"([[VAR_10_]]) {axis = 0 : si64} : (memref<16x4xf32>) -> (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Transpose"([[VAR_17_]]#0) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.Transpose"([[VAR_17_]]#1) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.Transpose"([[VAR_17_]]#2) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.Transpose"([[VAR_17_]]#3) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_22_:%.+]]:4 = "onnx.Split"([[VAR_8_]]) {axis = 0 : si64} : (memref<16x3xf32>) -> (memref<4x3xf32>, memref<4x3xf32>, memref<4x3xf32>, memref<4x3xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_23_:%.+]] = "onnx.Transpose"([[VAR_22_]]#0) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_24_:%.+]] = "onnx.Transpose"([[VAR_22_]]#1) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_25_:%.+]] = "onnx.Transpose"([[VAR_22_]]#2) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_26_:%.+]] = "onnx.Transpose"([[VAR_22_]]#3) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_27_:%.+]]:4 = "onnx.Split"([[VAR_11_]]) {axis = 0 : si64} : (memref<16x4xf32>) -> (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_28_:%.+]] = "onnx.Transpose"([[VAR_27_]]#0) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_29_:%.+]] = "onnx.Transpose"([[VAR_27_]]#1) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_30_:%.+]] = "onnx.Transpose"([[VAR_27_]]#2) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_31_:%.+]] = "onnx.Transpose"([[VAR_27_]]#3) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_32_:%.+]]:2 = "onnx.Split"([[PARAM_3_]]) {axis = 0 : si64} : (memref<2x32xf32>) -> (memref<1x32xf32>, memref<1x32xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_33_:%.+]] = "onnx.Squeeze"([[VAR_32_]]#0) {axes = [0]} : (memref<1x32xf32>) -> memref<32xf32>
// CHECK-DAG:       [[VAR_34_:%.+]] = "onnx.Squeeze"([[VAR_32_]]#1) {axes = [0]} : (memref<1x32xf32>) -> memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_35_:%.+]]:8 = "onnx.Split"([[VAR_33_]]) {axis = 0 : si64} : (memref<32xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[VAR_36_:%.+]]:8 = "onnx.Split"([[VAR_34_]]) {axis = 0 : si64} : (memref<32xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[VAR_37_:%.+]]:2 = "onnx.Split"([[PARAM_6_]]) {axis = 0 : si64} : (memref<2x12xf32>) -> (memref<1x12xf32>, memref<1x12xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_38_:%.+]] = "onnx.Squeeze"([[VAR_37_]]#0) {axes = [0]} : (memref<1x12xf32>) -> memref<12xf32>
// CHECK-DAG:       [[VAR_39_:%.+]] = "onnx.Squeeze"([[VAR_37_]]#1) {axes = [0]} : (memref<1x12xf32>) -> memref<12xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_40_:%.+]]:3 = "onnx.Split"([[VAR_38_]]) {axis = 0 : si64} : (memref<12xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[VAR_41_:%.+]]:3 = "onnx.Split"([[VAR_39_]]) {axis = 0 : si64} : (memref<12xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7) {
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_2_:%.+]] = memref.alloc() : memref<2x3xf32>
// CHECK-DAG:         [[CST_0_1_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_3_:%.+]] = constant 3 : index
// CHECK-DAG:         [[CST_0_2_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_3_:%.+]] = constant 0 : index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_2_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_3_]] to [[CST_3_]]) {
// CHECK:               [[VAR_56_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_56_]]#0, [[VAR_56_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_4_MEM_2_]]{{.}}[[VAR_56_]]#0, [[VAR_56_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_1_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_4_MEM_2_]], [[VAR_13_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[LOAD_PARAM_5_MEM_1_:%.+]] = "onnx.MatMul"([[VAR_3_]], [[VAR_18_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_49_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_4_MEM_2_]], [[VAR_15_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_50_:%.+]] = "onnx.MatMul"([[VAR_3_]], [[VAR_20_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_51_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_4_MEM_2_]], [[VAR_16_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_52_:%.+]] = "onnx.MatMul"([[VAR_3_]], [[VAR_21_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_53_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_4_MEM_2_]], [[VAR_14_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_54_:%.+]] = "onnx.MatMul"([[VAR_3_]], [[VAR_19_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[CST_0_4_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_5_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_1_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_1_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_4_]] to [[CST_2_1_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_4_]] to [[CST_4_]]) {
// CHECK:               [[VAR_56_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_2_]]{{.}}[[VAR_56_1_]]#0, [[VAR_56_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_LOAD_PARAM_4_MEM_1_MEM_:%.+]] = krnl.load [[LOAD_PARAM_4_MEM_1_]]{{.}}[[VAR_56_1_]]#0, [[VAR_56_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_LOAD_PARAM_5_MEM_1_MEM_:%.+]] = krnl.load [[LOAD_PARAM_5_MEM_1_]]{{.}}[[VAR_56_1_]]#0, [[VAR_56_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_60_:%.+]] = addf [[LOAD_LOAD_PARAM_4_MEM_1_MEM_]], [[LOAD_LOAD_PARAM_5_MEM_1_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_35_MEM_:%.+]] = krnl.load [[VAR_35_]]#0{{.}}[[VAR_56_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_35_MEM_1_:%.+]] = krnl.load [[VAR_35_]]#4{{.}}[[VAR_56_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_63_:%.+]] = addf [[VAR_60_]], [[LOAD_VAR_35_MEM_]] : f32
// CHECK-DAG:           [[VAR_64_:%.+]] = addf [[VAR_63_]], [[LOAD_VAR_35_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_40_MEM_:%.+]] = krnl.load [[VAR_40_]]#0{{.}}[[VAR_56_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_66_:%.+]] = mulf [[LOAD_VAR_40_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_67_:%.+]] = addf [[VAR_64_]], [[VAR_66_]] : f32
// CHECK-DAG:           [[VAR_68_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_67_]], [[VAR_68_]][] : memref<f32>
// CHECK:               [[VAR_69_:%.+]] = "onnx.Sigmoid"([[VAR_68_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_69_MEM_:%.+]] = krnl.load [[VAR_69_]][] : memref<f32>
// CHECK-DAG:           [[LOAD_VAR_49_MEM_:%.+]] = krnl.load [[VAR_49_]]{{.}}[[VAR_56_1_]]#0, [[VAR_56_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_50_MEM_:%.+]] = krnl.load [[VAR_50_]]{{.}}[[VAR_56_1_]]#0, [[VAR_56_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_73_:%.+]] = addf [[LOAD_VAR_49_MEM_]], [[LOAD_VAR_50_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_35_MEM_2_:%.+]] = krnl.load [[VAR_35_]]#2{{.}}[[VAR_56_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_35_MEM_3_:%.+]] = krnl.load [[VAR_35_]]#6{{.}}[[VAR_56_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_76_:%.+]] = addf [[VAR_73_]], [[LOAD_VAR_35_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_77_:%.+]] = addf [[VAR_76_]], [[LOAD_VAR_35_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_40_MEM_1_:%.+]] = krnl.load [[VAR_40_]]#2{{.}}[[VAR_56_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_79_:%.+]] = mulf [[LOAD_VAR_40_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_80_:%.+]] = addf [[VAR_77_]], [[VAR_79_]] : f32
// CHECK-DAG:           [[VAR_81_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_80_]], [[VAR_81_]][] : memref<f32>
// CHECK:               [[VAR_82_:%.+]] = "onnx.Sigmoid"([[VAR_81_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_82_MEM_:%.+]] = krnl.load [[VAR_82_]][] : memref<f32>
// CHECK-DAG:           [[LOAD_VAR_51_MEM_:%.+]] = krnl.load [[VAR_51_]]{{.}}[[VAR_56_1_]]#0, [[VAR_56_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_52_MEM_:%.+]] = krnl.load [[VAR_52_]]{{.}}[[VAR_56_1_]]#0, [[VAR_56_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_86_:%.+]] = addf [[LOAD_VAR_51_MEM_]], [[LOAD_VAR_52_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_35_MEM_4_:%.+]] = krnl.load [[VAR_35_]]#3{{.}}[[VAR_56_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_35_MEM_5_:%.+]] = krnl.load [[VAR_35_]]#7{{.}}[[VAR_56_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_89_:%.+]] = addf [[VAR_86_]], [[LOAD_VAR_35_MEM_4_]] : f32
// CHECK-DAG:           [[VAR_90_:%.+]] = addf [[VAR_89_]], [[LOAD_VAR_35_MEM_5_]] : f32
// CHECK-DAG:           [[VAR_91_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_90_]], [[VAR_91_]][] : memref<f32>
// CHECK:               [[VAR_92_:%.+]] = "onnx.Tanh"([[VAR_91_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_92_MEM_:%.+]] = krnl.load [[VAR_92_]][] : memref<f32>
// CHECK-DAG:           [[VAR_94_:%.+]] = mulf [[LOAD_VAR_82_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_95_:%.+]] = mulf [[LOAD_VAR_69_MEM_]], [[LOAD_VAR_92_MEM_]] : f32
// CHECK-DAG:           [[VAR_96_:%.+]] = addf [[VAR_94_]], [[VAR_95_]] : f32
// CHECK-DAG:           [[LOAD_VAR_53_MEM_:%.+]] = krnl.load [[VAR_53_]]{{.}}[[VAR_56_1_]]#0, [[VAR_56_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_54_MEM_:%.+]] = krnl.load [[VAR_54_]]{{.}}[[VAR_56_1_]]#0, [[VAR_56_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_99_:%.+]] = addf [[LOAD_VAR_53_MEM_]], [[LOAD_VAR_54_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_35_MEM_6_:%.+]] = krnl.load [[VAR_35_]]#1{{.}}[[VAR_56_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_35_MEM_7_:%.+]] = krnl.load [[VAR_35_]]#5{{.}}[[VAR_56_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_102_:%.+]] = addf [[VAR_99_]], [[LOAD_VAR_35_MEM_6_]] : f32
// CHECK-DAG:           [[VAR_103_:%.+]] = addf [[VAR_102_]], [[LOAD_VAR_35_MEM_7_]] : f32
// CHECK-DAG:           [[LOAD_VAR_40_MEM_2_:%.+]] = krnl.load [[VAR_40_]]#1{{.}}[[VAR_56_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_105_:%.+]] = mulf [[LOAD_VAR_40_MEM_2_]], [[VAR_96_]] : f32
// CHECK-DAG:           [[VAR_106_:%.+]] = addf [[VAR_103_]], [[VAR_105_]] : f32
// CHECK-DAG:           [[VAR_107_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_106_]], [[VAR_107_]][] : memref<f32>
// CHECK:               [[VAR_108_:%.+]] = "onnx.Sigmoid"([[VAR_107_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_108_MEM_:%.+]] = krnl.load [[VAR_108_]][] : memref<f32>
// CHECK-DAG:           [[VAR_110_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_96_]], [[VAR_110_]][] : memref<f32>
// CHECK:               [[VAR_111_:%.+]] = "onnx.Tanh"([[VAR_110_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_VAR_111_MEM_:%.+]] = krnl.load [[VAR_111_]][] : memref<f32>
// CHECK:               [[VAR_113_:%.+]] = mulf [[LOAD_VAR_108_MEM_]], [[LOAD_VAR_111_MEM_]] : f32
// CHECK:               krnl.store [[VAR_96_]], [[VAR_2_]]{{.}}[[VAR_56_1_]]#0, [[VAR_56_1_]]#1] : memref<2x4xf32>
// CHECK:               krnl.store [[VAR_113_]], [[VAR_3_]]{{.}}[[VAR_56_1_]]#0, [[VAR_56_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc [[LOAD_PARAM_4_MEM_2_]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_4_]]) with ([[LOOP_4_]] -> [[I_7_:%.+]] = 0 to 7) {
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_2_:%.+]] = memref.alloc() : memref<2x3xf32>
// CHECK-DAG:         [[CST_1_2_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_7_:%.+]] = constant 7 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOOP_2_:%.+]] = affine.apply #map([[I_7_]]){{.}}[[CST_7_]]{{.}}
// CHECK-DAG:         [[CST_2_2_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_3_1_:%.+]] = constant 3 : index
// CHECK-DAG:         [[CST_0_6_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_7_:%.+]] = constant 0 : index
// CHECK-DAG:         [[LOOP_5_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_5_]]#0, [[LOOP_5_]]#1) with ([[LOOP_5_]]#0 -> [[I_8_:%.+]] = [[CST_0_6_]] to [[CST_2_2_]], [[LOOP_5_]]#1 -> [[I_9_:%.+]] = [[CST_0_7_]] to [[CST_3_1_]]) {
// CHECK:               [[LOAD_PARAM_0_MEM_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_5_]]#0, [[LOOP_5_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[LOOP_2_]], [[LOAD_PARAM_0_MEM_1_]]#0, [[LOAD_PARAM_0_MEM_1_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_2_]], [[LOAD_PARAM_4_MEM_2_]]{{.}}[[LOAD_PARAM_0_MEM_1_]]#0, [[LOAD_PARAM_0_MEM_1_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[LOAD_PARAM_5_MEM_1_1_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_4_MEM_2_]], [[VAR_23_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_49_1_:%.+]] = "onnx.MatMul"([[VAR_1_]], [[VAR_28_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_50_1_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_4_MEM_2_]], [[VAR_25_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_51_1_:%.+]] = "onnx.MatMul"([[VAR_1_]], [[VAR_30_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_52_1_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_4_MEM_2_]], [[VAR_26_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_53_1_:%.+]] = "onnx.MatMul"([[VAR_1_]], [[VAR_31_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_54_1_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_4_MEM_2_]], [[VAR_24_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]] = "onnx.MatMul"([[VAR_1_]], [[VAR_29_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[CST_0_8_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_9_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_3_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_3_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_4_1_:%.+]] = constant 4 : index
// CHECK-DAG:         [[LOOP_6_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_6_]]#0, [[LOOP_6_]]#1) with ([[LOOP_6_]]#0 -> [[I_10_:%.+]] = [[CST_0_8_]] to [[CST_2_3_]], [[LOOP_6_]]#1 -> [[I_11_:%.+]] = [[CST_0_8_]] to [[CST_4_1_]]) {
// CHECK:               [[LOAD_PARAM_0_MEM_1_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_6_]]#0, [[LOOP_6_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_LOAD_PARAM_5_MEM_1_MEM_1_:%.+]] = krnl.load [[LOAD_PARAM_5_MEM_1_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[VAR_60_1_:%.+]] = krnl.load [[VAR_49_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_35_MEM_8_:%.+]] = addf [[LOAD_LOAD_PARAM_5_MEM_1_MEM_1_]], [[VAR_60_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_35_MEM_1_:%.+]] = krnl.load [[VAR_36_]]#0{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_63_1_:%.+]] = krnl.load [[VAR_36_]]#4{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_64_1_:%.+]] = addf [[LOAD_VAR_35_MEM_8_]], [[LOAD_VAR_35_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_40_MEM_3_:%.+]] = addf [[VAR_64_1_]], [[VAR_63_1_]] : f32
// CHECK-DAG:           [[VAR_66_1_:%.+]] = krnl.load [[VAR_41_]]#0{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_67_1_:%.+]] = mulf [[VAR_66_1_]], [[LOAD_PARAM_0_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_68_1_:%.+]] = addf [[LOAD_VAR_40_MEM_3_]], [[VAR_67_1_]] : f32
// CHECK-DAG:           [[VAR_69_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_68_1_]], [[VAR_69_1_]][] : memref<f32>
// CHECK:               [[LOAD_VAR_69_MEM_1_:%.+]] = "onnx.Sigmoid"([[VAR_69_1_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_49_MEM_1_:%.+]] = krnl.load [[LOAD_VAR_69_MEM_1_]][] : memref<f32>
// CHECK-DAG:           [[LOAD_VAR_50_MEM_1_:%.+]] = krnl.load [[VAR_50_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[VAR_73_1_:%.+]] = krnl.load [[VAR_51_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_35_MEM_2_:%.+]] = addf [[LOAD_VAR_50_MEM_1_]], [[VAR_73_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_35_MEM_3_:%.+]] = krnl.load [[VAR_36_]]#2{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_76_1_:%.+]] = krnl.load [[VAR_36_]]#6{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_77_1_:%.+]] = addf [[LOAD_VAR_35_MEM_2_]], [[LOAD_VAR_35_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_40_MEM_1_:%.+]] = addf [[VAR_77_1_]], [[VAR_76_1_]] : f32
// CHECK-DAG:           [[VAR_79_1_:%.+]] = krnl.load [[VAR_41_]]#2{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_80_1_:%.+]] = mulf [[VAR_79_1_]], [[LOAD_PARAM_0_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_81_1_:%.+]] = addf [[LOAD_VAR_40_MEM_1_]], [[VAR_80_1_]] : f32
// CHECK-DAG:           [[VAR_82_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_81_1_]], [[VAR_82_1_]][] : memref<f32>
// CHECK:               [[LOAD_VAR_82_MEM_1_:%.+]] = "onnx.Sigmoid"([[VAR_82_1_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_51_MEM_1_:%.+]] = krnl.load [[LOAD_VAR_82_MEM_1_]][] : memref<f32>
// CHECK-DAG:           [[LOAD_VAR_52_MEM_1_:%.+]] = krnl.load [[VAR_52_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[VAR_86_1_:%.+]] = krnl.load [[VAR_53_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_35_MEM_4_:%.+]] = addf [[LOAD_VAR_52_MEM_1_]], [[VAR_86_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_35_MEM_5_:%.+]] = krnl.load [[VAR_36_]]#3{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_89_1_:%.+]] = krnl.load [[VAR_36_]]#7{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_90_1_:%.+]] = addf [[LOAD_VAR_35_MEM_4_]], [[LOAD_VAR_35_MEM_5_]] : f32
// CHECK-DAG:           [[VAR_91_1_:%.+]] = addf [[VAR_90_1_]], [[VAR_89_1_]] : f32
// CHECK-DAG:           [[VAR_92_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_91_1_]], [[VAR_92_1_]][] : memref<f32>
// CHECK:               [[LOAD_VAR_92_MEM_1_:%.+]] = "onnx.Tanh"([[VAR_92_1_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[VAR_94_1_:%.+]] = krnl.load [[LOAD_VAR_92_MEM_1_]][] : memref<f32>
// CHECK-DAG:           [[VAR_95_1_:%.+]] = mulf [[LOAD_VAR_51_MEM_1_]], [[LOAD_PARAM_0_MEM_2_]] : f32
// CHECK:               [[VAR_96_1_:%.+]] = mulf [[LOAD_VAR_49_MEM_1_]], [[VAR_94_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_53_MEM_1_:%.+]] = addf [[VAR_95_1_]], [[VAR_96_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_54_MEM_1_:%.+]] = krnl.load [[VAR_54_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[VAR_99_1_:%.+]] = krnl.load [[LOOP_3_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_35_MEM_6_:%.+]] = addf [[LOAD_VAR_54_MEM_1_]], [[VAR_99_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_35_MEM_7_:%.+]] = krnl.load [[VAR_36_]]#1{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_102_1_:%.+]] = krnl.load [[VAR_36_]]#5{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_103_1_:%.+]] = addf [[LOAD_VAR_35_MEM_6_]], [[LOAD_VAR_35_MEM_7_]] : f32
// CHECK-DAG:           [[LOAD_VAR_40_MEM_2_:%.+]] = addf [[VAR_103_1_]], [[VAR_102_1_]] : f32
// CHECK-DAG:           [[VAR_105_1_:%.+]] = krnl.load [[VAR_41_]]#1{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_106_1_:%.+]] = mulf [[VAR_105_1_]], [[LOAD_VAR_53_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_107_1_:%.+]] = addf [[LOAD_VAR_40_MEM_2_]], [[VAR_106_1_]] : f32
// CHECK-DAG:           [[VAR_108_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_107_1_]], [[VAR_108_1_]][] : memref<f32>
// CHECK:               [[LOAD_VAR_108_MEM_1_:%.+]] = "onnx.Sigmoid"([[VAR_108_1_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[VAR_110_1_:%.+]] = krnl.load [[LOAD_VAR_108_MEM_1_]][] : memref<f32>
// CHECK-DAG:           [[VAR_111_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[LOAD_VAR_53_MEM_1_]], [[VAR_111_1_]][] : memref<f32>
// CHECK:               [[LOAD_VAR_111_MEM_1_:%.+]] = "onnx.Tanh"([[VAR_111_1_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[VAR_113_1_:%.+]] = krnl.load [[LOAD_VAR_111_MEM_1_]][] : memref<f32>
// CHECK:               [[VAR_114_:%.+]] = mulf [[VAR_110_1_]], [[VAR_113_1_]] : f32
// CHECK:               krnl.store [[LOAD_VAR_53_MEM_1_]], [[VAR_0_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK:               krnl.store [[VAR_114_]], [[VAR_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc [[LOAD_PARAM_4_MEM_2_]] : memref<2x3xf32>
// CHECK:           }
// CHECK-DAG:       [[CST_2_4_:%.+]] = constant 2 : index
// CHECK-DAG:       [[CST_4_2_:%.+]] = constant 4 : index
// CHECK-DAG:       [[CST_0_10_:%.+]] = constant 0 : index
// CHECK-DAG:       [[CST_0_11_:%.+]] = constant 0 : index
// CHECK-DAG:       [[CST_0_12_:%.+]] = constant 0 : index
// CHECK-DAG:       [[CST_1_4_:%.+]] = constant 1 : index
// CHECK-DAG:       [[LOOP_7_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_7_]]#0, [[LOOP_7_]]#1) with ([[LOOP_7_]]#0 -> [[I_12_:%.+]] = [[CST_0_10_]] to [[CST_2_4_]], [[LOOP_7_]]#1 -> [[I_13_:%.+]] = [[CST_0_11_]] to [[CST_4_2_]]) {
// CHECK:             [[LOAD_PARAM_4_MEM_2_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_7_]]#0, [[LOOP_7_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOOP_2_1_:%.+]] = krnl.load [[VAR_3_]]{{.}}[[LOAD_PARAM_4_MEM_2_1_]]#0, [[LOAD_PARAM_4_MEM_2_1_]]#1] : memref<2x4xf32>
// CHECK:             krnl.store [[LOOP_2_1_]], [[VAR_4_]]{{.}}[[CST_0_12_]], [[VAR_4_]]5#0, [[VAR_4_]]5#1] : memref<2x2x4xf32>
// CHECK:             [[LOOP_5_:%.+]] = krnl.load [[VAR_1_]]{{.}}[[LOAD_PARAM_4_MEM_2_1_]]#0, [[LOAD_PARAM_4_MEM_2_1_]]#1] : memref<2x4xf32>
// CHECK:             krnl.store [[LOOP_5_]], [[VAR_4_]]{{.}}[[CST_1_4_]], [[VAR_4_]]5#0, [[VAR_4_]]5#1] : memref<2x2x4xf32>
// CHECK:           }
// CHECK:           memref.dealloc [[VAR_3_]] : memref<2x4xf32>
// CHECK:           memref.dealloc [[VAR_2_]] : memref<2x4xf32>
// CHECK:           memref.dealloc [[VAR_1_]] : memref<2x4xf32>
// CHECK:           memref.dealloc [[VAR_0_]] : memref<2x4xf32>
// CHECK:           return [[VAR_4_]] : memref<2x2x4xf32>
// CHECK:         }
}
// -----

func private @test_lstm_unknown_dims(%arg0: tensor<?x?x?xf32>, %arg1: tensor<1x16x?xf32>, %arg2: tensor<1x16x4xf32>, %arg3: tensor<1x32xf32>, %arg4: tensor<1x?x4xf32>, %arg5: tensor<1x?x4xf32>, %arg6: tensor<1x12xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4, %arg5, %arg6) {hidden_size = 4 : si64} : (tensor<?x?x?xf32>, tensor<1x16x?xf32>, tensor<1x16x4xf32>, tensor<1x32xf32>, none, tensor<1x?x4xf32>, tensor<1x?x4xf32>, tensor<1x12xf32>) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>
// CHECK-LABEL:  func private @test_lstm_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x?xf32>, [[PARAM_1_:%.+]]: memref<1x16x?xf32>, [[PARAM_2_:%.+]]: memref<1x16x4xf32>, [[PARAM_3_:%.+]]: memref<1x32xf32>, [[PARAM_4_:%.+]]: memref<1x?x4xf32>, [[PARAM_5_:%.+]]: memref<1x?x4xf32>, [[PARAM_6_:%.+]]: memref<1x12xf32>) -> memref<1x?x4xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = constant unit
// CHECK-DAG:       [[CST_1_:%.+]] = constant 1 : index
// CHECK:           [[VAR_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.alloc([[VAR_0_]]) : memref<1x?x4xf32>
// CHECK-DAG:       [[CST_1_1_:%.+]] = constant 1 : index
// CHECK:           [[VAR_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = memref.alloc([[VAR_2_]]) : memref<?x4xf32>
// CHECK-DAG:       [[CST_1_2_:%.+]] = constant 1 : index
// CHECK:           [[VAR_4_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_2_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = memref.alloc([[VAR_4_]]) : memref<?x4xf32>
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK-DAG:       [[CST_1_3_:%.+]] = constant 1 : index
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[CST_0_1_:%.+]] = constant 0 : index
// CHECK:           [[VAR_7_:%.+]] = memref.dim [[VAR_3_]], [[CST_0_1_]] : memref<?x4xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[VAR_7_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4) {
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x?x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[VAR_3_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<?x4xf32>
// CHECK:             [[LOAD_PARAM_5_MEM_:%.+]] = krnl.load [[PARAM_5_]]{{.}}[[CST_0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x?x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_5_MEM_]], [[VAR_5_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<?x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Squeeze"([[PARAM_1_]]) {axes = [0]} : (memref<1x16x?xf32>) -> memref<16x?xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Squeeze"([[PARAM_2_]]) {axes = [0]} : (memref<1x16x4xf32>) -> memref<16x4xf32>
// CHECK:           [[VAR_10_:%.+]]:4 = "onnx.Split"([[VAR_8_]]) {axis = 0 : si64} : (memref<16x?xf32>) -> (memref<4x?xf32>, memref<4x?xf32>, memref<4x?xf32>, memref<4x?xf32>)
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Transpose"([[VAR_10_]]#0) {perm = [1, 0]} : (memref<4x?xf32>) -> memref<?x4xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Transpose"([[VAR_10_]]#1) {perm = [1, 0]} : (memref<4x?xf32>) -> memref<?x4xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Transpose"([[VAR_10_]]#2) {perm = [1, 0]} : (memref<4x?xf32>) -> memref<?x4xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.Transpose"([[VAR_10_]]#3) {perm = [1, 0]} : (memref<4x?xf32>) -> memref<?x4xf32>
// CHECK-DAG:       [[VAR_15_:%.+]]:4 = "onnx.Split"([[VAR_9_]]) {axis = 0 : si64} : (memref<16x4xf32>) -> (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Transpose"([[VAR_15_]]#0) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Transpose"([[VAR_15_]]#1) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Transpose"([[VAR_15_]]#2) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.Transpose"([[VAR_15_]]#3) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.Squeeze"([[PARAM_3_]]) {axes = [0]} : (memref<1x32xf32>) -> memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_21_:%.+]]:8 = "onnx.Split"([[VAR_20_]]) {axis = 0 : si64} : (memref<32xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[VAR_22_:%.+]] = "onnx.Squeeze"([[PARAM_6_]]) {axes = [0]} : (memref<1x12xf32>) -> memref<12xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_23_:%.+]]:3 = "onnx.Split"([[VAR_22_]]) {axis = 0 : si64} : (memref<12xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK-DAG:       [[CST_0_2_:%.+]] = constant 0 : index
// CHECK:           [[VAR_25_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_2_]] : memref<?x?x?xf32>
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to [[VAR_25_]]) {
// CHECK-DAG:         [[CST_0_3_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_1_4_:%.+]] = constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_4_]] : memref<?x?x?xf32>
// CHECK-DAG:         [[CST_2_:%.+]] = constant 2 : index
// CHECK:             [[LOAD_PARAM_5_MEM_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<?x?x?xf32>
// CHECK-DAG:         [[VAR_31_:%.+]] = memref.alloc([[LOAD_PARAM_4_MEM_1_]], [[LOAD_PARAM_5_MEM_1_]]) : memref<?x?xf32>
// CHECK-DAG:         [[CST_0_4_:%.+]] = constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_32_:%.+]] = memref.dim [[VAR_31_]], [[CST_0_4_]] : memref<?x?xf32>
// CHECK-DAG:         [[CST_1_5_:%.+]] = constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_33_:%.+]] = memref.dim [[VAR_31_]], [[CST_1_5_]] : memref<?x?xf32>
// CHECK-DAG:         [[CST_0_5_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_6_:%.+]] = constant 0 : index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_5_]] to [[VAR_32_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_6_]] to [[VAR_33_]]) {
// CHECK:               [[VAR_44_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_44_]]#0, [[VAR_44_]]#1] : memref<?x?x?xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[VAR_31_]]{{.}}[[VAR_44_]]#0, [[VAR_44_]]#1] : memref<?x?xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_35_:%.+]] = "onnx.MatMul"([[VAR_31_]], [[VAR_11_]]) : (memref<?x?xf32>, memref<?x4xf32>) -> memref<?x4xf32>
// CHECK-DAG:         [[VAR_36_:%.+]] = "onnx.MatMul"([[VAR_3_]], [[VAR_16_]]) : (memref<?x4xf32>, memref<4x4xf32>) -> memref<?x4xf32>
// CHECK-DAG:         [[VAR_37_:%.+]] = "onnx.MatMul"([[VAR_31_]], [[VAR_13_]]) : (memref<?x?xf32>, memref<?x4xf32>) -> memref<?x4xf32>
// CHECK-DAG:         [[VAR_38_:%.+]] = "onnx.MatMul"([[VAR_3_]], [[VAR_18_]]) : (memref<?x4xf32>, memref<4x4xf32>) -> memref<?x4xf32>
// CHECK-DAG:         [[VAR_39_:%.+]] = "onnx.MatMul"([[VAR_31_]], [[VAR_14_]]) : (memref<?x?xf32>, memref<?x4xf32>) -> memref<?x4xf32>
// CHECK-DAG:         [[VAR_40_:%.+]] = "onnx.MatMul"([[VAR_3_]], [[VAR_19_]]) : (memref<?x4xf32>, memref<4x4xf32>) -> memref<?x4xf32>
// CHECK-DAG:         [[VAR_41_:%.+]] = "onnx.MatMul"([[VAR_31_]], [[VAR_12_]]) : (memref<?x?xf32>, memref<?x4xf32>) -> memref<?x4xf32>
// CHECK-DAG:         [[VAR_42_:%.+]] = "onnx.MatMul"([[VAR_3_]], [[VAR_17_]]) : (memref<?x4xf32>, memref<4x4xf32>) -> memref<?x4xf32>
// CHECK-DAG:         [[CST_0_7_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_8_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_1_6_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_7_]] to [[VAR_2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_7_]] to [[CST_4_]]) {
// CHECK:               [[VAR_44_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_5_]]{{.}}[[VAR_44_1_]]#0, [[VAR_44_1_]]#1] : memref<?x4xf32>
// CHECK-DAG:           [[LOAD_VAR_35_MEM_:%.+]] = krnl.load [[VAR_35_]]{{.}}[[VAR_44_1_]]#0, [[VAR_44_1_]]#1] : memref<?x4xf32>
// CHECK-DAG:           [[LOAD_VAR_36_MEM_:%.+]] = krnl.load [[VAR_36_]]{{.}}[[VAR_44_1_]]#0, [[VAR_44_1_]]#1] : memref<?x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_48_:%.+]] = addf [[LOAD_VAR_35_MEM_]], [[LOAD_VAR_36_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_21_MEM_:%.+]] = krnl.load [[VAR_21_]]#0{{.}}[[VAR_44_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_21_MEM_1_:%.+]] = krnl.load [[VAR_21_]]#4{{.}}[[VAR_44_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_51_:%.+]] = addf [[VAR_48_]], [[LOAD_VAR_21_MEM_]] : f32
// CHECK-DAG:           [[VAR_52_:%.+]] = addf [[VAR_51_]], [[LOAD_VAR_21_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_23_MEM_:%.+]] = krnl.load [[VAR_23_]]#0{{.}}[[VAR_44_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_54_:%.+]] = mulf [[LOAD_VAR_23_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_55_:%.+]] = addf [[VAR_52_]], [[VAR_54_]] : f32
// CHECK-DAG:           [[VAR_56_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_55_]], [[VAR_56_]][] : memref<f32>
// CHECK:               [[VAR_57_:%.+]] = "onnx.Sigmoid"([[VAR_56_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_57_MEM_:%.+]] = krnl.load [[VAR_57_]][] : memref<f32>
// CHECK-DAG:           [[LOAD_VAR_37_MEM_:%.+]] = krnl.load [[VAR_37_]]{{.}}[[VAR_44_1_]]#0, [[VAR_44_1_]]#1] : memref<?x4xf32>
// CHECK-DAG:           [[LOAD_VAR_38_MEM_:%.+]] = krnl.load [[VAR_38_]]{{.}}[[VAR_44_1_]]#0, [[VAR_44_1_]]#1] : memref<?x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_61_:%.+]] = addf [[LOAD_VAR_37_MEM_]], [[LOAD_VAR_38_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_21_MEM_2_:%.+]] = krnl.load [[VAR_21_]]#2{{.}}[[VAR_44_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_21_MEM_3_:%.+]] = krnl.load [[VAR_21_]]#6{{.}}[[VAR_44_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_64_:%.+]] = addf [[VAR_61_]], [[LOAD_VAR_21_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_65_:%.+]] = addf [[VAR_64_]], [[LOAD_VAR_21_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_23_MEM_1_:%.+]] = krnl.load [[VAR_23_]]#2{{.}}[[VAR_44_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_67_:%.+]] = mulf [[LOAD_VAR_23_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_68_:%.+]] = addf [[VAR_65_]], [[VAR_67_]] : f32
// CHECK-DAG:           [[VAR_69_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_68_]], [[VAR_69_]][] : memref<f32>
// CHECK:               [[VAR_70_:%.+]] = "onnx.Sigmoid"([[VAR_69_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_70_MEM_:%.+]] = krnl.load [[VAR_70_]][] : memref<f32>
// CHECK-DAG:           [[LOAD_VAR_39_MEM_:%.+]] = krnl.load [[VAR_39_]]{{.}}[[VAR_44_1_]]#0, [[VAR_44_1_]]#1] : memref<?x4xf32>
// CHECK-DAG:           [[LOAD_VAR_40_MEM_:%.+]] = krnl.load [[VAR_40_]]{{.}}[[VAR_44_1_]]#0, [[VAR_44_1_]]#1] : memref<?x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_74_:%.+]] = addf [[LOAD_VAR_39_MEM_]], [[LOAD_VAR_40_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_21_MEM_4_:%.+]] = krnl.load [[VAR_21_]]#3{{.}}[[VAR_44_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_21_MEM_5_:%.+]] = krnl.load [[VAR_21_]]#7{{.}}[[VAR_44_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_77_:%.+]] = addf [[VAR_74_]], [[LOAD_VAR_21_MEM_4_]] : f32
// CHECK-DAG:           [[VAR_78_:%.+]] = addf [[VAR_77_]], [[LOAD_VAR_21_MEM_5_]] : f32
// CHECK-DAG:           [[VAR_79_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_78_]], [[VAR_79_]][] : memref<f32>
// CHECK:               [[VAR_80_:%.+]] = "onnx.Tanh"([[VAR_79_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_80_MEM_:%.+]] = krnl.load [[VAR_80_]][] : memref<f32>
// CHECK-DAG:           [[VAR_82_:%.+]] = mulf [[LOAD_VAR_70_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_83_:%.+]] = mulf [[LOAD_VAR_57_MEM_]], [[LOAD_VAR_80_MEM_]] : f32
// CHECK-DAG:           [[VAR_84_:%.+]] = addf [[VAR_82_]], [[VAR_83_]] : f32
// CHECK-DAG:           [[LOAD_VAR_41_MEM_:%.+]] = krnl.load [[VAR_41_]]{{.}}[[VAR_44_1_]]#0, [[VAR_44_1_]]#1] : memref<?x4xf32>
// CHECK-DAG:           [[LOAD_VAR_42_MEM_:%.+]] = krnl.load [[VAR_42_]]{{.}}[[VAR_44_1_]]#0, [[VAR_44_1_]]#1] : memref<?x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_87_:%.+]] = addf [[LOAD_VAR_41_MEM_]], [[LOAD_VAR_42_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_21_MEM_6_:%.+]] = krnl.load [[VAR_21_]]#1{{.}}[[VAR_44_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_21_MEM_7_:%.+]] = krnl.load [[VAR_21_]]#5{{.}}[[VAR_44_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_90_:%.+]] = addf [[VAR_87_]], [[LOAD_VAR_21_MEM_6_]] : f32
// CHECK-DAG:           [[VAR_91_:%.+]] = addf [[VAR_90_]], [[LOAD_VAR_21_MEM_7_]] : f32
// CHECK-DAG:           [[LOAD_VAR_23_MEM_2_:%.+]] = krnl.load [[VAR_23_]]#1{{.}}[[VAR_44_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_93_:%.+]] = mulf [[LOAD_VAR_23_MEM_2_]], [[VAR_84_]] : f32
// CHECK-DAG:           [[VAR_94_:%.+]] = addf [[VAR_91_]], [[VAR_93_]] : f32
// CHECK-DAG:           [[VAR_95_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_94_]], [[VAR_95_]][] : memref<f32>
// CHECK:               [[VAR_96_:%.+]] = "onnx.Sigmoid"([[VAR_95_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_96_MEM_:%.+]] = krnl.load [[VAR_96_]][] : memref<f32>
// CHECK-DAG:           [[VAR_98_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_84_]], [[VAR_98_]][] : memref<f32>
// CHECK:               [[VAR_99_:%.+]] = "onnx.Tanh"([[VAR_98_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_VAR_99_MEM_:%.+]] = krnl.load [[VAR_99_]][] : memref<f32>
// CHECK:               [[VAR_101_:%.+]] = mulf [[LOAD_VAR_96_MEM_]], [[LOAD_VAR_99_MEM_]] : f32
// CHECK:               krnl.store [[VAR_84_]], [[VAR_5_]]{{.}}[[VAR_44_1_]]#0, [[VAR_44_1_]]#1] : memref<?x4xf32>
// CHECK:               krnl.store [[VAR_101_]], [[VAR_3_]]{{.}}[[VAR_44_1_]]#0, [[VAR_44_1_]]#1] : memref<?x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc [[VAR_31_]] : memref<?x?xf32>
// CHECK:           }
// CHECK-DAG:       [[CST_16_:%.+]] = constant 16 : i64
// CHECK-DAG:       [[CST_0_9_:%.+]] = constant 0 : index
// CHECK:           [[VAR_26_:%.+]] = memref.dim [[VAR_3_]], [[CST_0_9_]] : memref<?x4xf32>
// CHECK:           [[VAR_27_:%.+]] = index_cast [[VAR_26_]] : index to i64
// CHECK:           [[VAR_28_:%.+]] = muli [[CST_16_]], [[VAR_27_]] : i64
// CHECK:           "krnl.memcpy"([[VAR_1_]], [[VAR_3_]], [[VAR_28_]]) : (memref<1x?x4xf32>, memref<?x4xf32>, i64) -> ()
// CHECK:           memref.dealloc [[VAR_3_]] : memref<?x4xf32>
// CHECK:           memref.dealloc [[VAR_5_]] : memref<?x4xf32>
// CHECK:           return [[VAR_1_]] : memref<1x?x4xf32>
// CHECK:         }
}