// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl='check-rnn-ops-lowering' %s -split-input-file

// COM: | FileCheck %s

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
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Squeeze"([[PARAM_3_]]) {axes = [0]} : (memref<1x32xf32>) -> memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_7_:%.+]]:8 = "onnx.Split"([[VAR_6_]]) {axis = 0 : si64} : (memref<32xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Squeeze"([[PARAM_6_]]) {axes = [0]} : (memref<1x12xf32>) -> memref<12xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_9_:%.+]]:3 = "onnx.Split"([[VAR_8_]]) {axis = 0 : si64} : (memref<12xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>)
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
// CHECK:               [[VAR_18_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_18_]]#0, [[VAR_18_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_4_MEM_1_]]{{.}}[[VAR_18_]]#0, [[VAR_18_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             [[VAR_13_:%.+]] = "onnx.Transpose"([[LOAD_PARAM_4_MEM_1_]]) {perm = [1, 0]} : (memref<2x3xf32>) -> memref<3x2xf32>
// CHECK-DAG:         [[VAR_14_:%.+]] = "onnx.MatMul"([[VAR_4_]], [[VAR_13_]]) : (memref<16x3xf32>, memref<3x2xf32>) -> memref<16x2xf32>
// CHECK-DAG:         [[VAR_15_:%.+]] = "onnx.Transpose"([[VAR_1_]]) {perm = [1, 0]} : (memref<2x4xf32>) -> memref<4x2xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_16_:%.+]] = "onnx.MatMul"([[VAR_5_]], [[VAR_15_]]) : (memref<16x4xf32>, memref<4x2xf32>) -> memref<16x2xf32>
// CHECK-DAG:         [[CST_0_4_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_5_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_1_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_2_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_4_]] to [[CST_2_1_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_4_]] to [[CST_4_]]) {
// CHECK:               [[VAR_18_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_14_MEM_:%.+]] = krnl.load [[VAR_14_]]{{.}}[[VAR_18_1_]]#1, [[VAR_18_1_]]#0] : memref<16x2xf32>
// CHECK-DAG:           [[LOAD_VAR_16_MEM_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_18_1_]]#1, [[VAR_18_1_]]#0] : memref<16x2xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_22_:%.+]] = addf [[LOAD_VAR_14_MEM_]], [[LOAD_VAR_16_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_7_MEM_:%.+]] = krnl.load [[VAR_7_]]#0{{.}}[[VAR_18_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_7_MEM_1_:%.+]] = krnl.load [[VAR_7_]]#4{{.}}[[VAR_18_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_25_:%.+]] = addf [[VAR_22_]], [[LOAD_VAR_7_MEM_]] : f32
// CHECK-DAG:           [[VAR_26_:%.+]] = addf [[VAR_25_]], [[LOAD_VAR_7_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_9_MEM_:%.+]] = krnl.load [[VAR_9_]]#0{{.}}[[VAR_18_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_28_:%.+]] = mulf [[LOAD_VAR_9_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_29_:%.+]] = addf [[VAR_26_]], [[VAR_28_]] : f32
// CHECK-DAG:           [[VAR_30_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_29_]], [[VAR_30_]][] : memref<f32>
// CHECK:               [[VAR_31_:%.+]] = "onnx.Sigmoid"([[VAR_30_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_31_MEM_:%.+]] = krnl.load [[VAR_31_]][] : memref<f32>
// CHECK-DAG:           [[CST_8_:%.+]] = constant 8 : index
// CHECK-DAG:           [[VAR_33_:%.+]] = affine.apply #map0(){{.}}[[VAR_18_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_14_MEM_1_:%.+]] = krnl.load [[VAR_14_]]{{.}}[[VAR_33_]], [[VAR_18_1_]]#0] : memref<16x2xf32>
// CHECK-DAG:           [[CST_8_1_:%.+]] = constant 8 : index
// CHECK-DAG:           [[VAR_35_:%.+]] = affine.apply #map0(){{.}}[[VAR_18_1_]]#1]
// CHECK:               [[LOAD_VAR_16_MEM_1_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_35_]], [[VAR_18_1_]]#0] : memref<16x2xf32>
// CHECK-DAG:           [[VAR_37_:%.+]] = addf [[LOAD_VAR_14_MEM_1_]], [[LOAD_VAR_16_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_7_MEM_2_:%.+]] = krnl.load [[VAR_7_]]#2{{.}}[[VAR_18_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_7_MEM_3_:%.+]] = krnl.load [[VAR_7_]]#6{{.}}[[VAR_18_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_40_:%.+]] = addf [[VAR_37_]], [[LOAD_VAR_7_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_41_:%.+]] = addf [[VAR_40_]], [[LOAD_VAR_7_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_9_MEM_1_:%.+]] = krnl.load [[VAR_9_]]#2{{.}}[[VAR_18_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_43_:%.+]] = mulf [[LOAD_VAR_9_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_44_:%.+]] = addf [[VAR_41_]], [[VAR_43_]] : f32
// CHECK-DAG:           [[VAR_45_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_44_]], [[VAR_45_]][] : memref<f32>
// CHECK:               [[VAR_46_:%.+]] = "onnx.Sigmoid"([[VAR_45_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_46_MEM_:%.+]] = krnl.load [[VAR_46_]][] : memref<f32>
// CHECK-DAG:           [[CST_12_:%.+]] = constant 12 : index
// CHECK-DAG:           [[VAR_48_:%.+]] = affine.apply #map1(){{.}}[[VAR_18_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_14_MEM_2_:%.+]] = krnl.load [[VAR_14_]]{{.}}[[VAR_48_]], [[VAR_18_1_]]#0] : memref<16x2xf32>
// CHECK-DAG:           [[CST_12_1_:%.+]] = constant 12 : index
// CHECK-DAG:           [[VAR_50_:%.+]] = affine.apply #map1(){{.}}[[VAR_18_1_]]#1]
// CHECK:               [[LOAD_VAR_16_MEM_2_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_50_]], [[VAR_18_1_]]#0] : memref<16x2xf32>
// CHECK-DAG:           [[VAR_52_:%.+]] = addf [[LOAD_VAR_14_MEM_2_]], [[LOAD_VAR_16_MEM_2_]] : f32
// CHECK-DAG:           [[LOAD_VAR_7_MEM_4_:%.+]] = krnl.load [[VAR_7_]]#3{{.}}[[VAR_18_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_7_MEM_5_:%.+]] = krnl.load [[VAR_7_]]#7{{.}}[[VAR_18_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_55_:%.+]] = addf [[VAR_52_]], [[LOAD_VAR_7_MEM_4_]] : f32
// CHECK-DAG:           [[VAR_56_:%.+]] = addf [[VAR_55_]], [[LOAD_VAR_7_MEM_5_]] : f32
// CHECK-DAG:           [[VAR_57_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_56_]], [[VAR_57_]][] : memref<f32>
// CHECK:               [[VAR_58_:%.+]] = "onnx.Tanh"([[VAR_57_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_58_MEM_:%.+]] = krnl.load [[VAR_58_]][] : memref<f32>
// CHECK-DAG:           [[VAR_60_:%.+]] = mulf [[LOAD_VAR_46_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_61_:%.+]] = mulf [[LOAD_VAR_31_MEM_]], [[LOAD_VAR_58_MEM_]] : f32
// CHECK-DAG:           [[VAR_62_:%.+]] = addf [[VAR_60_]], [[VAR_61_]] : f32
// CHECK-DAG:           [[CST_4_1_:%.+]] = constant 4 : index
// CHECK-DAG:           [[VAR_63_:%.+]] = affine.apply #map2(){{.}}[[VAR_18_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_14_MEM_3_:%.+]] = krnl.load [[VAR_14_]]{{.}}[[VAR_63_]], [[VAR_18_1_]]#0] : memref<16x2xf32>
// CHECK-DAG:           [[CST_4_2_:%.+]] = constant 4 : index
// CHECK-DAG:           [[VAR_65_:%.+]] = affine.apply #map2(){{.}}[[VAR_18_1_]]#1]
// CHECK:               [[LOAD_VAR_16_MEM_3_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_65_]], [[VAR_18_1_]]#0] : memref<16x2xf32>
// CHECK-DAG:           [[VAR_67_:%.+]] = addf [[LOAD_VAR_14_MEM_3_]], [[LOAD_VAR_16_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_7_MEM_6_:%.+]] = krnl.load [[VAR_7_]]#1{{.}}[[VAR_18_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_7_MEM_7_:%.+]] = krnl.load [[VAR_7_]]#5{{.}}[[VAR_18_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_70_:%.+]] = addf [[VAR_67_]], [[LOAD_VAR_7_MEM_6_]] : f32
// CHECK-DAG:           [[VAR_71_:%.+]] = addf [[VAR_70_]], [[LOAD_VAR_7_MEM_7_]] : f32
// CHECK-DAG:           [[LOAD_VAR_9_MEM_2_:%.+]] = krnl.load [[VAR_9_]]#1{{.}}[[VAR_18_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_73_:%.+]] = mulf [[LOAD_VAR_9_MEM_2_]], [[VAR_62_]] : f32
// CHECK-DAG:           [[VAR_74_:%.+]] = addf [[VAR_71_]], [[VAR_73_]] : f32
// CHECK-DAG:           [[VAR_75_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_74_]], [[VAR_75_]][] : memref<f32>
// CHECK:               [[VAR_76_:%.+]] = "onnx.Sigmoid"([[VAR_75_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_76_MEM_:%.+]] = krnl.load [[VAR_76_]][] : memref<f32>
// CHECK-DAG:           [[VAR_78_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_62_]], [[VAR_78_]][] : memref<f32>
// CHECK:               [[VAR_79_:%.+]] = "onnx.Tanh"([[VAR_78_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_VAR_79_MEM_:%.+]] = krnl.load [[VAR_79_]][] : memref<f32>
// CHECK:               [[VAR_81_:%.+]] = mulf [[LOAD_VAR_76_MEM_]], [[LOAD_VAR_79_MEM_]] : f32
// CHECK:               krnl.store [[VAR_62_]], [[VAR_0_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1] : memref<2x4xf32>
// CHECK:               krnl.store [[VAR_81_]], [[VAR_1_]]{{.}}[[VAR_1_]]8#0, [[VAR_1_]]8#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc [[LOAD_PARAM_4_MEM_1_]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           [[CST_32_:%.+]] = constant 32 : i64
// CHECK:           "krnl.memcpy"([[VAR_2_]], [[VAR_1_]], [[CST_32_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
// CHECK:           memref.dealloc [[VAR_1_]] : memref<2x4xf32>
// CHECK:           memref.dealloc [[VAR_0_]] : memref<2x4xf32>
// CHECK:           return [[VAR_2_]] : memref<1x2x4xf32>
// CHECK:         }
}

// -----

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
// CHECK-DAG:       [[VAR_10_:%.+]] = "krnl.global"() {name = "constant_6", shape = [32], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01]> : tensor<32xf32>} : () -> memref<32xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "krnl.global"() {name = "constant_7", shape = [4], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "krnl.global"() {name = "constant_8", shape = [4], value = dense<[5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "krnl.global"() {name = "constant_9", shape = [4], value = dense<[9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = "krnl.global"() {name = "constant_10", shape = [4], value = dense<[1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = "krnl.global"() {name = "constant_11", shape = [4], value = dense<[1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "krnl.global"() {name = "constant_12", shape = [4], value = dense<[2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = "krnl.global"() {name = "constant_13", shape = [4], value = dense<[2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "krnl.global"() {name = "constant_14", shape = [4], value = dense<[2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "krnl.global"() {name = "constant_15", shape = [12], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01]> : tensor<12xf32>} : () -> memref<12xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = "krnl.global"() {name = "constant_16", shape = [4], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "krnl.global"() {name = "constant_17", shape = [4], value = dense<[5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = "krnl.global"() {name = "constant_18", shape = [4], value = dense<[9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7) {
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_1_:%.+]] = memref.alloc() : memref<2x3xf32>
// CHECK-DAG:         [[CST_0_1_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_2_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_3_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_1_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_3_:%.+]] = constant 3 : index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_2_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_2_]] to [[CST_3_]]) {
// CHECK:               [[VAR_31_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_31_]]#0, [[VAR_31_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_1_]]{{.}}[[VAR_31_]]#0, [[VAR_31_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             [[VAR_26_:%.+]] = "onnx.Transpose"([[LOAD_PARAM_1_MEM_1_]]) {perm = [1, 0]} : (memref<2x3xf32>) -> memref<3x2xf32>
// CHECK-DAG:         [[VAR_27_:%.+]] = "onnx.MatMul"([[VAR_8_]], [[VAR_26_]]) : (memref<16x3xf32>, memref<3x2xf32>) -> memref<16x2xf32>
// CHECK-DAG:         [[VAR_28_:%.+]] = "onnx.Transpose"([[VAR_1_]]) {perm = [1, 0]} : (memref<2x4xf32>) -> memref<4x2xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_29_:%.+]] = "onnx.MatMul"([[VAR_9_]], [[VAR_28_]]) : (memref<16x4xf32>, memref<4x2xf32>) -> memref<16x2xf32>
// CHECK-DAG:         [[CST_0_4_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_5_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_1_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_2_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_4_]] to [[CST_2_1_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_4_]] to [[CST_4_]]) {
// CHECK:               [[VAR_31_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[VAR_31_1_]]#0, [[VAR_31_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_27_MEM_:%.+]] = krnl.load [[VAR_27_]]{{.}}[[VAR_31_1_]]#1, [[VAR_31_1_]]#0] : memref<16x2xf32>
// CHECK-DAG:           [[LOAD_VAR_29_MEM_:%.+]] = krnl.load [[VAR_29_]]{{.}}[[VAR_31_1_]]#1, [[VAR_31_1_]]#0] : memref<16x2xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_35_:%.+]] = addf [[LOAD_VAR_27_MEM_]], [[LOAD_VAR_29_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_:%.+]] = krnl.load [[VAR_11_]]{{.}}[[VAR_31_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_15_MEM_:%.+]] = krnl.load [[VAR_15_]]{{.}}[[VAR_31_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_38_:%.+]] = addf [[VAR_35_]], [[LOAD_VAR_11_MEM_]] : f32
// CHECK-DAG:           [[VAR_39_:%.+]] = addf [[VAR_38_]], [[LOAD_VAR_15_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_20_MEM_:%.+]] = krnl.load [[VAR_20_]]{{.}}[[VAR_31_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_41_:%.+]] = mulf [[LOAD_VAR_20_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_42_:%.+]] = addf [[VAR_39_]], [[VAR_41_]] : f32
// CHECK-DAG:           [[VAR_43_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_42_]], [[VAR_43_]][] : memref<f32>
// CHECK:               [[VAR_44_:%.+]] = "onnx.Sigmoid"([[VAR_43_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_44_MEM_:%.+]] = krnl.load [[VAR_44_]][] : memref<f32>
// CHECK-DAG:           [[CST_8_:%.+]] = constant 8 : index
// CHECK-DAG:           [[VAR_46_:%.+]] = affine.apply #map0(){{.}}[[VAR_31_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_27_MEM_1_:%.+]] = krnl.load [[VAR_27_]]{{.}}[[VAR_46_]], [[VAR_31_1_]]#0] : memref<16x2xf32>
// CHECK-DAG:           [[CST_8_1_:%.+]] = constant 8 : index
// CHECK-DAG:           [[VAR_48_:%.+]] = affine.apply #map0(){{.}}[[VAR_31_1_]]#1]
// CHECK:               [[LOAD_VAR_29_MEM_1_:%.+]] = krnl.load [[VAR_29_]]{{.}}[[VAR_48_]], [[VAR_31_1_]]#0] : memref<16x2xf32>
// CHECK-DAG:           [[VAR_50_:%.+]] = addf [[LOAD_VAR_27_MEM_1_]], [[LOAD_VAR_29_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_13_MEM_:%.+]] = krnl.load [[VAR_13_]]{{.}}[[VAR_31_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_17_MEM_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_31_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_53_:%.+]] = addf [[VAR_50_]], [[LOAD_VAR_13_MEM_]] : f32
// CHECK-DAG:           [[VAR_54_:%.+]] = addf [[VAR_53_]], [[LOAD_VAR_17_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_22_MEM_:%.+]] = krnl.load [[VAR_22_]]{{.}}[[VAR_31_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_56_:%.+]] = mulf [[LOAD_VAR_22_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_57_:%.+]] = addf [[VAR_54_]], [[VAR_56_]] : f32
// CHECK-DAG:           [[VAR_58_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_57_]], [[VAR_58_]][] : memref<f32>
// CHECK:               [[VAR_59_:%.+]] = "onnx.Sigmoid"([[VAR_58_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_59_MEM_:%.+]] = krnl.load [[VAR_59_]][] : memref<f32>
// CHECK-DAG:           [[CST_12_:%.+]] = constant 12 : index
// CHECK-DAG:           [[VAR_61_:%.+]] = affine.apply #map1(){{.}}[[VAR_31_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_27_MEM_2_:%.+]] = krnl.load [[VAR_27_]]{{.}}[[VAR_61_]], [[VAR_31_1_]]#0] : memref<16x2xf32>
// CHECK-DAG:           [[CST_12_1_:%.+]] = constant 12 : index
// CHECK-DAG:           [[VAR_63_:%.+]] = affine.apply #map1(){{.}}[[VAR_31_1_]]#1]
// CHECK:               [[LOAD_VAR_29_MEM_2_:%.+]] = krnl.load [[VAR_29_]]{{.}}[[VAR_63_]], [[VAR_31_1_]]#0] : memref<16x2xf32>
// CHECK-DAG:           [[VAR_65_:%.+]] = addf [[LOAD_VAR_27_MEM_2_]], [[LOAD_VAR_29_MEM_2_]] : f32
// CHECK-DAG:           [[LOAD_VAR_14_MEM_:%.+]] = krnl.load [[VAR_14_]]{{.}}[[VAR_31_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_18_MEM_:%.+]] = krnl.load [[VAR_18_]]{{.}}[[VAR_31_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_68_:%.+]] = addf [[VAR_65_]], [[LOAD_VAR_14_MEM_]] : f32
// CHECK-DAG:           [[VAR_69_:%.+]] = addf [[VAR_68_]], [[LOAD_VAR_18_MEM_]] : f32
// CHECK-DAG:           [[VAR_70_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_69_]], [[VAR_70_]][] : memref<f32>
// CHECK:               [[VAR_71_:%.+]] = "onnx.Tanh"([[VAR_70_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_71_MEM_:%.+]] = krnl.load [[VAR_71_]][] : memref<f32>
// CHECK-DAG:           [[VAR_73_:%.+]] = mulf [[LOAD_VAR_59_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_74_:%.+]] = mulf [[LOAD_VAR_44_MEM_]], [[LOAD_VAR_71_MEM_]] : f32
// CHECK-DAG:           [[VAR_75_:%.+]] = addf [[VAR_73_]], [[VAR_74_]] : f32
// CHECK-DAG:           [[CST_4_1_:%.+]] = constant 4 : index
// CHECK-DAG:           [[VAR_76_:%.+]] = affine.apply #map2(){{.}}[[VAR_31_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_27_MEM_3_:%.+]] = krnl.load [[VAR_27_]]{{.}}[[VAR_76_]], [[VAR_31_1_]]#0] : memref<16x2xf32>
// CHECK-DAG:           [[CST_4_2_:%.+]] = constant 4 : index
// CHECK-DAG:           [[VAR_78_:%.+]] = affine.apply #map2(){{.}}[[VAR_31_1_]]#1]
// CHECK:               [[LOAD_VAR_29_MEM_3_:%.+]] = krnl.load [[VAR_29_]]{{.}}[[VAR_78_]], [[VAR_31_1_]]#0] : memref<16x2xf32>
// CHECK-DAG:           [[VAR_80_:%.+]] = addf [[LOAD_VAR_27_MEM_3_]], [[LOAD_VAR_29_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_12_MEM_:%.+]] = krnl.load [[VAR_12_]]{{.}}[[VAR_31_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_16_MEM_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_31_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_83_:%.+]] = addf [[VAR_80_]], [[LOAD_VAR_12_MEM_]] : f32
// CHECK-DAG:           [[VAR_84_:%.+]] = addf [[VAR_83_]], [[LOAD_VAR_16_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_21_MEM_:%.+]] = krnl.load [[VAR_21_]]{{.}}[[VAR_31_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_86_:%.+]] = mulf [[LOAD_VAR_21_MEM_]], [[VAR_75_]] : f32
// CHECK-DAG:           [[VAR_87_:%.+]] = addf [[VAR_84_]], [[VAR_86_]] : f32
// CHECK-DAG:           [[VAR_88_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_87_]], [[VAR_88_]][] : memref<f32>
// CHECK:               [[VAR_89_:%.+]] = "onnx.Sigmoid"([[VAR_88_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_89_MEM_:%.+]] = krnl.load [[VAR_89_]][] : memref<f32>
// CHECK-DAG:           [[VAR_91_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_75_]], [[VAR_91_]][] : memref<f32>
// CHECK:               [[VAR_92_:%.+]] = "onnx.Tanh"([[VAR_91_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_VAR_92_MEM_:%.+]] = krnl.load [[VAR_92_]][] : memref<f32>
// CHECK:               [[VAR_94_:%.+]] = mulf [[LOAD_VAR_89_MEM_]], [[LOAD_VAR_92_MEM_]] : f32
// CHECK:               krnl.store [[VAR_75_]], [[VAR_0_]]{{.}}[[VAR_31_1_]]#0, [[VAR_31_1_]]#1] : memref<2x4xf32>
// CHECK:               krnl.store [[VAR_94_]], [[VAR_1_]]{{.}}[[VAR_31_1_]]#0, [[VAR_31_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc [[LOAD_PARAM_1_MEM_1_]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           [[CST_32_:%.+]] = constant 32 : i64
// CHECK:           "krnl.memcpy"([[VAR_2_]], [[VAR_1_]], [[CST_32_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
// CHECK:           memref.dealloc [[VAR_1_]] : memref<2x4xf32>
// CHECK:           memref.dealloc [[VAR_0_]] : memref<2x4xf32>
// CHECK:           return [[VAR_2_]] : memref<1x2x4xf32>
// CHECK:         }
}

// -----

func private @test_lstm_reverse_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x16x3xf32>, %arg2: tensor<1x16x4xf32>, %arg3: tensor<1x32xf32>, %arg4: tensor<1x2x4xf32>, %arg5: tensor<1x2x4xf32>, %arg6: tensor<1x12xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4, %arg5, %arg6) {hidden_size = 4 : si64, direction = "reverse"} : (tensor<7x2x3xf32>, tensor<1x16x3xf32>, tensor<1x16x4xf32>, tensor<1x32xf32>, none, tensor<1x2x4xf32>, tensor<1x2x4xf32>, tensor<1x12xf32>) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

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
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Squeeze"([[PARAM_3_]]) {axes = [0]} : (memref<1x32xf32>) -> memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_7_:%.+]]:8 = "onnx.Split"([[VAR_6_]]) {axis = 0 : si64} : (memref<32xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Squeeze"([[PARAM_6_]]) {axes = [0]} : (memref<1x12xf32>) -> memref<12xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_9_:%.+]]:3 = "onnx.Split"([[VAR_8_]]) {axis = 0 : si64} : (memref<12xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7) {
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_1_:%.+]] = memref.alloc() : memref<2x3xf32>
// CHECK-DAG:         [[CST_0_1_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_7_:%.+]] = constant 7 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_5_MEM_1_:%.+]] = affine.apply #map0([[I_2_]]){{.}}[[CST_7_]]{{.}}
// CHECK-DAG:         [[CST_0_2_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_3_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_1_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_3_:%.+]] = constant 3 : index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_2_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_2_]] to [[CST_3_]]) {
// CHECK:               [[VAR_19_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[LOAD_PARAM_5_MEM_1_]], [[VAR_19_]]#0, [[VAR_19_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_4_MEM_1_]]{{.}}[[VAR_19_]]#0, [[VAR_19_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             [[VAR_14_:%.+]] = "onnx.Transpose"([[LOAD_PARAM_4_MEM_1_]]) {perm = [1, 0]} : (memref<2x3xf32>) -> memref<3x2xf32>
// CHECK-DAG:         [[VAR_15_:%.+]] = "onnx.MatMul"([[VAR_4_]], [[VAR_14_]]) : (memref<16x3xf32>, memref<3x2xf32>) -> memref<16x2xf32>
// CHECK-DAG:         [[VAR_16_:%.+]] = "onnx.Transpose"([[VAR_1_]]) {perm = [1, 0]} : (memref<2x4xf32>) -> memref<4x2xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_17_:%.+]] = "onnx.MatMul"([[VAR_5_]], [[VAR_16_]]) : (memref<16x4xf32>, memref<4x2xf32>) -> memref<16x2xf32>
// CHECK-DAG:         [[CST_0_4_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_5_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_1_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_2_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_4_]] to [[CST_2_1_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_4_]] to [[CST_4_]]) {
// CHECK:               [[VAR_19_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[VAR_19_1_]]#0, [[VAR_19_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_15_MEM_:%.+]] = krnl.load [[VAR_15_]]{{.}}[[VAR_19_1_]]#1, [[VAR_19_1_]]#0] : memref<16x2xf32>
// CHECK-DAG:           [[LOAD_VAR_17_MEM_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_19_1_]]#1, [[VAR_19_1_]]#0] : memref<16x2xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_23_:%.+]] = addf [[LOAD_VAR_15_MEM_]], [[LOAD_VAR_17_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_7_MEM_:%.+]] = krnl.load [[VAR_7_]]#0{{.}}[[VAR_19_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_7_MEM_1_:%.+]] = krnl.load [[VAR_7_]]#4{{.}}[[VAR_19_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_26_:%.+]] = addf [[VAR_23_]], [[LOAD_VAR_7_MEM_]] : f32
// CHECK-DAG:           [[VAR_27_:%.+]] = addf [[VAR_26_]], [[LOAD_VAR_7_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_9_MEM_:%.+]] = krnl.load [[VAR_9_]]#0{{.}}[[VAR_19_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_29_:%.+]] = mulf [[LOAD_VAR_9_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_30_:%.+]] = addf [[VAR_27_]], [[VAR_29_]] : f32
// CHECK-DAG:           [[VAR_31_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_30_]], [[VAR_31_]][] : memref<f32>
// CHECK:               [[VAR_32_:%.+]] = "onnx.Sigmoid"([[VAR_31_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_32_MEM_:%.+]] = krnl.load [[VAR_32_]][] : memref<f32>
// CHECK-DAG:           [[CST_8_:%.+]] = constant 8 : index
// CHECK-DAG:           [[VAR_34_:%.+]] = affine.apply #map1(){{.}}[[VAR_19_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_15_MEM_1_:%.+]] = krnl.load [[VAR_15_]]{{.}}[[VAR_34_]], [[VAR_19_1_]]#0] : memref<16x2xf32>
// CHECK-DAG:           [[CST_8_1_:%.+]] = constant 8 : index
// CHECK-DAG:           [[VAR_36_:%.+]] = affine.apply #map1(){{.}}[[VAR_19_1_]]#1]
// CHECK:               [[LOAD_VAR_17_MEM_1_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_36_]], [[VAR_19_1_]]#0] : memref<16x2xf32>
// CHECK-DAG:           [[VAR_38_:%.+]] = addf [[LOAD_VAR_15_MEM_1_]], [[LOAD_VAR_17_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_7_MEM_2_:%.+]] = krnl.load [[VAR_7_]]#2{{.}}[[VAR_19_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_7_MEM_3_:%.+]] = krnl.load [[VAR_7_]]#6{{.}}[[VAR_19_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_41_:%.+]] = addf [[VAR_38_]], [[LOAD_VAR_7_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_42_:%.+]] = addf [[VAR_41_]], [[LOAD_VAR_7_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_9_MEM_1_:%.+]] = krnl.load [[VAR_9_]]#2{{.}}[[VAR_19_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_44_:%.+]] = mulf [[LOAD_VAR_9_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_45_:%.+]] = addf [[VAR_42_]], [[VAR_44_]] : f32
// CHECK-DAG:           [[VAR_46_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_45_]], [[VAR_46_]][] : memref<f32>
// CHECK:               [[VAR_47_:%.+]] = "onnx.Sigmoid"([[VAR_46_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_47_MEM_:%.+]] = krnl.load [[VAR_47_]][] : memref<f32>
// CHECK-DAG:           [[CST_12_:%.+]] = constant 12 : index
// CHECK-DAG:           [[VAR_49_:%.+]] = affine.apply #map2(){{.}}[[VAR_19_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_15_MEM_2_:%.+]] = krnl.load [[VAR_15_]]{{.}}[[VAR_49_]], [[VAR_19_1_]]#0] : memref<16x2xf32>
// CHECK-DAG:           [[CST_12_1_:%.+]] = constant 12 : index
// CHECK-DAG:           [[VAR_51_:%.+]] = affine.apply #map2(){{.}}[[VAR_19_1_]]#1]
// CHECK:               [[LOAD_VAR_17_MEM_2_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_51_]], [[VAR_19_1_]]#0] : memref<16x2xf32>
// CHECK-DAG:           [[VAR_53_:%.+]] = addf [[LOAD_VAR_15_MEM_2_]], [[LOAD_VAR_17_MEM_2_]] : f32
// CHECK-DAG:           [[LOAD_VAR_7_MEM_4_:%.+]] = krnl.load [[VAR_7_]]#3{{.}}[[VAR_19_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_7_MEM_5_:%.+]] = krnl.load [[VAR_7_]]#7{{.}}[[VAR_19_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_56_:%.+]] = addf [[VAR_53_]], [[LOAD_VAR_7_MEM_4_]] : f32
// CHECK-DAG:           [[VAR_57_:%.+]] = addf [[VAR_56_]], [[LOAD_VAR_7_MEM_5_]] : f32
// CHECK-DAG:           [[VAR_58_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_57_]], [[VAR_58_]][] : memref<f32>
// CHECK:               [[VAR_59_:%.+]] = "onnx.Tanh"([[VAR_58_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_59_MEM_:%.+]] = krnl.load [[VAR_59_]][] : memref<f32>
// CHECK-DAG:           [[VAR_61_:%.+]] = mulf [[LOAD_VAR_47_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_62_:%.+]] = mulf [[LOAD_VAR_32_MEM_]], [[LOAD_VAR_59_MEM_]] : f32
// CHECK-DAG:           [[VAR_63_:%.+]] = addf [[VAR_61_]], [[VAR_62_]] : f32
// CHECK-DAG:           [[CST_4_1_:%.+]] = constant 4 : index
// CHECK-DAG:           [[VAR_64_:%.+]] = affine.apply #map3(){{.}}[[VAR_19_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_15_MEM_3_:%.+]] = krnl.load [[VAR_15_]]{{.}}[[VAR_64_]], [[VAR_19_1_]]#0] : memref<16x2xf32>
// CHECK-DAG:           [[CST_4_2_:%.+]] = constant 4 : index
// CHECK-DAG:           [[VAR_66_:%.+]] = affine.apply #map3(){{.}}[[VAR_19_1_]]#1]
// CHECK:               [[LOAD_VAR_17_MEM_3_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_66_]], [[VAR_19_1_]]#0] : memref<16x2xf32>
// CHECK-DAG:           [[VAR_68_:%.+]] = addf [[LOAD_VAR_15_MEM_3_]], [[LOAD_VAR_17_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_7_MEM_6_:%.+]] = krnl.load [[VAR_7_]]#1{{.}}[[VAR_19_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_7_MEM_7_:%.+]] = krnl.load [[VAR_7_]]#5{{.}}[[VAR_19_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_71_:%.+]] = addf [[VAR_68_]], [[LOAD_VAR_7_MEM_6_]] : f32
// CHECK-DAG:           [[VAR_72_:%.+]] = addf [[VAR_71_]], [[LOAD_VAR_7_MEM_7_]] : f32
// CHECK-DAG:           [[LOAD_VAR_9_MEM_2_:%.+]] = krnl.load [[VAR_9_]]#1{{.}}[[VAR_19_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_74_:%.+]] = mulf [[LOAD_VAR_9_MEM_2_]], [[VAR_63_]] : f32
// CHECK-DAG:           [[VAR_75_:%.+]] = addf [[VAR_72_]], [[VAR_74_]] : f32
// CHECK-DAG:           [[VAR_76_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_75_]], [[VAR_76_]][] : memref<f32>
// CHECK:               [[VAR_77_:%.+]] = "onnx.Sigmoid"([[VAR_76_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_77_MEM_:%.+]] = krnl.load [[VAR_77_]][] : memref<f32>
// CHECK-DAG:           [[VAR_79_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_63_]], [[VAR_79_]][] : memref<f32>
// CHECK:               [[VAR_80_:%.+]] = "onnx.Tanh"([[VAR_79_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_VAR_80_MEM_:%.+]] = krnl.load [[VAR_80_]][] : memref<f32>
// CHECK:               [[VAR_82_:%.+]] = mulf [[LOAD_VAR_77_MEM_]], [[LOAD_VAR_80_MEM_]] : f32
// CHECK:               krnl.store [[VAR_63_]], [[VAR_0_]]{{.}}[[VAR_19_1_]]#0, [[VAR_19_1_]]#1] : memref<2x4xf32>
// CHECK:               krnl.store [[VAR_82_]], [[VAR_1_]]{{.}}[[VAR_1_]]9#0, [[VAR_1_]]9#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc [[LOAD_PARAM_4_MEM_1_]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           [[CST_32_:%.+]] = constant 32 : i64
// CHECK:           "krnl.memcpy"([[VAR_2_]], [[VAR_1_]], [[CST_32_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
// CHECK:           memref.dealloc [[VAR_1_]] : memref<2x4xf32>
// CHECK:           memref.dealloc [[VAR_0_]] : memref<2x4xf32>
// CHECK:           return [[VAR_2_]] : memref<1x2x4xf32>
// CHECK:         }
}

// -----

func private @test_lstm_bidirectional_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<2x16x3xf32>, %arg2: tensor<2x16x4xf32>, %arg3: tensor<2x32xf32>, %arg4: tensor<2x2x4xf32>, %arg5: tensor<2x2x4xf32>, %arg6: tensor<2x12xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4, %arg5, %arg6) {hidden_size = 4 : si64, direction = "bidirectional"} : (tensor<7x2x3xf32>, tensor<2x16x3xf32>, tensor<2x16x4xf32>, tensor<2x32xf32>, none, tensor<2x2x4xf32>, tensor<2x2x4xf32>, tensor<2x12xf32>) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

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
// CHECK-DAG:       [[VAR_12_:%.+]]:2 = "onnx.Split"([[PARAM_3_]]) {axis = 0 : si64} : (memref<2x32xf32>) -> (memref<1x32xf32>, memref<1x32xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Squeeze"([[VAR_12_]]#0) {axes = [0]} : (memref<1x32xf32>) -> memref<32xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.Squeeze"([[VAR_12_]]#1) {axes = [0]} : (memref<1x32xf32>) -> memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_15_:%.+]]:8 = "onnx.Split"([[VAR_13_]]) {axis = 0 : si64} : (memref<32xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[VAR_16_:%.+]]:8 = "onnx.Split"([[VAR_14_]]) {axis = 0 : si64} : (memref<32xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[VAR_17_:%.+]]:2 = "onnx.Split"([[PARAM_6_]]) {axis = 0 : si64} : (memref<2x12xf32>) -> (memref<1x12xf32>, memref<1x12xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Squeeze"([[VAR_17_]]#0) {axes = [0]} : (memref<1x12xf32>) -> memref<12xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.Squeeze"([[VAR_17_]]#1) {axes = [0]} : (memref<1x12xf32>) -> memref<12xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_20_:%.+]]:3 = "onnx.Split"([[VAR_18_]]) {axis = 0 : si64} : (memref<12xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[VAR_21_:%.+]]:3 = "onnx.Split"([[VAR_19_]]) {axis = 0 : si64} : (memref<12xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7) {
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_2_:%.+]] = memref.alloc() : memref<2x3xf32>
// CHECK-DAG:         [[CST_0_1_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_2_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_3_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_1_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_3_:%.+]] = constant 3 : index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_2_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_2_]] to [[CST_3_]]) {
// CHECK:               [[VAR_32_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_32_]]#0, [[VAR_32_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_4_MEM_2_]]{{.}}[[VAR_32_]]#0, [[VAR_32_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             [[LOAD_PARAM_4_MEM_1_:%.+]] = "onnx.Transpose"([[LOAD_PARAM_4_MEM_2_]]) {perm = [1, 0]} : (memref<2x3xf32>) -> memref<3x2xf32>
// CHECK-DAG:         [[LOAD_PARAM_5_MEM_1_:%.+]] = "onnx.MatMul"([[VAR_7_]], [[LOAD_PARAM_4_MEM_1_]]) : (memref<16x3xf32>, memref<3x2xf32>) -> memref<16x2xf32>
// CHECK-DAG:         [[VAR_29_:%.+]] = "onnx.Transpose"([[VAR_3_]]) {perm = [1, 0]} : (memref<2x4xf32>) -> memref<4x2xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_30_:%.+]] = "onnx.MatMul"([[VAR_10_]], [[VAR_29_]]) : (memref<16x4xf32>, memref<4x2xf32>) -> memref<16x2xf32>
// CHECK-DAG:         [[CST_0_4_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_5_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_1_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_2_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_4_]] to [[CST_2_1_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_4_]] to [[CST_4_]]) {
// CHECK:               [[VAR_32_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_2_]]{{.}}[[VAR_32_1_]]#0, [[VAR_32_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_LOAD_PARAM_5_MEM_1_MEM_:%.+]] = krnl.load [[LOAD_PARAM_5_MEM_1_]]{{.}}[[VAR_32_1_]]#1, [[VAR_32_1_]]#0] : memref<16x2xf32>
// CHECK-DAG:           [[LOAD_VAR_30_MEM_:%.+]] = krnl.load [[VAR_30_]]{{.}}[[VAR_32_1_]]#1, [[VAR_32_1_]]#0] : memref<16x2xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_36_:%.+]] = addf [[LOAD_LOAD_PARAM_5_MEM_1_MEM_]], [[LOAD_VAR_30_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_15_MEM_:%.+]] = krnl.load [[VAR_15_]]#0{{.}}[[VAR_32_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_15_MEM_1_:%.+]] = krnl.load [[VAR_15_]]#4{{.}}[[VAR_32_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_39_:%.+]] = addf [[VAR_36_]], [[LOAD_VAR_15_MEM_]] : f32
// CHECK-DAG:           [[VAR_40_:%.+]] = addf [[VAR_39_]], [[LOAD_VAR_15_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_20_MEM_:%.+]] = krnl.load [[VAR_20_]]#0{{.}}[[VAR_32_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_42_:%.+]] = mulf [[LOAD_VAR_20_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_43_:%.+]] = addf [[VAR_40_]], [[VAR_42_]] : f32
// CHECK-DAG:           [[VAR_44_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_43_]], [[VAR_44_]][] : memref<f32>
// CHECK:               [[VAR_45_:%.+]] = "onnx.Sigmoid"([[VAR_44_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_45_MEM_:%.+]] = krnl.load [[VAR_45_]][] : memref<f32>
// CHECK-DAG:           [[CST_8_:%.+]] = constant 8 : index
// CHECK-DAG:           [[VAR_47_:%.+]] = affine.apply #map0(){{.}}[[VAR_32_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_LOAD_PARAM_5_MEM_1_MEM_1_:%.+]] = krnl.load [[LOAD_PARAM_5_MEM_1_]]{{.}}[[VAR_47_]], [[VAR_32_1_]]#0] : memref<16x2xf32>
// CHECK-DAG:           [[CST_8_1_:%.+]] = constant 8 : index
// CHECK-DAG:           [[VAR_49_:%.+]] = affine.apply #map0(){{.}}[[VAR_32_1_]]#1]
// CHECK:               [[LOAD_VAR_30_MEM_1_:%.+]] = krnl.load [[VAR_30_]]{{.}}[[VAR_49_]], [[VAR_32_1_]]#0] : memref<16x2xf32>
// CHECK-DAG:           [[VAR_51_:%.+]] = addf [[LOAD_LOAD_PARAM_5_MEM_1_MEM_1_]], [[LOAD_VAR_30_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_15_MEM_2_:%.+]] = krnl.load [[VAR_15_]]#2{{.}}[[VAR_32_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_15_MEM_3_:%.+]] = krnl.load [[VAR_15_]]#6{{.}}[[VAR_32_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_54_:%.+]] = addf [[VAR_51_]], [[LOAD_VAR_15_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_55_:%.+]] = addf [[VAR_54_]], [[LOAD_VAR_15_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_20_MEM_1_:%.+]] = krnl.load [[VAR_20_]]#2{{.}}[[VAR_32_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_57_:%.+]] = mulf [[LOAD_VAR_20_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_58_:%.+]] = addf [[VAR_55_]], [[VAR_57_]] : f32
// CHECK-DAG:           [[VAR_59_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_58_]], [[VAR_59_]][] : memref<f32>
// CHECK:               [[VAR_60_:%.+]] = "onnx.Sigmoid"([[VAR_59_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_60_MEM_:%.+]] = krnl.load [[VAR_60_]][] : memref<f32>
// CHECK-DAG:           [[CST_12_:%.+]] = constant 12 : index
// CHECK-DAG:           [[VAR_62_:%.+]] = affine.apply #map1(){{.}}[[VAR_32_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_LOAD_PARAM_5_MEM_1_MEM_2_:%.+]] = krnl.load [[LOAD_PARAM_5_MEM_1_]]{{.}}[[VAR_62_]], [[VAR_32_1_]]#0] : memref<16x2xf32>
// CHECK-DAG:           [[CST_12_1_:%.+]] = constant 12 : index
// CHECK-DAG:           [[VAR_64_:%.+]] = affine.apply #map1(){{.}}[[VAR_32_1_]]#1]
// CHECK:               [[LOAD_VAR_30_MEM_2_:%.+]] = krnl.load [[VAR_30_]]{{.}}[[VAR_64_]], [[VAR_32_1_]]#0] : memref<16x2xf32>
// CHECK-DAG:           [[VAR_66_:%.+]] = addf [[LOAD_LOAD_PARAM_5_MEM_1_MEM_2_]], [[LOAD_VAR_30_MEM_2_]] : f32
// CHECK-DAG:           [[LOAD_VAR_15_MEM_4_:%.+]] = krnl.load [[VAR_15_]]#3{{.}}[[VAR_32_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_15_MEM_5_:%.+]] = krnl.load [[VAR_15_]]#7{{.}}[[VAR_32_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_69_:%.+]] = addf [[VAR_66_]], [[LOAD_VAR_15_MEM_4_]] : f32
// CHECK-DAG:           [[VAR_70_:%.+]] = addf [[VAR_69_]], [[LOAD_VAR_15_MEM_5_]] : f32
// CHECK-DAG:           [[VAR_71_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_70_]], [[VAR_71_]][] : memref<f32>
// CHECK:               [[VAR_72_:%.+]] = "onnx.Tanh"([[VAR_71_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_72_MEM_:%.+]] = krnl.load [[VAR_72_]][] : memref<f32>
// CHECK-DAG:           [[VAR_74_:%.+]] = mulf [[LOAD_VAR_60_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_75_:%.+]] = mulf [[LOAD_VAR_45_MEM_]], [[LOAD_VAR_72_MEM_]] : f32
// CHECK-DAG:           [[VAR_76_:%.+]] = addf [[VAR_74_]], [[VAR_75_]] : f32
// CHECK-DAG:           [[CST_4_1_:%.+]] = constant 4 : index
// CHECK-DAG:           [[VAR_77_:%.+]] = affine.apply #map2(){{.}}[[VAR_32_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_LOAD_PARAM_5_MEM_1_MEM_3_:%.+]] = krnl.load [[LOAD_PARAM_5_MEM_1_]]{{.}}[[VAR_77_]], [[VAR_32_1_]]#0] : memref<16x2xf32>
// CHECK-DAG:           [[CST_4_2_:%.+]] = constant 4 : index
// CHECK-DAG:           [[VAR_79_:%.+]] = affine.apply #map2(){{.}}[[VAR_32_1_]]#1]
// CHECK:               [[LOAD_VAR_30_MEM_3_:%.+]] = krnl.load [[VAR_30_]]{{.}}[[VAR_79_]], [[VAR_32_1_]]#0] : memref<16x2xf32>
// CHECK-DAG:           [[VAR_81_:%.+]] = addf [[LOAD_LOAD_PARAM_5_MEM_1_MEM_3_]], [[LOAD_VAR_30_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_15_MEM_6_:%.+]] = krnl.load [[VAR_15_]]#1{{.}}[[VAR_32_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_15_MEM_7_:%.+]] = krnl.load [[VAR_15_]]#5{{.}}[[VAR_32_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_84_:%.+]] = addf [[VAR_81_]], [[LOAD_VAR_15_MEM_6_]] : f32
// CHECK-DAG:           [[VAR_85_:%.+]] = addf [[VAR_84_]], [[LOAD_VAR_15_MEM_7_]] : f32
// CHECK-DAG:           [[LOAD_VAR_20_MEM_2_:%.+]] = krnl.load [[VAR_20_]]#1{{.}}[[VAR_32_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_87_:%.+]] = mulf [[LOAD_VAR_20_MEM_2_]], [[VAR_76_]] : f32
// CHECK-DAG:           [[VAR_88_:%.+]] = addf [[VAR_85_]], [[VAR_87_]] : f32
// CHECK-DAG:           [[VAR_89_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_88_]], [[VAR_89_]][] : memref<f32>
// CHECK:               [[VAR_90_:%.+]] = "onnx.Sigmoid"([[VAR_89_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_90_MEM_:%.+]] = krnl.load [[VAR_90_]][] : memref<f32>
// CHECK-DAG:           [[VAR_92_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_76_]], [[VAR_92_]][] : memref<f32>
// CHECK:               [[VAR_93_:%.+]] = "onnx.Tanh"([[VAR_92_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_VAR_93_MEM_:%.+]] = krnl.load [[VAR_93_]][] : memref<f32>
// CHECK:               [[VAR_95_:%.+]] = mulf [[LOAD_VAR_90_MEM_]], [[LOAD_VAR_93_MEM_]] : f32
// CHECK:               krnl.store [[VAR_76_]], [[VAR_2_]]{{.}}[[VAR_32_1_]]#0, [[VAR_32_1_]]#1] : memref<2x4xf32>
// CHECK:               krnl.store [[VAR_95_]], [[VAR_3_]]{{.}}[[VAR_3_]]2#0, [[VAR_3_]]2#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc [[LOAD_PARAM_4_MEM_2_]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_4_]]) with ([[LOOP_4_]] -> [[I_7_:%.+]] = 0 to 7) {
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_2_:%.+]] = memref.alloc() : memref<2x3xf32>
// CHECK-DAG:         [[CST_1_3_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_7_:%.+]] = constant 7 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOOP_2_:%.+]] = affine.apply #map3([[I_7_]]){{.}}[[CST_7_]]{{.}}
// CHECK-DAG:         [[CST_0_6_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_7_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_2_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_4_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_3_1_:%.+]] = constant 3 : index
// CHECK-DAG:         [[LOOP_5_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_5_]]#0, [[LOOP_5_]]#1) with ([[LOOP_5_]]#0 -> [[I_8_:%.+]] = [[CST_0_6_]] to [[CST_2_2_]], [[LOOP_5_]]#1 -> [[I_9_:%.+]] = [[CST_0_6_]] to [[CST_3_1_]]) {
// CHECK:               [[LOAD_PARAM_0_MEM_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_5_]]#0, [[LOOP_5_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[LOOP_2_]], [[LOAD_PARAM_0_MEM_1_]]#0, [[LOAD_PARAM_0_MEM_1_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_2_]], [[LOAD_PARAM_4_MEM_2_]]{{.}}[[LOAD_PARAM_0_MEM_1_]]#0, [[LOAD_PARAM_0_MEM_1_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             [[LOAD_PARAM_5_MEM_1_1_:%.+]] = "onnx.Transpose"([[LOAD_PARAM_4_MEM_2_]]) {perm = [1, 0]} : (memref<2x3xf32>) -> memref<3x2xf32>
// CHECK-DAG:         [[VAR_29_1_:%.+]] = "onnx.MatMul"([[VAR_8_]], [[LOAD_PARAM_5_MEM_1_1_]]) : (memref<16x3xf32>, memref<3x2xf32>) -> memref<16x2xf32>
// CHECK-DAG:         [[VAR_30_1_:%.+]] = "onnx.Transpose"([[VAR_1_]]) {perm = [1, 0]} : (memref<2x4xf32>) -> memref<4x2xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOOP_3_:%.+]] = "onnx.MatMul"([[VAR_11_]], [[VAR_30_1_]]) : (memref<16x4xf32>, memref<4x2xf32>) -> memref<16x2xf32>
// CHECK-DAG:         [[CST_0_8_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_9_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_3_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_5_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_4_3_:%.+]] = constant 4 : index
// CHECK-DAG:         [[LOOP_6_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_6_]]#0, [[LOOP_6_]]#1) with ([[LOOP_6_]]#0 -> [[I_10_:%.+]] = [[CST_0_8_]] to [[CST_2_3_]], [[LOOP_6_]]#1 -> [[I_11_:%.+]] = [[CST_0_8_]] to [[CST_4_3_]]) {
// CHECK:               [[LOAD_PARAM_0_MEM_1_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_6_]]#0, [[LOOP_6_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_30_MEM_4_:%.+]] = krnl.load [[VAR_29_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1, [[LOAD_PARAM_0_MEM_1_1_]]#0] : memref<16x2xf32>
// CHECK-DAG:           [[VAR_36_1_:%.+]] = krnl.load [[LOOP_3_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1, [[LOAD_PARAM_0_MEM_1_1_]]#0] : memref<16x2xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_15_MEM_8_:%.+]] = addf [[LOAD_VAR_30_MEM_4_]], [[VAR_36_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_15_MEM_1_:%.+]] = krnl.load [[VAR_16_]]#0{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_39_1_:%.+]] = krnl.load [[VAR_16_]]#4{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_40_1_:%.+]] = addf [[LOAD_VAR_15_MEM_8_]], [[LOAD_VAR_15_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_20_MEM_3_:%.+]] = addf [[VAR_40_1_]], [[VAR_39_1_]] : f32
// CHECK-DAG:           [[VAR_42_1_:%.+]] = krnl.load [[VAR_21_]]#0{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_43_1_:%.+]] = mulf [[VAR_42_1_]], [[LOAD_PARAM_0_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_44_1_:%.+]] = addf [[LOAD_VAR_20_MEM_3_]], [[VAR_43_1_]] : f32
// CHECK-DAG:           [[VAR_45_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_44_1_]], [[VAR_45_1_]][] : memref<f32>
// CHECK:               [[LOAD_VAR_45_MEM_1_:%.+]] = "onnx.Sigmoid"([[VAR_45_1_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[VAR_47_1_:%.+]] = krnl.load [[LOAD_VAR_45_MEM_1_]][] : memref<f32>
// CHECK-DAG:           [[CST_8_2_:%.+]] = constant 8 : index
// CHECK-DAG:           [[LOAD_LOAD_PARAM_5_MEM_1_MEM_1_:%.+]] = affine.apply #map0(){{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_49_1_:%.+]] = krnl.load [[VAR_29_1_]]{{.}}[[LOAD_LOAD_PARAM_5_MEM_1_MEM_1_]], [[LOAD_PARAM_0_MEM_1_1_]]#0] : memref<16x2xf32>
// CHECK-DAG:           [[CST_8_3_:%.+]] = constant 8 : index
// CHECK-DAG:           [[LOAD_VAR_30_MEM_1_:%.+]] = affine.apply #map0(){{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1]
// CHECK:               [[VAR_51_1_:%.+]] = krnl.load [[LOOP_3_]]{{.}}[[LOAD_VAR_30_MEM_1_]], [[LOAD_PARAM_0_MEM_1_1_]]#0] : memref<16x2xf32>
// CHECK-DAG:           [[LOAD_VAR_15_MEM_2_:%.+]] = addf [[VAR_49_1_]], [[VAR_51_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_15_MEM_3_:%.+]] = krnl.load [[VAR_16_]]#2{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_54_1_:%.+]] = krnl.load [[VAR_16_]]#6{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_55_1_:%.+]] = addf [[LOAD_VAR_15_MEM_2_]], [[LOAD_VAR_15_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_20_MEM_1_:%.+]] = addf [[VAR_55_1_]], [[VAR_54_1_]] : f32
// CHECK-DAG:           [[VAR_57_1_:%.+]] = krnl.load [[VAR_21_]]#2{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_58_1_:%.+]] = mulf [[VAR_57_1_]], [[LOAD_PARAM_0_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_59_1_:%.+]] = addf [[LOAD_VAR_20_MEM_1_]], [[VAR_58_1_]] : f32
// CHECK-DAG:           [[VAR_60_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_59_1_]], [[VAR_60_1_]][] : memref<f32>
// CHECK:               [[LOAD_VAR_60_MEM_1_:%.+]] = "onnx.Sigmoid"([[VAR_60_1_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[VAR_62_1_:%.+]] = krnl.load [[LOAD_VAR_60_MEM_1_]][] : memref<f32>
// CHECK-DAG:           [[CST_12_2_:%.+]] = constant 12 : index
// CHECK-DAG:           [[LOAD_LOAD_PARAM_5_MEM_1_MEM_2_:%.+]] = affine.apply #map1(){{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_64_1_:%.+]] = krnl.load [[VAR_29_1_]]{{.}}[[LOAD_LOAD_PARAM_5_MEM_1_MEM_2_]], [[LOAD_PARAM_0_MEM_1_1_]]#0] : memref<16x2xf32>
// CHECK-DAG:           [[CST_12_3_:%.+]] = constant 12 : index
// CHECK-DAG:           [[LOAD_VAR_30_MEM_2_:%.+]] = affine.apply #map1(){{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1]
// CHECK:               [[VAR_66_1_:%.+]] = krnl.load [[LOOP_3_]]{{.}}[[LOAD_VAR_30_MEM_2_]], [[LOAD_PARAM_0_MEM_1_1_]]#0] : memref<16x2xf32>
// CHECK-DAG:           [[LOAD_VAR_15_MEM_4_:%.+]] = addf [[VAR_64_1_]], [[VAR_66_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_15_MEM_5_:%.+]] = krnl.load [[VAR_16_]]#3{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_69_1_:%.+]] = krnl.load [[VAR_16_]]#7{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_70_1_:%.+]] = addf [[LOAD_VAR_15_MEM_4_]], [[LOAD_VAR_15_MEM_5_]] : f32
// CHECK-DAG:           [[VAR_71_1_:%.+]] = addf [[VAR_70_1_]], [[VAR_69_1_]] : f32
// CHECK-DAG:           [[VAR_72_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_71_1_]], [[VAR_72_1_]][] : memref<f32>
// CHECK:               [[LOAD_VAR_72_MEM_1_:%.+]] = "onnx.Tanh"([[VAR_72_1_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[VAR_74_1_:%.+]] = krnl.load [[LOAD_VAR_72_MEM_1_]][] : memref<f32>
// CHECK-DAG:           [[VAR_75_1_:%.+]] = mulf [[VAR_62_1_]], [[LOAD_PARAM_0_MEM_2_]] : f32
// CHECK:               [[VAR_76_1_:%.+]] = mulf [[VAR_47_1_]], [[VAR_74_1_]] : f32
// CHECK-DAG:           [[VAR_77_1_:%.+]] = addf [[VAR_75_1_]], [[VAR_76_1_]] : f32
// CHECK-DAG:           [[CST_4_4_:%.+]] = constant 4 : index
// CHECK-DAG:           [[LOAD_LOAD_PARAM_5_MEM_1_MEM_3_:%.+]] = affine.apply #map2(){{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_79_1_:%.+]] = krnl.load [[VAR_29_1_]]{{.}}[[LOAD_LOAD_PARAM_5_MEM_1_MEM_3_]], [[LOAD_PARAM_0_MEM_1_1_]]#0] : memref<16x2xf32>
// CHECK-DAG:           [[CST_4_5_:%.+]] = constant 4 : index
// CHECK-DAG:           [[LOAD_VAR_30_MEM_3_:%.+]] = affine.apply #map2(){{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1]
// CHECK:               [[VAR_81_1_:%.+]] = krnl.load [[LOOP_3_]]{{.}}[[LOAD_VAR_30_MEM_3_]], [[LOAD_PARAM_0_MEM_1_1_]]#0] : memref<16x2xf32>
// CHECK-DAG:           [[LOAD_VAR_15_MEM_6_:%.+]] = addf [[VAR_79_1_]], [[VAR_81_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_15_MEM_7_:%.+]] = krnl.load [[VAR_16_]]#1{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_84_1_:%.+]] = krnl.load [[VAR_16_]]#5{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_85_1_:%.+]] = addf [[LOAD_VAR_15_MEM_6_]], [[LOAD_VAR_15_MEM_7_]] : f32
// CHECK-DAG:           [[LOAD_VAR_20_MEM_2_:%.+]] = addf [[VAR_85_1_]], [[VAR_84_1_]] : f32
// CHECK-DAG:           [[VAR_87_1_:%.+]] = krnl.load [[VAR_21_]]#1{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_88_1_:%.+]] = mulf [[VAR_87_1_]], [[VAR_77_1_]] : f32
// CHECK-DAG:           [[VAR_89_1_:%.+]] = addf [[LOAD_VAR_20_MEM_2_]], [[VAR_88_1_]] : f32
// CHECK-DAG:           [[VAR_90_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_89_1_]], [[VAR_90_1_]][] : memref<f32>
// CHECK:               [[LOAD_VAR_90_MEM_1_:%.+]] = "onnx.Sigmoid"([[VAR_90_1_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[VAR_92_1_:%.+]] = krnl.load [[LOAD_VAR_90_MEM_1_]][] : memref<f32>
// CHECK-DAG:           [[VAR_93_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_77_1_]], [[VAR_93_1_]][] : memref<f32>
// CHECK:               [[LOAD_VAR_93_MEM_1_:%.+]] = "onnx.Tanh"([[VAR_93_1_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[VAR_95_1_:%.+]] = krnl.load [[LOAD_VAR_93_MEM_1_]][] : memref<f32>
// CHECK:               [[VAR_96_:%.+]] = mulf [[VAR_92_1_]], [[VAR_95_1_]] : f32
// CHECK:               krnl.store [[VAR_77_1_]], [[VAR_0_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK:               krnl.store [[VAR_96_]], [[VAR_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc [[LOAD_PARAM_4_MEM_2_]] : memref<2x3xf32>
// CHECK:           }
// CHECK-DAG:       [[CST_0_10_:%.+]] = constant 0 : index
// CHECK-DAG:       [[CST_1_6_:%.+]] = constant 1 : index
// CHECK-DAG:       [[CST_0_11_:%.+]] = constant 0 : index
// CHECK-DAG:       [[CST_2_4_:%.+]] = constant 2 : index
// CHECK-DAG:       [[CST_1_7_:%.+]] = constant 1 : index
// CHECK-DAG:       [[CST_4_6_:%.+]] = constant 4 : index
// CHECK-DAG:       [[LOOP_7_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_7_]]#0, [[LOOP_7_]]#1) with ([[LOOP_7_]]#0 -> [[I_12_:%.+]] = [[CST_0_10_]] to [[CST_2_4_]], [[LOOP_7_]]#1 -> [[I_13_:%.+]] = [[CST_0_10_]] to [[CST_4_6_]]) {
// CHECK:             [[LOAD_PARAM_4_MEM_2_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_7_]]#0, [[LOOP_7_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOOP_2_1_:%.+]] = krnl.load [[VAR_3_]]{{.}}[[LOAD_PARAM_4_MEM_2_1_]]#0, [[LOAD_PARAM_4_MEM_2_1_]]#1] : memref<2x4xf32>
// CHECK:             krnl.store [[LOOP_2_1_]], [[VAR_4_]]{{.}}[[CST_0_10_]], [[LOAD_PARAM_4_MEM_2_1_]]#0, [[LOAD_PARAM_4_MEM_2_1_]]#1] : memref<2x2x4xf32>
// CHECK:             [[LOOP_5_:%.+]] = krnl.load [[VAR_1_]]{{.}}[[LOAD_PARAM_4_MEM_2_1_]]#0, [[LOAD_PARAM_4_MEM_2_1_]]#1] : memref<2x4xf32>
// CHECK:             krnl.store [[LOOP_5_]], [[VAR_4_]]{{.}}[[CST_1_6_]], [[LOAD_PARAM_4_MEM_2_1_]]#0, [[LOAD_PARAM_4_MEM_2_1_]]#1] : memref<2x2x4xf32>
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
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Squeeze"([[PARAM_3_]]) {axes = [0]} : (memref<1x32xf32>) -> memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]]:8 = "onnx.Split"([[VAR_10_]]) {axis = 0 : si64} : (memref<32xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Squeeze"([[PARAM_6_]]) {axes = [0]} : (memref<1x12xf32>) -> memref<12xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_13_:%.+]]:3 = "onnx.Split"([[VAR_12_]]) {axis = 0 : si64} : (memref<12xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK-DAG:       [[CST_0_2_:%.+]] = constant 0 : index
// CHECK:           [[VAR_15_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_2_]] : memref<?x?x?xf32>
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to [[VAR_15_]]) {
// CHECK-DAG:         [[CST_0_3_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_1_4_:%.+]] = constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_4_]] : memref<?x?x?xf32>
// CHECK-DAG:         [[CST_2_:%.+]] = constant 2 : index
// CHECK:             [[LOAD_PARAM_5_MEM_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<?x?x?xf32>
// CHECK-DAG:         [[VAR_21_:%.+]] = memref.alloc([[LOAD_PARAM_4_MEM_1_]], [[LOAD_PARAM_5_MEM_1_]]) : memref<?x?xf32>
// CHECK-DAG:         [[CST_0_4_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_5_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_1_5_:%.+]] = constant 1 : index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_4_]] to [[LOAD_PARAM_4_MEM_1_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_4_]] to [[LOAD_PARAM_5_MEM_1_]]) {
// CHECK:               [[VAR_28_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_28_]]#0, [[VAR_28_]]#1] : memref<?x?x?xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[VAR_21_]]{{.}}[[VAR_28_]]#0, [[VAR_28_]]#1] : memref<?x?xf32>
// CHECK:             }
// CHECK:             [[VAR_23_:%.+]] = "onnx.Transpose"([[VAR_21_]]) {perm = [1, 0]} : (memref<?x?xf32>) -> memref<?x?xf32>
// CHECK-DAG:         [[VAR_24_:%.+]] = "onnx.MatMul"([[VAR_8_]], [[VAR_23_]]) : (memref<16x?xf32>, memref<?x?xf32>) -> memref<16x?xf32>
// CHECK-DAG:         [[VAR_25_:%.+]] = "onnx.Transpose"([[VAR_3_]]) {perm = [1, 0]} : (memref<?x4xf32>) -> memref<4x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_26_:%.+]] = "onnx.MatMul"([[VAR_9_]], [[VAR_25_]]) : (memref<16x4xf32>, memref<4x?xf32>) -> memref<16x?xf32>
// CHECK-DAG:         [[CST_0_6_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_7_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_1_6_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_6_]] to [[VAR_2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_6_]] to [[CST_4_]]) {
// CHECK:               [[VAR_28_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_5_]]{{.}}[[VAR_28_1_]]#0, [[VAR_28_1_]]#1] : memref<?x4xf32>
// CHECK-DAG:           [[LOAD_VAR_24_MEM_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_28_1_]]#1, [[VAR_28_1_]]#0] : memref<16x?xf32>
// CHECK-DAG:           [[LOAD_VAR_26_MEM_:%.+]] = krnl.load [[VAR_26_]]{{.}}[[VAR_28_1_]]#1, [[VAR_28_1_]]#0] : memref<16x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_32_:%.+]] = addf [[LOAD_VAR_24_MEM_]], [[LOAD_VAR_26_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_:%.+]] = krnl.load [[VAR_11_]]#0{{.}}[[VAR_28_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_11_MEM_1_:%.+]] = krnl.load [[VAR_11_]]#4{{.}}[[VAR_28_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_35_:%.+]] = addf [[VAR_32_]], [[LOAD_VAR_11_MEM_]] : f32
// CHECK-DAG:           [[VAR_36_:%.+]] = addf [[VAR_35_]], [[LOAD_VAR_11_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_13_MEM_:%.+]] = krnl.load [[VAR_13_]]#0{{.}}[[VAR_28_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_38_:%.+]] = mulf [[LOAD_VAR_13_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_39_:%.+]] = addf [[VAR_36_]], [[VAR_38_]] : f32
// CHECK-DAG:           [[VAR_40_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_39_]], [[VAR_40_]][] : memref<f32>
// CHECK:               [[VAR_41_:%.+]] = "onnx.Sigmoid"([[VAR_40_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_41_MEM_:%.+]] = krnl.load [[VAR_41_]][] : memref<f32>
// CHECK-DAG:           [[CST_8_:%.+]] = constant 8 : index
// CHECK-DAG:           [[VAR_43_:%.+]] = affine.apply #map0(){{.}}[[VAR_28_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_24_MEM_1_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_43_]], [[VAR_28_1_]]#0] : memref<16x?xf32>
// CHECK-DAG:           [[CST_8_1_:%.+]] = constant 8 : index
// CHECK-DAG:           [[VAR_45_:%.+]] = affine.apply #map0(){{.}}[[VAR_28_1_]]#1]
// CHECK:               [[LOAD_VAR_26_MEM_1_:%.+]] = krnl.load [[VAR_26_]]{{.}}[[VAR_45_]], [[VAR_28_1_]]#0] : memref<16x?xf32>
// CHECK-DAG:           [[VAR_47_:%.+]] = addf [[LOAD_VAR_24_MEM_1_]], [[LOAD_VAR_26_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_2_:%.+]] = krnl.load [[VAR_11_]]#2{{.}}[[VAR_28_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_11_MEM_3_:%.+]] = krnl.load [[VAR_11_]]#6{{.}}[[VAR_28_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_50_:%.+]] = addf [[VAR_47_]], [[LOAD_VAR_11_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_51_:%.+]] = addf [[VAR_50_]], [[LOAD_VAR_11_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_13_MEM_1_:%.+]] = krnl.load [[VAR_13_]]#2{{.}}[[VAR_28_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_53_:%.+]] = mulf [[LOAD_VAR_13_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_54_:%.+]] = addf [[VAR_51_]], [[VAR_53_]] : f32
// CHECK-DAG:           [[VAR_55_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_54_]], [[VAR_55_]][] : memref<f32>
// CHECK:               [[VAR_56_:%.+]] = "onnx.Sigmoid"([[VAR_55_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_56_MEM_:%.+]] = krnl.load [[VAR_56_]][] : memref<f32>
// CHECK-DAG:           [[CST_12_:%.+]] = constant 12 : index
// CHECK-DAG:           [[VAR_58_:%.+]] = affine.apply #map1(){{.}}[[VAR_28_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_24_MEM_2_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_58_]], [[VAR_28_1_]]#0] : memref<16x?xf32>
// CHECK-DAG:           [[CST_12_1_:%.+]] = constant 12 : index
// CHECK-DAG:           [[VAR_60_:%.+]] = affine.apply #map1(){{.}}[[VAR_28_1_]]#1]
// CHECK:               [[LOAD_VAR_26_MEM_2_:%.+]] = krnl.load [[VAR_26_]]{{.}}[[VAR_60_]], [[VAR_28_1_]]#0] : memref<16x?xf32>
// CHECK-DAG:           [[VAR_62_:%.+]] = addf [[LOAD_VAR_24_MEM_2_]], [[LOAD_VAR_26_MEM_2_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_4_:%.+]] = krnl.load [[VAR_11_]]#3{{.}}[[VAR_28_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_11_MEM_5_:%.+]] = krnl.load [[VAR_11_]]#7{{.}}[[VAR_28_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_65_:%.+]] = addf [[VAR_62_]], [[LOAD_VAR_11_MEM_4_]] : f32
// CHECK-DAG:           [[VAR_66_:%.+]] = addf [[VAR_65_]], [[LOAD_VAR_11_MEM_5_]] : f32
// CHECK-DAG:           [[VAR_67_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_66_]], [[VAR_67_]][] : memref<f32>
// CHECK:               [[VAR_68_:%.+]] = "onnx.Tanh"([[VAR_67_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_68_MEM_:%.+]] = krnl.load [[VAR_68_]][] : memref<f32>
// CHECK-DAG:           [[VAR_70_:%.+]] = mulf [[LOAD_VAR_56_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_71_:%.+]] = mulf [[LOAD_VAR_41_MEM_]], [[LOAD_VAR_68_MEM_]] : f32
// CHECK-DAG:           [[VAR_72_:%.+]] = addf [[VAR_70_]], [[VAR_71_]] : f32
// CHECK-DAG:           [[CST_4_1_:%.+]] = constant 4 : index
// CHECK-DAG:           [[VAR_73_:%.+]] = affine.apply #map2(){{.}}[[VAR_28_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_24_MEM_3_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_73_]], [[VAR_28_1_]]#0] : memref<16x?xf32>
// CHECK-DAG:           [[CST_4_2_:%.+]] = constant 4 : index
// CHECK-DAG:           [[VAR_75_:%.+]] = affine.apply #map2(){{.}}[[VAR_28_1_]]#1]
// CHECK:               [[LOAD_VAR_26_MEM_3_:%.+]] = krnl.load [[VAR_26_]]{{.}}[[VAR_75_]], [[VAR_28_1_]]#0] : memref<16x?xf32>
// CHECK-DAG:           [[VAR_77_:%.+]] = addf [[LOAD_VAR_24_MEM_3_]], [[LOAD_VAR_26_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_6_:%.+]] = krnl.load [[VAR_11_]]#1{{.}}[[VAR_28_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_11_MEM_7_:%.+]] = krnl.load [[VAR_11_]]#5{{.}}[[VAR_28_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_80_:%.+]] = addf [[VAR_77_]], [[LOAD_VAR_11_MEM_6_]] : f32
// CHECK-DAG:           [[VAR_81_:%.+]] = addf [[VAR_80_]], [[LOAD_VAR_11_MEM_7_]] : f32
// CHECK-DAG:           [[LOAD_VAR_13_MEM_2_:%.+]] = krnl.load [[VAR_13_]]#1{{.}}[[VAR_28_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_83_:%.+]] = mulf [[LOAD_VAR_13_MEM_2_]], [[VAR_72_]] : f32
// CHECK-DAG:           [[VAR_84_:%.+]] = addf [[VAR_81_]], [[VAR_83_]] : f32
// CHECK-DAG:           [[VAR_85_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_84_]], [[VAR_85_]][] : memref<f32>
// CHECK:               [[VAR_86_:%.+]] = "onnx.Sigmoid"([[VAR_85_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_86_MEM_:%.+]] = krnl.load [[VAR_86_]][] : memref<f32>
// CHECK-DAG:           [[VAR_88_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_72_]], [[VAR_88_]][] : memref<f32>
// CHECK:               [[VAR_89_:%.+]] = "onnx.Tanh"([[VAR_88_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_VAR_89_MEM_:%.+]] = krnl.load [[VAR_89_]][] : memref<f32>
// CHECK:               [[VAR_91_:%.+]] = mulf [[LOAD_VAR_86_MEM_]], [[LOAD_VAR_89_MEM_]] : f32
// CHECK:               krnl.store [[VAR_72_]], [[VAR_5_]]{{.}}[[VAR_28_1_]]#0, [[VAR_28_1_]]#1] : memref<?x4xf32>
// CHECK:               krnl.store [[VAR_91_]], [[VAR_3_]]{{.}}[[VAR_28_1_]]#0, [[VAR_28_1_]]#1] : memref<?x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc [[VAR_21_]] : memref<?x?xf32>
// CHECK:           }
// CHECK-DAG:       [[CST_16_:%.+]] = constant 16 : i64
// CHECK-DAG:       [[CST_0_8_:%.+]] = constant 0 : index
// CHECK:           [[VAR_16_:%.+]] = memref.dim [[VAR_3_]], [[CST_0_8_]] : memref<?x4xf32>
// CHECK:           [[VAR_17_:%.+]] = index_cast [[VAR_16_]] : index to i64
// CHECK:           [[VAR_18_:%.+]] = muli [[CST_16_]], [[VAR_17_]] : i64
// CHECK:           "krnl.memcpy"([[VAR_1_]], [[VAR_3_]], [[VAR_1_]]8) : (memref<1x?x4xf32>, memref<?x4xf32>, i64) -> ()
// CHECK:           memref.dealloc [[VAR_3_]] : memref<?x4xf32>
// CHECK:           memref.dealloc [[VAR_5_]] : memref<?x4xf32>
// CHECK:           return [[VAR_1_]] : memref<1x?x4xf32>
// CHECK:         }
}
