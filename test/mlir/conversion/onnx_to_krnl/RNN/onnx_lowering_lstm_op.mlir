// RUN: onnx-mlir-opt -O3 --shape-inference --convert-onnx-to-krnl='emit-intermediate-ir' --canonicalize %s -split-input-file | FileCheck %s

func.func private @test_lstm_forward_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x16x3xf32>, %arg2: tensor<1x16x4xf32>, %arg3: tensor<1x32xf32>, %arg4: tensor<1x2x4xf32>, %arg5: tensor<1x2x4xf32>, %arg6: tensor<1x12xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4, %arg5, %arg6) {hidden_size = 4 : si64} : (tensor<7x2x3xf32>, tensor<1x16x3xf32>, tensor<1x16x4xf32>, tensor<1x32xf32>, none, tensor<1x2x4xf32>, tensor<1x2x4xf32>, tensor<1x12xf32>) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>
// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 8)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 12)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 + 4)>
// CHECK-LABEL:  func.func private @test_lstm_forward_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x16x3xf32>, [[PARAM_2_:%.+]]: memref<1x16x4xf32>, [[PARAM_3_:%.+]]: memref<1x32xf32>, [[PARAM_4_:%.+]]: memref<1x2x4xf32>, [[PARAM_5_:%.+]]: memref<1x2x4xf32>, [[PARAM_6_:%.+]]: memref<1x12xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : i64
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_6_]] : memref<1x12xf32> to tensor<1x12xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_3_]] : memref<1x32xf32> to tensor<1x32xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_2_]] : memref<1x16x4xf32> to tensor<1x16x4xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_1_]] : memref<1x16x3xf32> to tensor<1x16x3xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[VAR_25_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_]], [[VAR_25_]]#0, [[VAR_25_]]#1] : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]]{{.}}[[VAR_25_]]#0, [[VAR_25_]]#1] : memref<2x4xf32>
// CHECK:             [[LOAD_PARAM_5_MEM_:%.+]] = krnl.load [[PARAM_5_]]{{.}}[[CST_0_]], [[VAR_25_]]#0, [[VAR_25_]]#1] : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_5_MEM_]], [[RES_2_]]{{.}}[[VAR_25_]]#0, [[VAR_25_]]#1] : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.SqueezeV11"([[VAR_3_]]) <{axes = [0]}> : (tensor<1x16x3xf32>) -> tensor<16x3xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.SqueezeV11"([[VAR_2_]]) <{axes = [0]}> : (tensor<1x16x4xf32>) -> tensor<16x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Transpose"([[VAR_5_]]) <{perm = [1, 0]}> : (tensor<16x3xf32>) -> tensor<3x16xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Transpose"([[VAR_6_]]) <{perm = [1, 0]}> : (tensor<16x4xf32>) -> tensor<4x16xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.SqueezeV11"([[VAR_1_]]) <{axes = [0]}> : (tensor<1x32xf32>) -> tensor<32xf32>
// CHECK:           [[VAR_10_:%.+]]:8 = "onnx.SplitV11"([[VAR_9_]]) <{axis = 0 : si64}> : (tensor<32xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
// CHECK-DAG:       [[VAR_11_:%.+]] = builtin.unrealized_conversion_cast [[VAR_10_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = builtin.unrealized_conversion_cast [[VAR_10_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = builtin.unrealized_conversion_cast [[VAR_10_]]#2 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = builtin.unrealized_conversion_cast [[VAR_10_]]#3 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = builtin.unrealized_conversion_cast [[VAR_10_]]#4 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = builtin.unrealized_conversion_cast [[VAR_10_]]#5 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = builtin.unrealized_conversion_cast [[VAR_10_]]#6 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = builtin.unrealized_conversion_cast [[VAR_10_]]#7 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.SqueezeV11"([[VAR_0_]]) <{axes = [0]}> : (tensor<1x12xf32>) -> tensor<12xf32>
// CHECK:           [[VAR_20_:%.+]]:3 = "onnx.SplitV11"([[VAR_19_]]) <{axis = 0 : si64}> : (tensor<12xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
// CHECK-DAG:       [[VAR_21_:%.+]] = builtin.unrealized_conversion_cast [[VAR_20_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = builtin.unrealized_conversion_cast [[VAR_20_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_23_:%.+]] = builtin.unrealized_conversion_cast [[VAR_20_]]#2 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7){
// CHECK-DAG:         [[VAR_25_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_]] to [[CST_3_]]){
// CHECK:               [[VAR_34_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_25_1_]], [[VAR_34_]]#0, [[VAR_34_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_3_]]{{.}}[[VAR_34_]]#0, [[VAR_34_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             [[LOAD_PARAM_5_MEM_1_:%.+]] = builtin.unrealized_conversion_cast [[RES_3_]] : memref<2x3xf32> to tensor<2x3xf32>
// CHECK:             [[VAR_28_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_5_MEM_1_]], [[VAR_7_]]) : (tensor<2x3xf32>, tensor<3x16xf32>) -> tensor<2x16xf32>
// CHECK-DAG:         [[VAR_29_:%.+]] = builtin.unrealized_conversion_cast [[VAR_28_]] : tensor<2x16xf32> to memref<2x16xf32>
// CHECK-DAG:         [[VAR_30_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_31_:%.+]] = "onnx.MatMul"([[VAR_30_]], [[VAR_8_]]) : (tensor<2x4xf32>, tensor<4x16xf32>) -> tensor<2x16xf32>
// CHECK-DAG:         [[VAR_32_:%.+]] = builtin.unrealized_conversion_cast [[VAR_31_]] : tensor<2x16xf32> to memref<2x16xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_]] to [[CST_4_]]){
// CHECK:               [[VAR_34_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_34_1_]]#0, [[VAR_34_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_29_MEM_:%.+]] = krnl.load [[VAR_29_]]{{.}}[[VAR_34_1_]]#0, [[VAR_34_1_]]#1] : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_VAR_32_MEM_:%.+]] = krnl.load [[VAR_32_]]{{.}}[[VAR_34_1_]]#0, [[VAR_34_1_]]#1] : memref<2x16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_38_:%.+]] = arith.addf [[LOAD_VAR_29_MEM_]], [[LOAD_VAR_32_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_:%.+]] = krnl.load [[VAR_11_]]{{.}}[[VAR_34_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_15_MEM_:%.+]] = krnl.load [[VAR_15_]]{{.}}[[VAR_34_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_41_:%.+]] = arith.addf [[VAR_38_]], [[LOAD_VAR_11_MEM_]] : f32
// CHECK-DAG:           [[VAR_42_:%.+]] = arith.addf [[VAR_41_]], [[LOAD_VAR_15_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_21_MEM_:%.+]] = krnl.load [[VAR_21_]]{{.}}[[VAR_34_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_44_:%.+]] = arith.mulf [[LOAD_VAR_21_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_45_:%.+]] = arith.addf [[VAR_42_]], [[VAR_44_]] : f32
// CHECK:               [[VAR_46_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_45_]] : f32
// CHECK:               [[VAR_47_:%.+]] = math.exp [[VAR_46_]] : f32
// CHECK:               [[VAR_48_:%.+]] = arith.addf [[VAR_47_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:           [[VAR_49_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_48_]] : f32
// CHECK-DAG:           [[VAR_50_:%.+]] = affine.apply [[MAP_0_]]([[VAR_34_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_29_MEM_1_:%.+]] = krnl.load [[VAR_29_]]{{.}}[[VAR_34_1_]]#0, [[VAR_50_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_52_:%.+]] = affine.apply [[MAP_0_]]([[VAR_34_1_]]#1)
// CHECK:               [[LOAD_VAR_32_MEM_1_:%.+]] = krnl.load [[VAR_32_]]{{.}}[[VAR_34_1_]]#0, [[VAR_52_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_54_:%.+]] = arith.addf [[LOAD_VAR_29_MEM_1_]], [[LOAD_VAR_32_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_13_MEM_:%.+]] = krnl.load [[VAR_13_]]{{.}}[[VAR_34_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_17_MEM_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_34_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_57_:%.+]] = arith.addf [[VAR_54_]], [[LOAD_VAR_13_MEM_]] : f32
// CHECK-DAG:           [[VAR_58_:%.+]] = arith.addf [[VAR_57_]], [[LOAD_VAR_17_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_23_MEM_:%.+]] = krnl.load [[VAR_23_]]{{.}}[[VAR_34_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_60_:%.+]] = arith.mulf [[LOAD_VAR_23_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_61_:%.+]] = arith.addf [[VAR_58_]], [[VAR_60_]] : f32
// CHECK:               [[VAR_62_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_61_]] : f32
// CHECK:               [[VAR_63_:%.+]] = math.exp [[VAR_62_]] : f32
// CHECK:               [[VAR_64_:%.+]] = arith.addf [[VAR_63_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:           [[VAR_65_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_64_]] : f32
// CHECK-DAG:           [[VAR_66_:%.+]] = affine.apply [[MAP_1_]]([[VAR_34_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_29_MEM_2_:%.+]] = krnl.load [[VAR_29_]]{{.}}[[VAR_34_1_]]#0, [[VAR_66_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_68_:%.+]] = affine.apply [[MAP_1_]]([[VAR_34_1_]]#1)
// CHECK:               [[LOAD_VAR_32_MEM_2_:%.+]] = krnl.load [[VAR_32_]]{{.}}[[VAR_34_1_]]#0, [[VAR_68_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_70_:%.+]] = arith.addf [[LOAD_VAR_29_MEM_2_]], [[LOAD_VAR_32_MEM_2_]] : f32
// CHECK-DAG:           [[LOAD_VAR_14_MEM_:%.+]] = krnl.load [[VAR_14_]]{{.}}[[VAR_34_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_18_MEM_:%.+]] = krnl.load [[VAR_18_]]{{.}}[[VAR_34_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_73_:%.+]] = arith.addf [[VAR_70_]], [[LOAD_VAR_14_MEM_]] : f32
// CHECK:               [[VAR_74_:%.+]] = arith.addf [[VAR_73_]], [[LOAD_VAR_18_MEM_]] : f32
// CHECK-DAG:           [[VAR_75_:%.+]] = math.tanh [[VAR_74_]] : f32
// CHECK-DAG:           [[VAR_76_:%.+]] = arith.mulf [[VAR_65_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_77_:%.+]] = arith.mulf [[VAR_49_]], [[VAR_75_]] : f32
// CHECK-DAG:           [[VAR_78_:%.+]] = arith.addf [[VAR_76_]], [[VAR_77_]] : f32
// CHECK-DAG:           [[VAR_79_:%.+]] = affine.apply [[MAP_2_]]([[VAR_34_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_29_MEM_3_:%.+]] = krnl.load [[VAR_29_]]{{.}}[[VAR_34_1_]]#0, [[VAR_79_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_81_:%.+]] = affine.apply [[MAP_2_]]([[VAR_34_1_]]#1)
// CHECK:               [[LOAD_VAR_32_MEM_3_:%.+]] = krnl.load [[VAR_32_]]{{.}}[[VAR_34_1_]]#0, [[VAR_81_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_83_:%.+]] = arith.addf [[LOAD_VAR_29_MEM_3_]], [[LOAD_VAR_32_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_12_MEM_:%.+]] = krnl.load [[VAR_12_]]{{.}}[[VAR_34_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_16_MEM_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_34_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_86_:%.+]] = arith.addf [[VAR_83_]], [[LOAD_VAR_12_MEM_]] : f32
// CHECK-DAG:           [[VAR_87_:%.+]] = arith.addf [[VAR_86_]], [[LOAD_VAR_16_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_22_MEM_:%.+]] = krnl.load [[VAR_22_]]{{.}}[[VAR_34_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_89_:%.+]] = arith.mulf [[LOAD_VAR_22_MEM_]], [[VAR_78_]] : f32
// CHECK:               [[VAR_90_:%.+]] = arith.addf [[VAR_87_]], [[VAR_89_]] : f32
// CHECK:               [[VAR_91_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_90_]] : f32
// CHECK:               [[VAR_92_:%.+]] = math.exp [[VAR_91_]] : f32
// CHECK:               [[VAR_93_:%.+]] = arith.addf [[VAR_92_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:           [[VAR_94_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_93_]] : f32
// CHECK-DAG:           [[VAR_95_:%.+]] = math.tanh [[VAR_78_]] : f32
// CHECK:               [[VAR_96_:%.+]] = arith.mulf [[VAR_94_]], [[VAR_95_]] : f32
// CHECK:               krnl.store [[VAR_78_]], [[RES_2_]]{{.}}[[VAR_34_1_]]#0, [[VAR_34_1_]]#1] : memref<2x4xf32>
// CHECK:               krnl.store [[VAR_96_]], [[RES_1_]]{{.}}[[VAR_34_1_]]#0, [[VAR_34_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_]]_1, [[CST_8_]], [[CST_0_]], [[CST_0_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64, index, index) -> ()
// CHECK:           return [[RES_]] : memref<1x2x4xf32>
// CHECK:         }
}

// -----

func.func private @test_lstm_forward_mode_constant_weight_and_bias(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x2x4xf32>, %arg2: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %w = onnx.Constant dense<[[[1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]]> : tensor<1x16x3xf32>
  %r = onnx.Constant dense<[[[2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.]]]> : tensor<1x16x4xf32>
  %b = onnx.Constant dense<[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32.]]> : tensor<1x32xf32>
  %p = onnx.Constant dense<[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]]> : tensor<1x12xf32>

  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %w, %r, %b, %cst, %arg1, %arg2, %p) {hidden_size = 4 : si64} : (tensor<7x2x3xf32>, tensor<1x16x3xf32>, tensor<1x16x4xf32>, tensor<1x32xf32>, none, tensor<1x2x4xf32>, tensor<1x2x4xf32>, tensor<1x12xf32>) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>
// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 8)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 12)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 + 4)>
// CHECK-LABEL:  func.func private @test_lstm_forward_mode_constant_weight_and_bias
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x2x4xf32>, [[PARAM_2_:%.+]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : i64
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[VAR_15_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_]], [[VAR_15_]]#0, [[VAR_15_]]#1] : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_1_]]{{.}}[[VAR_15_]]#0, [[VAR_15_]]#1] : memref<2x4xf32>
// CHECK:             [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_0_]], [[VAR_15_]]#0, [[VAR_15_]]#1] : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_2_MEM_]], [[RES_2_]]{{.}}[[VAR_15_]]#0, [[VAR_15_]]#1] : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [3, 16], value = dense<1.000000e+00> : tensor<3x16xf32>}> : () -> memref<3x16xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [4, 16], value = dense<2.000000e+00> : tensor<4x16xf32>}> : () -> memref<4x16xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [4], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>}> : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [4], value = dense<[5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]> : tensor<4xf32>}> : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [4], value = dense<[9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01]> : tensor<4xf32>}> : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [4], value = dense<[1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01]> : tensor<4xf32>}> : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [4], value = dense<[1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01]> : tensor<4xf32>}> : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [4], value = dense<[2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01]> : tensor<4xf32>}> : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [4], value = dense<[2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01]> : tensor<4xf32>}> : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [4], value = dense<[2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01]> : tensor<4xf32>}> : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [4], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>}> : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [4], value = dense<[5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]> : tensor<4xf32>}> : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [4], value = dense<[9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01]> : tensor<4xf32>}> : () -> memref<4xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7){
// CHECK-DAG:         [[VAR_15_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_]] to [[CST_3_]]){
// CHECK:               [[VAR_26_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_15_1_]], [[VAR_26_]]#0, [[VAR_26_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_3_]]{{.}}[[VAR_26_]]#0, [[VAR_26_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_1_:%.+]] = builtin.unrealized_conversion_cast [[RES_3_]] : memref<2x3xf32> to tensor<2x3xf32>
// CHECK-DAG:         [[VAR_18_:%.+]] = builtin.unrealized_conversion_cast [[VAR_1_]] : memref<3x16xf32> to tensor<3x16xf32>
// CHECK:             [[VAR_19_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_2_MEM_1_]], [[VAR_18_]]) : (tensor<2x3xf32>, tensor<3x16xf32>) -> tensor<2x16xf32>
// CHECK-DAG:         [[VAR_20_:%.+]] = builtin.unrealized_conversion_cast [[VAR_19_]] : tensor<2x16xf32> to memref<2x16xf32>
// CHECK-DAG:         [[VAR_21_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK-DAG:         [[VAR_22_:%.+]] = builtin.unrealized_conversion_cast [[VAR_2_]] : memref<4x16xf32> to tensor<4x16xf32>
// CHECK:             [[VAR_23_:%.+]] = "onnx.MatMul"([[VAR_21_]], [[VAR_22_]]) : (tensor<2x4xf32>, tensor<4x16xf32>) -> tensor<2x16xf32>
// CHECK-DAG:         [[VAR_24_:%.+]] = builtin.unrealized_conversion_cast [[VAR_23_]] : tensor<2x16xf32> to memref<2x16xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_]] to [[CST_4_]]){
// CHECK:               [[VAR_26_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_20_MEM_:%.+]] = krnl.load [[VAR_20_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_VAR_24_MEM_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_30_:%.+]] = arith.addf [[LOAD_VAR_20_MEM_]], [[LOAD_VAR_24_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_3_MEM_:%.+]] = krnl.load [[VAR_3_]]{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_7_MEM_:%.+]] = krnl.load [[VAR_7_]]{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_33_:%.+]] = arith.addf [[VAR_30_]], [[LOAD_VAR_3_MEM_]] : f32
// CHECK-DAG:           [[VAR_34_:%.+]] = arith.addf [[VAR_33_]], [[LOAD_VAR_7_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_:%.+]] = krnl.load [[VAR_11_]]{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_36_:%.+]] = arith.mulf [[LOAD_VAR_11_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_37_:%.+]] = arith.addf [[VAR_34_]], [[VAR_36_]] : f32
// CHECK:               [[VAR_38_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_37_]] : f32
// CHECK:               [[VAR_39_:%.+]] = math.exp [[VAR_38_]] : f32
// CHECK:               [[VAR_40_:%.+]] = arith.addf [[VAR_39_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:           [[VAR_41_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_40_]] : f32
// CHECK-DAG:           [[VAR_42_:%.+]] = affine.apply [[MAP_0_]]([[VAR_26_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_20_MEM_1_:%.+]] = krnl.load [[VAR_20_]]{{.}}[[VAR_26_1_]]#0, [[VAR_42_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_44_:%.+]] = affine.apply [[MAP_0_]]([[VAR_26_1_]]#1)
// CHECK:               [[LOAD_VAR_24_MEM_1_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_26_1_]]#0, [[VAR_44_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_46_:%.+]] = arith.addf [[LOAD_VAR_20_MEM_1_]], [[LOAD_VAR_24_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_5_MEM_:%.+]] = krnl.load [[VAR_5_]]{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_9_MEM_:%.+]] = krnl.load [[VAR_9_]]{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_49_:%.+]] = arith.addf [[VAR_46_]], [[LOAD_VAR_5_MEM_]] : f32
// CHECK-DAG:           [[VAR_50_:%.+]] = arith.addf [[VAR_49_]], [[LOAD_VAR_9_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_13_MEM_:%.+]] = krnl.load [[VAR_13_]]{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_52_:%.+]] = arith.mulf [[LOAD_VAR_13_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_53_:%.+]] = arith.addf [[VAR_50_]], [[VAR_52_]] : f32
// CHECK:               [[VAR_54_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_53_]] : f32
// CHECK:               [[VAR_55_:%.+]] = math.exp [[VAR_54_]] : f32
// CHECK:               [[VAR_56_:%.+]] = arith.addf [[VAR_55_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:           [[VAR_57_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_56_]] : f32
// CHECK-DAG:           [[VAR_58_:%.+]] = affine.apply [[MAP_1_]]([[VAR_26_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_20_MEM_2_:%.+]] = krnl.load [[VAR_20_]]{{.}}[[VAR_26_1_]]#0, [[VAR_58_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_60_:%.+]] = affine.apply [[MAP_1_]]([[VAR_26_1_]]#1)
// CHECK:               [[LOAD_VAR_24_MEM_2_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_26_1_]]#0, [[VAR_60_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_62_:%.+]] = arith.addf [[LOAD_VAR_20_MEM_2_]], [[LOAD_VAR_24_MEM_2_]] : f32
// CHECK-DAG:           [[LOAD_VAR_6_MEM_:%.+]] = krnl.load [[VAR_6_]]{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_10_MEM_:%.+]] = krnl.load [[VAR_10_]]{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_65_:%.+]] = arith.addf [[VAR_62_]], [[LOAD_VAR_6_MEM_]] : f32
// CHECK:               [[VAR_66_:%.+]] = arith.addf [[VAR_65_]], [[LOAD_VAR_10_MEM_]] : f32
// CHECK-DAG:           [[VAR_67_:%.+]] = math.tanh [[VAR_66_]] : f32
// CHECK-DAG:           [[VAR_68_:%.+]] = arith.mulf [[VAR_57_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_69_:%.+]] = arith.mulf [[VAR_41_]], [[VAR_67_]] : f32
// CHECK-DAG:           [[VAR_70_:%.+]] = arith.addf [[VAR_68_]], [[VAR_69_]] : f32
// CHECK-DAG:           [[VAR_71_:%.+]] = affine.apply [[MAP_2_]]([[VAR_26_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_20_MEM_3_:%.+]] = krnl.load [[VAR_20_]]{{.}}[[VAR_26_1_]]#0, [[VAR_71_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_73_:%.+]] = affine.apply [[MAP_2_]]([[VAR_26_1_]]#1)
// CHECK:               [[LOAD_VAR_24_MEM_3_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_26_1_]]#0, [[VAR_73_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_75_:%.+]] = arith.addf [[LOAD_VAR_20_MEM_3_]], [[LOAD_VAR_24_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_4_MEM_:%.+]] = krnl.load [[VAR_4_]]{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_8_MEM_:%.+]] = krnl.load [[VAR_8_]]{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_78_:%.+]] = arith.addf [[VAR_75_]], [[LOAD_VAR_4_MEM_]] : f32
// CHECK-DAG:           [[VAR_79_:%.+]] = arith.addf [[VAR_78_]], [[LOAD_VAR_8_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_12_MEM_:%.+]] = krnl.load [[VAR_12_]]{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_81_:%.+]] = arith.mulf [[LOAD_VAR_12_MEM_]], [[VAR_70_]] : f32
// CHECK:               [[VAR_82_:%.+]] = arith.addf [[VAR_79_]], [[VAR_81_]] : f32
// CHECK:               [[VAR_83_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_82_]] : f32
// CHECK:               [[VAR_84_:%.+]] = math.exp [[VAR_83_]] : f32
// CHECK:               [[VAR_85_:%.+]] = arith.addf [[VAR_84_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:           [[VAR_86_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_85_]] : f32
// CHECK-DAG:           [[VAR_87_:%.+]] = math.tanh [[VAR_70_]] : f32
// CHECK:               [[VAR_88_:%.+]] = arith.mulf [[VAR_86_]], [[VAR_87_]] : f32
// CHECK:               krnl.store [[VAR_70_]], [[RES_2_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x4xf32>
// CHECK:               krnl.store [[VAR_88_]], [[RES_1_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_]]_1, [[CST_8_]], [[CST_0_]], [[CST_0_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64, index, index) -> ()
// CHECK:           return [[RES_]] : memref<1x2x4xf32>
// CHECK:         }
}

// -----

func.func private @test_lstm_reverse_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x16x3xf32>, %arg2: tensor<1x16x4xf32>, %arg3: tensor<1x32xf32>, %arg4: tensor<1x2x4xf32>, %arg5: tensor<1x2x4xf32>, %arg6: tensor<1x12xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4, %arg5, %arg6) {hidden_size = 4 : si64, direction = "reverse"} : (tensor<7x2x3xf32>, tensor<1x16x3xf32>, tensor<1x16x4xf32>, tensor<1x32xf32>, none, tensor<1x2x4xf32>, tensor<1x2x4xf32>, tensor<1x12xf32>) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>
// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (-d0 + 6)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 8)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 + 12)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 + 4)>
// CHECK-LABEL:  func.func private @test_lstm_reverse_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x16x3xf32>, [[PARAM_2_:%.+]]: memref<1x16x4xf32>, [[PARAM_3_:%.+]]: memref<1x32xf32>, [[PARAM_4_:%.+]]: memref<1x2x4xf32>, [[PARAM_5_:%.+]]: memref<1x2x4xf32>, [[PARAM_6_:%.+]]: memref<1x12xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : i64
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_6_]] : memref<1x12xf32> to tensor<1x12xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_3_]] : memref<1x32xf32> to tensor<1x32xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_2_]] : memref<1x16x4xf32> to tensor<1x16x4xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_1_]] : memref<1x16x3xf32> to tensor<1x16x3xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[VAR_25_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_]], [[VAR_25_]]#0, [[VAR_25_]]#1] : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]]{{.}}[[VAR_25_]]#0, [[VAR_25_]]#1] : memref<2x4xf32>
// CHECK:             [[LOAD_PARAM_5_MEM_:%.+]] = krnl.load [[PARAM_5_]]{{.}}[[CST_0_]], [[VAR_25_]]#0, [[VAR_25_]]#1] : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_5_MEM_]], [[RES_2_]]{{.}}[[VAR_25_]]#0, [[VAR_25_]]#1] : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.SqueezeV11"([[VAR_3_]]) <{axes = [0]}> : (tensor<1x16x3xf32>) -> tensor<16x3xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.SqueezeV11"([[VAR_2_]]) <{axes = [0]}> : (tensor<1x16x4xf32>) -> tensor<16x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Transpose"([[VAR_5_]]) <{perm = [1, 0]}> : (tensor<16x3xf32>) -> tensor<3x16xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Transpose"([[VAR_6_]]) <{perm = [1, 0]}> : (tensor<16x4xf32>) -> tensor<4x16xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.SqueezeV11"([[VAR_1_]]) <{axes = [0]}> : (tensor<1x32xf32>) -> tensor<32xf32>
// CHECK:           [[VAR_10_:%.+]]:8 = "onnx.SplitV11"([[VAR_9_]]) <{axis = 0 : si64}> : (tensor<32xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
// CHECK-DAG:       [[VAR_11_:%.+]] = builtin.unrealized_conversion_cast [[VAR_10_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = builtin.unrealized_conversion_cast [[VAR_10_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = builtin.unrealized_conversion_cast [[VAR_10_]]#2 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = builtin.unrealized_conversion_cast [[VAR_10_]]#3 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = builtin.unrealized_conversion_cast [[VAR_10_]]#4 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = builtin.unrealized_conversion_cast [[VAR_10_]]#5 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = builtin.unrealized_conversion_cast [[VAR_10_]]#6 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = builtin.unrealized_conversion_cast [[VAR_10_]]#7 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.SqueezeV11"([[VAR_0_]]) <{axes = [0]}> : (tensor<1x12xf32>) -> tensor<12xf32>
// CHECK:           [[VAR_20_:%.+]]:3 = "onnx.SplitV11"([[VAR_19_]]) <{axis = 0 : si64}> : (tensor<12xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
// CHECK-DAG:       [[VAR_21_:%.+]] = builtin.unrealized_conversion_cast [[VAR_20_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = builtin.unrealized_conversion_cast [[VAR_20_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_23_:%.+]] = builtin.unrealized_conversion_cast [[VAR_20_]]#2 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7){
// CHECK:             [[VAR_25_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_1_:%.+]] = affine.apply [[MAP_0_]]([[VAR_25_1_]])
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_]] to [[CST_3_]]){
// CHECK:               [[VAR_35_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[LOAD_PARAM_4_MEM_1_]], [[VAR_35_]]#0, [[VAR_35_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_3_]]{{.}}[[VAR_35_]]#0, [[VAR_35_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             [[VAR_28_:%.+]] = builtin.unrealized_conversion_cast [[RES_3_]] : memref<2x3xf32> to tensor<2x3xf32>
// CHECK:             [[VAR_29_:%.+]] = "onnx.MatMul"([[VAR_28_]], [[VAR_7_]]) : (tensor<2x3xf32>, tensor<3x16xf32>) -> tensor<2x16xf32>
// CHECK-DAG:         [[VAR_30_:%.+]] = builtin.unrealized_conversion_cast [[VAR_29_]] : tensor<2x16xf32> to memref<2x16xf32>
// CHECK-DAG:         [[VAR_31_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_32_:%.+]] = "onnx.MatMul"([[VAR_31_]], [[VAR_8_]]) : (tensor<2x4xf32>, tensor<4x16xf32>) -> tensor<2x16xf32>
// CHECK-DAG:         [[VAR_33_:%.+]] = builtin.unrealized_conversion_cast [[VAR_32_]] : tensor<2x16xf32> to memref<2x16xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_]] to [[CST_4_]]){
// CHECK:               [[VAR_35_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_35_1_]]#0, [[VAR_35_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_30_MEM_:%.+]] = krnl.load [[VAR_30_]]{{.}}[[VAR_35_1_]]#0, [[VAR_35_1_]]#1] : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_VAR_33_MEM_:%.+]] = krnl.load [[VAR_33_]]{{.}}[[VAR_35_1_]]#0, [[VAR_35_1_]]#1] : memref<2x16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_39_:%.+]] = arith.addf [[LOAD_VAR_30_MEM_]], [[LOAD_VAR_33_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_:%.+]] = krnl.load [[VAR_11_]]{{.}}[[VAR_35_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_15_MEM_:%.+]] = krnl.load [[VAR_15_]]{{.}}[[VAR_35_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_42_:%.+]] = arith.addf [[VAR_39_]], [[LOAD_VAR_11_MEM_]] : f32
// CHECK-DAG:           [[VAR_43_:%.+]] = arith.addf [[VAR_42_]], [[LOAD_VAR_15_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_21_MEM_:%.+]] = krnl.load [[VAR_21_]]{{.}}[[VAR_35_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_45_:%.+]] = arith.mulf [[LOAD_VAR_21_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_46_:%.+]] = arith.addf [[VAR_43_]], [[VAR_45_]] : f32
// CHECK:               [[VAR_47_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_46_]] : f32
// CHECK:               [[VAR_48_:%.+]] = math.exp [[VAR_47_]] : f32
// CHECK:               [[VAR_49_:%.+]] = arith.addf [[VAR_48_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:           [[VAR_50_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_49_]] : f32
// CHECK-DAG:           [[VAR_51_:%.+]] = affine.apply [[MAP_1_]]([[VAR_35_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_30_MEM_1_:%.+]] = krnl.load [[VAR_30_]]{{.}}[[VAR_35_1_]]#0, [[VAR_51_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_53_:%.+]] = affine.apply [[MAP_1_]]([[VAR_35_1_]]#1)
// CHECK:               [[LOAD_VAR_33_MEM_1_:%.+]] = krnl.load [[VAR_33_]]{{.}}[[VAR_35_1_]]#0, [[VAR_53_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_55_:%.+]] = arith.addf [[LOAD_VAR_30_MEM_1_]], [[LOAD_VAR_33_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_13_MEM_:%.+]] = krnl.load [[VAR_13_]]{{.}}[[VAR_35_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_17_MEM_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_35_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_58_:%.+]] = arith.addf [[VAR_55_]], [[LOAD_VAR_13_MEM_]] : f32
// CHECK-DAG:           [[VAR_59_:%.+]] = arith.addf [[VAR_58_]], [[LOAD_VAR_17_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_23_MEM_:%.+]] = krnl.load [[VAR_23_]]{{.}}[[VAR_35_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_61_:%.+]] = arith.mulf [[LOAD_VAR_23_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_62_:%.+]] = arith.addf [[VAR_59_]], [[VAR_61_]] : f32
// CHECK:               [[VAR_63_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_62_]] : f32
// CHECK:               [[VAR_64_:%.+]] = math.exp [[VAR_63_]] : f32
// CHECK:               [[VAR_65_:%.+]] = arith.addf [[VAR_64_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:           [[VAR_66_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_65_]] : f32
// CHECK-DAG:           [[VAR_67_:%.+]] = affine.apply [[MAP_2_]]([[VAR_35_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_30_MEM_2_:%.+]] = krnl.load [[VAR_30_]]{{.}}[[VAR_35_1_]]#0, [[VAR_67_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_69_:%.+]] = affine.apply [[MAP_2_]]([[VAR_35_1_]]#1)
// CHECK:               [[LOAD_VAR_33_MEM_2_:%.+]] = krnl.load [[VAR_33_]]{{.}}[[VAR_35_1_]]#0, [[VAR_69_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_71_:%.+]] = arith.addf [[LOAD_VAR_30_MEM_2_]], [[LOAD_VAR_33_MEM_2_]] : f32
// CHECK-DAG:           [[LOAD_VAR_14_MEM_:%.+]] = krnl.load [[VAR_14_]]{{.}}[[VAR_35_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_18_MEM_:%.+]] = krnl.load [[VAR_18_]]{{.}}[[VAR_35_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_74_:%.+]] = arith.addf [[VAR_71_]], [[LOAD_VAR_14_MEM_]] : f32
// CHECK:               [[VAR_75_:%.+]] = arith.addf [[VAR_74_]], [[LOAD_VAR_18_MEM_]] : f32
// CHECK-DAG:           [[VAR_76_:%.+]] = math.tanh [[VAR_75_]] : f32
// CHECK-DAG:           [[VAR_77_:%.+]] = arith.mulf [[VAR_66_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_78_:%.+]] = arith.mulf [[VAR_50_]], [[VAR_76_]] : f32
// CHECK-DAG:           [[VAR_79_:%.+]] = arith.addf [[VAR_77_]], [[VAR_78_]] : f32
// CHECK-DAG:           [[VAR_80_:%.+]] = affine.apply [[MAP_3_]]([[VAR_35_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_30_MEM_3_:%.+]] = krnl.load [[VAR_30_]]{{.}}[[VAR_35_1_]]#0, [[VAR_80_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_82_:%.+]] = affine.apply [[MAP_3_]]([[VAR_35_1_]]#1)
// CHECK:               [[LOAD_VAR_33_MEM_3_:%.+]] = krnl.load [[VAR_33_]]{{.}}[[VAR_35_1_]]#0, [[VAR_82_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_84_:%.+]] = arith.addf [[LOAD_VAR_30_MEM_3_]], [[LOAD_VAR_33_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_12_MEM_:%.+]] = krnl.load [[VAR_12_]]{{.}}[[VAR_35_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_16_MEM_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_35_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_87_:%.+]] = arith.addf [[VAR_84_]], [[LOAD_VAR_12_MEM_]] : f32
// CHECK-DAG:           [[VAR_88_:%.+]] = arith.addf [[VAR_87_]], [[LOAD_VAR_16_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_22_MEM_:%.+]] = krnl.load [[VAR_22_]]{{.}}[[VAR_35_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_90_:%.+]] = arith.mulf [[LOAD_VAR_22_MEM_]], [[VAR_79_]] : f32
// CHECK:               [[VAR_91_:%.+]] = arith.addf [[VAR_88_]], [[VAR_90_]] : f32
// CHECK:               [[VAR_92_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_91_]] : f32
// CHECK:               [[VAR_93_:%.+]] = math.exp [[VAR_92_]] : f32
// CHECK:               [[VAR_94_:%.+]] = arith.addf [[VAR_93_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:           [[VAR_95_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_94_]] : f32
// CHECK-DAG:           [[VAR_96_:%.+]] = math.tanh [[VAR_79_]] : f32
// CHECK:               [[VAR_97_:%.+]] = arith.mulf [[VAR_95_]], [[VAR_96_]] : f32
// CHECK:               krnl.store [[VAR_79_]], [[RES_2_]]{{.}}[[VAR_35_1_]]#0, [[VAR_35_1_]]#1] : memref<2x4xf32>
// CHECK:               krnl.store [[VAR_97_]], [[RES_1_]]{{.}}[[VAR_35_1_]]#0, [[VAR_35_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_]]_1, [[CST_8_]], [[CST_0_]], [[CST_0_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64, index, index) -> ()
// CHECK:           return [[RES_]] : memref<1x2x4xf32>
// CHECK:         }
}


// -----

func.func private @test_lstm_bidirectional_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<2x16x3xf32>, %arg2: tensor<2x16x4xf32>, %arg3: tensor<2x32xf32>, %arg4: tensor<2x2x4xf32>, %arg5: tensor<2x2x4xf32>, %arg6: tensor<2x12xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4, %arg5, %arg6) {hidden_size = 4 : si64, direction = "bidirectional"} : (tensor<7x2x3xf32>, tensor<2x16x3xf32>, tensor<2x16x4xf32>, tensor<2x32xf32>, none, tensor<2x2x4xf32>, tensor<2x2x4xf32>, tensor<2x12xf32>) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>
// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 8)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 12)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 + 4)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (-d0 + 6)>
// CHECK-LABEL:  func.func private @test_lstm_bidirectional_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<2x16x3xf32>, [[PARAM_2_:%.+]]: memref<2x16x4xf32>, [[PARAM_3_:%.+]]: memref<2x32xf32>, [[PARAM_4_:%.+]]: memref<2x2x4xf32>, [[PARAM_5_:%.+]]: memref<2x2x4xf32>, [[PARAM_6_:%.+]]: memref<2x12xf32>) -> memref<2x2x4xf32> {
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_6_]] : memref<2x12xf32> to tensor<2x12xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_3_]] : memref<2x32xf32> to tensor<2x32xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_2_]] : memref<2x16x4xf32> to tensor<2x16x4xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_1_]] : memref<2x16x3xf32> to tensor<2x16x3xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[VAR_50_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_]], [[VAR_50_]]#0, [[VAR_50_]]#1] : memref<2x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]]{{.}}[[VAR_50_]]#0, [[VAR_50_]]#1] : memref<2x4xf32>
// CHECK:             [[LOAD_PARAM_5_MEM_:%.+]] = krnl.load [[PARAM_5_]]{{.}}[[CST_0_]], [[VAR_50_]]#0, [[VAR_50_]]#1] : memref<2x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_5_MEM_]], [[RES_2_]]{{.}}[[VAR_50_]]#0, [[VAR_50_]]#1] : memref<2x4xf32>
// CHECK:             [[LOAD_PARAM_4_MEM_1_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_1_]], [[VAR_50_]]#0, [[VAR_50_]]#1] : memref<2x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_1_]], [[RES_3_]]{{.}}[[VAR_50_]]#0, [[VAR_50_]]#1] : memref<2x4xf32>
// CHECK:             [[LOAD_PARAM_5_MEM_1_:%.+]] = krnl.load [[PARAM_5_]]{{.}}[[CST_1_]], [[VAR_50_]]#0, [[VAR_50_]]#1] : memref<2x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_5_MEM_1_]], [[RES_4_]]{{.}}[[VAR_50_]]#0, [[VAR_50_]]#1] : memref<2x4xf32>
// CHECK:           }
// CHECK:           [[VAR_5_:%.+]]:2 = "onnx.SplitV11"([[VAR_3_]]) <{axis = 0 : si64}> : (tensor<2x16x3xf32>) -> (tensor<1x16x3xf32>, tensor<1x16x3xf32>)
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.SqueezeV11"([[VAR_5_]]#0) <{axes = [0]}> : (tensor<1x16x3xf32>) -> tensor<16x3xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.SqueezeV11"([[VAR_5_]]#1) <{axes = [0]}> : (tensor<1x16x3xf32>) -> tensor<16x3xf32>
// CHECK-DAG:       [[VAR_8_:%.+]]:2 = "onnx.SplitV11"([[VAR_2_]]) <{axis = 0 : si64}> : (tensor<2x16x4xf32>) -> (tensor<1x16x4xf32>, tensor<1x16x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.SqueezeV11"([[VAR_8_]]#0) <{axes = [0]}> : (tensor<1x16x4xf32>) -> tensor<16x4xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.SqueezeV11"([[VAR_8_]]#1) <{axes = [0]}> : (tensor<1x16x4xf32>) -> tensor<16x4xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Transpose"([[VAR_6_]]) <{perm = [1, 0]}> : (tensor<16x3xf32>) -> tensor<3x16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Transpose"([[VAR_9_]]) <{perm = [1, 0]}> : (tensor<16x4xf32>) -> tensor<4x16xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Transpose"([[VAR_7_]]) <{perm = [1, 0]}> : (tensor<16x3xf32>) -> tensor<3x16xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.Transpose"([[VAR_10_]]) <{perm = [1, 0]}> : (tensor<16x4xf32>) -> tensor<4x16xf32>
// CHECK-DAG:       [[VAR_15_:%.+]]:2 = "onnx.SplitV11"([[VAR_1_]]) <{axis = 0 : si64}> : (tensor<2x32xf32>) -> (tensor<1x32xf32>, tensor<1x32xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.SqueezeV11"([[VAR_15_]]#0) <{axes = [0]}> : (tensor<1x32xf32>) -> tensor<32xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.SqueezeV11"([[VAR_15_]]#1) <{axes = [0]}> : (tensor<1x32xf32>) -> tensor<32xf32>
// CHECK:           [[VAR_18_:%.+]]:8 = "onnx.SplitV11"([[VAR_16_]]) <{axis = 0 : si64}> : (tensor<32xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
// CHECK-DAG:       [[VAR_19_:%.+]] = builtin.unrealized_conversion_cast [[VAR_18_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = builtin.unrealized_conversion_cast [[VAR_18_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = builtin.unrealized_conversion_cast [[VAR_18_]]#2 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = builtin.unrealized_conversion_cast [[VAR_18_]]#3 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_23_:%.+]] = builtin.unrealized_conversion_cast [[VAR_18_]]#4 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_24_:%.+]] = builtin.unrealized_conversion_cast [[VAR_18_]]#5 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_25_:%.+]] = builtin.unrealized_conversion_cast [[VAR_18_]]#6 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_26_:%.+]] = builtin.unrealized_conversion_cast [[VAR_18_]]#7 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_27_:%.+]]:8 = "onnx.SplitV11"([[VAR_17_]]) <{axis = 0 : si64}> : (tensor<32xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_28_:%.+]] = builtin.unrealized_conversion_cast [[VAR_27_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_29_:%.+]] = builtin.unrealized_conversion_cast [[VAR_27_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_30_:%.+]] = builtin.unrealized_conversion_cast [[VAR_27_]]#2 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_31_:%.+]] = builtin.unrealized_conversion_cast [[VAR_27_]]#3 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_32_:%.+]] = builtin.unrealized_conversion_cast [[VAR_27_]]#4 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_33_:%.+]] = builtin.unrealized_conversion_cast [[VAR_27_]]#5 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_34_:%.+]] = builtin.unrealized_conversion_cast [[VAR_27_]]#6 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_35_:%.+]] = builtin.unrealized_conversion_cast [[VAR_27_]]#7 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_36_:%.+]]:2 = "onnx.SplitV11"([[VAR_0_]]) <{axis = 0 : si64}> : (tensor<2x12xf32>) -> (tensor<1x12xf32>, tensor<1x12xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_37_:%.+]] = "onnx.SqueezeV11"([[VAR_36_]]#0) <{axes = [0]}> : (tensor<1x12xf32>) -> tensor<12xf32>
// CHECK-DAG:       [[VAR_38_:%.+]] = "onnx.SqueezeV11"([[VAR_36_]]#1) <{axes = [0]}> : (tensor<1x12xf32>) -> tensor<12xf32>
// CHECK:           [[VAR_39_:%.+]]:3 = "onnx.SplitV11"([[VAR_37_]]) <{axis = 0 : si64}> : (tensor<12xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
// CHECK-DAG:       [[VAR_40_:%.+]] = builtin.unrealized_conversion_cast [[VAR_39_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_41_:%.+]] = builtin.unrealized_conversion_cast [[VAR_39_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_42_:%.+]] = builtin.unrealized_conversion_cast [[VAR_39_]]#2 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_43_:%.+]]:3 = "onnx.SplitV11"([[VAR_38_]]) <{axis = 0 : si64}> : (tensor<12xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_44_:%.+]] = builtin.unrealized_conversion_cast [[VAR_43_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_45_:%.+]] = builtin.unrealized_conversion_cast [[VAR_43_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_46_:%.+]] = builtin.unrealized_conversion_cast [[VAR_43_]]#2 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7){
// CHECK-DAG:         [[VAR_50_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_]] to [[CST_3_]]){
// CHECK:               [[VAR_59_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_50_1_]], [[VAR_59_]]#0, [[VAR_59_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_5_]]{{.}}[[VAR_59_]]#0, [[VAR_59_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             [[LOAD_PARAM_5_MEM_2_:%.+]] = builtin.unrealized_conversion_cast [[RES_5_]] : memref<2x3xf32> to tensor<2x3xf32>
// CHECK:             [[LOAD_PARAM_4_MEM_1_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_5_MEM_2_]], [[VAR_11_]]) : (tensor<2x3xf32>, tensor<3x16xf32>) -> tensor<2x16xf32>
// CHECK-DAG:         [[LOAD_PARAM_5_MEM_1_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_4_MEM_1_]] : tensor<2x16xf32> to memref<2x16xf32>
// CHECK-DAG:         [[VAR_55_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_56_:%.+]] = "onnx.MatMul"([[VAR_55_]], [[VAR_12_]]) : (tensor<2x4xf32>, tensor<4x16xf32>) -> tensor<2x16xf32>
// CHECK-DAG:         [[VAR_57_:%.+]] = builtin.unrealized_conversion_cast [[VAR_56_]] : tensor<2x16xf32> to memref<2x16xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_]] to [[CST_4_]]){
// CHECK:               [[VAR_59_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_59_1_]]#0, [[VAR_59_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_LOAD_PARAM_5_MEM_1_MEM_:%.+]] = krnl.load [[LOAD_PARAM_5_MEM_1_]]{{.}}[[VAR_59_1_]]#0, [[VAR_59_1_]]#1] : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_VAR_57_MEM_:%.+]] = krnl.load [[VAR_57_]]{{.}}[[VAR_59_1_]]#0, [[VAR_59_1_]]#1] : memref<2x16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_63_:%.+]] = arith.addf [[LOAD_LOAD_PARAM_5_MEM_1_MEM_]], [[LOAD_VAR_57_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_19_MEM_:%.+]] = krnl.load [[VAR_19_]]{{.}}[[VAR_59_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_23_MEM_:%.+]] = krnl.load [[VAR_23_]]{{.}}[[VAR_59_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_66_:%.+]] = arith.addf [[VAR_63_]], [[LOAD_VAR_19_MEM_]] : f32
// CHECK-DAG:           [[VAR_67_:%.+]] = arith.addf [[VAR_66_]], [[LOAD_VAR_23_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_40_MEM_:%.+]] = krnl.load [[VAR_40_]]{{.}}[[VAR_59_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_69_:%.+]] = arith.mulf [[LOAD_VAR_40_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_70_:%.+]] = arith.addf [[VAR_67_]], [[VAR_69_]] : f32
// CHECK:               [[VAR_71_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_70_]] : f32
// CHECK:               [[VAR_72_:%.+]] = math.exp [[VAR_71_]] : f32
// CHECK:               [[VAR_73_:%.+]] = arith.addf [[VAR_72_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:           [[VAR_74_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_73_]] : f32
// CHECK-DAG:           [[VAR_75_:%.+]] = affine.apply [[MAP_0_]]([[VAR_59_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_LOAD_PARAM_5_MEM_1_MEM_1_:%.+]] = krnl.load [[LOAD_PARAM_5_MEM_1_]]{{.}}[[VAR_59_1_]]#0, [[VAR_75_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_77_:%.+]] = affine.apply [[MAP_0_]]([[VAR_59_1_]]#1)
// CHECK:               [[LOAD_VAR_57_MEM_1_:%.+]] = krnl.load [[VAR_57_]]{{.}}[[VAR_59_1_]]#0, [[VAR_77_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_79_:%.+]] = arith.addf [[LOAD_LOAD_PARAM_5_MEM_1_MEM_1_]], [[LOAD_VAR_57_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_21_MEM_:%.+]] = krnl.load [[VAR_21_]]{{.}}[[VAR_59_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_25_MEM_:%.+]] = krnl.load [[VAR_25_]]{{.}}[[VAR_59_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_82_:%.+]] = arith.addf [[VAR_79_]], [[LOAD_VAR_21_MEM_]] : f32
// CHECK-DAG:           [[VAR_83_:%.+]] = arith.addf [[VAR_82_]], [[LOAD_VAR_25_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_42_MEM_:%.+]] = krnl.load [[VAR_42_]]{{.}}[[VAR_59_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_85_:%.+]] = arith.mulf [[LOAD_VAR_42_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_86_:%.+]] = arith.addf [[VAR_83_]], [[VAR_85_]] : f32
// CHECK:               [[VAR_87_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_86_]] : f32
// CHECK:               [[VAR_88_:%.+]] = math.exp [[VAR_87_]] : f32
// CHECK:               [[VAR_89_:%.+]] = arith.addf [[VAR_88_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:           [[VAR_90_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_89_]] : f32
// CHECK-DAG:           [[VAR_91_:%.+]] = affine.apply [[MAP_1_]]([[VAR_59_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_LOAD_PARAM_5_MEM_1_MEM_2_:%.+]] = krnl.load [[LOAD_PARAM_5_MEM_1_]]{{.}}[[VAR_59_1_]]#0, [[VAR_91_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_93_:%.+]] = affine.apply [[MAP_1_]]([[VAR_59_1_]]#1)
// CHECK:               [[LOAD_VAR_57_MEM_2_:%.+]] = krnl.load [[VAR_57_]]{{.}}[[VAR_59_1_]]#0, [[VAR_93_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_95_:%.+]] = arith.addf [[LOAD_LOAD_PARAM_5_MEM_1_MEM_2_]], [[LOAD_VAR_57_MEM_2_]] : f32
// CHECK-DAG:           [[LOAD_VAR_22_MEM_:%.+]] = krnl.load [[VAR_22_]]{{.}}[[VAR_59_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_26_MEM_:%.+]] = krnl.load [[VAR_26_]]{{.}}[[VAR_59_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_98_:%.+]] = arith.addf [[VAR_95_]], [[LOAD_VAR_22_MEM_]] : f32
// CHECK:               [[VAR_99_:%.+]] = arith.addf [[VAR_98_]], [[LOAD_VAR_26_MEM_]] : f32
// CHECK-DAG:           [[VAR_100_:%.+]] = math.tanh [[VAR_99_]] : f32
// CHECK-DAG:           [[VAR_101_:%.+]] = arith.mulf [[VAR_90_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_102_:%.+]] = arith.mulf [[VAR_74_]], [[VAR_100_]] : f32
// CHECK-DAG:           [[VAR_103_:%.+]] = arith.addf [[VAR_101_]], [[VAR_102_]] : f32
// CHECK-DAG:           [[VAR_104_:%.+]] = affine.apply [[MAP_2_]]([[VAR_59_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_LOAD_PARAM_5_MEM_1_MEM_3_:%.+]] = krnl.load [[LOAD_PARAM_5_MEM_1_]]{{.}}[[VAR_59_1_]]#0, [[VAR_104_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_106_:%.+]] = affine.apply [[MAP_2_]]([[VAR_59_1_]]#1)
// CHECK:               [[LOAD_VAR_57_MEM_3_:%.+]] = krnl.load [[VAR_57_]]{{.}}[[VAR_59_1_]]#0, [[VAR_106_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_108_:%.+]] = arith.addf [[LOAD_LOAD_PARAM_5_MEM_1_MEM_3_]], [[LOAD_VAR_57_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_20_MEM_:%.+]] = krnl.load [[VAR_20_]]{{.}}[[VAR_59_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_24_MEM_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_59_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_111_:%.+]] = arith.addf [[VAR_108_]], [[LOAD_VAR_20_MEM_]] : f32
// CHECK-DAG:           [[VAR_112_:%.+]] = arith.addf [[VAR_111_]], [[LOAD_VAR_24_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_41_MEM_:%.+]] = krnl.load [[VAR_41_]]{{.}}[[VAR_59_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_114_:%.+]] = arith.mulf [[LOAD_VAR_41_MEM_]], [[VAR_103_]] : f32
// CHECK:               [[VAR_115_:%.+]] = arith.addf [[VAR_112_]], [[VAR_114_]] : f32
// CHECK:               [[VAR_116_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_115_]] : f32
// CHECK:               [[VAR_117_:%.+]] = math.exp [[VAR_116_]] : f32
// CHECK:               [[VAR_118_:%.+]] = arith.addf [[VAR_117_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:           [[VAR_119_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_118_]] : f32
// CHECK-DAG:           [[VAR_120_:%.+]] = math.tanh [[VAR_103_]] : f32
// CHECK:               [[VAR_121_:%.+]] = arith.mulf [[VAR_119_]], [[VAR_120_]] : f32
// CHECK:               krnl.store [[VAR_103_]], [[RES_2_]]{{.}}[[VAR_59_1_]]#0, [[VAR_59_1_]]#1] : memref<2x4xf32>
// CHECK:               krnl.store [[VAR_121_]], [[RES_1_]]{{.}}[[VAR_59_1_]]#0, [[VAR_59_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_4_]]) with ([[LOOP_4_]] -> [[I_7_:%.+]] = 0 to 7){
// CHECK:             [[VAR_50_2_:%.+]] = krnl.get_induction_var_value([[LOOP_4_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOOP_2_:%.+]] = affine.apply [[MAP_3_]]([[VAR_50_2_]])
// CHECK-DAG:         [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_5_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_5_]]#0, [[LOOP_5_]]#1) with ([[LOOP_5_]]#0 -> [[I_8_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_5_]]#1 -> [[I_9_:%.+]] = [[CST_0_]] to [[CST_3_]]){
// CHECK:               [[LOAD_PARAM_0_MEM_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_5_]]#0, [[LOOP_5_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[LOOP_2_]], [[LOAD_PARAM_0_MEM_1_]]#0, [[LOAD_PARAM_0_MEM_1_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_2_]], [[RES_6_]]{{.}}[[LOAD_PARAM_0_MEM_1_]]#0, [[LOAD_PARAM_0_MEM_1_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             [[LOAD_PARAM_4_MEM_1_1_:%.+]] = builtin.unrealized_conversion_cast [[RES_6_]] : memref<2x3xf32> to tensor<2x3xf32>
// CHECK:             [[LOAD_PARAM_5_MEM_1_1_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_4_MEM_1_1_]], [[VAR_13_]]) : (tensor<2x3xf32>, tensor<3x16xf32>) -> tensor<2x16xf32>
// CHECK-DAG:         [[VAR_55_1_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_5_MEM_1_1_]] : tensor<2x16xf32> to memref<2x16xf32>
// CHECK-DAG:         [[VAR_56_1_:%.+]] = builtin.unrealized_conversion_cast [[RES_3_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_57_1_:%.+]] = "onnx.MatMul"([[VAR_56_1_]], [[VAR_14_]]) : (tensor<2x4xf32>, tensor<4x16xf32>) -> tensor<2x16xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]] = builtin.unrealized_conversion_cast [[VAR_57_1_]] : tensor<2x16xf32> to memref<2x16xf32>
// CHECK-DAG:         [[LOOP_6_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_6_]]#0, [[LOOP_6_]]#1) with ([[LOOP_6_]]#0 -> [[I_10_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_6_]]#1 -> [[I_11_:%.+]] = [[CST_0_]] to [[CST_4_]]){
// CHECK:               [[LOAD_PARAM_0_MEM_1_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_6_]]#0, [[LOOP_6_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[RES_4_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_57_MEM_4_:%.+]] = krnl.load [[VAR_55_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x16xf32>
// CHECK-DAG:           [[VAR_63_1_:%.+]] = krnl.load [[LOOP_3_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_19_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_57_MEM_4_]], [[VAR_63_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_23_MEM_1_:%.+]] = krnl.load [[VAR_28_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_66_1_:%.+]] = krnl.load [[VAR_32_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_67_1_:%.+]] = arith.addf [[LOAD_VAR_19_MEM_1_]], [[LOAD_VAR_23_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_40_MEM_1_:%.+]] = arith.addf [[VAR_67_1_]], [[VAR_66_1_]] : f32
// CHECK-DAG:           [[VAR_69_1_:%.+]] = krnl.load [[VAR_44_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_70_1_:%.+]] = arith.mulf [[VAR_69_1_]], [[LOAD_PARAM_0_MEM_2_]] : f32
// CHECK:               [[VAR_71_1_:%.+]] = arith.addf [[LOAD_VAR_40_MEM_1_]], [[VAR_70_1_]] : f32
// CHECK:               [[VAR_72_1_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_71_1_]] : f32
// CHECK:               [[VAR_73_1_:%.+]] = math.exp [[VAR_72_1_]] : f32
// CHECK:               [[VAR_74_1_:%.+]] = arith.addf [[VAR_73_1_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:           [[VAR_75_1_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_74_1_]] : f32
// CHECK-DAG:           [[LOAD_LOAD_PARAM_5_MEM_1_MEM_1_:%.+]] = affine.apply [[MAP_0_]]([[LOAD_PARAM_0_MEM_1_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_77_1_:%.+]] = krnl.load [[VAR_55_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_LOAD_PARAM_5_MEM_1_MEM_1_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_VAR_57_MEM_1_:%.+]] = affine.apply [[MAP_0_]]([[LOAD_PARAM_0_MEM_1_1_]]#1)
// CHECK:               [[VAR_79_1_:%.+]] = krnl.load [[LOOP_3_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_VAR_57_MEM_1_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_VAR_21_MEM_1_:%.+]] = arith.addf [[VAR_77_1_]], [[VAR_79_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_25_MEM_1_:%.+]] = krnl.load [[VAR_30_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_82_1_:%.+]] = krnl.load [[VAR_34_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_83_1_:%.+]] = arith.addf [[LOAD_VAR_21_MEM_1_]], [[LOAD_VAR_25_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_42_MEM_1_:%.+]] = arith.addf [[VAR_83_1_]], [[VAR_82_1_]] : f32
// CHECK-DAG:           [[VAR_85_1_:%.+]] = krnl.load [[VAR_46_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_86_1_:%.+]] = arith.mulf [[VAR_85_1_]], [[LOAD_PARAM_0_MEM_2_]] : f32
// CHECK:               [[VAR_87_1_:%.+]] = arith.addf [[LOAD_VAR_42_MEM_1_]], [[VAR_86_1_]] : f32
// CHECK:               [[VAR_88_1_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_87_1_]] : f32
// CHECK:               [[VAR_89_1_:%.+]] = math.exp [[VAR_88_1_]] : f32
// CHECK:               [[VAR_90_1_:%.+]] = arith.addf [[VAR_89_1_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:           [[VAR_91_1_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_90_1_]] : f32
// CHECK-DAG:           [[LOAD_LOAD_PARAM_5_MEM_1_MEM_2_:%.+]] = affine.apply [[MAP_1_]]([[LOAD_PARAM_0_MEM_1_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_93_1_:%.+]] = krnl.load [[VAR_55_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_LOAD_PARAM_5_MEM_1_MEM_2_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_VAR_57_MEM_2_:%.+]] = affine.apply [[MAP_1_]]([[LOAD_PARAM_0_MEM_1_1_]]#1)
// CHECK:               [[VAR_95_1_:%.+]] = krnl.load [[LOOP_3_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_VAR_57_MEM_2_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_VAR_22_MEM_1_:%.+]] = arith.addf [[VAR_93_1_]], [[VAR_95_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_26_MEM_1_:%.+]] = krnl.load [[VAR_31_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_98_1_:%.+]] = krnl.load [[VAR_35_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_99_1_:%.+]] = arith.addf [[LOAD_VAR_22_MEM_1_]], [[LOAD_VAR_26_MEM_1_]] : f32
// CHECK:               [[VAR_100_1_:%.+]] = arith.addf [[VAR_99_1_]], [[VAR_98_1_]] : f32
// CHECK-DAG:           [[VAR_101_1_:%.+]] = math.tanh [[VAR_100_1_]] : f32
// CHECK-DAG:           [[VAR_102_1_:%.+]] = arith.mulf [[VAR_91_1_]], [[LOAD_PARAM_0_MEM_2_]] : f32
// CHECK:               [[VAR_103_1_:%.+]] = arith.mulf [[VAR_75_1_]], [[VAR_101_1_]] : f32
// CHECK-DAG:           [[VAR_104_1_:%.+]] = arith.addf [[VAR_102_1_]], [[VAR_103_1_]] : f32
// CHECK-DAG:           [[LOAD_LOAD_PARAM_5_MEM_1_MEM_3_:%.+]] = affine.apply [[MAP_2_]]([[LOAD_PARAM_0_MEM_1_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_106_1_:%.+]] = krnl.load [[VAR_55_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_LOAD_PARAM_5_MEM_1_MEM_3_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_VAR_57_MEM_3_:%.+]] = affine.apply [[MAP_2_]]([[LOAD_PARAM_0_MEM_1_1_]]#1)
// CHECK:               [[VAR_108_1_:%.+]] = krnl.load [[LOOP_3_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_VAR_57_MEM_3_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_VAR_20_MEM_1_:%.+]] = arith.addf [[VAR_106_1_]], [[VAR_108_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_24_MEM_1_:%.+]] = krnl.load [[VAR_29_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_111_1_:%.+]] = krnl.load [[VAR_33_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_112_1_:%.+]] = arith.addf [[LOAD_VAR_20_MEM_1_]], [[LOAD_VAR_24_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_41_MEM_1_:%.+]] = arith.addf [[VAR_112_1_]], [[VAR_111_1_]] : f32
// CHECK-DAG:           [[VAR_114_1_:%.+]] = krnl.load [[VAR_45_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_115_1_:%.+]] = arith.mulf [[VAR_114_1_]], [[VAR_104_1_]] : f32
// CHECK:               [[VAR_116_1_:%.+]] = arith.addf [[LOAD_VAR_41_MEM_1_]], [[VAR_115_1_]] : f32
// CHECK:               [[VAR_117_1_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_116_1_]] : f32
// CHECK:               [[VAR_118_1_:%.+]] = math.exp [[VAR_117_1_]] : f32
// CHECK:               [[VAR_119_1_:%.+]] = arith.addf [[VAR_118_1_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:           [[VAR_120_1_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_119_1_]] : f32
// CHECK-DAG:           [[VAR_121_1_:%.+]] = math.tanh [[VAR_104_1_]] : f32
// CHECK:               [[VAR_122_:%.+]] = arith.mulf [[VAR_120_1_]], [[VAR_121_1_]] : f32
// CHECK:               krnl.store [[VAR_104_1_]], [[RES_4_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK:               krnl.store [[VAR_122_]], [[RES_3_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOOP_7_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_7_]]#0, [[LOOP_7_]]#1) with ([[LOOP_7_]]#0 -> [[I_12_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_7_]]#1 -> [[I_13_:%.+]] = [[CST_0_]] to [[CST_4_]]){
// CHECK:             [[VAR_50_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_7_]]#0, [[LOOP_7_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOOP_2_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_50_3_]]#0, [[VAR_50_3_]]#1] : memref<2x4xf32>
// CHECK:             krnl.store [[LOOP_2_1_]], [[RES_]]{{.}}[[CST_0_]], [[VAR_50_3_]]#0, [[VAR_50_3_]]#1] : memref<2x2x4xf32>
// CHECK:             [[LOOP_5_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_50_3_]]#0, [[VAR_50_3_]]#1] : memref<2x4xf32>
// CHECK:             krnl.store [[LOOP_5_]], [[RES_]]{{.}}[[CST_1_]], [[VAR_50_3_]]#0, [[VAR_50_3_]]#1] : memref<2x2x4xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x2x4xf32>
// CHECK:         }
}

// -----

func.func private @test_lstm_unknown_dims(%arg0: tensor<?x?x?xf32>, %arg1: tensor<1x16x?xf32>, %arg2: tensor<1x16x4xf32>, %arg3: tensor<1x32xf32>, %arg4: tensor<1x?x4xf32>, %arg5: tensor<1x?x4xf32>, %arg6: tensor<1x12xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4, %arg5, %arg6) {hidden_size = 4 : si64} : (tensor<?x?x?xf32>, tensor<1x16x?xf32>, tensor<1x16x4xf32>, tensor<1x32xf32>, none, tensor<1x?x4xf32>, tensor<1x?x4xf32>, tensor<1x12xf32>) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>
// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 8)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 + 12)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 + 4)>
// CHECK-LABEL:  func.func private @test_lstm_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x?xf32>, [[PARAM_1_:%.+]]: memref<1x16x?xf32>, [[PARAM_2_:%.+]]: memref<1x16x4xf32>, [[PARAM_3_:%.+]]: memref<1x32xf32>, [[PARAM_4_:%.+]]: memref<1x?x4xf32>, [[PARAM_5_:%.+]]: memref<1x?x4xf32>, [[PARAM_6_:%.+]]: memref<1x12xf32>) -> memref<1x?x4xf32> {
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : i64
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_4_1_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_6_]] : memref<1x12xf32> to tensor<1x12xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_3_]] : memref<1x32xf32> to tensor<1x32xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_2_]] : memref<1x16x4xf32> to tensor<1x16x4xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_1_]] : memref<1x16x?xf32> to tensor<1x16x?xf32>
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<1x?x4xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[VAR_dim_1_]]) {{.*}}: memref<?x4xf32>
// CHECK-DAG:       [[VAR_dim_3_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc([[VAR_dim_3_]]) {{.*}}: memref<?x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_1_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[VAR_27_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_]], [[VAR_27_]]#0, [[VAR_27_]]#1] : memref<1x?x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]]{{.}}[[VAR_27_]]#0, [[VAR_27_]]#1] : memref<?x4xf32>
// CHECK:             [[LOAD_PARAM_5_MEM_:%.+]] = krnl.load [[PARAM_5_]]{{.}}[[CST_0_]], [[VAR_27_]]#0, [[VAR_27_]]#1] : memref<1x?x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_5_MEM_]], [[RES_2_]]{{.}}[[VAR_27_]]#0, [[VAR_27_]]#1] : memref<?x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.SqueezeV11"([[VAR_3_]]) <{axes = [0]}> : (tensor<1x16x?xf32>) -> tensor<16x?xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.SqueezeV11"([[VAR_2_]]) <{axes = [0]}> : (tensor<1x16x4xf32>) -> tensor<16x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Transpose"([[VAR_5_]]) <{perm = [1, 0]}> : (tensor<16x?xf32>) -> tensor<?x16xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Transpose"([[VAR_6_]]) <{perm = [1, 0]}> : (tensor<16x4xf32>) -> tensor<4x16xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.SqueezeV11"([[VAR_1_]]) <{axes = [0]}> : (tensor<1x32xf32>) -> tensor<32xf32>
// CHECK:           [[VAR_10_:%.+]]:8 = "onnx.SplitV11"([[VAR_9_]]) <{axis = 0 : si64}> : (tensor<32xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
// CHECK-DAG:       [[VAR_11_:%.+]] = builtin.unrealized_conversion_cast [[VAR_10_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = builtin.unrealized_conversion_cast [[VAR_10_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = builtin.unrealized_conversion_cast [[VAR_10_]]#2 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = builtin.unrealized_conversion_cast [[VAR_10_]]#3 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = builtin.unrealized_conversion_cast [[VAR_10_]]#4 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = builtin.unrealized_conversion_cast [[VAR_10_]]#5 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = builtin.unrealized_conversion_cast [[VAR_10_]]#6 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = builtin.unrealized_conversion_cast [[VAR_10_]]#7 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.SqueezeV11"([[VAR_0_]]) <{axes = [0]}> : (tensor<1x12xf32>) -> tensor<12xf32>
// CHECK:           [[VAR_20_:%.+]]:3 = "onnx.SplitV11"([[VAR_19_]]) <{axis = 0 : si64}> : (tensor<12xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
// CHECK-DAG:       [[VAR_21_:%.+]] = builtin.unrealized_conversion_cast [[VAR_20_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = builtin.unrealized_conversion_cast [[VAR_20_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_23_:%.+]] = builtin.unrealized_conversion_cast [[VAR_20_]]#2 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_dim_5_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_5_]])){
// CHECK-DAG:         [[VAR_27_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[VAR_dim_6_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x?xf32>
// CHECK-DAG:         [[VAR_dim_7_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<?x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc([[VAR_dim_6_]], [[VAR_dim_7_]]) {{.*}}: memref<?x?xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_]] to [[VAR_dim_6_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_]] to [[VAR_dim_7_]]){
// CHECK:               [[VAR_36_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_27_1_]], [[VAR_36_]]#0, [[VAR_36_]]#1] : memref<?x?x?xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_3_]]{{.}}[[VAR_36_]]#0, [[VAR_36_]]#1] : memref<?x?xf32>
// CHECK:             }
// CHECK:             [[LOAD_PARAM_5_MEM_1_:%.+]] = builtin.unrealized_conversion_cast [[RES_3_]] : memref<?x?xf32> to tensor<?x?xf32>
// CHECK:             [[VAR_30_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_5_MEM_1_]], [[VAR_7_]]) : (tensor<?x?xf32>, tensor<?x16xf32>) -> tensor<?x16xf32>
// CHECK-DAG:         [[VAR_31_:%.+]] = builtin.unrealized_conversion_cast [[VAR_30_]] : tensor<?x16xf32> to memref<?x16xf32>
// CHECK-DAG:         [[VAR_32_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<?x4xf32> to tensor<?x4xf32>
// CHECK:             [[VAR_33_:%.+]] = "onnx.MatMul"([[VAR_32_]], [[VAR_8_]]) : (tensor<?x4xf32>, tensor<4x16xf32>) -> tensor<?x16xf32>
// CHECK-DAG:         [[VAR_34_:%.+]] = builtin.unrealized_conversion_cast [[VAR_33_]] : tensor<?x16xf32> to memref<?x16xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_]] to [[VAR_dim_1_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_]] to [[CST_4_1_]]){
// CHECK:               [[VAR_36_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_36_1_]]#0, [[VAR_36_1_]]#1] : memref<?x4xf32>
// CHECK-DAG:           [[LOAD_VAR_31_MEM_:%.+]] = krnl.load [[VAR_31_]]{{.}}[[VAR_36_1_]]#0, [[VAR_36_1_]]#1] : memref<?x16xf32>
// CHECK-DAG:           [[LOAD_VAR_34_MEM_:%.+]] = krnl.load [[VAR_34_]]{{.}}[[VAR_36_1_]]#0, [[VAR_36_1_]]#1] : memref<?x16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_40_:%.+]] = arith.addf [[LOAD_VAR_31_MEM_]], [[LOAD_VAR_34_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_:%.+]] = krnl.load [[VAR_11_]]{{.}}[[VAR_36_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_15_MEM_:%.+]] = krnl.load [[VAR_15_]]{{.}}[[VAR_36_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_43_:%.+]] = arith.addf [[VAR_40_]], [[LOAD_VAR_11_MEM_]] : f32
// CHECK-DAG:           [[VAR_44_:%.+]] = arith.addf [[VAR_43_]], [[LOAD_VAR_15_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_21_MEM_:%.+]] = krnl.load [[VAR_21_]]{{.}}[[VAR_36_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_46_:%.+]] = arith.mulf [[LOAD_VAR_21_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_47_:%.+]] = arith.addf [[VAR_44_]], [[VAR_46_]] : f32
// CHECK:               [[VAR_48_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_47_]] : f32
// CHECK:               [[VAR_49_:%.+]] = math.exp [[VAR_48_]] : f32
// CHECK:               [[VAR_50_:%.+]] = arith.addf [[VAR_49_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:           [[VAR_51_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_50_]] : f32
// CHECK-DAG:           [[VAR_52_:%.+]] = affine.apply [[MAP_1_]]([[VAR_36_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_31_MEM_1_:%.+]] = krnl.load [[VAR_31_]]{{.}}[[VAR_36_1_]]#0, [[VAR_52_]]{{.}} : memref<?x16xf32>
// CHECK-DAG:           [[VAR_54_:%.+]] = affine.apply [[MAP_1_]]([[VAR_36_1_]]#1)
// CHECK:               [[LOAD_VAR_34_MEM_1_:%.+]] = krnl.load [[VAR_34_]]{{.}}[[VAR_36_1_]]#0, [[VAR_54_]]{{.}} : memref<?x16xf32>
// CHECK-DAG:           [[VAR_56_:%.+]] = arith.addf [[LOAD_VAR_31_MEM_1_]], [[LOAD_VAR_34_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_13_MEM_:%.+]] = krnl.load [[VAR_13_]]{{.}}[[VAR_36_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_17_MEM_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_36_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_59_:%.+]] = arith.addf [[VAR_56_]], [[LOAD_VAR_13_MEM_]] : f32
// CHECK-DAG:           [[VAR_60_:%.+]] = arith.addf [[VAR_59_]], [[LOAD_VAR_17_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_23_MEM_:%.+]] = krnl.load [[VAR_23_]]{{.}}[[VAR_36_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_62_:%.+]] = arith.mulf [[LOAD_VAR_23_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_63_:%.+]] = arith.addf [[VAR_60_]], [[VAR_62_]] : f32
// CHECK:               [[VAR_64_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_63_]] : f32
// CHECK:               [[VAR_65_:%.+]] = math.exp [[VAR_64_]] : f32
// CHECK:               [[VAR_66_:%.+]] = arith.addf [[VAR_65_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:           [[VAR_67_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_66_]] : f32
// CHECK-DAG:           [[VAR_68_:%.+]] = affine.apply [[MAP_2_]]([[VAR_36_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_31_MEM_2_:%.+]] = krnl.load [[VAR_31_]]{{.}}[[VAR_36_1_]]#0, [[VAR_68_]]{{.}} : memref<?x16xf32>
// CHECK-DAG:           [[VAR_70_:%.+]] = affine.apply [[MAP_2_]]([[VAR_36_1_]]#1)
// CHECK:               [[LOAD_VAR_34_MEM_2_:%.+]] = krnl.load [[VAR_34_]]{{.}}[[VAR_36_1_]]#0, [[VAR_70_]]{{.}} : memref<?x16xf32>
// CHECK-DAG:           [[VAR_72_:%.+]] = arith.addf [[LOAD_VAR_31_MEM_2_]], [[LOAD_VAR_34_MEM_2_]] : f32
// CHECK-DAG:           [[LOAD_VAR_14_MEM_:%.+]] = krnl.load [[VAR_14_]]{{.}}[[VAR_36_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_18_MEM_:%.+]] = krnl.load [[VAR_18_]]{{.}}[[VAR_36_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_75_:%.+]] = arith.addf [[VAR_72_]], [[LOAD_VAR_14_MEM_]] : f32
// CHECK:               [[VAR_76_:%.+]] = arith.addf [[VAR_75_]], [[LOAD_VAR_18_MEM_]] : f32
// CHECK-DAG:           [[VAR_77_:%.+]] = math.tanh [[VAR_76_]] : f32
// CHECK-DAG:           [[VAR_78_:%.+]] = arith.mulf [[VAR_67_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_79_:%.+]] = arith.mulf [[VAR_51_]], [[VAR_77_]] : f32
// CHECK-DAG:           [[VAR_80_:%.+]] = arith.addf [[VAR_78_]], [[VAR_79_]] : f32
// CHECK-DAG:           [[VAR_81_:%.+]] = affine.apply [[MAP_3_]]([[VAR_36_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_31_MEM_3_:%.+]] = krnl.load [[VAR_31_]]{{.}}[[VAR_36_1_]]#0, [[VAR_81_]]{{.}} : memref<?x16xf32>
// CHECK-DAG:           [[VAR_83_:%.+]] = affine.apply [[MAP_3_]]([[VAR_36_1_]]#1)
// CHECK:               [[LOAD_VAR_34_MEM_3_:%.+]] = krnl.load [[VAR_34_]]{{.}}[[VAR_36_1_]]#0, [[VAR_83_]]{{.}} : memref<?x16xf32>
// CHECK-DAG:           [[VAR_85_:%.+]] = arith.addf [[LOAD_VAR_31_MEM_3_]], [[LOAD_VAR_34_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_12_MEM_:%.+]] = krnl.load [[VAR_12_]]{{.}}[[VAR_36_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_16_MEM_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_36_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_88_:%.+]] = arith.addf [[VAR_85_]], [[LOAD_VAR_12_MEM_]] : f32
// CHECK-DAG:           [[VAR_89_:%.+]] = arith.addf [[VAR_88_]], [[LOAD_VAR_16_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_22_MEM_:%.+]] = krnl.load [[VAR_22_]]{{.}}[[VAR_36_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_91_:%.+]] = arith.mulf [[LOAD_VAR_22_MEM_]], [[VAR_80_]] : f32
// CHECK:               [[VAR_92_:%.+]] = arith.addf [[VAR_89_]], [[VAR_91_]] : f32
// CHECK:               [[VAR_93_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_92_]] : f32
// CHECK:               [[VAR_94_:%.+]] = math.exp [[VAR_93_]] : f32
// CHECK:               [[VAR_95_:%.+]] = arith.addf [[VAR_94_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:           [[VAR_96_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_95_]] : f32
// CHECK-DAG:           [[VAR_97_:%.+]] = math.tanh [[VAR_80_]] : f32
// CHECK:               [[VAR_98_:%.+]] = arith.mulf [[VAR_96_]], [[VAR_97_]] : f32
// CHECK:               krnl.store [[VAR_80_]], [[RES_2_]]{{.}}[[VAR_36_1_]]#0, [[VAR_36_1_]]#1] : memref<?x4xf32>
// CHECK:               krnl.store [[VAR_98_]], [[RES_1_]]{{.}}[[VAR_36_1_]]#0, [[VAR_36_1_]]#1] : memref<?x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           [[VAR_25_:%.+]] = arith.index_cast [[VAR_dim_1_]] : index to i64
// CHECK:           [[VAR_26_:%.+]] = arith.muli [[VAR_25_]], [[CST_4_]] : i64
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_]]_2, [[VAR_26_]], [[CST_0_]], [[CST_0_]]) : (memref<1x?x4xf32>, memref<?x4xf32>, i64, index, index) -> ()
// CHECK:           return [[RES_]] : memref<1x?x4xf32>
// CHECK:         }
}
