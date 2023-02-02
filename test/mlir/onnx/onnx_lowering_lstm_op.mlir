// RUN: onnx-mlir-opt -O3 --shape-inference --convert-onnx-to-krnl='emit-intermediate-ir' --canonicalize %s -split-input-file | FileCheck %s

func.func private @test_lstm_forward_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x16x3xf32>, %arg2: tensor<1x16x4xf32>, %arg3: tensor<1x32xf32>, %arg4: tensor<1x2x4xf32>, %arg5: tensor<1x2x4xf32>, %arg6: tensor<1x12xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4, %arg5, %arg6) {hidden_size = 4 : si64} : (tensor<7x2x3xf32>, tensor<1x16x3xf32>, tensor<1x16x4xf32>, tensor<1x32xf32>, none, tensor<1x2x4xf32>, tensor<1x2x4xf32>, tensor<1x12xf32>) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>
// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 + 8)>
// CHECK-DAG: [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 + 12)>
// CHECK-DAG: [[MAP_2_:#.+]] = affine_map<()[s0] -> (s0 + 4)>
// CHECK-LABEL:  func private @test_lstm_forward_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x16x3xf32>, [[PARAM_2_:%.+]]: memref<1x16x4xf32>, [[PARAM_3_:%.+]]: memref<1x32xf32>, [[PARAM_4_:%.+]]: memref<1x2x4xf32>, [[PARAM_5_:%.+]]: memref<1x2x4xf32>, [[PARAM_6_:%.+]]: memref<1x12xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[VAR_c8_i64_:%.+]] = arith.constant 8 : i64
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_6_]] : memref<1x12xf32> to tensor<1x12xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_3_]] : memref<1x32xf32> to tensor<1x32xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_2_]] : memref<1x16x4xf32> to tensor<1x16x4xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_1_]] : memref<1x16x3xf32> to tensor<1x16x3xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[VAR_28_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[VAR_c0_]], [[VAR_28_]]#0, [[VAR_28_]]#1] : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]]{{.}}[[VAR_28_]]#0, [[VAR_28_]]#1] : memref<2x4xf32>
// CHECK:             [[LOAD_PARAM_5_MEM_:%.+]] = krnl.load [[PARAM_5_]]{{.}}[[VAR_c0_]], [[VAR_28_]]#0, [[VAR_28_]]#1] : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_5_MEM_]], [[RES_2_]]{{.}}[[VAR_28_]]#0, [[VAR_28_]]#1] : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.SqueezeV11"([[VAR_3_]]) {axes = [0]} : (tensor<1x16x3xf32>) -> tensor<16x3xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.SqueezeV11"([[VAR_2_]]) {axes = [0]} : (tensor<1x16x4xf32>) -> tensor<16x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Transpose"([[VAR_8_]]) {perm = [1, 0]} : (tensor<16x3xf32>) -> tensor<3x16xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Transpose"([[VAR_9_]]) {perm = [1, 0]} : (tensor<16x4xf32>) -> tensor<4x16xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.SqueezeV11"([[VAR_1_]]) {axes = [0]} : (tensor<1x32xf32>) -> tensor<32xf32>
// CHECK:           [[VAR_13_:%.+]]:8 = "onnx.SplitV11"([[VAR_12_]]) {axis = 0 : si64} : (tensor<32xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
// CHECK-DAG:       [[VAR_14_:%.+]] = builtin.unrealized_conversion_cast [[VAR_13_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = builtin.unrealized_conversion_cast [[VAR_13_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = builtin.unrealized_conversion_cast [[VAR_13_]]#2 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = builtin.unrealized_conversion_cast [[VAR_13_]]#3 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = builtin.unrealized_conversion_cast [[VAR_13_]]#4 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = builtin.unrealized_conversion_cast [[VAR_13_]]#5 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = builtin.unrealized_conversion_cast [[VAR_13_]]#6 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = builtin.unrealized_conversion_cast [[VAR_13_]]#7 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = "onnx.SqueezeV11"([[VAR_0_]]) {axes = [0]} : (tensor<1x12xf32>) -> tensor<12xf32>
// CHECK:           [[VAR_23_:%.+]]:3 = "onnx.SplitV11"([[VAR_22_]]) {axis = 0 : si64} : (tensor<12xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
// CHECK-DAG:       [[VAR_24_:%.+]] = builtin.unrealized_conversion_cast [[VAR_23_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_25_:%.+]] = builtin.unrealized_conversion_cast [[VAR_23_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_26_:%.+]] = builtin.unrealized_conversion_cast [[VAR_23_]]#2 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7){
// CHECK-DAG:         [[VAR_28_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]){
// CHECK:               [[VAR_38_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_28_1_]], [[VAR_38_]]#0, [[VAR_38_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_3_]]{{.}}[[VAR_38_]]#0, [[VAR_38_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             [[VAR_31_:%.+]] = builtin.unrealized_conversion_cast [[RES_3_]] : memref<2x3xf32> to tensor<2x3xf32>
// CHECK:             [[VAR_32_:%.+]] = "onnx.MatMul"([[VAR_31_]], [[VAR_10_]]) : (tensor<2x3xf32>, tensor<3x16xf32>) -> tensor<2x16xf32>
// CHECK-DAG:         [[VAR_33_:%.+]] = builtin.unrealized_conversion_cast [[VAR_32_]] : tensor<2x16xf32> to memref<2x16xf32>
// CHECK-DAG:         [[VAR_34_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_35_:%.+]] = "onnx.MatMul"([[VAR_34_]], [[VAR_11_]]) : (tensor<2x4xf32>, tensor<4x16xf32>) -> tensor<2x16xf32>
// CHECK-DAG:         [[VAR_36_:%.+]] = builtin.unrealized_conversion_cast [[VAR_35_]] : tensor<2x16xf32> to memref<2x16xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[VAR_38_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_38_1_]]#0, [[VAR_38_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_33_MEM_:%.+]] = krnl.load [[VAR_33_]]{{.}}[[VAR_38_1_]]#0, [[VAR_38_1_]]#1] : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_VAR_36_MEM_:%.+]] = krnl.load [[VAR_36_]]{{.}}[[VAR_38_1_]]#0, [[VAR_38_1_]]#1] : memref<2x16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_42_:%.+]] = arith.addf [[LOAD_VAR_33_MEM_]], [[LOAD_VAR_36_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_14_MEM_:%.+]] = krnl.load [[VAR_14_]]{{.}}[[VAR_38_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_18_MEM_:%.+]] = krnl.load [[VAR_18_]]{{.}}[[VAR_38_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_45_:%.+]] = arith.addf [[VAR_42_]], [[LOAD_VAR_14_MEM_]] : f32
// CHECK-DAG:           [[VAR_46_:%.+]] = arith.addf [[VAR_45_]], [[LOAD_VAR_18_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_24_MEM_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_38_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_48_:%.+]] = arith.mulf [[LOAD_VAR_24_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_49_:%.+]] = arith.addf [[VAR_46_]], [[VAR_48_]] : f32
// CHECK:               [[VAR_50_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_49_]] : f32
// CHECK:               [[VAR_51_:%.+]] = math.exp [[VAR_50_]] : f32
// CHECK:               [[VAR_52_:%.+]] = arith.addf [[VAR_51_]], [[VAR_cst_]] : f32
// CHECK-DAG:           [[VAR_53_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_52_]] : f32
// CHECK-DAG:           [[VAR_54_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_38_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_33_MEM_1_:%.+]] = krnl.load [[VAR_33_]]{{.}}[[VAR_38_1_]]#0, [[VAR_54_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_56_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_38_1_]]#1]
// CHECK:               [[LOAD_VAR_36_MEM_1_:%.+]] = krnl.load [[VAR_36_]]{{.}}[[VAR_38_1_]]#0, [[VAR_56_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_58_:%.+]] = arith.addf [[LOAD_VAR_33_MEM_1_]], [[LOAD_VAR_36_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_16_MEM_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_38_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_20_MEM_:%.+]] = krnl.load [[VAR_20_]]{{.}}[[VAR_38_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_61_:%.+]] = arith.addf [[VAR_58_]], [[LOAD_VAR_16_MEM_]] : f32
// CHECK-DAG:           [[VAR_62_:%.+]] = arith.addf [[VAR_61_]], [[LOAD_VAR_20_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_26_MEM_:%.+]] = krnl.load [[VAR_26_]]{{.}}[[VAR_38_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_64_:%.+]] = arith.mulf [[LOAD_VAR_26_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_65_:%.+]] = arith.addf [[VAR_62_]], [[VAR_64_]] : f32
// CHECK:               [[VAR_66_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_65_]] : f32
// CHECK:               [[VAR_67_:%.+]] = math.exp [[VAR_66_]] : f32
// CHECK:               [[VAR_68_:%.+]] = arith.addf [[VAR_67_]], [[VAR_cst_]] : f32
// CHECK-DAG:           [[VAR_69_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_68_]] : f32
// CHECK-DAG:           [[VAR_70_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_38_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_33_MEM_2_:%.+]] = krnl.load [[VAR_33_]]{{.}}[[VAR_38_1_]]#0, [[VAR_70_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_72_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_38_1_]]#1]
// CHECK:               [[LOAD_VAR_36_MEM_2_:%.+]] = krnl.load [[VAR_36_]]{{.}}[[VAR_38_1_]]#0, [[VAR_72_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_74_:%.+]] = arith.addf [[LOAD_VAR_33_MEM_2_]], [[LOAD_VAR_36_MEM_2_]] : f32
// CHECK-DAG:           [[LOAD_VAR_17_MEM_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_38_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_21_MEM_:%.+]] = krnl.load [[VAR_21_]]{{.}}[[VAR_38_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_77_:%.+]] = arith.addf [[VAR_74_]], [[LOAD_VAR_17_MEM_]] : f32
// CHECK:               [[VAR_78_:%.+]] = arith.addf [[VAR_77_]], [[LOAD_VAR_21_MEM_]] : f32
// CHECK-DAG:           [[VAR_79_:%.+]] = math.tanh [[VAR_78_]] : f32
// CHECK-DAG:           [[VAR_80_:%.+]] = arith.mulf [[VAR_69_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_81_:%.+]] = arith.mulf [[VAR_53_]], [[VAR_79_]] : f32
// CHECK-DAG:           [[VAR_82_:%.+]] = arith.addf [[VAR_80_]], [[VAR_81_]] : f32
// CHECK-DAG:           [[VAR_83_:%.+]] = affine.apply [[MAP_2_]](){{.}}[[VAR_38_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_33_MEM_3_:%.+]] = krnl.load [[VAR_33_]]{{.}}[[VAR_38_1_]]#0, [[VAR_83_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_85_:%.+]] = affine.apply [[MAP_2_]](){{.}}[[VAR_38_1_]]#1]
// CHECK:               [[LOAD_VAR_36_MEM_3_:%.+]] = krnl.load [[VAR_36_]]{{.}}[[VAR_38_1_]]#0, [[VAR_85_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_87_:%.+]] = arith.addf [[LOAD_VAR_33_MEM_3_]], [[LOAD_VAR_36_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_15_MEM_:%.+]] = krnl.load [[VAR_15_]]{{.}}[[VAR_38_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_19_MEM_:%.+]] = krnl.load [[VAR_19_]]{{.}}[[VAR_38_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_90_:%.+]] = arith.addf [[VAR_87_]], [[LOAD_VAR_15_MEM_]] : f32
// CHECK-DAG:           [[VAR_91_:%.+]] = arith.addf [[VAR_90_]], [[LOAD_VAR_19_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_25_MEM_:%.+]] = krnl.load [[VAR_25_]]{{.}}[[VAR_38_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_93_:%.+]] = arith.mulf [[LOAD_VAR_25_MEM_]], [[VAR_82_]] : f32
// CHECK:               [[VAR_94_:%.+]] = arith.addf [[VAR_91_]], [[VAR_93_]] : f32
// CHECK:               [[VAR_95_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_94_]] : f32
// CHECK:               [[VAR_96_:%.+]] = math.exp [[VAR_95_]] : f32
// CHECK:               [[VAR_97_:%.+]] = arith.addf [[VAR_96_]], [[VAR_cst_]] : f32
// CHECK-DAG:           [[VAR_98_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_97_]] : f32
// CHECK-DAG:           [[VAR_99_:%.+]] = math.tanh [[VAR_82_]] : f32
// CHECK:               [[VAR_100_:%.+]] = arith.mulf [[VAR_98_]], [[VAR_99_]] : f32
// CHECK:               krnl.store [[VAR_82_]], [[RES_2_]]{{.}}[[VAR_38_1_]]#0, [[VAR_38_1_]]#1] : memref<2x4xf32>
// CHECK:               krnl.store [[VAR_100_]], [[RES_1_]]{{.}}[[VAR_38_1_]]#0, [[VAR_38_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_1_]], [[VAR_c8_i64_]], [[VAR_c0_]], [[VAR_c0_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64, index, index) -> ()
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
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<()[s0] -> (s0 + 8)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<()[s0] -> (s0 + 12)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<()[s0] -> (s0 + 4)>
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
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [3, 16], value = dense<1.000000e+00> : tensor<3x16xf32>} : () -> memref<3x16xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [4, 16], value = dense<2.000000e+00> : tensor<4x16xf32>} : () -> memref<4x16xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [4], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [4], value = dense<[5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [4], value = dense<[9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [4], value = dense<[1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [4], value = dense<[1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [4], value = dense<[2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [4], value = dense<[2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [4], value = dense<[2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [4], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [4], value = dense<[5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [4], value = dense<[9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
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
// CHECK-DAG:           [[VAR_42_:%.+]] = affine.apply [[MAP_3_]](){{.}}[[VAR_26_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_20_MEM_1_:%.+]] = krnl.load [[VAR_20_]]{{.}}[[VAR_26_1_]]#0, [[VAR_42_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_44_:%.+]] = affine.apply [[MAP_3_]](){{.}}[[VAR_26_1_]]#1]
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
// CHECK-DAG:           [[VAR_58_:%.+]] = affine.apply [[MAP_4_]](){{.}}[[VAR_26_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_20_MEM_2_:%.+]] = krnl.load [[VAR_20_]]{{.}}[[VAR_26_1_]]#0, [[VAR_58_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_60_:%.+]] = affine.apply [[MAP_4_]](){{.}}[[VAR_26_1_]]#1]
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
// CHECK-DAG:           [[VAR_71_:%.+]] = affine.apply [[MAP_5_]](){{.}}[[VAR_26_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_20_MEM_3_:%.+]] = krnl.load [[VAR_20_]]{{.}}[[VAR_26_1_]]#0, [[VAR_71_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_73_:%.+]] = affine.apply [[MAP_5_]](){{.}}[[VAR_26_1_]]#1]
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
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_]]_1, [[CST_8_]], [[VAR_c0_]], [[VAR_c0_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64, index, index) -> ()
// CHECK:           return [[RES_]] : memref<1x2x4xf32>
// CHECK:         }
}

// -----

func.func private @test_lstm_reverse_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x16x3xf32>, %arg2: tensor<1x16x4xf32>, %arg3: tensor<1x32xf32>, %arg4: tensor<1x2x4xf32>, %arg5: tensor<1x2x4xf32>, %arg6: tensor<1x12xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4, %arg5, %arg6) {hidden_size = 4 : si64, direction = "reverse"} : (tensor<7x2x3xf32>, tensor<1x16x3xf32>, tensor<1x16x4xf32>, tensor<1x32xf32>, none, tensor<1x2x4xf32>, tensor<1x2x4xf32>, tensor<1x12xf32>) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>
// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<(d0) -> (-d0 + 6)>
// CHECK-DAG: [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 + 8)>
// CHECK-DAG: [[MAP_2_:#.+]] = affine_map<()[s0] -> (s0 + 12)>
// CHECK-DAG: [[MAP_3_:#.+]] = affine_map<()[s0] -> (s0 + 4)>
// CHECK-LABEL:  func private @test_lstm_reverse_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x16x3xf32>, [[PARAM_2_:%.+]]: memref<1x16x4xf32>, [[PARAM_3_:%.+]]: memref<1x32xf32>, [[PARAM_4_:%.+]]: memref<1x2x4xf32>, [[PARAM_5_:%.+]]: memref<1x2x4xf32>, [[PARAM_6_:%.+]]: memref<1x12xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[VAR_c8_i64_:%.+]] = arith.constant 8 : i64
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_6_]] : memref<1x12xf32> to tensor<1x12xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_3_]] : memref<1x32xf32> to tensor<1x32xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_2_]] : memref<1x16x4xf32> to tensor<1x16x4xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_1_]] : memref<1x16x3xf32> to tensor<1x16x3xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[VAR_28_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[VAR_c0_]], [[VAR_28_]]#0, [[VAR_28_]]#1] : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]]{{.}}[[VAR_28_]]#0, [[VAR_28_]]#1] : memref<2x4xf32>
// CHECK:             [[LOAD_PARAM_5_MEM_:%.+]] = krnl.load [[PARAM_5_]]{{.}}[[VAR_c0_]], [[VAR_28_]]#0, [[VAR_28_]]#1] : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_5_MEM_]], [[RES_2_]]{{.}}[[VAR_28_]]#0, [[VAR_28_]]#1] : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.SqueezeV11"([[VAR_3_]]) {axes = [0]} : (tensor<1x16x3xf32>) -> tensor<16x3xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.SqueezeV11"([[VAR_2_]]) {axes = [0]} : (tensor<1x16x4xf32>) -> tensor<16x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Transpose"([[VAR_8_]]) {perm = [1, 0]} : (tensor<16x3xf32>) -> tensor<3x16xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Transpose"([[VAR_9_]]) {perm = [1, 0]} : (tensor<16x4xf32>) -> tensor<4x16xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.SqueezeV11"([[VAR_1_]]) {axes = [0]} : (tensor<1x32xf32>) -> tensor<32xf32>
// CHECK:           [[VAR_13_:%.+]]:8 = "onnx.SplitV11"([[VAR_12_]]) {axis = 0 : si64} : (tensor<32xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
// CHECK-DAG:       [[VAR_14_:%.+]] = builtin.unrealized_conversion_cast [[VAR_13_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = builtin.unrealized_conversion_cast [[VAR_13_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = builtin.unrealized_conversion_cast [[VAR_13_]]#2 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = builtin.unrealized_conversion_cast [[VAR_13_]]#3 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = builtin.unrealized_conversion_cast [[VAR_13_]]#4 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = builtin.unrealized_conversion_cast [[VAR_13_]]#5 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = builtin.unrealized_conversion_cast [[VAR_13_]]#6 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = builtin.unrealized_conversion_cast [[VAR_13_]]#7 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = "onnx.SqueezeV11"([[VAR_0_]]) {axes = [0]} : (tensor<1x12xf32>) -> tensor<12xf32>
// CHECK:           [[VAR_23_:%.+]]:3 = "onnx.SplitV11"([[VAR_22_]]) {axis = 0 : si64} : (tensor<12xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
// CHECK-DAG:       [[VAR_24_:%.+]] = builtin.unrealized_conversion_cast [[VAR_23_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_25_:%.+]] = builtin.unrealized_conversion_cast [[VAR_23_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_26_:%.+]] = builtin.unrealized_conversion_cast [[VAR_23_]]#2 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7){
// CHECK:             [[VAR_28_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_1_:%.+]] = affine.apply [[MAP_0_]]([[VAR_28_1_]])
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]){
// CHECK:               [[VAR_39_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[LOAD_PARAM_4_MEM_1_]], [[VAR_39_]]#0, [[VAR_39_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_3_]]{{.}}[[VAR_39_]]#0, [[VAR_39_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             [[VAR_32_:%.+]] = builtin.unrealized_conversion_cast [[RES_3_]] : memref<2x3xf32> to tensor<2x3xf32>
// CHECK:             [[VAR_33_:%.+]] = "onnx.MatMul"([[VAR_32_]], [[VAR_10_]]) : (tensor<2x3xf32>, tensor<3x16xf32>) -> tensor<2x16xf32>
// CHECK-DAG:         [[VAR_34_:%.+]] = builtin.unrealized_conversion_cast [[VAR_33_]] : tensor<2x16xf32> to memref<2x16xf32>
// CHECK-DAG:         [[VAR_35_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_36_:%.+]] = "onnx.MatMul"([[VAR_35_]], [[VAR_11_]]) : (tensor<2x4xf32>, tensor<4x16xf32>) -> tensor<2x16xf32>
// CHECK-DAG:         [[VAR_37_:%.+]] = builtin.unrealized_conversion_cast [[VAR_36_]] : tensor<2x16xf32> to memref<2x16xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[VAR_39_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_39_1_]]#0, [[VAR_39_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_34_MEM_:%.+]] = krnl.load [[VAR_34_]]{{.}}[[VAR_39_1_]]#0, [[VAR_39_1_]]#1] : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_VAR_37_MEM_:%.+]] = krnl.load [[VAR_37_]]{{.}}[[VAR_39_1_]]#0, [[VAR_39_1_]]#1] : memref<2x16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_43_:%.+]] = arith.addf [[LOAD_VAR_34_MEM_]], [[LOAD_VAR_37_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_14_MEM_:%.+]] = krnl.load [[VAR_14_]]{{.}}[[VAR_39_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_18_MEM_:%.+]] = krnl.load [[VAR_18_]]{{.}}[[VAR_39_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_46_:%.+]] = arith.addf [[VAR_43_]], [[LOAD_VAR_14_MEM_]] : f32
// CHECK-DAG:           [[VAR_47_:%.+]] = arith.addf [[VAR_46_]], [[LOAD_VAR_18_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_24_MEM_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_39_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_49_:%.+]] = arith.mulf [[LOAD_VAR_24_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_50_:%.+]] = arith.addf [[VAR_47_]], [[VAR_49_]] : f32
// CHECK:               [[VAR_51_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_50_]] : f32
// CHECK:               [[VAR_52_:%.+]] = math.exp [[VAR_51_]] : f32
// CHECK:               [[VAR_53_:%.+]] = arith.addf [[VAR_52_]], [[VAR_cst_]] : f32
// CHECK-DAG:           [[VAR_54_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_53_]] : f32
// CHECK-DAG:           [[VAR_55_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_39_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_34_MEM_1_:%.+]] = krnl.load [[VAR_34_]]{{.}}[[VAR_39_1_]]#0, [[VAR_55_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_57_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_39_1_]]#1]
// CHECK:               [[LOAD_VAR_37_MEM_1_:%.+]] = krnl.load [[VAR_37_]]{{.}}[[VAR_39_1_]]#0, [[VAR_57_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_59_:%.+]] = arith.addf [[LOAD_VAR_34_MEM_1_]], [[LOAD_VAR_37_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_16_MEM_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_39_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_20_MEM_:%.+]] = krnl.load [[VAR_20_]]{{.}}[[VAR_39_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_62_:%.+]] = arith.addf [[VAR_59_]], [[LOAD_VAR_16_MEM_]] : f32
// CHECK-DAG:           [[VAR_63_:%.+]] = arith.addf [[VAR_62_]], [[LOAD_VAR_20_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_26_MEM_:%.+]] = krnl.load [[VAR_26_]]{{.}}[[VAR_39_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_65_:%.+]] = arith.mulf [[LOAD_VAR_26_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_66_:%.+]] = arith.addf [[VAR_63_]], [[VAR_65_]] : f32
// CHECK:               [[VAR_67_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_66_]] : f32
// CHECK:               [[VAR_68_:%.+]] = math.exp [[VAR_67_]] : f32
// CHECK:               [[VAR_69_:%.+]] = arith.addf [[VAR_68_]], [[VAR_cst_]] : f32
// CHECK-DAG:           [[VAR_70_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_69_]] : f32
// CHECK-DAG:           [[VAR_71_:%.+]] = affine.apply [[MAP_2_]](){{.}}[[VAR_39_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_34_MEM_2_:%.+]] = krnl.load [[VAR_34_]]{{.}}[[VAR_39_1_]]#0, [[VAR_71_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_73_:%.+]] = affine.apply [[MAP_2_]](){{.}}[[VAR_39_1_]]#1]
// CHECK:               [[LOAD_VAR_37_MEM_2_:%.+]] = krnl.load [[VAR_37_]]{{.}}[[VAR_39_1_]]#0, [[VAR_73_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_75_:%.+]] = arith.addf [[LOAD_VAR_34_MEM_2_]], [[LOAD_VAR_37_MEM_2_]] : f32
// CHECK-DAG:           [[LOAD_VAR_17_MEM_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_39_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_21_MEM_:%.+]] = krnl.load [[VAR_21_]]{{.}}[[VAR_39_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_78_:%.+]] = arith.addf [[VAR_75_]], [[LOAD_VAR_17_MEM_]] : f32
// CHECK:               [[VAR_79_:%.+]] = arith.addf [[VAR_78_]], [[LOAD_VAR_21_MEM_]] : f32
// CHECK-DAG:           [[VAR_80_:%.+]] = math.tanh [[VAR_79_]] : f32
// CHECK-DAG:           [[VAR_81_:%.+]] = arith.mulf [[VAR_70_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_82_:%.+]] = arith.mulf [[VAR_54_]], [[VAR_80_]] : f32
// CHECK-DAG:           [[VAR_83_:%.+]] = arith.addf [[VAR_81_]], [[VAR_82_]] : f32
// CHECK-DAG:           [[VAR_84_:%.+]] = affine.apply [[MAP_3_]](){{.}}[[VAR_39_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_34_MEM_3_:%.+]] = krnl.load [[VAR_34_]]{{.}}[[VAR_39_1_]]#0, [[VAR_84_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_86_:%.+]] = affine.apply [[MAP_3_]](){{.}}[[VAR_39_1_]]#1]
// CHECK:               [[LOAD_VAR_37_MEM_3_:%.+]] = krnl.load [[VAR_37_]]{{.}}[[VAR_39_1_]]#0, [[VAR_86_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_88_:%.+]] = arith.addf [[LOAD_VAR_34_MEM_3_]], [[LOAD_VAR_37_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_15_MEM_:%.+]] = krnl.load [[VAR_15_]]{{.}}[[VAR_39_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_19_MEM_:%.+]] = krnl.load [[VAR_19_]]{{.}}[[VAR_39_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_91_:%.+]] = arith.addf [[VAR_88_]], [[LOAD_VAR_15_MEM_]] : f32
// CHECK-DAG:           [[VAR_92_:%.+]] = arith.addf [[VAR_91_]], [[LOAD_VAR_19_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_25_MEM_:%.+]] = krnl.load [[VAR_25_]]{{.}}[[VAR_39_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_94_:%.+]] = arith.mulf [[LOAD_VAR_25_MEM_]], [[VAR_83_]] : f32
// CHECK:               [[VAR_95_:%.+]] = arith.addf [[VAR_92_]], [[VAR_94_]] : f32
// CHECK:               [[VAR_96_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_95_]] : f32
// CHECK:               [[VAR_97_:%.+]] = math.exp [[VAR_96_]] : f32
// CHECK:               [[VAR_98_:%.+]] = arith.addf [[VAR_97_]], [[VAR_cst_]] : f32
// CHECK-DAG:           [[VAR_99_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_98_]] : f32
// CHECK-DAG:           [[VAR_100_:%.+]] = math.tanh [[VAR_83_]] : f32
// CHECK:               [[VAR_101_:%.+]] = arith.mulf [[VAR_99_]], [[VAR_100_]] : f32
// CHECK:               krnl.store [[VAR_83_]], [[RES_2_]]{{.}}[[VAR_39_1_]]#0, [[VAR_39_1_]]#1] : memref<2x4xf32>
// CHECK:               krnl.store [[VAR_101_]], [[RES_1_]]{{.}}[[VAR_39_1_]]#0, [[VAR_39_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_1_]], [[VAR_c8_i64_]], [[VAR_c0_]], [[VAR_c0_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64, index, index) -> ()
// CHECK:           return [[RES_]] : memref<1x2x4xf32>
// CHECK:         }
}


// -----

func.func private @test_lstm_bidirectional_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<2x16x3xf32>, %arg2: tensor<2x16x4xf32>, %arg3: tensor<2x32xf32>, %arg4: tensor<2x2x4xf32>, %arg5: tensor<2x2x4xf32>, %arg6: tensor<2x12xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4, %arg5, %arg6) {hidden_size = 4 : si64, direction = "bidirectional"} : (tensor<7x2x3xf32>, tensor<2x16x3xf32>, tensor<2x16x4xf32>, tensor<2x32xf32>, none, tensor<2x2x4xf32>, tensor<2x2x4xf32>, tensor<2x12xf32>) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>
// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 + 8)>
// CHECK-DAG: [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 + 12)>
// CHECK-DAG: [[MAP_2_:#.+]] = affine_map<()[s0] -> (s0 + 4)>
// CHECK-DAG: [[MAP_3_:#.+]] = affine_map<(d0) -> (-d0 + 6)>
// CHECK-LABEL:  func private @test_lstm_bidirectional_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<2x16x3xf32>, [[PARAM_2_:%.+]]: memref<2x16x4xf32>, [[PARAM_3_:%.+]]: memref<2x32xf32>, [[PARAM_4_:%.+]]: memref<2x2x4xf32>, [[PARAM_5_:%.+]]: memref<2x2x4xf32>, [[PARAM_6_:%.+]]: memref<2x12xf32>) -> memref<2x2x4xf32> {
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
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
// CHECK:             [[VAR_55_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[VAR_c0_]], [[VAR_55_]]#0, [[VAR_55_]]#1] : memref<2x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]]{{.}}[[VAR_55_]]#0, [[VAR_55_]]#1] : memref<2x4xf32>
// CHECK:             [[LOAD_PARAM_5_MEM_:%.+]] = krnl.load [[PARAM_5_]]{{.}}[[VAR_c0_]], [[VAR_55_]]#0, [[VAR_55_]]#1] : memref<2x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_5_MEM_]], [[RES_2_]]{{.}}[[VAR_55_]]#0, [[VAR_55_]]#1] : memref<2x4xf32>
// CHECK:             [[LOAD_PARAM_4_MEM_1_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[VAR_c1_]], [[VAR_55_]]#0, [[VAR_55_]]#1] : memref<2x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_1_]], [[RES_3_]]{{.}}[[VAR_55_]]#0, [[VAR_55_]]#1] : memref<2x4xf32>
// CHECK:             [[LOAD_PARAM_5_MEM_1_:%.+]] = krnl.load [[PARAM_5_]]{{.}}[[VAR_c1_]], [[VAR_55_]]#0, [[VAR_55_]]#1] : memref<2x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_5_MEM_1_]], [[RES_4_]]{{.}}[[VAR_55_]]#0, [[VAR_55_]]#1] : memref<2x4xf32>
// CHECK:           }
// CHECK:           [[VAR_10_:%.+]]:2 = "onnx.SplitV11"([[VAR_3_]]) {axis = 0 : si64} : (tensor<2x16x3xf32>) -> (tensor<1x16x3xf32>, tensor<1x16x3xf32>)
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.SqueezeV11"([[VAR_10_]]#0) {axes = [0]} : (tensor<1x16x3xf32>) -> tensor<16x3xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.SqueezeV11"([[VAR_10_]]#1) {axes = [0]} : (tensor<1x16x3xf32>) -> tensor<16x3xf32>
// CHECK-DAG:       [[VAR_13_:%.+]]:2 = "onnx.SplitV11"([[VAR_2_]]) {axis = 0 : si64} : (tensor<2x16x4xf32>) -> (tensor<1x16x4xf32>, tensor<1x16x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.SqueezeV11"([[VAR_13_]]#0) {axes = [0]} : (tensor<1x16x4xf32>) -> tensor<16x4xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.SqueezeV11"([[VAR_13_]]#1) {axes = [0]} : (tensor<1x16x4xf32>) -> tensor<16x4xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Transpose"([[VAR_11_]]) {perm = [1, 0]} : (tensor<16x3xf32>) -> tensor<3x16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Transpose"([[VAR_14_]]) {perm = [1, 0]} : (tensor<16x4xf32>) -> tensor<4x16xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Transpose"([[VAR_12_]]) {perm = [1, 0]} : (tensor<16x3xf32>) -> tensor<3x16xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.Transpose"([[VAR_15_]]) {perm = [1, 0]} : (tensor<16x4xf32>) -> tensor<4x16xf32>
// CHECK-DAG:       [[VAR_20_:%.+]]:2 = "onnx.SplitV11"([[VAR_1_]]) {axis = 0 : si64} : (tensor<2x32xf32>) -> (tensor<1x32xf32>, tensor<1x32xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.SqueezeV11"([[VAR_20_]]#0) {axes = [0]} : (tensor<1x32xf32>) -> tensor<32xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = "onnx.SqueezeV11"([[VAR_20_]]#1) {axes = [0]} : (tensor<1x32xf32>) -> tensor<32xf32>
// CHECK:           [[VAR_23_:%.+]]:8 = "onnx.SplitV11"([[VAR_21_]]) {axis = 0 : si64} : (tensor<32xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
// CHECK-DAG:       [[VAR_24_:%.+]] = builtin.unrealized_conversion_cast [[VAR_23_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_25_:%.+]] = builtin.unrealized_conversion_cast [[VAR_23_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_26_:%.+]] = builtin.unrealized_conversion_cast [[VAR_23_]]#2 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_27_:%.+]] = builtin.unrealized_conversion_cast [[VAR_23_]]#3 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_28_:%.+]] = builtin.unrealized_conversion_cast [[VAR_23_]]#4 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_29_:%.+]] = builtin.unrealized_conversion_cast [[VAR_23_]]#5 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_30_:%.+]] = builtin.unrealized_conversion_cast [[VAR_23_]]#6 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_31_:%.+]] = builtin.unrealized_conversion_cast [[VAR_23_]]#7 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_32_:%.+]]:8 = "onnx.SplitV11"([[VAR_22_]]) {axis = 0 : si64} : (tensor<32xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_33_:%.+]] = builtin.unrealized_conversion_cast [[VAR_32_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_34_:%.+]] = builtin.unrealized_conversion_cast [[VAR_32_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_35_:%.+]] = builtin.unrealized_conversion_cast [[VAR_32_]]#2 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_36_:%.+]] = builtin.unrealized_conversion_cast [[VAR_32_]]#3 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_37_:%.+]] = builtin.unrealized_conversion_cast [[VAR_32_]]#4 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_38_:%.+]] = builtin.unrealized_conversion_cast [[VAR_32_]]#5 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_39_:%.+]] = builtin.unrealized_conversion_cast [[VAR_32_]]#6 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_40_:%.+]] = builtin.unrealized_conversion_cast [[VAR_32_]]#7 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_41_:%.+]]:2 = "onnx.SplitV11"([[VAR_0_]]) {axis = 0 : si64} : (tensor<2x12xf32>) -> (tensor<1x12xf32>, tensor<1x12xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_42_:%.+]] = "onnx.SqueezeV11"([[VAR_41_]]#0) {axes = [0]} : (tensor<1x12xf32>) -> tensor<12xf32>
// CHECK-DAG:       [[VAR_43_:%.+]] = "onnx.SqueezeV11"([[VAR_41_]]#1) {axes = [0]} : (tensor<1x12xf32>) -> tensor<12xf32>
// CHECK:           [[VAR_44_:%.+]]:3 = "onnx.SplitV11"([[VAR_42_]]) {axis = 0 : si64} : (tensor<12xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
// CHECK-DAG:       [[VAR_45_:%.+]] = builtin.unrealized_conversion_cast [[VAR_44_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_46_:%.+]] = builtin.unrealized_conversion_cast [[VAR_44_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_47_:%.+]] = builtin.unrealized_conversion_cast [[VAR_44_]]#2 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_48_:%.+]]:3 = "onnx.SplitV11"([[VAR_43_]]) {axis = 0 : si64} : (tensor<12xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_49_:%.+]] = builtin.unrealized_conversion_cast [[VAR_48_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_50_:%.+]] = builtin.unrealized_conversion_cast [[VAR_48_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_51_:%.+]] = builtin.unrealized_conversion_cast [[VAR_48_]]#2 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7){
// CHECK-DAG:         [[VAR_55_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]){
// CHECK:               [[VAR_65_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_55_1_]], [[VAR_65_]]#0, [[VAR_65_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_5_]]{{.}}[[VAR_65_]]#0, [[VAR_65_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             [[LOAD_PARAM_4_MEM_1_:%.+]] = builtin.unrealized_conversion_cast [[RES_5_]] : memref<2x3xf32> to tensor<2x3xf32>
// CHECK:             [[LOAD_PARAM_5_MEM_1_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_4_MEM_1_]], [[VAR_16_]]) : (tensor<2x3xf32>, tensor<3x16xf32>) -> tensor<2x16xf32>
// CHECK-DAG:         [[VAR_60_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_5_MEM_1_]] : tensor<2x16xf32> to memref<2x16xf32>
// CHECK-DAG:         [[VAR_61_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_62_:%.+]] = "onnx.MatMul"([[VAR_61_]], [[VAR_17_]]) : (tensor<2x4xf32>, tensor<4x16xf32>) -> tensor<2x16xf32>
// CHECK-DAG:         [[VAR_63_:%.+]] = builtin.unrealized_conversion_cast [[VAR_62_]] : tensor<2x16xf32> to memref<2x16xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[VAR_65_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_65_1_]]#0, [[VAR_65_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_60_MEM_:%.+]] = krnl.load [[VAR_60_]]{{.}}[[VAR_65_1_]]#0, [[VAR_65_1_]]#1] : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_VAR_63_MEM_:%.+]] = krnl.load [[VAR_63_]]{{.}}[[VAR_65_1_]]#0, [[VAR_65_1_]]#1] : memref<2x16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_69_:%.+]] = arith.addf [[LOAD_VAR_60_MEM_]], [[LOAD_VAR_63_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_24_MEM_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_65_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_28_MEM_:%.+]] = krnl.load [[VAR_28_]]{{.}}[[VAR_65_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_72_:%.+]] = arith.addf [[VAR_69_]], [[LOAD_VAR_24_MEM_]] : f32
// CHECK-DAG:           [[VAR_73_:%.+]] = arith.addf [[VAR_72_]], [[LOAD_VAR_28_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_45_MEM_:%.+]] = krnl.load [[VAR_45_]]{{.}}[[VAR_65_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_75_:%.+]] = arith.mulf [[LOAD_VAR_45_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_76_:%.+]] = arith.addf [[VAR_73_]], [[VAR_75_]] : f32
// CHECK:               [[VAR_77_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_76_]] : f32
// CHECK:               [[VAR_78_:%.+]] = math.exp [[VAR_77_]] : f32
// CHECK:               [[VAR_79_:%.+]] = arith.addf [[VAR_78_]], [[VAR_cst_]] : f32
// CHECK-DAG:           [[VAR_80_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_79_]] : f32
// CHECK-DAG:           [[VAR_81_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_65_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_60_MEM_1_:%.+]] = krnl.load [[VAR_60_]]{{.}}[[VAR_65_1_]]#0, [[VAR_81_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_83_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_65_1_]]#1]
// CHECK:               [[LOAD_VAR_63_MEM_1_:%.+]] = krnl.load [[VAR_63_]]{{.}}[[VAR_65_1_]]#0, [[VAR_83_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_85_:%.+]] = arith.addf [[LOAD_VAR_60_MEM_1_]], [[LOAD_VAR_63_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_26_MEM_:%.+]] = krnl.load [[VAR_26_]]{{.}}[[VAR_65_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_30_MEM_:%.+]] = krnl.load [[VAR_30_]]{{.}}[[VAR_65_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_88_:%.+]] = arith.addf [[VAR_85_]], [[LOAD_VAR_26_MEM_]] : f32
// CHECK-DAG:           [[VAR_89_:%.+]] = arith.addf [[VAR_88_]], [[LOAD_VAR_30_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_47_MEM_:%.+]] = krnl.load [[VAR_47_]]{{.}}[[VAR_65_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_91_:%.+]] = arith.mulf [[LOAD_VAR_47_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_92_:%.+]] = arith.addf [[VAR_89_]], [[VAR_91_]] : f32
// CHECK:               [[VAR_93_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_92_]] : f32
// CHECK:               [[VAR_94_:%.+]] = math.exp [[VAR_93_]] : f32
// CHECK:               [[VAR_95_:%.+]] = arith.addf [[VAR_94_]], [[VAR_cst_]] : f32
// CHECK-DAG:           [[VAR_96_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_95_]] : f32
// CHECK-DAG:           [[VAR_97_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_65_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_60_MEM_2_:%.+]] = krnl.load [[VAR_60_]]{{.}}[[VAR_65_1_]]#0, [[VAR_97_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_99_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_65_1_]]#1]
// CHECK:               [[LOAD_VAR_63_MEM_2_:%.+]] = krnl.load [[VAR_63_]]{{.}}[[VAR_65_1_]]#0, [[VAR_99_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_101_:%.+]] = arith.addf [[LOAD_VAR_60_MEM_2_]], [[LOAD_VAR_63_MEM_2_]] : f32
// CHECK-DAG:           [[LOAD_VAR_27_MEM_:%.+]] = krnl.load [[VAR_27_]]{{.}}[[VAR_65_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_31_MEM_:%.+]] = krnl.load [[VAR_31_]]{{.}}[[VAR_65_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_104_:%.+]] = arith.addf [[VAR_101_]], [[LOAD_VAR_27_MEM_]] : f32
// CHECK:               [[VAR_105_:%.+]] = arith.addf [[VAR_104_]], [[LOAD_VAR_31_MEM_]] : f32
// CHECK-DAG:           [[VAR_106_:%.+]] = math.tanh [[VAR_105_]] : f32
// CHECK-DAG:           [[VAR_107_:%.+]] = arith.mulf [[VAR_96_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_108_:%.+]] = arith.mulf [[VAR_80_]], [[VAR_106_]] : f32
// CHECK-DAG:           [[VAR_109_:%.+]] = arith.addf [[VAR_107_]], [[VAR_108_]] : f32
// CHECK-DAG:           [[VAR_110_:%.+]] = affine.apply [[MAP_2_]](){{.}}[[VAR_65_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_60_MEM_3_:%.+]] = krnl.load [[VAR_60_]]{{.}}[[VAR_65_1_]]#0, [[VAR_110_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_112_:%.+]] = affine.apply [[MAP_2_]](){{.}}[[VAR_65_1_]]#1]
// CHECK:               [[LOAD_VAR_63_MEM_3_:%.+]] = krnl.load [[VAR_63_]]{{.}}[[VAR_65_1_]]#0, [[VAR_112_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_114_:%.+]] = arith.addf [[LOAD_VAR_60_MEM_3_]], [[LOAD_VAR_63_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_25_MEM_:%.+]] = krnl.load [[VAR_25_]]{{.}}[[VAR_65_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_29_MEM_:%.+]] = krnl.load [[VAR_29_]]{{.}}[[VAR_65_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_117_:%.+]] = arith.addf [[VAR_114_]], [[LOAD_VAR_25_MEM_]] : f32
// CHECK-DAG:           [[VAR_118_:%.+]] = arith.addf [[VAR_117_]], [[LOAD_VAR_29_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_46_MEM_:%.+]] = krnl.load [[VAR_46_]]{{.}}[[VAR_65_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_120_:%.+]] = arith.mulf [[LOAD_VAR_46_MEM_]], [[VAR_109_]] : f32
// CHECK:               [[VAR_121_:%.+]] = arith.addf [[VAR_118_]], [[VAR_120_]] : f32
// CHECK:               [[VAR_122_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_121_]] : f32
// CHECK:               [[VAR_123_:%.+]] = math.exp [[VAR_122_]] : f32
// CHECK:               [[VAR_124_:%.+]] = arith.addf [[VAR_123_]], [[VAR_cst_]] : f32
// CHECK-DAG:           [[VAR_125_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_124_]] : f32
// CHECK-DAG:           [[VAR_126_:%.+]] = math.tanh [[VAR_109_]] : f32
// CHECK:               [[VAR_127_:%.+]] = arith.mulf [[VAR_125_]], [[VAR_126_]] : f32
// CHECK:               krnl.store [[VAR_109_]], [[RES_2_]]{{.}}[[VAR_65_1_]]#0, [[VAR_65_1_]]#1] : memref<2x4xf32>
// CHECK:               krnl.store [[VAR_127_]], [[RES_1_]]{{.}}[[VAR_65_1_]]#0, [[VAR_65_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_4_]]) with ([[LOOP_4_]] -> [[I_7_:%.+]] = 0 to 7){
// CHECK:             [[VAR_55_2_:%.+]] = krnl.get_induction_var_value([[LOOP_4_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[RES_5_:%.+]] = affine.apply [[MAP_3_]]([[VAR_55_2_]])
// CHECK-DAG:         [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_5_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_5_]]#0, [[LOOP_5_]]#1) with ([[LOOP_5_]]#0 -> [[I_8_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_5_]]#1 -> [[I_9_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]){
// CHECK:               [[LOAD_PARAM_0_MEM_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_5_]]#0, [[LOOP_5_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[RES_5_]], [[LOAD_PARAM_0_MEM_1_]]#0, [[LOAD_PARAM_0_MEM_1_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_2_]], [[RES_6_]]{{.}}[[LOAD_PARAM_0_MEM_1_]]#0, [[LOAD_PARAM_0_MEM_1_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             [[LOAD_PARAM_5_MEM_1_1_:%.+]] = builtin.unrealized_conversion_cast [[RES_6_]] : memref<2x3xf32> to tensor<2x3xf32>
// CHECK:             [[VAR_60_1_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_5_MEM_1_1_]], [[VAR_18_]]) : (tensor<2x3xf32>, tensor<3x16xf32>) -> tensor<2x16xf32>
// CHECK-DAG:         [[VAR_61_1_:%.+]] = builtin.unrealized_conversion_cast [[VAR_60_1_]] : tensor<2x16xf32> to memref<2x16xf32>
// CHECK-DAG:         [[VAR_62_1_:%.+]] = builtin.unrealized_conversion_cast [[RES_3_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_63_1_:%.+]] = "onnx.MatMul"([[VAR_62_1_]], [[VAR_19_]]) : (tensor<2x4xf32>, tensor<4x16xf32>) -> tensor<2x16xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]] = builtin.unrealized_conversion_cast [[VAR_63_1_]] : tensor<2x16xf32> to memref<2x16xf32>
// CHECK-DAG:         [[LOOP_6_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_6_]]#0, [[LOOP_6_]]#1) with ([[LOOP_6_]]#0 -> [[I_10_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_6_]]#1 -> [[I_11_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[LOAD_PARAM_0_MEM_1_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_6_]]#0, [[LOOP_6_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[RES_4_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_63_MEM_4_:%.+]] = krnl.load [[VAR_61_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x16xf32>
// CHECK-DAG:           [[VAR_69_1_:%.+]] = krnl.load [[LOOP_3_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_24_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_63_MEM_4_]], [[VAR_69_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_28_MEM_1_:%.+]] = krnl.load [[VAR_33_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_72_1_:%.+]] = krnl.load [[VAR_37_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_73_1_:%.+]] = arith.addf [[LOAD_VAR_24_MEM_1_]], [[LOAD_VAR_28_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_45_MEM_1_:%.+]] = arith.addf [[VAR_73_1_]], [[VAR_72_1_]] : f32
// CHECK-DAG:           [[VAR_75_1_:%.+]] = krnl.load [[VAR_49_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_76_1_:%.+]] = arith.mulf [[VAR_75_1_]], [[LOAD_PARAM_0_MEM_2_]] : f32
// CHECK:               [[VAR_77_1_:%.+]] = arith.addf [[LOAD_VAR_45_MEM_1_]], [[VAR_76_1_]] : f32
// CHECK:               [[VAR_78_1_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_77_1_]] : f32
// CHECK:               [[VAR_79_1_:%.+]] = math.exp [[VAR_78_1_]] : f32
// CHECK:               [[VAR_80_1_:%.+]] = arith.addf [[VAR_79_1_]], [[VAR_cst_]] : f32
// CHECK-DAG:           [[VAR_81_1_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_80_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_60_MEM_1_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_83_1_:%.+]] = krnl.load [[VAR_61_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_VAR_60_MEM_1_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_VAR_63_MEM_1_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1]
// CHECK:               [[VAR_85_1_:%.+]] = krnl.load [[LOOP_3_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_VAR_63_MEM_1_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_VAR_26_MEM_1_:%.+]] = arith.addf [[VAR_83_1_]], [[VAR_85_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_30_MEM_1_:%.+]] = krnl.load [[VAR_35_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_88_1_:%.+]] = krnl.load [[VAR_39_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_89_1_:%.+]] = arith.addf [[LOAD_VAR_26_MEM_1_]], [[LOAD_VAR_30_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_47_MEM_1_:%.+]] = arith.addf [[VAR_89_1_]], [[VAR_88_1_]] : f32
// CHECK-DAG:           [[VAR_91_1_:%.+]] = krnl.load [[VAR_51_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_92_1_:%.+]] = arith.mulf [[VAR_91_1_]], [[LOAD_PARAM_0_MEM_2_]] : f32
// CHECK:               [[VAR_93_1_:%.+]] = arith.addf [[LOAD_VAR_47_MEM_1_]], [[VAR_92_1_]] : f32
// CHECK:               [[VAR_94_1_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_93_1_]] : f32
// CHECK:               [[VAR_95_1_:%.+]] = math.exp [[VAR_94_1_]] : f32
// CHECK:               [[VAR_96_1_:%.+]] = arith.addf [[VAR_95_1_]], [[VAR_cst_]] : f32
// CHECK-DAG:           [[VAR_97_1_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_96_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_60_MEM_2_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_99_1_:%.+]] = krnl.load [[VAR_61_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_VAR_60_MEM_2_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_VAR_63_MEM_2_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1]
// CHECK:               [[VAR_101_1_:%.+]] = krnl.load [[LOOP_3_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_VAR_63_MEM_2_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_VAR_27_MEM_1_:%.+]] = arith.addf [[VAR_99_1_]], [[VAR_101_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_31_MEM_1_:%.+]] = krnl.load [[VAR_36_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_104_1_:%.+]] = krnl.load [[VAR_40_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_105_1_:%.+]] = arith.addf [[LOAD_VAR_27_MEM_1_]], [[LOAD_VAR_31_MEM_1_]] : f32
// CHECK:               [[VAR_106_1_:%.+]] = arith.addf [[VAR_105_1_]], [[VAR_104_1_]] : f32
// CHECK-DAG:           [[VAR_107_1_:%.+]] = math.tanh [[VAR_106_1_]] : f32
// CHECK-DAG:           [[VAR_108_1_:%.+]] = arith.mulf [[VAR_97_1_]], [[LOAD_PARAM_0_MEM_2_]] : f32
// CHECK:               [[VAR_109_1_:%.+]] = arith.mulf [[VAR_81_1_]], [[VAR_107_1_]] : f32
// CHECK-DAG:           [[VAR_110_1_:%.+]] = arith.addf [[VAR_108_1_]], [[VAR_109_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_60_MEM_3_:%.+]] = affine.apply [[MAP_2_]](){{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_112_1_:%.+]] = krnl.load [[VAR_61_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_VAR_60_MEM_3_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_VAR_63_MEM_3_:%.+]] = affine.apply [[MAP_2_]](){{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1]
// CHECK:               [[VAR_114_1_:%.+]] = krnl.load [[LOOP_3_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_VAR_63_MEM_3_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_VAR_25_MEM_1_:%.+]] = arith.addf [[VAR_112_1_]], [[VAR_114_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_29_MEM_1_:%.+]] = krnl.load [[VAR_34_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_117_1_:%.+]] = krnl.load [[VAR_38_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_118_1_:%.+]] = arith.addf [[LOAD_VAR_25_MEM_1_]], [[LOAD_VAR_29_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_46_MEM_1_:%.+]] = arith.addf [[VAR_118_1_]], [[VAR_117_1_]] : f32
// CHECK-DAG:           [[VAR_120_1_:%.+]] = krnl.load [[VAR_50_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_121_1_:%.+]] = arith.mulf [[VAR_120_1_]], [[VAR_110_1_]] : f32
// CHECK:               [[VAR_122_1_:%.+]] = arith.addf [[LOAD_VAR_46_MEM_1_]], [[VAR_121_1_]] : f32
// CHECK:               [[VAR_123_1_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_122_1_]] : f32
// CHECK:               [[VAR_124_1_:%.+]] = math.exp [[VAR_123_1_]] : f32
// CHECK:               [[VAR_125_1_:%.+]] = arith.addf [[VAR_124_1_]], [[VAR_cst_]] : f32
// CHECK-DAG:           [[VAR_126_1_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_125_1_]] : f32
// CHECK-DAG:           [[VAR_127_1_:%.+]] = math.tanh [[VAR_110_1_]] : f32
// CHECK:               [[VAR_128_:%.+]] = arith.mulf [[VAR_126_1_]], [[VAR_127_1_]] : f32
// CHECK:               krnl.store [[VAR_110_1_]], [[RES_4_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK:               krnl.store [[VAR_128_]], [[RES_3_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOOP_7_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_7_]]#0, [[LOOP_7_]]#1) with ([[LOOP_7_]]#0 -> [[I_12_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_7_]]#1 -> [[I_13_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:             [[VAR_55_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_7_]]#0, [[LOOP_7_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[RES_5_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_55_3_]]#0, [[VAR_55_3_]]#1] : memref<2x4xf32>
// CHECK:             krnl.store [[RES_5_1_]], [[RES_]]{{.}}[[VAR_c0_]], [[VAR_55_3_]]#0, [[VAR_55_3_]]#1] : memref<2x2x4xf32>
// CHECK:             [[RES_6_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_55_3_]]#0, [[VAR_55_3_]]#1] : memref<2x4xf32>
// CHECK:             krnl.store [[RES_6_]], [[RES_]]{{.}}[[VAR_c1_]], [[VAR_55_3_]]#0, [[VAR_55_3_]]#1] : memref<2x2x4xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x2x4xf32>
// CHECK:         }
}

// -----

func.func private @test_lstm_unknown_dims(%arg0: tensor<?x?x?xf32>, %arg1: tensor<1x16x?xf32>, %arg2: tensor<1x16x4xf32>, %arg3: tensor<1x32xf32>, %arg4: tensor<1x?x4xf32>, %arg5: tensor<1x?x4xf32>, %arg6: tensor<1x12xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4, %arg5, %arg6) {hidden_size = 4 : si64} : (tensor<?x?x?xf32>, tensor<1x16x?xf32>, tensor<1x16x4xf32>, tensor<1x32xf32>, none, tensor<1x?x4xf32>, tensor<1x?x4xf32>, tensor<1x12xf32>) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>
// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 + 8)>
// CHECK-DAG: [[MAP_2_:#.+]] = affine_map<()[s0] -> (s0 + 12)>
// CHECK-DAG: [[MAP_3_:#.+]] = affine_map<()[s0] -> (s0 + 4)>
// CHECK-LABEL:  func private @test_lstm_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x?xf32>, [[PARAM_1_:%.+]]: memref<1x16x?xf32>, [[PARAM_2_:%.+]]: memref<1x16x4xf32>, [[PARAM_3_:%.+]]: memref<1x32xf32>, [[PARAM_4_:%.+]]: memref<1x?x4xf32>, [[PARAM_5_:%.+]]: memref<1x?x4xf32>, [[PARAM_6_:%.+]]: memref<1x12xf32>) -> memref<1x?x4xf32> {
// CHECK-DAG:       [[VAR_c4_i64_:%.+]] = arith.constant 4 : i64
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_6_]] : memref<1x12xf32> to tensor<1x12xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_3_]] : memref<1x32xf32> to tensor<1x32xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_2_]] : memref<1x16x4xf32> to tensor<1x16x4xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_1_]] : memref<1x16x?xf32> to tensor<1x16x?xf32>
// CHECK:           [[VAR_4_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_4_]]) {{.*}}: memref<1x?x4xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<?x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[VAR_6_]]) {{.*}}: memref<?x4xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<?x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc([[VAR_8_]]) {{.*}}: memref<?x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_6_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[VAR_34_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[VAR_c0_]], [[VAR_34_]]#0, [[VAR_34_]]#1] : memref<1x?x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]]{{.}}[[VAR_34_]]#0, [[VAR_34_]]#1] : memref<?x4xf32>
// CHECK:             [[LOAD_PARAM_5_MEM_:%.+]] = krnl.load [[PARAM_5_]]{{.}}[[VAR_c0_]], [[VAR_34_]]#0, [[VAR_34_]]#1] : memref<1x?x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_5_MEM_]], [[RES_2_]]{{.}}[[VAR_34_]]#0, [[VAR_34_]]#1] : memref<?x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.SqueezeV11"([[VAR_3_]]) {axes = [0]} : (tensor<1x16x?xf32>) -> tensor<16x?xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.SqueezeV11"([[VAR_2_]]) {axes = [0]} : (tensor<1x16x4xf32>) -> tensor<16x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Transpose"([[VAR_11_]]) {perm = [1, 0]} : (tensor<16x?xf32>) -> tensor<?x16xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.Transpose"([[VAR_12_]]) {perm = [1, 0]} : (tensor<16x4xf32>) -> tensor<4x16xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.SqueezeV11"([[VAR_1_]]) {axes = [0]} : (tensor<1x32xf32>) -> tensor<32xf32>
// CHECK:           [[VAR_16_:%.+]]:8 = "onnx.SplitV11"([[VAR_15_]]) {axis = 0 : si64} : (tensor<32xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
// CHECK-DAG:       [[VAR_17_:%.+]] = builtin.unrealized_conversion_cast [[VAR_16_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = builtin.unrealized_conversion_cast [[VAR_16_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = builtin.unrealized_conversion_cast [[VAR_16_]]#2 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = builtin.unrealized_conversion_cast [[VAR_16_]]#3 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = builtin.unrealized_conversion_cast [[VAR_16_]]#4 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = builtin.unrealized_conversion_cast [[VAR_16_]]#5 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_23_:%.+]] = builtin.unrealized_conversion_cast [[VAR_16_]]#6 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_24_:%.+]] = builtin.unrealized_conversion_cast [[VAR_16_]]#7 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_25_:%.+]] = "onnx.SqueezeV11"([[VAR_0_]]) {axes = [0]} : (tensor<1x12xf32>) -> tensor<12xf32>
// CHECK:           [[VAR_26_:%.+]]:3 = "onnx.SplitV11"([[VAR_25_]]) {axis = 0 : si64} : (tensor<12xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
// CHECK-DAG:       [[VAR_27_:%.+]] = builtin.unrealized_conversion_cast [[VAR_26_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_28_:%.+]] = builtin.unrealized_conversion_cast [[VAR_26_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_29_:%.+]] = builtin.unrealized_conversion_cast [[VAR_26_]]#2 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK-DAG:       [[VAR_31_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?x?x?xf32>
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to [[MAP_0_]]([[VAR_31_]])){
// CHECK-DAG:         [[VAR_34_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_1_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<?x?x?xf32>
// CHECK-DAG:         [[LOAD_PARAM_5_MEM_1_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c2_]] : memref<?x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc([[LOAD_PARAM_4_MEM_1_]], [[LOAD_PARAM_5_MEM_1_]]) {{.*}}: memref<?x?xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[VAR_c0_]] to [[LOAD_PARAM_4_MEM_1_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[VAR_c0_]] to [[LOAD_PARAM_5_MEM_1_]]){
// CHECK:               [[VAR_46_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_34_1_]], [[VAR_46_]]#0, [[VAR_46_]]#1] : memref<?x?x?xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_3_]]{{.}}[[VAR_46_]]#0, [[VAR_46_]]#1] : memref<?x?xf32>
// CHECK:             }
// CHECK:             [[VAR_39_:%.+]] = builtin.unrealized_conversion_cast [[RES_3_]] : memref<?x?xf32> to tensor<?x?xf32>
// CHECK:             [[VAR_40_:%.+]] = "onnx.MatMul"([[VAR_39_]], [[VAR_13_]]) : (tensor<?x?xf32>, tensor<?x16xf32>) -> tensor<?x16xf32>
// CHECK-DAG:         [[VAR_41_:%.+]] = builtin.unrealized_conversion_cast [[VAR_40_]] : tensor<?x16xf32> to memref<?x16xf32>
// CHECK-DAG:         [[VAR_42_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<?x4xf32> to tensor<?x4xf32>
// CHECK:             [[VAR_43_:%.+]] = "onnx.MatMul"([[VAR_42_]], [[VAR_14_]]) : (tensor<?x4xf32>, tensor<4x16xf32>) -> tensor<?x16xf32>
// CHECK-DAG:         [[VAR_44_:%.+]] = builtin.unrealized_conversion_cast [[VAR_43_]] : tensor<?x16xf32> to memref<?x16xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[VAR_c0_]] to [[VAR_6_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[VAR_46_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_46_1_]]#0, [[VAR_46_1_]]#1] : memref<?x4xf32>
// CHECK-DAG:           [[LOAD_VAR_41_MEM_:%.+]] = krnl.load [[VAR_41_]]{{.}}[[VAR_46_1_]]#0, [[VAR_46_1_]]#1] : memref<?x16xf32>
// CHECK-DAG:           [[LOAD_VAR_44_MEM_:%.+]] = krnl.load [[VAR_44_]]{{.}}[[VAR_46_1_]]#0, [[VAR_46_1_]]#1] : memref<?x16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_50_:%.+]] = arith.addf [[LOAD_VAR_41_MEM_]], [[LOAD_VAR_44_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_17_MEM_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_46_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_21_MEM_:%.+]] = krnl.load [[VAR_21_]]{{.}}[[VAR_46_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_53_:%.+]] = arith.addf [[VAR_50_]], [[LOAD_VAR_17_MEM_]] : f32
// CHECK-DAG:           [[VAR_54_:%.+]] = arith.addf [[VAR_53_]], [[LOAD_VAR_21_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_27_MEM_:%.+]] = krnl.load [[VAR_27_]]{{.}}[[VAR_46_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_56_:%.+]] = arith.mulf [[LOAD_VAR_27_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_57_:%.+]] = arith.addf [[VAR_54_]], [[VAR_56_]] : f32
// CHECK:               [[VAR_58_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_57_]] : f32
// CHECK:               [[VAR_59_:%.+]] = math.exp [[VAR_58_]] : f32
// CHECK:               [[VAR_60_:%.+]] = arith.addf [[VAR_59_]], [[VAR_cst_]] : f32
// CHECK-DAG:           [[VAR_61_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_60_]] : f32
// CHECK-DAG:           [[VAR_62_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_46_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_41_MEM_1_:%.+]] = krnl.load [[VAR_41_]]{{.}}[[VAR_46_1_]]#0, [[VAR_62_]]{{.}} : memref<?x16xf32>
// CHECK-DAG:           [[VAR_64_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_46_1_]]#1]
// CHECK:               [[LOAD_VAR_44_MEM_1_:%.+]] = krnl.load [[VAR_44_]]{{.}}[[VAR_46_1_]]#0, [[VAR_64_]]{{.}} : memref<?x16xf32>
// CHECK-DAG:           [[VAR_66_:%.+]] = arith.addf [[LOAD_VAR_41_MEM_1_]], [[LOAD_VAR_44_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_19_MEM_:%.+]] = krnl.load [[VAR_19_]]{{.}}[[VAR_46_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_23_MEM_:%.+]] = krnl.load [[VAR_23_]]{{.}}[[VAR_46_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_69_:%.+]] = arith.addf [[VAR_66_]], [[LOAD_VAR_19_MEM_]] : f32
// CHECK-DAG:           [[VAR_70_:%.+]] = arith.addf [[VAR_69_]], [[LOAD_VAR_23_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_29_MEM_:%.+]] = krnl.load [[VAR_29_]]{{.}}[[VAR_46_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_72_:%.+]] = arith.mulf [[LOAD_VAR_29_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_73_:%.+]] = arith.addf [[VAR_70_]], [[VAR_72_]] : f32
// CHECK:               [[VAR_74_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_73_]] : f32
// CHECK:               [[VAR_75_:%.+]] = math.exp [[VAR_74_]] : f32
// CHECK:               [[VAR_76_:%.+]] = arith.addf [[VAR_75_]], [[VAR_cst_]] : f32
// CHECK-DAG:           [[VAR_77_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_76_]] : f32
// CHECK-DAG:           [[VAR_78_:%.+]] = affine.apply [[MAP_2_]](){{.}}[[VAR_46_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_41_MEM_2_:%.+]] = krnl.load [[VAR_41_]]{{.}}[[VAR_46_1_]]#0, [[VAR_78_]]{{.}} : memref<?x16xf32>
// CHECK-DAG:           [[VAR_80_:%.+]] = affine.apply [[MAP_2_]](){{.}}[[VAR_46_1_]]#1]
// CHECK:               [[LOAD_VAR_44_MEM_2_:%.+]] = krnl.load [[VAR_44_]]{{.}}[[VAR_46_1_]]#0, [[VAR_80_]]{{.}} : memref<?x16xf32>
// CHECK-DAG:           [[VAR_82_:%.+]] = arith.addf [[LOAD_VAR_41_MEM_2_]], [[LOAD_VAR_44_MEM_2_]] : f32
// CHECK-DAG:           [[LOAD_VAR_20_MEM_:%.+]] = krnl.load [[VAR_20_]]{{.}}[[VAR_46_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_24_MEM_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_46_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_85_:%.+]] = arith.addf [[VAR_82_]], [[LOAD_VAR_20_MEM_]] : f32
// CHECK:               [[VAR_86_:%.+]] = arith.addf [[VAR_85_]], [[LOAD_VAR_24_MEM_]] : f32
// CHECK-DAG:           [[VAR_87_:%.+]] = math.tanh [[VAR_86_]] : f32
// CHECK-DAG:           [[VAR_88_:%.+]] = arith.mulf [[VAR_77_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_89_:%.+]] = arith.mulf [[VAR_61_]], [[VAR_87_]] : f32
// CHECK-DAG:           [[VAR_90_:%.+]] = arith.addf [[VAR_88_]], [[VAR_89_]] : f32
// CHECK-DAG:           [[VAR_91_:%.+]] = affine.apply [[MAP_3_]](){{.}}[[VAR_46_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_41_MEM_3_:%.+]] = krnl.load [[VAR_41_]]{{.}}[[VAR_46_1_]]#0, [[VAR_91_]]{{.}} : memref<?x16xf32>
// CHECK-DAG:           [[VAR_93_:%.+]] = affine.apply [[MAP_3_]](){{.}}[[VAR_46_1_]]#1]
// CHECK:               [[LOAD_VAR_44_MEM_3_:%.+]] = krnl.load [[VAR_44_]]{{.}}[[VAR_46_1_]]#0, [[VAR_93_]]{{.}} : memref<?x16xf32>
// CHECK-DAG:           [[VAR_95_:%.+]] = arith.addf [[LOAD_VAR_41_MEM_3_]], [[LOAD_VAR_44_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_18_MEM_:%.+]] = krnl.load [[VAR_18_]]{{.}}[[VAR_46_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_22_MEM_:%.+]] = krnl.load [[VAR_22_]]{{.}}[[VAR_46_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_98_:%.+]] = arith.addf [[VAR_95_]], [[LOAD_VAR_18_MEM_]] : f32
// CHECK-DAG:           [[VAR_99_:%.+]] = arith.addf [[VAR_98_]], [[LOAD_VAR_22_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_28_MEM_:%.+]] = krnl.load [[VAR_28_]]{{.}}[[VAR_46_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_101_:%.+]] = arith.mulf [[LOAD_VAR_28_MEM_]], [[VAR_90_]] : f32
// CHECK:               [[VAR_102_:%.+]] = arith.addf [[VAR_99_]], [[VAR_101_]] : f32
// CHECK:               [[VAR_103_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_102_]] : f32
// CHECK:               [[VAR_104_:%.+]] = math.exp [[VAR_103_]] : f32
// CHECK:               [[VAR_105_:%.+]] = arith.addf [[VAR_104_]], [[VAR_cst_]] : f32
// CHECK-DAG:           [[VAR_106_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_105_]] : f32
// CHECK-DAG:           [[VAR_107_:%.+]] = math.tanh [[VAR_90_]] : f32
// CHECK:               [[VAR_108_:%.+]] = arith.mulf [[VAR_106_]], [[VAR_107_]] : f32
// CHECK:               krnl.store [[VAR_90_]], [[RES_2_]]{{.}}[[VAR_46_1_]]#0, [[VAR_46_1_]]#1] : memref<?x4xf32>
// CHECK:               krnl.store [[VAR_108_]], [[RES_1_]]{{.}}[[VAR_46_1_]]#0, [[VAR_46_1_]]#1] : memref<?x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           [[VAR_32_:%.+]] = arith.index_cast [[VAR_6_]] : index to i64
// CHECK:           [[VAR_33_:%.+]] = arith.muli [[VAR_32_]], [[VAR_c4_i64_]] : i64
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_1_]], [[VAR_33_]], [[VAR_c0_]], [[VAR_c0_]]) : (memref<1x?x4xf32>, memref<?x4xf32>, i64, index, index) -> ()
// CHECK:           return [[RES_]] : memref<1x?x4xf32>
// CHECK:         }
}
