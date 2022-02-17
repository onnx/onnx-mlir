// RUN: onnx-mlir-opt -O3 --shape-inference --convert-onnx-to-krnl='emit-intermediate-ir' --canonicalize %s -split-input-file | FileCheck %s

func private @test_lstm_forward_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x16x3xf32>, %arg2: tensor<1x16x4xf32>, %arg3: tensor<1x32xf32>, %arg4: tensor<1x2x4xf32>, %arg5: tensor<1x2x4xf32>, %arg6: tensor<1x12xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4, %arg5, %arg6) {hidden_size = 4 : si64} : (tensor<7x2x3xf32>, tensor<1x16x3xf32>, tensor<1x16x4xf32>, tensor<1x32xf32>, none, tensor<1x2x4xf32>, tensor<1x2x4xf32>, tensor<1x12xf32>) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

// CHECK-DAG: #map0 = affine_map<()[s0] -> (s0 + 8)>
// CHECK-DAG: #map1 = affine_map<()[s0] -> (s0 + 12)>
// CHECK-DAG: #map2 = affine_map<()[s0] -> (s0 + 4)>
// CHECK-LABEL:  func private @test_lstm_forward_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x16x3xf32>, [[PARAM_2_:%.+]]: memref<1x16x4xf32>, [[PARAM_3_:%.+]]: memref<1x32xf32>, [[PARAM_4_:%.+]]: memref<1x2x4xf32>, [[PARAM_5_:%.+]]: memref<1x2x4xf32>, [[PARAM_6_:%.+]]: memref<1x12xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[VAR_c32_i64_:%.+]] = arith.constant 32 : i64
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[UCC_PARAM_6:%.+]] = builtin.unrealized_conversion_cast %arg6 : memref<1x12xf32> to tensor<1x12xf32>
// CHECK-DAG:       [[UCC_PARAM_3:%.+]] = builtin.unrealized_conversion_cast %arg3 : memref<1x32xf32> to tensor<1x32xf32>
// CHECK-DAG:       [[UCC_PARAM_2:%.+]] = builtin.unrealized_conversion_cast %arg2 : memref<1x16x4xf32> to tensor<1x16x4xf32>
// CHECK-DAG:       [[UCC_PARAM_1:%.+]] = builtin.unrealized_conversion_cast %arg1 : memref<1x16x3xf32> to tensor<1x16x3xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[VAR_c0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:             [[LOAD_PARAM_5_MEM_:%.+]] = krnl.load [[PARAM_5_]]{{.}}[[VAR_c0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_5_MEM_]], [[RES_2_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.SqueezeV11"([[UCC_PARAM_1]]) {axes = [0]} : (tensor<1x16x3xf32>) -> memref<16x3xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.SqueezeV11"([[UCC_PARAM_2]]) {axes = [0]} : (tensor<1x16x4xf32>) -> memref<16x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Transpose"([[VAR_4_]]) {perm = [1, 0]} : (memref<16x3xf32>) -> memref<3x16xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Transpose"([[VAR_5_]]) {perm = [1, 0]} : (memref<16x4xf32>) -> memref<4x16xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.SqueezeV11"([[UCC_PARAM_3]]) {axes = [0]} : (tensor<1x32xf32>) -> memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_9_:%.+]]:8 = "onnx.SplitV11"([[VAR_8_]]) {axis = 0 : si64} : (memref<32xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.SqueezeV11"([[UCC_PARAM_6]]) {axes = [0]} : (tensor<1x12xf32>) -> memref<12xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]]:3 = "onnx.SplitV11"([[VAR_10_]]) {axis = 0 : si64} : (memref<12xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7){
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]){
// CHECK:               [[VAR_18_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_18_]]#0, [[VAR_18_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_3_]]{{.}}[[VAR_18_]]#0, [[VAR_18_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_15_:%.+]] = "onnx.MatMul"([[RES_3_]], [[VAR_6_]]) : (memref<2x3xf32>, memref<3x16xf32>) -> memref<2x16xf32>
// CHECK-DAG:         [[VAR_16_:%.+]] = "onnx.MatMul"([[RES_1_]], [[VAR_7_]]) : (memref<2x4xf32>, memref<4x16xf32>) -> memref<2x16xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[VAR_18_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_15_MEM_:%.+]] = krnl.load [[VAR_15_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1] : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_VAR_16_MEM_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1] : memref<2x16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_22_:%.+]] = arith.addf [[LOAD_VAR_15_MEM_]], [[LOAD_VAR_16_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_9_MEM_:%.+]] = krnl.load [[VAR_9_]]#0{{.}}[[VAR_18_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_9_MEM_1_:%.+]] = krnl.load [[VAR_9_]]#4{{.}}[[VAR_18_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_25_:%.+]] = arith.addf [[VAR_22_]], [[LOAD_VAR_9_MEM_]] : f32
// CHECK-DAG:           [[VAR_26_:%.+]] = arith.addf [[VAR_25_]], [[LOAD_VAR_9_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_:%.+]] = krnl.load [[VAR_11_]]#0{{.}}[[VAR_18_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_28_:%.+]] = arith.mulf [[LOAD_VAR_11_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_29_:%.+]] = arith.addf [[VAR_26_]], [[VAR_28_]] : f32
// CHECK:               [[VAR_30_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_29_]] : f32
// CHECK:               [[VAR_31_:%.+]] = math.exp [[VAR_30_]] : f32
// CHECK:               [[VAR_32_:%.+]] = arith.addf [[VAR_31_]], [[VAR_cst_]] : f32
// CHECK-DAG:           [[VAR_33_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_32_]] : f32
// CHECK-DAG:           [[VAR_34_:%.+]] = affine.apply #map0(){{.}}[[VAR_18_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_15_MEM_1_:%.+]] = krnl.load [[VAR_15_]]{{.}}[[VAR_18_1_]]#0, [[VAR_34_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_36_:%.+]] = affine.apply #map0(){{.}}[[VAR_18_1_]]#1]
// CHECK:               [[LOAD_VAR_16_MEM_1_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_18_1_]]#0, [[VAR_36_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_38_:%.+]] = arith.addf [[LOAD_VAR_15_MEM_1_]], [[LOAD_VAR_16_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_9_MEM_2_:%.+]] = krnl.load [[VAR_9_]]#2{{.}}[[VAR_18_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_9_MEM_3_:%.+]] = krnl.load [[VAR_9_]]#6{{.}}[[VAR_18_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_41_:%.+]] = arith.addf [[VAR_38_]], [[LOAD_VAR_9_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_42_:%.+]] = arith.addf [[VAR_41_]], [[LOAD_VAR_9_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_1_:%.+]] = krnl.load [[VAR_11_]]#2{{.}}[[VAR_18_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_44_:%.+]] = arith.mulf [[LOAD_VAR_11_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_45_:%.+]] = arith.addf [[VAR_42_]], [[VAR_44_]] : f32
// CHECK:               [[VAR_46_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_45_]] : f32
// CHECK:               [[VAR_47_:%.+]] = math.exp [[VAR_46_]] : f32
// CHECK:               [[VAR_48_:%.+]] = arith.addf [[VAR_47_]], [[VAR_cst_]] : f32
// CHECK-DAG:           [[VAR_49_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_48_]] : f32
// CHECK-DAG:           [[VAR_50_:%.+]] = affine.apply #map1(){{.}}[[VAR_18_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_15_MEM_2_:%.+]] = krnl.load [[VAR_15_]]{{.}}[[VAR_18_1_]]#0, [[VAR_50_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_52_:%.+]] = affine.apply #map1(){{.}}[[VAR_18_1_]]#1]
// CHECK:               [[LOAD_VAR_16_MEM_2_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_18_1_]]#0, [[VAR_52_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_54_:%.+]] = arith.addf [[LOAD_VAR_15_MEM_2_]], [[LOAD_VAR_16_MEM_2_]] : f32
// CHECK-DAG:           [[LOAD_VAR_9_MEM_4_:%.+]] = krnl.load [[VAR_9_]]#3{{.}}[[VAR_18_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_9_MEM_5_:%.+]] = krnl.load [[VAR_9_]]#7{{.}}[[VAR_18_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_57_:%.+]] = arith.addf [[VAR_54_]], [[LOAD_VAR_9_MEM_4_]] : f32
// CHECK:               [[VAR_58_:%.+]] = arith.addf [[VAR_57_]], [[LOAD_VAR_9_MEM_5_]] : f32
// CHECK-DAG:           [[VAR_59_:%.+]] = math.tanh [[VAR_58_]] : f32
// CHECK-DAG:           [[VAR_60_:%.+]] = arith.mulf [[VAR_49_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_61_:%.+]] = arith.mulf [[VAR_33_]], [[VAR_59_]] : f32
// CHECK-DAG:           [[VAR_62_:%.+]] = arith.addf [[VAR_60_]], [[VAR_61_]] : f32
// CHECK-DAG:           [[VAR_63_:%.+]] = affine.apply #map2(){{.}}[[VAR_18_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_15_MEM_3_:%.+]] = krnl.load [[VAR_15_]]{{.}}[[VAR_18_1_]]#0, [[VAR_63_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_65_:%.+]] = affine.apply #map2(){{.}}[[VAR_18_1_]]#1]
// CHECK:               [[LOAD_VAR_16_MEM_3_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_18_1_]]#0, [[VAR_65_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_67_:%.+]] = arith.addf [[LOAD_VAR_15_MEM_3_]], [[LOAD_VAR_16_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_9_MEM_6_:%.+]] = krnl.load [[VAR_9_]]#1{{.}}[[VAR_18_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_9_MEM_7_:%.+]] = krnl.load [[VAR_9_]]#5{{.}}[[VAR_18_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_70_:%.+]] = arith.addf [[VAR_67_]], [[LOAD_VAR_9_MEM_6_]] : f32
// CHECK-DAG:           [[VAR_71_:%.+]] = arith.addf [[VAR_70_]], [[LOAD_VAR_9_MEM_7_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_2_:%.+]] = krnl.load [[VAR_11_]]#1{{.}}[[VAR_18_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_73_:%.+]] = arith.mulf [[LOAD_VAR_11_MEM_2_]], [[VAR_62_]] : f32
// CHECK:               [[VAR_74_:%.+]] = arith.addf [[VAR_71_]], [[VAR_73_]] : f32
// CHECK:               [[VAR_75_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_74_]] : f32
// CHECK:               [[VAR_76_:%.+]] = math.exp [[VAR_75_]] : f32
// CHECK:               [[VAR_77_:%.+]] = arith.addf [[VAR_76_]], [[VAR_cst_]] : f32
// CHECK-DAG:           [[VAR_78_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_77_]] : f32
// CHECK-DAG:           [[VAR_79_:%.+]] = math.tanh [[VAR_62_]] : f32
// CHECK:               [[VAR_80_:%.+]] = arith.mulf [[VAR_78_]], [[VAR_79_]] : f32
// CHECK:               krnl.store [[VAR_62_]], [[RES_2_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1] : memref<2x4xf32>
// CHECK:               krnl.store [[VAR_80_]], [[RES_1_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_1_]], [[VAR_c32_i64_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
// CHECK:           return [[RES_]] : memref<1x2x4xf32>
// CHECK:         }
}

// -----

func private @test_lstm_forward_mode_constant_weight_and_bias(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x2x4xf32>, %arg2: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %w = "onnx.Constant"() {value = dense<[[[1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]]> : tensor<1x16x3xf32>} : () -> tensor<1x16x3xf32> 
  %r = "onnx.Constant"() {value = dense<[[[2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.]]]> : tensor<1x16x4xf32>} : () -> tensor<1x16x4xf32> 
  %b = "onnx.Constant"() {value = dense<[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32.]]> : tensor<1x32xf32>} : () -> tensor<1x32xf32> 
  %p = "onnx.Constant"() {value = dense<[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]]> : tensor<1x12xf32>} : () -> tensor<1x12xf32> 

  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %w, %r, %b, %cst, %arg1, %arg2, %p) {hidden_size = 4 : si64} : (tensor<7x2x3xf32>, tensor<1x16x3xf32>, tensor<1x16x4xf32>, tensor<1x32xf32>, none, tensor<1x2x4xf32>, tensor<1x2x4xf32>, tensor<1x12xf32>) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

// CHECK-DAG: #map0 = affine_map<()[s0] -> (s0 + 8)>
// CHECK-DAG: #map1 = affine_map<()[s0] -> (s0 + 12)>
// CHECK-DAG: #map2 = affine_map<()[s0] -> (s0 + 4)>
// CHECK-LABEL:  func private @test_lstm_forward_mode_constant_weight_and_bias
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x2x4xf32>, [[PARAM_2_:%.+]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[VAR_c32_i64_:%.+]] = arith.constant 32 : i64
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_c0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_1_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:             [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_c0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_2_MEM_]], [[RES_2_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_4_:%.+]] = "krnl.global"() {name = "constant_6", shape = [3, 16], value = dense<1.000000e+00> : tensor<3x16xf32>} : () -> memref<3x16xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "krnl.global"() {name = "constant_7", shape = [4, 16], value = dense<2.000000e+00> : tensor<4x16xf32>} : () -> memref<4x16xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "krnl.global"() {name = "constant_9", shape = [4], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "krnl.global"() {name = "constant_10", shape = [4], value = dense<[5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "krnl.global"() {name = "constant_11", shape = [4], value = dense<[9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "krnl.global"() {name = "constant_12", shape = [4], value = dense<[1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "krnl.global"() {name = "constant_13", shape = [4], value = dense<[1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "krnl.global"() {name = "constant_14", shape = [4], value = dense<[2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "krnl.global"() {name = "constant_15", shape = [4], value = dense<[2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "krnl.global"() {name = "constant_16", shape = [4], value = dense<[2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = "krnl.global"() {name = "constant_18", shape = [4], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = "krnl.global"() {name = "constant_19", shape = [4], value = dense<[5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "krnl.global"() {name = "constant_20", shape = [4], value = dense<[9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7){
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]){
// CHECK:               [[VAR_23_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_23_]]#0, [[VAR_23_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_3_]]{{.}}[[VAR_23_]]#0, [[VAR_23_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_20_:%.+]] = "onnx.MatMul"([[RES_3_]], [[VAR_4_]]) : (memref<2x3xf32>, memref<3x16xf32>) -> memref<2x16xf32>
// CHECK-DAG:         [[VAR_21_:%.+]] = "onnx.MatMul"([[RES_1_]], [[VAR_5_]]) : (memref<2x4xf32>, memref<4x16xf32>) -> memref<2x16xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[VAR_23_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[RES_2_]]3#0, [[RES_2_]]3#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_20_MEM_:%.+]] = krnl.load [[VAR_20_]]{{.}}[[VAR_23_1_]]#0, [[VAR_23_1_]]#1] : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_VAR_21_MEM_:%.+]] = krnl.load [[VAR_21_]]{{.}}[[VAR_23_1_]]#0, [[VAR_23_1_]]#1] : memref<2x16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_27_:%.+]] = arith.addf [[LOAD_VAR_20_MEM_]], [[LOAD_VAR_21_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_6_MEM_:%.+]] = krnl.load [[VAR_6_]]{{.}}[[VAR_23_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_10_MEM_:%.+]] = krnl.load [[VAR_10_]]{{.}}[[VAR_23_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_30_:%.+]] = arith.addf [[VAR_27_]], [[LOAD_VAR_6_MEM_]] : f32
// CHECK-DAG:           [[VAR_31_:%.+]] = arith.addf [[VAR_30_]], [[LOAD_VAR_10_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_14_MEM_:%.+]] = krnl.load [[VAR_14_]]{{.}}[[VAR_23_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_33_:%.+]] = arith.mulf [[LOAD_VAR_14_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_34_:%.+]] = arith.addf [[VAR_31_]], [[VAR_33_]] : f32
// CHECK:               [[VAR_35_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_34_]] : f32
// CHECK:               [[VAR_36_:%.+]] = math.exp [[VAR_35_]] : f32
// CHECK:               [[VAR_37_:%.+]] = arith.addf [[VAR_36_]], [[VAR_cst_]] : f32
// CHECK-DAG:           [[VAR_38_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_37_]] : f32
// CHECK-DAG:           [[VAR_39_:%.+]] = affine.apply #map0(){{.}}[[VAR_23_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_20_MEM_1_:%.+]] = krnl.load [[VAR_20_]]{{.}}[[VAR_23_1_]]#0, [[VAR_39_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_41_:%.+]] = affine.apply #map0(){{.}}[[VAR_23_1_]]#1]
// CHECK:               [[LOAD_VAR_21_MEM_1_:%.+]] = krnl.load [[VAR_21_]]{{.}}[[VAR_23_1_]]#0, [[VAR_41_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_43_:%.+]] = arith.addf [[LOAD_VAR_20_MEM_1_]], [[LOAD_VAR_21_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_8_MEM_:%.+]] = krnl.load [[VAR_8_]]{{.}}[[VAR_23_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_12_MEM_:%.+]] = krnl.load [[VAR_12_]]{{.}}[[VAR_23_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_46_:%.+]] = arith.addf [[VAR_43_]], [[LOAD_VAR_8_MEM_]] : f32
// CHECK-DAG:           [[VAR_47_:%.+]] = arith.addf [[VAR_46_]], [[LOAD_VAR_12_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_16_MEM_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_23_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_49_:%.+]] = arith.mulf [[LOAD_VAR_16_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_50_:%.+]] = arith.addf [[VAR_47_]], [[VAR_49_]] : f32
// CHECK:               [[VAR_51_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_50_]] : f32
// CHECK:               [[VAR_52_:%.+]] = math.exp [[VAR_51_]] : f32
// CHECK:               [[VAR_53_:%.+]] = arith.addf [[VAR_52_]], [[VAR_cst_]] : f32
// CHECK-DAG:           [[VAR_54_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_53_]] : f32
// CHECK-DAG:           [[VAR_55_:%.+]] = affine.apply #map1(){{.}}[[VAR_23_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_20_MEM_2_:%.+]] = krnl.load [[VAR_20_]]{{.}}[[VAR_23_1_]]#0, [[VAR_55_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_57_:%.+]] = affine.apply #map1(){{.}}[[VAR_23_1_]]#1]
// CHECK:               [[LOAD_VAR_21_MEM_2_:%.+]] = krnl.load [[VAR_21_]]{{.}}[[VAR_23_1_]]#0, [[VAR_57_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_59_:%.+]] = arith.addf [[LOAD_VAR_20_MEM_2_]], [[LOAD_VAR_21_MEM_2_]] : f32
// CHECK-DAG:           [[LOAD_VAR_9_MEM_:%.+]] = krnl.load [[VAR_9_]]{{.}}[[VAR_23_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_13_MEM_:%.+]] = krnl.load [[VAR_13_]]{{.}}[[VAR_23_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_62_:%.+]] = arith.addf [[VAR_59_]], [[LOAD_VAR_9_MEM_]] : f32
// CHECK:               [[VAR_63_:%.+]] = arith.addf [[VAR_62_]], [[LOAD_VAR_13_MEM_]] : f32
// CHECK-DAG:           [[VAR_64_:%.+]] = math.tanh [[VAR_63_]] : f32
// CHECK-DAG:           [[VAR_65_:%.+]] = arith.mulf [[VAR_54_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_66_:%.+]] = arith.mulf [[VAR_38_]], [[VAR_64_]] : f32
// CHECK-DAG:           [[VAR_67_:%.+]] = arith.addf [[VAR_65_]], [[VAR_66_]] : f32
// CHECK-DAG:           [[VAR_68_:%.+]] = affine.apply #map2(){{.}}[[VAR_23_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_20_MEM_3_:%.+]] = krnl.load [[VAR_20_]]{{.}}[[VAR_23_1_]]#0, [[VAR_68_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_70_:%.+]] = affine.apply #map2(){{.}}[[VAR_23_1_]]#1]
// CHECK:               [[LOAD_VAR_21_MEM_3_:%.+]] = krnl.load [[VAR_21_]]{{.}}[[VAR_23_1_]]#0, [[VAR_70_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_72_:%.+]] = arith.addf [[LOAD_VAR_20_MEM_3_]], [[LOAD_VAR_21_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_7_MEM_:%.+]] = krnl.load [[VAR_7_]]{{.}}[[VAR_23_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_11_MEM_:%.+]] = krnl.load [[VAR_11_]]{{.}}[[VAR_23_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_75_:%.+]] = arith.addf [[VAR_72_]], [[LOAD_VAR_7_MEM_]] : f32
// CHECK-DAG:           [[VAR_76_:%.+]] = arith.addf [[VAR_75_]], [[LOAD_VAR_11_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_15_MEM_:%.+]] = krnl.load [[VAR_15_]]{{.}}[[VAR_23_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_78_:%.+]] = arith.mulf [[LOAD_VAR_15_MEM_]], [[VAR_67_]] : f32
// CHECK:               [[VAR_79_:%.+]] = arith.addf [[VAR_76_]], [[VAR_78_]] : f32
// CHECK:               [[VAR_80_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_79_]] : f32
// CHECK:               [[VAR_81_:%.+]] = math.exp [[VAR_80_]] : f32
// CHECK:               [[VAR_82_:%.+]] = arith.addf [[VAR_81_]], [[VAR_cst_]] : f32
// CHECK-DAG:           [[VAR_83_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_82_]] : f32
// CHECK-DAG:           [[VAR_84_:%.+]] = math.tanh [[VAR_67_]] : f32
// CHECK:               [[VAR_85_:%.+]] = arith.mulf [[VAR_83_]], [[VAR_84_]] : f32
// CHECK:               krnl.store [[VAR_67_]], [[RES_2_]]{{.}}[[RES_2_]]3#0, [[RES_2_]]3#1] : memref<2x4xf32>
// CHECK:               krnl.store [[VAR_85_]], [[RES_1_]]{{.}}[[VAR_23_1_]]#0, [[VAR_23_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_1_]], [[VAR_c32_i64_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
// CHECK:           return [[RES_]] : memref<1x2x4xf32>
// CHECK:         }
}

// -----

func private @test_lstm_reverse_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x16x3xf32>, %arg2: tensor<1x16x4xf32>, %arg3: tensor<1x32xf32>, %arg4: tensor<1x2x4xf32>, %arg5: tensor<1x2x4xf32>, %arg6: tensor<1x12xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4, %arg5, %arg6) {hidden_size = 4 : si64, direction = "reverse"} : (tensor<7x2x3xf32>, tensor<1x16x3xf32>, tensor<1x16x4xf32>, tensor<1x32xf32>, none, tensor<1x2x4xf32>, tensor<1x2x4xf32>, tensor<1x12xf32>) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

// CHECK-DAG: #map0 = affine_map<(d0) -> (-d0 + 6)>
// CHECK-DAG: #map1 = affine_map<()[s0] -> (s0 + 8)>
// CHECK-DAG: #map2 = affine_map<()[s0] -> (s0 + 12)>
// CHECK-DAG: #map3 = affine_map<()[s0] -> (s0 + 4)>
// CHECK-LABEL:  func private @test_lstm_reverse_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x16x3xf32>, [[PARAM_2_:%.+]]: memref<1x16x4xf32>, [[PARAM_3_:%.+]]: memref<1x32xf32>, [[PARAM_4_:%.+]]: memref<1x2x4xf32>, [[PARAM_5_:%.+]]: memref<1x2x4xf32>, [[PARAM_6_:%.+]]: memref<1x12xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[VAR_c32_i64_:%.+]] = arith.constant 32 : i64
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[UCC_PARAM_6:%.+]] = builtin.unrealized_conversion_cast %arg6 : memref<1x12xf32> to tensor<1x12xf32>
// CHECK-DAG:       [[UCC_PARAM_3:%.+]] = builtin.unrealized_conversion_cast %arg3 : memref<1x32xf32> to tensor<1x32xf32>
// CHECK-DAG:       [[UCC_PARAM_2:%.+]] = builtin.unrealized_conversion_cast %arg2 : memref<1x16x4xf32> to tensor<1x16x4xf32>
// CHECK-DAG:       [[UCC_PARAM_1:%.+]] = builtin.unrealized_conversion_cast %arg1 : memref<1x16x3xf32> to tensor<1x16x3xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[VAR_c0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:             [[LOAD_PARAM_5_MEM_:%.+]] = krnl.load [[PARAM_5_]]{{.}}[[VAR_c0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_5_MEM_]], [[RES_2_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.SqueezeV11"([[UCC_PARAM_1]]) {axes = [0]} : (tensor<1x16x3xf32>) -> memref<16x3xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.SqueezeV11"([[UCC_PARAM_2]]) {axes = [0]} : (tensor<1x16x4xf32>) -> memref<16x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Transpose"([[VAR_4_]]) {perm = [1, 0]} : (memref<16x3xf32>) -> memref<3x16xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Transpose"([[VAR_5_]]) {perm = [1, 0]} : (memref<16x4xf32>) -> memref<4x16xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.SqueezeV11"([[UCC_PARAM_3]]) {axes = [0]} : (tensor<1x32xf32>) -> memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_9_:%.+]]:8 = "onnx.SplitV11"([[VAR_8_]]) {axis = 0 : si64} : (memref<32xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.SqueezeV11"([[UCC_PARAM_6]]) {axes = [0]} : (tensor<1x12xf32>) -> memref<12xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]]:3 = "onnx.SplitV11"([[VAR_10_]]) {axis = 0 : si64} : (memref<12xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7){
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_1_:%.+]] = affine.apply #map0([[I_2_]])
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]){
// CHECK:               [[VAR_19_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[LOAD_PARAM_4_MEM_1_]], [[VAR_19_]]#0, [[VAR_19_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_3_]]{{.}}[[VAR_19_]]#0, [[VAR_19_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_16_:%.+]] = "onnx.MatMul"([[RES_3_]], [[VAR_6_]]) : (memref<2x3xf32>, memref<3x16xf32>) -> memref<2x16xf32>
// CHECK-DAG:         [[VAR_17_:%.+]] = "onnx.MatMul"([[RES_1_]], [[VAR_7_]]) : (memref<2x4xf32>, memref<4x16xf32>) -> memref<2x16xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[VAR_19_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_19_1_]]#0, [[VAR_19_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_16_MEM_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_19_1_]]#0, [[VAR_19_1_]]#1] : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_VAR_17_MEM_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_19_1_]]#0, [[VAR_19_1_]]#1] : memref<2x16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_23_:%.+]] = arith.addf [[LOAD_VAR_16_MEM_]], [[LOAD_VAR_17_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_9_MEM_:%.+]] = krnl.load [[VAR_9_]]#0{{.}}[[VAR_19_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_9_MEM_1_:%.+]] = krnl.load [[VAR_9_]]#4{{.}}[[VAR_19_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_26_:%.+]] = arith.addf [[VAR_23_]], [[LOAD_VAR_9_MEM_]] : f32
// CHECK-DAG:           [[VAR_27_:%.+]] = arith.addf [[VAR_26_]], [[LOAD_VAR_9_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_:%.+]] = krnl.load [[VAR_11_]]#0{{.}}[[VAR_19_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_29_:%.+]] = arith.mulf [[LOAD_VAR_11_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_30_:%.+]] = arith.addf [[VAR_27_]], [[VAR_29_]] : f32
// CHECK:               [[VAR_31_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_30_]] : f32
// CHECK:               [[VAR_32_:%.+]] = math.exp [[VAR_31_]] : f32
// CHECK:               [[VAR_33_:%.+]] = arith.addf [[VAR_32_]], [[VAR_cst_]] : f32
// CHECK-DAG:           [[VAR_34_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_33_]] : f32
// CHECK-DAG:           [[VAR_35_:%.+]] = affine.apply #map1(){{.}}[[VAR_19_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_16_MEM_1_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_19_1_]]#0, [[VAR_35_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_37_:%.+]] = affine.apply #map1(){{.}}[[VAR_19_1_]]#1]
// CHECK:               [[LOAD_VAR_17_MEM_1_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_19_1_]]#0, [[VAR_37_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_39_:%.+]] = arith.addf [[LOAD_VAR_16_MEM_1_]], [[LOAD_VAR_17_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_9_MEM_2_:%.+]] = krnl.load [[VAR_9_]]#2{{.}}[[VAR_19_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_9_MEM_3_:%.+]] = krnl.load [[VAR_9_]]#6{{.}}[[VAR_19_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_42_:%.+]] = arith.addf [[VAR_39_]], [[LOAD_VAR_9_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_43_:%.+]] = arith.addf [[VAR_42_]], [[LOAD_VAR_9_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_1_:%.+]] = krnl.load [[VAR_11_]]#2{{.}}[[VAR_19_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_45_:%.+]] = arith.mulf [[LOAD_VAR_11_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_46_:%.+]] = arith.addf [[VAR_43_]], [[VAR_45_]] : f32
// CHECK:               [[VAR_47_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_46_]] : f32
// CHECK:               [[VAR_48_:%.+]] = math.exp [[VAR_47_]] : f32
// CHECK:               [[VAR_49_:%.+]] = arith.addf [[VAR_48_]], [[VAR_cst_]] : f32
// CHECK-DAG:           [[VAR_50_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_49_]] : f32
// CHECK-DAG:           [[VAR_51_:%.+]] = affine.apply #map2(){{.}}[[VAR_19_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_16_MEM_2_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_19_1_]]#0, [[VAR_51_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_53_:%.+]] = affine.apply #map2(){{.}}[[VAR_19_1_]]#1]
// CHECK:               [[LOAD_VAR_17_MEM_2_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_19_1_]]#0, [[VAR_53_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_55_:%.+]] = arith.addf [[LOAD_VAR_16_MEM_2_]], [[LOAD_VAR_17_MEM_2_]] : f32
// CHECK-DAG:           [[LOAD_VAR_9_MEM_4_:%.+]] = krnl.load [[VAR_9_]]#3{{.}}[[VAR_19_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_9_MEM_5_:%.+]] = krnl.load [[VAR_9_]]#7{{.}}[[VAR_19_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_58_:%.+]] = arith.addf [[VAR_55_]], [[LOAD_VAR_9_MEM_4_]] : f32
// CHECK:               [[VAR_59_:%.+]] = arith.addf [[VAR_58_]], [[LOAD_VAR_9_MEM_5_]] : f32
// CHECK-DAG:           [[VAR_60_:%.+]] = math.tanh [[VAR_59_]] : f32
// CHECK-DAG:           [[VAR_61_:%.+]] = arith.mulf [[VAR_50_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_62_:%.+]] = arith.mulf [[VAR_34_]], [[VAR_60_]] : f32
// CHECK-DAG:           [[VAR_63_:%.+]] = arith.addf [[VAR_61_]], [[VAR_62_]] : f32
// CHECK-DAG:           [[VAR_64_:%.+]] = affine.apply #map3(){{.}}[[VAR_19_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_16_MEM_3_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_19_1_]]#0, [[VAR_64_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_66_:%.+]] = affine.apply #map3(){{.}}[[VAR_19_1_]]#1]
// CHECK:               [[LOAD_VAR_17_MEM_3_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_19_1_]]#0, [[VAR_66_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_68_:%.+]] = arith.addf [[LOAD_VAR_16_MEM_3_]], [[LOAD_VAR_17_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_9_MEM_6_:%.+]] = krnl.load [[VAR_9_]]#1{{.}}[[VAR_19_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_9_MEM_7_:%.+]] = krnl.load [[VAR_9_]]#5{{.}}[[VAR_19_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_71_:%.+]] = arith.addf [[VAR_68_]], [[LOAD_VAR_9_MEM_6_]] : f32
// CHECK-DAG:           [[VAR_72_:%.+]] = arith.addf [[VAR_71_]], [[LOAD_VAR_9_MEM_7_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_2_:%.+]] = krnl.load [[VAR_11_]]#1{{.}}[[VAR_19_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_74_:%.+]] = arith.mulf [[LOAD_VAR_11_MEM_2_]], [[VAR_63_]] : f32
// CHECK:               [[VAR_75_:%.+]] = arith.addf [[VAR_72_]], [[VAR_74_]] : f32
// CHECK:               [[VAR_76_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_75_]] : f32
// CHECK:               [[VAR_77_:%.+]] = math.exp [[VAR_76_]] : f32
// CHECK:               [[VAR_78_:%.+]] = arith.addf [[VAR_77_]], [[VAR_cst_]] : f32
// CHECK-DAG:           [[VAR_79_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_78_]] : f32
// CHECK-DAG:           [[VAR_80_:%.+]] = math.tanh [[VAR_63_]] : f32
// CHECK:               [[VAR_81_:%.+]] = arith.mulf [[VAR_79_]], [[VAR_80_]] : f32
// CHECK:               krnl.store [[VAR_63_]], [[RES_2_]]{{.}}[[VAR_19_1_]]#0, [[VAR_19_1_]]#1] : memref<2x4xf32>
// CHECK:               krnl.store [[VAR_81_]], [[RES_1_]]{{.}}[[VAR_19_1_]]#0, [[VAR_19_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_1_]], [[VAR_c32_i64_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
// CHECK:           return [[RES_]] : memref<1x2x4xf32>
// CHECK:         }
}


// -----

func private @test_lstm_bidirectional_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<2x16x3xf32>, %arg2: tensor<2x16x4xf32>, %arg3: tensor<2x32xf32>, %arg4: tensor<2x2x4xf32>, %arg5: tensor<2x2x4xf32>, %arg6: tensor<2x12xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4, %arg5, %arg6) {hidden_size = 4 : si64, direction = "bidirectional"} : (tensor<7x2x3xf32>, tensor<2x16x3xf32>, tensor<2x16x4xf32>, tensor<2x32xf32>, none, tensor<2x2x4xf32>, tensor<2x2x4xf32>, tensor<2x12xf32>) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

// CHECK-DAG: #map0 = affine_map<()[s0] -> (s0 + 8)>
// CHECK-DAG: #map1 = affine_map<()[s0] -> (s0 + 12)>
// CHECK-DAG: #map2 = affine_map<()[s0] -> (s0 + 4)>
// CHECK-DAG: #map3 = affine_map<(d0) -> (-d0 + 6)>
// CHECK-LABEL:  func private @test_lstm_bidirectional_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<2x16x3xf32>, [[PARAM_2_:%.+]]: memref<2x16x4xf32>, [[PARAM_3_:%.+]]: memref<2x32xf32>, [[PARAM_4_:%.+]]: memref<2x2x4xf32>, [[PARAM_5_:%.+]]: memref<2x2x4xf32>, [[PARAM_6_:%.+]]: memref<2x12xf32>) -> memref<2x2x4xf32> {
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[UCC_PARAM_6:%.+]] = builtin.unrealized_conversion_cast %arg6 : memref<2x12xf32> to tensor<2x12xf32>
// CHECK-DAG:       [[UCC_PARAM_3:%.+]] = builtin.unrealized_conversion_cast %arg3 : memref<2x32xf32> to tensor<2x32xf32>
// CHECK-DAG:       [[UCC_PARAM_2:%.+]] = builtin.unrealized_conversion_cast %arg2 : memref<2x16x4xf32> to tensor<2x16x4xf32>
// CHECK-DAG:       [[UCC_PARAM_1:%.+]] = builtin.unrealized_conversion_cast %arg1 : memref<2x16x3xf32> to tensor<2x16x3xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[VAR_c0_]], [[I_0_]], [[I_1_]]{{.}} : memref<2x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:             [[LOAD_PARAM_5_MEM_:%.+]] = krnl.load [[PARAM_5_]]{{.}}[[VAR_c0_]], [[I_0_]], [[I_1_]]{{.}} : memref<2x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_5_MEM_]], [[RES_2_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:             [[LOAD_PARAM_4_MEM_1_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[VAR_c1_]], [[I_0_]], [[I_1_]]{{.}} : memref<2x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_1_]], [[RES_3_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:             [[LOAD_PARAM_5_MEM_1_:%.+]] = krnl.load [[PARAM_5_]]{{.}}[[VAR_c1_]], [[I_0_]], [[I_1_]]{{.}} : memref<2x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_5_MEM_1_]], [[RES_4_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:           }
// CHECK:           [[VAR_6_:%.+]]:2 = "onnx.SplitV11"([[UCC_PARAM_1]]) {axis = 0 : si64} : (tensor<2x16x3xf32>) -> (memref<1x16x3xf32>, memref<1x16x3xf32>)
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.SqueezeV11"([[VAR_6_]]#0) {axes = [0]} : (memref<1x16x3xf32>) -> memref<16x3xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.SqueezeV11"([[VAR_6_]]#1) {axes = [0]} : (memref<1x16x3xf32>) -> memref<16x3xf32>
// CHECK-DAG:       [[VAR_9_:%.+]]:2 = "onnx.SplitV11"([[UCC_PARAM_2]]) {axis = 0 : si64} : (tensor<2x16x4xf32>) -> (memref<1x16x4xf32>, memref<1x16x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.SqueezeV11"([[VAR_9_]]#0) {axes = [0]} : (memref<1x16x4xf32>) -> memref<16x4xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.SqueezeV11"([[VAR_9_]]#1) {axes = [0]} : (memref<1x16x4xf32>) -> memref<16x4xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Transpose"([[VAR_7_]]) {perm = [1, 0]} : (memref<16x3xf32>) -> memref<3x16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Transpose"([[VAR_10_]]) {perm = [1, 0]} : (memref<16x4xf32>) -> memref<4x16xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.Transpose"([[VAR_8_]]) {perm = [1, 0]} : (memref<16x3xf32>) -> memref<3x16xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.Transpose"([[VAR_11_]]) {perm = [1, 0]} : (memref<16x4xf32>) -> memref<4x16xf32>
// CHECK-DAG:       [[VAR_16_:%.+]]:2 = "onnx.SplitV11"([[UCC_PARAM_3]]) {axis = 0 : si64} : (tensor<2x32xf32>) -> (memref<1x32xf32>, memref<1x32xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.SqueezeV11"([[VAR_16_]]#0) {axes = [0]} : (memref<1x32xf32>) -> memref<32xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.SqueezeV11"([[VAR_16_]]#1) {axes = [0]} : (memref<1x32xf32>) -> memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_19_:%.+]]:8 = "onnx.SplitV11"([[VAR_17_]]) {axis = 0 : si64} : (memref<32xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[VAR_20_:%.+]]:8 = "onnx.SplitV11"([[VAR_18_]]) {axis = 0 : si64} : (memref<32xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[VAR_21_:%.+]]:2 = "onnx.SplitV11"([[UCC_PARAM_6]]) {axis = 0 : si64} : (tensor<2x12xf32>) -> (memref<1x12xf32>, memref<1x12xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_22_:%.+]] = "onnx.SqueezeV11"([[VAR_21_]]#0) {axes = [0]} : (memref<1x12xf32>) -> memref<12xf32>
// CHECK-DAG:       [[VAR_23_:%.+]] = "onnx.SqueezeV11"([[VAR_21_]]#1) {axes = [0]} : (memref<1x12xf32>) -> memref<12xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_24_:%.+]]:3 = "onnx.SplitV11"([[VAR_22_]]) {axis = 0 : si64} : (memref<12xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[VAR_25_:%.+]]:3 = "onnx.SplitV11"([[VAR_23_]]) {axis = 0 : si64} : (memref<12xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7){
// CHECK-DAG:         [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]){
// CHECK:               [[VAR_34_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_34_]]#0, [[VAR_34_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_5_]]{{.}}[[VAR_34_]]#0, [[VAR_34_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_1_:%.+]] = "onnx.MatMul"([[RES_5_]], [[VAR_12_]]) : (memref<2x3xf32>, memref<3x16xf32>) -> memref<2x16xf32>
// CHECK-DAG:         [[LOAD_PARAM_5_MEM_1_:%.+]] = "onnx.MatMul"([[RES_1_]], [[VAR_13_]]) : (memref<2x4xf32>, memref<4x16xf32>) -> memref<2x16xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[VAR_34_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_34_1_]]#0, [[VAR_34_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_LOAD_PARAM_4_MEM_1_MEM_:%.+]] = krnl.load [[LOAD_PARAM_4_MEM_1_]]{{.}}[[VAR_34_1_]]#0, [[VAR_34_1_]]#1] : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_LOAD_PARAM_5_MEM_1_MEM_:%.+]] = krnl.load [[LOAD_PARAM_5_MEM_1_]]{{.}}[[VAR_34_1_]]#0, [[VAR_34_1_]]#1] : memref<2x16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_38_:%.+]] = arith.addf [[LOAD_LOAD_PARAM_4_MEM_1_MEM_]], [[LOAD_LOAD_PARAM_5_MEM_1_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_19_MEM_:%.+]] = krnl.load [[VAR_19_]]#0{{.}}[[VAR_34_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_19_MEM_1_:%.+]] = krnl.load [[VAR_19_]]#4{{.}}[[VAR_34_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_41_:%.+]] = arith.addf [[VAR_38_]], [[LOAD_VAR_19_MEM_]] : f32
// CHECK-DAG:           [[VAR_42_:%.+]] = arith.addf [[VAR_41_]], [[LOAD_VAR_19_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_24_MEM_:%.+]] = krnl.load [[VAR_24_]]#0{{.}}[[VAR_34_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_44_:%.+]] = arith.mulf [[LOAD_VAR_24_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_45_:%.+]] = arith.addf [[VAR_42_]], [[VAR_44_]] : f32
// CHECK:               [[VAR_46_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_45_]] : f32
// CHECK:               [[VAR_47_:%.+]] = math.exp [[VAR_46_]] : f32
// CHECK:               [[VAR_48_:%.+]] = arith.addf [[VAR_47_]], [[VAR_cst_]] : f32
// CHECK-DAG:           [[VAR_49_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_48_]] : f32
// CHECK-DAG:           [[VAR_50_:%.+]] = affine.apply #map0(){{.}}[[VAR_34_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_LOAD_PARAM_4_MEM_1_MEM_1_:%.+]] = krnl.load [[LOAD_PARAM_4_MEM_1_]]{{.}}[[VAR_34_1_]]#0, [[VAR_50_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_52_:%.+]] = affine.apply #map0(){{.}}[[VAR_34_1_]]#1]
// CHECK:               [[LOAD_LOAD_PARAM_5_MEM_1_MEM_1_:%.+]] = krnl.load [[LOAD_PARAM_5_MEM_1_]]{{.}}[[VAR_34_1_]]#0, [[VAR_52_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_54_:%.+]] = arith.addf [[LOAD_LOAD_PARAM_4_MEM_1_MEM_1_]], [[LOAD_LOAD_PARAM_5_MEM_1_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_19_MEM_2_:%.+]] = krnl.load [[VAR_19_]]#2{{.}}[[VAR_34_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_19_MEM_3_:%.+]] = krnl.load [[VAR_19_]]#6{{.}}[[VAR_34_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_57_:%.+]] = arith.addf [[VAR_54_]], [[LOAD_VAR_19_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_58_:%.+]] = arith.addf [[VAR_57_]], [[LOAD_VAR_19_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_24_MEM_1_:%.+]] = krnl.load [[VAR_24_]]#2{{.}}[[VAR_34_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_60_:%.+]] = arith.mulf [[LOAD_VAR_24_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_61_:%.+]] = arith.addf [[VAR_58_]], [[VAR_60_]] : f32
// CHECK:               [[VAR_62_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_61_]] : f32
// CHECK:               [[VAR_63_:%.+]] = math.exp [[VAR_62_]] : f32
// CHECK:               [[VAR_64_:%.+]] = arith.addf [[VAR_63_]], [[VAR_cst_]] : f32
// CHECK-DAG:           [[VAR_65_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_64_]] : f32
// CHECK-DAG:           [[VAR_66_:%.+]] = affine.apply #map1(){{.}}[[VAR_34_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_LOAD_PARAM_4_MEM_1_MEM_2_:%.+]] = krnl.load [[LOAD_PARAM_4_MEM_1_]]{{.}}[[VAR_34_1_]]#0, [[VAR_66_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_68_:%.+]] = affine.apply #map1(){{.}}[[VAR_34_1_]]#1]
// CHECK:               [[LOAD_LOAD_PARAM_5_MEM_1_MEM_2_:%.+]] = krnl.load [[LOAD_PARAM_5_MEM_1_]]{{.}}[[VAR_34_1_]]#0, [[VAR_68_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_70_:%.+]] = arith.addf [[LOAD_LOAD_PARAM_4_MEM_1_MEM_2_]], [[LOAD_LOAD_PARAM_5_MEM_1_MEM_2_]] : f32
// CHECK-DAG:           [[LOAD_VAR_19_MEM_4_:%.+]] = krnl.load [[VAR_19_]]#3{{.}}[[VAR_34_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_19_MEM_5_:%.+]] = krnl.load [[VAR_19_]]#7{{.}}[[VAR_34_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_73_:%.+]] = arith.addf [[VAR_70_]], [[LOAD_VAR_19_MEM_4_]] : f32
// CHECK:               [[VAR_74_:%.+]] = arith.addf [[VAR_73_]], [[LOAD_VAR_19_MEM_5_]] : f32
// CHECK-DAG:           [[VAR_75_:%.+]] = math.tanh [[VAR_74_]] : f32
// CHECK-DAG:           [[VAR_76_:%.+]] = arith.mulf [[VAR_65_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_77_:%.+]] = arith.mulf [[VAR_49_]], [[VAR_75_]] : f32
// CHECK-DAG:           [[VAR_78_:%.+]] = arith.addf [[VAR_76_]], [[VAR_77_]] : f32
// CHECK-DAG:           [[VAR_79_:%.+]] = affine.apply #map2(){{.}}[[VAR_34_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_LOAD_PARAM_4_MEM_1_MEM_3_:%.+]] = krnl.load [[LOAD_PARAM_4_MEM_1_]]{{.}}[[VAR_34_1_]]#0, [[VAR_79_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_81_:%.+]] = affine.apply #map2(){{.}}[[VAR_34_1_]]#1]
// CHECK:               [[LOAD_LOAD_PARAM_5_MEM_1_MEM_3_:%.+]] = krnl.load [[LOAD_PARAM_5_MEM_1_]]{{.}}[[VAR_34_1_]]#0, [[VAR_81_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_83_:%.+]] = arith.addf [[LOAD_LOAD_PARAM_4_MEM_1_MEM_3_]], [[LOAD_LOAD_PARAM_5_MEM_1_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_19_MEM_6_:%.+]] = krnl.load [[VAR_19_]]#1{{.}}[[VAR_34_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_19_MEM_7_:%.+]] = krnl.load [[VAR_19_]]#5{{.}}[[VAR_34_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_86_:%.+]] = arith.addf [[VAR_83_]], [[LOAD_VAR_19_MEM_6_]] : f32
// CHECK-DAG:           [[VAR_87_:%.+]] = arith.addf [[VAR_86_]], [[LOAD_VAR_19_MEM_7_]] : f32
// CHECK-DAG:           [[LOAD_VAR_24_MEM_2_:%.+]] = krnl.load [[VAR_24_]]#1{{.}}[[VAR_34_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_89_:%.+]] = arith.mulf [[LOAD_VAR_24_MEM_2_]], [[VAR_78_]] : f32
// CHECK:               [[VAR_90_:%.+]] = arith.addf [[VAR_87_]], [[VAR_89_]] : f32
// CHECK:               [[VAR_91_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_90_]] : f32
// CHECK:               [[VAR_92_:%.+]] = math.exp [[VAR_91_]] : f32
// CHECK:               [[VAR_93_:%.+]] = arith.addf [[VAR_92_]], [[VAR_cst_]] : f32
// CHECK-DAG:           [[VAR_94_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_93_]] : f32
// CHECK-DAG:           [[VAR_95_:%.+]] = math.tanh [[VAR_78_]] : f32
// CHECK:               [[VAR_96_:%.+]] = arith.mulf [[VAR_94_]], [[VAR_95_]] : f32
// CHECK:               krnl.store [[VAR_78_]], [[RES_2_]]{{.}}[[VAR_34_1_]]#0, [[VAR_34_1_]]#1] : memref<2x4xf32>
// CHECK:               krnl.store [[VAR_96_]], [[RES_1_]]{{.}}[[VAR_34_1_]]#0, [[VAR_34_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_4_]]) with ([[LOOP_4_]] -> [[I_7_:%.+]] = 0 to 7){
// CHECK-DAG:         [[RES_5_:%.+]] = affine.apply #map3([[I_7_]])
// CHECK-DAG:         [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_5_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_5_]]#0, [[LOOP_5_]]#1) with ([[LOOP_5_]]#0 -> [[I_8_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_5_]]#1 -> [[I_9_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]){
// CHECK:               [[LOAD_PARAM_0_MEM_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_5_]]#0, [[LOOP_5_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[RES_5_]], [[LOAD_PARAM_0_MEM_1_]]#0, [[LOAD_PARAM_0_MEM_1_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_2_]], [[RES_6_]]{{.}}[[LOAD_PARAM_0_MEM_1_]]#0, [[LOAD_PARAM_0_MEM_1_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[LOAD_PARAM_5_MEM_1_1_:%.+]] = "onnx.MatMul"([[RES_6_]], [[VAR_14_]]) : (memref<2x3xf32>, memref<3x16xf32>) -> memref<2x16xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]] = "onnx.MatMul"([[RES_3_]], [[VAR_15_]]) : (memref<2x4xf32>, memref<4x16xf32>) -> memref<2x16xf32>
// CHECK-DAG:         [[LOOP_6_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_6_]]#0, [[LOOP_6_]]#1) with ([[LOOP_6_]]#0 -> [[I_10_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_6_]]#1 -> [[I_11_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[LOAD_PARAM_0_MEM_1_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_6_]]#0, [[LOOP_6_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[RES_4_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_LOAD_PARAM_5_MEM_1_MEM_4_:%.+]] = krnl.load [[LOAD_PARAM_5_MEM_1_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x16xf32>
// CHECK-DAG:           [[VAR_38_1_:%.+]] = krnl.load [[LOOP_3_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_19_MEM_8_:%.+]] = arith.addf [[LOAD_LOAD_PARAM_5_MEM_1_MEM_4_]], [[VAR_38_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_19_MEM_1_:%.+]] = krnl.load [[VAR_20_]]#0{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_41_1_:%.+]] = krnl.load [[VAR_20_]]#4{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_42_1_:%.+]] = arith.addf [[LOAD_VAR_19_MEM_8_]], [[LOAD_VAR_19_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_24_MEM_3_:%.+]] = arith.addf [[VAR_42_1_]], [[VAR_41_1_]] : f32
// CHECK-DAG:           [[VAR_44_1_:%.+]] = krnl.load [[VAR_25_]]#0{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_45_1_:%.+]] = arith.mulf [[VAR_44_1_]], [[LOAD_PARAM_0_MEM_2_]] : f32
// CHECK:               [[VAR_46_1_:%.+]] = arith.addf [[LOAD_VAR_24_MEM_3_]], [[VAR_45_1_]] : f32
// CHECK:               [[VAR_47_1_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_46_1_]] : f32
// CHECK:               [[VAR_48_1_:%.+]] = math.exp [[VAR_47_1_]] : f32
// CHECK:               [[VAR_49_1_:%.+]] = arith.addf [[VAR_48_1_]], [[VAR_cst_]] : f32
// CHECK-DAG:           [[VAR_50_1_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_49_1_]] : f32
// CHECK-DAG:           [[LOAD_LOAD_PARAM_4_MEM_1_MEM_1_:%.+]] = affine.apply #map0(){{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_52_1_:%.+]] = krnl.load [[LOAD_PARAM_5_MEM_1_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_LOAD_PARAM_4_MEM_1_MEM_1_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_LOAD_PARAM_5_MEM_1_MEM_1_:%.+]] = affine.apply #map0(){{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1]
// CHECK:               [[VAR_54_1_:%.+]] = krnl.load [[LOOP_3_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_LOAD_PARAM_5_MEM_1_MEM_1_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_VAR_19_MEM_2_:%.+]] = arith.addf [[VAR_52_1_]], [[VAR_54_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_19_MEM_3_:%.+]] = krnl.load [[VAR_20_]]#2{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_57_1_:%.+]] = krnl.load [[VAR_20_]]#6{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_58_1_:%.+]] = arith.addf [[LOAD_VAR_19_MEM_2_]], [[LOAD_VAR_19_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_24_MEM_1_:%.+]] = arith.addf [[VAR_58_1_]], [[VAR_57_1_]] : f32
// CHECK-DAG:           [[VAR_60_1_:%.+]] = krnl.load [[VAR_25_]]#2{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_61_1_:%.+]] = arith.mulf [[VAR_60_1_]], [[LOAD_PARAM_0_MEM_2_]] : f32
// CHECK:               [[VAR_62_1_:%.+]] = arith.addf [[LOAD_VAR_24_MEM_1_]], [[VAR_61_1_]] : f32
// CHECK:               [[VAR_63_1_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_62_1_]] : f32
// CHECK:               [[VAR_64_1_:%.+]] = math.exp [[VAR_63_1_]] : f32
// CHECK:               [[VAR_65_1_:%.+]] = arith.addf [[VAR_64_1_]], [[VAR_cst_]] : f32
// CHECK-DAG:           [[VAR_66_1_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_65_1_]] : f32
// CHECK-DAG:           [[LOAD_LOAD_PARAM_4_MEM_1_MEM_2_:%.+]] = affine.apply #map1(){{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_68_1_:%.+]] = krnl.load [[LOAD_PARAM_5_MEM_1_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_LOAD_PARAM_4_MEM_1_MEM_2_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_LOAD_PARAM_5_MEM_1_MEM_2_:%.+]] = affine.apply #map1(){{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1]
// CHECK:               [[VAR_70_1_:%.+]] = krnl.load [[LOOP_3_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_LOAD_PARAM_5_MEM_1_MEM_2_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_VAR_19_MEM_4_:%.+]] = arith.addf [[VAR_68_1_]], [[VAR_70_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_19_MEM_5_:%.+]] = krnl.load [[VAR_20_]]#3{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_73_1_:%.+]] = krnl.load [[VAR_20_]]#7{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_74_1_:%.+]] = arith.addf [[LOAD_VAR_19_MEM_4_]], [[LOAD_VAR_19_MEM_5_]] : f32
// CHECK:               [[VAR_75_1_:%.+]] = arith.addf [[VAR_74_1_]], [[VAR_73_1_]] : f32
// CHECK-DAG:           [[VAR_76_1_:%.+]] = math.tanh [[VAR_75_1_]] : f32
// CHECK-DAG:           [[VAR_77_1_:%.+]] = arith.mulf [[VAR_66_1_]], [[LOAD_PARAM_0_MEM_2_]] : f32
// CHECK:               [[VAR_78_1_:%.+]] = arith.mulf [[VAR_50_1_]], [[VAR_76_1_]] : f32
// CHECK-DAG:           [[VAR_79_1_:%.+]] = arith.addf [[VAR_77_1_]], [[VAR_78_1_]] : f32
// CHECK-DAG:           [[LOAD_LOAD_PARAM_4_MEM_1_MEM_3_:%.+]] = affine.apply #map2(){{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_81_1_:%.+]] = krnl.load [[LOAD_PARAM_5_MEM_1_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_LOAD_PARAM_4_MEM_1_MEM_3_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_LOAD_PARAM_5_MEM_1_MEM_3_:%.+]] = affine.apply #map2(){{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1]
// CHECK:               [[VAR_83_1_:%.+]] = krnl.load [[LOOP_3_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_LOAD_PARAM_5_MEM_1_MEM_3_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_VAR_19_MEM_6_:%.+]] = arith.addf [[VAR_81_1_]], [[VAR_83_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_19_MEM_7_:%.+]] = krnl.load [[VAR_20_]]#1{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_86_1_:%.+]] = krnl.load [[VAR_20_]]#5{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_87_1_:%.+]] = arith.addf [[LOAD_VAR_19_MEM_6_]], [[LOAD_VAR_19_MEM_7_]] : f32
// CHECK-DAG:           [[LOAD_VAR_24_MEM_2_:%.+]] = arith.addf [[VAR_87_1_]], [[VAR_86_1_]] : f32
// CHECK-DAG:           [[VAR_89_1_:%.+]] = krnl.load [[VAR_25_]]#1{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_90_1_:%.+]] = arith.mulf [[VAR_89_1_]], [[VAR_79_1_]] : f32
// CHECK:               [[VAR_91_1_:%.+]] = arith.addf [[LOAD_VAR_24_MEM_2_]], [[VAR_90_1_]] : f32
// CHECK:               [[VAR_92_1_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_91_1_]] : f32
// CHECK:               [[VAR_93_1_:%.+]] = math.exp [[VAR_92_1_]] : f32
// CHECK:               [[VAR_94_1_:%.+]] = arith.addf [[VAR_93_1_]], [[VAR_cst_]] : f32
// CHECK-DAG:           [[VAR_95_1_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_94_1_]] : f32
// CHECK-DAG:           [[VAR_96_1_:%.+]] = math.tanh [[VAR_79_1_]] : f32
// CHECK:               [[VAR_97_:%.+]] = arith.mulf [[VAR_95_1_]], [[VAR_96_1_]] : f32
// CHECK:               krnl.store [[VAR_79_1_]], [[RES_4_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK:               krnl.store [[VAR_97_]], [[RES_3_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOOP_7_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_7_]]#0, [[LOOP_7_]]#1) with ([[LOOP_7_]]#0 -> [[I_12_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_7_]]#1 -> [[I_13_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:             [[RES_5_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_7_]]#0, [[LOOP_7_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[RES_6_:%.+]] = krnl.load [[RES_1_]]{{.}}[[RES_5_1_]]#0, [[RES_5_1_]]#1] : memref<2x4xf32>
// CHECK:             krnl.store [[RES_6_]], [[RES_]]{{.}}[[VAR_c0_]], [[RES_5_1_]]#0, [[RES_5_1_]]#1] : memref<2x2x4xf32>
// CHECK:             [[LOOP_5_:%.+]] = krnl.load [[RES_3_]]{{.}}[[RES_5_1_]]#0, [[RES_5_1_]]#1] : memref<2x4xf32>
// CHECK:             krnl.store [[LOOP_5_]], [[RES_]]{{.}}[[VAR_c1_]], [[RES_5_1_]]#0, [[RES_5_1_]]#1] : memref<2x2x4xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x2x4xf32>
// CHECK:         }
}

// -----

func private @test_lstm_unknown_dims(%arg0: tensor<?x?x?xf32>, %arg1: tensor<1x16x?xf32>, %arg2: tensor<1x16x4xf32>, %arg3: tensor<1x32xf32>, %arg4: tensor<1x?x4xf32>, %arg5: tensor<1x?x4xf32>, %arg6: tensor<1x12xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4, %arg5, %arg6) {hidden_size = 4 : si64} : (tensor<?x?x?xf32>, tensor<1x16x?xf32>, tensor<1x16x4xf32>, tensor<1x32xf32>, none, tensor<1x?x4xf32>, tensor<1x?x4xf32>, tensor<1x12xf32>) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

// CHECK-DAG: #map0 = affine_map<()[s0] -> (s0 + 8)>
// CHECK-DAG: #map1 = affine_map<()[s0] -> (s0 + 12)>
// CHECK-DAG: #map2 = affine_map<()[s0] -> (s0 + 4)>
// CHECK-LABEL:  func private @test_lstm_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x?xf32>, [[PARAM_1_:%.+]]: memref<1x16x?xf32>, [[PARAM_2_:%.+]]: memref<1x16x4xf32>, [[PARAM_3_:%.+]]: memref<1x32xf32>, [[PARAM_4_:%.+]]: memref<1x?x4xf32>, [[PARAM_5_:%.+]]: memref<1x?x4xf32>, [[PARAM_6_:%.+]]: memref<1x12xf32>) -> memref<1x?x4xf32> {
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c16_i64_:%.+]] = arith.constant 16 : i64
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[UCC_PARAM_6:%.+]] = builtin.unrealized_conversion_cast %arg6 : memref<1x12xf32> to tensor<1x12xf32>
// CHECK-DAG:       [[UCC_PARAM_3:%.+]] = builtin.unrealized_conversion_cast %arg3 : memref<1x32xf32> to tensor<1x32xf32>
// CHECK-DAG:       [[UCC_PARAM_2:%.+]] = builtin.unrealized_conversion_cast %arg2 : memref<1x16x4xf32> to tensor<1x16x4xf32>
// CHECK-DAG:       [[UCC_PARAM_1:%.+]] = builtin.unrealized_conversion_cast %arg1 : memref<1x16x?xf32> to tensor<1x16x?xf32>
// CHECK:           [[VAR_0_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<1x?x4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<?x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[VAR_2_]]) {{.*}}: memref<?x4xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<?x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc([[VAR_4_]]) {{.*}}: memref<?x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[VAR_2_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[VAR_c0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x?x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<?x4xf32>
// CHECK:             [[LOAD_PARAM_5_MEM_:%.+]] = krnl.load [[PARAM_5_]]{{.}}[[VAR_c0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x?x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_5_MEM_]], [[RES_2_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<?x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.SqueezeV11"([[UCC_PARAM_1]]) {axes = [0]} : (tensor<1x16x?xf32>) -> memref<16x?xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.SqueezeV11"([[UCC_PARAM_2]]) {axes = [0]} : (tensor<1x16x4xf32>) -> memref<16x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Transpose"([[VAR_7_]]) {perm = [1, 0]} : (memref<16x?xf32>) -> memref<?x16xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Transpose"([[VAR_8_]]) {perm = [1, 0]} : (memref<16x4xf32>) -> memref<4x16xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.SqueezeV11"([[UCC_PARAM_3]]) {axes = [0]} : (tensor<1x32xf32>) -> memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_12_:%.+]]:8 = "onnx.SplitV11"([[VAR_11_]]) {axis = 0 : si64} : (memref<32xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.SqueezeV11"([[UCC_PARAM_6]]) {axes = [0]} : (tensor<1x12xf32>) -> memref<12xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_14_:%.+]]:3 = "onnx.SplitV11"([[VAR_13_]]) {axis = 0 : si64} : (memref<12xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK-DAG:       [[VAR_16_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?x?x?xf32>
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to [[VAR_16_]]){
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_1_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<?x?x?xf32>
// CHECK-DAG:         [[LOAD_PARAM_5_MEM_1_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c2_]] : memref<?x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc([[LOAD_PARAM_4_MEM_1_]], [[LOAD_PARAM_5_MEM_1_]]) {{.*}}: memref<?x?xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[VAR_c0_]] to [[LOAD_PARAM_4_MEM_1_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[VAR_c0_]] to [[LOAD_PARAM_5_MEM_1_]]){
// CHECK:               [[VAR_26_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_26_]]#0, [[VAR_26_]]#1] : memref<?x?x?xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_3_]]{{.}}[[VAR_26_]]#0, [[VAR_26_]]#1] : memref<?x?xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_23_:%.+]] = "onnx.MatMul"([[RES_3_]], [[VAR_9_]]) : (memref<?x?xf32>, memref<?x16xf32>) -> memref<?x16xf32>
// CHECK-DAG:         [[VAR_24_:%.+]] = "onnx.MatMul"([[RES_1_]], [[VAR_10_]]) : (memref<?x4xf32>, memref<4x16xf32>) -> memref<?x16xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[VAR_c0_]] to [[VAR_2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[VAR_26_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<?x4xf32>
// CHECK-DAG:           [[LOAD_VAR_23_MEM_:%.+]] = krnl.load [[VAR_23_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<?x16xf32>
// CHECK-DAG:           [[LOAD_VAR_24_MEM_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<?x16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_30_:%.+]] = arith.addf [[LOAD_VAR_23_MEM_]], [[LOAD_VAR_24_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_12_MEM_:%.+]] = krnl.load [[VAR_12_]]#0{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_12_MEM_1_:%.+]] = krnl.load [[VAR_12_]]#4{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_33_:%.+]] = arith.addf [[VAR_30_]], [[LOAD_VAR_12_MEM_]] : f32
// CHECK-DAG:           [[VAR_34_:%.+]] = arith.addf [[VAR_33_]], [[LOAD_VAR_12_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_14_MEM_:%.+]] = krnl.load [[VAR_14_]]#0{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_36_:%.+]] = arith.mulf [[LOAD_VAR_14_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_37_:%.+]] = arith.addf [[VAR_34_]], [[VAR_36_]] : f32
// CHECK:               [[VAR_38_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_37_]] : f32
// CHECK:               [[VAR_39_:%.+]] = math.exp [[VAR_38_]] : f32
// CHECK:               [[VAR_40_:%.+]] = arith.addf [[VAR_39_]], [[VAR_cst_]] : f32
// CHECK-DAG:           [[VAR_41_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_40_]] : f32
// CHECK-DAG:           [[VAR_42_:%.+]] = affine.apply #map0(){{.}}[[VAR_26_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_23_MEM_1_:%.+]] = krnl.load [[VAR_23_]]{{.}}[[VAR_26_1_]]#0, [[VAR_42_]]{{.}} : memref<?x16xf32>
// CHECK-DAG:           [[VAR_44_:%.+]] = affine.apply #map0(){{.}}[[VAR_26_1_]]#1]
// CHECK:               [[LOAD_VAR_24_MEM_1_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_26_1_]]#0, [[VAR_44_]]{{.}} : memref<?x16xf32>
// CHECK-DAG:           [[VAR_46_:%.+]] = arith.addf [[LOAD_VAR_23_MEM_1_]], [[LOAD_VAR_24_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_12_MEM_2_:%.+]] = krnl.load [[VAR_12_]]#2{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_12_MEM_3_:%.+]] = krnl.load [[VAR_12_]]#6{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_49_:%.+]] = arith.addf [[VAR_46_]], [[LOAD_VAR_12_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_50_:%.+]] = arith.addf [[VAR_49_]], [[LOAD_VAR_12_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_14_MEM_1_:%.+]] = krnl.load [[VAR_14_]]#2{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_52_:%.+]] = arith.mulf [[LOAD_VAR_14_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_53_:%.+]] = arith.addf [[VAR_50_]], [[VAR_52_]] : f32
// CHECK:               [[VAR_54_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_53_]] : f32
// CHECK:               [[VAR_55_:%.+]] = math.exp [[VAR_54_]] : f32
// CHECK:               [[VAR_56_:%.+]] = arith.addf [[VAR_55_]], [[VAR_cst_]] : f32
// CHECK-DAG:           [[VAR_57_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_56_]] : f32
// CHECK-DAG:           [[VAR_58_:%.+]] = affine.apply #map1(){{.}}[[VAR_26_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_23_MEM_2_:%.+]] = krnl.load [[VAR_23_]]{{.}}[[VAR_26_1_]]#0, [[VAR_58_]]{{.}} : memref<?x16xf32>
// CHECK-DAG:           [[VAR_60_:%.+]] = affine.apply #map1(){{.}}[[VAR_26_1_]]#1]
// CHECK:               [[LOAD_VAR_24_MEM_2_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_26_1_]]#0, [[VAR_60_]]{{.}} : memref<?x16xf32>
// CHECK-DAG:           [[VAR_62_:%.+]] = arith.addf [[LOAD_VAR_23_MEM_2_]], [[LOAD_VAR_24_MEM_2_]] : f32
// CHECK-DAG:           [[LOAD_VAR_12_MEM_4_:%.+]] = krnl.load [[VAR_12_]]#3{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_12_MEM_5_:%.+]] = krnl.load [[VAR_12_]]#7{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_65_:%.+]] = arith.addf [[VAR_62_]], [[LOAD_VAR_12_MEM_4_]] : f32
// CHECK:               [[VAR_66_:%.+]] = arith.addf [[VAR_65_]], [[LOAD_VAR_12_MEM_5_]] : f32
// CHECK-DAG:           [[VAR_67_:%.+]] = math.tanh [[VAR_66_]] : f32
// CHECK-DAG:           [[VAR_68_:%.+]] = arith.mulf [[VAR_57_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_69_:%.+]] = arith.mulf [[VAR_41_]], [[VAR_67_]] : f32
// CHECK-DAG:           [[VAR_70_:%.+]] = arith.addf [[VAR_68_]], [[VAR_69_]] : f32
// CHECK-DAG:           [[VAR_71_:%.+]] = affine.apply #map2(){{.}}[[VAR_26_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_23_MEM_3_:%.+]] = krnl.load [[VAR_23_]]{{.}}[[VAR_26_1_]]#0, [[VAR_71_]]{{.}} : memref<?x16xf32>
// CHECK-DAG:           [[VAR_73_:%.+]] = affine.apply #map2(){{.}}[[VAR_26_1_]]#1]
// CHECK:               [[LOAD_VAR_24_MEM_3_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_26_1_]]#0, [[VAR_73_]]{{.}} : memref<?x16xf32>
// CHECK-DAG:           [[VAR_75_:%.+]] = arith.addf [[LOAD_VAR_23_MEM_3_]], [[LOAD_VAR_24_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_12_MEM_6_:%.+]] = krnl.load [[VAR_12_]]#1{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_12_MEM_7_:%.+]] = krnl.load [[VAR_12_]]#5{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_78_:%.+]] = arith.addf [[VAR_75_]], [[LOAD_VAR_12_MEM_6_]] : f32
// CHECK-DAG:           [[VAR_79_:%.+]] = arith.addf [[VAR_78_]], [[LOAD_VAR_12_MEM_7_]] : f32
// CHECK-DAG:           [[LOAD_VAR_14_MEM_2_:%.+]] = krnl.load [[VAR_14_]]#1{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_81_:%.+]] = arith.mulf [[LOAD_VAR_14_MEM_2_]], [[VAR_70_]] : f32
// CHECK:               [[VAR_82_:%.+]] = arith.addf [[VAR_79_]], [[VAR_81_]] : f32
// CHECK:               [[VAR_83_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_82_]] : f32
// CHECK:               [[VAR_84_:%.+]] = math.exp [[VAR_83_]] : f32
// CHECK:               [[VAR_85_:%.+]] = arith.addf [[VAR_84_]], [[VAR_cst_]] : f32
// CHECK-DAG:           [[VAR_86_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_85_]] : f32
// CHECK-DAG:           [[VAR_87_:%.+]] = math.tanh [[VAR_70_]] : f32
// CHECK:               [[VAR_88_:%.+]] = arith.mulf [[VAR_86_]], [[VAR_87_]] : f32
// CHECK:               krnl.store [[VAR_70_]], [[RES_2_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<?x4xf32>
// CHECK:               krnl.store [[VAR_88_]], [[RES_1_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<?x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           [[VAR_17_:%.+]] = arith.index_cast [[VAR_2_]] : index to i64
// CHECK:           [[VAR_18_:%.+]] = arith.muli [[VAR_17_]], [[VAR_c16_i64_]] : i64
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_1_]], [[VAR_18_]]) : (memref<1x?x4xf32>, memref<?x4xf32>, i64) -> ()
// CHECK:           return [[RES_]] : memref<1x?x4xf32>
// CHECK:         }
}
