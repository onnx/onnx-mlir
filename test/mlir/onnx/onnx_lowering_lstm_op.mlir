// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl='check-rnn-ops-lowering' --canonicalize %s -split-input-file | FileCheck %s

func private @test_lstm_forward_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x16x3xf32>, %arg2: tensor<1x16x4xf32>, %arg3: tensor<1x32xf32>, %arg4: tensor<1x2x4xf32>, %arg5: tensor<1x2x4xf32>, %arg6: tensor<1x12xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4, %arg5, %arg6) {hidden_size = 4 : si64} : (tensor<7x2x3xf32>, tensor<1x16x3xf32>, tensor<1x16x4xf32>, tensor<1x32xf32>, none, tensor<1x2x4xf32>, tensor<1x2x4xf32>, tensor<1x12xf32>) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

// mlir2FileCheck.py
// CHECK-DAG: #map0 = affine_map<()[s0] -> (s0 + 8)>
// CHECK-DAG: #map1 = affine_map<()[s0] -> (s0 + 12)>
// CHECK-DAG: #map2 = affine_map<()[s0] -> (s0 + 4)>
// CHECK-LABEL:  builtin.func private @test_lstm_forward_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x16x3xf32>, [[PARAM_2_:%.+]]: memref<1x16x4xf32>, [[PARAM_3_:%.+]]: memref<1x32xf32>, [[PARAM_4_:%.+]]: memref<1x2x4xf32>, [[PARAM_5_:%.+]]: memref<1x2x4xf32>, [[PARAM_6_:%.+]]: memref<1x12xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[CST_32_:%.+]] = constant 32 : i64
// CHECK-DAG:       [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:       [[CST_3_:%.+]] = constant 3 : index
// CHECK-DAG:       [[CST_2_:%.+]] = constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1x2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4) {
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:             [[LOAD_PARAM_5_MEM_:%.+]] = krnl.load [[PARAM_5_]]{{.}}[[CST_0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_5_MEM_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.SqueezeV11"([[PARAM_1_]]) {axes = [0]} : (memref<1x16x3xf32>) -> memref<16x3xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.SqueezeV11"([[PARAM_2_]]) {axes = [0]} : (memref<1x16x4xf32>) -> memref<16x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Transpose"([[VAR_4_]]) {perm = [1, 0]} : (memref<16x3xf32>) -> memref<3x16xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Transpose"([[VAR_5_]]) {perm = [1, 0]} : (memref<16x4xf32>) -> memref<4x16xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.SqueezeV11"([[PARAM_3_]]) {axes = [0]} : (memref<1x32xf32>) -> memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_9_:%.+]]:8 = "onnx.SplitV11"([[VAR_8_]]) {axis = 0 : si64} : (memref<32xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.SqueezeV11"([[PARAM_6_]]) {axes = [0]} : (memref<1x12xf32>) -> memref<12xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]]:3 = "onnx.SplitV11"([[VAR_10_]]) {axis = 0 : si64} : (memref<12xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7) {
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_]] to [[CST_3_]]) {
// CHECK:               [[VAR_18_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_18_]]#0, [[VAR_18_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_3_]]{{.}}[[VAR_18_]]#0, [[VAR_18_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_15_:%.+]] = "onnx.MatMul"([[RES_3_]], [[VAR_6_]]) : (memref<2x3xf32>, memref<3x16xf32>) -> memref<2x16xf32>
// CHECK-DAG:         [[VAR_16_:%.+]] = "onnx.MatMul"([[RES_1_]], [[VAR_7_]]) : (memref<2x4xf32>, memref<4x16xf32>) -> memref<2x16xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_]] to [[CST_4_]]) {
// CHECK:               [[VAR_18_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_15_MEM_:%.+]] = krnl.load [[VAR_15_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1] : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_VAR_16_MEM_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1] : memref<2x16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_22_:%.+]] = addf [[LOAD_VAR_15_MEM_]], [[LOAD_VAR_16_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_9_MEM_:%.+]] = krnl.load [[VAR_9_]]#0{{.}}[[VAR_18_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_9_MEM_1_:%.+]] = krnl.load [[VAR_9_]]#4{{.}}[[VAR_18_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_25_:%.+]] = addf [[VAR_22_]], [[LOAD_VAR_9_MEM_]] : f32
// CHECK-DAG:           [[VAR_26_:%.+]] = addf [[VAR_25_]], [[LOAD_VAR_9_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_:%.+]] = krnl.load [[VAR_11_]]#0{{.}}[[VAR_18_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_28_:%.+]] = mulf [[LOAD_VAR_11_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_29_:%.+]] = addf [[VAR_26_]], [[VAR_28_]] : f32
// CHECK-DAG:           [[RES_4_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_29_]], [[RES_4_]][] : memref<f32>
// CHECK:               [[VAR_31_:%.+]] = "onnx.Sigmoid"([[RES_4_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_31_MEM_:%.+]] = krnl.load [[VAR_31_]][] : memref<f32>
// CHECK-DAG:           [[VAR_33_:%.+]] = affine.apply #map0(){{.}}[[VAR_18_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_15_MEM_1_:%.+]] = krnl.load [[VAR_15_]]{{.}}[[VAR_18_1_]]#0, [[VAR_33_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_35_:%.+]] = affine.apply #map0(){{.}}[[VAR_18_1_]]#1]
// CHECK:               [[LOAD_VAR_16_MEM_1_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_18_1_]]#0, [[VAR_35_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_37_:%.+]] = addf [[LOAD_VAR_15_MEM_1_]], [[LOAD_VAR_16_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_9_MEM_2_:%.+]] = krnl.load [[VAR_9_]]#2{{.}}[[VAR_18_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_9_MEM_3_:%.+]] = krnl.load [[VAR_9_]]#6{{.}}[[VAR_18_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_40_:%.+]] = addf [[VAR_37_]], [[LOAD_VAR_9_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_41_:%.+]] = addf [[VAR_40_]], [[LOAD_VAR_9_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_1_:%.+]] = krnl.load [[VAR_11_]]#2{{.}}[[VAR_18_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_43_:%.+]] = mulf [[LOAD_VAR_11_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_44_:%.+]] = addf [[VAR_41_]], [[VAR_43_]] : f32
// CHECK-DAG:           [[RES_5_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_44_]], [[RES_5_]][] : memref<f32>
// CHECK:               [[VAR_46_:%.+]] = "onnx.Sigmoid"([[RES_5_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_46_MEM_:%.+]] = krnl.load [[VAR_46_]][] : memref<f32>
// CHECK-DAG:           [[VAR_48_:%.+]] = affine.apply #map1(){{.}}[[VAR_18_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_15_MEM_2_:%.+]] = krnl.load [[VAR_15_]]{{.}}[[VAR_18_1_]]#0, [[VAR_48_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_50_:%.+]] = affine.apply #map1(){{.}}[[VAR_18_1_]]#1]
// CHECK:               [[LOAD_VAR_16_MEM_2_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_18_1_]]#0, [[VAR_50_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_52_:%.+]] = addf [[LOAD_VAR_15_MEM_2_]], [[LOAD_VAR_16_MEM_2_]] : f32
// CHECK-DAG:           [[LOAD_VAR_9_MEM_4_:%.+]] = krnl.load [[VAR_9_]]#3{{.}}[[VAR_18_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_9_MEM_5_:%.+]] = krnl.load [[VAR_9_]]#7{{.}}[[VAR_18_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_55_:%.+]] = addf [[VAR_52_]], [[LOAD_VAR_9_MEM_4_]] : f32
// CHECK-DAG:           [[VAR_56_:%.+]] = addf [[VAR_55_]], [[LOAD_VAR_9_MEM_5_]] : f32
// CHECK-DAG:           [[RES_6_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_56_]], [[RES_6_]][] : memref<f32>
// CHECK:               [[VAR_58_:%.+]] = "onnx.Tanh"([[RES_6_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_58_MEM_:%.+]] = krnl.load [[VAR_58_]][] : memref<f32>
// CHECK-DAG:           [[VAR_60_:%.+]] = mulf [[LOAD_VAR_46_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_61_:%.+]] = mulf [[LOAD_VAR_31_MEM_]], [[LOAD_VAR_58_MEM_]] : f32
// CHECK-DAG:           [[VAR_62_:%.+]] = addf [[VAR_60_]], [[VAR_61_]] : f32
// CHECK-DAG:           [[VAR_63_:%.+]] = affine.apply #map2(){{.}}[[VAR_18_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_15_MEM_3_:%.+]] = krnl.load [[VAR_15_]]{{.}}[[VAR_18_1_]]#0, [[VAR_63_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_65_:%.+]] = affine.apply #map2(){{.}}[[VAR_18_1_]]#1]
// CHECK:               [[LOAD_VAR_16_MEM_3_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_18_1_]]#0, [[VAR_65_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_67_:%.+]] = addf [[LOAD_VAR_15_MEM_3_]], [[LOAD_VAR_16_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_9_MEM_6_:%.+]] = krnl.load [[VAR_9_]]#1{{.}}[[VAR_18_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_9_MEM_7_:%.+]] = krnl.load [[VAR_9_]]#5{{.}}[[VAR_18_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_70_:%.+]] = addf [[VAR_67_]], [[LOAD_VAR_9_MEM_6_]] : f32
// CHECK-DAG:           [[VAR_71_:%.+]] = addf [[VAR_70_]], [[LOAD_VAR_9_MEM_7_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_2_:%.+]] = krnl.load [[VAR_11_]]#1{{.}}[[VAR_18_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_73_:%.+]] = mulf [[LOAD_VAR_11_MEM_2_]], [[VAR_62_]] : f32
// CHECK-DAG:           [[VAR_74_:%.+]] = addf [[VAR_71_]], [[VAR_73_]] : f32
// CHECK-DAG:           [[RES_7_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_74_]], [[RES_7_]][] : memref<f32>
// CHECK:               [[VAR_76_:%.+]] = "onnx.Sigmoid"([[RES_7_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_76_MEM_:%.+]] = krnl.load [[VAR_76_]][] : memref<f32>
// CHECK-DAG:           [[RES_8_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_62_]], [[RES_8_]][] : memref<f32>
// CHECK:               [[VAR_79_:%.+]] = "onnx.Tanh"([[RES_8_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_VAR_79_MEM_:%.+]] = krnl.load [[VAR_79_]][] : memref<f32>
// CHECK:               [[VAR_81_:%.+]] = mulf [[LOAD_VAR_76_MEM_]], [[LOAD_VAR_79_MEM_]] : f32
// CHECK:               krnl.store [[VAR_62_]], [[RES_]]{{.}}[[VAR_18_1_]]#0, [[VAR_18_1_]]#1] : memref<2x4xf32>
// CHECK:               krnl.store [[VAR_81_]], [[RES_1_]]{{.}}[[RES_1_]]8#0, [[RES_1_]]8#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           "krnl.memcpy"([[RES_2_]], [[RES_1_]], [[CST_32_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
// CHECK:           return [[RES_2_]] : memref<1x2x4xf32>
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

// CHECK-DAG: #map0 = affine_map<()[s0] -> (s0 + 8)>
// CHECK-DAG: #map1 = affine_map<()[s0] -> (s0 + 12)>
// CHECK-DAG: #map2 = affine_map<()[s0] -> (s0 + 4)>
// CHECK-LABEL:  builtin.func private @test_lstm_forward_mode_constant_weight_and_bias
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x2x4xf32>, [[PARAM_2_:%.+]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[CST_32_:%.+]] = constant 32 : i64
// CHECK-DAG:       [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:       [[CST_3_:%.+]] = constant 3 : index
// CHECK-DAG:       [[CST_2_:%.+]] = constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1x2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4) {
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_1_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:             [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_2_MEM_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
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
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7) {
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_]] to [[CST_3_]]) {
// CHECK:               [[VAR_23_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_23_]]#0, [[VAR_23_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_3_]]{{.}}[[VAR_23_]]#0, [[VAR_23_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_20_:%.+]] = "onnx.MatMul"([[RES_3_]], [[VAR_4_]]) : (memref<2x3xf32>, memref<3x16xf32>) -> memref<2x16xf32>
// CHECK-DAG:         [[VAR_21_:%.+]] = "onnx.MatMul"([[RES_1_]], [[VAR_5_]]) : (memref<2x4xf32>, memref<4x16xf32>) -> memref<2x16xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_]] to [[CST_4_]]) {
// CHECK:               [[VAR_23_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_23_1_]]#0, [[VAR_23_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_20_MEM_:%.+]] = krnl.load [[VAR_20_]]{{.}}[[VAR_23_1_]]#0, [[VAR_23_1_]]#1] : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_VAR_21_MEM_:%.+]] = krnl.load [[VAR_21_]]{{.}}[[VAR_23_1_]]#0, [[VAR_23_1_]]#1] : memref<2x16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_27_:%.+]] = addf [[LOAD_VAR_20_MEM_]], [[LOAD_VAR_21_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_6_MEM_:%.+]] = krnl.load [[VAR_6_]]{{.}}[[VAR_23_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_10_MEM_:%.+]] = krnl.load [[VAR_10_]]{{.}}[[VAR_23_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_30_:%.+]] = addf [[VAR_27_]], [[LOAD_VAR_6_MEM_]] : f32
// CHECK-DAG:           [[VAR_31_:%.+]] = addf [[VAR_30_]], [[LOAD_VAR_10_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_14_MEM_:%.+]] = krnl.load [[VAR_14_]]{{.}}[[VAR_23_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_33_:%.+]] = mulf [[LOAD_VAR_14_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_34_:%.+]] = addf [[VAR_31_]], [[VAR_33_]] : f32
// CHECK-DAG:           [[RES_4_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_34_]], [[RES_4_]][] : memref<f32>
// CHECK:               [[VAR_36_:%.+]] = "onnx.Sigmoid"([[RES_4_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_36_MEM_:%.+]] = krnl.load [[VAR_36_]][] : memref<f32>
// CHECK-DAG:           [[VAR_38_:%.+]] = affine.apply #map0(){{.}}[[VAR_23_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_20_MEM_1_:%.+]] = krnl.load [[VAR_20_]]{{.}}[[VAR_23_1_]]#0, [[VAR_38_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_40_:%.+]] = affine.apply #map0(){{.}}[[VAR_23_1_]]#1]
// CHECK:               [[LOAD_VAR_21_MEM_1_:%.+]] = krnl.load [[VAR_21_]]{{.}}[[VAR_23_1_]]#0, [[VAR_40_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_42_:%.+]] = addf [[LOAD_VAR_20_MEM_1_]], [[LOAD_VAR_21_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_8_MEM_:%.+]] = krnl.load [[VAR_8_]]{{.}}[[VAR_23_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_12_MEM_:%.+]] = krnl.load [[VAR_12_]]{{.}}[[VAR_23_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_45_:%.+]] = addf [[VAR_42_]], [[LOAD_VAR_8_MEM_]] : f32
// CHECK-DAG:           [[VAR_46_:%.+]] = addf [[VAR_45_]], [[LOAD_VAR_12_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_16_MEM_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_23_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_48_:%.+]] = mulf [[LOAD_VAR_16_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_49_:%.+]] = addf [[VAR_46_]], [[VAR_48_]] : f32
// CHECK-DAG:           [[RES_5_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_49_]], [[RES_5_]][] : memref<f32>
// CHECK:               [[VAR_51_:%.+]] = "onnx.Sigmoid"([[RES_5_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_51_MEM_:%.+]] = krnl.load [[VAR_51_]][] : memref<f32>
// CHECK-DAG:           [[VAR_53_:%.+]] = affine.apply #map1(){{.}}[[VAR_23_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_20_MEM_2_:%.+]] = krnl.load [[VAR_20_]]{{.}}[[VAR_23_1_]]#0, [[VAR_53_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_55_:%.+]] = affine.apply #map1(){{.}}[[VAR_23_1_]]#1]
// CHECK:               [[LOAD_VAR_21_MEM_2_:%.+]] = krnl.load [[VAR_21_]]{{.}}[[VAR_23_1_]]#0, [[VAR_55_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_57_:%.+]] = addf [[LOAD_VAR_20_MEM_2_]], [[LOAD_VAR_21_MEM_2_]] : f32
// CHECK-DAG:           [[LOAD_VAR_9_MEM_:%.+]] = krnl.load [[VAR_9_]]{{.}}[[VAR_23_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_13_MEM_:%.+]] = krnl.load [[VAR_13_]]{{.}}[[VAR_23_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_60_:%.+]] = addf [[VAR_57_]], [[LOAD_VAR_9_MEM_]] : f32
// CHECK-DAG:           [[VAR_61_:%.+]] = addf [[VAR_60_]], [[LOAD_VAR_13_MEM_]] : f32
// CHECK-DAG:           [[RES_6_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_61_]], [[RES_6_]][] : memref<f32>
// CHECK:               [[VAR_63_:%.+]] = "onnx.Tanh"([[RES_6_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_63_MEM_:%.+]] = krnl.load [[VAR_63_]][] : memref<f32>
// CHECK-DAG:           [[VAR_65_:%.+]] = mulf [[LOAD_VAR_51_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_66_:%.+]] = mulf [[LOAD_VAR_36_MEM_]], [[LOAD_VAR_63_MEM_]] : f32
// CHECK-DAG:           [[VAR_67_:%.+]] = addf [[VAR_65_]], [[VAR_66_]] : f32
// CHECK-DAG:           [[VAR_68_:%.+]] = affine.apply #map2(){{.}}[[VAR_23_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_20_MEM_3_:%.+]] = krnl.load [[VAR_20_]]{{.}}[[VAR_23_1_]]#0, [[VAR_68_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_70_:%.+]] = affine.apply #map2(){{.}}[[VAR_23_1_]]#1]
// CHECK:               [[LOAD_VAR_21_MEM_3_:%.+]] = krnl.load [[VAR_21_]]{{.}}[[VAR_23_1_]]#0, [[VAR_70_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_72_:%.+]] = addf [[LOAD_VAR_20_MEM_3_]], [[LOAD_VAR_21_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_7_MEM_:%.+]] = krnl.load [[VAR_7_]]{{.}}[[VAR_23_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_11_MEM_:%.+]] = krnl.load [[VAR_11_]]{{.}}[[VAR_23_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_75_:%.+]] = addf [[VAR_72_]], [[LOAD_VAR_7_MEM_]] : f32
// CHECK-DAG:           [[VAR_76_:%.+]] = addf [[VAR_75_]], [[LOAD_VAR_11_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_15_MEM_:%.+]] = krnl.load [[VAR_15_]]{{.}}[[VAR_23_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_78_:%.+]] = mulf [[LOAD_VAR_15_MEM_]], [[VAR_67_]] : f32
// CHECK-DAG:           [[VAR_79_:%.+]] = addf [[VAR_76_]], [[VAR_78_]] : f32
// CHECK-DAG:           [[RES_7_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_79_]], [[RES_7_]][] : memref<f32>
// CHECK:               [[VAR_81_:%.+]] = "onnx.Sigmoid"([[RES_7_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_81_MEM_:%.+]] = krnl.load [[VAR_81_]][] : memref<f32>
// CHECK-DAG:           [[RES_8_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_67_]], [[RES_8_]][] : memref<f32>
// CHECK:               [[VAR_84_:%.+]] = "onnx.Tanh"([[RES_8_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_VAR_84_MEM_:%.+]] = krnl.load [[VAR_84_]][] : memref<f32>
// CHECK:               [[VAR_86_:%.+]] = mulf [[LOAD_VAR_81_MEM_]], [[LOAD_VAR_84_MEM_]] : f32
// CHECK:               krnl.store [[VAR_67_]], [[RES_]]{{.}}[[VAR_23_1_]]#0, [[VAR_23_1_]]#1] : memref<2x4xf32>
// CHECK:               krnl.store [[VAR_86_]], [[RES_1_]]{{.}}[[VAR_23_1_]]#0, [[VAR_23_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           "krnl.memcpy"([[RES_2_]], [[RES_1_]], [[CST_32_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
// CHECK:           return [[RES_2_]] : memref<1x2x4xf32>
// CHECK:         }

}

// -----

func private @test_lstm_reverse_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x16x3xf32>, %arg2: tensor<1x16x4xf32>, %arg3: tensor<1x32xf32>, %arg4: tensor<1x2x4xf32>, %arg5: tensor<1x2x4xf32>, %arg6: tensor<1x12xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4, %arg5, %arg6) {hidden_size = 4 : si64, direction = "reverse"} : (tensor<7x2x3xf32>, tensor<1x16x3xf32>, tensor<1x16x4xf32>, tensor<1x32xf32>, none, tensor<1x2x4xf32>, tensor<1x2x4xf32>, tensor<1x12xf32>) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

// CHECK-DAG: #map0 = affine_map<(d0) -> (-d0 + 6)>
// CHECK-DAG: #map1 = affine_map<()[s0] -> (s0 + 8)>
// CHECK-DAG: #map2 = affine_map<()[s0] -> (s0 + 12)>
// CHECK-DAG: #map3 = affine_map<()[s0] -> (s0 + 4)>
// CHECK-LABEL:  builtin.func private @test_lstm_reverse_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x16x3xf32>, [[PARAM_2_:%.+]]: memref<1x16x4xf32>, [[PARAM_3_:%.+]]: memref<1x32xf32>, [[PARAM_4_:%.+]]: memref<1x2x4xf32>, [[PARAM_5_:%.+]]: memref<1x2x4xf32>, [[PARAM_6_:%.+]]: memref<1x12xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[CST_32_:%.+]] = constant 32 : i64
// CHECK-DAG:       [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:       [[CST_3_:%.+]] = constant 3 : index
// CHECK-DAG:       [[CST_2_:%.+]] = constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1x2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4) {
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:             [[LOAD_PARAM_5_MEM_:%.+]] = krnl.load [[PARAM_5_]]{{.}}[[CST_0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_5_MEM_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.SqueezeV11"([[PARAM_1_]]) {axes = [0]} : (memref<1x16x3xf32>) -> memref<16x3xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.SqueezeV11"([[PARAM_2_]]) {axes = [0]} : (memref<1x16x4xf32>) -> memref<16x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Transpose"([[VAR_4_]]) {perm = [1, 0]} : (memref<16x3xf32>) -> memref<3x16xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Transpose"([[VAR_5_]]) {perm = [1, 0]} : (memref<16x4xf32>) -> memref<4x16xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.SqueezeV11"([[PARAM_3_]]) {axes = [0]} : (memref<1x32xf32>) -> memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_9_:%.+]]:8 = "onnx.SplitV11"([[VAR_8_]]) {axis = 0 : si64} : (memref<32xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.SqueezeV11"([[PARAM_6_]]) {axes = [0]} : (memref<1x12xf32>) -> memref<12xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]]:3 = "onnx.SplitV11"([[VAR_10_]]) {axis = 0 : si64} : (memref<12xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7) {
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOAD_PARAM_5_MEM_1_:%.+]] = affine.apply #map0([[I_2_]])
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_]] to [[CST_3_]]) {
// CHECK:               [[VAR_19_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[LOAD_PARAM_5_MEM_1_]], [[VAR_19_]]#0, [[VAR_19_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_3_]]{{.}}[[VAR_19_]]#0, [[VAR_19_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_16_:%.+]] = "onnx.MatMul"([[RES_3_]], [[VAR_6_]]) : (memref<2x3xf32>, memref<3x16xf32>) -> memref<2x16xf32>
// CHECK-DAG:         [[VAR_17_:%.+]] = "onnx.MatMul"([[RES_1_]], [[VAR_7_]]) : (memref<2x4xf32>, memref<4x16xf32>) -> memref<2x16xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_]] to [[CST_4_]]) {
// CHECK:               [[VAR_19_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_19_1_]]#0, [[VAR_19_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_16_MEM_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_19_1_]]#0, [[VAR_19_1_]]#1] : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_VAR_17_MEM_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_19_1_]]#0, [[VAR_19_1_]]#1] : memref<2x16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_23_:%.+]] = addf [[LOAD_VAR_16_MEM_]], [[LOAD_VAR_17_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_9_MEM_:%.+]] = krnl.load [[VAR_9_]]#0{{.}}[[VAR_19_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_9_MEM_1_:%.+]] = krnl.load [[VAR_9_]]#4{{.}}[[VAR_19_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_26_:%.+]] = addf [[VAR_23_]], [[LOAD_VAR_9_MEM_]] : f32
// CHECK-DAG:           [[VAR_27_:%.+]] = addf [[VAR_26_]], [[LOAD_VAR_9_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_:%.+]] = krnl.load [[VAR_11_]]#0{{.}}[[VAR_19_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_29_:%.+]] = mulf [[LOAD_VAR_11_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_30_:%.+]] = addf [[VAR_27_]], [[VAR_29_]] : f32
// CHECK-DAG:           [[RES_4_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_30_]], [[RES_4_]][] : memref<f32>
// CHECK:               [[VAR_32_:%.+]] = "onnx.Sigmoid"([[RES_4_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_32_MEM_:%.+]] = krnl.load [[VAR_32_]][] : memref<f32>
// CHECK-DAG:           [[VAR_34_:%.+]] = affine.apply #map1(){{.}}[[VAR_19_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_16_MEM_1_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_19_1_]]#0, [[VAR_34_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_36_:%.+]] = affine.apply #map1(){{.}}[[VAR_19_1_]]#1]
// CHECK:               [[LOAD_VAR_17_MEM_1_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_19_1_]]#0, [[VAR_36_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_38_:%.+]] = addf [[LOAD_VAR_16_MEM_1_]], [[LOAD_VAR_17_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_9_MEM_2_:%.+]] = krnl.load [[VAR_9_]]#2{{.}}[[VAR_19_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_9_MEM_3_:%.+]] = krnl.load [[VAR_9_]]#6{{.}}[[VAR_19_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_41_:%.+]] = addf [[VAR_38_]], [[LOAD_VAR_9_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_42_:%.+]] = addf [[VAR_41_]], [[LOAD_VAR_9_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_1_:%.+]] = krnl.load [[VAR_11_]]#2{{.}}[[VAR_19_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_44_:%.+]] = mulf [[LOAD_VAR_11_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_45_:%.+]] = addf [[VAR_42_]], [[VAR_44_]] : f32
// CHECK-DAG:           [[RES_5_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_45_]], [[RES_5_]][] : memref<f32>
// CHECK:               [[VAR_47_:%.+]] = "onnx.Sigmoid"([[RES_5_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_47_MEM_:%.+]] = krnl.load [[VAR_47_]][] : memref<f32>
// CHECK-DAG:           [[VAR_49_:%.+]] = affine.apply #map2(){{.}}[[VAR_19_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_16_MEM_2_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_19_1_]]#0, [[VAR_49_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_51_:%.+]] = affine.apply #map2(){{.}}[[VAR_19_1_]]#1]
// CHECK:               [[LOAD_VAR_17_MEM_2_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_19_1_]]#0, [[VAR_51_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_53_:%.+]] = addf [[LOAD_VAR_16_MEM_2_]], [[LOAD_VAR_17_MEM_2_]] : f32
// CHECK-DAG:           [[LOAD_VAR_9_MEM_4_:%.+]] = krnl.load [[VAR_9_]]#3{{.}}[[VAR_19_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_9_MEM_5_:%.+]] = krnl.load [[VAR_9_]]#7{{.}}[[VAR_19_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_56_:%.+]] = addf [[VAR_53_]], [[LOAD_VAR_9_MEM_4_]] : f32
// CHECK-DAG:           [[VAR_57_:%.+]] = addf [[VAR_56_]], [[LOAD_VAR_9_MEM_5_]] : f32
// CHECK-DAG:           [[RES_6_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_57_]], [[RES_6_]][] : memref<f32>
// CHECK:               [[VAR_59_:%.+]] = "onnx.Tanh"([[RES_6_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_59_MEM_:%.+]] = krnl.load [[VAR_59_]][] : memref<f32>
// CHECK-DAG:           [[VAR_61_:%.+]] = mulf [[LOAD_VAR_47_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_62_:%.+]] = mulf [[LOAD_VAR_32_MEM_]], [[LOAD_VAR_59_MEM_]] : f32
// CHECK-DAG:           [[VAR_63_:%.+]] = addf [[VAR_61_]], [[VAR_62_]] : f32
// CHECK-DAG:           [[VAR_64_:%.+]] = affine.apply #map3(){{.}}[[VAR_19_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_16_MEM_3_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_19_1_]]#0, [[VAR_64_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_66_:%.+]] = affine.apply #map3(){{.}}[[VAR_19_1_]]#1]
// CHECK:               [[LOAD_VAR_17_MEM_3_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_19_1_]]#0, [[VAR_66_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_68_:%.+]] = addf [[LOAD_VAR_16_MEM_3_]], [[LOAD_VAR_17_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_9_MEM_6_:%.+]] = krnl.load [[VAR_9_]]#1{{.}}[[VAR_19_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_9_MEM_7_:%.+]] = krnl.load [[VAR_9_]]#5{{.}}[[VAR_19_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_71_:%.+]] = addf [[VAR_68_]], [[LOAD_VAR_9_MEM_6_]] : f32
// CHECK-DAG:           [[VAR_72_:%.+]] = addf [[VAR_71_]], [[LOAD_VAR_9_MEM_7_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_2_:%.+]] = krnl.load [[VAR_11_]]#1{{.}}[[VAR_19_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_74_:%.+]] = mulf [[LOAD_VAR_11_MEM_2_]], [[VAR_63_]] : f32
// CHECK-DAG:           [[VAR_75_:%.+]] = addf [[VAR_72_]], [[VAR_74_]] : f32
// CHECK-DAG:           [[RES_7_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_75_]], [[RES_7_]][] : memref<f32>
// CHECK:               [[VAR_77_:%.+]] = "onnx.Sigmoid"([[RES_7_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_77_MEM_:%.+]] = krnl.load [[VAR_77_]][] : memref<f32>
// CHECK-DAG:           [[RES_8_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_63_]], [[RES_8_]][] : memref<f32>
// CHECK:               [[VAR_80_:%.+]] = "onnx.Tanh"([[RES_8_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_VAR_80_MEM_:%.+]] = krnl.load [[VAR_80_]][] : memref<f32>
// CHECK:               [[VAR_82_:%.+]] = mulf [[LOAD_VAR_77_MEM_]], [[LOAD_VAR_80_MEM_]] : f32
// CHECK:               krnl.store [[VAR_63_]], [[RES_]]{{.}}[[VAR_19_1_]]#0, [[VAR_19_1_]]#1] : memref<2x4xf32>
// CHECK:               krnl.store [[VAR_82_]], [[RES_1_]]{{.}}[[RES_1_]]9#0, [[RES_1_]]9#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           "krnl.memcpy"([[RES_2_]], [[RES_1_]], [[CST_32_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
// CHECK:           return [[RES_2_]] : memref<1x2x4xf32>
// CHECK:         }

}


// -----

func private @test_lstm_bidirectional_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<2x16x3xf32>, %arg2: tensor<2x16x4xf32>, %arg3: tensor<2x32xf32>, %arg4: tensor<2x2x4xf32>, %arg5: tensor<2x2x4xf32>, %arg6: tensor<2x12xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4, %arg5, %arg6) {hidden_size = 4 : si64, direction = "bidirectional"} : (tensor<7x2x3xf32>, tensor<2x16x3xf32>, tensor<2x16x4xf32>, tensor<2x32xf32>, none, tensor<2x2x4xf32>, tensor<2x2x4xf32>, tensor<2x12xf32>) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

// CHECK-DAG: #map0 = affine_map<()[s0] -> (s0 + 8)>
// CHECK-DAG: #map1 = affine_map<()[s0] -> (s0 + 12)>
// CHECK-DAG: #map2 = affine_map<()[s0] -> (s0 + 4)>
// CHECK-DAG: #map3 = affine_map<(d0) -> (-d0 + 6)>
// CHECK-LABEL:  builtin.func private @test_lstm_bidirectional_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<2x16x3xf32>, [[PARAM_2_:%.+]]: memref<2x16x4xf32>, [[PARAM_3_:%.+]]: memref<2x32xf32>, [[PARAM_4_:%.+]]: memref<2x2x4xf32>, [[PARAM_5_:%.+]]: memref<2x2x4xf32>, [[PARAM_6_:%.+]]: memref<2x12xf32>) -> memref<2x2x4xf32> {
// CHECK-DAG:       [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:       [[CST_3_:%.+]] = constant 3 : index
// CHECK-DAG:       [[CST_2_:%.+]] = constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<2x2x4xf32>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4) {
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_]], [[I_0_]], [[I_1_]]{{.}} : memref<2x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_3_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:             [[LOAD_PARAM_5_MEM_:%.+]] = krnl.load [[PARAM_5_]]{{.}}[[CST_0_]], [[I_0_]], [[I_1_]]{{.}} : memref<2x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_5_MEM_]], [[RES_2_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:             [[LOAD_PARAM_4_MEM_1_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_1_]], [[I_0_]], [[I_1_]]{{.}} : memref<2x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_1_]], [[RES_1_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:             [[LOAD_PARAM_5_MEM_1_:%.+]] = krnl.load [[PARAM_5_]]{{.}}[[CST_1_]], [[I_0_]], [[I_1_]]{{.}} : memref<2x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_5_MEM_1_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:           }
// CHECK:           [[VAR_6_:%.+]]:2 = "onnx.SplitV11"([[PARAM_1_]]) {axis = 0 : si64} : (memref<2x16x3xf32>) -> (memref<1x16x3xf32>, memref<1x16x3xf32>)
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.SqueezeV11"([[VAR_6_]]#0) {axes = [0]} : (memref<1x16x3xf32>) -> memref<16x3xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.SqueezeV11"([[VAR_6_]]#1) {axes = [0]} : (memref<1x16x3xf32>) -> memref<16x3xf32>
// CHECK-DAG:       [[VAR_9_:%.+]]:2 = "onnx.SplitV11"([[PARAM_2_]]) {axis = 0 : si64} : (memref<2x16x4xf32>) -> (memref<1x16x4xf32>, memref<1x16x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.SqueezeV11"([[VAR_9_]]#0) {axes = [0]} : (memref<1x16x4xf32>) -> memref<16x4xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.SqueezeV11"([[VAR_9_]]#1) {axes = [0]} : (memref<1x16x4xf32>) -> memref<16x4xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Transpose"([[VAR_7_]]) {perm = [1, 0]} : (memref<16x3xf32>) -> memref<3x16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Transpose"([[VAR_10_]]) {perm = [1, 0]} : (memref<16x4xf32>) -> memref<4x16xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.Transpose"([[VAR_8_]]) {perm = [1, 0]} : (memref<16x3xf32>) -> memref<3x16xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.Transpose"([[VAR_11_]]) {perm = [1, 0]} : (memref<16x4xf32>) -> memref<4x16xf32>
// CHECK-DAG:       [[VAR_16_:%.+]]:2 = "onnx.SplitV11"([[PARAM_3_]]) {axis = 0 : si64} : (memref<2x32xf32>) -> (memref<1x32xf32>, memref<1x32xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.SqueezeV11"([[VAR_16_]]#0) {axes = [0]} : (memref<1x32xf32>) -> memref<32xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.SqueezeV11"([[VAR_16_]]#1) {axes = [0]} : (memref<1x32xf32>) -> memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_19_:%.+]]:8 = "onnx.SplitV11"([[VAR_17_]]) {axis = 0 : si64} : (memref<32xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[VAR_20_:%.+]]:8 = "onnx.SplitV11"([[VAR_18_]]) {axis = 0 : si64} : (memref<32xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[VAR_21_:%.+]]:2 = "onnx.SplitV11"([[PARAM_6_]]) {axis = 0 : si64} : (memref<2x12xf32>) -> (memref<1x12xf32>, memref<1x12xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_22_:%.+]] = "onnx.SqueezeV11"([[VAR_21_]]#0) {axes = [0]} : (memref<1x12xf32>) -> memref<12xf32>
// CHECK-DAG:       [[VAR_23_:%.+]] = "onnx.SqueezeV11"([[VAR_21_]]#1) {axes = [0]} : (memref<1x12xf32>) -> memref<12xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_24_:%.+]]:3 = "onnx.SplitV11"([[VAR_22_]]) {axis = 0 : si64} : (memref<12xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[VAR_25_:%.+]]:3 = "onnx.SplitV11"([[VAR_23_]]) {axis = 0 : si64} : (memref<12xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7) {
// CHECK-DAG:         [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_]] to [[CST_3_]]) {
// CHECK:               [[VAR_34_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_34_]]#0, [[VAR_34_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_5_]]{{.}}[[VAR_34_]]#0, [[VAR_34_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_1_:%.+]] = "onnx.MatMul"([[RES_5_]], [[VAR_12_]]) : (memref<2x3xf32>, memref<3x16xf32>) -> memref<2x16xf32>
// CHECK-DAG:         [[LOAD_PARAM_5_MEM_1_:%.+]] = "onnx.MatMul"([[RES_3_]], [[VAR_13_]]) : (memref<2x4xf32>, memref<4x16xf32>) -> memref<2x16xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_]] to [[CST_4_]]) {
// CHECK:               [[VAR_34_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_34_1_]]#0, [[VAR_34_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_LOAD_PARAM_4_MEM_1_MEM_:%.+]] = krnl.load [[LOAD_PARAM_4_MEM_1_]]{{.}}[[VAR_34_1_]]#0, [[VAR_34_1_]]#1] : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_LOAD_PARAM_5_MEM_1_MEM_:%.+]] = krnl.load [[LOAD_PARAM_5_MEM_1_]]{{.}}[[VAR_34_1_]]#0, [[VAR_34_1_]]#1] : memref<2x16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_38_:%.+]] = addf [[LOAD_LOAD_PARAM_4_MEM_1_MEM_]], [[LOAD_LOAD_PARAM_5_MEM_1_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_19_MEM_:%.+]] = krnl.load [[VAR_19_]]#0{{.}}[[VAR_34_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_19_MEM_1_:%.+]] = krnl.load [[VAR_19_]]#4{{.}}[[VAR_34_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_41_:%.+]] = addf [[VAR_38_]], [[LOAD_VAR_19_MEM_]] : f32
// CHECK-DAG:           [[VAR_42_:%.+]] = addf [[VAR_41_]], [[LOAD_VAR_19_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_24_MEM_:%.+]] = krnl.load [[VAR_24_]]#0{{.}}[[VAR_34_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_44_:%.+]] = mulf [[LOAD_VAR_24_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_45_:%.+]] = addf [[VAR_42_]], [[VAR_44_]] : f32
// CHECK-DAG:           [[RES_6_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_45_]], [[RES_6_]][] : memref<f32>
// CHECK:               [[VAR_47_:%.+]] = "onnx.Sigmoid"([[RES_6_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_47_MEM_:%.+]] = krnl.load [[VAR_47_]][] : memref<f32>
// CHECK-DAG:           [[VAR_49_:%.+]] = affine.apply #map0(){{.}}[[VAR_34_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_LOAD_PARAM_4_MEM_1_MEM_1_:%.+]] = krnl.load [[LOAD_PARAM_4_MEM_1_]]{{.}}[[VAR_34_1_]]#0, [[VAR_49_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_51_:%.+]] = affine.apply #map0(){{.}}[[VAR_34_1_]]#1]
// CHECK:               [[LOAD_LOAD_PARAM_5_MEM_1_MEM_1_:%.+]] = krnl.load [[LOAD_PARAM_5_MEM_1_]]{{.}}[[VAR_34_1_]]#0, [[VAR_51_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_53_:%.+]] = addf [[LOAD_LOAD_PARAM_4_MEM_1_MEM_1_]], [[LOAD_LOAD_PARAM_5_MEM_1_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_19_MEM_2_:%.+]] = krnl.load [[VAR_19_]]#2{{.}}[[VAR_34_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_19_MEM_3_:%.+]] = krnl.load [[VAR_19_]]#6{{.}}[[VAR_34_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_56_:%.+]] = addf [[VAR_53_]], [[LOAD_VAR_19_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_57_:%.+]] = addf [[VAR_56_]], [[LOAD_VAR_19_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_24_MEM_1_:%.+]] = krnl.load [[VAR_24_]]#2{{.}}[[VAR_34_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_59_:%.+]] = mulf [[LOAD_VAR_24_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_60_:%.+]] = addf [[VAR_57_]], [[VAR_59_]] : f32
// CHECK-DAG:           [[RES_7_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_60_]], [[RES_7_]][] : memref<f32>
// CHECK:               [[VAR_62_:%.+]] = "onnx.Sigmoid"([[RES_7_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_62_MEM_:%.+]] = krnl.load [[VAR_62_]][] : memref<f32>
// CHECK-DAG:           [[VAR_64_:%.+]] = affine.apply #map1(){{.}}[[VAR_34_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_LOAD_PARAM_4_MEM_1_MEM_2_:%.+]] = krnl.load [[LOAD_PARAM_4_MEM_1_]]{{.}}[[VAR_34_1_]]#0, [[VAR_64_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_66_:%.+]] = affine.apply #map1(){{.}}[[VAR_34_1_]]#1]
// CHECK:               [[LOAD_LOAD_PARAM_5_MEM_1_MEM_2_:%.+]] = krnl.load [[LOAD_PARAM_5_MEM_1_]]{{.}}[[VAR_34_1_]]#0, [[VAR_66_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_68_:%.+]] = addf [[LOAD_LOAD_PARAM_4_MEM_1_MEM_2_]], [[LOAD_LOAD_PARAM_5_MEM_1_MEM_2_]] : f32
// CHECK-DAG:           [[LOAD_VAR_19_MEM_4_:%.+]] = krnl.load [[VAR_19_]]#3{{.}}[[VAR_34_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_19_MEM_5_:%.+]] = krnl.load [[VAR_19_]]#7{{.}}[[VAR_34_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_71_:%.+]] = addf [[VAR_68_]], [[LOAD_VAR_19_MEM_4_]] : f32
// CHECK-DAG:           [[VAR_72_:%.+]] = addf [[VAR_71_]], [[LOAD_VAR_19_MEM_5_]] : f32
// CHECK-DAG:           [[RES_8_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_72_]], [[RES_8_]][] : memref<f32>
// CHECK:               [[VAR_74_:%.+]] = "onnx.Tanh"([[RES_8_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_74_MEM_:%.+]] = krnl.load [[VAR_74_]][] : memref<f32>
// CHECK-DAG:           [[VAR_76_:%.+]] = mulf [[LOAD_VAR_62_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_77_:%.+]] = mulf [[LOAD_VAR_47_MEM_]], [[LOAD_VAR_74_MEM_]] : f32
// CHECK-DAG:           [[VAR_78_:%.+]] = addf [[VAR_76_]], [[VAR_77_]] : f32
// CHECK-DAG:           [[VAR_79_:%.+]] = affine.apply #map2(){{.}}[[VAR_34_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_LOAD_PARAM_4_MEM_1_MEM_3_:%.+]] = krnl.load [[LOAD_PARAM_4_MEM_1_]]{{.}}[[VAR_34_1_]]#0, [[VAR_79_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_81_:%.+]] = affine.apply #map2(){{.}}[[VAR_34_1_]]#1]
// CHECK:               [[LOAD_LOAD_PARAM_5_MEM_1_MEM_3_:%.+]] = krnl.load [[LOAD_PARAM_5_MEM_1_]]{{.}}[[VAR_34_1_]]#0, [[VAR_81_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[VAR_83_:%.+]] = addf [[LOAD_LOAD_PARAM_4_MEM_1_MEM_3_]], [[LOAD_LOAD_PARAM_5_MEM_1_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_19_MEM_6_:%.+]] = krnl.load [[VAR_19_]]#1{{.}}[[VAR_34_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_19_MEM_7_:%.+]] = krnl.load [[VAR_19_]]#5{{.}}[[VAR_34_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_86_:%.+]] = addf [[VAR_83_]], [[LOAD_VAR_19_MEM_6_]] : f32
// CHECK-DAG:           [[VAR_87_:%.+]] = addf [[VAR_86_]], [[LOAD_VAR_19_MEM_7_]] : f32
// CHECK-DAG:           [[LOAD_VAR_24_MEM_2_:%.+]] = krnl.load [[VAR_24_]]#1{{.}}[[VAR_34_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_89_:%.+]] = mulf [[LOAD_VAR_24_MEM_2_]], [[VAR_78_]] : f32
// CHECK-DAG:           [[VAR_90_:%.+]] = addf [[VAR_87_]], [[VAR_89_]] : f32
// CHECK-DAG:           [[RES_9_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_90_]], [[RES_9_]][] : memref<f32>
// CHECK:               [[VAR_92_:%.+]] = "onnx.Sigmoid"([[RES_9_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_92_MEM_:%.+]] = krnl.load [[VAR_92_]][] : memref<f32>
// CHECK-DAG:           [[RES_10_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_78_]], [[RES_10_]][] : memref<f32>
// CHECK:               [[VAR_95_:%.+]] = "onnx.Tanh"([[RES_10_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_VAR_95_MEM_:%.+]] = krnl.load [[VAR_95_]][] : memref<f32>
// CHECK:               [[VAR_97_:%.+]] = mulf [[LOAD_VAR_92_MEM_]], [[LOAD_VAR_95_MEM_]] : f32
// CHECK:               krnl.store [[VAR_78_]], [[RES_2_]]{{.}}[[VAR_34_1_]]#0, [[VAR_34_1_]]#1] : memref<2x4xf32>
// CHECK:               krnl.store [[VAR_97_]], [[RES_3_]]{{.}}[[VAR_34_1_]]#0, [[VAR_34_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_4_]]) with ([[LOOP_4_]] -> [[I_7_:%.+]] = 0 to 7) {
// CHECK-DAG:         [[RES_11_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]] = affine.apply #map3([[I_7_]])
// CHECK-DAG:         [[LOOP_5_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_5_]]#0, [[LOOP_5_]]#1) with ([[LOOP_5_]]#0 -> [[I_8_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_5_]]#1 -> [[I_9_:%.+]] = [[CST_0_]] to [[CST_3_]]) {
// CHECK:               [[LOAD_PARAM_0_MEM_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_5_]]#0, [[LOOP_5_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[LOOP_2_]], [[LOAD_PARAM_0_MEM_1_]]#0, [[LOAD_PARAM_0_MEM_1_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_2_]], [[RES_11_]]{{.}}[[LOAD_PARAM_0_MEM_1_]]#0, [[LOAD_PARAM_0_MEM_1_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[LOAD_PARAM_5_MEM_1_1_:%.+]] = "onnx.MatMul"([[RES_11_]], [[VAR_14_]]) : (memref<2x3xf32>, memref<3x16xf32>) -> memref<2x16xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]] = "onnx.MatMul"([[RES_1_]], [[VAR_15_]]) : (memref<2x4xf32>, memref<4x16xf32>) -> memref<2x16xf32>
// CHECK-DAG:         [[LOOP_6_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_6_]]#0, [[LOOP_6_]]#1) with ([[LOOP_6_]]#0 -> [[I_10_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_6_]]#1 -> [[I_11_:%.+]] = [[CST_0_]] to [[CST_4_]]) {
// CHECK:               [[LOAD_PARAM_0_MEM_1_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_6_]]#0, [[LOOP_6_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[RES_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_LOAD_PARAM_5_MEM_1_MEM_4_:%.+]] = krnl.load [[LOAD_PARAM_5_MEM_1_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x16xf32>
// CHECK-DAG:           [[VAR_38_1_:%.+]] = krnl.load [[LOOP_3_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_19_MEM_8_:%.+]] = addf [[LOAD_LOAD_PARAM_5_MEM_1_MEM_4_]], [[VAR_38_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_19_MEM_1_:%.+]] = krnl.load [[VAR_20_]]#0{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_41_1_:%.+]] = krnl.load [[VAR_20_]]#4{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_42_1_:%.+]] = addf [[LOAD_VAR_19_MEM_8_]], [[LOAD_VAR_19_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_24_MEM_3_:%.+]] = addf [[VAR_42_1_]], [[VAR_41_1_]] : f32
// CHECK-DAG:           [[VAR_44_1_:%.+]] = krnl.load [[VAR_25_]]#0{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_45_1_:%.+]] = mulf [[VAR_44_1_]], [[LOAD_PARAM_0_MEM_2_]] : f32
// CHECK-DAG:           [[RES_6_:%.+]] = addf [[LOAD_VAR_24_MEM_3_]], [[VAR_45_1_]] : f32
// CHECK-DAG:           [[RES_12_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[RES_6_]], [[RES_12_]][] : memref<f32>
// CHECK:               [[LOAD_VAR_47_MEM_1_:%.+]] = "onnx.Sigmoid"([[RES_12_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[VAR_49_1_:%.+]] = krnl.load [[LOAD_VAR_47_MEM_1_]][] : memref<f32>
// CHECK-DAG:           [[LOAD_LOAD_PARAM_4_MEM_1_MEM_1_:%.+]] = affine.apply #map0(){{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_51_1_:%.+]] = krnl.load [[LOAD_PARAM_5_MEM_1_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_LOAD_PARAM_4_MEM_1_MEM_1_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_LOAD_PARAM_5_MEM_1_MEM_1_:%.+]] = affine.apply #map0(){{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1]
// CHECK:               [[VAR_53_1_:%.+]] = krnl.load [[LOOP_3_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_LOAD_PARAM_5_MEM_1_MEM_1_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_VAR_19_MEM_2_:%.+]] = addf [[VAR_51_1_]], [[VAR_53_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_19_MEM_3_:%.+]] = krnl.load [[VAR_20_]]#2{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_56_1_:%.+]] = krnl.load [[VAR_20_]]#6{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_57_1_:%.+]] = addf [[LOAD_VAR_19_MEM_2_]], [[LOAD_VAR_19_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_24_MEM_1_:%.+]] = addf [[VAR_57_1_]], [[VAR_56_1_]] : f32
// CHECK-DAG:           [[VAR_59_1_:%.+]] = krnl.load [[VAR_25_]]#2{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_60_1_:%.+]] = mulf [[VAR_59_1_]], [[LOAD_PARAM_0_MEM_2_]] : f32
// CHECK-DAG:           [[RES_7_:%.+]] = addf [[LOAD_VAR_24_MEM_1_]], [[VAR_60_1_]] : f32
// CHECK-DAG:           [[RES_13_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[RES_7_]], [[RES_13_]][] : memref<f32>
// CHECK:               [[LOAD_VAR_62_MEM_1_:%.+]] = "onnx.Sigmoid"([[RES_13_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[VAR_64_1_:%.+]] = krnl.load [[LOAD_VAR_62_MEM_1_]][] : memref<f32>
// CHECK-DAG:           [[LOAD_LOAD_PARAM_4_MEM_1_MEM_2_:%.+]] = affine.apply #map1(){{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_66_1_:%.+]] = krnl.load [[LOAD_PARAM_5_MEM_1_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_LOAD_PARAM_4_MEM_1_MEM_2_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_LOAD_PARAM_5_MEM_1_MEM_2_:%.+]] = affine.apply #map1(){{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1]
// CHECK:               [[VAR_68_1_:%.+]] = krnl.load [[LOOP_3_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_LOAD_PARAM_5_MEM_1_MEM_2_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_VAR_19_MEM_4_:%.+]] = addf [[VAR_66_1_]], [[VAR_68_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_19_MEM_5_:%.+]] = krnl.load [[VAR_20_]]#3{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_71_1_:%.+]] = krnl.load [[VAR_20_]]#7{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_72_1_:%.+]] = addf [[LOAD_VAR_19_MEM_4_]], [[LOAD_VAR_19_MEM_5_]] : f32
// CHECK-DAG:           [[RES_8_:%.+]] = addf [[VAR_72_1_]], [[VAR_71_1_]] : f32
// CHECK-DAG:           [[RES_14_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[RES_8_]], [[RES_14_]][] : memref<f32>
// CHECK:               [[LOAD_VAR_74_MEM_1_:%.+]] = "onnx.Tanh"([[RES_14_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[VAR_76_1_:%.+]] = krnl.load [[LOAD_VAR_74_MEM_1_]][] : memref<f32>
// CHECK-DAG:           [[VAR_77_1_:%.+]] = mulf [[VAR_64_1_]], [[LOAD_PARAM_0_MEM_2_]] : f32
// CHECK:               [[VAR_78_1_:%.+]] = mulf [[VAR_49_1_]], [[VAR_76_1_]] : f32
// CHECK-DAG:           [[VAR_79_1_:%.+]] = addf [[VAR_77_1_]], [[VAR_78_1_]] : f32
// CHECK-DAG:           [[LOAD_LOAD_PARAM_4_MEM_1_MEM_3_:%.+]] = affine.apply #map2(){{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_81_1_:%.+]] = krnl.load [[LOAD_PARAM_5_MEM_1_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_LOAD_PARAM_4_MEM_1_MEM_3_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_LOAD_PARAM_5_MEM_1_MEM_3_:%.+]] = affine.apply #map2(){{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1]
// CHECK:               [[VAR_83_1_:%.+]] = krnl.load [[LOOP_3_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_LOAD_PARAM_5_MEM_1_MEM_3_]]{{.}} : memref<2x16xf32>
// CHECK-DAG:           [[LOAD_VAR_19_MEM_6_:%.+]] = addf [[VAR_81_1_]], [[VAR_83_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_19_MEM_7_:%.+]] = krnl.load [[VAR_20_]]#1{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_86_1_:%.+]] = krnl.load [[VAR_20_]]#5{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_87_1_:%.+]] = addf [[LOAD_VAR_19_MEM_6_]], [[LOAD_VAR_19_MEM_7_]] : f32
// CHECK-DAG:           [[LOAD_VAR_24_MEM_2_:%.+]] = addf [[VAR_87_1_]], [[VAR_86_1_]] : f32
// CHECK-DAG:           [[VAR_89_1_:%.+]] = krnl.load [[VAR_25_]]#1{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_90_1_:%.+]] = mulf [[VAR_89_1_]], [[VAR_79_1_]] : f32
// CHECK-DAG:           [[RES_9_:%.+]] = addf [[LOAD_VAR_24_MEM_2_]], [[VAR_90_1_]] : f32
// CHECK-DAG:           [[RES_15_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[RES_9_]], [[RES_15_]][] : memref<f32>
// CHECK:               [[LOAD_VAR_92_MEM_1_:%.+]] = "onnx.Sigmoid"([[RES_15_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[RES_10_:%.+]] = krnl.load [[LOAD_VAR_92_MEM_1_]][] : memref<f32>
// CHECK-DAG:           [[RES_16_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_79_1_]], [[RES_16_]][] : memref<f32>
// CHECK:               [[LOAD_VAR_95_MEM_1_:%.+]] = "onnx.Tanh"([[RES_16_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[VAR_97_1_:%.+]] = krnl.load [[LOAD_VAR_95_MEM_1_]][] : memref<f32>
// CHECK:               [[VAR_98_:%.+]] = mulf [[RES_10_]], [[VAR_97_1_]] : f32
// CHECK:               krnl.store [[VAR_79_1_]], [[RES_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK:               krnl.store [[VAR_98_]], [[RES_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOOP_7_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_7_]]#0, [[LOOP_7_]]#1) with ([[LOOP_7_]]#0 -> [[I_12_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_7_]]#1 -> [[I_13_:%.+]] = [[CST_0_]] to [[CST_4_]]) {
// CHECK:             [[RES_11_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_7_]]#0, [[LOOP_7_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOOP_2_1_:%.+]] = krnl.load [[RES_3_]]{{.}}[[RES_11_]]#0, [[RES_11_]]#1] : memref<2x4xf32>
// CHECK:             krnl.store [[LOOP_2_1_]], [[RES_4_]]{{.}}[[CST_0_]], [[RES_11_]]#0, [[RES_11_]]#1] : memref<2x2x4xf32>
// CHECK:             [[LOOP_5_:%.+]] = krnl.load [[RES_1_]]{{.}}[[RES_11_]]#0, [[RES_11_]]#1] : memref<2x4xf32>
// CHECK:             krnl.store [[LOOP_5_]], [[RES_4_]]{{.}}[[CST_1_]], [[RES_11_]]#0, [[RES_11_]]#1] : memref<2x2x4xf32>
// CHECK:           }
// CHECK:           return [[RES_4_]] : memref<2x2x4xf32>
// CHECK:         }

}

// -----

func private @test_lstm_unknown_dims(%arg0: tensor<?x?x?xf32>, %arg1: tensor<1x16x?xf32>, %arg2: tensor<1x16x4xf32>, %arg3: tensor<1x32xf32>, %arg4: tensor<1x?x4xf32>, %arg5: tensor<1x?x4xf32>, %arg6: tensor<1x12xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4, %arg5, %arg6) {hidden_size = 4 : si64} : (tensor<?x?x?xf32>, tensor<1x16x?xf32>, tensor<1x16x4xf32>, tensor<1x32xf32>, none, tensor<1x?x4xf32>, tensor<1x?x4xf32>, tensor<1x12xf32>) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

// CHECK-DAG: #map0 = affine_map<()[s0] -> (s0 + 8)>
// CHECK-DAG: #map1 = affine_map<()[s0] -> (s0 + 12)>
// CHECK-DAG: #map2 = affine_map<()[s0] -> (s0 + 4)>
// CHECK-LABEL:  builtin.func private @test_lstm_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x?xf32>, [[PARAM_1_:%.+]]: memref<1x16x?xf32>, [[PARAM_2_:%.+]]: memref<1x16x4xf32>, [[PARAM_3_:%.+]]: memref<1x32xf32>, [[PARAM_4_:%.+]]: memref<1x?x4xf32>, [[PARAM_5_:%.+]]: memref<1x?x4xf32>, [[PARAM_6_:%.+]]: memref<1x12xf32>) -> memref<1x?x4xf32> {
// CHECK-DAG:       [[CST_16_:%.+]] = constant 16 : i64
// CHECK-DAG:       [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:       [[CST_2_:%.+]] = constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = constant 1 : index
// CHECK:           [[VAR_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<1x?x4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[VAR_2_]]) {{.*}}: memref<?x4xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc([[VAR_4_]]) {{.*}}: memref<?x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[VAR_2_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4) {
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x?x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<?x4xf32>
// CHECK:             [[LOAD_PARAM_5_MEM_:%.+]] = krnl.load [[PARAM_5_]]{{.}}[[CST_0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x?x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_5_MEM_]], [[RES_2_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<?x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.SqueezeV11"([[PARAM_1_]]) {axes = [0]} : (memref<1x16x?xf32>) -> memref<16x?xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.SqueezeV11"([[PARAM_2_]]) {axes = [0]} : (memref<1x16x4xf32>) -> memref<16x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Transpose"([[VAR_7_]]) {perm = [1, 0]} : (memref<16x?xf32>) -> memref<?x16xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Transpose"([[VAR_8_]]) {perm = [1, 0]} : (memref<16x4xf32>) -> memref<4x16xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.SqueezeV11"([[PARAM_3_]]) {axes = [0]} : (memref<1x32xf32>) -> memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_12_:%.+]]:8 = "onnx.SplitV11"([[VAR_11_]]) {axis = 0 : si64} : (memref<32xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.SqueezeV11"([[PARAM_6_]]) {axes = [0]} : (memref<1x12xf32>) -> memref<12xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_14_:%.+]]:3 = "onnx.SplitV11"([[VAR_13_]]) {axis = 0 : si64} : (memref<12xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK-DAG:       [[VAR_16_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x?xf32>
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to [[VAR_16_]]) {
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x?xf32>
// CHECK-DAG:         [[LOAD_PARAM_5_MEM_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<?x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc([[LOAD_PARAM_4_MEM_1_]], [[LOAD_PARAM_5_MEM_1_]]) {{.*}}: memref<?x?xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_]] to [[LOAD_PARAM_4_MEM_1_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_]] to [[LOAD_PARAM_5_MEM_1_]]) {
// CHECK:               [[VAR_26_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_26_]]#0, [[VAR_26_]]#1] : memref<?x?x?xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_3_]]{{.}}[[VAR_26_]]#0, [[VAR_26_]]#1] : memref<?x?xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_23_:%.+]] = "onnx.MatMul"([[RES_3_]], [[VAR_9_]]) : (memref<?x?xf32>, memref<?x16xf32>) -> memref<?x16xf32>
// CHECK-DAG:         [[VAR_24_:%.+]] = "onnx.MatMul"([[RES_1_]], [[VAR_10_]]) : (memref<?x4xf32>, memref<4x16xf32>) -> memref<?x16xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_]] to [[VAR_2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_]] to [[CST_4_]]) {
// CHECK:               [[VAR_26_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<?x4xf32>
// CHECK-DAG:           [[LOAD_VAR_23_MEM_:%.+]] = krnl.load [[VAR_23_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<?x16xf32>
// CHECK-DAG:           [[LOAD_VAR_24_MEM_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<?x16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_30_:%.+]] = addf [[LOAD_VAR_23_MEM_]], [[LOAD_VAR_24_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_12_MEM_:%.+]] = krnl.load [[VAR_12_]]#0{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_12_MEM_1_:%.+]] = krnl.load [[VAR_12_]]#4{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_33_:%.+]] = addf [[VAR_30_]], [[LOAD_VAR_12_MEM_]] : f32
// CHECK-DAG:           [[VAR_34_:%.+]] = addf [[VAR_33_]], [[LOAD_VAR_12_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_14_MEM_:%.+]] = krnl.load [[VAR_14_]]#0{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_36_:%.+]] = mulf [[LOAD_VAR_14_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_37_:%.+]] = addf [[VAR_34_]], [[VAR_36_]] : f32
// CHECK-DAG:           [[RES_4_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_37_]], [[RES_4_]][] : memref<f32>
// CHECK:               [[VAR_39_:%.+]] = "onnx.Sigmoid"([[RES_4_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_39_MEM_:%.+]] = krnl.load [[VAR_39_]][] : memref<f32>
// CHECK-DAG:           [[VAR_41_:%.+]] = affine.apply #map0(){{.}}[[VAR_26_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_23_MEM_1_:%.+]] = krnl.load [[VAR_23_]]{{.}}[[VAR_26_1_]]#0, [[VAR_41_]]{{.}} : memref<?x16xf32>
// CHECK-DAG:           [[VAR_43_:%.+]] = affine.apply #map0(){{.}}[[VAR_26_1_]]#1]
// CHECK:               [[LOAD_VAR_24_MEM_1_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_26_1_]]#0, [[VAR_43_]]{{.}} : memref<?x16xf32>
// CHECK-DAG:           [[VAR_45_:%.+]] = addf [[LOAD_VAR_23_MEM_1_]], [[LOAD_VAR_24_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_12_MEM_2_:%.+]] = krnl.load [[VAR_12_]]#2{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_12_MEM_3_:%.+]] = krnl.load [[VAR_12_]]#6{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_48_:%.+]] = addf [[VAR_45_]], [[LOAD_VAR_12_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_49_:%.+]] = addf [[VAR_48_]], [[LOAD_VAR_12_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_14_MEM_1_:%.+]] = krnl.load [[VAR_14_]]#2{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_51_:%.+]] = mulf [[LOAD_VAR_14_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_52_:%.+]] = addf [[VAR_49_]], [[VAR_51_]] : f32
// CHECK-DAG:           [[RES_5_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_52_]], [[RES_5_]][] : memref<f32>
// CHECK:               [[VAR_54_:%.+]] = "onnx.Sigmoid"([[RES_5_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_54_MEM_:%.+]] = krnl.load [[VAR_54_]][] : memref<f32>
// CHECK-DAG:           [[VAR_56_:%.+]] = affine.apply #map1(){{.}}[[VAR_26_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_23_MEM_2_:%.+]] = krnl.load [[VAR_23_]]{{.}}[[VAR_26_1_]]#0, [[VAR_56_]]{{.}} : memref<?x16xf32>
// CHECK-DAG:           [[VAR_58_:%.+]] = affine.apply #map1(){{.}}[[VAR_26_1_]]#1]
// CHECK:               [[LOAD_VAR_24_MEM_2_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_26_1_]]#0, [[VAR_58_]]{{.}} : memref<?x16xf32>
// CHECK-DAG:           [[VAR_60_:%.+]] = addf [[LOAD_VAR_23_MEM_2_]], [[LOAD_VAR_24_MEM_2_]] : f32
// CHECK-DAG:           [[LOAD_VAR_12_MEM_4_:%.+]] = krnl.load [[VAR_12_]]#3{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_12_MEM_5_:%.+]] = krnl.load [[VAR_12_]]#7{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_63_:%.+]] = addf [[VAR_60_]], [[LOAD_VAR_12_MEM_4_]] : f32
// CHECK-DAG:           [[VAR_64_:%.+]] = addf [[VAR_63_]], [[LOAD_VAR_12_MEM_5_]] : f32
// CHECK-DAG:           [[RES_6_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_64_]], [[RES_6_]][] : memref<f32>
// CHECK:               [[VAR_66_:%.+]] = "onnx.Tanh"([[RES_6_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_66_MEM_:%.+]] = krnl.load [[VAR_66_]][] : memref<f32>
// CHECK-DAG:           [[VAR_68_:%.+]] = mulf [[LOAD_VAR_54_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_69_:%.+]] = mulf [[LOAD_VAR_39_MEM_]], [[LOAD_VAR_66_MEM_]] : f32
// CHECK-DAG:           [[VAR_70_:%.+]] = addf [[VAR_68_]], [[VAR_69_]] : f32
// CHECK-DAG:           [[VAR_71_:%.+]] = affine.apply #map2(){{.}}[[VAR_26_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_23_MEM_3_:%.+]] = krnl.load [[VAR_23_]]{{.}}[[VAR_26_1_]]#0, [[VAR_71_]]{{.}} : memref<?x16xf32>
// CHECK-DAG:           [[VAR_73_:%.+]] = affine.apply #map2(){{.}}[[VAR_26_1_]]#1]
// CHECK:               [[LOAD_VAR_24_MEM_3_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_26_1_]]#0, [[VAR_73_]]{{.}} : memref<?x16xf32>
// CHECK-DAG:           [[VAR_75_:%.+]] = addf [[LOAD_VAR_23_MEM_3_]], [[LOAD_VAR_24_MEM_3_]] : f32
// CHECK-DAG:           [[LOAD_VAR_12_MEM_6_:%.+]] = krnl.load [[VAR_12_]]#1{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_12_MEM_7_:%.+]] = krnl.load [[VAR_12_]]#5{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_78_:%.+]] = addf [[VAR_75_]], [[LOAD_VAR_12_MEM_6_]] : f32
// CHECK-DAG:           [[VAR_79_:%.+]] = addf [[VAR_78_]], [[LOAD_VAR_12_MEM_7_]] : f32
// CHECK-DAG:           [[LOAD_VAR_14_MEM_2_:%.+]] = krnl.load [[VAR_14_]]#1{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_81_:%.+]] = mulf [[LOAD_VAR_14_MEM_2_]], [[VAR_70_]] : f32
// CHECK-DAG:           [[VAR_82_:%.+]] = addf [[VAR_79_]], [[VAR_81_]] : f32
// CHECK-DAG:           [[RES_7_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_82_]], [[RES_7_]][] : memref<f32>
// CHECK:               [[VAR_84_:%.+]] = "onnx.Sigmoid"([[RES_7_]]) : (memref<f32>) -> memref<f32>
// CHECK-DAG:           [[LOAD_VAR_84_MEM_:%.+]] = krnl.load [[VAR_84_]][] : memref<f32>
// CHECK-DAG:           [[RES_8_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_70_]], [[RES_8_]][] : memref<f32>
// CHECK:               [[VAR_87_:%.+]] = "onnx.Tanh"([[RES_8_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_VAR_87_MEM_:%.+]] = krnl.load [[VAR_87_]][] : memref<f32>
// CHECK:               [[VAR_89_:%.+]] = mulf [[LOAD_VAR_84_MEM_]], [[LOAD_VAR_87_MEM_]] : f32
// CHECK:               krnl.store [[VAR_70_]], [[RES_2_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<?x4xf32>
// CHECK:               krnl.store [[VAR_89_]], [[RES_1_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<?x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           [[VAR_17_:%.+]] = index_cast [[VAR_2_]] : index to i64
// CHECK:           [[VAR_18_:%.+]] = muli [[VAR_17_]], [[CST_16_]] : i64
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_1_]], [[RES_]]8) : (memref<1x?x4xf32>, memref<?x4xf32>, i64) -> ()
// CHECK:           return [[RES_]] : memref<1x?x4xf32>
// CHECK:         } 

}
