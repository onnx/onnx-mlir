// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl='check-rnn-ops-lowering' --canonicalize %s -split-input-file | FileCheck %s

func private @test_rnn_forward_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x4x3xf32>, %arg2: tensor<1x4x4xf32>, %arg3: tensor<1x8xf32>, %arg4: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64} : (tensor<7x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x4xf32>, tensor<1x8xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  builtin.func private @test_rnn_forward_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x4x3xf32>, [[PARAM_2_:%.+]]: memref<1x4x4xf32>, [[PARAM_3_:%.+]]: memref<1x8xf32>, [[PARAM_4_:%.+]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[CST_32_:%.+]] = constant 32 : i64
// CHECK-DAG:       [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:       [[CST_3_:%.+]] = constant 3 : index
// CHECK-DAG:       [[CST_2_:%.+]] = constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1x2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4) {
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.SqueezeV11"([[PARAM_1_]]) {axes = [0]} : (memref<1x4x3xf32>) -> memref<4x3xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.SqueezeV11"([[PARAM_2_]]) {axes = [0]} : (memref<1x4x4xf32>) -> memref<4x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Transpose"([[VAR_3_]]) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Transpose"([[VAR_4_]]) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.SqueezeV11"([[PARAM_3_]]) {axes = [0]} : (memref<1x8xf32>) -> memref<8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]]:2 = "onnx.Split"([[VAR_7_]]) {axis = 0 : si64} : (memref<8xf32>) -> (memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7) {
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_]] to [[CST_3_]]) {
// CHECK:               [[VAR_15_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_15_]]#0, [[VAR_15_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_2_]]{{.}}[[VAR_15_]]#0, [[VAR_15_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_12_:%.+]] = "onnx.MatMul"([[RES_2_]], [[VAR_5_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_13_:%.+]] = "onnx.MatMul"([[RES_]], [[VAR_6_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_]] to [[CST_4_]]) {
// CHECK:               [[VAR_15_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_12_]]{{.}}[[VAR_15_1_]]#0, [[VAR_15_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_13_MEM_:%.+]] = krnl.load [[VAR_13_]]{{.}}[[VAR_15_1_]]#0, [[VAR_15_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_18_:%.+]] = addf [[LOAD_PARAM_0_MEM_1_]], [[LOAD_VAR_13_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_8_MEM_:%.+]] = krnl.load [[VAR_8_]]#0{{.}}[[VAR_15_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_8_MEM_1_:%.+]] = krnl.load [[VAR_8_]]#1{{.}}[[VAR_15_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_21_:%.+]] = addf [[VAR_18_]], [[LOAD_VAR_8_MEM_]] : f32
// CHECK-DAG:           [[VAR_22_:%.+]] = addf [[VAR_21_]], [[LOAD_VAR_8_MEM_1_]] : f32
// CHECK-DAG:           [[RES_3_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_22_]], [[RES_3_]][] : memref<f32>
// CHECK:               [[VAR_24_:%.+]] = "onnx.Tanh"([[RES_3_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_VAR_24_MEM_:%.+]] = krnl.load [[VAR_24_]][] : memref<f32>
// CHECK:               krnl.store [[LOAD_VAR_24_MEM_]], [[RES_]]{{.}}[[VAR_15_1_]]#0, [[VAR_15_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc [[RES_2_]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           "krnl.memcpy"([[RES_1_]], [[RES_]], [[CST_32_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
// CHECK:           memref.dealloc [[RES_]] : memref<2x4xf32>
// CHECK:           return [[RES_1_]] : memref<1x2x4xf32>
// CHECK:         }

}

// -----

func private @test_rnn_forward_mode_constant_weight_and_bias(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %w = "onnx.Constant"() {value = dense<[[[1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]]> : tensor<1x4x3xf32>} : () -> tensor<1x4x3xf32> 
  %r = "onnx.Constant"() {value = dense<[[[2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.]]]> : tensor<1x4x4xf32>} : () -> tensor<1x4x4xf32> 
  %b = "onnx.Constant"() {value = dense<[[1., 2., 3., 4., 5., 6., 7., 8.]]> : tensor<1x8xf32>} : () -> tensor<1x8xf32> 

  %Y, %Y_h = "onnx.RNN"(%arg0, %w, %r, %b, %cst, %arg1) {hidden_size = 4 : si64} : (tensor<7x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x4xf32>, tensor<1x8xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:  builtin.func private @test_rnn_forward_mode_constant_weight_and_bias
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[CST_32_:%.+]] = constant 32 : i64
// CHECK-DAG:       [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:       [[CST_3_:%.+]] = constant 3 : index
// CHECK-DAG:       [[CST_2_:%.+]] = constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1x2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4) {
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_3_:%.+]] = "krnl.global"() {name = "constant_5", shape = [3, 4], value = dense<1.000000e+00> : tensor<3x4xf32>} : () -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "krnl.global"() {name = "constant_6", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "krnl.global"() {name = "constant_8", shape = [4], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "krnl.global"() {name = "constant_9", shape = [4], value = dense<[5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7) {
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_]] to [[CST_3_]]) {
// CHECK:               [[VAR_13_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_13_]]#0, [[VAR_13_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_2_]]{{.}}[[VAR_13_]]#0, [[VAR_13_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_10_:%.+]] = "onnx.MatMul"([[RES_2_]], [[VAR_3_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_11_:%.+]] = "onnx.MatMul"([[RES_]], [[VAR_4_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_]] to [[CST_4_]]) {
// CHECK:               [[VAR_13_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_10_]]{{.}}[[VAR_13_1_]]#0, [[VAR_13_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_11_MEM_:%.+]] = krnl.load [[VAR_11_]]{{.}}[[VAR_13_1_]]#0, [[VAR_13_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_16_:%.+]] = addf [[LOAD_PARAM_0_MEM_1_]], [[LOAD_VAR_11_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_5_MEM_:%.+]] = krnl.load [[VAR_5_]]{{.}}[[VAR_13_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_6_MEM_:%.+]] = krnl.load [[VAR_6_]]{{.}}[[VAR_13_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_19_:%.+]] = addf [[VAR_16_]], [[LOAD_VAR_5_MEM_]] : f32
// CHECK-DAG:           [[VAR_20_:%.+]] = addf [[VAR_19_]], [[LOAD_VAR_6_MEM_]] : f32
// CHECK-DAG:           [[RES_3_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_20_]], [[RES_3_]][] : memref<f32>
// CHECK:               [[VAR_22_:%.+]] = "onnx.Tanh"([[RES_3_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_VAR_22_MEM_:%.+]] = krnl.load [[VAR_22_]][] : memref<f32>
// CHECK:               krnl.store [[LOAD_VAR_22_MEM_]], [[RES_]]{{.}}[[VAR_13_1_]]#0, [[VAR_13_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc [[RES_2_]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           "krnl.memcpy"([[RES_1_]], [[RES_]], [[CST_32_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
// CHECK:           memref.dealloc [[RES_]] : memref<2x4xf32>
// CHECK:           return [[RES_1_]] : memref<1x2x4xf32>
// CHECK:         }

}

// -----

func private @test_rnn_reverse_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x4x3xf32>, %arg2: tensor<1x4x4xf32>, %arg3: tensor<1x8xf32>, %arg4: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64, direction = "reverse"} : (tensor<7x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x4xf32>, tensor<1x8xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-DAG: #map = affine_map<(d0) -> (-d0 + 6)>
// CHECK-LABEL:  builtin.func private @test_rnn_reverse_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x4x3xf32>, [[PARAM_2_:%.+]]: memref<1x4x4xf32>, [[PARAM_3_:%.+]]: memref<1x8xf32>, [[PARAM_4_:%.+]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[CST_32_:%.+]] = constant 32 : i64
// CHECK-DAG:       [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:       [[CST_3_:%.+]] = constant 3 : index
// CHECK-DAG:       [[CST_2_:%.+]] = constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1x2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4) {
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.SqueezeV11"([[PARAM_1_]]) {axes = [0]} : (memref<1x4x3xf32>) -> memref<4x3xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.SqueezeV11"([[PARAM_2_]]) {axes = [0]} : (memref<1x4x4xf32>) -> memref<4x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Transpose"([[VAR_3_]]) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Transpose"([[VAR_4_]]) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.SqueezeV11"([[PARAM_3_]]) {axes = [0]} : (memref<1x8xf32>) -> memref<8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]]:2 = "onnx.Split"([[VAR_7_]]) {axis = 0 : si64} : (memref<8xf32>) -> (memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7) {
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[VAR_11_:%.+]] = affine.apply #map([[I_2_]])
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_]] to [[CST_3_]]) {
// CHECK:               [[VAR_16_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_11_]], [[VAR_16_]]#0, [[VAR_16_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_2_]]{{.}}[[VAR_16_]]#0, [[VAR_16_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_13_:%.+]] = "onnx.MatMul"([[RES_2_]], [[VAR_5_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_14_:%.+]] = "onnx.MatMul"([[RES_]], [[VAR_6_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_]] to [[CST_4_]]) {
// CHECK:               [[VAR_16_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_13_]]{{.}}[[VAR_16_1_]]#0, [[VAR_16_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_14_MEM_:%.+]] = krnl.load [[VAR_14_]]{{.}}[[VAR_16_1_]]#0, [[VAR_16_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_19_:%.+]] = addf [[LOAD_PARAM_0_MEM_1_]], [[LOAD_VAR_14_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_8_MEM_:%.+]] = krnl.load [[VAR_8_]]#0{{.}}[[VAR_16_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_8_MEM_1_:%.+]] = krnl.load [[VAR_8_]]#1{{.}}[[VAR_16_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_22_:%.+]] = addf [[VAR_19_]], [[LOAD_VAR_8_MEM_]] : f32
// CHECK-DAG:           [[VAR_23_:%.+]] = addf [[VAR_22_]], [[LOAD_VAR_8_MEM_1_]] : f32
// CHECK-DAG:           [[RES_3_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_23_]], [[RES_3_]][] : memref<f32>
// CHECK:               [[VAR_25_:%.+]] = "onnx.Tanh"([[RES_3_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_VAR_25_MEM_:%.+]] = krnl.load [[VAR_25_]][] : memref<f32>
// CHECK:               krnl.store [[LOAD_VAR_25_MEM_]], [[RES_]]{{.}}[[VAR_16_1_]]#0, [[VAR_16_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc [[RES_2_]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           "krnl.memcpy"([[RES_1_]], [[RES_]], [[CST_32_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
// CHECK:           memref.dealloc [[RES_]] : memref<2x4xf32>
// CHECK:           return [[RES_1_]] : memref<1x2x4xf32>
// CHECK:         }

}

// -----

func private @test_rnn_bidirectional_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<2x4x3xf32>, %arg2: tensor<2x4x4xf32>, %arg3: tensor<2x8xf32>, %arg4: tensor<2x2x4xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64, direction = "bidirectional"} : (tensor<7x2x3xf32>, tensor<2x4x3xf32>, tensor<2x4x4xf32>, tensor<2x8xf32>, none, tensor<2x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-DAG: #map = affine_map<(d0) -> (-d0 + 6)>
// CHECK-LABEL:  builtin.func private @test_rnn_bidirectional_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<2x4x3xf32>, [[PARAM_2_:%.+]]: memref<2x4x4xf32>, [[PARAM_3_:%.+]]: memref<2x8xf32>, [[PARAM_4_:%.+]]: memref<2x2x4xf32>) -> memref<2x2x4xf32> {
// CHECK-DAG:       [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:       [[CST_3_:%.+]] = constant 3 : index
// CHECK-DAG:       [[CST_2_:%.+]] = constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2x2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4) {
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_]], [[I_0_]], [[I_1_]]{{.}} : memref<2x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:             [[LOAD_PARAM_4_MEM_1_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_1_]], [[I_0_]], [[I_1_]]{{.}} : memref<2x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_1_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:           }
// CHECK:           [[VAR_4_:%.+]]:2 = "onnx.Split"([[PARAM_1_]]) {axis = 0 : si64} : (memref<2x4x3xf32>) -> (memref<1x4x3xf32>, memref<1x4x3xf32>)
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.SqueezeV11"([[VAR_4_]]#0) {axes = [0]} : (memref<1x4x3xf32>) -> memref<4x3xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.SqueezeV11"([[VAR_4_]]#1) {axes = [0]} : (memref<1x4x3xf32>) -> memref<4x3xf32>
// CHECK-DAG:       [[VAR_7_:%.+]]:2 = "onnx.Split"([[PARAM_2_]]) {axis = 0 : si64} : (memref<2x4x4xf32>) -> (memref<1x4x4xf32>, memref<1x4x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.SqueezeV11"([[VAR_7_]]#0) {axes = [0]} : (memref<1x4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.SqueezeV11"([[VAR_7_]]#1) {axes = [0]} : (memref<1x4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Transpose"([[VAR_5_]]) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Transpose"([[VAR_8_]]) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Transpose"([[VAR_6_]]) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Transpose"([[VAR_9_]]) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_14_:%.+]]:2 = "onnx.Split"([[PARAM_3_]]) {axis = 0 : si64} : (memref<2x8xf32>) -> (memref<1x8xf32>, memref<1x8xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.SqueezeV11"([[VAR_14_]]#0) {axes = [0]} : (memref<1x8xf32>) -> memref<8xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.SqueezeV11"([[VAR_14_]]#1) {axes = [0]} : (memref<1x8xf32>) -> memref<8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_17_:%.+]]:2 = "onnx.Split"([[VAR_15_]]) {axis = 0 : si64} : (memref<8xf32>) -> (memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[VAR_18_:%.+]]:2 = "onnx.Split"([[VAR_16_]]) {axis = 0 : si64} : (memref<8xf32>) -> (memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7) {
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_]] to [[CST_3_]]) {
// CHECK:               [[VAR_27_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_27_]]#0, [[VAR_27_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_3_]]{{.}}[[VAR_27_]]#0, [[VAR_27_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_24_:%.+]] = "onnx.MatMul"([[RES_3_]], [[VAR_10_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_25_:%.+]] = "onnx.MatMul"([[RES_1_]], [[RES_1_]]1) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_]] to [[CST_4_]]) {
// CHECK:               [[VAR_27_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_27_1_]]#0, [[VAR_27_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_25_MEM_:%.+]] = krnl.load [[VAR_25_]]{{.}}[[VAR_27_1_]]#0, [[VAR_27_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_30_:%.+]] = addf [[LOAD_PARAM_0_MEM_1_]], [[LOAD_VAR_25_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_17_MEM_:%.+]] = krnl.load [[VAR_17_]]#0{{.}}[[VAR_27_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_17_MEM_1_:%.+]] = krnl.load [[VAR_17_]]#1{{.}}[[VAR_27_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_33_:%.+]] = addf [[VAR_30_]], [[LOAD_VAR_17_MEM_]] : f32
// CHECK-DAG:           [[VAR_34_:%.+]] = addf [[VAR_33_]], [[LOAD_VAR_17_MEM_1_]] : f32
// CHECK-DAG:           [[RES_4_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_34_]], [[RES_4_]][] : memref<f32>
// CHECK:               [[VAR_36_:%.+]] = "onnx.Tanh"([[RES_4_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_VAR_36_MEM_:%.+]] = krnl.load [[VAR_36_]][] : memref<f32>
// CHECK:               krnl.store [[LOAD_VAR_36_MEM_]], [[RES_1_]]{{.}}[[VAR_27_1_]]#0, [[VAR_27_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc [[RES_3_]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_4_]]) with ([[LOOP_4_]] -> [[I_7_:%.+]] = 0 to 7) {
// CHECK-DAG:         [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]] = affine.apply #map([[I_7_]])
// CHECK-DAG:         [[LOOP_5_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_5_]]#0, [[LOOP_5_]]#1) with ([[LOOP_5_]]#0 -> [[I_8_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_5_]]#1 -> [[I_9_:%.+]] = [[CST_0_]] to [[CST_3_]]) {
// CHECK:               [[LOAD_PARAM_0_MEM_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_5_]]#0, [[LOOP_5_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[LOOP_2_]], [[LOAD_PARAM_0_MEM_1_]]#0, [[LOAD_PARAM_0_MEM_1_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_2_]], [[RES_5_]]{{.}}[[LOAD_PARAM_0_MEM_1_]]#0, [[LOAD_PARAM_0_MEM_1_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_25_1_:%.+]] = "onnx.MatMul"([[RES_5_]], [[VAR_12_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]] = "onnx.MatMul"([[RES_]], [[VAR_13_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[LOOP_6_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_6_]]#0, [[LOOP_6_]]#1) with ([[LOOP_6_]]#0 -> [[I_10_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_6_]]#1 -> [[I_11_:%.+]] = [[CST_0_]] to [[CST_4_]]) {
// CHECK:               [[LOAD_PARAM_0_MEM_1_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_6_]]#0, [[LOOP_6_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[VAR_25_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[VAR_30_1_:%.+]] = krnl.load [[LOOP_3_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_17_MEM_2_:%.+]] = addf [[LOAD_PARAM_0_MEM_2_]], [[VAR_30_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_17_MEM_1_:%.+]] = krnl.load [[VAR_18_]]#0{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_33_1_:%.+]] = krnl.load [[VAR_18_]]#1{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_34_1_:%.+]] = addf [[LOAD_VAR_17_MEM_2_]], [[LOAD_VAR_17_MEM_1_]] : f32
// CHECK-DAG:           [[RES_4_:%.+]] = addf [[VAR_34_1_]], [[VAR_33_1_]] : f32
// CHECK-DAG:           [[RES_6_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[RES_4_]], [[RES_6_]][] : memref<f32>
// CHECK:               [[LOAD_VAR_36_MEM_1_:%.+]] = "onnx.Tanh"([[RES_6_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_LOAD_VAR_36_MEM_1_MEM_:%.+]] = krnl.load [[LOAD_VAR_36_MEM_1_]][] : memref<f32>
// CHECK:               krnl.store [[LOAD_LOAD_VAR_36_MEM_1_MEM_]], [[RES_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc [[RES_5_]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           [[LOOP_7_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_7_]]#0, [[LOOP_7_]]#1) with ([[LOOP_7_]]#0 -> [[I_12_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_7_]]#1 -> [[I_13_:%.+]] = [[CST_0_]] to [[CST_4_]]) {
// CHECK:             [[RES_5_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_7_]]#0, [[LOOP_7_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOOP_2_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[RES_5_]]#0, [[RES_5_]]#1] : memref<2x4xf32>
// CHECK:             krnl.store [[LOOP_2_1_]], [[RES_2_]]{{.}}[[CST_0_]], [[RES_2_]]2#0, [[RES_2_]]2#1] : memref<2x2x4xf32>
// CHECK:             [[LOOP_5_:%.+]] = krnl.load [[RES_]]{{.}}[[RES_5_]]#0, [[RES_5_]]#1] : memref<2x4xf32>
// CHECK:             krnl.store [[LOOP_5_]], [[RES_2_]]{{.}}[[CST_1_]], [[RES_2_]]2#0, [[RES_2_]]2#1] : memref<2x2x4xf32>
// CHECK:           }
// CHECK:           memref.dealloc [[RES_1_]] : memref<2x4xf32>
// CHECK:           memref.dealloc [[RES_]] : memref<2x4xf32>
// CHECK:           return [[RES_2_]] : memref<2x2x4xf32>
// CHECK:         }

}

// -----

func private @test_rnn_unknown_dims(%arg0: tensor<?x?x?xf32>, %arg1: tensor<1x4x?xf32>, %arg2: tensor<1x4x4xf32>, %arg3: tensor<1x8xf32>, %arg4: tensor<1x?x4xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64} : (tensor<?x?x?xf32>, tensor<1x4x?xf32>, tensor<1x4x4xf32>, tensor<1x8xf32>, none, tensor<1x?x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:  builtin.func private @test_rnn_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x?xf32>, [[PARAM_1_:%.+]]: memref<1x4x?xf32>, [[PARAM_2_:%.+]]: memref<1x4x4xf32>, [[PARAM_3_:%.+]]: memref<1x8xf32>, [[PARAM_4_:%.+]]: memref<1x?x4xf32>) -> memref<1x?x4xf32> {
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
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[VAR_2_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4) {
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x?x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<?x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.SqueezeV11"([[PARAM_1_]]) {axes = [0]} : (memref<1x4x?xf32>) -> memref<4x?xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.SqueezeV11"([[PARAM_2_]]) {axes = [0]} : (memref<1x4x4xf32>) -> memref<4x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Transpose"([[VAR_5_]]) {perm = [1, 0]} : (memref<4x?xf32>) -> memref<?x4xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Transpose"([[VAR_6_]]) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.SqueezeV11"([[PARAM_3_]]) {axes = [0]} : (memref<1x8xf32>) -> memref<8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_10_:%.+]]:2 = "onnx.Split"([[VAR_9_]]) {axis = 0 : si64} : (memref<8xf32>) -> (memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK-DAG:       [[VAR_12_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x?xf32>
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to [[VAR_12_]]) {
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x?xf32>
// CHECK-DAG:         [[VAR_16_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<?x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloc([[LOAD_PARAM_4_MEM_1_]], [[VAR_16_]]) {{.*}}: memref<?x?xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_]] to [[LOAD_PARAM_4_MEM_1_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_]] to [[VAR_16_]]) {
// CHECK:               [[VAR_22_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_22_]]#0, [[VAR_22_]]#1] : memref<?x?x?xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_2_]]{{.}}[[VAR_22_]]#0, [[VAR_22_]]#1] : memref<?x?xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_19_:%.+]] = "onnx.MatMul"([[RES_2_]], [[VAR_7_]]) : (memref<?x?xf32>, memref<?x4xf32>) -> memref<?x4xf32>
// CHECK-DAG:         [[VAR_20_:%.+]] = "onnx.MatMul"([[RES_1_]], [[VAR_8_]]) : (memref<?x4xf32>, memref<4x4xf32>) -> memref<?x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_]] to [[VAR_2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_]] to [[CST_4_]]) {
// CHECK:               [[VAR_22_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_19_]]{{.}}[[VAR_22_1_]]#0, [[VAR_22_1_]]#1] : memref<?x4xf32>
// CHECK-DAG:           [[LOAD_VAR_20_MEM_:%.+]] = krnl.load [[VAR_20_]]{{.}}[[VAR_22_1_]]#0, [[VAR_22_1_]]#1] : memref<?x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_25_:%.+]] = addf [[LOAD_PARAM_0_MEM_1_]], [[LOAD_VAR_20_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_10_MEM_:%.+]] = krnl.load [[VAR_10_]]#0{{.}}[[VAR_22_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_10_MEM_1_:%.+]] = krnl.load [[VAR_10_]]#1{{.}}[[VAR_22_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_28_:%.+]] = addf [[VAR_25_]], [[LOAD_VAR_10_MEM_]] : f32
// CHECK-DAG:           [[VAR_29_:%.+]] = addf [[VAR_28_]], [[LOAD_VAR_10_MEM_1_]] : f32
// CHECK-DAG:           [[RES_3_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_29_]], [[RES_3_]][] : memref<f32>
// CHECK:               [[VAR_31_:%.+]] = "onnx.Tanh"([[RES_3_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_VAR_31_MEM_:%.+]] = krnl.load [[VAR_31_]][] : memref<f32>
// CHECK:               krnl.store [[LOAD_VAR_31_MEM_]], [[RES_1_]]{{.}}[[VAR_22_1_]]#0, [[VAR_22_1_]]#1] : memref<?x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc [[RES_2_]] : memref<?x?xf32>
// CHECK:           }
// CHECK:           [[VAR_13_:%.+]] = index_cast [[VAR_2_]] : index to i64
// CHECK:           [[VAR_14_:%.+]] = muli [[VAR_13_]], [[CST_16_]] : i64
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_1_]], [[RES_]]4) : (memref<1x?x4xf32>, memref<?x4xf32>, i64) -> ()
// CHECK:           memref.dealloc [[RES_1_]] : memref<?x4xf32>
// CHECK:           return [[RES_]] : memref<1x?x4xf32>
// CHECK:         }

}
