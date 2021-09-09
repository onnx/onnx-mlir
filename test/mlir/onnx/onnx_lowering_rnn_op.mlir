// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl='check-rnn-ops-lowering' %s -split-input-file | FileCheck %s

func private @test_rnn_forward_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x4x3xf32>, %arg2: tensor<1x4x4xf32>, %arg3: tensor<1x8xf32>, %arg4: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64} : (tensor<7x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x4xf32>, tensor<1x8xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:  func private @test_rnn_forward_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x4x3xf32>, [[PARAM_2_:%.+]]: memref<1x4x4xf32>, [[PARAM_3_:%.+]]: memref<1x8xf32>, [[PARAM_4_:%.+]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
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
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.SqueezeV11"([[PARAM_1_]]) {axes = [0]} : (memref<1x4x3xf32>) -> memref<4x3xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.SqueezeV11"([[PARAM_2_]]) {axes = [0]} : (memref<1x4x4xf32>) -> memref<4x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Transpose"([[VAR_3_]]) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Transpose"([[VAR_4_]]) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.SqueezeV11"([[PARAM_3_]]) {axes = [0]} : (memref<1x8xf32>) -> memref<8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]]:2 = "onnx.SplitV11"([[VAR_7_]]) {axis = 0 : si64} : (memref<8xf32>) -> (memref<4xf32>, memref<4xf32>)
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
// CHECK-DAG:         [[VAR_12_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_4_MEM_1_]], [[VAR_5_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_13_:%.+]] = "onnx.MatMul"([[VAR_0_]], [[VAR_6_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[CST_0_4_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_5_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_1_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_2_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_4_]] to [[CST_2_1_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_4_]] to [[CST_4_]]) {
// CHECK:               [[VAR_15_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_12_]]{{.}}[[VAR_15_1_]]#0, [[VAR_15_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_13_MEM_:%.+]] = krnl.load [[VAR_13_]]{{.}}[[VAR_15_1_]]#0, [[VAR_15_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_18_:%.+]] = addf [[LOAD_PARAM_0_MEM_1_]], [[LOAD_VAR_13_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_8_MEM_:%.+]] = krnl.load [[VAR_8_]]#0{{.}}[[VAR_15_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_8_MEM_1_:%.+]] = krnl.load [[VAR_8_]]#1{{.}}[[VAR_15_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_21_:%.+]] = addf [[VAR_18_]], [[LOAD_VAR_8_MEM_]] : f32
// CHECK-DAG:           [[VAR_22_:%.+]] = addf [[VAR_21_]], [[LOAD_VAR_8_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_23_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_22_]], [[VAR_23_]][] : memref<f32>
// CHECK:               [[VAR_24_:%.+]] = "onnx.Tanh"([[VAR_23_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_VAR_24_MEM_:%.+]] = krnl.load [[VAR_24_]][] : memref<f32>
// CHECK:               krnl.store [[LOAD_VAR_24_MEM_]], [[VAR_0_]]{{.}}[[VAR_15_1_]]#0, [[VAR_15_1_]]#1] : memref<2x4xf32>
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

func private @test_rnn_forward_mode_constant_weight_and_bias(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %w = "onnx.Constant"() {value = dense<[[[1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]]> : tensor<1x4x3xf32>} : () -> tensor<1x4x3xf32> 
  %r = "onnx.Constant"() {value = dense<[[[2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.]]]> : tensor<1x4x4xf32>} : () -> tensor<1x4x4xf32> 
  %b = "onnx.Constant"() {value = dense<[[1., 2., 3., 4., 5., 6., 7., 8.]]> : tensor<1x8xf32>} : () -> tensor<1x8xf32> 

  %Y, %Y_h = "onnx.RNN"(%arg0, %w, %r, %b, %cst, %arg1) {hidden_size = 4 : si64} : (tensor<7x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x4xf32>, tensor<1x8xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:  func private @test_rnn_forward_mode_constant_weight_and_bias
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.alloc() {{.*}}: memref<1x2x4xf32>
// CHECK-DAG:       [[VAR_cst_:%.+]] = constant unit
// CHECK-DAG:       [[VAR_2_:%.+]] = "krnl.global"() {name = "constant_0", shape = [1, 4, 3], value = dense<1.000000e+00> : tensor<1x4x3xf32>} : () -> memref<1x4x3xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "krnl.global"() {name = "constant_1", shape = [1, 4, 4], value = dense<2.000000e+00> : tensor<1x4x4xf32>} : () -> memref<1x4x4xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "krnl.global"() {name = "constant_2", shape = [1, 8], value = dense<{{.}}[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]{{.}}> : tensor<1x8xf32>} : () -> memref<1x8xf32>
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = constant 1 : index
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4) {
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_]], [[I_0_]], [[I_1_]]{{.}} : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[VAR_0_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_6_:%.+]] = "krnl.global"() {name = "constant_3", shape = [4, 3], value = dense<1.000000e+00> : tensor<4x3xf32>} : () -> memref<4x3xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "krnl.global"() {name = "constant_4", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "krnl.global"() {name = "constant_5", shape = [3, 4], value = dense<1.000000e+00> : tensor<3x4xf32>} : () -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "krnl.global"() {name = "constant_6", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "krnl.global"() {name = "constant_7", shape = [8], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]> : tensor<8xf32>} : () -> memref<8xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "krnl.global"() {name = "constant_8", shape = [4], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "krnl.global"() {name = "constant_9", shape = [4], value = dense<[5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7) {
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_1_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[CST_0_1_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_2_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_3_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_1_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_3_:%.+]] = constant 3 : index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_2_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_2_]] to [[CST_3_]]) {
// CHECK:               [[VAR_19_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_19_]]#0, [[VAR_19_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_1_]]{{.}}[[VAR_19_]]#0, [[VAR_19_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_16_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_1_MEM_1_]], [[VAR_8_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_17_:%.+]] = "onnx.MatMul"([[VAR_0_]], [[VAR_9_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[CST_0_4_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_5_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_1_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_2_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_4_]] to [[CST_2_1_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_4_]] to [[CST_4_]]) {
// CHECK:               [[VAR_19_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_19_1_]]#0, [[VAR_19_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_17_MEM_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_19_1_]]#0, [[VAR_19_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_22_:%.+]] = addf [[LOAD_PARAM_0_MEM_1_]], [[LOAD_VAR_17_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_:%.+]] = krnl.load [[VAR_11_]]{{.}}[[VAR_19_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_12_MEM_:%.+]] = krnl.load [[VAR_12_]]{{.}}[[VAR_19_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_25_:%.+]] = addf [[VAR_22_]], [[LOAD_VAR_11_MEM_]] : f32
// CHECK-DAG:           [[VAR_26_:%.+]] = addf [[VAR_25_]], [[LOAD_VAR_12_MEM_]] : f32
// CHECK-DAG:           [[VAR_27_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_26_]], [[VAR_27_]][] : memref<f32>
// CHECK:               [[VAR_28_:%.+]] = "onnx.Tanh"([[VAR_27_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_VAR_28_MEM_:%.+]] = krnl.load [[VAR_28_]][] : memref<f32>
// CHECK:               krnl.store [[LOAD_VAR_28_MEM_]], [[VAR_0_]]{{.}}[[VAR_19_1_]]#0, [[VAR_19_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc [[LOAD_PARAM_1_MEM_1_]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           [[CST_32_:%.+]] = constant 32 : i64
// CHECK:           "krnl.memcpy"([[VAR_1_]], [[VAR_0_]], [[CST_32_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
// CHECK:           memref.dealloc [[VAR_0_]] : memref<2x4xf32>
// CHECK:           return [[VAR_1_]] : memref<1x2x4xf32>
// CHECK:         }

}

// -----

func private @test_rnn_reverse_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x4x3xf32>, %arg2: tensor<1x4x4xf32>, %arg3: tensor<1x8xf32>, %arg4: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64, direction = "reverse"} : (tensor<7x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x4xf32>, tensor<1x8xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:  func private @test_rnn_reverse_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x4x3xf32>, [[PARAM_2_:%.+]]: memref<1x4x4xf32>, [[PARAM_3_:%.+]]: memref<1x8xf32>, [[PARAM_4_:%.+]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
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
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.SqueezeV11"([[PARAM_1_]]) {axes = [0]} : (memref<1x4x3xf32>) -> memref<4x3xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.SqueezeV11"([[PARAM_2_]]) {axes = [0]} : (memref<1x4x4xf32>) -> memref<4x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Transpose"([[VAR_3_]]) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Transpose"([[VAR_4_]]) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.SqueezeV11"([[PARAM_3_]]) {axes = [0]} : (memref<1x8xf32>) -> memref<8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]]:2 = "onnx.SplitV11"([[VAR_7_]]) {axis = 0 : si64} : (memref<8xf32>) -> (memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7) {
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_1_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[CST_0_1_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_7_:%.+]] = constant 7 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_11_:%.+]] = affine.apply #map([[I_2_]]){{.}}[[CST_7_]]{{.}}
// CHECK-DAG:         [[CST_0_2_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_3_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_1_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_3_:%.+]] = constant 3 : index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_2_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_2_]] to [[CST_3_]]) {
// CHECK:               [[VAR_16_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_11_]], [[VAR_16_]]#0, [[VAR_16_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_4_MEM_1_]]{{.}}[[VAR_16_]]#0, [[VAR_16_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_13_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_4_MEM_1_]], [[VAR_5_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_14_:%.+]] = "onnx.MatMul"([[VAR_0_]], [[VAR_6_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[CST_0_4_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_5_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_1_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_2_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_4_]] to [[CST_2_1_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_4_]] to [[CST_4_]]) {
// CHECK:               [[VAR_16_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_13_]]{{.}}[[VAR_16_1_]]#0, [[VAR_16_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_14_MEM_:%.+]] = krnl.load [[VAR_14_]]{{.}}[[VAR_16_1_]]#0, [[VAR_16_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_19_:%.+]] = addf [[LOAD_PARAM_0_MEM_1_]], [[LOAD_VAR_14_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_8_MEM_:%.+]] = krnl.load [[VAR_8_]]#0{{.}}[[VAR_16_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_8_MEM_1_:%.+]] = krnl.load [[VAR_8_]]#1{{.}}[[VAR_16_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_22_:%.+]] = addf [[VAR_19_]], [[LOAD_VAR_8_MEM_]] : f32
// CHECK-DAG:           [[VAR_23_:%.+]] = addf [[VAR_22_]], [[LOAD_VAR_8_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_24_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_23_]], [[VAR_24_]][] : memref<f32>
// CHECK:               [[VAR_25_:%.+]] = "onnx.Tanh"([[VAR_24_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_VAR_25_MEM_:%.+]] = krnl.load [[VAR_25_]][] : memref<f32>
// CHECK:               krnl.store [[LOAD_VAR_25_MEM_]], [[VAR_0_]]{{.}}[[VAR_16_1_]]#0, [[VAR_16_1_]]#1] : memref<2x4xf32>
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

func private @test_rnn_bidirectional_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<2x4x3xf32>, %arg2: tensor<2x4x4xf32>, %arg3: tensor<2x8xf32>, %arg4: tensor<2x2x4xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64, direction = "bidirectional"} : (tensor<7x2x3xf32>, tensor<2x4x3xf32>, tensor<2x4x4xf32>, tensor<2x8xf32>, none, tensor<2x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:  func private @test_rnn_bidirectional_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<2x4x3xf32>, [[PARAM_2_:%.+]]: memref<2x4x4xf32>, [[PARAM_3_:%.+]]: memref<2x8xf32>, [[PARAM_4_:%.+]]: memref<2x2x4xf32>) -> memref<2x2x4xf32> {
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
// CHECK:           [[VAR_4_:%.+]]:2 = "onnx.SplitV11"([[PARAM_1_]]) {axis = 0 : si64} : (memref<2x4x3xf32>) -> (memref<1x4x3xf32>, memref<1x4x3xf32>)
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.SqueezeV11"([[VAR_4_]]#0) {axes = [0]} : (memref<1x4x3xf32>) -> memref<4x3xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.SqueezeV11"([[VAR_4_]]#1) {axes = [0]} : (memref<1x4x3xf32>) -> memref<4x3xf32>
// CHECK-DAG:       [[VAR_7_:%.+]]:2 = "onnx.SplitV11"([[PARAM_2_]]) {axis = 0 : si64} : (memref<2x4x4xf32>) -> (memref<1x4x4xf32>, memref<1x4x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.SqueezeV11"([[VAR_7_]]#0) {axes = [0]} : (memref<1x4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.SqueezeV11"([[VAR_7_]]#1) {axes = [0]} : (memref<1x4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Transpose"([[VAR_5_]]) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Transpose"([[VAR_8_]]) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Transpose"([[VAR_6_]]) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Transpose"([[VAR_9_]]) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_14_:%.+]]:2 = "onnx.SplitV11"([[PARAM_3_]]) {axis = 0 : si64} : (memref<2x8xf32>) -> (memref<1x8xf32>, memref<1x8xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.SqueezeV11"([[VAR_14_]]#0) {axes = [0]} : (memref<1x8xf32>) -> memref<8xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.SqueezeV11"([[VAR_14_]]#1) {axes = [0]} : (memref<1x8xf32>) -> memref<8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_17_:%.+]]:2 = "onnx.SplitV11"([[VAR_15_]]) {axis = 0 : si64} : (memref<8xf32>) -> (memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[VAR_18_:%.+]]:2 = "onnx.SplitV11"([[VAR_16_]]) {axis = 0 : si64} : (memref<8xf32>) -> (memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7) {
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_2_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[CST_0_1_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_2_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_3_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_1_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_3_:%.+]] = constant 3 : index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_2_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_2_]] to [[CST_3_]]) {
// CHECK:               [[VAR_27_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_27_]]#0, [[VAR_27_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_4_MEM_2_]]{{.}}[[VAR_27_]]#0, [[VAR_27_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_24_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_4_MEM_2_]], [[VAR_10_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_25_:%.+]] = "onnx.MatMul"([[VAR_1_]], [[VAR_1_]]1) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[CST_0_4_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_5_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_1_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_2_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_4_]] to [[CST_2_1_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_4_]] to [[CST_4_]]) {
// CHECK:               [[VAR_27_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_27_1_]]#0, [[VAR_27_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_25_MEM_:%.+]] = krnl.load [[VAR_25_]]{{.}}[[VAR_27_1_]]#0, [[VAR_27_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_30_:%.+]] = addf [[LOAD_PARAM_0_MEM_1_]], [[LOAD_VAR_25_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_17_MEM_:%.+]] = krnl.load [[VAR_17_]]#0{{.}}[[VAR_27_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_17_MEM_1_:%.+]] = krnl.load [[VAR_17_]]#1{{.}}[[VAR_27_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_33_:%.+]] = addf [[VAR_30_]], [[LOAD_VAR_17_MEM_]] : f32
// CHECK-DAG:           [[VAR_34_:%.+]] = addf [[VAR_33_]], [[LOAD_VAR_17_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_35_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_34_]], [[VAR_35_]][] : memref<f32>
// CHECK:               [[VAR_36_:%.+]] = "onnx.Tanh"([[VAR_35_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_VAR_36_MEM_:%.+]] = krnl.load [[VAR_36_]][] : memref<f32>
// CHECK:               krnl.store [[LOAD_VAR_36_MEM_]], [[VAR_1_]]{{.}}[[VAR_27_1_]]#0, [[VAR_27_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc [[LOAD_PARAM_4_MEM_2_]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_4_]]) with ([[LOOP_4_]] -> [[I_7_:%.+]] = 0 to 7) {
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_2_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[CST_1_3_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_7_:%.+]] = constant 7 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOOP_2_:%.+]] = affine.apply #map([[I_7_]]){{.}}[[CST_7_]]{{.}}
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
// CHECK-DAG:         [[VAR_25_1_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_4_MEM_2_]], [[VAR_12_]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]] = "onnx.MatMul"([[VAR_0_]], [[VAR_13_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[CST_0_8_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_9_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_2_3_:%.+]] = constant 2 : index
// CHECK-DAG:         [[CST_1_5_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_4_1_:%.+]] = constant 4 : index
// CHECK-DAG:         [[LOOP_6_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_6_]]#0, [[LOOP_6_]]#1) with ([[LOOP_6_]]#0 -> [[I_10_:%.+]] = [[CST_0_8_]] to [[CST_2_3_]], [[LOOP_6_]]#1 -> [[I_11_:%.+]] = [[CST_0_8_]] to [[CST_4_1_]]) {
// CHECK:               [[LOAD_PARAM_0_MEM_1_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_6_]]#0, [[LOOP_6_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[VAR_25_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[VAR_30_1_:%.+]] = krnl.load [[LOOP_3_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_17_MEM_2_:%.+]] = addf [[LOAD_PARAM_0_MEM_2_]], [[VAR_30_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_17_MEM_1_:%.+]] = krnl.load [[VAR_18_]]#0{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_33_1_:%.+]] = krnl.load [[VAR_18_]]#1{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_34_1_:%.+]] = addf [[LOAD_VAR_17_MEM_2_]], [[LOAD_VAR_17_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_35_1_:%.+]] = addf [[VAR_34_1_]], [[VAR_33_1_]] : f32
// CHECK-DAG:           [[VAR_36_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_35_1_]], [[VAR_36_1_]][] : memref<f32>
// CHECK:               [[LOAD_VAR_36_MEM_1_:%.+]] = "onnx.Tanh"([[VAR_36_1_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_LOAD_VAR_36_MEM_1_MEM_:%.+]] = krnl.load [[LOAD_VAR_36_MEM_1_]][] : memref<f32>
// CHECK:               krnl.store [[LOAD_LOAD_VAR_36_MEM_1_MEM_]], [[VAR_0_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc [[LOAD_PARAM_4_MEM_2_]] : memref<2x3xf32>
// CHECK:           }
// CHECK-DAG:       [[CST_0_10_:%.+]] = constant 0 : index
// CHECK-DAG:       [[CST_1_6_:%.+]] = constant 1 : index
// CHECK-DAG:       [[CST_0_11_:%.+]] = constant 0 : index
// CHECK-DAG:       [[CST_2_4_:%.+]] = constant 2 : index
// CHECK-DAG:       [[CST_1_7_:%.+]] = constant 1 : index
// CHECK-DAG:       [[CST_4_2_:%.+]] = constant 4 : index
// CHECK-DAG:       [[LOOP_7_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_7_]]#0, [[LOOP_7_]]#1) with ([[LOOP_7_]]#0 -> [[I_12_:%.+]] = [[CST_0_10_]] to [[CST_2_4_]], [[LOOP_7_]]#1 -> [[I_13_:%.+]] = [[CST_0_10_]] to [[CST_4_2_]]) {
// CHECK:             [[LOAD_PARAM_4_MEM_2_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_7_]]#0, [[LOOP_7_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOOP_2_1_:%.+]] = krnl.load [[VAR_1_]]{{.}}[[LOAD_PARAM_4_MEM_2_1_]]#0, [[LOAD_PARAM_4_MEM_2_1_]]#1] : memref<2x4xf32>
// CHECK:             krnl.store [[LOOP_2_1_]], [[VAR_2_]]{{.}}[[CST_0_10_]], [[VAR_2_]]2#0, [[VAR_2_]]2#1] : memref<2x2x4xf32>
// CHECK:             [[LOOP_5_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[LOAD_PARAM_4_MEM_2_1_]]#0, [[LOAD_PARAM_4_MEM_2_1_]]#1] : memref<2x4xf32>
// CHECK:             krnl.store [[LOOP_5_]], [[VAR_2_]]{{.}}[[CST_1_6_]], [[VAR_2_]]2#0, [[VAR_2_]]2#1] : memref<2x2x4xf32>
// CHECK:           }
// CHECK:           memref.dealloc [[VAR_1_]] : memref<2x4xf32>
// CHECK:           memref.dealloc [[VAR_0_]] : memref<2x4xf32>
// CHECK:           return [[VAR_2_]] : memref<2x2x4xf32>
// CHECK:         }

}

// -----

func private @test_rnn_unknown_dims(%arg0: tensor<?x?x?xf32>, %arg1: tensor<1x4x?xf32>, %arg2: tensor<1x4x4xf32>, %arg3: tensor<1x8xf32>, %arg4: tensor<1x?x4xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64} : (tensor<?x?x?xf32>, tensor<1x4x?xf32>, tensor<1x4x4xf32>, tensor<1x8xf32>, none, tensor<1x?x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:  func private @test_rnn_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x?xf32>, [[PARAM_1_:%.+]]: memref<1x4x?xf32>, [[PARAM_2_:%.+]]: memref<1x4x4xf32>, [[PARAM_3_:%.+]]: memref<1x8xf32>, [[PARAM_4_:%.+]]: memref<1x?x4xf32>) -> memref<1x?x4xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = constant unit
// CHECK-DAG:       [[CST_1_:%.+]] = constant 1 : index
// CHECK:           [[VAR_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<1x?x4xf32>
// CHECK-DAG:       [[CST_1_1_:%.+]] = constant 1 : index
// CHECK:           [[VAR_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = memref.alloc([[VAR_2_]]) {{.*}}: memref<?x4xf32>
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
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.SqueezeV11"([[PARAM_1_]]) {axes = [0]} : (memref<1x4x?xf32>) -> memref<4x?xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.SqueezeV11"([[PARAM_2_]]) {axes = [0]} : (memref<1x4x4xf32>) -> memref<4x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Transpose"([[VAR_6_]]) {perm = [1, 0]} : (memref<4x?xf32>) -> memref<?x4xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Transpose"([[VAR_7_]]) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.SqueezeV11"([[PARAM_3_]]) {axes = [0]} : (memref<1x8xf32>) -> memref<8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]]:2 = "onnx.SplitV11"([[VAR_10_]]) {axis = 0 : si64} : (memref<8xf32>) -> (memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK-DAG:       [[CST_0_2_:%.+]] = constant 0 : index
// CHECK:           [[VAR_13_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_2_]] : memref<?x?x?xf32>
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to [[VAR_13_]]) {
// CHECK-DAG:         [[CST_0_3_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_1_3_:%.+]] = constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_3_]] : memref<?x?x?xf32>
// CHECK-DAG:         [[CST_2_:%.+]] = constant 2 : index
// CHECK:             [[VAR_18_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<?x?x?xf32>
// CHECK-DAG:         [[VAR_19_:%.+]] = memref.alloc([[LOAD_PARAM_4_MEM_1_]], [[VAR_18_]]) {{.*}}: memref<?x?xf32>
// CHECK-DAG:         [[CST_0_4_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_5_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_1_4_:%.+]] = constant 1 : index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_4_]] to [[LOAD_PARAM_4_MEM_1_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_4_]] to [[VAR_18_]]) {
// CHECK:               [[VAR_24_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[VAR_24_]]#0, [[VAR_24_]]#1] : memref<?x?x?xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[VAR_19_]]{{.}}[[VAR_24_]]#0, [[VAR_24_]]#1] : memref<?x?xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_21_:%.+]] = "onnx.MatMul"([[VAR_19_]], [[VAR_8_]]) : (memref<?x?xf32>, memref<?x4xf32>) -> memref<?x4xf32>
// CHECK-DAG:         [[VAR_22_:%.+]] = "onnx.MatMul"([[VAR_3_]], [[VAR_9_]]) : (memref<?x4xf32>, memref<4x4xf32>) -> memref<?x4xf32>
// CHECK-DAG:         [[CST_0_6_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_0_7_:%.+]] = constant 0 : index
// CHECK-DAG:         [[CST_1_5_:%.+]] = constant 1 : index
// CHECK-DAG:         [[CST_4_:%.+]] = constant 4 : index
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_6_]] to [[VAR_2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_6_]] to [[CST_4_]]) {
// CHECK:               [[VAR_24_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_21_]]{{.}}[[VAR_24_1_]]#0, [[VAR_24_1_]]#1] : memref<?x4xf32>
// CHECK-DAG:           [[LOAD_VAR_22_MEM_:%.+]] = krnl.load [[VAR_22_]]{{.}}[[VAR_24_1_]]#0, [[VAR_24_1_]]#1] : memref<?x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_27_:%.+]] = addf [[LOAD_PARAM_0_MEM_1_]], [[LOAD_VAR_22_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_:%.+]] = krnl.load [[VAR_11_]]#0{{.}}[[VAR_24_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_11_MEM_1_:%.+]] = krnl.load [[VAR_11_]]#1{{.}}[[VAR_24_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_30_:%.+]] = addf [[VAR_27_]], [[LOAD_VAR_11_MEM_]] : f32
// CHECK-DAG:           [[VAR_31_:%.+]] = addf [[VAR_30_]], [[LOAD_VAR_11_MEM_1_]] : f32
// CHECK-DAG:           [[VAR_32_:%.+]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store [[VAR_31_]], [[VAR_32_]][] : memref<f32>
// CHECK:               [[VAR_33_:%.+]] = "onnx.Tanh"([[VAR_32_]]) : (memref<f32>) -> memref<f32>
// CHECK:               [[LOAD_VAR_33_MEM_:%.+]] = krnl.load [[VAR_33_]][] : memref<f32>
// CHECK:               krnl.store [[LOAD_VAR_33_MEM_]], [[VAR_3_]]{{.}}[[VAR_24_1_]]#0, [[VAR_24_1_]]#1] : memref<?x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc [[VAR_19_]] : memref<?x?xf32>
// CHECK:           }
// CHECK-DAG:       [[CST_16_:%.+]] = constant 16 : i64
// CHECK-DAG:       [[CST_0_8_:%.+]] = constant 0 : index
// CHECK:           [[VAR_14_:%.+]] = memref.dim [[VAR_3_]], [[CST_0_8_]] : memref<?x4xf32>
// CHECK:           [[VAR_15_:%.+]] = index_cast [[VAR_14_]] : index to i64
// CHECK:           [[VAR_16_:%.+]] = muli [[CST_16_]], [[VAR_15_]] : i64
// CHECK:           "krnl.memcpy"([[VAR_1_]], [[VAR_3_]], [[VAR_1_]]6) : (memref<1x?x4xf32>, memref<?x4xf32>, i64) -> ()
// CHECK:           memref.dealloc [[VAR_3_]] : memref<?x4xf32>
// CHECK:           return [[VAR_1_]] : memref<1x?x4xf32>
// CHECK:         }
}
