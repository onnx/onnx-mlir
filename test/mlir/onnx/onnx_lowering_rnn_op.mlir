// RUN: onnx-mlir-opt -O3 --shape-inference --convert-onnx-to-krnl='emit-intermediate-ir' --canonicalize %s -split-input-file | FileCheck %s

func.func private @test_rnn_forward_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x4x3xf32>, %arg2: tensor<1x4x4xf32>, %arg3: tensor<1x8xf32>, %arg4: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64} : (tensor<7x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x4xf32>, tensor<1x8xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>
// CHECK-LABEL:  func private @test_rnn_forward_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x4x3xf32>, [[PARAM_2_:%.+]]: memref<1x4x4xf32>, [[PARAM_3_:%.+]]: memref<1x8xf32>, [[PARAM_4_:%.+]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[VAR_c8_i64_:%.+]] = arith.constant 8 : i64
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_3_]] : memref<1x8xf32> to tensor<1x8xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_2_]] : memref<1x4x4xf32> to tensor<1x4x4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_1_]] : memref<1x4x3xf32> to tensor<1x4x3xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[VAR_15_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[VAR_c0_]], [[VAR_15_]]#0, [[VAR_15_]]#1] : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]]{{.}}[[VAR_15_]]#0, [[VAR_15_]]#1] : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.SqueezeV11"([[VAR_2_]]) {axes = [0]} : (tensor<1x4x3xf32>) -> tensor<4x3xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.SqueezeV11"([[VAR_1_]]) {axes = [0]} : (tensor<1x4x4xf32>) -> tensor<4x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Transpose"([[VAR_6_]]) {perm = [1, 0]} : (tensor<4x3xf32>) -> tensor<3x4xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Transpose"([[VAR_7_]]) {perm = [1, 0]} : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.SqueezeV11"([[VAR_0_]]) {axes = [0]} : (tensor<1x8xf32>) -> tensor<8xf32>
// CHECK:           [[VAR_11_:%.+]]:2 = "onnx.SplitV11"([[VAR_10_]]) {axis = 0 : si64} : (tensor<8xf32>) -> (tensor<4xf32>, tensor<4xf32>)
// CHECK-DAG:       [[VAR_12_:%.+]] = builtin.unrealized_conversion_cast [[VAR_11_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = builtin.unrealized_conversion_cast [[VAR_11_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7){
// CHECK-DAG:         [[VAR_15_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]){
// CHECK:               [[VAR_25_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_15_1_]], [[VAR_25_]]#0, [[VAR_25_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_2_]]{{.}}[[VAR_25_]]#0, [[VAR_25_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             [[VAR_18_:%.+]] = builtin.unrealized_conversion_cast [[RES_2_]] : memref<2x3xf32> to tensor<2x3xf32>
// CHECK:             [[VAR_19_:%.+]] = "onnx.MatMul"([[VAR_18_]], [[VAR_8_]]) : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[VAR_20_:%.+]] = builtin.unrealized_conversion_cast [[VAR_19_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[VAR_21_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_22_:%.+]] = "onnx.MatMul"([[VAR_21_]], [[VAR_9_]]) : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[VAR_23_:%.+]] = builtin.unrealized_conversion_cast [[VAR_22_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[VAR_25_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_20_]]{{.}}[[VAR_25_1_]]#0, [[VAR_25_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_23_MEM_:%.+]] = krnl.load [[VAR_23_]]{{.}}[[VAR_25_1_]]#0, [[VAR_25_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_28_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_1_]], [[LOAD_VAR_23_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_12_MEM_:%.+]] = krnl.load [[VAR_12_]]{{.}}[[VAR_25_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_13_MEM_:%.+]] = krnl.load [[VAR_13_]]{{.}}[[VAR_25_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_31_:%.+]] = arith.addf [[VAR_28_]], [[LOAD_VAR_12_MEM_]] : f32
// CHECK:               [[VAR_32_:%.+]] = arith.addf [[VAR_31_]], [[LOAD_VAR_13_MEM_]] : f32
// CHECK:               [[VAR_33_:%.+]] = math.tanh [[VAR_32_]] : f32
// CHECK:               krnl.store [[VAR_33_]], [[RES_1_]]{{.}}[[VAR_25_1_]]#0, [[VAR_25_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_1_]], [[VAR_c8_i64_]], [[VAR_c0_]], [[VAR_c0_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64, index, index) -> ()
// CHECK:           return [[RES_]] : memref<1x2x4xf32>
// CHECK:         }
}

// -----

func.func private @test_rnn_forward_mode_constant_weight_and_bias(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %w = onnx.Constant dense<[[[1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]]> : tensor<1x4x3xf32>
  %r = onnx.Constant dense<[[[2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.]]]> : tensor<1x4x4xf32>
  %b = onnx.Constant dense<[[1., 2., 3., 4., 5., 6., 7., 8.]]> : tensor<1x8xf32>

  %Y, %Y_h = "onnx.RNN"(%arg0, %w, %r, %b, %cst, %arg1) {hidden_size = 4 : si64} : (tensor<7x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x4xf32>, tensor<1x8xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:  func.func private @test_rnn_forward_mode_constant_weight_and_bias
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : i64
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[VAR_6_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_]], [[VAR_6_]]#0, [[VAR_6_]]#1] : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_1_]]{{.}}[[VAR_6_]]#0, [[VAR_6_]]#1] : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [3, 4], value = dense<1.000000e+00> : tensor<3x4xf32>} : () -> memref<3x4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [4], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [4], value = dense<[5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7){
// CHECK-DAG:         [[VAR_6_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_]] to [[CST_3_]]){
// CHECK:               [[VAR_17_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_1_]], [[VAR_17_]]#0, [[VAR_17_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_2_]]{{.}}[[VAR_17_]]#0, [[VAR_17_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_8_:%.+]] = builtin.unrealized_conversion_cast [[RES_2_]] : memref<2x3xf32> to tensor<2x3xf32>
// CHECK-DAG:         [[VAR_9_:%.+]] = builtin.unrealized_conversion_cast [[VAR_1_]] : memref<3x4xf32> to tensor<3x4xf32>
// CHECK:             [[VAR_10_:%.+]] = "onnx.MatMul"([[VAR_8_]], [[VAR_9_]]) : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[VAR_11_:%.+]] = builtin.unrealized_conversion_cast [[VAR_10_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[VAR_12_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK-DAG:         [[VAR_13_:%.+]] = builtin.unrealized_conversion_cast [[VAR_2_]] : memref<4x4xf32> to tensor<4x4xf32>
// CHECK:             [[VAR_14_:%.+]] = "onnx.MatMul"([[VAR_12_]], [[VAR_13_]]) : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[VAR_15_:%.+]] = builtin.unrealized_conversion_cast [[VAR_14_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_]] to [[CST_4_]]){
// CHECK:               [[VAR_17_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_11_]]{{.}}[[VAR_17_1_]]#0, [[VAR_17_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_15_MEM_:%.+]] = krnl.load [[VAR_15_]]{{.}}[[VAR_17_1_]]#0, [[VAR_17_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_20_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_1_]], [[LOAD_VAR_15_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_3_MEM_:%.+]] = krnl.load [[VAR_3_]]{{.}}[[VAR_17_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_4_MEM_:%.+]] = krnl.load [[VAR_4_]]{{.}}[[VAR_17_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_23_:%.+]] = arith.addf [[VAR_20_]], [[LOAD_VAR_3_MEM_]] : f32
// CHECK:               [[VAR_24_:%.+]] = arith.addf [[VAR_23_]], [[LOAD_VAR_4_MEM_]] : f32
// CHECK:               [[VAR_25_:%.+]] = math.tanh [[VAR_24_]] : f32
// CHECK:               krnl.store [[VAR_25_]], [[RES_1_]]{{.}}[[VAR_17_1_]]#0, [[VAR_17_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_]]_0, [[CST_8_]], [[VAR_c0_]], [[VAR_c0_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64, index, index) -> ()
// CHECK:           return [[RES_]] : memref<1x2x4xf32>
// CHECK:         }
}

// -----

func.func private @test_rnn_reverse_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x4x3xf32>, %arg2: tensor<1x4x4xf32>, %arg3: tensor<1x8xf32>, %arg4: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64, direction = "reverse"} : (tensor<7x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x4xf32>, tensor<1x8xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<(d0) -> (-d0 + 6)>
// CHECK-LABEL:  func private @test_rnn_reverse_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x4x3xf32>, [[PARAM_2_:%.+]]: memref<1x4x4xf32>, [[PARAM_3_:%.+]]: memref<1x8xf32>, [[PARAM_4_:%.+]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[VAR_c8_i64_:%.+]] = arith.constant 8 : i64
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_3_]] : memref<1x8xf32> to tensor<1x8xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_2_]] : memref<1x4x4xf32> to tensor<1x4x4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_1_]] : memref<1x4x3xf32> to tensor<1x4x3xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[VAR_15_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[VAR_c0_]], [[VAR_15_]]#0, [[VAR_15_]]#1] : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]]{{.}}[[VAR_15_]]#0, [[VAR_15_]]#1] : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.SqueezeV11"([[VAR_2_]]) {axes = [0]} : (tensor<1x4x3xf32>) -> tensor<4x3xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.SqueezeV11"([[VAR_1_]]) {axes = [0]} : (tensor<1x4x4xf32>) -> tensor<4x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Transpose"([[VAR_6_]]) {perm = [1, 0]} : (tensor<4x3xf32>) -> tensor<3x4xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Transpose"([[VAR_7_]]) {perm = [1, 0]} : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.SqueezeV11"([[VAR_0_]]) {axes = [0]} : (tensor<1x8xf32>) -> tensor<8xf32>
// CHECK:           [[VAR_11_:%.+]]:2 = "onnx.SplitV11"([[VAR_10_]]) {axis = 0 : si64} : (tensor<8xf32>) -> (tensor<4xf32>, tensor<4xf32>)
// CHECK-DAG:       [[VAR_12_:%.+]] = builtin.unrealized_conversion_cast [[VAR_11_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = builtin.unrealized_conversion_cast [[VAR_11_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7){
// CHECK:             [[VAR_15_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_1_:%.+]] = affine.apply [[MAP_0_]]([[VAR_15_1_]])
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]){
// CHECK:               [[VAR_26_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[LOAD_PARAM_4_MEM_1_]], [[VAR_26_]]#0, [[VAR_26_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_2_]]{{.}}[[VAR_26_]]#0, [[VAR_26_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             [[VAR_19_:%.+]] = builtin.unrealized_conversion_cast [[RES_2_]] : memref<2x3xf32> to tensor<2x3xf32>
// CHECK:             [[VAR_20_:%.+]] = "onnx.MatMul"([[VAR_19_]], [[VAR_8_]]) : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[VAR_21_:%.+]] = builtin.unrealized_conversion_cast [[VAR_20_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[VAR_22_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_23_:%.+]] = "onnx.MatMul"([[VAR_22_]], [[VAR_9_]]) : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[VAR_24_:%.+]] = builtin.unrealized_conversion_cast [[VAR_23_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[VAR_26_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_21_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_24_MEM_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_29_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_1_]], [[LOAD_VAR_24_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_12_MEM_:%.+]] = krnl.load [[VAR_12_]]{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_13_MEM_:%.+]] = krnl.load [[VAR_13_]]{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_32_:%.+]] = arith.addf [[VAR_29_]], [[LOAD_VAR_12_MEM_]] : f32
// CHECK:               [[VAR_33_:%.+]] = arith.addf [[VAR_32_]], [[LOAD_VAR_13_MEM_]] : f32
// CHECK:               [[VAR_34_:%.+]] = math.tanh [[VAR_33_]] : f32
// CHECK:               krnl.store [[VAR_34_]], [[RES_1_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_1_]], [[VAR_c8_i64_]], [[VAR_c0_]], [[VAR_c0_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64, index, index) -> ()
// CHECK:           return [[RES_]] : memref<1x2x4xf32>
// CHECK:         }
}

// -----

func.func private @test_rnn_bidirectional_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<2x4x3xf32>, %arg2: tensor<2x4x4xf32>, %arg3: tensor<2x8xf32>, %arg4: tensor<2x2x4xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64, direction = "bidirectional"} : (tensor<7x2x3xf32>, tensor<2x4x3xf32>, tensor<2x4x4xf32>, tensor<2x8xf32>, none, tensor<2x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>
// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<(d0) -> (-d0 + 6)>
// CHECK-LABEL:  func private @test_rnn_bidirectional_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<2x4x3xf32>, [[PARAM_2_:%.+]]: memref<2x4x4xf32>, [[PARAM_3_:%.+]]: memref<2x8xf32>, [[PARAM_4_:%.+]]: memref<2x2x4xf32>) -> memref<2x2x4xf32> {
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_3_]] : memref<2x8xf32> to tensor<2x8xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_2_]] : memref<2x4x4xf32> to tensor<2x4x4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_1_]] : memref<2x4x3xf32> to tensor<2x4x3xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[VAR_29_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[VAR_c0_]], [[VAR_29_]]#0, [[VAR_29_]]#1] : memref<2x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]]{{.}}[[VAR_29_]]#0, [[VAR_29_]]#1] : memref<2x4xf32>
// CHECK:             [[LOAD_PARAM_4_MEM_1_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[VAR_c1_]], [[VAR_29_]]#0, [[VAR_29_]]#1] : memref<2x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_1_]], [[RES_2_]]{{.}}[[VAR_29_]]#0, [[VAR_29_]]#1] : memref<2x4xf32>
// CHECK:           }
// CHECK:           [[VAR_7_:%.+]]:2 = "onnx.SplitV11"([[VAR_2_]]) {axis = 0 : si64} : (tensor<2x4x3xf32>) -> (tensor<1x4x3xf32>, tensor<1x4x3xf32>)
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.SqueezeV11"([[VAR_7_]]#0) {axes = [0]} : (tensor<1x4x3xf32>) -> tensor<4x3xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.SqueezeV11"([[VAR_7_]]#1) {axes = [0]} : (tensor<1x4x3xf32>) -> tensor<4x3xf32>
// CHECK-DAG:       [[VAR_10_:%.+]]:2 = "onnx.SplitV11"([[VAR_1_]]) {axis = 0 : si64} : (tensor<2x4x4xf32>) -> (tensor<1x4x4xf32>, tensor<1x4x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.SqueezeV11"([[VAR_10_]]#0) {axes = [0]} : (tensor<1x4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.SqueezeV11"([[VAR_10_]]#1) {axes = [0]} : (tensor<1x4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Transpose"([[VAR_8_]]) {perm = [1, 0]} : (tensor<4x3xf32>) -> tensor<3x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.Transpose"([[VAR_11_]]) {perm = [1, 0]} : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.Transpose"([[VAR_9_]]) {perm = [1, 0]} : (tensor<4x3xf32>) -> tensor<3x4xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Transpose"([[VAR_12_]]) {perm = [1, 0]} : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_17_:%.+]]:2 = "onnx.SplitV11"([[VAR_0_]]) {axis = 0 : si64} : (tensor<2x8xf32>) -> (tensor<1x8xf32>, tensor<1x8xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.SqueezeV11"([[VAR_17_]]#0) {axes = [0]} : (tensor<1x8xf32>) -> tensor<8xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.SqueezeV11"([[VAR_17_]]#1) {axes = [0]} : (tensor<1x8xf32>) -> tensor<8xf32>
// CHECK:           [[VAR_20_:%.+]]:2 = "onnx.SplitV11"([[VAR_18_]]) {axis = 0 : si64} : (tensor<8xf32>) -> (tensor<4xf32>, tensor<4xf32>)
// CHECK-DAG:       [[VAR_21_:%.+]] = builtin.unrealized_conversion_cast [[VAR_20_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = builtin.unrealized_conversion_cast [[VAR_20_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_23_:%.+]]:2 = "onnx.SplitV11"([[VAR_19_]]) {axis = 0 : si64} : (tensor<8xf32>) -> (tensor<4xf32>, tensor<4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_24_:%.+]] = builtin.unrealized_conversion_cast [[VAR_23_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_25_:%.+]] = builtin.unrealized_conversion_cast [[VAR_23_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7){
// CHECK-DAG:         [[VAR_29_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]){
// CHECK:               [[VAR_39_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_29_1_]], [[VAR_39_]]#0, [[VAR_39_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_3_]]{{.}}[[VAR_39_]]#0, [[VAR_39_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             [[VAR_32_:%.+]] = builtin.unrealized_conversion_cast [[RES_3_]] : memref<2x3xf32> to tensor<2x3xf32>
// CHECK:             [[VAR_33_:%.+]] = "onnx.MatMul"([[VAR_32_]], [[VAR_13_]]) : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[VAR_34_:%.+]] = builtin.unrealized_conversion_cast [[VAR_33_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[VAR_35_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_36_:%.+]] = "onnx.MatMul"([[VAR_35_]], [[VAR_14_]]) : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[VAR_37_:%.+]] = builtin.unrealized_conversion_cast [[VAR_36_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[VAR_39_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_34_]]{{.}}[[VAR_39_1_]]#0, [[VAR_39_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_37_MEM_:%.+]] = krnl.load [[VAR_37_]]{{.}}[[VAR_39_1_]]#0, [[VAR_39_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_42_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_1_]], [[LOAD_VAR_37_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_21_MEM_:%.+]] = krnl.load [[VAR_21_]]{{.}}[[VAR_39_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_22_MEM_:%.+]] = krnl.load [[VAR_22_]]{{.}}[[VAR_39_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_45_:%.+]] = arith.addf [[VAR_42_]], [[LOAD_VAR_21_MEM_]] : f32
// CHECK:               [[VAR_46_:%.+]] = arith.addf [[VAR_45_]], [[LOAD_VAR_22_MEM_]] : f32
// CHECK:               [[VAR_47_:%.+]] = math.tanh [[VAR_46_]] : f32
// CHECK:               krnl.store [[VAR_47_]], [[RES_1_]]{{.}}[[VAR_39_1_]]#0, [[VAR_39_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_4_]]) with ([[LOOP_4_]] -> [[I_7_:%.+]] = 0 to 7){
// CHECK:             [[VAR_29_2_:%.+]] = krnl.get_induction_var_value([[LOOP_4_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[RES_3_:%.+]] = affine.apply [[MAP_0_]]([[VAR_29_2_]])
// CHECK-DAG:         [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_5_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_5_]]#0, [[LOOP_5_]]#1) with ([[LOOP_5_]]#0 -> [[I_8_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_5_]]#1 -> [[I_9_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]){
// CHECK:               [[LOAD_PARAM_0_MEM_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_5_]]#0, [[LOOP_5_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[RES_3_]], [[LOAD_PARAM_0_MEM_1_]]#0, [[LOAD_PARAM_0_MEM_1_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_2_]], [[RES_4_]]{{.}}[[LOAD_PARAM_0_MEM_1_]]#0, [[LOAD_PARAM_0_MEM_1_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             [[VAR_33_1_:%.+]] = builtin.unrealized_conversion_cast [[RES_4_]] : memref<2x3xf32> to tensor<2x3xf32>
// CHECK:             [[VAR_34_1_:%.+]] = "onnx.MatMul"([[VAR_33_1_]], [[VAR_15_]]) : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[VAR_35_1_:%.+]] = builtin.unrealized_conversion_cast [[VAR_34_1_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[VAR_36_1_:%.+]] = builtin.unrealized_conversion_cast [[RES_2_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_37_1_:%.+]] = "onnx.MatMul"([[VAR_36_1_]], [[VAR_16_]]) : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]] = builtin.unrealized_conversion_cast [[VAR_37_1_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[LOOP_6_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_6_]]#0, [[LOOP_6_]]#1) with ([[LOOP_6_]]#0 -> [[I_10_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_6_]]#1 -> [[I_11_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[LOAD_PARAM_0_MEM_1_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_6_]]#0, [[LOOP_6_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[VAR_35_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[VAR_42_1_:%.+]] = krnl.load [[LOOP_3_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_21_MEM_1_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_2_]], [[VAR_42_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_22_MEM_1_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_45_1_:%.+]] = krnl.load [[VAR_25_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_46_1_:%.+]] = arith.addf [[LOAD_VAR_21_MEM_1_]], [[LOAD_VAR_22_MEM_1_]] : f32
// CHECK:               [[VAR_47_1_:%.+]] = arith.addf [[VAR_46_1_]], [[VAR_45_1_]] : f32
// CHECK:               [[VAR_48_:%.+]] = math.tanh [[VAR_47_1_]] : f32
// CHECK:               krnl.store [[VAR_48_]], [[RES_2_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOOP_7_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_7_]]#0, [[LOOP_7_]]#1) with ([[LOOP_7_]]#0 -> [[I_12_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_7_]]#1 -> [[I_13_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:             [[VAR_29_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_7_]]#0, [[LOOP_7_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[RES_3_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_29_3_]]#0, [[VAR_29_3_]]#1] : memref<2x4xf32>
// CHECK:             krnl.store [[RES_3_1_]], [[RES_]]{{.}}[[VAR_c0_]], [[VAR_29_3_]]#0, [[VAR_29_3_]]#1] : memref<2x2x4xf32>
// CHECK:             [[RES_4_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_29_3_]]#0, [[VAR_29_3_]]#1] : memref<2x4xf32>
// CHECK:             krnl.store [[RES_4_]], [[RES_]]{{.}}[[VAR_c1_]], [[VAR_29_3_]]#0, [[VAR_29_3_]]#1] : memref<2x2x4xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x2x4xf32>
// CHECK:         }
}

// -----

func.func private @test_rnn_unknown_dims(%arg0: tensor<?x?x?xf32>, %arg1: tensor<1x4x?xf32>, %arg2: tensor<1x4x4xf32>, %arg3: tensor<1x8xf32>, %arg4: tensor<1x?x4xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64} : (tensor<?x?x?xf32>, tensor<1x4x?xf32>, tensor<1x4x4xf32>, tensor<1x8xf32>, none, tensor<1x?x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func private @test_rnn_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x?xf32>, [[PARAM_1_:%.+]]: memref<1x4x?xf32>, [[PARAM_2_:%.+]]: memref<1x4x4xf32>, [[PARAM_3_:%.+]]: memref<1x8xf32>, [[PARAM_4_:%.+]]: memref<1x?x4xf32>) -> memref<1x?x4xf32> {
// CHECK-DAG:       [[VAR_c4_i64_:%.+]] = arith.constant 4 : i64
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_3_]] : memref<1x8xf32> to tensor<1x8xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_2_]] : memref<1x4x4xf32> to tensor<1x4x4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_1_]] : memref<1x4x?xf32> to tensor<1x4x?xf32>
// CHECK:           [[VAR_3_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_3_]]) {{.*}}: memref<1x?x4xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<?x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[VAR_5_]]) {{.*}}: memref<?x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_5_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[VAR_20_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[VAR_c0_]], [[VAR_20_]]#0, [[VAR_20_]]#1] : memref<1x?x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]]{{.}}[[VAR_20_]]#0, [[VAR_20_]]#1] : memref<?x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.SqueezeV11"([[VAR_2_]]) {axes = [0]} : (tensor<1x4x?xf32>) -> tensor<4x?xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.SqueezeV11"([[VAR_1_]]) {axes = [0]} : (tensor<1x4x4xf32>) -> tensor<4x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Transpose"([[VAR_8_]]) {perm = [1, 0]} : (tensor<4x?xf32>) -> tensor<?x4xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Transpose"([[VAR_9_]]) {perm = [1, 0]} : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.SqueezeV11"([[VAR_0_]]) {axes = [0]} : (tensor<1x8xf32>) -> tensor<8xf32>
// CHECK:           [[VAR_13_:%.+]]:2 = "onnx.SplitV11"([[VAR_12_]]) {axis = 0 : si64} : (tensor<8xf32>) -> (tensor<4xf32>, tensor<4xf32>)
// CHECK-DAG:       [[VAR_14_:%.+]] = builtin.unrealized_conversion_cast [[VAR_13_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = builtin.unrealized_conversion_cast [[VAR_13_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK-DAG:       [[VAR_17_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?x?x?xf32>
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to [[MAP_0_]]([[VAR_17_]])){
// CHECK-DAG:         [[VAR_20_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_1_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<?x?x?xf32>
// CHECK-DAG:         [[VAR_22_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c2_]] : memref<?x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloc([[LOAD_PARAM_4_MEM_1_]], [[VAR_22_]]) {{.*}}: memref<?x?xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[VAR_c0_]] to [[LOAD_PARAM_4_MEM_1_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[VAR_c0_]] to [[VAR_22_]]){
// CHECK:               [[VAR_32_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_20_1_]], [[VAR_32_]]#0, [[VAR_32_]]#1] : memref<?x?x?xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_2_]]{{.}}[[VAR_32_]]#0, [[VAR_32_]]#1] : memref<?x?xf32>
// CHECK:             }
// CHECK:             [[VAR_25_:%.+]] = builtin.unrealized_conversion_cast [[RES_2_]] : memref<?x?xf32> to tensor<?x?xf32>
// CHECK:             [[VAR_26_:%.+]] = "onnx.MatMul"([[VAR_25_]], [[VAR_10_]]) : (tensor<?x?xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
// CHECK-DAG:         [[VAR_27_:%.+]] = builtin.unrealized_conversion_cast [[VAR_26_]] : tensor<?x4xf32> to memref<?x4xf32>
// CHECK-DAG:         [[VAR_28_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<?x4xf32> to tensor<?x4xf32>
// CHECK:             [[VAR_29_:%.+]] = "onnx.MatMul"([[VAR_28_]], [[VAR_11_]]) : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
// CHECK-DAG:         [[VAR_30_:%.+]] = builtin.unrealized_conversion_cast [[VAR_29_]] : tensor<?x4xf32> to memref<?x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[VAR_c0_]] to [[VAR_5_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[VAR_32_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_27_]]{{.}}[[VAR_32_1_]]#0, [[VAR_32_1_]]#1] : memref<?x4xf32>
// CHECK-DAG:           [[LOAD_VAR_30_MEM_:%.+]] = krnl.load [[VAR_30_]]{{.}}[[VAR_32_1_]]#0, [[VAR_32_1_]]#1] : memref<?x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_35_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_1_]], [[LOAD_VAR_30_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_14_MEM_:%.+]] = krnl.load [[VAR_14_]]{{.}}[[VAR_32_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_15_MEM_:%.+]] = krnl.load [[VAR_15_]]{{.}}[[VAR_32_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_38_:%.+]] = arith.addf [[VAR_35_]], [[LOAD_VAR_14_MEM_]] : f32
// CHECK:               [[VAR_39_:%.+]] = arith.addf [[VAR_38_]], [[LOAD_VAR_15_MEM_]] : f32
// CHECK:               [[VAR_40_:%.+]] = math.tanh [[VAR_39_]] : f32
// CHECK:               krnl.store [[VAR_40_]], [[RES_1_]]{{.}}[[VAR_32_1_]]#0, [[VAR_32_1_]]#1] : memref<?x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           [[VAR_18_:%.+]] = arith.index_cast [[VAR_5_]] : index to i64
// CHECK:           [[VAR_19_:%.+]] = arith.muli [[VAR_18_]], [[VAR_c4_i64_]] : i64
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_1_]], [[VAR_19_]], [[VAR_c0_]], [[VAR_c0_]]) : (memref<1x?x4xf32>, memref<?x4xf32>, i64, index, index) -> ()
// CHECK:           return [[RES_]] : memref<1x?x4xf32>
// CHECK:         }
}
