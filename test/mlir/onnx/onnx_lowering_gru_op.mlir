// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl='emit-intermediate-ir' --canonicalize %s -split-input-file | FileCheck %s

func.func private @test_gru_forward_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x12x3xf32>, %arg2: tensor<1x12x4xf32>, %arg3: tensor<1x24xf32>, %arg4: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64} : (tensor<7x2x3xf32>, tensor<1x12x3xf32>, tensor<1x12x4xf32>, tensor<1x24xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>
// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 + 4)>
// CHECK-DAG: [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 + 8)>
// CHECK-LABEL:  func private @test_gru_forward_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x12x3xf32>, [[PARAM_2_:%.+]]: memref<1x12x4xf32>, [[PARAM_3_:%.+]]: memref<1x24xf32>, [[PARAM_4_:%.+]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[VAR_c8_i64_:%.+]] = arith.constant 8 : i64
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_3_]] : memref<1x24xf32> to tensor<1x24xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_2_]] : memref<1x12x4xf32> to tensor<1x12x4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_1_]] : memref<1x12x3xf32> to tensor<1x12x3xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[VAR_22_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[VAR_c0_]], [[VAR_22_]]#0, [[VAR_22_]]#1] : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]]{{.}}[[VAR_22_]]#0, [[VAR_22_]]#1] : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.SqueezeV11"([[VAR_2_]]) {axes = [0]} : (tensor<1x12x3xf32>) -> tensor<12x3xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.SqueezeV11"([[VAR_1_]]) {axes = [0]} : (tensor<1x12x4xf32>) -> tensor<12x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Transpose"([[VAR_6_]]) {perm = [1, 0]} : (tensor<12x3xf32>) -> tensor<3x12xf32>
// CHECK-DAG:       [[VAR_9_:%.+]]:3 = "onnx.SplitV11"([[VAR_7_]]) {axis = 0 : si64} : (tensor<12x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Transpose"([[VAR_9_]]#0) {perm = [1, 0]} : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Transpose"([[VAR_9_]]#1) {perm = [1, 0]} : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Transpose"([[VAR_9_]]#2) {perm = [1, 0]} : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.SqueezeV11"([[VAR_0_]]) {axes = [0]} : (tensor<1x24xf32>) -> tensor<24xf32>
// CHECK:           [[VAR_14_:%.+]]:6 = "onnx.SplitV11"([[VAR_13_]]) {axis = 0 : si64} : (tensor<24xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
// CHECK-DAG:       [[VAR_15_:%.+]] = builtin.unrealized_conversion_cast [[VAR_14_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = builtin.unrealized_conversion_cast [[VAR_14_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = builtin.unrealized_conversion_cast [[VAR_14_]]#2 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = builtin.unrealized_conversion_cast [[VAR_14_]]#3 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = builtin.unrealized_conversion_cast [[VAR_14_]]#4 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = builtin.unrealized_conversion_cast [[VAR_14_]]#5 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7){
// CHECK-DAG:         [[VAR_22_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]){
// CHECK:               [[VAR_41_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_22_1_]], [[VAR_41_]]#0, [[VAR_41_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_2_]]{{.}}[[VAR_41_]]#0, [[VAR_41_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             [[VAR_25_:%.+]] = builtin.unrealized_conversion_cast [[RES_2_]] : memref<2x3xf32> to tensor<2x3xf32>
// CHECK:             [[VAR_26_:%.+]] = "onnx.MatMul"([[VAR_25_]], [[VAR_8_]]) : (tensor<2x3xf32>, tensor<3x12xf32>) -> tensor<2x12xf32>
// CHECK-DAG:         [[VAR_27_:%.+]] = builtin.unrealized_conversion_cast [[VAR_26_]] : tensor<2x12xf32> to memref<2x12xf32>
// CHECK-DAG:         [[VAR_28_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_29_:%.+]] = "onnx.MatMul"([[VAR_28_]], [[VAR_10_]]) : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[VAR_30_:%.+]] = builtin.unrealized_conversion_cast [[VAR_29_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[VAR_31_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_32_:%.+]] = "onnx.MatMul"([[VAR_31_]], [[VAR_11_]]) : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[VAR_33_:%.+]] = builtin.unrealized_conversion_cast [[VAR_32_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[VAR_41_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_41_]]#0, [[VAR_41_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[VAR_43_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_41_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_27_MEM_:%.+]] = krnl.load [[VAR_27_]]{{.}}[[VAR_41_1_]]#0, [[VAR_43_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_33_MEM_:%.+]] = krnl.load [[VAR_33_]]{{.}}[[VAR_41_1_]]#0, [[VAR_41_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_46_:%.+]] = arith.addf [[LOAD_VAR_27_MEM_]], [[LOAD_VAR_33_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_16_MEM_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_41_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_19_MEM_:%.+]] = krnl.load [[VAR_19_]]{{.}}[[VAR_41_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_49_:%.+]] = arith.addf [[VAR_46_]], [[LOAD_VAR_16_MEM_]] : f32
// CHECK:               [[VAR_50_:%.+]] = arith.addf [[VAR_49_]], [[LOAD_VAR_19_MEM_]] : f32
// CHECK:               [[VAR_51_:%.+]] = arith.subf [[VAR_cst_]], [[VAR_50_]] : f32
// CHECK:               [[VAR_52_:%.+]] = math.exp [[VAR_51_]] : f32
// CHECK:               [[VAR_53_:%.+]] = arith.addf [[VAR_52_]], [[VAR_cst_0_]] : f32
// CHECK:               [[VAR_54_:%.+]] = arith.divf [[VAR_cst_0_]], [[VAR_53_]] : f32
// CHECK:               krnl.store [[VAR_54_]], [[RES_3_]]{{.}}[[VAR_41_1_]]#0, [[VAR_41_1_]]#1] : memref<2x4xf32>
// CHECK:               [[VAR_55_:%.+]] = arith.mulf [[VAR_54_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               krnl.store [[VAR_55_]], [[RES_4_]]{{.}}[[VAR_41_1_]]#0, [[VAR_41_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             [[VAR_37_:%.+]] = builtin.unrealized_conversion_cast [[RES_4_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_38_:%.+]] = "onnx.MatMul"([[VAR_37_]], [[VAR_12_]]) : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[VAR_39_:%.+]] = builtin.unrealized_conversion_cast [[VAR_38_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_7_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_4_]]#1 -> [[I_8_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[VAR_41_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_41_]]#0, [[VAR_41_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_27_MEM_1_:%.+]] = krnl.load [[VAR_27_]]{{.}}[[VAR_41_2_]]#0, [[VAR_41_2_]]#1] : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_27_MEM_2_:%.+]] = krnl.load [[VAR_30_]]{{.}}[[VAR_41_2_]]#0, [[VAR_41_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_33_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_27_MEM_1_]], [[LOAD_VAR_27_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_46_1_:%.+]] = krnl.load [[VAR_15_]]{{.}}[[VAR_41_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_16_MEM_1_:%.+]] = krnl.load [[VAR_18_]]{{.}}[[VAR_41_2_]]#1] : memref<4xf32>
// CHECK:               [[LOAD_VAR_19_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_33_MEM_1_]], [[VAR_46_1_]] : f32
// CHECK:               [[VAR_49_1_:%.+]] = arith.addf [[LOAD_VAR_19_MEM_1_]], [[LOAD_VAR_16_MEM_1_]] : f32
// CHECK:               [[VAR_50_1_:%.+]] = arith.subf [[VAR_cst_]], [[VAR_49_1_]] : f32
// CHECK:               [[VAR_51_1_:%.+]] = math.exp [[VAR_50_1_]] : f32
// CHECK:               [[VAR_52_1_:%.+]] = arith.addf [[VAR_51_1_]], [[VAR_cst_0_]] : f32
// CHECK-DAG:           [[VAR_53_1_:%.+]] = arith.divf [[VAR_cst_0_]], [[VAR_52_1_]] : f32
// CHECK-DAG:           [[VAR_54_1_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_41_2_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_27_MEM_3_:%.+]] = krnl.load [[VAR_27_]]{{.}}[[VAR_41_2_]]#0, [[VAR_54_1_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_39_MEM_:%.+]] = krnl.load [[VAR_39_]]{{.}}[[VAR_41_2_]]#0, [[VAR_41_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_57_:%.+]] = arith.addf [[LOAD_VAR_27_MEM_3_]], [[LOAD_VAR_39_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_17_MEM_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_41_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_20_MEM_:%.+]] = krnl.load [[VAR_20_]]{{.}}[[VAR_41_2_]]#1] : memref<4xf32>
// CHECK:               [[VAR_60_:%.+]] = arith.addf [[VAR_57_]], [[LOAD_VAR_17_MEM_]] : f32
// CHECK:               [[VAR_61_:%.+]] = arith.addf [[VAR_60_]], [[LOAD_VAR_20_MEM_]] : f32
// CHECK-DAG:           [[VAR_62_:%.+]] = math.tanh [[VAR_61_]] : f32
// CHECK-DAG:           [[VAR_63_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_53_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_64_:%.+]] = arith.mulf [[VAR_63_]], [[VAR_62_]] : f32
// CHECK-DAG:           [[VAR_65_:%.+]] = arith.mulf [[VAR_53_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_66_:%.+]] = arith.addf [[VAR_64_]], [[VAR_65_]] : f32
// CHECK:               krnl.store [[VAR_66_]], [[RES_1_]]{{.}}[[VAR_41_]]#0, [[VAR_41_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_1_]], [[VAR_c8_i64_]], [[VAR_c0_]], [[VAR_c0_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64, index, index) -> ()
// CHECK:           return [[RES_]] : memref<1x2x4xf32>
// CHECK:         }
}

// -----

func.func private @test_gru_forward_mode_linear_before_reset(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x12x3xf32>, %arg2: tensor<1x12x4xf32>, %arg3: tensor<1x24xf32>, %arg4: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64, linear_before_reset = 1 : si64} : (tensor<7x2x3xf32>, tensor<1x12x3xf32>, tensor<1x12x4xf32>, tensor<1x24xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>
// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 + 4)>
// CHECK-DAG: [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 + 8)>
// CHECK-LABEL:  func private @test_gru_forward_mode_linear_before_reset
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x12x3xf32>, [[PARAM_2_:%.+]]: memref<1x12x4xf32>, [[PARAM_3_:%.+]]: memref<1x24xf32>, [[PARAM_4_:%.+]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[VAR_c8_i64_:%.+]] = arith.constant 8 : i64
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_3_]] : memref<1x24xf32> to tensor<1x24xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_2_]] : memref<1x12x4xf32> to tensor<1x12x4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_1_]] : memref<1x12x3xf32> to tensor<1x12x3xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[VAR_19_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[VAR_c0_]], [[VAR_19_]]#0, [[VAR_19_]]#1] : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]]{{.}}[[VAR_19_]]#0, [[VAR_19_]]#1] : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.SqueezeV11"([[VAR_2_]]) {axes = [0]} : (tensor<1x12x3xf32>) -> tensor<12x3xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.SqueezeV11"([[VAR_1_]]) {axes = [0]} : (tensor<1x12x4xf32>) -> tensor<12x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Transpose"([[VAR_6_]]) {perm = [1, 0]} : (tensor<12x3xf32>) -> tensor<3x12xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Transpose"([[VAR_7_]]) {perm = [1, 0]} : (tensor<12x4xf32>) -> tensor<4x12xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.SqueezeV11"([[VAR_0_]]) {axes = [0]} : (tensor<1x24xf32>) -> tensor<24xf32>
// CHECK:           [[VAR_11_:%.+]]:6 = "onnx.SplitV11"([[VAR_10_]]) {axis = 0 : si64} : (tensor<24xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
// CHECK-DAG:       [[VAR_12_:%.+]] = builtin.unrealized_conversion_cast [[VAR_11_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = builtin.unrealized_conversion_cast [[VAR_11_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = builtin.unrealized_conversion_cast [[VAR_11_]]#2 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = builtin.unrealized_conversion_cast [[VAR_11_]]#3 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = builtin.unrealized_conversion_cast [[VAR_11_]]#4 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = builtin.unrealized_conversion_cast [[VAR_11_]]#5 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7){
// CHECK-DAG:         [[VAR_19_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]){
// CHECK:               [[VAR_29_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_19_1_]], [[VAR_29_]]#0, [[VAR_29_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_2_]]{{.}}[[VAR_29_]]#0, [[VAR_29_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             [[VAR_22_:%.+]] = builtin.unrealized_conversion_cast [[RES_2_]] : memref<2x3xf32> to tensor<2x3xf32>
// CHECK:             [[VAR_23_:%.+]] = "onnx.MatMul"([[VAR_22_]], [[VAR_8_]]) : (tensor<2x3xf32>, tensor<3x12xf32>) -> tensor<2x12xf32>
// CHECK-DAG:         [[VAR_24_:%.+]] = builtin.unrealized_conversion_cast [[VAR_23_]] : tensor<2x12xf32> to memref<2x12xf32>
// CHECK-DAG:         [[VAR_25_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_26_:%.+]] = "onnx.MatMul"([[VAR_25_]], [[VAR_9_]]) : (tensor<2x4xf32>, tensor<4x12xf32>) -> tensor<2x12xf32>
// CHECK-DAG:         [[VAR_27_:%.+]] = builtin.unrealized_conversion_cast [[VAR_26_]] : tensor<2x12xf32> to memref<2x12xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[VAR_29_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_29_1_]]#0, [[VAR_29_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_24_MEM_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_29_1_]]#0, [[VAR_29_1_]]#1] : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_27_MEM_:%.+]] = krnl.load [[VAR_27_]]{{.}}[[VAR_29_1_]]#0, [[VAR_29_1_]]#1] : memref<2x12xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_33_:%.+]] = arith.addf [[LOAD_VAR_24_MEM_]], [[LOAD_VAR_27_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_12_MEM_:%.+]] = krnl.load [[VAR_12_]]{{.}}[[VAR_29_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_15_MEM_:%.+]] = krnl.load [[VAR_15_]]{{.}}[[VAR_29_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_36_:%.+]] = arith.addf [[VAR_33_]], [[LOAD_VAR_12_MEM_]] : f32
// CHECK:               [[VAR_37_:%.+]] = arith.addf [[VAR_36_]], [[LOAD_VAR_15_MEM_]] : f32
// CHECK:               [[VAR_38_:%.+]] = arith.subf [[VAR_cst_]], [[VAR_37_]] : f32
// CHECK:               [[VAR_39_:%.+]] = math.exp [[VAR_38_]] : f32
// CHECK:               [[VAR_40_:%.+]] = arith.addf [[VAR_39_]], [[VAR_cst_0_]] : f32
// CHECK-DAG:           [[VAR_41_:%.+]] = arith.divf [[VAR_cst_0_]], [[VAR_40_]] : f32
// CHECK-DAG:           [[VAR_42_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_29_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_24_MEM_1_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_29_1_]]#0, [[VAR_42_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[VAR_44_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_29_1_]]#1]
// CHECK:               [[LOAD_VAR_27_MEM_1_:%.+]] = krnl.load [[VAR_27_]]{{.}}[[VAR_29_1_]]#0, [[VAR_44_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[VAR_46_:%.+]] = arith.addf [[LOAD_VAR_24_MEM_1_]], [[LOAD_VAR_27_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_13_MEM_:%.+]] = krnl.load [[VAR_13_]]{{.}}[[VAR_29_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_16_MEM_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_29_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_49_:%.+]] = arith.addf [[VAR_46_]], [[LOAD_VAR_13_MEM_]] : f32
// CHECK:               [[VAR_50_:%.+]] = arith.addf [[VAR_49_]], [[LOAD_VAR_16_MEM_]] : f32
// CHECK:               [[VAR_51_:%.+]] = arith.subf [[VAR_cst_]], [[VAR_50_]] : f32
// CHECK:               [[VAR_52_:%.+]] = math.exp [[VAR_51_]] : f32
// CHECK:               [[VAR_53_:%.+]] = arith.addf [[VAR_52_]], [[VAR_cst_0_]] : f32
// CHECK-DAG:           [[VAR_54_:%.+]] = arith.divf [[VAR_cst_0_]], [[VAR_53_]] : f32
// CHECK-DAG:           [[VAR_55_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_29_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_24_MEM_2_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_29_1_]]#0, [[VAR_55_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[VAR_57_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_29_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_27_MEM_2_:%.+]] = krnl.load [[VAR_27_]]{{.}}[[VAR_29_1_]]#0, [[VAR_57_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_17_MEM_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_29_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_60_:%.+]] = arith.addf [[LOAD_VAR_27_MEM_2_]], [[LOAD_VAR_17_MEM_]] : f32
// CHECK:               [[VAR_61_:%.+]] = arith.mulf [[VAR_54_]], [[VAR_60_]] : f32
// CHECK-DAG:           [[VAR_62_:%.+]] = arith.addf [[LOAD_VAR_24_MEM_2_]], [[VAR_61_]] : f32
// CHECK-DAG:           [[LOAD_VAR_14_MEM_:%.+]] = krnl.load [[VAR_14_]]{{.}}[[VAR_29_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_64_:%.+]] = arith.addf [[VAR_62_]], [[LOAD_VAR_14_MEM_]] : f32
// CHECK-DAG:           [[VAR_65_:%.+]] = math.tanh [[VAR_64_]] : f32
// CHECK-DAG:           [[VAR_66_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_41_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_67_:%.+]] = arith.mulf [[VAR_66_]], [[VAR_65_]] : f32
// CHECK-DAG:           [[VAR_68_:%.+]] = arith.mulf [[VAR_41_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_69_:%.+]] = arith.addf [[VAR_67_]], [[VAR_68_]] : f32
// CHECK:               krnl.store [[VAR_69_]], [[RES_1_]]{{.}}[[VAR_29_1_]]#0, [[VAR_29_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_1_]], [[VAR_c8_i64_]], [[VAR_c0_]], [[VAR_c0_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64, index, index) -> ()
// CHECK:           return [[RES_]] : memref<1x2x4xf32>
// CHECK:         }
}

// -----

func.func private @test_gru_forward_mode_constant_weight_and_bias(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %w = onnx.Constant dense<[[[1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]]> : tensor<1x12x3xf32>
  %r = onnx.Constant dense<[[[2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.]]]> : tensor<1x12x4xf32>
  %b = onnx.Constant dense<[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.]]> : tensor<1x24xf32>

  %Y, %Y_h = "onnx.GRU"(%arg0, %w, %r, %b, %cst, %arg1) {hidden_size = 4 : si64} : (tensor<7x2x3xf32>, tensor<1x12x3xf32>, tensor<1x12x4xf32>, tensor<1x24xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<()[s0] -> (s0 + 4)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<()[s0] -> (s0 + 8)>
// CHECK-LABEL:  func.func private @test_gru_forward_mode_constant_weight_and_bias
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : i64
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[VAR_12_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_]], [[VAR_12_]]#0, [[VAR_12_]]#1] : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_1_]]{{.}}[[VAR_12_]]#0, [[VAR_12_]]#1] : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [3, 12], value = dense<1.000000e+00> : tensor<3x12xf32>} : () -> memref<3x12xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [4], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [4], value = dense<[5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [4], value = dense<[9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [4], value = dense<[1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [4], value = dense<[1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [4], value = dense<[2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7){
// CHECK-DAG:         [[VAR_12_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_]] to [[CST_3_]]){
// CHECK:               [[VAR_32_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_12_1_]], [[VAR_32_]]#0, [[VAR_32_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_2_]]{{.}}[[VAR_32_]]#0, [[VAR_32_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_14_:%.+]] = builtin.unrealized_conversion_cast [[RES_2_]] : memref<2x3xf32> to tensor<2x3xf32>
// CHECK-DAG:         [[VAR_15_:%.+]] = builtin.unrealized_conversion_cast [[VAR_1_]] : memref<3x12xf32> to tensor<3x12xf32>
// CHECK:             [[VAR_16_:%.+]] = "onnx.MatMul"([[VAR_14_]], [[VAR_15_]]) : (tensor<2x3xf32>, tensor<3x12xf32>) -> tensor<2x12xf32>
// CHECK-DAG:         [[VAR_17_:%.+]] = builtin.unrealized_conversion_cast [[VAR_16_]] : tensor<2x12xf32> to memref<2x12xf32>
// CHECK-DAG:         [[VAR_18_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK-DAG:         [[VAR_19_:%.+]] = builtin.unrealized_conversion_cast [[VAR_2_]] : memref<4x4xf32> to tensor<4x4xf32>
// CHECK:             [[VAR_20_:%.+]] = "onnx.MatMul"([[VAR_18_]], [[VAR_19_]]) : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[VAR_21_:%.+]] = builtin.unrealized_conversion_cast [[VAR_20_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[VAR_22_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK-DAG:         [[VAR_23_:%.+]] = builtin.unrealized_conversion_cast [[VAR_3_]] : memref<4x4xf32> to tensor<4x4xf32>
// CHECK:             [[VAR_24_:%.+]] = "onnx.MatMul"([[VAR_22_]], [[VAR_23_]]) : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[VAR_25_:%.+]] = builtin.unrealized_conversion_cast [[VAR_24_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_]] to [[CST_4_]]){
// CHECK:               [[VAR_32_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_32_1_]]#0, [[VAR_32_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[VAR_34_:%.+]] = affine.apply [[MAP_3_]](){{.}}[[VAR_32_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_17_MEM_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_32_1_]]#0, [[VAR_34_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_25_MEM_:%.+]] = krnl.load [[VAR_25_]]{{.}}[[VAR_32_1_]]#0, [[VAR_32_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_37_:%.+]] = arith.addf [[LOAD_VAR_17_MEM_]], [[LOAD_VAR_25_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_6_MEM_:%.+]] = krnl.load [[VAR_6_]]{{.}}[[VAR_32_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_9_MEM_:%.+]] = krnl.load [[VAR_9_]]{{.}}[[VAR_32_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_40_:%.+]] = arith.addf [[VAR_37_]], [[LOAD_VAR_6_MEM_]] : f32
// CHECK:               [[VAR_41_:%.+]] = arith.addf [[VAR_40_]], [[LOAD_VAR_9_MEM_]] : f32
// CHECK:               [[VAR_42_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_41_]] : f32
// CHECK:               [[VAR_43_:%.+]] = math.exp [[VAR_42_]] : f32
// CHECK:               [[VAR_44_:%.+]] = arith.addf [[VAR_43_]], [[CST_1_dot_000000_]] : f32
// CHECK:               [[VAR_45_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_44_]] : f32
// CHECK:               krnl.store [[VAR_45_]], [[RES_3_]]{{.}}[[VAR_32_1_]]#0, [[VAR_32_1_]]#1] : memref<2x4xf32>
// CHECK:               [[VAR_46_:%.+]] = arith.mulf [[VAR_45_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               krnl.store [[VAR_46_]], [[RES_4_]]{{.}}[[VAR_32_1_]]#0, [[VAR_32_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_27_:%.+]] = builtin.unrealized_conversion_cast [[RES_4_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK-DAG:         [[VAR_28_:%.+]] = builtin.unrealized_conversion_cast [[VAR_4_]] : memref<4x4xf32> to tensor<4x4xf32>
// CHECK:             [[VAR_29_:%.+]] = "onnx.MatMul"([[VAR_27_]], [[VAR_28_]]) : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[VAR_30_:%.+]] = builtin.unrealized_conversion_cast [[VAR_29_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_7_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_4_]]#1 -> [[I_8_:%.+]] = [[CST_0_]] to [[CST_4_]]){
// CHECK:               [[VAR_32_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_32_2_]]#0, [[VAR_32_2_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_17_MEM_1_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_32_2_]]#0, [[VAR_32_2_]]#1] : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_17_MEM_2_:%.+]] = krnl.load [[VAR_21_]]{{.}}[[VAR_32_2_]]#0, [[VAR_32_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_25_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_17_MEM_1_]], [[LOAD_VAR_17_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_37_1_:%.+]] = krnl.load [[VAR_5_]]{{.}}[[VAR_32_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_6_MEM_1_:%.+]] = krnl.load [[VAR_8_]]{{.}}[[VAR_32_2_]]#1] : memref<4xf32>
// CHECK:               [[LOAD_VAR_9_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_25_MEM_1_]], [[VAR_37_1_]] : f32
// CHECK:               [[VAR_40_1_:%.+]] = arith.addf [[LOAD_VAR_9_MEM_1_]], [[LOAD_VAR_6_MEM_1_]] : f32
// CHECK:               [[VAR_41_1_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_40_1_]] : f32
// CHECK:               [[VAR_42_1_:%.+]] = math.exp [[VAR_41_1_]] : f32
// CHECK:               [[VAR_43_1_:%.+]] = arith.addf [[VAR_42_1_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:           [[VAR_44_1_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_43_1_]] : f32
// CHECK-DAG:           [[VAR_45_1_:%.+]] = affine.apply [[MAP_4_]](){{.}}[[VAR_32_2_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_17_MEM_3_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_32_2_]]#0, [[VAR_45_1_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_30_MEM_:%.+]] = krnl.load [[VAR_30_]]{{.}}[[VAR_32_2_]]#0, [[VAR_32_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_48_:%.+]] = arith.addf [[LOAD_VAR_17_MEM_3_]], [[LOAD_VAR_30_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_7_MEM_:%.+]] = krnl.load [[VAR_7_]]{{.}}[[VAR_32_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_10_MEM_:%.+]] = krnl.load [[VAR_10_]]{{.}}[[VAR_32_2_]]#1] : memref<4xf32>
// CHECK:               [[VAR_51_:%.+]] = arith.addf [[VAR_48_]], [[LOAD_VAR_7_MEM_]] : f32
// CHECK:               [[VAR_52_:%.+]] = arith.addf [[VAR_51_]], [[LOAD_VAR_10_MEM_]] : f32
// CHECK-DAG:           [[VAR_53_:%.+]] = math.tanh [[VAR_52_]] : f32
// CHECK-DAG:           [[VAR_54_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_44_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_55_:%.+]] = arith.mulf [[VAR_54_]], [[VAR_53_]] : f32
// CHECK-DAG:           [[VAR_56_:%.+]] = arith.mulf [[VAR_44_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_57_:%.+]] = arith.addf [[VAR_55_]], [[VAR_56_]] : f32
// CHECK:               krnl.store [[VAR_57_]], [[RES_1_]]{{.}}[[VAR_32_2_]]#0, [[VAR_32_2_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_]]_1, [[CST_8_]], [[VAR_c0_]], [[VAR_c0_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64, index, index) -> ()
// CHECK:           return [[RES_]] : memref<1x2x4xf32>
// CHECK:         }
}

// -----

func.func private @test_gru_reverse_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x12x3xf32>, %arg2: tensor<1x12x4xf32>, %arg3: tensor<1x24xf32>, %arg4: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64, direction = "reverse"} : (tensor<7x2x3xf32>, tensor<1x12x3xf32>, tensor<1x12x4xf32>, tensor<1x24xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>
// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<(d0) -> (-d0 + 6)>
// CHECK-DAG: [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 + 4)>
// CHECK-DAG: [[MAP_2_:#.+]] = affine_map<()[s0] -> (s0 + 8)>
// CHECK-LABEL:  func private @test_gru_reverse_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x12x3xf32>, [[PARAM_2_:%.+]]: memref<1x12x4xf32>, [[PARAM_3_:%.+]]: memref<1x24xf32>, [[PARAM_4_:%.+]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[VAR_c8_i64_:%.+]] = arith.constant 8 : i64
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_3_]] : memref<1x24xf32> to tensor<1x24xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_2_]] : memref<1x12x4xf32> to tensor<1x12x4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_1_]] : memref<1x12x3xf32> to tensor<1x12x3xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[VAR_22_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[VAR_c0_]], [[VAR_22_]]#0, [[VAR_22_]]#1] : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]]{{.}}[[VAR_22_]]#0, [[VAR_22_]]#1] : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.SqueezeV11"([[VAR_2_]]) {axes = [0]} : (tensor<1x12x3xf32>) -> tensor<12x3xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.SqueezeV11"([[VAR_1_]]) {axes = [0]} : (tensor<1x12x4xf32>) -> tensor<12x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Transpose"([[VAR_6_]]) {perm = [1, 0]} : (tensor<12x3xf32>) -> tensor<3x12xf32>
// CHECK-DAG:       [[VAR_9_:%.+]]:3 = "onnx.SplitV11"([[VAR_7_]]) {axis = 0 : si64} : (tensor<12x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Transpose"([[VAR_9_]]#0) {perm = [1, 0]} : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Transpose"([[VAR_9_]]#1) {perm = [1, 0]} : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Transpose"([[VAR_9_]]#2) {perm = [1, 0]} : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.SqueezeV11"([[VAR_0_]]) {axes = [0]} : (tensor<1x24xf32>) -> tensor<24xf32>
// CHECK:           [[VAR_14_:%.+]]:6 = "onnx.SplitV11"([[VAR_13_]]) {axis = 0 : si64} : (tensor<24xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
// CHECK-DAG:       [[VAR_15_:%.+]] = builtin.unrealized_conversion_cast [[VAR_14_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = builtin.unrealized_conversion_cast [[VAR_14_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = builtin.unrealized_conversion_cast [[VAR_14_]]#2 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = builtin.unrealized_conversion_cast [[VAR_14_]]#3 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = builtin.unrealized_conversion_cast [[VAR_14_]]#4 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = builtin.unrealized_conversion_cast [[VAR_14_]]#5 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7){
// CHECK:             [[VAR_22_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_1_:%.+]] = affine.apply [[MAP_0_]]([[VAR_22_1_]])
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]){
// CHECK:               [[VAR_42_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[LOAD_PARAM_4_MEM_1_]], [[VAR_42_]]#0, [[VAR_42_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_2_]]{{.}}[[VAR_42_]]#0, [[VAR_42_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             [[VAR_26_:%.+]] = builtin.unrealized_conversion_cast [[RES_2_]] : memref<2x3xf32> to tensor<2x3xf32>
// CHECK:             [[VAR_27_:%.+]] = "onnx.MatMul"([[VAR_26_]], [[VAR_8_]]) : (tensor<2x3xf32>, tensor<3x12xf32>) -> tensor<2x12xf32>
// CHECK-DAG:         [[VAR_28_:%.+]] = builtin.unrealized_conversion_cast [[VAR_27_]] : tensor<2x12xf32> to memref<2x12xf32>
// CHECK-DAG:         [[VAR_29_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_30_:%.+]] = "onnx.MatMul"([[VAR_29_]], [[VAR_10_]]) : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[VAR_31_:%.+]] = builtin.unrealized_conversion_cast [[VAR_30_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[VAR_32_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_33_:%.+]] = "onnx.MatMul"([[VAR_32_]], [[VAR_11_]]) : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[VAR_34_:%.+]] = builtin.unrealized_conversion_cast [[VAR_33_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[VAR_42_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_42_1_]]#0, [[VAR_42_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[VAR_44_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_42_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_28_MEM_:%.+]] = krnl.load [[VAR_28_]]{{.}}[[VAR_42_1_]]#0, [[VAR_44_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_34_MEM_:%.+]] = krnl.load [[VAR_34_]]{{.}}[[VAR_42_1_]]#0, [[VAR_42_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_47_:%.+]] = arith.addf [[LOAD_VAR_28_MEM_]], [[LOAD_VAR_34_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_16_MEM_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_42_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_19_MEM_:%.+]] = krnl.load [[VAR_19_]]{{.}}[[VAR_42_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_50_:%.+]] = arith.addf [[VAR_47_]], [[LOAD_VAR_16_MEM_]] : f32
// CHECK:               [[VAR_51_:%.+]] = arith.addf [[VAR_50_]], [[LOAD_VAR_19_MEM_]] : f32
// CHECK:               [[VAR_52_:%.+]] = arith.subf [[VAR_cst_]], [[VAR_51_]] : f32
// CHECK:               [[VAR_53_:%.+]] = math.exp [[VAR_52_]] : f32
// CHECK:               [[VAR_54_:%.+]] = arith.addf [[VAR_53_]], [[VAR_cst_0_]] : f32
// CHECK:               [[VAR_55_:%.+]] = arith.divf [[VAR_cst_0_]], [[VAR_54_]] : f32
// CHECK:               krnl.store [[VAR_55_]], [[RES_3_]]{{.}}[[VAR_42_1_]]#0, [[VAR_42_1_]]#1] : memref<2x4xf32>
// CHECK:               [[VAR_56_:%.+]] = arith.mulf [[VAR_55_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               krnl.store [[VAR_56_]], [[RES_4_]]{{.}}[[VAR_42_1_]]#0, [[VAR_42_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             [[VAR_38_:%.+]] = builtin.unrealized_conversion_cast [[RES_4_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_39_:%.+]] = "onnx.MatMul"([[VAR_38_]], [[VAR_12_]]) : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[VAR_40_:%.+]] = builtin.unrealized_conversion_cast [[VAR_39_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_7_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_4_]]#1 -> [[I_8_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[VAR_42_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_42_1_]]#0, [[VAR_42_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_28_MEM_1_:%.+]] = krnl.load [[VAR_28_]]{{.}}[[VAR_42_2_]]#0, [[VAR_42_2_]]#1] : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_28_MEM_2_:%.+]] = krnl.load [[VAR_31_]]{{.}}[[VAR_42_2_]]#0, [[VAR_42_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_34_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_28_MEM_1_]], [[LOAD_VAR_28_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_47_1_:%.+]] = krnl.load [[VAR_15_]]{{.}}[[VAR_42_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_16_MEM_1_:%.+]] = krnl.load [[VAR_18_]]{{.}}[[VAR_42_2_]]#1] : memref<4xf32>
// CHECK:               [[LOAD_VAR_19_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_34_MEM_1_]], [[VAR_47_1_]] : f32
// CHECK:               [[VAR_50_1_:%.+]] = arith.addf [[LOAD_VAR_19_MEM_1_]], [[LOAD_VAR_16_MEM_1_]] : f32
// CHECK:               [[VAR_51_1_:%.+]] = arith.subf [[VAR_cst_]], [[VAR_50_1_]] : f32
// CHECK:               [[VAR_52_1_:%.+]] = math.exp [[VAR_51_1_]] : f32
// CHECK:               [[VAR_53_1_:%.+]] = arith.addf [[VAR_52_1_]], [[VAR_cst_0_]] : f32
// CHECK-DAG:           [[VAR_54_1_:%.+]] = arith.divf [[VAR_cst_0_]], [[VAR_53_1_]] : f32
// CHECK-DAG:           [[VAR_55_1_:%.+]] = affine.apply [[MAP_2_]](){{.}}[[VAR_42_2_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_28_MEM_3_:%.+]] = krnl.load [[VAR_28_]]{{.}}[[VAR_42_2_]]#0, [[VAR_55_1_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_40_MEM_:%.+]] = krnl.load [[VAR_40_]]{{.}}[[VAR_42_2_]]#0, [[VAR_42_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_58_:%.+]] = arith.addf [[LOAD_VAR_28_MEM_3_]], [[LOAD_VAR_40_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_17_MEM_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_42_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_20_MEM_:%.+]] = krnl.load [[VAR_20_]]{{.}}[[VAR_42_2_]]#1] : memref<4xf32>
// CHECK:               [[VAR_61_:%.+]] = arith.addf [[VAR_58_]], [[LOAD_VAR_17_MEM_]] : f32
// CHECK:               [[VAR_62_:%.+]] = arith.addf [[VAR_61_]], [[LOAD_VAR_20_MEM_]] : f32
// CHECK-DAG:           [[VAR_63_:%.+]] = math.tanh [[VAR_62_]] : f32
// CHECK-DAG:           [[VAR_64_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_54_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_65_:%.+]] = arith.mulf [[VAR_64_]], [[VAR_63_]] : f32
// CHECK-DAG:           [[VAR_66_:%.+]] = arith.mulf [[VAR_54_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_67_:%.+]] = arith.addf [[VAR_65_]], [[VAR_66_]] : f32
// CHECK:               krnl.store [[VAR_67_]], [[RES_1_]]{{.}}[[VAR_42_1_]]#0, [[VAR_42_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_1_]], [[VAR_c8_i64_]], [[VAR_c0_]], [[VAR_c0_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64, index, index) -> ()
// CHECK:           return [[RES_]] : memref<1x2x4xf32>
// CHECK:         }
}

// -----

func.func private @test_gru_bidirectional_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<2x12x3xf32>, %arg2: tensor<2x12x4xf32>, %arg3: tensor<2x24xf32>, %arg4: tensor<2x2x4xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64, direction = "bidirectional"} : (tensor<7x2x3xf32>, tensor<2x12x3xf32>, tensor<2x12x4xf32>, tensor<2x24xf32>, none, tensor<2x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>
// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 + 4)>
// CHECK-DAG: [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 + 8)>
// CHECK-DAG: [[MAP_2_:#.+]] = affine_map<(d0) -> (-d0 + 6)>
// CHECK-LABEL:  func private @test_gru_bidirectional_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<2x12x3xf32>, [[PARAM_2_:%.+]]: memref<2x12x4xf32>, [[PARAM_3_:%.+]]: memref<2x24xf32>, [[PARAM_4_:%.+]]: memref<2x2x4xf32>) -> memref<2x2x4xf32> {
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_3_]] : memref<2x24xf32> to tensor<2x24xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_2_]] : memref<2x12x4xf32> to tensor<2x12x4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_1_]] : memref<2x12x3xf32> to tensor<2x12x3xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[VAR_43_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[VAR_c0_]], [[VAR_43_]]#0, [[VAR_43_]]#1] : memref<2x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]]{{.}}[[VAR_43_]]#0, [[VAR_43_]]#1] : memref<2x4xf32>
// CHECK:             [[LOAD_PARAM_4_MEM_1_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[VAR_c1_]], [[VAR_43_]]#0, [[VAR_43_]]#1] : memref<2x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_1_]], [[RES_2_]]{{.}}[[VAR_43_]]#0, [[VAR_43_]]#1] : memref<2x4xf32>
// CHECK:           }
// CHECK:           [[VAR_7_:%.+]]:2 = "onnx.SplitV11"([[VAR_2_]]) {axis = 0 : si64} : (tensor<2x12x3xf32>) -> (tensor<1x12x3xf32>, tensor<1x12x3xf32>)
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.SqueezeV11"([[VAR_7_]]#0) {axes = [0]} : (tensor<1x12x3xf32>) -> tensor<12x3xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.SqueezeV11"([[VAR_7_]]#1) {axes = [0]} : (tensor<1x12x3xf32>) -> tensor<12x3xf32>
// CHECK-DAG:       [[VAR_10_:%.+]]:2 = "onnx.SplitV11"([[VAR_1_]]) {axis = 0 : si64} : (tensor<2x12x4xf32>) -> (tensor<1x12x4xf32>, tensor<1x12x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.SqueezeV11"([[VAR_10_]]#0) {axes = [0]} : (tensor<1x12x4xf32>) -> tensor<12x4xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.SqueezeV11"([[VAR_10_]]#1) {axes = [0]} : (tensor<1x12x4xf32>) -> tensor<12x4xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Transpose"([[VAR_8_]]) {perm = [1, 0]} : (tensor<12x3xf32>) -> tensor<3x12xf32>
// CHECK:           [[VAR_14_:%.+]]:3 = "onnx.SplitV11"([[VAR_11_]]) {axis = 0 : si64} : (tensor<12x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.Transpose"([[VAR_14_]]#0) {perm = [1, 0]} : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = "onnx.Transpose"([[VAR_14_]]#1) {perm = [1, 0]} : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Transpose"([[VAR_14_]]#2) {perm = [1, 0]} : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Transpose"([[VAR_9_]]) {perm = [1, 0]} : (tensor<12x3xf32>) -> tensor<3x12xf32>
// CHECK-DAG:       [[VAR_19_:%.+]]:3 = "onnx.SplitV11"([[VAR_12_]]) {axis = 0 : si64} : (tensor<12x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_20_:%.+]] = "onnx.Transpose"([[VAR_19_]]#0) {perm = [1, 0]} : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.Transpose"([[VAR_19_]]#1) {perm = [1, 0]} : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = "onnx.Transpose"([[VAR_19_]]#2) {perm = [1, 0]} : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_23_:%.+]]:2 = "onnx.SplitV11"([[VAR_0_]]) {axis = 0 : si64} : (tensor<2x24xf32>) -> (tensor<1x24xf32>, tensor<1x24xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_24_:%.+]] = "onnx.SqueezeV11"([[VAR_23_]]#0) {axes = [0]} : (tensor<1x24xf32>) -> tensor<24xf32>
// CHECK-DAG:       [[VAR_25_:%.+]] = "onnx.SqueezeV11"([[VAR_23_]]#1) {axes = [0]} : (tensor<1x24xf32>) -> tensor<24xf32>
// CHECK:           [[VAR_26_:%.+]]:6 = "onnx.SplitV11"([[VAR_24_]]) {axis = 0 : si64} : (tensor<24xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
// CHECK-DAG:       [[VAR_27_:%.+]] = builtin.unrealized_conversion_cast [[VAR_26_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_28_:%.+]] = builtin.unrealized_conversion_cast [[VAR_26_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_29_:%.+]] = builtin.unrealized_conversion_cast [[VAR_26_]]#2 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_30_:%.+]] = builtin.unrealized_conversion_cast [[VAR_26_]]#3 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_31_:%.+]] = builtin.unrealized_conversion_cast [[VAR_26_]]#4 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_32_:%.+]] = builtin.unrealized_conversion_cast [[VAR_26_]]#5 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_33_:%.+]]:6 = "onnx.SplitV11"([[VAR_25_]]) {axis = 0 : si64} : (tensor<24xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_34_:%.+]] = builtin.unrealized_conversion_cast [[VAR_33_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_35_:%.+]] = builtin.unrealized_conversion_cast [[VAR_33_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_36_:%.+]] = builtin.unrealized_conversion_cast [[VAR_33_]]#2 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_37_:%.+]] = builtin.unrealized_conversion_cast [[VAR_33_]]#3 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_38_:%.+]] = builtin.unrealized_conversion_cast [[VAR_33_]]#4 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_39_:%.+]] = builtin.unrealized_conversion_cast [[VAR_33_]]#5 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7){
// CHECK-DAG:         [[VAR_43_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]){
// CHECK:               [[VAR_62_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_43_1_]], [[VAR_62_]]#0, [[VAR_62_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_3_]]{{.}}[[VAR_62_]]#0, [[VAR_62_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             [[VAR_46_:%.+]] = builtin.unrealized_conversion_cast [[RES_3_]] : memref<2x3xf32> to tensor<2x3xf32>
// CHECK:             [[VAR_47_:%.+]] = "onnx.MatMul"([[VAR_46_]], [[VAR_13_]]) : (tensor<2x3xf32>, tensor<3x12xf32>) -> tensor<2x12xf32>
// CHECK-DAG:         [[VAR_48_:%.+]] = builtin.unrealized_conversion_cast [[VAR_47_]] : tensor<2x12xf32> to memref<2x12xf32>
// CHECK-DAG:         [[VAR_49_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_50_:%.+]] = "onnx.MatMul"([[VAR_49_]], [[VAR_15_]]) : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[VAR_51_:%.+]] = builtin.unrealized_conversion_cast [[VAR_50_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[VAR_52_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_53_:%.+]] = "onnx.MatMul"([[VAR_52_]], [[VAR_16_]]) : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[VAR_54_:%.+]] = builtin.unrealized_conversion_cast [[VAR_53_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[VAR_62_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_62_1_]]#0, [[VAR_62_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[VAR_64_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_62_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_48_MEM_:%.+]] = krnl.load [[VAR_48_]]{{.}}[[VAR_62_1_]]#0, [[VAR_64_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_54_MEM_:%.+]] = krnl.load [[VAR_54_]]{{.}}[[VAR_62_1_]]#0, [[VAR_62_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_67_:%.+]] = arith.addf [[LOAD_VAR_48_MEM_]], [[LOAD_VAR_54_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_28_MEM_:%.+]] = krnl.load [[VAR_28_]]{{.}}[[VAR_62_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_31_MEM_:%.+]] = krnl.load [[VAR_31_]]{{.}}[[VAR_62_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_70_:%.+]] = arith.addf [[VAR_67_]], [[LOAD_VAR_28_MEM_]] : f32
// CHECK:               [[VAR_71_:%.+]] = arith.addf [[VAR_70_]], [[LOAD_VAR_31_MEM_]] : f32
// CHECK:               [[VAR_72_:%.+]] = arith.subf [[VAR_cst_]], [[VAR_71_]] : f32
// CHECK:               [[VAR_73_:%.+]] = math.exp [[VAR_72_]] : f32
// CHECK:               [[VAR_74_:%.+]] = arith.addf [[VAR_73_]], [[VAR_cst_0_]] : f32
// CHECK:               [[VAR_75_:%.+]] = arith.divf [[VAR_cst_0_]], [[VAR_74_]] : f32
// CHECK:               krnl.store [[VAR_75_]], [[RES_4_]]{{.}}[[VAR_62_1_]]#0, [[VAR_62_1_]]#1] : memref<2x4xf32>
// CHECK:               [[VAR_76_:%.+]] = arith.mulf [[VAR_75_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               krnl.store [[VAR_76_]], [[RES_5_]]{{.}}[[VAR_62_1_]]#0, [[VAR_62_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             [[VAR_58_:%.+]] = builtin.unrealized_conversion_cast [[RES_5_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_59_:%.+]] = "onnx.MatMul"([[VAR_58_]], [[VAR_17_]]) : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[VAR_60_:%.+]] = builtin.unrealized_conversion_cast [[VAR_59_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_7_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_4_]]#1 -> [[I_8_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[VAR_62_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_62_2_]]#0, [[VAR_62_2_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_48_MEM_1_:%.+]] = krnl.load [[VAR_48_]]{{.}}[[VAR_62_2_]]#0, [[VAR_62_2_]]#1] : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_48_MEM_2_:%.+]] = krnl.load [[VAR_51_]]{{.}}[[VAR_62_2_]]#0, [[VAR_62_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_54_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_48_MEM_1_]], [[LOAD_VAR_48_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_67_1_:%.+]] = krnl.load [[VAR_27_]]{{.}}[[VAR_62_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_28_MEM_1_:%.+]] = krnl.load [[VAR_30_]]{{.}}[[VAR_62_2_]]#1] : memref<4xf32>
// CHECK:               [[LOAD_VAR_31_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_54_MEM_1_]], [[VAR_67_1_]] : f32
// CHECK:               [[VAR_70_1_:%.+]] = arith.addf [[LOAD_VAR_31_MEM_1_]], [[LOAD_VAR_28_MEM_1_]] : f32
// CHECK:               [[VAR_71_1_:%.+]] = arith.subf [[VAR_cst_]], [[VAR_70_1_]] : f32
// CHECK:               [[VAR_72_1_:%.+]] = math.exp [[VAR_71_1_]] : f32
// CHECK:               [[VAR_73_1_:%.+]] = arith.addf [[VAR_72_1_]], [[VAR_cst_0_]] : f32
// CHECK-DAG:           [[VAR_74_1_:%.+]] = arith.divf [[VAR_cst_0_]], [[VAR_73_1_]] : f32
// CHECK-DAG:           [[VAR_75_1_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_62_2_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_48_MEM_3_:%.+]] = krnl.load [[VAR_48_]]{{.}}[[VAR_62_2_]]#0, [[VAR_75_1_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_60_MEM_:%.+]] = krnl.load [[VAR_60_]]{{.}}[[VAR_62_2_]]#0, [[VAR_62_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_78_:%.+]] = arith.addf [[LOAD_VAR_48_MEM_3_]], [[LOAD_VAR_60_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_29_MEM_:%.+]] = krnl.load [[VAR_29_]]{{.}}[[VAR_62_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_32_MEM_:%.+]] = krnl.load [[VAR_32_]]{{.}}[[VAR_62_2_]]#1] : memref<4xf32>
// CHECK:               [[VAR_81_:%.+]] = arith.addf [[VAR_78_]], [[LOAD_VAR_29_MEM_]] : f32
// CHECK:               [[VAR_82_:%.+]] = arith.addf [[VAR_81_]], [[LOAD_VAR_32_MEM_]] : f32
// CHECK-DAG:           [[VAR_83_:%.+]] = math.tanh [[VAR_82_]] : f32
// CHECK-DAG:           [[VAR_84_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_74_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_85_:%.+]] = arith.mulf [[VAR_84_]], [[VAR_83_]] : f32
// CHECK-DAG:           [[VAR_86_:%.+]] = arith.mulf [[VAR_74_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_87_:%.+]] = arith.addf [[VAR_85_]], [[VAR_86_]] : f32
// CHECK:               krnl.store [[VAR_87_]], [[RES_1_]]{{.}}[[VAR_62_2_]]#0, [[VAR_62_2_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_5_]]) with ([[LOOP_5_]] -> [[I_9_:%.+]] = 0 to 7){
// CHECK:             [[VAR_43_2_:%.+]] = krnl.get_induction_var_value([[LOOP_5_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[RES_3_:%.+]] = affine.apply [[MAP_2_]]([[VAR_43_2_]])
// CHECK-DAG:         [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_6_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_6_]]#0, [[LOOP_6_]]#1) with ([[LOOP_6_]]#0 -> [[I_10_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_6_]]#1 -> [[I_11_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]){
// CHECK:               [[LOAD_PARAM_0_MEM_1_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_6_]]#0, [[LOOP_6_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[RES_3_]], [[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_2_]], [[RES_6_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             [[VAR_47_1_:%.+]] = builtin.unrealized_conversion_cast [[RES_6_]] : memref<2x3xf32> to tensor<2x3xf32>
// CHECK:             [[VAR_48_1_:%.+]] = "onnx.MatMul"([[VAR_47_1_]], [[VAR_18_]]) : (tensor<2x3xf32>, tensor<3x12xf32>) -> tensor<2x12xf32>
// CHECK-DAG:         [[VAR_49_1_:%.+]] = builtin.unrealized_conversion_cast [[VAR_48_1_]] : tensor<2x12xf32> to memref<2x12xf32>
// CHECK-DAG:         [[VAR_50_1_:%.+]] = builtin.unrealized_conversion_cast [[RES_2_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_51_1_:%.+]] = "onnx.MatMul"([[VAR_50_1_]], [[VAR_20_]]) : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[VAR_52_1_:%.+]] = builtin.unrealized_conversion_cast [[VAR_51_1_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[VAR_53_1_:%.+]] = builtin.unrealized_conversion_cast [[RES_2_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_54_1_:%.+]] = "onnx.MatMul"([[VAR_53_1_]], [[VAR_21_]]) : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[RES_4_:%.+]] = builtin.unrealized_conversion_cast [[VAR_54_1_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[RES_7_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[LOOP_7_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_7_]]#0, [[LOOP_7_]]#1) with ([[LOOP_7_]]#0 -> [[I_12_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_7_]]#1 -> [[I_13_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[LOAD_PARAM_0_MEM_1_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_7_]]#0, [[LOOP_7_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[RES_2_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_48_MEM_2_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_54_MEM_1_:%.+]] = krnl.load [[VAR_49_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_VAR_48_MEM_2_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[VAR_67_1_:%.+]] = krnl.load [[RES_4_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_28_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_54_MEM_1_]], [[VAR_67_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_31_MEM_1_:%.+]] = krnl.load [[VAR_35_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_70_1_:%.+]] = krnl.load [[VAR_38_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_71_2_:%.+]] = arith.addf [[LOAD_VAR_28_MEM_1_]], [[LOAD_VAR_31_MEM_1_]] : f32
// CHECK:               [[VAR_72_2_:%.+]] = arith.addf [[VAR_71_2_]], [[VAR_70_1_]] : f32
// CHECK:               [[VAR_73_2_:%.+]] = arith.subf [[VAR_cst_]], [[VAR_72_2_]] : f32
// CHECK:               [[VAR_74_2_:%.+]] = math.exp [[VAR_73_2_]] : f32
// CHECK:               [[VAR_75_2_:%.+]] = arith.addf [[VAR_74_2_]], [[VAR_cst_0_]] : f32
// CHECK:               [[VAR_76_1_:%.+]] = arith.divf [[VAR_cst_0_]], [[VAR_75_2_]] : f32
// CHECK:               krnl.store [[VAR_76_1_]], [[RES_7_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK:               [[LOAD_VAR_60_MEM_1_:%.+]] = arith.mulf [[VAR_76_1_]], [[LOAD_PARAM_0_MEM_2_]] : f32
// CHECK:               krnl.store [[LOAD_VAR_60_MEM_1_]], [[RES_8_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             [[VAR_59_1_:%.+]] = builtin.unrealized_conversion_cast [[RES_8_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_60_1_:%.+]] = "onnx.MatMul"([[VAR_59_1_]], [[VAR_22_]]) : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[LOOP_4_:%.+]] = builtin.unrealized_conversion_cast [[VAR_60_1_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[LOOP_8_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_8_]]#0, [[LOOP_8_]]#1) with ([[LOOP_8_]]#0 -> [[I_14_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_8_]]#1 -> [[I_15_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[LOAD_PARAM_0_MEM_1_1_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_8_]]#0, [[LOOP_8_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_2_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_48_MEM_2_1_:%.+]] = krnl.load [[VAR_49_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_54_MEM_1_1_:%.+]] = krnl.load [[VAR_52_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_67_2_:%.+]] = arith.addf [[LOAD_VAR_48_MEM_2_1_]], [[LOAD_VAR_54_MEM_1_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_28_MEM_1_1_:%.+]] = krnl.load [[VAR_34_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_31_MEM_1_1_:%.+]] = krnl.load [[VAR_37_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_70_2_:%.+]] = arith.addf [[VAR_67_2_]], [[LOAD_VAR_28_MEM_1_1_]] : f32
// CHECK:               [[VAR_71_3_:%.+]] = arith.addf [[VAR_70_2_]], [[LOAD_VAR_31_MEM_1_1_]] : f32
// CHECK:               [[VAR_72_3_:%.+]] = arith.subf [[VAR_cst_]], [[VAR_71_3_]] : f32
// CHECK:               [[VAR_73_3_:%.+]] = math.exp [[VAR_72_3_]] : f32
// CHECK:               [[VAR_74_3_:%.+]] = arith.addf [[VAR_73_3_]], [[VAR_cst_0_]] : f32
// CHECK-DAG:           [[VAR_75_3_:%.+]] = arith.divf [[VAR_cst_0_]], [[VAR_74_3_]] : f32
// CHECK-DAG:           [[VAR_76_2_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_60_MEM_1_:%.+]] = krnl.load [[VAR_49_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#0, [[VAR_76_2_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[VAR_78_1_:%.+]] = krnl.load [[LOOP_4_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_29_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_60_MEM_1_]], [[VAR_78_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_32_MEM_1_:%.+]] = krnl.load [[VAR_36_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_81_1_:%.+]] = krnl.load [[VAR_39_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_82_1_:%.+]] = arith.addf [[LOAD_VAR_29_MEM_1_]], [[LOAD_VAR_32_MEM_1_]] : f32
// CHECK:               [[VAR_83_1_:%.+]] = arith.addf [[VAR_82_1_]], [[VAR_81_1_]] : f32
// CHECK-DAG:           [[VAR_84_1_:%.+]] = math.tanh [[VAR_83_1_]] : f32
// CHECK-DAG:           [[VAR_85_1_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_75_3_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_86_1_:%.+]] = arith.mulf [[VAR_85_1_]], [[VAR_84_1_]] : f32
// CHECK-DAG:           [[VAR_87_1_:%.+]] = arith.mulf [[VAR_75_3_]], [[LOAD_PARAM_0_MEM_2_1_]] : f32
// CHECK:               [[VAR_88_:%.+]] = arith.addf [[VAR_86_1_]], [[VAR_87_1_]] : f32
// CHECK:               krnl.store [[VAR_88_]], [[RES_2_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOOP_9_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_9_]]#0, [[LOOP_9_]]#1) with ([[LOOP_9_]]#0 -> [[I_16_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_9_]]#1 -> [[I_17_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:             [[VAR_43_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_9_]]#0, [[LOOP_9_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[RES_3_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_43_]]#0, [[VAR_43_]]#1] : memref<2x4xf32>
// CHECK:             krnl.store [[RES_3_1_]], [[RES_]]{{.}}[[VAR_c0_]], [[VAR_43_3_]]#0, [[VAR_43_3_]]#1] : memref<2x2x4xf32>
// CHECK:             [[RES_6_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_43_3_]]#0, [[VAR_43_3_]]#1] : memref<2x4xf32>
// CHECK:             krnl.store [[RES_6_]], [[RES_]]{{.}}[[VAR_c1_]], [[VAR_43_3_]]#0, [[VAR_43_3_]]#1] : memref<2x2x4xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x2x4xf32>
// CHECK:         }
}

// -----

func.func private @test_gru_unknown_dims(%arg0: tensor<?x?x?xf32>, %arg1: tensor<1x12x?xf32>, %arg2: tensor<1x12x4xf32>, %arg3: tensor<1x24xf32>, %arg4: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64} : (tensor<?x?x?xf32>, tensor<1x12x?xf32>, tensor<1x12x4xf32>, tensor<1x24xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>
// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 + 4)>
// CHECK-DAG: [[MAP_2_:#.+]] = affine_map<()[s0] -> (s0 + 8)>
// CHECK-LABEL:  func private @test_gru_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x?xf32>, [[PARAM_1_:%.+]]: memref<1x12x?xf32>, [[PARAM_2_:%.+]]: memref<1x12x4xf32>, [[PARAM_3_:%.+]]: memref<1x24xf32>, [[PARAM_4_:%.+]]: memref<1x2x4xf32>) -> memref<1x?x4xf32> {
// CHECK-DAG:       [[VAR_c4_i64_:%.+]] = arith.constant 4 : i64
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_3_]] : memref<1x24xf32> to tensor<1x24xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_2_]] : memref<1x12x4xf32> to tensor<1x12x4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_1_]] : memref<1x12x?xf32> to tensor<1x12x?xf32>
// CHECK:           [[VAR_3_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_3_]]) {{.*}}: memref<1x?x4xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<?x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[VAR_5_]]) {{.*}}: memref<?x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_5_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[VAR_27_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[VAR_c0_]], [[VAR_27_]]#0, [[VAR_27_]]#1] : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]]{{.}}[[VAR_27_]]#0, [[VAR_27_]]#1] : memref<?x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.SqueezeV11"([[VAR_2_]]) {axes = [0]} : (tensor<1x12x?xf32>) -> tensor<12x?xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.SqueezeV11"([[VAR_1_]]) {axes = [0]} : (tensor<1x12x4xf32>) -> tensor<12x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Transpose"([[VAR_8_]]) {perm = [1, 0]} : (tensor<12x?xf32>) -> tensor<?x12xf32>
// CHECK-DAG:       [[VAR_11_:%.+]]:3 = "onnx.SplitV11"([[VAR_9_]]) {axis = 0 : si64} : (tensor<12x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Transpose"([[VAR_11_]]#0) {perm = [1, 0]} : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Transpose"([[VAR_11_]]#1) {perm = [1, 0]} : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.Transpose"([[VAR_11_]]#2) {perm = [1, 0]} : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.SqueezeV11"([[VAR_0_]]) {axes = [0]} : (tensor<1x24xf32>) -> tensor<24xf32>
// CHECK:           [[VAR_16_:%.+]]:6 = "onnx.SplitV11"([[VAR_15_]]) {axis = 0 : si64} : (tensor<24xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
// CHECK-DAG:       [[VAR_17_:%.+]] = builtin.unrealized_conversion_cast [[VAR_16_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = builtin.unrealized_conversion_cast [[VAR_16_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = builtin.unrealized_conversion_cast [[VAR_16_]]#2 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = builtin.unrealized_conversion_cast [[VAR_16_]]#3 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = builtin.unrealized_conversion_cast [[VAR_16_]]#4 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = builtin.unrealized_conversion_cast [[VAR_16_]]#5 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK-DAG:       [[VAR_24_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?x?x?xf32>
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to [[MAP_0_]]([[VAR_24_]])){
// CHECK-DAG:         [[VAR_27_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_1_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<?x?x?xf32>
// CHECK-DAG:         [[VAR_29_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c2_]] : memref<?x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloc([[LOAD_PARAM_4_MEM_1_]], [[VAR_29_]]) {{.*}}: memref<?x?xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[VAR_c0_]] to [[LOAD_PARAM_4_MEM_1_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[VAR_c0_]] to [[VAR_29_]]){
// CHECK:               [[VAR_48_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_27_1_]], [[VAR_48_]]#0, [[VAR_48_]]#1] : memref<?x?x?xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_2_]]{{.}}[[VAR_48_]]#0, [[VAR_48_]]#1] : memref<?x?xf32>
// CHECK:             }
// CHECK:             [[VAR_32_:%.+]] = builtin.unrealized_conversion_cast [[RES_2_]] : memref<?x?xf32> to tensor<?x?xf32>
// CHECK:             [[VAR_33_:%.+]] = "onnx.MatMul"([[VAR_32_]], [[VAR_10_]]) : (tensor<?x?xf32>, tensor<?x12xf32>) -> tensor<?x12xf32>
// CHECK-DAG:         [[VAR_34_:%.+]] = builtin.unrealized_conversion_cast [[VAR_33_]] : tensor<?x12xf32> to memref<?x12xf32>
// CHECK-DAG:         [[VAR_35_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<?x4xf32> to tensor<?x4xf32>
// CHECK:             [[VAR_36_:%.+]] = "onnx.MatMul"([[VAR_35_]], [[VAR_12_]]) : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
// CHECK-DAG:         [[VAR_37_:%.+]] = builtin.unrealized_conversion_cast [[VAR_36_]] : tensor<?x4xf32> to memref<?x4xf32>
// CHECK-DAG:         [[VAR_38_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<?x4xf32> to tensor<?x4xf32>
// CHECK:             [[VAR_39_:%.+]] = "onnx.MatMul"([[VAR_38_]], [[VAR_13_]]) : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
// CHECK-DAG:         [[VAR_40_:%.+]] = builtin.unrealized_conversion_cast [[VAR_39_]] : tensor<?x4xf32> to memref<?x4xf32>
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc([[VAR_5_]]) {{.*}}: memref<?x4xf32>
// CHECK-DAG:         [[RES_4_:%.+]] = memref.alloc([[VAR_5_]]) {{.*}}: memref<?x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[VAR_c0_]] to [[VAR_5_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[VAR_48_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_48_1_]]#0, [[VAR_48_1_]]#1] : memref<?x4xf32>
// CHECK-DAG:           [[VAR_50_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_48_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_34_MEM_:%.+]] = krnl.load [[VAR_34_]]{{.}}[[VAR_48_1_]]#0, [[VAR_50_]]{{.}} : memref<?x12xf32>
// CHECK-DAG:           [[LOAD_VAR_40_MEM_:%.+]] = krnl.load [[VAR_40_]]{{.}}[[VAR_48_1_]]#0, [[VAR_48_1_]]#1] : memref<?x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_53_:%.+]] = arith.addf [[LOAD_VAR_34_MEM_]], [[LOAD_VAR_40_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_18_MEM_:%.+]] = krnl.load [[VAR_18_]]{{.}}[[VAR_48_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_21_MEM_:%.+]] = krnl.load [[VAR_21_]]{{.}}[[VAR_48_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_56_:%.+]] = arith.addf [[VAR_53_]], [[LOAD_VAR_18_MEM_]] : f32
// CHECK:               [[VAR_57_:%.+]] = arith.addf [[VAR_56_]], [[LOAD_VAR_21_MEM_]] : f32
// CHECK:               [[VAR_58_:%.+]] = arith.subf [[VAR_cst_]], [[VAR_57_]] : f32
// CHECK:               [[VAR_59_:%.+]] = math.exp [[VAR_58_]] : f32
// CHECK:               [[VAR_60_:%.+]] = arith.addf [[VAR_59_]], [[VAR_cst_0_]] : f32
// CHECK:               [[VAR_61_:%.+]] = arith.divf [[VAR_cst_0_]], [[VAR_60_]] : f32
// CHECK:               krnl.store [[VAR_61_]], [[RES_3_]]{{.}}[[VAR_48_1_]]#0, [[VAR_48_1_]]#1] : memref<?x4xf32>
// CHECK:               [[VAR_62_:%.+]] = arith.mulf [[VAR_61_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               krnl.store [[VAR_62_]], [[RES_4_]]{{.}}[[VAR_48_1_]]#0, [[VAR_48_1_]]#1] : memref<?x4xf32>
// CHECK:             }
// CHECK:             [[VAR_44_:%.+]] = builtin.unrealized_conversion_cast [[RES_4_]] : memref<?x4xf32> to tensor<?x4xf32>
// CHECK:             [[VAR_45_:%.+]] = "onnx.MatMul"([[VAR_44_]], [[VAR_14_]]) : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
// CHECK-DAG:         [[VAR_46_:%.+]] = builtin.unrealized_conversion_cast [[VAR_45_]] : tensor<?x4xf32> to memref<?x4xf32>
// CHECK-DAG:         [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_7_:%.+]] = [[VAR_c0_]] to [[VAR_5_]], [[LOOP_4_]]#1 -> [[I_8_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[VAR_48_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_48_2_]]#0, [[VAR_48_2_]]#1] : memref<?x4xf32>
// CHECK-DAG:           [[LOAD_VAR_34_MEM_1_:%.+]] = krnl.load [[VAR_34_]]{{.}}[[VAR_48_2_]]#0, [[VAR_48_2_]]#1] : memref<?x12xf32>
// CHECK-DAG:           [[LOAD_VAR_34_MEM_2_:%.+]] = krnl.load [[VAR_37_]]{{.}}[[VAR_48_2_]]#0, [[VAR_48_2_]]#1] : memref<?x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_40_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_34_MEM_1_]], [[LOAD_VAR_34_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_53_1_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_48_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_18_MEM_1_:%.+]] = krnl.load [[VAR_20_]]{{.}}[[VAR_48_2_]]#1] : memref<4xf32>
// CHECK:               [[LOAD_VAR_21_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_40_MEM_1_]], [[VAR_53_1_]] : f32
// CHECK:               [[VAR_56_1_:%.+]] = arith.addf [[LOAD_VAR_21_MEM_1_]], [[LOAD_VAR_18_MEM_1_]] : f32
// CHECK:               [[VAR_57_1_:%.+]] = arith.subf [[VAR_cst_]], [[VAR_56_1_]] : f32
// CHECK:               [[VAR_58_1_:%.+]] = math.exp [[VAR_57_1_]] : f32
// CHECK:               [[VAR_59_1_:%.+]] = arith.addf [[VAR_58_1_]], [[VAR_cst_0_]] : f32
// CHECK-DAG:           [[VAR_60_1_:%.+]] = arith.divf [[VAR_cst_0_]], [[VAR_59_1_]] : f32
// CHECK-DAG:           [[VAR_61_1_:%.+]] = affine.apply [[MAP_2_]](){{.}}[[VAR_48_2_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_34_MEM_3_:%.+]] = krnl.load [[VAR_34_]]{{.}}[[VAR_48_2_]]#0, [[VAR_61_1_]]{{.}} : memref<?x12xf32>
// CHECK-DAG:           [[LOAD_VAR_46_MEM_:%.+]] = krnl.load [[VAR_46_]]{{.}}[[VAR_48_2_]]#0, [[VAR_48_2_]]#1] : memref<?x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_64_:%.+]] = arith.addf [[LOAD_VAR_34_MEM_3_]], [[LOAD_VAR_46_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_19_MEM_:%.+]] = krnl.load [[VAR_19_]]{{.}}[[VAR_48_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_22_MEM_:%.+]] = krnl.load [[VAR_22_]]{{.}}[[VAR_48_2_]]#1] : memref<4xf32>
// CHECK:               [[VAR_67_:%.+]] = arith.addf [[VAR_64_]], [[LOAD_VAR_19_MEM_]] : f32
// CHECK:               [[VAR_68_:%.+]] = arith.addf [[VAR_67_]], [[LOAD_VAR_22_MEM_]] : f32
// CHECK-DAG:           [[VAR_69_:%.+]] = math.tanh [[VAR_68_]] : f32
// CHECK-DAG:           [[VAR_70_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_60_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_71_:%.+]] = arith.mulf [[VAR_70_]], [[VAR_69_]] : f32
// CHECK-DAG:           [[VAR_72_:%.+]] = arith.mulf [[VAR_60_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_73_:%.+]] = arith.addf [[VAR_71_]], [[VAR_72_]] : f32
// CHECK:               krnl.store [[VAR_73_]], [[RES_1_]]{{.}}[[VAR_48_2_]]#0, [[VAR_48_2_]]#1] : memref<?x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           [[VAR_25_:%.+]] = arith.index_cast [[VAR_5_]] : index to i64
// CHECK:           [[VAR_26_:%.+]] = arith.muli [[VAR_25_]], [[VAR_c4_i64_]] : i64
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_1_]], [[VAR_26_]], [[VAR_c0_]], [[VAR_c0_]]) : (memref<1x?x4xf32>, memref<?x4xf32>, i64, index, index) -> ()
// CHECK:           return [[RES_]] : memref<1x?x4xf32>
// CHECK:         }
}

// -----

func.func @gru_default_backend(%arg0: tensor<1x3x2xf32>, %arg1: tensor<1x15x2xf32>, %arg2: tensor<1x15x5xf32>) -> tensor<1x3x5xf32>  {
    %0 = "onnx.NoValue"() {value} : () -> none
    %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %0, %0, %0) {hidden_size = 5 : si64} : (tensor<1x3x2xf32>, tensor<1x15x2xf32>, tensor<1x15x5xf32>, none, none, none) -> (none, tensor<1x3x5xf32>)
    return %Y_h : tensor<1x3x5xf32>
// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 + 5)>
// CHECK-DAG: [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 + 10)>
// CHECK-LABEL:  func @gru_default_backend
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x3x2xf32>, [[PARAM_1_:%.+]]: memref<1x15x2xf32>, [[PARAM_2_:%.+]]: memref<1x15x5xf32>) -> memref<1x3x5xf32> {
// CHECK-DAG:       [[VAR_c15_i64_:%.+]] = arith.constant 15 : i64
// CHECK-DAG:       [[VAR_c5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_0_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_2_]] : memref<1x15x5xf32> to tensor<1x15x5xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_1_]] : memref<1x15x2xf32> to tensor<1x15x2xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x3x5xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<3x5xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 5){
// CHECK:             [[VAR_13_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[VAR_cst_0_]], [[RES_1_]]{{.}}[[VAR_13_]]#0, [[VAR_13_]]#1] : memref<3x5xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.SqueezeV11"([[VAR_1_]]) {axes = [0]} : (tensor<1x15x2xf32>) -> tensor<15x2xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.SqueezeV11"([[VAR_0_]]) {axes = [0]} : (tensor<1x15x5xf32>) -> tensor<15x5xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Transpose"([[VAR_5_]]) {perm = [1, 0]} : (tensor<15x2xf32>) -> tensor<2x15xf32>
// CHECK-DAG:       [[VAR_8_:%.+]]:3 = "onnx.SplitV11"([[VAR_6_]]) {axis = 0 : si64} : (tensor<15x5xf32>) -> (tensor<5x5xf32>, tensor<5x5xf32>, tensor<5x5xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Transpose"([[VAR_8_]]#0) {perm = [1, 0]} : (tensor<5x5xf32>) -> tensor<5x5xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Transpose"([[VAR_8_]]#1) {perm = [1, 0]} : (tensor<5x5xf32>) -> tensor<5x5xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Transpose"([[VAR_8_]]#2) {perm = [1, 0]} : (tensor<5x5xf32>) -> tensor<5x5xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 1){
// CHECK-DAG:         [[VAR_13_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<3x2xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]]){
// CHECK:               [[VAR_32_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_13_1_]], [[VAR_32_]]#0, [[VAR_32_]]#1] : memref<1x3x2xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_2_]]{{.}}[[VAR_32_]]#0, [[VAR_32_]]#1] : memref<3x2xf32>
// CHECK:             }
// CHECK:             [[VAR_16_:%.+]] = builtin.unrealized_conversion_cast [[RES_2_]] : memref<3x2xf32> to tensor<3x2xf32>
// CHECK:             [[VAR_17_:%.+]] = "onnx.MatMul"([[VAR_16_]], [[VAR_7_]]) : (tensor<3x2xf32>, tensor<2x15xf32>) -> tensor<3x15xf32>
// CHECK-DAG:         [[VAR_18_:%.+]] = builtin.unrealized_conversion_cast [[VAR_17_]] : tensor<3x15xf32> to memref<3x15xf32>
// CHECK-DAG:         [[VAR_19_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<3x5xf32> to tensor<3x5xf32>
// CHECK:             [[VAR_20_:%.+]] = "onnx.MatMul"([[VAR_19_]], [[VAR_9_]]) : (tensor<3x5xf32>, tensor<5x5xf32>) -> tensor<3x5xf32>
// CHECK-DAG:         [[VAR_21_:%.+]] = builtin.unrealized_conversion_cast [[VAR_20_]] : tensor<3x5xf32> to memref<3x5xf32>
// CHECK-DAG:         [[VAR_22_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<3x5xf32> to tensor<3x5xf32>
// CHECK:             [[VAR_23_:%.+]] = "onnx.MatMul"([[VAR_22_]], [[VAR_10_]]) : (tensor<3x5xf32>, tensor<5x5xf32>) -> tensor<3x5xf32>
// CHECK-DAG:         [[VAR_24_:%.+]] = builtin.unrealized_conversion_cast [[VAR_23_]] : tensor<3x5xf32> to memref<3x5xf32>
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<3x5xf32>
// CHECK-DAG:         [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<3x5xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[VAR_c0_]] to [[VAR_c5_]]){
// CHECK:               [[VAR_32_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_32_1_]]#0, [[VAR_32_1_]]#1] : memref<3x5xf32>
// CHECK-DAG:           [[VAR_34_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_32_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_18_MEM_:%.+]] = krnl.load [[VAR_18_]]{{.}}[[VAR_32_1_]]#0, [[VAR_34_]]{{.}} : memref<3x15xf32>
// CHECK-DAG:           [[LOAD_VAR_24_MEM_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_32_1_]]#0, [[VAR_32_1_]]#1] : memref<3x5xf32>
// CHECK:               [[VAR_37_:%.+]] = arith.addf [[LOAD_VAR_18_MEM_]], [[LOAD_VAR_24_MEM_]] : f32
// CHECK:               [[VAR_38_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_37_]] : f32
// CHECK:               [[VAR_39_:%.+]] = math.exp [[VAR_38_]] : f32
// CHECK:               [[VAR_40_:%.+]] = arith.addf [[VAR_39_]], [[VAR_cst_]] : f32
// CHECK:               [[VAR_41_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_40_]] : f32
// CHECK:               krnl.store [[VAR_41_]], [[RES_3_]]{{.}}[[VAR_32_1_]]#0, [[VAR_32_1_]]#1] : memref<3x5xf32>
// CHECK:               [[VAR_42_:%.+]] = arith.mulf [[VAR_41_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               krnl.store [[VAR_42_]], [[RES_4_]]{{.}}[[VAR_32_1_]]#0, [[VAR_32_1_]]#1] : memref<3x5xf32>
// CHECK:             }
// CHECK:             [[VAR_28_:%.+]] = builtin.unrealized_conversion_cast [[RES_4_]] : memref<3x5xf32> to tensor<3x5xf32>
// CHECK:             [[VAR_29_:%.+]] = "onnx.MatMul"([[VAR_28_]], [[VAR_11_]]) : (tensor<3x5xf32>, tensor<5x5xf32>) -> tensor<3x5xf32>
// CHECK-DAG:         [[VAR_30_:%.+]] = builtin.unrealized_conversion_cast [[VAR_29_]] : tensor<3x5xf32> to memref<3x5xf32>
// CHECK-DAG:         [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_7_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]], [[LOOP_4_]]#1 -> [[I_8_:%.+]] = [[VAR_c0_]] to [[VAR_c5_]]){
// CHECK:               [[VAR_32_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_32_1_]]#0, [[VAR_32_1_]]#1] : memref<3x5xf32>
// CHECK-DAG:           [[LOAD_VAR_18_MEM_1_:%.+]] = krnl.load [[VAR_18_]]{{.}}[[VAR_32_2_]]#0, [[VAR_32_2_]]#1] : memref<3x15xf32>
// CHECK-DAG:           [[LOAD_VAR_18_MEM_2_:%.+]] = krnl.load [[VAR_21_]]{{.}}[[VAR_32_2_]]#0, [[VAR_32_2_]]#1] : memref<3x5xf32>
// CHECK:               [[LOAD_VAR_24_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_18_MEM_1_]], [[LOAD_VAR_18_MEM_2_]] : f32
// CHECK:               [[VAR_37_1_:%.+]] = arith.subf [[VAR_cst_0_]], [[LOAD_VAR_24_MEM_1_]] : f32
// CHECK:               [[VAR_38_1_:%.+]] = math.exp [[VAR_37_1_]] : f32
// CHECK:               [[VAR_39_1_:%.+]] = arith.addf [[VAR_38_1_]], [[VAR_cst_]] : f32
// CHECK-DAG:           [[VAR_40_1_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_39_1_]] : f32
// CHECK-DAG:           [[VAR_41_1_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_32_2_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_18_MEM_3_:%.+]] = krnl.load [[VAR_18_]]{{.}}[[VAR_32_2_]]#0, [[VAR_41_1_]]{{.}} : memref<3x15xf32>
// CHECK-DAG:           [[LOAD_VAR_30_MEM_:%.+]] = krnl.load [[VAR_30_]]{{.}}[[VAR_32_2_]]#0, [[VAR_32_2_]]#1] : memref<3x5xf32>
// CHECK:               [[VAR_44_:%.+]] = arith.addf [[LOAD_VAR_18_MEM_3_]], [[LOAD_VAR_30_MEM_]] : f32
// CHECK-DAG:           [[VAR_45_:%.+]] = math.tanh [[VAR_44_]] : f32
// CHECK-DAG:           [[VAR_46_:%.+]] = arith.subf [[VAR_cst_]], [[VAR_40_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_47_:%.+]] = arith.mulf [[VAR_46_]], [[VAR_45_]] : f32
// CHECK-DAG:           [[VAR_48_:%.+]] = arith.mulf [[VAR_40_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_49_:%.+]] = arith.addf [[VAR_47_]], [[VAR_48_]] : f32
// CHECK:               krnl.store [[VAR_49_]], [[RES_1_]]{{.}}[[VAR_32_1_]]#0, [[VAR_32_1_]]#1] : memref<3x5xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_1_]], [[VAR_c15_i64_]], [[VAR_c0_]], [[VAR_c0_]]) : (memref<1x3x5xf32>, memref<3x5xf32>, i64, index, index) -> ()
// CHECK:           return [[RES_]] : memref<1x3x5xf32>
// CHECK:         }
}
