// RUN: onnx-mlir-opt -O3 --shape-inference --convert-onnx-to-krnl='emit-intermediate-ir' --canonicalize %s -split-input-file | FileCheck %s

func private @test_gru_forward_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x12x3xf32>, %arg2: tensor<1x12x4xf32>, %arg3: tensor<1x24xf32>, %arg4: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64} : (tensor<7x2x3xf32>, tensor<1x12x3xf32>, tensor<1x12x4xf32>, tensor<1x24xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-DAG: #map0 = affine_map<()[s0] -> (s0 + 4)>
// CHECK-DAG: #map1 = affine_map<()[s0] -> (s0 + 8)>
// CHECK-LABEL:  func private @test_gru_forward_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x12x3xf32>, [[PARAM_2_:%.+]]: memref<1x12x4xf32>, [[PARAM_3_:%.+]]: memref<1x24xf32>, [[PARAM_4_:%.+]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[VAR_c32_i64_:%.+]] = arith.constant 32 : i64
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[UCC_PARAM_3:%.+]] = builtin.unrealized_conversion_cast %arg3 : memref<1x24xf32> to tensor<1x24xf32>
// CHECK-DAG:       [[UCC_PARAM_2:%.+]] = builtin.unrealized_conversion_cast %arg2 : memref<1x12x4xf32> to tensor<1x12x4xf32>
// CHECK-DAG:       [[UCC_PARAM_1:%.+]] = builtin.unrealized_conversion_cast %arg1 : memref<1x12x3xf32> to tensor<1x12x3xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[VAR_c0_]], [[IV]]#0, [[IV]]#1{{.}} : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]][[[IV]]#0, [[IV]]#1] : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.SqueezeV11"([[UCC_PARAM_1]]) {axes = [0]} : (tensor<1x12x3xf32>) -> memref<12x3xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.SqueezeV11"([[UCC_PARAM_2]]) {axes = [0]} : (tensor<1x12x4xf32>) -> memref<12x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Transpose"([[VAR_3_]]) {perm = [1, 0]} : (memref<12x3xf32>) -> memref<3x12xf32>
// CHECK-DAG:       [[VAR_6_:%.+]]:3 = "onnx.SplitV11"([[VAR_4_]]) {axis = 0 : si64} : (memref<12x4xf32>) -> (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Transpose"([[VAR_6_]]#0) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Transpose"([[VAR_6_]]#1) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Transpose"([[VAR_6_]]#2) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.SqueezeV11"([[UCC_PARAM_3]]) {axes = [0]} : (tensor<1x24xf32>) -> memref<24xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]]:6 = "onnx.SplitV11"([[VAR_10_]]) {axis = 0 : si64} : (memref<24xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7){
// CHECK:             [[IV:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]){
// CHECK:               [[VAR_23_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[IV]], [[VAR_23_]]#0, [[VAR_23_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_2_]]{{.}}[[VAR_23_]]#0, [[VAR_23_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_15_:%.+]] = "onnx.MatMul"([[RES_2_]], [[VAR_5_]]) : (memref<2x3xf32>, memref<3x12xf32>) -> memref<2x12xf32>
// CHECK-DAG:         [[VAR_16_:%.+]] = "onnx.MatMul"([[RES_1_]], [[VAR_7_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_17_:%.+]] = "onnx.MatMul"([[RES_1_]], [[VAR_8_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[VAR_23_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_23_1_]]#0, [[VAR_23_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[VAR_25_:%.+]] = affine.apply #map0(){{.}}[[VAR_23_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_15_MEM_:%.+]] = krnl.load [[VAR_15_]]{{.}}[[VAR_23_1_]]#0, [[VAR_25_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_17_MEM_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_23_1_]]#0, [[VAR_23_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_28_:%.+]] = arith.addf [[LOAD_VAR_15_MEM_]], [[LOAD_VAR_17_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_:%.+]] = krnl.load [[VAR_11_]]#1{{.}}[[VAR_23_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_11_MEM_1_:%.+]] = krnl.load [[VAR_11_]]#4{{.}}[[VAR_23_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_31_:%.+]] = arith.addf [[VAR_28_]], [[LOAD_VAR_11_MEM_]] : f32
// CHECK:               [[VAR_32_:%.+]] = arith.addf [[VAR_31_]], [[LOAD_VAR_11_MEM_1_]] : f32
// CHECK:               [[VAR_33_:%.+]] = arith.subf [[VAR_cst_]], [[VAR_32_]] : f32
// CHECK:               [[VAR_34_:%.+]] = math.exp [[VAR_33_]] : f32
// CHECK:               [[VAR_35_:%.+]] = arith.addf [[VAR_34_]], [[VAR_cst_0_]] : f32
// CHECK:               [[VAR_36_:%.+]] = arith.divf [[VAR_cst_0_]], [[VAR_35_]] : f32
// CHECK:               krnl.store [[VAR_36_]], [[RES_3_]]{{.}}[[VAR_23_1_]]#0, [[VAR_23_1_]]#1] : memref<2x4xf32>
// CHECK:               [[VAR_37_:%.+]] = arith.mulf [[VAR_36_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               krnl.store [[VAR_37_]], [[RES_4_]]{{.}}[[VAR_23_1_]]#0, [[VAR_23_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_21_:%.+]] = "onnx.MatMul"([[RES_4_]], [[VAR_9_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_7_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_4_]]#1 -> [[I_8_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[VAR_23_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_15_MEM_1_:%.+]] = krnl.load [[VAR_15_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1] : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_15_MEM_2_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_17_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_15_MEM_1_]], [[LOAD_VAR_15_MEM_2_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_2_:%.+]] = krnl.load [[VAR_11_]]#0{{.}}[[VAR_23_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_11_MEM_3_:%.+]] = krnl.load [[VAR_11_]]#3{{.}}[[VAR_23_2_]]#1] : memref<4xf32>
// CHECK:               [[LOAD_VAR_11_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_17_MEM_1_]], [[LOAD_VAR_11_MEM_2_]] : f32
// CHECK:               [[VAR_31_1_:%.+]] = arith.addf [[LOAD_VAR_11_MEM_1_]], [[LOAD_VAR_11_MEM_3_]] : f32
// CHECK:               [[VAR_32_1_:%.+]] = arith.subf [[VAR_cst_]], [[VAR_31_1_]] : f32
// CHECK:               [[VAR_33_1_:%.+]] = math.exp [[VAR_32_1_]] : f32
// CHECK:               [[VAR_34_1_:%.+]] = arith.addf [[VAR_33_1_]], [[VAR_cst_0_]] : f32
// CHECK-DAG:           [[VAR_35_1_:%.+]] = arith.divf [[VAR_cst_0_]], [[VAR_34_1_]] : f32
// CHECK-DAG:           [[VAR_36_1_:%.+]] = affine.apply #map1(){{.}}[[VAR_23_2_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_15_MEM_3_:%.+]] = krnl.load [[VAR_15_]]{{.}}[[VAR_23_2_]]#0, [[VAR_36_1_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_21_MEM_:%.+]] = krnl.load [[VAR_21_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_39_:%.+]] = arith.addf [[LOAD_VAR_15_MEM_3_]], [[LOAD_VAR_21_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_4_:%.+]] = krnl.load [[VAR_11_]]#2{{.}}[[VAR_23_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_11_MEM_5_:%.+]] = krnl.load [[VAR_11_]]#5{{.}}[[VAR_23_2_]]#1] : memref<4xf32>
// CHECK:               [[VAR_42_:%.+]] = arith.addf [[VAR_39_]], [[LOAD_VAR_11_MEM_4_]] : f32
// CHECK:               [[VAR_43_:%.+]] = arith.addf [[VAR_42_]], [[LOAD_VAR_11_MEM_5_]] : f32
// CHECK-DAG:           [[VAR_44_:%.+]] = math.tanh [[VAR_43_]] : f32
// CHECK-DAG:           [[VAR_45_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_35_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_46_:%.+]] = arith.mulf [[VAR_45_]], [[VAR_44_]] : f32
// CHECK-DAG:           [[VAR_47_:%.+]] = arith.mulf [[VAR_35_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_48_:%.+]] = arith.addf [[VAR_46_]], [[VAR_47_]] : f32
// CHECK:               krnl.store [[VAR_48_]], [[RES_1_]]{{.}}[[VAR_23_2_]]#0, [[VAR_23_2_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_1_]], [[VAR_c32_i64_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
// CHECK:           return [[RES_]] : memref<1x2x4xf32>
// CHECK:         }
}

// -----

func private @test_gru_forward_mode_linear_before_reset(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x12x3xf32>, %arg2: tensor<1x12x4xf32>, %arg3: tensor<1x24xf32>, %arg4: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64, linear_before_reset = 1 : si64} : (tensor<7x2x3xf32>, tensor<1x12x3xf32>, tensor<1x12x4xf32>, tensor<1x24xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>
  
// CHECK-DAG: #map0 = affine_map<()[s0] -> (s0 + 4)>
// CHECK-DAG: #map1 = affine_map<()[s0] -> (s0 + 8)>
// CHECK-LABEL:  func private @test_gru_forward_mode_linear_before_reset
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x12x3xf32>, [[PARAM_2_:%.+]]: memref<1x12x4xf32>, [[PARAM_3_:%.+]]: memref<1x24xf32>, [[PARAM_4_:%.+]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[VAR_c32_i64_:%.+]] = arith.constant 32 : i64
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[UCC_PARAM_3:%.+]] = builtin.unrealized_conversion_cast %arg3 : memref<1x24xf32> to tensor<1x24xf32>
// CHECK-DAG:       [[UCC_PARAM_2:%.+]] = builtin.unrealized_conversion_cast %arg2 : memref<1x12x4xf32> to tensor<1x12x4xf32>
// CHECK-DAG:       [[UCC_PARAM_1:%.+]] = builtin.unrealized_conversion_cast %arg1 : memref<1x12x3xf32> to tensor<1x12x3xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]][[[VAR_c0_]], [[IV]]#0, [[IV]]#1] : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]][[[IV]]#0, [[IV]]#1] : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.SqueezeV11"([[UCC_PARAM_1]]) {axes = [0]} : (tensor<1x12x3xf32>) -> memref<12x3xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.SqueezeV11"([[UCC_PARAM_2]]) {axes = [0]} : (tensor<1x12x4xf32>) -> memref<12x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Transpose"([[VAR_3_]]) {perm = [1, 0]} : (memref<12x3xf32>) -> memref<3x12xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Transpose"([[VAR_4_]]) {perm = [1, 0]} : (memref<12x4xf32>) -> memref<4x12xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.SqueezeV11"([[UCC_PARAM_3]]) {axes = [0]} : (tensor<1x24xf32>) -> memref<24xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]]:6 = "onnx.SplitV11"([[VAR_7_]]) {axis = 0 : si64} : (memref<24xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7){
// CHECK:             [[IV:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]){
// CHECK:               [[VAR_15_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[IV]], [[VAR_15_]]#0, [[VAR_15_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_2_]]{{.}}[[VAR_15_]]#0, [[VAR_15_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_12_:%.+]] = "onnx.MatMul"([[RES_2_]], [[VAR_5_]]) : (memref<2x3xf32>, memref<3x12xf32>) -> memref<2x12xf32>
// CHECK-DAG:         [[VAR_13_:%.+]] = "onnx.MatMul"([[RES_1_]], [[VAR_6_]]) : (memref<2x4xf32>, memref<4x12xf32>) -> memref<2x12xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[VAR_15_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_15_1_]]#0, [[VAR_15_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_12_MEM_:%.+]] = krnl.load [[VAR_12_]]{{.}}[[VAR_15_1_]]#0, [[VAR_15_1_]]#1] : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_13_MEM_:%.+]] = krnl.load [[VAR_13_]]{{.}}[[VAR_15_1_]]#0, [[VAR_15_1_]]#1] : memref<2x12xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_19_:%.+]] = arith.addf [[LOAD_VAR_12_MEM_]], [[LOAD_VAR_13_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_8_MEM_:%.+]] = krnl.load [[VAR_8_]]#0{{.}}[[VAR_15_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_8_MEM_1_:%.+]] = krnl.load [[VAR_8_]]#3{{.}}[[VAR_15_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_22_:%.+]] = arith.addf [[VAR_19_]], [[LOAD_VAR_8_MEM_]] : f32
// CHECK:               [[VAR_23_:%.+]] = arith.addf [[VAR_22_]], [[LOAD_VAR_8_MEM_1_]] : f32
// CHECK:               [[VAR_24_:%.+]] = arith.subf [[VAR_cst_]], [[VAR_23_]] : f32
// CHECK:               [[VAR_25_:%.+]] = math.exp [[VAR_24_]] : f32
// CHECK:               [[VAR_26_:%.+]] = arith.addf [[VAR_25_]], [[VAR_cst_0_]] : f32
// CHECK-DAG:           [[VAR_27_:%.+]] = arith.divf [[VAR_cst_0_]], [[VAR_26_]] : f32
// CHECK-DAG:           [[VAR_28_:%.+]] = affine.apply #map0(){{.}}[[VAR_15_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_12_MEM_1_:%.+]] = krnl.load [[VAR_12_]]{{.}}[[VAR_15_1_]]#0, [[VAR_28_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[VAR_30_:%.+]] = affine.apply #map0(){{.}}[[VAR_15_1_]]#1]
// CHECK:               [[LOAD_VAR_13_MEM_1_:%.+]] = krnl.load [[VAR_13_]]{{.}}[[VAR_15_1_]]#0, [[VAR_30_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[VAR_32_:%.+]] = arith.addf [[LOAD_VAR_12_MEM_1_]], [[LOAD_VAR_13_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_8_MEM_2_:%.+]] = krnl.load [[VAR_8_]]#1{{.}}[[VAR_15_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_8_MEM_3_:%.+]] = krnl.load [[VAR_8_]]#4{{.}}[[VAR_15_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_35_:%.+]] = arith.addf [[VAR_32_]], [[LOAD_VAR_8_MEM_2_]] : f32
// CHECK:               [[VAR_36_:%.+]] = arith.addf [[VAR_35_]], [[LOAD_VAR_8_MEM_3_]] : f32
// CHECK:               [[VAR_37_:%.+]] = arith.subf [[VAR_cst_]], [[VAR_36_]] : f32
// CHECK:               [[VAR_38_:%.+]] = math.exp [[VAR_37_]] : f32
// CHECK:               [[VAR_39_:%.+]] = arith.addf [[VAR_38_]], [[VAR_cst_0_]] : f32
// CHECK-DAG:           [[VAR_40_:%.+]] = arith.divf [[VAR_cst_0_]], [[VAR_39_]] : f32
// CHECK-DAG:           [[VAR_41_:%.+]] = affine.apply #map1(){{.}}[[VAR_15_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_12_MEM_2_:%.+]] = krnl.load [[VAR_12_]]{{.}}[[VAR_15_1_]]#0, [[VAR_41_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[VAR_43_:%.+]] = affine.apply #map1(){{.}}[[VAR_15_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_13_MEM_2_:%.+]] = krnl.load [[VAR_13_]]{{.}}[[VAR_15_1_]]#0, [[VAR_43_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_8_MEM_4_:%.+]] = krnl.load [[VAR_8_]]#5{{.}}[[VAR_15_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_46_:%.+]] = arith.addf [[LOAD_VAR_13_MEM_2_]], [[LOAD_VAR_8_MEM_4_]] : f32
// CHECK:               [[VAR_47_:%.+]] = arith.mulf [[VAR_40_]], [[VAR_46_]] : f32
// CHECK-DAG:           [[VAR_48_:%.+]] = arith.addf [[LOAD_VAR_12_MEM_2_]], [[VAR_47_]] : f32
// CHECK-DAG:           [[LOAD_VAR_8_MEM_5_:%.+]] = krnl.load [[VAR_8_]]#2{{.}}[[VAR_15_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_50_:%.+]] = arith.addf [[VAR_48_]], [[LOAD_VAR_8_MEM_5_]] : f32
// CHECK-DAG:           [[VAR_51_:%.+]] = math.tanh [[VAR_50_]] : f32
// CHECK-DAG:           [[VAR_52_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_27_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_53_:%.+]] = arith.mulf [[VAR_52_]], [[VAR_51_]] : f32
// CHECK-DAG:           [[VAR_54_:%.+]] = arith.mulf [[VAR_27_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_55_:%.+]] = arith.addf [[VAR_53_]], [[VAR_54_]] : f32
// CHECK:               krnl.store [[VAR_55_]], [[RES_1_]]{{.}}[[VAR_15_1_]]#0, [[VAR_15_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_1_]], [[VAR_c32_i64_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
// CHECK:           return [[RES_]] : memref<1x2x4xf32>
// CHECK:         }
}

// -----

func private @test_gru_forward_mode_constant_weight_and_bias(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %w = "onnx.Constant"() {value = dense<[[[1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]]> : tensor<1x12x3xf32>} : () -> tensor<1x12x3xf32> 
  %r = "onnx.Constant"() {value = dense<[[[2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.]]]> : tensor<1x12x4xf32>} : () -> tensor<1x12x4xf32> 
  %b = "onnx.Constant"() {value = dense<[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.]]> : tensor<1x24xf32>} : () -> tensor<1x24xf32> 

  %Y, %Y_h = "onnx.GRU"(%arg0, %w, %r, %b, %cst, %arg1) {hidden_size = 4 : si64} : (tensor<7x2x3xf32>, tensor<1x12x3xf32>, tensor<1x12x4xf32>, tensor<1x24xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-DAG: #map0 = affine_map<()[s0] -> (s0 + 4)>
// CHECK-DAG: #map1 = affine_map<()[s0] -> (s0 + 8)>
// CHECK-LABEL:  func private @test_gru_forward_mode_constant_weight_and_bias
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[VAR_c32_i64_:%.+]] = arith.constant 32 : i64
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]][[[VAR_c0_]], [[IV]]#0, [[IV]]#1] : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_1_]][[[IV]]#0, [[IV]]#1] : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_3_:%.+]] = "krnl.global"() {name = "constant_5", shape = [3, 12], value = dense<1.000000e+00> : tensor<3x12xf32>} : () -> memref<3x12xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "krnl.global"() {name = "constant_9", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "krnl.global"() {name = "constant_10", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "krnl.global"() {name = "constant_11", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "krnl.global"() {name = "constant_13", shape = [4], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "krnl.global"() {name = "constant_14", shape = [4], value = dense<[5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "krnl.global"() {name = "constant_15", shape = [4], value = dense<[9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "krnl.global"() {name = "constant_16", shape = [4], value = dense<[1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "krnl.global"() {name = "constant_17", shape = [4], value = dense<[1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "krnl.global"() {name = "constant_18", shape = [4], value = dense<[2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7){
// CHECK:             [[IV:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]){
// CHECK:               [[VAR_24_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[IV]], [[VAR_24_]]#0, [[VAR_24_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_2_]]{{.}}[[VAR_24_]]#0, [[VAR_24_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_16_:%.+]] = "onnx.MatMul"([[RES_2_]], [[VAR_3_]]) : (memref<2x3xf32>, memref<3x12xf32>) -> memref<2x12xf32>
// CHECK-DAG:         [[VAR_17_:%.+]] = "onnx.MatMul"([[RES_1_]], [[VAR_4_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_18_:%.+]] = "onnx.MatMul"([[RES_1_]], [[VAR_5_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[VAR_24_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_24_1_]]#0, [[VAR_24_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[VAR_26_:%.+]] = affine.apply #map0(){{.}}[[VAR_24_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_16_MEM_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_24_1_]]#0, [[VAR_26_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_18_MEM_:%.+]] = krnl.load [[VAR_18_]]{{.}}[[VAR_24_1_]]#0, [[VAR_24_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_29_:%.+]] = arith.addf [[LOAD_VAR_16_MEM_]], [[LOAD_VAR_18_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_8_MEM_:%.+]] = krnl.load [[VAR_8_]]{{.}}[[VAR_24_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_11_MEM_:%.+]] = krnl.load [[VAR_11_]]{{.}}[[VAR_24_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_32_:%.+]] = arith.addf [[VAR_29_]], [[LOAD_VAR_8_MEM_]] : f32
// CHECK:               [[VAR_33_:%.+]] = arith.addf [[VAR_32_]], [[LOAD_VAR_11_MEM_]] : f32
// CHECK:               [[VAR_34_:%.+]] = arith.subf [[VAR_cst_]], [[VAR_33_]] : f32
// CHECK:               [[VAR_35_:%.+]] = math.exp [[VAR_34_]] : f32
// CHECK:               [[VAR_36_:%.+]] = arith.addf [[VAR_35_]], [[VAR_cst_0_]] : f32
// CHECK:               [[VAR_37_:%.+]] = arith.divf [[VAR_cst_0_]], [[VAR_36_]] : f32
// CHECK:               krnl.store [[VAR_37_]], [[RES_3_]]{{.}}[[VAR_24_1_]]#0, [[VAR_24_1_]]#1] : memref<2x4xf32>
// CHECK:               [[VAR_38_:%.+]] = arith.mulf [[VAR_37_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               krnl.store [[VAR_38_]], [[RES_4_]]{{.}}[[VAR_24_1_]]#0, [[VAR_24_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_22_:%.+]] = "onnx.MatMul"([[RES_4_]], [[VAR_6_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_7_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_4_]]#1 -> [[I_8_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[VAR_24_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_24_2_]]#0, [[VAR_24_2_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_16_MEM_1_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_24_2_]]#0, [[VAR_24_2_]]#1] : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_16_MEM_2_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_24_2_]]#0, [[VAR_24_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_18_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_16_MEM_1_]], [[LOAD_VAR_16_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_29_1_:%.+]] = krnl.load [[VAR_7_]]{{.}}[[VAR_24_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_8_MEM_1_:%.+]] = krnl.load [[VAR_10_]]{{.}}[[VAR_24_2_]]#1] : memref<4xf32>
// CHECK:               [[LOAD_VAR_11_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_18_MEM_1_]], [[VAR_29_1_]] : f32
// CHECK:               [[VAR_32_1_:%.+]] = arith.addf [[LOAD_VAR_11_MEM_1_]], [[LOAD_VAR_8_MEM_1_]] : f32
// CHECK:               [[VAR_33_1_:%.+]] = arith.subf [[VAR_cst_]], [[VAR_32_1_]] : f32
// CHECK:               [[VAR_34_1_:%.+]] = math.exp [[VAR_33_1_]] : f32
// CHECK:               [[VAR_35_1_:%.+]] = arith.addf [[VAR_34_1_]], [[VAR_cst_0_]] : f32
// CHECK-DAG:           [[VAR_36_1_:%.+]] = arith.divf [[VAR_cst_0_]], [[VAR_35_1_]] : f32
// CHECK-DAG:           [[VAR_37_1_:%.+]] = affine.apply #map1(){{.}}[[VAR_24_2_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_16_MEM_3_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_24_2_]]#0, [[VAR_37_1_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_22_MEM_:%.+]] = krnl.load [[VAR_22_]]{{.}}[[VAR_24_2_]]#0, [[VAR_24_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_40_:%.+]] = arith.addf [[LOAD_VAR_16_MEM_3_]], [[LOAD_VAR_22_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_9_MEM_:%.+]] = krnl.load [[VAR_9_]]{{.}}[[VAR_24_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_12_MEM_:%.+]] = krnl.load [[VAR_12_]]{{.}}[[VAR_24_2_]]#1] : memref<4xf32>
// CHECK:               [[VAR_43_:%.+]] = arith.addf [[VAR_40_]], [[LOAD_VAR_9_MEM_]] : f32
// CHECK:               [[VAR_44_:%.+]] = arith.addf [[VAR_43_]], [[LOAD_VAR_12_MEM_]] : f32
// CHECK-DAG:           [[VAR_45_:%.+]] = math.tanh [[VAR_44_]] : f32
// CHECK-DAG:           [[VAR_46_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_36_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_47_:%.+]] = arith.mulf [[VAR_46_]], [[VAR_45_]] : f32
// CHECK-DAG:           [[VAR_48_:%.+]] = arith.mulf [[VAR_36_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_49_:%.+]] = arith.addf [[VAR_47_]], [[VAR_48_]] : f32
// CHECK:               krnl.store [[VAR_49_]], [[RES_1_]]{{.}}[[VAR_24_2_]]#0, [[VAR_24_2_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_1_]], [[VAR_c32_i64_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
// CHECK:           return [[RES_]] : memref<1x2x4xf32>
// CHECK:         }
}

// -----

func private @test_gru_reverse_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x12x3xf32>, %arg2: tensor<1x12x4xf32>, %arg3: tensor<1x24xf32>, %arg4: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64, direction = "reverse"} : (tensor<7x2x3xf32>, tensor<1x12x3xf32>, tensor<1x12x4xf32>, tensor<1x24xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>
  
// CHECK-DAG: #map0 = affine_map<(d0) -> (-d0 + 6)>
// CHECK-DAG: #map1 = affine_map<()[s0] -> (s0 + 4)>
// CHECK-DAG: #map2 = affine_map<()[s0] -> (s0 + 8)>
// CHECK-LABEL:  func private @test_gru_reverse_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x12x3xf32>, [[PARAM_2_:%.+]]: memref<1x12x4xf32>, [[PARAM_3_:%.+]]: memref<1x24xf32>, [[PARAM_4_:%.+]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[VAR_c32_i64_:%.+]] = arith.constant 32 : i64
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[UCC_PARAM_3:%.+]] = builtin.unrealized_conversion_cast %arg3 : memref<1x24xf32> to tensor<1x24xf32>
// CHECK-DAG:       [[UCC_PARAM_2:%.+]] = builtin.unrealized_conversion_cast %arg2 : memref<1x12x4xf32> to tensor<1x12x4xf32>
// CHECK-DAG:       [[UCC_PARAM_1:%.+]] = builtin.unrealized_conversion_cast %arg1 : memref<1x12x3xf32> to tensor<1x12x3xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]][[[VAR_c0_]], [[IV]]#0, [[IV]]#1] : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]][[[IV]]#0, [[IV]]#1] : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.SqueezeV11"([[UCC_PARAM_1]]) {axes = [0]} : (tensor<1x12x3xf32>) -> memref<12x3xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.SqueezeV11"([[UCC_PARAM_2]]) {axes = [0]} : (tensor<1x12x4xf32>) -> memref<12x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Transpose"([[VAR_3_]]) {perm = [1, 0]} : (memref<12x3xf32>) -> memref<3x12xf32>
// CHECK-DAG:       [[VAR_6_:%.+]]:3 = "onnx.SplitV11"([[VAR_4_]]) {axis = 0 : si64} : (memref<12x4xf32>) -> (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Transpose"([[VAR_6_]]#0) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Transpose"([[VAR_6_]]#1) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Transpose"([[VAR_6_]]#2) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.SqueezeV11"([[UCC_PARAM_3]]) {axes = [0]} : (tensor<1x24xf32>) -> memref<24xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]]:6 = "onnx.SplitV11"([[VAR_10_]]) {axis = 0 : si64} : (memref<24xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7){
// CHECK:             [[IV:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_1_:%.+]] = affine.apply #map0([[IV]])
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]){
// CHECK:               [[VAR_24_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[LOAD_PARAM_4_MEM_1_]], [[VAR_24_]]#0, [[VAR_24_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_2_]]{{.}}[[VAR_24_]]#0, [[VAR_24_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_16_:%.+]] = "onnx.MatMul"([[RES_2_]], [[VAR_5_]]) : (memref<2x3xf32>, memref<3x12xf32>) -> memref<2x12xf32>
// CHECK-DAG:         [[VAR_17_:%.+]] = "onnx.MatMul"([[RES_1_]], [[VAR_7_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_18_:%.+]] = "onnx.MatMul"([[RES_1_]], [[VAR_8_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[VAR_24_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_24_1_]]#0, [[VAR_24_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[VAR_26_:%.+]] = affine.apply #map1(){{.}}[[VAR_24_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_16_MEM_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_24_1_]]#0, [[VAR_26_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_18_MEM_:%.+]] = krnl.load [[VAR_18_]]{{.}}[[VAR_24_1_]]#0, [[VAR_24_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_29_:%.+]] = arith.addf [[LOAD_VAR_16_MEM_]], [[LOAD_VAR_18_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_:%.+]] = krnl.load [[VAR_11_]]#1{{.}}[[VAR_24_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_11_MEM_1_:%.+]] = krnl.load [[VAR_11_]]#4{{.}}[[VAR_24_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_32_:%.+]] = arith.addf [[VAR_29_]], [[LOAD_VAR_11_MEM_]] : f32
// CHECK:               [[VAR_33_:%.+]] = arith.addf [[VAR_32_]], [[LOAD_VAR_11_MEM_1_]] : f32
// CHECK:               [[VAR_34_:%.+]] = arith.subf [[VAR_cst_]], [[VAR_33_]] : f32
// CHECK:               [[VAR_35_:%.+]] = math.exp [[VAR_34_]] : f32
// CHECK:               [[VAR_36_:%.+]] = arith.addf [[VAR_35_]], [[VAR_cst_0_]] : f32
// CHECK:               [[VAR_37_:%.+]] = arith.divf [[VAR_cst_0_]], [[VAR_36_]] : f32
// CHECK:               krnl.store [[VAR_37_]], [[RES_3_]]{{.}}[[VAR_24_1_]]#0, [[VAR_24_1_]]#1] : memref<2x4xf32>
// CHECK:               [[VAR_38_:%.+]] = arith.mulf [[VAR_37_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               krnl.store [[VAR_38_]], [[RES_4_]]{{.}}[[VAR_24_1_]]#0, [[VAR_24_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_22_:%.+]] = "onnx.MatMul"([[RES_4_]], [[VAR_9_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_7_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_4_]]#1 -> [[I_8_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[VAR_24_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_24_2_]]#0, [[VAR_24_2_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_16_MEM_1_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_24_2_]]#0, [[VAR_24_2_]]#1] : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_16_MEM_2_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_24_2_]]#0, [[VAR_24_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_18_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_16_MEM_1_]], [[LOAD_VAR_16_MEM_2_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_2_:%.+]] = krnl.load [[VAR_11_]]#0{{.}}[[VAR_24_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_11_MEM_3_:%.+]] = krnl.load [[VAR_11_]]#3{{.}}[[VAR_24_2_]]#1] : memref<4xf32>
// CHECK:               [[LOAD_VAR_11_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_18_MEM_1_]], [[LOAD_VAR_11_MEM_2_]] : f32
// CHECK:               [[VAR_32_1_:%.+]] = arith.addf [[LOAD_VAR_11_MEM_1_]], [[LOAD_VAR_11_MEM_3_]] : f32
// CHECK:               [[VAR_33_1_:%.+]] = arith.subf [[VAR_cst_]], [[VAR_32_1_]] : f32
// CHECK:               [[VAR_34_1_:%.+]] = math.exp [[VAR_33_1_]] : f32
// CHECK:               [[VAR_35_1_:%.+]] = arith.addf [[VAR_34_1_]], [[VAR_cst_0_]] : f32
// CHECK-DAG:           [[VAR_36_1_:%.+]] = arith.divf [[VAR_cst_0_]], [[VAR_35_1_]] : f32
// CHECK-DAG:           [[VAR_37_1_:%.+]] = affine.apply #map2(){{.}}[[VAR_24_2_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_16_MEM_3_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_24_2_]]#0, [[VAR_37_1_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_22_MEM_:%.+]] = krnl.load [[VAR_22_]]{{.}}[[VAR_24_2_]]#0, [[VAR_24_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_40_:%.+]] = arith.addf [[LOAD_VAR_16_MEM_3_]], [[LOAD_VAR_22_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_4_:%.+]] = krnl.load [[VAR_11_]]#2{{.}}[[VAR_24_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_11_MEM_5_:%.+]] = krnl.load [[VAR_11_]]#5{{.}}[[VAR_24_2_]]#1] : memref<4xf32>
// CHECK:               [[VAR_43_:%.+]] = arith.addf [[VAR_40_]], [[LOAD_VAR_11_MEM_4_]] : f32
// CHECK:               [[VAR_44_:%.+]] = arith.addf [[VAR_43_]], [[LOAD_VAR_11_MEM_5_]] : f32
// CHECK-DAG:           [[VAR_45_:%.+]] = math.tanh [[VAR_44_]] : f32
// CHECK-DAG:           [[VAR_46_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_36_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_47_:%.+]] = arith.mulf [[VAR_46_]], [[VAR_45_]] : f32
// CHECK-DAG:           [[VAR_48_:%.+]] = arith.mulf [[VAR_36_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_49_:%.+]] = arith.addf [[VAR_47_]], [[VAR_48_]] : f32
// CHECK:               krnl.store [[VAR_49_]], [[RES_1_]]{{.}}[[VAR_24_2_]]#0, [[VAR_24_2_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_1_]], [[VAR_c32_i64_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
// CHECK:           return [[RES_]] : memref<1x2x4xf32>
// CHECK:         }
}

// -----

func private @test_gru_bidirectional_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<2x12x3xf32>, %arg2: tensor<2x12x4xf32>, %arg3: tensor<2x24xf32>, %arg4: tensor<2x2x4xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64, direction = "bidirectional"} : (tensor<7x2x3xf32>, tensor<2x12x3xf32>, tensor<2x12x4xf32>, tensor<2x24xf32>, none, tensor<2x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>
  
// CHECK-DAG: #map0 = affine_map<()[s0] -> (s0 + 4)>
// CHECK-DAG: #map1 = affine_map<()[s0] -> (s0 + 8)>
// CHECK-DAG: #map2 = affine_map<(d0) -> (-d0 + 6)>
// CHECK-LABEL:  func private @test_gru_bidirectional_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<2x12x3xf32>, [[PARAM_2_:%.+]]: memref<2x12x4xf32>, [[PARAM_3_:%.+]]: memref<2x24xf32>, [[PARAM_4_:%.+]]: memref<2x2x4xf32>) -> memref<2x2x4xf32> {
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[UCC_PARAM_3:%.+]] = builtin.unrealized_conversion_cast %arg3 : memref<2x24xf32> to tensor<2x24xf32>
// CHECK-DAG:       [[UCC_PARAM_2:%.+]] = builtin.unrealized_conversion_cast %arg2 : memref<2x12x4xf32> to tensor<2x12x4xf32>
// CHECK-DAG:       [[UCC_PARAM_1:%.+]] = builtin.unrealized_conversion_cast %arg1 : memref<2x12x3xf32> to tensor<2x12x3xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]][[[VAR_c0_]], [[IV]]#0, [[IV]]#1] : memref<2x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]][[[IV]]#0, [[IV]]#1] : memref<2x4xf32>
// CHECK:             [[LOAD_PARAM_4_MEM_1_:%.+]] = krnl.load [[PARAM_4_]][[[VAR_c1_]], [[IV]]#0, [[IV]]#1] : memref<2x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_1_]], [[RES_2_]][[[IV]]#0, [[IV]]#1] : memref<2x4xf32>
// CHECK:           }
// CHECK:           [[VAR_4_:%.+]]:2 = "onnx.SplitV11"([[UCC_PARAM_1]]) {axis = 0 : si64} : (tensor<2x12x3xf32>) -> (memref<1x12x3xf32>, memref<1x12x3xf32>)
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.SqueezeV11"([[VAR_4_]]#0) {axes = [0]} : (memref<1x12x3xf32>) -> memref<12x3xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.SqueezeV11"([[VAR_4_]]#1) {axes = [0]} : (memref<1x12x3xf32>) -> memref<12x3xf32>
// CHECK-DAG:       [[VAR_7_:%.+]]:2 = "onnx.SplitV11"([[UCC_PARAM_2]]) {axis = 0 : si64} : (tensor<2x12x4xf32>) -> (memref<1x12x4xf32>, memref<1x12x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.SqueezeV11"([[VAR_7_]]#0) {axes = [0]} : (memref<1x12x4xf32>) -> memref<12x4xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.SqueezeV11"([[VAR_7_]]#1) {axes = [0]} : (memref<1x12x4xf32>) -> memref<12x4xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Transpose"([[VAR_5_]]) {perm = [1, 0]} : (memref<12x3xf32>) -> memref<3x12xf32>
// CHECK:           [[VAR_11_:%.+]]:3 = "onnx.SplitV11"([[VAR_8_]]) {axis = 0 : si64} : (memref<12x4xf32>) -> (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>)
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Transpose"([[VAR_11_]]#0) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Transpose"([[VAR_11_]]#1) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.Transpose"([[VAR_11_]]#2) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.Transpose"([[VAR_6_]]) {perm = [1, 0]} : (memref<12x3xf32>) -> memref<3x12xf32>
// CHECK-DAG:       [[VAR_16_:%.+]]:3 = "onnx.SplitV11"([[VAR_9_]]) {axis = 0 : si64} : (memref<12x4xf32>) -> (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Transpose"([[VAR_16_]]#0) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Transpose"([[VAR_16_]]#1) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.Transpose"([[VAR_16_]]#2) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_20_:%.+]]:2 = "onnx.SplitV11"([[UCC_PARAM_3]]) {axis = 0 : si64} : (tensor<2x24xf32>) -> (memref<1x24xf32>, memref<1x24xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.SqueezeV11"([[VAR_20_]]#0) {axes = [0]} : (memref<1x24xf32>) -> memref<24xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = "onnx.SqueezeV11"([[VAR_20_]]#1) {axes = [0]} : (memref<1x24xf32>) -> memref<24xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_23_:%.+]]:6 = "onnx.SplitV11"([[VAR_21_]]) {axis = 0 : si64} : (memref<24xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[VAR_24_:%.+]]:6 = "onnx.SplitV11"([[VAR_22_]]) {axis = 0 : si64} : (memref<24xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7){
// CHECK:             [[IV:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]){
// CHECK:               [[VAR_38_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[IV]], [[VAR_38_]]#0, [[VAR_38_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_3_]]{{.}}[[VAR_38_]]#0, [[VAR_38_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_30_:%.+]] = "onnx.MatMul"([[RES_3_]], [[VAR_10_]]) : (memref<2x3xf32>, memref<3x12xf32>) -> memref<2x12xf32>
// CHECK-DAG:         [[VAR_31_:%.+]] = "onnx.MatMul"([[RES_1_]], [[VAR_12_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[VAR_32_:%.+]] = "onnx.MatMul"([[RES_1_]], [[VAR_13_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[VAR_38_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_38_1_]]#0, [[VAR_38_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[VAR_40_:%.+]] = affine.apply #map0(){{.}}[[VAR_38_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_30_MEM_:%.+]] = krnl.load [[VAR_30_]]{{.}}[[VAR_38_1_]]#0, [[VAR_40_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_32_MEM_:%.+]] = krnl.load [[VAR_32_]]{{.}}[[VAR_38_1_]]#0, [[VAR_38_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_43_:%.+]] = arith.addf [[LOAD_VAR_30_MEM_]], [[LOAD_VAR_32_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_23_MEM_:%.+]] = krnl.load [[VAR_23_]]#1{{.}}[[VAR_38_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_23_MEM_1_:%.+]] = krnl.load [[VAR_23_]]#4{{.}}[[VAR_38_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_46_:%.+]] = arith.addf [[VAR_43_]], [[LOAD_VAR_23_MEM_]] : f32
// CHECK:               [[VAR_47_:%.+]] = arith.addf [[VAR_46_]], [[LOAD_VAR_23_MEM_1_]] : f32
// CHECK:               [[VAR_48_:%.+]] = arith.subf [[VAR_cst_]], [[VAR_47_]] : f32
// CHECK:               [[VAR_49_:%.+]] = math.exp [[VAR_48_]] : f32
// CHECK:               [[VAR_50_:%.+]] = arith.addf [[VAR_49_]], [[VAR_cst_0_]] : f32
// CHECK:               [[VAR_51_:%.+]] = arith.divf [[VAR_cst_0_]], [[VAR_50_]] : f32
// CHECK:               krnl.store [[VAR_51_]], [[RES_4_]]{{.}}[[VAR_38_1_]]#0, [[VAR_38_1_]]#1] : memref<2x4xf32>
// CHECK:               [[VAR_52_:%.+]] = arith.mulf [[VAR_51_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               krnl.store [[VAR_52_]], [[RES_5_]]{{.}}[[VAR_38_1_]]#0, [[VAR_38_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_36_:%.+]] = "onnx.MatMul"([[RES_5_]], [[VAR_14_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_7_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_4_]]#1 -> [[I_8_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[VAR_38_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_38_2_]]#0, [[VAR_38_2_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_30_MEM_1_:%.+]] = krnl.load [[VAR_30_]]{{.}}[[VAR_38_2_]]#0, [[VAR_38_2_]]#1] : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_30_MEM_2_:%.+]] = krnl.load [[VAR_31_]]{{.}}[[VAR_38_2_]]#0, [[VAR_38_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_32_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_30_MEM_1_]], [[LOAD_VAR_30_MEM_2_]] : f32
// CHECK-DAG:           [[LOAD_VAR_23_MEM_2_:%.+]] = krnl.load [[VAR_23_]]#0{{.}}[[VAR_38_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_23_MEM_3_:%.+]] = krnl.load [[VAR_23_]]#3{{.}}[[VAR_38_2_]]#1] : memref<4xf32>
// CHECK:               [[LOAD_VAR_23_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_32_MEM_1_]], [[LOAD_VAR_23_MEM_2_]] : f32
// CHECK:               [[VAR_46_1_:%.+]] = arith.addf [[LOAD_VAR_23_MEM_1_]], [[LOAD_VAR_23_MEM_3_]] : f32
// CHECK:               [[VAR_47_1_:%.+]] = arith.subf [[VAR_cst_]], [[VAR_46_1_]] : f32
// CHECK:               [[VAR_48_1_:%.+]] = math.exp [[VAR_47_1_]] : f32
// CHECK:               [[VAR_49_1_:%.+]] = arith.addf [[VAR_48_1_]], [[VAR_cst_0_]] : f32
// CHECK-DAG:           [[VAR_50_1_:%.+]] = arith.divf [[VAR_cst_0_]], [[VAR_49_1_]] : f32
// CHECK-DAG:           [[VAR_51_1_:%.+]] = affine.apply #map1(){{.}}[[VAR_38_2_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_30_MEM_3_:%.+]] = krnl.load [[VAR_30_]]{{.}}[[VAR_38_2_]]#0, [[VAR_51_1_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_36_MEM_:%.+]] = krnl.load [[VAR_36_]]{{.}}[[VAR_38_2_]]#0, [[VAR_38_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_54_:%.+]] = arith.addf [[LOAD_VAR_30_MEM_3_]], [[LOAD_VAR_36_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_23_MEM_4_:%.+]] = krnl.load [[VAR_23_]]#2{{.}}[[VAR_38_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_23_MEM_5_:%.+]] = krnl.load [[VAR_23_]]#5{{.}}[[VAR_38_2_]]#1] : memref<4xf32>
// CHECK:               [[VAR_57_:%.+]] = arith.addf [[VAR_54_]], [[LOAD_VAR_23_MEM_4_]] : f32
// CHECK:               [[VAR_58_:%.+]] = arith.addf [[VAR_57_]], [[LOAD_VAR_23_MEM_5_]] : f32
// CHECK-DAG:           [[VAR_59_:%.+]] = math.tanh [[VAR_58_]] : f32
// CHECK-DAG:           [[VAR_60_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_50_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_61_:%.+]] = arith.mulf [[VAR_60_]], [[VAR_59_]] : f32
// CHECK-DAG:           [[VAR_62_:%.+]] = arith.mulf [[VAR_50_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_63_:%.+]] = arith.addf [[VAR_61_]], [[VAR_62_]] : f32
// CHECK:               krnl.store [[VAR_63_]], [[RES_1_]]{{.}}[[VAR_38_2_]]#0, [[VAR_38_2_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_5_]]) with ([[LOOP_5_]] -> [[I_9_:%.+]] = 0 to 7){
// CHECK:             [[IV:%.+]] = krnl.get_induction_var_value([[LOOP_5_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[RES_3_:%.+]] = affine.apply #map2([[IV]])
// CHECK-DAG:         [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_6_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_6_]]#0, [[LOOP_6_]]#1) with ([[LOOP_6_]]#0 -> [[I_10_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_6_]]#1 -> [[I_11_:%.+]] = [[VAR_c0_]] to [[VAR_c3_]]){
// CHECK:               [[LOAD_PARAM_0_MEM_1_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_6_]]#0, [[LOOP_6_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[RES_3_]], [[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_2_]], [[RES_6_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_31_1_:%.+]] = "onnx.MatMul"([[RES_6_]], [[VAR_15_]]) : (memref<2x3xf32>, memref<3x12xf32>) -> memref<2x12xf32>
// CHECK-DAG:         [[VAR_32_1_:%.+]] = "onnx.MatMul"([[RES_2_]], [[VAR_17_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[RES_4_:%.+]] = "onnx.MatMul"([[RES_2_]], [[VAR_18_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[RES_7_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[LOOP_7_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_7_]]#0, [[LOOP_7_]]#1) with ([[LOOP_7_]]#0 -> [[I_12_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_7_]]#1 -> [[I_13_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[LOAD_PARAM_0_MEM_1_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_7_]]#0, [[LOOP_7_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[RES_2_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_30_MEM_2_:%.+]] = affine.apply #map0(){{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_32_MEM_1_:%.+]] = krnl.load [[VAR_31_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_VAR_30_MEM_2_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_23_MEM_2_:%.+]] = krnl.load [[RES_4_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_23_MEM_3_:%.+]] = arith.addf [[LOAD_VAR_32_MEM_1_]], [[LOAD_VAR_23_MEM_2_]] : f32
// CHECK-DAG:           [[LOAD_VAR_23_MEM_1_1_:%.+]] = krnl.load [[VAR_24_]]#1{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_46_1_:%.+]] = krnl.load [[VAR_24_]]#4{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_47_2_:%.+]] = arith.addf [[LOAD_VAR_23_MEM_3_]], [[LOAD_VAR_23_MEM_1_1_]] : f32
// CHECK:               [[VAR_48_2_:%.+]] = arith.addf [[VAR_47_2_]], [[VAR_46_1_]] : f32
// CHECK:               [[VAR_49_2_:%.+]] = arith.subf [[VAR_cst_]], [[VAR_48_2_]] : f32
// CHECK:               [[VAR_50_2_:%.+]] = math.exp [[VAR_49_2_]] : f32
// CHECK:               [[VAR_51_2_:%.+]] = arith.addf [[VAR_50_2_]], [[VAR_cst_0_]] : f32
// CHECK:               [[VAR_52_1_:%.+]] = arith.divf [[VAR_cst_0_]], [[VAR_51_2_]] : f32
// CHECK:               krnl.store [[VAR_52_1_]], [[RES_7_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK:               [[LOAD_VAR_36_MEM_1_:%.+]] = arith.mulf [[VAR_52_1_]], [[LOAD_PARAM_0_MEM_2_]] : f32
// CHECK:               krnl.store [[LOAD_VAR_36_MEM_1_]], [[RES_8_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK-DAG:         [[LOOP_4_:%.+]] = "onnx.MatMul"([[RES_8_]], [[VAR_19_]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK-DAG:         [[LOOP_8_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_8_]]#0, [[LOOP_8_]]#1) with ([[LOOP_8_]]#0 -> [[I_14_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_8_]]#1 -> [[I_15_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[LOAD_PARAM_0_MEM_1_1_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_8_]]#0, [[LOOP_8_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_2_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_30_MEM_2_1_:%.+]] = krnl.load [[VAR_31_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_32_MEM_1_1_:%.+]] = krnl.load [[VAR_32_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_43_1_:%.+]] = arith.addf [[LOAD_VAR_30_MEM_2_1_]], [[LOAD_VAR_32_MEM_1_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_23_MEM_3_1_:%.+]] = krnl.load [[VAR_24_]]#0{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_23_MEM_1_1_:%.+]] = krnl.load [[VAR_24_]]#3{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_46_2_:%.+]] = arith.addf [[VAR_43_1_]], [[LOAD_VAR_23_MEM_3_1_]] : f32
// CHECK:               [[VAR_47_3_:%.+]] = arith.addf [[VAR_46_2_]], [[LOAD_VAR_23_MEM_1_1_]] : f32
// CHECK:               [[VAR_48_3_:%.+]] = arith.subf [[VAR_cst_]], [[VAR_47_3_]] : f32
// CHECK:               [[VAR_49_3_:%.+]] = math.exp [[VAR_48_3_]] : f32
// CHECK:               [[VAR_50_3_:%.+]] = arith.addf [[VAR_49_3_]], [[VAR_cst_0_]] : f32
// CHECK-DAG:           [[VAR_51_3_:%.+]] = arith.divf [[VAR_cst_0_]], [[VAR_50_3_]] : f32
// CHECK-DAG:           [[VAR_52_2_:%.+]] = affine.apply #map1(){{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_36_MEM_1_:%.+]] = krnl.load [[VAR_31_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#0, [[VAR_52_2_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[VAR_54_1_:%.+]] = krnl.load [[LOOP_4_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_23_MEM_4_:%.+]] = arith.addf [[LOAD_VAR_36_MEM_1_]], [[VAR_54_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_23_MEM_5_:%.+]] = krnl.load [[VAR_24_]]#2{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_57_1_:%.+]] = krnl.load [[VAR_24_]]#5{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_58_1_:%.+]] = arith.addf [[LOAD_VAR_23_MEM_4_]], [[LOAD_VAR_23_MEM_5_]] : f32
// CHECK:               [[VAR_59_1_:%.+]] = arith.addf [[VAR_58_1_]], [[VAR_57_1_]] : f32
// CHECK-DAG:           [[VAR_60_1_:%.+]] = math.tanh [[VAR_59_1_]] : f32
// CHECK-DAG:           [[VAR_61_1_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_51_3_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_62_1_:%.+]] = arith.mulf [[VAR_61_1_]], [[VAR_60_1_]] : f32
// CHECK-DAG:           [[VAR_63_1_:%.+]] = arith.mulf [[VAR_51_3_]], [[LOAD_PARAM_0_MEM_2_1_]] : f32
// CHECK:               [[VAR_64_:%.+]] = arith.addf [[VAR_62_1_]], [[VAR_63_1_]] : f32
// CHECK:               krnl.store [[VAR_64_]], [[RES_2_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOOP_9_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_9_]]#0, [[LOOP_9_]]#1) with ([[LOOP_9_]]#0 -> [[I_16_:%.+]] = [[VAR_c0_]] to [[VAR_c2_]], [[LOOP_9_]]#1 -> [[I_17_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:             [[RES_3_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_9_]]#0, [[LOOP_9_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[RES_6_:%.+]] = krnl.load [[RES_1_]]{{.}}[[RES_3_1_]]#0, [[RES_3_1_]]#1] : memref<2x4xf32>
// CHECK:             krnl.store [[RES_6_]], [[RES_]]{{.}}[[VAR_c0_]], [[RES_3_1_]]#0, [[RES_3_1_]]#1] : memref<2x2x4xf32>
// CHECK:             [[LOOP_6_:%.+]] = krnl.load [[RES_2_]]{{.}}[[RES_3_1_]]#0, [[RES_3_1_]]#1] : memref<2x4xf32>
// CHECK:             krnl.store [[LOOP_6_]], [[RES_]]{{.}}[[VAR_c1_]], [[RES_3_1_]]#0, [[RES_3_1_]]#1] : memref<2x2x4xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x2x4xf32>
// CHECK:         }
}

// -----

func private @test_gru_unknown_dims(%arg0: tensor<?x?x?xf32>, %arg1: tensor<1x12x?xf32>, %arg2: tensor<1x12x4xf32>, %arg3: tensor<1x24xf32>, %arg4: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64} : (tensor<?x?x?xf32>, tensor<1x12x?xf32>, tensor<1x12x4xf32>, tensor<1x24xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-DAG: #map0 = affine_map<(d0) -> (d0)>
// CHECK-DAG: #map1 = affine_map<()[s0] -> (s0 + 4)>
// CHECK-DAG: #map2 = affine_map<()[s0] -> (s0 + 8)>
// CHECK-LABEL:  func private @test_gru_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x?xf32>, [[PARAM_1_:%.+]]: memref<1x12x?xf32>, [[PARAM_2_:%.+]]: memref<1x12x4xf32>, [[PARAM_3_:%.+]]: memref<1x24xf32>, [[PARAM_4_:%.+]]: memref<1x2x4xf32>) -> memref<1x?x4xf32> {
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c16_i64_:%.+]] = arith.constant 16 : i64
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[UCC_PARAM_3:%.+]] = builtin.unrealized_conversion_cast %arg3 : memref<1x24xf32> to tensor<1x24xf32>
// CHECK-DAG:       [[UCC_PARAM_2:%.+]] = builtin.unrealized_conversion_cast %arg2 : memref<1x12x4xf32> to tensor<1x12x4xf32>
// CHECK-DAG:       [[UCC_PARAM_1:%.+]] = builtin.unrealized_conversion_cast %arg1 : memref<1x12x?xf32> to tensor<1x12x?xf32>
// CHECK:           [[VAR_0_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<1x?x4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<?x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[VAR_2_]]) {{.*}}: memref<?x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to #map0([[VAR_2_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]][[[VAR_c0_]], [[IV]]#0, [[IV]]#1] : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]][[[IV]]#0, [[IV]]#1] : memref<?x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.SqueezeV11"([[UCC_PARAM_1]]) {axes = [0]} : (tensor<1x12x?xf32>) -> memref<12x?xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.SqueezeV11"([[UCC_PARAM_2]]) {axes = [0]} : (tensor<1x12x4xf32>) -> memref<12x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Transpose"([[VAR_5_]]) {perm = [1, 0]} : (memref<12x?xf32>) -> memref<?x12xf32>
// CHECK-DAG:       [[VAR_8_:%.+]]:3 = "onnx.SplitV11"([[VAR_6_]]) {axis = 0 : si64} : (memref<12x4xf32>) -> (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Transpose"([[VAR_8_]]#0) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Transpose"([[VAR_8_]]#1) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Transpose"([[VAR_8_]]#2) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.SqueezeV11"([[UCC_PARAM_3]]) {axes = [0]} : (tensor<1x24xf32>) -> memref<24xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_13_:%.+]]:6 = "onnx.SplitV11"([[VAR_12_]]) {axis = 0 : si64} : (memref<24xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK-DAG:       [[VAR_15_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?x?x?xf32>
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to #map0([[VAR_15_]])){
// CHECK:             [[IV:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_1_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<?x?x?xf32>
// CHECK-DAG:         [[VAR_19_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c2_]] : memref<?x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloc([[LOAD_PARAM_4_MEM_1_]], [[VAR_19_]]) {{.*}}: memref<?x?xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[VAR_c0_]] to [[LOAD_PARAM_4_MEM_1_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[VAR_c0_]] to [[VAR_19_]]){
// CHECK:               [[VAR_30_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[IV]], [[VAR_30_]]#0, [[VAR_30_]]#1] : memref<?x?x?xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_2_]]{{.}}[[VAR_30_]]#0, [[VAR_30_]]#1] : memref<?x?xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_22_:%.+]] = "onnx.MatMul"([[RES_2_]], [[VAR_7_]]) : (memref<?x?xf32>, memref<?x12xf32>) -> memref<?x12xf32>
// CHECK-DAG:         [[VAR_23_:%.+]] = "onnx.MatMul"([[RES_1_]], [[VAR_9_]]) : (memref<?x4xf32>, memref<4x4xf32>) -> memref<?x4xf32>
// CHECK-DAG:         [[VAR_24_:%.+]] = "onnx.MatMul"([[RES_1_]], [[VAR_10_]]) : (memref<?x4xf32>, memref<4x4xf32>) -> memref<?x4xf32>
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc([[VAR_2_]]) {{.*}}: memref<?x4xf32>
// CHECK-DAG:         [[RES_4_:%.+]] = memref.alloc([[VAR_2_]]) {{.*}}: memref<?x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[VAR_c0_]] to [[VAR_2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[VAR_30_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_30_1_]]#0, [[VAR_30_1_]]#1] : memref<?x4xf32>
// CHECK-DAG:           [[VAR_32_:%.+]] = affine.apply #map1(){{.}}[[VAR_30_1_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_22_MEM_:%.+]] = krnl.load [[VAR_22_]]{{.}}[[VAR_30_1_]]#0, [[VAR_32_]]{{.}} : memref<?x12xf32>
// CHECK-DAG:           [[LOAD_VAR_24_MEM_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_30_1_]]#0, [[VAR_30_1_]]#1] : memref<?x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_35_:%.+]] = arith.addf [[LOAD_VAR_22_MEM_]], [[LOAD_VAR_24_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_13_MEM_:%.+]] = krnl.load [[VAR_13_]]#1{{.}}[[VAR_30_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_13_MEM_1_:%.+]] = krnl.load [[VAR_13_]]#4{{.}}[[VAR_30_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_38_:%.+]] = arith.addf [[VAR_35_]], [[LOAD_VAR_13_MEM_]] : f32
// CHECK:               [[VAR_39_:%.+]] = arith.addf [[VAR_38_]], [[LOAD_VAR_13_MEM_1_]] : f32
// CHECK:               [[VAR_40_:%.+]] = arith.subf [[VAR_cst_]], [[VAR_39_]] : f32
// CHECK:               [[VAR_41_:%.+]] = math.exp [[VAR_40_]] : f32
// CHECK:               [[VAR_42_:%.+]] = arith.addf [[VAR_41_]], [[VAR_cst_0_]] : f32
// CHECK:               [[VAR_43_:%.+]] = arith.divf [[VAR_cst_0_]], [[VAR_42_]] : f32
// CHECK:               krnl.store [[VAR_43_]], [[RES_3_]]{{.}}[[VAR_30_1_]]#0, [[VAR_30_1_]]#1] : memref<?x4xf32>
// CHECK:               [[VAR_44_:%.+]] = arith.mulf [[VAR_43_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               krnl.store [[VAR_44_]], [[RES_4_]]{{.}}[[VAR_30_1_]]#0, [[VAR_30_1_]]#1] : memref<?x4xf32>
// CHECK:             }
// CHECK-DAG:         [[VAR_28_:%.+]] = "onnx.MatMul"([[RES_4_]], [[VAR_11_]]) : (memref<?x4xf32>, memref<4x4xf32>) -> memref<?x4xf32>
// CHECK-DAG:         [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_7_:%.+]] = [[VAR_c0_]] to [[VAR_2_]], [[LOOP_4_]]#1 -> [[I_8_:%.+]] = [[VAR_c0_]] to [[VAR_c4_]]){
// CHECK:               [[VAR_30_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_30_2_]]#0, [[VAR_30_2_]]#1] : memref<?x4xf32>
// CHECK-DAG:           [[LOAD_VAR_22_MEM_1_:%.+]] = krnl.load [[VAR_22_]]{{.}}[[VAR_30_2_]]#0, [[VAR_30_2_]]#1] : memref<?x12xf32>
// CHECK-DAG:           [[LOAD_VAR_22_MEM_2_:%.+]] = krnl.load [[VAR_23_]]{{.}}[[VAR_30_2_]]#0, [[VAR_30_2_]]#1] : memref<?x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_24_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_22_MEM_1_]], [[LOAD_VAR_22_MEM_2_]] : f32
// CHECK-DAG:           [[LOAD_VAR_13_MEM_2_:%.+]] = krnl.load [[VAR_13_]]#0{{.}}[[VAR_30_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_13_MEM_3_:%.+]] = krnl.load [[VAR_13_]]#3{{.}}[[VAR_30_2_]]#1] : memref<4xf32>
// CHECK:               [[LOAD_VAR_13_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_24_MEM_1_]], [[LOAD_VAR_13_MEM_2_]] : f32
// CHECK:               [[VAR_38_1_:%.+]] = arith.addf [[LOAD_VAR_13_MEM_1_]], [[LOAD_VAR_13_MEM_3_]] : f32
// CHECK:               [[VAR_39_1_:%.+]] = arith.subf [[VAR_cst_]], [[VAR_38_1_]] : f32
// CHECK:               [[VAR_40_1_:%.+]] = math.exp [[VAR_39_1_]] : f32
// CHECK:               [[VAR_41_1_:%.+]] = arith.addf [[VAR_40_1_]], [[VAR_cst_0_]] : f32
// CHECK-DAG:           [[VAR_42_1_:%.+]] = arith.divf [[VAR_cst_0_]], [[VAR_41_1_]] : f32
// CHECK-DAG:           [[VAR_43_1_:%.+]] = affine.apply #map2(){{.}}[[VAR_30_2_]]#1]
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_22_MEM_3_:%.+]] = krnl.load [[VAR_22_]]{{.}}[[VAR_30_2_]]#0, [[VAR_43_1_]]{{.}} : memref<?x12xf32>
// CHECK-DAG:           [[LOAD_VAR_28_MEM_:%.+]] = krnl.load [[VAR_28_]]{{.}}[[VAR_30_2_]]#0, [[VAR_30_2_]]#1] : memref<?x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_46_:%.+]] = arith.addf [[LOAD_VAR_22_MEM_3_]], [[LOAD_VAR_28_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_13_MEM_4_:%.+]] = krnl.load [[VAR_13_]]#2{{.}}[[VAR_30_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_13_MEM_5_:%.+]] = krnl.load [[VAR_13_]]#5{{.}}[[VAR_30_2_]]#1] : memref<4xf32>
// CHECK:               [[VAR_49_:%.+]] = arith.addf [[VAR_46_]], [[LOAD_VAR_13_MEM_4_]] : f32
// CHECK:               [[VAR_50_:%.+]] = arith.addf [[VAR_49_]], [[LOAD_VAR_13_MEM_5_]] : f32
// CHECK-DAG:           [[VAR_51_:%.+]] = math.tanh [[VAR_50_]] : f32
// CHECK-DAG:           [[VAR_52_:%.+]] = arith.subf [[VAR_cst_0_]], [[VAR_42_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_53_:%.+]] = arith.mulf [[VAR_52_]], [[VAR_51_]] : f32
// CHECK-DAG:           [[VAR_54_:%.+]] = arith.mulf [[VAR_42_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_55_:%.+]] = arith.addf [[VAR_53_]], [[VAR_54_]] : f32
// CHECK:               krnl.store [[VAR_55_]], [[RES_1_]]{{.}}[[VAR_30_2_]]#0, [[VAR_30_2_]]#1] : memref<?x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           [[VAR_16_:%.+]] = arith.index_cast [[VAR_2_]] : index to i64
// CHECK:           [[VAR_17_:%.+]] = arith.muli [[VAR_16_]], [[VAR_c16_i64_]] : i64
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_1_]], [[VAR_17_]]) : (memref<1x?x4xf32>, memref<?x4xf32>, i64) -> ()
// CHECK:           return [[RES_]] : memref<1x?x4xf32>
// CHECK:         }
}
