// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl='emit-intermediate-ir' --canonicalize %s -split-input-file | FileCheck %s

func.func private @test_gru_forward_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x12x3xf32>, %arg2: tensor<1x12x4xf32>, %arg3: tensor<1x24xf32>, %arg4: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64} : (tensor<7x2x3xf32>, tensor<1x12x3xf32>, tensor<1x12x4xf32>, tensor<1x24xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>
// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 4)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 8)>
// CHECK-LABEL:  func.func private @test_gru_forward_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x12x3xf32>, [[PARAM_2_:%.+]]: memref<1x12x4xf32>, [[PARAM_3_:%.+]]: memref<1x24xf32>, [[PARAM_4_:%.+]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : i64
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_3_]] : memref<1x24xf32> to tensor<1x24xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_2_]] : memref<1x12x4xf32> to tensor<1x12x4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_1_]] : memref<1x12x3xf32> to tensor<1x12x3xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[VAR_20_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_]], [[VAR_20_]]#0, [[VAR_20_]]#1] : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]]{{.}}[[VAR_20_]]#0, [[VAR_20_]]#1] : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.SqueezeV11"([[VAR_2_]]) <{axes = [0]}> : (tensor<1x12x3xf32>) -> tensor<12x3xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.SqueezeV11"([[VAR_1_]]) <{axes = [0]}> : (tensor<1x12x4xf32>) -> tensor<12x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Transpose"([[VAR_4_]]) <{perm = [1, 0]}> : (tensor<12x3xf32>) -> tensor<3x12xf32>
// CHECK-DAG:       [[VAR_7_:%.+]]:3 = "onnx.SplitV11"([[VAR_5_]]) <{axis = 0 : si64}> : (tensor<12x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Transpose"([[VAR_7_]]#0) <{perm = [1, 0]}> : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Transpose"([[VAR_7_]]#1) <{perm = [1, 0]}> : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Transpose"([[VAR_7_]]#2) <{perm = [1, 0]}> : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.SqueezeV11"([[VAR_0_]]) <{axes = [0]}> : (tensor<1x24xf32>) -> tensor<24xf32>
// CHECK:           [[VAR_12_:%.+]]:6 = "onnx.SplitV11"([[VAR_11_]]) <{axis = 0 : si64}> : (tensor<24xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
// CHECK-DAG:       [[VAR_13_:%.+]] = builtin.unrealized_conversion_cast [[VAR_12_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = builtin.unrealized_conversion_cast [[VAR_12_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = builtin.unrealized_conversion_cast [[VAR_12_]]#2 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = builtin.unrealized_conversion_cast [[VAR_12_]]#3 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = builtin.unrealized_conversion_cast [[VAR_12_]]#4 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = builtin.unrealized_conversion_cast [[VAR_12_]]#5 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7){
// CHECK-DAG:         [[VAR_20_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_]] to [[CST_3_]]){
// CHECK:               [[VAR_36_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_20_1_]], [[VAR_36_]]#0, [[VAR_36_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_2_]]{{.}}[[VAR_36_]]#0, [[VAR_36_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             [[VAR_22_:%.+]] = builtin.unrealized_conversion_cast [[RES_2_]] : memref<2x3xf32> to tensor<2x3xf32>
// CHECK:             [[VAR_23_:%.+]] = "onnx.MatMul"([[VAR_22_]], [[VAR_6_]]) : (tensor<2x3xf32>, tensor<3x12xf32>) -> tensor<2x12xf32>
// CHECK-DAG:         [[VAR_24_:%.+]] = builtin.unrealized_conversion_cast [[VAR_23_]] : tensor<2x12xf32> to memref<2x12xf32>
// CHECK-DAG:         [[VAR_25_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_26_:%.+]] = "onnx.MatMul"([[VAR_25_]], [[VAR_8_]]) : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[VAR_27_:%.+]] = builtin.unrealized_conversion_cast [[VAR_26_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[VAR_28_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_29_:%.+]] = "onnx.MatMul"([[VAR_28_]], [[VAR_9_]]) : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[VAR_30_:%.+]] = builtin.unrealized_conversion_cast [[VAR_29_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_]] to [[CST_4_]]){
// CHECK:               [[VAR_36_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_36_1_]]#0, [[VAR_36_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[VAR_38_:%.+]] = affine.apply [[MAP_0_]]([[VAR_36_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_24_MEM_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_36_1_]]#0, [[VAR_38_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_30_MEM_:%.+]] = krnl.load [[VAR_30_]]{{.}}[[VAR_36_1_]]#0, [[VAR_36_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_41_:%.+]] = arith.addf [[LOAD_VAR_24_MEM_]], [[LOAD_VAR_30_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_14_MEM_:%.+]] = krnl.load [[VAR_14_]]{{.}}[[VAR_36_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_17_MEM_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_36_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_44_:%.+]] = arith.addf [[VAR_41_]], [[LOAD_VAR_14_MEM_]] : f32
// CHECK:               [[VAR_45_:%.+]] = arith.addf [[VAR_44_]], [[LOAD_VAR_17_MEM_]] : f32
// CHECK:               [[VAR_46_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_45_]] : f32
// CHECK:               [[VAR_47_:%.+]] = math.exp [[VAR_46_]] : f32
// CHECK:               [[VAR_48_:%.+]] = arith.addf [[VAR_47_]], [[CST_1_dot_000000_]] : f32
// CHECK:               [[VAR_49_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_48_]] : f32
// CHECK:               krnl.store [[VAR_49_]], [[RES_3_]]{{.}}[[VAR_36_1_]]#0, [[VAR_36_1_]]#1] : memref<2x4xf32>
// CHECK:               [[VAR_50_:%.+]] = arith.mulf [[VAR_49_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               krnl.store [[VAR_50_]], [[RES_4_]]{{.}}[[VAR_36_1_]]#0, [[VAR_36_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             [[VAR_32_:%.+]] = builtin.unrealized_conversion_cast [[RES_4_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_33_:%.+]] = "onnx.MatMul"([[VAR_32_]], [[VAR_10_]]) : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[VAR_34_:%.+]] = builtin.unrealized_conversion_cast [[VAR_33_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_7_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_4_]]#1 -> [[I_8_:%.+]] = [[CST_0_]] to [[CST_4_]]){
// CHECK:               [[VAR_36_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_36_2_]]#0, [[VAR_36_2_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_24_MEM_1_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_36_2_]]#0, [[VAR_36_2_]]#1] : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_24_MEM_2_:%.+]] = krnl.load [[VAR_27_]]{{.}}[[VAR_36_2_]]#0, [[VAR_36_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_30_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_24_MEM_1_]], [[LOAD_VAR_24_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_41_1_:%.+]] = krnl.load [[VAR_13_]]{{.}}[[VAR_36_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_14_MEM_1_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_36_2_]]#1] : memref<4xf32>
// CHECK:               [[LOAD_VAR_17_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_30_MEM_1_]], [[VAR_41_1_]] : f32
// CHECK:               [[VAR_44_1_:%.+]] = arith.addf [[LOAD_VAR_17_MEM_1_]], [[LOAD_VAR_14_MEM_1_]] : f32
// CHECK:               [[VAR_45_1_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_44_1_]] : f32
// CHECK:               [[VAR_46_1_:%.+]] = math.exp [[VAR_45_1_]] : f32
// CHECK:               [[VAR_47_1_:%.+]] = arith.addf [[VAR_46_1_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:           [[VAR_48_1_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_47_1_]] : f32
// CHECK-DAG:           [[VAR_49_1_:%.+]] = affine.apply [[MAP_1_]]([[VAR_36_2_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_24_MEM_3_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_36_2_]]#0, [[VAR_49_1_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_34_MEM_:%.+]] = krnl.load [[VAR_34_]]{{.}}[[VAR_36_2_]]#0, [[VAR_36_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_52_:%.+]] = arith.addf [[LOAD_VAR_24_MEM_3_]], [[LOAD_VAR_34_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_15_MEM_:%.+]] = krnl.load [[VAR_15_]]{{.}}[[VAR_36_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_18_MEM_:%.+]] = krnl.load [[VAR_18_]]{{.}}[[VAR_36_2_]]#1] : memref<4xf32>
// CHECK:               [[VAR_55_:%.+]] = arith.addf [[VAR_52_]], [[LOAD_VAR_15_MEM_]] : f32
// CHECK:               [[VAR_56_:%.+]] = arith.addf [[VAR_55_]], [[LOAD_VAR_18_MEM_]] : f32
// CHECK-DAG:           [[VAR_57_:%.+]] = math.tanh [[VAR_56_]] : f32
// CHECK-DAG:           [[VAR_58_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_48_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_59_:%.+]] = arith.mulf [[VAR_58_]], [[VAR_57_]] : f32
// CHECK-DAG:           [[VAR_60_:%.+]] = arith.mulf [[VAR_48_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_61_:%.+]] = arith.addf [[VAR_59_]], [[VAR_60_]] : f32
// CHECK:               krnl.store [[VAR_61_]], [[RES_1_]]{{.}}[[VAR_36_2_]]#0, [[VAR_36_2_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_]]_1, [[CST_8_]], [[CST_0_]], [[CST_0_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64, index, index) -> ()
// CHECK:           return [[RES_]] : memref<1x2x4xf32>
// CHECK:         }
}

// -----

func.func private @test_gru_forward_mode_linear_before_reset(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x12x3xf32>, %arg2: tensor<1x12x4xf32>, %arg3: tensor<1x24xf32>, %arg4: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64, linear_before_reset = 1 : si64} : (tensor<7x2x3xf32>, tensor<1x12x3xf32>, tensor<1x12x4xf32>, tensor<1x24xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>
// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 4)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 8)>
// CHECK-LABEL:  func.func private @test_gru_forward_mode_linear_before_reset
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x12x3xf32>, [[PARAM_2_:%.+]]: memref<1x12x4xf32>, [[PARAM_3_:%.+]]: memref<1x24xf32>, [[PARAM_4_:%.+]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : i64
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_3_]] : memref<1x24xf32> to tensor<1x24xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_2_]] : memref<1x12x4xf32> to tensor<1x12x4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_1_]] : memref<1x12x3xf32> to tensor<1x12x3xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[VAR_17_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_]], [[VAR_17_]]#0, [[VAR_17_]]#1] : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]]{{.}}[[VAR_17_]]#0, [[VAR_17_]]#1] : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.SqueezeV11"([[VAR_2_]]) <{axes = [0]}> : (tensor<1x12x3xf32>) -> tensor<12x3xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.SqueezeV11"([[VAR_1_]]) <{axes = [0]}> : (tensor<1x12x4xf32>) -> tensor<12x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Transpose"([[VAR_4_]]) <{perm = [1, 0]}> : (tensor<12x3xf32>) -> tensor<3x12xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Transpose"([[VAR_5_]]) <{perm = [1, 0]}> : (tensor<12x4xf32>) -> tensor<4x12xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.SqueezeV11"([[VAR_0_]]) <{axes = [0]}> : (tensor<1x24xf32>) -> tensor<24xf32>
// CHECK:           [[VAR_9_:%.+]]:6 = "onnx.SplitV11"([[VAR_8_]]) <{axis = 0 : si64}> : (tensor<24xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
// CHECK-DAG:       [[VAR_10_:%.+]] = builtin.unrealized_conversion_cast [[VAR_9_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = builtin.unrealized_conversion_cast [[VAR_9_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = builtin.unrealized_conversion_cast [[VAR_9_]]#2 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = builtin.unrealized_conversion_cast [[VAR_9_]]#3 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = builtin.unrealized_conversion_cast [[VAR_9_]]#4 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = builtin.unrealized_conversion_cast [[VAR_9_]]#5 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7){
// CHECK-DAG:         [[VAR_17_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_]] to [[CST_3_]]){
// CHECK:               [[VAR_26_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_17_1_]], [[VAR_26_]]#0, [[VAR_26_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_2_]]{{.}}[[VAR_26_]]#0, [[VAR_26_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             [[VAR_19_:%.+]] = builtin.unrealized_conversion_cast [[RES_2_]] : memref<2x3xf32> to tensor<2x3xf32>
// CHECK:             [[VAR_20_:%.+]] = "onnx.MatMul"([[VAR_19_]], [[VAR_6_]]) : (tensor<2x3xf32>, tensor<3x12xf32>) -> tensor<2x12xf32>
// CHECK-DAG:         [[VAR_21_:%.+]] = builtin.unrealized_conversion_cast [[VAR_20_]] : tensor<2x12xf32> to memref<2x12xf32>
// CHECK-DAG:         [[VAR_22_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_23_:%.+]] = "onnx.MatMul"([[VAR_22_]], [[VAR_7_]]) : (tensor<2x4xf32>, tensor<4x12xf32>) -> tensor<2x12xf32>
// CHECK-DAG:         [[VAR_24_:%.+]] = builtin.unrealized_conversion_cast [[VAR_23_]] : tensor<2x12xf32> to memref<2x12xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_]] to [[CST_4_]]){
// CHECK:               [[VAR_26_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_21_MEM_:%.+]] = krnl.load [[VAR_21_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_24_MEM_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x12xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_30_:%.+]] = arith.addf [[LOAD_VAR_21_MEM_]], [[LOAD_VAR_24_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_10_MEM_:%.+]] = krnl.load [[VAR_10_]]{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_13_MEM_:%.+]] = krnl.load [[VAR_13_]]{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_33_:%.+]] = arith.addf [[VAR_30_]], [[LOAD_VAR_10_MEM_]] : f32
// CHECK:               [[VAR_34_:%.+]] = arith.addf [[VAR_33_]], [[LOAD_VAR_13_MEM_]] : f32
// CHECK:               [[VAR_35_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_34_]] : f32
// CHECK:               [[VAR_36_:%.+]] = math.exp [[VAR_35_]] : f32
// CHECK:               [[VAR_37_:%.+]] = arith.addf [[VAR_36_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:           [[VAR_38_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_37_]] : f32
// CHECK-DAG:           [[VAR_39_:%.+]] = affine.apply [[MAP_0_]]([[VAR_26_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_21_MEM_1_:%.+]] = krnl.load [[VAR_21_]]{{.}}[[VAR_26_1_]]#0, [[VAR_39_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[VAR_41_:%.+]] = affine.apply [[MAP_0_]]([[VAR_26_1_]]#1)
// CHECK:               [[LOAD_VAR_24_MEM_1_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_26_1_]]#0, [[VAR_41_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[VAR_43_:%.+]] = arith.addf [[LOAD_VAR_21_MEM_1_]], [[LOAD_VAR_24_MEM_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_11_MEM_:%.+]] = krnl.load [[VAR_11_]]{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_14_MEM_:%.+]] = krnl.load [[VAR_14_]]{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_46_:%.+]] = arith.addf [[VAR_43_]], [[LOAD_VAR_11_MEM_]] : f32
// CHECK:               [[VAR_47_:%.+]] = arith.addf [[VAR_46_]], [[LOAD_VAR_14_MEM_]] : f32
// CHECK:               [[VAR_48_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_47_]] : f32
// CHECK:               [[VAR_49_:%.+]] = math.exp [[VAR_48_]] : f32
// CHECK:               [[VAR_50_:%.+]] = arith.addf [[VAR_49_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:           [[VAR_51_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_50_]] : f32
// CHECK-DAG:           [[VAR_52_:%.+]] = affine.apply [[MAP_1_]]([[VAR_26_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_21_MEM_2_:%.+]] = krnl.load [[VAR_21_]]{{.}}[[VAR_26_1_]]#0, [[VAR_52_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[VAR_54_:%.+]] = affine.apply [[MAP_1_]]([[VAR_26_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_24_MEM_2_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_26_1_]]#0, [[VAR_54_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_15_MEM_:%.+]] = krnl.load [[VAR_15_]]{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_57_:%.+]] = arith.addf [[LOAD_VAR_24_MEM_2_]], [[LOAD_VAR_15_MEM_]] : f32
// CHECK:               [[VAR_58_:%.+]] = arith.mulf [[VAR_51_]], [[VAR_57_]] : f32
// CHECK-DAG:           [[VAR_59_:%.+]] = arith.addf [[LOAD_VAR_21_MEM_2_]], [[VAR_58_]] : f32
// CHECK-DAG:           [[LOAD_VAR_12_MEM_:%.+]] = krnl.load [[VAR_12_]]{{.}}[[VAR_26_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_61_:%.+]] = arith.addf [[VAR_59_]], [[LOAD_VAR_12_MEM_]] : f32
// CHECK-DAG:           [[VAR_62_:%.+]] = math.tanh [[VAR_61_]] : f32
// CHECK-DAG:           [[VAR_63_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_38_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_64_:%.+]] = arith.mulf [[VAR_63_]], [[VAR_62_]] : f32
// CHECK-DAG:           [[VAR_65_:%.+]] = arith.mulf [[VAR_38_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_66_:%.+]] = arith.addf [[VAR_64_]], [[VAR_65_]] : f32
// CHECK:               krnl.store [[VAR_66_]], [[RES_1_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_]]_1, [[CST_8_]], [[CST_0_]], [[CST_0_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64, index, index) -> ()
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
// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 4)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 8)>
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
// CHECK-DAG:           [[VAR_34_:%.+]] = affine.apply [[MAP_0_]]([[VAR_32_1_]]#1)
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
// CHECK-DAG:           [[VAR_45_1_:%.+]] = affine.apply [[MAP_1_]]([[VAR_32_2_]]#1)
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
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_]]_1, [[CST_8_]], [[CST_0_]], [[CST_0_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64, index, index) -> ()
// CHECK:           return [[RES_]] : memref<1x2x4xf32>
// CHECK:         }
}

// -----

func.func private @test_gru_reverse_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x12x3xf32>, %arg2: tensor<1x12x4xf32>, %arg3: tensor<1x24xf32>, %arg4: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64, direction = "reverse"} : (tensor<7x2x3xf32>, tensor<1x12x3xf32>, tensor<1x12x4xf32>, tensor<1x24xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>
// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (-d0 + 6)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 4)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 + 8)>
// CHECK-LABEL:  func.func private @test_gru_reverse_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<1x12x3xf32>, [[PARAM_2_:%.+]]: memref<1x12x4xf32>, [[PARAM_3_:%.+]]: memref<1x24xf32>, [[PARAM_4_:%.+]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : i64
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_3_]] : memref<1x24xf32> to tensor<1x24xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_2_]] : memref<1x12x4xf32> to tensor<1x12x4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_1_]] : memref<1x12x3xf32> to tensor<1x12x3xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[VAR_20_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_]], [[VAR_20_]]#0, [[VAR_20_]]#1] : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]]{{.}}[[VAR_20_]]#0, [[VAR_20_]]#1] : memref<2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.SqueezeV11"([[VAR_2_]]) <{axes = [0]}> : (tensor<1x12x3xf32>) -> tensor<12x3xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.SqueezeV11"([[VAR_1_]]) <{axes = [0]}> : (tensor<1x12x4xf32>) -> tensor<12x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Transpose"([[VAR_4_]]) <{perm = [1, 0]}> : (tensor<12x3xf32>) -> tensor<3x12xf32>
// CHECK-DAG:       [[VAR_7_:%.+]]:3 = "onnx.SplitV11"([[VAR_5_]]) <{axis = 0 : si64}> : (tensor<12x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Transpose"([[VAR_7_]]#0) <{perm = [1, 0]}> : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Transpose"([[VAR_7_]]#1) <{perm = [1, 0]}> : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Transpose"([[VAR_7_]]#2) <{perm = [1, 0]}> : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.SqueezeV11"([[VAR_0_]]) <{axes = [0]}> : (tensor<1x24xf32>) -> tensor<24xf32>
// CHECK:           [[VAR_12_:%.+]]:6 = "onnx.SplitV11"([[VAR_11_]]) <{axis = 0 : si64}> : (tensor<24xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
// CHECK-DAG:       [[VAR_13_:%.+]] = builtin.unrealized_conversion_cast [[VAR_12_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = builtin.unrealized_conversion_cast [[VAR_12_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = builtin.unrealized_conversion_cast [[VAR_12_]]#2 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = builtin.unrealized_conversion_cast [[VAR_12_]]#3 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = builtin.unrealized_conversion_cast [[VAR_12_]]#4 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = builtin.unrealized_conversion_cast [[VAR_12_]]#5 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7){
// CHECK:             [[VAR_20_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_4_MEM_1_:%.+]] = affine.apply [[MAP_0_]]([[VAR_20_1_]])
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_]] to [[CST_3_]]){
// CHECK:               [[VAR_37_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[LOAD_PARAM_4_MEM_1_]], [[VAR_37_]]#0, [[VAR_37_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_2_]]{{.}}[[VAR_37_]]#0, [[VAR_37_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             [[VAR_23_:%.+]] = builtin.unrealized_conversion_cast [[RES_2_]] : memref<2x3xf32> to tensor<2x3xf32>
// CHECK:             [[VAR_24_:%.+]] = "onnx.MatMul"([[VAR_23_]], [[VAR_6_]]) : (tensor<2x3xf32>, tensor<3x12xf32>) -> tensor<2x12xf32>
// CHECK-DAG:         [[VAR_25_:%.+]] = builtin.unrealized_conversion_cast [[VAR_24_]] : tensor<2x12xf32> to memref<2x12xf32>
// CHECK-DAG:         [[VAR_26_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_27_:%.+]] = "onnx.MatMul"([[VAR_26_]], [[VAR_8_]]) : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[VAR_28_:%.+]] = builtin.unrealized_conversion_cast [[VAR_27_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[VAR_29_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_30_:%.+]] = "onnx.MatMul"([[VAR_29_]], [[VAR_9_]]) : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[VAR_31_:%.+]] = builtin.unrealized_conversion_cast [[VAR_30_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_]] to [[CST_4_]]){
// CHECK:               [[VAR_37_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_37_1_]]#0, [[VAR_37_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[VAR_39_:%.+]] = affine.apply [[MAP_1_]]([[VAR_37_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_25_MEM_:%.+]] = krnl.load [[VAR_25_]]{{.}}[[VAR_37_1_]]#0, [[VAR_39_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_31_MEM_:%.+]] = krnl.load [[VAR_31_]]{{.}}[[VAR_37_1_]]#0, [[VAR_37_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_42_:%.+]] = arith.addf [[LOAD_VAR_25_MEM_]], [[LOAD_VAR_31_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_14_MEM_:%.+]] = krnl.load [[VAR_14_]]{{.}}[[VAR_37_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_17_MEM_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_37_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_45_:%.+]] = arith.addf [[VAR_42_]], [[LOAD_VAR_14_MEM_]] : f32
// CHECK:               [[VAR_46_:%.+]] = arith.addf [[VAR_45_]], [[LOAD_VAR_17_MEM_]] : f32
// CHECK:               [[VAR_47_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_46_]] : f32
// CHECK:               [[VAR_48_:%.+]] = math.exp [[VAR_47_]] : f32
// CHECK:               [[VAR_49_:%.+]] = arith.addf [[VAR_48_]], [[CST_1_dot_000000_]] : f32
// CHECK:               [[VAR_50_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_49_]] : f32
// CHECK:               krnl.store [[VAR_50_]], [[RES_3_]]{{.}}[[VAR_37_1_]]#0, [[VAR_37_1_]]#1] : memref<2x4xf32>
// CHECK:               [[VAR_51_:%.+]] = arith.mulf [[VAR_50_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               krnl.store [[VAR_51_]], [[RES_4_]]{{.}}[[VAR_37_1_]]#0, [[VAR_37_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             [[VAR_33_:%.+]] = builtin.unrealized_conversion_cast [[RES_4_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_34_:%.+]] = "onnx.MatMul"([[VAR_33_]], [[VAR_10_]]) : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[VAR_35_:%.+]] = builtin.unrealized_conversion_cast [[VAR_34_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_7_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_4_]]#1 -> [[I_8_:%.+]] = [[CST_0_]] to [[CST_4_]]){
// CHECK:               [[VAR_37_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_37_2_]]#0, [[VAR_37_2_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_25_MEM_1_:%.+]] = krnl.load [[VAR_25_]]{{.}}[[VAR_37_2_]]#0, [[VAR_37_2_]]#1] : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_25_MEM_2_:%.+]] = krnl.load [[VAR_28_]]{{.}}[[VAR_37_2_]]#0, [[VAR_37_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_31_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_25_MEM_1_]], [[LOAD_VAR_25_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_42_1_:%.+]] = krnl.load [[VAR_13_]]{{.}}[[VAR_37_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_14_MEM_1_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_37_2_]]#1] : memref<4xf32>
// CHECK:               [[LOAD_VAR_17_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_31_MEM_1_]], [[VAR_42_1_]] : f32
// CHECK:               [[VAR_45_1_:%.+]] = arith.addf [[LOAD_VAR_17_MEM_1_]], [[LOAD_VAR_14_MEM_1_]] : f32
// CHECK:               [[VAR_46_1_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_45_1_]] : f32
// CHECK:               [[VAR_47_1_:%.+]] = math.exp [[VAR_46_1_]] : f32
// CHECK:               [[VAR_48_1_:%.+]] = arith.addf [[VAR_47_1_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:           [[VAR_49_1_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_48_1_]] : f32
// CHECK-DAG:           [[VAR_50_1_:%.+]] = affine.apply [[MAP_2_]]([[VAR_37_2_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_25_MEM_3_:%.+]] = krnl.load [[VAR_25_]]{{.}}[[VAR_37_2_]]#0, [[VAR_50_1_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_35_MEM_:%.+]] = krnl.load [[VAR_35_]]{{.}}[[VAR_37_2_]]#0, [[VAR_37_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_53_:%.+]] = arith.addf [[LOAD_VAR_25_MEM_3_]], [[LOAD_VAR_35_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_15_MEM_:%.+]] = krnl.load [[VAR_15_]]{{.}}[[VAR_37_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_18_MEM_:%.+]] = krnl.load [[VAR_18_]]{{.}}[[VAR_37_2_]]#1] : memref<4xf32>
// CHECK:               [[VAR_56_:%.+]] = arith.addf [[VAR_53_]], [[LOAD_VAR_15_MEM_]] : f32
// CHECK:               [[VAR_57_:%.+]] = arith.addf [[VAR_56_]], [[LOAD_VAR_18_MEM_]] : f32
// CHECK-DAG:           [[VAR_58_:%.+]] = math.tanh [[VAR_57_]] : f32
// CHECK-DAG:           [[VAR_59_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_49_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_60_:%.+]] = arith.mulf [[VAR_59_]], [[VAR_58_]] : f32
// CHECK-DAG:           [[VAR_61_:%.+]] = arith.mulf [[VAR_49_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_62_:%.+]] = arith.addf [[VAR_60_]], [[VAR_61_]] : f32
// CHECK:               krnl.store [[VAR_62_]], [[RES_1_]]{{.}}[[VAR_37_2_]]#0, [[VAR_37_2_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_]]_1, [[CST_8_]], [[CST_0_]], [[CST_0_]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64, index, index) -> ()
// CHECK:           return [[RES_]] : memref<1x2x4xf32>
// CHECK:         }
}

// -----

func.func private @test_gru_bidirectional_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<2x12x3xf32>, %arg2: tensor<2x12x4xf32>, %arg3: tensor<2x24xf32>, %arg4: tensor<2x2x4xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64, direction = "bidirectional"} : (tensor<7x2x3xf32>, tensor<2x12x3xf32>, tensor<2x12x4xf32>, tensor<2x24xf32>, none, tensor<2x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>
// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 4)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 8)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (-d0 + 6)>
// CHECK-LABEL:  func.func private @test_gru_bidirectional_mode
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<7x2x3xf32>, [[PARAM_1_:%.+]]: memref<2x12x3xf32>, [[PARAM_2_:%.+]]: memref<2x12x4xf32>, [[PARAM_3_:%.+]]: memref<2x24xf32>, [[PARAM_4_:%.+]]: memref<2x2x4xf32>) -> memref<2x2x4xf32> {
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_3_]] : memref<2x24xf32> to tensor<2x24xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_2_]] : memref<2x12x4xf32> to tensor<2x12x4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_1_]] : memref<2x12x3xf32> to tensor<2x12x3xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x2x4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[VAR_40_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_]], [[VAR_40_]]#0, [[VAR_40_]]#1] : memref<2x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]]{{.}}[[VAR_40_]]#0, [[VAR_40_]]#1] : memref<2x4xf32>
// CHECK:             [[LOAD_PARAM_4_MEM_1_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_1_]], [[VAR_40_]]#0, [[VAR_40_]]#1] : memref<2x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_1_]], [[RES_2_]]{{.}}[[VAR_40_]]#0, [[VAR_40_]]#1] : memref<2x4xf32>
// CHECK:           }
// CHECK:           [[VAR_4_:%.+]]:2 = "onnx.SplitV11"([[VAR_2_]]) <{axis = 0 : si64}> : (tensor<2x12x3xf32>) -> (tensor<1x12x3xf32>, tensor<1x12x3xf32>)
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.SqueezeV11"([[VAR_4_]]#0) <{axes = [0]}> : (tensor<1x12x3xf32>) -> tensor<12x3xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.SqueezeV11"([[VAR_4_]]#1) <{axes = [0]}> : (tensor<1x12x3xf32>) -> tensor<12x3xf32>
// CHECK-DAG:       [[VAR_7_:%.+]]:2 = "onnx.SplitV11"([[VAR_1_]]) <{axis = 0 : si64}> : (tensor<2x12x4xf32>) -> (tensor<1x12x4xf32>, tensor<1x12x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.SqueezeV11"([[VAR_7_]]#0) <{axes = [0]}> : (tensor<1x12x4xf32>) -> tensor<12x4xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.SqueezeV11"([[VAR_7_]]#1) <{axes = [0]}> : (tensor<1x12x4xf32>) -> tensor<12x4xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Transpose"([[VAR_5_]]) <{perm = [1, 0]}> : (tensor<12x3xf32>) -> tensor<3x12xf32>
// CHECK:           [[VAR_11_:%.+]]:3 = "onnx.SplitV11"([[VAR_8_]]) <{axis = 0 : si64}> : (tensor<12x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Transpose"([[VAR_11_]]#0) <{perm = [1, 0]}> : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Transpose"([[VAR_11_]]#1) <{perm = [1, 0]}> : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.Transpose"([[VAR_11_]]#2) <{perm = [1, 0]}> : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.Transpose"([[VAR_6_]]) <{perm = [1, 0]}> : (tensor<12x3xf32>) -> tensor<3x12xf32>
// CHECK-DAG:       [[VAR_16_:%.+]]:3 = "onnx.SplitV11"([[VAR_9_]]) <{axis = 0 : si64}> : (tensor<12x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_17_:%.+]] = "onnx.Transpose"([[VAR_16_]]#0) <{perm = [1, 0]}> : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = "onnx.Transpose"([[VAR_16_]]#1) <{perm = [1, 0]}> : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = "onnx.Transpose"([[VAR_16_]]#2) <{perm = [1, 0]}> : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_20_:%.+]]:2 = "onnx.SplitV11"([[VAR_0_]]) <{axis = 0 : si64}> : (tensor<2x24xf32>) -> (tensor<1x24xf32>, tensor<1x24xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_21_:%.+]] = "onnx.SqueezeV11"([[VAR_20_]]#0) <{axes = [0]}> : (tensor<1x24xf32>) -> tensor<24xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = "onnx.SqueezeV11"([[VAR_20_]]#1) <{axes = [0]}> : (tensor<1x24xf32>) -> tensor<24xf32>
// CHECK:           [[VAR_23_:%.+]]:6 = "onnx.SplitV11"([[VAR_21_]]) <{axis = 0 : si64}> : (tensor<24xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
// CHECK-DAG:       [[VAR_24_:%.+]] = builtin.unrealized_conversion_cast [[VAR_23_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_25_:%.+]] = builtin.unrealized_conversion_cast [[VAR_23_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_26_:%.+]] = builtin.unrealized_conversion_cast [[VAR_23_]]#2 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_27_:%.+]] = builtin.unrealized_conversion_cast [[VAR_23_]]#3 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_28_:%.+]] = builtin.unrealized_conversion_cast [[VAR_23_]]#4 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_29_:%.+]] = builtin.unrealized_conversion_cast [[VAR_23_]]#5 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_30_:%.+]]:6 = "onnx.SplitV11"([[VAR_22_]]) <{axis = 0 : si64}> : (tensor<24xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_31_:%.+]] = builtin.unrealized_conversion_cast [[VAR_30_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_32_:%.+]] = builtin.unrealized_conversion_cast [[VAR_30_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_33_:%.+]] = builtin.unrealized_conversion_cast [[VAR_30_]]#2 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_34_:%.+]] = builtin.unrealized_conversion_cast [[VAR_30_]]#3 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_35_:%.+]] = builtin.unrealized_conversion_cast [[VAR_30_]]#4 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_36_:%.+]] = builtin.unrealized_conversion_cast [[VAR_30_]]#5 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 7){
// CHECK-DAG:         [[VAR_40_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_]] to [[CST_3_]]){
// CHECK:               [[VAR_56_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_40_1_]], [[VAR_56_]]#0, [[VAR_56_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_3_]]{{.}}[[VAR_56_]]#0, [[VAR_56_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             [[LOAD_PARAM_4_MEM_1_:%.+]] = builtin.unrealized_conversion_cast [[RES_3_]] : memref<2x3xf32> to tensor<2x3xf32>
// CHECK:             [[VAR_43_:%.+]] = "onnx.MatMul"([[LOAD_PARAM_4_MEM_1_]], [[VAR_10_]]) : (tensor<2x3xf32>, tensor<3x12xf32>) -> tensor<2x12xf32>
// CHECK-DAG:         [[VAR_44_:%.+]] = builtin.unrealized_conversion_cast [[VAR_43_]] : tensor<2x12xf32> to memref<2x12xf32>
// CHECK-DAG:         [[VAR_45_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_46_:%.+]] = "onnx.MatMul"([[VAR_45_]], [[VAR_12_]]) : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[VAR_47_:%.+]] = builtin.unrealized_conversion_cast [[VAR_46_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[VAR_48_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_49_:%.+]] = "onnx.MatMul"([[VAR_48_]], [[VAR_13_]]) : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[VAR_50_:%.+]] = builtin.unrealized_conversion_cast [[VAR_49_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_]] to [[CST_4_]]){
// CHECK:               [[VAR_56_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_56_1_]]#0, [[VAR_56_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[VAR_58_:%.+]] = affine.apply [[MAP_0_]]([[VAR_56_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_44_MEM_:%.+]] = krnl.load [[VAR_44_]]{{.}}[[VAR_56_1_]]#0, [[VAR_58_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_50_MEM_:%.+]] = krnl.load [[VAR_50_]]{{.}}[[VAR_56_1_]]#0, [[VAR_56_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_61_:%.+]] = arith.addf [[LOAD_VAR_44_MEM_]], [[LOAD_VAR_50_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_25_MEM_:%.+]] = krnl.load [[VAR_25_]]{{.}}[[VAR_56_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_28_MEM_:%.+]] = krnl.load [[VAR_28_]]{{.}}[[VAR_56_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_64_:%.+]] = arith.addf [[VAR_61_]], [[LOAD_VAR_25_MEM_]] : f32
// CHECK:               [[VAR_65_:%.+]] = arith.addf [[VAR_64_]], [[LOAD_VAR_28_MEM_]] : f32
// CHECK:               [[VAR_66_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_65_]] : f32
// CHECK:               [[VAR_67_:%.+]] = math.exp [[VAR_66_]] : f32
// CHECK:               [[VAR_68_:%.+]] = arith.addf [[VAR_67_]], [[CST_1_dot_000000_]] : f32
// CHECK:               [[VAR_69_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_68_]] : f32
// CHECK:               krnl.store [[VAR_69_]], [[RES_4_]]{{.}}[[VAR_56_1_]]#0, [[VAR_56_1_]]#1] : memref<2x4xf32>
// CHECK:               [[VAR_70_:%.+]] = arith.mulf [[VAR_69_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               krnl.store [[VAR_70_]], [[RES_5_]]{{.}}[[VAR_56_1_]]#0, [[VAR_56_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             [[VAR_52_:%.+]] = builtin.unrealized_conversion_cast [[RES_5_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_53_:%.+]] = "onnx.MatMul"([[VAR_52_]], [[VAR_14_]]) : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[VAR_54_:%.+]] = builtin.unrealized_conversion_cast [[VAR_53_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_7_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_4_]]#1 -> [[I_8_:%.+]] = [[CST_0_]] to [[CST_4_]]){
// CHECK:               [[VAR_56_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_56_2_]]#0, [[VAR_56_2_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_44_MEM_1_:%.+]] = krnl.load [[VAR_44_]]{{.}}[[VAR_56_2_]]#0, [[VAR_56_2_]]#1] : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_44_MEM_2_:%.+]] = krnl.load [[VAR_47_]]{{.}}[[VAR_56_2_]]#0, [[VAR_56_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_50_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_44_MEM_1_]], [[LOAD_VAR_44_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_61_1_:%.+]] = krnl.load [[VAR_24_]]{{.}}[[VAR_56_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_25_MEM_1_:%.+]] = krnl.load [[VAR_27_]]{{.}}[[VAR_56_2_]]#1] : memref<4xf32>
// CHECK:               [[LOAD_VAR_28_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_50_MEM_1_]], [[VAR_61_1_]] : f32
// CHECK:               [[VAR_64_1_:%.+]] = arith.addf [[LOAD_VAR_28_MEM_1_]], [[LOAD_VAR_25_MEM_1_]] : f32
// CHECK:               [[VAR_65_1_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_64_1_]] : f32
// CHECK:               [[VAR_66_1_:%.+]] = math.exp [[VAR_65_1_]] : f32
// CHECK:               [[VAR_67_1_:%.+]] = arith.addf [[VAR_66_1_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:           [[VAR_68_1_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_67_1_]] : f32
// CHECK-DAG:           [[VAR_69_1_:%.+]] = affine.apply [[MAP_1_]]([[VAR_56_2_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_44_MEM_3_:%.+]] = krnl.load [[VAR_44_]]{{.}}[[VAR_56_2_]]#0, [[VAR_69_1_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_54_MEM_:%.+]] = krnl.load [[VAR_54_]]{{.}}[[VAR_56_2_]]#0, [[VAR_56_2_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_72_:%.+]] = arith.addf [[LOAD_VAR_44_MEM_3_]], [[LOAD_VAR_54_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_26_MEM_:%.+]] = krnl.load [[VAR_26_]]{{.}}[[VAR_56_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_29_MEM_:%.+]] = krnl.load [[VAR_29_]]{{.}}[[VAR_56_2_]]#1] : memref<4xf32>
// CHECK:               [[VAR_75_:%.+]] = arith.addf [[VAR_72_]], [[LOAD_VAR_26_MEM_]] : f32
// CHECK:               [[VAR_76_:%.+]] = arith.addf [[VAR_75_]], [[LOAD_VAR_29_MEM_]] : f32
// CHECK-DAG:           [[VAR_77_:%.+]] = math.tanh [[VAR_76_]] : f32
// CHECK-DAG:           [[VAR_78_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_68_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_79_:%.+]] = arith.mulf [[VAR_78_]], [[VAR_77_]] : f32
// CHECK-DAG:           [[VAR_80_:%.+]] = arith.mulf [[VAR_68_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_81_:%.+]] = arith.addf [[VAR_79_]], [[VAR_80_]] : f32
// CHECK:               krnl.store [[VAR_81_]], [[RES_1_]]{{.}}[[VAR_56_2_]]#0, [[VAR_56_2_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_5_]]) with ([[LOOP_5_]] -> [[I_9_:%.+]] = 0 to 7){
// CHECK:             [[VAR_40_2_:%.+]] = krnl.get_induction_var_value([[LOOP_5_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOOP_2_:%.+]] = affine.apply [[MAP_2_]]([[VAR_40_2_]])
// CHECK-DAG:         [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:         [[LOOP_6_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_6_]]#0, [[LOOP_6_]]#1) with ([[LOOP_6_]]#0 -> [[I_10_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_6_]]#1 -> [[I_11_:%.+]] = [[CST_0_]] to [[CST_3_]]){
// CHECK:               [[LOAD_PARAM_0_MEM_1_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_6_]]#0, [[LOOP_6_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[LOOP_2_]], [[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_2_]], [[RES_6_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             [[VAR_43_1_:%.+]] = builtin.unrealized_conversion_cast [[RES_6_]] : memref<2x3xf32> to tensor<2x3xf32>
// CHECK:             [[VAR_44_1_:%.+]] = "onnx.MatMul"([[VAR_43_1_]], [[VAR_15_]]) : (tensor<2x3xf32>, tensor<3x12xf32>) -> tensor<2x12xf32>
// CHECK-DAG:         [[VAR_45_1_:%.+]] = builtin.unrealized_conversion_cast [[VAR_44_1_]] : tensor<2x12xf32> to memref<2x12xf32>
// CHECK-DAG:         [[VAR_46_1_:%.+]] = builtin.unrealized_conversion_cast [[RES_2_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_47_1_:%.+]] = "onnx.MatMul"([[VAR_46_1_]], [[VAR_17_]]) : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[VAR_48_1_:%.+]] = builtin.unrealized_conversion_cast [[VAR_47_1_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[VAR_49_1_:%.+]] = builtin.unrealized_conversion_cast [[RES_2_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_50_1_:%.+]] = "onnx.MatMul"([[VAR_49_1_]], [[VAR_18_]]) : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]] = builtin.unrealized_conversion_cast [[VAR_50_1_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[RES_7_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<2x4xf32>
// CHECK-DAG:         [[LOOP_7_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_7_]]#0, [[LOOP_7_]]#1) with ([[LOOP_7_]]#0 -> [[I_12_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_7_]]#1 -> [[I_13_:%.+]] = [[CST_0_]] to [[CST_4_]]){
// CHECK:               [[LOAD_PARAM_0_MEM_1_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_7_]]#0, [[LOOP_7_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[RES_2_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_44_MEM_2_:%.+]] = affine.apply [[MAP_0_]]([[LOAD_PARAM_0_MEM_1_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_50_MEM_1_:%.+]] = krnl.load [[VAR_45_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_VAR_44_MEM_2_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[VAR_61_1_:%.+]] = krnl.load [[LOOP_3_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_25_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_50_MEM_1_]], [[VAR_61_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_28_MEM_1_:%.+]] = krnl.load [[VAR_32_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_64_1_:%.+]] = krnl.load [[VAR_35_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_65_2_:%.+]] = arith.addf [[LOAD_VAR_25_MEM_1_]], [[LOAD_VAR_28_MEM_1_]] : f32
// CHECK:               [[VAR_66_2_:%.+]] = arith.addf [[VAR_65_2_]], [[VAR_64_1_]] : f32
// CHECK:               [[VAR_67_2_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_66_2_]] : f32
// CHECK:               [[VAR_68_2_:%.+]] = math.exp [[VAR_67_2_]] : f32
// CHECK:               [[VAR_69_2_:%.+]] = arith.addf [[VAR_68_2_]], [[CST_1_dot_000000_]] : f32
// CHECK:               [[VAR_70_1_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_69_2_]] : f32
// CHECK:               krnl.store [[VAR_70_1_]], [[RES_7_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK:               [[LOAD_VAR_54_MEM_1_:%.+]] = arith.mulf [[VAR_70_1_]], [[LOAD_PARAM_0_MEM_2_]] : f32
// CHECK:               krnl.store [[LOAD_VAR_54_MEM_1_]], [[RES_8_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             [[VAR_53_1_:%.+]] = builtin.unrealized_conversion_cast [[RES_8_]] : memref<2x4xf32> to tensor<2x4xf32>
// CHECK:             [[VAR_54_1_:%.+]] = "onnx.MatMul"([[VAR_53_1_]], [[VAR_19_]]) : (tensor<2x4xf32>, tensor<4x4xf32>) -> tensor<2x4xf32>
// CHECK-DAG:         [[LOOP_4_:%.+]] = builtin.unrealized_conversion_cast [[VAR_54_1_]] : tensor<2x4xf32> to memref<2x4xf32>
// CHECK-DAG:         [[LOOP_8_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_8_]]#0, [[LOOP_8_]]#1) with ([[LOOP_8_]]#0 -> [[I_14_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_8_]]#1 -> [[I_15_:%.+]] = [[CST_0_]] to [[CST_4_]]){
// CHECK:               [[LOAD_PARAM_0_MEM_1_1_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_8_]]#0, [[LOOP_8_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_2_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<2x4xf32>
// CHECK-DAG:           [[LOAD_VAR_44_MEM_2_1_:%.+]] = krnl.load [[VAR_45_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<2x12xf32>
// CHECK-DAG:           [[LOAD_VAR_50_MEM_1_1_:%.+]] = krnl.load [[VAR_48_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_61_2_:%.+]] = arith.addf [[LOAD_VAR_44_MEM_2_1_]], [[LOAD_VAR_50_MEM_1_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_25_MEM_1_1_:%.+]] = krnl.load [[VAR_31_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_28_MEM_1_1_:%.+]] = krnl.load [[VAR_34_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_64_2_:%.+]] = arith.addf [[VAR_61_2_]], [[LOAD_VAR_25_MEM_1_1_]] : f32
// CHECK:               [[VAR_65_3_:%.+]] = arith.addf [[VAR_64_2_]], [[LOAD_VAR_28_MEM_1_1_]] : f32
// CHECK:               [[VAR_66_3_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_65_3_]] : f32
// CHECK:               [[VAR_67_3_:%.+]] = math.exp [[VAR_66_3_]] : f32
// CHECK:               [[VAR_68_3_:%.+]] = arith.addf [[VAR_67_3_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:           [[VAR_69_3_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_68_3_]] : f32
// CHECK-DAG:           [[VAR_70_2_:%.+]] = affine.apply [[MAP_1_]]([[LOAD_PARAM_0_MEM_1_1_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_54_MEM_1_:%.+]] = krnl.load [[VAR_45_1_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#0, [[VAR_70_2_]]{{.}} : memref<2x12xf32>
// CHECK-DAG:           [[VAR_72_1_:%.+]] = krnl.load [[LOOP_4_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<2x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_26_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_54_MEM_1_]], [[VAR_72_1_]] : f32
// CHECK-DAG:           [[LOAD_VAR_29_MEM_1_:%.+]] = krnl.load [[VAR_33_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[VAR_75_1_:%.+]] = krnl.load [[VAR_36_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_76_1_:%.+]] = arith.addf [[LOAD_VAR_26_MEM_1_]], [[LOAD_VAR_29_MEM_1_]] : f32
// CHECK:               [[VAR_77_1_:%.+]] = arith.addf [[VAR_76_1_]], [[VAR_75_1_]] : f32
// CHECK-DAG:           [[VAR_78_1_:%.+]] = math.tanh [[VAR_77_1_]] : f32
// CHECK-DAG:           [[VAR_79_1_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_69_3_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_80_1_:%.+]] = arith.mulf [[VAR_79_1_]], [[VAR_78_1_]] : f32
// CHECK-DAG:           [[VAR_81_1_:%.+]] = arith.mulf [[VAR_69_3_]], [[LOAD_PARAM_0_MEM_2_1_]] : f32
// CHECK:               [[VAR_82_:%.+]] = arith.addf [[VAR_80_1_]], [[VAR_81_1_]] : f32
// CHECK:               krnl.store [[VAR_82_]], [[RES_2_]]{{.}}[[LOAD_PARAM_0_MEM_1_1_1_]]#0, [[LOAD_PARAM_0_MEM_1_1_1_]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOOP_9_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_9_]]#0, [[LOOP_9_]]#1) with ([[LOOP_9_]]#0 -> [[I_16_:%.+]] = [[CST_0_]] to [[CST_2_]], [[LOOP_9_]]#1 -> [[I_17_:%.+]] = [[CST_0_]] to [[CST_4_]]){
// CHECK:             [[VAR_40_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_9_]]#0, [[LOOP_9_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOOP_2_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_40_3_]]#0, [[VAR_40_3_]]#1] : memref<2x4xf32>
// CHECK:             krnl.store [[LOOP_2_1_]], [[RES_]]{{.}}[[CST_0_]], [[VAR_40_3_]]#0, [[VAR_40_3_]]#1] : memref<2x2x4xf32>
// CHECK:             [[LOOP_6_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_40_3_]]#0, [[VAR_40_3_]]#1] : memref<2x4xf32>
// CHECK:             krnl.store [[LOOP_6_]], [[RES_]]{{.}}[[CST_1_]], [[VAR_40_3_]]#0, [[VAR_40_3_]]#1] : memref<2x2x4xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x2x4xf32>
// CHECK:         }
}

// -----

func.func private @test_gru_unknown_dims(%arg0: tensor<?x?x?xf32>, %arg1: tensor<1x12x?xf32>, %arg2: tensor<1x12x4xf32>, %arg3: tensor<1x24xf32>, %arg4: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64} : (tensor<?x?x?xf32>, tensor<1x12x?xf32>, tensor<1x12x4xf32>, tensor<1x24xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>
// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 4)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 + 8)>
// CHECK-LABEL:  func.func private @test_gru_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x?xf32>, [[PARAM_1_:%.+]]: memref<1x12x?xf32>, [[PARAM_2_:%.+]]: memref<1x12x4xf32>, [[PARAM_3_:%.+]]: memref<1x24xf32>, [[PARAM_4_:%.+]]: memref<1x2x4xf32>) -> memref<1x?x4xf32> {
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : i64
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_4_1_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_3_]] : memref<1x24xf32> to tensor<1x24xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_2_]] : memref<1x12x4xf32> to tensor<1x12x4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_1_]] : memref<1x12x?xf32> to tensor<1x12x?xf32>
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<1x?x4xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[VAR_dim_1_]]) {{.*}}: memref<?x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_1_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[VAR_22_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_]], [[VAR_22_]]#0, [[VAR_22_]]#1] : memref<1x2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_4_MEM_]], [[RES_1_]]{{.}}[[VAR_22_]]#0, [[VAR_22_]]#1] : memref<?x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.SqueezeV11"([[VAR_2_]]) <{axes = [0]}> : (tensor<1x12x?xf32>) -> tensor<12x?xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.SqueezeV11"([[VAR_1_]]) <{axes = [0]}> : (tensor<1x12x4xf32>) -> tensor<12x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Transpose"([[VAR_4_]]) <{perm = [1, 0]}> : (tensor<12x?xf32>) -> tensor<?x12xf32>
// CHECK-DAG:       [[VAR_7_:%.+]]:3 = "onnx.SplitV11"([[VAR_5_]]) <{axis = 0 : si64}> : (tensor<12x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Transpose"([[VAR_7_]]#0) <{perm = [1, 0]}> : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Transpose"([[VAR_7_]]#1) <{perm = [1, 0]}> : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Transpose"([[VAR_7_]]#2) <{perm = [1, 0]}> : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.SqueezeV11"([[VAR_0_]]) <{axes = [0]}> : (tensor<1x24xf32>) -> tensor<24xf32>
// CHECK:           [[VAR_12_:%.+]]:6 = "onnx.SplitV11"([[VAR_11_]]) <{axis = 0 : si64}> : (tensor<24xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>)
// CHECK-DAG:       [[VAR_13_:%.+]] = builtin.unrealized_conversion_cast [[VAR_12_]]#0 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = builtin.unrealized_conversion_cast [[VAR_12_]]#1 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = builtin.unrealized_conversion_cast [[VAR_12_]]#2 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = builtin.unrealized_conversion_cast [[VAR_12_]]#3 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = builtin.unrealized_conversion_cast [[VAR_12_]]#4 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = builtin.unrealized_conversion_cast [[VAR_12_]]#5 : tensor<4xf32> to memref<4xf32>
// CHECK-DAG:       [[VAR_dim_3_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_3_]])){
// CHECK-DAG:         [[VAR_22_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[VAR_dim_4_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x?xf32>
// CHECK-DAG:         [[VAR_dim_5_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<?x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloc([[VAR_dim_4_]], [[VAR_dim_5_]]) {{.*}}: memref<?x?xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_]] to [[VAR_dim_4_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_]] to [[VAR_dim_5_]]){
// CHECK:               [[VAR_38_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_22_1_]], [[VAR_38_]]#0, [[VAR_38_]]#1] : memref<?x?x?xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_2_]]{{.}}[[VAR_38_]]#0, [[VAR_38_]]#1] : memref<?x?xf32>
// CHECK:             }
// CHECK:             [[VAR_24_:%.+]] = builtin.unrealized_conversion_cast [[RES_2_]] : memref<?x?xf32> to tensor<?x?xf32>
// CHECK:             [[VAR_25_:%.+]] = "onnx.MatMul"([[VAR_24_]], [[VAR_6_]]) : (tensor<?x?xf32>, tensor<?x12xf32>) -> tensor<?x12xf32>
// CHECK-DAG:         [[VAR_26_:%.+]] = builtin.unrealized_conversion_cast [[VAR_25_]] : tensor<?x12xf32> to memref<?x12xf32>
// CHECK-DAG:         [[VAR_27_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<?x4xf32> to tensor<?x4xf32>
// CHECK:             [[VAR_28_:%.+]] = "onnx.MatMul"([[VAR_27_]], [[VAR_8_]]) : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
// CHECK-DAG:         [[VAR_29_:%.+]] = builtin.unrealized_conversion_cast [[VAR_28_]] : tensor<?x4xf32> to memref<?x4xf32>
// CHECK-DAG:         [[VAR_30_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<?x4xf32> to tensor<?x4xf32>
// CHECK:             [[VAR_31_:%.+]] = "onnx.MatMul"([[VAR_30_]], [[VAR_9_]]) : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
// CHECK-DAG:         [[VAR_32_:%.+]] = builtin.unrealized_conversion_cast [[VAR_31_]] : tensor<?x4xf32> to memref<?x4xf32>
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc([[VAR_dim_1_]]) {{.*}}: memref<?x4xf32>
// CHECK-DAG:         [[RES_4_:%.+]] = memref.alloc([[VAR_dim_1_]]) {{.*}}: memref<?x4xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_]] to [[VAR_dim_1_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_]] to [[CST_4_1_]]){
// CHECK:               [[VAR_38_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_38_1_]]#0, [[VAR_38_1_]]#1] : memref<?x4xf32>
// CHECK-DAG:           [[VAR_40_:%.+]] = affine.apply [[MAP_1_]]([[VAR_38_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_26_MEM_:%.+]] = krnl.load [[VAR_26_]]{{.}}[[VAR_38_1_]]#0, [[VAR_40_]]{{.}} : memref<?x12xf32>
// CHECK-DAG:           [[LOAD_VAR_32_MEM_:%.+]] = krnl.load [[VAR_32_]]{{.}}[[VAR_38_1_]]#0, [[VAR_38_1_]]#1] : memref<?x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_43_:%.+]] = arith.addf [[LOAD_VAR_26_MEM_]], [[LOAD_VAR_32_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_14_MEM_:%.+]] = krnl.load [[VAR_14_]]{{.}}[[VAR_38_1_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_17_MEM_:%.+]] = krnl.load [[VAR_17_]]{{.}}[[VAR_38_1_]]#1] : memref<4xf32>
// CHECK:               [[VAR_46_:%.+]] = arith.addf [[VAR_43_]], [[LOAD_VAR_14_MEM_]] : f32
// CHECK:               [[VAR_47_:%.+]] = arith.addf [[VAR_46_]], [[LOAD_VAR_17_MEM_]] : f32
// CHECK:               [[VAR_48_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_47_]] : f32
// CHECK:               [[VAR_49_:%.+]] = math.exp [[VAR_48_]] : f32
// CHECK:               [[VAR_50_:%.+]] = arith.addf [[VAR_49_]], [[CST_1_dot_000000_]] : f32
// CHECK:               [[VAR_51_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_50_]] : f32
// CHECK:               krnl.store [[VAR_51_]], [[RES_3_]]{{.}}[[VAR_38_1_]]#0, [[VAR_38_1_]]#1] : memref<?x4xf32>
// CHECK:               [[VAR_52_:%.+]] = arith.mulf [[VAR_51_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               krnl.store [[VAR_52_]], [[RES_4_]]{{.}}[[VAR_38_1_]]#0, [[VAR_38_1_]]#1] : memref<?x4xf32>
// CHECK:             }
// CHECK:             [[VAR_34_:%.+]] = builtin.unrealized_conversion_cast [[RES_4_]] : memref<?x4xf32> to tensor<?x4xf32>
// CHECK:             [[VAR_35_:%.+]] = "onnx.MatMul"([[VAR_34_]], [[VAR_10_]]) : (tensor<?x4xf32>, tensor<4x4xf32>) -> tensor<?x4xf32>
// CHECK-DAG:         [[VAR_36_:%.+]] = builtin.unrealized_conversion_cast [[VAR_35_]] : tensor<?x4xf32> to memref<?x4xf32>
// CHECK-DAG:         [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_7_:%.+]] = [[CST_0_]] to [[VAR_dim_1_]], [[LOOP_4_]]#1 -> [[I_8_:%.+]] = [[CST_0_]] to [[CST_4_1_]]){
// CHECK:               [[VAR_38_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_38_2_]]#0, [[VAR_38_2_]]#1] : memref<?x4xf32>
// CHECK-DAG:           [[LOAD_VAR_26_MEM_1_:%.+]] = krnl.load [[VAR_26_]]{{.}}[[VAR_38_2_]]#0, [[VAR_38_2_]]#1] : memref<?x12xf32>
// CHECK-DAG:           [[LOAD_VAR_26_MEM_2_:%.+]] = krnl.load [[VAR_29_]]{{.}}[[VAR_38_2_]]#0, [[VAR_38_2_]]#1] : memref<?x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_32_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_26_MEM_1_]], [[LOAD_VAR_26_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_43_1_:%.+]] = krnl.load [[VAR_13_]]{{.}}[[VAR_38_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_14_MEM_1_:%.+]] = krnl.load [[VAR_16_]]{{.}}[[VAR_38_2_]]#1] : memref<4xf32>
// CHECK:               [[LOAD_VAR_17_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_32_MEM_1_]], [[VAR_43_1_]] : f32
// CHECK:               [[VAR_46_1_:%.+]] = arith.addf [[LOAD_VAR_17_MEM_1_]], [[LOAD_VAR_14_MEM_1_]] : f32
// CHECK:               [[VAR_47_1_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_46_1_]] : f32
// CHECK:               [[VAR_48_1_:%.+]] = math.exp [[VAR_47_1_]] : f32
// CHECK:               [[VAR_49_1_:%.+]] = arith.addf [[VAR_48_1_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:           [[VAR_50_1_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_49_1_]] : f32
// CHECK-DAG:           [[VAR_51_1_:%.+]] = affine.apply [[MAP_2_]]([[VAR_38_2_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_26_MEM_3_:%.+]] = krnl.load [[VAR_26_]]{{.}}[[VAR_38_2_]]#0, [[VAR_51_1_]]{{.}} : memref<?x12xf32>
// CHECK-DAG:           [[LOAD_VAR_36_MEM_:%.+]] = krnl.load [[VAR_36_]]{{.}}[[VAR_38_2_]]#0, [[VAR_38_2_]]#1] : memref<?x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_54_:%.+]] = arith.addf [[LOAD_VAR_26_MEM_3_]], [[LOAD_VAR_36_MEM_]] : f32
// CHECK-DAG:           [[LOAD_VAR_15_MEM_:%.+]] = krnl.load [[VAR_15_]]{{.}}[[VAR_38_2_]]#1] : memref<4xf32>
// CHECK-DAG:           [[LOAD_VAR_18_MEM_:%.+]] = krnl.load [[VAR_18_]]{{.}}[[VAR_38_2_]]#1] : memref<4xf32>
// CHECK:               [[VAR_57_:%.+]] = arith.addf [[VAR_54_]], [[LOAD_VAR_15_MEM_]] : f32
// CHECK:               [[VAR_58_:%.+]] = arith.addf [[VAR_57_]], [[LOAD_VAR_18_MEM_]] : f32
// CHECK-DAG:           [[VAR_59_:%.+]] = math.tanh [[VAR_58_]] : f32
// CHECK-DAG:           [[VAR_60_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_50_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_61_:%.+]] = arith.mulf [[VAR_60_]], [[VAR_59_]] : f32
// CHECK-DAG:           [[VAR_62_:%.+]] = arith.mulf [[VAR_50_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_63_:%.+]] = arith.addf [[VAR_61_]], [[VAR_62_]] : f32
// CHECK:               krnl.store [[VAR_63_]], [[RES_1_]]{{.}}[[VAR_38_2_]]#0, [[VAR_38_2_]]#1] : memref<?x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           [[VAR_20_:%.+]] = arith.index_cast [[VAR_dim_1_]] : index to i64
// CHECK:           [[VAR_21_:%.+]] = arith.muli [[VAR_20_]], [[CST_4_]] : i64
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_]]_2, [[VAR_21_]], [[CST_0_]], [[CST_0_]]) : (memref<1x?x4xf32>, memref<?x4xf32>, i64, index, index) -> ()
// CHECK:           return [[RES_]] : memref<1x?x4xf32>
// CHECK:         }
}

// -----

func.func @gru_default_backend(%arg0: tensor<1x3x2xf32>, %arg1: tensor<1x15x2xf32>, %arg2: tensor<1x15x5xf32>) -> tensor<1x3x5xf32>  {
    %0 = "onnx.NoValue"() {value} : () -> none
    %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %0, %0, %0) {hidden_size = 5 : si64} : (tensor<1x3x2xf32>, tensor<1x15x2xf32>, tensor<1x15x5xf32>, none, none, none) -> (none, tensor<1x3x5xf32>)
    return %Y_h : tensor<1x3x5xf32>
// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 5)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 10)>
// CHECK-LABEL:  func.func @gru_default_backend
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x3x2xf32>, [[PARAM_1_:%.+]]: memref<1x15x2xf32>, [[PARAM_2_:%.+]]: memref<1x15x5xf32>) -> memref<1x3x5xf32> {
// CHECK-DAG:       [[CST_15_:%.+]] = arith.constant 15 : i64
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_2_]] : memref<1x15x5xf32> to tensor<1x15x5xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_1_]] : memref<1x15x2xf32> to tensor<1x15x2xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x3x5xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<3x5xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 5){
// CHECK:             [[VAR_11_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_1_]]{{.}}[[VAR_11_]]#0, [[VAR_11_]]#1] : memref<3x5xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.SqueezeV11"([[VAR_1_]]) <{axes = [0]}> : (tensor<1x15x2xf32>) -> tensor<15x2xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.SqueezeV11"([[VAR_0_]]) <{axes = [0]}> : (tensor<1x15x5xf32>) -> tensor<15x5xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Transpose"([[VAR_3_]]) <{perm = [1, 0]}> : (tensor<15x2xf32>) -> tensor<2x15xf32>
// CHECK-DAG:       [[VAR_6_:%.+]]:3 = "onnx.SplitV11"([[VAR_4_]]) <{axis = 0 : si64}> : (tensor<15x5xf32>) -> (tensor<5x5xf32>, tensor<5x5xf32>, tensor<5x5xf32>)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Transpose"([[VAR_6_]]#0) <{perm = [1, 0]}> : (tensor<5x5xf32>) -> tensor<5x5xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Transpose"([[VAR_6_]]#1) <{perm = [1, 0]}> : (tensor<5x5xf32>) -> tensor<5x5xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Transpose"([[VAR_6_]]#2) <{perm = [1, 0]}> : (tensor<5x5xf32>) -> tensor<5x5xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 1){
// CHECK-DAG:         [[VAR_11_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<3x2xf32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = [[CST_0_]] to [[CST_3_]], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = [[CST_0_]] to [[CST_2_]]){
// CHECK:               [[VAR_27_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_11_1_]], [[VAR_27_]]#0, [[VAR_27_]]#1] : memref<1x3x2xf32>
// CHECK:               krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_2_]]{{.}}[[VAR_27_]]#0, [[VAR_27_]]#1] : memref<3x2xf32>
// CHECK:             }
// CHECK:             [[VAR_13_:%.+]] = builtin.unrealized_conversion_cast [[RES_2_]] : memref<3x2xf32> to tensor<3x2xf32>
// CHECK:             [[VAR_14_:%.+]] = "onnx.MatMul"([[VAR_13_]], [[VAR_5_]]) : (tensor<3x2xf32>, tensor<2x15xf32>) -> tensor<3x15xf32>
// CHECK-DAG:         [[VAR_15_:%.+]] = builtin.unrealized_conversion_cast [[VAR_14_]] : tensor<3x15xf32> to memref<3x15xf32>
// CHECK-DAG:         [[VAR_16_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<3x5xf32> to tensor<3x5xf32>
// CHECK:             [[VAR_17_:%.+]] = "onnx.MatMul"([[VAR_16_]], [[VAR_7_]]) : (tensor<3x5xf32>, tensor<5x5xf32>) -> tensor<3x5xf32>
// CHECK-DAG:         [[VAR_18_:%.+]] = builtin.unrealized_conversion_cast [[VAR_17_]] : tensor<3x5xf32> to memref<3x5xf32>
// CHECK-DAG:         [[VAR_19_:%.+]] = builtin.unrealized_conversion_cast [[RES_1_]] : memref<3x5xf32> to tensor<3x5xf32>
// CHECK:             [[VAR_20_:%.+]] = "onnx.MatMul"([[VAR_19_]], [[VAR_8_]]) : (tensor<3x5xf32>, tensor<5x5xf32>) -> tensor<3x5xf32>
// CHECK-DAG:         [[VAR_21_:%.+]] = builtin.unrealized_conversion_cast [[VAR_20_]] : tensor<3x5xf32> to memref<3x5xf32>
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<3x5xf32>
// CHECK-DAG:         [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<3x5xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = [[CST_0_]] to [[CST_3_]], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = [[CST_0_]] to [[CST_5_]]){
// CHECK:               [[VAR_27_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_27_1_]]#0, [[VAR_27_1_]]#1] : memref<3x5xf32>
// CHECK-DAG:           [[VAR_29_:%.+]] = affine.apply [[MAP_0_]]([[VAR_27_1_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_15_MEM_:%.+]] = krnl.load [[VAR_15_]]{{.}}[[VAR_27_1_]]#0, [[VAR_29_]]{{.}} : memref<3x15xf32>
// CHECK-DAG:           [[LOAD_VAR_21_MEM_:%.+]] = krnl.load [[VAR_21_]]{{.}}[[VAR_27_1_]]#0, [[VAR_27_1_]]#1] : memref<3x5xf32>
// CHECK:               [[VAR_32_:%.+]] = arith.addf [[LOAD_VAR_15_MEM_]], [[LOAD_VAR_21_MEM_]] : f32
// CHECK:               [[VAR_33_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_32_]] : f32
// CHECK:               [[VAR_34_:%.+]] = math.exp [[VAR_33_]] : f32
// CHECK:               [[VAR_35_:%.+]] = arith.addf [[VAR_34_]], [[CST_1_dot_000000_]] : f32
// CHECK:               [[VAR_36_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_35_]] : f32
// CHECK:               krnl.store [[VAR_36_]], [[RES_3_]]{{.}}[[VAR_27_1_]]#0, [[VAR_27_1_]]#1] : memref<3x5xf32>
// CHECK:               [[VAR_37_:%.+]] = arith.mulf [[VAR_36_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               krnl.store [[VAR_37_]], [[RES_4_]]{{.}}[[VAR_27_1_]]#0, [[VAR_27_1_]]#1] : memref<3x5xf32>
// CHECK:             }
// CHECK:             [[VAR_23_:%.+]] = builtin.unrealized_conversion_cast [[RES_4_]] : memref<3x5xf32> to tensor<3x5xf32>
// CHECK:             [[VAR_24_:%.+]] = "onnx.MatMul"([[VAR_23_]], [[VAR_9_]]) : (tensor<3x5xf32>, tensor<5x5xf32>) -> tensor<3x5xf32>
// CHECK-DAG:         [[VAR_25_:%.+]] = builtin.unrealized_conversion_cast [[VAR_24_]] : tensor<3x5xf32> to memref<3x5xf32>
// CHECK-DAG:         [[LOOP_4_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1) with ([[LOOP_4_]]#0 -> [[I_7_:%.+]] = [[CST_0_]] to [[CST_3_]], [[LOOP_4_]]#1 -> [[I_8_:%.+]] = [[CST_0_]] to [[CST_5_]]){
// CHECK:               [[VAR_27_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_27_2_]]#0, [[VAR_27_2_]]#1] : memref<3x5xf32>
// CHECK-DAG:           [[LOAD_VAR_15_MEM_1_:%.+]] = krnl.load [[VAR_15_]]{{.}}[[VAR_27_2_]]#0, [[VAR_27_2_]]#1] : memref<3x15xf32>
// CHECK-DAG:           [[LOAD_VAR_15_MEM_2_:%.+]] = krnl.load [[VAR_18_]]{{.}}[[VAR_27_2_]]#0, [[VAR_27_2_]]#1] : memref<3x5xf32>
// CHECK:               [[LOAD_VAR_21_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_15_MEM_1_]], [[LOAD_VAR_15_MEM_2_]] : f32
// CHECK:               [[VAR_32_1_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[LOAD_VAR_21_MEM_1_]] : f32
// CHECK:               [[VAR_33_1_:%.+]] = math.exp [[VAR_32_1_]] : f32
// CHECK:               [[VAR_34_1_:%.+]] = arith.addf [[VAR_33_1_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:           [[VAR_35_1_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_34_1_]] : f32
// CHECK-DAG:           [[VAR_36_1_:%.+]] = affine.apply [[MAP_1_]]([[VAR_27_2_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_15_MEM_3_:%.+]] = krnl.load [[VAR_15_]]{{.}}[[VAR_27_2_]]#0, [[VAR_36_1_]]{{.}} : memref<3x15xf32>
// CHECK-DAG:           [[LOAD_VAR_25_MEM_:%.+]] = krnl.load [[VAR_25_]]{{.}}[[VAR_27_2_]]#0, [[VAR_27_2_]]#1] : memref<3x5xf32>
// CHECK:               [[VAR_39_:%.+]] = arith.addf [[LOAD_VAR_15_MEM_3_]], [[LOAD_VAR_25_MEM_]] : f32
// CHECK-DAG:           [[VAR_40_:%.+]] = math.tanh [[VAR_39_]] : f32
// CHECK-DAG:           [[VAR_41_:%.+]] = arith.subf [[CST_1_dot_000000_]], [[VAR_35_1_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_42_:%.+]] = arith.mulf [[VAR_41_]], [[VAR_40_]] : f32
// CHECK-DAG:           [[VAR_43_:%.+]] = arith.mulf [[VAR_35_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:               [[VAR_44_:%.+]] = arith.addf [[VAR_42_]], [[VAR_43_]] : f32
// CHECK:               krnl.store [[VAR_44_]], [[RES_1_]]{{.}}[[VAR_27_2_]]#0, [[VAR_27_2_]]#1] : memref<3x5xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_]]_1, [[CST_15_]], [[CST_0_]], [[CST_0_]]) : (memref<1x3x5xf32>, memref<3x5xf32>, i64, index, index) -> ()
// CHECK:           return [[RES_]] : memref<1x3x5xf32>
// CHECK:         }
}
