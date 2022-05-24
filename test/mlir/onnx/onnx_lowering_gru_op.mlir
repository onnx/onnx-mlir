// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl='emit-intermediate-ir' --canonicalize %s -split-input-file | FileCheck %s

func @gru_default_backend(%arg0: tensor<1x3x2xf32>, %arg1: tensor<1x15x2xf32>, %arg2: tensor<1x15x5xf32>) -> tensor<1x3x5xf32>  {
    %0 = "onnx.NoValue"() {value} : () -> none
    %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %0, %0, %0) {hidden_size = 5 : si64} : (tensor<1x3x2xf32>, tensor<1x15x2xf32>, tensor<1x15x5xf32>, none, none, none) -> (none, tensor<1x3x5xf32>)
    return %Y_h : tensor<1x3x5xf32>
// CHECK-DAG: #map0 = affine_map<()[s0] -> (s0 + 5)>
// CHECK-DAG: #map1 = affine_map<()[s0] -> (s0 + 10)>
// CHECK-LABEL:  func @gru_default_backend
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x3x2xf32>, [[PARAM_1_:%.+]]: memref<1x15x2xf32>, [[PARAM_2_:%.+]]: memref<1x15x5xf32>) -> memref<1x3x5xf32> {
// CHECK-DAG:       [[VAR_c60_i64_:%.+]] = arith.constant 60 : i64
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
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[RES_1_]]2#0, [[RES_1_]]2#1] : memref<3x5xf32>
// CHECK-DAG:           [[VAR_34_:%.+]] = affine.apply #map0(){{.}}[[VAR_32_1_]]#1]
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
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[RES_1_]]2#0, [[RES_1_]]2#1] : memref<3x5xf32>
// CHECK-DAG:           [[LOAD_VAR_18_MEM_1_:%.+]] = krnl.load [[VAR_18_]]{{.}}[[VAR_32_2_]]#0, [[VAR_32_2_]]#1] : memref<3x15xf32>
// CHECK-DAG:           [[LOAD_VAR_18_MEM_2_:%.+]] = krnl.load [[VAR_21_]]{{.}}[[VAR_32_2_]]#0, [[VAR_32_2_]]#1] : memref<3x5xf32>
// CHECK:               [[LOAD_VAR_24_MEM_1_:%.+]] = arith.addf [[LOAD_VAR_18_MEM_1_]], [[LOAD_VAR_18_MEM_2_]] : f32
// CHECK:               [[VAR_37_1_:%.+]] = arith.subf [[VAR_cst_0_]], [[LOAD_VAR_24_MEM_1_]] : f32
// CHECK:               [[VAR_38_1_:%.+]] = math.exp [[VAR_37_1_]] : f32
// CHECK:               [[VAR_39_1_:%.+]] = arith.addf [[VAR_38_1_]], [[VAR_cst_]] : f32
// CHECK-DAG:           [[VAR_40_1_:%.+]] = arith.divf [[VAR_cst_]], [[VAR_39_1_]] : f32
// CHECK-DAG:           [[VAR_41_1_:%.+]] = affine.apply #map1(){{.}}[[VAR_32_2_]]#1]
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
// CHECK:               krnl.store [[VAR_49_]], [[RES_1_]]{{.}}[[RES_1_]]2#0, [[RES_1_]]2#1] : memref<3x5xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           "krnl.memcpy"([[RES_]], [[RES_1_]], [[VAR_c60_i64_]]) : (memref<1x3x5xf32>, memref<3x5xf32>, i64) -> ()
// CHECK:           return [[RES_]] : memref<1x3x5xf32>
// CHECK:         }
}
