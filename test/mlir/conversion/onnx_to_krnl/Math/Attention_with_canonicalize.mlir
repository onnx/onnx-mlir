// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Fixed-size KV cache pattern of onnx.Attention: (Q, K, V, attn_mask = None,
// past_key = None, past_value = None, nonpad_kv_seqlen = <given>). K/V are the
// full padded cache; nonpad_kv_seqlen says how many leading positions per batch
// are valid. attn_mask is synthesized from is_causal/nonpad_kv_seqlen (see
// lowerFixedSizeKVCacheAttention in Attention.cpp).

// -----

func.func @test_attention_fixed_kv_cache(%Q: tensor<1x1x1x2xf32>, %K: tensor<1x1x4x2xf32>, %V: tensor<1x1x4x2xf32>, %nonpad: tensor<1xi64>) -> tensor<1x1x1x2xf32> {
  %none0 = "onnx.NoValue"() : () -> none
  %none1 = "onnx.NoValue"() : () -> none
  %none2 = "onnx.NoValue"() : () -> none
  %Y, %pk, %pv, %qkmm = "onnx.Attention"(%Q, %K, %V, %none0, %none1, %none2, %nonpad) {is_causal = 0 : si64, qk_matmul_output_mode = 0 : si64, scale = 5.000000e-01 : f32, softcap = 0.000000e+00 : f32} : (tensor<1x1x1x2xf32>, tensor<1x1x4x2xf32>, tensor<1x1x4x2xf32>, none, none, none, tensor<1xi64>) -> (tensor<1x1x1x2xf32>, none, none, none)
  return %Y : tensor<1x1x1x2xf32>

// mlir2FileCheck.py -a '["Q","K","V","nonpad"]'
// CHECK-LABEL:  func.func @test_attention_fixed_kv_cache
// CHECK-SAME:   ([[Q_:%.+]]: memref<1x1x1x2xf32>, [[K_:%.+]]: memref<1x1x4x2xf32>, [[V_:%.+]]: memref<1x1x4x2xf32>, [[NONPAD_:%.+]]: memref<1xi64>) -> memref<1x1x1x2xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [1], value = dense<0.000000e+00> : tensor<1xf32>}> : () -> memref<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [1], value = dense<-1.000000e+09> : tensor<1xf32>}> : () -> memref<1xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [4], value = dense<[0, 1, 2, 3]> : tensor<4xi64>}> : () -> memref<4xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[VAR_2_]] to offset: [0], sizes: [1, 1, 1, 4], strides: [4, 4, 4, 1] : memref<4xi64> to memref<1x1x1x4xi64>
// CHECK-DAG:       [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[NONPAD_]] to offset: [0], sizes: [1, 1, 1, 1], strides: [1, 1, 1, 1] : memref<1xi64> to memref<1x1x1x1xi64>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 16 : memref<1x1x1x4xi1>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 1, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 4){
// CHECK:             [[VAR_12_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_MEM_:%.+]] = krnl.load [[VAR_reinterpret_cast_]]{{.}}[[CST_0_1_]], [[CST_0_1_]], [[CST_0_1_]], [[VAR_12_]]#3] : memref<1x1x1x4xi64>
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_1_MEM_:%.+]] = krnl.load [[VAR_reinterpret_cast_1_]]{{.}}[[CST_0_1_]], [[CST_0_1_]], [[CST_0_1_]], [[CST_0_1_]]{{.}} : memref<1x1x1x1xi64>
// CHECK:             [[VAR_15_:%.+]] = arith.cmpi slt, [[LOAD_VAR_reinterpret_cast_MEM_]], [[LOAD_VAR_reinterpret_cast_1_MEM_]] : i64
// CHECK:             krnl.store [[VAR_15_]], [[RES_]]{{.}}[[VAR_12_]]#0, [[VAR_12_]]#1, [[VAR_12_]]#2, [[VAR_12_]]#3] : memref<1x1x1x4xi1>
// CHECK:           }
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() alignment = 16 : memref<1x1x1x4xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2, [[LOOP_1_]]#3) with ([[LOOP_1_]]#0 -> [[I_4_:%.+]] = 0 to 1, [[LOOP_1_]]#1 -> [[I_5_:%.+]] = 0 to 1, [[LOOP_1_]]#2 -> [[I_6_:%.+]] = 0 to 1, [[LOOP_1_]]#3 -> [[I_7_:%.+]] = 0 to 4){
// CHECK:             [[VAR_12_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2, [[LOOP_1_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[CST_0_1_]], [[CST_0_1_]], [[CST_0_1_]], [[VAR_12_1_]]#3] : memref<1x1x1x4xi1>
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_1_MEM_1_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:         [[VAR_15_1_:%.+]] = krnl.load [[VAR_1_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK:             [[VAR_16_:%.+]] = arith.select [[LOAD_VAR_reinterpret_cast_MEM_1_]], [[LOAD_VAR_reinterpret_cast_1_MEM_1_]], [[VAR_15_1_]] : f32
// CHECK:             krnl.store [[VAR_16_]], [[RES_1_]]{{.}}[[VAR_12_1_]]#0, [[VAR_12_1_]]#1, [[VAR_12_1_]]#2, [[VAR_12_1_]]#3] : memref<1x1x1x4xf32>
// CHECK:           }
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() alignment = 16 : memref<1x1x2x4xf32>
// CHECK-DAG:       [[LOOP_2_:%.+]]:4 = krnl.define_loops 4
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_2_]]#3 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.unroll [[BLOCK_IN__0_]] : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2, [[BLOCK_TILE__0_]], [[BLOCK_IN__0_]]) with ([[LOOP_2_]]#0 -> [[I_8_:%.+]] = 0 to 1, [[LOOP_2_]]#1 -> [[I_9_:%.+]] = 0 to 1, [[LOOP_2_]]#2 -> [[I_10_:%.+]] = 0 to 2, [[LOOP_2_]]#3 -> [[I_11_:%.+]] = 0 to 4){
// CHECK:             [[VAR_12_1_:%.+]] = krnl.load [[K_]]{{.}}[[I_8_]], [[I_9_]], [[I_11_]], [[I_10_]]{{.}} : memref<1x1x4x2xf32>
// CHECK:             krnl.store [[VAR_12_1_]], [[RES_2_]]{{.}}[[I_8_]], [[I_9_]], [[I_10_]], [[I_11_]]{{.}} : memref<1x1x2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() alignment = 16 : memref<1x1x1x4xf32>
// CHECK-DAG:       [[LOOP_3_:%.+]]:5 = krnl.define_loops 5
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1, [[LOOP_3_]]#2, [[LOOP_3_]]#3) with ([[LOOP_3_]]#0 -> [[I_12_:%.+]] = 0 to 1, [[LOOP_3_]]#1 -> [[I_13_:%.+]] = 0 to 1, [[LOOP_3_]]#2 -> [[I_14_:%.+]] = 0 to 1, [[LOOP_3_]]#3 -> [[I_15_:%.+]] = 0 to 4, [[LOOP_3_]]#4 -> [[I_16_:%.+]] = 0 to 2){
// CHECK-DAG:         [[VAR_12_2_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1, [[LOOP_3_]]#2, [[LOOP_3_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_MEM_1_:%.+]] = krnl.iterate([[LOOP_3_]]#4) with () iter_args([[VAR_arg9_:%.+]] = [[CST_0_dot_000000_]]) -> (f32){
// CHECK-DAG:           [[LOAD_VAR_reinterpret_cast_1_MEM_1_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]#4) : (!krnl.loop) -> index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_15_1_:%.+]] = krnl.load [[Q_]]{{.}}[[VAR_12_2_]]#0, [[VAR_12_2_]]#1, [[VAR_12_2_]]#2, [[LOAD_VAR_reinterpret_cast_1_MEM_1_]]{{.}} : memref<1x1x1x2xf32>
// CHECK-DAG:           [[VAR_16_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_12_2_]]#0, [[VAR_12_2_]]#1, [[LOAD_VAR_reinterpret_cast_1_MEM_1_]], [[VAR_12_2_]]#3] : memref<1x1x2x4xf32>
// CHECK:               [[VAR_17_:%.+]] = arith.mulf [[VAR_15_1_]], [[VAR_16_1_]] : f32
// CHECK:               [[VAR_18_:%.+]] = arith.addf [[VAR_arg9_]], [[VAR_17_]] : f32
// CHECK:               krnl.yield [[VAR_18_]] : f32
// CHECK:             }
// CHECK:             krnl.store [[LOAD_VAR_reinterpret_cast_MEM_1_]], [[RES_3_]]{{.}}[[VAR_12_2_]]#0, [[VAR_12_2_]]#1, [[VAR_12_2_]]#2, [[VAR_12_2_]]#3] : memref<1x1x1x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_7_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [1], value = dense<5.000000e-01> : tensor<1xf32>}> : () -> memref<1xf32>
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() alignment = 16 : memref<1x1x1x4xf32>
// CHECK-DAG:       [[LOOP_4_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1, [[LOOP_4_]]#2, [[LOOP_4_]]#3) with ([[LOOP_4_]]#0 -> [[I_17_:%.+]] = 0 to 1, [[LOOP_4_]]#1 -> [[I_18_:%.+]] = 0 to 1, [[LOOP_4_]]#2 -> [[I_19_:%.+]] = 0 to 1, [[LOOP_4_]]#3 -> [[I_20_:%.+]] = 0 to 4){
// CHECK:             [[VAR_12_3_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1, [[LOOP_4_]]#2, [[LOOP_4_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_MEM_1_1_:%.+]] = krnl.load [[RES_3_]]{{.}}[[CST_0_1_]], [[CST_0_1_]], [[CST_0_1_]], [[VAR_12_3_]]#3] : memref<1x1x1x4xf32>
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_1_MEM_1_1_:%.+]] = krnl.load [[VAR_7_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK:             [[VAR_15_2_:%.+]] = arith.mulf [[LOAD_VAR_reinterpret_cast_MEM_1_1_]], [[LOAD_VAR_reinterpret_cast_1_MEM_1_1_]] : f32
// CHECK:             krnl.store [[VAR_15_2_]], [[RES_4_]]{{.}}[[VAR_12_3_]]#0, [[VAR_12_3_]]#1, [[VAR_12_3_]]#2, [[VAR_12_3_]]#3] : memref<1x1x1x4xf32>
// CHECK:           }
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() alignment = 16 : memref<1x1x1x4xf32>
// CHECK-DAG:       [[LOOP_5_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_5_]]#0, [[LOOP_5_]]#1, [[LOOP_5_]]#2, [[LOOP_5_]]#3) with ([[LOOP_5_]]#0 -> [[I_21_:%.+]] = 0 to 1, [[LOOP_5_]]#1 -> [[I_22_:%.+]] = 0 to 1, [[LOOP_5_]]#2 -> [[I_23_:%.+]] = 0 to 1, [[LOOP_5_]]#3 -> [[I_24_:%.+]] = 0 to 4){
// CHECK:             [[VAR_12_4_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_5_]]#0, [[LOOP_5_]]#1, [[LOOP_5_]]#2, [[LOOP_5_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_MEM_1_1_:%.+]] = krnl.load [[RES_4_]]{{.}}[[CST_0_1_]], [[CST_0_1_]], [[CST_0_1_]], [[VAR_12_4_]]#3] : memref<1x1x1x4xf32>
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_1_MEM_1_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_0_1_]], [[CST_0_1_]], [[CST_0_1_]], [[VAR_12_4_]]#3] : memref<1x1x1x4xf32>
// CHECK:             [[VAR_15_3_:%.+]] = arith.addf [[LOAD_VAR_reinterpret_cast_MEM_1_1_]], [[LOAD_VAR_reinterpret_cast_1_MEM_1_1_]] : f32
// CHECK:             krnl.store [[VAR_15_3_]], [[RES_5_]]{{.}}[[VAR_12_4_]]#0, [[VAR_12_4_]]#1, [[VAR_12_4_]]#2, [[VAR_12_4_]]#3] : memref<1x1x1x4xf32>
// CHECK:           }
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() alignment = 16 : memref<1x1x1x4xf32>
// CHECK-DAG:       [[LOOP_6_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_6_]]#0, [[LOOP_6_]]#1, [[LOOP_6_]]#2) with ([[LOOP_6_]]#0 -> [[I_25_:%.+]] = 0 to 1, [[LOOP_6_]]#1 -> [[I_26_:%.+]] = 0 to 1, [[LOOP_6_]]#2 -> [[I_27_:%.+]] = 0 to 1){
// CHECK-DAG:         [[VAR_12_5_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_6_]]#0, [[LOOP_6_]]#1, [[LOOP_6_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOOP_7_:%.+]] = krnl.define_loops 1
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_1_MEM_1_1_1_:%.+]] = krnl.iterate([[LOOP_7_]]) with ([[LOOP_7_]] -> [[I_24_:%.+]] = 0 to 4) iter_args([[I_16_:%.+]] = [[CST_0_]]) -> (f32){
// CHECK-DAG:           [[VAR_18_1_:%.+]] = krnl.get_induction_var_value([[LOOP_7_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_RES_5_MEM_:%.+]] = krnl.load [[RES_5_]]{{.}}[[VAR_12_5_]]#0, [[VAR_12_5_]]#1, [[VAR_12_5_]]#2, [[VAR_18_1_]]{{.}} : memref<1x1x1x4xf32>
// CHECK:               [[VAR_20_:%.+]] = arith.maxnumf [[I_16_]], [[LOAD_RES_5_MEM_]] : f32
// CHECK:               krnl.yield [[VAR_20_]] : f32
// CHECK:             }
// CHECK:             [[LOOP_8_:%.+]] = krnl.define_loops 1
// CHECK-DAG:         [[VAR_16_2_:%.+]] = krnl.iterate([[LOOP_8_]]) with ([[LOOP_8_]] -> [[I_24_1_:%.+]] = 0 to 4) iter_args([[I_16_1_:%.+]] = [[CST_0_dot_000000_]]) -> (f32){
// CHECK-DAG:           [[VAR_18_2_:%.+]] = krnl.get_induction_var_value([[LOOP_8_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_RES_5_MEM_1_:%.+]] = krnl.load [[RES_5_]]{{.}}[[VAR_12_5_]]#0, [[VAR_12_5_]]#1, [[VAR_12_5_]]#2, [[VAR_18_2_]]{{.}} : memref<1x1x1x4xf32>
// CHECK:               [[VAR_20_1_:%.+]] = arith.subf [[LOAD_RES_5_MEM_1_]], [[LOAD_VAR_reinterpret_cast_1_MEM_1_1_1_]] : f32
// CHECK:               [[VAR_21_:%.+]] = math.exp [[VAR_20_1_]] : f32
// CHECK:               [[VAR_22_:%.+]] = arith.addf [[I_16_1_]], [[VAR_21_]] : f32
// CHECK:               krnl.store [[VAR_21_]], [[RES_6_]]{{.}}[[VAR_12_5_]]#0, [[VAR_12_5_]]#1, [[VAR_12_5_]]#2, [[VAR_18_2_]]{{.}} : memref<1x1x1x4xf32>
// CHECK:               krnl.yield [[VAR_22_]] : f32
// CHECK:             }
// CHECK:             [[LOOP_9_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_9_]]) with ([[LOOP_9_]] -> [[I_28_:%.+]] = 0 to 4){
// CHECK:               [[VAR_18_3_:%.+]] = krnl.get_induction_var_value([[LOOP_9_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_RES_5_MEM_1_:%.+]] = krnl.load [[RES_6_]]{{.}}[[VAR_12_5_]]#0, [[VAR_12_5_]]#1, [[VAR_12_5_]]#2, [[VAR_18_3_]]{{.}} : memref<1x1x1x4xf32>
// CHECK:               [[VAR_20_2_:%.+]] = arith.divf [[LOAD_RES_5_MEM_1_]], [[VAR_16_2_]] : f32
// CHECK:               krnl.store [[VAR_20_2_]], [[RES_6_]]{{.}}[[VAR_12_5_]]#0, [[VAR_12_5_]]#1, [[VAR_12_5_]]#2, [[VAR_18_3_]]{{.}} : memref<1x1x1x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc() alignment = 16 : memref<1x1x1x2xf32>
// CHECK-DAG:       [[LOOP_10_:%.+]]:5 = krnl.define_loops 5
// CHECK:           krnl.iterate([[LOOP_10_]]#0, [[LOOP_10_]]#1, [[LOOP_10_]]#2, [[LOOP_10_]]#3) with ([[LOOP_10_]]#0 -> [[I_29_:%.+]] = 0 to 1, [[LOOP_10_]]#1 -> [[I_30_:%.+]] = 0 to 1, [[LOOP_10_]]#2 -> [[I_31_:%.+]] = 0 to 1, [[LOOP_10_]]#3 -> [[I_32_:%.+]] = 0 to 2, [[LOOP_10_]]#4 -> [[I_33_:%.+]] = 0 to 4){
// CHECK-DAG:         [[VAR_12_6_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_10_]]#0, [[LOOP_10_]]#1, [[LOOP_10_]]#2, [[LOOP_10_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[LOOP_7_:%.+]] = krnl.iterate([[LOOP_10_]]#4) with () iter_args([[VAR_arg9_1_:%.+]] = [[CST_0_dot_000000_]]) -> (f32){
// CHECK-DAG:           [[LOAD_VAR_reinterpret_cast_1_MEM_1_1_1_:%.+]] = krnl.get_induction_var_value([[LOOP_10_]]#4) : (!krnl.loop) -> index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOOP_8_:%.+]] = krnl.load [[RES_6_]]{{.}}[[VAR_12_6_]]#0, [[VAR_12_6_]]#1, [[VAR_12_6_]]#2, [[LOAD_VAR_reinterpret_cast_1_MEM_1_1_1_]]{{.}} : memref<1x1x1x4xf32>
// CHECK-DAG:           [[VAR_16_2_:%.+]] = krnl.load [[V_]]{{.}}[[VAR_12_6_]]#0, [[VAR_12_6_]]#1, [[LOAD_VAR_reinterpret_cast_1_MEM_1_1_1_]], [[VAR_12_6_]]#3] : memref<1x1x4x2xf32>
// CHECK:               [[VAR_17_1_:%.+]] = arith.mulf [[LOOP_8_]], [[VAR_16_2_]] : f32
// CHECK:               [[VAR_18_4_:%.+]] = arith.addf [[VAR_arg9_1_]], [[VAR_17_1_]] : f32
// CHECK:               krnl.yield [[VAR_18_4_]] : f32
// CHECK:             }
// CHECK:             krnl.store [[LOOP_7_]], [[RES_7_]]{{.}}[[VAR_12_6_]]#0, [[VAR_12_6_]]#1, [[VAR_12_6_]]#2, [[VAR_12_6_]]#3] : memref<1x1x1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_7_]] : memref<1x1x1x2xf32>
// CHECK:         }
}

// -----

func.func @test_attention_fixed_kv_cache_causal(%Q: tensor<1x1x2x2xf32>, %K: tensor<1x1x4x2xf32>, %V: tensor<1x1x4x2xf32>, %nonpad: tensor<1xi64>) -> tensor<1x1x2x2xf32> {
  %none0 = "onnx.NoValue"() : () -> none
  %none1 = "onnx.NoValue"() : () -> none
  %none2 = "onnx.NoValue"() : () -> none
  %Y, %pk, %pv, %qkmm = "onnx.Attention"(%Q, %K, %V, %none0, %none1, %none2, %nonpad) {is_causal = 1 : si64, qk_matmul_output_mode = 0 : si64, scale = 5.000000e-01 : f32, softcap = 0.000000e+00 : f32} : (tensor<1x1x2x2xf32>, tensor<1x1x4x2xf32>, tensor<1x1x4x2xf32>, none, none, none, tensor<1xi64>) -> (tensor<1x1x2x2xf32>, none, none, none)
  return %Y : tensor<1x1x2x2xf32>

// mlir2FileCheck.py -a '["Q","K","V","nonpad"]'
// CHECK-LABEL:  func.func @test_attention_fixed_kv_cache_causal
// CHECK-SAME:   ([[Q_:%.+]]: memref<1x1x2x2xf32>, [[K_:%.+]]: memref<1x1x4x2xf32>, [[V_:%.+]]: memref<1x1x4x2xf32>, [[NONPAD_:%.+]]: memref<1xi64>) -> memref<1x1x2x2xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [1], value = dense<0.000000e+00> : tensor<1xf32>}> : () -> memref<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [1], value = dense<-1.000000e+09> : tensor<1xf32>}> : () -> memref<1xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [4], value = dense<[0, 1, 2, 3]> : tensor<4xi64>}> : () -> memref<4xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[VAR_2_]] to offset: [0], sizes: [1, 1, 1, 4], strides: [4, 4, 4, 1] : memref<4xi64> to memref<1x1x1x4xi64>
// CHECK-DAG:       [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[NONPAD_]] to offset: [0], sizes: [1, 1, 1, 1], strides: [1, 1, 1, 1] : memref<1xi64> to memref<1x1x1x1xi64>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() alignment = 16 : memref<1x1x1x4xi1>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 1, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 4){
// CHECK:             [[VAR_19_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_MEM_:%.+]] = krnl.load [[VAR_reinterpret_cast_]]{{.}}[[CST_0_1_]], [[CST_0_1_]], [[CST_0_1_]], [[VAR_19_]]#3] : memref<1x1x1x4xi64>
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_1_MEM_:%.+]] = krnl.load [[VAR_reinterpret_cast_1_]]{{.}}[[CST_0_1_]], [[CST_0_1_]], [[CST_0_1_]], [[CST_0_1_]]{{.}} : memref<1x1x1x1xi64>
// CHECK:             [[VAR_22_:%.+]] = arith.cmpi slt, [[LOAD_VAR_reinterpret_cast_MEM_]], [[LOAD_VAR_reinterpret_cast_1_MEM_]] : i64
// CHECK:             krnl.store [[VAR_22_]], [[RES_]]{{.}}[[VAR_19_]]#0, [[VAR_19_]]#1, [[VAR_19_]]#2, [[VAR_19_]]#3] : memref<1x1x1x4xi1>
// CHECK:           }
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() alignment = 16 : memref<1x1x1x4xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2, [[LOOP_1_]]#3) with ([[LOOP_1_]]#0 -> [[I_4_:%.+]] = 0 to 1, [[LOOP_1_]]#1 -> [[I_5_:%.+]] = 0 to 1, [[LOOP_1_]]#2 -> [[I_6_:%.+]] = 0 to 1, [[LOOP_1_]]#3 -> [[I_7_:%.+]] = 0 to 4){
// CHECK:             [[VAR_19_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2, [[LOOP_1_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[CST_0_1_]], [[CST_0_1_]], [[CST_0_1_]], [[VAR_19_1_]]#3] : memref<1x1x1x4xi1>
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_1_MEM_1_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:         [[VAR_22_1_:%.+]] = krnl.load [[VAR_1_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK:             [[VAR_23_:%.+]] = arith.select [[LOAD_VAR_reinterpret_cast_MEM_1_]], [[LOAD_VAR_reinterpret_cast_1_MEM_1_]], [[VAR_22_1_]] : f32
// CHECK:             krnl.store [[VAR_23_]], [[RES_1_]]{{.}}[[VAR_19_1_]]#0, [[VAR_19_1_]]#1, [[VAR_19_1_]]#2, [[VAR_19_1_]]#3] : memref<1x1x1x4xf32>
// CHECK:           }
// CHECK:           [[VAR_5_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [2], value = dense<[0, 1]> : tensor<2xi64>}> : () -> memref<2xi64>
// CHECK-DAG:       [[VAR_reinterpret_cast_3_:%.+]] = memref.reinterpret_cast [[VAR_5_]] to offset: [0], sizes: [1, 1, 2, 1], strides: [2, 2, 1, 1] : memref<2xi64> to memref<1x1x2x1xi64>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() alignment = 16 : memref<1x1x2x4xi64>
// CHECK-DAG:       [[LOOP_2_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2, [[LOOP_2_]]#3) with ([[LOOP_2_]]#0 -> [[I_8_:%.+]] = 0 to 1, [[LOOP_2_]]#1 -> [[I_9_:%.+]] = 0 to 1, [[LOOP_2_]]#2 -> [[I_10_:%.+]] = 0 to 2, [[LOOP_2_]]#3 -> [[I_11_:%.+]] = 0 to 4){
// CHECK:             [[VAR_19_2_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2, [[LOOP_2_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_MEM_2_:%.+]] = krnl.load [[VAR_reinterpret_cast_]]{{.}}[[CST_0_1_]], [[CST_0_1_]], [[CST_0_1_]], [[VAR_19_2_]]#3] : memref<1x1x1x4xi64>
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_1_MEM_1_:%.+]] = krnl.load [[VAR_reinterpret_cast_3_]]{{.}}[[CST_0_1_]], [[CST_0_1_]], [[VAR_19_2_]]#2, [[CST_0_1_]]{{.}} : memref<1x1x2x1xi64>
// CHECK:             [[VAR_22_2_:%.+]] = arith.subi [[LOAD_VAR_reinterpret_cast_MEM_2_]], [[LOAD_VAR_reinterpret_cast_1_MEM_1_]] : i64
// CHECK:             krnl.store [[VAR_22_2_]], [[RES_2_]]{{.}}[[VAR_19_2_]]#0, [[VAR_19_2_]]#1, [[VAR_19_2_]]#2, [[VAR_19_2_]]#3] : memref<1x1x2x4xi64>
// CHECK:           }
// CHECK-DAG:       [[VAR_7_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [1], value = dense<2> : tensor<1xi64>}> : () -> memref<1xi64>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() alignment = 16 : memref<1x1x1x1xi64>
// CHECK-DAG:       [[LOOP_3_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1, [[LOOP_3_]]#2, [[LOOP_3_]]#3) with ([[LOOP_3_]]#0 -> [[I_12_:%.+]] = 0 to 1, [[LOOP_3_]]#1 -> [[I_13_:%.+]] = 0 to 1, [[LOOP_3_]]#2 -> [[I_14_:%.+]] = 0 to 1, [[LOOP_3_]]#3 -> [[I_15_:%.+]] = 0 to 1){
// CHECK-DAG:         [[VAR_19_3_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1, [[LOOP_3_]]#2, [[LOOP_3_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_1_MEM_2_:%.+]] = krnl.load [[VAR_reinterpret_cast_1_]]{{.}}[[CST_0_1_]], [[CST_0_1_]], [[CST_0_1_]], [[CST_0_1_]]{{.}} : memref<1x1x1x1xi64>
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_1_MEM_1_1_:%.+]] = krnl.load [[VAR_7_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi64>
// CHECK:             [[VAR_22_3_:%.+]] = arith.subi [[LOAD_VAR_reinterpret_cast_1_MEM_2_]], [[LOAD_VAR_reinterpret_cast_1_MEM_1_1_]] : i64
// CHECK:             krnl.store [[VAR_22_3_]], [[RES_3_]]{{.}}[[VAR_19_3_]]#0, [[VAR_19_3_]]#1, [[VAR_19_3_]]#2, [[VAR_19_3_]]#3] : memref<1x1x1x1xi64>
// CHECK:           }
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() alignment = 16 : memref<1x1x2x4xi1>
// CHECK-DAG:       [[LOOP_4_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_4_]]#0, [[LOOP_4_]]#1, [[LOOP_4_]]#2, [[LOOP_4_]]#3) with ([[LOOP_4_]]#0 -> [[I_16_:%.+]] = 0 to 1, [[LOOP_4_]]#1 -> [[I_17_:%.+]] = 0 to 1, [[LOOP_4_]]#2 -> [[I_18_:%.+]] = 0 to 2, [[LOOP_4_]]#3 -> [[I_19_:%.+]] = 0 to 4){
// CHECK:             [[VAR_19_4_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_4_]]#0, [[LOOP_4_]]#1, [[LOOP_4_]]#2, [[LOOP_4_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_1_MEM_2_:%.+]] = krnl.load [[RES_2_]]{{.}}[[CST_0_1_]], [[CST_0_1_]], [[VAR_19_4_]]#2, [[VAR_19_4_]]#3] : memref<1x1x2x4xi64>
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_1_MEM_1_1_:%.+]] = krnl.load [[RES_3_]]{{.}}[[CST_0_1_]], [[CST_0_1_]], [[CST_0_1_]], [[CST_0_1_]]{{.}} : memref<1x1x1x1xi64>
// CHECK:             [[VAR_22_4_:%.+]] = arith.cmpi sle, [[LOAD_VAR_reinterpret_cast_1_MEM_2_]], [[LOAD_VAR_reinterpret_cast_1_MEM_1_1_]] : i64
// CHECK:             krnl.store [[VAR_22_4_]], [[RES_4_]]{{.}}[[VAR_19_4_]]#0, [[VAR_19_4_]]#1, [[VAR_19_4_]]#2, [[VAR_19_4_]]#3] : memref<1x1x2x4xi1>
// CHECK:           }
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() alignment = 16 : memref<1x1x2x4xf32>
// CHECK-DAG:       [[LOOP_5_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_5_]]#0, [[LOOP_5_]]#1, [[LOOP_5_]]#2, [[LOOP_5_]]#3) with ([[LOOP_5_]]#0 -> [[I_20_:%.+]] = 0 to 1, [[LOOP_5_]]#1 -> [[I_21_:%.+]] = 0 to 1, [[LOOP_5_]]#2 -> [[I_22_:%.+]] = 0 to 2, [[LOOP_5_]]#3 -> [[I_23_:%.+]] = 0 to 4){
// CHECK:             [[VAR_19_5_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_5_]]#0, [[LOOP_5_]]#1, [[LOOP_5_]]#2, [[LOOP_5_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_1_MEM_2_1_:%.+]] = krnl.load [[RES_4_]]{{.}}[[CST_0_1_]], [[CST_0_1_]], [[VAR_19_5_]]#2, [[VAR_19_5_]]#3] : memref<1x1x2x4xi1>
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_1_MEM_1_1_1_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK-DAG:         [[VAR_22_4_:%.+]] = krnl.load [[VAR_1_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK:             [[VAR_23_1_:%.+]] = arith.select [[LOAD_VAR_reinterpret_cast_1_MEM_2_1_]], [[LOAD_VAR_reinterpret_cast_1_MEM_1_1_1_]], [[VAR_22_4_]] : f32
// CHECK:             krnl.store [[VAR_23_1_]], [[RES_5_]]{{.}}[[VAR_19_5_]]#0, [[VAR_19_5_]]#1, [[VAR_19_5_]]#2, [[VAR_19_5_]]#3] : memref<1x1x2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() alignment = 16 : memref<1x1x2x4xf32>
// CHECK-DAG:       [[LOOP_6_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_6_]]#0, [[LOOP_6_]]#1, [[LOOP_6_]]#2, [[LOOP_6_]]#3) with ([[LOOP_6_]]#0 -> [[I_24_:%.+]] = 0 to 1, [[LOOP_6_]]#1 -> [[I_25_:%.+]] = 0 to 1, [[LOOP_6_]]#2 -> [[I_26_:%.+]] = 0 to 2, [[LOOP_6_]]#3 -> [[I_27_:%.+]] = 0 to 4){
// CHECK:             [[VAR_19_6_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_6_]]#0, [[LOOP_6_]]#1, [[LOOP_6_]]#2, [[LOOP_6_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_1_MEM_2_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_0_1_]], [[CST_0_1_]], [[CST_0_1_]], [[VAR_19_6_]]#3] : memref<1x1x1x4xf32>
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_1_MEM_1_1_1_:%.+]] = krnl.load [[RES_5_]]{{.}}[[CST_0_1_]], [[CST_0_1_]], [[VAR_19_6_]]#2, [[VAR_19_6_]]#3] : memref<1x1x2x4xf32>
// CHECK:             [[VAR_22_5_:%.+]] = arith.addf [[LOAD_VAR_reinterpret_cast_1_MEM_2_1_]], [[LOAD_VAR_reinterpret_cast_1_MEM_1_1_1_]] : f32
// CHECK:             krnl.store [[VAR_22_5_]], [[RES_6_]]{{.}}[[VAR_19_6_]]#0, [[VAR_19_6_]]#1, [[VAR_19_6_]]#2, [[VAR_19_6_]]#3] : memref<1x1x2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc() alignment = 16 : memref<1x1x2x4xf32>
// CHECK-DAG:       [[LOOP_7_:%.+]]:4 = krnl.define_loops 4
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_7_]]#3 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.unroll [[BLOCK_IN__0_]] : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_7_]]#0, [[LOOP_7_]]#1, [[LOOP_7_]]#2, [[BLOCK_TILE__0_]], [[BLOCK_IN__0_]]) with ([[LOOP_7_]]#0 -> [[I_28_:%.+]] = 0 to 1, [[LOOP_7_]]#1 -> [[I_29_:%.+]] = 0 to 1, [[LOOP_7_]]#2 -> [[I_30_:%.+]] = 0 to 2, [[LOOP_7_]]#3 -> [[I_31_:%.+]] = 0 to 4){
// CHECK:             [[VAR_19_6_:%.+]] = krnl.load [[K_]]{{.}}[[I_28_]], [[I_29_]], [[I_31_]], [[I_30_]]{{.}} : memref<1x1x4x2xf32>
// CHECK:             krnl.store [[VAR_19_6_]], [[RES_7_]]{{.}}[[I_28_]], [[I_29_]], [[I_30_]], [[I_31_]]{{.}} : memref<1x1x2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[RES_8_:%.+]] = memref.alloc() alignment = 16 : memref<1x1x2x4xf32>
// CHECK-DAG:       [[LOOP_8_:%.+]]:5 = krnl.define_loops 5
// CHECK:           krnl.iterate([[LOOP_8_]]#0, [[LOOP_8_]]#1, [[LOOP_8_]]#2, [[LOOP_8_]]#3) with ([[LOOP_8_]]#0 -> [[I_32_:%.+]] = 0 to 1, [[LOOP_8_]]#1 -> [[I_33_:%.+]] = 0 to 1, [[LOOP_8_]]#2 -> [[I_34_:%.+]] = 0 to 2, [[LOOP_8_]]#3 -> [[I_35_:%.+]] = 0 to 4, [[LOOP_8_]]#4 -> [[I_36_:%.+]] = 0 to 2){
// CHECK-DAG:         [[VAR_19_7_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_8_]]#0, [[LOOP_8_]]#1, [[LOOP_8_]]#2, [[LOOP_8_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_1_MEM_2_1_1_:%.+]] = krnl.iterate([[LOOP_8_]]#4) with () iter_args([[VAR_arg9_:%.+]] = [[CST_0_dot_000000_]]) -> (f32){
// CHECK-DAG:           [[LOAD_VAR_reinterpret_cast_1_MEM_1_1_1_1_:%.+]] = krnl.get_induction_var_value([[LOOP_8_]]#4) : (!krnl.loop) -> index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_22_5_:%.+]] = krnl.load [[Q_]]{{.}}[[VAR_19_7_]]#0, [[VAR_19_7_]]#1, [[VAR_19_7_]]#2, [[LOAD_VAR_reinterpret_cast_1_MEM_1_1_1_1_]]{{.}} : memref<1x1x2x2xf32>
// CHECK-DAG:           [[VAR_23_1_:%.+]] = krnl.load [[RES_7_]]{{.}}[[VAR_19_7_]]#0, [[VAR_19_7_]]#1, [[LOAD_VAR_reinterpret_cast_1_MEM_1_1_1_1_]], [[VAR_19_7_]]#3] : memref<1x1x2x4xf32>
// CHECK:               [[VAR_24_:%.+]] = arith.mulf [[VAR_22_5_]], [[VAR_23_1_]] : f32
// CHECK:               [[VAR_25_:%.+]] = arith.addf [[VAR_arg9_]], [[VAR_24_]] : f32
// CHECK:               krnl.yield [[VAR_25_]] : f32
// CHECK:             }
// CHECK:             krnl.store [[LOAD_VAR_reinterpret_cast_1_MEM_2_1_1_]], [[RES_8_]]{{.}}[[VAR_19_7_]]#0, [[VAR_19_7_]]#1, [[VAR_19_7_]]#2, [[VAR_19_7_]]#3] : memref<1x1x2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_14_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [1], value = dense<5.000000e-01> : tensor<1xf32>}> : () -> memref<1xf32>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc() alignment = 16 : memref<1x1x2x4xf32>
// CHECK-DAG:       [[LOOP_9_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_9_]]#0, [[LOOP_9_]]#1, [[LOOP_9_]]#2, [[LOOP_9_]]#3) with ([[LOOP_9_]]#0 -> [[I_37_:%.+]] = 0 to 1, [[LOOP_9_]]#1 -> [[I_38_:%.+]] = 0 to 1, [[LOOP_9_]]#2 -> [[I_39_:%.+]] = 0 to 2, [[LOOP_9_]]#3 -> [[I_40_:%.+]] = 0 to 4){
// CHECK:             [[VAR_19_8_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_9_]]#0, [[LOOP_9_]]#1, [[LOOP_9_]]#2, [[LOOP_9_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_1_MEM_2_1_1_:%.+]] = krnl.load [[RES_8_]]{{.}}[[CST_0_1_]], [[CST_0_1_]], [[VAR_19_8_]]#2, [[VAR_19_8_]]#3] : memref<1x1x2x4xf32>
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_1_MEM_1_1_1_1_:%.+]] = krnl.load [[VAR_14_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xf32>
// CHECK:             [[VAR_22_6_:%.+]] = arith.mulf [[LOAD_VAR_reinterpret_cast_1_MEM_2_1_1_]], [[LOAD_VAR_reinterpret_cast_1_MEM_1_1_1_1_]] : f32
// CHECK:             krnl.store [[VAR_22_6_]], [[RES_9_]]{{.}}[[VAR_19_8_]]#0, [[VAR_19_8_]]#1, [[VAR_19_8_]]#2, [[VAR_19_8_]]#3] : memref<1x1x2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[RES_10_:%.+]] = memref.alloc() alignment = 16 : memref<1x1x2x4xf32>
// CHECK-DAG:       [[LOOP_10_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_10_]]#0, [[LOOP_10_]]#1, [[LOOP_10_]]#2, [[LOOP_10_]]#3) with ([[LOOP_10_]]#0 -> [[I_41_:%.+]] = 0 to 1, [[LOOP_10_]]#1 -> [[I_42_:%.+]] = 0 to 1, [[LOOP_10_]]#2 -> [[I_43_:%.+]] = 0 to 2, [[LOOP_10_]]#3 -> [[I_44_:%.+]] = 0 to 4){
// CHECK:             [[VAR_19_9_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_10_]]#0, [[LOOP_10_]]#1, [[LOOP_10_]]#2, [[LOOP_10_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_1_MEM_2_1_1_1_:%.+]] = krnl.load [[RES_9_]]{{.}}[[CST_0_1_]], [[CST_0_1_]], [[VAR_19_9_]]#2, [[VAR_19_9_]]#3] : memref<1x1x2x4xf32>
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_1_MEM_1_1_1_1_1_:%.+]] = krnl.load [[RES_6_]]{{.}}[[CST_0_1_]], [[CST_0_1_]], [[VAR_19_9_]]#2, [[VAR_19_9_]]#3] : memref<1x1x2x4xf32>
// CHECK:             [[VAR_22_7_:%.+]] = arith.addf [[LOAD_VAR_reinterpret_cast_1_MEM_2_1_1_1_]], [[LOAD_VAR_reinterpret_cast_1_MEM_1_1_1_1_1_]] : f32
// CHECK:             krnl.store [[VAR_22_7_]], [[RES_10_]]{{.}}[[VAR_19_9_]]#0, [[VAR_19_9_]]#1, [[VAR_19_9_]]#2, [[VAR_19_9_]]#3] : memref<1x1x2x4xf32>
// CHECK:           }
// CHECK-DAG:       [[RES_11_:%.+]] = memref.alloc() alignment = 16 : memref<1x1x2x4xf32>
// CHECK-DAG:       [[LOOP_11_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_11_]]#0, [[LOOP_11_]]#1, [[LOOP_11_]]#2) with ([[LOOP_11_]]#0 -> [[I_45_:%.+]] = 0 to 1, [[LOOP_11_]]#1 -> [[I_46_:%.+]] = 0 to 1, [[LOOP_11_]]#2 -> [[I_47_:%.+]] = 0 to 2){
// CHECK-DAG:         [[VAR_19_10_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_11_]]#0, [[LOOP_11_]]#1, [[LOOP_11_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOOP_12_:%.+]] = krnl.define_loops 1
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_1_MEM_1_1_1_1_1_:%.+]] = krnl.iterate([[LOOP_12_]]) with ([[LOOP_12_]] -> [[I_44_:%.+]] = 0 to 4) iter_args([[I_36_:%.+]] = [[CST_0_]]) -> (f32){
// CHECK-DAG:           [[VAR_25_1_:%.+]] = krnl.get_induction_var_value([[LOOP_12_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_RES_10_MEM_:%.+]] = krnl.load [[RES_10_]]{{.}}[[VAR_19_10_]]#0, [[VAR_19_10_]]#1, [[VAR_19_10_]]#2, [[VAR_25_1_]]{{.}} : memref<1x1x2x4xf32>
// CHECK:               [[VAR_27_:%.+]] = arith.maxnumf [[I_36_]], [[LOAD_RES_10_MEM_]] : f32
// CHECK:               krnl.yield [[VAR_27_]] : f32
// CHECK:             }
// CHECK:             [[LOOP_13_:%.+]] = krnl.define_loops 1
// CHECK-DAG:         [[VAR_23_2_:%.+]] = krnl.iterate([[LOOP_13_]]) with ([[LOOP_13_]] -> [[I_44_1_:%.+]] = 0 to 4) iter_args([[I_36_1_:%.+]] = [[CST_0_dot_000000_]]) -> (f32){
// CHECK-DAG:           [[VAR_25_2_:%.+]] = krnl.get_induction_var_value([[LOOP_13_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_RES_10_MEM_1_:%.+]] = krnl.load [[RES_10_]]{{.}}[[VAR_19_10_]]#0, [[VAR_19_10_]]#1, [[VAR_19_10_]]#2, [[VAR_25_2_]]{{.}} : memref<1x1x2x4xf32>
// CHECK:               [[VAR_27_1_:%.+]] = arith.subf [[LOAD_RES_10_MEM_1_]], [[LOAD_VAR_reinterpret_cast_1_MEM_1_1_1_1_1_]] : f32
// CHECK:               [[VAR_28_:%.+]] = math.exp [[VAR_27_1_]] : f32
// CHECK:               [[VAR_29_:%.+]] = arith.addf [[I_36_1_]], [[VAR_28_]] : f32
// CHECK:               krnl.store [[VAR_28_]], [[RES_11_]]{{.}}[[VAR_19_10_]]#0, [[VAR_19_10_]]#1, [[VAR_19_10_]]#2, [[VAR_25_2_]]{{.}} : memref<1x1x2x4xf32>
// CHECK:               krnl.yield [[VAR_29_]] : f32
// CHECK:             }
// CHECK:             [[LOOP_14_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_14_]]) with ([[LOOP_14_]] -> [[I_48_:%.+]] = 0 to 4){
// CHECK:               [[VAR_25_3_:%.+]] = krnl.get_induction_var_value([[LOOP_14_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_RES_10_MEM_1_:%.+]] = krnl.load [[RES_11_]]{{.}}[[VAR_19_10_]]#0, [[VAR_19_10_]]#1, [[VAR_19_10_]]#2, [[VAR_25_3_]]{{.}} : memref<1x1x2x4xf32>
// CHECK:               [[VAR_27_2_:%.+]] = arith.divf [[LOAD_RES_10_MEM_1_]], [[VAR_23_2_]] : f32
// CHECK:               krnl.store [[VAR_27_2_]], [[RES_11_]]{{.}}[[VAR_19_10_]]#0, [[VAR_19_10_]]#1, [[VAR_19_10_]]#2, [[VAR_25_3_]]{{.}} : memref<1x1x2x4xf32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_12_:%.+]] = memref.alloc() alignment = 16 : memref<1x1x2x2xf32>
// CHECK-DAG:       [[LOOP_15_:%.+]]:5 = krnl.define_loops 5
// CHECK:           krnl.iterate([[LOOP_15_]]#0, [[LOOP_15_]]#1, [[LOOP_15_]]#2, [[LOOP_15_]]#3) with ([[LOOP_15_]]#0 -> [[I_49_:%.+]] = 0 to 1, [[LOOP_15_]]#1 -> [[I_50_:%.+]] = 0 to 1, [[LOOP_15_]]#2 -> [[I_51_:%.+]] = 0 to 2, [[LOOP_15_]]#3 -> [[I_52_:%.+]] = 0 to 2, [[LOOP_15_]]#4 -> [[I_53_:%.+]] = 0 to 4){
// CHECK-DAG:         [[VAR_19_11_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_15_]]#0, [[LOOP_15_]]#1, [[LOOP_15_]]#2, [[LOOP_15_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[LOOP_12_:%.+]] = krnl.iterate([[LOOP_15_]]#4) with () iter_args([[VAR_arg9_1_:%.+]] = [[CST_0_dot_000000_]]) -> (f32){
// CHECK-DAG:           [[LOAD_VAR_reinterpret_cast_1_MEM_1_1_1_1_1_1_:%.+]] = krnl.get_induction_var_value([[LOOP_15_]]#4) : (!krnl.loop) -> index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOOP_13_:%.+]] = krnl.load [[RES_11_]]{{.}}[[VAR_19_11_]]#0, [[VAR_19_11_]]#1, [[VAR_19_11_]]#2, [[LOAD_VAR_reinterpret_cast_1_MEM_1_1_1_1_1_1_]]{{.}} : memref<1x1x2x4xf32>
// CHECK-DAG:           [[VAR_23_2_:%.+]] = krnl.load [[V_]]{{.}}[[VAR_19_11_]]#0, [[VAR_19_11_]]#1, [[LOAD_VAR_reinterpret_cast_1_MEM_1_1_1_1_1_1_]], [[VAR_19_11_]]#3] : memref<1x1x4x2xf32>
// CHECK:               [[VAR_24_1_:%.+]] = arith.mulf [[LOOP_13_]], [[VAR_23_2_]] : f32
// CHECK:               [[VAR_25_4_:%.+]] = arith.addf [[VAR_arg9_1_]], [[VAR_24_1_]] : f32
// CHECK:               krnl.yield [[VAR_25_4_]] : f32
// CHECK:             }
// CHECK:             krnl.store [[LOOP_12_]], [[RES_12_]]{{.}}[[VAR_19_11_]]#0, [[VAR_19_11_]]#1, [[VAR_19_11_]]#2, [[VAR_19_11_]]#3] : memref<1x1x2x2xf32>
// CHECK:           }
// CHECK:           return [[RES_12_]] : memref<1x1x2x2xf32>
// CHECK:         }
}
