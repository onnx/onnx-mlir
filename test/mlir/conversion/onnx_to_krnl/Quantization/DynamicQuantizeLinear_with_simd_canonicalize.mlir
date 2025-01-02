// RUN: onnx-mlir-opt -O3 --march=z16 --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

// -----


func.func @test_dynamic_quantize_linear_simd_only(%arg0: tensor<256x16xf32>) -> (tensor<256x16xui8>, tensor<f32>, tensor<ui8>) {
  %y, %y_scale, %y_zero_point = "onnx.DynamicQuantizeLinear"(%arg0) : (tensor<256x16xf32>) -> (tensor<256x16xui8>, tensor<f32>, tensor<ui8>)
  return %y, %y_scale, %y_zero_point:  tensor<256x16xui8>, tensor<f32>, tensor<ui8>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_dynamic_quantize_linear_simd_only
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<256x16xf32>) -> (memref<256x16xui8>, memref<f32>, memref<ui8>) {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<2.550000e+02> : vector<16xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<0.000000e+00> : vector<16xf32>
// CHECK-DAG:       [[VAR_cst_1_:%.+]] = arith.constant dense<0xFF800000> : vector<32xf32>
// CHECK-DAG:       [[VAR_cst_2_:%.+]] = arith.constant dense<0x7F800000> : vector<32xf32>
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_4096_:%.+]] = arith.constant 4096 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_2_dot_550000_:%.+]] = arith.constant 2.550000e+02 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<256x16xui8>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() : memref<f32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() : memref<ui8>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_4096_]], [[RES_3_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_3_]]) : (memref<256x16xf32>, memref<1xindex>) -> memref<4096xf32>
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<32xf32>
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() : memref<f32>
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<32xf32>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           vector.store [[VAR_cst_2_]], [[RES_4_]]{{.}}[[CST_0_]]{{.}} : memref<32xf32>, vector<32xf32>
// CHECK:           vector.store [[VAR_cst_1_]], [[RES_6_]]{{.}}[[CST_0_]]{{.}} : memref<32xf32>, vector<32xf32>
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 4096){
// CHECK:             [[VAR_20_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_20_]]{{.}} : memref<4096xf32>, vector<32xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_1_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_20_]]{{.}} : memref<4096xf32>, vector<32xf32>
// CHECK-DAG:         [[LOAD_RES_4_MEM_:%.+]] = vector.load [[RES_4_]]{{.}}[[CST_0_]]{{.}} : memref<32xf32>, vector<32xf32>
// CHECK-DAG:         [[LOAD_RES_6_MEM_:%.+]] = vector.load [[RES_6_]]{{.}}[[CST_0_]]{{.}} : memref<32xf32>, vector<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_25_:%.+]] = arith.minnumf [[LOAD_RES_4_MEM_]], [[LOAD_VAR_reshape_MEM_]] : vector<32xf32>
// CHECK-DAG:         [[VAR_26_:%.+]] = arith.maxnumf [[LOAD_RES_6_MEM_]], [[LOAD_VAR_reshape_MEM_1_]] : vector<32xf32>
// CHECK:             vector.store [[VAR_25_]], [[RES_4_]]{{.}}[[CST_0_]]{{.}} : memref<32xf32>, vector<32xf32>
// CHECK:             vector.store [[VAR_26_]], [[RES_6_]]{{.}}[[CST_0_]]{{.}} : memref<32xf32>, vector<32xf32>
// CHECK:           }
// CHECK-DAG:       [[LOAD_RES_4_MEM_1_:%.+]] = vector.load [[RES_4_]]{{.}}[[CST_0_]]{{.}} : memref<32xf32>, vector<32xf32>
// CHECK-DAG:       [[LOAD_RES_6_MEM_1_:%.+]] = vector.load [[RES_6_]]{{.}}[[CST_0_]]{{.}} : memref<32xf32>, vector<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = vector.reduction <minnumf>, [[LOAD_RES_4_MEM_1_]] : vector<32xf32> into f32
// CHECK-DAG:       [[VAR_4_:%.+]] = vector.reduction <maxnumf>, [[LOAD_RES_6_MEM_1_]] : vector<32xf32> into f32
// CHECK:           krnl.store [[VAR_3_]], [[RES_5_]][] : memref<f32>
// CHECK:           krnl.store [[VAR_4_]], [[RES_7_]][] : memref<f32>
// CHECK-DAG:       [[LOAD_RES_5_MEM_:%.+]] = krnl.load [[RES_5_]][] : memref<f32>
// CHECK-DAG:       [[LOAD_RES_7_MEM_:%.+]] = krnl.load [[RES_7_]][] : memref<f32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_7_:%.+]] = arith.maxnumf [[LOAD_RES_7_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.minnumf [[LOAD_RES_5_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK:           [[VAR_9_:%.+]] = arith.subf [[VAR_7_]], [[VAR_8_]] : f32
// CHECK:           [[VAR_10_:%.+]] = arith.divf [[VAR_9_]], [[CST_2_dot_550000_]] : f32
// CHECK:           [[VAR_11_:%.+]] = arith.divf [[VAR_8_]], [[VAR_10_]] : f32
// CHECK:           [[VAR_12_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_11_]] : f32
// CHECK:           [[VAR_13_:%.+]] = arith.maxnumf [[VAR_12_]], [[CST_0_dot_000000_]] : f32
// CHECK:           [[VAR_14_:%.+]] = arith.minnumf [[VAR_13_]], [[CST_2_dot_550000_]] : f32
// CHECK:           [[VAR_15_:%.+]] = "krnl.round_even"([[VAR_14_]]) : (f32) -> f32
// CHECK:           [[VAR_16_:%.+]] = arith.fptoui [[VAR_15_]] : f32 to i32
// CHECK:           [[VAR_17_:%.+]] = arith.trunci [[VAR_16_]] : i32 to i8
// CHECK:           [[VAR_18_:%.+]] = builtin.unrealized_conversion_cast [[VAR_17_]] : i8 to ui8
// CHECK:           krnl.store [[VAR_10_]], [[RES_1_]][] : memref<f32>
// CHECK:           krnl.store [[VAR_18_]], [[RES_2_]][] : memref<ui8>
// CHECK:           [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_4096_]], [[RES_8_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_13_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_8_]]) : (memref<256x16xf32>, memref<1xindex>) -> memref<4096xf32>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_4096_]], [[RES_9_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_15_:%.+]] = memref.reshape [[RES_]]([[RES_]]_14) : (memref<256x16xui8>, memref<1xindex>) -> memref<4096xui8>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_1_]] 16 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 4096){
// CHECK:             [[VAR_20_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_2_:%.+]] = vector.load [[VAR_reshape_13_]]{{.}}[[VAR_20_1_]]{{.}} : memref<4096xf32>, vector<16xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_1_:%.+]] = vector.splat [[VAR_10_]] : vector<16xf32>
// CHECK:             [[LOAD_RES_4_MEM_2_:%.+]] = arith.divf [[LOAD_VAR_reshape_MEM_2_]], [[LOAD_VAR_reshape_MEM_1_]] : vector<16xf32>
// CHECK:             [[LOAD_RES_6_MEM_2_:%.+]] = vector.shape_cast [[LOAD_RES_4_MEM_2_]] : vector<16xf32> to vector<4x4xf32>
// CHECK:             [[VAR_25_1_:%.+]] = vector.extract [[LOAD_RES_6_MEM_2_]][0] : vector<4xf32> from vector<4x4xf32>
// CHECK:             [[VAR_26_1_:%.+]] = "krnl.round_even"([[VAR_25_1_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:         [[VAR_27_:%.+]] = vector.insert [[VAR_26_1_]], [[LOAD_RES_6_MEM_2_]] [0] : vector<4xf32> into vector<4x4xf32>
// CHECK-DAG:         [[VAR_28_:%.+]] = vector.extract [[LOAD_RES_6_MEM_2_]][1] : vector<4xf32> from vector<4x4xf32>
// CHECK:             [[VAR_29_:%.+]] = "krnl.round_even"([[VAR_28_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:         [[VAR_30_:%.+]] = vector.insert [[VAR_29_]], [[VAR_27_]] [1] : vector<4xf32> into vector<4x4xf32>
// CHECK-DAG:         [[VAR_31_:%.+]] = vector.extract [[LOAD_RES_6_MEM_2_]][2] : vector<4xf32> from vector<4x4xf32>
// CHECK:             [[VAR_32_:%.+]] = "krnl.round_even"([[VAR_31_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:         [[VAR_33_:%.+]] = vector.insert [[VAR_32_]], [[VAR_30_]] [2] : vector<4xf32> into vector<4x4xf32>
// CHECK-DAG:         [[VAR_34_:%.+]] = vector.extract [[LOAD_RES_6_MEM_2_]][3] : vector<4xf32> from vector<4x4xf32>
// CHECK:             [[VAR_35_:%.+]] = "krnl.round_even"([[VAR_34_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK:             [[VAR_36_:%.+]] = vector.insert [[VAR_35_]], [[VAR_33_]] [3] : vector<4xf32> into vector<4x4xf32>
// CHECK-DAG:         [[VAR_37_:%.+]] = vector.shape_cast [[VAR_36_]] : vector<4x4xf32> to vector<16xf32>
// CHECK-DAG:         [[VAR_38_:%.+]] = vector.splat [[VAR_15_]] : vector<16xf32>
// CHECK:             [[VAR_39_:%.+]] = arith.addf [[VAR_37_]], [[VAR_38_]] : vector<16xf32>
// CHECK:             [[VAR_40_:%.+]] = arith.maxnumf [[VAR_39_]], [[VAR_cst_0_]] : vector<16xf32>
// CHECK:             [[VAR_41_:%.+]] = arith.minnumf [[VAR_40_]], [[VAR_cst_]] : vector<16xf32>
// CHECK:             [[VAR_42_:%.+]] = arith.fptoui [[VAR_41_]] : vector<16xf32> to vector<16xi32>
// CHECK:             [[VAR_43_:%.+]] = arith.trunci [[VAR_42_]] : vector<16xi32> to vector<16xi8>
// CHECK:             [[VAR_44_:%.+]] = builtin.unrealized_conversion_cast [[VAR_43_]] : vector<16xi8> to vector<16xui8>
// CHECK:             vector.store [[VAR_44_]], [[VAR_reshape_15_]]{{.}}[[VAR_20_1_]]{{.}} : memref<4096xui8>, vector<16xui8>
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_]]_5, [[RES_]]_6 : memref<256x16xui8>, memref<f32>, memref<ui8>
// CHECK:         }
}

// -----


func.func @test_dynamic_quantize_linear_simd_and_scalar(%arg0: tensor<255x17xf32>) -> (tensor<255x17xui8>, tensor<f32>, tensor<ui8>) {
  %y, %y_scale, %y_zero_point = "onnx.DynamicQuantizeLinear"(%arg0) : (tensor<255x17xf32>) -> (tensor<255x17xui8>, tensor<f32>, tensor<ui8>)
  return %y, %y_scale, %y_zero_point:  tensor<255x17xui8>, tensor<f32>, tensor<ui8>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_dynamic_quantize_linear_simd_and_scalar
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<255x17xf32>) -> (memref<255x17xui8>, memref<f32>, memref<ui8>) {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<2.550000e+02> : vector<16xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<0.000000e+00> : vector<16xf32>
// CHECK-DAG:       [[VAR_cst_1_:%.+]] = arith.constant dense<0xFF800000> : vector<32xf32>
// CHECK-DAG:       [[VAR_cst_2_:%.+]] = arith.constant dense<0x7F800000> : vector<32xf32>
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_4335_:%.+]] = arith.constant 4335 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_2_dot_550000_:%.+]] = arith.constant 2.550000e+02 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<255x17xui8>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() : memref<f32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() : memref<ui8>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_4335_]], [[RES_3_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_3_]]) : (memref<255x17xf32>, memref<1xindex>) -> memref<4335xf32>
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<32xf32>
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() : memref<f32>
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<32xf32>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           vector.store [[VAR_cst_2_]], [[RES_4_]]{{.}}[[CST_0_]]{{.}} : memref<32xf32>, vector<32xf32>
// CHECK:           vector.store [[VAR_cst_1_]], [[RES_6_]]{{.}}[[CST_0_]]{{.}} : memref<32xf32>, vector<32xf32>
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 4304){
// CHECK:             [[VAR_22_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_22_]]{{.}} : memref<4335xf32>, vector<32xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_1_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_22_]]{{.}} : memref<4335xf32>, vector<32xf32>
// CHECK-DAG:         [[LOAD_RES_4_MEM_:%.+]] = vector.load [[RES_4_]]{{.}}[[CST_0_]]{{.}} : memref<32xf32>, vector<32xf32>
// CHECK-DAG:         [[LOAD_RES_6_MEM_:%.+]] = vector.load [[RES_6_]]{{.}}[[CST_0_]]{{.}} : memref<32xf32>, vector<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_27_:%.+]] = arith.minnumf [[LOAD_RES_4_MEM_]], [[LOAD_VAR_reshape_MEM_]] : vector<32xf32>
// CHECK-DAG:         [[VAR_28_:%.+]] = arith.maxnumf [[LOAD_RES_6_MEM_]], [[LOAD_VAR_reshape_MEM_1_]] : vector<32xf32>
// CHECK:             vector.store [[VAR_27_]], [[RES_4_]]{{.}}[[CST_0_]]{{.}} : memref<32xf32>, vector<32xf32>
// CHECK:             vector.store [[VAR_28_]], [[RES_6_]]{{.}}[[CST_0_]]{{.}} : memref<32xf32>, vector<32xf32>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 4320 to 4335){
// CHECK:             [[VAR_22_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_2_:%.+]] = krnl.load [[VAR_reshape_]]{{.}}[[VAR_22_1_]]{{.}} : memref<4335xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_3_:%.+]] = krnl.load [[VAR_reshape_]]{{.}}[[VAR_22_1_]]{{.}} : memref<4335xf32>
// CHECK-DAG:         [[LOAD_RES_4_MEM_1_:%.+]] = krnl.load [[RES_4_]]{{.}}[[CST_0_]]{{.}} : memref<32xf32>
// CHECK-DAG:         [[LOAD_RES_6_MEM_1_:%.+]] = krnl.load [[RES_6_]]{{.}}[[CST_0_]]{{.}} : memref<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_27_1_:%.+]] = arith.minnumf [[LOAD_RES_4_MEM_1_]], [[LOAD_VAR_reshape_MEM_2_]] : f32
// CHECK-DAG:         [[VAR_28_1_:%.+]] = arith.maxnumf [[LOAD_RES_6_MEM_1_]], [[LOAD_VAR_reshape_MEM_3_]] : f32
// CHECK:             krnl.store [[VAR_27_1_]], [[RES_4_]]{{.}}[[CST_0_]]{{.}} : memref<32xf32>
// CHECK:             krnl.store [[VAR_28_1_]], [[RES_6_]]{{.}}[[CST_0_]]{{.}} : memref<32xf32>
// CHECK:           }
// CHECK-DAG:       [[LOAD_RES_4_MEM_2_:%.+]] = vector.load [[RES_4_]]{{.}}[[CST_0_]]{{.}} : memref<32xf32>, vector<32xf32>
// CHECK-DAG:       [[LOAD_RES_6_MEM_2_:%.+]] = vector.load [[RES_6_]]{{.}}[[CST_0_]]{{.}} : memref<32xf32>, vector<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = vector.reduction <minnumf>, [[LOAD_RES_4_MEM_2_]] : vector<32xf32> into f32
// CHECK-DAG:       [[VAR_5_:%.+]] = vector.reduction <maxnumf>, [[LOAD_RES_6_MEM_2_]] : vector<32xf32> into f32
// CHECK:           krnl.store [[VAR_4_]], [[RES_5_]][] : memref<f32>
// CHECK:           krnl.store [[VAR_5_]], [[RES_7_]][] : memref<f32>
// CHECK-DAG:       [[LOAD_RES_5_MEM_:%.+]] = krnl.load [[RES_5_]][] : memref<f32>
// CHECK-DAG:       [[LOAD_RES_7_MEM_:%.+]] = krnl.load [[RES_7_]][] : memref<f32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.maxnumf [[LOAD_RES_7_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:       [[VAR_9_:%.+]] = arith.minnumf [[LOAD_RES_5_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK:           [[VAR_10_:%.+]] = arith.subf [[VAR_8_]], [[VAR_9_]] : f32
// CHECK:           [[VAR_11_:%.+]] = arith.divf [[VAR_10_]], [[CST_2_dot_550000_]] : f32
// CHECK:           [[VAR_12_:%.+]] = arith.divf [[VAR_9_]], [[VAR_11_]] : f32
// CHECK:           [[VAR_13_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_12_]] : f32
// CHECK:           [[VAR_14_:%.+]] = arith.maxnumf [[VAR_13_]], [[CST_0_dot_000000_]] : f32
// CHECK:           [[VAR_15_:%.+]] = arith.minnumf [[VAR_14_]], [[CST_2_dot_550000_]] : f32
// CHECK:           [[VAR_16_:%.+]] = "krnl.round_even"([[VAR_15_]]) : (f32) -> f32
// CHECK:           [[VAR_17_:%.+]] = arith.fptoui [[VAR_16_]] : f32 to i32
// CHECK:           [[VAR_18_:%.+]] = arith.trunci [[VAR_17_]] : i32 to i8
// CHECK:           [[VAR_19_:%.+]] = builtin.unrealized_conversion_cast [[VAR_18_]] : i8 to ui8
// CHECK:           krnl.store [[VAR_11_]], [[RES_1_]][] : memref<f32>
// CHECK:           krnl.store [[VAR_19_]], [[RES_2_]][] : memref<ui8>
// CHECK:           [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_4335_]], [[RES_8_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_13_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_8_]]) : (memref<255x17xf32>, memref<1xindex>) -> memref<4335xf32>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_4335_]], [[RES_9_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_15_:%.+]] = memref.reshape [[RES_]]([[RES_]]_14) : (memref<255x17xui8>, memref<1xindex>) -> memref<4335xui8>
// CHECK-DAG:       [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_2_]] 16 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__1_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 4320){
// CHECK:             [[VAR_22_2_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_2_:%.+]] = vector.load [[VAR_reshape_13_]]{{.}}[[VAR_22_2_]]{{.}} : memref<4335xf32>, vector<16xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_3_:%.+]] = vector.splat [[VAR_11_]] : vector<16xf32>
// CHECK:             [[LOAD_RES_4_MEM_1_:%.+]] = arith.divf [[LOAD_VAR_reshape_MEM_2_]], [[LOAD_VAR_reshape_MEM_3_]] : vector<16xf32>
// CHECK:             [[LOAD_RES_6_MEM_1_:%.+]] = vector.shape_cast [[LOAD_RES_4_MEM_1_]] : vector<16xf32> to vector<4x4xf32>
// CHECK:             [[VAR_27_2_:%.+]] = vector.extract [[LOAD_RES_6_MEM_1_]][0] : vector<4xf32> from vector<4x4xf32>
// CHECK:             [[VAR_28_2_:%.+]] = "krnl.round_even"([[VAR_27_2_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:         [[VAR_29_:%.+]] = vector.insert [[VAR_28_2_]], [[LOAD_RES_6_MEM_1_]] [0] : vector<4xf32> into vector<4x4xf32>
// CHECK-DAG:         [[VAR_30_:%.+]] = vector.extract [[LOAD_RES_6_MEM_1_]][1] : vector<4xf32> from vector<4x4xf32>
// CHECK:             [[VAR_31_:%.+]] = "krnl.round_even"([[VAR_30_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:         [[VAR_32_:%.+]] = vector.insert [[VAR_31_]], [[VAR_29_]] [1] : vector<4xf32> into vector<4x4xf32>
// CHECK-DAG:         [[VAR_33_:%.+]] = vector.extract [[LOAD_RES_6_MEM_1_]][2] : vector<4xf32> from vector<4x4xf32>
// CHECK:             [[VAR_34_:%.+]] = "krnl.round_even"([[VAR_33_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:         [[VAR_35_:%.+]] = vector.insert [[VAR_34_]], [[VAR_32_]] [2] : vector<4xf32> into vector<4x4xf32>
// CHECK-DAG:         [[VAR_36_:%.+]] = vector.extract [[LOAD_RES_6_MEM_1_]][3] : vector<4xf32> from vector<4x4xf32>
// CHECK:             [[VAR_37_:%.+]] = "krnl.round_even"([[VAR_36_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK:             [[VAR_38_:%.+]] = vector.insert [[VAR_37_]], [[VAR_35_]] [3] : vector<4xf32> into vector<4x4xf32>
// CHECK-DAG:         [[VAR_39_:%.+]] = vector.shape_cast [[VAR_38_]] : vector<4x4xf32> to vector<16xf32>
// CHECK-DAG:         [[VAR_40_:%.+]] = vector.splat [[VAR_16_]] : vector<16xf32>
// CHECK:             [[VAR_41_:%.+]] = arith.addf [[VAR_39_]], [[VAR_40_]] : vector<16xf32>
// CHECK:             [[VAR_42_:%.+]] = arith.maxnumf [[VAR_41_]], [[VAR_cst_0_]] : vector<16xf32>
// CHECK:             [[VAR_43_:%.+]] = arith.minnumf [[VAR_42_]], [[VAR_cst_]] : vector<16xf32>
// CHECK:             [[VAR_44_:%.+]] = arith.fptoui [[VAR_43_]] : vector<16xf32> to vector<16xi32>
// CHECK:             [[VAR_45_:%.+]] = arith.trunci [[VAR_44_]] : vector<16xi32> to vector<16xi8>
// CHECK:             [[VAR_46_:%.+]] = builtin.unrealized_conversion_cast [[VAR_45_]] : vector<16xi8> to vector<16xui8>
// CHECK:             vector.store [[VAR_46_]], [[VAR_reshape_15_]]{{.}}[[VAR_22_2_]]{{.}} : memref<4335xui8>, vector<16xui8>
// CHECK:           }
// CHECK:           [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[I_3_:%.+]] = 4320 to 4335){
// CHECK:             [[VAR_22_3_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_VAR_reshape_MEM_2_1_:%.+]] = krnl.load [[VAR_reshape_13_]]{{.}}[[VAR_22_3_]]{{.}} : memref<4335xf32>
// CHECK:             [[LOAD_VAR_reshape_MEM_3_1_:%.+]] = arith.divf [[LOAD_VAR_reshape_MEM_2_1_]], [[VAR_11_]] : f32
// CHECK:             [[LOAD_RES_4_MEM_1_1_:%.+]] = "krnl.round_even"([[LOAD_VAR_reshape_MEM_3_1_]]) : (f32) -> f32
// CHECK:             [[LOAD_RES_6_MEM_1_1_:%.+]] = arith.addf [[LOAD_RES_4_MEM_1_1_]], [[VAR_16_]] : f32
// CHECK:             [[VAR_27_3_:%.+]] = arith.maxnumf [[LOAD_RES_6_MEM_1_1_]], [[CST_0_dot_000000_]] : f32
// CHECK:             [[VAR_28_3_:%.+]] = arith.minnumf [[VAR_27_3_]], [[CST_2_dot_550000_]] : f32
// CHECK:             [[VAR_29_1_:%.+]] = arith.fptoui [[VAR_28_3_]] : f32 to i32
// CHECK:             [[VAR_30_1_:%.+]] = arith.trunci [[VAR_29_1_]] : i32 to i8
// CHECK:             [[VAR_31_1_:%.+]] = builtin.unrealized_conversion_cast [[VAR_30_1_]] : i8 to ui8
// CHECK:             krnl.store [[VAR_31_1_]], [[VAR_reshape_15_]]{{.}}[[VAR_22_3_]]{{.}} : memref<4335xui8>
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_]]_5, [[RES_]]_6 : memref<255x17xui8>, memref<f32>, memref<ui8>
// CHECK:         }
}

// -----


func.func @test_dynamic_quantize_linear_reduced_simd_only(%arg0: tensor<1x8xf32>) -> (tensor<1x8xui8>, tensor<f32>, tensor<ui8>) {
  %y, %y_scale, %y_zero_point = "onnx.DynamicQuantizeLinear"(%arg0) : (tensor<1x8xf32>) -> (tensor<1x8xui8>, tensor<f32>, tensor<ui8>)
  return %y, %y_scale, %y_zero_point:  tensor<1x8xui8>, tensor<f32>, tensor<ui8>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_dynamic_quantize_linear_reduced_simd_only
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x8xf32>) -> (memref<1x8xui8>, memref<f32>, memref<ui8>) {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<2.550000e+02> : vector<8xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<0.000000e+00> : vector<8xf32>
// CHECK-DAG:       [[VAR_cst_1_:%.+]] = arith.constant dense<0xFF800000> : vector<8xf32>
// CHECK-DAG:       [[VAR_cst_2_:%.+]] = arith.constant dense<0x7F800000> : vector<8xf32>
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_2_dot_550000_:%.+]] = arith.constant 2.550000e+02 : f32
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x8xui8>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() : memref<f32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() : memref<ui8>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_8_]], [[RES_3_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_3_]]) : (memref<1x8xf32>, memref<1xindex>) -> memref<8xf32>
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<8xf32>
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() : memref<f32>
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<8xf32>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           vector.store [[VAR_cst_2_]], [[RES_4_]]{{.}}[[CST_0_]]{{.}} : memref<8xf32>, vector<8xf32>
// CHECK:           vector.store [[VAR_cst_1_]], [[RES_6_]]{{.}}[[CST_0_]]{{.}} : memref<8xf32>, vector<8xf32>
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 8){
// CHECK:             [[VAR_20_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_20_]]{{.}} : memref<8xf32>, vector<8xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_1_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_20_]]{{.}} : memref<8xf32>, vector<8xf32>
// CHECK-DAG:         [[LOAD_RES_4_MEM_:%.+]] = vector.load [[RES_4_]]{{.}}[[CST_0_]]{{.}} : memref<8xf32>, vector<8xf32>
// CHECK-DAG:         [[LOAD_RES_6_MEM_:%.+]] = vector.load [[RES_6_]]{{.}}[[CST_0_]]{{.}} : memref<8xf32>, vector<8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_25_:%.+]] = arith.minnumf [[LOAD_RES_4_MEM_]], [[LOAD_VAR_reshape_MEM_]] : vector<8xf32>
// CHECK-DAG:         [[VAR_26_:%.+]] = arith.maxnumf [[LOAD_RES_6_MEM_]], [[LOAD_VAR_reshape_MEM_1_]] : vector<8xf32>
// CHECK:             vector.store [[VAR_25_]], [[RES_4_]]{{.}}[[CST_0_]]{{.}} : memref<8xf32>, vector<8xf32>
// CHECK:             vector.store [[VAR_26_]], [[RES_6_]]{{.}}[[CST_0_]]{{.}} : memref<8xf32>, vector<8xf32>
// CHECK:           }
// CHECK-DAG:       [[LOAD_RES_4_MEM_1_:%.+]] = vector.load [[RES_4_]]{{.}}[[CST_0_]]{{.}} : memref<8xf32>, vector<8xf32>
// CHECK-DAG:       [[LOAD_RES_6_MEM_1_:%.+]] = vector.load [[RES_6_]]{{.}}[[CST_0_]]{{.}} : memref<8xf32>, vector<8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = vector.reduction <minnumf>, [[LOAD_RES_4_MEM_1_]] : vector<8xf32> into f32
// CHECK-DAG:       [[VAR_4_:%.+]] = vector.reduction <maxnumf>, [[LOAD_RES_6_MEM_1_]] : vector<8xf32> into f32
// CHECK:           krnl.store [[VAR_3_]], [[RES_5_]][] : memref<f32>
// CHECK:           krnl.store [[VAR_4_]], [[RES_7_]][] : memref<f32>
// CHECK-DAG:       [[LOAD_RES_5_MEM_:%.+]] = krnl.load [[RES_5_]][] : memref<f32>
// CHECK-DAG:       [[LOAD_RES_7_MEM_:%.+]] = krnl.load [[RES_7_]][] : memref<f32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_7_:%.+]] = arith.maxnumf [[LOAD_RES_7_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.minnumf [[LOAD_RES_5_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK:           [[VAR_9_:%.+]] = arith.subf [[VAR_7_]], [[VAR_8_]] : f32
// CHECK:           [[VAR_10_:%.+]] = arith.divf [[VAR_9_]], [[CST_2_dot_550000_]] : f32
// CHECK:           [[VAR_11_:%.+]] = arith.divf [[VAR_8_]], [[VAR_10_]] : f32
// CHECK:           [[VAR_12_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_11_]] : f32
// CHECK:           [[VAR_13_:%.+]] = arith.maxnumf [[VAR_12_]], [[CST_0_dot_000000_]] : f32
// CHECK:           [[VAR_14_:%.+]] = arith.minnumf [[VAR_13_]], [[CST_2_dot_550000_]] : f32
// CHECK:           [[VAR_15_:%.+]] = "krnl.round_even"([[VAR_14_]]) : (f32) -> f32
// CHECK:           [[VAR_16_:%.+]] = arith.fptoui [[VAR_15_]] : f32 to i32
// CHECK:           [[VAR_17_:%.+]] = arith.trunci [[VAR_16_]] : i32 to i8
// CHECK:           [[VAR_18_:%.+]] = builtin.unrealized_conversion_cast [[VAR_17_]] : i8 to ui8
// CHECK:           krnl.store [[VAR_10_]], [[RES_1_]][] : memref<f32>
// CHECK:           krnl.store [[VAR_18_]], [[RES_2_]][] : memref<ui8>
// CHECK:           [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_8_]], [[RES_8_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_13_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_8_]]) : (memref<1x8xf32>, memref<1xindex>) -> memref<8xf32>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_8_]], [[RES_9_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_15_:%.+]] = memref.reshape [[RES_]]([[RES_]]_14) : (memref<1x8xui8>, memref<1xindex>) -> memref<8xui8>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_1_]] 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 8){
// CHECK:             [[VAR_20_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_2_:%.+]] = vector.load [[VAR_reshape_13_]]{{.}}[[VAR_20_1_]]{{.}} : memref<8xf32>, vector<8xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_1_:%.+]] = vector.splat [[VAR_10_]] : vector<8xf32>
// CHECK:             [[LOAD_RES_4_MEM_2_:%.+]] = arith.divf [[LOAD_VAR_reshape_MEM_2_]], [[LOAD_VAR_reshape_MEM_1_]] : vector<8xf32>
// CHECK:             [[LOAD_RES_6_MEM_2_:%.+]] = vector.shape_cast [[LOAD_RES_4_MEM_2_]] : vector<8xf32> to vector<2x4xf32>
// CHECK:             [[VAR_25_1_:%.+]] = vector.extract [[LOAD_RES_6_MEM_2_]][0] : vector<4xf32> from vector<2x4xf32>
// CHECK:             [[VAR_26_1_:%.+]] = "krnl.round_even"([[VAR_25_1_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK-DAG:         [[VAR_27_:%.+]] = vector.insert [[VAR_26_1_]], [[LOAD_RES_6_MEM_2_]] [0] : vector<4xf32> into vector<2x4xf32>
// CHECK-DAG:         [[VAR_28_:%.+]] = vector.extract [[LOAD_RES_6_MEM_2_]][1] : vector<4xf32> from vector<2x4xf32>
// CHECK:             [[VAR_29_:%.+]] = "krnl.round_even"([[VAR_28_]]) : (vector<4xf32>) -> vector<4xf32>
// CHECK:             [[VAR_30_:%.+]] = vector.insert [[VAR_29_]], [[VAR_27_]] [1] : vector<4xf32> into vector<2x4xf32>
// CHECK-DAG:         [[VAR_31_:%.+]] = vector.shape_cast [[VAR_30_]] : vector<2x4xf32> to vector<8xf32>
// CHECK-DAG:         [[VAR_32_:%.+]] = vector.splat [[VAR_15_]] : vector<8xf32>
// CHECK:             [[VAR_33_:%.+]] = arith.addf [[VAR_31_]], [[VAR_32_]] : vector<8xf32>
// CHECK:             [[VAR_34_:%.+]] = arith.maxnumf [[VAR_33_]], [[VAR_cst_0_]] : vector<8xf32>
// CHECK:             [[VAR_35_:%.+]] = arith.minnumf [[VAR_34_]], [[VAR_cst_]] : vector<8xf32>
// CHECK:             [[VAR_36_:%.+]] = arith.fptoui [[VAR_35_]] : vector<8xf32> to vector<8xi32>
// CHECK:             [[VAR_37_:%.+]] = arith.trunci [[VAR_36_]] : vector<8xi32> to vector<8xi8>
// CHECK:             [[VAR_38_:%.+]] = builtin.unrealized_conversion_cast [[VAR_37_]] : vector<8xi8> to vector<8xui8>
// CHECK:             vector.store [[VAR_38_]], [[VAR_reshape_15_]]{{.}}[[VAR_20_1_]]{{.}} : memref<8xui8>, vector<8xui8>
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_]]_5, [[RES_]]_6 : memref<1x8xui8>, memref<f32>, memref<ui8>
// CHECK:         }
}

