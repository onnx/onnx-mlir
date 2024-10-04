// RUN: onnx-mlir-opt -O3 -mcpu=z16 --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

// -----


func.func @test_dynamic_quantize_linear_simd_only(%arg0: tensor<256x16xf32>) -> (tensor<256x16xui8>, tensor<f32>, tensor<ui8>) {
  %y, %y_scale, %y_zero_point = "onnx.DynamicQuantizeLinear"(%arg0) : (tensor<256x16xf32>) -> (tensor<256x16xui8>, tensor<f32>, tensor<ui8>)
  return %y, %y_scale, %y_zero_point:  tensor<256x16xui8>, tensor<f32>, tensor<ui8>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_dynamic_quantize_linear_simd_only
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<256x16xf32>) -> (memref<256x16xui8>, memref<f32>, memref<ui8>) {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<2.550000e+02> : vector<8xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<0.000000e+00> : vector<8xf32>
// CHECK-DAG:       [[VAR_cst_1_:%.+]] = arith.constant dense<5.000000e-01> : vector<8xf32>
// CHECK-DAG:       [[VAR_cst_2_:%.+]] = arith.constant dense<2.000000e+00> : vector<8xf32>
// CHECK-DAG:       [[VAR_cst_3_:%.+]] = arith.constant dense<1.000000e+00> : vector<8xf32>
// CHECK-DAG:       [[VAR_cst_4_:%.+]] = arith.constant dense<0xFF800000> : vector<32xf32>
// CHECK-DAG:       [[VAR_cst_5_:%.+]] = arith.constant dense<0x7F800000> : vector<32xf32>
// CHECK-DAG:       [[CST_5_dot_000000_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:       [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
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
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() : memref<f32>
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<32xf32>
// CHECK:           vector.store [[VAR_cst_5_]], [[RES_5_]]{{.}}[[CST_0_]]{{.}} : memref<32xf32>, vector<32xf32>
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() : memref<f32>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc() {{.*}}: memref<32xf32>
// CHECK:           vector.store [[VAR_cst_4_]], [[RES_7_]]{{.}}[[CST_0_]]{{.}} : memref<32xf32>, vector<32xf32>
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = [[CST_0_]] to [[CST_4096_]]){
// CHECK:             [[VAR_32_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_32_]]{{.}} : memref<4096xf32>, vector<32xf32>
// CHECK-DAG:         [[LOAD_RES_5_MEM_:%.+]] = vector.load [[RES_5_]]{{.}}[[CST_0_]]{{.}} : memref<32xf32>, vector<32xf32>
// CHECK:             [[VAR_35_:%.+]] = arith.minnumf [[LOAD_RES_5_MEM_]], [[LOAD_VAR_reshape_MEM_]] : vector<32xf32>
// CHECK:             vector.store [[VAR_35_]], [[RES_5_]]{{.}}[[CST_0_]]{{.}} : memref<32xf32>, vector<32xf32>
// CHECK:             [[LOAD_RES_7_MEM_:%.+]] = vector.load [[RES_7_]]{{.}}[[CST_0_]]{{.}} : memref<32xf32>, vector<32xf32>
// CHECK:             [[VAR_37_:%.+]] = arith.maxnumf [[LOAD_RES_7_MEM_]], [[LOAD_VAR_reshape_MEM_]] : vector<32xf32>
// CHECK:             vector.store [[VAR_37_]], [[RES_7_]]{{.}}[[CST_0_]]{{.}} : memref<32xf32>, vector<32xf32>
// CHECK:           }
// CHECK:           [[LOAD_RES_5_MEM_1_:%.+]] = vector.load [[RES_5_]]{{.}}[[CST_0_]]{{.}} : memref<32xf32>, vector<32xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = vector.reduction <minnumf>, [[LOAD_RES_5_MEM_1_]] : vector<32xf32> into f32
// CHECK-DAG:       [[LOAD_RES_7_MEM_1_:%.+]] = vector.load [[RES_7_]]{{.}}[[CST_0_]]{{.}} : memref<32xf32>, vector<32xf32>
// CHECK:           [[VAR_4_:%.+]] = vector.reduction <maxnumf>, [[LOAD_RES_7_MEM_1_]] : vector<32xf32> into f32
// CHECK:           krnl.store [[VAR_2_]], [[RES_4_]][] : memref<f32>
// CHECK:           krnl.store [[VAR_4_]], [[RES_6_]][] : memref<f32>
// CHECK-DAG:       [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]][] : memref<f32>
// CHECK-DAG:       [[LOAD_RES_6_MEM_:%.+]] = krnl.load [[RES_6_]][] : memref<f32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_7_:%.+]] = arith.maxnumf [[LOAD_RES_6_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.minnumf [[LOAD_RES_4_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK:           [[VAR_9_:%.+]] = arith.subf [[VAR_7_]], [[VAR_8_]] : f32
// CHECK:           [[VAR_10_:%.+]] = arith.divf [[VAR_9_]], [[CST_2_dot_550000_]] : f32
// CHECK:           [[VAR_11_:%.+]] = arith.divf [[VAR_8_]], [[VAR_10_]] : f32
// CHECK:           [[VAR_12_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_11_]] : f32
// CHECK:           [[VAR_13_:%.+]] = arith.maxnumf [[VAR_12_]], [[CST_0_dot_000000_]] : f32
// CHECK:           [[VAR_14_:%.+]] = arith.minnumf [[VAR_13_]], [[CST_2_dot_550000_]] : f32
// CHECK:           [[VAR_15_:%.+]] = math.floor [[VAR_14_]] : f32
// CHECK:           [[VAR_16_:%.+]] = arith.subf [[VAR_14_]], [[VAR_15_]] : f32
// CHECK-DAG:       [[VAR_17_:%.+]] = arith.cmpf ogt, [[VAR_16_]], [[CST_5_dot_000000_]] : f32
// CHECK-DAG:       [[VAR_18_:%.+]] = arith.addf [[VAR_15_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_19_:%.+]] = arith.select [[VAR_17_]], [[VAR_18_]], [[VAR_15_]] : f32
// CHECK-DAG:       [[VAR_20_:%.+]] = arith.mulf [[VAR_15_]], [[CST_5_dot_000000_]] : f32
// CHECK:           [[VAR_21_:%.+]] = math.floor [[VAR_20_]] : f32
// CHECK:           [[VAR_22_:%.+]] = arith.mulf [[VAR_21_]], [[CST_2_dot_000000_]] : f32
// CHECK:           [[VAR_23_:%.+]] = arith.subf [[VAR_15_]], [[VAR_22_]] : f32
// CHECK-DAG:       [[VAR_24_:%.+]] = arith.cmpf oeq, [[VAR_23_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:       [[VAR_25_:%.+]] = arith.addf [[VAR_15_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_26_:%.+]] = arith.select [[VAR_24_]], [[VAR_25_]], [[VAR_15_]] : f32
// CHECK-DAG:       [[VAR_27_:%.+]] = arith.cmpf oeq, [[VAR_16_]], [[CST_5_dot_000000_]] : f32
// CHECK:           [[VAR_28_:%.+]] = arith.select [[VAR_27_]], [[VAR_26_]], [[VAR_19_]] : f32
// CHECK:           [[VAR_29_:%.+]] = arith.fptoui [[VAR_28_]] : f32 to i8
// CHECK:           [[VAR_30_:%.+]] = builtin.unrealized_conversion_cast [[VAR_29_]] : i8 to ui8
// CHECK:           krnl.store [[VAR_10_]], [[RES_1_]][] : memref<f32>
// CHECK:           krnl.store [[VAR_30_]], [[RES_2_]][] : memref<ui8>
// CHECK:           [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_4096_]], [[RES_8_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_19_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_8_]]) : (memref<256x16xf32>, memref<1xindex>) -> memref<4096xf32>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_4096_]], [[RES_9_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_21_:%.+]] = memref.reshape [[RES_]]([[RES_]]_20) : (memref<256x16xui8>, memref<1xindex>) -> memref<4096xui8>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_1_]] 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 4096){
// CHECK:             [[VAR_32_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_1_:%.+]] = vector.load [[VAR_reshape_19_]]{{.}}[[VAR_32_1_]]{{.}} : memref<4096xf32>, vector<8xf32>
// CHECK-DAG:         [[LOAD_RES_5_MEM_2_:%.+]] = vector.splat [[VAR_10_]] : vector<8xf32>
// CHECK:             [[VAR_35_1_:%.+]] = arith.divf [[LOAD_VAR_reshape_MEM_1_]], [[LOAD_RES_5_MEM_2_]] : vector<8xf32>
// CHECK:             [[LOAD_RES_7_MEM_2_:%.+]] = math.floor [[VAR_35_1_]] : vector<8xf32>
// CHECK:             [[VAR_37_1_:%.+]] = arith.subf [[VAR_35_1_]], [[LOAD_RES_7_MEM_2_]] : vector<8xf32>
// CHECK-DAG:         [[VAR_38_:%.+]] = arith.cmpf ogt, [[VAR_37_1_]], [[VAR_cst_1_]] : vector<8xf32>
// CHECK-DAG:         [[VAR_39_:%.+]] = arith.addf [[LOAD_RES_7_MEM_2_]], [[VAR_cst_3_]] : vector<8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_40_:%.+]] = arith.select [[VAR_38_]], [[VAR_39_]], [[LOAD_RES_7_MEM_2_]] : vector<8xi1>, vector<8xf32>
// CHECK-DAG:         [[VAR_41_:%.+]] = arith.mulf [[LOAD_RES_7_MEM_2_]], [[VAR_cst_1_]] : vector<8xf32>
// CHECK:             [[VAR_42_:%.+]] = math.floor [[VAR_41_]] : vector<8xf32>
// CHECK:             [[VAR_43_:%.+]] = arith.mulf [[VAR_42_]], [[VAR_cst_2_]] : vector<8xf32>
// CHECK:             [[VAR_44_:%.+]] = arith.subf [[LOAD_RES_7_MEM_2_]], [[VAR_43_]] : vector<8xf32>
// CHECK-DAG:         [[VAR_45_:%.+]] = arith.cmpf oeq, [[VAR_44_]], [[VAR_cst_3_]] : vector<8xf32>
// CHECK-DAG:         [[VAR_46_:%.+]] = arith.addf [[LOAD_RES_7_MEM_2_]], [[VAR_cst_3_]] : vector<8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_47_:%.+]] = arith.select [[VAR_45_]], [[VAR_46_]], [[LOAD_RES_7_MEM_2_]] : vector<8xi1>, vector<8xf32>
// CHECK-DAG:         [[VAR_48_:%.+]] = arith.cmpf oeq, [[VAR_37_1_]], [[VAR_cst_1_]] : vector<8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_49_:%.+]] = arith.select [[VAR_48_]], [[VAR_47_]], [[VAR_40_]] : vector<8xi1>, vector<8xf32>
// CHECK-DAG:         [[VAR_50_:%.+]] = vector.splat [[VAR_28_]] : vector<8xf32>
// CHECK:             [[VAR_51_:%.+]] = arith.addf [[VAR_49_]], [[VAR_50_]] : vector<8xf32>
// CHECK:             [[VAR_52_:%.+]] = arith.maxnumf [[VAR_51_]], [[VAR_cst_0_]] : vector<8xf32>
// CHECK:             [[VAR_53_:%.+]] = arith.minnumf [[VAR_52_]], [[VAR_cst_]] : vector<8xf32>
// CHECK:             [[VAR_54_:%.+]] = arith.fptoui [[VAR_53_]] : vector<8xf32> to vector<8xi8>
// CHECK:             [[VAR_55_:%.+]] = builtin.unrealized_conversion_cast [[VAR_54_]] : vector<8xi8> to vector<8xui8>
// CHECK:             vector.store [[VAR_55_]], [[VAR_reshape_21_]]{{.}}[[VAR_32_1_]]{{.}} : memref<4096xui8>, vector<8xui8>
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_]]_11, [[RES_]]_12 : memref<256x16xui8>, memref<f32>, memref<ui8>
// CHECK:         }
}

// -----


func.func @test_dynamic_quantize_linear_simd_and_scalar(%arg0: tensor<255x17xf32>) -> (tensor<255x17xui8>, tensor<f32>, tensor<ui8>) {
  %y, %y_scale, %y_zero_point = "onnx.DynamicQuantizeLinear"(%arg0) : (tensor<255x17xf32>) -> (tensor<255x17xui8>, tensor<f32>, tensor<ui8>)
  return %y, %y_scale, %y_zero_point:  tensor<255x17xui8>, tensor<f32>, tensor<ui8>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_dynamic_quantize_linear_simd_and_scalar
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<255x17xf32>) -> (memref<255x17xui8>, memref<f32>, memref<ui8>) {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<2.550000e+02> : vector<8xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<0.000000e+00> : vector<8xf32>
// CHECK-DAG:       [[VAR_cst_1_:%.+]] = arith.constant dense<5.000000e-01> : vector<8xf32>
// CHECK-DAG:       [[VAR_cst_2_:%.+]] = arith.constant dense<2.000000e+00> : vector<8xf32>
// CHECK-DAG:       [[VAR_cst_3_:%.+]] = arith.constant dense<1.000000e+00> : vector<8xf32>
// CHECK-DAG:       [[CST_5_dot_000000_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:       [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0x7F800000 : f32
// CHECK-DAG:       [[CST_4335_:%.+]] = arith.constant 4335 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_2_dot_550000_:%.+]] = arith.constant 2.550000e+02 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<255x17xui8>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() : memref<f32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() : memref<ui8>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_4335_]], [[RES_3_]][0] : memref<1xindex>
// CHECK:           [[RES_4_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.memset [[RES_4_]], [[CST_0_1_]] : memref<f32>
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 255, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 17){
// CHECK:             [[VAR_30_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_30_]]#0, [[VAR_30_]]#1] : memref<255x17xf32>
// CHECK-DAG:         [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]][] : memref<f32>
// CHECK:             [[VAR_33_:%.+]] = arith.minnumf [[LOAD_RES_4_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_33_]], [[RES_4_]][] : memref<f32>
// CHECK:           }
// CHECK:           [[RES_5_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.memset [[RES_5_]], [[CST_0_]] : memref<f32>
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 255, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 17){
// CHECK:             [[VAR_30_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_30_1_]]#0, [[VAR_30_1_]]#1] : memref<255x17xf32>
// CHECK-DAG:         [[LOAD_RES_4_MEM_1_:%.+]] = krnl.load [[RES_5_]][] : memref<f32>
// CHECK:             [[VAR_33_1_:%.+]] = arith.maxnumf [[LOAD_RES_4_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:             krnl.store [[VAR_33_1_]], [[RES_5_]][] : memref<f32>
// CHECK:           }
// CHECK-DAG:       [[LOAD_RES_4_MEM_2_:%.+]] = krnl.load [[RES_4_]][] : memref<f32>
// CHECK-DAG:       [[LOAD_RES_5_MEM_:%.+]] = krnl.load [[RES_5_]][] : memref<f32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.maxnumf [[LOAD_RES_5_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.minnumf [[LOAD_RES_4_MEM_2_]], [[CST_0_dot_000000_]] : f32
// CHECK:           [[VAR_6_:%.+]] = arith.subf [[VAR_4_]], [[VAR_5_]] : f32
// CHECK:           [[VAR_7_:%.+]] = arith.divf [[VAR_6_]], [[CST_2_dot_550000_]] : f32
// CHECK:           [[VAR_8_:%.+]] = arith.divf [[VAR_5_]], [[VAR_7_]] : f32
// CHECK:           [[VAR_9_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_8_]] : f32
// CHECK:           [[VAR_10_:%.+]] = arith.maxnumf [[VAR_9_]], [[CST_0_dot_000000_]] : f32
// CHECK:           [[VAR_11_:%.+]] = arith.minnumf [[VAR_10_]], [[CST_2_dot_550000_]] : f32
// CHECK:           [[VAR_12_:%.+]] = math.floor [[VAR_11_]] : f32
// CHECK:           [[VAR_13_:%.+]] = arith.subf [[VAR_11_]], [[VAR_12_]] : f32
// CHECK-DAG:       [[VAR_14_:%.+]] = arith.cmpf ogt, [[VAR_13_]], [[CST_5_dot_000000_]] : f32
// CHECK-DAG:       [[VAR_15_:%.+]] = arith.addf [[VAR_12_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_16_:%.+]] = arith.select [[VAR_14_]], [[VAR_15_]], [[VAR_12_]] : f32
// CHECK-DAG:       [[VAR_17_:%.+]] = arith.mulf [[VAR_12_]], [[CST_5_dot_000000_]] : f32
// CHECK:           [[VAR_18_:%.+]] = math.floor [[VAR_17_]] : f32
// CHECK:           [[VAR_19_:%.+]] = arith.mulf [[VAR_18_]], [[CST_2_dot_000000_]] : f32
// CHECK:           [[VAR_20_:%.+]] = arith.subf [[VAR_12_]], [[VAR_19_]] : f32
// CHECK-DAG:       [[VAR_21_:%.+]] = arith.cmpf oeq, [[VAR_20_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:       [[VAR_22_:%.+]] = arith.addf [[VAR_12_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_23_:%.+]] = arith.select [[VAR_21_]], [[VAR_22_]], [[VAR_12_]] : f32
// CHECK-DAG:       [[VAR_24_:%.+]] = arith.cmpf oeq, [[VAR_13_]], [[CST_5_dot_000000_]] : f32
// CHECK:           [[VAR_25_:%.+]] = arith.select [[VAR_24_]], [[VAR_23_]], [[VAR_16_]] : f32
// CHECK:           [[VAR_26_:%.+]] = arith.fptoui [[VAR_25_]] : f32 to i8
// CHECK:           [[VAR_27_:%.+]] = builtin.unrealized_conversion_cast [[VAR_26_]] : i8 to ui8
// CHECK:           krnl.store [[VAR_7_]], [[RES_1_]][] : memref<f32>
// CHECK:           krnl.store [[VAR_27_]], [[RES_2_]][] : memref<ui8>
// CHECK:           [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_4335_]], [[RES_6_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_6_]]) : (memref<255x17xf32>, memref<1xindex>) -> memref<4335xf32>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_4335_]], [[RES_7_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_18_:%.+]] = memref.reshape [[RES_]]([[RES_]]_17) : (memref<255x17xui8>, memref<1xindex>) -> memref<4335xui8>
// CHECK-DAG:       [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_2_]] 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_2_]] -> [[I_4_:%.+]] = 0 to 4328){
// CHECK:             [[VAR_30_2_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_30_2_]]{{.}} : memref<4335xf32>, vector<8xf32>
// CHECK-DAG:         [[LOAD_RES_4_MEM_1_:%.+]] = vector.splat [[VAR_7_]] : vector<8xf32>
// CHECK:             [[VAR_33_2_:%.+]] = arith.divf [[LOAD_PARAM_0_MEM_1_]], [[LOAD_RES_4_MEM_1_]] : vector<8xf32>
// CHECK:             [[VAR_34_:%.+]] = math.floor [[VAR_33_2_]] : vector<8xf32>
// CHECK:             [[VAR_35_:%.+]] = arith.subf [[VAR_33_2_]], [[VAR_34_]] : vector<8xf32>
// CHECK-DAG:         [[VAR_36_:%.+]] = arith.cmpf ogt, [[VAR_35_]], [[VAR_cst_1_]] : vector<8xf32>
// CHECK-DAG:         [[VAR_37_:%.+]] = arith.addf [[VAR_34_]], [[VAR_cst_3_]] : vector<8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_38_:%.+]] = arith.select [[VAR_36_]], [[VAR_37_]], [[VAR_34_]] : vector<8xi1>, vector<8xf32>
// CHECK-DAG:         [[VAR_39_:%.+]] = arith.mulf [[VAR_34_]], [[VAR_cst_1_]] : vector<8xf32>
// CHECK:             [[VAR_40_:%.+]] = math.floor [[VAR_39_]] : vector<8xf32>
// CHECK:             [[VAR_41_:%.+]] = arith.mulf [[VAR_40_]], [[VAR_cst_2_]] : vector<8xf32>
// CHECK:             [[VAR_42_:%.+]] = arith.subf [[VAR_34_]], [[VAR_41_]] : vector<8xf32>
// CHECK-DAG:         [[VAR_43_:%.+]] = arith.cmpf oeq, [[VAR_42_]], [[VAR_cst_3_]] : vector<8xf32>
// CHECK-DAG:         [[VAR_44_:%.+]] = arith.addf [[VAR_34_]], [[VAR_cst_3_]] : vector<8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_45_:%.+]] = arith.select [[VAR_43_]], [[VAR_44_]], [[VAR_34_]] : vector<8xi1>, vector<8xf32>
// CHECK-DAG:         [[VAR_46_:%.+]] = arith.cmpf oeq, [[VAR_35_]], [[VAR_cst_1_]] : vector<8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_47_:%.+]] = arith.select [[VAR_46_]], [[VAR_45_]], [[VAR_38_]] : vector<8xi1>, vector<8xf32>
// CHECK-DAG:         [[VAR_48_:%.+]] = vector.splat [[VAR_25_]] : vector<8xf32>
// CHECK:             [[VAR_49_:%.+]] = arith.addf [[VAR_47_]], [[VAR_48_]] : vector<8xf32>
// CHECK:             [[VAR_50_:%.+]] = arith.maxnumf [[VAR_49_]], [[VAR_cst_0_]] : vector<8xf32>
// CHECK:             [[VAR_51_:%.+]] = arith.minnumf [[VAR_50_]], [[VAR_cst_]] : vector<8xf32>
// CHECK:             [[VAR_52_:%.+]] = arith.fptoui [[VAR_51_]] : vector<8xf32> to vector<8xi8>
// CHECK:             [[VAR_53_:%.+]] = builtin.unrealized_conversion_cast [[VAR_52_]] : vector<8xi8> to vector<8xui8>
// CHECK:             vector.store [[VAR_53_]], [[VAR_reshape_18_]]{{.}}[[VAR_30_2_]]{{.}} : memref<4335xui8>, vector<8xui8>
// CHECK:           }
// CHECK:           [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[I_5_:%.+]] = 4328 to 4335){
// CHECK:             [[VAR_30_3_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_0_MEM_1_1_:%.+]] = krnl.load [[VAR_reshape_]]{{.}}[[VAR_30_3_]]{{.}} : memref<4335xf32>
// CHECK:             [[LOAD_RES_4_MEM_1_1_:%.+]] = arith.divf [[LOAD_PARAM_0_MEM_1_1_]], [[VAR_7_]] : f32
// CHECK:             [[VAR_33_3_:%.+]] = math.floor [[LOAD_RES_4_MEM_1_1_]] : f32
// CHECK:             [[VAR_34_1_:%.+]] = arith.subf [[LOAD_RES_4_MEM_1_1_]], [[VAR_33_3_]] : f32
// CHECK-DAG:         [[VAR_35_1_:%.+]] = arith.cmpf ogt, [[VAR_34_1_]], [[CST_5_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_36_1_:%.+]] = arith.addf [[VAR_33_3_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_37_1_:%.+]] = arith.select [[VAR_35_1_]], [[VAR_36_1_]], [[VAR_33_3_]] : f32
// CHECK-DAG:         [[VAR_38_1_:%.+]] = arith.mulf [[VAR_33_3_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_39_1_:%.+]] = math.floor [[VAR_38_1_]] : f32
// CHECK:             [[VAR_40_1_:%.+]] = arith.mulf [[VAR_39_1_]], [[CST_2_dot_000000_]] : f32
// CHECK:             [[VAR_41_1_:%.+]] = arith.subf [[VAR_33_3_]], [[VAR_40_1_]] : f32
// CHECK-DAG:         [[VAR_42_1_:%.+]] = arith.cmpf oeq, [[VAR_41_1_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_43_1_:%.+]] = arith.addf [[VAR_33_3_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_44_1_:%.+]] = arith.select [[VAR_42_1_]], [[VAR_43_1_]], [[VAR_33_3_]] : f32
// CHECK-DAG:         [[VAR_45_1_:%.+]] = arith.cmpf oeq, [[VAR_34_1_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_46_1_:%.+]] = arith.select [[VAR_45_1_]], [[VAR_44_1_]], [[VAR_37_1_]] : f32
// CHECK:             [[VAR_47_1_:%.+]] = arith.addf [[VAR_46_1_]], [[VAR_25_]] : f32
// CHECK:             [[VAR_48_1_:%.+]] = arith.maxnumf [[VAR_47_1_]], [[CST_0_dot_000000_]] : f32
// CHECK:             [[VAR_49_1_:%.+]] = arith.minnumf [[VAR_48_1_]], [[CST_2_dot_550000_]] : f32
// CHECK:             [[VAR_50_1_:%.+]] = arith.fptoui [[VAR_49_1_]] : f32 to i8
// CHECK:             [[VAR_51_1_:%.+]] = builtin.unrealized_conversion_cast [[VAR_50_1_]] : i8 to ui8
// CHECK:             krnl.store [[VAR_51_1_]], [[VAR_reshape_18_]]{{.}}[[VAR_30_3_]]{{.}} : memref<4335xui8>
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_]]_11, [[RES_]]_12 : memref<255x17xui8>, memref<f32>, memref<ui8>
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
// CHECK-DAG:       [[VAR_cst_1_:%.+]] = arith.constant dense<5.000000e-01> : vector<8xf32>
// CHECK-DAG:       [[VAR_cst_2_:%.+]] = arith.constant dense<2.000000e+00> : vector<8xf32>
// CHECK-DAG:       [[VAR_cst_3_:%.+]] = arith.constant dense<1.000000e+00> : vector<8xf32>
// CHECK-DAG:       [[VAR_cst_4_:%.+]] = arith.constant dense<0xFF800000> : vector<8xf32>
// CHECK-DAG:       [[VAR_cst_5_:%.+]] = arith.constant dense<0x7F800000> : vector<8xf32>
// CHECK-DAG:       [[CST_5_dot_000000_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:       [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
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
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() : memref<f32>
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<8xf32>
// CHECK:           vector.store [[VAR_cst_5_]], [[RES_5_]]{{.}}[[CST_0_]]{{.}} : memref<8xf32>, vector<8xf32>
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() : memref<f32>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc() {{.*}}: memref<8xf32>
// CHECK:           vector.store [[VAR_cst_4_]], [[RES_7_]]{{.}}[[CST_0_]]{{.}} : memref<8xf32>, vector<8xf32>
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = [[CST_0_]] to [[CST_8_]]){
// CHECK:             [[VAR_32_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_32_]]{{.}} : memref<8xf32>, vector<8xf32>
// CHECK-DAG:         [[LOAD_RES_5_MEM_:%.+]] = vector.load [[RES_5_]]{{.}}[[CST_0_]]{{.}} : memref<8xf32>, vector<8xf32>
// CHECK:             [[VAR_35_:%.+]] = arith.minnumf [[LOAD_RES_5_MEM_]], [[LOAD_VAR_reshape_MEM_]] : vector<8xf32>
// CHECK:             vector.store [[VAR_35_]], [[RES_5_]]{{.}}[[CST_0_]]{{.}} : memref<8xf32>, vector<8xf32>
// CHECK:             [[LOAD_RES_7_MEM_:%.+]] = vector.load [[RES_7_]]{{.}}[[CST_0_]]{{.}} : memref<8xf32>, vector<8xf32>
// CHECK:             [[VAR_37_:%.+]] = arith.maxnumf [[LOAD_RES_7_MEM_]], [[LOAD_VAR_reshape_MEM_]] : vector<8xf32>
// CHECK:             vector.store [[VAR_37_]], [[RES_7_]]{{.}}[[CST_0_]]{{.}} : memref<8xf32>, vector<8xf32>
// CHECK:           }
// CHECK:           [[LOAD_RES_5_MEM_1_:%.+]] = vector.load [[RES_5_]]{{.}}[[CST_0_]]{{.}} : memref<8xf32>, vector<8xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = vector.reduction <minnumf>, [[LOAD_RES_5_MEM_1_]] : vector<8xf32> into f32
// CHECK-DAG:       [[LOAD_RES_7_MEM_1_:%.+]] = vector.load [[RES_7_]]{{.}}[[CST_0_]]{{.}} : memref<8xf32>, vector<8xf32>
// CHECK:           [[VAR_4_:%.+]] = vector.reduction <maxnumf>, [[LOAD_RES_7_MEM_1_]] : vector<8xf32> into f32
// CHECK:           krnl.store [[VAR_2_]], [[RES_4_]][] : memref<f32>
// CHECK:           krnl.store [[VAR_4_]], [[RES_6_]][] : memref<f32>
// CHECK-DAG:       [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]][] : memref<f32>
// CHECK-DAG:       [[LOAD_RES_6_MEM_:%.+]] = krnl.load [[RES_6_]][] : memref<f32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_7_:%.+]] = arith.maxnumf [[LOAD_RES_6_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.minnumf [[LOAD_RES_4_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK:           [[VAR_9_:%.+]] = arith.subf [[VAR_7_]], [[VAR_8_]] : f32
// CHECK:           [[VAR_10_:%.+]] = arith.divf [[VAR_9_]], [[CST_2_dot_550000_]] : f32
// CHECK:           [[VAR_11_:%.+]] = arith.divf [[VAR_8_]], [[VAR_10_]] : f32
// CHECK:           [[VAR_12_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_11_]] : f32
// CHECK:           [[VAR_13_:%.+]] = arith.maxnumf [[VAR_12_]], [[CST_0_dot_000000_]] : f32
// CHECK:           [[VAR_14_:%.+]] = arith.minnumf [[VAR_13_]], [[CST_2_dot_550000_]] : f32
// CHECK:           [[VAR_15_:%.+]] = math.floor [[VAR_14_]] : f32
// CHECK:           [[VAR_16_:%.+]] = arith.subf [[VAR_14_]], [[VAR_15_]] : f32
// CHECK-DAG:       [[VAR_17_:%.+]] = arith.cmpf ogt, [[VAR_16_]], [[CST_5_dot_000000_]] : f32
// CHECK-DAG:       [[VAR_18_:%.+]] = arith.addf [[VAR_15_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_19_:%.+]] = arith.select [[VAR_17_]], [[VAR_18_]], [[VAR_15_]] : f32
// CHECK-DAG:       [[VAR_20_:%.+]] = arith.mulf [[VAR_15_]], [[CST_5_dot_000000_]] : f32
// CHECK:           [[VAR_21_:%.+]] = math.floor [[VAR_20_]] : f32
// CHECK:           [[VAR_22_:%.+]] = arith.mulf [[VAR_21_]], [[CST_2_dot_000000_]] : f32
// CHECK:           [[VAR_23_:%.+]] = arith.subf [[VAR_15_]], [[VAR_22_]] : f32
// CHECK-DAG:       [[VAR_24_:%.+]] = arith.cmpf oeq, [[VAR_23_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:       [[VAR_25_:%.+]] = arith.addf [[VAR_15_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_26_:%.+]] = arith.select [[VAR_24_]], [[VAR_25_]], [[VAR_15_]] : f32
// CHECK-DAG:       [[VAR_27_:%.+]] = arith.cmpf oeq, [[VAR_16_]], [[CST_5_dot_000000_]] : f32
// CHECK:           [[VAR_28_:%.+]] = arith.select [[VAR_27_]], [[VAR_26_]], [[VAR_19_]] : f32
// CHECK:           [[VAR_29_:%.+]] = arith.fptoui [[VAR_28_]] : f32 to i8
// CHECK:           [[VAR_30_:%.+]] = builtin.unrealized_conversion_cast [[VAR_29_]] : i8 to ui8
// CHECK:           krnl.store [[VAR_10_]], [[RES_1_]][] : memref<f32>
// CHECK:           krnl.store [[VAR_30_]], [[RES_2_]][] : memref<ui8>
// CHECK:           [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_8_]], [[RES_8_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_19_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_8_]]) : (memref<1x8xf32>, memref<1xindex>) -> memref<8xf32>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_8_]], [[RES_9_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_21_:%.+]] = memref.reshape [[RES_]]([[RES_]]_20) : (memref<1x8xui8>, memref<1xindex>) -> memref<8xui8>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_1_]] 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 8){
// CHECK:             [[VAR_32_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_1_:%.+]] = vector.load [[VAR_reshape_19_]]{{.}}[[VAR_32_1_]]{{.}} : memref<8xf32>, vector<8xf32>
// CHECK-DAG:         [[LOAD_RES_5_MEM_2_:%.+]] = vector.splat [[VAR_10_]] : vector<8xf32>
// CHECK:             [[VAR_35_1_:%.+]] = arith.divf [[LOAD_VAR_reshape_MEM_1_]], [[LOAD_RES_5_MEM_2_]] : vector<8xf32>
// CHECK:             [[LOAD_RES_7_MEM_2_:%.+]] = math.floor [[VAR_35_1_]] : vector<8xf32>
// CHECK:             [[VAR_37_1_:%.+]] = arith.subf [[VAR_35_1_]], [[LOAD_RES_7_MEM_2_]] : vector<8xf32>
// CHECK-DAG:         [[VAR_38_:%.+]] = arith.cmpf ogt, [[VAR_37_1_]], [[VAR_cst_1_]] : vector<8xf32>
// CHECK-DAG:         [[VAR_39_:%.+]] = arith.addf [[LOAD_RES_7_MEM_2_]], [[VAR_cst_3_]] : vector<8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_40_:%.+]] = arith.select [[VAR_38_]], [[VAR_39_]], [[LOAD_RES_7_MEM_2_]] : vector<8xi1>, vector<8xf32>
// CHECK-DAG:         [[VAR_41_:%.+]] = arith.mulf [[LOAD_RES_7_MEM_2_]], [[VAR_cst_1_]] : vector<8xf32>
// CHECK:             [[VAR_42_:%.+]] = math.floor [[VAR_41_]] : vector<8xf32>
// CHECK:             [[VAR_43_:%.+]] = arith.mulf [[VAR_42_]], [[VAR_cst_2_]] : vector<8xf32>
// CHECK:             [[VAR_44_:%.+]] = arith.subf [[LOAD_RES_7_MEM_2_]], [[VAR_43_]] : vector<8xf32>
// CHECK-DAG:         [[VAR_45_:%.+]] = arith.cmpf oeq, [[VAR_44_]], [[VAR_cst_3_]] : vector<8xf32>
// CHECK-DAG:         [[VAR_46_:%.+]] = arith.addf [[LOAD_RES_7_MEM_2_]], [[VAR_cst_3_]] : vector<8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_47_:%.+]] = arith.select [[VAR_45_]], [[VAR_46_]], [[LOAD_RES_7_MEM_2_]] : vector<8xi1>, vector<8xf32>
// CHECK-DAG:         [[VAR_48_:%.+]] = arith.cmpf oeq, [[VAR_37_1_]], [[VAR_cst_1_]] : vector<8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_49_:%.+]] = arith.select [[VAR_48_]], [[VAR_47_]], [[VAR_40_]] : vector<8xi1>, vector<8xf32>
// CHECK-DAG:         [[VAR_50_:%.+]] = vector.splat [[VAR_28_]] : vector<8xf32>
// CHECK:             [[VAR_51_:%.+]] = arith.addf [[VAR_49_]], [[VAR_50_]] : vector<8xf32>
// CHECK:             [[VAR_52_:%.+]] = arith.maxnumf [[VAR_51_]], [[VAR_cst_0_]] : vector<8xf32>
// CHECK:             [[VAR_53_:%.+]] = arith.minnumf [[VAR_52_]], [[VAR_cst_]] : vector<8xf32>
// CHECK:             [[VAR_54_:%.+]] = arith.fptoui [[VAR_53_]] : vector<8xf32> to vector<8xi8>
// CHECK:             [[VAR_55_:%.+]] = builtin.unrealized_conversion_cast [[VAR_54_]] : vector<8xi8> to vector<8xui8>
// CHECK:             vector.store [[VAR_55_]], [[VAR_reshape_21_]]{{.}}[[VAR_32_1_]]{{.}} : memref<8xui8>, vector<8xui8>
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_]]_11, [[RES_]]_12 : memref<1x8xui8>, memref<f32>, memref<ui8>
// CHECK:         }
}

