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
// CHECK-DAG:       [[VAR_cst_4_:%.+]] = arith.constant dense<0xFF800000> : vector<16xf32>
// CHECK-DAG:       [[VAR_cst_5_:%.+]] = arith.constant dense<0x7F800000> : vector<16xf32>
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
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<16xf32>
// CHECK:           vector.store [[VAR_cst_5_]], [[RES_5_]]{{.}}[[CST_0_]]{{.}} : memref<16xf32>, vector<16xf32>
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() : memref<f32>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc() {{.*}}: memref<16xf32>
// CHECK:           vector.store [[VAR_cst_4_]], [[RES_7_]]{{.}}[[CST_0_]]{{.}} : memref<16xf32>, vector<16xf32>
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 16 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = [[CST_0_]] to [[CST_4096_]]){
// CHECK:             [[VAR_32_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_32_]]{{.}} : memref<4096xf32>, vector<16xf32>
// CHECK-DAG:         [[LOAD_RES_5_MEM_:%.+]] = vector.load [[RES_5_]]{{.}}[[CST_0_]]{{.}} : memref<16xf32>, vector<16xf32>
// CHECK:             [[VAR_35_:%.+]] = arith.minnumf [[LOAD_RES_5_MEM_]], [[LOAD_VAR_reshape_MEM_]] : vector<16xf32>
// CHECK:             vector.store [[VAR_35_]], [[RES_5_]]{{.}}[[CST_0_]]{{.}} : memref<16xf32>, vector<16xf32>
// CHECK:             [[LOAD_RES_7_MEM_:%.+]] = vector.load [[RES_7_]]{{.}}[[CST_0_]]{{.}} : memref<16xf32>, vector<16xf32>
// CHECK:             [[VAR_37_:%.+]] = arith.maxnumf [[LOAD_RES_7_MEM_]], [[LOAD_VAR_reshape_MEM_]] : vector<16xf32>
// CHECK:             vector.store [[VAR_37_]], [[RES_7_]]{{.}}[[CST_0_]]{{.}} : memref<16xf32>, vector<16xf32>
// CHECK:           }
// CHECK:           [[LOAD_RES_5_MEM_1_:%.+]] = vector.load [[RES_5_]]{{.}}[[CST_0_]]{{.}} : memref<16xf32>, vector<16xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = vector.reduction <minnumf>, [[LOAD_RES_5_MEM_1_]] : vector<16xf32> into f32
// CHECK-DAG:       [[LOAD_RES_7_MEM_1_:%.+]] = vector.load [[RES_7_]]{{.}}[[CST_0_]]{{.}} : memref<16xf32>, vector<16xf32>
// CHECK:           [[VAR_4_:%.+]] = vector.reduction <maxnumf>, [[LOAD_RES_7_MEM_1_]] : vector<16xf32> into f32
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
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 256){
// CHECK-DAG:         [[VAR_32_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_2_]] 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__1_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 16){
// CHECK:               [[LOAD_RES_5_MEM_2_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[VAR_35_1_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_32_1_]], [[LOAD_RES_5_MEM_2_]]{{.}} : memref<256x16xf32>, vector<8xf32>
// CHECK-DAG:           [[LOAD_RES_7_MEM_2_:%.+]] = vector.splat [[VAR_10_]] : vector<8xf32>
// CHECK:               [[VAR_37_1_:%.+]] = arith.divf [[VAR_35_1_]], [[LOAD_RES_7_MEM_2_]] : vector<8xf32>
// CHECK:               [[VAR_38_:%.+]] = math.floor [[VAR_37_1_]] : vector<8xf32>
// CHECK:               [[VAR_39_:%.+]] = arith.subf [[VAR_37_1_]], [[VAR_38_]] : vector<8xf32>
// CHECK-DAG:           [[VAR_40_:%.+]] = arith.cmpf ogt, [[VAR_39_]], [[VAR_cst_1_]] : vector<8xf32>
// CHECK-DAG:           [[VAR_41_:%.+]] = arith.addf [[VAR_38_]], [[VAR_cst_3_]] : vector<8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_42_:%.+]] = arith.select [[VAR_40_]], [[VAR_41_]], [[VAR_38_]] : vector<8xi1>, vector<8xf32>
// CHECK-DAG:           [[VAR_43_:%.+]] = arith.mulf [[VAR_38_]], [[VAR_cst_1_]] : vector<8xf32>
// CHECK:               [[VAR_44_:%.+]] = math.floor [[VAR_43_]] : vector<8xf32>
// CHECK:               [[VAR_45_:%.+]] = arith.mulf [[VAR_44_]], [[VAR_cst_2_]] : vector<8xf32>
// CHECK:               [[VAR_46_:%.+]] = arith.subf [[VAR_38_]], [[VAR_45_]] : vector<8xf32>
// CHECK-DAG:           [[VAR_47_:%.+]] = arith.cmpf oeq, [[VAR_46_]], [[VAR_cst_3_]] : vector<8xf32>
// CHECK-DAG:           [[VAR_48_:%.+]] = arith.addf [[VAR_38_]], [[VAR_cst_3_]] : vector<8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_49_:%.+]] = arith.select [[VAR_47_]], [[VAR_48_]], [[VAR_38_]] : vector<8xi1>, vector<8xf32>
// CHECK-DAG:           [[VAR_50_:%.+]] = arith.cmpf oeq, [[VAR_39_]], [[VAR_cst_1_]] : vector<8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_51_:%.+]] = arith.select [[VAR_50_]], [[VAR_49_]], [[VAR_42_]] : vector<8xi1>, vector<8xf32>
// CHECK-DAG:           [[VAR_52_:%.+]] = vector.splat [[VAR_28_]] : vector<8xf32>
// CHECK:               [[VAR_53_:%.+]] = arith.addf [[VAR_51_]], [[VAR_52_]] : vector<8xf32>
// CHECK:               [[VAR_54_:%.+]] = arith.maxnumf [[VAR_53_]], [[VAR_cst_0_]] : vector<8xf32>
// CHECK:               [[VAR_55_:%.+]] = arith.minnumf [[VAR_54_]], [[VAR_cst_]] : vector<8xf32>
// CHECK:               [[VAR_56_:%.+]] = arith.fptoui [[VAR_55_]] : vector<8xf32> to vector<8xi8>
// CHECK:               [[VAR_57_:%.+]] = builtin.unrealized_conversion_cast [[VAR_56_]] : vector<8xi8> to vector<8xui8>
// CHECK:               vector.store [[VAR_57_]], [[RES_]]{{.}}[[VAR_32_1_]], [[LOAD_RES_5_MEM_2_]]{{.}} : memref<256x16xui8>, vector<8xui8>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_]]_11, [[RES_]]_12 : memref<256x16xui8>, memref<f32>, memref<ui8>
// CHECK:         }
}

// -----


func.func @test_dynamic_quantize_linear_simd_and_scalar(%arg0: tensor<256x17xf32>) -> (tensor<256x17xui8>, tensor<f32>, tensor<ui8>) {
  %y, %y_scale, %y_zero_point = "onnx.DynamicQuantizeLinear"(%arg0) : (tensor<256x17xf32>) -> (tensor<256x17xui8>, tensor<f32>, tensor<ui8>)
  return %y, %y_scale, %y_zero_point:  tensor<256x17xui8>, tensor<f32>, tensor<ui8>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_dynamic_quantize_linear_simd_and_scalar
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<256x17xf32>) -> (memref<256x17xui8>, memref<f32>, memref<ui8>) {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<2.550000e+02> : vector<8xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<0.000000e+00> : vector<8xf32>
// CHECK-DAG:       [[VAR_cst_1_:%.+]] = arith.constant dense<5.000000e-01> : vector<8xf32>
// CHECK-DAG:       [[VAR_cst_2_:%.+]] = arith.constant dense<2.000000e+00> : vector<8xf32>
// CHECK-DAG:       [[VAR_cst_3_:%.+]] = arith.constant dense<1.000000e+00> : vector<8xf32>
// CHECK-DAG:       [[VAR_cst_4_:%.+]] = arith.constant dense<0xFF800000> : vector<16xf32>
// CHECK-DAG:       [[VAR_cst_5_:%.+]] = arith.constant dense<0x7F800000> : vector<16xf32>
// CHECK-DAG:       [[CST_5_dot_000000_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:       [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_4352_:%.+]] = arith.constant 4352 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_2_dot_550000_:%.+]] = arith.constant 2.550000e+02 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<256x17xui8>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() : memref<f32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() : memref<ui8>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_4352_]], [[RES_3_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_3_]]) : (memref<256x17xf32>, memref<1xindex>) -> memref<4352xf32>
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() : memref<f32>
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<16xf32>
// CHECK:           vector.store [[VAR_cst_5_]], [[RES_5_]]{{.}}[[CST_0_]]{{.}} : memref<16xf32>, vector<16xf32>
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() : memref<f32>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc() {{.*}}: memref<16xf32>
// CHECK:           vector.store [[VAR_cst_4_]], [[RES_7_]]{{.}}[[CST_0_]]{{.}} : memref<16xf32>, vector<16xf32>
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 16 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = [[CST_0_]] to [[CST_4352_]]){
// CHECK:             [[VAR_32_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_32_]]{{.}} : memref<4352xf32>, vector<16xf32>
// CHECK-DAG:         [[LOAD_RES_5_MEM_:%.+]] = vector.load [[RES_5_]]{{.}}[[CST_0_]]{{.}} : memref<16xf32>, vector<16xf32>
// CHECK:             [[VAR_35_:%.+]] = arith.minnumf [[LOAD_RES_5_MEM_]], [[LOAD_VAR_reshape_MEM_]] : vector<16xf32>
// CHECK:             vector.store [[VAR_35_]], [[RES_5_]]{{.}}[[CST_0_]]{{.}} : memref<16xf32>, vector<16xf32>
// CHECK:             [[LOAD_RES_7_MEM_:%.+]] = vector.load [[RES_7_]]{{.}}[[CST_0_]]{{.}} : memref<16xf32>, vector<16xf32>
// CHECK:             [[VAR_37_:%.+]] = arith.maxnumf [[LOAD_RES_7_MEM_]], [[LOAD_VAR_reshape_MEM_]] : vector<16xf32>
// CHECK:             vector.store [[VAR_37_]], [[RES_7_]]{{.}}[[CST_0_]]{{.}} : memref<16xf32>, vector<16xf32>
// CHECK:           }
// CHECK:           [[LOAD_RES_5_MEM_1_:%.+]] = vector.load [[RES_5_]]{{.}}[[CST_0_]]{{.}} : memref<16xf32>, vector<16xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = vector.reduction <minnumf>, [[LOAD_RES_5_MEM_1_]] : vector<16xf32> into f32
// CHECK-DAG:       [[LOAD_RES_7_MEM_1_:%.+]] = vector.load [[RES_7_]]{{.}}[[CST_0_]]{{.}} : memref<16xf32>, vector<16xf32>
// CHECK:           [[VAR_4_:%.+]] = vector.reduction <maxnumf>, [[LOAD_RES_7_MEM_1_]] : vector<16xf32> into f32
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
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 256){
// CHECK-DAG:         [[VAR_32_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_2_]] 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__1_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 10){
// CHECK:               [[VAR_35_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_RES_7_MEM_2_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_32_1_]], [[VAR_35_1_]]{{.}} : memref<256x17xf32>, vector<8xf32>
// CHECK-DAG:           [[VAR_37_1_:%.+]] = vector.splat [[VAR_10_]] : vector<8xf32>
// CHECK:               [[VAR_38_:%.+]] = arith.divf [[LOAD_RES_7_MEM_2_]], [[VAR_37_1_]] : vector<8xf32>
// CHECK:               [[VAR_39_:%.+]] = math.floor [[VAR_38_]] : vector<8xf32>
// CHECK:               [[VAR_40_:%.+]] = arith.subf [[VAR_38_]], [[VAR_39_]] : vector<8xf32>
// CHECK-DAG:           [[VAR_41_:%.+]] = arith.cmpf ogt, [[VAR_40_]], [[VAR_cst_1_]] : vector<8xf32>
// CHECK-DAG:           [[VAR_42_:%.+]] = arith.addf [[VAR_39_]], [[VAR_cst_3_]] : vector<8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_43_:%.+]] = arith.select [[VAR_41_]], [[VAR_42_]], [[VAR_39_]] : vector<8xi1>, vector<8xf32>
// CHECK-DAG:           [[VAR_44_:%.+]] = arith.mulf [[VAR_39_]], [[VAR_cst_1_]] : vector<8xf32>
// CHECK:               [[VAR_45_:%.+]] = math.floor [[VAR_44_]] : vector<8xf32>
// CHECK:               [[VAR_46_:%.+]] = arith.mulf [[VAR_45_]], [[VAR_cst_2_]] : vector<8xf32>
// CHECK:               [[VAR_47_:%.+]] = arith.subf [[VAR_39_]], [[VAR_46_]] : vector<8xf32>
// CHECK-DAG:           [[VAR_48_:%.+]] = arith.cmpf oeq, [[VAR_47_]], [[VAR_cst_3_]] : vector<8xf32>
// CHECK-DAG:           [[VAR_49_:%.+]] = arith.addf [[VAR_39_]], [[VAR_cst_3_]] : vector<8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_50_:%.+]] = arith.select [[VAR_48_]], [[VAR_49_]], [[VAR_39_]] : vector<8xi1>, vector<8xf32>
// CHECK-DAG:           [[VAR_51_:%.+]] = arith.cmpf oeq, [[VAR_40_]], [[VAR_cst_1_]] : vector<8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_52_:%.+]] = arith.select [[VAR_51_]], [[VAR_50_]], [[VAR_43_]] : vector<8xi1>, vector<8xf32>
// CHECK-DAG:           [[VAR_53_:%.+]] = vector.splat [[VAR_28_]] : vector<8xf32>
// CHECK:               [[VAR_54_:%.+]] = arith.addf [[VAR_52_]], [[VAR_53_]] : vector<8xf32>
// CHECK:               [[VAR_55_:%.+]] = arith.maxnumf [[VAR_54_]], [[VAR_cst_0_]] : vector<8xf32>
// CHECK:               [[VAR_56_:%.+]] = arith.minnumf [[VAR_55_]], [[VAR_cst_]] : vector<8xf32>
// CHECK:               [[VAR_57_:%.+]] = arith.fptoui [[VAR_56_]] : vector<8xf32> to vector<8xi8>
// CHECK:               [[VAR_58_:%.+]] = builtin.unrealized_conversion_cast [[VAR_57_]] : vector<8xi8> to vector<8xui8>
// CHECK:               vector.store [[VAR_58_]], [[RES_]]{{.}}[[VAR_32_1_]], [[VAR_35_1_]]{{.}} : memref<256x17xui8>, vector<8xui8>
// CHECK:             }
// CHECK:             [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[I_3_:%.+]] = 16 to 17){
// CHECK:               [[VAR_35_2_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_RES_7_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_32_1_]], [[VAR_35_2_]]{{.}} : memref<256x17xf32>
// CHECK:               [[VAR_37_2_:%.+]] = arith.divf [[LOAD_RES_7_MEM_2_]], [[VAR_10_]] : f32
// CHECK:               [[VAR_38_1_:%.+]] = math.floor [[VAR_37_2_]] : f32
// CHECK:               [[VAR_39_1_:%.+]] = arith.subf [[VAR_37_2_]], [[VAR_38_1_]] : f32
// CHECK-DAG:           [[VAR_40_1_:%.+]] = arith.cmpf ogt, [[VAR_39_1_]], [[CST_5_dot_000000_]] : f32
// CHECK-DAG:           [[VAR_41_1_:%.+]] = arith.addf [[VAR_38_1_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_42_1_:%.+]] = arith.select [[VAR_40_1_]], [[VAR_41_1_]], [[VAR_38_1_]] : f32
// CHECK-DAG:           [[VAR_43_1_:%.+]] = arith.mulf [[VAR_38_1_]], [[CST_5_dot_000000_]] : f32
// CHECK:               [[VAR_44_1_:%.+]] = math.floor [[VAR_43_1_]] : f32
// CHECK:               [[VAR_45_1_:%.+]] = arith.mulf [[VAR_44_1_]], [[CST_2_dot_000000_]] : f32
// CHECK:               [[VAR_46_1_:%.+]] = arith.subf [[VAR_38_1_]], [[VAR_45_1_]] : f32
// CHECK-DAG:           [[VAR_47_1_:%.+]] = arith.cmpf oeq, [[VAR_46_1_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:           [[VAR_48_1_:%.+]] = arith.addf [[VAR_38_1_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_49_1_:%.+]] = arith.select [[VAR_47_1_]], [[VAR_48_1_]], [[VAR_38_1_]] : f32
// CHECK-DAG:           [[VAR_50_1_:%.+]] = arith.cmpf oeq, [[VAR_39_1_]], [[CST_5_dot_000000_]] : f32
// CHECK:               [[VAR_51_1_:%.+]] = arith.select [[VAR_50_1_]], [[VAR_49_1_]], [[VAR_42_1_]] : f32
// CHECK:               [[VAR_52_1_:%.+]] = arith.addf [[VAR_51_1_]], [[VAR_28_]] : f32
// CHECK:               [[VAR_53_1_:%.+]] = arith.maxnumf [[VAR_52_1_]], [[CST_0_dot_000000_]] : f32
// CHECK:               [[VAR_54_1_:%.+]] = arith.minnumf [[VAR_53_1_]], [[CST_2_dot_550000_]] : f32
// CHECK:               [[VAR_55_1_:%.+]] = arith.fptoui [[VAR_54_1_]] : f32 to i8
// CHECK:               [[VAR_56_1_:%.+]] = builtin.unrealized_conversion_cast [[VAR_55_1_]] : i8 to ui8
// CHECK:               krnl.store [[VAR_56_1_]], [[RES_]]{{.}}[[VAR_32_1_]], [[VAR_35_2_]]{{.}} : memref<256x17xui8>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_]]_11, [[RES_]]_12 : memref<256x17xui8>, memref<f32>, memref<ui8>
// CHECK:         }
}

// -----


func.func @test_dynamic_quantize_linear_reduced_simd_only(%arg0: tensor<256x4xf32>) -> (tensor<256x4xui8>, tensor<f32>, tensor<ui8>) {
  %y, %y_scale, %y_zero_point = "onnx.DynamicQuantizeLinear"(%arg0) : (tensor<256x4xf32>) -> (tensor<256x4xui8>, tensor<f32>, tensor<ui8>)
  return %y, %y_scale, %y_zero_point:  tensor<256x4xui8>, tensor<f32>, tensor<ui8>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_dynamic_quantize_linear_reduced_simd_only
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<256x4xf32>) -> (memref<256x4xui8>, memref<f32>, memref<ui8>) {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<2.550000e+02> : vector<4xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
// CHECK-DAG:       [[VAR_cst_1_:%.+]] = arith.constant dense<5.000000e-01> : vector<4xf32>
// CHECK-DAG:       [[VAR_cst_2_:%.+]] = arith.constant dense<2.000000e+00> : vector<4xf32>
// CHECK-DAG:       [[VAR_cst_3_:%.+]] = arith.constant dense<1.000000e+00> : vector<4xf32>
// CHECK-DAG:       [[VAR_cst_4_:%.+]] = arith.constant dense<0xFF800000> : vector<16xf32>
// CHECK-DAG:       [[VAR_cst_5_:%.+]] = arith.constant dense<0x7F800000> : vector<16xf32>
// CHECK-DAG:       [[CST_5_dot_000000_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:       [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1024_:%.+]] = arith.constant 1024 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_2_dot_550000_:%.+]] = arith.constant 2.550000e+02 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<256x4xui8>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() : memref<f32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() : memref<ui8>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_1024_]], [[RES_3_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_3_]]) : (memref<256x4xf32>, memref<1xindex>) -> memref<1024xf32>
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() : memref<f32>
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<16xf32>
// CHECK:           vector.store [[VAR_cst_5_]], [[RES_5_]]{{.}}[[CST_0_]]{{.}} : memref<16xf32>, vector<16xf32>
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() : memref<f32>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc() {{.*}}: memref<16xf32>
// CHECK:           vector.store [[VAR_cst_4_]], [[RES_7_]]{{.}}[[CST_0_]]{{.}} : memref<16xf32>, vector<16xf32>
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 16 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = [[CST_0_]] to [[CST_1024_]]){
// CHECK:             [[VAR_32_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_32_]]{{.}} : memref<1024xf32>, vector<16xf32>
// CHECK-DAG:         [[LOAD_RES_5_MEM_:%.+]] = vector.load [[RES_5_]]{{.}}[[CST_0_]]{{.}} : memref<16xf32>, vector<16xf32>
// CHECK:             [[VAR_35_:%.+]] = arith.minnumf [[LOAD_RES_5_MEM_]], [[LOAD_VAR_reshape_MEM_]] : vector<16xf32>
// CHECK:             vector.store [[VAR_35_]], [[RES_5_]]{{.}}[[CST_0_]]{{.}} : memref<16xf32>, vector<16xf32>
// CHECK:             [[LOAD_RES_7_MEM_:%.+]] = vector.load [[RES_7_]]{{.}}[[CST_0_]]{{.}} : memref<16xf32>, vector<16xf32>
// CHECK:             [[VAR_37_:%.+]] = arith.maxnumf [[LOAD_RES_7_MEM_]], [[LOAD_VAR_reshape_MEM_]] : vector<16xf32>
// CHECK:             vector.store [[VAR_37_]], [[RES_7_]]{{.}}[[CST_0_]]{{.}} : memref<16xf32>, vector<16xf32>
// CHECK:           }
// CHECK:           [[LOAD_RES_5_MEM_1_:%.+]] = vector.load [[RES_5_]]{{.}}[[CST_0_]]{{.}} : memref<16xf32>, vector<16xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = vector.reduction <minnumf>, [[LOAD_RES_5_MEM_1_]] : vector<16xf32> into f32
// CHECK-DAG:       [[LOAD_RES_7_MEM_1_:%.+]] = vector.load [[RES_7_]]{{.}}[[CST_0_]]{{.}} : memref<16xf32>, vector<16xf32>
// CHECK:           [[VAR_4_:%.+]] = vector.reduction <maxnumf>, [[LOAD_RES_7_MEM_1_]] : vector<16xf32> into f32
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
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 256){
// CHECK-DAG:         [[VAR_32_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_2_]] 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__1_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 4){
// CHECK:               [[LOAD_RES_5_MEM_2_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[VAR_35_1_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[VAR_32_1_]], [[LOAD_RES_5_MEM_2_]]{{.}} : memref<256x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_7_MEM_2_:%.+]] = vector.splat [[VAR_10_]] : vector<4xf32>
// CHECK:               [[VAR_37_1_:%.+]] = arith.divf [[VAR_35_1_]], [[LOAD_RES_7_MEM_2_]] : vector<4xf32>
// CHECK:               [[VAR_38_:%.+]] = math.floor [[VAR_37_1_]] : vector<4xf32>
// CHECK:               [[VAR_39_:%.+]] = arith.subf [[VAR_37_1_]], [[VAR_38_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_40_:%.+]] = arith.cmpf ogt, [[VAR_39_]], [[VAR_cst_1_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_41_:%.+]] = arith.addf [[VAR_38_]], [[VAR_cst_3_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_42_:%.+]] = arith.select [[VAR_40_]], [[VAR_41_]], [[VAR_38_]] : vector<4xi1>, vector<4xf32>
// CHECK-DAG:           [[VAR_43_:%.+]] = arith.mulf [[VAR_38_]], [[VAR_cst_1_]] : vector<4xf32>
// CHECK:               [[VAR_44_:%.+]] = math.floor [[VAR_43_]] : vector<4xf32>
// CHECK:               [[VAR_45_:%.+]] = arith.mulf [[VAR_44_]], [[VAR_cst_2_]] : vector<4xf32>
// CHECK:               [[VAR_46_:%.+]] = arith.subf [[VAR_38_]], [[VAR_45_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_47_:%.+]] = arith.cmpf oeq, [[VAR_46_]], [[VAR_cst_3_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_48_:%.+]] = arith.addf [[VAR_38_]], [[VAR_cst_3_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_49_:%.+]] = arith.select [[VAR_47_]], [[VAR_48_]], [[VAR_38_]] : vector<4xi1>, vector<4xf32>
// CHECK-DAG:           [[VAR_50_:%.+]] = arith.cmpf oeq, [[VAR_39_]], [[VAR_cst_1_]] : vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_51_:%.+]] = arith.select [[VAR_50_]], [[VAR_49_]], [[VAR_42_]] : vector<4xi1>, vector<4xf32>
// CHECK-DAG:           [[VAR_52_:%.+]] = vector.splat [[VAR_28_]] : vector<4xf32>
// CHECK:               [[VAR_53_:%.+]] = arith.addf [[VAR_51_]], [[VAR_52_]] : vector<4xf32>
// CHECK:               [[VAR_54_:%.+]] = arith.maxnumf [[VAR_53_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK:               [[VAR_55_:%.+]] = arith.minnumf [[VAR_54_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:               [[VAR_56_:%.+]] = arith.fptoui [[VAR_55_]] : vector<4xf32> to vector<4xi8>
// CHECK:               [[VAR_57_:%.+]] = builtin.unrealized_conversion_cast [[VAR_56_]] : vector<4xi8> to vector<4xui8>
// CHECK:               vector.store [[VAR_57_]], [[RES_]]{{.}}[[VAR_32_1_]], [[LOAD_RES_5_MEM_2_]]{{.}} : memref<256x4xui8>, vector<4xui8>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_]]_11, [[RES_]]_12 : memref<256x4xui8>, memref<f32>, memref<ui8>
// CHECK:         }
}

