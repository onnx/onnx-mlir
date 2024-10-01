// RUN: onnx-mlir-opt -O3 -mcpu=z16 --shape-inference --convert-onnx-to-krnl=enable-parallel --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

// -----


func.func @test_dynamic_quantize_linear_simd_only(%arg0: tensor<256x16xf32>) -> (tensor<256x16xui8>, tensor<f32>, tensor<ui8>) {
  %y, %y_scale, %y_zero_point = "onnx.DynamicQuantizeLinear"(%arg0) : (tensor<256x16xf32>) -> (tensor<256x16xui8>, tensor<f32>, tensor<ui8>)
  return %y, %y_scale, %y_zero_point:  tensor<256x16xui8>, tensor<f32>, tensor<ui8>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 * 512)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (4096, d0 * 512 + 512)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 * 32)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (4065, d0 * 512 + 481)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 * -512 + 4096, 512)>
// CHECK-LABEL:  func.func @test_dynamic_quantize_linear_simd_only
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<256x16xf32>) -> (memref<256x16xui8>, memref<f32>, memref<ui8>) {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<2.550000e+02> : vector<16xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<0.000000e+00> : vector<16xf32>
// CHECK-DAG:       [[VAR_cst_1_:%.+]] = arith.constant dense<5.000000e-01> : vector<16xf32>
// CHECK-DAG:       [[VAR_cst_2_:%.+]] = arith.constant dense<2.000000e+00> : vector<16xf32>
// CHECK-DAG:       [[VAR_cst_3_:%.+]] = arith.constant dense<1.000000e+00> : vector<16xf32>
// CHECK-DAG:       [[VAR_cst_4_:%.+]] = arith.constant dense<0xFF800000> : vector<1xf32>
// CHECK-DAG:       [[VAR_cst_5_:%.+]] = arith.constant dense<0x7F800000> : vector<1xf32>
// CHECK-DAG:       [[VAR_cst_6_:%.+]] = arith.constant dense<0xFF800000> : vector<32xf32>
// CHECK-DAG:       [[VAR_cst_7_:%.+]] = arith.constant dense<0x7F800000> : vector<32xf32>
// CHECK-DAG:       [[CST_5_dot_000000_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:       [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_4096_:%.+]] = arith.constant 4096 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_2_dot_550000_:%.+]] = arith.constant 2.550000e+02 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<256x16xui8>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() : memref<f32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() : memref<ui8>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_4096_]], [[RES_3_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_3_]]) : (memref<256x16xf32>, memref<1xindex>) -> memref<4096xf32>
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<256xf32>
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() : memref<8xf32>
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<256xf32>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc() : memref<8xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.parallel([[LOOP_0_]]) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 8){
// CHECK:             [[VAR_34_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[VAR_35_:%.+]] = affine.apply [[MAP_0_]]([[VAR_34_]])
// CHECK-DAG:         [[VAR_36_:%.+]] = affine.min [[MAP_1_]]([[VAR_34_]])
// CHECK-DAG:         [[VAR_37_:%.+]] = affine.apply [[MAP_2_]]([[VAR_34_]])
// CHECK:             vector.store [[VAR_cst_7_]], [[RES_4_]]{{.}}[[VAR_37_]]{{.}} : memref<256xf32>, vector<32xf32>
// CHECK:             vector.store [[VAR_cst_6_]], [[RES_6_]]{{.}}[[VAR_37_]]{{.}} : memref<256xf32>, vector<32xf32>
// CHECK:             [[VAR_38_:%.+]] = affine.min [[MAP_3_]]([[VAR_34_]])
// CHECK:             scf.for [[I_1_:%.+]] = [[VAR_35_]] to [[VAR_38_]] step [[CST_32_]] {
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[I_1_]]{{.}} : memref<4096xf32>, vector<32xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_1_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[I_1_]]{{.}} : memref<4096xf32>, vector<32xf32>
// CHECK-DAG:           [[LOAD_RES_4_MEM_:%.+]] = vector.load [[RES_4_]]{{.}}[[VAR_37_]]{{.}} : memref<256xf32>, vector<32xf32>
// CHECK-DAG:           [[LOAD_RES_6_MEM_:%.+]] = vector.load [[RES_6_]]{{.}}[[VAR_37_]]{{.}} : memref<256xf32>, vector<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_51_:%.+]] = arith.minnumf [[LOAD_RES_4_MEM_]], [[LOAD_VAR_reshape_MEM_]] : vector<32xf32>
// CHECK-DAG:           [[VAR_52_:%.+]] = arith.maxnumf [[LOAD_RES_6_MEM_]], [[LOAD_VAR_reshape_MEM_1_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_51_]], [[RES_4_]]{{.}}[[VAR_37_]]{{.}} : memref<256xf32>, vector<32xf32>
// CHECK:               vector.store [[VAR_52_]], [[RES_6_]]{{.}}[[VAR_37_]]{{.}} : memref<256xf32>, vector<32xf32>
// CHECK:             }
// CHECK:             [[VAR_39_:%.+]] = affine.min [[MAP_4_]]([[VAR_34_]])
// CHECK:             [[VAR_40_:%.+]] = arith.remsi [[VAR_39_]], [[CST_32_]] : index
// CHECK:             [[VAR_41_:%.+]] = arith.subi [[VAR_39_]], [[VAR_40_]] : index
// CHECK:             [[VAR_42_:%.+]] = arith.addi [[VAR_35_]], [[VAR_41_]] : index
// CHECK:             scf.for [[I_2_:%.+]] = [[VAR_42_]] to [[VAR_36_]] step [[CST_1_]] {
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_2_:%.+]] = memref.load [[VAR_reshape_]]{{.}}[[I_2_]]{{.}} : memref<4096xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_3_:%.+]] = memref.load [[VAR_reshape_]]{{.}}[[I_2_]]{{.}} : memref<4096xf32>
// CHECK-DAG:           [[LOAD_RES_4_MEM_1_:%.+]] = memref.load [[RES_4_]]{{.}}[[VAR_37_]]{{.}} : memref<256xf32>
// CHECK-DAG:           [[LOAD_RES_6_MEM_1_:%.+]] = memref.load [[RES_6_]]{{.}}[[VAR_37_]]{{.}} : memref<256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_51_1_:%.+]] = arith.minnumf [[LOAD_RES_4_MEM_1_]], [[LOAD_VAR_reshape_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_52_1_:%.+]] = arith.maxnumf [[LOAD_RES_6_MEM_1_]], [[LOAD_VAR_reshape_MEM_3_]] : f32
// CHECK:               memref.store [[VAR_51_1_]], [[RES_4_]]{{.}}[[VAR_37_]]{{.}} : memref<256xf32>
// CHECK:               memref.store [[VAR_52_1_]], [[RES_6_]]{{.}}[[VAR_37_]]{{.}} : memref<256xf32>
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_4_MEM_2_:%.+]] = vector.load [[RES_4_]]{{.}}[[VAR_37_]]{{.}} : memref<256xf32>, vector<32xf32>
// CHECK-DAG:         [[LOAD_RES_6_MEM_2_:%.+]] = vector.load [[RES_6_]]{{.}}[[VAR_37_]]{{.}} : memref<256xf32>, vector<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_45_:%.+]] = vector.reduction <minnumf>, [[LOAD_RES_4_MEM_2_]] : vector<32xf32> into f32
// CHECK-DAG:         [[VAR_46_:%.+]] = vector.reduction <maxnumf>, [[LOAD_RES_6_MEM_2_]] : vector<32xf32> into f32
// CHECK:             memref.store [[VAR_45_]], [[RES_5_]]{{.}}[[VAR_34_]]{{.}} : memref<8xf32>
// CHECK:             memref.store [[VAR_46_]], [[RES_7_]]{{.}}[[VAR_34_]]{{.}} : memref<8xf32>
// CHECK:           }
// CHECK-DAG:       [[RES_8_:%.+]] = memref.alloc() : memref<f32>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           vector.store [[VAR_cst_5_]], [[RES_4_]]{{.}}[[CST_0_]]{{.}} : memref<256xf32>, vector<1xf32>
// CHECK:           vector.store [[VAR_cst_4_]], [[RES_6_]]{{.}}[[CST_0_]]{{.}} : memref<256xf32>, vector<1xf32>
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_3_:%.+]] = 0 to 8){
// CHECK:             [[VAR_34_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[VAR_35_1_:%.+]] = krnl.load [[RES_5_]]{{.}}[[VAR_34_1_]]{{.}} : memref<8xf32>
// CHECK-DAG:         [[VAR_36_1_:%.+]] = krnl.load [[RES_7_]]{{.}}[[VAR_34_1_]]{{.}} : memref<8xf32>
// CHECK-DAG:         [[LOAD_RES_4_MEM_3_:%.+]] = krnl.load [[RES_4_]]{{.}}[[CST_0_]]{{.}} : memref<256xf32>
// CHECK-DAG:         [[LOAD_RES_6_MEM_3_:%.+]] = krnl.load [[RES_6_]]{{.}}[[CST_0_]]{{.}} : memref<256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_39_1_:%.+]] = arith.minnumf [[LOAD_RES_4_MEM_3_]], [[VAR_35_1_]] : f32
// CHECK-DAG:         [[VAR_40_1_:%.+]] = arith.maxnumf [[LOAD_RES_6_MEM_3_]], [[VAR_36_1_]] : f32
// CHECK:             krnl.store [[VAR_39_1_]], [[RES_4_]]{{.}}[[CST_0_]]{{.}} : memref<256xf32>
// CHECK:             krnl.store [[VAR_40_1_]], [[RES_6_]]{{.}}[[CST_0_]]{{.}} : memref<256xf32>
// CHECK:           }
// CHECK-DAG:       [[LOAD_RES_4_MEM_4_:%.+]] = vector.load [[RES_4_]]{{.}}[[CST_0_]]{{.}} : memref<256xf32>, vector<1xf32>
// CHECK-DAG:       [[LOAD_RES_6_MEM_4_:%.+]] = vector.load [[RES_6_]]{{.}}[[CST_0_]]{{.}} : memref<256xf32>, vector<1xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = vector.extract [[LOAD_RES_4_MEM_4_]][0] : f32 from vector<1xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = vector.extract [[LOAD_RES_6_MEM_4_]][0] : f32 from vector<1xf32>
// CHECK:           krnl.store [[VAR_4_]], [[RES_8_]][] : memref<f32>
// CHECK:           krnl.store [[VAR_5_]], [[RES_9_]][] : memref<f32>
// CHECK-DAG:       [[LOAD_RES_8_MEM_:%.+]] = krnl.load [[RES_8_]][] : memref<f32>
// CHECK-DAG:       [[LOAD_RES_9_MEM_:%.+]] = krnl.load [[RES_9_]][] : memref<f32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.maxnumf [[LOAD_RES_9_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:       [[VAR_9_:%.+]] = arith.minnumf [[LOAD_RES_8_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK:           [[VAR_10_:%.+]] = arith.subf [[VAR_8_]], [[VAR_9_]] : f32
// CHECK:           [[VAR_11_:%.+]] = arith.divf [[VAR_10_]], [[CST_2_dot_550000_]] : f32
// CHECK:           [[VAR_12_:%.+]] = arith.divf [[VAR_9_]], [[VAR_11_]] : f32
// CHECK:           [[VAR_13_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_12_]] : f32
// CHECK:           [[VAR_14_:%.+]] = arith.maxnumf [[VAR_13_]], [[CST_0_dot_000000_]] : f32
// CHECK:           [[VAR_15_:%.+]] = arith.minnumf [[VAR_14_]], [[CST_2_dot_550000_]] : f32
// CHECK:           [[VAR_16_:%.+]] = math.floor [[VAR_15_]] : f32
// CHECK:           [[VAR_17_:%.+]] = arith.subf [[VAR_15_]], [[VAR_16_]] : f32
// CHECK-DAG:       [[VAR_18_:%.+]] = arith.cmpf ogt, [[VAR_17_]], [[CST_5_dot_000000_]] : f32
// CHECK-DAG:       [[VAR_19_:%.+]] = arith.addf [[VAR_16_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_20_:%.+]] = arith.select [[VAR_18_]], [[VAR_19_]], [[VAR_16_]] : f32
// CHECK-DAG:       [[VAR_21_:%.+]] = arith.mulf [[VAR_16_]], [[CST_5_dot_000000_]] : f32
// CHECK:           [[VAR_22_:%.+]] = math.floor [[VAR_21_]] : f32
// CHECK:           [[VAR_23_:%.+]] = arith.mulf [[VAR_22_]], [[CST_2_dot_000000_]] : f32
// CHECK:           [[VAR_24_:%.+]] = arith.subf [[VAR_16_]], [[VAR_23_]] : f32
// CHECK-DAG:       [[VAR_25_:%.+]] = arith.cmpf oeq, [[VAR_24_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:       [[VAR_26_:%.+]] = arith.addf [[VAR_16_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_27_:%.+]] = arith.select [[VAR_25_]], [[VAR_26_]], [[VAR_16_]] : f32
// CHECK-DAG:       [[VAR_28_:%.+]] = arith.cmpf oeq, [[VAR_17_]], [[CST_5_dot_000000_]] : f32
// CHECK:           [[VAR_29_:%.+]] = arith.select [[VAR_28_]], [[VAR_27_]], [[VAR_20_]] : f32
// CHECK:           [[VAR_30_:%.+]] = arith.fptoui [[VAR_29_]] : f32 to i32
// CHECK:           [[VAR_31_:%.+]] = arith.trunci [[VAR_30_]] : i32 to i8
// CHECK:           [[VAR_32_:%.+]] = builtin.unrealized_conversion_cast [[VAR_31_]] : i8 to ui8
// CHECK:           krnl.store [[VAR_11_]], [[RES_1_]][] : memref<f32>
// CHECK:           krnl.store [[VAR_32_]], [[RES_2_]][] : memref<ui8>
// CHECK:           [[RES_10_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_4096_]], [[RES_10_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_23_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_10_]]) : (memref<256x16xf32>, memref<1xindex>) -> memref<4096xf32>
// CHECK-DAG:       [[RES_11_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_4096_]], [[RES_11_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_25_:%.+]] = memref.reshape [[RES_]]([[RES_]]_24) : (memref<256x16xui8>, memref<1xindex>) -> memref<4096xui8>
// CHECK-DAG:       [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_2_]] 16 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.parallel([[BLOCK_TILE__0_]]) : !krnl.loop
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_2_]] -> [[I_4_:%.+]] = 0 to 4096){
// CHECK:             [[VAR_34_2_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[VAR_35_1_:%.+]] = vector.load [[VAR_reshape_23_]]{{.}}[[VAR_34_2_]]{{.}} : memref<4096xf32>, vector<16xf32>
// CHECK-DAG:         [[VAR_36_2_:%.+]] = vector.splat [[VAR_11_]] : vector<16xf32>
// CHECK:             [[VAR_37_1_:%.+]] = arith.divf [[VAR_35_1_]], [[VAR_36_2_]] : vector<16xf32>
// CHECK:             [[VAR_38_1_:%.+]] = math.floor [[VAR_37_1_]] : vector<16xf32>
// CHECK:             [[VAR_39_2_:%.+]] = arith.subf [[VAR_37_1_]], [[VAR_38_1_]] : vector<16xf32>
// CHECK-DAG:         [[VAR_40_2_:%.+]] = arith.cmpf ogt, [[VAR_39_2_]], [[VAR_cst_1_]] : vector<16xf32>
// CHECK-DAG:         [[VAR_41_1_:%.+]] = arith.addf [[VAR_38_1_]], [[VAR_cst_3_]] : vector<16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_42_1_:%.+]] = arith.select [[VAR_40_2_]], [[VAR_41_1_]], [[VAR_38_1_]] : vector<16xi1>, vector<16xf32>
// CHECK-DAG:         [[LOAD_RES_4_MEM_2_:%.+]] = arith.mulf [[VAR_38_1_]], [[VAR_cst_1_]] : vector<16xf32>
// CHECK:             [[LOAD_RES_6_MEM_2_:%.+]] = math.floor [[LOAD_RES_4_MEM_2_]] : vector<16xf32>
// CHECK:             [[VAR_45_1_:%.+]] = arith.mulf [[LOAD_RES_6_MEM_2_]], [[VAR_cst_2_]] : vector<16xf32>
// CHECK:             [[VAR_46_1_:%.+]] = arith.subf [[VAR_38_1_]], [[VAR_45_1_]] : vector<16xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_2_:%.+]] = arith.cmpf oeq, [[VAR_46_1_]], [[VAR_cst_3_]] : vector<16xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_3_:%.+]] = arith.addf [[VAR_38_1_]], [[VAR_cst_3_]] : vector<16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_RES_4_MEM_1_:%.+]] = arith.select [[LOAD_VAR_reshape_MEM_2_]], [[LOAD_VAR_reshape_MEM_3_]], [[VAR_38_1_]] : vector<16xi1>, vector<16xf32>
// CHECK-DAG:         [[LOAD_RES_6_MEM_1_:%.+]] = arith.cmpf oeq, [[VAR_39_2_]], [[VAR_cst_1_]] : vector<16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_51_2_:%.+]] = arith.select [[LOAD_RES_6_MEM_1_]], [[LOAD_RES_4_MEM_1_]], [[VAR_42_1_]] : vector<16xi1>, vector<16xf32>
// CHECK-DAG:         [[VAR_52_2_:%.+]] = vector.splat [[VAR_29_]] : vector<16xf32>
// CHECK:             [[VAR_53_:%.+]] = arith.addf [[VAR_51_2_]], [[VAR_52_2_]] : vector<16xf32>
// CHECK:             [[VAR_54_:%.+]] = arith.maxnumf [[VAR_53_]], [[VAR_cst_0_]] : vector<16xf32>
// CHECK:             [[VAR_55_:%.+]] = arith.minnumf [[VAR_54_]], [[VAR_cst_]] : vector<16xf32>
// CHECK:             [[VAR_56_:%.+]] = arith.fptoui [[VAR_55_]] : vector<16xf32> to vector<16xi32>
// CHECK:             [[VAR_57_:%.+]] = arith.trunci [[VAR_56_]] : vector<16xi32> to vector<16xi8>
// CHECK:             [[VAR_58_:%.+]] = builtin.unrealized_conversion_cast [[VAR_57_]] : vector<16xi8> to vector<16xui8>
// CHECK:             vector.store [[VAR_58_]], [[VAR_reshape_25_]]{{.}}[[VAR_34_2_]]{{.}} : memref<4096xui8>, vector<16xui8>
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_]]_13, [[RES_]]_14 : memref<256x16xui8>, memref<f32>, memref<ui8>
// CHECK:         }
}

// -----


func.func @test_dynamic_quantize_linear_simd_and_scalar(%arg0: tensor<255x17xf32>) -> (tensor<255x17xui8>, tensor<f32>, tensor<ui8>) {
  %y, %y_scale, %y_zero_point = "onnx.DynamicQuantizeLinear"(%arg0) : (tensor<255x17xf32>) -> (tensor<255x17xui8>, tensor<f32>, tensor<ui8>)
  return %y, %y_scale, %y_zero_point:  tensor<255x17xui8>, tensor<f32>, tensor<ui8>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 * 542)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (4335, d0 * 542 + 542)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 * 32)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (4304, d0 * 542 + 511)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 * -542 + 4335, 542)>
// CHECK-LABEL:  func.func @test_dynamic_quantize_linear_simd_and_scalar
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<255x17xf32>) -> (memref<255x17xui8>, memref<f32>, memref<ui8>) {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<2.550000e+02> : vector<16xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<0.000000e+00> : vector<16xf32>
// CHECK-DAG:       [[VAR_cst_1_:%.+]] = arith.constant dense<5.000000e-01> : vector<16xf32>
// CHECK-DAG:       [[VAR_cst_2_:%.+]] = arith.constant dense<2.000000e+00> : vector<16xf32>
// CHECK-DAG:       [[VAR_cst_3_:%.+]] = arith.constant dense<1.000000e+00> : vector<16xf32>
// CHECK-DAG:       [[VAR_cst_4_:%.+]] = arith.constant dense<0xFF800000> : vector<1xf32>
// CHECK-DAG:       [[VAR_cst_5_:%.+]] = arith.constant dense<0x7F800000> : vector<1xf32>
// CHECK-DAG:       [[VAR_cst_6_:%.+]] = arith.constant dense<0xFF800000> : vector<32xf32>
// CHECK-DAG:       [[VAR_cst_7_:%.+]] = arith.constant dense<0x7F800000> : vector<32xf32>
// CHECK-DAG:       [[CST_5_dot_000000_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:       [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_4335_:%.+]] = arith.constant 4335 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_2_dot_550000_:%.+]] = arith.constant 2.550000e+02 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<255x17xui8>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() : memref<f32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() : memref<ui8>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_4335_]], [[RES_3_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_3_]]) : (memref<255x17xf32>, memref<1xindex>) -> memref<4335xf32>
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<256xf32>
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() : memref<8xf32>
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<256xf32>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc() : memref<8xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.parallel([[LOOP_0_]]) : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 8){
// CHECK:             [[VAR_35_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[VAR_36_:%.+]] = affine.apply [[MAP_0_]]([[VAR_35_]])
// CHECK-DAG:         [[VAR_37_:%.+]] = affine.min [[MAP_1_]]([[VAR_35_]])
// CHECK-DAG:         [[VAR_38_:%.+]] = affine.apply [[MAP_2_]]([[VAR_35_]])
// CHECK:             vector.store [[VAR_cst_7_]], [[RES_4_]]{{.}}[[VAR_38_]]{{.}} : memref<256xf32>, vector<32xf32>
// CHECK:             vector.store [[VAR_cst_6_]], [[RES_6_]]{{.}}[[VAR_38_]]{{.}} : memref<256xf32>, vector<32xf32>
// CHECK:             [[VAR_39_:%.+]] = affine.min [[MAP_3_]]([[VAR_35_]])
// CHECK:             scf.for [[I_1_:%.+]] = [[VAR_36_]] to [[VAR_39_]] step [[CST_32_]] {
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[I_1_]]{{.}} : memref<4335xf32>, vector<32xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_1_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[I_1_]]{{.}} : memref<4335xf32>, vector<32xf32>
// CHECK-DAG:           [[LOAD_RES_4_MEM_:%.+]] = vector.load [[RES_4_]]{{.}}[[VAR_38_]]{{.}} : memref<256xf32>, vector<32xf32>
// CHECK-DAG:           [[LOAD_RES_6_MEM_:%.+]] = vector.load [[RES_6_]]{{.}}[[VAR_38_]]{{.}} : memref<256xf32>, vector<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_52_:%.+]] = arith.minnumf [[LOAD_RES_4_MEM_]], [[LOAD_VAR_reshape_MEM_]] : vector<32xf32>
// CHECK-DAG:           [[VAR_53_:%.+]] = arith.maxnumf [[LOAD_RES_6_MEM_]], [[LOAD_VAR_reshape_MEM_1_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_52_]], [[RES_4_]]{{.}}[[VAR_38_]]{{.}} : memref<256xf32>, vector<32xf32>
// CHECK:               vector.store [[VAR_53_]], [[RES_6_]]{{.}}[[VAR_38_]]{{.}} : memref<256xf32>, vector<32xf32>
// CHECK:             }
// CHECK:             [[VAR_40_:%.+]] = affine.min [[MAP_4_]]([[VAR_35_]])
// CHECK:             [[VAR_41_:%.+]] = arith.remsi [[VAR_40_]], [[CST_32_]] : index
// CHECK:             [[VAR_42_:%.+]] = arith.subi [[VAR_40_]], [[VAR_41_]] : index
// CHECK:             [[VAR_43_:%.+]] = arith.addi [[VAR_36_]], [[VAR_42_]] : index
// CHECK:             scf.for [[I_2_:%.+]] = [[VAR_43_]] to [[VAR_37_]] step [[CST_1_]] {
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_2_:%.+]] = memref.load [[VAR_reshape_]]{{.}}[[I_2_]]{{.}} : memref<4335xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_3_:%.+]] = memref.load [[VAR_reshape_]]{{.}}[[I_2_]]{{.}} : memref<4335xf32>
// CHECK-DAG:           [[LOAD_RES_4_MEM_1_:%.+]] = memref.load [[RES_4_]]{{.}}[[VAR_38_]]{{.}} : memref<256xf32>
// CHECK-DAG:           [[LOAD_RES_6_MEM_1_:%.+]] = memref.load [[RES_6_]]{{.}}[[VAR_38_]]{{.}} : memref<256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_52_1_:%.+]] = arith.minnumf [[LOAD_RES_4_MEM_1_]], [[LOAD_VAR_reshape_MEM_2_]] : f32
// CHECK-DAG:           [[VAR_53_1_:%.+]] = arith.maxnumf [[LOAD_RES_6_MEM_1_]], [[LOAD_VAR_reshape_MEM_3_]] : f32
// CHECK:               memref.store [[VAR_52_1_]], [[RES_4_]]{{.}}[[VAR_38_]]{{.}} : memref<256xf32>
// CHECK:               memref.store [[VAR_53_1_]], [[RES_6_]]{{.}}[[VAR_38_]]{{.}} : memref<256xf32>
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_4_MEM_2_:%.+]] = vector.load [[RES_4_]]{{.}}[[VAR_38_]]{{.}} : memref<256xf32>, vector<32xf32>
// CHECK-DAG:         [[LOAD_RES_6_MEM_2_:%.+]] = vector.load [[RES_6_]]{{.}}[[VAR_38_]]{{.}} : memref<256xf32>, vector<32xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_46_:%.+]] = vector.reduction <minnumf>, [[LOAD_RES_4_MEM_2_]] : vector<32xf32> into f32
// CHECK-DAG:         [[VAR_47_:%.+]] = vector.reduction <maxnumf>, [[LOAD_RES_6_MEM_2_]] : vector<32xf32> into f32
// CHECK:             memref.store [[VAR_46_]], [[RES_5_]]{{.}}[[VAR_35_]]{{.}} : memref<8xf32>
// CHECK:             memref.store [[VAR_47_]], [[RES_7_]]{{.}}[[VAR_35_]]{{.}} : memref<8xf32>
// CHECK:           }
// CHECK-DAG:       [[RES_8_:%.+]] = memref.alloc() : memref<f32>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           vector.store [[VAR_cst_5_]], [[RES_4_]]{{.}}[[CST_0_]]{{.}} : memref<256xf32>, vector<1xf32>
// CHECK:           vector.store [[VAR_cst_4_]], [[RES_6_]]{{.}}[[CST_0_]]{{.}} : memref<256xf32>, vector<1xf32>
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_3_:%.+]] = 0 to 8){
// CHECK:             [[VAR_35_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[VAR_36_1_:%.+]] = krnl.load [[RES_5_]]{{.}}[[VAR_35_1_]]{{.}} : memref<8xf32>
// CHECK-DAG:         [[VAR_37_1_:%.+]] = krnl.load [[RES_7_]]{{.}}[[VAR_35_1_]]{{.}} : memref<8xf32>
// CHECK-DAG:         [[LOAD_RES_4_MEM_3_:%.+]] = krnl.load [[RES_4_]]{{.}}[[CST_0_]]{{.}} : memref<256xf32>
// CHECK-DAG:         [[LOAD_RES_6_MEM_3_:%.+]] = krnl.load [[RES_6_]]{{.}}[[CST_0_]]{{.}} : memref<256xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_40_1_:%.+]] = arith.minnumf [[LOAD_RES_4_MEM_3_]], [[VAR_36_1_]] : f32
// CHECK-DAG:         [[VAR_41_1_:%.+]] = arith.maxnumf [[LOAD_RES_6_MEM_3_]], [[VAR_37_1_]] : f32
// CHECK:             krnl.store [[VAR_40_1_]], [[RES_4_]]{{.}}[[CST_0_]]{{.}} : memref<256xf32>
// CHECK:             krnl.store [[VAR_41_1_]], [[RES_6_]]{{.}}[[CST_0_]]{{.}} : memref<256xf32>
// CHECK:           }
// CHECK-DAG:       [[LOAD_RES_4_MEM_4_:%.+]] = vector.load [[RES_4_]]{{.}}[[CST_0_]]{{.}} : memref<256xf32>, vector<1xf32>
// CHECK-DAG:       [[LOAD_RES_6_MEM_4_:%.+]] = vector.load [[RES_6_]]{{.}}[[CST_0_]]{{.}} : memref<256xf32>, vector<1xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = vector.extract [[LOAD_RES_4_MEM_4_]][0] : f32 from vector<1xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = vector.extract [[LOAD_RES_6_MEM_4_]][0] : f32 from vector<1xf32>
// CHECK:           krnl.store [[VAR_4_]], [[RES_8_]][] : memref<f32>
// CHECK:           krnl.store [[VAR_5_]], [[RES_9_]][] : memref<f32>
// CHECK-DAG:       [[LOAD_RES_8_MEM_:%.+]] = krnl.load [[RES_8_]][] : memref<f32>
// CHECK-DAG:       [[LOAD_RES_9_MEM_:%.+]] = krnl.load [[RES_9_]][] : memref<f32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.maxnumf [[LOAD_RES_9_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:       [[VAR_9_:%.+]] = arith.minnumf [[LOAD_RES_8_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK:           [[VAR_10_:%.+]] = arith.subf [[VAR_8_]], [[VAR_9_]] : f32
// CHECK:           [[VAR_11_:%.+]] = arith.divf [[VAR_10_]], [[CST_2_dot_550000_]] : f32
// CHECK:           [[VAR_12_:%.+]] = arith.divf [[VAR_9_]], [[VAR_11_]] : f32
// CHECK:           [[VAR_13_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_12_]] : f32
// CHECK:           [[VAR_14_:%.+]] = arith.maxnumf [[VAR_13_]], [[CST_0_dot_000000_]] : f32
// CHECK:           [[VAR_15_:%.+]] = arith.minnumf [[VAR_14_]], [[CST_2_dot_550000_]] : f32
// CHECK:           [[VAR_16_:%.+]] = math.floor [[VAR_15_]] : f32
// CHECK:           [[VAR_17_:%.+]] = arith.subf [[VAR_15_]], [[VAR_16_]] : f32
// CHECK-DAG:       [[VAR_18_:%.+]] = arith.cmpf ogt, [[VAR_17_]], [[CST_5_dot_000000_]] : f32
// CHECK-DAG:       [[VAR_19_:%.+]] = arith.addf [[VAR_16_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_20_:%.+]] = arith.select [[VAR_18_]], [[VAR_19_]], [[VAR_16_]] : f32
// CHECK-DAG:       [[VAR_21_:%.+]] = arith.mulf [[VAR_16_]], [[CST_5_dot_000000_]] : f32
// CHECK:           [[VAR_22_:%.+]] = math.floor [[VAR_21_]] : f32
// CHECK:           [[VAR_23_:%.+]] = arith.mulf [[VAR_22_]], [[CST_2_dot_000000_]] : f32
// CHECK:           [[VAR_24_:%.+]] = arith.subf [[VAR_16_]], [[VAR_23_]] : f32
// CHECK-DAG:       [[VAR_25_:%.+]] = arith.cmpf oeq, [[VAR_24_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:       [[VAR_26_:%.+]] = arith.addf [[VAR_16_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_27_:%.+]] = arith.select [[VAR_25_]], [[VAR_26_]], [[VAR_16_]] : f32
// CHECK-DAG:       [[VAR_28_:%.+]] = arith.cmpf oeq, [[VAR_17_]], [[CST_5_dot_000000_]] : f32
// CHECK:           [[VAR_29_:%.+]] = arith.select [[VAR_28_]], [[VAR_27_]], [[VAR_20_]] : f32
// CHECK:           [[VAR_30_:%.+]] = arith.fptoui [[VAR_29_]] : f32 to i32
// CHECK:           [[VAR_31_:%.+]] = arith.trunci [[VAR_30_]] : i32 to i8
// CHECK:           [[VAR_32_:%.+]] = builtin.unrealized_conversion_cast [[VAR_31_]] : i8 to ui8
// CHECK:           krnl.store [[VAR_11_]], [[RES_1_]][] : memref<f32>
// CHECK:           krnl.store [[VAR_32_]], [[RES_2_]][] : memref<ui8>
// CHECK:           [[RES_10_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_4335_]], [[RES_10_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_23_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_10_]]) : (memref<255x17xf32>, memref<1xindex>) -> memref<4335xf32>
// CHECK-DAG:       [[RES_11_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_4335_]], [[RES_11_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_25_:%.+]] = memref.reshape [[RES_]]([[RES_]]_24) : (memref<255x17xui8>, memref<1xindex>) -> memref<4335xui8>
// CHECK-DAG:       [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_2_]] 16 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.parallel([[BLOCK_TILE__0_]]) : !krnl.loop
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_2_]] -> [[I_4_:%.+]] = 0 to 4320){
// CHECK:             [[VAR_35_2_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[VAR_36_1_:%.+]] = vector.load [[VAR_reshape_23_]]{{.}}[[VAR_35_2_]]{{.}} : memref<4335xf32>, vector<16xf32>
// CHECK-DAG:         [[VAR_37_2_:%.+]] = vector.splat [[VAR_11_]] : vector<16xf32>
// CHECK:             [[VAR_38_1_:%.+]] = arith.divf [[VAR_36_1_]], [[VAR_37_2_]] : vector<16xf32>
// CHECK:             [[VAR_39_1_:%.+]] = math.floor [[VAR_38_1_]] : vector<16xf32>
// CHECK:             [[VAR_40_2_:%.+]] = arith.subf [[VAR_38_1_]], [[VAR_39_1_]] : vector<16xf32>
// CHECK-DAG:         [[VAR_41_2_:%.+]] = arith.cmpf ogt, [[VAR_40_2_]], [[VAR_cst_1_]] : vector<16xf32>
// CHECK-DAG:         [[VAR_42_1_:%.+]] = arith.addf [[VAR_39_1_]], [[VAR_cst_3_]] : vector<16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_43_1_:%.+]] = arith.select [[VAR_41_2_]], [[VAR_42_1_]], [[VAR_39_1_]] : vector<16xi1>, vector<16xf32>
// CHECK-DAG:         [[LOAD_RES_4_MEM_2_:%.+]] = arith.mulf [[VAR_39_1_]], [[VAR_cst_1_]] : vector<16xf32>
// CHECK:             [[LOAD_RES_6_MEM_2_:%.+]] = math.floor [[LOAD_RES_4_MEM_2_]] : vector<16xf32>
// CHECK:             [[VAR_46_1_:%.+]] = arith.mulf [[LOAD_RES_6_MEM_2_]], [[VAR_cst_2_]] : vector<16xf32>
// CHECK:             [[VAR_47_1_:%.+]] = arith.subf [[VAR_39_1_]], [[VAR_46_1_]] : vector<16xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_2_:%.+]] = arith.cmpf oeq, [[VAR_47_1_]], [[VAR_cst_3_]] : vector<16xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_3_:%.+]] = arith.addf [[VAR_39_1_]], [[VAR_cst_3_]] : vector<16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_RES_4_MEM_1_:%.+]] = arith.select [[LOAD_VAR_reshape_MEM_2_]], [[LOAD_VAR_reshape_MEM_3_]], [[VAR_39_1_]] : vector<16xi1>, vector<16xf32>
// CHECK-DAG:         [[LOAD_RES_6_MEM_1_:%.+]] = arith.cmpf oeq, [[VAR_40_2_]], [[VAR_cst_1_]] : vector<16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_52_2_:%.+]] = arith.select [[LOAD_RES_6_MEM_1_]], [[LOAD_RES_4_MEM_1_]], [[VAR_43_1_]] : vector<16xi1>, vector<16xf32>
// CHECK-DAG:         [[VAR_53_2_:%.+]] = vector.splat [[VAR_29_]] : vector<16xf32>
// CHECK:             [[VAR_54_:%.+]] = arith.addf [[VAR_52_2_]], [[VAR_53_2_]] : vector<16xf32>
// CHECK:             [[VAR_55_:%.+]] = arith.maxnumf [[VAR_54_]], [[VAR_cst_0_]] : vector<16xf32>
// CHECK:             [[VAR_56_:%.+]] = arith.minnumf [[VAR_55_]], [[VAR_cst_]] : vector<16xf32>
// CHECK:             [[VAR_57_:%.+]] = arith.fptoui [[VAR_56_]] : vector<16xf32> to vector<16xi32>
// CHECK:             [[VAR_58_:%.+]] = arith.trunci [[VAR_57_]] : vector<16xi32> to vector<16xi8>
// CHECK:             [[VAR_59_:%.+]] = builtin.unrealized_conversion_cast [[VAR_58_]] : vector<16xi8> to vector<16xui8>
// CHECK:             vector.store [[VAR_59_]], [[VAR_reshape_25_]]{{.}}[[VAR_35_2_]]{{.}} : memref<4335xui8>, vector<16xui8>
// CHECK:           }
// CHECK:           [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[I_5_:%.+]] = 4320 to 4335){
// CHECK:             [[VAR_35_3_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK:             [[VAR_36_1_1_:%.+]] = krnl.load [[VAR_reshape_23_]]{{.}}[[VAR_35_3_]]{{.}} : memref<4335xf32>
// CHECK:             [[VAR_37_3_:%.+]] = arith.divf [[VAR_36_1_1_]], [[VAR_11_]] : f32
// CHECK:             [[VAR_38_2_:%.+]] = math.floor [[VAR_37_3_]] : f32
// CHECK:             [[VAR_39_2_:%.+]] = arith.subf [[VAR_37_3_]], [[VAR_38_2_]] : f32
// CHECK-DAG:         [[VAR_40_3_:%.+]] = arith.cmpf ogt, [[VAR_39_2_]], [[CST_5_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_41_3_:%.+]] = arith.addf [[VAR_38_2_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_42_2_:%.+]] = arith.select [[VAR_40_3_]], [[VAR_41_3_]], [[VAR_38_2_]] : f32
// CHECK-DAG:         [[VAR_43_2_:%.+]] = arith.mulf [[VAR_38_2_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[LOAD_RES_4_MEM_2_1_:%.+]] = math.floor [[VAR_43_2_]] : f32
// CHECK:             [[LOAD_RES_6_MEM_2_1_:%.+]] = arith.mulf [[LOAD_RES_4_MEM_2_1_]], [[CST_2_dot_000000_]] : f32
// CHECK:             [[VAR_46_2_:%.+]] = arith.subf [[VAR_38_2_]], [[LOAD_RES_6_MEM_2_1_]] : f32
// CHECK-DAG:         [[VAR_47_2_:%.+]] = arith.cmpf oeq, [[VAR_46_2_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_2_1_:%.+]] = arith.addf [[VAR_38_2_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_3_1_:%.+]] = arith.select [[VAR_47_2_]], [[LOAD_VAR_reshape_MEM_2_1_]], [[VAR_38_2_]] : f32
// CHECK-DAG:         [[LOAD_RES_4_MEM_1_1_:%.+]] = arith.cmpf oeq, [[VAR_39_2_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[LOAD_RES_6_MEM_1_1_:%.+]] = arith.select [[LOAD_RES_4_MEM_1_1_]], [[LOAD_VAR_reshape_MEM_3_1_]], [[VAR_42_2_]] : f32
// CHECK:             [[VAR_52_3_:%.+]] = arith.addf [[LOAD_RES_6_MEM_1_1_]], [[VAR_29_]] : f32
// CHECK:             [[VAR_53_3_:%.+]] = arith.maxnumf [[VAR_52_3_]], [[CST_0_dot_000000_]] : f32
// CHECK:             [[VAR_54_1_:%.+]] = arith.minnumf [[VAR_53_3_]], [[CST_2_dot_550000_]] : f32
// CHECK:             [[VAR_55_1_:%.+]] = arith.fptoui [[VAR_54_1_]] : f32 to i32
// CHECK:             [[VAR_56_1_:%.+]] = arith.trunci [[VAR_55_1_]] : i32 to i8
// CHECK:             [[VAR_57_1_:%.+]] = builtin.unrealized_conversion_cast [[VAR_56_1_]] : i8 to ui8
// CHECK:             krnl.store [[VAR_57_1_]], [[VAR_reshape_25_]]{{.}}[[VAR_35_3_]]{{.}} : memref<4335xui8>
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_]]_13, [[RES_]]_14 : memref<255x17xui8>, memref<f32>, memref<ui8>
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
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<8xf32>
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() : memref<f32>
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<8xf32>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           vector.store [[VAR_cst_5_]], [[RES_4_]]{{.}}[[CST_0_]]{{.}} : memref<8xf32>, vector<8xf32>
// CHECK:           vector.store [[VAR_cst_4_]], [[RES_6_]]{{.}}[[CST_0_]]{{.}} : memref<8xf32>, vector<8xf32>
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 8){
// CHECK:             [[VAR_33_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_33_]]{{.}} : memref<8xf32>, vector<8xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_1_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_33_]]{{.}} : memref<8xf32>, vector<8xf32>
// CHECK-DAG:         [[LOAD_RES_4_MEM_:%.+]] = vector.load [[RES_4_]]{{.}}[[CST_0_]]{{.}} : memref<8xf32>, vector<8xf32>
// CHECK-DAG:         [[LOAD_RES_6_MEM_:%.+]] = vector.load [[RES_6_]]{{.}}[[CST_0_]]{{.}} : memref<8xf32>, vector<8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_38_:%.+]] = arith.minnumf [[LOAD_RES_4_MEM_]], [[LOAD_VAR_reshape_MEM_]] : vector<8xf32>
// CHECK-DAG:         [[VAR_39_:%.+]] = arith.maxnumf [[LOAD_RES_6_MEM_]], [[LOAD_VAR_reshape_MEM_1_]] : vector<8xf32>
// CHECK:             vector.store [[VAR_38_]], [[RES_4_]]{{.}}[[CST_0_]]{{.}} : memref<8xf32>, vector<8xf32>
// CHECK:             vector.store [[VAR_39_]], [[RES_6_]]{{.}}[[CST_0_]]{{.}} : memref<8xf32>, vector<8xf32>
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
// CHECK:           [[VAR_29_:%.+]] = arith.fptoui [[VAR_28_]] : f32 to i32
// CHECK:           [[VAR_30_:%.+]] = arith.trunci [[VAR_29_]] : i32 to i8
// CHECK:           [[VAR_31_:%.+]] = builtin.unrealized_conversion_cast [[VAR_30_]] : i8 to ui8
// CHECK:           krnl.store [[VAR_10_]], [[RES_1_]][] : memref<f32>
// CHECK:           krnl.store [[VAR_31_]], [[RES_2_]][] : memref<ui8>
// CHECK:           [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_8_]], [[RES_8_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_19_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_8_]]) : (memref<1x8xf32>, memref<1xindex>) -> memref<8xf32>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_8_]], [[RES_9_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_21_:%.+]] = memref.reshape [[RES_]]([[RES_]]_20) : (memref<1x8xui8>, memref<1xindex>) -> memref<8xui8>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_1_]] 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.parallel([[BLOCK_TILE__1_]]) : !krnl.loop
// CHECK:           krnl.iterate([[BLOCK_TILE__1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 8){
// CHECK:             [[VAR_33_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_2_:%.+]] = vector.load [[VAR_reshape_19_]]{{.}}[[VAR_33_1_]]{{.}} : memref<8xf32>, vector<8xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_1_:%.+]] = vector.splat [[VAR_10_]] : vector<8xf32>
// CHECK:             [[LOAD_RES_4_MEM_2_:%.+]] = arith.divf [[LOAD_VAR_reshape_MEM_2_]], [[LOAD_VAR_reshape_MEM_1_]] : vector<8xf32>
// CHECK:             [[LOAD_RES_6_MEM_2_:%.+]] = math.floor [[LOAD_RES_4_MEM_2_]] : vector<8xf32>
// CHECK:             [[VAR_38_1_:%.+]] = arith.subf [[LOAD_RES_4_MEM_2_]], [[LOAD_RES_6_MEM_2_]] : vector<8xf32>
// CHECK-DAG:         [[VAR_39_1_:%.+]] = arith.cmpf ogt, [[VAR_38_1_]], [[VAR_cst_1_]] : vector<8xf32>
// CHECK-DAG:         [[VAR_40_:%.+]] = arith.addf [[LOAD_RES_6_MEM_2_]], [[VAR_cst_3_]] : vector<8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_41_:%.+]] = arith.select [[VAR_39_1_]], [[VAR_40_]], [[LOAD_RES_6_MEM_2_]] : vector<8xi1>, vector<8xf32>
// CHECK-DAG:         [[VAR_42_:%.+]] = arith.mulf [[LOAD_RES_6_MEM_2_]], [[VAR_cst_1_]] : vector<8xf32>
// CHECK:             [[VAR_43_:%.+]] = math.floor [[VAR_42_]] : vector<8xf32>
// CHECK:             [[VAR_44_:%.+]] = arith.mulf [[VAR_43_]], [[VAR_cst_2_]] : vector<8xf32>
// CHECK:             [[VAR_45_:%.+]] = arith.subf [[LOAD_RES_6_MEM_2_]], [[VAR_44_]] : vector<8xf32>
// CHECK-DAG:         [[VAR_46_:%.+]] = arith.cmpf oeq, [[VAR_45_]], [[VAR_cst_3_]] : vector<8xf32>
// CHECK-DAG:         [[VAR_47_:%.+]] = arith.addf [[LOAD_RES_6_MEM_2_]], [[VAR_cst_3_]] : vector<8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_48_:%.+]] = arith.select [[VAR_46_]], [[VAR_47_]], [[LOAD_RES_6_MEM_2_]] : vector<8xi1>, vector<8xf32>
// CHECK-DAG:         [[VAR_49_:%.+]] = arith.cmpf oeq, [[VAR_38_1_]], [[VAR_cst_1_]] : vector<8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_50_:%.+]] = arith.select [[VAR_49_]], [[VAR_48_]], [[VAR_41_]] : vector<8xi1>, vector<8xf32>
// CHECK-DAG:         [[VAR_51_:%.+]] = vector.splat [[VAR_28_]] : vector<8xf32>
// CHECK:             [[VAR_52_:%.+]] = arith.addf [[VAR_50_]], [[VAR_51_]] : vector<8xf32>
// CHECK:             [[VAR_53_:%.+]] = arith.maxnumf [[VAR_52_]], [[VAR_cst_0_]] : vector<8xf32>
// CHECK:             [[VAR_54_:%.+]] = arith.minnumf [[VAR_53_]], [[VAR_cst_]] : vector<8xf32>
// CHECK:             [[VAR_55_:%.+]] = arith.fptoui [[VAR_54_]] : vector<8xf32> to vector<8xi32>
// CHECK:             [[VAR_56_:%.+]] = arith.trunci [[VAR_55_]] : vector<8xi32> to vector<8xi8>
// CHECK:             [[VAR_57_:%.+]] = builtin.unrealized_conversion_cast [[VAR_56_]] : vector<8xi8> to vector<8xui8>
// CHECK:             vector.store [[VAR_57_]], [[VAR_reshape_21_]]{{.}}[[VAR_33_1_]]{{.}} : memref<8xui8>, vector<8xui8>
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_]]_11, [[RES_]]_12 : memref<1x8xui8>, memref<f32>, memref<ui8>
// CHECK:         }
}

