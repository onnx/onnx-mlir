// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// -----

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

func.func @test_dynamic_quantize_linear(%arg0: tensor<?x2xf32>) -> (tensor<?x2xui8>, tensor<f32>, tensor<ui8>) {
  %y, %y_scale, %y_zero_point = "onnx.DynamicQuantizeLinear"(%arg0) : (tensor<?x2xf32>) -> (tensor<?x2xui8>, tensor<f32>, tensor<ui8>)
  return %y, %y_scale, %y_zero_point:  tensor<?x2xui8>, tensor<f32>, tensor<ui8>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 2)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-LABEL:  func.func @test_dynamic_quantize_linear
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x2xf32>) -> (memref<?x2xui8>, memref<f32>, memref<ui8>) {
// CHECK-DAG:       [[CST_5_dot_000000_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:       [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0x7F800000 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_2_dot_550000_:%.+]] = arith.constant 2.550000e+02 : f32
// CHECK-DAG:       [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_2_]] : memref<?x2xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x2xui8>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() : memref<f32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() : memref<ui8>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.memset [[RES_3_]], [[CST_0_1_]] : memref<f32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_9_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_2_]] : memref<?x2xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[VAR_dim_9_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[VAR_32_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_32_]]#0, [[VAR_32_]]#1] : memref<?x2xf32>
// CHECK-DAG:         [[LOAD_RES_3_MEM_:%.+]] = krnl.load [[RES_3_]][] : memref<f32>
// CHECK:             [[VAR_35_:%.+]] = arith.minnumf [[LOAD_RES_3_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_35_]], [[RES_3_]][] : memref<f32>
// CHECK:           }
// CHECK:           [[RES_4_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.memset [[RES_4_]], [[CST_0_]] : memref<f32>
// CHECK-DAG:       [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_11_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_2_]] : memref<?x2xf32>
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to [[VAR_dim_11_]], [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 2){
// CHECK:             [[VAR_32_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_32_1_]]#0, [[VAR_32_1_]]#1] : memref<?x2xf32>
// CHECK-DAG:         [[LOAD_RES_3_MEM_1_:%.+]] = krnl.load [[RES_4_]][] : memref<f32>
// CHECK:             [[VAR_35_1_:%.+]] = arith.maxnumf [[LOAD_RES_3_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:             krnl.store [[VAR_35_1_]], [[RES_4_]][] : memref<f32>
// CHECK:           }
// CHECK-DAG:       [[LOAD_RES_3_MEM_2_:%.+]] = krnl.load [[RES_3_]][] : memref<f32>
// CHECK-DAG:       [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]][] : memref<f32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.maxnumf [[LOAD_RES_4_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.minnumf [[LOAD_RES_3_MEM_2_]], [[CST_0_dot_000000_]] : f32
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
// CHECK:           [[VAR_26_:%.+]] = arith.fptoui [[VAR_25_]] : f32 to i32
// CHECK:           [[VAR_27_:%.+]] = arith.trunci [[VAR_26_]] : i32 to i8
// CHECK:           [[VAR_28_:%.+]] = builtin.unrealized_conversion_cast [[VAR_27_]] : i8 to ui8
// CHECK:           krnl.store [[VAR_7_]], [[RES_1_]][] : memref<f32>
// CHECK:           krnl.store [[VAR_28_]], [[RES_2_]][] : memref<ui8>
// CHECK-DAG:       [[VAR_29_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_29_]], [[RES_5_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_5_]]) : (memref<?x2xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK-DAG:       [[VAR_30_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_30_]], [[RES_6_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_14_:%.+]] = memref.reshape [[RES_]]([[RES_]]_13) : (memref<?x2xui8>, memref<1xindex>) -> memref<?xui8>
// CHECK-DAG:       [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_4_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]])){
// CHECK:             [[VAR_32_2_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_reshape_]]{{.}}[[VAR_32_2_]]{{.}} : memref<?xf32>
// CHECK:             [[LOAD_RES_3_MEM_1_:%.+]] = arith.divf [[LOAD_PARAM_0_MEM_1_]], [[VAR_7_]] : f32
// CHECK:             [[VAR_35_2_:%.+]] = math.floor [[LOAD_RES_3_MEM_1_]] : f32
// CHECK:             [[VAR_36_:%.+]] = arith.subf [[LOAD_RES_3_MEM_1_]], [[VAR_35_2_]] : f32
// CHECK-DAG:         [[VAR_37_:%.+]] = arith.cmpf ogt, [[VAR_36_]], [[CST_5_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_38_:%.+]] = arith.addf [[VAR_35_2_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_39_:%.+]] = arith.select [[VAR_37_]], [[VAR_38_]], [[VAR_35_2_]] : f32
// CHECK-DAG:         [[VAR_40_:%.+]] = arith.mulf [[VAR_35_2_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_41_:%.+]] = math.floor [[VAR_40_]] : f32
// CHECK:             [[VAR_42_:%.+]] = arith.mulf [[VAR_41_]], [[CST_2_dot_000000_]] : f32
// CHECK:             [[VAR_43_:%.+]] = arith.subf [[VAR_35_2_]], [[VAR_42_]] : f32
// CHECK-DAG:         [[VAR_44_:%.+]] = arith.cmpf oeq, [[VAR_43_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_45_:%.+]] = arith.addf [[VAR_35_2_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_46_:%.+]] = arith.select [[VAR_44_]], [[VAR_45_]], [[VAR_35_2_]] : f32
// CHECK-DAG:         [[VAR_47_:%.+]] = arith.cmpf oeq, [[VAR_36_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_48_:%.+]] = arith.select [[VAR_47_]], [[VAR_46_]], [[VAR_39_]] : f32
// CHECK:             [[VAR_49_:%.+]] = arith.addf [[VAR_48_]], [[VAR_25_]] : f32
// CHECK:             [[VAR_50_:%.+]] = arith.maxnumf [[VAR_49_]], [[CST_0_dot_000000_]] : f32
// CHECK:             [[VAR_51_:%.+]] = arith.minnumf [[VAR_50_]], [[CST_2_dot_550000_]] : f32
// CHECK:             [[VAR_52_:%.+]] = arith.fptoui [[VAR_51_]] : f32 to i32
// CHECK:             [[VAR_53_:%.+]] = arith.trunci [[VAR_52_]] : i32 to i8
// CHECK:             [[VAR_54_:%.+]] = builtin.unrealized_conversion_cast [[VAR_53_]] : i8 to ui8
// CHECK:             krnl.store [[VAR_54_]], [[VAR_reshape_14_]]{{.}}[[VAR_32_2_]]{{.}} : memref<?xui8>
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_]]_6, [[RES_]]_7 : memref<?x2xui8>, memref<f32>, memref<ui8>
// CHECK:         }
}

