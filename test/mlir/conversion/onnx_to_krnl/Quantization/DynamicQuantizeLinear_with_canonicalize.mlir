// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// -----

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

func.func @test_dynamic_quantize_linear(%arg0: tensor<?x2xf32>) -> (tensor<?x2xui8>, tensor<f32>, tensor<ui8>) {
  %y, %y_scale, %y_zero_point = "onnx.DynamicQuantizeLinear"(%arg0) : (tensor<?x2xf32>) -> (tensor<?x2xui8>, tensor<f32>, tensor<ui8>)
  return %y, %y_scale, %y_zero_point:  tensor<?x2xui8>, tensor<f32>, tensor<ui8>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @test_dynamic_quantize_linear
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x2xf32>) -> (memref<?x2xui8>, memref<f32>, memref<ui8>) {
// CHECK-DAG:       [[CST_5_dot_000000_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:       [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0x7F800000 : f32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_2_dot_550000_:%.+]] = arith.constant 2.550000e+02 : f32
// CHECK-DAG:       [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_2_]] : memref<?x2xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x2xui8>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() : memref<f32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() : memref<ui8>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.store [[CST_2_dot_550000_]], [[RES_3_]][] : memref<f32>
// CHECK:           [[RES_4_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.store [[CST_0_dot_000000_]], [[RES_4_]][] : memref<f32>
// CHECK:           [[RES_5_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.memset [[RES_5_]], [[CST_0_1_]] : memref<f32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_11_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_2_]] : memref<?x2xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[VAR_dim_11_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[VAR_31_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_31_]]#0, [[VAR_31_]]#1] : memref<?x2xf32>
// CHECK-DAG:         [[LOAD_RES_5_MEM_:%.+]] = krnl.load [[RES_5_]][] : memref<f32>
// CHECK:             [[VAR_34_:%.+]] = arith.maxnumf [[LOAD_RES_5_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_34_]], [[RES_5_]][] : memref<f32>
// CHECK:           }
// CHECK:           [[RES_6_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.memset [[RES_6_]], [[CST_0_]] : memref<f32>
// CHECK-DAG:       [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_13_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_2_]] : memref<?x2xf32>
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to [[VAR_dim_13_]], [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 2){
// CHECK:             [[VAR_31_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_31_1_]]#0, [[VAR_31_1_]]#1] : memref<?x2xf32>
// CHECK-DAG:         [[LOAD_RES_5_MEM_1_:%.+]] = krnl.load [[RES_6_]][] : memref<f32>
// CHECK:             [[VAR_34_1_:%.+]] = arith.minnumf [[LOAD_RES_5_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:             krnl.store [[VAR_34_1_]], [[RES_6_]][] : memref<f32>
// CHECK:           }
// CHECK-DAG:       [[LOAD_RES_5_MEM_2_:%.+]] = krnl.load [[RES_5_]][] : memref<f32>
// CHECK-DAG:       [[LOAD_RES_6_MEM_:%.+]] = krnl.load [[RES_6_]][] : memref<f32>
// CHECK:           [[VAR_4_:%.+]] = arith.cmpf ogt, [[LOAD_RES_5_MEM_2_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.select [[VAR_4_]], [[LOAD_RES_5_MEM_2_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.cmpf olt, [[LOAD_RES_6_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK:           [[VAR_7_:%.+]] = arith.select [[VAR_6_]], [[LOAD_RES_6_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK:           [[VAR_8_:%.+]] = arith.subf [[VAR_5_]], [[VAR_7_]] : f32
// CHECK:           [[VAR_9_:%.+]] = arith.divf [[VAR_8_]], [[CST_2_dot_550000_]] : f32
// CHECK:           krnl.store [[VAR_9_]], [[RES_1_]][] : memref<f32>
// CHECK:           [[VAR_10_:%.+]] = arith.divf [[VAR_7_]], [[VAR_9_]] : f32
// CHECK:           [[VAR_11_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_10_]] : f32
// CHECK:           [[VAR_12_:%.+]] = arith.maxnumf [[VAR_11_]], [[CST_0_dot_000000_]] : f32
// CHECK:           [[VAR_13_:%.+]] = arith.minnumf [[VAR_12_]], [[CST_2_dot_550000_]] : f32
// CHECK:           [[VAR_14_:%.+]] = math.floor [[VAR_13_]] : f32
// CHECK:           [[VAR_15_:%.+]] = arith.subf [[VAR_13_]], [[VAR_14_]] : f32
// CHECK-DAG:       [[VAR_16_:%.+]] = arith.cmpf ogt, [[VAR_15_]], [[CST_5_dot_000000_]] : f32
// CHECK-DAG:       [[VAR_17_:%.+]] = arith.addf [[VAR_14_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_18_:%.+]] = arith.select [[VAR_16_]], [[VAR_17_]], [[VAR_14_]] : f32
// CHECK-DAG:       [[VAR_19_:%.+]] = arith.mulf [[VAR_14_]], [[CST_5_dot_000000_]] : f32
// CHECK:           [[VAR_20_:%.+]] = math.floor [[VAR_19_]] : f32
// CHECK:           [[VAR_21_:%.+]] = arith.mulf [[VAR_20_]], [[CST_2_dot_000000_]] : f32
// CHECK:           [[VAR_22_:%.+]] = arith.subf [[VAR_14_]], [[VAR_21_]] : f32
// CHECK-DAG:       [[VAR_23_:%.+]] = arith.cmpf oeq, [[VAR_22_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:       [[VAR_24_:%.+]] = arith.addf [[VAR_14_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_25_:%.+]] = arith.select [[VAR_23_]], [[VAR_24_]], [[VAR_14_]] : f32
// CHECK-DAG:       [[VAR_26_:%.+]] = arith.cmpf oeq, [[VAR_15_]], [[CST_5_dot_000000_]] : f32
// CHECK:           [[VAR_27_:%.+]] = arith.select [[VAR_26_]], [[VAR_25_]], [[VAR_18_]] : f32
// CHECK:           [[VAR_28_:%.+]] = arith.fptoui [[VAR_27_]] : f32 to i8
// CHECK:           [[VAR_29_:%.+]] = builtin.unrealized_conversion_cast [[VAR_28_]] : i8 to ui8
// CHECK:           krnl.store [[VAR_29_]], [[RES_2_]][] : memref<ui8>
// CHECK:           [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_4_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]]), [[LOOP_2_]]#1 -> [[I_5_:%.+]] = 0 to 2){
// CHECK:             [[VAR_31_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_31_2_]]#0, [[VAR_31_2_]]#1] : memref<?x2xf32>
// CHECK:             [[LOAD_RES_5_MEM_1_:%.+]] = arith.divf [[LOAD_PARAM_0_MEM_2_]], [[VAR_9_]] : f32
// CHECK:             [[VAR_34_2_:%.+]] = math.floor [[LOAD_RES_5_MEM_1_]] : f32
// CHECK:             [[VAR_35_:%.+]] = arith.subf [[LOAD_RES_5_MEM_1_]], [[VAR_34_2_]] : f32
// CHECK-DAG:         [[VAR_36_:%.+]] = arith.cmpf ogt, [[VAR_35_]], [[CST_5_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_37_:%.+]] = arith.addf [[VAR_34_2_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_38_:%.+]] = arith.select [[VAR_36_]], [[VAR_37_]], [[VAR_34_2_]] : f32
// CHECK-DAG:         [[VAR_39_:%.+]] = arith.mulf [[VAR_34_2_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_40_:%.+]] = math.floor [[VAR_39_]] : f32
// CHECK:             [[VAR_41_:%.+]] = arith.mulf [[VAR_40_]], [[CST_2_dot_000000_]] : f32
// CHECK:             [[VAR_42_:%.+]] = arith.subf [[VAR_34_2_]], [[VAR_41_]] : f32
// CHECK-DAG:         [[VAR_43_:%.+]] = arith.cmpf oeq, [[VAR_42_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_44_:%.+]] = arith.addf [[VAR_34_2_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_45_:%.+]] = arith.select [[VAR_43_]], [[VAR_44_]], [[VAR_34_2_]] : f32
// CHECK-DAG:         [[VAR_46_:%.+]] = arith.cmpf oeq, [[VAR_35_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_47_:%.+]] = arith.select [[VAR_46_]], [[VAR_45_]], [[VAR_38_]] : f32
// CHECK:             [[VAR_48_:%.+]] = arith.addf [[VAR_47_]], [[VAR_27_]] : f32
// CHECK:             [[VAR_49_:%.+]] = arith.maxnumf [[VAR_48_]], [[CST_0_dot_000000_]] : f32
// CHECK:             [[VAR_50_:%.+]] = arith.minnumf [[VAR_49_]], [[CST_2_dot_550000_]] : f32
// CHECK:             [[VAR_51_:%.+]] = arith.fptoui [[VAR_50_]] : f32 to i8
// CHECK:             [[VAR_52_:%.+]] = builtin.unrealized_conversion_cast [[VAR_51_]] : i8 to ui8
// CHECK:             krnl.store [[VAR_52_]], [[RES_]]{{.}}[[VAR_31_2_]]#0, [[VAR_31_2_]]#1] : memref<?x2xui8>
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_]]_6, [[RES_]]_7 : memref<?x2xui8>, memref<f32>, memref<ui8>
// CHECK:         }
}

