// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

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
// CHECK:           krnl.define_loops 0
// CHECK:           krnl.iterate() with (){
// CHECK:             krnl.get_induction_var_value() : () -> ()
// CHECK:             krnl.store [[CST_0_1_]], [[RES_5_]][] : memref<f32>
// CHECK:           }
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_11_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_2_]] : memref<?x2xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[VAR_dim_11_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[VAR_33_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_33_]]#0, [[VAR_33_]]#1] : memref<?x2xf32>
// CHECK-DAG:         [[LOAD_RES_5_MEM_:%.+]] = krnl.load [[RES_5_]][] : memref<f32>
// CHECK:             [[VAR_36_:%.+]] = arith.cmpf ogt, [[LOAD_RES_5_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             [[VAR_37_:%.+]] = arith.select [[VAR_36_]], [[LOAD_RES_5_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_37_]], [[RES_5_]][] : memref<f32>
// CHECK:           }
// CHECK:           [[RES_6_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.define_loops 0
// CHECK:           krnl.iterate() with (){
// CHECK:             krnl.get_induction_var_value() : () -> ()
// CHECK:             krnl.store [[CST_0_]], [[RES_6_]][] : memref<f32>
// CHECK:           }
// CHECK-DAG:       [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_13_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_2_]] : memref<?x2xf32>
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to [[VAR_dim_13_]], [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 2){
// CHECK:             [[VAR_33_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_33_1_]]#0, [[VAR_33_1_]]#1] : memref<?x2xf32>
// CHECK-DAG:         [[LOAD_RES_5_MEM_1_:%.+]] = krnl.load [[RES_6_]][] : memref<f32>
// CHECK:             [[VAR_36_1_:%.+]] = arith.cmpf olt, [[LOAD_RES_5_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:             [[VAR_37_1_:%.+]] = arith.select [[VAR_36_1_]], [[LOAD_RES_5_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:             krnl.store [[VAR_37_1_]], [[RES_6_]][] : memref<f32>
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
// CHECK:           [[VAR_10_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_7_]] : f32
// CHECK:           [[VAR_11_:%.+]] = arith.divf [[VAR_10_]], [[VAR_9_]] : f32
// CHECK:           [[VAR_12_:%.+]] = arith.cmpf olt, [[VAR_11_]], [[CST_0_dot_000000_]] : f32
// CHECK:           [[VAR_13_:%.+]] = arith.select [[VAR_12_]], [[CST_0_dot_000000_]], [[VAR_11_]] : f32
// CHECK:           [[VAR_14_:%.+]] = arith.cmpf olt, [[VAR_13_]], [[CST_2_dot_550000_]] : f32
// CHECK:           [[VAR_15_:%.+]] = arith.select [[VAR_14_]], [[VAR_13_]], [[CST_2_dot_550000_]] : f32
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
// CHECK:           [[VAR_30_:%.+]] = arith.fptoui [[VAR_29_]] : f32 to i8
// CHECK:           [[VAR_31_:%.+]] = builtin.unrealized_conversion_cast [[VAR_30_]] : i8 to ui8
// CHECK:           krnl.store [[VAR_31_]], [[RES_2_]][] : memref<ui8>
// CHECK:           [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_4_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]]), [[LOOP_2_]]#1 -> [[I_5_:%.+]] = 0 to 2){
// CHECK:             [[VAR_33_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_33_2_]]#0, [[VAR_33_2_]]#1] : memref<?x2xf32>
// CHECK:             [[LOAD_RES_5_MEM_1_:%.+]] = arith.divf [[LOAD_PARAM_0_MEM_2_]], [[VAR_9_]] : f32
// CHECK:             [[VAR_36_2_:%.+]] = math.floor [[LOAD_RES_5_MEM_1_]] : f32
// CHECK:             [[VAR_37_2_:%.+]] = arith.subf [[LOAD_RES_5_MEM_1_]], [[VAR_36_2_]] : f32
// CHECK-DAG:         [[VAR_38_:%.+]] = arith.cmpf ogt, [[VAR_37_2_]], [[CST_5_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_39_:%.+]] = arith.addf [[VAR_36_2_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_40_:%.+]] = arith.select [[VAR_38_]], [[VAR_39_]], [[VAR_36_2_]] : f32
// CHECK-DAG:         [[VAR_41_:%.+]] = arith.mulf [[VAR_36_2_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_42_:%.+]] = math.floor [[VAR_41_]] : f32
// CHECK:             [[VAR_43_:%.+]] = arith.mulf [[VAR_42_]], [[CST_2_dot_000000_]] : f32
// CHECK:             [[VAR_44_:%.+]] = arith.subf [[VAR_36_2_]], [[VAR_43_]] : f32
// CHECK-DAG:         [[VAR_45_:%.+]] = arith.cmpf oeq, [[VAR_44_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_46_:%.+]] = arith.addf [[VAR_36_2_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_47_:%.+]] = arith.select [[VAR_45_]], [[VAR_46_]], [[VAR_36_2_]] : f32
// CHECK-DAG:         [[VAR_48_:%.+]] = arith.cmpf oeq, [[VAR_37_2_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_49_:%.+]] = arith.select [[VAR_48_]], [[VAR_47_]], [[VAR_40_]] : f32
// CHECK:             [[VAR_50_:%.+]] = arith.addf [[VAR_49_]], [[VAR_29_]] : f32
// CHECK:             [[VAR_51_:%.+]] = arith.cmpf olt, [[VAR_50_]], [[CST_0_dot_000000_]] : f32
// CHECK:             [[VAR_52_:%.+]] = arith.select [[VAR_51_]], [[CST_0_dot_000000_]], [[VAR_50_]] : f32
// CHECK:             [[VAR_53_:%.+]] = arith.cmpf olt, [[VAR_52_]], [[CST_2_dot_550000_]] : f32
// CHECK:             [[VAR_54_:%.+]] = arith.select [[VAR_53_]], [[VAR_52_]], [[CST_2_dot_550000_]] : f32
// CHECK:             [[VAR_55_:%.+]] = arith.fptoui [[VAR_54_]] : f32 to i8
// CHECK:             [[VAR_56_:%.+]] = builtin.unrealized_conversion_cast [[VAR_55_]] : i8 to ui8
// CHECK:             krnl.store [[VAR_56_]], [[RES_]]{{.}}[[VAR_33_2_]]#0, [[VAR_33_2_]]#1] : memref<?x2xui8>
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_]]_6, [[RES_]]_7 : memref<?x2xui8>, memref<f32>, memref<ui8>
// CHECK:         }
}
