// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl=enable-fast-math --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

// Test fast math where the divide by scale is replaced by mutiply by the reciprocal of the scale.
// -----


func.func @test_quantize_linear_ui8(%arg0: tensor<6xf32>, %arg1: tensor<f32>, %arg2: tensor<ui8>) -> tensor<6xui8> {
  %0 = "onnx.QuantizeLinear"(%arg0, %arg1, %arg2) {axis = 1 : si64} : (tensor<6xf32>, tensor<f32>, tensor<ui8>) -> tensor<6xui8>
  return %0 : tensor<6xui8>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_quantize_linear_ui8
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<6xf32>, [[PARAM_1_:%.+]]: memref<f32>, [[PARAM_2_:%.+]]: memref<ui8>) -> memref<6xui8> {
// CHECK-DAG:       [[CST_5_dot_000000_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:       [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_2_dot_550000_:%.+]] = arith.constant 2.550000e+02 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<6xui8>
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]][] : memref<f32>
// CHECK-DAG:       [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]][] : memref<ui8>
// CHECK:           [[VAR_2_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_2_MEM_]] : ui8 to i8
// CHECK:           [[VAR_3_:%.+]] = arith.extui [[VAR_2_]] : i8 to i32
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.uitofp [[VAR_3_]] : i32 to f32
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 6){
// CHECK:             [[VAR_7_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_7_]]{{.}} : memref<6xf32>
// CHECK:             [[VAR_9_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_]], [[VAR_5_]] : f32
// CHECK:             [[VAR_10_:%.+]] = math.floor [[VAR_9_]] : f32
// CHECK:             [[VAR_11_:%.+]] = arith.subf [[VAR_9_]], [[VAR_10_]] : f32
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.cmpf ogt, [[VAR_11_]], [[CST_5_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.addf [[VAR_10_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.select [[VAR_12_]], [[VAR_13_]], [[VAR_10_]] : f32
// CHECK-DAG:         [[VAR_15_:%.+]] = arith.mulf [[VAR_10_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_16_:%.+]] = math.floor [[VAR_15_]] : f32
// CHECK:             [[VAR_17_:%.+]] = arith.mulf [[VAR_16_]], [[CST_2_dot_000000_]] : f32
// CHECK:             [[VAR_18_:%.+]] = arith.subf [[VAR_10_]], [[VAR_17_]] : f32
// CHECK-DAG:         [[VAR_19_:%.+]] = arith.cmpf oeq, [[VAR_18_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_20_:%.+]] = arith.addf [[VAR_10_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_21_:%.+]] = arith.select [[VAR_19_]], [[VAR_20_]], [[VAR_10_]] : f32
// CHECK-DAG:         [[VAR_22_:%.+]] = arith.cmpf oeq, [[VAR_11_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_23_:%.+]] = arith.select [[VAR_22_]], [[VAR_21_]], [[VAR_14_]] : f32
// CHECK:             [[VAR_24_:%.+]] = arith.addf [[VAR_23_]], [[VAR_4_]] : f32
// CHECK:             [[VAR_25_:%.+]] = arith.maxnumf [[VAR_24_]], [[CST_0_dot_000000_]] : f32
// CHECK:             [[VAR_26_:%.+]] = arith.minnumf [[VAR_25_]], [[CST_2_dot_550000_]] : f32
// CHECK:             [[VAR_27_:%.+]] = arith.fptoui [[VAR_26_]] : f32 to i32
// CHECK:             [[VAR_28_:%.+]] = arith.trunci [[VAR_27_]] : i32 to i8
// CHECK:             [[VAR_29_:%.+]] = builtin.unrealized_conversion_cast [[VAR_28_]] : i8 to ui8
// CHECK:             krnl.store [[VAR_29_]], [[RES_]]{{.}}[[VAR_7_]]{{.}} : memref<6xui8>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<6xui8>
// CHECK:         }
}

// -----


func.func @test_quantize_linear_i8(%arg0: tensor<6xf32>, %arg1: tensor<f32>, %arg2: tensor<i8>) -> tensor<6xi8> {
  %0 = "onnx.QuantizeLinear"(%arg0, %arg1, %arg2) {axis = 1 : si64} : (tensor<6xf32>, tensor<f32>, tensor<i8>) -> tensor<6xi8>
  return %0 : tensor<6xi8>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_quantize_linear_i8
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<6xf32>, [[PARAM_1_:%.+]]: memref<f32>, [[PARAM_2_:%.+]]: memref<i8>) -> memref<6xi8> {
// CHECK-DAG:       [[CST_5_dot_000000_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:       [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_minus_1_dot_280000_:%.+]] = arith.constant -1.280000e+02 : f32
// CHECK-DAG:       [[CST_1_dot_270000_:%.+]] = arith.constant 1.270000e+02 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<6xi8>
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]][] : memref<f32>
// CHECK-DAG:       [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]][] : memref<i8>
// CHECK:           [[VAR_2_:%.+]] = arith.extsi [[LOAD_PARAM_2_MEM_]] : i8 to i32
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.sitofp [[VAR_2_]] : i32 to f32
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 6){
// CHECK:             [[VAR_6_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]]{{.}} : memref<6xf32>
// CHECK:             [[VAR_8_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_]], [[VAR_4_]] : f32
// CHECK:             [[VAR_9_:%.+]] = math.floor [[VAR_8_]] : f32
// CHECK:             [[VAR_10_:%.+]] = arith.subf [[VAR_8_]], [[VAR_9_]] : f32
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.cmpf ogt, [[VAR_10_]], [[CST_5_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.addf [[VAR_9_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.select [[VAR_11_]], [[VAR_12_]], [[VAR_9_]] : f32
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.mulf [[VAR_9_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_15_:%.+]] = math.floor [[VAR_14_]] : f32
// CHECK:             [[VAR_16_:%.+]] = arith.mulf [[VAR_15_]], [[CST_2_dot_000000_]] : f32
// CHECK:             [[VAR_17_:%.+]] = arith.subf [[VAR_9_]], [[VAR_16_]] : f32
// CHECK-DAG:         [[VAR_18_:%.+]] = arith.cmpf oeq, [[VAR_17_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_19_:%.+]] = arith.addf [[VAR_9_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_20_:%.+]] = arith.select [[VAR_18_]], [[VAR_19_]], [[VAR_9_]] : f32
// CHECK-DAG:         [[VAR_21_:%.+]] = arith.cmpf oeq, [[VAR_10_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_22_:%.+]] = arith.select [[VAR_21_]], [[VAR_20_]], [[VAR_13_]] : f32
// CHECK:             [[VAR_23_:%.+]] = arith.addf [[VAR_22_]], [[VAR_3_]] : f32
// CHECK:             [[VAR_24_:%.+]] = arith.maxnumf [[VAR_23_]], [[CST_minus_1_dot_280000_]] : f32
// CHECK:             [[VAR_25_:%.+]] = arith.minnumf [[VAR_24_]], [[CST_1_dot_270000_]] : f32
// CHECK:             [[VAR_26_:%.+]] = arith.fptosi [[VAR_25_]] : f32 to i32
// CHECK:             [[VAR_27_:%.+]] = arith.trunci [[VAR_26_]] : i32 to i8
// CHECK:             krnl.store [[VAR_27_]], [[RES_]]{{.}}[[VAR_6_]]{{.}} : memref<6xi8>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<6xi8>
// CHECK:         }
}

