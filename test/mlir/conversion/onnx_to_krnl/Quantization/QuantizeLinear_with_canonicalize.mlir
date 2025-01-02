// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

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
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 6){
// CHECK:             [[VAR_6_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]]{{.}} : memref<6xf32>
// CHECK:             [[VAR_8_:%.+]] = arith.divf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
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
// CHECK:             [[VAR_23_:%.+]] = arith.addf [[VAR_22_]], [[VAR_4_]] : f32
// CHECK:             [[VAR_24_:%.+]] = arith.maxnumf [[VAR_23_]], [[CST_0_dot_000000_]] : f32
// CHECK:             [[VAR_25_:%.+]] = arith.minnumf [[VAR_24_]], [[CST_2_dot_550000_]] : f32
// CHECK:             [[VAR_26_:%.+]] = arith.fptoui [[VAR_25_]] : f32 to i32
// CHECK:             [[VAR_27_:%.+]] = arith.trunci [[VAR_26_]] : i32 to i8
// CHECK:             [[VAR_28_:%.+]] = builtin.unrealized_conversion_cast [[VAR_27_]] : i8 to ui8
// CHECK:             krnl.store [[VAR_28_]], [[RES_]]{{.}}[[VAR_6_]]{{.}} : memref<6xui8>
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
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 6){
// CHECK:             [[VAR_5_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_5_]]{{.}} : memref<6xf32>
// CHECK:             [[VAR_7_:%.+]] = arith.divf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             [[VAR_8_:%.+]] = math.floor [[VAR_7_]] : f32
// CHECK:             [[VAR_9_:%.+]] = arith.subf [[VAR_7_]], [[VAR_8_]] : f32
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.cmpf ogt, [[VAR_9_]], [[CST_5_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.addf [[VAR_8_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.select [[VAR_10_]], [[VAR_11_]], [[VAR_8_]] : f32
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.mulf [[VAR_8_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_14_:%.+]] = math.floor [[VAR_13_]] : f32
// CHECK:             [[VAR_15_:%.+]] = arith.mulf [[VAR_14_]], [[CST_2_dot_000000_]] : f32
// CHECK:             [[VAR_16_:%.+]] = arith.subf [[VAR_8_]], [[VAR_15_]] : f32
// CHECK-DAG:         [[VAR_17_:%.+]] = arith.cmpf oeq, [[VAR_16_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_18_:%.+]] = arith.addf [[VAR_8_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_19_:%.+]] = arith.select [[VAR_17_]], [[VAR_18_]], [[VAR_8_]] : f32
// CHECK-DAG:         [[VAR_20_:%.+]] = arith.cmpf oeq, [[VAR_9_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_21_:%.+]] = arith.select [[VAR_20_]], [[VAR_19_]], [[VAR_12_]] : f32
// CHECK:             [[VAR_22_:%.+]] = arith.addf [[VAR_21_]], [[VAR_3_]] : f32
// CHECK:             [[VAR_23_:%.+]] = arith.maxnumf [[VAR_22_]], [[CST_minus_1_dot_280000_]] : f32
// CHECK:             [[VAR_24_:%.+]] = arith.minnumf [[VAR_23_]], [[CST_1_dot_270000_]] : f32
// CHECK:             [[VAR_25_:%.+]] = arith.fptosi [[VAR_24_]] : f32 to i32
// CHECK:             [[VAR_26_:%.+]] = arith.trunci [[VAR_25_]] : i32 to i8
// CHECK:             krnl.store [[VAR_26_]], [[RES_]]{{.}}[[VAR_5_]]{{.}} : memref<6xi8>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<6xi8>
// CHECK:         }
}

// -----

func.func @test_quantize_linear_ui8_scalar(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<ui8>) -> tensor<ui8> {
  %0 = "onnx.QuantizeLinear"(%arg0, %arg1, %arg2) {axis = 1 : si64} : (tensor<f32>, tensor<f32>, tensor<ui8>) -> tensor<ui8>
  return %0 : tensor<ui8>

// CHECK-LABEL:  func.func @test_quantize_linear_ui8_scalar
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<f32>, [[PARAM_1_:%.+]]: memref<f32>, [[PARAM_2_:%.+]]: memref<ui8>) -> memref<ui8> {
// CHECK-DAG:       [[CST_5_dot_000000_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:       [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_2_dot_550000_:%.+]] = arith.constant 2.550000e+02 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<ui8>
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]][] : memref<f32>
// CHECK-DAG:       [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]][] : memref<ui8>
// CHECK:           [[VAR_2_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_2_MEM_]] : ui8 to i8
// CHECK:           [[VAR_3_:%.+]] = arith.extui [[VAR_2_]] : i8 to i32
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.uitofp [[VAR_3_]] : i32 to f32
// CHECK-DAG:       [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]][] : memref<f32>
// CHECK:           [[VAR_6_:%.+]] = arith.divf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:           [[VAR_7_:%.+]] = math.floor [[VAR_6_]] : f32
// CHECK:           [[VAR_8_:%.+]] = arith.subf [[VAR_6_]], [[VAR_7_]] : f32
// CHECK-DAG:       [[VAR_9_:%.+]] = arith.cmpf ogt, [[VAR_8_]], [[CST_5_dot_000000_]] : f32
// CHECK-DAG:       [[VAR_10_:%.+]] = arith.addf [[VAR_7_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]] = arith.select [[VAR_9_]], [[VAR_10_]], [[VAR_7_]] : f32
// CHECK-DAG:       [[VAR_12_:%.+]] = arith.mulf [[VAR_7_]], [[CST_5_dot_000000_]] : f32
// CHECK:           [[VAR_13_:%.+]] = math.floor [[VAR_12_]] : f32
// CHECK:           [[VAR_14_:%.+]] = arith.mulf [[VAR_13_]], [[CST_2_dot_000000_]] : f32
// CHECK:           [[VAR_15_:%.+]] = arith.subf [[VAR_7_]], [[VAR_14_]] : f32
// CHECK-DAG:       [[VAR_16_:%.+]] = arith.cmpf oeq, [[VAR_15_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:       [[VAR_17_:%.+]] = arith.addf [[VAR_7_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_18_:%.+]] = arith.select [[VAR_16_]], [[VAR_17_]], [[VAR_7_]] : f32
// CHECK-DAG:       [[VAR_19_:%.+]] = arith.cmpf oeq, [[VAR_8_]], [[CST_5_dot_000000_]] : f32
// CHECK:           [[VAR_20_:%.+]] = arith.select [[VAR_19_]], [[VAR_18_]], [[VAR_11_]] : f32
// CHECK:           [[VAR_21_:%.+]] = arith.addf [[VAR_20_]], [[VAR_4_]] : f32
// CHECK:           [[VAR_22_:%.+]] = arith.maxnumf [[VAR_21_]], [[CST_0_dot_000000_]] : f32
// CHECK:           [[VAR_23_:%.+]] = arith.minnumf [[VAR_22_]], [[CST_2_dot_550000_]] : f32
// CHECK:           [[VAR_24_:%.+]] = arith.fptoui [[VAR_23_]] : f32 to i32
// CHECK:           [[VAR_25_:%.+]] = arith.trunci [[VAR_24_]] : i32 to i8
// CHECK:           [[VAR_26_:%.+]] = builtin.unrealized_conversion_cast [[VAR_25_]] : i8 to ui8
// CHECK:           krnl.store [[VAR_26_]], [[RES_]][] : memref<ui8>
// CHECK:           return [[RES_]] : memref<ui8>
// CHECK:         }
}
