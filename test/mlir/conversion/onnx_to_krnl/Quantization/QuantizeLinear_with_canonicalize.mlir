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
// CHECK:             [[VAR_9_:%.+]] = math.roundeven [[VAR_8_]] : f32
// CHECK:             [[VAR_10_:%.+]] = arith.addf [[VAR_9_]], [[VAR_4_]] : f32
// CHECK:             [[VAR_11_:%.+]] = arith.maxnumf [[VAR_10_]], [[CST_0_dot_000000_]] : f32
// CHECK:             [[VAR_12_:%.+]] = arith.minnumf [[VAR_11_]], [[CST_2_dot_550000_]] : f32
// CHECK:             [[VAR_13_:%.+]] = arith.fptoui [[VAR_12_]] : f32 to i32
// CHECK:             [[VAR_14_:%.+]] = arith.trunci [[VAR_13_]] : i32 to i8
// CHECK:             [[VAR_15_:%.+]] = builtin.unrealized_conversion_cast [[VAR_14_]] : i8 to ui8
// CHECK:             krnl.store [[VAR_15_]], [[RES_]]{{.}}[[VAR_6_]]{{.}} : memref<6xui8>
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
// CHECK:             [[VAR_8_:%.+]] = math.roundeven [[VAR_7_]] : f32
// CHECK:             [[VAR_9_:%.+]] = arith.addf [[VAR_8_]], [[VAR_3_]] : f32
// CHECK:             [[VAR_10_:%.+]] = arith.maxnumf [[VAR_9_]], [[CST_minus_1_dot_280000_]] : f32
// CHECK:             [[VAR_11_:%.+]] = arith.minnumf [[VAR_10_]], [[CST_1_dot_270000_]] : f32
// CHECK:             [[VAR_12_:%.+]] = arith.fptosi [[VAR_11_]] : f32 to i32
// CHECK:             [[VAR_13_:%.+]] = arith.trunci [[VAR_12_]] : i32 to i8
// CHECK:             krnl.store [[VAR_13_]], [[RES_]]{{.}}[[VAR_5_]]{{.}} : memref<6xi8>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<6xi8>
// CHECK:         }
}

