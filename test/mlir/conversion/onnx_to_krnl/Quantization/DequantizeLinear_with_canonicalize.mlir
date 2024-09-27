// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

// -----


func.func @test_dequantizelinear_i8(%arg0: tensor<4xi8>, %arg1: tensor<f32>, %arg2: tensor<i8>) -> tensor<4xf32> {
  %0 = "onnx.DequantizeLinear"(%arg0, %arg1, %arg2) {axis = 1 : si64} : (tensor<4xi8>, tensor<f32>, tensor<i8>) -> tensor<4xf32>
  return %0 : tensor<4xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_dequantizelinear_i8
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4xi8>, [[PARAM_1_:%.+]]: memref<f32>, [[PARAM_2_:%.+]]: memref<i8>) -> memref<4xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 4){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]{{.}} : memref<4xi8>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]][] : memref<f32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]][] : memref<i8>
// CHECK:             [[VAR_5_:%.+]] = arith.extsi [[LOAD_PARAM_0_MEM_]] : i8 to i32
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.sitofp [[VAR_5_]] : i32 to f32
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.extsi [[LOAD_PARAM_2_MEM_]] : i8 to i32
// CHECK:             [[VAR_8_:%.+]] = arith.sitofp [[VAR_7_]] : i32 to f32
// CHECK:             [[VAR_9_:%.+]] = arith.subf [[VAR_6_]], [[VAR_8_]] : f32
// CHECK:             [[VAR_10_:%.+]] = arith.mulf [[VAR_9_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             krnl.store [[VAR_10_]], [[RES_]]{{.}}[[VAR_1_]]{{.}} : memref<4xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<4xf32>
// CHECK:         }
}

// -----


func.func @test_dequantizelinear_ui8(%arg0: tensor<4xui8>, %arg1: tensor<f32>, %arg2: tensor<ui8>) -> tensor<4xf32> {
  %0 = "onnx.DequantizeLinear"(%arg0, %arg1, %arg2) {axis = 1 : si64} : (tensor<4xui8>, tensor<f32>, tensor<ui8>) -> tensor<4xf32>
  return %0 : tensor<4xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_dequantizelinear_ui8
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4xui8>, [[PARAM_1_:%.+]]: memref<f32>, [[PARAM_2_:%.+]]: memref<ui8>) -> memref<4xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 4){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]{{.}} : memref<4xui8>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]][] : memref<f32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]][] : memref<ui8>
// CHECK:             [[VAR_5_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_]] : ui8 to i8
// CHECK:             [[VAR_6_:%.+]] = arith.extui [[VAR_5_]] : i8 to i32
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.uitofp [[VAR_6_]] : i32 to f32
// CHECK-DAG:         [[VAR_8_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_2_MEM_]] : ui8 to i8
// CHECK:             [[VAR_9_:%.+]] = arith.extui [[VAR_8_]] : i8 to i32
// CHECK:             [[VAR_10_:%.+]] = arith.uitofp [[VAR_9_]] : i32 to f32
// CHECK:             [[VAR_11_:%.+]] = arith.subf [[VAR_7_]], [[VAR_10_]] : f32
// CHECK:             [[VAR_12_:%.+]] = arith.mulf [[VAR_11_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             krnl.store [[VAR_12_]], [[RES_]]{{.}}[[VAR_1_]]{{.}} : memref<4xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<4xf32>
// CHECK:         }
}

// -----

func.func @test_dequantizelinear_i32(%arg0: tensor<4xi32>, %arg1: tensor<f32>, %arg2: tensor<i32>) -> tensor<4xf32> {
  %0 = "onnx.DequantizeLinear"(%arg0, %arg1, %arg2) {axis = 1 : si64} : (tensor<4xi32>, tensor<f32>, tensor<i32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>

// CHECK-LABEL:  func.func @test_dequantizelinear_i32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4xi32>, [[PARAM_1_:%.+]]: memref<f32>, [[PARAM_2_:%.+]]: memref<i32>) -> memref<4xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 4){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]{{.}} : memref<4xi32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]][] : memref<f32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]][] : memref<i32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.sitofp [[LOAD_PARAM_2_MEM_]] : i32 to f32
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.sitofp [[LOAD_PARAM_0_MEM_]] : i32 to f32
// CHECK:             [[VAR_7_:%.+]] = arith.subf [[VAR_6_]], [[VAR_5_]] : f32
// CHECK:             [[VAR_8_:%.+]] = arith.mulf [[VAR_7_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             krnl.store [[VAR_8_]], [[RES_]]{{.}}[[VAR_1_]]{{.}} : memref<4xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<4xf32>
// CHECK:         }
}
