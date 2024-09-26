// RUN: onnx-mlir-opt --disable-quantization-zero-point --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Test quantization with disabled zero point

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

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
// CHECK:             [[VAR_4_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_]] : ui8 to i8
// CHECK:             [[VAR_5_:%.+]] = arith.extui [[VAR_4_]] : i8 to i32
// CHECK:             [[VAR_6_:%.+]] = arith.uitofp [[VAR_5_]] : i32 to f32
// CHECK:             [[VAR_7_:%.+]] = arith.mulf [[VAR_6_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             krnl.store [[VAR_7_]], [[RES_]]{{.}}[[VAR_1_]]{{.}} : memref<4xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<4xf32>
// CHECK:         }
}

// -----


func.func @test_dynamic_quantize_linear(%arg0: tensor<?x2xf32>) -> (tensor<?x2xui8>, tensor<f32>, tensor<ui8>) {
  %y, %y_scale, %y_zero_point = "onnx.DynamicQuantizeLinear"(%arg0) : (tensor<?x2xf32>) -> (tensor<?x2xui8>, tensor<f32>, tensor<ui8>)
  return %y, %y_scale, %y_zero_point:  tensor<?x2xui8>, tensor<f32>, tensor<ui8>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 2)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-LABEL:  func.func @test_dynamic_quantize_linear
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x2xf32>) -> (memref<?x2xui8>, memref<f32>, memref<ui8>) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i8
// CHECK-DAG:       [[CST_5_dot_000000_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:       [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[CST_0_2_:%.+]] = arith.constant 0x7F800000 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_2_dot_550000_:%.+]] = arith.constant 2.550000e+02 : f32
// CHECK-DAG:       [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_3_]] : memref<?x2xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x2xui8>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() : memref<f32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() : memref<ui8>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.memset [[RES_3_]], [[CST_0_2_]] : memref<f32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_9_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_3_]] : memref<?x2xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[VAR_dim_9_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[VAR_12_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_12_]]#0, [[VAR_12_]]#1] : memref<?x2xf32>
// CHECK-DAG:         [[LOAD_RES_3_MEM_:%.+]] = krnl.load [[RES_3_]][] : memref<f32>
// CHECK:             [[VAR_15_:%.+]] = arith.minnumf [[LOAD_RES_3_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_15_]], [[RES_3_]][] : memref<f32>
// CHECK:           }
// CHECK:           [[RES_4_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.memset [[RES_4_]], [[CST_0_1_]] : memref<f32>
// CHECK-DAG:       [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_11_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_3_]] : memref<?x2xf32>
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to [[VAR_dim_11_]], [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 2){
// CHECK:             [[VAR_12_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_12_1_]]#0, [[VAR_12_1_]]#1] : memref<?x2xf32>
// CHECK-DAG:         [[LOAD_RES_3_MEM_1_:%.+]] = krnl.load [[RES_4_]][] : memref<f32>
// CHECK:             [[VAR_15_1_:%.+]] = arith.maxnumf [[LOAD_RES_3_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:             krnl.store [[VAR_15_1_]], [[RES_4_]][] : memref<f32>
// CHECK:           }
// CHECK-DAG:       [[LOAD_RES_3_MEM_2_:%.+]] = krnl.load [[RES_3_]][] : memref<f32>
// CHECK-DAG:       [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]][] : memref<f32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.maxnumf [[LOAD_RES_4_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.minnumf [[LOAD_RES_3_MEM_2_]], [[CST_0_dot_000000_]] : f32
// CHECK:           [[VAR_6_:%.+]] = arith.subf [[VAR_4_]], [[VAR_5_]] : f32
// CHECK-DAG:       [[VAR_7_:%.+]] = arith.divf [[VAR_6_]], [[CST_2_dot_550000_]] : f32
// CHECK-DAG:       [[VAR_8_:%.+]] = builtin.unrealized_conversion_cast [[CST_0_]] : i8 to ui8
// CHECK:           krnl.store [[VAR_7_]], [[RES_1_]][] : memref<f32>
// CHECK:           krnl.store [[VAR_8_]], [[RES_2_]][] : memref<ui8>
// CHECK-DAG:       [[VAR_9_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_9_]], [[RES_5_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_5_]]) : (memref<?x2xf32>, memref<1xindex>) -> memref<?xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[VAR_10_]], [[RES_6_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_14_:%.+]] = memref.reshape [[RES_]]([[RES_]]_13) : (memref<?x2xui8>, memref<1xindex>) -> memref<?xui8>
// CHECK-DAG:       [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_4_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]])){
// CHECK:             [[VAR_12_2_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[VAR_reshape_]]{{.}}[[VAR_12_2_]]{{.}} : memref<?xf32>
// CHECK:             [[LOAD_RES_3_MEM_1_:%.+]] = arith.divf [[LOAD_PARAM_0_MEM_1_]], [[VAR_7_]] : f32
// CHECK:             [[VAR_15_2_:%.+]] = math.floor [[LOAD_RES_3_MEM_1_]] : f32
// CHECK:             [[VAR_16_:%.+]] = arith.subf [[LOAD_RES_3_MEM_1_]], [[VAR_15_2_]] : f32
// CHECK-DAG:         [[VAR_17_:%.+]] = arith.cmpf ogt, [[VAR_16_]], [[CST_5_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_18_:%.+]] = arith.addf [[VAR_15_2_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_19_:%.+]] = arith.select [[VAR_17_]], [[VAR_18_]], [[VAR_15_2_]] : f32
// CHECK-DAG:         [[VAR_20_:%.+]] = arith.mulf [[VAR_15_2_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_21_:%.+]] = math.floor [[VAR_20_]] : f32
// CHECK:             [[VAR_22_:%.+]] = arith.mulf [[VAR_21_]], [[CST_2_dot_000000_]] : f32
// CHECK:             [[VAR_23_:%.+]] = arith.subf [[VAR_15_2_]], [[VAR_22_]] : f32
// CHECK-DAG:         [[VAR_24_:%.+]] = arith.cmpf oeq, [[VAR_23_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_25_:%.+]] = arith.addf [[VAR_15_2_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_26_:%.+]] = arith.select [[VAR_24_]], [[VAR_25_]], [[VAR_15_2_]] : f32
// CHECK-DAG:         [[VAR_27_:%.+]] = arith.cmpf oeq, [[VAR_16_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_28_:%.+]] = arith.select [[VAR_27_]], [[VAR_26_]], [[VAR_19_]] : f32
// CHECK:             [[VAR_29_:%.+]] = arith.maxnumf [[VAR_28_]], [[CST_0_dot_000000_]] : f32
// CHECK:             [[VAR_30_:%.+]] = arith.minnumf [[VAR_29_]], [[CST_2_dot_550000_]] : f32
// CHECK:             [[VAR_31_:%.+]] = arith.fptoui [[VAR_30_]] : f32 to i32
// CHECK:             [[VAR_32_:%.+]] = arith.trunci [[VAR_31_]] : i32 to i8
// CHECK:             [[VAR_33_:%.+]] = builtin.unrealized_conversion_cast [[VAR_32_]] : i8 to ui8
// CHECK:             krnl.store [[VAR_33_]], [[VAR_reshape_14_]]{{.}}[[VAR_12_2_]]{{.}} : memref<?xui8>
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_]]_6, [[RES_]]_7 : memref<?x2xui8>, memref<f32>, memref<ui8>
// CHECK:         }
}

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
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 6){
// CHECK:             [[VAR_2_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]]{{.}} : memref<6xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.divf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             [[VAR_5_:%.+]] = math.floor [[VAR_4_]] : f32
// CHECK:             [[VAR_6_:%.+]] = arith.subf [[VAR_4_]], [[VAR_5_]] : f32
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.cmpf ogt, [[VAR_6_]], [[CST_5_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.addf [[VAR_5_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.select [[VAR_7_]], [[VAR_8_]], [[VAR_5_]] : f32
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.mulf [[VAR_5_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_11_:%.+]] = math.floor [[VAR_10_]] : f32
// CHECK:             [[VAR_12_:%.+]] = arith.mulf [[VAR_11_]], [[CST_2_dot_000000_]] : f32
// CHECK:             [[VAR_13_:%.+]] = arith.subf [[VAR_5_]], [[VAR_12_]] : f32
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.cmpf oeq, [[VAR_13_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_15_:%.+]] = arith.addf [[VAR_5_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_16_:%.+]] = arith.select [[VAR_14_]], [[VAR_15_]], [[VAR_5_]] : f32
// CHECK-DAG:         [[VAR_17_:%.+]] = arith.cmpf oeq, [[VAR_6_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_18_:%.+]] = arith.select [[VAR_17_]], [[VAR_16_]], [[VAR_9_]] : f32
// CHECK:             [[VAR_19_:%.+]] = arith.maxnumf [[VAR_18_]], [[CST_0_dot_000000_]] : f32
// CHECK:             [[VAR_20_:%.+]] = arith.minnumf [[VAR_19_]], [[CST_2_dot_550000_]] : f32
// CHECK:             [[VAR_21_:%.+]] = arith.fptoui [[VAR_20_]] : f32 to i32
// CHECK:             [[VAR_22_:%.+]] = arith.trunci [[VAR_21_]] : i32 to i8
// CHECK:             [[VAR_23_:%.+]] = builtin.unrealized_conversion_cast [[VAR_22_]] : i8 to ui8
// CHECK:             krnl.store [[VAR_23_]], [[RES_]]{{.}}[[VAR_2_]]{{.}} : memref<6xui8>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<6xui8>
// CHECK:         }
}

