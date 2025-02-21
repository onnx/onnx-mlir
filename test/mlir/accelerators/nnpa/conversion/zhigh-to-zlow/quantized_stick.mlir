// RUN: onnx-mlir-opt --march=arch15 --maccel=NNPA --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

func.func @test_zhigh_quantized_stick_dlfloat16(%arg0: tensor<1x3x5xf32>) -> tensor<*xf16> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0:3 = "zhigh.QuantizedStick"(%arg0, %none, %none) {layout = "3DS", quantized_type = "dlfloat16", sym_mode = 0 : i64} : (tensor<1x3x5xf32>, none, none) -> (tensor<*xf16>, tensor<f32>, tensor<f32>)
  return %0#0: tensor<*xf16>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-LABEL:  func.func @test_zhigh_quantized_stick_dlfloat16
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x3x5xf32>) -> memref<1x3x5xf16, #map> {
// CHECK-DAG:       [[CST_2_dot_550000_:%.+]] = arith.constant 2.550000e+02 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0x7F800000 : f32
// CHECK-DAG:       [[CST_1_dot_270000_:%.+]] = arith.constant 1.270000e+02 : f32
// CHECK-DAG:       [[CST_minus_1_dot_280000_:%.+]] = arith.constant -1.280000e+02 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.memset [[RES_]], [[CST_0_1_]] : memref<f32>
// CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[VAR_14_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_14_]]#0, [[VAR_14_]]#1, [[VAR_14_]]#2] : memref<1x3x5xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<f32>
// CHECK:             [[VAR_17_:%.+]] = arith.minnumf [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_17_]], [[RES_]][] : memref<f32>
// CHECK:           }
// CHECK:           [[RES_1_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.memset [[RES_1_]], [[CST_0_]] : memref<f32>
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to 1, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 3, [[LOOP_1_]]#2 -> [[I_5_:%.+]] = 0 to 5){
// CHECK:             [[VAR_14_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_14_1_]]#0, [[VAR_14_1_]]#1, [[VAR_14_1_]]#2] : memref<1x3x5xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:             [[VAR_17_1_:%.+]] = arith.maxnumf [[LOAD_RES_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:             krnl.store [[VAR_17_1_]], [[RES_1_]][] : memref<f32>
// CHECK:           }
// CHECK-DAG:       [[LOAD_RES_MEM_2_:%.+]] = krnl.load [[RES_]][] : memref<f32>
// CHECK-DAG:       [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.maxnumf [[LOAD_RES_1_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.minnumf [[LOAD_RES_MEM_2_]], [[CST_0_dot_000000_]] : f32
// CHECK:           [[VAR_6_:%.+]] = arith.subf [[VAR_4_]], [[VAR_5_]] : f32
// CHECK:           [[VAR_7_:%.+]] = arith.divf [[VAR_6_]], [[CST_2_dot_550000_]] : f32
// CHECK:           [[VAR_8_:%.+]] = arith.divf [[VAR_5_]], [[VAR_7_]] : f32
// CHECK:           [[VAR_9_:%.+]] = arith.subf [[CST_minus_1_dot_280000_]], [[VAR_8_]] : f32
// CHECK:           [[VAR_10_:%.+]] = arith.maxnumf [[VAR_9_]], [[CST_minus_1_dot_280000_]] : f32
// CHECK:           [[VAR_11_:%.+]] = arith.minnumf [[VAR_10_]], [[CST_1_dot_270000_]] : f32
// CHECK-DAG:       [[VAR_12_:%.+]] = "krnl.round_even"([[VAR_11_]]) : (f32) -> f32
// CHECK-DAG:       [[VAR_13_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_7_]] : f32
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.store [[VAR_13_]], [[RES_2_]][] : memref<f32>
// CHECK:           [[RES_3_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.store [[VAR_12_]], [[RES_3_]][] : memref<f32>
// CHECK:           [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<1x3x5xf16, #map>
// CHECK:           "zlow.stick"([[PARAM_0_]], [[RES_4_]]) {layout = "3DS", saturation = -1 : si64} : (memref<1x3x5xf32>, memref<1x3x5xf16, #map>) -> ()
// CHECK:           return [[RES_4_]] : memref<1x3x5xf16, #map>
// CHECK:         }
}

// -----


func.func @test_zhigh_quantized_stick_dlfloat16_symmetric(%arg0: tensor<1x3x5xf32>) -> tensor<*xf16> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0:3 = "zhigh.QuantizedStick"(%arg0, %none, %none) {layout = "3DS", quantized_type = "dlfloat16", sym_mode = 1 : i64} : (tensor<1x3x5xf32>, none, none) -> (tensor<*xf16>, tensor<f32>, tensor<f32>)
  return %0#0: tensor<*xf16>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-LABEL:  func.func @test_zhigh_quantized_stick_dlfloat16_symmetric
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x3x5xf32>) -> memref<1x3x5xf16, #map> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [], value = dense<1.270000e+02> : tensor<f32>} : () -> memref<f32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x3x5xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[VAR_7_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_7_]]#0, [[VAR_7_]]#1, [[VAR_7_]]#2] : memref<1x3x5xf32>
// CHECK:             [[VAR_9_:%.+]] = math.absf [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_9_]], [[RES_]]{{.}}[[VAR_7_]]#0, [[VAR_7_]]#1, [[VAR_7_]]#2] : memref<1x3x5xf32>
// CHECK:           }
// CHECK:           [[RES_1_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.memset [[RES_1_]], [[CST_0_]] : memref<f32>
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to 1, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 3, [[LOOP_1_]]#2 -> [[I_5_:%.+]] = 0 to 5){
// CHECK:             [[VAR_7_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_7_1_]]#0, [[VAR_7_1_]]#1, [[VAR_7_1_]]#2] : memref<1x3x5xf32>
// CHECK-DAG:         [[VAR_9_1_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:             [[VAR_10_:%.+]] = arith.maxnumf [[VAR_9_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:             krnl.store [[VAR_10_]], [[RES_1_]][] : memref<f32>
// CHECK:           }
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() : memref<f32>
// CHECK-DAG:       [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]][] : memref<f32>
// CHECK-DAG:       [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:           [[VAR_5_:%.+]] = arith.divf [[LOAD_VAR_0_MEM_]], [[LOAD_RES_1_MEM_]] : f32
// CHECK:           krnl.store [[VAR_5_]], [[RES_2_]][] : memref<f32>
// CHECK-DAG:       [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]][] : memref<f32>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.store [[LOAD_RES_2_MEM_]], [[RES_3_]][] : memref<f32>
// CHECK:           [[RES_4_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.store [[CST_0_dot_000000_]], [[RES_4_]][] : memref<f32>
// CHECK:           [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<1x3x5xf16, #map>
// CHECK:           "zlow.stick"([[PARAM_0_]], [[RES_5_]]) {layout = "3DS", saturation = -1 : si64} : (memref<1x3x5xf32>, memref<1x3x5xf16, #map>) -> ()
// CHECK:           return [[RES_5_]] : memref<1x3x5xf16, #map>
// CHECK:         }
}

// -----

func.func @test_zhigh_quantized_stick_int8(%arg0: tensor<1x3x5xf32>) -> tensor<*xi8> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0:3 = "zhigh.QuantizedStick"(%arg0, %none, %none) {layout = "3DS", quantized_type = "int8"} : (tensor<1x3x5xf32>, none, none) -> (tensor<*xi8>, tensor<f32>, tensor<f32>)
  return %0#0: tensor<*xi8>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 128, 0, d1 floordiv 32, d1 mod 32, d2 mod 128)>
// CHECK-LABEL:  func.func @test_zhigh_quantized_stick_int8
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x3x5xf32>) -> memref<1x3x5xi8, #map> {
// CHECK-DAG:       [[CST_2_dot_550000_:%.+]] = arith.constant 2.550000e+02 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0x7F800000 : f32
// CHECK-DAG:       [[CST_1_dot_270000_:%.+]] = arith.constant 1.270000e+02 : f32
// CHECK-DAG:       [[CST_minus_1_dot_280000_:%.+]] = arith.constant -1.280000e+02 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.memset [[RES_]], [[CST_0_1_]] : memref<f32>
// CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[VAR_14_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_14_]]#0, [[VAR_14_]]#1, [[VAR_14_]]#2] : memref<1x3x5xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<f32>
// CHECK:             [[VAR_17_:%.+]] = arith.minnumf [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_17_]], [[RES_]][] : memref<f32>
// CHECK:           }
// CHECK:           [[RES_1_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.memset [[RES_1_]], [[CST_0_]] : memref<f32>
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to 1, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 3, [[LOOP_1_]]#2 -> [[I_5_:%.+]] = 0 to 5){
// CHECK:             [[VAR_14_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_14_1_]]#0, [[VAR_14_1_]]#1, [[VAR_14_1_]]#2] : memref<1x3x5xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:             [[VAR_17_1_:%.+]] = arith.maxnumf [[LOAD_RES_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:             krnl.store [[VAR_17_1_]], [[RES_1_]][] : memref<f32>
// CHECK:           }
// CHECK-DAG:       [[LOAD_RES_MEM_2_:%.+]] = krnl.load [[RES_]][] : memref<f32>
// CHECK-DAG:       [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.maxnumf [[LOAD_RES_1_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.minnumf [[LOAD_RES_MEM_2_]], [[CST_0_dot_000000_]] : f32
// CHECK:           [[VAR_6_:%.+]] = arith.subf [[VAR_4_]], [[VAR_5_]] : f32
// CHECK:           [[VAR_7_:%.+]] = arith.divf [[VAR_6_]], [[CST_2_dot_550000_]] : f32
// CHECK:           [[VAR_8_:%.+]] = arith.divf [[VAR_5_]], [[VAR_7_]] : f32
// CHECK:           [[VAR_9_:%.+]] = arith.subf [[CST_minus_1_dot_280000_]], [[VAR_8_]] : f32
// CHECK:           [[VAR_10_:%.+]] = arith.maxnumf [[VAR_9_]], [[CST_minus_1_dot_280000_]] : f32
// CHECK:           [[VAR_11_:%.+]] = arith.minnumf [[VAR_10_]], [[CST_1_dot_270000_]] : f32
// CHECK-DAG:       [[VAR_12_:%.+]] = "krnl.round_even"([[VAR_11_]]) : (f32) -> f32
// CHECK-DAG:       [[VAR_13_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_7_]] : f32
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.store [[VAR_13_]], [[RES_2_]][] : memref<f32>
// CHECK:           [[RES_3_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.store [[VAR_12_]], [[RES_3_]][] : memref<f32>
// CHECK:           [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<1x3x5xi8, #map>
// CHECK:           "zlow.quantizedStick"([[PARAM_0_]], [[RES_2_]], [[RES_3_]], [[RES_4_]]) {layout = "3DS", q_type = "int8"} : (memref<1x3x5xf32>, memref<f32>, memref<f32>, memref<1x3x5xi8, #map>) -> ()
// CHECK:           return [[RES_4_]] : memref<1x3x5xi8, #map>
// CHECK:         }
}

// -----


func.func @test_zhigh_quantized_stick_weights(%arg0: tensor<1x3x5xf32>) -> tensor<*xi8> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0:3 = "zhigh.QuantizedStick"(%arg0, %none, %none) {layout = "3DS", quantized_type = "weights"} : (tensor<1x3x5xf32>, none, none) -> (tensor<*xi8>, tensor<f32>, tensor<f32>)
  return %0#0: tensor<*xi8>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 64, d1 mod 64, d2 mod 64)>
// CHECK-LABEL:  func.func @test_zhigh_quantized_stick_weights
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x3x5xf32>) -> memref<1x3x5xi8, #map> {
// CHECK-DAG:       [[CST_2_dot_550000_:%.+]] = arith.constant 2.550000e+02 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0x7F800000 : f32
// CHECK-DAG:       [[CST_1_dot_270000_:%.+]] = arith.constant 1.270000e+02 : f32
// CHECK-DAG:       [[CST_minus_1_dot_280000_:%.+]] = arith.constant -1.280000e+02 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.memset [[RES_]], [[CST_0_1_]] : memref<f32>
// CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[VAR_14_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_14_]]#0, [[VAR_14_]]#1, [[VAR_14_]]#2] : memref<1x3x5xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<f32>
// CHECK:             [[VAR_17_:%.+]] = arith.minnumf [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_17_]], [[RES_]][] : memref<f32>
// CHECK:           }
// CHECK:           [[RES_1_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.memset [[RES_1_]], [[CST_0_]] : memref<f32>
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to 1, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 3, [[LOOP_1_]]#2 -> [[I_5_:%.+]] = 0 to 5){
// CHECK:             [[VAR_14_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_14_1_]]#0, [[VAR_14_1_]]#1, [[VAR_14_1_]]#2] : memref<1x3x5xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:             [[VAR_17_1_:%.+]] = arith.maxnumf [[LOAD_RES_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:             krnl.store [[VAR_17_1_]], [[RES_1_]][] : memref<f32>
// CHECK:           }
// CHECK-DAG:       [[LOAD_RES_MEM_2_:%.+]] = krnl.load [[RES_]][] : memref<f32>
// CHECK-DAG:       [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.maxnumf [[LOAD_RES_1_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.minnumf [[LOAD_RES_MEM_2_]], [[CST_0_dot_000000_]] : f32
// CHECK:           [[VAR_6_:%.+]] = arith.subf [[VAR_4_]], [[VAR_5_]] : f32
// CHECK:           [[VAR_7_:%.+]] = arith.divf [[VAR_6_]], [[CST_2_dot_550000_]] : f32
// CHECK:           [[VAR_8_:%.+]] = arith.divf [[VAR_5_]], [[VAR_7_]] : f32
// CHECK:           [[VAR_9_:%.+]] = arith.subf [[CST_minus_1_dot_280000_]], [[VAR_8_]] : f32
// CHECK:           [[VAR_10_:%.+]] = arith.maxnumf [[VAR_9_]], [[CST_minus_1_dot_280000_]] : f32
// CHECK:           [[VAR_11_:%.+]] = arith.minnumf [[VAR_10_]], [[CST_1_dot_270000_]] : f32
// CHECK-DAG:       [[VAR_12_:%.+]] = "krnl.round_even"([[VAR_11_]]) : (f32) -> f32
// CHECK-DAG:       [[VAR_13_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_7_]] : f32
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.store [[VAR_13_]], [[RES_2_]][] : memref<f32>
// CHECK:           [[RES_3_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.store [[VAR_12_]], [[RES_3_]][] : memref<f32>
// CHECK:           [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<1x3x5xi8, #map>
// CHECK:           "zlow.quantizedStick"([[PARAM_0_]], [[RES_2_]], [[RES_3_]], [[RES_4_]]) {layout = "3DS", q_type = "weights"} : (memref<1x3x5xf32>, memref<f32>, memref<f32>, memref<1x3x5xi8, #map>) -> ()
// CHECK:           return [[RES_4_]] : memref<1x3x5xi8, #map>
// CHECK:         }
}
