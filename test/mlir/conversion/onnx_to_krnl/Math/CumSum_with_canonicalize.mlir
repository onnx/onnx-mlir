// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// -----

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

func.func @test_cumsum_constant_axis(%arg0: tensor<2x3xf64>) -> tensor<*xf64> {
  %axis = onnx.Constant dense<1> : tensor<i32>
  %0 = "onnx.CumSum"(%arg0, %axis) : (tensor<2x3xf64>, tensor<i32>) -> tensor<*xf64>
  return %0 : tensor<*xf64>

// mlir2FileCheck.py -a'["input"]'
// CHECK-LABEL:  func @test_cumsum_constant_axis
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<2x3xf64>) -> memref<2x3xf64> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[AXIS_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_0_:%.+]] = 0 to 3){
// CHECK:             [[VAR_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1] : memref<2x3xf64>
// CHECK:             krnl.store [[LOAD_INPUT_MEM_]], [[RES_1_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1] : memref<2x3xf64>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[VAR_4_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_INPUT_MEM_1_:%.+]] = arith.index_cast [[VAR_4_1_]] : index to i64
// CHECK:             [[VAR_6_:%.+]] = arith.sitofp [[LOAD_INPUT_MEM_1_]] : i64 to f32
// CHECK:             [[VAR_7_:%.+]] = math.exp2 [[VAR_6_]] : f32
// CHECK:             [[VAR_8_:%.+]] = arith.fptosi [[VAR_7_]] : f32 to i64
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.index_cast [[VAR_8_]] : i64 to index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_2_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[I_3_:%.+]] = 0 to 3){
// CHECK:               [[VAR_12_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_12_]]#0, [[VAR_12_]]#1] : memref<2x3xf64>
// CHECK-DAG:           [[VAR_14_:%.+]] = arith.subi [[VAR_12_]]#1, [[VAR_9_]] : index
// CHECK:               [[VAR_15_:%.+]] = arith.cmpi sge, [[VAR_14_]], [[CST_0_]] : index
// CHECK:               [[VAR_16_:%.+]] = arith.select [[VAR_15_]], [[VAR_14_]], [[VAR_12_]]#1 : index
// CHECK:               [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_12_]]#0, [[VAR_16_]]] : memref<2x3xf64>
// CHECK:               [[VAR_18_:%.+]] = arith.select [[VAR_15_]], [[LOAD_RES_1_MEM_1_]], [[CST_0_dot_000000_]] : f64
// CHECK:               [[VAR_19_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[VAR_18_]] : f64
// CHECK:               krnl.store [[VAR_19_]], [[RES_]]{{.}}[[VAR_12_]]#0, [[VAR_12_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:             [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_4_:%.+]] = 0 to 2, [[LOOP_3_]]#1 -> [[I_5_:%.+]] = 0 to 3){
// CHECK:               [[VAR_12_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_12_1_]]#0, [[VAR_12_1_]]#1] : memref<2x3xf64>
// CHECK:               krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_1_]]{{.}}[[VAR_12_1_]]#0, [[VAR_12_1_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3xf64>
// CHECK:         }
}

// -----

func.func @test_cumsum_constant_axis_reverse_mode(%arg0: tensor<2x3xf64>) -> tensor<*xf64> {
  %axis = onnx.Constant dense<1> : tensor<i32>
  %0 = "onnx.CumSum"(%arg0, %axis) {reverse = 1 : si64} : (tensor<2x3xf64>, tensor<i32>) -> tensor<*xf64>
  return %0 : tensor<*xf64>

// mlir2FileCheck.py -a'["input"]'
// CHECK-LABEL:  func @test_cumsum_constant_axis_reverse_mode
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<2x3xf64>) -> memref<2x3xf64> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[AXIS_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_0_:%.+]] = 0 to 3){
// CHECK:             [[VAR_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1] : memref<2x3xf64>
// CHECK:             krnl.store [[LOAD_INPUT_MEM_]], [[RES_1_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1] : memref<2x3xf64>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[VAR_4_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_INPUT_MEM_1_:%.+]] = arith.index_cast [[VAR_4_1_]] : index to i64
// CHECK:             [[VAR_6_:%.+]] = arith.sitofp [[LOAD_INPUT_MEM_1_]] : i64 to f32
// CHECK:             [[VAR_7_:%.+]] = math.exp2 [[VAR_6_]] : f32
// CHECK:             [[VAR_8_:%.+]] = arith.fptosi [[VAR_7_]] : f32 to i64
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.index_cast [[VAR_8_]] : i64 to index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_2_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[I_3_:%.+]] = 0 to 3){
// CHECK:               [[VAR_12_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_12_]]#0, [[VAR_12_]]#1] : memref<2x3xf64>
// CHECK-DAG:           [[VAR_14_:%.+]] = arith.addi [[VAR_12_]]#1, [[VAR_9_]] : index
// CHECK:               [[VAR_15_:%.+]] = arith.cmpi slt, [[VAR_14_]], [[CST_3_]] : index
// CHECK:               [[VAR_16_:%.+]] = arith.select [[VAR_15_]], [[VAR_14_]], [[VAR_12_]]#1 : index
// CHECK:               [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_12_]]#0, [[VAR_16_]]] : memref<2x3xf64>
// CHECK:               [[VAR_18_:%.+]] = arith.select [[VAR_15_]], [[LOAD_RES_1_MEM_1_]], [[CST_0_dot_000000_]] : f64
// CHECK:               [[VAR_19_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[VAR_18_]] : f64
// CHECK:               krnl.store [[VAR_19_]], [[RES_]]{{.}}[[VAR_12_]]#0, [[VAR_12_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:             [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_4_:%.+]] = 0 to 2, [[LOOP_3_]]#1 -> [[I_5_:%.+]] = 0 to 3){
// CHECK:               [[VAR_12_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_12_1_]]#0, [[VAR_12_1_]]#1] : memref<2x3xf64>
// CHECK:               krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_1_]]{{.}}[[VAR_12_]]#0, [[VAR_12_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3xf64>
// CHECK:         }
}

// -----

func.func @test_cumsum_constant_axis_exclusive_mode(%arg0: tensor<2x3xf64>) -> tensor<*xf64> {
  %axis = onnx.Constant dense<1> : tensor<i32>
  %0 = "onnx.CumSum"(%arg0, %axis) {exclusive = 1 : si64} : (tensor<2x3xf64>, tensor<i32>) -> tensor<*xf64>
  return %0 : tensor<*xf64>

// mlir2FileCheck.py -a'["input"]'
// CHECK-LABEL:  func @test_cumsum_constant_axis_exclusive_mode
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<2x3xf64>) -> memref<2x3xf64> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[AXIS_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_0_:%.+]] = 0 to 3){
// CHECK:             [[VAR_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[VAR_5_:%.+]] = arith.subi [[VAR_4_]]#1, [[CST_1_]] : index
// CHECK:             [[VAR_6_:%.+]] = arith.cmpi sge, [[VAR_5_]], [[CST_0_]] : index
// CHECK:             [[VAR_7_:%.+]] = arith.select [[VAR_6_]], [[VAR_5_]], [[VAR_4_]]#1 : index
// CHECK:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_4_]]#0, [[VAR_7_]]{{.}} : memref<2x3xf64>
// CHECK:             [[VAR_9_:%.+]] = arith.select [[VAR_6_]], [[LOAD_INPUT_MEM_]], [[CST_0_dot_000000_]] : f64
// CHECK:             krnl.store [[VAR_9_]], [[RES_1_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1] : memref<2x3xf64>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[VAR_4_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[VAR_5_1_:%.+]] = arith.index_cast [[VAR_4_1_]] : index to i64
// CHECK:             [[VAR_6_1_:%.+]] = arith.sitofp [[VAR_5_1_]] : i64 to f32
// CHECK:             [[VAR_7_1_:%.+]] = math.exp2 [[VAR_6_1_]] : f32
// CHECK:             [[LOAD_INPUT_MEM_1_:%.+]] = arith.fptosi [[VAR_7_1_]] : f32 to i64
// CHECK-DAG:         [[VAR_9_1_:%.+]] = arith.index_cast [[LOAD_INPUT_MEM_1_]] : i64 to index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_2_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[I_3_:%.+]] = 0 to 3){
// CHECK:               [[VAR_12_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_12_]]#0, [[VAR_12_]]#1] : memref<2x3xf64>
// CHECK-DAG:           [[VAR_14_:%.+]] = arith.subi [[VAR_12_]]#1, [[VAR_9_1_]] : index
// CHECK:               [[VAR_15_:%.+]] = arith.cmpi sge, [[VAR_14_]], [[CST_0_]] : index
// CHECK:               [[VAR_16_:%.+]] = arith.select [[VAR_15_]], [[VAR_14_]], [[VAR_12_]]#1 : index
// CHECK:               [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_12_]]#0, [[VAR_16_]]] : memref<2x3xf64>
// CHECK:               [[VAR_18_:%.+]] = arith.select [[VAR_15_]], [[LOAD_RES_1_MEM_1_]], [[CST_0_dot_000000_]] : f64
// CHECK:               [[VAR_19_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[VAR_18_]] : f64
// CHECK:               krnl.store [[VAR_19_]], [[RES_]]{{.}}[[VAR_12_]]#0, [[VAR_12_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:             [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_4_:%.+]] = 0 to 2, [[LOOP_3_]]#1 -> [[I_5_:%.+]] = 0 to 3){
// CHECK:               [[VAR_12_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_12_1_]]#0, [[VAR_12_1_]]#1] : memref<2x3xf64>
// CHECK:               krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_1_]]{{.}}[[VAR_12_]]#0, [[VAR_12_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3xf64>
// CHECK:         }
}

// -----

func.func @test_cumsum_constant_axis_exclusive_reverse_mode(%arg0: tensor<2x3xf64>) -> tensor<*xf64> {
  %axis = onnx.Constant dense<1> : tensor<i32>
  %0 = "onnx.CumSum"(%arg0, %axis) {exclusive = 1 : si64, reverse = 1 : si64} : (tensor<2x3xf64>, tensor<i32>) -> tensor<*xf64>
  return %0 : tensor<*xf64>

// mlir2FileCheck.py -a'["input"]'
// CHECK-LABEL:  func @test_cumsum_constant_axis_exclusive_reverse_mode
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<2x3xf64>) -> memref<2x3xf64> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[AXIS_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_0_:%.+]] = 0 to 3){
// CHECK:             [[VAR_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[VAR_5_:%.+]] = arith.addi [[VAR_4_]]#1, [[CST_1_]] : index
// CHECK:             [[VAR_6_:%.+]] = arith.cmpi slt, [[VAR_5_]], [[CST_3_]] : index
// CHECK:             [[VAR_7_:%.+]] = arith.select [[VAR_6_]], [[VAR_5_]], [[VAR_4_]]#1 : index
// CHECK:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_4_]]#0, [[VAR_7_]]{{.}} : memref<2x3xf64>
// CHECK:             [[VAR_9_:%.+]] = arith.select [[VAR_6_]], [[LOAD_INPUT_MEM_]], [[CST_0_dot_000000_]] : f64
// CHECK:             krnl.store [[VAR_9_]], [[RES_1_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1] : memref<2x3xf64>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[VAR_4_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[VAR_5_1_:%.+]] = arith.index_cast [[VAR_4_1_]] : index to i64
// CHECK:             [[VAR_6_1_:%.+]] = arith.sitofp [[VAR_5_1_]] : i64 to f32
// CHECK:             [[VAR_7_1_:%.+]] = math.exp2 [[VAR_6_1_]] : f32
// CHECK:             [[LOAD_INPUT_MEM_1_:%.+]] = arith.fptosi [[VAR_7_1_]] : f32 to i64
// CHECK-DAG:         [[VAR_9_1_:%.+]] = arith.index_cast [[LOAD_INPUT_MEM_1_]] : i64 to index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_2_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[I_3_:%.+]] = 0 to 3){
// CHECK:               [[VAR_12_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_12_]]#0, [[VAR_12_]]#1] : memref<2x3xf64>
// CHECK-DAG:           [[VAR_14_:%.+]] = arith.addi [[VAR_12_]]#1, [[VAR_9_1_]] : index
// CHECK:               [[VAR_15_:%.+]] = arith.cmpi slt, [[VAR_14_]], [[CST_3_]] : index
// CHECK:               [[VAR_16_:%.+]] = arith.select [[VAR_15_]], [[VAR_14_]], [[VAR_12_]]#1 : index
// CHECK:               [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_12_]]#0, [[VAR_16_]]] : memref<2x3xf64>
// CHECK:               [[VAR_18_:%.+]] = arith.select [[VAR_15_]], [[LOAD_RES_1_MEM_1_]], [[CST_0_dot_000000_]] : f64
// CHECK:               [[VAR_19_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[VAR_18_]] : f64
// CHECK:               krnl.store [[VAR_19_]], [[RES_]]{{.}}[[VAR_12_]]#0, [[VAR_12_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:             [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_4_:%.+]] = 0 to 2, [[LOOP_3_]]#1 -> [[I_5_:%.+]] = 0 to 3){
// CHECK:               [[VAR_12_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_12_1_]]#0, [[VAR_12_1_]]#1] : memref<2x3xf64>
// CHECK:               krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_1_]]{{.}}[[VAR_12_]]#0, [[VAR_12_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3xf64>
// CHECK:         }
}

// -----

func.func @test_cumsum_dynamic_axis(%arg0: tensor<2x3xf64>, %arg1:tensor<i32>) -> tensor<*xf64> {
  %0 = "onnx.CumSum"(%arg0, %arg1) : (tensor<2x3xf64>, tensor<i32>) -> tensor<*xf64>
  return %0 : tensor<*xf64>

// mlir2FileCheck.py -a'["input", "axis"]'
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 + 2)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s1 + 1)>
// CHECK-LABEL:  func @test_cumsum_dynamic_axis
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<2x3xf64>, [[AXIS_:%.+]]: memref<i32>) -> memref<2x3xf64> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[LOAD_AXIS_MEM_:%.+]] = krnl.load [[AXIS_]][] : memref<i32>
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[LOAD_AXIS_MEM_]] : i32 to index
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.cmpi slt, [[VAR_1_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_1_]]{{.}}
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.select [[VAR_2_]], [[VAR_3_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK:           [[VAR_7_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[CST_2_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_9_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_1_]] : index
// CHECK:           [[VAR_10_:%.+]] = arith.select [[VAR_9_]], [[CST_3_]], [[VAR_8_]] : index
// CHECK:           [[VAR_11_:%.+]] = arith.index_cast [[VAR_10_]] : index to i64
// CHECK:           [[VAR_12_:%.+]] = arith.sitofp [[VAR_11_]] : i64 to f32
// CHECK:           [[VAR_13_:%.+]] = math.log2 [[VAR_12_]] : f32
// CHECK:           [[VAR_14_:%.+]] = arith.fptosi [[VAR_13_]] : f32 to i64
// CHECK-DAG:       [[VAR_15_:%.+]] = arith.index_cast [[VAR_14_]] : i64 to index
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3){
// CHECK:             [[VAR_18_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_18_]]#0, [[VAR_18_]]#1] : memref<2x3xf64>
// CHECK:             krnl.store [[LOAD_INPUT_MEM_]], [[RES_1_]]{{.}}[[VAR_18_]]#0, [[VAR_18_]]#1] : memref<2x3xf64>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to [[MAP_1_]](){{.}}[[VAR_1_]], [[VAR_15_]]]){
// CHECK:             [[VAR_18_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_INPUT_MEM_1_:%.+]] = arith.index_cast [[VAR_18_1_]] : index to i64
// CHECK:             [[VAR_20_:%.+]] = arith.sitofp [[LOAD_INPUT_MEM_1_]] : i64 to f32
// CHECK:             [[VAR_21_:%.+]] = math.exp2 [[VAR_20_]] : f32
// CHECK:             [[VAR_22_:%.+]] = arith.fptosi [[VAR_21_]] : f32 to i64
// CHECK-DAG:         [[VAR_23_:%.+]] = arith.index_cast [[VAR_22_]] : i64 to index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 3){
// CHECK:               [[VAR_26_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_26_]]#0, [[VAR_26_]]#1] : memref<2x3xf64>
// CHECK-DAG:           [[VAR_28_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:           [[VAR_29_:%.+]] = arith.subi [[VAR_26_]]#0, [[VAR_23_]] : index
// CHECK:               [[VAR_30_:%.+]] = arith.cmpi sge, [[VAR_29_]], [[CST_0_]] : index
// CHECK:               [[VAR_31_:%.+]] = arith.andi [[VAR_28_]], [[VAR_30_]] : i1
// CHECK-DAG:           [[VAR_32_:%.+]] = arith.select [[VAR_31_]], [[VAR_29_]], [[VAR_26_]]#0 : index
// CHECK-DAG:           [[VAR_33_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_1_]] : index
// CHECK-DAG:           [[VAR_34_:%.+]] = arith.subi [[VAR_26_]]#1, [[VAR_23_]] : index
// CHECK:               [[VAR_35_:%.+]] = arith.cmpi sge, [[VAR_34_]], [[CST_0_]] : index
// CHECK:               [[VAR_36_:%.+]] = arith.andi [[VAR_33_]], [[VAR_35_]] : i1
// CHECK-DAG:           [[VAR_37_:%.+]] = arith.ori [[VAR_36_]], [[VAR_31_]] : i1
// CHECK-DAG:           [[VAR_38_:%.+]] = arith.select [[VAR_36_]], [[VAR_34_]], [[VAR_26_]]#1 : index
// CHECK:               [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_32_]], [[VAR_38_]]{{.}} : memref<2x3xf64>
// CHECK:               [[VAR_40_:%.+]] = arith.select [[VAR_37_]], [[LOAD_RES_1_MEM_1_]], [[CST_0_dot_000000_]] : f64
// CHECK:               [[VAR_41_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[VAR_40_]] : f64
// CHECK:               krnl.store [[VAR_41_]], [[RES_]]{{.}}[[VAR_26_]]#0, [[VAR_26_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:             [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to 2, [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to 3){
// CHECK:               [[VAR_26_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x3xf64>
// CHECK:               krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_1_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3xf64>
// CHECK:         }
}

// -----

func.func @test_cumsum_dynamic_axis_reverse_mode(%arg0: tensor<2x3xf64>, %arg1:tensor<i32>) -> tensor<*xf64> {
  %0 = "onnx.CumSum"(%arg0, %arg1) {reverse = 1 : si64} : (tensor<2x3xf64>, tensor<i32>) -> tensor<*xf64>
  return %0 : tensor<*xf64>

// mlir2FileCheck.py -a'["input", "axis"]'
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 + 2)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s1 + 1)>
// CHECK-LABEL:  func @test_cumsum_dynamic_axis_reverse_mode
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<2x3xf64>, [[AXIS_:%.+]]: memref<i32>) -> memref<2x3xf64> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[LOAD_AXIS_MEM_:%.+]] = krnl.load [[AXIS_]][] : memref<i32>
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[LOAD_AXIS_MEM_]] : i32 to index
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.cmpi slt, [[VAR_1_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_1_]]{{.}}
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.select [[VAR_2_]], [[VAR_3_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK:           [[VAR_7_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[CST_2_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_9_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_1_]] : index
// CHECK:           [[VAR_10_:%.+]] = arith.select [[VAR_9_]], [[CST_3_]], [[VAR_8_]] : index
// CHECK:           [[VAR_11_:%.+]] = arith.index_cast [[VAR_10_]] : index to i64
// CHECK:           [[VAR_12_:%.+]] = arith.sitofp [[VAR_11_]] : i64 to f32
// CHECK:           [[VAR_13_:%.+]] = math.log2 [[VAR_12_]] : f32
// CHECK:           [[VAR_14_:%.+]] = arith.fptosi [[VAR_13_]] : f32 to i64
// CHECK-DAG:       [[VAR_15_:%.+]] = arith.index_cast [[VAR_14_]] : i64 to index
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3){
// CHECK:             [[VAR_18_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_18_]]#0, [[VAR_18_]]#1] : memref<2x3xf64>
// CHECK:             krnl.store [[LOAD_INPUT_MEM_]], [[RES_1_]]{{.}}[[VAR_18_]]#0, [[VAR_18_]]#1] : memref<2x3xf64>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to [[MAP_1_]](){{.}}[[VAR_1_]], [[VAR_15_]]]){
// CHECK:             [[VAR_18_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_INPUT_MEM_1_:%.+]] = arith.index_cast [[VAR_18_1_]] : index to i64
// CHECK:             [[VAR_20_:%.+]] = arith.sitofp [[LOAD_INPUT_MEM_1_]] : i64 to f32
// CHECK:             [[VAR_21_:%.+]] = math.exp2 [[VAR_20_]] : f32
// CHECK:             [[VAR_22_:%.+]] = arith.fptosi [[VAR_21_]] : f32 to i64
// CHECK-DAG:         [[VAR_23_:%.+]] = arith.index_cast [[VAR_22_]] : i64 to index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 3){
// CHECK:               [[VAR_26_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_26_]]#0, [[VAR_26_]]#1] : memref<2x3xf64>
// CHECK-DAG:           [[VAR_28_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:           [[VAR_29_:%.+]] = arith.addi [[VAR_26_]]#0, [[VAR_23_]] : index
// CHECK:               [[VAR_30_:%.+]] = arith.cmpi slt, [[VAR_29_]], [[CST_2_]] : index
// CHECK:               [[VAR_31_:%.+]] = arith.andi [[VAR_28_]], [[VAR_30_]] : i1
// CHECK-DAG:           [[VAR_32_:%.+]] = arith.select [[VAR_31_]], [[VAR_29_]], [[VAR_26_]]#0 : index
// CHECK-DAG:           [[VAR_33_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_1_]] : index
// CHECK-DAG:           [[VAR_34_:%.+]] = arith.addi [[VAR_26_]]#1, [[VAR_23_]] : index
// CHECK:               [[VAR_35_:%.+]] = arith.cmpi slt, [[VAR_34_]], [[CST_3_]] : index
// CHECK:               [[VAR_36_:%.+]] = arith.andi [[VAR_33_]], [[VAR_35_]] : i1
// CHECK-DAG:           [[VAR_37_:%.+]] = arith.ori [[VAR_36_]], [[VAR_31_]] : i1
// CHECK-DAG:           [[VAR_38_:%.+]] = arith.select [[VAR_36_]], [[VAR_34_]], [[VAR_26_]]#1 : index
// CHECK:               [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_32_]], [[VAR_38_]]{{.}} : memref<2x3xf64>
// CHECK:               [[VAR_40_:%.+]] = arith.select [[VAR_37_]], [[LOAD_RES_1_MEM_1_]], [[CST_0_dot_000000_]] : f64
// CHECK:               [[VAR_41_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[VAR_40_]] : f64
// CHECK:               krnl.store [[VAR_41_]], [[RES_]]{{.}}[[VAR_26_]]#0, [[VAR_26_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:             [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to 2, [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to 3){
// CHECK:               [[VAR_26_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x3xf64>
// CHECK:               krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_1_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3xf64>
// CHECK:         }
}

// -----

func.func @test_cumsum_dynamic_axis_exclusive_mode(%arg0: tensor<2x3xf64>, %arg1:tensor<i32>) -> tensor<*xf64> {
  %0 = "onnx.CumSum"(%arg0, %arg1) {exclusive = 1 : si64} : (tensor<2x3xf64>, tensor<i32>) -> tensor<*xf64>
  return %0 : tensor<*xf64>

// mlir2FileCheck.py -a'["input", "axis"]'
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 + 2)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s1 + 1)>
// CHECK-LABEL:  func @test_cumsum_dynamic_axis_exclusive_mode
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<2x3xf64>, [[AXIS_:%.+]]: memref<i32>) -> memref<2x3xf64> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[LOAD_AXIS_MEM_:%.+]] = krnl.load [[AXIS_]][] : memref<i32>
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[LOAD_AXIS_MEM_]] : i32 to index
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.cmpi slt, [[VAR_1_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_1_]]{{.}}
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.select [[VAR_2_]], [[VAR_3_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK:           [[VAR_7_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[CST_2_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_9_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_1_]] : index
// CHECK:           [[VAR_10_:%.+]] = arith.select [[VAR_9_]], [[CST_3_]], [[VAR_8_]] : index
// CHECK:           [[VAR_11_:%.+]] = arith.index_cast [[VAR_10_]] : index to i64
// CHECK:           [[VAR_12_:%.+]] = arith.sitofp [[VAR_11_]] : i64 to f32
// CHECK:           [[VAR_13_:%.+]] = math.log2 [[VAR_12_]] : f32
// CHECK:           [[VAR_14_:%.+]] = arith.fptosi [[VAR_13_]] : f32 to i64
// CHECK-DAG:       [[VAR_15_:%.+]] = arith.index_cast [[VAR_14_]] : i64 to index
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3){
// CHECK-DAG:         [[VAR_18_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_19_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_0_]] : index
// CHECK:             [[VAR_20_:%.+]] = arith.subi [[VAR_18_]]#0, [[CST_1_]] : index
// CHECK:             [[VAR_21_:%.+]] = arith.cmpi sge, [[VAR_20_]], [[CST_0_]] : index
// CHECK:             [[VAR_22_:%.+]] = arith.andi [[VAR_19_]], [[VAR_21_]] : i1
// CHECK-DAG:         [[VAR_23_:%.+]] = arith.select [[VAR_22_]], [[VAR_20_]], [[VAR_18_]]#0 : index
// CHECK-DAG:         [[VAR_24_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_25_:%.+]] = arith.subi [[VAR_18_]]#1, [[CST_1_]] : index
// CHECK:             [[VAR_26_:%.+]] = arith.cmpi sge, [[VAR_25_]], [[CST_0_]] : index
// CHECK:             [[VAR_27_:%.+]] = arith.andi [[VAR_24_]], [[VAR_26_]] : i1
// CHECK-DAG:         [[VAR_28_:%.+]] = arith.ori [[VAR_27_]], [[VAR_22_]] : i1
// CHECK-DAG:         [[VAR_29_:%.+]] = arith.select [[VAR_27_]], [[VAR_25_]], [[VAR_18_]]#1 : index
// CHECK:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_23_]], [[VAR_29_]]{{.}} : memref<2x3xf64>
// CHECK:             [[VAR_31_:%.+]] = arith.select [[VAR_28_]], [[LOAD_INPUT_MEM_]], [[CST_0_dot_000000_]] : f64
// CHECK:             krnl.store [[VAR_31_]], [[RES_1_]]{{.}}[[VAR_18_]]#0, [[VAR_18_]]#1] : memref<2x3xf64>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to [[MAP_1_]](){{.}}[[VAR_1_]], [[VAR_15_]]]){
// CHECK:             [[VAR_18_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[VAR_19_1_:%.+]] = arith.index_cast [[VAR_18_1_]] : index to i64
// CHECK:             [[VAR_20_1_:%.+]] = arith.sitofp [[VAR_19_1_]] : i64 to f32
// CHECK:             [[VAR_21_1_:%.+]] = math.exp2 [[VAR_20_1_]] : f32
// CHECK:             [[VAR_22_1_:%.+]] = arith.fptosi [[VAR_21_1_]] : f32 to i64
// CHECK-DAG:         [[VAR_23_1_:%.+]] = arith.index_cast [[VAR_22_1_]] : i64 to index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 3){
// CHECK:               [[VAR_26_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[VAR_27_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x3xf64>
// CHECK-DAG:           [[VAR_28_1_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:           [[VAR_29_1_:%.+]] = arith.subi [[VAR_26_1_]]#0, [[VAR_23_1_]] : index
// CHECK:               [[LOAD_INPUT_MEM_1_:%.+]] = arith.cmpi sge, [[VAR_29_1_]], [[CST_0_]] : index
// CHECK:               [[VAR_31_1_:%.+]] = arith.andi [[VAR_28_1_]], [[LOAD_INPUT_MEM_1_]] : i1
// CHECK-DAG:           [[VAR_32_:%.+]] = arith.select [[VAR_31_1_]], [[VAR_29_1_]], [[VAR_26_1_]]#0 : index
// CHECK-DAG:           [[VAR_33_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_1_]] : index
// CHECK-DAG:           [[VAR_34_:%.+]] = arith.subi [[VAR_26_1_]]#1, [[VAR_23_1_]] : index
// CHECK:               [[VAR_35_:%.+]] = arith.cmpi sge, [[VAR_34_]], [[CST_0_]] : index
// CHECK:               [[VAR_36_:%.+]] = arith.andi [[VAR_33_]], [[VAR_35_]] : i1
// CHECK-DAG:           [[VAR_37_:%.+]] = arith.ori [[VAR_36_]], [[VAR_31_1_]] : i1
// CHECK-DAG:           [[VAR_38_:%.+]] = arith.select [[VAR_36_]], [[VAR_34_]], [[VAR_26_1_]]#1 : index
// CHECK:               [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_32_]], [[VAR_38_]]{{.}} : memref<2x3xf64>
// CHECK:               [[VAR_40_:%.+]] = arith.select [[VAR_37_]], [[LOAD_RES_1_MEM_]], [[CST_0_dot_000000_]] : f64
// CHECK:               [[VAR_41_:%.+]] = arith.addf [[VAR_27_1_]], [[VAR_40_]] : f64
// CHECK:               krnl.store [[VAR_41_]], [[RES_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:             [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to 2, [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to 3){
// CHECK:               [[VAR_26_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[VAR_27_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_26_2_]]#0, [[VAR_26_2_]]#1] : memref<2x3xf64>
// CHECK:               krnl.store [[VAR_27_1_]], [[RES_1_]]{{.}}[[VAR_26_2_]]#0, [[VAR_26_2_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3xf64>
// CHECK:         }
}

// -----

func.func @test_cumsum_dynamic_axis_exclusive_reverse_mode(%arg0: tensor<2x3xf64>, %arg1:tensor<i32>) -> tensor<*xf64> {
  %0 = "onnx.CumSum"(%arg0, %arg1) {exclusive = 1 : si64, reverse = 1 : si64} : (tensor<2x3xf64>, tensor<i32>) -> tensor<*xf64>
  return %0 : tensor<*xf64>

// mlir2FileCheck.py -a'["input", "axis"]'
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 + 2)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s1 + 1)>
// CHECK-LABEL:  func @test_cumsum_dynamic_axis_exclusive_reverse_mode
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<2x3xf64>, [[AXIS_:%.+]]: memref<i32>) -> memref<2x3xf64> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[LOAD_AXIS_MEM_:%.+]] = krnl.load [[AXIS_]][] : memref<i32>
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[LOAD_AXIS_MEM_]] : i32 to index
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.cmpi slt, [[VAR_1_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_1_]]{{.}}
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.select [[VAR_2_]], [[VAR_3_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK:           [[VAR_7_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[CST_2_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_9_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_1_]] : index
// CHECK:           [[VAR_10_:%.+]] = arith.select [[VAR_9_]], [[CST_3_]], [[VAR_8_]] : index
// CHECK:           [[VAR_11_:%.+]] = arith.index_cast [[VAR_10_]] : index to i64
// CHECK:           [[VAR_12_:%.+]] = arith.sitofp [[VAR_11_]] : i64 to f32
// CHECK:           [[VAR_13_:%.+]] = math.log2 [[VAR_12_]] : f32
// CHECK:           [[VAR_14_:%.+]] = arith.fptosi [[VAR_13_]] : f32 to i64
// CHECK-DAG:       [[VAR_15_:%.+]] = arith.index_cast [[VAR_14_]] : i64 to index
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3){
// CHECK-DAG:         [[VAR_18_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_19_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_0_]] : index
// CHECK:             [[VAR_20_:%.+]] = arith.addi [[VAR_18_]]#0, [[CST_1_]] : index
// CHECK:             [[VAR_21_:%.+]] = arith.cmpi slt, [[VAR_20_]], [[CST_2_]] : index
// CHECK:             [[VAR_22_:%.+]] = arith.andi [[VAR_19_]], [[VAR_21_]] : i1
// CHECK-DAG:         [[VAR_23_:%.+]] = arith.select [[VAR_22_]], [[VAR_20_]], [[VAR_18_]]#0 : index
// CHECK-DAG:         [[VAR_24_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_25_:%.+]] = arith.addi [[VAR_18_]]#1, [[CST_1_]] : index
// CHECK:             [[VAR_26_:%.+]] = arith.cmpi slt, [[VAR_25_]], [[CST_3_]] : index
// CHECK:             [[VAR_27_:%.+]] = arith.andi [[VAR_24_]], [[VAR_26_]] : i1
// CHECK-DAG:         [[VAR_28_:%.+]] = arith.ori [[VAR_27_]], [[VAR_22_]] : i1
// CHECK-DAG:         [[VAR_29_:%.+]] = arith.select [[VAR_27_]], [[VAR_25_]], [[VAR_18_]]#1 : index
// CHECK:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_23_]], [[VAR_29_]]{{.}} : memref<2x3xf64>
// CHECK:             [[VAR_31_:%.+]] = arith.select [[VAR_28_]], [[LOAD_INPUT_MEM_]], [[CST_0_dot_000000_]] : f64
// CHECK:             krnl.store [[VAR_31_]], [[RES_1_]]{{.}}[[VAR_18_]]#0, [[VAR_18_]]#1] : memref<2x3xf64>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to [[MAP_1_]](){{.}}[[VAR_1_]], [[VAR_15_]]]){
// CHECK:             [[VAR_18_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[VAR_19_1_:%.+]] = arith.index_cast [[VAR_18_1_]] : index to i64
// CHECK:             [[VAR_20_1_:%.+]] = arith.sitofp [[VAR_19_1_]] : i64 to f32
// CHECK:             [[VAR_21_1_:%.+]] = math.exp2 [[VAR_20_1_]] : f32
// CHECK:             [[VAR_22_1_:%.+]] = arith.fptosi [[VAR_21_1_]] : f32 to i64
// CHECK-DAG:         [[VAR_23_1_:%.+]] = arith.index_cast [[VAR_22_1_]] : i64 to index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 3){
// CHECK:               [[VAR_26_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[VAR_27_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x3xf64>
// CHECK-DAG:           [[VAR_28_1_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:           [[VAR_29_1_:%.+]] = arith.addi [[VAR_26_1_]]#0, [[VAR_23_1_]] : index
// CHECK:               [[LOAD_INPUT_MEM_1_:%.+]] = arith.cmpi slt, [[VAR_29_1_]], [[CST_2_]] : index
// CHECK:               [[VAR_31_1_:%.+]] = arith.andi [[VAR_28_1_]], [[LOAD_INPUT_MEM_1_]] : i1
// CHECK-DAG:           [[VAR_32_:%.+]] = arith.select [[VAR_31_1_]], [[VAR_29_1_]], [[VAR_26_1_]]#0 : index
// CHECK-DAG:           [[VAR_33_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_1_]] : index
// CHECK-DAG:           [[VAR_34_:%.+]] = arith.addi [[VAR_26_1_]]#1, [[VAR_23_1_]] : index
// CHECK:               [[VAR_35_:%.+]] = arith.cmpi slt, [[VAR_34_]], [[CST_3_]] : index
// CHECK:               [[VAR_36_:%.+]] = arith.andi [[VAR_33_]], [[VAR_35_]] : i1
// CHECK-DAG:           [[VAR_37_:%.+]] = arith.ori [[VAR_36_]], [[VAR_31_1_]] : i1
// CHECK-DAG:           [[VAR_38_:%.+]] = arith.select [[VAR_36_]], [[VAR_34_]], [[VAR_26_1_]]#1 : index
// CHECK:               [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_32_]], [[VAR_38_]]{{.}} : memref<2x3xf64>
// CHECK:               [[VAR_40_:%.+]] = arith.select [[VAR_37_]], [[LOAD_RES_1_MEM_]], [[CST_0_dot_000000_]] : f64
// CHECK:               [[VAR_41_:%.+]] = arith.addf [[VAR_27_1_]], [[VAR_40_]] : f64
// CHECK:               krnl.store [[VAR_41_]], [[RES_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:             [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to 2, [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to 3){
// CHECK:               [[VAR_26_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[VAR_27_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_26_2_]]#0, [[VAR_26_2_]]#1] : memref<2x3xf64>
// CHECK:               krnl.store [[VAR_27_1_]], [[RES_1_]]{{.}}[[VAR_26_2_]]#0, [[VAR_26_2_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3xf64>
// CHECK:         }
}

// -----

func.func @test_cumsum_dynamic_dims(%arg0: tensor<?x?xf64>, %arg1:tensor<i32>) -> tensor<*xf64> {
  %0 = "onnx.CumSum"(%arg0, %arg1) : (tensor<?x?xf64>, tensor<i32>) -> tensor<*xf64>
  return %0 : tensor<*xf64>

// mlir2FileCheck.py -a '["input","axis"]'
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0] -> (s0 + 2)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0)[s0, s1] -> (d0)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0, d1)[s0, s1] -> (d1)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0, d1)[s0, s1] -> (s1 + 1)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0, d1)[s0, s1] -> (d0)>
// CHECK-LABEL:  func.func @test_cumsum_dynamic_dims
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<?x?xf64>, [[AXIS_:%.+]]: memref<i32>) -> memref<?x?xf64> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[INPUT_]], [[CST_0_]] : memref<?x?xf64>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[INPUT_]], [[CST_1_]] : memref<?x?xf64>
// CHECK-DAG:       [[LOAD_AXIS_MEM_:%.+]] = krnl.load [[AXIS_]][] : memref<i32>
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[LOAD_AXIS_MEM_]] : i32 to index
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply [[MAP_2_]](){{.}}[[VAR_1_]]{{.}}
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.cmpi slt, [[VAR_1_]], [[CST_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.select [[VAR_3_]], [[VAR_2_]], [[VAR_1_]] : index
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[INPUT_]], [[CST_0_]] : memref<?x?xf64>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[INPUT_]], [[CST_1_]] : memref<?x?xf64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_1_]], [[VAR_dim_2_]]) {{.*}}: memref<?x?xf64>
// CHECK-DAG:       [[VAR_dim_3_:%.+]] = memref.dim [[INPUT_]], [[CST_0_]] : memref<?x?xf64>
// CHECK-DAG:       [[VAR_dim_4_:%.+]] = memref.dim [[INPUT_]], [[CST_1_]] : memref<?x?xf64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[VAR_dim_3_]], [[VAR_dim_4_]]) {{.*}}: memref<?x?xf64>
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.select [[VAR_5_]], [[VAR_dim_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_7_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_1_]] : index
// CHECK:           [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[VAR_dim_0_]], [[VAR_6_]] : index
// CHECK:           [[VAR_9_:%.+]] = arith.index_cast [[VAR_8_]] : index to i64
// CHECK:           [[VAR_10_:%.+]] = arith.sitofp [[VAR_9_]] : i64 to f32
// CHECK:           [[VAR_11_:%.+]] = math.log2 [[VAR_10_]] : f32
// CHECK:           [[VAR_12_:%.+]] = arith.fptosi [[VAR_11_]] : f32 to i64
// CHECK-DAG:       [[VAR_13_:%.+]] = arith.index_cast [[VAR_12_]] : i64 to index
// CHECK-DAG:       [[VAR_dim_6_:%.+]] = memref.dim [[INPUT_]], [[CST_0_]] : memref<?x?xf64>
// CHECK-DAG:       [[VAR_dim_7_:%.+]] = memref.dim [[INPUT_]], [[CST_1_]] : memref<?x?xf64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_3_]]([[VAR_dim_6_]]){{.}}[[VAR_1_]], [[VAR_1_]]3], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_4_]]([[VAR_dim_6_]], [[VAR_dim_7_]]){{.}}[[VAR_1_]], [[VAR_1_]]3]){
// CHECK:             [[VAR_16_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_16_]]#0, [[VAR_16_]]#1] : memref<?x?xf64>
// CHECK:             krnl.store [[LOAD_INPUT_MEM_]], [[RES_1_]]{{.}}[[VAR_16_]]#0, [[VAR_16_]]#1] : memref<?x?xf64>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to [[MAP_5_]]([[VAR_dim_6_]], [[VAR_dim_7_]]){{.}}[[VAR_1_]], [[VAR_1_]]3]){
// CHECK:             [[VAR_16_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_INPUT_MEM_1_:%.+]] = arith.index_cast [[VAR_16_1_]] : index to i64
// CHECK:             [[VAR_18_:%.+]] = arith.sitofp [[LOAD_INPUT_MEM_1_]] : i64 to f32
// CHECK:             [[VAR_19_:%.+]] = math.exp2 [[VAR_18_]] : f32
// CHECK:             [[VAR_20_:%.+]] = arith.fptosi [[VAR_19_]] : f32 to i64
// CHECK-DAG:         [[VAR_21_:%.+]] = arith.index_cast [[VAR_20_]] : i64 to index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to [[MAP_6_]]([[VAR_dim_6_]], [[VAR_dim_7_]]){{.}}[[VAR_1_]], [[VAR_1_]]3], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to [[MAP_4_]]([[VAR_dim_6_]], [[VAR_dim_7_]]){{.}}[[VAR_1_]], [[VAR_1_]]3]){
// CHECK:               [[VAR_24_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_24_]]#0, [[VAR_24_]]#1] : memref<?x?xf64>
// CHECK-DAG:           [[VAR_26_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:           [[VAR_27_:%.+]] = arith.subi [[VAR_24_]]#0, [[VAR_21_]] : index
// CHECK:               [[VAR_28_:%.+]] = arith.cmpi sge, [[VAR_27_]], [[CST_0_]] : index
// CHECK:               [[VAR_29_:%.+]] = arith.andi [[VAR_26_]], [[VAR_28_]] : i1
// CHECK-DAG:           [[VAR_30_:%.+]] = arith.select [[VAR_29_]], [[VAR_27_]], [[VAR_24_]]#0 : index
// CHECK-DAG:           [[VAR_31_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_1_]] : index
// CHECK-DAG:           [[VAR_32_:%.+]] = arith.subi [[VAR_24_]]#1, [[VAR_21_]] : index
// CHECK:               [[VAR_33_:%.+]] = arith.cmpi sge, [[VAR_32_]], [[CST_0_]] : index
// CHECK:               [[VAR_34_:%.+]] = arith.andi [[VAR_31_]], [[VAR_33_]] : i1
// CHECK-DAG:           [[VAR_35_:%.+]] = arith.ori [[VAR_34_]], [[VAR_29_]] : i1
// CHECK-DAG:           [[VAR_36_:%.+]] = arith.select [[VAR_34_]], [[VAR_32_]], [[VAR_24_]]#1 : index
// CHECK:               [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_30_]], [[VAR_36_]]{{.}} : memref<?x?xf64>
// CHECK:               [[VAR_38_:%.+]] = arith.select [[VAR_35_]], [[LOAD_RES_1_MEM_1_]], [[CST_0_dot_000000_]] : f64
// CHECK:               [[VAR_39_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[VAR_38_]] : f64
// CHECK:               krnl.store [[VAR_39_]], [[RES_]]{{.}}[[VAR_24_]]#0, [[VAR_24_]]#1] : memref<?x?xf64>
// CHECK:             }
// CHECK:             [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to [[MAP_6_]]([[VAR_dim_6_]], [[VAR_dim_7_]]){{.}}[[VAR_1_]], [[VAR_1_]]3], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to [[MAP_4_]]([[VAR_dim_6_]], [[VAR_dim_7_]]){{.}}[[VAR_1_]], [[VAR_1_]]3]){
// CHECK:               [[VAR_24_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_24_1_]]#0, [[VAR_24_1_]]#1] : memref<?x?xf64>
// CHECK:               krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_1_]]{{.}}[[VAR_24_1_]]#0, [[VAR_24_1_]]#1] : memref<?x?xf64>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?xf64>
// CHECK:         }
}
