// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

// Compress on axis 0, with enough conditions, test elided

func.func @compress_axis0(%arg0: tensor<3x2xf32>, %arg1: tensor<3xi1>) -> tensor<?x2xf32> {
  %0 = "onnx.Compress"(%arg0, %arg1) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<3xi1>) -> tensor<?x2xf32>
  return %0 : tensor<?x2xf32>
//  use arg names: ['input', 'condition']
// mlir2FileCheck.py -a'["input", "condition"]'
// CHECK-LABEL:  func @compress_axis0
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<3x2xf32>, [[CONDITION_:%.+]]: memref<3xi1>) -> memref<?x2xf32> {
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 3){
// CHECK:             [[VAR_5_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_CONDITION_MEM_:%.+]] = krnl.load [[CONDITION_]]{{.}}[[VAR_5_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_7_:%.+]] = arith.cmpi ne, [[LOAD_CONDITION_MEM_]], [[VAR_false_]] : i1
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[VAR_c1_]], [[VAR_c0_]] : index
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:             [[VAR_10_:%.+]] = arith.addi [[LOAD_RES_MEM_]], [[VAR_8_]] : index
// CHECK:             krnl.store [[VAR_10_]], [[RES_]][] : memref<index>
// CHECK:           }
// CHECK:           [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:           [[RES_1_:%.+]] = memref.alloc([[LOAD_RES_MEM_1_]]) {{.*}}: memref<?x2xf32>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_]][] : memref<index>
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 3){
// CHECK:             [[VAR_5_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_CONDITION_MEM_1_:%.+]] = krnl.load [[CONDITION_]]{{.}}[[VAR_5_1_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_7_1_:%.+]] = arith.cmpi ne, [[LOAD_CONDITION_MEM_1_]], [[VAR_false_]] : i1
// CHECK:             scf.if [[VAR_7_1_]] {
// CHECK-DAG:           [[LOAD_RES_MEM_2_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:               krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 2){
// CHECK:                 [[VAR_11_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK:                 [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_5_1_]], [[VAR_11_]]{{.}} : memref<3x2xf32>
// CHECK:                 krnl.store [[LOAD_INPUT_MEM_]], [[RES_1_]]{{.}}[[LOAD_RES_MEM_2_]], [[VAR_11_]]{{.}} : memref<?x2xf32>
// CHECK:               }
// CHECK:               [[VAR_10_1_:%.+]] = arith.addi [[LOAD_RES_MEM_2_]], [[VAR_c1_]] : index
// CHECK:               krnl.store [[VAR_10_1_]], [[RES_]][] : memref<index>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<?x2xf32>
// CHECK:         }
}

// -----

// Compress on axis 0, with not enough conditions, test not elided

func.func @compress_axis0_not_enough(%arg0: tensor<3x2xf32>, %arg1: tensor<2xi1>) -> tensor<?x2xf32> {
  %0 = "onnx.Compress"(%arg0, %arg1) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<2xi1>) -> tensor<?x2xf32>
  return %0 : tensor<?x2xf32>
// mlir2FileCheck.py -a'["input", "condition"]'
// CHECK-LABEL:  func @compress_axis0_not_enough
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<3x2xf32>, [[CONDITION_:%.+]]: memref<2xi1>) -> memref<?x2xf32> {
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 2){
// CHECK:             [[VAR_5_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_CONDITION_MEM_:%.+]] = krnl.load [[CONDITION_]]{{.}}[[VAR_5_]]{{.}} : memref<2xi1>
// CHECK:             [[VAR_7_:%.+]] = arith.cmpi ne, [[LOAD_CONDITION_MEM_]], [[VAR_false_]] : i1
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[VAR_c1_]], [[VAR_c0_]] : index
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:             [[VAR_10_:%.+]] = arith.addi [[LOAD_RES_MEM_]], [[VAR_8_]] : index
// CHECK:             krnl.store [[VAR_10_]], [[RES_]][] : memref<index>
// CHECK:           }
// CHECK:           [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:           [[RES_1_:%.+]] = memref.alloc([[LOAD_RES_MEM_1_]]) {{.*}}: memref<?x2xf32>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_]][] : memref<index>
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 3){
// CHECK:             [[VAR_5_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_CONDITION_MEM_1_:%.+]] = arith.cmpi slt, [[VAR_5_1_]], [[VAR_c2_]] : index
// CHECK:             scf.if [[LOAD_CONDITION_MEM_1_]] {
// CHECK:               [[LOAD_CONDITION_MEM_2_:%.+]] = krnl.load [[CONDITION_]]{{.}}[[VAR_5_1_]]{{.}} : memref<2xi1>
// CHECK:               [[VAR_8_1_:%.+]] = arith.cmpi ne, [[LOAD_CONDITION_MEM_2_]], [[VAR_false_]] : i1
// CHECK:               scf.if [[VAR_8_1_]] {
// CHECK-DAG:             [[LOAD_RES_MEM_2_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:             [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 2){
// CHECK:                   [[VAR_12_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK:                   [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_5_1_]], [[VAR_12_]]{{.}} : memref<3x2xf32>
// CHECK:                   krnl.store [[LOAD_INPUT_MEM_]], [[RES_1_]]{{.}}[[LOAD_RES_MEM_2_]], [[VAR_12_]]{{.}} : memref<?x2xf32>
// CHECK:                 }
// CHECK:                 [[VAR_11_:%.+]] = arith.addi [[LOAD_RES_MEM_2_]], [[VAR_c1_]] : index
// CHECK:                 krnl.store [[VAR_11_]], [[RES_]][] : memref<index>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<?x2xf32>
// CHECK:         }
}

// -----

// Compress on axis 1, with enough conditions, test elided

func.func @compress_axis1(%arg0: tensor<3x2xf32>, %arg1: tensor<3xi1>) -> tensor<3x?xf32> {
  %0 = "onnx.Compress"(%arg0, %arg1) {axis = 1 : si64} : (tensor<3x2xf32>, tensor<3xi1>) -> tensor<3x?xf32>
  return %0 : tensor<3x?xf32>
// mlir2FileCheck.py -a'["input", "condition"]'
// CHECK-LABEL:  func @compress_axis1
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<3x2xf32>, [[CONDITION_:%.+]]: memref<3xi1>) -> memref<3x?xf32> {
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 3){
// CHECK:             [[VAR_5_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_CONDITION_MEM_:%.+]] = krnl.load [[CONDITION_]]{{.}}[[VAR_5_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_7_:%.+]] = arith.cmpi ne, [[LOAD_CONDITION_MEM_]], [[VAR_false_]] : i1
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[VAR_c1_]], [[VAR_c0_]] : index
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:             [[VAR_10_:%.+]] = arith.addi [[LOAD_RES_MEM_]], [[VAR_8_]] : index
// CHECK:             krnl.store [[VAR_10_]], [[RES_]][] : memref<index>
// CHECK:           }
// CHECK:           [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:           [[RES_1_:%.+]] = memref.alloc([[LOAD_RES_MEM_1_]]) {{.*}}: memref<3x?xf32>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_]][] : memref<index>
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[VAR_5_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_CONDITION_MEM_1_:%.+]] = krnl.load [[CONDITION_]]{{.}}[[VAR_5_1_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_7_1_:%.+]] = arith.cmpi ne, [[LOAD_CONDITION_MEM_1_]], [[VAR_false_]] : i1
// CHECK:             scf.if [[VAR_7_1_]] {
// CHECK-DAG:           [[LOAD_RES_MEM_2_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:               krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 3){
// CHECK:                 [[VAR_11_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK:                 [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_11_]], [[VAR_5_1_]]{{.}} : memref<3x2xf32>
// CHECK:                 krnl.store [[LOAD_INPUT_MEM_]], [[RES_1_]]{{.}}[[VAR_11_]], [[LOAD_RES_MEM_2_]]{{.}} : memref<3x?xf32>
// CHECK:               }
// CHECK:               [[VAR_10_1_:%.+]] = arith.addi [[LOAD_RES_MEM_2_]], [[VAR_c1_]] : index
// CHECK:               krnl.store [[VAR_10_1_]], [[RES_]][] : memref<index>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<3x?xf32>
// CHECK:         }
}

// -----

// Compress witn no axis , with not enough conditions, test not elided

func.func @compress_no_axis_not_elided(%arg0: tensor<3x2xf32>, %arg1: tensor<3xi1>) -> tensor<?xf32> {
  %0 = "onnx.Compress"(%arg0, %arg1) : (tensor<3x2xf32>, tensor<3xi1>) -> tensor<?xf32>
  return %0 : tensor<?xf32>

// CHECK-LABEL:  func @compress_no_axis_not_elided
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<3x2xf32>, [[CONDITION_:%.+]]: memref<3xi1>) -> memref<?xf32> {
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 3){
// CHECK:             [[VAR_6_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_CONDITION_MEM_:%.+]] = krnl.load [[CONDITION_]]{{.}}[[VAR_6_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_8_:%.+]] = arith.cmpi ne, [[LOAD_CONDITION_MEM_]], [[VAR_false_]] : i1
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.select [[VAR_8_]], [[VAR_c1_]], [[VAR_c0_]] : index
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:             [[VAR_11_:%.+]] = arith.addi [[LOAD_RES_MEM_]], [[VAR_9_]] : index
// CHECK:             krnl.store [[VAR_11_]], [[RES_]][] : memref<index>
// CHECK:           }
// CHECK:           [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:           [[RES_1_:%.+]] = memref.alloc([[LOAD_RES_MEM_1_]]) {{.*}}: memref<?xf32>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_]][] : memref<index>
// CHECK:           [[RES_2_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_2_]][] : memref<index>
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_2_:%.+]] = 0 to 2){
// CHECK-DAG:         [[VAR_6_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_CONDITION_MEM_1_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:             [[VAR_8_1_:%.+]] = arith.cmpi slt, [[LOAD_CONDITION_MEM_1_]], [[VAR_c3_]] : index
// CHECK:             scf.if [[VAR_8_1_]] {
// CHECK:               [[LOAD_CONDITION_MEM_2_:%.+]] = krnl.load [[CONDITION_]]{{.}}[[LOAD_CONDITION_MEM_1_]]{{.}} : memref<3xi1>
// CHECK:               [[LOAD_RES_MEM_2_:%.+]] = arith.cmpi ne, [[LOAD_CONDITION_MEM_2_]], [[VAR_false_]] : i1
// CHECK:               scf.if [[LOAD_RES_MEM_2_]] {
// CHECK-DAG:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_6_1_]]#0, [[VAR_6_1_]]#1] : memref<3x2xf32>
// CHECK-DAG:             [[LOAD_RES_MEM_3_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:                 krnl.store [[LOAD_INPUT_MEM_]], [[RES_1_]]{{.}}[[LOAD_RES_MEM_3_]]{{.}} : memref<?xf32>
// CHECK:                 [[VAR_14_:%.+]] = arith.addi [[LOAD_RES_MEM_3_]], [[VAR_c1_]] : index
// CHECK:                 krnl.store [[VAR_14_]], [[RES_]][] : memref<index>
// CHECK:               }
// CHECK:               [[VAR_11_1_:%.+]] = arith.addi [[LOAD_CONDITION_MEM_1_]], [[VAR_c1_]] : index
// CHECK:               krnl.store [[VAR_11_1_]], [[RES_2_]][] : memref<index>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<?xf32>
// CHECK:         }
}

// -----

// Compress witn no axis , with enough conditions, test elided

func.func @compress_no_axis_enough_cond(%arg0: tensor<3x2xf32>, %arg1: tensor<6xi1>) -> tensor<?xf32> {
  %0 = "onnx.Compress"(%arg0, %arg1) : (tensor<3x2xf32>, tensor<6xi1>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
// CHECK-LABEL:  func @compress_no_axis_enough_cond
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<3x2xf32>, [[CONDITION_:%.+]]: memref<6xi1>) -> memref<?xf32> {
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 6){
// CHECK:             [[VAR_6_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_CONDITION_MEM_:%.+]] = krnl.load [[CONDITION_]]{{.}}[[VAR_6_]]{{.}} : memref<6xi1>
// CHECK:             [[VAR_8_:%.+]] = arith.cmpi ne, [[LOAD_CONDITION_MEM_]], [[VAR_false_]] : i1
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.select [[VAR_8_]], [[VAR_c1_]], [[VAR_c0_]] : index
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:             [[VAR_11_:%.+]] = arith.addi [[LOAD_RES_MEM_]], [[VAR_9_]] : index
// CHECK:             krnl.store [[VAR_11_]], [[RES_]][] : memref<index>
// CHECK:           }
// CHECK:           [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:           [[RES_1_:%.+]] = memref.alloc([[LOAD_RES_MEM_1_]]) {{.*}}: memref<?xf32>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_]][] : memref<index>
// CHECK:           [[RES_2_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_2_]][] : memref<index>
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_2_:%.+]] = 0 to 2){
// CHECK-DAG:         [[VAR_6_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_CONDITION_MEM_1_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:             [[LOAD_CONDITION_MEM_2_:%.+]] = krnl.load [[CONDITION_]]{{.}}[[LOAD_CONDITION_MEM_1_]]{{.}} : memref<6xi1>
// CHECK:             [[VAR_9_1_:%.+]] = arith.cmpi ne, [[LOAD_CONDITION_MEM_2_]], [[VAR_false_]] : i1
// CHECK:             scf.if [[VAR_9_1_]] {
// CHECK-DAG:           [[VAR_11_1_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_6_1_]]#0, [[VAR_6_1_]]#1] : memref<3x2xf32>
// CHECK-DAG:           [[LOAD_RES_MEM_2_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:               krnl.store [[VAR_11_1_]], [[RES_1_]]{{.}}[[LOAD_RES_MEM_2_]]{{.}} : memref<?xf32>
// CHECK:               [[VAR_13_:%.+]] = arith.addi [[LOAD_RES_MEM_2_]], [[VAR_c1_]] : index
// CHECK:               krnl.store [[VAR_13_]], [[RES_]][] : memref<index>
// CHECK:             }
// CHECK:             [[LOAD_RES_MEM_3_:%.+]] = arith.addi [[LOAD_CONDITION_MEM_1_]], [[VAR_c1_]] : index
// CHECK:             krnl.store [[LOAD_RES_MEM_3_]], [[RES_2_]][] : memref<index>
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<?xf32>
// CHECK:         }
}
