// RUN: onnx-mlir-opt -O3 --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// -----

// Test whether lowering is correct for a string tensor input.

func.func private @test_category_mapper_string_to_int64(%arg0 : tensor<2x2x!onnx.String>) -> tensor<2x2xi64> {
  %0 = "onnx.CategoryMapper"(%arg0) {cats_int64s = [1, 2, 3], cats_strings = ["cat", "dog", "cow"], default_int64 = -1: si64} : (tensor<2x2x!onnx.String>) -> tensor<2x2xi64>
  "func.return"(%0) : (tensor<2x2xi64>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_category_mapper_string_to_int64
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x2x!krnl.string>) -> memref<2x2xi64> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : i64
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : i32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x2xi64>
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() <{name = "G0", shape = [3], value = dense<[1, 0, -3]> : tensor<3xi32>}> : () -> memref<3xi32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() <{name = "V1", shape = [3], value = dense<[1, 2, 0]> : tensor<3xi32>}> : () -> memref<3xi32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "krnl.global"() <{name = "cats_int64s2", shape = [3], value = dense<[1, 2, 3]> : tensor<3xi64>}> : () -> memref<3xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = "krnl.global"() <{name = "cats_strings3", shape = [3], value = dense<["cat", "dog", "cow"]> : tensor<3x!krnl.string>}> : () -> memref<3x!krnl.string>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK-DAG:         [[VAR_5_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_6_:%.+]] = "krnl.getref"([[PARAM_0_]], [[CST_0_1_]]) : (memref<2x2x!krnl.string>, i64) -> memref<2x2x!krnl.string>
// CHECK:             [[LOAD_VAR_6_MEM_:%.+]] = krnl.load [[VAR_6_]]{{.}}[[VAR_5_]]#0, [[VAR_5_]]#1] : memref<2x2x!krnl.string>
// CHECK:             [[VAR_8_:%.+]] = "krnl.find_index"([[LOAD_VAR_6_MEM_]], [[VAR_0_]], [[VAR_1_]], [[CST_3_]]) : (!krnl.string, memref<3xi32>, memref<3xi32>, i32) -> index
// CHECK:             [[LOAD_VAR_3_MEM_:%.+]] = krnl.load [[VAR_3_]]{{.}}[[VAR_8_]]{{.}} : memref<3x!krnl.string>
// CHECK:             [[VAR_10_:%.+]] = "krnl.strlen"([[LOAD_VAR_3_MEM_]]) : (!krnl.string) -> i64
// CHECK:             [[VAR_11_:%.+]] = "krnl.strncmp"([[LOAD_VAR_6_MEM_]], [[LOAD_VAR_3_MEM_]], [[VAR_10_]]) : (!krnl.string, !krnl.string, i64) -> i32
// CHECK:             [[VAR_12_:%.+]] = arith.cmpi eq, [[VAR_11_]], [[CST_0_]] : i32
// CHECK:             scf.if [[VAR_12_]] {
// CHECK:               [[LOAD_VAR_2_MEM_:%.+]] = krnl.load [[VAR_2_]]{{.}}[[VAR_8_]]{{.}} : memref<3xi64>
// CHECK:               krnl.store [[LOAD_VAR_2_MEM_]], [[RES_]]{{.}}[[VAR_5_]]#0, [[VAR_5_]]#1] : memref<2x2xi64>
// CHECK:             } else {
// CHECK:               krnl.store [[CST_minus_1_]], [[RES_]]{{.}}[[VAR_5_]]#0, [[VAR_5_]]#1] : memref<2x2xi64>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x2xi64>
// CHECK:         }
}

// -----

// Test whether lowering is correct for a int64_t tensor input.

func.func private @test_category_mapper_int64_to_string(%arg0 : tensor<2x2xi64>) -> tensor<2x2x!onnx.String> {
  %0 = "onnx.CategoryMapper"(%arg0) {cats_int64s = [1, 2, 3], cats_strings = ["cat", "dog", "cow"], default_string = "none"} : (tensor<2x2xi64>) -> tensor<2x2x!onnx.String>
  "func.return"(%0) : (tensor<2x2x!onnx.String>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_category_mapper_int64_to_string
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x2xi64>) -> memref<2x2x!krnl.string> {
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : i32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x2x!krnl.string>
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() <{name = "G0", shape = [3], value = dense<[-1, 1, 0]> : tensor<3xi32>}> : () -> memref<3xi32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() <{name = "V1", shape = [3], value = dense<[2, 1, 0]> : tensor<3xi32>}> : () -> memref<3xi32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "krnl.global"() <{name = "cats_int64s2", shape = [3], value = dense<[1, 2, 3]> : tensor<3xi64>}> : () -> memref<3xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = "krnl.global"() <{name = "cats_strings3", shape = [3], value = dense<["cat", "dog", "cow"]> : tensor<3x!krnl.string>}> : () -> memref<3x!krnl.string>
// CHECK-DAG:       [[VAR_4_:%.+]] = "krnl.global"() <{name = "default_string4", shape = [], value = dense<"none"> : tensor<!krnl.string>}> : () -> memref<!krnl.string>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[VAR_6_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]]#0, [[VAR_6_]]#1] : memref<2x2xi64>
// CHECK:             [[VAR_8_:%.+]] = "krnl.find_index"([[LOAD_PARAM_0_MEM_]], [[VAR_0_]], [[VAR_1_]], [[CST_3_]]) : (i64, memref<3xi32>, memref<3xi32>, i32) -> index
// CHECK:             [[LOAD_VAR_2_MEM_:%.+]] = krnl.load [[VAR_2_]]{{.}}[[VAR_8_]]{{.}} : memref<3xi64>
// CHECK:             [[VAR_10_:%.+]] = arith.cmpi eq, [[LOAD_PARAM_0_MEM_]], [[LOAD_VAR_2_MEM_]] : i64
// CHECK:             scf.if [[VAR_10_]] {
// CHECK:               [[LOAD_VAR_3_MEM_:%.+]] = krnl.load [[VAR_3_]]{{.}}[[VAR_8_]]{{.}} : memref<3x!krnl.string>
// CHECK:               krnl.store [[LOAD_VAR_3_MEM_]], [[RES_]]{{.}}[[VAR_6_]]#0, [[VAR_6_]]#1] : memref<2x2x!krnl.string>
// CHECK:             } else {
// CHECK:               [[LOAD_VAR_3_MEM_1_:%.+]] = krnl.load [[VAR_4_]][] : memref<!krnl.string>
// CHECK:               krnl.store [[LOAD_VAR_3_MEM_1_]], [[RES_]]{{.}}[[VAR_6_]]#0, [[VAR_6_]]#1] : memref<2x2x!krnl.string>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x2x!krnl.string>
// CHECK:         }
}

// -----

// Test whether lowering is correct for a rank-3 string tensor input.

func.func private @test_rank3_category_mapper_string_to_int64(%arg0 : tensor<2x2x2x!onnx.String>) -> tensor<2x2x2xi64> {
  %0 = "onnx.CategoryMapper"(%arg0) {cats_int64s = [1, 2, 3], cats_strings = ["cat", "dog", "cow"], default_int64 = -1: si64} : (tensor<2x2x2x!onnx.String>) -> tensor<2x2x2xi64>
  "func.return"(%0) : (tensor<2x2x2xi64>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_rank3_category_mapper_string_to_int64
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x2x2x!krnl.string>) -> memref<2x2x2xi64> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : i64
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : i32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x2x2xi64>
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() <{name = "G0", shape = [3], value = dense<[1, 0, -3]> : tensor<3xi32>}> : () -> memref<3xi32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() <{name = "V1", shape = [3], value = dense<[1, 2, 0]> : tensor<3xi32>}> : () -> memref<3xi32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "krnl.global"() <{name = "cats_int64s2", shape = [3], value = dense<[1, 2, 3]> : tensor<3xi64>}> : () -> memref<3xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = "krnl.global"() <{name = "cats_strings3", shape = [3], value = dense<["cat", "dog", "cow"]> : tensor<3x!krnl.string>}> : () -> memref<3x!krnl.string>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2){
// CHECK-DAG:         [[VAR_5_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[VAR_6_:%.+]] = "krnl.getref"([[PARAM_0_]], [[CST_0_1_]]) : (memref<2x2x2x!krnl.string>, i64) -> memref<2x2x2x!krnl.string>
// CHECK:             [[LOAD_VAR_6_MEM_:%.+]] = krnl.load [[VAR_6_]]{{.}}[[VAR_5_]]#0, [[VAR_5_]]#1, [[VAR_5_]]#2] : memref<2x2x2x!krnl.string>
// CHECK:             [[VAR_8_:%.+]] = "krnl.find_index"([[LOAD_VAR_6_MEM_]], [[VAR_0_]], [[VAR_1_]], [[CST_3_]]) : (!krnl.string, memref<3xi32>, memref<3xi32>, i32) -> index
// CHECK:             [[LOAD_VAR_3_MEM_:%.+]] = krnl.load [[VAR_3_]]{{.}}[[VAR_8_]]{{.}} : memref<3x!krnl.string>
// CHECK:             [[VAR_10_:%.+]] = "krnl.strlen"([[LOAD_VAR_3_MEM_]]) : (!krnl.string) -> i64
// CHECK:             [[VAR_11_:%.+]] = "krnl.strncmp"([[LOAD_VAR_6_MEM_]], [[LOAD_VAR_3_MEM_]], [[VAR_10_]]) : (!krnl.string, !krnl.string, i64) -> i32
// CHECK:             [[VAR_12_:%.+]] = arith.cmpi eq, [[VAR_11_]], [[CST_0_]] : i32
// CHECK:             scf.if [[VAR_12_]] {
// CHECK:               [[LOAD_VAR_2_MEM_:%.+]] = krnl.load [[VAR_2_]]{{.}}[[VAR_8_]]{{.}} : memref<3xi64>
// CHECK:               krnl.store [[LOAD_VAR_2_MEM_]], [[RES_]]{{.}}[[VAR_5_]]#0, [[VAR_5_]]#1, [[VAR_5_]]#2] : memref<2x2x2xi64>
// CHECK:             } else {
// CHECK:               krnl.store [[CST_minus_1_]], [[RES_]]{{.}}[[VAR_5_]]#0, [[VAR_5_]]#1, [[VAR_5_]]#2] : memref<2x2x2xi64>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x2x2xi64>
// CHECK:         }
}

// -----

// Test whether lowering is correct for a int64_t tensor input.

func.func private @test_rank3_category_mapper_int64_to_string(%arg0 : tensor<2x2x2xi64>) -> tensor<2x2x2x!onnx.String> {
  %0 = "onnx.CategoryMapper"(%arg0) {cats_int64s = [1, 2, 3], cats_strings = ["cat", "dog", "cow"], default_string = "none"} : (tensor<2x2x2xi64>) -> tensor<2x2x2x!onnx.String>
  "func.return"(%0) : (tensor<2x2x2x!onnx.String>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_rank3_category_mapper_int64_to_string
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x2x2xi64>) -> memref<2x2x2x!krnl.string> {
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : i32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x2x2x!krnl.string>
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() <{name = "G0", shape = [3], value = dense<[-1, 1, 0]> : tensor<3xi32>}> : () -> memref<3xi32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() <{name = "V1", shape = [3], value = dense<[2, 1, 0]> : tensor<3xi32>}> : () -> memref<3xi32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "krnl.global"() <{name = "cats_int64s2", shape = [3], value = dense<[1, 2, 3]> : tensor<3xi64>}> : () -> memref<3xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = "krnl.global"() <{name = "cats_strings3", shape = [3], value = dense<["cat", "dog", "cow"]> : tensor<3x!krnl.string>}> : () -> memref<3x!krnl.string>
// CHECK-DAG:       [[VAR_4_:%.+]] = "krnl.global"() <{name = "default_string4", shape = [], value = dense<"none"> : tensor<!krnl.string>}> : () -> memref<!krnl.string>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2){
// CHECK:             [[VAR_6_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]]#0, [[VAR_6_]]#1, [[VAR_6_]]#2] : memref<2x2x2xi64>
// CHECK:             [[VAR_8_:%.+]] = "krnl.find_index"([[LOAD_PARAM_0_MEM_]], [[VAR_0_]], [[VAR_1_]], [[CST_3_]]) : (i64, memref<3xi32>, memref<3xi32>, i32) -> index
// CHECK:             [[LOAD_VAR_2_MEM_:%.+]] = krnl.load [[VAR_2_]]{{.}}[[VAR_8_]]{{.}} : memref<3xi64>
// CHECK:             [[VAR_10_:%.+]] = arith.cmpi eq, [[LOAD_PARAM_0_MEM_]], [[LOAD_VAR_2_MEM_]] : i64
// CHECK:             scf.if [[VAR_10_]] {
// CHECK:               [[LOAD_VAR_3_MEM_:%.+]] = krnl.load [[VAR_3_]]{{.}}[[VAR_8_]]{{.}} : memref<3x!krnl.string>
// CHECK:               krnl.store [[LOAD_VAR_3_MEM_]], [[RES_]]{{.}}[[VAR_6_]]#0, [[VAR_6_]]#1, [[VAR_6_]]#2] : memref<2x2x2x!krnl.string>
// CHECK:             } else {
// CHECK:               [[LOAD_VAR_3_MEM_1_:%.+]] = krnl.load [[VAR_4_]][] : memref<!krnl.string>
// CHECK:               krnl.store [[LOAD_VAR_3_MEM_1_]], [[RES_]]{{.}}[[VAR_6_]]#0, [[VAR_6_]]#1, [[VAR_6_]]#2] : memref<2x2x2x!krnl.string>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x2x2x!krnl.string>
// CHECK:         }
}

