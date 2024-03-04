// RUN: onnx-mlir-opt -O3 --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// -----

// Test whether lowering is correct for a string tensor input.
func.func private @test_category_mapper_string_to_int64(%arg0 : tensor<2x2x!onnx.String>) -> tensor<2x2xi64> {
  %0 = "onnx.CategoryMapper"(%arg0) {cats_int64s = [1, 2, 3], cats_strings = ["cat", "dog", "cow"], default_int64 = -1: si64} : (tensor<2x2x!onnx.String>) -> tensor<2x2xi64>
  "func.return"(%0) : (tensor<2x2xi64>) -> ()

  // CHECK-LABEL: test_category_mapper_string_to_int64
  // CHECK-DAG: [[LEN:%.+]] = arith.constant 3 : i32
  // CHECK-DAG: [[ALLOCA:%.+]] = memref.alloc() {alignment = 16 : i64} : memref<2x2xi64>
  // CHECK-DAG: [[G:%.+]] = "krnl.global"() {name = {{.*}}, shape = [3], value = dense<[1, 0, -3]> : tensor<3xi32>} : () -> memref<3xi32>
  // CHECK-DAG: [[V:%.+]] = "krnl.global"() {name = {{.*}}, shape = [3], value = dense<[1, 2, 0]> : tensor<3xi32>} : () -> memref<3xi32>
  // CHECK-DAG: [[CAT_INT64s:%.+]] = "krnl.global"() {name = {{.*}}, shape = [3], value = dense<[1, 2, 3]> : tensor<3xi64>} : () -> memref<3xi64>
  // CHECK-DAG: [[CAT_STRINGS:%.+]] = "krnl.global"() {name = {{.*}}, shape = [3], value = dense<["cat", "dog", "cow"]> : tensor<3x!krnl.string>} : () -> memref<3x!krnl.string>
  // CHECK-DAG: [[DEFAULT_INT64:%.+]] = arith.constant -1 : i64
  // CHECK-DAG: [[ZERO:%.+]] = arith.constant 0 : i32
  // CHECK-DAG: [[LOOP_0:%.+]]:2 = krnl.define_loops 2
  // CHECK:     krnl.iterate([[LOOP_0]]#0, [[LOOP_0]]#1) with ([[LOOP_0]]#0 -> [[I_0:%.+]] = 0 to 2, [[LOOP_0]]#1 -> [[I_1:%.+]] = 0 to 2){  
  // CHECK:     [[IVS:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0]]#0, [[LOOP_0]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
  // CHECK:     [[LOAD1:%.+]] = krnl.load %arg0{{.}}[[IVS]]#0, [[IVS]]#1{{.}} : memref<2x2x!krnl.string>
  // CHECK:     [[INDEX:%.+]] = "krnl.find_index"([[LOAD1]], [[G]], [[V]], [[LEN]]) : (!krnl.string, memref<3xi32>, memref<3xi32>, i32) -> index
  // CHECK:     [[LOAD2:%.+]] = krnl.load [[CAT_STRINGS]]{{.}}[[INDEX]]{{.}} : memref<3x!krnl.string>
  // CHECK:     [[STRLEN:%.+]] = "krnl.strlen"([[LOAD2]]) : (!krnl.string) -> i64
  // CHECK:     [[STRNCMP:%.+]] = "krnl.strncmp"([[LOAD1]], [[LOAD2]], [[STRLEN]]) : (!krnl.string, !krnl.string, i64) -> i32
  // CHECK:     [[VALID:%.+]] = arith.cmpi eq, [[STRNCMP]], [[ZERO]] : i32
  // CHECK:     scf.if [[VALID]] {
  // CHECK:     [[LOAD3:%.+]] = krnl.load [[CAT_INT64s]]{{.}}[[INDEX]]{{.}} : memref<3xi64>
  // CHECK:     krnl.store [[LOAD3]], [[ALLOCA]]{{.}}[[IVS]]#0, [[IVS]]#1{{.}} : memref<2x2xi64>
  // CHECK:     } else {
  // CHECK:     krnl.store [[DEFAULT_INT64]], [[ALLOCA]]{{.}}[[IVS]]#0, [[IVS]]#1{{.}} : memref<2x2xi64>
  // CHECK:     }
  // CHECK:     return [[ALLOCA]] : memref<2x2xi64>
}

// -----

// Test whether lowering is correct for a int64_t tensor input.
func.func private @test_category_mapper_int64_to_string(%arg0 : tensor<2x2xi64>) -> tensor<2x2x!onnx.String> {
  %0 = "onnx.CategoryMapper"(%arg0) {cats_int64s = [1, 2, 3], cats_strings = ["cat", "dog", "cow"], default_string = "none"} : (tensor<2x2xi64>) -> tensor<2x2x!onnx.String>
  "func.return"(%0) : (tensor<2x2x!onnx.String>) -> ()

  // CHECK-LABEL: test_category_mapper_int64_to_string
  // CHECK-DAG: [[LEN:%.+]] = arith.constant 3 : i32
  // CHECK-DAG: [[ALLOCA:%.+]] = memref.alloc() {alignment = 16 : i64} : memref<2x2x!krnl.string>
  // CHECK-DAG: [[G:%.+]] = "krnl.global"() {name = {{.*}}, shape = [3], value = dense<[-1, 1, 0]> : tensor<3xi32>} : () -> memref<3xi32>
  // CHECK-DAG: [[V:%.+]] = "krnl.global"() {name = {{.*}}, shape = [3], value = dense<[2, 1, 0]> : tensor<3xi32>} : () -> memref<3xi32>
  // CHECK-DAG: [[CAT_INT64s:%.+]] = "krnl.global"() {name = {{.*}}, shape = [3], value = dense<[1, 2, 3]> : tensor<3xi64>} : () -> memref<3xi64>
  // CHECK-DAG: [[CAT_STRINGS:%.+]] = "krnl.global"() {name = {{.*}}, shape = [3], value = dense<["cat", "dog", "cow"]> : tensor<3x!krnl.string>} : () -> memref<3x!krnl.string>
  // CHECK-DAG: [[DEFAULT_STRING:%.+]] = "krnl.global"() {name = {{.*}}, shape = [], value = dense<"none"> : tensor<!krnl.string>} : () -> memref<!krnl.string>
  // CHECK-DAG: [[LOOP_0:%.+]]:2 = krnl.define_loops 2
  // CHECK:     krnl.iterate([[LOOP_0]]#0, [[LOOP_0]]#1) with ([[LOOP_0]]#0 -> [[I_0:%.+]] = 0 to 2, [[LOOP_0]]#1 -> [[I_1:%.+]] = 0 to 2){
  // CHECK:     [[IVS:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0]]#0, [[LOOP_0]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
  // CHECK:     [[LOAD1:%.+]] = krnl.load %arg0{{.}}[[IVS]]#0, [[IVS]]#1{{.}} : memref<2x2xi64>
  // CHECK:     [[INDEX:%.+]] = "krnl.find_index"([[LOAD1]], [[G]], [[V]], [[LEN]]) : (i64, memref<3xi32>, memref<3xi32>, i32) -> index
  // CHECK:     [[LOAD2:%.+]] = krnl.load [[CAT_INT64s]]{{.}}[[INDEX]]{{.}} : memref<3xi64>
  // CHECK:     [[VALID:%.+]] = arith.cmpi eq, [[LOAD1]], [[LOAD2]] : i64
  // CHECK:     scf.if [[VALID]] {
  // CHECK:     [[LOAD3:%.+]] = krnl.load [[CAT_STRINGS]]{{.}}[[INDEX]]{{.}} : memref<3x!krnl.string>
  // CHECK:     krnl.store [[LOAD3]], [[ALLOCA]]{{.}}[[IVS]]#0, [[IVS]]#1{{.}} : memref<2x2x!krnl.string>
  // CHECK:     } else {
  // CHECK:     [[LOAD4:%.+]] = krnl.load [[DEFAULT_STRING]][] : memref<!krnl.string>    
  // CHECK:     krnl.store [[LOAD4]], [[ALLOCA]]{{.}}[[IVS]]#0, [[IVS]]#1{{.}} : memref<2x2x!krnl.string>
  // CHECK:     }
  // CHECK:     return [[ALLOCA]] : memref<2x2x!krnl.string>
}

// -----

// Test whether lowering is correct for a rank-3 string tensor input.
func.func private @test_rank3_category_mapper_string_to_int64(%arg0 : tensor<2x2x2x!onnx.String>) -> tensor<2x2x2xi64> {
  %0 = "onnx.CategoryMapper"(%arg0) {cats_int64s = [1, 2, 3], cats_strings = ["cat", "dog", "cow"], default_int64 = -1: si64} : (tensor<2x2x2x!onnx.String>) -> tensor<2x2x2xi64>
  "func.return"(%0) : (tensor<2x2x2xi64>) -> ()

  // CHECK-LABEL: test_rank3_category_mapper_string_to_int64
  // CHECK-DAG: [[LEN:%.+]] = arith.constant 3 : i32
  // CHECK-DAG: [[ALLOCA:%.+]] = memref.alloc() {alignment = 16 : i64} : memref<2x2x2xi64>
  // CHECK-DAG: [[G:%.+]] = "krnl.global"() {name = {{.*}}, shape = [3], value = dense<[1, 0, -3]> : tensor<3xi32>} : () -> memref<3xi32>
  // CHECK-DAG: [[V:%.+]] = "krnl.global"() {name = {{.*}}, shape = [3], value = dense<[1, 2, 0]> : tensor<3xi32>} : () -> memref<3xi32>
  // CHECK-DAG: [[CAT_INT64s:%.+]] = "krnl.global"() {name = {{.*}}, shape = [3], value = dense<[1, 2, 3]> : tensor<3xi64>} : () -> memref<3xi64>
  // CHECK-DAG: [[CAT_STRINGS:%.+]] = "krnl.global"() {name = {{.*}}, shape = [3], value = dense<["cat", "dog", "cow"]> : tensor<3x!krnl.string>} : () -> memref<3x!krnl.string>
  // CHECK-DAG: [[DEFAULT_INT64:%.+]] = arith.constant -1 : i64
  // CHECK-DAG: [[ZERO:%.+]] = arith.constant 0 : i32
  // CHECK-DAG: [[LOOP_0:%.+]]:3 = krnl.define_loops 3
  // CHECK:     krnl.iterate([[LOOP_0]]#0, [[LOOP_0]]#1, [[LOOP_0]]#2) with ([[LOOP_0]]#0 -> %arg1 = 0 to 2, [[LOOP_0]]#1 -> %arg2 = 0 to 2, [[LOOP_0]]#2 -> %arg3 = 0 to 2){
  // CHECK:     [[IVS:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0]]#0, [[LOOP_0]]#1, [[LOOP_0]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
  // CHECK:     [[LOAD1:%.+]] = krnl.load %arg0{{.}}[[IVS]]#0, [[IVS]]#1, [[IVS]]#2{{.}} : memref<2x2x2x!krnl.string>
  // CHECK:     [[INDEX:%.+]] = "krnl.find_index"([[LOAD1]], [[G]], [[V]], [[LEN]]) : (!krnl.string, memref<3xi32>, memref<3xi32>, i32) -> index
  // CHECK:     [[LOAD2:%.+]] = krnl.load [[CAT_STRINGS]]{{.}}[[INDEX]]{{.}} : memref<3x!krnl.string>
  // CHECK:     [[STRLEN:%.+]] = "krnl.strlen"([[LOAD2]]) : (!krnl.string) -> i64
  // CHECK:     [[STRNCMP:%.+]] = "krnl.strncmp"([[LOAD1]], [[LOAD2]], [[STRLEN]]) : (!krnl.string, !krnl.string, i64) -> i32
  // CHECK:     [[VALID:%.+]] = arith.cmpi eq, [[STRNCMP]], [[ZERO]] : i32
  // CHECK:     scf.if [[VALID]] {
  // CHECK:     [[LOAD3:%.+]] = krnl.load [[CAT_INT64s]]{{.}}[[INDEX]]{{.}} : memref<3xi64>
  // CHECK:     krnl.store [[LOAD3]], [[ALLOCA]]{{.}}[[IVS]]#0, [[IVS]]#1, [[IVS]]#2{{.}} : memref<2x2x2xi64>
  // CHECK:     } else {
  // CHECK:     krnl.store [[DEFAULT_INT64]], [[ALLOCA]]{{.}}[[IVS]]#0, [[IVS]]#1, [[IVS]]#2{{.}} : memref<2x2x2xi64>
  // CHECK:     }
  // CHECK:     return [[ALLOCA]] : memref<2x2x2xi64>
}

// -----

// Test whether lowering is correct for a int64_t tensor input.
func.func private @test_rank3_category_mapper_int64_to_string(%arg0 : tensor<2x2x2xi64>) -> tensor<2x2x2x!onnx.String> {
  %0 = "onnx.CategoryMapper"(%arg0) {cats_int64s = [1, 2, 3], cats_strings = ["cat", "dog", "cow"], default_string = "none"} : (tensor<2x2x2xi64>) -> tensor<2x2x2x!onnx.String>
  "func.return"(%0) : (tensor<2x2x2x!onnx.String>) -> ()

  // CHECK-LABEL: test_rank3_category_mapper_int64_to_string
  // CHECK-DAG: [[LEN:%.+]] = arith.constant 3 : i32
  // CHECK-DAG: [[ALLOCA:%.+]] = memref.alloc() {alignment = 16 : i64} : memref<2x2x2x!krnl.string>
  // CHECK-DAG: [[G:%.+]] = "krnl.global"() {name = {{.*}}, shape = [3], value = dense<[-1, 1, 0]> : tensor<3xi32>} : () -> memref<3xi32>
  // CHECK-DAG: [[V:%.+]] = "krnl.global"() {name = {{.*}}, shape = [3], value = dense<[2, 1, 0]> : tensor<3xi32>} : () -> memref<3xi32>
  // CHECK-DAG: [[CAT_INT64s:%.+]] = "krnl.global"() {name = {{.*}}, shape = [3], value = dense<[1, 2, 3]> : tensor<3xi64>} : () -> memref<3xi64>
  // CHECK-DAG: [[CAT_STRINGS:%.+]] = "krnl.global"() {name = {{.*}}, shape = [3], value = dense<["cat", "dog", "cow"]> : tensor<3x!krnl.string>} : () -> memref<3x!krnl.string>
  // CHECK-DAG: [[DEFAULT_STRING:%.+]] = "krnl.global"() {name = {{.*}}, shape = [], value = dense<"none"> : tensor<!krnl.string>} : () -> memref<!krnl.string>
  // CHECK-DAG: [[LOOP_0:%.+]]:3 = krnl.define_loops 3
  // CHECK:     krnl.iterate([[LOOP_0]]#0, [[LOOP_0]]#1, [[LOOP_0]]#2) with ([[LOOP_0]]#0 -> [[I_0:%.+]] = 0 to 2, [[LOOP_0]]#1 -> [[I_1:%.+]] = 0 to 2, [[LOOP_0]]#2 -> [[I_2:%.+]] = 0 to 2){
  // CHECK:     [[IVS:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0]]#0, [[LOOP_0]]#1, [[LOOP_0]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
  // CHECK:     [[LOAD1:%.+]] = krnl.load %arg0{{.}}[[IVS]]#0, [[IVS]]#1, [[IVS]]#2{{.}} : memref<2x2x2xi64>
  // CHECK:     [[INDEX:%.+]] = "krnl.find_index"([[LOAD1]], [[G]], [[V]], [[LEN]]) : (i64, memref<3xi32>, memref<3xi32>, i32) -> index
  // CHECK:     [[LOAD2:%.+]] = krnl.load [[CAT_INT64s]]{{.}}[[INDEX]]{{.}} : memref<3xi64>
  // CHECK:     [[VALID:%.+]] = arith.cmpi eq, [[LOAD1]], [[LOAD2]] : i64
  // CHECK:     scf.if [[VALID]] {
  // CHECK:     [[LOAD3:%.+]] = krnl.load [[CAT_STRINGS]]{{.}}[[INDEX]]{{.}} : memref<3x!krnl.string>
  // CHECK:     krnl.store [[LOAD3]], [[ALLOCA]]{{.}}[[IVS]]#0, [[IVS]]#1, [[IVS]]#2{{.}} : memref<2x2x2x!krnl.string>
  // CHECK:     } else {
  // CHECK:     [[LOAD4:%.+]] = krnl.load [[DEFAULT_STRING]][] : memref<!krnl.string>
  // CHECK:     krnl.store [[LOAD4]], [[ALLOCA]]{{.}}[[IVS]]#0, [[IVS]]#1, [[IVS]]#2{{.}} : memref<2x2x2x!krnl.string>
  // CHECK:     }
  // CHECK:     return [[ALLOCA]] : memref<2x2x2x!krnl.string>
}
