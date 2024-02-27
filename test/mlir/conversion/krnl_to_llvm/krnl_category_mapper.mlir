// RUN: onnx-mlir-opt -O3 --convert-krnl-to-affine --convert-krnl-to-llvm %s -split-input-file | FileCheck %s

// -----

// Test that 'krnl.find_index' can be called when the first argument is a string.
func.func private @test_find_index_str(%str: !krnl.string) -> index {
  %G = "krnl.global"() {name = "G", shape = [3], value = dense<[1,0,-3]> : tensor<3xi32>} : () -> memref<3xi32>
  %V = "krnl.global"() {name = "V", shape = [3], value = dense<[1,2,0]> : tensor<3xi32>} : () -> memref<3xi32>
  %c3 = arith.constant 3 : i32  
  %index = "krnl.find_index"(%str, %G, %V, %c3) : (!krnl.string, memref<3xi32>, memref<3xi32>, i32) -> index
  return %index : index
}

// CHECK-DAG:   llvm.func @find_index_str(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> i64
// CHECK-DAG:   llvm.mlir.global internal constant @V(dense<[1, 2, 0]> : tensor<3xi32>) {addr_space = 0 : i32, alignment = 16 : i64} : !llvm.array<3 x i32>
// CHECK-DAG:   llvm.mlir.global internal constant @G(dense<[1, 0, -3]> : tensor<3xi32>) {addr_space = 0 : i32, alignment = 16 : i64} : !llvm.array<3 x i32>

// CHECK-LABEL: @test_find_index_str(%arg0: i64) -> i64
// CHECK-DAG:   [[LEN:%.+]] = llvm.mlir.constant(3 : i32) : i32
// CHECK-DAG:   [[STR:%.+]] = llvm.inttoptr %arg0 : i64 to !llvm.ptr
// CHECK-DAG:   [[G:%.+]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:   [[V:%.+]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:       [[INDEX:%.+]] = llvm.call @find_index_str([[STR]], [[G]], [[V]], [[LEN]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> i64
// CHECK:       llvm.return [[INDEX]] : i64

// -----

// Test that 'krnl.find_index' can be called when the first argument is a int64_t.
func.func private @test_find_index_int(%val: i64) -> index {
  %G = "krnl.global"() {name = "G", shape = [3], value = dense<[1,0,-3]> : tensor<3xi32>} : () -> memref<3xi32>
  %V = "krnl.global"() {name = "V", shape = [3], value = dense<[1,2,0]> : tensor<3xi32>} : () -> memref<3xi32>
  %c3 = arith.constant 3 : i32  
  %index = "krnl.find_index"(%val, %G, %V, %c3) : (i64, memref<3xi32>, memref<3xi32>, i32) -> index
  return %index : index

// CHECK-DAG:   llvm.func @find_index_i64(i64, !llvm.ptr, !llvm.ptr, i32) -> i64
// CHECK-DAG:   llvm.mlir.global internal constant @V(dense<[1, 2, 0]> : tensor<3xi32>) {addr_space = 0 : i32, alignment = 16 : i64} : !llvm.array<3 x i32>
// CHECK-DAG:   llvm.mlir.global internal constant @G(dense<[1, 0, -3]> : tensor<3xi32>) {addr_space = 0 : i32, alignment = 16 : i64} : !llvm.array<3 x i32>

// CHECK-LABEL: llvm.func @test_find_index_int(%arg0: i64) -> i64
// CHECK-DAG:   [[LEN:%.+]] = llvm.mlir.constant(3 : i32) : i32
// CHECK-DAG:   [[G:%.+]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:   [[V:%.+]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:       [[INDEX:%.+]] = llvm.call @find_index_i64(%arg0, [[G]], [[V]], [[LEN]]) : (i64, !llvm.ptr, !llvm.ptr, i32) -> i64
// CHECK:       llvm.return [[INDEX]] : i64
}

// -----

// Test CategorMapper lowering when the input is a list of strings.
func.func private @test_category_mapper_string_to_int64(%arg0: memref<2x2x!krnl.string>) -> memref<2x2xi64> {
  %c0_i32 = arith.constant 0 : i32
  %c-1_i64 = arith.constant -1 : i64
  %c3_i32 = arith.constant 3 : i32
  %0 = memref.alloc() {alignment = 16 : i64} : memref<2x2xi64>
  %1 = "krnl.global"() {name = "G", shape = [3], value = dense<[1, 0, -3]> : tensor<3xi32>} : () -> memref<3xi32>
  %2 = "krnl.global"() {name = "V", shape = [3], value = dense<[1, 2, 0]> : tensor<3xi32>} : () -> memref<3xi32>
  %3 = "krnl.global"() {name = "cats_int64s", shape = [3], value = dense<[1, 2, 3]> : tensor<3xi64>} : () -> memref<3xi64>
  %4 = "krnl.global"() {name = "cats_strings", shape = [3], value = dense<["cat", "dog", "cow"]> : tensor<3x!krnl.string>} : () -> memref<3x!krnl.string>
  %5:2 = krnl.define_loops 2
  krnl.iterate(%5#0, %5#1) with (%5#0 -> %arg1 = 0 to 2, %5#1 -> %arg2 = 0 to 2) {
    %6:2 = krnl.get_induction_var_value(%5#0, %5#1) : (!krnl.loop, !krnl.loop) -> (index, index)
    %7 = krnl.load %arg0[%6#0, %6#1] : memref<2x2x!krnl.string>
    %8 = "krnl.find_index"(%7, %1, %2, %c3_i32) : (!krnl.string, memref<3xi32>, memref<3xi32>, i32) -> index
    %9 = krnl.load %4[%8] : memref<3x!krnl.string>
    %10 = "krnl.strlen"(%9) : (!krnl.string) -> i64
    %11 = "krnl.strncmp"(%7, %9, %10) : (!krnl.string, !krnl.string, i64) -> i32
    %12 = arith.cmpi eq, %11, %c0_i32 : i32
    scf.if %12 {
      %13 = krnl.load %3[%8] : memref<3xi64>
      krnl.store %13, %0[%6#0, %6#1] : memref<2x2xi64>
    } else {
      krnl.store %c-1_i64, %0[%6#0, %6#1] : memref<2x2xi64>
    }
  }
  return %0 : memref<2x2xi64>

  // CHECK-DAG: llvm.func @strncmp(!llvm.ptr, !llvm.ptr, i64) -> i32
  // CHECK-DAG: llvm.func @strlen(!llvm.ptr) -> i64
  // CHECK-DAG: llvm.func @find_index_str(!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> i64
  // CHECK-DAG: llvm.mlir.global internal constant @om.strArray.cats_strings("cat\00dog\00cow\00") {addr_space = 0 : i32}
  // CHECK:    llvm.mlir.global internal constant @cats_strings() {addr_space = 0 : i32, alignment = 16 : i64} : !llvm.array<3 x ptr> {
  // CHECK:    [[ARRAY:%.+]] = llvm.mlir.undef : !llvm.array<3 x ptr>
  // CHECK:    [[BASE_ADDR:%.+]] = llvm.mlir.addressof @om.strArray.cats_strings : !llvm.ptr
  // CHECK:    [[I8_BASE_ADDR:%.+]] = llvm.bitcast %1 : !llvm.ptr to !llvm.ptr
  // CHECK:    [[CAT_GEP:%.+]] = llvm.getelementptr [[I8_BASE_ADDR]][0] : (!llvm.ptr) -> !llvm.ptr, i8
  // CHECK:    [[CAT_INS_VAL:%.+]] = llvm.insertvalue [[CAT_GEP]], [[ARRAY]][0] : !llvm.array<3 x ptr> 
  // CHECK:    [[DOG_GEP:%.+]] = llvm.getelementptr [[I8_BASE_ADDR]][4] : (!llvm.ptr) -> !llvm.ptr, i8
  // CHECK:    [[DOG_INS_VAL:%.+]] = llvm.insertvalue [[DOG_GEP]], [[CAT_INS_VAL]][1] : !llvm.array<3 x ptr> 
  // CHECK:    [[COW_GEP:%.+]] = llvm.getelementptr [[I8_BASE_ADDR]][8] : (!llvm.ptr) -> !llvm.ptr, i8
  // CHECK:    [[COW_INS_VAL:%.+]] = llvm.insertvalue [[COW_GEP]], [[DOG_INS_VAL]][2] : !llvm.array<3 x ptr> 
  // CHECK:    llvm.return [[COW_INS_VAL]] : !llvm.array<3 x ptr>
  // CHECK:  }
  // CHECK-DAG: llvm.mlir.global internal constant @cats_int64s{{.*}}(dense<[1, 2, 3]> : tensor<3xi64>) {addr_space = 0 : i32, alignment = 16 : i64} : !llvm.array<3 x i64>
  // CHECK-DAG: llvm.mlir.global internal constant @V{{.*}}(dense<[1, 2, 0]> : tensor<3xi32>) {addr_space = 0 : i32, alignment = 16 : i64} : !llvm.array<3 x i32>
  // CHECK-DAG: llvm.mlir.global internal constant @G{{.*}}(dense<[1, 0, -3]> : tensor<3xi32>) {addr_space = 0 : i32, alignment = 16 : i64} : !llvm.array<3 x i32>

  // CHECK-LABEL: @test_category_mapper_string_to_int64(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32  
  // CHECK-DAG:   [[DEF_VAL:%.+]] = llvm.mlir.constant(-1 : i64) : i64  
  // CHECK-DAG:   [[LEN:%.+]] = llvm.mlir.constant(3 : i32) : i32

  /// Find the index of the input string:

  // CHECK:       [[LOAD:%.+]] = llvm.load {{.*}} : !llvm.ptr -> i64
  // CHECK:       [[STR:%.+]] = llvm.inttoptr [[LOAD]] : i64 to !llvm.ptr
  // CHECK:       [[G:%.+]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK:       [[V:%.+]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK:       [[INDEX:%.+]] = llvm.call @find_index_str([[STR]], [[G]], [[V]], [[LEN]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32) -> i64

  /// Determine whether the index is valid:
  // CHECK:       [[STR1:%.+]]  = llvm.inttoptr {{.*}} : i64 to !llvm.ptr
  // CHECK-DAG:   [[STRLEN:%.+]] = llvm.call @strlen([[STR1]]) : (!llvm.ptr) -> i64
  // CHECK-DAG:   [[STR2:%.+]] = llvm.inttoptr {{.*}} : i64 to !llvm.ptr
  // CHECK-DAG:   [[STR3:%.+]] = llvm.inttoptr {{.*}} : i64 to !llvm.ptr
  // CHECK:       [[STREQ:%.+]] = llvm.call @strncmp([[STR2]], [[STR3]], [[STRLEN]]) : (!llvm.ptr, !llvm.ptr, i64) -> i32

  /// Store the index if valid, otherwise store the default value:
  // CHECK-NEXT:  [[IS_EQUAL:%.+]] = llvm.icmp "eq" [[STREQ]], [[C0]] : i32  
  // CHECK-NEXT:  llvm.cond_br [[IS_EQUAL]], [[LAB_TRUE:\^.+]], [[LAB_FALSE:\^.+]]
  // CHECK:       [[LAB_TRUE]]:
  // CHECK:       [[GEP1:%.+]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr  
  // CHECK:       [[LOAD1:%.+]] = llvm.load [[GEP1]] : !llvm.ptr  
  // CHECK:       llvm.store [[LOAD1]], {{.*}} : i64, !llvm.ptr 
  // CHECK-NEXT:  llvm.br [[IF_END:\^.+]]
  // CHECK:       [[LAB_FALSE]]:  
  // CHECK:       llvm.store [[DEF_VAL]], {{.*}} : i64, !llvm.ptr
  // CHECK-NEXT:  llvm.br [[IF_END]]
  // CHECK:       [[IF_END]]:
}

// -----

// Test CategorMapper lowering when the input is a list of int64_t.
func.func private @test_category_mapper_int64_to_string(%arg0: memref<2x2xi64>) -> memref<2x2x!krnl.string> {
  %c3_i32 = arith.constant 3 : i32
  %0 = memref.alloc() {alignment = 16 : i64} : memref<2x2x!krnl.string>
  %1 = "krnl.global"() {name = "G", shape = [3], value = dense<[-1, 1, 0]> : tensor<3xi32>} : () -> memref<3xi32>
  %2 = "krnl.global"() {name = "V", shape = [3], value = dense<[2, 1, 0]> : tensor<3xi32>} : () -> memref<3xi32>
  %3 = "krnl.global"() {name = "cats_int64s", shape = [3], value = dense<[1, 2, 3]> : tensor<3xi64>} : () -> memref<3xi64>
  %4 = "krnl.global"() {name = "cats_strings", shape = [3], value = dense<["cat", "dog", "cow"]> : tensor<3x!krnl.string>} : () -> memref<3x!krnl.string>
  %5 = "krnl.global"() {name = "default_string", shape = [], value = dense<"none"> : tensor<!krnl.string>} : () -> memref<!krnl.string>
  %6:2 = krnl.define_loops 2
  krnl.iterate(%6#0, %6#1) with (%6#0 -> %arg1 = 0 to 2, %6#1 -> %arg2 = 0 to 2) {
    %7:2 = krnl.get_induction_var_value(%6#0, %6#1) : (!krnl.loop, !krnl.loop) -> (index, index)
    %8 = krnl.load %arg0[%7#0, %7#1] : memref<2x2xi64>
    %9 = "krnl.find_index"(%8, %1, %2, %c3_i32) : (i64, memref<3xi32>, memref<3xi32>, i32) -> index
    %10 = krnl.load %3[%9] : memref<3xi64>
    %11 = arith.cmpi eq, %8, %10 : i64
    scf.if %11 {
      %12 = krnl.load %4[%9] : memref<3x!krnl.string>
      krnl.store %12, %0[%7#0, %7#1] : memref<2x2x!krnl.string>
    } else {
      %12 = krnl.load %5[] : memref<!krnl.string>
      krnl.store %12, %0[%7#0, %7#1] : memref<2x2x!krnl.string>
    }
  }
  return %0 : memref<2x2x!krnl.string>

  // CHECK-DAG:  llvm.func @find_index_i64(i64, !llvm.ptr, !llvm.ptr, i32) -> i64
  // CHECK-DAG:  llvm.mlir.global internal constant @om.strArray.default_string("none\00") {addr_space = 0 : i32}
  // CHECK-DAG: llvm.mlir.global internal constant @om.strArray.cats_strings("cat\00dog\00cow\00") {addr_space = 0 : i32}
  // CHECK:    llvm.mlir.global internal constant @cats_strings() {addr_space = 0 : i32, alignment = 16 : i64} : !llvm.array<3 x ptr> {
  // CHECK:    [[ARRAY:%.+]] = llvm.mlir.undef : !llvm.array<3 x ptr>
  // CHECK:    [[BASE_ADDR:%.+]] = llvm.mlir.addressof @om.strArray.cats_strings : !llvm.ptr
  // CHECK:    [[I8_BASE_ADDR:%.+]] = llvm.bitcast %1 : !llvm.ptr to !llvm.ptr
  // CHECK:    [[CAT_GEP:%.+]] = llvm.getelementptr [[I8_BASE_ADDR]][0] : (!llvm.ptr) -> !llvm.ptr, i8
  // CHECK:    [[CAT_INS_VAL:%.+]] = llvm.insertvalue [[CAT_GEP]], [[ARRAY]][0] : !llvm.array<3 x ptr> 
  // CHECK:    [[DOG_GEP:%.+]] = llvm.getelementptr [[I8_BASE_ADDR]][4] : (!llvm.ptr) -> !llvm.ptr, i8
  // CHECK:    [[DOG_INS_VAL:%.+]] = llvm.insertvalue [[DOG_GEP]], [[CAT_INS_VAL]][1] : !llvm.array<3 x ptr> 
  // CHECK:    [[COW_GEP:%.+]] = llvm.getelementptr [[I8_BASE_ADDR]][8] : (!llvm.ptr) -> !llvm.ptr, i8
  // CHECK:    [[COW_INS_VAL:%.+]] = llvm.insertvalue [[COW_GEP]], [[DOG_INS_VAL]][2] : !llvm.array<3 x ptr> 
  // CHECK:    llvm.return [[COW_INS_VAL]] : !llvm.array<3 x ptr>
  // CHECK:  }
  // CHECK-DAG:  llvm.mlir.global internal constant @cats_int64s{{.*}}(dense<[1, 2, 3]> : tensor<3xi64>) {addr_space = 0 : i32, alignment = 16 : i64} : !llvm.array<3 x i64>
  // CHECK-DAG:  llvm.mlir.global internal constant @V{{.*}}(dense<[2, 1, 0]> : tensor<3xi32>) {addr_space = 0 : i32, alignment = 16 : i64} : !llvm.array<3 x i32>
  // CHECK-DAG:  llvm.mlir.global internal constant @G{{.*}}(dense<[-1, 1, 0]> : tensor<3xi32>) {addr_space = 0 : i32, alignment = 16 : i64} : !llvm.array<3 x i32>

  // CHECK-LABEL: @test_category_mapper_int64_to_string(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
  // CHECK-DAG:   [[LEN:%.+]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK:       [[MALLOC:%.+]] = llvm.call @malloc({{.*}}) : (i64) -> !llvm.ptr
  // CHECK:       [[UNDEF:%.+]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
  // CHECK:       [[EV_1:%.+]] = llvm.insertvalue {{.*}}, [[UNDEF]][0]
  // CHECK:       [[EV_2:%.+]] = llvm.insertvalue {{.*}}, [[EV_1]][1]
  // CHECK:       [[C0:%.+]] = llvm.mlir.constant(0 : index) : i64
  // CHECK:       [[DEF_VAL:%.+]] = llvm.insertvalue [[C0]], [[EV_2]][2] : !llvm.struct<(ptr, ptr, i64)>

  /// Find the index of the input string:
  // CHECK-DAG:   [[INPUT:%.+]] = llvm.load {{.*}} : !llvm.ptr  
  // CHECK-DAG:   [[G:%.+]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK-DAG:   [[V:%.+]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK:       [[INDEX:%.+]] = llvm.call @find_index_i64([[INPUT]], [[G]], [[V]], [[LEN]]) : (i64, !llvm.ptr, !llvm.ptr, i32) -> i64

  /// Determine whether the index is valid:
  // CHECK:       [[EV1:%.+]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
  // CHECK-DAG:   [[GEP1:%.+]] = llvm.getelementptr [[EV1]]{{.*}}[[INDEX]]{{.*}} : (!llvm.ptr, i64) -> !llvm.ptr
  // CHECK-DAG:   [[INDEX1:%.+]] = llvm.load [[GEP1]] : !llvm.ptr

  /// Store the index if valid, otherwise store the default value:
  // CHECK-NEXT:  [[IS_EQUAL:%.+]] = llvm.icmp "eq" {{.*}}, [[INDEX1]] : i64  
  // CHECK-NEXT:  llvm.cond_br [[IS_EQUAL]], [[LAB_TRUE:\^.+]], [[LAB_FALSE:\^.+]]
  // CHECK:       [[LAB_TRUE]]:
  // CHECK:       [[GEP1:%.+]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr 
  // CHECK:       [[LOAD1:%.+]] = llvm.load [[GEP1]] : !llvm.ptr 
  // CHECK:       llvm.store [[LOAD1]], {{.*}} : i64, !llvm.ptr 
  // CHECK-NEXT:  llvm.br [[IF_END:\^.+]]
  // CHECK:       [[LAB_FALSE]]:  
  // CHECK:       [[EV2:%.+]] = llvm.extractvalue [[DEF_VAL]][1] : !llvm.struct<(ptr, ptr, i64)>
  // CHECK:       [[LOAD_EXT_VAL:%.+]] = llvm.load [[EV2]] : !llvm.ptr
  // CHECK:       llvm.store [[LOAD_EXT_VAL]], {{.*}} : i64, !llvm.ptr
  // CHECK-NEXT:  llvm.br [[IF_END]]
  // CHECK:       [[IF_END]]:
}

// -----

// Test that 'krnl.global' with 129+ strings can be handled without errors.
func.func private @test_krnl_global_with_129_elements() -> memref<129x!krnl.string> {
  %4 = "krnl.global"() {name = "cats_strings", shape = [129], value = dense<["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "110", "111", "112", "113", "114", "115", "116", "117", "118", "119", "120", "121", "122", "123", "124", "125", "126", "127", "128", "129"]> : tensor<129x!krnl.string>} : () -> memref<129x!krnl.string>
  return %4 : memref<129x!krnl.string>

  // CHECK:         llvm.func @test_krnl_global_with_129_elements() -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> attributes {llvm.emit_c_interface, sym_visibility = "private"} {
  // CHECK:           [[VAR_0_1_:%.+]] = llvm.mlir.addressof @cats_strings : !llvm.ptr
  // CHECK-DAG:       [[VAR_1_1_:%.+]] = llvm.bitcast [[VAR_0_1_]] : !llvm.ptr to !llvm.ptr
  // CHECK-DAG:       [[VAR_2_1_:%.+]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK:           [[VAR_3_1_:%.+]] = llvm.insertvalue [[VAR_1_1_]], [[VAR_2_1_]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK-DAG:       [[VAR_4_1_:%.+]] = llvm.insertvalue [[VAR_1_1_]], [[VAR_3_1_]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK-DAG:       [[VAR_5_1_:%.+]] = llvm.mlir.constant(0 : index) : i64
  // CHECK-NOT: separator of consecutive DAGs
  // CHECK-DAG:       [[VAR_6_1_:%.+]] = llvm.insertvalue [[VAR_5_1_]], [[VAR_4_1_]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK-DAG:       [[VAR_7_1_:%.+]] = llvm.mlir.constant(129 : index) : i64
  // CHECK-NOT: separator of consecutive DAGs
  // CHECK-DAG:       [[VAR_8_1_:%.+]] = llvm.insertvalue [[VAR_7_1_]], [[VAR_6_1_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK-DAG:       [[VAR_9_1_:%.+]] = llvm.mlir.constant(1 : index) : i64
  // CHECK:           [[VAR_10_1_:%.+]] = llvm.insertvalue [[VAR_9_1_]], [[VAR_8_1_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK:           llvm.return [[VAR_10_1_]] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>

}
