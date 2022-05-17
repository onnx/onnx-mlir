// RUN: onnx-mlir-opt -O3 --convert-krnl-to-affine --convert-krnl-to-llvm %s -split-input-file | FileCheck %s

// -----

// Test that 'krnl.find_index' can be called when the first argument is a string.
func private @test_find_index_str(%str: !krnl.string) -> index {
  %G = "krnl.global"() {name = "G", shape = [3], value = dense<[1,0,-3]> : tensor<3xi32>} : () -> memref<3xi32>
  %V = "krnl.global"() {name = "V", shape = [3], value = dense<[1,2,0]> : tensor<3xi32>} : () -> memref<3xi32>
  %c3 = arith.constant 3 : i32  
  %index = "krnl.find_index"(%str, %G, %V, %c3) : (!krnl.string, memref<3xi32>, memref<3xi32>, i32) -> index
  return %index : index
}

// CHECK-DAG:   llvm.func @find_index_str(!llvm.ptr<i8>, !llvm.ptr<i32>, !llvm.ptr<i32>, i32) -> i64
// CHECK-DAG:   llvm.mlir.global internal constant @V(dense<[1, 2, 0]> : tensor<3xi32>) {alignment = 16 : i64} : !llvm.array<3 x i32>
// CHECK-DAG:   llvm.mlir.global internal constant @G(dense<[1, 0, -3]> : tensor<3xi32>) {alignment = 16 : i64} : !llvm.array<3 x i32>

// CHECK-LABEL: @test_find_index_str(%arg0: i64) -> i64
// CHECK-DAG:   [[LEN:%.+]] = llvm.mlir.constant(3 : i32) : i32
// CHECK-DAG:   [[STR:%.+]] = llvm.inttoptr %arg0 : i64 to !llvm.ptr<i8>
// CHECK-DAG:   [[G:%.+]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:   [[V:%.+]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:       [[INDEX:%.+]] = llvm.call @find_index_str([[STR]], [[G]], [[V]], [[LEN]]) : (!llvm.ptr<i8>, !llvm.ptr<i32>, !llvm.ptr<i32>, i32) -> i64
// CHECK:       llvm.return [[INDEX]] : i64

// -----

// Test that 'krnl.find_index' can be called when the first argument is a int64_t.
func private @test_find_index_int(%val: i64) -> index {
  %G = "krnl.global"() {name = "G", shape = [3], value = dense<[1,0,-3]> : tensor<3xi32>} : () -> memref<3xi32>
  %V = "krnl.global"() {name = "V", shape = [3], value = dense<[1,2,0]> : tensor<3xi32>} : () -> memref<3xi32>
  %c3 = arith.constant 3 : i32  
  %index = "krnl.find_index"(%val, %G, %V, %c3) : (i64, memref<3xi32>, memref<3xi32>, i32) -> index
  return %index : index

// CHECK-DAG:   llvm.func @find_index_i64(i64, !llvm.ptr<i32>, !llvm.ptr<i32>, i32) -> i64
// CHECK-DAG:   llvm.mlir.global internal constant @V(dense<[1, 2, 0]> : tensor<3xi32>) {alignment = 16 : i64} : !llvm.array<3 x i32>
// CHECK-DAG:   llvm.mlir.global internal constant @G(dense<[1, 0, -3]> : tensor<3xi32>) {alignment = 16 : i64} : !llvm.array<3 x i32>

// CHECK-LABEL: llvm.func @test_find_index_int(%arg0: i64) -> i64
// CHECK-DAG:   [[LEN:%.+]] = llvm.mlir.constant(3 : i32) : i32
// CHECK-DAG:   [[G:%.+]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:   [[V:%.+]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:       [[INDEX:%.+]] = llvm.call @find_index_i64(%arg0, [[G]], [[V]], [[LEN]]) : (i64, !llvm.ptr<i32>, !llvm.ptr<i32>, i32) -> i64
// CHECK:       llvm.return [[INDEX]] : i64
}

// -----

// Test CategorMapper lowering when the input is a list of strings.
func private @test_category_mapper_string_to_int64(%arg0: memref<2x2x!krnl.string>) -> memref<2x2xi64> {
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

  // CHECK-DAG: llvm.func @strncmp(!llvm.ptr<i8>, !llvm.ptr<i8>, i64) -> i32
  // CHECK-DAG: llvm.func @strlen(!llvm.ptr<i8>) -> i64
  // CHECK-DAG: llvm.func @find_index_str(!llvm.ptr<i8>, !llvm.ptr<i32>, !llvm.ptr<i32>, i32) -> i64
  // CHECK-DAG: llvm.mlir.global internal constant @cat("cat")
  // CHECK-DAG: llvm.mlir.global internal constant @dog("dog")
  // CHECK-DAG: llvm.mlir.global internal constant @cow("cow")    
  // CHECK:     llvm.mlir.global internal constant @cats_strings{{.*}}() {alignment = 16 : i64} : !llvm.array<3 x ptr<i8>> { 
  // CHECK:       [[ARRAY:%.+]] = llvm.mlir.undef : !llvm.array<3 x ptr<i8>>
  // CHECK:       [[CAT_ADDR:%.+]] = llvm.mlir.addressof @cat : !llvm.ptr<array<3 x i8>>
  // CHECK:       [[ZERO:%.+]] = llvm.mlir.constant(0 : index) : i64
  // CHECK:       [[CAT_GEP:%.+]] = llvm.getelementptr [[CAT_ADDR]]{{.*}}[[ZERO]], [[ZERO]]{{.*}} : (!llvm.ptr<array<3 x i8>>, i64, i64) -> !llvm.ptr<i8>
  // CHECK:       [[CAT_INS_VAL:%.+]] = llvm.insertvalue [[CAT_GEP]], [[ARRAY]][0 : index] : !llvm.array<3 x ptr<i8>>
  // CHECK:       [[DOG_ADDR:%.+]] = llvm.mlir.addressof @dog : !llvm.ptr<array<3 x i8>>
  // CHECK:       [[ZERO:%.+]] = llvm.mlir.constant(0 : index) : i64  
  // CHECK:       [[DOG_GEP:%.+]] = llvm.getelementptr [[DOG_ADDR]]{{.*}}[[ZERO]], [[ZERO]]{{.*}} : (!llvm.ptr<array<3 x i8>>, i64, i64) -> !llvm.ptr<i8>
  // CHECK:       [[DOG_INS_VAL:%.+]] = llvm.insertvalue [[DOG_GEP]], [[CAT_INS_VAL]][1 : index] : !llvm.array<3 x ptr<i8>>
  // CHECK:       [[COW_ADDR:%.+]] = llvm.mlir.addressof @cow : !llvm.ptr<array<3 x i8>>
  // CHECK:       [[ZERO:%.+]] = llvm.mlir.constant(0 : index) : i64    
  // CHECK:       [[COW_GEP:%.+]] = llvm.getelementptr [[COW_ADDR]]{{.*}}[[ZERO]], [[ZERO]]{{.*}} : (!llvm.ptr<array<3 x i8>>, i64, i64) -> !llvm.ptr<i8>
  // CHECK:       [[COW_INS_VAL:%.+]] = llvm.insertvalue [[COW_GEP]], [[DOG_INS_VAL]][2 : index] : !llvm.array<3 x ptr<i8>>
  // CHECK:       llvm.return [[COW_INS_VAL]] : !llvm.array<3 x ptr<i8>>
  // CHECK:     }
  // CHECK-DAG: llvm.mlir.global internal constant @cats_int64s{{.*}}(dense<[1, 2, 3]> : tensor<3xi64>) {alignment = 16 : i64} : !llvm.array<3 x i64>
  // CHECK-DAG: llvm.mlir.global internal constant @V{{.*}}(dense<[1, 2, 0]> : tensor<3xi32>) {alignment = 16 : i64} : !llvm.array<3 x i32>
  // CHECK-DAG: llvm.mlir.global internal constant @G{{.*}}(dense<[1, 0, -3]> : tensor<3xi32>) {alignment = 16 : i64} : !llvm.array<3 x i32>

  // CHECK-LABEL: @test_category_mapper_string_to_int64(%arg0: !llvm.ptr<i64>, %arg1: !llvm.ptr<i64>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) -> !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK-DAG:   [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32  
  // CHECK-DAG:   [[DEF_VAL:%.+]] = llvm.mlir.constant(-1 : i64) : i64  
  // CHECK-DAG:   [[LEN:%.+]] = llvm.mlir.constant(3 : i32) : i32

  /// Find the index of the input string:
  // CHECK-DAG:   [[STR:%.+]] = llvm.inttoptr {{.*}} : i64 to !llvm.ptr<i8>
  // CHECK-DAG:   [[G:%.+]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK-DAG:   [[V:%.+]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK:       [[INDEX:%.+]] = llvm.call @find_index_str([[STR]], [[G]], [[V]], [[LEN]]) : (!llvm.ptr<i8>, !llvm.ptr<i32>, !llvm.ptr<i32>, i32) -> i64

  /// Determine whether the index is valid:
  // CHECK:       [[STR1:%.+]]  = llvm.inttoptr {{.*}} : i64 to !llvm.ptr<i8>
  // CHECK-DAG:   [[STRLEN:%.+]] = llvm.call @strlen([[STR1]]) : (!llvm.ptr<i8>) -> i64
  // CHECK-DAG:   [[STR2:%.+]] = llvm.inttoptr {{.*}} : i64 to !llvm.ptr<i8>
  // CHECK-DAG:   [[STR3:%.+]] = llvm.inttoptr {{.*}} : i64 to !llvm.ptr<i8>
  // CHECK:       [[STREQ:%.+]] = llvm.call @strncmp([[STR2]], [[STR3]], [[STRLEN]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64) -> i32

  /// Store the index if valid, otherwise store the default value:
  // CHECK-NEXT:  [[IS_EQUAL:%.+]] = llvm.icmp "eq" [[STREQ]], [[C0]] : i32  
  // CHECK-NEXT:  llvm.cond_br [[IS_EQUAL]], [[LAB_TRUE:\^.+]], [[LAB_FALSE:\^.+]]
  // CHECK:       [[LAB_TRUE]]:
  // CHECK:       [[GEP1:%.+]] = llvm.getelementptr {{.*}} : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>  
  // CHECK:       [[LOAD1:%.+]] = llvm.load [[GEP1]] : !llvm.ptr<i64>  
  // CHECK:       llvm.store [[LOAD1]], {{.*}} : !llvm.ptr<i64>
  // CHECK-NEXT:  llvm.br [[IF_END:\^.+]]
  // CHECK:       [[LAB_FALSE]]:  
  // CHECK:       llvm.store [[DEF_VAL]], {{.*}} : !llvm.ptr<i64>
  // CHECK-NEXT:  llvm.br [[IF_END]]
  // CHECK:       [[IF_END]]:
}

// -----

// Test CategorMapper lowering when the input is a list of int64_t.
func private @test_category_mapper_int64_to_string(%arg0: memref<2x2xi64>) -> memref<2x2x!krnl.string> {
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

  // CHECK-DAG:  llvm.func @find_index_i64(i64, !llvm.ptr<i32>, !llvm.ptr<i32>, i32) -> i64
  // CHECK-DAG:  llvm.mlir.global internal constant @none("none")
  // CHECK-DAG:  llvm.mlir.global internal constant @cat("cat")
  // CHECK-DAG:  llvm.mlir.global internal constant @dog("dog")
  // CHECK-DAG:  llvm.mlir.global internal constant @cow("cow")    
  // CHECK:      llvm.mlir.global internal constant @cats_strings{{.*}}() {alignment = 16 : i64} : !llvm.array<3 x ptr<i8>> { 
  // CHECK:        [[ARRAY:%.+]] = llvm.mlir.undef : !llvm.array<3 x ptr<i8>>
  // CHECK:        [[CAT_ADDR:%.+]] = llvm.mlir.addressof @cat : !llvm.ptr<array<3 x i8>>
  // CHECK:        [[ZERO:%.+]] = llvm.mlir.constant(0 : index) : i64
  // CHECK:        [[CAT_GEP:%.+]] = llvm.getelementptr [[CAT_ADDR]]{{.*}}[[ZERO]], [[ZERO]]{{.*}} : (!llvm.ptr<array<3 x i8>>, i64, i64) -> !llvm.ptr<i8>
  // CHECK:        [[CAT_INS_VAL:%.+]] = llvm.insertvalue [[CAT_GEP]], [[ARRAY]][0 : index] : !llvm.array<3 x ptr<i8>>
  // CHECK:        [[DOG_ADDR:%.+]] = llvm.mlir.addressof @dog : !llvm.ptr<array<3 x i8>>
  // CHECK:        [[ZERO:%.+]] = llvm.mlir.constant(0 : index) : i64  
  // CHECK:        [[DOG_GEP:%.+]] = llvm.getelementptr [[DOG_ADDR]]{{.*}}[[ZERO]], [[ZERO]]{{.*}} : (!llvm.ptr<array<3 x i8>>, i64, i64) -> !llvm.ptr<i8>
  // CHECK:        [[DOG_INS_VAL:%.+]] = llvm.insertvalue [[DOG_GEP]], [[CAT_INS_VAL]][1 : index] : !llvm.array<3 x ptr<i8>>
  // CHECK:        [[COW_ADDR:%.+]] = llvm.mlir.addressof @cow : !llvm.ptr<array<3 x i8>>
  // CHECK:        [[ZERO:%.+]] = llvm.mlir.constant(0 : index) : i64    
  // CHECK:        [[COW_GEP:%.+]] = llvm.getelementptr [[COW_ADDR]]{{.*}}[[ZERO]], [[ZERO]]{{.*}} : (!llvm.ptr<array<3 x i8>>, i64, i64) -> !llvm.ptr<i8>
  // CHECK:        [[COW_INS_VAL:%.+]] = llvm.insertvalue [[COW_GEP]], [[DOG_INS_VAL]][2 : index] : !llvm.array<3 x ptr<i8>>
  // CHECK:        llvm.return [[COW_INS_VAL]] : !llvm.array<3 x ptr<i8>>
  // CHECK:      }
  // CHECK-DAG:  llvm.mlir.global internal constant @cats_int64s{{.*}}(dense<[1, 2, 3]> : tensor<3xi64>) {alignment = 16 : i64} : !llvm.array<3 x i64>
  // CHECK-DAG:  llvm.mlir.global internal constant @V{{.*}}(dense<[2, 1, 0]> : tensor<3xi32>) {alignment = 16 : i64} : !llvm.array<3 x i32>
  // CHECK-DAG:  llvm.mlir.global internal constant @G{{.*}}(dense<[-1, 1, 0]> : tensor<3xi32>) {alignment = 16 : i64} : !llvm.array<3 x i32>

  // CHECK-LABEL: @test_category_mapper_int64_to_string(%arg0: !llvm.ptr<i64>, %arg1: !llvm.ptr<i64>, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) -> !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<2 x i64>, array<2 x i64>)> 
  // CHECK-DAG:   [[LEN:%.+]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK:       [[MALLOC:%.+]] = llvm.call @malloc({{.*}}) : (i64) -> !llvm.ptr<i8>
  // CHECK:       [[UNDEF:%.+]] = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64)>
  // CHECK:       [[EV_1:%.+]] = llvm.insertvalue {{.*}}, [[UNDEF]][0]
  // CHECK:       [[EV_2:%.+]] = llvm.insertvalue {{.*}}, [[EV_1]][1]
  // CHECK:       [[C0:%.+]] = llvm.mlir.constant(0 : index) : i64
  // CHECK:       [[DEF_VAL:%.+]] = llvm.insertvalue [[C0]], [[EV_2]][2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64)>

  /// Find the index of the input string:
  // CHECK-DAG:   [[INPUT:%.+]] = llvm.load {{.*}} : !llvm.ptr<i64>  
  // CHECK-DAG:   [[G:%.+]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK-DAG:   [[V:%.+]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK:       [[INDEX:%.+]] = llvm.call @find_index_i64([[INPUT]], [[G]], [[V]], [[LEN]]) : (i64, !llvm.ptr<i32>, !llvm.ptr<i32>, i32) -> i64

  /// Determine whether the index is valid:
  // CHECK:       [[EV1:%.+]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)> 
  // CHECK-DAG:   [[GEP1:%.+]] = llvm.getelementptr [[EV1]]{{.*}}[[INDEX]]{{.*}} : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
  // CHECK-DAG:   [[INDEX1:%.+]] = llvm.load [[GEP1]] : !llvm.ptr<i64>

  /// Store the index if valid, otherwise store the default value:
  // CHECK-NEXT:  [[IS_EQUAL:%.+]] = llvm.icmp "eq" {{.*}}, [[INDEX1]] : i64  
  // CHECK-NEXT:  llvm.cond_br [[IS_EQUAL]], [[LAB_TRUE:\^.+]], [[LAB_FALSE:\^.+]]
  // CHECK:       [[LAB_TRUE]]:
  // CHECK:       [[GEP1:%.+]] = llvm.getelementptr {{.*}} : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64> 
  // CHECK:       [[LOAD1:%.+]] = llvm.load [[GEP1]] : !llvm.ptr<i64> 
  // CHECK:       llvm.store [[LOAD1]], {{.*}} : !llvm.ptr<i64>
  // CHECK-NEXT:  llvm.br [[IF_END:\^.+]]
  // CHECK:       [[LAB_FALSE]]:  
  // CHECK:       [[EV2:%.+]] = llvm.extractvalue [[DEF_VAL]][1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64)>
  // CHECK:       [[LOAD_EXT_VAL:%.+]] = llvm.load [[EV2]] : !llvm.ptr<i64>
  // CHECK:       llvm.store [[LOAD_EXT_VAL]], {{.*}} : !llvm.ptr<i64>
  // CHECK-NEXT:  llvm.br [[IF_END]]
  // CHECK:       [[IF_END]]:
}

