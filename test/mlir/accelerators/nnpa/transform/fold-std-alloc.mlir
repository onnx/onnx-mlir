// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --fold-std-alloc %s -split-input-file | FileCheck %s

// -----

func.func @should_fold() -> memref<3xi64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  %c7 = arith.constant 7 : i64
  %c8 = arith.constant 8 : i64
  %c9 = arith.constant 9 : i64
 
  %0 = memref.alloc() : memref<3xi64>
  memref.store %c7, %0[%c0] : memref<3xi64>
  affine.store %c8, %0[%c1] : memref<3xi64>
  krnl.store %c9, %0[%c2] : memref<3xi64>
  return %0: memref<3xi64>

  // CHECK-LABEL: should_fold
  // CHECK: "krnl.global"() {name = "constant_fold_std_alloc_0", shape = [3], value = dense<[7, 8, 9]> : tensor<3xi64>} : () -> memref<3xi64>
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: krnl.store 
}

// -----

func.func @should_not_fold_not_constant_value(%arg0 : memref<1xi64>) -> memref<1xi64> {
  %c0 = arith.constant 0 : index
  %1 = krnl.load %arg0[%c0] : memref<1xi64>

  %0 = memref.alloc() : memref<1xi64>
  krnl.store %1, %0[%c0] : memref<1xi64>
  return %0: memref<1xi64>

  // CHECK-LABEL: should_not_fold_not_constant_value
  // CHECK: memref.alloc
  // CHECK: krnl.store 
}

// -----

func.func @should_not_fold_not_constant_i64(%arg0 : memref<1xindex>) -> memref<1xi64> {
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %1 = krnl.load %arg0[%c0] : memref<1xindex>

  %0 = memref.alloc() : memref<1xi64>
  krnl.store %c0_i64, %0[%1] : memref<1xi64>
  return %0: memref<1xi64>

  // CHECK-LABEL: should_not_fold_not_constant_i64
  // CHECK: memref.alloc
  // CHECK: krnl.store 
}

// -----

func.func @shoud_not_fold_different_blocks() -> memref<3xi64> {
  %c0 = arith.constant 0 : i64

  %0 = memref.alloc() : memref<3xi64>
  affine.for %i = 0 to 3 {
    krnl.store %c0, %0[%i] : memref<3xi64>
  }
  return %0: memref<3xi64>

  // CHECK-LABEL: shoud_not_fold_different_blocks 
  // CHECK: memref.alloc
  // CHECK: affine.for 
  // CHECK: krnl.store 
}

// -----

func.func @should_not_fold_number_of_stores_mismatch() -> memref<3xi64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %c7 = arith.constant 7 : i64
  %c8 = arith.constant 8 : i64
 
  %0 = memref.alloc() : memref<3xi64>
  krnl.store %c7, %0[%c0] : memref<3xi64>
  krnl.store %c8, %0[%c1] : memref<3xi64>
  return %0: memref<3xi64>

  // CHECK-LABEL: should_not_fold_number_of_stores_mismatch 
  // CHECK: memref.alloc
  // CHECK: krnl.store 
  // CHECK: krnl.store 
}

// -----

func.func @should_not_fold_not_int_type() -> memref<1xf32> {
  %c0 = arith.constant 0 : index
  %c1_f32 = arith.constant 1. : f32
 
  %0 = memref.alloc() : memref<1xf32>
  krnl.store %c1_f32, %0[%c0] : memref<1xf32>
  return %0: memref<1xf32>

  // CHECK-LABEL: should_not_fold_not_int_type
  // CHECK: memref.alloc
  // CHECK: krnl.store 
}
