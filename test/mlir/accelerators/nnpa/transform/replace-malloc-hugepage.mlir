// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --replace-malloc-by-hugepage-malloc %s -split-input-file | FileCheck %s

llvm.func @malloc(i64) -> !llvm.ptr

// CHECK-LABEL: llvm.func @OMHugePageMalloc(i64) -> !llvm.ptr
// CHECK-LABEL: func @test_replace_malloc
func.func @test_replace_malloc(%arg0: i64) -> !llvm.ptr {
  // CHECK: %[[RES:.*]] = llvm.call @OMHugePageMalloc(%arg0) : (i64) -> !llvm.ptr
  // CHECK-NOT: llvm.call @malloc
  %0 = llvm.call @malloc(%arg0) : (i64) -> !llvm.ptr
  return %0 : !llvm.ptr
}

// -----

llvm.func @malloc(i64) -> !llvm.ptr

// CHECK-LABEL: llvm.func @OMHugePageMalloc(i64) -> !llvm.ptr
// CHECK-LABEL: func @test_multiple_malloc
func.func @test_multiple_malloc(%arg0: i64, %arg1: i64) -> (!llvm.ptr, !llvm.ptr) {
  // CHECK: %[[RES1:.*]] = llvm.call @OMHugePageMalloc(%arg0) : (i64) -> !llvm.ptr
  // CHECK: %[[RES2:.*]] = llvm.call @OMHugePageMalloc(%arg1) : (i64) -> !llvm.ptr
  // CHECK-NOT: llvm.call @malloc
  %0 = llvm.call @malloc(%arg0) : (i64) -> !llvm.ptr
  %1 = llvm.call @malloc(%arg1) : (i64) -> !llvm.ptr
  return %0, %1 : !llvm.ptr, !llvm.ptr
}

// -----

llvm.func @malloc(i64) -> !llvm.ptr
llvm.func @free(!llvm.ptr)

// CHECK-LABEL: llvm.func @OMHugePageMalloc(i64) -> !llvm.ptr
// CHECK-LABEL: func @test_other_calls_unchanged
func.func @test_other_calls_unchanged(%arg0: i64) -> !llvm.ptr {
  // CHECK: %[[RES:.*]] = llvm.call @OMHugePageMalloc(%arg0) : (i64) -> !llvm.ptr
  // CHECK: llvm.call @free(%[[RES]]) : (!llvm.ptr) -> ()
  %0 = llvm.call @malloc(%arg0) : (i64) -> !llvm.ptr
  llvm.call @free(%0) : (!llvm.ptr) -> ()
  return %0 : !llvm.ptr
}