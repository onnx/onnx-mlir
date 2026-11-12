// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --replace-malloc-by-hugepage-malloc %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --replace-malloc-by-hugepage-malloc="enable-memory-cache=false" %s -split-input-file | FileCheck %s --check-prefix=DISABLED

// A temporary of the model: the malloc result is only used to compute the
// aligned pointer and is then freed. The pair goes to the memory cache.

llvm.func @malloc(i64) -> !llvm.ptr
llvm.func @free(!llvm.ptr)

// CHECK-DAG: llvm.func @OMCacheAlloc(i64) -> !llvm.ptr
// CHECK-DAG: llvm.func @OMCacheFree(!llvm.ptr)
// CHECK-LABEL: llvm.func @test_temporary
llvm.func @test_temporary(%size: i64, %align: i64, %val: f32) {
  // CHECK: %[[P:.*]] = llvm.call @OMCacheAlloc(%arg0) : (i64) -> !llvm.ptr
  // CHECK: llvm.ptrtoint %[[P]]
  // CHECK: llvm.call @OMCacheFree(%[[P]]) : (!llvm.ptr) -> ()
  // CHECK-NOT: llvm.call @malloc
  // CHECK-NOT: llvm.call @free
  // DISABLED-LABEL: llvm.func @test_temporary
  // DISABLED: %[[P:.*]] = llvm.call @OMHugePageMalloc(%arg0) : (i64) -> !llvm.ptr
  // DISABLED: llvm.call @free(%[[P]]) : (!llvm.ptr) -> ()
  %0 = llvm.call @malloc(%size) : (i64) -> !llvm.ptr
  %1 = llvm.ptrtoint %0 : !llvm.ptr to i64
  %2 = llvm.add %1, %align : i64
  %3 = llvm.inttoptr %2 : i64 to !llvm.ptr
  llvm.store %val, %3 : f32, !llvm.ptr
  llvm.call @free(%0) : (!llvm.ptr) -> ()
  llvm.return
}

// -----

// A model output: the malloc result is put into the returned memref
// descriptor and never freed. It keeps using OMHugePageMalloc.

llvm.func @malloc(i64) -> !llvm.ptr

// CHECK-LABEL: llvm.func @test_output
llvm.func @test_output(%size: i64) -> !llvm.struct<(ptr, ptr, i64)> {
  // CHECK: %[[P:.*]] = llvm.call @OMHugePageMalloc(%arg0) : (i64) -> !llvm.ptr
  // CHECK: llvm.insertvalue %[[P]]
  %0 = llvm.call @malloc(%size) : (i64) -> !llvm.ptr
  %1 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
  %2 = llvm.insertvalue %0, %1[0] : !llvm.struct<(ptr, ptr, i64)>
  %3 = llvm.insertvalue %0, %2[1] : !llvm.struct<(ptr, ptr, i64)>
  llvm.return %3 : !llvm.struct<(ptr, ptr, i64)>
}

// -----

// Freed, but the pointer is also stored: it may escape, so it is not cached.

llvm.func @malloc(i64) -> !llvm.ptr
llvm.func @free(!llvm.ptr)

// CHECK-LABEL: llvm.func @test_stored
llvm.func @test_stored(%size: i64, %slot: !llvm.ptr) {
  // CHECK: %[[P:.*]] = llvm.call @OMHugePageMalloc(%arg0) : (i64) -> !llvm.ptr
  // CHECK: llvm.store %[[P]], %arg1
  // CHECK: llvm.call @free(%[[P]]) : (!llvm.ptr) -> ()
  %0 = llvm.call @malloc(%size) : (i64) -> !llvm.ptr
  llvm.store %0, %slot : !llvm.ptr, !llvm.ptr
  llvm.call @free(%0) : (!llvm.ptr) -> ()
  llvm.return
}

// -----

// Freed on two paths: both frees go to the cache.

llvm.func @malloc(i64) -> !llvm.ptr
llvm.func @free(!llvm.ptr)
llvm.func @other()

// CHECK-LABEL: llvm.func @test_two_frees
llvm.func @test_two_frees(%size: i64, %cond: i1) {
  // CHECK: %[[P:.*]] = llvm.call @OMCacheAlloc(%arg0) : (i64) -> !llvm.ptr
  // CHECK: llvm.call @OMCacheFree(%[[P]]) : (!llvm.ptr) -> ()
  // CHECK: llvm.call @OMCacheFree(%[[P]]) : (!llvm.ptr) -> ()
  // CHECK-NOT: llvm.call @free
  %0 = llvm.call @malloc(%size) : (i64) -> !llvm.ptr
  llvm.cond_br %cond, ^bb1, ^bb2
^bb1:
  llvm.call @free(%0) : (!llvm.ptr) -> ()
  llvm.return
^bb2:
  llvm.call @other() : () -> ()
  llvm.call @free(%0) : (!llvm.ptr) -> ()
  llvm.return
}

// -----

// A per-thread buffer inside an OpenMP region is left alone.

llvm.func @malloc(i64) -> !llvm.ptr
llvm.func @free(!llvm.ptr)

// CHECK-LABEL: llvm.func @test_openmp
llvm.func @test_openmp(%size: i64) {
  // CHECK: omp.parallel
  // CHECK: %[[P:.*]] = llvm.call @OMHugePageMalloc(%arg0) : (i64) -> !llvm.ptr
  // CHECK: llvm.call @free(%[[P]]) : (!llvm.ptr) -> ()
  omp.parallel {
    %0 = llvm.call @malloc(%size) : (i64) -> !llvm.ptr
    llvm.call @free(%0) : (!llvm.ptr) -> ()
    omp.terminator
  }
  llvm.return
}
