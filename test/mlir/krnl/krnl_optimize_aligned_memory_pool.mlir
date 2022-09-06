// RUN: onnx-mlir-opt -O3 -allow-unregistered-dialect --optimize-memory-pools --canonicalize %s -split-input-file | FileCheck %s

/// 1. Single-chain workflow with alignment.
func.func @single_chain_dataflow(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>) -> memref<10x10xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0_i64 = arith.constant 0 : i64
  %c1600_i64 = arith.constant 1600 : i64
  %c1200_i64 = arith.constant 1200 : i64
  %c800_i64 = arith.constant 800 : i64
  %c400_i64 = arith.constant 400 : i64
  %0 = memref.alloc() : memref<10x10xf32>
  %1 = memref.alloc() : memref<1200xi8>
  %aligned = memref.alloc() {alignment = 4096 : i64} : memref<800xi8>
  %2 = "krnl.getref"(%aligned, %c400_i64) : (memref<800xi8>, i64) -> memref<10x10xf32>
  %3 = "krnl.getref"(%1, %c800_i64) : (memref<1200xi8>, i64) -> memref<10x10xf32>
  %4 = "krnl.getref"(%aligned, %c0_i64) : (memref<800xi8>, i64) -> memref<10x10xf32>
  %5 = "krnl.getref"(%1, %c400_i64) : (memref<1200xi8>, i64) -> memref<10x10xf32>
  %6 = "krnl.getref"(%1, %c0_i64) : (memref<1200xi8>, i64) -> memref<10x10xf32>
  %7:2 = krnl.define_loops 2
  krnl.iterate(%7#0, %7#1) with (%7#0 -> %arg2 = 0 to 10, %7#1 -> %arg3 = 0 to 10) {
    krnl.store %cst, %6[%arg2, %arg3] : memref<10x10xf32>
    %13 = krnl.define_loops 1
    krnl.iterate(%13) with (%13 -> %arg4 = 0 to 10) {
      %14 = krnl.load %arg0[%arg2, %arg4] : memref<10x10xf32>
      %15 = krnl.load %arg1[%arg4, %arg3] : memref<10x10xf32>
      %16 = krnl.load %6[%arg2, %arg3] : memref<10x10xf32>
      %17 = arith.mulf %14, %15 : f32
      %18 = arith.addf %16, %17 : f32
      krnl.store %18, %6[%arg2, %arg3] : memref<10x10xf32>
    }
  }
  %8:2 = krnl.define_loops 2
  krnl.iterate(%8#0, %8#1) with (%8#0 -> %arg2 = 0 to 10, %8#1 -> %arg3 = 0 to 10) {
    %13 = krnl.load %arg0[%arg2, %arg3] : memref<10x10xf32>
    %14 = krnl.load %6[%arg2, %arg3] : memref<10x10xf32>
    %15 = arith.addf %13, %14 : f32
    krnl.store %15, %5[%arg2, %arg3] : memref<10x10xf32>
  }
  %9:2 = krnl.define_loops 2
  krnl.iterate(%9#0, %9#1) with (%9#0 -> %arg2 = 0 to 10, %9#1 -> %arg3 = 0 to 10) {
    krnl.store %cst, %4[%arg2, %arg3] : memref<10x10xf32>
    %13 = krnl.define_loops 1
    krnl.iterate(%13) with (%13 -> %arg4 = 0 to 10) {
      %14 = krnl.load %arg0[%arg2, %arg4] : memref<10x10xf32>
      %15 = krnl.load %5[%arg4, %arg3] : memref<10x10xf32>
      %16 = krnl.load %4[%arg2, %arg3] : memref<10x10xf32>
      %17 = arith.mulf %14, %15 : f32
      %18 = arith.addf %16, %17 : f32
      krnl.store %18, %4[%arg2, %arg3] : memref<10x10xf32>
    }
  }
  %10:2 = krnl.define_loops 2
  krnl.iterate(%10#0, %10#1) with (%10#0 -> %arg2 = 0 to 10, %10#1 -> %arg3 = 0 to 10) {
    %13 = krnl.load %4[%arg2, %arg3] : memref<10x10xf32>
    %14 = krnl.load %arg1[%arg2, %arg3] : memref<10x10xf32>
    %15 = arith.addf %13, %14 : f32
    krnl.store %15, %3[%arg2, %arg3] : memref<10x10xf32>
  }
  %11:2 = krnl.define_loops 2
  krnl.iterate(%11#0, %11#1) with (%11#0 -> %arg2 = 0 to 10, %11#1 -> %arg3 = 0 to 10) {
    krnl.store %cst, %2[%arg2, %arg3] : memref<10x10xf32>
    %13 = krnl.define_loops 1
    krnl.iterate(%13) with (%13 -> %arg4 = 0 to 10) {
      %14 = krnl.load %arg0[%arg2, %arg4] : memref<10x10xf32>
      %15 = krnl.load %3[%arg4, %arg3] : memref<10x10xf32>
      %16 = krnl.load %2[%arg2, %arg3] : memref<10x10xf32>
      %17 = arith.mulf %14, %15 : f32
      %18 = arith.addf %16, %17 : f32
      krnl.store %18, %2[%arg2, %arg3] : memref<10x10xf32>
    }
  }
  %12:2 = krnl.define_loops 2
  krnl.iterate(%12#0, %12#1) with (%12#0 -> %arg2 = 0 to 10, %12#1 -> %arg3 = 0 to 10) {
    %13 = krnl.load %2[%arg2, %arg3] : memref<10x10xf32>
    %14 = krnl.load %arg1[%arg2, %arg3] : memref<10x10xf32>
    %15 = arith.addf %13, %14 : f32
    krnl.store %15, %0[%arg2, %arg3] : memref<10x10xf32>
  }
  memref.dealloc %1 : memref<1200xi8>
  memref.dealloc %aligned : memref<800xi8>
  return %0 : memref<10x10xf32>

  // CHECK-LABEL: single_chain_dataflow
  // CHECK-DAG: [[C400:%.+]] = arith.constant 400 : i64
  // CHECK-DAG: [[C0:%.+]] = arith.constant 0 : i64
  // CHECK-DAG: [[MEMPOOL:%.+]] = memref.alloc() : memref<800xi8>
  // CHECK-DAG: [[MEMPOOL_ALIGNED:%.+]] = memref.alloc() {alignment = 4096 : i64} : memref<4096xi8>
  // CHECK: "krnl.getref"([[MEMPOOL_ALIGNED]], [[C0]]) : (memref<4096xi8>, i64) -> memref<10x10xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C0]]) : (memref<800xi8>, i64) -> memref<10x10xf32>
  // CHECK: "krnl.getref"([[MEMPOOL_ALIGNED]], [[C0]]) : (memref<4096xi8>, i64) -> memref<10x10xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C400]]) : (memref<800xi8>, i64) -> memref<10x10xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C0]]) : (memref<800xi8>, i64) -> memref<10x10xf32>
}

