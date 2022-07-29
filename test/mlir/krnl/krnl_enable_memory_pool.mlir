// RUN: onnx-mlir-opt -O3 --enable-memory-pool %s -split-input-file | FileCheck %s

#map0 = affine_map<(d0, d1) -> (0, d1 floordiv 64, 0, d0 floordiv 32, d0 mod 32, d1 mod 64)>
func.func @test_allocs_not_lowered(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>) -> memref<10x10xf32> {
    %0 = memref.alloc() : memref<10x10xf32>
    %1 = memref.alloc() {alignment = 4096 : i64} : memref<10x10xf32>
    %2 = memref.alloc() : memref<10x10xf32, #map0>
    %3 = memref.alloc() : memref<10x10xf32>
    %4:2 = krnl.define_loops 2
    krnl.iterate(%4#0, %4#1) with (%4#0 -> %arg2 = 0 to 10, %4#1 -> %arg3 = 0 to 10) {
      %8 = krnl.load %arg0[%arg2, %arg3] : memref<10x10xf32>
      %9 = krnl.load %arg1[%arg2, %arg3] : memref<10x10xf32>
      %10 = arith.addf %8, %9 : f32
      krnl.store %10, %3[%arg2, %arg3] : memref<10x10xf32>
    }
    %cst = arith.constant 0.000000e+00 : f32
    %5:2 = krnl.define_loops 2
    krnl.iterate(%5#0, %5#1) with (%5#0 -> %arg2 = 0 to 10, %5#1 -> %arg3 = 0 to 10) {
      krnl.store %cst, %2[%arg2, %arg3] : memref<10x10xf32, #map0>
      %8 = krnl.define_loops 1
      krnl.iterate(%8) with (%8 -> %arg4 = 0 to 10) {
        %9 = krnl.load %arg0[%arg2, %arg4] : memref<10x10xf32>
        %10 = krnl.load %3[%arg4, %arg3] : memref<10x10xf32>
        %11 = krnl.load %2[%arg2, %arg3] : memref<10x10xf32, #map0>
        %12 = arith.mulf %9, %10 : f32
        %13 = arith.addf %11, %12 : f32
        krnl.store %13, %2[%arg2, %arg3] : memref<10x10xf32, #map0>
      }
    }
    %6:2 = krnl.define_loops 2
    krnl.iterate(%6#0, %6#1) with (%6#0 -> %arg2 = 0 to 10, %6#1 -> %arg3 = 0 to 10) {
      %8 = krnl.load %3[%arg2, %arg3] : memref<10x10xf32>
      %9 = krnl.load %2[%arg2, %arg3] : memref<10x10xf32, #map0>
      %10 = arith.addf %8, %9 : f32
      krnl.store %10, %1[%arg2, %arg3] : memref<10x10xf32>
    }
    %7:2 = krnl.define_loops 2
    krnl.iterate(%7#0, %7#1) with (%7#0 -> %arg2 = 0 to 10, %7#1 -> %arg3 = 0 to 10) {
      %8 = krnl.load %2[%arg2, %arg3] : memref<10x10xf32, #map0>
      %9 = krnl.load %1[%arg2, %arg3] : memref<10x10xf32>
      %10 = arith.addf %8, %9 : f32
      krnl.store %10, %0[%arg2, %arg3] : memref<10x10xf32>
    }
    memref.dealloc %3 : memref<10x10xf32>
    memref.dealloc %2 : memref<10x10xf32, #map0>
    memref.dealloc %1 : memref<10x10xf32>
    return %0 : memref<10x10xf32>
}

// CHECK: [[MAP:#.+]] = affine_map<(d0, d1)
// CHECK: test_allocs_not_lowered

/// AllocOps with alignment attributes are preserved.
// CHECK: [[ALLOC1:%.+]] = memref.alloc() {alignment = 4096 : i64} : memref<400xi8>
// CHECK: "krnl.getref"([[ALLOC1]], {{.*}}) : (memref<400xi8>, i64) -> memref<10x10xf32>

/// AllocOps with unresolved maps cannot be lowered.
// CHECK: [[ALLOC2:%.+]] = memref.alloc() : memref<10x10xf32, [[MAP]]>
// CHECK-NOT: "krnl.getref"([[ALLOC2]], {{.*}})

// CHECK: [[ALLOC3:%.+]] = memref.alloc() : memref<400xi8>
// CHECK: "krnl.getref"([[ALLOC3]], {{.*}})

// -----

// Apply memorypool in `test_unsqueeze_squeeze_dealloc` in `test/mlir/onnx/onnx_lowering.mlir`.
// Memorypool is not enabled in %0 because it is a return value via reinterpret_cast.

func.func @test_unsqueeze_squeeze_dealloc_mempool(%arg0: memref<10x20xf32>) -> memref<20x10xf32> {
    %0 = memref.alloc() : memref<20x1x1x10xf32>
    %1 = memref.alloc() : memref<20x10xf32>
    %c20 = arith.constant 20 : index
    %c10 = arith.constant 10 : index
    %2:2 = krnl.define_loops 2
    krnl.iterate(%2#0, %2#1) with (%2#0 -> %arg1 = 0 to 10, %2#1 -> %arg2 = 0 to 20) {
      %6 = krnl.load %arg0[%arg1, %arg2] : memref<10x20xf32>
      krnl.store %6, %1[%arg2, %arg1] : memref<20x10xf32>
    }
    %c20_0 = arith.constant 20 : index
    %c1 = arith.constant 1 : index
    %c10_1 = arith.constant 10 : index
    %c1_2 = arith.constant 1 : index
    %c1_3 = arith.constant 1 : index
    %c1_4 = arith.constant 1 : index
    %c10_5 = arith.constant 10 : index
    %c10_6 = arith.constant 10 : index
    %3 = memref.reinterpret_cast %1 to offset: [0], sizes: [20, 1, 10, 1], strides: [10, 10, 1, 1] : memref<20x10xf32> to memref<20x1x10x1xf32>
    %c20_7 = arith.constant 20 : index
    %c1_8 = arith.constant 1 : index
    %c1_9 = arith.constant 1 : index
    %c10_10 = arith.constant 10 : index
    %4:4 = krnl.define_loops 4
    krnl.iterate(%4#0, %4#1, %4#2, %4#3) with (%4#0 -> %arg1 = 0 to 20, %4#1 -> %arg2 = 0 to 1, %4#2 -> %arg3 = 0 to 10, %4#3 -> %arg4 = 0 to 1) {
      %6 = krnl.load %3[%arg1, %arg2, %arg3, %arg4] : memref<20x1x10x1xf32>
      krnl.store %6, %0[%arg1, %arg4, %arg2, %arg3] : memref<20x1x1x10xf32>
    }
    %c20_11 = arith.constant 20 : index
    %c10_12 = arith.constant 10 : index
    %c1_13 = arith.constant 1 : index
    %c10_14 = arith.constant 10 : index
    %5 = memref.reinterpret_cast %0 to offset: [0], sizes: [20, 10], strides: [10, 1] : memref<20x1x1x10xf32> to memref<20x10xf32>
    memref.dealloc %1 : memref<20x10xf32>
    return %5 : memref<20x10xf32>

    // CHECK-LABEL: func @test_unsqueeze_squeeze_dealloc_mempool
    // CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i64
    // CHECK-DAG:       [[VAR_0_:%.+]] = memref.alloc() : memref<20x1x1x10xf32>
    // CHECK-DAG:       [[VAR_1_:%.+]] = memref.alloc() : memref<800xi8>
    // CHECK:           [[VAR_3_:%.+]] = "krnl.getref"([[VAR_1_]], [[CST_0_]]) : (memref<800xi8>, i64) -> memref<20x10xf32>
    // CHECK-DAG:       [[VAR_4_:%.+]] = memref.reinterpret_cast [[VAR_3_]] to offset: [0], sizes: [20, 1, 10, 1], strides: [10, 10, 1, 1] : memref<20x10xf32> to memref<20x1x10x1xf32>
    // CHECK-DAG:       [[VAR_6_:%.+]] = memref.reinterpret_cast [[VAR_0_]] to offset: [0], sizes: [20, 10], strides: [10, 1] : memref<20x1x1x10xf32> to memref<20x10xf32>
    // CHECK:           memref.dealloc [[VAR_1_]] : memref<800xi8>
    // CHECK:           return [[VAR_6_]] : memref<20x10xf32>
}

// -----

func.func @test_return_cast(%arg0: memref<2x1xf32>) -> memref<1x2xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = memref.alloc() {alignment = 16 : i64} : memref<2x1xf32>
  %1:2 = krnl.define_loops 2
  krnl.iterate(%1#0, %1#1) with (%1#0 -> %arg1 = 0 to 2, %1#1 -> %arg2 = 0 to 1) {
    %7 = krnl.load %arg0[%arg1, %arg2] : memref<2x1xf32>
    krnl.store %7, %0[%arg1, %arg2] : memref<2x1xf32>
  }
  %2 = memref.alloc() {alignment = 16 : i64} : memref<2xindex>
  krnl.store %c1, %2[%c0] : memref<2xindex>
  krnl.store %c2, %2[%c1] : memref<2xindex>
  %3 = krnl.load %2[%c0] : memref<2xindex>
  %4 = krnl.load %2[%c1] : memref<2xindex>
  memref.dealloc %2 : memref<2xindex>
  %5 = memref.reinterpret_cast %0 to offset: [0], sizes: [%3, %4], strides: [%4, 1] : memref<2x1xf32> to memref<?x?xf32>
  %6 = memref.cast %5 : memref<?x?xf32> to memref<1x2xf32>
  return %6 : memref<1x2xf32>

  // CHECK-LABEL: func @test_return_cast
  // CHECK: [[VAR_0_:%.+]] = memref.alloc() {alignment = 16 : i64} : memref<2x1xf32>
  // CHECK-NOT: memref.dealloc [[VAR_0_]]
}
