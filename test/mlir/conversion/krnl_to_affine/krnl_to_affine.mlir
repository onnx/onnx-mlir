// RUN: onnx-mlir-opt -O3 --convert-krnl-to-affine %s -split-input-file | FileCheck %s

func.func @test_lower_degenerate_iterate(%arg0: memref<f32>) -> memref<f32> {
  %0 = memref.alloc() : memref<f32>
  krnl.iterate() with () {
    %1 = memref.load %arg0[] : memref<f32>
    memref.store %1, %0[] : memref<f32>
  }
  return %0 : memref<f32>
  // CHECK-LABEL: test_lower_degenerate_iterate
  // CHECK-NEXT: [[ALLOC:%.+]] = memref.alloc() : memref<f32>
  // CHECK-NEXT: [[LOAD:%.+]] = memref.load %{{.*}}[] : memref<f32>
  // CHECK-NEXT: store [[LOAD]], [[ALLOC]][] : memref<f32>
  // CHECK-NEXT: return [[ALLOC]] : memref<f32>
}

// -----

// COM: Simple krnl.load/store to affine.load/store.
func.func @test_krnl_load_store(%arg0: memref<10x10xf32>) -> memref<1xf32> {
  %c0 = arith.constant 0 : index
  %1 = krnl.load %arg0[%c0, %c0] : memref<10x10xf32>
  %2 = memref.alloc() : memref<1xf32>
  krnl.store %1, %2[%c0] : memref<1xf32>
  return %2 : memref<1xf32>

  // CHECK-LABEL: test_krnl_load_store
  // CHECK: [[C0:%.+]] = arith.constant 0 : index
  // CHECK: [[LOAD:%.+]] = affine.load %arg0{{\[}}[[C0]], [[C0]]{{\]}} : memref<10x10xf32>
  // CHECK: [[RES:%.+]]  = memref.alloc() : memref<1xf32>
  // CHECK: affine.store [[LOAD]], [[RES]]{{\[}}[[C0]]{{\]}} : memref<1xf32>

}

// -----

// COM: Check whether krnl.load is lowered to std.load due to non-affine indices.
func.func @test_krnl_load_with_krnl_iterate(%arg0: memref<10x10xf32>, %arg1: memref<10x?xf32>) -> memref<10x10xf32> {
  %0 = memref.alloc() : memref<10x10xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %1 = memref.dim %arg1, %c1 : memref<10x?xf32>
  %2:2 = krnl.define_loops 2
  krnl.iterate(%2#0, %2#1) with (%2#0 -> %arg2 = 0 to 10, %2#1 -> %arg3 = 0 to 10) {
    %3 = krnl.load %arg0[%arg2, %arg3] : memref<10x10xf32>
    %4 = arith.cmpi sgt, %1, %c1 : index
    %5 = arith.select %4, %arg3, %c0 : index
    %6 = krnl.load %arg1[%arg2, %5] : memref<10x?xf32>
    %7 = arith.addf %3, %6 : f32
    krnl.store %7, %0[%arg2, %arg3] : memref<10x10xf32>
  }
  return %0 : memref<10x10xf32>

  // CHECK-LABEL:  @test_krnl_load_with_krnl_iterate
  // CHECK:  affine.for {{.*}}
  // CHECK:    affine.for {{.*}}
  // CHECK:      {{.*}} = affine.load {{.*}} : memref<10x10xf32>
  // CHECK:      {{.*}} = memref.load {{.*}} : memref<10x?xf32>
  // CHECK:      affine.store {{.*}} : memref<10x10xf32>
}

// -----

// COM: Check whether krnl.store is lowered to std.load due to non-affine indices.
func.func @test_krnl_store_with_krnl_iterate(%arg0: memref<10x10xf32>, %arg1: memref<10x?xf32>) -> memref<10x10xf32> {
  %0 = memref.alloc() : memref<10x10xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %1 = memref.dim %arg1, %c1 : memref<10x?xf32>
  %2:2 = krnl.define_loops 2
  krnl.iterate(%2#0, %2#1) with (%2#0 -> %arg2 = 0 to 10, %2#1 -> %arg3 = 0 to 10) {
    %3 = krnl.load %arg0[%arg2, %arg3] : memref<10x10xf32>
    %4 = arith.cmpi sgt, %1, %c1 : index
    %5 = arith.select %4, %arg3, %c0 : index
    %6 = krnl.load %arg1[%arg2, %5] : memref<10x?xf32>
    %7 = arith.addf %3, %6 : f32
    krnl.store %7, %0[%arg2, %5] : memref<10x10xf32>
  }
  return %0 : memref<10x10xf32>

  // CHECK-LABEL:  @test_krnl_store_with_krnl_iterate
  // CHECK:  affine.for {{.*}}
  // CHECK:    affine.for {{.*}}
  // CHECK:      {{.*}} = affine.load {{.*}} : memref<10x10xf32>
  // CHECK:      {{.*}} = memref.load {{.*}} : memref<10x?xf32>
  // CHECK:      store {{.*}} : memref<10x10xf32>
}

// -----

// COM: Check whether krnl.load/store is lowered to affine.load/store due to affine indices.
#map = affine_map<(d0) -> (d0 + 1)>
func.func @test_krnl_load_store_with_affine(%arg0: memref<10x10xf32>, %arg1: memref<10x?xf32>) -> memref<10x10xf32> {
  %0 = memref.alloc() : memref<10x10xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %1 = memref.dim %arg1, %c1 : memref<10x?xf32>
  %2:2 = krnl.define_loops 2
  krnl.iterate(%2#0, %2#1) with (%2#0 -> %arg2 = 0 to 10, %2#1 -> %arg3 = 0 to 10) {
    %3 = krnl.load %arg0[%arg2, %arg3] : memref<10x10xf32>
    %4 = affine.apply #map(%arg3)
    %5 = krnl.load %arg1[%arg2, %4] : memref<10x?xf32>
    krnl.store %5, %0[%arg2, %4] : memref<10x10xf32>
  }
  return %0 : memref<10x10xf32>

  // CHECK-LABEL:  @test_krnl_load_store_with_affine
  // CHECK:  affine.for {{.*}}
  // CHECK:    affine.for {{.*}}
  // CHECK:      {{.*}} = affine.load {{.*}} : memref<10x10xf32>
  // CHECK:      {{.*}} = affine.load {{.*}} : memref<10x?xf32>
  // CHECK:      affine.store {{.*}} : memref<10x10xf32>
}
