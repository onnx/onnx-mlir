// RUN: onnx-mlir-opt --enable-memory-pool %s -split-input-file | FileCheck %s

#map0 = affine_map<(d0, d1) -> (0, d1 floordiv 64, 0, d0 floordiv 32, d0 mod 32, d1 mod 64)>
func @test_allocs_not_lowered(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>) -> memref<10x10xf32> {
    %0 = alloc() : memref<10x10xf32>
    %1 = alloc() {alignment = 4096 : i64} : memref<10x10xf32>
    %2 = alloc() : memref<10x10xf32, #map0>
    %3 = alloc() : memref<10x10xf32>
    %4:2 = krnl.define_loops 2
    krnl.iterate(%4#0, %4#1) with (%4#0 -> %arg2 = 0 to 10, %4#1 -> %arg3 = 0 to 10) {
      %8 = krnl.load %arg0[%arg2, %arg3] : memref<10x10xf32>
      %9 = krnl.load %arg1[%arg2, %arg3] : memref<10x10xf32>
      %10 = addf %8, %9 : f32
      krnl.store %10, %3[%arg2, %arg3] : memref<10x10xf32>
    }
    %cst = constant 0.000000e+00 : f32
    %5:2 = krnl.define_loops 2
    krnl.iterate(%5#0, %5#1) with (%5#0 -> %arg2 = 0 to 10, %5#1 -> %arg3 = 0 to 10) {
      krnl.store %cst, %2[%arg2, %arg3] : memref<10x10xf32, #map0>
      %8 = krnl.define_loops 1
      krnl.iterate(%8) with (%8 -> %arg4 = 0 to 10) {
        %9 = krnl.load %arg0[%arg2, %arg4] : memref<10x10xf32>
        %10 = krnl.load %3[%arg4, %arg3] : memref<10x10xf32>
        %11 = krnl.load %2[%arg2, %arg3] : memref<10x10xf32, #map0>
        %12 = mulf %9, %10 : f32
        %13 = addf %11, %12 : f32
        krnl.store %13, %2[%arg2, %arg3] : memref<10x10xf32, #map0>
      }
    }
    %6:2 = krnl.define_loops 2
    krnl.iterate(%6#0, %6#1) with (%6#0 -> %arg2 = 0 to 10, %6#1 -> %arg3 = 0 to 10) {
      %8 = krnl.load %3[%arg2, %arg3] : memref<10x10xf32>
      %9 = krnl.load %2[%arg2, %arg3] : memref<10x10xf32, #map0>
      %10 = addf %8, %9 : f32
      krnl.store %10, %1[%arg2, %arg3] : memref<10x10xf32>
    }
    %7:2 = krnl.define_loops 2
    krnl.iterate(%7#0, %7#1) with (%7#0 -> %arg2 = 0 to 10, %7#1 -> %arg3 = 0 to 10) {
      %8 = krnl.load %2[%arg2, %arg3] : memref<10x10xf32, #map0>
      %9 = krnl.load %1[%arg2, %arg3] : memref<10x10xf32>
      %10 = addf %8, %9 : f32
      krnl.store %10, %0[%arg2, %arg3] : memref<10x10xf32>
    }
    dealloc %3 : memref<10x10xf32>
    dealloc %2 : memref<10x10xf32, #map0>
    dealloc %1 : memref<10x10xf32>
    return %0 : memref<10x10xf32>
}

// CHECK: [[MAP:#.+]] = affine_map<(d0, d1)
// CHECK: test_allocs_not_lowered

/// AllocOps with alignment attributes are preserved.
// CHECK: [[ALLOC1:%.+]] = alloc() {alignment = 4096 : i64} : memref<400xi8>
// CHECK: "krnl.getref"([[ALLOC1]], {{.*}}) : (memref<400xi8>, i64) -> memref<10x10xf32>

/// AllocOps with unresolved maps cannot be lowered.
// CHECK: [[ALLOC2:%.+]] = alloc() : memref<10x10xf32, [[MAP]]>
// CHECK-NOT: "krnl.getref"([[ALLOC2]], {{.*}})

// CHECK: [[ALLOC3:%.+]] = alloc() : memref<400xi8>
// CHECK: "krnl.getref"([[ALLOC3]], {{.*}})

