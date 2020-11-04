// RUN: onnx-mlir-opt --optimize-memory-pools --canonicalize %s -split-input-file | FileCheck %s

/// 1. Base case where we have a single-chain workflow.
func @single_chain_dataflow(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>) -> memref<10x10xf32> {
  %cst = constant 0.000000e+00 : f32
  %c0_i64 = constant 0 : i64
  %c1600_i64 = constant 1600 : i64
  %c1200_i64 = constant 1200 : i64
  %c800_i64 = constant 800 : i64
  %c400_i64 = constant 400 : i64
  %0 = alloc() : memref<10x10xf32>
  %1 = alloc() : memref<2000xi8>
  %2 = "krnl.getref"(%1, %c1600_i64) : (memref<2000xi8>, i64) -> memref<10x10xf32>
  %3 = "krnl.getref"(%1, %c1200_i64) : (memref<2000xi8>, i64) -> memref<10x10xf32>
  %4 = "krnl.getref"(%1, %c800_i64) : (memref<2000xi8>, i64) -> memref<10x10xf32>
  %5 = "krnl.getref"(%1, %c400_i64) : (memref<2000xi8>, i64) -> memref<10x10xf32>
  %6 = "krnl.getref"(%1, %c0_i64) : (memref<2000xi8>, i64) -> memref<10x10xf32>
  %7:2 = krnl.define_loops 2
  krnl.iterate(%7#0, %7#1) with (%7#0 -> %arg2 = 0 to 10, %7#1 -> %arg3 = 0 to 10) {
    affine.store %cst, %6[symbol(%arg2), symbol(%arg3)] : memref<10x10xf32>
    %13 = krnl.define_loops 1
    krnl.iterate(%13) with (%13 -> %arg4 = 0 to 10) {
      %14 = affine.load %arg0[symbol(%arg2), symbol(%arg4)] : memref<10x10xf32>
      %15 = affine.load %arg1[symbol(%arg4), symbol(%arg3)] : memref<10x10xf32>
      %16 = affine.load %6[symbol(%arg2), symbol(%arg3)] : memref<10x10xf32>
      %17 = mulf %14, %15 : f32
      %18 = addf %16, %17 : f32
      affine.store %18, %6[symbol(%arg2), symbol(%arg3)] : memref<10x10xf32>
    }
  }
  %8:2 = krnl.define_loops 2
  krnl.iterate(%8#0, %8#1) with (%8#0 -> %arg2 = 0 to 10, %8#1 -> %arg3 = 0 to 10) {
    %13 = affine.load %arg0[symbol(%arg2), symbol(%arg3)] : memref<10x10xf32>
    %14 = affine.load %6[symbol(%arg2), symbol(%arg3)] : memref<10x10xf32>
    %15 = addf %13, %14 : f32
    affine.store %15, %5[symbol(%arg2), symbol(%arg3)] : memref<10x10xf32>
  }
  %9:2 = krnl.define_loops 2
  krnl.iterate(%9#0, %9#1) with (%9#0 -> %arg2 = 0 to 10, %9#1 -> %arg3 = 0 to 10) {
    affine.store %cst, %4[symbol(%arg2), symbol(%arg3)] : memref<10x10xf32>
    %13 = krnl.define_loops 1
    krnl.iterate(%13) with (%13 -> %arg4 = 0 to 10) {
      %14 = affine.load %arg0[symbol(%arg2), symbol(%arg4)] : memref<10x10xf32>
      %15 = affine.load %5[symbol(%arg4), symbol(%arg3)] : memref<10x10xf32>
      %16 = affine.load %4[symbol(%arg2), symbol(%arg3)] : memref<10x10xf32>
      %17 = mulf %14, %15 : f32
      %18 = addf %16, %17 : f32
      affine.store %18, %4[symbol(%arg2), symbol(%arg3)] : memref<10x10xf32>
    }
  }
  %10:2 = krnl.define_loops 2
  krnl.iterate(%10#0, %10#1) with (%10#0 -> %arg2 = 0 to 10, %10#1 -> %arg3 = 0 to 10) {
    %13 = affine.load %4[symbol(%arg2), symbol(%arg3)] : memref<10x10xf32>
    %14 = affine.load %arg1[symbol(%arg2), symbol(%arg3)] : memref<10x10xf32>
    %15 = addf %13, %14 : f32
    affine.store %15, %3[symbol(%arg2), symbol(%arg3)] : memref<10x10xf32>
  }
  %11:2 = krnl.define_loops 2
  krnl.iterate(%11#0, %11#1) with (%11#0 -> %arg2 = 0 to 10, %11#1 -> %arg3 = 0 to 10) {
    affine.store %cst, %2[symbol(%arg2), symbol(%arg3)] : memref<10x10xf32>
    %13 = krnl.define_loops 1
    krnl.iterate(%13) with (%13 -> %arg4 = 0 to 10) {
      %14 = affine.load %arg0[symbol(%arg2), symbol(%arg4)] : memref<10x10xf32>
      %15 = affine.load %3[symbol(%arg4), symbol(%arg3)] : memref<10x10xf32>
      %16 = affine.load %2[symbol(%arg2), symbol(%arg3)] : memref<10x10xf32>
      %17 = mulf %14, %15 : f32
      %18 = addf %16, %17 : f32
      affine.store %18, %2[symbol(%arg2), symbol(%arg3)] : memref<10x10xf32>
    }
  }
  %12:2 = krnl.define_loops 2
  krnl.iterate(%12#0, %12#1) with (%12#0 -> %arg2 = 0 to 10, %12#1 -> %arg3 = 0 to 10) {
    %13 = affine.load %2[symbol(%arg2), symbol(%arg3)] : memref<10x10xf32>
    %14 = affine.load %arg1[symbol(%arg2), symbol(%arg3)] : memref<10x10xf32>
    %15 = addf %13, %14 : f32
    affine.store %15, %0[symbol(%arg2), symbol(%arg3)] : memref<10x10xf32>
  }
  dealloc %1 : memref<2000xi8>
  return %0 : memref<10x10xf32>

  // CHECK-LABEL: single_chain_dataflow
  // CHECK: [[C0:%.+]] = constant 0 : i64
  // CHECK: [[C400:%.+]] = constant 400 : i64
  // CHECK: [[MEMPOOL:%.+]] = alloc() : memref<800xi8>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C0]]) : (memref<800xi8>, i64) -> memref<10x10xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C400]]) : (memref<800xi8>, i64) -> memref<10x10xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C0]]) : (memref<800xi8>, i64) -> memref<10x10xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C400]]) : (memref<800xi8>, i64) -> memref<10x10xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C0]]) : (memref<800xi8>, i64) -> memref<10x10xf32>
}

// -----

/// 2. Test for MemRefs with different shapes that can share the same slot.
func @multiple_shaped_memrefs(%arg0: memref<10x5xf32>, %arg1: memref<5x5xf32>, %arg2: memref<10x10xf32>) -> memref<10x5xf32> {
    %cst = constant 0.000000e+00 : f32
    %c0_i64 = constant 0 : i64
    %c1200_i64 = constant 1200 : i64
    %c1000_i64 = constant 1000 : i64
    %c800_i64 = constant 800 : i64
    %c600_i64 = constant 600 : i64
    %c400_i64 = constant 400 : i64
    %c200_i64 = constant 200 : i64
    %0 = alloc() : memref<10x5xf32>
    %1 = alloc() : memref<1400xi8>
    %2 = "krnl.getref"(%1, %c1200_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %3 = "krnl.getref"(%1, %c1000_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %4 = "krnl.getref"(%1, %c800_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %5 = "krnl.getref"(%1, %c600_i64) : (memref<1400xi8>, i64) -> memref<5x10xf32>
    %6 = "krnl.getref"(%1, %c400_i64) : (memref<1400xi8>, i64) -> memref<5x10xf32>
    %7 = "krnl.getref"(%1, %c200_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %8 = "krnl.getref"(%1, %c0_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %9:2 = krnl.define_loops 2
    krnl.iterate(%9#0, %9#1) with (%9#0 -> %arg3 = 0 to 10, %9#1 -> %arg4 = 0 to 5) {
      affine.store %cst, %8[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      %17 = krnl.define_loops 1
      krnl.iterate(%17) with (%17 -> %arg5 = 0 to 5) {
        %18 = affine.load %arg0[symbol(%arg3), symbol(%arg5)] : memref<10x5xf32>
        %19 = affine.load %arg1[symbol(%arg5), symbol(%arg4)] : memref<5x5xf32>
        %20 = affine.load %8[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
        %21 = mulf %18, %19 : f32
        %22 = addf %20, %21 : f32
        affine.store %22, %8[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      }
    }
    %10:2 = krnl.define_loops 2
    krnl.iterate(%10#0, %10#1) with (%10#0 -> %arg3 = 0 to 10, %10#1 -> %arg4 = 0 to 5) {
      %17 = affine.load %arg0[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      %18 = affine.load %8[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      %19 = addf %17, %18 : f32
      affine.store %19, %7[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
    }
    %11:2 = krnl.define_loops 2
    krnl.iterate(%11#0, %11#1) with (%11#0 -> %arg3 = 0 to 10, %11#1 -> %arg4 = 0 to 5) {
      %17 = affine.load %7[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      affine.store %17, %6[symbol(%arg4), symbol(%arg3)] : memref<5x10xf32>
    }
    %12:2 = krnl.define_loops 2
    krnl.iterate(%12#0, %12#1) with (%12#0 -> %arg3 = 0 to 5, %12#1 -> %arg4 = 0 to 10) {
      affine.store %cst, %5[symbol(%arg3), symbol(%arg4)] : memref<5x10xf32>
      %17 = krnl.define_loops 1
      krnl.iterate(%17) with (%17 -> %arg5 = 0 to 10) {
        %18 = affine.load %6[symbol(%arg3), symbol(%arg5)] : memref<5x10xf32>
        %19 = affine.load %arg2[symbol(%arg5), symbol(%arg4)] : memref<10x10xf32>
        %20 = affine.load %5[symbol(%arg3), symbol(%arg4)] : memref<5x10xf32>
        %21 = mulf %18, %19 : f32
        %22 = addf %20, %21 : f32
        affine.store %22, %5[symbol(%arg3), symbol(%arg4)] : memref<5x10xf32>
      }
    }
    %13:2 = krnl.define_loops 2
    krnl.iterate(%13#0, %13#1) with (%13#0 -> %arg3 = 0 to 5, %13#1 -> %arg4 = 0 to 10) {
      %17 = affine.load %5[symbol(%arg3), symbol(%arg4)] : memref<5x10xf32>
      affine.store %17, %4[symbol(%arg4), symbol(%arg3)] : memref<10x5xf32>
    }
    %14:2 = krnl.define_loops 2
    krnl.iterate(%14#0, %14#1) with (%14#0 -> %arg3 = 0 to 10, %14#1 -> %arg4 = 0 to 5) {
      %17 = affine.load %4[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      %18 = affine.load %arg0[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      %19 = addf %17, %18 : f32
      affine.store %19, %3[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
    }
    %15:2 = krnl.define_loops 2
    krnl.iterate(%15#0, %15#1) with (%15#0 -> %arg3 = 0 to 10, %15#1 -> %arg4 = 0 to 5) {
      affine.store %cst, %2[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      %17 = krnl.define_loops 1
      krnl.iterate(%17) with (%17 -> %arg5 = 0 to 5) {
        %18 = affine.load %3[symbol(%arg3), symbol(%arg5)] : memref<10x5xf32>
        %19 = affine.load %arg1[symbol(%arg5), symbol(%arg4)] : memref<5x5xf32>
        %20 = affine.load %2[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
        %21 = mulf %18, %19 : f32
        %22 = addf %20, %21 : f32
        affine.store %22, %2[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      }
    }
    %16:2 = krnl.define_loops 2
    krnl.iterate(%16#0, %16#1) with (%16#0 -> %arg3 = 0 to 10, %16#1 -> %arg4 = 0 to 5) {
      %17 = affine.load %2[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      %18 = affine.load %arg0[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      %19 = addf %17, %18 : f32
      affine.store %19, %0[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
    }
    dealloc %1 : memref<1400xi8>
    return %0 : memref<10x5xf32>

  // CHECK-LABEL: multiple_shaped_memrefs
  // CHECK: [[C0:%.+]] = constant 0 : i64
  // CHECK: [[C200:%.+]] = constant 200 : i64
  // CHECK: [[MEMPOOL:%.+]] = alloc() : memref<400xi8>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C0]]) : (memref<400xi8>, i64) -> memref<10x5xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C200]]) : (memref<400xi8>, i64) -> memref<10x5xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C0]]) : (memref<400xi8>, i64) -> memref<10x5xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C200]]) : (memref<400xi8>, i64) -> memref<5x10xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C0]]) : (memref<400xi8>, i64) -> memref<5x10xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C200]]) : (memref<400xi8>, i64) -> memref<10x5xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C0]]) : (memref<400xi8>, i64) -> memref<10x5xf32>
}

// -----

/// 3. Test dependency analysis for MemRefs copied using the krnl.memcpy instruction.
func @analysis_krnl_memcpy(%arg0: memref<10x5xf32>, %arg1: memref<5x5xf32>, %arg2: memref<10x10xf32>) -> memref<10x5xf32> {
    %cst = constant 0.000000e+00 : f32
    %c0_i64 = constant 0 : i64
    %c1200_i64 = constant 1200 : i64
    %c1000_i64 = constant 1000 : i64
    %c800_i64 = constant 800 : i64
    %c600_i64 = constant 600 : i64
    %c400_i64 = constant 400 : i64
    %c200_i64 = constant 200 : i64
    %0 = alloc() : memref<10x5xf32>
    %1 = alloc() : memref<1400xi8>
    %2 = "krnl.getref"(%1, %c1200_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %3 = "krnl.getref"(%1, %c1000_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %4 = "krnl.getref"(%1, %c800_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %5 = "krnl.getref"(%1, %c600_i64) : (memref<1400xi8>, i64) -> memref<5x10xf32>
    %6 = "krnl.getref"(%1, %c400_i64) : (memref<1400xi8>, i64) -> memref<5x10xf32>
    %7 = "krnl.getref"(%1, %c200_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %8 = "krnl.getref"(%1, %c0_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %9:2 = krnl.define_loops 2
    krnl.iterate(%9#0, %9#1) with (%9#0 -> %arg3 = 0 to 10, %9#1 -> %arg4 = 0 to 5) {
      affine.store %cst, %8[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      %17 = krnl.define_loops 1
      krnl.iterate(%17) with (%17 -> %arg5 = 0 to 5) {
        %18 = affine.load %arg0[symbol(%arg3), symbol(%arg5)] : memref<10x5xf32>
        %19 = affine.load %arg1[symbol(%arg5), symbol(%arg4)] : memref<5x5xf32>
        %20 = affine.load %8[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
        %21 = mulf %18, %19 : f32
        %22 = addf %20, %21 : f32
        affine.store %22, %8[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      }
    }
    %10:2 = krnl.define_loops 2
    krnl.iterate(%10#0, %10#1) with (%10#0 -> %arg3 = 0 to 10, %10#1 -> %arg4 = 0 to 5) {
      %17 = affine.load %arg0[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      %18 = affine.load %8[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      %19 = addf %17, %18 : f32
      affine.store %19, %7[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
    }
    "krnl.memcpy"(%6, %7, %c200_i64) : (memref<5x10xf32>, memref<10x5xf32>, i64) -> ()
    %12:2 = krnl.define_loops 2
    krnl.iterate(%12#0, %12#1) with (%12#0 -> %arg3 = 0 to 5, %12#1 -> %arg4 = 0 to 10) {
      affine.store %cst, %5[symbol(%arg3), symbol(%arg4)] : memref<5x10xf32>
      %17 = krnl.define_loops 1
      krnl.iterate(%17) with (%17 -> %arg5 = 0 to 10) {
        %18 = affine.load %6[symbol(%arg3), symbol(%arg5)] : memref<5x10xf32>
        %19 = affine.load %arg2[symbol(%arg5), symbol(%arg4)] : memref<10x10xf32>
        %20 = affine.load %5[symbol(%arg3), symbol(%arg4)] : memref<5x10xf32>
        %21 = mulf %18, %19 : f32
        %22 = addf %20, %21 : f32
        affine.store %22, %5[symbol(%arg3), symbol(%arg4)] : memref<5x10xf32>
      }
    }
    "krnl.memcpy"(%4, %5, %c200_i64) : (memref<10x5xf32>, memref<5x10xf32>, i64) -> ()
    %14:2 = krnl.define_loops 2
    krnl.iterate(%14#0, %14#1) with (%14#0 -> %arg3 = 0 to 10, %14#1 -> %arg4 = 0 to 5) {
      %17 = affine.load %4[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      %18 = affine.load %arg0[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      %19 = addf %17, %18 : f32
      affine.store %19, %3[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
    }
    %15:2 = krnl.define_loops 2
    krnl.iterate(%15#0, %15#1) with (%15#0 -> %arg3 = 0 to 10, %15#1 -> %arg4 = 0 to 5) {
      affine.store %cst, %2[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      %17 = krnl.define_loops 1
      krnl.iterate(%17) with (%17 -> %arg5 = 0 to 5) {
        %18 = affine.load %3[symbol(%arg3), symbol(%arg5)] : memref<10x5xf32>
        %19 = affine.load %arg1[symbol(%arg5), symbol(%arg4)] : memref<5x5xf32>
        %20 = affine.load %2[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
        %21 = mulf %18, %19 : f32
        %22 = addf %20, %21 : f32
        affine.store %22, %2[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      }
    }
    %16:2 = krnl.define_loops 2
    krnl.iterate(%16#0, %16#1) with (%16#0 -> %arg3 = 0 to 10, %16#1 -> %arg4 = 0 to 5) {
      %17 = affine.load %2[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      %18 = affine.load %arg0[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      %19 = addf %17, %18 : f32
      affine.store %19, %0[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
    }
    dealloc %1 : memref<1400xi8>
    return %0 : memref<10x5xf32>

  // CHECK-LABEL: analysis_krnl_memcpy
  // CHECK: [[C0:%.+]] = constant 0 : i64
  // CHECK: [[C200:%.+]] = constant 200 : i64
  // CHECK: [[MEMPOOL:%.+]] = alloc() : memref<400xi8>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C0]]) : (memref<400xi8>, i64) -> memref<10x5xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C200]]) : (memref<400xi8>, i64) -> memref<10x5xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C0]]) : (memref<400xi8>, i64) -> memref<10x5xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C200]]) : (memref<400xi8>, i64) -> memref<5x10xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C0]]) : (memref<400xi8>, i64) -> memref<5x10xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C200]]) : (memref<400xi8>, i64) -> memref<10x5xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C0]]) : (memref<400xi8>, i64) -> memref<10x5xf32>
}

// -----

/// 4. Test dependency analysis for MemRefs with intersecting live ranges.
/// %8 now has bigger live range that leads to less reuse.
/// %8 cannot share a slot with %2, %3, and %7 since there is a direct load/store relationship between them.
/// %8 cannot share a slot with %5 and %6 because their live ranges intersect.
func @analysis_krnl_memcpy(%arg0: memref<10x5xf32>, %arg1: memref<5x5xf32>, %arg2: memref<10x10xf32>) -> memref<10x5xf32> {
    %cst = constant 0.000000e+00 : f32
    %c0_i64 = constant 0 : i64
    %c1200_i64 = constant 1200 : i64
    %c1000_i64 = constant 1000 : i64
    %c800_i64 = constant 800 : i64
    %c600_i64 = constant 600 : i64
    %c400_i64 = constant 400 : i64
    %c200_i64 = constant 200 : i64
    %0 = alloc() : memref<10x5xf32>
    %1 = alloc() : memref<1400xi8>
    %2 = "krnl.getref"(%1, %c1200_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %3 = "krnl.getref"(%1, %c1000_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %4 = "krnl.getref"(%1, %c800_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %5 = "krnl.getref"(%1, %c600_i64) : (memref<1400xi8>, i64) -> memref<5x10xf32>
    %6 = "krnl.getref"(%1, %c400_i64) : (memref<1400xi8>, i64) -> memref<5x10xf32>
    %7 = "krnl.getref"(%1, %c200_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %8 = "krnl.getref"(%1, %c0_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %9:2 = krnl.define_loops 2
    krnl.iterate(%9#0, %9#1) with (%9#0 -> %arg3 = 0 to 10, %9#1 -> %arg4 = 0 to 5) {
      affine.store %cst, %8[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      %17 = krnl.define_loops 1
      krnl.iterate(%17) with (%17 -> %arg5 = 0 to 5) {
        %18 = affine.load %arg0[symbol(%arg3), symbol(%arg5)] : memref<10x5xf32>
        %19 = affine.load %arg1[symbol(%arg5), symbol(%arg4)] : memref<5x5xf32>
        %20 = affine.load %8[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
        %21 = mulf %18, %19 : f32
        %22 = addf %20, %21 : f32
        affine.store %22, %8[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      }
    }
    %10:2 = krnl.define_loops 2
    krnl.iterate(%10#0, %10#1) with (%10#0 -> %arg3 = 0 to 10, %10#1 -> %arg4 = 0 to 5) {
      %17 = affine.load %arg0[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      %18 = affine.load %8[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      %19 = addf %17, %18 : f32
      affine.store %19, %7[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
    }
    "krnl.memcpy"(%6, %7, %c200_i64) : (memref<5x10xf32>, memref<10x5xf32>, i64) -> ()
    %12:2 = krnl.define_loops 2
    krnl.iterate(%12#0, %12#1) with (%12#0 -> %arg3 = 0 to 5, %12#1 -> %arg4 = 0 to 10) {
      affine.store %cst, %5[symbol(%arg3), symbol(%arg4)] : memref<5x10xf32>
      %17 = krnl.define_loops 1
      krnl.iterate(%17) with (%17 -> %arg5 = 0 to 10) {
        %18 = affine.load %6[symbol(%arg3), symbol(%arg5)] : memref<5x10xf32>
        %19 = affine.load %arg2[symbol(%arg5), symbol(%arg4)] : memref<10x10xf32>
        %20 = affine.load %5[symbol(%arg3), symbol(%arg4)] : memref<5x10xf32>
        %21 = mulf %18, %19 : f32
        %22 = addf %20, %21 : f32
        affine.store %22, %5[symbol(%arg3), symbol(%arg4)] : memref<5x10xf32>
      }
    }
    "krnl.memcpy"(%4, %5, %c200_i64) : (memref<10x5xf32>, memref<5x10xf32>, i64) -> ()
    %14:2 = krnl.define_loops 2
    krnl.iterate(%14#0, %14#1) with (%14#0 -> %arg3 = 0 to 10, %14#1 -> %arg4 = 0 to 5) {
      %17 = affine.load %4[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      /// Change this to use %8 instead of an arg argument.
      %18 = affine.load %8[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      %19 = addf %17, %18 : f32
      affine.store %19, %3[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
    }
    %15:2 = krnl.define_loops 2
    krnl.iterate(%15#0, %15#1) with (%15#0 -> %arg3 = 0 to 10, %15#1 -> %arg4 = 0 to 5) {
      affine.store %cst, %2[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      %17 = krnl.define_loops 1
      krnl.iterate(%17) with (%17 -> %arg5 = 0 to 5) {
        %18 = affine.load %3[symbol(%arg3), symbol(%arg5)] : memref<10x5xf32>
        /// Add new val that uses %8.
        %newVal = affine.load %8[symbol(%arg3), symbol(%arg5)] : memref<10x5xf32>
        %newAdd = addf %18, %newVal : f32
        %19 = affine.load %arg1[symbol(%arg5), symbol(%arg4)] : memref<5x5xf32>
        %20 = affine.load %2[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
        %21 = mulf %18, %19 : f32
        %22 = addf %20, %21 : f32
        %newStoreVal = addf %22, %newAdd : f32
        affine.store %newStoreVal, %2[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      }
    }
    %16:2 = krnl.define_loops 2
    krnl.iterate(%16#0, %16#1) with (%16#0 -> %arg3 = 0 to 10, %16#1 -> %arg4 = 0 to 5) {
      %17 = affine.load %2[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      %18 = affine.load %arg0[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      %19 = addf %17, %18 : f32
      affine.store %19, %0[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
    }
    dealloc %1 : memref<1400xi8>
    return %0 : memref<10x5xf32>

  // CHECK-LABEL: analysis_krnl_memcpy
  // CHECK: [[C0:%.+]] = constant 0 : i64
  // CHECK: [[C200:%.+]] = constant 200 : i64
  // CHECK: [[C400:%.+]] = constant 400 : i64
  // CHECK: [[C600:%.+]] = constant 600 : i64
  // CHECK: [[MEMPOOL:%.+]] = alloc() : memref<800xi8>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C0]]) : (memref<800xi8>, i64) -> memref<10x5xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C200]]) : (memref<800xi8>, i64) -> memref<10x5xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C0]]) : (memref<800xi8>, i64) -> memref<10x5xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C400]]) : (memref<800xi8>, i64) -> memref<5x10xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C200]]) : (memref<800xi8>, i64) -> memref<5x10xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C0]]) : (memref<800xi8>, i64) -> memref<10x5xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C600]]) : (memref<800xi8>, i64) -> memref<10x5xf32>
}

/// 5. Test for MemRefs which share a krnl.iterate outermost loop. These MemRefs cannot
/// share the same slot.
///    %4 and %2 have disjoint live ranges.
///    %4 and %2 cannot share a slot because they both are under the same outermost krnl.iterate.
func @multiple_shaped_memrefs(%arg0: memref<10x5xf32>, %arg1: memref<5x5xf32>, %arg2: memref<10x10xf32>) -> memref<10x5xf32> {
    %cst = constant 0.000000e+00 : f32
    %c0_i64 = constant 0 : i64
    %c1200_i64 = constant 1200 : i64
    %c1000_i64 = constant 1000 : i64
    %c800_i64 = constant 800 : i64
    %c600_i64 = constant 600 : i64
    %c400_i64 = constant 400 : i64
    %c200_i64 = constant 200 : i64
    %0 = alloc() : memref<10x5xf32>
    %1 = alloc() : memref<1400xi8>
    %2 = "krnl.getref"(%1, %c1200_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %3 = "krnl.getref"(%1, %c1000_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %4 = "krnl.getref"(%1, %c800_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %5 = "krnl.getref"(%1, %c600_i64) : (memref<1400xi8>, i64) -> memref<5x10xf32>
    %6 = "krnl.getref"(%1, %c400_i64) : (memref<1400xi8>, i64) -> memref<5x10xf32>
    %7 = "krnl.getref"(%1, %c200_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %8 = "krnl.getref"(%1, %c0_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %9:2 = krnl.define_loops 2
    krnl.iterate(%9#0, %9#1) with (%9#0 -> %arg3 = 0 to 10, %9#1 -> %arg4 = 0 to 5) {
      affine.store %cst, %8[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      %17 = krnl.define_loops 1
      krnl.iterate(%17) with (%17 -> %arg5 = 0 to 5) {
        %18 = affine.load %arg0[symbol(%arg3), symbol(%arg5)] : memref<10x5xf32>
        %19 = affine.load %arg1[symbol(%arg5), symbol(%arg4)] : memref<5x5xf32>
        %20 = affine.load %8[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
        %21 = mulf %18, %19 : f32
        %22 = addf %20, %21 : f32
        affine.store %22, %8[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      }
    }
    %10:2 = krnl.define_loops 2
    krnl.iterate(%10#0, %10#1) with (%10#0 -> %arg3 = 0 to 10, %10#1 -> %arg4 = 0 to 5) {
      %17 = affine.load %arg0[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      %18 = affine.load %8[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      %19 = addf %17, %18 : f32
      affine.store %19, %7[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
    }
    %11:2 = krnl.define_loops 2
    krnl.iterate(%11#0, %11#1) with (%11#0 -> %arg3 = 0 to 10, %11#1 -> %arg4 = 0 to 5) {
      %17 = affine.load %7[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      affine.store %17, %6[symbol(%arg4), symbol(%arg3)] : memref<5x10xf32>
    }
    %12:2 = krnl.define_loops 2
    krnl.iterate(%12#0, %12#1) with (%12#0 -> %arg3 = 0 to 5, %12#1 -> %arg4 = 0 to 10) {
      affine.store %cst, %5[symbol(%arg3), symbol(%arg4)] : memref<5x10xf32>
      %17 = krnl.define_loops 1
      krnl.iterate(%17) with (%17 -> %arg5 = 0 to 10) {
        %18 = affine.load %6[symbol(%arg3), symbol(%arg5)] : memref<5x10xf32>
        %19 = affine.load %arg2[symbol(%arg5), symbol(%arg4)] : memref<10x10xf32>
        %20 = affine.load %5[symbol(%arg3), symbol(%arg4)] : memref<5x10xf32>
        %21 = mulf %18, %19 : f32
        %22 = addf %20, %21 : f32
        affine.store %22, %5[symbol(%arg3), symbol(%arg4)] : memref<5x10xf32>
      }
    }
    %14:2 = krnl.define_loops 2
    krnl.iterate(%14#0, %14#1) with (%14#0 -> %arg3 = 0 to 10, %14#1 -> %arg4 = 0 to 5) {
      %17 = affine.load %5[symbol(%arg4), symbol(%arg3)] : memref<5x10xf32>
      %18 = affine.load %arg0[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      %19 = addf %17, %18 : f32
      affine.store %19, %3[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
    }
    %15:2 = krnl.define_loops 2
    krnl.iterate(%15#0, %15#1) with (%15#0 -> %arg3 = 0 to 10, %15#1 -> %arg4 = 0 to 5) {
      affine.store %cst, %2[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      %17 = krnl.define_loops 1
      krnl.iterate(%17) with (%17 -> %arg5 = 0 to 5) {
        %18 = affine.load %3[symbol(%arg3), symbol(%arg5)] : memref<10x5xf32>
        %19 = affine.load %arg1[symbol(%arg5), symbol(%arg4)] : memref<5x5xf32>
        %20 = affine.load %2[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
        %21 = mulf %18, %19 : f32
        %22 = addf %20, %21 : f32
        affine.store %22, %2[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      }
      /// Newly added code.
      %newLoad = affine.load %3[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      affine.store %newLoad, %4[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
    }
    %16:2 = krnl.define_loops 2
    krnl.iterate(%16#0, %16#1) with (%16#0 -> %arg3 = 0 to 10, %16#1 -> %arg4 = 0 to 5) {
      %17 = affine.load %4[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      %18 = affine.load %arg0[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
      %19 = addf %17, %18 : f32
      affine.store %19, %0[symbol(%arg3), symbol(%arg4)] : memref<10x5xf32>
    }
    dealloc %1 : memref<1400xi8>
    return %0 : memref<10x5xf32>

  // CHECK-LABEL: multiple_shaped_memrefs
  // CHECK: [[C0:%.+]] = constant 0 : i64
  // CHECK: [[C200:%.+]] = constant 200 : i64
  // CHECK: [[C400:%.+]] = constant 400 : i64
  // CHECK: [[MEMPOOL:%.+]] = alloc() : memref<600xi8>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C0]]) : (memref<600xi8>, i64) -> memref<10x5xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C200]]) : (memref<600xi8>, i64) -> memref<10x5xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C400]]) : (memref<600xi8>, i64) -> memref<10x5xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C0]]) : (memref<600xi8>, i64) -> memref<5x10xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C400]]) : (memref<600xi8>, i64) -> memref<5x10xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C200]]) : (memref<600xi8>, i64) -> memref<10x5xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C0]]) : (memref<600xi8>, i64) -> memref<10x5xf32>
}
