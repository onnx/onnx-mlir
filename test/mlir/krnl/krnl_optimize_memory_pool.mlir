// RUN: onnx-mlir-opt -O3 -allow-unregistered-dialect --optimize-memory-pools --canonicalize %s -split-input-file | FileCheck %s

/// 1. Base case where we have a single-chain workflow.
func.func @single_chain_dataflow(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>) -> memref<10x10xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0_i64 = arith.constant 0 : i64
  %c1600_i64 = arith.constant 1600 : i64
  %c1200_i64 = arith.constant 1200 : i64
  %c800_i64 = arith.constant 800 : i64
  %c400_i64 = arith.constant 400 : i64
  %0 = memref.alloc() : memref<10x10xf32>
  %1 = memref.alloc() : memref<2000xi8>
  %2 = "krnl.getref"(%1, %c1600_i64) : (memref<2000xi8>, i64) -> memref<10x10xf32>
  %3 = "krnl.getref"(%1, %c1200_i64) : (memref<2000xi8>, i64) -> memref<10x10xf32>
  %4 = "krnl.getref"(%1, %c800_i64) : (memref<2000xi8>, i64) -> memref<10x10xf32>
  %5 = "krnl.getref"(%1, %c400_i64) : (memref<2000xi8>, i64) -> memref<10x10xf32>
  %6 = "krnl.getref"(%1, %c0_i64) : (memref<2000xi8>, i64) -> memref<10x10xf32>
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
  memref.dealloc %1 : memref<2000xi8>
  return %0 : memref<10x10xf32>

  // CHECK-LABEL: single_chain_dataflow
  // CHECK-DAG: [[C0:%.+]] = arith.constant 0 : i64
  // CHECK-DAG: [[C400:%.+]] = arith.constant 400 : i64
  // CHECK-DAG: [[MEMPOOL:%.+]] = memref.alloc() : memref<800xi8>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C0]]) : (memref<800xi8>, i64) -> memref<10x10xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C400]]) : (memref<800xi8>, i64) -> memref<10x10xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C0]]) : (memref<800xi8>, i64) -> memref<10x10xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C400]]) : (memref<800xi8>, i64) -> memref<10x10xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C0]]) : (memref<800xi8>, i64) -> memref<10x10xf32>
}

// -----

/// 2. Test for MemRefs with different shapes that can share the same slot.
func.func @multiple_shaped_memrefs(%arg0: memref<10x5xf32>, %arg1: memref<5x5xf32>, %arg2: memref<10x10xf32>) -> memref<10x5xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %c1200_i64 = arith.constant 1200 : i64
    %c1000_i64 = arith.constant 1000 : i64
    %c800_i64 = arith.constant 800 : i64
    %c600_i64 = arith.constant 600 : i64
    %c400_i64 = arith.constant 400 : i64
    %c200_i64 = arith.constant 200 : i64
    %0 = memref.alloc() : memref<10x5xf32>
    %1 = memref.alloc() : memref<1400xi8>
    %2 = "krnl.getref"(%1, %c1200_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %3 = "krnl.getref"(%1, %c1000_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %4 = "krnl.getref"(%1, %c800_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %5 = "krnl.getref"(%1, %c600_i64) : (memref<1400xi8>, i64) -> memref<5x10xf32>
    %6 = "krnl.getref"(%1, %c400_i64) : (memref<1400xi8>, i64) -> memref<5x10xf32>
    %7 = "krnl.getref"(%1, %c200_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %8 = "krnl.getref"(%1, %c0_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %9:2 = krnl.define_loops 2
    krnl.iterate(%9#0, %9#1) with (%9#0 -> %arg3 = 0 to 10, %9#1 -> %arg4 = 0 to 5) {
      krnl.store %cst, %8[%arg3, %arg4] : memref<10x5xf32>
      %17 = krnl.define_loops 1
      krnl.iterate(%17) with (%17 -> %arg5 = 0 to 5) {
        %18 = krnl.load %arg0[%arg3, %arg5] : memref<10x5xf32>
        %19 = krnl.load %arg1[%arg5, %arg4] : memref<5x5xf32>
        %20 = krnl.load %8[%arg3, %arg4] : memref<10x5xf32>
        %21 = arith.mulf %18, %19 : f32
        %22 = arith.addf %20, %21 : f32
        krnl.store %22, %8[%arg3, %arg4] : memref<10x5xf32>
      }
    }
    %10:2 = krnl.define_loops 2
    krnl.iterate(%10#0, %10#1) with (%10#0 -> %arg3 = 0 to 10, %10#1 -> %arg4 = 0 to 5) {
      %17 = krnl.load %arg0[%arg3, %arg4] : memref<10x5xf32>
      %18 = krnl.load %8[%arg3, %arg4] : memref<10x5xf32>
      %19 = arith.addf %17, %18 : f32
      krnl.store %19, %7[%arg3, %arg4] : memref<10x5xf32>
    }
    %11:2 = krnl.define_loops 2
    krnl.iterate(%11#0, %11#1) with (%11#0 -> %arg3 = 0 to 10, %11#1 -> %arg4 = 0 to 5) {
      %17 = krnl.load %7[%arg3, %arg4] : memref<10x5xf32>
      krnl.store %17, %6[%arg4, %arg3] : memref<5x10xf32>
    }
    %12:2 = krnl.define_loops 2
    krnl.iterate(%12#0, %12#1) with (%12#0 -> %arg3 = 0 to 5, %12#1 -> %arg4 = 0 to 10) {
      krnl.store %cst, %5[%arg3, %arg4] : memref<5x10xf32>
      %17 = krnl.define_loops 1
      krnl.iterate(%17) with (%17 -> %arg5 = 0 to 10) {
        %18 = krnl.load %6[%arg3, %arg5] : memref<5x10xf32>
        %19 = krnl.load %arg2[%arg5, %arg4] : memref<10x10xf32>
        %20 = krnl.load %5[%arg3, %arg4] : memref<5x10xf32>
        %21 = arith.mulf %18, %19 : f32
        %22 = arith.addf %20, %21 : f32
        krnl.store %22, %5[%arg3, %arg4] : memref<5x10xf32>
      }
    }
    %13:2 = krnl.define_loops 2
    krnl.iterate(%13#0, %13#1) with (%13#0 -> %arg3 = 0 to 5, %13#1 -> %arg4 = 0 to 10) {
      %17 = krnl.load %5[%arg3, %arg4] : memref<5x10xf32>
      krnl.store %17, %4[%arg4, %arg3] : memref<10x5xf32>
    }
    %14:2 = krnl.define_loops 2
    krnl.iterate(%14#0, %14#1) with (%14#0 -> %arg3 = 0 to 10, %14#1 -> %arg4 = 0 to 5) {
      %17 = krnl.load %4[%arg3, %arg4] : memref<10x5xf32>
      %18 = krnl.load %arg0[%arg3, %arg4] : memref<10x5xf32>
      %19 = arith.addf %17, %18 : f32
      krnl.store %19, %3[%arg3, %arg4] : memref<10x5xf32>
    }
    %15:2 = krnl.define_loops 2
    krnl.iterate(%15#0, %15#1) with (%15#0 -> %arg3 = 0 to 10, %15#1 -> %arg4 = 0 to 5) {
      krnl.store %cst, %2[%arg3, %arg4] : memref<10x5xf32>
      %17 = krnl.define_loops 1
      krnl.iterate(%17) with (%17 -> %arg5 = 0 to 5) {
        %18 = krnl.load %3[%arg3, %arg5] : memref<10x5xf32>
        %19 = krnl.load %arg1[%arg5, %arg4] : memref<5x5xf32>
        %20 = krnl.load %2[%arg3, %arg4] : memref<10x5xf32>
        %21 = arith.mulf %18, %19 : f32
        %22 = arith.addf %20, %21 : f32
        krnl.store %22, %2[%arg3, %arg4] : memref<10x5xf32>
      }
    }
    %16:2 = krnl.define_loops 2
    krnl.iterate(%16#0, %16#1) with (%16#0 -> %arg3 = 0 to 10, %16#1 -> %arg4 = 0 to 5) {
      %17 = krnl.load %2[%arg3, %arg4] : memref<10x5xf32>
      %18 = krnl.load %arg0[%arg3, %arg4] : memref<10x5xf32>
      %19 = arith.addf %17, %18 : f32
      krnl.store %19, %0[%arg3, %arg4] : memref<10x5xf32>
    }
    memref.dealloc %1 : memref<1400xi8>
    return %0 : memref<10x5xf32>

  // CHECK-LABEL: multiple_shaped_memrefs
  // CHECK-DAG: [[C0:%.+]] = arith.constant 0 : i64
  // CHECK-DAG: [[C200:%.+]] = arith.constant 200 : i64
  // CHECK-DAG: [[MEMPOOL:%.+]] = memref.alloc() : memref<400xi8>
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
func.func @analysis_krnl_memcpy(%arg0: memref<10x5xf32>, %arg1: memref<5x5xf32>, %arg2: memref<10x10xf32>) -> memref<10x5xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %c1200_i64 = arith.constant 1200 : i64
    %c1000_i64 = arith.constant 1000 : i64
    %c800_i64 = arith.constant 800 : i64
    %c600_i64 = arith.constant 600 : i64
    %c400_i64 = arith.constant 400 : i64
    %c200_i64 = arith.constant 200 : i64
    %0 = memref.alloc() : memref<10x5xf32>
    %1 = memref.alloc() : memref<1400xi8>
    %2 = "krnl.getref"(%1, %c1200_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %3 = "krnl.getref"(%1, %c1000_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %4 = "krnl.getref"(%1, %c800_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %5 = "krnl.getref"(%1, %c600_i64) : (memref<1400xi8>, i64) -> memref<5x10xf32>
    %6 = "krnl.getref"(%1, %c400_i64) : (memref<1400xi8>, i64) -> memref<5x10xf32>
    %7 = "krnl.getref"(%1, %c200_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %8 = "krnl.getref"(%1, %c0_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %9:2 = krnl.define_loops 2
    krnl.iterate(%9#0, %9#1) with (%9#0 -> %arg3 = 0 to 10, %9#1 -> %arg4 = 0 to 5) {
      krnl.store %cst, %8[%arg3, %arg4] : memref<10x5xf32>
      %17 = krnl.define_loops 1
      krnl.iterate(%17) with (%17 -> %arg5 = 0 to 5) {
        %18 = krnl.load %arg0[%arg3, %arg5] : memref<10x5xf32>
        %19 = krnl.load %arg1[%arg5, %arg4] : memref<5x5xf32>
        %20 = krnl.load %8[%arg3, %arg4] : memref<10x5xf32>
        %21 = arith.mulf %18, %19 : f32
        %22 = arith.addf %20, %21 : f32
        krnl.store %22, %8[%arg3, %arg4] : memref<10x5xf32>
      }
    }
    %10:2 = krnl.define_loops 2
    krnl.iterate(%10#0, %10#1) with (%10#0 -> %arg3 = 0 to 10, %10#1 -> %arg4 = 0 to 5) {
      %17 = krnl.load %arg0[%arg3, %arg4] : memref<10x5xf32>
      %18 = krnl.load %8[%arg3, %arg4] : memref<10x5xf32>
      %19 = arith.addf %17, %18 : f32
      krnl.store %19, %7[%arg3, %arg4] : memref<10x5xf32>
    }
    "krnl.memcpy"(%6, %7, %c200_i64) : (memref<5x10xf32>, memref<10x5xf32>, i64) -> ()
    %12:2 = krnl.define_loops 2
    krnl.iterate(%12#0, %12#1) with (%12#0 -> %arg3 = 0 to 5, %12#1 -> %arg4 = 0 to 10) {
      krnl.store %cst, %5[%arg3, %arg4] : memref<5x10xf32>
      %17 = krnl.define_loops 1
      krnl.iterate(%17) with (%17 -> %arg5 = 0 to 10) {
        %18 = krnl.load %6[%arg3, %arg5] : memref<5x10xf32>
        %19 = krnl.load %arg2[%arg5, %arg4] : memref<10x10xf32>
        %20 = krnl.load %5[%arg3, %arg4] : memref<5x10xf32>
        %21 = arith.mulf %18, %19 : f32
        %22 = arith.addf %20, %21 : f32
        krnl.store %22, %5[%arg3, %arg4] : memref<5x10xf32>
      }
    }
    "krnl.memcpy"(%4, %5, %c200_i64) : (memref<10x5xf32>, memref<5x10xf32>, i64) -> ()
    %14:2 = krnl.define_loops 2
    krnl.iterate(%14#0, %14#1) with (%14#0 -> %arg3 = 0 to 10, %14#1 -> %arg4 = 0 to 5) {
      %17 = krnl.load %4[%arg3, %arg4] : memref<10x5xf32>
      %18 = krnl.load %arg0[%arg3, %arg4] : memref<10x5xf32>
      %19 = arith.addf %17, %18 : f32
      krnl.store %19, %3[%arg3, %arg4] : memref<10x5xf32>
    }
    %15:2 = krnl.define_loops 2
    krnl.iterate(%15#0, %15#1) with (%15#0 -> %arg3 = 0 to 10, %15#1 -> %arg4 = 0 to 5) {
      krnl.store %cst, %2[%arg3, %arg4] : memref<10x5xf32>
      %17 = krnl.define_loops 1
      krnl.iterate(%17) with (%17 -> %arg5 = 0 to 5) {
        %18 = krnl.load %3[%arg3, %arg5] : memref<10x5xf32>
        %19 = krnl.load %arg1[%arg5, %arg4] : memref<5x5xf32>
        %20 = krnl.load %2[%arg3, %arg4] : memref<10x5xf32>
        %21 = arith.mulf %18, %19 : f32
        %22 = arith.addf %20, %21 : f32
        krnl.store %22, %2[%arg3, %arg4] : memref<10x5xf32>
      }
    }
    %16:2 = krnl.define_loops 2
    krnl.iterate(%16#0, %16#1) with (%16#0 -> %arg3 = 0 to 10, %16#1 -> %arg4 = 0 to 5) {
      %17 = krnl.load %2[%arg3, %arg4] : memref<10x5xf32>
      %18 = krnl.load %arg0[%arg3, %arg4] : memref<10x5xf32>
      %19 = arith.addf %17, %18 : f32
      krnl.store %19, %0[%arg3, %arg4] : memref<10x5xf32>
    }
    memref.dealloc %1 : memref<1400xi8>
    return %0 : memref<10x5xf32>

  // CHECK-LABEL: analysis_krnl_memcpy
  // CHECK-DAG: [[C0:%.+]] = arith.constant 0 : i64
  // CHECK-DAG: [[C200:%.+]] = arith.constant 200 : i64
  // CHECK-DAG: [[MEMPOOL:%.+]] = memref.alloc() : memref<400xi8>
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
func.func @analysis_krnl_memcpy(%arg0: memref<10x5xf32>, %arg1: memref<5x5xf32>, %arg2: memref<10x10xf32>) -> memref<10x5xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %c1200_i64 = arith.constant 1200 : i64
    %c1000_i64 = arith.constant 1000 : i64
    %c800_i64 = arith.constant 800 : i64
    %c600_i64 = arith.constant 600 : i64
    %c400_i64 = arith.constant 400 : i64
    %c200_i64 = arith.constant 200 : i64
    %0 = memref.alloc() : memref<10x5xf32>
    %1 = memref.alloc() : memref<1400xi8>
    %2 = "krnl.getref"(%1, %c1200_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %3 = "krnl.getref"(%1, %c1000_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %4 = "krnl.getref"(%1, %c800_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %5 = "krnl.getref"(%1, %c600_i64) : (memref<1400xi8>, i64) -> memref<5x10xf32>
    %6 = "krnl.getref"(%1, %c400_i64) : (memref<1400xi8>, i64) -> memref<5x10xf32>
    %7 = "krnl.getref"(%1, %c200_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %8 = "krnl.getref"(%1, %c0_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %9:2 = krnl.define_loops 2
    krnl.iterate(%9#0, %9#1) with (%9#0 -> %arg3 = 0 to 10, %9#1 -> %arg4 = 0 to 5) {
      krnl.store %cst, %8[%arg3, %arg4] : memref<10x5xf32>
      %17 = krnl.define_loops 1
      krnl.iterate(%17) with (%17 -> %arg5 = 0 to 5) {
        %18 = krnl.load %arg0[%arg3, %arg5] : memref<10x5xf32>
        %19 = krnl.load %arg1[%arg5, %arg4] : memref<5x5xf32>
        %20 = krnl.load %8[%arg3, %arg4] : memref<10x5xf32>
        %21 = arith.mulf %18, %19 : f32
        %22 = arith.addf %20, %21 : f32
        krnl.store %22, %8[%arg3, %arg4] : memref<10x5xf32>
      }
    }
    %10:2 = krnl.define_loops 2
    krnl.iterate(%10#0, %10#1) with (%10#0 -> %arg3 = 0 to 10, %10#1 -> %arg4 = 0 to 5) {
      %17 = krnl.load %arg0[%arg3, %arg4] : memref<10x5xf32>
      %18 = krnl.load %8[%arg3, %arg4] : memref<10x5xf32>
      %19 = arith.addf %17, %18 : f32
      krnl.store %19, %7[%arg3, %arg4] : memref<10x5xf32>
    }
    "krnl.memcpy"(%6, %7, %c200_i64) : (memref<5x10xf32>, memref<10x5xf32>, i64) -> ()
    %12:2 = krnl.define_loops 2
    krnl.iterate(%12#0, %12#1) with (%12#0 -> %arg3 = 0 to 5, %12#1 -> %arg4 = 0 to 10) {
      krnl.store %cst, %5[%arg3, %arg4] : memref<5x10xf32>
      %17 = krnl.define_loops 1
      krnl.iterate(%17) with (%17 -> %arg5 = 0 to 10) {
        %18 = krnl.load %6[%arg3, %arg5] : memref<5x10xf32>
        %19 = krnl.load %arg2[%arg5, %arg4] : memref<10x10xf32>
        %20 = krnl.load %5[%arg3, %arg4] : memref<5x10xf32>
        %21 = arith.mulf %18, %19 : f32
        %22 = arith.addf %20, %21 : f32
        krnl.store %22, %5[%arg3, %arg4] : memref<5x10xf32>
      }
    }
    "krnl.memcpy"(%4, %5, %c200_i64) : (memref<10x5xf32>, memref<5x10xf32>, i64) -> ()
    %14:2 = krnl.define_loops 2
    krnl.iterate(%14#0, %14#1) with (%14#0 -> %arg3 = 0 to 10, %14#1 -> %arg4 = 0 to 5) {
      %17 = krnl.load %4[%arg3, %arg4] : memref<10x5xf32>
      /// Change this to use %8 instead of an arg argument.
      %18 = krnl.load %8[%arg3, %arg4] : memref<10x5xf32>
      %19 = arith.addf %17, %18 : f32
      krnl.store %19, %3[%arg3, %arg4] : memref<10x5xf32>
    }
    %15:2 = krnl.define_loops 2
    krnl.iterate(%15#0, %15#1) with (%15#0 -> %arg3 = 0 to 10, %15#1 -> %arg4 = 0 to 5) {
      krnl.store %cst, %2[%arg3, %arg4] : memref<10x5xf32>
      %17 = krnl.define_loops 1
      krnl.iterate(%17) with (%17 -> %arg5 = 0 to 5) {
        %18 = krnl.load %3[%arg3, %arg5] : memref<10x5xf32>
        /// Add new val that uses %8.
        %newVal = krnl.load %8[%arg3, %arg5] : memref<10x5xf32>
        %newAdd = arith.addf %18, %newVal : f32
        %19 = krnl.load %arg1[%arg5, %arg4] : memref<5x5xf32>
        %20 = krnl.load %2[%arg3, %arg4] : memref<10x5xf32>
        %21 = arith.mulf %18, %19 : f32
        %22 = arith.addf %20, %21 : f32
        %newStoreVal = arith.addf %22, %newAdd : f32
        krnl.store %newStoreVal, %2[%arg3, %arg4] : memref<10x5xf32>
      }
    }
    %16:2 = krnl.define_loops 2
    krnl.iterate(%16#0, %16#1) with (%16#0 -> %arg3 = 0 to 10, %16#1 -> %arg4 = 0 to 5) {
      %17 = krnl.load %2[%arg3, %arg4] : memref<10x5xf32>
      %18 = krnl.load %arg0[%arg3, %arg4] : memref<10x5xf32>
      %19 = arith.addf %17, %18 : f32
      krnl.store %19, %0[%arg3, %arg4] : memref<10x5xf32>
    }
    memref.dealloc %1 : memref<1400xi8>
    return %0 : memref<10x5xf32>

  // CHECK-LABEL: analysis_krnl_memcpy
  // CHECK-DAG: [[C0:%.+]] = arith.constant 0 : i64
  // CHECK-DAG: [[C200:%.+]] = arith.constant 200 : i64
  // CHECK-DAG: [[C400:%.+]] = arith.constant 400 : i64
  // CHECK-DAG: [[C600:%.+]] = arith.constant 600 : i64
  // CHECK-DAG: [[MEMPOOL:%.+]] = memref.alloc() : memref<800xi8>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C0]]) : (memref<800xi8>, i64) -> memref<10x5xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C200]]) : (memref<800xi8>, i64) -> memref<10x5xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C0]]) : (memref<800xi8>, i64) -> memref<10x5xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C400]]) : (memref<800xi8>, i64) -> memref<5x10xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C200]]) : (memref<800xi8>, i64) -> memref<5x10xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C0]]) : (memref<800xi8>, i64) -> memref<10x5xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C600]]) : (memref<800xi8>, i64) -> memref<10x5xf32>
}

// -----

/// 5. Test for MemRefs which share a krnl.iterate outermost loop. These MemRefs cannot
/// share the same slot.
///    %4 and %2 have disjoint live ranges.
///    %4 and %2 cannot share a slot because they both are under the same outermost krnl.iterate.
func.func @multiple_shaped_memrefs(%arg0: memref<10x5xf32>, %arg1: memref<5x5xf32>, %arg2: memref<10x10xf32>) -> memref<10x5xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %c1200_i64 = arith.constant 1200 : i64
    %c1000_i64 = arith.constant 1000 : i64
    %c800_i64 = arith.constant 800 : i64
    %c600_i64 = arith.constant 600 : i64
    %c400_i64 = arith.constant 400 : i64
    %c200_i64 = arith.constant 200 : i64
    %0 = memref.alloc() : memref<10x5xf32>
    %1 = memref.alloc() : memref<1400xi8>
    %2 = "krnl.getref"(%1, %c1200_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %3 = "krnl.getref"(%1, %c1000_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %4 = "krnl.getref"(%1, %c800_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %5 = "krnl.getref"(%1, %c600_i64) : (memref<1400xi8>, i64) -> memref<5x10xf32>
    %6 = "krnl.getref"(%1, %c400_i64) : (memref<1400xi8>, i64) -> memref<5x10xf32>
    %7 = "krnl.getref"(%1, %c200_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %8 = "krnl.getref"(%1, %c0_i64) : (memref<1400xi8>, i64) -> memref<10x5xf32>
    %9:2 = krnl.define_loops 2
    krnl.iterate(%9#0, %9#1) with (%9#0 -> %arg3 = 0 to 10, %9#1 -> %arg4 = 0 to 5) {
      krnl.store %cst, %8[%arg3, %arg4] : memref<10x5xf32>
      %17 = krnl.define_loops 1
      krnl.iterate(%17) with (%17 -> %arg5 = 0 to 5) {
        %18 = krnl.load %arg0[%arg3, %arg5] : memref<10x5xf32>
        %19 = krnl.load %arg1[%arg5, %arg4] : memref<5x5xf32>
        %20 = krnl.load %8[%arg3, %arg4] : memref<10x5xf32>
        %21 = arith.mulf %18, %19 : f32
        %22 = arith.addf %20, %21 : f32
        krnl.store %22, %8[%arg3, %arg4] : memref<10x5xf32>
      }
    }
    %10:2 = krnl.define_loops 2
    krnl.iterate(%10#0, %10#1) with (%10#0 -> %arg3 = 0 to 10, %10#1 -> %arg4 = 0 to 5) {
      %17 = krnl.load %arg0[%arg3, %arg4] : memref<10x5xf32>
      %18 = krnl.load %8[%arg3, %arg4] : memref<10x5xf32>
      %19 = arith.addf %17, %18 : f32
      krnl.store %19, %7[%arg3, %arg4] : memref<10x5xf32>
    }
    %11:2 = krnl.define_loops 2
    krnl.iterate(%11#0, %11#1) with (%11#0 -> %arg3 = 0 to 10, %11#1 -> %arg4 = 0 to 5) {
      %17 = krnl.load %7[%arg3, %arg4] : memref<10x5xf32>
      krnl.store %17, %6[%arg4, %arg3] : memref<5x10xf32>
    }
    %12:2 = krnl.define_loops 2
    krnl.iterate(%12#0, %12#1) with (%12#0 -> %arg3 = 0 to 5, %12#1 -> %arg4 = 0 to 10) {
      krnl.store %cst, %5[%arg3, %arg4] : memref<5x10xf32>
      %17 = krnl.define_loops 1
      krnl.iterate(%17) with (%17 -> %arg5 = 0 to 10) {
        %18 = krnl.load %6[%arg3, %arg5] : memref<5x10xf32>
        %19 = krnl.load %arg2[%arg5, %arg4] : memref<10x10xf32>
        %20 = krnl.load %5[%arg3, %arg4] : memref<5x10xf32>
        %21 = arith.mulf %18, %19 : f32
        %22 = arith.addf %20, %21 : f32
        krnl.store %22, %5[%arg3, %arg4] : memref<5x10xf32>
      }
    }
    %14:2 = krnl.define_loops 2
    krnl.iterate(%14#0, %14#1) with (%14#0 -> %arg3 = 0 to 10, %14#1 -> %arg4 = 0 to 5) {
      %17 = krnl.load %5[%arg4, %arg3] : memref<5x10xf32>
      %18 = krnl.load %arg0[%arg3, %arg4] : memref<10x5xf32>
      %19 = arith.addf %17, %18 : f32
      krnl.store %19, %3[%arg3, %arg4] : memref<10x5xf32>
    }
    %15:2 = krnl.define_loops 2
    krnl.iterate(%15#0, %15#1) with (%15#0 -> %arg3 = 0 to 10, %15#1 -> %arg4 = 0 to 5) {
      krnl.store %cst, %2[%arg3, %arg4] : memref<10x5xf32>
      %17 = krnl.define_loops 1
      krnl.iterate(%17) with (%17 -> %arg5 = 0 to 5) {
        %18 = krnl.load %3[%arg3, %arg5] : memref<10x5xf32>
        %19 = krnl.load %arg1[%arg5, %arg4] : memref<5x5xf32>
        %20 = krnl.load %2[%arg3, %arg4] : memref<10x5xf32>
        %21 = arith.mulf %18, %19 : f32
        %22 = arith.addf %20, %21 : f32
        krnl.store %22, %2[%arg3, %arg4] : memref<10x5xf32>
      }
      /// Newly added code.
      %newLoad = krnl.load %3[%arg3, %arg4] : memref<10x5xf32>
      krnl.store %newLoad, %4[%arg3, %arg4] : memref<10x5xf32>
    }
    %16:2 = krnl.define_loops 2
    krnl.iterate(%16#0, %16#1) with (%16#0 -> %arg3 = 0 to 10, %16#1 -> %arg4 = 0 to 5) {
      %17 = krnl.load %4[%arg3, %arg4] : memref<10x5xf32>
      %18 = krnl.load %arg0[%arg3, %arg4] : memref<10x5xf32>
      %19 = arith.addf %17, %18 : f32
      krnl.store %19, %0[%arg3, %arg4] : memref<10x5xf32>
    }
    memref.dealloc %1 : memref<1400xi8>
    return %0 : memref<10x5xf32>

  // CHECK-LABEL: multiple_shaped_memrefs
  // CHECK-DAG: [[C0:%.+]] = arith.constant 0 : i64
  // CHECK-DAG: [[C200:%.+]] = arith.constant 200 : i64
  // CHECK-DAG: [[C400:%.+]] = arith.constant 400 : i64
  // CHECK-DAG: [[MEMPOOL:%.+]] = memref.alloc() : memref<600xi8>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C0]]) : (memref<600xi8>, i64) -> memref<10x5xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C200]]) : (memref<600xi8>, i64) -> memref<10x5xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C400]]) : (memref<600xi8>, i64) -> memref<10x5xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C0]]) : (memref<600xi8>, i64) -> memref<5x10xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C400]]) : (memref<600xi8>, i64) -> memref<5x10xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C200]]) : (memref<600xi8>, i64) -> memref<10x5xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C0]]) : (memref<600xi8>, i64) -> memref<10x5xf32>
}

// -----

/// 6. Test for MemRefs which are used by the same operation, in this case an unknown operation.
/// The outcome of this is that by augmenting Test 1 above with an unknown operation, we see a
/// new slot being used increasing the memory usage from 800 to 1200 bytes.
/// Value %4 does not share a slot with %2 and %6 anymore.
func.func @unknown_op_reuse(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>) -> memref<10x10xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0_i64 = arith.constant 0 : i64
  %c1600_i64 = arith.constant 1600 : i64
  %c1200_i64 = arith.constant 1200 : i64
  %c800_i64 = arith.constant 800 : i64
  %c400_i64 = arith.constant 400 : i64
  %0 = memref.alloc() : memref<10x10xf32>
  %1 = memref.alloc() : memref<2000xi8>
  %2 = "krnl.getref"(%1, %c1600_i64) : (memref<2000xi8>, i64) -> memref<10x10xf32>
  %3 = "krnl.getref"(%1, %c1200_i64) : (memref<2000xi8>, i64) -> memref<10x10xf32>
  %4 = "krnl.getref"(%1, %c800_i64) : (memref<2000xi8>, i64) -> memref<10x10xf32>
  %5 = "krnl.getref"(%1, %c400_i64) : (memref<2000xi8>, i64) -> memref<10x10xf32>
  %6 = "krnl.getref"(%1, %c0_i64) : (memref<2000xi8>, i64) -> memref<10x10xf32>
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
  /// Newly added operation, unknown dialect and semantics.
  "unknown.newOp"(%4, %6) : (memref<10x10xf32>, memref<10x10xf32>) -> ()
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
  memref.dealloc %1 : memref<2000xi8>
  return %0 : memref<10x10xf32>

  // CHECK-LABEL: unknown_op_reuse
  // CHECK-DAG: [[C0:%.+]] = arith.constant 0 : i64
  // CHECK-DAG: [[C400:%.+]] = arith.constant 400 : i64
  // CHECK-DAG: [[C800:%.+]] = arith.constant 800 : i64
  // CHECK-DAG: [[MEMPOOL:%.+]] = memref.alloc() : memref<1200xi8>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C0]]) : (memref<1200xi8>, i64) -> memref<10x10xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C400]]) : (memref<1200xi8>, i64) -> memref<10x10xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C800]]) : (memref<1200xi8>, i64) -> memref<10x10xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C400]]) : (memref<1200xi8>, i64) -> memref<10x10xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[C0]]) : (memref<1200xi8>, i64) -> memref<10x10xf32>
}
