// RUN: onnx-mlir-opt -O3 --bundle-memory-pools --canonicalize %s -split-input-file | FileCheck %s

func.func @test_pool_bundling(%arg0: memref<10x10xf32>, %arg1: memref<10x20xf32>) -> memref<10x20xf32> {
  %c0_i64 = arith.constant 0 : i64
  %ind = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = memref.alloc() : memref<10x20xf32>
  %1 = memref.alloc() : memref<800xi8>
  %2 = "krnl.getref"(%1, %c0_i64) : (memref<800xi8>, i64) -> memref<10x20xf32>
  %3 = memref.alloc() : memref<400xi8>
  %4 = "krnl.getref"(%3, %c0_i64) : (memref<400xi8>, i64) -> memref<10x10xf32>
  %5 = memref.alloc() : memref<800xi8>
  %6 = "krnl.getref"(%5, %c0_i64) : (memref<800xi8>, i64) -> memref<10x20xf32>
  %7 = memref.alloc() : memref<800xi8>
  %8 = "krnl.getref"(%7, %c0_i64) : (memref<800xi8>, i64) -> memref<10x20xf32>
  %9 = memref.alloc() : memref<400xi8>
  %10 = "krnl.getref"(%9, %c0_i64) : (memref<400xi8>, i64) -> memref<10x10xf32>
  krnl.store %cst, %10[%ind, %ind] : memref<10x10xf32>
  krnl.store %cst, %8[%ind, %ind] : memref<10x20xf32>
  krnl.store %cst, %6[%ind, %ind] : memref<10x20xf32>
  krnl.store %cst, %4[%ind, %ind] : memref<10x10xf32>
  krnl.store %cst, %2[%ind, %ind] : memref<10x20xf32>
  krnl.store %cst, %0[%ind, %ind] : memref<10x20xf32>
  memref.dealloc %9 : memref<400xi8>
  memref.dealloc %7 : memref<800xi8>
  memref.dealloc %5 : memref<800xi8>
  memref.dealloc %3 : memref<400xi8>
  memref.dealloc %1 : memref<800xi8>
  return %0 : memref<10x20xf32>

  // CHECK-LABEL: test_pool_bundling
  // CHECK-DAG: [[CONST_0:%.+]] = arith.constant 0 : i64
  // CHECK-DAG: [[CONST_0_INDEX:%.+]] = arith.constant 0 : index
  // CHECK-DAG: [[CONST_CST:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: [[CONST_2400:%.+]] = arith.constant 2400 : i64
  // CHECK-DAG: [[CONST_2000:%.+]] = arith.constant 2000 : i64
  // CHECK-DAG: [[CONST_1200:%.+]] = arith.constant 1200 : i64
  // CHECK-DAG: [[CONST_400:%.+]] = arith.constant 400 : i64
  // CHECK-DAG: [[RES:%.+]] = memref.alloc() : memref<10x20xf32>
  // CHECK: [[MEMPOOL:%.+]] = memref.alloc() : memref<3200xi8>
  // CHECK: [[MEMREF1:%.+]] = "krnl.getref"([[MEMPOOL]], [[CONST_2400]]) : (memref<3200xi8>, i64) -> memref<10x20xf32>
  // CHECK: [[MEMREF2:%.+]] = "krnl.getref"([[MEMPOOL]], [[CONST_2000]]) : (memref<3200xi8>, i64) -> memref<10x10xf32>
  // CHECK: [[MEMREF3:%.+]] = "krnl.getref"([[MEMPOOL]], [[CONST_1200]]) : (memref<3200xi8>, i64) -> memref<10x20xf32>
  // CHECK: [[MEMREF4:%.+]] = "krnl.getref"([[MEMPOOL]], [[CONST_400]]) : (memref<3200xi8>, i64) -> memref<10x20xf32>
  // CHECK: [[MEMREF5:%.+]] = "krnl.getref"([[MEMPOOL]], [[CONST_0]]) : (memref<3200xi8>, i64) -> memref<10x10xf32>
  // CHECK: krnl.store %cst, [[MEMREF5]]{{\[}}[[CONST_0_INDEX]], [[CONST_0_INDEX]]{{\]}} : memref<10x10xf32>
  // CHECK: krnl.store %cst, [[MEMREF4]]{{\[}}[[CONST_0_INDEX]], [[CONST_0_INDEX]]{{\]}} : memref<10x20xf32>
  // CHECK: krnl.store %cst, [[MEMREF3]]{{\[}}[[CONST_0_INDEX]], [[CONST_0_INDEX]]{{\]}} : memref<10x20xf32>
  // CHECK: krnl.store %cst, [[MEMREF2]]{{\[}}[[CONST_0_INDEX]], [[CONST_0_INDEX]]{{\]}} : memref<10x10xf32>
  // CHECK: krnl.store %cst, [[MEMREF1]]{{\[}}[[CONST_0_INDEX]], [[CONST_0_INDEX]]{{\]}} : memref<10x20xf32>
  // CHECK: krnl.store %cst, [[RES]]{{\[}}[[CONST_0_INDEX]], [[CONST_0_INDEX]]{{\]}} : memref<10x20xf32>
  // CHECK: memref.dealloc [[MEMPOOL]] : memref<3200xi8>
  // CHECK: return [[RES]] : memref<10x20xf32>
}

// -----

/// Test bundling inside a sub-block.
func.func @static_mem_pool_rnn_subblock(%arg0: memref<1x3x2xf32>, %arg1: memref<1x4x2xf32>, %arg2: memref<1x4x4xf32>) -> memref<1x3x4xf32> attributes {input_names = ["X", "W", "R"], output_names = ["Y"]} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %0 = memref.alloc() : memref<1x3x4xf32>
  %2 = krnl.define_loops 1
  krnl.iterate(%2) with (%2 -> %arg3 = 0 to 1) {
    %3:2 = krnl.define_loops 2
    krnl.iterate(%3#0, %3#1) with (%3#0 -> %arg4 = 0 to 3, %3#1 -> %arg5 = 0 to 4) {
      %4 = memref.alloc() : memref<4xi8>
      %5 = "krnl.getref"(%4, %c0_i64) : (memref<4xi8>, i64) -> memref<f32>
      %6 = krnl.load %0[%c0, %arg4, %arg5] : memref<1x3x4xf32>
      %7 = memref.alloc() : memref<4xi8>
      %8 = "krnl.getref"(%7, %c0_i64) : (memref<4xi8>, i64) -> memref<f32>
      krnl.store %cst, %8[] : memref<f32>
      %9 = memref.alloc() : memref<4xi8>
      %10 = "krnl.getref"(%9, %c0_i64) : (memref<4xi8>, i64) -> memref<f32>
      krnl.store %cst, %10[] : memref<f32>
      %11 = krnl.define_loops 1
      krnl.iterate(%11) with (%11 -> %arg6 = 0 to 2) {
        %25 = krnl.load %arg0[%arg3, %arg4, %arg6] : memref<1x3x2xf32>
        %26 = krnl.load %arg1[%c0, %arg5, %arg6] : memref<1x4x2xf32>
        %27 = arith.mulf %25, %26 : f32
        %28 = krnl.load %8[] : memref<f32>
        %29 = arith.addf %28, %27 : f32
        krnl.store %29, %8[] : memref<f32>
        %30 = krnl.load %arg2[%c0, %arg5, %arg6] : memref<1x4x4xf32>
        %31 = arith.mulf %6, %30 : f32
        %32 = krnl.load %10[] : memref<f32>
        %33 = arith.addf %32, %31 : f32
        krnl.store %33, %10[] : memref<f32>
      }
      %12 = krnl.load %8[] : memref<f32>
      %13 = krnl.load %10[] : memref<f32>
      %14 = arith.addf %12, %13 : f32
      %15 = memref.alloc() : memref<4xi8>
      %16 = "krnl.getref"(%15, %c0_i64) : (memref<4xi8>, i64) -> memref<f32>
      krnl.store %14, %16[] : memref<f32>
      %17 = krnl.load %16[] : memref<f32>
      %18 = arith.subf %cst, %17 : f32
      %19 = math.exp %17 : f32
      %20 = math.exp %18 : f32
      %21 = arith.subf %19, %20 : f32
      %22 = arith.addf %19, %20 : f32
      %23 = arith.divf %21, %22 : f32
      krnl.store %23, %5[] : memref<f32>
      %24 = krnl.load %5[] : memref<f32>
      krnl.store %24, %0[%c0, %arg4, %arg5] : memref<1x3x4xf32>
      memref.dealloc %15 : memref<4xi8>
      memref.dealloc %9 : memref<4xi8>
      memref.dealloc %7 : memref<4xi8>
      memref.dealloc %4 : memref<4xi8>
    }
  }
  return %0 : memref<1x3x4xf32>

  // CHECK-LABEL: static_mem_pool_rnn_subblock
  // CHECK-DAG: [[CST:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: [[C0:%.+]] = arith.constant 0 : i64
  // CHECK-DAG: [[C12:%.+]] = arith.constant 12 : i64
  // CHECK-DAG: [[C8:%.+]] = arith.constant 8 : i64
  // CHECK-DAG: [[C4:%.+]] = arith.constant 4 : i64
  // CHECK-DAG: [[RES:%.+]] = memref.alloc() : memref<1x3x4xf32>
  // CHECK: krnl.define_loops 1
  // CHECK: krnl.iterate
  // CHECK: krnl.define_loops 2
  // CHECK: krnl.iterate
  // CHECK: [[STATIC_MEM_POOL:%.+]] = memref.alloc() : memref<16xi8>
  // CHECK: [[REF1:%.+]] = "krnl.getref"([[STATIC_MEM_POOL]], [[C12]]) : (memref<16xi8>, i64) -> memref<f32>
  // CHECK: krnl.load
  // CHECK: [[REF2:%.+]] = "krnl.getref"([[STATIC_MEM_POOL]], [[C8]]) : (memref<16xi8>, i64) -> memref<f32>
  // CHECK: krnl.store
  // CHECK: [[REF3:%.+]] = "krnl.getref"([[STATIC_MEM_POOL]], [[C4]]) : (memref<16xi8>, i64) -> memref<f32>
  // CHECK: krnl.store
  // CHECK: krnl.define_loops 1
  // CHECK: krnl.iterate
  // CHECK: [[REF4:%.+]] = "krnl.getref"([[STATIC_MEM_POOL]], [[C0]]) : (memref<16xi8>, i64) -> memref<f32>
  // CHECK: memref.dealloc [[STATIC_MEM_POOL]] : memref<16xi8>
  // CHECK: return [[RES]] : memref<1x3x4xf32>
}

// -----

/// Test bundling inside a sub-block and in the main block.
func.func @static_mem_pool_rnn_sub_and_main_block(%arg0: memref<1x3x2xf32>, %arg1: memref<1x4x2xf32>, %arg2: memref<1x4x4xf32>) -> memref<1x3x4xf32> attributes {input_names = ["X", "W", "R"], output_names = ["Y"]} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index 
  %0 = memref.alloc() : memref<1x3x4xf32>
  %mem0 = memref.alloc() : memref<4xi8>
  %ref0 = "krnl.getref"(%mem0, %c0_i64) : (memref<4xi8>, i64) -> memref<f32>
  %mem1 = memref.alloc() : memref<4xi8>
  %ref1 = "krnl.getref"(%mem1, %c0_i64) : (memref<4xi8>, i64) -> memref<f32>
  %2 = krnl.define_loops 1
  krnl.iterate(%2) with (%2 -> %arg3 = 0 to 1) {
    %3:2 = krnl.define_loops 2
    krnl.iterate(%3#0, %3#1) with (%3#0 -> %arg4 = 0 to 3, %3#1 -> %arg5 = 0 to 4) {
      %4 = memref.alloc() : memref<4xi8>
      %5 = "krnl.getref"(%4, %c0_i64) : (memref<4xi8>, i64) -> memref<f32>
      %6 = krnl.load %0[%c0, %arg4, %arg5] : memref<1x3x4xf32>
      %7 = memref.alloc() : memref<4xi8>
      %8 = "krnl.getref"(%7, %c0_i64) : (memref<4xi8>, i64) -> memref<f32>
      krnl.store %cst, %8[] : memref<f32>
      %9 = memref.alloc() : memref<4xi8>
      %10 = "krnl.getref"(%9, %c0_i64) : (memref<4xi8>, i64) -> memref<f32>
      krnl.store %cst, %10[] : memref<f32>
      %11 = krnl.define_loops 1
      krnl.iterate(%11) with (%11 -> %arg6 = 0 to 2) {
        %25 = krnl.load %arg0[%arg3, %arg4, %arg6] : memref<1x3x2xf32>
        %26 = krnl.load %arg1[%c0, %arg5, %arg6] : memref<1x4x2xf32>
        %27 = arith.mulf %25, %26 : f32
        %28 = krnl.load %8[] : memref<f32>
        %29 = arith.addf %28, %27 : f32
        krnl.store %29, %8[] : memref<f32>
        %30 = krnl.load %arg2[%c0, %arg5, %arg6] : memref<1x4x4xf32>
        %31 = arith.mulf %6, %30 : f32
        %32 = krnl.load %10[] : memref<f32>
        %33 = arith.addf %32, %31 : f32
        krnl.store %33, %10[] : memref<f32>
        krnl.store %33, %ref0[] : memref<f32>
      }
      %12 = krnl.load %8[] : memref<f32>
      %13 = krnl.load %10[] : memref<f32>
      %14 = arith.addf %12, %13 : f32
      %15 = memref.alloc() : memref<4xi8>
      %16 = "krnl.getref"(%15, %c0_i64) : (memref<4xi8>, i64) -> memref<f32>
      krnl.store %14, %16[] : memref<f32>
      %17 = krnl.load %16[] : memref<f32>
      %18 = arith.subf %cst, %17 : f32
      %19 = math.exp %17 : f32
      %20 = math.exp %18 : f32
      %21 = arith.subf %19, %20 : f32
      %22 = arith.addf %19, %20 : f32
      %23 = arith.divf %21, %22 : f32
      krnl.store %23, %5[] : memref<f32>
      %24 = krnl.load %5[] : memref<f32>
      krnl.store %24, %0[%c0, %arg4, %arg5] : memref<1x3x4xf32>
      krnl.store %24, %ref1[] : memref<f32>
      memref.dealloc %15 : memref<4xi8>
      memref.dealloc %9 : memref<4xi8>
      memref.dealloc %7 : memref<4xi8>
      memref.dealloc %4 : memref<4xi8>
    }
  }
  %mem2 = memref.alloc() : memref<4xi8>
  %ref2 = "krnl.getref"(%mem2, %c0_i64) : (memref<4xi8>, i64) -> memref<f32>
  %val = krnl.load %ref1[] : memref<f32>
  krnl.store %val, %ref2[] : memref<f32>
  memref.dealloc %mem2 : memref<4xi8>
  memref.dealloc %mem1 : memref<4xi8>
  memref.dealloc %mem0 : memref<4xi8>
  return %0 : memref<1x3x4xf32>

  // CHECK-LABEL: static_mem_pool_rnn_sub_and_main_block
  // CHECK-DAG: [[CST:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: [[C0:%.+]] = arith.constant 0 : i64
  // CHECK-DAG: [[C12:%.+]] = arith.constant 12 : i64
  // CHECK-DAG: [[C8:%.+]] = arith.constant 8 : i64
  // CHECK-DAG: [[C4:%.+]] = arith.constant 4 : i64
  // CHECK-DAG: [[RES:%.+]] = memref.alloc() : memref<1x3x4xf32>
  // CHECK-DAG: [[STATIC_MEM_POOL_MAIN:%.+]] = memref.alloc() : memref<12xi8>
  // CHECK: [[MAIN_REF_0:%.+]] = "krnl.getref"([[STATIC_MEM_POOL_MAIN]], [[C8]]) : (memref<12xi8>, i64) -> memref<f32>
  // CHECK: [[MAIN_REF_1:%.+]] = "krnl.getref"([[STATIC_MEM_POOL_MAIN]], [[C4]]) : (memref<12xi8>, i64) -> memref<f32>
  // CHECK: krnl.define_loops 1
  // CHECK: krnl.iterate
  // CHECK: krnl.define_loops 2
  // CHECK: krnl.iterate
  // CHECK: [[STATIC_MEM_POOL:%.+]] = memref.alloc() : memref<16xi8>
  // CHECK: [[REF1:%.+]] = "krnl.getref"([[STATIC_MEM_POOL]], [[C12]]) : (memref<16xi8>, i64) -> memref<f32>
  // CHECK: krnl.load
  // CHECK: [[REF2:%.+]] = "krnl.getref"([[STATIC_MEM_POOL]], [[C8]]) : (memref<16xi8>, i64) -> memref<f32>
  // CHECK: krnl.store
  // CHECK: [[REF3:%.+]] = "krnl.getref"([[STATIC_MEM_POOL]], [[C4]]) : (memref<16xi8>, i64) -> memref<f32>
  // CHECK: krnl.store
  // CHECK: krnl.define_loops 1
  // CHECK: krnl.iterate
  // CHECK: [[REF4:%.+]] = "krnl.getref"([[STATIC_MEM_POOL]], [[C0]]) : (memref<16xi8>, i64) -> memref<f32>
  // CHECK: memref.dealloc [[STATIC_MEM_POOL]] : memref<16xi8>
  // CHECK: [[MAIN_REF_2:%.+]] = "krnl.getref"([[STATIC_MEM_POOL_MAIN]], [[C0]]) : (memref<12xi8>, i64) -> memref<f32>
  // CHECK: [[LOAD:%.+]] = krnl.load [[MAIN_REF_1]][] : memref<f32>
  // CHECK: krnl.store [[LOAD]], [[MAIN_REF_2]][] : memref<f32>
  // CHECK: memref.dealloc [[STATIC_MEM_POOL_MAIN]] : memref<12xi8>
  // CHECK: return [[RES]] : memref<1x3x4xf32>
}
