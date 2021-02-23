// RUN: onnx-mlir-opt --bundle-memory-pools --canonicalize %s -split-input-file | FileCheck %s

func @test_pool_bundling(%arg0: memref<10x10xf32>, %arg1: memref<10x20xf32>) -> memref<10x20xf32> {
  %c0_i64 = constant 0 : i64
  %ind = constant 0 : index
  %cst = constant 0.000000e+00 : f32
  %0 = alloc() : memref<10x20xf32>
  %1 = alloc() : memref<800xi8>
  %2 = "krnl.getref"(%1, %c0_i64) : (memref<800xi8>, i64) -> memref<10x20xf32>
  %3 = alloc() : memref<400xi8>
  %4 = "krnl.getref"(%3, %c0_i64) : (memref<400xi8>, i64) -> memref<10x10xf32>
  %5 = alloc() : memref<800xi8>
  %6 = "krnl.getref"(%5, %c0_i64) : (memref<800xi8>, i64) -> memref<10x20xf32>
  %7 = alloc() : memref<800xi8>
  %8 = "krnl.getref"(%7, %c0_i64) : (memref<800xi8>, i64) -> memref<10x20xf32>
  %9 = alloc() : memref<400xi8>
  %10 = "krnl.getref"(%9, %c0_i64) : (memref<400xi8>, i64) -> memref<10x10xf32>
  krnl.store %cst, %10[%ind, %ind] : memref<10x10xf32>
  krnl.store %cst, %8[%ind, %ind] : memref<10x20xf32>
  krnl.store %cst, %6[%ind, %ind] : memref<10x20xf32>
  krnl.store %cst, %4[%ind, %ind] : memref<10x10xf32>
  krnl.store %cst, %2[%ind, %ind] : memref<10x20xf32>
  krnl.store %cst, %0[%ind, %ind] : memref<10x20xf32>
  dealloc %9 : memref<400xi8>
  dealloc %7 : memref<800xi8>
  dealloc %5 : memref<800xi8>
  dealloc %3 : memref<400xi8>
  dealloc %1 : memref<800xi8>
  return %0 : memref<10x20xf32>

  // CHECK-LABEL: test_pool_bundling
  // CHECK: [[CONST_0:%.+]] = constant 0 : i64
  // CHECK-DAG: [[CONST_0_INDEX:%.+]] = constant 0 : index
  // CHECK: [[CONST_CST:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[CONST_2400:%.+]] = constant 2400 : i64
  // CHECK: [[CONST_2000:%.+]] = constant 2000 : i64
  // CHECK: [[CONST_1200:%.+]] = constant 1200 : i64
  // CHECK: [[CONST_400:%.+]] = constant 400 : i64
  // CHECK: [[RES:%.+]] = alloc() : memref<10x20xf32>
  // CHECK: [[MEMPOOL:%.+]] = alloc() : memref<3200xi8>
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
  // CHECK: dealloc [[MEMPOOL]] : memref<3200xi8>
  // CHECK: return [[RES]] : memref<10x20xf32>
}

// -----

func @test_dynamic_pool_bundling(%arg0: memref<?x?xf32>) -> memref<?x10xf32> {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %cst = constant 0.000000e+00 : f32
  %ind = constant 0 : index
  %c4 = constant 4 : index
  %c10 = constant 10 : index
  %c0_i64 = constant 0 : i64
  %0 = dim %arg0, %c0 : memref<?x?xf32>
  %1 = muli %0, %c4 : index
  %2 = muli %1, %c10 : index
  %3 = alloc(%2) : memref<?xi8>
  %4 = "krnl.getref"(%3, %c0_i64, %0) : (memref<?xi8>, i64, index) -> memref<?x10xf32>
  %6 = cmpi sgt, %0, %0 : index
  %7 = select %6, %0, %0 : index
  %8 = muli %7, %c4 : index
  %9 = muli %8, %c10 : index
  %10 = alloc(%9) : memref<?xi8>
  %11 = "krnl.getref"(%10, %c0_i64, %7) : (memref<?xi8>, i64, index) -> memref<?x10xf32>
  %12 = cmpi eq, %0, %c1 : index
  %13 = cmpi eq, %0, %c1 : index
  %15 = alloc(%0) : memref<?x10xf32>
  krnl.store %cst, %4[%ind, %ind] : memref<?x10xf32>
  krnl.store %cst, %11[%ind, %ind] : memref<?x10xf32>
  krnl.store %cst, %15[%ind, %ind] : memref<?x10xf32>
  dealloc %10 : memref<?xi8>
  dealloc %3 : memref<?xi8>
  return %15 : memref<?x10xf32>

  // CHECK-LABEL: test_dynamic_pool_bundling
  // CHECK: [[CST:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[C4:%.+]] = constant 4 : index
  // CHECK: [[C10:%.+]] = constant 10 : index
  // CHECK: [[C0_I64:%.+]] = constant 0 : i64
  // CHECK: [[DIM:%.+]] = dim %arg0, [[C0]] : memref<?x?xf32>
  // CHECK: [[MUL2:%.+]] = muli [[DIM]], [[C4]] : index
  // CHECK: [[OFFSET2:%.+]] = muli [[MUL2]], [[C10]] : index
  // CHECK: [[MUL1:%.+]] = muli [[DIM]], [[C4]] : index
  // CHECK: [[OFFSET1:%.+]] = muli [[MUL1]], [[C10]] : index
  // CHECK: [[MEMPOOL_SIZE:%.+]] = addi [[OFFSET1]], [[OFFSET2]] : index
  // CHECK: [[OFFSET1_I64:%.+]] = index_cast [[OFFSET1]] : index to i64
  // CHECK: [[DYN_MEMPOOL:%.+]] = alloc([[MEMPOOL_SIZE]]) : memref<?xi8>
  // CHECK: [[DATA1:%.+]] = "krnl.getref"([[DYN_MEMPOOL]], [[OFFSET1_I64]], [[DIM]]) : (memref<?xi8>, i64, index) -> memref<?x10xf32>
  // CHECK: [[DATA2:%.+]] = "krnl.getref"([[DYN_MEMPOOL]], [[C0_I64]], [[DIM]]) : (memref<?xi8>, i64, index) -> memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM]]) : memref<?x10xf32>
  // CHECK: krnl.store [[CST]], [[DATA1]]{{\[}}[[C0]], [[C0]]{{\]}} : memref<?x10xf32>
  // CHECK: krnl.store [[CST]], [[DATA2]]{{\[}}[[C0]], [[C0]]{{\]}} : memref<?x10xf32>
  // CHECK: krnl.store [[CST]], [[RES]]{{\[}}[[C0]], [[C0]]{{\]}} : memref<?x10xf32>
  // CHECK: dealloc [[DYN_MEMPOOL]] : memref<?xi8>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_dynamic_and_static_pool_bundling(%arg0: memref<?x?xf32>, %arg1: memref<10x10xf32>) -> memref<?x10xf32> {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %cst = constant 0.000000e+00 : f32
  %ind = constant 0 : index
  %c4 = constant 4 : index
  %c10 = constant 10 : index
  %c0_i64 = constant 0 : i64
  %0 = dim %arg0, %c0 : memref<?x?xf32>
  %1 = muli %0, %c4 : index
  %2 = muli %1, %c10 : index
  %3 = alloc(%2) : memref<?xi8>
  %4 = "krnl.getref"(%3, %c0_i64, %0) : (memref<?xi8>, i64, index) -> memref<?x10xf32>
  %const_alloc1 = alloc() : memref<800xi8>
  %const_ref1 = "krnl.getref"(%const_alloc1, %c0_i64) : (memref<800xi8>, i64) -> memref<10x20xf32>
  %const_alloc2 = alloc() : memref<400xi8>
  %const_ref2 = "krnl.getref"(%const_alloc2, %c0_i64) : (memref<400xi8>, i64) -> memref<10x10xf32>
  %6 = cmpi sgt, %0, %0 : index
  %7 = select %6, %0, %0 : index
  %8 = muli %7, %c4 : index
  %9 = muli %8, %c10 : index
  %10 = alloc(%9) : memref<?xi8>
  %11 = "krnl.getref"(%10, %c0_i64, %7) : (memref<?xi8>, i64, index) -> memref<?x10xf32>
  %12 = cmpi eq, %0, %c1 : index
  %13 = cmpi eq, %0, %c1 : index
  %15 = alloc(%0) : memref<?x10xf32>
  %const_alloc3 = alloc() : memref<1600xi8>
  %const_ref3 = "krnl.getref"(%const_alloc3, %c0_i64) : (memref<1600xi8>, i64) -> memref<10x40xf32>
  krnl.store %cst, %4[%ind, %ind] : memref<?x10xf32>
  krnl.store %cst, %11[%ind, %ind] : memref<?x10xf32>
  krnl.store %cst, %15[%ind, %ind] : memref<?x10xf32>
  krnl.store %cst, %const_ref2[%ind, %ind] : memref<10x10xf32>
  krnl.store %cst, %const_ref1[%ind, %ind] : memref<10x20xf32>
  krnl.store %cst, %const_ref3[%ind, %ind] : memref<10x40xf32>
  dealloc %10 : memref<?xi8>
  dealloc %3 : memref<?xi8>
  dealloc %const_alloc1 : memref<800xi8>
  dealloc %const_alloc2 : memref<400xi8>
  dealloc %const_alloc3 : memref<1600xi8>
  return %15 : memref<?x10xf32>

  // CHECK-LABEL: test_dynamic_and_static_pool_bundling
  // CHECK: [[CST:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[C4:%.+]] = constant 4 : index
  // CHECK: [[C10:%.+]] = constant 10 : index
  // CHECK: [[C2000_I64:%.+]] = constant 2000 : i64
  // CHECK: [[C1600_I64:%.+]] = constant 1600 : i64
  // CHECK: [[C0_I64:%.+]] = constant 0 : i64
  // CHECK: [[DIM:%.+]] = dim %arg0, [[C0]] : memref<?x?xf32>
  // CHECK: [[MUL2:%.+]] = muli [[DIM]], [[C4]] : index
  // CHECK: [[OFFSET2:%.+]] = muli [[MUL2]], [[C10]] : index
  // CHECK: [[MUL1:%.+]] = muli [[DIM]], [[C4]] : index
  // CHECK: [[OFFSET1:%.+]] = muli [[MUL1]], [[C10]] : index
  // CHECK: [[MEMPOOL_SIZE:%.+]] = addi [[OFFSET1]], [[OFFSET2]] : index
  // CHECK: [[OFFSET1_I64:%.+]] = index_cast [[OFFSET1]] : index to i64
  // CHECK: [[DYN_MEMPOOL:%.+]] = alloc([[MEMPOOL_SIZE]]) : memref<?xi8>
  // CHECK: [[DATA2:%.+]] = "krnl.getref"([[DYN_MEMPOOL]], [[OFFSET1_I64]], [[DIM]]) : (memref<?xi8>, i64, index) -> memref<?x10xf32>
  // CHECK: [[STATIC_MEMPOOL:%.+]] = alloc() : memref<2800xi8>
  // CHECK: [[DATA3:%.+]] = "krnl.getref"([[STATIC_MEMPOOL]], [[C2000_I64]]) : (memref<2800xi8>, i64) -> memref<10x20xf32>
  // CHECK: [[DATA4:%.+]] = "krnl.getref"([[STATIC_MEMPOOL]], [[C1600_I64]]) : (memref<2800xi8>, i64) -> memref<10x10xf32>
  // CHECK: [[DATA1:%.+]] = "krnl.getref"([[DYN_MEMPOOL]], [[C0_I64]], [[DIM]]) : (memref<?xi8>, i64, index) -> memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM]]) : memref<?x10xf32>
  // CHECK: [[DATA5:%.+]] = "krnl.getref"([[STATIC_MEMPOOL]], [[C0_I64]]) : (memref<2800xi8>, i64) -> memref<10x40xf32>
  // CHECK: krnl.store [[CST]], [[DATA2]]{{\[}}[[C0]], [[C0]]{{\]}} : memref<?x10xf32>
  // CHECK: krnl.store [[CST]], [[DATA1]]{{\[}}[[C0]], [[C0]]{{\]}} : memref<?x10xf32>
  // CHECK: krnl.store [[CST]], [[RES]]{{\[}}[[C0]], [[C0]]{{\]}} : memref<?x10xf32>
  // CHECK: krnl.store [[CST]], [[DATA4]]{{\[}}[[C0]], [[C0]]{{\]}} : memref<10x10xf32>
  // CHECK: krnl.store [[CST]], [[DATA3]]{{\[}}[[C0]], [[C0]]{{\]}} : memref<10x20xf32>
  // CHECK: krnl.store [[CST]], [[DATA5]]{{\[}}[[C0]], [[C0]]{{\]}} : memref<10x40xf32>
  // CHECK: dealloc [[DYN_MEMPOOL]] : memref<?xi8>
  // CHECK: dealloc [[STATIC_MEMPOOL]] : memref<2800xi8>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

/// Test bundling inside a sub-block.
func @static_mem_pool_rnn_subblock(%arg0: memref<1x3x2xf32>, %arg1: memref<1x4x2xf32>, %arg2: memref<1x4x4xf32>) -> memref<1x3x4xf32> attributes {input_names = ["X", "W", "R"], output_names = ["Y"]} {
  %cst = constant 0.000000e+00 : f32
  %c0_i64 = constant 0 : i64
  %c0 = constant 0 : index
  %0 = alloc() : memref<1x3x4xf32>
  %2 = krnl.define_loops 1
  krnl.iterate(%2) with (%2 -> %arg3 = 0 to 1) {
    %3:2 = krnl.define_loops 2
    krnl.iterate(%3#0, %3#1) with (%3#0 -> %arg4 = 0 to 3, %3#1 -> %arg5 = 0 to 4) {
      %4 = alloc() : memref<4xi8>
      %5 = "krnl.getref"(%4, %c0_i64) : (memref<4xi8>, i64) -> memref<f32>
      %6 = krnl.load %0[%c0, %arg4, %arg5] : memref<1x3x4xf32>
      %7 = alloc() : memref<4xi8>
      %8 = "krnl.getref"(%7, %c0_i64) : (memref<4xi8>, i64) -> memref<f32>
      krnl.store %cst, %8[] : memref<f32>
      %9 = alloc() : memref<4xi8>
      %10 = "krnl.getref"(%9, %c0_i64) : (memref<4xi8>, i64) -> memref<f32>
      krnl.store %cst, %10[] : memref<f32>
      %11 = krnl.define_loops 1
      krnl.iterate(%11) with (%11 -> %arg6 = 0 to 2) {
        %25 = krnl.load %arg0[%arg3, %arg4, %arg6] : memref<1x3x2xf32>
        %26 = krnl.load %arg1[%c0, %arg5, %arg6] : memref<1x4x2xf32>
        %27 = mulf %25, %26 : f32
        %28 = krnl.load %8[] : memref<f32>
        %29 = addf %28, %27 : f32
        krnl.store %29, %8[] : memref<f32>
        %30 = krnl.load %arg2[%c0, %arg5, %arg6] : memref<1x4x4xf32>
        %31 = mulf %6, %30 : f32
        %32 = krnl.load %10[] : memref<f32>
        %33 = addf %32, %31 : f32
        krnl.store %33, %10[] : memref<f32>
      }
      %12 = krnl.load %8[] : memref<f32>
      %13 = krnl.load %10[] : memref<f32>
      %14 = addf %12, %13 : f32
      %15 = alloc() : memref<4xi8>
      %16 = "krnl.getref"(%15, %c0_i64) : (memref<4xi8>, i64) -> memref<f32>
      krnl.store %14, %16[] : memref<f32>
      %17 = krnl.load %16[] : memref<f32>
      %18 = subf %cst, %17 : f32
      %19 = math.exp %17 : f32
      %20 = math.exp %18 : f32
      %21 = subf %19, %20 : f32
      %22 = addf %19, %20 : f32
      %23 = divf %21, %22 : f32
      krnl.store %23, %5[] : memref<f32>
      %24 = krnl.load %5[] : memref<f32>
      krnl.store %24, %0[%c0, %arg4, %arg5] : memref<1x3x4xf32>
      dealloc %15 : memref<4xi8>
      dealloc %9 : memref<4xi8>
      dealloc %7 : memref<4xi8>
      dealloc %4 : memref<4xi8>
    }
  }
  return %0 : memref<1x3x4xf32>

  // CHECK-LABEL: static_mem_pool_rnn_subblock
  // CHECK: [[CST:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[C0:%.+]] = constant 0 : i64
  // CHECK: [[C12:%.+]] = constant 12 : i64
  // CHECK: [[C8:%.+]] = constant 8 : i64
  // CHECK: [[C4:%.+]] = constant 4 : i64
  // CHECK: [[RES:%.+]] = alloc() : memref<1x3x4xf32>
  // CHECK: krnl.define_loops 1
  // CHECK: krnl.iterate
  // CHECK: krnl.define_loops 2
  // CHECK: krnl.iterate
  // CHECK: [[STATIC_MEM_POOL:%.+]] = alloc() : memref<16xi8>
  // CHECK: [[REF1:%.+]] = "krnl.getref"([[STATIC_MEM_POOL]], [[C12]]) : (memref<16xi8>, i64) -> memref<f32>
  // CHECK: krnl.load
  // CHECK: [[REF2:%.+]] = "krnl.getref"([[STATIC_MEM_POOL]], [[C8]]) : (memref<16xi8>, i64) -> memref<f32>
  // CHECK: krnl.store
  // CHECK: [[REF3:%.+]] = "krnl.getref"([[STATIC_MEM_POOL]], [[C4]]) : (memref<16xi8>, i64) -> memref<f32>
  // CHECK: krnl.store
  // CHECK: krnl.define_loops 1
  // CHECK: krnl.iterate
  // CHECK: [[REF4:%.+]] = "krnl.getref"([[STATIC_MEM_POOL]], [[C0]]) : (memref<16xi8>, i64) -> memref<f32>
  // CHECK: dealloc [[STATIC_MEM_POOL]] : memref<16xi8>
  // CHECK: return [[RES]] : memref<1x3x4xf32>
}

// -----

/// Test bundling inside a sub-block and in the main block.
func @static_mem_pool_rnn_sub_and_main_block(%arg0: memref<1x3x2xf32>, %arg1: memref<1x4x2xf32>, %arg2: memref<1x4x4xf32>) -> memref<1x3x4xf32> attributes {input_names = ["X", "W", "R"], output_names = ["Y"]} {
  %cst = constant 0.000000e+00 : f32
  %c0_i64 = constant 0 : i64
  %c0 = constant 0 : index 
  %0 = alloc() : memref<1x3x4xf32>
  %mem0 = alloc() : memref<4xi8>
  %ref0 = "krnl.getref"(%mem0, %c0_i64) : (memref<4xi8>, i64) -> memref<f32>
  %mem1 = alloc() : memref<4xi8>
  %ref1 = "krnl.getref"(%mem1, %c0_i64) : (memref<4xi8>, i64) -> memref<f32>
  %2 = krnl.define_loops 1
  krnl.iterate(%2) with (%2 -> %arg3 = 0 to 1) {
    %3:2 = krnl.define_loops 2
    krnl.iterate(%3#0, %3#1) with (%3#0 -> %arg4 = 0 to 3, %3#1 -> %arg5 = 0 to 4) {
      %4 = alloc() : memref<4xi8>
      %5 = "krnl.getref"(%4, %c0_i64) : (memref<4xi8>, i64) -> memref<f32>
      %6 = krnl.load %0[%c0, %arg4, %arg5] : memref<1x3x4xf32>
      %7 = alloc() : memref<4xi8>
      %8 = "krnl.getref"(%7, %c0_i64) : (memref<4xi8>, i64) -> memref<f32>
      krnl.store %cst, %8[] : memref<f32>
      %9 = alloc() : memref<4xi8>
      %10 = "krnl.getref"(%9, %c0_i64) : (memref<4xi8>, i64) -> memref<f32>
      krnl.store %cst, %10[] : memref<f32>
      %11 = krnl.define_loops 1
      krnl.iterate(%11) with (%11 -> %arg6 = 0 to 2) {
        %25 = krnl.load %arg0[%arg3, %arg4, %arg6] : memref<1x3x2xf32>
        %26 = krnl.load %arg1[%c0, %arg5, %arg6] : memref<1x4x2xf32>
        %27 = mulf %25, %26 : f32
        %28 = krnl.load %8[] : memref<f32>
        %29 = addf %28, %27 : f32
        krnl.store %29, %8[] : memref<f32>
        %30 = krnl.load %arg2[%c0, %arg5, %arg6] : memref<1x4x4xf32>
        %31 = mulf %6, %30 : f32
        %32 = krnl.load %10[] : memref<f32>
        %33 = addf %32, %31 : f32
        krnl.store %33, %10[] : memref<f32>
        krnl.store %33, %ref0[] : memref<f32>
      }
      %12 = krnl.load %8[] : memref<f32>
      %13 = krnl.load %10[] : memref<f32>
      %14 = addf %12, %13 : f32
      %15 = alloc() : memref<4xi8>
      %16 = "krnl.getref"(%15, %c0_i64) : (memref<4xi8>, i64) -> memref<f32>
      krnl.store %14, %16[] : memref<f32>
      %17 = krnl.load %16[] : memref<f32>
      %18 = subf %cst, %17 : f32
      %19 = math.exp %17 : f32
      %20 = math.exp %18 : f32
      %21 = subf %19, %20 : f32
      %22 = addf %19, %20 : f32
      %23 = divf %21, %22 : f32
      krnl.store %23, %5[] : memref<f32>
      %24 = krnl.load %5[] : memref<f32>
      krnl.store %24, %0[%c0, %arg4, %arg5] : memref<1x3x4xf32>
      krnl.store %24, %ref1[] : memref<f32>
      dealloc %15 : memref<4xi8>
      dealloc %9 : memref<4xi8>
      dealloc %7 : memref<4xi8>
      dealloc %4 : memref<4xi8>
    }
  }
  %mem2 = alloc() : memref<4xi8>
  %ref2 = "krnl.getref"(%mem2, %c0_i64) : (memref<4xi8>, i64) -> memref<f32>
  %val = krnl.load %ref1[] : memref<f32>
  krnl.store %val, %ref2[] : memref<f32>
  dealloc %mem2 : memref<4xi8>
  dealloc %mem1 : memref<4xi8>
  dealloc %mem0 : memref<4xi8>
  return %0 : memref<1x3x4xf32>

  // CHECK-LABEL: static_mem_pool_rnn_sub_and_main_block
  // CHECK: [[CST:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[C0:%.+]] = constant 0 : i64
  // CHECK: [[C12:%.+]] = constant 12 : i64
  // CHECK: [[C8:%.+]] = constant 8 : i64
  // CHECK: [[C4:%.+]] = constant 4 : i64
  // CHECK: [[RES:%.+]] = alloc() : memref<1x3x4xf32>
  // CHECK: [[STATIC_MEM_POOL_MAIN:%.+]] = alloc() : memref<12xi8>
  // CHECK: [[MAIN_REF_0:%.+]] = "krnl.getref"([[STATIC_MEM_POOL_MAIN]], [[C8]]) : (memref<12xi8>, i64) -> memref<f32>
  // CHECK: [[MAIN_REF_1:%.+]] = "krnl.getref"([[STATIC_MEM_POOL_MAIN]], [[C4]]) : (memref<12xi8>, i64) -> memref<f32>
  // CHECK: krnl.define_loops 1
  // CHECK: krnl.iterate
  // CHECK: krnl.define_loops 2
  // CHECK: krnl.iterate
  // CHECK: [[STATIC_MEM_POOL:%.+]] = alloc() : memref<16xi8>
  // CHECK: [[REF1:%.+]] = "krnl.getref"([[STATIC_MEM_POOL]], [[C12]]) : (memref<16xi8>, i64) -> memref<f32>
  // CHECK: krnl.load
  // CHECK: [[REF2:%.+]] = "krnl.getref"([[STATIC_MEM_POOL]], [[C8]]) : (memref<16xi8>, i64) -> memref<f32>
  // CHECK: krnl.store
  // CHECK: [[REF3:%.+]] = "krnl.getref"([[STATIC_MEM_POOL]], [[C4]]) : (memref<16xi8>, i64) -> memref<f32>
  // CHECK: krnl.store
  // CHECK: krnl.define_loops 1
  // CHECK: krnl.iterate
  // CHECK: [[REF4:%.+]] = "krnl.getref"([[STATIC_MEM_POOL]], [[C0]]) : (memref<16xi8>, i64) -> memref<f32>
  // CHECK: dealloc [[STATIC_MEM_POOL]] : memref<16xi8>
  // CHECK: [[MAIN_REF_2:%.+]] = "krnl.getref"([[STATIC_MEM_POOL_MAIN]], [[C0]]) : (memref<12xi8>, i64) -> memref<f32>
  // CHECK: [[LOAD:%.+]] = krnl.load [[MAIN_REF_1]][] : memref<f32>
  // CHECK: krnl.store [[LOAD]], [[MAIN_REF_2]][] : memref<f32>
  // CHECK: dealloc [[STATIC_MEM_POOL_MAIN]] : memref<12xi8>
  // CHECK: return [[RES]] : memref<1x3x4xf32>
}

// -----

/// Test dynamic pooling in sub-block.
func @test_dynamic_pool_rnn(%arg0: memref<1x3x2xf32>, %arg1: memref<1x4x2xf32>, %arg2: memref<1x?x?xf32>) -> memref<1x3x?xf32> attributes {input_names = ["X", "W", "R"], output_names = ["Y"]} {
  %cst = constant 0.000000e+00 : f32
  %c0_i64 = constant 0 : i64
  %c0 = constant 0 : index
  %c_ind = constant 1 : index
  %dim0 = dim %arg1, %c_ind : memref<1x4x2xf32>
  %0 = alloc(%dim0) : memref<1x3x?xf32>
  %1:3 = krnl.define_loops 3
  krnl.iterate(%1#0, %1#1, %1#2) with (%1#0 -> %arg3 = 0 to 1, %1#1 -> %arg4 = 0 to 3, %1#2 -> %arg5 = 0 to 4) {
    krnl.store %cst, %0[%arg3, %arg4, %arg5] : memref<1x3x?xf32>
  }
  %2 = krnl.define_loops 1
  krnl.iterate(%2) with (%2 -> %arg3 = 0 to 1) {
    %3:2 = krnl.define_loops 2
    krnl.iterate(%3#0, %3#1) with (%3#0 -> %arg4 = 0 to 3, %3#1 -> %arg5 = 0 to 4) {
      %dim1 = dim %arg2, %c_ind : memref<1x?x?xf32>
      %4 = alloc(%dim1) : memref<?xi8>
      %5 = "krnl.getref"(%4, %c0_i64) : (memref<?xi8>, i64) -> memref<f32>
      %6 = krnl.load %0[%c0, %arg4, %arg5] : memref<1x3x?xf32>
      %7 = alloc(%dim1) : memref<?xi8>
      %8 = "krnl.getref"(%7, %c0_i64) : (memref<?xi8>, i64) -> memref<f32>
      krnl.store %cst, %8[] : memref<f32>
      %c_ind_2 = constant 2 : index
      %dim2 = dim %arg2, %c_ind_2 : memref<1x?x?xf32>
      %9 = alloc(%dim2) : memref<?xi8>
      %10 = "krnl.getref"(%9, %c0_i64) : (memref<?xi8>, i64) -> memref<f32>
      krnl.store %cst, %10[] : memref<f32>
      %11 = krnl.define_loops 1
      krnl.iterate(%11) with (%11 -> %arg6 = 0 to 2) {
        %25 = krnl.load %arg0[%arg3, %arg4, %arg6] : memref<1x3x2xf32>
        %26 = krnl.load %arg1[%c0, %arg5, %arg6] : memref<1x4x2xf32>
        %27 = mulf %25, %26 : f32
        %28 = krnl.load %8[] : memref<f32>
        %29 = addf %28, %27 : f32
        krnl.store %29, %8[] : memref<f32>
        %30 = krnl.load %arg2[%c0, %arg5, %arg6] : memref<1x?x?xf32>
        %31 = mulf %6, %30 : f32
        %32 = krnl.load %10[] : memref<f32>
        %33 = addf %32, %31 : f32
        krnl.store %33, %10[] : memref<f32>
      }
      %12 = krnl.load %8[] : memref<f32>
      %13 = krnl.load %10[] : memref<f32>
      %14 = addf %12, %13 : f32
      %15 = alloc(%dim2) : memref<?xi8>
      %16 = "krnl.getref"(%15, %c0_i64) : (memref<?xi8>, i64) -> memref<f32>
      krnl.store %14, %16[] : memref<f32>
      %17 = krnl.load %16[] : memref<f32>
      %18 = subf %cst, %17 : f32
      %19 = math.exp %17 : f32
      %20 = math.exp %18 : f32
      %21 = subf %19, %20 : f32
      %22 = addf %19, %20 : f32
      %23 = divf %21, %22 : f32
      krnl.store %23, %5[] : memref<f32>
      %24 = krnl.load %5[] : memref<f32>
      krnl.store %24, %0[%c0, %arg4, %arg5] : memref<1x3x?xf32>
      dealloc %15 : memref<?xi8>
      dealloc %9 : memref<?xi8>
      dealloc %7 : memref<?xi8>
      dealloc %4 : memref<?xi8>
    }
  }
  return %0 : memref<1x3x?xf32>

  // CHECK-LABEL: test_dynamic_pool_rnn
  // CHECK: [[C1:%.+]] = constant 1 : index
  // CHECK: [[C2:%.+]] = constant 2 : index
  // CHECK: [[C0:%.+]] = constant 0 : i64
  // CHECK: krnl.define_loops 1
  // CHECK: krnl.iterate
  // CHECK: krnl.define_loops 2
  // CHECK: krnl.iterate
  // CHECK: [[DIM1:%.+]] = dim %arg2, [[C1]] : memref<1x?x?xf32>
  // CHECK: [[DIM2:%.+]] = dim %arg2, [[C2]] : memref<1x?x?xf32>
  // CHECK: [[ADD1:%.+]] = addi [[DIM2]], [[DIM2]] : index
  // CHECK: [[OFFSET1:%.+]] = index_cast [[DIM2]] : index to i64
  // CHECK: [[ADD2:%.+]] = addi [[ADD1]], [[DIM1]] : index
  // CHECK: [[OFFSET2:%.+]] = index_cast [[ADD1]] : index to i64
  // CHECK: [[ADD3:%.+]] = addi [[ADD2]], [[DIM1]] : index
  // CHECK: [[OFFSET3:%.+]] = index_cast [[ADD2]] : index to i64
  // CHECK: [[DYNAMIC_MEMORY_POOL:%.+]] = alloc([[ADD3]]) : memref<?xi8>
  // CHECK: [[DATA2:%.+]] = "krnl.getref"([[DYNAMIC_MEMORY_POOL]], [[OFFSET3]]) : (memref<?xi8>, i64) -> memref<f32>
  // CHECK: krnl.load
  // CHECK: [[DATA3:%.+]] = "krnl.getref"([[DYNAMIC_MEMORY_POOL]], [[OFFSET2]]) : (memref<?xi8>, i64) -> memref<f32>
  // CHECK: krnl.store
  // CHECK: [[DATA4:%.+]] = "krnl.getref"([[DYNAMIC_MEMORY_POOL]], [[OFFSET1]]) : (memref<?xi8>, i64) -> memref<f32>
  // CHECK: krnl.store
  // CHECK: krnl.define_loops
  // CHECK: krnl.iterate
  // CHECK: [[DATA1:%.+]] = "krnl.getref"([[DYNAMIC_MEMORY_POOL]], [[C0]]) : (memref<?xi8>, i64) -> memref<f32>
  // CHECK: dealloc [[DYNAMIC_MEMORY_POOL]] : memref<?xi8>
}
