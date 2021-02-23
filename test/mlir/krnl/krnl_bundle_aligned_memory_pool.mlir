// RUN: onnx-mlir-opt --bundle-memory-pools --canonicalize %s -split-input-file | FileCheck %s

func @test_pool_bundling(%arg0: memref<10x10xf32>, %arg1: memref<10x20xf32>) -> memref<10x20xf32> {
  %c0_i64 = constant 0 : i64
  %ind = constant 0 : index
  %cst = constant 0.000000e+00 : f32
  %0 = alloc() : memref<10x20xf32>
  %1 = alloc() {alignment = 4096 : i64} : memref<800xi8>
  %2 = "krnl.getref"(%1, %c0_i64) : (memref<800xi8>, i64) -> memref<10x20xf32>
  %3 = alloc() {alignment = 4096 : i64} : memref<400xi8>
  %4 = "krnl.getref"(%3, %c0_i64) : (memref<400xi8>, i64) -> memref<10x10xf32>
  %5 = alloc() : memref<800xi8>
  %6 = "krnl.getref"(%5, %c0_i64) : (memref<800xi8>, i64) -> memref<10x20xf32>
  %7 = alloc() {alignment = 4096 : i64} : memref<800xi8>
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
  // CHECK: [[CONST_1200:%.+]] = constant 1200 : i64
  // CHECK: [[CONST_800:%.+]] = constant 800 : i64
  // CHECK: [[CONST_400:%.+]] = constant 400 : i64
  // CHECK: [[RES:%.+]] = alloc() : memref<10x20xf32>
  // CHECK: [[MEMPOOL_ALIGNED:%.+]] = alloc() {alignment = 4096 : i64} : memref<2000xi8>
  // CHECK: [[MEMREF1:%.+]] = "krnl.getref"([[MEMPOOL_ALIGNED]], [[CONST_1200]]) : (memref<2000xi8>, i64) -> memref<10x20xf32>
  // CHECK: [[MEMREF2:%.+]] = "krnl.getref"([[MEMPOOL_ALIGNED]], [[CONST_800]]) : (memref<2000xi8>, i64) -> memref<10x10xf32>
  // CHECK: [[MEMPOOL:%.+]] = alloc() : memref<1200xi8>
  // CHECK: [[MEMREF3:%.+]] = "krnl.getref"([[MEMPOOL]], [[CONST_400]]) : (memref<1200xi8>, i64) -> memref<10x20xf32>
  // CHECK: [[MEMREF4:%.+]] = "krnl.getref"([[MEMPOOL_ALIGNED]], [[CONST_0]]) : (memref<2000xi8>, i64) -> memref<10x20xf32>
  // CHECK: [[MEMREF5:%.+]] = "krnl.getref"([[MEMPOOL]], [[CONST_0]]) : (memref<1200xi8>, i64) -> memref<10x10xf32>
  // CHECK: krnl.store %cst, [[MEMREF5]]{{\[}}[[CONST_0_INDEX]], [[CONST_0_INDEX]]{{\]}} : memref<10x10xf32>
  // CHECK: krnl.store %cst, [[MEMREF4]]{{\[}}[[CONST_0_INDEX]], [[CONST_0_INDEX]]{{\]}} : memref<10x20xf32>
  // CHECK: krnl.store %cst, [[MEMREF3]]{{\[}}[[CONST_0_INDEX]], [[CONST_0_INDEX]]{{\]}} : memref<10x20xf32>
  // CHECK: krnl.store %cst, [[MEMREF2]]{{\[}}[[CONST_0_INDEX]], [[CONST_0_INDEX]]{{\]}} : memref<10x10xf32>
  // CHECK: krnl.store %cst, [[MEMREF1]]{{\[}}[[CONST_0_INDEX]], [[CONST_0_INDEX]]{{\]}} : memref<10x20xf32>
  // CHECK: krnl.store %cst, [[RES]]{{\[}}[[CONST_0_INDEX]], [[CONST_0_INDEX]]{{\]}} : memref<10x20xf32>
  // CHECK: dealloc [[MEMPOOL]] : memref<1200xi8>
  // CHECK: dealloc [[MEMPOOL_ALIGNED]] : memref<2000xi8>
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
  %3 = alloc(%2) {alignment = 4096 : i64} : memref<?xi8>
  %4 = "krnl.getref"(%3, %c0_i64, %0) : (memref<?xi8>, i64, index) -> memref<?x10xf32>
  %6 = cmpi "sgt", %0, %0 : index
  %7 = select %6, %0, %0 : index
  %8 = muli %7, %c4 : index
  %9 = muli %8, %c10 : index
  %10 = alloc(%9) : memref<?xi8>
  %11 = "krnl.getref"(%10, %c0_i64, %7) : (memref<?xi8>, i64, index) -> memref<?x10xf32>
  %12 = cmpi "eq", %0, %c1 : index
  %13 = cmpi "eq", %0, %c1 : index
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
  // CHECK: [[DYN_MEMPOOL:%.+]] = alloc([[OFFSET1]]) : memref<?xi8>
  // CHECK: [[DYN_MEMPOOL_ALIGNED:%.+]] = alloc([[OFFSET2]]) {alignment = 4096 : i64} : memref<?xi8>
  // CHECK: [[DATA1:%.+]] = "krnl.getref"([[DYN_MEMPOOL_ALIGNED]], [[C0_I64]], [[DIM]]) : (memref<?xi8>, i64, index) -> memref<?x10xf32>
  // CHECK: [[DATA2:%.+]] = "krnl.getref"([[DYN_MEMPOOL]], [[C0_I64]], [[DIM]]) : (memref<?xi8>, i64, index) -> memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM]]) : memref<?x10xf32>
  // CHECK: krnl.store [[CST]], [[DATA1]]{{\[}}[[C0]], [[C0]]{{\]}} : memref<?x10xf32>
  // CHECK: krnl.store [[CST]], [[DATA2]]{{\[}}[[C0]], [[C0]]{{\]}} : memref<?x10xf32>
  // CHECK: krnl.store [[CST]], [[RES]]{{\[}}[[C0]], [[C0]]{{\]}} : memref<?x10xf32>
  // CHECK: dealloc [[DYN_MEMPOOL]] : memref<?xi8>
  // CHECK: dealloc [[DYN_MEMPOOL_ALIGNED]] : memref<?xi8>
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
  %3 = alloc(%2) {alignment = 4096 : i64} : memref<?xi8>
  %4 = "krnl.getref"(%3, %c0_i64, %0) : (memref<?xi8>, i64, index) -> memref<?x10xf32>
  %const_alloc1 = alloc() : memref<800xi8>
  %const_ref1 = "krnl.getref"(%const_alloc1, %c0_i64) : (memref<800xi8>, i64) -> memref<10x20xf32>
  %const_alloc2 = alloc() {alignment = 4096 : i64} : memref<400xi8>
  %const_ref2 = "krnl.getref"(%const_alloc2, %c0_i64) : (memref<400xi8>, i64) -> memref<10x10xf32>
  %6 = cmpi "sgt", %0, %0 : index
  %7 = select %6, %0, %0 : index
  %8 = muli %7, %c4 : index
  %9 = muli %8, %c10 : index
  %10 = alloc(%9) : memref<?xi8>
  %11 = "krnl.getref"(%10, %c0_i64, %7) : (memref<?xi8>, i64, index) -> memref<?x10xf32>
  %12 = cmpi "eq", %0, %c1 : index
  %13 = cmpi "eq", %0, %c1 : index
  %15 = alloc(%0) {alignment = 4096 : i64} : memref<?x10xf32>
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
  // CHECK: [[C1600_I64:%.+]] = constant 1600 : i64
  // CHECK: [[C0_I64:%.+]] = constant 0 : i64
  // CHECK: [[DIM:%.+]] = dim %arg0, [[C0]] : memref<?x?xf32>
  // CHECK: [[MUL2:%.+]] = muli [[DIM]], [[C4]] : index
  // CHECK: [[OFFSET2:%.+]] = muli [[MUL2]], [[C10]] : index
  // CHECK: [[MUL1:%.+]] = muli [[DIM]], [[C4]] : index
  // CHECK: [[OFFSET1:%.+]] = muli [[MUL1]], [[C10]] : index
  // CHECK: [[DYN_MEMPOOL:%.+]] = alloc([[OFFSET1]]) : memref<?xi8>
  // CHECK: [[DYN_MEMPOOL_ALIGNED:%.+]] = alloc([[OFFSET2]]) {alignment = 4096 : i64} : memref<?xi8>
  // CHECK: [[DATA2:%.+]] = "krnl.getref"([[DYN_MEMPOOL_ALIGNED]], [[C0_I64]], [[DIM]]) : (memref<?xi8>, i64, index) -> memref<?x10xf32>
  // CHECK: [[STATIC_MEMPOOL:%.+]] = alloc() : memref<2400xi8>
  // CHECK: [[DATA3:%.+]] = "krnl.getref"([[STATIC_MEMPOOL]], [[C1600_I64]]) : (memref<2400xi8>, i64) -> memref<10x20xf32>
  // CHECK: [[STATIC_MEMPOOL_ALIGNED:%.+]] = alloc() {alignment = 4096 : i64} : memref<400xi8>
  // CHECK: [[DATA4:%.+]] = "krnl.getref"([[STATIC_MEMPOOL_ALIGNED]], [[C0_I64]]) : (memref<400xi8>, i64) -> memref<10x10xf32>
  // CHECK: [[DATA1:%.+]] = "krnl.getref"([[DYN_MEMPOOL]], [[C0_I64]], [[DIM]]) : (memref<?xi8>, i64, index) -> memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM]]) {alignment = 4096 : i64} : memref<?x10xf32>
  // CHECK: [[DATA5:%.+]] = "krnl.getref"([[STATIC_MEMPOOL]], [[C0_I64]]) : (memref<2400xi8>, i64) -> memref<10x40xf32>
  // CHECK: krnl.store [[CST]], [[DATA2]]{{\[}}[[C0]], [[C0]]{{\]}} : memref<?x10xf32>
  // CHECK: krnl.store [[CST]], [[DATA1]]{{\[}}[[C0]], [[C0]]{{\]}} : memref<?x10xf32>
  // CHECK: krnl.store [[CST]], [[RES]]{{\[}}[[C0]], [[C0]]{{\]}} : memref<?x10xf32>
  // CHECK: krnl.store [[CST]], [[DATA4]]{{\[}}[[C0]], [[C0]]{{\]}} : memref<10x10xf32>
  // CHECK: krnl.store [[CST]], [[DATA3]]{{\[}}[[C0]], [[C0]]{{\]}} : memref<10x20xf32>
  // CHECK: krnl.store [[CST]], [[DATA5]]{{\[}}[[C0]], [[C0]]{{\]}} : memref<10x40xf32>
  // CHECK: dealloc [[DYN_MEMPOOL]] : memref<?xi8>
  // CHECK: dealloc [[DYN_MEMPOOL_ALIGNED]] : memref<?xi8>
  // CHECK: dealloc [[STATIC_MEMPOOL_ALIGNED]] : memref<400xi8>
  // CHECK: dealloc [[STATIC_MEMPOOL]] : memref<2400xi8>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

