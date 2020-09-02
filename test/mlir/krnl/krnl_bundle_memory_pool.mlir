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
  affine.store %cst, %10[%ind, %ind] : memref<10x10xf32>
  affine.store %cst, %8[%ind, %ind] : memref<10x20xf32>
  affine.store %cst, %6[%ind, %ind] : memref<10x20xf32>
  affine.store %cst, %4[%ind, %ind] : memref<10x10xf32>
  affine.store %cst, %2[%ind, %ind] : memref<10x20xf32>
  affine.store %cst, %0[%ind, %ind] : memref<10x20xf32>
  dealloc %9 : memref<400xi8>
  dealloc %7 : memref<800xi8>
  dealloc %5 : memref<800xi8>
  dealloc %3 : memref<400xi8>
  dealloc %1 : memref<800xi8>
  return %0 : memref<10x20xf32>

  // CHECK-LABEL: test_pool_bundling
  // CHECK: [[CONST_0:%.+]] = constant 0 : i64
  // CHECK: [[CONST_CST:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[CONST_400:%.+]] = constant 400 : i64
  // CHECK: [[CONST_1200:%.+]] = constant 1200 : i64
  // CHECK: [[CONST_2000:%.+]] = constant 2000 : i64
  // CHECK: [[CONST_2400:%.+]] = constant 2400 : i64
  // CHECK: [[RES:%.+]] = alloc() : memref<10x20xf32>
  // CHECK: [[MEMPOOL:%.+]] = alloc() : memref<3200xi8>
  // CHECK: [[MEMREF1:%.+]] = "krnl.getref"([[MEMPOOL]], [[CONST_2400]]) : (memref<3200xi8>, i64) -> memref<10x20xf32>
  // CHECK: [[MEMREF2:%.+]] = "krnl.getref"([[MEMPOOL]], [[CONST_2000]]) : (memref<3200xi8>, i64) -> memref<10x10xf32>
  // CHECK: [[MEMREF3:%.+]] = "krnl.getref"([[MEMPOOL]], [[CONST_1200]]) : (memref<3200xi8>, i64) -> memref<10x20xf32>
  // CHECK: [[MEMREF4:%.+]] = "krnl.getref"([[MEMPOOL]], [[CONST_400]]) : (memref<3200xi8>, i64) -> memref<10x20xf32>
  // CHECK: [[MEMREF5:%.+]] = "krnl.getref"([[MEMPOOL]], [[CONST_0]]) : (memref<3200xi8>, i64) -> memref<10x10xf32>
  // CHECK: affine.store %cst, [[MEMREF5]][0, 0] : memref<10x10xf32>
  // CHECK: affine.store %cst, [[MEMREF4]][0, 0] : memref<10x20xf32>
  // CHECK: affine.store %cst, [[MEMREF3]][0, 0] : memref<10x20xf32>
  // CHECK: affine.store %cst, [[MEMREF2]][0, 0] : memref<10x10xf32>
  // CHECK: affine.store %cst, [[MEMREF1]][0, 0] : memref<10x20xf32>
  // CHECK: affine.store %cst, [[RES]][0, 0] : memref<10x20xf32>
  // CHECK: dealloc [[MEMPOOL]] : memref<3200xi8>
  // CHECK: return [[RES]] : memref<10x20xf32>
}

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
  %4 = "krnl.getref"(%3, %c0_i64) : (memref<?xi8>, i64) -> memref<?x10xf32>
  %6 = cmpi "sgt", %0, %0 : index
  %7 = select %6, %0, %0 : index
  %8 = muli %7, %c4 : index
  %9 = muli %8, %c10 : index
  %10 = alloc(%9) : memref<?xi8>
  %11 = "krnl.getref"(%10, %c0_i64) : (memref<?xi8>, i64) -> memref<?x10xf32>
  %12 = cmpi "eq", %0, %c1 : index
  %13 = cmpi "eq", %0, %c1 : index
  %15 = alloc(%0) : memref<?x10xf32>
  affine.store %cst, %4[%ind, %ind] : memref<?x10xf32>
  affine.store %cst, %11[%ind, %ind] : memref<?x10xf32>
  affine.store %cst, %15[%ind, %ind] : memref<?x10xf32>
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
  // CHECK: [[SGT:%.+]] = cmpi "sgt", [[DIM]], [[DIM]] : index
  // CHECK: [[SELECT:%.+]] = select [[SGT]], [[DIM]], [[DIM]] : index
  // CHECK: [[MUL1:%.+]] = muli [[SELECT]], [[C4]] : index
  // CHECK: [[OFFSET1:%.+]] = muli [[MUL1]], [[C10]] : index
  // CHECK: [[MUL2:%.+]] = muli [[DIM]], [[C4]] : index
  // CHECK: [[OFFSET2:%.+]] = muli [[MUL2]], [[C10]] : index
  // CHECK: [[MEMPOOL_SIZE:%.+]] = addi [[OFFSET1]], [[OFFSET2]] : index
  // CHECK: [[OFFSET1_I64:%.+]] = index_cast [[OFFSET1]] : index to i64
  // CHECK: [[DYN_MEMPOOL:%.+]] = alloc([[MEMPOOL_SIZE]]) : memref<?xi8>
  // CHECK: [[DATA1:%.+]] = "krnl.getref"([[DYN_MEMPOOL]], [[OFFSET1_I64]]) : (memref<?xi8>, i64) -> memref<?x10xf32>
  // CHECK: [[DATA2:%.+]] = "krnl.getref"([[DYN_MEMPOOL]], [[C0_I64]]) : (memref<?xi8>, i64) -> memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM]]) : memref<?x10xf32>
  // CHECK: affine.store [[CST]], [[DATA1]][0, 0] : memref<?x10xf32>
  // CHECK: affine.store [[CST]], [[DATA2]][0, 0] : memref<?x10xf32>
  // CHECK: affine.store [[CST]], [[RES]][0, 0] : memref<?x10xf32>
  // CHECK: dealloc [[DYN_MEMPOOL]] : memref<?xi8>
  // CHECK: return [[RES]] : memref<?x10xf32>
}
