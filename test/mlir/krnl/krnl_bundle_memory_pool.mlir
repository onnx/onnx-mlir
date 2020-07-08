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
