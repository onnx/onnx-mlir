// RUN: onnx-mlir-opt -O3 --bundle-memory-pools --canonicalize %s -split-input-file | FileCheck %s

func.func @test_pool_bundling(%arg0: memref<10x10xf32>, %arg1: memref<10x20xf32>) -> memref<10x20xf32> {
  %c0_i64 = arith.constant 0 : i64
  %ind = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = memref.alloc() : memref<10x20xf32>
  %1 = memref.alloc() {alignment = 4096 : i64} : memref<800xi8>
  %2 = "krnl.getref"(%1, %c0_i64) : (memref<800xi8>, i64) -> memref<10x20xf32>
  %3 = memref.alloc() {alignment = 4096 : i64} : memref<400xi8>
  %4 = "krnl.getref"(%3, %c0_i64) : (memref<400xi8>, i64) -> memref<10x10xf32>
  %5 = memref.alloc() : memref<800xi8>
  %6 = "krnl.getref"(%5, %c0_i64) : (memref<800xi8>, i64) -> memref<10x20xf32>
  %7 = memref.alloc() {alignment = 4096 : i64} : memref<800xi8>
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
  // CHECK-DAG: [[CONST_1200:%.+]] = arith.constant 8192 : i64
  // CHECK-DAG: [[CONST_800:%.+]] = arith.constant 4096 : i64
  // CHECK-DAG: [[CONST_400:%.+]] = arith.constant 400 : i64
  // CHECK-DAG: [[RES:%.+]] = memref.alloc() : memref<10x20xf32>
  // CHECK-DAG: [[MEMPOOL_ALIGNED:%.+]] = memref.alloc() {alignment = 4096 : i64} : memref<8992xi8>
  // CHECK: [[MEMREF1:%.+]] = "krnl.getref"([[MEMPOOL_ALIGNED]], [[CONST_1200]]) : (memref<8992xi8>, i64) -> memref<10x20xf32>
  // CHECK: [[MEMREF2:%.+]] = "krnl.getref"([[MEMPOOL_ALIGNED]], [[CONST_800]]) : (memref<8992xi8>, i64) -> memref<10x10xf32>
  // CHECK: [[MEMPOOL:%.+]] = memref.alloc() : memref<1200xi8>
  // CHECK: [[MEMREF3:%.+]] = "krnl.getref"([[MEMPOOL]], [[CONST_400]]) : (memref<1200xi8>, i64) -> memref<10x20xf32>
  // CHECK: [[MEMREF4:%.+]] = "krnl.getref"([[MEMPOOL_ALIGNED]], [[CONST_0]]) : (memref<8992xi8>, i64) -> memref<10x20xf32>
  // CHECK: [[MEMREF5:%.+]] = "krnl.getref"([[MEMPOOL]], [[CONST_0]]) : (memref<1200xi8>, i64) -> memref<10x10xf32>
  // CHECK: krnl.store %cst, [[MEMREF5]]{{\[}}[[CONST_0_INDEX]], [[CONST_0_INDEX]]{{\]}} : memref<10x10xf32>
  // CHECK: krnl.store %cst, [[MEMREF4]]{{\[}}[[CONST_0_INDEX]], [[CONST_0_INDEX]]{{\]}} : memref<10x20xf32>
  // CHECK: krnl.store %cst, [[MEMREF3]]{{\[}}[[CONST_0_INDEX]], [[CONST_0_INDEX]]{{\]}} : memref<10x20xf32>
  // CHECK: krnl.store %cst, [[MEMREF2]]{{\[}}[[CONST_0_INDEX]], [[CONST_0_INDEX]]{{\]}} : memref<10x10xf32>
  // CHECK: krnl.store %cst, [[MEMREF1]]{{\[}}[[CONST_0_INDEX]], [[CONST_0_INDEX]]{{\]}} : memref<10x20xf32>
  // CHECK: krnl.store %cst, [[RES]]{{\[}}[[CONST_0_INDEX]], [[CONST_0_INDEX]]{{\]}} : memref<10x20xf32>
  // CHECK: memref.dealloc [[MEMPOOL]] : memref<1200xi8>
  // CHECK: memref.dealloc [[MEMPOOL_ALIGNED]] : memref<8992xi8>
  // CHECK: return [[RES]] : memref<10x20xf32>
}
