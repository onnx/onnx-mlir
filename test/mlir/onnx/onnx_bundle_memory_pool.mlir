// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --enable-memory-pool --bundle-memory-pools --canonicalize %s -split-input-file | FileCheck %s
// XFAIL: *

func @test_bundle_memory_pool(%arg0: tensor<10x10xf32>, %arg1: tensor<10x20xf32>) -> tensor<10x20xf32> {
  %0 = "onnx.Add"(%arg0, %arg0) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %1 = "onnx.MatMul"(%0, %arg1) : (tensor<10x10xf32>, tensor<10x20xf32>) -> tensor<10x20xf32>
  %2 = "onnx.Add"(%1, %arg1) : (tensor<10x20xf32>, tensor<10x20xf32>) -> tensor<10x20xf32>
  %3 = "onnx.Add"(%0, %arg0) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %4 = "onnx.MatMul"(%3, %arg1) : (tensor<10x10xf32>, tensor<10x20xf32>) -> tensor<10x20xf32>
  %5 = "onnx.Add"(%4, %arg1) : (tensor<10x20xf32>, tensor<10x20xf32>) -> tensor<10x20xf32>
  return %5 : tensor<10x20xf32>

  // CHECK-LABEL: test_bundle_memory_pool
  // CHECK: [[CONST0:%.+]] = constant 0 : i64
  // CHECK: [[CONST00:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[CONST2400:%.+]] = constant 2400 : i64
  // CHECK: [[CONST2000:%.+]] = constant 2000 : i64
  // CHECK: [[CONST1200:%.+]] = constant 1200 : i64
  // CHECK: [[CONST400:%.+]] = constant 400 : i64
  // CHECK: [[RES:%.+]] = memref.alloc() : memref<10x20xf32>
  // CHECK: [[MEMPOOL:%.+]] = memref.alloc() : memref<3200xi8>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[CONST2400]]) : (memref<3200xi8>, i64) -> memref<10x20xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[CONST2000]]) : (memref<3200xi8>, i64) -> memref<10x10xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[CONST1200]]) : (memref<3200xi8>, i64) -> memref<10x20xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[CONST400]]) : (memref<3200xi8>, i64) -> memref<10x20xf32>
  // CHECK: "krnl.getref"([[MEMPOOL]], [[CONST0]]) : (memref<3200xi8>, i64) -> memref<10x10xf32>
  // CHECK: memref.dealloc [[MEMPOOL]] : memref<3200xi8>
  // CHECK: return [[RES]] : memref<10x20xf32>
}
