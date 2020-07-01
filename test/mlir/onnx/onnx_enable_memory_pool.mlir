// RUN: onnx-mlir-opt --shape-inference --lower-frontend --enable-memory-pool %s -split-input-file | FileCheck %s

/// One intermediate value to allocate in the memory pool.
func @test_enable_memory_pool(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Add"(%arg0, %arg0) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %1 = "onnx.Add"(%0, %arg0) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  return %1 : tensor<10x10xf32>

  // CHECK-LABEL: test_enable_memory_pool
  // CHECK: [[CONST0:%.+]] = constant 0 : i64
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[MEMPOOL:%.+]] = alloc() : memref<400xi8>
  // CHECK: [[GETREF:%.+]] = "krnl.getref"([[MEMPOOL]], [[CONST0]]) : (memref<400xi8>, i64) -> memref<10x10xf32>
  // CHECK: krnl.define_loops
  // CHECK: krnl.iterate
  // CHECK: [[LOAD1:%.+]] = load %arg0[%arg1, %arg2] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = load %arg0[%arg1, %arg2] : memref<10x10xf32>
  // CHECK: [[ADDF1:%.+]] = addf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: store [[ADDF1]], [[GETREF]][%arg1, %arg2] : memref<10x10xf32>
  // CHECK: krnl.define_loops
  // CHECK: krnl.iterate
  // CHECK: dealloc [[MEMPOOL]] : memref<400xi8>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

/// Two intermediate values to allocate in the memory pool.
func @test_enable_memory_pool_2(%arg0: tensor<10x10xf32>, %arg1: tensor<10x20xf32>) -> tensor<10x20xf32> {
  %0 = "onnx.Add"(%arg0, %arg0) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %1 = "onnx.MatMul"(%0, %arg1) : (tensor<10x10xf32>, tensor<10x20xf32>) -> tensor<10x20xf32>
  %2 = "onnx.Add"(%1, %arg1) : (tensor<10x20xf32>, tensor<10x20xf32>) -> tensor<10x20xf32>
  return %2 : tensor<10x20xf32>

  // CHECK-LABEL: test_enable_memory_pool_2
  // CHECK: [[CONST0:%.+]] = constant 0 : i64
  // CHECK: [[CONST1:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[RES:%.+]] = alloc() : memref<10x20xf32>
  // CHECK: [[MEMPOOL0:%.+]] = alloc() : memref<800xi8>
  // CHECK: [[GETREF0:%.+]] = "krnl.getref"([[MEMPOOL0]], [[CONST0]]) : (memref<800xi8>, i64) -> memref<10x20xf32>
  // CHECK: [[MEMPOOL1:%.+]] = alloc() : memref<400xi8>
  // CHECK: [[GETREF1:%.+]] = "krnl.getref"([[MEMPOOL1]], [[CONST0]]) : (memref<400xi8>, i64) -> memref<10x10xf32>
  // CHECK: krnl.define_loops
  // CHECK: krnl.iterate
  // CHECK: [[LOAD1:%.+]] = load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[ADDF1:%.+]] = addf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: store [[ADDF1]], [[GETREF1]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: krnl.define_loops
  // CHECK: krnl.iterate
  // CHECK: [[LOAD3:%.+]] = load [[GETREF1]][%arg2, %arg4] : memref<10x10xf32>
  // CHECK: [[LOAD4:%.+]] = load %arg1[%arg4, %arg3] : memref<10x20xf32>
  // CHECK: [[LOAD5:%.+]] = load [[GETREF0]][%arg2, %arg3] : memref<10x20xf32>
  // CHECK: [[MULF1:%.+]] = mulf [[LOAD3]], [[LOAD4]] : f32
  // CHECK: [[ADDF2:%.+]] = addf [[LOAD5]], [[MULF1]] : f32
  // CHECK: store [[ADDF2]], [[GETREF0]][%arg2, %arg3] : memref<10x20xf32>
  // CHECK: krnl.define_loops
  // CHECK: krnl.iterate
  // CHECK: [[LOAD6:%.+]] = load [[GETREF0]][%arg2, %arg3] : memref<10x20xf32>
  // CHECK: [[LOAD7:%.+]] = load %arg1[%arg2, %arg3] : memref<10x20xf32>
  // CHECK: [[ADDF3:%.+]] = addf [[LOAD6]], [[LOAD7]] : f32
  // CHECK: store [[ADDF3]], [[RES]][%arg2, %arg3] : memref<10x20xf32>
  // CHECK: dealloc [[MEMPOOL1]] : memref<400xi8>
  // CHECK: dealloc [[MEMPOOL0]] : memref<800xi8>
  // CHECK: return [[RES]] : memref<10x20xf32>
}
