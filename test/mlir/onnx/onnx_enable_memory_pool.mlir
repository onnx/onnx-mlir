// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --enable-memory-pool %s -split-input-file | FileCheck %s

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
  // CHECK: [[LOAD1:%.+]] = affine.load %arg0[%arg1, %arg2] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = affine.load %arg0[%arg1, %arg2] : memref<10x10xf32>
  // CHECK: [[ADDF1:%.+]] = addf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: affine.store [[ADDF1]], [[GETREF]][%arg1, %arg2] : memref<10x10xf32>
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
  // CHECK: [[LOAD1:%.+]] = affine.load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = affine.load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[ADDF1:%.+]] = addf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: affine.store [[ADDF1]], [[GETREF1]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: krnl.define_loops
  // CHECK: krnl.iterate
  // CHECK: [[LOAD3:%.+]] = affine.load [[GETREF1]][%arg2, %arg4] : memref<10x10xf32>
  // CHECK: [[LOAD4:%.+]] = affine.load %arg1[%arg4, %arg3] : memref<10x20xf32>
  // CHECK: [[LOAD5:%.+]] = affine.load [[GETREF0]][%arg2, %arg3] : memref<10x20xf32>
  // CHECK: [[MULF1:%.+]] = mulf [[LOAD3]], [[LOAD4]] : f32
  // CHECK: [[ADDF2:%.+]] = addf [[LOAD5]], [[MULF1]] : f32
  // CHECK: affine.store [[ADDF2]], [[GETREF0]][%arg2, %arg3] : memref<10x20xf32>
  // CHECK: krnl.define_loops
  // CHECK: krnl.iterate
  // CHECK: [[LOAD6:%.+]] = affine.load [[GETREF0]][%arg2, %arg3] : memref<10x20xf32>
  // CHECK: [[LOAD7:%.+]] = affine.load %arg1[%arg2, %arg3] : memref<10x20xf32>
  // CHECK: [[ADDF3:%.+]] = addf [[LOAD6]], [[LOAD7]] : f32
  // CHECK: affine.store [[ADDF3]], [[RES]][%arg2, %arg3] : memref<10x20xf32>
  // CHECK: dealloc [[MEMPOOL1]] : memref<400xi8>
  // CHECK: dealloc [[MEMPOOL0]] : memref<800xi8>
  // CHECK: return [[RES]] : memref<10x20xf32>
}

// Two intermediate dynamic sized MemRefs.
func @test_enable_memory_pool_3(%arg0: tensor<?x?xf32>, %arg1: tensor<?x10xf32>, %arg2: tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x?xf32>, tensor<?x10xf32>) -> tensor<*xf32>
  %1 = "onnx.Add"(%0, %0) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %2 = "onnx.MatMul"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %2 : tensor<*xf32>

  // CHECK-LABEL: test_enable_memory_pool_3
  // CHECK: [[CONST4:%.+]] = constant 4 : index
  // CHECK: [[CONST10:%.+]] = constant 10 : index
  // CHECK: [[CONST0_I64:%.+]] = constant 0 : i64
  // CHECK: [[CONST1:%.+]] = constant 1 : index
  // CHECK: [[CONST0:%.+]] = constant 0 : index
  // CHECK: [[CST:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[DIM1:%.+]] = dim %arg0, [[CONST0]] : memref<?x?xf32>
  // CHECK: [[TMP1:%.+]] = muli [[DIM1]], [[CONST4]] : index
  // CHECK: [[TMP2:%.+]] = muli [[TMP1]], [[CONST10]] : index
  // CHECK: [[MEMPOOL1:%.+]] = alloc([[TMP2]]) : memref<?xi8>
  // CHECK: [[DATA1:%.+]] = "krnl.getref"([[MEMPOOL1]], [[CONST0_I64]]) : (memref<?xi8>, i64) -> memref<?x10xf32>
  // CHECK: krnl.define_loops 2
  // CHECK: krnl.iterate
  // CHECK: affine.store {{.*}}, [[DATA1]][%arg3, %arg4] : memref<?x10xf32>
  // CHECK: krnl.define_loops 1
  // CHECK: krnl.iterate
  // CHECK: affine.store {{.*}}[%arg3, %arg4] : memref<?x10xf32>
  // CHECK: [[TMP3:%.+]] = muli [[DIM1]], [[CONST4]] : index
  // CHECK: [[TMP4:%.+]] = muli [[TMP3]], [[CONST10]] : index
  // CHECK: [[MEMPOOL2:%.+]] = alloc([[TMP4]]) : memref<?xi8>
  // CHECK: [[DATA2:%.+]] = "krnl.getref"([[MEMPOOL2]], [[CONST0_I64]]) : (memref<?xi8>, i64) -> memref<?x10xf32>
  // CHECK: [[CMP1:%.+]] = cmpi "eq", [[DIM1]], [[CONST1]] : index
  // CHECK: krnl.define_loops 2
  // CHECK: krnl.iterate
  // CHECK: affine.store {{.*}}, [[DATA2]][%arg3, %arg4] : memref<?x10xf32>
  // CHECK: [[DATA3:%.+]] = alloc([[DIM1]]) : memref<?x10xf32>
  // CHECK: krnl.define_loops 2
  // CHECK: krnl.iterate
  // CHECK: affine.store [[CST]], [[DATA3]][%arg3, %arg4] : memref<?x10xf32>
  // CHECK: dealloc [[MEMPOOL2]] : memref<?xi8>
  // CHECK: dealloc [[MEMPOOL1]] : memref<?xi8>
  // CHECK: return [[DATA3]] : memref<?x10xf32>
}
