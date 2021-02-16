// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --enable-memory-pool --convert-krnl-to-affine --convert-krnl-to-llvm %s -split-input-file | FileCheck %s

func @test_memory_pool(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Add"(%arg0, %arg0) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %1 = "onnx.Add"(%0, %arg0) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  return %1 : tensor<10x10xf32>

  /// Define the offset inside the memory pool.
  // CHECK: %[[OFFSET:.+]] = llvm.mlir.constant(0 : i64) : i64

  /// Allocate memory for the memory pool.
  // CHECK: %[[MEMPOOL_SIZE:.+]] = llvm.mlir.constant(400 : index) : i64
  // CHECK: [[TMP1:%.+]] = llvm.mlir.null : !llvm.ptr<i8>
  // CHECK: [[TMP2:%.+]] = llvm.getelementptr [[TMP1]][%[[MEMPOOL_SIZE]]] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
  // CHECK: [[TYPE_SIZE_IN_BYTES:%.+]] = llvm.ptrtoint [[TMP2]] : !llvm.ptr<i8> to i64
  // CHECK: [[ALLOC_MEM_POOL:%.+]] = llvm.call @malloc([[TYPE_SIZE_IN_BYTES]]) : (i64) -> !llvm.ptr<i8>
  // CHECK: [[BITCAST_ALLOC_MEM_POOL:%.+]] = llvm.bitcast [[ALLOC_MEM_POOL]] : !llvm.ptr<i8> to !llvm.ptr<i8>

  /// MemRef representing the memory pool and which contains the memory allocated above.
  // CHECK: [[MEMREF0:%.+]] = llvm.mlir.undef : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)> 
  // CHECK: [[TMP3:%.+]] = llvm.insertvalue [[BITCAST_ALLOC_MEM_POOL]], [[MEMREF0]][0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)> 
  // CHECK: llvm.insertvalue [[BITCAST_ALLOC_MEM_POOL]], [[TMP3]][1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)> 
  // CHECK: llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue
  // CHECK: llvm.insertvalue
  // CHECK: [[TMP4:%.+]] = llvm.insertvalue {{.*}}[4, 0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>

  /// Get reference within the memory pool where the data of the getref instruction has already been allocated.
  // CHECK: [[MEMPOOL_BASE:%.+]] = llvm.extractvalue [[TMP4]][1] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)>
  // CHECK: [[GETREF_MEMORY:%.+]] = llvm.getelementptr [[MEMPOOL_BASE]][%[[OFFSET]]] : (!llvm.ptr<i8>, i64) -> !llvm.ptr<i8>
  // CHECK: [[CASTED_GETREF_MEMORY:%.+]] = llvm.bitcast [[GETREF_MEMORY]] : !llvm.ptr<i8> to !llvm.ptr<f32>

  /// Create MemRef for krnl.getref.
  // CHECK: [[MEMREF1:%.+]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
  // CHECK: [[MEMREF1_TMP1:%.+]] = llvm.insertvalue [[CASTED_GETREF_MEMORY]], [[MEMREF1]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
  // CHECK: [[MEMREF1_TMP2:%.+]] = llvm.insertvalue [[CASTED_GETREF_MEMORY]], [[MEMREF1_TMP1]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
  // CHECK: [[CONST2:%.+]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: [[MEMREF1_TMP3:%.+]] = llvm.insertvalue [[CONST2]], [[MEMREF1_TMP2]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[CONST3:%.+]] = llvm.mlir.constant(10 : index) : i64
  // CHECK: [[MEMREF1_TMP4:%.+]] = llvm.insertvalue [[CONST3]], [[MEMREF1_TMP3]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[CONST4:%.+]] = llvm.mlir.constant(10 : index) : i64
  // CHECK: [[MEMREF1_TMP5:%.+]] = llvm.insertvalue [[CONST4]], [[MEMREF1_TMP4]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[CONST5:%.+]] = llvm.mlir.constant(10 : index) : i64
  // CHECK: [[MEMREF1_TMP6:%.+]] = llvm.insertvalue [[CONST5]], [[MEMREF1_TMP5]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[CONST6:%.+]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: [[MEMREF1_TMP7:%.+]] = llvm.insertvalue [[CONST6]], [[MEMREF1_TMP6]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 

  /// Usage of the getref MemRef.
  // CHECK: [[MEM0:%.+]] = llvm.extractvalue [[MEMREF1_TMP7]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[CONST7:%.+]] = llvm.mlir.constant(10 : index) : i64
  // CHECK: [[MUL1:%.+]] = llvm.mul {{.*}}, [[CONST7]] : i64
  // CHECK: %[[ADD1:.+]] = llvm.add [[MUL1]], {{.*}} : i64
  // CHECK: llvm.getelementptr [[MEM0]][%[[ADD1]]] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>

  /// Deallocation of the memory pool.
  // CHECK: [[MEMPOOL_BASE_UNALIGNED:%.+]] = llvm.extractvalue [[TMP4]][0] : !llvm.struct<(ptr<i8>, ptr<i8>, i64, array<1 x i64>, array<1 x i64>)> 
  // CHECK: [[CASTED_MEMPOOL_BASE_UNALIGNED:%.+]] = llvm.bitcast [[MEMPOOL_BASE_UNALIGNED]] : !llvm.ptr<i8> to !llvm.ptr<i8>
  // CHECK: llvm.call @free([[CASTED_MEMPOOL_BASE_UNALIGNED]]) : (!llvm.ptr<i8>) -> ()
}
