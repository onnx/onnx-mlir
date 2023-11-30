// RUN: onnx-mlir-opt -O3 --convert-krnl-to-affine %s -split-input-file | FileCheck %s

func.func @test_getref_lowering(%arg0: memref<2x2xf32>) -> memref<2x2xf32> {
  %c13_i64 = arith.constant 13 : i64
  %1 = memref.alloc() : memref<10x10xf32>
  %2 = "krnl.getref"(%1, %c13_i64) : (memref<10x10xf32>, i64) -> memref<2x2xf32>
  return %2 : memref<2x2xf32>

  // CHECK-LABEL: test_getref_lowering
  // CHECK: %[[OFFSET:.+]] = llvm.mlir.constant(13 : i64) : i64
  // CHECK: [[CONST_10_0:%.+]] = llvm.mlir.constant(10 : index) : i64
  // CHECK: [[CONST_10_1:%.+]] = llvm.mlir.constant(10 : index) : i64
  // CHECK: [[CONST_1:%.+]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: %[[CONST_100:.+]] = llvm.mlir.constant(100 : index) : i64
  // CHECK: [[FLOAT_STAR:%.+]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: [[ELEM1:%.+]] = llvm.getelementptr [[FLOAT_STAR]][%[[CONST_100]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  // CHECK: [[ELEM_SIZE:%.+]] = llvm.ptrtoint [[ELEM1]] : !llvm.ptr to i64
  // CHECK: [[MEMPOOL:%.+]] = llvm.call @malloc([[ELEM_SIZE]]) : (i64) -> !llvm.ptr
  // CHECK: [[MEMPOOL_MEMREF:%.+]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
  // CHECK: [[MEMREF1:%.+]] = llvm.insertvalue [[MEMPOOL]], [[MEMPOOL_MEMREF]][0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
  // CHECK: [[MEMREF2:%.+]] = llvm.insertvalue [[MEMPOOL]], [[MEMREF1]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
  // CHECK: llvm.mlir.constant
  // CHECK: llvm.insertvalue
  // CHECK: llvm.insertvalue
  // CHECK: llvm.insertvalue
  // CHECK: llvm.insertvalue
  // CHECK: llvm.insertvalue
  // CHECK: [[MEMPOOL1:%.+]] = llvm.extractvalue {{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
  // CHECK: [[MEMPOOL_ALLOC:%.+]] = llvm.getelementptr [[MEMPOOL1]][%[[OFFSET]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  // CHECK: [[TYPED_MEMPOOL_ALLOC:%.+]] = llvm.bitcast [[MEMPOOL_ALLOC]] : !llvm.ptr to !llvm.ptr
  // CHECK: [[MEMPOOLED_MEMREF:%.+]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
  // CHECK: [[MEMREF3:%.+]] = llvm.insertvalue [[TYPED_MEMPOOL_ALLOC]], [[MEMPOOLED_MEMREF]][0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
  // CHECK: [[MEMREF4:%.+]] = llvm.insertvalue [[TYPED_MEMPOOL_ALLOC]], [[MEMREF3]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
}

// -----

func.func @test_getref_lowering_dynamic(%arg0: memref<2x2xf32>) -> memref<2x?xf32> {
  %c13_i64 = arith.constant 13 : i64
  %c5_index = arith.constant 5 : index
  %1 = memref.alloc(%c5_index) : memref<10x?xf32>
  %2 = "krnl.getref"(%1, %c13_i64, %c5_index) : (memref<10x?xf32>, i64, index) -> memref<2x?xf32>
  return %2 : memref<2x?xf32>

  // CHECK-LABEL: test_getref_lowering_dynamic
  // CHECK: %[[C13_I64:.+]] = llvm.mlir.constant(13 : i64) : i64
  // CHECK: %[[C5_INDEX:.+]] = llvm.mlir.constant(5 : index) : i64
  // CHECK: %[[C10_INDEX:.+]] = llvm.mlir.constant(10 : index) : i64
  // CHECK: %[[C1_INDEX:.+]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: %[[MUL1:.+]] = llvm.mul %[[C5_INDEX]], %[[C10_INDEX]] : i64
  // CHECK: [[FLOAT_STAR:%.+]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: [[ELEM1:%.+]] = llvm.getelementptr [[FLOAT_STAR]][%[[MUL1]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  // CHECK: [[ELEM_SIZE:%.+]] = llvm.ptrtoint [[ELEM1]] : !llvm.ptr to i64

  /// Allocate the memory pool alloc.
  // CHECK: [[ALLOC:%.+]] = llvm.call @malloc([[ELEM_SIZE]]) : (i64) -> !llvm.ptr

  /// Definition of the Alloc output memref<10x?xf32>
  // CHECK: [[ALLOC_MEMREF_1:%.+]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[ALLOC_MEMREF_2:%.+]] = llvm.insertvalue [[ALLOC]], [[ALLOC_MEMREF_1]][0]
  // CHECK: [[ALLOC_MEMREF_3:%.+]] = llvm.insertvalue [[ALLOC]], [[ALLOC_MEMREF_2]][1]
  // CHECK: [[CONST_0:%.+]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: [[ALLOC_MEMREF_4:%.+]] = llvm.insertvalue [[CONST_0]], [[ALLOC_MEMREF_3]][2]
  // CHECK: [[ALLOC_MEMREF_5:%.+]] = llvm.insertvalue %[[C10_INDEX]], [[ALLOC_MEMREF_4]][3, 0]
  // CHECK: [[ALLOC_MEMREF_6:%.+]] = llvm.insertvalue %[[C5_INDEX]], [[ALLOC_MEMREF_5]][3, 1]
  // CHECK: [[ALLOC_MEMREF_7:%.+]] = llvm.insertvalue %[[C5_INDEX]], [[ALLOC_MEMREF_6]][4, 0]
  // CHECK: [[ALLOC_MEMREF_8:%.+]] = llvm.insertvalue %[[C1_INDEX]], [[ALLOC_MEMREF_7]][4, 1]

  /// Fetch the allocated memory from the memory pool alloc.
  // CHECK: [[MEMPOOL:%.+]] = llvm.extractvalue [[ALLOC_MEMREF_8]][1]
  // CHECK: [[GETREF_START:%.+]] = llvm.getelementptr [[MEMPOOL]][%[[C13_I64]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  // CHECK: [[TYPED_GETREF_START:%.+]] = llvm.bitcast [[GETREF_START]] : !llvm.ptr to !llvm.ptr

  /// Definition of the krnl.getref output memref<2x?xf32>
  // CHECK: [[GETREF_MEMREF_1:%.+]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[GETREF_MEMREF_2:%.+]] = llvm.insertvalue [[TYPED_GETREF_START]], [[GETREF_MEMREF_1]][0]
  // CHECK: [[GETREF_MEMREF_3:%.+]] = llvm.insertvalue [[TYPED_GETREF_START]], [[GETREF_MEMREF_2]][1]
  // CHECK: [[CONST_0:%.+]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: [[GETREF_MEMREF_4:%.+]] = llvm.insertvalue [[CONST_0]], [[GETREF_MEMREF_3]][2]
  // CHECK: [[CONST_2:%.+]] = llvm.mlir.constant(2 : index) : i64
  // CHECK: [[CONST_1:%.+]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: [[MUL3:%.+]] = llvm.mul [[CONST_1]], %[[C5_INDEX]] : i64
  // CHECK: [[GETREF_MEMREF_5:%.+]] = llvm.insertvalue [[CONST_2]], [[GETREF_MEMREF_4]][3, 0]
  // CHECK: [[GETREF_MEMREF_6:%.+]] = llvm.insertvalue [[MUL3]], [[GETREF_MEMREF_5]][4, 0]
  // CHECK: [[GETREF_MEMREF_7:%.+]] = llvm.insertvalue %[[C5_INDEX]], [[GETREF_MEMREF_6]][3, 1]
  // CHECK: [[GETREF_MEMREF_8:%.+]] = llvm.insertvalue [[CONST_1]], [[GETREF_MEMREF_7]][4, 1]
  // CHECK: llvm.return [[GETREF_MEMREF_8]]
}
