// RUN: onnx-mlir-opt --convert-krnl-to-llvm %s -split-input-file | FileCheck %s

func @test_krnl_global_constant_alignment() -> memref<3xf32> {
  %0 = "krnl.global"() {name = "constant", alignment = 1024 : i64, shape = [3], value = dense<[0.0, 0.1, 0.2]> : tensor<3xf32>} : () -> memref<3xf32>
  return %0 : memref<3xf32>

// CHECK-LABEL:   llvm.func @free(!llvm.ptr<i8>)
// CHECK:         llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr<i8>
// CHECK:         llvm.mlir.global internal constant @constant(dense<[0.000000e+00, 1.000000e-01, 2.000000e-01]> : tensor<3xf32>) : !llvm.array<3 x f32>

// CHECK-LABEL:   llvm.func @test_krnl_global_constant_alignment() -> !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> {
// CHECK:           %[[GLOBAL_VALUE:.*]] = llvm.mlir.addressof @constant : !llvm.ptr<array<3 x f32>>
// CHECK-DAG:       %[[CONSTANT_1024:.*]] = llvm.mlir.constant(1024 : index) : i64
// CHECK-DAG:       %[[CONSTANT_12:.*]] = llvm.mlir.constant(12 : index) : i64
// CHECK:           %[[ALLOCATION_SIZE:.*]] = llvm.add %[[CONSTANT_12]], %[[CONSTANT_1024]]  : i64

// COM: Allocate a local buffer considering alignment.
// CHECK:           %[[I8_PTR_LOCAL:.*]] = llvm.call @malloc(%[[ALLOCATION_SIZE]]) : (i64) -> !llvm.ptr<i8>
// CHECK:           %[[F32_PTR_LOCAL:.*]] = llvm.bitcast %[[I8_PTR_LOCAL]] : !llvm.ptr<i8> to !llvm.ptr<f32>

// COM: Compute the aligned pointer.
// CHECK:           %[[PTR_TO_INT:.*]] = llvm.ptrtoint %[[F32_PTR_LOCAL]] : !llvm.ptr<f32> to i64
// CHECK:           %[[INDEX_1:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[SUB1:.*]] = llvm.sub %[[CONSTANT_1024]], %[[INDEX_1]]  : i64
// CHECK:           %[[BUMPED:.*]] = llvm.add %[[PTR_TO_INT]], %[[SUB1]]  : i64
// CHECK:           %[[REM:.*]] = llvm.urem %[[BUMPED]], %[[CONSTANT_1024]]  : i64
// CHECK:           %[[ALIGNED:.*]] = llvm.sub %[[BUMPED]], %[[REM]]  : i64
// CHECK:           %[[ALIGNED_PTR_LOCAL:.*]] = llvm.inttoptr %[[ALIGNED]] : i64 to !llvm.ptr<f32>

// COM: Copy constant values to the aligned buffer.
// CHECK:           %[[I8_ALIGNED_PTR_LOCAL:.*]] = llvm.bitcast %[[ALIGNED_PTR_LOCAL]] : !llvm.ptr<f32> to !llvm.ptr<i8>
// CHECK:           %[[I8_ALIGNED_PTR_GLOBAL:.*]] = llvm.bitcast %[[GLOBAL_VALUE]] : !llvm.ptr<array<3 x f32>> to !llvm.ptr<i8>
// CHECK:           %[[VOLATILE:.*]] = llvm.mlir.constant(false) : i1
// CHECK:           llvm.call @llvm.memcpy.p0i8.p0i8.i64(%[[I8_ALIGNED_PTR_LOCAL]], %[[I8_ALIGNED_PTR_GLOBAL]], %[[CONSTANT_12]], %[[VOLATILE]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()

// COM: Create a MemRef to wrap the buffers.
// CHECK:           %[[MEMREF_0:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[MEMREF_1:.*]] = llvm.insertvalue %[[F32_PTR_LOCAL]], %[[MEMREF_0]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[MEMREF_2:.*]] = llvm.insertvalue %[[F32_PTR_LOCAL]], %[[MEMREF_1]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[CONSTANT_0:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[MEMREF_3:.*]] = llvm.insertvalue %[[CONSTANT_0]], %[[MEMREF_2]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[CONSTANT_3:.*]] = llvm.mlir.constant(3 : index) : i64
// CHECK:           %[[MEMREF_4:.*]] = llvm.insertvalue %[[CONSTANT_3]], %[[MEMREF_3]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           %[[CONSTANT_1:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[MEMREF_5:.*]] = llvm.insertvalue %[[CONSTANT_1]], %[[MEMREF_4]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>

// COM: Update the MemRef with the new aligned pointer.
// CHECK:           %[[RES:.*]] = llvm.insertvalue %[[ALIGNED_PTR_LOCAL]], %[[MEMREF_5]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           llvm.call @free(%[[I8_PTR_LOCAL]]) : (!llvm.ptr<i8>) -> ()
// CHECK:           llvm.return %[[RES]] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:         }

}
