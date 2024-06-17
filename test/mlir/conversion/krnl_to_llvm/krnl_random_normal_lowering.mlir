// RUN: onnx-mlir-opt -O3 --convert-krnl-to-affine --convert-krnl-to-llvm --canonicalize %s -split-input-file | FileCheck %s

func.func @test_random_normal_lowering() -> memref<3x4x5xf32> {
  %0 = memref.alloc() {alignment = 16 : i64} : memref<3x4x5xf32>
  %c60 = arith.constant 60 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %cst_1 = arith.constant 2.000000e+00 : f32
  "krnl.random_normal"(%0, %c60, %cst, %cst_0, %cst_1) : (memref<3x4x5xf32>, index, f32, f32, f32) -> ()
  return %0 : memref<3x4x5xf32>

  // CHECK-LABEL: llvm.func @get_random_normal_value_f32(!llvm.ptr, i64, f32, f32, f32)
  // CHECK: llvm.func @malloc(i64) -> !llvm.ptr
  // CHECK-LABEL: llvm.func @test_random_normal_lowering() -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> attributes {llvm.emit_c_interface} {

  // CHECK: [[SEED:%.+]] = llvm.mlir.constant(2.000000e+00 : f32) : f32
  // CHECK: [[SCALE:%.+]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
  // CHECK: [[MEAN:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
  // CHECK: [[ALL_VALUES:%.+]] = llvm.mlir.constant(60 : index) : i64

  /// Populate tensor:
  // CHECK: [[ALIGNED_TENSOR_MEMORY:%.+]] = llvm.inttoptr {{.*}} : i64 to !llvm.ptr
  // CHECK: [[OUTPUT_TENSOR:%.+]] = llvm.insertvalue {{.*}}, {{.*}}[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
  // CHECK: llvm.call @get_random_normal_value_f32([[ALIGNED_TENSOR_MEMORY]], [[ALL_VALUES]], [[MEAN]], [[SCALE]], [[SEED]]) : (!llvm.ptr, i64, f32, f32, f32) -> ()
  // CHECK: llvm.return [[OUTPUT_TENSOR]] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
}

// -----

func.func @test_random_normal_dynamic_lowering(%arg0: memref<3x4x?x?xf32>) -> memref<3x4x?x?xf32> {
  %c2 = arith.constant 2 : index
  %0 = memref.dim %arg0, %c2 : memref<3x4x?x?xf32>
  %c3 = arith.constant 3 : index
  %1 = memref.dim %arg0, %c3 : memref<3x4x?x?xf32>
  %2 = memref.alloc(%0, %1) {alignment = 16 : i64} : memref<3x4x?x?xf32>
  %c12 = arith.constant 12 : index
  %c2_0 = arith.constant 2 : index
  %3 = memref.dim %arg0, %c2_0 : memref<3x4x?x?xf32>
  %4 = arith.muli %c12, %3 : index
  %c3_1 = arith.constant 3 : index
  %5 = memref.dim %arg0, %c3_1 : memref<3x4x?x?xf32>
  %6 = arith.muli %4, %5 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_2 = arith.constant 1.000000e+00 : f32
  %cst_3 = arith.constant 2.000000e+00 : f32
  "krnl.random_normal"(%2, %6, %cst, %cst_2, %cst_3) : (memref<3x4x?x?xf32>, index, f32, f32, f32) -> ()
  return %2 : memref<3x4x?x?xf32>

  // CHECK-LABEL: llvm.func @get_random_normal_value_f32(!llvm.ptr, i64, f32, f32, f32)
  // CHECK: llvm.func @malloc(i64) -> !llvm.ptr
  // CHECK-LABEL: llvm.func @test_random_normal_dynamic_lowering(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64) -> !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> attributes {llvm.emit_c_interface} {

  // CHECK: [[SEED:%.+]] = llvm.mlir.constant(2.000000e+00 : f32) : f32
  // CHECK: [[SCALE:%.+]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
  // CHECK: [[MEAN:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
  // CHECK: [[ALL_VALUES1:%.+]] = llvm.mlir.constant(12 : index) : i64
  // CHECK: [[C0:%.+]] = llvm.mlir.zero : !llvm.ptr
  // CHECK: [[C4:%.+]] = llvm.mlir.constant(4 : index) : i64
  // CHECK: [[C3:%.+]] = llvm.mlir.constant(3 : index) : i64

  // CHECK: [[MUL1:%.+]] = llvm.mul %arg6, %arg5  : i64
  // CHECK: [[MUL2:%.+]] = llvm.mul [[MUL1]], [[C4]]  : i64
  // CHECK: %[[MUL3:.+]] = llvm.mul [[MUL2]], [[C3]]  : i64

  /// Allocate aligned tensor:
  // CHECK: llvm.getelementptr [[C0]][%[[MUL3]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  // CHECK: [[ALIGNED_TENSOR_MEMORY:%.+]] = llvm.inttoptr {{.*}} : i64 to !llvm.ptr

  /// Populate tensor:
  // CHECK: [[OUTPUT_TENSOR:%.+]] = llvm.insertvalue {{.*}}, {{.*}}[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
  
  // CHECK: [[ALL_VALUES2:%.+]] = llvm.mul %arg5, [[ALL_VALUES1]]  : i64
  // CHECK: [[ALL_VALUES3:%.+]] = llvm.mul [[ALL_VALUES2]], %arg6  : i64
  // CHECK: llvm.call @get_random_normal_value_f32([[ALIGNED_TENSOR_MEMORY]], [[ALL_VALUES3]], [[MEAN]], [[SCALE]], [[SEED]]) : (!llvm.ptr, i64, f32, f32, f32) -> ()
  // CHECK: llvm.return [[OUTPUT_TENSOR]] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
}
