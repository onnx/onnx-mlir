// RUN: onnx-mlir-opt --convert-krnl-to-affine --convert-krnl-to-llvm --canonicalize %s -split-input-file | FileCheck %s

func @test_random_normal_lowering() -> memref<3x4x5xf32> {
  %0 = memref.alloc() {alignment = 16 : i64} : memref<3x4x5xf32>
  %c60 = arith.constant 60 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %cst_1 = arith.constant 2.000000e+00 : f32
  "krnl.random_normal"(%0, %c60, %cst, %cst_0, %cst_1) : (memref<3x4x5xf32>, index, f32, f32, f32) -> ()
  return %0 : memref<3x4x5xf32>

  // CHECK: llvm.func @get_random_normal_value_f32(!llvm.ptr<f32>, i64, f32, f32, f32)
  // CHECK: llvm.func @malloc(i64) -> !llvm.ptr<i8>
  // CHECK: llvm.func @test_random_normal_lowering() -> !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)> {
  // CHECK: llvm.mlir.constant(3 : index) : i64
  // CHECK: llvm.mlir.constant(4 : index) : i64
  // CHECK: llvm.mlir.constant(5 : index) : i64
  // CHECK: llvm.mlir.constant(1 : index) : i64
  // CHECK: llvm.mlir.constant(20 : index) : i64
  // CHECK: llvm.mlir.constant(60 : index) : i64
  
  /// Allocate aligned tensor:
  // CHECK: llvm.mlir.null : !llvm.ptr<f32>
  // CHECK: llvm.getelementptr
  // CHECK: llvm.ptrtoint
  // CHECK: llvm.mlir.constant(16 : index) : i64
  // CHECK: llvm.add
  // CHECK: llvm.call @malloc
  // CHECK: llvm.bitcast
  // CHECK: llvm.ptrtoint
  // CHECK: llvm.mlir.constant(1 : index) : i64
  // CHECK: llvm.sub
  // CHECK: llvm.add
  // CHECK: llvm.urem
  // CHECK: llvm.sub
  // CHECK: [[ALIGNED_TENSOR_MEMORY:%.+]] = llvm.inttoptr %18 : i64 to !llvm.ptr<f32>

  /// Populate tensor:
  // CHECK: llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
  // CHECK: llvm.insertvalue
  // CHECK: llvm.insertvalue
  // CHECK: llvm.mlir.constant(0 : index) : i64
  // CHECK: llvm.insertvalue
  // CHECK: llvm.insertvalue
  // CHECK: llvm.insertvalue
  // CHECK: llvm.insertvalue
  // CHECK: llvm.insertvalue
  // CHECK: llvm.insertvalue
  // CHECK: [[OUTPUT_TENSOR:%.+]] = llvm.insertvalue
  // CHECK: [[ALL_VALUES:%.+]] = llvm.mlir.constant(60 : index) : i64
  // CHECK: [[MEAN:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
  // CHECK: [[SCALE:%.+]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
  // CHECK: [[SEED:%.+]] = llvm.mlir.constant(2.000000e+00 : f32) : f32
  // CHECK: llvm.call @get_random_normal_value_f32([[ALIGNED_TENSOR_MEMORY]], [[ALL_VALUES]], [[MEAN]], [[SCALE]], [[SEED]]) : (!llvm.ptr<f32>, i64, f32, f32, f32) -> ()
  // CHECK: llvm.return [[OUTPUT_TENSOR]] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<3 x i64>, array<3 x i64>)>
}
