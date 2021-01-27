// RUN: onnx-mlir-opt --convert-krnl-to-affine --convert-krnl-to-llvm %s -split-input-file | FileCheck %s

// ----

/// Test lowering of krnl.erf to LLVM math function call.
func @test_krnl_erf_lowering(%arg0: memref<10x10xf32>) -> memref<10x10xf32> {
  %0 = alloc() : memref<10x10xf32>
  %1:2 = krnl.define_loops 2
  krnl.iterate(%1#0, %1#1) with (%1#0 -> %arg1 = 0 to 10, %1#1 -> %arg2 = 0 to 10) {
    %2 = krnl.load %arg0[%arg1, %arg2] : memref<10x10xf32>
    %3 = "krnl.erf"(%2) : (f32) -> f32
    krnl.store %3, %0[%arg1, %arg2] : memref<10x10xf32>
  }
  return %0 : memref<10x10xf32>
}

// CHECK-LABEL: test_krnl_erf_lowering
// CHECK: [[MEMREF_IN:%.+]] = llvm.insertvalue %arg6, {{.*}}[4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: [[DATA:%.+]] = llvm.extractvalue [[MEMREF_IN]][1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: [[DATA_IN:%.+]] = llvm.getelementptr [[DATA]]{{.*}} : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
// CHECK: [[SCALAR_IN:%.+]] = llvm.load [[DATA_IN]] : !llvm.ptr<float>
// CHECK: [[ERF_RES:%.+]] = llvm.call @erff([[SCALAR_IN]]) : (!llvm.float) -> !llvm.float
// CHECK: [[DATA_OUT:%.+]] = llvm.getelementptr {{.*}} : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
// CHECK: llvm.store [[ERF_RES]], [[DATA_OUT]] : !llvm.ptr<float>

