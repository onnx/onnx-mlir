// RUN: onnx-mlir-opt -O3 --convert-krnl-to-affine --convert-krnl-to-llvm %s -split-input-file | FileCheck %s

// -----

/// Test lowering of krnl.erf to LLVM math function call.
func.func @test_krnl_erf_lowering(%arg0: memref<10x10xf32>) -> memref<10x10xf32> {
  %0 = memref.alloc() : memref<10x10xf32>
  %1:2 = krnl.define_loops 2
  krnl.iterate(%1#0, %1#1) with (%1#0 -> %arg1 = 0 to 10, %1#1 -> %arg2 = 0 to 10) {
    %2 = krnl.load %arg0[%arg1, %arg2] : memref<10x10xf32>
    %3 = "krnl.erf"(%2) : (f32) -> f32
    krnl.store %3, %0[%arg1, %arg2] : memref<10x10xf32>
  }
  return %0 : memref<10x10xf32>
}

// CHECK-LABEL: test_krnl_erf_lowering
// CHECK: [[MEMREF_IN:%.+]] = llvm.insertvalue %arg6, {{.*}}[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: [[DATA:%.+]] = llvm.extractvalue [[MEMREF_IN]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: [[DATA_IN:%.+]] = llvm.getelementptr [[DATA]]{{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: [[SCALAR_IN:%.+]] = llvm.load [[DATA_IN]] : !llvm.ptr
// CHECK: [[ERF_RES:%.+]] = llvm.call @erff([[SCALAR_IN]]) : (f32) -> f32
// CHECK: [[DATA_OUT:%.+]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: llvm.store [[ERF_RES]], [[DATA_OUT]] : f32, !llvm.ptr

// -----

/// Test lowering of krnl.acos to LLVM math function call.
func.func @test_krnl_acos_lowering(%arg0: memref<10x10xf32>) -> memref<10x10xf32> {
  %0 = memref.alloc() : memref<10x10xf32>
  %1:2 = krnl.define_loops 2
  krnl.iterate(%1#0, %1#1) with (%1#0 -> %arg1 = 0 to 10, %1#1 -> %arg2 = 0 to 10) {
    %2 = krnl.load %arg0[%arg1, %arg2] : memref<10x10xf32>
    %3 = "krnl.acos"(%2) : (f32) -> f32
    krnl.store %3, %0[%arg1, %arg2] : memref<10x10xf32>
  }
  return %0 : memref<10x10xf32>
}

// CHECK-LABEL: test_krnl_acos_lowering
// CHECK: [[MEMREF_IN:%.+]] = llvm.insertvalue %arg6, {{.*}}[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: [[DATA:%.+]] = llvm.extractvalue [[MEMREF_IN]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: [[DATA_IN:%.+]] = llvm.getelementptr [[DATA]]{{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: [[SCALAR_IN:%.+]] = llvm.load [[DATA_IN]] : !llvm.ptr
// CHECK: [[ACOS_RES:%.+]] = llvm.call @acosf([[SCALAR_IN]]) : (f32) -> f32
// CHECK: [[DATA_OUT:%.+]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: llvm.store [[ACOS_RES]], [[DATA_OUT]] : f32, !llvm.ptr

// -----

/// Test lowering of krnl.acosh to LLVM math function call.
func.func @test_krnl_acosh_lowering(%arg0: memref<10x10xf32>) -> memref<10x10xf32> {
  %0 = memref.alloc() : memref<10x10xf32>
  %1:2 = krnl.define_loops 2
  krnl.iterate(%1#0, %1#1) with (%1#0 -> %arg1 = 0 to 10, %1#1 -> %arg2 = 0 to 10) {
    %2 = krnl.load %arg0[%arg1, %arg2] : memref<10x10xf32>
    %3 = "krnl.acosh"(%2) : (f32) -> f32
    krnl.store %3, %0[%arg1, %arg2] : memref<10x10xf32>
  }
  return %0 : memref<10x10xf32>
}

// CHECK-LABEL: test_krnl_acosh_lowering
// CHECK: [[MEMREF_IN:%.+]] = llvm.insertvalue %arg6, {{.*}}[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: [[DATA:%.+]] = llvm.extractvalue [[MEMREF_IN]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: [[DATA_IN:%.+]] = llvm.getelementptr [[DATA]]{{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: [[SCALAR_IN:%.+]] = llvm.load [[DATA_IN]] : !llvm.ptr
// CHECK: [[ACOS_RES:%.+]] = llvm.call @acoshf([[SCALAR_IN]]) : (f32) -> f32
// CHECK: [[DATA_OUT:%.+]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: llvm.store [[ACOS_RES]], [[DATA_OUT]] : f32, !llvm.ptr

// -----

/// Test lowering of krnl.asin to LLVM math function call.
func.func @test_krnl_asin_lowering(%arg0: memref<10x10xf32>) -> memref<10x10xf32> {
  %0 = memref.alloc() : memref<10x10xf32>
  %1:2 = krnl.define_loops 2
  krnl.iterate(%1#0, %1#1) with (%1#0 -> %arg1 = 0 to 10, %1#1 -> %arg2 = 0 to 10) {
    %2 = krnl.load %arg0[%arg1, %arg2] : memref<10x10xf32>
    %3 = "krnl.asin"(%2) : (f32) -> f32
    krnl.store %3, %0[%arg1, %arg2] : memref<10x10xf32>
  }
  return %0 : memref<10x10xf32>
}

// CHECK-LABEL: test_krnl_asin_lowering
// CHECK: [[MEMREF_IN:%.+]] = llvm.insertvalue %arg6, {{.*}}[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: [[DATA:%.+]] = llvm.extractvalue [[MEMREF_IN]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: [[DATA_IN:%.+]] = llvm.getelementptr [[DATA]]{{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: [[SCALAR_IN:%.+]] = llvm.load [[DATA_IN]] : !llvm.ptr
// CHECK: [[ACOS_RES:%.+]] = llvm.call @asinf([[SCALAR_IN]]) : (f32) -> f32
// CHECK: [[DATA_OUT:%.+]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: llvm.store [[ACOS_RES]], [[DATA_OUT]] : f32, !llvm.ptr

// -----

/// Test lowering of krnl.asinh to LLVM math function call.
func.func @test_krnl_asinh_lowering(%arg0: memref<10x10xf32>) -> memref<10x10xf32> {
  %0 = memref.alloc() : memref<10x10xf32>
  %1:2 = krnl.define_loops 2
  krnl.iterate(%1#0, %1#1) with (%1#0 -> %arg1 = 0 to 10, %1#1 -> %arg2 = 0 to 10) {
    %2 = krnl.load %arg0[%arg1, %arg2] : memref<10x10xf32>
    %3 = "krnl.asinh"(%2) : (f32) -> f32
    krnl.store %3, %0[%arg1, %arg2] : memref<10x10xf32>
  }
  return %0 : memref<10x10xf32>
}

// CHECK-LABEL: test_krnl_asinh_lowering
// CHECK: [[MEMREF_IN:%.+]] = llvm.insertvalue %arg6, {{.*}}[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: [[DATA:%.+]] = llvm.extractvalue [[MEMREF_IN]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: [[DATA_IN:%.+]] = llvm.getelementptr [[DATA]]{{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: [[SCALAR_IN:%.+]] = llvm.load [[DATA_IN]] : !llvm.ptr
// CHECK: [[ACOS_RES:%.+]] = llvm.call @asinhf([[SCALAR_IN]]) : (f32) -> f32
// CHECK: [[DATA_OUT:%.+]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: llvm.store [[ACOS_RES]], [[DATA_OUT]] : f32, !llvm.ptr

// -----

/// Test lowering of krnl.atan to LLVM math function call.
func.func @test_krnl_atan_lowering(%arg0: memref<10x10xf32>) -> memref<10x10xf32> {
  %0 = memref.alloc() : memref<10x10xf32>
  %1:2 = krnl.define_loops 2
  krnl.iterate(%1#0, %1#1) with (%1#0 -> %arg1 = 0 to 10, %1#1 -> %arg2 = 0 to 10) {
    %2 = krnl.load %arg0[%arg1, %arg2] : memref<10x10xf32>
    %3 = "krnl.atan"(%2) : (f32) -> f32
    krnl.store %3, %0[%arg1, %arg2] : memref<10x10xf32>
  }
  return %0 : memref<10x10xf32>
}

// CHECK-LABEL: test_krnl_atan_lowering
// CHECK: [[MEMREF_IN:%.+]] = llvm.insertvalue %arg6, {{.*}}[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: [[DATA:%.+]] = llvm.extractvalue [[MEMREF_IN]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: [[DATA_IN:%.+]] = llvm.getelementptr [[DATA]]{{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: [[SCALAR_IN:%.+]] = llvm.load [[DATA_IN]] : !llvm.ptr
// CHECK: [[ACOS_RES:%.+]] = llvm.call @atanf([[SCALAR_IN]]) : (f32) -> f32
// CHECK: [[DATA_OUT:%.+]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: llvm.store [[ACOS_RES]], [[DATA_OUT]] : f32, !llvm.ptr

// -----

func.func @test_krnl_atanh_lowering(%arg0: memref<10x10xf32>) -> memref<10x10xf32> {
  %0 = memref.alloc() : memref<10x10xf32>
  %1:2 = krnl.define_loops 2
  krnl.iterate(%1#0, %1#1) with (%1#0 -> %arg1 = 0 to 10, %1#1 -> %arg2 = 0 to 10) {
    %2 = krnl.load %arg0[%arg1, %arg2] : memref<10x10xf32>
    %3 = "krnl.atanh"(%2) : (f32) -> f32
    krnl.store %3, %0[%arg1, %arg2] : memref<10x10xf32>
  }
  return %0 : memref<10x10xf32>
}

// CHECK-LABEL: test_krnl_atanh_lowering
// CHECK: [[MEMREF_IN:%.+]] = llvm.insertvalue %arg6, {{.*}}[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: [[DATA:%.+]] = llvm.extractvalue [[MEMREF_IN]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: [[DATA_IN:%.+]] = llvm.getelementptr [[DATA]]{{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: [[SCALAR_IN:%.+]] = llvm.load [[DATA_IN]] : !llvm.ptr
// CHECK: [[ACOS_RES:%.+]] = llvm.call @atanhf([[SCALAR_IN]]) : (f32) -> f32
// CHECK: [[DATA_OUT:%.+]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: llvm.store [[ACOS_RES]], [[DATA_OUT]] : f32, !llvm.ptr

// -----

func.func @test_krnl_tan_lowering(%arg0: memref<10x10xf32>) -> memref<10x10xf32> {
  %0 = memref.alloc() : memref<10x10xf32>
  %1:2 = krnl.define_loops 2
  krnl.iterate(%1#0, %1#1) with (%1#0 -> %arg1 = 0 to 10, %1#1 -> %arg2 = 0 to 10) {
    %2 = krnl.load %arg0[%arg1, %arg2] : memref<10x10xf32>
    %3 = "krnl.tan"(%2) : (f32) -> f32
    krnl.store %3, %0[%arg1, %arg2] : memref<10x10xf32>
  }
  return %0 : memref<10x10xf32>
}

// CHECK-LABEL: test_krnl_tan_lowering
// CHECK: [[MEMREF_IN:%.+]] = llvm.insertvalue %arg6, {{.*}}[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: [[DATA:%.+]] = llvm.extractvalue [[MEMREF_IN]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: [[DATA_IN:%.+]] = llvm.getelementptr [[DATA]]{{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: [[SCALAR_IN:%.+]] = llvm.load [[DATA_IN]] : !llvm.ptr
// CHECK: [[ACOS_RES:%.+]] = llvm.call @tanf([[SCALAR_IN]]) : (f32) -> f32
// CHECK: [[DATA_OUT:%.+]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: llvm.store [[ACOS_RES]], [[DATA_OUT]] : f32, !llvm.ptr
