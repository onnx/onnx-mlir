// RUN: onnx-mlir-opt -O3 --convert-krnl-to-llvm --canonicalize %s -split-input-file | FileCheck %s

module {
// Check that output OMTensor does not own the data pointer because data is a constant.
  func.func @return_constant() -> memref<8xf32> {
    %0 = "krnl.global"() {name = "cst0", shape = [8], value = dense<[1., 2., 3., 4., 5., 6., 7., 8.]> : tensor<8xf32>} : () -> memref<8xf32>
    return %0 : memref<8xf32>
  }
  "krnl.entry_point"() {func = @return_constant, numInputs = 0 : i32, numOutputs = 1 : i32, signature = ""} : () -> ()

  // CHECK-LABEL: return_constant
  // CHECK: [[OWNING:%.+]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK: llvm.call @_mlir_ciface_return_constant
  // CHECK: llvm.call @omTensorSetDataPtr({{.*}}, [[OWNING]], {{.*}}, {{.*}}) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> ()
}

// -----

module {
 // Check that output OMTensor does not own the data pointer because data is a constant via a view op.
  func.func @return_view_of_constant() -> memref<2x4xf32> {
    %0 = "krnl.global"() {name = "cst0", shape = [8], value = dense<[1., 2., 3., 4., 5., 6., 7., 8.]> : tensor<8xf32>} : () -> memref<8xf32>
    %1 = memref.reinterpret_cast %0 to offset: [0], sizes: [2, 4], strides: [4, 1] : memref<8xf32> to memref<2x4xf32>
    return %1 : memref<2x4xf32>
  }
  "krnl.entry_point"() {func = @return_view_of_constant, numInputs = 0 : i32, numOutputs = 1 : i32, signature = ""} : () -> ()
  // CHECK-LABEL: return_view_of_constant
  // CHECK: [[OWNING:%.+]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK: llvm.call @_mlir_ciface_return_view_of_constant
  // CHECK: llvm.call @omTensorSetDataPtr({{.*}}, [[OWNING]], {{.*}}, {{.*}}) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> ()
}

// -----

module {
  // Check that output OMTensor owns the data pointer because data is not a constant.
  func.func @return_view_of_argument() -> memref<2x4xf32> {
    %0 = memref.alloc() : memref<2x4xf32>
    return %0 : memref<2x4xf32>
  }
  "krnl.entry_point"() {func = @return_view_of_argument, numInputs = 0 : i32, numOutputs = 1 : i32, signature = ""} : () -> ()
  // CHECK-LABEL: return_view_of_argument
  // CHECK: [[OWNING:%.+]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: llvm.call @_mlir_ciface_return_view_of_argument
  // CHECK: llvm.call @omTensorCreateUntyped
  // CHECK: llvm.call @omTensorSetDataPtr({{.*}}, [[OWNING]], {{.*}}, {{.*}}) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> ()
}

// -----

module {
  // Check that output OMTensor does not own the data pointer because data is a block argument.
  func.func @return_argument(%arg0: memref<2x4xf32>) -> memref<2x4xf32> {
    return %arg0 : memref<2x4xf32>
  }
  "krnl.entry_point"() {func = @return_argument, numInputs = 1 : i32, numOutputs = 1 : i32, signature = ""} : () -> ()
  // CHECK-LABEL: return_argument
  // CHECK: [[OWNING:%.+]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK: llvm.call @_mlir_ciface_return_argument
  // CHECK: llvm.call @omTensorCreateUntyped
  // CHECK: llvm.call @omTensorSetDataPtr({{.*}}, [[OWNING]], {{.*}}, {{.*}}) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> ()
}
