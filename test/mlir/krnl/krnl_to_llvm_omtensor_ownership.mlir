// RUN: onnx-mlir-opt -O3 --convert-krnl-to-llvm --canonicalize %s -split-input-file | FileCheck %s

module {
  // Check that output OMTensor owns the data pointer because data is not a constant.
  func.func @return_view_of_argument() -> memref<2x4xf32> {
    %0 = memref.alloc() : memref<2x4xf32>
    return %0 : memref<2x4xf32>
  }
  "krnl.entry_point"() {func = @return_view_of_argument, numInputs = 0 : i32, numOutputs = 1 : i32} : () -> ()
  // CHECK-LABEL: return_view_of_argument
  // CHECK: llvm.call @_mlir_ciface_return_view_of_argument
  // CHECK: llvm.call @omTensorCreateUntyped
  // CHECK: llvm.call @omTensorSetDataPtr({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> ()
}

// -----

module {
  // Check that output OMTensor does not own the data pointer because data is a block argument.
  func.func @return_argument(%arg0: memref<2x4xf32>) -> memref<2x4xf32> {
    return %arg0 : memref<2x4xf32>
  }
  "krnl.entry_point"() {func = @return_argument, numInputs = 1 : i32, numOutputs = 1 : i32} : () -> ()
  // CHECK-LABEL: return_argument
  // CHECK: llvm.call @_mlir_ciface_return_argument
  // CHECK: llvm.call @omTensorCreateUntyped
  // CHECK: llvm.call @omTensorSetDataPtr({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> ()
}
