// RUN: onnx-mlir-opt --convert-krnl-to-llvm="use-opaque-pointers=true store-constants-to-file constants-to-file-single-threshold=0.03 constants-to-file-total-threshold=0.00000006" --canonicalize %s -split-input-file | FileCheck %s && rm model.constants.bin 

// Thresholds for this files: 
//  -constants-to-file-single-threshold=0.03: 30 bytes for a single constants 
//  -constants-to-file-total-threshold=0.00000006: 60 bytes for all constants 

// Donot save to file if a constant is a return value.
module {
func.func @test_constants_to_file_return_value() -> memref<10xi64> {
  %0 = "krnl.global"() {name = "constant", alignment = 4096: i64, shape = [10], value = dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]> : tensor<10xi64>} : () -> memref<10xi64>
  return %0 : memref<10xi64>

  // CHECK-LABEL: module
  // CHECK: llvm.mlir.global internal constant @constant(dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]> : tensor<10xi64>) {addr_space = 0 : i32, alignment = 4096 : i64} : !llvm.array<10 x i64> 
}
}

// -----

// Donot save to file if a constant is splat.
module {
func.func @test_constants_to_file_splat() -> memref<10xi64> {
  %0 = "krnl.global"() {name = "constant", alignment = 4096: i64, shape = [10], value = dense<1> : tensor<10xi64>} : () -> memref<10xi64>
  return %0 : memref<10xi64>

  // CHECK-LABEL: module
  // CHECK: llvm.mlir.global internal constant @constant(dense<1> : tensor<10xi64>) {addr_space = 0 : i32, alignment = 4096 : i64} : !llvm.array<10 x i64>
}
}

// -----

module {
func.func @test_constants_to_file() -> memref<10xi64> {
  %0 = "krnl.global"() {name = "constant_0", alignment = 4096: i64, shape = [10], value = dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]> : tensor<10xi64>} : () -> memref<10xi64>
  %1 = "krnl.global"() {name = "constant_1", alignment = 4096: i64, shape = [10], value = dense<[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]> : tensor<10xi64>} : () -> memref<10xi64>
  %2 = "krnl.global"() {name = "constant_2", alignment = 4096: i64, shape = [10], value = dense<[21, 22, 23, 24, 25, 26, 27, 28, 29, 30]> : tensor<10xi64>} : () -> memref<10xi64>
  return %2 : memref<10xi64>

// CHECK-LABEL: module
// CHECK:         llvm.func @omFreeBuffer(!llvm.ptr)
// CHECK:         llvm.func @omCloseFile(!llvm.ptr)
// CHECK:         llvm.func @omOpenBinaryFile(!llvm.ptr, i64) -> !llvm.ptr
// CHECK:         llvm.func @omLoadExternalConstant(!llvm.ptr, !llvm.ptr, i64, i64, i64)

// CHECK:         llvm.mlir.global internal constant @constant_2(dense<[21, 22, 23, 24, 25, 26, 27, 28, 29, 30]> : tensor<10xi64>) {addr_space = 0 : i32, alignment = 4096 : i64} : !llvm.array<10 x i64>

// CHECK:         llvm.mlir.global internal @om_external_constant_data_constant_1() {addr_space = 0 : i32, alignment = 4096 : i64} : !llvm.ptr {
// CHECK:           [[VAR_0_4_:%.+]] = llvm.mlir.null : !llvm.ptr
// CHECK:           llvm.return [[VAR_0_4_]] : !llvm.ptr
// CHECK:         }
// CHECK:         llvm.mlir.global internal constant @om_external_constant_offset_constant_1(80 : i64) {addr_space = 0 : i32} : i64
// CHECK:         llvm.mlir.global internal constant @om_external_constant_size_constant_1(80 : i64) {addr_space = 0 : i32} : i64
// CHECK:         llvm.mlir.global internal @om_external_constant_data_constant_0() {addr_space = 0 : i32, alignment = 4096 : i64} : !llvm.ptr {
// CHECK:           [[VAR_0_5_:%.+]] = llvm.mlir.null : !llvm.ptr
// CHECK:           llvm.return [[VAR_0_5_]] : !llvm.ptr
// CHECK:         }
// CHECK:         llvm.mlir.global internal constant @om_external_constant_offset_constant_0(0 : i64) {addr_space = 0 : i32} : i64
// CHECK:         llvm.mlir.global internal constant @om_external_constant_size_constant_0(80 : i64) {addr_space = 0 : i32} : i64
// CHECK:         llvm.mlir.global internal constant @om_external_constant_filename("model.constants.bin\00") {addr_space = 0 : i32}
// CHECK:         llvm.mlir.global internal constant @om_external_constant_isLE(0 : i8) {addr_space = 0 : i32} : i8

// CHECK:         llvm.func @omLoadConstantsFromFile([[arg0_:%.+]]: !llvm.ptr) {
// CHECK-DAG:       [[VAR_0_8_:%.+]] = llvm.mlir.constant(4096 : i64) : i64
// CHECK-DAG:       [[VAR_1_3_:%.+]] = llvm.mlir.constant(80 : i64) : i64
// CHECK-DAG:       [[VAR_2_3_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_3_:%.+]] = llvm.call @omOpenBinaryFile([[arg0_]], [[VAR_2_3_]]) : (!llvm.ptr, i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_4_3_:%.+]] = llvm.mlir.addressof @om_external_constant_data_constant_1 : !llvm.ptr<ptr>
// CHECK:           [[VAR_5_3_:%.+]] = llvm.bitcast [[VAR_4_3_]] : !llvm.ptr<ptr> to !llvm.ptr
// CHECK:           llvm.call @omLoadExternalConstant([[VAR_5_3_]], [[VAR_3_3_]], [[VAR_1_3_]], [[VAR_1_3_]], [[VAR_0_8_]]) : (!llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
// CHECK:           [[VAR_6_3_:%.+]] = llvm.mlir.addressof @om_external_constant_data_constant_0 : !llvm.ptr<ptr>
// CHECK:           [[VAR_7_3_:%.+]] = llvm.bitcast [[VAR_6_3_]] : !llvm.ptr<ptr> to !llvm.ptr
// CHECK:           llvm.call @omLoadExternalConstant([[VAR_7_3_]], [[VAR_3_3_]], [[VAR_2_3_]], [[VAR_1_3_]], [[VAR_0_8_]]) : (!llvm.ptr, !llvm.ptr, i64, i64, i64) -> ()
// CHECK:           llvm.call @omCloseFile([[VAR_3_3_]]) : (!llvm.ptr) -> ()
// CHECK:           llvm.return
// CHECK:         }

// CHECK:         llvm.func @omFreeBuffersForConstants() {
// CHECK:           [[VAR_1_4_:%.+]] = llvm.mlir.addressof @om_external_constant_data_constant_1 : !llvm.ptr<ptr>
// CHECK:           [[VAR_2_4_:%.+]] = llvm.load [[VAR_1_4_]] : !llvm.ptr<ptr>
// CHECK:           llvm.call @omFreeBuffer([[VAR_2_4_]]) : (!llvm.ptr) -> ()
// CHECK:           [[VAR_3_4_:%.+]] = llvm.mlir.addressof @om_external_constant_data_constant_0 : !llvm.ptr<ptr>
// CHECK:           [[VAR_4_4_:%.+]] = llvm.load [[VAR_3_4_]] : !llvm.ptr<ptr>
// CHECK:           llvm.call @omFreeBuffer([[VAR_4_4_]]) : (!llvm.ptr) -> ()
// CHECK:           llvm.return
// CHECK:         }
}
}
