// RUN: onnx-mlir-opt --convert-krnl-to-llvm="store-constants-to-file constants-to-file-single-threshold=0.03 constants-to-file-total-threshold=0.00000006" --canonicalize %s -split-input-file | FileCheck %s && rm -f model.constants.bin

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

// CHECK:         llvm.func @omGetExternalConstantAddr(!llvm.ptr, !llvm.ptr, i64)
// CHECK:         llvm.func @omMMapBinaryFile(!llvm.ptr, !llvm.ptr, i64, i64) -> i1
// CHECK:         llvm.mlir.global internal constant @constant_2(dense<[21, 22, 23, 24, 25, 26, 27, 28, 29, 30]> : tensor<10xi64>) {addr_space = 0 : i32, alignment = 4096 : i64} : !llvm.array<10 x i64>
// CHECK:         llvm.mlir.global internal @om_external_constant_data_constant_1() {addr_space = 0 : i32, alignment = 4096 : i64} : !llvm.ptr {
// CHECK:           [[VAR_0_4_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           llvm.return [[VAR_0_4_]] : !llvm.ptr
// CHECK:         }
// CHECK:         llvm.mlir.global internal constant @om_external_constant_offset_constant_1(0 : i64) {addr_space = 0 : i32} : i64
// CHECK:         llvm.mlir.global internal @om_external_constant_data_constant_0() {addr_space = 0 : i32, alignment = 4096 : i64} : !llvm.ptr {
// CHECK:           [[VAR_0_5_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           llvm.return [[VAR_0_5_]] : !llvm.ptr
// CHECK:         }
// CHECK:         llvm.mlir.global internal constant @om_external_constant_offset_constant_0(4096 : i64) {addr_space = 0 : i32} : i64
// CHECK:         llvm.mlir.global internal constant @om_external_constant_filename("model.constants.bin\00") {addr_space = 0 : i32}
// CHECK:         llvm.mlir.global internal constant @om_external_constant_filesize(4176 : i64) {addr_space = 0 : i32} : i64
// CHECK:         llvm.mlir.global internal constant @om_external_constant_isLE(1 : i8) {addr_space = 0 : i32} : i8
// CHECK:         llvm.mlir.global internal @om_external_constant_packedConst() {addr_space = 0 : i32} : !llvm.ptr {
// CHECK:           [[VAR_0_6_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           llvm.return [[VAR_0_6_]] : !llvm.ptr
// CHECK:         }

// CHECK:         llvm.func @omLoadConstantsFromFile() -> i1 {
// CHECK-DAG:       [[VAR_0_9_:%.+]] = llvm.mlir.constant(4096 : i64) : i64
// CHECK-DAG:       [[VAR_1_3_:%.+]] = llvm.mlir.addressof @om_external_constant_data_constant_0 : !llvm.ptr
// CHECK-DAG:       [[VAR_1_4_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_2_3_:%.+]] = llvm.mlir.addressof @om_external_constant_data_constant_1 : !llvm.ptr
// CHECK-DAG:       [[VAR_3_3_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_4_3_:%.+]] = llvm.mlir.constant(4176 : i64) : i64
// CHECK-DAG:       [[VAR_5_3_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_6_3_:%.+]] = llvm.mlir.addressof @om_external_constant_packedConst : !llvm.ptr
// CHECK-DAG:       [[VAR_7_3_:%.+]] = llvm.mlir.addressof @om_external_constant_filename : !llvm.ptr
// CHECK:           [[VAR_8_3_:%.+]] = llvm.call @omMMapBinaryFile([[VAR_6_3_]], [[VAR_7_3_]], [[VAR_4_3_]], [[VAR_5_3_]]) : (!llvm.ptr, !llvm.ptr, i64, i64) -> i1
// CHECK:           [[VAR_9_3_:%.+]] = llvm.icmp "ne" [[VAR_3_3_]], [[VAR_8_3_]] : i1
// CHECK:           llvm.cond_br [[VAR_9_3_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           llvm.return [[VAR_8_3_]] : i1
// CHECK:         ^bb2:  // pred: ^bb0
// CHECK:           llvm.call @omGetExternalConstantAddr([[VAR_2_3_]], [[VAR_6_3_]], [[VAR_1_4_]]) : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           llvm.call @omGetExternalConstantAddr([[VAR_1_3_]], [[VAR_6_3_]], [[VAR_0_9_]]) : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           llvm.return [[VAR_3_3_]] : i1
// CHECK:         }

// CHECK:         llvm.func @run_main_graph({{.*}}: !llvm.ptr) -> !llvm.ptr {
// CHECK-DAG:       [[VAR_3_4_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-DAG:       [[VAR_4_4_:%.+]] = llvm.mlir.constant(22 : i32) : i32
// CHECK-DAG:       [[VAR_5_4_:%.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG:       [[VAR_6_4_:%.+]] = llvm.call @omLoadConstantsFromFile() : () -> i1
// CHECK:           [[VAR_7_4_:%.+]] = llvm.icmp "ne" [[VAR_5_4_]], [[VAR_6_4_]] : i1
// CHECK:           llvm.cond_br [[VAR_7_4_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           [[VAR_8_4_:%.+]] = llvm.call @__errno_location() : () -> !llvm.ptr
// CHECK:           llvm.store [[VAR_4_4_]], [[VAR_8_4_]] : i32, !llvm.ptr
// CHECK:           llvm.return [[VAR_3_4_]] : !llvm.ptr
// CHECK:         ^bb2:  // pred: ^bb0
// CHECK:         }

}
"krnl.entry_point"() {func = @test_constants_to_file, numInputs = 0 : i32, numOutputs = 1 : i32, signature = "[in_sig]\00@[out_sig]\00"} : () -> ()
}
