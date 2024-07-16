// RUN: onnx-mlir-opt --convert-krnl-to-llvm --canonicalize %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --convert-krnl-to-llvm="store-constants-to-file constants-to-file-single-threshold=0.03 constants-to-file-total-threshold=0.00000006" --canonicalize %s -split-input-file | FileCheck %s -check-prefix=CHECK-CONST-TO-FILE && rm model.constants.bin

// -----

module attributes {"onnx-mlir.symbol-postfix" = "tag_symbols"} {
  func.func private @main_graph(%arg0: memref<10xf32>) -> memref<10xf32> {
    %0 = "krnl.global"() {name = "constant", alignment = 1024 : i64, shape = [3], value = dense<[0.0, 0.1, 0.2]> : tensor<3xf32>} : () -> memref<3xf32>
    return %arg0 : memref<10xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[in_sig]\00@[out_sig]\00"} : () -> ()

// CHECK-DAG:     llvm.mlir.global external constant @_entry_point_1_tag_symbols("run_main_graph_tag_symbols\00") {addr_space = 0 : i32}
// CHECK-DAG:     llvm.mlir.global external constant @_entry_point_1_in_sig_tag_symbols("[in_sig]\00") {addr_space = 0 : i32}
// CHECK-DAG:     llvm.mlir.global external constant @_entry_point_1_out_sig_tag_symbols("[out_sig]\00") {addr_space = 0 : i32}
// CHECK-DAG:     llvm.mlir.global external constant @_entry_point_0_tag_symbols("run_main_graph\00") {addr_space = 0 : i32}
// CHECK-DAG:     llvm.mlir.global external constant @_entry_point_0_in_sig_tag_symbols("[in_sig]\00") {addr_space = 0 : i32}
// CHECK-DAG:     llvm.mlir.global external constant @_entry_point_0_out_sig_tag_symbols("[out_sig]\00") {addr_space = 0 : i32}
// CHECK:         llvm.mlir.global internal constant @constant_tag_symbols(dense<[0.000000e+00, 1.000000e-01, 2.000000e-01]> : tensor<3xf32>) {addr_space = 0 : i32, alignment = 1024 : i64} : !llvm.array<3 x f32>

// CHECK:         llvm.func @main_graph_tag_symbols{{.*}}

// CHECK:         llvm.func @_mlir_ciface_main_graph_tag_symbols{{.*}} {
// CHECK:           {{.*}} = llvm.call @main_graph_tag_symbols
// CHECK:         }

// CHECK:         llvm.func @run_main_graph_tag_symbols{{.*}} {
// CHECK:           llvm.call @_mlir_ciface_main_graph_tag_symbols
// CHECK:         }

// CHECK:         llvm.func @run_main_graph{{.*}} {
// CHECK:           [[VAR_0_2_:%.+]] = llvm.call @run_main_graph_tag_symbols
// CHECK:           llvm.return [[VAR_0_2_]] : !llvm.ptr
// CHECK:         }

// CHECK:         llvm.mlir.global internal constant @_entry_point_arrays_tag_symbols() {addr_space = 0 : i32} : !llvm.array<3 x ptr> {
// CHECK-DAG:       [[VAR_0_4_:%.+]] = llvm.mlir.undef : !llvm.array<3 x ptr>
// CHECK-DAG:       [[VAR_1_3_:%.+]] = llvm.mlir.addressof @_entry_point_0_tag_symbols : !llvm.ptr
// CHECK-DAG:       [[VAR_3_3_:%.+]] = llvm.insertvalue [[VAR_1_3_]], [[VAR_0_4_]][0] : !llvm.array<3 x ptr>
// CHECK-DAG:       [[VAR_4_2_:%.+]] = llvm.mlir.addressof @_entry_point_1_tag_symbols : !llvm.ptr
// CHECK-DAG:       [[VAR_6_2_:%.+]] = llvm.insertvalue [[VAR_4_2_]], [[VAR_3_3_]][1] : !llvm.array<3 x ptr>
// CHECK-DAG:       [[VAR_7_1_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_8_1_:%.+]] = llvm.insertvalue [[VAR_7_1_]], [[VAR_6_2_]][2] : !llvm.array<3 x ptr>
// CHECK:           llvm.return [[VAR_8_1_]] : !llvm.array<3 x ptr>
// CHECK:         }

// CHECK:         llvm.func @omQueryEntryPoints_tag_symbols([[arg0_:%.+]]: !llvm.ptr) -> !llvm.ptr {
// CHECK-DAG:       [[VAR_0_5_:%.+]] = llvm.mlir.addressof @_entry_point_arrays_tag_symbols : !llvm.ptr
// CHECK-DAG:       [[VAR_1_4_:%.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-DAG:       [[VAR_2_4_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_3_4_:%.+]] = llvm.icmp "ne" [[arg0_]], [[VAR_2_4_]] : !llvm.ptr
// CHECK:           llvm.cond_br [[VAR_3_4_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           llvm.store [[VAR_1_4_]], [[arg0_]] : i64, !llvm.ptr
// CHECK:           llvm.br ^bb2
// CHECK:         ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK:           llvm.return [[VAR_0_5_]] : !llvm.ptr
// CHECK:         }
// CHECK:         llvm.func @omQueryEntryPoints([[arg0_:%.+]]: !llvm.ptr) -> !llvm.ptr {
// CHECK:           [[VAR_0_5_:%.+]] = llvm.call @omQueryEntryPoints_tag_symbols([[arg0_]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK:           llvm.return [[VAR_0_5_]] : !llvm.ptr
// CHECK:         }

// CHECK:         llvm.func @omInputSignature_tag_symbols([[arg0_:%.+]]: !llvm.ptr) -> !llvm.ptr {
// CHECK-DAG:       [[VAR_0:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-DAG:       [[VAR_0_7_:%.+]] = llvm.mlir.addressof @_entry_point_1_in_sig_tag_symbols : !llvm.ptr
// CHECK-DAG:       [[VAR_1_5_:%.+]] = llvm.mlir.constant(27 : i64) : i64
// CHECK-DAG:       [[VAR_2_5_:%.+]] = llvm.mlir.addressof @_entry_point_1_tag_symbols : !llvm.ptr
// CHECK-DAG:       [[VAR_3_5_:%.+]] = llvm.mlir.addressof @_entry_point_0_in_sig_tag_symbols : !llvm.ptr
// CHECK-DAG:       [[VAR_4_3_:%.+]] = llvm.mlir.constant(15 : i64) : i64
// CHECK-DAG:       [[VAR_5_4_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:       [[VAR_6_3_:%.+]] = llvm.mlir.addressof @_entry_point_0_tag_symbols : !llvm.ptr
// CHECK:           [[VAR_7_1_:%.+]] = llvm.call @strncmp([[arg0_]], [[VAR_6_3_]], [[VAR_4_3_]]) : (!llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK:           [[VAR_8_1_:%.+]] = llvm.icmp "eq" [[VAR_7_1_]], [[VAR_5_4_]] : i32
// CHECK:           llvm.cond_br [[VAR_8_1_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           llvm.return [[VAR_3_5_]] : !llvm.ptr
// CHECK:         ^bb2:  // pred: ^bb0
// CHECK:           [[VAR_10_1_:%.+]] = llvm.call @strncmp([[arg0_]], [[VAR_2_5_]], [[VAR_1_5_]]) : (!llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK:           [[VAR_11_1_:%.+]] = llvm.icmp "eq" [[VAR_10_1_]], [[VAR_5_4_]] : i32
// CHECK:           llvm.cond_br [[VAR_11_1_]], ^bb3, ^bb4
// CHECK:         ^bb3:  // pred: ^bb2
// CHECK:           llvm.return [[VAR_0_7_]] : !llvm.ptr
// CHECK:         ^bb4:  // pred: ^bb2
// CHECK:           llvm.return [[VAR_0]] : !llvm.ptr
// CHECK:         }

// CHECK:         llvm.func @omInputSignature([[arg0_:%.+]]: !llvm.ptr) -> !llvm.ptr {
// CHECK:           [[VAR_0_7_:%.+]] = llvm.call @omInputSignature_tag_symbols([[arg0_]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK:           llvm.return [[VAR_0_7_]] : !llvm.ptr
// CHECK:         }

// CHECK:         llvm.func @omOutputSignature_tag_symbols([[arg0_]]: !llvm.ptr) -> !llvm.ptr {
// CHECK-DAG:       [[VAR_0:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-DAG:       [[VAR_0_9_:%.+]] = llvm.mlir.addressof @_entry_point_1_out_sig_tag_symbols : !llvm.ptr
// CHECK-DAG:       [[VAR_1_6_:%.+]] = llvm.mlir.constant(27 : i64) : i64
// CHECK-DAG:       [[VAR_2_6_:%.+]] = llvm.mlir.addressof @_entry_point_1_tag_symbols : !llvm.ptr
// CHECK-DAG:       [[VAR_3_6_:%.+]] = llvm.mlir.addressof @_entry_point_0_out_sig_tag_symbols : !llvm.ptr
// CHECK-DAG:       [[VAR_4_4_:%.+]] = llvm.mlir.constant(15 : i64) : i64
// CHECK-DAG:       [[VAR_5_5_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:       [[VAR_6_4_:%.+]] = llvm.mlir.addressof @_entry_point_0_tag_symbols : !llvm.ptr
// CHECK:           [[VAR_7_2_:%.+]] = llvm.call @strncmp([[arg0_]], [[VAR_6_4_]], [[VAR_4_4_]]) : (!llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK:           [[VAR_8_2_:%.+]] = llvm.icmp "eq" [[VAR_7_2_]], [[VAR_5_5_]] : i32
// CHECK:           llvm.cond_br [[VAR_8_2_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           llvm.return [[VAR_3_6_]] : !llvm.ptr
// CHECK:         ^bb2:  // pred: ^bb0
// CHECK:           [[VAR_10_2_:%.+]] = llvm.call @strncmp([[arg0_]], [[VAR_2_6_]], [[VAR_1_6_]]) : (!llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK:           [[VAR_11_2_:%.+]] = llvm.icmp "eq" [[VAR_10_2_]], [[VAR_5_5_]] : i32
// CHECK:           llvm.cond_br [[VAR_11_2_]], ^bb3, ^bb4
// CHECK:         ^bb3:  // pred: ^bb2
// CHECK:           llvm.return [[VAR_0_9_]] : !llvm.ptr
// CHECK:         ^bb4:  // pred: ^bb2
// CHECK:           llvm.return [[VAR_0]] : !llvm.ptr
// CHECK:         }
// CHECK:         llvm.func @omOutputSignature([[arg0_:%.+]]: !llvm.ptr) -> !llvm.ptr {
// CHECK:           [[VAR_0_9_:%.+]] = llvm.call @omOutputSignature_tag_symbols([[arg0_]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK:           llvm.return [[VAR_0_9_]] : !llvm.ptr
// CHECK:         }
}

// -----

module attributes {"onnx-mlir.symbol-postfix" = "tag_constants_to_file"} {
  func.func @main_graph() -> memref<10xi64> {
    %0 = "krnl.global"() {name = "constant_0", alignment = 4096: i64, shape = [10], value = dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]> : tensor<10xi64>} : () -> memref<10xi64>
    %1 = "krnl.global"() {name = "constant_1", alignment = 4096: i64, shape = [10], value = dense<[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]> : tensor<10xi64>} : () -> memref<10xi64>
    %2 = "krnl.global"() {name = "constant_2", alignment = 4096: i64, shape = [10], value = dense<[21, 22, 23, 24, 25, 26, 27, 28, 29, 30]> : tensor<10xi64>} : () -> memref<10xi64>
    return %2 : memref<10xi64>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 0 : i32, numOutputs = 1 : i32, signature = "[in_sig]\00@[out_sig]\00"} : () -> ()

// CHECK-CONST-TO-FILE:         llvm.mlir.global internal constant @constant_2_tag_constants_to_file(dense<[21, 22, 23, 24, 25, 26, 27, 28, 29, 30]> : tensor<10xi64>) {addr_space = 0 : i32, alignment = 4096 : i64} : !llvm.array<10 x i64>

// CHECK-CONST-TO-FILE:         llvm.mlir.global internal @om_external_constant_data_constant_1_tag_constants_to_file() {addr_space = 0 : i32, alignment = 4096 : i64} : !llvm.ptr {
// CHECK-CONST-TO-FILE:           [[VAR_0_13_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-CONST-TO-FILE:           llvm.return [[VAR_0_13_]] : !llvm.ptr
// CHECK-CONST-TO-FILE:         }
// CHECK-CONST-TO-FILE:         llvm.mlir.global internal constant @om_external_constant_offset_constant_1_tag_constants_to_file(0 : i64) {addr_space = 0 : i32} : i64

// CHECK-CONST-TO-FILE:         llvm.mlir.global internal @om_external_constant_data_constant_0_tag_constants_to_file() {addr_space = 0 : i32, alignment = 4096 : i64} : !llvm.ptr {
// CHECK-CONST-TO-FILE:           [[VAR_0_14_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-CONST-TO-FILE:           llvm.return [[VAR_0_14_]] : !llvm.ptr
// CHECK-CONST-TO-FILE:         }
// CHECK-CONST-TO-FILE:         llvm.mlir.global internal constant @om_external_constant_offset_constant_0_tag_constants_to_file(4096 : i64) {addr_space = 0 : i32} : i64

// CHECK-CONST-TO-FILE:         llvm.mlir.global internal constant @om_external_constant_filename_tag_constants_to_file("model.constants.bin\00") {addr_space = 0 : i32}
// CHECK-CONST-TO-FILE:         llvm.mlir.global internal constant @om_external_constant_filesize_tag_constants_to_file(4176 : i64) {addr_space = 0 : i32} : i64
// CHECK-CONST-TO-FILE:         llvm.mlir.global internal constant @om_external_constant_isLE_tag_constants_to_file({{.*}} : i8) {addr_space = 0 : i32} : i8
// CHECK-CONST-TO-FILE:         llvm.mlir.global internal @om_external_constant_packedConst_tag_constants_to_file() {addr_space = 0 : i32} : !llvm.ptr {
// CHECK-CONST-TO-FILE:           [[VAR_0_15_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-CONST-TO-FILE:           llvm.return [[VAR_0_15_]] : !llvm.ptr
// CHECK-CONST-TO-FILE:         }

// CHECK-CONST-TO-FILE:         llvm.func @omLoadConstantsFromFile_tag_constants_to_file() {
// CHECK-CONST-TO-FILE-DAG:       [[VAR_0_18_:%.+]] = llvm.mlir.constant(4096 : i64) : i64
// CHECK-CONST-TO-FILE-DAG:       [[VAR_1_9_:%.+]] = llvm.mlir.addressof @om_external_constant_data_constant_0_tag_constants_to_file : !llvm.ptr
// CHECK-CONST-TO-FILE-DAG:       [[VAR_2_9_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-CONST-TO-FILE-DAG:       [[VAR_3_9_:%.+]] = llvm.mlir.addressof @om_external_constant_data_constant_1_tag_constants_to_file : !llvm.ptr
// CHECK-CONST-TO-FILE-DAG:       [[VAR_4_7_:%.+]] = llvm.mlir.constant(4176 : i64) : i64
// CHECK-CONST-TO-FILE-DAG:       [[VAR_5_8_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-CONST-TO-FILE-DAG:       [[VAR_6_6_:%.+]] = llvm.mlir.addressof @om_external_constant_packedConst_tag_constants_to_file : !llvm.ptr
// CHECK-CONST-TO-FILE-DAG:       [[VAR_7_4_:%.+]] = llvm.mlir.addressof @om_external_constant_filename_tag_constants_to_file : !llvm.ptr
// CHECK-CONST-TO-FILE:           llvm.call @omMMapBinaryFile([[VAR_6_6_]], [[VAR_7_4_]], [[VAR_4_7_]], [[VAR_5_8_]]) : (!llvm.ptr, !llvm.ptr, i64, i64) -> ()
// CHECK-CONST-TO-FILE:           llvm.call @omGetExternalConstantAddr([[VAR_3_9_]], [[VAR_6_6_]], [[VAR_2_9_]]) : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK-CONST-TO-FILE:           llvm.call @omGetExternalConstantAddr([[VAR_1_9_]], [[VAR_6_6_]], [[VAR_0_18_]]) : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK-CONST-TO-FILE:           llvm.return
// CHECK-CONST-TO-FILE:         }
// CHECK-CONST-TO-FILE:         llvm.func @omLoadConstantsFromFile() {
// CHECK-CONST-TO-FILE:           llvm.call @omLoadConstantsFromFile_tag_constants_to_file() : () -> ()
// CHECK-CONST-TO-FILE:           llvm.return
// CHECK-CONST-TO-FILE:         }

// CHECK-CONST-TO-FILE:         llvm.func @run_main_graph_tag_constants_to_file([[arg0_:%.+]]: !llvm.ptr) -> !llvm.ptr {
// CHECK-CONST-TO-FILE:           llvm.call @omLoadConstantsFromFile_tag_constants_to_file() : () -> ()
// CHECK-CONST-TO-FILE:         }
// CHECK-CONST-TO-FILE:         llvm.func @run_main_graph([[arg0_:%.+]]: !llvm.ptr) -> !llvm.ptr {
// CHECK-CONST-TO-FILE:           llvm.call @omLoadConstantsFromFile_tag_constants_to_file() : () -> ()
// CHECK-CONST-TO-FILE:           [[VAR_0_20_:%.+]] = llvm.call @run_main_graph_tag_constants_to_file([[arg0_]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK-CONST-TO-FILE:           llvm.return [[VAR_0_20_]] : !llvm.ptr
// CHECK-CONST-TO-FILE:         }
}
