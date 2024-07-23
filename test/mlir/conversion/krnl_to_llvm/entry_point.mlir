// RUN: onnx-mlir-opt --convert-krnl-to-llvm --canonicalize %s -split-input-file | FileCheck %s

// COM: Generate the default entry point "run_main_graph" since there is only
// COM: one single point.
module {
  func.func private @first_entry(%arg0: memref<10xf32>) -> memref<10xf32> {
    return %arg0 : memref<10xf32>
  }
  "krnl.entry_point"() {func = @first_entry, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[in_sig]\00@[out_sig]\00"} : () -> ()

// CHECK:         llvm.func @strncmp(!llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK:         llvm.mlir.global external constant @_entry_point_0("run_main_graph\00")
// CHECK:         llvm.mlir.global external constant @_entry_point_0_in_sig("[in_sig]\00")
// CHECK:         llvm.mlir.global external constant @_entry_point_0_out_sig("[out_sig]\00")

// CHECK-LABEL:   llvm.func @run_main_graph
// CHECK:             ([[ARG0:%.+]]: !llvm.ptr) -> !llvm.ptr
// CHECK:           {{.*}} = llvm.call @omTensorListGetOmtArray([[ARG0]]) : (!llvm.ptr) -> !llvm.ptr

// CHECK:         llvm.mlir.global internal constant @_entry_point_arrays() {addr_space = 0 : i32} : !llvm.array<2 x ptr> {
// CHECK-DAG:       [[VAR_0:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-DAG:       [[VAR_0_3_:%.+]] = llvm.mlir.addressof @_entry_point_0 : !llvm.ptr
// CHECK-DAG:       [[VAR_1_3_:%.+]] = llvm.mlir.undef : !llvm.array<2 x ptr>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_3_:%.+]] = llvm.insertvalue [[VAR_0_3_]], [[VAR_1_3_]][0] : !llvm.array<2 x ptr>
// CHECK:           [[VAR_4_2_:%.+]] = llvm.insertvalue [[VAR_0]], [[VAR_2_3_]][1] : !llvm.array<2 x ptr>
// CHECK:           llvm.return [[VAR_4_2_]] : !llvm.array<2 x ptr>
// CHECK:         }

// CHECK:         llvm.func @omQueryEntryPoints([[arg0_:%.+]]: !llvm.ptr) -> !llvm.ptr {
// CHECK-DAG:       [[VAR_0_4_:%.+]] = llvm.mlir.addressof @_entry_point_arrays : !llvm.ptr
// CHECK-DAG:       [[VAR_1_4_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_2_4_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_3_4_:%.+]] = llvm.icmp "ne" [[arg0_]], [[VAR_2_4_]] : !llvm.ptr
// CHECK:           llvm.cond_br [[VAR_3_4_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           llvm.store [[VAR_1_4_]], [[arg0_]] : i64, !llvm.ptr
// CHECK:           llvm.br ^bb2
// CHECK:         ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK:           llvm.return [[VAR_0_4_]] : !llvm.ptr
// CHECK:         }

// CHECK:         llvm.func @omInputSignature([[arg0_:%.+]]: !llvm.ptr) -> !llvm.ptr {
// CHECK-DAG:       [[VAR_0:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-DAG:       [[VAR_0_5_:%.+]] = llvm.mlir.addressof @_entry_point_0_in_sig : !llvm.ptr
// CHECK-DAG:       [[VAR_1_5_:%.+]] = llvm.mlir.constant(15 : i64) : i64
// CHECK-DAG:       [[VAR_2_5_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:       [[VAR_3_5_:%.+]] = llvm.mlir.addressof @_entry_point_0 : !llvm.ptr
// CHECK:           [[VAR_4_3_:%.+]] = llvm.call @strncmp([[arg0_]], [[VAR_3_5_]], [[VAR_1_5_]]) : (!llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK:           [[VAR_5_3_:%.+]] = llvm.icmp "eq" [[VAR_4_3_]], [[VAR_2_5_]] : i32
// CHECK:           llvm.cond_br [[VAR_5_3_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           llvm.return [[VAR_0_5_]] : !llvm.ptr
// CHECK:         ^bb2:  // pred: ^bb0
// CHECK:           llvm.return [[VAR_0]] : !llvm.ptr
// CHECK:         }

// CHECK:         llvm.func @omOutputSignature([[arg0_:%.+]]: !llvm.ptr) -> !llvm.ptr {
// CHECK-DAG:       [[VAR_0:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-DAG:       [[VAR_0_6_:%.+]] = llvm.mlir.addressof @_entry_point_0_out_sig : !llvm.ptr
// CHECK-DAG:       [[VAR_1_6_:%.+]] = llvm.mlir.constant(15 : i64) : i64
// CHECK-DAG:       [[VAR_2_6_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:       [[VAR_3_6_:%.+]] = llvm.mlir.addressof @_entry_point_0 : !llvm.ptr
// CHECK:           [[VAR_4_4_:%.+]] = llvm.call @strncmp([[arg0_]], [[VAR_3_6_]], [[VAR_1_6_]]) : (!llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK:           [[VAR_5_4_:%.+]] = llvm.icmp "eq" [[VAR_4_4_]], [[VAR_2_6_]] : i32
// CHECK:           llvm.cond_br [[VAR_5_4_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           llvm.return [[VAR_0_6_]] : !llvm.ptr
// CHECK:         ^bb2:  // pred: ^bb0
// CHECK:           llvm.return [[VAR_0]] : !llvm.ptr
// CHECK:         }
}

// -----

// COM: Generate multiple entry points.
module {
  func.func private @first_entry(%arg0: memref<10xf32>) -> memref<10xf32> {
    return %arg0 : memref<10xf32>
  }
  func.func private @second_entry(%arg0: memref<10xf32>) -> memref<10xf32> {
    return %arg0 : memref<10xf32>
  }
  "krnl.entry_point"() {func = @first_entry, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[in_sig_0]\00@[out_sig_0]\00"} : () -> ()
  "krnl.entry_point"() {func = @second_entry, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[in_sig_1]\00@[out_sig_1]\00"} : () -> ()

// CHECK:         llvm.func @strncmp(!llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK-DAG:     llvm.mlir.global external constant @_entry_point_0("run_first_entry\00")
// CHECK-DAG:     llvm.mlir.global external constant @_entry_point_0_in_sig("[in_sig_0]\00")
// CHECK-DAG:     llvm.mlir.global external constant @_entry_point_0_out_sig("[out_sig_0]\00")
// CHECK-DAG:     llvm.mlir.global external constant @_entry_point_1("run_second_entry\00")
// CHECK-DAG:     llvm.mlir.global external constant @_entry_point_1_in_sig("[in_sig_1]\00")
// CHECK-DAG:     llvm.mlir.global external constant @_entry_point_1_out_sig("[out_sig_1]\00")

// CHECK:         llvm.func @run_first_entry([[ARG0:%.+]]: !llvm.ptr) -> !llvm.ptr {
// CHECK:           {{.*}} = llvm.call @omTensorListGetOmtArray([[ARG0]]) : (!llvm.ptr) -> !llvm.ptr

// CHECK:         llvm.func @run_second_entry([[ARG0:%.+]]: !llvm.ptr) -> !llvm.ptr {
// CHECK:           {{.*}} = llvm.call @omTensorListGetOmtArray([[ARG0]]) : (!llvm.ptr) -> !llvm.ptr

// CHECK:         llvm.mlir.global internal constant @_entry_point_arrays() {addr_space = 0 : i32} : !llvm.array<3 x ptr> {
// CHECK-DAG:       [[VAR_0:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-DAG:       [[VAR_0_11_:%.+]] = llvm.mlir.addressof @_entry_point_1 : !llvm.ptr
// CHECK-DAG:       [[VAR_1_13_:%.+]] = llvm.mlir.addressof @_entry_point_0 : !llvm.ptr
// CHECK-DAG:       [[VAR_2_13_:%.+]] = llvm.mlir.undef : !llvm.array<3 x ptr>
// CHECK:           [[VAR_3_13_:%.+]] = llvm.insertvalue [[VAR_1_13_]], [[VAR_2_13_]][0] : !llvm.array<3 x ptr>
// CHECK-DAG:       [[VAR_4_9_:%.+]] = llvm.insertvalue [[VAR_0_11_]], [[VAR_3_13_]][1] : !llvm.array<3 x ptr>
// CHECK:           [[VAR_6_8_:%.+]] = llvm.insertvalue [[VAR_0]], [[VAR_4_9_]][2] : !llvm.array<3 x ptr>
// CHECK:           llvm.return [[VAR_6_8_]] : !llvm.array<3 x ptr>
// CHECK:         }

// CHECK:         llvm.func @omQueryEntryPoints([[arg0_:%.+]]: !llvm.ptr) -> !llvm.ptr {
// CHECK-DAG:       [[VAR_0_12_:%.+]] = llvm.mlir.addressof @_entry_point_arrays : !llvm.ptr
// CHECK-DAG:       [[VAR_1_14_:%.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-DAG:       [[VAR_2_14_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_3_14_:%.+]] = llvm.icmp "ne" [[arg0_]], [[VAR_2_14_]] : !llvm.ptr
// CHECK:           llvm.cond_br [[VAR_3_14_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           llvm.store [[VAR_1_14_]], [[arg0_]] : i64, !llvm.ptr
// CHECK:           llvm.br ^bb2
// CHECK:         ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK:           llvm.return [[VAR_0_12_]] : !llvm.ptr
// CHECK:         }

// CHECK:         llvm.func @omInputSignature([[arg0_:%.+]]: !llvm.ptr) -> !llvm.ptr {
// CHECK-DAG:       [[VAR_0:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-DAG:       [[VAR_0_13_:%.+]] = llvm.mlir.addressof @_entry_point_1_in_sig : !llvm.ptr
// CHECK-DAG:       [[VAR_1_15_:%.+]] = llvm.mlir.constant(17 : i64) : i64
// CHECK-DAG:       [[VAR_2_15_:%.+]] = llvm.mlir.addressof @_entry_point_1 : !llvm.ptr
// CHECK-DAG:       [[VAR_3_15_:%.+]] = llvm.mlir.addressof @_entry_point_0_in_sig : !llvm.ptr
// CHECK-DAG:       [[VAR_4_10_:%.+]] = llvm.mlir.constant(16 : i64) : i64
// CHECK-DAG:       [[VAR_5_12_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:       [[VAR_6_9_:%.+]] = llvm.mlir.addressof @_entry_point_0 : !llvm.ptr
// CHECK:           [[VAR_7_3_:%.+]] = llvm.call @strncmp([[arg0_]], [[VAR_6_9_]], [[VAR_4_10_]]) : (!llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK:           [[VAR_8_3_:%.+]] = llvm.icmp "eq" [[VAR_7_3_]], [[VAR_5_12_]] : i32
// CHECK:           llvm.cond_br [[VAR_8_3_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           llvm.return [[VAR_3_15_]] : !llvm.ptr
// CHECK:         ^bb2:  // pred: ^bb0
// CHECK:           [[VAR_10_3_:%.+]] = llvm.call @strncmp([[arg0_]], [[VAR_2_15_]], [[VAR_1_15_]]) : (!llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK:           [[VAR_11_3_:%.+]] = llvm.icmp "eq" [[VAR_10_3_]], [[VAR_5_12_]] : i32
// CHECK:           llvm.cond_br [[VAR_11_3_]], ^bb3, ^bb4
// CHECK:         ^bb3:  // pred: ^bb2
// CHECK:           llvm.return [[VAR_0_13_]] : !llvm.ptr
// CHECK:         ^bb4:  // pred: ^bb2
// CHECK:           llvm.return [[VAR_0]] : !llvm.ptr
// CHECK:         }

// CHECK:         llvm.func @omOutputSignature([[arg0_:%.+]]: !llvm.ptr) -> !llvm.ptr {
// CHECK-DAG:       [[VAR_0:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-DAG:       [[VAR_0_14_:%.+]] = llvm.mlir.addressof @_entry_point_1_out_sig : !llvm.ptr
// CHECK-DAG:       [[VAR_1_16_:%.+]] = llvm.mlir.constant(17 : i64) : i64
// CHECK-DAG:       [[VAR_2_16_:%.+]] = llvm.mlir.addressof @_entry_point_1 : !llvm.ptr
// CHECK-DAG:       [[VAR_3_16_:%.+]] = llvm.mlir.addressof @_entry_point_0_out_sig : !llvm.ptr
// CHECK-DAG:       [[VAR_4_11_:%.+]] = llvm.mlir.constant(16 : i64) : i64
// CHECK-DAG:       [[VAR_5_13_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:       [[VAR_6_10_:%.+]] = llvm.mlir.addressof @_entry_point_0 : !llvm.ptr
// CHECK:           [[VAR_7_4_:%.+]] = llvm.call @strncmp([[arg0_]], [[VAR_6_10_]], [[VAR_4_11_]]) : (!llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK:           [[VAR_8_4_:%.+]] = llvm.icmp "eq" [[VAR_7_4_]], [[VAR_5_13_]] : i32
// CHECK:           llvm.cond_br [[VAR_8_4_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           llvm.return [[VAR_3_16_]] : !llvm.ptr
// CHECK:         ^bb2:  // pred: ^bb0
// CHECK:           [[VAR_10_4_:%.+]] = llvm.call @strncmp([[arg0_]], [[VAR_2_16_]], [[VAR_1_16_]]) : (!llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK:           [[VAR_11_4_:%.+]] = llvm.icmp "eq" [[VAR_10_4_]], [[VAR_5_13_]] : i32
// CHECK:           llvm.cond_br [[VAR_11_4_]], ^bb3, ^bb4
// CHECK:         ^bb3:  // pred: ^bb2
// CHECK:           llvm.return [[VAR_0_14_]] : !llvm.ptr
// CHECK:         ^bb4:  // pred: ^bb2
// CHECK:           llvm.return [[VAR_0]] : !llvm.ptr
// CHECK:         }
}

// -----

// COM: Generate calls that initialize accelerators.
module attributes {"onnx-mlir.accels" = ["Pseudo-0x10001", "NNPA-0x10000"]} {
  func.func private @main_graph(%arg0: memref<10xf32>) -> memref<10xf32> {
    return %arg0 : memref<10xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[in_sig]\00@[out_sig]\00"} : () -> ()
// CHECK:      llvm.func @OMInitCompatibleAccelNNPA(i64)
// CHECK:      llvm.func @OMInitCompatibleAccelPseudo(i64)
// CHECK:      llvm.func @run_main_graph({{.*}}: !llvm.ptr) -> !llvm.ptr {
// CHECK-DAG:    [[VAR_0:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-DAG:    [[FALSE:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:    [[VERSION_NUMBER_0:%.+]] = llvm.mlir.constant(65537 : i64) : i64
// CHECK-DAG:    [[VERSION_NUMBER_1:%.+]] = llvm.mlir.constant(65536 : i64) : i64
// CHECK:        [[COMPATIBLE:%.+]] = llvm.call @OMInitCompatibleAccelPseudo([[VERSION_NUMBER_0]]) : (i64) -> i64
// CHECK-NEXT:   [[FAILED:%.+]] = llvm.icmp "eq" [[COMPATIBLE]], [[FALSE]] : i64
// CHECK-NEXT:   llvm.cond_br [[FAILED]], ^bb1, ^bb2
// CHECK-NEXT: ^bb1:  // pred: ^bb0
// CHECK-NEXT:   llvm.return [[VAR_0]] : !llvm.ptr
// CHECK-NEXT: ^bb2:  // pred: ^bb0
// CHECK-NEXT:   [[COMPATIBLE:%.+]] = llvm.call @OMInitCompatibleAccelNNPA([[VERSION_NUMBER_1]]) : (i64) -> i64
// CHECK-NEXT:   [[FAILED:%.+]] = llvm.icmp "eq" [[COMPATIBLE]], [[FALSE]] : i64
// CHECK-NEXT:   llvm.cond_br [[FAILED]], ^bb3, ^bb4
// CHECK-NEXT: ^bb3:  // pred: ^bb2
// CHECK-NEXT:   llvm.return [[VAR_0]] : !llvm.ptr
// CHECK-NEXT: ^bb4:  // pred: ^bb2
// CHECK-NEXT:   {{.*}} = llvm.call @omTensorListGetOmtArray(%arg0) : (!llvm.ptr) -> !llvm.ptr
}

