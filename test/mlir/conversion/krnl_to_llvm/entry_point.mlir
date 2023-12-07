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
// CHECK-DAG:       [[VAR_0_:%.+]] = llvm.mlir.undef : !llvm.array<2 x ptr>
// CHECK-DAG:       [[VAR_2_:%.+]] = llvm.mlir.addressof @_entry_point_0 : !llvm.ptr
// CHECK:           [[VAR_4_:%.+]] = llvm.insertvalue [[VAR_2_]], [[VAR_0_]][0] : !llvm.array<2 x ptr>
// CHECK:           [[VAR_5_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_6_:%.+]] = llvm.insertvalue [[VAR_5_]], [[VAR_4_]][1] : !llvm.array<2 x ptr>
// CHECK:           llvm.return [[VAR_6_]] : !llvm.array<2 x ptr>
// CHECK:         }

// CHECK:         llvm.func @omQueryEntryPoints([[arg0_:%.+]]: !llvm.ptr) -> !llvm.ptr {
// CHECK-DAG:       [[VAR_2_4_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_0_3_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_1_4_:%.+]] = llvm.icmp "ne" [[arg0_]], [[VAR_0_3_]] : !llvm.ptr
// CHECK:           llvm.cond_br [[VAR_1_4_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           llvm.store [[VAR_2_4_]], [[arg0_]] : i64, !llvm.ptr
// CHECK:           llvm.br ^bb2
// CHECK:         ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK:           [[VAR_3_3_:%.+]] = llvm.mlir.addressof @_entry_point_arrays : !llvm.ptr
// CHECK:           llvm.return [[VAR_3_3_]] : !llvm.ptr
// CHECK:         }

// CHECK:         llvm.func @omInputSignature([[arg0_:%.+]]: !llvm.ptr) -> !llvm.ptr {
// CHECK-DAG:       [[VAR_0_4_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:       [[VAR_4_5_:%.+]] = llvm.mlir.addressof @_entry_point_0 : !llvm.ptr
// CHECK-DAG:       [[VAR_6_3_:%.+]] = llvm.mlir.constant(15 : i64) : i64
// CHECK:           [[VAR_7_1_:%.+]] = llvm.call @strncmp([[arg0_]], [[VAR_4_5_]], [[VAR_6_3_]]) : (!llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK:           [[VAR_8_1_:%.+]] = llvm.icmp "eq" [[VAR_7_1_]], [[VAR_0_4_]] : i32
// CHECK:           llvm.cond_br [[VAR_8_1_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           [[VAR_9_1_:%.+]] = llvm.mlir.addressof @_entry_point_0_in_sig : !llvm.ptr
// CHECK:           llvm.return [[VAR_9_1_]] : !llvm.ptr
// CHECK:         ^bb2:  // pred: ^bb0
// CHECK:           [[VAR_11_1_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           llvm.return [[VAR_11_1_]] : !llvm.ptr
// CHECK:         }

// CHECK:         llvm.func @omOutputSignature([[arg0_:%.+]]: !llvm.ptr) -> !llvm.ptr {
// CHECK-DAG:       [[VAR_0_5_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:       [[VAR_4_6_:%.+]] = llvm.mlir.addressof @_entry_point_0 : !llvm.ptr
// CHECK-DAG:       [[VAR_6_4_:%.+]] = llvm.mlir.constant(15 : i64) : i64
// CHECK:           [[VAR_7_2_:%.+]] = llvm.call @strncmp([[arg0_]], [[VAR_4_6_]], [[VAR_6_4_]]) : (!llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK:           [[VAR_8_2_:%.+]] = llvm.icmp "eq" [[VAR_7_2_]], [[VAR_0_5_]] : i32
// CHECK:           llvm.cond_br [[VAR_8_2_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           [[VAR_9_2_:%.+]] = llvm.mlir.addressof @_entry_point_0_out_sig : !llvm.ptr
// CHECK:           llvm.return [[VAR_9_2_]] : !llvm.ptr
// CHECK:         ^bb2:  // pred: ^bb0
// CHECK:           [[VAR_11_2_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           llvm.return [[VAR_11_2_]] : !llvm.ptr
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
// CHECK-DAG:       [[VAR_0_6_:%.+]] = llvm.mlir.undef : !llvm.array<3 x ptr>
// CHECK-DAG:       [[VAR_2_6_:%.+]] = llvm.mlir.addressof @_entry_point_0 : !llvm.ptr
// CHECK:           [[VAR_4_6_:%.+]] = llvm.insertvalue [[VAR_2_6_]], [[VAR_0_6_]][0] : !llvm.array<3 x ptr>
// CHECK:           [[VAR_6_5_:%.+]] = llvm.mlir.addressof @_entry_point_1 : !llvm.ptr
// CHECK:           [[VAR_8_3_:%.+]] = llvm.insertvalue [[VAR_6_5_]], [[VAR_4_6_]][1] : !llvm.array<3 x ptr>
// CHECK:           [[VAR_9_3_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_10_3_:%.+]] = llvm.insertvalue [[VAR_9_3_]], [[VAR_8_3_]][2] : !llvm.array<3 x ptr>
// CHECK:           llvm.return [[VAR_10_3_]] : !llvm.array<3 x ptr>
// CHECK:         }

// CHECK:         llvm.func @omQueryEntryPoints([[arg0_:%.+]]: !llvm.ptr) -> !llvm.ptr {
// CHECK-DAG:       [[VAR_0_11_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-DAG:       [[VAR_2_14_:%.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK:           [[VAR_1_14_:%.+]] = llvm.icmp "ne" [[arg0_]], [[VAR_0_11_]] : !llvm.ptr
// CHECK:           llvm.cond_br [[VAR_1_14_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           llvm.store [[VAR_2_14_]], [[arg0_]] : i64, !llvm.ptr
// CHECK:           llvm.br ^bb2
// CHECK:         ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK:           [[VAR_3_11_:%.+]] = llvm.mlir.addressof @_entry_point_arrays : !llvm.ptr
// CHECK:           llvm.return [[VAR_3_11_]] : !llvm.ptr
// CHECK:         }

// CHECK:         llvm.func @omInputSignature([[arg0_:%.+]]: !llvm.ptr) -> !llvm.ptr {
// CHECK-DAG:       [[VAR_0_12_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:       [[VAR_4_15_:%.+]] = llvm.mlir.addressof @_entry_point_0 : !llvm.ptr
// CHECK-DAG:       [[VAR_6_10_:%.+]] = llvm.mlir.constant(16 : i64) : i64
// CHECK-DAG:       [[LOAD_VAR_12_MEM_1_1_:%.+]] = llvm.mlir.constant(17 : i64) : i64
// CHECK:           [[VAR_7_6_:%.+]] = llvm.call @strncmp([[arg0_]], [[VAR_4_15_]], [[VAR_6_10_]]) : (!llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK:           [[VAR_8_6_:%.+]] = llvm.icmp "eq" [[VAR_7_6_]], [[VAR_0_12_]] : i32
// CHECK:           llvm.cond_br [[VAR_8_6_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           [[VAR_9_6_:%.+]] = llvm.mlir.addressof @_entry_point_0_in_sig : !llvm.ptr
// CHECK:           llvm.return [[VAR_9_6_]] : !llvm.ptr
// CHECK:         ^bb2:  // pred: ^bb0
// CHECK:           [[VAR_12_4_:%.+]] = llvm.mlir.addressof @_entry_point_1 : !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK:           [[VAR_15_3_:%.+]] = llvm.call @strncmp([[arg0_]], [[VAR_12_4_]], [[LOAD_VAR_12_MEM_1_1_]]) : (!llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK:           [[LOAD_VAR_13_MEM_1_1_:%.+]] = llvm.icmp "eq" [[VAR_15_3_]], [[VAR_0_12_]] : i32
// CHECK:           llvm.cond_br [[LOAD_VAR_13_MEM_1_1_]], ^bb3, ^bb4
// CHECK:         ^bb3:  // pred: ^bb2
// CHECK:           [[VAR_17_3_:%.+]] = llvm.mlir.addressof @_entry_point_1_in_sig : !llvm.ptr
// CHECK:           llvm.return [[VAR_17_3_]] : !llvm.ptr
// CHECK:         ^bb4:  // pred: ^bb2
// CHECK:           [[VAR_19_3_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           llvm.return [[VAR_19_3_]] : !llvm.ptr
// CHECK:         }

// CHECK:         llvm.func @omOutputSignature([[arg0_:%.+]]: !llvm.ptr) -> !llvm.ptr {
// CHECK-DAG:       [[VAR_0_13_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:       [[VAR_4_16_:%.+]] = llvm.mlir.addressof @_entry_point_0 : !llvm.ptr
// CHECK-DAG:       [[VAR_6_11_:%.+]] = llvm.mlir.constant(16 : i64) : i64
// CHECK-DAG:       [[LOAD_VAR_12_MEM_1_1_:%.+]] = llvm.mlir.constant(17 : i64) : i64
// CHECK:           [[VAR_7_7_:%.+]] = llvm.call @strncmp([[arg0_]], [[VAR_4_16_]], [[VAR_6_11_]]) : (!llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK:           [[VAR_8_7_:%.+]] = llvm.icmp "eq" [[VAR_7_7_]], [[VAR_0_13_]] : i32
// CHECK:           llvm.cond_br [[VAR_8_7_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           [[VAR_9_7_:%.+]] = llvm.mlir.addressof @_entry_point_0_out_sig : !llvm.ptr
// CHECK:           llvm.return [[VAR_9_7_]] : !llvm.ptr
// CHECK:         ^bb2:  // pred: ^bb0
// CHECK:           [[VAR_12_5_:%.+]] = llvm.mlir.addressof @_entry_point_1 : !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK:           [[VAR_15_4_:%.+]] = llvm.call @strncmp([[arg0_]], [[VAR_12_5_]], [[LOAD_VAR_12_MEM_1_1_]]) : (!llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK:           [[LOAD_VAR_13_MEM_1_1_:%.+]] = llvm.icmp "eq" [[VAR_15_4_]], [[VAR_0_13_]] : i32
// CHECK:           llvm.cond_br [[LOAD_VAR_13_MEM_1_1_]], ^bb3, ^bb4
// CHECK:         ^bb3:  // pred: ^bb2
// CHECK:           [[VAR_17_4_:%.+]] = llvm.mlir.addressof @_entry_point_1_out_sig : !llvm.ptr
// CHECK:           llvm.return [[VAR_17_4_]] : !llvm.ptr
// CHECK:         ^bb4:  // pred: ^bb2
// CHECK:           [[VAR_19_4_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           llvm.return [[VAR_19_4_]] : !llvm.ptr
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
// CHECK-DAG:    [[FALSE:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:    [[VERSION_NUMBER_0:%.+]] = llvm.mlir.constant(65537 : i64) : i64
// CHECK-DAG:    [[VERSION_NUMBER_1:%.+]] = llvm.mlir.constant(65536 : i64) : i64
// CHECK:        [[COMPATIBLE:%.+]] = llvm.call @OMInitCompatibleAccelPseudo([[VERSION_NUMBER_0]]) : (i64) -> i64
// CHECK-NEXT:   [[FAILED:%.+]] = llvm.icmp "eq" [[COMPATIBLE]], [[FALSE]] : i64
// CHECK-NEXT:   llvm.cond_br [[FAILED]], ^bb1, ^bb2
// CHECK-NEXT: ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:   [[NULL:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT:   llvm.return [[NULL]] : !llvm.ptr
// CHECK-NEXT: ^bb2:  // pred: ^bb0
// CHECK-NEXT:   [[COMPATIBLE:%.+]] = llvm.call @OMInitCompatibleAccelNNPA([[VERSION_NUMBER_1]]) : (i64) -> i64
// CHECK-NEXT:   [[FAILED:%.+]] = llvm.icmp "eq" [[COMPATIBLE]], [[FALSE]] : i64
// CHECK-NEXT:   llvm.cond_br [[FAILED]], ^bb1, ^bb3
// CHECK-NEXT: ^bb3:  // pred: ^bb2
// CHECK-NEXT:   {{.*}} = llvm.call @omTensorListGetOmtArray(%arg0) : (!llvm.ptr) -> !llvm.ptr
}

