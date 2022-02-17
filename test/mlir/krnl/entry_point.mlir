// RUN: onnx-mlir-opt --convert-krnl-to-llvm --canonicalize %s -split-input-file | FileCheck %s

// COM: Generate the default entry point "run_main_graph" since there is only
// COM: one single point.
module {
  func private @first_entry(%arg0: memref<10xf32>) -> memref<10xf32> {
    return %arg0 : memref<10xf32>
  }
  "krnl.entry_point"() {func = @first_entry, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[in_sig]\00@[out_sig]\00"} : () -> ()

// CHECK-DAG:     llvm.func @strncmp(!llvm.ptr<i8>, !llvm.ptr<i8>, i64) -> i32
// CHECK-DAG:     llvm.mlir.global external constant @_entry_point_0("run_main_graph\00")
// CHECK-DAG:     llvm.mlir.global external constant @_entry_point_0_in_sig("[in_sig]\00")
// CHECK-DAG:     llvm.mlir.global external constant @_entry_point_0_out_sig("[out_sig]\00")

// CHECK-LABEL:   llvm.func @run_main_graph({{.*}}: !llvm.ptr<i8>) -> !llvm.ptr<i8>

// CHECK:         llvm.func @omInputSignature([[arg0_:%.+]]: !llvm.ptr<i8>) -> !llvm.ptr<i8> {
// CHECK-DAG:       [[VAR_0_3_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:       [[VAR_1_3_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_2_3_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_3_2_:%.+]] = llvm.alloca [[VAR_2_3_]] x !llvm.ptr<i8> : (i64) -> !llvm.ptr<ptr<i8>>
// CHECK-DAG:       [[VAR_4_3_:%.+]] = llvm.mlir.addressof @_entry_point_0 : !llvm.ptr<array<15 x i8>>
// CHECK-DAG:       [[VAR_5_3_:%.+]] = llvm.getelementptr [[VAR_4_3_]]{{.}}[[VAR_1_3_]], [[VAR_1_3_]]{{.}} : (!llvm.ptr<array<15 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK-DAG:       [[VAR_6_2_:%.+]] = llvm.mlir.constant(15 : i64) : i64
// CHECK:           [[VAR_7_1_:%.+]] = llvm.call @strncmp([[arg0_]], [[VAR_5_3_]], [[VAR_6_2_]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64) -> i32
// CHECK:           [[VAR_8_1_:%.+]] = llvm.icmp "eq" [[VAR_7_1_]], [[VAR_0_3_]] : i32
// CHECK:           llvm.cond_br [[VAR_8_1_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           [[VAR_9_1_:%.+]] = llvm.mlir.addressof @_entry_point_0_in_sig : !llvm.ptr<array<9 x i8>>
// CHECK:           [[VAR_10_1_:%.+]] = llvm.bitcast [[VAR_9_1_]] : !llvm.ptr<array<9 x i8>> to !llvm.ptr<i8>
// CHECK:           llvm.store [[VAR_10_1_]], [[VAR_3_2_]] : !llvm.ptr<ptr<i8>>
// CHECK:           llvm.br ^bb3
// CHECK:         ^bb2:  // pred: ^bb0
// CHECK:           llvm.br ^bb3
// CHECK:         ^bb3:  // 2 preds: ^bb1, ^bb2
// CHECK:           [[VAR_11_1_:%.+]] = llvm.load [[VAR_3_2_]] : !llvm.ptr<ptr<i8>>
// CHECK:           llvm.return [[VAR_11_1_]] : !llvm.ptr<i8>
// CHECK:         }

// CHECK:         llvm.func @omOutputSignature([[arg0_:%.+]]: !llvm.ptr<i8>) -> !llvm.ptr<i8> {
// CHECK-DAG:       [[VAR_0_4_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:       [[VAR_1_4_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_2_4_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_3_3_:%.+]] = llvm.alloca [[VAR_2_4_]] x !llvm.ptr<i8> : (i64) -> !llvm.ptr<ptr<i8>>
// CHECK-DAG:       [[VAR_4_4_:%.+]] = llvm.mlir.addressof @_entry_point_0 : !llvm.ptr<array<15 x i8>>
// CHECK-DAG:       [[VAR_5_4_:%.+]] = llvm.getelementptr [[VAR_4_4_]]{{.}}[[VAR_1_4_]], [[VAR_1_4_]]{{.}} : (!llvm.ptr<array<15 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK-DAG:       [[VAR_6_3_:%.+]] = llvm.mlir.constant(15 : i64) : i64
// CHECK:           [[VAR_7_2_:%.+]] = llvm.call @strncmp([[arg0_]], [[VAR_5_4_]], [[VAR_6_3_]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64) -> i32
// CHECK:           [[VAR_8_2_:%.+]] = llvm.icmp "eq" [[VAR_7_2_]], [[VAR_0_4_]] : i32
// CHECK:           llvm.cond_br [[VAR_8_2_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           [[VAR_9_2_:%.+]] = llvm.mlir.addressof @_entry_point_0_out_sig : !llvm.ptr<array<10 x i8>>
// CHECK:           [[VAR_10_2_:%.+]] = llvm.bitcast [[VAR_9_2_]] : !llvm.ptr<array<10 x i8>> to !llvm.ptr<i8>
// CHECK:           llvm.store [[VAR_10_2_]], [[VAR_3_3_]] : !llvm.ptr<ptr<i8>>
// CHECK:           llvm.br ^bb3
// CHECK:         ^bb2:  // pred: ^bb0
// CHECK:           llvm.br ^bb3
// CHECK:         ^bb3:  // 2 preds: ^bb1, ^bb2
// CHECK:           [[VAR_11_1_:%.+]] = llvm.load [[VAR_3_3_]] : !llvm.ptr<ptr<i8>>
// CHECK:           llvm.return [[VAR_11_1_]] : !llvm.ptr<i8>
// CHECK:         }
}

// -----

// COM: Generate multiple entry points.
module {
  func private @first_entry(%arg0: memref<10xf32>) -> memref<10xf32> {
    return %arg0 : memref<10xf32>
  }
  func private @second_entry(%arg0: memref<10xf32>) -> memref<10xf32> {
    return %arg0 : memref<10xf32>
  }
  "krnl.entry_point"() {func = @first_entry, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[in_sig_0]\00@[out_sig_0]\00"} : () -> ()
  "krnl.entry_point"() {func = @second_entry, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[in_sig_1]\00@[out_sig_1]\00"} : () -> ()

// CHECK-DAG:     llvm.func @strncmp(!llvm.ptr<i8>, !llvm.ptr<i8>, i64) -> i32
// CHECK-DAG:     llvm.mlir.global external constant @_entry_point_0("run_first_entry\00")
// CHECK-DAG:     llvm.mlir.global external constant @_entry_point_0_in_sig("[in_sig_0]\00")
// CHECK-DAG:     llvm.mlir.global external constant @_entry_point_0_out_sig("[out_sig_0]\00")
// CHECK-DAG:     llvm.mlir.global external constant @_entry_point_1("run_second_entry\00")
// CHECK-DAG:     llvm.mlir.global external constant @_entry_point_1_in_sig("[in_sig_1]\00")
// CHECK-DAG:     llvm.mlir.global external constant @_entry_point_1_out_sig("[out_sig_1]\00")

// CHECK-LABEL:   llvm.func @run_first_entry({{.*}}: !llvm.ptr<i8>) -> !llvm.ptr<i8> {
// CHECK-LABEL:   llvm.func @run_second_entry({{.*}}: !llvm.ptr<i8>) -> !llvm.ptr<i8> {

// CHECK:         llvm.func @omInputSignature([[arg0_:%.+]]: !llvm.ptr<i8>) -> !llvm.ptr<i8> {
// CHECK-DAG:       [[VAR_0_5_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:       [[VAR_1_6_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_2_6_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_3_4_:%.+]] = llvm.alloca [[VAR_2_6_]] x !llvm.ptr<i8> : (i64) -> !llvm.ptr<ptr<i8>>
// CHECK-DAG:       [[VAR_4_6_:%.+]] = llvm.mlir.addressof @_entry_point_0 : !llvm.ptr<array<16 x i8>>
// CHECK-DAG:       [[VAR_5_6_:%.+]] = llvm.getelementptr [[VAR_4_6_]]{{.}}[[VAR_1_6_]], [[VAR_1_6_]]{{.}} : (!llvm.ptr<array<16 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK-DAG:       [[VAR_6_4_:%.+]] = llvm.mlir.constant(16 : i64) : i64
// CHECK:           [[VAR_7_2_:%.+]] = llvm.call @strncmp([[arg0_]], [[VAR_5_6_]], [[VAR_6_4_]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64) -> i32
// CHECK:           [[VAR_8_2_:%.+]] = llvm.icmp "eq" [[VAR_7_2_]], [[VAR_0_5_]] : i32
// CHECK:           llvm.cond_br [[VAR_8_2_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           [[VAR_9_2_:%.+]] = llvm.mlir.addressof @_entry_point_0_in_sig : !llvm.ptr<array<11 x i8>>
// CHECK:           [[VAR_10_2_:%.+]] = llvm.bitcast [[VAR_9_2_]] : !llvm.ptr<array<11 x i8>> to !llvm.ptr<i8>
// CHECK:           llvm.store [[VAR_10_2_]], [[VAR_3_4_]] : !llvm.ptr<ptr<i8>>
// CHECK:           llvm.br ^bb5
// CHECK:         ^bb2:  // pred: ^bb0
// CHECK:           [[VAR_11_2_:%.+]] = llvm.mlir.addressof @_entry_point_1 : !llvm.ptr<array<17 x i8>>
// CHECK-DAG:       [[VAR_12_2_:%.+]] = llvm.getelementptr [[VAR_11_2_]]{{.}}[[VAR_1_6_]], [[VAR_1_6_]]{{.}} : (!llvm.ptr<array<17 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK-DAG:       [[VAR_13_2_:%.+]] = llvm.mlir.constant(17 : i64) : i64
// CHECK:           [[LOAD_VAR_12_MEM_1_:%.+]] = llvm.call @strncmp([[arg0_]], [[VAR_12_2_]], [[VAR_13_2_]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64) -> i32
// CHECK:           [[VAR_15_2_:%.+]] = llvm.icmp "eq" [[LOAD_VAR_12_MEM_1_]], [[VAR_0_5_]] : i32
// CHECK:           llvm.cond_br [[VAR_15_2_]], ^bb3, ^bb4
// CHECK:         ^bb3:  // pred: ^bb2
// CHECK:           [[LOAD_VAR_13_MEM_1_:%.+]] = llvm.mlir.addressof @_entry_point_1_in_sig : !llvm.ptr<array<11 x i8>>
// CHECK:           [[VAR_17_2_:%.+]] = llvm.bitcast [[LOAD_VAR_13_MEM_1_]] : !llvm.ptr<array<11 x i8>> to !llvm.ptr<i8>
// CHECK:           llvm.store [[VAR_17_2_]], [[VAR_3_4_]] : !llvm.ptr<ptr<i8>>
// CHECK:           llvm.br ^bb5
// CHECK:         ^bb4:  // pred: ^bb2
// CHECK:           llvm.br ^bb5
// CHECK:         ^bb5:  // 3 preds: ^bb1, ^bb3, ^bb4
// CHECK:           [[LOAD_VAR_2_4_MEM_1_:%.+]] = llvm.load [[VAR_3_4_]] : !llvm.ptr<ptr<i8>>
// CHECK:           llvm.return [[LOAD_VAR_2_4_MEM_1_]] : !llvm.ptr<i8>
// CHECK:         }

// CHECK:         llvm.func @omOutputSignature([[arg0_:%.+]]: !llvm.ptr<i8>) -> !llvm.ptr<i8> {
// CHECK-DAG:       [[VAR_0_6_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:       [[VAR_1_7_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_2_7_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_3_5_:%.+]] = llvm.alloca [[VAR_2_7_]] x !llvm.ptr<i8> : (i64) -> !llvm.ptr<ptr<i8>>
// CHECK-DAG:       [[VAR_4_7_:%.+]] = llvm.mlir.addressof @_entry_point_0 : !llvm.ptr<array<16 x i8>>
// CHECK-DAG:       [[VAR_5_7_:%.+]] = llvm.getelementptr [[VAR_4_7_]]{{.}}[[VAR_1_7_]], [[VAR_1_7_]]{{.}} : (!llvm.ptr<array<16 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK-DAG:       [[VAR_6_5_:%.+]] = llvm.mlir.constant(16 : i64) : i64
// CHECK:           [[VAR_7_3_:%.+]] = llvm.call @strncmp([[arg0_]], [[VAR_5_7_]], [[VAR_6_5_]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64) -> i32
// CHECK:           [[VAR_8_3_:%.+]] = llvm.icmp "eq" [[VAR_7_3_]], [[VAR_0_6_]] : i32
// CHECK:           llvm.cond_br [[VAR_8_3_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           [[VAR_9_3_:%.+]] = llvm.mlir.addressof @_entry_point_0_out_sig : !llvm.ptr<array<12 x i8>>
// CHECK:           [[VAR_10_3_:%.+]] = llvm.bitcast [[VAR_9_3_]] : !llvm.ptr<array<12 x i8>> to !llvm.ptr<i8>
// CHECK:           llvm.store [[VAR_10_3_]], [[VAR_3_5_]] : !llvm.ptr<ptr<i8>>
// CHECK:           llvm.br ^bb5
// CHECK:         ^bb2:  // pred: ^bb0
// CHECK:           [[VAR_11_3_:%.+]] = llvm.mlir.addressof @_entry_point_1 : !llvm.ptr<array<17 x i8>>
// CHECK-DAG:       [[VAR_12_3_:%.+]] = llvm.getelementptr [[VAR_11_3_]]{{.}}[[VAR_1_7_]], [[VAR_1_7_]]{{.}} : (!llvm.ptr<array<17 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK-DAG:       [[VAR_13_3_:%.+]] = llvm.mlir.constant(17 : i64) : i64
// CHECK:           [[LOAD_VAR_12_MEM_1_1_:%.+]] = llvm.call @strncmp([[arg0_]], [[VAR_12_3_]], [[VAR_13_3_]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64) -> i32
// CHECK:           [[VAR_15_3_:%.+]] = llvm.icmp "eq" [[LOAD_VAR_12_MEM_1_1_]], [[VAR_0_6_]] : i32
// CHECK:           llvm.cond_br [[VAR_15_3_]], ^bb3, ^bb4
// CHECK:         ^bb3:  // pred: ^bb2
// CHECK:           [[LOAD_VAR_13_MEM_1_1_:%.+]] = llvm.mlir.addressof @_entry_point_1_out_sig : !llvm.ptr<array<12 x i8>>
// CHECK:           [[VAR_17_3_:%.+]] = llvm.bitcast [[LOAD_VAR_13_MEM_1_1_]] : !llvm.ptr<array<12 x i8>> to !llvm.ptr<i8>
// CHECK:           llvm.store [[VAR_17_3_]], [[VAR_3_5_]] : !llvm.ptr<ptr<i8>>
// CHECK:           llvm.br ^bb5
// CHECK:         ^bb4:  // pred: ^bb2
// CHECK:           llvm.br ^bb5
// CHECK:         ^bb5:  // 3 preds: ^bb1, ^bb3, ^bb4
// CHECK:           [[LOAD_VAR_2_4_MEM_1_1_:%.+]] = llvm.load [[VAR_3_5_]] : !llvm.ptr<ptr<i8>>
// CHECK:           llvm.return [[LOAD_VAR_2_4_MEM_1_1_]] : !llvm.ptr<i8>
// CHECK:         }
}
