// RUN: onnx-mlir-opt --convert-krnl-to-llvm --canonicalize %s -split-input-file | FileCheck %s

// COM: Generate the default entry point "run_main_graph" since there is only
// COM: one single point.
module {
  func private @first_entry(%arg0: memref<10xf32>) -> memref<10xf32> {
    return %arg0 : memref<10xf32>
  }
  "krnl.entry_point"() {func = @first_entry, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[in_sig]\00@[out_sig]\00"} : () -> ()

// CHECK:         llvm.func @strncmp(!llvm.ptr<i8>, !llvm.ptr<i8>, i64) -> i32
// CHECK:         llvm.mlir.global external constant @_entry_point_0("run_main_graph\00")
// CHECK:         llvm.mlir.global external constant @_entry_point_0_in_sig("[in_sig]\00")
// CHECK:         llvm.mlir.global external constant @_entry_point_0_out_sig("[out_sig]\00")
// CHECK:         llvm.mlir.global internal constant @_entry_point_arrays() : !llvm.array<2 x ptr<i8>> {
// CHECK:           [[VAR_0_:%.+]] = llvm.mlir.undef : !llvm.array<2 x ptr<i8>>
// CHECK:           [[VAR_1_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:           [[VAR_2_:%.+]] = llvm.mlir.addressof @_entry_point_0 : !llvm.ptr<array<15 x i8>>
// CHECK:           [[VAR_3_:%.+]] = llvm.getelementptr [[VAR_2_]]{{.}}[[VAR_1_]], [[VAR_1_]]{{.}} : (!llvm.ptr<array<15 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK:           [[VAR_4_:%.+]] = llvm.insertvalue [[VAR_3_]], [[VAR_0_]][0 : index] : !llvm.array<2 x ptr<i8>>
// CHECK:           [[VAR_5_:%.+]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK:           [[VAR_6_:%.+]] = llvm.insertvalue [[VAR_5_]], [[VAR_4_]][1 : index] : !llvm.array<2 x ptr<i8>>
// CHECK:           llvm.return [[VAR_6_]] : !llvm.array<2 x ptr<i8>>
// CHECK:         }

// CHECK-LABEL:   llvm.func @run_main_graph({{.*}}: !llvm.ptr<i8>) -> !llvm.ptr<i8>

// CHECK:         llvm.func @omQueryEntryPoints([[arg0_:%.+]]: !llvm.ptr<i64>) -> !llvm.ptr<ptr<i8>> {
// CHECK:           [[VAR_0_3_:%.+]] = llvm.mlir.null : !llvm.ptr<i64>
// CHECK:           [[VAR_1_4_:%.+]] = llvm.icmp "ne" [[arg0_]], [[VAR_0_3_]] : !llvm.ptr<i64>
// CHECK:           llvm.cond_br [[VAR_1_4_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           [[VAR_2_4_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           llvm.store [[VAR_2_4_]], [[arg0_]] : !llvm.ptr<i64>
// CHECK:           llvm.br ^bb3
// CHECK:         ^bb2:  // pred: ^bb0
// CHECK:           llvm.br ^bb3
// CHECK:         ^bb3:  // 2 preds: ^bb1, ^bb2
// CHECK:           [[VAR_3_3_:%.+]] = llvm.mlir.addressof @_entry_point_arrays : !llvm.ptr<array<2 x ptr<i8>>>
// CHECK:           [[VAR_4_4_:%.+]] = llvm.bitcast [[VAR_3_3_]] : !llvm.ptr<array<2 x ptr<i8>>> to !llvm.ptr<ptr<i8>>
// CHECK:           llvm.return [[VAR_4_4_]] : !llvm.ptr<ptr<i8>>
// CHECK:         }

// CHECK:         llvm.func @omInputSignature([[arg0_:%.+]]: !llvm.ptr<i8>) -> !llvm.ptr<i8> {
// CHECK-DAG:       [[VAR_0_4_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:       [[VAR_1_5_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_5_:%.+]] = llvm.alloca [[VAR_1_5_]] x !llvm.ptr<i8> : (i64) -> !llvm.ptr<ptr<i8>>
// CHECK-DAG:       [[VAR_3_4_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_4_5_:%.+]] = llvm.mlir.addressof @_entry_point_0 : !llvm.ptr<array<15 x i8>>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_5_4_:%.+]] = llvm.getelementptr [[VAR_4_5_]]{{.}}[[VAR_3_4_]], [[VAR_3_4_]]{{.}} : (!llvm.ptr<array<15 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK-DAG:       [[VAR_6_3_:%.+]] = llvm.mlir.constant(15 : i64) : i64
// CHECK:           [[VAR_7_1_:%.+]] = llvm.call @strncmp([[arg0_]], [[VAR_5_4_]], [[VAR_6_3_]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64) -> i32
// CHECK:           [[VAR_8_1_:%.+]] = llvm.icmp "eq" [[VAR_7_1_]], [[VAR_0_4_]] : i32
// CHECK:           llvm.cond_br [[VAR_8_1_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           [[VAR_9_1_:%.+]] = llvm.mlir.addressof @_entry_point_0_in_sig : !llvm.ptr<array<9 x i8>>
// CHECK:           [[VAR_10_1_:%.+]] = llvm.bitcast [[VAR_9_1_]] : !llvm.ptr<array<9 x i8>> to !llvm.ptr<i8>
// CHECK:           llvm.store [[VAR_10_1_]], [[VAR_2_5_]] : !llvm.ptr<ptr<i8>>
// CHECK:           llvm.br ^bb3
// CHECK:         ^bb2:  // pred: ^bb0
// CHECK:           [[VAR_11_1_:%.+]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK:           llvm.return [[VAR_11_1_]] : !llvm.ptr<i8>
// CHECK:         ^bb3:  // pred: ^bb1
// CHECK:           [[VAR_12_1_:%.+]] = llvm.load [[VAR_2_5_]] : !llvm.ptr<ptr<i8>>
// CHECK:           llvm.return [[VAR_12_1_]] : !llvm.ptr<i8>
// CHECK:         }

// CHECK:         llvm.func @omOutputSignature([[arg0_:%.+]]: !llvm.ptr<i8>) -> !llvm.ptr<i8> {
// CHECK-DAG:       [[VAR_0_5_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:       [[VAR_1_6_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_6_:%.+]] = llvm.alloca [[VAR_1_6_]] x !llvm.ptr<i8> : (i64) -> !llvm.ptr<ptr<i8>>
// CHECK-DAG:       [[VAR_3_5_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_4_6_:%.+]] = llvm.mlir.addressof @_entry_point_0 : !llvm.ptr<array<15 x i8>>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_5_5_:%.+]] = llvm.getelementptr [[VAR_4_6_]]{{.}}[[VAR_3_5_]], [[VAR_3_5_]]{{.}} : (!llvm.ptr<array<15 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK-DAG:       [[VAR_6_4_:%.+]] = llvm.mlir.constant(15 : i64) : i64
// CHECK:           [[VAR_7_2_:%.+]] = llvm.call @strncmp([[arg0_]], [[VAR_5_5_]], [[VAR_6_4_]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64) -> i32
// CHECK:           [[VAR_8_2_:%.+]] = llvm.icmp "eq" [[VAR_7_2_]], [[VAR_0_5_]] : i32
// CHECK:           llvm.cond_br [[VAR_8_2_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           [[VAR_9_2_:%.+]] = llvm.mlir.addressof @_entry_point_0_out_sig : !llvm.ptr<array<10 x i8>>
// CHECK:           [[VAR_10_2_:%.+]] = llvm.bitcast [[VAR_9_2_]] : !llvm.ptr<array<10 x i8>> to !llvm.ptr<i8>
// CHECK:           llvm.store [[VAR_10_2_]], [[VAR_2_6_]] : !llvm.ptr<ptr<i8>>
// CHECK:           llvm.br ^bb3
// CHECK:         ^bb2:  // pred: ^bb0
// CHECK:           [[VAR_11_2_:%.+]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK:           llvm.return [[VAR_11_2_]] : !llvm.ptr<i8>
// CHECK:         ^bb3:  // pred: ^bb1
// CHECK:           [[VAR_12_1_:%.+]] = llvm.load [[VAR_2_6_]] : !llvm.ptr<ptr<i8>>
// CHECK:           llvm.return [[VAR_12_1_]] : !llvm.ptr<i8>
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

// CHECK:         llvm.func @strncmp(!llvm.ptr<i8>, !llvm.ptr<i8>, i64) -> i32
// CHECK:         llvm.mlir.global external constant @_entry_point_0("run_first_entry\00")
// CHECK:         llvm.mlir.global external constant @_entry_point_0_in_sig("[in_sig_0]\00")
// CHECK:         llvm.mlir.global external constant @_entry_point_0_out_sig("[out_sig_0]\00")
// CHECK:         llvm.mlir.global external constant @_entry_point_1("run_second_entry\00")
// CHECK:         llvm.mlir.global external constant @_entry_point_1_in_sig("[in_sig_1]\00")
// CHECK:         llvm.mlir.global external constant @_entry_point_1_out_sig("[out_sig_1]\00")
// CHECK:         llvm.mlir.global internal constant @_entry_point_arrays() : !llvm.array<3 x ptr<i8>> {
// CHECK:           [[VAR_0_6_:%.+]] = llvm.mlir.undef : !llvm.array<3 x ptr<i8>>
// CHECK:           [[VAR_1_7_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:           [[VAR_2_6_:%.+]] = llvm.mlir.addressof @_entry_point_0 : !llvm.ptr<array<16 x i8>>
// CHECK:           [[VAR_3_5_:%.+]] = llvm.getelementptr [[VAR_2_6_]]{{.}}[[VAR_1_7_]], [[VAR_1_7_]]{{.}} : (!llvm.ptr<array<16 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK:           [[VAR_4_6_:%.+]] = llvm.insertvalue [[VAR_3_5_]], [[VAR_0_6_]][0 : index] : !llvm.array<3 x ptr<i8>>
// CHECK:           [[VAR_5_6_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK:           [[VAR_6_5_:%.+]] = llvm.mlir.addressof @_entry_point_1 : !llvm.ptr<array<17 x i8>>
// CHECK:           [[VAR_7_3_:%.+]] = llvm.getelementptr [[VAR_6_5_]]{{.}}[[VAR_5_6_]], [[VAR_5_6_]]{{.}} : (!llvm.ptr<array<17 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK:           [[VAR_8_3_:%.+]] = llvm.insertvalue [[VAR_7_3_]], [[VAR_4_6_]][1 : index] : !llvm.array<3 x ptr<i8>>
// CHECK:           [[VAR_9_3_:%.+]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK:           [[VAR_10_3_:%.+]] = llvm.insertvalue [[VAR_9_3_]], [[VAR_8_3_]][2 : index] : !llvm.array<3 x ptr<i8>>
// CHECK:           llvm.return [[VAR_10_3_]] : !llvm.array<3 x ptr<i8>>
// CHECK:         }

// CHECK-LABEL:   llvm.func @run_first_entry({{.*}}: !llvm.ptr<i8>) -> !llvm.ptr<i8> {
// CHECK-LABEL:   llvm.func @run_second_entry({{.*}}: !llvm.ptr<i8>) -> !llvm.ptr<i8> {

// CHECK:         llvm.func @omQueryEntryPoints([[arg0_:%.+]]: !llvm.ptr<i64>) -> !llvm.ptr<ptr<i8>> {
// CHECK:           [[VAR_0_11_:%.+]] = llvm.mlir.null : !llvm.ptr<i64>
// CHECK:           [[VAR_1_14_:%.+]] = llvm.icmp "ne" [[arg0_]], [[VAR_0_11_]] : !llvm.ptr<i64>
// CHECK:           llvm.cond_br [[VAR_1_14_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           [[VAR_2_14_:%.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK:           llvm.store [[VAR_2_14_]], [[arg0_]] : !llvm.ptr<i64>
// CHECK:           llvm.br ^bb3
// CHECK:         ^bb2:  // pred: ^bb0
// CHECK:           llvm.br ^bb3
// CHECK:         ^bb3:  // 2 preds: ^bb1, ^bb2
// CHECK:           [[VAR_3_11_:%.+]] = llvm.mlir.addressof @_entry_point_arrays : !llvm.ptr<array<3 x ptr<i8>>>
// CHECK:           [[VAR_4_14_:%.+]] = llvm.bitcast [[VAR_3_11_]] : !llvm.ptr<array<3 x ptr<i8>>> to !llvm.ptr<ptr<i8>>
// CHECK:           llvm.return [[VAR_4_14_]] : !llvm.ptr<ptr<i8>>
// CHECK:         }

// CHECK:         llvm.func @omInputSignature([[arg0_:%.+]]: !llvm.ptr<i8>) -> !llvm.ptr<i8> {
// CHECK-DAG:       [[VAR_0_12_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:       [[VAR_1_15_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_15_:%.+]] = llvm.alloca [[VAR_1_15_]] x !llvm.ptr<i8> : (i64) -> !llvm.ptr<ptr<i8>>
// CHECK-DAG:       [[VAR_3_12_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_4_15_:%.+]] = llvm.mlir.addressof @_entry_point_0 : !llvm.ptr<array<16 x i8>>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_5_13_:%.+]] = llvm.getelementptr [[VAR_4_15_]]{{.}}[[VAR_3_12_]], [[VAR_3_12_]]{{.}} : (!llvm.ptr<array<16 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK-DAG:       [[VAR_6_10_:%.+]] = llvm.mlir.constant(16 : i64) : i64
// CHECK:           [[VAR_7_6_:%.+]] = llvm.call @strncmp([[arg0_]], [[VAR_5_13_]], [[VAR_6_10_]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64) -> i32
// CHECK:           [[VAR_8_6_:%.+]] = llvm.icmp "eq" [[VAR_7_6_]], [[VAR_0_12_]] : i32
// CHECK:           llvm.cond_br [[VAR_8_6_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           [[VAR_9_6_:%.+]] = llvm.mlir.addressof @_entry_point_0_in_sig : !llvm.ptr<array<11 x i8>>
// CHECK:           [[VAR_10_6_:%.+]] = llvm.bitcast [[VAR_9_6_]] : !llvm.ptr<array<11 x i8>> to !llvm.ptr<i8>
// CHECK:           llvm.store [[VAR_10_6_]], [[VAR_2_15_]] : !llvm.ptr<ptr<i8>>
// CHECK:           llvm.br ^bb5
// CHECK:         ^bb2:  // pred: ^bb0
// CHECK-DAG:       [[VAR_11_5_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_12_4_:%.+]] = llvm.mlir.addressof @_entry_point_1 : !llvm.ptr<array<17 x i8>>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_13_3_:%.+]] = llvm.getelementptr [[VAR_12_4_]]{{.}}[[VAR_11_5_]], [[VAR_11_5_]]{{.}} : (!llvm.ptr<array<17 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK-DAG:       [[LOAD_VAR_12_MEM_1_1_:%.+]] = llvm.mlir.constant(17 : i64) : i64
// CHECK:           [[VAR_15_3_:%.+]] = llvm.call @strncmp([[arg0_]], [[VAR_13_3_]], [[LOAD_VAR_12_MEM_1_1_]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64) -> i32
// CHECK:           [[LOAD_VAR_13_MEM_1_1_:%.+]] = llvm.icmp "eq" [[VAR_15_3_]], [[VAR_0_12_]] : i32
// CHECK:           llvm.cond_br [[LOAD_VAR_13_MEM_1_1_]], ^bb3, ^bb4
// CHECK:         ^bb3:  // pred: ^bb2
// CHECK:           [[VAR_17_3_:%.+]] = llvm.mlir.addressof @_entry_point_1_in_sig : !llvm.ptr<array<11 x i8>>
// CHECK:           [[LOAD_VAR_2_3_MEM_1_1_:%.+]] = llvm.bitcast [[VAR_17_3_]] : !llvm.ptr<array<11 x i8>> to !llvm.ptr<i8>
// CHECK:           llvm.store [[LOAD_VAR_2_3_MEM_1_1_]], [[VAR_2_15_]] : !llvm.ptr<ptr<i8>>
// CHECK:           llvm.br ^bb5
// CHECK:         ^bb4:  // pred: ^bb2
// CHECK:           [[VAR_19_3_:%.+]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK:           llvm.return [[VAR_19_3_]] : !llvm.ptr<i8>
// CHECK:         ^bb5:  // 2 preds: ^bb1, ^bb3
// CHECK:           [[VAR_20_2_:%.+]] = llvm.load [[VAR_2_15_]] : !llvm.ptr<ptr<i8>>
// CHECK:           llvm.return [[VAR_20_2_]] : !llvm.ptr<i8>
// CHECK:         }

// CHECK:         llvm.func @omOutputSignature([[arg0_:%.+]]: !llvm.ptr<i8>) -> !llvm.ptr<i8> {
// CHECK-DAG:       [[VAR_0_13_:%.+]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:       [[VAR_1_16_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_16_:%.+]] = llvm.alloca [[VAR_1_16_]] x !llvm.ptr<i8> : (i64) -> !llvm.ptr<ptr<i8>>
// CHECK-DAG:       [[VAR_3_13_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_4_16_:%.+]] = llvm.mlir.addressof @_entry_point_0 : !llvm.ptr<array<16 x i8>>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_5_14_:%.+]] = llvm.getelementptr [[VAR_4_16_]]{{.}}[[VAR_3_13_]], [[VAR_3_13_]]{{.}} : (!llvm.ptr<array<16 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK-DAG:       [[VAR_6_11_:%.+]] = llvm.mlir.constant(16 : i64) : i64
// CHECK:           [[VAR_7_7_:%.+]] = llvm.call @strncmp([[arg0_]], [[VAR_5_14_]], [[VAR_6_11_]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64) -> i32
// CHECK:           [[VAR_8_7_:%.+]] = llvm.icmp "eq" [[VAR_7_7_]], [[VAR_0_13_]] : i32
// CHECK:           llvm.cond_br [[VAR_8_7_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK:           [[VAR_9_7_:%.+]] = llvm.mlir.addressof @_entry_point_0_out_sig : !llvm.ptr<array<12 x i8>>
// CHECK:           [[VAR_10_7_:%.+]] = llvm.bitcast [[VAR_9_7_]] : !llvm.ptr<array<12 x i8>> to !llvm.ptr<i8>
// CHECK:           llvm.store [[VAR_10_7_]], [[VAR_2_16_]] : !llvm.ptr<ptr<i8>>
// CHECK:           llvm.br ^bb5
// CHECK:         ^bb2:  // pred: ^bb0
// CHECK-DAG:       [[VAR_11_6_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_12_5_:%.+]] = llvm.mlir.addressof @_entry_point_1 : !llvm.ptr<array<17 x i8>>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_13_4_:%.+]] = llvm.getelementptr [[VAR_12_5_]]{{.}}[[VAR_11_6_]], [[VAR_11_6_]]{{.}} : (!llvm.ptr<array<17 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK-DAG:       [[LOAD_VAR_12_MEM_1_1_:%.+]] = llvm.mlir.constant(17 : i64) : i64
// CHECK:           [[VAR_15_4_:%.+]] = llvm.call @strncmp([[arg0_]], [[VAR_13_4_]], [[LOAD_VAR_12_MEM_1_1_]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64) -> i32
// CHECK:           [[LOAD_VAR_13_MEM_1_1_:%.+]] = llvm.icmp "eq" [[VAR_15_4_]], [[VAR_0_13_]] : i32
// CHECK:           llvm.cond_br [[LOAD_VAR_13_MEM_1_1_]], ^bb3, ^bb4
// CHECK:         ^bb3:  // pred: ^bb2
// CHECK:           [[VAR_17_4_:%.+]] = llvm.mlir.addressof @_entry_point_1_out_sig : !llvm.ptr<array<12 x i8>>
// CHECK:           [[LOAD_VAR_2_3_MEM_1_1_:%.+]] = llvm.bitcast [[VAR_17_4_]] : !llvm.ptr<array<12 x i8>> to !llvm.ptr<i8>
// CHECK:           llvm.store [[LOAD_VAR_2_3_MEM_1_1_]], [[VAR_2_16_]] : !llvm.ptr<ptr<i8>>
// CHECK:           llvm.br ^bb5
// CHECK:         ^bb4:  // pred: ^bb2
// CHECK:           [[VAR_19_4_:%.+]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK:           llvm.return [[VAR_19_4_]] : !llvm.ptr<i8>
// CHECK:         ^bb5:  // 2 preds: ^bb1, ^bb3
// CHECK:           [[VAR_20_2_1_:%.+]] = llvm.load [[VAR_2_16_]] : !llvm.ptr<ptr<i8>>
// CHECK:           llvm.return [[VAR_20_2_1_]] : !llvm.ptr<i8>
// CHECK:         }
}

// -----

// COM: Generate calls that initialize accelerators.
module attributes {"onnx-mlir.accels" = ["Pseudo-0x10001", "NNPA-0x10000"]} {
  func private @main_graph(%arg0: memref<10xf32>) -> memref<10xf32> {
    return %arg0 : memref<10xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[in_sig]\00@[out_sig]\00"} : () -> ()
// CHECK: llvm.func @OMInitCompatibleAccelNNPA(i64)
// CHECK: llvm.func @OMInitCompatibleAccelPseudo(i64)
// CHECK-LABEL: llvm.func @run_main_graph({{.*}}: !llvm.ptr<i8>) -> !llvm.ptr<i8> {
// CHECK-NEXT: [[FALSE:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT: [[VERSION_NUMBER:%.+]] = llvm.mlir.constant(65537 : i64) : i64
// CHECK-NEXT: [[COMPATIBLE:%.+]] = llvm.call @OMInitCompatibleAccelPseudo([[VERSION_NUMBER]]) : (i64) -> i64
// CHECK-NEXT: [[FAILED:%.+]] = llvm.icmp "eq" [[COMPATIBLE]], [[FALSE]] : i64
// CHECK-NEXT: llvm.cond_br [[FAILED]], ^bb1, ^bb2
// CHECK-NEXT: ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:   [[NULL:%.+]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK-NEXT:   llvm.return [[NULL]] : !llvm.ptr<i8>
// CHECK-NEXT: ^bb2:  // pred: ^bb0
// CHECK-NEXT:  [[VERSION_NUMBER:%.+]] = llvm.mlir.constant(65536 : i64) : i64
// CHECK-NEXT:  [[COMPATIBLE:%.+]] = llvm.call @OMInitCompatibleAccelNNPA([[VERSION_NUMBER]]) : (i64) -> i64
// CHECK-NEXT:  [[FAILED:%.+]] = llvm.icmp "eq" [[COMPATIBLE]], [[FALSE]] : i64
// CHECK-NEXT:  llvm.cond_br [[FAILED]], ^bb1, ^bb3
// CHECK-NEXT: ^bb3:  // pred: ^bb2
// CHECK-NEXT:   {{.*}} = llvm.call @omTensorListGetOmtArray(%arg0) : (!llvm.ptr<i8>) -> !llvm.ptr<ptr<i8>>
}
