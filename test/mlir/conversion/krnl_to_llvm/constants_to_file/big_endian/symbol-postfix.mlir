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
// CHECK-DAG:       [[VAR_0_:%.+]] = llvm.mlir.zero : !llvm.ptr
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
// CHECK:           llvm.return [[VAR_0_]] : !llvm.ptr
// CHECK:         }

// CHECK:         llvm.func @omInputSignature([[arg0_:%.+]]: !llvm.ptr) -> !llvm.ptr {
// CHECK:           [[VAR_0_7_:%.+]] = llvm.call @omInputSignature_tag_symbols([[arg0_]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK:           llvm.return [[VAR_0_7_]] : !llvm.ptr
// CHECK:         }

// CHECK:         llvm.func @omOutputSignature_tag_symbols([[arg0_]]: !llvm.ptr) -> !llvm.ptr {
// CHECK-DAG:       [[VAR_0_:%.+]] = llvm.mlir.zero : !llvm.ptr
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
// CHECK:         ^bb3: // pred: ^bb2
// CHECK:           llvm.return [[VAR_0_9_]]
// CHECK:         ^bb4: // pred: ^bb2
// CHECK:           llvm.return [[VAR_0_]]
// CHECK:         }
// CHECK:         llvm.func @omOutputSignature([[arg0_:%.+]]: !llvm.ptr) -> !llvm.ptr {
// CHECK:           [[VAR_0_9_:%.+]] = llvm.call @omOutputSignature_tag_symbols([[arg0_]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK:           llvm.return [[VAR_0_9_]] : !llvm.ptr
// CHECK:         }
}
