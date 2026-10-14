// RUN: onnx-mlir-opt --convert-krnl-to-llvm="omit-compile-info=true" --canonicalize %s | FileCheck %s

// COM: Test that omCompilationInfo is always emitted, returning "{}" when
// COM: --omit-compile-info is set.
module attributes {"onnx-mlir.compile_options" = "test-options", "onnx-mlir.op_stats" = "{\"op1\": 5}"} {
  func.func private @main_graph(%arg0: memref<10xf32>) -> memref<10xf32> {
    return %arg0 : memref<10xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[in_sig]\00@[out_sig]\00"} : () -> ()

// CHECK:         llvm.mlir.global internal constant @om_compilation_info_json("{}\00") {addr_space = 0 : i32}

// CHECK:         llvm.func @omCompilationInfo() -> !llvm.ptr {
// CHECK:           [[VAR_0:%.+]] = llvm.mlir.addressof @om_compilation_info_json : !llvm.ptr
// CHECK:           llvm.return [[VAR_0]] : !llvm.ptr
// CHECK:         }
}
