// RUN: onnx-mlir-opt --convert-krnl-to-llvm --canonicalize %s -split-input-file | FileCheck %s

// COM: Test that omCompilationInfo function and global JSON string are generated correctly
module attributes {"onnx-mlir.compile_options" = "test-options", "onnx-mlir.op_stats" = "{\"op1\": 5}"} {
  func.func private @main_graph(%arg0: memref<10xf32>) -> memref<10xf32> {
    return %arg0 : memref<10xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[in_sig]\00@[out_sig]\00"} : () -> ()

// CHECK:         llvm.mlir.global internal constant @om_compilation_info_json("{\22compile_options\22: \22test-options\22, \22op_stats\22: {\22op1\22: 5}}\00") {addr_space = 0 : i32}

// CHECK:         llvm.func @omCompilationInfo() -> !llvm.ptr {
// CHECK:           [[VAR_0:%.+]] = llvm.mlir.addressof @om_compilation_info_json : !llvm.ptr
// CHECK:           llvm.return [[VAR_0]] : !llvm.ptr
// CHECK:         }
}

// -----

// COM: Test with empty op_stats
module attributes {"onnx-mlir.compile_options" = "-O3 --EmitLib", "onnx-mlir.op_stats" = ""} {
  func.func private @test_entry(%arg0: memref<5xi32>) -> memref<5xi32> {
    return %arg0 : memref<5xi32>
  }
  "krnl.entry_point"() {func = @test_entry, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[test_in]\00@[test_out]\00"} : () -> ()

// CHECK:         llvm.mlir.global internal constant @om_compilation_info_json("{\22compile_options\22: \22-O3 --EmitLib\22, \22op_stats\22: }\00") {addr_space = 0 : i32}

// CHECK:         llvm.func @omCompilationInfo() -> !llvm.ptr {
// CHECK:           [[VAR_0:%.+]] = llvm.mlir.addressof @om_compilation_info_json : !llvm.ptr
// CHECK:           llvm.return [[VAR_0]] : !llvm.ptr
// CHECK:         }
}

// -----

// COM: Test with missing attributes (should generate empty values)
module {
  func.func private @simple_entry(%arg0: memref<3xf64>) -> memref<3xf64> {
    return %arg0 : memref<3xf64>
  }
  "krnl.entry_point"() {func = @simple_entry, numInputs = 1 : i32, numOutputs = 1 : i32, signature = "[simple_in]\00@[simple_out]\00"} : () -> ()

// CHECK:         llvm.mlir.global internal constant @om_compilation_info_json("{\22compile_options\22: \22\22, \22op_stats\22: }\00") {addr_space = 0 : i32}

// CHECK:         llvm.func @omCompilationInfo() -> !llvm.ptr {
// CHECK:           [[VAR_0:%.+]] = llvm.mlir.addressof @om_compilation_info_json : !llvm.ptr
// CHECK:           llvm.return [[VAR_0]] : !llvm.ptr
// CHECK:         }
}
