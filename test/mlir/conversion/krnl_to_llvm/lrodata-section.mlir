// RUN: onnx-mlir-opt -O3 --convert-krnl-to-llvm="use-lrodata-section=true" %s -split-input-file | FileCheck %s

// Test if global constants are annotated with .lrodata section or not.
module {
  func.func @test_lrodata_section() -> memref<3xf32> {
    %0 = "krnl.global"() {name = "constant", alignment = 1024 : i64, shape = [3], value = dense<[0.0, 0.1, 0.2]> : tensor<3xf32>} : () -> memref<3xf32>
    return %0 : memref<3xf32>
  }
  "krnl.entry_point"() {func = @test_lrodata_section, numInputs = 0 : i32, numOutputs = 1 : i32, signature = "[in_sig]\00@[out_sig]\00"} : () -> ()

// CHECK:         llvm.mlir.global external constant @_entry_point_0("run_main_graph\00") {addr_space = 0 : i32, section = ".lrodata"}
// CHECK:         llvm.mlir.global external constant @_entry_point_0_in_sig("[in_sig]\00") {addr_space = 0 : i32, section = ".lrodata"}
// CHECK:         llvm.mlir.global external constant @_entry_point_0_out_sig("[out_sig]\00") {addr_space = 0 : i32, section = ".lrodata"}
// CHECK:         llvm.mlir.global internal constant @constant(dense<[0.000000e+00, 1.000000e-01, 2.000000e-01]> : tensor<3xf32>) {addr_space = 0 : i32, alignment = 1024 : i64, section = ".lrodata"} : !llvm.array<3 x f32>
// CHECK:         llvm.mlir.global internal constant @_entry_point_arrays() {addr_space = 0 : i32, section = ".lrodata"} : !llvm.array<2 x ptr> {
}
