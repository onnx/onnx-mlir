// RUN: onnx-mlir-opt --convert-krnl-to-llvm="verify-input-tensors=true" --canonicalize %s -split-input-file | FileCheck %s

// COM: Test symbol consistency verification for dynamic dimensions.
// COM: This test verifies that when multiple inputs share the same symbolic dimension,
// COM: the generated code checks that the actual dimension values are consistent.

module {
  func.func private @test_symbol_consistency(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>) -> memref<?x?xf32> {
    return %arg0 : memref<?x?xf32>
  }
  "krnl.entry_point"() {func = @test_symbol_consistency, numInputs = 2 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [\22batch_size\22 , \22seq_len\22] , \22name\22 : \22input0\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [\22batch_size\22 , \22seq_len\22] , \22name\22 : \22input1\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [\22batch_size\22 , \22seq_len\22], \22name\22 : \22output\22 }\0A\0A]\00"} : () -> ()

// CHECK-LABEL: llvm.func @run_main_graph
// CHECK-SAME: ([[ARG0:%.+]]: !llvm.ptr) -> !llvm.ptr

// COM: Check that symbol values array is allocated (2 symbols: batch_size, seq_len)
// CHECK: [[SYMBOL_ARRAY_SIZE:%.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK: [[SYMBOL_ARRAY:%.+]] = llvm.alloca [[SYMBOL_ARRAY_SIZE]] x i64

// COM: Check that symbol values are initialized to -1
// CHECK: [[MINUS_ONE:%.+]] = llvm.mlir.constant(-1 : i64) : i64
// CHECK: llvm.store [[MINUS_ONE]]

// COM: Check that error message globals are generated for symbol consistency
// CHECK-DAG: llvm.mlir.global internal constant @"om_Inconsistent dimension for symbol 'batch_size' at dimension 0 of input 0: expect %lld, but got %lld\0A"
// CHECK-DAG: llvm.mlir.global internal constant @"om_Inconsistent dimension for symbol 'seq_len' at dimension 1 of input 0: expect %lld, but got %lld\0A"
// CHECK-DAG: llvm.mlir.global internal constant @"om_Inconsistent dimension for symbol 'batch_size' at dimension 0 of input 1: expect %lld, but got %lld\0A"
// CHECK-DAG: llvm.mlir.global internal constant @"om_Inconsistent dimension for symbol 'seq_len' at dimension 1 of input 1: expect %lld, but got %lld\0A"

// COM: Check that error message globals are generated for non-negative validation
// CHECK-DAG: llvm.mlir.global internal constant @"om_Wrong size for dimension 0 ('batch_size') of input 0: expect a non-negative value\0A"
// CHECK-DAG: llvm.mlir.global internal constant @"om_Wrong size for dimension 1 ('seq_len') of input 0: expect a non-negative value\0A"
// CHECK-DAG: llvm.mlir.global internal constant @"om_Wrong size for dimension 0 ('batch_size') of input 1: expect a non-negative value\0A"
// CHECK-DAG: llvm.mlir.global internal constant @"om_Wrong size for dimension 1 ('seq_len') of input 1: expect a non-negative value\0A"

// COM: Verify consistency checking logic is present
// CHECK: llvm.icmp "eq"
// CHECK: llvm.cond_br
// CHECK: llvm.icmp "ne"
// CHECK: llvm.and
}
