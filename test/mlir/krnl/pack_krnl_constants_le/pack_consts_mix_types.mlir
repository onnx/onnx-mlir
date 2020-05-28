// RUN: onnx-mlir-opt --pack-krnl-constants='elision-threshold=3 move-to-file=true filename=test1.bin' %s -split-input-file && binary-decoder test1.bin -s 0 -n 16 -FLOAT | FileCheck %s -check-prefix=BINARY_DECODER_1
// RUN: onnx-mlir-opt --pack-krnl-constants='elision-threshold=3 move-to-file=true filename=test2.bin' %s -split-input-file && binary-decoder test2.bin -s 16 -n 32 -INT32 -rm | FileCheck %s -check-prefix=BINARY_DECODER_2

// BINARY_DECODER_1: 0.1 0.2 0.3 0.4
// BINARY_DECODER_2: 1 2 3 4
func @test_krnl_const_packing_file_mixing_types() -> memref<1x4xf32> {
  %0 = "krnl.global"() {name = "constant_0", shape = [1, 4], value = dense<[[0.1, 0.2, 0.3, 0.4]]> : tensor<1x4xf32>} : () -> memref<1x4xf32>
  %1 = "krnl.global"() {name = "constant_1", shape = [1, 4], value = dense<[[1, 2, 3, 4]]> : tensor<1x4xi32>} : () -> memref<1x4xi32>
  return %0 : memref<1x4xf32>
}