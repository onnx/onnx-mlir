// RUN: onnx-mlir-opt --convert-krnl-to-llvm="store-constants-to-file constants-to-file-single-threshold=0.03 constants-to-file-total-threshold=0.00000006" --canonicalize %s -split-input-file && binary-decoder model.constants.bin -s 0 -n 40 --onnx::TensorProto::FLOAT -rm | FileCheck %s -check-prefix=BINARY_DECODER_0

// RUN: onnx-mlir-opt --convert-krnl-to-llvm="store-constants-to-file constants-to-file-single-threshold=0.03 constants-to-file-total-threshold=0.00000006" --canonicalize %s -split-input-file && binary-decoder model.constants.bin -s 4096 -n 80 --onnx::TensorProto::INT64 -rm | FileCheck %s -check-prefix=BINARY_DECODER_1

// Thresholds for this files: 
//  -constants-to-file-single-threshold=0.03: 30 bytes for a single constants 
//  -constants-to-file-total-threshold=0.00000006: 60 bytes for all constants 

// BINARY_DECODER_0: 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.1
// BINARY_DECODER_1: 1 2 3 4 5 6 7 8 9 10
module {
  func.func @test_constants_to_file() -> memref<10xi64> {
    %0 = "krnl.global"() {name = "constant_0", alignment = 4096: i64, shape = [10], value = dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]> : tensor<10xi64>} : () -> memref<10xi64>
    %1 = "krnl.global"() {name = "constant_1", alignment = 4096: i64, shape = [10], value = dense<[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.1]> : tensor<10xf32>} : () -> memref<10xf32>
    %2 = "krnl.global"() {name = "constant_2", alignment = 4096: i64, shape = [10], value = dense<[21, 22, 23, 24, 25, 26, 27, 28, 29, 30]> : tensor<10xi64>} : () -> memref<10xi64>
    return %2 : memref<10xi64>
  }
  "krnl.entry_point"() {func = @test_constants_to_file, numInputs = 0 : i32, numOutputs = 1 : i32, signature = "[in_sig]\00@[out_sig]\00"} : () -> ()
}
