// RUN: onnx-mlir-opt --pack-krnl-constants='elision-threshold=3 move-to-file=true filename=test-pack-consts-to-file-same-type.bin' %s -split-input-file && binary-decoder test-pack-consts-to-file-same-type.bin -s 0 -n 32 --onnx::TensorProto::FLOAT -rm | FileCheck %s
// RUN: onnx-mlir-opt --pack-krnl-constants='elision-threshold=3 move-to-file=true filename=test-pack-consts-to-file-same-type.bin' %s -split-input-file | FileCheck %s --check-prefix=ALIGN

// CHECK: 0 1 2 3 0 1 2 3
func @test_krnl_const_packing_file() -> memref<1x4xf32> {
  %0 = "krnl.global"() {name = "constant_0", shape = [1, 4], value = dense<[[0., 1., 2., 3.]]> : tensor<1x4xf32>} : () -> memref<1x4xf32>
  %1 = "krnl.global"() {name = "constant_1", shape = [1, 4], value = dense<[[0., 1., 2., 3.]]> : tensor<1x4xf32>} : () -> memref<1x4xf32>
  return %0 : memref<1x4xf32>
}

// -----

// COM: do not pack aligned constants.
func @test_krnl_const_packing_file_alignment() -> memref<1x4xf32> {
  %0 = "krnl.global"() {name = "constant_0", shape = [1, 4], value = dense<[[0., 1., 2., 3.]]> : tensor<1x4xf32>} : () -> memref<1x4xf32>
  %1 = "krnl.global"() {name = "constant_1", shape = [1, 4], value = dense<[[0., 1., 2., 3.]]> : tensor<1x4xf32>} : () -> memref<1x4xf32>
  %2 = "krnl.global"() {name = "constant_2", shape = [1, 4], value = dense<[[0., 1., 2., 3.]]> : tensor<1x4xf32>, alignment = 1024} : () -> memref<1x4xf32>
  %3 = "krnl.global"() {name = "constant_3", shape = [1, 4], value = dense<[[0., 1., 2., 3.]]> : tensor<1x4xf32>, alignment = 1024} : () -> memref<1x4xf32>
  return %0 : memref<1x4xf32>

  // ALIGN: {{.*}} = "krnl.global"()  {{.*}}
  // ALIGN: {{.*}} = "krnl.global"()  {{.*}}
  // ALIGN: {{.*}} = "krnl.global"() {alignment = 1024 : i64, name = "constant_2", shape = [1, 4]} : () -> memref<1x4xf32>
  // ALIGN: {{.*}} = "krnl.global"() {alignment = 1024 : i64, name = "constant_3", shape = [1, 4]} : () -> memref<1x4xf32>
}
