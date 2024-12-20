// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --convert-krnl-to-llvm %s -split-input-file | FileCheck %s

// -----

// COM: Check the lowering of an zlow operation when its shape includes constant dims.
// COM: In this case, the constant values will be passed directly to
// COM: 'zdnn_init_pre_transformed_desc' that initializes a zTensor descriptor.
// COM: Using zlow.softmax as an example.
func.func @test_zlow_softmax_constant_shape() -> () {
  // %0 = "onnx.Softmax"(%arg0) : (memref<5x10xf32>) -> memref<5x10xf32>
  // "func.return"(%0) : (memref<5x10xf32>) -> ()
  %shape = "krnl.global"() {name = "constant_fold_std_alloc_0", shape = [3], value = dense<[1, 5, 10]> : tensor<3xi64>} : () -> memref<3xi64>
  %res = memref.alloc() {alignment = 4096 : i64} : memref<1x1x1x1x32x64xf16>
  %input = memref.alloc() {alignment = 4096 : i64} : memref<1x1x1x1x32x64xf16>
  %work_area = memref.alloc() {alignment = 4096 : i64} : memref<8192xi8>
  "zlow.softmax"(%input, %work_area, %shape, %res) {act_func = "ACT_NONE"} : (memref<1x1x1x1x32x64xf16>, memref<8192xi8>, memref<3xi64>, memref<1x1x1x1x32x64xf16>) -> ()
  return

  // CHECK-LABEL:   llvm.func @test_zlow_softmax_constant_shape() {{.*}} {
  // CHECK:           %[[DIM0:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK:           %[[DIM1:.*]] = llvm.mlir.constant(5 : i64) : i64
  // CHECK:           %[[DIM2:.*]] = llvm.mlir.constant(10 : i64) : i64
  // CHECK:           llvm.call @zdnn_init_pre_transformed_desc({{.*}}, {{.*}}, {{.*}}, %[[DIM0]], %[[DIM1]], %[[DIM2]]) vararg(!llvm.func<void (i64, i64, ptr, i64, i64, i64, ...)>) : (i64, i64, !llvm.ptr, i64, i64, i64) -> ()

}
