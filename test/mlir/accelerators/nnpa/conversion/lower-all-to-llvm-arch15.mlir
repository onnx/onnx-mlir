// RUN: onnx-mlir-opt --march=arch15 --maccel=NNPA --convert-krnl-to-llvm %s -split-input-file | FileCheck %s

// -----

// Check stickification with saturation.
func.func @test_stick_with_saturation() -> () {
  %0 = memref.alloc() : memref<10x10xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  "zlow.stick"(%0, %1) {saturation = -1 : si64} : (memref<10x10xf32>, memref<1x1x32x64xf16>) -> ()
  return

  // CHECK-LABEL: test_stick_with_saturation
  // CHECK: llvm.call @zdnn_transform_ztensor_with_saturation({{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr) -> i32
}

// -----

// Check stickification without saturation.
func.func @test_stick_without_saturation() -> () {
  %0 = memref.alloc() : memref<10x10xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  "zlow.stick"(%0, %1) {saturation = 0 : si64} : (memref<10x10xf32>, memref<1x1x32x64xf16>) -> ()
  return

  // CHECK-LABEL: test_stick_without_saturation
  // CHECK: llvm.call @zdnn_transform_ztensor({{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr) -> i32
}

// -----

// Check whether the lowering of zlow.gelu calls the correct zDNN API or not.
func.func @test_call_zdnn_gelu() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf16>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.gelu"(%0, %shape, %1) {layout = "2D"} : (memref<1x1x32x64xf16>, memref<2xi64>, memref<1x1x32x64xf16>) -> ()
  return

  // CHECK-LABEL: test_call_zdnn_gelu
  // CHECK: {{.*}} = llvm.call @zdnn_gelu_ext({{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr) -> i32
}

// -----

// Check whether the lowering of zlow.leakyrelu calls the correct zDNN API or not. 
func.func @test_call_zdnn_leaky_relu() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf16>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.leakyrelu"(%0, %shape, %1) {layout = "2D"} : (memref<1x1x32x64xf16>, memref<2xi64>, memref<1x1x32x64xf16>) -> ()
  return

  // CHECK-LABEL: test_call_zdnn_leaky_relu
  // CHECK: {{.*}} = llvm.call @zdnn_leaky_relu_ext({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr, f32, !llvm.ptr) -> i32
}

// -----

// Check whether the lowering of zlow.invsqrt calls the correct zDNN API or not.
func.func @test_call_zdnn_invsqrt() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf16>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  %shape = memref.alloc() :  memref<2xi64>
  "zlow.invsqrt"(%0, %shape, %1) {layout = "2D"} : (memref<1x1x32x64xf16>, memref<2xi64>, memref<1x1x32x64xf16>) -> ()
  return

  // CHECK-LABEL: test_call_zdnn_invsqrt
  // CHECK: {{.*}} = llvm.call @zdnn_invsqrt_ext({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr, f32, !llvm.ptr) -> i32
}

// -----

// Check whether the lowering of zlow.reducemax calls the correct zDNN API or not.
func.func @test_call_zdnn_reducemax() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf16>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  %work_area = memref.alloc() {alignment = 4096 : i64} : memref<8192xi8>
  %shape = memref.alloc() : memref<i64>
  "zlow.reducemax"(%0, %work_area, %shape, %1) {layout = "2D", op_type = "REDUCE_OP_MAXIMUM" : i64} : (memref<1x1x32x64xf16>,  memref<8192xi8>, memref<i64>, memref<1x1x32x64xf16>) -> ()
  return

  // CHECK-LABEL: test_call_zdnn_reducemax
  // CHECK: {{.*}} = llvm.call @zdnn_reduce_ext({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32
}

// -----

// Check whether the lowering of zlow.reducemin calls the correct zDNN API or not.
func.func @test_call_zdnn_reducemin() -> () {
  %0 = memref.alloc() : memref<3x2x32x64xf16>
  %1 = memref.alloc() : memref<3x2x32x64xf16>
  %work_area = memref.alloc() {alignment = 4096 : i64} : memref<8192xi8>
  %shape = memref.alloc() : memref<i64>
  "zlow.reducemin"(%0, %work_area, %shape, %1) {layout = "2D", op_type = "REDUCE_OP_MINIMUM" : i64} : (memref<3x2x32x64xf16>,  memref<8192xi8>, memref<i64>, memref<3x2x32x64xf16>) -> ()
  return

  // CHECK-LABEL: test_call_zdnn_reducemin
  // CHECK: {{.*}} = llvm.call @zdnn_reduce_ext({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32
}

// -----

// Check whether the lowering of zlow.sqrt calls the correct zDNN API or not.
func.func @test_call_zdnn_sqrt() -> () {
  %0 = memref.alloc() {alignment = 4096 : i64} : memref<2048xf16>
  %1 = memref.alloc() {alignment = 4096 : i64} : memref<2048xf16>
  %shape = memref.alloc() : memref<4xi64>
  "zlow.sqrt"(%0, %shape, %1) {layout = "2D"} : (memref<2048xf16>, memref<4xi64>, memref<2048xf16>) -> ()
  return

  // CHECK-LABEL: test_call_zdnn_sqrt
  // CHECK: {{.*}} = llvm.call @zdnn_sqrt_ext({{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr) -> i32
}

// -----

// Check whether the lowering of zlow.matmul calls the correct zDNN API or not.
func.func @test_matmul_bcast1(%x: memref<2048xf16>,%y: memref<2048xf16>,%bias: memref<2048xf16>, %shape: memref<3xi64>) -> memref<2048xf16> {
  %res = memref.alloc() {alignment = 4096 : i64} : memref<2048xf16>
  "zlow.matmul"(%x, %y, %bias, %shape, %res) {is_bcast1 = -1 : si64, is_bcast23 = 0 : si64, is_stacked = 0 : si64} : (memref<2048xf16>, memref<2048xf16>, memref<2048xf16>, memref<3xi64>, memref<2048xf16>) -> ()
  return %res : memref<2048xf16>
  // CHECK-LABEL: test_matmul_bcast1
  // CHECK: %{{.*}} = llvm.call @zdnn_matmul_bcast_op_ext(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32
}

// -----

// Check whether the lowering of zlow.quantized_matmul calls the correct zDNN API or not.
func.func @test_call_zdnn_quantized_matmul_op(%arg0: memref<1x1x1x1x32x64xf16>, %arg1: memref<f32>, %arg2: memref<f32>, %arg3: memref<1x1x1x1x32x64xi8>, %arg4: memref<f32>, %arg5: memref<f32>, %arg6: memref<1x1x1x1x32x64xi8>, %arg7: memref<f32>, %arg8: memref<f32>, %arg9: memref<1x1x1x1x32x64xf16>, %arg10: memref<4xi64>, %arg11: memref<f32>, %arg12: memref<f32>) -> memref<1x1x1x1x32x64xf16> {
  %alloc = memref.alloc() {alignment = 4096 : i64} : memref<1x1x1x1x32x64xf16>
  "zlow.quantizedMatmul"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %alloc, %arg11, %arg12) {bias_q_type = "INT8", dequantize_output = 0 : si64, is_bcast = -1 : si64, is_stacked = 0 : si64, out_q_type = "DLFLOAT16", x_q_type = "DLFLOAT16", y_q_type = "WEIGHTS"} : (memref<1x1x1x1x32x64xf16>, memref<f32>, memref<f32>, memref<1x1x1x1x32x64xi8>, memref<f32>, memref<f32>, memref<1x1x1x1x32x64xi8>, memref<f32>, memref<f32>, memref<1x1x1x1x32x64xf16>, memref<4xi64>, memref<1x1x1x1x32x64xf16>, memref<f32>, memref<f32>) -> ()
  return %alloc : memref<1x1x1x1x32x64xf16>

  // CHECK-LABEL: test_call_zdnn_quantized_matmul_op
  // CHECK: {{.*}} = llvm.call @zdnn_quantized_matmul_op({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr) -> i32
}

// -----

// Check whether the lowering of zlow.quantized_matmul calls the correct zDNN API or not.
func.func @test_call_zdnn_quantized_matmul_dequantized_op(%arg0: memref<1x1x1x1x32x64xf16>, %arg1: memref<f32>, %arg2: memref<f32>, %arg3: memref<1x1x1x1x32x64xi8>, %arg4: memref<f32>, %arg5: memref<f32>, %arg6: memref<1x1x1x1x32x64xi8>, %arg7: memref<f32>, %arg8: memref<f32>, %arg9: memref<1x1x1x1x32x64xf16>, %arg10: memref<4xi64>, %arg11: memref<f32>, %arg12: memref<f32>) -> memref<1x1x1x1x32x64xf16> {
  %alloc = memref.alloc() {alignment = 4096 : i64} : memref<1x1x1x1x32x64xf16>
  "zlow.quantizedMatmul"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %alloc, %arg11, %arg12) {bias_q_type = "INT8", dequantize_output = -1 : si64, is_bcast = -1 : si64, is_stacked = 0 : si64, out_q_type = "DLFLOAT16", x_q_type = "DLFLOAT16", y_q_type = "WEIGHTS"} : (memref<1x1x1x1x32x64xf16>, memref<f32>, memref<f32>, memref<1x1x1x1x32x64xi8>, memref<f32>, memref<f32>, memref<1x1x1x1x32x64xi8>, memref<f32>, memref<f32>, memref<1x1x1x1x32x64xf16>, memref<4xi64>, memref<1x1x1x1x32x64xf16>, memref<f32>, memref<f32>) -> ()
  return %alloc : memref<1x1x1x1x32x64xf16>

  // CHECK-LABEL: test_call_zdnn_quantized_matmul_dequantized_op
  // CHECK: {{.*}} = llvm.call @zdnn_quantized_matmul_op({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr) -> i32
}
