// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s

func.func @test_onnx_to_matmul2d(%arg0 : tensor<4x8xf32>, %arg1 : tensor<8x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK: func.func @test_onnx_to_matmul2d(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>) -> tensor<4x16xf32>
  // CHECK:  %0 = tosa.reshape %arg0 {new_shape = array<i64: 1, 4, 8>} : (tensor<4x8xf32>) -> tensor<1x4x8xf32>
  // CHECK:  %1 = tosa.reshape %arg1 {new_shape = array<i64: 1, 8, 16>} : (tensor<8x16xf32>) -> tensor<1x8x16xf32>
  // CHECK:  %2 = tosa.matmul %0, %1 : (tensor<1x4x8xf32>, tensor<1x8x16xf32>) -> tensor<1x4x16xf32>
  // CHECK:  %3 = tosa.reshape %2 {new_shape = array<i64: 4, 16>} : (tensor<1x4x16xf32>) -> tensor<4x16xf32>
  // CHECK:  return %3 : tensor<4x16xf32>
}

// -----

func.func @test_onnx_to_matmul3dbcast(%arg0 : tensor<100x4x8xf32>, %arg1 : tensor<8x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<100x4x8xf32>, tensor<8x16xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK: func.func @test_onnx_to_matmul3dbcast(%arg0: tensor<100x4x8xf32>, %arg1: tensor<8x16xf32>) -> tensor<100x4x16xf32> {
  // CHECK:   %0 = tosa.reshape %arg1 {new_shape = array<i64: 1, 8, 16>} : (tensor<8x16xf32>) -> tensor<1x8x16xf32>
  // CHECK:   %1 = tosa.reshape %arg0 {new_shape = array<i64: 1, 400, 8>} : (tensor<100x4x8xf32>) -> tensor<1x400x8xf32>
  // CHECK:   %2 = "tosa.const"() <{value = dense<[1, 0, 2]> : tensor<3xi32>}> : () -> tensor<3xi32>
  // CHECK:   %3 = tosa.transpose %0, %2 : (tensor<1x8x16xf32>, tensor<3xi32>) -> tensor<8x1x16xf32>
  // CHECK:   %4 = tosa.reshape %3 {new_shape = array<i64: 1, 8, 16>} : (tensor<8x1x16xf32>) -> tensor<1x8x16xf32>
  // CHECK:   %5 = tosa.matmul %1, %4 : (tensor<1x400x8xf32>, tensor<1x8x16xf32>) -> tensor<1x400x16xf32>
  // CHECK:   %6 = tosa.reshape %5 {new_shape = array<i64: 100, 4, 16>} : (tensor<1x400x16xf32>) -> tensor<100x4x16xf32>
  // CHECK:   return %6 : tensor<100x4x16xf32>
}

// -----

func.func @test_onnx_1d(%arg0 : tensor<6xf32>, %arg1 : tensor<6xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<6xf32>, tensor<6xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK: func.func @test_onnx_1d(%arg0: tensor<6xf32>, %arg1: tensor<6xf32>) -> tensor<f32> {
  // CHECK:   %0 = tosa.reshape %arg0 {new_shape = array<i64: 1, 6>} : (tensor<6xf32>) -> tensor<1x6xf32>
  // CHECK:   %1 = tosa.reshape %arg1 {new_shape = array<i64: 6, 1>} : (tensor<6xf32>) -> tensor<6x1xf32>
  // CHECK:   %2 = tosa.reshape %0 {new_shape = array<i64: 1, 1, 6>} : (tensor<1x6xf32>) -> tensor<1x1x6xf32>
  // CHECK:   %3 = tosa.reshape %1 {new_shape = array<i64: 1, 6, 1>} : (tensor<6x1xf32>) -> tensor<1x6x1xf32>
  // CHECK:   %4 = tosa.matmul %2, %3 : (tensor<1x1x6xf32>, tensor<1x6x1xf32>) -> tensor<1x1x1xf32>
  // CHECK:   %5 = tosa.reshape %4 {new_shape = array<i64>} : (tensor<1x1x1xf32>) -> tensor<f32>
  // CHECK:   return %5 : tensor<f32>
}

// -----

func.func @test_onnx_12d(%arg0 : tensor<6xf32>, %arg1 : tensor<6x1xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<6xf32>, tensor<6x1xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK: func.func @test_onnx_12d(%arg0: tensor<6xf32>, %arg1: tensor<6x1xf32>) -> tensor<1xf32> {
  // CHECK:   %0 = tosa.reshape %arg0 {new_shape = array<i64: 1, 6>} : (tensor<6xf32>) -> tensor<1x6xf32>
  // CHECK:   %1 = tosa.reshape %0 {new_shape = array<i64: 1, 1, 6>} : (tensor<1x6xf32>) -> tensor<1x1x6xf32>
  // CHECK:   %2 = tosa.reshape %arg1 {new_shape = array<i64: 1, 6, 1>} : (tensor<6x1xf32>) -> tensor<1x6x1xf32>
  // CHECK:   %3 = tosa.matmul %1, %2 : (tensor<1x1x6xf32>, tensor<1x6x1xf32>) -> tensor<1x1x1xf32>
  // CHECK:   %4 = tosa.reshape %3 {new_shape = array<i64: 1>} : (tensor<1x1x1xf32>) -> tensor<1xf32>
  // CHECK:   return %4 : tensor<1xf32>
}

// -----

func.func @test_onnx_21d(%arg0 : tensor<2x6xf32>, %arg1 : tensor<6xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<2x6xf32>, tensor<6xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK: func.func @test_onnx_21d(%arg0: tensor<2x6xf32>, %arg1: tensor<6xf32>) -> tensor<2xf32> {
  // CHECK:   %0 = tosa.reshape %arg1 {new_shape = array<i64: 6, 1>} : (tensor<6xf32>) -> tensor<6x1xf32>
  // CHECK:   %1 = tosa.reshape %arg0 {new_shape = array<i64: 1, 2, 6>} : (tensor<2x6xf32>) -> tensor<1x2x6xf32>
  // CHECK:   %2 = tosa.reshape %0 {new_shape = array<i64: 1, 6, 1>} : (tensor<6x1xf32>) -> tensor<1x6x1xf32>
  // CHECK:   %3 = tosa.matmul %1, %2 : (tensor<1x2x6xf32>, tensor<1x6x1xf32>) -> tensor<1x2x1xf32>
  // CHECK:   %4 = tosa.reshape %3 {new_shape = array<i64: 2>} : (tensor<1x2x1xf32>) -> tensor<2xf32>
  // CHECK:   return %4 : tensor<2xf32>
}

// -----

func.func @test_onnx_4d(%arg0 : tensor<10x10x6x2xf32>, %arg1 : tensor<10x10x2x6xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<10x10x6x2xf32>, tensor<10x10x2x6xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK: func.func @test_onnx_4d(%arg0: tensor<10x10x6x2xf32>, %arg1: tensor<10x10x2x6xf32>) -> tensor<10x10x6x6xf32> {
  // CHECK:   %0 = tosa.reshape %arg0 {new_shape = array<i64: 100, 6, 2>} : (tensor<10x10x6x2xf32>) -> tensor<100x6x2xf32>
  // CHECK:   %1 = tosa.reshape %arg1 {new_shape = array<i64: 100, 2, 6>} : (tensor<10x10x2x6xf32>) -> tensor<100x2x6xf32>
  // CHECK:   %2 = tosa.matmul %0, %1 : (tensor<100x6x2xf32>, tensor<100x2x6xf32>) -> tensor<100x6x6xf32>
  // CHECK:   %3 = tosa.reshape %2 {new_shape = array<i64: 10, 10, 6, 6>} : (tensor<100x6x6xf32>) -> tensor<10x10x6x6xf32>
  // CHECK:   return %3 : tensor<10x10x6x6xf32>
}

// -----

func.func @test_onnx_4d_mixed(%arg0 : tensor<10x6x2xf32>, %arg1 : tensor<10x10x2x6xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<10x6x2xf32>, tensor<10x10x2x6xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK: func.func @test_onnx_4d_mixed(%arg0: tensor<10x6x2xf32>, %arg1: tensor<10x10x2x6xf32>) -> tensor<10x10x6x6xf32> {
  // CHECK:   %0 = tosa.reshape %arg0 {new_shape = array<i64: 1, 10, 6, 2>} : (tensor<10x6x2xf32>) -> tensor<1x10x6x2xf32>
  // CHECK:   %1 = "tosa.const"() <{value = dense<[1, 0, 2, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
  // CHECK:   %2 = tosa.transpose %0, %1 : (tensor<1x10x6x2xf32>, tensor<4xi32>) -> tensor<10x1x6x2xf32>
  // CHECK:   %3 = tosa.reshape %2 {new_shape = array<i64: 10, 6, 2>} : (tensor<10x1x6x2xf32>) -> tensor<10x6x2xf32>
  // CHECK:   %4 = "tosa.const"() <{value = dense<[1, 2, 0, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
  // CHECK:   %5 = tosa.transpose %arg1, %4 : (tensor<10x10x2x6xf32>, tensor<4xi32>) -> tensor<10x2x10x6xf32>
  // CHECK:   %6 = tosa.reshape %5 {new_shape = array<i64: 10, 2, 60>} : (tensor<10x2x10x6xf32>) -> tensor<10x2x60xf32>
  // CHECK:   %7 = tosa.matmul %3, %6 : (tensor<10x6x2xf32>, tensor<10x2x60xf32>) -> tensor<10x6x60xf32>
  // CHECK:   %8 = tosa.reshape %7 {new_shape = array<i64: 10, 6, 10, 6>} : (tensor<10x6x60xf32>) -> tensor<10x6x10x6xf32>
  // CHECK:   %9 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
  // CHECK:   %10 = tosa.transpose %8, %9 : (tensor<10x6x10x6xf32>, tensor<4xi32>) -> tensor<10x10x6x6xf32>
  // CHECK:   return %10 : tensor<10x10x6x6xf32>
}

// -----

func.func @test_onnx_to_matmul4d_non_broadcastable(%arg0 : tensor<4x1x5x6xf32>, %arg1 : tensor<1x3x6x7xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<4x1x5x6xf32>, tensor<1x3x6x7xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK: func.func @test_onnx_to_matmul4d_non_broadcastable(%arg0: tensor<4x1x5x6xf32>, %arg1: tensor<1x3x6x7xf32>) -> tensor<4x3x5x7xf32> {
  // CHECK:   %0 = tosa.reshape %arg0 {new_shape = array<i64: 1, 20, 6>} : (tensor<4x1x5x6xf32>) -> tensor<1x20x6xf32>
  // CHECK:   %1 = "tosa.const"() <{value = dense<[2, 0, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
  // CHECK:   %2 = tosa.transpose %arg1, %1 : (tensor<1x3x6x7xf32>, tensor<4xi32>) -> tensor<6x1x3x7xf32>
  // CHECK:   %3 = tosa.reshape %2 {new_shape = array<i64: 1, 6, 21>} : (tensor<6x1x3x7xf32>) -> tensor<1x6x21xf32>
  // CHECK:   %4 = tosa.matmul %0, %3 : (tensor<1x20x6xf32>, tensor<1x6x21xf32>) -> tensor<1x20x21xf32>
  // CHECK:   %5 = tosa.reshape %4 {new_shape = array<i64: 4, 5, 3, 7>} : (tensor<1x20x21xf32>) -> tensor<4x5x3x7xf32>
  // CHECK:   %6 = "tosa.const"() <{value = dense<[0, 2, 1, 3]> : tensor<4xi32>}> : () -> tensor<4xi32>
  // CHECK:   %7 = tosa.transpose %5, %6 : (tensor<4x5x3x7xf32>, tensor<4xi32>) -> tensor<4x3x5x7xf32>
  // CHECK:   return %7 : tensor<4x3x5x7xf32>
}

// -----

func.func @test_onnx_to_matmul3d_fp16(%arg0 : tensor<100x4x8xf16>, %arg1 : tensor<100x8x16xf16>) -> tensor<*xf16> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<100x4x8xf16>, tensor<100x8x16xf16>) -> tensor<*xf16>
  "func.return"(%0) : (tensor<*xf16>) -> ()
  // CHECK:  %0 = tosa.matmul %arg0, %arg1 : (tensor<100x4x8xf16>, tensor<100x8x16xf16>) -> tensor<100x4x16xf32>
  // CHECK:  %1 = tosa.cast %0 : (tensor<100x4x16xf32>) -> tensor<100x4x16xf16>
  // CHECK:  return %1 : tensor<100x4x16xf16>
}

// -----

func.func @test_onnx_to_matmul3d_bf16(%arg0 : tensor<100x4x8xbf16>, %arg1 : tensor<100x8x16xbf16>) -> tensor<*xbf16> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<100x4x8xbf16>, tensor<100x8x16xbf16>) -> tensor<*xbf16>
  "func.return"(%0) : (tensor<*xbf16>) -> ()
  // CHECK:   %0 = tosa.matmul %arg0, %arg1 : (tensor<100x4x8xbf16>, tensor<100x8x16xbf16>) -> tensor<100x4x16xf32>
  // CHECK:   %1 = tosa.cast %0 : (tensor<100x4x16xf32>) -> tensor<100x4x16xbf16>
  // CHECK:   return %1 : tensor<100x4x16xbf16>
}

// -----

func.func @test_onnx_to_matmul3d_fp32(%arg0 : tensor<100x4x8xf32>, %arg1 : tensor<100x8x16xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<100x4x8xf32>, tensor<100x8x16xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK:   %0 = tosa.matmul %arg0, %arg1 : (tensor<100x4x8xf32>, tensor<100x8x16xf32>) -> tensor<100x4x16xf32>
  // CHECK:   return %0 : tensor<100x4x16xf32>
}

// -----

func.func @test_onnx_to_matmul2d_dyn(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK-NOT: tosa.matmul
}

// -----

func.func @test_onnx_to_matmul3d_dyn(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x?x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK-NOT: tosa.matmul
}
