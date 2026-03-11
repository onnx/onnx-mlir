// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa=disable-cast-lowering=true -cse %s -split-input-file | FileCheck %s

func.func @test_cast_f32_i8(%arg0: tensor<13x21x1xf32>) -> tensor<13x21x1xi8> {
  %0 = "onnx.Cast"(%arg0) {to = i8} : (tensor<13x21x1xf32>) -> tensor<13x21x1xi8>
  "func.return"(%0) : (tensor<13x21x1xi8>) -> ()
// CHECK-LABEL:   func.func @test_cast_f32_i8
// CHECK: onnx.Cast
}

// -----

func.func @test_cast_int4_and_uint4_to_from_int8_uint8(%arg0: tensor<1xi4>, %arg1: tensor<1xui4>) -> (tensor<1xi4>, tensor<1xui4>) {
    %0 = "onnx.Cast"(%arg0) {saturate = 1 : si64, to = i8} : (tensor<1xi4>) -> tensor<1xi8>
    %1 = "onnx.Cast"(%0) {saturate = 1 : si64, to = i4} : (tensor<1xi8>) -> tensor<1xi4>
    %2 = "onnx.Cast"(%arg1) {saturate = 1 : si64, to = ui8} : (tensor<1xui4>) -> tensor<1xui8>
    %3 = "onnx.Cast"(%2) {saturate = 1 : si64, to = ui4} : (tensor<1xui8>) -> tensor<1xui4>
    onnx.Return %1, %3 : tensor<1xi4>, tensor<1xui4>
// CHECK-LABEL:  func.func @test_cast_int4_and_uint4_to_from_int8_uint8
// CHECK: onnx.Cast
// CHECK: onnx.Cast
// CHECK: onnx.Cast
// CHECK: onnx.Cast
}

// -----

func.func @test_cast_int4_and_uint4_to_float_and_back(%arg0: tensor<1xi4>, %arg1: tensor<1xui4>) -> (tensor<1xi4>, tensor<1xui4>) {
    %0 = "onnx.Cast"(%arg0) {saturate = 1 : si64, to = f32} : (tensor<1xi4>) -> tensor<1xf32>
    %1 = "onnx.Cast"(%0) {saturate = 1 : si64, to = i4} : (tensor<1xf32>) -> tensor<1xi4>
    %2 = "onnx.Cast"(%arg1) {saturate = 1 : si64, to = f32} : (tensor<1xui4>) -> tensor<1xf32>
    %3 = "onnx.Cast"(%2) {saturate = 1 : si64, to = ui4} : (tensor<1xf32>) -> tensor<1xui4>
    onnx.Return %1, %3 : tensor<1xi4>, tensor<1xui4>
// CHECK-LABEL:  func.func @test_cast_int4_and_uint4_to_float_and_back
// CHECK: onnx.Cast
// CHECK: onnx.Cast
// CHECK: onnx.Cast
// CHECK: onnx.Cast
}

// -----

func.func @test_cast_f16_i8(%arg0: tensor<13x21x1xf16>) -> tensor<13x21x1xi8> {
  %0 = "onnx.Cast"(%arg0) {to = i8} : (tensor<13x21x1xf16>) -> tensor<13x21x1xi8>
  "func.return"(%0) : (tensor<13x21x1xi8>) -> ()
// CHECK: onnx.Cast
}

// -----

func.func @test_cast_i8_i1(%arg0: tensor<1x21x1x1xi8>) -> tensor<1x21x1x1xi1> {
  %0 = "onnx.Cast"(%arg0) {to = i1} : (tensor<1x21x1x1xi8>) -> tensor<1x21x1x1xi1>
  "func.return"(%0) : (tensor<1x21x1x1xi1>) -> ()
// CHECK-LABEL:  func @test_cast_i8_i1
// CHECK: onnx.Cast
}

// -----

func.func @test_cast_f32_i1(%arg0: tensor<13x21x1xf32>) -> tensor<13x21x1xi1> {
  %0 = "onnx.Cast"(%arg0) {to = i1} : (tensor<13x21x1xf32>) -> tensor<13x21x1xi1>
  "func.return"(%0) : (tensor<13x21x1xi1>) -> ()
// CHECK-LABEL: func @test_cast_f32_i1
// CHECK: onnx.Cast
}
