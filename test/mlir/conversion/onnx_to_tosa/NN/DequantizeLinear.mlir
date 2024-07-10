// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s

func.func @test_dequantizeLinear(%arg0 : tensor<32x3x224x224xi8>) -> tensor<32x3x224x224xf32> {
  %0 = onnx.Constant dense<3.125000e-02> : tensor<f32>                       
  %1 = onnx.Constant dense<0> : tensor<i8>                                   
  %2 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 1 : si64} : (tensor<32x3x224x224xi8>, tensor<f32>, tensor<i8>) -> tensor<32x3x224x224xf32>
  "func.return"(%2) : (tensor<32x3x224x224xf32>) -> ()
}
// CHECK-LABEL:  @test_dequantizeLinear
// CHECK-SAME:    (%[[ARG_0:.*]]: tensor<32x3x224x224xi8>) -> tensor<32x3x224x224xf32>
// CHECK-DAG:    %[[ZP:.*]] = "tosa.const"() <{value = dense<0> : tensor<1x1x1x1xi8>}> : () -> tensor<1x1x1x1xi8>
// CHECK-DAG:    %[[SCALE:.*]] = "tosa.const"() <{value = dense<3.125000e-02> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
// CHECK-DAG:    %[[CAST_0:.*]] = tosa.cast %[[ARG_0]] : (tensor<32x3x224x224xi8>) -> tensor<32x3x224x224xf32>
// CHECK-DAG:    %[[CASTZP:.*]] = tosa.cast %[[ZP]] : (tensor<1x1x1x1xi8>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:    %[[SUB:.*]] = tosa.sub %[[CAST_0]], %[[CASTZP]] : (tensor<32x3x224x224xf32>, tensor<1x1x1x1xf32>) -> tensor<32x3x224x224xf32>
// CHECK-DAG:    %[[MUL:.*]] = tosa.mul %[[SUB]], %[[SCALE]] {shift = 0 : i8} : (tensor<32x3x224x224xf32>, tensor<1x1x1x1xf32>) -> tensor<32x3x224x224xf32>
// CHECK-DAG:    return %[[MUL]] : tensor<32x3x224x224xf32>

// -----

func.func @test_dequantizeLinear_f16(%arg0 : tensor<32x3x224x224xi8>) -> tensor<32x3x224x224xf16> {
  %0 = onnx.Constant dense<3.125000e-02> : tensor<f16>                       
  %1 = onnx.Constant dense<0> : tensor<i8>                                   
  %2 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 1 : si64} : (tensor<32x3x224x224xi8>, tensor<f16>, tensor<i8>) -> tensor<32x3x224x224xf16>
  "func.return"(%2) : (tensor<32x3x224x224xf16>) -> ()
}

// CHECK-LABEL:  @test_dequantizeLinear_f16
// CHECK-SAME:    (%[[ARG_0:.*]]: tensor<32x3x224x224xi8>) -> tensor<32x3x224x224xf16>
// CHECK-DAG:    %[[ZP:.*]] = "tosa.const"() <{value = dense<0> : tensor<1x1x1x1xi8>}> : () -> tensor<1x1x1x1xi8>
// CHECK-DAG:    %[[SCALE:.*]] = "tosa.const"() <{value = dense<3.125000e-02> : tensor<1x1x1x1xf16>}> : () -> tensor<1x1x1x1xf16>
// CHECK-DAG:    %[[CAST_0:.*]] = tosa.cast %[[ARG_0]] : (tensor<32x3x224x224xi8>) -> tensor<32x3x224x224xf32>
// CHECK-DAG:    %[[CASTZP:.*]] = tosa.cast %[[ZP]] : (tensor<1x1x1x1xi8>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:    %[[SUB:.*]] = tosa.sub %[[CAST_0]], %[[CASTZP]] : (tensor<32x3x224x224xf32>, tensor<1x1x1x1xf32>) -> tensor<32x3x224x224xf32>
// CHECK-DAG:    %[[CASTSCALE:.*]] = tosa.cast %[[SCALE]] : (tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:    %[[MUL:.*]] = tosa.mul %[[SUB]], %[[CASTSCALE]] {shift = 0 : i8} : (tensor<32x3x224x224xf32>, tensor<1x1x1x1xf32>) -> tensor<32x3x224x224xf32>
// CHECK-DAG:    %[[CAST:.*]] = tosa.cast %[[MUL]] : (tensor<32x3x224x224xf32>) -> tensor<32x3x224x224xf16>
// CHECK-DAG:    return %[[CAST]] : tensor<32x3x224x224xf16>

// -----

func.func @per_axis(%arg0: tensor<8x2xi8>) -> tensor<8x2xf32> {
  %0 = onnx.Constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  %1 = onnx.Constant dense<[0, 1]> : tensor<2xi8>
  %2 = "onnx.DequantizeLinear"(%arg0, %0, %1)
    {axis = 1 : si64,
     saturate = 1 : si64} : (tensor<8x2xi8>, tensor<2xf32>, tensor<2xi8>) -> tensor<8x2xf32>
  return %2 : tensor<8x2xf32>
}

// -----

// The default `axis` is `1` when it's absent in ONNX, which conflicts
// with the allowed range of `axis` when the input has rank 1.
// See https://github.com/onnx/onnx/issues/6067
func.func @default_axis(%arg0 : tensor<32xi8>) -> tensor<32xf32> {
  %0 = onnx.Constant dense<3.125000e-02> : tensor<f32>
  %1 = onnx.Constant dense<0> : tensor<i8>
  %2 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 1 : si64} : (tensor<32xi8>, tensor<f32>, tensor<i8>) -> tensor<32xf32>
  return %2 : tensor<32xf32>
}

// CHECK-LABEL: default_axis
// CHECK-NOT: onnx.DequantizeLinear

// -----

func.func @no_zeropoint(%arg0: tensor<5xi8>, %arg1: tensor<f32>) -> tensor<5xf32>  {
  %0 = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.DequantizeLinear"(%arg0, %arg1, %0) {axis = 0 : si64} : (tensor<5xi8>, tensor<f32>, none) -> tensor<5xf32>
  return %1 : tensor<5xf32>
}

// CHECK-LABEL: @no_zeropoint(
// CHECK-SAME:                            %[[VAL_0:.*]]: tensor<5xi8>,
// CHECK-SAME:                            %[[VAL_1:.*]]: tensor<f32>) -> tensor<5xf32> {
// CHECK:           %[[VAL_2:.*]] = tosa.cast %[[VAL_0]] : (tensor<5xi8>) -> tensor<5xf32>
// CHECK:           %[[VAL_3:.*]] = tosa.reshape %[[VAL_1]] {new_shape = array<i64: 1>} : (tensor<f32>) -> tensor<1xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.mul %[[VAL_2]], %[[VAL_3]] {shift = 0 : i8} : (tensor<5xf32>, tensor<1xf32>) -> tensor<5xf32>
// CHECK:           return %[[VAL_4]] : tensor<5xf32>

// -----

func.func @f8E4M3FN(%arg0: tensor<5xf8E4M3FN>, %arg1: tensor<f32>) -> tensor<5xf32>  {
  %0 = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.DequantizeLinear"(%arg0, %arg1, %0) {axis = 0 : si64} : (tensor<5xf8E4M3FN>, tensor<f32>, none) -> tensor<5xf32>
  return %1 : tensor<5xf32>
}

// CHECK-LABEL: @f8E4M3FN
// CHECK-SAME:                        %[[VAL_0:.*]]: tensor<5xf8E4M3FN>,
// CHECK-SAME:                        %[[VAL_1:.*]]: tensor<f32>) -> tensor<5xf32> {
// CHECK:           %[[VAL_2:.*]] = tosa.cast %[[VAL_0]] : (tensor<5xf8E4M3FN>) -> tensor<5xf32>
// CHECK:           %[[VAL_3:.*]] = tosa.reshape %[[VAL_1]] {new_shape = array<i64: 1>} : (tensor<f32>) -> tensor<1xf32>
// CHECK:           %[[VAL_4:.*]] = tosa.mul %[[VAL_2]], %[[VAL_3]] {shift = 0 : i8} : (tensor<5xf32>, tensor<1xf32>) -> tensor<5xf32>
// CHECK:           return %[[VAL_4]] : tensor<5xf32>
