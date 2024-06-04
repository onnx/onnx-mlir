// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s

func.func @test_quantizeLinear(%arg0 : tensor<32x3x224x224xf32>) -> tensor<32x3x224x224xi8> {
  %0 = onnx.Constant dense<3.125000e-02> : tensor<f32>                       
  %1 = onnx.Constant dense<0> : tensor<i8>                                   
  %2 = "onnx.QuantizeLinear"(%arg0, %0, %1) {axis = 1 : si64} : (tensor<32x3x224x224xf32>, tensor<f32>, tensor<i8>) -> tensor<32x3x224x224xi8>
  "func.return"(%2) : (tensor<32x3x224x224xi8>) -> ()
}
// CHECK-LABEL:  @test_quantizeLinear
// CHECK-SAME: (%[[ARG_0:.*]]: tensor<32x3x224x224xf32>) -> tensor<32x3x224x224xi8>
// CHECK-DAG:    %[[ZP:.*]] = "tosa.const"() <{value = dense<0> : tensor<1x1x1x1xi8>}> : () -> tensor<1x1x1x1xi8>
// CHECK-DAG:    %[[SCALE:.*]] = "tosa.const"() <{value = dense<3.125000e-02> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
// CHECK-DAG:    %[[REC:.*]] = tosa.reciprocal %[[SCALE]] : (tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:    %[[MUL:.*]] = tosa.mul %[[ARG_0]], %[[REC]] {shift = 0 : i8} : (tensor<32x3x224x224xf32>, tensor<1x1x1x1xf32>) -> tensor<32x3x224x224xf32>
// CHECK-DAG:    %[[ZPCAST:.*]] = tosa.cast %[[ZP]] : (tensor<1x1x1x1xi8>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:    %[[ADD:.*]] = tosa.add %3, %[[ZPCAST]] : (tensor<32x3x224x224xf32>, tensor<1x1x1x1xf32>) -> tensor<32x3x224x224xf32>
// CHECK-DAG:    %[[CAST:.*]]  = tosa.cast %[[ADD]] : (tensor<32x3x224x224xf32>) -> tensor<32x3x224x224xi8>
// CHECK-DAG:    return %[[CAST]] : tensor<32x3x224x224xi8>

// -----

func.func @test_quantizeLinear_per_axis(%arg0: tensor<8x2xf32>) -> tensor<8x2xi8> {
  %0 = onnx.Constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  %1 = onnx.Constant dense<[0, 1]> : tensor<2xi8>
  %2 = "onnx.QuantizeLinear"(%arg0, %0, %1)
    {axis = 1 : si64,
     saturate = 1 : si64} : (tensor<8x2xf32>, tensor<2xf32>, tensor<2xi8>) -> tensor<8x2xi8>
  return %2 : tensor<8x2xi8>
}
// CHECK-LABEL:   func.func @test_quantizeLinear_per_axis(
// CHECK-SAME:                                            %[[VAL_0:.*]]: tensor<8x2xf32>) -> tensor<8x2xi8> {
// CHECK:           %[[VAL_1:.*]] = "tosa.const"() <{value = dense<{{\[\[}}0, 1]]> : tensor<1x2xi8>}> : () -> tensor<1x2xi8>
// CHECK:           %[[VAL_2:.*]] = "tosa.const"() <{value = dense<{{\[\[}}1.000000e+00, 2.000000e+00]]> : tensor<1x2xf32>}> : () -> tensor<1x2xf32>
// CHECK:           %[[REC:.*]] = tosa.reciprocal %[[VAL_2]] : (tensor<1x2xf32>) -> tensor<1x2xf32>
// CHECK:           %[[MUL:.*]] = tosa.mul %[[VAL_0]], %[[REC]] {shift = 0 : i8} : (tensor<8x2xf32>, tensor<1x2xf32>) -> tensor<8x2xf32>
// CHECK:           %[[ZPCAST:.*]] = tosa.cast %[[VAL_1]] : (tensor<1x2xi8>) -> tensor<1x2xf32>
// CHECK:           %[[ADD:.*]] = tosa.add %[[MUL]], %[[ZPCAST]] : (tensor<8x2xf32>, tensor<1x2xf32>) -> tensor<8x2xf32>
// CHECK:           %[[CAST:.*]] = tosa.cast %[[ADD]] : (tensor<8x2xf32>) -> tensor<8x2xi8>
// CHECK:           return %[[CAST]] : tensor<8x2xi8>
// CHECK:         }

// -----

func.func @test_quantizeLinear_negative_axis(%arg0: tensor<8x2xf32>) -> tensor<8x2xi8> {
  %0 = onnx.Constant dense<2.000000e+00> : tensor<8xf32>
  %1 = onnx.Constant dense<1> : tensor<8xi8>
  %2 = "onnx.QuantizeLinear"(%arg0, %0, %1)
    {axis = -2 : si64,
     saturate = 1 : si64} : (tensor<8x2xf32>, tensor<8xf32>, tensor<8xi8>) -> tensor<8x2xi8>
  return %2 : tensor<8x2xi8>
}
// CHECK-LABEL: test_quantizeLinear_negative_axis
// CHECK: "tosa.const"() {{.*}} : tensor<8x1xi8>

// -----

// The default `axis` is `1` when it's absent in ONNX, which conflicts
// with the allowed range of `axis` when the input has rank 1.
// See https://github.com/onnx/onnx/issues/6067
func.func @default_axis(%arg0 : tensor<32xf32>) -> tensor<32xi8> {
  %0 = onnx.Constant dense<3.125000e-02> : tensor<f32>
  %1 = onnx.Constant dense<0> : tensor<i8>
  %2 = "onnx.QuantizeLinear"(%arg0, %0, %1) {axis = 1 : si64} : (tensor<32xf32>, tensor<f32>, tensor<i8>) -> tensor<32xi8>
  return %2 : tensor<32xi8>
}

// CHECK-LABEL: default_axis
// CHECK-NOT: onnx.QuantizeLinear
