// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s

func.func @test_dequantizeLinear(%arg0 : tensor<32x3x224x224xi8>) -> tensor<32x3x224x224xf32> {
  %0 = onnx.Constant dense<3.125000e-02> : tensor<f32>                       
  %1 = onnx.Constant dense<0> : tensor<i8>                                   
  %2 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 1 : si64} : (tensor<32x3x224x224xi8>, tensor<f32>, tensor<i8>) -> tensor<32x3x224x224xf32>
  "func.return"(%2) : (tensor<32x3x224x224xf32>) -> ()
}
// CHECK-LABEL:  @test_dequantizeLinear(%arg0: tensor<32x3x224x224xi8>) -> tensor<32x3x224x224xf32>
// CHECK-DAG:    %[[ZP:.*]] = "tosa.const"() <{value = dense<0> : tensor<1x1x1x1xi8>}> : () -> tensor<1x1x1x1xi8>
// CHECK-DAG:    %[[SCALE:.*]] = "tosa.const"() <{value = dense<3.125000e-02> : tensor<1x1x1x1xf32>}> : () -> tensor<1x1x1x1xf32>
// CHECK-DAG:    %[[SUB:.*]] = tosa.sub %arg0, %[[ZP]] : (tensor<32x3x224x224xi8>, tensor<1x1x1x1xi8>) -> tensor<32x3x224x224xi8>
// CHECK-DAG:    %[[CAST:.*]] = tosa.cast %[[SUB]] : (tensor<32x3x224x224xi8>) -> tensor<32x3x224x224xf32>
// CHECK-DAG:    %[[MUL:.*]] = tosa.mul %[[CAST]], %[[SCALE]] {shift = 0 : i8} : (tensor<32x3x224x224xf32>, tensor<1x1x1x1xf32>) -> tensor<32x3x224x224xf32>
// CHECK-DAG:    return %[[MUL]] : tensor<32x3x224x224xf32>

// -----

func.func @per_axis(%arg0: tensor<8x2xi8>) -> tensor<8x2xf32> {
  %0 = onnx.Constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
  %1 = onnx.Constant dense<[0, 1]> : tensor<2xi8>
  %2 = "onnx.DequantizeLinear"(%arg0, %0, %1)
    {axis = 1 : si64,
     saturate = 1 : si64} : (tensor<8x2xi8>, tensor<2xf32>, tensor<2xi8>) -> tensor<8x2xf32>
  return %2 : tensor<8x2xf32>
}
// CHECK-LABEL:   func.func @per_axis(
// CHECK-SAME:                        %[[VAL_0:.*]]: tensor<8x2xi8>) -> tensor<8x2xf32> {
// CHECK:           %[[VAL_1:.*]] = "tosa.const"() <{value = dense<{{\[\[}}0, 1]]> : tensor<1x2xi8>}> : () -> tensor<1x2xi8>
// CHECK:           %[[VAL_2:.*]] = "tosa.const"() <{value = dense<{{\[\[}}1.000000e+00, 2.000000e+00]]> : tensor<1x2xf32>}> : () -> tensor<1x2xf32>
// CHECK:           %[[VAL_3:.*]] = tosa.sub %[[VAL_0]], %[[VAL_1]] : (tensor<8x2xi8>, tensor<1x2xi8>) -> tensor<8x2xi8>
// CHECK:           %[[VAL_4:.*]] = tosa.cast %[[VAL_3]] : (tensor<8x2xi8>) -> tensor<8x2xf32>
// CHECK:           %[[VAL_5:.*]] = tosa.mul %[[VAL_4]], %[[VAL_2]] {shift = 0 : i8} : (tensor<8x2xf32>, tensor<1x2xf32>) -> tensor<8x2xf32>
// CHECK:           return %[[VAL_5]] : tensor<8x2xf32>
// CHECK:         }
