// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s

/// Test the default behavior of AveragePool with no padding
func.func @test_dequantizeLinear(%arg0 : tensor<32x3x224x224xi8>) -> tensor<32x3x224x224xf32> {
  %0 = onnx.Constant dense<3.125000e-02> : tensor<f32>                       
  %1 = onnx.Constant dense<0> : tensor<i8>                                   
  %2 = "onnx.DequantizeLinear"(%arg0, %0, %1) {axis = 1 : si64} : (tensor<32x3x224x224xi8>, tensor<f32>, tensor<i8>) -> tensor<32x3x224x224xf32>
  "func.return"(%2) : (tensor<32x3x224x224xf32>) -> ()
}
// CHECK-LABEL:  @test_dequantizeLinear
// CHECK-DAG:    %[[SCALE:.*]] = "tosa.const"() {value = dense<3.125000e-02> : tensor<f32>}
// CHECK-DAG:    %[[ZP:.*]] = "tosa.const"() {value = dense<0> : tensor<i8>}
// CHECK-DAG:    %[[CAST:.*]] = "tosa.cast"(%arg0)
// CHECK-DAG:    %[[SUB:.*]] = "tosa.sub"(%[[CAST]], %[[ZP]])
// CHECK-DAG:    %[[MUL:.*]] = "tosa.mul"(%[[SUB]], %[[SCALE]]) {shift = 0 : i32}
// CHECK-DAG:    return %4 : tensor<32x3x224x224xf32>