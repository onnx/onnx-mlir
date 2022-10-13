// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa %s -split-input-file | FileCheck %s

func.func @test_pad(%arg0: tensor<20x16x44x32xf32>) -> tensor<20x16x48x36xf32> {
    %0 = "onnx.Constant"() {value = dense<[0, 0, 2, 2, 0, 0, 2, 2]> : tensor<8xi64>} : () -> tensor<8xi64> 
    %1 = "onnx.Constant"() {value = dense<[0.0000]> : tensor<1xf32>} : () -> tensor<1xf32> 
    %2 = "onnx.Pad"(%arg0, %0, %1) {mode = "constant"} : (tensor<20x16x44x32xf32>, tensor<8xi64>, tensor<1xf32>) -> tensor<20x16x48x36xf32> 
    return %2 : tensor<20x16x48x36xf32>
// CHECK-LABEL: test_pad
// CHECK: %[[VAR1:.*]] = "tosa.pad"(%arg0, %[[VAR0:.*]], %[[PVAL:.*]])
}
