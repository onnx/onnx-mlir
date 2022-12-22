// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-mhlo --canonicalize %s -split-input-file | FileCheck %s

func.func @test_unsqueeze(%arg0 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[0, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Unsqueeze"(%arg0, %0) : (tensor<10x10xf32>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func @test_unsqueeze    
// CHECK-NEXT:         %0 = mhlo.reshape %arg0 : (tensor<10x10xf32>) -> tensor<1x10x10x1xf32>
// CHECK-NEXT:         return %0 : tensor<1x10x10x1xf32>
}

func.func @test_unsqueeze_negative_axis(%arg0 : tensor<16x32x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[-2]> : tensor<1xi64>} : () -> tensor<1xi64>
  %1 = "onnx.Unsqueeze"(%arg0, %0) : (tensor<16x32x64xf32>, tensor<1xi64>) -> (tensor<*xf32>)
  "func.return"(%1) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func @test_unsqueeze_negative_axis
// CHECK-NEXT:         %0 = mhlo.reshape %arg0 : (tensor<16x32x64xf32>) -> tensor<16x32x1x64xf32>
// CHECK-NEXT:         return %0 : tensor<16x32x1x64xf32>
}

func.func @test_unsqueeze_mix(%arg0 : tensor<16x32x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[1, -2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Unsqueeze"(%arg0, %0) : (tensor<16x32x64xf32>, tensor<2xi64>) -> (tensor<*xf32>)
  "func.return"(%1) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func @test_unsqueeze_mix
// CHECK-NEXT:         %0 = mhlo.reshape %arg0 : (tensor<16x32x64xf32>) -> tensor<16x1x32x1x64xf32>
// CHECK-NEXT:         return %0 : tensor<16x1x32x1x64xf32>
}