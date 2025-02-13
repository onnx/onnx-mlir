// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --shape-inference --convert-onnx-to-zhigh %s -split-input-file | FileCheck %s

// COM:  Do not lower element-wise ops with scalar tensor since it is not benefical. 
func.func @test_not_lowered_scalar_tensor(%arg0 : tensor<f32>, %arg1 : tensor<f32>, %arg2: tensor<2xf32>) -> tensor<*xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<*xf32>
  %1 = "onnx.Add"(%arg2, %arg2) : (tensor<2xf32>, tensor<2xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_not_lowered_scalar_tensor 
}
