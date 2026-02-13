// RUN: onnx-mlir-opt -maccel=NNPA -march=z17 --convert-onnx-to-zhigh -split-input-file %s | FileCheck %s

// COM: This is to test that small ONNX operations are not lowered to ZHigh for NNPA.

func.func @test_salar_sqrt(%arg0 : tensor<1xf32>) -> tensor<1xf32> {
  %x = "onnx.Sqrt"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
  "func.return"(%x) : (tensor<1xf32>) -> ()

// CHECK-LABEL:  func.func @test_salar_sqrt
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1xf32>) -> tensor<1xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Sqrt"([[PARAM_0_]]) : (tensor<1xf32>) -> tensor<1xf32>
// CHECK:           return [[VAR_0_]] : tensor<1xf32>
// CHECK:         }
}

// -----

func.func @test_scalar_invsqrt(%arg0 : tensor<1xf32>) -> tensor<1xf32> {
  %a = onnx.Constant dense<[1.0]> : tensor<1xf32>
  %x = "onnx.Sqrt"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
  %y = "onnx.Div"(%a, %x) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  "func.return"(%y) : (tensor<1xf32>) -> ()

// CHECK-LABEL:  func.func @test_scalar_invsqrt
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1xf32>) -> tensor<1xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Sqrt"([[PARAM_0_]]) : (tensor<1xf32>) -> tensor<1xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Div"([[VAR_0_]], [[VAR_1_]]) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
// CHECK:           return [[VAR_2_]] : tensor<1xf32>
// CHECK:         }
}

// -----

func.func @test_multiple_scalar_ops(%arg0 : tensor<1xf32>) -> tensor<1xf32> {
  %a = onnx.Constant dense<[1.0]> : tensor<1xf32>
  %x = "onnx.Sqrt"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
  %y = "onnx.Div"(%a, %x) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %z = "onnx.Sqrt"(%y) : (tensor<1xf32>) -> tensor<1xf32>
  "func.return"(%z) : (tensor<1xf32>) -> ()

// CHECK-LABEL:  func.func @test_multiple_scalar_ops
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1xf32>) -> tensor<1xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Sqrt"([[PARAM_0_]]) : (tensor<1xf32>) -> tensor<1xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Div"([[VAR_0_]], [[VAR_1_]]) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Sqrt"([[VAR_2_]]) : (tensor<1xf32>) -> tensor<1xf32>
// CHECK:           return [[VAR_3_]] : tensor<1xf32>
// CHECK:         }
}

// -----

func.func @test_scalar_add_1(%arg0 : tensor<f32>, %arg1 : tensor<f32>) -> tensor<f32> {
  %x = "onnx.Add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  "func.return"(%x) : (tensor<f32>) -> ()

// CHECK-LABEL:  func.func @test_scalar_add_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<f32>, [[PARAM_1_:%.+]]: tensor<f32>) -> tensor<f32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:           return [[VAR_0_]] : tensor<f32>
// CHECK:         }
}

// -----

func.func @test_scalar_add_2(%arg0 : tensor<1xf32>, %arg1 : tensor<1xf32>) -> tensor<1xf32> {
  %x = "onnx.Add"(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  "func.return"(%x) : (tensor<1xf32>) -> ()

// CHECK-LABEL:  func.func @test_scalar_add_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1xf32>, [[PARAM_1_:%.+]]: tensor<1xf32>) -> tensor<1xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Add"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
// CHECK:           return [[VAR_0_]] : tensor<1xf32>
// CHECK:         }
}
