// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s


func.func @test_slice_constant_default_steps(%arg0 : tensor<2x4xf32>) -> tensor<1x3xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, none) -> tensor<1x3xf32>
  "func.return"(%1) : (tensor<1x3xf32>) -> ()
// CHECK-LABEL: func @test_slice_constant_default_steps
// CHECK: %0 = tosa.slice %arg0 {size = array<i64: 1, 3>, start = array<i64: 1, 0>} : (tensor<2x4xf32>) -> tensor<1x3xf32>
}

func.func @test_slice_all_constant_negative(%arg0 : tensor<2x4xf32>) -> tensor<1x3xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, -1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[2, -1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x3xf32>
  "func.return"(%1) : (tensor<1x3xf32>) -> ()
// CHECK-LABEL: func @test_slice_all_constant_negative
// CHECK: %0 = tosa.slice %arg0 {size = array<i64: 1, 3>, start = array<i64: 1, 0>} : (tensor<2x4xf32>) -> tensor<1x3xf32>
}

func.func @test_slice_all_constant_end_outofbound(%arg0 : tensor<2x4xf32>) -> tensor<1x3xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[5, 3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x3xf32>
  "func.return"(%1) : (tensor<1x3xf32>) -> ()
// CHECK-LABEL: func @test_slice_all_constant_end_outofbound
// CHECK: %0 = tosa.slice %arg0 {size = array<i64: 1, 3>, start = array<i64: 1, 0>} : (tensor<2x4xf32>) -> tensor<1x3xf32>
}

// -----

func.func @slice_all_dynamic(%arg0: tensor<20x10x5xf32>,
                             %arg1: tensor<1xi64>,
                             %arg2: tensor<1xi64>,
                             %arg3: tensor<1xi64>,
                             %arg4: tensor<1xi64>)
                              -> tensor<20x9x5xf32> {
  %0 = "onnx.Slice"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<20x10x5xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<20x9x5xf32>
  return %0 : tensor<20x9x5xf32>
}

// -----

func.func @slice_default_axes(%arg0: tensor<20x10x5xf32>,
                             %arg1: tensor<3xi64>,
                             %arg2: tensor<3xi64>)
                              -> tensor<20x10x1xf32> {
  %0 = onnx.Constant dense<[0, 1, 2]> : tensor<3xi64>
  %1 = onnx.Constant dense<1> : tensor<3xi64>
  %2 = "onnx.Slice"(%arg0, %arg1, %arg2, %0, %1) {onnx_node_name = "onnx.Slice_0"} : (tensor<20x10x5xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<20x10x1xf32>
  return %2 : tensor<20x10x1xf32>
}
