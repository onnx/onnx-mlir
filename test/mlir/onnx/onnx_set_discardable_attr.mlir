// RUN: onnx-mlir-opt --convert-attr-to-discardable --attr-names=onnx_node_name --cse %s -split-input-file | FileCheck %s

// COM: Test that CSE can eliminate duplicate operations after converting to discardable attributes
// COM: Two Sqrt operations with same input but different onnx_node_name should be merged by CSE

func.func @test_cse_with_discardable_attr(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Sqrt"(%arg0) {onnx_node_name = "Sqrt_1"} : (tensor<10x10xf32>) -> tensor<10x10xf32>
  %1 = "onnx.Sqrt"(%arg0) {onnx_node_name = "Sqrt_2"} : (tensor<10x10xf32>) -> tensor<10x10xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  onnx.Return %2 : tensor<10x10xf32>

  // CHECK-LABEL: func @test_cse_with_discardable_attr
  // CHECK:       [[VAR_0_:%.+]] = "onnx.Sqrt"(%arg0)
  // CHECK-NOT:   "onnx.Sqrt"
  // CHECK:       [[VAR_1_:%.+]] = "onnx.Add"([[VAR_0_]], [[VAR_0_]])
}

// -----

// RUN: onnx-mlir-opt --convert-attr-to-discardable --attr-names=onnx_node_name --cse %s -split-input-file | FileCheck %s --check-prefix=NOCSE

// COM: Test that CSE does NOT eliminate operations with non-discardable attributes
// COM: Two Sqrt operations with different unknown_attr should NOT be merged by CSE
// COM: because unknown_attr is not in the attr-names list and remains non-discardable

func.func @test_no_cse_with_non_discardable_attr(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Sqrt"(%arg0) {unknown_attr = "Sqrt_1"} : (tensor<10x10xf32>) -> tensor<10x10xf32>
  %1 = "onnx.Sqrt"(%arg0) {unknown_attr = "Sqrt_2"} : (tensor<10x10xf32>) -> tensor<10x10xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  onnx.Return %2 : tensor<10x10xf32>

  // NOCSE-LABEL: func @test_no_cse_with_non_discardable_attr
  // NOCSE:       [[VAR_0_:%.+]] = "onnx.Sqrt"(%arg0) {unknown_attr = "Sqrt_1"}
  // NOCSE:       [[VAR_1_:%.+]] = "onnx.Sqrt"(%arg0) {unknown_attr = "Sqrt_2"}
  // NOCSE:       [[VAR_2_:%.+]] = "onnx.Add"([[VAR_0_]], [[VAR_1_]])
}
