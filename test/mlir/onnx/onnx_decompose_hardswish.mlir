// RUN: onnx-mlir-opt --decompose-onnx --decompose-op-in-onnx HardSwish %s | FileCheck %s
func.func @test_hardswish(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "onnx.HardSwish"(%arg0) {onnx_node_name = "/hardswish/HardSwish"} :
       (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  onnx.Return %0 : tensor<?x?x?xf32>

  // CHECK-LABEL:       func @test_hardswish
  // CHECK-NOT: "onnx.HardSwish"
  // CHECK-SAME:     (%[[ARG0:.*]]: {{.*}})
  // CHECK-NEXT: %[[C1:.*]] = onnx.Constant dense<0.166666672> : tensor<1xf32>
  // CHECK-NEXT: %[[C2:.*]] = onnx.Constant dense<5.000000e-01> : tensor<1xf32>
  // CHECK-NEXT: %[[C3:.*]] = onnx.Constant dense<1.000000e+00> : tensor<1xf32>
  // CHECK-NEXT: %[[C4:.*]] = onnx.Constant dense<0.000000e+00> : tensor<1xf32>
  // CHECK-NEXT: %[[MUL1:.*]] = "onnx.Mul"(%[[ARG0]], %[[C1]]) : (tensor<?x?x?xf32>, tensor<1xf32>) -> tensor<?x?x?xf32>
  // CHECK-NEXT: %[[ADD:.*]] = "onnx.Add"(%[[MUL1]], %[[C2]]) : (tensor<?x?x?xf32>, tensor<1xf32>) -> tensor<?x?x?xf32>
  // CHECK-NEXT: %[[MIN:.*]] = "onnx.Min"(%[[ADD]], %[[C3]]) : (tensor<?x?x?xf32>, tensor<1xf32>) -> tensor<?x?x?xf32>
  // CHECK-NEXT: %[[MAX:.*]] = "onnx.Max"(%[[MIN]], %[[C4]]) : (tensor<?x?x?xf32>, tensor<1xf32>) -> tensor<?x?x?xf32>
  // CHECK-NEXT: %[[MUL2:.*]] = "onnx.Mul"(%[[ARG0]], %[[MAX]]) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  // CHECK-NEXT: onnx.Return %[[MUL2]] : tensor<?x?x?xf32>
}