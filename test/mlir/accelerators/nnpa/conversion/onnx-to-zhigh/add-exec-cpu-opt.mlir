// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --shape-inference --convert-onnx-to-zhigh %s | FileCheck %s

func.func @test_add_force_cpu_opt(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) {device = "cpu", onnx_node_name = "test/add0"} : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  %1 = "onnx.Add"(%0, %arg0) {onnx_node_name = "test/add1"} : (tensor<*xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  %2 = "onnx.Add"(%1, %arg1) {device = "cpu", onnx_node_name = "test/add2"} : (tensor<*xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "func.return"(%2) : (tensor<*xf32>) -> ()

  // CHECK-LABEL:  func @test_add_force_cpu_opt
  // CHECK:           "onnx.Add"({{.*}}, {{.*}}) {device = "cpu", onnx_node_name = "test/add0"}
  // CHECK:           "zhigh.Stick"
  // CHECK:           "zhigh.Stick"
  // CHECK:           "zhigh.Add"
  // CHECK:           "zhigh.Unstick"
  // CHECK:           "onnx.Add"({{.*}}, {{.*}}) {device = "cpu", onnx_node_name = "test/add2"}
  // CHECK:           return
}

