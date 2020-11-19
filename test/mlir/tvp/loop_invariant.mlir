// RUN: onnx-mlir --EmitMLIR %s -o %t 
// RUN: FileCheck %s --input-file %t.onnx.mlir

module {
  // CHECK-LABEL: func @main_graph
  // CHECK: %cst = constant 0.00392156886 : f32
  // CHECK: affine.for
  // CHECK: affine.for
  func @main_graph(%arg0: tensor<?x784xf32>) -> tensor<?x784xf32> attributes {input_names = ["X"], output_names = ["predictions"]} {
    %0 = "onnx.Constant"() {value = dense<0.00392156886> : tensor<f32>} : () -> tensor<f32>
    %1 = "onnx.Mul"(%arg0, %0) {onnx_node_name = "mul0"} : (tensor<?x784xf32>, tensor<f32>) -> tensor<?x784xf32>
    return %1 : tensor<?x784xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 1 : i32, numOutputs = 1 : i32} : () -> ()
}
