// RUN: onnx-mlir --instrument-onnx-node-return="add1:out0" --instrument-onnx-node="add1:out0" --EmitONNXIR --printIR %s | FileCheck %s

// Basic case: --instrument-onnx-node-return appends the selected output of
// "add1" as an extra function result, named "__instrumented__add1__out0"
// and returned alongside the model's real output. Combined here with
// --instrument-onnx-node on the same node/selector to confirm the two flags
// are orthogonal: the print op is still inserted too.

module {
  func.func @main_graph(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> (tensor<3xf32> {onnx.name = "z"}) {
    %0 = "onnx.Add"(%arg0, %arg1) {onnx_node_name = "add1"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    %1 = "onnx.Neg"(%0) {onnx_node_name = "neg1"} : (tensor<3xf32>) -> tensor<3xf32>
    onnx.Return %1 : tensor<3xf32>
  }
  "onnx.EntryPoint"() <{func = @main_graph}> : () -> ()

// CHECK-LABEL:  func.func @main_graph
// CHECK-SAME:   ({{.*}}, {{.*}}) -> (tensor<3xf32> {onnx.name = "z"}, tensor<3xf32> {onnx.name = "__instrumented__add1__out0"}) {
// CHECK:        [[VAR_0_:%.+]] = "onnx.Add"
// CHECK:        "onnx.PrintSignature"([[VAR_0_]]) <{{{.*}}op_name = "onnx.Add, add1"{{.*}}}>
// CHECK:        [[VAR_1_:%.+]] = "onnx.Neg"([[VAR_0_]])
// CHECK:        return [[VAR_1_]], [[VAR_0_]] : tensor<3xf32>, tensor<3xf32>
// CHECK:      }
}
