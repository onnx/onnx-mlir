// RUN: onnx-mlir --instrument-onnx-node="add1,onnx.Mul_*,add2:in1+out0" --EmitONNXIR --printIR %s | FileCheck %s

// Three --instrument-onnx-node matching styles in one flag value:
//  - "add1": a fully-specified (exact, no wildcard) node name -- matches
//    only that node and prints all of its inputs and outputs.
//  - "onnx.Mul_*": a glob/regexp pattern -- matches "onnx.Mul_7" and prints
//    all of its inputs and outputs.
//  - "add2:in1+out0": an exact node name with an explicit selector --
//    prints only input 1 and output 0 of "add2", not input 0.

module {
  func.func @main_graph(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>, %arg2: tensor<3xf32>) -> (tensor<3xf32> {onnx.name = "z"}) {
    %0 = "onnx.Add"(%arg0, %arg1) {onnx_node_name = "add1"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    %1 = "onnx.Mul"(%0, %arg2) {onnx_node_name = "onnx.Mul_7"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    %2 = "onnx.Add"(%1, %arg2) {onnx_node_name = "add2"} : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    onnx.Return %2 : tensor<3xf32>
  }
  "onnx.EntryPoint"() <{func = @main_graph}> : () -> ()

// CHECK-LABEL:  func.func @main_graph
// CHECK-SAME:   ({{.*}}) -> (tensor<3xf32> {onnx.name = "z"}) {
// CHECK:        [[VAR_0_:%.+]] = "onnx.Add"({{.*}}) {onnx_node_name = "add1"}
// CHECK:        "onnx.PrintSignature"({{.*}}, {{.*}}, [[VAR_0_]]) <{io_labels = ["in0", "in1", "out0"], op_name = "onnx.Add, add1", print_data = 1 : si64}>
// CHECK:        [[VAR_1_:%.+]] = "onnx.Mul"([[VAR_0_]], {{.*}}) {onnx_node_name = "onnx.Mul_7"}
// CHECK:        "onnx.PrintSignature"([[VAR_0_]], {{.*}}, [[VAR_1_]]) <{io_labels = ["in0", "in1", "out0"], op_name = "onnx.Mul, onnx.Mul_7", print_data = 1 : si64}>
// CHECK:        [[VAR_2_:%.+]] = "onnx.Add"([[VAR_1_]], {{.*}}) {onnx_node_name = "add2"}
// CHECK:        "onnx.PrintSignature"({{.*}}, [[VAR_2_]]) <{io_labels = ["in1", "out0"], op_name = "onnx.Add, add2", print_data = 1 : si64}>
// CHECK:        return [[VAR_2_]] : tensor<3xf32>
// CHECK:      }
}
