// RUN: onnx-mlir --instrument-onnx-node-return="neg1:out0,inner_add" --EmitONNXIR --printIR %s | FileCheck %s

// Two cases where --instrument-onnx-node-return must NOT add a new output:
//  - "neg1:out0"'s selected result is already the function's only return
//    value (dedup by Value identity).
//  - "inner_add" only exists inside an onnx.Loop body, so it is never found
//    by the pass's top-level-only scan (returning it would violate SSA
//    dominance) -- neither the function signature nor the terminator change.

module {
  func.func @main_graph(%arg0: tensor<3xf32>) -> (tensor<3xf32> {onnx.name = "z"}) {
    %0 = "onnx.Neg"(%arg0) {onnx_node_name = "neg1"} : (tensor<3xf32>) -> tensor<3xf32>
    onnx.Return %0 : tensor<3xf32>
  }
  "onnx.EntryPoint"() <{func = @main_graph}> : () -> ()

  func.func @main_graph2(%arg0: tensor<i64>, %arg1: tensor<i1>, %arg2: tensor<1xi64>) -> tensor<1xi64> {
    %0 = "onnx.Loop"(%arg0, %arg1, %arg2) ({
    ^bb0(%i: tensor<i64>, %cond: tensor<i1>, %y: tensor<1xi64>):
      %1 = "onnx.Add"(%y, %i) {onnx_node_name = "inner_add"} : (tensor<1xi64>, tensor<i64>) -> tensor<1xi64>
      onnx.Yield %cond, %1 : tensor<i1>, tensor<1xi64>
    }) : (tensor<i64>, tensor<i1>, tensor<1xi64>) -> tensor<1xi64>
    onnx.Return %0 : tensor<1xi64>
  }
  "onnx.EntryPoint"() <{func = @main_graph2}> : () -> ()

// CHECK-LABEL:  func.func @main_graph
// CHECK-SAME:   ({{.*}}) -> (tensor<3xf32> {onnx.name = "z"}) {
// CHECK:        [[VAR_0_:%.+]] = "onnx.Neg"
// CHECK-NEXT:   return [[VAR_0_]] : tensor<3xf32>
// CHECK:        }

// CHECK-LABEL:  func.func @main_graph2
// CHECK-SAME:   ({{.*}}) -> tensor<1xi64> {
// CHECK:        [[VAR_0_:%.+]] = "onnx.Loop"
// CHECK:        return [[VAR_0_]] : tensor<1xi64>
// CHECK:        }
}
