
// RUN: onnx-mlir --EmitMLIR  --preserveLocations --printIR %s |  FileCheck %s ; rm %p/*.onnx.mlir ; rm %p/*.tmp

module {
  func @main_graph(%arg0: tensor<1x16xf32>, %arg1: tensor<1x16xf32>, %arg2: tensor<1x16xf32>) -> tensor<1x16xf32> attributes {input_names = ["X", "Y", "U"], output_names = ["Z"]} {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<1x16xf32>, tensor<1x16xf32>) -> tensor<1x16xf32> loc("/build/workspace/addop.onnx":1:0)
    %1 = "onnx.Add"(%0, %arg2) : (tensor<1x16xf32>, tensor<1x16xf32>) -> tensor<1x16xf32> loc("/build/workspace/addop.onnx":2:0)
    return %1 : tensor<1x16xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 3 : i32, numOutputs = 1 : i32} : () -> ()

// CHECK: loc("{{(/[[:alnum:]]+)+}}.onnx":1:0)
}

// RUN: onnx-mlir --EmitMLIR  %s | FileCheck %s ; rm %p/*.onnx.mlir ; rm %p/*.tmp

module {
  func @main_graph(%arg0: tensor<1x16xf32>, %arg1: tensor<1x16xf32>, %arg2: tensor<1x16xf32>) -> tensor<1x16xf32> attributes {input_names = ["X", "Y", "U"], output_names = ["Z"]} {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<1x16xf32>, tensor<1x16xf32>) -> tensor<1x16xf32> loc("/build/workspace/addop.onnx":1:0)
    %1 = "onnx.Add"(%0, %arg2) : (tensor<1x16xf32>, tensor<1x16xf32>) -> tensor<1x16xf32> loc("/build/workspace/addop.onnx":2:0)
    return %1 : tensor<1x16xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 3 : i32, numOutputs = 1 : i32} : () -> ()

// CHECK-NOT: loc("{{(/[[:alnum:]]+)+}}.onnx":1:0)
}
