
// RUN: onnx-mlir --EmitMLIR  --preserveLocations %s | FileCheck %s

module {
  func @main_graph(%arg0: tensor<1x16xf32>, %arg1: tensor<1x16xf32>, %arg2: tensor<1x16xf32>) -> tensor<1x16xf32> attributes {input_names = ["X", "Y", "U"], output_names = ["Z"]} {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<1x16xf32>, tensor<1x16xf32>) -> tensor<1x16xf32> loc("/build/workspace/addop.onnx":1:0)
    %1 = "onnx.Add"(%0, %arg2) : (tensor<1x16xf32>, tensor<1x16xf32>) -> tensor<1x16xf32> loc("/build/workspace/addop.onnx":2:0)
    return %1 : tensor<1x16xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 3 : i32, numOutputs = 1 : i32} : () -> ()

// CHECK: loc("/build/workspace/addop.onnx":1:0)
// CHECK: loc("/build/workspace/addop.onnx":2:0)
// CHECK: loc("addop.onnx.mlir":7:5)
// CHECK: loc("addop.onnx.mlir":4:3)
// CHECK: loc("addop.onnx.mlir":9:3)
}

// RUN: onnx-mlir --EmitMLIR  %s | FileCheck %s

module {
  func @main_graph(%arg0: tensor<1x16xf32>, %arg1: tensor<1x16xf32>, %arg2: tensor<1x16xf32>) -> tensor<1x16xf32> attributes {input_names = ["X", "Y", "U"], output_names = ["Z"]} {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<1x16xf32>, tensor<1x16xf32>) -> tensor<1x16xf32> loc("/build/workspace/addop.onnx":1:0)
    %1 = "onnx.Add"(%0, %arg2) : (tensor<1x16xf32>, tensor<1x16xf32>) -> tensor<1x16xf32> loc("/build/workspace/addop.onnx":2:0)
    return %1 : tensor<1x16xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 3 : i32, numOutputs = 1 : i32} : () -> ()

// CHECK-NOT: loc("/build/workspace/addop.onnx":1:0)
// CHECK-NOT: loc("/build/workspace/addop.onnx":2:0)
// CHECK-NOT: loc("addop.onnx.mlir":7:5)
// CHECK-NOT: loc("addop.onnx.mlir":4:3)
// CHECK-NOT: loc("addop.onnx.mlir":9:3)
}
