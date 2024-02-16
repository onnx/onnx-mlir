// RUN: onnx-mlir --preserveLLVMIR %s -o %t
// RUN: cat %t.ll | FileCheck %s

// REQUIRES: system-windows
// CHECK: define dso_local dllexport ptr @run_main_graph_1
// CHECK: define dso_local dllexport ptr @run_main_graph_2
// CHECK: define dso_local dllexport ptr @omQueryEntryPoints
// CHECK: define dso_local dllexport ptr @omInputSignature
// CHECK: define dso_local dllexport ptr @omOutputSignature
module  {
  func.func @main_graph_1(%arg0: tensor<1x1xf32>) -> tensor<1x1xf32> {
    %0 = "onnx.Relu"(%arg0) : (tensor<1x1xf32>) -> tensor<1x1xf32>
    onnx.Return %0 : tensor<1x1xf32>
  }
  func.func @main_graph_2(%arg0: tensor<1x1xf32>) -> tensor<1x1xf32> {
    %0 = "onnx.Relu"(%arg0) : (tensor<1x1xf32>) -> tensor<1x1xf32>
    onnx.Return %0 : tensor<1x1xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph_1} : () -> ()
  "onnx.EntryPoint"() {func = @main_graph_2} : () -> ()
}
