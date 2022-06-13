// RUN: onnx-mlir --preserveLLVMIR %s -o %t
// RUN: cat %t.ll | FileCheck %s

// REQUIRES: system-windows
// CHECK: define dso_local dllexport i8* @run_main_graph_1
// CHECK: define dso_local dllexport i8* @run_main_graph_2
// CHECK: define dso_local dllexport i8** @omQueryEntryPoints
// CHECK: define dso_local dllexport i8* @omInputSignature
// CHECK: define dso_local dllexport i8* @omOutputSignature
module  {
  func @main_graph_1(%arg0: tensor<1x1xf32>) -> tensor<1x1xf32> {
    %0 = "onnx.Relu"(%arg0) : (tensor<1x1xf32>) -> tensor<1x1xf32>
    return %0 : tensor<1x1xf32>
  }
  func @main_graph_2(%arg0: tensor<1x1xf32>) -> tensor<1x1xf32> {
    %0 = "onnx.Relu"(%arg0) : (tensor<1x1xf32>) -> tensor<1x1xf32>
    return %0 : tensor<1x1xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph_1, numInputs = 1 : i32, numOutputs = 1 : i32, signature = ""} : () -> ()
  "onnx.EntryPoint"() {func = @main_graph_2, numInputs = 1 : i32, numOutputs = 1 : i32, signature = ""} : () -> ()
}
