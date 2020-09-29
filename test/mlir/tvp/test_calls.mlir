// RUN: onnx-mlir --EmitMLIR %s -o %t 
// RUN: FileCheck %s --input-file %t.onnx.mlir

module {
  func @tvp_1(%3: tensor<1024xf32>) -> tensor<1024xf32> {
    return %3 : tensor<1024xf32>
  }
  func @main_graph(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    // CHECK: %1 = call @tvp_1(%arg0) : (memref<1024xf32>) -> memref<1024xf32>
    %1 = std.call @tvp_1(%arg0) : (tensor<1024xf32>) -> tensor<1024xf32>
    "std.return"(%1) : (tensor<1024xf32>) -> ()
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = -4 : i32, numOutputs = 1 : i32} : () -> ()
}
