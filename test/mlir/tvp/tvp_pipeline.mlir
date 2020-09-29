// RUN: onnx-mlir --EmitMLIR %s -o %t 
// RUN: FileCheck %s --input-file %t.onnx.mlir
// XFAIL: *

module {
// CHECK: error: failed to legalize operation 'tvp.module'
// Although we add code to execute lowering to LLVM on tvp.module, the pipeline is not yet setup
// to fully handle that.
  tvp.module @kernels {
    func @tvp_0(%arg0: tensor<784xf32>, %0: tensor<f32>) -> tensor<784xf32> {
       %1 = "onnx.Add"(%arg0, %0) : (tensor<784xf32>, tensor<f32>) -> tensor<784xf32>
       return %1 : tensor<784xf32>
    }
  }
  func @main_graph(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    return %arg0 : tensor<1024xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = -4 : i32, numOutputs = 1 : i32} : () -> ()
}