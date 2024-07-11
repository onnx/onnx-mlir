// RUN: onnx-mlir %s | FileCheck %s

// CHECK: [1/5] {{.*}} Importing ONNX Model to MLIR Module
// CHECK: [2/5] {{.*}} Compiling and Optimizing MLIR Module
// CHECK: [3/5] {{.*}} Translating MLIR Module to LLVM and Generating LLVM Optimized Bitcode
// CHECK: [4/5] {{.*}} Generating Object from LLVM Bitcode
// CHECK: [5/5] {{.*}} Linking and Generating the Output Shared Library
module {
  func.func @main_graph(%arg0: tensor<?xf32>) -> tensor<?xf32> {
    onnx.Return %arg0 : tensor<?xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
