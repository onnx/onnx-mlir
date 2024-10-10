// RUN: onnx-mlir %s -o %t 2>&1 | FileCheck %s && rm %t.so

// CHECK: [1/6] {{.*}} Importing ONNX Model to MLIR Module from
// CHECK: [2/6] {{.*}} Compiling and Optimizing MLIR Module
// CHECK: [3/6] {{.*}} Translating MLIR Module to LLVM and Generating LLVM Optimized Bitcode
// CHECK: [4/6] {{.*}} Generating Object from LLVM Bitcode
// CHECK: [5/6] {{.*}} Linking and Generating the Output Shared Library
// CHECK: [6/6] {{.*}} Compilation completed
module {
  func.func @main_graph(%arg0: tensor<?xf32>) -> tensor<?xf32> {
    onnx.Return %arg0 : tensor<?xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
