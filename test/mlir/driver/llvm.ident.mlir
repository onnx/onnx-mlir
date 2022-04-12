// RUN: onnx-mlir --preserveBitcode %s -o %t
// RUN: llvm-dis %t.bc -o %t.ll
// RUN: cat %t.ll | FileCheck %s
// XFAIL: onnx-mlir-vendor

// CHECK: !llvm.ident = !{![[MD:[0-9]*]]}
// CHECK: ![[MD]] = !{!"onnx-mlir version 1.0.0 ({{.*}}/onnx-mlir{{.*}} {{[a-zA-Z0-9]+}}, {{.*}}/llvm-project{{.*}} {{[a-zA-Z0-9]+}})"}
module {
  func @main_graph(%arg0: tensor<1x1xf32>, %arg1: tensor<1x1xf32>) -> tensor<1x1xf32> {
    %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
    return %0 : tensor<1x1xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 1 : i32, signature = ""} : () -> ()
}
