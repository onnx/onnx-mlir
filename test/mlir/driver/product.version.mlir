// RUN: onnx-mlir --preserveBitcode %s -o %t
// RUN: llvm-dis %t.bc -o %t.ll
// RUN: cat %t.ll | FileCheck %s
// XFAIL: onnx-mlir-product-version

// CHECK: !{{[0-9]+}} = !{i32 2, !"Product Major Version", i32 0}
// CHECK: !{{[0-9]+}} = !{i32 2, !"Product Minor Version", i32 0}
// CHECK: !{{[0-9]+}} = !{i32 2, !"Product Patchlevel", i32 0}
// CHECK: !{{[0-9]+}} = !{i32 2, !"Product Id", !"NOT_SPECIFIED"}
module {
  func @main_graph(%arg0: tensor<1x1xf32>, %arg1: tensor<1x1xf32>) -> tensor<1x1xf32> {
    %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
    return %0 : tensor<1x1xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 1 : i32, signature = ""} : () -> ()
}
