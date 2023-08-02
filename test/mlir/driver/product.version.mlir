// RUN: onnx-mlir --preserveBitcode -tag="test" %s -o %t
// RUN: llvm-dis %t.bc -o %t.ll
// RUN: cat %t.ll | FileCheck %s

// CHECK: !{{[0-9]+}} = !{i32 2, !"Product Major Version", i32 {{.*}}}
// CHECK: !{{[0-9]+}} = !{i32 2, !"Product Minor Version", i32 {{.*}}}
// CHECK: !{{[0-9]+}} = !{i32 2, !"Product Patchlevel", i32 {{.*}}}
// CHECK: !{{[0-9]+}} = !{i32 2, !"Product Id", !"{{.*}}"}
module {
  func.func @main_graph(%arg0: tensor<1x1xf32>, %arg1: tensor<1x1xf32>) -> tensor<1x1xf32> {
    %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
    onnx.Return %0 : tensor<1x1xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
