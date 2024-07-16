
// RUN: onnx-mlir --mcpu=z16 --maccel=NNPA -v -tag="test" %s -o %t 2>&1 | FileCheck %s

// -----

// REQUIRES: system-linux
module {
  func.func @main_graph(%arg0: tensor<1x1xf32>, %arg1: tensor<1x1xf32>) -> tensor<1x1xf32> {
    %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
    onnx.Return %0 : tensor<1x1xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
// CHECK: {{.*}} opt {{.*}} -o {{.*}}.bc
// CHECK: {{.*}} llc {{.*}}  {{.*}} {{.*}}.bc
// CHECK: {{.*}} {{clang|c|g}}++{{.*}} {{.*}}.o -o {{.*}}.so -shared -fPIC -L{{.*}}/lib -lRuntimeNNPA -lzdnn -lcruntime
