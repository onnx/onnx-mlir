// RUN: onnx-mlir -tag="test" -v %s -o %t 2>&1 | FileCheck %s

// REQUIRES: system-linux
// CHECK:      opt {{.*}} -o {{.*}}check_verbosity{{.*}}.bc
// CHECK:      llc {{.*}} -filetype=obj {{.*}} -o {{.*}}check_verbosity{{.*}}.o {{.*}}check_verbosity{{.*}}.bc
// CHECK:      {{clang|c|g}}++{{.*}} {{.*}}check_verbosity{{.*}}.o -o {{.*}}check_verbosity{{.*}}.so -shared -fPIC -L{{.*}}/lib -lcruntime
module {
  func.func @main_graph(%arg0: tensor<1x1xf32>, %arg1: tensor<1x1xf32>) -> tensor<1x1xf32> {
    %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
    onnx.Return %0 : tensor<1x1xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
