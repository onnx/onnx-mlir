// RUN: onnx-mlir -tag="test" -v %s -o %t 2>&1 | FileCheck %s

// REQUIRES: system-windows
// CHECK:      opt{{(\.exe)?}} {{.*}} -o {{.*}}check_verbosity{{.*}}.bc
// CHECK-NEXT: llc{{(\.exe)?}} {{.*}} -filetype=obj {{.*}} -o {{.*}}check_verbosity{{.*}}.obj {{.*}}check_verbosity{{.*}}.bc
// CHECK-NEXT: cl{{(\.exe)?}} {{.*}}check_verbosity{{.*}}.obj /Fe:{{.*}}check_verbosity{{.*}}.dll {{.*}} cruntime.lib
module {
  func.func @main_graph(%arg0: tensor<1x1xf32>, %arg1: tensor<1x1xf32>) -> tensor<1x1xf32> {
    %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
    onnx.Return %0 : tensor<1x1xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
