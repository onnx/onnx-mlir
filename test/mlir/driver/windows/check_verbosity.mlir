// RUN: onnx-mlir -v %s 2>&1 |  FileCheck %s

// REQUIRES: system-windows
// CHECK:      opt{{(\.exe)?}} {{.*}} -o {{.*}}check_verbosity.bc
// CHECK-NEXT: llc{{(\.exe)?}} {{.*}} -filetype=obj {{.*}} -o {{.*}}check_verbosity.obj {{.*}}check_verbosity.bc
// CHECK-NEXT: cl{{(\.exe)?}} {{.*}}check_verbosity.obj /Fe:{{.*}}check_verbosity.dll {{.*}}
module  {
  func @main_graph(%arg0: tensor<1x1xf32>, %arg1: tensor<1x1xf32>) -> tensor<1x1xf32> {
    %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
    return %0 : tensor<1x1xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 1 : i32, signature = ""} : () -> ()
}
