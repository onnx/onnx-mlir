// RUN: onnx-mlir -Xopt --data-sections -v %s -o %t 2>&1 | FileCheck --check-prefix=OPT %s
// RUN: onnx-mlir -Xllc --data-sections -v %s -o %t 2>&1 | FileCheck --check-prefix=LLC %s
// RUN: onnx-mlir -mllvm --data-sections -v %s -o %t 2>&1 | FileCheck --check-prefix=LLVM %s

// OPT:       opt{{.*}} --data-sections -o {{.*}}check_passthrough_options{{.*}}.bc
// OPT-NOT:   llc{{.*}} --data-sections {{.*}}
// LLC-NOT:   opt{{.*}} --data-sections -o {{.*}}check_passthrough_options{{.*}}.bc
// LLC:       llc{{.*}} --data-sections {{.*}}
// LLVM:      opt{{.*}} --data-sections -o {{.*}}check_passthrough_options{{.*}}.bc
// LLVM-NEXT: llc{{.*}} --data-sections {{.*}}
module {
  func.func @main_graph(%arg0: tensor<1x1xf32>, %arg1: tensor<1x1xf32>) -> tensor<1x1xf32> {
    %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
    return %0 : tensor<1x1xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
