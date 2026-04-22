// RUN: onnx-mlir --EmitLLVMIR --printIR %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %0 = "onnx.Relu"(%arg0) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    onnx.Return %0 : tensor<?x?x?xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// CHECK: llvm.mlir.global internal constant @om_compilation_info_json{{.*}}("{{.*}}compile_options{{.*}}op_stats{{.*}}onnx.Relu{{.*}}")

// CHECK: llvm.func @omCompilationInfo{{.*}}() -> !llvm.ptr
// CHECK:   {{.*}} = llvm.mlir.addressof @om_compilation_info_json
// CHECK:   llvm.return {{.*}} : !llvm.ptr
