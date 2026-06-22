// RUN: onnx-mlir --omit-compile-info --EmitLLVMIR --printIR %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %0 = "onnx.Relu"(%arg0) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    onnx.Return %0 : tensor<?x?x?xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()

// CHECK:     om_compilation_info_json{{.*}}("{}\00")
// CHECK:     llvm.func @omCompilationInfo
}

