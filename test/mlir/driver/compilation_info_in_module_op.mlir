// RUN: onnx-mlir --EmitMLIR --printIR %s | FileCheck %s

module {
  func.func @main_graph(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %0 = "onnx.Relu"(%arg0) : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    onnx.Return %0 : tensor<?x?x?xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// CHECK: module attributes {{{.*}}"onnx-mlir.compile_options" = "--EmitMLIR --printIR {{.*}}", "onnx-mlir.op_stats" = "{\0A  \22func.func\22 : 1,\0A  \22func.return\22 : 1,\0A  \22onnx.Relu\22 : 1\0A}\0A"{{.*}}}
