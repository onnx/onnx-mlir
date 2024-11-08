// RUN: onnx-mlir --EmitMLIR --do-not-emit-full-mlir-code %s -o %t && test ! -f %t.onnx.mlir && rm %t.tmp

module {
  func.func @main_graph(%arg0: tensor<?xf32>) -> tensor<?xf32> {
    onnx.Return %arg0 : tensor<?xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

