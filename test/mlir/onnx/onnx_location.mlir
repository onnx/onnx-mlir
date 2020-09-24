
// RUN: onnx-mlir --EmitMLIR  --preserveLocations --printIR %s |  FileCheck %s --check-prefix=PRESENT; rm %p/*.onnx.mlir ; rm %p/*.tmp
// RUN: onnx-mlir --EmitMLIR  --printIR %s | FileCheck %s --check-prefix=ABSENT; rm %p/*.onnx.mlir ; rm %p/*.tmp

  func @main_graph(%arg0: tensor<1x16xf32>, %arg1: tensor<1x16xf32>) -> tensor<1x16xf32>  {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<1x16xf32>, tensor<1x16xf32>) -> tensor<1x16xf32> loc("/build/workspace/addop.onnx":1:0)
     return %0 : tensor<1x16xf32>
  }

// PRESENT: loc("{{(/[[:alnum:]]+)+}}.onnx":1:0)
// ABSENT-NOT: loc("{{(/[[:alnum:]]+)+}}.onnx":1:0)
