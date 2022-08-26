// RUN: onnx-mlir --preserveLocations --printIR %s |  FileCheck %s --check-prefix=PRESENT
// RUN: onnx-mlir --printIR %s | FileCheck %s --check-prefix=ABSENT

  func.func @main_graph(%arg0: tensor<1x16xf32>, %arg1: tensor<1x16xf32>) -> tensor<1x16xf32>  {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<1x16xf32>, tensor<1x16xf32>) -> tensor<1x16xf32> loc("/build/workspace/addop.onnx":1:0)
     return %0 : tensor<1x16xf32>
  }

// PRESENT: loc("onnx.Add"("{{(/[[:alnum:]]+)+}}.onnx":1:0))
// ABSENT-NOT: loc("onnx.Add"("{{(/[[:alnum:]]+)+}}.onnx":1:0))
