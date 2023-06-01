// RUN: onnx-mlir --preserveLocations --printIR %s |  FileCheck %s --check-prefix=PRESENT
// RUN: onnx-mlir --printIR %s | FileCheck %s --check-prefix=ABSENT

  func.func @main_graph(%arg0: tensor<1x16xf32>, %arg1: tensor<1x16xf32>) -> tensor<1x16xf32>  {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<1x16xf32>, tensor<1x16xf32>) -> tensor<1x16xf32> loc("/build/workspace/addop.onnx":1:0)
    onnx.Return %0 : tensor<1x16xf32>
  }

// PRESENT: #[[onnx_file_loc:.+]] = loc("{{/.+}}.onnx":1:0)
// PRESENT: loc("onnx.Add"(#[[onnx_file_loc]]))
// ABSENT-NOT: loc("onnx.Add"(
