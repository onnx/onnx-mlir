// Copy the test source file to the output directory, so that the tests do not pollute the source
// tree even when the given output path is invalid (because output defaults to location of source).

// RUN: rm -rf %t && mkdir %t
// RUN: cp %s %t/invalid_output_path.mlir

// RUN: onnx-mlir %t/invalid_output_path.mlir -tag="test" -v -o %t\abc 2>&1 | FileCheck --check-prefix=VALID %s
// RUN: onnx-mlir %t/invalid_output_path.mlir -tag="test" -v -o %t/abc 2>&1 | FileCheck --check-prefix=VALID %s
// RUN: onnx-mlir %t/invalid_output_path.mlir -tag="test" -v -o %t/abc. 2>&1 | FileCheck --check-prefix=VALID %s
// RUN: onnx-mlir %t/invalid_output_path.mlir -tag="test" -v -o %t/abc.. 2>&1 | FileCheck --check-prefix=VALID %s
// RUN: onnx-mlir %t/invalid_output_path.mlir -tag="test" -v -o %t/..abc.. 2>&1 | FileCheck --check-prefix=VALID %s

// RUN: onnx-mlir %t/invalid_output_path.mlir -tag="test" -v -o %t\\ 2>&1 | FileCheck --check-prefix=INVALID %s
// RUN: onnx-mlir %t/invalid_output_path.mlir -tag="test" -v -o %t/ 2>&1 | FileCheck --check-prefix=INVALID %s
// RUN: onnx-mlir %t/invalid_output_path.mlir -tag="test" -v -o %t/. 2>&1 | FileCheck --check-prefix=INVALID %s
// RUN: onnx-mlir %t/invalid_output_path.mlir -tag="test" -v -o . 2>&1 | FileCheck --check-prefix=INVALID %s
// RUN: onnx-mlir %t/invalid_output_path.mlir -tag="test" -v -o .. 2>&1 | FileCheck --check-prefix=INVALID %s

// INVALID: Invalid -o option value {{.*}} ignored.
// VALID-NOT: Invalid -o option value {{.*}} ignored.

module {
  func.func @main_graph(%arg0: tensor<1x1xf32>, %arg1: tensor<1x1xf32>) -> tensor<1x1xf32> {
    %0 = "onnx.MatMul"(%arg0, %arg1) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
    onnx.Return %0 : tensor<1x1xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
