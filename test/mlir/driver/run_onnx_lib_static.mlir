// Test that build-run-onnx-lib.sh can produce a statically-linked binary from
// a compiled model and that the binary executes successfully.
//
// Step 1: compile the MLIR model below to a shared library.
// RUN: onnx-mlir %s -o %t 2>&1 | FileCheck --check-prefix=COMPILE %s
//
// Step 2: link it statically into a standalone runner binary.
// RUN: env ONNX_MLIR_HOME=%onnx-mlir-home \
// RUN:   sh %S/../../../utils/build-run-onnx-lib.sh %t.so %t-run 2>&1 \
// RUN:   | FileCheck --check-prefix=BUILD %s
//
// Step 3: run the binary — just verify it exits cleanly.
// RUN: %t-run | FileCheck --check-prefix=EXEC %s
//
// Cleanup
// RUN: rm -f %t.so %t-run

// REQUIRES: system-linux || system-darwin

// COMPILE: Compilation completed

// BUILD: Success

// EXEC: Finish computing 1 iterations

module {
  func.func @main_graph(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<3x4x5xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xf32>
    onnx.Return %0 : tensor<3x4x5xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
