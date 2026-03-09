// RUN: out=$(dirname %s)/test && onnx-mlir --store-constants-to-file --constants-to-file-single-threshold=0.03 --constants-to-file-total-threshold=0.00000006 -o ${out} %s | FileCheck %s && rm ${out}.so ${out}.constants.bin

module {
  func.func @main_graph() -> tensor<10xi64> {
      %0 = onnx.Constant dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]> : tensor<10xi64>
        return %0 : tensor<10xi64>
  }
    "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// CHECK: [1/6] {{.*}} Importing ONNX Model to MLIR Module from
// CHECK: [2/6] {{.*}} Compiling and Optimizing MLIR Module
// CHECK-NEXT: Constants in the model exceeds the thresholds (single constant <= {{.*}} KB, total constants <= {{.*}} GB). Stored them in an external file: {{.*}}.constants.bin". Make sure to put this file in the same folder as the generated model or set OM_CONSTANT_PATH to the folder having this file. For constants-related settings, see options --store-constants-to-file, --constants-to-file-single-threshold and --constants-to-file-total-threshold.
