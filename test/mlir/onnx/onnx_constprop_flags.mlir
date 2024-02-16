// RUN: onnx-mlir-opt --constprop-onnx --onnx-const-prop-expansion-bound=2 %s -split-input-file | FileCheck --check-prefix=EXPANSIONBOUND2 %s
// RUN: onnx-mlir-opt --constprop-onnx --onnx-const-prop-round-fp-to-int=true %s -split-input-file | FileCheck --check-prefix=ROUND %s
// RUN: onnx-mlir-opt --constprop-onnx --onnx-const-prop-round-fp-to-int=false %s -split-input-file | FileCheck --check-prefix=TRUNCATE %s

//===----------------------------------------------------------------------===//
// Constant propagate ONNXAddOp only if expansion bound satisfied
//===----------------------------------------------------------------------===//

func.func @test_add_propagates() -> tensor<2x5xf32> {
  %0 = "onnx.Constant"() {value = dense<1.0> : tensor<2x1xf32>} : () -> tensor<2x1xf32>
  %1 = "onnx.Constant"() {value = dense<2.0> : tensor<1x5xf32>} : () -> tensor<1x5xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<2x1xf32> , tensor<1x5xf32>) -> tensor<2x5xf32>
  onnx.Return %2 : tensor<2x5xf32>
}
// EXPANSIONBOUND2-LABEL: @test_add_propagates() -> tensor<2x5xf32>
// EXPANSIONBOUND2: onnx.Constant {{.*}} : tensor<2x5xf32>

// -----

func.func @test_add_doesnt_propagate() -> tensor<5x5xf32> {
  %0 = "onnx.Constant"() {value = dense<1.0> : tensor<5x1xf32>} : () -> tensor<5x1xf32>
  %1 = "onnx.Constant"() {value = dense<2.0> : tensor<1x5xf32>} : () -> tensor<1x5xf32>
  %2 = "onnx.Add"(%0, %1) : (tensor<5x1xf32> , tensor<1x5xf32>) -> tensor<5x5xf32>
  onnx.Return %2 : tensor<5x5xf32>
}
// EXPANSIONBOUND2-LABEL: @test_add_doesnt_propagate() -> tensor<5x5xf32>
// EXPANSIONBOUND2: "onnx.Add"(%0, %1) : (tensor<5x1xf32>, tensor<1x5xf32>) -> tensor<5x5xf32>

// -----

func.func @test_cast_f16_i16() -> tensor<6xi16> {
  %0 = onnx.Constant dense<[-1.5, -0.5, 0.4, 0.5, 1.5, 1.6]> : tensor<6xf16>
  %1 = "onnx.Cast"(%0) {to = i16} : (tensor<6xf16>) -> tensor<6xi16>
  onnx.Return %1 : tensor<6xi16>
}
// ROUND-LABEL: @test_cast_f16_i16() -> tensor<6xi16>
// ROUND: onnx.Constant dense<[-2, 0, 0, 0, 2, 2]> : tensor<6xi16>
//
// TRUNCATE-LABEL: @test_cast_f16_i16() -> tensor<6xi16>
// TRUNCATE: onnx.Constant dense<[-1, 0, 0, 0, 1, 1]> : tensor<6xi16>
