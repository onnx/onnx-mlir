// RUN: onnx-mlir-opt --shape-inference %s | FileCheck %s

// CHECK-LABEL: func.func @check_opt_identity(%arg0: !onnx.Opt<tensor<2xf32>>) -> !onnx.Opt<tensor<2xf32>> {
func.func @check_opt_identity(%arg0: !onnx.Opt<tensor<2xf32>>) -> !onnx.Opt<tensor<*xf32>> {
  %0 = "onnx.Identity"(%arg0) : (!onnx.Opt<tensor<2xf32>>) -> !onnx.Opt<tensor<*xf32>>
  return %0 : !onnx.Opt<tensor<*xf32>>
  // CHECK-NEXT: [[VAR_0_:%.+]] = "onnx.Identity"(%arg0) : (!onnx.Opt<tensor<2xf32>>) -> !onnx.Opt<tensor<2xf32>>
  // CHECK-NEXT: return [[VAR_0_]] : !onnx.Opt<tensor<2xf32>>
}
