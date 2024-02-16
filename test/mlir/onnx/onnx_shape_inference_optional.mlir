// RUN: onnx-mlir-opt --shape-inference %s | FileCheck %s

// -----

// CHECK-LABEL: func.func @check_opt_identity(%arg0: !onnx.Opt<tensor<2xf32>>) -> !onnx.Opt<tensor<2xf32>> {
func.func @check_opt_identity(%arg0: !onnx.Opt<tensor<2xf32>>) -> !onnx.Opt<tensor<*xf32>> {
  %0 = "onnx.Identity"(%arg0) : (!onnx.Opt<tensor<2xf32>>) -> !onnx.Opt<tensor<*xf32>>
  onnx.Return %0 : !onnx.Opt<tensor<*xf32>>
  // CHECK-NEXT: [[VAR_0_:%.+]] = "onnx.Identity"(%arg0) : (!onnx.Opt<tensor<2xf32>>) -> !onnx.Opt<tensor<2xf32>>
  // CHECK-NEXT: onnx.Return [[VAR_0_]] : !onnx.Opt<tensor<2xf32>>
}

// -----

// CHECK-LABEL: func.func @check_optional(%arg0: tensor<2xf32>) -> !onnx.Opt<tensor<2xf32>> {
func.func @check_optional(%arg0: tensor<2xf32>) -> !onnx.Opt<tensor<*xf32>> {
  %0 = "onnx.Optional"(%arg0) : (tensor<2xf32>) -> !onnx.Opt<tensor<*xf32>>
  onnx.Return %0 : !onnx.Opt<tensor<*xf32>>
  // CHECK-NEXT: [[VAR_0_:%.+]] = "onnx.Optional"(%arg0) : (tensor<2xf32>) -> !onnx.Opt<tensor<2xf32>>
  // CHECK-NEXT: onnx.Return [[VAR_0_]] : !onnx.Opt<tensor<2xf32>>
}

// -----

// CHECK-LABEL: func.func @check_optional_none() -> !onnx.Opt<tensor<2xf32>> {
func.func @check_optional_none() -> !onnx.Opt<tensor<*xf32>> {
  %0 = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.Optional"(%0) {type = tensor<2xf32>} : (none) -> !onnx.Opt<tensor<*xf32>>
  onnx.Return %1 : !onnx.Opt<tensor<*xf32>>
  // CHECK-NEXT: [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
  // CHECK-NEXT: [[VAR_1_:%.+]] = "onnx.Optional"([[VAR_0_]]) {type = tensor<2xf32>} : (none) -> !onnx.Opt<tensor<2xf32>>
  // CHECK-NEXT: onnx.Return [[VAR_1_]] : !onnx.Opt<tensor<2xf32>>
}

// -----

// CHECK-LABEL: func.func @check_optionalgetelement(%arg0: !onnx.Opt<tensor<2xf32>>) -> tensor<2xf32> {
func.func @check_optionalgetelement(%arg0: !onnx.Opt<tensor<2xf32>>) -> tensor<*xf32> {
  %0 = "onnx.OptionalGetElement"(%arg0) : (!onnx.Opt<tensor<2xf32>>) -> tensor<*xf32>
  onnx.Return %0 : tensor<*xf32>
  // CHECK-NEXT: [[VAR_0_:%.+]] = "onnx.OptionalGetElement"(%arg0) : (!onnx.Opt<tensor<2xf32>>) -> tensor<2xf32>
  // CHECK-NEXT: onnx.Return [[VAR_0_]] : tensor<2xf32>
}

// -----

// CHECK-LABEL: func.func @check_optionalhaselement(%arg0: !onnx.Opt<tensor<*xf32>>) -> tensor<i1> {
func.func @check_optionalhaselement(%arg0: !onnx.Opt<tensor<*xf32>>) -> tensor<*xi1> {
  %0 = "onnx.OptionalHasElement"(%arg0) : (!onnx.Opt<tensor<*xf32>>) -> tensor<*xi1>
  onnx.Return %0 : tensor<*xi1>
  // CHECK-NEXT: [[VAR_0_:%.+]] = "onnx.OptionalHasElement"(%arg0) : (!onnx.Opt<tensor<*xf32>>) -> tensor<i1>
  // CHECK-NEXT: onnx.Return [[VAR_0_]] : tensor<i1>
}
