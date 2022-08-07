// RUN: onnx-mlir-opt %s -split-input-file | FileCheck %s

//===----------------------------------------------------------------------===//
// CHECK-LABEL: @check_map1(%arg0: tuple<i64, f32>) -> tensor<*xf32> {
func.func @check_map1(%arg0: tuple<i64, f32>) -> tensor<*xf32> {
  %0 = "onnx.CastMap"(%arg0) {cast_to = "TO_FLOAT", map_form = "DENSE", max_map = 1 : si64} : (tuple<i64, f32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
  // CHECK-NEXT: %0 = "onnx.CastMap"(%arg0) {cast_to = "TO_FLOAT", map_form = "DENSE", max_map = 1 : si64} : (tuple<i64, f32>) -> tensor<*xf32>
}

// CHECK-LABEL: @check_string(%arg0: tensor<10x20x!onnx.String>) -> tensor<10x20x!onnx.String> {
func.func @check_string(%arg0: tensor<10x20x!onnx.String>) -> tensor<10x20x!onnx.String> {
  return %arg0 : tensor<10x20x!onnx.String>
  // CHECK-NEXT: return %arg0 : tensor<10x20x!onnx.String>
}

// CHECK-LABEL: @check_seq(%arg0: tensor<10x20xf32>, %arg1: tensor<5x20xf32>) -> tensor<*xf32> {
func.func @check_seq(%arg0: tensor<10x20xf32>, %arg1: tensor<5x20xf32>) -> tensor<*xf32> {
  %cst = "onnx.Constant"() {value = dense<[0]> : tensor<1xi32>} : () -> tensor<1xi32>
  %0 = "onnx.SequenceConstruct"(%arg0, %arg1) : (tensor<10x20xf32>, tensor<5x20xf32>) -> !onnx.Seq<tensor<*xf32>>
  %1 = "onnx.SequenceAt"(%0, %cst) : (!onnx.Seq<tensor<*xf32>>, tensor<1xi32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>
  // CHECK-NEXT: %0 = "onnx.Constant"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK-NEXT: %1 = "onnx.SequenceConstruct"(%arg0, %arg1) : (tensor<10x20xf32>, tensor<5x20xf32>) -> !onnx.Seq<tensor<*xf32>>
  // CHECK-NEXT: %2 = "onnx.SequenceAt"(%1, %0) : (!onnx.Seq<tensor<*xf32>>, tensor<1xi32>) -> tensor<*xf32>
}

// CHECK-LABEL: func.func @check_opt(%arg0: !onnx.Opt<tensor<*xf32>>) -> !onnx.Opt<tensor<*xf32>> {
func.func @check_opt(%arg0: !onnx.Opt<tensor<*xf32>>) -> !onnx.Opt<tensor<*xf32>> {
  %0 = "onnx.Identity"(%arg0) : (!onnx.Opt<tensor<*xf32>>) -> !onnx.Opt<tensor<*xf32>>
  return %0 : !onnx.Opt<tensor<*xf32>>
  // CHECK-NEXT: [[VAR_0_:%.+]] = "onnx.Identity"(%arg0) : (!onnx.Opt<tensor<*xf32>>) -> !onnx.Opt<tensor<*xf32>>
  // CHECK-NEXT: return [[VAR_0_]] : !onnx.Opt<tensor<*xf32>>
}

// CHECK-LABEL: func.func @check_opt_seq(%arg0: !onnx.Opt<!onnx.Seq<tensor<*xf32>>>) -> !onnx.Opt<!onnx.Seq<tensor<*xf32>>> {
func.func @check_opt_seq(%arg0: !onnx.Opt<!onnx.Seq<tensor<*xf32>>>) -> !onnx.Opt<!onnx.Seq<tensor<*xf32>>> {
  %0 = "onnx.Identity"(%arg0) : (!onnx.Opt<!onnx.Seq<tensor<*xf32>>>) -> !onnx.Opt<!onnx.Seq<tensor<*xf32>>>
  return %0 : !onnx.Opt<!onnx.Seq<tensor<*xf32>>>
  // CHECK-NEXT: [[VAR_0_:%.+]] = "onnx.Identity"(%arg0) : (!onnx.Opt<!onnx.Seq<tensor<*xf32>>>) -> !onnx.Opt<!onnx.Seq<tensor<*xf32>>>
  // CHECK-NEXT: return [[VAR_0_]] : !onnx.Opt<!onnx.Seq<tensor<*xf32>>>
}

// CHECK-LABEL: func.func @check_optional(%arg0: tensor<*xf32>) -> !onnx.Opt<tensor<*xf32>> {
func.func @check_optional(%arg0: tensor<*xf32>) -> !onnx.Opt<tensor<*xf32>> {
  %0 = "onnx.Optional"(%arg0) : (tensor<*xf32>) -> !onnx.Opt<tensor<*xf32>>
  return %0 : !onnx.Opt<tensor<*xf32>>
  // CHECK-NEXT: [[VAR_0_:%.+]] = "onnx.Optional"(%arg0) : (tensor<*xf32>) -> !onnx.Opt<tensor<*xf32>>
  // CHECK-NEXT: return [[VAR_0_]] : !onnx.Opt<tensor<*xf32>>
}

// CHECK-LABEL: func.func @check_optional_none() -> !onnx.Opt<tensor<*xf32>> {
func.func @check_optional_none() -> !onnx.Opt<tensor<*xf32>> {
  %0 = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.Optional"(%0) {type = tensor<*xf32>} : (none) -> !onnx.Opt<tensor<*xf32>>
  return %1 : !onnx.Opt<tensor<*xf32>>
  // CHECK-NEXT: [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
  // CHECK-NEXT: [[VAR_1_:%.+]] = "onnx.Optional"([[VAR_0_]]) {type = tensor<*xf32>} : (none) -> !onnx.Opt<tensor<*xf32>>
  // CHECK-NEXT: return [[VAR_1_]] : !onnx.Opt<tensor<*xf32>>
}

// CHECK-LABEL: func.func @check_optionalgetelement(%arg0: !onnx.Opt<tensor<*xf32>>) -> tensor<*xf32> {
func.func @check_optionalgetelement(%arg0: !onnx.Opt<tensor<*xf32>>) -> tensor<*xf32> {
  %0 = "onnx.OptionalGetElement"(%arg0) : (!onnx.Opt<tensor<*xf32>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
  // CHECK-NEXT: [[VAR_0_:%.+]] = "onnx.OptionalGetElement"(%arg0) : (!onnx.Opt<tensor<*xf32>>) -> tensor<*xf32>
  // CHECK-NEXT: return [[VAR_0_]] : tensor<*xf32>
}

// CHECK-LABEL: func.func @check_optionalhaselement(%arg0: !onnx.Opt<tensor<*xf32>>) -> tensor<i1> {
func.func @check_optionalhaselement(%arg0: !onnx.Opt<tensor<*xf32>>) -> tensor<i1> {
  %0 = "onnx.OptionalHasElement"(%arg0) : (!onnx.Opt<tensor<*xf32>>) -> tensor<i1>
  return %0 : tensor<i1>
  // CHECK-NEXT: [[VAR_0_:%.+]] = "onnx.OptionalHasElement"(%arg0) : (!onnx.Opt<tensor<*xf32>>) -> tensor<i1>
  // CHECK-NEXT: return [[VAR_0_]] : tensor<i1>
}
