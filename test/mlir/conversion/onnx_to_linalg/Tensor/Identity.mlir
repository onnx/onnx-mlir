// RUN: onnx-mlir-opt --convert-onnx-to-linalg='linalg-ops=onnx.Identity' %s -split-input-file | FileCheck %s --check-prefix=ENABLED
// RUN: onnx-mlir-opt --convert-onnx-to-linalg='linalg-ops=NONE' %s -split-input-file | FileCheck %s --check-prefix=DISABLED

// -----

func.func @identity_static(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "onnx.Identity"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>

  // ENABLED-LABEL: func.func @identity_static(
  // ENABLED-SAME: %[[ARG:.*]]: tensor<2x3xf32>
  // ENABLED-NOT: "onnx.Identity"
  // ENABLED: return %[[ARG]] : tensor<2x3xf32>

  // DISABLED-LABEL: func.func @identity_static(
  // DISABLED-SAME: %[[ARG:.*]]: tensor<2x3xf32>
  // DISABLED: %[[RESULT:.*]] = "onnx.Identity"(%[[ARG]])
  // DISABLED: return %[[RESULT]] : tensor<2x3xf32>
}

// -----

func.func @identity_dynamic(%arg0: tensor<?x3xi64>) -> tensor<?x3xi64> {
  %0 = "onnx.Identity"(%arg0) : (tensor<?x3xi64>) -> tensor<?x3xi64>
  return %0 : tensor<?x3xi64>

  // ENABLED-LABEL: func.func @identity_dynamic(
  // ENABLED-SAME: %[[ARG:.*]]: tensor<?x3xi64>
  // ENABLED-NOT: "onnx.Identity"
  // ENABLED: return %[[ARG]] : tensor<?x3xi64>
}

// -----

func.func @identity_scalar(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "onnx.Identity"(%arg0) : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>

  // ENABLED-LABEL: func.func @identity_scalar(
  // ENABLED-SAME: %[[ARG:.*]]: tensor<f32>
  // ENABLED-NOT: "onnx.Identity"
  // ENABLED: return %[[ARG]] : tensor<f32>
}

// -----

func.func @identity_unranked(%arg0: tensor<*xf16>) -> tensor<*xf16> {
  %0 = "onnx.Identity"(%arg0) : (tensor<*xf16>) -> tensor<*xf16>
  return %0 : tensor<*xf16>

  // ENABLED-LABEL: func.func @identity_unranked(
  // ENABLED-SAME: %[[ARG:.*]]: tensor<*xf16>
  // ENABLED-NOT: "onnx.Identity"
  // ENABLED: return %[[ARG]] : tensor<*xf16>
}

// -----

func.func @identity_chain(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  %0 = "onnx.Identity"(%arg0) : (tensor<4xi32>) -> tensor<4xi32>
  %1 = "onnx.Identity"(%0) : (tensor<4xi32>) -> tensor<4xi32>
  return %1 : tensor<4xi32>

  // ENABLED-LABEL: func.func @identity_chain(
  // ENABLED-SAME: %[[ARG:.*]]: tensor<4xi32>
  // ENABLED-NOT: "onnx.Identity"
  // ENABLED: return %[[ARG]] : tensor<4xi32>
}

// -----

// Replacing an Identity with a differently typed input would invalidate its
// users. Leave it unchanged until shape inference makes the types identical.
func.func @identity_different_types(
    %arg0: tensor<2x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.Identity"(%arg0) : (tensor<2x3xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

  // ENABLED-LABEL: func.func @identity_different_types(
  // ENABLED: %[[RESULT:.*]] = "onnx.Identity"(%arg0)
  // ENABLED-SAME: : (tensor<2x3xf32>) -> tensor<*xf32>
  // ENABLED: return %[[RESULT]] : tensor<*xf32>
}

// -----

// Identity also supports ONNX optional values, not only tensors.
func.func @identity_optional(
    %arg0: !onnx.Opt<tensor<*xf32>>) -> !onnx.Opt<tensor<*xf32>> {
  %0 = "onnx.Identity"(%arg0)
      : (!onnx.Opt<tensor<*xf32>>) -> !onnx.Opt<tensor<*xf32>>
  return %0 : !onnx.Opt<tensor<*xf32>>

  // ENABLED-LABEL: func.func @identity_optional(
  // ENABLED-SAME: %[[ARG:.*]]: !onnx.Opt<tensor<*xf32>>
  // ENABLED-NOT: "onnx.Identity"
  // ENABLED: return %[[ARG]] : !onnx.Opt<tensor<*xf32>>
}
