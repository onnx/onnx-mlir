// RUN: onnx-mlir-opt --convert-onnx-to-linalg='linalg-ops=onnx.Constant' %s -split-input-file | FileCheck %s --check-prefix=ENABLED
// RUN: onnx-mlir-opt --convert-onnx-to-linalg='linalg-ops=NONE' %s -split-input-file | FileCheck %s --check-prefix=DISABLED

// -----

func.func @constant_dense_f32() -> tensor<2x2xf32> {
  %0 = "onnx.Constant"() {
    value = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  } : () -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>

  // ENABLED-LABEL: func.func @constant_dense_f32()
  // ENABLED-NOT: onnx.Constant
  // ENABLED: %[[RESULT:.*]] = arith.constant dense<{{\[\[}}1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>
  // ENABLED: return %[[RESULT]] : tensor<2x2xf32>

  // DISABLED-LABEL: func.func @constant_dense_f32()
  // DISABLED: %[[RESULT:.*]] = onnx.Constant dense<{{\[\[}}1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>
  // DISABLED: return %[[RESULT]] : tensor<2x2xf32>
}

// -----

func.func @constant_dense_i64() -> tensor<3xi64> {
  %0 = "onnx.Constant"() {
    value = dense<[1, -2, 3]> : tensor<3xi64>
  } : () -> tensor<3xi64>
  return %0 : tensor<3xi64>

  // ENABLED-LABEL: func.func @constant_dense_i64()
  // ENABLED-NOT: onnx.Constant
  // ENABLED: %[[RESULT:.*]] = arith.constant dense<[1, -2, 3]> : tensor<3xi64>
  // ENABLED: return %[[RESULT]] : tensor<3xi64>
}

// -----

func.func @constant_scalar() -> tensor<i32> {
  %0 = "onnx.Constant"() {
    value = dense<42> : tensor<i32>
  } : () -> tensor<i32>
  return %0 : tensor<i32>

  // ENABLED-LABEL: func.func @constant_scalar()
  // ENABLED-NOT: onnx.Constant
  // ENABLED: %[[RESULT:.*]] = arith.constant dense<42> : tensor<i32>
  // ENABLED: return %[[RESULT]] : tensor<i32>
}

// -----

func.func @constant_not_normalized() -> tensor<f32> {
  %0 = "onnx.Constant"() {
    value_float = 1.5 : f32
  } : () -> tensor<f32>
  return %0 : tensor<f32>

  // ENABLED-LABEL: func.func @constant_not_normalized()
  // ENABLED: %[[RESULT:.*]] = onnx.Constant
  // ENABLED-SAME: {value_float = 1.500000e+00 : f32} : tensor<f32>
  // ENABLED: return %[[RESULT]] : tensor<f32>
}

// -----

func.func @constant_different_types() -> tensor<?xf32> {
  %0 = "onnx.Constant"() {
    value = dense<[1.0, 2.0]> : tensor<2xf32>
  } : () -> tensor<?xf32>
  return %0 : tensor<?xf32>

  // ENABLED-LABEL: func.func @constant_different_types()
  // ENABLED: %[[RESULT:.*]] = onnx.Constant
  // ENABLED-SAME: {value = dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>}
  // ENABLED-SAME: : tensor<?xf32>
  // ENABLED: return %[[RESULT]] : tensor<?xf32>
}

// -----

func.func @constant_scalar_ui32() -> tensor<ui32> {
  %0 = "onnx.Constant"() {
    value = dense<1> : tensor<ui32>
  } : () -> tensor<ui32>
  return %0 : tensor<ui32>

  // ENABLED-LABEL: func.func @constant_scalar_ui32()
  // ENABLED-NOT: onnx.Constant
  // ENABLED: %[[SIGNLESS:.*]] = arith.constant dense<1> : tensor<i32>
  // ENABLED-NEXT: %[[RESULT:.*]] = builtin.unrealized_conversion_cast %[[SIGNLESS]] : tensor<i32> to tensor<ui32>
  // ENABLED-NEXT: return %[[RESULT]] : tensor<ui32>

  // DISABLED-LABEL: func.func @constant_scalar_ui32()
  // DISABLED: %[[RESULT:.*]] = onnx.Constant dense<1> : tensor<ui32>
  // DISABLED: return %[[RESULT]] : tensor<ui32>
}

// -----

func.func @constant_dense_ui32() -> tensor<4xui32> {
  %0 = "onnx.Constant"() {
    value = dense<[0, 1, 2147483648, 4294967295]> : tensor<4xui32>
  } : () -> tensor<4xui32>
  return %0 : tensor<4xui32>

  // ENABLED-LABEL: func.func @constant_dense_ui32()
  // ENABLED-NOT: onnx.Constant
  // ENABLED: %[[SIGNLESS:.*]] = arith.constant dense<[0, 1, -2147483648, -1]> : tensor<4xi32>
  // ENABLED-NEXT: %[[RESULT:.*]] = builtin.unrealized_conversion_cast %[[SIGNLESS]] : tensor<4xi32> to tensor<4xui32>
  // ENABLED-NEXT: return %[[RESULT]] : tensor<4xui32>

  // DISABLED-LABEL: func.func @constant_dense_ui32()
  // DISABLED: %[[RESULT:.*]] = onnx.Constant dense<[0, 1, 2147483648, 4294967295]> : tensor<4xui32>
  // DISABLED: return %[[RESULT]] : tensor<4xui32>
}
