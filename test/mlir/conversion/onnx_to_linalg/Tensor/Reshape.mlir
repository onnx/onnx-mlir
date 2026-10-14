// RUN: onnx-mlir-opt --convert-onnx-to-linalg='linalg-ops=onnx.Reshape' %s -split-input-file | FileCheck %s --check-prefix=ENABLED
// RUN: onnx-mlir-opt --convert-onnx-to-linalg='linalg-ops=NONE' %s -split-input-file | FileCheck %s --check-prefix=DISABLED

// -----

func.func @reshape_static(%arg0: tensor<2x3x4xf32>) -> tensor<6x4xf32> {
  %shape = onnx.Constant dense<[6, 4]> : tensor<2xi64>
  %0 = "onnx.Reshape"(%arg0, %shape) {allowzero = 0 : si64}
      : (tensor<2x3x4xf32>, tensor<2xi64>) -> tensor<6x4xf32>
  return %0 : tensor<6x4xf32>

  // ENABLED-LABEL: func.func @reshape_static(
  // ENABLED: %[[SHAPE:.*]] = arith.constant dense<[6, 4]> : tensor<2xindex>
  // ENABLED-NEXT: %[[RESULT:.*]] = tensor.reshape %arg0(%[[SHAPE]]) : (tensor<2x3x4xf32>, tensor<2xindex>) -> tensor<6x4xf32>
  // ENABLED-NEXT: return %[[RESULT]] : tensor<6x4xf32>
  // ENABLED-NOT: "onnx.Reshape"

  // DISABLED-LABEL: func.func @reshape_static(
  // DISABLED: %[[RESULT:.*]] = "onnx.Reshape"
  // DISABLED-SAME: -> tensor<6x4xf32>
  // DISABLED: return %[[RESULT]] : tensor<6x4xf32>
}

// -----

func.func @reshape_inferred_dimension(
    %arg0: tensor<2x3x4xf32>) -> tensor<2x12xf32> {
  %shape = onnx.Constant dense<[2, -1]> : tensor<2xi64>
  %0 = "onnx.Reshape"(%arg0, %shape) {allowzero = 0 : si64}
      : (tensor<2x3x4xf32>, tensor<2xi64>) -> tensor<2x12xf32>
  return %0 : tensor<2x12xf32>

  // ENABLED-LABEL: func.func @reshape_inferred_dimension(
  // ENABLED: %[[SHAPE:.*]] = arith.constant dense<[2, 12]> : tensor<2xindex>
  // ENABLED: %[[RESULT:.*]] = tensor.reshape %arg0(%[[SHAPE]]) : (tensor<2x3x4xf32>, tensor<2xindex>) -> tensor<2x12xf32>
  // ENABLED: return %[[RESULT]] : tensor<2x12xf32>
  // ENABLED-NOT: "onnx.Reshape"
}

// -----

func.func @reshape_copy_dimension(
    %arg0: tensor<2x3x4xf32>) -> tensor<2x12xf32> {
  %shape = onnx.Constant dense<[0, 12]> : tensor<2xi64>
  %0 = "onnx.Reshape"(%arg0, %shape) {allowzero = 0 : si64}
      : (tensor<2x3x4xf32>, tensor<2xi64>) -> tensor<2x12xf32>
  return %0 : tensor<2x12xf32>

  // ENABLED-LABEL: func.func @reshape_copy_dimension(
  // ENABLED: %[[SHAPE:.*]] = arith.constant dense<[2, 12]> : tensor<2xindex>
  // ENABLED: %[[RESULT:.*]] = tensor.reshape %arg0(%[[SHAPE]]) : (tensor<2x3x4xf32>, tensor<2xindex>) -> tensor<2x12xf32>
  // ENABLED: return %[[RESULT]] : tensor<2x12xf32>
  // ENABLED-NOT: "onnx.Reshape"
}

// -----

func.func @reshape_dynamic_input(
    %arg0: tensor<?x3x4xf32>) -> tensor<?x12xf32> {
  %shape = onnx.Constant dense<[0, -1]> : tensor<2xi64>
  %0 = "onnx.Reshape"(%arg0, %shape) {allowzero = 0 : si64}
      : (tensor<?x3x4xf32>, tensor<2xi64>) -> tensor<?x12xf32>
  return %0 : tensor<?x12xf32>

  // ENABLED-LABEL: func.func @reshape_dynamic_input(
  // ENABLED: %[[DIM:.*]] = tensor.dim %arg0, %{{.*}} : tensor<?x3x4xf32>
  // ENABLED: %[[SHAPE:.*]] = tensor.from_elements %{{.*}}, %{{.*}} : tensor<2xindex>
  // ENABLED: %[[RESULT:.*]] = tensor.reshape %arg0(%[[SHAPE]]) : (tensor<?x3x4xf32>, tensor<2xindex>) -> tensor<?x12xf32>
  // ENABLED: return %[[RESULT]] : tensor<?x12xf32>
  // ENABLED-NOT: "onnx.Reshape"
}

// -----

func.func @reshape_runtime_shape(%arg0: tensor<2x3x4xf32>,
    %shape: tensor<2xi64>) -> tensor<?x?xf32> {
  %0 = "onnx.Reshape"(%arg0, %shape) {allowzero = 0 : si64}
      : (tensor<2x3x4xf32>, tensor<2xi64>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>

  // ENABLED-LABEL: func.func @reshape_runtime_shape(
  // ENABLED-SAME: %[[DATA:.*]]: tensor<2x3x4xf32>, %[[INPUT_SHAPE:.*]]: tensor<2xi64>
  // ENABLED: tensor.extract %[[INPUT_SHAPE]][%{{.*}}] : tensor<2xi64>
  // ENABLED: %[[RESHAPE_SHAPE:.*]] = tensor.from_elements %{{.*}}, %{{.*}} : tensor<2xindex>
  // ENABLED: %[[RESULT:.*]] = tensor.reshape %[[DATA]](%[[RESHAPE_SHAPE]]) : (tensor<2x3x4xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  // ENABLED: return %[[RESULT]] : tensor<?x?xf32>
  // ENABLED-NOT: "onnx.Reshape"
}

// -----

func.func @reshape_to_scalar(%arg0: tensor<1xf32>) -> tensor<f32> {
  %shape = onnx.Constant dense<> : tensor<0xi64>
  %0 = "onnx.Reshape"(%arg0, %shape) {allowzero = 0 : si64}
      : (tensor<1xf32>, tensor<0xi64>) -> tensor<f32>
  return %0 : tensor<f32>

  // ENABLED-LABEL: func.func @reshape_to_scalar(
  // ENABLED: %[[SHAPE:.*]] = arith.constant dense<> : tensor<0xindex>
  // ENABLED: %[[RESULT:.*]] = tensor.reshape %arg0(%[[SHAPE]]) : (tensor<1xf32>, tensor<0xindex>) -> tensor<f32>
  // ENABLED: return %[[RESULT]] : tensor<f32>
  // ENABLED-NOT: "onnx.Reshape"
}

// -----

func.func @reshape_allowzero_not_lowered(
    %arg0: tensor<0x3xf32>) -> tensor<0x3xf32> {
  %shape = onnx.Constant dense<[0, 3]> : tensor<2xi64>
  %0 = "onnx.Reshape"(%arg0, %shape) {allowzero = 1 : si64}
      : (tensor<0x3xf32>, tensor<2xi64>) -> tensor<0x3xf32>
  return %0 : tensor<0x3xf32>

  // ENABLED-LABEL: func.func @reshape_allowzero_not_lowered(
  // ENABLED: %[[RESULT:.*]] = "onnx.Reshape"
  // ENABLED-SAME: <{allowzero = 1 : si64}>
  // ENABLED-SAME: -> tensor<0x3xf32>
  // ENABLED: return %[[RESULT]] : tensor<0x3xf32>
}

// -----

func.func @reshape_dynamic_shape_length_not_lowered(
    %arg0: tensor<2x3xf32>, %shape: tensor<?xi64>) -> tensor<*xf32> {
  %0 = "onnx.Reshape"(%arg0, %shape) {allowzero = 0 : si64}
      : (tensor<2x3xf32>, tensor<?xi64>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

  // ENABLED-LABEL: func.func @reshape_dynamic_shape_length_not_lowered(
  // ENABLED: %[[RESULT:.*]] = "onnx.Reshape"
  // ENABLED-SAME: -> tensor<*xf32>
  // ENABLED: return %[[RESULT]] : tensor<*xf32>
}

// -----

func.func @reshape_unranked_input_not_lowered(
    %arg0: tensor<*xf32>) -> tensor<2x?xf32> {
  %shape = onnx.Constant dense<[2, -1]> : tensor<2xi64>
  %0 = "onnx.Reshape"(%arg0, %shape) {allowzero = 0 : si64}
      : (tensor<*xf32>, tensor<2xi64>) -> tensor<2x?xf32>
  return %0 : tensor<2x?xf32>

  // ENABLED-LABEL: func.func @reshape_unranked_input_not_lowered(
  // ENABLED: %[[RESULT:.*]] = "onnx.Reshape"
  // ENABLED-SAME: -> tensor<2x?xf32>
  // ENABLED: return %[[RESULT]] : tensor<2x?xf32>
}
