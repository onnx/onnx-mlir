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

// ENABLED-LABEL:  func.func @reshape_dynamic_input
// ENABLED-SAME:   ([[PARAM_0_:%.+]]: tensor<?x3x4xf32>) -> tensor<?x12xf32> {
// ENABLED-DAG:       [[CST_12_:%.+]] = arith.constant 12 : index
// ENABLED-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// ENABLED:           [[VAR_dim_:%.+]] = tensor.dim [[PARAM_0_]], [[CST_0_]] : tensor<?x3x4xf32>
// ENABLED:           [[VAR_from_elements_:%.+]] = tensor.from_elements [[VAR_dim_]], [[CST_12_]] : tensor<2xindex>
// ENABLED:           [[VAR_reshape_:%.+]] = tensor.reshape [[PARAM_0_]]([[VAR_from_elements_]]) : (tensor<?x3x4xf32>, tensor<2xindex>) -> tensor<?x12xf32>
// ENABLED:           return [[VAR_reshape_]] : tensor<?x12xf32>
// ENABLED:         }
}

// -----

func.func @reshape_runtime_shape(%arg0: tensor<2x3x4xf32>,
    %shape: tensor<2xi64>) -> tensor<?x?xf32> {
  %0 = "onnx.Reshape"(%arg0, %shape) {allowzero = 0 : si64}
      : (tensor<2x3x4xf32>, tensor<2xi64>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>

// ENABLED-LABEL:  func.func @reshape_runtime_shape
// ENABLED-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xf32>, [[PARAM_1_:%.+]]: tensor<2xi64>) -> tensor<?x?xf32> {
// ENABLED-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// ENABLED-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// ENABLED-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// ENABLED-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// ENABLED-DAG:       [[CST_24_:%.+]] = arith.constant 24 : index
// ENABLED-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// ENABLED:           [[VAR_extracted_:%.+]] = tensor.extract [[PARAM_1_]]{{.}}[[CST_0_]]{{.}} : tensor<2xi64>
// ENABLED:           [[VAR_0_:%.+]] = arith.index_cast [[VAR_extracted_]] : i64 to index
// ENABLED:           [[VAR_1_:%.+]] = arith.cmpi eq, [[VAR_0_]], [[CST_0_]] : index
// ENABLED:           [[VAR_2_:%.+]] = arith.select [[VAR_1_]], [[CST_2_]], [[VAR_0_]] : index
// ENABLED:           [[VAR_3_:%.+]] = arith.cmpi eq, [[VAR_2_]], [[CST_minus_1_]] : index
// ENABLED-DAG:       [[VAR_4_:%.+]] = arith.select [[VAR_3_]], [[CST_1_]], [[VAR_2_]] : index
// ENABLED-DAG:       [[VAR_extracted_0_:%.+]] = tensor.extract [[PARAM_1_]]{{.}}[[CST_1_]]{{.}} : tensor<2xi64>
// ENABLED:           [[VAR_5_:%.+]] = arith.index_cast [[VAR_extracted_0_]] : i64 to index
// ENABLED:           [[VAR_6_:%.+]] = arith.cmpi eq, [[VAR_5_]], [[CST_0_]] : index
// ENABLED:           [[VAR_7_:%.+]] = arith.select [[VAR_6_]], [[CST_3_]], [[VAR_5_]] : index
// ENABLED:           [[VAR_8_:%.+]] = arith.cmpi eq, [[VAR_7_]], [[CST_minus_1_]] : index
// ENABLED:           [[VAR_9_:%.+]] = arith.select [[VAR_8_]], [[CST_1_]], [[VAR_7_]] : index
// ENABLED-DAG:       [[VAR_10_:%.+]] = arith.muli [[VAR_4_]], [[VAR_9_]] : index
// ENABLED-DAG:       [[VAR_extracted_1_:%.+]] = tensor.extract [[PARAM_1_]]{{.}}[[CST_0_]]{{.}} : tensor<2xi64>
// ENABLED-NOT: separator of consecutive DAGs
// ENABLED-DAG:       [[VAR_11_:%.+]] = arith.index_cast [[VAR_extracted_1_]] : i64 to index
// ENABLED-DAG:       [[VAR_12_:%.+]] = arith.floordivsi [[CST_24_]], [[VAR_10_]] : index
// ENABLED:           [[VAR_13_:%.+]] = arith.cmpi eq, [[VAR_11_]], [[CST_minus_1_]] : index
// ENABLED-DAG:       [[VAR_14_:%.+]] = arith.select [[VAR_13_]], [[VAR_12_]], [[VAR_2_]] : index
// ENABLED-DAG:       [[VAR_extracted_2_:%.+]] = tensor.extract [[PARAM_1_]]{{.}}[[CST_1_]]{{.}} : tensor<2xi64>
// ENABLED-NOT: separator of consecutive DAGs
// ENABLED-DAG:       [[VAR_15_:%.+]] = arith.index_cast [[VAR_extracted_2_]] : i64 to index
// ENABLED-DAG:       [[VAR_16_:%.+]] = arith.floordivsi [[CST_24_]], [[VAR_10_]] : index
// ENABLED:           [[VAR_17_:%.+]] = arith.cmpi eq, [[VAR_15_]], [[CST_minus_1_]] : index
// ENABLED:           [[VAR_18_:%.+]] = arith.select [[VAR_17_]], [[VAR_16_]], [[VAR_7_]] : index
// ENABLED:           [[VAR_from_elements_:%.+]] = tensor.from_elements [[VAR_14_]], [[VAR_18_]] : tensor<2xindex>
// ENABLED:           [[VAR_reshape_:%.+]] = tensor.reshape [[PARAM_0_]]([[VAR_from_elements_]]) : (tensor<2x3x4xf32>, tensor<2xindex>) -> tensor<?x?xf32>
// ENABLED:           return [[VAR_reshape_]] : tensor<?x?xf32>
// ENABLED:         }
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

// -----

// GROUND-THIS: -r=--O0 -t="--O0 --use-linalg-path" --shape-info=0:2x3x4
func.func @reshape_identity_dynamic(
    %arg0: tensor<?x?x?xf32> {onnx.name = "input"})
    -> (tensor<?x?x?xf32> {onnx.name = "output"}) {
  %minus_one = onnx.Constant dense<-1> : tensor<1xi64>
  %dim1 = "onnx.Dim"(%arg0) {axis = 1 : si64}
      : (tensor<?x?x?xf32>) -> tensor<1xi64>
  %dim2 = "onnx.Dim"(%arg0) {axis = 2 : si64}
      : (tensor<?x?x?xf32>) -> tensor<1xi64>
  %shape = "onnx.Concat"(%minus_one, %dim1, %dim2) {axis = 0 : si64}
      : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
  %0 = "onnx.Reshape"(%arg0, %shape) {allowzero = 0 : si64}
      : (tensor<?x?x?xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>

  // ENABLED-LABEL: func.func @reshape_identity_dynamic(
  // ENABLED-NOT: tensor.reshape
  // ENABLED: return %arg0 : tensor<?x?x?xf32>

  // DISABLED-LABEL: func.func @reshape_identity_dynamic(
  // DISABLED: %[[RESULT:.*]] = "onnx.Reshape"
  // DISABLED: return %[[RESULT]] : tensor<?x?x?xf32>
}
