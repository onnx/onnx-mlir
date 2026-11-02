// RUN: onnx-mlir-opt --convert-onnx-to-linalg='linalg-ops=onnx.Shape' %s -split-input-file | FileCheck %s --check-prefix=ENABLED
// RUN: onnx-mlir-opt --convert-onnx-to-linalg='linalg-ops=NONE' %s -split-input-file | FileCheck %s --check-prefix=DISABLED

// The default range returns the complete shape. Static dimensions fold to i64
// constants, while dynamic dimensions are queried and cast from index to i64.
func.func @shape_default(
    %arg0: tensor<?x3x2xf32>) -> tensor<3xi64> {
  %0 = "onnx.Shape"(%arg0) :
      (tensor<?x3x2xf32>) -> tensor<3xi64>
  return %0 : tensor<3xi64>
}

// ENABLED-LABEL:  func.func @shape_default
// ENABLED-SAME:   ([[PARAM_0_:%.+]]: tensor<?x3x2xf32>) -> tensor<3xi64> {
// ENABLED-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i64
// ENABLED-DAG:       [[CST_3_:%.+]] = arith.constant 3 : i64
// ENABLED-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// ENABLED:           [[VAR_dim_:%.+]] = tensor.dim [[PARAM_0_]], [[CST_0_]] : tensor<?x3x2xf32>
// ENABLED:           [[VAR_0_:%.+]] = arith.index_cast [[VAR_dim_]] : index to i64
// ENABLED:           [[VAR_from_elements_:%.+]] = tensor.from_elements [[VAR_0_]], [[CST_3_]], [[CST_2_]] : tensor<3xi64>
// ENABLED:           return [[VAR_from_elements_]] : tensor<3xi64>
// ENABLED:         }

// DISABLED-LABEL:  func.func @shape_default
// DISABLED-SAME:   ([[PARAM_0_:%.+]]: tensor<?x3x2xf32>) -> tensor<3xi64> {
// DISABLED:           [[VAR_0_:%.+]] = "onnx.Shape"([[PARAM_0_]]) <{start = 0 : si64}> : (tensor<?x3x2xf32>) -> tensor<3xi64>
// DISABLED:           return [[VAR_0_]] : tensor<3xi64>
// DISABLED:         }

// -----

// A fully dynamic input queries every selected dimension at runtime.
func.func @shape_all_dynamic(
    %arg0: tensor<?x?x?xf32>) -> tensor<3xi64> {
  %0 = "onnx.Shape"(%arg0) :
      (tensor<?x?x?xf32>) -> tensor<3xi64>
  return %0 : tensor<3xi64>
}

// ENABLED-LABEL:  func.func @shape_all_dynamic
// ENABLED-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?xf32>) -> tensor<3xi64> {
// ENABLED-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// ENABLED-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// ENABLED-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// ENABLED:           [[VAR_dim_:%.+]] = tensor.dim [[PARAM_0_]], [[CST_0_]] : tensor<?x?x?xf32>
// ENABLED-DAG:       [[VAR_0_:%.+]] = arith.index_cast [[VAR_dim_]] : index to i64
// ENABLED-DAG:       [[VAR_dim_0_:%.+]] = tensor.dim [[PARAM_0_]], [[CST_1_]] : tensor<?x?x?xf32>
// ENABLED-NOT: separator of consecutive DAGs
// ENABLED-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[VAR_dim_0_]] : index to i64
// ENABLED-DAG:       [[VAR_dim_1_:%.+]] = tensor.dim [[PARAM_0_]], [[CST_2_]] : tensor<?x?x?xf32>
// ENABLED:           [[VAR_2_:%.+]] = arith.index_cast [[VAR_dim_1_]] : index to i64
// ENABLED:           [[VAR_from_elements_:%.+]] = tensor.from_elements [[VAR_0_]], [[VAR_1_]], [[VAR_2_]] : tensor<3xi64>
// ENABLED:           return [[VAR_from_elements_]] : tensor<3xi64>
// ENABLED:         }

// -----

// Explicit start and end select [start, end). All selected dimensions are
// static, so the complete result folds to a constant tensor.
func.func @shape_explicit_range(
    %arg0: tensor<2x3x4x5xf32>) -> tensor<2xi64> {
  %0 = "onnx.Shape"(%arg0) {start = 1 : si64, end = 3 : si64} :
      (tensor<2x3x4x5xf32>) -> tensor<2xi64>
  return %0 : tensor<2xi64>
}

// ENABLED-LABEL:  func.func @shape_explicit_range
// ENABLED-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4x5xf32>) -> tensor<2xi64> {
// ENABLED:           [[VAR_cst_:%.+]] = arith.constant dense<[3, 4]> : tensor<2xi64>
// ENABLED:           return [[VAR_cst_]] : tensor<2xi64>
// ENABLED:         }

// -----

// When only start is present, end defaults to the input rank. The selected
// range mixes dynamic and static dimensions and starts at a nonzero index.
func.func @shape_start_only(
    %arg0: tensor<2x?x4x?xf32>) -> tensor<3xi64> {
  %0 = "onnx.Shape"(%arg0) {start = 1 : si64} :
      (tensor<2x?x4x?xf32>) -> tensor<3xi64>
  return %0 : tensor<3xi64>
}

// ENABLED-LABEL:  func.func @shape_start_only
// ENABLED-SAME:   ([[PARAM_0_:%.+]]: tensor<2x?x4x?xf32>) -> tensor<3xi64> {
// ENABLED-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// ENABLED-DAG:       [[CST_4_:%.+]] = arith.constant 4 : i64
// ENABLED-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// ENABLED:           [[VAR_dim_:%.+]] = tensor.dim [[PARAM_0_]], [[CST_1_]] : tensor<2x?x4x?xf32>
// ENABLED-DAG:       [[VAR_0_:%.+]] = arith.index_cast [[VAR_dim_]] : index to i64
// ENABLED-DAG:       [[VAR_dim_0_:%.+]] = tensor.dim [[PARAM_0_]], [[CST_3_]] : tensor<2x?x4x?xf32>
// ENABLED:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_dim_0_]] : index to i64
// ENABLED:           [[VAR_from_elements_:%.+]] = tensor.from_elements [[VAR_0_]], [[CST_4_]], [[VAR_1_]] : tensor<3xi64>
// ENABLED:           return [[VAR_from_elements_]] : tensor<3xi64>
// ENABLED:         }

// -----

// When only end is present, start defaults to zero. A negative end counts
// backward from the input rank.
func.func @shape_end_only(
    %arg0: tensor<?x3x?x5xf32>) -> tensor<3xi64> {
  %0 = "onnx.Shape"(%arg0) {end = -1 : si64} :
      (tensor<?x3x?x5xf32>) -> tensor<3xi64>
  return %0 : tensor<3xi64>
}

// ENABLED-LABEL:  func.func @shape_end_only
// ENABLED-SAME:   ([[PARAM_0_:%.+]]: tensor<?x3x?x5xf32>) -> tensor<3xi64> {
// ENABLED-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// ENABLED-DAG:       [[CST_3_:%.+]] = arith.constant 3 : i64
// ENABLED-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// ENABLED:           [[VAR_dim_:%.+]] = tensor.dim [[PARAM_0_]], [[CST_0_]] : tensor<?x3x?x5xf32>
// ENABLED-DAG:       [[VAR_0_:%.+]] = arith.index_cast [[VAR_dim_]] : index to i64
// ENABLED-DAG:       [[VAR_dim_0_:%.+]] = tensor.dim [[PARAM_0_]], [[CST_2_]] : tensor<?x3x?x5xf32>
// ENABLED:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_dim_0_]] : index to i64
// ENABLED:           [[VAR_from_elements_:%.+]] = tensor.from_elements [[VAR_0_]], [[CST_3_]], [[VAR_1_]] : tensor<3xi64>
// ENABLED:           return [[VAR_from_elements_]] : tensor<3xi64>
// ENABLED:         }

// -----

// Negative start and end values count backward from the input rank.
func.func @shape_negative_range(
    %arg0: tensor<2x3x4x5xf32>) -> tensor<2xi64> {
  %0 = "onnx.Shape"(%arg0) {start = -3 : si64, end = -1 : si64} :
      (tensor<2x3x4x5xf32>) -> tensor<2xi64>
  return %0 : tensor<2xi64>
}

// ENABLED-LABEL:  func.func @shape_negative_range
// ENABLED-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4x5xf32>) -> tensor<2xi64> {
// ENABLED:           [[VAR_cst_:%.+]] = arith.constant dense<[3, 4]> : tensor<2xi64>
// ENABLED:           return [[VAR_cst_]] : tensor<2xi64>
// ENABLED:         }

// -----

// Out-of-range start and end values are clamped to [0, inputRank].
func.func @shape_clamped_range(
    %arg0: tensor<2x3x4xf32>) -> tensor<3xi64> {
  %0 = "onnx.Shape"(%arg0) {start = -10 : si64, end = 10 : si64} :
      (tensor<2x3x4xf32>) -> tensor<3xi64>
  return %0 : tensor<3xi64>
}

// ENABLED-LABEL:  func.func @shape_clamped_range
// ENABLED-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3x4xf32>) -> tensor<3xi64> {
// ENABLED:           [[VAR_cst_:%.+]] = arith.constant dense<[2, 3, 4]> : tensor<3xi64>
// ENABLED:           return [[VAR_cst_]] : tensor<3xi64>
// ENABLED:         }

// -----

// Equal start and end values select no dimensions.
func.func @shape_empty_range(
    %arg0: tensor<2x3xf32>) -> tensor<0xi64> {
  %0 = "onnx.Shape"(%arg0) {start = 1 : si64, end = 1 : si64} :
      (tensor<2x3xf32>) -> tensor<0xi64>
  return %0 : tensor<0xi64>
}

// ENABLED-LABEL:  func.func @shape_empty_range
// ENABLED-SAME:   ([[PARAM_0_:%.+]]: tensor<2x3xf32>) -> tensor<0xi64> {
// ENABLED:           [[VAR_cst_:%.+]] = arith.constant dense<> : tensor<0xi64>
// ENABLED:           return [[VAR_cst_]] : tensor<0xi64>
// ENABLED:         }

// -----

// A scalar has rank zero, so its complete shape is an empty i64 tensor.
func.func @shape_scalar(
    %arg0: tensor<f32>) -> tensor<0xi64> {
  %0 = "onnx.Shape"(%arg0) :
      (tensor<f32>) -> tensor<0xi64>
  return %0 : tensor<0xi64>
}

// ENABLED-LABEL:  func.func @shape_scalar
// ENABLED-SAME:   ([[PARAM_0_:%.+]]: tensor<f32>) -> tensor<0xi64> {
// ENABLED:           [[VAR_cst_:%.+]] = arith.constant dense<> : tensor<0xi64>
// ENABLED:           return [[VAR_cst_]] : tensor<0xi64>
// ENABLED:         }

// -----

// Without a known input rank, the number of shape values is unknown and the
// operation is intentionally not lowered.
func.func @shape_unranked(
    %arg0: tensor<*xf32>) -> tensor<*xi64> {
  %0 = "onnx.Shape"(%arg0) :
      (tensor<*xf32>) -> tensor<*xi64>
  return %0 : tensor<*xi64>
}

// ENABLED-LABEL:  func.func @shape_unranked
// ENABLED-SAME:   ([[PARAM_0_:%.+]]: tensor<*xf32>) -> tensor<*xi64> {
// ENABLED:           [[VAR_0_:%.+]] = "onnx.Shape"([[PARAM_0_]]) <{start = 0 : si64}> : (tensor<*xf32>) -> tensor<*xi64>
// ENABLED:           return [[VAR_0_]] : tensor<*xi64>
// ENABLED:         }
