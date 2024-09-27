/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ ResultTypeInferenceOpInterface.hpp --------------===//
//===------- Infer Data Type for Result of Op Interface Definition -------===//
//
// Copyright 2020-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the declaration of the data type reference for op
// interface.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_RESULT_TYPE_INFERENCE_H
#define ONNX_MLIR_RESULT_TYPE_INFERENCE_H

#include <map>
#include <string>

#include "mlir/IR/OpDefinition.h"

namespace mlir {

/// Include the auto-generated declarations.
#include "src/Interface/ResultTypeInferenceOpInterface.hpp.inc"

} // end namespace mlir
#endif
