//===- shape_inference_interface.hpp - Definition for ShapeInference --------=//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the declarations of the shape inference interfaces defined
// in ShapeInferenceInterface.td.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <map>
#include <string>

#include "mlir/IR/OpDefinition.h"

namespace mlir {

/// Include the auto-generated declarations.
#include "src/dialect/onnx/promotable_const_operand.hpp.inc"

}  // end namespace mlir