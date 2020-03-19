//===---- ShapeInferenceInterface.cpp - Definition for ShapeInference -----===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the declarations of the shape inference interfaces defined
// in ShapeInferenceInterface.td.
//
//===----------------------------------------------------------------------===//

#include "src/Interface/ShapeInferenceInterface.hpp"

namespace mlir {

/// Include the auto-generated declarations.
#include "src/Interface/shape_inference.cpp.inc"

}  // end namespace mlir
