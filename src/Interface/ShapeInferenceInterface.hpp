//===---- ShapeInferenceInterface.hpp - Definition for ShapeInference -----===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the declarations of the shape inference interfaces defined
// in ShapeInferenceInterface.td.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Function.h"

namespace mlir {

/// Include the auto-generated declarations.
#include "src/Interface/ShapeInference.hpp.inc"

} // end namespace mlir
