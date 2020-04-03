//===------------ PromotableConstOperandsOpInterface.hpp --------------===//
//===-------- Promotable Const Operands Op Interface Definition -----------===//
//
// Copyright 2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the declaration of the promotable const operands op
// interface.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <map>
#include <string>

#include "mlir/IR/OpDefinition.h"

namespace mlir {

/// Include the auto-generated declarations.
#include "src/Interface/PromotableConstOperandsOpInterface.hpp.inc"

}  // end namespace mlir