/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------- HasOnnxSubgraphOpInterface.hpp ------------------===//
//===------------- Has Onnx Subgraph Op Interface Definition -------------===//
//
// Copyright 2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the declaration of the HasOnnxSubgraph Op Interface.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <map>
#include <string>

#include "mlir/IR/OpDefinition.h"

namespace mlir {

/// Include the auto-generated declarations.
#include "src/Interface/HasOnnxSubgraphOpInterface.hpp.inc"

} // end namespace mlir
