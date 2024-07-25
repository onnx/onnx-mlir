/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------- HasOnnxSubgraphOpInterface.hpp ------------------===//
//===------------- Has Onnx Subgraph Op Interface Definition -------------===//
//
// Copyright 2020-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the declaration of the HasOnnxSubgraph Op Interface.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_HAS_ONNX_INTERFACE_H
#define ONNX_MLIR_HAS_ONNX_INTERFACE_H

#include <map>
#include <string>

#include "mlir/IR/OpDefinition.h"

namespace mlir {

/// Include the auto-generated declarations.
#include "src/Interface/HasOnnxSubgraphOpInterface.hpp.inc"

} // end namespace mlir
#endif
