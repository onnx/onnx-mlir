/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------- SpecializedKernelOpInterface.hpp
//------------------===//
//===------------- Specialized Kernel Op Interface Definition -------------===//
//
// Copyright 2020-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the declaration of the SpecializedKernel Op Interface.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_SPECIALIZED_KERNEL_INTERFACE_H
#define ONNX_MLIR_SPECIALIZED_KERNEL_INTERFACE_H

#include <map>
#include <string>

#include "mlir/IR/OpDefinition.h"

namespace mlir {

/// Include the auto-generated declarations.
#include "src/Interface/SpecializedKernelOpInterface.hpp.inc"

} // end namespace mlir
#endif
