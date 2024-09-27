/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------- KrnlOps.hpp - Krnl Operations ------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declarations of krnl operations.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_KRNL_H
#define ONNX_MLIR_KRNL_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlTypes.hpp"
#include "src/Interface/SpecializedKernelOpInterface.hpp"

#include "src/Dialect/Krnl/KrnlDialect.hpp.inc"

#define GET_OP_CLASSES
#include "src/Dialect/Krnl/KrnlOps.hpp.inc"
#endif
