/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===========-- ModelHelper.hpp - Helper function for building models -=======//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains helper functions.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <string>

#include "llvm/ADT/SmallVector.h"

#include "src/Compiler/CompilerUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Runtime/ExecutionSession.hpp"
#include "src/Runtime/OMTensorHelper.h"


// Helper function
mlir::ONNXConstantOp
buildONNXConstantOp(mlir::MLIRContext *ctx, mlir::OpBuilder builder,
    OMTensor *omt, mlir::RankedTensorType resultType);
