/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ ZLowOps.cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file defines the ZLow operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"

#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace zlow {

//===----------------------------------------------------------------------===//
// ZLowDialect
//===----------------------------------------------------------------------===//

void ZLowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.cpp.inc"
      >();
}

} // namespace zlow
} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.cpp.inc"

#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowDialect.cpp.inc"
