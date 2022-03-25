/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ ZLowOps.hpp - ONNX Operations ---------------------===//
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

#include "ZLowOps.hpp"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ZLowDialect
//===----------------------------------------------------------------------===//

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
ZLowDialect::ZLowDialect(MLIRContext *ctx)
    : Dialect(getDialectNamespace(), ctx, TypeID::get<ZLowDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "Dialect/ZLow/ZLowOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Dialect/ZLow/ZLowOps.cpp.inc"
