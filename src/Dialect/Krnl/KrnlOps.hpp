/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------- KrnlOps.hpp - Krnl Operations ------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declarations of krnl operations.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "src/Interface/SpecializedKernelOpInterface.hpp"
#include "llvm/ADT/TypeSwitch.h"

#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlTypes.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/Mlir/IndexExpr.hpp"

namespace mlir {

class KrnlOpsDialect : public Dialect {
public:
  KrnlOpsDialect(MLIRContext *context);
  KrnlOpsDialect() = delete;

  static StringRef getDialectNamespace() { return "krnl"; }

  /// Parse a type registered to this dialect.
  Type parseType(DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect.
  void printType(Type type, DialectAsmPrinter &os) const override;
};

} // namespace mlir

#define GET_OP_CLASSES
#include "src/Dialect/Krnl/KrnlOps.hpp.inc"
