/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ ZHighOps.hpp - ZHigh Operations -------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file defines the ZHigh operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <map>
#include <string>

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "src/Interface/ShapeInferenceOpInterface.hpp"

namespace mlir {

//===----------------------------------------------------------------------===//
// Traits

namespace OpTrait {
namespace impl {
LogicalResult verifySameOperandsAndResultLayout(Operation *op);
}

/// This class provides verification for ops that are known to have the same
/// operand and result layout.
template <typename ConcreteType>
class SameOperandsAndResultLayout
    : public TraitBase<ConcreteType, SameOperandsAndResultLayout> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifySameOperandsAndResultLayout(op);
  }
};
} // namespace OpTrait
} // namespace mlir

namespace onnx_mlir {
namespace zhigh {

class ZHighDialect : public mlir::Dialect {
public:
  ZHighDialect(mlir::MLIRContext *context);

  /// Parse an instance of an attribute registered to the zhigh dialect.
  mlir::Attribute parseAttribute(
      mlir::DialectAsmParser &parser, mlir::Type type) const override;

  /// Print an instance of an attribute registered to the zhigh dialect.
  void printAttribute(
      mlir::Attribute attr, mlir::DialectAsmPrinter &printer) const override;

  /// Provide a utility accessor to the dialect namespace. This is used by
  /// several utilities for casting between dialects.
  static mlir::StringRef getDialectNamespace() { return "zhigh"; }
};

} // namespace zhigh
} // namespace onnx_mlir

/// Include the auto-generated header file containing the declarations of the
/// ZHigh operations.
#define GET_ATTRDEF_CLASSES
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighAttributes.hpp.inc"

#define GET_OP_CLASSES
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp.inc"
