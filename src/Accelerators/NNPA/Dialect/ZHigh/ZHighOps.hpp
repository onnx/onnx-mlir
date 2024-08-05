/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ ZHighOps.hpp - ZHigh Operations -------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file defines the ZHigh operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ZHIGH_H
#define ONNX_MLIR_ZHIGH_H

#include <map>
#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

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

/// Include the auto-generated header files containing the declarations of the
/// ZHigh dialect and operations.
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighDialect.hpp.inc"

#define GET_ATTRDEF_CLASSES
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighAttributes.hpp.inc"

#define GET_OP_CLASSES
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp.inc"
#endif