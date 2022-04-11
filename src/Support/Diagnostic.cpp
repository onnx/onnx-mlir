/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====--------------- Diagnostic.cpp - Diagnostic Utilities ---------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the implementation for diagnosing utilities.
//
//===----------------------------------------------------------------------===//

#include "src/Support/Diagnostic.hpp"

using namespace mlir;

namespace onnx_mlir {

template <typename T>
LogicalResult Diagnostic::attributeOutOfRange(Operation &op,
    const llvm::Twine &attrName, T attrVal, Range<T> validRange) {
  static_assert(std::is_arithmetic<T>::value, "Expecting an arithmetic type");

  llvm::Twine msg(op.getName().getStringRef() + ": ");
  return emitError(op.getLoc(), msg.concat("'" + attrName + "'")
                                    .concat(" value is ")
                                    .concat(std::to_string(attrVal))
                                    .concat(", accepted range is [")
                                    .concat(std::to_string(validRange.min))
                                    .concat(", ")
                                    .concat(std::to_string(validRange.max))
                                    .concat("]"));
}

LogicalResult Diagnostic::operandHasUnexpectedRank(Operation &op,
    Value &operand, uint64_t operandRank, uint64_t expectedRank) {
  llvm::Twine msg(op.getName().getStringRef() + ": ");
  return emitError(op.getLoc(), msg.concat("operand '" + getName(operand) + "'")
                                    .concat(" must have rank ")
                                    .concat(std::to_string(expectedRank))
                                    .concat(" instead of ")
                                    .concat(std::to_string(operandRank)));
}

LogicalResult Diagnostic::operandHasUnexpectedDimensionValue(Operation &op,
    Value &operand, uint64_t operandDimension, uint64_t dimensionValue,
    uint64_t expectedDimensionValue) {
  llvm::Twine msg(op.getName().getStringRef() + ": ");
  return emitError(
      op.getLoc(), msg.concat("operand '" + getName(operand) + "'")
                       .concat(" has value ")
                       .concat(std::to_string(dimensionValue))
                       .concat(" instead of ")
                       .concat(std::to_string(expectedDimensionValue))
                       .concat(" for dimension ")
                       .concat(std::to_string(operandDimension)));
}

std::string Diagnostic::getName(Value &v) {
  std::string str;
  llvm::raw_string_ostream os(str);
  v.print(os);
  return str;
}

// Template instantiations - keep at the end of the file.
template mlir::LogicalResult Diagnostic::attributeOutOfRange(
    mlir::Operation &op, const llvm::Twine &attrName, int64_t attrVal,
    Range<int64_t> validRange);

} // namespace onnx_mlir
