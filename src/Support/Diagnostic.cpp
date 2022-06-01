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
LogicalResult Diagnostic::emitAttributeOutOfRangeError(Operation &op,
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

template <typename T>
LogicalResult Diagnostic::emitInputsMustHaveSameRankError(Operation &op,
    const llvm::Twine &inputName1, T rank1, const llvm::Twine &inputName2,
    T rank2) {
  static_assert(std::is_arithmetic<T>::value, "Expecting an arithmetic type");

  llvm::Twine msg(op.getName().getStringRef() + ": ");
  return emitError(
      op.getLoc(), msg.concat("'" + inputName1 + "'")
                       .concat(" has rank ")
                       .concat(std::to_string(rank1))
                       .concat(", '" + inputName2 + "'")
                       .concat(" has rank ")
                       .concat(std::to_string(rank2))
                       .concat(". The two inputs must have the same rank."));
}

LogicalResult Diagnostic::emitOperandHasUnexpectedRankError(Operation &op,
    Value &operand, uint64_t operandRank, StringRef expectedRank) {
  llvm::Twine msg(op.getName().getStringRef() + ": ");
  return emitError(op.getLoc(), msg.concat("operand '" + getName(operand) + "'")
                                    .concat(" has rank ")
                                    .concat(std::to_string(operandRank))
                                    .concat(", rank should be ")
                                    .concat(expectedRank));
}

LogicalResult Diagnostic::emitDimensionHasUnexpectedValueError(Operation &op,
    Value &operand, int64_t index, int64_t value, StringRef expectedValue) {
  llvm::Twine msg(op.getName().getStringRef() + ": ");
  return emitError(op.getLoc(), msg.concat("operand '" + getName(operand) + "'")
                                    .concat(" has dimension at index ")
                                    .concat(std::to_string(index))
                                    .concat(" with value ")
                                    .concat(std::to_string(value))
                                    .concat(", value should be ")
                                    .concat(expectedValue));
}

std::string Diagnostic::getName(Value &v) {
  std::string str;
  llvm::raw_string_ostream os(str);
  v.print(os);
  return str;
}

// Template instantiations - keep at the end of the file.
template LogicalResult Diagnostic::emitAttributeOutOfRangeError(
    Operation &, const llvm::Twine &, int64_t, Range<int64_t>);
template LogicalResult Diagnostic::emitInputsMustHaveSameRankError(
    Operation &, const llvm::Twine &, int64_t, const llvm::Twine &, int64_t);

} // namespace onnx_mlir
