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
LogicalResult Diagnostic::emitAttributeOutOfRangeError(
    Operation &op, const llvm::Twine &attrName, T attrVal, Range<T> range) {
  static_assert(std::is_arithmetic<T>::value, "Expecting an arithmetic type");

  llvm::Twine msg(op.getName().getStringRef() + ": ");
  std::string rangeMessage =
      range.isValid() ? "" : " <<Warning, ill-formed range>>";
  return emitError(op.getLoc(), msg.concat("'" + attrName + "'")
                                    .concat(" value is ")
                                    .concat(std::to_string(attrVal))
                                    .concat(", accepted range is [")
                                    .concat(std::to_string(range.min))
                                    .concat(", ")
                                    .concat(std::to_string(range.max))
                                    .concat("]")
                                    .concat(rangeMessage));
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

LogicalResult Diagnostic::emitDimensionsMustHaveSameValueError(Operation &op,
    const llvm::Twine &inputName1, uint64_t axisDim1, int64_t dim1,
    const llvm::Twine &inputName2, uint64_t axisDim2, int64_t dim2) {
  llvm::Twine msg(op.getName().getStringRef() + ": ");
  return emitError(op.getLoc(),
      msg.concat("'" + inputName1 + "'")
          .concat(" dimension at index ")
          .concat(std::to_string(axisDim1))
          .concat(" has value " + std::to_string(dim1))
          .concat(", '" + inputName2 + "'")
          .concat(" dimension at index ")
          .concat(std::to_string(axisDim2))
          .concat(" has value " + std::to_string(dim2))
          .concat(". The two dimensions must have the same value."));
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
  // print without calling verify() on v's defining op to
  // avoid infinite recursion when getName() is called from an emit
  // method called from verify()
  v.print(os, OpPrintingFlags().assumeVerified());
  return str;
}

// Template instantiations - keep at the end of the file.
template LogicalResult Diagnostic::emitAttributeOutOfRangeError(
    Operation &, const llvm::Twine &, int64_t, Range<int64_t>);
template LogicalResult Diagnostic::emitInputsMustHaveSameRankError(
    Operation &, const llvm::Twine &, int64_t, const llvm::Twine &, int64_t);
template LogicalResult Diagnostic::emitInputsMustHaveSameRankError(
    Operation &, const llvm::Twine &, uint64_t, const llvm::Twine &, uint64_t);

} // namespace onnx_mlir
