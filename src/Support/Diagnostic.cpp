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

  Twine msg(op.getName().getStringRef() + " ");
  return emitError(op.getLoc(), msg.concat("'" + attrName + "'")
                                    .concat(" value is ")
                                    .concat(std::to_string(attrVal))
                                    .concat(", accepted range is [")
                                    .concat(std::to_string(validRange.min))
                                    .concat(", ")
                                    .concat(std::to_string(validRange.max))
                                    .concat("]"));
};

template <typename T>
mlir::LogicalResult Diagnostic::inputsMustHaveSameRank(Operation &op,
    const llvm::Twine &inputName1, T rank1, const llvm::Twine &inputName2,
    T rank2) {
  static_assert(std::is_arithmetic<T>::value, "Expecting an arithmetic type");

  llvm::Twine msg(op.getName().getStringRef() + " ");
  return emitError(
      op.getLoc(), msg.concat("'" + inputName1 + "'")
                       .concat(" has rank ")
                       .concat(std::to_string(rank1))
                       .concat(", '" + inputName2 + "'")
                       .concat(" has rank ")
                       .concat(std::to_string(rank2))
                       .concat(". The two inputs must have the same rank."));
}

// Template instantiations - keep at the end of the file.
template mlir::LogicalResult Diagnostic::attributeOutOfRange(
    Operation &, const Twine &, int64_t, Range<int64_t>);
template mlir::LogicalResult Diagnostic::inputsMustHaveSameRank(
    Operation &, const Twine &, int64_t, const Twine &, int64_t);

} // namespace onnx_mlir
