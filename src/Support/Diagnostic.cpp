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

namespace onnx_mlir {

template <typename T>
mlir::LogicalResult Diagnostic::attributeOutOfRange(mlir::Operation &op,
    const llvm::Twine &attrName, T attrVal, Range<T> validRange) {
  static_assert(std::is_arithmetic<T>::value, "Expecting an arithmetic type");

  llvm::Twine msg(op.getName().getStringRef() + " ");
  return emitError(op.getLoc(), msg.concat("'" + attrName + "'")
                                    .concat(" value is ")
                                    .concat(std::to_string(attrVal))
                                    .concat(", accepted range is [")
                                    .concat(std::to_string(validRange.min))
                                    .concat(", ")
                                    .concat(std::to_string(validRange.max))
                                    .concat("]"));
};

// Template instantiations - keep at the end of the file.
template mlir::LogicalResult Diagnostic::attributeOutOfRange(
    mlir::Operation &op, const llvm::Twine &attrName, int64_t attrVal,
    Range<int64_t> validRange);

} // namespace onnx_mlir
