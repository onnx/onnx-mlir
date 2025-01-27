/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====--------------- Diagnostic.hpp - Diagnostic Utilities ---------------===//
//
// Copyright 2022-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains common utilities for diagnosing incorrect ONNX operations.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_DIAGNOSTIC_H
#define ONNX_MLIR_DIAGNOSTIC_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/Twine.h"
#include <type_traits>

namespace onnx_mlir {

/// This convenience class groups diagnostic API used to emit error messages for
/// ONNX operators.
class Diagnostic {
public:
  Diagnostic() = delete;

  template <typename T>
  class Range {
    static_assert(std::is_arithmetic<T>::value, "Expecting an arithmetic type");
    friend class Diagnostic;
    T min;
    T max;

  public:
    // Range is used in error situations, so having an assert is not very useful
    // as that assert may crash the program instead of reporting the error
    // condition. New approach is to report the error with an additional
    // warning.
    Range(T min, T max) : min(min), max(max) {
      if (!isValid())
        llvm::errs() << "Warning: badly formed range(min=" << min
                     << ", max=" << max << ")\n";
    }
    bool isValid() { return min <= max; }
  };

  /// Diagnostic message for attribute value outside of a supplied range.
  template <typename T>
  static mlir::LogicalResult emitAttributeOutOfRangeError(mlir::Operation &op,
      const llvm::Twine &attrName, T attrVal, Range<T> range);

  /// Verifies whether 2 inputs have the same rank.
  template <typename T>
  static mlir::LogicalResult emitInputsMustHaveSameRankError(
      mlir::Operation &op, const llvm::Twine &inputName1, T rank1,
      const llvm::Twine &inputName2, T rank2);

  /// Diagnostic message for operand with unexpected rank.
  static mlir::LogicalResult emitOperandHasUnexpectedRankError(
      mlir::Operation &op, mlir::Value &operand, uint64_t operandRank,
      mlir::StringRef expectedRank);

  /// Verifies whether two dimensions from two inputs have the same value.
  static mlir::LogicalResult emitDimensionsMustHaveSameValueError(
      mlir::Operation &op, const llvm::Twine &inputName1, uint64_t axisDim1,
      int64_t dim1, const llvm::Twine &inputName2, uint64_t axisDim2,
      int64_t dim2);

  /// Diagnostic message for dimension with unexpected value.
  static mlir::LogicalResult emitDimensionHasUnexpectedValueError(
      mlir::Operation &op, mlir::Value &operand, int64_t index, int64_t value,
      mlir::StringRef expectedValue);

  /// Return the name of the given value.
  static std::string getName(mlir::Value &v);
};

} // namespace onnx_mlir
#endif
