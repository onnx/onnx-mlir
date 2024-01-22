/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ DialectBuilder.hpp - TOSA dialect builder --------------------===//
//
// Copyright (c) 2022-2023 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file contains the dialect build for the TOSA dialect. Uses the same
// implementation as ONNXToMhlo with minor differences.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"

#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/Mlir/IndexExprBuilder.hpp"

namespace onnx_mlir {

// =============================================================================
// TOSA Builder
// =============================================================================

struct TosaBuilder : DialectBuilder {
  TosaBuilder(mlir::Location loc) : DialectBuilder(loc) {}
  TosaBuilder(mlir::PatternRewriter &b, mlir::Location loc)
      : DialectBuilder(b, loc), patternRewriter(&b) {}
  TosaBuilder(const DialectBuilder &db) : DialectBuilder(db) {}
  virtual ~TosaBuilder() {}

  std::optional<mlir::Value> gather(mlir::Value resultValue,
      mlir::Value inputValue, mlir::Value indicesValue, int32_t batchDims,
      int32_t axis);
  template <typename T>
  mlir::Value binaryOp(mlir::Value &lhs, mlir::Value &rhs);
  mlir::Value mul(mlir::Value &lhs, mlir::Value &rhs, int32_t shift = 0);
  mlir::Value intdiv(mlir::Value &lhs, mlir::Value &rhs);

  mlir::Value transpose(mlir::Value &value, llvm::ArrayRef<int32_t> perm);
  mlir::Value slice(mlir::Value &inputConst, llvm::ArrayRef<int64_t> size,
      llvm::ArrayRef<int64_t> start);
  mlir::Value reshape(mlir::Value &value, llvm::ArrayRef<int64_t> shape);

  template <typename T>
  mlir::Value unaryOp(mlir::Value &input);
  mlir::Value sqrt(mlir::Value &input);

  template <typename T>
  mlir::Value compareOp(mlir::PatternRewriter &rewriter, mlir::Location loc,
      mlir::Value &lhs, mlir::Value &rhs);
  mlir::Value equal(mlir::Value &lhs, mlir::Value &rhs);
  mlir::Value greater(mlir::Value &lhs, mlir::Value &rhs);
  mlir::Value greaterEqual(mlir::Value &lhs, mlir::Value &rhs);
  mlir::Value less(mlir::Value &lhs, mlir::Value &rhs);
  mlir::Value lessEqual(mlir::Value &lhs, mlir::Value &rhs);

  mlir::Value select(mlir::Value &cond, mlir::Value &lhs, mlir::Value &rhs);

  /// When using window based ops like maxpool or conv2d, we sometimes have
  /// unused values at the end of a spatial dimension. TOSA does not allow that,
  /// the input can only have values that are actually used. To achieve this we
  /// have to reduce padding and if this is not enough, we even have to insert a
  /// slice op.
  mlir::FailureOr<mlir::Value> resizeWindowBasedOps(mlir::Value &value,
      llvm::ArrayRef<int64_t> inputShape,
      llvm::ArrayRef<int64_t> weightSpatialShape,
      llvm::SmallVectorImpl<int64_t> &padding,
      llvm::ArrayRef<int64_t> strides = {1, 1},
      llvm::ArrayRef<int64_t> dilation = {0, 0});

  mlir::Value getConst(
      llvm::ArrayRef<int64_t> vec, llvm::ArrayRef<int64_t> shape);
  mlir::Value getConst(
      llvm::ArrayRef<int32_t> vec, llvm::ArrayRef<int64_t> shape);
  mlir::Value getConst(
      llvm::ArrayRef<int8_t> vec, llvm::ArrayRef<int64_t> shape);
  mlir::Value getConst(
      llvm::ArrayRef<float> vec, llvm::ArrayRef<int64_t> shape);
  // Create a floating-point constant operator from a float
  // The tensor will have the same rank as shape but all dimensions will
  // have size 1 (differs from tensorflow impl.)
  // If dtype is provided, it also cast the value to the appropriate dtype.
  mlir::Value getSplattedConst(float val, llvm::ArrayRef<int64_t> shape = {},
      std::optional<mlir::Type> dtype = {});

  // Creates a constant of shape <1x1x...x1> of rank `rank` with all values set
  // to `value`.
  template <typename T>
  mlir::Value getSplattedConst(T value, uint rank) {
    llvm::SmallVector<int64_t, 4> tmpTensor;
    for (uint i = 0; i < rank; ++i) {
      tmpTensor.emplace_back(1);
    }
    std::vector zpVec = std::vector<T>{value};
    return getConst(zpVec, tmpTensor);
  }

  // Adds reshape ops to expand the rank to the max rank of the values.
  llvm::SmallVector<mlir::Value, 4> equalizeRanks(mlir::ValueRange valueRange);

protected:
  template <typename T>
  bool testNumberOfElementsMatch(
      llvm::ArrayRef<T> vec, llvm::ArrayRef<int64_t> shape);
  template <typename T>
  mlir::Value createConstFromRankedTensorAndVec(
      llvm::ArrayRef<T> vec, mlir::RankedTensorType &constType);
  template <typename T>
  mlir::Value createConst(
      llvm::ArrayRef<T> vec, llvm::ArrayRef<int64_t> shape, mlir::Type &type);

  mlir::Value expandRank(mlir::Value input, int64_t rank);
  bool needsRankBroadcast(mlir::ValueRange valueRange);

  // Private getters of builder (concise version).
  mlir::PatternRewriter &rewriter() const {
    assert(patternRewriter && "rewriter is null");
    return *patternRewriter;
  }

private:
  mlir::PatternRewriter *patternRewriter;
};

// =============================================================================
// IndexExpr Builder for Shape lowering
// =============================================================================

struct IndexExprBuilderForTosa : IndexExprBuilder {
  IndexExprBuilderForTosa(mlir::Location loc) : IndexExprBuilder(loc) {}
  IndexExprBuilderForTosa(mlir::OpBuilder &b, mlir::Location loc)
      : IndexExprBuilder(b, loc) {}
  IndexExprBuilderForTosa(const DialectBuilder &db) : IndexExprBuilder(db) {}
  virtual ~IndexExprBuilderForTosa() {}

protected:
  mlir::ElementsAttr getConst(mlir::Value value) final;
  mlir::Value getVal(mlir::Value intArrayVal, uint64_t i) final;
  mlir::Value getShapeVal(mlir::Value tensorOrMemrefValue, uint64_t i) final;
};

// Recursive class specialized for AffineBuilder refereed to as affine.
template <class... Ts>
struct MultiDialectBuilder<IndexExprBuilderForTosa, Ts...>
    : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(mlir::OpBuilder &b, mlir::Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), tosaIE(b, loc) {}
  MultiDialectBuilder(const DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), tosaIE(db) {}
  IndexExprBuilderForTosa tosaIE;
};

} // namespace onnx_mlir
