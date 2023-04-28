/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ DialectBuilder.hpp - TOSA dialect builder --------------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file contains the dialect build for the TOSA dialect. Uses the same
// implementation as MHLO with minor differences.
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
// IndexExpr Builder for Shape lowering
// =============================================================================

struct TosaBuilder : DialectBuilder {
  TosaBuilder(mlir::Location loc) : DialectBuilder(loc) {}
  TosaBuilder(mlir::PatternRewriter &b, mlir::Location loc)
      : DialectBuilder(b, loc), patternRewriter(&b) {}
  TosaBuilder(const DialectBuilder &db) : DialectBuilder(db) {}
  virtual ~TosaBuilder() {}

  mlir::Value reshape(mlir::Value &value, llvm::ArrayRef<int64_t> shape);
  mlir::Value transpose(mlir::Value &value, llvm::ArrayRef<int32_t> perm);
  mlir::Value slice(mlir::Value &inputConst, llvm::ArrayRef<int64_t> size,
      llvm::ArrayRef<int64_t> start);
  llvm::Optional<mlir::Value> gather(mlir::Value resultValue,
      mlir::Value inputValue, mlir::Value indicesValue, int32_t batchDims,
      int32_t axis);

  mlir::Value getConst(
      llvm::ArrayRef<int64_t> vec, llvm::ArrayRef<int64_t> shape);
  mlir::Value getConst(
      llvm::ArrayRef<int32_t> vec, llvm::ArrayRef<int64_t> shape);
  mlir::Value getConst(
      llvm::ArrayRef<int8_t> vec, llvm::ArrayRef<int64_t> shape);
  mlir::Value getConst(
      llvm::ArrayRef<float> vec, llvm::ArrayRef<int64_t> shape);
  // Create a 32-bit float constant operator from a float
  // The tensor will have the same rank as shape but with axis 1 (differs from
  // tensorflow impl.)
  mlir::Value getConst(float val, llvm::ArrayRef<int64_t> shape = {});
  
  // Creates a constant of shape <1x1x...x1> of rank `rank` with all values set to
  // `value`.
  template<typename T>
  mlir::Value getSplattedConst(T value, uint rank) {
    llvm::SmallVector<int64_t, 4> tmpTensor;
    for (uint i = 0; i < rank; ++i) {
      tmpTensor.emplace_back(1);
    }
    std::vector zpVec = std::vector<T>{value};
    return getConst(zpVec, tmpTensor);
  }


protected:
  template <typename T>
  mlir::Value createConstFromRankedTensorAndVec(
      llvm::ArrayRef<T> vec, mlir::RankedTensorType &constType);

  // Private getters of builder (concise version).
  mlir::PatternRewriter &rewriter() const {
    assert(patternRewriter && "rewriter is null");
    return *patternRewriter;
  }

private:
  mlir::PatternRewriter *patternRewriter;
};

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
