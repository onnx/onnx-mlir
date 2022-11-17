/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- NewShapeHelper.hpp - help for shapes ---------------===//
//
// Copyright 2020-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file has the computations to compute the shapes using the new index expr
// approach.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <utility>

#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/Mlir/IndexExprBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

namespace onnx_mlir {

using DimsExpr = llvm::SmallVector<IndexExpr, 4>;

//===----------------------------------------------------------------------===//
// Top shape helper class
//===----------------------------------------------------------------------===//

template <class OP>
struct NewONNXOpShapeHelper {
  // Constructor for shape inference.
  NewONNXOpShapeHelper(OP *op, mlir::ValueRange operands,
      IndexExprBuilder *ieBuilder, IndexExprScope *scope);
  virtual ~NewONNXOpShapeHelper();

  // Every child class is expected to create a computeShape with the following
  // signature. This method is responsible to compute at a minimum the output
  // dims.
  virtual mlir::LogicalResult computeShape() = 0;

  // Use the op to get attributes, and operandAdaptor to get the input/output
  // tensors.

  // Set/get output dims for the N-th output.
  DimsExpr &getOutputDims(int n = 0) { return outputsDims[n]; }
  void setOutputDims(DimsExpr inferredDims, int n = 0);

  // Set the number of outputs.(hi alex: if only needed in constructor, remove)
  void setNumberOfOutputs(int n) { outputsDims.resize(n); }

  // Obtain the n-th output value.
  mlir::Value getOutput(int n) { return op->getResult(n); }

  // get index expression scope
  IndexExprScope *getScope() { return scope; }

protected:
  // Data that must be present for every ShapeHelper operation. Op and scope
  // are initialized in the constructor, and outputsDims is computed by the
  // child's struct `computeShape` function.
  OP *op;
  mlir::ValueRange operands;
  IndexExprBuilder *createIE;
  IndexExprScope *scope;

private:
  llvm::SmallVector<DimsExpr, 1> outputsDims;
  bool ownScope;
};

//===----------------------------------------------------------------------===//
// Unary
//===----------------------------------------------------------------------===//

/// Compute an output shape for a unary element-wise operation. The output and
/// input of an unary element-wise operation have the same shape.
struct NewONNXGenericOpUnaryShapeHelper
    : public NewONNXOpShapeHelper<mlir::Operation> {
  NewONNXGenericOpUnaryShapeHelper(mlir::Operation *op,
      mlir::ValueRange operands, IndexExprBuilder *ieBuilder,
      IndexExprScope *scope = nullptr);
  ~NewONNXGenericOpUnaryShapeHelper() {}

  mlir::LogicalResult computeShape() final;
};

//===----------------------------------------------------------------------===//
// Broadcast
//===----------------------------------------------------------------------===//

/// Compute a broadcasted shape from the shapes of given operands. Operands must
/// be ranked in advance.
template <class OP>
struct NewONNXOpBroadcastedShapeHelper : public NewONNXOpShapeHelper<OP> {
  NewONNXOpBroadcastedShapeHelper(mlir::Operation *op,
      mlir::ValueRange operands, DimsExpr *additionalOperand,
      IndexExprBuilder *ieBuilder, IndexExprScope *scope = nullptr,
      bool hasUniBroadcasting = false, bool hasNoBroadcasting = false);
  ~NewONNXOpBroadcastedShapeHelper() {}

  mlir::LogicalResult computeShape() final;

  // Compute access indices to load/store value from/to a given 'operand'.
  // Used in a loop to access the operand.
  // Parameters:
  //   - operand: operand to access.
  //   - operandIndex: index of the operand in 'this->inputsDims'.
  //   - loopAccessExprs: IndexExprs for the loop's IVs.
  //   - operandAccessExprs: access indices to access the operand.
  //     This is the output of this function. Use it in subsequent load/stores.
  // hi alex: rename getAccessExprs
  mlir::LogicalResult GetAccessExprs(mlir::Value operand, uint64_t i,
      const llvm::SmallVectorImpl<IndexExpr> &outputAccessExprs,
      llvm::SmallVectorImpl<IndexExpr> &operandAccessExprs);

  // A vector of input shapes where dimensions are padded with 1 if necessary,
  // so that all inputs have the same rank. Instantiated during ComputeShape.
  llvm::SmallVector<DimsExpr, 4> inputsDims;
  // A vector of IndexExprs representing the output shape.
  // in upper DimsExpr outputDims;
  uint64_t outputRank;

protected:
  // Because of templates, have to be specific about vars from parent class.
  using NewONNXOpShapeHelper<OP>::createIE;
  using NewONNXOpShapeHelper<OP>::operands;

  // Some ops need an additional operand passed as an IndexExpression vector.
  // Ignored when null.
  DimsExpr *additionalOperand;
  // When unidirectional broadcasting is true, the other operands are always
  // unidirectional broadcastable to the first operand.
  bool hasUniBroadcasting;
  // When isNoBroadcasting is true, the shape of all input is assumed to be
  // same. This flag is used to test dynamic shape. There is no impact on static
  // shape.
  bool hasNoBroadcasting;
};

struct NewONNXGenericOpBroadcastedShapeHelper
    : public NewONNXOpBroadcastedShapeHelper<mlir::Operation> {
  NewONNXGenericOpBroadcastedShapeHelper(mlir::Operation *op,
      mlir::ValueRange operands, IndexExprBuilder *ieBuilder,
      IndexExprScope *scope = nullptr, bool uniBroadcasting = false,
      bool noBroadcasting = false);
  ~NewONNXGenericOpBroadcastedShapeHelper() {}
};

} // namespace onnx_mlir