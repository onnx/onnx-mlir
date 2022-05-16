/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====---------------- KrnlHelper.hpp - Krnl Dialect Helper----------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements helper methods to build Krnl Dialect ops.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <queue>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/Mlir/IndexExpr.hpp"

namespace mlir {
class KrnlIterateOp;
class KrnlGetRefOp;
class KrnlMovableOp;
} // namespace mlir

namespace onnx_mlir {
namespace krnl {

class KrnlDialectOperandParser {
public:
  explicit KrnlDialectOperandParser(mlir::OpAsmParser &parser)
      : parser(parser), builder(parser.getBuilder()){};

  // Parse an optional operand.
  mlir::ParseResult ParseOptionalOperand(
      const mlir::Type &operandType, mlir::Value &operand);

  // Parse an optional operand and push it to an operand list.
  mlir::ParseResult ParseOptionalOperand(const mlir::Type &operandType,
      llvm::SmallVectorImpl<mlir::Value> &operandList);

  // Parse a required operand.
  mlir::ParseResult ParseOperand(
      const mlir::Type &operandType, mlir::Value &operand);

  // Parse a required operand and push it to an operand list.
  mlir::ParseResult ParseOperand(const mlir::Type &operandType,
      llvm::SmallVectorImpl<mlir::Value> &operandList);

  // Do we have more operands to parse?
  bool hasOperandLeft() { return !operandRefQueue.empty(); }

private:
  mlir::OpAsmParser &parser;
  mlir::Builder &builder;

  // A queue storing the parsed SSA id references.
  std::queue<mlir::OpAsmParser::UnresolvedOperand> operandRefQueue;
};

// Adapted from:
// https://github.com/tensorflow/mlir/blob/6a150d70c7e06fb37cddd7188fa48cde9a90fe59/lib/Dialect/StandardOps/Ops.cpp#L197
// Main difference is that it advances the iterator `begin` as it consumes
// dimension and symbol operands.
void printDimAndSymbolList(mlir::Operation::operand_iterator &begin,
    unsigned numDims, unsigned numSymbols, mlir::OpAsmPrinter &p);

// Adapted from:
// https://github.com/tensorflow/mlir/blob/5cb42c914fed14cebbbe5c170b4e2784d2628304/lib/Dialect/AffineOps/AffineOps.cpp#L1272
// Main difference is that it advances the iterator `boundOperandsBeg` as it
// prints bound.
void printBound(mlir::AffineMapAttr boundMap,
    mlir::Operation::operand_iterator &boundOperandsBeg, const char *prefix,
    mlir::OpAsmPrinter &p);

struct KrnlIterateOperandPack {
  KrnlIterateOperandPack(mlir::Builder &builder,
      llvm::ArrayRef<mlir::Value> inputLoops,
      llvm::ArrayRef<mlir::Value> optimizedLoops)
      : inputLoops(inputLoops), optimizedLoops(optimizedLoops),
        builder(builder) {
    operands.insert(
        operands.end(), optimizedLoops.begin(), optimizedLoops.end());
  }

  // Create a pack with optimizedLoops = inputLoops (ie., no optimization).
  KrnlIterateOperandPack(
      mlir::Builder &builder, llvm::ArrayRef<mlir::Value> inputLoops)
      : inputLoops(inputLoops), optimizedLoops(inputLoops), builder(builder) {
    operands.insert(operands.end(), inputLoops.begin(), inputLoops.end());
  }

  void pushConstantBound(int64_t bound);

  void pushOperandBound(mlir::Value operand);

  void pushAffineMapBound(
      mlir::AffineMap map, mlir::ArrayRef<mlir::Value> operands);

  // When used in a lower bound, set isLb to true, when used in an upper bound,
  // set isLb to false.
  void pushIndexExprBound(IndexExpr expr, bool isLb);

  void pushIndexExprsBound(llvm::SmallVectorImpl<IndexExpr> &exprVector);

  llvm::SmallVector<mlir::Value, 8> getOperands() const { return operands; }

  mlir::ArrayAttr getAttributes() const {
    return builder.getArrayAttr(boundMaps);
  }

  size_t getNumOptimizedLoops() const { return optimizedLoops.size(); }

  size_t getNumInputLoops() const { return inputLoops.size(); }

private:
  llvm::SmallVector<mlir::Value, 8> operands;
  llvm::SmallVector<mlir::Attribute, 8> boundMaps;
  llvm::ArrayRef<mlir::Value> inputLoops, optimizedLoops;
  mlir::Builder &builder;
};

// This function satisfies the ArrayValueIndexCapture::DenseElementsAttr lambda
// type, using ONNX and Krnl operations.
mlir::DenseElementsAttr getDenseElementAttributeFromKrnlValue(
    mlir::Value value);

// This function satisfies the ArrayValueIndexCapture::LoadVal lambda
// type, using Krnl operations.
mlir::Value loadDenseElementArrayValueAtIndex(mlir::OpBuilder &rewriter,
    mlir::Location loc, mlir::Value array, int64_t index);

//====---------------- Support for simple transpose ----------------------===//

void generateIndexMap(
    llvm::SmallVectorImpl<int64_t> &map, int64_t size, bool transposeInner2);

//====---------------- Common helper functions --------------------------===//

/// Check whether a value is produced by a dense KrnlGlobalOp.
bool isKrnlGlobalConstant(mlir::Value result);

} // namespace krnl
} // namespace onnx_mlir
