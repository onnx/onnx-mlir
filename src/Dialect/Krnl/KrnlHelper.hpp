//====---------------- KrnlHelper.hpp - Krnl Dialect Helper----------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
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
#include "src/Dialect/ONNX/IndexExpr.hpp"

namespace onnx_mlir {

class KrnlDialectOperandParser {
public:
  explicit KrnlDialectOperandParser(mlir::OpAsmParser &parser)
      : _parser(parser), _builder(parser.getBuilder()){};

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
  bool hasOperandLeft() { return !_operandRefQueue.empty(); }

private:
  mlir::OpAsmParser &_parser;

  mlir::Builder &_builder;

  // A queue storing the parsed SSA id references.
  std::queue<mlir::OpAsmParser::OperandType> _operandRefQueue;
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
} // namespace onnx_mlir

namespace mlir {

struct KrnlIterateOperandPack {
  KrnlIterateOperandPack(mlir::Builder &builder,
      llvm::ArrayRef<mlir::Value> inputLoops,
      llvm::ArrayRef<mlir::Value> optimizedLoops)
      : builder(builder), inputLoops(inputLoops),
        optimizedLoops(optimizedLoops) {
    _operands.insert(
        _operands.end(), optimizedLoops.begin(), optimizedLoops.end());
  }

  // Create a pack with optimizedLoops = inputLoops (ie., no optimization).
  KrnlIterateOperandPack(
      mlir::Builder &builder, llvm::ArrayRef<mlir::Value> inputLoops)
      : builder(builder), inputLoops(inputLoops), optimizedLoops(inputLoops) {
    _operands.insert(_operands.end(), inputLoops.begin(), inputLoops.end());
  }

  void pushConstantBound(int64_t bound);

  void pushOperandBound(mlir::Value operand);

  void pushAffineMapBound(mlir::AffineMap map, ArrayRef<Value> operands);

  void pushIndexExprBound(IndexExpr expr);

  void pushIndexExprsBound(SmallVectorImpl<IndexExpr> &exprVector);

  llvm::SmallVector<mlir::Value, 8> getOperands() const { return _operands; }

  mlir::ArrayAttr getAttributes() const {
    return builder.getArrayAttr(boundMaps);
  }

  size_t getNumOptimizedLoops() const { return optimizedLoops.size(); }

  size_t getNumInputLoops() const { return inputLoops.size(); }

private:
  int _boundIdx = 0;

  llvm::SmallVector<mlir::Value, 8> _operands;

  llvm::SmallVector<mlir::Attribute, 8> boundMaps;

  llvm::ArrayRef<mlir::Value> inputLoops, optimizedLoops;

  mlir::Builder &builder;
};

// Helper function to write kernel loops. This class will let us build a single
// define/iterate operation combo. We can then insert operations in the body of
// the iterate operation.
//
// The sequence is as follow:
//
//   1) Create an object giving the rewriter, location, and number of loop in
//   the original (non optimized) loop.
//
//   2) Create define_loops ops to define new loop variables.
//
//   3) Push the bounds for each of the original loops. Bounds are pushed in
//   pairs (lower & upper bounds). There are a few methods to do it depending
//   on the type of the bounds. When pushing bounds, the method returns a
//   number that represent the index associated with that iteration (induction
//   variable and bounds). That index can be used later to extract the
//   induction variable for reference in computation and/or index calculations
//   of mem refs.
//
//   4) Once all the bounds are pushed, create the iterate operation. Once this
//   is done, we can add operations within the iterate blocks by setting the
//   insertion point to it. Value of the induction variables can be retrieved
//   using the proper index (determined when pushin the bounds).

class BuildKrnlLoop {
public:
  // Create kernel loop builder for a loop nest of depth loopNum.
  BuildKrnlLoop(ConversionPatternRewriter &rewriter, Location loc, int loopNum);

  // Create kernel loop builder for a loop nest of depth equal to the
  // dimensionality of the operand. An operand of MemRef type is requied.
  BuildKrnlLoop(
      ConversionPatternRewriter &rewriter, Location loc, Value memRefOperand);
  ~BuildKrnlLoop();

  // Create define and optimize loop with loopNum original loops. If
  // withEmptyOptimization is true, the optimization is simply the identity
  // function (no optimizations).
  void createDefineOp();

  // Push bounds (lower and upper) for each of the loops (order matters).
  // The function returns the order number associated with the loop iteration.
  // This index is used by the getInductionVar call. Non-constant operands
  // must be of MemRef type.
  int pushBounds(int64_t lowerBound, int64_t upperBound);
  int pushBounds(int64_t lowerBound, Value upperBound);
  int pushBounds(int64_t lowerBound, IndexExpr upperBound);
  int pushBounds(int64_t lowerBound, SmallVectorImpl<IndexExpr> &upperBound);
  int pushBounds(SmallVectorImpl<IndexExpr> &lowerBound, SmallVectorImpl<IndexExpr> &upperBound);
  int pushBounds(IndexExpr lowerBound, IndexExpr upperBound);
  int pushBounds(int64_t lowerBound, AffineMap upperBound,
      ArrayRef<Value> operandsForUpperBoundMap);
  int pushBounds(Value lowerBound, Value upperBound);
  int pushBounds(int64_t lowerBound, Value upperBoundMemRefOperand,
      int upperBoundMemRefIndex, bool upperBoundMustBeConstant = false);
  // for each index expression i in upperBounds, push 0..upperBound[i].
  void pushAllBounds(SmallVectorImpl<IndexExpr> &upperBounds);

  // Create the KrnlIterateOp assiciated with this loop nest. The loops
  // iteration will be created if the definition and the optimization
  // operations associated with this loop nest have been emitted already.
  void createIterateOp();

  // Create the loop nest definition and iteration operations
  // for a given operand of MemRef type. The loop nest has a depth equal to the
  // rank of the MemRef operand. The lower bound of each loop is zero. The
  // upper bound of each loop is given by the corresponding dimension of the
  // MemRef operand.
  void createDefineAndIterateOp(Value memRefOperand);

  // Get the (original loop) induction variable associated with the given
  // index. Use the index returned when pushing the bounds.
  BlockArgument &getInductionVar(int originalLoopIndex);

  // Get all of the (original loop) induction variables.
  ArrayRef<BlockArgument> getAllInductionVar();

  // Get a reference to the code region of the optimization operation.
  // This allows us to set the insertion point to the inner block of the
  // loop nest optimization operation.
  Block *getOptimizationBlock() { return optBlock; }

  // Get a reference to the code region of the iteration operation.
  // This allows us to set the insertion point to the inner block of the
  // loop nest iteration operation.
  Block *getIterateBlock() { return iterBlock; }

  // Get original loop nest.
  std::vector<Value> &getOriginalLoops() { return originalLoops; }

  // Get optimized loop nest.
  std::vector<Value> &getOptimizedLoops() { return optLoops; }

private:
  // Required for emitting operations.
  ConversionPatternRewriter &rewriter;
  Location loc;
  int originalLoopNum;

  // List of original, un-optimized loops.
  std::vector<Value> originalLoops;

  // List of optimized loops.
  std::vector<Value> optLoops;

  // List of lower-upper bound pairs needed by the KrnlIterateOp.
  KrnlIterateOperandPack *pack;

  // Number of lower-upper bound pairs pushed.
  int pushCount;

  // Flags that keep track of emitted operations.
  bool createdDefineOp;
  bool createdIterateOp;

  // Saved insertion point in the code region of the KrnlOptimizeLoopsOp.
  Block *optBlock;

  // Saved insertion point in the code region of the KrnlIterateOp.
  Block *iterBlock;
};

} // namespace mlir
