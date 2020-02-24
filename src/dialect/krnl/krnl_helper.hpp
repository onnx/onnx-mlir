#pragma once

#include <queue>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace onnf {

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
} // namespace onnf

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

  void pushConstantBound(int64_t bound);

  void pushOperandBound(mlir::Value operand);

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
// define/optimize/iterate operation combo. We can then insert optimizations in
// the body of the optimization operation, and operations in the body of the
// iterate operation.
//
// The sequence is as follow:
//
//   1) Create a object giving the rewriter, location, and number of loop in the
//   original (non optimized) loop.
//
//   2) Create define & optimize ops (currently paired). Optimizations can then
//   be added to the inner block of the optimize operation. Make sure to set the
//   insertion point to that block for optimizations to go in the right place.
//
//   3) Push the bounds for each of the original loops. Bounds are pushed in
//   pairs (lower & upper bounds). THere are a few methods to do it depending on
//   the type of the bounds. When pushing bounds, the method returns a number
//   that represent the index associated with that iteration (induction variable
//   and bounds). That index can be used later to extract the induction variable
//   for reference in computation and/or index calculations of mem refs.
//
//   4) Once all the bounds are pushed, create the iterate operation. Once this
//   is done, we can add operations within the iterate blocks by setting the
//   insertion point to it. Value of the induction variables can be retrieved
//   using the proper index (determined when pushin the bounds).

class BuildKrnlLoop {
public:
  // Create a build kernel loop for the given location and loop number.
  BuildKrnlLoop(ConversionPatternRewriter &rewriter, Location loc, int loopNum);
  // Do the same, but where the loop number corresponds to the dimensionality of
  // the mem ref operand.
  BuildKrnlLoop(
      ConversionPatternRewriter &rewriter, Location loc, Value memRefOperand);
  ~BuildKrnlLoop();

  // Create define and optimize loop with loopNum original loops. If
  // withEmptyOptimization, the optimization is simply the identity function (no
  // optimizations).
  void createDefineAndOptimizeOp(bool withEmptyOptimization = true);

  // Push bounds (lower and upper) for each of the loops, in order. It returns
  // the index associated with the loop iteration. This index is in the range
  // from zero to original loop number -1, and is monotonally increasing from
  // call to call. This index is later used in the getInductionVar call.
  int pushBounds(int64_t lowerBound, int64_t upperBound);
  int pushBounds(int64_t lowerBound, Value upperBound);
  int pushBounds(Value lowerBound, Value upperBound);
  // same, where the lower bound is an integer, and the uppoer bound is given by
  // the size of the mem ref operand along the upperBoundMemRefIndex dimension.
  int pushBounds(int64_t lowerBound, Value upperBoundMemRefOperand,
      int upperBoundMemRefIndex, bool upperBoundMustBeConstant = false);

  // Create an iterate op.
  void createIterateOp();
  // Create an define, optimize and iterate op, with the same loop nummber as
  // the rank of the memRefOperand. The lower bound of each loops is zero, and
  // the upper bound of each loops is the dimension given by the mem refs
  void createDefineOptimizeAndIterateOp(
      Value memRefOperand, bool withEmptyOptimization = true);

  // Get the (original loop) induction variable associated with the given index.
  // Use the index returned when pushing the bounds.
  BlockArgument &getInductionVar(int originalLoopIndex);

  // Get blocks. This allow us to set the insertion point to the inner block of
  // the optimize and the iterate Operation
  Block *getOptimizationBlock() { return optBlock; }
  Block *getIterateBlock() { return iterBlock; }

  // get original or optimized loops
  std::vector<Value> &getOriginalLoops() { return originalLoops; }
  std::vector<Value> &getOptimizedLoops() { return optLoops; }

private:
  // inputs
  ConversionPatternRewriter &rewriter;
  Location loc;
  int originalLoopNum;
  // track loops and bounds
  std::vector<Value> originalLoops;
  std::vector<Value> optLoops;
  KrnlIterateOperandPack *pack;
  int pushCount;
  bool createdDefineOp;
  bool createdOptimizeOp;
  bool createdIterateOp;
  // insertion points (opt block, iterate)
  Block *optBlock;
  Block *iterBlock;
};

} // namespace mlir
