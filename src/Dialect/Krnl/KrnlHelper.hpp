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
#include "src/Dialect/ONNX/IndexExpr.hpp"
#include "src/Dialect/ONNX/MLIRDialectBuilder.hpp"

namespace mlir {
class KrnlIterateOp;
class KrnlGetRefOp;
class KrnlMovableOp;
} // namespace mlir

namespace onnx_mlir {

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
  std::queue<mlir::OpAsmParser::OperandType> operandRefQueue;
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

  void pushAffineMapBound(mlir::AffineMap map, ArrayRef<Value> operands);

  // When used in a lower bound, set isLb to true, when used in an upper bound,
  // set isLb to false.
  void pushIndexExprBound(IndexExpr expr, bool isLb);

  void pushIndexExprsBound(SmallVectorImpl<IndexExpr> &exprVector);

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

class BuildKrnlLoop final {
public:
  // Create kernel loop builder for a loop nest of depth loopNum.
  BuildKrnlLoop(OpBuilder &builder, Location loc, int loopNum);
  BuildKrnlLoop(const BuildKrnlLoop &) = delete;
  BuildKrnlLoop(BuildKrnlLoop &&) = delete;
  BuildKrnlLoop &operator=(const BuildKrnlLoop &) = delete;
  BuildKrnlLoop &operator=(BuildKrnlLoop &&) = delete;

  // Create kernel loop builder for a loop nest of depth equal to the
  // dimensionality of the operand. An operand of MemRef type is requied.
  BuildKrnlLoop(OpBuilder &builder, Location loc, Value memRefOperand);
  ~BuildKrnlLoop() {
    if (pack)
      delete pack;
  }

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
  int pushBounds(SmallVectorImpl<IndexExpr> &lowerBound,
      SmallVectorImpl<IndexExpr> &upperBound);
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
  BlockArgument &getInductionVar(int originalLoopIndex) const;

  // Get all of the (original loop) induction variables.
  ArrayRef<BlockArgument> getAllInductionVar() const;

  // Get a reference to the code region of the optimization operation.
  // This allows us to set the insertion point to the inner block of the
  // loop nest optimization operation.
  // Deprecated.
  Block *getOptimizationBlock() const { return optBlock; }

  // Get a reference to the code region of the iteration operation.
  // This allows us to set the insertion point to the inner block of the
  // loop nest iteration operation.
  Block *getIterateBlock() const { return iterBlock; }

  // Get original loop nest.
  const std::vector<Value> &getOriginalLoops() const { return originalLoops; }

  // Get optimized loop nest.
  const std::vector<Value> &getOptimizedLoops() const { return optLoops; }

private:
  // Required for emitting operations.
  OpBuilder &builder;
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

// This function satisfies the ArrayValueIndexCapture::DenseElementsAttr lambda
// type, using ONNX and Krnl operations.
DenseElementsAttr getDenseElementAttributeFromKrnlValue(Value value);

// This function satisfies the ArrayValueIndexCapture::LoadVal lambda
// type, using Krnl operations.
Value loadDenseElementArrayValueAtIndex(
    OpBuilder &rewriter, Location loc, Value array, int64_t index);

//====---------------- Support for simple transpose ----------------------===//

void generateIndexMap(
    SmallVectorImpl<int64_t> &map, int64_t size, bool transposeInner2);

//====-------------------- Support for Krnl Builder ----------------------===//

struct KrnlBuilder : public DialectBuilder {
  KrnlBuilder(OpBuilder &b, Location loc) : DialectBuilder(b, loc) {}
  KrnlBuilder(DialectBuilder &db) : DialectBuilder(db) {}

  Value load(Value memref, ValueRange indices = {}) const;
  Value loadIE(Value memref, ArrayRef<IndexExpr> indices) const;
  void store(Value val, Value memref, ValueRange indices = {}) const;
  void storeIE(Value val, Value memref, ArrayRef<IndexExpr> indices) const;

  Value vectorTypeCast(Value sourceMemref, int64_t vectorLen) const;

  ValueRange defineLoops(int64_t originalLoopNum) const;
  ValueRange block(Value loop, int64_t blockSize) const;
  void permute(ValueRange loops, ArrayRef<int64_t> map) const;
  ValueRange getInductionVarValue(ValueRange loops) const;

  // Lambda passes loop indices as 2nd parameter.
  void iterate(ValueRange originalLoops, ValueRange optimizedLoops,
      ValueRange lbs, ValueRange ubs,
      function_ref<void(KrnlBuilder &createKrnl, ValueRange indices)>
          bodyBuilderFn) const;
  mlir::KrnlIterateOp iterate(const KrnlIterateOperandPack &operands) const;

  // Lambda passes loop indices as 2nd parameter.
  void iterateIE(ValueRange originalLoops, ValueRange optimizedLoops,
      ArrayRef<IndexExpr> lbs, ArrayRef<IndexExpr> ubs,
      function_ref<void(KrnlBuilder &createKrnl, ValueRange indices)>
          bodyBuilderFn) const;

  void copyToBuffer(
      // Buffer and source memory. Source memref may have a higher rank than
      // buffer.
      Value bufferMemref, Value sourceMemref,
      // Indices that points to the first data to be copied from source.
      // Starts has the same rank as sourceMemref.
      ValueRange starts,
      // If padding is needed, value to pad.
      Value padValue,
      // Now the bufferMemref may be larger than the actual data to be stored
      // in the buffer, if the user want to pad the data to a higher size.
      // TileSize enables the user to
      ArrayRef<int64_t> tileSize, ArrayRef<int64_t> padToNext,
      bool transpose = false) const;
  void copyToBuffer(Value bufferMemref, Value sourceMemref, ValueRange starts,
      Value padValue, bool transpose = false) const;

  void copyFromBuffer(Value bufferMemref, Value memref, ValueRange starts,
      ArrayRef<int64_t> tileSize) const;
  void copyFromBuffer(
      Value bufferMemref, Value memref, ValueRange starts) const;

  void matmul(
      // The a/b/cStart are the indices at the begining of the buffer/mem
      // A/B/C.
      Value A, ValueRange aStart, Value B, ValueRange bStart, Value C,
      ValueRange cStart,
      // Loops are the krnl loop indices that this matmul replaces
      ValueRange loops,
      // the computeStarts indicate the i/j/k indices pointing to the begining
      // of the matmul computation.
      ValueRange computeStarts,
      // The globalUBs are the global bounds on the original I, J, K
      // dimensions.
      ValueRange globalUBs,
      // If not the full A, B, C buffers are used by this matmul, meaning the
      // matmul uses a subtile of the buffers, this compute tile size
      // specifies the actual size of the i/j/k computations. Empty means
      // compute tiles encompass the entire buffer A, B, and C as defined by
      // their tile sizes.
      ArrayRef<int64_t> computeTileSize,
      // If buffers A, B, or C were padded, then the tile sizes give the size
      // of the non-padded data, basically the size of the data when the tile
      // is full. Partial tiles (due to computation on the edges of the
      // matrices) are handled differently (using the UBs), so no need to
      // worry about this. Empty means no padding was used.
      ArrayRef<int64_t> aTileSize, ArrayRef<int64_t> bTileSize,
      ArrayRef<int64_t> cTileSize,
      // Optimizations for code gen.
      bool simdize, bool unroll, bool overcompute) const;
  void matmul(Value A, ValueRange aStart, Value B, ValueRange bStart, Value C,
      ValueRange cStart, ValueRange loops, ValueRange computeStarts,
      ValueRange globalUBs, bool simdize, bool unroll, bool overcompute) const;

  Value dim(Type type, Value alloc, Value index) const;

  mlir::KrnlMovableOp movable() const;

  mlir::KrnlGetRefOp getRef(
      Type type, Value memref, Value offset, ValueRange indices = {}) const;

  Value constant(MemRefType type, StringRef name, Optional<Attribute> value,
      Optional<IntegerAttr> offset = None,
      Optional<IntegerAttr> alignment = None) const;

  // C library functions.
  void memcpy(Value dest, Value src, Value size) const;
  void memset(Value dest, Value val) const;
  Value strncmp(Value str1, Value str2, Value len) const;
  Value strlen(Value str) const;
  void printf(StringRef msg) const;
  void printf(StringRef msg, Value input, Type inputType) const;

  // Onnx-mlir runtime functions.
  void randomNormal(Value alloc, Value numberOfRandomValues, Value mean,
      Value scale, Value seed) const;
  Value findIndex(Value input, Value G, Value V, Value len) const;
  void printTensor(StringRef msg, Value input) const;
};

// Recursive class specialized for KrnlBuilder referred to as krnl.
template <class... Ts>
struct MultiDialectBuilder<KrnlBuilder, Ts...> : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(OpBuilder &b, Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), krnl(b, loc) {}
  MultiDialectBuilder(DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), krnl(db) {}
  KrnlBuilder krnl;
};

//====---------------- Common helper functions --------------------------===//

/// Check whether a value is produced by a dense KrnlGlobalOp.
bool isKrnlGlobalConstant(Value result);

} // namespace mlir
