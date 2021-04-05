/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====---------------- KrnlHelper.cpp - Krnl Dialect Helper----------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file declares helper methods to build Krnl Dialect Ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExpr.h"

#include "src/Dialect/Krnl/KrnlOps.hpp"

#include "KrnlHelper.hpp"

namespace onnx_mlir {

using namespace mlir;

ParseResult KrnlDialectOperandParser::ParseOptionalOperand(
    const Type &operandType, Value &operand) {
  // If operand queue is empty, parse more operands and cache them.
  if (_operandRefQueue.empty()) {
    // Parse operand types:
    llvm::SmallVector<OpAsmParser::OperandType, 2> operand_refs;
    _parser.parseOperandList(operand_refs);

    // Record operands:
    for (auto &operand_ref : operand_refs)
      _operandRefQueue.emplace(operand_ref);
  }

  // If we parsed some operand reference(s), resolve the ref to an operand:
  if (!_operandRefQueue.empty()) {
    auto operand_ref = _operandRefQueue.front();
    _operandRefQueue.pop();

    llvm::SmallVector<Value, 1> operands;
    _parser.resolveOperand(operand_ref, operandType, operands);
    operand = operands.front();
    return success();
  } else {
    operand = nullptr;
    return failure();
  }
}

ParseResult KrnlDialectOperandParser::ParseOptionalOperand(
    const Type &operandType, llvm::SmallVectorImpl<Value> &operandList) {
  Value operand = nullptr;
  if (ParseOptionalOperand(operandType, operand))
    return failure();

  operandList.emplace_back(operand);
  return success();
}

ParseResult KrnlDialectOperandParser::ParseOperand(
    const Type &operandType, Value &operand) {
  if (ParseOptionalOperand(operandType, operand))
    return _parser.emitError(
        _parser.getCurrentLocation(), "Expecting an operand.");
  return success();
}

ParseResult KrnlDialectOperandParser::ParseOperand(
    const Type &operandType, llvm::SmallVectorImpl<Value> &operandList) {
  if (ParseOptionalOperand(operandType, operandList))
    return _parser.emitError(
        _parser.getCurrentLocation(), "Expecting an operand.");

  return success();
}

void printDimAndSymbolList(Operation::operand_iterator &begin, unsigned numDims,
    unsigned numSymbols, OpAsmPrinter &p) {
  p << '(';
  p.printOperands(begin, begin + numDims);
  p << ')';

  if (numSymbols) {
    p << '[';
    p.printOperands(begin + numDims, begin + numDims + numSymbols);
    p << ']';
  }

  begin = std::next(begin, numDims + numSymbols);
}

void printBound(AffineMapAttr boundMap,
    Operation::operand_iterator &boundOperandsBeg, const char *prefix,
    OpAsmPrinter &p) {
  AffineMap map = boundMap.getValue();

  // Check if this bound should be printed using custom assembly form.
  // The decision to restrict printing custom assembly form to trivial cases
  // comes from the will to roundtrip MLIR binary -> text -> binary in a
  // lossless way.
  // Therefore, custom assembly form parsing and printing is only supported for
  // zero-operand constant maps and single symbol operand identity maps.
  if (map.getNumResults() == 1) {
    AffineExpr expr = map.getResult(0);

    // Print constant bound.
    if (map.getNumDims() == 0 && map.getNumSymbols() == 0) {
      if (auto constExpr = expr.dyn_cast<AffineConstantExpr>()) {
        p << constExpr.getValue();
        return;
      }
    }

    // Print bound that consists of a single SSA symbol if the map is over a
    // single symbol.
    if (map.getNumDims() == 0 && map.getNumSymbols() == 1) {
      if (auto symExpr = expr.dyn_cast<AffineSymbolExpr>()) {
        p.printOperand(*(boundOperandsBeg++));
        return;
      }
    }
  } else {
    // Map has multiple results. Print 'min' or 'max' prefix.
    p << prefix << ' ';
  }

  // Print the map and its operands.
  p << boundMap;
  printDimAndSymbolList(
      boundOperandsBeg, map.getNumDims(), map.getNumSymbols(), p);
}
} // namespace onnx_mlir

namespace mlir {

//====---------------- KrnlIterateOperandPack -----------------------------===//

void KrnlIterateOperandPack::pushConstantBound(int64_t bound) {
  if (boundMaps.size() % 2 == 0)
    _operands.emplace_back(inputLoops[boundMaps.size() / 2]);
  AffineMap map = builder.getConstantAffineMap(bound);
  boundMaps.emplace_back(AffineMapAttr::get(map));
}

void KrnlIterateOperandPack::pushOperandBound(Value operand) {
  if (boundMaps.size() % 2 == 0)
    _operands.emplace_back(inputLoops[boundMaps.size() / 2]);
  AffineMap map = builder.getSymbolIdentityMap();
  boundMaps.emplace_back(AffineMapAttr::get(map));
  _operands.emplace_back(operand);
}

void KrnlIterateOperandPack::pushAffineMapBound(
    AffineMap map, ArrayRef<Value> operands) {
  if (boundMaps.size() % 2 == 0)
    _operands.emplace_back(inputLoops[boundMaps.size() / 2]);
  boundMaps.emplace_back(AffineMapAttr::get(map));
  for (auto operand : operands)
    _operands.emplace_back(operand);
}

// Bound could be a constant, Value or AffineMap
void KrnlIterateOperandPack::pushIndexExprBound(IndexExpr expr) {
  if (expr.isLiteral()) {
    pushConstantBound(expr.getLiteral());
  } else if (expr.isAffine() && !expr.isPredType()) {
    int dimNum = expr.getScope().getNumDims();
    int symNum = expr.getScope().getNumSymbols();
    AffineMap map = AffineMap::get(dimNum, symNum, {expr.getAffineExpr()},
        expr.getRewriter().getContext());
    SmallVector<Value, 4> list;
    expr.getScope().getDimAndSymbolList(list);
    pushAffineMapBound(map, list);
  } else {
    // Assume the expr is loop invariant if there is any outer loop
    pushOperandBound(expr.getValue());
  }
}

void KrnlIterateOperandPack::pushIndexExprsBound(
    SmallVectorImpl<IndexExpr> &exprVector) {
  SmallVector<AffineExpr, 4> AEVector;
  for (IndexExpr expr : exprVector) {
    assert(!expr.isPredType() && "no affine support for predicate type");
    AEVector.push_back(expr.getAffineExpr());
  }
  IndexExpr expr = exprVector.front();
  int dimNum = expr.getScope().getNumDims();
  int symNum = expr.getScope().getNumSymbols();
  AffineMap map =
      AffineMap::get(dimNum, symNum, AEVector, builder.getContext());
  SmallVector<Value, 4> list;
  expr.getScope().getDimAndSymbolList(list);
  pushAffineMapBound(map, list);
}

//====---------------- BuildKrnlLoop --------------------------------------===//

BuildKrnlLoop::BuildKrnlLoop(
    ConversionPatternRewriter &rewriter, Location loc, int loopNum)
    : rewriter(rewriter), loc(loc), originalLoopNum(loopNum), pack(NULL),
      pushCount(0), createdDefineOp(false), createdIterateOp(false) {
  if (originalLoopNum <= 0)
    emitError(loc, "Expected positive number of original loops.");
}

BuildKrnlLoop::BuildKrnlLoop(
    ConversionPatternRewriter &rewriter, Location loc, Value memRefOperand)
    : BuildKrnlLoop(rewriter, loc,
          memRefOperand.getType().cast<MemRefType>().getShape().size()) {}

BuildKrnlLoop::~BuildKrnlLoop() {
  if (pack)
    delete pack;
}

void BuildKrnlLoop::createDefineOp() {
  // Insert define loop operation.
  auto loopsOp = rewriter.create<KrnlDefineLoopsOp>(loc, originalLoopNum);
  originalLoops.reserve(originalLoopNum);
  for (auto result : loopsOp.getResults())
    originalLoops.push_back(result);
  createdDefineOp = true;

  // prepare data structure to push bounds
  pack = new KrnlIterateOperandPack(rewriter, originalLoops);
}

int BuildKrnlLoop::pushBounds(int64_t lowerBound, int64_t upperBound) {
  pack->pushConstantBound(lowerBound);
  pack->pushConstantBound(upperBound);
  return pushCount++;
}

int BuildKrnlLoop::pushBounds(int64_t lowerBound, Value upperBound) {
  pack->pushConstantBound(lowerBound);
  pack->pushOperandBound(upperBound);
  return pushCount++;
}

int BuildKrnlLoop::pushBounds(int64_t lowerBound, IndexExpr upperBound) {
  if (upperBound.isLiteral()) {
    return pushBounds(lowerBound, upperBound.getLiteral());
  }
  return pushBounds(lowerBound, upperBound.getValue());
}

int BuildKrnlLoop::pushBounds(
    int64_t lowerBound, SmallVectorImpl<IndexExpr> &upperBound) {
  pack->pushConstantBound(lowerBound);
  pack->pushIndexExprsBound(upperBound);
  return pushCount++;
}

int BuildKrnlLoop::pushBounds(SmallVectorImpl<IndexExpr> &lowerBound,
    SmallVectorImpl<IndexExpr> &upperBound) {
  pack->pushIndexExprsBound(lowerBound);
  pack->pushIndexExprsBound(upperBound);
  return pushCount++;
}

int BuildKrnlLoop::pushBounds(int64_t lowerBound, AffineMap upperBound,
    ArrayRef<Value> operandsForUpperBoundMap) {
  pack->pushConstantBound(lowerBound);
  pack->pushAffineMapBound(upperBound, operandsForUpperBoundMap);
  return pushCount++;
}

int BuildKrnlLoop::pushBounds(int64_t lowerBound, Value upperBoundMemRefOperand,
    int upperBoundMemRefIndex, bool upperBoundMustBeConstant) {
  pack->pushConstantBound(lowerBound);

  // Process upperBound as a dimension of the MemRef. Non-constant dimensions
  // are supported.
  auto shape = upperBoundMemRefOperand.getType().cast<MemRefType>().getShape();
  if (shape[upperBoundMemRefIndex] < 0) {
    assert(!upperBoundMustBeConstant && "Bound expected to be constant.");
    pack->pushOperandBound(
        rewriter
            .create<DimOp>(loc, upperBoundMemRefOperand, upperBoundMemRefIndex)
            .getResult());
  } else
    pack->pushConstantBound(shape[upperBoundMemRefIndex]);

  return pushCount++;
}

int BuildKrnlLoop::pushBounds(Value lowerBound, Value upperBound) {
  pack->pushOperandBound(lowerBound);
  pack->pushOperandBound(upperBound);
  return pushCount++;
}

void BuildKrnlLoop::pushAllBounds(SmallVectorImpl<IndexExpr> &upperBounds) {
  for (IndexExpr ie : upperBounds) {
    pushBounds(0, ie);
  }
}

void BuildKrnlLoop::createIterateOp() {
  // Loop definition operation is mandatory.
  assert(createdDefineOp && "Must create define op before iterate op.");

  // Check if all bounds have been defined.
  assert(pushCount == originalLoopNum &&
         "Must push bounds for all original loops.");

  // Emit iteration operation.
  auto iterateOp = rewriter.create<KrnlIterateOp>(loc, *pack);
  iterBlock = &iterateOp.bodyRegion().front();
  createdIterateOp = true;
}

void BuildKrnlLoop::createDefineAndIterateOp(Value memRefOperand) {
  // Rank of the MemRef operand. We will emit a loop for each dimension.
  int loopNum = memRefOperand.getType().cast<MemRefType>().getShape().size();
  assert(originalLoopNum == loopNum &&
         "Mismatch in loop numbers from constructor and define.");

  // Emit the definition and the optimization operations for the loop nest.
  createDefineOp();

  // Push a lower-upper bound pair for each dimension of the MemRef operand.
  // The lower bound in this case is always zero.
  for (int i = 0; i < originalLoopNum; ++i)
    pushBounds(0, memRefOperand, i);

  // Emit the iteration operation over the current loop nest.
  createIterateOp();
}

BlockArgument &BuildKrnlLoop::getInductionVar(int originalLoopIndex) {
  // Check if loop iteration variable is within bounds.
  assert(originalLoopIndex >= 0 && originalLoopIndex < originalLoopNum &&
         "Original loop index is out of bounds.");
  return iterBlock->getArguments()[originalLoopIndex];
}

ArrayRef<BlockArgument> BuildKrnlLoop::getAllInductionVar() {
  return ArrayRef<BlockArgument>(
      iterBlock->getArguments().begin(), iterBlock->getArguments().end());
}

// TODO: only in the EDSC scope

//====---------------- EDSC Support with Value ---------------------------===//

Value krnl_load(Value memref, ValueRange indices) {
  using namespace mlir::edsc;
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  return ScopedContext::getBuilderRef().create<KrnlLoadOp>(
      ScopedContext::getLocation(), memref, indices);
}

#if 0
Value krnl_load(Value memref, ArrayRef<Value> indices) {
  return krnl_load(memref, ValueRange(indices));
}
#endif

void krnl_store(Value val, Value memref, ValueRange indices) {
  using namespace mlir::edsc;
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  ScopedContext::getBuilderRef().create<KrnlStoreOp>(
      ScopedContext::getLocation(), val, memref, indices);
}

#if 0
void krnl_store(Value val, Value memref, ArrayRef<Value> indices) {
  krnl_store(val, memref, ValueRange(indices));
}
#endif

ValueRange krnl_define_loop(int64_t originalLoopNum) {
  using namespace mlir::edsc;
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  KrnlDefineLoopsOp newOp =
      ScopedContext::getBuilderRef().create<KrnlDefineLoopsOp>(
          ScopedContext::getLocation(), originalLoopNum);
  return newOp.getResults();
}

ValueRange krnl_block(Value loop, int64_t blockSize) {
  using namespace mlir::edsc;
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  return ScopedContext::getBuilderRef()
      .create<KrnlBlockOp>(ScopedContext::getLocation(), loop, blockSize)
      .getResults();
}

void krnl_permute(ArrayRef<Value> loops, ArrayRef<int64_t> map) {
  using namespace mlir::edsc;
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  ScopedContext::getBuilderRef().create<KrnlPermuteOp>(
      ScopedContext::getLocation(), loops, map);
}

#if 0
ValueRange krnl_get_induction_var_value(ArrayRef<Value> loops) {
  using namespace mlir::edsc;
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  return ScopedContext::getBuilderRef()
      .create<KrnlGetInductionVariableValueOp>(
          ScopedContext::getLocation(), loops)
      .getResults();
}
#endif

ValueRange krnl_get_induction_var_value(ValueRange loops) {
  using namespace mlir::edsc;
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  return ScopedContext::getBuilderRef()
      .create<KrnlGetInductionVariableValueOp>(
          ScopedContext::getLocation(), loops)
      .getResults();
}

#if 0
void krnl_iterate(ArrayRef<Value> originalLoop, ArrayRef<Value> optimizedLoop,
    ArrayRef<Value> lb, ArrayRef<Value> ub, ArrayRef<Value> iterArgs,
    function_ref<void(ArrayRef<Value>)> bodyBuilderFn) {
  using namespace mlir::edsc;
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  assert(lb.size() == ub.size() && "expected matching number of lb & ub");
  OpBuilder &builder = ScopedContext::getBuilderRef();
  Location loc = ScopedContext::getLocation();
  KrnlIterateOperandPack pack(builder, originalLoop, optimizedLoop);
  for (int i = 0; i < lb.size(); ++i) {
    pack.pushOperandBound(lb[i]);
    pack.pushOperandBound(ub[i]);
  }
  KrnlIterateOp iterateOp =
      builder.create<KrnlIterateOp>(ScopedContext::getLocation(), pack);
  // auto savedInsertionPoint = builder.saveInsertionPoint();
  Block *iterBlock = &iterateOp.bodyRegion().front();

  if (bodyBuilderFn) { // Scope for the scoped context of the loop.
    ScopedContext nestedContext(builder, loc);
    builder.setInsertionPointToStart(iterBlock);
    bodyBuilderFn(iterArgs);
  }
}

void krnl_iterate(ArrayRef<Value> originalLoop, ArrayRef<Value> lb,
    ArrayRef<Value> ub, ArrayRef<Value> iterArgs,
    function_ref<void(ArrayRef<Value>)> bodyBuilderFn) {
  // When no optimized loops are given, use original for the optimized.
  krnl_iterate(originalLoop, originalLoop, lb, ub, iterArgs, bodyBuilderFn);
}
#endif

void krnl_iterate(ValueRange originalLoops, ValueRange optimizedLoops,
    ValueRange lbs, ValueRange ubs, ValueRange iterArgs,
    function_ref<void(ValueRange)> bodyBuilderFn) {
  using namespace mlir::edsc;
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  assert(lbs.size() == ubs.size() && "expected matching number of lb & ub");
  OpBuilder &builder = ScopedContext::getBuilderRef();
  Location loc = ScopedContext::getLocation();
  // May want to change KrnlIterateOperandPack to use ValueRanges...
  SmallVector<Value, 4> origLoops, optLoops;
  for (auto org : originalLoops)
    origLoops.emplace_back(org);
  for (auto opt : optimizedLoops)
    optLoops.emplace_back(opt);
  KrnlIterateOperandPack pack(builder, origLoops, optLoops);
  for (int i = 0; i < lbs.size(); ++i) {
    pack.pushOperandBound(lbs[i]);
    pack.pushOperandBound(ubs[i]);
  }
  KrnlIterateOp iterateOp =
      builder.create<KrnlIterateOp>(ScopedContext::getLocation(), pack);
  // auto savedInsertionPoint = builder.saveInsertionPoint();
  Block *iterBlock = &iterateOp.bodyRegion().front();

  if (bodyBuilderFn) { // Scope for the scoped context of the loop.
    ScopedContext nestedContext(builder, loc);
    builder.setInsertionPointToStart(iterBlock);
    bodyBuilderFn(iterArgs);
  }
}

void krnl_iterate(ValueRange originalLoops, ValueRange lbs, ValueRange ubs,
    ValueRange iterArgs, function_ref<void(ValueRange)> bodyBuilderFn) {
  // When no optimized loops are given, use original for the optimized.
  krnl_iterate(originalLoops, originalLoops, lbs, ubs, iterArgs, bodyBuilderFn);
}

void krnl_copy_to_buffer(Value bufferMemref, Value memref,
    ArrayRef<Value> starts, Value padValue, ArrayRef<int64_t> tileSize,
    ArrayRef<int64_t> padToNext) {
  using namespace mlir::edsc;
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  ScopedContext::getBuilderRef().create<KrnlCopyToBufferOp>(
      ScopedContext::getLocation(), bufferMemref, memref, starts, padValue,
      tileSize, padToNext);
}

void krnl_copy_to_buffer(
    Value bufferMemref, Value memref, ArrayRef<Value> starts, Value padValue) {
  ArrayRef<int64_t> empty;
  krnl_copy_to_buffer(bufferMemref, memref, starts, padValue, empty, empty);
}

void krnl_copy_from_buffer(Value bufferMemref, Value memref,
    ArrayRef<Value> starts, ArrayRef<int64_t> tileSize) {
  using namespace mlir::edsc;
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  ScopedContext::getBuilderRef().create<KrnlCopyFromBufferOp>(
      ScopedContext::getLocation(), bufferMemref, memref, starts, tileSize);
}
void krnl_copy_from_buffer(
    Value bufferMemref, Value memref, ArrayRef<Value> starts) {
  ArrayRef<int64_t> empty;
  krnl_copy_from_buffer(bufferMemref, memref, starts, empty);
}

void krnl_matmul(Value A, ArrayRef<Value> aStart, Value B,
    ArrayRef<Value> bStart, Value C, ArrayRef<Value> cStart,
    ArrayRef<Value> loops, ArrayRef<Value> computeStarts,
    ArrayRef<Value> globalUBs, ArrayRef<int64_t> computeTileSize,
    ArrayRef<int64_t> aTileSize, ArrayRef<int64_t> bTileSize,
    ArrayRef<int64_t> cTileSize, bool simdize, bool unroll, bool overcompute) {
  using namespace mlir::edsc;
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  ScopedContext::getBuilderRef().create<KrnlMatMulOp>(
      ScopedContext::getLocation(), A, aStart, B, bStart, C, cStart, loops,
      computeStarts[0], computeStarts[1], computeStarts[2], globalUBs[0],
      globalUBs[1], globalUBs[2], computeTileSize, aTileSize, bTileSize,
      cTileSize, simdize, unroll, overcompute);
}

void krnl_matmul(Value A, ArrayRef<Value> aStart, Value B,
    ArrayRef<Value> bStart, Value C, ArrayRef<Value> cStart,
    ArrayRef<Value> loops, ArrayRef<Value> computeStarts,
    ArrayRef<Value> globalUBs, bool simdize, bool unroll, bool overcompute) {
  ArrayRef<int64_t> empty;
  krnl_matmul(A, aStart, B, bStart, C, cStart, loops, computeStarts, globalUBs,
      empty, empty, empty, empty, simdize, unroll, overcompute);
}

//====---------------- EDSC Support with IndexExpr -----------------------===//

Value krnl_load(Value memref, ArrayRef<IndexExpr> indices) {
  SmallVector<Value, 4> indexValues;
  IndexExpr::getValues(indices, indexValues);
  return krnl_load(memref, ValueRange(indexValues));
}

void krnl_store(Value val, Value memref, ArrayRef<IndexExpr> indices) {
  SmallVector<Value, 4> indexValues;
  IndexExpr::getValues(indices, indexValues);
  krnl_store(val, memref, ValueRange(indexValues));
}

void krnl_iterate(ArrayRef<Value> originalLoop, ArrayRef<Value> optimizedLoop,
    ArrayRef<IndexExpr> lb, ArrayRef<IndexExpr> ub, ArrayRef<Value> iterArgs,
    function_ref<void(ArrayRef<Value>)> bodyBuilderFn) {
  using namespace mlir::edsc;
  assert(ScopedContext::getContext() && "EDSC ScopedContext not set up");
  assert(lb.size() == ub.size() && "expected matching number of lb & ub");
  OpBuilder &builder = ScopedContext::getBuilderRef();
  Location loc = ScopedContext::getLocation();
  KrnlIterateOperandPack pack(builder, originalLoop, optimizedLoop);
  for (int i = 0; i < lb.size(); ++i) {
    pack.pushIndexExprBound(lb[i]);
    pack.pushIndexExprBound(ub[i]);
  }
  KrnlIterateOp iterateOp =
      builder.create<KrnlIterateOp>(ScopedContext::getLocation(), pack);
  // auto savedInsertionPoint = builder.saveInsertionPoint();
  Block *iterBlock = &iterateOp.bodyRegion().front();

  if (bodyBuilderFn) { // Scope for the scoped context of the loop.
    ScopedContext nestedContext(builder, loc);
    builder.setInsertionPointToStart(iterBlock);
    bodyBuilderFn(iterArgs);
  }
}

void krnl_iterate(ArrayRef<Value> originalLoop, ArrayRef<IndexExpr> lb,
    ArrayRef<IndexExpr> ub, ArrayRef<Value> iterArgs,
    function_ref<void(ArrayRef<Value>)> bodyBuilderFn) {
  // When no optimized loops are given, use original for the optimized.
  krnl_iterate(originalLoop, originalLoop, lb, ub, iterArgs, bodyBuilderFn);
}

void krnl_copy_to_buffer(Value bufferMemref, Value memref,
    ArrayRef<IndexExpr> starts, Value padValue, ArrayRef<int64_t> tileSize,
    ArrayRef<int64_t> padToNext) {
  SmallVector<Value, 4> startValues;
  IndexExpr::getValues(starts, startValues);
  krnl_copy_to_buffer(
      bufferMemref, memref, startValues, padValue, tileSize, padToNext);
}

void krnl_copy_to_buffer(Value bufferMemref, Value memref,
    ArrayRef<IndexExpr> starts, Value padValue) {
  ArrayRef<int64_t> empty;
  krnl_copy_to_buffer(bufferMemref, memref, starts, padValue, empty, empty);
}

void krnl_copy_from_buffer(Value bufferMemref, Value memref,
    ArrayRef<IndexExpr> starts, ArrayRef<int64_t> tileSize) {
  SmallVector<Value, 4> startValues;
  IndexExpr::getValues(starts, startValues);
  krnl_copy_from_buffer(bufferMemref, memref, starts, tileSize);
}
void krnl_copy_from_buffer(
    Value bufferMemref, Value memref, ArrayRef<IndexExpr> starts) {
  ArrayRef<int64_t> empty;
  krnl_copy_from_buffer(bufferMemref, memref, starts, empty);
}

} // namespace mlir
