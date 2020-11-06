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

#include "KrnlOps.hpp"

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
    free(pack);
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

int BuildKrnlLoop::pushBounds(
    IndexExprContext &context, int64_t lowerBound, IndexExpr upperBound) {
  if (upperBound.isLiteral()) {
    return pushBounds(0, upperBound.getLiteral());
  }
  return pushBounds(0, upperBound.getValue());
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

} // namespace mlir
