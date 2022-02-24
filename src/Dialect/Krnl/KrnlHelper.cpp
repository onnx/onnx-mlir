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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
void KrnlIterateOperandPack::pushIndexExprBound(IndexExpr expr, bool isLb) {
  if (expr.isLiteral()) {
    pushConstantBound(expr.getLiteral());
  } else if (expr.isAffine() && !expr.isPredType()) {
    AffineMap map;
    SmallVector<Value, 4> list;
    expr.getAffineMapAndOperands(map, list);
    pushAffineMapBound(map, list);
  } else {
    Value val = expr.getValue();
    if ((val.getDefiningOp<AffineMinOp>() && !isLb) ||
        (val.getDefiningOp<AffineMaxOp>() && isLb)) {
      // Have a Affine Min in an upper bound computation, or have an Affine Max
      // in a lower bound computation,  will extract the list of affine min/max
      // for the loop bounds.
      AffineMap map;
      SmallVector<Value, 4> list;
      expr.getAffineMapAndOperands(map, list);
      pushAffineMapBound(map, list);
    } else {
      // Assume the expr is loop invariant if there is any outer loop
      pushOperandBound(val);
    }
  }
}

void KrnlIterateOperandPack::pushIndexExprsBound(
    SmallVectorImpl<IndexExpr> &exprVector) {
  SmallVector<AffineExpr, 4> AEVector;
  // Important to get the affine expressions before getting the num
  // Dim/Symbols as it may add some dims and symbol itself.
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

//====---------------- BuildKrnlLoop --------------------------------===//

BuildKrnlLoop::BuildKrnlLoop(
    ConversionPatternRewriter &rewriter, Location loc, int loopNum)
    : rewriter(rewriter), loc(loc), originalLoopNum(loopNum), pack(nullptr),
      pushCount(0), createdDefineOp(false), createdIterateOp(false) {
  if (originalLoopNum < 0)
    emitError(loc, "Expected non-negative number of original loops.");
}

BuildKrnlLoop::BuildKrnlLoop(
    ConversionPatternRewriter &rewriter, Location loc, Value memRefOperand)
    : BuildKrnlLoop(rewriter, loc,
          memRefOperand.getType().cast<MemRefType>().getShape().size()) {}

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
            .create<memref::DimOp>(
                loc, upperBoundMemRefOperand, upperBoundMemRefIndex)
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

// This function satisfies the ArrayValueIndexCapture::DenseElementsAttr
// lambda type, using ONNX and Krnl operations.
DenseElementsAttr getDenseElementAttributeFromKrnlValue(Value value) {
  KrnlGlobalOp globalOp =
      dyn_cast_or_null<mlir::KrnlGlobalOp>(value.getDefiningOp());
  if (globalOp) {
    if (globalOp.value().hasValue())
      return globalOp.valueAttr().dyn_cast<DenseElementsAttr>();
  }
  return nullptr;
}

// This function satisfies the ArrayValueIndexCapture::LoadVal lambda
// type, using Krnl operations.
Value loadDenseElementArrayValueAtIndex(
    OpBuilder &rewriter, Location loc, Value array, int64_t index) {
  // Scalar tensor.
  if (array.getType().cast<ShapedType>().getShape().size() == 0)
    return rewriter.create<KrnlLoadOp>(loc, array);
  Attribute constAttr = rewriter.getIntegerAttr(rewriter.getIndexType(), index);
  Value indexVal = rewriter.create<arith::ConstantOp>(loc, constAttr);
  SmallVector<Value, 1> memrefVal = {indexVal};
  return rewriter.create<KrnlLoadOp>(loc, array, memrefVal);
}

//====---------------- Support for simple transpose -------------------===//

// create an identity
void generateIndexMap(
    SmallVectorImpl<int64_t> &map, int64_t size, bool transposeInner2) {
  for (int i = 0; i < size; ++i)
    map.emplace_back(i); // Indentity map.
  if (size < 2)
    return;
  if (transposeInner2) {
    map[size - 2] = size - 1;
    map[size - 1] = size - 2;
  }
}

//====---------------- Support for Krnl Builder ----------------------===//

Value KrnlBuilder::load(Value memref, ValueRange indices) const {
  return b.create<KrnlLoadOp>(loc, memref, indices);
}

Value KrnlBuilder::loadIE(Value memref, ArrayRef<IndexExpr> indices) const {
  SmallVector<Value, 4> indexValues;
  IndexExpr::getValues(indices, indexValues);
  return b.create<KrnlLoadOp>(loc, memref, indexValues);
}

void KrnlBuilder::store(Value val, Value memref, ValueRange indices) const {
  b.create<KrnlStoreOp>(loc, val, memref, indices);
}

void KrnlBuilder::storeIE(
    Value val, Value memref, ArrayRef<IndexExpr> indices) const {
  SmallVector<Value, 4> indexValues;
  IndexExpr::getValues(indices, indexValues);
  b.create<KrnlStoreOp>(loc, val, memref, indexValues);
}

Value KrnlBuilder::vectorTypeCast(Value sourceMemref, int64_t vectorLen) {
  return b.create<KrnlVectorTypeCastOp>(loc, sourceMemref, vectorLen);
}

ValueRange KrnlBuilder::defineLoops(int64_t originalLoopNum) {
  return b.create<KrnlDefineLoopsOp>(loc, originalLoopNum).getResults();
}

ValueRange KrnlBuilder::block(Value loop, int64_t blockSize) {
  return b.create<KrnlBlockOp>(loc, loop, blockSize).getResults();
}

void KrnlBuilder::permute(ValueRange loops, ArrayRef<int64_t> map) {
  b.create<KrnlPermuteOp>(loc, loops, map);
}

ValueRange KrnlBuilder::getInductionVarValue(ValueRange loops) {
  return b.create<KrnlGetInductionVariableValueOp>(loc, loops).getResults();
}

void KrnlBuilder::iterate(ValueRange originalLoops, ValueRange optimizedLoops,
    ValueRange lbs, ValueRange ubs,
    function_ref<void(KrnlBuilder &createKrnl, ValueRange indices)>
        bodyBuilderFn) {
  // Check that originalLoops, lbs, and ubs have the same rank.
  assert(originalLoops.size() == lbs.size() && "expected same rank");
  assert(originalLoops.size() == ubs.size() && "expected same rank");
  ValueRange empty;
  b.create<KrnlIterateOp>(loc, originalLoops, optimizedLoops, lbs, ubs, empty,
      [&](OpBuilder &builder, Location loc, ValueRange args) {
        KrnlBuilder createKrnl(builder, loc);
        ValueRange indices = createKrnl.getInductionVarValue(optimizedLoops);
        bodyBuilderFn(createKrnl, indices);
      });
}

void KrnlBuilder::iterateIE(ValueRange originalLoops, ValueRange optimizedLoops,
    ArrayRef<IndexExpr> lbs, ArrayRef<IndexExpr> ubs,
    function_ref<void(KrnlBuilder &createKrnl, ValueRange indices)>
        bodyBuilderFn) {
  // Check that originalLoops, lbs, and ubs have the same rank.
  assert(originalLoops.size() == lbs.size() && "expected same rank");
  assert(originalLoops.size() == ubs.size() && "expected same rank");
  ValueRange empty;
  b.create<KrnlIterateOp>(loc, originalLoops, optimizedLoops, lbs, ubs, empty,
      [&](OpBuilder &builder, Location loc, ValueRange args) {
        KrnlBuilder createKrnl(builder, loc);
        ValueRange indices = createKrnl.getInductionVarValue(optimizedLoops);
        bodyBuilderFn(createKrnl, indices);
      });
}

void KrnlBuilder::copyToBuffer(Value bufferMemref, Value sourceMemref,
    ValueRange starts, Value padValue, ArrayRef<int64_t> tileSize,
    ArrayRef<int64_t> padToNext, bool transpose) {
  b.create<KrnlCopyToBufferOp>(loc, bufferMemref, sourceMemref, starts,
      padValue, tileSize, padToNext, transpose);
}

void KrnlBuilder::copyToBuffer(Value bufferMemref, Value sourceMemref,
    ValueRange starts, Value padValue, bool transpose) {
  b.create<KrnlCopyToBufferOp>(
      loc, bufferMemref, sourceMemref, starts, padValue, transpose);
}

void KrnlBuilder::copyFromBuffer(Value bufferMemref, Value memref,
    ValueRange starts, ArrayRef<int64_t> tileSize) {
  b.create<KrnlCopyFromBufferOp>(loc, bufferMemref, memref, starts, tileSize);
}

void KrnlBuilder::copyFromBuffer(
    Value bufferMemref, Value memref, ValueRange starts) {
  b.create<KrnlCopyFromBufferOp>(loc, bufferMemref, memref, starts);
}

void KrnlBuilder::matmul(Value A, ValueRange aStart, Value B, ValueRange bStart,
    Value C, ValueRange cStart, ValueRange loops, ValueRange computeStarts,
    ValueRange globalUBs, ArrayRef<int64_t> computeTileSize,
    ArrayRef<int64_t> aTileSize, ArrayRef<int64_t> bTileSize,
    ArrayRef<int64_t> cTileSize, bool simdize, bool unroll, bool overcompute) {
  b.create<KrnlMatMulOp>(loc, A, aStart, B, bStart, C, cStart, loops,
      computeStarts[0], computeStarts[1], computeStarts[2], globalUBs[0],
      globalUBs[1], globalUBs[2], computeTileSize, aTileSize, bTileSize,
      cTileSize, simdize, unroll, overcompute);
}

void KrnlBuilder::matmul(Value A, ValueRange aStart, Value B, ValueRange bStart,
    Value C, ValueRange cStart, ValueRange loops, ValueRange computeStarts,
    ValueRange globalUBs, bool simdize, bool unroll, bool overcompute) {
  b.create<KrnlMatMulOp>(loc, A, aStart, B, bStart, C, cStart, loops,
      computeStarts[0], computeStarts[1], computeStarts[2], globalUBs[0],
      globalUBs[1], globalUBs[2], simdize, unroll, overcompute);
}

Value KrnlBuilder::constant(MemRefType type, StringRef name,
    DenseElementsAttr value, Optional<IntegerAttr> offset,
    Optional<IntegerAttr> alignment) const {
  static int32_t constantID = 0;
  return b.create<KrnlGlobalOp>(loc, type, b.getI64ArrayAttr(type.getShape()),
      b.getStringAttr(name + std::to_string(constantID++)), value,
      offset.hasValue() ? offset.getValue() : nullptr,
      alignment.hasValue() ? alignment.getValue() : nullptr);
}

void KrnlBuilder::memcpy(Value dest, Value src, Value size) const {
  b.create<KrnlMemcpyOp>(loc, dest, src, size);
}

void KrnlBuilder::memset(Value dest, Value val) const {
  b.create<KrnlMemsetOp>(loc, dest, val);
}

Value KrnlBuilder::strncmp(Value str1, Value str2, Value len) const {
  return b.create<KrnlStrncmpOp>(loc, b.getI32Type(), str1, str2, len);
}

Value KrnlBuilder::strlen(Value str) const {
  return b.create<KrnlStrlenOp>(loc, b.getI64Type(), str);
}

Value KrnlBuilder::findIndex(Value input, Value G, Value V, Value len) const {
  return b.create<KrnlFindIndexOp>(loc, b.getIndexType(), input, G, V, len);
}

void KrnlBuilder::printTensor(StringRef msg, Value input) const {
  b.create<KrnlPrintTensorOp>(loc, msg, input);
}

void KrnlBuilder::printf(StringRef msg) const {
  Value noneValue;
  b.create<KrnlPrintOp>(loc, msg, noneValue);
}

void KrnlBuilder::printf(StringRef msg, Value input, Type inputType) const {
  StringRef format;
  TypeSwitch<Type>(inputType)
      .Case<mlir::Float16Type>([&](mlir::Float16Type) { format = "%g\n"; })
      .Case<mlir::Float32Type>([&](mlir::Float32Type) { format = "%g\n"; })
      .Case<mlir::Float64Type>([&](mlir::Float64Type) { format = "%g\n"; })
      .Case<IntegerType>([&](IntegerType type) {
        switch (type.getWidth()) {
        case 1:
        case 8:
        case 16:
        case 32:
          format = type.isUnsigned() ? "%u\n" : "%d\n";
          break;
        case 64:
          format = type.isUnsigned() ? "%llu\n" : "%lld\n";
          break;
        }
      })
      .Case<IndexType>([&](IndexType type) { format = "%lld\n"; })
      .Case<StringType>([&](StringType type) { format = "%s\n"; })
      .Case<LLVM::LLVMPointerType>(
          [&](LLVM::LLVMPointerType type) { format = "%s\n"; })
      .Default([&](Type type) {
        llvm::errs() << "type: " << type << "\n";
        llvm_unreachable("Unhandled type");
      });

  std::string concat(msg.str() + format.str());
  StringRef newFormat(concat);
  b.create<KrnlPrintOp>(loc, newFormat, input);
}

//====---------------- Common helper functions --------------------------===//

bool isKrnlGlobalConstant(Value result) {
  Operation *op = result.getDefiningOp();

  KrnlGlobalOp constOp = llvm::dyn_cast_or_null<KrnlGlobalOp>(op);
  // Not a constant.
  if (!constOp)
    return false;

  if (!(op->getAttrOfType<::mlir::Attribute>("value")))
    return false;

  return true;
}

} // namespace mlir
