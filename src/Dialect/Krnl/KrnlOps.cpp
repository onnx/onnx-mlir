/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------- KrnlOps.cpp - Krnl Operations -----------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains implementations of krnl operations.
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <queue>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"

#include "KrnlHelper.hpp"

#include "KrnlOps.hpp"

using namespace mlir;

KrnlOpsDialect::KrnlOpsDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<KrnlOpsDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "src/Dialect/Krnl/KrnlOps.cpp.inc"
      >();
  addTypes<LoopType>();
}

namespace mlir {

//===----------------------------------------------------------------------===//
// KrnlDefineLoopsOp
//===----------------------------------------------------------------------===//

void KrnlDefineLoopsOp::build(
    OpBuilder &builder, OperationState &result, int64_t num_loops) {
  // Create the same number of dimension handlers as the number of
  // dimensions in the associated integer set.
  result.types.append(num_loops, LoopType::get(builder.getContext()));
  result.addAttribute(
      getNumLoopsAttrName(), builder.getI64IntegerAttr(num_loops));
}

void print(OpAsmPrinter &p, KrnlDefineLoopsOp &op) {
  auto numLoopAttr =
      op->getAttrOfType<IntegerAttr>(KrnlDefineLoopsOp::getNumLoopsAttrName());
  p << "krnl.define_loops " << numLoopAttr.getValue().getSExtValue();
}

ParseResult parseKrnlDefineLoopsOp(
    OpAsmParser &parser, OperationState &result) {
  // Parse the attribute indicating number of loops defined.
  IntegerAttr numLoops;
  auto &builder = parser.getBuilder();
  auto intType = builder.getIntegerType(64);
  if (parser.parseAttribute(numLoops, intType,
          KrnlDefineLoopsOp::getNumLoopsAttrName(), result.attributes))
    return failure();

  auto loopTypes = llvm::SmallVector<Type, 4>(
      numLoops.getValue().getSExtValue(), LoopType::get(builder.getContext()));
  if (parser.addTypesToList(loopTypes, result.types))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// KrnlIterateOp
//===----------------------------------------------------------------------===//

/*!
 * Build a Krnl Dialect iterate operation.
 * input_loops: a collection of input krnl.loops being optimized.
 * optimized_loops: a collection of optimized (scheduled) krnl.loops.
 * operand_bounds: a collection of SSA value bounds.
 * const_bounds: a collection of constant bounds.
 * bound_types: a collection of integer values indicating how bounds are given.
 *   0 : bound is given as an integer in const_bounds.
 *   1 : bound is given as an operand in operand_bounds.
 *   2 : bound is given as an affine map. (TODO).
 *
 * The following example illustrates how induction variable bounds are parsed
 * from builder function inputs:
 *
 * - operand_bounds = [N, M]
 * - const_bounds = [10, 20]
 * - bound_types = [0, 1, 1, 0]
 *
 * Then the bounds will be parsed as:
 *   %i0 = 10 to N : %i1 = M to 20
 */
void KrnlIterateOp::build(OpBuilder &builder, OperationState &result,
    KrnlIterateOperandPack operandPack) {
  // Record optimized loops and the number of such loops.
  result.addOperands(operandPack.getOperands());
  result.addAttribute(
      KrnlIterateOp::getBoundsAttrName(), operandPack.getAttributes());

  result.addAttribute(getNumOptimizedLoopsAttrName(),
      builder.getI64IntegerAttr(operandPack.getNumOptimizedLoops()));

  // Create a region and a block for the body. The arguments of the region are
  // the loop induction variables; there can be multiple induction variables
  // associated with the same krnl.iterate operation.
  Region *bodyRegion = result.addRegion();
  auto *body = new Block();
  auto body_args = llvm::SmallVector<Type, 4>(
      operandPack.getNumInputLoops(), IndexType::get(builder.getContext()));
  body->addArguments(body_args);
  bodyRegion->push_back(body);

  ensureTerminator(*bodyRegion, builder, result.location);
}

void print(OpAsmPrinter &p, KrnlIterateOp &op) {
  p << "krnl.iterate(";
  // Print optimized loops:
  auto numOptimizedLoops = op.getNumOptimizedLoops();
  p.printOperands(op.operand_begin(), op.operand_begin() + numOptimizedLoops);
  p << ") with (";

  // In the event where body region has been lowered, do not print body.
  if (op.bodyRegion().empty()) {
    p << ")";
    return;
  }
  auto inductionVars = op.bodyRegion().begin()->getArguments();
  auto boundItr =
      op->getAttrOfType<ArrayAttr>(KrnlIterateOp::getBoundsAttrName())
          .getValue()
          .begin();
  auto operandItr = op.operand_begin() + numOptimizedLoops;

  std::string delimiter;
  for (auto &var : inductionVars) {
    p << delimiter;
    p.printOperand(*operandItr++);
    p << " -> ";
    p.printOperand(var);
    p << " = ";
    onnx_mlir::printBound(
        (*boundItr++).cast<AffineMapAttr>(), operandItr, "max", p);
    p << " to ";
    onnx_mlir::printBound(
        (*boundItr++).cast<AffineMapAttr>(), operandItr, "min", p);
    delimiter = ", ";
  }

  p << ")";
  p.printRegion(op.bodyRegion(), /*printEntryBlockArgs=*/false,
      /*printBlockTerminators=*/false);
}

ParseResult parseKrnlIterateOp(OpAsmParser &parser, OperationState &result) {
  auto builder = parser.getBuilder();
  auto context = builder.getContext();
  onnx_mlir::KrnlDialectOperandParser operandParser(parser);

  // Parse optimized loops:
  SmallVector<OpAsmParser::OperandType, 4> optimizedLoopRefs;
  if (parser.parseOperandList(
          optimizedLoopRefs, OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(optimizedLoopRefs,
          LoopType::get(result.getContext()), result.operands))
    return failure();

  // Record how many optimized loops did we parse.
  result.addAttribute(KrnlIterateOp::getNumOptimizedLoopsAttrName(),
      builder.getI64IntegerAttr(optimizedLoopRefs.size()));

  // Parse input loops and their lower and upper bounds.
  SmallVector<OpAsmParser::OperandType, 4> inductionVarRefs;
  SmallVector<Attribute, 4> boundMaps;

  if (parser.parseKeyword("with") || parser.parseLParen())
    return failure();

  // A function to parse a lower or upper bound.
  auto parseBound = [&result, &builder, &parser, &operandParser, &boundMaps](
                        bool isUpper) -> ParseResult {
    // 'min' / 'max' prefixes are generally syntactic sugar, but are required if
    // the map has multiple results.
    bool failedToParsedMinMax =
        failed(parser.parseOptionalKeyword(isUpper ? "min" : "max"));

    // Try parse an SSA operand.
    if (succeeded(operandParser.ParseOptionalOperand(
            builder.getIndexType(), result.operands))) {
      AffineMap map = builder.getSymbolIdentityMap();
      boundMaps.emplace_back(AffineMapAttr::get(map));
      return success();
    }

    // Bound is not an SSA id, then it must be an integer.
    // Parse an integer constant attribute.
    // Get the attribute location.
    llvm::SMLoc attrLoc = parser.getCurrentLocation();
    Attribute boundAttr;
    NamedAttrList tempBoundAttrContainer;
    if (parser.parseAttribute(
            boundAttr, builder.getIndexType(), "temp", tempBoundAttrContainer))
      return failure();

    if (auto affineMapAttr = boundAttr.dyn_cast<AffineMapAttr>()) {
      unsigned currentNumOperands = result.operands.size();
      unsigned numDims = 0;
      if (parseDimAndSymbolList(parser, result.operands, numDims))
        return failure();

      auto map = affineMapAttr.getValue();
      if (map.getNumDims() != numDims)
        return parser.emitError(parser.getNameLoc(),
            "dim operand count and integer set dim count must match");

      unsigned numDimAndSymbolOperands =
          result.operands.size() - currentNumOperands;
      if (numDims + map.getNumSymbols() != numDimAndSymbolOperands)
        return parser.emitError(parser.getNameLoc(),
            "symbol operand count and integer set symbol count must match");

      // If the map has multiple results, make sure that we parsed the min/max
      // prefix.
      if (map.getNumResults() > 1 && failedToParsedMinMax) {
        if (isUpper)
          return parser.emitError(attrLoc,
              "upper loop bound affine map with multiple "
              "results requires 'min' prefix");
        return parser.emitError(attrLoc,
            "lower loop bound affine mapwith "
            "multiple results requires 'max' prefix");
      }
      boundMaps.emplace_back(AffineMapAttr::get(map));
      return success();
    }

    if (auto integerAttr = boundAttr.dyn_cast<IntegerAttr>()) {
      AffineMap map =
          builder.getConstantAffineMap(integerAttr.getValue().getSExtValue());
      boundMaps.emplace_back(AffineMapAttr::get(map));
    }
    return success();
  };

  while (failed(parser.parseOptionalRParen())) {
    // Parse an input loop operand;
    operandParser.ParseOperand(LoopType::get(context), result.operands);
    parser.parseArrow();

    // Parse induction variable.
    OpAsmParser::OperandType inductionVar;
    if (parser.parseRegionArgument(inductionVar) || parser.parseEqual())
      return failure();
    inductionVarRefs.emplace_back(inductionVar);

    // Parse bound par (min to max).
    if (parseBound(/*isUpper=*/false) || parser.parseKeyword("to") ||
        parseBound(/*isUpper=*/true))
      return failure();

    // We may fail to parse a comma if an operand bound is followed by
    // a comma and the next input loop operand, in which case
    // the entire "{operand bound}, {input_loop_operand}" sequence will
    // be parsed as an operand list.
    parser.parseOptionalComma();
  }

  // At this point, there shouldn't be any operands left to parse.
  if (operandParser.hasOperandLeft())
    return parser.emitError(parser.getCurrentLocation());
  result.addAttribute(
      KrnlIterateOp::getBoundsAttrName(), builder.getArrayAttr(boundMaps));

  Region *region = result.addRegion();
  SmallVector<Type, 4> inductionVarTypes(
      inductionVarRefs.size(), builder.getIndexType());
  if (parser.parseRegion(*region, inductionVarRefs, inductionVarTypes))
    return failure();

  // Ensure iterate region is closed off with krnl.terminate.
  KrnlIterateOp::ensureTerminator(
      *region, parser.getBuilder(), result.location);

  return success();
}

static LogicalResult verify(KrnlIterateOp op) {
  // TODO: Verify number of induction variable bounds matches the number of
  // input loops.
  return success();
}

//===----------------------------------------------------------------------===//
// KrnlEntryPointOp
//===----------------------------------------------------------------------===//

void KrnlEntryPointOp::build(mlir::OpBuilder &builder, OperationState &state,
    SymbolRefAttr funcAttr, IntegerAttr numInputs, IntegerAttr numOutputs,
    StringAttr signature) {
  state.addAttribute(KrnlEntryPointOp::getEntryPointFuncAttrName(), funcAttr);
  state.addAttribute(KrnlEntryPointOp::getNumInputsAttrName(), numInputs);
  state.addAttribute(KrnlEntryPointOp::getNumOutputsAttrName(), numOutputs);
  state.addAttribute(KrnlEntryPointOp::getSignatureAttrName(), signature);
}

//===----------------------------------------------------------------------===//
// KrnlBlockOp
//===----------------------------------------------------------------------===//

void KrnlBlockOp::build(::mlir::OpBuilder &odsBuilder,
    ::mlir::OperationState &odsState, Value odsLoop, int64_t odsTileSize) {
  Type loopType = LoopType::get(odsBuilder.getContext());
  TypeRange blockResType({loopType, loopType});
  build(odsBuilder, odsState, blockResType, odsLoop,
      odsBuilder.getI64IntegerAttr(odsTileSize));
}

//===----------------------------------------------------------------------===//
// KrnlPermuteOp
//===----------------------------------------------------------------------===//

void KrnlPermuteOp::build(::mlir::OpBuilder &odsBuilder,
    ::mlir::OperationState &odsState, ValueRange odsLoops,
    ArrayRef<int64_t> odsMap) {
  int64_t rank = odsLoops.size();
  assert(rank >= 2 && "permute needs 2 or more loops");
  assert(odsMap.size() == rank && "loop and size size must be identical");
  for (int i = 0; i < rank; ++i) {
    assert(odsMap[i] >= 0 && odsMap[i] < rank && "bad permute");
    for (int j = i + 1; j < rank; ++j)
      assert(
          odsMap[i] != odsMap[j] && "map should be a strict permute pattern");
  }
  ValueRange loopRange(odsLoops);
  ArrayAttr mapAttr = odsBuilder.getI64ArrayAttr(odsMap);
  build(odsBuilder, odsState, loopRange, mapAttr);
}

//===----------------------------------------------------------------------===//
// KrnlGetInductionVariableValueOp
//===----------------------------------------------------------------------===//

void KrnlGetInductionVariableValueOp::build(::mlir::OpBuilder &odsBuilder,
    ::mlir::OperationState &odsState, ValueRange odsLoops) {
  int64_t rank = odsLoops.size();
  Type loopType = LoopType::get(odsBuilder.getContext());
  SmallVector<Type, 6> types(rank, odsBuilder.getIndexType());
  TypeRange typeRange(types);
  ArrayRef<NamedAttribute> noAttr({});
  build(odsBuilder, odsState, typeRange, odsLoops, noAttr);
}

//===----------------------------------------------------------------------===//
// KrnlDummyCastOp
//===----------------------------------------------------------------------===//

void KrnlDummyCastOp::build(
    OpBuilder &builder, OperationState &state, Value in, Type outType) {
  state.operands.emplace_back(in);
  state.types.emplace_back(outType);
}

//===----------------------------------------------------------------------===//
// KrnlVectorTypeCastOp
//===----------------------------------------------------------------------===//

// Use the sourceMemRef as a template to create the result type, where all the
// dimensions are copied but for the last one that is divided by vectorLen, as
// the elementary type of the result is vectorLen x elementary type. Supports
// only 1D vectors.
void KrnlVectorTypeCastOp::build(OpBuilder &builder, OperationState &state,
    Value sourceMemRef, int64_t vectorLen) {
  MemRefType sourceType = sourceMemRef.getType().cast<MemRefType>();
  Type elementType = sourceType.getElementType();
  auto sourceShape = sourceType.getShape();
  int rank = sourceShape.size();
  VectorType vecType = VectorType::get({vectorLen}, elementType);
  SmallVector<int64_t, 4> vectorShape;
  for (int i = 0; i < rank - 1; ++i)
    vectorShape.emplace_back(sourceShape[i]);
  assert(sourceShape[rank - 1] > 0 &&
         "expected compile time, strictly positive last dim");
  assert(sourceShape[rank - 1] % vectorLen == 0 &&
         "last dim must be a multiple of vector length");
  vectorShape.emplace_back(sourceShape[rank - 1] / vectorLen);
  MemRefType resultType = MemRefType::get(vectorShape, vecType);
  build(builder, state, resultType, sourceMemRef);
}

bool KrnlVectorTypeCastOp::areCastCompatible(Type a, Type b) {
  auto aT = a.dyn_cast<MemRefType>();
  auto bT = b.dyn_cast<MemRefType>();

  if (!aT || !bT)
    return false;

  if (aT.getAffineMaps() != bT.getAffineMaps())
    return false;

  if (aT.getMemorySpace() != bT.getMemorySpace())
    return false;

  if (aT.getRank() != bT.getRank())
    return false;

  // With rank 0, there is no vec cast.
  if (aT.getRank() == 0)
    return false;

  // Should have the same shape up until the last n-1 dimensions.
  // Replace this by std::equal.
  for (unsigned i = 0, e = aT.getRank() - 1; i < e; ++i)
    if (aT.getDimSize(i) != bT.getDimSize(i))
      return false;

  // Source memref can't have vector element type.
  if (auto shapedEltType = aT.getElementType().dyn_cast<ShapedType>())
    return false;

  auto shapedEltTypeB = bT.getElementType().dyn_cast<ShapedType>();
  if (!shapedEltTypeB)
    return false;

  auto eltA = aT.getElementType();
  auto eltB = shapedEltTypeB.getElementType();
  if (eltA != eltB)
    return false;

  int64_t lastDimA = aT.getShape().back();
  int64_t lastDimB = bT.getShape().back();

  // If one of them is dynamic but not the other, they are incompatible.
  if (lastDimA * lastDimB < 0)
    return false;

  if (lastDimA != MemRefType::kDynamicSize &&
      lastDimB != MemRefType::kDynamicSize &&
      lastDimA / shapedEltTypeB.getNumElements() != lastDimB)
    return false;

  return true;
}

/// This is a common class used for patterns of the form
/// "someop(memrefcast) -> someop".  It folds the source of any memref_cast
/// into the root operation directly.
static LogicalResult foldMemRefCast(Operation *op) {
  bool folded = false;
  for (OpOperand &operand : op->getOpOperands()) {
    auto cast = operand.get().getDefiningOp<MemRefCastOp>();
    if (cast && !cast.getOperand().getType().isa<UnrankedMemRefType>()) {
      operand.set(cast.getOperand());
      folded = true;
    }
  }
  return success(folded);
}

OpFoldResult KrnlVectorTypeCastOp::fold(ArrayRef<Attribute> operands) {
  if (Value folded = impl::foldCastOp(*this))
    return folded;
  return succeeded(foldMemRefCast(*this)) ? getResult() : Value();
}

MutableOperandRange KrnlSpecializedKernel::getLoopRefs() {
  return loopsMutable();
}

//===----------------------------------------------------------------------===//
// KrnlMatMulOp
//===----------------------------------------------------------------------===//

void KrnlMatMulOp::build(::mlir::OpBuilder &odsBuilder,
    ::mlir::OperationState &odsState, Value odsA, ValueRange aOdsStart,
    Value odsB, ValueRange bOdsStart, Value odsC, ValueRange cOdsStart,
    ValueRange odsLoops, Value iOdsComputeStart, Value jOdsComputeStart,
    Value kOdsComputeStart, Value iOdsGlobalUB, Value jOdsGlobalUB,
    Value kOdsGlobalUB, ArrayRef<int64_t> odsComputeTileSize,
    ArrayRef<int64_t> aOdsTileSize, ArrayRef<int64_t> bOdsTileSize,
    ArrayRef<int64_t> cOdsTileSize, bool odsSimdize, bool odsUnroll,
    bool odsOvercompute) {
  // Massage types.
  ValueRange loopRange(odsLoops);
  ArrayAttr computeTileSizeAttr =
      odsBuilder.getI64ArrayAttr(odsComputeTileSize);
  ArrayAttr aTileSizeAttr = odsBuilder.getI64ArrayAttr(aOdsTileSize);
  ArrayAttr bTileSizeAttr = odsBuilder.getI64ArrayAttr(bOdsTileSize);
  ArrayAttr cTileSizeAttr = odsBuilder.getI64ArrayAttr(cOdsTileSize);

  build(odsBuilder, odsState, odsA, aOdsStart, odsB, bOdsStart, odsC, cOdsStart,
      loopRange, iOdsComputeStart, jOdsComputeStart, kOdsComputeStart,
      iOdsGlobalUB, jOdsGlobalUB, kOdsGlobalUB, computeTileSizeAttr,
      aTileSizeAttr, bTileSizeAttr, cTileSizeAttr, odsSimdize, odsUnroll,
      odsOvercompute);
}

static LogicalResult verify(KrnlMatMulOp op) {
  KrnlMatMulOpAdaptor operandAdaptor = KrnlMatMulOpAdaptor(op);
  int64_t aRank =
      operandAdaptor.A().getType().cast<MemRefType>().getShape().size();
  int64_t bRank =
      operandAdaptor.B().getType().cast<MemRefType>().getShape().size();
  int64_t cRank =
      operandAdaptor.C().getType().cast<MemRefType>().getShape().size();
  if (!(aRank >= 2 && bRank >= 2 && cRank >= 2))
    return op.emitOpError("currently only support ranks >=2");
  if (operandAdaptor.aMemStart().size() != aRank)
    return op.emitOpError("aMemStart should have same rank as memref A");
  if (operandAdaptor.bMemStart().size() != bRank)
    return op.emitOpError("bMemStart should have same rank as memref A");
  if (operandAdaptor.cMemStart().size() != cRank)
    return op.emitOpError("cMemStart should have same rank as memref A");
  if (operandAdaptor.loops().size() != 3)
    return op.emitOpError("loops rank should be 3 (i,j,k)");
  ArrayAttr computeAttr = operandAdaptor.computeTileSize();
  if (computeAttr && !(computeAttr.size() == 0 || computeAttr.size() == 3))
    return op.emitOpError("computeTileSize rank should be 0 or 3");
  ArrayAttr aTileAttr = operandAdaptor.aTileSize();
  if (aTileAttr && !(aTileAttr.size() == 0 || aTileAttr.size() == 2))
    return op.emitOpError("aTileSize rank should be 0 or 2");
  ArrayAttr bTileAttr = operandAdaptor.bTileSize();
  if (bTileAttr && !(bTileAttr.size() == 0 || bTileAttr.size() == 2))
    return op.emitOpError("bTileSize rank should be 0 or 2");
  ArrayAttr cTileAttr = operandAdaptor.cTileSize();
  if (cTileAttr && !(cTileAttr.size() == 0 || cTileAttr.size() == 2))
    return op.emitOpError("cTileSize rank should be 0 or 2");
  return success();
}

MutableOperandRange KrnlMatMulOp::getLoopRefs() { return loopsMutable(); }

//===----------------------------------------------------------------------===//
// KrnlCopyToBufferOp
//===----------------------------------------------------------------------===//

void KrnlCopyToBufferOp::build(::mlir::OpBuilder &odsBuilder,
    ::mlir::OperationState &odsState, Value odsBufferMemref, Value odsMemref,
    ValueRange odsStarts, Value odsPadValue, ArrayRef<int64_t> odsTileSize,
    ArrayRef<int64_t> odsPadToNext, bool odsTranspose) {
  // Massage types.
  ValueRange startsRange(odsStarts);
  ArrayAttr tileSizeAttr = odsBuilder.getI64ArrayAttr(odsTileSize);
  ArrayAttr padToNextAttr = odsBuilder.getI64ArrayAttr(odsPadToNext);
  build(odsBuilder, odsState, odsBufferMemref, odsMemref, startsRange,
      odsPadValue, tileSizeAttr, padToNextAttr, odsTranspose);
}

static LogicalResult verify(KrnlCopyToBufferOp op) {
  KrnlCopyToBufferOpAdaptor opAdaptor = KrnlCopyToBufferOpAdaptor(op);
  MemRefBoundsIndexCapture buffCapture(opAdaptor.buffer());
  MemRefBoundsIndexCapture srcCapture(opAdaptor.source());
  int64_t bufferRank = buffCapture.getRank();
  int64_t srcRank = srcCapture.getRank();
  int64_t startRank = opAdaptor.starts().size();
  if (!buffCapture.areAllLiteral())
    return op.emitOpError("buffer expect constant dimensions");
  if (srcRank < bufferRank)
    return op.emitOpError("Rank of memref cannot be smaller than buffer");
  if (startRank != srcRank)
    return op.emitOpError("Rank of starts and memrefs must be identical");
  if (opAdaptor.tileSize()) {
    int64_t tRank = opAdaptor.tileSize().size();
    if (!(tRank == 0 || tRank == bufferRank))
      return op.emitOpError("Rank of tileSize must be identical to buffer");
  }
  if (opAdaptor.padToNext()) {
    int64_t padRank = opAdaptor.padToNext().size();
    if (!(padRank == 0 || padRank == bufferRank))
      return op.emitOpError("Rank of padToNext must be identical to buffer");
  }
  if (opAdaptor.transpose()) {
    if (bufferRank < 2)
      return op.emitOpError(
          "To transpose buffer, its rank must be greater than 1");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// KrnlCopyFromBufferOp
//===----------------------------------------------------------------------===//

void KrnlCopyFromBufferOp::build(::mlir::OpBuilder &odsBuilder,
    ::mlir::OperationState &odsState, Value odsBufferMemref, Value odsMemref,
    ValueRange odsStarts, ArrayRef<int64_t> odsTileSize) {
  // Massage types.
  ValueRange startsRange(odsStarts);
  ArrayAttr tileSizeAttr = odsBuilder.getI64ArrayAttr(odsTileSize);
  build(odsBuilder, odsState, odsBufferMemref, odsMemref, startsRange,
      tileSizeAttr);
}

static LogicalResult verify(KrnlCopyFromBufferOp op) {
  KrnlCopyFromBufferOpAdaptor opAdaptor = KrnlCopyFromBufferOpAdaptor(op);
  MemRefBoundsIndexCapture buffCapture(opAdaptor.buffer());
  int64_t bufferRank = buffCapture.getRank();
  int64_t destRank =
      opAdaptor.dest().getType().cast<MemRefType>().getShape().size();
  int64_t startRank = opAdaptor.starts().size();
  if (!buffCapture.areAllLiteral())
    return op.emitOpError("buffer expect constant dimensions");
  if (destRank < bufferRank)
    return op.emitOpError("Rank of memref cannot be smaller than buffer");
  if (startRank != destRank)
    return op.emitOpError("Rank of starts and memrefs must be identical");
  if (opAdaptor.tileSize()) {
    int64_t tRank = opAdaptor.tileSize().size();
    if (!(tRank == 0 || tRank == bufferRank))
      return op.emitOpError("Rank of tileSize must be identical to buffer");
  }
  return success();
}

} // namespace mlir

#define GET_OP_CLASSES
#include "src/Dialect/Krnl/KrnlOps.cpp.inc"
