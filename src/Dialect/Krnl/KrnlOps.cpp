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
      getNumLoopsAttrName(), builder.getI32IntegerAttr(num_loops));
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
    SymbolRefAttr funcAttr, IntegerAttr numInputs, IntegerAttr numOutputs) {
  state.addAttribute(KrnlEntryPointOp::getEntryPointFuncAttrName(), funcAttr);
  state.addAttribute(KrnlEntryPointOp::getNumInputsAttrName(), numInputs);
  state.addAttribute(KrnlEntryPointOp::getNumOutputsAttrName(), numOutputs);
}

//===----------------------------------------------------------------------===//
// KrnlIterateOp
//===----------------------------------------------------------------------===//

void KrnlDummyCastOp::build(
    OpBuilder &builder, OperationState &state, Value in, Type outType) {
  state.operands.emplace_back(in);
  state.types.emplace_back(outType);
}

//===----------------------------------------------------------------------===//
// KrnlVectorTypeCastOp
//===----------------------------------------------------------------------===//

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

MutableOperandRange KrnlMatMulOp::getLoopRefs() { return loopsMutable(); }

} // namespace mlir

#define GET_OP_CLASSES
#include "src/Dialect/Krnl/KrnlOps.cpp.inc"
