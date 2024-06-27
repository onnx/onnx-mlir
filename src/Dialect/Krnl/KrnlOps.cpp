/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------------- KrnlOps.cpp - Krnl Operations -----------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains implementations of krnl operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/Value.h"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// KrnlDialect
//===----------------------------------------------------------------------===//

void KrnlDialect::initialize() {
  addTypes<krnl::LoopType, krnl::StringType>();
  addOperations<
#define GET_OP_LIST
#include "src/Dialect/Krnl/KrnlOps.cpp.inc"
      >();
}

Type KrnlDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();
  MLIRContext *context = getContext();

  if (keyword == "loop")
    return krnl::LoopType::get(context);
  if (keyword == "string")
    return krnl::StringType::get(context);

  parser.emitError(parser.getNameLoc(), "unknown krnl type: " + keyword);
  return Type();
}

void KrnlDialect::printType(Type type, DialectAsmPrinter &os) const {
  TypeSwitch<Type>(type)
      .Case<krnl::LoopType>([&](Type) { os << "loop"; })
      .Case<krnl::StringType>([&](Type) { os << "string"; })
      .Default([](Type) { llvm_unreachable("Unexpected 'krnl' type kind"); });
}

namespace mlir {

//===----------------------------------------------------------------------===//
// KrnlCallOp
//===----------------------------------------------------------------------===//

static std::string typeToString(Type ty) {
  std::string str;
  llvm::raw_string_ostream out(str);
  ty.print(out);
  return out.str();
}

void KrnlCallOp::build(OpBuilder &builder, ::mlir::OperationState &odsState,
    std::string funcNameStr, ValueRange resultVals, Operation *op,
    ValueRange operands, bool copyAttrs) {
  // Creates parameters for KrnlCall for Optional input (with NoneType)
  // The semantics of optional input is ONNX Op specific and should be
  // handled when lowering ONNX Op, not lowering KrnlCall.
  // For now, None input is picked out from parameters of KrnCall.
  // The Op will decide which external function to call based on the input.
  // For future work: it might be possible to assume None type is
  // always for a tensor and implemented with a nullptr in llvm.
  // Then the None input can be handled inside the external function.
  // Currently, onnx-mlir::NoneType is not handled by typeConverter of
  // ONNXToKrnl conversion.
  SmallVector<Value, 4> allInputs(resultVals);
  for (auto operand : operands) {
    if (!isNoneValue(operand))
      allInputs.emplace_back(operand);
  }

  StringAttr funcNameAttr = builder.getStringAttr(funcNameStr);
  IntegerAttr numOfOutputAttr =
      IntegerAttr::get(builder.getIntegerType(64, /*isSigned=*/true),
          APInt(64, /*value=*/resultVals.size(), /*isSigned=*/true));
  if (!copyAttrs) {
    build(builder, odsState, funcNameAttr, numOfOutputAttr,
        ValueRange(allInputs));
  } else {
    std::vector<NamedAttribute> attributes;
    auto namedAttr1 = builder.getNamedAttr("funcName", funcNameAttr);
    auto namedAttr2 = builder.getNamedAttr("numOfOutput", numOfOutputAttr);
    attributes.emplace_back(namedAttr1);
    attributes.emplace_back(namedAttr2);
    for (auto namedAttr : op->getAttrs()) {
      attributes.emplace_back(namedAttr);
    }
    build(builder, odsState, TypeRange(), ValueRange(allInputs), attributes);
  }
}

void KrnlCallOp::build(OpBuilder &builder, ::mlir::OperationState &odsState,
    std::string funcNameStr, ValueRange resultVals, Operation *op,
    ValueRange operands, std::vector<std::string> attributeNames) {
  SmallVector<Value, 4> allInputs(resultVals);
  for (auto operand : operands) {
    if (!isNoneValue(operand))
      allInputs.emplace_back(operand);
  }

  std::vector<NamedAttribute> attributes;
  StringAttr funcNameAttr = builder.getStringAttr(funcNameStr);
  auto namedAttr1 = builder.getNamedAttr("funcName", funcNameAttr);
  attributes.emplace_back(namedAttr1);

  IntegerAttr numOfOutputAttr =
      IntegerAttr::get(builder.getIntegerType(64, /*isSigned=*/true),
          APInt(64, /*value=*/resultVals.size(), /*isSigned=*/true));
  auto namedAttr2 = builder.getNamedAttr("numOfOutput", numOfOutputAttr);
  attributes.emplace_back(namedAttr2);

  for (auto attributeName : attributeNames) {
    if (Attribute attr = op->getAttr(attributeName)) {
      attributes.emplace_back(builder.getNamedAttr(attributeName, attr));
    }
  }
  build(builder, odsState, TypeRange(), ValueRange(allInputs), attributes);
}

void KrnlCallOp::build(OpBuilder &builder, ::mlir::OperationState &odsState,
    ValueRange resultVals, Operation *op, ValueRange operands, bool copyAttrs) {
  // Create funcName
  std::string name = op->getName().getStringRef().str();
  std::replace(name.begin(), name.end(), '.', '_');
  ShapedType resultType = mlir::cast<ShapedType>(resultVals[0].getType());
  Type elementType = resultType.getElementType();
  std::string funcNameStr = name + "_" + typeToString(elementType);

  build(builder, odsState, funcNameStr, resultVals, op, operands, copyAttrs);
}

void KrnlCallOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {

  for (size_t i = 0; i < getParameters().size(); i++) {
    if (i < (size_t)getNumOfOutput())
      effects.emplace_back(MemoryEffects::Write::get(),
          &getParametersMutable()[i], SideEffects::DefaultResource::get());
    else
      effects.emplace_back(MemoryEffects::Read::get(),
          &getParametersMutable()[i], SideEffects::DefaultResource::get());
  }
}

//===----------------------------------------------------------------------===//
// KrnlDefineLoopsOp
//===----------------------------------------------------------------------===//

void KrnlDefineLoopsOp::build(
    OpBuilder &builder, OperationState &result, int64_t num_loops) {
  // Create the same number of dimension handlers as the number of
  // dimensions in the associated integer set.
  result.types.append(num_loops, krnl::LoopType::get(builder.getContext()));
  result.addAttribute(
      getNumLoopsAttrName(), builder.getI64IntegerAttr(num_loops));
}

void KrnlDefineLoopsOp::print(OpAsmPrinter &printer) {
  auto numLoopAttr = (*this)->getAttrOfType<IntegerAttr>(
      KrnlDefineLoopsOp::getNumLoopsAttrName());
  printer << ' ' << numLoopAttr.getValue().getSExtValue();
}

ParseResult KrnlDefineLoopsOp::parse(
    OpAsmParser &parser, OperationState &result) {
  // Parse the attribute indicating number of loops defined.
  IntegerAttr numLoops;
  auto &builder = parser.getBuilder();
  auto intType = builder.getIntegerType(64);
  if (parser.parseAttribute(numLoops, intType,
          KrnlDefineLoopsOp::getNumLoopsAttrName(), result.attributes))
    return failure();

  auto loopTypes =
      llvm::SmallVector<Type, 4>(numLoops.getValue().getSExtValue(),
          krnl::LoopType::get(builder.getContext()));
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
 * bound_types: a collection of integer values indicating how bounds are
 * given. 0 : bound is given as an integer in const_bounds. 1 : bound is given
 * as an operand in operand_bounds. 2 : bound is given as an affine map.
 * (TODO).
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
    krnl::KrnlIterateOperandPack operandPack, ValueRange iterArgInits,
    function_ref<void(OpBuilder &, Location, ValueRange, ValueRange)>
        bodyBuilderFn) {
  // Record optimized loops and the number of such loops.
  result.addOperands(operandPack.getOperands());

  // Add result based on iterArgInits.
  result.addOperands(iterArgInits);
  for (auto iterArgInit : iterArgInits)
    result.addTypes(iterArgInit.getType());

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
  auto body_arg_locs = llvm::SmallVector<Location, 4>(
      operandPack.getNumInputLoops(), result.location);
  body->addArguments(body_args, body_arg_locs);
  SmallVector<Value> iterArgs;
  // Add iterArgs after loop args.
  for (Value val : iterArgInits)
    iterArgs.emplace_back(body->addArgument(val.getType(), val.getLoc()));
  bodyRegion->push_back(body);

  // If nonnull, invoke the lambda function that creates the loop body. This
  // feature is used to build structured operations using lambda. Parameters
  // to the functions are the builder, location, and arguments passed as
  // iterArgs.
  if (bodyBuilderFn) {
    PatternRewriter::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(body);
    bodyBuilderFn(builder, result.location, iterArgInits, iterArgs);
    ensureTerminator(*bodyRegion, builder, result.location);
  } else {
    ensureTerminator(*bodyRegion, builder, result.location);
  }
}

void KrnlIterateOp::build(OpBuilder &builder, OperationState &result,
    ValueRange originalLoops, ValueRange optimizedLoops, ValueRange lbs,
    ValueRange ubs, ValueRange iterArgs,
    function_ref<void(OpBuilder &, Location, ValueRange, ValueRange)>
        bodyBuilderFn) {
  assert(lbs.size() == ubs.size() && "expected matching number of lb & ub");
  // TODO: May want to change KrnlIterateOperandPack to use ValueRanges...
  SmallVector<Value, 4> origLoops, optLoops;
  for (auto org : originalLoops)
    origLoops.emplace_back(org);
  for (auto opt : optimizedLoops)
    optLoops.emplace_back(opt);
  krnl::KrnlIterateOperandPack pack(builder, origLoops, optLoops);
  for (unsigned int i = 0; i < lbs.size(); ++i) {
    pack.pushOperandBound(lbs[i]);
    pack.pushOperandBound(ubs[i]);
  }
  // Fill in this iterate op using the main build function.
  build(builder, result, pack, iterArgs, bodyBuilderFn);
}

void KrnlIterateOp::build(OpBuilder &builder, OperationState &result,
    ValueRange originalLoops, ValueRange optimizedLoops,
    ArrayRef<IndexExpr> lbs, ArrayRef<IndexExpr> ubs, ValueRange iterArgs,
    function_ref<void(OpBuilder &, Location, ValueRange, ValueRange)>
        bodyBuilderFn) {
  assert(lbs.size() == ubs.size() && "expected matching number of lb & ub");
  SmallVector<Value, 4> origLoops, optLoops;
  for (auto org : originalLoops)
    origLoops.emplace_back(org);
  for (auto opt : optimizedLoops)
    optLoops.emplace_back(opt);
  krnl::KrnlIterateOperandPack pack(builder, origLoops, optLoops);
  for (unsigned int i = 0; i < lbs.size(); ++i) {
    pack.pushIndexExprBound(lbs[i], /*isLb*/ true);
    pack.pushIndexExprBound(ubs[i], /*isLb*/ false);
  }
  // Fill in this iterate op using the main build function.
  build(builder, result, pack, iterArgs, bodyBuilderFn);
}

MutableArrayRef<OpOperand> KrnlIterateOp::getInitsMutable() {
  size_t numControlOperands = getNumOperands() - getNumIterArgs();
  return getOperation()->getOpOperands().drop_front(numControlOperands);
}

std::optional<::llvm::MutableArrayRef<::mlir::OpOperand>>
KrnlIterateOp::getYieldedValuesMutable() {
  return cast<KrnlYieldOp>(getBody()->getTerminator()).getOperandsMutable();
}

void KrnlIterateOp::print(OpAsmPrinter &printer) {
  printer << "(";
  // Print optimized loops:
  auto numOptimizedLoops = getNumOptimizedLoops();
  printer.printOperands(operand_begin(), operand_begin() + numOptimizedLoops);
  printer << ") with (";

  // In the event where body region has been lowered, do not print body.
  if (getBodyRegion().empty()) {
    printer << ")";
    return;
  }
  auto entryBBArgs = getBodyRegion().begin()->getArguments();
  auto numEntryBBArgs = entryBBArgs.size();
  auto numIterArgs = getNumIterArgs();
  auto numInductionVars = numEntryBBArgs - numIterArgs;
  auto inductionVars = Block::BlockArgListType(
      entryBBArgs.begin(), entryBBArgs.begin() + numInductionVars);
  auto boundItr =
      (*this)
          ->getAttrOfType<ArrayAttr>(KrnlIterateOp::getBoundsAttrName())
          .getValue()
          .begin();
  auto operandItr = operand_begin() + numOptimizedLoops;

  std::string delimiter;
  for (auto &var : inductionVars) {
    printer << delimiter;
    printer.printOperand(*operandItr++);
    printer << " -> ";
    printer.printOperand(var);
    printer << " = ";
    krnl::printBound(
        mlir::cast<AffineMapAttr>(*boundItr++), operandItr, "max", printer);
    printer << " to ";
    krnl::printBound(
        mlir::cast<AffineMapAttr>(*boundItr++), operandItr, "min", printer);
    delimiter = ", ";
  }

  printer << ")";

  // iterArgs in format of iter_args(% arg = % init)
  if (getNumIterArgs()) {
    auto iterArgs = Block::BlockArgListType(
        entryBBArgs.begin() + numInductionVars, entryBBArgs.end());
    printer << " iter_args(";
    for (auto it : llvm::zip(iterArgs, getIterArgInits())) {
      printer.printOperand(std::get<0>(it));
      printer << " = ";
      printer.printOperand(std::get<1>(it));
    }
    printer << ") -> (";
    // -> (f32)
    for (auto it : getResults()) {
      it.getType().print(printer.getStream());
    }
    printer << ")";
  }
  printer.printRegion(getBodyRegion(), /*printEntryBlockArgs=*/false,
      /*printBlockTerminators=*/getNumIterArgs());
}

//===----------------------------------------------------------------------===//
// KrnlYieldOp
//===----------------------------------------------------------------------===//

LogicalResult KrnlYieldOp::verify() {
  auto *parentOp = (*this)->getParentOp();
  auto results = parentOp->getResults();
  auto operands = getOperands();

  if (!isa<KrnlIterateOp>(parentOp))
    return emitOpError() << "only terminates krnl.iterate regions";
  if (parentOp->getNumResults() != getNumOperands())
    return emitOpError() << "parent of yield must have same number of "
                            "results as the yield operands";
  for (auto it : llvm::zip(results, operands)) {
    if (std::get<0>(it).getType() != std::get<1>(it).getType())
      return emitOpError() << "types mismatch between yield op and its parent";
  }

  return success();
}

namespace {

struct LoopParser {
  // Parse input loops and their lower and upper bounds.
  SmallVector<OpAsmParser::UnresolvedOperand, 4> inductionVarRefs;
  SmallVector<Attribute, 4> boundMaps;
  OpAsmParser &parser;
  const Type loopType;

  LoopParser(OpAsmParser &parser, Type loopType)
      : parser(parser), loopType(loopType) {}

  // A method to parse a lower or upper bound.
  ParseResult parseBound(bool isUpper, OperationState &result) {
    Builder builder = parser.getBuilder();

    // 'min' / 'max' prefixes are generally syntactic sugar, but are required
    // if the map has multiple results.
    bool failedToParsedMinMax =
        failed(parser.parseOptionalKeyword(isUpper ? "min" : "max"));

    // Try parse an SSA operand.
    OpAsmParser::UnresolvedOperand ssa;
    OptionalParseResult ssaParseResult = parser.parseOptionalOperand(ssa);
    if (ssaParseResult.has_value()) {
      if (failed(ssaParseResult.value()) ||
          parser.resolveOperand(ssa, builder.getIndexType(), result.operands))
        return failure();

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

    if (auto affineMapAttr = mlir::dyn_cast<AffineMapAttr>(boundAttr)) {
      unsigned currentNumOperands = result.operands.size();
      unsigned numDims = 0;
      if (affine::parseDimAndSymbolList(parser, result.operands, numDims))
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
            "lower loop bound affine map with multiple "
            "results requires 'max' prefix");
      }
      boundMaps.emplace_back(AffineMapAttr::get(map));
      return success();
    }

    if (auto integerAttr = mlir::dyn_cast<IntegerAttr>(boundAttr)) {
      AffineMap map =
          builder.getConstantAffineMap(integerAttr.getValue().getSExtValue());
      boundMaps.emplace_back(AffineMapAttr::get(map));
    }
    return success();
  }

  ParseResult parse(OperationState &result) {
    // Parse an input loop operand;
    OpAsmParser::UnresolvedOperand loop;
    if (parser.parseOperand(loop) ||
        parser.resolveOperand(loop, loopType, result.operands) ||
        parser.parseArrow())
      return failure();

    // Parse induction variable.
    OpAsmParser::UnresolvedOperand inductionVar;
    if (parser.parseOperand(inductionVar, /*allowResultNumber=*/false) ||
        parser.parseEqual())
      return failure();
    inductionVarRefs.emplace_back(inductionVar);

    // Parse bound par (min to max).
    if (parseBound(/*isUpper=*/false, result) || parser.parseKeyword("to") ||
        parseBound(/*isUpper=*/true, result))
      return failure();

    return success();
  }

}; // struct LoopParser

} // namespace

ParseResult KrnlIterateOp::parse(OpAsmParser &parser, OperationState &result) {
  auto builder = parser.getBuilder();

  Type loopType = krnl::LoopType::get(result.getContext());

  // Parse optimized loops:
  SmallVector<OpAsmParser::UnresolvedOperand, 4> optimizedLoopRefs;
  if (parser.parseOperandList(
          optimizedLoopRefs, OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(optimizedLoopRefs, loopType, result.operands))
    return failure();

  // Record how many optimized loops did we parse.
  result.addAttribute(KrnlIterateOp::getNumOptimizedLoopsAttrName(),
      builder.getI64IntegerAttr(optimizedLoopRefs.size()));

  LoopParser loopParser(parser, loopType);

  if (parser.parseKeyword("with") || parser.parseLParen())
    return failure();

  if (failed(parser.parseOptionalRParen())) {
    if (parser.parseCommaSeparatedList(
            [&] { return loopParser.parse(result); }) ||
        parser.parseRParen())
      return failure();
  }

  result.addAttribute(KrnlIterateOp::getBoundsAttrName(),
      builder.getArrayAttr(loopParser.boundMaps));

  // Parse the optional initial iteration arguments.
  SmallVector<OpAsmParser::Argument, 4> regionArgs;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;

  Region *region = result.addRegion();

  SmallVector<OpAsmParser::Argument> entryArgs;
  for (OpAsmParser::UnresolvedOperand name : loopParser.inductionVarRefs) {
    OpAsmParser::Argument arg;
    arg.ssaName = name;
    arg.type = builder.getIndexType();
    entryArgs.push_back(arg);
  }

  if (succeeded(parser.parseOptionalKeyword("iter_args"))) {
    // Parse assignment list and results type list.
    if (parser.parseAssignmentList(regionArgs, operands) ||
        parser.parseArrowTypeList(result.types))
      return failure();
    // Resolve input operands.
    for (auto argOperandType : llvm::zip(regionArgs, operands, result.types)) {
      Type type = std::get<2>(argOperandType);
      std::get<0>(argOperandType).type = type;
      if (parser.resolveOperand(
              std::get<1>(argOperandType), type, result.operands))
        return failure();
      // Add to entryArgs.
      entryArgs.push_back(std::get<0>(argOperandType));
    }
  }

  if (parser.parseRegion(*region, entryArgs))
    return failure();

  // Ensure iterate region is closed off with krnl.terminate.
  KrnlIterateOp::ensureTerminator(
      *region, parser.getBuilder(), result.location);

  return success();
}

::llvm::SmallVector<mlir::Region *> KrnlIterateOp::getLoopRegions() {
  return {&getBodyRegion()};
}

LogicalResult KrnlIterateOp::verify() {
  // TODO: Verify number of induction variable bounds matches the number of
  // input loops.
  return success();
}

void KrnlRegionOp::build(OpBuilder &builder, OperationState &result,
    function_ref<void(OpBuilder &, Location)> bodyBuilderFn) {

  Region *bodyRegion = result.addRegion();
  auto *body = new Block();
  llvm::SmallVector<Type, 4> body_args;
  llvm::SmallVector<Location, 4> body_arg_locs;
  body->addArguments(body_args, body_arg_locs);
  bodyRegion->push_back(body);

  if (bodyBuilderFn) {
    PatternRewriter::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(body);
    bodyBuilderFn(builder, result.location);
  }
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

void KrnlInstrumentOp::build(mlir::OpBuilder &builder, OperationState &state,
    Operation *op, int tag = 0) {
  const char *opName = op->getName().getStringRef().data();
  StringAttr opNameAttr = builder.getStringAttr(StringRef(opName));
  IntegerAttr tagAttr = builder.getI64IntegerAttr(tag);
  StringAttr nodeNameAttr =
      op->getAttrOfType<::mlir::StringAttr>("onnx_node_name");
  build(builder, state, opNameAttr, tagAttr, nodeNameAttr);
}

//===----------------------------------------------------------------------===//
// KrnlBlockOp
//===----------------------------------------------------------------------===//

void KrnlBlockOp::build(::mlir::OpBuilder &odsBuilder,
    ::mlir::OperationState &odsState, Value odsLoop, int64_t odsTileSize) {
  SmallVector<Type, 4> blockResType(
      2, krnl::LoopType::get(odsBuilder.getContext()));
  build(odsBuilder, odsState, blockResType, odsLoop,
      odsBuilder.getI64IntegerAttr(odsTileSize));
}

//===----------------------------------------------------------------------===//
// KrnlPermuteOp
//===----------------------------------------------------------------------===//

void KrnlPermuteOp::build(::mlir::OpBuilder &odsBuilder,
    ::mlir::OperationState &odsState, ValueRange odsLoops,
    ArrayRef<int64_t> odsMap) {
  uint64_t rank = odsLoops.size();
  assert(rank >= 2 && "permute needs 2 or more loops");
  assert(odsMap.size() == rank && "loop and size size must be identical");
  for (unsigned int i = 0; i < rank; ++i) {
    assert(odsMap[i] >= 0 && odsMap[i] < (int64_t)rank && "bad permute");
    for (unsigned int j = i + 1; j < rank; ++j)
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
  SmallVector<Type, 6> types(rank, odsBuilder.getIndexType());
  ArrayRef<NamedAttribute> noAttr({});
  build(odsBuilder, odsState, types, odsLoops, noAttr);
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
  MemRefType sourceType = mlir::cast<MemRefType>(sourceMemRef.getType());
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

bool KrnlVectorTypeCastOp::areCastCompatible(
    TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  Type a = inputs.front(), b = outputs.front();

  auto aT = mlir::dyn_cast<MemRefType>(a);
  auto bT = mlir::dyn_cast<MemRefType>(b);

  if (!aT || !bT)
    return false;

  if (aT.getLayout() != bT.getLayout())
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
  if (auto shapedEltType = mlir::dyn_cast<ShapedType>(aT.getElementType()))
    return false;

  auto shapedEltTypeB = mlir::dyn_cast<ShapedType>(bT.getElementType());
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

  if (!ShapedType::isDynamic(lastDimA) && !ShapedType::isDynamic(lastDimB) &&
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
    auto cast = operand.get().getDefiningOp<memref::CastOp>();
    if (cast && !mlir::isa<UnrankedMemRefType>(cast.getOperand().getType())) {
      operand.set(cast.getOperand());
      folded = true;
    }
  }
  return success(folded);
}

OpFoldResult KrnlVectorTypeCastOp::fold(FoldAdaptor adaptor) {
  if (OpFoldResult folded = OpFoldResult())
    return folded;
  return succeeded(foldMemRefCast(*this)) ? getResult() : Value();
}

MutableOperandRange KrnlSpecializedKernel::getLoopRefs() {
  return getLoopsMutable();
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
    bool odsOverCompute) {
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
      odsOverCompute);
}

void KrnlMatMulOp::build(::mlir::OpBuilder &odsBuilder,
    ::mlir::OperationState &odsState, Value odsA, ValueRange aOdsStart,
    Value odsB, ValueRange bOdsStart, Value odsC, ValueRange cOdsStart,
    ValueRange odsLoops, Value iOdsComputeStart, Value jOdsComputeStart,
    Value kOdsComputeStart, Value iOdsGlobalUB, Value jOdsGlobalUB,
    Value kOdsGlobalUB, bool odsSimdize, bool odsUnroll, bool odsOverCompute) {
  // Massage types.
  ValueRange loopRange(odsLoops);
  ArrayRef<int64_t> empty;

  build(odsBuilder, odsState, odsA, aOdsStart, odsB, bOdsStart, odsC, cOdsStart,
      loopRange, iOdsComputeStart, jOdsComputeStart, kOdsComputeStart,
      iOdsGlobalUB, jOdsGlobalUB, kOdsGlobalUB, empty, empty, empty, empty,
      odsSimdize, odsUnroll, odsOverCompute);
}

LogicalResult KrnlMatMulOp::verify() {
  KrnlMatMulOpAdaptor operandAdaptor = KrnlMatMulOpAdaptor(*this);
  uint64_t aRank =
      mlir::cast<MemRefType>(operandAdaptor.getA().getType()).getShape().size();
  uint64_t bRank =
      mlir::cast<MemRefType>(operandAdaptor.getB().getType()).getShape().size();
  uint64_t cRank =
      mlir::cast<MemRefType>(operandAdaptor.getC().getType()).getShape().size();
  if (!(aRank >= 2 && bRank >= 2 && cRank >= 2))
    return emitOpError("currently only support ranks >=2");
  if (operandAdaptor.getAGlobalIndexMemStart().size() != aRank)
    return emitOpError(
        "aGlobalIndexMemStart should have same rank as memref A");
  if (operandAdaptor.getBGlobalIndexMemStart().size() != bRank)
    return emitOpError(
        "bGlobalIndexMemStart should have same rank as memref A");
  if (operandAdaptor.getCGlobalIndexMemStart().size() != cRank)
    return emitOpError(
        "cGlobalIndexMemStart should have same rank as memref A");
  if (operandAdaptor.getLoops().size() != 3)
    return emitOpError("loops rank should be 3 (i,j,k)");

  if (operandAdaptor.getComputeTileSize().has_value()) {
    ArrayAttr computeAttr = operandAdaptor.getComputeTileSize().value();
    if (!(computeAttr.size() == 0 || computeAttr.size() == 3))
      return emitOpError("computeTileSize rank should be 0 or 3");
  }
  if (operandAdaptor.getATileSize().has_value()) {
    ArrayAttr aTileAttr = operandAdaptor.getATileSize().value();
    if (!(aTileAttr.size() == 0 || aTileAttr.size() == 2))
      return emitOpError("aTileSize rank should be 0 or 2");
  }
  if (operandAdaptor.getBTileSize().has_value()) {
    ArrayAttr bTileAttr = operandAdaptor.getBTileSize().value();
    if (!(bTileAttr.size() == 0 || bTileAttr.size() == 2))
      return emitOpError("bTileSize rank should be 0 or 2");
  }
  if (operandAdaptor.getCTileSize().has_value()) {
    ArrayAttr cTileAttr = operandAdaptor.getCTileSize().value();
    if (!(cTileAttr.size() == 0 || cTileAttr.size() == 2))
      return emitOpError("cTileSize rank should be 0 or 2");
  }
  return success();
}

MutableOperandRange KrnlMatMulOp::getLoopRefs() { return getLoopsMutable(); }

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

void KrnlCopyToBufferOp::build(::mlir::OpBuilder &odsBuilder,
    ::mlir::OperationState &odsState, Value odsBufferMemref, Value odsMemref,
    ValueRange odsStarts, Value odsPadValue, bool odsTranspose) {
  // Massage types.
  ValueRange startsRange(odsStarts);
  ArrayRef<int64_t> empty;
  build(odsBuilder, odsState, odsBufferMemref, odsMemref, startsRange,
      odsPadValue, empty, empty, odsTranspose);
}

LogicalResult KrnlCopyToBufferOp::verify() {
  KrnlCopyToBufferOpAdaptor opAdaptor = KrnlCopyToBufferOpAdaptor(*this);
  IndexExprBuilderForAnalysis createIE(getLoc());
  SmallVector<IndexExpr, 4> buff, source;
  int64_t bufferRank = createIE.getShapedTypeRank(opAdaptor.getBuffer());
  int64_t srcRank = createIE.getShapedTypeRank(opAdaptor.getSource());
  int64_t startRank = opAdaptor.getStarts().size();
  if (!createIE.isLiteralShape(opAdaptor.getBuffer()))
    return emitOpError("buffer expect constant dimensions");
  if (srcRank < bufferRank)
    return emitOpError("Rank of memref cannot be smaller than buffer");
  if (startRank != srcRank)
    return emitOpError("Rank of starts and memrefs must be identical");
  if (opAdaptor.getTileSize()) {
    int64_t tRank = opAdaptor.getTileSize().value().size();
    if (!(tRank == 0 || tRank == bufferRank))
      return emitOpError("Rank of tileSize must be identical to buffer");
  }
  if (opAdaptor.getPadToNext()) {
    int64_t padRank = opAdaptor.getPadToNext().value().size();
    if (!(padRank == 0 || padRank == bufferRank))
      return emitOpError("Rank of padToNext must be identical to buffer");
  }
  if (opAdaptor.getTranspose()) {
    if (bufferRank < 2)
      return emitOpError(
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

void KrnlCopyFromBufferOp::build(::mlir::OpBuilder &odsBuilder,
    ::mlir::OperationState &odsState, Value odsBufferMemref, Value odsMemref,
    ValueRange odsStarts) {
  // Massage types.
  ValueRange startsRange(odsStarts);
  ArrayRef<int64_t> empty;
  build(odsBuilder, odsState, odsBufferMemref, odsMemref, startsRange, empty);
}

LogicalResult KrnlCopyFromBufferOp::verify() {
  KrnlCopyFromBufferOpAdaptor opAdaptor = KrnlCopyFromBufferOpAdaptor(*this);
  IndexExprBuilderForAnalysis createIE(getLoc());
  int64_t bufferRank = createIE.getShapedTypeRank(opAdaptor.getBuffer());
  int64_t destRank =
      mlir::cast<MemRefType>(opAdaptor.getDest().getType()).getShape().size();
  int64_t startRank = opAdaptor.getStarts().size();
  if (!createIE.isLiteralShape(opAdaptor.getBuffer()))
    return emitOpError("buffer expect constant dimensions");
  if (destRank < bufferRank)
    return emitOpError("Rank of memref cannot be smaller than buffer");
  if (startRank != destRank)
    return emitOpError("Rank of starts and memrefs must be identical");
  if (opAdaptor.getTileSize()) {
    int64_t tRank = opAdaptor.getTileSize().value().size();
    if (!(tRank == 0 || tRank == bufferRank))
      return emitOpError("Rank of tileSize must be identical to buffer");
  }
  return success();
}

void KrnlSeqExtractOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSeqMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getIndexMutable(),
      SideEffects::DefaultResource::get());
  OpResult output = getOperation()->getOpResults()[0];
  effects.emplace_back(
      MemoryEffects::Write::get(), output, SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Allocate::get(), output,
      SideEffects::DefaultResource::get());
}

void KrnlSeqStoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getSeqMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getInputMutable(),
      SideEffects::DefaultResource::get());
}

std::optional<Operation *> KrnlSeqExtractOp::buildDealloc(
    OpBuilder &builder, Value alloc) {
  Location loc = alloc.getLoc();
  MultiDialectBuilder<MemRefBuilder> create(builder, loc);
  return create.mem.dealloc(alloc).getOperation();
}

std::optional<Value> KrnlSeqExtractOp::buildClone(
    OpBuilder &builder, Value alloc) {
  return builder.create<bufferization::CloneOp>(alloc.getLoc(), alloc)
      .getResult();
}

void KrnlSeqAllocOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  for (auto inp = getLengthMutable().begin(); inp != getLengthMutable().end();
       ++inp)
    effects.emplace_back(
        MemoryEffects::Read::get(), inp, SideEffects::DefaultResource::get());
  OpResult output = getOperation()->getOpResults()[0];
  effects.emplace_back(
      MemoryEffects::Write::get(), output, SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Allocate::get(), output,
      SideEffects::DefaultResource::get());
}

std::optional<Operation *> KrnlSeqAllocOp::buildDealloc(
    OpBuilder &builder, Value alloc) {
  Location loc = alloc.getLoc();
  // MultiDialectBuilder<KrnlBuilder> create(builder, loc);
  return builder.create<KrnlSeqDeallocOp>(loc, alloc).getOperation();
}

std::optional<Value> KrnlSeqAllocOp::buildClone(
    OpBuilder &builder, Value alloc) {
  return builder.create<bufferization::CloneOp>(alloc.getLoc(), alloc)
      .getResult();
}

//===----------------------------------------------------------------------===//
// KrnlGetLinearOffsetIndexOp
//===----------------------------------------------------------------------===//

void KrnlGetLinearOffsetIndexOp::build(OpBuilder &builder,
    OperationState &result, AffineMap map, ValueRange operands) {
  assert(operands.size() == 1 + map.getNumInputs() && "inconsistent operands");
  result.addOperands(operands);
  if (map)
    result.addAttribute(getMapAttrStrName(), AffineMapAttr::get(map));
  auto memrefType = llvm::cast<MemRefType>(operands[0].getType());
  result.types.push_back(memrefType.getElementType());
}

void KrnlGetLinearOffsetIndexOp::build(OpBuilder &builder,
    OperationState &result, Value memref, AffineMap map,
    ValueRange mapOperands) {
  assert(map.getNumInputs() == mapOperands.size() && "inconsistent index info");
  result.addOperands(memref);
  result.addOperands(mapOperands);
  result.addAttribute(getMapAttrStrName(), AffineMapAttr::get(map));
  result.types.push_back(builder.getIndexType());
}

void KrnlGetLinearOffsetIndexOp::build(OpBuilder &builder,
    OperationState &result, Value memref, ValueRange indices) {
  auto memrefType = llvm::cast<MemRefType>(memref.getType());
  int64_t rank = memrefType.getRank();
  // Create identity map for memrefs with at least one dimension or () -> ()
  // for zero-dimensional memrefs.
  auto map =
      rank ? builder.getMultiDimIdentityMap(rank) : builder.getEmptyAffineMap();
  build(builder, result, memref, map, indices);
}

ParseResult KrnlGetLinearOffsetIndexOp::parse(
    OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  auto indexTy = builder.getIndexType();

  MemRefType type;
  OpAsmParser::UnresolvedOperand memrefInfo;
  AffineMapAttr mapAttr;
  SmallVector<OpAsmParser::UnresolvedOperand, 1> mapOperands;
  return failure(
      parser.parseOperand(memrefInfo) || parser.parseKeyword("at") ||
      parser.parseAffineMapOfSSAIds(mapOperands, mapAttr,
          KrnlGetLinearOffsetIndexOp::getMapAttrStrName(), result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(memrefInfo, type, result.operands) ||
      parser.resolveOperands(mapOperands, indexTy, result.operands) ||
      parser.addTypeToList(indexTy, result.types));
}

void KrnlGetLinearOffsetIndexOp::print(OpAsmPrinter &p) {
  p << " " << getMemRef() << " at [";
  if (AffineMapAttr mapAttr =
          (*this)->getAttrOfType<AffineMapAttr>(getMapAttrStrName()))
    p.printAffineMapOfSSAIds(mapAttr, getMapOperands());
  p << ']';
  p.printOptionalAttrDict((*this)->getAttrs(),
      /*elidedAttrs=*/{getMapAttrStrName()});
  p << " : " << getMemRefType();
}

//===----------------------------------------------------------------------===//
// KrnlPrefetchOp
//===----------------------------------------------------------------------===//

void KrnlPrefetchOp::build(OpBuilder &builder, OperationState &result,
    Value memref, AffineMap map, ValueRange mapOperands, bool isWrite,
    unsigned localityHint, bool isDataCache) {
  assert(map.getNumInputs() == mapOperands.size() && "inconsistent index info");
  result.addOperands(memref);
  result.addOperands(mapOperands);
  result.addAttribute(getMapAttrStrName(), AffineMapAttr::get(map));
  result.addAttribute(getIsWriteAttrStrName(), builder.getBoolAttr(isWrite));
  result.addAttribute(
      getLocalityHintAttrStrName(), builder.getI32IntegerAttr(localityHint));
  result.addAttribute(
      getIsDataCacheAttrStrName(), builder.getBoolAttr(isDataCache));
}

void KrnlPrefetchOp::build(OpBuilder &builder, OperationState &result,
    Value memref, ValueRange indices, bool isWrite, unsigned localityHint,
    bool isDataCache) {
  auto memrefType = llvm::cast<MemRefType>(memref.getType());
  int64_t rank = memrefType.getRank();
  // Create identity map for memrefs with at least one dimension or () -> ()
  // for zero-dimensional memrefs.
  auto map =
      rank ? builder.getMultiDimIdentityMap(rank) : builder.getEmptyAffineMap();
  build(builder, result, memref, map, indices, isWrite, localityHint,
      isDataCache);
}

void KrnlPrefetchOp::build(OpBuilder &builder, OperationState &result,
    Value memref, bool isWrite, unsigned localityHint, bool isDataCache) {
  build(builder, result, memref, {}, isWrite, localityHint, isDataCache);
}

//
// krnl.prefetch %0[%i, %j + 5], read, locality<3>, data : memref<400x400xi32>
// Code lifted from affine prefetch as is.
// I have seen parsing errors when multiple '#x' are used in the indices,
// could not tell why.
//   krnl.prefetch %arg0[%1#0, %1#1, %3], read, locality<3>, data :
//     memref<8x256x512xf32>
// With only one, it works.
//

ParseResult KrnlPrefetchOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  auto indexTy = builder.getIndexType();

  MemRefType type;
  OpAsmParser::UnresolvedOperand memrefInfo;
  IntegerAttr hintInfo;
  auto i32Type = parser.getBuilder().getIntegerType(32);
  StringRef readOrWrite, cacheType;

  AffineMapAttr mapAttr;
  SmallVector<OpAsmParser::UnresolvedOperand, 1> mapOperands;
  if (parser.parseOperand(memrefInfo) ||
      parser.parseAffineMapOfSSAIds(mapOperands, mapAttr,
          KrnlPrefetchOp::getMapAttrStrName(), result.attributes) ||
      parser.parseComma() || parser.parseKeyword(&readOrWrite) ||
      parser.parseComma() || parser.parseKeyword("locality") ||
      parser.parseLess() ||
      parser.parseAttribute(hintInfo, i32Type,
          KrnlPrefetchOp::getLocalityHintAttrStrName(), result.attributes) ||
      parser.parseGreater() || parser.parseComma() ||
      parser.parseKeyword(&cacheType) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(memrefInfo, type, result.operands) ||
      parser.resolveOperands(mapOperands, indexTy, result.operands))
    return failure();

  if (!(readOrWrite == "read") && !(readOrWrite == "write"))
    return parser.emitError(
        parser.getNameLoc(), "rw specifier has to be 'read' or 'write'");
  result.addAttribute(KrnlPrefetchOp::getIsWriteAttrStrName(),
      parser.getBuilder().getBoolAttr(readOrWrite == "write"));

  if (!(cacheType == "data") && !(cacheType == "instr"))
    return parser.emitError(
        parser.getNameLoc(), "cache type has to be 'data' or 'instr'");

  result.addAttribute(KrnlPrefetchOp::getIsDataCacheAttrStrName(),
      parser.getBuilder().getBoolAttr(cacheType == "data"));

  return success();
}

void KrnlPrefetchOp::print(OpAsmPrinter &p) {
  p << " " << getMemref() << '[';
  AffineMapAttr mapAttr =
      (*this)->getAttrOfType<AffineMapAttr>(getMapAttrStrName());
  if (mapAttr)
    p.printAffineMapOfSSAIds(mapAttr, getMapOperands());
  p << ']' << ", " << (getIsWrite() ? "write" : "read") << ", "
    << "locality<" << getLocalityHint() << ">, "
    << (getIsDataCache() ? "data" : "instr");
  p.printOptionalAttrDict((*this)->getAttrs(),
      /*elidedAttrs=*/{getMapAttrStrName(), getLocalityHintAttrStrName(),
          getIsDataCacheAttrStrName(), getIsWriteAttrStrName()});
  p << " : " << getMemRefType();
}

//===----------------------------------------------------------------------===//
// KrnlMemcpyOp
//===----------------------------------------------------------------------===//

void KrnlMemcpyOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestMutable(),
      SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// KrnlMemsetOp
//===----------------------------------------------------------------------===//

void KrnlMemsetOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDestMutable(),
      SideEffects::DefaultResource::get());
}

} // namespace mlir

#define GET_OP_CLASSES
#include "src/Dialect/Krnl/KrnlOps.cpp.inc"

#include "src/Dialect/Krnl/KrnlDialect.cpp.inc"
