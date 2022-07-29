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

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"

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
    std::string funcNameStr, Value resultVal, Operation *op,
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
  SmallVector<Value, 4> allInputs;
  allInputs.emplace_back(resultVal);
  for (auto operand : operands) {
    if (!isFromNone(operand))
      allInputs.emplace_back(operand);
  }

  StringAttr funcNameAttr = builder.getStringAttr(funcNameStr);
  auto namedAttr = builder.getNamedAttr("funcName", funcNameAttr);
  if (!copyAttrs) {
    build(builder, odsState, funcNameAttr, resultVal, allInputs);
  } else {
    std::vector<NamedAttribute> attributes;
    attributes.emplace_back(namedAttr);
    for (auto namedAttr : op->getAttrs()) {
      attributes.emplace_back(namedAttr);
    }
    build(builder, odsState, TypeRange(), ValueRange(allInputs), attributes);
  }
}

void KrnlCallOp::build(OpBuilder &builder, ::mlir::OperationState &odsState,
    Value resultVal, Operation *op, ValueRange operands, bool copyAttrs) {
  // Create funcName
  std::string name = op->getName().getStringRef().str();
  std::replace(name.begin(), name.end(), '.', '_');
  ShapedType resultType = resultVal.getType().cast<ShapedType>();
  Type elementType = resultType.getElementType();
  std::string funcNameStr = name + "_" + typeToString(elementType);

  build(builder, odsState, funcNameStr, resultVal, op, operands, copyAttrs);
}

void KrnlCallOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  for (auto parameter : parameters()) {
    effects.emplace_back(MemoryEffects::Read::get(), parameter,
        SideEffects::DefaultResource::get());
  }
  effects.emplace_back(MemoryEffects::Write::get(), result(),
      SideEffects::DefaultResource::get());
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
    krnl::KrnlIterateOperandPack operandPack, ValueRange iterArgs,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilderFn) {
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
  auto body_arg_locs = llvm::SmallVector<Location, 4>(
      operandPack.getNumInputLoops(), result.location);
  body->addArguments(body_args, body_arg_locs);
  bodyRegion->push_back(body);

  // If nonnull, invoke the lambda function that creates the loop body. This
  // feature is used to build structured operations using lambda. Parameters
  // to the functions are the builder, location, and arguments passed as
  // iterArgs.
  if (bodyBuilderFn) {
    PatternRewriter::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(body);
    bodyBuilderFn(builder, result.location, iterArgs);
    ensureTerminator(*bodyRegion, builder, result.location);
  } else {
    ensureTerminator(*bodyRegion, builder, result.location);
  }
}

void KrnlIterateOp::build(OpBuilder &builder, OperationState &result,
    ValueRange originalLoops, ValueRange optimizedLoops, ValueRange lbs,
    ValueRange ubs, ValueRange iterArgs,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilderFn) {
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
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilderFn) {
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

void KrnlIterateOp::print(OpAsmPrinter &printer) {
  printer << "(";
  // Print optimized loops:
  auto numOptimizedLoops = getNumOptimizedLoops();
  printer.printOperands(operand_begin(), operand_begin() + numOptimizedLoops);
  printer << ") with (";

  // In the event where body region has been lowered, do not print body.
  if (bodyRegion().empty()) {
    printer << ")";
    return;
  }
  auto inductionVars = bodyRegion().begin()->getArguments();
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
        (*boundItr++).cast<AffineMapAttr>(), operandItr, "max", printer);
    printer << " to ";
    krnl::printBound(
        (*boundItr++).cast<AffineMapAttr>(), operandItr, "min", printer);
    delimiter = ", ";
  }

  printer << ")";
  printer.printRegion(bodyRegion(), /*printEntryBlockArgs=*/false,
      /*printBlockTerminators=*/false);
}

ParseResult KrnlIterateOp::parse(OpAsmParser &parser, OperationState &result) {
  auto builder = parser.getBuilder();
  auto context = builder.getContext();
  onnx_mlir::krnl::KrnlDialectOperandParser operandParser(parser);

  // Parse optimized loops:
  SmallVector<OpAsmParser::UnresolvedOperand, 4> optimizedLoopRefs;
  if (parser.parseOperandList(
          optimizedLoopRefs, OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(optimizedLoopRefs,
          krnl::LoopType::get(result.getContext()), result.operands))
    return failure();

  // Record how many optimized loops did we parse.
  result.addAttribute(KrnlIterateOp::getNumOptimizedLoopsAttrName(),
      builder.getI64IntegerAttr(optimizedLoopRefs.size()));

  // Parse input loops and their lower and upper bounds.
  SmallVector<OpAsmParser::UnresolvedOperand, 4> inductionVarRefs;
  SmallVector<Attribute, 4> boundMaps;

  if (parser.parseKeyword("with") || parser.parseLParen())
    return failure();

  // A function to parse a lower or upper bound.
  auto parseBound = [&result, &builder, &parser, &operandParser, &boundMaps](
                        bool isUpper) -> ParseResult {
    // 'min' / 'max' prefixes are generally syntactic sugar, but are required
    // if the map has multiple results.
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
    operandParser.ParseOperand(krnl::LoopType::get(context), result.operands);
    parser.parseArrow();

    // Parse induction variable.
    OpAsmParser::UnresolvedOperand inductionVar;
    if (parser.parseOperand(inductionVar, /*allowResultNumber=*/false) ||
        parser.parseEqual())
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

  SmallVector<OpAsmParser::Argument> entryArgs;
  for (auto it : llvm::zip(inductionVarRefs, inductionVarTypes)) {
    OpAsmParser::Argument arg;
    arg.ssaName = std::get<0>(it);
    arg.type = std::get<1>(it);
    entryArgs.push_back(arg);
  }

  if (parser.parseRegion(*region, entryArgs))
    return failure();

  // Ensure iterate region is closed off with krnl.terminate.
  KrnlIterateOp::ensureTerminator(
      *region, parser.getBuilder(), result.location);

  return success();
}

Region &KrnlIterateOp::getLoopBody() { return bodyRegion(); }

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
  int64_t opID = 0;
  // getName() result is "onnx.opName"
  // Put only the opName part in the opID within its size
  strncpy((char *)&opID, opName + 5, sizeof(decltype(opID)) - 1);
  IntegerAttr attr = builder.getI64IntegerAttr(opID);
  auto tagAttr = builder.getI64IntegerAttr(tag);
  StringAttr nameAttr = builder.getStringAttr(StringRef(opName));
  state.addAttribute("opName", nameAttr);
  state.addAttribute("opID", attr);
  state.addAttribute("tag", tagAttr);
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

bool KrnlVectorTypeCastOp::areCastCompatible(
    TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  Type a = inputs.front(), b = outputs.front();

  auto aT = a.dyn_cast<MemRefType>();
  auto bT = b.dyn_cast<MemRefType>();

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
    if (cast && !cast.getOperand().getType().isa<UnrankedMemRefType>()) {
      operand.set(cast.getOperand());
      folded = true;
    }
  }
  return success(folded);
}

OpFoldResult KrnlVectorTypeCastOp::fold(ArrayRef<Attribute> operands) {
  if (OpFoldResult folded = OpFoldResult())
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

void KrnlMatMulOp::build(::mlir::OpBuilder &odsBuilder,
    ::mlir::OperationState &odsState, Value odsA, ValueRange aOdsStart,
    Value odsB, ValueRange bOdsStart, Value odsC, ValueRange cOdsStart,
    ValueRange odsLoops, Value iOdsComputeStart, Value jOdsComputeStart,
    Value kOdsComputeStart, Value iOdsGlobalUB, Value jOdsGlobalUB,
    Value kOdsGlobalUB, bool odsSimdize, bool odsUnroll, bool odsOvercompute) {
  // Massage types.
  ValueRange loopRange(odsLoops);
  ArrayRef<int64_t> empty;

  build(odsBuilder, odsState, odsA, aOdsStart, odsB, bOdsStart, odsC, cOdsStart,
      loopRange, iOdsComputeStart, jOdsComputeStart, kOdsComputeStart,
      iOdsGlobalUB, jOdsGlobalUB, kOdsGlobalUB, empty, empty, empty, empty,
      odsSimdize, odsUnroll, odsOvercompute);
}

LogicalResult KrnlMatMulOp::verify() {
  KrnlMatMulOpAdaptor operandAdaptor = KrnlMatMulOpAdaptor(*this);
  uint64_t aRank =
      operandAdaptor.A().getType().cast<MemRefType>().getShape().size();
  uint64_t bRank =
      operandAdaptor.B().getType().cast<MemRefType>().getShape().size();
  uint64_t cRank =
      operandAdaptor.C().getType().cast<MemRefType>().getShape().size();
  if (!(aRank >= 2 && bRank >= 2 && cRank >= 2))
    return emitOpError("currently only support ranks >=2");
  if (operandAdaptor.aGlobalIndexMemStart().size() != aRank)
    return emitOpError(
        "aGlobalIndexMemStart should have same rank as memref A");
  if (operandAdaptor.bGlobalIndexMemStart().size() != bRank)
    return emitOpError(
        "bGlobalIndexMemStart should have same rank as memref A");
  if (operandAdaptor.cGlobalIndexMemStart().size() != cRank)
    return emitOpError(
        "cGlobalIndexMemStart should have same rank as memref A");
  if (operandAdaptor.loops().size() != 3)
    return emitOpError("loops rank should be 3 (i,j,k)");

  if (operandAdaptor.computeTileSize().hasValue()) {
    ArrayAttr computeAttr = operandAdaptor.computeTileSize().getValue();
    if (!(computeAttr.size() == 0 || computeAttr.size() == 3))
      return emitOpError("computeTileSize rank should be 0 or 3");
  }
  if (operandAdaptor.aTileSize().hasValue()) {
    ArrayAttr aTileAttr = operandAdaptor.aTileSize().getValue();
    if (!(aTileAttr.size() == 0 || aTileAttr.size() == 2))
      return emitOpError("aTileSize rank should be 0 or 2");
  }
  if (operandAdaptor.bTileSize().hasValue()) {
    ArrayAttr bTileAttr = operandAdaptor.bTileSize().getValue();
    if (!(bTileAttr.size() == 0 || bTileAttr.size() == 2))
      return emitOpError("bTileSize rank should be 0 or 2");
  }
  if (operandAdaptor.cTileSize().hasValue()) {
    ArrayAttr cTileAttr = operandAdaptor.cTileSize().getValue();
    if (!(cTileAttr.size() == 0 || cTileAttr.size() == 2))
      return emitOpError("cTileSize rank should be 0 or 2");
  }
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
  MemRefBoundsIndexCapture buffCapture(opAdaptor.buffer());
  MemRefBoundsIndexCapture srcCapture(opAdaptor.source());
  int64_t bufferRank = buffCapture.getRank();
  int64_t srcRank = srcCapture.getRank();
  int64_t startRank = opAdaptor.starts().size();
  if (!buffCapture.areAllLiteral())
    return emitOpError("buffer expect constant dimensions");
  if (srcRank < bufferRank)
    return emitOpError("Rank of memref cannot be smaller than buffer");
  if (startRank != srcRank)
    return emitOpError("Rank of starts and memrefs must be identical");
  if (opAdaptor.tileSize()) {
    int64_t tRank = opAdaptor.tileSize().getValue().size();
    if (!(tRank == 0 || tRank == bufferRank))
      return emitOpError("Rank of tileSize must be identical to buffer");
  }
  if (opAdaptor.padToNext()) {
    int64_t padRank = opAdaptor.padToNext().getValue().size();
    if (!(padRank == 0 || padRank == bufferRank))
      return emitOpError("Rank of padToNext must be identical to buffer");
  }
  if (opAdaptor.transpose()) {
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
  MemRefBoundsIndexCapture buffCapture(opAdaptor.buffer());
  int64_t bufferRank = buffCapture.getRank();
  int64_t destRank =
      opAdaptor.dest().getType().cast<MemRefType>().getShape().size();
  int64_t startRank = opAdaptor.starts().size();
  if (!buffCapture.areAllLiteral())
    return emitOpError("buffer expect constant dimensions");
  if (destRank < bufferRank)
    return emitOpError("Rank of memref cannot be smaller than buffer");
  if (startRank != destRank)
    return emitOpError("Rank of starts and memrefs must be identical");
  if (opAdaptor.tileSize()) {
    int64_t tRank = opAdaptor.tileSize().getValue().size();
    if (!(tRank == 0 || tRank == bufferRank))
      return emitOpError("Rank of tileSize must be identical to buffer");
  }
  return success();
}

void KrnlSeqExtractOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(
      MemoryEffects::Read::get(), seq(), SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), output(),
      SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Allocate::get(), output(),
      SideEffects::DefaultResource::get());
}

Optional<Operation *> KrnlSeqExtractOp::buildDealloc(
    OpBuilder &builder, Value alloc) {
  auto loc = alloc.getLoc();
  MultiDialectBuilder<MemRefBuilder> create(builder, loc);
  return create.mem.dealloc(alloc).getOperation();
}

Optional<Value> KrnlSeqExtractOp::buildClone(OpBuilder &builder, Value alloc) {
  return builder.create<bufferization::CloneOp>(alloc.getLoc(), alloc)
      .getResult();
}

} // namespace mlir

#define GET_OP_CLASSES
#include "src/Dialect/Krnl/KrnlOps.cpp.inc"

#include "src/Dialect/Krnl/KrnlDialect.cpp.inc"
