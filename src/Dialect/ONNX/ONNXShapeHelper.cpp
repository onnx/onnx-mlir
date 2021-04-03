/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------ONNXShapeHelper.cpp - help for shapes----------------=== //
//
// Copyright 2020 The IBM Research Authors.
//
// =============================================================================
//
// This file has the computations to compute the shapes using the new index expr
// approach.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"

#include <algorithm>

using namespace mlir;

//===----------------------------------------------------------------------===//
// ONNX Helper functions
//===----------------------------------------------------------------------===//

// Returns the ConstantOp which defines an MLIR Value or null.
ONNXConstantOp getONNXConstantOp(Value value) {
  return dyn_cast_or_null<mlir::ONNXConstantOp>(value.getDefiningOp());
}

//===----------------------------------------------------------------------===//
// ONNX Op Shape Helper
//===----------------------------------------------------------------------===//

template <class OP>
ONNXOpShapeHelper<OP>::ONNXOpShapeHelper(
    OP *newOp, ConversionPatternRewriter *rewriter)
    : op(newOp), scope(rewriter, newOp->getLoc()), outputsDims() {
  setNumberOfOutputs(op->getOperation()->getNumResults());
}

//===----------------------------------------------------------------------===//
// ONNX Op Shape Helper for ArgMax
//===----------------------------------------------------------------------===//
ONNXArgMaxOpShapeHelper::ONNXArgMaxOpShapeHelper(
    ONNXArgMaxOp *newOp, ConversionPatternRewriter *rewriter)
    : ONNXOpShapeHelper<ONNXArgMaxOp>(newOp, rewriter) {}

LogicalResult ONNXArgMaxOpShapeHelper::Compute(
    ONNXArgMaxOpAdaptor operandAdaptor) {
  // Get info about input data operand.
  Value data = operandAdaptor.data();
  int64_t dataRank = data.getType().cast<ShapedType>().getRank();

  // axis is a required attribute and should have default value of 0.
  int64_t axisValue = op->axis();

  // Accepted axis range is [-r, r-1] where r = rank(data).
  if (axisValue < -1 * (int64_t)dataRank || axisValue >= (int64_t)dataRank) {
    return op->emitError("ArgMax axis value out of bound");
  }

  if (axisValue < 0) {
    axisValue = dataRank + axisValue;
    auto builder = mlir::Builder(op->getContext());
    op->axisAttr(IntegerAttr::get(builder.getIntegerType(64, /*isSigned=*/true),
        APInt(64, /*value=*/axisValue, /*isSigned=*/true)));
  }

  // keepdims is a required attribute and should have default value of 1.
  int64_t keepdims = op->keepdims();
  bool isKeepdims = (keepdims == 1) ? true : false;

  // Compute outputDims
  DimsExpr outputDims;
  MemRefBoundIndexCapture dataBounds(data);
  int reducedRank = isKeepdims ? dataRank : dataRank - 1;
  outputDims.resize(reducedRank);
  for (auto i = 0; i < reducedRank; i++) {
    DimIndexExpr dimOutput;
    if (isKeepdims) {
      if (i != axisValue)
        dimOutput = dataBounds.getDim(i);
      else
        dimOutput = LiteralIndexExpr(1);
    } else {
      if (i < axisValue)
        dimOutput = dataBounds.getDim(i);
      else
        dimOutput = dataBounds.getDim(i + 1);
    }
    outputDims[i] = dimOutput;
  }

  // Save the final result.
  dimsForOutput(0) = outputDims;

  return success();
}

//===----------------------------------------------------------------------===//
// ONNX Op Shape Helper for Broadcasting
//===----------------------------------------------------------------------===//

ONNXOpBroadcastedShapeHelper::ONNXOpBroadcastedShapeHelper(
    ConversionPatternRewriter *rewriter, Location loc, bool uniBroadcasting)
    : scope(rewriter, loc), isUniBroadcasting(uniBroadcasting) {}

LogicalResult ONNXOpBroadcastedShapeHelper::Compute(ArrayRef<Value> operands) {
  // A temporary IndexExpr vector for the output.
  DimsExpr dimsExpr;
  int numOfInputs = operands.size();

  // Compute rank of the output. Rank of the output is the maximum rank of all
  // operands.
  for (int64_t i = 0; i < numOfInputs; ++i) {
    int64_t r = operands[i].getType().cast<ShapedType>().getRank();
    outputRank = std::max(outputRank, r);
  }
  dimsExpr.resize(outputRank);

  // Prepare dims for every input. Prepend 1s if the input's shape has smaller
  // rank, so that all the shapes have the same rank.
  for (int64_t i = 0; i < numOfInputs; ++i) {
    DimsExpr dims;
    int64_t r = operands[i].getType().cast<ShapedType>().getRank();
    MemRefBoundIndexCapture bounds(operands[i]);
    // Prepend 1s.
    for (int64_t k = 0; k < outputRank - r; ++k)
      dims.emplace_back(LiteralIndexExpr(1));
    // Get from the input.
    for (int64_t k = outputRank - r; k < outputRank; ++k)
      dims.emplace_back(bounds.getDim(k - outputRank + r));
    inputsDims.emplace_back(dims);
  }

  // Initialize the output with the first operand.
  dimsExpr = inputsDims[0];

  // Now compute each broadcasted dimension for the output.
  // folding over the other operands along the current dimension index.
  for (int64_t i = 1; i < numOfInputs; ++i) {
    for (int64_t j = 0; j < outputRank; ++j) {
      // Set the output dimension based on the two dimension values.
      // Dimension value can be one of 1, QuestionMark, LiteralNot1.
      IndexExpr currentDimExpr = dimsExpr[j];
      IndexExpr nextDimExpr = inputsDims[i][j];

      // 1 - QuestionMark
      // 1 - LiteralNot1
      // 1 - 1
      if (currentDimExpr.isLiteralAndIdenticalTo(1)) {
        if (!isUniBroadcasting)
          dimsExpr[j] = nextDimExpr;
        continue;
      }

      if (currentDimExpr.isLiteralAndDifferentThan(1)) {
        // LiteralNot1 - LiteralNot1 => keep unchanged with verifying.
        if (nextDimExpr.isLiteralAndDifferentThan(1))
          assert(currentDimExpr.isLiteralAndIdenticalTo(nextDimExpr));
        // Keep unchanged wihout verifying:
        //   - LiteralNot1 - QuestionMark
        //   - LiteralNot1 - 1
        continue;
      }

      // QuestionMark - 1 => keep unchanged.
      if (currentDimExpr.isQuestionmark() &&
          nextDimExpr.isLiteralAndIdenticalTo(1))
        continue;

      // QuestionMark - LiteralNot1 => set to LiteralNot1 without verifying.
      if (currentDimExpr.isQuestionmark() &&
          nextDimExpr.isLiteralAndDifferentThan(1)) {
        dimsExpr[j] = nextDimExpr;
        continue;
      }

      // QuestionMark - QuestionMark
      if (!isUniBroadcasting) {
        SmallVector<IndexExpr, 2> exprs({currentDimExpr, nextDimExpr});
        dimsExpr[j] = IndexExpr::max(exprs);
      }
    }
  }

  // Set the final output.
  outputDims = dimsExpr;

  return success();
}

LogicalResult ONNXOpBroadcastedShapeHelper::GetAccessExprs(Value operand,
    unsigned operandIndex, const SmallVectorImpl<IndexExpr> &outputAccessExprs,
    SmallVectorImpl<IndexExpr> &operandAccessExprs) {
  if (isUniBroadcasting && operandIndex == 0) {
    for (IndexExpr ie : outputAccessExprs)
      operandAccessExprs.emplace_back(ie);
    return success();
  }

  auto operandRank = operand.getType().cast<ShapedType>().getRank();
  for (decltype(operandRank) i = 0; i < operandRank; ++i) {
    // Shape helper may pretend 1s, thus adjust dimension index accordingly.
    auto dimIndex = outputRank - operandRank + i;
    SymbolIndexExpr dim(inputsDims[operandIndex][dimIndex]);

    // Compute access index based on broadcasting rules.
    // If all other operand dims are 1, just use the output access index.
    // Otherwise, emit a select op.
    bool allOtherInputDimsAreOne = true;
    for (int i = 0; i < inputsDims.size(); ++i) {
      if (i == operandIndex)
        continue;
      IndexExpr dim = inputsDims[i][dimIndex];
      if (!dim.isLiteralAndIdenticalTo(1)) {
        allOtherInputDimsAreOne = false;
        break;
      }
    }
    if (allOtherInputDimsAreOne) {
      operandAccessExprs.emplace_back(outputAccessExprs[dimIndex]);
    } else
      operandAccessExprs.emplace_back(
          IndexExpr::select(dim > 1, outputAccessExprs[dimIndex], 0));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ONNX Slice Op Shape Helper
//===----------------------------------------------------------------------===//

ONNXSliceOpShapeHelper::ONNXSliceOpShapeHelper(
    ONNXSliceOp *newOp, ConversionPatternRewriter *rewriter)
    : ONNXOpShapeHelper<ONNXSliceOp>(newOp, rewriter), starts(), ends(),
      steps() {}

LogicalResult ONNXSliceOpShapeHelper::Compute(
    ONNXSliceOpAdaptor operandAdaptor) {
  // Shape inference indicated by passing a null rewriter pointer.
  Operation *genericOp = reinterpret_cast<Operation *>(op);

  // Output dims of results.
  DimsExpr outputDims;

  // Get info about input data operand.
  Value data = operandAdaptor.data();
  int64_t dataRank = data.getType().cast<ShapedType>().getShape().size();

  // Get each of the axes, and save the literal values in axesIntLit.
  SmallVector<int64_t, 2> axesIntLit;
  Value axes = operandAdaptor.axes();
  if (axes.getType().isa<NoneType>()) {
    // If `axes` are omitted, they are set to `[0, ..., nDim-1]`."
    for (int i = 0; i < dataRank; ++i)
      axesIntLit.emplace_back(i);
  } else if (auto valueAttribute = getDenseElementAttributeFromValue(axes)) {
    // If `axes` are constants, read them."
    for (IntegerAttr value : valueAttribute.getValues<IntegerAttr>()) {
      int64_t axis = value.cast<IntegerAttr>().getInt();
      if (axis < 0)
        axis += dataRank;
      if (!(axis >= 0 && axis < dataRank))
        return op->emitError("Axes contains an out-of-bound index");
      axesIntLit.emplace_back(axis);
    }
  } else {
    return op->emitError("Axes must be known at compile time");
  }
  int sliceRank = axesIntLit.size();

  // Initialize context and results (start & output)
  starts.resize(dataRank);
  steps.resize(dataRank);
  ends.resize(dataRank);
  outputDims.resize(dataRank);

  // SmallVector<uint64_t, 1> index1D(1, 0);
  ArrayValueIndexCapture startsCapture(genericOp, operandAdaptor.starts());
  ArrayValueIndexCapture endsCapture(genericOp, operandAdaptor.ends());
  ArrayValueIndexCapture stepsCapture(genericOp, operandAdaptor.steps());
  MemRefBoundIndexCapture dataBounds(data);
  for (uint64_t i = 0; i < sliceRank; i++) {
    // i is index in start/step/end/output
    // ii is logical index in mem/loop bounds
    int ii = axesIntLit[i];
    // Get start, end, step, and dim index expressions.
    // Get start.
    SymbolIndexExpr startInput(startsCapture.getSymbol(i));
    if (startInput.isUndefined())
      return op->emitError("start input parameter could not be processed");
    // Get end.
    SymbolIndexExpr endInput(endsCapture.getSymbol(i));
    if (endInput.isUndefined())
      return op->emitError("end input parameter could not be processed");
    // Get step.
    SymbolIndexExpr stepInput(stepsCapture.getSymbol(i));
    if (stepInput.isUndefined())
      return op->emitError("step input parameter could not be processed");
    if (stepInput.isLiteral() && stepInput.getLiteral() == 0)
      return op->emitError("step input parameter cannot be zero");
    // Get dim.
    DimIndexExpr dimInput(dataBounds.getDim(ii));

    // Now proceed with the computations for start/end/dim.
    // Calculation for start: start < 0 ? start + dim : start.
    IndexExpr startPos =
        IndexExpr::select(startInput < 0, startInput + dimInput, startInput);
    // Step < 0: clamp(0, start, dim -1) else clamp(0, start, dim)
    IndexExpr neg = startPos.clamp(0, dimInput - 1);
    IndexExpr pos = startPos.clamp(0, dimInput);
    IndexExpr startFinal = IndexExpr::select(stepInput < 0, neg, pos);

    // Calculation for end: end<0 -> end + dim else -> end;
    // special case end <= -inf -> -1;  end >= inf -> dim;
    int64_t negInf = std::numeric_limits<int32_t>::min();
    int64_t posInf = std::numeric_limits<int32_t>::max();
    IndexExpr endPos =
        IndexExpr::select(endInput < 0, endInput + dimInput, endInput);
    endPos = endPos.selectOrSelf(endInput <= negInf, -1);
    endPos = endPos.selectOrSelf(endInput >= posInf, dimInput);
    // End: step<0: clamp(-1, end, dim); step>0 clamp(0, end, dim)
    neg = endPos.clamp(-1, dimInput);
    pos = endPos.clamp(0, dimInput);
    IndexExpr endFinal = IndexExpr::select(stepInput < 0, neg, pos);

    // Calculation for output size.
    IndexExpr dimOutputFinal = (endFinal - startFinal).ceilDiv(stepInput);
    // should use a max
    dimOutputFinal = dimOutputFinal.selectOrSelf(dimOutputFinal < 0, 0);

    // Save results
    starts[ii] = startFinal;
    steps[ii] = stepInput;
    ends[ii] = endFinal;
    outputDims[ii] = dimOutputFinal;
  }

  // Handle the default for the non-axis arrays; they are detected with 0 steps
  // (illegal value).
  bool allOutputLit;
  for (uint64_t i = 0; i < dataRank; ++i) {
    if (steps[i].isUndefined()) {
      // have one unset, put the defaults (start was already at zero, so we are
      // fine).
      starts[i] = LiteralIndexExpr(0);
      steps[i] = LiteralIndexExpr(1);
      DimIndexExpr dimInput(dataBounds.getDim(i));
      ends[i] = dimInput;
      outputDims[i] = dimInput;
    }
  }

  // Save the final result.
  dimsForOutput(0) = outputDims;

  return success();
}

//===----------------------------------------------------------------------===//
// ONNX Tile Op Shape Helper
//===----------------------------------------------------------------------===//

ONNXTileOpShapeHelper::ONNXTileOpShapeHelper(
    ONNXTileOp *newOp, ConversionPatternRewriter *rewriter)
    : ONNXOpShapeHelper<ONNXTileOp>(newOp, rewriter) {}

LogicalResult ONNXTileOpShapeHelper::Compute(ONNXTileOpAdaptor operandAdaptor) {
  // Shape inference indicated by passing a null rewriter pointer.
  Operation *genericOp = reinterpret_cast<Operation *>(op);

  // Get info about input data operand.
  Value input = operandAdaptor.input();
  // TOFIX: need to check is_a<ShapedType>?
  int64_t inputRank = input.getType().cast<ShapedType>().getShape().size();
  Value repeats = operandAdaptor.repeats();

  // Compute outputDims
  DimsExpr outputDims;
  outputDims.resize(inputRank);
  MemRefBoundIndexCapture inputBounds(input);
  ArrayValueIndexCapture repeatsCapture(genericOp, repeats);
  for (auto i = 0; i < inputRank; i++) {
    DimIndexExpr dimInput(inputBounds.getDim(i));
    SymbolIndexExpr repeatsValue(repeatsCapture.getSymbol(i));
    IndexExpr dimOutput = dimInput * repeatsValue;
    outputDims[i] = dimOutput;
  }
  dimsForOutput(0) = outputDims;
  return success();
}

//===----------------------------------------------------------------------===//
// ONNX Gemm Op Shape Helper
//===----------------------------------------------------------------------===//

ONNXGemmOpShapeHelper::ONNXGemmOpShapeHelper(
    ONNXGemmOp *newOp, ConversionPatternRewriter *rewriter)
    : ONNXOpShapeHelper<ONNXGemmOp>(newOp, rewriter), aDims(), bDims(), cDims(),
      hasBias(false), cRank(-1) {}

LogicalResult ONNXGemmOpShapeHelper::Compute(ONNXGemmOpAdaptor operandAdaptor) {
  // Shape inference indicated by passing a null rewriter pointer.
  // Output dims of result.
  DimsExpr outputDims;

  // Get info.
  Value A = operandAdaptor.A();
  Value B = operandAdaptor.B();
  Value C = operandAdaptor.C();
  hasBias = !C.getType().isa<NoneType>();

  // Test ranks.
  if (A.getType().cast<ShapedType>().getShape().size() != 2)
    return op->emitError("Gemm with A should be a 2D tensor");
  if (B.getType().cast<ShapedType>().getShape().size() != 2)
    return op->emitError("Gemm with B should be a 2D tensor");
  cRank = 0;
  if (hasBias) {
    cRank = C.getType().cast<ShapedType>().getShape().size();
    if (cRank > 2)
      return op->emitError("Gemm with C should be a 1D or 2D tensor");
  }
  // Scan dimensions of A with/without transpose.
  MemRefBoundIndexCapture ABounds(A);
  if (op->transA() == 0) {
    aDims = {ABounds.getDim(0), ABounds.getDim(1)};
  } else {
    aDims = {ABounds.getDim(1), ABounds.getDim(0)};
  }
  // Scan dimensions of B with/without transpose.
  MemRefBoundIndexCapture BBounds(B);
  if (op->transB() == 0) {
    bDims = {BBounds.getDim(0), BBounds.getDim(1)};
  } else {
    bDims = {BBounds.getDim(1), BBounds.getDim(0)};
  }
  // Set output dims of result, creating a copy of it to be safe.
  outputDims = {aDims[0].deepCopy(), bDims[1].deepCopy()};
  // Bias C can be a (unidirectional) broadcast.
  MemRefBoundIndexCapture CBounds(C);
  if (hasBias) {
    if (cRank == 0) {
      // Broadcast for scalar: both dims are 1.
      cDims = {LiteralIndexExpr(1), LiteralIndexExpr(1)};
    } else if (cRank == 1) {
      // First dim is the one padded.
      cDims = {LiteralIndexExpr(1), CBounds.getDim(0)};
    } else {
      assert(cRank == 2 && "illegal path");
      cDims = {CBounds.getDim(0), CBounds.getDim(1)};
    }
  }
  // Check static dimensions, if we can.
  if (aDims[1].isLiteral() && bDims[0].isLiteral() &&
      aDims[1].getLiteral() != bDims[0].getLiteral()) {
    return op->emitError("Gemm 2nd dim of A is different than 1st dim of B");
  }
  if (hasBias) {
    // Check first dim.
    if (outputDims[0].isLiteral() && cDims[0].isLiteral()) {
      if (cDims[0].getLiteral() == 1 ||
          cDims[0].getLiteral() == outputDims[0].getLiteral()) {
        // We are fine.
      } else {
        return op->emitError("bias add has bad dimension on first dim");
      }
    }
    // Check second dim.
    if (outputDims[1].isLiteral() && cDims[1].isLiteral()) {
      if (cDims[1].getLiteral() == 1 ||
          cDims[1].getLiteral() == outputDims[1].getLiteral()) {
        // We are fine.
      } else {
        return op->emitError("bias add has bad dimension on second dim");
      }
    }
  }
  // Save the final result.
  dimsForOutput(0) = outputDims;
  return success();
}

//===----------------------------------------------------------------------===//
// ONNX MatMul Op Shape Helper
//===----------------------------------------------------------------------===//

ONNXMatMulOpShapeHelper::ONNXMatMulOpShapeHelper(
    ONNXMatMulOp *newOp, ConversionPatternRewriter *rewriter)
    : ONNXOpShapeHelper<ONNXMatMulOp>(newOp, rewriter), aDims(), bDims(),
      aPadDims(), bPadDims() {}

LogicalResult ONNXMatMulOpShapeHelper::Compute(
    ONNXMatMulOpAdaptor operandAdaptor) {
  // Shape inference indicated by passing a null rewriter pointer.
  // Output dims of result.
  DimsExpr outputDims;

  // Get info.
  Value A = operandAdaptor.A();
  Value B = operandAdaptor.B();
  int aRank = A.getType().cast<ShapedType>().getShape().size();
  int bRank = B.getType().cast<ShapedType>().getShape().size();

  // Size all the arrays to padded length.
  int paddedRank = std::max(aRank, bRank);
  paddedRank = std::max(paddedRank, 2);
  aDims.resize(paddedRank);
  bDims.resize(paddedRank);
  aPadDims.resize(paddedRank, false);
  bPadDims.resize(paddedRank, false);
  // Add the dims of A. All of the aDim[0]...aDim[aRank-1] are in the rightmost
  // positions, prepended by 1s to fit the paddedRankSize.
  // (1,1,1... 1, aDim[0]...aDim[aRank-1])

  LiteralIndexExpr one(1);
  MemRefBoundIndexCapture ABounds(A);
  int aOffset = paddedRank - aRank;
  for (int i = 0; i < aOffset; ++i) {
    aDims[i] = one;
    aPadDims[i] = true;
  }
  for (int i = 0; i < aRank; ++i) {
    aDims[i + aOffset] = ABounds.getDim(i);
    aPadDims[i + aOffset] = false; // Pad false even if dim is sized 1.
  }
  // for B: two cases. If bRank = 1, we pad the rightmost position. Namely we
  // get (1...,1, bDim[0], 1). We use one padding credit for the rightmost
  // position. Otherwise, when bRank>1, we only pad the leading positions.
  // Namely we get (1,1,1...,1, bDim[0],.... bDim[bRank-1])
  MemRefBoundIndexCapture BBounds(B);
  int bOffset = paddedRank - bRank;
  if (bRank == 1) {
    bDims[paddedRank - 1] = one;
    bPadDims[paddedRank - 1] = true;
    bOffset--;
  }
  for (int i = 0; i < bOffset; ++i) {
    bDims[i] = one;
    bPadDims[i] = true;
  }
  for (int i = 0; i < bRank; ++i) {
    bDims[i + bOffset] = BBounds.getDim(i);
    bPadDims[i + bOffset] = false; // Pad false even if dim is sized 1.
  }
  assert(aDims.size() == bDims.size() && "padded A&B must have same size");

  // Fill in the output dimensions, start with the non-matmul dims.
  for (int i = 0; i < paddedRank - 2; ++i) {
    // Check for broadcast, then literals, then runtime for both.
    if (aDims[i].isLiteralAndIdenticalTo(1)) {
      // A is broadcast, use B dim.
      outputDims.emplace_back(bDims[i]);
    } else if (bDims[i].isLiteralAndIdenticalTo(1)) {
      // B is a broadcast, use A dim.
      outputDims.emplace_back(aDims[i]);
    } else if (aDims[i].isLiteral() && bDims[i].isLiteral()) {
      // No broadcast, both literals, make sure they have the same value.
      if (aDims[i].getLiteral() != bDims[i].getLiteral())
        return op->emitError("Incompatible size detected");
      outputDims.emplace_back(aDims[i]);
    } else if (aDims[i].isLiteral()) {
      // A dim is a literal; use it here for output and b, since b
      // is guaranteed not to be a broadcast (earlier tests).
      outputDims.emplace_back(aDims[i]);
      bDims[i] = aDims[i]; // Add runtime check if desired.
    } else if (bDims[i].isLiteral()) {
      // A dim is a literal; use it here for output and a, since a
      // is guaranteed not to be a broadcast (earlier tests).
      outputDims.emplace_back(bDims[i]);
      aDims[i] = bDims[i]; // Add runtime check if desired.
    } else {
      // Have no broadcast or literal, just pick a for output; add runtime check
      // if desired.
      outputDims.emplace_back(aDims[i]);
    }
  }
  // We now check get the last two dimensions: NxK times KxM.
  int aN = paddedRank - 2;
  int aK = paddedRank - 1;
  int bK = paddedRank - 2;
  int bM = paddedRank - 1;
  // And test the K dimensions.
  if (aDims[aK].isLiteral() && bDims[bK].isLiteral()) {
    if (aDims[aK].getLiteral() != bDims[bK].getLiteral())
      return op->emitError("reduction dimension must be the same");
  } else if (aDims[aK].isLiteral()) {
    // Save aK dims into bK dims, in case bK dims was runtime
    bDims[bK] = aDims[aK];
  } else if (bDims[bK].isLiteral()) {
    // Save bK dims into aK dims, in case aK dims was runtime
    aDims[aK] = bDims[bK];
  }
  // Add lower N x M dimensions if they are not padded dimensions.
  if (!aPadDims[aN])
    outputDims.emplace_back(aDims[aN]);
  if (!bPadDims[bM])
    outputDims.emplace_back(bDims[bM]);
  // For the case where both aRank == bRank == 1
  if (aRank == 1 && bRank == 1) {
    outputDims.emplace_back(one);
  }
  // Save the final result.
  dimsForOutput(0) = outputDims;
  return success();
}

//===----------------------------------------------------------------------===//
// ONNX Split Op Shape Helper
//===----------------------------------------------------------------------===//

ONNXSplitOpShapeHelper::ONNXSplitOpShapeHelper(
    ONNXSplitOp *newOp, ConversionPatternRewriter *rewriter)
    : ONNXOpShapeHelper<ONNXSplitOp>(newOp, rewriter) {}

LogicalResult ONNXSplitOpShapeHelper::Compute(
    ONNXSplitOpAdaptor operandAdaptor) {
  // Shape inference indicated by passing a null rewriter pointer.
  // Get info about input and output data.
  int numOfResults = op->getNumResults();
  auto rank = operandAdaptor.input().getType().cast<ShapedType>().getRank();

  // Checking value of axis parameter.
  int64_t axisIndex = op->axis();
  if (axisIndex < -rank || axisIndex >= rank)
    return op->emitError("Split axis value out of bound");
  // Negative axis means values are counted from the opposite side.
  if (axisIndex < 0) {
    axisIndex = rank + axisIndex;
    auto builder = mlir::Builder(op->getContext());
    op->axisAttr(IntegerAttr::get(builder.getIntegerType(64, /*isSigned=*/true),
        APInt(64, /*value=*/axisIndex, /*isSigned=*/true)));
  }

  // Checking value of split parameter.
  auto splitAttribute = op->split();
  SmallVector<IndexExpr, 4> splitDims;
  MemRefBoundIndexCapture inputBounds(operandAdaptor.input());
  if (splitAttribute.hasValue()) {
    if (ArrayAttrSize(splitAttribute) != numOfResults)
      return op->emitError("Split size not equal to the number of results");
    for (int i = 0; i < numOfResults; ++i) {
      LiteralIndexExpr dim(ArrayAttrIntVal(splitAttribute, i));
      splitDims.emplace_back(dim);
    }
  } else {
    // If split parameter is not specified, the dimension is split to
    // equal-sized parts.
    DimIndexExpr splitInputDim(inputBounds.getDim(axisIndex));
    LiteralIndexExpr numOfPartitions(numOfResults);
    if (splitInputDim.isLiteral() &&
        (splitInputDim.getLiteral() % numOfResults != 0))
      return op->emitError("The dimension at the split axis is "
                           "expected to be divisible by the number of results");
    for (int i = 0; i < numOfResults; ++i) {
      IndexExpr splitDim = splitInputDim.ceilDiv(numOfPartitions);
      splitDims.emplace_back(splitDim);
    }
  }

  // Build result types.
  for (int i = 0; i < numOfResults; ++i) {
    DimsExpr outputDims;
    outputDims.resize(rank);
    for (int j = 0; j < rank; ++j) {
      if (j == axisIndex) {
        outputDims[j] = splitDims[i];
      } else {
        outputDims[j] = inputBounds.getDim(j);
      }
    }
    dimsForOutput(i) = outputDims;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ONNX Gather Op Shape Helper
//===----------------------------------------------------------------------===//

ONNXGatherOpShapeHelper::ONNXGatherOpShapeHelper(
    ONNXGatherOp *newOp, ConversionPatternRewriter *rewriter)
    : ONNXOpShapeHelper<ONNXGatherOp>(newOp, rewriter), dataDims(),
      indicesDims(), positiveConstantIndices(false) {}

LogicalResult ONNXGatherOpShapeHelper::Compute(
    ONNXGatherOpAdaptor operandAdaptor) {
  // Shape inference indicated by passing a null rewriter pointer.
  // Read data and indices shapes as dim indices.
  MemRefBoundIndexCapture dataBounds(operandAdaptor.data());
  MemRefBoundIndexCapture indicesBounds(operandAdaptor.indices());
  dataBounds.getDimList(dataDims);
  indicesBounds.getDimList(indicesDims);

  // Read constant 'axis' attribute and normalize when negative.
  int64_t axisIndex = op->axis();
  // The 'axis' value must be in [-rank, rank-1].
  int dataRank = dataDims.size();
  if (axisIndex < -dataRank || axisIndex >= dataRank)
    return op->emitError("Gather axis value out of bound");
  // Convert a negative axis to a positive axis.
  if (axisIndex < 0) {
    axisIndex += dataRank;
    auto builder = mlir::Builder(op->getContext());
    op->axisAttr(IntegerAttr::get(builder.getIntegerType(64, /*isSigned=*/true),
        APInt(64, /*value=*/axisIndex, /*isSigned=*/true)));
  }

  // If 'indices' is a constant tensor, check whether its values are valid.
  if (dataDims[axisIndex].isLiteral()) {
    auto valueAttribute =
        getDenseElementAttributeFromValue(operandAdaptor.indices());
    if (valueAttribute) {
      int64_t dataDimAtAxis = dataDims[axisIndex].getLiteral();
      positiveConstantIndices = true;
      for (auto value : valueAttribute.getValues<IntegerAttr>()) {
        auto index = value.cast<IntegerAttr>().getInt();
        if (index < -dataDimAtAxis || index >= dataDimAtAxis)
          return op->emitError("Indices tensor contains an out-of-bound index");
        if (index < 0)
          // TODO: make the negative consant number positive.
          positiveConstantIndices = false;
      }
    }
  }

  // Output has rank of 'indicesRank + (dataRank - 1).
  // Output shape is constructed from 'input' by:
  //    replacing the dimension at 'axis' in 'input' by the shape of 'indices'.
  for (int i = 0; i < dataRank; ++i) {
    if (i == axisIndex)
      for (IndexExpr d : indicesDims)
        dimsForOutput(0).emplace_back(d);
    else
      dimsForOutput(0).emplace_back(dataDims[i]);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ONNX Concat Op Shape Helper
//===----------------------------------------------------------------------===//

ONNXConcatOpShapeHelper::ONNXConcatOpShapeHelper(
    ONNXConcatOp *newOp, ConversionPatternRewriter *rewriter)
    : ONNXOpShapeHelper<ONNXConcatOp>(newOp, rewriter) {}

LogicalResult ONNXConcatOpShapeHelper::Compute(
    ONNXConcatOpAdaptor operandAdaptor) {

  int inputNum = op->getNumOperands();
  Value firstInput = operandAdaptor.getODSOperands(0)[0];
  auto commonType = firstInput.getType().cast<ShapedType>();
  auto commonShape = commonType.getShape();
  auto commonRank = commonShape.size();
  int64_t axisIndex = op->axis();

  // Negative axis means values are counted from the opposite side.
  // TOFIX should be in normalization pass
  if (axisIndex < 0) {
    axisIndex = commonRank + axisIndex;
  }

  IndexExpr cumulativeAxisSize = LiteralIndexExpr(0);
  for (int i = 0; i < inputNum; ++i) {
    Value currentInput = operandAdaptor.getODSOperands(0)[i];
    MemRefBoundIndexCapture currInputBounds(currentInput);
    DimIndexExpr currentSize(currInputBounds.getDim(axisIndex));
    cumulativeAxisSize = cumulativeAxisSize + currentSize;
  }

  DimsExpr outputDims;
  MemRefBoundIndexCapture firstInputBounds(firstInput);
  outputDims.resize(commonRank);
  for (int i = 0; i < commonRank; i++) {
    if (i == axisIndex) {
      outputDims[i] = cumulativeAxisSize;
    } else {
      outputDims[i] = firstInputBounds.getDim(i);
    }
  }

  dimsForOutput(0) = outputDims;
  return success();
}

//===----------------------------------------------------------------------===//
// ONNX Transpose Op Shape Helper
//===----------------------------------------------------------------------===//

ONNXTransposeOpShapeHelper::ONNXTransposeOpShapeHelper(
    ONNXTransposeOp *newOp, ConversionPatternRewriter *rewriter)
    : ONNXOpShapeHelper<ONNXTransposeOp>(newOp, rewriter) {}

LogicalResult ONNXTransposeOpShapeHelper::Compute(
    ONNXTransposeOpAdaptor operandAdaptor) {
  // Shape inference indicated by passing a null rewriter pointer.
  Operation *genericOp = reinterpret_cast<Operation *>(op);

  // Basic information.
  auto rank = operandAdaptor.data().getType().cast<ShapedType>().getRank();

  // Transposition which handles the default case of
  // reversing the shape of the tensor (similar to numpy.transpose).
  ArrayAttr permAttr = op->permAttr();
  if (!permAttr) {
    // Generate reverse order for default transpose operation.
    SmallVector<int64_t, 4> defaultVals;
    auto builder = mlir::Builder(op->getContext());
    for (int i = rank - 1; i >= 0; --i)
      defaultVals.emplace_back(i);
    // Set default attribute.
    ArrayRef<int64_t> defaultRefs(defaultVals);
    op->permAttr(builder.getI64ArrayAttr(defaultRefs));
    permAttr = op->permAttr();
  }

  // Perform transposition according to perm attribute.
  DimsExpr transposedDims;
  MemRefBoundIndexCapture dataBounds(operandAdaptor.data());
  for (decltype(rank) i = 0; i < rank; ++i) {
    int64_t inputIndex = ArrayAttrIntVal(permAttr, i);
    transposedDims.emplace_back(dataBounds.getDim(inputIndex));
  }

  // Set type for the first output.
  dimsForOutput(0) = transposedDims;
  return success();
}

//===----------------------------------------------------------------------===//
// ONNX LRN Op Shape Helper
//===----------------------------------------------------------------------===//

ONNXLRNOpShapeHelper::ONNXLRNOpShapeHelper(
    ONNXLRNOp *newOp, ConversionPatternRewriter *rewriter)
    : ONNXOpShapeHelper<ONNXLRNOp>(newOp, rewriter) {}

LogicalResult ONNXLRNOpShapeHelper::Compute(ONNXLRNOpAdaptor operandAdaptor) {
  // Shape inference indicated by passing a null rewriter pointer.
  Operation *genericOp = reinterpret_cast<Operation *>(op);

  // Basic information.
  auto rank = operandAdaptor.X().getType().cast<ShapedType>().getRank();

  // Perform transposition according to perm attribute.
  DimsExpr outputDims;
  MemRefBoundIndexCapture XBounds(operandAdaptor.X());
  for (decltype(rank) i = 0; i < rank; ++i) {
    outputDims.emplace_back(XBounds.getDim(i));
  }

  // Set type for the first output.
  dimsForOutput(0) = outputDims;
  return success();
}
