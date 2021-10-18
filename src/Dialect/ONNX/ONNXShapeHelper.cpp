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
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"

#include <algorithm>

using namespace mlir;

#define DEBUG_TYPE "shape-helper"

//===----------------------------------------------------------------------===//
// ONNX Op Shape Helper
//===----------------------------------------------------------------------===//

// Reuse scope if given, otherwise create one now and free in destructor.
template <class OP>
ONNXOpShapeHelper<OP>::ONNXOpShapeHelper(
    OP *newOp, int numResults, IndexExprScope *inScope)
    : op(newOp), fGetDenseVal(getDenseElementAttributeFromONNXValue),
      fLoadVal(nullptr), outputsDims(), ownScope(inScope == nullptr) {
  assert(op && "Expecting a valid pointer");
  if (ownScope)
    scope = new IndexExprScope(nullptr, newOp->getLoc());
  assert(scope && "expected a fully formed scope");
  setNumberOfOutputs(numResults);
}

template <class OP>
ONNXOpShapeHelper<OP>::ONNXOpShapeHelper(OP *newOp, int numResults,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseValInput,
    ArrayValueIndexCapture::LoadVal fLoadVal, IndexExprScope *inScope)
    : op(newOp), fLoadVal(fLoadVal), outputsDims(),
      ownScope(inScope == nullptr) {
  assert(op && "Expecting a valid pointer");
  if (ownScope)
    scope = new IndexExprScope(rewriter, newOp->getLoc());
  assert(scope && "expected a fully formed scope");
  setNumberOfOutputs(numResults);
  // Get the dense value by combining provided function (if any) with the
  // default one.
  fGetDenseVal = [=](Value array) {
    DenseElementsAttr res = nullptr;
    // Try with the provided method, if any.
    if (fGetDenseValInput)
      res = fGetDenseValInput(array);
    // If provided method was not provided or failed, try default ONNX method.
    if (!res)
      res = getDenseElementAttributeFromONNXValue(array);
    return res;
  };
}

//===----------------------------------------------------------------------===//
// ONNX Op Shape Helper for Broadcasting
//===----------------------------------------------------------------------===//

// Since Broadcasted ops are generic (due to implementation in Elementwise.cpp
// templates), we assume below that each of the broadcast ops have exactly one
// output. If that were not the case, then we would need to pass the number of
// results as a parameter to both constructors below.

template <class OP>
ONNXOpBroadcastedShapeHelper<OP>::ONNXOpBroadcastedShapeHelper(OP *newOp,
    IndexExprScope *inScope, bool uniBroadcasting, bool noBroadcasting)
    : ONNXOpShapeHelper<OP>(newOp, 1, inScope), inputsDims(), outputRank(-1),
      isUniBroadcasting(uniBroadcasting), isNoBroadcasting(noBroadcasting) {}

template <class OP>
ONNXOpBroadcastedShapeHelper<OP>::ONNXOpBroadcastedShapeHelper(OP *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal, IndexExprScope *inScope,
    bool uniBroadcasting, bool noBroadcasting)
    : ONNXOpShapeHelper<OP>(
          newOp, 1, rewriter, fGetDenseVal, fLoadVal, inScope),
      inputsDims(), outputRank(-1), isUniBroadcasting(uniBroadcasting),
      isNoBroadcasting(noBroadcasting) {}

template <class OP>
LogicalResult ONNXOpBroadcastedShapeHelper<OP>::computeShape(
    ArrayRef<Value> operands, DimsExpr &additionalOperand) {
  // if additionalOperand is not used, we expect a zero-sized vector.
  // A temporary IndexExpr vector for the output.
  DimsExpr dimsExpr;
  int64_t numOfInputs = operands.size();

  // Compute rank of the output. Rank of the output is the maximum rank of all
  // operands.
  int additionalOperRank = additionalOperand.size();
  bool hasAdditionalOper = additionalOperRank > 0;
  outputRank = hasAdditionalOper ? additionalOperRank : -1;
  for (int64_t i = 0; i < numOfInputs; ++i) {
    int64_t r = operands[i].getType().cast<ShapedType>().getRank();
    outputRank = std::max(outputRank, r);
  }
  dimsExpr.resize(outputRank);

  // Prepare dims for every input. Prepend 1s if the input's shape has smaller
  // rank, so that all the shapes have the same rank.
  LiteralIndexExpr one(1);
  for (int64_t i = 0; i < numOfInputs; ++i) {
    MemRefBoundsIndexCapture bounds(operands[i]);
    int64_t r = bounds.getRank();
    // Prepend 1s.
    DimsExpr dims;
    for (int64_t k = 0; k < outputRank - r; ++k)
      dims.emplace_back(one);
    // Get from the input.
    for (int64_t k = 0; k < r; ++k)
      dims.emplace_back(bounds.getDim(k));
    inputsDims.emplace_back(dims);
  }
  // Handle the additional operand here.
  if (hasAdditionalOper) {
    DimsExpr dims(outputRank - additionalOperRank, one);
    for (int64_t k = 0; k < additionalOperRank; ++k)
      dims.emplace_back(additionalOperand[k]);
    inputsDims.emplace_back(dims);
    numOfInputs++;
  }

  // Initialize the output with the first operand.
  dimsExpr = inputsDims[0];

  // Note on IndexExpr. When we are not allowed to generate code, QuestionMark
  // stands for anything but a literal. When we are allowed to generate code,
  // there should be no more QuestionMarks as we are allowed to generate
  // affine/symbols/dims/non-affine expressions. Since this code predominantly
  // runs when we can gen code (as it actually does gen max ops), we should use
  // !isLiteral() for anything that is runtime. The comments were left
  // unchanged.

  //  Now compute each broadcasted dimension for the output. folding over the
  //  other operands along the current dimension index.
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
        if (!isUniBroadcasting && !isNoBroadcasting)
          dimsExpr[j] = nextDimExpr;
        continue;
      }

      if (currentDimExpr.isLiteralAndDifferentThan(1)) {
        // LiteralNot1 - LiteralNot1 => keep unchanged with verifying.
        if (nextDimExpr.isLiteralAndDifferentThan(1))
          assert(currentDimExpr.isLiteralAndIdenticalTo(nextDimExpr));
        // Keep unchanged without verifying:
        //   - LiteralNot1 - QuestionMark
        //   - LiteralNot1 - 1
        continue;
      }

      // QuestionMark - 1 => keep unchanged.
      if (!currentDimExpr.isLiteral() &&
          nextDimExpr.isLiteralAndIdenticalTo(1)) {
        continue;
      }

      // QuestionMark - LiteralNot1 => set to LiteralNot1 without verifying.
      if (!currentDimExpr.isLiteral() &&
          nextDimExpr.isLiteralAndDifferentThan(1)) {
        dimsExpr[j] = nextDimExpr;
        continue;
      }

      // QuestionMark - QuestionMark
      if (!isUniBroadcasting) {
        dimsExpr[j] = IndexExpr::max(currentDimExpr, nextDimExpr);
      }
    }
  }
  // Set the final output.
  ONNXOpShapeHelper<OP>::dimsForOutput(0) = dimsExpr;
  return success();
}

template <class OP>
LogicalResult ONNXOpBroadcastedShapeHelper<OP>::GetAccessExprs(Value operand,
    unsigned operandIndex, const SmallVectorImpl<IndexExpr> &outputAccessExprs,
    SmallVectorImpl<IndexExpr> &operandAccessExprs) {
  if (isNoBroadcasting || (isUniBroadcasting && operandIndex == 0)) {
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
    for (unsigned int i = 0; i < inputsDims.size(); ++i) {
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
    } else {
      operandAccessExprs.emplace_back(
          IndexExpr::select(dim > 1, outputAccessExprs[dimIndex], 0));
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ONNX Generic Broadcast Op Shape Helper
//===----------------------------------------------------------------------===//

ONNXGenericOpBroadcastedShapeHelper::ONNXGenericOpBroadcastedShapeHelper(
    Operation *newOp, IndexExprScope *inScope, bool uniBroadcasting,
    bool noBroadcasting)
    : ONNXOpBroadcastedShapeHelper<Operation>(
          newOp, inScope, uniBroadcasting, noBroadcasting) {}

ONNXGenericOpBroadcastedShapeHelper::ONNXGenericOpBroadcastedShapeHelper(
    Operation *newOp, OpBuilder *rewriter,
    ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal, IndexExprScope *inScope,
    bool uniBroadcasting, bool noBroadcasting)
    : ONNXOpBroadcastedShapeHelper<Operation>(newOp, rewriter, fGetDenseVal,
          fLoadVal, inScope, uniBroadcasting, noBroadcasting) {}

//===----------------------------------------------------------------------===//
// ONNX Op Shape Helper for ArgMax
//===----------------------------------------------------------------------===//

ONNXArgMaxOpShapeHelper::ONNXArgMaxOpShapeHelper(ONNXArgMaxOp *newOp)
    : ONNXOpShapeHelper<ONNXArgMaxOp>(
          newOp, newOp->getOperation()->getNumResults()) {}

ONNXArgMaxOpShapeHelper::ONNXArgMaxOpShapeHelper(ONNXArgMaxOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXOpShapeHelper<ONNXArgMaxOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal) {}

LogicalResult ONNXArgMaxOpShapeHelper::computeShape(
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
  MemRefBoundsIndexCapture dataBounds(data);
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

// ONNX DepthToSpace Op Shape Helper
//===----------------------------------------------------------------------===//

ONNXDepthToSpaceOpShapeHelper::ONNXDepthToSpaceOpShapeHelper(
    ONNXDepthToSpaceOp *newOp)
    : ONNXOpShapeHelper<ONNXDepthToSpaceOp>(
          newOp, newOp->getOperation()->getNumResults()) {}

ONNXDepthToSpaceOpShapeHelper::ONNXDepthToSpaceOpShapeHelper(
    ONNXDepthToSpaceOp *newOp, OpBuilder *rewriter,
    ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXOpShapeHelper<ONNXDepthToSpaceOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal) {}

LogicalResult ONNXDepthToSpaceOpShapeHelper::computeShape(
    ONNXDepthToSpaceOpAdaptor operandAdaptor) {
  // Get info about input data operand and blocksize.
  Value input = operandAdaptor.input();
  int64_t blocksize = op->blocksize();
  assert(input.getType().isa<ShapedType>() && "Input should have a shape");
  assert(blocksize > 0 && "blocksize should be strictly positive");

  int64_t inputRank = input.getType().cast<ShapedType>().getShape().size();
  assert(inputRank == 4 && "Unexpected input tensor rank");

  // Compute outputDims.
  // The input tensor has format [N,C,H,W], where N is the batch axis, C is the
  // channel or depth, H is the height and W is the width. The output tensor has
  // shape [N, C / (blocksize * blocksize), H * blocksize, W * blocksize].
  DimsExpr outputDims;
  outputDims.resize(inputRank);
  MemRefBoundsIndexCapture inputBounds(input);
  DimIndexExpr N(inputBounds.getDim(0));
  DimIndexExpr C(inputBounds.getDim(1));
  DimIndexExpr H(inputBounds.getDim(2));
  DimIndexExpr W(inputBounds.getDim(3));

  outputDims[0] = N;
  outputDims[1] = C.floorDiv(blocksize * blocksize);
  outputDims[2] = H * blocksize;
  outputDims[3] = W * blocksize;

  // Save the final result.
  dimsForOutput(0) = outputDims;
  return success();
}

//===----------------------------------------------------------------------===//
// ONNX Slice Op Shape Helper
//===----------------------------------------------------------------------===//

ONNXSliceOpShapeHelper::ONNXSliceOpShapeHelper(ONNXSliceOp *newOp)
    : ONNXOpShapeHelper<ONNXSliceOp>(
          newOp, newOp->getOperation()->getNumResults()),
      starts(), ends(), steps() {}

ONNXSliceOpShapeHelper::ONNXSliceOpShapeHelper(ONNXSliceOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXOpShapeHelper<ONNXSliceOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal),
      starts(), ends(), steps() {}

LogicalResult ONNXSliceOpShapeHelper::computeShape(
    ONNXSliceOpAdaptor operandAdaptor) {
  // Shape inference indicated by passing a null rewriter pointer.
  // Output dims of results.
  DimsExpr outputDims;

  // Get info about input data operand.
  Value data = operandAdaptor.data();
  uint64_t dataRank = data.getType().cast<ShapedType>().getShape().size();

  // Get each of the axes, and save the literal values in axesIntLit.
  SmallVector<int64_t, 2> axesIntLit;
  Value axes = operandAdaptor.axes();
  if (axes.getType().isa<NoneType>()) {
    // If `axes` are omitted, they are set to `[0, ..., nDim-1]`."
    for (unsigned int i = 0; i < dataRank; ++i)
      axesIntLit.emplace_back(i);
  } else if (auto valueAttribute = fGetDenseVal(axes)) {
    // If `axes` are constants, read them."
    for (IntegerAttr value : valueAttribute.getValues<IntegerAttr>()) {
      int64_t axis = value.cast<IntegerAttr>().getInt();
      if (axis < 0)
        axis += dataRank;
      if (!(axis >= 0 && axis < (int64_t)dataRank))
        return op->emitError("Axes contains an out-of-bound index");
      axesIntLit.emplace_back(axis);
    }
  } else {
    return op->emitError("Axes must be known at compile time");
  }
  uint64_t sliceRank = axesIntLit.size();

  // Initialize context and results (start & output)
  starts.resize(dataRank);
  steps.resize(dataRank);
  ends.resize(dataRank);
  outputDims.resize(dataRank);

  // SmallVector<uint64_t, 1> index1D(1, 0);
  ArrayValueIndexCapture startsCapture(
      operandAdaptor.starts(), fGetDenseVal, fLoadVal);
  ArrayValueIndexCapture endsCapture(
      operandAdaptor.ends(), fGetDenseVal, fLoadVal);
  ArrayValueIndexCapture stepsCapture(
      operandAdaptor.steps(), fGetDenseVal, fLoadVal);
  MemRefBoundsIndexCapture dataBounds(data);
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

  // Handle the default for the non-axis arrays; they are detected with 0
  // steps (illegal value).
  for (uint64_t i = 0; i < dataRank; ++i) {
    if (steps[i].isUndefined()) {
      // have one unset, put the defaults (start was already at zero, so we
      // are fine).
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

ONNXTileOpShapeHelper::ONNXTileOpShapeHelper(ONNXTileOp *newOp)
    : ONNXOpShapeHelper<ONNXTileOp>(
          newOp, newOp->getOperation()->getNumResults()) {}

ONNXTileOpShapeHelper::ONNXTileOpShapeHelper(ONNXTileOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXOpShapeHelper<ONNXTileOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal) {}

LogicalResult ONNXTileOpShapeHelper::computeShape(
    ONNXTileOpAdaptor operandAdaptor) {
  // Shape inference indicated by passing a null rewriter pointer.
  // Get info about input data operand.
  Value input = operandAdaptor.input();
  // TOFIX: need to check is_a<ShapedType>?
  int64_t inputRank = input.getType().cast<ShapedType>().getShape().size();
  Value repeats = operandAdaptor.repeats();

  // Compute outputDims
  DimsExpr outputDims;
  outputDims.resize(inputRank);
  MemRefBoundsIndexCapture inputBounds(input);
  ArrayValueIndexCapture repeatsCapture(repeats, fGetDenseVal, fLoadVal);
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

ONNXGemmOpShapeHelper::ONNXGemmOpShapeHelper(ONNXGemmOp *newOp)
    : ONNXOpShapeHelper<ONNXGemmOp>(
          newOp, newOp->getOperation()->getNumResults()),
      aDims(), bDims(), cDims(), hasBias(false), cRank(-1) {}

ONNXGemmOpShapeHelper::ONNXGemmOpShapeHelper(ONNXGemmOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXOpShapeHelper<ONNXGemmOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal),
      aDims(), bDims(), cDims(), hasBias(false), cRank(-1) {}

LogicalResult ONNXGemmOpShapeHelper::computeShape(
    ONNXGemmOpAdaptor operandAdaptor) {
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
  MemRefBoundsIndexCapture ABounds(A);
  if (op->transA() == 0) {
    aDims = {ABounds.getDim(0), ABounds.getDim(1)};
  } else {
    aDims = {ABounds.getDim(1), ABounds.getDim(0)};
  }
  // Scan dimensions of B with/without transpose.
  MemRefBoundsIndexCapture BBounds(B);
  if (op->transB() == 0) {
    bDims = {BBounds.getDim(0), BBounds.getDim(1)};
  } else {
    bDims = {BBounds.getDim(1), BBounds.getDim(0)};
  }
  // Set output dims of result, creating a copy of it to be safe.
  outputDims = {aDims[0].deepCopy(), bDims[1].deepCopy()};
  // Bias C can be a (unidirectional) broadcast.
  MemRefBoundsIndexCapture CBounds(C);
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

ONNXMatMulOpShapeHelper::ONNXMatMulOpShapeHelper(ONNXMatMulOp *newOp)
    : ONNXOpShapeHelper<ONNXMatMulOp>(
          newOp, newOp->getOperation()->getNumResults()),
      aDims(), bDims(), aPadDims(), bPadDims() {}

ONNXMatMulOpShapeHelper::ONNXMatMulOpShapeHelper(ONNXMatMulOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXOpShapeHelper<ONNXMatMulOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal),
      aDims(), bDims(), aPadDims(), bPadDims() {}

LogicalResult ONNXMatMulOpShapeHelper::computeShape(
    ONNXMatMulOpAdaptor operandAdaptor) {
  // Shape inference indicated by passing a null rewriter pointer.
  // Output dims of result.
  DimsExpr outputDims;

  // Get info.
  Value A = operandAdaptor.A();
  Value B = operandAdaptor.B();
  MemRefBoundsIndexCapture ABounds(A);
  MemRefBoundsIndexCapture BBounds(B);

  // Size all the arrays to padded length.
  int paddedRank = std::max(ABounds.getRank(), BBounds.getRank());
  paddedRank = std::max(paddedRank, 2);
  aDims.resize(paddedRank);
  bDims.resize(paddedRank);
  aPadDims.resize(paddedRank, false);
  bPadDims.resize(paddedRank, false);

  // Add the dims of A. All of the aDim[0]...aDim[aRank-1] are in the
  // rightmost positions, prepended by 1s to fit the paddedRankSize. (1,1,1...
  // 1, aDim[0]...aDim[aRank-1])
  LiteralIndexExpr one(1);
  int aOffset = paddedRank - ABounds.getRank();
  for (int i = 0; i < aOffset; ++i) {
    aDims[i] = one;
    aPadDims[i] = true;
  }
  for (unsigned int i = 0; i < ABounds.getRank(); ++i) {
    aDims[i + aOffset] = ABounds.getDim(i);
    aPadDims[i + aOffset] = false; // Pad false even if dim is sized 1.
  }
  // for B: two cases. If bRank = 1, we pad the rightmost position. Namely we
  // get (1...,1, bDim[0], 1). We use one padding credit for the rightmost
  // position. Otherwise, when bRank>1, we only pad the leading positions.
  // Namely we get (1,1,1...,1, bDim[0],.... bDim[bRank-1])
  int bOffset = paddedRank - BBounds.getRank();
  if (BBounds.getRank() == 1) {
    bDims[paddedRank - 1] = one;
    bPadDims[paddedRank - 1] = true;
    bOffset--;
  }
  for (int i = 0; i < bOffset; ++i) {
    bDims[i] = one;
    bPadDims[i] = true;
  }
  for (unsigned int i = 0; i < BBounds.getRank(); ++i) {
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
      // Have no broadcast or literal, just pick a for output; add runtime
      // check if desired.
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
  if (ABounds.getRank() == 1 && BBounds.getRank() == 1) {
    outputDims.emplace_back(one);
  }
  // Save the final result.
  dimsForOutput(0) = outputDims;
  return success();
}

//===----------------------------------------------------------------------===//
// ONNX SpaceToDepth Op Shape Helper
//===----------------------------------------------------------------------===//

ONNXSpaceToDepthOpShapeHelper::ONNXSpaceToDepthOpShapeHelper(
    ONNXSpaceToDepthOp *newOp)
    : ONNXOpShapeHelper<ONNXSpaceToDepthOp>(
          newOp, newOp->getOperation()->getNumResults()) {}

ONNXSpaceToDepthOpShapeHelper::ONNXSpaceToDepthOpShapeHelper(
    ONNXSpaceToDepthOp *newOp, OpBuilder *rewriter,
    ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXOpShapeHelper<ONNXSpaceToDepthOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal) {}

LogicalResult ONNXSpaceToDepthOpShapeHelper::computeShape(
    ONNXSpaceToDepthOpAdaptor operandAdaptor) {
  // Get info about input data operand and blocksize.
  Value input = operandAdaptor.input();
  int64_t blocksize = op->blocksize();
  assert(input.getType().isa<ShapedType>() && "Input should have a shape");
  assert(blocksize > 0 && "blocksize should be strictly positive");

  int64_t inputRank = input.getType().cast<ShapedType>().getShape().size();
  assert(inputRank == 4 && "Unexpected input tensor rank");

  // Compute outputDims.
  // The input tensor has format [N,C,H,W], where N is the batch axis, C is the
  // channel or depth, H is the height and W is the width. The output tensor has
  // shape [N, C * blocksize * blocksize, H/blocksize, W/blocksize].
  DimsExpr outputDims;
  outputDims.resize(inputRank);
  MemRefBoundsIndexCapture inputBounds(input);
  DimIndexExpr N(inputBounds.getDim(0));
  DimIndexExpr C(inputBounds.getDim(1));
  DimIndexExpr H(inputBounds.getDim(2));
  DimIndexExpr W(inputBounds.getDim(3));

  outputDims[0] = N;
  outputDims[1] = C * blocksize * blocksize;
  outputDims[2] = H.floorDiv(blocksize);
  outputDims[3] = W.floorDiv(blocksize);

  // Save the final result.
  dimsForOutput(0) = outputDims;
  return success();
}

//===----------------------------------------------------------------------===//
// ONNX Split Op Shape Helper
//===----------------------------------------------------------------------===//

template <typename ShapeHelper, typename OperandAdaptor>
LogicalResult ONNXSplitOpShapeHelperCommon(ShapeHelper *shapeHelper,
    OperandAdaptor operandAdaptor, ArrayRef<IndexExpr> indexExprArray) {
  // Shape inference indicated by passing a null rewriter pointer.
  // Get info about input and output data.
  auto op = shapeHelper->op;
  unsigned int numOfResults = op->getNumResults();
  auto rank =
      operandAdaptor.input().getType().template cast<ShapedType>().getRank();

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

  SmallVector<IndexExpr, 4> splitDims;
  MemRefBoundsIndexCapture inputBounds(operandAdaptor.input());
  if (!indexExprArray.empty()) {
    if (indexExprArray.size() != numOfResults)
      return op->emitError("Split size not equal to the number of results");
    for (unsigned int i = 0; i < numOfResults; ++i) {
      LiteralIndexExpr dim(indexExprArray[i]);
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
    for (unsigned int i = 0; i < numOfResults; ++i) {
      IndexExpr splitDim = splitInputDim.ceilDiv(numOfPartitions);
      splitDims.emplace_back(splitDim);
    }
  }

  // Build result types.
  for (unsigned int i = 0; i < numOfResults; ++i) {
    DimsExpr outputDims;
    outputDims.resize(rank);
    for (unsigned int j = 0; j < rank; ++j) {
      if (j == axisIndex) {
        outputDims[j] = splitDims[i];
      } else {
        outputDims[j] = inputBounds.getDim(j);
      }
    }
    shapeHelper->dimsForOutput(i) = outputDims;
  }
  return success();
}

ONNXSplitOpShapeHelper::ONNXSplitOpShapeHelper(ONNXSplitOp *newOp)
    : ONNXOpShapeHelper<ONNXSplitOp>(
          newOp, newOp->getOperation()->getNumResults()) {}

ONNXSplitOpShapeHelper::ONNXSplitOpShapeHelper(ONNXSplitOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXOpShapeHelper<ONNXSplitOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal) {}

LogicalResult ONNXSplitOpShapeHelper::computeShape(
    ONNXSplitOpAdaptor operandAdaptor) {

  auto split = op->split();

  SmallVector<IndexExpr, 4> indexExprArray;
  // TODO: getONNXConstantOp might be a problem during code gen as ONNX
  // constant get lowered to global constants.
  if (auto splitConstOp = getONNXConstantOp(split)) {
    ArrayValueIndexCapture splitCapture(split, fGetDenseVal, fLoadVal);
    auto splitRank =
        splitConstOp.valueAttr().dyn_cast_or_null<DenseElementsAttr>().size();
    splitCapture.getSymbolList(splitRank, indexExprArray);
  } else if (!split.getType().template isa<NoneType>()) {
    llvm_unreachable("dynamic split not yet supported");
  }

  return ONNXSplitOpShapeHelperCommon(this, operandAdaptor, indexExprArray);
}

ONNXSplitV11OpShapeHelper::ONNXSplitV11OpShapeHelper(ONNXSplitV11Op *newOp)
    : ONNXOpShapeHelper<ONNXSplitV11Op>(
          newOp, newOp->getOperation()->getNumResults()) {}

ONNXSplitV11OpShapeHelper::ONNXSplitV11OpShapeHelper(ONNXSplitV11Op *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXOpShapeHelper<ONNXSplitV11Op>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal) {}

LogicalResult ONNXSplitV11OpShapeHelper::computeShape(
    ONNXSplitV11OpAdaptor operandAdaptor) {
  auto splitAttr = op->split();
  SmallVector<IndexExpr, 4> indexExprArray;
  if (splitAttr.hasValue()) {
    ArrayAttributeIndexCapture splitCapture(splitAttr.getValue());
    auto splitRank = splitCapture.size();
    for (unsigned i = 0; i < splitRank; ++i) {
      indexExprArray.emplace_back(splitCapture.getLiteral(i));
    }
  }
  return ONNXSplitOpShapeHelperCommon(this, operandAdaptor, indexExprArray);
}

//===----------------------------------------------------------------------===//
// ONNX Gather Op Shape Helper
//===----------------------------------------------------------------------===//

ONNXGatherOpShapeHelper::ONNXGatherOpShapeHelper(ONNXGatherOp *newOp)
    : ONNXOpShapeHelper<ONNXGatherOp>(
          newOp, newOp->getOperation()->getNumResults()),
      dataDims(), indicesDims(), positiveConstantIndices(false) {}

ONNXGatherOpShapeHelper::ONNXGatherOpShapeHelper(ONNXGatherOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXOpShapeHelper<ONNXGatherOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal),
      dataDims(), indicesDims(), positiveConstantIndices(false) {}

LogicalResult ONNXGatherOpShapeHelper::computeShape(
    ONNXGatherOpAdaptor operandAdaptor) {
  // Shape inference indicated by passing a null rewriter pointer.
  // Read data and indices shapes as dim indices.
  MemRefBoundsIndexCapture dataBounds(operandAdaptor.data());
  MemRefBoundsIndexCapture indicesBounds(operandAdaptor.indices());
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
    auto valueAttribute = fGetDenseVal(operandAdaptor.indices());
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
  //    replacing the dimension at 'axis' in 'input' by the shape of
  //    'indices'.
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

ONNXConcatOpShapeHelper::ONNXConcatOpShapeHelper(ONNXConcatOp *newOp)
    : ONNXOpShapeHelper<ONNXConcatOp>(
          newOp, newOp->getOperation()->getNumResults()) {}

ONNXConcatOpShapeHelper::ONNXConcatOpShapeHelper(ONNXConcatOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXOpShapeHelper<ONNXConcatOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal) {}

LogicalResult ONNXConcatOpShapeHelper::computeShape(
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
    MemRefBoundsIndexCapture currInputBounds(currentInput);
    DimIndexExpr currentSize(currInputBounds.getDim(axisIndex));
    cumulativeAxisSize = cumulativeAxisSize + currentSize;
  }

  DimsExpr outputDims;
  MemRefBoundsIndexCapture firstInputBounds(firstInput);
  outputDims.resize(commonRank);
  for (unsigned int i = 0; i < commonRank; i++) {
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

ONNXTransposeOpShapeHelper::ONNXTransposeOpShapeHelper(ONNXTransposeOp *newOp)
    : ONNXOpShapeHelper<ONNXTransposeOp>(
          newOp, newOp->getOperation()->getNumResults()) {}

ONNXTransposeOpShapeHelper::ONNXTransposeOpShapeHelper(ONNXTransposeOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXOpShapeHelper<ONNXTransposeOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal) {}

LogicalResult ONNXTransposeOpShapeHelper::computeShape(
    ONNXTransposeOpAdaptor operandAdaptor) {
  // Shape inference indicated by passing a null rewriter pointer.
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
  MemRefBoundsIndexCapture dataBounds(operandAdaptor.data());
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
ONNXLRNOpShapeHelper::ONNXLRNOpShapeHelper(ONNXLRNOp *newOp)
    : ONNXOpShapeHelper<ONNXLRNOp>(
          newOp, newOp->getOperation()->getNumResults()) {}

ONNXLRNOpShapeHelper::ONNXLRNOpShapeHelper(ONNXLRNOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXOpShapeHelper<ONNXLRNOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal) {}

LogicalResult ONNXLRNOpShapeHelper::computeShape(
    ONNXLRNOpAdaptor operandAdaptor) {
  // Shape inference indicated by passing a null rewriter pointer.
  // Basic information.
  auto rank = operandAdaptor.X().getType().cast<ShapedType>().getRank();

  // Perform transposition according to perm attribute.
  DimsExpr outputDims;
  MemRefBoundsIndexCapture XBounds(operandAdaptor.X());
  for (decltype(rank) i = 0; i < rank; ++i) {
    outputDims.emplace_back(XBounds.getDim(i));
  }

  // Set type for the first output.
  dimsForOutput(0) = outputDims;
  return success();
}

//===----------------------------------------------------------------------===//
// ONNX Generic Pooling Op Shape Helper
//===----------------------------------------------------------------------===//

template <typename OP_TYPE, typename OP_ADAPTOR>
ONNXGenericPoolShapeHelper<OP_TYPE, OP_ADAPTOR>::ONNXGenericPoolShapeHelper(
    OP_TYPE *newOp, bool hasFilter, bool ceilMode)
    : ONNXOpShapeHelper<OP_TYPE>(newOp, newOp->getOperation()->getNumResults()),
      hasFilter(hasFilter), ceilMode(ceilMode) {}

template <typename OP_TYPE, typename OP_ADAPTOR>
ONNXGenericPoolShapeHelper<OP_TYPE, OP_ADAPTOR>::ONNXGenericPoolShapeHelper(
    OP_TYPE *newOp, bool hasFilter, bool ceilMode, OpBuilder *rewriter,
    ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXOpShapeHelper<OP_TYPE>(newOp, newOp->getOperation()->getNumResults(),
          rewriter, fGetDenseVal, fLoadVal),
      hasFilter(hasFilter), ceilMode(ceilMode) {}

template <typename OP_TYPE, typename OP_ADAPTOR>
LogicalResult ONNXGenericPoolShapeHelper<OP_TYPE, OP_ADAPTOR>::computeShape(
    OP_ADAPTOR operandAdaptor, Value filterValue,
    Optional<ArrayAttr> kernelShapeOpt, Optional<ArrayAttr> padOpt,
    Optional<ArrayAttr> strideOpt, Optional<ArrayAttr> dilationOpt) {
  // Shape inference indicated by passing a null rewriter pointer.
  // Basic information.
  Value xValue = (Value)operandAdaptor.X();
  int64_t rank = xValue.getType().cast<ShapedType>().getRank();
  int64_t spatialOffset = 2;
  int64_t spatialRank = rank - spatialOffset;

  MemRefBoundsIndexCapture XBounds(operandAdaptor.X());
  MemRefBoundsIndexCapture WBounds(filterValue);

  // Fill the stride, dilation, kernel.
  for (int i = 0; i < spatialRank; ++i) {
    // Strides, default 1.
    strides.emplace_back(
        strideOpt.hasValue() ? ArrayAttrIntVal(strideOpt, i) : 1);
    // Dilations, default 1.
    dilations.emplace_back(
        dilationOpt.hasValue() ? ArrayAttrIntVal(dilationOpt, i) : 1);
    // Kernel shape from attribute, default from Weight's spatial dims.
    if (kernelShapeOpt.hasValue()) {
      kernelShape.emplace_back(
          LiteralIndexExpr(ArrayAttrIntVal(kernelShapeOpt, i)));
    } else if (hasFilter) {
      int ii = i + spatialOffset;
      kernelShape.emplace_back(WBounds.getSymbol(ii));
    } else {
      llvm_unreachable("should have tested the availability of kernel shape");
    }
  }
  // Pads, at this stage a given compile-time literal or default 0.
  for (int i = 0; i < 2 * spatialRank; ++i) {
    int64_t p = padOpt.hasValue() ? ArrayAttrIntVal(padOpt, i) : 0;
    pads.emplace_back(LiteralIndexExpr(p));
  }

  // Handle output size: start by inserting batch size and output channels.
  DimsExpr outputDims;
  outputDims.emplace_back(XBounds.getDim(0));
  if (hasFilter)
    outputDims.emplace_back(WBounds.getDim(0)); // CO may be different from CI.
  else
    outputDims.emplace_back(XBounds.getDim(1)); // CO is CI.

  // Insert dimensions for the spatial axes. From MaxPool:
  // https://github.com/onnx/onnx/blob/master/docs/Operators.md#maxpool
  //
  // NOSET:
  //  * O[i] = floor((I[i] + P[i] - ((K[i] - 1) * d[i] + 1)) / s[i] + 1)
  // VALID:
  // * O[i] = floor((I[i] - {(K[i] - 1) * d[i] + 1} + 1) / s[i])
  // * P = 0
  // SAME_LOWER or SAME_UPPER:
  // * O[i] = ceil(I[i] / s[i])
  // * p' = (O[i] - 1) * s[i] + ((K[i] - 1) * d[i] + 1) - I[i]
  // * P[i] = p' / 2, if odd, first or second are increased by one.
  auto autoPad = ONNXOpShapeHelper<OP_TYPE>::op->auto_pad();
  LiteralIndexExpr zero(0);
  LiteralIndexExpr one(1);
  for (int64_t i = 0; i < spatialRank; ++i) {
    int64_t ii = i + spatialOffset;
    IndexExpr I = XBounds.getDim(ii);
    IndexExpr K = kernelShape[i];
    LiteralIndexExpr d(dilations[i]);
    LiteralIndexExpr s(strides[i]);
    IndexExpr t1 = K - one;
    IndexExpr kdTerm = t1 * d + one; // (k - 1) * d + 1
    if (autoPad == "NOTSET") {
      IndexExpr p = pads[i] + pads[i + spatialRank]; // Sum both pads.
      IndexExpr t1 = I + p; // Compute floor/ceil((I + p - kdTerm) / s) + 1.
      IndexExpr t2 = t1 - kdTerm;
      IndexExpr O;
      if (ceilMode)
        O = t2.ceilDiv(s);
      else
        O = t2.floorDiv(s);
      O = O + one;
      // Set output dim, and pads already set, nothing more to do.
      outputDims.emplace_back(O);
    } else if (autoPad == "VALID") {
      IndexExpr t1 = I - kdTerm; // Compute ceil((I - kdTerm +1)/s).
      IndexExpr t2 = t1 + one;
      IndexExpr O = t2.ceilDiv(s);
      // Set output dim, and pads already set to zero, nothing more to do.
      outputDims.emplace_back(O);
    } else if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER") {
      // Compute output as O = ceil(I/s).
      IndexExpr O = I.ceilDiv(s);
      outputDims.emplace_back(O);
      // Compute sum of pads padSum = (O -1)*s + kdTerm - I.
      IndexExpr t1 = O - one;
      IndexExpr t2 = t1 * s + kdTerm;
      IndexExpr t3 = t2 - I;
      IndexExpr padSum = IndexExpr::max(t3, zero);
      // Single pad value is padSump / 2.
      IndexExpr p = padSum.floorDiv(2);
      // Increment is 1 when pp % 2 != 0
      IndexExpr test = (padSum % 2) != zero;
      IndexExpr inc = IndexExpr::select(test, one, zero);
      // Increment 1st value for SAME_LOWER and 2nd for SAME_UPPER.
      if (autoPad == "SAME_UPPER") {
        pads[i] = p;
        pads[i + spatialRank] = p + inc;
      } else { // SAME_LOWER.
        pads[i] = p + inc;
        pads[i + spatialRank] = p;
      }
    }
  }

#if DEBUG
  if (outputDims.size() == 4) {
    cerr << "2d conv const params";
    if (outputDims[0].isLiteral())
      cerr << ", N " << outputDims[0].getLiteral();
    if (outputDims[1].isLiteral())
      cerr << ", CO " << outputDims[1].getLiteral();
    if (outputDims[2].isLiteral())
      cerr << ", WO " << outputDims[2].getLiteral();
    if (outputDims[3].isLiteral())
      cerr << ", HO " << outputDims[3].getLiteral();
    if (pads[0].isLiteral())
      cerr << ", ph begin " << pads[0].getLiteral();
    if (pads[2].isLiteral())
      cerr << ", ph end " << pads[2].getLiteral();
    if (pads[1].isLiteral())
      cerr << ", pw begin " << pads[1].getLiteral();
    if (pads[3].isLiteral())
      cerr << ", pw end " << pads[3].getLiteral();
    cerr << endl;
  }
#endif

  // Set type for the first output.
  ONNXOpShapeHelper<OP_TYPE>::dimsForOutput(0) = outputDims;
  return success();
}

//===----------------------------------------------------------------------===//
// ONNX Conv Op Shape Helper
//===----------------------------------------------------------------------===//

ONNXConvOpShapeHelper::ONNXConvOpShapeHelper(ONNXConvOp *newOp)
    : ONNXGenericPoolShapeHelper<ONNXConvOp, ONNXConvOpAdaptor>(
          newOp, true /*hasFilter*/, false /*hasCeil*/) {}

ONNXConvOpShapeHelper::ONNXConvOpShapeHelper(ONNXConvOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXGenericPoolShapeHelper<ONNXConvOp, ONNXConvOpAdaptor>(newOp,
          true /*hasFilter*/, false /*hasCeil*/, rewriter, fGetDenseVal,
          fLoadVal) {}

LogicalResult ONNXConvOpShapeHelper::computeShape(
    ONNXConvOpAdaptor operandAdaptor) {
  return ONNXGenericPoolShapeHelper<ONNXConvOp,
      ONNXConvOpAdaptor>::computeShape(operandAdaptor, operandAdaptor.W(),
      op->kernel_shape(), op->pads(), op->strides(), op->dilations());
}

//===----------------------------------------------------------------------===//
// ONNX Max Pool Single Out Ops Shape Helper
//===----------------------------------------------------------------------===//

ONNXMaxPoolSingleOutOpShapeHelper::ONNXMaxPoolSingleOutOpShapeHelper(
    ONNXMaxPoolSingleOutOp *newOp)
    : ONNXGenericPoolShapeHelper<ONNXMaxPoolSingleOutOp,
          ONNXMaxPoolSingleOutOpAdaptor>(
          newOp, false /*hasFilter*/, newOp->ceil_mode()) {}

ONNXMaxPoolSingleOutOpShapeHelper::ONNXMaxPoolSingleOutOpShapeHelper(
    ONNXMaxPoolSingleOutOp *newOp, OpBuilder *rewriter,
    ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXGenericPoolShapeHelper<ONNXMaxPoolSingleOutOp,
          ONNXMaxPoolSingleOutOpAdaptor>(newOp, false /*hasFilter*/,
          newOp->ceil_mode(), rewriter, fGetDenseVal, fLoadVal) {}

LogicalResult ONNXMaxPoolSingleOutOpShapeHelper::computeShape(
    ONNXMaxPoolSingleOutOpAdaptor operandAdaptor) {
  return ONNXGenericPoolShapeHelper<ONNXMaxPoolSingleOutOp,
      ONNXMaxPoolSingleOutOpAdaptor>::computeShape(operandAdaptor, nullptr,
      op->kernel_shape(), op->pads(), op->strides(), op->dilations());
}

//===----------------------------------------------------------------------===//
// ONNX Max Pool Single Out Ops Shape Helper
//===----------------------------------------------------------------------===//

ONNXAveragePoolOpShapeHelper::ONNXAveragePoolOpShapeHelper(
    ONNXAveragePoolOp *newOp)
    : ONNXGenericPoolShapeHelper<ONNXAveragePoolOp, ONNXAveragePoolOpAdaptor>(
          newOp, false /*hasFilter*/, newOp->ceil_mode()) {}

ONNXAveragePoolOpShapeHelper::ONNXAveragePoolOpShapeHelper(
    ONNXAveragePoolOp *newOp, OpBuilder *rewriter,
    ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXGenericPoolShapeHelper<ONNXAveragePoolOp, ONNXAveragePoolOpAdaptor>(
          newOp, false /*hasFilter*/, newOp->ceil_mode(), rewriter,
          fGetDenseVal, fLoadVal) {}

LogicalResult ONNXAveragePoolOpShapeHelper::computeShape(
    ONNXAveragePoolOpAdaptor operandAdaptor) {
  return ONNXGenericPoolShapeHelper<ONNXAveragePoolOp,
      ONNXAveragePoolOpAdaptor>::computeShape(operandAdaptor, nullptr,
      op->kernel_shape(), op->pads(), op->strides(), None);
}

//===----------------------------------------------------------------------===//
// ONNX Reshape Op Shape Helper
//===----------------------------------------------------------------------===//

ONNXReshapeOpShapeHelper::ONNXReshapeOpShapeHelper(ONNXReshapeOp *newOp)
    : ONNXOpShapeHelper<ONNXReshapeOp>(
          newOp, newOp->getOperation()->getNumResults()) {}

ONNXReshapeOpShapeHelper::ONNXReshapeOpShapeHelper(ONNXReshapeOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXOpShapeHelper<ONNXReshapeOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal) {}

LogicalResult ONNXReshapeOpShapeHelper::computeShape(
    ONNXReshapeOpAdaptor operandAdaptor) {
  // Shape inference indicated by passing a null rewriter pointer.
  // Output dims of results.
  DimsExpr outputDims;

  // Get info about input data operand.
  Value data = operandAdaptor.data();
  MemRefBoundsIndexCapture dataBounds(data);
  int64_t dataRank = data.getType().cast<ShapedType>().getShape().size();

  // Get info about shape operand.
  Value shape = operandAdaptor.shape();
  ArrayValueIndexCapture shapeCapture(shape, fGetDenseVal, fLoadVal);
  int64_t outputRank = shape.getType().cast<ShapedType>().getShape()[0];
  assert(outputRank != -1 && "Shape tensor must have constant shape");

  // Initialize context and results.
  outputDims.resize(outputRank);

  // Shape values can be 0, -1, or N (N > 0).
  //   - 0: the output dim is setting to the input dim at the same index.
  //   Thus, it must happen at the index < dataRank.
  //   - -1: the output dim is calculated from the other output dims. No more
  //   than one dim in the output has value -1.

  // Compute the total number of elements using the input data operand.
  IndexExpr numOfElements = LiteralIndexExpr(1);
  for (unsigned i = 0; i < dataRank; ++i)
    numOfElements = numOfElements * dataBounds.getDim(i);

  // Compute the total number of elements from the shape values.
  IndexExpr numOfElementsFromShape = LiteralIndexExpr(1);
  for (unsigned i = 0; i < outputRank; ++i) {
    SymbolIndexExpr dimShape(shapeCapture.getSymbol(i));
    if (dimShape.isUndefined())
      return op->emitError("shape input parameter could not be processed");
    IndexExpr dim;
    if (i < dataRank)
      // dimShape == 0: use dim from the input.
      dim = dimShape.selectOrSelf(dimShape == 0, dataBounds.getDim(i));
    else
      dim = dimShape;

    // Just store the dim as it is. Real value for -1 will be computed later.
    outputDims[i] = dim;

    // dimShape == -1: use 1 to compute the number of elements to avoid
    // negative value.
    dim = dim.selectOrSelf(dim == -1, LiteralIndexExpr(1));
    numOfElementsFromShape = numOfElementsFromShape * dim;
  }

  // All the output dims except the one with -1 are computed. Thus, only
  // update the dim with -1 here.
  for (unsigned i = 0; i < outputRank; ++i)
    outputDims[i] = outputDims[i].selectOrSelf(
        outputDims[i] == -1, numOfElements.floorDiv(numOfElementsFromShape));

  // Save the final result.
  dimsForOutput(0) = outputDims;

  return success();
}

//===----------------------------------------------------------------------===//
// ONNX Squeeze Op Shape Helper
//===----------------------------------------------------------------------===//

template <typename ShapeHelper, typename OperandAdaptor>
LogicalResult ONNXSqueezeOpShapeHelperCommon(ShapeHelper *shapeHelper,
    OperandAdaptor operandAdaptor, ArrayRef<IndexExpr> indexExprArray) {
  // Output dims of results.
  DimsExpr outputDims;

  // Get info about input data operand.
  Value data = operandAdaptor.data();
  MemRefBoundsIndexCapture dataBounds(data);
  int64_t dataRank = data.getType().cast<ShapedType>().getShape().size();

  // Get axis values. They are expected to be normalized before so that there
  // is no negative values.
  SmallVector<int64_t, 4> axes;
  for (auto axisAttr : indexExprArray) {
    int64_t axis = axisAttr.getLiteral();
    assert(axis >= 0 && "Invalid axis");
    axes.emplace_back(axis);
  }

  for (int i = 0; i < dataRank; ++i)
    if (std::find(axes.begin(), axes.end(), i) == axes.end())
      outputDims.emplace_back(dataBounds.getDim(i));

  // Save the final result.
  shapeHelper->dimsForOutput(0) = outputDims;

  return success();
}

ONNXSqueezeOpShapeHelper::ONNXSqueezeOpShapeHelper(ONNXSqueezeOp *newOp)
    : ONNXOpShapeHelper<ONNXSqueezeOp>(
          newOp, newOp->getOperation()->getNumResults()) {}

ONNXSqueezeOpShapeHelper::ONNXSqueezeOpShapeHelper(ONNXSqueezeOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXOpShapeHelper<ONNXSqueezeOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal) {}

LogicalResult ONNXSqueezeOpShapeHelper::computeShape(
    ONNXSqueezeOpAdaptor operandAdaptor) {
  auto axes = op->axes();
  SmallVector<IndexExpr, 4> indexExprArray;
  if (auto axesConstOp = getONNXConstantOp(axes)) {
    ArrayValueIndexCapture axesCapture(axes, fGetDenseVal, fLoadVal);
    axesCapture.getSymbolList(indexExprArray);
  } else if (!axes.getType().template isa<NoneType>()) {
    llvm_unreachable("dynamic axes not yet supported");
  }

  return ONNXSqueezeOpShapeHelperCommon(this, operandAdaptor, indexExprArray);
}

ONNXSqueezeV11OpShapeHelper::ONNXSqueezeV11OpShapeHelper(
    ONNXSqueezeV11Op *newOp)
    : ONNXOpShapeHelper<ONNXSqueezeV11Op>(
          newOp, newOp->getOperation()->getNumResults()) {}

ONNXSqueezeV11OpShapeHelper::ONNXSqueezeV11OpShapeHelper(
    ONNXSqueezeV11Op *newOp, OpBuilder *rewriter,
    ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXOpShapeHelper<ONNXSqueezeV11Op>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal) {}

LogicalResult ONNXSqueezeV11OpShapeHelper::computeShape(
    ONNXSqueezeV11OpAdaptor operandAdaptor) {
  auto axesAttr = op->axes();
  SmallVector<IndexExpr, 4> indexExprArray;
  if (axesAttr.hasValue()) {
    ArrayAttributeIndexCapture axesCapture(axesAttr.getValue());
    auto axesRank = axesCapture.size();
    for (unsigned i = 0; i < axesRank; ++i) {
      indexExprArray.emplace_back(axesCapture.getLiteral(i));
    }
  }
  return ONNXSqueezeOpShapeHelperCommon(this, operandAdaptor, indexExprArray);
}

//===----------------------------------------------------------------------===//
// ONNX Unsqueeze Op Shape Helper
//===----------------------------------------------------------------------===//

template <typename ShapeHelper, typename OperandAdaptor>
LogicalResult ONNXUnsqueezeOpShapeHelperCommon(ShapeHelper *shapeHelper,
    OperandAdaptor operandAdaptor, ArrayRef<IndexExpr> indexExprArray) {
  // Output dims of results.
  DimsExpr outputDims;

  // Get info about input data operand.
  Value data = operandAdaptor.data();
  MemRefBoundsIndexCapture dataBounds(data);
  int64_t dataRank = data.getType().cast<ShapedType>().getShape().size();

  // Get axis values. They are expected to be normalized before so that there
  // is no negative values.
  SmallVector<int64_t, 4> axes;
  for (auto axisAttr : indexExprArray) {
    int64_t axis = axisAttr.getLiteral();
    assert(axis >= 0 && "Invalid axis");
    axes.emplace_back(axis);
  }

  int64_t outRank = dataRank + axes.size();
  for (int i = 0, j = 0; i < outRank || j < dataRank; ++i)
    if (std::find(axes.begin(), axes.end(), i) != axes.end())
      outputDims.emplace_back(LiteralIndexExpr(1));
    else
      outputDims.emplace_back(dataBounds.getDim(j++));

  // Save the final result.
  shapeHelper->dimsForOutput(0) = outputDims;

  return success();
}

ONNXUnsqueezeOpShapeHelper::ONNXUnsqueezeOpShapeHelper(ONNXUnsqueezeOp *newOp)
    : ONNXOpShapeHelper<ONNXUnsqueezeOp>(
          newOp, newOp->getOperation()->getNumResults()) {}

ONNXUnsqueezeOpShapeHelper::ONNXUnsqueezeOpShapeHelper(ONNXUnsqueezeOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXOpShapeHelper<ONNXUnsqueezeOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal) {}

LogicalResult ONNXUnsqueezeOpShapeHelper::computeShape(
    ONNXUnsqueezeOpAdaptor operandAdaptor) {
  auto axes = op->axes();
  SmallVector<IndexExpr, 4> indexExprArray;
  if (auto axesConstOp = getONNXConstantOp(axes)) {
    ArrayValueIndexCapture axesCapture(axes, fGetDenseVal, fLoadVal);
    axesCapture.getSymbolList(indexExprArray);
  } else if (!axes.getType().template isa<NoneType>()) {
    llvm_unreachable("dynamic axes not yet supported");
  }

  return ONNXUnsqueezeOpShapeHelperCommon(this, operandAdaptor, indexExprArray);
}

ONNXUnsqueezeV11OpShapeHelper::ONNXUnsqueezeV11OpShapeHelper(
    ONNXUnsqueezeV11Op *newOp)
    : ONNXOpShapeHelper<ONNXUnsqueezeV11Op>(
          newOp, newOp->getOperation()->getNumResults()) {}

ONNXUnsqueezeV11OpShapeHelper::ONNXUnsqueezeV11OpShapeHelper(
    ONNXUnsqueezeV11Op *newOp, OpBuilder *rewriter,
    ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXOpShapeHelper<ONNXUnsqueezeV11Op>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal) {}

LogicalResult ONNXUnsqueezeV11OpShapeHelper::computeShape(
    ONNXUnsqueezeV11OpAdaptor operandAdaptor) {
  auto axesAttr = op->axes();
  SmallVector<IndexExpr, 4> indexExprArray;
  ArrayAttributeIndexCapture axesCapture(axesAttr);
  auto axesRank = axesCapture.size();
  for (unsigned i = 0; i < axesRank; ++i) {
    indexExprArray.emplace_back(axesCapture.getLiteral(i));
  }
  return ONNXUnsqueezeOpShapeHelperCommon(this, operandAdaptor, indexExprArray);
}

//===----------------------------------------------------------------------===//
// ONNX Shape Op Shape Helper
//===----------------------------------------------------------------------===//

ONNXShapeOpShapeHelper::ONNXShapeOpShapeHelper(
    ONNXShapeOp *newOp, IndexExprScope *inScope)
    : ONNXOpShapeHelper<ONNXShapeOp>(
          newOp, newOp->getOperation()->getNumResults(), inScope) {}

ONNXShapeOpShapeHelper::ONNXShapeOpShapeHelper(ONNXShapeOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal, IndexExprScope *inScope)
    : ONNXOpShapeHelper<ONNXShapeOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal, inScope) {}

LogicalResult ONNXShapeOpShapeHelper::computeShape(
    ONNXShapeOpAdaptor operandAdaptor) {

  // Get info about input data operand.
  Value data = operandAdaptor.data();
  MemRefBoundsIndexCapture dataBounds(data);
  int64_t dataRank = dataBounds.getRank();

  // To be initalized from op (opset > 13)
  int64_t start = 0;
  int64_t end = dataRank; // Default value if option not defined.

  // Handle negative values.
  if (start < 0)
    start = start + dataRank;
  if (end < 0)
    end = end + dataRank;
  if (start < 0 || start > dataRank)
    return op->emitError("start value is out of bound");
  if (end < 0 || end > dataRank)
    return op->emitError("end value is out of bound");

  // Save actual values in selected data
  for (int64_t i = start; i < end; ++i)
    selectedData.emplace_back(dataBounds.getDim(i));
  // Output is the actual number of values (1D)
  dimsForOutput(0).emplace_back(LiteralIndexExpr(selectedData.size()));
  return success();
}

//===----------------------------------------------------------------------===//
// ONNX Pad Op Shape Helper
//===----------------------------------------------------------------------===//

ONNXPadOpShapeHelper::ONNXPadOpShapeHelper(ONNXPadOp *newOp)
    : ONNXOpShapeHelper<ONNXPadOp>(
          newOp, newOp->getOperation()->getNumResults()),
      pads() {}

ONNXPadOpShapeHelper::ONNXPadOpShapeHelper(ONNXPadOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXOpShapeHelper<ONNXPadOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal),
      pads() {}

LogicalResult ONNXPadOpShapeHelper::computeShape(
    ONNXPadOpAdaptor operandAdaptor) {
  // Shape inference indicated by passing a null rewriter pointer.
  // Output dims of results.
  DimsExpr outputDims;

  // Get info about input data operand.
  MemRefBoundsIndexCapture dataBounds(operandAdaptor.data());
  uint64_t dataRank = dataBounds.getRank();

  // Initialize context and results (pads & output)
  pads.resize(2 * dataRank); // pads two sides of each axis.
  outputDims.resize(dataRank);

  // `pads` format is : [x1_begin, x2_begin,...,x1_end, x2_end,...],
  // where
  // - xi_begin: the number of pad values added at the beginning of axis `i`
  // - xi_end: the number of pad values added at the end of axis `i`.
  ArrayValueIndexCapture padsCapture(
      operandAdaptor.pads(), fGetDenseVal, fLoadVal);

  // Calculate output dimension sizes.
  for (uint64_t i = 0; i < dataRank; i++) {
    // Get begin/end pads.
    SymbolIndexExpr padBegin(padsCapture.getSymbol(i));
    SymbolIndexExpr padEnd(padsCapture.getSymbol(i + dataRank));
    if (padBegin.isUndefined() || padEnd.isUndefined())
      return op->emitError("pad parameter could not be processed");
    // Get input dim.
    DimIndexExpr dimInput(dataBounds.getDim(i));

    // Calculation for output size.
    IndexExpr dimOutputFinal = padBegin + dimInput + padEnd;

    // Save results.
    pads[i] = padBegin;
    pads[i + dataRank] = padEnd;
    outputDims[i] = dimOutputFinal;
  }

  // Save the final result.
  dimsForOutput(0) = outputDims;

  return success();
}

//===----------------------------------------------------------------------===//
// ONNX Expand Op Shape Helper
//===----------------------------------------------------------------------===//

ONNXExpandOpShapeHelper::ONNXExpandOpShapeHelper(ONNXExpandOp *newOp)
    : ONNXOpBroadcastedShapeHelper<ONNXExpandOp>(newOp), expandOp(newOp) {}

ONNXExpandOpShapeHelper::ONNXExpandOpShapeHelper(ONNXExpandOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXOpBroadcastedShapeHelper<ONNXExpandOp>(
          newOp, rewriter, fGetDenseVal, fLoadVal),
      expandOp(newOp) {}

LogicalResult ONNXExpandOpShapeHelper::computeShape(
    ONNXExpandOpAdaptor operandAdaptor) {
  // Get info about input operands.
  Value input = operandAdaptor.input();
  Value shape = operandAdaptor.shape();
  Operation *shapeDefOp = shape.getDefiningOp();

  ShapedType shapeType = shape.getType().dyn_cast_or_null<ShapedType>();
  if (!shapeType)
    return op->emitError("expected shape parameter to be defined");
  if (shapeType.getShape()[0] == -1)
    return op->emitError("expected size of shape parameter to be defined");
  if (mlir::ONNXShapeOp shapeOp =
          dyn_cast_or_null<mlir::ONNXShapeOp>(shapeDefOp)) {
    assert(shapeOp.data().getType().isa<ShapedType>() && "expected");
    // Consider a first case where the expand.shape is produced by a shape op.
    // Infer its shape and use it as the requested shape.
    // Compute the output of the shape operation. We have to use its shape
    // helper as we need to connect to the actual expressions used to compute
    // it, not just a shape, in presence of runtime dimensions.

    // Use the full constructor as this is called from shape helper which may
    // be used in either shape inference or lowering to ONNX context. We also
    // pass here the scope of the ExpandOp shape helper so that the
    // computations performed in the ShapeOp shape helper can be used in the
    // context of the ExpandOp.
    ONNXShapeOpShapeHelper shapeOpShapeHelper(
        &shapeOp, scope->getRewriterPtr(), fGetDenseVal, fLoadVal, scope);
    ONNXShapeOpAdaptor shapeOpOperandAdaptor(shapeOp);
    if (failed(shapeOpShapeHelper.computeShape(shapeOpOperandAdaptor)))
      return op->emitError("failed to get shape op shape");
    // Now that we have the shape's actual computation in
    if (failed(ONNXOpBroadcastedShapeHelper::computeShape(
            {input}, shapeOpShapeHelper.selectedData)))
      return op->emitError("failed to broadcast");

  } else {
    assert(shape.getType().isa<ShapedType>());
    SmallVector<IndexExpr, 4> constVals;
    ArrayValueIndexCapture arrayCapture(shape, fGetDenseVal, fLoadVal);
    if (!arrayCapture.getSymbolList(constVals)) {
      return op->emitError(
          "Shape argument of Expand is the output of an unexpected "
          "operation. Supported operations are: onnx.Constant and "
          "onnx.Shape");
    }
    if (failed(ONNXOpBroadcastedShapeHelper::computeShape({input}, constVals)))
      return op->emitError("failed to broadcast");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ONNX Compress Op Shape Helper
//===----------------------------------------------------------------------===//

ONNXCompressOpShapeHelper::ONNXCompressOpShapeHelper(ONNXCompressOp *newOp)
    : ONNXOpShapeHelper<ONNXCompressOp>(
          newOp, newOp->getOperation()->getNumResults()) {}

ONNXCompressOpShapeHelper::ONNXCompressOpShapeHelper(ONNXCompressOp *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal)
    : ONNXOpShapeHelper<ONNXCompressOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, fGetDenseVal,
          fLoadVal) {}

LogicalResult ONNXCompressOpShapeHelper::computeShape(
    ONNXCompressOpAdaptor operandAdaptor) {
  // Check that input and condition are ranked.
  Value input = operandAdaptor.input();
  ShapedType inputType = input.getType().dyn_cast_or_null<ShapedType>();
  assert(inputType && inputType.hasRank() &&
         "Input should have a known shape and rank");
  int64_t inputRank = inputType.getRank();
  Value cond = operandAdaptor.condition();
  ShapedType condType = cond.getType().dyn_cast_or_null<ShapedType>();
  assert(condType && condType.hasRank() &&
         "Condition should have a known and rank");
  // Get the dimension derived from the condition. Assume in shape helper
  // that it is only going to be a question mark. ONNX to Krnl lowering will
  // compute the actual value.
  // TODO: if cond is constant, the compute the actual value.
  printf("hi alex shape 1\n");
  IndexExpr dynDim;
  if (scope->isShapeInferencePass())
    dynDim = QuestionmarkIndexExpr(); // Value for runtime dim.
  else
    dynDim = LiteralIndexExpr(-1); // Dummy value to be replaced in lowering.
  printf("hi alex shape 1.1\n");
  // Get axis. Value -1 signify axis was not specified. Verifier already checked
  // that the axis, if given, is in range.
  axis = -1;
  if (op->axis().hasValue()) {
    axis = op->axis().getValue();
    if (axis < 0)
      axis += inputRank;
  }
  // Compute dims for output.
  DimsExpr outputDims;
  printf("hi alex shape 2\n");
  if (axis == -1) {
    // Reduced to a single dimensional array, of dynamic size.
    outputDims.emplace_back(dynDim);
  } else {
    // Has same dimensionality as input, with axis dimension being the dynamic
    // size.
    MemRefBoundsIndexCapture inputBounds(input);
    inputBounds.getDimList(outputDims);
    outputDims[axis] = dynDim;
  }
  printf("hi alex shape 3\n");
  dimsForOutput(0) = outputDims;
  return success();
}

// Keep template instantiation at the end of the file.

//===----------------------------------------------------------------------===//
// ONNX Shape Helper template instantiation
//===----------------------------------------------------------------------===//

template struct ONNXOpBroadcastedShapeHelper<Operation>;
template struct ONNXOpBroadcastedShapeHelper<ONNXExpandOp>;

// Keep template instantiation at the end of the file.
