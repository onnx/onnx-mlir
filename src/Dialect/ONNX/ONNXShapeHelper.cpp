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
    : op(newOp), context(rewriter, newOp->getLoc()), outputDims() {}

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

  // Get info about input data operand.
  Value data = operandAdaptor.data();
  int64_t dataRank = data.getType().cast<ShapedType>().getShape().size();

  // Get each of the axes, and save the literal values in axesIntLit.
  SmallVector<int64_t, 2> axesIntLit;
  Value axes = operandAdaptor.axes();
  if (axes.getType().isa<NoneType>()) {
    // If `axes` are omitted, they are set to `[0, ..., ndim-1]`."
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
  for (uint64_t i = 0; i < sliceRank; i++) {
    // i is index in start/step/end/output
    // ii is logical index in mem/loop bounds
    int ii = axesIntLit[i];
    // Get start, end, step, and dim index expressions.
    // Get start.
    IndexExpr startInput = context.createSymbolIndexFromArrayAtIndex(
        genericOp, operandAdaptor.starts(), i);
    if (startInput.isUndefined())
      return op->emitError("start input parameter could not be processed");
    startInput.debugPrint("start input");
    // Get end.
    IndexExpr endInput = context.createSymbolIndexFromArrayAtIndex(
        genericOp, operandAdaptor.ends(), i);
    if (endInput.isUndefined())
      return op->emitError("end input parameter could not be processed");
    endInput.debugPrint("end input");
    // Get step.
    IndexExpr stepInput = context.createSymbolIndexFromArrayAtIndex(
        genericOp, operandAdaptor.steps(), i, 1);
    if (stepInput.isUndefined())
      return op->emitError("step input parameter could not be processed");
    if (stepInput.isLiteral() && stepInput.getLiteral() == 0)
      return op->emitError("step input parameter cannot be zero");
    stepInput.debugPrint("step input");
    // Get dim.
    IndexExpr dimInput = context.createDimIndexFromShapedType(data, ii);
    dimInput.debugPrint("dim input");

    // Now proceed with the computations for start/end/dim.
    // Calculation for start: start < 0 ? start + dim : start.
    IndexExpr startPos =
        IndexExpr::select(startInput < 0, startInput + dimInput, startInput);
    // Step < 0: clamp(0, start, dim -1) else clamp(0, start, dim)
    IndexExpr neg = startPos.clamp(0, dimInput - 1);
    IndexExpr pos = startPos.clamp(0, dimInput);
    IndexExpr startFinal = IndexExpr::select(stepInput < 0, neg, pos);
    startFinal.debugPrint("start final");

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
    endFinal.debugPrint("end final");

    // Calculation for output size.
    IndexExpr dimOutputFinal = (endFinal - startFinal).ceilDiv(stepInput);
    // should use a max
    dimOutputFinal = dimOutputFinal.selectOrSelf(dimOutputFinal < 0, 0);
    dimOutputFinal.debugPrint("output dim final");

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
      starts[i] = context.createLiteralIndex(0);
      steps[i] = context.createLiteralIndex(1);
      IndexExpr dimInput = context.createDimIndexFromShapedType(data, i);
      ends[i] = dimInput;
      outputDims[i] = dimInput;
    }
#if 1
    starts[i].debugPrint("New Dim\n  start");
    ends[i].debugPrint("  end");
    steps[i].debugPrint("  step");
    outputDims[i].debugPrint("  output dim");
#endif
  }
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
  outputDims.resize(inputRank);
  for (auto i = 0; i < inputRank; i++) {
    IndexExpr dimInput = context.createDimIndexFromShapedType(input, i);
    IndexExpr repeatsValue =
        context.createSymbolIndexFromArrayAtIndex(genericOp, repeats, i);
    IndexExpr dimOutput = dimInput * repeatsValue;
    outputDims[i] = dimOutput;
  }
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
  Operation *genericOp = reinterpret_cast<Operation *>(op);
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
  if (op->transA() == 0) {
    aDims.emplace_back(context.createDimIndexFromShapedType(A, 0));
    aDims.emplace_back(context.createDimIndexFromShapedType(A, 1));
  } else {
    aDims.emplace_back(context.createDimIndexFromShapedType(A, 1));
    aDims.emplace_back(context.createDimIndexFromShapedType(A, 0));
  }
  aDims[0].debugPrint("a0");
  aDims[1].debugPrint("a1");
  // Scan dimensions of B with/without transpose.
  if (op->transB() == 0) {
    bDims.emplace_back(context.createDimIndexFromShapedType(B, 0));
    bDims.emplace_back(context.createDimIndexFromShapedType(B, 1));
  } else {
    bDims.emplace_back(context.createDimIndexFromShapedType(B, 1));
    bDims.emplace_back(context.createDimIndexFromShapedType(B, 0));
  }
  bDims[0].debugPrint("b0");
  bDims[1].debugPrint("b1");
  // Set output dims of result, creating a copy of it to be safe.
  outputDims.emplace_back(context.createIndex(aDims[0]));
  outputDims.emplace_back(context.createIndex(bDims[1]));
  outputDims[0].debugPrint("out0");
  outputDims[1].debugPrint("out1");
  // Bias C can be a (unidirectional) broadcast.
  if (hasBias) {
    if (cRank == 0) {
      // Broadcast for scalar: both dims are 1.
      cDims.emplace_back(context.createLiteralIndex(1));
      cDims.emplace_back(context.createLiteralIndex(1));
    } else if (cRank == 1) {
      // First dim is the one padded.
      cDims.emplace_back(context.createLiteralIndex(1));
      cDims.emplace_back(context.createDimIndexFromShapedType(C, 0));
    } else {
      assert(cRank == 2 && "illegal path");
      cDims.emplace_back(context.createDimIndexFromShapedType(C, 0));
      cDims.emplace_back(context.createDimIndexFromShapedType(C, 1));
    }
    cDims[0].debugPrint("c0");
    cDims[1].debugPrint("c1");
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
  Operation *genericOp = reinterpret_cast<Operation *>(op);
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
  // Add the dims of A. All of the aDim[0]...adim[arank-1] are in the rightmost
  // positions, prepended by 1s to fit the paddedRankSize.
  // (1,1,1... 1, aDim[0]...adim[aRank-1])
  IndexExpr one = context.createLiteralIndex(1);
  int aOffset = paddedRank - aRank;
  for (int i = 0; i < aOffset; ++i) {
    aDims[i] = one;
    aPadDims[i] = true;
  }
  for (int i = 0; i < aRank; ++i) {
    aDims[i + aOffset] = context.createDimIndexFromShapedType(A, i);
    aPadDims[i + aOffset] = false; // Pad false evein if dim is sized 1.
  }
  // for B: two cases. If bRank = 1, we pad the rightmost position. Namely we
  // get (1...,1, bDim[0], 1). We use one padding credit for the rightmost
  // position. Otherwise, when bRank>1, we only padd the leading positions.
  // Namely we get (1,1,1...,1, bDim[0],.... bDim[bRank-1])
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
    bDims[i + bOffset] = context.createDimIndexFromShapedType(B, i);
    bPadDims[i + bOffset] = false; // Pad false evein if dim is sized 1.
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
  return success();
}
