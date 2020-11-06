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
// ONNX Gemm Op Shape Helper
//===----------------------------------------------------------------------===//

ONNXGemmOpShapeHelper::ONNXGemmOpShapeHelper(
    ONNXGemmOp *newOp, ConversionPatternRewriter *rewriter)
    : ONNXOpShapeHelper<ONNXGemmOp>(newOp, rewriter), aDims(), bDims(),
      cDims() {}

LogicalResult ONNXGemmOpShapeHelper::Compute(ONNXGemmOpAdaptor operandAdaptor) {
  // Shape inference indicated by passing a null rewriter pointer.
  Operation *genericOp = reinterpret_cast<Operation *>(op);
  // Get info.
  Value A = operandAdaptor.A();
  Value B = operandAdaptor.B();
  Value C = operandAdaptor.C();
  bool hasBias = !C.getType().isa<NoneType>();

  // Test ranks.
  if (A.getType().cast<ShapedType>().getShape().size() != 2)
    return op->emitError("Gemm with A should be a 2D tensor");
  if (B.getType().cast<ShapedType>().getShape().size() != 2)
    return op->emitError("Gemm with B should be a 2D tensor");
  int cRank = 0;
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
