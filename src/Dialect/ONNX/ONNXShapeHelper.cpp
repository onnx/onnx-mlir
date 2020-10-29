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
// ONNX Helper for Shape inference
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ONNX Helper for Slice
//===----------------------------------------------------------------------===//


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
  auto dataType = data.getType().cast<ShapedType>();
  auto elementType = dataType.getElementType();
  auto dataShape = dataType.getShape();
  int64_t dataRank = dataShape.size();

  // Get each of the axes, and save the litteral values in axesIntLit.
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
    IndexExpr startInput, endInput, stepInput, dimInput, dimMinOneInput;
    // Get start.
    startInput = context.CreateSymbolIndexFromArrayAtIndex(
        genericOp, operandAdaptor.starts(), i);
    if (startInput.IsUndefined())
      return op->emitError("start input parameter could not be processed");
    startInput.DebugPrint("start input");
    // Get end.
    endInput = context.CreateSymbolIndexFromArrayAtIndex(
        genericOp, operandAdaptor.ends(), i);
    if (endInput.IsUndefined())
      return op->emitError("end input parameter could not be processed");
    endInput.DebugPrint("end input");
    // Get step.
    stepInput = context.CreateSymbolIndexFromArrayAtIndex(
        genericOp, operandAdaptor.steps(), i, 1);
    if (stepInput.IsUndefined())
      return op->emitError("step input parameter could not be processed");
    if (stepInput.IsLiteral() && stepInput.GetLiteral() == 0)
      return op->emitError("step input parameter cannot be zero");
    stepInput.DebugPrint("step input");
    // Get dim.
    dimInput = context.CreateDimIndexFromMemref(data, dataShape, ii);
    dimInput.DebugPrint("dim input");

    // Now proceed with the computations for start/end/dim.
    // Calculation for start: start < 0 ? start + dim : start.
    IndexExpr startPlusDim, startPos, startFinal, neg, pos;
    startPlusDim.Add(startInput, dimInput);
    startPos.Select(
        startInput, CmpIPredicate::slt, 0, startPlusDim, startInput);
    // Step < 0: clamp(0, start, dim -1) else clamp(0, start, dim)
    dimMinOneInput.Sub(dimInput, 1);
    neg.Clamp(startPos, 0, dimMinOneInput);
    pos.Clamp(startPos, 0, dimInput);
    startFinal.Select(stepInput, CmpIPredicate::slt, 0, neg, pos);
    startFinal.DebugPrint("start final");

    // Calculation for end: end<0 -> end + dim else -> end;
    // special case end <= -inf -> -1;  end >= inf -> dim;
    int64_t negInf = std::numeric_limits<int32_t>::min();
    int64_t posInf = std::numeric_limits<int32_t>::max();
    IndexExpr endPlusDim, endPos, endFinal;
    endPlusDim.Add(endInput, dimInput);
    endPos.Select(endInput, CmpIPredicate::slt, 0, endPlusDim, endInput);
    endPos.AssignIf(endInput, CmpIPredicate::sle, negInf, -1);
    endPos.AssignIf(endInput, CmpIPredicate::sge, posInf, dimInput);
    // End: step<0: clamp(-1, end, dim); step>0 clamp(0, end, dim)
    neg.Clamp(endPos, -1, dimInput);
    pos.Clamp(endPos, 0, dimInput);
    endFinal.Select(stepInput, CmpIPredicate::slt, 0, neg, pos);
    endFinal.DebugPrint("end final");

    // Calculation for output size.
    IndexExpr dimOutputFinal;
    dimOutputFinal.Sub(endFinal, startFinal).CeilDivBy(stepInput);
    // should use a max
    dimOutputFinal.AssignIf(dimOutputFinal, CmpIPredicate::slt, 0, 0);
    dimOutputFinal.DebugPrint("output dim final");

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
    if (steps[i].IsUndefined()) {
      // have one unset, put the defaults (start was already at zero, so we are
      // fine).
      starts[i] = context.CreateLiteralIndex(0);
      steps[i] = context.CreateLiteralIndex(1);
      IndexExpr dimInput = context.CreateDimIndexFromMemref(data, dataShape, i);
      ends[i] = dimInput;
      outputDims[i] = dimInput;
    }
#if 1
    starts[i].DebugPrint("New Dim\n  start");
    ends[i].DebugPrint("  end");
    steps[i].DebugPrint("  step");
    outputDims[i].DebugPrint("  output dim");
#endif
  }
  return success();
}
