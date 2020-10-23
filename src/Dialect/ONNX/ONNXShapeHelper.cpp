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

size_t ArrayAttrSize(ArrayAttr a) { return a.size(); }

size_t ArrayAttrSize(Optional<ArrayAttr> a) { return a.getValue().size(); }

int64_t ArrayAttrIntVal(ArrayAttr a, int i) {
  return (a.getValue()[i]).cast<IntegerAttr>().getInt();
}

int64_t ArrayAttrIntVal(Optional<ArrayAttr> a, int i) {
  return (a.getValue().getValue()[i]).cast<IntegerAttr>().getInt();
}

// Returns the ConstantOp which defines an MLIR Value or null.
ONNXConstantOp getONNXConstantOp(Value value) {
  return dyn_cast_or_null<mlir::ONNXConstantOp>(value.getDefiningOp());
}

DenseElementsAttr getDenseElementAttributeFromValue(Value value) {
  auto definingOp = value.getDefiningOp();
  if (auto constantOp = dyn_cast_or_null<mlir::ONNXConstantOp>(definingOp))
    return constantOp.valueAttr().dyn_cast<DenseElementsAttr>();
  else if (auto globalOp = dyn_cast_or_null<mlir::KrnlGlobalOp>(definingOp))
    if (globalOp.value().hasValue())
      return globalOp.valueAttr().dyn_cast<DenseElementsAttr>();
  return nullptr;
}

bool getIntegerLiteralFromValue(Value value, int64_t &intLit) {
  // From lib/Dialect/LinAlg/Transform/Promotion.cpp
  if (auto constantOp = value.getDefiningOp<ConstantOp>()) {
    if (constantOp.getType().isa<IndexType>())
      intLit = constantOp.value().cast<IntegerAttr>().getInt();
    return true;
  }
  // Since ConsantIndexOp is a subclass of ConstantOp, not sure if this one is
  // useful.
  if (auto constantOp = value.getDefiningOp<ConstantIndexOp>()) {
    if (constantOp.getType().isa<IndexType>())
      intLit = constantOp.value().cast<IntegerAttr>().getInt();
    return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// ONNX Helper for Shape inference
//===----------------------------------------------------------------------===//

IndexExpr GetIndexExprFromArrayAt(
    IndexExprContainer &container, Operation *op, Value operand, uint64_t i) {
  if (auto attrArray = getDenseElementAttributeFromValue(operand)) {
    // We extracted an dense attribute from definition of operand.
    if (i >= attrArray.getType().getDimSize(0)) {
      printf("error 1\n");
      op->emitError("operand literal has wrong shape");
      return container.CreateUndefinedIndexExpr();
    }
    auto attrVal = attrArray.getValue(ArrayRef<uint64_t>({i}));
    int64_t attrInt = attrVal.cast<IntegerAttr>().getInt();
    return container.CreateLiteralIndexExpr(attrInt);
  }
  // We must read value from an array.
  if (container.IsShapeInferencePass()) {
    // Not a constant; don't add code.
    return container.CreateQuestionmarkIndexExpr();
  }
  // Emit code to rad array.
  Value indexVal = emitConstantOp(container.GetRewriter(),
      container.GetLocation(), container.GetRewriter().getIndexType(), i);
  SmallVector<Value, 1> memrefVal = {indexVal};
  Value loadVal = container.GetRewriter().create<AffineLoadOp>(
      container.GetLocation(), operand, memrefVal);
  return container.CreateSymbolIndexExpr(loadVal);
}

IndexExpr GetIndexExprFromArrayAt(IndexExprContainer &container, Operation *op,
    Value operand, uint64_t i, int64_t defaultIntLit) {
  // Check if we have an operand.
  if (operand.getType().isa<NoneType>()) {
    // Operand undefined, we use the default value.
    return container.CreateLiteralIndexExpr(defaultIntLit);
  }
  if (auto attrArray = getDenseElementAttributeFromValue(operand)) {
    // We extracted an dense attribute from definition of operand.
    if (i > attrArray.getType().getDimSize(0)) {
      // Not enought attributes for this index, return the default value.
      return container.CreateLiteralIndexExpr(defaultIntLit);
    }
    // We have enought attributes for this index, get the value.
    Attribute attrVal = attrArray.getValue(ArrayRef<uint64_t>({i}));
    int64_t attrInt = attrVal.cast<IntegerAttr>().getInt();
    return container.CreateLiteralIndexExpr(attrInt);
  }
  // Read the value from an array.
  if (container.IsShapeInferencePass()) {
    // Not a constant; don't add code.
    return container.CreateQuestionmarkIndexExpr();
  }
  // Emit the code to read array.
  Value indexVal = emitConstantOp(container.GetRewriter(),
      container.GetLocation(), container.GetRewriter().getIndexType(), i);
  SmallVector<Value, 1> memrefVal = {indexVal};
  Value loadVal = container.GetRewriter().create<AffineLoadOp>(
      container.GetLocation(), operand, memrefVal);
  return container.CreateSymbolIndexExpr(loadVal);
}

//===----------------------------------------------------------------------===//
// ONNX Helper for Slice
//===----------------------------------------------------------------------===//

LogicalResult HandleSliceOpParams(ONNXSliceOp *sliceOp,
    ONNXSliceOpAdaptor operandAdaptor, IndexExprContainer &container,
    SmallVectorImpl<IndexExpr> &startIndices,
    SmallVectorImpl<IndexExpr> &endIndices,
    SmallVectorImpl<IndexExpr> &stepIndices,
    SmallVectorImpl<IndexExpr> &outputDims) {
  // Shape inference indicated by passing a null rewriter pointer.
  Operation *op = reinterpret_cast<Operation *>(sliceOp);

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
        return sliceOp->emitError("Axes contains an out-of-bound index");
      axesIntLit.emplace_back(axis);
    }
  } else {
    return sliceOp->emitError("Axes must be known at compile time");
  }
  int sliceRank = axesIntLit.size();

  // Initialize container and results (start & output)
  startIndices.resize(dataRank);
  stepIndices.resize(dataRank);
  endIndices.resize(dataRank);
  outputDims.resize(dataRank);

  // SmallVector<uint64_t, 1> index1D(1, 0);
  for (uint64_t i = 0; i < sliceRank; i++) {
    // i is index in start/step/end/output
    // ii is logical index in mem/loop bounds
    int ii = axesIntLit[i];
    // Get start, end, step, and dim index expressions.
    IndexExpr startInput, endInput, stepInput, dimInput, dimMinOneInput;
    // Get start.
    startInput =
        GetIndexExprFromArrayAt(container, op, operandAdaptor.starts(), i);
    if (startInput.IsUndefined())
      return sliceOp->emitError("start input parameter could not be processed");
    startInput.DebugPrint("start input");
    // Get end.
    endInput = GetIndexExprFromArrayAt(container, op, operandAdaptor.ends(), i);
    if (endInput.IsUndefined())
      return sliceOp->emitError("end input parameter could not be processed");
    endInput.DebugPrint("end input");
    // Get step.
    stepInput =
        GetIndexExprFromArrayAt(container, op, operandAdaptor.steps(), i, 1);
    if (stepInput.IsUndefined())
      return sliceOp->emitError("step input parameter could not be processed");
    if (stepInput.IsLiteral() && stepInput.GetLiteral() == 0)
      return sliceOp->emitError("step input parameter cannot be zero");
    stepInput.DebugPrint("step input");
    // Get dim.
    dimInput = container.CreateDimIndexExpr(data, dataShape, ii);
    dimInput.DebugPrint("dim input");
    dimMinOneInput.Sub(dimInput, 1);

    // If in shape inference mode and we don't have the constant info, take
    // early break.
    if (container.IsShapeInferencePass() &&
        (startInput.IsQuestionmark() || endInput.IsQuestionmark() ||
            stepInput.IsQuestionmark() || dimInput.IsQuestionmark()))
      return failure();

    // Now proceed with the computations for start/end/dim.
    // Calculation for start: start < 0 ? start + dim : start.
    IndexExpr startPlusDim, startPos, startFinal, neg, pos;
    startPlusDim.Add(startInput, dimInput);
    startPos.Select(
        startInput, CmpIPredicate::slt, 0, startPlusDim, startInput);
    // Step < 0: clamp(0, start, dim -1) else clamp(0, start, dim)
    neg.Clamp(startPos, 0, dimMinOneInput);
    pos.Clamp(startPos, 0, dimInput);
    startFinal.Select(stepInput, CmpIPredicate::slt, 0, neg, pos);
    startPlusDim.DebugPrint("start plus dim input");
    startPos.DebugPrint("start pos");
    neg.DebugPrint("start clamp neg");
    pos.DebugPrint("start clamp pos");
    startFinal.DebugPrint("start final");

    // Calculation for end: end<0 -> end + dim else -> end;
    // special case end <= -inf -> -1;  end >= inf -> dim;
    int64_t negInf = std::numeric_limits<int32_t>::min();
    int64_t posInf = std::numeric_limits<int32_t>::max();
    IndexExpr endPlusDim, endPos, endFinal;
    endPlusDim.Add(endInput, dimInput);
    endPos.Select(endInput, CmpIPredicate::slt, 0, endPlusDim, endInput);
    endPos.Select(endInput, CmpIPredicate::sle, negInf, -1, endPos);
    endPos.Select(endInput, CmpIPredicate::sge, posInf, dimInput, endPos);

    // End: step<0: clamp(-1, end, dim); step>0 clamp(0, end, dim)
    neg.Clamp(endPos, -1, dimInput);
    pos.Clamp(endPos, 0, dimInput);
    endFinal.Select(stepInput, CmpIPredicate::slt, 0, neg, pos);
    endFinal.DebugPrint("end final");

    // Calculation for output size.
    IndexExpr dimOutputFinal;
    dimOutputFinal.Sub(endFinal, startFinal);
    dimOutputFinal.CeilDiv(dimOutputFinal, stepInput);
    // should use a max
    dimOutputFinal.Select(
        dimOutputFinal, CmpIPredicate::slt, 0, 0, dimOutputFinal);
    dimOutputFinal.DebugPrint("output dim final");

    if (container.IsShapeInferencePass() && dimOutputFinal.IsQuestionmark()) {
      // Return failure as we could not find a constant output size.
      return failure();
    }

    // Save results
    startIndices[ii] = startFinal;
    stepIndices[ii] = stepInput;
    endIndices[ii] = endFinal;
    outputDims[ii] = dimOutputFinal;
  }

  // Handle the default for the non-axis arrays; they are detected with 0 steps
  // (illegal value).
  bool allOutputLit;
  for (uint64_t i = 0; i < dataRank; ++i) {
    if (stepIndices[i].IsUndefined()) {
      // have one unset, put the defaults (start was already at zero, so we are
      // fine).
      startIndices[i] = container.CreateLiteralIndexExpr(0);
      stepIndices[i] = container.CreateLiteralIndexExpr(1);
      IndexExpr dimInput = container.CreateDimIndexExpr(data, dataShape, i);
      endIndices[i] = dimInput;
      outputDims[i] = dimInput;
      if (container.IsShapeInferencePass() && dimInput.IsQuestionmark()) {
        // Return failure as we could not find a constant output size.
        return failure();
      }
    }
#if 1
    startIndices[i].DebugPrint("New Dim\n  start");
    endIndices[i].DebugPrint("  end");
    stepIndices[i].DebugPrint("  step");
    outputDims[i].DebugPrint("  output dim");
#endif
  }
  return success();
}
