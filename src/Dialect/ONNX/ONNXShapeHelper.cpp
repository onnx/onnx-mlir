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
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

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

//===----------------------------------------------------------------------===//
// ONNX Helper for Shape inference
//===----------------------------------------------------------------------===//

LogicalResult GetIndexExprFromOperandValueAtIndex(Operation *op, Value operand,
    uint64_t i, IndexExprContainer &container, IndexExpr &indexExpr) {
  if (auto attrArray = getDenseElementAttributeFromValue(operand)) {
    // Could extract an dense attribute from definition of operand.
    if (i >= attrArray.getType().getDimSize(0)) {
      printf("error 1\n");
      return op->emitError("operand literal has wrong shape");
    }
    auto attrVal = attrArray.getValue(ArrayRef<uint64_t>({i}));
    int64_t attrInt = attrVal.cast<IntegerAttr>().getInt();
    indexExpr.InitAsIntLit(attrInt);
  } else {
    // Read starts as an array.
    if (container.IsShapeInferencePass()) {
      // Not a constant; don't add code.
      indexExpr.InitAsQuestionmark();
      return success();
    }
    auto indexVal = emitConstantOp(container.GetRewriter(),
        container.GetLocation(), container.GetRewriter().getIndexType(), i);
    SmallVector<Value, 1> memrefVal = {indexVal};
    auto loadVal = container.GetRewriter().create<AffineLoadOp>(
        container.GetLocation(), operand, memrefVal);
    indexExpr.InitAsSymbol(loadVal);
  }
  return success();
}

LogicalResult GetIndexExprFromOperandValueAtIndex(Operation *op, Value operand,
    uint64_t i, int64_t defaultIntLit, IndexExprContainer &container,
    IndexExpr &indexExpr) {

  if (operand.getType().isa<NoneType>()) {
    // Argument undefined, so use the default value.
    indexExpr.InitAsIntLit(defaultIntLit);
  } else if (auto attrArray = getDenseElementAttributeFromValue(operand)) {
    if (i > attrArray.getType().getDimSize(0)) {
      // Not enought attributes for this index, install the default value.
      indexExpr.InitAsIntLit(defaultIntLit);
    } else {
      // We have enought attributes for this index, get the value.
      auto attrVal = attrArray.getValue(ArrayRef<uint64_t>({i}));
      int64_t attrInt = attrVal.cast<IntegerAttr>().getInt();
      indexExpr.InitAsIntLit(attrInt);
    }
  } else {
    // Read starts as an array.
    if (container.IsShapeInferencePass()) {
      // Not a constant; don't add code.
      indexExpr.InitAsQuestionmark();
      return success();
    }
    auto indexVal = emitConstantOp(container.GetRewriter(),
        container.GetLocation(), container.GetRewriter().getIndexType(), i);
    SmallVector<Value, 1> memrefVal = {indexVal};
    auto loadVal = container.GetRewriter().create<AffineLoadOp>(
        container.GetLocation(), operand, memrefVal);
    indexExpr.InitAsSymbol(loadVal);
  }
  return success();
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
    for (auto value : valueAttribute.getValues<IntegerAttr>()) {
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
  IndexExpr zeroIE(0);
  IndexExpr oneIE(1);
  startIndices.resize(dataRank);
  stepIndices.resize(dataRank);
  endIndices.resize(dataRank);
  outputDims.resize(dataRank);

  // SmallVector<uint64_t, 1> index1D(1, 0);
  for (uint64_t i = 0; i < sliceRank; i++) {
    // i is index in start/step/end/output
    // ii is logical index in mem/loop bounds
    int ii = axesIntLit[i];
    // Get start.
    IndexExpr startInputIE;
    Value starts = operandAdaptor.starts();
    if (failed(GetIndexExprFromOperandValueAtIndex(
            op, starts, i, container, startInputIE))) {
      return sliceOp->emitError("start input parameter could not be processed");
    }
    startInputIE.DebugPrint("start input");
    // Get end.
    IndexExpr endInputIE;
    Value ends = operandAdaptor.ends();
    if (failed(GetIndexExprFromOperandValueAtIndex(
            op, ends, i, container, endInputIE))) {
      return sliceOp->emitError("end input parameter could not be processed");
    }
    endInputIE.DebugPrint("end input");
    // Get steps.
    IndexExpr stepInputIE;
    Value steps = operandAdaptor.steps();
    if (failed(GetIndexExprFromOperandValueAtIndex(
            op, steps, i, 1, container, stepInputIE))) {
      return sliceOp->emitError("step input parameter could not be processed");
    }
    if (stepInputIE.IsIntLit() && stepInputIE.GetIntLit() == 0) {
      return sliceOp->emitError("step input parameter cannot be zero");
    }
    stepInputIE.DebugPrint("step input");
    // Get dim.
    IndexExpr dimInputIE(container, data, dataShape, ii);
    dimInputIE.DebugPrint("dim input");
    // If in shape inference mode and we don't have the constant info, take
    // early break.
    if (container.PerformShapeInference() &&
        (startInputIE.IsQuestionmark() || endInputIE.IsQuestionmark() ||
            stepInputIE.IsQuestionmark() || dimInputIE.IsQuestionmark())) {
      // return failure as we could not find a constant shape
      return failure();
    }
    // Now proceed with the computations for start/end/dim.
    // Calculation for start: start < 0 ? start + dim : start.
    IndexExpr startPlusDimIE, startPosIE, startNegClampIE, startPosClampIE,
        startFinalIE;
    startPlusDimIE.Add(container, startInputIE, dimInputIE);
    startPlusDimIE.DebugPrint("start plus dim input");
    startPosIE.Select(container, startInputIE, CmpIPredicate::slt, zeroIE,
        startPlusDimIE, startInputIE);
    startPosIE.DebugPrint("start pos");
    // Step < 0: clamp(0, start, dim -1) else clamp(0, start, dim)
    startNegClampIE.Clamp(container, startPosIE, zeroIE, 0, dimInputIE, -1);
    startNegClampIE.DebugPrint("start clamp neg");
    startPosClampIE.Clamp(container, startPosIE, zeroIE, 0, dimInputIE, 0);
    startPosClampIE.DebugPrint("start clamp pos");
    startFinalIE.Select(container, stepInputIE, CmpIPredicate::slt, zeroIE,
        startNegClampIE, startPosClampIE);
    startFinalIE.DebugPrint("start final");
    // Calculation for end: end <= -inf -> -1;  end >= inf -> dim
    // otherwise end<0 -> end + dim.
    IndexExpr posInfinityIE(std::numeric_limits<int32_t>::max());
    IndexExpr negInfinityIE(std::numeric_limits<int32_t>::min());
    IndexExpr negOneIE(-1);
    IndexExpr endPlusDimIE, endPosIE, endWithNegInfIE, endWithPosInfIE,
        endNegClampIE, endPosClampIE, endFinalIE;
    endPlusDimIE.Add(container, endInputIE, dimInputIE);
    endPosIE.Select(container, endInputIE, CmpIPredicate::slt, zeroIE,
        endPlusDimIE, endInputIE);
    endWithNegInfIE.Select(container, endInputIE, CmpIPredicate::sle,
        negInfinityIE, negOneIE, endPosIE);
    endWithPosInfIE.Select(container, endInputIE, CmpIPredicate::sge,
        posInfinityIE, zeroIE, endWithNegInfIE);
    // End: step<0: clamp(-1, end, dim); step>0 clamp(0, end, dim)
    endNegClampIE.Clamp(container, endWithPosInfIE, negOneIE, 0, dimInputIE, 0);
    endPosClampIE.Clamp(container, endWithPosInfIE, zeroIE, 0, dimInputIE, 0);
    endFinalIE.Select(container, stepInputIE, CmpIPredicate::slt, zeroIE,
        endNegClampIE, endPosClampIE);
    endFinalIE.DebugPrint("end final");

    // Calculation for output size.
    IndexExpr dimOutputSubIE, dimOutputCeilIE, dimOutputFinalIE;
    dimOutputSubIE.Sub(container, endFinalIE, startFinalIE);
    dimOutputCeilIE.CeilDiv(container, dimOutputSubIE, stepInputIE);
    // TODO: add min and max.
    dimOutputFinalIE.Select(container, dimOutputCeilIE, CmpIPredicate::slt,
        zeroIE, zeroIE, dimOutputCeilIE);
    dimOutputFinalIE.DebugPrint("output dim final");
    if (container.PerformShapeInference() &&
        dimOutputFinalIE.IsQuestionmark()) {
      // Return failure as we could not find a constant output size.
      return failure();
    }

    // Save results
    startIndices[ii] = startFinalIE;
    stepIndices[ii] = stepInputIE;
    endIndices[ii] = endFinalIE;
    outputDims[ii] = dimOutputFinalIE;
  }

  // Handle the default for the non-axis arrays; they are detected with 0 steps
  // (illegal value).
  bool allOutputLit;
  for (uint64_t i = 0; i < dataRank; ++i) {
    if (!stepIndices[i].IsDefined()) {
      // have one unset, put the defaults (start was already at zero, so we are
      // fine).
      startIndices[i] = zeroIE;
      stepIndices[i] = oneIE;
      IndexExpr dimInputIE(container, data, dataShape, i);
      endIndices[i] = dimInputIE;
      outputDims[i] = dimInputIE;
      if (container.PerformShapeInference() && dimInputIE.IsQuestionmark()) {
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
