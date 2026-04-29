/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Slice.cpp - ONNX Operations ---------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Slice operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

LogicalResult ONNXSliceOpShapeHelper::computeShape() {
  // Get info about input data operand.
  ONNXSliceOpAdaptor operandAdaptor(operands);
  Value data = operandAdaptor.getData();
  uint64_t dataRank = mlir::cast<ShapedType>(data.getType()).getShape().size();

  // Get each of the axes, and save the literal values in axesIntLit.
  SmallVector<int64_t, 4> axesIntLit;
  Value axes = operandAdaptor.getAxes();
  if (isNoneValue(axes)) {
    // If `axes` are omitted, they are set to `[0, ..., nDim-1]`."
    for (uint64_t i = 0; i < dataRank; ++i)
      axesIntLit.emplace_back(i);
  } else {
    SmallVector<IndexExpr, 4> axesSymbol;
    createIE->getIntFromArrayAsSymbols(axes, axesSymbol);
    for (IndexExpr val : axesSymbol) {
      if (!val.isLiteral())
        return success();
      int64_t axis = val.getLiteral();
      if (axis < 0)
        axis += dataRank;
      if (!(axis >= 0 && axis < static_cast<int64_t>(dataRank)))
        return op->emitError("Axes contains an out-of-bound index");
      axesIntLit.emplace_back(axis);
    }
  }
  uint64_t sliceRank = axesIntLit.size();

  // Initialize context and results (start & output)
  starts.resize(dataRank);
  steps.resize(dataRank);
  ends.resize(dataRank);
  DimsExpr outputDims;
  outputDims.resize(dataRank);

  for (uint64_t i = 0; i < sliceRank; i++) {
    // i is index in start/step/end/output
    // ii is logical index in mem/loop bounds
    int ii = axesIntLit[i];
    // Get start, end, step, and dim index expressions.
    // Get start.
    SymbolIndexExpr startInput =
        createIE->getIntFromArrayAsSymbol(operandAdaptor.getStarts(), i);
    if (startInput.isUndefined())
      return op->emitError("start input parameter could not be processed");
    // Get end.
    SymbolIndexExpr endInput =
        createIE->getIntFromArrayAsSymbol(operandAdaptor.getEnds(), i);
    if (endInput.isUndefined())
      return op->emitError("end input parameter could not be processed");
    // Get step.
    SymbolIndexExpr stepInput =
        isNoneValue(operandAdaptor.getSteps())
            ? LitIE(1)
            : createIE->getIntFromArrayAsSymbol(operandAdaptor.getSteps(), i);
    if (stepInput.isUndefined())
      return op->emitError("step input parameter could not be processed");
    if (stepInput.isLiteral() && stepInput.getLiteral() == 0)
      return op->emitError("step input parameter cannot be zero");
    // Get dim.
    DimIndexExpr dimInput = createIE->getShapeAsDim(data, ii);

    // Now proceed with the computations for start/end/dim.
    // Calculation for start: start < 0 ? start + dim : start.
    IndexExpr startPos =
        IndexExpr::select(startInput < 0, startInput + dimInput, startInput);
    // Step < 0: clamp(0, start, dim -1) else clamp(0, start, dim)
    IndexExpr neg = startPos.clamp(0, dimInput - 1);
    IndexExpr pos = startPos.clamp(0, dimInput);
    IndexExpr startFinal = IndexExpr::select(stepInput < 0, neg, pos);

    IndexExpr endPos = endInput;
    IndexExpr endInputIsNeg = endInput < 0;
    int64_t maxI64 = std::numeric_limits<int64_t>::max();
    IndexExpr maxMinusDim = LitIE(maxI64) - dimInput;
    IndexExpr endInputSafe = IndexExpr::min({endPos, maxMinusDim});
    endPos = IndexExpr::select(endInputIsNeg, endInputSafe + dimInput, endPos);

    // End: step<0: clamp(-1, end, dim - 1); step>0 clamp(0, end, dim)
    neg = endPos.clamp(-1, dimInput - 1);
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
      starts[i] = LitIE(0);
      steps[i] = LitIE(1);
      DimIndexExpr dimInput = createIE->getShapeAsDim(data, i);
      ends[i] = dimInput;
      outputDims[i] = dimInput;
    }
  }

  // Save the final result.
  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Verify
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shape Inference
//===----------------------------------------------------------------------===//

LogicalResult ONNXSliceOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Cannot infer shape if no shape exists.
  if (!hasShapeAndRank(getData()))
    return success();

  Value axes = getAxes();
  Value steps = getSteps();

  // Cannot infer shape if axes is not a constant. It can be a constant after
  // several rounds of shape-inference and constant propagation.
  if (!isNoneValue(axes) && !isConstLikeValue(axes))
    return success();

  {
    OpBuilder builder(this->getContext());
    OnnxBuilder createONNX(builder, this->getLoc());
    builder.setInsertionPoint(*this);

    auto buildI64Const = [&](ArrayRef<int64_t> vals) -> Value {
      auto ty = RankedTensorType::get(
          {static_cast<int64_t>(vals.size())}, builder.getI64Type());
      return createONNX.constant(DenseElementsAttr::get(ty, vals));
    };

    // Helper: return the static length of the starts tensor, or -1 if not
    // yet known (unranked type or dynamic first dim).  Callers skip
    // materialisation and let a later inferShapes round handle it.
    auto getStartsLen = [&]() -> std::optional<int64_t> {
      auto ty = mlir::dyn_cast<RankedTensorType>(getStarts().getType());
      if (!ty)
        return std::nullopt;
      int64_t n = ty.getShape()[0];
      return (n == ShapedType::kDynamic) ? std::nullopt
                                         : std::optional<int64_t>(n);
    };

    // If axes is not specified, default to [0, ..., len(starts)-1].
    if (isNoneValue(axes)) {
      auto maybeN = getStartsLen();
      if (!maybeN)
        return success(); // starts shape not yet known; retry later
      int64_t n = *maybeN;
      SmallVector<int64_t> vals;
      for (int64_t s = 0; s < n; ++s)
        vals.push_back(s);
      this->setOperand(3, buildI64Const(vals));
    }

    // If steps is not specified, default to [1, ..., 1] (same length as
    // starts).
    if (isNoneValue(steps)) {
      auto maybeN = getStartsLen();
      if (!maybeN)
        return success(); // starts shape not yet known; retry later
      int64_t n = *maybeN;
      SmallVector<int64_t> vals(n, 1);
      this->setOperand(4, buildI64Const(vals));
    }

    // Normalize axes to non-negative, and starts/ends steps. Ends are only
    // normalized for positive steps. This runs after None axes/steps have been
    // materialized above, so all four operands are now explicit constants.

    auto canonicalize = [&]() {
      const auto dataTy = mlir::dyn_cast<RankedTensorType>(getData().getType());
      if (!dataTy || !dataTy.hasStaticShape()) {
        return;
      }
      SmallVector<int64_t> axesVals, stepsVals, startsVals, endsVals;
      if (!onnx_mlir::getI64ValuesFromONNXConstantOp(getAxes(), axesVals) ||
          !onnx_mlir::getI64ValuesFromONNXConstantOp(getSteps(), stepsVals) ||
          !onnx_mlir::getI64ValuesFromONNXConstantOp(getStarts(), startsVals) ||
          !onnx_mlir::getI64ValuesFromONNXConstantOp(getEnds(), endsVals)) {
        return;
      }
      const int64_t rank = dataTy.getRank();
      const auto dataShape = dataTy.getShape();
      const auto numAxes = static_cast<int64_t>(axesVals.size());
      SmallVector<int64_t> newAxes(axesVals);
      SmallVector<int64_t> newStarts(startsVals);
      SmallVector<int64_t> newEnds(endsVals);

      // A step of 0 is invalid
      if (llvm::any_of(stepsVals, [](int64_t s) { return s == 0; })) {
        return;
      }

      for (int64_t i = 0; i < numAxes; ++i) {
        int64_t axis = newAxes[i];
        if (axis < 0)
          axis += rank;
        if (axis < 0 || axis >= rank) {
          return;
        }
        newAxes[i] = axis;

        const int64_t step = stepsVals[i];
        const int64_t dim = dataShape[axis];

        auto wrapAndClamp = [dim](int64_t v) -> int64_t {
          if (v < 0)
            v = (v < -dim) ? 0 : v + dim;
          return std::clamp<int64_t>(v, 0LL, dim);
        };
        const int64_t startHi =
            (step > 0) ? dim : std::max<int64_t>(0, dim - 1);
        newStarts[i] =
            std::clamp<int64_t>(wrapAndClamp(newStarts[i]), 0, startHi);
        if (step < 0)
          continue; // skip end: negative-step -1 sentinel not
                    // idempotent
        newEnds[i] = wrapAndClamp(newEnds[i]);
      }

      if (newAxes != axesVals)
        this->setOperand(3, buildI64Const(newAxes));
      if (newStarts != startsVals)
        this->setOperand(1, buildI64Const(newStarts));
      if (newEnds != endsVals)
        this->setOperand(2, buildI64Const(newEnds));
    };

    canonicalize();
  }

  Type elementType =
      mlir::cast<ShapedType>(getData().getType()).getElementType();
  ONNXSliceOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// Folder
//===----------------------------------------------------------------------===//
OpFoldResult ONNXSliceOp::fold(FoldAdaptor adaptor) {

  auto isZero = [&](auto start) { return start.getLiteral() == 0; };
  auto isOne = [&](auto step) { return step.getLiteral() == 1; };

  auto inputTy = llvm::dyn_cast<RankedTensorType>(getData().getType());
  auto outputTy = llvm::dyn_cast<RankedTensorType>(getOutput().getType());
  if (inputTy && inputTy == outputTy && inputTy.hasStaticShape()) {
    // Get starts and steps via ShapeHelper.
    ONNXSliceOpShapeHelper shapeHelper(getOperation(), {});
    if (failed(shapeHelper.computeShape()))
      return nullptr;

    // All starts must be 0.
    if (!llvm::all_of(shapeHelper.starts, isZero)) {
      return nullptr;
    }
    // All steps must be 1.
    if (!llvm::all_of(shapeHelper.steps, isOne)) {
      return nullptr;
    }
    return getData();
  }
  return nullptr;
}
