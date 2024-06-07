/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- Reduction.cpp - Lowering Reduction Ops ----------------===//
//
// Copyright 2022-2024
//
// =============================================================================
//
// This file lowers the ONNX Reduction Operators to Stablehlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStablehlo/ONNXToStablehloCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {
// Identity values

template <typename Op>
Value getIdentityValue(
    ConversionPatternRewriter &rewriter, Location loc, Type elemType) {
  return nullptr;
}

Value getReduceMaxIdentityValue(
    ConversionPatternRewriter &rewriter, Location loc, Type elemType) {
  MathBuilder createMath(rewriter, loc);
  return rewriter.create<stablehlo::ConstantOp>(
      loc, createMath.negativeInfAttr(elemType));
}

Value getReduceMinIdentityValue(
    ConversionPatternRewriter &rewriter, Location loc, Type elemType) {
  MathBuilder createMath(rewriter, loc);
  return rewriter.create<stablehlo::ConstantOp>(
      loc, createMath.positiveInfAttr(elemType));
}

Value getReduceSumIdentityValue(
    ConversionPatternRewriter &rewriter, Location loc, Type elemType) {
  return rewriter.create<stablehlo::ConstantOp>(
      loc, rewriter.getZeroAttr(elemType));
}

Value getReduceMeanIdentityValue(
    ConversionPatternRewriter &rewriter, Location loc, Type elemType) {
  return rewriter.create<stablehlo::ConstantOp>(
      loc, rewriter.getZeroAttr(elemType));
}

template <>
Value getIdentityValue<ONNXReduceMaxOp>(
    ConversionPatternRewriter &rewriter, Location loc, Type elemType) {
  return getReduceMaxIdentityValue(rewriter, loc, elemType);
}

template <>
Value getIdentityValue<ONNXReduceMaxV18Op>(
    ConversionPatternRewriter &rewriter, Location loc, Type elemType) {
  return getReduceMaxIdentityValue(rewriter, loc, elemType);
}

template <>
Value getIdentityValue<ONNXReduceMaxV13Op>(
    ConversionPatternRewriter &rewriter, Location loc, Type elemType) {
  return getReduceMaxIdentityValue(rewriter, loc, elemType);
}

template <>
Value getIdentityValue<ONNXReduceMinOp>(
    ConversionPatternRewriter &rewriter, Location loc, Type elemType) {
  return getReduceMinIdentityValue(rewriter, loc, elemType);
}

template <>
Value getIdentityValue<ONNXReduceMinV18Op>(
    ConversionPatternRewriter &rewriter, Location loc, Type elemType) {
  return getReduceMinIdentityValue(rewriter, loc, elemType);
}

template <>
Value getIdentityValue<ONNXReduceMinV13Op>(
    ConversionPatternRewriter &rewriter, Location loc, Type elemType) {
  return getReduceMinIdentityValue(rewriter, loc, elemType);
}

template <>
Value getIdentityValue<ONNXReduceSumOp>(
    ConversionPatternRewriter &rewriter, Location loc, Type elemType) {
  return getReduceSumIdentityValue(rewriter, loc, elemType);
}

template <>
Value getIdentityValue<ONNXReduceSumV11Op>(
    ConversionPatternRewriter &rewriter, Location loc, Type elemType) {
  return getReduceSumIdentityValue(rewriter, loc, elemType);
}

template <>
Value getIdentityValue<ONNXReduceMeanOp>(
    ConversionPatternRewriter &rewriter, Location loc, Type elemType) {
  return getReduceMeanIdentityValue(rewriter, loc, elemType);
}

template <>
Value getIdentityValue<ONNXReduceMeanV13Op>(
    ConversionPatternRewriter &rewriter, Location loc, Type elemType) {
  return getReduceMeanIdentityValue(rewriter, loc, elemType);
}

template <typename ONNXReductionOp>
llvm::SmallVector<int64_t, 4> getDefinedAxes(Operation *op) {
  llvm::SmallVector<int64_t, 4> definedAxes;
  ArrayAttr axisAttrs = llvm::dyn_cast<ONNXReductionOp>(op).getAxesAttr();
  if (axisAttrs) {
    for (Attribute axisAttr : axisAttrs.getValue()) {
      int64_t axis = mlir::cast<IntegerAttr>(axisAttr).getInt();
      definedAxes.push_back(axis);
    }
  }
  return definedAxes;
}

llvm::SmallVector<int64_t, 4> getDefinedAxesFromConstAxes(
    Operation *op, Value axesValue, bool keepDims) {
  llvm::SmallVector<int64_t, 4> definedAxes;
  // Assume it is verified that axes are known. Convert DenseElementsAttr to
  // ArrayAttr.
  if (!isNoneValue(axesValue) && getONNXConstantOp(axesValue)) {
    mlir::ElementsAttr constAxes = mlir::dyn_cast_or_null<mlir::ElementsAttr>(
        getONNXConstantOp(axesValue).getValueAttr());
    for (mlir::IntegerAttr element : constAxes.getValues<IntegerAttr>())
      definedAxes.push_back(element.getInt());
    return definedAxes;
  }
  if (isNoneValue(axesValue))
    return definedAxes;
  // Dynamic axes
  RankedTensorType inputType =
      mlir::dyn_cast<RankedTensorType>(op->getOperands()[0].getType());
  RankedTensorType outputType =
      mlir::dyn_cast<RankedTensorType>(op->getResultTypes()[0]);
  assert(inputType != nullptr && outputType != nullptr &&
         "not implemented for dynamic axes when either input or output is not "
         "ranked");

  int64_t inputRank = inputType.getRank();
  int64_t outputRank = outputType.getRank();
  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  llvm::ArrayRef<int64_t> outputShape = outputType.getShape();
  if (keepDims) {
    assert(inputRank == outputRank && "keepdims is true but input and output "
                                      "ranks are not the same");
    for (int64_t i = 0; i < inputRank; ++i)
      if (outputShape[i] == 1 && inputShape[i] != 1)
        definedAxes.push_back(i);
  } else {
    // match greedily if not keepdims
    for (int64_t i = 0, j = 0; i < inputRank; ++i)
      if (j == outputRank || inputShape[i] != outputShape[j])
        definedAxes.push_back(i);
      else
        ++j;
  }

  return definedAxes;
}

template <>
llvm::SmallVector<int64_t, 4> getDefinedAxes<ONNXReduceMaxOp>(Operation *op) {
  ONNXReduceMaxOp reduceMaxOp = cast<ONNXReduceMaxOp>(op);
  Value axesValue = reduceMaxOp.getAxes();
  bool keepDims = reduceMaxOp.getKeepdims() == 1;
  return getDefinedAxesFromConstAxes(op, axesValue, keepDims);
}

template <>
llvm::SmallVector<int64_t, 4> getDefinedAxes<ONNXReduceMaxV18Op>(
    Operation *op) {
  ONNXReduceMaxV18Op reduceMaxOp = cast<ONNXReduceMaxV18Op>(op);
  Value axesValue = reduceMaxOp.getAxes();
  bool keepDims = reduceMaxOp.getKeepdims() == 1;
  return getDefinedAxesFromConstAxes(op, axesValue, keepDims);
}

template <>
llvm::SmallVector<int64_t, 4> getDefinedAxes<ONNXReduceMinOp>(Operation *op) {
  ONNXReduceMinOp reduceMinOp = cast<ONNXReduceMinOp>(op);
  Value axesValue = reduceMinOp.getAxes();
  bool keepDims = reduceMinOp.getKeepdims() == 1;
  return getDefinedAxesFromConstAxes(op, axesValue, keepDims);
}

template <>
llvm::SmallVector<int64_t, 4> getDefinedAxes<ONNXReduceMinV18Op>(
    Operation *op) {
  ONNXReduceMinV18Op reduceMinOp = cast<ONNXReduceMinV18Op>(op);
  Value axesValue = reduceMinOp.getAxes();
  bool keepDims = reduceMinOp.getKeepdims() == 1;
  return getDefinedAxesFromConstAxes(op, axesValue, keepDims);
}

template <>
llvm::SmallVector<int64_t, 4> getDefinedAxes<ONNXReduceSumOp>(Operation *op) {
  ONNXReduceSumOp reduceSumOp = cast<ONNXReduceSumOp>(op);
  Value axesValue = reduceSumOp.getAxes();
  bool keepDims = reduceSumOp.getKeepdims() == 1;
  return getDefinedAxesFromConstAxes(op, axesValue, keepDims);
}

template <>
llvm::SmallVector<int64_t, 4> getDefinedAxes<ONNXReduceMeanOp>(Operation *op) {
  ONNXReduceMeanOp reduceMeanOp = cast<ONNXReduceMeanOp>(op);
  Value axesValue = reduceMeanOp.getAxes();
  bool keepDims = reduceMeanOp.getKeepdims() == 1;
  return getDefinedAxesFromConstAxes(op, axesValue, keepDims);
}

// Block reduce ops
template <typename ReductionOp>
struct BlockReduceOp {
  using Op = void;
};

template <>
struct BlockReduceOp<ONNXReduceMaxOp> {
  using Op = stablehlo::MaxOp;
};

template <>
struct BlockReduceOp<ONNXReduceMaxV18Op> {
  using Op = stablehlo::MaxOp;
};

template <>
struct BlockReduceOp<ONNXReduceMaxV13Op> {
  using Op = stablehlo::MaxOp;
};

template <>
struct BlockReduceOp<ONNXReduceMinOp> {
  using Op = stablehlo::MinOp;
};

template <>
struct BlockReduceOp<ONNXReduceMinV18Op> {
  using Op = stablehlo::MinOp;
};

template <>
struct BlockReduceOp<ONNXReduceMinV13Op> {
  using Op = stablehlo::MinOp;
};

template <>
struct BlockReduceOp<ONNXReduceMeanOp> {
  using Op = stablehlo::AddOp;
};

template <>
struct BlockReduceOp<ONNXReduceMeanV13Op> {
  using Op = stablehlo::AddOp;
};

template <>
struct BlockReduceOp<ONNXReduceSumOp> {
  using Op = stablehlo::AddOp;
};

template <>
struct BlockReduceOp<ONNXReduceSumV11Op> {
  using Op = stablehlo::AddOp;
};

template <typename ReductionOp>
using BlockOp = typename BlockReduceOp<ReductionOp>::Op;

SmallVector<int64_t> getReductionShape(ShapedType inputType,
    const llvm::SmallVector<int64_t, 4> &axes, bool isKeepdims) {
  SmallVector<int64_t> reduceShape;
  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t rank = inputType.getRank();

  // Mark reduction axes.
  llvm::SmallVector<bool, 4> isReductionAxis;
  for (int64_t i = 0; i < rank; ++i) {
    if (std::find(axes.begin(), axes.end(), i) != axes.end())
      isReductionAxis.push_back(true);
    else
      isReductionAxis.push_back(false);
  }

  for (int64_t i = 0; i < rank; ++i) {
    if (!isReductionAxis[i])
      reduceShape.push_back(inputShape[i]);
    else if (isKeepdims)
      reduceShape.push_back(1);
  }
  return reduceShape;
}

Value getReductionShapeValue(Location loc, PatternRewriter &rewriter,
    Value operand, llvm::SmallVector<int64_t, 4> axes, bool keepDims) {
  int64_t rank = mlir::cast<RankedTensorType>(operand.getType()).getRank();
  // Mark reduction axes.
  llvm::SmallVector<bool, 4> isReductionAxis;
  for (int64_t i = 0; i < rank; ++i) {
    if (std::find(axes.begin(), axes.end(), i) != axes.end())
      isReductionAxis.push_back(true);
    else
      isReductionAxis.push_back(false);
  }
  Value inputShape = rewriter.create<shape::ShapeOfOp>(loc, operand);
  SmallVector<Value> dims;
  for (int64_t i = 0; i < rank; i++) {
    if (!isReductionAxis[i]) {
      Value dim = rewriter.create<shape::GetExtentOp>(loc, inputShape, i);
      dims.push_back(dim);
    } else if (keepDims) {
      Value dim = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      dims.push_back(dim);
    }
  }
  Value reduceShapeValue = rewriter.create<shape::FromExtentsOp>(loc, dims);
  reduceShapeValue = rewriter.create<shape::ToExtentTensorOp>(loc,
      RankedTensorType::get({rank}, rewriter.getIndexType()), reduceShapeValue);
  return reduceShapeValue;
}

int64_t getReductionFactor(
    ShapedType inputType, const llvm::SmallVector<int64_t, 4> &axes) {
  SmallVector<int64_t> reduceShape;
  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t rank = inputType.getRank();

  // Mark reduction axes.
  llvm::SmallVector<bool, 4> isReductionAxis;
  for (int64_t i = 0; i < rank; ++i) {
    if (std::find(axes.begin(), axes.end(), i) != axes.end())
      isReductionAxis.push_back(true);
    else
      isReductionAxis.push_back(false);
  }

  int64_t reduceFactor = 1;

  for (int64_t i = 0; i < rank; ++i)
    if (isReductionAxis[i])
      reduceFactor *= inputShape[i];
  return reduceFactor;
}

// Create "stablehlo.reduce", "operand" is reduce input and "identity" is init
// value, reduce from operand to operand[reduceIdx].
template <typename BlockReduceOp>
Value createReduce(Location loc, Value operand, Value identity,
    SmallVector<int64_t> &reduceShape, llvm::SmallVector<int64_t, 4> axes,
    PatternRewriter &rewriter, bool keepDims, ShapedType outputType) {
  RankedTensorType operandType =
      mlir::cast<RankedTensorType>(operand.getType());
  Type reduceResultType =
      RankedTensorType::get(reduceShape, operandType.getElementType());
  stablehlo::ReduceOp reduce = rewriter.create<stablehlo::ReduceOp>(loc,
      reduceResultType, operand, identity, rewriter.getDenseI64ArrayAttr(axes));

  // setup "stablehlo.reduce"'s body
  Region &region = reduce.getBody();
  Block &block = region.emplaceBlock();
  RankedTensorType blockArgumentType =
      RankedTensorType::get({}, operandType.getElementType());
  block.addArgument(blockArgumentType, loc);
  block.addArgument(blockArgumentType, loc);
  BlockArgument firstArgument = *block.args_begin();
  BlockArgument secondArgument = *block.args_rbegin();
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&block);
    Value reduceResult =
        rewriter.create<BlockReduceOp>(loc, firstArgument, secondArgument);
    rewriter.create<stablehlo::ReturnOp>(loc, reduceResult);
  }
  Value result = reduce.getResult(0);
  if (keepDims) {
    Value reduceShapeValue =
        getReductionShapeValue(loc, rewriter, operand, axes, true);
    result = rewriter.create<stablehlo::DynamicReshapeOp>(
        loc, outputType, result, reduceShapeValue);
  }
  return result;
}

// ONNXReductionOp is implemented as stablehlo.reduce with its body built
// correspondingly.
template <typename ONNXReductionOp>
struct ONNXReductionOpLoweringToStablehlo : public ConversionPattern {
  bool computeMean = false;

  ONNXReductionOpLoweringToStablehlo(MLIRContext *ctx, bool computeMean = false)
      : ConversionPattern(ONNXReductionOp::getOperationName(), 1, ctx) {
    this->computeMean = computeMean;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    Value input = operands[0];
    // Type
    RankedTensorType inputType = mlir::cast<RankedTensorType>(input.getType());
    if (inputType == nullptr)
      return failure();
    Type resultType = *op->result_type_begin();
    ShapedType outputType = mlir::cast<ShapedType>(resultType);
    if (outputType == nullptr)
      return failure();
    Type elemType = inputType.getElementType();
    Value identity = getIdentityValue<ONNXReductionOp>(rewriter, loc, elemType);
    int64_t inRank = inputType.getRank();

    // Get axes value defined by op
    // Leave empty is not defined
    llvm::SmallVector<int64_t, 4> definedAxes =
        getDefinedAxes<ONNXReductionOp>(op);
    llvm::SmallVector<int64_t, 4> axes;
    if (definedAxes.size()) {
      for (int64_t axis : definedAxes) {
        if (axis < -inRank || axis > inRank - 1)
          return emitError(loc, "axes value out of range");
        int64_t newaxis = axis >= 0 ? axis : (inRank + axis);
        if (std::find(axes.begin(), axes.end(), newaxis) == axes.end())
          axes.push_back(newaxis);
      }
    } else
      for (decltype(inRank) i = 0; i < inRank; ++i)
        axes.push_back(i);

    // KeepDims
    int64_t keepdims = llvm::dyn_cast<ONNXReductionOp>(op).getKeepdims();
    bool isKeepdims = (keepdims == 1) ? true : false;

    SmallVector<int64_t> reducedShape =
        getReductionShape(inputType, axes, false);
    Value reduceResult = createReduce<BlockOp<ONNXReductionOp>>(loc, input,
        identity, reducedShape, axes, rewriter, isKeepdims, outputType);
    if (computeMean) {
      // TODO: support dynamic shape
      if (inputType.hasStaticShape()) {
        int64_t reduceFactor = getReductionFactor(inputType, axes);
        Value reduceFactorValue =
            getShapedFloat(loc, rewriter, reduceFactor, reduceResult);
        reduceResult = rewriter.create<stablehlo::DivOp>(
            loc, reduceResult, reduceFactorValue);
      } else {
        Value ones;
        if (mlir::isa<IntegerType>(elemType))
          ones = getShapedInt(loc, rewriter, 1, input);
        else
          ones = getShapedFloat(loc, rewriter, 1.0, input);
        Value reduceSum = createReduce<stablehlo::AddOp>(loc, ones, identity,
            reducedShape, axes, rewriter, isKeepdims, outputType);
        reduceResult = rewriter.create<stablehlo::DivOp>(
            loc, outputType, reduceResult, reduceSum);
      }
    }
    rewriter.replaceOp(op, reduceResult);
    return success();
  }
};

} // namespace

void populateLoweringONNXReductionOpToStablehloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXReductionOpLoweringToStablehlo<mlir::ONNXReduceMaxOp>,
      ONNXReductionOpLoweringToStablehlo<mlir::ONNXReduceMaxV18Op>,
      ONNXReductionOpLoweringToStablehlo<mlir::ONNXReduceMaxV13Op>,
      ONNXReductionOpLoweringToStablehlo<mlir::ONNXReduceMinOp>,
      ONNXReductionOpLoweringToStablehlo<mlir::ONNXReduceMinV18Op>,
      ONNXReductionOpLoweringToStablehlo<mlir::ONNXReduceMinV13Op>,
      ONNXReductionOpLoweringToStablehlo<mlir::ONNXReduceSumOp>,
      ONNXReductionOpLoweringToStablehlo<mlir::ONNXReduceSumV11Op>>(ctx);
  patterns.insert<ONNXReductionOpLoweringToStablehlo<mlir::ONNXReduceMeanOp>>(
      ctx, /*computeMean=*/true);
  patterns
      .insert<ONNXReductionOpLoweringToStablehlo<mlir::ONNXReduceMeanV13Op>>(
          ctx, /*computeMean=*/true);
}

} // namespace onnx_mlir
