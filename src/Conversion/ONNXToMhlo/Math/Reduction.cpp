/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- Reduction.cpp - Lowering Reduction Ops ----------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Reduction Operators to Mhlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToMhlo/ONNXToMhloCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {
// Identity values

template <typename Op>
Value getIdentityValue(
    ConversionPatternRewriter &rewriter, Location loc, FloatType elemType) {
  return nullptr;
}

template <>
Value getIdentityValue<ONNXReduceMaxOp>(
    ConversionPatternRewriter &rewriter, Location loc, FloatType elemType) {
  return rewriter.create<mhlo::ConstOp>(
      loc, rewriter.getFloatAttr(
               elemType, APFloat::getInf(elemType.getFloatSemantics(),
                             /*isNegative=*/true)));
}

template <>
Value getIdentityValue<ONNXReduceMinOp>(
    ConversionPatternRewriter &rewriter, Location loc, FloatType elemType) {
  return rewriter.create<mhlo::ConstOp>(
      loc, rewriter.getFloatAttr(
               elemType, APFloat::getInf(elemType.getFloatSemantics(),
                             /*isNegative=*/false)));
}

template <>
Value getIdentityValue<ONNXReduceSumOp>(
    ConversionPatternRewriter &rewriter, Location loc, FloatType elemType) {
  return rewriter.create<mhlo::ConstOp>(loc, rewriter.getZeroAttr(elemType));
}

template <>
Value getIdentityValue<ONNXReduceSumV11Op>(
    ConversionPatternRewriter &rewriter, Location loc, FloatType elemType) {
  return rewriter.create<mhlo::ConstOp>(loc, rewriter.getZeroAttr(elemType));
}

template <>
Value getIdentityValue<ONNXReduceMeanOp>(
    ConversionPatternRewriter &rewriter, Location loc, FloatType elemType) {
  return rewriter.create<mhlo::ConstOp>(loc, rewriter.getZeroAttr(elemType));
}

template <typename ONNXReductionOp>
std::vector<int64_t> getDefinedAxes(Operation *op) {
  std::vector<int64_t> definedAxes;
  ArrayAttr axisAttrs = llvm::dyn_cast<ONNXReductionOp>(op).axesAttr();
  if (axisAttrs) {
    for (auto axisAttr : axisAttrs.getValue()) {
      int64_t axis = axisAttr.cast<IntegerAttr>().getInt();
      definedAxes.push_back(axis);
    }
  }
  return definedAxes;
}

template <>
std::vector<int64_t> getDefinedAxes<ONNXReduceSumOp>(Operation *op) {
  std::vector<int64_t> definedAxes;
  ONNXReduceSumOp reduceSumOp = cast<ONNXReduceSumOp>(op);
  Value axesValue = reduceSumOp.axes();

  // Assume it is verified that axes are known. Convert DenseElementsAttr to
  // ArrayAttr.
  if (!isFromNone(axesValue) && getONNXConstantOp(axesValue)) {
    auto constAxes = getONNXConstantOp(axesValue)
                         .valueAttr()
                         .dyn_cast_or_null<mlir::DenseElementsAttr>();
    for (auto element : constAxes.getValues<IntegerAttr>())
      definedAxes.push_back(element.getInt());
    return definedAxes;
  }
  if (isFromNone(axesValue))
    return definedAxes;
  // Dynamic axes
  RankedTensorType inputType =
      op->getOperands()[0].getType().dyn_cast<RankedTensorType>();
  RankedTensorType outputType =
      op->getResultTypes()[0].dyn_cast<RankedTensorType>();
  assert(inputType != nullptr && outputType != nullptr &&
         "not implemented for dynamic axes when either input or output is not "
         "ranked");
  bool keepDims = reduceSumOp.keepdims() == 1;
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
    for (int64_t i = 0, j = 0; i < inputRank; ++i)
      if (j == outputRank || inputShape[i] != outputShape[j])
        definedAxes.push_back(i);
      else
        ++j;
  }

  return definedAxes;
}

// Block reduce ops
template <typename ReductionOp>
struct BlockReduceOp {
  using Op = void;
};

template <>
struct BlockReduceOp<ONNXReduceMaxOp> {
  using Op = mhlo::MaxOp;
};

template <>
struct BlockReduceOp<ONNXReduceMinOp> {
  using Op = mhlo::MinOp;
};

template <>
struct BlockReduceOp<ONNXReduceMeanOp> {
  using Op = mhlo::AddOp;
};

template <>
struct BlockReduceOp<ONNXReduceSumOp> {
  using Op = mhlo::AddOp;
};

template <>
struct BlockReduceOp<ONNXReduceSumV11Op> {
  using Op = mhlo::AddOp;
};

template <typename ReductionOp>
using BlockOp = typename BlockReduceOp<ReductionOp>::Op;

SmallVector<int64_t> getReductionShape(
    ShapedType inputType, const std::vector<int64_t> &axes, bool isKeepdims) {
  SmallVector<int64_t> reduceShape;
  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t rank = inputType.getRank();

  // Mark reduction axes.
  std::vector<bool> isReductionAxis;
  for (int64_t i = 0; i < rank; ++i) {
    if (std::find(axes.begin(), axes.end(), i) != axes.end())
      isReductionAxis.push_back(true);
    else
      isReductionAxis.push_back(false);
  }

  for (int64_t i = 0; i < rank; ++i) {
    if (isReductionAxis[i]) {
      if (isKeepdims)
        reduceShape.push_back(1);
    } else
      reduceShape.push_back(inputShape[i]);
  }
  return reduceShape;
}

int64_t getReductionFactor(
    ShapedType inputType, const std::vector<int64_t> &axes) {
  SmallVector<int64_t> reduceShape;
  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t rank = inputType.getRank();

  // Mark reduction axes.
  std::vector<bool> isReductionAxis;
  for (int64_t i = 0; i < rank; ++i) {
    if (std::find(axes.begin(), axes.end(), i) != axes.end())
      isReductionAxis.push_back(true);
    else
      isReductionAxis.push_back(false);
  }

  int64_t reduceFactor = 1;

  for (int64_t i = 0; i < rank; ++i) {
    if (isReductionAxis[i]) {
      reduceFactor *= inputShape[i];
    }
  }
  return reduceFactor;
}

// Create "mhlo.reduce", "operand" is reduce input and "zero" is init value,
// reduce from operand to operand[reduceIdx].
template <typename BlockReduceOp>
Value createReduce(Location loc, Value operand, Value zero,
    SmallVector<int64_t> &reduceShape, std::vector<int64_t> axes,
    PatternRewriter &rewriter, bool keepDims) {
  auto operandType = operand.getType().cast<RankedTensorType>();
  Type reduceResultType =
      RankedTensorType::get(reduceShape, operandType.getElementType());
  mhlo::ReduceOp reduce = rewriter.create<mhlo::ReduceOp>(
      loc, reduceResultType, operand, zero, rewriter.getI64TensorAttr(axes));

  // setup "mhlo.reduce"'s body
  Region &region = reduce.body();
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
    rewriter.create<mhlo::ReturnOp>(loc, reduceResult);
  }
  Value result = reduce.getResult(0);
  if (keepDims) {
    RankedTensorType inputType = operand.getType().cast<RankedTensorType>();
    SmallVector<int64_t> resultShape =
        getReductionShape(inputType, axes, true);
    Type resultType =
      RankedTensorType::get(resultShape, inputType.getElementType());
    Value shape = rewriter.create<mhlo::ConstOp>(loc, rewriter.getI64TensorAttr(resultShape));
    result = rewriter.create<mhlo::DynamicReshapeOp>(loc, resultType, result, shape);
  }
  return result;
}

template <typename ONNXReductionOp>
struct ONNXReductionOpLoweringToMhlo : public ConversionPattern {
  bool computeMean = false;

  ONNXReductionOpLoweringToMhlo(MLIRContext *ctx, bool computeMean = false)
      : ConversionPattern(ONNXReductionOp::getOperationName(), 1, ctx) {
    this->computeMean = computeMean;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    Value input = operands[0];
    // Type
    RankedTensorType inputType = input.getType().cast<RankedTensorType>();
    if (inputType == nullptr)
      return failure();
    Type resultType = *op->result_type_begin();
    ShapedType outputType = resultType.cast<ShapedType>();
    if (outputType == nullptr)
      return failure();
    FloatType elemType = inputType.getElementType().cast<FloatType>();
    Value zero = getIdentityValue<ONNXReductionOp>(rewriter, loc, elemType);
    int64_t inRank = inputType.getRank();

    // Get axes value defined by op
    // Leave empty is not defined
    std::vector<int64_t> definedAxes = getDefinedAxes<ONNXReductionOp>(op);
    std::vector<int64_t> axes;
    if (definedAxes.size()) {
      for (auto axis : definedAxes) {
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
    auto keepdims = llvm::dyn_cast<ONNXReductionOp>(op).keepdims();
    bool isKeepdims = (keepdims == 1) ? true : false;

    SmallVector<int64_t> reducedShape =
        getReductionShape(inputType, axes, false);
    auto reduceResult = createReduce<BlockOp<ONNXReductionOp>>(
        loc, input, zero, reducedShape, axes, rewriter, isKeepdims);
    if (computeMean) {
      // TODO: support dynamic shape
      int64_t reduceFactor = getReductionFactor(inputType, axes);
      Value reduceFactorValue = getShapedFloat(loc, rewriter, outputType,
          1.0 / reduceFactor, reduceResult, outputType);
      reduceResult =
          rewriter.create<mhlo::DivOp>(loc, reduceResult, reduceFactorValue);
    }
    rewriter.replaceOp(op, reduceResult);
    return success();
  }
};

} // namespace

void populateLoweringONNXReductionOpToMhloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXReductionOpLoweringToMhlo<mlir::ONNXReduceMaxOp>,
      ONNXReductionOpLoweringToMhlo<mlir::ONNXReduceMinOp>,
      ONNXReductionOpLoweringToMhlo<mlir::ONNXReduceSumOp>,
      ONNXReductionOpLoweringToMhlo<mlir::ONNXReduceSumV11Op>>(ctx);
  patterns.insert<ONNXReductionOpLoweringToMhlo<mlir::ONNXReduceMeanOp>>(
      ctx, /*computeMean=*/true);
}

} // namespace onnx_mlir
