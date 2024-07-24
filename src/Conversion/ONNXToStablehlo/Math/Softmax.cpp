/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Softmax.cpp - Softmax Ops -------------------===//
//
// Copyright 2022-2024
//
// =============================================================================
//
// This file lowers ONNX softmax operators to Stablehlo dialect.
//

#include "src/Conversion/ONNXToStablehlo/DialectBuilder.hpp"
#include "src/Conversion/ONNXToStablehlo/ONNXToStablehloCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "stablehlo/dialect/BroadcastUtils.h"
#include <iostream>


using namespace mlir;

namespace onnx_mlir{

namespace{


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

//Calutes Broadcast dimensions
SmallVector<int64_t> getBroadcastDims(Value operand, llvm::SmallVector<int64_t, 4> axes)
{
  int64_t rank = mlir::cast<RankedTensorType>(operand.getType()).getRank();
  // Mark reduction axes.
  llvm::SmallVector<bool, 4> isReductionAxis;
  for (int64_t i = 0; i < rank; ++i) {
    if (std::find(axes.begin(), axes.end(), i) != axes.end())
      isReductionAxis.push_back(true);
    else
      isReductionAxis.push_back(false);
  }
  SmallVector<int64_t> dims;
  for (int64_t i = 0; i < rank; i++) {
    if (!isReductionAxis[i]) {
      dims.push_back(i);
    } 
  }

  return dims;
}


Value computeReduceSum(Location loc, Value operand, Value identity,
    SmallVector<int64_t> &reduceShape, llvm::SmallVector<int64_t, 4> axes,
    PatternRewriter &rewriter, bool keepDims, ShapedType outputType){
    
    RankedTensorType operandType =
        mlir::cast<RankedTensorType>(operand.getType());
    Type reduceResultType =
        RankedTensorType::get(reduceShape, operandType.getElementType());
    stablehlo::ReduceOp reduce = rewriter.create<stablehlo::ReduceOp>(loc,
        reduceResultType, operand, identity, rewriter.getDenseI64ArrayAttr(axes));

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
            rewriter.create<stablehlo::AddOp>(loc, firstArgument, secondArgument);
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

bool hasStaticShape(Value val) {
  // Get the type of the value
  Type type = val.getType();

  // Check if the type is a RankedTensorType
  if (auto rankedTensorType = mlir::dyn_cast<RankedTensorType>(type)) {
    // Check if all dimensions are static
    for (int64_t dim : rankedTensorType.getShape()) {
      if (dim == ShapedType::kDynamic) {
        return false; // Found a dynamic dimension
      }
    }
    return true; // All dimensions are static
  }

  // The type is not a RankedTensorType or has dynamic dimensions
  return false;
}

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

struct ONNXSoftmaxOpLoweringToStablehlo : public ConversionPattern {
  ONNXSoftmaxOpLoweringToStablehlo(MLIRContext *ctx)
      : ConversionPattern(ONNXSoftmaxOp::getOperationName(), 1, ctx) {}
    
  
  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    Value operand = operands[0];
    assert(hasStaticShape(operand) && "Only Static shapes are accepted");

    Location loc = op->getLoc();
    Type outputType = *op->result_type_begin();
    assert(isRankedShapedType(outputType) && "Expected Ranked ShapedType");
    assert(mlir::cast<RankedTensorType>(operand.getType()).getElementType().isF32() && "Currently Only float32 is supported for input");

    //Exponential operation
    Value ElementwiseExpStableHLO = rewriter.create<stablehlo::ExpOp>(
        loc, op->getResultTypes(), op->getOperands());

    if(ElementwiseExpStableHLO == nullptr)
      return failure();

    RankedTensorType ExpOutputType = mlir::cast<RankedTensorType>(ElementwiseExpStableHLO.getType());

    //Converting negative indices to Postive indices
    int64_t axis = mlir::cast<ONNXSoftmaxOp>(*op).getAxis();
    if(axis < 0)
        axis = ExpOutputType.getRank() + axis;
    
    SmallVector<int64_t, 4> axes = {axis};
    //Sum of the all the exponents for the denominator
    SmallVector<int64_t> reducedShape = getReductionShape(ExpOutputType, axes, false);
    ShapedType ReducedShapeType = mlir::cast<ShapedType>(RankedTensorType::get(reducedShape, ExpOutputType.getElementType()));
    Value identity = rewriter.create<stablehlo::ConstantOp>(loc, rewriter.getZeroAttr(ExpOutputType.getElementType()));
    Value ReduceSum = computeReduceSum(loc, ElementwiseExpStableHLO, identity, reducedShape, axes, rewriter, false, ReducedShapeType);
    if(ReduceSum == nullptr)
      return failure();

    SmallVector <int64_t> broadcast_dims = getBroadcastDims(ElementwiseExpStableHLO, axes);
    Value BroadCastOp = rewriter.create<stablehlo::BroadcastInDimOp>(loc, ExpOutputType, ReduceSum, rewriter.getDenseI64ArrayAttr(broadcast_dims));
    if(BroadCastOp == nullptr)
      return failure();

    Value Softmax_output = rewriter.create<stablehlo::DivOp>(loc, ElementwiseExpStableHLO, BroadCastOp);
    if(Softmax_output == nullptr)
      return failure();

    rewriter.replaceOp(op, Softmax_output);
    return success();
  }
};
}

void populateLoweringONNXSoftmaxOpToStablehloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
        patterns.
            insert<ONNXSoftmaxOpLoweringToStablehlo>(ctx);
    }
}