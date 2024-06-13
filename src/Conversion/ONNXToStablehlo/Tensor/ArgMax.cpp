/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- ArgMax.cpp - Lowering ArgMax Op -------------------===//
//
// Copyright 2022-2024
//
// =============================================================================
//
// This file lowers the ONNX ArgMax Operator to Stablehlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStablehlo/DialectBuilder.hpp"
#include "src/Conversion/ONNXToStablehlo/ONNXToStablehloCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

static void BuildArgmaxReductionBody(
    Type elementType, Type indexElementType, Region *body, OpBuilder *builder) {
  OpBuilder::InsertionGuard guard(*builder);
  Type inputType = RankedTensorType::get(/*shape=*/{}, elementType);
  Type indexType = RankedTensorType::get(/*shape=*/{}, indexElementType);
  Block *block = builder->createBlock(body);
  Location loc = body->getLoc();
  block->addArguments({inputType, indexType, inputType, indexType},
      SmallVector<Location, 4>(4, loc));

  Value lhsVal = block->getArgument(0);
  Value lhsIndex = block->getArgument(1);
  Value rhsVal = block->getArgument(2);
  Value rhsIndex = block->getArgument(3);

  ImplicitLocOpBuilder b(loc, *builder);
  Value compareDt = b.create<stablehlo::CompareOp>(
      lhsVal, rhsVal, stablehlo::ComparisonDirection::GE);
  Value selectedInput =
      b.create<stablehlo::SelectOp>(inputType, compareDt, lhsVal, rhsVal);

  Value compareEq = b.create<stablehlo::CompareOp>(
      lhsVal, rhsVal, stablehlo::ComparisonDirection::EQ);
  Value minIndex = b.create<stablehlo::MinOp>(lhsIndex, rhsIndex);
  Value minValIndex =
      b.create<stablehlo::SelectOp>(indexType, compareDt, lhsIndex, rhsIndex);
  Value selectedIndex = b.create<stablehlo::SelectOp>(
      indexType, compareEq, minIndex, minValIndex);
  Value returnValues[] = {selectedInput, selectedIndex};
  b.create<stablehlo::ReturnOp>(returnValues);
}

// ONNXArgMaxOp is mainly implemented using Stablehlo DynamicIotaOp and
// ReduceOp.
struct ONNXArgMaxOpLoweringToStablehlo : public ConversionPattern {
  ONNXArgMaxOpLoweringToStablehlo(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXArgMaxOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    // Gather info.
    Location loc = op->getLoc();
    ONNXArgMaxOpAdaptor operandAdaptor(operands);
    ONNXArgMaxOp argMaxOp = llvm::cast<ONNXArgMaxOp>(op);

    // Shape helper (not really used).
    IndexExprBuilderForStablehlo createIE(rewriter, loc);
    ONNXArgMaxOpShapeHelper shapeHelper(op, operands, &createIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    Type outputType = *op->result_type_begin();
    assert(isRankedShapedType(outputType) && "Expected Ranked ShapedType");
    ShapedType outputShapedType = mlir::cast<ShapedType>(outputType);
    Type indexElementType = outputShapedType.getElementType();
    Value indexInitValue = rewriter.create<stablehlo::ConstantOp>(
        loc, rewriter.getZeroAttr(indexElementType));

    // data input
    Value data = operandAdaptor.getData();
    assert(isRankedShapedType(data.getType()) &&
           "data must be ranked Shaped Type");
    ShapedType dataType = mlir::cast<ShapedType>(data.getType());
    Type elementType = dataType.getElementType();
    int64_t dataRank = dataType.getRank();

    // axis & keepdims attribute
    int64_t axis = argMaxOp.getAxis();
    assert(axis >= -dataRank && axis <= dataRank - 1);
    axis = axis >= 0 ? axis : (dataRank + axis);

    int64_t keepdims = argMaxOp.getKeepdims();
    bool isKeepdims = (keepdims == 1) ? true : false;

    Value initValue = rewriter.create<stablehlo::ConstantOp>(
        loc, rewriter.getFloatAttr(elementType,
                 APFloat::getInf(
                     mlir::cast<FloatType>(elementType).getFloatSemantics(),
                     /*isNegative=*/true)));
    RankedTensorType indexType =
        RankedTensorType::get(dataType.getShape(), indexElementType);

    IntegerAttr iotaDimension = IntegerAttr::get(rewriter.getI64Type(), axis);
    Value inputShapeOp = rewriter.create<shape::ShapeOfOp>(loc, data);
    Value indexValues = rewriter.create<stablehlo::DynamicIotaOp>(
        loc, indexType, inputShapeOp, iotaDimension);

    ArrayRef<int64_t> inputShape = dataType.getShape();
    SmallVector<int64_t> resultShape;
    for (int64_t i = 0; i < dataRank; i++)
      if (i != axis)
        resultShape.push_back(inputShape[i]);
    stablehlo::ReduceOp reduction = rewriter.create<stablehlo::ReduceOp>(loc,
        /*resultType0*/
        TypeRange{RankedTensorType::get(resultShape, elementType),
            RankedTensorType::get(resultShape, indexElementType)},
        /*inputs*/ ValueRange{data, indexValues},
        /*init_values*/ ValueRange{initValue, indexInitValue},
        /*dimensions*/ rewriter.getDenseI64ArrayAttr({axis}));

    BuildArgmaxReductionBody(
        elementType, indexElementType, &reduction.getBody(), &rewriter);

    Value result = reduction.getResult(1);
    if (isKeepdims) {
      if (outputShapedType.hasStaticShape())
        result = rewriter.create<stablehlo::ReshapeOp>(loc, outputType, result);
      else {
        SmallVector<Value> dims;
        for (int64_t i = 0; i < dataRank; i++) {
          if (i != axis) {
            Value dim =
                rewriter.create<shape::GetExtentOp>(loc, inputShapeOp, i);
            dims.push_back(dim);
          } else {
            Value dim = rewriter.create<arith::ConstantIndexOp>(loc, 1);
            dims.push_back(dim);
          }
        }
        Type outputShapeType =
            RankedTensorType::get({dataRank}, rewriter.getIndexType());
        Value newShapeValue = rewriter.create<shape::FromExtentsOp>(loc, dims);
        newShapeValue = rewriter.create<shape::ToExtentTensorOp>(
            loc, outputShapeType, newShapeValue);
        result = rewriter.create<stablehlo::DynamicReshapeOp>(
            loc, outputType, result, newShapeValue);
      }
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void populateLoweringONNXArgMaxOpToStablehloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXArgMaxOpLoweringToStablehlo>(ctx);
}

} // namespace onnx_mlir
