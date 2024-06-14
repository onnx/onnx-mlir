/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Pooling.cpp - Lowering Pooling Ops ------------------===//
//
// Copyright 2022-2024
//
// =============================================================================
//
// This file lowers the ONNX Pooling Operators to Stablehlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToStablehlo/DialectBuilder.hpp"
#include "src/Conversion/ONNXToStablehlo/ONNXToStablehloCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

static Value createInitialValueForPoolingOp(
    Operation *op, Type elemType, ConversionPatternRewriter &rewriter) {
  Location loc = op->getLoc();
  if (isa<ONNXMaxPoolSingleOutOp>(op)) {
    // returns negative infinity
    return rewriter.create<stablehlo::ConstantOp>(loc,
        rewriter.getFloatAttr(elemType,
            APFloat::getInf(mlir::cast<FloatType>(elemType).getFloatSemantics(),
                /*isNegative=*/true)));
  }
  if (isa<ONNXAveragePoolOp>(op)) {
    // returns negative infinity
    return rewriter.create<stablehlo::ConstantOp>(
        loc, rewriter.getFloatAttr(elemType,
                 APFloat::getZero(
                     mlir::cast<FloatType>(elemType).getFloatSemantics(),
                     /*isNegative=*/false)));
  }
  op->emitError("unimplemented lowering for onnx pooling op\n");
  return nullptr;
}

// Builds body for reduce op by using the template binary op as the
// reducer op.
template <typename Op>
void buildReduceBody(Type elementType, Region *body, OpBuilder *builder) {
  OpBuilder::InsertionGuard guard(*builder);
  Block *block = builder->createBlock(body);
  // Block arguments are scalars of the given element type.
  auto type = RankedTensorType::get(/*shape=*/{}, elementType);
  Location loc = body->getLoc();
  block->addArguments({type, type}, SmallVector<Location, 2>(2, loc));
  Value reducer =
      builder->create<Op>(loc, block->getArgument(0), block->getArgument(1));
  builder->create<stablehlo::ReturnOp>(loc, reducer);
}

template <typename Op>
void buildReduceBodyFor(Type elementType, Region *body, OpBuilder *builder);

template <>
void buildReduceBodyFor<ONNXMaxPoolSingleOutOp>(
    Type elementType, Region *body, OpBuilder *builder) {
  buildReduceBody<stablehlo::MaxOp>(elementType, body, builder);
}

template <>
void buildReduceBodyFor<ONNXAveragePoolOp>(
    Type elementType, Region *body, OpBuilder *builder) {
  buildReduceBody<stablehlo::AddOp>(elementType, body, builder);
}

// Returns 1D 64-bit dense elements attribute padded with the given values.
static DenseI64ArrayAttr getKernelAttr(ArrayRef<IndexExpr> values,
    Builder *builder, int64_t spatialOffset, int64_t defaultValue = 1) {
  SmallVector<int64_t> vectorValues(spatialOffset, defaultValue);
  int64_t size = values.size();
  for (int64_t i = 0; i < size; i++) {
    assert(values[i].isLiteral() && "kernel dim is not literal");
    vectorValues.push_back(values[i].getLiteral());
  }
  return builder->getDenseI64ArrayAttr(vectorValues);
}

void padVector(
    SmallVectorImpl<int64_t> &inputVector, int64_t numPad, int64_t value) {
  inputVector.insert(inputVector.begin(), numPad, value);
}

//===----------------------------------------------------------------------===//
// Template function that does pooling.
//
template <typename PoolOp, typename PoolOpAdaptor, typename PoolOpShapeHelper>
struct ONNXPoolOpLoweringToStablehlo : public ConversionPattern {
  ONNXPoolOpLoweringToStablehlo(MLIRContext *ctx)
      : ConversionPattern(PoolOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    PoolOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
    Location loc = op->getLoc();
    PoolOp poolOp = llvm::cast<PoolOp>(op);

    // Get shape.
    IndexExprBuilderForStablehlo createStablehloIE(rewriter, loc);
    PoolOpShapeHelper shapeHelper(op, operands, &createStablehloIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    llvm::SmallVector<IndexExpr, 2> kernelShape = shapeHelper.kernelShape;
    llvm::SmallVector<int64_t, 2> strides = shapeHelper.strides;
    llvm::SmallVector<int64_t, 2> dilations = shapeHelper.dilations;
    DimsExpr outputDims = shapeHelper.getOutputDims();

    // Type information about the input and result of this operation.
    Value inputOperand = operandAdaptor.getX();
    RankedTensorType inputType =
        mlir::dyn_cast_or_null<RankedTensorType>(inputOperand.getType());
    if (inputType == nullptr)
      return failure();
    llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
    Type elemType = inputType.getElementType();
    Type outputType = *op->result_type_begin();
    int64_t spatialOffset = 2;
    int64_t rank = inputType.getRank();
    int64_t ceilMode = poolOp.getCeilMode();

    Value initVal = createInitialValueForPoolingOp(op, elemType, rewriter);
    if (initVal == nullptr)
      return failure();

    // paddings
    llvm::SmallVector<IndexExpr, 4> pads = shapeHelper.pads;
    llvm::StringRef padding = poolOp.getAutoPad();
    int64_t spatialRank = rank - spatialOffset;
    SmallVector<int64_t> flattenPaddings;
    for (int64_t i = 0; i < 2 * spatialOffset; i++)
      flattenPaddings.push_back(0);
    bool needPadding = (padding == "NOTSET") && (ceilMode == 1);
    for (int64_t i = 0; i < spatialRank; i++) {
      if (!needPadding) {
        flattenPaddings.push_back(pads[i].getLiteral());
        flattenPaddings.push_back(pads[i + spatialRank].getLiteral());
      } else {
        int64_t kdTerm = (kernelShape[i].getLiteral() - 1) * dilations[i] + 1;
        int64_t padFront = pads[i].getLiteral();
        int64_t padBack =
            (outputDims[i + spatialOffset].getLiteral() - 1) * strides[i] +
            kdTerm - inputShape[i + spatialOffset] - padFront;
        flattenPaddings.push_back(padFront);
        flattenPaddings.push_back(padBack);
      }
    }

    padVector(strides, spatialOffset, 1);
    padVector(dilations, spatialOffset, 1);
    stablehlo::ReduceWindowOp reduce =
        rewriter.create<stablehlo::ReduceWindowOp>(loc, outputType,
            inputOperand, initVal,
            getKernelAttr(kernelShape, &rewriter, spatialOffset),
            rewriter.getDenseI64ArrayAttr(strides),
            /*base_dilations=*/DenseI64ArrayAttr(),
            /*window_dilations=*/rewriter.getDenseI64ArrayAttr(dilations),
            DenseIntElementsAttr::get(
                RankedTensorType::get({rank, 2}, rewriter.getI64Type()),
                flattenPaddings));
    buildReduceBodyFor<PoolOp>(elemType, &reduce.getBody(), &rewriter);

    if (isa<ONNXAveragePoolOp>(op)) {
      Value reduceResult = reduce.getResult(0);
      int64_t countIncludePad =
          llvm::cast<ONNXAveragePoolOp>(op).getCountIncludePad();
      if (countIncludePad) {
        // Use kernel size as the divisor
        int64_t kernelSize = 1;
        for (int64_t i = 0; i < spatialRank; i++) {
          kernelSize *= kernelShape[i].getLiteral();
        }
        Value divisor = getShapedFloat(loc, rewriter, kernelSize, reduceResult);
        Value divResult = rewriter.create<stablehlo::DivOp>(
            loc, outputType, reduceResult, divisor);
        rewriter.replaceOp(op, divResult);
      } else {
        // Use another stablehlo.ReduceWindowOp to get the divisor
        Value one = getShapedFloat(loc, rewriter, 1.0, inputOperand);
        stablehlo::ReduceWindowOp reduceDivisor =
            rewriter.create<stablehlo::ReduceWindowOp>(loc, outputType, one,
                initVal, getKernelAttr(kernelShape, &rewriter, spatialOffset),
                rewriter.getDenseI64ArrayAttr(strides),
                /*base_dilations=*/DenseI64ArrayAttr(),
                /*window_dilations=*/rewriter.getDenseI64ArrayAttr(dilations),
                DenseIntElementsAttr::get(
                    RankedTensorType::get({rank, 2}, rewriter.getI64Type()),
                    flattenPaddings));
        buildReduceBodyFor<ONNXAveragePoolOp>(
            elemType, &reduceDivisor.getBody(), &rewriter);
        Value divResult = rewriter.create<stablehlo::DivOp>(
            loc, outputType, reduceResult, reduceDivisor.getResult(0));
        rewriter.replaceOp(op, divResult);
      }
    } else {
      rewriter.replaceOp(op, reduce->getResults());
    }
    return success();
  }
};

} // namespace

void populateLoweringONNXPoolingOpToStablehloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXPoolOpLoweringToStablehlo<ONNXMaxPoolSingleOutOp,
      ONNXMaxPoolSingleOutOpAdaptor, ONNXMaxPoolSingleOutOpShapeHelper>>(ctx);
  patterns.insert<ONNXPoolOpLoweringToStablehlo<ONNXAveragePoolOp,
      ONNXAveragePoolOpAdaptor, ONNXAveragePoolOpShapeHelper>>(ctx);
}

} // namespace onnx_mlir
