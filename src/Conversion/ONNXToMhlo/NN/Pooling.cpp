/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Pooling.cpp - Lowering Pooling Ops ------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Pooling Operators to Mhlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToMhlo/ONNXToMhloCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

// Builds body for reduce op by using the template binary op as the
// reducer op.
template <typename Op>
void buildReduceBody(Type elementType, Region *body, OpBuilder *builder) {
  OpBuilder::InsertionGuard guard(*builder);
  Block *block = builder->createBlock(body);
  // Block arguments are scalars of the given element type.
  Type type = RankedTensorType::get(/*shape=*/{}, elementType);
  Location loc = body->getLoc();
  block->addArguments({type, type}, SmallVector<Location, 2>(2, loc));
  Value reducer =
      builder->create<Op>(loc, block->getArgument(0), block->getArgument(1));
  builder->create<mhlo::ReturnOp>(loc, reducer);
}

template <typename Op>
void buildReduceBodyFor(Type elementType, Region *body, OpBuilder *builder) {}

template <>
void buildReduceBodyFor<ONNXMaxPoolSingleOutOp>(
    Type elementType, Region *body, OpBuilder *builder) {
  buildReduceBody<mhlo::MaxOp>(elementType, body, builder);
}

// Returns 1D 64-bit dense elements attribute padded with the given values.
static DenseIntElementsAttr getKernelAttr(ArrayRef<IndexExpr> values,
    Builder *builder, int64_t spatialOffset, int64_t defaultValue = 1) {
  SmallVector<int64_t> vectorValues(spatialOffset, defaultValue);
  int64_t size = values.size();
  for (int64_t i = 0; i < size; i++) {
    assert(values[i].isLiteral() && "kernel dim is not literal");
    vectorValues.push_back(values[i].getLiteral());
  }
  return builder->getI64VectorAttr(vectorValues);
}

void padVector(
    SmallVectorImpl<int64_t> &inputVector, int64_t numPad, int64_t value) {
  inputVector.insert(inputVector.begin(), numPad, value);
}

//===----------------------------------------------------------------------===//
// Template function that does pooling.
//
template <typename PoolOp, typename PoolOpAdaptor, typename PoolOpShapeHelper>
struct ONNXPoolOpLoweringToMhlo : public ConversionPattern {
  ONNXPoolOpLoweringToMhlo(MLIRContext *ctx)
      : ConversionPattern(PoolOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    PoolOpAdaptor operandAdaptor(operands, op->getAttrDictionary());
    Location loc = op->getLoc();
    PoolOp poolOp = llvm::cast<PoolOp>(op);

    // Get shape.
    PoolOpShapeHelper shapeHelper(&poolOp);
    LogicalResult shapecomputed = shapeHelper.computeShape(operandAdaptor);
    assert(succeeded(shapecomputed) && "Could not compute output shape");

    llvm::SmallVector<IndexExpr, 2> kernelShape = shapeHelper.kernelShape;
    llvm::SmallVector<int64_t, 2> strides = shapeHelper.strides;
    llvm::SmallVector<int64_t, 2> dilations = shapeHelper.dilations;
    DimsExpr outputDims = shapeHelper.dimsForOutput();

    // Type information about the input and result of this operation.
    Value inputOperand = operandAdaptor.X();
    RankedTensorType inputType =
        inputOperand.getType().dyn_cast_or_null<RankedTensorType>();
    if (inputType == nullptr)
      return failure();
    llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
    Type elemType = inputType.getElementType();
    Type outputType = *op->result_type_begin();
    int64_t spatialOffset = 2;
    int64_t rank = inputType.getRank();
    int64_t ceilMode = poolOp.ceil_mode();

    Value negInfinity = rewriter.create<mhlo::ConstantOp>(
        loc, rewriter.getFloatAttr(elemType,
                 APFloat::getInf(elemType.cast<FloatType>().getFloatSemantics(),
                     /*isNegative=*/true)));

    // paddings
    llvm::SmallVector<IndexExpr, 4> pads = shapeHelper.pads;
    llvm::StringRef padding = poolOp.auto_pad();
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
    mhlo::ReduceWindowOp reduce =
        rewriter.create<mhlo::ReduceWindowOp>(loc, outputType, inputOperand,
            negInfinity, getKernelAttr(kernelShape, &rewriter, spatialOffset),
            rewriter.getI64VectorAttr(strides),
            /*base_dilations=*/DenseIntElementsAttr(),
            /*window_dilations=*/rewriter.getI64VectorAttr(dilations),
            DenseIntElementsAttr::get(
                RankedTensorType::get({rank, 2}, rewriter.getI64Type()),
                flattenPaddings));
    buildReduceBodyFor<PoolOp>(elemType, &reduce.body(), &rewriter);
    rewriter.replaceOp(op, reduce->getResults());
    return success();
  }
};

} // namespace

void populateLoweringONNXPoolingOpToMhloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXPoolOpLoweringToMhlo<ONNXMaxPoolSingleOutOp,
      ONNXMaxPoolSingleOutOpAdaptor, ONNXMaxPoolSingleOutOpShapeHelper>>(ctx);
}

} // namespace onnx_mlir
