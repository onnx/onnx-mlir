//===---------------- Reshape.cpp - Lowering Reshape Op -------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Reshape Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

Value getDimOrConstant(ConversionPatternRewriter &rewriter, Location loc,
    Value operand, int64_t axis) {
  ArrayRef<int64_t> shape = operand.getType().cast<ShapedType>().getShape();
  Value dimVal;
  if (shape[axis] < 0) {
    Value dim = rewriter.create<DimOp>(loc, operand, axis);
    dimVal =
        rewriter.create<IndexCastOp>(loc, dim, rewriter.getIntegerType(64));
  } else {
    dimVal =
        emitConstantOp(rewriter, loc, rewriter.getIntegerType(64), shape[axis]);
  }
  return dimVal;
}

struct ONNXReshapeOpLowering : public ConversionPattern {
  ONNXReshapeOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXReshapeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXReshapeOpAdaptor operandAdaptor(operands);
    ONNXReshapeOp reshapeOp = dyn_cast_or_null<ONNXReshapeOp>(op);

    auto loc = op->getLoc();
    Value data = operandAdaptor.data();
    Value shape = operandAdaptor.shape();
    auto dataShape = data.getType().cast<MemRefType>().getShape();
    // If shape tensor was be promoted to attribute, get its values from the
    // attribute.
    SmallVector<int64_t, 4> shapeAttrValues;
    DenseElementsAttr shapeAttr =
        reshapeOp.getAttr("shape").dyn_cast_or_null<DenseElementsAttr>();
    if (shapeAttr) {
      auto shapeAttrIt = shapeAttr.getValues<IntegerAttr>().begin();
      auto itEnd = shapeAttr.getValues<IntegerAttr>().end();
      for (; shapeAttrIt != itEnd;)
        shapeAttrValues.emplace_back(
            (*shapeAttrIt++).cast<IntegerAttr>().getInt());
    }

    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    auto memRefShape = memRefType.getShape();
    Value alloc;

    // Compute size in bytes using the input tensor.
    Value tensorSize = emitConstantOp(rewriter, loc,
        rewriter.getIntegerType(64), getMemRefEltSizeInBytes(memRefType));
    for (int i = 0; i < dataShape.size(); ++i) {
      Value dimVal = getDimOrConstant(rewriter, loc, data, i);
      tensorSize = rewriter.create<MulIOp>(loc, tensorSize, dimVal);
    }

    bool insertDealloc = checkInsertDealloc(op);
    if (hasAllConstantDimensions(memRefType)) {
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    } else {
      // Calculate the unknown output dimensions using given shape information.
      // Shape information is store in the shape input. However, the shape input
      // will be promoted to attribute if it is a constant. Thus, we need to
      // check both cases.
      //
      // Dimensions in the shape information can be: 0, -1, or N.
      // - If a dimension is 0, the actual dimension value is taken from the
      // input tensor.
      //
      // - If a dimension is -1, we compute its actual dimension value from the
      // other dimensions. But we don't have enough information about the other
      // dimensions at this point. So, we need to scan the shape first to
      // calculate reduction of all of the dimensions. If the reduction is
      // negative, then the shape array contains a negative dimension.
      // Otherwise, the reduction is the same as the one computed from the input
      // tensor.
      Value tensorSizeFromShape = emitConstantOp(rewriter, loc,
          rewriter.getIntegerType(64), getMemRefEltSizeInBytes(memRefType));
      SmallVector<Value, 4> DimInfo;
      for (int i = 0; i < memRefShape.size(); ++i) {
        // If a dimension is N (N != 0 && N != -1), use it.
        // If a dimension is - 1, it will be computed from the other dimensions.
        // But we don't have enough information about the other dimensions at
        // this point. So, we let it as it is (-1), and compute it later.
        Value loadedVal;
        if (!shapeAttrValues.empty()) {
          loadedVal = emitConstantOp(
              rewriter, loc, rewriter.getIntegerType(64), shapeAttrValues[i]);
        } else {
          Value index =
              emitConstantOp(rewriter, loc, rewriter.getIndexType(), i);
          loadedVal = rewriter.create<KrnlLoadOp>(loc, shape, index);
        }

        // The output dimension cannot be 0 if the output dimension position is
        // out of the input dimension position.
        if (i < dataShape.size()) {
          // If a dimension is 0, the actual dimension value is taken from the
          // input tensor.
          if (!shapeAttrValues.empty()) {
            if (shapeAttrValues[i] == 0)
              loadedVal = getDimOrConstant(rewriter, loc, data, i);
          } else {
            auto dimVal = getDimOrConstant(rewriter, loc, data, i);
            auto zero =
                emitConstantOp(rewriter, loc, rewriter.getIntegerType(64), 0);
            auto isZero = rewriter.create<CmpIOp>(
                loc, CmpIPredicate::eq, loadedVal, zero);
            loadedVal =
                rewriter.create<SelectOp>(loc, isZero, dimVal, loadedVal);
          }
        }

        // Compute tensor size using shape information.
        tensorSizeFromShape =
            rewriter.create<MulIOp>(loc, tensorSizeFromShape, loadedVal);
        // Store intermediate results to use later.
        DimInfo.emplace_back(loadedVal);
      }
      // Reverse tensorSizeFromShape since it is negative if the shape array has
      // a negative dimension. This is safe since we only use it to compute the
      // actual value for the negative dimension.
      auto zero = emitConstantOp(rewriter, loc, rewriter.getIntegerType(64), 0);
      tensorSizeFromShape =
          rewriter.create<SubIOp>(loc, zero, tensorSizeFromShape);

      // Obtain operands for AllocOp.
      SmallVector<Value, 4> allocOperands;
      auto negOne =
          emitConstantOp(rewriter, loc, rewriter.getIntegerType(64), -1);

      for (int i = 0; i < memRefShape.size(); ++i) {
        if (memRefShape[i] != -1)
          continue;
        auto dimVal = DimInfo[i];
        auto isNegOne =
            rewriter.create<CmpIOp>(loc, CmpIPredicate::eq, dimVal, negOne);
        // If dimension is -1, compute its value from the other dimensions.
        auto actualDimVal =
            rewriter.create<SignedDivIOp>(loc, tensorSize, tensorSizeFromShape);
        auto loadedVal =
            rewriter.create<SelectOp>(loc, isNegOne, actualDimVal, dimVal);
        allocOperands.push_back(rewriter.create<IndexCastOp>(
            loc, loadedVal, rewriter.getIndexType()));
      }
      AllocOp allocateMemref =
          rewriter.create<AllocOp>(loc, memRefType, allocOperands);

      // Make sure to allocate at the beginning of the block if
      // all dimensions are known.
      auto *parentBlock = allocateMemref.getOperation()->getBlock();
      if (insertDealloc) {
        auto dealloc = rewriter.create<DeallocOp>(loc, allocateMemref);
        dealloc.getOperation()->moveBefore(&parentBlock->back());
      }

      alloc = allocateMemref;
    }

    rewriter.create<KrnlMemcpyOp>(loc, alloc, data, tensorSize);
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXReshapeOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXReshapeOpLowering>(ctx);
}
