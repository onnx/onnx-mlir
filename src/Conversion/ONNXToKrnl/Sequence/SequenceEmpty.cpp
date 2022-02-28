/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------SequenceEmpty.cpp - Lowering SequenceEmpty Op----------------------=== //
//
// Copyright 2020-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX SequenceEmpty Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

struct ONNXSequenceEmptyOpLowering : public ConversionPattern {
  ONNXSequenceEmptyOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXSequenceEmptyOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ONNXSequenceEmptyOp sequenceEmptyOp = cast<ONNXSequenceEmptyOp>(op);

    // Get element type
    //auto elementType = convertONNXTypeToMLIRType(sequenceEmptyOp.dtype());

    // Default type, F32, for the Optional attribute
    Type elementType = rewriter.getF32Type();
    if (sequenceEmptyOp.dtypeAttr())
      elementType = convertONNXTypeToMLIRType(rewriter, (onnx::TensorProto_DataType)sequenceEmptyOp.dtypeAttr().getValue().getSExtValue());

    // Get element shape
    int64_t rank = -1;

    // Will add an attribute for shape in SequenceEmpty

    Type seqElementType;
    if (rank == -1) {
      // unranked memref
      seqElementType = UnrankedMemRefType::get(elementType, 0);
    } else {
      // temporarily assume all dim unknown
      SmallVector<int64_t, 1> dims(-1, rank);
      llvm::ArrayRef<int64_t> shape(dims.data(), dims.size());
      seqElementType = MemRefType::get(shape, elementType);
    }
      
    SmallVector<int64_t, 1> dims;
    dims.emplace_back(0);
    llvm::ArrayRef<int64_t> shape(dims.data(), dims.size());
 
    //auto itemType = UnrankedMemRefType::get(rewriter.getF32Type(),0);
    //auto outputTensorType = RankedTensorType::get(shape, rewriter.getIntegerType(64));
    //auto outputMemRefType = convertToMemRefType(outputTensorType);
    MemRefType outputMemRefType = MemRefType::get(shape, seqElementType);
    bool insertDealloc = checkInsertDealloc(op);
    Value alloc = insertAllocAndDealloc(outputMemRefType, loc, rewriter, insertDealloc);
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXSequenceEmptyOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXSequenceEmptyOpLowering>(typeConverter, ctx);
}
