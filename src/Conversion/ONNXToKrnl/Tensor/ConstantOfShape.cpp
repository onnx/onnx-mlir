/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ ConstantOfShape.cpp - Lowering ConstantOfShape Op -------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX ConstantOfShape Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXConstantOfShapeOpLowering : public ConversionPattern {
  ONNXConstantOfShapeOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter,
            mlir::ONNXConstantOfShapeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ONNXConstantOfShapeOpAdaptor operandAdaptor(operands);

    auto valueAttr = llvm::cast<ONNXConstantOfShapeOp>(op)
                         .getValue()
                         .value()
                         .cast<DenseElementsAttr>();

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = convertedType.cast<MemRefType>();
    Type elementType = memRefType.getElementType();
    ArrayRef<int64_t> outputShape = memRefType.getShape();
    size_t rank = outputShape.size();

    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder,
        MathBuilder>
        create(rewriter, loc);

    // Allocate memory for the output.
    Value alloc;
    if (hasAllConstantDimensions(memRefType))
      alloc = create.mem.alignedAlloc(memRefType);
    else {
      SmallVector<Value, 2> allocOperands;
      // Load dimensions from the input.
      for (decltype(rank) i = 0; i < rank; ++i) {
        if (outputShape[i] == ShapedType::kDynamic) {
          Value index = create.math.constantIndex(i);
          Value dim = create.krnl.load(operandAdaptor.getInput(), index);
          Value dimIndex = create.math.castToIndex(dim);
          allocOperands.emplace_back(dimIndex);
        }
      }
      // Allocate memory.
      alloc = create.mem.alignedAlloc(memRefType, allocOperands);
    }

    // Get the constant value from the attribute 'value'.
    Value constantVal;
    if (elementType.isa<IntegerType>()) {
      auto valueIt = valueAttr.getValues<IntegerAttr>().begin();
      auto valueInt = (*valueIt++).cast<IntegerAttr>().getInt();
      constantVal = create.math.constant(elementType, valueInt);
    } else if (elementType.isa<FloatType>()) {
      auto valueIt = valueAttr.getValues<FloatAttr>().begin();
      auto valueFloat = (*valueIt++).cast<FloatAttr>().getValueAsDouble();
      constantVal = create.math.constant(elementType, valueFloat);
    } else
      llvm_unreachable("unsupported element type");

    // Create a Krnl iterate if the output is not a scalar tensor.
    if (!hasAllScalarValues({alloc})) {
      IndexExprScope childScope(&rewriter, loc);
      ValueRange loopDef = create.krnl.defineLoops(rank);
      SmallVector<IndexExpr, 4> lbs(rank, LiteralIndexExpr(0));
      SmallVector<IndexExpr, 4> ubs;
      create.krnlIE.getShapeAsDims(alloc, ubs);
      create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
          [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
            createKrnl.store(constantVal, alloc, loopInd);
          });
    } else
      create.krnl.store(constantVal, alloc);

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXConstantOfShapeOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXConstantOfShapeOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
