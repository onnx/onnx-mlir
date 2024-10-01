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

struct ONNXConstantOfShapeOpLowering
    : public OpConversionPattern<ONNXConstantOfShapeOp> {
  ONNXConstantOfShapeOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXConstantOfShapeOp constantOp,
      ONNXConstantOfShapeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = constantOp.getOperation();
    Location loc = ONNXLoc<ONNXConstantOfShapeOp>(op);

    auto valueAttr = mlir::cast<DenseElementsAttr>(adaptor.getValue().value());

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = mlir::cast<MemRefType>(convertedType);
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
          Value dim = create.krnl.load(adaptor.getInput(), index);
          Value dimIndex = create.math.castToIndex(dim);
          allocOperands.emplace_back(dimIndex);
        }
      }
      // Allocate memory.
      alloc = create.mem.alignedAlloc(memRefType, allocOperands);
    }

    // Get the constant value from the attribute 'value'.
    Value constantVal;
    if (mlir::isa<IntegerType>(elementType)) {
      auto valueIt = valueAttr.getValues<IntegerAttr>().begin();
      auto valueInt = mlir::cast<IntegerAttr>(*valueIt++).getInt();
      constantVal = create.math.constant(elementType, valueInt);
    } else if (mlir::isa<FloatType>(elementType)) {
      auto valueIt = valueAttr.getValues<FloatAttr>().begin();
      auto valueFloat = mlir::cast<FloatAttr>(*valueIt++).getValueAsDouble();
      constantVal = create.math.constant(elementType, valueFloat);
    } else
      llvm_unreachable("unsupported element type");

    // Create a Krnl iterate if the output is not a scalar tensor.
    if (!hasAllScalarValues({alloc})) {
      IndexExprScope childScope(&rewriter, loc);
      ValueRange loopDef = create.krnl.defineLoops(rank);
      SmallVector<IndexExpr, 4> lbs(rank, LitIE(0));
      SmallVector<IndexExpr, 4> ubs;
      create.krnlIE.getShapeAsDims(alloc, ubs);
      create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
          [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
            createKrnl.store(constantVal, alloc, loopInd);
          });
    } else
      create.krnl.store(constantVal, alloc);

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);

    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXConstantOfShapeOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXConstantOfShapeOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
