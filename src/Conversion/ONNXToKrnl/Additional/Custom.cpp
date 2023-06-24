/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- Custom.cpp - Lowering Custom Op--------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNXCustomOp to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXCustomOpLowering : public OpConversionPattern<ONNXCustomOp> {
  ONNXCustomOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXCustomOp customOp,
      ONNXCustomOpAdaptor operandAdaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = customOp.getOperation();
    Location loc = op->getLoc();
    ValueRange operands = operandAdaptor.getOperands();

    // Helper builders.
    MultiDialectBuilder<AffineBuilder, IndexExprBuilderForKrnl, KrnlBuilder,
        MemRefBuilder>
        create(rewriter, loc);
    IndexExprScope scope(create.krnlIE);

    // Get shape.
    // Commented out, waiting for the new PR
    //ONNXCustomOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    //shapeHelper.computeShapeAndAssertOnFailure();

    // Output types.
    SmallVector<Type, 4> outputMemRefTypes;
    SmallVector<Value, 4> allocOutputs;
    for(Type ty : op->getResultTypes()) {
      if(!hasStaticShape(ty)) {
	return emitError(loc, "Custom Op has dynamic output without shape inference pattern");
      }
      MemRefType outputMemRefType = typeConverter->convertType(ty).cast<MemRefType>();
      outputMemRefTypes.emplace_back(outputMemRefType);
      allocOutputs.emplace_back(create.mem.alignedAlloc(outputMemRefType));
    }

    // Handle the attributes: exclude the attributes used for analysis
    // function_name is passed explicitly. Others may include shape inference
    std::vector<std::string> excludeStrings={"function_name", "shape_infer_pattern", "inputs_for_infer", "output_element_type"};
    std::vector<std::string> attributeNames;
    for(NamedAttribute namedAttr : customOp->getAttrs()) {
      std::string attrName = namedAttr.getName().getValue().str();
      if(std::find(excludeStrings.begin(), excludeStrings.end(), attrName) == excludeStrings.end())
        attributeNames.push_back(attrName);
    }

    rewriter.create<KrnlCallOp>(loc, customOp.getFunctionName().str(), allocOutputs, op, operands, attributeNames);
     
    rewriter.replaceOp(op, allocOutputs);
    return success();
  }
};

void populateLoweringONNXCustomOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXCustomOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
