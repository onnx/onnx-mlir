//===----------- Constant.cpp - Lowering Constant Op to Linalg ------------===//
//
// Copyright 2026 HighTec-EDV Systeme Gmbh Authors.
//
// =============================================================================
//
// This file lowers the ONNX Constant Operator to Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "src/Conversion/ONNXToLinalg/ONNXToLinalgCommon.hpp"
#include "src/Dialect/ONNX/OnnxElementsAttrBuilder.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXConstantOpLoweringToLinalg
    : public OpRewritePattern<ONNXConstantOp> {
  ONNXConstantOpLoweringToLinalg(
      MLIRContext *ctx, const std::string &linalgOps, bool useLinalgPath)
      : OpRewritePattern<ONNXConstantOp>(ctx), linalgOps(linalgOps),
        useLinalgPath(useLinalgPath) {}

  LogicalResult matchAndRewrite(
      ONNXConstantOp constantOp, PatternRewriter &rewriter) const final {
    if (!shouldConvertToLinalg(
            constantOp.getOperation(), linalgOps, useLinalgPath)) {
      return rewriter.notifyMatchFailure(
          constantOp, "operation not selected for Linalg conversion");
    }
    auto valueAttr = dyn_cast_or_null<TypedAttr>(constantOp.getValueAttr());
    if (!valueAttr)
      return rewriter.notifyMatchFailure(
          constantOp, "expected normalized dense value attribute");

    if (valueAttr.getType() != constantOp.getOutput().getType())
      return rewriter.notifyMatchFailure(
          constantOp, "constant value and result types do not match");

    Location loc = constantOp->getLoc();
    auto elementsAttr = dyn_cast<ElementsAttr>(valueAttr);
    ShapedType shapedType = elementsAttr.getShapedType();
    auto integerType = dyn_cast<IntegerType>(shapedType.getElementType());

    Value result;
    // Arith constants only support signless integer element types. Create the
    // constant with a signless element type and then cast it back to the ONNX
    // signedness while preserving the bit pattern.
    if (integerType && !integerType.isSignless()) {
      IntegerType signlessElementType =
          IntegerType::get(rewriter.getContext(), integerType.getWidth());

      OnnxElementsAttrBuilder elementsBuilder{rewriter.getContext()};
      ElementsAttr signlessElementsAttr = elementsBuilder.castToIntElementType(
          elementsAttr, signlessElementType);
      DenseElementsAttr signlessAttr =
          ElementsAttrBuilder::toDenseElementsAttr(signlessElementsAttr);

      Value signlessConstant =
          arith::ConstantOp::create(rewriter, loc, signlessAttr);
      result = UnrealizedConversionCastOp::create(
          rewriter, loc, shapedType, signlessConstant)
                   ->getResult(0);

    } else {
      result = arith::ConstantOp::create(rewriter, loc, valueAttr);
    }

    rewriter.replaceOp(constantOp, result);
    return success();
  }

private:
  std::string linalgOps;
  bool useLinalgPath;
};

void populateLoweringONNXConstantOpToLinalgPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx,
    const std::string &linalgOps, bool useLinalgPath) {
  patterns.add<ONNXConstantOpLoweringToLinalg>(ctx, linalgOps, useLinalgPath);
}
} // namespace onnx_mlir
