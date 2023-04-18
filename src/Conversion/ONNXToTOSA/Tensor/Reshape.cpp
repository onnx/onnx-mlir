/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Reshape.cpp - Reshape Op ----------------------------===//
//
// Copyright (c) d-Matrix Inc. 2023
//
// =============================================================================
//
// This file lowers ONNX reshape operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXReshapeOpLoweringToTOSA : public OpConversionPattern<ONNXReshapeOp> {
public:
  using OpConversionPattern<ONNXReshapeOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(ONNXReshapeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    TosaBuilder tosaBuilder(rewriter, op.getLoc());

    ONNXConstantOp constOp = op.getShape().getDefiningOp<ONNXConstantOp>();
    if (!constOp)
      return op.emitError(
          "Dynamic ONNXReshapeOp lowering to tosa not supported");

    Optional<Attribute> valueAttr = constOp.getValue();
    if (!valueAttr.has_value())
      return op.emitError("Const operator needs a 'value' attribute");

    DenseElementsAttr valueDenseAttr =
        valueAttr.value().template cast<DenseElementsAttr>();
    std::vector<int64_t> shape;
    for (auto v : valueDenseAttr.getValues<APInt>())
      shape.push_back(v.getSExtValue());

    Value data = op.getData();
    auto tosaReshapeOp = tosaBuilder.reshape(data, shape);
    rewriter.replaceOp(op, {tosaReshapeOp});
    return success();
  }
};

} // namespace

void populateLoweringONNXReshapeOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXReshapeOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
