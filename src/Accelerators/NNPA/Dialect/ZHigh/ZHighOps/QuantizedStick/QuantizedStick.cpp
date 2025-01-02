/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Stick.cpp - ZHigh Operations ----------------------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/Compiler/NNPACompilerOptions.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace zhigh {

//===----------------------------------------------------------------------===//
// Custom builders
//===----------------------------------------------------------------------===//

void ZHighQuantizedStickOp::build(OpBuilder &builder, OperationState &state,
    Value input, Value recScale, Value offset, StringAttr layout,
    StringAttr qtype, IntegerAttr symMode) {
  // Quantized type.
  auto quantizedType = convertStringAttrToZTensorQuantizedType(qtype);

  Type resElementType;
  if (quantizedType == ZTensorEncodingAttr::QuantizedType::DLFLOAT16)
    resElementType = builder.getF16Type();
  else if (quantizedType == ZTensorEncodingAttr::QuantizedType::INT8)
    resElementType = builder.getI8Type();
  else if (quantizedType == ZTensorEncodingAttr::QuantizedType::WEIGHTS)
    resElementType = builder.getI8Type();
  else
    llvm_unreachable("Unsupported quantized transform type");

  Type resType = builder.getNoneType();
  if (!mlir::isa<NoneType>(input.getType())) {
    ShapedType inputType = mlir::cast<ShapedType>(input.getType());
    int64_t rank = -1;
    if (inputType.hasRank()) {
      rank = inputType.getRank();
      ZTensorEncodingAttr::DataLayout dataLayout;
      if (layout)
        dataLayout = convertStringAttrToZTensorDataLayout(layout);
      else {
        dataLayout = getZTensorDataLayoutByRank(rank);
        // Create a layout attribute.
        layout = convertZTensorDataLayoutToStringAttr(builder, dataLayout);
      }
      // Compute shape.
      ArrayRef<int64_t> inputShape = inputType.getShape();
      SmallVector<int64_t, 4> resShape(inputShape.begin(), inputShape.end());
      resType = RankedTensorType::get(resShape, resElementType,
          ZTensorEncodingAttr::get(
              builder.getContext(), dataLayout, quantizedType));
    } else {
      resType = UnrankedTensorType::get(resElementType);
    }
  }
  RankedTensorType scalarTensorF32Type =
      RankedTensorType::get({}, builder.getF32Type());
  build(builder, state, {resType, scalarTensorF32Type, scalarTensorF32Type},
      input, recScale, offset, layout, qtype, symMode);
}

void ZHighQuantizedStickOp::build(OpBuilder &builder, OperationState &state,
    Value input, Value recScale, Value offset, StringAttr layout,
    StringAttr qtype) {
  // By default, sym_mode is off.
  IntegerAttr symMode = builder.getIntegerAttr(builder.getI64Type(), 0);
  build(builder, state, input, recScale, offset, layout, qtype, symMode);
}

//===----------------------------------------------------------------------===//
// ShapeHelper
//===----------------------------------------------------------------------===//

LogicalResult ZHighQuantizedStickOpShapeHelper::computeShape() {
  ZHighQuantizedStickOp::Adaptor operandAdaptor(operands);
  Value input = operandAdaptor.getIn();

  // Output dims of result.
  DimsExpr outputDims;

  // Get operands and bounds.
  SmallVector<IndexExpr, 4> inputDims;
  createIE->getShapeAsDims(input, inputDims);
  int64_t rank = inputDims.size();

  for (int64_t i = 0; i < rank; ++i)
    outputDims.emplace_back(inputDims[i]);

  // Save the final result.
  setOutputDims(outputDims);
  return success();
}

//===----------------------------------------------------------------------===//
// Shape inference
//===----------------------------------------------------------------------===//

LogicalResult ZHighQuantizedStickOp::inferShapes(
    std::function<void(mlir::Region &)> doShapeInference) {
  Operation *op = getOperation();
  OpBuilder builder(op);

  Value input = getIn();
  if (isa<NoneType>(input.getType()) || !hasRankedType(input))
    return success();

  auto inputType = mlir::cast<RankedTensorType>(input.getType());
  StringAttr layout = getLayoutAttr();
  StringAttr qtype = getQuantizedTypeAttr();
  int64_t rank = inputType.getRank();

  ZTensorEncodingAttr::DataLayout dataLayout;
  if (layout)
    dataLayout = convertStringAttrToZTensorDataLayout(layout);
  else
    dataLayout = getZTensorDataLayoutByRank(rank);
  ZTensorEncodingAttr::QuantizedType quantizedType =
      convertStringAttrToZTensorQuantizedType(qtype);
  auto encoding =
      ZTensorEncodingAttr::get(this->getContext(), dataLayout, quantizedType);

  Type resElementType;
  if (quantizedType == ZTensorEncodingAttr::QuantizedType::DLFLOAT16)
    resElementType = builder.getF16Type();
  else if (quantizedType == ZTensorEncodingAttr::QuantizedType::INT8)
    resElementType = builder.getI8Type();
  else if (quantizedType == ZTensorEncodingAttr::QuantizedType::WEIGHTS)
    resElementType = builder.getI8Type();
  else
    llvm_unreachable("Unsupported quantized transform type");

  ZHighQuantizedStickOpShapeHelper shapeHelper(getOperation());
  shapeHelper.computeShapeAndAssertOnFailure();
  SmallVector<int64_t, 4> outputDims;
  IndexExpr::getShape(shapeHelper.getOutputDims(0), outputDims);

  updateType(op, getResults()[0], outputDims, resElementType, encoding);
  getResults()[1].setType(RankedTensorType::get({}, builder.getF32Type()));
  getResults()[2].setType(RankedTensorType::get({}, builder.getF32Type()));
  return success();
}

//===----------------------------------------------------------------------===//
// Canonicalization patterns
//===----------------------------------------------------------------------===//

class QuantizedStickUnstickRemovalPattern
    : public OpRewritePattern<ZHighQuantizedStickOp> {
public:
  using OpRewritePattern<ZHighQuantizedStickOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ZHighQuantizedStickOp qStickOp,
      PatternRewriter &rewriter) const override {
    Location loc = qStickOp.getLoc();
    Value input = qStickOp.getIn();
    StringAttr quantizedType = qStickOp.getQuantizedTypeAttr();

    // ZHighQuantizedStickOp's type is dlfloat16.
    if (!quantizedType.getValue().equals_insensitive(QTYPE_DLFLOAT16))
      return failure();

    // ZHighQuantizedStickOp's input was defined by ZHighUnstickOp.
    auto unstickOp = input.getDefiningOp<ZHighUnstickOp>();
    if (!unstickOp)
      return failure();
    // Stickified input's layout is 3D, 2DS or 3DS.
    Value stickInput = unstickOp.getIn();
    StringAttr stickLayout =
        getZTensorLayoutAttr(rewriter, stickInput.getType());
    if (!(stickLayout.getValue().equals_insensitive("3D") ||
            stickLayout.getValue().equals_insensitive("2DS") ||
            stickLayout.getValue().equals_insensitive("3DS")))
      return failure();
    // Match layout.
    StringAttr qStickLayout = qStickOp.getLayoutAttr();
    if (stickLayout != qStickLayout)
      return failure();

    // Rewrite by passing the stickified input directly to ZHighQuantizedStick.
    ZHighQuantizedStickOp newQStickOp = rewriter.create<ZHighQuantizedStickOp>(
        loc, stickInput, qStickOp.getInRecScale(), qStickOp.getInOffset(),
        qStickOp.getLayoutAttr(), qStickOp.getQuantizedTypeAttr());
    rewriter.replaceOp(qStickOp, newQStickOp.getResults());
    return success();
  }
};

void ZHighQuantizedStickOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  if (nnpaUseDynamicQuantizeLinearOnCPUForScaleOffset)
    results.insert<QuantizedStickUnstickRemovalPattern>(context);
}

} // namespace zhigh
} // namespace onnx_mlir
