/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- DLF16ToF32.cpp - ZHigh Operations -------------------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/ShapeHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/DLF16ToF32/ONNXZHighDLF16ToF32.inc"

//===----------------------------------------------------------------------===//
// Delay zhigh.DLF16ToF32 as long as possible, and finally it can be cancelled
// when it meets zhigh.F32ToDLF16.
//
// This pattern works on ONNX operations that have one main input and other
// inputs are, for example, indices or shapes, etc.
//===----------------------------------------------------------------------===//
template <typename ONNX_OP>
class DelayDLF16ToF32Pattern : public OpRewritePattern<ONNX_OP> {
public:
  using OpRewritePattern<ONNX_OP>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNX_OP onnxOp, PatternRewriter &rewriter) const override {
    Operation *op = onnxOp.getOperation();

    // Match: ONNX_Op (ZHighDLF16ToF32Op X), args, attrs
    Value onnxInput = op->getOperand(0);
    zhigh::ZHighDLF16ToF32Op dlf16ToF32Op =
        onnxInput.getDefiningOp<zhigh::ZHighDLF16ToF32Op>();
    if (!dlf16ToF32Op)
      return failure();
    Value X = dlf16ToF32Op.getOperand();

    // Rewrite
    //  ONNX_Op (ZHighDLF16ToF32Op X), args, attrs
    // into
    //  ZHighDLF16ToF32Op (ONNX_Op X, args, attrs)

    // Build a new ONNXOp accepting X as input.
    Operation *clonedONNXOp = rewriter.clone(*op);
    clonedONNXOp->setOperand(0, X);
    for (int64_t i = 0; i < op->getNumResults(); ++i) {
      // Set elementType of the cloned op to f16.
      ShapedType f16Type = dyn_cast<ShapedType>(op->getResult(i).getType())
                               .clone(rewriter.getF16Type());
      clonedONNXOp->getResult(i).setType(f16Type);
    }

    // Build a new ZHighDLF16ToF32Op.
    SmallVector<Value, 4> newResults;
    for (int64_t i = 0; i < op->getNumResults(); ++i) {
      Operation *clonedZHighOp = rewriter.clone(*dlf16ToF32Op.getOperation());
      clonedZHighOp->setOperand(0, clonedONNXOp->getResult(i));
      Value result = clonedZHighOp->getResult(0);
      result.setType(op->getResult(i).getType());
      newResults.emplace_back(result);
    }

    rewriter.replaceOp(onnxOp, newResults);
    return success();
  }
};
} // end anonymous namespace

namespace onnx_mlir {
namespace zhigh {

//===----------------------------------------------------------------------===//
// Custom builders
//===----------------------------------------------------------------------===//

void ZHighDLF16ToF32Op::build(
    OpBuilder &builder, OperationState &state, Value input) {
  Type elementType = builder.getF32Type();
  Type resType = UnrankedTensorType::get(elementType);

  if (auto inType = mlir::dyn_cast<RankedTensorType>(input.getType()))
    resType = RankedTensorType::get(inType.getShape(), elementType);

  build(builder, state, resType, input);
}

//===----------------------------------------------------------------------===//
// Shape inference
//===----------------------------------------------------------------------===//

LogicalResult ZHighDLF16ToF32Op::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}

//===----------------------------------------------------------------------===//
// Canonicalization patterns
//===----------------------------------------------------------------------===//

void ZHighDLF16ToF32Op::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<ConversionRemovalPattern>(context);

  // This pattern works on ONNX operations that have one main input and other
  // inputs are, for example, indices or shapes, etc.
  results.insert<DelayDLF16ToF32Pattern<ONNXExpandOp>>(context);
  results.insert<DelayDLF16ToF32Pattern<ONNXFlattenOp>>(context);
  results.insert<DelayDLF16ToF32Pattern<ONNXGatherOp>>(context);
  results.insert<DelayDLF16ToF32Pattern<ONNXReshapeOp>>(context);
  results.insert<DelayDLF16ToF32Pattern<ONNXSliceOp>>(context);
  results.insert<DelayDLF16ToF32Pattern<ONNXSplitOp>>(context);
  results.insert<DelayDLF16ToF32Pattern<ONNXSqueezeOp>>(context);
  results.insert<DelayDLF16ToF32Pattern<ONNXTransposeOp>>(context);
  results.insert<DelayDLF16ToF32Pattern<ONNXUnsqueezeOp>>(context);

  results.insert<DimDLF16ToF32RemovalPattern>(context);
}
} // namespace zhigh
} // namespace onnx_mlir
