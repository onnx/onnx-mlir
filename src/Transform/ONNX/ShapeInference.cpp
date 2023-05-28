/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------- ShapeInference.cpp ---------------------------===//
//
// Shape inference patterns and helper functions.
//
//===----------------------------------------------------------------------===//

#include "ShapeInference.hpp"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Interface/ShapeInferenceOpInterface.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

bool hasDynamicOrUnknownShape(Type type) {
  if (auto tensorType = dyn_cast<TensorType>(type))
    return !tensorType.hasStaticShape();

  if (type.isa<NoneType>())
    return false;

  if (auto seqType = dyn_cast<SeqType>(type))
    return ShapedType::isDynamic(seqType.getLength()) ||
           hasDynamicOrUnknownShape(seqType.getElementType());

  if (auto optType = dyn_cast<OptType>(type))
    return hasDynamicOrUnknownShape(optType.getElementType());

  llvm_unreachable("unknown type");
}

// Returns failure if the op can be verified and failed verification.
LogicalResult verifyOp(Operation *op) {
  if (auto info = op->getName().getRegisteredInfo()) {
    return info->verifyInvariants(op);
  } else {
    return success(); // Unregistered ops are unverifiable.
  }
}

// Returns failure if the op and its results are unchanged,
// like RewritePattern::matchAndRewrite().
// Calls shapeInfOp.emitOpError() if there is any actual failure.
LogicalResult inferShapes(ShapeInferenceOpInterface shapeInfOp) {
  OperationFingerPrint before(shapeInfOp.getOperation());
  if (failed(shapeInfOp.inferShapes([](Region &region) {})))
    return shapeInfOp.emitOpError("shape inference failed");
  OperationFingerPrint after(shapeInfOp.getOperation());
  if (after == before) {
    return failure(); // Operation and its result types are unchanged.
  } else {
    return success(); // Operation or its result types changed.
  }
}

// Shape inference pattern for regular ONNX ops from the ONNX specification,
// which all implement the ShapeInference interface.
struct InferShapesPattern
    : public OpInterfaceRewritePattern<ShapeInferenceOpInterface> {
  using OpInterfaceRewritePattern<
      ShapeInferenceOpInterface>::OpInterfaceRewritePattern;

  // Returns success if shapeInfOp or its result types changed.
  LogicalResult matchAndRewrite(ShapeInferenceOpInterface shapeInfOp,
      PatternRewriter &rewriter) const override {
    // Optimization: Don't (re)infer shapes if shapeInfOp is simple (has no
    // subgraphs) and its result shapes are known and static.
    if (!isa<HasOnnxSubgraphOpInterface>(shapeInfOp.getOperation()) &&
        !returnsDynamicOrUnknownShape(shapeInfOp))
      return failure();

    // Verify the operation before attempting to infer the shape of the
    // produced output(s).
    if (failed(verifyOp(shapeInfOp)))
      return shapeInfOp.emitOpError("verification failed");

    // Infer the results shapes.
    return inferShapes(shapeInfOp);
  }
};

// Shape inference pattern for ONNXYieldOp which terminates
// if/loop/scan subgraphs.
struct YieldShapesPattern : public OpRewritePattern<ONNXYieldOp> {
  using OpRewritePattern<ONNXYieldOp>::OpRewritePattern;

  // Returns success if parent op or parent result types changed.
  LogicalResult matchAndRewrite(
      ONNXYieldOp yieldOp, PatternRewriter &rewriter) const override {
    Operation *parent = yieldOp->getParentOp();
    assert((isa<ONNXIfOp, ONNXLoopOp, ONNXScanOp>(parent)) &&
           "onnx.Yield has if/loop/scan parent");
    return inferShapes(cast<ShapeInferenceOpInterface>(parent));
  }
};

} // namespace

bool returnsDynamicOrUnknownShape(Operation *op) {
  return llvm::any_of(op->getResultTypes(), hasDynamicOrUnknownShape);
}

void getShapeInferencePatterns(RewritePatternSet &set) {
  // Bump up the pattern benefit of the shape inference patterns to run them
  // before other patterns, because most other patterns (e.g. canonicalization)
  // work best after shapes are inferred.
  PatternBenefit highPriority(10000);
  set.insert<InferShapesPattern>(set.getContext(), highPriority);
  set.insert<YieldShapesPattern>(set.getContext(), highPriority);
}

// TODO: Consider whether to do this in a Func::ReturnOp pattern.
void inferFunctionReturnShapes(func::FuncOp f) {
  Operation *returnOp = f.getBody().back().getTerminator();
  assert(returnOp && "function must return");
  FunctionType fty = f.getFunctionType();
  assert(f.getNumResults() == returnOp->getNumOperands() &&
         "returned results count much match function type");
  f.setType(fty.clone(fty.getInputs(), returnOp->getOperandTypes()));
}

} // namespace onnx_mlir
