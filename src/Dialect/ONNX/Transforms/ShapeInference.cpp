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
#include "src/Interface/HasOnnxSubgraphOpInterface.hpp"
#include "src/Interface/ShapeInferenceOpInterface.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

bool hasDynamicOrUnknownShape(Type type) {
  if (auto tensorType = mlir::dyn_cast<TensorType>(type))
    return !tensorType.hasStaticShape();

  if (mlir::isa<NoneType>(type))
    return false;

  if (auto seqType = mlir::dyn_cast<SeqType>(type))
    return ShapedType::isDynamic(seqType.getLength()) ||
           hasDynamicOrUnknownShape(seqType.getElementType());

  if (auto optType = mlir::dyn_cast<OptType>(type))
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

// Variant of RewriterBase::updateRootInPlace(op) which finalizes/cancels the
// root update if the callback succeeds/fails.
template <typename Callback = std::function<LogicalResult()>>
LogicalResult tryUpdateRootInPlace(
    Operation *op, PatternRewriter &rewriter, Callback &&callback) {
  rewriter.startOpModification(op);
  if (failed(callback())) {
    rewriter.cancelOpModification(op);
    return failure();
  } else {
    rewriter.finalizeOpModification(op);
    return success();
  }
}

// Returns failure if the op and its results are unchanged,
// like RewritePattern::matchAndRewrite().
// Calls shapeInfOp.emitOpError() if there is any actual failure.
LogicalResult inferShapes(
    ShapeInferenceOpInterface shapeInfOp, PatternRewriter &rewriter) {
  return tryUpdateRootInPlace(shapeInfOp, rewriter, [&]() -> LogicalResult {
    OperationFingerPrint before(shapeInfOp);
    LogicalResult outcome = shapeInfOp.inferShapes([](Region &region) {});
    OperationFingerPrint after(shapeInfOp);
    if (failed(outcome)) {
      assert(after == before && "op must be unchanged on failure");
      return shapeInfOp.emitOpError("shape inference failed");
    }
    // succeed only shapeInfOp or its result types changed
    return after == before ? failure() : success();
  });
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
    // TODO: Consider fixing all the things to reenable the optimization below.
    //       It is disabled because inferShapes() sometimes "canonicalizes"
    //       operations with small rewrites and some other logic depends on it,
    //       e.g., ONNXTransposeOp::inferShapes() populates ONNXTransposeOp's
    //       perm attribute if absent and ONNXTransposeOpLowering depends on
    //       the attribute being set.
    // // Optimization: Don't (re)infer shapes if shapeInfOp is simple (has no
    // // subgraphs) and its result shapes are known and static.
    // if (!isa<HasOnnxSubgraphOpInterface>(shapeInfOp.getOperation()) &&
    //     !returnsDynamicOrUnknownShape(shapeInfOp))
    //   return failure();

    // Verify the operation before attempting to infer the shape of the
    // produced output(s).
    if (failed(verifyOp(shapeInfOp)))
      return shapeInfOp.emitOpError("verification failed");

    // Infer the results shapes.
    return inferShapes(shapeInfOp, rewriter);
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
    assert(parent->getName().hasInterface<HasOnnxSubgraphOpInterface>() &&
           "onnx.Yield has parent with subgraph");
    return inferShapes(cast<ShapeInferenceOpInterface>(parent), rewriter);
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

// TODO: Consider whether to do this in a ONNXReturnOp pattern
//       once lit tests are converted to use onnx.Return.
void inferFunctionReturnShapes(func::FuncOp f) {
  Operation *returnOp = f.getBody().back().getTerminator();
  assert(returnOp && "function must return");
  FunctionType fty = f.getFunctionType();
  assert(f.getNumResults() == returnOp->getNumOperands() &&
         "returned results count much match function type");
  f.setType(fty.clone(fty.getInputs(), returnOp->getOperandTypes()));
}

} // namespace onnx_mlir
