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

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Interface/ShapeInferenceOpInterface.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

// Shape inference pattern for regular ONNX ops from the ONNX specification,
// which all implement the ShapeInference interface.
struct InferShapesPattern
    : public OpInterfaceRewritePattern<ShapeInferenceOpInterface> {
  using OpInterfaceRewritePattern<
      ShapeInferenceOpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(ShapeInferenceOpInterface shapeInfOp,
      PatternRewriter &rewriter) const override {
    // TODO: check if it's necessary to skip ops that satisfy
    // !returnsDynamicOrUnknownShape (see ShapeInferencePass.cpp)

    // Verify the operation before attempting to infer the shape of the
    // produced output(s).
    Optional<RegisteredOperationName> registeredInfo =
        shapeInfOp->getName().getRegisteredInfo();
    if (registeredInfo &&
        failed(registeredInfo->verifyInvariants(&*shapeInfOp)))
      return shapeInfOp.emitOpError("verification failed");

    // Infer the results shapes.
    if (failed(shapeInfOp.inferShapes([](Region &region) {})))
      return shapeInfOp.emitOpError("shape inference failed");

    return success();
  }
};

// Shape inference pattern for ONNXReturnOp which terminates
// if/loop/scan subgraphs.
struct ReturnShapesPattern : public OpRewritePattern<ONNXReturnOp> {
  using OpRewritePattern<ONNXReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXReturnOp returnOp, PatternRewriter &rewriter) const override {
    Operation *parent = returnOp->getParentOp();
    assert(parent && "every onnx.Return op has a parent");
    if (auto shapeInfOp = dyn_cast<ShapeInferenceOpInterface>(parent)) {
      if (failed(shapeInfOp.inferShapes([](Region &region) {})))
        return shapeInfOp.emitOpError("shape inference failed");
    } else {
      llvm_unreachable("onnx.Return always has if/loop/scan parent");
    }
    return success();
  }
};

} // namespace

void getShapeInferencePatterns(RewritePatternSet &set) {
  // Bump up the pattern benefit of the shape inference patterns to run them
  // before other patterns, because most other patterns (e.g. canonicalization)
  // work best after shapes are inferred.
  PatternBenefit highPriority(10000);
  set.insert<InferShapesPattern>(set.getContext(), highPriority);
  set.insert<ReturnShapesPattern>(set.getContext(), highPriority);
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
