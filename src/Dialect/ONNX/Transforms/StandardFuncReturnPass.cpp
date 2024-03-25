/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------- StandardFuncReturnPass.cpp -----------------------===//
//
// Replaces each ONNXReturnOp with a func::ReturnOp.
// Assumes that shape inference has matched the operand types with the
// parent function signature return types.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/Transforms/ShapeInference.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

struct StandardReturnPattern : public OpConversionPattern<ONNXReturnOp> {
  StandardReturnPattern(MLIRContext *context) : OpConversionPattern(context) {}
  LogicalResult matchAndRewrite(ONNXReturnOp ReturnOp,
      ONNXReturnOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    // Propagate return types to the function signature in case
    // they changed since last ShapeInferencePass.
    func::FuncOp function = ReturnOp.getParentOp();
    inferFunctionReturnShapes(function);

    rewriter.create<func::ReturnOp>(ReturnOp->getLoc(), ReturnOp.getOperands());
    rewriter.eraseOp(ReturnOp);
    return success();
  }
};

struct StandardFuncReturnPass
    : public PassWrapper<StandardFuncReturnPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StandardFuncReturnPass)

  StringRef getArgument() const override { return "standard-func-return"; }

  void runOnOperation() final {
    func::FuncOp function = getOperation();
    MLIRContext *context = &getContext();

    ConversionTarget target(*context);
    target
        .addLegalDialect<ONNXDialect, arith::ArithDialect, func::FuncDialect>();
    target.addIllegalOp<ONNXReturnOp>();

    RewritePatternSet patterns(context);
    patterns.insert<StandardReturnPattern>(context);

    if (failed(applyPartialConversion(function, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createStandardFuncReturnPass() {
  return std::make_unique<StandardFuncReturnPass>();
}

} // namespace onnx_mlir
