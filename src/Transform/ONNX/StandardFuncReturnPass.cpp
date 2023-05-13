/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------- StandardFuncReturnPass.cpp -----------------------===//
//
// Replaces each ONNXFuncReturnOp with a func::ReturnOp.
// Assumes that shape inference has matched the operand types with the
// parent function signature return types.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

struct StandardFuncReturnPattern
    : public OpConversionPattern<ONNXFuncReturnOp> {
  StandardFuncReturnPattern(MLIRContext *context)
      : OpConversionPattern(context) {}
  LogicalResult matchAndRewrite(ONNXFuncReturnOp funcReturnOp,
      ONNXFuncReturnOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    rewriter.create<func::ReturnOp>(
        funcReturnOp->getLoc(), funcReturnOp.getOperands());
    rewriter.eraseOp(funcReturnOp);
    return success();
  }
};

struct StandardFuncReturnPass
    : public PassWrapper<StandardFuncReturnPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StandardFuncReturnPass)

  StringRef getArgument() const override { return "scrub-disposable"; }

  void runOnOperation() final {
    func::FuncOp function = getOperation();
    MLIRContext *context = &getContext();

    ConversionTarget target(*context);
    target
        .addLegalDialect<ONNXDialect, arith::ArithDialect, func::FuncDialect>();
    target.addIllegalOp<ONNXFuncReturnOp>();

    RewritePatternSet patterns(context);
    patterns.insert<StandardFuncReturnPattern>(context);

    if (failed(applyPartialConversion(function, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createStandardFuncReturnPass() {
  return std::make_unique<StandardFuncReturnPass>();
}

} // namespace onnx_mlir
