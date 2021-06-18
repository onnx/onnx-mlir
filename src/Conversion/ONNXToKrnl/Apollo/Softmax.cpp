/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- Softmax.cpp - Softmax Op ---------------------------===//
//
// Copyright 2019 The IBM Research Authors.
// Copyright 2021 Microsoft.
//
// =============================================================================
//
// This file lowers ONNX softmax operator to a function call 
// that will be lowered to Apollo-specific code.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"

using namespace mlir;

namespace {
struct ONNXSoftmaxOpApolloLowering : public ConversionPattern {
private:
  static int nextCallId;

public:
  ONNXSoftmaxOpApolloLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXSoftmaxOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    int64_t rank = memRefType.getRank();
    int64_t axis = llvm::dyn_cast<ONNXSoftmaxOp>(op).axis();
    axis = axis >= 0 ? axis : rank + axis;
    assert(axis >= -rank && axis <= rank - 1);

    auto funcName = "apollo_tvp_Softmax_" + std::to_string(nextCallId++);
    auto moduleOp = op->template getParentOfType<ModuleOp>();
    assert(!moduleOp.lookupSymbol<FuncOp>(funcName));
    SmallVector<Type> resultTypes;
    for (auto rType : op->getResultTypes()) {
      resultTypes.emplace_back(!rType.isa<NoneType>() ? convertToMemRefType(rType) : rType);
    }
    auto callOp = rewriter.replaceOpWithNewOp<mlir::CallOp>(
        op, funcName, resultTypes, operands);

    auto fnType = rewriter.getFunctionType(op->getOperandTypes(), resultTypes);
    // Insert the function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());

    rewriter.create<FuncOp>(moduleOp.getLoc(), funcName, fnType).setPrivate();

    auto ctx = getContext();

    // Call ops don't seem to retain custom attributes, so attach it to the
    // function itself.
    Operation * funcOp = moduleOp.lookupSymbol<FuncOp>(funcName);
    // Record a boolean attribute to identify function in mlir passes.
    funcOp->setAttr("tvp.Softmax", BoolAttr::get(ctx, true));
    // Record axis as an attribute.
    funcOp->setAttr("axis", IntegerAttr::get(
      rewriter.getIntegerType(64, /*isSigned=*/true),
        APInt(64, axis, /*isSigned=*/true)));
    return success();
  }
};

int ONNXSoftmaxOpApolloLowering::nextCallId = 0;

} // namespace 

void populateLoweringONNXSoftmaxOpApolloPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXSoftmaxOpApolloLowering>(ctx);
}
