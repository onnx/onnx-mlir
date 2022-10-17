/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------- Loop.cpp - Lowering Loop Op ---------------------===//
//
// This file lowers the ONNX If Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXIfOpLowering : public ConversionPattern {
  explicit ONNXIfOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXIfOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = ONNXLoc<ONNXIfOp>(op);
    auto ifOp = dyn_cast<ONNXIfOp>(op);
    ONNXIfOpAdaptor ifOpAdaptor(operands, op->getAttrDictionary());

    KrnlBuilder createKrnl(rewriter, loc);
    Value cond = createKrnl.load(ifOpAdaptor.cond());

    auto resultTypes = ifOp.getResultTypes();
    SmallVector<Type> convertedResultTypes;
    if (failed(
            typeConverter->convertTypes(resultTypes, convertedResultTypes))) {
      return failure();
    }
    scf::IfOp scfIfOp = rewriter.create<scf::IfOp>(
        loc, convertedResultTypes, cond, /*withElseRegion=*/true);
    graphToScfBranch(
        rewriter, loc, ifOp.then_branch(), scfIfOp.getThenRegion());
    graphToScfBranch(
        rewriter, loc, ifOp.else_branch(), scfIfOp.getElseRegion());
    rewriter.replaceOp(op, scfIfOp.getResults());
    return success();
  }

private:
  void graphToScfBranch(ConversionPatternRewriter &rewriter, Location loc,
      Region &graph, Region &scfBranch) const {
    OpBuilder::InsertionGuard insertGuard(rewriter);

    rewriter.eraseBlock(&scfBranch.back());
    scfBranch.takeBody(graph);
    rewriter.setInsertionPointToEnd(&scfBranch.back());

    Operation *returnOp = scfBranch.back().getTerminator();
    llvm::SmallVector<Value> outputs;
    if (failed(rewriter.getRemappedValues(returnOp->getOperands(), outputs))) {
      llvm_unreachable("failed to convert branch return values");
    }
    rewriter.replaceOpWithNewOp<scf::YieldOp>(returnOp, outputs);
  }
};

void populateLoweringONNXIfOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXIfOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
