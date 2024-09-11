/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------- Loop.cpp - Lowering Loop Op ---------------------===//
//
// This file lowers the ONNX If Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXIfOpLowering : public OpConversionPattern<ONNXIfOp> {
  explicit ONNXIfOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXIfOp ifOp, ONNXIfOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = ifOp.getOperation();
    Location loc = ONNXLoc<ONNXIfOp>(op);

    KrnlBuilder createKrnl(rewriter, loc);
    Value cond = createKrnl.load(adaptor.getCond());

    auto resultTypes = ifOp.getResultTypes();
    SmallVector<Type> convertedResultTypes;
    if (failed(
            typeConverter->convertTypes(resultTypes, convertedResultTypes))) {
      return failure();
    }
    scf::IfOp scfIfOp = rewriter.create<scf::IfOp>(
        loc, convertedResultTypes, cond, /*withElseRegion=*/true);
    graphToScfBranch(
        rewriter, loc, ifOp.getThenBranch(), scfIfOp.getThenRegion());
    graphToScfBranch(
        rewriter, loc, ifOp.getElseBranch(), scfIfOp.getElseRegion());
    rewriter.replaceOp(op, scfIfOp.getResults());
    onnxToKrnlSimdReport(op);
    return success();
  }

private:
  void graphToScfBranch(ConversionPatternRewriter &rewriter, Location loc,
      Region &graph, Region &scfBranch) const {
    OpBuilder::InsertionGuard insertGuard(rewriter);

    rewriter.eraseBlock(&scfBranch.back());
    scfBranch.takeBody(graph);
  }
};

void populateLoweringONNXIfOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXIfOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
