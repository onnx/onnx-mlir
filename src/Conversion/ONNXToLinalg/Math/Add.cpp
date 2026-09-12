//===--- Add.cpp - Lowering ONNX Add Op to Linalg -------------------===//
//
// SPDX-License-Identifier: Apache-2.0
//
// Lowers `onnx.Add` into tensor-based `linalg.generic`.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"

#include "src/Conversion/ONNXToLinalg/ONNXToLinalgCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {

struct ONNXAddOpLoweringToLinalg : public OpRewritePattern<ONNXAddOp> {
  ONNXAddOpLoweringToLinalg(
      MLIRContext *ctx, const std::string &linalgOps, bool useLinalgPath)
      : OpRewritePattern<ONNXAddOp>(ctx), linalgOps(linalgOps),
        useLinalgPath(useLinalgPath) {}

  LogicalResult matchAndRewrite(
      ONNXAddOp addOp, PatternRewriter &rewriter) const final {
    Operation *op = addOp.getOperation();
    if (!shouldConvertToLinalg(op, linalgOps, useLinalgPath))
      return rewriter.notifyMatchFailure(
          addOp, "operation not selected for Linalg conversion");

    ONNXAddOpAdaptor adaptor(addOp);
    Value lhs = adaptor.getA();
    Value rhs = adaptor.getB();

    Location loc = addOp.getLoc();

    auto lhsTy = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsTy = dyn_cast<RankedTensorType>(rhs.getType());
    auto outTy = dyn_cast<RankedTensorType>(addOp.getType());

    if (!lhsTy || !rhsTy || !outTy)
      return rewriter.notifyMatchFailure(addOp, "expected ranked tensor types");

    // no broadcasting yet.
    if (lhsTy.getShape() != rhsTy.getShape())
      return rewriter.notifyMatchFailure(
          addOp, "no broadcasting supported yet");

    if (lhsTy.getElementType() != rhsTy.getElementType())
      return rewriter.notifyMatchFailure(addOp, "mismatched element types");

    // only elementwise float/int right now.
    Type elemTy = outTy.getElementType();
    bool isFloat = isa<FloatType>(elemTy);
    bool isInt = isa<IntegerType>(elemTy);
    if (!isFloat && !isInt)
      return rewriter.notifyMatchFailure(addOp, "unsupported element type");

    Value empty = tensor::EmptyOp::create(
        rewriter, loc, outTy.getShape(), outTy.getElementType());

    int64_t rank = outTy.getRank();

    AffineMap idMap =
        AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    SmallVector<AffineMap> indexingMaps{idMap, idMap, idMap};

    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);

    auto generic = linalg::GenericOp::create(rewriter, loc, TypeRange{outTy},
        ValueRange{lhs, rhs}, ValueRange{empty}, indexingMaps, iteratorTypes,
        [&](OpBuilder &b, Location bodyLoc, ValueRange blockArgs) {
          Value lhsElem = blockArgs[0];
          Value rhsElem = blockArgs[1];

          Value sum;
          if (isFloat) {
            sum = arith::AddFOp::create(b, bodyLoc, lhsElem, rhsElem);
          } else {
            sum = arith::AddIOp::create(b, bodyLoc, lhsElem, rhsElem);
          }

          linalg::YieldOp::create(b, bodyLoc, sum);
        });

    rewriter.replaceOp(addOp, generic.getResult(0));
    return success();
  }

private:
  std::string linalgOps;
  bool useLinalgPath;
};

void populateLoweringONNXAddOpToLinalgPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx,
    const std::string &linalgOps, bool useLinalgPath) {
  (void)typeConverter;
  patterns.add<ONNXAddOpLoweringToLinalg>(ctx, linalgOps, useLinalgPath);
}

} // namespace onnx_mlir
