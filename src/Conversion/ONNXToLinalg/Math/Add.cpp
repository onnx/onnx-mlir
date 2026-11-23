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
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

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

    Value lhs = addOp.getA();
    Value rhs = addOp.getB();
    Location loc = addOp.getLoc();

    auto lhsTy = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsTy = dyn_cast<RankedTensorType>(rhs.getType());
    auto outTy = dyn_cast<RankedTensorType>(addOp.getType());

    if (!lhsTy || !rhsTy || !outTy)
      return rewriter.notifyMatchFailure(addOp, "expected ranked tensor types");

    if (!lhsTy.hasStaticShape() || !rhsTy.hasStaticShape() ||
        !outTy.hasStaticShape())
      return rewriter.notifyMatchFailure(
          addOp, "dynamic dimensions are not supported yet");

    if (lhsTy.getRank() != rhsTy.getRank() ||
        lhsTy.getRank() != outTy.getRank())
      return rewriter.notifyMatchFailure(
          addOp, "rank broadcasting is not supported yet");

    ONNXAddOpShapeHelper shapeHelper(op, op->getOperands());
    if (failed(shapeHelper.computeShape()))
      return rewriter.notifyMatchFailure(
          addOp, "failed to compute result shape");

    if (!shapeHelper.hasNoBroadcast())
      return rewriter.notifyMatchFailure(
          addOp, "broadcasting is not supported yet");

    if (outTy.getShape() != lhsTy.getShape())
      return rewriter.notifyMatchFailure(addOp, "unexpected result shape");

    if (lhsTy.getElementType() != rhsTy.getElementType() ||
        lhsTy.getElementType() != outTy.getElementType())
      return rewriter.notifyMatchFailure(addOp, "mismatched element types");

    Type elemTy = outTy.getElementType();
    if (!isa<FloatType>(elemTy) && !elemTy.isSignlessInteger() &&
        !elemTy.isUnsignedInteger())
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

          MathBuilder createMath(b, bodyLoc);
          Value sum = createMath.add(lhsElem, rhsElem);

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
