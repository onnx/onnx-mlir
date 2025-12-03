/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Gemm.cpp - Gemm Op ----------------------------------===//
//
// Copyright (c) 2022-2024 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX Gemm operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "src/Conversion/ONNXToTOSA/DialectBuilder.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include <cstdint>

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXGemmOpLoweringToTOSA : public OpConversionPattern<ONNXGemmOp> {
public:
  using OpConversionPattern<ONNXGemmOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXGemmOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    TosaBuilder tosaBuilder(rewriter, op->getLoc());
    return rewriteToTosaMatMul(op, adaptor, rewriter, tosaBuilder);
  }

  LogicalResult rewriteToTosaMatMul(ONNXGemmOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter, TosaBuilder &tosaBuilder) const {
    Location loc = op->getLoc();
    Value A = op.getA();
    Value B = op.getB();
    Value C = op.getC();
    int64_t transA = adaptor.getTransA();
    int64_t transB = adaptor.getTransB();
    FloatAttr alpha = adaptor.getAlphaAttr();
    FloatAttr beta = adaptor.getBetaAttr();
    auto AType = mlir::cast<TensorType>(A.getType());
    auto BType = mlir::cast<TensorType>(B.getType());
    auto shapeA = AType.getShape();
    auto shapeB = BType.getShape();
    auto resultType = mlir::cast<TensorType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    auto context = op.getContext();

    // C is optional, if it's not there, we need to be aware of it for later
    // computations
    bool isCPresent = mlir::isa<TensorType>(C.getType());
    // ONNX uses HW matrix as input and output as it runs a matrix
    // multiplication. TOSA implements it as a batch matrix multiplication,
    // meaning the input and output are NHW. As such, there is a need to add
    // reshapes operators before and after we do any computation to add a batch
    // of 1.
    llvm::SmallVector<int64_t> newShapeA{1, shapeA[0], shapeA[1]};
    llvm::SmallVector<int64_t> newShapeB{1, shapeB[0], shapeB[1]};

    llvm::SmallVector<int64_t> dynamicTensorShape = {
        ShapedType::kDynamic, ShapedType::kDynamic, ShapedType::kDynamic};

    A = tosa::CreateOpAndInfer<mlir::tosa::ReshapeOp>(rewriter, op->getLoc(),
        RankedTensorType::get(dynamicTensorShape, AType.getElementType()), A,
        mlir::tosa::getTosaConstShape(rewriter, op.getLoc(), newShapeA))
            .getResult();
    B = tosa::CreateOpAndInfer<mlir::tosa::ReshapeOp>(rewriter, op->getLoc(),
        RankedTensorType::get(dynamicTensorShape, BType.getElementType()), B,
        mlir::tosa::getTosaConstShape(rewriter, op.getLoc(), newShapeB))
            .getResult();

    // If transA or transB are present, create Transpose operators.
    if (transA) {
      auto permsAttr = mlir::DenseI32ArrayAttr::get(context, {0, 2, 1});
      Type outputType =
          RankedTensorType::get(dynamicTensorShape, AType.getElementType());
      A = tosa::CreateOpAndInfer<mlir::tosa::TransposeOp>(
          rewriter, loc, outputType, A, permsAttr)
              .getResult();
    }
    if (transB) {
      auto permsAttr = mlir::DenseI32ArrayAttr::get(context, {0, 2, 1});
      Type outputType =
          RankedTensorType::get(dynamicTensorShape, BType.getElementType());
      B = tosa::CreateOpAndInfer<mlir::tosa::TransposeOp>(
          rewriter, loc, outputType, B, permsAttr)
              .getResult();
    }

    Value alphaMulResult = A;
    Value betaMulResult = C;
    // If Alpha is present and not 1, we create a multiply operation for alpha *
    // A
    if (alpha && alpha.getValueAsDouble() != 1.) {
      Value splattedConstAlpha = tosaBuilder.getSplattedConst(
          static_cast<float>(alpha.getValueAsDouble()), newShapeA);
      alphaMulResult = tosaBuilder.mul(splattedConstAlpha, A, 0);
    }

    // If C and Beta are set, and beta is different from 1, we also need to add
    // a multiplication for beta * C
    if (beta && isCPresent && beta.getValueAsDouble() != 1.) {
      Value splattedConstBeta = tosaBuilder.getSplattedConst(
          static_cast<float>(beta.getValueAsDouble()), newShapeA);
      betaMulResult = tosaBuilder.mul(splattedConstBeta, C, 0);
    }

    // A * B
    Value matmulRes = tosa::CreateOpAndInfer<mlir::tosa::MatMulOp>(rewriter,
        loc,
        RankedTensorType::get(dynamicTensorShape, resultType.getElementType()),
        alphaMulResult, B)
                          .getResult();

    Value addRes = NULL;
    //(A*B) + Beta * C or (A*B) + C
    if (isCPresent) {
      addRes =
          tosaBuilder.binaryOp<mlir::tosa::AddOp>(matmulRes, betaMulResult);
    } else {
      addRes = matmulRes;
    }

    // Add reshape to go back to the original shape
    Value reshape = tosaBuilder.reshape(addRes, resultType.getShape());
    rewriter.replaceOp(op, {reshape});
    return success();
  }

  /// Check if the bias (C) needs broadcasting when we convert GEMM to FC.
  static bool hasCCorrectShape(TensorType A, TensorType B, Value C) {
    if (!mlir::isa<mlir::RankedTensorType>(C.getType()))
      return false;
    ArrayRef<int64_t> AShape = A.getShape();
    ArrayRef<int64_t> BShape = B.getShape();
    ArrayRef<int64_t> CShape =
        mlir::cast<RankedTensorType>(C.getType()).getShape();
    // In the case of GemmToFC, transB is set meaning that B shapes will be
    // interverted so we check B[0]. Also, C is supposed to be of rank 1 so we
    // only need to check C[0].
    return CShape[0] == AShape[0] || CShape[0] == BShape[0];
  }
};

} // namespace

void populateLoweringONNXGemmOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXGemmOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
