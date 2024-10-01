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
    // If legal, create a FullyConnected operator instead
    if (rewriteToTosaFC(op, adaptor, rewriter, tosaBuilder))
      return success();
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
        rewriter.getDenseI64ArrayAttr(newShapeA))
            .getResult();
    B = tosa::CreateOpAndInfer<mlir::tosa::ReshapeOp>(rewriter, op->getLoc(),
        RankedTensorType::get(dynamicTensorShape, BType.getElementType()), B,
        rewriter.getDenseI64ArrayAttr(newShapeB))
            .getResult();

    // If transA or transB are present, create Transpose operators.
    if (transA) {
      Value targetTensor =
          tosaBuilder.getConst(llvm::SmallVector<int32_t>{0, 2, 1}, {3});
      Type outputType =
          RankedTensorType::get(dynamicTensorShape, AType.getElementType());
      A = tosa::CreateOpAndInfer<mlir::tosa::TransposeOp>(
          rewriter, loc, outputType, A, targetTensor)
              .getResult();
    }
    if (transB) {
      Value targetTensor =
          tosaBuilder.getConst(llvm::SmallVector<int32_t>{0, 2, 1}, {3});
      Type outputType =
          RankedTensorType::get(dynamicTensorShape, BType.getElementType());
      B = tosa::CreateOpAndInfer<mlir::tosa::TransposeOp>(
          rewriter, loc, outputType, B, targetTensor)
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

  /// The GEMM can be described as a FullyConnected operator.
  /// Y = AB^T + C if we perform a transpose on B only with.
  /// alpha and beta factors set to 1.
  /// Input A must be of rank 2 (input).
  /// Input B must be of rank 2 (weights).
  /// Input C must be of rank 1 (bias).
  bool rewriteToTosaFC(ONNXGemmOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter, TosaBuilder &tosaBuilder) const {
    Value A = op.getA();
    Value B = op.getB();
    Value C = op.getC();

    auto AType = mlir::cast<TensorType>(A.getType());
    auto BType = mlir::cast<TensorType>(B.getType());

    bool isCPresent = !mlir::isa<mlir::NoneType>(C.getType());
    // If C is present, it can only be of rank 1, if the rank is not 1, return
    // false.
    if (mlir::isa<RankedTensorType>(C.getType()) &&
        mlir::cast<RankedTensorType>(C.getType()).getRank() != 1)
      return false;

    // Input tensor must be of rank 2.
    // Weights must also be of rank 2.
    if (AType.getRank() != 2 || BType.getRank() != 2)
      return false;

    // Both alpha and beta must be 1.
    if ((adaptor.getAlpha().convertToFloat() != 1.0F) ||
        (adaptor.getBeta().convertToFloat() != 1.0F))
      return false;

    // Only Transpose B must be enabled.
    if (adaptor.getTransA() != 0 || adaptor.getTransB() != 1)
      return false;

    // If all check passed, we replace the GEMM by a FC operator
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());

    // Because the bias is not broadcastable for TOSA while it is for ONNX,
    // we create an empty bias and use an add (broadcastable for tosa)
    // afterwards.
    // Base dummy C shape on B[0] shape.
    bool needsBroadcasting = !hasCCorrectShape(AType, BType, C);
    Value dummyC = C;
    if (!isCPresent || needsBroadcasting) {
      ArrayRef<int64_t> cformat(
          mlir::cast<TensorType>(resultType).getShape()[1]);
      std::vector<float> elements = {};
      for (int i = 0; i < cformat[0]; ++i)
        elements.push_back(0.0F);
      dummyC = tosaBuilder.getConst(elements, cformat);
    }

    Value fcRes = tosa::CreateOpAndInfer<mlir::tosa::FullyConnectedOp>(
        rewriter, op->getLoc(), resultType, A, B, dummyC)
                      .getResult();
    // If C was present in the original GEMM, we create an add to take the bias
    // into account.
    if (isCPresent && needsBroadcasting)
      fcRes = tosaBuilder.binaryOp<mlir::tosa::AddOp>(fcRes, C);

    rewriter.replaceOp(op, fcRes);

    return true;
  }
};

} // namespace

void populateLoweringONNXGemmOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXGemmOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir
