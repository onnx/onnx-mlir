/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Gemm.cpp - Gemm Op --------------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file lowers ONNX Gemm operator to TOSA dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXGemmOpLoweringToTOSA : public OpConversionPattern<ONNXGemmOp> {
public:
  using OpConversionPattern<ONNXGemmOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ONNXGemmOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    // If legal, create a FullyConnected operator instead
    if (rewriteToTosaFC(op, adaptor, rewriter)) {
      return success();
    }
    return rewriteToTosaMatMul(op, adaptor, rewriter);
  }

  LogicalResult rewriteToTosaMatMul(ONNXGemmOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const {
    Location loc = op->getLoc();
    Value A = op.A();
    Value B = op.B();
    Value C = op.C();
    int64_t transA = adaptor.transA();
    int64_t transB = adaptor.transB();
    FloatAttr alpha = adaptor.alphaAttr();
    FloatAttr beta = adaptor.betaAttr();
    auto AType = A.getType().cast<RankedTensorType>();
    auto BType = B.getType().cast<RankedTensorType>();
    auto shapeA = AType.getShape();
    auto shapeB = BType.getShape();
    auto resultType = getTypeConverter()
                          ->convertType(op.getResult().getType())
                          .cast<TensorType>();

    // C is optional, if it's not there, we need to be aware of it for later
    // computations
    bool isCPresent = C.getType().isa<TensorType>();
    // ONNX uses HW matrix as input and output as it runs a matrix
    // multiplication. TOSA implements it as a batch matrix multiplication,
    // meaning the input and output are NHW. As such, there is a need to add
    // reshapes operators before and after we do any computation to add a batch
    // of 1.
    llvm::SmallVector<int64_t> newShapeA{1, shapeA[0], shapeA[1]};
    llvm::SmallVector<int64_t> newShapeB{1, shapeB[0], shapeB[1]};

    A = tosa::CreateOpAndInfer<mlir::tosa::ReshapeOp>(rewriter, op->getLoc(),
        RankedTensorType::get({-1, -1, -1}, AType.getElementType()), A,
        rewriter.getI64ArrayAttr(newShapeA))
            .getResult();
    B = tosa::CreateOpAndInfer<mlir::tosa::ReshapeOp>(rewriter, op->getLoc(),
        RankedTensorType::get({-1, -1, -1}, BType.getElementType()), B,
        rewriter.getI64ArrayAttr(newShapeB))
            .getResult();

    auto tosaResult =
        RankedTensorType::get({-1, -1, -1}, resultType.getElementType());

    // If transA or transB are present, create Transpose operators.
    if (transA) {
      Value targetTensor =
          tosa::getConstTensor<int32_t>(rewriter, op, {0, 2, 1}, {3}).value();
      Type outputType =
          RankedTensorType::get({-1, -1, -1}, AType.getElementType());
      A = tosa::CreateOpAndInfer<mlir::tosa::TransposeOp>(
          rewriter, loc, outputType, A, targetTensor)
              .getResult();
    }
    if (transB) {
      Value targetTensor =
          tosa::getConstTensor<int32_t>(rewriter, op, {0, 2, 1}, {3}).value();
      Type outputType =
          RankedTensorType::get({-1, -1, -1}, BType.getElementType());
      B = tosa::CreateOpAndInfer<mlir::tosa::TransposeOp>(
          rewriter, loc, outputType, B, targetTensor)
              .getResult();
    }

    Value alphaMulResult = A;
    Value betaMulResult = C;
    // If Alpha is present and not 1, we create a multiply operation for alpha *
    // A
    if (alpha && alpha.getValueAsDouble() != 1.) {
      alphaMulResult =
          tosa::CreateOpAndInfer<mlir::tosa::MulOp>(rewriter, loc, tosaResult,
              tosa::getTosaConstTensorSingleF32(
                  rewriter, op, alpha.getValueAsDouble(), newShapeA),
              A, 0)
              .getResult();
    }

    // If C and Beta are set, and beta is different from 1, we also need to add
    // a multiplication for beta * C
    if (beta && isCPresent && beta.getValueAsDouble() != 1.) {
      auto shapeC = C.getType().dyn_cast<ShapedType>();
      betaMulResult = tosa::CreateOpAndInfer<mlir::tosa::MulOp>(rewriter, loc,
          shapeC,
          tosa::getTosaConstTensorSingleF32(rewriter, op,
              beta.getValueAsDouble(), shapeC.cast<TensorType>().getShape()),
          C, 0)
                          .getResult();
    }

    // A * B
    Value matmulRes = tosa::CreateOpAndInfer<mlir::tosa::MatMulOp>(rewriter,
        loc, RankedTensorType::get({-1, -1, -1}, resultType.getElementType()),
        alphaMulResult, B)
                          .getResult();

    Value addRes = NULL;
    //(A*B) + Beta * C or (A*B) + C
    if (isCPresent) {
      addRes = tosa::CreateOpAndInfer<mlir::tosa::AddOp>(
          rewriter, loc, tosaResult, matmulRes, betaMulResult)
                   .getResult();
    } else {
      addRes = matmulRes;
    }

    // Add reshape to go back to the original shape
    tosa::CreateReplaceOpAndInfer<mlir::tosa::ReshapeOp>(rewriter, op,
        resultType, addRes, rewriter.getI64ArrayAttr(resultType.getShape()));

    return success();
  }

  /// The GEMM can be described as a FullyConnected operator
  /// Y = AB^T + C if we perform a transpose on B only with
  /// alpha and beta factors set to 1.
  /// Input A must be of rank 2 (input)
  /// Input B must be of rank 2 (weights)
  /// Input C must be of rank 1 (bias)
  bool rewriteToTosaFC(ONNXGemmOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const {
    Value A = op.A();
    Value B = op.B();
    Value C = op.C();

    auto AType = A.getType().cast<RankedTensorType>();
    auto BType = B.getType().cast<RankedTensorType>();

    // If C is present, it can only be of rank 1
    if (C.getType().isa<RankedTensorType>() &&
        C.getType().cast<RankedTensorType>().getRank() != 1) {
      return false;
    }
    // Input tensor must be of rank 2
    // Weights must also be of rank 2
    if (AType.getRank() != 2 || BType.getRank() != 2) {
      return false;
    }
    // Both alpha and beta must be 1
    if ((adaptor.alpha().convertToFloat() != 1.0F) ||
        (adaptor.beta().convertToFloat() != 1.0F)) {
      return false;
    }
    // Only Transpose B must be enabled
    if (adaptor.transA() != 0 || adaptor.transB() != 1) {
      return false;
    }

    // If all check passed, we replace the GEMM by a FC operator
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());

    // If no bias is given to the GEMM operator, we create a 1D bias with all
    // zeroes
    if (C.getType().isa<mlir::NoneType>()) {
      // Base C format on B[0] format
      ArrayRef<int64_t> cformat(B.getType().cast<TensorType>().getShape()[0]);
      std::vector<float> elements = {};
      for (int i = 0; i < cformat[0]; ++i)
        elements.push_back(0.0F);
      C = tosa::getConstTensor<float>(rewriter, op, elements, cformat).value();
    }

    rewriter.replaceOpWithNewOp<mlir::tosa::FullyConnectedOp>(
        op, resultType, A, B, C);
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