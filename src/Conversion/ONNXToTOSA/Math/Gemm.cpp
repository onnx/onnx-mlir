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
  //using OpAdaptor = typename ONNXGemmOp::Adaptor;
  LogicalResult rewriteToTosaFC(ONNXGemmOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const {
    Value A = adaptor.A();
    Value B = adaptor.B();
    Value C = adaptor.C();

    Type resultType = getTypeConverter()->convertType(op.getResult().getType());

    // If no bias is given to the GEMM operator, we create a 1D bias with all
    // zeroes
    if (C.getType().isa<mlir::NoneType>()) {
      // B is supposed to have 
      ArrayRef<int64_t> cformat(B.getType().cast<TensorType>().getShape()[1]);
      std::vector<float> elements = {};
      for (int i = 0; i < cformat[0]; ++i)
        elements.push_back(0.0F);
      C = mlir::tosa::getConstTensor<float>(rewriter, op, elements, cformat).value();
    }

    rewriter.replaceOpWithNewOp<tosa::FullyConnectedOp>(op, resultType, A, B, C);
    return success();
  }

  /// The GEMM can be described as a FullyConnected operator
  /// Y = AB^T + C if we perform a transpose on B only with
  /// alpha and beta factors set to 1.
  /// The constraints on rank were taken from the torch implementation check
  /// Input A must be of rank 2 (input)
  /// Input B must be of rank 2 (weights)
  /// Input C must be of rank 1 (bias)
  static bool checkLegalTosaFCOp(ONNXGemmOp op, OpAdaptor adaptor) {
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
    // onnx-mlir lowering to host only supports rank of 2
    // torch-mlir supports 2 and 3
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
    return true;
  }
  LogicalResult matchAndRewrite(ONNXGemmOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
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
    auto shapeA = A.getType().dyn_cast<ShapedType>().getShape();
    auto shapeB = B.getType().dyn_cast<ShapedType>().getShape();
    auto resultType = getTypeConverter()->convertType(op.getResult().getType()).cast<TensorType>();
    
    // If legal, create a FullyConnected operator instead
    if (checkLegalTosaFCOp(op, adaptor)) {
      return rewriteToTosaFC(op, adaptor, rewriter);
    }

    // ONNX gives 2d matrix as input and expect a 2d output. TOSA expects everything to be 3D. As such, there
    // is a need to add reshapes operators before and after we do any computation. We set the batch as 1 as it
    // is unknown.
    llvm::SmallVector<int64_t> newShapeA{1};
    llvm::SmallVector<int64_t> newShapeB{1};
    newShapeA.append({shapeA[0], shapeA[1]});
    newShapeB.append({shapeB[0], shapeB[1]});

    A = tosa::CreateOpAndInfer<tosa::ReshapeOp>(
        rewriter, op->getLoc(),
        UnrankedTensorType::get(AType.getElementType()), A,
        rewriter.getI64ArrayAttr(newShapeA)).getResult();
    B = tosa::CreateOpAndInfer<tosa::ReshapeOp>(
        rewriter, op->getLoc(),
        UnrankedTensorType::get(BType.getElementType()), B,
        rewriter.getI64ArrayAttr(newShapeB)).getResult();

    auto tosaResult = UnrankedTensorType::get(resultType.getElementType());
    // C is optional, if it's not there, we need to be aware of it for later computations
    bool isCPresent = C.getType().isa<TensorType>();

    if (transA) {
      Value targetTensor = mlir::tosa::getConstTensor<int32_t>(rewriter, op, {0, 2, 1}, {3}).value();
      Type outputType = UnrankedTensorType::get(AType.getElementType());
      A = tosa::CreateOpAndInfer<tosa::TransposeOp>(rewriter, loc, outputType, A, targetTensor).getResult();
    }
    if (transB) {
      Value targetTensor = mlir::tosa::getConstTensor<int32_t>(rewriter, op, {0, 2, 1}, {3}).value();
      Type outputType = UnrankedTensorType::get(AType.getElementType());
      B = tosa::CreateOpAndInfer<tosa::TransposeOp>(rewriter, loc, outputType, B, targetTensor).getResult();
    }
    
    Value alphaMulResult = NULL;
    Value betaMulResult = NULL;
    // If Alpha is present and not 1, we create a multiply operation for alpha * A
    if (alpha && alpha.getValueAsDouble() != 1.) {
      alphaMulResult = tosa::CreateOpAndInfer<tosa::MulOp>(rewriter, loc,
            tosaResult,
            tosa::getTosaConstTensorSingleF32(rewriter, op, alpha.getValueAsDouble(), newShapeA),
            A, 0).getResult();
    }

    // If C and Beta are set, and beta is different from 1, we also need to add a multiplication for beta * C
    if (beta && isCPresent && beta.getValueAsDouble() != 1.) {
      auto shapeC = C.getType().dyn_cast<ShapedType>();
      betaMulResult = tosa::CreateOpAndInfer<tosa::MulOp>(rewriter, loc, shapeC, 
            tosa::getTosaConstTensorSingleF32(rewriter, op, beta.getValueAsDouble(), shapeC.cast<TensorType>().getShape()),
            C, 0).getResult();
    }

    Value matmulRes = NULL;
    //A * B
    if (alphaMulResult) {
      //matmulRes = tosa::CreateOpAndInfer<tosa::MatMulOp>(rewriter, loc, tosaResult, alphaMulResult, B).getResult();
      matmulRes = tosa::CreateOpAndInfer<tosa::MatMulOp>(rewriter, loc, UnrankedTensorType::get(resultType.getElementType()), alphaMulResult, B).getResult();
    }
    else {
      //matmulRes = tosa::CreateOpAndInfer<tosa::MatMulOp>(rewriter, loc, tosaResult, A, B).getResult();
      matmulRes = tosa::CreateOpAndInfer<tosa::MatMulOp>(rewriter, loc, UnrankedTensorType::get(resultType.getElementType()), A, B).getResult();
    }

    Value addRes = NULL;
    //(A*B) + Beta * C or (A*B) + C
    if (betaMulResult) {
      addRes = tosa::CreateOpAndInfer<tosa::AddOp>(rewriter, loc, tosaResult, matmulRes, betaMulResult).getResult();
    }
    else if (isCPresent) {
      addRes = tosa::CreateOpAndInfer<tosa::AddOp>(rewriter, loc, tosaResult, matmulRes, adaptor.C()).getResult();
    }
    else {
      addRes = matmulRes;
    }

    // Add reshape to go back to the original size
    tosa::CreateReplaceOpAndInfer<tosa::ReshapeOp>(rewriter, op, resultType, addRes, rewriter.getI64ArrayAttr(resultType.getShape()));

    return success();
  }
};


} // namespace

void populateLoweringONNXGemmOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXGemmOpLoweringToTOSA>(typeConverter, ctx);
}

} // namespace onnx_mlir