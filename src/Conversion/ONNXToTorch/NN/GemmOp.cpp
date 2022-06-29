/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---- Gemm.cpp - Lowering Convolution Op ----===//
//
// Copyright 2022, Helprack LLC.
//
// ========================================================================
//
// This file lowers the ONNX Gemm Operation to Torch dialect.
//
//===-----------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/NN/CommonUtils.h"
#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"
#include <vector>
#include <numeric>

/*
 * ONNX Gemm operation

 * General Matrix multiplication: https://en.wikipedia.org/wiki/
 * Basic_Linear_Algebra_Subprograms#Level_3 A' = transpose(A) if
 * transA else A' B' = transpose(B) if transB else B'. Compute
 * Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K)
 * or (K, M), input tensor B has shape (K, N) or (N, K), input tensor C
 * is broadcastable to shape (M, N), and output tensor Y has shape (M, N).
 * A will be transposed before doing the computation if attribute transA
 * is non-zero, same for B and transB.
 *
 * Attributes:
 * Attribute	    MLIR Type	           Description
    alpha	::mlir::FloatAttr	32-bit float attribute
    beta	::mlir::FloatAttr	32-bit float attribute
    transA	::mlir::IntegerAttr	64-bit signed integer attribute
    transB	::mlir::IntegerAttr	64-bit signed integer attribute

 * Operands:
 * Operand Description
 *   A   tensor of 16-bit/32-bit/64-bit float values or
 *       tensor of 32-bit/64-bit unsigned integer values or
 *       tensor of 32-bit/64-bit signless integer values or
 *       tensor of bfloat16 type values or memref of any type values.
 *
 *   B   tensor of 16-bit/32-bit/64-bit float values or
 *       tensor of 32-bit/64-bit unsigned integer values or
 *       tensor of 32-bit/64-bit signless integer values or
 *       tensor of bfloat16 type values or memref of any type values.
 *
 *   C   tensor of 16-bit/32-bit/64-bit float values or
 *       tensor of 32-bit/64-bit unsigned integer values or
 *       tensor of 32-bit/64-bit signless integer values or
 *       tensor of bfloat16 type values or memref of any type values.
 * Results:
 * Result Description
 *   Y   tensor of 16-bit/32-bit/64-bit float values or
 *       tensor of 32-bit/64-bit unsigned integer values or
 *       tensor of 32-bit/64-bit signless integer values or
 *       tensor of bfloat16 type values or memref of any type values.
 */

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

struct ONNXGemmOpToTorchLowering : public ConversionPattern {

  Value getFloatValue(mlir::FloatAttr val, ConversionPatternRewriter &rewriter,
                      Location loc) const {
    auto fVal =
        FloatAttr::get(rewriter.getF64Type(), val.getValue().convertToFloat());
    return rewriter.create<ConstantFloatOp>(loc, fVal);
  }

  int getRank(Value operand) const {
    auto operandType = operand.getType().cast<TensorType>();
    ArrayRef<int64_t> operandShape = operandType.getShape();
    return operandShape.size();
  }

  SmallVector<int64_t, 4> getTransposedShape2D(ShapedType operandType) const {
    ArrayRef<int64_t> operandShape = operandType.getShape();

    SmallVector<int64_t, 4> transposedShape;
    transposedShape.emplace_back(operandShape[1]);
    transposedShape.emplace_back(operandShape[0]);

    return transposedShape;
  }

  ONNXGemmOpToTorchLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter, mlir::ONNXGemmOp::getOperationName(),
                          1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    ONNXGemmOp gemmOp = llvm::dyn_cast_or_null<ONNXGemmOp>(op);

    if(!gemmOp)
      return op->emitError("Expecting op to have a strong type");

    auto loc = gemmOp.getLoc();
    mlir::MLIRContext *context = gemmOp.getContext();

    auto alpha = gemmOp.alphaAttr(); // ::mlir::FloatAttr
    auto beta = gemmOp.betaAttr();   // ::mlir::FloatAttr
    auto transA = gemmOp.transA();
    auto transB = gemmOp.transB();

    Value iOne = getIntValue(1, rewriter, context, loc);

    auto A = gemmOp.A();
    auto B = gemmOp.B();
    auto C = gemmOp.C();

    auto aTensor = getTorchTensor(A, rewriter, context, loc);
    auto aType = toTorchType(context, A.getType());
    auto bTensor = getTorchTensor(B, rewriter, context, loc);
    auto bType = toTorchType(context, B.getType());
    auto resultType = toTorchType(context, gemmOp.getResult().getType());
    auto cTensor = getTorchTensor(C, rewriter, context, loc);

    // TODO: `gemmOp.C()` can broadcast its shape. XTen does not expect broadcast
    // so for now we arrange the broadcast here. When this is fixed in XTen we
    // will remove the explicit broadcasting from here. The fix is only applied
    // to constant ops and will not work in a generalized case.
  
    if (mlir::isa<ONNXConstantOp>(C.getDefiningOp()) &&
        C.getDefiningOp()->hasAttr("value")) {
      auto cTensorOp = C.getDefiningOp()->getAttr("value").getType()
          .cast<TensorType>();
      auto cTensorOpShape = cTensorOp.getShape();
      auto cTensorOpType = cTensorOp.getElementType();

      auto resultOp = op->getResult(0).getType().cast<TensorType>();
      auto resultOpShape = resultOp.getShape();
      auto resultOpType = resultOp.getElementType();

      int cShapeElements = std::accumulate(cTensorOpShape.begin(),
          cTensorOpShape.end(), 1, std::multiplies<int>());
      int resultOpShapeElements = std::accumulate(resultOpShape.begin(),
          resultOpShape.end(), 1, std::multiplies<int>());

      assert((cShapeElements == resultOpShapeElements) &&
          "C and result tensor shapes do not match");
      assert((cTensorOpType == resultOpType) &&
          "C and result tensor types do not match");

      auto denseElementType = ::mlir::FloatType::getF32(context);
      ShapedType denseValueType = RankedTensorType::get(resultOpShape,
          denseElementType);
      DenseElementsAttr denseValueAttr =
          C.getDefiningOp()->getAttr("value").cast<DenseElementsAttr>()
          .reshape(denseValueType);
      cTensor = rewriter.create<Torch::ValueTensorLiteralOp>(loc,
          resultType, denseValueAttr);
    }

    // Transpose A and B. Transpose on Torch is only 2d or less.
    if(!(getRank(A) == 2 && getRank(B) == 2 && getRank(C) <= 2))
      return op->emitError("Gemm only supports rank 2 tensors");

    auto aShapedType = A.getType().dyn_cast<ShapedType>();
    auto bShapedType = B.getType().dyn_cast<ShapedType>();

    mlir::Type transposeAType =
        (transA != 0)
            ? Torch::ValueTensorType::get(
                  context, ArrayRef<int64_t>(getTransposedShape2D(aShapedType)),
                  aShapedType.getElementType())
            : aType;
    mlir::Type transposeBType =
        (transB != 0)
            ? Torch::ValueTensorType::get(
                  context, ArrayRef<int64_t>(getTransposedShape2D(bShapedType)),
                  bShapedType.getElementType())
            : bType;

    Value transposeAVal =
        (transA != 0) ? rewriter.create<AtenTOp>(loc, transposeAType, aTensor)
                 : aTensor;
    Value transposeBVal =
        (transB != 0) ? rewriter.create<AtenTOp>(loc, transposeBType, bTensor)
                 : bTensor;

    // Compute Y = alpha * A' * B' + beta * C
    // Scalar multiplication with alpha(alpha * A')
    // and beta(beta * C) values.
    Value alphaMulResult = NULL, betaMulResult = NULL;
    if (alpha && alpha.getValueAsDouble() != 1.) {
      Value alpha3v = getFloatValue(alpha, rewriter, loc);
      alphaMulResult = rewriter.create<AtenMulScalarOp>(loc, transposeAType,
                                                        transposeAVal, alpha3v);
    }

    if (beta && beta.getValueAsDouble() != 1.) {
      Value beta3v = getFloatValue(beta, rewriter, loc);
      betaMulResult =
          rewriter.create<AtenMulScalarOp>(loc, cTensor.getType(), cTensor, beta3v);
    }

    // Bmm Operation ((alpha * A') * B')
    Value bmmValue;
    if (alphaMulResult)
      bmmValue = rewriter.create<AtenMmOp>(loc, resultType, alphaMulResult,
                                            transposeBVal);
    else
      bmmValue = rewriter.create<AtenMmOp>(loc, resultType, transposeAVal,
                                            transposeBVal);

    // Addition ((alpha * A' * B') + (beta * C))
    Value result;
    if (betaMulResult)
      result = rewriter.create<AtenAddTensorOp>(loc, resultType, bmmValue,
                                                betaMulResult, iOne);
    else
      result = rewriter.create<AtenAddTensorOp>(loc, resultType, bmmValue,
                                                cTensor, iOne);

    rewriter.replaceOpWithNewOp<torch::TorchConversion::ToBuiltinTensorOp>(
        op, op->getResult(0).getType(), result);

    return success();
  }
};

void populateLoweringONNXToTorchGemmOpPattern(RewritePatternSet &patterns,
                                              TypeConverter &typeConverter,
                                              MLIRContext *ctx) {
  patterns.insert<ONNXGemmOpToTorchLowering>(typeConverter, ctx);
}
