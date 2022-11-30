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

#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "src/Conversion/ONNXToTorch/NN/CommonUtils.h"
#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"
#include <numeric>
#include <vector>

/// ONNX Gemm operation

/// General Matrix multiplication: https://en.wikipedia.org/wiki/
/// Basic_Linear_Algebra_Subprograms#Level_3 A' = transpose(A) if
/// transA else A' B' = transpose(B) if transB else B'. Compute
/// Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K)
/// or (K, M), input tensor B has shape (K, N) or (N, K), input tensor C
/// is broadcastable to shape (M, N), and output tensor Y has shape (M, N).
/// A will be transposed before doing the computation if attribute transA
/// is non-zero, same for B and transB.
///
/// Attributes:
/// Attribute	    MLIR Type	           Description
///  alpha	::mlir::FloatAttr	32-bit float attribute
///  beta	::mlir::FloatAttr	32-bit float attribute
///  transA	::mlir::IntegerAttr	64-bit signed integer attribute
///  transB	::mlir::IntegerAttr	64-bit signed integer attribute

/// Operands:
/// Operand Description
///   A   tensor of 16-bit/32-bit/64-bit float values or
///       tensor of 32-bit/64-bit unsigned integer values or
///       tensor of 32-bit/64-bit signless integer values or
///       tensor of bfloat16 type values or memref of any type values.
///
///   B   tensor of 16-bit/32-bit/64-bit float values or
///       tensor of 32-bit/64-bit unsigned integer values or
///       tensor of 32-bit/64-bit signless integer values or
///       tensor of bfloat16 type values or memref of any type values.
///
///   C   tensor of 16-bit/32-bit/64-bit float values or
///       tensor of 32-bit/64-bit unsigned integer values or
///       tensor of 32-bit/64-bit signless integer values or
///       tensor of bfloat16 type values or memref of any type values.
/// Results:
/// Result Description
///   Y   tensor of 16-bit/32-bit/64-bit float values or
///       tensor of 32-bit/64-bit unsigned integer values or
///       tensor of 32-bit/64-bit signless integer values or
///       tensor of bfloat16 type values or memref of any type values.

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

class ONNXGemmOpToTorchLowering : public OpConversionPattern<ONNXGemmOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  Value getFloatValue(mlir::FloatAttr val, ConversionPatternRewriter &rewriter,
      Location loc) const {
    auto fVal =
        FloatAttr::get(rewriter.getF64Type(), val.getValue().convertToFloat());
    return rewriter.create<ConstantFloatOp>(loc, fVal);
  }

  int getRank(Value operand) const {
    if (auto operandType = operand.getType().dyn_cast<TensorType>()) {
      ArrayRef<int64_t> operandShape = operandType.getShape();
      return operandShape.size();
    }
    return 0;
  }

  SmallVector<int64_t, 4> getTransposedShape2D(ShapedType operandType) const {
    ArrayRef<int64_t> operandShape = operandType.getShape();

    SmallVector<int64_t, 4> transposedShape;
    transposedShape.emplace_back(operandShape[1]);
    transposedShape.emplace_back(operandShape[0]);

    return transposedShape;
  }

  /// Creates a tensor literal in torch using a f32 element type given
  /// a float constant.
  /// The bias is assumed to be 1 dimensional as that is the rank that is
  /// accepted by the torch.aten.linear operator. The size of the dimension
  /// is taken from the trailing dimensions of the op return result type
  static Value createLiteralTensor(ONNXGemmOp &op,
      ConversionPatternRewriter &rewriter, TypeConverter &typeConverter,
      float constant) {
    // We need to get the shape of the return type from the original result
    auto originalResultOpType = op->getResult(0).getType().cast<TensorType>();

    // Create a dense constant type tensor with the given value. Have the single
    // dimension be the size of the trailing dimension of the return type
    auto denseElementType = ::mlir::FloatType::getF32(rewriter.getContext());
    ShapedType denseValueType =
        RankedTensorType::get(originalResultOpType.getShape().drop_front(
                                  originalResultOpType.getRank() - 1),
            denseElementType);
    auto denseAttr = DenseElementsAttr::get(denseValueType, constant);

    // Convert the tensor type to the torch equivalent
    auto torchTensorType =
        typeConverter.convertType(denseValueType.cast<Type>());
    return rewriter.create<Torch::ValueTensorLiteralOp>(
        op.getLoc(), torchTensorType, denseAttr);
  }

  LogicalResult rewriteToAtenLinear(ONNXGemmOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const {
    Value A = adaptor.A();
    Value B = adaptor.B();
    Value C = adaptor.C();

    Type resultType = getTypeConverter()->convertType(op.getResult().getType());

    // If no bias is given to the GEMM operator, we create a 1D bias with all
    // zeroes
    if (C.getType().isa<mlir::NoneType>()) {
      C = createLiteralTensor(op, rewriter, *getTypeConverter(), 0.0F);
    }

    auto newOp =
        rewriter.replaceOpWithNewOp<AtenLinearOp>(op, resultType, A, B, C);
    setLayerNameAttr(op, newOp);
    return success();
  }

  /// The GEMM can also be described as a linear function
  /// Y = AB^T + C if we perform a transpose on B only with
  /// alpha and beta factors set to 1.
  /// The constraints on rank were taken from the torch implementation check
  /// Input A must be of rank 2 (input)
  /// Input B must be of rank 2 (weights)
  /// Input C must be of rank 1 (bias)
  static bool checkLegalAtenLinearOp(ONNXGemmOp op, OpAdaptor adaptor) {
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

    Location loc = op.getLoc();
    MLIRContext *context = op.getContext();

    Value A = op.A();
    Value B = op.B();
    Value C = op.C();

    if (checkLegalAtenLinearOp(op, adaptor)) {
      return rewriteToAtenLinear(op, adaptor, rewriter);
    }

    Type resultType = getTypeConverter()->convertType(op.getResult().getType());

    // C is an optional. However, when not present, it is replaced by a None.
    // Before using C we need to be sure that it is really there and it's not
    // just a None.
    bool IsCPresent = false;
    if (C.getType().isa<TensorType>())
      IsCPresent = true;

    // TODO: `op.C()` can broadcast its shape. XTen does not expect broadcast
    // so for now we arrange the broadcast here. When this is fixed in XTen we
    // will remove the explicit broadcasting from here. The fix is only applied
    // to constant ops and will not work in a generalized case.
    Value cTensor = adaptor.C();

    if (IsCPresent && C.getDefiningOp<ONNXConstantOp>() &&
        C.getDefiningOp()->hasAttr("value")) {
      auto cTensorOp =
          C.getDefiningOp()->getAttr("value").cast<TypedAttr>().getType().cast<TensorType>();
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
      ShapedType denseValueType =
          RankedTensorType::get(resultOpShape, denseElementType);
      DenseElementsAttr denseValueAttr =
          C.getDefiningOp()->getAttr("value").cast<DenseElementsAttr>().reshape(
              denseValueType);
      cTensor = rewriter.create<Torch::ValueTensorLiteralOp>(
          loc, resultType, denseValueAttr);
    }

    // Transpose A and B. Transpose on Torch is only 2d or less.
    if (!(getRank(A) == 2 && getRank(B) == 2 && getRank(C) <= 2))
      return op->emitError("Gemm only supports rank 2 tensors");

    auto aShapedType = A.getType().dyn_cast<ShapedType>();
    auto bShapedType = B.getType().dyn_cast<ShapedType>();
    int64_t transA = adaptor.transA();
    ::mlir::Type transposeAType =
        (transA != 0)
            ? Torch::ValueTensorType::get(context,
                  ArrayRef<int64_t>(getTransposedShape2D(aShapedType)),
                  aShapedType.getElementType())
            : adaptor.A().getType();
    int64_t transB = adaptor.transB();
    mlir::Type transposeBType =
        (transB != 0)
            ? Torch::ValueTensorType::get(context,
                  ArrayRef<int64_t>(getTransposedShape2D(bShapedType)),
                  bShapedType.getElementType())
            : adaptor.B().getType();

    Value transposeAVal = (transA != 0) ? rewriter.create<AtenTOp>(
                                              loc, transposeAType, adaptor.A())
                                        : adaptor.A();
    Value transposeBVal = (transB != 0) ? rewriter.create<AtenTOp>(
                                              loc, transposeBType, adaptor.B())
                                        : adaptor.B();
    setLayerNameAttr(op, transposeAVal.getDefiningOp());
    setLayerNameAttr(op, transposeBVal.getDefiningOp());

    // Compute Y = alpha * A' * B' + beta * C
    // Scalar multiplication with alpha(alpha * A')
    // and beta(beta * C) values.
    FloatAttr alpha = adaptor.alphaAttr();
    Value alphaMulResult = NULL, betaMulResult = NULL;
    if (alpha && alpha.getValueAsDouble() != 1.) {
      Value alpha3v = getFloatValue(alpha, rewriter, loc);
      alphaMulResult = rewriter.create<AtenMulScalarOp>(
          loc, transposeAType, transposeAVal, alpha3v);
      setLayerNameAttr(op, alphaMulResult.getDefiningOp());
    }

    FloatAttr beta = adaptor.betaAttr();
    if (beta && beta.getValueAsDouble() != 1. && IsCPresent) {
      Value beta3v = getFloatValue(beta, rewriter, loc);
      betaMulResult = rewriter.create<AtenMulScalarOp>(
          loc, cTensor.getType(), cTensor, beta3v);
      setLayerNameAttr(op, betaMulResult.getDefiningOp());
    }

    // Mm Operation ((alpha * A') * B')
    AtenMmOp mmValue;
    if (alphaMulResult)
      mmValue = rewriter.create<AtenMmOp>(
          loc, resultType, alphaMulResult, transposeBVal);
    else
      mmValue = rewriter.create<AtenMmOp>(
          loc, resultType, transposeAVal, transposeBVal);
    setLayerNameAttr(op, mmValue);

    // Addition ((alpha * A' * B') + (beta * C))
    Value iOne = getIntValue(1, rewriter, context, loc);
    if (betaMulResult) {
      auto newOp = rewriter.replaceOpWithNewOp<AtenAddTensorOp>(
          op, resultType, mmValue, betaMulResult, iOne);
      setLayerNameAttr(op, newOp);
    }
    // C is optional. If C is present but Beta was 1
    else if (IsCPresent) {
      auto newOp = rewriter.replaceOpWithNewOp<AtenAddTensorOp>(
          op, resultType, mmValue, cTensor, iOne);
      setLayerNameAttr(op, newOp);
    }
    // If neither C nor Beta is present, this is equivalent to a Matrix
    // multiply. We return (alpha * A' * B')
    else {
      rewriter.replaceOp(op, mmValue->getResults());
    }
    return success();
  }
};

void populateLoweringONNXToTorchGemmOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXGemmOpToTorchLowering>(typeConverter, ctx);
}
