/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---- Gemm.cpp - Lowering Convolution Op ----===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Convolution Operators to Torch dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/NN/CommonUtils.h"
#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"
#include <fstream>
#include <iostream>
#include <set>
#include <vector>

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/ToolOutputFile.h"

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/OMOptions.hpp"

#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/StringExtras.h"

#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

#ifdef _WIN32
#include <io.h>
#endif

/*
 * ONNX Gemm operation

 * “General Matrix multiplication:” “https://en.wikipedia.org/wiki/
 * Basic_Linear_Algebra_Subprograms#Level_3” “” “A’ = transpose(A) if 
 * transA else A” “” “B’ = transpose(B) if transB else B” “” “Compute 
 * Y = alpha * A’ * B’ + beta * C, where input tensor A has shape (M, K) 
 * or (K, M),” “input tensor B has shape (K, N) or (N, K), input tensor C 
 * is broadcastable to shape (M, N),” “and output tensor Y has shape (M, N). 
 * A will be transposed before doing the” “computation if attribute transA 
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
 * A tensor of 16-bit float values or tensor of 32-bit float values or tensor
 * of 64-bit float values or tensor of 32-bit unsigned integer values or 
 * tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer
 * values or tensor of 64-bit signless integer values or tensor of bfloat16 type
 * values or memref of any type values.
 *
 * B tensor of 16-bit float values or tensor of 32-bit float values or tensor
 * of 64-bit float values or tensor of 32-bit unsigned integer values or 
 * tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer
 * values or tensor of 64-bit signless integer values or tensor of bfloat16 type 
 * values or memref of any type values.
 *
 * C tensor of 16-bit float values or tensor of 32-bit float values or tensor 
 * of 64-bit float values or tensor of 32-bit unsigned integer values or 
 * tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer
 * values or tensor of 64-bit signless integer values or tensor of bfloat16 type
 * values or memref of any type values or none type.
 *
 * Results:
 * Result Description
 * Y tensor of 16-bit float values or tensor of 32-bit float values or tensor 
 * of 64-bit float values or tensor of 32-bit unsigned integer values or 
 * tensor of 64-bit unsigned integer values or tensor of 32-bit signless integer
 * values or tensor of 64-bit signless integer values or tensor of bfloat16 type
 * values or memref of any type values or none type
 */

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

struct ONNXGemmOpToTorchLowering : public ConversionPattern {
  ONNXGemmOpToTorchLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXGemmOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    ONNXGemmOpAdaptor adaptor(operands);
    ONNXGemmOp op1 = llvm::dyn_cast<ONNXGemmOp>(op);

    mlir::MLIRContext *context = op1.getContext();

    Value a = op1.A(); // ONNX operands
    Value b = op1.B(); // ONNX operands
    Value c = op1.C(); // ONNX operands

    auto alpha = op1.alphaAttr();   // ::mlir::FloatAttr
    auto beta = op1.betaAttr();     // ::mlir::FloatAttr
    auto transA = op1.transAAttr(); // ::mlir::IntegerAttr
    auto transB = op1.transBAttr(); // ::mlir::IntegerAttr

    TensorType a_tensor_type = a.getType().cast<TensorType>();
    TensorType b_tensor_type = b.getType().cast<TensorType>();
    TensorType c_tensor_type = c.getType().cast<TensorType>();
    TensorType op_tensor_type = op->getResult(0).getType().cast<TensorType>();

    auto ty = IntegerType::get(op1.getContext(), 64);

    auto zero = 0;
    auto f0 = IntegerAttr::get(ty, zero);
    Value f0v = rewriter.create<ConstantIntOp>(loc, f0);

    auto one = 1;
    auto f1 = IntegerAttr::get(ty, one);
    Value f1v = rewriter.create<ConstantIntOp>(loc, f1);

    auto aTy = Torch::ValueTensorType::get(
        context, a_tensor_type.getShape(), a_tensor_type.getElementType());
    auto bTy = Torch::ValueTensorType::get(
        context, b_tensor_type.getShape(), b_tensor_type.getElementType());
    auto cTy = Torch::ValueTensorType::get(
        context, c_tensor_type.getShape(), c_tensor_type.getElementType());

    auto att = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
        loc, aTy, a);
    auto btt = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
        loc, bTy, b);
    auto ctt = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
        loc, cTy, c);

    auto neg_slope = alpha.getValue();
    auto f3 = FloatAttr::get(rewriter.getF64Type(), neg_slope.convertToFloat());
    Value alpha3v = rewriter.create<ConstantFloatOp>(loc, f3);

    neg_slope = beta.getValue();
    f3 = FloatAttr::get(rewriter.getF64Type(), neg_slope.convertToFloat());
    Value beta3v = rewriter.create<ConstantFloatOp>(loc, f3);

    auto resultTy = Torch::ValueTensorType::get(op1.getContext(),
        op_tensor_type.getShape(), op_tensor_type.getElementType());

    // Transpose the A and B.
    Value transposeAVal, transposeBVal;
    if (transA)
      transposeAVal =
          rewriter.create<AtenTransposeIntOp>(loc, resultTy, att, f0v, f1v);
    else
      transposeAVal = att;
    llvm::outs() << "\n transposeAVal : "
                 << "\n"
                 << transposeAVal << "\n"
                 << "\n";

    if (transB)
      transposeBVal =
          rewriter.create<AtenTransposeIntOp>(loc, resultTy, btt, f0v, f1v);
    else
      transposeBVal = btt;

    llvm::outs() << "\n transposeBVal : "
                 << "\n"
                 << transposeBVal << "\n"
                 << "\n";

    // Compute Y = alpha * A’ * B’ + beta * C
    // Scalar multiplication with alpha(alpha * A’) and beta(beta * C) values.
    Value alphaMulResult = NULL, betaMulResult = NULL;
    if (alpha)
      alphaMulResult = rewriter.create<AtenMulScalarOp>(
          loc, resultTy, transposeAVal, alpha3v);
    llvm::outs() << "alphaMulResult Value"
                 << "\n"
                 << alphaMulResult << "\n"
                 << "\n";

    if (beta)
      betaMulResult =
          rewriter.create<AtenMulScalarOp>(loc, resultTy, ctt, beta3v);

    llvm::outs() << "betaMulResult Value"
                 << "\n"
                 << betaMulResult << "\n"
                 << "\n";

    // Bmm Operation ((alpha * A’) * B’)
    Value bmmValue;
    if (alphaMulResult)
      bmmValue = rewriter.create<AtenBmmOp>(
          loc, resultTy, alphaMulResult, transposeBVal);
    else
      bmmValue = rewriter.create<AtenBmmOp>(
          loc, resultTy, transposeAVal, transposeBVal);

    llvm::outs() << "bmmValue operation creation"
                 << "\n"
                 << bmmValue << "\n"
                 << "\n";

    // Addition ((alpha * A’ * B’) + (beta * C))
    Value addValue;
    if (betaMulResult)
      addValue =
          rewriter.create<AtenAddTensorOp>(loc, resultTy, bmmValue,
			  betaMulResult, f1v);
    else
      addValue =
          rewriter.create<AtenAddTensorOp>(loc, resultTy, bmmValue,
			  transposeBVal, f1v);
    llvm::outs() << "Gemm operation creation"
                 << "\n"
                 << addValue << "\n"
                 << "\n";

    Value result = addValue;

    rewriter.replaceOpWithNewOp<torch::TorchConversion::ToBuiltinTensorOp>(
        op, resultTy, result);

    return success();
  }
};

void populateLoweringONNXToTorchGemmOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXGemmOpToTorchLowering>(typeConverter, ctx);
}
