/*
 * SPDX-License-Identifier: Apache-2.0
 */
//===------ KrnlSeqInsert.cpp - Lower KrnlSeqInsertOp
//----------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlSeqInsertOp operator.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "krnl_to_llvm"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
namespace krnl {

class KrnlSeqInsertOpLowering : public ConversionPattern {
public:
  explicit KrnlSeqInsertOpLowering(
      TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(
            typeConverter, KrnlSeqInsertOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    KrnlSeqInsertOpAdaptor operandAdaptor(operands);
    KrnlSeqInsertOp thisOp = dyn_cast<KrnlSeqInsertOp>(op);
    auto loc = op->getLoc();
    MultiDialectBuilder<MathBuilder, MemRefBuilder> create(rewriter, loc);
    IndexExprScope IEScope(&rewriter, loc);
    IndexExpr positionIE = SymbolIndexExpr(operandAdaptor.index());

    // Allocate output sequence
    Type convertedType =
        typeConverter->convertType(thisOp.getResult().getType());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = convertedType.cast<MemRefType>();

    auto seqElementConvertedType =
        outputMemRefType.getElementType().cast<MemRefType>();
    auto input_sequence = operandAdaptor.input_sequence();
    auto dimSize = create.mem.dim(input_sequence, 0);
    SymbolIndexExpr boundIE(dimSize);

    auto outputBound = boundIE + 1;
    SmallVector<IndexExpr, 1> ubsIE;
    ubsIE.emplace_back(outputBound);
    Value allocOutput =
        insertAllocAndDeallocSimple(rewriter, op, outputMemRefType, loc, ubsIE);

    // Allocate a new tensor and copy input tensor into it
    if (!operandAdaptor.input_element().getType().isa<MemRefType>())
      llvm_unreachable("Not supported: type of onnx seq element is not tensor");
    auto inputType =
        operandAdaptor.input_element().getType().cast<MemRefType>();
    SmallVector<mlir::Value, 4> allocParams;
    for (size_t i = 0; i < inputType.getShape().size(); i++) {
      if (inputType.getShape()[i] == -1) {
        allocParams.emplace_back(
            create.mem.dim(operandAdaptor.input_element(), i));
      }
    }
    Value alloc = create.mem.alignedAlloc(inputType, allocParams);
    rewriter.create<memref::CopyOp>(loc, operandAdaptor.input_element(), alloc);

    // Cast the input tensor to the element type of the sequence
    auto seq = operandAdaptor.input_sequence();
    auto seqElementType =
        seq.getType().cast<MemRefType>().getElementType().cast<MemRefType>();
    auto casted = create.mem.cast(alloc, seqElementType);

    // Copy the tensors in the sequence without duplicating the object
    // ToFix: An analysis pass is needed to determine whether duplicating
    // is needed.

    // Copy elements before the insertion position
    rewriter.create<scf::ForOp>(loc, create.math.constantIndex(0),
        positionIE.getValue(), create.math.constantIndex(1), ValueRange(),
        [&](OpBuilder &bodyBuilder, Location bodyLoc, Value forInduction,
            ValueRange iterArgs) {
          MultiDialectBuilder<MathBuilder, MemRefBuilder> create(
              bodyBuilder, bodyLoc);
          // onnx_mlir memref builder does not support load/store
          auto element = bodyBuilder.create<memref::LoadOp>(
              bodyLoc, operandAdaptor.input_sequence(), forInduction);
          auto converted = create.mem.cast(element, seqElementConvertedType);
          bodyBuilder.create<memref::StoreOp>(
              bodyLoc, converted, allocOutput, forInduction);
          bodyBuilder.create<scf::YieldOp>(bodyLoc);
        });

    // Store the tensor
    rewriter.create<memref::StoreOp>(
        loc, casted, allocOutput, operandAdaptor.index());

    // Copy elements after the insertion position
    rewriter.create<scf::ForOp>(loc, (positionIE + 1).getValue(),
        outputBound.getValue(), create.math.constantIndex(1), ValueRange(),
        [&](OpBuilder &bodyBuilder, Location bodyLoc, Value forInduction,
            ValueRange iterArgs) {
          MultiDialectBuilder<MathBuilder, MemRefBuilder> create(
              bodyBuilder, bodyLoc);
          auto element = bodyBuilder.create<memref::LoadOp>(
              bodyLoc, operandAdaptor.input_sequence(), forInduction);
          auto converted = create.mem.cast(element, seqElementConvertedType);
          auto outputIndex =
              create.math.add(forInduction, create.math.constantIndex(1));
          bodyBuilder.create<memref::StoreOp>(
              bodyLoc, converted, allocOutput, outputIndex);
          bodyBuilder.create<scf::YieldOp>(bodyLoc);
        });

    rewriter.replaceOp(op, allocOutput);
    return success();
  }
};

void populateLoweringKrnlSeqInsertOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlSeqInsertOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
