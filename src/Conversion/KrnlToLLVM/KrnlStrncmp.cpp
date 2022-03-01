/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlStrncmp.cpp - Lower KrnlStrncmpOp -------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlStrncmpOp operator.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "krnl_to_llvm"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
namespace krnl {

class KrnlStrncmpOpLowering : public ConversionPattern {
public:
  explicit KrnlStrncmpOpLowering(
      TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(
            typeConverter, KrnlStrncmpOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    KrnlStrncmpOpAdaptor operandAdaptor(operands);
    Location loc = op->getLoc();

    // Get a symbol reference to the strncmp function, inserting it if
    // necessary.
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto StrncmpRef = getOrInsertStrncmp(rewriter, parentModule);

    // Operands.
    Type strType = operandAdaptor.str1()
                       .getType()
                       .cast<LLVM::LLVMStructType>()
                       .getBody()[1];
    Value extractedStrPtr1 = rewriter.create<LLVM::ExtractValueOp>(
        loc, strType, operandAdaptor.str1(), rewriter.getI64ArrayAttr(1));
    Value extractedStrPtr2 = rewriter.create<LLVM::ExtractValueOp>(
        loc, strType, operandAdaptor.str2(), rewriter.getI64ArrayAttr(1));
    Value length = operandAdaptor.len();

    // Strncmp call.
    MLIRContext *ctx = op->getContext();
    Type i32Type = IntegerType::get(ctx, 32);
    auto funcCall = rewriter.create<CallOp>(loc, StrncmpRef, i32Type,
        ArrayRef<Value>({extractedStrPtr1, extractedStrPtr2, length}));

    rewriter.replaceOp(op, funcCall.getResults()[0]);
    return success();
  }
};

void populateLoweringKrnlStrncmpOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlStrncmpOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir