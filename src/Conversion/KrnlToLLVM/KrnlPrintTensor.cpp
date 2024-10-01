/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlPrintTensor.cpp - Lower KrnlPrintTensorOp ----------------===//
//
// Copyright 2022-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlPrintTensorOp operator.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Conversion/KrnlToLLVM/RuntimeAPI.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "krnl_to_llvm"

using namespace mlir;

namespace onnx_mlir {
namespace krnl {

class KrnlPrintTensorOpLowering : public ConversionPattern {
public:
  explicit KrnlPrintTensorOpLowering(
      LLVMTypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(
            typeConverter, KrnlPrintTensorOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto printTensorOp = mlir::cast<KrnlPrintTensorOp>(op);
    MLIRContext *context = printTensorOp.getContext();
    Location loc = printTensorOp.getLoc();
    KrnlPrintTensorOpAdaptor operandAdaptor(operands);
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
    const LLVMTypeConverter *typeConverter =
        static_cast<const LLVMTypeConverter *>(getTypeConverter());

    StringRef msg = printTensorOp.getMsg();
    Value input = operandAdaptor.getInput();
    Value originalInput = printTensorOp.getInput();
    assert(mlir::isa<LLVM::LLVMStructType>(input.getType()) &&
           "expecting LLVMStructType");

    ModuleOp module = printTensorOp->getParentOfType<ModuleOp>();
    const auto &apiRegistry =
        RuntimeAPIRegistry(module, rewriter, *typeConverter);

    // Get a symbol reference to the runtime function to use, creating one if
    // necessary.
    auto int64Ty = IntegerType::get(context, 64);
    auto memRefTy = mlir::dyn_cast<LLVM::LLVMStructType>(input.getType());
    auto memRefRank = krnl::getRankFromMemRefType(memRefTy);
    Value memRefRankVal =
        create.llvm.constant(int64Ty, static_cast<int64_t>(memRefRank));
    Value omTensor = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
        RuntimeAPI::API::CREATE_OMTENSOR, {memRefRankVal});

    Type elemTy =
        mlir::cast<MemRefType>(originalInput.getType()).getElementType();
    krnl::fillOMTensorWithMemRef(input, elemTy, omTensor, false /*outOwning*/,
        rewriter, loc, apiRegistry, module);
    LLVM::GlobalOp globalStr = krnl::getOrCreateGlobalString(
        msg, loc, rewriter, module, typeConverter);
    Value strPtr = krnl::getPtrToGlobalString(globalStr, loc, rewriter);

    RuntimeAPI::callApi(rewriter, loc, apiRegistry,
        RuntimeAPI::API::PRINT_OMTENSOR, {strPtr, omTensor});

    rewriter.eraseOp(op);
    return success();
  }
};

void populateLoweringKrnlPrintTensorOpPattern(LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlPrintTensorOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
