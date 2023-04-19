/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlPrintTensor.cpp - Lower KrnlPrintTensorOp ----------------===//
//
// Copyright 2022 The IBM Research Authors.
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
      TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(
            typeConverter, KrnlPrintTensorOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto printTensorOp = cast<KrnlPrintTensorOp>(op);
    MLIRContext *context = printTensorOp.getContext();
    Location loc = printTensorOp.getLoc();
    KrnlPrintTensorOpAdaptor operandAdaptor(operands);
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
    LLVMTypeConverter *typeConverter =
        static_cast<LLVMTypeConverter *>(getTypeConverter());

    StringRef msg = printTensorOp.getMsg();
    Value input = operandAdaptor.getInput();
    Value originalInput = printTensorOp.getInput();
    assert(input.getType().isa<LLVM::LLVMStructType>() &&
           "expecting LLVMStructType");

    ModuleOp module = printTensorOp->getParentOfType<ModuleOp>();
    const auto &apiRegistry =
        RuntimeAPIRegistry(module, rewriter, *typeConverter);

    // Get a symbol reference to the runtime function to use, creating one if
    // necessary.
    auto int64Ty = IntegerType::get(context, 64);
    auto memRefTy = input.getType().dyn_cast<LLVM::LLVMStructType>();
    auto memRefRank = krnl::getRankFromMemRefType(memRefTy);
    Value memRefRankVal = create.llvm.constant(int64Ty, (int64_t)memRefRank);
    Value omTensor = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
        RuntimeAPI::API::CREATE_OMTENSOR, {memRefRankVal});

    Type elemTy = originalInput.getType().cast<MemRefType>().getElementType();
    krnl::fillOMTensorWithMemRef(input, elemTy, omTensor, false /*outOwning*/,
        rewriter, loc, apiRegistry, module, *typeConverter);
    LLVM::GlobalOp globalStr = krnl::getOrCreateGlobalString(
        msg, loc, rewriter, module, typeConverter);
    Value strPtr =
        krnl::getPtrToGlobalString(globalStr, loc, rewriter, typeConverter);

    RuntimeAPI::callApi(rewriter, loc, apiRegistry,
        RuntimeAPI::API::PRINT_OMTENSOR, {strPtr, omTensor});

    rewriter.eraseOp(op);
    return success();
  }
};

void populateLoweringKrnlPrintTensorOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlPrintTensorOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
