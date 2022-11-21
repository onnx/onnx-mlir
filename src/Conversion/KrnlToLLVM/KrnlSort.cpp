/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlSort.cpp - Lower KrnlSortOp -----------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlSortOp operator.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "krnl_to_llvm"

using namespace mlir;

namespace onnx_mlir {
namespace krnl {

class KrnlSortOpLowering : public ConversionPattern {
public:
  explicit KrnlSortOpLowering(
      TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(
            typeConverter, KrnlSortOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto sortOp = cast<KrnlSortOp>(op);
    Location loc = sortOp.getLoc();
    KrnlSortOpAdaptor operandAdaptor(operands);
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
    MLIRContext *context = op->getContext();

    ModuleOp module = sortOp->getParentOfType<ModuleOp>();
    const auto &apiRegistry = RuntimeAPIRegistry(module, rewriter);

    // Get a symbol reference to the runtime function to use, creating one if
    // necessary.
    Value order = operandAdaptor.order();
    assert(order.getType().isa<LLVM::LLVMStructType>() &&
           "expecting LLVMStructType");
    Value input = operandAdaptor.input();
    assert(input.getType().isa<LLVM::LLVMStructType>() &&
           "expecting LLVMStructType");
    // Get a symbol reference to the runtime function to use, creating one if
    // necessary.
    auto int64Ty = IntegerType::get(context, 64);
    auto memRefTy = input.getType().dyn_cast<LLVM::LLVMStructType>();
    auto memRefRank = krnl::getRankFromMemRefType(memRefTy);
    Value memRefRankVal = create.llvm.constant(int64Ty, (int64_t)memRefRank);
    Value omTensorOrder = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
        RuntimeAPI::API::CREATE_OMTENSOR, {memRefRankVal});
    krnl::fillOMTensorWithMemRef(order, omTensorOrder, false /*outOwning*/,
        rewriter, loc, apiRegistry, module);
    Value omTensorInput = RuntimeAPI::callApi(rewriter, loc, apiRegistry,
        RuntimeAPI::API::CREATE_OMTENSOR, {memRefRankVal});
    krnl::fillOMTensorWithMemRef(input, omTensorInput, false /*outOwning*/,
        rewriter, loc, apiRegistry, module);
    Value axis = create.llvm.constant(int64Ty, (int64_t)sortOp.axis());
    Value ascending =
        create.llvm.constant(int64Ty, (int64_t)sortOp.ascending());
    Value algorithm =
        create.llvm.constant(int64Ty, (int64_t)sortOp.algorithm());

    // Sort func call.
    RuntimeAPI::callApi(rewriter, loc, apiRegistry, RuntimeAPI::API::SORT,
        {omTensorOrder, omTensorInput, axis, ascending, algorithm});

    rewriter.eraseOp(op);
    return success();
  }
};

void populateLoweringKrnlSortOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlSortOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
