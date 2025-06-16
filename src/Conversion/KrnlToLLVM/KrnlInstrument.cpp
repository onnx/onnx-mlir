
/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlInstrument.cpp - Lower KrnlInstrumentOp -------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlInstrumentOp operator.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"

#include "onnx-mlir/Compiler/OMCompilerRuntimeTypes.h"
#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

#define DEBUG_TYPE "krnl_to_llvm"

using namespace mlir;

namespace onnx_mlir {
namespace krnl {

class KrnlInstrumentOpLowering : public ConversionPattern {
public:
  explicit KrnlInstrumentOpLowering(
      LLVMTypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(
            typeConverter, KrnlInstrumentOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto *context = op->getContext();
    KrnlInstrumentOpAdaptor operandAdaptor(operands);
    Location loc = op->getLoc();
    KrnlInstrumentOp instrumentOp = llvm::dyn_cast<KrnlInstrumentOp>(op);

    StringRef opNameStr = instrumentOp.getOpName();

    StringRef nodeName;
    if (instrumentOp.getNodeName().has_value()) {
      // If we can get it from the instrument op direct, do so
      nodeName = instrumentOp.getNodeName().value();
    } else {
      // Otherwise, backup by creating it from the op.
      std::string nodeNameStr = getNodeNameInPresenceOfOpt(op);
      nodeName = rewriter.getStringAttr(nodeNameStr).strref();
    }
    LLVM_DEBUG(
        llvm::dbgs() << "Instrumentation_nodeName: " << nodeName << "\n");

    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
    const LLVMTypeConverter *typeConverter =
        static_cast<const LLVMTypeConverter *>(getTypeConverter());

    // Get a symbol reference to the memcpy function, inserting it if necessary.
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto instrumentRef = getOrInsertInstrument(rewriter, parentModule);

    LLVM::GlobalOp globalOpNameStr = krnl::getOrCreateGlobalString(
        opNameStr, loc, rewriter, parentModule, typeConverter);
    Value opNamePtr =
        krnl::getPtrToGlobalString(globalOpNameStr, loc, rewriter);
    // Encode the tag with the length of the op and node name strings
    uint64_t opNameLen = opNameStr.size();
    uint64_t nodeNameLen = nodeName.size();
    uint64_t tagWithLen = instrumentOp.getTag();
    SET_INSTRUMENT_OP_NAME_LEN(tagWithLen, opNameLen);
    SET_INSTRUMENT_NODE_NAME_LEN(tagWithLen, nodeNameLen);
    Value tag = create.llvm.constant(
        IntegerType::get(context, 64), static_cast<int64_t>(tagWithLen));
    LLVM::GlobalOp globalStr = krnl::getOrCreateGlobalString(
        nodeName, loc, rewriter, parentModule, typeConverter);
    Value nodeNamePtr = krnl::getPtrToGlobalString(globalStr, loc, rewriter);
    create.llvm.call({}, instrumentRef, {opNamePtr, tag, nodeNamePtr});

    rewriter.eraseOp(op);
    return success();
  }

private:
  // Create a function declaration for OMInstrumentPoint, the signature is:
  //   `void (ptr, i64, ptr)`
  FlatSymbolRefAttr getOrInsertInstrument(
      PatternRewriter &rewriter, ModuleOp module) const {
    MLIRContext *context = module.getContext();
    MultiDialectBuilder<LLVMBuilder> create(rewriter, module.getLoc());
    Type llvmVoidTy = LLVM::LLVMVoidType::get(context);
    Type llvmI64Ty = IntegerType::get(context, 64);
    Type opaquePtrTy = getI8PointerType(context);
    return create.llvm.getOrInsertSymbolRef(module,
        StringRef("OMInstrumentPoint"), llvmVoidTy,
        {opaquePtrTy, llvmI64Ty, opaquePtrTy});
  }
};

void populateLoweringKrnlInstrumentOpPattern(LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlInstrumentOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
