
/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlInstrument.cpp - Lower KrnlInstrumentOp -------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
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

#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"

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
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
    LLVMTypeConverter *typeConverter =
        static_cast<LLVMTypeConverter *>(getTypeConverter());

    // Get a symbol reference to the memcpy function, inserting it if necessary.
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto instrumentRef = getOrInsertInstrument(rewriter, parentModule);

    StringRef opNameStr = instrumentOp.getOpName();
    LLVM::GlobalOp globalOpNameStr = krnl::getOrCreateGlobalString(
        opNameStr, loc, rewriter, parentModule, typeConverter);
    Value opNamePtr =
        krnl::getPtrToGlobalString(globalOpNameStr, loc, rewriter);
    Value tag = create.llvm.constant(
        IntegerType::get(context, 64), (int64_t)instrumentOp.getTag());
    StringRef nodeName;
    if (instrumentOp.getNodeName().has_value())
      nodeName = instrumentOp.getNodeName().value();
    else if (auto nameLoc = loc.dyn_cast<NameLoc>())
      nodeName = nameLoc.getName();
    else if (auto fusedLoc = loc.dyn_cast<FusedLoc>()) {
      // Combine each location name and set it as nodeName.
      std::string name;
      for (Location locIt : fusedLoc.getLocations()) {
        if (auto nameLocIt = locIt.dyn_cast<NameLoc>())
          name += nameLocIt.getName().str() + "-";
        else if (auto fileLineColLoc = locIt.dyn_cast<FileLineColLoc>()) {
          StringRef filename =
              llvm::sys::path::filename(fileLineColLoc.getFilename().str());
          name += filename.str() + ":" +
                  std::to_string(fileLineColLoc.getLine()) + "-";
        }
      }
      if (name.empty())
        name = "NOTSET";
      else
        name.pop_back(); // remove last "-"
      loc = NameLoc::get(rewriter.getStringAttr(name));
      nodeName = cast<NameLoc>(loc).getName();
    } else if (auto fileLineColLoc = loc.dyn_cast<FileLineColLoc>()) {
      StringRef filename =
          llvm::sys::path::filename(fileLineColLoc.getFilename().str());
      std::string name =
          filename.str() + ":" + std::to_string(fileLineColLoc.getLine());
      loc = NameLoc::get(rewriter.getStringAttr(name));
      nodeName = cast<NameLoc>(loc).getName();
    } else
      nodeName = StringRef("NOTSET");
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
