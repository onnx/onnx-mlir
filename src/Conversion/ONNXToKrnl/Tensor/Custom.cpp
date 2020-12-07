//===----------------- Custom.cpp - Lowering Custom Op ----------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Custom Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

struct ONNXCustomOpLowering : public ConversionPattern {
  ONNXCustomOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXCustomOp::getOperationName(), 1, ctx) {}

//static FlatSymbolRefAttr findOrCreateFunction(PatternRewriter &rewriter,
static FuncOp findOrCreateFunction(PatternRewriter &rewriter,
    ModuleOp module, StringRef funcName, mlir::FunctionType funcType) {
  auto *context = module.getContext();
  if (auto sym = module.lookupSymbol<FuncOp>(funcName)) {
//if (auto sym = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName)) {
    assert(sym.getType() == funcType && "wrong symbol type");
    return sym;
    //return SymbolRefAttr::get(funcName, context);
  }

  // Insert the function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  return rewriter.create<FuncOp>(module.getLoc(), funcName, funcType);
  //return SymbolRefAttr::get(funcName, context);
}


  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXCustomOpAdaptor operandAdaptor(operands);
    rewriter.startRootUpdate(op);
    rewriter.setInsertionPointAfter(op);
    //rewriter.replaceOp(CallOp, operandAdaptor.input());
    //op->dump();
    auto module = op->getParentOfType<ModuleOp>();
    auto loc = op->getLoc();
    auto funcAttr = op->getAttr("function_name");
    auto xxx = funcAttr.cast<StringAttr>();
    auto sref = xxx.getValue();
    llvm::StringRef fname = sref;
    MLIRContext* context = (op->getContext());
    OpBuilder builder(context);
    auto it = mlir::IntegerType::get(64,context);
    std::vector<int64_t> v = {4};
    llvm::ArrayRef<int64_t> shape(v);
    auto t = mlir::MemRefType::get(shape,it);
  auto funcType = mlir::FunctionType::get({t}, {t}, context);
    auto function = findOrCreateFunction(rewriter, module, fname, funcType);
    function.dump();
    auto funcCallOp = rewriter.create<CallOp>(loc, function, operandAdaptor.input());
    funcCallOp.dump();
    //funcCallOp.setAttr("callee",funcAttr);
    printf("before dump\n");
    funcCallOp.dump();
    printf("\nafter dump\n");
    fflush(stdout);
    //rewriter.replaceOp(op,llvm::None);
    //rewriter.replaceOp(op,funcCallOp);
    //rewriter.eraseOp(op);
    //rewriter.replaceOpWithNewOp<CallOp>(
    //    op, op->getResult(0).getType(), operandAdaptor.input());
    rewriter.finalizeRootUpdate(op);
    return success();
  }
};

void populateLoweringONNXCustomOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXCustomOpLowering>(ctx);
}
