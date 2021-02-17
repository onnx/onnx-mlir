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

template <>
struct ScalarOp<ONNXCustomOp> {
  using FOp = KrnlCustomOp;
  using IOp = KrnlCustomOp; 
};


template <>
Value emitScalarOpFor<ONNXCustomOp>(ConversionPatternRewriter &rewriter, Location loc,
    Operation *op, Type elementType, ArrayRef<Value> scalarOperands) {
    auto funcAttr = op->getAttr("function_name");
    llvm::StringRef funcName = funcAttr.cast<StringAttr>().getValue();
    if (elementType.isa<IntegerType>()) {
    return rewriter.create<ScalarIOp<ONNXCustomOp>>(
        loc, elementType, 
        scalarOperands[0], rewriter.getStringAttr(funcName));
//        rewriter.getStringAttr(funcName),scalarOperands, mlir::None);
  } else if (elementType.isa<FloatType>()) {
    return rewriter.create<ScalarFOp<ONNXCustomOp>>(
        loc, elementType,
        scalarOperands[0], rewriter.getStringAttr(funcName));
//        rewriter.getStringAttr(funcName), scalarOperands, mlir::None);
  } else {
    llvm_unreachable("unsupported element type");
  }
}


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
    auto loc = op->getLoc();
    auto X = operands[0];

    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertToMemRefType(*op->result_type_begin());

    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);

    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else
      alloc =
          insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc, X);

    SmallVector<Value, 4> loopIVs;
    // Only create krnl.iterate if one of the operands is not scalar tensor.
    if (!hasAllScalarValues(operands)) {
      // Create iterateOp & get block within iterate op.
      BuildKrnlLoop loops(rewriter, loc, memRefType.getRank());
      loops.createDefineAndIterateOp(X);
      Block *iterationBlock = loops.getIterateBlock();

      // Insert instructions inside the KernelIterateOp body.
      rewriter.setInsertionPointToStart(iterationBlock);

      // Handle the operation:
      for (auto arg : iterationBlock->getArguments())
        loopIVs.push_back(arg);
    }

    auto loadedVal = rewriter.create<KrnlLoadOp>(loc, X, loopIVs);
    //auto loweredOpResult = rewriter.create<KrnlCustomOp>(
    //    loc, memRefType.getElementType(),  {loadedVal}, mlir::None);
    auto loweredOpResult = emitScalarOpFor<ONNXCustomOp>(
        rewriter, loc, op, memRefType.getElementType(), {loadedVal});
    // Store result in the resulting array.
    rewriter.create<KrnlStoreOp>(loc, loweredOpResult, alloc, loopIVs);

    rewriter.replaceOp(op, alloc);
    return success();
    /*
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
    auto it = mlir::IntegerType::get(context,64);
    std::vector<int64_t> v = {4};
    llvm::ArrayRef<int64_t> shape(v);
    auto t = mlir::MemRefType::get(shape,it);
  auto funcType = mlir::FunctionType::get(context,{t}, {t});
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
    return success();*/
  }
};

void populateLoweringONNXCustomOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXCustomOpLowering>(ctx);
}

