//===-------------------- Loop.cpp - Lowering Loop Op ---------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Loop Operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

struct ONNXLoopOpLowering : public ConversionPattern {
  ONNXLoopOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXLoopOp::getOperationName(), 1, ctx) {}

  void allocateMemoryForVFinal(mlir::Location loc,
      ConversionPatternRewriter &rewriter, Operation *op,
      ONNXLoopOpAdaptor loopOpAdapter,
      SmallVectorImpl<mlir::Value> &outputs) const {
    auto vFinalAndScanOutputs = op->getOpResults();
    auto opVFinalOutputs = llvm::make_range(vFinalAndScanOutputs.begin(),
        vFinalAndScanOutputs.begin() + loopOpAdapter.v_initial().size());
    auto vInitIter = loopOpAdapter.v_initial();

    for (const auto &ioPair : llvm::zip(vInitIter, opVFinalOutputs)) {
      auto vInit = std::get<0>(ioPair);
      auto vFinal = std::get<1>(ioPair);

      auto memRefType = convertToMemRefType(vFinal.getType());
      Value alloc;
      bool shouldDealloc = checkInsertDealloc(op);
      if (hasAllConstantDimensions(memRefType))
        alloc = insertAllocAndDealloc(memRefType, loc, rewriter, shouldDealloc);
      else
        alloc = insertAllocAndDealloc(
            memRefType, loc, rewriter, shouldDealloc, {vInit});
      outputs.emplace_back(alloc);
    }
  }

  void allocateMemoryForScanOutput(mlir::Location loc,
      ConversionPatternRewriter &rewriter, Operation *op,
      ONNXLoopOpAdaptor loopOpAdapter,
      SmallVectorImpl<mlir::Value> &outputs) const {
    auto vFinalAndScanOutputs = op->getOpResults();
    auto opScanOutputIter = llvm::make_range(
        vFinalAndScanOutputs.begin() + loopOpAdapter.v_initial().size(),
        vFinalAndScanOutputs.end());
    auto vInitIter = loopOpAdapter.v_initial();

    for (const auto &opScanOutput : opScanOutputIter) {
      auto memRefType = convertToMemRefType(opScanOutput.getType());
      Value alloc;
      bool shouldDealloc = checkInsertDealloc(op);
      if (hasAllConstantDimensions(memRefType))
        alloc = insertAllocAndDealloc(memRefType, loc, rewriter, shouldDealloc);
      else {
        auto rankedScanOutTy = memRefType;
        SmallVector<mlir::Value, 4> allocParams;
        for (int i = 0; i < rankedScanOutTy.getRank(); i++) {
          if (rankedScanOutTy.getShape()[i] == -1) {
            if (i == 0) {
              // TODO(tjingrant): in general, it is not correct to expect
              // loop operation scan output to have the leading dimension extent
              // equal to the trip count, due to the possibility of early
              // termination.
              assert(!loopOpAdapter.M().getType().isa<NoneType>());
              Value maxTripCount =
                  rewriter.create<LoadOp>(loc, loopOpAdapter.M()).getResult();
              allocParams.emplace_back(rewriter.create<IndexCastOp>(
                  loc, maxTripCount, rewriter.getIndexType()));
            } else {
              llvm_unreachable("Error.");
            }
          }
        }
        alloc = rewriter.create<AllocOp>(loc, rankedScanOutTy, allocParams);
      }
      outputs.emplace_back(alloc);
    }
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    ONNXLoopOpAdaptor loopOpAdapter(operands, op->getAttrDictionary());

    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto symbolName =
        loopOpAdapter.body().cast<SymbolRefAttr>().getLeafReference();
    auto func = dyn_cast<mlir::FuncOp>(module.lookupSymbol(symbolName));
    auto &loopBody = func.getBody();

    // Allocate memory for two kinds of outputs:
    // - final values of loop dependent variables, and
    // - scan output (all intermediate values of) loop dependent variables.
    SmallVector<Value, 4> outputs;
    allocateMemoryForVFinal(loc, rewriter, op, loopOpAdapter, outputs);
    allocateMemoryForScanOutput(loc, rewriter, op, loopOpAdapter, outputs);

    // Copy content of vInit to vFinal, which is used to host intermediate
    // values produced by loop body function invocation.
    for (const auto &vInitAndFinal :
        llvm::zip(loopOpAdapter.v_initial(), outputs)) {
      const auto &vInit = std::get<0>(vInitAndFinal);
      const auto &vFinal = std::get<1>(vInitAndFinal);
      EmitCopy(rewriter, loc, vInit, vFinal);
    }

    BuildKrnlLoop loop(rewriter, loc, 1);
    loop.createDefineOp();
    Value maxTripCount =
        rewriter.create<LoadOp>(loc, loopOpAdapter.M()).getResult();
    maxTripCount = rewriter.create<IndexCastOp>(
        loc, maxTripCount, rewriter.getIndexType());
    loop.pushBounds(0, maxTripCount);
    loop.createIterateOp();
    rewriter.setInsertionPointToStart(loop.getIterateBlock());

    // Create a scalar tensor out of iv, as the first argument passed to the
    // body graph function.
    Value origIV = loop.getInductionVar(0);
    auto iv = rewriter.create<IndexCastOp>(loc, origIV, rewriter.getI64Type())
                  .getResult();
    Value ivMemRef =
        rewriter
            .create<AllocOp>(loc, MemRefType::get({}, rewriter.getI64Type()))
            .getResult();
    rewriter.create<StoreOp>(loc, iv, ivMemRef);

    // Make the call to loop body function.
    SmallVector<Value, 4> params = {ivMemRef, loopOpAdapter.cond()};
    for (auto value : llvm::make_range(outputs.begin(),
             outputs.begin() + loopOpAdapter.v_initial().size()))
      params.emplace_back(value);

    auto callOp = rewriter.create<CallOp>(loc, func, params);

    // Post values from loop body function.
    auto resultsRange = callOp.getResults();
    SmallVector<Value, 4> bodyOutputs(resultsRange.begin(), resultsRange.end());

    for (int i = 0; i < bodyOutputs.size(); i++) {
      auto output = bodyOutputs[i];
      assert(output.getType().isa<TensorType>() ||
             output.getType().isa<MemRefType>() &&
                 "Expecting loop body function output to consist of "
                 "tensors/memrefs.");
      auto outputTy = output.getType().cast<ShapedType>();
      bodyOutputs[i] = rewriter
                           .create<KrnlDummyCastOp>(loc, output,
                               MemRefType::get(outputTy.getShape(),
                                   outputTy.getElementType()))
                           .getResult();
    }

    auto vIntermediate = llvm::make_range(bodyOutputs.begin() + 1,
        bodyOutputs.begin() + 1 + loopOpAdapter.v_initial().size());
    for (auto vIntermediateToFinal : llvm::zip(vIntermediate, outputs))
      EmitCopy(rewriter, loc, std::get<0>(vIntermediateToFinal),
          std::get<1>(vIntermediateToFinal));

    auto scanIntermediate =
        llvm::make_range(vIntermediate.end(), bodyOutputs.end());
    auto scanOutputs = llvm::make_range(
        outputs.begin() + loopOpAdapter.v_initial().size(), outputs.end());
    for (auto scanIntermediateToFinal :
        llvm::zip(scanIntermediate, scanOutputs))
      EmitCopy(rewriter, loc, std::get<0>(scanIntermediateToFinal),
          std::get<1>(scanIntermediateToFinal), {origIV});

    rewriter.replaceOp(op, outputs);
    return success();
  }

  void EmitCopy(ConversionPatternRewriter &rewriter, const Location &loc,
      const Value &vInit, const Value &vFinal,
      std::vector<Value> writePrefix = {}) const {
    OpBuilder::InsertionGuard insertGuard(rewriter);
    auto vInitTy = vInit.getType().cast<MemRefType>();
    BuildKrnlLoop loop(rewriter, loc, vInitTy.getRank());
    loop.createDefineOp();
    for (int i = 0; i < vInitTy.getRank(); i++)
      loop.pushBounds(0, vInit, i);
    loop.createIterateOp();
    rewriter.setInsertionPointToStart(loop.getIterateBlock());
    SmallVector<Value, 4> writeIV(writePrefix.begin(), writePrefix.end());
    auto readIV = loop.getAllInductionVar();
    writeIV.insert(writeIV.end(), readIV.begin(), readIV.end());
    auto v = rewriter.create<AffineLoadOp>(loc, vInit, readIV).getResult();
    rewriter.create<AffineStoreOp>(loc, v, vFinal, writeIV);
  }
};

void populateLoweringONNXLoopOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXLoopOpLowering>(ctx);
}