/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------- Loop.cpp - Lowering Loop Op ---------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Loop Operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

struct ONNXScanOpLowering : public ConversionPattern {
  explicit ONNXScanOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXScanOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = ONNXLoc<ONNXScanOp>(op);
    auto loopOp = dyn_cast<ONNXScanOp>(op);
    ONNXScanOpAdaptor loopOpAdapter(operands, op->getAttrDictionary());

    auto &loopBody = loopOp.body();
    auto bodyArgs = loopBody.getArguments();

    // Allocate memory for two kinds of outputs:
    // - final values of loop carried dependencies, and
    // - scan output (all intermediate values returned from body func
    // concatenated together).
    SmallVector<Value, 4> outputs;
    allocateMemoryForVFinal(loc, rewriter, op, loopOpAdapter, outputs);
    allocateMemoryForScanOutput(loc, rewriter, op, loopOpAdapter, outputs);

    // Copy content of vInit to vFinal, which is used to host intermediate
    // values produced by loop body function invocation in a scope accessible by
    // all loop iterations.
    auto v_initials = llvm::make_range(
        operands.begin(), operands.end() - loopOp.num_scan_inputs());
    for (const auto &vInitAndFinal : llvm::zip(v_initials, outputs))
      emitCopy(rewriter, loc, std::get<0>(vInitAndFinal),
          std::get<1>(vInitAndFinal));

    // Create the loop iteration.
    BuildKrnlLoop loop(rewriter, loc, 1);
    loop.createDefineOp();
    Value maxTripCount =
        rewriter.create<DimOp>(loc, loopOp.scan_inputs().front(), 0);

    loop.pushBounds(0, maxTripCount);
    loop.createIterateOp();
    rewriter.setInsertionPointToStart(loop.getIterateBlock());

    {

      OpBuilder::InsertionGuard insertGuard(rewriter);
      Value iv = loop.getInductionVar(0);

      // Initialize body graph parameter to be all the loop-carried
      // dependencies.
      SmallVector<Value, 4> params(
          outputs.begin(), outputs.begin() + loopOp.v_final().size());

      auto opScanInputRange = llvm::make_range(
          operands.end() - loopOp.num_scan_inputs(), operands.end());
      auto bodyScanInputRange = llvm::make_range(
          bodyArgs.end() - loopOp.num_scan_inputs(), bodyArgs.end());
      for (const auto &opAndBodyScanInput :
          llvm::zip(opScanInputRange, bodyScanInputRange)) {
        auto opScanInput = std::get<0>(opAndBodyScanInput);
        auto bodyScanInput = std::get<1>(opAndBodyScanInput);
        auto bodyScanInputMemRef = allocateMemoryForBodyScanInput(
            loopOp->getLoc(), rewriter, bodyScanInput.getType());
        params.emplace_back(bodyScanInputMemRef);
        emitCopyEx(
            rewriter, loopOp->getLoc(), opScanInput, bodyScanInputMemRef, {iv});
      }

      auto &loopBodyEntryBlock = loopBody.front();
      BlockAndValueMapping mapper;
      for (unsigned i = 0, e = params.size(); i != e; ++i) {
        params[i].dump();
        // Verify that the types of the provided values match the function
        // argument types.
        BlockArgument regionArg = loopBodyEntryBlock.getArgument(i);
        mapper.map(regionArg, params[i]);
      }

      auto *thenBlock = loop.getIterateBlock();
      auto &thenRegion = *thenBlock->getParent();

      // Split the insertion block into two, where the second block
      // `postInsertBlock` contains only the terminator operation, insert loop
      // body right before `postInsertBlock`, after all other operations created
      // within the if Op.
      Block *postInsertBlock =
          thenBlock->splitBlock(thenBlock->getTerminator());
      assert(loopBody.getBlocks().size() == 1 &&
             "Currently only support loop body with 1 block.");
      thenRegion.getBlocks().splice(
          postInsertBlock->getIterator(), loopBody.getBlocks());
      auto newBlocks = llvm::make_range(
          std::next(thenBlock->getIterator()), postInsertBlock->getIterator());
      auto &loopBodyBlock = *newBlocks.begin();
      auto loopBodyTerminator = loopBodyBlock.getTerminator();

      // Within inlined blocks, substitute reference to block arguments with
      // values produced by the lowered loop operation bootstrapping IR.
      auto remapOperands = [&](Operation *op1) {
        for (auto &operand : op1->getOpOperands())
          if (auto mappedOp = mapper.lookupOrNull(operand.get()))
            operand.set(mappedOp);
      };
      for (auto &block : thenRegion.getBlocks())
        block.walk(remapOperands);
      auto resultsRange =
          llvm::SmallVector<Value, 4>(loopBodyTerminator->getOperands().begin(),
              loopBodyTerminator->getOperands().end());
      rewriter.setInsertionPointToStart(postInsertBlock);

      // Cast loop body outputs from tensor type to memref type in case it has
      // not already been lowered via dummy_cast. Eventually, dummy cast becomes
      // a cast from memref type to a memref type when everything is lowered and
      // thus becomes redundant.
      SmallVector<Value, 4> bodyOutputs(
          resultsRange.begin(), resultsRange.end());
      for (int i = 0; i < bodyOutputs.size(); i++) {
        auto output = bodyOutputs[i];
        assert((output.getType().isa<TensorType>() ||
                   output.getType().isa<MemRefType>()) &&
               "Expecting loop body function output to consist of"
               "tensors/memrefs.");
        auto outputTy = output.getType().cast<ShapedType>();
        bodyOutputs[i] = rewriter
                             .create<KrnlDummyCastOp>(loc, output,
                                 MemRefType::get(outputTy.getShape(),
                                     outputTy.getElementType()))
                             .getResult();
      }

      // Copy intermediate values of loop carried dependencies to MemRef outside
      // the iteration scope so next iteration can have use them as init value.
      auto vIntermediate = llvm::make_range(
          bodyOutputs.begin(), bodyOutputs.begin() + loopOp.v_initial().size());
      for (auto vIntermediateToFinal : llvm::zip(vIntermediate, outputs))
        emitCopy(rewriter, loc, std::get<0>(vIntermediateToFinal),
            std::get<1>(vIntermediateToFinal));

      // Copy intermediate values of scan outputs to their corresponding slice
      // in the loop scan output tensor.
      auto scanIntermediate =
          llvm::make_range(vIntermediate.end(), bodyOutputs.end());
      auto scanOutputs = llvm::make_range(
          outputs.begin() + loopOp.v_initial().size(), outputs.end());
      for (auto scanIntermediateToFinal :
          llvm::zip(scanIntermediate, scanOutputs))
        emitCopy(rewriter, loc, std::get<0>(scanIntermediateToFinal),
            std::get<1>(scanIntermediateToFinal),
            /*writePrefix=*/{iv});

      // Remove loop body terminator op.
      rewriter.eraseOp(loopBodyTerminator);

      // Merge the post insert block into the cloned entry block.
      loopBodyBlock.getOperations().splice(
          loopBodyBlock.end(), postInsertBlock->getOperations());
      rewriter.eraseBlock(postInsertBlock);

      // Merge the loop body block into the then block.
      thenBlock->getOperations().splice(
          thenBlock->end(), loopBodyBlock.getOperations());
      rewriter.eraseBlock(&loopBodyBlock);
    }

    rewriter.replaceOp(op, outputs);
    return success();
  }

  void allocateMemoryForVFinal(mlir::Location loc,
      ConversionPatternRewriter &rewriter, Operation *op,
      ONNXScanOpAdaptor loopOpAdapter,
      SmallVectorImpl<mlir::Value> &outputs) const {
    auto loopOp = dyn_cast<ONNXScanOp>(op);
    for (const auto &ioPair : llvm::zip(loopOp.v_initial(), loopOp.v_final())) {
      auto vInit = std::get<0>(ioPair);
      auto vFinal = std::get<1>(ioPair);

      // Allocate memory for the loop-carried dependencies, since they are
      // guaranteed to have the same shape throughout all iterations, use their
      // initial value tensors as reference when allocating memory.
      auto memRefType = convertToMemRefType(vFinal.getType());
      Value alloc;
      bool shouldDealloc = checkInsertDealloc(op);
      if (hasAllConstantDimensions(memRefType))
        alloc = insertAllocAndDealloc(memRefType, loc, rewriter, shouldDealloc);
      else
        alloc = insertAllocAndDealloc(
            memRefType, loc, rewriter, shouldDealloc, vInit);
      outputs.emplace_back(alloc);
    }
  }

  void allocateMemoryForScanOutput(mlir::Location loc,
      ConversionPatternRewriter &rewriter, Operation *op,
      ONNXScanOpAdaptor loopOpAdapter,
      SmallVectorImpl<mlir::Value> &outputs) const {
    auto loopOp = dyn_cast<ONNXScanOp>(op);
    for (const auto &opScanOutput : loopOp.scan_outputs()) {
      // Allocate memory for the scan outputs. There're no good "reference"
      // shape for scan outputs. So if the scan outputs do not have constant
      // dimensions in all except the leading dimensions, we simply give up. The
      // leading dimension is simply the number of iterations executed, which is
      // easier to obtain.
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
              // equal to the max trip count, due to the possibility of early
              // termination.
              auto dim =
                  rewriter.create<DimOp>(loc, loopOp.scan_inputs().front(), 0);
              allocParams.emplace_back(dim);
            } else {
              // TODO(tjingrant): we can support dynamic dimensions for scan
              // output, however, then we will be unable to allocate memory
              // before any loop body is called.
              llvm_unreachable("Loop op doesn't support dynamic dimensions for "
                               "scan output.");
            }
          }
        }
        alloc = rewriter.create<AllocOp>(loc, rankedScanOutTy, allocParams);
      }
      outputs.emplace_back(alloc);
    }
  }

  mlir::Value allocateMemoryForBodyScanInput(mlir::Location loc,
      ConversionPatternRewriter &rewriter, mlir::Type bodyScanInputTy) const {
    // Allocate memory for the scan outputs. There're no good "reference"
    // shape for scan outputs. So if the scan outputs do not have constant
    // dimensions in all except the leading dimensions, we simply give up. The
    // leading dimension is simply the number of iterations executed, which is
    // easier to obtain.
    auto memRefType = convertToMemRefType(bodyScanInputTy);
    Value alloc;
    assert(hasAllConstantDimensions(memRefType) &&
           "Body scan input must have constant shape.");
    // TODO(tjingrant): Dealloc!!
    alloc =
        insertAllocAndDealloc(memRefType, loc, rewriter, /*shouldDealloc=*/0);
    return alloc;
  }

  // Helper function to emit code that copies data from src to dest.
  //
  // writePrefix enables copying to a contiguous subtensor of the same shape
  // within dest. For instance, we can copy a (4x2) tensor as the first tensor
  // into a higher dimensional tensor with shape (10x4x2), i.e., a batch of 10
  // tensors, each with shape (4x2). To do so, we can invoke emitCopy(src, dest,
  // {0}).
  void emitCopy(ConversionPatternRewriter &rewriter, const Location &loc,
      const Value &src, const Value &dest,
      std::vector<Value> writePrefix = {}) const {
    OpBuilder::InsertionGuard insertGuard(rewriter);

    auto srcTy = src.getType().cast<MemRefType>();
    SmallVector<Value, 4> readIV;
    if (srcTy.getRank() > 0) {
      BuildKrnlLoop loop(rewriter, loc, srcTy.getRank());
      loop.createDefineOp();
      for (int i = 0; i < srcTy.getRank(); i++)
        loop.pushBounds(0, src, i);
      loop.createIterateOp();
      rewriter.setInsertionPointToStart(loop.getIterateBlock());
      auto loopIVs = loop.getAllInductionVar();
      readIV = SmallVector<Value, 4>(loopIVs.begin(), loopIVs.end());
    }

    SmallVector<Value, 4> writeIV(writePrefix.begin(), writePrefix.end());
    writeIV.insert(writeIV.end(), readIV.begin(), readIV.end());
    auto val = rewriter.create<KrnlLoadOp>(loc, src, readIV).getResult();
    rewriter.create<KrnlStoreOp>(loc, val, dest, writeIV);
  }

  void emitCopyEx(ConversionPatternRewriter &rewriter, const Location &loc,
      const Value &src, const Value &dest,
      std::vector<Value> readPrefix = {}) const {
    OpBuilder::InsertionGuard insertGuard(rewriter);

    auto srcTy = src.getType().cast<MemRefType>();
    SmallVector<Value, 4> readIV(readPrefix.begin(), readPrefix.end());
    SmallVector<Value, 4> writeIV;
    if (srcTy.getRank() > readIV.size()) {
      BuildKrnlLoop loop(rewriter, loc, srcTy.getRank() - readPrefix.size());
      loop.createDefineOp();
      for (int i = readIV.size(); i < srcTy.getRank(); i++)
        loop.pushBounds(0, src, i);
      loop.createIterateOp();
      rewriter.setInsertionPointToStart(loop.getIterateBlock());
      auto IVs = loop.getAllInductionVar();
      writeIV.insert(writeIV.end(), IVs.begin(), IVs.end());
      readIV.insert(readIV.end(), writeIV.begin(), writeIV.end());
    }

    auto val = rewriter.create<KrnlLoadOp>(loc, src, readIV).getResult();
    rewriter.create<KrnlStoreOp>(loc, val, dest, writeIV);
  }
};

void populateLoweringONNXScanOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXScanOpLowering>(ctx);
}
