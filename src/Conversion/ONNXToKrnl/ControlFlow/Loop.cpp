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

struct ONNXLoopOpLowering : public ConversionPattern {
  explicit ONNXLoopOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXLoopOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = ONNXLoc<ONNXLoopOp>(op);
    auto loopOp = dyn_cast<ONNXLoopOp>(op);
    ONNXLoopOpAdaptor loopOpAdapter(operands, op->getAttrDictionary());

    auto &loopBody = loopOp.body();

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
    for (const auto &vInitAndFinal :
        llvm::zip(loopOpAdapter.v_initial(), outputs))
      emitCopy(rewriter, loc, std::get<0>(vInitAndFinal),
          std::get<1>(vInitAndFinal));

    // Create a memref for recording loop condition, initialize it with the
    // initial loop condition.
    auto condMemRefTy = convertToMemRefType(loopOpAdapter.cond().getType());
    Value cond;
    if (hasAllConstantDimensions(condMemRefTy))
      cond = insertAllocAndDealloc(
          condMemRefTy, loc, rewriter, /*insertDealloc=*/true);
    emitCopy(rewriter, loc, loopOpAdapter.cond(), cond);

    // Create the loop iteration.
    BuildKrnlLoop loop(rewriter, loc, 1);
    loop.createDefineOp();
    Value maxTripCount =
        rewriter.create<KrnlLoadOp>(loc, loopOpAdapter.M()).getResult();
    maxTripCount = rewriter.create<IndexCastOp>(
        loc, maxTripCount, rewriter.getIndexType());
    loop.pushBounds(0, maxTripCount);
    loop.createIterateOp();
    rewriter.setInsertionPointToStart(loop.getIterateBlock());

    {
      OpBuilder::InsertionGuard insertGuard(rewriter);

      auto condReg = rewriter.create<KrnlLoadOp>(loc, cond).getResult();
      auto ifOp = rewriter.create<scf::IfOp>(loc, condReg, false);
      rewriter.setInsertionPointToStart(&ifOp.thenRegion().front());

      // Create a scalar tensor out of loop iteration variable, as the first
      // argument passed to the body graph function.
      Value origIV = loop.getInductionVar(0);
      auto iv = rewriter.create<IndexCastOp>(loc, origIV, rewriter.getI64Type())
                    .getResult();
      Value ivMemRef = rewriter
                           .create<memref::AllocOp>(
                               loc, MemRefType::get({}, rewriter.getI64Type()))
                           .getResult();
      rewriter.create<KrnlStoreOp>(loc, iv, ivMemRef);

      // Make the call to loop body function.
      SmallVector<Value, 4> params = {ivMemRef, loopOpAdapter.cond()};
      for (auto value : llvm::make_range(outputs.begin(),
               outputs.begin() + loopOpAdapter.v_initial().size()))
        params.emplace_back(value);

      auto &loopBodyEntryBlock = loopBody.front();
      BlockAndValueMapping mapper;
      for (unsigned i = 0, e = params.size(); i != e; ++i) {
        // Verify that the types of the provided values match the function
        // argument types.
        BlockArgument regionArg = loopBodyEntryBlock.getArgument(i);
        mapper.map(regionArg, params[i]);
      }

      auto &thenRegion = ifOp.thenRegion();
      auto &thenBlock = thenRegion.front();

      // Split the insertion block into two, where the second block
      // `postInsertBlock` contains only the terminator operation, insert loop
      // body right before `postInsertBlock`, after all other operations created
      // within the if Op.
      Block *postInsertBlock = thenBlock.splitBlock(thenBlock.getTerminator());
      assert(loopBody.getBlocks().size() == 1 &&
             "Currently only support loop body with 1 block.");
      thenRegion.getBlocks().splice(
          postInsertBlock->getIterator(), loopBody.getBlocks());
      auto newBlocks = llvm::make_range(
          std::next(thenBlock.getIterator()), postInsertBlock->getIterator());
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
               "Expecting loop body function output to consist of "
               "tensors/memrefs.");
        auto outputTy = output.getType().cast<ShapedType>();
        bodyOutputs[i] = rewriter
                             .create<KrnlDummyCastOp>(loc, output,
                                 MemRefType::get(outputTy.getShape(),
                                     outputTy.getElementType()))
                             .getResult();
      }

      // Copy the newly computed loop condition to pre-allocated buffer.
      emitCopy(rewriter, loc, bodyOutputs[0], cond);

      // Copy intermediate values of loop carried dependencies to MemRef outside
      // the iteration scope so next iteration can have use them as init value.
      auto vIntermediate = llvm::make_range(bodyOutputs.begin() + 1,
          bodyOutputs.begin() + 1 + loopOpAdapter.v_initial().size());
      for (auto vIntermediateToFinal : llvm::zip(vIntermediate, outputs))
        emitCopy(rewriter, loc, std::get<0>(vIntermediateToFinal),
            std::get<1>(vIntermediateToFinal));

      // Copy intermediate values of scan outputs to their corresponding slice
      // in the loop scan output tensor.
      auto scanIntermediate =
          llvm::make_range(vIntermediate.end(), bodyOutputs.end());
      auto scanOutputs = llvm::make_range(
          outputs.begin() + loopOpAdapter.v_initial().size(), outputs.end());
      for (auto scanIntermediateToFinal :
          llvm::zip(scanIntermediate, scanOutputs))
        emitCopy(rewriter, loc, std::get<0>(scanIntermediateToFinal),
            std::get<1>(scanIntermediateToFinal),
            /*writePrefix=*/{origIV});

      // Remove loop body terminator op.
      rewriter.eraseOp(loopBodyTerminator);

      // Merge the post insert block into the cloned entry block.
      loopBodyBlock.getOperations().splice(
          loopBodyBlock.end(), postInsertBlock->getOperations());
      rewriter.eraseBlock(postInsertBlock);

      // Merge the loop body block into the then block.
      thenBlock.getOperations().splice(
          thenBlock.end(), loopBodyBlock.getOperations());
      rewriter.eraseBlock(&loopBodyBlock);
    }
    rewriter.replaceOp(op, outputs);
    return success();
  }

  void allocateMemoryForVFinal(mlir::Location loc,
      ConversionPatternRewriter &rewriter, Operation *op,
      ONNXLoopOpAdaptor loopOpAdapter,
      SmallVectorImpl<mlir::Value> &outputs) const {
    auto loopOp = dyn_cast<ONNXLoopOp>(op);
    for (const auto &ioPair :
        llvm::zip(loopOpAdapter.v_initial(), loopOp.v_final())) {
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
      ONNXLoopOpAdaptor loopOpAdapter,
      SmallVectorImpl<mlir::Value> &outputs) const {
    auto loopOp = dyn_cast<ONNXLoopOp>(op);
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
              assert(!loopOpAdapter.M().getType().isa<NoneType>());
              Value maxTripCount =
                  rewriter.create<KrnlLoadOp>(loc, loopOpAdapter.M())
                      .getResult();
              allocParams.emplace_back(rewriter.create<IndexCastOp>(
                  loc, maxTripCount, rewriter.getIndexType()));
            } else {
              // TODO(tjingrant): we can support dynamic dimensions for scan
              // output, however, then we will be unable to allocate memory
              // before any loop body is called.
              llvm_unreachable("Loop op doesn't support dynamic dimensions for "
                               "scan output.");
            }
          }
        }
        alloc =
            rewriter.create<memref::AllocOp>(loc, rankedScanOutTy, allocParams);
      }
      outputs.emplace_back(alloc);
    }
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
      // Do not create defineLoo
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
};

void populateLoweringONNXLoopOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXLoopOpLowering>(ctx);
}
