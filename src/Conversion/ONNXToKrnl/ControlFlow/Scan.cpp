/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------- Scan.cpp - Lowering Scan Op ---------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Scan Operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

struct ONNXScanOpLowering : public ConversionPattern {
  explicit ONNXScanOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXScanOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = ONNXLoc<ONNXScanOp>(op);
    auto scanOp = dyn_cast<ONNXScanOp>(op);
    ONNXScanOpAdaptor scanOpAdapter(operands, op->getAttrDictionary());

    auto &scanBody = scanOp.body();
    auto bodyArgs = scanBody.getArguments();

    // Allocate memory for two kinds of outputs:
    // - final values of scan carried dependencies, and
    // - scan output (all intermediate values returned from body func
    // concatenated together).
    SmallVector<Value, 4> outputs;
    allocateMemoryForVFinal(loc, rewriter, op, scanOpAdapter, outputs);
    allocateMemoryForScanOutput(loc, rewriter, op, scanOpAdapter, outputs);

    // Copy content of vInit to vFinal, which is used to host intermediate
    // values produced by scan body function invocation in a scope accessible by
    // all scan iterations.
    int64_t numInputs = scanOp.num_scan_inputs();
    auto v_initials =
        llvm::make_range(operands.begin(), operands.end() - numInputs);
    for (const auto &vInitAndFinal : llvm::zip(v_initials, outputs))
      emitCopy(rewriter, loc, std::get<0>(vInitAndFinal),
          std::get<1>(vInitAndFinal));

    auto inputOperands = llvm::make_range(
        operands.begin() + (operands.size() - numInputs), operands.end());
    MemRefBuilder createMemRef(rewriter, loc);
    Value maxTripCount = createMemRef.dim(*inputOperands.begin(), 0);

    // Create the scan iteration.
    BuildKrnlLoop loop(rewriter, loc, 1);
    loop.createDefineOp();
    loop.pushBounds(0, maxTripCount);
    loop.createIterateOp();
    rewriter.setInsertionPointToStart(loop.getIterateBlock());

    {
      OpBuilder::InsertionGuard insertGuard(rewriter);
      Value iv = loop.getInductionVar(0);

      // Initialize scan body function parameter to be all the
      // loop-carried dependencies.
      SmallVector<Value, 4> params(
          outputs.begin(), outputs.begin() + scanOp.v_final().size());
      // Variables local to the subgraph.
      SmallVector<Value, 4> localVars;

      auto opScanInputRange = llvm::make_range(
          operands.end() - scanOp.num_scan_inputs(), operands.end());
      auto bodyScanInputRange = llvm::make_range(
          bodyArgs.end() - scanOp.num_scan_inputs(), bodyArgs.end());
      for (const auto &opAndBodyScanInput :
          llvm::zip(opScanInputRange, bodyScanInputRange)) {
        auto opScanInput = std::get<0>(opAndBodyScanInput);
        auto bodyScanInput = std::get<1>(opAndBodyScanInput);
        auto bodyScanInputMemRef = allocateMemoryForBodyScanInput(
            scanOp->getLoc(), rewriter, bodyScanInput.getType());
        emitCopyFromTensorSlice(
            rewriter, scanOp->getLoc(), opScanInput, bodyScanInputMemRef, {iv});
        params.emplace_back(bodyScanInputMemRef);
        localVars.emplace_back(bodyScanInputMemRef);
      }

      auto &scanBodyEntryBlock = scanBody.front();
      BlockAndValueMapping mapper;
      for (unsigned i = 0, e = params.size(); i != e; ++i) {
        // Verify that the types of the provided values match the function
        // argument types.
        BlockArgument regionArg = scanBodyEntryBlock.getArgument(i);
        mapper.map(regionArg, params[i]);
      }

      auto *loopBodyBlock = loop.getIterateBlock();
      auto &loopBodyRegion = *loopBodyBlock->getParent();

      // Split the insertion block into two, where the second block
      // `postInsertBlock` contains only the terminator operation, insert scan
      // body right before `postInsertBlock`, after all other operations created
      // within the if Op.
      Block *postInsertBlock =
          loopBodyBlock->splitBlock(loopBodyBlock->getTerminator());
      assert(scanBody.getBlocks().size() == 1 &&
             "Currently only support scan body with 1 block.");
      loopBodyRegion.getBlocks().splice(
          postInsertBlock->getIterator(), scanBody.getBlocks());
      auto newBlocks = llvm::make_range(std::next(loopBodyBlock->getIterator()),
          postInsertBlock->getIterator());
      auto &scanBodyBlock = *newBlocks.begin();
      auto scanBodyTerminator = scanBodyBlock.getTerminator();

      // Within inlined blocks, substitute reference to block arguments with
      // values produced by the lowered scan operation bootstrapping IR.
      auto remapOperands = [&](Operation *op1) {
        for (auto &operand : op1->getOpOperands())
          if (auto mappedOp = mapper.lookupOrNull(operand.get()))
            operand.set(mappedOp);
      };
      for (auto &block : loopBodyRegion.getBlocks())
        block.walk(remapOperands);
      auto resultsRange =
          llvm::SmallVector<Value, 4>(scanBodyTerminator->getOperands().begin(),
              scanBodyTerminator->getOperands().end());
      rewriter.setInsertionPointToStart(postInsertBlock);

      // Cast scan body outputs from tensor type to memref type in case it has
      // not already been lowered. Eventually, 'UnrealizedConversionCastOp'
      // becomes a cast from memref type to a memref type when everything is
      // lowered and thus becomes redundant.
      SmallVector<Value, 4> bodyOutputs(
          resultsRange.begin(), resultsRange.end());
      for (unsigned i = 0; i < bodyOutputs.size(); i++) {
        auto output = bodyOutputs[i];
        assert((output.getType().isa<TensorType>() ||
                   output.getType().isa<MemRefType>()) &&
               "Expecting scan body function output to consist of"
               "tensors/memrefs.");
        auto outputTy = output.getType().cast<ShapedType>();
        bodyOutputs[i] = rewriter
                             .create<UnrealizedConversionCastOp>(loc,
                                 MemRefType::get(outputTy.getShape(),
                                     outputTy.getElementType()),
                                 output)
                             .getResult(0);
      }

      // Copy intermediate values of scan carried dependencies to MemRef outside
      // the iteration scope so next iteration can have use them as init value.
      auto vIntermediate = llvm::make_range(
          bodyOutputs.begin(), bodyOutputs.begin() + scanOp.v_initial().size());
      for (auto vIntermediateToFinal : llvm::zip(vIntermediate, outputs))
        emitCopy(rewriter, loc, std::get<0>(vIntermediateToFinal),
            std::get<1>(vIntermediateToFinal));

      // Copy intermediate values of scan outputs to their corresponding slice
      // in the scan scan output tensor.
      auto scanIntermediate =
          llvm::make_range(vIntermediate.end(), bodyOutputs.end());
      auto scanOutputs = llvm::make_range(
          outputs.begin() + scanOp.v_initial().size(), outputs.end());
      for (auto scanIntermediateToFinal :
          llvm::zip(scanIntermediate, scanOutputs))
        emitCopy(rewriter, loc, std::get<0>(scanIntermediateToFinal),
            std::get<1>(scanIntermediateToFinal),
            /*writePrefix=*/{iv});

      // Remove scan body terminator op.
      rewriter.eraseOp(scanBodyTerminator);

      // Merge the post insert block into the cloned entry block.
      scanBodyBlock.getOperations().splice(
          scanBodyBlock.end(), postInsertBlock->getOperations());
      rewriter.eraseBlock(postInsertBlock);

      // Merge the scan body block into the then block.
      loopBodyBlock->getOperations().splice(
          loopBodyBlock->end(), scanBodyBlock.getOperations());
      rewriter.eraseBlock(&scanBodyBlock);
    }

    rewriter.replaceOp(op, outputs);
    return success();
  }

  static void allocateMemoryForVFinal(mlir::Location loc,
      ConversionPatternRewriter &rewriter, Operation *op,
      ONNXScanOpAdaptor scanOpAdapter, SmallVectorImpl<mlir::Value> &outputs) {
    auto scanOp = dyn_cast<ONNXScanOp>(op);
    for (const auto &ioPair : llvm::zip(scanOp.v_initial(), scanOp.v_final())) {
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

  static void allocateMemoryForScanOutput(mlir::Location loc,
      ConversionPatternRewriter &rewriter, Operation *op,
      ONNXScanOpAdaptor scanOpAdapter, SmallVectorImpl<mlir::Value> &outputs) {
    auto scanOp = dyn_cast<ONNXScanOp>(op);
    for (const auto &opScanOutput : scanOp.scan_outputs()) {
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
        MemRefBuilder createMemRef(rewriter, loc);
        auto rankedScanOutTy = memRefType;
        SmallVector<mlir::Value, 4> allocParams;
        for (int i = 0; i < rankedScanOutTy.getRank(); i++) {
          if (rankedScanOutTy.getShape()[i] == -1) {
            if (i == 0) {
              // TODO(tjingrant): in general, it is not correct to expect
              // scan operation scan output to have the leading dimension extent
              // equal to the max trip count, due to the possibility of early
              // termination.
              auto dim = createMemRef.dim(scanOp.scan_inputs().front(), 0);
              allocParams.emplace_back(dim);
            } else {
              // TODO(tjingrant): we can support dynamic dimensions for scan
              // output, however, then we will be unable to allocate memory
              // before any scan body is called.
              llvm_unreachable("Scan op doesn't support dynamic dimensions for "
                               "scan output.");
            }
          }
        }
        alloc = createMemRef.alignedAlloc(rankedScanOutTy, allocParams);
      }
      outputs.emplace_back(alloc);
    }
  }

  static mlir::Value allocateMemoryForBodyScanInput(mlir::Location loc,
      ConversionPatternRewriter &rewriter, mlir::Type bodyScanInputTy) {
    // Allocate memory for the scan outputs. There're no good "reference"
    // shape for scan outputs. So if the scan outputs do not have constant
    // dimensions in all except the leading dimensions, we simply give up. The
    // leading dimension is simply the number of iterations executed, which is
    // easier to obtain.
    auto memRefType = convertToMemRefType(bodyScanInputTy);
    Value alloc;
    assert(hasAllConstantDimensions(memRefType) &&
           "Body scan input must have constant shape.");
    // TODO(tjingrant): figure out why insertDealloc=1 doesn't work. Our current
    // mechanism for pulling the dealloc to the end of function doesn't work
    // alongside subgraph inlining.
    alloc =
        insertAllocAndDealloc(memRefType, loc, rewriter, /*insertDealloc=*/0);
    return alloc;
  }

  // Helper function to emit code that copies data from src to dest.
  //
  // writePrefix enables copying to a contiguous subtensor of the same shape
  // within dest. For instance, we can copy a (4x2) tensor as the first tensor
  // into a higher dimensional tensor with shape (10x4x2), i.e., a batch of 10
  // tensors, each with shape (4x2). To do so, we can invoke emitCopy(src, dest,
  // {0}).
  static void emitCopy(OpBuilder &builder, const Location &loc,
      const Value &src, const Value &dest,
      std::vector<Value> writePrefix = {}) {
    OpBuilder::InsertionGuard insertGuard(builder);

    auto srcTy = src.getType().cast<MemRefType>();
    SmallVector<Value, 4> readIV;
    if (srcTy.getRank() > 0) {
      BuildKrnlLoop loop(builder, loc, srcTy.getRank());
      loop.createDefineOp();
      for (int i = 0; i < srcTy.getRank(); i++)
        loop.pushBounds(0, src, i);
      loop.createIterateOp();
      builder.setInsertionPointToStart(loop.getIterateBlock());
      auto loopIVs = loop.getAllInductionVar();
      readIV = SmallVector<Value, 4>(loopIVs.begin(), loopIVs.end());
    }

    SmallVector<Value, 4> writeIV(writePrefix.begin(), writePrefix.end());
    writeIV.insert(writeIV.end(), readIV.begin(), readIV.end());

    KrnlBuilder createKrnl(builder, loc);
    Value val = createKrnl.load(src, readIV);
    createKrnl.store(val, dest, writeIV);
  }

  static void emitCopyFromTensorSlice(OpBuilder &builder, const Location &loc,
      const Value &src, const Value &dest, std::vector<Value> readPrefix = {}) {
    OpBuilder::InsertionGuard insertGuard(builder);

    auto srcTy = src.getType().cast<MemRefType>();
    SmallVector<Value, 4> readIV(readPrefix.begin(), readPrefix.end());
    SmallVector<Value, 4> writeIV;
    if ((size_t)srcTy.getRank() > readIV.size()) {
      BuildKrnlLoop loop(builder, loc, srcTy.getRank() - readPrefix.size());
      loop.createDefineOp();
      for (int i = readIV.size(); i < srcTy.getRank(); i++)
        loop.pushBounds(0, src, i);
      loop.createIterateOp();
      builder.setInsertionPointToStart(loop.getIterateBlock());
      auto IVs = loop.getAllInductionVar();
      writeIV.insert(writeIV.end(), IVs.begin(), IVs.end());
      readIV.insert(readIV.end(), writeIV.begin(), writeIV.end());
    }

    KrnlBuilder createKrnl(builder, loc);
    Value val = createKrnl.load(src, readIV);
    createKrnl.store(val, dest, writeIV);
  }
};

void populateLoweringONNXScanOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXScanOpLowering>(typeConverter, ctx);
}
