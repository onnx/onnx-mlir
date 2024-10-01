/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------- Scan.cpp - Lowering Scan Op ---------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Scan Operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXScanOpLowering : public OpConversionPattern<ONNXScanOp> {
  explicit ONNXScanOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXScanOp scanOp, ONNXScanOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = scanOp.getOperation();
    Location loc = ONNXLoc<ONNXScanOp>(op);
    ValueRange operands = adaptor.getOperands();

    auto &scanBody = scanOp.getBody();
    auto bodyArgs = scanBody.getArguments();

    // Allocate memory for two kinds of outputs:
    // - final values of scan carried dependencies, and
    // - scan output (all intermediate values returned from body func
    // concatenated together).
    SmallVector<Value, 4> outputs;
    allocateMemoryForVFinal(loc, rewriter, typeConverter, op, adaptor, outputs);
    allocateMemoryForScanOutput(
        loc, rewriter, typeConverter, op, adaptor, outputs);

    // Copy content of vInit to vFinal, which is used to host intermediate
    // values produced by scan body function invocation in a scope accessible
    // by all scan iterations.
    int64_t numInputs = scanOp.getNumScanInputs();
    auto v_initials = llvm::make_range(
        operands.begin(), operands.begin() + (operands.size() - numInputs));
    for (const auto &vInitAndFinal : llvm::zip(v_initials, outputs))
      emitCopy(rewriter, loc, std::get<0>(vInitAndFinal),
          std::get<1>(vInitAndFinal));

    auto inputOperands = llvm::make_range(
        operands.begin() + (operands.size() - numInputs), operands.end());
    MemRefBuilder createMemRef(rewriter, loc);
    Value maxTripCount = createMemRef.dim(*inputOperands.begin(), 0);

    // Create the scan iteration.
    std::vector<Value> loop;
    defineLoops(rewriter, loc, loop, 1);
    krnl::KrnlIterateOperandPack pack(rewriter, loop);
    pack.pushConstantBound(0);
    pack.pushOperandBound(maxTripCount);
    KrnlBuilder createKrnl(rewriter, loc);
    KrnlIterateOp iterateOp = createKrnl.iterate(pack);
    Block &iterationBlock = iterateOp.getBodyRegion().front();
    rewriter.setInsertionPointToStart(&iterationBlock);

    {
      OpBuilder::InsertionGuard insertGuard(rewriter);
      Value iv = *iterationBlock.getArguments().begin();

      // Initialize scan body function parameter to be all the
      // loop-carried dependencies.
      SmallVector<Value, 4> params(
          outputs.begin(), outputs.begin() + scanOp.v_final().size());
      // Variables local to the subgraph.
      SmallVector<Value, 4> localVars;

      auto opScanInputRange = llvm::make_range(
          operands.begin() + (operands.size() - numInputs), operands.end());
      auto bodyScanInputRange = llvm::make_range(
          bodyArgs.begin() + (bodyArgs.size() - numInputs), bodyArgs.end());
      for (const auto &opAndBodyScanInput :
          llvm::zip(opScanInputRange, bodyScanInputRange)) {
        auto opScanInput = std::get<0>(opAndBodyScanInput);
        auto bodyScanInput = std::get<1>(opAndBodyScanInput);
        auto bodyScanInputMemRef = allocateMemoryForBodyScanInput(
            scanOp->getLoc(), rewriter, typeConverter, bodyScanInput.getType());
        emitCopyFromTensorSlice(
            rewriter, scanOp->getLoc(), opScanInput, bodyScanInputMemRef, {iv});
        params.emplace_back(bodyScanInputMemRef);
        localVars.emplace_back(bodyScanInputMemRef);
      }

      auto &scanBodyEntryBlock = scanBody.front();
      IRMapping mapper;
      for (unsigned i = 0, e = params.size(); i != e; ++i) {
        // Verify that the types of the provided values match the function
        // argument types.
        BlockArgument regionArg = scanBodyEntryBlock.getArgument(i);
        mapper.map(regionArg, params[i]);
      }

      Block *loopBodyBlock = &iterationBlock;
      Region &loopBodyRegion = *loopBodyBlock->getParent();

      // Split the insertion block into two, where the second block
      // `postInsertBlock` contains only the terminator operation, insert scan
      // body right before `postInsertBlock`, after all other operations
      // created within the if Op.
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
        assert((mlir::isa<TensorType>(output.getType()) ||
                   mlir::isa<MemRefType>(output.getType())) &&
               "Expecting scan body function output to consist of"
               "tensors/memrefs.");
        auto outputTy = mlir::cast<ShapedType>(output.getType());
        bodyOutputs[i] = rewriter
                             .create<UnrealizedConversionCastOp>(loc,
                                 MemRefType::get(outputTy.getShape(),
                                     outputTy.getElementType()),
                                 output)
                             .getResult(0);
      }

      // Copy intermediate values of scan carried dependencies to MemRef
      // outside the iteration scope so next iteration can have use them as
      // init value.
      auto vIntermediate = llvm::make_range(bodyOutputs.begin(),
          bodyOutputs.begin() + scanOp.getVInitial().size());
      for (auto vIntermediateToFinal : llvm::zip(vIntermediate, outputs))
        emitCopy(rewriter, loc, std::get<0>(vIntermediateToFinal),
            std::get<1>(vIntermediateToFinal));

      // Copy intermediate values of scan outputs to their corresponding slice
      // in the scan scan output tensor.
      auto scanIntermediate =
          llvm::make_range(vIntermediate.end(), bodyOutputs.end());
      auto scanOutputs = llvm::make_range(
          outputs.begin() + scanOp.getVInitial().size(), outputs.end());
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
    onnxToKrnlSimdReport(op);
    return success();
  }

  static void allocateMemoryForVFinal(Location loc,
      ConversionPatternRewriter &rewriter, const TypeConverter *typeConverter,
      Operation *op, ONNXScanOpAdaptor adaptor,
      SmallVectorImpl<Value> &outputs) {
    auto scanOp = mlir::dyn_cast<ONNXScanOp>(op);
    for (const auto &ioPair :
        llvm::zip(scanOp.getVInitial(), scanOp.v_final())) {
      auto vInit = std::get<0>(ioPair);
      auto vFinal = std::get<1>(ioPair);

      // Convert vFinal's type to MemRefType.
      Type convertedType = typeConverter->convertType(vFinal.getType());
      assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
             "Failed to convert type to MemRefType");
      MemRefType memRefType = mlir::cast<MemRefType>(convertedType);

      // Allocate memory for the loop-carried dependencies, since they are
      // guaranteed to have the same shape throughout all iterations, use
      // their initial value tensors as reference when allocating memory.
      MultiDialectBuilder<MemRefBuilder> create(rewriter, loc);
      Value alloc = create.mem.alignedAlloc(vInit, memRefType);
      outputs.emplace_back(alloc);
    }
  }

  static void allocateMemoryForScanOutput(Location loc,
      ConversionPatternRewriter &rewriter, const TypeConverter *typeConverter,
      Operation *op, ONNXScanOpAdaptor adaptor,
      SmallVectorImpl<Value> &outputs) {
    auto scanOp = mlir::dyn_cast<ONNXScanOp>(op);
    for (const auto &opScanOutput : scanOp.scan_outputs()) {
      // Convert opScanOutput's type to MemRefType.
      Type convertedType = typeConverter->convertType(opScanOutput.getType());
      assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
             "Failed to convert type to MemRefType");
      MemRefType memRefType = mlir::cast<MemRefType>(convertedType);

      // Allocate memory for the scan outputs. There're no good "reference"
      // shape for scan outputs. So if the scan outputs do not have constant
      // dimensions in all except the leading dimensions, we simply give up.
      // The leading dimension is simply the number of iterations executed,
      // which is easier to obtain.
      Value alloc;
      MemRefBuilder createMemRef(rewriter, loc);
      if (hasAllConstantDimensions(memRefType))
        alloc = createMemRef.alignedAlloc(memRefType);
      else {
        MemRefBuilder createMemRef(rewriter, loc);
        OnnxBuilder onnxBuilder(rewriter, loc);
        auto rankedScanOutTy = memRefType;
        SmallVector<Value, 4> allocParams;
        for (int i = 0; i < rankedScanOutTy.getRank(); i++) {
          if (rankedScanOutTy.isDynamicDim(i)) {
            if (i == 0) {
              // TODO(tjingrant): in general, it is not correct to expect
              // scan operation scan output to have the leading dimension
              // extent equal to the max trip count, due to the possibility of
              // early termination.
              Value val = onnxBuilder.toMemref(scanOp.scan_inputs().front());
              auto dim = createMemRef.dim(val, 0);
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

  static Value allocateMemoryForBodyScanInput(Location loc,
      ConversionPatternRewriter &rewriter, const TypeConverter *typeConverter,
      Type bodyScanInputTy) {
    // Convert type to MemRefType.
    Type convertedType = typeConverter->convertType(bodyScanInputTy);
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = mlir::cast<MemRefType>(convertedType);

    // Allocate memory for the scan outputs. There're no good "reference"
    // shape for scan outputs. So if the scan outputs do not have constant
    // dimensions in all except the leading dimensions, we simply give up. The
    // leading dimension is simply the number of iterations executed, which is
    // easier to obtain.
    Value alloc;
    assert(hasAllConstantDimensions(memRefType) &&
           "Body scan input must have constant shape.");
    // TODO(tjingrant): figure out why insertDealloc=1 doesn't work. Our
    // current mechanism for pulling the dealloc to the end of function
    // doesn't work alongside subgraph inlining.
    MultiDialectBuilder<MemRefBuilder> create(rewriter, loc);
    alloc = create.mem.alignedAlloc(memRefType);
    return alloc;
  }

  // Helper function to emit code that copies data from src to dest.
  //
  // writePrefix enables copying to a contiguous subtensor of the same shape
  // within dest. For instance, we can copy a (4x2) tensor as the first tensor
  // into a higher dimensional tensor with shape (10x4x2), i.e., a batch of 10
  // tensors, each with shape (4x2). To do so, we can invoke emitCopy(src,
  // dest, {0}).
  static void emitCopy(OpBuilder &builder, const Location &loc,
      const Value &src, const Value &dest,
      std::vector<Value> writePrefix = {}) {
    OpBuilder::InsertionGuard insertGuard(builder);

    auto srcTy = mlir::cast<MemRefType>(src.getType());
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl> create(
        builder, loc);
    if (srcTy.getRank() > 0) {
      IndexExprScope childScope(create.krnl);
      ValueRange loopDef = create.krnl.defineLoops(srcTy.getRank());
      SmallVector<IndexExpr, 4> lbs(srcTy.getRank(), LitIE(0));
      SmallVector<IndexExpr, 4> ubs;
      create.krnlIE.getShapeAsDims(src, ubs);
      create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
          [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
            SmallVector<Value, 4> writeIV(
                writePrefix.begin(), writePrefix.end());
            writeIV.insert(writeIV.end(), loopInd.begin(), loopInd.end());

            Value val = createKrnl.load(src, loopInd);
            createKrnl.store(val, dest, writeIV);
          });
    } else {
      Value val = create.krnl.load(src);
      create.krnl.store(val, dest, writePrefix);
    }
  }

  static void emitCopyFromTensorSlice(OpBuilder &builder, const Location &loc,
      const Value &src, const Value &dest, std::vector<Value> readPrefix = {}) {
    OpBuilder::InsertionGuard insertGuard(builder);

    auto srcTy = mlir::cast<MemRefType>(src.getType());
    SmallVector<Value, 4> readIV(readPrefix.begin(), readPrefix.end());
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl> create(
        builder, loc);
    if (static_cast<size_t>(srcTy.getRank()) > readIV.size()) {
      IndexExprScope childScope(create.krnl);
      ValueRange loopDef =
          create.krnl.defineLoops(srcTy.getRank() - readPrefix.size());
      SmallVector<IndexExpr, 4> lbs(
          srcTy.getRank() - readPrefix.size(), LitIE(0));
      SmallVector<IndexExpr, 4> ubs;
      for (int i = readIV.size(); i < srcTy.getRank(); i++)
        ubs.emplace_back(create.krnlIE.getShapeAsDim(src, i));
      create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
          [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
            readIV.insert(readIV.end(), loopInd.begin(), loopInd.end());
            Value val = createKrnl.load(src, readIV);
            createKrnl.store(val, dest, loopInd);
          });
    } else {
      Value val = create.krnl.load(src, readIV);
      create.krnl.store(val, dest);
    }
  }
};

void populateLoweringONNXScanOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXScanOpLowering>(typeConverter, ctx);
}
} // namespace onnx_mlir
