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
#include "src/Dialect/Krnl/DialectBuilder.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXLoopOpLowering : public ConversionPattern {
  explicit ONNXLoopOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXLoopOp::getOperationName(), 1, ctx) {}

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
    MultiDialectBuilder<KrnlBuilder, MemRefBuilder, MathBuilder> create(
        rewriter, loc);

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
    krnl::BuildKrnlLoop loop(rewriter, loc, 1);
    loop.createDefineOp();
    KrnlBuilder createKrnl(rewriter, loc);
    Value maxTripCount = createKrnl.load(loopOpAdapter.M());

    maxTripCount = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), maxTripCount);
    loop.pushBounds(0, maxTripCount);
    loop.createIterateOp();
    auto afterLoop = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(loop.getIterateBlock());

    {
      OpBuilder::InsertionGuard insertGuard(rewriter);

      Value condReg = createKrnl.load(cond);
      auto ifOp = rewriter.create<scf::IfOp>(loc, condReg, false);
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());

      // Contain the ThenRegion with KrnlRegionOp
      // The krnl loop inside may use symbol computed in LoopOp body
      KrnlRegionOp regionOp = rewriter.create<KrnlRegionOp>(loc);
      rewriter.setInsertionPointToStart(&regionOp.bodyRegion().front());

      // Create a scalar tensor out of loop iteration variable, as the first
      // argument passed to the body graph function.
      Value origIV = loop.getInductionVar(0);
      auto iv =
          rewriter
              .create<arith::IndexCastOp>(loc, rewriter.getI64Type(), origIV)
              .getResult();
      MemRefBuilder createMemRef(rewriter, loc);
      Value ivMemRef =
          createMemRef.alloc(MemRefType::get({}, rewriter.getI64Type()));
      createKrnl.store(iv, ivMemRef);

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

      // Previous code is intended to create the loop body without RegionOp
      // Current code just splice the loopBody into the RegionOp after it was
      // built.
      // ToDo: code could be simplified if not built on top of the previous code

      auto &thenRegion = ifOp.getThenRegion();
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

      // Add the loop end code into the loopBodyBlock
      // Previous code adds them to postInsertBlock
      // rewriter.setInsertionPointToStart(postInsertBlock);
      rewriter.setInsertionPointToEnd(&loopBodyBlock);

      // Cast loop body outputs from tensor type to memref type in case it has
      // not already been lowered. Eventually, 'UnrealizedConversionCastOp'
      // becomes a cast from memref type to a memref type when everything is
      // lowered and thus becomes redundant.
      SmallVector<Value, 4> bodyOutputs(
          resultsRange.begin(), resultsRange.end());
      for (unsigned i = 0; i < bodyOutputs.size(); i++) {
        auto output = bodyOutputs[i];
        assert((output.getType().isa<TensorType>() ||
                   output.getType().isa<MemRefType>()) &&
               "Expecting loop body function output to consist of "
               "tensors/memrefs.");
        auto outputTy = output.getType().cast<ShapedType>();
        bodyOutputs[i] = rewriter
                             .create<UnrealizedConversionCastOp>(loc,
                                 MemRefType::get(outputTy.getShape(),
                                     outputTy.getElementType()),
                                 output)
                             .getResult(0);
      }

      // Copy the newly computed loop condition to pre-allocated buffer.
      emitCopy(rewriter, loc, bodyOutputs[0], cond);

      // Copy intermediate values of scan outputs to their corresponding slice
      // in the loop scan output tensor.
      auto vIntermediate = llvm::make_range(bodyOutputs.begin() + 1,
          bodyOutputs.begin() + 1 + loopOpAdapter.v_initial().size());
      auto scanIntermediate =
          llvm::make_range(vIntermediate.end(), bodyOutputs.end());
      auto scanOutputs = llvm::make_range(
          outputs.begin() + loopOpAdapter.v_initial().size(), outputs.end());
      for (auto scanIntermediateToFinal :
          llvm::zip(scanIntermediate, scanOutputs)) {
        auto elementType = std::get<1>(scanIntermediateToFinal)
                               .getType()
                               .cast<MemRefType>()
                               .getElementType();
        if (elementType.dyn_cast<MemRefType>()) {
          // accumulate dynamic tensor
          rewriter.create<KrnlSeqStoreOp>(loc,
              std::get<0>(scanIntermediateToFinal),
              std::get<1>(scanIntermediateToFinal), origIV);
        } else {
          emitCopy(rewriter, loc, std::get<0>(scanIntermediateToFinal),
              std::get<1>(scanIntermediateToFinal),
              /*writePrefix=*/{origIV});
        }
      }

      // Copy intermediate values of loop carried dependencies to MemRef outside
      // the iteration scope so next iteration can use them as init value.
      for (auto vIntermediateToFinal : llvm::zip(vIntermediate, outputs))
        emitCopy(rewriter, loc, std::get<0>(vIntermediateToFinal),
            std::get<1>(vIntermediateToFinal));

      // Remove loop body terminator op.
      rewriter.eraseOp(loopBodyTerminator);

      // Merge the post block (with only Terminator) into the thenBody
      thenBlock.getOperations().splice(
          thenBlock.end(), postInsertBlock->getOperations());
      rewriter.eraseBlock(postInsertBlock);

      // Merge the loop body block into the RegionOp.
      regionOp.bodyRegion().front().getOperations().splice(
          regionOp.bodyRegion().front().end(), loopBodyBlock.getOperations());
      rewriter.eraseBlock(&loopBodyBlock);
    }

    rewriter.restoreInsertionPoint(afterLoop);
    // accumulate dynamic tensor
    // Convert the memref<?xmemref<>> to a memref
    SmallVector<Value, 4> newOutputs;
    for (auto output : outputs) {
      auto seqElementType =
          output.getType().cast<MemRefType>().getElementType();
      if (seqElementType.isa<MemRefType>()) {
        // need to convert memref<memrefs<xT>> to memref<xT>
        // TODO: need a IF statement to handle output is empty
        // we can safely give 0 to the dynamic dim for alloc
        // Here loop is assumed to be executed at least once.
        auto firstElement =
            create.krnl.load(output, create.math.constantIndex(0));
        SmallVector<mlir::Value, 4> allocParams;
        SmallVector<int64_t, 4> dims;
        dims.emplace_back(output.getType().cast<MemRefType>().getShape()[0]);
        if (output.getType().cast<MemRefType>().getShape()[0] == -1)
          allocParams.emplace_back(create.mem.dim(output, 0));
        for (auto i = 0;
             i < firstElement.getType().cast<MemRefType>().getRank(); i++) {
          dims.emplace_back(
              firstElement.getType().cast<MemRefType>().getShape()[i]);
          if (firstElement.getType().cast<MemRefType>().getShape()[i] == -1)
            allocParams.emplace_back(create.mem.dim(firstElement, i));
        }
        ArrayRef<int64_t> shape(dims.data(), dims.size());
        auto flatType = MemRefType::get(
            shape, firstElement.getType().cast<MemRefType>().getElementType());
        auto alloc = create.mem.alignedAlloc(flatType, allocParams);
        // copy the value
        krnl::BuildKrnlLoop loop(rewriter, loc, 1);
        loop.createDefineOp();
        loop.pushBounds(0, maxTripCount);
        loop.createIterateOp();
        auto afterCopyLoop = rewriter.saveInsertionPoint();
        rewriter.setInsertionPointToStart(loop.getIterateBlock());
        // Wrap with KrnlRegionOp because emitCopy uses the result of SeqExtract
        // for loop bound.
        KrnlRegionOp regionOp = rewriter.create<KrnlRegionOp>(loc);
        rewriter.setInsertionPointToStart(&regionOp.bodyRegion().front());
        Value origIV = loop.getInductionVar(0);
        auto src = rewriter.create<KrnlSeqExtractOp>(
            loc, seqElementType, output, origIV);
        emitCopy(rewriter, loc, src, alloc, {origIV});
        rewriter.restoreInsertionPoint(afterCopyLoop);
        newOutputs.emplace_back(alloc);
      } else {
        newOutputs.emplace_back(output);
      }
    }
    // end accumulate dynamic tensor

    rewriter.replaceOp(op, newOutputs);
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
        MultiDialectBuilder<KrnlBuilder, MathBuilder, MemRefBuilder> create(
            rewriter, loc);
        auto rankedScanOutTy = memRefType;
        SmallVector<mlir::Value, 4> allocParams;

        // Check the loop accumulation dimension
        if (rankedScanOutTy.getShape()[0] == -1) {
          // TODO(tjingrant): in general, it is not correct to expect
          // loop operation scan output to have the leading dimension extent
          // equal to the max trip count, due to the possibility of early
          // termination.
          assert(!loopOpAdapter.M().getType().isa<NoneType>());
          Value maxTripCount =
              rewriter.create<KrnlLoadOp>(loc, loopOpAdapter.M()).getResult();
          allocParams.emplace_back(rewriter.create<arith::IndexCastOp>(
              loc, rewriter.getIndexType(), maxTripCount));
        }

        bool isDynamic = false;
        // If one of the rest dimension is dynamic, we cannot allocate the
        // memref before the loop because the size of the dynamic dim is not
        // known yet. The accumulation has to be done like sequence. The
        // sequence will be converted to a tensor after loop when the size is
        // known All the related code will be marked with 'accumulation for
        // dynamic tensor'
        for (int i = 1; i < rankedScanOutTy.getRank(); i++) {
          if (rankedScanOutTy.getShape()[i] == -1) {
            isDynamic = true;
            break;
          }
        }
        MemRefBuilder createMemRef(rewriter, loc);
        if (isDynamic) {
          // Suppose the scanout type is is <d1 , d2,... dnxT>
          // Use memref<d1xmemref<d2, ..., dnxT>>
          // seqElementType: memref<d2, ..., dnxT>
          auto elementType = rankedScanOutTy.getElementType();
          ArrayRef<int64_t> shape1 =
              llvm::makeArrayRef(rankedScanOutTy.getShape().begin() + 1,
                  rankedScanOutTy.getShape().end());
          auto seqElementType = MemRefType::get(shape1, elementType);
          auto seqType =
              MemRefType::get({rankedScanOutTy.getShape()[0]}, seqElementType);
          alloc = createMemRef.alignedAlloc(seqType, allocParams);
        } else {
          alloc = createMemRef.alignedAlloc(rankedScanOutTy, allocParams);
        }
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
      krnl::BuildKrnlLoop loop(rewriter, loc, srcTy.getRank());
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
    KrnlBuilder createKrnl(rewriter, loc);
    Value val = createKrnl.load(src, readIV);
    createKrnl.store(val, dest, writeIV);
  }
};

void populateLoweringONNXLoopOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXLoopOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
