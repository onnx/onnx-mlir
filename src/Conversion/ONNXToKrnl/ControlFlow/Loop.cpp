/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------- Loop.cpp - Lowering Loop Op ---------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Loop Operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXLoopOpLowering : public OpConversionPattern<ONNXLoopOp> {
  explicit ONNXLoopOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXLoopOp loopOp, ONNXLoopOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = loopOp.getOperation();
    Location loc = ONNXLoc<ONNXLoopOp>(op);
    ValueRange operands = adaptor.getOperands();

    if (isWhileLoop(op)) {
      return rewriteWithSCFWhile(op, operands, rewriter);
    }

    auto &loopBody = loopOp.getBody();

    // Allocate memory for two kinds of outputs:
    // - final values of loop carried dependencies, and
    // - scan output (all intermediate values returned from body func
    // concatenated together).
    // Implementation details:
    // *  For final values with SeqType. SeqType is implemented with
    //    memref<*xmemref<*xT>>. The size of the memref is dynamic for
    //    every iteration. The SeqType value is stored in a Value of
    //    type memref<1xmemref<*xmemref<*xT>>. In loop body, argument of
    //    SeqType is loaded from this value in the beginning, and the new
    //    value is stored into this value at the end.
    // *  Scan output with unknown shape. For example, the intermediate
    //    result is of shape memref<?xT>, where the ? is only known inside
    //    loop body. Therefore, the scan result of memref<?x?xT> cannot be
    //    allocated outside of the loop. Instead, memref<?xmemref<?xT>> is
    //    used to accumulate the scan output. This seqType result is
    //    transformed into memref<?x?xT> after the loop when the shape is
    //    known.

    SmallVector<Value, 4> outputs;
    allocateMemoryForVFinal(loc, rewriter, op, adaptor, outputs);
    allocateMemoryForScanOutput(loc, rewriter, op, adaptor, outputs);
    MultiDialectBuilder<KrnlBuilder, MemRefBuilder, MathBuilder> create(
        rewriter, loc);

    // Copy content of vInit to vFinal, which is used to host intermediate
    // values produced by loop body function invocation in a scope accessible by
    // all loop iterations.
    for (unsigned long i = 0; i < adaptor.getVInitial().size(); i++) {
      Value origInput = loopOp.getVInitial()[i];
      if (mlir::isa<SeqType>(origInput.getType())) {
        Value zero = create.math.constantIndex(0);
        create.krnl.store(adaptor.getVInitial()[i], outputs[i], zero);
      } else {
        emitCopy(rewriter, loc, adaptor.getVInitial()[i], outputs[i]);
      }
    }

    // Convert the cond type to MemRefType.
    Type convertedType =
        typeConverter->convertType(adaptor.getCond().getType());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType condMemRefTy = mlir::cast<MemRefType>(convertedType);

    // Create a memref for recording loop condition, initialize it with the
    // initial loop condition.
    Value cond;
    if (hasAllConstantDimensions(condMemRefTy))
      cond = create.mem.alignedAlloc(condMemRefTy);
    emitCopy(rewriter, loc, adaptor.getCond(), cond);

    // Create the loop iteration.
    IndexExprScope childScope(&rewriter, loc);
    KrnlBuilder createKrnl(rewriter, loc);

    Value maxTripCount = createKrnl.load(adaptor.getM());
    maxTripCount = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), maxTripCount);
    ValueRange loopDef = createKrnl.defineLoops(1);
    Value zero = create.math.constantIndex(0);
    createKrnl.iterate(loopDef, loopDef, {zero}, {maxTripCount},
        [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
          OpBuilder::InsertionGuard insertGuard(rewriter);

          Value condReg = createKrnl.load(cond);
          auto ifOp = rewriter.create<scf::IfOp>(loc, condReg, false);
          rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());

          // Contain the ThenRegion with KrnlRegionOp
          // The krnl loop inside may use symbol computed in LoopOp body
          KrnlRegionOp regionOp = rewriter.create<KrnlRegionOp>(loc);
          rewriter.setInsertionPointToStart(&regionOp.getBodyRegion().front());

          // Create a scalar tensor out of loop iteration variable, as the first
          // argument passed to the body graph function.
          Value origIV = loopInd[0];
          Value iv = rewriter
                         .create<arith::IndexCastOp>(
                             loc, rewriter.getI64Type(), origIV)
                         .getResult();
          MemRefBuilder createMemRef(rewriter, loc);
          Value ivMemRef =
              createMemRef.alloc(MemRefType::get({}, rewriter.getI64Type()));
          createKrnl.store(iv, ivMemRef);

          // Make the call to loop body function.
          SmallVector<Value, 4> params = {ivMemRef, adaptor.getCond()};

          // For SeqType, load the value for the storage
          for (unsigned long i = 0; i < loopOp.getVInitial().size(); i++) {
            if (mlir::isa<SeqType>(loopOp.getVInitial()[i].getType())) {
              Value seqValue = create.krnl.load(outputs[i], zero);
              params.emplace_back(seqValue);
            } else {
              params.emplace_back(outputs[i]);
            }
          }

          Block &loopBodyEntryBlock = loopBody.front();
          IRMapping mapper;
          for (unsigned i = 0, e = params.size(); i != e; ++i) {
            // Verify that the types of the provided values match the function
            // argument types.
            BlockArgument regionArg = loopBodyEntryBlock.getArgument(i);
            mapper.map(regionArg, params[i]);
          }

          // Previous code is intended to create the loop body without RegionOp
          // Current code just splice the loopBody into the RegionOp after it
          // was built. ToDo: code could be simplified if not built on top of
          // the previous code

          Region &thenRegion = ifOp.getThenRegion();
          Block &thenBlock = thenRegion.front();

          // Split the insertion block into two, where the second block
          // `postInsertBlock` contains only the terminator operation, insert
          // loop body right before `postInsertBlock`, after all other
          // operations created within the if Op.
          Block *postInsertBlock =
              thenBlock.splitBlock(thenBlock.getTerminator());
          assert(loopBody.getBlocks().size() == 1 &&
                 "Currently only support loop body with 1 block.");
          thenRegion.getBlocks().splice(
              postInsertBlock->getIterator(), loopBody.getBlocks());

          auto newBlocks = llvm::make_range(std::next(thenBlock.getIterator()),
              postInsertBlock->getIterator());
          Block &loopBodyBlock = *newBlocks.begin();

          Operation *loopBodyTerminator = loopBodyBlock.getTerminator();

          // Within inlined blocks, substitute reference to block arguments with
          // values produced by the lowered loop operation bootstrapping IR.
          auto remapOperands = [&](Operation *op1) {
            for (auto &operand : op1->getOpOperands())
              if (auto mappedOp = mapper.lookupOrNull(operand.get()))
                operand.set(mappedOp);
          };
          for (auto &block : thenRegion.getBlocks())
            block.walk(remapOperands);
          auto resultsRange = llvm::SmallVector<Value, 4>(
              loopBodyTerminator->getOperands().begin(),
              loopBodyTerminator->getOperands().end());

          // Add the loop end code into the loopBodyBlock
          // Previous code adds them to postInsertBlock
          // rewriter.setInsertionPointToStart(postInsertBlock);
          rewriter.setInsertionPointToEnd(&loopBodyBlock);

          // Cast loop body outputs from tensor type to memref type in case it
          // has not already been lowered. Eventually,
          // 'UnrealizedConversionCastOp' becomes a cast from memref type to a
          // memref type when everything is lowered and thus becomes redundant.
          SmallVector<Value, 4> bodyOutputs(
              resultsRange.begin(), resultsRange.end());
          for (unsigned i = 0; i < bodyOutputs.size(); i++) {
            Value output = bodyOutputs[i];
            Type outputTy = output.getType();
            bodyOutputs[i] =
                rewriter
                    .create<UnrealizedConversionCastOp>(
                        loc, typeConverter->convertType(outputTy), output)
                    .getResult(0);
          }

          // Copy the newly computed loop condition to pre-allocated buffer.
          emitCopy(rewriter, loc, bodyOutputs[0], cond);

          // Copy intermediate values of scan outputs to their corresponding
          // slice in the loop scan output tensor.
          // Intermediate value with SeqType should not in scan output
          auto vIntermediate = llvm::make_range(bodyOutputs.begin() + 1,
              bodyOutputs.begin() + 1 + adaptor.getVInitial().size());
          auto scanIntermediate =
              llvm::make_range(vIntermediate.end(), bodyOutputs.end());
          auto scanOutputs = llvm::make_range(
              outputs.begin() + adaptor.getVInitial().size(), outputs.end());
          for (auto scanIntermediateToFinal :
              llvm::zip(scanIntermediate, scanOutputs)) {
            Type elementType = mlir::cast<MemRefType>(
                std::get<1>(scanIntermediateToFinal).getType())
                                   .getElementType();
            if (mlir::dyn_cast<MemRefType>(elementType)) {
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

          // Copy intermediate values of loop carried dependencies to MemRef
          // outside the iteration scope so next iteration can use them as init
          // value.
          for (unsigned long i = 0; i < loopOp.getVInitial().size(); i++) {
            if (mlir::isa<SeqType>(loopOp.getVInitial()[i].getType())) {
              create.krnl.store(bodyOutputs[i + 1], outputs[i], zero);
            } else {
              emitCopy(rewriter, loc, bodyOutputs[i + 1], outputs[i]);
            }
          }

          // Remove loop body terminator op.
          rewriter.eraseOp(loopBodyTerminator);

          // Merge the post block (with only Terminator) into the thenBody
          thenBlock.getOperations().splice(
              thenBlock.end(), postInsertBlock->getOperations());
          rewriter.eraseBlock(postInsertBlock);

          // Merge the loop body block into the RegionOp.
          regionOp.getBodyRegion().front().getOperations().splice(
              regionOp.getBodyRegion().front().end(),
              loopBodyBlock.getOperations());
          rewriter.eraseBlock(&loopBodyBlock);
        });

    // accumulate dynamic tensor
    // Convert the memref<?xmemref<>> to a memref
    SmallVector<Value, 4> newOutputs;
    for (unsigned long i = 0; i < outputs.size(); i++) {
      Value output = outputs[i];
      auto seqElementType =
          mlir::cast<MemRefType>(output.getType()).getElementType();
      if (mlir::isa<MemRefType>(seqElementType)) {
        // need to distinguish seqType in v_final and scan
        if (i < loopOp.v_final().size()) {
          // In v_final
          Value v = create.krnl.load(output, zero);
          newOutputs.emplace_back(v);
        } else {
          // scan output
          // need to convert memref<memrefs<xT>> to memref<xT>
          // TODO: need a IF statement to handle output is empty
          // we can safely give 0 to the dynamic dim for alloc
          // Here loop is assumed to be executed at least once.
          Value firstElement =
              create.krnl.load(output, create.math.constantIndex(0));
          SmallVector<Value, 4> allocParams;
          SmallVector<int64_t, 4> dims;
          dims.emplace_back(
              mlir::cast<MemRefType>(output.getType()).getShape()[0]);
          if (mlir::cast<MemRefType>(output.getType()).isDynamicDim(0))
            allocParams.emplace_back(create.mem.dim(output, 0));
          for (auto i = 0;
               i < mlir::cast<MemRefType>(firstElement.getType()).getRank();
               i++) {
            dims.emplace_back(
                mlir::cast<MemRefType>(firstElement.getType()).getShape()[i]);
            if (mlir::cast<MemRefType>(firstElement.getType()).isDynamicDim(i))
              allocParams.emplace_back(create.mem.dim(firstElement, i));
          }
          ArrayRef<int64_t> shape(dims.data(), dims.size());
          auto flatType = MemRefType::get(shape,
              mlir::cast<MemRefType>(firstElement.getType()).getElementType());
          Value alloc = create.mem.alignedAlloc(flatType, allocParams);
          // copy the value
          KrnlBuilder createKrnl(rewriter, loc);
          ValueRange loopDef = createKrnl.defineLoops(1);
          createKrnl.iterate(loopDef, loopDef, {zero}, {maxTripCount},
              [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
                // Wrap with KrnlRegionOp because emitCopy uses the result of
                // SeqExtract for loop bound.
                KrnlRegionOp regionOp = rewriter.create<KrnlRegionOp>(loc);
                rewriter.setInsertionPointToStart(
                    &regionOp.getBodyRegion().front());
                Value origIV = loopInd[0];
                auto src = rewriter.create<KrnlSeqExtractOp>(loc,
                    seqElementType, output, origIV,
                    IntegerAttr::get(rewriter.getIntegerType(1, false), 0));
                emitCopy(rewriter, loc, src, alloc, {origIV});
              });
          newOutputs.emplace_back(alloc);
        }
      } else {
        newOutputs.emplace_back(output);
      }
    }
    // end accumulate dynamic tensor

    rewriter.replaceOp(op, newOutputs);
    onnxToKrnlSimdReport(op);
    return success();
  }

  void allocateMemoryForVFinal(Location loc,
      ConversionPatternRewriter &rewriter, Operation *op,
      ONNXLoopOpAdaptor adaptor, SmallVectorImpl<Value> &outputs) const {
    auto loopOp = mlir::dyn_cast<ONNXLoopOp>(op);
    for (const auto &ioPair :
        llvm::zip(adaptor.getVInitial(), loopOp.v_final())) {
      auto vInit = std::get<0>(ioPair);
      auto vFinal = std::get<1>(ioPair);

      // Convert vFinal's type to MemRefType.
      Type convertedType = typeConverter->convertType(vFinal.getType());
      assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
             "Failed to convert type to MemRefType");
      MemRefType memRefType = mlir::cast<MemRefType>(convertedType);

      if (mlir::isa<SeqType>(vFinal.getType())) {
        memRefType = MemRefType::get({1}, memRefType);
      }

      // Allocate memory for the loop-carried dependencies, since they are
      // guaranteed to have the same shape throughout all iterations, use their
      // initial value tensors as reference when allocating memory.
      MultiDialectBuilder<MemRefBuilder> create(rewriter, loc);
      Value alloc = create.mem.alignedAlloc(vInit, memRefType);
      outputs.emplace_back(alloc);
    }
  }

  void allocateMemoryForScanOutput(Location loc,
      ConversionPatternRewriter &rewriter, Operation *op,
      ONNXLoopOpAdaptor adaptor, SmallVectorImpl<Value> &outputs,
      bool isWhile = false) const {
    auto loopOp = mlir::dyn_cast<ONNXLoopOp>(op);
    for (const auto &opScanOutput : loopOp.scan_outputs()) {
      // Convert opScanOutput's type to MemRefType.
      Type convertedType = typeConverter->convertType(opScanOutput.getType());
      assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
             "Failed to convert type to MemRefType");
      MemRefType memRefType = mlir::cast<MemRefType>(convertedType);

      // Allocate memory for the scan outputs. There're no good "reference"
      // shape for scan outputs. So if the scan outputs do not have constant
      // dimensions in all except the leading dimensions, we simply give up. The
      // leading dimension is simply the number of iterations executed, which is
      // easier to obtain.
      Value alloc;
      MultiDialectBuilder<KrnlBuilder, MathBuilder, MemRefBuilder> create(
          rewriter, loc);
      if (hasAllConstantDimensions(memRefType))
        alloc = create.mem.alignedAlloc(memRefType);
      else {
        auto rankedScanOutTy = memRefType;
        SmallVector<Value, 4> allocParams;

        // Check the loop accumulation dimension
        if (rankedScanOutTy.isDynamicDim(0)) {
          // TODO(tjingrant): in general, it is not correct to expect
          // loop operation scan output to have the leading dimension extent
          // equal to the max trip count, due to the possibility of early
          // termination.
          // TODO(chentong): will use dynamic data structure(e.g. Sequence)
          // to support the scan output for while.
          if (isWhile) {
            llvm_unreachable("Scan output for while loop is not supported");
          }
          assert(!mlir::isa<NoneType>(adaptor.getM().getType()));
          Value maxTripCount =
              rewriter.create<KrnlLoadOp>(loc, adaptor.getM()).getResult();
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
          if (rankedScanOutTy.isDynamicDim(i)) {
            isDynamic = true;
            break;
          }
        }
        MemRefBuilder createMemRef(rewriter, loc);
        if (isDynamic) {
          // Suppose the scan out type is is <d1 , d2,... dnxT>
          // Use memref<d1xmemref<d2, ..., dnxT>>
          // seqElementType: memref<d2, ..., dnxT>
          Type elementType = rankedScanOutTy.getElementType();
          ArrayRef<int64_t> shape1 =
              llvm::ArrayRef(rankedScanOutTy.getShape().begin() + 1,
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
    auto srcTy = mlir::cast<MemRefType>(src.getType());
    SmallVector<Value, 4> readIV;
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl> create(
        rewriter, loc);
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

  // Check whether scf.While has to be used to Loop instead of krnl loop
  // krnl loop can be used only when the condition for LoopOp iteration
  // is a loop invariant. In LoopOp structure, the condition at the end
  // of loop body (the first operand in yieldOp) is the condition passed to
  // loop body at the beginning (the second argument of loop body)

  // If there is a seq in the loop carried variable list, scf.while is
  // needed. This is a temporary implementation issue: the krnl.iterate
  // does not support return of loop-carried variable other than the
  // iteration variable

  bool isWhileLoop(Operation *op) const {
    auto onnxLoopOp = mlir::dyn_cast<ONNXLoopOp>(op);

    // Check whether continue condition is modified or not
    // Code copied from src/Dialect/ONNX/Rewrite.cpp

    // Check whether the condition is optional
    if (isNoneValue(onnxLoopOp.getCond()))
      return false;

    // Get the loop region.
    Region &loopBody = onnxLoopOp.getBody();
    // Make sure the region has only one block.
    if (!loopBody.hasOneBlock())
      return true;

    // Get YieldOp of the body block.
    Block &bodyBlock = loopBody.front();
    Operation *yieldOp = bodyBlock.getTerminator();
    if (!isa<ONNXYieldOp>(yieldOp))
      return true;

    // The break condition is the first argument of YieldOp.
    // `ONNXYieldOp (cond, ..., ubValue, ..., newCounterValue, ...)`
    // which means the condition is loop invariant.
    Value breakCond = yieldOp->getOperands()[0];
    if (mlir::isa<BlockArgument>(breakCond) &&
        mlir::cast<BlockArgument>(breakCond).getArgNumber() == 1) {
    } else
      return true;

    return false;
  }

  LogicalResult rewriteWithSCFWhile(Operation *op, ValueRange operands,
      ConversionPatternRewriter &rewriter) const {
    Location loc = ONNXLoc<ONNXLoopOp>(op);
    auto loopOp = mlir::dyn_cast<ONNXLoopOp>(op);
    MultiDialectBuilder<KrnlBuilder, MemRefBuilder, MathBuilder> create(
        rewriter, loc);

    // Create memref object for induction variable and outputs
    // As a result, there is no memref object returned from LoopOp,
    // just the value in these "global" memref objects.
    Value ivMemRef =
        create.mem.alloc(MemRefType::get({}, rewriter.getI64Type()));
    Value cond = create.mem.alloc(MemRefType::get({}, rewriter.getI1Type()));
    ONNXLoopOpAdaptor adaptor(operands, op->getAttrDictionary());

    // Construct inputs for WhileOp, which should be (0, cond, v_initial)
    // The initial value for iteration should be zero
    SmallVector<Value, 4> whileInputValues;
    SmallVector<Type, 4> whileInputTypes;
    SmallVector<Location, 4> locs;
    bool hasM = false;
    bool hasCond = false;
    Value c0 = create.math.constant(rewriter.getI64Type(), 0);
    Value c1 = create.math.constant(rewriter.getI64Type(), 1);
    Value ubV;
    if (!isNoneValue(loopOp.getM())) {
      hasM = true;
      Value mInitial = c0;
      whileInputValues.emplace_back(mInitial);
      whileInputTypes.emplace_back(mInitial.getType());
      ubV = create.krnl.load(adaptor.getM());
      locs.emplace_back(loc);
    }

    if (!isNoneValue(loopOp.getCond())) {
      hasCond = true;
      whileInputValues.emplace_back(adaptor.getCond());
      whileInputTypes.emplace_back(adaptor.getCond().getType());
      locs.emplace_back(loc);
    }

    // add v_initial
    for (auto v : adaptor.getVInitial()) {
      whileInputValues.emplace_back(v);
      whileInputTypes.emplace_back(v.getType());
      locs.emplace_back(loc);
    }

    SmallVector<Value, 4> outputs;
    allocateMemoryForVFinal(loc, rewriter, op, adaptor, outputs);

    // Need to handle the scan out specially because the trip count cannot
    // be expressed for a while loop
    allocateMemoryForScanOutput(loc, rewriter, op, adaptor, outputs, true);
    SmallVector<Type, 4> outputTypes;
    for (auto v : outputs) {
      outputTypes.emplace_back(v.getType());
    }

    // Create the skeleton of WhileOp
    // auto whileOp = rewriter.create<scf::WhileOp>(loc, outputTypes,
    // whileInputValues);
    auto whileOp =
        rewriter.create<scf::WhileOp>(loc, whileInputTypes, whileInputValues);
    Block *beforeBlock =
        rewriter.createBlock(&whileOp.getBefore(), {}, whileInputTypes, locs);
    Block *afterBlock =
        rewriter.createBlock(&whileOp.getAfter(), {}, whileInputTypes, locs);

    // Construct the condition block
    {
      auto arguments = beforeBlock->getArguments();
      rewriter.setInsertionPointToStart(&whileOp.getBefore().front());
      Value mCond;
      int argIndex = 0;
      if (hasM) {
        auto iv = beforeBlock->getArgument(argIndex);
        mCond = create.math.slt(iv, ubV);
        argIndex++;
      }

      Value condV;
      if (hasCond) {
        condV = create.krnl.load(beforeBlock->getArgument(argIndex));
      }

      Value combinedV;
      if (hasM && hasCond) {
        combinedV = create.math.andi(mCond, condV);
      } else if (hasM) {
        combinedV = mCond;
      } else {
        combinedV = condV;
      }

      rewriter.create<scf::ConditionOp>(loc, combinedV, arguments);
    }

    // Construct the body of while loop
    {
      rewriter.setInsertionPointToStart(&whileOp.getAfter().front());

      // Handle loop body
      // Most code is copied from the lambda function of krnl.iterate
      // for LoopOp

      // Differences: no need to construct scf.if since the condition
      // is checked with iteration variable in the first block of
      // WhileOp

      // Contain the ThenRegion with KrnlRegionOp
      // The krnl loop inside may use symbol computed in LoopOp body
      KrnlRegionOp regionOp = rewriter.create<KrnlRegionOp>(loc);
      rewriter.setInsertionPointToStart(&regionOp.getBodyRegion().front());

      SmallVector<Value, 4> params;
      int argIndex = 0;
      // Create a scalar tensor out of loop iteration variable, as the first
      // argument passed to the body graph function.
      if (hasM) {
        Value iv = afterBlock->getArgument(argIndex);
        create.krnl.store(iv, ivMemRef);
        params.emplace_back(ivMemRef);
        argIndex++;
      }

      for (auto v :
          llvm::make_range(afterBlock->getArguments().begin() + argIndex,
              afterBlock->getArguments().end())) {
        params.emplace_back(v);
      }

      Region &loopBody = loopOp.getBody();
      Block &loopBodyEntryBlock = loopBody.front();
      IRMapping mapper;
      for (unsigned i = 0, e = params.size(); i != e; ++i) {
        // Verify that the types of the provided values match the function
        // argument types.
        BlockArgument regionArg = loopBodyEntryBlock.getArgument(i);
        mapper.map(regionArg, params[i]);
      }

      Region &containRegion = regionOp.getBodyRegion();
      Block &firstBlock = containRegion.front();
      assert(loopBody.getBlocks().size() == 1 &&
             "Currently only support loop body with 1 block.");
      containRegion.getBlocks().splice(
          containRegion.getBlocks().end(), loopBody.getBlocks());
      Block &loopBodyBlock = *std::next(firstBlock.getIterator());
      Operation *loopBodyTerminator = loopBodyBlock.getTerminator();

      // Within inlined blocks, substitute reference to block arguments with
      // values produced by the lowered loop operation bootstrapping IR.
      auto remapOperands = [&](Operation *op1) {
        for (auto &operand : op1->getOpOperands())
          if (auto mappedOp = mapper.lookupOrNull(operand.get()))
            operand.set(mappedOp);
      };
      for (auto &block : containRegion.getBlocks())
        block.walk(remapOperands);
      auto resultsRange =
          llvm::SmallVector<Value, 4>(loopBodyTerminator->getOperands().begin(),
              loopBodyTerminator->getOperands().end());

      // Add Ops at the end of the loopBody
      rewriter.setInsertionPointToEnd(&loopBodyBlock);

      // Cast loop body outputs from tensor type to memref type in case it
      // has not already been lowered. Eventually,
      // 'UnrealizedConversionCastOp' becomes a cast from memref type to a
      // memref type when everything is lowered and thus becomes redundant.
      SmallVector<Value, 4> bodyOutputs(
          resultsRange.begin(), resultsRange.end());
      for (unsigned long i = 0; i < bodyOutputs.size(); i++) {
        Value output = bodyOutputs[i];
        assert((mlir::isa<TensorType>(output.getType()) ||
                   mlir::isa<MemRefType>(output.getType())) &&
               "Expecting loop body function output to consist of "
               "tensors/memrefs.");
        auto outputTy = mlir::cast<ShapedType>(output.getType());
        bodyOutputs[i] = rewriter
                             .create<UnrealizedConversionCastOp>(loc,
                                 MemRefType::get(outputTy.getShape(),
                                     outputTy.getElementType()),
                                 output)
                             .getResult(0);
      }

      // In while loop, the scan output has to be supported with dynamic
      // sequence because a static-sized sequence can not be allocated
      // due to the dynamic iteration count of loop. Another PR.

      // This copy is needed for KrnlRegion, otherwise the symbol created
      // in loop body cannot be exported out of KrnlRegion
      // ToDo(chentong): extend KrnlRegion to generate output

      // Copy the newly computed loop condition to pre-allocated buffer.
      // It can be assumed that hasCond
      int condIndex = 0;
      if (hasCond) {
        emitCopy(rewriter, loc, bodyOutputs[0], cond);
        condIndex++;
      }

      // Copy intermediate values of scan outputs to their corresponding
      // slice in the loop scan output tensor.
      auto vIntermediate = llvm::make_range(bodyOutputs.begin() + condIndex,
          bodyOutputs.begin() + condIndex + adaptor.getVInitial().size());
      auto scanIntermediate =
          llvm::make_range(vIntermediate.end(), bodyOutputs.end());
      auto scanOutputs = llvm::make_range(
          outputs.begin() + adaptor.getVInitial().size(), outputs.end());

      for (auto scanIntermediateToFinal :
          llvm::zip(scanIntermediate, scanOutputs)) {
        Type elementType = mlir::cast<MemRefType>(
            std::get<1>(scanIntermediateToFinal).getType())
                               .getElementType();
        if (mlir::dyn_cast<MemRefType>(elementType)) {
          // TODO(chentong): handle dynamic scan output for while loop
          llvm_unreachable("Not implemented yet");
        } else {
          emitCopy(rewriter, loc, std::get<0>(scanIntermediateToFinal),
              std::get<1>(scanIntermediateToFinal),
              /*writePrefix=*/{ivMemRef});
        }
      }

      // Copy intermediate values of loop carried dependencies to MemRef
      // outside the iteration scope so next iteration can use them as init
      // value.
      for (auto vIntermediateToFinal : llvm::zip(vIntermediate, outputs))
        emitCopy(rewriter, loc, std::get<0>(vIntermediateToFinal),
            std::get<1>(vIntermediateToFinal));

      // Erase the yieldOp of loopBody
      rewriter.eraseOp(loopBodyTerminator);
      // Move the Ops in loopBody into the afterBlock of while
      firstBlock.getOperations().splice(
          firstBlock.end(), loopBodyBlock.getOperations());
      rewriter.eraseBlock(&loopBodyBlock);

      // Add YieldOp for WhileOp
      rewriter.setInsertionPointToEnd(&whileOp.getAfter().front());
      SmallVector<Value> yieldList;
      // Add the IV for WhileOp, which is not explicitly in the output
      // of loop body of LoopOp
      if (hasM) {
        Value newIV = create.math.add(afterBlock->getArgument(0), c1);
        yieldList.emplace_back(newIV);
      }
      if (hasCond) {
        yieldList.emplace_back(cond);
      }
      for (auto v : outputs) {
        yieldList.emplace_back(v);
      }
      rewriter.create<scf::YieldOp>(loc, yieldList);
    }

    rewriter.replaceOp(op, outputs);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXLoopOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXLoopOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
