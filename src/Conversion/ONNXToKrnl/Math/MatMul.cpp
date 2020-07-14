//===----------------- Matmul.cpp - Lowering Matmul Op --------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Matmul Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

struct ONNXMatMulOpLowering : public ConversionPattern {
  ONNXMatMulOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXMatMulOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    ONNXMatMulOpAdaptor operandAdaptor(operands);
    Value A = operandAdaptor.A();
    Value B = operandAdaptor.B();
    auto AShape = A.getType().cast<MemRefType>().getShape();
    auto BShape = B.getType().cast<MemRefType>().getShape();

    // There are three cases related to the shapes of the two arguments:
    // - Both arguments are N-D, N >= 2
    // - Either argument is 1-D, the other is N-D, N >= 2
    // - Both arguments are 1-D

    // Result type
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    auto elementType = memRefType.getElementType();
    auto memRefShape = memRefType.getShape();

    // A value zero
    auto zero = emitConstantOp(rewriter, loc, memRefType.getElementType(), 0);

    // Create init block if this is the first operation in the function.
    createInitState(rewriter, loc, op);

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc;
    bool insertDealloc = checkInsertDealloc(op);
    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc, op);
    else {
      PatternRewriter::InsertionGuard insertGuard(rewriter);
      FuncOp function = cast<FuncOp>(op->getParentOp());
      bool allOperandsAreInInitBlock = operandsInInitOrArgList(function, {A, B});
      if (allOperandsAreInInitBlock)
        rewriter.setInsertionPoint(getInitInsertionPoint(function));

      SmallVector<Value, 4> allocOperands;
      if (AShape.size() >= 2 && BShape.size() >= 2) {
        // Both arguments are N-D, N >= 2
        // (s1 x s2 x... x sK x M x K) MATMUL (K x N)
        // =>
        // (s1 x s2 x... x sK x M x N)
        for (int i = 0; i < memRefShape.size() - 2; ++i) {
          if (memRefShape[i] < 0) {
            if ((AShape.size() == 2) && (BShape.size() > 2))
              allocOperands.emplace_back(rewriter.create<DimOp>(loc, B, i));
            else if ((AShape.size() > 2) && (BShape.size() == 2))
              allocOperands.emplace_back(rewriter.create<DimOp>(loc, A, i));
          }
        }
        if (memRefShape[memRefShape.size() - 2] < 0) {
          auto dim = rewriter.create<DimOp>(loc, A, memRefShape.size() - 2);
          allocOperands.emplace_back(dim);
        }
        if (memRefShape[memRefShape.size() - 1] < 0) {
          auto dim = rewriter.create<DimOp>(loc, B, memRefShape.size() - 1);
          allocOperands.emplace_back(dim);
        }
      } else if (AShape.size() == 1 && BShape.size() >= 2) {
        // Either argument is 1-D
        // K MATMUL (s1 x s2 x... x sK x K x N)
        // =>
        // (s1 x s2 x... x sK x N)
        for (int i = 0; i < memRefShape.size() - 1; ++i) {
          if (memRefShape[i] < 0) {
            auto dim = rewriter.create<DimOp>(loc, B, i);
            allocOperands.emplace_back(dim);
          }
        }
        if (memRefShape[memRefShape.size() - 1] < 0) {
          auto dim = rewriter.create<DimOp>(loc, B, BShape.size() - 1);
          allocOperands.emplace_back(dim);
        }
      } else if (AShape.size() >= 2 && BShape.size() == 1) {
        // Either argument is 1-D
        // (s1 x s2 x... x sK x M x K) MATMUL K
        // =>
        // (s1 x s2 x... x sK x M)
        for (int i = 0; i < memRefShape.size() - 1; ++i) {
          if (memRefShape[i] < 0) {
            auto dim = rewriter.create<DimOp>(loc, A, i);
            allocOperands.emplace_back(dim);
          }
        }
        if (memRefShape[memRefShape.size() - 1] < 0) {
          auto dim = rewriter.create<DimOp>(loc, A, AShape.size() - 2);
          allocOperands.emplace_back(dim);
        }
      } else if (AShape.size() == 1 && BShape.size() == 1) {
        // Both arguments are 1-D
        if (memRefShape[0] < 0) {
          auto dim = rewriter.create<DimOp>(loc, A, 0);
          allocOperands.emplace_back(dim);
        }
      } else {
        return emitError(loc, "Invalid shapes");
      }

      alloc = rewriter.create<AllocOp>(loc, memRefType, allocOperands);

      if (allOperandsAreInInitBlock)
        markOperandInInitBlock(function, alloc);
    }

    if (AShape.size() >= 2 || BShape.size() >= 2) {
      // Cases 1 and 2:
      // - Both arguments are N-D, N >= 2
      // - Either argument is 1-D, the other is N-D, N >= 2

      // Define loops for batch dimensions.
      std::vector<Value> originalLoops;
      defineLoops(rewriter, loc, originalLoops, memRefShape.size());

      // Outer KrnlIterateOp
      SmallVector<Value, 4> loopBatchIVs;
      bool hasBatchLoop = false;
      if (AShape.size() > 2 || BShape.size() > 2) {
        SmallVector<int, 4> batchAxes;
        int matmulResultDims =
            ((AShape.size() == 1 || BShape.size() == 1)) ? 1 : 2;
        for (int i = 0; i < memRefShape.size() - matmulResultDims; ++i)
          batchAxes.emplace_back(i);

        std::vector<Value> outerLoops;
        outerLoops.reserve(batchAxes.size());
        for (int i = 0; i < batchAxes.size(); ++i)
          outerLoops.push_back(originalLoops[i]);

        KrnlIterateOperandPack outerPack(rewriter, outerLoops);
        for (int i = 0; i < batchAxes.size(); ++i) {
          addDimensionToPack(rewriter, loc, outerPack, alloc, i);
        }
        auto outerIterateOp = rewriter.create<KrnlIterateOp>(loc, outerPack);

        // Insert instructions into the outer KrnlIterateOp.
        Block &outerIterationBlock = outerIterateOp.bodyRegion().front();
        rewriter.setInsertionPointToStart(&outerIterationBlock);

        // Induction variables: non-matrix-multiplication variables.
        for (auto arg : outerIterationBlock.getArguments()) {
          loopBatchIVs.emplace_back(arg);
        }

        hasBatchLoop = true;
      }

      // Now, we define loops for matrix multiplication.

      // Create a KrnlIterateOp for matrix multiplication.
      KrnlIterateOp matmulIterateOp;
      std::vector<Value> matmulLoops;
      if (AShape.size() >= 2 && BShape.size() >= 2) {
        // 2-D x 2-D. Result has two dimensions.
        matmulLoops.reserve(2);
        for (int i = 2; i > 0; --i) {
          matmulLoops.emplace_back(originalLoops[memRefShape.size() - i]);
        }
        KrnlIterateOperandPack matmulPack(rewriter, matmulLoops);
        for (int i = 2; i > 0; --i) {
          addDimensionToPack(
              rewriter, loc, matmulPack, alloc, memRefShape.size() - i);
        }
        matmulIterateOp = rewriter.create<KrnlIterateOp>(loc, matmulPack);
      } else {
        // 1-D x 2-D, and vice versa. Result has one dimension.
        matmulLoops.reserve(1);
        matmulLoops.emplace_back(originalLoops[memRefShape.size() - 1]);
        KrnlIterateOperandPack matmulPack(rewriter, matmulLoops);
        addDimensionToPack(
            rewriter, loc, matmulPack, alloc, memRefShape.size() - 1);
        matmulIterateOp = rewriter.create<KrnlIterateOp>(loc, matmulPack);
      }

      // Insert instructions into the matmul KrnlIterateOp.
      Block &matmulIterationBlock = matmulIterateOp.bodyRegion().front();
      rewriter.setInsertionPointToStart(&matmulIterationBlock);

      // Induction variables: M, N
      SmallVector<Value, 4> loopMNIVs;
      for (auto arg : matmulIterationBlock.getArguments()) {
        loopMNIVs.emplace_back(arg);
      }
      // Induction variables for the final result.
      SmallVector<Value, 4> loopBatchMNIVs;
      for (auto arg : loopBatchIVs) {
        loopBatchMNIVs.emplace_back(arg);
      }
      for (auto arg : loopMNIVs) {
        loopBatchMNIVs.emplace_back(arg);
      }

      // Fill the output with value 0.
      rewriter.create<AffineStoreOp>(loc, zero, alloc, loopBatchMNIVs);

      //  Iterate along the reduction dimension.
      //  Use a value from A.
      std::vector<Value> reduceLoops;
      defineLoops(rewriter, loc, reduceLoops, 1);
      KrnlIterateOperandPack reducePack(rewriter, reduceLoops);
      addDimensionToPack(rewriter, loc, reducePack, A, AShape.size() - 1);
      auto reduceIterateOp = rewriter.create<KrnlIterateOp>(loc, reducePack);

      // Insert instructions into the reduction KrnlIterateOp.
      Block &reduceIterationBlock = reduceIterateOp.bodyRegion().front();
      rewriter.setInsertionPointToStart(&reduceIterationBlock);

      // Induction variables
      SmallVector<Value, 4> loopKIVs, loopBatchMKIVs, loopBatchKNIVs;
      // K
      loopKIVs.emplace_back(reduceIterationBlock.getArguments()[0]);
      // MK
      if (AShape.size() > 2)
        for (auto arg : loopBatchIVs)
          loopBatchMKIVs.emplace_back(arg);
      if (AShape.size() >= 2)
        loopBatchMKIVs.emplace_back(loopMNIVs[0]);
      loopBatchMKIVs.emplace_back(loopKIVs[0]);
      // KN
      if (BShape.size() > 2)
        for (auto arg : loopBatchIVs)
          loopBatchKNIVs.emplace_back(arg);
      loopBatchKNIVs.emplace_back(loopKIVs[0]);
      if (BShape.size() >= 2) {
        if (AShape.size() >= 2)
          loopBatchKNIVs.emplace_back(loopMNIVs[1]);
        else
          loopBatchKNIVs.emplace_back(loopMNIVs[0]);
      }
      // Matmul computation
      auto loadedA = rewriter.create<AffineLoadOp>(loc, A, loopBatchMKIVs);
      auto loadedB = rewriter.create<AffineLoadOp>(loc, B, loopBatchKNIVs);
      auto loadedY = rewriter.create<AffineLoadOp>(loc, alloc, loopBatchMNIVs);
      if (elementType.isa<IntegerType>()) {
        auto AB = rewriter.create<MulIOp>(loc, loadedA, loadedB);
        auto accumulated = rewriter.create<AddIOp>(loc, loadedY, AB);
        rewriter.create<AffineStoreOp>(loc, accumulated, alloc, loopBatchMNIVs);
      } else if (elementType.isa<FloatType>()) {
        auto AB = rewriter.create<MulFOp>(loc, loadedA, loadedB);
        auto accumulated = rewriter.create<AddFOp>(loc, loadedY, AB);
        rewriter.create<AffineStoreOp>(loc, accumulated, alloc, loopBatchMNIVs);
      }
    } else if ((AShape.size() == 1) && (BShape.size() == 1)) {
      // Case 3:
      // - Both arguments are 1-D

      // Fill the output with value 0.
      Value zeroIndex = rewriter.create<ConstantIndexOp>(loc, 0);
      rewriter.create<AffineStoreOp>(loc, zero, alloc, zeroIndex);

      //  Iterate along the reduction dimension.
      //  Use a value from A.
      std::vector<Value> reduceLoops;

      defineLoops(rewriter, loc, reduceLoops, 1);
      KrnlIterateOperandPack reducePack(rewriter, reduceLoops);
      addDimensionToPack(rewriter, loc, reducePack, A, 0);
      auto reduceIterateOp = rewriter.create<KrnlIterateOp>(loc, reducePack);

      // Insert instructions into the reduction KrnlIterateOp.
      Block &reduceIterationBlock = reduceIterateOp.bodyRegion().front();
      rewriter.setInsertionPointToStart(&reduceIterationBlock);

      // Induction variables
      SmallVector<Value, 4> loopKIVs;
      // K
      loopKIVs.emplace_back(reduceIterationBlock.getArgument(0));

      // Matmul computation
      auto loadedA = rewriter.create<AffineLoadOp>(loc, A, loopKIVs);
      auto loadedB = rewriter.create<AffineLoadOp>(loc, B, loopKIVs);
      auto loadedY = rewriter.create<AffineLoadOp>(loc, alloc, zeroIndex);
      if (elementType.isa<IntegerType>()) {
        auto AB = rewriter.create<MulIOp>(loc, loadedA, loadedB);
        auto accumulated = rewriter.create<AddIOp>(loc, loadedY, AB);
        rewriter.create<AffineStoreOp>(loc, accumulated, alloc, zeroIndex);
      } else if (elementType.isa<FloatType>()) {
        auto AB = rewriter.create<MulFOp>(loc, loadedA, loadedB);
        auto accumulated = rewriter.create<AddFOp>(loc, loadedY, AB);
        rewriter.create<AffineStoreOp>(loc, accumulated, alloc, zeroIndex);
      }
    } else {
      // No scalar matrix multiplication.
      llvm_unreachable("Unsupported scalar matrix multiplication.");
    }

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXMatMulOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXMatMulOpLowering>(ctx);
}
