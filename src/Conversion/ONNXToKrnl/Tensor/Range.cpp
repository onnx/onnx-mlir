/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------- Range.cpp - Lowering Range Op --------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Range Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"

using namespace mlir;

struct ONNXRangeOpLowering : public ConversionPattern {
  ONNXRangeOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXRangeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXRangeOpAdaptor operandAdaptor(operands);
    auto loc = op->getLoc();

    // Create an index expression scope.
    // Scope for krnl ops
    IndexExprScope ieScope(rewriter, loc);
    KrnlBuilder createKrnl(rewriter, loc);

    Value start = operandAdaptor.start();
    Value limit = operandAdaptor.limit();
    Value delta = operandAdaptor.delta();

    auto startShape = start.getType().cast<MemRefType>().getShape();
    auto limitShape = limit.getType().cast<MemRefType>().getShape();
    auto deltaShape = delta.getType().cast<MemRefType>().getShape();

    // Insert an allocation and deallocation for the result of this operation.
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    Type elementType = memRefType.getElementType();

    // Allocate result.
    Value alloc;
    Value zero = emitConstantOp(rewriter, loc, rewriter.getIndexType(), 0);

    // Load values depending on shape.
    Value loadedStart;
    if (startShape.size() == 0)
      loadedStart = rewriter.create<KrnlLoadOp>(loc, start, ArrayRef<Value>{});
    else if (startShape.size() == 1 && startShape[0] == 1)
      loadedStart = rewriter.create<KrnlLoadOp>(loc, start, zero);
    else
      llvm_unreachable("start shape must be 0 or if 1, size must be 1");

    Value loadedDelta;
    if (deltaShape.size() == 0)
      loadedDelta = rewriter.create<KrnlLoadOp>(loc, delta, ArrayRef<Value>{});
    else if (deltaShape.size() == 1 && deltaShape[0] == 1)
      loadedDelta = rewriter.create<KrnlLoadOp>(loc, delta, zero);
    else
      llvm_unreachable("delta shape must be 0 or if 1, size must be 1");

    bool insertDealloc = checkInsertDealloc(op);
    if (hasAllConstantDimensions(memRefType)) {
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    } else {
      Value loadedLimit;
      if (limitShape.size() == 0)
        loadedLimit =
            rewriter.create<KrnlLoadOp>(loc, limit, ArrayRef<Value>{});
      else if (limitShape.size() == 1 && limitShape[0] == 1)
        loadedLimit = rewriter.create<KrnlLoadOp>(loc, limit, zero);
      else
        llvm_unreachable("limit shape must be 0 or if 1, size must be 1");

      Value numberOfElements;
      TypeSwitch<Type>(elementType)
          .Case<Float16Type>([&](Type) {
            llvm_unreachable("Float 16 type not supported for Range op.");
          })
          .Case<Float32Type>([&](Type) {
            Value elements = rewriter.create<DivFOp>(loc,
                rewriter.create<SubFOp>(loc, loadedLimit, loadedStart),
                loadedDelta);
            numberOfElements = rewriter.create<IndexCastOp>(loc,
                rewriter.create<mlir::FPToUIOp>(loc,
                    rewriter.create<mlir::CeilFOp>(loc, elements),
                    rewriter.getIntegerType(64)),
                rewriter.getIndexType());
          })
          .Case<Float64Type>([&](Type) {
            Value elements = rewriter.create<DivFOp>(loc,
                rewriter.create<SubFOp>(loc, loadedLimit, loadedStart),
                loadedDelta);
            numberOfElements = rewriter.create<IndexCastOp>(loc,
                rewriter.create<mlir::FPToUIOp>(loc,
                    rewriter.create<mlir::CeilFOp>(loc, elements),
                    rewriter.getIntegerType(64)),
                rewriter.getIndexType());
          })
          .Case<IntegerType>([&](Type) {
            Value elements = rewriter.create<SignedCeilDivIOp>(loc,
                rewriter.create<SubIOp>(loc, loadedLimit, loadedStart),
                loadedDelta);
            numberOfElements = rewriter.create<IndexCastOp>(
                loc, elements, rewriter.getIndexType());
          });

      SmallVector<Value, 4> allocOperands;
      allocOperands.push_back(numberOfElements);
      IntegerAttr alignAttr = rewriter.getI64IntegerAttr(gDefaultAllocAlign);
      memref::AllocOp allocateMemref = rewriter.create<memref::AllocOp>(
          loc, memRefType, allocOperands, alignAttr);
      alloc = allocateMemref;
    }

    // Create a single loop.
    BuildKrnlLoop krnlLoop(rewriter, loc, 1);

    // Emit the definition.
    krnlLoop.createDefineOp();

    SmallVector<int64_t, 1> accShape;
    accShape.emplace_back(1);

    MemRefType accType;
    TypeSwitch<Type>(elementType)
        .Case<Float32Type>([&](Type) {
          accType = MemRefType::get(accShape, rewriter.getF32Type());
        })
        .Case<Float64Type>([&](Type) {
          accType = MemRefType::get(accShape, rewriter.getF64Type());
        })
        .Case<IntegerType>([&](Type) {
          auto width = elementType.cast<IntegerType>().getWidth();
          if (width == 8) {
            llvm_unreachable("Integer 8 type not supported for Range op.");
          } else if (width == 16) {
            accType = MemRefType::get(accShape, rewriter.getIntegerType(16));
          } else if (width == 32) {
            accType = MemRefType::get(accShape, rewriter.getIntegerType(32));
          } else if (width == 64) {
            accType = MemRefType::get(accShape, rewriter.getIntegerType(64));
          } else {
            llvm_unreachable(
                "Integer type over 64 bits not supported for Range op.");
          }
        });
    IntegerAttr alignAttr = rewriter.getI64IntegerAttr(gDefaultAllocAlign);
    auto acc = rewriter.create<memref::AllocOp>(loc, accType, alignAttr);

    // Acc index:
    SmallVector<IndexExpr, 4> accIndex;
    accIndex.emplace_back(LiteralIndexExpr(0));

    // Initialize accumulator with value:
    createKrnl.storeIE(loadedStart, acc, accIndex);

    // Emit body of the loop:
    // output[i] = start + (i * delta);
    int nIndex = krnlLoop.pushBounds(0, alloc, 0);
    krnlLoop.createIterateOp();
    rewriter.setInsertionPointToStart(krnlLoop.getIterateBlock());
    {
      // Read value:
      Value result = createKrnl.loadIE(acc, accIndex);

      // Store result:
      SmallVector<IndexExpr, 4> resultIndices;
      resultIndices.emplace_back(
          DimIndexExpr(krnlLoop.getInductionVar(nIndex)));
      createKrnl.storeIE(result, alloc, resultIndices);

      // Increment result:
      Value accResult;
      TypeSwitch<Type>(elementType)
          .Case<Float32Type>([&](Type) {
            accResult = rewriter.create<AddFOp>(loc, result, loadedDelta);
          })
          .Case<Float64Type>([&](Type) {
            accResult = rewriter.create<AddFOp>(loc, result, loadedDelta);
          })
          .Case<IntegerType>([&](Type) {
            accResult = rewriter.create<AddIOp>(loc, result, loadedDelta);
          });
      createKrnl.storeIE(accResult, acc, accIndex);
    }

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXRangeOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXRangeOpLowering>(ctx);
}
