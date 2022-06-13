/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------- Range.cpp - Lowering Range Op --------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Range Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXRangeOpLowering : public ConversionPattern {
  ONNXRangeOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXRangeOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXRangeOpAdaptor operandAdaptor(operands);
    auto loc = op->getLoc();

    // Create an index expression scope.
    // Scope for krnl ops
    IndexExprScope ieScope(&rewriter, loc);
    KrnlBuilder createKrnl(rewriter, loc);

    Value start = operandAdaptor.start();
    Value limit = operandAdaptor.limit();
    Value delta = operandAdaptor.delta();

    auto startShape = start.getType().cast<MemRefType>().getShape();
    auto limitShape = limit.getType().cast<MemRefType>().getShape();
    auto deltaShape = delta.getType().cast<MemRefType>().getShape();

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = convertedType.cast<MemRefType>();
    Type elementType = memRefType.getElementType();

    // Insert an allocation and deallocation for the result of this operation.
    // Allocate result.
    Value alloc;
    MathBuilder createMath(rewriter, loc);
    Value zero = createMath.constant(rewriter.getIndexType(), 0);

    MultiDialectBuilder<KrnlBuilder, MemRefBuilder, MathBuilder> create(
        rewriter, loc);

    // Load values depending on shape.
    Value loadedStart = (startShape.size() == 0) ? createKrnl.load(start)
                                                 : createKrnl.load(start, zero);
    assert((startShape.size() == 0 ||
               (startShape.size() == 1 && startShape[0] == 1)) &&
           "start shape must be 0 or if 1, size must be 1");

    Value loadedDelta = (deltaShape.size() == 0) ? createKrnl.load(delta)
                                                 : createKrnl.load(delta, zero);
    assert((deltaShape.size() == 0 ||
               (deltaShape.size() == 1 && deltaShape[0] == 1)) &&
           "delta shape must be 0 or if 1, size must be 1");

    bool insertDealloc = checkInsertDealloc(op);
    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else {
      Value loadedLimit = (limitShape.size() == 0)
                              ? createKrnl.load(limit)
                              : createKrnl.load(limit, zero);
      assert((limitShape.size() == 0 ||
                 (limitShape.size() == 1 && limitShape[0] == 1)) &&
             "limit shape must be 0 or if 1, size must be 1");

      Value numberOfElements;
      TypeSwitch<Type>(elementType)
          .Case<Float16Type>([&](Type) {
            llvm_unreachable("Float 16 type not supported for Range op.");
          })
          .Case<Float32Type>([&](Type) {
            Value elements = create.math.div(
                create.math.sub(loadedLimit, loadedStart), loadedDelta);
            numberOfElements = rewriter.create<arith::IndexCastOp>(loc,
                rewriter.getIndexType(),
                rewriter.create<arith::FPToUIOp>(loc,
                    rewriter.getIntegerType(64),
                    rewriter.create<math::CeilOp>(loc, elements)));
          })
          .Case<Float64Type>([&](Type) {
            Value elements = create.math.div(
                create.math.sub(loadedLimit, loadedStart), loadedDelta);
            numberOfElements = rewriter.create<arith::IndexCastOp>(loc,
                rewriter.getIndexType(),
                rewriter.create<arith::FPToUIOp>(loc,
                    rewriter.getIntegerType(64),
                    rewriter.create<math::CeilOp>(loc, elements)));
          })
          .Case<IntegerType>([&](Type) {
            Value elements = rewriter.create<arith::CeilDivSIOp>(
                loc, create.math.sub(loadedLimit, loadedStart), loadedDelta);
            numberOfElements = rewriter.create<arith::IndexCastOp>(
                loc, rewriter.getIndexType(), elements);
          });

      SmallVector<Value, 4> allocOperands;
      allocOperands.push_back(numberOfElements);
      alloc = create.mem.alignedAlloc(memRefType, allocOperands);
    }

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
        .Case<IntegerType>([&](IntegerType type) {
          switch (type.getWidth()) {
          case 8:
            llvm_unreachable("Integer 8 type not supported for Range op.");
            break;
          case 16:
            accType = MemRefType::get(accShape, rewriter.getIntegerType(16));
            break;
          case 32:
            accType = MemRefType::get(accShape, rewriter.getIntegerType(32));
            break;
          case 64:
            accType = MemRefType::get(accShape, rewriter.getIntegerType(64));
            break;
          default:
            llvm_unreachable(
                "Integer type over 64 bits not supported for Range op.");
          }
        });
    Value acc = create.mem.alignedAlloc(accType);

    // Acc index:
    SmallVector<IndexExpr, 4> accIndex;
    accIndex.emplace_back(LiteralIndexExpr(0));

    // Initialize accumulator with value:
    createKrnl.storeIE(loadedStart, acc, accIndex);

    ValueRange loopDef = createKrnl.defineLoops(1);
    MemRefBoundsIndexCapture allocBounds(alloc);
    SmallVector<IndexExpr, 4> ubs;
    allocBounds.getDimList(ubs);
    createKrnl.iterateIE(loopDef, loopDef, {LiteralIndexExpr(0)}, ubs,
        [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
          // Emit body of the loop:
          // output[i] = start + (i * delta);
          // Read value:
          Value result = createKrnl.loadIE(acc, accIndex);

          // Store result:
          SmallVector<IndexExpr, 4> resultIndices;
          resultIndices.emplace_back(DimIndexExpr(loopInd[0]));
          createKrnl.storeIE(result, alloc, resultIndices);

          // Increment result:
          Value accResult = create.math.add(result, loadedDelta);
          createKrnl.storeIE(accResult, acc, accIndex);
        });

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXRangeOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXRangeOpLowering>(typeConverter, ctx);
}
} // namespace onnx_mlir
