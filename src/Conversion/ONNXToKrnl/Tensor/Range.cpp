/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------- Range.cpp - Lowering Range Op --------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Range Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXRangeOpLowering : public OpConversionPattern<ONNXRangeOp> {
  ONNXRangeOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXRangeOp rangeOp, ONNXRangeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = rangeOp.getOperation();
    Location loc = ONNXLoc<ONNXRangeOp>(op);

    // Create an index expression scope.
    // Scope for krnl ops
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder,
        MathBuilder>
        create(rewriter, loc);
    IndexExprScope ieScope(create.krnl);

    Value start = adaptor.getStart();
    Value limit = adaptor.getLimit();
    Value delta = adaptor.getDelta();

    auto startShape = mlir::cast<MemRefType>(start.getType()).getShape();
    auto limitShape = mlir::cast<MemRefType>(limit.getType()).getShape();
    auto deltaShape = mlir::cast<MemRefType>(delta.getType()).getShape();

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = mlir::cast<MemRefType>(convertedType);
    Type elementType = memRefType.getElementType();

    // Insert an allocation and deallocation for the result of this operation.
    // Allocate result.
    Value alloc;
    Value zero = create.math.constant(rewriter.getIndexType(), 0);

    // Load values depending on shape.
    Value loadedStart = (startShape.size() == 0)
                            ? create.krnl.load(start)
                            : create.krnl.load(start, zero);
    assert((startShape.size() == 0 ||
               (startShape.size() == 1 && startShape[0] == 1)) &&
           "start shape must be 0 or if 1, size must be 1");

    Value loadedDelta = (deltaShape.size() == 0)
                            ? create.krnl.load(delta)
                            : create.krnl.load(delta, zero);
    assert((deltaShape.size() == 0 ||
               (deltaShape.size() == 1 && deltaShape[0] == 1)) &&
           "delta shape must be 0 or if 1, size must be 1");

    if (hasAllConstantDimensions(memRefType))
      alloc = create.mem.alignedAlloc(memRefType);
    else {
      Value loadedLimit = (limitShape.size() == 0)
                              ? create.krnl.load(limit)
                              : create.krnl.load(limit, zero);
      assert((limitShape.size() == 0 ||
                 (limitShape.size() == 1 && limitShape[0] == 1)) &&
             "limit shape must be 0 or if 1, size must be 1");

      Value numberOfElements;
      // TODO: many of the ops below exists in the create.math
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
    accIndex.emplace_back(LitIE(0));

    // Initialize accumulator with value:
    create.krnl.storeIE(loadedStart, acc, accIndex);

    ValueRange loopDef = create.krnl.defineLoops(1);
    SmallVector<IndexExpr, 4> ubs;
    create.krnlIE.getShapeAsDims(alloc, ubs);
    create.krnl.iterateIE(loopDef, loopDef, {LitIE(0)}, ubs,
        [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
          // Emit body of the loop:
          // output[i] = start + (i * delta);
          // Read value:
          Value result = createKrnl.loadIE(acc, accIndex);

          // Store result:
          SmallVector<IndexExpr, 4> resultIndices;
          resultIndices.emplace_back(DimIE(loopInd[0]));
          createKrnl.storeIE(result, alloc, resultIndices);

          // Increment result:
          Value accResult = create.math.add(result, loadedDelta);
          createKrnl.storeIE(accResult, acc, accIndex);
        });

    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXRangeOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXRangeOpLowering>(typeConverter, ctx);
}
} // namespace onnx_mlir
