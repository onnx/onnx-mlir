//===---------------- OneHot.cpp - Lowering OneHot Op -------------------===//
//
// This file lowers the ONNX OneHot Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXShapeHelper.hpp"

using namespace mlir;

struct ONNXOneHotOpLowering : public ConversionPattern {
  ONNXOneHotOpLowering(MLIRContext *ctx)
      : ConversionPattern(mlir::ONNXOneHotOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    ONNXOneHotOpAdaptor operandAdaptor(operands);
    ONNXOneHotOp oneHotOp = llvm::cast<ONNXOneHotOp>(op);
    auto loc = op->getLoc();

    Value indices = operandAdaptor.indices();
    Value depth = operandAdaptor.depth();
    Value values = operandAdaptor.values();

    auto depthShape = depth.getType().cast<MemRefType>().getShape();
    auto indicesShape = indices.getType().cast<MemRefType>().getShape();
    int64_t indicesRank = indicesShape.size();

    // Axis is an attribute with default value.
    int64_t axisValue = oneHotOp.axis();
    assert(axisValue >= 0 && axisValue <= indicesRank);

    // Insert an allocation and deallocation for the result of this operation.
    auto memRefOutType = convertToMemRefType(*op->result_type_begin());
    auto memRefOutShape = memRefOutType.getShape();
    int64_t outRank = memRefOutType.getRank();

    IndexExprScope ieScope(&rewriter, loc);
    DimsExpr outputDims(outRank);
    MemRefBoundsIndexCapture indicesBounds(indices);

    // Allocate result.
    Value alloc;
    Value zero = emitConstantOp(rewriter, loc, rewriter.getIndexType(), 0);
    Value one = emitConstantOp(rewriter, loc, rewriter.getIndexType(), 1);

    // Load values depending on shape.
    Value loadedDepth;
    // Inputs of `depth` should be a scalar in the description
    // (https://github.com/onnx/onnx/blob/master/docs/Operators.md#OneHot).
    // However, some test cases for onehot node accept tensor input.
    if (depthShape.size() == 0)
      loadedDepth = rewriter.create<KrnlLoadOp>(loc, depth, ArrayRef<Value>{});
    else if (depthShape.size() == 1 && depthShape[0] == 1)
      loadedDepth = rewriter.create<KrnlLoadOp>(loc, depth, zero);
    else
      llvm_unreachable("depth shape must be 0 or if 1, size must be 1");

    // convert float type to int64 for depth value
    Type depthElmentType =
        depth.getType().dyn_cast<MemRefType>().getElementType();

    TypeSwitch<Type>(depthElmentType)
        .Case<Float32Type>([&](Type) {
          loadedDepth = rewriter.create<mlir::FPToUIOp>(loc,
              rewriter.create<mlir::CeilFOp>(loc, loadedDepth),
              rewriter.getIntegerType(64));
        })
        .Case<IntegerType>([&](Type) {
          auto width = depthElmentType.cast<IntegerType>().getWidth();
          if (width > 64) {
            llvm_unreachable(
                "Integer type over 64 bits not supported for OneHot op.");
          }
        })
        .Default([](Type) {
          llvm_unreachable("Unsupported element type for OneHot op.");
        });

    bool insertDealloc = checkInsertDealloc(op);
    if (hasAllConstantDimensions(memRefOutType)) {
      alloc =
          insertAllocAndDealloc(memRefOutType, loc, rewriter, insertDealloc);
    } else {
      for (decltype(outRank) i = 0; i < outRank; i++) {
        if (i == axisValue) {
          Value outDim = rewriter.create<IndexCastOp>(
              loc, loadedDepth, rewriter.getIndexType());
          SymbolIndexExpr outDimIE(outDim);
          outputDims[i] = SymbolIndexExpr(outDimIE);
        } else if (memRefOutShape[i] != -1) {
          outputDims[i] = LiteralIndexExpr(memRefOutShape[i]);
        } else {
          if (i < axisValue) {
            Value outDim = indicesBounds.getDim(i).getValue();
            SymbolIndexExpr outDimIE(outDim);
            outputDims[i] = SymbolIndexExpr(outDimIE);
          } else {
            Value outDim = indicesBounds.getDim(i - 1).getValue();
            SymbolIndexExpr outDimIE(outDim);
            outputDims[i] = SymbolIndexExpr(outDimIE);
          }
        }
      }
      alloc = insertAllocAndDeallocSimple(
          rewriter, op, memRefOutType, loc, outputDims, insertDealloc);
    }

    Value off_value = rewriter.create<KrnlLoadOp>(loc, values, zero);
    Value on_value = rewriter.create<KrnlLoadOp>(loc, values, one);

    // 1. Krnl loops to initialize the result.
    BuildKrnlLoop initLoops(rewriter, loc, outRank);
    initLoops.createDefineOp();
    for (int i = 0; i < outRank; ++i)
      initLoops.pushBounds(0, alloc, i);
    initLoops.createIterateOp();
    auto initLoopBody = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(initLoops.getIterateBlock());
    {
      auto loopIVs = initLoops.getAllInductionVar();
      auto readIV = SmallVector<Value, 4>(loopIVs.begin(), loopIVs.end());
      rewriter.create<KrnlStoreOp>(loc, off_value, alloc, loopIVs);
    }
    rewriter.restoreInsertionPoint(initLoopBody);

    // Create loop.
    BuildKrnlLoop krnlLoop(rewriter, loc, indicesRank);

    // Emit the definition.
    krnlLoop.createDefineOp();

    for (int i = 0; i < indicesRank; ++i)
      krnlLoop.pushBounds(0, indices, i);

    krnlLoop.createIterateOp();
    rewriter.setInsertionPointToStart(krnlLoop.getIterateBlock());
    {
      auto loopIVs = krnlLoop.getAllInductionVar();
      auto readIV = SmallVector<Value, 4>(loopIVs.begin(), loopIVs.end());
      Value indiceValue = rewriter.create<KrnlLoadOp>(loc, indices, readIV);

      // convert float type to int64 for indiceValue value
      Type indicesElmentType =
          indices.getType().dyn_cast<MemRefType>().getElementType();
      TypeSwitch<Type>(indicesElmentType)
          .Case<Float32Type>([&](Type) {
            indiceValue = rewriter.create<mlir::FPToUIOp>(loc,
                rewriter.create<mlir::CeilFOp>(loc, indiceValue),
                rewriter.getIntegerType(64));
          })
          .Case<IntegerType>([&](Type) {
            auto width = indicesElmentType.cast<IntegerType>().getWidth();
            if (width > 64) {
              llvm_unreachable(
                  "Integer type over 64 bits not supported for OneHot op.");
            }
          })
          .Default([](Type) {
            llvm_unreachable("Unsupported element type for OneHot op.");
          });

      Value lessThanZero = rewriter.create<CmpIOp>(loc, CmpIPredicate::slt,
          indiceValue, emitConstantOp(rewriter, loc, rewriter.getI64Type(), 0));
      indiceValue = rewriter.create<SelectOp>(loc, lessThanZero,
          rewriter.create<AddIOp>(loc, indiceValue, loadedDepth), indiceValue);

      SmallVector<Value, 4> writeIV;
      for (int i = 0; i < outRank; ++i) {
        if (i == axisValue) {
          writeIV.push_back(rewriter.create<IndexCastOp>(
              loc, indiceValue, rewriter.getIndexType()));
        } else if (i < axisValue) {
          writeIV.push_back(readIV[i]);
        } else {
          writeIV.push_back(readIV[i - 1]);
        }
      }

      rewriter.create<KrnlStoreOp>(loc, on_value, alloc, writeIV);
    }

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXOneHotOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXOneHotOpLowering>(ctx);
}
