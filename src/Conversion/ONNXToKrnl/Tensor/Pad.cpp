/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-----------------------Pad.cpp - Lowering Pad Op -------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Pad  Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXPadOpLowering : public ConversionPattern {
  ONNXPadOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXPadOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    ONNXPadOp padOp = llvm::dyn_cast<ONNXPadOp>(op);
    ONNXPadOpAdaptor operandAdaptor(operands);
    Value data = operandAdaptor.data();
    Value constantValue = operandAdaptor.constant_value();
    StringRef padMode = padOp.mode();

    // Builder helper.
    KrnlBuilder createKrnl(rewriter, loc);

    // Shape helper.
    ONNXPadOpShapeHelper shapeHelper(&padOp, &rewriter,
        krnl::getDenseElementAttributeFromKrnlValue,
        krnl::loadDenseElementArrayValueAtIndex);
    auto shapecomputed = shapeHelper.computeShape(operandAdaptor);
    assert(succeeded(shapecomputed) && "Could not compute output shape");

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType resMemRefType = convertedType.cast<MemRefType>();
    Type resElementType = resMemRefType.getElementType();

    // Insert an allocation and deallocation for the output of this operation.
    Value resMemRef = insertAllocAndDeallocSimple(
        rewriter, op, resMemRefType, loc, shapeHelper.dimsForOutput());

    // Bounds.
    MemRefBoundsIndexCapture dataBounds(data);
    MemRefBoundsIndexCapture resBounds(resMemRef);
    uint64_t rank = dataBounds.getRank();

    // Literal indices.
    LiteralIndexExpr zeroIE(0);
    LiteralIndexExpr oneIE(1);
    LiteralIndexExpr twoIE(2);

    if (padMode.equals_insensitive("constant")) {
      // 'constant' mode.
      // We first initialize the result tensor with the constant value, and then
      // iterate over the input and copy values from the input to the result.
      // This way is to avoid using `select` in computing indices as doing for
      // 'edge' and 'reflect' modes.
      Value cValue;
      if (constantValue.getType().isa<NoneType>()) {
        // Default to 0 if constant_value is not specified.
        MathBuilder createMath(rewriter, loc);
        cValue = createMath.constant(resElementType, 0);
      } else
        cValue = createKrnl.load(constantValue, {});

      // Initialize the result to the constant value.
      createKrnl.memset(resMemRef, cValue);

      // Copy values from the input to the result.
      // Iterate over the input tensor dimensions.
      SmallVector<IndexExpr, 4> lbs(rank, zeroIE);
      SmallVector<IndexExpr, 4> ubs;
      dataBounds.getDimList(ubs);
      ValueRange mainLoopDef = createKrnl.defineLoops(rank);
      createKrnl.iterateIE(mainLoopDef, mainLoopDef, lbs, ubs,
          [&](KrnlBuilder &createKrnl, ValueRange dataLoopInd) {
            SmallVector<IndexExpr, 4> resLoopInd;
            for (uint64_t i = 0; i < rank; ++i) {
              IndexExpr resInd =
                  DimIndexExpr(dataLoopInd[i]) + shapeHelper.pads[i];
              resLoopInd.emplace_back(resInd);
            }
            Value dataValue = createKrnl.load(data, dataLoopInd);
            createKrnl.storeIE(dataValue, resMemRef, resLoopInd);
          });
    } else {
      // 'edge' and 'reflect' modes.
      SmallVector<IndexExpr, 4> lbs(rank, zeroIE);
      SmallVector<IndexExpr, 4> ubs;
      resBounds.getDimList(ubs);
      // Copy values from the input to the result.
      // Iterate over the result tensor dimensions.
      ValueRange mainLoopDef = createKrnl.defineLoops(rank);
      createKrnl.iterateIE(mainLoopDef, mainLoopDef, lbs, ubs,
          [&](KrnlBuilder &createKrnl, ValueRange resLoopInd) {
            SmallVector<IndexExpr, 4> dataLoopInd;
            for (uint64_t i = 0; i < rank; ++i) {
              IndexExpr dataInd = DimIndexExpr(resLoopInd[i]);
              IndexExpr pad = shapeHelper.pads[i];
              IndexExpr dim = dataBounds.getDim(i);
              if (padMode.equals_insensitive("edge")) {
                // Before the left side of input. Use values on the left edge.
                dataInd = dataInd.select(dataInd <= pad, zeroIE, dataInd - pad);
                // After the right side of input. Use values on the right edge.
                dataInd = dataInd.selectOrSelf(dataInd >= dim, dim - oneIE);
              }
              if (padMode.equals_insensitive("reflect")) {
                // Before the left side of input. Reflect on the left edge.
                dataInd =
                    dataInd.select(dataInd < pad, pad - dataInd, dataInd - pad);
                // After the right side of input. Reflect on the right edge.
                dataInd = dataInd.selectOrSelf(
                    dataInd >= dim, dim - (dataInd - dim) - twoIE);
              }
              dataLoopInd.emplace_back(dataInd);
            }
            Value dataValue = createKrnl.loadIE(data, dataLoopInd);
            createKrnl.store(dataValue, resMemRef, resLoopInd);
          });
    }

    // Replace the original op with the generated code.
    rewriter.replaceOp(op, resMemRef);

    return success();
  }
};

void populateLoweringONNXPadOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXPadOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
