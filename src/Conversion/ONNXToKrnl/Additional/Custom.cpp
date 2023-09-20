/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- Custom.cpp - Lowering Custom Op--------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNXCustomOp to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

void FixGRUY(Location loc, ConversionPatternRewriter &rewriter,
    ONNXCustomOp customOp, ValueRange operands, ValueRange outputAllocs) {
  Value Y = operands[0];
  Value sequenceLens = operands[1];
  Value initialH = operands[2];
  Value output = outputAllocs[0];

  MultiDialectBuilder<KrnlBuilder, MathBuilder, MemRefBuilder, OnnxBuilder>
      create(rewriter, loc);

  // Code copied from GRU.cpp: calculateState
  int64_t yRank = 4;
  Value iZero = create.math.constantIndex(0);
  SmallVector<Value, 2> yLbs(yRank, iZero);
  SmallVector<Value, 2> yUbs;
  for (unsigned r = 0; r < yRank; ++r) {
    yUbs.emplace_back(create.mem.dim(Y, r));
  }
  ValueRange loops = create.krnl.defineLoops(yRank);
  create.krnl.iterate(loops, loops, yLbs, yUbs,
      [&](KrnlBuilder &createKrnl, ValueRange indices) {
        MathBuilder createMath(createKrnl);
        IndexExprScope ieScope(createKrnl);
        Value sequenceIV(indices[0]);
        Value directionIV(indices[1]);
        Value bs(indices[2]), hs(indices[3]);

        Value currentV = createKrnl.load(Y, indices);
        Value sequenceUB = createKrnl.load(sequenceLens, {bs});
        Value initial;
        if (isNoneValue(initialH)) {
          initial = createMath.constant(currentV.getType(), 0.);
        } else {
          initial = createKrnl.load(initialH, {directionIV, bs, hs});
        }
        Value cond = createMath.sge(
            createMath.cast(sequenceUB.getType(), sequenceIV), sequenceUB);
        Value newV = createMath.select(cond, /*padding*/ initial, currentV);
        createKrnl.store(newV, output, indices);
      });
}

void FixGRUYh(Location loc, ConversionPatternRewriter &rewriter,
    ONNXCustomOp customOp, ValueRange operands, ValueRange outputAllocs) {
  Value Y = operands[0];
  Value sequenceLens = operands[1];
  Value Yh = outputAllocs[0];

  MultiDialectBuilder<KrnlBuilder, MathBuilder, MemRefBuilder, OnnxBuilder>
      create(rewriter, loc);

  // Code copied from GRU.cpp: calculateState
  int64_t htRank = 3;
  Value iZero = create.math.constantIndex(0);
  Value iOne = create.math.constantIndex(1);
  SmallVector<Value, 2> htLbs(htRank, iZero);
  SmallVector<Value, 2> htUbs;
  for (unsigned r = 0; r < htRank; ++r) {
    // skip the first two dim for sequence and batch
    htUbs.emplace_back(create.mem.dim(Y, r + 1));
  }
  Value seqSize = create.mem.dim(Y, 0);
  ValueRange loops = create.krnl.defineLoops(htRank);
  create.krnl.iterate(loops, loops, htLbs, htUbs,
      [&](KrnlBuilder &createKrnl, ValueRange indices) {
        MathBuilder createMath(createKrnl);
        IndexExprScope ieScope(createKrnl);
        Value bs(indices[1]), hs(indices[2]);
        Value directionIV(indices[0]);
        Value sequenceUB = createKrnl.load(sequenceLens, {bs});
        Value bound = createMath.min(
            createMath.cast(seqSize.getType(), sequenceUB), seqSize);
        Value index = createMath.sub(bound, iOne);
        Value lastHt = createKrnl.load(Y, {index, directionIV, bs, hs});
        createKrnl.store(lastHt, Yh, {directionIV, bs, hs});
      });
}

struct ONNXCustomOpLowering : public OpConversionPattern<ONNXCustomOp> {
  ONNXCustomOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXCustomOp customOp,
      ONNXCustomOpAdaptor operandAdaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = customOp.getOperation();
    Location loc = op->getLoc();
    ValueRange operands = operandAdaptor.getOperands();

    // Helper builders.
    MultiDialectBuilder<AffineBuilder, IndexExprBuilderForKrnl, KrnlBuilder,
        MemRefBuilder>
        create(rewriter, loc);
    IndexExprScope scope(create.krnlIE);

    // Get shape.
    ONNXCustomOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Prepare outputs for krnl.call
    SmallVector<Type, 4> outputMemRefTypes;
    SmallVector<Value, 4> outputAllocs;
    for (size_t idx = 0; idx < op->getResultTypes().size(); idx++) {
      Type ty = op->getResultTypes()[idx];
      MemRefType outputMemRefType =
          typeConverter->convertType(ty).cast<MemRefType>();
      outputMemRefTypes.emplace_back(outputMemRefType);
      Value alloc = create.mem.alignedAlloc(
          outputMemRefType, shapeHelper.getOutputDims(idx));
      outputAllocs.emplace_back(alloc);
    }

    // Lower to Krnl for special CustomOp
    if (customOp.getFunctionName() == "FixGRUYh") {
      FixGRUYh(loc, rewriter, customOp, operands, outputAllocs);
    } else if (customOp.getFunctionName() == "FixGRUY") {
      FixGRUY(loc, rewriter, customOp, operands, outputAllocs);
    } else {
      // Create Krnl.Call

      // Handle the attributes: exclude the attributes used for analysis
      // function_name is passed explicitly. Others may include shape inference
      std::vector<std::string> excludeStrings = {"function_name",
          "shape_infer_pattern", "inputs_for_infer", "output_element_type"};
      std::vector<std::string> attributeNames;
      for (NamedAttribute namedAttr : customOp->getAttrs()) {
        std::string attrName = namedAttr.getName().getValue().str();
        if (std::find(excludeStrings.begin(), excludeStrings.end(), attrName) ==
            excludeStrings.end())
          attributeNames.push_back(attrName);
      }
      rewriter.create<KrnlCallOp>(loc, customOp.getFunctionName().str(),
          outputAllocs, op, operands, attributeNames);
    }

    rewriter.replaceOp(op, outputAllocs);
    return success();
  }
};

void populateLoweringONNXCustomOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXCustomOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
