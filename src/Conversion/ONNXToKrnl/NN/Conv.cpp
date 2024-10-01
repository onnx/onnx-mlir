/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- Conv.cpp - Lowering Convolution Op -------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Convolution Operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXConvOpLowering : public OpConversionPattern<ONNXConvOp> {
  ONNXConvOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel)
      : OpConversionPattern(typeConverter, ctx) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ONNXConvOp::getOperationName());
  }

  bool enableParallel;

  void convUnoptimized(ConversionPatternRewriter &rewriter, ONNXConvOp &convOp,
      ONNXConvOpAdaptor &operandAdaptor, ONNXConvOpShapeHelper &shapeHelper,
      MemRefType &memRefType, Value alloc) const {
    Operation *op = convOp.getOperation();
    Location loc = convOp.getLoc();
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, SCFBuilder,
        MathBuilder, MemRefBuilder>
        create(rewriter, loc);
    // Spatial data starts from the second dimension.
    int spatialStartIndex = 2;

    auto inputOperand = operandAdaptor.getX();
    auto filterOperand = operandAdaptor.getW();
    auto biasOperand = operandAdaptor.getB();
    bool hasBias = !mlir::isa<NoneType>(biasOperand.getType());
    int64_t groupNum = convOp.getGroup();
    IndexExpr G = LitIE(groupNum);
    Value fZero = create.math.constant(memRefType.getElementType(), 0);

    // Bounds for output sizes: [N x CO x HO x WO]:
    // where N is Batch Size,
    // where CO (or M) is Channel Out (multiple of group num)
    // and where HO & WO are spacial dimensions of the output.
    int outputRank = shapeHelper.getOutputDims().size();
    IndexExpr N = shapeHelper.getOutputDims()[0];
    IndexExpr CO = shapeHelper.getOutputDims()[1];
    IndexExpr COPerGroup = CO.ceilDiv(G);

    // Bounds for input image X: [N x CI x HI x WI]:
    // where N is Batch Size,
    // where CI (or C) is Channel In (multiple of group num),
    // and where HI & WI are spacial dimensions of the input image.

    // Bounds for kernel/filter W: [CO x CIPerGroup x KH x KW]:
    // where CO (or M) is Channel Out,
    // where CIPerGroup (or C/G) is number of channel in per group,
    // and where KH x KW are the kernel / filter size (e.g. 3x3, 1x1).
    IndexExpr CIPerGroup = create.krnlIE.getShapeAsSymbol(filterOperand, 1);

    // Determine the bounds for the loops over batch & channel out.
    IndexExpr iZero = LitIE(0);
    IndexExpr iOne = LitIE(1);

    SmallVector<Value, 3> lbsStorage, ubsStorage, stepsStorage;
    SmallVector<IndexExpr, 3> outerLbs = {iZero, iZero, iZero};
    SmallVector<IndexExpr, 3> outerUbs = {N, G, COPerGroup};
    SmallVector<IndexExpr, 3> outerSteps = {iOne, iOne, iOne};
    IndexExpr::getValues(outerLbs, lbsStorage);
    IndexExpr::getValues(outerUbs, ubsStorage);
    IndexExpr::getValues(outerSteps, stepsStorage);
    ValueRange parLbs(lbsStorage);
    ValueRange steps(stepsStorage);
    ValueRange parUbs(ubsStorage);
    // Iterate over the outer loops
    // for n = 0 .. N:
    //   for g = 0 .. G:
    //     for coPerGroup = 0 .. COPerGroup:
    //       co = g * COPerGroup + coPerGroup;

    auto bodyFunction = [&](ValueRange outerIndices) {
      // Compute the Channel In Indices.
      IndexExprScope outerScope(create.krnl);
      // Compute the channel out index "co".
      DimIndexExpr g(outerIndices[1]);
      DimIndexExpr coPerGroup(outerIndices[2]);
      IndexExpr co = g * SymIE(COPerGroup) + coPerGroup;
      // Compute g * CIPerGroup for later use.
      IndexExpr gTimesCIPerGroup = g * SymIE(CIPerGroup);
      // Determine the bounds for the output spacial dimensions.
      int spacialRank = outputRank - spatialStartIndex;
      ValueRange outputSpacialLoops = create.krnl.defineLoops(spacialRank);
      SmallVector<IndexExpr, 3> outputSpacialLbs, outputSpacialUbs;
      for (int i = spatialStartIndex; i < outputRank; ++i) {
        outputSpacialLbs.emplace_back(iZero);
        outputSpacialUbs.emplace_back(SymIE(shapeHelper.getOutputDims()[i]));
      }
      // Spacial loops.
      // for ho = 0 .. HO:
      //    for wo = 0 .. WO:
      create.krnl.iterateIE(outputSpacialLoops, outputSpacialLoops,
          outputSpacialLbs, outputSpacialUbs,
          [&](const KrnlBuilder &createKrnl, ValueRange outputSpatialIndices) {
            IndexExprScope outputSpacialScope(createKrnl);
            MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl,
                MathBuilder>
                create(createKrnl);

            ValueRange inits = ValueRange(fZero);

            // Bounds for reduction loops.
            ValueRange redLoops = create.krnl.defineLoops(spacialRank + 1);
            SmallVector<IndexExpr, 4> redLbs, redUbs, pMinOS;
            // First: loop over channel in per group.
            redLbs.emplace_back(iZero);
            redUbs.emplace_back(SymIE(CIPerGroup));
            // For each spacial dim, do the following.
            for (int i = 0; i < spacialRank; ++i) {
              // Get data for dis spacial dimension.
              DimIndexExpr o(outputSpatialIndices[i]);
              SymbolIndexExpr I(create.krnlIE.getShapeAsSymbol(
                  inputOperand, spatialStartIndex + i));
              SymbolIndexExpr K(create.krnlIE.getShapeAsSymbol(
                  filterOperand, spatialStartIndex + i));
              SymbolIndexExpr p(shapeHelper.pads[i]); // Beginning/left/top pad.
              LiteralIndexExpr s(shapeHelper.strides[i]);
              LiteralIndexExpr d(shapeHelper.dilations[i]);
              // lb = ceil((p - o * s) / d)
              IndexExpr pos = p - (o * s);
              IndexExpr lb = pos.ceilDiv(d);
              lb = IndexExpr::max(lb, 0);
              redLbs.emplace_back(lb);
              // ub = ceil((I + p - o * s) / d)
              IndexExpr ipos = I + pos;
              IndexExpr ub = ipos.ceilDiv(d);
              ub = IndexExpr::min(ub, K);
              redUbs.emplace_back(ub);
              // Save p - o * s for later use.
              pMinOS.emplace_back(pos);
            }
            // for ciPerGroup = 0 .. CIPerGroup:
            //   for kh in lb .. ub:
            //     for kw in lb .. ub:
            auto innerIterate =
                create.krnl.iterateIE(redLoops, redLoops, redLbs, redUbs, inits,
                    [&](const KrnlBuilder &createKrnl, ValueRange redIndices,
                        ValueRange iterArgs) {
                      // Get last argument for the iterate body.
                      Value iterArg = iterArgs.back();
                      IndexExprScope redScope(createKrnl);
                      MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl,
                          MathBuilder>
                          create(createKrnl);
                      // Create access function for input image:
                      // [n, ci, ho * sh + kh * dh - ph, wo * sw + kw * dw -
                      // pw].
                      SmallVector<IndexExpr, 4> inputAccessFct;
                      DimIndexExpr n(outerIndices[0]);
                      inputAccessFct.emplace_back(n);
                      // ci = g * CIPerG + ciPerG
                      DimIndexExpr ciPerG(redIndices[0]);
                      IndexExpr ci = SymIE(gTimesCIPerGroup) + ciPerG;
                      inputAccessFct.emplace_back(ci);
                      for (int i = 0; i < spacialRank; ++i) {
                        // for each spacial dims: access is o * s + k * d - p.
                        DimIndexExpr k(redIndices[1 + i]);
                        SymbolIndexExpr pos(pMinOS[i]);
                        LiteralIndexExpr d(shapeHelper.dilations[i]);
                        // k*d - (p - o*s) = k*d + o*s - p
                        IndexExpr t = (k * d) - pos;
                        inputAccessFct.emplace_back(t);
                      }
                      Value image =
                          create.krnl.loadIE(inputOperand, inputAccessFct);
                      // Create access fct for filter: [co, ciPerG, kh, kw].
                      SmallVector<IndexExpr, 4> filterAccessFct;
                      filterAccessFct.emplace_back(DimIE(co));
                      filterAccessFct.emplace_back(DimIE(ciPerG));

                      for (int i = 0; i < spacialRank; ++i) {
                        DimIndexExpr k(redIndices[1 + i]);
                        filterAccessFct.emplace_back(k);
                      }
                      Value filter =
                          create.krnl.loadIE(filterOperand, filterAccessFct);
                      Value oldRed = iterArg;
                      Value mul = create.math.mul(image, filter);
                      Value newRed = create.math.add(oldRed, mul);
                      create.krnl.yield(newRed);
                    }); // Reduction loops.
                        // Finish the reduction and store in result array.
            Value result = innerIterate.getResult(0);
            // Store the result. Optionally add bias.
            SymbolIndexExpr coInOutputSpacial(co);
            if (hasBias) {
              Value bias = create.krnl.loadIE(biasOperand, {coInOutputSpacial});
              result = create.math.add(result, bias);
            }
            SmallVector<IndexExpr, 4> resAccessFunc;
            resAccessFunc.emplace_back(SymIE(outerIndices[0]));
            resAccessFunc.emplace_back(coInOutputSpacial);
            for (Value o : outputSpatialIndices)
              resAccessFunc.emplace_back(DimIE(o));
            create.krnl.storeIE(result, alloc, resAccessFunc);
          }); // Output spacial loops.
    };

    ValueRange outerLoops = create.krnl.defineLoops(3);
    if (enableParallel) {
      int64_t parId;
      if (findSuitableParallelDimension(outerLbs, outerUbs, 0, 1, parId,
              /*min iter for going parallel*/ 4)) {
        create.krnl.parallel(outerLoops[0]);
        onnxToKrnlParallelReport(op, true, 0, outerLbs[0], outerUbs[0], "conv");
      } else {
        onnxToKrnlParallelReport(
            op, false, 0, outerLbs[0], outerUbs[0], "not enough work in conv");
      }
    }
    create.krnl.iterateIE(outerLoops, outerLoops, outerLbs, outerUbs,
        [&](const KrnlBuilder &create, ValueRange outerIndices) {
          bodyFunction(outerIndices);
        });
  }

  LogicalResult matchAndRewrite(ONNXConvOp convOp, ONNXConvOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = convOp.getOperation();
    Location loc = ONNXLoc<ONNXConvOp>(op);
    ValueRange operands = adaptor.getOperands();

    // Get shape.
    MultiDialectBuilder<IndexExprBuilderForKrnl, MemRefBuilder> create(
        rewriter, loc);

    ONNXConvOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Insert allocation for the result of this operation.
    Value alloc = allocForONNXOp<ONNXConvOp>(
        convOp, rewriter, typeConverter, shapeHelper)[0];
    MemRefType memRefType = mlir::cast<MemRefType>(alloc.getType());
    convUnoptimized(rewriter, convOp, adaptor, shapeHelper, memRefType, alloc);

    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXConvOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel,
    std::string opsForCall) {
  patterns.insert<ONNXConvOpToCall>(typeConverter, ctx, opsForCall);
  patterns.insert<ONNXConvOpLowering>(typeConverter, ctx, enableParallel);
}

} // namespace onnx_mlir
