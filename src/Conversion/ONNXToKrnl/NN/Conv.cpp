/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- Conv.cpp - Lowering Convolution Op -------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Convolution Operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

//#include "src/Compiler/CompilerOptions.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXConvOpLowering : public ConversionPattern {
  ONNXConvOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel)
      : ConversionPattern(
            typeConverter, mlir::ONNXConvOp::getOperationName(), 1, ctx),
        enableParallel(enableParallel) {}
  bool enableParallel;

  void convUnoptimized(ConversionPatternRewriter &rewriter, ONNXConvOp &convOp,
      ONNXConvOpAdaptor &operandAdaptor, ONNXConvOpShapeHelper &shapeHelper,
      MemRefType &memRefType, Value alloc) const {
    Location loc = convOp.getLoc();
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, SCFBuilder,
        MathBuilder, MemRefBuilder>
        create(rewriter, loc);
    // Spatial data starts from the second dimension.
    int spatialStartIndex = 2;

    auto inputOperand = operandAdaptor.getX();
    auto filterOperand = operandAdaptor.getW();
    auto biasOperand = operandAdaptor.getB();
    bool hasBias = !biasOperand.getType().isa<NoneType>();
    int64_t groupNum = convOp.getGroup();
    IndexExpr G = LiteralIndexExpr(groupNum);
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
    IndexExpr iZero = LiteralIndexExpr(0);
    IndexExpr iOne = LiteralIndexExpr(1);

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

    // Create a local reduction value.
    MemRefType tmpType = MemRefType::get({}, memRefType.getElementType());
    // Single scalar, no need for default alignment.
    Value reductionVal = create.mem.alloca(tmpType);
    auto bodyFunction = [&](ValueRange outerIndices) {
      // Compute the Channel In Indices.
      IndexExprScope outerScope(create.krnl);
      // Compute the channel out index "co".
      DimIndexExpr g(outerIndices[1]);
      DimIndexExpr coPerGroup(outerIndices[2]);
      IndexExpr co = g * SymbolIndexExpr(COPerGroup) + coPerGroup;
      // Compute g * CIPerGroup for later use.
      IndexExpr gTimesCIPerGroup = g * SymbolIndexExpr(CIPerGroup);
      // Determine the bounds for the output spacial dimensions.
      int spacialRank = outputRank - spatialStartIndex;
      ValueRange outputSpacialLoops = create.krnl.defineLoops(spacialRank);
      SmallVector<IndexExpr, 3> outputSpacialLbs, outputSpacialUbs;
      for (int i = spatialStartIndex; i < outputRank; ++i) {
        outputSpacialLbs.emplace_back(iZero);
        outputSpacialUbs.emplace_back(
            SymbolIndexExpr(shapeHelper.getOutputDims()[i]));
      }
      // Spacial loops.
      // for ho = 0 .. HO:
      //    for wo = 0 .. WO:
      create.krnl.iterateIE(outputSpacialLoops, outputSpacialLoops,
          outputSpacialLbs, outputSpacialUbs,
          [&](KrnlBuilder &createKrnl, ValueRange outputSpatialIndices) {
            IndexExprScope outputSpacialScope(createKrnl);
            MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl,
                MathBuilder>
                create(createKrnl);
            // Reset reduction value to zero.
            create.krnl.store(fZero, reductionVal);

            // Bounds for reduction loops.
            ValueRange redLoops = create.krnl.defineLoops(spacialRank + 1);
            SmallVector<IndexExpr, 4> redLbs, redUbs, pMinOS;
            // First: loop over channel in per group.
            redLbs.emplace_back(iZero);
            redUbs.emplace_back(SymbolIndexExpr(CIPerGroup));
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
            create.krnl.iterateIE(redLoops, redLoops, redLbs, redUbs,
                [&](KrnlBuilder &createKrnl, ValueRange redIndices) {
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
                  IndexExpr ci = SymbolIndexExpr(gTimesCIPerGroup) + ciPerG;
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
                  filterAccessFct.emplace_back(DimIndexExpr(co));
                  filterAccessFct.emplace_back(DimIndexExpr(ciPerG));

                  for (int i = 0; i < spacialRank; ++i) {
                    DimIndexExpr k(redIndices[1 + i]);
                    filterAccessFct.emplace_back(k);
                  }
                  Value filter =
                      create.krnl.loadIE(filterOperand, filterAccessFct);
                  Value oldRed = create.krnl.load(reductionVal);
                  Value mul = create.math.mul(image, filter);
                  Value newRed = create.math.add(oldRed, mul);
                  create.krnl.store(newRed, reductionVal);
                }); // Reduction loops.
                    // Finish the reduction and store in result array.
            Value result = create.krnl.load(reductionVal);
            // Store the result. Optionally add bias.
            SymbolIndexExpr coInOutputSpacial(co);
            if (hasBias) {
              Value bias = create.krnl.loadIE(biasOperand, {coInOutputSpacial});
              result = create.math.add(result, bias);
            }
            SmallVector<IndexExpr, 4> resAccessFunc;
            resAccessFunc.emplace_back(SymbolIndexExpr(outerIndices[0]));
            resAccessFunc.emplace_back(coInOutputSpacial);
            for (Value o : outputSpatialIndices)
              resAccessFunc.emplace_back(DimIndexExpr(o));
            create.krnl.storeIE(result, alloc, resAccessFunc);
          }); // Output spacial loops.
    };

    if (enableParallel) {
      create.scf.parallelLoop(parLbs, parUbs, steps,
          [&](SCFBuilder &create, ValueRange outerIndices) {
            bodyFunction(outerIndices);
          });
    } else {
      ValueRange outerLoops = create.krnl.defineLoops(3);
      create.krnl.iterateIE(outerLoops, outerLoops, outerLbs, outerUbs,
          [&](KrnlBuilder &create, ValueRange outerIndices) {
            bodyFunction(outerIndices);
          });
    }
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = op->getLoc();
    ONNXConvOpAdaptor operandAdaptor(operands);
    ONNXConvOp convOp = llvm::dyn_cast<ONNXConvOp>(op);

    // Get shape.
    MultiDialectBuilder<IndexExprBuilderForKrnl, MemRefBuilder> create(
        rewriter, loc);

    ONNXConvOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && convertedType.isa<MemRefType>() &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = convertedType.cast<MemRefType>();

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc =
        create.mem.alignedAlloc(memRefType, shapeHelper.getOutputDims());

    convUnoptimized(
        rewriter, convOp, operandAdaptor, shapeHelper, memRefType, alloc);

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXConvOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel) {
  patterns.insert<ONNXConvOpLowering>(typeConverter, ctx, enableParallel);
}

} // namespace onnx_mlir
