/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- Conv.cpp - Lowering Convolution Op -------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Convolution Operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXConvOpLowering : public ConversionPattern {
  ONNXConvOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, mlir::ONNXConvOp::getOperationName(), 1, ctx) {}

  void convUnoptimized(ConversionPatternRewriter &rewriter,
      IndexExprScope *topScope, ONNXConvOp &convOp,
      ONNXConvOpAdaptor &operandAdaptor, ONNXConvOpShapeHelper &shapeHelper,
      MemRefType &memRefType, Value alloc) const {
    auto loc = convOp.getLoc();
    KrnlBuilder createKrnl(rewriter, loc);

    // Spatial data starts from the second dimension.
    int spatialStartIndex = 2;

    auto inputOperand = operandAdaptor.X();
    auto filterOperand = operandAdaptor.W();
    auto biasOperand = operandAdaptor.B();
    bool hasBias = !biasOperand.getType().isa<NoneType>();
    int64_t groupNum = convOp.group();
    IndexExpr G = LiteralIndexExpr(groupNum);
    Value fZero = emitConstantOp(rewriter, loc, memRefType.getElementType(), 0);

    // Bounds for output sizes: [N x CO x HO x WO]:
    // where N is Batch Size,
    // where CO (or M) is Channel Out (multiple of group num)
    // and where HO & WO are spacial dimensions of the output.
    int outputRank = shapeHelper.dimsForOutput().size();
    IndexExpr N = shapeHelper.dimsForOutput()[0];
    IndexExpr CO = shapeHelper.dimsForOutput()[1];
    IndexExpr COPerGroup = CO.ceilDiv(G);

    // Bounds for input image X: [N x CI x HI x WI]:
    // where N is Batch Size,
    // where CI (or C) is Channel In (multiple of group num),
    // and where HI & WI are spacial dimensions of the input image.
    MemRefBoundsIndexCapture inputBounds(inputOperand);

    // Bounds for kernel/filter W: [CO x CIPerGroup x KH x KW]:
    // where CO (or M) is Channel Out,
    // where CIPerGroup (or C/G) is number of channel in per group,
    // and where KH x KW are the kernel / filter size (e.g. 3x3, 1x1).
    MemRefBoundsIndexCapture filterBounds(filterOperand);
    IndexExpr CIPerGroup = filterBounds.getSymbol(1);

    // Determine the bounds for the loops over batch & channel out.
    IndexExpr iZero = LiteralIndexExpr(0);
    ValueRange outerLoops = createKrnl.defineLoops(3);
    SmallVector<IndexExpr, 3> outerLbs = {iZero, iZero, iZero};
    SmallVector<IndexExpr, 3> outerUbs = {N, G, COPerGroup};
    // Iterate over the outer loops
    // for n = 0 .. N:
    //   for g = 0 .. G:
    //     for coPerGroup = 0 .. COPerGroup:
    //       co = g * COPerGroup + coPerGroup;

    createKrnl.iterateIE(outerLoops, outerLoops, outerLbs, outerUbs,
        [&](KrnlBuilder &createKrnl, ValueRange outerIndices) {
          // Compute the Channel In Indices.
          IndexExprScope outerScope(createKrnl);
          // Compute the channel out index "co".
          DimIndexExpr g(outerIndices[1]);
          DimIndexExpr coPerGroup(outerIndices[2]);
          IndexExpr co = g * SymbolIndexExpr(COPerGroup) + coPerGroup;
          // Compute g * CIPerGroup for later use.
          IndexExpr gTimesCIPerGroup = g * SymbolIndexExpr(CIPerGroup);
          // Determine the bounds for the output spacial dimensions.
          int spacialRank = outputRank - spatialStartIndex;
          ValueRange outputSpacialLoops = createKrnl.defineLoops(spacialRank);
          SmallVector<IndexExpr, 3> outputSpacialLbs, outputSpacialUbs;
          for (int i = spatialStartIndex; i < outputRank; ++i) {
            outputSpacialLbs.emplace_back(iZero);
            outputSpacialUbs.emplace_back(
                SymbolIndexExpr(shapeHelper.dimsForOutput()[i]));
          }
          // Spacial loops.
          // for ho = 0 .. HO:
          //    for wo = 0 .. WO:
          createKrnl.iterateIE(outputSpacialLoops, outputSpacialLoops,
              outputSpacialLbs, outputSpacialUbs,
              [&](KrnlBuilder &createKrnl, ValueRange outputSpatialIndices) {
                IndexExprScope outputSpacialScope(createKrnl);
                MemRefBuilder createMemRef(createKrnl);
                // Create a local reduction value and set to zero.
                MemRefType tmpType =
                    MemRefType::get({}, memRefType.getElementType());
                // Single scalar, no need for default alignment.
                Value reductionVal = createMemRef.alloca(tmpType);
                createKrnl.store(fZero, reductionVal);

                // Bounds for reduction loops.
                ValueRange redLoops = createKrnl.defineLoops(spacialRank + 1);
                SmallVector<IndexExpr, 4> redLbs, redUbs, pMinOS;
                // First: loop over channel in per group.
                redLbs.emplace_back(iZero);
                redUbs.emplace_back(SymbolIndexExpr(CIPerGroup));
                // For each spacial dim, do the following.
                for (int i = 0; i < spacialRank; ++i) {
                  // Get data for dis spacial dimension.
                  DimIndexExpr o(outputSpatialIndices[i]);
                  SymbolIndexExpr I(
                      inputBounds.getSymbol(spatialStartIndex + i));
                  SymbolIndexExpr K(
                      filterBounds.getSymbol(spatialStartIndex + i));
                  SymbolIndexExpr p(
                      shapeHelper.pads[i]); // Begining/left/top pad.
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
                createKrnl.iterateIE(redLoops, redLoops, redLbs, redUbs,
                    [&](KrnlBuilder &createKrnl, ValueRange redIndices) {
                      IndexExprScope redScope(createKrnl);
                      MathBuilder createMath(createKrnl);
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
                          createKrnl.loadIE(inputOperand, inputAccessFct);
                      // Create access fct for filter: [co, ciPerG, kh, kw].
                      SmallVector<IndexExpr, 4> filterAccessFct;
                      filterAccessFct.emplace_back(DimIndexExpr(co));
                      filterAccessFct.emplace_back(DimIndexExpr(ciPerG));

                      for (int i = 0; i < spacialRank; ++i) {
                        DimIndexExpr k(redIndices[1 + i]);
                        filterAccessFct.emplace_back(k);
                      }
                      Value filter =
                          createKrnl.loadIE(filterOperand, filterAccessFct);
                      Value oldRed = createKrnl.load(reductionVal);
                      Value mul = createMath.mul(image, filter);
                      Value newRed = createMath.add(oldRed, mul);
                      createKrnl.store(newRed, reductionVal);
                    }); // Reduction loops.
                        // Finish the reduction and store in result array.
                Value result = createKrnl.load(reductionVal);
                // Store the result. Optionally add bias.
                SymbolIndexExpr coInOutputSpacial(co);
                if (hasBias) {
                  MathBuilder createMath(createKrnl);
                  Value bias =
                      createKrnl.loadIE(biasOperand, {coInOutputSpacial});
                  result = createMath.add(result, bias);
                }
                SmallVector<IndexExpr, 4> resAccessFunc;
                resAccessFunc.emplace_back(SymbolIndexExpr(outerIndices[0]));
                resAccessFunc.emplace_back(coInOutputSpacial);
                for (Value o : outputSpatialIndices)
                  resAccessFunc.emplace_back(DimIndexExpr(o));
                createKrnl.storeIE(result, alloc, resAccessFunc);
              }); // Output spacial loops.
        });       // Outer loops;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    ONNXConvOpAdaptor operandAdaptor(operands);
    ONNXConvOp convOp = llvm::dyn_cast<ONNXConvOp>(op);

    // Get shape.
    ONNXConvOpShapeHelper shapeHelper(&convOp, &rewriter,
        krnl::getDenseElementAttributeFromKrnlValue,
        krnl::loadDenseElementArrayValueAtIndex);
    auto shapecomputed = shapeHelper.computeShape(operandAdaptor);
    assert(succeeded(shapecomputed) && "Could not compute output shape");

    // Insert an allocation and deallocation for the result of this operation.
    MemRefType memRefType = convertToMemRefType(*op->result_type_begin());
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, memRefType, loc, shapeHelper.dimsForOutput());

    convUnoptimized(rewriter, shapeHelper.scope, convOp, operandAdaptor,
        shapeHelper, memRefType, alloc);

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNXConvOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXConvOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
