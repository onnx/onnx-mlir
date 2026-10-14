/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- DeformConv.cpp - Lowering DeformConv Op ----------------===//
//
// This file lowers the ONNX DeformConv Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

// Largest spatial rank we lower. Multilinear interpolation needs 2^rank
// corners per sampled point, so the generated code grows exponentially; 3D is
// already past what the ONNX spec calls a common use case.
static constexpr int64_t deformConvMaxSpatialRank = 3;

// Multilinear (bilinear for 2D) interpolation of `input[n, c, ...]` at the
// floating point location `coords`, one coordinate per spatial axis. Following
// the ONNX spec, samples that fall outside the input contribute zero, which
// matches a GridSample with "zeros" padding over the already padded input.
static Value sampleMultilinear(const KrnlBuilder &createKrnl,
    const MathBuilder &math, Value input, Value n, Value c,
    ArrayRef<Value> coords, ArrayRef<Value> inputSpatialDims,
    Type elementType) {
  int64_t spatialRank = coords.size();
  Value zeroIdx = math.constantIndex(0);
  Value oneIdx = math.constantIndex(1);
  Value fZero = math.constant(elementType, 0.0);
  Value fOne = math.constant(elementType, 1.0);

  // Lower corner of the interpolation cell, and the fractional distance from
  // it, along each spatial axis.
  SmallVector<Value, 3> lowerIdx, frac;
  for (int64_t i = 0; i < spatialRank; ++i) {
    Value floored = math.floor(coords[i]);
    lowerIdx.emplace_back(math.castToIndex(floored));
    frac.emplace_back(math.sub(coords[i], floored));
  }

  // Accumulate over the 2^rank corners of the cell. Bit i of `corner` selects
  // the upper neighbor along spatial axis i.
  Value result = fZero;
  for (int64_t corner = 0; corner < (int64_t(1) << spatialRank); ++corner) {
    Value weight = fOne;
    Value inBounds = nullptr;
    SmallVector<Value, 5> accessFct = {n, c};
    for (int64_t i = 0; i < spatialRank; ++i) {
      bool upper = (corner >> i) & 1;
      weight = math.mul(weight, upper ? frac[i] : math.sub(fOne, frac[i]));
      Value idx = upper ? math.add(lowerIdx[i], oneIdx) : lowerIdx[i];
      Value dimMax = math.sub(inputSpatialDims[i], oneIdx);
      // Hoisted so that the two comparisons are emitted in a fixed order:
      // as arguments of one call their evaluation order is unspecified, and
      // each one creates an op, so the generated IR would differ by compiler.
      Value geZero = math.sge(idx, zeroIdx);
      Value leMax = math.sle(idx, dimMax);
      Value valid = math.andi(geZero, leMax);
      inBounds = inBounds ? math.andi(inBounds, valid) : valid;
      // Clamp so that the load below stays in range; out of bounds corners
      // are masked out to zero right after.
      accessFct.emplace_back(math.min(math.max(idx, zeroIdx), dimMax));
    }
    Value val = createKrnl.load(input, accessFct);
    val = math.select(inBounds, val, fZero);
    result = math.add(result, math.mul(weight, val));
  }
  return result;
}

struct ONNXDeformConvOpLowering : public OpConversionPattern<ONNXDeformConvOp> {
  ONNXDeformConvOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel)
      : OpConversionPattern(typeConverter, ctx) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ONNXDeformConvOp::getOperationName());
  }

private:
  bool enableParallel = false;

public:
  LogicalResult matchAndRewrite(ONNXDeformConvOp deformConvOp,
      ONNXDeformConvOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = deformConvOp.getOperation();
    Location loc = ONNXLoc<ONNXDeformConvOp>(op);
    ValueRange operands = adaptor.getOperands();

    Value inputOperand = adaptor.getX();
    Value filterOperand = adaptor.getW();
    Value offsetOperand = adaptor.getOffset();
    Value biasOperand = adaptor.getB();
    Value maskOperand = adaptor.getMask();
    bool hasBias = !mlir::isa<NoneType>(biasOperand.getType());
    bool hasMask = !mlir::isa<NoneType>(maskOperand.getType());

    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder,
        MemRefBuilder>
        create(rewriter, loc);

    // Get shape.
    ONNXDeformConvOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    DimsExpr outputDims = shapeHelper.getOutputDims();
    int64_t outputRank = outputDims.size();
    int64_t spatialRank = outputRank - 2;
    if (spatialRank < 1 || spatialRank > deformConvMaxSpatialRank)
      return rewriter.notifyMatchFailure(
          op, "DeformConv lowering supports 1-D to 3-D spatial data only");

    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);
    Type elementType = outputMemRefType.getElementType();
    Value alloc = create.mem.alignedAlloc(outputMemRefType, outputDims);

    int64_t groupNum = deformConvOp.getGroup();
    int64_t offsetGroupNum = deformConvOp.getOffsetGroup();

    // Output: [N x oC x o1 x ... x on], weights: [oC x C/group x k1 x ... kn].
    IndexExpr N = outputDims[0];
    IndexExpr G = LitIE(groupNum);
    IndexExpr COPerGroup = outputDims[1].floorDiv(G);
    IndexExpr CIPerGroup = create.krnlIE.getShapeAsSymbol(filterOperand, 1);

    // Values that are loop invariant and only ever used inside the nest below.
    // `offset` channels are laid out as
    //   (((og * k1 + i1) * k2 + i2) ... * kn + in) * spatialRank + axis,
    // and `mask` channels as the same expression without the last two terms.
    Value coPerGroupVal = COPerGroup.getValue();
    Value ciPerGroupVal = CIPerGroup.getValue();
    Value ciPerOffsetGroupVal = create.krnlIE.getShapeAsSymbol(inputOperand, 1)
                                    .floorDiv(LitIE(offsetGroupNum))
                                    .getValue();
    Value spatialRankVal = create.math.constantIndex(spatialRank);
    // DeformConv has no auto_pad, so the pads are plain literals; the shape
    // helper has already defaulted strides and dilations to 1.
    SmallVector<int64_t, 3> padsBegin(spatialRank, 0);
    if (std::optional<ArrayAttr> pads = deformConvOp.getPads())
      for (int64_t i = 0; i < spatialRank; ++i)
        padsBegin[i] = mlir::cast<IntegerAttr>((*pads)[i]).getInt();
    SmallVector<Value, 3> kernelDimVals, inputSpatialDims;
    Value kernelElemsVal = create.math.constantIndex(1);
    for (int64_t i = 0; i < spatialRank; ++i) {
      Value k = shapeHelper.kernelShape[i].getValue();
      kernelDimVals.emplace_back(k);
      kernelElemsVal = create.math.mul(kernelElemsVal, k);
      inputSpatialDims.emplace_back(
          create.krnlIE.getShapeAsSymbol(inputOperand, 2 + i).getValue());
    }
    Value fZero = create.math.constant(elementType, 0);

    // Iterate over the outer loops:
    // for n = 0 .. N:
    //   for g = 0 .. G:
    //     for coPerGroup = 0 .. COPerGroup:
    //       co = g * COPerGroup + coPerGroup;
    SmallVector<IndexExpr, 3> outerLbs(3, LitIE(0));
    SmallVector<IndexExpr, 3> outerUbs = {N, G, COPerGroup};
    ValueRange outerLoops = create.krnl.defineLoops(3);
    // Every (n, g, coPerGroup) iteration stores to its own output element and
    // the reduction accumulates in an iterArg, so any of the three may be the
    // parallel level. Searching all of them matters because `n` alone leaves
    // the common batch of one with nothing to parallelize over.
    auto plan = KrnlParallelPlan::noCollapse(outerLoops, /*first*/ 0,
        /*last excl*/ 3, {.minTripCountForParallel = 16});
    if (enableParallel)
      plan.tryCreateParallel(create.krnl, op, "deformconv", outerLbs, outerUbs);
    create.krnl.iterateIE(outerLoops, plan.optimizedLoopDef(), outerLbs,
        outerUbs, [&](const KrnlBuilder &createKrnl, ValueRange outerInd) {
          MultiDialectBuilder<KrnlBuilder, MathBuilder> create(createKrnl);
          Value n = outerInd[0];
          Value g = outerInd[1];
          Value co =
              create.math.add(create.math.mul(g, coPerGroupVal), outerInd[2]);
          Value gTimesCIPerGroup = create.math.mul(g, ciPerGroupVal);

          // Spatial loops over the output: for o1 .. for on.
          SmallVector<IndexExpr, 3> spatialLbs(spatialRank, LitIE(0));
          SmallVector<IndexExpr, 3> spatialUbs;
          for (int64_t i = 0; i < spatialRank; ++i)
            spatialUbs.emplace_back(outputDims[2 + i]);
          ValueRange spatialLoops = create.krnl.defineLoops(spatialRank);
          create.krnl.iterateIE(spatialLoops, spatialLoops, spatialLbs,
              spatialUbs,
              [&](const KrnlBuilder &createKrnl, ValueRange spatialInd) {
                MultiDialectBuilder<KrnlBuilder, MathBuilder> create(
                    createKrnl);
                SmallVector<Value, 3> outSpatial(
                    spatialInd.begin(), spatialInd.end());

                // Reduction loops over the input channels of the group and
                // over every kernel position:
                // for ciPerG = 0 .. CIPerGroup:
                //   for k1 = 0 .. K1: ... for kn = 0 .. Kn:
                SmallVector<IndexExpr, 4> redLbs(spatialRank + 1, LitIE(0));
                SmallVector<IndexExpr, 4> redUbs = {CIPerGroup};
                for (int64_t i = 0; i < spatialRank; ++i)
                  redUbs.emplace_back(shapeHelper.kernelShape[i]);
                ValueRange redLoops = create.krnl.defineLoops(spatialRank + 1);
                auto redIterate = create.krnl.iterateIE(redLoops, redLoops,
                    redLbs, redUbs, ValueRange(fZero),
                    [&](const KrnlBuilder &createKrnl, ValueRange redInd,
                        ValueRange iterArgs) {
                      MultiDialectBuilder<KrnlBuilder, MathBuilder> create(
                          createKrnl);
                      MathBuilder &math = create.math;
                      Value ciPerG = redInd[0];
                      Value ci = math.add(gTimesCIPerGroup, ciPerG);
                      Value og = math.floorDiv(ci, ciPerOffsetGroupVal);

                      // Flatten the kernel position, then turn it into the
                      // first offset/mask channel it owns.
                      Value kFlat = math.constantIndex(0);
                      for (int64_t i = 0; i < spatialRank; ++i)
                        kFlat = math.add(
                            math.mul(kFlat, kernelDimVals[i]), redInd[1 + i]);
                      Value maskChan =
                          math.add(math.mul(og, kernelElemsVal), kFlat);
                      Value offChanBase = math.mul(maskChan, spatialRankVal);

                      // Sampling location: the location this kernel element
                      // would read in a plain convolution, shifted by the
                      // offset given for this output position.
                      SmallVector<Value, 3> coords;
                      for (int64_t i = 0; i < spatialRank; ++i) {
                        // o_i * stride_i + k_i * dilation_i - padBegin_i, the
                        // location a plain convolution would read.
                        // Hoisted for a fixed emission order, see the note
                        // in sampleMultilinear.
                        Value strided = math.mul(outSpatial[i],
                            math.constantIndex(shapeHelper.strides[i]));
                        Value dilated = math.mul(redInd[1 + i],
                            math.constantIndex(shapeHelper.dilations[i]));
                        Value base = math.add(strided, dilated);
                        base = math.sub(base, math.constantIndex(padsBegin[i]));
                        SmallVector<Value, 5> offAccessFct = {
                            n, math.add(offChanBase, math.constantIndex(i))};
                        offAccessFct.append(
                            outSpatial.begin(), outSpatial.end());
                        Value offVal =
                            create.krnl.load(offsetOperand, offAccessFct);
                        coords.emplace_back(
                            math.add(math.cast(elementType, base), offVal));
                      }

                      Value sampled =
                          sampleMultilinear(createKrnl, math, inputOperand, n,
                              ci, coords, inputSpatialDims, elementType);

                      SmallVector<Value, 5> filterAccessFct = {co, ciPerG};
                      for (int64_t i = 0; i < spatialRank; ++i)
                        filterAccessFct.emplace_back(redInd[1 + i]);
                      Value filter =
                          create.krnl.load(filterOperand, filterAccessFct);
                      Value prod = math.mul(sampled, filter);
                      if (hasMask) {
                        SmallVector<Value, 5> maskAccessFct = {n, maskChan};
                        maskAccessFct.append(
                            outSpatial.begin(), outSpatial.end());
                        prod = math.mul(
                            prod, create.krnl.load(maskOperand, maskAccessFct));
                      }
                      create.krnl.yield(math.add(iterArgs.back(), prod));
                    });

                Value result = redIterate.getResult(0);
                if (hasBias) {
                  SmallVector<Value, 1> biasAccessFct = {co};
                  result = create.math.add(
                      result, create.krnl.load(biasOperand, biasAccessFct));
                }
                SmallVector<Value, 5> resAccessFct = {n, co};
                resAccessFct.append(outSpatial.begin(), outSpatial.end());
                create.krnl.store(result, alloc, resAccessFct);
              });
        });

    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXDeformConvOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, bool enableParallel) {
  patterns.insert<ONNXDeformConvOpLowering>(typeConverter, ctx, enableParallel);
}

} // namespace onnx_mlir
