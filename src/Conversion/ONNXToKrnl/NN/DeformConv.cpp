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

// One corner of the interpolation cell around a sampling location: the weight
// it contributes, whether it lies inside the input, and the clamped spatial
// indices to load from. All of it is a function of the location alone, never
// of the channel being sampled, so it is built once and reused for every input
// channel that is sampled there.
struct InterpolationCorner {
  Value weight;
  Value inBounds;
  SmallVector<Value, 3> spatialIdx;
};

// Build the 2^rank corners of the interpolation cell around the floating point
// location `coords`, one coordinate per spatial axis.
static SmallVector<InterpolationCorner, 8> prepareMultilinear(
    const MathBuilder &math, ArrayRef<Value> coords,
    ArrayRef<Value> inputSpatialDims, Type elementType) {
  int64_t spatialRank = coords.size();
  Value zeroIdx = math.constantIndex(0);
  Value oneIdx = math.constantIndex(1);
  Value fOne = math.constant(elementType, 1.0);

  // Lower corner of the interpolation cell, and the fractional distance from
  // it, along each spatial axis.
  //
  // The coordinate is sanitized first. `offset` is model data, so a
  // non-finite sampling location is reachable, and an infinity is easy to
  // reach in f16. Casting such a value to an index lowers to `arith.fptosi`,
  // which yields poison for a NaN, for an infinity, and for anything that
  // overflows the index type; the integer clamp further down cannot repair a
  // poison value, so the load would read an arbitrary address. Clamping in
  // float space to [-2, dim + 1] avoids that without changing any finite
  // result: a coordinate outside (-1, dim) already has every interpolation
  // corner out of bounds and contributes zero. NaN needs no separate test
  // because `arith.maxnumf` and `arith.minnumf` return the non-NaN operand,
  // which folds it onto the same path.
  SmallVector<Value, 3> lowerIdx, frac;
  Value fLowLimit = math.constant(elementType, -2.0);
  for (int64_t i = 0; i < spatialRank; ++i) {
    Value fHighLimit =
        math.add(math.cast(elementType, inputSpatialDims[i]), fOne);
    Value coord = math.min(math.max(coords[i], fLowLimit), fHighLimit);
    Value floored = math.floor(coord);
    lowerIdx.emplace_back(math.castToIndex(floored));
    frac.emplace_back(math.sub(coord, floored));
  }

  // Bit i of `corner` selects the upper neighbor along spatial axis i.
  SmallVector<InterpolationCorner, 8> corners;
  for (int64_t corner = 0; corner < (int64_t(1) << spatialRank); ++corner) {
    InterpolationCorner entry;
    entry.weight = fOne;
    for (int64_t i = 0; i < spatialRank; ++i) {
      bool upper = (corner >> i) & 1;
      entry.weight =
          math.mul(entry.weight, upper ? frac[i] : math.sub(fOne, frac[i]));
      Value idx = upper ? math.add(lowerIdx[i], oneIdx) : lowerIdx[i];
      Value dimMax = math.sub(inputSpatialDims[i], oneIdx);
      // Hoisted so that the two comparisons are emitted in a fixed order:
      // as arguments of one call their evaluation order is unspecified, and
      // each one creates an op, so the generated IR would differ by compiler.
      Value geZero = math.sge(idx, zeroIdx);
      Value leMax = math.sle(idx, dimMax);
      Value valid = math.andi(geZero, leMax);
      entry.inBounds =
          entry.inBounds ? math.andi(entry.inBounds, valid) : valid;
      // Clamp so that the load stays in range; out of bounds corners are
      // masked out to zero when the corner is applied. `dimMax` is -1 when the
      // input spatial dim is 0 at runtime, which dynamic shapes allow while
      // padding keeps the output non-empty, so clamp it up to 0 too instead of
      // letting the index go negative. `valid` is false for every corner in
      // that case, so whatever is loaded is discarded.
      Value dimMaxNonNeg = math.max(dimMax, zeroIdx);
      entry.spatialIdx.emplace_back(
          math.min(math.max(idx, zeroIdx), dimMaxNonNeg));
    }
    corners.emplace_back(std::move(entry));
  }
  return corners;
}

// Multilinear (bilinear for 2D) interpolation of `input[n, c, ...]` over the
// corners `prepareMultilinear` built. Following the ONNX spec, samples that
// fall outside the input contribute zero, which matches a GridSample with
// "zeros" padding over the already padded input. This is the only part of a
// sample that depends on the channel `c`.
static Value applyMultilinear(const KrnlBuilder &createKrnl,
    const MathBuilder &math, Value input, Value n, Value c,
    ArrayRef<InterpolationCorner> corners, Type elementType) {
  Value fZero = math.constant(elementType, 0.0);
  Value result = fZero;
  for (const InterpolationCorner &corner : corners) {
    SmallVector<Value, 5> accessFct = {n, c};
    accessFct.append(corner.spatialIdx.begin(), corner.spatialIdx.end());
    Value val = createKrnl.load(input, accessFct);
    val = math.select(corner.inBounds, val, fZero);
    result = math.add(result, math.mul(corner.weight, val));
  }
  return result;
}

// The sampling location for one (kernel position, output position) pair: the
// offset/mask channel it owns, its interpolation corners, and its modulation
// scalar. None of it depends on the input channel.
struct SampleSetup {
  Value maskChan;
  Value maskVal;
  SmallVector<InterpolationCorner, 8> corners;
};

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
    // The offset group of a channel is `ci / (C / offset_group)`. When
    // `group % offset_group == 0` every channel of a conv group lands in the
    // same offset group, so it is `g / (group / offset_group)` and the whole
    // sampling setup becomes invariant in the reduction over input channels.
    // That is worth hoisting, and it covers the usual `offset_group == 1`.
    bool ogIsGroupInvariant = (groupNum % offsetGroupNum) == 0;

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
    Value ciPerOffsetGroupVal;
    if (!ogIsGroupInvariant)
      ciPerOffsetGroupVal = create.krnlIE.getShapeAsSymbol(inputOperand, 1)
                                .floorDiv(LitIE(offsetGroupNum))
                                .getValue();
    Value spatialRankVal = create.math.constantIndex(spatialRank);
    // DeformConv has no auto_pad, so the pads the shape helper computed are
    // plain literals. Reuse them rather than reading the attribute a second
    // time, so the output size and the sampling base cannot drift apart. The
    // helper has also already defaulted strides and dilations to 1.
    SmallVector<int64_t, 3> padsBegin(spatialRank, 0);
    for (int64_t i = 0; i < spatialRank; ++i)
      padsBegin[i] = shapeHelper.pads[i].getLiteral();
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
          Value ogInvariant;
          if (ogIsGroupInvariant)
            ogInvariant = create.math.floorDiv(
                g, create.math.constantIndex(groupNum / offsetGroupNum));

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

                // Reduction over the input channels of the group and
                // over every kernel position. The kernel loops are outermost
                // so that the sampling setup, which depends on the output
                // position and the kernel position but not on the input
                // channel, is computed once per kernel position instead of
                // once per channel. Only the 2^rank loads and the multiply
                // add remain per channel.
                // for k1 = 0 .. K1: ... for kn = 0 .. Kn:
                //   for ciPerG = 0 .. CIPerGroup:
                SmallVector<IndexExpr, 3> kLbs(spatialRank, LitIE(0));
                SmallVector<IndexExpr, 3> kUbs;
                for (int64_t i = 0; i < spatialRank; ++i)
                  kUbs.emplace_back(shapeHelper.kernelShape[i]);
                ValueRange kLoops = create.krnl.defineLoops(spatialRank);
                auto redIterate = create.krnl.iterateIE(kLoops, kLoops, kLbs,
                    kUbs, ValueRange(fZero),
                    [&](const KrnlBuilder &createKrnl, ValueRange kInd,
                        ValueRange kIterArgs) {
                      MultiDialectBuilder<KrnlBuilder, MathBuilder> create(
                          createKrnl);
                      MathBuilder &math = create.math;

                      // Flatten the kernel position. Together with the offset
                      // group it names the offset/mask channel this kernel
                      // element owns.
                      Value kFlat = math.constantIndex(0);
                      for (int64_t i = 0; i < spatialRank; ++i)
                        kFlat = math.add(
                            math.mul(kFlat, kernelDimVals[i]), kInd[i]);

                      // Everything about where this kernel element samples,
                      // given the offset group it belongs to.
                      auto buildSample = [&](const KrnlBuilder &createKrnl,
                                             MathBuilder &math,
                                             Value og) -> SampleSetup {
                        SampleSetup setup;
                        setup.maskChan =
                            math.add(math.mul(og, kernelElemsVal), kFlat);
                        Value offChanBase =
                            math.mul(setup.maskChan, spatialRankVal);
                        SmallVector<Value, 3> coords;
                        for (int64_t i = 0; i < spatialRank; ++i) {
                          // o_i * stride_i + k_i * dilation_i - padBegin_i,
                          // the location a plain convolution would read.
                          // Hoisted for a fixed emission order, see the note
                          // in prepareMultilinear.
                          Value strided = math.mul(outSpatial[i],
                              math.constantIndex(shapeHelper.strides[i]));
                          Value dilated = math.mul(kInd[i],
                              math.constantIndex(shapeHelper.dilations[i]));
                          Value base = math.add(strided, dilated);
                          base =
                              math.sub(base, math.constantIndex(padsBegin[i]));
                          SmallVector<Value, 5> offAccessFct = {
                              n, math.add(offChanBase, math.constantIndex(i))};
                          offAccessFct.append(
                              outSpatial.begin(), outSpatial.end());
                          Value offVal =
                              createKrnl.load(offsetOperand, offAccessFct);
                          coords.emplace_back(
                              math.add(math.cast(elementType, base), offVal));
                        }
                        setup.corners = prepareMultilinear(
                            math, coords, inputSpatialDims, elementType);
                        if (hasMask) {
                          SmallVector<Value, 5> maskAccessFct = {
                              n, setup.maskChan};
                          maskAccessFct.append(
                              outSpatial.begin(), outSpatial.end());
                          setup.maskVal =
                              createKrnl.load(maskOperand, maskAccessFct);
                        }
                        return setup;
                      };

                      SampleSetup hoistedSetup;
                      if (ogIsGroupInvariant)
                        hoistedSetup =
                            buildSample(createKrnl, math, ogInvariant);

                      ValueRange ciLoop = create.krnl.defineLoops(1);
                      auto ciIterate = create.krnl.iterateIE(ciLoop, ciLoop,
                          {LitIE(0)}, {CIPerGroup},
                          ValueRange(kIterArgs.back()),
                          [&](const KrnlBuilder &createKrnl, ValueRange ciInd,
                              ValueRange ciIterArgs) {
                            MultiDialectBuilder<KrnlBuilder, MathBuilder>
                                create(createKrnl);
                            MathBuilder &math = create.math;
                            Value ciPerG = ciInd[0];
                            Value ci = math.add(gTimesCIPerGroup, ciPerG);
                            // On the general path the offset group varies with
                            // the channel, so the setup cannot be hoisted and
                            // is rebuilt here, as it was before.
                            SampleSetup localSetup;
                            if (!ogIsGroupInvariant) {
                              Value og = math.floorDiv(ci, ciPerOffsetGroupVal);
                              localSetup = buildSample(createKrnl, math, og);
                            }
                            const SampleSetup &setup =
                                ogIsGroupInvariant ? hoistedSetup : localSetup;

                            Value sampled =
                                applyMultilinear(createKrnl, math, inputOperand,
                                    n, ci, setup.corners, elementType);
                            SmallVector<Value, 5> filterAccessFct = {
                                co, ciPerG};
                            for (int64_t i = 0; i < spatialRank; ++i)
                              filterAccessFct.emplace_back(kInd[i]);
                            Value filter =
                                createKrnl.load(filterOperand, filterAccessFct);
                            Value prod = math.mul(sampled, filter);
                            if (hasMask)
                              prod = math.mul(prod, setup.maskVal);
                            create.krnl.yield(
                                math.add(ciIterArgs.back(), prod));
                          });
                      create.krnl.yield(ciIterate.getResult(0));
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
