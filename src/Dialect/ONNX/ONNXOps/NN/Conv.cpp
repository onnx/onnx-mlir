/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ Conv.cpp - ONNX Operations ------------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file provides definition of ONNX dialect Conv operation.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

#include "src/Dialect/ONNX/ONNXOps/NN/NNHelper.cpp.inc"

// Currently, QLinearConv has no lit tests. The shape inference was moved to the
// new scheme, but there is no way to ensure it is correct. So I suggest that we
// keep the old code until we get actual tests. Once it is verified, we can
// delete all of the code guarded by this #define.
#define PRESERVE_UNTESTED_OLD_QLINEARCONV_CODE 0

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace {

//===----------------------------------------------------------------------===//
// Support function that computes default values for dilations.
//===----------------------------------------------------------------------===//

template <class T>
LogicalResult processConvDilationParam(T *op, Optional<ArrayAttr> kernelShape) {
  auto builder = Builder(op->getContext());
  auto kernelRank = ArrayAttrSize(kernelShape);

  auto dilationsOpt = op->getDilations();
  if (dilationsOpt.has_value()) {
    if (ArrayAttrSize(dilationsOpt) != kernelRank) {
      return op->emitError("dilation rank is not the same as the spatial rank");
    }
    // Test values to be greater than 0.
    for (decltype(kernelRank) i = 0; i < kernelRank; ++i) {
      if (ArrayAttrIntVal(dilationsOpt, i) < 1) {
        return op->emitError("dilation value must be nonzero positive");
      }
    }
  } else {
    // Default dilatation is needed, all dimensions init with 1.
    SmallVector<int64_t, 4> defaultVals(kernelRank, 1);
    // Convert to ArrayRef, then build attribute, then store attribute.
    ArrayRef<int64_t> defaultRefs(defaultVals);
    op->setDilationsAttr(builder.getI64ArrayAttr(defaultRefs));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Support function that computes default values for kernel_shape.
//===----------------------------------------------------------------------===//

template <class T>
LogicalResult processConvKernelParam(
    T *op, ArrayRef<int64_t> inputShape, ArrayRef<int64_t> weightShape) {
  // Deduce shape from weight input.
  // Number of spatial dimensions.
  if (!op->getKernelShape().has_value()) {
    auto spatialOffset = 2;
    int32_t spatialRank = inputShape.size() - spatialOffset;

    SmallVector<int64_t, 2> defaultVals;
    for (int i = 0; i < spatialRank; ++i)
      defaultVals.emplace_back(weightShape[spatialOffset + i]);
    // Convert to ArrayRef, then build attribute, then store attribute.
    ArrayRef<int64_t> defaultRefs(defaultVals);
    auto builder = Builder(op->getContext());
    op->setKernelShapeAttr(builder.getI64ArrayAttr(defaultRefs));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Support function that computes default values for pads.
//===----------------------------------------------------------------------===//

template <class T>
LogicalResult processConvPadParam(T *op, ArrayRef<int64_t> inputShape,
    Optional<ArrayAttr> kernelShape, Optional<ArrayAttr> stridesOpt,
    Optional<ArrayAttr> dilationsOpt = std::nullopt) {
  auto builder = Builder(op->getContext());

  auto inputRank = inputShape.size();
  auto kernelRank = ArrayAttrSize(kernelShape);
  auto kernelOffset = inputRank - kernelRank;

  // Try to find padding, getting auto_pad attribute first.
  auto autoPad = op->getAutoPad();
  // And then investigate the various different cases. Prefill pad values with
  // zeros, the most common case.
  SmallVector<int64_t, 4> actualPads(2 * kernelRank, 0);
  bool updatedPad = false;
  if (autoPad == "NOTSET") {
    auto padsOpt = op->getPads();
    if (padsOpt.has_value()) {
      // Only option where pads are not updated. Pads consists of two entries
      // for each spatial axis.
      if (ArrayAttrSize(padsOpt) != 2 * kernelRank) {
        return op->emitError("pads rank is not twice the spatial rank");
      }
      // Check values, pads cannot be negative.
      for (decltype(kernelRank) i = 0; i < 2 * kernelRank; ++i) {
        if (ArrayAttrIntVal(padsOpt, i) < 0) {
          return op->emitError("pads value must be nonnegative");
        }
      }
    } else {
      // We have notset with no pads, they are assumed to be all zero.
      updatedPad = true;
    }
  } else if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER") {
    // Reload dilation and strides as they may have gotten default values.
    updatedPad = true;
    int64_t dilationVal = 1;
    for (decltype(kernelRank) i = 0; i < kernelRank; ++i) {
      auto inputSize = inputShape[kernelOffset + i];
      if (inputSize < 0)
        return op->emitError("Conv Pads defined as SAME_UPPER or SAME_LOWER "
                             "requires compile time X sizes");
      auto kernelSize = ArrayAttrIntVal(kernelShape, i);
      if (dilationsOpt.has_value())
        dilationVal = ArrayAttrIntVal(dilationsOpt, i);
      auto strideVal = ArrayAttrIntVal(stridesOpt, i);
      // Output size is input size divided by stride. When stride is 1, then
      // input and output are the same size, which is the usual case. When
      // stride is greater than 1, take the ceil to be sure to have each input
      // value used, as padding will be used to fill the gaps.
      int64_t outputSize = ceil((1.0 * inputSize) / (1.0 * strideVal));
      // Formula is from ONNX MaxPool, and can be explained as follows. Pads
      // is the difference between the needed values for the computations,
      // minus the input values. The needed values for the computation is the
      // effective side of the kernel plus the number of times we jump to the
      // next kernel. Number of time we jump is (outputSize - 1). That number
      // is multiplied with the size of the jump, namely strideVal. Now for
      // the effective kernel size. It is the kernelSize + the number of times
      // we have dilation holes time the dilation. The number of dilation
      // holes is (kernelSize -1). Thus the effective size is "kernelSize +
      // (kernelSize-1)*dilation". This simplifies to "(kernelSize
      // -1)*dilation + 1".
      auto sumOfPad = (outputSize - 1) * strideVal +
                      ((kernelSize - 1) * dilationVal + 1) - inputSize;

      // If filter size for dimension is 1, and dilation for dimension is 1,
      // the above pattern can be negative, in which case the padding should
      // be zero.
      if (sumOfPad < 0)
        sumOfPad = 0;
      // Pad values are assumed equal on both size, at half the total value.
      actualPads[i] = actualPads[kernelRank + i] = sumOfPad / 2;
      // But if the total pad value is odd, we add 1 to beginning or end
      // depending on autoPad value.
      if (sumOfPad % 2 != 0) {
        if (autoPad == "SAME_UPPER") {
          actualPads[kernelRank + i] += 1;
        } else {
          actualPads[i] += 1;
        }
      }
    }
  } else if (autoPad == "VALID") {
    // No pad, default value was set to zero, we are all set.
    updatedPad = true;
  } else {
    return op->emitError("auto_pad of unknown / unsupported value");
  }
  // Set pads values in attributes, if it is needed.
  if (updatedPad) {
    ArrayRef<int64_t> defaultRefs(actualPads);
    op->setPadsAttr(builder.getI64ArrayAttr(defaultRefs));
  }
  // In all cases now, the actual pad values are found in the pads attribute.
  op->setAutoPadAttr(builder.getStringAttr("NOTSET"));
  return success();
}

//===----------------------------------------------------------------------===//
// Support function that computes default values for strides.
//===----------------------------------------------------------------------===//

template <class T>
LogicalResult processConvStrideParam(T *op, Optional<ArrayAttr> kernelShape) {
  auto builder = Builder(op->getContext());
  auto kernelRank = ArrayAttrSize(kernelShape);

  auto stridesOpt = op->getStrides();
  if (stridesOpt.has_value()) {
    if (ArrayAttrSize(stridesOpt) != kernelRank)
      return op->emitError("strides rank is not the same as the spatial rank");
    // Check values to be greater than 0.
    for (decltype(kernelRank) i = 0; i < kernelRank; ++i) {
      if (ArrayAttrIntVal(stridesOpt, i) < 1)
        return op->emitError("strides value must be nonzero positive");
    }
  } else {
    // Default stride is needed, all dimensions init with 1.
    SmallVector<int64_t, 4> defaultVals(kernelRank, 1);
    // Convert to ArrayRef, then build attribute, then store attribute.
    ArrayRef<int64_t> defaultRefs(defaultVals);
    op->setStridesAttr(builder.getI64ArrayAttr(defaultRefs));
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Support function computing default values for dilations, strides,
// kernel_shape and pads.
//===----------------------------------------------------------------------===//

template <class T>
LogicalResult processConvTypeParams(T *op, Value inputOperand, Value W) {
  // 1) Get shape of input. Shape is not guaranteed to be compile time constant.
  auto inputShape = inputOperand.getType().cast<RankedTensorType>().getShape();
  auto wShape = W.getType().cast<RankedTensorType>().getShape();

  // If kernel_shape isn't provided, add kernel_shape to the the op based on the
  // shape of the input and weights.
  LogicalResult res = processConvKernelParam<T>(op, inputShape, wShape);
  if (failed(res))
    return res;

  // 2) Get kernel_shape attribute. They were previously computed. At this time,
  // they are guaranteed to be compile time constant.
  auto kernelShape = op->getKernelShape();

  // Dilation. It is compile time constants (filled to default 1 value if not
  // explicitly given as input).
  res = processConvDilationParam<T>(op, kernelShape);
  if (failed(res))
    return res;
  auto dilationsOpt = op->getDilations();

  // Strides. It is compile time constants (filled to default 1 value if not
  // explicitly given as input).
  res = processConvStrideParam<T>(op, kernelShape);
  if (failed(res))
    return res;
  auto stridesOpt = op->getStrides();

  // Pads.
  return processConvPadParam<T>(
      op, inputShape, kernelShape, stridesOpt, dilationsOpt);
}

#if PRESERVE_UNTESTED_OLD_QLINEARCONV_CODE
//===----------------------------------------------------------------------===//
// Compute spatial dimensions given dilations, strides, pads, and ceil mode.
//===----------------------------------------------------------------------===//

// Helper method for computation of spatial dimensions.

// This method substitutes any uses of dimensions and symbols (e.g.
// dim#0 with dimReplacements[0]) in an affine map, simplifies the modified
// affine map, and returns an integer constant.
static int64_t AffineMapIntConstant(Builder &builder, AffineMap map,
    ArrayRef<int64_t> dimReplacements, ArrayRef<int64_t> symReplacements,
    unsigned numResultDims, unsigned numResultSyms) {
  // Prepare affine expressions.
  SmallVector<AffineExpr, 4> dimExprs, symExprs;
  for (int64_t dim : dimReplacements) {
    AffineExpr exp = builder.getAffineConstantExpr(dim);
    dimExprs.emplace_back(exp);
  }
  for (int64_t sym : symReplacements) {
    AffineExpr exp = builder.getAffineConstantExpr(sym);
    symExprs.emplace_back(exp);
  }
  // Replace all the affine map's arguments with real values and evaluate the
  // map.
  AffineMap replacedDimMap = map.replaceDimsAndSymbols(
      dimExprs, symExprs, numResultDims, numResultSyms);
  AffineMap simplifiedMap = simplifyAffineMap(replacedDimMap);
  return simplifiedMap.getSingleConstantResult();
}

static void insertConvSpatialDim(SmallVector<int64_t, 4> *outputDims,
    Builder &builder, ArrayRef<int64_t> xShape, Optional<ArrayAttr> kernelShape,
    Optional<ArrayAttr> padsOpt, Optional<ArrayAttr> stridesOpt,
    Optional<ArrayAttr> dilationsOpt = std::nullopt, bool ceilMode = false) {
  auto spatialRank = ArrayAttrSize(kernelShape);
  auto spatialOffset = xShape.size() - spatialRank;

  // Get an affine map to compute the output dimension.
  AffineMap dimMap = getConvDimMap(builder, ceilMode);
  for (unsigned int i = 0; i < spatialRank; ++i) {
    int64_t res = ShapedType::kDynamic;
    if (!ShapedType::isDynamic(xShape[spatialOffset + i])) {
      auto inputSize = xShape[spatialOffset + i];
      auto kernelSize = ArrayAttrIntVal(kernelShape, i);
      auto sumOfPads = ArrayAttrIntVal(padsOpt, i) +
                       ArrayAttrIntVal(padsOpt, spatialRank + i);
      auto strideVal = ArrayAttrIntVal(stridesOpt, i);
      int64_t dilationVal = 1;
      if (dilationsOpt.has_value())
        dilationVal = ArrayAttrIntVal(dilationsOpt, i);
      res = AffineMapIntConstant(builder, dimMap, {inputSize},
          {kernelSize, sumOfPads, strideVal, dilationVal}, 1, 4);
    }
    outputDims->emplace_back(res);
  }
}
#endif // PRESERVE_UNTESTED_OLD_QLINEARCONV_CODE

} // namespace

namespace onnx_mlir {

template <>
LogicalResult ONNXConvOpShapeHelper::computeShape() {
  ONNXConvOp poolOp = llvm::cast<ONNXConvOp>(op);
  ONNXConvOpAdaptor operandAdaptor = ONNXConvOpAdaptor(operands);
  return customComputeShape(operandAdaptor.getX(), operandAdaptor.getW(),
      poolOp.getKernelShape(), poolOp.getAutoPad(), poolOp.getPads(),
      poolOp.getStrides(), poolOp.getDilations(), /*hasFilter*/ true,
      /*ceil mode*/ false);
}

template <>
LogicalResult ONNXConvIntegerOpShapeHelper::computeShape() {
  ONNXConvIntegerOp poolOp = llvm::cast<ONNXConvIntegerOp>(op);
  ONNXConvIntegerOpAdaptor operandAdaptor = ONNXConvIntegerOpAdaptor(operands);
  return customComputeShape(operandAdaptor.getX(), operandAdaptor.getW(),
      poolOp.getKernelShape(), poolOp.getAutoPad(), poolOp.getPads(),
      poolOp.getStrides(), poolOp.getDilations(), /*hasFilter*/ true,
      /*ceil mode*/ false);
}

template <>
LogicalResult ONNXQLinearConvOpShapeHelper::computeShape() {
  ONNXQLinearConvOp poolOp = llvm::cast<ONNXQLinearConvOp>(op);
  ONNXQLinearConvOpAdaptor operandAdaptor = ONNXQLinearConvOpAdaptor(operands);
  return customComputeShape(operandAdaptor.getX(), operandAdaptor.getW(),
      poolOp.getKernelShape(), poolOp.getAutoPad(), poolOp.getPads(),
      poolOp.getStrides(), poolOp.getDilations(), /*hasFilter*/ true,
      /*ceil mode*/ false);
}

LogicalResult ONNXConvTransposeOpShapeHelper::computeShape() {
  ONNXConvTransposeOp convTransposeOp = llvm::cast<ONNXConvTransposeOp>(op);
  ONNXConvTransposeOpAdaptor operandAdaptor(operands);
  Optional<ArrayAttr> kernelShapeOpt = convTransposeOp.getKernelShape();
  Optional<ArrayAttr> padOpt = convTransposeOp.getPads();
  Optional<ArrayAttr> strideOpt = convTransposeOp.getStrides();
  Optional<ArrayAttr> dilationOpt = convTransposeOp.getDilations();
  Optional<ArrayAttr> outputPaddingOpt = convTransposeOp.getOutputPadding();
  Optional<ArrayAttr> outputShapeOpt = convTransposeOp.getOutputShape();
  int64_t groupNum = convTransposeOp.getGroup();
  llvm::StringRef autoPad = convTransposeOp.getAutoPad();

  Value xValue = (Value)operandAdaptor.getX();
  Value wValue = operandAdaptor.getW();

  // Basic information.
  int64_t rank = createIE->getShapedTypeRank(xValue);
  int64_t spatialOffset = 2;
  int64_t spatialRank = rank - spatialOffset;

  // Fill the stride, dilation, kernel.
  for (int i = 0; i < spatialRank; ++i) {
    // Strides, default 1.
    strides.emplace_back(
        strideOpt.has_value() ? ArrayAttrIntVal(strideOpt, i) : 1);
    // Dilations, default 1.
    dilations.emplace_back(
        dilationOpt.has_value() ? ArrayAttrIntVal(dilationOpt, i) : 1);
    // Kernel shape from attribute, default from Weight's spatial dims.
    if (kernelShapeOpt.has_value()) {
      kernelShape.emplace_back(
          LiteralIndexExpr(ArrayAttrIntVal(kernelShapeOpt, i)));
    } else {
      int ii = i + spatialOffset;
      kernelShape.emplace_back(createIE->getShapeAsSymbol(wValue, ii));
    }
    // Output Padding, default 0.
    outputPadding.emplace_back(outputPaddingOpt.has_value()
                                   ? ArrayAttrIntVal(outputPaddingOpt, i)
                                   : 0);
  }
  // Pads, at this stage a given compile-time literal or default 0.
  for (int i = 0; i < 2 * spatialRank; ++i) {
    int64_t p = padOpt.has_value() ? ArrayAttrIntVal(padOpt, i) : 0;
    pads.emplace_back(LiteralIndexExpr(p));
  }

  // Handle output size: start by inserting batch size and output channels.
  DimsExpr outputDims;
  outputDims.emplace_back(createIE->getShapeAsDim(xValue, 0));
  outputDims.emplace_back(
      createIE->getShapeAsDim(wValue, 1) *
      LiteralIndexExpr(groupNum)); // CO may be different from CI.

  LiteralIndexExpr zeroIE(0);
  LiteralIndexExpr oneIE(1);
  for (int i = 0; i < spatialRank; ++i) {
    int64_t ii = i + spatialOffset;
    IndexExpr I = createIE->getShapeAsDim(xValue, ii);
    IndexExpr K = kernelShape[i];
    LiteralIndexExpr d(dilations[i]);
    LiteralIndexExpr s(strides[i]);
    LiteralIndexExpr outPad(outputPadding[i]);

    IndexExpr t0 = K - oneIE;
    IndexExpr kdTerm = t0 * d + oneIE; // (k - 1) * d + 1
    IndexExpr t1 = I - oneIE;
    if (outputShapeOpt.has_value()) {
      // Set output dim
      LiteralIndexExpr O(ArrayAttrIntVal(outputShapeOpt, i));
      outputDims.emplace_back(O);
      // Set pads
      // P = max(0, s * (I - 1) + outPad + ((K - 1) * d + 1) - O);
      IndexExpr pSum = IndexExpr::max(zeroIE, s * t1 + outPad + kdTerm - O);
      IndexExpr pSmall = pSum.floorDiv(2);
      if (autoPad == "SAME_UPPER") {
        pads[i] = pSmall;
        pads[i + spatialRank] = pSum - pSmall;
      } else if (autoPad == "NOTSET" || autoPad == "VALID" ||
                 autoPad == "SAME_LOWER") {
        pads[i] = pSum - pSmall;
        pads[i + spatialRank] = pSmall;
      } else {
        return op->emitError("auto_pad of unknown/unsupported value");
      }
    } else {
      // Set pads
      IndexExpr pSum;
      if (autoPad == "NOTSET") {
        pSum = pads[i] + pads[i + spatialRank]; // Sum both pads.
        // pads already set, nothing more to do.
      } else if (autoPad == "VALID") {
        pSum = zeroIE;
        pads[i] = zeroIE;
        pads[i + spatialRank] = zeroIE;
      } else if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER") {
        return op->emitError(
            "auto_pad of SAME_UPPER/SAME_LOWER not unsupported yet");
      } else {
        return op->emitError("auto_pad of unknown/unsupported value");
      }
      // Set output dim
      // O = s * (I - 1) + outPad + ((K - 1) * d + 1) - P
      IndexExpr O = s * t1 + outPad + kdTerm - pSum;
      outputDims.emplace_back(O); // Set output dim
    }
  }
  // Save the final result.
  setOutputDims(outputDims);

  dimsNoOutputPadding.emplace_back(outputDims[0]);
  dimsNoOutputPadding.emplace_back(outputDims[1]);
  for (int i = 0; i < spatialRank; ++i) {
    LiteralIndexExpr outPad(outputPadding[i]);
    IndexExpr dimNoOutPad =
        IndexExpr::max(zeroIE, outputDims[i + spatialOffset] - outPad);
    dimsNoOutputPadding.emplace_back(dimNoOutPad);
  }
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Conv
//===----------------------------------------------------------------------===//

LogicalResult ONNXConvOp::verify() {
  ONNXConvOpAdaptor operandAdaptor = ONNXConvOpAdaptor(*this);
  // Get operands.
  auto X = operandAdaptor.getX();
  auto W = operandAdaptor.getW();
  auto B = operandAdaptor.getB();
  bool hasBias = !isFromNone(B);
  int64_t g = getGroup();
  if (g < 1)
    return emitOpError("group must be strictly positive");
  // Get spatial rank.
  if (!hasShapeAndRank(W)) {
    // Won't be able to do any checking at this stage.
    return success();
  }
  auto wShape = W.getType().cast<ShapedType>().getShape();
  int64_t spatialRank = wShape.size() - 2;
  // If ranked, verify ranks of inputs.
  if (spatialRank < 1)
    return emitOpError("Spatial rank must be strictly positive");

  if (wShape[0] != ShapedType::kDynamic && wShape[0] % g != 0) {
    // This rule is not enforced in the spec but is present in Keras,
    // Pytorch, and simplifies the code.
    // Note: Pytorch requires both channel in (CI) and channel out (CO) to be
    // multiple of group number (G).
    // https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    // ONNX clearly states that C (channel in or CI here) is a multiple of group
    // number (G).
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv
    // Quote: X.shape[1] == (W.shape[1] * group) == C
    // Keras also specifies it: Input channels and filters must both be
    // divisible by groups.
    // https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
    return emitOpError(
        "Channel Out (M) must be a multiple of the number of groups");
  }
  if (hasShapeAndRank(X)) {
    auto xShape = X.getType().cast<ShapedType>().getShape();
    if ((int64_t)xShape.size() - 2 != spatialRank)
      return emitOpError("Input and filter rank mismatch");
    if (xShape[1] != ShapedType::kDynamic && xShape[1] % g != 0)
      return emitOpError(
          "Channel In (C) must be a multiple of the number of groups");
    if (xShape[1] != ShapedType::kDynamic &&
        wShape[1] != ShapedType::kDynamic && xShape[1] != wShape[1] * g) {
      return emitOpError("Channel In (C) of input must be equal 2nd dim "
                         "of weights times g");
    }
  }
  if (hasBias && hasShapeAndRank(B)) {
    auto bShape = B.getType().cast<ShapedType>().getShape();
    if (bShape.size() != 1)
      return emitOpError("Bias should have a rank of one");
    if (bShape[0] != ShapedType::kDynamic &&
        wShape[0] != ShapedType::kDynamic && wShape[0] != bShape[0])
      return emitOpError(
          "Bias should have same dimension as first dimension of weights");
  }
  // Verify parameters.
  if (failed(verifyKernelShape<ONNXConvOp>(
          this, W, getKernelShape(), spatialRank)))
    return failure();
  if (failed(verifyStrides<ONNXConvOp>(this, spatialRank)))
    return failure();
  if (failed(verifyDilations<ONNXConvOp>(this, spatialRank)))
    return failure();
  if (failed(verifyPadding<ONNXConvOp>(this, spatialRank)))
    return failure();
  return success();
}

// For this operation, we define the attributes once in the original Conv
// operation class. There is no need to redefine the attribute names for the
// other classes based on Conv.
// Conv attributes output: no changes to the op but the output.
// ShapeHelper get
//   -  dilations, strides: set to 1 if not defined by user;
//   -  kernelShape: inferred from weight matrix if not defined by user;
//   -  pads: set to proper value

LogicalResult ONNXConvOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Generic shape for data input X, weight tensor W, and optional bias B
  // X: (N x C x D1 x D2 ... x Dn)
  // W: (M x C/group x k1 x k2 x ... x kn)
  // B: (M) Optional

  // Cannot infer shape if no shape exists.
  bool hasBias = !isFromNone(getB());
  if (!hasShapeAndRank(getX()) || !hasShapeAndRank(getW()) ||
      (hasBias && !hasShapeAndRank(getB())))
    return success();

  Type elementType = getX().getType().cast<ShapedType>().getElementType();
  ONNXConvOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

//===----------------------------------------------------------------------===//
// ConvTranspose
//===----------------------------------------------------------------------===//

LogicalResult ONNXConvTransposeOp::verify() {
  ONNXConvTransposeOpAdaptor operandAdaptor = ONNXConvTransposeOpAdaptor(*this);
  // Get operands.
  auto X = operandAdaptor.getX();
  auto W = operandAdaptor.getW();
  auto B = operandAdaptor.getB();
  bool hasBias = !B.getType().isa<NoneType>();
  int64_t g = getGroup();
  if (g < 1)
    return emitOpError("group must be strictly positive");
  // Get spatial rank.
  if (!hasShapeAndRank(W)) {
    // Won't be able to do any checking at this stage.
    return success();
  }
  auto wShape = W.getType().cast<ShapedType>().getShape();
  int64_t spatialRank = wShape.size() - 2;
  // If ranked, verify ranks of inputs.
  if (spatialRank < 1)
    return emitOpError("Spatial rank must be strictly positive");

  if (hasShapeAndRank(X)) {
    auto xShape = X.getType().cast<ShapedType>().getShape();
    if ((int64_t)xShape.size() - 2 != spatialRank)
      return emitOpError("Input and filter rank mismatch");
    if (xShape[1] != ShapedType::kDynamic &&
        wShape[0] != ShapedType::kDynamic && xShape[1] != wShape[0]) {
      return emitOpError("Channel In (C) of input must be equal 1st dim "
                         "of weights");
    }
  }
  if (hasBias && hasShapeAndRank(B)) {
    auto bShape = B.getType().cast<ShapedType>().getShape();
    if (bShape.size() != 1)
      return emitOpError("Bias should have a rank of one");
    if (bShape[0] != ShapedType::kDynamic &&
        wShape[1] != ShapedType::kDynamic && bShape[0] != wShape[1] * g)
      return emitOpError("Bias should have same dimension as second dimension "
                         "of weights times g");
  }
  // Verify parameters.
  if (failed(verifyKernelShape<ONNXConvTransposeOp>(
          this, W, getKernelShape(), spatialRank)))
    return failure();
  if (failed(verifyStrides<ONNXConvTransposeOp>(this, spatialRank)))
    return failure();
  if (failed(verifyDilations<ONNXConvTransposeOp>(this, spatialRank)))
    return failure();
  if (failed(verifyPadding<ONNXConvTransposeOp>(this, spatialRank)))
    return failure();
  if (failed(verifyOutputShape<ONNXConvTransposeOp>(this, spatialRank)))
    return failure();
  return success();
}

static void insertConvTransposeSpatialDim(SmallVectorImpl<int64_t> &outputDims,
    ArrayRef<int64_t> xShape, Optional<ArrayAttr> kernelShape,
    Optional<ArrayAttr> padsOpt, Optional<ArrayAttr> stridesOpt,
    Optional<ArrayAttr> outputPadsOpt,
    Optional<ArrayAttr> dilationsOpt = std::nullopt, bool ceilMode = false) {
  auto xRank = xShape.size();
  auto spatialRank = ArrayAttrSize(kernelShape);
  auto spatialOffset = xRank - spatialRank;

  int64_t dilationVal = 1;
  int64_t outputPadsVal = 0;
  // output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] +
  // ((kernel_shape[i] - 1) * dilations[i] + 1) - pads[start_i] - pads[end_i]
  for (unsigned int i = 0; i < spatialRank; ++i) {
    auto inputSize = xShape[spatialOffset + i];
    auto sumOfPads =
        ArrayAttrIntVal(padsOpt, i) + ArrayAttrIntVal(padsOpt, spatialRank + i);
    auto kernelSize = ArrayAttrIntVal(kernelShape, i);
    if (dilationsOpt.has_value())
      dilationVal = ArrayAttrIntVal(dilationsOpt, i);
    auto strideVal = ArrayAttrIntVal(stridesOpt, i);
    if (outputPadsOpt.has_value())
      outputPadsVal = ArrayAttrIntVal(outputPadsOpt, i);
    // Number of useful values: input plus pad - effective size of kernel (see
    // processConvTypeParams comments to see how this value is derived).
    int64_t res = strideVal * (inputSize - 1) + outputPadsVal +
                  ((kernelSize - 1) * dilationVal + 1) - sumOfPads;
    outputDims.emplace_back(res);
  }
}

static void insertConvTransposePads(SmallVectorImpl<int64_t> &inferredPads,
    llvm::StringRef autoPad, ArrayRef<int64_t> xShape,
    Optional<ArrayAttr> kernelShape, Optional<ArrayAttr> padsOpt,
    Optional<ArrayAttr> stridesOpt, Optional<ArrayAttr> outputPadsOpt,
    Optional<ArrayAttr> outputShapeOpt,
    Optional<ArrayAttr> dilationsOpt = std::nullopt, bool ceilMode = false) {
  auto xRank = xShape.size();
  auto spatialRank = ArrayAttrSize(kernelShape);
  auto spatialOffset = xRank - spatialRank;

  int64_t dilationVal = 1;
  int64_t outputPadsVal = 0;
  // total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] +
  // ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i]
  // If (auto_pads == SAME_UPPER): pads[start_i] = total_padding[i]/2;
  // pads[end_i] = total_padding[i] - (total_padding[i]/2)
  // Else: pads[start_i] = total_padding[i] - (total_padding[i]/2);
  // pads[end_i] = (total_padding[i]/2)
  inferredPads.resize(spatialRank * 2);
  for (unsigned int i = 0; i < spatialRank; ++i) {
    auto inputSize = xShape[spatialOffset + i];
    auto outputSize = ArrayAttrIntVal(outputShapeOpt, i);
    auto kernelSize = ArrayAttrIntVal(kernelShape, i);
    if (dilationsOpt.has_value())
      dilationVal = ArrayAttrIntVal(dilationsOpt, i);
    auto strideVal = ArrayAttrIntVal(stridesOpt, i);
    if (outputPadsOpt.has_value())
      outputPadsVal = ArrayAttrIntVal(outputPadsOpt, i);
    int64_t totalPadding = strideVal * (inputSize - 1) + outputPadsVal +
                           ((kernelSize - 1) * dilationVal + 1) - outputSize;
    int64_t beginPad = totalPadding / 2;
    int64_t endPad = totalPadding - beginPad;
    if (autoPad != "SAME_UPPER") {
      std::swap(beginPad, endPad);
    }
    inferredPads[i] = beginPad;
    inferredPads[spatialRank + i] = endPad;
  }
}

// For this operation, we define the attributes once in the original Conv
// operation class. There is no need to redefine the attribute names for the
// other classes based on Conv.
// Conv attributes output:
//   -  auto_pad set to NOTSET;
//   -  dilations, strides: set to 1 if not defined by user;
//   -  kernelShape: inferred from weight matrix if not defined by user;
//   -  pads: set to proper value, 0 if not defined by user.

LogicalResult ONNXConvTransposeOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Generic shape for data input X, weight tensor W, and optional bias B
  // X: (N x C x D1 x D2 ... x Dn)
  // W: (C x M/group x k1 x k2 x ... x kn)
  // B: (M) Optional

  bool hasBias = !isFromNone(getB());

  // Cannot infer shape if no shape exists.
  if (!hasShapeAndRank(getX()) || !hasShapeAndRank(getW()) ||
      (hasBias && !hasShapeAndRank(getB()))) {
    return success();
  }

  auto xTy = getX().getType().cast<RankedTensorType>();
  auto xShape = xTy.getShape();
  auto weightTy = getW().getType().cast<RankedTensorType>();
  auto weightShape = weightTy.getShape();
  auto builder = Builder(this->getContext());

  // Lowest supported convolution is a one dimensional convolution.
  if (xShape.size() < 3) {
    return emitError("Data input shape must be at least (NxCxD1)");
  }

  // Check that shape of weight and data have same length.
  if (xShape.size() != weightShape.size()) {
    return emitError("Weight size not compatible with data size");
  }

  // Group is a required attribute and should have default value of 1.
  int64_t group = ONNXConvTransposeOp::getGroup();

  // Check if the attribute actually exists. If it does not then add it.
  if (!getGroupAttr())
    setGroupAttr(IntegerAttr::get(builder.getIntegerType(64, /*isSigned=*/true),
        APInt(64, group, /*isSigned=*/true)));

  int64_t inChannels = weightShape[0];
  int64_t outChannels = weightShape[1] * group;

  // Check that the X.shape[1] == W.shape[0] == C && X.shape[1] % group == 0
  // condition holds.
  if (!ShapedType::isDynamic(xShape[1]) && !ShapedType::isDynamic(inChannels) &&
      xShape[1] != inChannels && xShape[1] % group != 0) {
    return emitOpError("Channel dimension mismatch")
           << xTy << " " << weightTy << " " << group;
  }

  // Check the size of bias.
  if (hasBias) {
    auto bTx = getB().getType().cast<RankedTensorType>();
    auto bShape = bTx.getShape();
    if (bShape.size() != 1) {
      return emitError("bias should be one dimensional");
    }
    if (bShape[0] != outChannels) {
      return emitError(
          "bias should have same dimensions as number of output channels");
    }
  }

  // Note: the value of the group attribute only impacts the way the
  // computation is carried out and not the actual output size.

  // Number of spatial dimensions.
  auto spatialOffset = 2;
  int32_t spatialRank = xShape.size() - spatialOffset;

  // Use kernel_shape attribute if present otherwise use size from weight
  // argument.
  auto kernelShape = getKernelShape();
  if (kernelShape.has_value()) {
    if ((int32_t)ArrayAttrSize(kernelShape) != spatialRank) {
      return emitError(
          "kernel_shape length incompatible with spatial dimensions");
    }
    // Have the right number of values, check them.
    for (int i = 0; i < spatialRank; ++i)
      if (ArrayAttrIntVal(kernelShape, i) < 1) {
        return emitError("bad kernel_shape value");
      }
  }

  // Process strides, dilations, kernel_shape and pads.
  LogicalResult res =
      processConvTypeParams<ONNXConvTransposeOp>(this, getX(), getW());
  assert(succeeded(res));
  kernelShape = getKernelShape();

  auto dilationsOpt = getDilations();
  auto stridesOpt = getStrides();
  auto padsOpt = getPads();
  auto outputPads = getOutputPadding();

  if (getOutputShape().has_value()) {
    SmallVector<int64_t, 4> inferredPads;
    // Determine padding values based on output shape.
    auto outputShape = getOutputShape();
    auto autoPad = getAutoPad();
    insertConvTransposePads(inferredPads, autoPad, xShape, kernelShape, padsOpt,
        stridesOpt, outputPads, outputShape, dilationsOpt);
    setPadsAttr(builder.getI64ArrayAttr(inferredPads));
  } else {
    llvm::SmallVector<int64_t, 4> outputShapeSpatial;
    insertConvTransposeSpatialDim(outputShapeSpatial, xShape, kernelShape,
        padsOpt, stridesOpt, outputPads, dilationsOpt);
    setOutputShapeAttr(builder.getI64ArrayAttr(outputShapeSpatial));
  }
  auto outputShape = getOutputShape(); // spatial dimensions

  llvm::SmallVector<int64_t, 4> outputShapeFinal;
  // First two output dimensions consist of the number of batches and the
  // number of kernels being applied.
  // Insert batch size.
  outputShapeFinal.emplace_back(xShape[0]);
  // Insert number of filters being applied (number of output channels *
  // groups).
  outputShapeFinal.emplace_back(outChannels);
  // Insert spatial dims.
  for (int32_t i = 0; i < spatialRank; ++i) {
    outputShapeFinal.emplace_back(ArrayAttrIntVal(outputShape, i));
  }
  getResult().setType(
      RankedTensorType::get(outputShapeFinal, xTy.getElementType()));

  return success();
}

//===----------------------------------------------------------------------===//
// QLinearConv
//===----------------------------------------------------------------------===//

// Enable this one default; but keep the else code as there is currently no lit
// test of any kind to very functionality.

LogicalResult ONNXQLinearConvOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Generic shape for data input X, weight tensor W, and optional bias B
  // X: (N x C x D1 x D2 ... x Dn)
  // W: (M x C/group x k1 x k2 x ... x kn)
  // B: (M) Optional

  // Cannot infer shape if no shape exists.
  bool hasBias = !isFromNone(getB());
  if (!hasShapeAndRank(getX()) || !hasShapeAndRank(getW()) ||
      (hasBias && !hasShapeAndRank(getB())))
    return success();

  Type elementType = getX().getType().cast<ShapedType>().getElementType();
  ONNXQLinearConvOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(elementType);
}

#if PRESERVE_UNTESTED_OLD_QLINEARCONV_CODE
LogicalResult ONNXQLinearConvOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Generic shape for data input X, weight tensor W, and optional bias B
  // X: (N x C x D1 x D2 ... x Dn)
  // W: (M x C/group x k1 x k2 x ... x kn)
  // B: (M) Optional

  bool hasBias = !isFromNone(B());

  // Cannot infer shape if no shape exists.
  if (!getX().getType().isa<RankedTensorType>() ||
      !getW().getType().isa<RankedTensorType>() ||
      (hasBias && !getB().getType().isa<RankedTensorType>()))
    return success();

  auto xTy = getX().getType().cast<RankedTensorType>();
  auto xShape = xTy.getShape();
  auto weightTy = getW().getType().cast<RankedTensorType>();
  auto weightShape = weightTy.getShape();
  auto builder = Builder(this->getContext());

  // Lowest supported convolution is a one dimensional convolution.
  if (xShape.size() < 3)
    return emitError("Data input shape must be at least (NxCxD1)");

  // Check that shape of weight and data have same length.
  if (xShape.size() != weightShape.size())
    return emitError("Weight size not compatible with data size");

  // Group is a required attribute and should have default value of 1.
  int64_t group = ONNXQLinearConvOp::getGroup();

  // Check if the attribute actually exists. If it does not then add it.
  if (!getGroupAttr())
    setGroupAttr(builder.getI64IntegerAttr(group));

  // Check that the X.shape[1] == (W.shape[1] * group) == C condition holds.
  if (!ShapedType::isDynamic(xShape[1]) &&
      !ShapedType::isDynamic(weightShape[1]) &&
      xShape[1] != (weightShape[1] * group))
    return emitError("Channel dimension mismatch");

  // Check the size of bias.
  if (hasBias) {
    auto bTx = getB().getType().cast<RankedTensorType>();
    auto bShape = bTx.getShape();
    if (bShape.size() != 1)
      return emitError("bias should be one dimensional");
    if (bShape[0] != weightShape[0])
      return emitError("bias should have same dimensions "
                       "as weight's first dimension");
  }

  // Note: the value of the group attribute only impacts the way the
  // computation is carried out and not the actual output size.

  // Number of spatial dimensions.
  auto spatialOffset = 2;
  int32_t spatialRank = xShape.size() - spatialOffset;

  // Use kernel_shape attribute if present otherwise use size from weight
  // argument.
  auto kernelShape = getKernelShape();
  if (kernelShape.has_value()) {
    if ((int32_t)ArrayAttrSize(kernelShape) != spatialRank)
      return emitError(
          "kernel_shape length incompatible with spatial dimensions");
    // Have the right number of values, check them.
    for (int i = 0; i < spatialRank; ++i)
      if (ArrayAttrIntVal(kernelShape, i) < 1)
        return emitError("bad kernel_shape value");
  }

  // Process strides, dilations, kernel_shape and pads.
  LogicalResult res =
      processConvTypeParams<ONNXQLinearConvOp>(this, getX(), getW());
  assert(succeeded(res));
  kernelShape = getKernelShape();

  auto dilationsOpt = getDilations();
  auto stridesOpt = getStrides();
  auto padsOpt = getPads();

  // First two output dimensions consist of the number of batches and the
  // number of kernels being applied.
  SmallVector<int64_t, 4> outputDims;
  // Insert batch size.
  outputDims.emplace_back(xShape[0]);
  // Insert number of filters being applied (number of output channels).
  outputDims.emplace_back(weightShape[0]);
  // Compute and insert spatial dims.
  insertConvSpatialDim(&outputDims, builder, xShape, kernelShape, padsOpt,
      stridesOpt, dilationsOpt);

  getResult().setType(RankedTensorType::get(outputDims, xTy.getElementType()));
  return success();
}
#endif // PRESERVE_UNTESTED_OLD_QLINEARCONV_CODE

//===----------------------------------------------------------------------===//
// ConvInteger - copied almost exactly from Conv (X -> x, W -> w, no bias)
//===----------------------------------------------------------------------===//

LogicalResult ONNXConvIntegerOp::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  // Generic shape for data input X, weight tensor W, and optional bias B
  // X: (N x C x D1 x D2 ... x Dn)
  // W: (M x C/group x k1 x k2 x ... x kn)
  // B: (M) Optional

  // Cannot infer shape if no shape exists.
  if (!hasShapeAndRank(getX()) || !hasShapeAndRank(getW()))
    return success();

  Type outputElementType = IntegerType::get(getContext(), 32);
  ONNXConvIntegerOpShapeHelper shapeHelper(getOperation(), {});
  return shapeHelper.computeShapeAndUpdateType(outputElementType);
}

//===----------------------------------------------------------------------===//
// Template instantiation; keep at the end of the file.
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template struct ONNXGenericPoolOpShapeHelper<ONNXConvOp>;
template struct ONNXGenericPoolOpShapeHelper<ONNXConvIntegerOp>;
template struct ONNXGenericPoolOpShapeHelper<ONNXQLinearConvOp>;

} // namespace onnx_mlir
