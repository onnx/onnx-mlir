/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ NNHelper.cpp.inc - ONNX Operations ----------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file provides specific template code for Conv and Pool operations.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace mlir::OpTrait::util;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// Generic pool shape computation
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

template <typename OP_TYPE>
LogicalResult ONNXGenericPoolOpShapeHelper<OP_TYPE>::customComputeShape(
    Value xValue, Value wValue, std::optional<ArrayAttr> kernelShapeOpt,
    llvm::StringRef autoPad, std::optional<ArrayAttr> padOpt,
    std::optional<ArrayAttr> strideOpt, std::optional<ArrayAttr> dilationOpt,
    bool hasFilter, bool ceilMode) {
  // Basic information.
  if(!hasShapeAndRank(xValue)) {
    return failure();
  }
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
      kernelShape.emplace_back(LitIE(ArrayAttrIntVal(kernelShapeOpt, i)));
    } else {
      assert(hasFilter && "no kernel shape and no filter: unkown kernel shape");
      int ii = i + spatialOffset;
      kernelShape.emplace_back(createIE->getShapeAsSymbol(wValue, ii));
    }
  }
  // Pads, at this stage a given compile-time literal or default 0.
  for (int i = 0; i < 2 * spatialRank; ++i) {
    int64_t p = padOpt.has_value() ? ArrayAttrIntVal(padOpt, i) : 0;
    pads.emplace_back(LitIE(p));
  }

  // Handle output size: start by inserting batch size and output channels.
  DimsExpr outputDims;
  outputDims.emplace_back(createIE->getShapeAsDim(xValue, 0));
  if (hasFilter)
    // CO may be different from CI.
    outputDims.emplace_back(createIE->getShapeAsDim(wValue, 0));
  else
    // CO is CI.
    outputDims.emplace_back(createIE->getShapeAsDim(xValue, 1));

  // Insert dimensions for the spatial axes. From MaxPool:
  // https://github.com/onnx/onnx/blob/main/docs/Operators.md#maxpool
  //
  // NOSET:
  //  * O[i] = floor((I[i] + P[i] - ((K[i] - 1) * d[i] + 1)) / s[i] + 1)
  // VALID:
  // * O[i] = floor((I[i] - {(K[i] - 1) * d[i] + 1} + 1) / s[i])
  // * P = 0
  // SAME_LOWER or SAME_UPPER:
  // * O[i] = ceil(I[i] / s[i])
  // * p' = (O[i] - 1) * s[i] + ((K[i] - 1) * d[i] + 1) - I[i]
  // * P[i] = p' / 2, if odd, first or second are increased by one.
  LiteralIndexExpr zero(0);
  LiteralIndexExpr one(1);
  for (int64_t i = 0; i < spatialRank; ++i) {
    int64_t ii = i + spatialOffset;
    IndexExpr I = createIE->getShapeAsDim(xValue, ii);
    IndexExpr K = kernelShape[i];
    LiteralIndexExpr d(dilations[i]);
    LiteralIndexExpr s(strides[i]);
    IndexExpr t1 = K - one;
    IndexExpr kdTerm = t1 * d + one; // (k - 1) * d + 1
    if (autoPad == "NOTSET") {
      IndexExpr p = pads[i] + pads[i + spatialRank]; // Sum both pads.
      IndexExpr t1 = I + p; // Compute floor/ceil((I + p - kdTerm) / s) + 1.
      IndexExpr t2 = t1 - kdTerm;
      IndexExpr O;
      if (ceilMode)
        O = t2.ceilDiv(s);
      else
        O = t2.floorDiv(s);
      O = O + one;
      // Set output dim, and pads already set, nothing more to do.
      outputDims.emplace_back(O);
    } else if (autoPad == "VALID") {
      IndexExpr t1 = I - kdTerm; // Compute ceil((I - kdTerm +1)/s).
      IndexExpr t2 = t1 + one;
      IndexExpr O = t2.ceilDiv(s);
      // Set output dim, and pads already set to zero, nothing more to do.
      outputDims.emplace_back(O);
    } else if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER") {
      // Compute output as O = ceil(I/s).
      IndexExpr O = I.ceilDiv(s);
      outputDims.emplace_back(O);
      // Compute sum of pads padSum = (O -1)*s + kdTerm - I.
      IndexExpr t1 = O - one;
      IndexExpr t2 = t1 * s + kdTerm;
      IndexExpr t3 = t2 - I;
      IndexExpr padSum = IndexExpr::max(t3, zero);
      // Single pad value is padSump / 2.
      IndexExpr p = padSum.floorDiv(2);
      // Increment is 1 when pp % 2 != 0
      IndexExpr test = (padSum % 2) != zero;
      IndexExpr inc = IndexExpr::select(test, one, zero);
      // Increment 1st value for SAME_LOWER and 2nd for SAME_UPPER.
      if (autoPad == "SAME_UPPER") {
        pads[i] = p;
        pads[i + spatialRank] = p + inc;
      } else { // SAME_LOWER.
        pads[i] = p + inc;
        pads[i + spatialRank] = p;
      }
    }
  }

#if DEBUG
  if (outputDims.size() == 4) {
    cerr << "2d conv const params";
    if (outputDims[0].isLiteral())
      cerr << ", N " << outputDims[0].getLiteral();
    if (outputDims[1].isLiteral())
      cerr << ", CO " << outputDims[1].getLiteral();
    if (outputDims[2].isLiteral())
      cerr << ", WO " << outputDims[2].getLiteral();
    if (outputDims[3].isLiteral())
      cerr << ", HO " << outputDims[3].getLiteral();
    if (pads[0].isLiteral())
      cerr << ", ph begin " << pads[0].getLiteral();
    if (pads[2].isLiteral())
      cerr << ", ph end " << pads[2].getLiteral();
    if (pads[1].isLiteral())
      cerr << ", pw begin " << pads[1].getLiteral();
    if (pads[3].isLiteral())
      cerr << ", pw end " << pads[3].getLiteral();
    cerr << endl;
  }
#endif

  // Set type for the first output.
  setOutputDims(outputDims);
  return success();
}

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Support function for verifiers.
//===----------------------------------------------------------------------===//

namespace {

// For ops without filter, pass nullptr in filterOperand.
template <class T>
static LogicalResult verifyKernelShape(T *op, Value filterOperand,
    std::optional<ArrayAttr> kernelShapeOpt, int64_t spatialRank) {
  if (filterOperand && !hasShapeAndRank(filterOperand)) {
    // Won't be able to do any checking at this stage.
    return success();
  }
  // 1) Get shape of filter. Shape is not guaranteed to be compile time
  // constant.
  ArrayRef<int64_t> filterShape =
      filterOperand ? mlir::cast<ShapedType>(filterOperand.getType()).getShape()
                    : ArrayRef<int64_t>();
  // 2) Get kernel_shape attribute
  if (!kernelShapeOpt.has_value()) {
    assert(
        filterOperand && "ops without filter have mandatory kernel_shape arg");
    // Don't have a kernel shape explicitly, still make sure that the filter
    // shape are fine if known. If size is negative, ok since this is runtime.
    // If positive, ok since it must be strictly positive. If zero, that is
    // bad.
    for (int i = 0; i < spatialRank; ++i)
      if (filterShape[2 + i] == 0)
        return op->emitError("Bad spatial filter size: cannot be zero");
    return success();
  }
  // 3) Verify that we have the right number.
  if ((int64_t)ArrayAttrSize(kernelShapeOpt) != spatialRank)
    return op->emitError(
        "kernel_shape length incompatible with spatial dimensions");
  // 4) Verify that they are all positive.
  for (int i = 0; i < spatialRank; ++i) {
    auto attrSize = ArrayAttrIntVal(kernelShapeOpt, i);
    if (attrSize < 1)
      return op->emitError("Bad kernel_shape value: must be strictly positive");
    if (filterOperand) {
      // Has a shape from filter, make sure its consistent.
      auto filterSize = filterShape[2 + i];
      if (filterSize >= 0 && filterSize != attrSize)
        return op->emitError(
            "Bad kernel_shape value: does not match filter sizes");
    }
  }
  return success();
}

template <class T>
static LogicalResult verifyStrides(T *op, int64_t spatialRank) {
  // 1) Get strides attribute.
  auto strides = op->getStrides();
  if (!strides.has_value())
    return success();
  // 2) Verify that we have the right number.
  if ((int64_t)ArrayAttrSize(strides) != spatialRank)
    return op->emitError("strides length incompatible with spatial dimensions");
  // 3) Verify that they are all positive.
  for (int i = 0; i < spatialRank; ++i) {
    auto attrSize = ArrayAttrIntVal(strides, i);
    if (attrSize < 1)
      return op->emitError("Bad stride value: must be strictly positive");
  }
  return success();
}

template <class T>
static LogicalResult verifyDilations(T *op, int64_t spatialRank) {
  // 1) Get dilation attribute.
  auto dilations = op->getDilations();
  if (!dilations.has_value())
    return success();
  // 2) Verify that we have the right number.
  if ((int64_t)ArrayAttrSize(dilations) != spatialRank)
    return op->emitError(
        "dilations length incompatible with spatial dimensions");
  // 3) Verify that they are all positive.
  for (int i = 0; i < spatialRank; ++i) {
    auto attrSize = ArrayAttrIntVal(dilations, i);
    if (attrSize < 1)
      return op->emitError("Bad dilation value: must be strictly positive");
  }
  return success();
}

template <class T>
static LogicalResult verifyPadding(T *op, int64_t spatialRank) {
  // Verify auto pad field.
  auto autoPad = op->getAutoPad();
  if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER" ||
      autoPad == "VALID" || autoPad == "NOTSET") {
    // Ok, known auto pad value.
  } else {
    return op->emitError("Unknown auto pad option");
  }
  // Verify pad values, if defined.
  auto pads = op->getPads();
  if (!pads.has_value())
    return success();
  // Verify that we have the right number of pad values.
  if ((int32_t)ArrayAttrSize(pads) != 2 * spatialRank)
    return op->emitError("pads length incompatible with spatial dimensions");
  // Verify the values of the pads.
  if (autoPad == "NOTSET") {
    for (int i = 0; i < 2 * spatialRank; ++i)
      if (ArrayAttrIntVal(pads, i) < 0)
        return op->emitError("Bad pad value: must be nonnegative");
  } else {
    for (int i = 0; i < 2 * spatialRank; ++i)
      if (ArrayAttrIntVal(pads, i) != 0)
        return op->emitError("Bad pad value: nonzero pad value only allowed "
                             "with NOTSET option");
  }
  return success();
}

template <class T>
static LogicalResult verifyOutputShape(T *op, int64_t spatialRank) {
  // 1) Get output shape attribute.
  auto outputShape = op->getOutputShape();
  if (!outputShape.has_value())
    return success();
  // 2) Verify that we have the right number.
  if ((int64_t)ArrayAttrSize(outputShape) != spatialRank)
    return op->emitError(
        "output shape length incompatible with spatial dimensions");
  // 3) Verify that they are all positive.
  for (int i = 0; i < spatialRank; ++i) {
    auto attrSize = ArrayAttrIntVal(outputShape, i);
    if (attrSize < 1)
      return op->emitError("Bad stride value: must be strictly positive");
  }
  return success();
}

} // namespace
