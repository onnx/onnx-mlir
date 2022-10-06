/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------ONNXShapeHelper.cpp - help for shapes----------------=== //
//
// Copyright 2020 The IBM Research Authors.
//
// =============================================================================
//
// This file has the computations to compute the shapes using the new index expr
// approach.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

#include <algorithm>

#define DEBUG_TYPE "shape-helper"

using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// ONNX Op Shape Helper
//===----------------------------------------------------------------------===//

/// Refine `inferredDims` using the output's shape if possbile. For example,
/// replacing a dynamic dim in `inferredDims` by a static dim in the output's
/// shape.
static void refineDims(DimsExpr &inferredDims, Value output) {
  // Nothing to do if the output is unranked.
  if (!isRankedShapedType(output.getType()))
    return;

  llvm::ArrayRef<int64_t> existingDims = getShape(output.getType());
  // Do not handle the case of scalar tensor whose type can be tensor<f32>
  // or tensor<1xf32>. Just use the inferredShape in this case.
  if (existingDims.size() < 1 || inferredDims.size() < 1)
    return;

  assert((existingDims.size() == inferredDims.size()) &&
         "Inferred shape and existing shape are inconsistent in the number "
         "of elements");

  // Try to update inferredDim if existingDim is static.
  for (unsigned i = 0; i < existingDims.size(); ++i) {
    // existingDim is dynamic, nothing to do.
    if (existingDims[i] == -1)
      continue;

    // inferredDim is unknown at shape inference: update it.
    if (inferredDims[i].isQuestionmark()) {
      inferredDims[i] = LiteralIndexExpr(existingDims[i]);
      continue;
    }
    // inferredDim is unknown at lowering: use exising dim for efficiency.
    if (!inferredDims[i].isLiteral()) {
      inferredDims[i] = LiteralIndexExpr(existingDims[i]);
      continue;
    }
    // inferedDim is different from existingDim. Believe in existingDim.
    if (inferredDims[i].isLiteral() &&
        (existingDims[i] != inferredDims[i].getLiteral())) {
      // Warning for users.
      llvm::outs() << "Warning: [Shape inference] the inferred dim ("
                   << inferredDims[i].getLiteral()
                   << ") is different from the existing dim ("
                   << existingDims[i] << "). Use the existing dim instead.\n";
      inferredDims[i] = LiteralIndexExpr(existingDims[i]);
    }
  }
}

// Reuse scope if given, otherwise create one now and free in destructor.
template <class OP>
ONNXOpShapeHelper<OP>::ONNXOpShapeHelper(
    OP *newOp, int numResults, IndexExprScope *inScope)
    : op(newOp), fGetDenseVal(getDenseElementAttributeFromONNXValue),
      fLoadVal(nullptr), outputsDims(), ownScope(inScope == nullptr) {
  assert(op && "Expecting a valid pointer");
  if (ownScope)
    scope = new IndexExprScope(nullptr, newOp->getLoc());
  setNumberOfOutputs(numResults);
}

template <class OP>
ONNXOpShapeHelper<OP>::ONNXOpShapeHelper(OP *newOp, int numResults,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseValInput,
    ArrayValueIndexCapture::LoadVal fLoadVal, IndexExprScope *inScope)
    : op(newOp), fLoadVal(fLoadVal), outputsDims(),
      ownScope(inScope == nullptr) {
  assert(op && "Expecting a valid pointer");
  if (ownScope)
    scope = new IndexExprScope(rewriter, newOp->getLoc());
  setNumberOfOutputs(numResults);
  // Get the dense value by combining provided function (if any) with the
  // default one.
  fGetDenseVal = [=](Value array) {
    DenseElementsAttr res = nullptr;
    // Try with the provided method, if any.
    if (fGetDenseValInput)
      res = fGetDenseValInput(array);
    // If provided method was not provided or failed, try default ONNX method.
    if (!res)
      res = getDenseElementAttributeFromONNXValue(array);
    return res;
  };
}

template <>
Value ONNXOpShapeHelper<Operation>::getOutput(int n) {
  return op->getResult(n);
}

// Set output dims for the N-th output.
template <class OP>
void ONNXOpShapeHelper<OP>::setOutputDims(DimsExpr inferredDims, int n) {
  outputsDims[n] = inferredDims;
  // Try to refine outputsDims[n] using the output's shape if possbile. For
  // example, replacing a dynamic dim in outputsDims[n] by a static dim in the
  // output's shape.
  Value output = getOutput(n);
  refineDims(outputsDims[n], output);
}

//===----------------------------------------------------------------------===//
// ONNX Op Shape Helper for Broadcasting
//===----------------------------------------------------------------------===//

// Since Broadcasted ops are generic (due to implementation in Elementwise.cpp
// templates), we assume below that each of the broadcast ops have exactly one
// output. If that were not the case, then we would need to pass the number of
// results as a parameter to both constructors below.

template <class OP>
ONNXOpBroadcastedShapeHelper<OP>::ONNXOpBroadcastedShapeHelper(OP *newOp,
    IndexExprScope *inScope, bool uniBroadcasting, bool noBroadcasting)
    : ONNXOpShapeHelper<OP>(newOp, 1, inScope), inputsDims(), outputRank(-1),
      isUniBroadcasting(uniBroadcasting), isNoBroadcasting(noBroadcasting) {}

template <class OP>
ONNXOpBroadcastedShapeHelper<OP>::ONNXOpBroadcastedShapeHelper(OP *newOp,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal, IndexExprScope *inScope,
    bool uniBroadcasting, bool noBroadcasting)
    : ONNXOpShapeHelper<OP>(
          newOp, 1, rewriter, fGetDenseVal, fLoadVal, inScope),
      inputsDims(), outputRank(-1), isUniBroadcasting(uniBroadcasting),
      isNoBroadcasting(noBroadcasting) {}

template <class OP>
LogicalResult ONNXOpBroadcastedShapeHelper<OP>::computeShape(
    ArrayRef<Value> operands, DimsExpr &additionalOperand) {
  // if additionalOperand is not used, we expect a zero-sized vector.
  // A temporary IndexExpr vector for the output.
  DimsExpr dimsExpr;
  int64_t numOfInputs = operands.size();

  // Compute rank of the output. Rank of the output is the maximum rank of all
  // operands.
  int additionalOperRank = additionalOperand.size();
  bool hasAdditionalOper = additionalOperRank > 0;
  outputRank = hasAdditionalOper ? additionalOperRank : -1;
  for (int64_t i = 0; i < numOfInputs; ++i) {
    int64_t r = operands[i].getType().cast<ShapedType>().getRank();
    outputRank = std::max(outputRank, r);
  }
  dimsExpr.resize(outputRank);

  // Prepare dims for every input. Prepend 1s if the input's shape has smaller
  // rank, so that all the shapes have the same rank.
  LiteralIndexExpr oneIE(1);
  for (int64_t i = 0; i < numOfInputs; ++i) {
    MemRefBoundsIndexCapture bounds(operands[i]);
    int64_t r = bounds.getRank();
    // Prepend 1s.
    DimsExpr dims;
    for (int64_t k = 0; k < outputRank - r; ++k)
      dims.emplace_back(oneIE);
    // Get from the input.
    for (int64_t k = 0; k < r; ++k)
      dims.emplace_back(bounds.getDim(k));
    inputsDims.emplace_back(dims);
  }
  // Handle the additional operand here.
  if (hasAdditionalOper) {
    DimsExpr dims(outputRank - additionalOperRank, oneIE);
    for (int64_t k = 0; k < additionalOperRank; ++k)
      dims.emplace_back(additionalOperand[k]);
    inputsDims.emplace_back(dims);
    numOfInputs++;
  }

  // Initialize the output with the first operand.
  dimsExpr = inputsDims[0];

  // Note on IndexExpr. When we are not allowed to generate code, QuestionMark
  // stands for anything but a literal. When we are allowed to generate code,
  // there should be no more QuestionMarks as we are allowed to generate
  // affine/symbols/dims/non-affine expressions. Since this code predominantly
  // runs when we can gen code (as it actually does gen max ops), we should use
  // !isLiteral() for anything that is runtime. The comments were left
  // unchanged.

  //  Now compute each broadcasted dimension for the output. folding over the
  //  other operands along the current dimension index.
  for (int64_t i = 1; i < numOfInputs; ++i) {
    for (int64_t j = 0; j < outputRank; ++j) {
      // Set the output dimension based on the two dimension values.
      // Dimension value can be one of 1, QuestionMark, LiteralNot1.
      IndexExpr currentDimExpr = dimsExpr[j];
      IndexExpr nextDimExpr = inputsDims[i][j];

      // 1 - QuestionMark
      // 1 - LiteralNot1
      // 1 - 1
      if (currentDimExpr.isLiteralAndIdenticalTo(1)) {
        if (!isUniBroadcasting && !isNoBroadcasting)
          dimsExpr[j] = nextDimExpr;
        continue;
      }

      if (currentDimExpr.isLiteralAndDifferentThan(1)) {
        // LiteralNot1 - LiteralNot1 => keep unchanged with verifying.
        if (nextDimExpr.isLiteralAndDifferentThan(1))
          assert(currentDimExpr.isLiteralAndIdenticalTo(nextDimExpr));
        // Keep unchanged without verifying:
        //   - LiteralNot1 - QuestionMark
        //   - LiteralNot1 - 1
        continue;
      }

      // QuestionMark - 1 => keep unchanged.
      if (!currentDimExpr.isLiteral() &&
          nextDimExpr.isLiteralAndIdenticalTo(1)) {
        continue;
      }

      // QuestionMark - LiteralNot1 => set to LiteralNot1 without verifying.
      if (!currentDimExpr.isLiteral() &&
          nextDimExpr.isLiteralAndDifferentThan(1)) {
        dimsExpr[j] = nextDimExpr;
        continue;
      }

      // QuestionMark - QuestionMark
      if (!isUniBroadcasting) {
        dimsExpr[j] = IndexExpr::max(currentDimExpr, nextDimExpr);
      }
    }
  }
  // Set the final output.
  ONNXOpShapeHelper<OP>::setOutputDims(dimsExpr);
  return success();
}

template <class OP>
LogicalResult ONNXOpBroadcastedShapeHelper<OP>::GetAccessExprs(Value operand,
    unsigned operandIndex, const SmallVectorImpl<IndexExpr> &outputAccessExprs,
    SmallVectorImpl<IndexExpr> &operandAccessExprs) {
  if (isNoBroadcasting || (isUniBroadcasting && operandIndex == 0)) {
    for (IndexExpr ie : outputAccessExprs)
      operandAccessExprs.emplace_back(ie);
    return success();
  }

  auto operandRank = operand.getType().cast<ShapedType>().getRank();
  for (decltype(operandRank) i = 0; i < operandRank; ++i) {
    // Shape helper may pretend 1s, thus adjust dimension index accordingly.
    auto dimIndex = outputRank - operandRank + i;
    SymbolIndexExpr dim(inputsDims[operandIndex][dimIndex]);

    // Compute access index based on broadcasting rules.
    // If all other operand dims are 1, just use the output access index.
    // Otherwise, emit a select op.
    bool allOtherInputDimsAreOne = true;
    for (unsigned int i = 0; i < inputsDims.size(); ++i) {
      if (i == operandIndex)
        continue;
      IndexExpr dim = inputsDims[i][dimIndex];
      if (!dim.isLiteralAndIdenticalTo(1)) {
        allOtherInputDimsAreOne = false;
        break;
      }
    }
    if (allOtherInputDimsAreOne) {
      operandAccessExprs.emplace_back(outputAccessExprs[dimIndex]);
    } else {
      operandAccessExprs.emplace_back(
          IndexExpr::select(dim > 1, outputAccessExprs[dimIndex], 0));
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ONNX Generic Broadcast Op Shape Helper
//===----------------------------------------------------------------------===//

ONNXGenericOpBroadcastedShapeHelper::ONNXGenericOpBroadcastedShapeHelper(
    Operation *newOp, IndexExprScope *inScope, bool uniBroadcasting,
    bool noBroadcasting)
    : ONNXOpBroadcastedShapeHelper<Operation>(
          newOp, inScope, uniBroadcasting, noBroadcasting) {}

ONNXGenericOpBroadcastedShapeHelper::ONNXGenericOpBroadcastedShapeHelper(
    Operation *newOp, OpBuilder *rewriter,
    ArrayValueIndexCapture::GetDenseVal fGetDenseVal,
    ArrayValueIndexCapture::LoadVal fLoadVal, IndexExprScope *inScope,
    bool uniBroadcasting, bool noBroadcasting)
    : ONNXOpBroadcastedShapeHelper<Operation>(newOp, rewriter, fGetDenseVal,
          fLoadVal, inScope, uniBroadcasting, noBroadcasting) {}

//===----------------------------------------------------------------------===//
// ONNX Generic Pool Op Shape Helper
//===----------------------------------------------------------------------===//

template <typename OP_TYPE, typename OP_ADAPTOR>
LogicalResult ONNXGenericPoolShapeHelper<OP_TYPE, OP_ADAPTOR>::computeShape(
    OP_ADAPTOR operandAdaptor, Value filterValue,
    Optional<ArrayAttr> kernelShapeOpt, Optional<ArrayAttr> padOpt,
    Optional<ArrayAttr> strideOpt, Optional<ArrayAttr> dilationOpt) {
  // Shape inference indicated by passing a null rewriter pointer.
  // Basic information.
  Value xValue = (Value)operandAdaptor.X();
  int64_t rank = xValue.getType().cast<ShapedType>().getRank();
  int64_t spatialOffset = 2;
  int64_t spatialRank = rank - spatialOffset;

  MemRefBoundsIndexCapture XBounds(operandAdaptor.X());
  MemRefBoundsIndexCapture WBounds(filterValue);

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
    } else if (hasFilter) {
      int ii = i + spatialOffset;
      kernelShape.emplace_back(WBounds.getSymbol(ii));
    } else {
      llvm_unreachable("should have tested the availability of kernel shape");
    }
  }
  // Pads, at this stage a given compile-time literal or default 0.
  for (int i = 0; i < 2 * spatialRank; ++i) {
    int64_t p = padOpt.has_value() ? ArrayAttrIntVal(padOpt, i) : 0;
    pads.emplace_back(LiteralIndexExpr(p));
  }

  // Handle output size: start by inserting batch size and output channels.
  DimsExpr outputDims;
  outputDims.emplace_back(XBounds.getDim(0));
  if (hasFilter)
    outputDims.emplace_back(WBounds.getDim(0)); // CO may be different from CI.
  else
    outputDims.emplace_back(XBounds.getDim(1)); // CO is CI.

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
  auto autoPad = ONNXOpShapeHelper<OP_TYPE>::op->auto_pad();
  LiteralIndexExpr zeroIE(0);
  LiteralIndexExpr oneIE(1);
  for (int64_t i = 0; i < spatialRank; ++i) {
    int64_t ii = i + spatialOffset;
    IndexExpr I = XBounds.getDim(ii);
    IndexExpr K = kernelShape[i];
    LiteralIndexExpr d(dilations[i]);
    LiteralIndexExpr s(strides[i]);
    IndexExpr t1 = K - oneIE;
    IndexExpr kdTerm = t1 * d + oneIE; // (k - 1) * d + 1
    if (autoPad == "NOTSET") {
      IndexExpr p = pads[i] + pads[i + spatialRank]; // Sum both pads.
      IndexExpr t1 = I + p; // Compute floor/ceil((I + p - kdTerm) / s) + 1.
      IndexExpr t2 = t1 - kdTerm;
      IndexExpr O;
      if (ceilMode)
        O = t2.ceilDiv(s);
      else
        O = t2.floorDiv(s);
      O = O + oneIE;
      // Set output dim, and pads already set, nothing more to do.
      outputDims.emplace_back(O);
    } else if (autoPad == "VALID") {
      IndexExpr t1 = I - kdTerm; // Compute ceil((I - kdTerm +1)/s).
      IndexExpr t2 = t1 + oneIE;
      IndexExpr O = t2.ceilDiv(s);
      // Set output dim, and pads already set to zero, nothing more to do.
      outputDims.emplace_back(O);
    } else if (autoPad == "SAME_UPPER" || autoPad == "SAME_LOWER") {
      // Compute output as O = ceil(I/s).
      IndexExpr O = I.ceilDiv(s);
      outputDims.emplace_back(O);
      // Compute sum of pads padSum = (O -1)*s + kdTerm - I.
      IndexExpr t1 = O - oneIE;
      IndexExpr t2 = t1 * s + kdTerm;
      IndexExpr t3 = t2 - I;
      IndexExpr padSum = IndexExpr::max(t3, zeroIE);
      // Single pad value is padSump / 2.
      IndexExpr p = padSum.floorDiv(2);
      // Increment is 1 when pp % 2 != 0
      IndexExpr test = (padSum % 2) != zeroIE;
      IndexExpr inc = IndexExpr::select(test, oneIE, zeroIE);
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
  ONNXOpShapeHelper<OP_TYPE>::setOutputDims(outputDims);
  return success();
}

/// Handle shape inference for unary element-wise operators.
LogicalResult inferShapeForUnaryElementwiseOps(Operation *op) {
  Value input = op->getOperand(0);
  Value output = op->getResult(0);

  if (!hasShapeAndRank(input))
    return success();

  // Inferred shape is getting from the input's shape.
  RankedTensorType inputType = input.getType().dyn_cast<RankedTensorType>();
  updateType(output, inputType.getShape(), inputType.getElementType(),
      inputType.getEncoding());
  return success();
}

/// Update a tensor type by using the given shape, elementType and encoding.
void updateType(Value val, ArrayRef<int64_t> shape, Type elementType,
    Attribute encoding, bool refineShape) {
  // Try to combine the given shape and the output's shape if possbile.
  IndexExprScope scope(nullptr, val.getLoc());
  DimsExpr inferredDims;
  for (int64_t d : shape) {
    if (d == -1)
      inferredDims.emplace_back(QuestionmarkIndexExpr());
    else
      inferredDims.emplace_back(LiteralIndexExpr(d));
  }
  if (refineShape)
    refineDims(inferredDims, val);
  SmallVector<int64_t, 4> inferredShape;
  IndexExpr::getShape(inferredDims, inferredShape);

  // Get element type.
  if (!elementType)
    elementType = getElementType(val.getType());

  // Get encoding.
  if (auto valType = val.getType().dyn_cast<RankedTensorType>())
    if (!encoding)
      encoding = valType.getEncoding();

  // Build result type.
  RankedTensorType resType;
  if (encoding)
    resType = RankedTensorType::get(inferredShape, elementType, encoding);
  else
    resType = RankedTensorType::get(inferredShape, elementType);

  val.setType(resType);
}

//===----------------------------------------------------------------------===//
// ONNX Shape Helper template instantiation
// Keep template instantiation at the end of the file.
//===----------------------------------------------------------------------===//

template struct ONNXOpShapeHelper<ONNXArgMaxOp>;
template struct ONNXOpShapeHelper<ONNXArgMinOp>;
template struct ONNXOpShapeHelper<ONNXAveragePoolOp>;
template struct ONNXOpShapeHelper<ONNXCategoryMapperOp>;
template struct ONNXOpShapeHelper<ONNXClipOp>;
template struct ONNXOpShapeHelper<ONNXCompressOp>;
template struct ONNXOpShapeHelper<ONNXConcatOp>;
template struct ONNXOpShapeHelper<ONNXConvOp>;
template struct ONNXOpShapeHelper<ONNXDepthToSpaceOp>;
template struct ONNXOpShapeHelper<ONNXExpandOp>;
template struct ONNXOpShapeHelper<ONNXFlattenOp>;
template struct ONNXOpShapeHelper<ONNXGatherOp>;
template struct ONNXOpShapeHelper<ONNXGatherElementsOp>;
template struct ONNXOpShapeHelper<ONNXGatherNDOp>;
template struct ONNXOpShapeHelper<ONNXGemmOp>;
template struct ONNXOpShapeHelper<ONNXMatMulOp>;
template struct ONNXOpShapeHelper<ONNXMaxPoolSingleOutOp>;
template struct ONNXOpShapeHelper<ONNXOneHotOp>;
template struct ONNXOpShapeHelper<ONNXPadOp>;
template struct ONNXOpShapeHelper<ONNXReduceSumOp>;
template struct ONNXOpShapeHelper<ONNXReshapeOp>;
template struct ONNXOpShapeHelper<ONNXLRNOp>;
template struct ONNXOpShapeHelper<ONNXReverseSequenceOp>;
template struct ONNXOpShapeHelper<ONNXRoiAlignOp>;
template struct ONNXOpShapeHelper<ONNXShapeOp>;
template struct ONNXOpShapeHelper<ONNXSliceOp>;
template struct ONNXOpShapeHelper<ONNXSpaceToDepthOp>;
template struct ONNXOpShapeHelper<ONNXSplitOp>;
template struct ONNXOpShapeHelper<ONNXSplitV11Op>;
template struct ONNXOpShapeHelper<ONNXSqueezeOp>;
template struct ONNXOpShapeHelper<ONNXSqueezeV11Op>;
template struct ONNXOpShapeHelper<ONNXTileOp>;
template struct ONNXOpShapeHelper<ONNXTopKOp>;
template struct ONNXOpShapeHelper<ONNXTransposeOp>;
template struct ONNXOpShapeHelper<ONNXUnsqueezeOp>;
template struct ONNXOpShapeHelper<ONNXUnsqueezeV11Op>;

template struct ONNXOpBroadcastedShapeHelper<Operation>;
template struct ONNXOpBroadcastedShapeHelper<ONNXExpandOp>;

template struct ONNXGenericPoolShapeHelper<ONNXAveragePoolOp,
    ONNXAveragePoolOpAdaptor>;
template struct ONNXGenericPoolShapeHelper<ONNXConvOp, ONNXConvOpAdaptor>;
template struct ONNXGenericPoolShapeHelper<ONNXMaxPoolSingleOutOp,
    ONNXMaxPoolSingleOutOpAdaptor>;

// Keep template instantiation at the end of the file.

} // namespace onnx_mlir
