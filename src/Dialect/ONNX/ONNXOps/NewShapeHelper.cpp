/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- NewShapeHelper.hpp - help for shapes ---------------===//
//
// Copyright 2020-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file has the computations to compute the shapes using the new index expr
// approach.
//
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
#define AEE_MIGRATE 1
#if AEE_MIGRATE

// nothing

//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
#else

#include "src/Dialect/ONNX/ONNXOps/NewShapeHelper.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

#include <algorithm>

#define DEBUG_TYPE "shape-helper"

using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// ONNX Op Shape Helper
//===----------------------------------------------------------------------===//

/// Refine `inferredDims` using the output's shape if possible. For example,
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
    // inferredDim is unknown at lowering: use existing dim for efficiency.
    if (!inferredDims[i].isLiteral()) {
      inferredDims[i] = LiteralIndexExpr(existingDims[i]);
      continue;
    }
    // inferredDim is different from existingDim. Believe in existingDim.
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

//===----------------------------------------------------------------------===//
// ONNX Op Shape Helper
//===----------------------------------------------------------------------===//

NewONNXOpShapeHelper::NewONNXOpShapeHelper(Operation *inputOp,
    ArrayRef<Value> inputOperands, IndexExprBuilder *inputIeBuilder,
    IndexExprScope *inputScope)
    : op(inputOp), operands(inputOperands), createIE(inputIeBuilder),
      scope(inputScope), privateOutputsDims(), ownScope(inputScope == nullptr),
      ownBuilder(inputIeBuilder == nullptr) {
  assert(op && "Expecting a valid operation pointer");
  if (ownBuilder) {
    createIE = new IndexExprBuilderForAnalysis(op->getLoc());
    assert(createIE && "failed to create a new builder");
  }
  if (ownScope) {
    scope = new IndexExprScope(createIE->getBuilderPtr(), createIE->getLoc());
    assert(scope && "failed to create a new scope");
  }
  privateOutputsDims.resize(op->getNumResults());
  // When we have no inputOperands, get them from the operation.
  if (inputOperands.size() == 0) {
    // A operand cache is used here, as we don't want to rely on the live range
    // of the passed parameter. Possibly a more elegant solution can be used, I
    // could not find one at this time.
    privateOperandsCache = llvm::SmallVector<Value, 4>(
        op->getOperands().begin(), op->getOperands().end());
    operands = ArrayRef<Value>(privateOperandsCache);
  }
}

NewONNXOpShapeHelper::~NewONNXOpShapeHelper() {
  if (ownScope)
    delete scope;
  if (ownBuilder)
    delete createIE;
}

void NewONNXOpShapeHelper::computeShapeAndAssertOnFailure() {
  // Invoke virtual compute shape.
  LogicalResult res = computeShape();
  assert(succeeded(res) && "Failed to compute shape");
}

void NewONNXOpShapeHelper::setOutputDims(const DimsExpr &inferredDims, int n) {
  Value output = getOutput(n);
  privateOutputsDims[n] = inferredDims;
  refineDims(privateOutputsDims[n], output);
}

LogicalResult NewONNXOpShapeHelper::computeShapeFromOperand(Value operand) {
  // Output and operand have the same shape. Just pass the operand shape to the
  // output.
  DimsExpr outputDims;
  createIE->getShapeAsDims(operand, outputDims);
  setOutputDims(outputDims);
  return success();
}

// Reuse the same type for each of the outputs.
mlir::LogicalResult NewONNXOpShapeHelper::computeShapeAndUpdateType(
    Type elementType) {
  // Invoke virtual compute shape.
  if (failed(computeShape()))
    return op->emitError("Failed to scan parameters successfully");
  uint64_t resNum = op->getNumResults();
  for (uint64_t i = 0; i < resNum; ++i) {
    llvm::SmallVector<int64_t, 4> shapeVect;
    IndexExpr::getShape(getOutputDims(i), shapeVect);
    updateType(op->getResults()[i], shapeVect, elementType);
  }
  return mlir::success();
}

// Use a distinct type for each of the output.
LogicalResult NewONNXOpShapeHelper::computeShapeAndUpdateTypes(
    TypeRange elementTypeRange) {
  uint64_t resNum = op->getNumResults();
  assert((elementTypeRange.size() == resNum) && "Incorrect elementTypes size");
  // Invoke virtual compute.
  if (failed(computeShape()))
    return op->emitError("Failed to scan " + op->getName().getStringRef() +
                         " parameters successfully");
  for (uint64_t i = 0; i < resNum; ++i) {
    llvm::SmallVector<int64_t, 4> shapeVect;
    IndexExpr::getShape(getOutputDims(i), shapeVect);
    Type currElementType = elementTypeRange[i];
    updateType(op->getResults()[i], shapeVect, currElementType);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ONNX Op Shape Helper for Generic Unary Elementwise Operations
//===----------------------------------------------------------------------===//

LogicalResult NewONNXUnaryOpShapeHelper::computeShape() {
  // Output and input have the same shape. Just pass the input shape to the
  // output.
  return computeShapeFromOperand(operands[0]);
}

//===----------------------------------------------------------------------===//
// ONNX Broadcast Op Shape Helper
//===----------------------------------------------------------------------===//

LogicalResult NewONNXBroadcastOpShapeHelper::customComputeShape(
    ArrayRef<Value> initialOperands, DimsExpr *additionalOperand) {
  // if additionalOperand is not used, we expect a zero-sized vector.
  // A temporary IndexExpr vector for the output.
  DimsExpr dimsExpr;
  uint64_t numOfInputs = initialOperands.size();

  // Compute rank of the output. Rank of the output is the maximum rank of all
  // initial operands.
  uint64_t additionalOperRank =
      additionalOperand ? additionalOperand->size() : 0;
  outputRank = additionalOperRank;
  for (uint64_t i = 0; i < numOfInputs; ++i)
    outputRank =
        std::max(outputRank, createIE->getTypeRank(initialOperands[i]));
  dimsExpr.resize(outputRank);

  // Prepare dims for every input. Prepend 1s if the input's shape has smaller
  // rank, so that all the shapes have the same rank.
  LiteralIndexExpr one(1);
  for (uint64_t i = 0; i < numOfInputs; ++i) {
    uint64_t r = createIE->getTypeRank(initialOperands[i]);
    // Prepend 1s.
    DimsExpr dims(outputRank - r, one);
    // Get from the input.
    for (uint64_t k = 0; k < r; ++k)
      dims.emplace_back(createIE->getShapeAsDim(initialOperands[i], k));
    inputsDims.emplace_back(dims);
  }

  // Handle the additional operand here.
  if (additionalOperRank > 0) {
    DimsExpr dims(outputRank - additionalOperRank, one);
    for (uint64_t k = 0; k < additionalOperRank; ++k)
      dims.emplace_back((*additionalOperand)[k]);
    inputsDims.emplace_back(dims);
    numOfInputs++;
  }

  // Initialize the output with the first operand.
  dimsExpr = inputsDims[0];

  // Note on IndexExpr. When we are not allowed to generate code, QuestionMark
  // stands for anything but a literal. When we are allowed to generate code,
  // there should be no more QuestionMarks as we are allowed to generate
  // affine/symbols/dims/non-affine expressions. Since this code predominantly
  // runs when we can gen code (as it actually does gen max ops), we should
  // use !isLiteral() for anything that is runtime. The comments were left
  // unchanged.

  //  Now compute each broadcasted dimension for the output. Folding over the
  //  other operands along the current dimension index.
  for (uint64_t i = 1; i < numOfInputs; ++i) {
    for (uint64_t j = 0; j < outputRank; ++j) {
      // Set the output dimension based on the two dimension values.
      // Dimension value can be one of 1, QuestionMark, LiteralNot1.
      IndexExpr currentDimExpr = dimsExpr[j];
      IndexExpr nextDimExpr = inputsDims[i][j];
      // Case: 1 - *.
      if (currentDimExpr.isLiteralAndIdenticalTo(1)) {
        if (!hasUniBroadcasting && !hasNoBroadcasting)
          dimsExpr[j] = nextDimExpr;
        continue;
      }
      // Case: LiteralNot1 - *.
      if (currentDimExpr.isLiteralAndDifferentThan(1)) {
        // LiteralNot1 - LiteralNot1 => keep unchanged with verifying.
        if (nextDimExpr.isLiteralAndDifferentThan(1) &&
            !currentDimExpr.isLiteralAndIdenticalTo(nextDimExpr))
          return op->emitError("Incompatible broadcast matching " +
                               std::to_string(currentDimExpr.getLiteral()) +
                               " with " +
                               std::to_string(currentDimExpr.getLiteral()));
        // Case: LiteralNot1 - (QuestionMark or 1) => Keep unchanged without
        // verifying.
        continue;
      }
      // Case: QuestionMark - 1 => keep unchanged.
      if (!currentDimExpr.isLiteral() &&
          nextDimExpr.isLiteralAndIdenticalTo(1)) {
        continue;
      }
      // Case QuestionMark - LiteralNot1 => set to LiteralNot1 without
      // verifying.
      if (!currentDimExpr.isLiteral() &&
          nextDimExpr.isLiteralAndDifferentThan(1)) {
        dimsExpr[j] = nextDimExpr;
        continue;
      }
      // Case: QuestionMark - QuestionMark
      if (!hasUniBroadcasting) {
        dimsExpr[j] = IndexExpr::max(currentDimExpr, nextDimExpr);
      }
    }
  }
  // Set the final output.
  setOutputDims(dimsExpr);
  return success();
}

LogicalResult NewONNXBroadcastOpShapeHelper::getAccessExprs(Value operand,
    uint64_t operandIndex, const SmallVectorImpl<IndexExpr> &outputAccessExprs,
    SmallVectorImpl<IndexExpr> &operandAccessExprs) {
  if (hasNoBroadcasting || (hasUniBroadcasting && operandIndex == 0)) {
    for (IndexExpr ie : outputAccessExprs)
      operandAccessExprs.emplace_back(ie);
    return success();
  }

  uint64_t operandRank = operand.getType().cast<ShapedType>().getRank();
  for (uint64_t i = 0; i < operandRank; ++i) {
    // Shape helper may pretend 1s, thus adjust dimension index accordingly.
    uint64_t dimIndex = outputRank - operandRank + i;
    SymbolIndexExpr dim(inputsDims[operandIndex][dimIndex]);

    // Compute access index based on broadcasting rules.
    // If all other operand dims are 1, just use the output access index.
    // Otherwise, emit a select op.
    bool allOtherInputDimsAreOne = true;
    for (uint64_t i = 0; i < inputsDims.size(); ++i) {
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
// Pooling ops.
//===----------------------------------------------------------------------===//

LogicalResult NewONNXPoolOpShapeHelper::customComputeShape(Value xValue,
    Value wValue, Optional<ArrayAttr> kernelShapeOpt, llvm::StringRef autoPad,
    Optional<ArrayAttr> padOpt, Optional<ArrayAttr> strideOpt,
    Optional<ArrayAttr> dilationOpt) {
  // Basic information.
  int64_t rank = createIE->getTypeRank(xValue);
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
      assert(hasFilter && "no kernel shape and no filter: unkown kernel shape");
      int ii = i + spatialOffset;
      kernelShape.emplace_back(createIE->getShapeAsSymbol(wValue, ii));
    }
  }
  // Pads, at this stage a given compile-time literal or default 0.
  for (int i = 0; i < 2 * spatialRank; ++i) {
    int64_t p = padOpt.has_value() ? ArrayAttrIntVal(padOpt, i) : 0;
    pads.emplace_back(LiteralIndexExpr(p));
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

//===----------------------------------------------------------------------===//
// Setting a new constant or attribute value.
//===----------------------------------------------------------------------===//

void SaveOnnxConstInOp(mlir::Operation *op,
    const llvm::SmallVectorImpl<int64_t> &vals, int operandId) {
  OpBuilder builder(op->getContext());
  builder.setInsertionPoint(op);
  OnnxBuilder createONNX(builder, op->getLoc());
  Value constVal = createONNX.constantInt64(vals);
  op->setOperand(operandId, constVal);
}

void SaveOnnxConstInOp(mlir::Operation *op, MutableOperandRange operand,
    const llvm::SmallVectorImpl<int64_t> &vals) {
  OpBuilder builder(op->getContext());
  builder.setInsertionPoint(op);
  OnnxBuilder createONNX(builder, op->getLoc());
  Value constVal = createONNX.constantInt64(vals);
  operand.assign(constVal);
}

//===----------------------------------------------------------------------===//
// Template instantiation (last).
//===----------------------------------------------------------------------===//

// Ideally should be in more specific files

} // namespace onnx_mlir

//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
#endif