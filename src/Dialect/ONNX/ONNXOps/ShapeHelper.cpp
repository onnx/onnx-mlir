/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------ONNXShapeHelper.cpp - help for shapes----------------=== //
//
// Copyright 2020-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file has the computations to compute the shapes using the new index expr
// approach.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/BitVector.h"

#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
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
    // Safety checks for old convention of using -1 for dynamic.
    assert(existingDims[i] != -1 && "dynamic use kDynamic now");
    if (inferredDims[i].isLiteral()) {
      // Index expressions should not use the ShapedType::kDynamic ever to
      // signal dynamic shape. Questionmarks are used for that.
      assert(inferredDims[i].getLiteral() != -1 && "dynamic use questionmark");
      assert(inferredDims[i].getLiteral() != ShapedType::kDynamic &&
             "dynamic use questionmark");
    }
    // ExistingDim is dynamic, nothing to learn from.
    if (existingDims[i] == ShapedType::kDynamic)
      continue;

    // InferredDim is unknown at shape inference: update it.
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
    assert(inferredDims[i].isLiteral() && "isLiteral failed");
    if (existingDims[i] != inferredDims[i].getLiteral()) {
      // Warning for users.
      llvm::outs() << "Warning: [Shape inference, dim " << i
                   << "] the inferred dim (" << inferredDims[i].getLiteral()
                   << ") is different from the existing dim ("
                   << existingDims[i] << "). Use the existing dim instead.\n";
      inferredDims[i] = LiteralIndexExpr(existingDims[i]);
    }
  }
}

//===----------------------------------------------------------------------===//
// ONNX Op Shape Helper
//===----------------------------------------------------------------------===//

ONNXOpShapeHelper::ONNXOpShapeHelper(Operation *inputOp,
    ValueRange inputOperands, IndexExprBuilder *inputIeBuilder,
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
    operands = ValueRange(privateOperandsCache);
  }
}

ONNXOpShapeHelper::~ONNXOpShapeHelper() {
  if (ownScope)
    delete scope;
  if (ownBuilder)
    delete createIE;
}

void ONNXOpShapeHelper::computeShapeAndAssertOnFailure() {
  // Invoke virtual compute shape.
  LogicalResult res = computeShape();
  assert(succeeded(res) && "Failed to compute shape");
}

void ONNXOpShapeHelper::setOutputDims(
    const DimsExpr &inferredDims, int n, bool refineShape) {
  privateOutputsDims[n] = inferredDims;
  if (refineShape) {
    Value output = getOutput(n);
    refineDims(privateOutputsDims[n], output);
  }
}

LogicalResult ONNXOpShapeHelper::setOutputDimsFromOperand(
    Value operand, int n, bool refineShape) {
  // Output and operand have the same shape. Just pass the operand shape to the
  // output.
  DimsExpr outputDims;
  createIE->getShapeAsDims(operand, outputDims);
  setOutputDims(outputDims, n, refineShape);
  return success();
}

LogicalResult ONNXOpShapeHelper::setOutputDimsFromLiterals(
    SmallVector<int64_t, 4> shape, int n, bool refineShape) {
  // Output has the shape given by the vector of integer numbers. Number
  // ShapedType::kDynamic is transformed into a questionmark.
  DimsExpr outputDims;
  getIndexExprListFromShape(shape, outputDims);
  setOutputDims(outputDims, n, refineShape);
  return success();
}

LogicalResult ONNXOpShapeHelper::setOutputDimsFromTypeWithConstantShape(
    Type type, int n, bool refineShape) {
  RankedTensorType rankedType = type.dyn_cast<RankedTensorType>();
  if (!rankedType)
    return failure();
  DimsExpr outputDims;
  getIndexExprListFromShape(rankedType.getShape(), outputDims);
  if (!IndexExpr::isNonNegativeLiteral(outputDims))
    return failure();
  setOutputDims(outputDims, n, refineShape);
  return success();
}

// Reuse the same type for each of the outputs.
LogicalResult ONNXOpShapeHelper::computeShapeAndUpdateType(
    Type elementType, Attribute encoding) {
  // Invoke virtual compute shape.
  if (failed(computeShape()))
    return op->emitError("Failed to scan parameters successfully");
  uint64_t resNum = op->getNumResults();
  for (uint64_t i = 0; i < resNum; ++i) {
    // If we have an optional type, leave it as is.
    if (op->getResults()[i].getType().isa<NoneType>())
      continue;
    llvm::SmallVector<int64_t, 4> shapeVect;
    IndexExpr::getShape(getOutputDims(i), shapeVect);
    // Set refineShape to false here because we refine it (or not) when setting
    // the output shape. So there is no need to perform this again here.
    updateType(op->getResults()[i], shapeVect, elementType, encoding,
        /*refineShape*/ false);
  }
  return success();
}

// Use a distinct type for each of the output.
LogicalResult ONNXOpShapeHelper::computeShapeAndUpdateTypes(
    TypeRange elementTypeRange, mlir::ArrayRef<mlir::Attribute> encodingList) {
  uint64_t resNum = op->getNumResults();
  assert((elementTypeRange.size() == resNum) &&
         "Incorrect number of elementTypes");
  bool hasEncoding = encodingList.size() > 0;
  assert((!hasEncoding || encodingList.size() == resNum) &&
         "Incorrect number of encoding");
  // Invoke virtual compute.
  if (failed(computeShape()))
    return op->emitError("Failed to scan " + op->getName().getStringRef() +
                         " parameters successfully");
  for (uint64_t i = 0; i < resNum; ++i) {
    // If we have an optional type, leave it as is.
    if (op->getResults()[i].getType().isa<NoneType>())
      continue;
    llvm::SmallVector<int64_t, 4> shapeVect;
    IndexExpr::getShape(getOutputDims(i), shapeVect);
    Type currElementType = elementTypeRange[i];
    // Set refineShape to false here because we refine it (or not) when setting
    // the output shape. So there is no need to perform this again here.
    updateType(op->getResults()[i], shapeVect, currElementType,
        hasEncoding ? encodingList[i] : nullptr, /*refineShape*/ false);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ONNX Broadcast Op Shape Helper
//===----------------------------------------------------------------------===//

LogicalResult ONNXBroadcastOpShapeHelper::customComputeShape(
    ValueRange initialOperands, DimsExpr *additionalOperand) {
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
        std::max(outputRank, createIE->getShapedTypeRank(initialOperands[i]));
  dimsExpr.resize(outputRank);

  // Prepare dims for every input. Prepend 1s if the input's shape has smaller
  // rank, so that all the shapes have the same rank.
  LiteralIndexExpr one(1);
  for (uint64_t i = 0; i < numOfInputs; ++i) {
    uint64_t r = createIE->getShapedTypeRank(initialOperands[i]);
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
        if (!hasUniBroadcasting) {
          dimsExpr[j] = nextDimExpr;
        }
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

// Attempt to rule out broadcasting at compile time, using dim analysis when
// available (i.e. nonnull). Must be called after computeShape.
bool ONNXBroadcastOpShapeHelper::hasNoBroadcast(DimAnalysis *dimAnalysis) {
  // First use static analysis to rule out broadcast. If we cannot rule out
  // broadcasting for any reasons, hasNoBroadcast is set to false.
  bool hasNoBroadcast = true;
  for (uint64_t r = 0; r < outputRank && hasNoBroadcast; ++r) {
    bool hasOne, hasOtherThanOne;
    hasOne = hasOtherThanOne = false;
    for (DimsExpr dims : inputsDims) {
      if (!dims[r].isLiteral()) {
        // Has dynamic values.. possible broadcast, assume the worst.
        hasNoBroadcast = false;
        break;
      }
      int64_t lit = dims[r].getLiteral();
      if (lit == 1)
        hasOne = true;
      else
        hasOtherThanOne = true;
    }
    if (hasOne && hasOtherThanOne)
      // Has a known broadcast situation. No need for further analysis,
      // broadcasting has been detected.
      return false;
  }

  // Using the most conservative analysis, we did not detect any broadcasting,
  // we are good.
  if (hasNoBroadcast)
    return true;

  // We have dynamic dimensions that prevented us to rule out broadcasting, try
  // the more expensive dimAnalysis approach now, if available.
  if (!dimAnalysis)
    return false;
  // In some cases, we can have more inputDims than operands (custom broadcast
  // operators, e.g. ONNXExtendOp). Dismiss such cases as we need here the
  // values of each of the inputs.
  int64_t inputNum = operands.size();
  if ((int64_t)inputsDims.size() != inputNum)
    return false;
  // Check if we can prove that each operand has the same shape.
  for (int i = 1; i < inputNum; ++i)
    if (!dimAnalysis->sameShape(operands[0], operands[i]))
      return false;
  // All have the same shape.
  return true;
}

// Determine if all but one input is a scalar, in which case the broadcasting is
// trivial. For example: (4x16xf32 and 1xf32), (?x?xf32 and 1xf32).
bool ONNXBroadcastOpShapeHelper::hasScalarBroadcast(DimAnalysis *dimAnalysis) {
  // Find the inputs that are scalar.
  int scalarNum = 0;
  int nonScalarID = -1;
  int dimNum = inputsDims.size();
  llvm::SmallVector<int, 4> scalarInput(dimNum, true);
  for (int d = 0; d < dimNum; ++d) {
    for (uint64_t r = 0; r < outputRank; ++r) {
      if (!inputsDims[d][r].isLiteralAndIdenticalTo(1)) {
        scalarInput[d] = false;
        nonScalarID = d;
        break;
      }
    }
    if (scalarInput[d])
      scalarNum++;
  }
  if (scalarNum == 0 || scalarNum == dimNum) {
    // No scalars/all scalars. There is no scalar broadcast.
    return false;
  }
  assert(nonScalarID != -1 && "expected one non-scalar input");

  // Now find out if all the non-scalar inputs are identical. If there is
  // only one non-scalar input, we are fine by definition.
  if (dimNum - scalarNum == 1)
    return true;
  // To use dim analysis, it must be defined and we must have an operand for
  // each dim.
  bool canUseDimAnalysis = dimAnalysis && (int)operands.size() == dimNum;
  int rank = inputsDims[nonScalarID].size();
  for (int d = 0; d < dimNum; ++d) {
    if (scalarInput[d] || d == nonScalarID)
      // Scalar or self, nothing to test.
      continue;
    if (canUseDimAnalysis &&
        !dimAnalysis->sameShape(operands[nonScalarID], operands[d])) {
      // Cannot have 2 non-scalar that are different.
      return false;
    } else {
      // Cannot use analysis, just ensure that both have the same static values.
      assert((int)inputsDims[d].size() == rank && "must have the same rank");
      for (int r = 0; r < rank; ++r) {
        if (!inputsDims[nonScalarID][r].isLiteral() ||
            !inputsDims[d][r].isLiteral() ||
            inputsDims[nonScalarID][r].getLiteral() !=
                inputsDims[d][r].getLiteral())
          return false;
      }
    }
  }
  // Checked all non-scalar inputs and they are all identical.
  return true;
}

bool ONNXBroadcastOpShapeHelper::hasManageableBroadcastForInnerDims(
    int64_t &innerDimNum, int64_t &innerDimLiteralSize,
    IndexExpr &innerDimDynamicSize, DimAnalysis *dimAnalysis) {

  int64_t dimNum = inputsDims.size();
  bool canUseDimAnalysis = dimAnalysis && (int64_t)operands.size() == dimNum;
  // fprintf(stderr, "hi alex: can use dim analysis %d\n",
  // (int)canUseDimAnalysis);
  innerDimLiteralSize = 1;
  innerDimDynamicSize = LiteralIndexExpr(1);
  llvm::SmallBitVector isScalar(dimNum, true);
  llvm::SmallBitVector isScalarUpToNow(dimNum, true);
  // Walk through the rank from innermost to outermost;
  for (int64_t r = outputRank - 1; r >= 0; --r) {
    // fprintf(stderr, "hi alex: iter %d\n", (int)r);
    // Detect scalars and non-scalars at this rank.
    int64_t nonScalarID = -1;
    int64_t scalarNum = 0;
    int64_t scalarUpToNowNum = 0;
    for (int64_t d = 0; d < dimNum; ++d) {
      isScalar[d] = inputsDims[d][r].isLiteralAndIdenticalTo(1);
      if (isScalar[d])
        scalarNum++;
      else
        nonScalarID = d;
      isScalarUpToNow[d] = isScalar[d] & isScalarUpToNow[d];
      if (isScalarUpToNow[d])
        scalarUpToNowNum++;
    }
    int64_t nonScalarNum = dimNum - scalarNum;
    // See if we have a conflicts among the non scalars.
    if (nonScalarNum >= 2) {
      // fprintf(stderr, "hi alex: non scalar>2\n");
      // Has 2 or more non scalar, make sure they are compatible
      bool possibleNonScalarBroadcast = false;
      for (int64_t d = 0; d < dimNum; ++d) {
        // Consider only dims d that are not scalar; check identity with
        // nonScalarID (so also skip compare with self).
        if (isScalar[d] || d == nonScalarID)
          continue;
        // Compare nonScalarID with d
        if (inputsDims[nonScalarID][r].isLiteral() &&
            inputsDims[d][r].isLiteral()) {
          // Both literal, do a literal check.
          if (inputsDims[nonScalarID][r].getLiteral() ==
              inputsDims[d][r].getLiteral())
            // same dims, we are fine.
            continue;
          // Has hard broadcast; this dim is no good
          possibleNonScalarBroadcast = true;
          break;
        }
        if (canUseDimAnalysis &&
            dimAnalysis->sameDim(operands[nonScalarID], r, operands[d], r)) {
          // Analysis demonstrated them to be the same, we are fine.
          // fprintf(stderr, "hi alex: dyn analysis say its fine\n");
          continue;
        }
        // Analysis could not prove operands's r dim to be identical.
        // fprintf(stderr, "hi alex: dyn analysis say its ambiguous\n");
        possibleNonScalarBroadcast = true;
        break;
      }
      if (possibleNonScalarBroadcast) {
        // fprintf(stderr, "hi alex: possibleNonScalarBroadcast\n");
        // Had 2+ non scalar dims that are incompatible, e.g. (2, ?) or (?, ?).
        // Abort at this dim; previous is fine.
        innerDimNum = outputRank - (r + 1);
        return innerDimNum > 0;
      }
    }
    // Now here, we are guaranteed that if there are non-scalars, they must be
    // the same. So if we have 1+ non scalar, and 1+ scalar, then we know we
    // have a scalar broadcast here.
    bool hasScalarBroadcast = false;
    if (nonScalarNum > 0 && scalarNum > 0) {
      // fprintf(stderr, "hi alex: hasScalarBroadcast\n");
      hasScalarBroadcast = true;
    }
    // If we have a scalar broadcast situation, all of the currently seen 1s
    // must be from scalars, that is there must have 1s from the inner dimension
    // up to the current dimension d.
    if (hasScalarBroadcast && scalarNum != scalarUpToNowNum) {
      // Abort as we have something like this (1x4x1, 1x1x1 2x4x1), namely where
      // the first dim has a broadcast that is not for a scalar.
      // fprintf(stderr, "hi alex: scalarNum != scalarUpToNowNum\n");
      innerDimNum = outputRank - (r + 1);
      return innerDimNum > 0;
    }
    // This dim is fine for manageable broadcasting. Account for the cumulative
    // size of the inner dimensions.
    if (nonScalarNum > 0) {
      // If all scalar, then dim is 1, and there is nothing to do. Otherwise
      // accumulate in literal or dynamic sizes.
      if (inputsDims[nonScalarID][r].isLiteral()) {
        innerDimLiteralSize *= inputsDims[nonScalarID][r].getLiteral();
      } else {
        innerDimDynamicSize = innerDimDynamicSize * inputsDims[nonScalarID][r];
      }
    }
  }
  // Came up to here, we are able to collapse them all.
  innerDimNum = outputRank;
  return innerDimNum > 0;
}

LogicalResult ONNXBroadcastOpShapeHelper::getAccessExprs(Value operand,
    int64_t operandIndex, const SmallVectorImpl<IndexExpr> &loopAccessExprs,
    SmallVectorImpl<IndexExpr> &operandAccessExprs, bool flattenedInnerDims,
    bool ruledOutBroadcast) {
  // Get info.
  int64_t loopDepth = loopAccessExprs.size();
  int64_t inputSize = inputsDims.size();
  int64_t operandRank = operand.getType().cast<ShapedType>().getRank();
  // Note: if the loops were flattened, all of the operands must also have
  // flattened memref dimensions. Check that there.
  assert(operandRank <= loopDepth &&
         "operand cannot have more dims that the number of surrounding loops.");
  if (!flattenedInnerDims)
    assert(loopDepth == (int64_t)outputRank &&
           "without flattening, expect one loop iter variable per output rank");
  // Emtpy the access expr, just in case.
  operandAccessExprs.clear();
  // There is this case where we have no broadcast per se, but we have
  // mixtures of 1xTYPE vs TYPE scalars. Handle this case properly here.
  if (operandRank == 0)
    return success();

  // The ruledOutBroadcast pattern can be established by shape inference using
  // DimAnalysis. If that is available, and broadcasting was ruled out, then
  // more efficient code can be generated.
  bool noBroadcasting =
      ruledOutBroadcast || (hasUniBroadcasting && operandIndex == 0);

  for (int64_t r = 0; r < operandRank; ++r) {
    // Shape helper may pretend 1s, thus adjust dimension index accordingly.
    // Take loopDepth instead of outputRank as loopDepth reflect the (possible)
    // flattening of the loops.
    int64_t dimIndex = loopDepth - operandRank + r;
    SymbolIndexExpr dim(inputsDims[operandIndex][dimIndex]);

    // Compute access index based on broadcasting rules.
    // If all other operand dims are 1, just use the output access index.
    // Otherwise, emit a select op.
    bool allOtherInputDimsAreOne = true;
    for (int64_t i = 0; i < inputSize; ++i) {
      if (i == operandIndex)
        continue;
      IndexExpr dim = inputsDims[i][dimIndex];
      if (!dim.isLiteralAndIdenticalTo(1)) {
        allOtherInputDimsAreOne = false;
        break;
      }
    }

    if (noBroadcasting || allOtherInputDimsAreOne) {
      // no Broadcasting -> no worries, just use the index.
      //
      // allOtherInputDimsAreOne -> use loop index without worries.
      // Our dim may be [*, dim, *] where all the others are [*, 1, *];
      // Regardless of the value of dim (constant, `?`) we can use the loop
      // index variable without reservation as if dim is 1, then its 0 by def,
      // and if dim>1, then its 0...dim-1 without issue.
      operandAccessExprs.emplace_back(loopAccessExprs[dimIndex]);
    } else if (flattenedInnerDims && r == operandRank - 1) {
      // Flattened dims: we only have manageable broadcast; either scalar
      // (access is 0) or non-broadcast (access is loop index).
      if (dim.isLiteralAndIdenticalTo(1))
        operandAccessExprs.emplace_back(LiteralIndexExpr(0));
      else
        operandAccessExprs.emplace_back(loopAccessExprs[dimIndex]);
    } else {
      // If dim is a compile time constant, then the test below will resolve
      // at compile time. If dim is dynamic (i.e. only known at runtime), then
      // we will issue code for the compare and select and the right value
      // will be used at runtime.
      operandAccessExprs.emplace_back(
          IndexExpr::select(dim > 1, loopAccessExprs[dimIndex], 0));
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Setting a new constant or attribute value.
//===----------------------------------------------------------------------===//

void SaveOnnxConstInOp(
    Operation *op, const llvm::SmallVectorImpl<int64_t> &vals, int operandId) {
  OpBuilder builder(op->getContext());
  builder.setInsertionPoint(op);
  OnnxBuilder createONNX(builder, op->getLoc());
  Value constVal = createONNX.constantInt64(vals);
  op->setOperand(operandId, constVal);
}

void SaveOnnxConstInOp(Operation *op, MutableOperandRange operand,
    const llvm::SmallVectorImpl<int64_t> &vals) {
  OpBuilder builder(op->getContext());
  builder.setInsertionPoint(op);
  OnnxBuilder createONNX(builder, op->getLoc());
  Value constVal = createONNX.constantInt64(vals);
  operand.assign(constVal);
}

//===----------------------------------------------------------------------===//
// Setting the type of the output using explict element type and shape.
//===----------------------------------------------------------------------===//

/// Update a tensor type by using the given shape, elementType and encoding.
void updateType(Value val, ArrayRef<int64_t> shape, Type elementType,
    Attribute encoding, bool refineShape) {
  // Try to combine the given shape and the output's shape if possible.
  SmallVector<int64_t, 4> inferredShape;
  if (refineShape) {
    IndexExprScope scope(nullptr, val.getLoc());
    DimsExpr inferredDims;
    for (int64_t d : shape) {
      // TODO: "-1" may be used if "shape" is coming from e.g. the parameters
      // of an `onnx.Reshape` op?
      if (ShapedType::isDynamic(d) || d == -1)
        inferredDims.emplace_back(QuestionmarkIndexExpr(/*isFloat*/ false));
      else
        inferredDims.emplace_back(LiteralIndexExpr(d));
    }
    refineDims(inferredDims, val);
    IndexExpr::getShape(inferredDims, inferredShape);
  } else {
    // TODO: "-1" may be used if "shape" is coming from e.g. the parameters of
    // an `onnx.Reshape` op?
    for (size_t i = 0; i < shape.size(); ++i)
      inferredShape.emplace_back(
          shape[i] != -1 ? shape[i] : ShapedType::kDynamic);
  }

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
  // Reset type
  val.setType(resType);
}

static void resetTypeShapeToQuestionmarks(Value val) {
  // Only deal with ranked tensor types here.
  RankedTensorType valType = val.getType().dyn_cast<RankedTensorType>();
  if (!valType)
    return;
  // Reset any compile time literal to unknown (aka question marks).
  SmallVector<int64_t, 4> newShape(valType.getRank(), ShapedType::kDynamic);
  auto resType = RankedTensorType::Builder(valType).setShape(newShape);
  // Reset type
  val.setType(resType);
}

void resetTypesShapeToQuestionmarks(Operation *op) {
  int numRes = op->getNumResults();
  for (int i = 0; i < numRes; ++i)
    resetTypeShapeToQuestionmarks(op->getResult(i));
}

//===----------------------------------------------------------------------===//
// Template instantiation (last).
//===----------------------------------------------------------------------===//

// Ideally should be in more specific files

} // namespace onnx_mlir
