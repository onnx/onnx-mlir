/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- RNNBase.cpp - Lowering RNN Ops -----------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file defines base functions for lowerng the ONNX RNN Operators.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/RNN/RNNBase.hpp"

using namespace mlir;

// Check a Value's type is none or not.
bool isNoneType(Value val) { return val.getType().isa<NoneType>(); }

// Get a dimension of the tensor's shape.
int64_t dimAt(Value val, int index) {
  return val.getType().cast<ShapedType>().getShape()[index];
}

/// Insert Allocate and Deallocate for the all hidden output.
/// Shape :: [seq_length, num_directions, batch_size, hidden_size]
Value allocAllHidden(ConversionPatternRewriter &rewriter, Location loc, Value X,
    Value W, Value R, Value output, bool insertDealloc) {
  Value alloc;
  if (!isNoneType(output)) {
    auto memRefType = convertToMemRefType(output.getType());
    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else {
      auto memRefShape = memRefType.getShape();
      SmallVector<Value, 2> allocOperands;
      if (memRefShape[0] < 0) {
        // Get seq_length from X.
        auto dim = rewriter.create<DimOp>(loc, X, 0);
        allocOperands.emplace_back(dim);
      }
      if (memRefShape[1] < 0) {
        // Get num_directions from W.
        auto dim = rewriter.create<DimOp>(loc, W, 0);
        allocOperands.emplace_back(dim);
      }
      if (memRefShape[2] < 0) {
        // Get batch_size from X.
        auto dim = rewriter.create<DimOp>(loc, X, 1);
        allocOperands.emplace_back(dim);
      }
      if (memRefShape[3] < 0) {
        // Get hidden_size from R.
        auto dim = rewriter.create<DimOp>(loc, R, 2);
        allocOperands.emplace_back(dim);
      }
      alloc = rewriter.create<AllocOp>(loc, memRefType, allocOperands);
      if (insertDealloc) {
        auto *parentBlock = alloc.getDefiningOp()->getBlock();
        auto dealloc = rewriter.create<DeallocOp>(loc, alloc);
        dealloc.getOperation()->moveBefore(&parentBlock->back());
      }
    }
  } else {
    alloc = output;
  }
  return alloc;
}

/// Insert Allocate and Deallocate for the hidden or cell output.
/// Shape :: [num_directions, batch_size, hidden_size]
Value allocHiddenOrCell(ConversionPatternRewriter &rewriter, Location loc,
    Value X, Value W, Value R, Value output, bool insertDealloc) {
  MemRefType memRefType;
  if (!isNoneType(output))
    memRefType = convertToMemRefType(output.getType());
  else {
    memRefType = MemRefType::get(
        {/*num_directions=*/dimAt(W, 0), /*batch_size=*/dimAt(X, 1),
            /*hidden_size=*/dimAt(R, 2)},
        X.getType().cast<ShapedType>().getElementType());
    // The hidden or cell is not a return value but a temporary value, so always
    // dealloc it.
    insertDealloc = true;
  }

  Value alloc;
  if (hasAllConstantDimensions(memRefType))
    alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
  else {
    auto memRefShape = memRefType.getShape();
    SmallVector<Value, 2> allocOperands;
    if (memRefShape[0] < 0) {
      // Get num_directions from W.
      auto dim = rewriter.create<DimOp>(loc, W, 0);
      allocOperands.emplace_back(dim);
    }
    if (memRefShape[1] < 0) {
      // Get batch_size from X.
      auto dim = rewriter.create<DimOp>(loc, X, 1);
      allocOperands.emplace_back(dim);
    }
    if (memRefShape[2] < 0) {
      // Get hidden_size from R.
      auto dim = rewriter.create<DimOp>(loc, R, 2);
      allocOperands.emplace_back(dim);
    }
    alloc = rewriter.create<AllocOp>(loc, memRefType, allocOperands);
    if (insertDealloc) {
      auto *parentBlock = alloc.getDefiningOp()->getBlock();
      auto dealloc = rewriter.create<DeallocOp>(loc, alloc);
      dealloc.getOperation()->moveBefore(&parentBlock->back());
    }
  }

  return alloc;
}

// Initialize the hidden and cell states.
void initializeHiddenAndCell(ConversionPatternRewriter &rewriter, Location loc,
    Value ht, Value ct, Value initialH, Value initialC, Type elementType,
    bool onlyHidden) {
  Value zero = emitConstantOp(rewriter, loc, elementType, 0);
  int nLoops = 3;
  BuildKrnlLoop initializationLoops(rewriter, loc, nLoops);
  initializationLoops.createDefineAndIterateOp(ht);
  auto ipInitializationLoops = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(initializationLoops.getIterateBlock());
  {
    SmallVector<Value, 4> IVs;
    for (int i = 0; i < nLoops; ++i)
      IVs.emplace_back(initializationLoops.getInductionVar(i));

    Value hiddenVal = zero;
    if (!isNoneType(initialH))
      hiddenVal = rewriter.create<KrnlLoadOp>(loc, initialH, IVs);
    rewriter.create<KrnlStoreOp>(loc, hiddenVal, ht, IVs);

    if (!onlyHidden) {
      Value cellVal = zero;
      if (!isNoneType(initialC))
        cellVal = rewriter.create<KrnlLoadOp>(loc, initialC, IVs);
      rewriter.create<KrnlStoreOp>(loc, cellVal, ct, IVs);
    }
  }
  rewriter.restoreInsertionPoint(ipInitializationLoops);
}

// Apply an activation function on a given scalar operand.
Value applyActivation(ConversionPatternRewriter &rewriter, Location loc,
    RNNActivation activation, Value scalarOperand) {
  Value res;

  MemRefType scalarMemRefType =
      MemRefType::get({}, scalarOperand.getType(), {}, 0);
  Value alloc = rewriter.create<AllocOp>(loc, scalarMemRefType);
  rewriter.create<KrnlStoreOp>(loc, scalarOperand, alloc, ArrayRef<Value>{});

  std::vector<mlir::NamedAttribute> attributes;
  if (activation.alpha) {
    attributes.emplace_back(
        rewriter.getNamedAttr("alpha", activation.alpha.getValue()));
  }
  if (activation.beta) {
    attributes.emplace_back(
        rewriter.getNamedAttr("beta", activation.beta.getValue()));
  }

  if (activation.name.equals_lower("relu"))
    res = rewriter.create<ONNXReluOp>(loc, scalarMemRefType, alloc);
  else if (activation.name.equals_lower("tanh"))
    res = rewriter.create<ONNXTanhOp>(loc, scalarMemRefType, alloc);
  else if (activation.name.equals_lower("sigmoid"))
    res = rewriter.create<ONNXSigmoidOp>(loc, scalarMemRefType, alloc);
  else if (activation.name.equals_lower("affine"))
    llvm_unreachable("Unsupported activation");
  else if (activation.name.equals_lower("leakyrelu"))
    res = rewriter.create<ONNXLeakyReluOp>(
        loc, scalarMemRefType, alloc, attributes);
  else if (activation.name.equals_lower("thresholdedrelu"))
    res = rewriter.create<ONNXThresholdedReluOp>(
        loc, scalarMemRefType, alloc, attributes);
  else if (activation.name.equals_lower("scaledtanh"))
    llvm_unreachable("Unsupported activation");
  else if (activation.name.equals_lower("hardsigmoid"))
    res = rewriter.create<ONNXHardSigmoidOp>(
        loc, scalarMemRefType, alloc, attributes);
  else if (activation.name.equals_lower("elu"))
    res = rewriter.create<ONNXEluOp>(loc, scalarMemRefType, alloc, attributes);
  else if (activation.name.equals_lower("softsign"))
    res = rewriter.create<ONNXSoftsignOp>(loc, scalarMemRefType, alloc);
  else if (activation.name.equals_lower("softplus"))
    res = rewriter.create<ONNXSoftplusOp>(loc, scalarMemRefType, alloc);
  else
    llvm_unreachable("Unsupported activation");

  Value result = rewriter.create<KrnlLoadOp>(loc, res);
  return result;
}
