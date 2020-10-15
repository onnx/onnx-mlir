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

// Insert Allocate and Deallocate for the all hidden output.
Value allocAllHidden(ConversionPatternRewriter &rewriter, Location loc, Value X,
    Value W, Value R, Value output, bool insertDealloc) {
  Value alloc;
  if (!isNoneType(output)) {
    auto yMemRefType = convertToMemRefType(output.getType());
    if (hasAllConstantDimensions(yMemRefType))
      alloc = insertAllocAndDealloc(yMemRefType, loc, rewriter, insertDealloc);
    else {
      llvm_unreachable("Unsupported dynamic dimensions.");
    }
  } else {
    alloc = output;
  }
  return alloc;
}

// Insert Allocate and Deallocate for the hidden or cell output.
Value allocHiddenOrCell(ConversionPatternRewriter &rewriter, Location loc,
    Value X, Value W, Value R, Value output, bool insertDealloc) {
  Value alloc;
  if (!isNoneType(output)) {
    auto yhMemRefType = convertToMemRefType(output.getType());
    if (hasAllConstantDimensions(yhMemRefType))
      alloc = insertAllocAndDealloc(yhMemRefType, loc, rewriter, insertDealloc);
    else
      llvm_unreachable("Unsupported dynamic dimensions.");
  } else {
    auto yhMemRefType = MemRefType::get(
        {/*num_directions=*/dimAt(W, 0), /*batch_size=*/dimAt(X, 1),
            /*hidden_size=*/dimAt(R, 2)},
        X.getType().cast<ShapedType>().getElementType());
    alloc = insertAllocAndDealloc(yhMemRefType, loc, rewriter, true);
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
      hiddenVal = rewriter.create<AffineLoadOp>(loc, initialH, IVs);
    rewriter.create<AffineStoreOp>(loc, hiddenVal, ht, IVs);

    if (!onlyHidden) {
      Value cellVal = zero;
      if (!isNoneType(initialC))
        cellVal = rewriter.create<AffineLoadOp>(loc, initialC, IVs);
      rewriter.create<AffineStoreOp>(loc, cellVal, ct, IVs);
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
  rewriter.create<AffineStoreOp>(loc, scalarOperand, alloc, ArrayRef<Value>{});

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

  Value result = rewriter.create<AffineLoadOp>(loc, res);
  return result;
}
