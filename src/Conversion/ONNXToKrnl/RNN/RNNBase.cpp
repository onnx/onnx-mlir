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
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

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
  IndexExprScope scope(rewriter, loc);
  Value alloc;
  if (!isNoneType(output)) {
    MemRefBoundsIndexCapture XBounds(X);
    MemRefBoundsIndexCapture WBounds(W);
    MemRefBoundsIndexCapture RBounds(R);
    SmallVector<IndexExpr, 4> dims;
    // Get seq_length from X.
    dims.emplace_back(XBounds.getDim(0));
    // Get num_directions from W.
    dims.emplace_back(WBounds.getDim(0));
    // Get batch_size from X.
    dims.emplace_back(XBounds.getDim(1));
    // Get hidden_size from R.
    dims.emplace_back(RBounds.getDim(2));
    auto memRefType = convertToMemRefType(output.getType());
    alloc = insertAllocAndDeallocSimple(
        rewriter, nullptr, memRefType, loc, dims, insertDealloc);
  } else {
    alloc = output;
  }
  return alloc;
}

/// Insert Allocate and Deallocate for the intermediate hidden or cell states.
/// Shape :: [batch_size, hidden_size]
Value allocIntermediateState(
    ConversionPatternRewriter &rewriter, Location loc, Value X, Value R) {
  IndexExprScope scope(rewriter, loc);
  auto memRefType = MemRefType::get({/*batch_size=*/dimAt(X, 1),
                                        /*hidden_size=*/dimAt(R, 2)},
      X.getType().cast<ShapedType>().getElementType());
  MemRefBoundsIndexCapture XBounds(X);
  MemRefBoundsIndexCapture RBounds(R);
  SmallVector<IndexExpr, 2> dims;
  // Get batch_size from X.
  dims.emplace_back(XBounds.getDim(1));
  // Get hidden_size from R.
  dims.emplace_back(RBounds.getDim(2));
  // The hidden or cell is not a return value but a temporary value, so always
  // dealloc it.
  Value alloc = insertAllocAndDeallocSimple(
      rewriter, nullptr, memRefType, loc, dims, /*insertDealloc=*/true);
  return alloc;
}

/// Initialize the intermediate hidden and cell states.
void initializeIntermediateStates(ConversionPatternRewriter &rewriter,
    Location loc, Value forwardHt, Value reverseHt, Value forwardCt,
    Value reverseCt, Value initialH, Value initialC, Type elementType,
    StringRef direction, bool onlyHidden) {
  Value zero = emitConstantOp(rewriter, loc, elementType, 0);
  Value zeroIndex = emitConstantOp(rewriter, loc, rewriter.getIndexType(), 0);
  Value oneIndex = emitConstantOp(rewriter, loc, rewriter.getIndexType(), 1);

  int nLoops = 2;
  BuildKrnlLoop initializationLoops(rewriter, loc, nLoops);
  if (direction == FORWARD || direction == BIDIRECTIONAL)
    initializationLoops.createDefineAndIterateOp(forwardHt);
  else
    initializationLoops.createDefineAndIterateOp(reverseHt);
  auto ipInitializationLoops = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(initializationLoops.getIterateBlock());
  {
    KrnlBuilder createKrnl(rewriter, loc);
    SmallVector<Value, 4> IVs;
    IVs.emplace_back(initializationLoops.getInductionVar(0));
    IVs.emplace_back(initializationLoops.getInductionVar(1));

    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      SmallVector<Value, 4> initialIVs;
      initialIVs.emplace_back(zeroIndex);
      initialIVs.emplace_back(initializationLoops.getInductionVar(0));
      initialIVs.emplace_back(initializationLoops.getInductionVar(1));
      if (isNoneType(initialH))
        createKrnl.store(zero, forwardHt, IVs);
      else {
        Value h = createKrnl.load(initialH, initialIVs);
        createKrnl.store(h, forwardHt, IVs);
      }
      if (!onlyHidden) {
        if (isNoneType(initialC))
          createKrnl.store(zero, forwardCt, IVs);
        else {
          Value c = createKrnl.load(initialC, initialIVs);
          createKrnl.store(c, forwardCt, IVs);
        }
      }
    }

    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      SmallVector<Value, 4> initialIVs;
      if (direction == REVERSE)
        initialIVs.emplace_back(zeroIndex);
      else
        initialIVs.emplace_back(oneIndex);
      initialIVs.emplace_back(initializationLoops.getInductionVar(0));
      initialIVs.emplace_back(initializationLoops.getInductionVar(1));
      if (isNoneType(initialH))
        createKrnl.store(zero, reverseHt, IVs);
      else {
        Value h = createKrnl.load(initialH, initialIVs);
        createKrnl.store(h, reverseHt, IVs);
      }
      if (!onlyHidden) {
        if (isNoneType(initialC))
          createKrnl.store(zero, reverseCt, IVs);
        else {
          Value c = createKrnl.load(initialC, initialIVs);
          createKrnl.store(c, reverseCt, IVs);
        }
      }
    }
  }
  rewriter.restoreInsertionPoint(ipInitializationLoops);
}

/// Insert Allocate and Deallocate for the hidden or cell output.
/// Shape :: [num_directions, batch_size, hidden_size]
Value allocHiddenOrCell(ConversionPatternRewriter &rewriter, Location loc,
    Value X, Value W, Value R, Value output, bool insertDealloc) {
  IndexExprScope scope(rewriter, loc);
  Value alloc;
  if (!isNoneType(output)) {
    MemRefBoundsIndexCapture XBounds(X);
    MemRefBoundsIndexCapture WBounds(W);
    MemRefBoundsIndexCapture RBounds(R);
    SmallVector<IndexExpr, 3> dims;
    // Get num_directions from W.
    dims.emplace_back(WBounds.getDim(0));
    // Get batch_size from X.
    dims.emplace_back(XBounds.getDim(1));
    // Get hidden_size from R.
    dims.emplace_back(RBounds.getDim(2));
    MemRefType memRefType = convertToMemRefType(output.getType());
    alloc = insertAllocAndDeallocSimple(
        rewriter, nullptr, memRefType, loc, dims, insertDealloc);
  } else {
    alloc = output;
  }
  return alloc;
}

// Initialize the hidden and cell states.
void initializeHiddenAndCell(ConversionPatternRewriter &rewriter, Location loc,
    Value ht, Value ct, Value initialH, Value initialC, Type elementType,
    bool onlyHidden) {
  // TODO remove
  // scope(rewriter, loc);
  ImplicitLocOpBuilder lb(loc, rewriter);
  KrnlBuilder createKrnl(lb);
  MemRefBuilder createMemRef(lb);
  Value zero = emitConstantOp(rewriter, loc, elementType, 0);
  unsigned htRank = ht.getType().cast<MemRefType>().getRank();
  Value iZero = lb.create<ConstantIndexOp>(0);
  SmallVector<Value, 4> htLbs(htRank, iZero);
  SmallVector<Value, 4> htUbs;
  for (unsigned r = 0; r < htRank; ++r) {
    htUbs.emplace_back(createMemRef.dim(ht, r));
  }
  ValueRange loops = createKrnl.defineLoops(htRank);
  createKrnl.iterate(loops, loops, htLbs, htUbs,
      [&](KrnlBuilder &createKrnl, ValueRange indices) {
        Value hiddenVal = zero;
        if (!isNoneType(initialH))
          hiddenVal = createKrnl.load(initialH, indices);
        createKrnl.store(hiddenVal, ht, indices);

        if (!onlyHidden) {
          Value cellVal = zero;
          if (!isNoneType(initialC))
            cellVal = createKrnl.load(initialC, indices);
          createKrnl.store(cellVal, ct, indices);
        }
      });
}

/// Store a state into the output of the RNN op.
/// The input state is 2D and the output state is 3D with '1' or '2' is
/// pretended, depending on 'direction'.
void stateToOutputForHiddenOrCell(ConversionPatternRewriter &rewriter,
    Location loc, Value forwardVal, Value reverseVal, StringRef direction,
    Value output) {
  // TODO remove
  ImplicitLocOpBuilder lb(loc, rewriter);
  KrnlBuilder createKrnl(lb);
  MemRefBuilder createMemRef(lb);
  if (direction == FORWARD || direction == REVERSE) {
    Value val = (direction == FORWARD) ? forwardVal : reverseVal;
    Value sizeInBytes = getDynamicMemRefSizeInBytes(rewriter, loc, val);
    createKrnl.memcpy(output, val, sizeInBytes);
  } else { // BIDIRECTIONAL
    unsigned rank = forwardVal.getType().cast<MemRefType>().getRank();
    Value zero = lb.create<ConstantIndexOp>(0);
    Value one = lb.create<ConstantIndexOp>(1);
    SmallVector<Value, 4> lbs(rank, zero);
    SmallVector<Value, 4> ubs;
    for (unsigned r = 0; r < rank; ++r) {
      ubs.emplace_back(createMemRef.dim(forwardVal, r));
    }
    ValueRange loops = createKrnl.defineLoops(2);
    createKrnl.iterate(loops, loops, lbs, ubs,
        [&](KrnlBuilder &createKrnl, ValueRange indices) {
          Value b(indices[0]), h(indices[1]);
          // Forward.
          Value val = createKrnl.load(forwardVal, {b, h});
          createKrnl.store(val, output, {zero, b, h});
          // Reverse.
          val = createKrnl.load(reverseVal, {b, h});
          createKrnl.store(val, output, {one, b, h});
        });
  }
}

// Apply an activation function on a given scalar operand.
Value applyActivation(OpBuilder &rewriter, Location loc,
    RNNActivation activation, Value operand) {
  Value res;

  bool isScalar = !operand.getType().isa<ShapedType>();
  assert(isScalar && "Not a scalar operand");

  MemRefType memRefType = MemRefType::get({}, operand.getType(), {}, 0);
  // Single scalar, no need for default alignment.
  MemRefBuilder createMemRef(rewriter, loc);
  Value alloc = createMemRef.alloca(memRefType);
  rewriter.create<KrnlStoreOp>(loc, operand, alloc, ArrayRef<Value>{});

  std::vector<mlir::NamedAttribute> attributes;
  if (activation.alpha) {
    attributes.emplace_back(
        rewriter.getNamedAttr("alpha", activation.alpha.getValue()));
  }
  if (activation.beta) {
    attributes.emplace_back(
        rewriter.getNamedAttr("beta", activation.beta.getValue()));
  }

  // Change equality to be case insensitive.
  if (activation.name.equals_insensitive("relu"))
    res = rewriter.create<ONNXReluOp>(loc, memRefType, alloc);
  else if (activation.name.equals_insensitive("tanh"))
    res = rewriter.create<ONNXTanhOp>(loc, memRefType, alloc);
  else if (activation.name.equals_insensitive("sigmoid"))
    res = rewriter.create<ONNXSigmoidOp>(loc, memRefType, alloc);
  else if (activation.name.equals_insensitive("affine"))
    llvm_unreachable("Unsupported activation");
  else if (activation.name.equals_insensitive("leakyrelu"))
    res = rewriter.create<ONNXLeakyReluOp>(loc, memRefType, alloc, attributes);
  else if (activation.name.equals_insensitive("thresholdedrelu"))
    res = rewriter.create<ONNXThresholdedReluOp>(
        loc, memRefType, alloc, attributes);
  else if (activation.name.equals_insensitive("scaledtanh"))
    llvm_unreachable("Unsupported activation");
  else if (activation.name.equals_insensitive("hardsigmoid"))
    res =
        rewriter.create<ONNXHardSigmoidOp>(loc, memRefType, alloc, attributes);
  else if (activation.name.equals_insensitive("elu"))
    res = rewriter.create<ONNXEluOp>(loc, memRefType, alloc, attributes);
  else if (activation.name.equals_insensitive("softsign"))
    res = rewriter.create<ONNXSoftsignOp>(loc, memRefType, alloc);
  else if (activation.name.equals_insensitive("softplus"))
    res = rewriter.create<ONNXSoftplusOp>(loc, memRefType, alloc);
  else
    llvm_unreachable("Unsupported activation");

  res = rewriter.create<KrnlLoadOp>(loc, res);

  return res;
}

/// Create a copy of a slice of X at a specific timestep.
/// This function is not able correctly to emit 'dealloc' for the copy since it
/// does not have enough information about the parent context. Users must
/// deallocate the copy by themselves.
Value emitXSliceAt(ConversionPatternRewriter &rewriter, Location loc, Value X,
    Value timestepIV) {
  // TODO remove
  IndexExprScope scope(rewriter, loc);
  ImplicitLocOpBuilder lb(loc, rewriter);
  KrnlBuilder createKrnl(rewriter, loc);
  MemRefBuilder createMemRef(rewriter, loc);

  int64_t batchSize = dimAt(X, 1);
  int64_t inputSize = dimAt(X, 2);
  auto elementType = X.getType().cast<ShapedType>().getElementType();
  MemRefType sliceXType = MemRefType::get({batchSize, inputSize}, elementType);

  // Allocate a buffer
  MemRefBoundsIndexCapture XBounds(X);
  SmallVector<IndexExpr, 2> dims;
  dims.emplace_back(XBounds.getDim(1));
  dims.emplace_back(XBounds.getDim(2));
  Value sliceX = insertAllocAndDeallocSimple(
      rewriter, nullptr, sliceXType, loc, dims, /*insertDealloc=*/false);

  // Copy data from X.
  Value iZero = lb.create<ConstantIndexOp>(0);
  SmallVector<Value, 2> lbs(2, iZero);
  SmallVector<Value, 2> ubs;
  for (unsigned r = 0; r < 2; ++r) {
    ubs.emplace_back(createMemRef.dim(sliceX, r));
  }
  ValueRange loops = createKrnl.defineLoops(2);
  createKrnl.iterate(
      loops, loops, lbs, ubs, [&](KrnlBuilder &createKrnl, ValueRange indices) {
        Value b(indices[0]), i(indices[1]);
        Value val = createKrnl.load(X, {timestepIV, b, i});
        createKrnl.store(val, sliceX, {b, i});
      });

  return sliceX;
}
