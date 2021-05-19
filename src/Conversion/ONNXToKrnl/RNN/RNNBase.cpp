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
        auto dim = rewriter.create<memref::DimOp>(loc, X, 0);
        allocOperands.emplace_back(dim);
      }
      if (memRefShape[1] < 0) {
        // Get num_directions from W.
        auto dim = rewriter.create<memref::DimOp>(loc, W, 0);
        allocOperands.emplace_back(dim);
      }
      if (memRefShape[2] < 0) {
        // Get batch_size from X.
        auto dim = rewriter.create<memref::DimOp>(loc, X, 1);
        allocOperands.emplace_back(dim);
      }
      if (memRefShape[3] < 0) {
        // Get hidden_size from R.
        auto dim = rewriter.create<memref::DimOp>(loc, R, 2);
        allocOperands.emplace_back(dim);
      }
      alloc = rewriter.create<memref::AllocOp>(loc, memRefType, allocOperands);
      if (insertDealloc) {
        auto *parentBlock = alloc.getDefiningOp()->getBlock();
        auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
        dealloc.getOperation()->moveBefore(&parentBlock->back());
      }
    }
  } else {
    alloc = output;
  }
  return alloc;
}

/// Insert Allocate and Deallocate for the hidden or cell output.
/// Shape :: [batch_size, hidden_size]
Value allocHiddenOrCell_(
    ConversionPatternRewriter &rewriter, Location loc, Value X, Value R) {
  // The hidden or cell is not a return value but a temporary value, so always
  // dealloc it.
  bool insertDealloc = true;

  auto memRefType = MemRefType::get({/*batch_size=*/dimAt(X, 1),
                                        /*hidden_size=*/dimAt(R, 2)},
      X.getType().cast<ShapedType>().getElementType());

  Value alloc;
  if (hasAllConstantDimensions(memRefType))
    alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
  else {
    auto memRefShape = memRefType.getShape();
    SmallVector<Value, 2> allocOperands;
    if (memRefShape[0] < 0) {
      // Get batch_size from X.
      auto dim = rewriter.create<memref::DimOp>(loc, X, 1);
      allocOperands.emplace_back(dim);
    }
    if (memRefShape[1] < 0) {
      // Get hidden_size from R.
      auto dim = rewriter.create<memref::DimOp>(loc, R, 2);
      allocOperands.emplace_back(dim);
    }
    alloc = rewriter.create<memref::AllocOp>(loc, memRefType, allocOperands);
    if (insertDealloc) {
      auto *parentBlock = alloc.getDefiningOp()->getBlock();
      auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
      dealloc.getOperation()->moveBefore(&parentBlock->back());
    }
  }

  return alloc;
}

// Initialize the hidden and cell states.
void initializeHiddenAndCell_(ConversionPatternRewriter &rewriter, Location loc,
    Value forwardHt, Value backwardHt, Value forwardCt, Value backwardCt,
    Value initialH, Value initialC, Type elementType, StringRef direction,
    bool onlyHidden) {
  Value zero = emitConstantOp(rewriter, loc, elementType, 0);
  Value zeroIndex = emitConstantOp(rewriter, loc, rewriter.getIndexType(), 0);
  Value oneIndex = emitConstantOp(rewriter, loc, rewriter.getIndexType(), 1);

  int nLoops = 2;
  BuildKrnlLoop initializationLoops(rewriter, loc, nLoops);
  initializationLoops.createDefineAndIterateOp(forwardHt);
  auto ipInitializationLoops = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(initializationLoops.getIterateBlock());
  {
    SmallVector<Value, 4> IVs;
    IVs.emplace_back(initializationLoops.getInductionVar(0));
    IVs.emplace_back(initializationLoops.getInductionVar(1));

    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      SmallVector<Value, 4> initialIVs;
      initialIVs.emplace_back(zeroIndex);
      initialIVs.emplace_back(initializationLoops.getInductionVar(0));
      initialIVs.emplace_back(initializationLoops.getInductionVar(1));
      if (isNoneType(initialH))
        rewriter.create<KrnlStoreOp>(loc, zero, forwardHt, IVs);
      else {
        Value h = rewriter.create<KrnlLoadOp>(loc, initialH, initialIVs);
        rewriter.create<KrnlStoreOp>(loc, h, forwardHt, IVs);
      }
      if (!onlyHidden) {
        if (isNoneType(initialC))
          rewriter.create<KrnlStoreOp>(loc, zero, forwardCt, IVs);
        else {
          Value c = rewriter.create<KrnlLoadOp>(loc, initialC, initialIVs);
          rewriter.create<KrnlStoreOp>(loc, c, forwardCt, IVs);
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
        rewriter.create<KrnlStoreOp>(loc, zero, backwardHt, IVs);
      else {
        Value h = rewriter.create<KrnlLoadOp>(loc, initialH, initialIVs);
        rewriter.create<KrnlStoreOp>(loc, h, backwardHt, IVs);
      }
      if (!onlyHidden) {
        if (isNoneType(initialC))
          rewriter.create<KrnlStoreOp>(loc, zero, backwardCt, IVs);
        else {
          Value c = rewriter.create<KrnlLoadOp>(loc, initialC, initialIVs);
          rewriter.create<KrnlStoreOp>(loc, c, backwardCt, IVs);
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
  Value alloc;
  if (!isNoneType(output)) {
    MemRefType memRefType = convertToMemRefType(output.getType());
    if (hasAllConstantDimensions(memRefType))
      alloc = insertAllocAndDealloc(memRefType, loc, rewriter, insertDealloc);
    else {
      auto memRefShape = memRefType.getShape();
      SmallVector<Value, 2> allocOperands;
      if (memRefShape[0] < 0) {
        // Get num_directions from W.
        auto dim = rewriter.create<memref::DimOp>(loc, W, 0);
        allocOperands.emplace_back(dim);
      }
      if (memRefShape[1] < 0) {
        // Get batch_size from X.
        auto dim = rewriter.create<memref::DimOp>(loc, X, 1);
        allocOperands.emplace_back(dim);
      }
      if (memRefShape[2] < 0) {
        // Get hidden_size from R.
        auto dim = rewriter.create<memref::DimOp>(loc, R, 2);
        allocOperands.emplace_back(dim);
      }
      alloc = rewriter.create<memref::AllocOp>(loc, memRefType, allocOperands);
      if (insertDealloc) {
        auto *parentBlock = alloc.getDefiningOp()->getBlock();
        auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
        dealloc.getOperation()->moveBefore(&parentBlock->back());
      }
    }
  } else {
    alloc = output;
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

void stateToOutputForHiddenOrCell(ConversionPatternRewriter &rewriter,
    Location loc, Value forwardVal, Value reverseVal, StringRef direction,
    Value output) {
  // Scope for krnl EDSC ops
  using namespace mlir::edsc;
  // Scope for std EDSC ops
  using namespace edsc::intrinsics;
  ScopedContext scope(rewriter, loc);

  if (direction == FORWARD || direction == REVERSE) {
    Value val = (direction == FORWARD) ? forwardVal : reverseVal;
    Value sizeInBytes = getDynamicMemRefSizeInBytes(rewriter, loc, val);
    rewriter.create<KrnlMemcpyOp>(loc, output, val, sizeInBytes);
  } else { // BIDIRECTIONAL
    MemRefBoundsCapture bounds(forwardVal);
    Value zero = std_constant_index(0);
    Value one = std_constant_index(1);
    ValueRange loops = krnl_define_loop(2);
    krnl_iterate(
        loops, bounds.getLbs(), bounds.getUbs(), {}, [&](ValueRange args) {
          ValueRange indices = krnl_get_induction_var_value(loops);
          Value b(indices[0]), h(indices[1]);
          // Forward.
          Value val = krnl_load(forwardVal, {b, h});
          krnl_store(val, output, {zero, b, h});
          // Reverse.
          val = krnl_load(reverseVal, {b, h});
          krnl_store(val, output, {one, b, h});
        });
  }
}

void storeIntermediateState(ConversionPatternRewriter &rewriter, Location loc,
    Value state, Value output) {
  Value sizeInBytes = getDynamicMemRefSizeInBytes(rewriter, loc, state);
  rewriter.create<KrnlMemcpyOp>(loc, output, state, sizeInBytes);
}

void storeIntermediateStateToAllH(ConversionPatternRewriter &rewriter,
    Location loc, Value Ht, Value sequenceIV, Value directionIV, Value allH) {
  // Scope for krnl EDSC ops
  using namespace mlir::edsc;
  // Scope for std EDSC ops
  using namespace edsc::intrinsics;
  ScopedContext scope(rewriter, loc);

  MemRefBoundsCapture bounds(Ht);
  ValueRange loops = krnl_define_loop(2);
  krnl_iterate(
      loops, bounds.getLbs(), bounds.getUbs(), {}, [&](ValueRange args) {
        ValueRange indices = krnl_get_induction_var_value(loops);
        Value s(sequenceIV), d(directionIV), b(indices[0]), h(indices[1]);
        Value val = krnl_load(Ht, {b, h});
        krnl_store(val, allH, {s, d, b, h});
      });
}

// Apply an activation function on a given operand.
Value applyActivation(ConversionPatternRewriter &rewriter, Location loc,
    RNNActivation activation, Value operand) {
  Value res;

  // TODO: remove this once all implementations has changed.
  bool isScalar = !operand.getType().isa<ShapedType>();

  MemRefType memRefType;
  Value alloc;
  if (isScalar) {
    memRefType = MemRefType::get({}, operand.getType(), {}, 0);
    alloc = rewriter.create<memref::AllocOp>(loc, memRefType);
    rewriter.create<KrnlStoreOp>(loc, operand, alloc, ArrayRef<Value>{});
  } else {
    memRefType = operand.getType().cast<MemRefType>();
    alloc = operand;
  }

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
    res = rewriter.create<ONNXReluOp>(loc, memRefType, alloc);
  else if (activation.name.equals_lower("tanh"))
    res = rewriter.create<ONNXTanhOp>(loc, memRefType, alloc);
  else if (activation.name.equals_lower("sigmoid"))
    res = rewriter.create<ONNXSigmoidOp>(loc, memRefType, alloc);
  else if (activation.name.equals_lower("affine"))
    llvm_unreachable("Unsupported activation");
  else if (activation.name.equals_lower("leakyrelu"))
    res = rewriter.create<ONNXLeakyReluOp>(loc, memRefType, alloc, attributes);
  else if (activation.name.equals_lower("thresholdedrelu"))
    res = rewriter.create<ONNXThresholdedReluOp>(
        loc, memRefType, alloc, attributes);
  else if (activation.name.equals_lower("scaledtanh"))
    llvm_unreachable("Unsupported activation");
  else if (activation.name.equals_lower("hardsigmoid"))
    res =
        rewriter.create<ONNXHardSigmoidOp>(loc, memRefType, alloc, attributes);
  else if (activation.name.equals_lower("elu"))
    res = rewriter.create<ONNXEluOp>(loc, memRefType, alloc, attributes);
  else if (activation.name.equals_lower("softsign"))
    res = rewriter.create<ONNXSoftsignOp>(loc, memRefType, alloc);
  else if (activation.name.equals_lower("softplus"))
    res = rewriter.create<ONNXSoftplusOp>(loc, memRefType, alloc);
  else
    llvm_unreachable("Unsupported activation");

  if (isScalar)
    res = rewriter.create<KrnlLoadOp>(loc, res);

  return res;
}

/// Get a slice of X at a specific timestep.
Value emitXSliceAt(ConversionPatternRewriter &rewriter, Location loc, Value X,
    Value timestepIV) {

  Value sliceX;

  int64_t batchSize = dimAt(X, 1);
  int64_t inputSize = dimAt(X, 2);
  Value batchSizeVal =
      getDimOrConstant(rewriter, loc, X, 1, rewriter.getIndexType());
  Value inputSizeVal =
      getDimOrConstant(rewriter, loc, X, 2, rewriter.getIndexType());

  auto elementType = X.getType().cast<ShapedType>().getElementType();
  MemRefType sliceXType = MemRefType::get({batchSize, inputSize}, elementType);
  if (hasAllConstantDimensions(sliceXType))
    sliceX = insertAllocAndDealloc(sliceXType, loc, rewriter, true);
  else {
    auto memRefShape = sliceXType.getShape();
    SmallVector<Value, 2> allocOperands;
    if (memRefShape[0] < 0) {
      allocOperands.emplace_back(batchSizeVal);
    }
    if (memRefShape[1] < 0) {
      allocOperands.emplace_back(inputSizeVal);
    }
    sliceX = rewriter.create<memref::AllocOp>(loc, sliceXType, allocOperands);
    // FIXME: deallocate
    // auto *parentBlock = X.getDefiningOp()->getBlock();
    // auto dealloc = rewriter.create<DeallocOp>(loc, X);
    // dealloc.getOperation()->moveBefore(&parentBlock->back());
  }
  BuildKrnlLoop xtLoops(rewriter, loc, 2);
  xtLoops.createDefineOp();
  xtLoops.pushBounds(0, batchSizeVal);
  xtLoops.pushBounds(0, inputSizeVal);
  xtLoops.createIterateOp();
  {
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(xtLoops.getIterateBlock());
    auto batchIV = xtLoops.getInductionVar(0);
    auto inputIV = xtLoops.getInductionVar(1);
    Value val = rewriter.create<KrnlLoadOp>(
        loc, X, ArrayRef<Value>{timestepIV, batchIV, inputIV});
    rewriter.create<KrnlStoreOp>(
        loc, val, sliceX, ArrayRef<Value>{batchIV, inputIV});
  }
  return sliceX;
}
