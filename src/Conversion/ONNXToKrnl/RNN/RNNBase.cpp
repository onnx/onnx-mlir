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
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;

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

/// Insert Allocate and Deallocate for the intermediate hidden or cell states.
/// Shape :: [batch_size, hidden_size]
Value allocIntermediateState(
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
        rewriter.create<KrnlStoreOp>(loc, zero, reverseHt, IVs);
      else {
        Value h = rewriter.create<KrnlLoadOp>(loc, initialH, initialIVs);
        rewriter.create<KrnlStoreOp>(loc, h, reverseHt, IVs);
      }
      if (!onlyHidden) {
        if (isNoneType(initialC))
          rewriter.create<KrnlStoreOp>(loc, zero, reverseCt, IVs);
        else {
          Value c = rewriter.create<KrnlLoadOp>(loc, initialC, initialIVs);
          rewriter.create<KrnlStoreOp>(loc, c, reverseCt, IVs);
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
  ScopedContext scope(rewriter, loc);
  Value zero = emitConstantOp(rewriter, loc, elementType, 0);
  MemRefBoundsCapture bounds(ht);
  ValueRange loops = krnl_define_loop(bounds.rank());
  krnl_iterate(
      loops, bounds.getLbs(), bounds.getUbs(), {}, [&](ValueRange args) {
        ValueRange indices = krnl_get_induction_var_value(loops);
        Value hiddenVal = zero;
        if (!isNoneType(initialH))
          hiddenVal = krnl_load(initialH, indices);
        krnl_store(hiddenVal, ht, indices);

        if (!onlyHidden) {
          Value cellVal = zero;
          if (!isNoneType(initialC))
            cellVal = krnl_load(initialC, indices);
          krnl_store(cellVal, ct, indices);
        }
      });
}

/// Store a state into the output of the RNN op.
/// The input state is 2D and the output state is 3D with '1' or '2' is
/// pretended, depending on 'direction'.
void stateToOutputForHiddenOrCell(ConversionPatternRewriter &rewriter,
    Location loc, Value forwardVal, Value reverseVal, StringRef direction,
    Value output) {
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

// Apply an activation function on a given scalar operand.
Value applyActivation(ConversionPatternRewriter &rewriter, Location loc,
    RNNActivation activation, Value operand) {
  Value res;

  bool isScalar = !operand.getType().isa<ShapedType>();
  assert(isScalar && "Not a scalar operand");

  MemRefType memRefType = MemRefType::get({}, operand.getType(), {}, 0);
  Value alloc = rewriter.create<memref::AllocaOp>(loc, memRefType);
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

  res = rewriter.create<KrnlLoadOp>(loc, res);

  return res;
}

/// Create a copy of a slice of X at a specific timestep.
/// This function is not able correctly to emit 'dealloc' for the copy since it
/// does not have enough information about the parent context. Users must
/// deallocate the copy by themselves.
Value emitXSliceAt(ConversionPatternRewriter &rewriter, Location loc, Value X,
    Value timestepIV) {
  ScopedContext scope(rewriter, loc);

  Value sliceX;

  int64_t batchSize = dimAt(X, 1);
  int64_t inputSize = dimAt(X, 2);
  auto elementType = X.getType().cast<ShapedType>().getElementType();
  MemRefType sliceXType = MemRefType::get({batchSize, inputSize}, elementType);

  // Allocate a buffer
  if (hasAllConstantDimensions(sliceXType))
    sliceX =
        insertAllocAndDealloc(sliceXType, loc, rewriter, /*deallocate=*/false);
  else {
    auto memRefShape = sliceXType.getShape();
    SmallVector<Value, 2> allocOperands;
    if (memRefShape[0] < 0) {
      Value batchSizeVal =
          getDimOrConstant(rewriter, loc, X, 1, rewriter.getIndexType());
      allocOperands.emplace_back(batchSizeVal);
    }
    if (memRefShape[1] < 0) {
      Value inputSizeVal =
          getDimOrConstant(rewriter, loc, X, 2, rewriter.getIndexType());
      allocOperands.emplace_back(inputSizeVal);
    }
    sliceX = rewriter.create<memref::AllocOp>(loc, sliceXType, allocOperands);
  }

  // Copy data from X.
  MemRefBoundsCapture bounds(sliceX);
  ValueRange loops = krnl_define_loop(2);
  krnl_iterate(
      loops, bounds.getLbs(), bounds.getUbs(), {}, [&](ValueRange args) {
        ValueRange indices = krnl_get_induction_var_value(loops);
        Value b(indices[0]), i(indices[1]);
        Value val = krnl_load(X, {timestepIV, b, i});
        krnl_store(val, sliceX, {b, i});
      });

  return sliceX;
}
