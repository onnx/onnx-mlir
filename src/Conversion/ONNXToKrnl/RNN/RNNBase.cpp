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
  ScopedContext scope(rewriter, loc);
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
      alloc = memref_alloc(memRefType, allocOperands);
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
  ScopedContext scope(rewriter, loc);
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
    alloc = memref_alloc(memRefType, allocOperands);
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
  ScopedContext scope(rewriter, loc);
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
        krnl_store(zero, forwardHt, IVs);
      else {
        Value h = krnl_load(initialH, initialIVs);
        krnl_store(h, forwardHt, IVs);
      }
      if (!onlyHidden) {
        if (isNoneType(initialC))
          krnl_store(zero, forwardCt, IVs);
        else {
          Value c = krnl_load(initialC, initialIVs);
          krnl_store(c, forwardCt, IVs);
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
        Value h = krnl_load(initialH, initialIVs);
        krnl_store(h, reverseHt, IVs);
      }
      if (!onlyHidden) {
        if (isNoneType(initialC))
          krnl_store(zero, reverseCt, IVs);
        else {
          Value c = krnl_load(initialC, initialIVs);
          krnl_store(c, reverseCt, IVs);
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
  ScopedContext scope(rewriter, loc);
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
      alloc = memref_alloc(memRefType, allocOperands);
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
    sliceX = memref_alloc(sliceXType, allocOperands);
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

void emitFusedMatMul(ConversionPatternRewriter &rewriter, Location loc,
    MemRefType matrixType, Value A, ArrayRef<Value> Bs, Value zero,
    Value zeroVal, ArrayRef<Value> Cs) {
  ScopedContext scope(rewriter, loc);

  Type elementType = matrixType.getElementType();
  // Get bounds I, J, K.
  MemRefBoundsCapture aBounds(A), bBounds(Bs[0]);
  Value I(aBounds.ub(0)), J(bBounds.ub(1)), K(aBounds.ub(1));

  // Initialize alloc/C to zero.
  ValueRange zeroLoop = krnl_define_loop(2);
  krnl_iterate(zeroLoop, {zero, zero}, {I, J}, {}, [&](ValueRange args) {
    ValueRange indices = krnl_get_induction_var_value(zeroLoop);
    for (Value C : Cs)
      krnl_store(zeroVal, C, indices);
  });

  // Prepare for the computations.
  // 1) Define blocking, with simdization along the j axis.
  const int64_t iCacheTile(64), jCacheTile(128), kCacheTile(512);
  const int64_t iRegTile(4), jRegTile(8);

  bool unrollAndJam = true;
  // Simdize with jRegTile as the vector length.
  bool simdize = true;

  bool mustTileR = false;
  // J is hidden size which is always literal.
  int64_t jVal = matrixType.getShape()[1];
  if (jVal < jRegTile) {
    // Very small computation, give up on SIMD.
    simdize = false;
  } else if (jVal % jRegTile != 0) {
    // Unfortunately, J is not divisible by the vector length. Could try
    // to change the vector length, but right now, just go to buffering.
    mustTileR = true;
  } else {
    // Best of all world, large computation, of sizes compatible with vector
    // length.
  }

  // 2) Alloc data for tiles.
  MemRefType aTileType = MemRefType::get({iCacheTile, kCacheTile}, elementType);
  MemRefType bTileType = MemRefType::get({kCacheTile, jCacheTile}, elementType);
  MemRefType cTileType = MemRefType::get({iCacheTile, jCacheTile}, elementType);
  IntegerAttr alignAttr = rewriter.getI64IntegerAttr(BUFFER_ALIGN);
  ValueRange empty;
  Value aBuff = memref_alloc(aTileType, empty, alignAttr);
  SmallVector<Value, 4> bBuffs;
  for (unsigned int i = 0; i < Bs.size(); ++i) {
    Value bBuff = memref_alloc(bTileType, empty, alignAttr);
    bBuffs.emplace_back(bBuff);
  }
  SmallVector<Value, 4> cBuffs;
  if (mustTileR) {
    for (unsigned int i = 0; i < Cs.size(); ++i) {
      Value cBuff = memref_alloc(cTileType, empty, alignAttr);
      cBuffs.emplace_back(cBuff);
    }
  }

  // 3) introduce the loops and permute them
  // I, J, K loop.
  ValueRange origLoop = krnl_define_loop(3);
  Value ii(origLoop[0]), jj(origLoop[1]), kk(origLoop[2]);
  // Tile I.
  ValueRange iCacheBlock = krnl_block(ii, iCacheTile);
  ValueRange iRegBlock = krnl_block(iCacheBlock[1], iRegTile);
  Value ii1(iCacheBlock[0]), ii2(iRegBlock[0]), ii3(iRegBlock[1]);
  // Tile J.
  ValueRange jCacheBlock = krnl_block(jj, jCacheTile);
  ValueRange jRegBlock = krnl_block(jCacheBlock[1], jRegTile);
  Value jj1(jCacheBlock[0]), jj2(jRegBlock[0]), jj3(jRegBlock[1]);
  // Tile K.
  ValueRange kCacheBlock = krnl_block(kk, kCacheTile);
  Value kk1(kCacheBlock[0]), kk2(kCacheBlock[1]);

  // If we must tile the result R, then we put I & J in the outermost.
  // Otherwise, we follow the more traditional scheme of having J & K in the
  // outermost.
  if (mustTileR) {
    // (cache) ii1 jj1 kk1,    (reg) jj2, ii2,    (matmul) ii3, jj3, kk3
    krnl_permute({ii1, ii2, ii3, jj1, jj2, jj3, kk1, kk2},
        {/*i*/ 0, 4, 5, /*j*/ 1, 3, 6, /*k*/ 2, 7});
    // Compute: A[i, k] * b[k, j] -> R[i, j])
    krnl_iterate(
        {ii, jj}, {ii1, jj1}, {zero, zero}, {I, J}, {}, [&](ValueRange args) {
          ValueRange i1_j1_indices = krnl_get_induction_var_value({ii1, jj1});
          Value i1(i1_j1_indices[0]), j1(i1_j1_indices[1]);
          for (unsigned int n = 0; n < cBuffs.size(); ++n)
            krnl_copy_to_buffer(cBuffs[n], Cs[n], {i1, j1}, zeroVal, false);
          krnl_iterate({kk}, {kk1}, {zero}, {K}, {}, [&](ValueRange args) {
            ValueRange k1_index = krnl_get_induction_var_value({kk1});
            Value k1(k1_index[0]);
            krnl_copy_to_buffer(aBuff, A, {i1, k1}, zeroVal, false);
            for (unsigned int n = 0; n < bBuffs.size(); ++n)
              krnl_copy_to_buffer(bBuffs[n], Bs[n], {k1, j1}, zeroVal, false);
            krnl_iterate({}, {jj2, ii2}, {}, {}, {}, [&](ValueRange args) {
              ValueRange j2_i2_indices =
                  krnl_get_induction_var_value({jj2, ii2});
              Value j2(j2_i2_indices[0]), i2(j2_i2_indices[1]);
              for (unsigned int n = 0; n < bBuffs.size(); ++n) {
                krnl_matmul(aBuff, {i1, k1}, bBuffs[n], {k1, j1}, cBuffs[n],
                    {i1, j1},
                    /*loops*/ {ii3, jj3, kk2},
                    /*compute start*/ {i2, j2, k1},
                    /*ubs*/ {I, J, K},
                    /*compute tile*/ {iRegTile, jRegTile, kCacheTile},
                    /* a/b/c tiles*/ {}, {}, {}, simdize, unrollAndJam, false);
              }
            });
          });
          for (unsigned int n = 0; n < cBuffs.size(); ++n)
            krnl_copy_from_buffer(cBuffs[n], Cs[n], {i1, j1});
        });
  } else {
    // Does not have to tile the result.
    // (cache) jj1 kk1, ii1, (reg) jj2, ii2, (matmul) ii3, jj3, kk3
    krnl_permute({jj1, jj2, jj3, kk1, kk2, ii1, ii2, ii3},
        {/*j*/ 0, 3, 5, /*k*/ 1, 6, /*i*/ 2, 4, 7});
    // Compute: A[i, k] * b[k, j] -> C[i, j])
    krnl_iterate(
        {jj, kk}, {jj1, kk1}, {zero, zero}, {J, K}, {}, [&](ValueRange args) {
          ValueRange j1_k1_indices = krnl_get_induction_var_value({jj1, kk1});
          Value j1(j1_k1_indices[0]), k1(j1_k1_indices[1]);
          for (unsigned int n = 0; n < bBuffs.size(); ++n)
            krnl_copy_to_buffer(bBuffs[n], Bs[n], {k1, j1}, zeroVal, false);
          krnl_iterate({ii}, {ii1}, {zero}, {I}, {}, [&](ValueRange args) {
            ValueRange i1_index = krnl_get_induction_var_value({ii1});
            Value i1(i1_index[0]);
            krnl_copy_to_buffer(aBuff, A, {i1, k1}, zeroVal, false);
            krnl_iterate({}, {jj2, ii2}, {}, {}, {}, [&](ValueRange args) {
              ValueRange j2_i2_indices =
                  krnl_get_induction_var_value({jj2, ii2});
              Value j2(j2_i2_indices[0]), i2(j2_i2_indices[1]);
              for (unsigned int n = 0; n < bBuffs.size(); ++n)
                krnl_matmul(aBuff, {i1, k1}, bBuffs[n], {k1, j1}, Cs[n],
                    {zero, zero},
                    /*loops*/ {ii3, jj3, kk2},
                    /*compute start*/ {i2, j2, k1},
                    /*ubs*/ {I, J, K},
                    /*compute tile*/ {iRegTile, jRegTile, kCacheTile},
                    /* a/b/c tiles*/ {}, {}, {}, simdize, unrollAndJam, false);
            });
          });
        });
  }
}
