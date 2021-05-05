/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- GRU.cpp - Lowering GRU Op --------------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX GRU Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/RNN/RNNBase.hpp"

using namespace mlir;

struct GruState {
  Value allH;
  Value ht;
  bool linearBeforeReset;
};

struct GruActivationPack {
  RNNActivation f;
  RNNActivation g;
};

template <>
bool hasAllNoneOutput<ONNXGRUOp>(ONNXGRUOp *op) {
  return (isNoneType(op->Y()) && isNoneType(op->Y_h()));
}

template <>
std::tuple<GruActivationPack, GruActivationPack>
getActivationPack<ONNXGRUOp, GruActivationPack>(ONNXGRUOp *op) {
  auto direction = op->direction();
  auto activations = op->activations();
  auto activationAlpha = op->activation_alpha();
  auto activationBeta = op->activation_beta();

  GruActivationPack activationForward, activationReverse;

  // Get activation function name.
  // Default forward functions
  activationForward.f.name = "sigmoid";
  activationForward.g.name = "tanh";
  // Default backward functions
  activationReverse.f.name = "sigmoid";
  activationReverse.g.name = "tanh";
  if (activations) {
    ArrayAttr activationArrAttr = activations.getValue();
    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      // Forward activations.
      if (activationArrAttr.size() > 0) {
        activationForward.f.name =
            activationArrAttr[0].cast<StringAttr>().getValue();
      }
      if (activationArrAttr.size() > 1) {
        activationForward.g.name =
            activationArrAttr[1].cast<StringAttr>().getValue();
      }
    }

    // Reverse activations.
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      int startIndex = (direction == REVERSE) ? 0 : 2;
      if (activationArrAttr.size() > startIndex) {
        activationReverse.f.name =
            activationArrAttr[startIndex].cast<StringAttr>().getValue();
      }
      if (activationArrAttr.size() > startIndex + 1) {
        activationReverse.g.name =
            activationArrAttr[startIndex + 1].cast<StringAttr>().getValue();
      }
    }
  }

  // Get alpha attributes.
  if (activationAlpha) {
    ArrayAttr activationArrAttr = activationAlpha.getValue();
    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      // Forward activations.
      if (activationArrAttr.size() > 0) {
        activationForward.f.alpha = activationArrAttr[0].cast<FloatAttr>();
      }
      if (activationArrAttr.size() > 1) {
        activationForward.g.alpha = activationArrAttr[1].cast<FloatAttr>();
      }
    }

    // Reverse activations.
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      int startIndex = (direction == REVERSE) ? 0 : 2;
      if (activationArrAttr.size() > startIndex) {
        activationReverse.f.alpha =
            activationArrAttr[startIndex].cast<FloatAttr>();
      }
      if (activationArrAttr.size() > startIndex + 1) {
        activationReverse.g.alpha =
            activationArrAttr[startIndex + 1].cast<FloatAttr>();
      }
    }
  }

  // Get beta attributes.
  if (activationBeta) {
    ArrayAttr activationArrAttr = activationBeta.getValue();
    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      // Forward activations.
      if (activationArrAttr.size() > 0) {
        activationForward.f.beta = activationArrAttr[0].cast<FloatAttr>();
      }
      if (activationArrAttr.size() > 1) {
        activationForward.g.beta = activationArrAttr[1].cast<FloatAttr>();
      }
    }

    // Reverse activations.
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      int startIndex = (direction == REVERSE) ? 0 : 2;
      if (activationArrAttr.size() > startIndex) {
        activationReverse.f.beta =
            activationArrAttr[startIndex].cast<FloatAttr>();
      }
      if (activationArrAttr.size() > startIndex + 1) {
        activationReverse.g.beta =
            activationArrAttr[startIndex + 1].cast<FloatAttr>();
      }
    }
  }

  return std::make_tuple(activationForward, activationReverse);
}

template <>
GruState allocAndInitializeStates<ONNXGRUOp, GruState>(
    ConversionPatternRewriter &rewriter, Location loc, ONNXGRUOp *op,
    typename ONNXGRUOp::Adaptor operandAdaptor) {
  GruState state;

  // Insert allocation and deallocation for the results of this operation.
  // Y :: [seq_length, num_directions, batch_size, hidden_size]
  state.allH = allocAllHidden(rewriter, loc, operandAdaptor.X(),
      operandAdaptor.W(), operandAdaptor.R(), op->Y(),
      checkInsertDealloc(op->getOperation(), 0));
  // Y_h :: [num_directions, batch_size, hidden_size]
  state.ht = allocHiddenOrCell(rewriter, loc, operandAdaptor.X(),
      operandAdaptor.W(), operandAdaptor.R(), op->Y_h(),
      checkInsertDealloc(op->getOperation(), 1));

  // Initialize ht.
  Value noneValue;
  initializeHiddenAndCell(rewriter, loc, state.ht, noneValue,
      operandAdaptor.initial_h(), noneValue,
      operandAdaptor.X().getType().cast<MemRefType>().getElementType(),
      /*onlyHidden=*/true);

  // Obtain the value of 'linear_before_reset' attribute.
  int64_t linearBeforeResetAttr = op->linear_before_reset();
  if (linearBeforeResetAttr == 0)
    state.linearBeforeReset = false;
  else
    state.linearBeforeReset = true;
  return state;
}

template <>
void calculateState<ONNXGRUOp, GruState, GruActivationPack>(
    ConversionPatternRewriter &rewriter, Location loc,
    typename ONNXGRUOp::Adaptor operandAdaptor, GruState state,
    GruActivationPack activationPack, Value directionIV, Value sequenceIV) {

  // GRU has 3 gates: Update, Reset, and Hidden.
  const int GATES = 3;

  bool hasBiasForInput = false;
  if (!isNoneType(operandAdaptor.B()))
    hasBiasForInput = true;

  // Prepare dimensions.
  auto hiddenDimSize = dimAt(operandAdaptor.R(), 2);
  Value hiddenDimVal =
      emitConstantOp(rewriter, loc, rewriter.getIndexType(), hiddenDimSize);

  auto elementType =
      operandAdaptor.X().getType().cast<ShapedType>().getElementType();

  // Prepare AffineMap to access the bias tensor.
  AffineMap accessByOffsetMap;
  {
    AffineExpr iv = rewriter.getAffineDimExpr(0);
    AffineExpr index = rewriter.getAffineSymbolExpr(0);
    AffineExpr size = rewriter.getAffineSymbolExpr(1);
    AffineExpr accessByOffsetExpr = index * size + iv;
    accessByOffsetMap = AffineMap::get(1, 2, accessByOffsetExpr);
  }

  // Prepare constant indices.
  SmallVector<Value, GATES> constantIndices;
  for (int i = 0; i < 2 * GATES; i++)
    constantIndices.emplace_back(
        emitConstantOp(rewriter, loc, rewriter.getIndexType(), i));

  // Equations (Default: f=Sigmoid, g=Tanh):"
  // zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)"
  // rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)"
  // if (linearBeforeReset)
  //   ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh)
  // else
  //   ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh)
  // Ht = (1 - zt) (.) ht + zt (.) Ht-1"
  //
  // Shape information:
  // Xt : [seq_length, batch_size, input_size]
  // W[zrh] : [num_directions, hidden_size, input_size]
  // R[zrh] : [num_directions, hidden_size, hidden_size]
  // Ht : [num_directions, batch_size, hidden_size]
  // Wb[zrh] : [num_directions, hidden_size]
  // Rb[zrh] : [num_directions, hidden_size]
  //
  // The following code will emit loops as follows:
  //     for b in 0 .. BatchDimSize
  //       for h in 0 .. HiddenDimSize {
  //         for i in 0 .. InputDimSize
  //           compute Xt*(Wz^T), Xt*(Wr^T), Xt*(Wh^T)
  //         for i in 0 .. HiddenDimSize {
  //           compute Ht-1*(Rz^T), Ht-1*(Rr^T),
  //           if (linearBeforeReset)
  //             Ht-1*(Rh^T)
  //         }
  //         compute zt, rt
  //         if (!linearBeforeReset)
  //           compute (rt (.) Ht-1)
  //         else
  //           compute ht
  //       }
  //     if (!linearBeforeReset)
  //       for b in 0 .. BatchDimSize
  //         for h in 0 .. HiddenDimSize {
  //           for i in 0 .. InputDimSize {
  //             compute (rt (.) Ht-1)*(Rh^T)
  //     for b in 0 .. BatchDimSize
  //       for h in 0 .. HiddenDimSize {
  //         if (!linearBeforeReset)
  //           compute ht
  //         compute Ht
  //         update the hidden state with the new state Ht.
  //
  // The reason to have two loops at the top level is to avoid updating any
  // element of the hidden state while computing Ht-1*(R[zrh]^T).

  // Create temporary buffers for ht and zt.
  // These tensors have shape of [batch_size, hidden_size],
  MemRefType bufMemRefType = MemRefType::get(
      {dimAt(operandAdaptor.X(), 1), hiddenDimSize}, elementType);
  bool staticDimensions = hasAllConstantDimensions(bufMemRefType);
  Value htMemRef, ztMemRef;
  if (staticDimensions) {
    htMemRef = insertAllocAndDealloc(bufMemRefType, loc, rewriter, false);
    ztMemRef = insertAllocAndDealloc(bufMemRefType, loc, rewriter, false);
  } else {
    // Hidden size is a constant, so the batch size must be unknown here.
    Value batchSizeDim =
        rewriter.create<memref::DimOp>(loc, operandAdaptor.X(), 1).getResult();
    htMemRef = rewriter.create<memref::AllocOp>(
        loc, bufMemRefType, llvm::makeArrayRef({batchSizeDim}));
    ztMemRef = rewriter.create<memref::AllocOp>(
        loc, bufMemRefType, llvm::makeArrayRef({batchSizeDim}));
  }

  // In case of not linearBeforeReset, create temporary buffers for Xt*(Wh^t),
  // (rt . Ht-1), (rt . Ht-1)*(Rh^t), used for computing ht.
  // These tensors have shape of [batch_size, hidden_size].
  Value xwHMemRef, rhrHMemRef, rhMemRef;
  if (!state.linearBeforeReset) {
    if (staticDimensions) {
      xwHMemRef = insertAllocAndDealloc(bufMemRefType, loc, rewriter, false);
      rhMemRef = insertAllocAndDealloc(bufMemRefType, loc, rewriter, false);
      rhrHMemRef = insertAllocAndDealloc(bufMemRefType, loc, rewriter, false);
    } else {
      // Hidden size is a constant, so the batch size must be unknown here.
      Value batchSizeDim =
          rewriter.create<memref::DimOp>(loc, operandAdaptor.X(), 1).getResult();
      xwHMemRef = rewriter.create<memref::AllocOp>(
          loc, bufMemRefType, llvm::makeArrayRef({batchSizeDim}));
      rhMemRef = rewriter.create<memref::AllocOp>(
          loc, bufMemRefType, llvm::makeArrayRef({batchSizeDim}));
      rhrHMemRef = rewriter.create<memref::AllocOp>(
          loc, bufMemRefType, llvm::makeArrayRef({batchSizeDim}));
    }
  }

  // Emit instructions for computing rt and zt.
  BuildKrnlLoop matrixLoops(rewriter, loc, 2);
  matrixLoops.createDefineOp();
  // Batch size dim.
  matrixLoops.pushBounds(0, operandAdaptor.X(), 1);
  // Hidden size dim.
  matrixLoops.pushBounds(0, hiddenDimSize);
  matrixLoops.createIterateOp();
  auto ipMatrixLoops = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(matrixLoops.getIterateBlock());
  {
    auto batchIV = matrixLoops.getInductionVar(0);
    auto hiddenIV = matrixLoops.getInductionVar(1);

    // IVs for the hidden state tensor.
    SmallVector<Value, 3> stateIVs = {directionIV, batchIV, hiddenIV};
    // IVs for the matrix multiplication results.
    SmallVector<Value, 2> mIVs = {batchIV, hiddenIV};

    // IVs for the bias tensors for W and R.
    SmallVector<SmallVector<Value, 2>, GATES> wbZRHIVs, rbZRHIVs;
    { // Compute IVs for bias.
      // Bias [Wb[zrh], Rb[zrh]] :: [num_directions, 2*GATES*hidden_size]
      if (hasBiasForInput) {
        // Wb[zrh]
        for (unsigned i = 0; i < GATES; ++i) {
          Value wHiddenIV =
              rewriter.create<AffineApplyOp>(loc, accessByOffsetMap,
                  std::vector<Value>{/*iv=*/hiddenIV,
                      /*index=*/constantIndices[i], /*size=*/hiddenDimVal});
          wbZRHIVs.emplace_back(SmallVector<Value, 2>{directionIV, wHiddenIV});
        }
        // Rb[zrh]
        for (unsigned i = GATES; i < 2 * GATES; ++i) {
          Value rHiddenIV =
              rewriter.create<AffineApplyOp>(loc, accessByOffsetMap,
                  std::vector<Value>{/*iv=*/hiddenIV,
                      /*index=*/constantIndices[i], /*size=*/hiddenDimVal});
          rbZRHIVs.emplace_back(SmallVector<Value, 2>{directionIV, rHiddenIV});
        }
      }
    }

    // Emit instructions for matrix multiplications:
    //   Xt*(Wz^T), Ht-1*(Rz^T),
    //   Xt*(Wr^T), Ht-1*(Rr^T),
    //   Xt*(Wh^T),
    //   if (linearBeforeReset)
    //     Ht-1*(Rh^T)

    // Allocate and initialize memory for matrix multiplication results.
    SmallVector<Value, GATES> xwZRH, hrZRH;
    Value zero = emitConstantOp(rewriter, loc, elementType, 0);
    MemRefType scalarMemRefType = MemRefType::get({}, elementType, {}, 0);
    for (unsigned i = 0; i < GATES; ++i) {
      // in case of not linearBeforeReset:
      // - do not need to allocate a local buffer here for Xt*(Wh^T), we will
      // store to a global buffer.
      // - Ht-1*(Rh^T) is unnecessary
      if (!state.linearBeforeReset && (i == GATES - 1))
        continue;
      Value xwAlloc = rewriter.create<memref::AllocOp>(loc, scalarMemRefType);
      rewriter.create<KrnlStoreOp>(loc, zero, xwAlloc, ArrayRef<Value>{});
      xwZRH.emplace_back(xwAlloc);
      Value hrAlloc = rewriter.create<memref::AllocOp>(loc, scalarMemRefType);
      rewriter.create<KrnlStoreOp>(loc, zero, hrAlloc, ArrayRef<Value>{});
      hrZRH.emplace_back(hrAlloc);
    }

    // Initialize the global buffer for Xt*(Wh^T).
    if (!state.linearBeforeReset)
      rewriter.create<KrnlStoreOp>(loc, zero, xwHMemRef, mIVs);

    { // Emit instructions for Xt*(Wz^T), Xt*(Wr^T), Xt*(Wh^T)
      // input_size is the reduction dimension.
      BuildKrnlLoop reductionLoops(rewriter, loc, 1);
      reductionLoops.createDefineOp();
      // Input size dim.
      reductionLoops.pushBounds(0, operandAdaptor.X(), 2);
      reductionLoops.createIterateOp();

      auto ipReductionLoops = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointToStart(reductionLoops.getIterateBlock());
      {
        auto reductionIV = reductionLoops.getInductionVar(0);
        // Prepare IVs for accessing the input tensor and parameters.
        SmallVector<Value, 3> xIVs;
        SmallVector<SmallVector<Value, 3>, GATES> wZRHIVs;

        // X :: [seq_length, batch_size, input_size]
        xIVs = {sequenceIV, batchIV, reductionIV};

        // W[zrh] :: [num_directions, GATES*hidden_size, input_size]
        for (unsigned i = 0; i < GATES; ++i) {
          SmallVector<Value, 3> wIVs;
          Value wHiddenIV = rewriter.create<AffineApplyOp>(loc,
              accessByOffsetMap,
              std::vector<Value>{hiddenIV, constantIndices[i], hiddenDimVal});
          wIVs = {directionIV, wHiddenIV, reductionIV};
          wZRHIVs.emplace_back(wIVs);
        }

        Value loadX =
            rewriter.create<KrnlLoadOp>(loc, operandAdaptor.X(), xIVs);
        for (unsigned i = 0; i < GATES; ++i) {
          // Xt * W[zrh]
          Value loadW =
              rewriter.create<KrnlLoadOp>(loc, operandAdaptor.W(), wZRHIVs[i]);
          Value xwVal = rewriter.create<MulFOp>(loc, loadX, loadW);
          Value loadXW;
          if (!state.linearBeforeReset && (i == GATES - 1))
            // in case of not linearBeforeReset, load from the global buffer.
            loadXW = rewriter.create<KrnlLoadOp>(loc, xwHMemRef, mIVs);
          else
            loadXW = rewriter.create<KrnlLoadOp>(loc, xwZRH[i]);
          Value nextXW = rewriter.create<AddFOp>(loc, loadXW, xwVal);
          if (!state.linearBeforeReset && (i == GATES - 1))
            // in case of not linearBeforeReset, store to the global buffer.
            rewriter.create<KrnlStoreOp>(loc, nextXW, xwHMemRef, mIVs);
          else
            rewriter.create<KrnlStoreOp>(
                loc, nextXW, xwZRH[i], ArrayRef<Value>{});
        }
      }
      rewriter.restoreInsertionPoint(ipReductionLoops);
    }

    { // Emit instructions for Ht-1*(Rz^T), Ht-1*(Rr^T), and Ht-1 * R[zrh]
      // hidden_size is the reduction dimension.
      BuildKrnlLoop reductionLoops(rewriter, loc, 1);
      reductionLoops.createDefineOp();
      reductionLoops.pushBounds(0, hiddenDimSize);
      reductionLoops.createIterateOp();

      auto ipReductionLoops = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointToStart(reductionLoops.getIterateBlock());
      {
        auto reductionIV = reductionLoops.getInductionVar(0);
        // Prepare IVs for accessing the input tensor and parameters.
        SmallVector<Value, 3> hIVs;
        SmallVector<SmallVector<Value, 3>, GATES> rZRHIVs;

        // H :: [num_directions, batch_size, hidden_size]
        hIVs = {directionIV, batchIV, reductionIV};

        // R[zrh] :: [num_directions, GATES*hidden_size, hidden_size]
        for (unsigned i = 0; i < GATES; ++i) {
          SmallVector<Value, 3> rIVs;
          Value rHiddenIV = rewriter.create<AffineApplyOp>(loc,
              accessByOffsetMap,
              std::vector<Value>{hiddenIV, constantIndices[i], hiddenDimVal});
          rIVs = {directionIV, rHiddenIV, reductionIV};
          rZRHIVs.emplace_back(rIVs);
        }

        // Ht-1 * R[zrh]
        Value loadH = rewriter.create<KrnlLoadOp>(loc, state.ht, hIVs);
        for (unsigned i = 0; i < GATES; ++i) {
          // Ht-1*(Rh^T) is unnecessary if not linearBeforeReset
          if (!state.linearBeforeReset && (i == GATES - 1))
            continue;
          Value loadR =
              rewriter.create<KrnlLoadOp>(loc, operandAdaptor.R(), rZRHIVs[i]);
          Value hrVal = rewriter.create<MulFOp>(loc, loadH, loadR);
          Value loadHR = rewriter.create<KrnlLoadOp>(loc, hrZRH[i]);
          Value nextHR = rewriter.create<AddFOp>(loc, loadHR, hrVal);
          rewriter.create<KrnlStoreOp>(
              loc, nextHR, hrZRH[i], ArrayRef<Value>{});
        }
      }
      rewriter.restoreInsertionPoint(ipReductionLoops);
    }

    // zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
    Value loadXWZ = rewriter.create<KrnlLoadOp>(loc, xwZRH[0]);
    Value loadHRZ = rewriter.create<KrnlLoadOp>(loc, hrZRH[0]);
    Value zt = rewriter.create<AddFOp>(loc, loadXWZ, loadHRZ);
    if (hasBiasForInput) {
      Value loadWB =
          rewriter.create<KrnlLoadOp>(loc, operandAdaptor.B(), wbZRHIVs[0]);
      zt = rewriter.create<AddFOp>(loc, zt, loadWB);
      Value loadRB =
          rewriter.create<KrnlLoadOp>(loc, operandAdaptor.B(), rbZRHIVs[0]);
      zt = rewriter.create<AddFOp>(loc, zt, loadRB);
    }
    zt = applyActivation(rewriter, loc, activationPack.f, zt);
    rewriter.create<KrnlStoreOp>(loc, zt, ztMemRef, mIVs);

    // rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
    Value loadXWR = rewriter.create<KrnlLoadOp>(loc, xwZRH[1]);
    Value loadHRR = rewriter.create<KrnlLoadOp>(loc, hrZRH[1]);
    Value rt = rewriter.create<AddFOp>(loc, loadXWR, loadHRR);
    if (hasBiasForInput) {
      Value loadWB =
          rewriter.create<KrnlLoadOp>(loc, operandAdaptor.B(), wbZRHIVs[1]);
      rt = rewriter.create<AddFOp>(loc, rt, loadWB);
      Value loadRB =
          rewriter.create<KrnlLoadOp>(loc, operandAdaptor.B(), rbZRHIVs[1]);
      rt = rewriter.create<AddFOp>(loc, rt, loadRB);
    }
    rt = applyActivation(rewriter, loc, activationPack.f, rt);

    if (!state.linearBeforeReset) {
      // compute and store (rt . Ht-1) to a global buffer.
      Value loadH = rewriter.create<KrnlLoadOp>(loc, state.ht, stateIVs);
      Value rtHt = rewriter.create<MulFOp>(loc, rt, loadH);
      rewriter.create<KrnlStoreOp>(loc, rtHt, rhMemRef, mIVs);
    } else {
      // compute ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh)
      Value ht = rewriter.create<KrnlLoadOp>(loc, xwZRH[2]);
      Value linear = rewriter.create<KrnlLoadOp>(loc, hrZRH[2]);
      if (hasBiasForInput) {
        Value loadRB =
            rewriter.create<KrnlLoadOp>(loc, operandAdaptor.B(), rbZRHIVs[2]);
        linear = rewriter.create<AddFOp>(loc, linear, loadRB);
      }
      Value reset = rewriter.create<MulFOp>(loc, rt, linear);
      ht = rewriter.create<AddFOp>(loc, ht, reset);
      if (hasBiasForInput) {
        Value loadWB =
            rewriter.create<KrnlLoadOp>(loc, operandAdaptor.B(), wbZRHIVs[2]);
        ht = rewriter.create<AddFOp>(loc, ht, loadWB);
      }
      ht = applyActivation(rewriter, loc, activationPack.g, ht);
      rewriter.create<KrnlStoreOp>(loc, ht, htMemRef, mIVs);
    }

    // Deallocate the temporary results of matrix multiplications.
    for (Value v : xwZRH)
      rewriter.create<memref::DeallocOp>(loc, v);
    for (Value v : hrZRH)
      rewriter.create<memref::DeallocOp>(loc, v);
  }
  rewriter.restoreInsertionPoint(ipMatrixLoops);

  // Emit instructions for computing (rt (.) Ht-1)*(Rh^T) in case of not
  // LinearBeforeReset.
  if (!state.linearBeforeReset) {
    BuildKrnlLoop matrixLoops(rewriter, loc, 2);
    matrixLoops.createDefineOp();
    // Batch size dim.
    matrixLoops.pushBounds(0, operandAdaptor.X(), 1);
    // Hidden size dim.
    matrixLoops.pushBounds(0, hiddenDimSize);
    matrixLoops.createIterateOp();
    auto ipMatrixLoops = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(matrixLoops.getIterateBlock());
    {
      auto batchIV = matrixLoops.getInductionVar(0);
      auto hiddenIV = matrixLoops.getInductionVar(1);

      // IVs for the matrix multiplication results.
      SmallVector<Value, 2> mIVs = {batchIV, hiddenIV};

      // IVs to access Rh.
      Value rOffsetIV = rewriter.create<AffineApplyOp>(loc, accessByOffsetMap,
          std::vector<Value>{hiddenIV, constantIndices[2], hiddenDimVal});

      Value zero = emitConstantOp(rewriter, loc, elementType, 0);
      rewriter.create<KrnlStoreOp>(loc, zero, rhrHMemRef, mIVs);
      {
        // Emit instructions for 'rtHt*(Rh^T)'.
        // hidden_size is the reduction dimension.
        BuildKrnlLoop reductionLoops(rewriter, loc, 1);
        reductionLoops.createDefineOp();
        // Hidden size dim.
        reductionLoops.pushBounds(0, hiddenDimSize);
        reductionLoops.createIterateOp();

        auto ipReductionLoops = rewriter.saveInsertionPoint();
        rewriter.setInsertionPointToStart(reductionLoops.getIterateBlock());
        {
          auto reductionIV = reductionLoops.getInductionVar(0);
          // rtHt :: [batch_size, hidden_size]
          SmallVector<Value, 3> rhIVs = {batchIV, reductionIV};
          // R[zrh] :: [num_directions, 3*hidden_size, hidden_size]
          SmallVector<Value, 3> rIVs = {directionIV, rOffsetIV, reductionIV};

          // 'rtHt*(Rh^T)'
          Value loadRtHt = rewriter.create<KrnlLoadOp>(loc, rhMemRef, rhIVs);
          Value loadR =
              rewriter.create<KrnlLoadOp>(loc, operandAdaptor.R(), rIVs);
          Value rhrVal = rewriter.create<MulFOp>(loc, loadRtHt, loadR);
          Value loadRHR = rewriter.create<KrnlLoadOp>(loc, rhrHMemRef, mIVs);
          Value nextRHR = rewriter.create<AddFOp>(loc, loadRHR, rhrVal);
          rewriter.create<KrnlStoreOp>(loc, nextRHR, rhrHMemRef, mIVs);
        }
        rewriter.restoreInsertionPoint(ipReductionLoops);
      }
    }
    rewriter.restoreInsertionPoint(ipMatrixLoops);
  }

  // Emit instructions for computing Ht.
  BuildKrnlLoop stateLoops(rewriter, loc, 2);
  stateLoops.createDefineOp();
  // Batch size dim.
  stateLoops.pushBounds(0, operandAdaptor.X(), 1);
  // Hidden size dim.
  stateLoops.pushBounds(0, hiddenDimSize);
  stateLoops.createIterateOp();
  auto ipStateLoops = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(stateLoops.getIterateBlock());
  {
    auto batchIV = stateLoops.getInductionVar(0);
    auto hiddenIV = stateLoops.getInductionVar(1);
    SmallVector<Value, 3> stateIVs = {directionIV, batchIV, hiddenIV};
    SmallVector<Value, 2> mIVs = {batchIV, hiddenIV};

    Value ht;
    if (!state.linearBeforeReset) {
      //   ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh)
      ht = rewriter.create<KrnlLoadOp>(loc, xwHMemRef, mIVs);
      Value linear = rewriter.create<KrnlLoadOp>(loc, rhrHMemRef, mIVs);
      ht = rewriter.create<AddFOp>(loc, ht, linear);
      if (hasBiasForInput) {
        Value rHiddenIV = rewriter.create<AffineApplyOp>(loc, accessByOffsetMap,
            std::vector<Value>{/*iv=*/hiddenIV,
                /*index=*/constantIndices[5], /*size=*/hiddenDimVal});
        Value loadRB = rewriter.create<KrnlLoadOp>(loc, operandAdaptor.B(),
            SmallVector<Value, 2>{directionIV, rHiddenIV});
        ht = rewriter.create<AddFOp>(loc, ht, loadRB);

        Value wHiddenIV = rewriter.create<AffineApplyOp>(loc, accessByOffsetMap,
            std::vector<Value>{/*iv=*/hiddenIV,
                /*index=*/constantIndices[2], /*size=*/hiddenDimVal});
        Value loadWB = rewriter.create<KrnlLoadOp>(loc, operandAdaptor.B(),
            SmallVector<Value, 2>{directionIV, wHiddenIV});
        ht = rewriter.create<AddFOp>(loc, ht, loadWB);
      }
      ht = applyActivation(rewriter, loc, activationPack.g, ht);
    } else
      ht = rewriter.create<KrnlLoadOp>(loc, htMemRef, mIVs);

    // Ht = (1 - zt) (.) ht + zt (.) Ht-1
    Value zt = rewriter.create<KrnlLoadOp>(loc, ztMemRef, mIVs);
    Value loadH = rewriter.create<KrnlLoadOp>(loc, state.ht, stateIVs);
    Value one = emitConstantOp(rewriter, loc, elementType, 1);
    Value subZt = rewriter.create<SubFOp>(loc, one, zt);
    Value mulHt = rewriter.create<MulFOp>(loc, subZt, ht);
    Value mulH = rewriter.create<MulFOp>(loc, zt, loadH);
    Value Ht = rewriter.create<AddFOp>(loc, mulHt, mulH);
    rewriter.create<KrnlStoreOp>(loc, Ht, state.ht, stateIVs);

    // Store the current Ht if required.
    if (!isNoneType(state.allH)) {
      SmallVector<Value, 4> allHIVs{sequenceIV, directionIV, batchIV, hiddenIV};
      rewriter.create<KrnlStoreOp>(loc, Ht, state.allH, allHIVs);
    }
  }
  rewriter.restoreInsertionPoint(ipStateLoops);
  // Deallocate the temporary results.
  rewriter.create<memref::DeallocOp>(loc, htMemRef);
  rewriter.create<memref::DeallocOp>(loc, ztMemRef);
  if (!state.linearBeforeReset) {
    rewriter.create<memref::DeallocOp>(loc, xwHMemRef);
    rewriter.create<memref::DeallocOp>(loc, rhMemRef);
    rewriter.create<memref::DeallocOp>(loc, rhrHMemRef);
  }
}

template <>
void stateToOutput<ONNXGRUOp, GruState>(
    ONNXGRUOp *op, GruState state, std::vector<Value> &outputs) {
  Value noneValue;
  outputs.emplace_back((isNoneType(op->Y()) ? noneValue : state.allH));
  outputs.emplace_back((isNoneType(op->Y_h()) ? noneValue : state.ht));
}

void populateLoweringONNXGRUOpPattern(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXRNNOpLowering<ONNXGRUOp, GruState, GruActivationPack>>(
      ctx);
}
