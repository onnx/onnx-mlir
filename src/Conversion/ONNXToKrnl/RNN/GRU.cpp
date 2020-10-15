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
  if (!isNoneType(op->Y())) {
    auto yMemRefType = convertToMemRefType(op->Y().getType());
    if (hasAllConstantDimensions(yMemRefType))
      state.allH = insertAllocAndDealloc(yMemRefType, loc, rewriter,
          checkInsertDealloc(op->getOperation(), 0));
    else {
      llvm_unreachable("Unsupported dynamic dimensions.");
    }
  } else {
    state.allH = op->Y();
  }

  // Y_h :: [num_directions, batch_size, hidden_size]
  if (!isNoneType(op->Y_h())) {
    auto yhMemRefType = convertToMemRefType(op->Y_h().getType());
    if (hasAllConstantDimensions(yhMemRefType))
      state.ht = insertAllocAndDealloc(yhMemRefType, loc, rewriter,
          checkInsertDealloc(op->getOperation(), 1));
    else
      llvm_unreachable("Unsupported dynamic dimensions.");
  } else {
    auto yhMemRefType = MemRefType::get(
        {dimAt(operandAdaptor.W(), 0), dimAt(operandAdaptor.X(), 1),
            dimAt(operandAdaptor.R(), 2)},
        operandAdaptor.X().getType().cast<ShapedType>().getElementType());
    state.ht = insertAllocAndDealloc(yhMemRefType, loc, rewriter, true);
  }

  // Initialize ht.
  Value zero = emitConstantOp(rewriter, loc,
      operandAdaptor.X().getType().cast<ShapedType>().getElementType(), 0);
  int nLoops = 3;
  BuildKrnlLoop initializationLoops(rewriter, loc, nLoops);
  initializationLoops.createDefineAndIterateOp(state.ht);
  auto ipInitializationLoops = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(initializationLoops.getIterateBlock());
  {
    SmallVector<Value, 4> IVs;
    for (int i = 0; i < nLoops; ++i)
      IVs.emplace_back(initializationLoops.getInductionVar(i));

    Value hiddenVal = zero;
    if (!isNoneType(operandAdaptor.initial_h()))
      hiddenVal =
          rewriter.create<AffineLoadOp>(loc, operandAdaptor.initial_h(), IVs);
    rewriter.create<AffineStoreOp>(loc, hiddenVal, state.ht, IVs);
  }
  rewriter.restoreInsertionPoint(ipInitializationLoops);

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
  auto batchDimSize = dimAt(operandAdaptor.X(), 1);
  auto inputDimSize = dimAt(operandAdaptor.X(), 2);
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

  // The following code will emit loops as follows:
  // for b in 0 .. BatchDimSize
  //   for h in 0 .. HiddenDimSize {
  //     for i in 0 .. InputDimSize {
  //       compute Xt*(Wz^T), Ht-1*(Rz^T),
  //               Xt*(Wr^T), Ht-1*(Rr^T),
  //               Xt*(Wh^T),
  //       if (linearBeforeReset)
  //         Ht-1*(Rh^T)
  //     }
  //     compute zt, rt
  //     if (!linearBeforeReset)
  //       compute RHt = (rt (.) Ht-1)
  //       for i in 0 .. InputDimSize {
  //         compute (RHt)*(Rh^T)
  //     compute ht
  //   }

  BuildKrnlLoop stateLoops(rewriter, loc, 2);
  stateLoops.createDefineOp();
  stateLoops.pushBounds(0, batchDimSize);
  stateLoops.pushBounds(0, hiddenDimSize);
  stateLoops.createIterateOp();

  rewriter.setInsertionPointToStart(stateLoops.getIterateBlock());
  {
    auto batchIV = stateLoops.getInductionVar(0);
    auto hiddenIV = stateLoops.getInductionVar(1);

    // IVs to access tensors.
    // IVs for the hidden state tensor.
    SmallVector<Value, 3> hIVs, cIVs;
    // IVs for the bias tensors for W and R.
    SmallVector<SmallVector<Value, 2>, GATES> wbZRHIVs, rbZRHIVs;

    { // Compute IVs.
      // H :: [num_directions, batch_size, hidden_size]
      hIVs = {directionIV, batchIV, hiddenIV};

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

    Value loadH = rewriter.create<AffineLoadOp>(loc, state.ht, hIVs);

    // Emit instructions for matrix multiplications:
    //   Xt*(Wz^T), Ht-1*(Rz^T),
    //   Xt*(Wr^T), Ht-1*(Rr^T),
    //   Xt*(Wh^T),
    //   if (linearBeforeReset)
    //     Ht-1*(Rh^T)

    // Allocate memory for storing matrix multiplication results.
    SmallVector<Value, GATES> xwZRH, hrZRH;
    Value zero = emitConstantOp(rewriter, loc, elementType, 0);
    MemRefType scalarMemRefType = MemRefType::get({}, elementType, {}, 0);
    for (unsigned i = 0; i < GATES; ++i) {
      Value xwAlloc = rewriter.create<AllocOp>(loc, scalarMemRefType);
      rewriter.create<AffineStoreOp>(loc, zero, xwAlloc, ArrayRef<Value>{});
      Value hrAlloc = rewriter.create<AllocOp>(loc, scalarMemRefType);
      rewriter.create<AffineStoreOp>(loc, zero, hrAlloc, ArrayRef<Value>{});
      xwZRH.emplace_back(xwAlloc);
      hrZRH.emplace_back(hrAlloc);
    }

    { // Emit instructions for matrix multiplications.
      // input_size is the reduction dimension.
      BuildKrnlLoop reductionLoops(rewriter, loc, 1);
      reductionLoops.createDefineOp();
      reductionLoops.pushBounds(0, inputDimSize);
      reductionLoops.createIterateOp();

      auto ipReductionLoops = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointToStart(reductionLoops.getIterateBlock());
      {
        auto reductionIV = reductionLoops.getInductionVar(0);
        // Prepare IVs for accessing the input tensor and parameters.
        SmallVector<Value, 3> xIVs;
        SmallVector<SmallVector<Value, 3>, GATES> wZRHIVs, rZRHIVs;

        // X :: [seq_length, batch_size, input_size]
        xIVs = {sequenceIV, batchIV, reductionIV};

        // W[zrh] :: [num_directions, GATES*hidden_size, input_size]
        // R[zrh] :: [num_directions, GATES*hidden_size, input_size]
        for (unsigned i = 0; i < GATES; ++i) {
          SmallVector<Value, 3> wIVs, rIVs;
          Value wHiddenIV = rewriter.create<AffineApplyOp>(loc,
              accessByOffsetMap,
              std::vector<Value>{hiddenIV, constantIndices[i], hiddenDimVal});

          wIVs = {directionIV, wHiddenIV, reductionIV};
          wZRHIVs.emplace_back(wIVs);

          rIVs = {directionIV, wHiddenIV, reductionIV};
          rZRHIVs.emplace_back(rIVs);
        }

        Value loadX =
            rewriter.create<AffineLoadOp>(loc, operandAdaptor.X(), xIVs);
        for (unsigned i = 0; i < GATES; ++i) {
          // Xt * W[zrh]
          Value loadW = rewriter.create<AffineLoadOp>(
              loc, operandAdaptor.W(), wZRHIVs[i]);
          Value xwVal = rewriter.create<MulFOp>(loc, loadX, loadW);
          Value loadXW = rewriter.create<AffineLoadOp>(loc, xwZRH[i]);
          Value nextXW = rewriter.create<AddFOp>(loc, loadXW, xwVal);
          rewriter.create<AffineStoreOp>(
              loc, nextXW, xwZRH[i], ArrayRef<Value>{});

          // Ht-1 * R[zrh]
          // Only compute Ht-1*(Rh^T) if not linearBeforeReset
          if (!state.linearBeforeReset && (i == GATES - 1))
            continue;
          Value loadR = rewriter.create<AffineLoadOp>(
              loc, operandAdaptor.R(), rZRHIVs[i]);
          Value hrVal = rewriter.create<MulFOp>(loc, loadH, loadR);
          Value loadHR = rewriter.create<AffineLoadOp>(loc, hrZRH[i]);
          Value nextHR = rewriter.create<AddFOp>(loc, loadHR, hrVal);
          rewriter.create<AffineStoreOp>(
              loc, nextHR, hrZRH[i], ArrayRef<Value>{});
        }
      }
      rewriter.restoreInsertionPoint(ipReductionLoops);
    }

    // zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
    Value loadXWZ = rewriter.create<AffineLoadOp>(loc, xwZRH[0]);
    Value loadHRZ = rewriter.create<AffineLoadOp>(loc, hrZRH[0]);
    Value zt = rewriter.create<AddFOp>(loc, loadXWZ, loadHRZ);
    if (hasBiasForInput) {
      Value loadWB =
          rewriter.create<AffineLoadOp>(loc, operandAdaptor.B(), wbZRHIVs[0]);
      zt = rewriter.create<AddFOp>(loc, zt, loadWB);
      Value loadRB =
          rewriter.create<AffineLoadOp>(loc, operandAdaptor.B(), rbZRHIVs[0]);
      zt = rewriter.create<AddFOp>(loc, zt, loadRB);
    }
    zt = applyActivation(rewriter, loc, activationPack.f, zt);

    // rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
    Value loadXWR = rewriter.create<AffineLoadOp>(loc, xwZRH[1]);
    Value loadHRR = rewriter.create<AffineLoadOp>(loc, hrZRH[1]);
    Value rt = rewriter.create<AddFOp>(loc, loadXWR, loadHRR);
    if (hasBiasForInput) {
      Value loadWB =
          rewriter.create<AffineLoadOp>(loc, operandAdaptor.B(), wbZRHIVs[1]);
      rt = rewriter.create<AddFOp>(loc, rt, loadWB);
      Value loadRB =
          rewriter.create<AffineLoadOp>(loc, operandAdaptor.B(), rbZRHIVs[1]);
      rt = rewriter.create<AddFOp>(loc, rt, loadRB);
    }
    rt = applyActivation(rewriter, loc, activationPack.f, rt);

    // if (linearBeforeReset)
    //   ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh)
    // else
    //   ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh)
    Value ht = rewriter.create<AffineLoadOp>(loc, xwZRH[2]);
    if (state.linearBeforeReset) {
      Value linear = rewriter.create<AffineLoadOp>(loc, hrZRH[2]);
      if (hasBiasForInput) {
        Value loadRB =
            rewriter.create<AffineLoadOp>(loc, operandAdaptor.B(), rbZRHIVs[2]);
        linear = rewriter.create<AddFOp>(loc, linear, loadRB);
      }
      Value reset = rewriter.create<MulFOp>(loc, rt, linear);
      ht = rewriter.create<AddFOp>(loc, ht, reset);
      if (hasBiasForInput) {
        Value loadWB =
            rewriter.create<AffineLoadOp>(loc, operandAdaptor.B(), wbZRHIVs[2]);
        ht = rewriter.create<AddFOp>(loc, ht, loadWB);
      }
    } else {
      // rtHt = rt (.) Ht-1)
      Value rtHt = rewriter.create<MulFOp>(loc, rt, loadH);
      {
        // Emit instructions for 'rtHt*(Rh^T)'.
        // input_size is the reduction dimension.
        BuildKrnlLoop reductionLoops(rewriter, loc, 1);
        reductionLoops.createDefineOp();
        reductionLoops.pushBounds(0, inputDimSize);
        reductionLoops.createIterateOp();

        auto ipReductionLoops = rewriter.saveInsertionPoint();
        rewriter.setInsertionPointToStart(reductionLoops.getIterateBlock());
        {
          auto reductionIV = reductionLoops.getInductionVar(0);
          // Rh :: [num_directions, hidden_size, input_size]
          Value wHiddenIV = rewriter.create<AffineApplyOp>(loc,
              accessByOffsetMap,
              std::vector<Value>{hiddenIV, constantIndices[2], hiddenDimVal});
          SmallVector<Value, 3> rhIVs = {directionIV, wHiddenIV, reductionIV};

          // 'rtHt*(Rh^T)'
          Value loadR =
              rewriter.create<AffineLoadOp>(loc, operandAdaptor.R(), rhIVs);
          Value hrVal = rewriter.create<MulFOp>(loc, rtHt, loadR);
          Value loadHR = rewriter.create<AffineLoadOp>(loc, hrZRH[2]);
          Value nextHR = rewriter.create<AddFOp>(loc, loadHR, hrVal);
          rewriter.create<AffineStoreOp>(
              loc, nextHR, hrZRH[2], ArrayRef<Value>{});
        }
        rewriter.restoreInsertionPoint(ipReductionLoops);
      }
      //   ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh)
      Value linear = rewriter.create<AffineLoadOp>(loc, hrZRH[2]);
      ht = rewriter.create<AddFOp>(loc, ht, linear);
      if (hasBiasForInput) {
        Value loadWB =
            rewriter.create<AffineLoadOp>(loc, operandAdaptor.B(), wbZRHIVs[2]);
        ht = rewriter.create<AddFOp>(loc, ht, loadWB);
        Value loadRB =
            rewriter.create<AffineLoadOp>(loc, operandAdaptor.B(), rbZRHIVs[2]);
        ht = rewriter.create<AddFOp>(loc, ht, loadRB);
      }
    }
    ht = applyActivation(rewriter, loc, activationPack.g, ht);

    // Ht = (1 - zt) (.) ht + zt (.) Ht-1
    Value one = emitConstantOp(rewriter, loc, elementType, 1);
    Value subZt = rewriter.create<SubFOp>(loc, one, zt);
    Value mulHt = rewriter.create<MulFOp>(loc, subZt, ht);
    Value mulH = rewriter.create<MulFOp>(loc, zt, loadH);
    Value Ht = rewriter.create<AddFOp>(loc, mulHt, mulH);
    rewriter.create<AffineStoreOp>(loc, Ht, state.ht, hIVs);

    // Store the current Ht if required.
    if (!isNoneType(state.allH)) {
      SmallVector<Value, 4> allHIVs{sequenceIV, directionIV, batchIV, hiddenIV};
      rewriter.create<AffineStoreOp>(loc, Ht, state.allH, allHIVs);
    }

    // Deallocate the temporary results of matrix multiplications.
    for (Value v : xwZRH)
      rewriter.create<DeallocOp>(loc, v);
    for (Value v : hrZRH)
      rewriter.create<DeallocOp>(loc, v);
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
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXRNNOpLowering<ONNXGRUOp, GruState, GruActivationPack>>(
      ctx);
}
