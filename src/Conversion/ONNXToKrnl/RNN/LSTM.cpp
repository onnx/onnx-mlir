//===--------------- LSTM.cpp - Lowering LSTM Op --------------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX LSTM Operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/RNN/RNNBase.hpp"

using namespace mlir;

struct LstmState {
  Value allH;
  Value ht;
  Value ct;
};

struct LstmActivationPack {
  RNNActivation f;
  RNNActivation g;
  RNNActivation h;
};

template <>
bool hasAllNoneOutput<ONNXLSTMOp>(ONNXLSTMOp *op) {
  return (
      isNoneType(op->Y()) && isNoneType(op->Y_h()) && isNoneType(op->Y_c()));
}

template <>
std::tuple<LstmActivationPack, LstmActivationPack>
getActivationPack<ONNXLSTMOp, LstmActivationPack>(ONNXLSTMOp *op) {
  auto direction = op->direction();
  auto activations = op->activations();
  auto activationAlpha = op->activation_alpha();
  auto activationBeta = op->activation_beta();

  LstmActivationPack activationForward, activationReverse;

  // Get activation function name.
  // Default forward functions
  activationForward.f.name = "sigmoid";
  activationForward.g.name = "tanh";
  activationForward.h.name = "tanh";
  // Default backward functions
  activationReverse.f.name = "sigmoid";
  activationReverse.g.name = "tanh";
  activationReverse.h.name = "tanh";
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
      if (activationArrAttr.size() > 2) {
        activationForward.h.name =
            activationArrAttr[2].cast<StringAttr>().getValue();
      }
    }

    // Reverse activations.
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      int startIndex = (direction == REVERSE) ? 0 : 3;
      if (activationArrAttr.size() > startIndex) {
        activationReverse.f.name =
            activationArrAttr[startIndex].cast<StringAttr>().getValue();
      }
      if (activationArrAttr.size() > startIndex + 1) {
        activationReverse.g.name =
            activationArrAttr[startIndex + 1].cast<StringAttr>().getValue();
      }
      if (activationArrAttr.size() > startIndex + 2) {
        activationReverse.h.name =
            activationArrAttr[startIndex + 2].cast<StringAttr>().getValue();
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
      if (activationArrAttr.size() > 2) {
        activationForward.h.alpha = activationArrAttr[2].cast<FloatAttr>();
      }
    }

    // Reverse activations.
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      int startIndex = (direction == REVERSE) ? 0 : 3;
      if (activationArrAttr.size() > startIndex) {
        activationReverse.f.alpha =
            activationArrAttr[startIndex].cast<FloatAttr>();
      }
      if (activationArrAttr.size() > startIndex + 1) {
        activationReverse.g.alpha =
            activationArrAttr[startIndex + 1].cast<FloatAttr>();
      }
      if (activationArrAttr.size() > startIndex + 2) {
        activationReverse.h.alpha =
            activationArrAttr[startIndex + 2].cast<FloatAttr>();
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
      if (activationArrAttr.size() > 2) {
        activationForward.h.beta = activationArrAttr[2].cast<FloatAttr>();
      }
    }

    // Reverse activations.
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      int startIndex = (direction == REVERSE) ? 0 : 3;
      if (activationArrAttr.size() > startIndex) {
        activationReverse.f.beta =
            activationArrAttr[startIndex].cast<FloatAttr>();
      }
      if (activationArrAttr.size() > startIndex + 1) {
        activationReverse.g.beta =
            activationArrAttr[startIndex + 1].cast<FloatAttr>();
      }
      if (activationArrAttr.size() > startIndex + 2) {
        activationReverse.h.beta =
            activationArrAttr[startIndex + 2].cast<FloatAttr>();
      }
    }
  }

  return std::make_tuple(activationForward, activationReverse);
}

template <>
LstmState allocAndInitializeStates<ONNXLSTMOp, LstmState>(
    ConversionPatternRewriter &rewriter, Location loc, ONNXLSTMOp *op,
    typename ONNXLSTMOp::Adaptor operandAdaptor) {
  LstmState state;
  FuncOp function = cast<FuncOp>(op->getParentOp());

  // Insert allocation and deallocation for the results of this operation.
  if (!isNoneType(op->Y())) {
    auto yMemRefType = convertToMemRefType(op->Y().getType());
    if (hasAllConstantDimensions(yMemRefType))
      state.allH = insertAllocAndDeallocWithFunction(yMemRefType, loc, rewriter,
          checkInsertDealloc(op->getOperation(), 0), function, true);
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
      state.ht = insertAllocAndDeallocWithFunction(yhMemRefType, loc, rewriter,
          checkInsertDealloc(op->getOperation(), 1), function, true);
    else
      llvm_unreachable("Unsupported dynamic dimensions.");
  } else {
    auto yhMemRefType = MemRefType::get(
        {dimAt(operandAdaptor.W(), 0), dimAt(operandAdaptor.X(), 1),
            dimAt(operandAdaptor.R(), 2)},
        operandAdaptor.X().getType().cast<ShapedType>().getElementType());
    state.ht = insertAllocAndDeallocWithFunction(
        yhMemRefType, loc, rewriter, true, function, true);
  }

  // Y_c :: [num_directions, batch_size, hidden_size]
  if (!isNoneType(op->Y_c())) {
    auto ycMemRefType = convertToMemRefType(op->Y_c().getType());
    if (hasAllConstantDimensions(ycMemRefType))
      state.ct = insertAllocAndDeallocWithFunction(ycMemRefType, loc, rewriter,
          checkInsertDealloc(op->getOperation(), 2), function, true);
    else
      llvm_unreachable("Unsupported dynamic dimensions.");
  } else {
    auto ycMemRefType = MemRefType::get(
        {dimAt(operandAdaptor.W(), 0), dimAt(operandAdaptor.X(), 1),
            dimAt(operandAdaptor.R(), 2)},
        operandAdaptor.X().getType().cast<ShapedType>().getElementType());
    state.ct = insertAllocAndDeallocWithFunction(
        ycMemRefType, loc, rewriter, true, function, true);
  }

  // Initialize ht and ct.
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

    Value cellVal = zero;
    if (!isNoneType(operandAdaptor.initial_c()))
      cellVal =
          rewriter.create<AffineLoadOp>(loc, operandAdaptor.initial_c(), IVs);
    rewriter.create<AffineStoreOp>(loc, cellVal, state.ct, IVs);
  }
  rewriter.restoreInsertionPoint(ipInitializationLoops);
  return state;
}

template <>
void calculateState<ONNXLSTMOp, LstmState, LstmActivationPack>(
    ConversionPatternRewriter &rewriter, Location loc,
    typename ONNXLSTMOp::Adaptor operandAdaptor, LstmState state,
    LstmActivationPack activationPack, Value directionIV, Value sequenceIV) {

  bool hasBiasForInput = false, hasPeepholes = false;
  if (!isNoneType(operandAdaptor.B()))
    hasBiasForInput = true;
  if (!isNoneType(operandAdaptor.P()))
    hasPeepholes = true;

  // Prepare dimensions.
  auto batchDimSize = dimAt(operandAdaptor.X(), 1);
  auto inputDimSize = dimAt(operandAdaptor.X(), 2);
  auto hiddenDimSize = dimAt(operandAdaptor.R(), 2);
  Value hiddenDimVal =
      emitConstantOp(rewriter, loc, rewriter.getIndexType(), hiddenDimSize);

  auto elementType =
      operandAdaptor.X().getType().cast<ShapedType>().getElementType();

  // Prepare AffineMap to access bias, peepholes tensors.
  AffineMap accessByOffsetMap;
  {
    AffineExpr iv = rewriter.getAffineDimExpr(0);
    AffineExpr index = rewriter.getAffineSymbolExpr(0);
    AffineExpr size = rewriter.getAffineSymbolExpr(1);
    AffineExpr accessByOffsetExpr = index * size + iv;
    accessByOffsetMap = AffineMap::get(1, 2, accessByOffsetExpr);
  }

  // Prepare constant indices.
  SmallVector<Value, 4> constantIndices;
  for (int i = 0; i < 8; i++)
    constantIndices.emplace_back(
        emitConstantOp(rewriter, loc, rewriter.getIndexType(), i));

  // Equations for LSTM.
  // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
  // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
  // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
  // Ct = ft (.) Ct-1 + it (.) ct
  // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
  // Ht = ot (.) h(Ct)
  //
  // The following code will emit loops as follows:
  // for b in 0 .. BatchDimSize
  //   for h in 0 .. HiddenDimSize
  //     for i in 0 .. InputDimSize {
  //       compute Xt*(Wi^T), Xt*(Wo^T), Xt*(Wf^t), Xt*(Wc^T),
  //               Ht-1*(Ri^T), Ht-1*(Ro^T), Ht-1*(Rf^t), Ht-1*(Rc^T)
  //     }
  //     compute it, ft, ct, Ct, ot, Ht

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
    // IVs for the hidden and cell state tensors.
    SmallVector<Value, 4> hIVs, cIVs;
    // IVs for the bias tensors for W and R.
    SmallVector<SmallVector<Value, 4>, 4> wbIOFCIVs, rbIOFCIVs;
    // IVs for the peepholes.
    SmallVector<SmallVector<Value, 4>, 4> pIOFIVs;

    { // Compute IVs.
      // H :: [num_directions, batch_size, hidden_size]
      hIVs = {directionIV, batchIV, hiddenIV};
      // C :: [num_directions, batch_size, hidden_size]
      cIVs = {directionIV, batchIV, hiddenIV};

      // Bias [Wb[iofc], Rb[iofc]] :: [num_directions, 8*hidden_size]
      if (hasBiasForInput) {
        // Wb[iofc]
        for (unsigned i = 0; i < 4; ++i) {
          Value wHiddenIV =
              rewriter.create<AffineApplyOp>(loc, accessByOffsetMap,
                  std::vector<Value>{/*iv=*/hiddenIV,
                      /*index=*/constantIndices[i], /*size=*/hiddenDimVal});
          wbIOFCIVs.emplace_back(SmallVector<Value, 2>{directionIV, wHiddenIV});
        }
        // Rb[iofc]
        for (unsigned i = 4; i < 8; ++i) {
          SmallVector<Value, 4> rbIVs;
          Value rHiddenIV =
              rewriter.create<AffineApplyOp>(loc, accessByOffsetMap,
                  std::vector<Value>{/*iv=*/hiddenIV,
                      /*index=*/constantIndices[i], /*size=*/hiddenDimVal});
          rbIOFCIVs.emplace_back(SmallVector<Value, 2>{directionIV, rHiddenIV});
        }
      }

      // Peepholes P[iof] :: [num_directions, 3*hidden_size]
      if (hasPeepholes) {
        for (unsigned i = 0; i < 3; ++i) {
          SmallVector<Value, 4> pIVs;
          Value pHiddenIV = rewriter.create<AffineApplyOp>(loc,
              accessByOffsetMap,
              std::vector<Value>{hiddenIV, constantIndices[i], hiddenDimVal});
          pIOFIVs.emplace_back(SmallVector<Value, 2>{directionIV, pHiddenIV});
        }
      }
    }

    Value loadH = rewriter.create<AffineLoadOp>(loc, state.ht, hIVs);
    Value loadC = rewriter.create<AffineLoadOp>(loc, state.ct, cIVs);

    // Emit instructions for matrix multiplications:
    //   Xt*(Wi^T), Xt*(Wo^T), Xt*(Wf^t), Xt*(Wc^T)
    //   Ht-1*(Ri^T), Ht-1*(Ro^T), Ht-1*(Rf^t), Ht-1*(Rc^T)

    // Allocate memory for storing matrix multiplication results.
    SmallVector<Value, 4> xwIOFC, hrIOFC;
    Value zero = emitConstantOp(rewriter, loc, elementType, 0);
    MemRefType scalarMemRefType = MemRefType::get({}, elementType, {}, 0);
    for (unsigned i = 0; i < 4; ++i) {
      Value xwAlloc = rewriter.create<AllocOp>(loc, scalarMemRefType);
      rewriter.create<AffineStoreOp>(loc, zero, xwAlloc, ArrayRef<Value>{});
      Value hrAlloc = rewriter.create<AllocOp>(loc, scalarMemRefType);
      rewriter.create<AffineStoreOp>(loc, zero, hrAlloc, ArrayRef<Value>{});
      xwIOFC.emplace_back(xwAlloc);
      hrIOFC.emplace_back(hrAlloc);
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
        SmallVector<Value, 4> xIVs;
        SmallVector<SmallVector<Value, 4>, 4> wIOFCIVs, rIOFCIVs;

        // X :: [seq_length, batch_size, input_size]
        xIVs = {sequenceIV, batchIV, reductionIV};

        // W[iofc] :: [num_directions, 4*hidden_size, input_size]
        // R[iofc] :: [num_directions, 4*hidden_size, input_size]
        for (unsigned i = 0; i < 4; ++i) {
          SmallVector<Value, 4> wIVs, rIVs;
          Value wHiddenIV = rewriter.create<AffineApplyOp>(loc,
              accessByOffsetMap,
              std::vector<Value>{hiddenIV, constantIndices[i], hiddenDimVal});

          wIVs = {directionIV, wHiddenIV, reductionIV};
          wIOFCIVs.emplace_back(wIVs);

          rIVs = {directionIV, wHiddenIV, reductionIV};
          rIOFCIVs.emplace_back(rIVs);
        }

        Value loadX =
            rewriter.create<AffineLoadOp>(loc, operandAdaptor.X(), xIVs);
        for (unsigned i = 0; i < 4; ++i) {
          // Xt * Wiofc
          Value loadW = rewriter.create<AffineLoadOp>(
              loc, operandAdaptor.W(), wIOFCIVs[i]);
          Value xwVal = rewriter.create<MulFOp>(loc, loadX, loadW);
          Value loadXW = rewriter.create<AffineLoadOp>(loc, xwIOFC[i]);
          Value nextXW = rewriter.create<AddFOp>(loc, loadXW, xwVal);
          rewriter.create<AffineStoreOp>(
              loc, nextXW, xwIOFC[i], ArrayRef<Value>{});
          // Ht-1 * Riofc
          Value loadR = rewriter.create<AffineLoadOp>(
              loc, operandAdaptor.R(), rIOFCIVs[i]);
          Value hrVal = rewriter.create<MulFOp>(loc, loadH, loadR);
          Value loadHR = rewriter.create<AffineLoadOp>(loc, hrIOFC[i]);
          Value nextHR = rewriter.create<AddFOp>(loc, loadHR, hrVal);
          rewriter.create<AffineStoreOp>(
              loc, nextHR, hrIOFC[i], ArrayRef<Value>{});
        }
      }
      rewriter.restoreInsertionPoint(ipReductionLoops);
    }

    // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
    Value loadXWI = rewriter.create<AffineLoadOp>(loc, xwIOFC[0]);
    Value loadHRI = rewriter.create<AffineLoadOp>(loc, hrIOFC[0]);
    Value it = rewriter.create<AddFOp>(loc, loadXWI, loadHRI);
    if (hasPeepholes) {
      Value loadP =
          rewriter.create<AffineLoadOp>(loc, operandAdaptor.P(), pIOFIVs[0]);
      Value PC = rewriter.create<MulFOp>(loc, loadP, loadC);
      it = rewriter.create<AddFOp>(loc, it, PC);
    }
    if (hasBiasForInput) {
      Value loadWB =
          rewriter.create<AffineLoadOp>(loc, operandAdaptor.B(), wbIOFCIVs[0]);
      it = rewriter.create<AddFOp>(loc, it, loadWB);
      Value loadRB =
          rewriter.create<AffineLoadOp>(loc, operandAdaptor.B(), rbIOFCIVs[0]);
      it = rewriter.create<AddFOp>(loc, it, loadRB);
    }
    it = applyActivation(rewriter, loc, activationPack.f, it);

    // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
    Value loadXWF = rewriter.create<AffineLoadOp>(loc, xwIOFC[2]);
    Value loadHRF = rewriter.create<AffineLoadOp>(loc, hrIOFC[2]);
    Value ft = rewriter.create<AddFOp>(loc, loadXWF, loadHRF);
    if (hasPeepholes) {
      Value loadP =
          rewriter.create<AffineLoadOp>(loc, operandAdaptor.P(), pIOFIVs[2]);
      Value PC = rewriter.create<MulFOp>(loc, loadP, loadC);
      ft = rewriter.create<AddFOp>(loc, ft, PC);
    }
    if (hasBiasForInput) {
      Value loadWB =
          rewriter.create<AffineLoadOp>(loc, operandAdaptor.B(), wbIOFCIVs[2]);
      ft = rewriter.create<AddFOp>(loc, ft, loadWB);
      Value loadRB =
          rewriter.create<AffineLoadOp>(loc, operandAdaptor.B(), rbIOFCIVs[2]);
      ft = rewriter.create<AddFOp>(loc, ft, loadRB);
    }
    ft = applyActivation(rewriter, loc, activationPack.f, ft);

    // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
    Value loadXWC = rewriter.create<AffineLoadOp>(loc, xwIOFC[3]);
    Value loadHRC = rewriter.create<AffineLoadOp>(loc, hrIOFC[3]);
    Value ct = rewriter.create<AddFOp>(loc, loadXWC, loadHRC);
    if (hasBiasForInput) {
      Value loadWB =
          rewriter.create<AffineLoadOp>(loc, operandAdaptor.B(), wbIOFCIVs[3]);
      ct = rewriter.create<AddFOp>(loc, ct, loadWB);
      Value loadRB =
          rewriter.create<AffineLoadOp>(loc, operandAdaptor.B(), rbIOFCIVs[3]);
      ct = rewriter.create<AddFOp>(loc, ct, loadRB);
    }
    ct = applyActivation(rewriter, loc, activationPack.g, ct);

    // Ct = ft (.) Ct-1 + it (.) ct
    Value FtCt1 = rewriter.create<MulFOp>(loc, ft, loadC);
    Value itct = rewriter.create<MulFOp>(loc, it, ct);
    Value Ct = rewriter.create<AddFOp>(loc, FtCt1, itct);
    rewriter.create<AffineStoreOp>(loc, Ct, state.ct, cIVs);

    // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
    Value loadXWO = rewriter.create<AffineLoadOp>(loc, xwIOFC[1]);
    Value loadHRO = rewriter.create<AffineLoadOp>(loc, hrIOFC[1]);
    Value ot = rewriter.create<AddFOp>(loc, loadXWO, loadHRO);
    if (hasPeepholes) {
      Value loadP =
          rewriter.create<AffineLoadOp>(loc, operandAdaptor.P(), pIOFIVs[1]);
      Value PC = rewriter.create<MulFOp>(loc, loadP, Ct);
      ot = rewriter.create<AddFOp>(loc, ot, PC);
    }
    if (hasBiasForInput) {
      Value loadWB =
          rewriter.create<AffineLoadOp>(loc, operandAdaptor.B(), wbIOFCIVs[1]);
      ot = rewriter.create<AddFOp>(loc, ot, loadWB);
      Value loadRB =
          rewriter.create<AffineLoadOp>(loc, operandAdaptor.B(), rbIOFCIVs[1]);
      ot = rewriter.create<AddFOp>(loc, ot, loadRB);
    }
    ot = applyActivation(rewriter, loc, activationPack.f, ot);

    // Ht = ot (.) h(Ct)
    Value hCt = applyActivation(rewriter, loc, activationPack.h, Ct);
    Value Ht = rewriter.create<MulFOp>(loc, ot, hCt);
    rewriter.create<AffineStoreOp>(loc, Ht, state.ht, hIVs);

    // Store the current Ht if required.
    if (!isNoneType(state.allH)) {
      SmallVector<Value, 4> allHIVs{sequenceIV, directionIV, batchIV, hiddenIV};
      rewriter.create<AffineStoreOp>(loc, Ht, state.allH, allHIVs);
    }

    // Deallocate the temporary results of matrix multiplications.
    for (Value v : xwIOFC)
      rewriter.create<DeallocOp>(loc, v);
    for (Value v : hrIOFC)
      rewriter.create<DeallocOp>(loc, v);
  }
}

template <>
void stateToOutput<ONNXLSTMOp, LstmState>(
    ONNXLSTMOp *op, LstmState state, std::vector<Value> &outputs) {
  Value noneValue;
  outputs.emplace_back((isNoneType(op->Y()) ? noneValue : state.allH));
  outputs.emplace_back((isNoneType(op->Y_h()) ? noneValue : state.ht));
  outputs.emplace_back((isNoneType(op->Y_c()) ? noneValue : state.ct));
}

void populateLoweringONNXLSTMOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXRNNOpLowering<ONNXLSTMOp, LstmState, LstmActivationPack>>(
      ctx);
}
