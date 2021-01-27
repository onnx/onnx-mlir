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

  // Insert allocation and deallocation for the results of this operation.
  // Y :: [seq_length, num_directions, batch_size, hidden_size]
  state.allH = allocAllHidden(rewriter, loc, operandAdaptor.X(),
      operandAdaptor.W(), operandAdaptor.R(), op->Y(),
      checkInsertDealloc(op->getOperation(), 0));
  // Y_h :: [num_directions, batch_size, hidden_size]
  state.ht = allocHiddenOrCell(rewriter, loc, operandAdaptor.X(),
      operandAdaptor.W(), operandAdaptor.R(), op->Y_h(),
      checkInsertDealloc(op->getOperation(), 1));
  // Y_c :: [num_directions, batch_size, hidden_size]
  state.ct = allocHiddenOrCell(rewriter, loc, operandAdaptor.X(),
      operandAdaptor.W(), operandAdaptor.R(), op->Y_c(),
      checkInsertDealloc(op->getOperation(), 2));

  // Initialize ht and ct.
  initializeHiddenAndCell(rewriter, loc, state.ht, state.ct,
      operandAdaptor.initial_h(), operandAdaptor.initial_c(),
      operandAdaptor.X().getType().cast<MemRefType>().getElementType(),
      /*onlyHidden=*/false);
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
  // Shape information:
  // Xt : [seq_length, batch_size, input_size]
  // W[iofc] : [num_directions, hidden_size, input_size]
  // R[iofc] : [num_directions, hidden_size, hidden_size]
  // Ht, Ct, it, ot, ft, ct: [num_directions, batch_size, hidden_size]
  // Wb[iofc] : [num_directions, hidden_size]
  // Rb[iofc] : [num_directions, hidden_size]
  //
  // The following code will emit loops as follows:
  //     for b in 0 .. BatchDimSize
  //       for h in 0 .. HiddenDimSize
  //         for i in 0 .. InputDimSize
  //           compute Xt*(Wi^T), Xt*(Wo^T), Xt*(Wf^t), Xt*(Wc^T),
  //         for i in 0 .. HiddenDimSize
  //            compute Ht-1*(Ri^T), Ht-1*(Ro^T), Ht-1*(Rf^t), Ht-1*(Rc^T)
  //     for b in 0 .. BatchDimSize
  //       for h in 0 .. HiddenDimSize
  //         compute it, ft, ct, Ct, ot, Ht
  //         update the hidden state with the new state Ht.
  //         update the cell state with the new state Ct.
  //
  // The reason to have two loops at the top level is to avoid updating any
  // element of the hidden state while computing Ht-1*(Ri^T), Ht-1*(Ro^T),
  // Ht-1*(Rf^t), Ht-1*(Rc^T)

  // Create temporary buffers for
  //   - Xt*(Wi^T), Xt*(Wo^T), Xt*(Wf^t), Xt*(Wc^T),
  //   - Ht-1*(Ri^T), Ht-1*(Ro^T), Ht-1*(Rf^t), Ht-1*(Rc^T)
  // These tensors have shape of [batch_size, hidden_size],
  MemRefType bufMemRefType = MemRefType::get(
      {dimAt(operandAdaptor.X(), 1), hiddenDimSize}, elementType);
  bool staticDimensions = hasAllConstantDimensions(bufMemRefType);
  SmallVector<Value, 4> xwIOFC, hrIOFC;
  for (unsigned i = 0; i < 4; ++i) {
    Value xwAlloc, hrAlloc;
    if (staticDimensions) {
      xwAlloc = insertAllocAndDealloc(bufMemRefType, loc, rewriter, false);
      hrAlloc = insertAllocAndDealloc(bufMemRefType, loc, rewriter, false);
    } else {
      // Hidden size is a constant, so the batch size must be unknown here.
      Value batchSizeDim =
          rewriter.create<DimOp>(loc, operandAdaptor.X(), 1).getResult();
      xwAlloc = rewriter.create<AllocOp>(
          loc, bufMemRefType, llvm::makeArrayRef({batchSizeDim}));
      hrAlloc = rewriter.create<AllocOp>(
          loc, bufMemRefType, llvm::makeArrayRef({batchSizeDim}));
    }
    xwIOFC.emplace_back(xwAlloc);
    hrIOFC.emplace_back(hrAlloc);
  }

  // Emit instructions for matrix multiplications:
  //   Xt*(Wi^T), Xt*(Wo^T), Xt*(Wf^t), Xt*(Wc^T)
  //   Ht-1*(Ri^T), Ht-1*(Ro^T), Ht-1*(Rf^t), Ht-1*(Rc^T)
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

    // IVs to access tensors.
    // [num_directions, batch_size, hidden_size]
    SmallVector<Value, 4> IVs = {directionIV, batchIV, hiddenIV};
    SmallVector<Value, 4> mIVs = {batchIV, hiddenIV};

    // Initialize matrix multiplication result.
    Value zero = emitConstantOp(rewriter, loc, elementType, 0);
    for (unsigned i = 0; i < 4; ++i) {
      rewriter.create<KrnlStoreOp>(loc, zero, xwIOFC[i], mIVs);
      rewriter.create<KrnlStoreOp>(loc, zero, hrIOFC[i], mIVs);
    }

    { // Emit instructions for matrix multiplications.
      //   Xt*(Wi^T), Xt*(Wo^T), Xt*(Wf^t), Xt*(Wc^T)
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
        SmallVector<Value, 4> xIVs;
        SmallVector<SmallVector<Value, 4>, 4> wIOFCIVs;

        // X :: [seq_length, batch_size, input_size]
        xIVs = {sequenceIV, batchIV, reductionIV};

        // W[iofc] :: [num_directions, 4*hidden_size, input_size]
        for (unsigned i = 0; i < 4; ++i) {
          SmallVector<Value, 4> wIVs, rIVs;
          Value wHiddenIV = rewriter.create<AffineApplyOp>(loc,
              accessByOffsetMap,
              std::vector<Value>{hiddenIV, constantIndices[i], hiddenDimVal});

          wIVs = {directionIV, wHiddenIV, reductionIV};
          wIOFCIVs.emplace_back(wIVs);
        }

        Value loadX =
            rewriter.create<KrnlLoadOp>(loc, operandAdaptor.X(), xIVs);
        for (unsigned i = 0; i < 4; ++i) {
          // Xt * Wiofc
          Value loadW =
              rewriter.create<KrnlLoadOp>(loc, operandAdaptor.W(), wIOFCIVs[i]);
          Value xwVal = rewriter.create<MulFOp>(loc, loadX, loadW);
          Value loadXW = rewriter.create<KrnlLoadOp>(loc, xwIOFC[i], mIVs);
          Value nextXW = rewriter.create<AddFOp>(loc, loadXW, xwVal);
          rewriter.create<KrnlStoreOp>(loc, nextXW, xwIOFC[i], mIVs);
        }
      }
      rewriter.restoreInsertionPoint(ipReductionLoops);
    }

    { // Emit instructions for matrix multiplications.
      //   Ht-1*(Ri^T), Ht-1*(Ro^T), Ht-1*(Rf^t), Ht-1*(Rc^T)
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
        SmallVector<Value, 4> hIVs;
        SmallVector<SmallVector<Value, 4>, 4> rIOFCIVs;

        // H :: [num_directions, batch_size, hidden_size]
        hIVs = {directionIV, batchIV, reductionIV};

        // R[iofc] :: [num_directions, 4*hidden_size, hidden_size]
        for (unsigned i = 0; i < 4; ++i) {
          SmallVector<Value, 4> rIVs;
          Value rHiddenIV = rewriter.create<AffineApplyOp>(loc,
              accessByOffsetMap,
              std::vector<Value>{hiddenIV, constantIndices[i], hiddenDimVal});
          rIVs = {directionIV, rHiddenIV, reductionIV};
          rIOFCIVs.emplace_back(rIVs);
        }

        Value loadH = rewriter.create<KrnlLoadOp>(loc, state.ht, hIVs);
        for (unsigned i = 0; i < 4; ++i) {
          // Ht-1 * Riofc
          Value loadR =
              rewriter.create<KrnlLoadOp>(loc, operandAdaptor.R(), rIOFCIVs[i]);
          Value hrVal = rewriter.create<MulFOp>(loc, loadH, loadR);
          Value loadHR = rewriter.create<KrnlLoadOp>(loc, hrIOFC[i], mIVs);
          Value nextHR = rewriter.create<AddFOp>(loc, loadHR, hrVal);
          rewriter.create<KrnlStoreOp>(loc, nextHR, hrIOFC[i], mIVs);
        }
      }
      rewriter.restoreInsertionPoint(ipReductionLoops);
    }
  }
  rewriter.restoreInsertionPoint(ipMatrixLoops);

  // Emit instructions for computing gate outputs.
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

    // IVs to access tensors.
    // IVs for the hidden and cell states, and matrix multiplication results.
    SmallVector<Value, 4> hIVs, cIVs, mIVs;
    // IVs for the bias tensors for W and R.
    SmallVector<SmallVector<Value, 4>, 4> wbIOFCIVs, rbIOFCIVs;
    // IVs for the peepholes.
    SmallVector<SmallVector<Value, 4>, 4> pIOFIVs;

    { // Compute IVs.
      // H :: [num_directions, batch_size, hidden_size]
      hIVs = {directionIV, batchIV, hiddenIV};
      // C :: [num_directions, batch_size, hidden_size]
      cIVs = {directionIV, batchIV, hiddenIV};
      // M :: [batch_size, hidden_size] for matmul
      mIVs = {batchIV, hiddenIV};

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

    // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
    Value loadC = rewriter.create<KrnlLoadOp>(loc, state.ct, cIVs);
    Value loadXWI = rewriter.create<KrnlLoadOp>(loc, xwIOFC[0], mIVs);
    Value loadHRI = rewriter.create<KrnlLoadOp>(loc, hrIOFC[0], mIVs);
    Value it = rewriter.create<AddFOp>(loc, loadXWI, loadHRI);
    if (hasPeepholes) {
      Value loadP =
          rewriter.create<KrnlLoadOp>(loc, operandAdaptor.P(), pIOFIVs[0]);
      Value PC = rewriter.create<MulFOp>(loc, loadP, loadC);
      it = rewriter.create<AddFOp>(loc, it, PC);
    }
    if (hasBiasForInput) {
      Value loadWB =
          rewriter.create<KrnlLoadOp>(loc, operandAdaptor.B(), wbIOFCIVs[0]);
      it = rewriter.create<AddFOp>(loc, it, loadWB);
      Value loadRB =
          rewriter.create<KrnlLoadOp>(loc, operandAdaptor.B(), rbIOFCIVs[0]);
      it = rewriter.create<AddFOp>(loc, it, loadRB);
    }
    it = applyActivation(rewriter, loc, activationPack.f, it);

    // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
    Value loadXWF = rewriter.create<KrnlLoadOp>(loc, xwIOFC[2], mIVs);
    Value loadHRF = rewriter.create<KrnlLoadOp>(loc, hrIOFC[2], mIVs);
    Value ft = rewriter.create<AddFOp>(loc, loadXWF, loadHRF);
    if (hasPeepholes) {
      Value loadP =
          rewriter.create<KrnlLoadOp>(loc, operandAdaptor.P(), pIOFIVs[2]);
      Value PC = rewriter.create<MulFOp>(loc, loadP, loadC);
      ft = rewriter.create<AddFOp>(loc, ft, PC);
    }
    if (hasBiasForInput) {
      Value loadWB =
          rewriter.create<KrnlLoadOp>(loc, operandAdaptor.B(), wbIOFCIVs[2]);
      ft = rewriter.create<AddFOp>(loc, ft, loadWB);
      Value loadRB =
          rewriter.create<KrnlLoadOp>(loc, operandAdaptor.B(), rbIOFCIVs[2]);
      ft = rewriter.create<AddFOp>(loc, ft, loadRB);
    }
    ft = applyActivation(rewriter, loc, activationPack.f, ft);

    // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
    Value loadXWC = rewriter.create<KrnlLoadOp>(loc, xwIOFC[3], mIVs);
    Value loadHRC = rewriter.create<KrnlLoadOp>(loc, hrIOFC[3], mIVs);
    Value ct = rewriter.create<AddFOp>(loc, loadXWC, loadHRC);
    if (hasBiasForInput) {
      Value loadWB =
          rewriter.create<KrnlLoadOp>(loc, operandAdaptor.B(), wbIOFCIVs[3]);
      ct = rewriter.create<AddFOp>(loc, ct, loadWB);
      Value loadRB =
          rewriter.create<KrnlLoadOp>(loc, operandAdaptor.B(), rbIOFCIVs[3]);
      ct = rewriter.create<AddFOp>(loc, ct, loadRB);
    }
    ct = applyActivation(rewriter, loc, activationPack.g, ct);

    // Ct = ft (.) Ct-1 + it (.) ct
    Value FtCt1 = rewriter.create<MulFOp>(loc, ft, loadC);
    Value itct = rewriter.create<MulFOp>(loc, it, ct);
    Value Ct = rewriter.create<AddFOp>(loc, FtCt1, itct);
    rewriter.create<KrnlStoreOp>(loc, Ct, state.ct, cIVs);

    // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
    Value loadXWO = rewriter.create<KrnlLoadOp>(loc, xwIOFC[1], mIVs);
    Value loadHRO = rewriter.create<KrnlLoadOp>(loc, hrIOFC[1], mIVs);
    Value ot = rewriter.create<AddFOp>(loc, loadXWO, loadHRO);
    if (hasPeepholes) {
      Value loadP =
          rewriter.create<KrnlLoadOp>(loc, operandAdaptor.P(), pIOFIVs[1]);
      Value PC = rewriter.create<MulFOp>(loc, loadP, Ct);
      ot = rewriter.create<AddFOp>(loc, ot, PC);
    }
    if (hasBiasForInput) {
      Value loadWB =
          rewriter.create<KrnlLoadOp>(loc, operandAdaptor.B(), wbIOFCIVs[1]);
      ot = rewriter.create<AddFOp>(loc, ot, loadWB);
      Value loadRB =
          rewriter.create<KrnlLoadOp>(loc, operandAdaptor.B(), rbIOFCIVs[1]);
      ot = rewriter.create<AddFOp>(loc, ot, loadRB);
    }
    ot = applyActivation(rewriter, loc, activationPack.f, ot);

    // Ht = ot (.) h(Ct)
    Value hCt = applyActivation(rewriter, loc, activationPack.h, Ct);
    Value Ht = rewriter.create<MulFOp>(loc, ot, hCt);
    rewriter.create<KrnlStoreOp>(loc, Ht, state.ht, hIVs);

    // Store the current Ht if required.
    if (!isNoneType(state.allH)) {
      SmallVector<Value, 4> allHIVs{sequenceIV, directionIV, batchIV, hiddenIV};
      rewriter.create<KrnlStoreOp>(loc, Ht, state.allH, allHIVs);
    }
  }
  rewriter.restoreInsertionPoint(ipStateLoops);
  // Deallocate the temporary results of matrix multiplications.
  for (Value v : xwIOFC)
    rewriter.create<DeallocOp>(loc, v);
  for (Value v : hrIOFC)
    rewriter.create<DeallocOp>(loc, v);
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
