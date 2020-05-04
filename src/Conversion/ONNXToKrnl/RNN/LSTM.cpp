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
  return (op->Y().getType().isa<NoneType>() &&
          op->Y_h().getType().isa<NoneType>() &&
          op->Y_c().getType().isa<NoneType>());
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
    OperandAdaptor<ONNXLSTMOp> operandAdaptor) {
  LstmState state;

  // Insert allocation and deallocation for the results of this operation.
  if (!op->Y().getType().isa<NoneType>()) {
    auto yMemRefType = convertToMemRefType(op->Y().getType());
    if (hasAllConstantDimensions(yMemRefType))
      state.allH = insertAllocAndDealloc(yMemRefType, loc, rewriter,
          checkInsertDealloc(op->getOperation(), 0));
    else
      emitError(loc, "Unsupported dynamic dimensions.");
  } else {
    state.allH = op->Y();
  }

  if (!op->Y_h().getType().isa<NoneType>()) {
    auto yhMemRefType = convertToMemRefType(op->Y_h().getType());
    if (hasAllConstantDimensions(yhMemRefType))
      state.ht = insertAllocAndDealloc(yhMemRefType, loc, rewriter,
          checkInsertDealloc(op->getOperation(), 1));
    else
      emitError(loc, "Unsupported dynamic dimensions.");
  } else {
    SmallVector<int64_t, 3> yhDims;
    yhDims.emplace_back(
        operandAdaptor.W().getType().cast<ShapedType>().getShape()[0]);
    yhDims.emplace_back(
        operandAdaptor.X().getType().cast<ShapedType>().getShape()[1]);
    yhDims.emplace_back(
        operandAdaptor.R().getType().cast<ShapedType>().getShape()[2]);
    auto yhMemRefType = MemRefType::get(yhDims,
        operandAdaptor.X().getType().cast<ShapedType>().getElementType());
    state.ht = insertAllocAndDealloc(yhMemRefType, loc, rewriter, true);
  }

  if (!op->Y_c().getType().isa<NoneType>()) {
    auto ycMemRefType = convertToMemRefType(op->Y_c().getType());
    if (hasAllConstantDimensions(ycMemRefType))
      state.ct = insertAllocAndDealloc(ycMemRefType, loc, rewriter,
          checkInsertDealloc(op->getOperation(), 2));
    else
      emitError(loc, "Unsupported dynamic dimensions.");
  } else {
    SmallVector<int64_t, 3> ycDims;
    ycDims.emplace_back(
        operandAdaptor.W().getType().cast<ShapedType>().getShape()[0]);
    ycDims.emplace_back(
        operandAdaptor.X().getType().cast<ShapedType>().getShape()[1]);
    ycDims.emplace_back(
        operandAdaptor.R().getType().cast<ShapedType>().getShape()[2]);
    auto ycMemRefType = MemRefType::get(ycDims,
        operandAdaptor.X().getType().cast<ShapedType>().getElementType());
    state.ct = insertAllocAndDealloc(ycMemRefType, loc, rewriter, true);
  }

  // Initialize ht and ct.
  Value zero = emitConstantOp(rewriter, loc,
      operandAdaptor.X().getType().cast<ShapedType>().getElementType(), 0);
  int nLoops = 3;
  BuildKrnlLoop initializationLoops(rewriter, loc, nLoops);
  initializationLoops.createDefineOptimizeAndIterateOp(state.ht);
  auto ipInitializationLoops = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(initializationLoops.getIterateBlock());
  {
    SmallVector<Value, 4> IVs;
    for (int i = 0; i < nLoops; ++i)
      IVs.emplace_back(initializationLoops.getInductionVar(i));
    Value hiddenVal =
        (!operandAdaptor.initial_h().getType().isa<NoneType>())
            ? rewriter.create<LoadOp>(loc, operandAdaptor.initial_h(), IVs)
            : zero;
    rewriter.create<StoreOp>(loc, hiddenVal, state.ht, IVs);
    Value cellVal =
        (!operandAdaptor.initial_c().getType().isa<NoneType>())
            ? rewriter.create<LoadOp>(loc, operandAdaptor.initial_c(), IVs)
            : zero;
    rewriter.create<StoreOp>(loc, cellVal, state.ct, IVs);
  }
  rewriter.restoreInsertionPoint(ipInitializationLoops);
  return state;
}

template <>
void calculateState<ONNXLSTMOp, LstmState, LstmActivationPack>(
    ConversionPatternRewriter &rewriter, Location loc,
    OperandAdaptor<ONNXLSTMOp> operandAdaptor, LstmState state,
    LstmActivationPack activationPack, Value numDirectionIV,
    Value sequenceLengthIV) {

  bool hasBiasForInput =
      (operandAdaptor.B().getType().isa<NoneType>()) ? false : true;
  bool hasPeepholes =
      (operandAdaptor.P().getType().isa<NoneType>()) ? false : true;
  bool hasSequenceLengths =
      (operandAdaptor.sequence_lens().getType().isa<NoneType>()) ? false : true;
  if (hasSequenceLengths)
    emitError(loc, "Does not support sequence_lens at this time");

  // Preapre dimensions.
  auto batchSizeDim =
      operandAdaptor.X().getType().cast<ShapedType>().getShape()[1];
  auto inputSizeDim =
      operandAdaptor.X().getType().cast<ShapedType>().getShape()[2];
  auto hiddenSizeDim =
      operandAdaptor.R().getType().cast<ShapedType>().getShape()[2];
  Value hiddenSizeVal =
      emitConstantOp(rewriter, loc, rewriter.getIndexType(), hiddenSizeDim);

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
  // for b in 0 .. BatchSizeDim
  //   for h in 0 .. HiddenSizeDim
  //     for i in 0 .. InputSizeDim {
  //       compute Xt*(Wi^T), Xt*(Wo^T), Xt*(Wf^t), Xt*(Wc^T),
  //               Ht-1*(Ri^T), Ht-1*(Ro^T), Ht-1*(Rf^t), Ht-1*(Rc^T)
  //     }
  //     compute it, ft, ct, Ct, ot, Ht

  BuildKrnlLoop stateLoops(rewriter, loc, 2);
  stateLoops.createDefineAndOptimizeOp();
  stateLoops.pushBounds(0, batchSizeDim);
  stateLoops.pushBounds(0, hiddenSizeDim);
  stateLoops.createIterateOp();

  rewriter.setInsertionPointToStart(stateLoops.getIterateBlock());
  {
    auto batchSizeIV = stateLoops.getInductionVar(0);
    auto hiddenSizeIV = stateLoops.getInductionVar(1);

    // IVs to access tensors.
    // IVs for the hidden and cell state tensors.
    SmallVector<Value, 4> hIVs, cIVs;
    // IVs for the bias tensors for W and R.
    SmallVector<SmallVector<Value, 4>, 4> wbIOFCIVs, rbIOFCIVs;
    // IVs for the peepholes.
    SmallVector<SmallVector<Value, 4>, 4> pIOFIVs;

    { // Compute IVs.
      // H :: [num_directions, batch_size, hidden_size]
      hIVs = {numDirectionIV, batchSizeIV, hiddenSizeIV};
      // C :: [num_directions, batch_size, hidden_size]
      cIVs = {numDirectionIV, batchSizeIV, hiddenSizeIV};

      // Bias [Wb[iofc], Rb[iofc]] :: [num_directions, 8*hidden_size]
      if (hasBiasForInput) {
        // Wb[iofc]
        for (unsigned i = 0; i < 4; ++i) {
          Value wHiddenIV =
              rewriter.create<AffineApplyOp>(loc, accessByOffsetMap,
                  ValueRange(std::vector<Value>{
                      hiddenSizeIV, constantIndices[i], hiddenSizeVal}));
          wbIOFCIVs.emplace_back(
              SmallVector<Value, 2>{numDirectionIV, wHiddenIV});
        }
        // Rb[iofc]
        for (unsigned i = 4; i < 8; ++i) {
          SmallVector<Value, 4> rbIVs;
          Value rHiddenIV =
              rewriter.create<AffineApplyOp>(loc, accessByOffsetMap,
                  ValueRange(std::vector<Value>{
                      hiddenSizeIV, constantIndices[i], hiddenSizeVal}));
          rbIOFCIVs.emplace_back(
              SmallVector<Value, 2>{numDirectionIV, rHiddenIV});
        }
      }

      // Peepholes P[iof] :: [num_directions, 3*hidden_size]
      if (hasPeepholes) {
        for (unsigned i = 0; i < 3; ++i) {
          SmallVector<Value, 4> pIVs;
          Value pHiddenIV =
              rewriter.create<AffineApplyOp>(loc, accessByOffsetMap,
                  ValueRange(std::vector<Value>{
                      hiddenSizeIV, constantIndices[i], hiddenSizeVal}));
          pIOFIVs.emplace_back(
              SmallVector<Value, 2>{numDirectionIV, pHiddenIV});
        }
      }
    }

    Value loadH = rewriter.create<LoadOp>(loc, state.ht, hIVs);
    Value loadC = rewriter.create<LoadOp>(loc, state.ct, cIVs);

    // Emit instructions for matrix multiplications:
    //   Xt*(Wi^T), Xt*(Wo^T), Xt*(Wf^t), Xt*(Wc^T)
    //   Ht-1*(Ri^T), Ht-1*(Ro^T), Ht-1*(Rf^t), Ht-1*(Rc^T)

    // Allocate memory for storing matrix multiplication results.
    SmallVector<Value, 4> xwIOFC, hrIOFC;
    Value zero = emitConstantOp(rewriter, loc, elementType, 0);
    MemRefType scalarMemRefType = MemRefType::get({}, elementType, {}, 0);
    for (unsigned i = 0; i < 4; ++i) {
      Value xwAlloc = rewriter.create<AllocOp>(loc, scalarMemRefType);
      rewriter.create<StoreOp>(loc, zero, xwAlloc);
      Value hrAlloc = rewriter.create<AllocOp>(loc, scalarMemRefType);
      rewriter.create<StoreOp>(loc, zero, hrAlloc);
      xwIOFC.emplace_back(xwAlloc);
      hrIOFC.emplace_back(hrAlloc);
    }

    { // Emit instructions for matrix multiplications.
      // input_size is the reduction dimension.
      BuildKrnlLoop reductionLoops(rewriter, loc, 1);
      reductionLoops.createDefineAndOptimizeOp();
      reductionLoops.pushBounds(0, inputSizeDim);
      reductionLoops.createIterateOp();

      auto ipReductionLoops = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointToStart(reductionLoops.getIterateBlock());
      {
        auto reductionIV = reductionLoops.getInductionVar(0);
        // Prepare IVs for accessing the input tensor and parameters.
        SmallVector<Value, 4> xIVs;
        SmallVector<SmallVector<Value, 4>, 4> wIOFCIVs, rIOFCIVs;

        // X :: [seq_length, batch_size, input_size]
        xIVs = {sequenceLengthIV, batchSizeIV, reductionIV};

        // W[iofc] :: [num_directions, 4*hidden_size, input_size]
        // R[iofc] :: [num_directions, 4*hidden_size, input_size]
        for (unsigned i = 0; i < 4; ++i) {
          SmallVector<Value, 4> wIVs, rIVs;
          Value wHiddenIV =
              rewriter.create<AffineApplyOp>(loc, accessByOffsetMap,
                  ValueRange(std::vector<Value>{
                      hiddenSizeIV, constantIndices[i], hiddenSizeVal}));

          wIVs = {numDirectionIV, wHiddenIV, reductionIV};
          wIOFCIVs.emplace_back(wIVs);

          rIVs = {numDirectionIV, wHiddenIV, reductionIV};
          rIOFCIVs.emplace_back(rIVs);
        }

        Value loadX = rewriter.create<LoadOp>(loc, operandAdaptor.X(), xIVs);
        for (unsigned i = 0; i < 4; ++i) {
          // Xt * Wiofc
          Value loadW =
              rewriter.create<LoadOp>(loc, operandAdaptor.W(), wIOFCIVs[i]);
          Value xwVal = rewriter.create<MulFOp>(loc, loadX, loadW);
          Value loadXW = rewriter.create<LoadOp>(loc, xwIOFC[i]);
          Value nextXW = rewriter.create<AddFOp>(loc, loadXW, xwVal);
          rewriter.create<StoreOp>(loc, nextXW, xwIOFC[i]);
          // Ht-1 * Riofc
          Value loadR =
              rewriter.create<LoadOp>(loc, operandAdaptor.R(), rIOFCIVs[i]);
          Value hrVal = rewriter.create<MulFOp>(loc, loadH, loadR);
          Value loadHR = rewriter.create<LoadOp>(loc, hrIOFC[i]);
          Value nextHR = rewriter.create<AddFOp>(loc, loadHR, hrVal);
          rewriter.create<StoreOp>(loc, nextHR, hrIOFC[i]);
        }
      }
      rewriter.restoreInsertionPoint(ipReductionLoops);
    }

    // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
    Value loadXWI = rewriter.create<LoadOp>(loc, xwIOFC[0]);
    Value loadHRI = rewriter.create<LoadOp>(loc, hrIOFC[0]);
    Value it = rewriter.create<AddFOp>(loc, loadXWI, loadHRI);
    if (hasPeepholes) {
      Value loadP =
          rewriter.create<LoadOp>(loc, operandAdaptor.P(), pIOFIVs[0]);
      Value PC = rewriter.create<MulFOp>(loc, loadP, loadC);
      it = rewriter.create<AddFOp>(loc, it, PC);
    }
    if (hasBiasForInput) {
      Value loadWB =
          rewriter.create<LoadOp>(loc, operandAdaptor.B(), wbIOFCIVs[0]);
      it = rewriter.create<AddFOp>(loc, it, loadWB);
      Value loadRB =
          rewriter.create<LoadOp>(loc, operandAdaptor.B(), rbIOFCIVs[0]);
      it = rewriter.create<AddFOp>(loc, it, loadRB);
    }
    it = applyActivation(rewriter, loc, activationPack.f, it);

    // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
    Value loadXWF = rewriter.create<LoadOp>(loc, xwIOFC[2]);
    Value loadHRF = rewriter.create<LoadOp>(loc, hrIOFC[2]);
    Value ft = rewriter.create<AddFOp>(loc, loadXWF, loadHRF);
    if (hasPeepholes) {
      Value loadP =
          rewriter.create<LoadOp>(loc, operandAdaptor.P(), pIOFIVs[2]);
      Value PC = rewriter.create<MulFOp>(loc, loadP, loadC);
      ft = rewriter.create<AddFOp>(loc, ft, PC);
    }
    if (hasBiasForInput) {
      Value loadWB =
          rewriter.create<LoadOp>(loc, operandAdaptor.B(), wbIOFCIVs[2]);
      ft = rewriter.create<AddFOp>(loc, ft, loadWB);
      Value loadRB =
          rewriter.create<LoadOp>(loc, operandAdaptor.B(), rbIOFCIVs[2]);
      ft = rewriter.create<AddFOp>(loc, ft, loadRB);
    }
    ft = applyActivation(rewriter, loc, activationPack.f, ft);

    // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
    Value loadXWC = rewriter.create<LoadOp>(loc, xwIOFC[3]);
    Value loadHRC = rewriter.create<LoadOp>(loc, hrIOFC[3]);
    Value ct = rewriter.create<AddFOp>(loc, loadXWC, loadHRC);
    if (hasBiasForInput) {
      Value loadWB =
          rewriter.create<LoadOp>(loc, operandAdaptor.B(), wbIOFCIVs[3]);
      ct = rewriter.create<AddFOp>(loc, ct, loadWB);
      Value loadRB =
          rewriter.create<LoadOp>(loc, operandAdaptor.B(), rbIOFCIVs[3]);
      ct = rewriter.create<AddFOp>(loc, ct, loadRB);
    }
    ct = applyActivation(rewriter, loc, activationPack.g, ct);

    // Ct = ft (.) Ct-1 + it (.) ct
    Value FtCt1 = rewriter.create<MulFOp>(loc, ft, loadC);
    Value itct = rewriter.create<MulFOp>(loc, it, ct);
    Value Ct = rewriter.create<AddFOp>(loc, FtCt1, itct);
    rewriter.create<StoreOp>(loc, Ct, state.ct, cIVs);

    // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
    Value loadXWO = rewriter.create<LoadOp>(loc, xwIOFC[1]);
    Value loadHRO = rewriter.create<LoadOp>(loc, hrIOFC[1]);
    Value ot = rewriter.create<AddFOp>(loc, loadXWO, loadHRO);
    if (hasPeepholes) {
      Value loadP =
          rewriter.create<LoadOp>(loc, operandAdaptor.P(), pIOFIVs[1]);
      Value PC = rewriter.create<MulFOp>(loc, loadP, Ct);
      ot = rewriter.create<AddFOp>(loc, ot, PC);
    }
    if (hasBiasForInput) {
      Value loadWB =
          rewriter.create<LoadOp>(loc, operandAdaptor.B(), wbIOFCIVs[1]);
      ot = rewriter.create<AddFOp>(loc, ot, loadWB);
      Value loadRB =
          rewriter.create<LoadOp>(loc, operandAdaptor.B(), rbIOFCIVs[1]);
      ot = rewriter.create<AddFOp>(loc, ot, loadRB);
    }
    ot = applyActivation(rewriter, loc, activationPack.f, ot);

    // Ht = ot (.) h(Ct)
    Value hCt = applyActivation(rewriter, loc, activationPack.h, Ct);
    Value Ht = rewriter.create<MulFOp>(loc, ot, hCt);
    rewriter.create<StoreOp>(loc, Ht, state.ht, hIVs);

    // Store the current Ht if required.
    if (!state.allH.getType().isa<NoneType>()) {
      SmallVector<Value, 4> allHIVs;
      allHIVs.emplace_back(sequenceLengthIV);
      allHIVs.emplace_back(numDirectionIV);
      allHIVs.emplace_back(batchSizeIV);
      allHIVs.emplace_back(hiddenSizeIV);
      rewriter.create<StoreOp>(loc, Ht, state.allH, allHIVs);
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
  outputs.emplace_back(
      (op->Y().getType().isa<NoneType>() ? noneValue : state.allH));
  outputs.emplace_back(
      (op->Y_h().getType().isa<NoneType>() ? noneValue : state.ht));
  outputs.emplace_back(
      (op->Y_c().getType().isa<NoneType>() ? noneValue : state.ct));
}

void populateLoweringONNXLSTMOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXRNNOpLowering<ONNXLSTMOp, LstmState, LstmActivationPack>>(
      ctx);
}
