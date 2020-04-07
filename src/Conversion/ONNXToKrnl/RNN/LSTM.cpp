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

struct LstmInputPack {
  Value X;
  Value W;
  Value R;
  Value Biofc;
  Value sequenceLength;
  Value initialHidden;
  Value initialCell;
  Value Piof;
};

struct LstmState {
  Value allH;
  Value ht;
  Value ct;
};

template <>
LstmInputPack getInputPack<LstmInputPack>(ArrayRef<Value> operands) {
  LstmInputPack inputPack;
  inputPack.X = operands[0];
  inputPack.W = operands[1];
  inputPack.R = operands[2];
  inputPack.Biofc = operands[3];
  inputPack.sequenceLength = operands[4];
  inputPack.initialHidden = operands[5];
  inputPack.initialCell = operands[6];
  inputPack.Piof = operands[7];
  return inputPack;
}

template <>
bool hasNoOutput<ONNXLSTMOp>(Operation *op) {
  ONNXLSTMOp rnnOp = llvm::dyn_cast<ONNXLSTMOp>(op);
  return (rnnOp.Y().getType().isa<NoneType>() &&
          rnnOp.Y_h().getType().isa<NoneType>() &&
          rnnOp.Y_c().getType().isa<NoneType>());
}

template <>
LstmState allocAndInitializeStates<ONNXLSTMOp, LstmInputPack>(
    ConversionPatternRewriter &rewriter, Location loc, Operation *op,
    LstmInputPack inputPack) {
  LstmState state;

  ONNXLSTMOp rnnOp = llvm::dyn_cast<ONNXLSTMOp>(op);

  // Insert allocation and deallocation for the results of this operation.
  if (!rnnOp.Y().getType().isa<NoneType>()) {
    auto yMemRefType = convertToMemRefType(rnnOp.Y().getType());
    if (hasAllConstantDimensions(yMemRefType))
      state.allH = insertAllocAndDealloc(
          yMemRefType, loc, rewriter, checkInsertDealloc(op, 0));
    else
      emitError(loc, "Unsupported dynamic dimensions.");
  }

  if (!rnnOp.Y_h().getType().isa<NoneType>()) {
    auto yhMemRefType = convertToMemRefType(rnnOp.Y_h().getType());
    if (hasAllConstantDimensions(yhMemRefType))
      state.ht = insertAllocAndDealloc(
          yhMemRefType, loc, rewriter, checkInsertDealloc(op, 1));
    else
      emitError(loc, "Unsupported dynamic dimensions.");
  } else {
    SmallVector<int64_t, 3> yhDims;
    yhDims.emplace_back(inputPack.W.getType().cast<ShapedType>().getShape()[0]);
    yhDims.emplace_back(inputPack.X.getType().cast<ShapedType>().getShape()[1]);
    yhDims.emplace_back(inputPack.R.getType().cast<ShapedType>().getShape()[2]);
    auto yhMemRefType = MemRefType::get(
        yhDims, inputPack.X.getType().cast<ShapedType>().getElementType());
    state.ht = insertAllocAndDealloc(yhMemRefType, loc, rewriter, true);
  }

  if (!rnnOp.Y_c().getType().isa<NoneType>()) {
    auto ycMemRefType = convertToMemRefType(rnnOp.Y_c().getType());
    if (hasAllConstantDimensions(ycMemRefType))
      state.ct = insertAllocAndDealloc(
          ycMemRefType, loc, rewriter, checkInsertDealloc(op, 2));
    else
      emitError(loc, "Unsupported dynamic dimensions.");
  } else {
    SmallVector<int64_t, 3> ycDims;
    ycDims.emplace_back(inputPack.W.getType().cast<ShapedType>().getShape()[0]);
    ycDims.emplace_back(inputPack.X.getType().cast<ShapedType>().getShape()[1]);
    ycDims.emplace_back(inputPack.R.getType().cast<ShapedType>().getShape()[2]);
    auto ycMemRefType = MemRefType::get(
        ycDims, inputPack.X.getType().cast<ShapedType>().getElementType());
    state.ct = insertAllocAndDealloc(ycMemRefType, loc, rewriter, true);
  }

  // Initialize ht and ct.
  Value zero = emitConstantOp(rewriter, loc,
      inputPack.X.getType().cast<ShapedType>().getElementType(), 0);
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
        (!inputPack.initialHidden.getType().isa<NoneType>())
            ? rewriter.create<LoadOp>(loc, inputPack.initialHidden, IVs)
            : zero;
    rewriter.create<StoreOp>(loc, hiddenVal, state.ht, IVs);
    Value cellVal =
        (!inputPack.initialCell.getType().isa<NoneType>())
            ? rewriter.create<LoadOp>(loc, inputPack.initialCell, IVs)
            : zero;
    rewriter.create<StoreOp>(loc, cellVal, state.ct, IVs);
  }
  rewriter.restoreInsertionPoint(ipInitializationLoops);
  return state;
}

template <>
void calculateState<ONNXLSTMOp, LstmInputPack, LstmState>(
    ConversionPatternRewriter &rewriter, Location loc, Operation *op,
    Value numDirectionIV, Value sequenceIV, LstmInputPack inputPack,
    LstmState state) {
  ONNXLSTMOp rnnOp = llvm::dyn_cast<ONNXLSTMOp>(op);

  bool hasBiasForInput =
      (inputPack.Biofc.getType().isa<NoneType>()) ? false : true;
  bool hasPeepholes = (inputPack.Piof.getType().isa<NoneType>()) ? false : true;
  bool hasSequenceLengths =
      (inputPack.sequenceLength.getType().isa<NoneType>()) ? false : true;
  if (hasSequenceLengths)
    emitError(loc, "Does not support sequence_lens at this time");

  auto elementType = inputPack.X.getType().cast<ShapedType>().getElementType();
  MemRefType scalarMemRefType = MemRefType::get({}, elementType, {}, 0);

  // Equations for LSTM.
  // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
  // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
  // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
  // Ct = ft (.) Ct-1 + it (.) ct
  // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
  // Ht = ot (.) h(Ct)

  BuildKrnlLoop stateLoops(rewriter, loc, 2);
  stateLoops.createDefineAndOptimizeOp();
  stateLoops.pushBounds(0, state.ht.getType().cast<ShapedType>().getShape()[1]);
  stateLoops.pushBounds(0, state.ht.getType().cast<ShapedType>().getShape()[2]);
  stateLoops.createIterateOp();

  rewriter.setInsertionPointToStart(stateLoops.getIterateBlock());
  {
    auto inputSizeDim = inputPack.X.getType().cast<ShapedType>().getShape()[2];
    auto hiddenSizeDim = inputPack.R.getType().cast<ShapedType>().getShape()[2];
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
      hIVs.emplace_back(numDirectionIV);
      hIVs.emplace_back(batchSizeIV);
      hIVs.emplace_back(hiddenSizeIV);
      // C :: [num_directions, batch_size, hidden_size]
      cIVs.emplace_back(numDirectionIV);
      cIVs.emplace_back(batchSizeIV);
      cIVs.emplace_back(hiddenSizeIV);

      // Bias [Wb[iofc], Rb[iofc]] :: [num_directions, 8*hidden_size]
      if (hasBiasForInput) {
        Value hiddenSizeVal = emitConstantOp(
            rewriter, loc, rewriter.getIndexType(), hiddenSizeDim);
        // Wb[iofc]
        for (unsigned i = 0; i < 4; ++i) {
          SmallVector<Value, 4> wbIVs;
          Value index =
              emitConstantOp(rewriter, loc, rewriter.getIndexType(), i);
          Value offsetIV = rewriter.create<MulIOp>(loc, hiddenSizeVal, index);
          Value wHiddenIV =
              rewriter.create<AddIOp>(loc, offsetIV, hiddenSizeIV);
          wbIVs.emplace_back(numDirectionIV);
          wbIVs.emplace_back(wHiddenIV);
          wbIOFCIVs.emplace_back(wbIVs);
        }
        // Rb[iofc]
        for (unsigned i = 4; i < 8; ++i) {
          SmallVector<Value, 4> rbIVs;
          Value index =
              emitConstantOp(rewriter, loc, rewriter.getIndexType(), i);
          Value offsetIV = rewriter.create<MulIOp>(loc, hiddenSizeVal, index);
          Value rHiddenIV =
              rewriter.create<AddIOp>(loc, offsetIV, hiddenSizeIV);
          // [Wb[iofc], Rb[iofc]] :: [num_directions, 8*hidden_size]
          rbIVs.emplace_back(numDirectionIV);
          rbIVs.emplace_back(rHiddenIV);
          rbIOFCIVs.emplace_back(rbIVs);
        }
      }

      // Peepholes.
      if (hasPeepholes) {
        Value hiddenSizeVal = emitConstantOp(
            rewriter, loc, rewriter.getIndexType(), hiddenSizeDim);
        for (unsigned i = 0; i < 3; ++i) {
          SmallVector<Value, 4> pIVs;
          Value index =
              emitConstantOp(rewriter, loc, rewriter.getIndexType(), i);
          Value offsetIV = rewriter.create<MulIOp>(loc, hiddenSizeVal, index);
          Value pHiddenIV =
              rewriter.create<AddIOp>(loc, offsetIV, hiddenSizeIV);
          // P[iof] :: [num_directions, 3*hidden_size]
          pIVs.emplace_back(numDirectionIV);
          pIVs.emplace_back(pHiddenIV);
          pIOFIVs.emplace_back(pIVs);
        }
      }
    }

    Value loadH = rewriter.create<LoadOp>(loc, state.ht, hIVs);
    Value loadC = rewriter.create<LoadOp>(loc, state.ct, cIVs);

    // Compute temporary results for matrix multiplications:
    //   Xt*(Wi^T), Xt*(Wo^T), Xt*(Wf^t), Xt*(Wc^T)
    //   Ht-1*(Ri^T), Ht-1*(Ro^T), Ht-1*(Rf^t), Ht-1*(Rc^T)
    SmallVector<Value, 4> xwIOFC, hrIOFC;

    // Allocate memory for the temporary results and initialize them.
    Value zero = emitConstantOp(rewriter, loc, elementType, 0);
    for (unsigned i = 0; i < 4; ++i) {
      Value xwAlloc = rewriter.create<AllocOp>(loc, scalarMemRefType);
      rewriter.create<StoreOp>(loc, zero, xwAlloc);
      Value hrAlloc = rewriter.create<AllocOp>(loc, scalarMemRefType);
      rewriter.create<StoreOp>(loc, zero, hrAlloc);
      xwIOFC.emplace_back(xwAlloc);
      hrIOFC.emplace_back(hrAlloc);
    }

    { // Emit computation for matrix multiplications.
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
        xIVs.emplace_back(sequenceIV);
        xIVs.emplace_back(batchSizeIV);
        xIVs.emplace_back(reductionIV);

        // W and R are transposed.
        Value hiddenSizeVal = emitConstantOp(
            rewriter, loc, rewriter.getIndexType(), hiddenSizeDim);
        for (unsigned i = 0; i < 4; ++i) {
          SmallVector<Value, 4> wIVs, rIVs;
          Value index =
              emitConstantOp(rewriter, loc, rewriter.getIndexType(), i);
          Value offsetIV = rewriter.create<MulIOp>(loc, hiddenSizeVal, index);
          Value wHiddenIV =
              rewriter.create<AddIOp>(loc, offsetIV, hiddenSizeIV);

          // W[iofc] :: [num_directions, 4*hidden_size, input_size]
          wIVs.emplace_back(numDirectionIV);
          wIVs.emplace_back(wHiddenIV);
          wIVs.emplace_back(reductionIV);
          wIOFCIVs.emplace_back(wIVs);
          // R[iofc] :: [num_directions, 4*hidden_size, input_size]
          rIVs.emplace_back(numDirectionIV);
          rIVs.emplace_back(wHiddenIV);
          rIVs.emplace_back(reductionIV);
          rIOFCIVs.emplace_back(rIVs);
        }

        Value loadX = rewriter.create<LoadOp>(loc, inputPack.X, xIVs);
        for (unsigned i = 0; i < 4; ++i) {
          // Xt * Wiofc
          Value loadW = rewriter.create<LoadOp>(loc, inputPack.W, wIOFCIVs[i]);
          Value xwVal = rewriter.create<MulFOp>(loc, loadX, loadW);
          Value loadXW = rewriter.create<LoadOp>(loc, xwIOFC[i]);
          Value nextXW = rewriter.create<AddFOp>(loc, loadXW, xwVal);
          rewriter.create<StoreOp>(loc, nextXW, xwIOFC[i]);
          // Ht-1 * Riofc
          Value loadR = rewriter.create<LoadOp>(loc, inputPack.R, rIOFCIVs[i]);
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
      Value loadP = rewriter.create<LoadOp>(loc, inputPack.Piof, pIOFIVs[0]);
      Value PC = rewriter.create<MulFOp>(loc, loadP, loadC);
      it = rewriter.create<AddFOp>(loc, it, PC);
    }
    if (hasBiasForInput) {
      Value loadWB =
          rewriter.create<LoadOp>(loc, inputPack.Biofc, wbIOFCIVs[0]);
      it = rewriter.create<AddFOp>(loc, it, loadWB);
      Value loadRB =
          rewriter.create<LoadOp>(loc, inputPack.Biofc, rbIOFCIVs[0]);
      it = rewriter.create<AddFOp>(loc, it, loadRB);
    }
    // TODO
    it = activation_f(rewriter, loc, op, it, elementType);

    // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
    Value loadXWF = rewriter.create<LoadOp>(loc, xwIOFC[2]);
    Value loadHRF = rewriter.create<LoadOp>(loc, hrIOFC[2]);
    Value ft = rewriter.create<AddFOp>(loc, loadXWF, loadHRF);
    if (hasPeepholes) {
      Value loadP = rewriter.create<LoadOp>(loc, inputPack.Piof, pIOFIVs[2]);
      Value PC = rewriter.create<MulFOp>(loc, loadP, loadC);
      ft = rewriter.create<AddFOp>(loc, ft, PC);
    }
    if (hasBiasForInput) {
      Value loadWB =
          rewriter.create<LoadOp>(loc, inputPack.Biofc, wbIOFCIVs[2]);
      ft = rewriter.create<AddFOp>(loc, ft, loadWB);
      Value loadRB =
          rewriter.create<LoadOp>(loc, inputPack.Biofc, rbIOFCIVs[2]);
      ft = rewriter.create<AddFOp>(loc, ft, loadRB);
    }
    // TODO
    ft = activation_f(rewriter, loc, op, ft, elementType);

    // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
    Value loadXWC = rewriter.create<LoadOp>(loc, xwIOFC[3]);
    Value loadHRC = rewriter.create<LoadOp>(loc, hrIOFC[3]);
    Value ct = rewriter.create<AddFOp>(loc, loadXWC, loadHRC);
    if (hasBiasForInput) {
      Value loadWB =
          rewriter.create<LoadOp>(loc, inputPack.Biofc, wbIOFCIVs[3]);
      ct = rewriter.create<AddFOp>(loc, ct, loadWB);
      Value loadRB =
          rewriter.create<LoadOp>(loc, inputPack.Biofc, rbIOFCIVs[3]);
      ct = rewriter.create<AddFOp>(loc, ct, loadRB);
    }
    // TODO
    ct = activation_g(rewriter, loc, op, ct, elementType);

    // Ct = ft (.) Ct-1 + it (.) ct
    Value Ct =
        rewriter.create<AddFOp>(loc, rewriter.create<MulFOp>(loc, ft, loadC),
            rewriter.create<MulFOp>(loc, it, ct));
    rewriter.create<StoreOp>(loc, Ct, state.ct, cIVs);

    // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
    Value loadXWO = rewriter.create<LoadOp>(loc, xwIOFC[1]);
    Value loadHRO = rewriter.create<LoadOp>(loc, hrIOFC[1]);
    Value ot = rewriter.create<AddFOp>(loc, loadXWO, loadHRO);
    if (hasPeepholes) {
      Value loadP = rewriter.create<LoadOp>(loc, inputPack.Piof, pIOFIVs[1]);
      Value PC = rewriter.create<MulFOp>(loc, loadP, Ct);
      ot = rewriter.create<AddFOp>(loc, ot, PC);
    }
    if (hasBiasForInput) {
      Value loadWB =
          rewriter.create<LoadOp>(loc, inputPack.Biofc, wbIOFCIVs[1]);
      ot = rewriter.create<AddFOp>(loc, ot, loadWB);
      Value loadRB =
          rewriter.create<LoadOp>(loc, inputPack.Biofc, rbIOFCIVs[1]);
      ot = rewriter.create<AddFOp>(loc, ot, loadRB);
    }
    // TODO
    ot = activation_f(rewriter, loc, op, ot, elementType);

    // Ht = ot (.) h(Ct)
    Value Ht = rewriter.create<MulFOp>(
        loc, ot, activation_h(rewriter, loc, op, Ct, elementType));
    rewriter.create<StoreOp>(loc, Ht, state.ht, hIVs);

    // Deallocate the temporary results.
    for (Value v : xwIOFC)
      rewriter.create<DeallocOp>(loc, v);
    for (Value v : hrIOFC)
      rewriter.create<DeallocOp>(loc, v);
  }
}

template <>
void stateToOutput<ONNXLSTMOp, LstmState>(
    Operation *op, LstmState state, std::vector<Value> &outputs) {
  ONNXLSTMOp rnnOp = llvm::dyn_cast<ONNXLSTMOp>(op);
  Value none_;
  outputs.emplace_back(
      (rnnOp.Y().getType().isa<NoneType>() ? none_ : state.allH));
  outputs.emplace_back(
      (rnnOp.Y_h().getType().isa<NoneType>() ? none_ : state.ht));
  outputs.emplace_back(
      (rnnOp.Y_c().getType().isa<NoneType>() ? none_ : state.ct));
}

void populateLoweringONNXLSTMOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXRNNOpLowering<ONNXLSTMOp, LstmInputPack, LstmState>>(ctx);
}
