//===--------------- LSTM.cpp - Lowering LSTM Op --------------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX LSTM Operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"

using namespace mlir;

static const std::string FORWARD = "forward";
static const std::string REVERSE = "reverse";
static const std::string BIDIRECTIONAL = "bidirectional";

Value activation_f(ConversionPatternRewriter &rewriter, Location loc,
    Value input, Type elementType) {
  auto zero = emitConstantOp(rewriter, loc, elementType, 0);
  auto one = emitConstantOp(rewriter, loc, elementType, 1);
  auto neg = rewriter.create<SubFOp>(loc, zero, input);
  auto negExp = rewriter.create<ExpOp>(loc, neg);
  auto result = rewriter.create<DivFOp>(
      loc, one, rewriter.create<AddFOp>(loc, one, negExp));
  return result;
}

Value activation_g(ConversionPatternRewriter &rewriter, Location loc,
    Value input, Type elementType) {
  auto zero = emitConstantOp(rewriter, loc, elementType, 0);
  auto two =  emitConstantOp(rewriter, loc, elementType, 2);
  auto neg = rewriter.create<SubFOp>(loc, zero, input);
  auto exp = rewriter.create<ExpOp>(loc, input);
  auto negExp = rewriter.create<ExpOp>(loc, neg);

  auto sinh = rewriter.create<DivFOp>(
      loc, rewriter.create<SubFOp>(loc, exp, negExp), two);
  auto cosh = rewriter.create<DivFOp>(
      loc, rewriter.create<AddFOp>(loc, exp, negExp), two);

  return rewriter.create<DivFOp>(loc, sinh, cosh);
}

Value activation_h(ConversionPatternRewriter &rewriter, Location loc,
    Value input, Type elementType) {
  auto zero = emitConstantOp(rewriter, loc, elementType, 0);
  auto two =  emitConstantOp(rewriter, loc, elementType, 2);
  auto neg = rewriter.create<SubFOp>(loc, zero, input);
  auto exp = rewriter.create<ExpOp>(loc, input);
  auto negExp = rewriter.create<ExpOp>(loc, neg);

  auto sinh = rewriter.create<DivFOp>(
      loc, rewriter.create<SubFOp>(loc, exp, negExp), two);
  auto cosh = rewriter.create<DivFOp>(
      loc, rewriter.create<AddFOp>(loc, exp, negExp), two);

  return rewriter.create<DivFOp>(loc, sinh, cosh);
}

template <typename RNNOp>
struct ONNXRNNOpLowering : public ConversionPattern {
  ONNXRNNOpLowering(MLIRContext *ctx)
      : ConversionPattern(RNNOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    RNNOp rnnOp = llvm::dyn_cast<RNNOp>(op);

    // Required inputs.
    Value X = operands[0];
    Value W = operands[1];
    Value R = operands[2];

    // Optional inputs.
    Value B = operands[3];
    Value sequenceLength = operands[4];
    Value initialHidden = operands[5];
    Value initialCell = operands[6];
    Value P = operands[7];
    bool hasBiasForInput = (B.getType().isa<NoneType>()) ? false : true;
    bool hasSequenceLengths =
        (sequenceLength.getType().isa<NoneType>()) ? false : true;
    bool hasInitialHidden =
        (initialHidden.getType().isa<NoneType>()) ? false : true;
    bool hasInitialCell =
        (initialCell.getType().isa<NoneType>()) ? false : true;
    bool hasPeepholes = (P.getType().isa<NoneType>()) ? false : true;
    if (hasSequenceLengths)
      emitError(loc, "Does not support sequence_lens at this time");

    // Optional outputs (0 - 3).
    // Y's shape :: [seq_length, num_directions, batch_size, hidden_size]
    Type yTy = rnnOp.Y().getType();
    // Y_h's shape :: [num_directions, batch_size, hidden_size]
    Type yhTy = rnnOp.Y_h().getType();
    // Y_c's shape :: [num_directions, batch_size, hidden_size]
    Type ycTy = rnnOp.Y_c().getType();

    bool returnAllHiddenStates = yTy.isa<NoneType>() ? false : true;
    bool returnLastHiddenState = yhTy.isa<NoneType>() ? false : true;
    bool returnLastCellState = ycTy.isa<NoneType>() ? false : true;

    // Delete this op if there is no output.
    if (!returnAllHiddenStates && !returnLastHiddenState &&
        !returnLastCellState) {
      rewriter.eraseOp(op);
      return success();
    }

    // xShape :: [seq_length, batch_size, input_size]
    auto xShape = X.getType().cast<ShapedType>().getShape();
    // wShape :: [num_directions, 4*hidden_size, input_size]
    auto wShape = W.getType().cast<ShapedType>().getShape();
    // rShape :: [num_directions, 4*hidden_size, hidden_size]
    auto rShape = R.getType().cast<ShapedType>().getShape();
    auto elementType = X.getType().cast<ShapedType>().getElementType();
    MemRefType scalarMemRefType = MemRefType::get({}, elementType, {}, 0);

    // Prepare necessary dimensions.
    auto sequenceLengthDim = xShape[0];
    auto batchSizeDim = xShape[1];
    auto inputSizeDim = xShape[2];
    auto hiddenSizeDim = rnnOp.hidden_size().getValue().getSExtValue();
    auto direction = rnnOp.direction();
    int numDirectionDim;
    if (direction == FORWARD || direction == REVERSE)
      numDirectionDim = 1;
    else if (direction == BIDIRECTIONAL)
      numDirectionDim = 2;
    else
      llvm_unreachable(
          "direction attribute must be one of forward, reverse and "
          "bidirectional");

    // Insert allocation and deallocation for the results of this operation.
    MemRefType yMemRefType, yhMemRefType, ycMemRefType;
    if (returnAllHiddenStates)
      yMemRefType = convertToMemRefType(yTy);
    if (returnLastHiddenState)
      yhMemRefType = convertToMemRefType(yhTy);
    if (returnLastCellState)
      ycMemRefType = convertToMemRefType(ycTy);

    Value allHiddenStates, lastHiddenState, lastCellState, none_;
    if (returnAllHiddenStates) {
      if (hasAllConstantDimensions(yMemRefType))
        allHiddenStates = insertAllocAndDealloc(
            yMemRefType, loc, rewriter, checkInsertDealloc(op, 0));
      else
        // TODO: add code.
        emitError(loc, "Unsupported dynamic dimensions.");
    }

    if (returnLastHiddenState) {
      if (hasAllConstantDimensions(yhMemRefType))
        lastHiddenState = insertAllocAndDealloc(
            yhMemRefType, loc, rewriter, checkInsertDealloc(op, 1));
      else
        // TODO: add code.
        emitError(loc, "Unsupported dynamic dimensions.");
    } else {
      SmallVector<int64_t, 3> yhDims;
      yhDims.emplace_back(numDirectionDim);
      yhDims.emplace_back(batchSizeDim);
      yhDims.emplace_back(hiddenSizeDim);
      yhMemRefType = MemRefType::get(yhDims, elementType);
      lastHiddenState =
          insertAllocAndDealloc(yhMemRefType, loc, rewriter, true);
    }

    if (returnLastCellState) {
      if (hasAllConstantDimensions(ycMemRefType))
        lastCellState = insertAllocAndDealloc(
            ycMemRefType, loc, rewriter, checkInsertDealloc(op, 2));
      else
        // TODO: add code.
        emitError(loc, "Unsupported dynamic dimensions.");
    } else {
      SmallVector<int64_t, 3> ycDims;
      ycDims.emplace_back(numDirectionDim);
      ycDims.emplace_back(batchSizeDim);
      ycDims.emplace_back(hiddenSizeDim);
      ycMemRefType = MemRefType::get(ycDims, elementType);
      lastCellState =
          insertAllocAndDealloc(ycMemRefType, loc, rewriter, true);
    }

    // Compute states
    // for t in [0..sequenceLength]:
    //   it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
    //   ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
    //   ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
    //   Ct = ft (.) Ct-1 + it (.) ct
    //   ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
    //   Ht = ot (.) h(Ct)

    // Initialize H and C.
    Value zero = emitConstantOp(rewriter, loc, elementType, 0);
    BuildKrnlLoop initializationLoops(rewriter, loc, 3);
    initializationLoops.createDefineOptimizeAndIterateOp(lastHiddenState);
    auto ipInitializationLoops = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(initializationLoops.getIterateBlock());
    {
      SmallVector<Value, 4> IVs;
      for (int i = 0; i < 3; ++i)
        IVs.emplace_back(initializationLoops.getInductionVar(i));
      Value hiddenVal = (hasInitialHidden)
                            ? rewriter.create<LoadOp>(loc, initialHidden, IVs)
                            : zero;
      rewriter.create<StoreOp>(loc, hiddenVal, lastHiddenState, IVs);
      Value cellVal = (hasInitialCell)
                          ? rewriter.create<LoadOp>(loc, initialCell, IVs)
                          : zero;
      rewriter.create<StoreOp>(loc, cellVal, lastCellState, IVs);
    }
    rewriter.restoreInsertionPoint(ipInitializationLoops);

    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      // Define loops over sequence length.
      BuildKrnlLoop sequenceLoops(rewriter, loc, 1);
      sequenceLoops.createDefineAndOptimizeOp();
      sequenceLoops.pushBounds(0, sequenceLengthDim);
      sequenceLoops.createIterateOp();

      rewriter.setInsertionPointToStart(sequenceLoops.getIterateBlock());
      {
        auto sequenceIV = sequenceLoops.getInductionVar(0);
        // Emit calculation for one RNN step.
        // Fuse all calculations in one RNN step into one Krnl iterate.
        BuildKrnlLoop stateLoops(rewriter, loc, 3);
        stateLoops.createDefineAndOptimizeOp();
        stateLoops.pushBounds(0, numDirectionDim);
        stateLoops.pushBounds(0, batchSizeDim);
        stateLoops.pushBounds(0, hiddenSizeDim);
        stateLoops.createIterateOp();

        rewriter.setInsertionPointToStart(stateLoops.getIterateBlock());
        {
          auto numDirectionIV = stateLoops.getInductionVar(0);
          auto batchSizeIV = stateLoops.getInductionVar(1);
          auto hiddenSizeIV = stateLoops.getInductionVar(2);

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

            // Bias
            if (hasBiasForInput) {
              Value hiddenSizeVal = emitConstantOp(
                  rewriter, loc, rewriter.getIndexType(), hiddenSizeDim);
              // Wb[iofc]
              for (unsigned i = 0; i < 4; ++i) {
                SmallVector<Value, 4> wbIVs;
                Value index =
                    emitConstantOp(rewriter, loc, rewriter.getIndexType(), i);
                Value offsetIV =
                    rewriter.create<MulIOp>(loc, hiddenSizeVal, index);
                Value wHiddenIV =
                    rewriter.create<AddIOp>(loc, offsetIV, hiddenSizeIV);
                // [Wb[iofc], Rb[iofc]] :: [num_directions, 8*hidden_size]
                wbIVs.emplace_back(numDirectionIV);
                wbIVs.emplace_back(wHiddenIV);
                wbIOFCIVs.emplace_back(wbIVs);
              }
              // Rb[iofc]
              for (unsigned i = 4; i < 8; ++i) {
                SmallVector<Value, 4> rbIVs;
                Value index =
                    emitConstantOp(rewriter, loc, rewriter.getIndexType(), i);
                Value offsetIV =
                    rewriter.create<MulIOp>(loc, hiddenSizeVal, index);
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
                Value offsetIV =
                    rewriter.create<MulIOp>(loc, hiddenSizeVal, index);
                Value pHiddenIV =
                    rewriter.create<AddIOp>(loc, offsetIV, hiddenSizeIV);
                // P[iof] :: [num_directions, 3*hidden_size]
                pIVs.emplace_back(numDirectionIV);
                pIVs.emplace_back(pHiddenIV);
                pIOFIVs.emplace_back(pIVs);
              }
            }
          }

          // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
          // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
          // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
          // Ct = ft (.) Ct-1 + it (.) ct
          // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
          // Ht = ot (.) h(Ct)

          Value loadH = rewriter.create<LoadOp>(loc, lastHiddenState, hIVs);
          Value loadC = rewriter.create<LoadOp>(loc, lastCellState, cIVs);

          // Compute temporary results for matrix multiplications:
          //   Xt*(Wi^T), Xt*(Wo^T), Xt*(Wf^t), Xt*(Wc^T)
          //   Ht-1*(Ri^T), Ht-1*(Ro^T), Ht-1*(Rf^t), Ht-1*(Rc^T)
          SmallVector<Value, 4> xwIOFC, hrIOFC;

          // Allocate memory for the temporary results and initialize them.
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
                Value offsetIV =
                    rewriter.create<MulIOp>(loc, hiddenSizeVal, index);
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

              Value loadX = rewriter.create<LoadOp>(loc, X, xIVs);
              for (unsigned i = 0; i < 4; ++i) {
                // Xt * Wiofc
                Value loadW = rewriter.create<LoadOp>(loc, W, wIOFCIVs[i]);
                Value xwVal = rewriter.create<MulFOp>(loc, loadX, loadW);
                Value loadXW = rewriter.create<LoadOp>(loc, xwIOFC[i]);
                Value nextXW = rewriter.create<AddFOp>(loc, loadXW, xwVal);
                rewriter.create<StoreOp>(loc, nextXW, xwIOFC[i]);
                // Ht-1 * Riofc
                Value loadR = rewriter.create<LoadOp>(loc, R, rIOFCIVs[i]);
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
            Value loadP = rewriter.create<LoadOp>(loc, P, pIOFIVs[0]);
            Value PC = rewriter.create<MulFOp>(loc, loadP, loadC);
            it = rewriter.create<AddFOp>(loc, it, PC);
          }
          if (hasBiasForInput) {
            Value loadWB = rewriter.create<LoadOp>(loc, B, wbIOFCIVs[0]);
            it = rewriter.create<AddFOp>(loc, it, loadWB);
            Value loadRB = rewriter.create<LoadOp>(loc, B, rbIOFCIVs[0]);
            it = rewriter.create<AddFOp>(loc, it, loadRB);
          }
          // TODO
          it = activation_f(rewriter, loc, it, elementType);

          // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
          Value loadXWO = rewriter.create<LoadOp>(loc, xwIOFC[1]);
          Value loadHRO = rewriter.create<LoadOp>(loc, hrIOFC[1]);
          Value ot = rewriter.create<AddFOp>(loc, loadXWO, loadHRO);
          if (hasPeepholes) {
            Value loadP = rewriter.create<LoadOp>(loc, P, pIOFIVs[1]);
            Value PC = rewriter.create<MulFOp>(loc, loadP, loadC);
            ot = rewriter.create<AddFOp>(loc, ot, PC);
          }
          if (hasBiasForInput) {
            Value loadWB = rewriter.create<LoadOp>(loc, B, wbIOFCIVs[1]);
            ot = rewriter.create<AddFOp>(loc, ot, loadWB);
            Value loadRB = rewriter.create<LoadOp>(loc, B, rbIOFCIVs[1]);
            ot = rewriter.create<AddFOp>(loc, ot, loadRB);
          }
          // TODO
          ot = activation_f(rewriter, loc, ot, elementType);

          // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
          Value loadXWF = rewriter.create<LoadOp>(loc, xwIOFC[2]);
          Value loadHRF = rewriter.create<LoadOp>(loc, hrIOFC[2]);
          Value ft = rewriter.create<AddFOp>(loc, loadXWF, loadHRF);
          if (hasPeepholes) {
            Value loadP = rewriter.create<LoadOp>(loc, P, pIOFIVs[2]);
            Value PC = rewriter.create<MulFOp>(loc, loadP, loadC);
            ft = rewriter.create<AddFOp>(loc, ft, PC);
          }
          if (hasBiasForInput) {
            Value loadWB = rewriter.create<LoadOp>(loc, B, wbIOFCIVs[2]);
            ft = rewriter.create<AddFOp>(loc, ft, loadWB);
            Value loadRB = rewriter.create<LoadOp>(loc, B, rbIOFCIVs[2]);
            ft = rewriter.create<AddFOp>(loc, ft, loadRB);
          }
          // TODO
          ft = activation_f(rewriter, loc, ft, elementType);

          // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
          Value loadXWC = rewriter.create<LoadOp>(loc, xwIOFC[3]);
          Value loadHRC = rewriter.create<LoadOp>(loc, hrIOFC[3]);
          Value ct = rewriter.create<AddFOp>(loc, loadXWC, loadHRC);
          if (hasBiasForInput) {
            Value loadWB = rewriter.create<LoadOp>(loc, B, wbIOFCIVs[3]);
            ct = rewriter.create<AddFOp>(loc, ct, loadWB);
            Value loadRB = rewriter.create<LoadOp>(loc, B, rbIOFCIVs[3]);
            ct = rewriter.create<AddFOp>(loc, ct, loadRB);
          }
          // TODO
          ct = activation_g(rewriter, loc, ct, elementType);

          // Ct = ft (.) Ct-1 + it (.) ct
          Value Ct = rewriter.create<AddFOp>(loc,
              rewriter.create<MulFOp>(loc, ft, loadC),
              rewriter.create<MulFOp>(loc, it, ct));
          rewriter.create<StoreOp>(loc, Ct, lastCellState, cIVs);

          // Ht = ot (.) h(Ct)
          Value Ht = rewriter.create<MulFOp>(
              loc, ot, activation_h(rewriter, loc, Ct, elementType));
          rewriter.create<StoreOp>(loc, Ht, lastHiddenState, hIVs);

          // Deallocate the temporary results.
          for (Value v : xwIOFC)
            rewriter.create<DeallocOp>(loc, v);
          for (Value v : hrIOFC)
            rewriter.create<DeallocOp>(loc, v);
        }
      }

      rewriter.replaceOp(op, {allHiddenStates, lastHiddenState, lastCellState});
    }

    return success();
  }
};

void populateLoweringONNXLSTMOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXRNNOpLowering<ONNXLSTMOp>>(ctx);
}
