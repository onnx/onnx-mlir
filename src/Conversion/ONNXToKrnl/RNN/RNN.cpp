//===----------------- RNN.cpp - Lowering RNN Op --------------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX RNN Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/RNN/RNNBase.hpp"

using namespace mlir;

struct RnnState {
  Value allH;
  Value ht;
};

struct RnnActivationPack {
  RNNActivation f;
};

template <>
bool hasAllNoneOutput<ONNXRNNOp>(ONNXRNNOp *op) {
  return (isNoneType(op->Y()) && isNoneType(op->Y_h()));
}

template <>
std::tuple<RnnActivationPack, RnnActivationPack>
getActivationPack<ONNXRNNOp, RnnActivationPack>(ONNXRNNOp *op) {
  auto direction = op->direction();
  auto activations = op->activations();
  auto activationAlpha = op->activation_alpha();
  auto activationBeta = op->activation_beta();

  RnnActivationPack activationForward, activationReverse;

  // Get activation function name.
  // Default forward functions
  activationForward.f.name = "tanh";
  // Default backward functions
  activationReverse.f.name = "tanh";
  if (activations) {
    ArrayRef<Attribute> activationArrAttr = activations.getValue();
    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      // Forward activations.
      if (activationArrAttr.size() > 0) {
        activationForward.f.name =
            activationArrAttr[0].cast<StringAttr>().getValue();
      }
    }

    // Reverse activations.
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      int startIndex = (direction == REVERSE) ? 0 : 1;
      if (activationArrAttr.size() > startIndex) {
        activationReverse.f.name =
            activationArrAttr[startIndex].cast<StringAttr>().getValue();
      }
    }
  }

  // Get alpha attributes.
  if (activationAlpha) {
    ArrayRef<Attribute> activationArrAttr = activationAlpha.getValue();
    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      // Forward activations.
      if (activationArrAttr.size() > 0) {
        activationForward.f.alpha = activationArrAttr[0].cast<FloatAttr>();
      }
    }

    // Reverse activations.
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      int startIndex = (direction == REVERSE) ? 0 : 1;
      if (activationArrAttr.size() > startIndex) {
        activationReverse.f.alpha =
            activationArrAttr[startIndex].cast<FloatAttr>();
      }
    }
  }

  // Get beta attributes.
  if (activationBeta) {
    ArrayRef<Attribute> activationArrAttr = activationBeta.getValue();
    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      // Forward activations.
      if (activationArrAttr.size() > 0) {
        activationForward.f.beta = activationArrAttr[0].cast<FloatAttr>();
      }
    }

    // Reverse activations.
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      int startIndex = (direction == REVERSE) ? 0 : 1;
      if (activationArrAttr.size() > startIndex) {
        activationReverse.f.beta =
            activationArrAttr[startIndex].cast<FloatAttr>();
      }
    }
  }

  return std::make_tuple(activationForward, activationReverse);
}

template <>
RnnState allocAndInitializeStates<ONNXRNNOp, RnnState>(
    ConversionPatternRewriter &rewriter, Location loc, ONNXRNNOp *op,
    typename ONNXRNNOp::Adaptor operandAdaptor) {
  RnnState state;

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
  return state;
}

template <>
void calculateState<ONNXRNNOp, RnnState, RnnActivationPack>(
    ConversionPatternRewriter &rewriter, Location loc,
    typename ONNXRNNOp::Adaptor operandAdaptor, RnnState state,
    RnnActivationPack activationPack, Value directionIV, Value sequenceIV) {

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

  // Prepare AffineMap to access bias tensor.
  AffineMap accessByOffsetMap;
  {
    AffineExpr iv = rewriter.getAffineDimExpr(0);
    AffineExpr index = rewriter.getAffineSymbolExpr(0);
    AffineExpr size = rewriter.getAffineSymbolExpr(1);
    AffineExpr accessByOffsetExpr = index * size + iv;
    accessByOffsetMap = AffineMap::get(1, 2, accessByOffsetExpr);
  }

  // Prepare constant indices.
  SmallVector<Value, 2> constantIndices;
  for (int i = 0; i < 2; i++)
    constantIndices.emplace_back(
        emitConstantOp(rewriter, loc, rewriter.getIndexType(), i));

  // Equations for RNN.
  // Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
  //
  // The following code will emit loops as follows:
  // for b in 0 .. BatchDimSize
  //   for h in 0 .. HiddenDimSize {
  //     for i in 0 .. InputDimSize {
  //       compute Xt*(Wi^T), Ht-1*(Ri^T)
  //     }
  //     compute Ht
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

    // IVs for the hidden state tensor.
    // H :: [num_directions, batch_size, hidden_size]
    SmallVector<Value, 3> hIVs = {directionIV, batchIV, hiddenIV};
    // IVs for the bias tensors for W and R.
    // Bias [Wb[i], Rb[i]] :: [num_directions, 2*hidden_size]
    SmallVector<Value, 2> wbiIVs, rbiIVs;
    if (hasBiasForInput) {
      // Wb[i]
      Value wHiddenIV = rewriter.create<AffineApplyOp>(loc, accessByOffsetMap,
          std::vector<Value>{/*iv=*/hiddenIV,
              /*index=*/constantIndices[0], /*size=*/hiddenDimVal});
      wbiIVs = {directionIV, wHiddenIV};
      // Rb[i]
      Value rHiddenIV = rewriter.create<AffineApplyOp>(loc, accessByOffsetMap,
          std::vector<Value>{/*iv=*/hiddenIV,
              /*index=*/constantIndices[1], /*size=*/hiddenDimVal});
      rbiIVs = {directionIV, rHiddenIV};
    }

    // Emit instructions for matrix multiplications:
    //   Xt*(Wi^T), Ht-1*(Ri^T)
    Value loadH = rewriter.create<AffineLoadOp>(loc, state.ht, hIVs);
    // Allocate memory for storing matrix multiplication results.
    Value zero = emitConstantOp(rewriter, loc, elementType, 0);
    MemRefType scalarMemRefType = MemRefType::get({}, elementType, {}, 0);
    Value xwi = rewriter.create<AllocOp>(loc, scalarMemRefType);
    rewriter.create<AffineStoreOp>(loc, zero, xwi, ArrayRef<Value>{});
    //
    Value hri = rewriter.create<AllocOp>(loc, scalarMemRefType);
    rewriter.create<AffineStoreOp>(loc, zero, hri, ArrayRef<Value>{});

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
        // X :: [seq_length, batch_size, input_size]
        SmallVector<Value, 3> xIVs = {sequenceIV, batchIV, reductionIV};
        // W[i] :: [num_directions, hidden_size, input_size]
        // R[i] :: [num_directions, hidden_size, input_size]
        SmallVector<Value, 3> wiIVs = {directionIV, hiddenIV, reductionIV};
        SmallVector<Value, 3> riIVs = {directionIV, hiddenIV, reductionIV};

        // Emit intructions for matrix multiplication.
        Value loadX =
            rewriter.create<AffineLoadOp>(loc, operandAdaptor.X(), xIVs);
        // Xt * Wi
        Value loadW =
            rewriter.create<AffineLoadOp>(loc, operandAdaptor.W(), wiIVs);
        Value xwVal = rewriter.create<MulFOp>(loc, loadX, loadW);
        Value loadXW = rewriter.create<AffineLoadOp>(loc, xwi);
        Value nextXW = rewriter.create<AddFOp>(loc, loadXW, xwVal);
        rewriter.create<AffineStoreOp>(loc, nextXW, xwi, ArrayRef<Value>{});
        // Ht-1 * Ri
        Value loadR =
            rewriter.create<AffineLoadOp>(loc, operandAdaptor.R(), riIVs);
        Value hrVal = rewriter.create<MulFOp>(loc, loadH, loadR);
        Value loadHR = rewriter.create<AffineLoadOp>(loc, hri);
        Value nextHR = rewriter.create<AddFOp>(loc, loadHR, hrVal);
        rewriter.create<AffineStoreOp>(loc, nextHR, hri, ArrayRef<Value>{});
      }
      rewriter.restoreInsertionPoint(ipReductionLoops);
    }

    // Emit instructions for 'Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)'
    Value loadXWI = rewriter.create<AffineLoadOp>(loc, xwi);
    Value loadHRI = rewriter.create<AffineLoadOp>(loc, hri);
    Value Ht = rewriter.create<AddFOp>(loc, loadXWI, loadHRI);
    if (hasBiasForInput) {
      Value loadWB =
          rewriter.create<AffineLoadOp>(loc, operandAdaptor.B(), wbiIVs);
      Ht = rewriter.create<AddFOp>(loc, Ht, loadWB);
      Value loadRB =
          rewriter.create<AffineLoadOp>(loc, operandAdaptor.B(), rbiIVs);
      Ht = rewriter.create<AddFOp>(loc, Ht, loadRB);
    }
    Ht = applyActivation(rewriter, loc, activationPack.f, Ht);
    rewriter.create<AffineStoreOp>(loc, Ht, state.ht, hIVs);

    // Store the current Ht if required.
    if (!isNoneType(state.allH)) {
      SmallVector<Value, 4> allHIVs{sequenceIV, directionIV, batchIV, hiddenIV};
      rewriter.create<AffineStoreOp>(loc, Ht, state.allH, allHIVs);
    }

    // Deallocate the temporary results of matrix multiplications.
    rewriter.create<DeallocOp>(loc, xwi);
    rewriter.create<DeallocOp>(loc, hri);
  }
}

template <>
void stateToOutput<ONNXRNNOp, RnnState>(
    ONNXRNNOp *op, RnnState state, std::vector<Value> &outputs) {
  Value noneValue;
  outputs.emplace_back((isNoneType(op->Y()) ? noneValue : state.allH));
  outputs.emplace_back((isNoneType(op->Y_h()) ? noneValue : state.ht));
}

void populateLoweringONNXRNNOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXRNNOpLowering<ONNXRNNOp, RnnState, RnnActivationPack>>(
      ctx);
}
