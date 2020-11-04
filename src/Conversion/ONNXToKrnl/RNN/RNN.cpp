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
  // Shape information:
  // Xt : [seq_length, batch_size, input_size]
  // Wi : [num_directions, hidden_size, input_size]
  // Ri : [num_directions, hidden_size, hidden_size]
  // Ht : [num_directions, batch_size, hidden_size]
  // Wbi: [num_directions, hidden_size]
  // Rbi: [num_directions, hidden_size]
  //
  // The following code will emit loops as follows:
  //     for b in 0 .. BatchDimSize
  //       for h in 0 .. HiddenDimSize
  //         for i in 0 .. InputDimSize
  //           compute Xt*(Wi^T)
  //         for i in 0 .. HiddenDimSize
  //           compute Ht-1*(Ri^T)
  //     for b in 0 .. BatchDimSize
  //       for h in 0 .. HiddenDimSize
  //         compute Ht
  //         update the hidden state with the new state Ht.
  //
  // The reason to have two loops at the top level is to avoid updating any
  // element of the hidden state while computing Ht-1*(Ri^T).

  // Create temporary buffers for
  //   - Xt*(Wi^T), Ht-1*(Ri^T)
  // These tensors have shape of [num_directions, batch_size, hidden_size],
  // similar to the shape of the hidden state. Thus, we use the shape of the
  // hidden state to allocate these buffers.
  auto htMemRefType = state.ht.getType().cast<MemRefType>();
  bool staticDimensions = hasAllConstantDimensions(htMemRefType);
  Value xwI, hrI;
  if (staticDimensions) {
    xwI = insertAllocAndDealloc(htMemRefType, loc, rewriter, false);
    hrI = insertAllocAndDealloc(htMemRefType, loc, rewriter, false);
  } else {
    xwI = insertAllocAndDealloc(htMemRefType, loc, rewriter, false, {state.ht});
    hrI = insertAllocAndDealloc(htMemRefType, loc, rewriter, false, {state.ht});
  }

  // Emit instructions for matrix multiplications: Xt*(Wi^T) and Ht-1*(Ri^T)
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
    SmallVector<Value, 3> IVs = {directionIV, batchIV, hiddenIV};

    // Initialize matrix multiplication result.
    Value zero = emitConstantOp(rewriter, loc, elementType, 0);
    rewriter.create<AffineStoreOp>(loc, zero, xwI, IVs);
    rewriter.create<AffineStoreOp>(loc, zero, hrI, IVs);

    { // Emit instructions for matrix multiplication Xt*(Wi^T).
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
        // X :: [seq_length, batch_size, input_size]
        SmallVector<Value, 3> xIVs = {sequenceIV, batchIV, reductionIV};
        // Wi :: [num_directions, hidden_size, input_size]
        SmallVector<Value, 3> wIIVs = {directionIV, hiddenIV, reductionIV};

        // Xt * Wi
        Value loadX =
            rewriter.create<AffineLoadOp>(loc, operandAdaptor.X(), xIVs);
        Value loadW =
            rewriter.create<AffineLoadOp>(loc, operandAdaptor.W(), wIIVs);
        Value xwVal = rewriter.create<MulFOp>(loc, loadX, loadW);
        Value loadXW = rewriter.create<AffineLoadOp>(loc, xwI, IVs);
        Value nextXW = rewriter.create<AddFOp>(loc, loadXW, xwVal);
        rewriter.create<AffineStoreOp>(loc, nextXW, xwI, IVs);
      }
      rewriter.restoreInsertionPoint(ipReductionLoops);
    }

    { // Emit instructions for matrix multiplication Ht-1*(Ri^T)
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
        // H :: [num_directions, batch_size, hidden_size]
        SmallVector<Value, 3> hIVs = {directionIV, batchIV, reductionIV};
        // Ri :: [num_directions, hidden_size, hidden_size]
        SmallVector<Value, 3> rIIVs = {directionIV, hiddenIV, reductionIV};

        // Ht-1 * Riofc
        Value loadH = rewriter.create<AffineLoadOp>(loc, state.ht, hIVs);
        Value loadR =
            rewriter.create<AffineLoadOp>(loc, operandAdaptor.R(), rIIVs);
        Value hrVal = rewriter.create<MulFOp>(loc, loadH, loadR);
        Value loadHR = rewriter.create<AffineLoadOp>(loc, hrI, IVs);
        Value nextHR = rewriter.create<AddFOp>(loc, loadHR, hrVal);
        rewriter.create<AffineStoreOp>(loc, nextHR, hrI, IVs);
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
    // IVs for the matrix multiplication results.
    // M :: [num_directions, batch_size, hidden_size] for matmul
    SmallVector<Value, 3> mIVs = {directionIV, batchIV, hiddenIV};

    // Emit instructions for 'Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)'
    Value loadXWI = rewriter.create<AffineLoadOp>(loc, xwI, mIVs);
    Value loadHRI = rewriter.create<AffineLoadOp>(loc, hrI, mIVs);
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
  }
  rewriter.restoreInsertionPoint(ipStateLoops);
  // Deallocate the temporary results of matrix multiplications.
  rewriter.create<DeallocOp>(loc, xwI);
  rewriter.create<DeallocOp>(loc, hrI);
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
