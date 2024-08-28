/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- LSTM.cpp - Lowering LSTM Op --------------------------===//
//
// Copyright 2023-2024
//
// =============================================================================
//
// This file lowers the ONNX LSTM Operators to Stablehlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXConversionCommon/RNN/LSTM.hpp"
#include "src/Conversion/ONNXToStablehlo/DialectBuilder.hpp"
#include "src/Conversion/ONNXToStablehlo/RNN/RNNBase.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace stablehlo {

struct LstmState {
  SmallVector<Value, 512> forwardAllH;
  SmallVector<Value, 512> reverseAllH;

  Value allHForward;
  Value allHReverse;

  Value ht;
  Value ct;
  // intermediate states.
  Value forwardHt;
  Value reverseHt;
  Value forwardCt;
  Value reverseCt;
};

template <>
std::tuple<LstmWeightPack, LstmWeightPack>
getWeightPack<ONNXLSTMOp, LstmWeightPack>(
    ConversionPatternRewriter &rewriter, Location loc, ONNXLSTMOp *op) {
  // Return values.
  LstmWeightPack weightForward, weightReverse;

  // parameter weight: [direction, 4*hiddenSize, inputSize]
  Value W = op->getW();
  // recurrence weight: [direction, 4*hiddenSize, hiddenSize]
  Value R = op->getR();
  // direction
  StringRef direction = op->getDirection();

  ArrayRef<int64_t> wShape = mlir::cast<ShapedType>(W.getType()).getShape();
  Type elementType = mlir::cast<ShapedType>(W.getType()).getElementType();
  int64_t hiddenSize = wShape[1] / 4;
  int64_t inputSize = wShape[2];

  // RankedTensorType types for parameter weights.
  auto w3DTy =
      RankedTensorType::get({1, 4 * hiddenSize, inputSize}, elementType);
  auto w2DTy = RankedTensorType::get({4 * hiddenSize, inputSize}, elementType);
  auto wTranspose2DTy =
      RankedTensorType::get({inputSize, 4 * hiddenSize}, elementType);
  SmallVector<Type, 4> w3D2Ty(2, w3DTy);

  // RankedTensorType types for recurrence weights.
  auto r3DTy =
      RankedTensorType::get({1, 4 * hiddenSize, hiddenSize}, elementType);
  auto r2DTy = RankedTensorType::get({4 * hiddenSize, hiddenSize}, elementType);
  auto rTranspose2DTy =
      RankedTensorType::get({hiddenSize, 4 * hiddenSize}, elementType);
  SmallVector<Type, 4> r3D2Ty(2, r3DTy);

  // Squeeze the direction axis from W and R.
  Value fW, bW, fR, bR;
  if (direction == FORWARD) {
    fW = foldOrEmitONNXSqueezeOpStablehlo(rewriter, loc, w2DTy, W, /*axis=*/0);
    fR = foldOrEmitONNXSqueezeOpStablehlo(rewriter, loc, r2DTy, R, /*axis=*/0);
  } else if (direction == REVERSE) {
    bW = foldOrEmitONNXSqueezeOpStablehlo(rewriter, loc, w2DTy, W, /*axis=*/0);
    bR = foldOrEmitONNXSqueezeOpStablehlo(rewriter, loc, r2DTy, R, /*axis=*/0);
  } else { // BIDIRECTIONAL
    // W
    std::vector<Value> vals =
        foldOrEmitONNXSplitOpStablehlo(rewriter, loc, w3D2Ty, W, 0);
    fW = foldOrEmitONNXSqueezeOpStablehlo(
        rewriter, loc, w2DTy, vals[0], /*axis=*/0);
    bW = foldOrEmitONNXSqueezeOpStablehlo(
        rewriter, loc, w2DTy, vals[1], /*axis=*/0);
    // R
    vals.clear();
    vals = foldOrEmitONNXSplitOpStablehlo(rewriter, loc, r3D2Ty, R, 0);
    fR = foldOrEmitONNXSqueezeOpStablehlo(
        rewriter, loc, r2DTy, vals[0], /*axis=*/0);
    bR = foldOrEmitONNXSqueezeOpStablehlo(
        rewriter, loc, r2DTy, vals[1], /*axis=*/0);
  }

  // Transpose W and R.
  ArrayAttr permAttr = rewriter.getI64ArrayAttr({1, 0});
  if (direction == FORWARD || direction == BIDIRECTIONAL) {
    // W
    weightForward.WT = foldOrEmitONNXTransposeOpStablehlo(
        rewriter, loc, wTranspose2DTy, fW, permAttr);
    // R
    weightForward.RT = foldOrEmitONNXTransposeOpStablehlo(
        rewriter, loc, rTranspose2DTy, fR, permAttr);
  }
  if (direction == REVERSE || direction == BIDIRECTIONAL) {
    // W
    weightReverse.WT = foldOrEmitONNXTransposeOpStablehlo(
        rewriter, loc, wTranspose2DTy, bW, permAttr);
    // R
    weightReverse.RT = foldOrEmitONNXTransposeOpStablehlo(
        rewriter, loc, rTranspose2DTy, bR, permAttr);
  }
  return std::make_tuple(weightForward, weightReverse);
}

template <>
std::tuple<LstmBiasPack, LstmBiasPack> getBiasPack<ONNXLSTMOp, LstmBiasPack>(
    ConversionPatternRewriter &rewriter, Location loc, ONNXLSTMOp *op) {
  // Return values.
  LstmBiasPack biasForward, biasReverse;

  // bias: [direction, 8*hiddenSize] for both parameter and recurrence weights.
  Value B = op->getB();
  // peephold: [direction, 3*hiddenSize] for input, output and forget gates.
  Value P = op->getP();

  // direction
  StringRef direction = op->getDirection();

  // Split B.
  if (!isNoneValue(B)) {
    ArrayRef<int64_t> bShape = mlir::cast<ShapedType>(B.getType()).getShape();
    Type elementType = mlir::cast<ShapedType>(B.getType()).getElementType();
    int64_t hiddenSize = bShape[1] / 8;

    // MemRef types.
    auto bType2D = RankedTensorType::get({1, 8 * hiddenSize}, elementType);
    auto bType1D = RankedTensorType::get({8 * hiddenSize}, elementType);
    auto bSplitType1D = RankedTensorType::get({hiddenSize}, elementType);
    SmallVector<Type, 4> split1D8Ty(8, bSplitType1D);
    SmallVector<Type, 4> split2D2Ty(2, bType2D);

    // Squeeze the direction axis from B.
    Value fB, bB;
    if (direction == FORWARD) {
      fB = foldOrEmitONNXSqueezeOpStablehlo(
          rewriter, loc, bType1D, B, /*axis=*/0);
    } else if (direction == REVERSE) {
      bB = foldOrEmitONNXSqueezeOpStablehlo(
          rewriter, loc, bType1D, B, /*axis=*/0);
    } else { // BIDIRECTIONAL
      std::vector<Value> vals;
      vals = foldOrEmitONNXSplitOpStablehlo(rewriter, loc, split2D2Ty, B, 0);
      fB = foldOrEmitONNXSqueezeOpStablehlo(
          rewriter, loc, bType1D, vals[0], /*axis=*/0);
      bB = foldOrEmitONNXSqueezeOpStablehlo(
          rewriter, loc, bType1D, vals[1], /*axis=*/0);
    }

    // Split B into individual bias tensors.
    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      std::vector<Value> vals =
          foldOrEmitONNXSplitOpStablehlo(rewriter, loc, split1D8Ty, fB, 0);
      biasForward.Wbi = vals[0];
      biasForward.Wbo = vals[1];
      biasForward.Wbf = vals[2];
      biasForward.Wbc = vals[3];
      biasForward.Rbi = vals[4];
      biasForward.Rbo = vals[5];
      biasForward.Rbf = vals[6];
      biasForward.Rbc = vals[7];
      biasForward.hasBias = true;
    }
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      std::vector<Value> vals =
          foldOrEmitONNXSplitOpStablehlo(rewriter, loc, split1D8Ty, bB, 0);
      biasReverse.Wbi = vals[0];
      biasReverse.Wbo = vals[1];
      biasReverse.Wbf = vals[2];
      biasReverse.Wbc = vals[3];
      biasReverse.Rbi = vals[4];
      biasReverse.Rbo = vals[5];
      biasReverse.Rbf = vals[6];
      biasReverse.Rbc = vals[7];
      biasReverse.hasBias = true;
    }
  }

  // Split P.
  if (!isNoneValue(P)) {
    ArrayRef<int64_t> pShape = mlir::cast<ShapedType>(P.getType()).getShape();
    Type elementType = mlir::cast<ShapedType>(P.getType()).getElementType();
    int64_t hiddenSize = pShape[1] / 3;

    // MemRef types.
    auto pType2D = RankedTensorType::get({1, 3 * hiddenSize}, elementType);
    auto pType1D = RankedTensorType::get({3 * hiddenSize}, elementType);
    auto pSplitType1D = RankedTensorType::get({hiddenSize}, elementType);
    SmallVector<Type, 4> split1D3Ty(3, pSplitType1D);
    SmallVector<Type, 4> split2D2Ty(2, pType2D);

    // Squeeze the direction axis from P.
    Value fP, bP;
    if (direction == FORWARD) {
      fP = foldOrEmitONNXSqueezeOpStablehlo(
          rewriter, loc, pType1D, P, /*axis=*/0);
    } else if (direction == REVERSE) {
      bP = foldOrEmitONNXSqueezeOpStablehlo(
          rewriter, loc, pType1D, P, /*axis=*/0);
    } else { // BIDIRECTIONAL
      std::vector<Value> vals =
          foldOrEmitONNXSplitOpStablehlo(rewriter, loc, split2D2Ty, P, 0);
      fP = foldOrEmitONNXSqueezeOpStablehlo(
          rewriter, loc, pType1D, vals[0], /*axis=*/0);
      bP = foldOrEmitONNXSqueezeOpStablehlo(
          rewriter, loc, pType1D, vals[1], /*axis=*/0);
    }

    // Split P into individual tensors.
    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      std::vector<Value> vals =
          foldOrEmitONNXSplitOpStablehlo(rewriter, loc, split1D3Ty, fP, 0);
      biasForward.Pi = vals[0];
      biasForward.Po = vals[1];
      biasForward.Pf = vals[2];
      biasForward.hasPeephole = true;
    }
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      std::vector<Value> vals =
          foldOrEmitONNXSplitOpStablehlo(rewriter, loc, split1D3Ty, bP, 0);
      biasReverse.Pi = vals[0];
      biasReverse.Po = vals[1];
      biasReverse.Pf = vals[2];
      biasReverse.hasPeephole = true;
    }
  }

  return std::make_tuple(biasForward, biasReverse);
}

template <>
LstmState allocAndInitializeStates<ONNXLSTMOp, LstmState>(
    ConversionPatternRewriter &rewriter, Location loc, ONNXLSTMOp *op,
    typename ONNXLSTMOp::Adaptor operandAdaptor, bool enableUnroll) {
  LstmState state;

  // direction
  StringRef direction = op->getDirection();

  // allocation for the results of this operation.
  // If the result is not returned, then no allocation happens.
  if (!enableUnroll) {
    if (direction == FORWARD || direction == BIDIRECTIONAL)
      state.allHForward = allocAllHidden(
          rewriter, loc, operandAdaptor.getX(), operandAdaptor.getR());
    if (direction == REVERSE || direction == BIDIRECTIONAL)
      state.allHReverse = allocAllHidden(
          rewriter, loc, operandAdaptor.getX(), operandAdaptor.getR());
  }
  // Y :: [seq_length, num_directions, batch_size, hidden_size]
  // Y_h :: [num_directions, batch_size, hidden_size]
  state.ht = allocHiddenOrCell(rewriter, loc, operandAdaptor.getX(),
      operandAdaptor.getW(), operandAdaptor.getR());
  // Y_c :: [num_directions, batch_size, hidden_size]
  state.ct = allocHiddenOrCell(rewriter, loc, operandAdaptor.getX(),
      operandAdaptor.getW(), operandAdaptor.getR());

  // Insert allocation and deallocation the intermediate Ht and Ct for the
  // forward and reverse directions.
  // Ht :: [batch_size, hidden_size]
  // Ct :: [batch_size, hidden_size]
  if (direction == FORWARD || direction == BIDIRECTIONAL) {
    state.forwardHt = allocIntermediateState(
        rewriter, loc, operandAdaptor.getX(), operandAdaptor.getR());
    state.forwardCt = allocIntermediateState(
        rewriter, loc, operandAdaptor.getX(), operandAdaptor.getR());
  }
  if (direction == REVERSE || direction == BIDIRECTIONAL) {
    state.reverseHt = allocIntermediateState(
        rewriter, loc, operandAdaptor.getX(), operandAdaptor.getR());
    state.reverseCt = allocIntermediateState(
        rewriter, loc, operandAdaptor.getX(), operandAdaptor.getR());
  }

  // Initialize Ht and Ct.
  initializeIntermediateStates(rewriter, loc, state.forwardHt, state.reverseHt,
      state.forwardCt, state.reverseCt, operandAdaptor.getInitialH(),
      operandAdaptor.getInitialC(),
      mlir::cast<RankedTensorType>(operandAdaptor.getX().getType())
          .getElementType(),
      direction, /*onlyHidden=*/false);
  return state;
}

template <>
void calculateState<LstmState, LstmActivationPack, LstmWeightPack,
    LstmBiasPack>(ConversionPatternRewriter &rewriter, Location loc, Value Xt,
    LstmState &state, LstmActivationPack activationPack,
    LstmWeightPack weightPack, LstmBiasPack biasPack, Value sequenceIV,
    Value directionIV, Value sequenceLens, Value initialH, bool enableUnroll,
    bool isForward) {
  // Equations for LSTM.
  // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
  // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
  // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
  // Ct = ft (.) Ct-1 + it (.) ct
  // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
  // Ht = ot (.) h(Ct)

  MultiDialectBuilder<OnnxBuilder, StablehloBuilder> create(rewriter, loc);

  ArrayRef<int64_t> xtShape = mlir::cast<ShapedType>(Xt.getType()).getShape();
  int64_t batchSize = xtShape[0];

  // Get Ht, Ct.
  Value Ht = (isForward) ? state.forwardHt : state.reverseHt;
  Value Ct = (isForward) ? state.forwardCt : state.reverseCt;

  ArrayRef<int64_t> htShape = mlir::cast<ShapedType>(Ht.getType()).getShape();
  int64_t hiddenSize = htShape[1];

  // Frequently used types.
  RankedTensorType matrixType = mlir::cast<RankedTensorType>(Ht.getType());
  Type elementType = matrixType.getElementType();
  RankedTensorType matrixAllGatesType =
      RankedTensorType::get({batchSize, 4 * hiddenSize}, elementType);

  // Do matrix multiplications.
  // Xt * (Wi^T ++ Wo^T ++ Wf^T ++ Wc^T)
  // Ht * (Ri^T ++ Ro^T ++ Rf^T ++ Rc^T)
  // where '++' is matrix concatenation.
  // XtWT: [B, 4H], HtRT: [B, 4H]
  Value XtWT = create.onnx.matmul(matrixAllGatesType, Xt, weightPack.WT);
  Value HtRT = create.onnx.matmul(matrixAllGatesType, Ht, weightPack.RT);
  Value commonSum = create.onnx.add(XtWT, HtRT);
  Value zeroIndex = create.stablehlo.constantI64(0);
  Value oneHiddenIndex = create.stablehlo.constantI64(hiddenSize);
  Value twoHiddenIndex = create.stablehlo.constantI64(2 * hiddenSize);
  Value threeHiddenIndex = create.stablehlo.constantI64(3 * hiddenSize);
  SmallVector<int64_t> sliceSizes = {batchSize, hiddenSize};
  SmallVector<Value> iStartIndices = {zeroIndex, zeroIndex};
  SmallVector<Value> oStartIndices = {zeroIndex, oneHiddenIndex};
  SmallVector<Value> fStartIndices = {zeroIndex, twoHiddenIndex};
  SmallVector<Value> cStartIndices = {zeroIndex, threeHiddenIndex};
  Value it =
      create.stablehlo.dynamic_slice(commonSum, iStartIndices, sliceSizes);
  Value ot =
      create.stablehlo.dynamic_slice(commonSum, oStartIndices, sliceSizes);
  Value ft =
      create.stablehlo.dynamic_slice(commonSum, fStartIndices, sliceSizes);
  Value ct =
      create.stablehlo.dynamic_slice(commonSum, cStartIndices, sliceSizes);
  if (biasPack.hasBias) {
    it = create.onnx.add(it, biasPack.Wbi);
    it = create.onnx.add(it, biasPack.Rbi);
  }
  if (biasPack.hasPeephole) {
    Value PiCt = create.onnx.mul(biasPack.Pi, Ct);
    it = create.onnx.add(it, PiCt);
  }
  it = applyActivation(rewriter, loc, activationPack.f, it);
  if (biasPack.hasBias) {
    ft = create.onnx.add(ft, biasPack.Wbf);
    ft = create.onnx.add(ft, biasPack.Rbf);
  }
  if (biasPack.hasPeephole) {
    Value PfCt = create.onnx.mul(biasPack.Pf, Ct);
    ft = create.onnx.add(ft, PfCt);
  }
  ft = applyActivation(rewriter, loc, activationPack.f, ft);
  if (biasPack.hasBias) {
    ct = create.onnx.add(ct, biasPack.Wbc);
    ct = create.onnx.add(ct, biasPack.Rbc);
  }
  ct = applyActivation(rewriter, loc, activationPack.g, ct);

  Value ftCt = create.onnx.mul(ft, Ct);
  Value itct = create.onnx.mul(it, ct);
  Value nextCt = create.onnx.add(ftCt, itct);

  if (biasPack.hasBias) {
    ot = create.onnx.add(ot, biasPack.Wbo);
    ot = create.onnx.add(ot, biasPack.Rbo);
  }
  if (biasPack.hasPeephole) {
    Value PoCt = create.onnx.mul(biasPack.Po, nextCt);
    ot = create.onnx.add(ot, PoCt);
  }
  ot = applyActivation(rewriter, loc, activationPack.f, ot);
  // Ht = ot (.) h(Ct)
  Value nextHt = applyActivation(rewriter, loc, activationPack.h, nextCt);
  nextHt = create.onnx.mul(ot, nextHt);
  if (isForward) {
    state.forwardHt = nextHt;
    state.forwardCt = nextCt;
  } else {
    state.reverseHt = nextHt;
    state.reverseCt = nextCt;
  }
  if (enableUnroll) {
    RankedTensorType unsqueezedHtType =
        RankedTensorType::get({1, 1, batchSize, hiddenSize}, elementType);
    if (isForward)
      state.forwardAllH.emplace_back(create.onnx.unsqueeze(
          unsqueezedHtType, nextHt, create.onnx.constantInt64({0, 1})));
    else
      state.reverseAllH.insert(state.reverseAllH.begin(),
          create.onnx.unsqueeze(
              unsqueezedHtType, nextHt, create.onnx.constantInt64({0, 1})));
  } else {
    RankedTensorType unsqueezedHtType =
        RankedTensorType::get({1, 1, batchSize, hiddenSize}, elementType);
    RankedTensorType unsqueezedIdxType =
        RankedTensorType::get({1, 1}, rewriter.getI64Type());
    Value unsqueezedHt = create.onnx.unsqueeze(
        unsqueezedHtType, nextHt, create.onnx.constantInt64({0, 1}));
    Value unsqueezedIdx = create.onnx.unsqueeze(
        unsqueezedIdxType, sequenceIV, create.onnx.constantInt64({0}));
    if (isForward)
      state.allHForward =
          rewriter.create<ONNXScatterNDOp>(loc, state.allHForward.getType(),
              state.allHForward, unsqueezedIdx, unsqueezedHt);
    else
      state.allHReverse =
          rewriter.create<ONNXScatterNDOp>(loc, state.allHReverse.getType(),
              state.allHReverse, unsqueezedIdx, unsqueezedHt);
  }
}

template <>
void stateToOutput<ONNXLSTMOp, LstmState>(ConversionPatternRewriter &rewriter,
    Location loc, ONNXLSTMOp *op, LstmState state, std::vector<Value> &outputs,
    bool enableUnroll) {
  Value noneValue;
  auto direction = op->getDirection();
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
  // First output: all sequences.
  if (isNoneValue(op->getY()))
    outputs.emplace_back(noneValue);
  else {
    if (enableUnroll) {
      if (direction == FORWARD) {
        outputs.emplace_back(create.onnx.concat(
            op->getY().getType(), ValueRange(state.forwardAllH), 0));
      } else if (direction == REVERSE) {
        outputs.emplace_back(create.onnx.concat(
            op->getY().getType(), ValueRange(state.reverseAllH), 0));
      } else {
        auto outputShape =
            mlir::cast<ShapedType>(op->getY().getType()).getShape();
        RankedTensorType singleDirectionType = RankedTensorType::get(
            {outputShape[0], 1, outputShape[2], outputShape[3]},
            mlir::cast<ShapedType>(op->getY().getType()).getElementType());
        outputs.emplace_back(create.onnx.concat(op->getY().getType(),
            {create.onnx.concat(
                 singleDirectionType, ValueRange(state.forwardAllH), 0),
                create.onnx.concat(
                    singleDirectionType, ValueRange(state.reverseAllH), 0)},
            1));
      }
    } else {
      if (direction == FORWARD) {
        outputs.emplace_back(state.allHForward);
      } else if (direction == REVERSE) {
        outputs.emplace_back(state.allHReverse);
      } else {
        outputs.emplace_back(create.onnx.concat(
            op->getY().getType(), {state.allHForward, state.allHReverse}, 1));
      }
    }
  }
  // Second output: hidden.
  if (isNoneValue(op->getYH()))
    outputs.emplace_back(noneValue);
  else {
    stateToOutputForHiddenOrCell(
        rewriter, loc, state.forwardHt, state.reverseHt, direction, state.ht);
    outputs.emplace_back(state.ht);
  }
  // Third output: cell.
  if (isNoneValue(op->getYC()))
    outputs.emplace_back(noneValue);
  else {
    stateToOutputForHiddenOrCell(
        rewriter, loc, state.forwardCt, state.reverseCt, direction, state.ct);
    outputs.emplace_back(state.ct);
  }
}

template <>
void calculateStateWithUnroll<ONNXLSTMOp, LstmState, LstmActivationPack,
    LstmWeightPack, LstmBiasPack>(mlir::ConversionPatternRewriter &rewriter,
    Location loc, llvm::StringRef direction, int64_t sequenceDimSize, Value X,
    LstmState &state, LstmActivationPack activationForward,
    LstmActivationPack activationReverse, LstmWeightPack weightForward,
    LstmWeightPack weightReverse, LstmBiasPack biasForward,
    LstmBiasPack biasReverse, Value sequenceLens, Value initialH) {
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);

  if (direction == FORWARD || direction == BIDIRECTIONAL) {
    for (int64_t i = 0; i < sequenceDimSize; i++) {
      Value directionIV = create.onnx.constantInt64({0});
      Value sequenceIV = create.onnx.constantInt64({i});
      // Get a slice of X at the current timestep.
      Value Xt = emitXSliceAt(rewriter, loc, X, sequenceIV);
      // Emit calculation for one RNN step.
      calculateState<LstmState, LstmActivationPack, LstmWeightPack,
          LstmBiasPack>(rewriter, loc, Xt, state, activationForward,
          weightForward, biasForward, sequenceIV, directionIV, sequenceLens,
          initialH, /*enableUnroll=*/true, /*isForward=*/true);
    }
  }

  if (direction == REVERSE || direction == BIDIRECTIONAL) {
    for (int64_t i = 0; i < sequenceDimSize; i++) {
      Value directionIV =
          create.onnx.constantInt64({(direction == REVERSE) ? 0 : 1});
      Value reverseSequenceIV =
          create.onnx.constantInt64({sequenceDimSize - i - 1});
      // Get a slice of X at the current timestep.
      Value Xt = emitXSliceAt(rewriter, loc, X, reverseSequenceIV);
      // Emit calculation for one RNN step.
      calculateState<LstmState, LstmActivationPack, LstmWeightPack,
          LstmBiasPack>(rewriter, loc, Xt, state, activationReverse,
          weightReverse, biasReverse, reverseSequenceIV, directionIV,
          sequenceLens, initialH, /*enableUnroll=*/true, /*isForward=*/false);
    }
  }
}

template <>
void calculateStateWithLoop<ONNXLSTMOp, LstmState, LstmActivationPack,
    LstmWeightPack, LstmBiasPack>(mlir::ConversionPatternRewriter &rewriter,
    Location loc, llvm::StringRef direction, int64_t sequenceDimSize, Value X,
    LstmState &state, LstmActivationPack activationForward,
    LstmActivationPack activationReverse, LstmWeightPack weightForward,
    LstmWeightPack weightReverse, LstmBiasPack biasForward,
    LstmBiasPack biasReverse, Value sequenceLens, Value initialH) {
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);

  if (direction == FORWARD || direction == BIDIRECTIONAL) {
    Value directionIV = create.onnx.constantInt64({0});
    Value sequenceIV = create.onnx.constantInt64({0});
    SmallVector<Value> operands = {
        sequenceIV, state.allHForward, state.forwardHt, state.forwardCt};
    SmallVector<Type> returnedTypes = {sequenceIV.getType(),
        state.allHForward.getType(), state.forwardHt.getType(),
        state.forwardCt.getType()};
    SmallVector<Location> locations(returnedTypes.size(), loc);
    ::stablehlo::WhileOp whileLoopOp =
        rewriter.create<::stablehlo::WhileOp>(loc, returnedTypes, operands);
    Region &condRegion = whileLoopOp.getCond();
    Region &bodyRegion = whileLoopOp.getBody();
    Block &condBlock = condRegion.emplaceBlock();
    Block &bodyBlock = bodyRegion.emplaceBlock();
    condBlock.addArguments(returnedTypes, locations);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&condBlock);
      BlockArgument lhs = condBlock.getArgument(0);
      Value rhs = create.onnx.constantInt64({sequenceDimSize});
      Value compareResult = rewriter.create<::stablehlo::CompareOp>(
          loc, lhs, rhs, ::stablehlo::ComparisonDirection::LT);
      compareResult = rewriter.create<::stablehlo::ReshapeOp>(
          loc, RankedTensorType::get({}, rewriter.getI1Type()), compareResult);
      rewriter.create<::stablehlo::ReturnOp>(loc, compareResult);
    }
    bodyBlock.addArguments(returnedTypes, locations);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&bodyBlock);
      BlockArgument seqIV = bodyBlock.getArgument(0);
      BlockArgument allH = bodyBlock.getArgument(1);
      BlockArgument ht = bodyBlock.getArgument(2);
      BlockArgument ct = bodyBlock.getArgument(3);
      state.allHForward = allH;
      state.forwardHt = ht;
      state.forwardCt = ct;
      Value Xt = emitXSliceAt(rewriter, loc, X, seqIV);
      calculateState<LstmState, LstmActivationPack, LstmWeightPack,
          LstmBiasPack>(rewriter, loc, Xt, state, activationForward,
          weightForward, biasForward, seqIV, directionIV, sequenceLens,
          initialH, /*enableUnroll=*/false, /*isForward=*/true);
      Value one = create.onnx.constantInt64({1});
      Value newSeqIV = create.onnx.add(seqIV, one);
      rewriter.create<::stablehlo::ReturnOp>(loc,
          ValueRange(
              {newSeqIV, state.allHForward, state.forwardHt, state.forwardCt}));
    }
    state.allHForward = whileLoopOp.getResults()[1];
    state.forwardHt = whileLoopOp.getResults()[2];
    state.forwardCt = whileLoopOp.getResults()[3];
  }

  if (direction == REVERSE || direction == BIDIRECTIONAL) {
    Value directionIV =
        create.onnx.constantInt64({(direction == REVERSE) ? 0 : 1});
    Value reverseSequenceIV = create.onnx.constantInt64({sequenceDimSize - 1});

    SmallVector<Value> operands = {
        reverseSequenceIV, state.allHReverse, state.reverseHt, state.reverseCt};
    SmallVector<Type> returnedTypes = {reverseSequenceIV.getType(),
        state.allHReverse.getType(), state.reverseHt.getType(),
        state.reverseCt.getType()};
    SmallVector<Location> locations(returnedTypes.size(), loc);
    ::stablehlo::WhileOp whileLoopOp =
        rewriter.create<::stablehlo::WhileOp>(loc, returnedTypes, operands);
    Region &condRegion = whileLoopOp.getCond();
    Region &bodyRegion = whileLoopOp.getBody();
    Block &condBlock = condRegion.emplaceBlock();
    Block &bodyBlock = bodyRegion.emplaceBlock();
    condBlock.addArguments(returnedTypes, locations);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&condBlock);
      BlockArgument lhs = condBlock.getArgument(0);
      Value rhs = create.onnx.constantInt64({0});
      Value compareResult = rewriter.create<::stablehlo::CompareOp>(
          loc, lhs, rhs, ::stablehlo::ComparisonDirection::GE);
      compareResult = rewriter.create<::stablehlo::ReshapeOp>(
          loc, RankedTensorType::get({}, rewriter.getI1Type()), compareResult);
      rewriter.create<::stablehlo::ReturnOp>(loc, compareResult);
    }
    bodyBlock.addArguments(returnedTypes, locations);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&bodyBlock);
      BlockArgument revseqIV = bodyBlock.getArgument(0);
      BlockArgument allH = bodyBlock.getArgument(1);
      BlockArgument ht = bodyBlock.getArgument(2);
      BlockArgument ct = bodyBlock.getArgument(3);
      state.allHReverse = allH;
      state.reverseHt = ht;
      state.reverseCt = ct;
      Value Xt = emitXSliceAt(rewriter, loc, X, revseqIV);
      calculateState<LstmState, LstmActivationPack, LstmWeightPack,
          LstmBiasPack>(rewriter, loc, Xt, state, activationReverse,
          weightReverse, biasReverse, revseqIV, directionIV, sequenceLens,
          initialH, /*enableUnroll=*/false, /*isForward=*/false);
      Value one = create.onnx.constantInt64({1});
      Value newrevseqIV = create.onnx.sub(revseqIV, one);
      rewriter.create<::stablehlo::ReturnOp>(
          loc, ValueRange({newrevseqIV, state.allHReverse, state.reverseHt,
                   state.reverseCt}));
    }
    state.allHReverse = whileLoopOp.getResults()[1];
    state.reverseHt = whileLoopOp.getResults()[2];
    state.reverseCt = whileLoopOp.getResults()[3];
  }
}

} // namespace stablehlo

void populateLoweringONNXLSTMOpToStablehloPattern(
    RewritePatternSet &patterns, MLIRContext *ctx, bool enableUnroll) {
  patterns.insert<onnx_mlir::stablehlo::ONNXRNNOpLowering<ONNXLSTMOp,
      onnx_mlir::stablehlo::LstmState, onnx_mlir::LstmActivationPack,
      onnx_mlir::LstmWeightPack, onnx_mlir::LstmBiasPack>>(ctx, enableUnroll);
}

} // namespace onnx_mlir
