/*
 * SPDX-License-Identifier: Apache-2.0
 */

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
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;

struct RnnState {
  // returned states.
  Value allH;
  Value ht;
  // intermediate states.
  Value forwardHt;
  Value reverseHt;
};

struct RnnActivationPack {
  RNNActivation f;
};

struct RnnWeightPack {
  Value Wi;
  Value Ri;
};

struct RnnBiasPack {
  bool hasBias = false;
  Value Wbi;
  Value Rbi;
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
std::tuple<RnnWeightPack, RnnWeightPack>
getWeightPack<ONNXRNNOp, RnnWeightPack>(
    ConversionPatternRewriter &rewriter, Location loc, ONNXRNNOp *op) {

  // Return values.
  RnnWeightPack weightForward, weightReverse;

  // parameter weight: [direction, hiddenSize, inputSize]
  Value W = op->W();
  // recurrence weight: [direction, hiddenSize, hiddenSize]
  Value R = op->R();
  // direction
  StringRef direction = op->direction();

  ArrayRef<int64_t> wShape = W.getType().cast<ShapedType>().getShape();
  Type elementType = W.getType().cast<ShapedType>().getElementType();
  int64_t hiddenSize = wShape[1];
  int64_t inputSize = wShape[2];

  // MemRef types for parameter weights.
  auto w3DTy = MemRefType::get({1, hiddenSize, inputSize}, elementType);
  auto w2DTy = MemRefType::get({hiddenSize, inputSize}, elementType);
  auto wTranspose2DTy = MemRefType::get({inputSize, hiddenSize}, elementType);
  SmallVector<Type, 4> w3D2Ty(2, w3DTy);

  // MemRef types for recurrence weights.
  auto r3DTy = MemRefType::get({1, hiddenSize, hiddenSize}, elementType);
  auto r2DTy = MemRefType::get({hiddenSize, hiddenSize}, elementType);
  auto rTranspose2DTy = MemRefType::get({hiddenSize, hiddenSize}, elementType);
  SmallVector<Type, 4> r3D2Ty(2, r3DTy);

  ArrayAttr permAttr = rewriter.getI64ArrayAttr({1, 0});

  // Unsqueeze the direction axis from W and R.
  Value fW, bW, fR, bR;
  if (direction == FORWARD) {
    fW = foldOrEmitONNXSqueezeOp(rewriter, loc, w2DTy, W, /*axis=*/0);
    fR = foldOrEmitONNXSqueezeOp(rewriter, loc, r2DTy, R, /*axis=*/0);
  } else if (direction == REVERSE) {
    bW = foldOrEmitONNXSqueezeOp(rewriter, loc, w2DTy, W, /*axis=*/0);
    bR = foldOrEmitONNXSqueezeOp(rewriter, loc, r2DTy, R, /*axis=*/0);
  } else { // BIDIRECTIONAL
    // W
    std::vector<Value> vals =
        foldOrEmitONNXSplitOp(rewriter, loc, w3D2Ty, W, 0);
    fW = foldOrEmitONNXSqueezeOp(rewriter, loc, w2DTy, vals[0], /*axis=*/0);
    bW = foldOrEmitONNXSqueezeOp(rewriter, loc, w2DTy, vals[1], /*axis=*/0);
    // R
    vals.clear();
    vals = foldOrEmitONNXSplitOp(rewriter, loc, r3D2Ty, R, 0);
    fR = foldOrEmitONNXSqueezeOp(rewriter, loc, r2DTy, vals[0], /*axis=*/0);
    bR = foldOrEmitONNXSqueezeOp(rewriter, loc, r2DTy, vals[1], /*axis=*/0);
  }

  // Split W and R into individual weight tensors, and transpose them.
  if (direction == FORWARD || direction == BIDIRECTIONAL) {
    weightForward.Wi =
        foldOrEmitONNXTransposeOp(rewriter, loc, wTranspose2DTy, fW, permAttr);
    weightForward.Ri =
        foldOrEmitONNXTransposeOp(rewriter, loc, rTranspose2DTy, fR, permAttr);
  }
  if (direction == REVERSE || direction == BIDIRECTIONAL) {
    weightReverse.Wi =
        foldOrEmitONNXTransposeOp(rewriter, loc, wTranspose2DTy, bW, permAttr);
    weightReverse.Ri =
        foldOrEmitONNXTransposeOp(rewriter, loc, rTranspose2DTy, bR, permAttr);
  }
  return std::make_tuple(weightForward, weightReverse);
}

template <>
std::tuple<RnnBiasPack, RnnBiasPack> getBiasPack<ONNXRNNOp, RnnBiasPack>(
    ConversionPatternRewriter &rewriter, Location loc, ONNXRNNOp *op) {
  // Return values.
  RnnBiasPack biasForward, biasReverse;

  // bias: [direction, 2*hiddenSize] for both parameter and recurrence weights.
  Value B = op->B();

  // direction
  StringRef direction = op->direction();

  // Split B.
  if (!isNoneType(B)) {
    ArrayRef<int64_t> bShape = B.getType().cast<ShapedType>().getShape();
    Type elementType = B.getType().cast<ShapedType>().getElementType();
    int64_t hiddenSize = bShape[1] / 2;

    // MemRef types.
    auto bType2D = MemRefType::get({1, 2 * hiddenSize}, elementType);
    auto bType1D = MemRefType::get({2 * hiddenSize}, elementType);
    auto bSplitType1D = MemRefType::get({hiddenSize}, elementType);
    SmallVector<Type, 4> split1D2Ty(2, bSplitType1D);
    SmallVector<Type, 4> split2D2Ty(2, bType2D);

    // Unsqueeze the direction axis from B.
    Value fB, bB;
    if (direction == FORWARD) {
      fB = foldOrEmitONNXSqueezeOp(rewriter, loc, bType1D, B, /*axis=*/0);
    } else if (direction == REVERSE) {
      bB = foldOrEmitONNXSqueezeOp(rewriter, loc, bType1D, B, /*axis=*/0);
    } else { // BIDIRECTIONAL
      std::vector<Value> vals;
      vals = foldOrEmitONNXSplitOp(rewriter, loc, split2D2Ty, B, 0);
      fB = foldOrEmitONNXSqueezeOp(rewriter, loc, bType1D, vals[0], /*axis=*/0);
      bB = foldOrEmitONNXSqueezeOp(rewriter, loc, bType1D, vals[1], /*axis=*/0);
    }

    // Split B into individual bias tensors.
    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      std::vector<Value> vals =
          foldOrEmitONNXSplitOp(rewriter, loc, split1D2Ty, fB, 0);
      biasForward.Wbi = vals[0];
      biasForward.Rbi = vals[1];
      biasForward.hasBias = true;
    }
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      std::vector<Value> vals =
          foldOrEmitONNXSplitOp(rewriter, loc, split1D2Ty, bB, 0);
      biasReverse.Wbi = vals[0];
      biasReverse.Rbi = vals[1];
      biasReverse.hasBias = true;
    }
  }

  return std::make_tuple(biasForward, biasReverse);
}

template <>
RnnState allocAndInitializeStates<ONNXRNNOp, RnnState>(
    ConversionPatternRewriter &rewriter, Location loc, ONNXRNNOp *op,
    typename ONNXRNNOp::Adaptor operandAdaptor) {
  RnnState state;

  // direction
  StringRef direction = op->direction();

  // Insert allocation and deallocation for the results of this operation.
  // Y :: [seq_length, num_directions, batch_size, hidden_size]
  state.allH = allocAllHidden(rewriter, loc, operandAdaptor.X(),
      operandAdaptor.W(), operandAdaptor.R(), op->Y(),
      checkInsertDealloc(op->getOperation(), 0));
  // Y_h :: [num_directions, batch_size, hidden_size]
  state.ht = allocHiddenOrCell(rewriter, loc, operandAdaptor.X(),
      operandAdaptor.W(), operandAdaptor.R(), op->Y_h(),
      checkInsertDealloc(op->getOperation(), 1));

  // Insert allocation and deallocation the intermedidate Ht for the forward and
  // reverse directions.
  // Ht :: [batch_size, hidden_size]
  // Ct :: [batch_size, hidden_size]
  if (direction == FORWARD || direction == BIDIRECTIONAL)
    state.forwardHt = allocIntermediateState(
        rewriter, loc, operandAdaptor.X(), operandAdaptor.R());
  if (direction == REVERSE || direction == BIDIRECTIONAL)
    state.reverseHt = allocIntermediateState(
        rewriter, loc, operandAdaptor.X(), operandAdaptor.R());

  // Initialize ht.
  Value noneValue;
  initializeIntermediateStates(rewriter, loc, state.forwardHt, state.reverseHt,
      noneValue, noneValue, operandAdaptor.initial_h(), noneValue,
      operandAdaptor.X().getType().cast<MemRefType>().getElementType(),
      direction, /*onlyHidden=*/true);
  return state;
}

template <>
void calculateState<RnnState, RnnActivationPack, RnnWeightPack, RnnBiasPack>(
    ConversionPatternRewriter &rewriter, Location loc, Value Xt, RnnState state,
    RnnActivationPack activationPack, RnnWeightPack weightPack,
    RnnBiasPack biasPack, Value sequenceIV, Value directionIV, bool isForward) {
  // Equations for RNN.
  // Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
  // Shape information:
  // Xt : [seq_length, batch_size, input_size]
  // Wi : [hidden_size, input_size]
  // Ri : [hidden_size, hidden_size]
  // Ht : [batch_size, hidden_size]
  // Wbi: [hidden_size]
  // Rbi: [hidden_size]

  ScopedContext scope(rewriter, loc);

  // Get Ht.
  Value Ht = (isForward) ? state.forwardHt : state.reverseHt;
  MemRefType matrixType = Ht.getType().cast<MemRefType>();

  // Do matrix multiplications.
  Value XtWi = onnx_matmul(matrixType, Xt, weightPack.Wi);
  Value HtRi = onnx_matmul(matrixType, Ht, weightPack.Ri);

  // Do element-wise computations. Fuse them into a single nested loop.
  MemRefBoundsCapture bounds(Ht);
  ValueRange loops = krnl_define_loop(bounds.rank());
  krnl_iterate(
      loops, bounds.getLbs(), bounds.getUbs(), {}, [&](ValueRange args) {
        ValueRange indices = krnl_get_induction_var_value(loops);
        Value bs(indices[0]), hs(indices[1]);
        // Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
        Value XtWiVal = krnl_load(XtWi, indices);
        Value HtRiVal = krnl_load(HtRi, indices);
        Value nextHt = std_addf(XtWiVal, HtRiVal);
        if (biasPack.hasBias) {
          Value WbiVal = krnl_load(biasPack.Wbi, {hs});
          Value RbiVal = krnl_load(biasPack.Rbi, {hs});
          nextHt = std_addf(nextHt, WbiVal);
          nextHt = std_addf(nextHt, RbiVal);
        }
        nextHt = applyActivation(rewriter, loc, activationPack.f, nextHt);

        // Store the intermediate Ht.
        krnl_store(nextHt, Ht, indices);
        if (!isNoneType(state.allH))
          krnl_store(nextHt, state.allH, {sequenceIV, directionIV, bs, hs});
      });
}

template <>
void stateToOutput<ONNXRNNOp, RnnState>(ConversionPatternRewriter &rewriter,
    Location loc, ONNXRNNOp *op, RnnState state, std::vector<Value> &outputs) {
  Value noneValue;
  auto direction = op->direction();

  // First output: all sequences.
  outputs.emplace_back((isNoneType(op->Y()) ? noneValue : state.allH));
  // Second output: hidden.
  if (isNoneType(op->Y_h()))
    outputs.emplace_back(noneValue);
  else {
    stateToOutputForHiddenOrCell(
        rewriter, loc, state.forwardHt, state.reverseHt, direction, state.ht);
    outputs.emplace_back(state.ht);
  }
}

void populateLoweringONNXRNNOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXRNNOpLowering<ONNXRNNOp, RnnState, RnnActivationPack,
      RnnWeightPack, RnnBiasPack>>(ctx);
}
