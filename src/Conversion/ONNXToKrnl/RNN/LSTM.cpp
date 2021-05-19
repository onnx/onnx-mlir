/*
 * SPDX-License-Identifier: Apache-2.0
 */

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

Value noneVal;

struct LstmState {
  // returned states.
  Value allH;
  Value ht;
  Value ct;
  // intermediate states.
  Value forwardHt;
  Value reverseHt;
  Value forwardCt;
  Value reverseCt;
};

struct LstmActivationPack {
  RNNActivation f;
  RNNActivation g;
  RNNActivation h;
};

struct LstmWeightPack {
  Value Wi;
  Value Wo;
  Value Wf;
  Value Wc;
  Value Ri;
  Value Ro;
  Value Rf;
  Value Rc;
};

struct LstmBiasPack {
  Value Wbi;
  Value Wbo;
  Value Wbf;
  Value Wbc;
  Value Rbi;
  Value Rbo;
  Value Rbf;
  Value Rbc;
  // Put peephole here.
  Value Pi;
  Value Po;
  Value Pf;
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
std::tuple<LstmWeightPack, LstmWeightPack>
getWeightPack<ONNXLSTMOp, LstmWeightPack>(
    ConversionPatternRewriter &rewriter, Location loc, ONNXLSTMOp *op) {
  // Return values.
  LstmWeightPack weightForward, weightReverse;

  // parameter weight: [direction, 4*hiddenSize, inputSize]
  Value W = op->W();
  // recurrence weight: [direction, 4*hiddenSize, hiddenSize]
  Value R = op->R();
  // direction
  StringRef direction = op->direction();

  ArrayRef<int64_t> wShape = W.getType().cast<ShapedType>().getShape();
  Type elementType = W.getType().cast<ShapedType>().getElementType();
  int64_t hiddenSize = wShape[1] / 4;
  int64_t inputSize = wShape[2];

  // MemRef types for parameter weights.
  auto w3DTy = MemRefType::get({1, 4 * hiddenSize, inputSize}, elementType);
  auto w2DTy = MemRefType::get({4 * hiddenSize, inputSize}, elementType);
  auto wSplit2DTy = MemRefType::get({hiddenSize, inputSize}, elementType);
  auto wTranspose2DTy = MemRefType::get({inputSize, hiddenSize}, elementType);
  SmallVector<Type, 4> w3D2Ty(2, w3DTy);
  SmallVector<Type, 4> wSplit2D4Ty(4, wSplit2DTy);

  // MemRef types for recurrence weights.
  auto r3DTy = MemRefType::get({1, 4 * hiddenSize, hiddenSize}, elementType);
  auto r2DTy = MemRefType::get({4 * hiddenSize, hiddenSize}, elementType);
  auto rSplit2DTy = MemRefType::get({hiddenSize, hiddenSize}, elementType);
  auto rTranspose2DTy = MemRefType::get({hiddenSize, hiddenSize}, elementType);
  SmallVector<Type, 4> r3D2Ty(2, r3DTy);
  SmallVector<Type, 4> rSplit2D4Ty(4, rSplit2DTy);

  ArrayAttr permAttr = rewriter.getI64ArrayAttr({1, 0});

  // Unsqueeze the direction axis from W and R.
  Value fW, bW, fR, bR;
  if (direction == FORWARD) {
    fW = emitSqueeze(rewriter, loc, w2DTy, W, /*axis=*/0);
    fR = emitSqueeze(rewriter, loc, r2DTy, R, /*axis=*/0);
  } else if (direction == REVERSE) {
    bW = emitSqueeze(rewriter, loc, w2DTy, W, /*axis=*/0);
    bR = emitSqueeze(rewriter, loc, r2DTy, R, /*axis=*/0);
  } else { // BIDIRECTIONAL
    // W
    std::vector<Value> vals = emitSplit(rewriter, loc, w3D2Ty, W, 0);
    fW = emitSqueeze(rewriter, loc, w2DTy, vals[0], /*axis=*/0);
    bW = emitSqueeze(rewriter, loc, w2DTy, vals[1], /*axis=*/0);
    // R
    vals.clear();
    vals = emitSplit(rewriter, loc, r3D2Ty, R, 0);
    fR = emitSqueeze(rewriter, loc, r2DTy, vals[0], /*axis=*/0);
    bR = emitSqueeze(rewriter, loc, r2DTy, vals[1], /*axis=*/0);
  }

  // Split W and R into individual weight tensors, and transpose them.
  if (direction == FORWARD || direction == BIDIRECTIONAL) {
    // W
    std::vector<Value> vals = emitSplit(rewriter, loc, wSplit2D4Ty, fW, 0);
    weightForward.Wi =
        emitTranspose(rewriter, loc, wTranspose2DTy, vals[0], permAttr);
    weightForward.Wo =
        emitTranspose(rewriter, loc, wTranspose2DTy, vals[1], permAttr);
    weightForward.Wf =
        emitTranspose(rewriter, loc, wTranspose2DTy, vals[2], permAttr);
    weightForward.Wc =
        emitTranspose(rewriter, loc, wTranspose2DTy, vals[3], permAttr);
    // R
    vals.clear();
    vals = emitSplit(rewriter, loc, rSplit2D4Ty, fR, 0);
    weightForward.Ri =
        emitTranspose(rewriter, loc, rTranspose2DTy, vals[0], permAttr);
    weightForward.Ro =
        emitTranspose(rewriter, loc, rTranspose2DTy, vals[1], permAttr);
    weightForward.Rf =
        emitTranspose(rewriter, loc, rTranspose2DTy, vals[2], permAttr);
    weightForward.Rc =
        emitTranspose(rewriter, loc, rTranspose2DTy, vals[3], permAttr);
  }
  if (direction == REVERSE || direction == BIDIRECTIONAL) {
    // W
    std::vector<Value> vals = emitSplit(rewriter, loc, wSplit2D4Ty, bW, 0);
    weightReverse.Ri =
        emitTranspose(rewriter, loc, wTranspose2DTy, vals[0], permAttr);
    weightReverse.Ro =
        emitTranspose(rewriter, loc, wTranspose2DTy, vals[1], permAttr);
    weightReverse.Rf =
        emitTranspose(rewriter, loc, wTranspose2DTy, vals[2], permAttr);
    weightReverse.Rc =
        emitTranspose(rewriter, loc, wTranspose2DTy, vals[3], permAttr);
    // R
    vals.clear();
    vals = emitSplit(rewriter, loc, rSplit2D4Ty, bR, 0);
    weightReverse.Ri =
        emitTranspose(rewriter, loc, rTranspose2DTy, vals[0], permAttr);
    weightReverse.Ro =
        emitTranspose(rewriter, loc, rTranspose2DTy, vals[1], permAttr);
    weightReverse.Rf =
        emitTranspose(rewriter, loc, rTranspose2DTy, vals[2], permAttr);
    weightReverse.Rc =
        emitTranspose(rewriter, loc, rTranspose2DTy, vals[3], permAttr);
  }
  return std::make_tuple(weightForward, weightReverse);
}

template <>
std::tuple<LstmBiasPack, LstmBiasPack> getBiasPack<ONNXLSTMOp, LstmBiasPack>(
    ConversionPatternRewriter &rewriter, Location loc, ONNXLSTMOp *op) {
  // Return values.
  LstmBiasPack biasForward, biasReverse;

  // bias: [direction, 8*hiddenSize] for both parameter and recurrence weights.
  Value B = op->B();
  // peephold: [direction, 3*hiddenSize] for input, output and forget gates.
  Value P = op->P();

  // direction
  StringRef direction = op->direction();

  // Split B.
  if (!isNoneType(B)) {
    ArrayRef<int64_t> bShape = B.getType().cast<ShapedType>().getShape();
    Type elementType = B.getType().cast<ShapedType>().getElementType();
    int64_t hiddenSize = bShape[1] / 8;

    // MemRef types.
    auto bType2D = MemRefType::get({1, 8 * hiddenSize}, elementType);
    auto bType1D = MemRefType::get({8 * hiddenSize}, elementType);
    auto bSplitType1D = MemRefType::get({hiddenSize}, elementType);
    SmallVector<Type, 4> split1D8Ty(8, bSplitType1D);
    SmallVector<Type, 4> split2D2Ty(2, bType2D);

    // Unsqueeze the direction axis from B.
    Value fB, bB;
    if (direction == FORWARD) {
      fB = emitSqueeze(rewriter, loc, bType1D, B, /*axis=*/0);
    } else if (direction == REVERSE) {
      bB = emitSqueeze(rewriter, loc, bType1D, B, /*axis=*/0);
    } else { // BIDIRECTIONAL
      std::vector<Value> vals;
      vals = emitSplit(rewriter, loc, split2D2Ty, B, 0);
      fB = emitSqueeze(rewriter, loc, bType1D, vals[0], /*axis=*/0);
      bB = emitSqueeze(rewriter, loc, bType1D, vals[1], /*axis=*/0);
    }

    // Split B into invidual bias tensors.
    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      std::vector<Value> vals = emitSplit(rewriter, loc, split1D8Ty, fB, 0);
      biasForward.Wbi = vals[0];
      biasForward.Wbo = vals[1];
      biasForward.Wbf = vals[2];
      biasForward.Wbc = vals[3];
      biasForward.Rbi = vals[4];
      biasForward.Rbo = vals[5];
      biasForward.Rbf = vals[6];
      biasForward.Rbc = vals[7];
    }
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      std::vector<Value> vals = emitSplit(rewriter, loc, split1D8Ty, bB, 0);
      biasReverse.Wbi = vals[0];
      biasReverse.Wbo = vals[1];
      biasReverse.Wbf = vals[2];
      biasReverse.Wbc = vals[3];
      biasReverse.Rbi = vals[4];
      biasReverse.Rbo = vals[5];
      biasReverse.Rbf = vals[6];
      biasReverse.Rbc = vals[7];
    }
  }

  // Split P.
  if (!isNoneType(P)) {
    ArrayRef<int64_t> pShape = P.getType().cast<ShapedType>().getShape();
    Type elementType = P.getType().cast<ShapedType>().getElementType();
    int64_t hiddenSize = pShape[1] / 3;

    // MemRef types.
    auto pType2D = MemRefType::get({1, 3 * hiddenSize}, elementType);
    auto pType1D = MemRefType::get({3 * hiddenSize}, elementType);
    auto pSplitType1D = MemRefType::get({hiddenSize}, elementType);
    SmallVector<Type, 4> split1D3Ty(3, pSplitType1D);
    SmallVector<Type, 4> split2D2Ty(2, pType2D);

    // Unsqueeze the direction axis from P.
    Value fP, bP;
    if (direction == FORWARD) {
      fP = emitSqueeze(rewriter, loc, pType1D, P, /*axis=*/0);
    } else if (direction == REVERSE) {
      bP = emitSqueeze(rewriter, loc, pType1D, P, /*axis=*/0);
    } else { // BIDIRECTIONAL
      std::vector<Value> vals = emitSplit(rewriter, loc, split2D2Ty, P, 0);
      fP = emitSqueeze(rewriter, loc, pType1D, vals[0], /*axis=*/0);
      bP = emitSqueeze(rewriter, loc, pType1D, vals[1], /*axis=*/0);
    }

    // Split P into invidual tensors.
    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      std::vector<Value> vals = emitSplit(rewriter, loc, split1D3Ty, fP, 0);
      biasForward.Pi = vals[0];
      biasForward.Po = vals[1];
      biasForward.Pf = vals[2];
    }
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      std::vector<Value> vals = emitSplit(rewriter, loc, split1D3Ty, bP, 0);
      biasReverse.Pi = vals[0];
      biasReverse.Po = vals[1];
      biasReverse.Pf = vals[2];
    }
  }

  return std::make_tuple(biasForward, biasReverse);
}

template <>
LstmState allocAndInitializeStates<ONNXLSTMOp, LstmState>(
    ConversionPatternRewriter &rewriter, Location loc, ONNXLSTMOp *op,
    typename ONNXLSTMOp::Adaptor operandAdaptor) {
  LstmState state;

  // direction
  StringRef direction = op->direction();

  // Insert allocation and deallocation for the results of this operation.
  // If the result is not returned, then no allocation happens.
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

  // Insert allocation and deallocation the intermedidate Ht and Ct for the
  // forward and reverse directions.
  // Ht :: [batch_size, hidden_size]
  // Ct :: [batch_size, hidden_size]
  if (direction == FORWARD || direction == BIDIRECTIONAL) {
    state.forwardHt = allocIntermediateState(
        rewriter, loc, operandAdaptor.X(), operandAdaptor.R());
    state.forwardCt = allocIntermediateState(
        rewriter, loc, operandAdaptor.X(), operandAdaptor.R());
  }
  if (direction == REVERSE || direction == BIDIRECTIONAL) {
    state.reverseHt = allocIntermediateState(
        rewriter, loc, operandAdaptor.X(), operandAdaptor.R());
    state.reverseCt = allocIntermediateState(
        rewriter, loc, operandAdaptor.X(), operandAdaptor.R());
  }

  // Initialize Ht and Ct.
  initializeIntermediateStates(rewriter, loc, state.forwardHt, state.reverseHt,
      state.forwardCt, state.reverseCt, operandAdaptor.initial_h(),
      operandAdaptor.initial_c(),
      operandAdaptor.X().getType().cast<MemRefType>().getElementType(),
      direction, /*onlyHidden=*/false);
  return state;
}

template <>
void calculateState<ONNXLSTMOp, LstmState, LstmActivationPack, LstmWeightPack,
    LstmBiasPack>(ConversionPatternRewriter &rewriter, Location loc,
    typename ONNXLSTMOp::Adaptor operandAdaptor, LstmState state,
    LstmActivationPack activationPack, LstmWeightPack weightPack,
    LstmBiasPack biasPack, Value sequenceIV, Value directionIV,
    bool isForward) {

  // Scope for krnl EDSC ops
  using namespace mlir::edsc;
  // Scope for std EDSC ops
  using namespace edsc::intrinsics;
  ScopedContext scope(rewriter, loc);

  // Prepare dimensions.
  int64_t batchSize = dimAt(operandAdaptor.X(), 1);
  int64_t hiddenSize = dimAt(operandAdaptor.R(), 2);

  // Frequently used types.
  auto elementType =
      operandAdaptor.X().getType().cast<ShapedType>().getElementType();
  auto matrixType = MemRefType::get({batchSize, hiddenSize}, elementType);

  bool hasBiasForInput = !isNoneType(operandAdaptor.B());
  bool hasPeepholes = !isNoneType(operandAdaptor.P());

  // Equations for LSTM.
  // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
  // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
  // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
  // Ct = ft (.) Ct-1 + it (.) ct
  // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
  // Ht = ot (.) h(Ct)
  //
  // Shape information:
  // Xt : [batch_size, input_size]
  // W[iofc] : [hidden_size, input_size]
  // R[iofc] : [hidden_size, hidden_size]
  // Ht, Ct, it, ot, ft, ct: [batch_size, hidden_size]
  // Wb[iofc] : [hidden_size]
  // Rb[iofc] : [hidden_size]

  // Get a slice of X at the current timestep.
  Value Xt = emitXSliceAt(rewriter, loc, operandAdaptor.X(), sequenceIV);
  // Get Ht, Ct.
  Value Ht = (isForward) ? state.forwardHt : state.reverseHt;
  Value Ct = (isForward) ? state.forwardCt : state.reverseCt;

  // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
  Value XtWi = onnx_matmul(matrixType, Xt, weightPack.Wi);
  Value HtRi = onnx_matmul(matrixType, Ht, weightPack.Ri);
  Value it = onnx_add(matrixType, XtWi, HtRi);
  if (hasBiasForInput) {
    it = onnx_add(matrixType, it, biasPack.Wbi);
    it = onnx_add(matrixType, it, biasPack.Rbi);
  }
  if (hasPeepholes) {
    Value PiCt = onnx_mul(matrixType, biasPack.Pi, Ct);
    it = onnx_add(matrixType, it, PiCt);
  }
  it = applyActivation(rewriter, loc, activationPack.f, it);

  // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
  Value XtWf = onnx_matmul(matrixType, Xt, weightPack.Wf);
  Value HtRf = onnx_matmul(matrixType, Ht, weightPack.Rf);
  Value ft = onnx_add(matrixType, XtWf, HtRf);
  if (hasBiasForInput) {
    ft = onnx_add(matrixType, ft, biasPack.Wbf);
    ft = onnx_add(matrixType, ft, biasPack.Rbf);
  }
  if (hasPeepholes) {
    Value PfCt = onnx_mul(matrixType, biasPack.Pf, Ct);
    ft = onnx_add(matrixType, ft, PfCt);
  }
  ft = applyActivation(rewriter, loc, activationPack.f, ft);

  // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
  Value XtWc = onnx_matmul(matrixType, Xt, weightPack.Wc);
  Value HtRc = onnx_matmul(matrixType, Ht, weightPack.Rc);
  Value ct = onnx_add(matrixType, XtWc, HtRc);
  if (hasBiasForInput) {
    ct = onnx_add(matrixType, ct, biasPack.Wbc);
    ct = onnx_add(matrixType, ct, biasPack.Rbc);
  }
  ct = applyActivation(rewriter, loc, activationPack.g, ct);

  // Ct = ft (.) Ct-1 + it (.) ct
  Value ftCt = onnx_mul(matrixType, ft, Ct);
  Value itct = onnx_mul(matrixType, it, ct);
  Value nextCt = onnx_add(matrixType, ftCt, itct);

  // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
  Value XtWo = onnx_matmul(matrixType, Xt, weightPack.Wo);
  Value HtRo = onnx_matmul(matrixType, Ht, weightPack.Ro);
  Value ot = onnx_add(matrixType, XtWo, HtRo);
  if (hasBiasForInput) {
    ot = onnx_add(matrixType, ot, biasPack.Wbo);
    ot = onnx_add(matrixType, ot, biasPack.Rbo);
  }
  if (hasPeepholes) {
    Value PoCt = onnx_mul(matrixType, biasPack.Po, nextCt);
    ot = onnx_add(matrixType, ot, PoCt);
  }
  ot = applyActivation(rewriter, loc, activationPack.f, ot);

  // Ht = ot (.) h(Ct)
  Value nextHt = applyActivation(rewriter, loc, activationPack.h, nextCt);
  nextHt = onnx_mul(matrixType, ot, nextHt);

  // Store the intermediate Ht, Ct.
  storeIntermediateState(rewriter, loc, nextHt, Ht);
  storeIntermediateState(rewriter, loc, nextCt, Ct);
  if (!isNoneType(state.allH))
    storeIntermediateStateToAllH(
        rewriter, loc, nextHt, sequenceIV, directionIV, state.allH);
}

template <>
void stateToOutput<ONNXLSTMOp, LstmState>(ConversionPatternRewriter &rewriter,
    Location loc, ONNXLSTMOp *op, LstmState state,
    std::vector<Value> &outputs) {
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
  // Second output: cell.
  if (isNoneType(op->Y_c()))
    outputs.emplace_back(noneValue);
  else {
    stateToOutputForHiddenOrCell(
        rewriter, loc, state.forwardCt, state.reverseCt, direction, state.ct);
    outputs.emplace_back(state.ct);
  }
}

void populateLoweringONNXLSTMOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXRNNOpLowering<ONNXLSTMOp, LstmState, LstmActivationPack,
      LstmWeightPack, LstmBiasPack>>(ctx);
}
