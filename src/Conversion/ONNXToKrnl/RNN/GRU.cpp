/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- GRU.cpp - Lowering GRU Op --------------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX GRU Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/RNN/RNNBase.hpp"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;

struct GruState {
  // returned states.
  Value allH;
  Value ht;
  // intermediate states.
  Value forwardHt;
  Value reverseHt;
  bool linearBeforeReset;
};

struct GruActivationPack {
  RNNActivation f;
  RNNActivation g;
};

struct GruWeightPack {
  Value Wz;
  Value Wr;
  Value Wh;
  Value Rz;
  Value Rr;
  Value Rh;
};

struct GruBiasPack {
  bool hasBias = false;
  Value Wbz;
  Value Wbr;
  Value Wbh;
  Value Rbz;
  Value Rbr;
  Value Rbh;
};

template <>
bool hasAllNoneOutput<ONNXGRUOp>(ONNXGRUOp *op) {
  return (isNoneType(op->Y()) && isNoneType(op->Y_h()));
}

template <>
std::tuple<GruActivationPack, GruActivationPack>
getActivationPack<ONNXGRUOp, GruActivationPack>(ONNXGRUOp *op) {
  auto direction = op->direction();
  auto activations = op->activations();
  auto activationAlpha = op->activation_alpha();
  auto activationBeta = op->activation_beta();

  GruActivationPack activationForward, activationReverse;

  // Get activation function name.
  // Default forward functions
  activationForward.f.name = "sigmoid";
  activationForward.g.name = "tanh";
  // Default backward functions
  activationReverse.f.name = "sigmoid";
  activationReverse.g.name = "tanh";
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
    }

    // Reverse activations.
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      int startIndex = (direction == REVERSE) ? 0 : 2;
      if (activationArrAttr.size() > startIndex) {
        activationReverse.f.name =
            activationArrAttr[startIndex].cast<StringAttr>().getValue();
      }
      if (activationArrAttr.size() > startIndex + 1) {
        activationReverse.g.name =
            activationArrAttr[startIndex + 1].cast<StringAttr>().getValue();
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
    }

    // Reverse activations.
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      int startIndex = (direction == REVERSE) ? 0 : 2;
      if (activationArrAttr.size() > startIndex) {
        activationReverse.f.alpha =
            activationArrAttr[startIndex].cast<FloatAttr>();
      }
      if (activationArrAttr.size() > startIndex + 1) {
        activationReverse.g.alpha =
            activationArrAttr[startIndex + 1].cast<FloatAttr>();
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
    }

    // Reverse activations.
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      int startIndex = (direction == REVERSE) ? 0 : 2;
      if (activationArrAttr.size() > startIndex) {
        activationReverse.f.beta =
            activationArrAttr[startIndex].cast<FloatAttr>();
      }
      if (activationArrAttr.size() > startIndex + 1) {
        activationReverse.g.beta =
            activationArrAttr[startIndex + 1].cast<FloatAttr>();
      }
    }
  }

  return std::make_tuple(activationForward, activationReverse);
}

template <>
std::tuple<GruWeightPack, GruWeightPack>
getWeightPack<ONNXGRUOp, GruWeightPack>(
    ConversionPatternRewriter &rewriter, Location loc, ONNXGRUOp *op) {
  // Return values.
  GruWeightPack weightForward, weightReverse;

  // parameter weight: [direction, 3*hiddenSize, inputSize]
  Value W = op->W();
  // recurrence weight: [direction, 3*hiddenSize, hiddenSize]
  Value R = op->R();
  // direction
  StringRef direction = op->direction();

  ArrayRef<int64_t> wShape = W.getType().cast<ShapedType>().getShape();
  Type elementType = W.getType().cast<ShapedType>().getElementType();
  int64_t hiddenSize = wShape[1] / 3;
  int64_t inputSize = wShape[2];

  // MemRef types for parameter weights.
  auto w3DTy = MemRefType::get({1, 3 * hiddenSize, inputSize}, elementType);
  auto w2DTy = MemRefType::get({3 * hiddenSize, inputSize}, elementType);
  auto wSplit2DTy = MemRefType::get({hiddenSize, inputSize}, elementType);
  auto wTranspose2DTy = MemRefType::get({inputSize, hiddenSize}, elementType);
  SmallVector<Type, 2> w3D2Ty(2, w3DTy);
  SmallVector<Type, 3> wSplit2D3Ty(3, wSplit2DTy);

  // MemRef types for recurrence weights.
  auto r3DTy = MemRefType::get({1, 3 * hiddenSize, hiddenSize}, elementType);
  auto r2DTy = MemRefType::get({3 * hiddenSize, hiddenSize}, elementType);
  auto rSplit2DTy = MemRefType::get({hiddenSize, hiddenSize}, elementType);
  auto rTranspose2DTy = MemRefType::get({hiddenSize, hiddenSize}, elementType);
  SmallVector<Type, 2> r3D2Ty(2, r3DTy);
  SmallVector<Type, 3> rSplit2D3Ty(3, rSplit2DTy);

  ArrayAttr permAttr = rewriter.getI64ArrayAttr({1, 0});

  // Squeeze the direction axis from W and R.
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
    std::vector<Value> vals = emitSplit(rewriter, loc, wSplit2D3Ty, fW, 0);
    weightForward.Wz =
        emitTranspose(rewriter, loc, wTranspose2DTy, vals[0], permAttr);
    weightForward.Wr =
        emitTranspose(rewriter, loc, wTranspose2DTy, vals[1], permAttr);
    weightForward.Wh =
        emitTranspose(rewriter, loc, wTranspose2DTy, vals[2], permAttr);
    // R
    vals.clear();
    vals = emitSplit(rewriter, loc, rSplit2D3Ty, fR, 0);
    weightForward.Rz =
        emitTranspose(rewriter, loc, rTranspose2DTy, vals[0], permAttr);
    weightForward.Rr =
        emitTranspose(rewriter, loc, rTranspose2DTy, vals[1], permAttr);
    weightForward.Rh =
        emitTranspose(rewriter, loc, rTranspose2DTy, vals[2], permAttr);
  }
  if (direction == REVERSE || direction == BIDIRECTIONAL) {
    // W
    std::vector<Value> vals = emitSplit(rewriter, loc, wSplit2D3Ty, bW, 0);
    weightReverse.Wz =
        emitTranspose(rewriter, loc, wTranspose2DTy, vals[0], permAttr);
    weightReverse.Wr =
        emitTranspose(rewriter, loc, wTranspose2DTy, vals[1], permAttr);
    weightReverse.Wh =
        emitTranspose(rewriter, loc, wTranspose2DTy, vals[2], permAttr);
    // R
    vals.clear();
    vals = emitSplit(rewriter, loc, rSplit2D3Ty, bR, 0);
    weightReverse.Rz =
        emitTranspose(rewriter, loc, rTranspose2DTy, vals[0], permAttr);
    weightReverse.Rr =
        emitTranspose(rewriter, loc, rTranspose2DTy, vals[1], permAttr);
    weightReverse.Rh =
        emitTranspose(rewriter, loc, rTranspose2DTy, vals[2], permAttr);
  }
  return std::make_tuple(weightForward, weightReverse);
}

template <>
std::tuple<GruBiasPack, GruBiasPack> getBiasPack<ONNXGRUOp, GruBiasPack>(
    ConversionPatternRewriter &rewriter, Location loc, ONNXGRUOp *op) {
  // Return values.
  GruBiasPack biasForward, biasReverse;

  // bias: [direction, 6*hiddenSize] for both parameter and recurrence weights.
  Value B = op->B();

  // direction
  StringRef direction = op->direction();

  // Split B.
  if (!isNoneType(B)) {
    ArrayRef<int64_t> bShape = B.getType().cast<ShapedType>().getShape();
    Type elementType = B.getType().cast<ShapedType>().getElementType();
    int64_t hiddenSize = bShape[1] / 6;

    // MemRef types.
    auto bType2D = MemRefType::get({1, 6 * hiddenSize}, elementType);
    auto bType1D = MemRefType::get({6 * hiddenSize}, elementType);
    auto bSplitType1D = MemRefType::get({hiddenSize}, elementType);
    SmallVector<Type, 6> split1D6Ty(6, bSplitType1D);
    SmallVector<Type, 2> split2D2Ty(2, bType2D);

    // Squeeze the direction axis from B.
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

    // Split B into individual bias tensors.
    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      std::vector<Value> vals = emitSplit(rewriter, loc, split1D6Ty, fB, 0);
      biasForward.Wbz = vals[0];
      biasForward.Wbr = vals[1];
      biasForward.Wbh = vals[2];
      biasForward.Rbz = vals[3];
      biasForward.Rbr = vals[4];
      biasForward.Rbh = vals[5];
      biasForward.hasBias = true;
    }
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      std::vector<Value> vals = emitSplit(rewriter, loc, split1D6Ty, bB, 0);
      biasReverse.Wbz = vals[0];
      biasReverse.Wbr = vals[1];
      biasReverse.Wbh = vals[2];
      biasReverse.Rbz = vals[3];
      biasReverse.Rbr = vals[4];
      biasReverse.Rbh = vals[5];
      biasReverse.hasBias = true;
    }
  }

  return std::make_tuple(biasForward, biasReverse);
}

template <>
GruState allocAndInitializeStates<ONNXGRUOp, GruState>(
    ConversionPatternRewriter &rewriter, Location loc, ONNXGRUOp *op,
    typename ONNXGRUOp::Adaptor operandAdaptor) {
  GruState state;

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
  if (direction == FORWARD || direction == BIDIRECTIONAL) {
    state.forwardHt = allocIntermediateState(
        rewriter, loc, operandAdaptor.X(), operandAdaptor.R());
  }
  if (direction == REVERSE || direction == BIDIRECTIONAL) {
    state.reverseHt = allocIntermediateState(
        rewriter, loc, operandAdaptor.X(), operandAdaptor.R());
  }

  // Initialize Ht.
  Value noneValue;
  initializeIntermediateStates(rewriter, loc, state.forwardHt, state.reverseHt,
      noneValue, noneValue, operandAdaptor.initial_h(), noneValue,
      operandAdaptor.X().getType().cast<MemRefType>().getElementType(),
      direction, /*onlyHidden=*/true);

  // Obtain the value of 'linear_before_reset' attribute.
  int64_t linearBeforeResetAttr = op->linear_before_reset();
  if (linearBeforeResetAttr == 0)
    state.linearBeforeReset = false;
  else
    state.linearBeforeReset = true;
  return state;
}

template <>
void calculateState<GruState, GruActivationPack, GruWeightPack, GruBiasPack>(
    ConversionPatternRewriter &rewriter, Location loc, Value Xt, GruState state,
    GruActivationPack activationPack, GruWeightPack weightPack,
    GruBiasPack biasPack, Value sequenceIV, Value directionIV, bool isForward) {
  // Equations (Default: f=Sigmoid, g=Tanh):"
  // zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)"
  // rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)"
  // if (linearBeforeReset)
  //   ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh)
  // else
  //   ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh)
  // Ht = (1 - zt) (.) ht + zt (.) Ht-1"

  ScopedContext scope(rewriter, loc);

  // Get Ht.
  Value Ht = (isForward) ? state.forwardHt : state.reverseHt;

  // Frequently used types.
  MemRefType matrixType = Ht.getType().cast<MemRefType>();
  Type elementType = matrixType.getElementType();

  // Constant one.
  DenseElementsAttr oneAttr = DenseElementsAttr::get<float>(
      RankedTensorType::get({1}, elementType), {1.0});
  Value one = rewriter.create<ONNXConstantOp>(loc,
      MemRefType::get({1}, elementType), Attribute(), oneAttr, FloatAttr(),
      ArrayAttr(), IntegerAttr(), ArrayAttr(), StringAttr(), ArrayAttr());

  // zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
  Value XtWz = onnx_matmul(matrixType, Xt, weightPack.Wz);
  Value HtRz = onnx_matmul(matrixType, Ht, weightPack.Rz);
  Value zt = onnx_add(matrixType, XtWz, HtRz);
  if (biasPack.hasBias) {
    zt = onnx_add(matrixType, zt, biasPack.Wbz);
    zt = onnx_add(matrixType, zt, biasPack.Rbz);
  }
  zt = applyActivation(rewriter, loc, activationPack.f, zt);

  // rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
  Value XtWr = onnx_matmul(matrixType, Xt, weightPack.Wr);
  Value HtRr = onnx_matmul(matrixType, Ht, weightPack.Rr);
  Value rt = onnx_add(matrixType, XtWr, HtRr);
  if (biasPack.hasBias) {
    rt = onnx_add(matrixType, rt, biasPack.Wbr);
    rt = onnx_add(matrixType, rt, biasPack.Rbr);
  }
  rt = applyActivation(rewriter, loc, activationPack.f, rt);

  // ht.
  Value ht;
  if (state.linearBeforeReset) {
    // ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh)
    Value XtWh = onnx_matmul(matrixType, Xt, weightPack.Wh);
    Value HtRh = onnx_matmul(matrixType, Ht, weightPack.Rh);
    if (biasPack.hasBias)
      HtRh = onnx_add(matrixType, HtRh, biasPack.Rbh);
    Value rtHtRh = onnx_mul(matrixType, rt, HtRh);
    ht = onnx_add(matrixType, XtWh, rtHtRh);
    if (biasPack.hasBias)
      ht = onnx_add(matrixType, ht, biasPack.Wbh);
  } else {
    // ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh)
    Value XtWh = onnx_matmul(matrixType, Xt, weightPack.Wh);
    Value rtHtRh =
        onnx_matmul(matrixType, onnx_mul(matrixType, rt, Ht), weightPack.Rh);
    ht = onnx_add(matrixType, XtWh, rtHtRh);
    if (biasPack.hasBias) {
      ht = onnx_add(matrixType, ht, biasPack.Rbh);
      ht = onnx_add(matrixType, ht, biasPack.Wbh);
    }
  }
  ht = applyActivation(rewriter, loc, activationPack.g, ht);

  // Ht = (1 - zt) (.) ht + zt (.) Ht-1
  Value oneMinusZt = onnx_sub(matrixType, one, zt);
  Value ztht = onnx_mul(matrixType, oneMinusZt, ht);
  Value ztHt = onnx_mul(matrixType, zt, Ht);
  Value nextHt = onnx_add(matrixType, ztht, ztHt);

  // Store the intermediate Ht, Ct.
  storeIntermediateState(rewriter, loc, nextHt, Ht);
  if (!isNoneType(state.allH))
    storeIntermediateStateToAllH(
        rewriter, loc, nextHt, sequenceIV, directionIV, state.allH);
}

template <>
void stateToOutput<ONNXGRUOp, GruState>(ConversionPatternRewriter &rewriter,
    Location loc, ONNXGRUOp *op, GruState state, std::vector<Value> &outputs) {
  auto direction = op->direction();
  Value noneValue;
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

void populateLoweringONNXGRUOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXRNNOpLowering<ONNXGRUOp, GruState, GruActivationPack,
      GruWeightPack, GruBiasPack>>(ctx);
}
