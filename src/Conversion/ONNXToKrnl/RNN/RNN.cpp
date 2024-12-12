/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- RNN.cpp - Lowering RNN Op --------------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX RNN Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/RNN/RNNBase.hpp"

using namespace mlir;

namespace onnx_mlir {

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
  return (isNoneValue(op->getY()) && isNoneValue(op->getYH()));
}

template <>
std::tuple<RnnActivationPack, RnnActivationPack>
getActivationPack<ONNXRNNOp, RnnActivationPack>(ONNXRNNOp *op) {
  auto direction = op->getDirection();
  auto activations = op->getActivations();
  auto activationAlpha = op->getActivationAlpha();
  auto activationBeta = op->getActivationBeta();

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
            mlir::cast<StringAttr>(activationArrAttr[0]).getValue();
      }
    }

    // Reverse activations.
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      unsigned int startIndex = (direction == REVERSE) ? 0 : 1;
      if (activationArrAttr.size() > startIndex) {
        activationReverse.f.name =
            mlir::cast<StringAttr>(activationArrAttr[startIndex]).getValue();
      }
    }
  }

  // Get alpha attributes.
  if (activationAlpha) {
    ArrayRef<Attribute> activationArrAttr = activationAlpha.value();
    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      // Forward activations.
      if (activationArrAttr.size() > 0) {
        activationForward.f.alpha = mlir::cast<FloatAttr>(activationArrAttr[0]);
      }
    }

    // Reverse activations.
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      unsigned int startIndex = (direction == REVERSE) ? 0 : 1;
      if (activationArrAttr.size() > startIndex) {
        activationReverse.f.alpha =
            mlir::cast<FloatAttr>(activationArrAttr[startIndex]);
      }
    }
  }

  // Get beta attributes.
  if (activationBeta) {
    ArrayRef<Attribute> activationArrAttr = activationBeta.value();
    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      // Forward activations.
      if (activationArrAttr.size() > 0) {
        activationForward.f.beta = mlir::cast<FloatAttr>(activationArrAttr[0]);
      }
    }

    // Reverse activations.
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      unsigned int startIndex = (direction == REVERSE) ? 0 : 1;
      if (activationArrAttr.size() > startIndex) {
        activationReverse.f.beta =
            mlir::cast<FloatAttr>(activationArrAttr[startIndex]);
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
  Value W = op->getW();
  // recurrence weight: [direction, hiddenSize, hiddenSize]
  Value R = op->getR();
  // direction
  StringRef direction = op->getDirection();

  ArrayRef<int64_t> wShape = mlir::cast<ShapedType>(W.getType()).getShape();
  Type elementType = mlir::cast<ShapedType>(W.getType()).getElementType();
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
    fW = foldOrEmitONNXSqueezeV11OpKrnl(rewriter, loc, w2DTy, W, /*axis=*/0);
    fR = foldOrEmitONNXSqueezeV11OpKrnl(rewriter, loc, r2DTy, R, /*axis=*/0);
  } else if (direction == REVERSE) {
    bW = foldOrEmitONNXSqueezeV11OpKrnl(rewriter, loc, w2DTy, W, /*axis=*/0);
    bR = foldOrEmitONNXSqueezeV11OpKrnl(rewriter, loc, r2DTy, R, /*axis=*/0);
  } else { // BIDIRECTIONAL
    // W
    std::vector<Value> vals =
        foldOrEmitONNXSplitV11OpKrnl(rewriter, loc, w3D2Ty, W, 0);
    fW = foldOrEmitONNXSqueezeV11OpKrnl(
        rewriter, loc, w2DTy, vals[0], /*axis=*/0);
    bW = foldOrEmitONNXSqueezeV11OpKrnl(
        rewriter, loc, w2DTy, vals[1], /*axis=*/0);
    // R
    vals.clear();
    vals = foldOrEmitONNXSplitV11OpKrnl(rewriter, loc, r3D2Ty, R, 0);
    fR = foldOrEmitONNXSqueezeV11OpKrnl(
        rewriter, loc, r2DTy, vals[0], /*axis=*/0);
    bR = foldOrEmitONNXSqueezeV11OpKrnl(
        rewriter, loc, r2DTy, vals[1], /*axis=*/0);
  }

  // Split W and R into individual weight tensors, and transpose them.
  if (direction == FORWARD || direction == BIDIRECTIONAL) {
    weightForward.Wi = foldOrEmitONNXTransposeOpKrnl(
        rewriter, loc, wTranspose2DTy, fW, permAttr);
    weightForward.Ri = foldOrEmitONNXTransposeOpKrnl(
        rewriter, loc, rTranspose2DTy, fR, permAttr);
  }
  if (direction == REVERSE || direction == BIDIRECTIONAL) {
    weightReverse.Wi = foldOrEmitONNXTransposeOpKrnl(
        rewriter, loc, wTranspose2DTy, bW, permAttr);
    weightReverse.Ri = foldOrEmitONNXTransposeOpKrnl(
        rewriter, loc, rTranspose2DTy, bR, permAttr);
  }
  return std::make_tuple(weightForward, weightReverse);
}

template <>
std::tuple<RnnBiasPack, RnnBiasPack> getBiasPack<ONNXRNNOp, RnnBiasPack>(
    ConversionPatternRewriter &rewriter, Location loc, ONNXRNNOp *op) {
  // Return values.
  RnnBiasPack biasForward, biasReverse;

  // bias: [direction, 2*hiddenSize] for both parameter and recurrence weights.
  Value B = op->getB();

  // direction
  StringRef direction = op->getDirection();

  // Split B.
  if (!isNoneValue(B)) {
    ArrayRef<int64_t> bShape = mlir::cast<ShapedType>(B.getType()).getShape();
    Type elementType = mlir::cast<ShapedType>(B.getType()).getElementType();
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
      fB =
          foldOrEmitONNXSqueezeV11OpKrnl(rewriter, loc, bType1D, B, /*axis=*/0);
    } else if (direction == REVERSE) {
      bB =
          foldOrEmitONNXSqueezeV11OpKrnl(rewriter, loc, bType1D, B, /*axis=*/0);
    } else { // BIDIRECTIONAL
      std::vector<Value> vals;
      vals = foldOrEmitONNXSplitV11OpKrnl(rewriter, loc, split2D2Ty, B, 0);
      fB = foldOrEmitONNXSqueezeV11OpKrnl(
          rewriter, loc, bType1D, vals[0], /*axis=*/0);
      bB = foldOrEmitONNXSqueezeV11OpKrnl(
          rewriter, loc, bType1D, vals[1], /*axis=*/0);
    }

    // Split B into individual bias tensors.
    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      std::vector<Value> vals =
          foldOrEmitONNXSplitV11OpKrnl(rewriter, loc, split1D2Ty, fB, 0);
      biasForward.Wbi = vals[0];
      biasForward.Rbi = vals[1];
      biasForward.hasBias = true;
    }
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      std::vector<Value> vals =
          foldOrEmitONNXSplitV11OpKrnl(rewriter, loc, split1D2Ty, bB, 0);
      biasReverse.Wbi = vals[0];
      biasReverse.Rbi = vals[1];
      biasReverse.hasBias = true;
    }
  }

  return std::make_tuple(biasForward, biasReverse);
}

template <>
RnnState allocAndInitializeStates<ONNXRNNOp, RnnState>(
    ConversionPatternRewriter &rewriter, Location loc,
    const TypeConverter *typeConverter, ONNXRNNOp *op,
    typename ONNXRNNOp::Adaptor operandAdaptor) {
  RnnState state;

  // direction
  StringRef direction = op->getDirection();

  // Insert allocation and deallocation for the results of this operation.
  // Y :: [seq_length, num_directions, batch_size, hidden_size]
  state.allH =
      allocAllHidden(rewriter, loc, typeConverter, operandAdaptor.getX(),
          operandAdaptor.getW(), operandAdaptor.getR(), op->getY());
  // Y_h :: [num_directions, batch_size, hidden_size]
  state.ht =
      allocHiddenOrCell(rewriter, loc, typeConverter, operandAdaptor.getX(),
          operandAdaptor.getW(), operandAdaptor.getR(), op->getYH());

  // Insert allocation and deallocation the intermediate Ht for the forward and
  // reverse directions.
  // Ht :: [batch_size, hidden_size]
  // Ct :: [batch_size, hidden_size]
  if (direction == FORWARD || direction == BIDIRECTIONAL)
    state.forwardHt = allocIntermediateState(
        rewriter, loc, operandAdaptor.getX(), operandAdaptor.getR());
  if (direction == REVERSE || direction == BIDIRECTIONAL)
    state.reverseHt = allocIntermediateState(
        rewriter, loc, operandAdaptor.getX(), operandAdaptor.getR());

  // Initialize ht.
  Value noneValue;
  initializeIntermediateStates(rewriter, loc, state.forwardHt, state.reverseHt,
      noneValue, noneValue, operandAdaptor.getInitialH(), noneValue,
      mlir::cast<MemRefType>(operandAdaptor.getX().getType()).getElementType(),
      direction, /*onlyHidden=*/true);
  return state;
}

template <>
void calculateState<RnnState, RnnActivationPack, RnnWeightPack, RnnBiasPack>(
    ConversionPatternRewriter &rewriter, Location loc, Value Xt, RnnState state,
    RnnActivationPack activationPack, RnnWeightPack weightPack,
    RnnBiasPack biasPack, Value sequenceIV, Value directionIV,
    Value sequenceLens, Value initialH, bool isForward) {
  // Equations for RNN.
  // Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
  // Shape information:
  // Xt : [seq_length, batch_size, input_size]
  // Wi : [hidden_size, input_size]
  // Ri : [hidden_size, hidden_size]
  // Ht : [batch_size, hidden_size]
  // Wbi: [hidden_size]
  // Rbi: [hidden_size]

  // ToFix: add support of sequenceLens for RNN
  assert(isNoneValue(sequenceLens) && "not implemented yet");

  MultiDialectBuilder<KrnlBuilder, MathBuilder, MemRefBuilder, OnnxBuilder>
      create(rewriter, loc);

  // Get Ht.
  Value Ht = (isForward) ? state.forwardHt : state.reverseHt;
  MemRefType matrixType = mlir::cast<MemRefType>(Ht.getType());
  unsigned htRank = matrixType.getRank();

  // Do matrix multiplications.
  Value XtWi =
      create.onnx.toMemref(create.onnx.matmul(matrixType, Xt, weightPack.Wi));
  Value HtRi =
      create.onnx.toMemref(create.onnx.matmul(matrixType, Ht, weightPack.Ri));

  // Do element-wise computations. Fuse them into a single nested loop.
  // Lower and upper bounds derived from Ht tensor.
  Value iZero = create.math.constantIndex(0);
  SmallVector<Value, 4> htLbs(htRank, iZero);
  SmallVector<Value, 4> htUbs;
  for (unsigned r = 0; r < htRank; ++r) {
    htUbs.emplace_back(create.mem.dim(Ht, r));
  }
  ValueRange loops = create.krnl.defineLoops(htRank);
  create.krnl.iterate(loops, loops, htLbs, htUbs,
      [&](const KrnlBuilder &createKrnl, ValueRange indices) {
        MathBuilder createMath(createKrnl);
        Value bs(indices[0]), hs(indices[1]);
        // Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
        Value XtWiVal = createKrnl.load(XtWi, indices);
        Value HtRiVal = createKrnl.load(HtRi, indices);
        Value nextHt = createMath.add(XtWiVal, HtRiVal);
        if (biasPack.hasBias) {
          Value WbiVal = createKrnl.load(biasPack.Wbi, {hs});
          Value RbiVal = createKrnl.load(biasPack.Rbi, {hs});
          nextHt = createMath.add(nextHt, WbiVal);
          nextHt = createMath.add(nextHt, RbiVal);
        }
        nextHt = applyActivation(
            createKrnl.getBuilder(), loc, activationPack.f, nextHt);

        // Store the intermediate Ht.
        createKrnl.store(nextHt, Ht, indices);
        if (!isNoneValue(state.allH))
          createKrnl.store(
              nextHt, state.allH, {sequenceIV, directionIV, bs, hs});
      });
}

template <>
void stateToOutput<ONNXRNNOp, RnnState>(ConversionPatternRewriter &rewriter,
    Location loc, ONNXRNNOp *op, RnnState state, std::vector<Value> &outputs) {
  Value noneValue;
  auto direction = op->getDirection();

  // First output: all sequences.
  outputs.emplace_back((isNoneValue(op->getY()) ? noneValue : state.allH));
  // Second output: hidden.
  if (isNoneValue(op->getYH()))
    outputs.emplace_back(noneValue);
  else {
    stateToOutputForHiddenOrCell(
        rewriter, loc, state.forwardHt, state.reverseHt, direction, state.ht);
    outputs.emplace_back(state.ht);
  }
}

void populateLoweringONNXRNNOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXRNNOpLowering<ONNXRNNOp, RnnState, RnnActivationPack,
      RnnWeightPack, RnnBiasPack>>(typeConverter, ctx);
}

} // namespace onnx_mlir
