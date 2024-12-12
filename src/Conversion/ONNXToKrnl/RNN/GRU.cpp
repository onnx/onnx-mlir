/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------- GRU.cpp - Lowering GRU Op --------------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX GRU Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/RNN/RNNBase.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"

using namespace mlir;

namespace onnx_mlir {

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
  Value WT;
  Value RT; // (Rz ++ Rr ++ Rh) if linearBeforeReset
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
  return (isNoneValue(op->getY()) && isNoneValue(op->getYH()));
}

template <>
std::tuple<GruActivationPack, GruActivationPack>
getActivationPack<ONNXGRUOp, GruActivationPack>(ONNXGRUOp *op) {
  auto direction = op->getDirection();
  auto activations = op->getActivations();
  auto activationAlpha = op->getActivationAlpha();
  auto activationBeta = op->getActivationBeta();

  GruActivationPack activationForward, activationReverse;

  // Get activation function name.
  // Default forward functions
  activationForward.f.name = "sigmoid";
  activationForward.g.name = "tanh";
  // Default backward functions
  activationReverse.f.name = "sigmoid";
  activationReverse.g.name = "tanh";
  if (activations) {
    ArrayAttr activationArrAttr = activations.value();
    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      // Forward activations.
      if (activationArrAttr.size() > 0) {
        activationForward.f.name =
            mlir::cast<StringAttr>(activationArrAttr[0]).getValue();
      }
      if (activationArrAttr.size() > 1) {
        activationForward.g.name =
            mlir::cast<StringAttr>(activationArrAttr[1]).getValue();
      }
    }

    // Reverse activations.
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      unsigned int startIndex = (direction == REVERSE) ? 0 : 2;
      if (activationArrAttr.size() > startIndex) {
        activationReverse.f.name =
            mlir::cast<StringAttr>(activationArrAttr[startIndex]).getValue();
      }
      if (activationArrAttr.size() > startIndex + 1) {
        activationReverse.g.name =
            mlir::cast<StringAttr>(activationArrAttr[startIndex + 1])
                .getValue();
      }
    }
  }

  // Get alpha attributes.
  if (activationAlpha) {
    ArrayAttr activationArrAttr = activationAlpha.value();
    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      // Forward activations.
      if (activationArrAttr.size() > 0) {
        activationForward.f.alpha = mlir::cast<FloatAttr>(activationArrAttr[0]);
      }
      if (activationArrAttr.size() > 1) {
        activationForward.g.alpha = mlir::cast<FloatAttr>(activationArrAttr[1]);
      }
    }

    // Reverse activations.
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      unsigned int startIndex = (direction == REVERSE) ? 0 : 2;
      if (activationArrAttr.size() > startIndex) {
        activationReverse.f.alpha =
            mlir::cast<FloatAttr>(activationArrAttr[startIndex]);
      }
      if (activationArrAttr.size() > startIndex + 1) {
        activationReverse.g.alpha =
            mlir::cast<FloatAttr>(activationArrAttr[startIndex + 1]);
      }
    }
  }

  // Get beta attributes.
  if (activationBeta) {
    ArrayAttr activationArrAttr = activationBeta.value();
    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      // Forward activations.
      if (activationArrAttr.size() > 0) {
        activationForward.f.beta = mlir::cast<FloatAttr>(activationArrAttr[0]);
      }
      if (activationArrAttr.size() > 1) {
        activationForward.g.beta = mlir::cast<FloatAttr>(activationArrAttr[1]);
      }
    }

    // Reverse activations.
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      unsigned int startIndex = (direction == REVERSE) ? 0 : 2;
      if (activationArrAttr.size() > startIndex) {
        activationReverse.f.beta =
            mlir::cast<FloatAttr>(activationArrAttr[startIndex]);
      }
      if (activationArrAttr.size() > startIndex + 1) {
        activationReverse.g.beta =
            mlir::cast<FloatAttr>(activationArrAttr[startIndex + 1]);
      }
    }
  }

  return std::make_tuple(activationForward, activationReverse);
}

template <>
std::tuple<GruWeightPack, GruWeightPack>
getWeightPack<ONNXGRUOp, GruWeightPack>(
    ConversionPatternRewriter &rewriter, Location loc, ONNXGRUOp *op) {
  MultiDialectBuilder<KrnlBuilder, MathBuilder, MemRefBuilder, OnnxBuilder>
      create(rewriter, loc);
  // Return values.
  GruWeightPack weightForward, weightReverse;

  // parameter weight: [direction, 3*hiddenSize, inputSize]
  Value W = op->getW();
  // recurrence weight: [direction, 3*hiddenSize, hiddenSize]
  Value R = op->getR();
  // direction
  StringRef direction = op->getDirection();
  // linear_before_reset.
  bool linearBeforeReset = true;
  if (op->getLinearBeforeReset() == 0)
    linearBeforeReset = false;

  ArrayRef<int64_t> wShape = mlir::cast<ShapedType>(W.getType()).getShape();
  Type elementType = mlir::cast<ShapedType>(W.getType()).getElementType();
  int64_t hiddenSize = wShape[1] / 3;
  int64_t inputSize = wShape[2];

  // MemRef types for parameter weights.
  auto w3DTy = MemRefType::get({1, 3 * hiddenSize, inputSize}, elementType);
  auto w2DTy = MemRefType::get({3 * hiddenSize, inputSize}, elementType);
  auto wTransposeTy = MemRefType::get({inputSize, 3 * hiddenSize}, elementType);
  auto wSplit2DTy = MemRefType::get({hiddenSize, inputSize}, elementType);
  SmallVector<Type, 2> w3D2Ty(2, w3DTy);
  SmallVector<Type, 3> wSplit2D3Ty(3, wSplit2DTy);

  // MemRef types for recurrence weights.
  auto r3DTy = MemRefType::get({1, 3 * hiddenSize, hiddenSize}, elementType);
  auto r2DTy = MemRefType::get({3 * hiddenSize, hiddenSize}, elementType);
  auto rTransposeTy =
      MemRefType::get({hiddenSize, 3 * hiddenSize}, elementType);
  auto rSplit2DTy = MemRefType::get({hiddenSize, hiddenSize}, elementType);
  auto rTranspose2DTy = MemRefType::get({hiddenSize, hiddenSize}, elementType);
  SmallVector<Type, 2> r3D2Ty(2, r3DTy);
  SmallVector<Type, 3> rSplit2D3Ty(3, rSplit2DTy);

  ArrayAttr permAttr = rewriter.getI64ArrayAttr({1, 0});

  // Squeeze the direction axis from W and R.
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
    // W
    weightForward.WT = foldOrEmitONNXTransposeOpKrnl(
        rewriter, loc, wTransposeTy, fW, permAttr);
    // R
    if (linearBeforeReset) {
      weightForward.RT = foldOrEmitONNXTransposeOpKrnl(
          rewriter, loc, rTransposeTy, fR, permAttr);
    } else {
      std::vector<Value> vals =
          foldOrEmitONNXSplitV11OpKrnl(rewriter, loc, rSplit2D3Ty, fR, 0);
      weightForward.Rz = foldOrEmitONNXTransposeOpKrnl(
          rewriter, loc, rTranspose2DTy, vals[0], permAttr);
      weightForward.Rr = foldOrEmitONNXTransposeOpKrnl(
          rewriter, loc, rTranspose2DTy, vals[1], permAttr);
      weightForward.Rh = foldOrEmitONNXTransposeOpKrnl(
          rewriter, loc, rTranspose2DTy, vals[2], permAttr);
    }
  }
  if (direction == REVERSE || direction == BIDIRECTIONAL) {
    // W
    weightReverse.WT = foldOrEmitONNXTransposeOpKrnl(
        rewriter, loc, wTransposeTy, bW, permAttr);
    // R
    if (linearBeforeReset) {
      weightReverse.RT = foldOrEmitONNXTransposeOpKrnl(
          rewriter, loc, rTransposeTy, bR, permAttr);
    } else {
      std::vector<Value> vals =
          foldOrEmitONNXSplitV11OpKrnl(rewriter, loc, rSplit2D3Ty, bR, 0);
      weightReverse.Rz = foldOrEmitONNXTransposeOpKrnl(
          rewriter, loc, rTranspose2DTy, vals[0], permAttr);
      weightReverse.Rr = foldOrEmitONNXTransposeOpKrnl(
          rewriter, loc, rTranspose2DTy, vals[1], permAttr);
      weightReverse.Rh = foldOrEmitONNXTransposeOpKrnl(
          rewriter, loc, rTranspose2DTy, vals[2], permAttr);
    }
  }
  return std::make_tuple(weightForward, weightReverse);
}

template <>
std::tuple<GruBiasPack, GruBiasPack> getBiasPack<ONNXGRUOp, GruBiasPack>(
    ConversionPatternRewriter &rewriter, Location loc, ONNXGRUOp *op) {
  // Return values.
  GruBiasPack biasForward, biasReverse;

  // bias: [direction, 6*hiddenSize] for both parameter and recurrence weights.
  Value B = op->getB();

  // direction
  StringRef direction = op->getDirection();

  MultiDialectBuilder<KrnlBuilder, MathBuilder, MemRefBuilder, OnnxBuilder>
      create(rewriter, loc);
  // Split B.
  if (!isNoneValue(B)) {
    ArrayRef<int64_t> bShape = mlir::cast<ShapedType>(B.getType()).getShape();
    Type elementType = mlir::cast<ShapedType>(B.getType()).getElementType();
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
          foldOrEmitONNXSplitV11OpKrnl(rewriter, loc, split1D6Ty, fB, 0);
      biasForward.Wbz = vals[0];
      biasForward.Wbr = vals[1];
      biasForward.Wbh = vals[2];
      biasForward.Rbz = vals[3];
      biasForward.Rbr = vals[4];
      biasForward.Rbh = vals[5];
      biasForward.hasBias = true;
    }
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      std::vector<Value> vals =
          foldOrEmitONNXSplitV11OpKrnl(rewriter, loc, split1D6Ty, bB, 0);
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
    ConversionPatternRewriter &rewriter, Location loc,
    const TypeConverter *typeConverter, ONNXGRUOp *op,
    typename ONNXGRUOp::Adaptor operandAdaptor) {
  GruState state;

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
  if (direction == FORWARD || direction == BIDIRECTIONAL) {
    state.forwardHt = allocIntermediateState(
        rewriter, loc, operandAdaptor.getX(), operandAdaptor.getR());
  }
  if (direction == REVERSE || direction == BIDIRECTIONAL) {
    state.reverseHt = allocIntermediateState(
        rewriter, loc, operandAdaptor.getX(), operandAdaptor.getR());
  }

  // Initialize Ht.
  Value noneValue;
  initializeIntermediateStates(rewriter, loc, state.forwardHt, state.reverseHt,
      noneValue, noneValue, operandAdaptor.getInitialH(), noneValue,
      mlir::cast<MemRefType>(operandAdaptor.getX().getType()).getElementType(),
      direction, /*onlyHidden=*/true);

  // Obtain the value of 'linear_before_reset' attribute.
  int64_t linearBeforeResetAttr = op->getLinearBeforeReset();
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
    GruBiasPack biasPack, Value sequenceIV, Value directionIV,
    Value sequenceLens, Value initialH, bool isForward) {
  // Equations (Default: f=Sigmoid, g=Tanh):"
  // zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)"
  // rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)"
  // if (linearBeforeReset)
  //   ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh)
  // else
  //   ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh)
  // Ht = (1 - zt) (.) ht + zt (.) Ht-1"

  MultiDialectBuilder<KrnlBuilder, MathBuilder, MemRefBuilder, OnnxBuilder>
      create(rewriter, loc);

  ArrayRef<int64_t> xtShape = mlir::cast<ShapedType>(Xt.getType()).getShape();
  int64_t batchSize = xtShape[0];

  // Get Ht.
  Value Ht = (isForward) ? state.forwardHt : state.reverseHt;

  ArrayRef<int64_t> htShape = mlir::cast<ShapedType>(Ht.getType()).getShape();
  int64_t hiddenSize = htShape[1];

  // Frequently used types.
  MemRefType matrixType = mlir::cast<MemRefType>(Ht.getType());
  unsigned htRank = matrixType.getRank();
  Type elementType = matrixType.getElementType();
  MemRefType matrixAllGatesType =
      MemRefType::get({batchSize, 3 * hiddenSize}, elementType);

  // Common matrix multiplications.
  Value XtWT = create.onnx.toMemref(
      create.onnx.matmul(matrixAllGatesType, Xt, weightPack.WT));
  Value one = create.math.constant(elementType, 1);

  // Lower and upper bounds derived from Ht tensor.
  Value iZero = create.math.constantIndex(0);
  SmallVector<Value, 4> htLbs(htRank, iZero);
  SmallVector<Value, 4> htUbs;
  for (unsigned r = 0; r < htRank; ++r) {
    htUbs.emplace_back(create.mem.dim(Ht, r));
  }

  if (state.linearBeforeReset) {
    // zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)"
    // rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)"
    // ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh)
    // Ht = (1 - zt) (.) ht + zt (.) Ht-1"
    // In this case, we can do all matrix multiplications first, then fuse all
    // element-wise computations into a single nested loop.
    Value HtRT = create.onnx.toMemref(
        create.onnx.matmul(matrixAllGatesType, Ht, weightPack.RT));

    // Do element-wise computations. Fuse them into a single nested loop.
    ValueRange loops = create.krnl.defineLoops(htRank);
    create.krnl.iterate(loops, loops, htLbs, htUbs,
        [&](const KrnlBuilder &createKrnl, ValueRange indices) {
          MathBuilder createMath(createKrnl);
          IndexExprScope ieScope(createKrnl);
          Value bs(indices[0]), hs(indices[1]);
          SymbolIndexExpr bsie(bs), hsie(hs);
          LiteralIndexExpr hsieLit(hiddenSize);

          Value HtVal = createKrnl.load(Ht, indices);
          // zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
          Value XtWzVal = createKrnl.loadIE(XtWT, {bsie, hsie});
          Value HtRzVal = createKrnl.loadIE(HtRT, {bsie, hsie});
          Value zt = createMath.add(XtWzVal, HtRzVal);
          if (biasPack.hasBias) {
            Value WbzVal = createKrnl.load(biasPack.Wbz, {hs});
            Value RbzVal = createKrnl.load(biasPack.Rbz, {hs});
            zt = createMath.add(zt, WbzVal);
            zt = createMath.add(zt, RbzVal);
          }
          zt = applyActivation(
              createKrnl.getBuilder(), loc, activationPack.f, zt);
          // rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)"
          Value XtWrVal = createKrnl.loadIE(XtWT, {bsie, hsie + hsieLit});
          Value HtRrVal = createKrnl.loadIE(HtRT, {bsie, hsie + hsieLit});
          Value rt = createMath.add(XtWrVal, HtRrVal);
          if (biasPack.hasBias) {
            Value WbrVal = createKrnl.load(biasPack.Wbr, {hs});
            Value RbrVal = createKrnl.load(biasPack.Rbr, {hs});
            rt = createMath.add(rt, WbrVal);
            rt = createMath.add(rt, RbrVal);
          }
          rt = applyActivation(
              createKrnl.getBuilder(), loc, activationPack.f, rt);
          // ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh)
          Value XtWhVal = createKrnl.loadIE(XtWT, {bsie, hsie + 2 * hsieLit});
          Value HtRhVal = createKrnl.loadIE(HtRT, {bsie, hsie + 2 * hsieLit});
          if (biasPack.hasBias) {
            Value RbhVal = createKrnl.load(biasPack.Rbh, {hs});
            HtRhVal = createMath.add(HtRhVal, RbhVal);
          }
          Value rtHtRhVal = createMath.mul(rt, HtRhVal);
          Value ht = createMath.add(XtWhVal, rtHtRhVal);
          if (biasPack.hasBias) {
            Value WbhVal = createKrnl.load(biasPack.Wbh, {hs});
            ht = createMath.add(ht, WbhVal);
          }
          ht = applyActivation(
              createKrnl.getBuilder(), loc, activationPack.g, ht);
          // Ht = (1 - zt) (.) ht + zt (.) Ht-1
          Value oneMinusZt = createMath.sub(one, zt);
          Value ztht = createMath.mul(oneMinusZt, ht);
          Value ztHt = createMath.mul(zt, HtVal);
          Value nextHt = createMath.add(ztht, ztHt);

          // Store the intermediate Ht.
          // Handle sequence_lens
          nextHt = handleSequenceLens(createKrnl, createMath, sequenceLens,
              initialH, nextHt, sequenceIV, directionIV, bs, hs, Ht);

          if (!isNoneValue(state.allH))
            createKrnl.store(
                nextHt, state.allH, {sequenceIV, directionIV, bs, hs});
        });
  } else {
    // zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)"
    // rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)"
    // ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh)
    // Ht = (1 - zt) (.) ht + zt (.) Ht-1"
    // In this case, besides computing matrix multiplications, we need to
    // compute rt and (rt (.) Ht-1) first, then fuse the remaining element-wise
    // computations into a single nested loop.
    Value HtRz =
        create.onnx.toMemref(create.onnx.matmul(matrixType, Ht, weightPack.Rz));
    Value HtRr =
        create.onnx.toMemref(create.onnx.matmul(matrixType, Ht, weightPack.Rr));
    Value rt, rtHt;
    if (hasAllConstantDimensions(matrixType)) {
      rt = create.mem.alignedAlloc(matrixType);
      rtHt = create.mem.alignedAlloc(matrixType);
    } else {
      // matrixType's shape is of [BatchSize, HiddenSize].
      // HiddenSize is always static. Thus, only BatchSize is dynamic.
      Value batchSize = create.mem.dim(Ht, 0);
      rt = create.mem.alignedAlloc(matrixType, {batchSize});
      rtHt = create.mem.alignedAlloc(matrixType, {batchSize});
    }

    // Emit rt and (rt (.) Ht-1).
    ValueRange loops1 = create.krnl.defineLoops(htRank);
    create.krnl.iterate(loops1, loops1, htLbs, htUbs,
        [&](const KrnlBuilder &createKrnl, ValueRange indices) {
          MathBuilder createMath(createKrnl);
          IndexExprScope ieScope(createKrnl);
          Value bs(indices[0]), hs(indices[1]);
          SymbolIndexExpr bsie(bs), hsie(hs);
          LiteralIndexExpr hsieLit(hiddenSize);

          Value HtVal = createKrnl.load(Ht, indices);
          // rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)"
          Value XtWrVal = createKrnl.loadIE(XtWT, {bsie, hsie + hsieLit});
          Value HtRrVal = createKrnl.load(HtRr, indices);
          Value rtVal = createMath.add(XtWrVal, HtRrVal);
          if (biasPack.hasBias) {
            Value WbrVal = createKrnl.load(biasPack.Wbr, {hs});
            Value RbrVal = createKrnl.load(biasPack.Rbr, {hs});
            rtVal = createMath.add(rtVal, WbrVal);
            rtVal = createMath.add(rtVal, RbrVal);
          }
          rtVal = applyActivation(
              createKrnl.getBuilder(), loc, activationPack.f, rtVal);
          createKrnl.store(rtVal, rt, indices);
          // rt (.) Ht-1
          Value rtHtVal = createMath.mul(rtVal, HtVal);
          createKrnl.store(rtHtVal, rtHt, indices);
        });

    // Emit (rt (.) Ht-1)*(Rh^T)
    Value rtHtRh = create.onnx.toMemref(
        create.onnx.matmul(matrixType, rtHt, weightPack.Rh));

    // Do element-wise computations. Fuse them into a single nested loop.
    ValueRange loops2 = create.krnl.defineLoops(htRank);
    create.krnl.iterate(loops2, loops2, htLbs, htUbs,
        [&](const KrnlBuilder &createKrnl, ValueRange indices) {
          MathBuilder createMath(createKrnl);
          IndexExprScope ieScope(createKrnl);
          Value bs(indices[0]), hs(indices[1]);
          SymbolIndexExpr bsie(bs), hsie(hs);
          LiteralIndexExpr hsieLit(hiddenSize);

          Value HtVal = createKrnl.load(Ht, indices);
          // zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
          Value XtWzVal = createKrnl.loadIE(XtWT, {bsie, hsie});
          Value HtRzVal = createKrnl.load(HtRz, indices);
          Value zt = createMath.add(XtWzVal, HtRzVal);
          if (biasPack.hasBias) {
            Value WbzVal = createKrnl.load(biasPack.Wbz, {hs});
            Value RbzVal = createKrnl.load(biasPack.Rbz, {hs});
            zt = createMath.add(zt, WbzVal);
            zt = createMath.add(zt, RbzVal);
          }
          zt = applyActivation(
              createKrnl.getBuilder(), loc, activationPack.f, zt);
          // ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh)
          Value XtWhVal = createKrnl.loadIE(XtWT, {bsie, hsie + 2 * hsieLit});
          Value rtHtRhVal = createKrnl.load(rtHtRh, indices);
          Value ht = createMath.add(XtWhVal, rtHtRhVal);
          if (biasPack.hasBias) {
            Value WbhVal = createKrnl.load(biasPack.Wbh, {hs});
            Value RbhVal = createKrnl.load(biasPack.Rbh, {hs});
            ht = createMath.add(ht, WbhVal);
            ht = createMath.add(ht, RbhVal);
          }
          ht = applyActivation(
              createKrnl.getBuilder(), loc, activationPack.g, ht);
          // Ht = (1 - zt) (.) ht + zt (.) Ht-1
          Value oneMinusZt = createMath.sub(one, zt);
          Value ztht = createMath.mul(oneMinusZt, ht);
          Value ztHt = createMath.mul(zt, HtVal);
          Value nextHt = createMath.add(ztht, ztHt);

          // Store the intermediate Ht.
          // Handle sequence_lens
          nextHt = handleSequenceLens(createKrnl, createMath, sequenceLens,
              initialH, nextHt, sequenceIV, directionIV, bs, hs, Ht);

          if (!isNoneValue(state.allH))
            createKrnl.store(
                nextHt, state.allH, {sequenceIV, directionIV, bs, hs});
        });
  }
}

template <>
void stateToOutput<ONNXGRUOp, GruState>(ConversionPatternRewriter &rewriter,
    Location loc, ONNXGRUOp *op, GruState state, std::vector<Value> &outputs) {
  auto direction = op->getDirection();
  Value noneValue;
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

void populateLoweringONNXGRUOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXRNNOpLowering<ONNXGRUOp, GruState, GruActivationPack,
      GruWeightPack, GruBiasPack>>(typeConverter, ctx);
}

} // namespace onnx_mlir
