/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- LSTM.cpp - Lowering LSTM Op --------------------------===//
//
// Copyright 2019-2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX LSTM Operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXConversionCommon/RNN/LSTM.hpp"
#include "src/Conversion/ONNXToKrnl/RNN/RNNBase.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"

using namespace mlir;

namespace onnx_mlir {

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

  // MemRef types for parameter weights.
  auto w3DTy = MemRefType::get({1, 4 * hiddenSize, inputSize}, elementType);
  auto w2DTy = MemRefType::get({4 * hiddenSize, inputSize}, elementType);
  auto wTranspose2DTy =
      MemRefType::get({inputSize, 4 * hiddenSize}, elementType);
  SmallVector<Type, 4> w3D2Ty(2, w3DTy);

  // MemRef types for recurrence weights.
  auto r3DTy = MemRefType::get({1, 4 * hiddenSize, hiddenSize}, elementType);
  auto r2DTy = MemRefType::get({4 * hiddenSize, hiddenSize}, elementType);
  auto rTranspose2DTy =
      MemRefType::get({hiddenSize, 4 * hiddenSize}, elementType);
  SmallVector<Type, 4> r3D2Ty(2, r3DTy);

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

  // Transpose W and R.
  ArrayAttr permAttr = rewriter.getI64ArrayAttr({1, 0});
  if (direction == FORWARD || direction == BIDIRECTIONAL) {
    // W
    weightForward.WT = foldOrEmitONNXTransposeOpKrnl(
        rewriter, loc, wTranspose2DTy, fW, permAttr);
    // R
    weightForward.RT = foldOrEmitONNXTransposeOpKrnl(
        rewriter, loc, rTranspose2DTy, fR, permAttr);
  }
  if (direction == REVERSE || direction == BIDIRECTIONAL) {
    // W
    weightReverse.WT = foldOrEmitONNXTransposeOpKrnl(
        rewriter, loc, wTranspose2DTy, bW, permAttr);
    // R
    weightReverse.RT = foldOrEmitONNXTransposeOpKrnl(
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
    auto bType2D = MemRefType::get({1, 8 * hiddenSize}, elementType);
    auto bType1D = MemRefType::get({8 * hiddenSize}, elementType);
    auto bSplitType1D = MemRefType::get({hiddenSize}, elementType);
    SmallVector<Type, 4> split1D8Ty(8, bSplitType1D);
    SmallVector<Type, 4> split2D2Ty(2, bType2D);

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
          foldOrEmitONNXSplitV11OpKrnl(rewriter, loc, split1D8Ty, fB, 0);
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
          foldOrEmitONNXSplitV11OpKrnl(rewriter, loc, split1D8Ty, bB, 0);
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
    auto pType2D = MemRefType::get({1, 3 * hiddenSize}, elementType);
    auto pType1D = MemRefType::get({3 * hiddenSize}, elementType);
    auto pSplitType1D = MemRefType::get({hiddenSize}, elementType);
    SmallVector<Type, 4> split1D3Ty(3, pSplitType1D);
    SmallVector<Type, 4> split2D2Ty(2, pType2D);

    // Squeeze the direction axis from P.
    Value fP, bP;
    if (direction == FORWARD) {
      fP =
          foldOrEmitONNXSqueezeV11OpKrnl(rewriter, loc, pType1D, P, /*axis=*/0);
    } else if (direction == REVERSE) {
      bP =
          foldOrEmitONNXSqueezeV11OpKrnl(rewriter, loc, pType1D, P, /*axis=*/0);
    } else { // BIDIRECTIONAL
      std::vector<Value> vals =
          foldOrEmitONNXSplitV11OpKrnl(rewriter, loc, split2D2Ty, P, 0);
      fP = foldOrEmitONNXSqueezeV11OpKrnl(
          rewriter, loc, pType1D, vals[0], /*axis=*/0);
      bP = foldOrEmitONNXSqueezeV11OpKrnl(
          rewriter, loc, pType1D, vals[1], /*axis=*/0);
    }

    // Split P into individual tensors.
    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      std::vector<Value> vals =
          foldOrEmitONNXSplitV11OpKrnl(rewriter, loc, split1D3Ty, fP, 0);
      biasForward.Pi = vals[0];
      biasForward.Po = vals[1];
      biasForward.Pf = vals[2];
      biasForward.hasPeephole = true;
    }
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      std::vector<Value> vals =
          foldOrEmitONNXSplitV11OpKrnl(rewriter, loc, split1D3Ty, bP, 0);
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
    ConversionPatternRewriter &rewriter, Location loc,
    const TypeConverter *typeConverter, ONNXLSTMOp *op,
    typename ONNXLSTMOp::Adaptor operandAdaptor) {
  LstmState state;

  // direction
  StringRef direction = op->getDirection();

  // Insert allocation and deallocation for the results of this operation.
  // If the result is not returned, then no allocation happens.
  // Y :: [seq_length, num_directions, batch_size, hidden_size]
  state.allH =
      allocAllHidden(rewriter, loc, typeConverter, operandAdaptor.getX(),
          operandAdaptor.getW(), operandAdaptor.getR(), op->getY());
  // Y_h :: [num_directions, batch_size, hidden_size]
  state.ht =
      allocHiddenOrCell(rewriter, loc, typeConverter, operandAdaptor.getX(),
          operandAdaptor.getW(), operandAdaptor.getR(), op->getYH());
  // Y_c :: [num_directions, batch_size, hidden_size]
  state.ct =
      allocHiddenOrCell(rewriter, loc, typeConverter, operandAdaptor.getX(),
          operandAdaptor.getW(), operandAdaptor.getR(), op->getYC());

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
      mlir::cast<MemRefType>(operandAdaptor.getX().getType()).getElementType(),
      direction, /*onlyHidden=*/false);
  return state;
}

template <>
void calculateState<LstmState, LstmActivationPack, LstmWeightPack,
    LstmBiasPack>(ConversionPatternRewriter &rewriter, Location loc, Value Xt,
    LstmState state, LstmActivationPack activationPack,
    LstmWeightPack weightPack, LstmBiasPack biasPack, Value sequenceIV,
    Value directionIV, Value sequenceLens, Value initialH, bool isForward) {
  // Equations for LSTM.
  // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
  // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
  // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
  // Ct = ft (.) Ct-1 + it (.) ct
  // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
  // Ht = ot (.) h(Ct)

  // ToFix: add support of sequence lens for LSTM
  // The assert will fail the test_lstm_with_peephole.
  // In that test case, the length of the input is used as sequence_lens.
  // Therefore, onnx-mlir can pass the test by ignoring the sequence_lens
  // paramenter.
  // assert(isNoneValue(sequenceLens) && "not implemented yet");

  // TODO remove scope
  MultiDialectBuilder<KrnlBuilder, MathBuilder, MemRefBuilder, OnnxBuilder>
      create(rewriter, loc);

  ArrayRef<int64_t> xtShape = mlir::cast<ShapedType>(Xt.getType()).getShape();
  int64_t batchSize = xtShape[0];

  // Get Ht, Ct.
  Value Ht = (isForward) ? state.forwardHt : state.reverseHt;
  Value Ct = (isForward) ? state.forwardCt : state.reverseCt;

  ArrayRef<int64_t> htShape = mlir::cast<ShapedType>(Ht.getType()).getShape();
  int64_t hiddenSize = htShape[1];

  // Frequently used types.
  MemRefType matrixType = mlir::cast<MemRefType>(Ht.getType());
  Type elementType = matrixType.getElementType();
  MemRefType matrixAllGatesType =
      MemRefType::get({batchSize, 4 * hiddenSize}, elementType);

  // Do matrix multiplications.
  // Xt * (Wi^T ++ Wo^T ++ Wf^T ++ Wc^T)
  // Ht * (Ri^T ++ Ro^T ++ Rf^T ++ Rc^T)
  // where '++' is matrix concatenation.
  Value XtWT = create.onnx.toMemref(
      create.onnx.matmul(matrixAllGatesType, Xt, weightPack.WT));
  Value HtRT = create.onnx.toMemref(
      create.onnx.matmul(matrixAllGatesType, Ht, weightPack.RT));

  // Do element-wise computations. Fuse them into a single nested loop.
  // Lower and upper bounds derived from Ht tensor.
  unsigned HtRank = matrixType.getRank();
  Value iZero = create.math.constantIndex(0);
  SmallVector<Value, 4> HtLbs(HtRank, iZero);
  SmallVector<Value, 4> HtUbs;
  for (unsigned r = 0; r < HtRank; ++r) {
    HtUbs.emplace_back(create.mem.dim(Ht, r));
  }

  ValueRange loops = create.krnl.defineLoops(HtRank);
  create.krnl.iterate(loops, loops, HtLbs, HtUbs,
      [&](const KrnlBuilder &createKrnl, ValueRange indices) {
        MathBuilder createMath(createKrnl);
        IndexExprScope ieScope(createKrnl);
        Value bs(indices[0]), hs(indices[1]);
        SymbolIndexExpr bsie(bs), hsie(hs);
        LiteralIndexExpr hsieLit(hiddenSize);

        Value CtVal = createKrnl.load(Ct, indices);
        // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
        Value XtWTiVal = createKrnl.loadIE(XtWT, {bsie, hsie});
        Value HtRTiVal = createKrnl.loadIE(HtRT, {bsie, hsie});
        Value it = createMath.add(XtWTiVal, HtRTiVal);
        if (biasPack.hasBias) {
          Value WbiVal = createKrnl.load(biasPack.Wbi, {hs});
          Value RbiVal = createKrnl.load(biasPack.Rbi, {hs});
          it = createMath.add(it, WbiVal);
          it = createMath.add(it, RbiVal);
        }
        if (biasPack.hasPeephole) {
          Value PiVal = createKrnl.load(biasPack.Pi, {hs});
          Value PiCt = createMath.mul(PiVal, CtVal);
          it = createMath.add(it, PiCt);
        }
        it =
            applyActivation(createKrnl.getBuilder(), loc, activationPack.f, it);

        // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
        Value XtWTfVal = createKrnl.loadIE(XtWT, {bsie, hsie + 2 * hsieLit});
        Value HtRTfVal = createKrnl.loadIE(HtRT, {bsie, hsie + 2 * hsieLit});
        Value ft = createMath.add(XtWTfVal, HtRTfVal);
        if (biasPack.hasBias) {
          Value WbfVal = createKrnl.load(biasPack.Wbf, {hs});
          Value RbfVal = createKrnl.load(biasPack.Rbf, {hs});
          ft = createMath.add(ft, WbfVal);
          ft = createMath.add(ft, RbfVal);
        }
        if (biasPack.hasPeephole) {
          Value PfVal = createKrnl.load(biasPack.Pf, {hs});
          Value PfCt = createMath.mul(PfVal, CtVal);
          ft = createMath.add(ft, PfCt);
        }
        ft =
            applyActivation(createKrnl.getBuilder(), loc, activationPack.f, ft);

        // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
        Value XtWTcVal = createKrnl.loadIE(XtWT, {bsie, hsie + 3 * hsieLit});
        Value HtRTcVal = createKrnl.loadIE(HtRT, {bsie, hsie + 3 * hsieLit});
        Value ct = createMath.add(XtWTcVal, HtRTcVal);
        if (biasPack.hasBias) {
          Value WbcVal = createKrnl.load(biasPack.Wbc, {hs});
          Value RbcVal = createKrnl.load(biasPack.Rbc, {hs});
          ct = createMath.add(ct, WbcVal);
          ct = createMath.add(ct, RbcVal);
        }
        ct =
            applyActivation(createKrnl.getBuilder(), loc, activationPack.g, ct);

        // Ct = ft (.) Ct-1 + it (.) ct
        Value ftCt = createMath.mul(ft, CtVal);
        Value itct = createMath.mul(it, ct);
        Value nextCt = createMath.add(ftCt, itct);

        // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
        Value XtWToVal = createKrnl.loadIE(XtWT, {bsie, hsie + hsieLit});
        Value HtRToVal = createKrnl.loadIE(HtRT, {bsie, hsie + hsieLit});
        Value ot = createMath.add(XtWToVal, HtRToVal);
        if (biasPack.hasBias) {
          Value WboVal = createKrnl.load(biasPack.Wbo, {hs});
          Value RboVal = createKrnl.load(biasPack.Rbo, {hs});
          ot = createMath.add(ot, WboVal);
          ot = createMath.add(ot, RboVal);
        }
        if (biasPack.hasPeephole) {
          Value PoVal = createKrnl.load(biasPack.Po, {hs});
          Value PoCt = createMath.mul(PoVal, nextCt);
          ot = createMath.add(ot, PoCt);
        }
        ot =
            applyActivation(createKrnl.getBuilder(), loc, activationPack.f, ot);

        // Ht = ot (.) h(Ct)
        Value nextHt = applyActivation(
            createKrnl.getBuilder(), loc, activationPack.h, nextCt);
        nextHt = createMath.mul(ot, nextHt);

        // Store the intermediate Ht, Ct.
        createKrnl.store(nextCt, Ct, indices);
        createKrnl.store(nextHt, Ht, indices);
        if (!isNoneValue(state.allH))
          createKrnl.store(
              nextHt, state.allH, {sequenceIV, directionIV, bs, hs});
      });
}

template <>
void stateToOutput<ONNXLSTMOp, LstmState>(ConversionPatternRewriter &rewriter,
    Location loc, ONNXLSTMOp *op, LstmState state,
    std::vector<Value> &outputs) {
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
  // Third output: cell.
  if (isNoneValue(op->getYC()))
    outputs.emplace_back(noneValue);
  else {
    stateToOutputForHiddenOrCell(
        rewriter, loc, state.forwardCt, state.reverseCt, direction, state.ct);
    outputs.emplace_back(state.ct);
  }
}

void populateLoweringONNXLSTMOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXRNNOpLowering<ONNXLSTMOp, LstmState, LstmActivationPack,
      LstmWeightPack, LstmBiasPack>>(typeConverter, ctx);
}

} // namespace onnx_mlir
