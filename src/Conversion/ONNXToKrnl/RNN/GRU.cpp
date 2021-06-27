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
      unsigned int startIndex = (direction == REVERSE) ? 0 : 2;
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
      unsigned int startIndex = (direction == REVERSE) ? 0 : 2;
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
      unsigned int startIndex = (direction == REVERSE) ? 0 : 2;
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
    // W
    std::vector<Value> vals =
        foldOrEmitONNXSplitOp(rewriter, loc, wSplit2D3Ty, fW, 0);
    weightForward.Wz = foldOrEmitONNXTransposeOp(
        rewriter, loc, wTranspose2DTy, vals[0], permAttr);
    weightForward.Wr = foldOrEmitONNXTransposeOp(
        rewriter, loc, wTranspose2DTy, vals[1], permAttr);
    weightForward.Wh = foldOrEmitONNXTransposeOp(
        rewriter, loc, wTranspose2DTy, vals[2], permAttr);
    // R
    vals.clear();
    vals = foldOrEmitONNXSplitOp(rewriter, loc, rSplit2D3Ty, fR, 0);
    weightForward.Rz = foldOrEmitONNXTransposeOp(
        rewriter, loc, rTranspose2DTy, vals[0], permAttr);
    weightForward.Rr = foldOrEmitONNXTransposeOp(
        rewriter, loc, rTranspose2DTy, vals[1], permAttr);
    weightForward.Rh = foldOrEmitONNXTransposeOp(
        rewriter, loc, rTranspose2DTy, vals[2], permAttr);
  }
  if (direction == REVERSE || direction == BIDIRECTIONAL) {
    // W
    std::vector<Value> vals =
        foldOrEmitONNXSplitOp(rewriter, loc, wSplit2D3Ty, bW, 0);
    weightReverse.Wz = foldOrEmitONNXTransposeOp(
        rewriter, loc, wTranspose2DTy, vals[0], permAttr);
    weightReverse.Wr = foldOrEmitONNXTransposeOp(
        rewriter, loc, wTranspose2DTy, vals[1], permAttr);
    weightReverse.Wh = foldOrEmitONNXTransposeOp(
        rewriter, loc, wTranspose2DTy, vals[2], permAttr);
    // R
    vals.clear();
    vals = foldOrEmitONNXSplitOp(rewriter, loc, rSplit2D3Ty, bR, 0);
    weightReverse.Rz = foldOrEmitONNXTransposeOp(
        rewriter, loc, rTranspose2DTy, vals[0], permAttr);
    weightReverse.Rr = foldOrEmitONNXTransposeOp(
        rewriter, loc, rTranspose2DTy, vals[1], permAttr);
    weightReverse.Rh = foldOrEmitONNXTransposeOp(
        rewriter, loc, rTranspose2DTy, vals[2], permAttr);
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
          foldOrEmitONNXSplitOp(rewriter, loc, split1D6Ty, fB, 0);
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
          foldOrEmitONNXSplitOp(rewriter, loc, split1D6Ty, bB, 0);
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

  // Common matrix multiplications.
  Value XtWz = onnx_matmul(matrixType, Xt, weightPack.Wz);
  Value HtRz = onnx_matmul(matrixType, Ht, weightPack.Rz);
  Value XtWr = onnx_matmul(matrixType, Xt, weightPack.Wr);
  Value HtRr = onnx_matmul(matrixType, Ht, weightPack.Rr);
  Value XtWh = onnx_matmul(matrixType, Xt, weightPack.Wh);
  Value one = emitConstantOp(rewriter, loc, elementType, 1);

  if (state.linearBeforeReset) {
    // zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)"
    // rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)"
    // ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh)
    // Ht = (1 - zt) (.) ht + zt (.) Ht-1"
    // In this case, we can do all matrix multiplications first, then fuse all
    // element-wise computations into a single nested loop.
    Value HtRh = onnx_matmul(matrixType, Ht, weightPack.Rh);

    // Do element-wise computations. Fuse them into a single nested loop.
    MemRefBoundsCapture bounds(Ht);
    ValueRange loops = krnl_define_loop(bounds.rank());
    krnl_iterate(
        loops, bounds.getLbs(), bounds.getUbs(), {}, [&](ValueRange args) {
          ValueRange indices = krnl_get_induction_var_value(loops);
          Value bs(indices[0]), hs(indices[1]);
          Value HtVal = krnl_load(Ht, indices);
          // zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
          Value XtWzVal = krnl_load(XtWz, indices);
          Value HtRzVal = krnl_load(HtRz, indices);
          Value zt = std_addf(XtWzVal, HtRzVal);
          if (biasPack.hasBias) {
            Value WbzVal = krnl_load(biasPack.Wbz, {hs});
            Value RbzVal = krnl_load(biasPack.Rbz, {hs});
            zt = std_addf(zt, WbzVal);
            zt = std_addf(zt, RbzVal);
          }
          zt = applyActivation(rewriter, loc, activationPack.f, zt);
          // rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)"
          Value XtWrVal = krnl_load(XtWr, indices);
          Value HtRrVal = krnl_load(HtRr, indices);
          Value rt = std_addf(XtWrVal, HtRrVal);
          if (biasPack.hasBias) {
            Value WbrVal = krnl_load(biasPack.Wbr, {hs});
            Value RbrVal = krnl_load(biasPack.Rbr, {hs});
            rt = std_addf(rt, WbrVal);
            rt = std_addf(rt, RbrVal);
          }
          rt = applyActivation(rewriter, loc, activationPack.f, rt);
          // ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh)
          Value XtWhVal = krnl_load(XtWh, indices);
          Value HtRhVal = krnl_load(HtRh, indices);
          if (biasPack.hasBias) {
            Value RbhVal = krnl_load(biasPack.Rbh, {hs});
            HtRhVal = std_addf(HtRhVal, RbhVal);
          }
          Value rtHtRhVal = std_mulf(rt, HtRhVal);
          Value ht = std_addf(XtWhVal, rtHtRhVal);
          if (biasPack.hasBias) {
            Value WbhVal = krnl_load(biasPack.Wbh, {hs});
            ht = std_addf(ht, WbhVal);
          }
          ht = applyActivation(rewriter, loc, activationPack.g, ht);
          // Ht = (1 - zt) (.) ht + zt (.) Ht-1
          Value oneMinusZt = std_subf(one, zt);
          Value ztht = std_mulf(oneMinusZt, ht);
          Value ztHt = std_mulf(zt, HtVal);
          Value nextHt = std_addf(ztht, ztHt);

          // Store the intermediate Ht.
          krnl_store(nextHt, Ht, indices);
          if (!isNoneType(state.allH))
            krnl_store(nextHt, state.allH, {sequenceIV, directionIV, bs, hs});
        });
  } else {
    // zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)"
    // rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)"
    // ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh)
    // Ht = (1 - zt) (.) ht + zt (.) Ht-1"
    // In this case, besides computing matrix multiplications, we need to
    // compute rt and (rt (.) Ht-1) first, then fuse the remaining element-wise
    // computations into a single nested loop.
    Value rt, rtHt;
    if (hasAllConstantDimensions(matrixType)) {
      rt = insertAllocAndDealloc(matrixType, loc, rewriter, false);
      rtHt = insertAllocAndDealloc(matrixType, loc, rewriter, false);
    } else {
      // matrixType's shape is of [BatchSize, HiddenSize].
      // HiddenSize is always static. Thus, only BatchSize is dynamic.
      Value batchSize = rewriter.create<memref::DimOp>(loc, Ht, 0).getResult();
      rt = memref_alloc(matrixType, llvm::makeArrayRef({batchSize}));
      rtHt = memref_alloc(matrixType, llvm::makeArrayRef({batchSize}));
    }

    // Emit rt and (rt (.) Ht-1).
    MemRefBoundsCapture bounds(Ht);
    ValueRange loops1 = krnl_define_loop(bounds.rank());
    krnl_iterate(
        loops1, bounds.getLbs(), bounds.getUbs(), {}, [&](ValueRange args) {
          ValueRange indices = krnl_get_induction_var_value(loops1);
          Value hs(indices[1]);
          Value HtVal = krnl_load(Ht, indices);
          // rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)"
          Value XtWrVal = krnl_load(XtWr, indices);
          Value HtRrVal = krnl_load(HtRr, indices);
          Value rtVal = std_addf(XtWrVal, HtRrVal);
          if (biasPack.hasBias) {
            Value WbrVal = krnl_load(biasPack.Wbr, {hs});
            Value RbrVal = krnl_load(biasPack.Rbr, {hs});
            rtVal = std_addf(rtVal, WbrVal);
            rtVal = std_addf(rtVal, RbrVal);
          }
          rtVal = applyActivation(rewriter, loc, activationPack.f, rtVal);
          krnl_store(rtVal, rt, indices);
          // rt (.) Ht-1
          Value rtHtVal = std_mulf(rtVal, HtVal);
          krnl_store(rtHtVal, rtHt, indices);
        });

    // Emit (rt (.) Ht-1)*(Rh^T)
    Value rtHtRh = onnx_matmul(matrixType, rtHt, weightPack.Rh);

    // Do element-wise computations. Fuse them into a single nested loop.
    ValueRange loops2 = krnl_define_loop(bounds.rank());
    krnl_iterate(
        loops2, bounds.getLbs(), bounds.getUbs(), {}, [&](ValueRange args) {
          ValueRange indices = krnl_get_induction_var_value(loops2);
          Value bs(indices[0]), hs(indices[1]);
          Value HtVal = krnl_load(Ht, indices);
          // zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
          Value XtWzVal = krnl_load(XtWz, indices);
          Value HtRzVal = krnl_load(HtRz, indices);
          Value zt = std_addf(XtWzVal, HtRzVal);
          if (biasPack.hasBias) {
            Value WbzVal = krnl_load(biasPack.Wbz, {hs});
            Value RbzVal = krnl_load(biasPack.Rbz, {hs});
            zt = std_addf(zt, WbzVal);
            zt = std_addf(zt, RbzVal);
          }
          zt = applyActivation(rewriter, loc, activationPack.f, zt);
          // ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh)
          Value XtWhVal = krnl_load(XtWh, indices);
          Value rtHtRhVal = krnl_load(rtHtRh, indices);
          Value ht = std_addf(XtWhVal, rtHtRhVal);
          if (biasPack.hasBias) {
            Value WbhVal = krnl_load(biasPack.Wbh, {hs});
            Value RbhVal = krnl_load(biasPack.Rbh, {hs});
            ht = std_addf(ht, WbhVal);
            ht = std_addf(ht, RbhVal);
          }
          ht = applyActivation(rewriter, loc, activationPack.g, ht);
          // Ht = (1 - zt) (.) ht + zt (.) Ht-1
          Value oneMinusZt = std_subf(one, zt);
          Value ztht = std_mulf(oneMinusZt, ht);
          Value ztHt = std_mulf(zt, HtVal);
          Value nextHt = std_addf(ztht, ztHt);

          // Store the intermediate Ht.
          krnl_store(nextHt, Ht, indices);
          if (!isNoneType(state.allH))
            krnl_store(nextHt, state.allH, {sequenceIV, directionIV, bs, hs});
        });

    // Clean up
    rewriter.create<memref::DeallocOp>(loc, rt);
    rewriter.create<memref::DeallocOp>(loc, rtHt);
  }
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
