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

#include "mlir/Dialect/Affine/EDSC/Intrinsics.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/Vector/EDSC/Intrinsics.h"

using namespace mlir;

Value noneVal;

struct LstmState {
  Value allH;
  Value ht;
  Value ct;
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
  Value Wbi = noneVal;
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

  // gate weight: [direction, 4*hiddenSize, inputSize]
  Value W = op->W();
  // recurrent weight: [direction, 4*hiddenSize, hiddenSize]
  Value R = op->R();
  // direction
  StringRef direction = op->direction();

  ArrayRef<int64_t> wShape = W.getType().cast<ShapedType>().getShape();
  Type elementType = W.getType().cast<ShapedType>().getElementType();
  int64_t hiddenSize = wShape[1] / 4;
  int64_t inputSize = wShape[2];

  // MemRef types.
  MemRefType wType3D =
      MemRefType::get({1, 4 * hiddenSize, inputSize}, elementType);
  MemRefType wType2D =
      MemRefType::get({4 * hiddenSize, inputSize}, elementType);
  MemRefType wSplitType2D =
      MemRefType::get({hiddenSize, inputSize}, elementType);
  MemRefType wTransposeType2D =
      MemRefType::get({inputSize, hiddenSize}, elementType);

  MemRefType rType3D =
      MemRefType::get({1, 4 * hiddenSize, hiddenSize}, elementType);
  MemRefType rType2D =
      MemRefType::get({4 * hiddenSize, hiddenSize}, elementType);
  MemRefType rSplitType2D =
      MemRefType::get({hiddenSize, hiddenSize}, elementType);
  MemRefType rTransposeType2D =
      MemRefType::get({hiddenSize, hiddenSize}, elementType);
  ArrayAttr permAttr = rewriter.getI64ArrayAttr({1, 0});

  // Eliminate the direction axis from W and R.
  Value fW, bW, fR, bR;
  if (direction == FORWARD) {
    fW = rewriter.create<ONNXUnsqueezeOp>(
        loc, wType2D, W, rewriter.getI64ArrayAttr(0));
    fR = rewriter.create<ONNXUnsqueezeOp>(
        loc, rType2D, R, rewriter.getI64ArrayAttr(0));
  } else if (direction == REVERSE) {
    bW = rewriter.create<ONNXUnsqueezeOp>(
        loc, wType2D, W, rewriter.getI64ArrayAttr(0));
    bR = rewriter.create<ONNXUnsqueezeOp>(
        loc, rType2D, R, rewriter.getI64ArrayAttr(0));
  } else { // BIDIRECTIONAL
    // W
    ONNXSplitOp splitW =
        rewriter.create<ONNXSplitOp>(loc, ArrayRef<Type>{wType3D, wType3D}, W,
            /*axis=*/0, rewriter.getI64ArrayAttr({1, 1}));
    fW = rewriter.create<ONNXUnsqueezeOp>(
        loc, wType2D, splitW.getResults()[0], rewriter.getI64ArrayAttr(0));
    bW = rewriter.create<ONNXUnsqueezeOp>(
        loc, wType2D, splitW.getResults()[1], rewriter.getI64ArrayAttr(0));
    // R
    ONNXSplitOp splitR =
        rewriter.create<ONNXSplitOp>(loc, ArrayRef<Type>{rType3D, rType3D}, R,
            /*axis=*/0, nullptr);
    fR = rewriter.create<ONNXUnsqueezeOp>(
        loc, rType2D, splitR.getResults()[0], rewriter.getI64ArrayAttr(0));
    bR = rewriter.create<ONNXUnsqueezeOp>(
        loc, rType2D, splitR.getResults()[1], rewriter.getI64ArrayAttr(0));
  }

  // Split W and R into invidual weight tensors, and transpose them.
  if (direction == FORWARD || direction == BIDIRECTIONAL) {
    // W
    ONNXSplitOp splitFW = rewriter.create<ONNXSplitOp>(loc,
        ArrayRef<Type>{wSplitType2D, wSplitType2D, wSplitType2D, wSplitType2D},
        fW,
        /*axis=*/0, nullptr);
    weightForward.Wi = rewriter.create<ONNXTransposeOp>(
        loc, wTransposeType2D, splitFW.getResults()[0], permAttr);
    weightForward.Wo = rewriter.create<ONNXTransposeOp>(
        loc, wTransposeType2D, splitFW.getResults()[1], permAttr);
    weightForward.Wf = rewriter.create<ONNXTransposeOp>(
        loc, wTransposeType2D, splitFW.getResults()[2], permAttr);
    weightForward.Wc = rewriter.create<ONNXTransposeOp>(
        loc, wTransposeType2D, splitFW.getResults()[3], permAttr);
    // R
    ONNXSplitOp splitFR = rewriter.create<ONNXSplitOp>(loc,
        ArrayRef<Type>{rSplitType2D, rSplitType2D, rSplitType2D, rSplitType2D},
        fR,
        /*axis=*/0, nullptr);
    weightForward.Ri = rewriter.create<ONNXTransposeOp>(
        loc, rTransposeType2D, splitFR.getResults()[0], permAttr);
    weightForward.Ro = rewriter.create<ONNXTransposeOp>(
        loc, rTransposeType2D, splitFR.getResults()[1], permAttr);
    weightForward.Rf = rewriter.create<ONNXTransposeOp>(
        loc, rTransposeType2D, splitFR.getResults()[2], permAttr);
    weightForward.Rc = rewriter.create<ONNXTransposeOp>(
        loc, rTransposeType2D, splitFR.getResults()[3], permAttr);
  }
  if (direction == REVERSE || direction == BIDIRECTIONAL) {
    // W
    ONNXSplitOp splitBW = rewriter.create<ONNXSplitOp>(loc,
        ArrayRef<Type>{wSplitType2D, wSplitType2D, wSplitType2D, wSplitType2D},
        bW,
        /*axis=*/0, nullptr);
    weightReverse.Ri = rewriter.create<ONNXTransposeOp>(
        loc, wTransposeType2D, splitBW.getResults()[0], permAttr);
    weightReverse.Ro = rewriter.create<ONNXTransposeOp>(
        loc, wTransposeType2D, splitBW.getResults()[1], permAttr);
    weightReverse.Rf = rewriter.create<ONNXTransposeOp>(
        loc, wTransposeType2D, splitBW.getResults()[2], permAttr);
    weightReverse.Rc = rewriter.create<ONNXTransposeOp>(
        loc, wTransposeType2D, splitBW.getResults()[3], permAttr);
    // R
    ONNXSplitOp splitBR = rewriter.create<ONNXSplitOp>(loc,
        ArrayRef<Type>{rSplitType2D, rSplitType2D, rSplitType2D, rSplitType2D},
        bR,
        /*axis=*/0, nullptr);
    weightReverse.Ri = rewriter.create<ONNXTransposeOp>(
        loc, rTransposeType2D, splitBR.getResults()[0], permAttr);
    weightReverse.Ro = rewriter.create<ONNXTransposeOp>(
        loc, rTransposeType2D, splitBR.getResults()[1], permAttr);
    weightReverse.Rf = rewriter.create<ONNXTransposeOp>(
        loc, rTransposeType2D, splitBR.getResults()[2], permAttr);
    weightReverse.Rc = rewriter.create<ONNXTransposeOp>(
        loc, rTransposeType2D, splitBR.getResults()[3], permAttr);
  }
  return std::make_tuple(weightForward, weightReverse);
}

template <>
std::tuple<LstmBiasPack, LstmBiasPack> getBiasPack<ONNXLSTMOp, LstmBiasPack>(
    ConversionPatternRewriter &rewriter, Location loc, ONNXLSTMOp *op) {
  // Return values.
  LstmBiasPack biasForward, biasReverse;

  // bias: [direction, 8*hiddenSize] for both gates and recurrent.
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
    MemRefType bType2D = MemRefType::get({1, 8 * hiddenSize}, elementType);
    MemRefType bType1D = MemRefType::get({8 * hiddenSize}, elementType);
    MemRefType bSplitType1D = MemRefType::get({hiddenSize}, elementType);

    // Eliminate the direction axis from B.
    Value fB, bB;
    if (direction == FORWARD) {
      fB = rewriter.create<ONNXUnsqueezeOp>(
          loc, bType1D, B, rewriter.getI64ArrayAttr(0));
    } else if (direction == REVERSE) {
      bB = rewriter.create<ONNXUnsqueezeOp>(
          loc, bType1D, B, rewriter.getI64ArrayAttr(0));
    } else { // BIDIRECTIONAL
      ONNXSplitOp splitW =
          rewriter.create<ONNXSplitOp>(loc, ArrayRef<Type>{bType2D, bType2D}, B,
              /*axis=*/0, rewriter.getI64ArrayAttr({1, 1}));
      fB = rewriter.create<ONNXUnsqueezeOp>(
          loc, bType1D, splitW.getResults()[0], rewriter.getI64ArrayAttr(0));
      bB = rewriter.create<ONNXUnsqueezeOp>(
          loc, bType1D, splitW.getResults()[1], rewriter.getI64ArrayAttr(0));
    }

    // Split B into invidual bias tensors.
    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      ONNXSplitOp splitB = rewriter.create<ONNXSplitOp>(loc,
          ArrayRef<Type>{bSplitType1D, bSplitType1D, bSplitType1D, bSplitType1D,
              bSplitType1D, bSplitType1D, bSplitType1D, bSplitType1D},
          fB,
          /*axis=*/0, nullptr);
      biasForward.Wbi = splitB.getResults()[0];
      biasForward.Wbo = splitB.getResults()[1];
      biasForward.Wbf = splitB.getResults()[2];
      biasForward.Wbc = splitB.getResults()[3];
      biasForward.Rbi = splitB.getResults()[4];
      biasForward.Rbo = splitB.getResults()[5];
      biasForward.Rbf = splitB.getResults()[6];
      biasForward.Rbc = splitB.getResults()[7];
    }
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      ONNXSplitOp splitB =
          rewriter.create<ONNXSplitOp>(loc, ArrayRef<Type>{bSplitType1D}, bB,
              /*axis=*/0, nullptr);
      biasReverse.Wbi = splitB.getResults()[0];
      biasReverse.Wbo = splitB.getResults()[1];
      biasReverse.Wbf = splitB.getResults()[2];
      biasReverse.Wbc = splitB.getResults()[3];
      biasReverse.Rbi = splitB.getResults()[4];
      biasReverse.Rbo = splitB.getResults()[5];
      biasReverse.Rbf = splitB.getResults()[6];
      biasReverse.Rbc = splitB.getResults()[7];
    }
  }

  // Split P.
  if (!isNoneType(P)) {
    ArrayRef<int64_t> pShape = P.getType().cast<ShapedType>().getShape();
    Type elementType = P.getType().cast<ShapedType>().getElementType();
    int64_t hiddenSize = pShape[1] / 3;

    // MemRef types.
    MemRefType pType2D = MemRefType::get({1, 3 * hiddenSize}, elementType);
    MemRefType pType1D = MemRefType::get({3 * hiddenSize}, elementType);
    MemRefType pSplitType1D = MemRefType::get({hiddenSize}, elementType);

    // Eliminate the direction axis from P.
    Value fP, bP;
    if (direction == FORWARD) {
      fP = rewriter.create<ONNXUnsqueezeOp>(
          loc, pType1D, P, rewriter.getI64ArrayAttr(0));
    } else if (direction == REVERSE) {
      bP = rewriter.create<ONNXUnsqueezeOp>(
          loc, pType1D, P, rewriter.getI64ArrayAttr(0));
    } else { // BIDIRECTIONAL
      ONNXSplitOp splitW =
          rewriter.create<ONNXSplitOp>(loc, ArrayRef<Type>{pType2D, pType2D}, P,
              /*axis=*/0, rewriter.getI64ArrayAttr({1, 1}));
      fP = rewriter.create<ONNXUnsqueezeOp>(
          loc, pType1D, splitW.getResults()[0], rewriter.getI64ArrayAttr(0));
      bP = rewriter.create<ONNXUnsqueezeOp>(
          loc, pType1D, splitW.getResults()[1], rewriter.getI64ArrayAttr(0));
    }

    // Split P into invidual tensors.
    if (direction == FORWARD || direction == BIDIRECTIONAL) {
      ONNXSplitOp splitP = rewriter.create<ONNXSplitOp>(loc,
          ArrayRef<Type>{pSplitType1D, pSplitType1D, pSplitType1D}, fP,
          /*axis=*/0, nullptr);
      biasForward.Pi = splitP.getResults()[0];
      biasForward.Po = splitP.getResults()[1];
      biasForward.Pf = splitP.getResults()[2];
    }
    if (direction == REVERSE || direction == BIDIRECTIONAL) {
      ONNXSplitOp splitP =
          rewriter.create<ONNXSplitOp>(loc, ArrayRef<Type>{pSplitType1D}, bP,
              /*axis=*/0, nullptr);
      biasReverse.Pi = splitP.getResults()[0];
      biasReverse.Po = splitP.getResults()[1];
      biasReverse.Pf = splitP.getResults()[2];
    }
  }

  return std::make_tuple(biasForward, biasReverse);
}

template <>
LstmState allocAndInitializeStates<ONNXLSTMOp, LstmState>(
    ConversionPatternRewriter &rewriter, Location loc, ONNXLSTMOp *op,
    typename ONNXLSTMOp::Adaptor operandAdaptor) {
  LstmState state;

  // Insert allocation and deallocation for the results of this operation.
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

  // Initialize ht and ct.
  initializeHiddenAndCell(rewriter, loc, state.ht, state.ct,
      operandAdaptor.initial_h(), operandAdaptor.initial_c(),
      operandAdaptor.X().getType().cast<MemRefType>().getElementType(),
      /*onlyHidden=*/false);
  return state;
}

template <>
void calculateState<ONNXLSTMOp, LstmState, LstmActivationPack, LstmWeightPack,
    LstmBiasPack>(ConversionPatternRewriter &rewriter, Location loc,
    typename ONNXLSTMOp::Adaptor operandAdaptor, LstmState state,
    LstmActivationPack activationPack, LstmWeightPack weightPack,
    LstmBiasPack biasPack, Value directionIV, Value sequenceIV) {

  // Scope for krnl EDSC ops
  using namespace mlir::edsc;
  // Scope for std EDSC ops
  using namespace edsc::intrinsics;
  ScopedContext scope(rewriter, loc);

  // Prepare dimensions.
  Value batchSizeVal = getDimOrConstant(
      rewriter, loc, operandAdaptor.X(), 1, rewriter.getIndexType());
  Value inputSizeVal = getDimOrConstant(
      rewriter, loc, operandAdaptor.X(), 2, rewriter.getIndexType());
  Value hiddenSizeVal = getDimOrConstant(
      rewriter, loc, operandAdaptor.R(), 2, rewriter.getIndexType());

  auto elementType =
      operandAdaptor.X().getType().cast<ShapedType>().getElementType();
  bool hasBiasForInput = false, hasPeepholes = false;
  if (!isNoneType(operandAdaptor.B()))
    hasBiasForInput = true;
  if (!isNoneType(operandAdaptor.P()))
    hasPeepholes = true;

  // Equations for LSTM.
  // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
  // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
  // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
  // Ct = ft (.) Ct-1 + it (.) ct
  // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
  // Ht = ot (.) h(Ct)
  //
  // Shape information:
  // Xt : [seq_length, batch_size, input_size]
  // W[iofc] : [num_directions, hidden_size, input_size]
  // R[iofc] : [num_directions, hidden_size, hidden_size]
  // Ht, Ct, it, ot, ft, ct: [num_directions, batch_size, hidden_size]
  // Wb[iofc] : [num_directions, hidden_size]
  // Rb[iofc] : [num_directions, hidden_size]

  // Copy Xt.
  Value Xt;
  MemRefType XtType = MemRefType::get(
      {dimAt(operandAdaptor.X(), 1), dimAt(operandAdaptor.X(), 2)},
      elementType);
  if (hasAllConstantDimensions(XtType))
    Xt = insertAllocAndDealloc(XtType, loc, rewriter, true);
  else {
    auto memRefShape = XtType.getShape();
    SmallVector<Value, 2> allocOperands;
    if (memRefShape[0] < 0) {
      allocOperands.emplace_back(batchSizeVal);
    }
    if (memRefShape[1] < 0) {
      allocOperands.emplace_back(inputSizeVal);
    }
    Xt = rewriter.create<AllocOp>(loc, XtType, allocOperands);
    //auto *parentBlock = Xt.getDefiningOp()->getBlock();
    //auto dealloc = rewriter.create<DeallocOp>(loc, Xt);
    //dealloc.getOperation()->moveBefore(&parentBlock->back());
  }
  BuildKrnlLoop xtLoops(rewriter, loc, 2);
  xtLoops.createDefineOp();
  xtLoops.pushBounds(0, batchSizeVal);
  xtLoops.pushBounds(0, inputSizeVal);
  xtLoops.createIterateOp();
  {
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(xtLoops.getIterateBlock());
    auto batchIV = xtLoops.getInductionVar(0);
    auto inputIV = xtLoops.getInductionVar(1);
    Value val = krnl_load(
        operandAdaptor.X(), ArrayRef<Value>{sequenceIV, batchIV, inputIV});
    krnl_store(val, Xt, ArrayRef<Value>{batchIV, inputIV});
  }

  // Copy Ht
  Value Ht;
  MemRefType HtType = MemRefType::get(
      {dimAt(operandAdaptor.X(), 1), dimAt(operandAdaptor.R(), 2)},
      elementType);
  if (hasAllConstantDimensions(HtType))
    Ht = insertAllocAndDealloc(HtType, loc, rewriter, true);
  else {
    auto memRefShape = HtType.getShape();
    SmallVector<Value, 2> allocOperands;
    if (memRefShape[0] < 0) {
      allocOperands.emplace_back(batchSizeVal);
    }
    if (memRefShape[1] < 0) {
      allocOperands.emplace_back(inputSizeVal);
    }
    Ht = rewriter.create<AllocOp>(loc, HtType, allocOperands);
    //auto *parentBlock = Ht.getDefiningOp()->getBlock();
    //auto dealloc = rewriter.create<DeallocOp>(loc, Ht);
    //dealloc.getOperation()->moveBefore(&parentBlock->back());
  }
  Value Ct;
  MemRefType CtType = MemRefType::get(
      {dimAt(operandAdaptor.X(), 1), dimAt(operandAdaptor.R(), 2)},
      elementType);
  if (hasAllConstantDimensions(CtType))
    Ct = insertAllocAndDealloc(CtType, loc, rewriter, true);
  else {
    auto memRefShape = CtType.getShape();
    SmallVector<Value, 2> allocOperands;
    if (memRefShape[0] < 0) {
      allocOperands.emplace_back(batchSizeVal);
    }
    if (memRefShape[1] < 0) {
      allocOperands.emplace_back(inputSizeVal);
    }
    Ct = rewriter.create<AllocOp>(loc, CtType, allocOperands);
    //auto *parentBlock = Ct.getDefiningOp()->getBlock();
    //auto dealloc = rewriter.create<DeallocOp>(loc, Ct);
    //dealloc.getOperation()->moveBefore(&parentBlock->back());
  }

  BuildKrnlLoop htLoops(rewriter, loc, 2);
  htLoops.createDefineOp();
  htLoops.pushBounds(0, batchSizeVal);
  htLoops.pushBounds(0, hiddenSizeVal);
  htLoops.createIterateOp();
  {
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(htLoops.getIterateBlock());
    auto batchIV = htLoops.getInductionVar(0);
    auto hiddenIV = htLoops.getInductionVar(1);
    // Copy Ht.
    Value val =
        krnl_load(state.ht, ArrayRef<Value>{sequenceIV, batchIV, hiddenIV});
    krnl_store(val, Ht, ArrayRef<Value>{batchIV, hiddenIV});
    // Copy Ct.
    val = krnl_load(state.ct, ArrayRef<Value>{sequenceIV, batchIV, hiddenIV});
    krnl_store(val, Ct, ArrayRef<Value>{batchIV, hiddenIV});
  }

  MemRefType matrixType = MemRefType::get(
      {dimAt(operandAdaptor.X(), 1), dimAt(operandAdaptor.R(), 2)},
      elementType);

  // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
  Value XtWi =
      rewriter.create<ONNXMatMulOp>(loc, matrixType, Xt, weightPack.Wi);
  Value HtRi =
      rewriter.create<ONNXMatMulOp>(loc, matrixType, Ht, weightPack.Ri);
  Value it = rewriter.create<ONNXAddOp>(loc, matrixType, XtWi, HtRi);
  if (hasBiasForInput) {
    it = rewriter.create<ONNXAddOp>(loc, matrixType, it, biasPack.Wbi);
    it = rewriter.create<ONNXAddOp>(loc, matrixType, it, biasPack.Rbi);
  }
  if (hasPeepholes) {
    Value PiCt = rewriter.create<ONNXMulOp>(loc, matrixType, biasPack.Pi, Ct);
    it = rewriter.create<ONNXAddOp>(loc, matrixType, it, PiCt);
  }
  it = applyActivation(rewriter, loc, activationPack.f, it);

  // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
  Value XtWf =
      rewriter.create<ONNXMatMulOp>(loc, matrixType, Xt, weightPack.Wf);
  Value HtRf =
      rewriter.create<ONNXMatMulOp>(loc, matrixType, Ht, weightPack.Rf);
  Value ft = rewriter.create<ONNXAddOp>(loc, matrixType, XtWf, HtRf);
  if (hasBiasForInput) {
    ft = rewriter.create<ONNXAddOp>(loc, matrixType, ft, biasPack.Wbf);
    ft = rewriter.create<ONNXAddOp>(loc, matrixType, ft, biasPack.Rbf);
  }
  if (hasPeepholes) {
    Value PfCt = rewriter.create<ONNXMulOp>(loc, matrixType, biasPack.Pf, Ct);
    ft = rewriter.create<ONNXAddOp>(loc, matrixType, ft, PfCt);
  }
  ft = applyActivation(rewriter, loc, activationPack.f, ft);

  // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
  Value XtWc =
      rewriter.create<ONNXMatMulOp>(loc, matrixType, Xt, weightPack.Wc);
  Value HtRc =
      rewriter.create<ONNXMatMulOp>(loc, matrixType, Ht, weightPack.Rc);
  Value ct = rewriter.create<ONNXAddOp>(loc, matrixType, XtWc, HtRc);
  if (hasBiasForInput) {
    ct = rewriter.create<ONNXAddOp>(loc, matrixType, ct, biasPack.Wbc);
    ct = rewriter.create<ONNXAddOp>(loc, matrixType, ct, biasPack.Rbc);
  }
  ct = applyActivation(rewriter, loc, activationPack.g, ct);

  // Ct = ft (.) Ct-1 + it (.) ct
  Value ftCt = rewriter.create<ONNXMulOp>(loc, matrixType, ft, Ct);
  Value itct = rewriter.create<ONNXMulOp>(loc, matrixType, it, ct);
  Value nextCt = rewriter.create<ONNXAddOp>(loc, matrixType, ftCt, itct);

  // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
  Value XtWo =
      rewriter.create<ONNXMatMulOp>(loc, matrixType, Xt, weightPack.Wo);
  Value HtRo =
      rewriter.create<ONNXMatMulOp>(loc, matrixType, Ht, weightPack.Ro);
  Value ot = rewriter.create<ONNXAddOp>(loc, matrixType, XtWo, HtRo);
  if (hasBiasForInput) {
    ot = rewriter.create<ONNXAddOp>(loc, matrixType, ot, biasPack.Wbo);
    ot = rewriter.create<ONNXAddOp>(loc, matrixType, ot, biasPack.Rbo);
  }
  if (hasPeepholes) {
    Value PoCt =
        rewriter.create<ONNXMulOp>(loc, matrixType, biasPack.Po, nextCt);
    ot = rewriter.create<ONNXAddOp>(loc, matrixType, ot, PoCt);
  }
  ot = applyActivation(rewriter, loc, activationPack.f, ot);

  // Ht = ot (.) h(Ct)
  Value nextHt = applyActivation(rewriter, loc, activationPack.h, nextCt);
  nextHt = rewriter.create<ONNXMulOp>(loc, matrixType, ot, nextHt);

  // Store the intermediate Ht if required.
  BuildKrnlLoop stateLoops(rewriter, loc, 2);
  stateLoops.createDefineOp();
  stateLoops.pushBounds(0, batchSizeVal);
  stateLoops.pushBounds(0, hiddenSizeVal);
  stateLoops.createIterateOp();
  {
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(stateLoops.getIterateBlock());

    auto batchIV = stateLoops.getInductionVar(0);
    auto hiddenIV = stateLoops.getInductionVar(1);

    Value val = krnl_load(nextHt, ArrayRef<Value>{batchIV, hiddenIV});
    krnl_store(val, state.ht, ArrayRef<Value>{directionIV, batchIV, hiddenIV});

    if (!isNoneType(state.allH)) {
      SmallVector<Value, 4> allHIVs{sequenceIV, directionIV, batchIV, hiddenIV};
      krnl_store(val, state.allH, allHIVs);
    }
  }
}

template <>
void stateToOutput<ONNXLSTMOp, LstmState>(
    ONNXLSTMOp *op, LstmState state, std::vector<Value> &outputs) {
  Value noneValue;
  outputs.emplace_back((isNoneType(op->Y()) ? noneValue : state.allH));
  outputs.emplace_back((isNoneType(op->Y_h()) ? noneValue : state.ht));
  outputs.emplace_back((isNoneType(op->Y_c()) ? noneValue : state.ct));
}

void populateLoweringONNXLSTMOpPattern(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  patterns.insert<ONNXRNNOpLowering<ONNXLSTMOp, LstmState, LstmActivationPack,
      LstmWeightPack, LstmBiasPack>>(ctx);
}
