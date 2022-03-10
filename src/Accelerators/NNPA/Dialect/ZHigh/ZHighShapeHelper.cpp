/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------ZHighShapeHelper.cpp - help for shapes ---------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains shape computation for ZHigh operations.
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighShapeHelper.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace zhigh {

//===----------------------------------------------------------------------===//
// ZHigh Op Shape Helper
// Same as ONNXOpsShapeHelper
//===----------------------------------------------------------------------===//

// Reuse scope if given, otherwise create one now and free in destructor.
template <class OP>
ZHighOpShapeHelper<OP>::ZHighOpShapeHelper(
    OP *newOp, int numResults, IndexExprScope *inScope)
    : op(newOp), outputsDims(), ownScope(inScope == nullptr) {
  assert(op && "Expecting a valid pointer");
  if (ownScope)
    scope = new IndexExprScope(nullptr, newOp->getLoc());
  setNumberOfOutputs(numResults);
}

template <class OP>
ZHighOpShapeHelper<OP>::ZHighOpShapeHelper(OP *newOp, int numResults,
    OpBuilder *rewriter, ArrayValueIndexCapture::GetDenseVal fGetDenseValInput,
    ArrayValueIndexCapture::LoadVal fLoadVal, IndexExprScope *inScope)
    : op(newOp), fGetDenseVal(fGetDenseValInput), fLoadVal(fLoadVal),
      outputsDims(), ownScope(inScope == nullptr) {
  assert(op && "Expecting a valid pointer");
  if (ownScope)
    scope = new IndexExprScope(rewriter, newOp->getLoc());
  setNumberOfOutputs(numResults);
  // Get the dense value by combining provided function (if any) with the
  // default one.
  fGetDenseVal = [=](Value array) {
    DenseElementsAttr res = nullptr;
    // Try with the provided method, if any.
    if (fGetDenseValInput)
      res = fGetDenseValInput(array);
    // If provided method was not provided or failed, try default ONNX method.
    if (!res)
      res = getDenseElementAttributeFromONNXValue(array);
    return res;
  };
}

//===----------------------------------------------------------------------===//
// StickForGRUOp
//===----------------------------------------------------------------------===//

ZHighStickForGRUOpShapeHelper::ZHighStickForGRUOpShapeHelper(
    ZHighStickForGRUOp *newOp)
    : ZHighOpShapeHelper<ZHighStickForGRUOp>(
          newOp, newOp->getOperation()->getNumResults()) {}

ZHighStickForGRUOpShapeHelper::ZHighStickForGRUOpShapeHelper(
    ZHighStickForGRUOp *newOp, OpBuilder *rewriter)
    : ZHighOpShapeHelper<ZHighStickForGRUOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, nullptr, nullptr) {}

LogicalResult ZHighStickForGRUOpShapeHelper::computeShape(
    ZHighStickForGRUOpAdaptor operandAdaptor) {
  // Output dims of result.
  DimsExpr outputDims;

  // Get operands and bounds.
  Value zGate = operandAdaptor.z_gate();
  MemRefBoundsIndexCapture zBounds(zGate);
  int64_t rank = zBounds.getRank();

  for (int64_t i = 0; i < rank - 1; ++i)
    outputDims.emplace_back(zBounds.getDim(i));
  IndexExpr lastDim = zBounds.getDim(rank - 1) * LiteralIndexExpr(3);
  outputDims.emplace_back(lastDim);

  // Save the final result.
  dimsForOutput(0) = outputDims;
  return success();
}

//===----------------------------------------------------------------------===//
// StickForLSTMOp
//===----------------------------------------------------------------------===//

ZHighStickForLSTMOpShapeHelper::ZHighStickForLSTMOpShapeHelper(
    ZHighStickForLSTMOp *newOp)
    : ZHighOpShapeHelper<ZHighStickForLSTMOp>(
          newOp, newOp->getOperation()->getNumResults()) {}

ZHighStickForLSTMOpShapeHelper::ZHighStickForLSTMOpShapeHelper(
    ZHighStickForLSTMOp *newOp, OpBuilder *rewriter)
    : ZHighOpShapeHelper<ZHighStickForLSTMOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, nullptr, nullptr) {}

LogicalResult ZHighStickForLSTMOpShapeHelper::computeShape(
    ZHighStickForLSTMOpAdaptor operandAdaptor) {
  // Output dims of result.
  DimsExpr outputDims;

  // Get operands and bounds.
  Value fGate = operandAdaptor.f_gate();
  MemRefBoundsIndexCapture fBounds(fGate);
  int64_t rank = fBounds.getRank();

  for (int64_t i = 0; i < rank - 1; ++i)
    outputDims.emplace_back(fBounds.getDim(i));
  IndexExpr lastDim = fBounds.getDim(rank - 1) * LiteralIndexExpr(4);
  outputDims.emplace_back(lastDim);

  // Save the final result.
  dimsForOutput(0) = outputDims;
  return success();
}

//===----------------------------------------------------------------------===//
// MatMulOp
//===----------------------------------------------------------------------===//

ZHighMatMulOpShapeHelper::ZHighMatMulOpShapeHelper(ZHighMatMulOp *newOp)
    : ZHighOpShapeHelper<ZHighMatMulOp>(
          newOp, newOp->getOperation()->getNumResults()) {}

ZHighMatMulOpShapeHelper::ZHighMatMulOpShapeHelper(
    ZHighMatMulOp *newOp, OpBuilder *rewriter)
    : ZHighOpShapeHelper<ZHighMatMulOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, nullptr, nullptr) {}

LogicalResult ZHighMatMulOpShapeHelper::computeShape(
    ZHighMatMulOpAdaptor operandAdaptor) {
  // Output dims of result.
  DimsExpr outputDims;

  // Get operands.
  Value X = operandAdaptor.X();
  Value Y = operandAdaptor.Y();

  // Get bounds
  MemRefBoundsIndexCapture XBounds(X);
  MemRefBoundsIndexCapture YBounds(Y);
  int64_t xRank = XBounds.getRank();
  int64_t yRank = YBounds.getRank();

  if (!(xRank == 2 || xRank == 3))
    return failure();

  if (xRank == 2) {
    // X :: MxN
    // Y :: NxP
    outputDims.emplace_back(XBounds.getDim(0));
    outputDims.emplace_back(YBounds.getDim(1));
  } else if (xRank == 3) {
    // X :: SxMxN
    outputDims.emplace_back(XBounds.getDim(0));
    outputDims.emplace_back(XBounds.getDim(1));
    if (yRank == 2) {
      // Y :: NxP
      outputDims.emplace_back(YBounds.getDim(1));
      isBroadcasted = true;
    } else if (yRank == 3) {
      // Y :: SxNxP
      outputDims.emplace_back(YBounds.getDim(2));
      isStacked = true;
    }
  }

  // Keep all original dimensions: M, N, P if 2D or S, M, N, P if 3D.
  if (xRank == 2) {
    // M
    allOriginalDims.emplace_back(XBounds.getDim(0));
    // N
    allOriginalDims.emplace_back(XBounds.getDim(1));
    // P
    allOriginalDims.emplace_back(YBounds.getDim(1));
  } else if (xRank == 3) {
    // S
    allOriginalDims.emplace_back(XBounds.getDim(0));
    // M
    allOriginalDims.emplace_back(XBounds.getDim(1));
    // N
    allOriginalDims.emplace_back(XBounds.getDim(2));
    // P
    if (yRank == 2)
      allOriginalDims.emplace_back(YBounds.getDim(1));
    else if (yRank == 3)
      allOriginalDims.emplace_back(YBounds.getDim(2));
  }

  // Save the final result.
  dimsForOutput(0) = outputDims;
  return success();
}

//===----------------------------------------------------------------------===//
// LSTMOp
//===----------------------------------------------------------------------===//

ZHighLSTMOpShapeHelper::ZHighLSTMOpShapeHelper(ZHighLSTMOp *newOp)
    : ZHighOpShapeHelper<ZHighLSTMOp>(
          newOp, newOp->getOperation()->getNumResults()) {}

ZHighLSTMOpShapeHelper::ZHighLSTMOpShapeHelper(
    ZHighLSTMOp *newOp, OpBuilder *rewriter)
    : ZHighOpShapeHelper<ZHighLSTMOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, nullptr, nullptr) {}

LogicalResult ZHighLSTMOpShapeHelper::computeShape(
    ZHighLSTMOpAdaptor operandAdaptor) {
  // Get operands.
  // X: [S, B, I]
  Value X = operandAdaptor.input();
  // R: [D, H, H]
  Value R = operandAdaptor.hidden_weights();

  // Return all timesteps or only the final step;
  bool isAllTimesteps = (op->return_all_steps() == -1) ? true : false;

  // Get bounds
  MemRefBoundsIndexCapture XBounds(X);
  MemRefBoundsIndexCapture RBounds(R);
  IndexExpr S = XBounds.getDim(0);
  IndexExpr B = XBounds.getDim(1);
  IndexExpr I = XBounds.getDim(2);
  IndexExpr D = RBounds.getDim(0);
  IndexExpr H = RBounds.getDim(1);

  // Shape for hn_ouput : [S, D, B, H] if return all  timesteps. [1, D, B, H] if
  // return the final step only.
  DimsExpr hnOutputDims;
  if (isAllTimesteps)
    hnOutputDims.emplace_back(S);
  else
    hnOutputDims.emplace_back(LiteralIndexExpr(1));
  hnOutputDims.emplace_back(D);
  hnOutputDims.emplace_back(B);
  hnOutputDims.emplace_back(H);

  // Shape for cf_ouput : [1, D, B, H]
  DimsExpr cfOutputDims;
  cfOutputDims.emplace_back(LiteralIndexExpr(1));
  cfOutputDims.emplace_back(D);
  cfOutputDims.emplace_back(B);
  cfOutputDims.emplace_back(H);

  // Shape for optional values.
  // Initialized value: [D, B, H]
  hc0Shape.emplace_back(D);
  hc0Shape.emplace_back(B);
  hc0Shape.emplace_back(H);
  // Bias value: [D, 4*H]
  biasShape.emplace_back(D);
  biasShape.emplace_back(H * 4);

  // Keep all original dimensions.
  allOriginalDims.emplace_back(D);
  allOriginalDims.emplace_back(S);
  allOriginalDims.emplace_back(B);
  allOriginalDims.emplace_back(I);
  allOriginalDims.emplace_back(H);

  // Save the final results.
  dimsForOutput(0) = hnOutputDims;
  dimsForOutput(1) = cfOutputDims;
  return success();
}

//===----------------------------------------------------------------------===//
// GRUOp
//===----------------------------------------------------------------------===//

ZHighGRUOpShapeHelper::ZHighGRUOpShapeHelper(ZHighGRUOp *newOp)
    : ZHighOpShapeHelper<ZHighGRUOp>(
          newOp, newOp->getOperation()->getNumResults()) {}

ZHighGRUOpShapeHelper::ZHighGRUOpShapeHelper(
    ZHighGRUOp *newOp, OpBuilder *rewriter)
    : ZHighOpShapeHelper<ZHighGRUOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, nullptr, nullptr) {}

LogicalResult ZHighGRUOpShapeHelper::computeShape(
    ZHighGRUOpAdaptor operandAdaptor) {
  // Get operands.
  // X: [S, B, I]
  Value X = operandAdaptor.input();
  // R: [D, H, H]
  Value R = operandAdaptor.hidden_weights();

  // Return all timesteps or only the final step;
  bool isAllTimesteps = (op->return_all_steps() == -1) ? true : false;

  // Get bounds
  MemRefBoundsIndexCapture XBounds(X);
  MemRefBoundsIndexCapture RBounds(R);
  IndexExpr S = XBounds.getDim(0);
  IndexExpr B = XBounds.getDim(1);
  IndexExpr I = XBounds.getDim(2);
  IndexExpr D = RBounds.getDim(0);
  IndexExpr H = RBounds.getDim(1);

  // Shape for hn_ouput : [S, D, B, H] if return all  timesteps. [1, D, B, H] if
  // return the final step only.
  DimsExpr hnOutputDims;
  if (isAllTimesteps)
    hnOutputDims.emplace_back(S);
  else
    hnOutputDims.emplace_back(LiteralIndexExpr(1));
  hnOutputDims.emplace_back(D);
  hnOutputDims.emplace_back(B);
  hnOutputDims.emplace_back(H);

  // Shape for cf_ouput : [1, B, H]
  DimsExpr cfOutputDims;
  cfOutputDims.emplace_back(LiteralIndexExpr(1));
  cfOutputDims.emplace_back(B);
  cfOutputDims.emplace_back(H);

  // Shape for optional values.
  // Initialized value: [D, B, H]
  h0Shape.emplace_back(D);
  h0Shape.emplace_back(B);
  h0Shape.emplace_back(H);
  // Bias value: [D, 3*H]
  biasShape.emplace_back(D);
  biasShape.emplace_back(H * 3);

  // Keep all original dimensions.
  allOriginalDims.emplace_back(D);
  allOriginalDims.emplace_back(S);
  allOriginalDims.emplace_back(B);
  allOriginalDims.emplace_back(I);
  allOriginalDims.emplace_back(H);

  // Save the final results.
  dimsForOutput(0) = hnOutputDims;
  return success();
}

//===----------------------------------------------------------------------===//
// Conv2DOp
//===----------------------------------------------------------------------===//

ZHighConv2DOpShapeHelper::ZHighConv2DOpShapeHelper(ZHighConv2DOp *newOp)
    : ZHighOpShapeHelper<ZHighConv2DOp>(
          newOp, newOp->getOperation()->getNumResults()) {}

ZHighConv2DOpShapeHelper::ZHighConv2DOpShapeHelper(
    ZHighConv2DOp *newOp, OpBuilder *rewriter)
    : ZHighOpShapeHelper<ZHighConv2DOp>(newOp,
          newOp->getOperation()->getNumResults(), rewriter, nullptr, nullptr) {}

LogicalResult ZHighConv2DOpShapeHelper::computeShape(
    ZHighConv2DOpAdaptor operandAdaptor) {
  // Get operands.
  // X: [B, HI, WI, CI]
  Value X = operandAdaptor.input();
  // W: [KH, KW, CI, CO]
  Value W = operandAdaptor.input_kernel();
  // Get attributes.
  ArrayAttr strides = op->strides();
  StringRef paddingType = op->padding_type();

  // Get bounds
  MemRefBoundsIndexCapture XBounds(X);
  MemRefBoundsIndexCapture WBounds(W);
  IndexExpr B = XBounds.getDim(0);
  IndexExpr HI = XBounds.getDim(1);
  IndexExpr WI = XBounds.getDim(2);
  IndexExpr CI = XBounds.getDim(3);
  IndexExpr KH = WBounds.getDim(0);
  IndexExpr KW = WBounds.getDim(1);
  IndexExpr CO = WBounds.getDim(3);
  IndexExpr strideH = LiteralIndexExpr(strides[0].cast<IntegerAttr>().getInt());
  IndexExpr strideW = LiteralIndexExpr(strides[1].cast<IntegerAttr>().getInt());

  // Compute output height and weight.
  IndexExpr HO, WO;
  if (paddingType.equals_insensitive("SAME_PADDING")) {
    HO = HI.ceilDiv(strideH);
    WO = WI.ceilDiv(strideW);
  } else if (paddingType.equals_insensitive("VALID_PADDING")) {
    IndexExpr newHI = HI - KH + 1;
    IndexExpr newWI = WI - KW + 1;
    HO = newHI.ceilDiv(strideH);
    WO = newWI.ceilDiv(strideW);
  } else {
    llvm_unreachable("Unsupported padding_type");
  }

  // Output shape: [B, HO, WO, CO]
  DimsExpr outputDims;
  outputDims.emplace_back(B);
  outputDims.emplace_back(HO);
  outputDims.emplace_back(WO);
  outputDims.emplace_back(CO);

  // Keep all original dimensions.
  allOriginalDims.emplace_back(B);
  allOriginalDims.emplace_back(CI);
  allOriginalDims.emplace_back(HI);
  allOriginalDims.emplace_back(WI);
  allOriginalDims.emplace_back(CO);
  allOriginalDims.emplace_back(HO);
  allOriginalDims.emplace_back(WO);

  // Save the final results.
  dimsForOutput(0) = outputDims;
  return success();
}

//===----------------------------------------------------------------------===//
// PoolingOp
//===----------------------------------------------------------------------===//

template <typename OP, typename OP_ADAPTOR>
ZHighPoolingOpShapeHelper<OP, OP_ADAPTOR>::ZHighPoolingOpShapeHelper(OP *newOp)
    : ZHighOpShapeHelper<OP>(newOp, newOp->getOperation()->getNumResults()) {}

template <typename OP, typename OP_ADAPTOR>
ZHighPoolingOpShapeHelper<OP, OP_ADAPTOR>::ZHighPoolingOpShapeHelper(
    OP *newOp, OpBuilder *rewriter)
    : ZHighOpShapeHelper<OP>(newOp, newOp->getOperation()->getNumResults(),
          rewriter, nullptr, nullptr) {}

template <typename OP, typename OP_ADAPTOR>
LogicalResult ZHighPoolingOpShapeHelper<OP, OP_ADAPTOR>::computeShape(
    OP_ADAPTOR operandAdaptor) {
  // Get operands.
  // X: [B, HI, WI, CI]
  Value X = operandAdaptor.input();
  // Get attributes.
  ArrayAttr kernelShape = ZHighOpShapeHelper<OP>::op->kernel_shape();
  ArrayAttr strides = ZHighOpShapeHelper<OP>::op->strides();
  StringRef paddingType = ZHighOpShapeHelper<OP>::op->padding_type();

  // Get bounds
  MemRefBoundsIndexCapture XBounds(X);
  IndexExpr B = XBounds.getDim(0);
  IndexExpr HI = XBounds.getDim(1);
  IndexExpr WI = XBounds.getDim(2);
  IndexExpr CI = XBounds.getDim(3);
  IndexExpr KH = LiteralIndexExpr(kernelShape[0].cast<IntegerAttr>().getInt());
  IndexExpr KW = LiteralIndexExpr(kernelShape[1].cast<IntegerAttr>().getInt());
  IndexExpr strideH = LiteralIndexExpr(strides[0].cast<IntegerAttr>().getInt());
  IndexExpr strideW = LiteralIndexExpr(strides[1].cast<IntegerAttr>().getInt());

  // Compute output height and weight.
  IndexExpr HO, WO;
  if (paddingType.equals_insensitive("SAME_PADDING")) {
    HO = HI.ceilDiv(strideH);
    WO = WI.ceilDiv(strideW);
  } else if (paddingType.equals_insensitive("VALID_PADDING")) {
    IndexExpr newHI = HI - KH + 1;
    IndexExpr newWI = WI - KW + 1;
    HO = newHI.ceilDiv(strideH);
    WO = newWI.ceilDiv(strideW);
  } else {
    llvm_unreachable("Unsupported padding_type");
  }

  // Output shape: [B, HO, WO, CI]
  DimsExpr outputDims;
  outputDims.emplace_back(B);
  outputDims.emplace_back(HO);
  outputDims.emplace_back(WO);
  outputDims.emplace_back(CI);

  // Keep all original dimensions.
  allOriginalDims.emplace_back(B);
  allOriginalDims.emplace_back(CI);
  allOriginalDims.emplace_back(HI);
  allOriginalDims.emplace_back(WI);
  allOriginalDims.emplace_back(HO);
  allOriginalDims.emplace_back(WO);

  // Save the final results.
  ZHighOpShapeHelper<OP>::dimsForOutput(0) = outputDims;
  return success();
}

//===----------------------------------------------------------------------===//
// ZHigh Shape Helper template instantiation
// Keep template instantiation at the end of the file.
//===----------------------------------------------------------------------===//

template struct ZHighPoolingOpShapeHelper<ZHighMaxPool2DOp,
    ZHighMaxPool2DOpAdaptor>;
template struct ZHighPoolingOpShapeHelper<ZHighAvgPool2DOp,
    ZHighAvgPool2DOpAdaptor>;

} // namespace zhigh
} // namespace onnx_mlir