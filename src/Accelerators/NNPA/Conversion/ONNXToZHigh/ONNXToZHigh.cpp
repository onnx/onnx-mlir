/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ONNXToZHigh.cpp - ONNX dialect to ZHigh lowering -------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the lowering of ONNX operations to a combination of
// ONNX and ZHigh operations.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Debug.h"

#include "src/Accelerators/NNPA/Compiler/NNPACompilerOptions.hpp"
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXToZHigh.hpp"
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXToZHighCommon.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/OpHelper.hpp"
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
#include "src/Conversion/ONNXToKrnl/RNN/RNNBase.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "src/Dialect/ONNX/Transforms/ShapeInference.hpp"

#define DEBUG_TYPE "onnx-to-zhigh"

using namespace mlir;

//
// LSTM/GRU specific functions
//

namespace onnx_mlir {

using namespace zhigh;

#define QUANT_PATTERN_BENEFIT 1000

/// Checks whether a constant tensor's elements are of type FloatType.
bool isFloatType(Value constValue) {
  ElementsAttr constElements = getElementAttributeFromONNXValue(constValue);
  Type elemType = constElements.getElementType();
  return mlir::isa<FloatType>(elemType);
}

ArrayAttr getLSTMGRUBiasSplitShape(
    Location loc, PatternRewriter &rewriter, ArrayRef<int64_t> shapeR) {
  int64_t hiddenSize = shapeR[2];
  int splitNum = (shapeR[1] / shapeR[2]) * 2;
  std::vector<int64_t> splitShape;
  for (int i = 0; i < splitNum; i++) {
    splitShape.emplace_back(hiddenSize);
  }
  return rewriter.getI64ArrayAttr(splitShape);
}

Value getLSTMGRUZDNNWeightFromONNXWeight(
    Location loc, PatternRewriter &rewriter, Value weight, int isLSTM) {
  int64_t splitNum = isLSTM ? 4 : 3;
  RankedTensorType weightType = mlir::cast<RankedTensorType>(weight.getType());
  Type elementType = weightType.getElementType();
  ArrayRef<int64_t> weightShape = weightType.getShape();
  int64_t direction = weightShape[0];
  int64_t hiddenSize = weightShape[1] / splitNum;
  int64_t weightHiddenSize = weightShape[1];
  int64_t feature = weightShape[2];
  SmallVector<int64_t, 3> transposeShape;
  transposeShape.emplace_back(direction);
  transposeShape.emplace_back(feature);
  transposeShape.emplace_back(weightHiddenSize);
  RankedTensorType transposeType =
      RankedTensorType::get(transposeShape, elementType);
  SmallVector<int64_t, 3> perms({0, 2, 1});
  ArrayRef<int64_t> permArrayW(perms);
  ArrayAttr permAttrW = rewriter.getI64ArrayAttr(permArrayW);
  Value transposeOp =
      rewriter.create<ONNXTransposeOp>(loc, transposeType, weight, permAttrW);
  SmallVector<int64_t, 3> splitShape;
  splitShape.emplace_back(direction);
  splitShape.emplace_back(feature);
  splitShape.emplace_back(hiddenSize);
  Type splitType = RankedTensorType::get(splitShape, elementType);
  int64_t axis = 2;
  Value stickForOp;
  if (isLSTM) {
    SmallVector<Type, 4> splitTypes(splitNum, splitType);
    ONNXSplitV11Op splitOp = rewriter.create<ONNXSplitV11Op>(
        loc, splitTypes, transposeOp, axis, nullptr);
    Value i_gate = splitOp.getResults()[0];
    Value o_gate = splitOp.getResults()[1];
    Value f_gate = splitOp.getResults()[2];
    Value c_gate = splitOp.getResults()[3];
    stickForOp = rewriter.create<zhigh::ZHighStickForLSTMOp>(
        loc, f_gate, i_gate, c_gate, o_gate);
  } else { // GRU
    SmallVector<Type, 3> splitTypes(splitNum, splitType);
    ONNXSplitV11Op splitOp = rewriter.create<ONNXSplitV11Op>(
        loc, splitTypes, transposeOp, axis, nullptr);
    Value z_gate = splitOp.getResults()[0];
    Value r_gate = splitOp.getResults()[1];
    Value h_gate = splitOp.getResults()[2];
    stickForOp =
        rewriter.create<zhigh::ZHighStickForGRUOp>(loc, z_gate, r_gate, h_gate);
  }
  return stickForOp;
}

Value getLSTMGRUGetY(
    Location loc, PatternRewriter &rewriter, Value val, Value resY) {
  Value noneValue;
  if (isNoneValue(resY)) {
    return noneValue;
  }
  return val;
}

Value getLSTMGRUGetYWithSequenceLens(Location loc, PatternRewriter &rewriter,
    Value val, Value resY, Value sequenceLens, Value initialH) {

  Value noneValue;
  if (isNoneValue(resY)) {
    return noneValue;
  }

  if (isNoneValue(sequenceLens))
    return getLSTMGRUGetY(loc, rewriter, val, resY);

  std::vector<Value> inputs = {val, sequenceLens, initialH};
  return rewriter.create<zhigh::ZHighFixGRUYOp>(loc, resY.getType(), inputs);
}

Value getLSTMGRUGetYh(Location loc, PatternRewriter &rewriter, Value val,
    Value resY, Value resYh, Value X, StringAttr direction) {
  Value noneValue;
  if (isNoneValue(resYh) || isNoneValue(val))
    return noneValue;

  ArrayRef<int64_t> shapeX = mlir::cast<ShapedType>(X.getType()).getShape();
  MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);
  // Generate Y_h for onnx.LSTM from hn_output for all timestep
  Value minusOne = create.onnx.constantInt64({-1});
  Value zero = create.onnx.constantInt64({0});
  Value one = create.onnx.constantInt64({1});
  // Use INT_MAX to get timestep dimension because timestep is the end of a
  // dimension. INT_MAX is recommended in ONNX.Slice to slice to the end of a
  // dimension with unknown size.
  Value intMax = create.onnx.constantInt64({INT_MAX});
  StringRef directionStr = direction.getValue();
  ArrayRef<int64_t> resYhShape =
      mlir::cast<RankedTensorType>(resYh.getType()).getShape();
  int64_t T = isNoneValue(resY) ? 1 : shapeX[0];
  int64_t D = resYhShape[0];
  int64_t B = resYhShape[1];
  int64_t H = resYhShape[2];
  Type elementType = mlir::cast<ShapedType>(resYh.getType()).getElementType();
  Value axis = zero;
  Value step = one;
  Value ret;
  if (directionStr.equals_insensitive("forward") ||
      directionStr.equals_insensitive("reverse")) {
    Value start = directionStr.equals_insensitive("forward") ? minusOne : zero;
    Value end = directionStr.equals_insensitive("forward") ? intMax : one;

    Type sliceType = RankedTensorType::get({1, D, B, H}, elementType);
    ONNXSliceOp sliceOp = rewriter.create<ONNXSliceOp>(
        loc, sliceType, val, start, end, axis, step);
    return rewriter.create<ONNXSqueezeV11Op>(
        loc, resYh.getType(), sliceOp.getResult(), rewriter.getI64ArrayAttr(0));
  } else if (directionStr.equals_insensitive("bidirectional")) {
    Type splitType = RankedTensorType::get({T, 1, B, H}, elementType);
    SmallVector<Type> splitTypes = {splitType, splitType};
    ONNXSplitV11Op splitOp = rewriter.create<ONNXSplitV11Op>(
        loc, splitTypes, val, /*splitAxis=*/1, nullptr);
    Type sliceType = RankedTensorType::get({1, 1, B, H}, elementType);
    Value fwdLastSlice = rewriter.create<ONNXSliceOp>(
        loc, sliceType, splitOp.getResults()[0], minusOne, intMax, axis, step);
    Value bkwFirstSlice = rewriter.create<ONNXSliceOp>(
        loc, sliceType, splitOp.getResults()[1], zero, one, axis, step);
    Type concatType = RankedTensorType::get({1, D, B, H}, elementType);
    Value concatOp = rewriter.create<ONNXConcatOp>(loc, concatType,
        ValueRange({fwdLastSlice, bkwFirstSlice}), /*concatAxis=*/1);
    Type squeezeType = RankedTensorType::get({D, B, H}, elementType);
    return rewriter.create<ONNXSqueezeV11Op>(
        loc, squeezeType, concatOp, rewriter.getI64ArrayAttr(0));
  } else {
    llvm_unreachable("Invalid direction.");
  }
  return ret;
}

Value getLSTMGRUGetYhWithSequenceLens(Location loc, PatternRewriter &rewriter,
    Value val, Value resY, Value resYh, Value X, StringAttr direction,
    Value sequenceLens) {
  Value noneValue;
  if (isNoneValue(resYh) || isNoneValue(val))
    return noneValue;

  if (isNoneValue(sequenceLens))
    return getLSTMGRUGetYh(loc, rewriter, val, resY, resYh, X, direction);

  std::vector<Value> inputs = {val, sequenceLens};
  return rewriter.create<zhigh::ZHighFixGRUYhOp>(loc, resYh.getType(), inputs);
}

Value getLSTMGRUGetYc(
    Location loc, PatternRewriter &rewriter, Value val, Value resYc) {
  Value noneValue;
  if (isNoneValue(resYc))
    return noneValue;

  zhigh::ZHighUnstickOp unstickOp =
      rewriter.create<zhigh::ZHighUnstickOp>(loc, val);
  return rewriter.create<ONNXSqueezeV11Op>(
      loc, resYc.getType(), unstickOp.getResult(), rewriter.getI64ArrayAttr(0));
}

SmallVector<Value, 4> emitONNXSplitOp(Location loc, PatternRewriter &rewriter,
    Value input, IntegerAttr axis, ArrayAttr split) {
  Type elementType = mlir::cast<ShapedType>(input.getType()).getElementType();
  SmallVector<Type> outputTypes;
  int64_t splitNum = split.size();
  ArrayRef<int64_t> inputShape =
      mlir::cast<RankedTensorType>(input.getType()).getShape();
  int64_t splitAxis = mlir::cast<IntegerAttr>(axis).getSInt();
  assert(splitAxis >= 0 && "Negative axis");
  for (int i = 0; i < splitNum; i++) {
    SmallVector<int64_t> outputShape;
    for (size_t dim = 0; dim < inputShape.size(); dim++) {
      outputShape.emplace_back(
          (dim == (unsigned int)splitAxis)
              ? mlir::cast<IntegerAttr>(split[dim]).getInt()
              : inputShape[dim]);
    }
    outputTypes.emplace_back(RankedTensorType::get(outputShape, elementType));
  }
  ONNXSplitV11Op splitOp =
      rewriter.create<ONNXSplitV11Op>(loc, outputTypes, input, axis, split);
  return splitOp.getResults();
}

/// Get kernelShapes using shape helper
template <typename OP, typename OPAdaptor, typename OPShapeHelper>
SmallVector<int64_t, 2> getArrayKernelShape(OP op) {
  OPShapeHelper shapeHelper(op.getOperation(), {});
  shapeHelper.computeShapeAndAssertOnFailure();

  // Check if kernelShape is literal. Only static value is supported.
  assert((llvm::any_of(shapeHelper.kernelShape, [](IndexExpr val) {
    return val.isLiteral();
  })) && "Only support static kernel_shape ");

  SmallVector<int64_t, 2> kernelShapesRet;
  kernelShapesRet.emplace_back(shapeHelper.kernelShape[0].getLiteral());
  kernelShapesRet.emplace_back(shapeHelper.kernelShape[1].getLiteral());
  return kernelShapesRet;
}

/// Get strides using shape helper
template <typename OP, typename OPAdaptor, typename OPShapeHelper>
SmallVector<int64_t, 2> getArrayStrides(OP op) {
  OPShapeHelper shapeHelper(op.getOperation(), {});
  shapeHelper.computeShapeAndAssertOnFailure();
  return shapeHelper.strides;
}

/// Get approximate
template <typename OP, typename OPAdaptor, typename OPShapeHelper>
StringRef getStrApproximateType(OP op) {
  return op.getApproximate();
}

// Computes the folded bias to be passed to quantized matmul call when
// operation is MATMUL_OP_ADDITION. Zb should be equal to 0, meaning the
// correction term for input_a is also equal to 0. This allows the
// correction term for input_b to be folded into qc_tilde, which removes the
// need for correction being applied after the quantized matmul call.
//
// The original equation for qc_tilde is:
//   M = (Sa * Sb) / Sy
//   qc_tilde = Zy - (Sc / Sy) * Zc + (Sc / Sy) * input_c[j] + M*N*Za*Zb
//
// Given Zb = 0, the equation becomes:
//   M = (Sa * Sb) / Sy
//   qc_tilde = Zy - (Sc / Sy) * Zc + (Sc / Sy) * input_c[j]
//
// Given scales are stored as the reciprocal in zTensor, the modified equation
// becomes:
//   M = RSy / (RSa * RSb)
//   qc_tilde = Zy - (RSy / RSc) * Zc + (RSy / RSc) * input_c[j]
//
//  where RS = 1/S.
//
// We can reorder this to:
//   M = RSy / (RSa * RSb)
//   qc_tilde = input_c[j] * (RSy / RSc) + Zy - (RSy / RSc) * Zc
//
// This allows us to pre-compute a scale and offset to apply to input_c[j]:
//   M = RSy / (RSa * RSb).
//   scale = (RSy / RSc)
//   offset = Zy - scale * Zc
//   qc_tilde[j] = input_c[j] * scale + offset
//
// The original equation for the correction term for input_b is:
//   M = (RSa * RSb) / RSy
//   term_b = M * Za * sum(input_b[:,j])
//
// Given scales are stored as the reciprocal, the modified equation becomes:
//   M = RSy / (RSa * RSb)
//   term_b = M * Za * sum(input_b[:,j])
//
// This gives us the equation:
//   M = RSy / (RSa * RSb)
//   MZa = M * Za
//   scale = (RSy / RSc)
//   offset = Zy - scale * Zc
//   qc_tilde[j] = input_c[j] * scale + offset - MZa * sum(input_b[:,j])
//
// In case of MatMulInteger, input_c = 0, RSc = 1, Zc = 0, the final equation
// is:
//   M = RSy / (RSa * RSb)
//   MZa = M * Za
//   scale = RSy
//   offset = Zy
//   qc_tilde[j] = offset - Za * (RSy / RSa / RSb) * sum(input_b[:,j])
//
// When Zy = 0, qc_tilde[j] = -Za * (RSy / RSa / RSb) * sum(input_b[:,j])
static void preComputeBias(MultiDialectBuilder<OnnxBuilder> &create, Value RSa,
    Value Za, Value BI8, Value RSb, Value RSy, Value Zy, Value &qcTilde,
    Value &RSqctilde, Value &Zqctilde) {
  OpBuilder rewriter = create.getBuilder();
  Location loc = create.getLoc();

  Type i64Ty = rewriter.getI64Type();
  Type f32Ty = rewriter.getF32Type();
  auto cstMinus2Attr = DenseElementsAttr::get(
      RankedTensorType::get({}, i64Ty), static_cast<int64_t>(-2));
  auto cst0Attr = DenseElementsAttr::get(
      RankedTensorType::get({}, f32Ty), static_cast<float>(0));
  auto cst1Attr = DenseElementsAttr::get(
      RankedTensorType::get({}, f32Ty), static_cast<float>(1));

  Value cst0 = create.onnx.constant(cst0Attr);
  Value cst1 = create.onnx.constant(cst1Attr);

  // Can be optimized further when Zy is zero.
  bool ZyIsZero = isDenseONNXConstant(Zy) && isConstOf(Zy, 0.);

  Value qcF32;
  Value B = create.onnx.cast(BI8, f32Ty);
  Value lastSecondAxis = create.onnx.constant(cstMinus2Attr);
  // Emit: sum(input_b[:,j])
  Value BSum = create.onnx.reduceSum(
      UnrankedTensorType::get(f32Ty), B, lastSecondAxis, false);
  // RSy, RSa, RSb, Za are scalar, do scalar computation.
  // Emit: Za * (RSy / RSa / RSb)
  Value RSyRSa = create.onnx.div(RSy, RSa);
  Value RSyRSaRSb = create.onnx.div(RSyRSa, RSb);
  Value MZa = create.onnx.mul(RSyRSaRSb, Za);
  // Negate ZaRSyRSa to avoid broadcasting Sub:
  // `Zy - Za * (RSy / RSa / RSb) * ...`
  MZa = create.onnx.sub(cst0, MZa);
  // Broadcast ops.
  // Emit: - Za * (RSy / RSa / RSb) * sum(input_b[:,j])
  Value MZaBSum = create.onnx.mul(MZa, BSum);
  // Emit: Zy - Za * (RSy / RSa / RSb) * sum(input_b[:,j])
  if (ZyIsZero) {
    qcF32 = MZaBSum;
  } else {
    qcF32 = create.onnx.add(Zy, MZaBSum);
  }

  // Use 1 for recscale and 0 for offset. This is a dlfloat16 stickification.
  int64_t rank = getRank(qcF32.getType());
  StringAttr layoutAttr =
      rewriter.getStringAttr((rank == 1) ? LAYOUT_1D : LAYOUT_2DS);
  ZHighQuantizedStickOp qcOp = rewriter.create<ZHighQuantizedStickOp>(loc,
      qcF32, cst1, cst0, layoutAttr, rewriter.getStringAttr(QTYPE_DLFLOAT16));
  qcTilde = qcOp.getResult(0);
  RSqctilde = qcOp.getResult(1);
  Zqctilde = qcOp.getResult(2);
}

static Value getOrCastToI8(Value val, MultiDialectBuilder<OnnxBuilder> &create,
    bool simpleCast = false) {
  if (!getElementType(val.getType()).isUnsignedInteger())
    return val;

  Type i8Ty = create.getBuilder().getI8Type();
  if (simpleCast)
    return create.onnx.cast(val, i8Ty);

  // Use int16 to avoid integer overflow.
  Type i16Ty = create.getBuilder().getI16Type();
  auto cst128Attr = DenseElementsAttr::get(
      RankedTensorType::get({}, i16Ty), static_cast<int16_t>(128));
  Value valI16 = create.onnx.cast(val, i16Ty);
  valI16 = create.onnx.sub(valI16, create.onnx.constant(cst128Attr));
  Value valI8 = create.onnx.cast(valI16, i8Ty);
  return valI8;
}

// Dynamic quantization helper to match and rewrite values A, B, C of A*B+C.
class DynQuantI8PatternHelper {
public:
  DynQuantI8PatternHelper(PatternRewriter &rewriter, Location loc,
      Operation *op, Value A, Value B, Value C, bool symForA)
      : rewriter(rewriter), loc(loc), op(op), A(A), B(B), C(C),
        symForA(symForA) {}

  // Check the inputs A, B, C of `A*B+C` to see if they are suitable for doing
  // dynamic quantization on NNPA.
  LogicalResult match() {
    // A is of f32.
    if (!mlir::isa<Float32Type>(getElementType(A.getType())))
      return rewriter.notifyMatchFailure(op, "MatMul's A is not of f32.");

    // Weight is a constant.
    if (!isDenseONNXConstant(B))
      return rewriter.notifyMatchFailure(op, "MatMul's B is not a constant.");

    if (C) {
      // Bias is a constant.
      if (!isDenseONNXConstant(C))
        return rewriter.notifyMatchFailure(op, "MatMul's C is not a constant");
      // B and C shapes must be consistent. The reduction shape of B on the
      // second dim from the last is the same as the shape of B, e.g. If B is
      // [2x3x4], C must be [2x4].
      ArrayRef<int64_t> bShape = getShape(B.getType());
      ArrayRef<int64_t> cShape = getShape(C.getType());
      int64_t bRank = bShape.size();
      int64_t cRank = cShape.size();
      if (bRank - 1 != cRank)
        return rewriter.notifyMatchFailure(
            op, "The ranks of B and C are imcompatible.");
      if (bShape[bRank - 1] != cShape[cRank - 1])
        return rewriter.notifyMatchFailure(
            op, "The last dimensions of B and C are not the same.");
      if (bShape.drop_back(2) != cShape.drop_back(1))
        return rewriter.notifyMatchFailure(
            op, "The shapes of B and C are imcompatible.");
    }

    return success();
  }

  // clang-format off
  /*
   * Emit the following code to compute `A*B+C` using i8 dynamic quantization.
   * A can be quantized using asymmetric or symmetric quantization depending on
   * the flag `symForA`, while B is always quantized using symmetric quantization.
   * (Note that: If C is given, it will be added into the pre_computed_bias)
   *
   * ```
   * (Quantize A using asymmetric/symmetric quant by setting `sym_mode` attr to the `symForA` flag)
   * %qa, %a_recscale, %a_offset = zhigh.QuantizedStick(%A, none, none) { quantized_type = QUANTIZED_DLFLOAT16, sym_mode = 1/0}
   *
   * (Quantize B using symmetric quant)
   * %b_offset = 0 // Symmetric quant mode for i8. Offset is always zero, qmin = * -127, qmax = 127.
   * %absmax = onnx.ReduceMax(onnx.Abs(%B))
   * %b_rescale = onnx.Div(127, absmax)
   * %qb = onnx.cast(onnx.Clip(onnx.Round(onnx.Mul(%B, %b_rescale)), qmin, qmax))
   * %qb, %b_recscale, %b_offset = zhigh.QuantizedStick(%qb, %b_recscale, %b_offset) { quantized_type = QUANTIZED_WEIGHTS_INT8 }
   *
   * (Pre computed bias, %C is added)
   * %qc = emit_ops_for_pre_computed_bias_at_compile_time
   * %qc = zhigh.Add(%qc, zhigh.Stick(%C)) // only done if C is given.
   * %qc_recscale = 1
   * %qc_offset = 0
   *
   * %Y_recscale = 1
   * %Y_offset = 0
   * %Y, %Y_recscale, %Y_offset = zhigh.QuantizedMatMul (%qa, %a_recscale, %a_offset,
   *                                                     %qb, %b_recscale, %b_offset,
   *                                                     %qc, %c_recscale, %c_offset,
   *                                                     %Y_recscale, %Y_offset) {
   *  PreComputedBias = true, DisableClipping = true, DequantizeOutput = false
   * }
   * ```
   *
   * where the computation of `%qb` and `%qb_recscale` are expected to be folded by constant
   * propagation so that they become constants.
   *
   * For more information about dynamic quantization, see https://www.maartengrootendorst.com/blog/quantization
   */
  // clang-format on
  Value rewriteSym() {
    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);

    Type i8Ty = rewriter.getIntegerType(8);
    Type si64Ty = rewriter.getIntegerType(64, true);
    Type f16Ty = rewriter.getF16Type();
    Type f32Ty = rewriter.getF32Type();
    RankedTensorType scalarTy = RankedTensorType::get({}, f32Ty);

    IntegerAttr trueAttr = rewriter.getIntegerAttr(si64Ty, -1);
    IntegerAttr falseAttr = rewriter.getIntegerAttr(si64Ty, 0);

    Value none = create.onnx.none();
    Value cst0 = create.onnx.constant(
        DenseElementsAttr::get(scalarTy, static_cast<float>(0)));
    Value cst1 = create.onnx.constant(
        DenseElementsAttr::get(scalarTy, static_cast<float>(1)));
    Value cst127 = create.onnx.constant(
        DenseElementsAttr::get(scalarTy, static_cast<float>(127)));
    Value cstNeg127 = create.onnx.constant(
        DenseElementsAttr::get(scalarTy, static_cast<float>(-127)));

    int64_t rankA = getRank(A.getType());
    int64_t rankB = getRank(B.getType());
    StringAttr aLayoutAttr =
        rewriter.getStringAttr((rankA == 2) ? LAYOUT_2D : LAYOUT_3DS);
    StringAttr bLayoutAttr =
        rewriter.getStringAttr((rankB == 2) ? LAYOUT_2D : LAYOUT_3DS);

    // Quantize and stickify A.
    IntegerAttr symModeAttr =
        rewriter.getIntegerAttr(rewriter.getI64Type(), symForA ? 1 : 0);
    ZHighQuantizedStickOp qAOp =
        rewriter.create<ZHighQuantizedStickOp>(loc, A, none, none, aLayoutAttr,
            rewriter.getStringAttr(QTYPE_DLFLOAT16), symModeAttr);
    Value AI8 = qAOp.getResult(0);
    Value ARecScale = qAOp.getResult(1);
    Value AOffset = qAOp.getResult(2);

    // Quantize B. All computations here would be folded by constprop.
    // Though computation here can be generalized for other integer types by
    // changing qmin and qmax, we optimize it for i8 since NNPA supports i8 only
    // at this moment.
    // Symmetric mode for i8, meaning offset = 0, qmin = -127, qmax = 127.
    Value BOffset = cst0;
    Value qmin = cstNeg127;
    Value qmax = cst127;
    // %absmax = onnx.ReduceMax(onnx.Abs(%B))
    // %b_rescale = onnx.Div(127, absmax)
    Value absMax =
        create.onnx.reduceMax(scalarTy, create.onnx.abs(B), none, false, false);
    Value BRecScale = create.onnx.div(cst127, absMax);
    // %qb = onnx.Cast(
    //  onnx.Clip(onnx.Round(onnx.Mul(%B, %b_rescale)), qmin, qmax))
    Value BI8 = create.onnx.cast(
        create.onnx.clip(
            create.onnx.round(create.onnx.mul(B, BRecScale)), qmin, qmax),
        i8Ty);
    // Stickify B.
    ZHighQuantizedStickOp qBOp =
        rewriter.create<ZHighQuantizedStickOp>(loc, BI8, BRecScale, BOffset,
            bLayoutAttr, rewriter.getStringAttr(QTYPE_WEIGHTS));

    // Output information.
    Value YRecScale = cst1;
    Value YOffset = cst0;

    // When A is also quantized using symmetric mode, both correction terms for
    // A and B are canceled out. Thus, no precomputation is needed.
    Value qcTilde = none, qcTildeRecScale = cst1, qcTildeOffset = cst0;
    if (!symForA) {
      // When only B is quantized using symmetric mode, precompute the
      // correction term for B only.
      preComputeBias(create, ARecScale, AOffset, BI8, BRecScale, YRecScale,
          YOffset, qcTilde, qcTildeRecScale, qcTildeOffset);
    }
    // Add up C into bias if C is given.
    if (C) {
      int64_t rankC = getRank(C.getType());
      assert((rankC == rankB - 1) &&
             "C has a wrong shape to be added into pre_computed_bias");
      assert((rankC == 1 || rankC == 2) && "Wrong rank for C");
      StringAttr cLayoutAttr =
          rewriter.getStringAttr((rankC == 1) ? LAYOUT_1D : LAYOUT_2DS);
      Value stickC = rewriter.create<ZHighStickOp>(loc, C, cLayoutAttr);
      if (symForA)
        qcTilde = stickC;
      else
        qcTilde = rewriter.create<ZHighAddOp>(
            loc, qcTilde.getType(), qcTilde, stickC);
    }

    // Emit zhigh.QuantizedMatMul.
    // No need to dequantize since Y's rescale is 1.
    // Do not clip the output values to i8, keep i32.
    SmallVector<Type, 3> resTypes;
    resTypes.emplace_back(UnrankedTensorType::get(f16Ty));
    resTypes.emplace_back(RankedTensorType::get({}, f32Ty));
    resTypes.emplace_back(RankedTensorType::get({}, f32Ty));
    ZHighQuantizedMatMulOp zhighQuantizedMatMulOp =
        rewriter.create<ZHighQuantizedMatMulOp>(loc, resTypes, AI8, ARecScale,
            AOffset, qBOp.getResult(0), BRecScale, BOffset, qcTilde,
            qcTildeRecScale, qcTildeOffset,
            /*OutRecScale*/ YRecScale, /*OutOffset*/ YOffset,
            /*PreComputedBias*/ trueAttr, /*DisableClipping*/ trueAttr,
            /*DequantizeOutput*/ falseAttr);
    (void)zhighQuantizedMatMulOp.inferShapes([](Region &region) {});

    // Unstickify the matmul result that is int8-as-float.
    Value res = rewriter.create<ZHighUnstickOp>(
        loc, zhighQuantizedMatMulOp.getResult(0));
    return res;
  }

private:
  PatternRewriter &rewriter;
  Location loc;
  Operation *op;
  Value A, B, C;
  // Whether do symmetric quant for activation input A or not.
  bool symForA = false;
};

//===----------------------------------------------------------------------===//
// ONNX to ZHigh Lowering Pass
//===----------------------------------------------------------------------===//

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXONNXToZHigh.inc"

// Enhance 'replaceONNXSumOpPatternRecursion' to allow operating recursively.
struct replaceONNXSumOpPatternEnhancedRecursion
    : public replaceONNXSumOpPatternRecursion {
  replaceONNXSumOpPatternEnhancedRecursion(MLIRContext *context)
      : replaceONNXSumOpPatternRecursion(context) {}
  void initialize() {
    // This pattern recursively unpacks one variadic operand at a time. The
    // recursion bounded as the number of variadic operands is strictly
    // decreasing.
    setHasBoundedRewriteRecursion(true);
  }
};

/**
 * This is a pattern for doing i8 dynamic quantization (symmetric mode) for
 * onnx.MatMul(%A, %B), where %B is a constant.
 */

class replaceONNXMatMulByDynQuantI8Pattern
    : public OpRewritePattern<ONNXMatMulOp> {
public:
  using OpRewritePattern<ONNXMatMulOp>::OpRewritePattern;

  replaceONNXMatMulByDynQuantI8Pattern(
      MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<ONNXMatMulOp>(context, benefit) {}

  LogicalResult matchAndRewrite(
      ONNXMatMulOp mmOp, PatternRewriter &rewriter) const override {
    Location loc = mmOp.getLoc();
    Operation *op = mmOp.getOperation();
    Value A = mmOp.getA();
    Value B = mmOp.getB();

    // Dynamic quantization helper.
    DynQuantI8PatternHelper dqHelper(rewriter, loc, op, A, B, nullptr,
        ONNXToZHighLoweringConfiguration::Quant::isActivationSym);

    // Match
    if (!isSuitableForZDNN<ONNXMatMulOp>(mmOp) || failed(dqHelper.match()))
      return rewriter.notifyMatchFailure(op, "MatMul is not suitable for zDNN");

    // Rewrite
    Value res = dqHelper.rewriteSym();
    rewriter.replaceOp(op, res);
    return success();
  }
};

/**
 * This is a pattern for doing i8 dynamic quantization (symmetric mode) for
 * `onnx.Add(onnx.MatMul(%A, %B), %C)`. where
 * - %B and %C are a constant and
 * - %B and %C must have compatible shape, i.e. the reduction shape on the last
 *   second dim of %B is the same as %C's shape.
 */
class replaceONNXMatMulAddByDynQuantI8Pattern
    : public OpRewritePattern<ONNXAddOp> {
public:
  using OpRewritePattern<ONNXAddOp>::OpRewritePattern;

  replaceONNXMatMulAddByDynQuantI8Pattern(
      MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<ONNXAddOp>(context, benefit) {}

  LogicalResult matchAndRewrite(
      ONNXAddOp addOp, PatternRewriter &rewriter) const override {
    Location loc = addOp.getLoc();
    Operation *op = addOp.getOperation();
    Value lhs = addOp.getOperand(0);
    Value rhs = addOp.getOperand(1);

    // Match A*B+C and C+A*B where B and C are constants, and then rewrite.
    Value AB, C;
    if (!areDefinedBy<ONNXMatMulOp, ONNXConstantOp>(lhs, rhs, AB, C))
      return rewriter.notifyMatchFailure(
          op, "MatMulAdd is not suitable for zDNN.");
    ONNXMatMulOp mmOp = AB.getDefiningOp<ONNXMatMulOp>();
    Value A = mmOp.getA();
    Value B = mmOp.getB();

    // Match A, B, C.
    DynQuantI8PatternHelper dqHelper(rewriter, loc, op, A, B, C,
        ONNXToZHighLoweringConfiguration::Quant::isActivationSym);
    if (succeeded(dqHelper.match())) {
      Value res = dqHelper.rewriteSym();
      rewriter.replaceOp(op, res);
      return success();
    }

    return failure();
  }
};

/**
 * This is a pattern for doing i8 dynamic quantization (symmetric mode) for
 * onnx.Gemm(%A, %B, %C), where %B and %C are constants.
 *
 * This pattern is applied only when the compiler option
 * `--nnpa-quantization={DynSymI8|SymSymI8}` is specified.
 *
 */

class replaceONNXGemmByDynQuantI8Pattern : public OpRewritePattern<ONNXGemmOp> {
public:
  using OpRewritePattern<ONNXGemmOp>::OpRewritePattern;

  replaceONNXGemmByDynQuantI8Pattern(
      MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<ONNXGemmOp>(context, benefit) {}

  LogicalResult matchAndRewrite(
      ONNXGemmOp gemmOp, PatternRewriter &rewriter) const override {
    Location loc = gemmOp.getLoc();
    Operation *op = gemmOp.getOperation();

    Value A = gemmOp.getA();
    Value B = gemmOp.getB();
    Value C = gemmOp.getC();
    bool transA = (gemmOp.getTransA() != 0);
    bool transB = (gemmOp.getTransB() != 0);

    // Dynamic quantization helper.
    DynQuantI8PatternHelper dqHelper(rewriter, loc, op, A, B,
        isNoneValue(C) ? nullptr : C,
        ONNXToZHighLoweringConfiguration::Quant::isActivationSym);

    // Match
    // TODO: if B is a constant and it is transposed, we can do transpose
    // explicitly.
    if (transA || transB)
      return rewriter.notifyMatchFailure(op, "Gemm is with transpose");
    if (!isSuitableForZDNN<ONNXGemmOp>(gemmOp))
      return rewriter.notifyMatchFailure(op, "Gemm is not suitable for zDNN");
    if (failed(dqHelper.match()))
      return failure();

    // Rewrite
    Value res = dqHelper.rewriteSym();
    rewriter.replaceOp(op, res);
    return success();
  }
};

class replaceONNXMatMulIntegerPattern
    : public OpRewritePattern<ONNXMatMulIntegerOp> {
public:
  using OpRewritePattern<ONNXMatMulIntegerOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXMatMulIntegerOp mmiOp, PatternRewriter &rewriter) const override {
    Location loc = mmiOp.getLoc();
    Operation *op = mmiOp.getOperation();
    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);

    // Match
    if (failed(canBeRewritten(rewriter, mmiOp)))
      return failure();

    Type si64Ty = rewriter.getIntegerType(64, true);
    Type f16Ty = rewriter.getF16Type();
    Type f32Ty = rewriter.getF32Type();
    Type outElemTy = getElementType(mmiOp.getY().getType());
    IntegerAttr trueAttr = rewriter.getIntegerAttr(si64Ty, -1);
    IntegerAttr falseAttr = rewriter.getIntegerAttr(si64Ty, 0);

    auto cst0Attr = DenseElementsAttr::get(
        RankedTensorType::get({}, f32Ty), static_cast<float>(0));
    auto cst1Attr = DenseElementsAttr::get(
        RankedTensorType::get({}, f32Ty), static_cast<float>(1));
    Value none = create.onnx.none();
    Value zero = create.onnx.constant(cst0Attr);
    Value zeroI64 = create.onnx.constantInt64({0});
    Value one = create.onnx.constant(cst1Attr);

    // Prepare inputs for zhigh QuantizedMatMul.

    // I8 tensors
    Value AI8 = getOrCastToI8(mmiOp.getA(), create, true);
    Value BI8 = getOrCastToI8(mmiOp.getB(), create, true);

    // Zero points in f32.
    Value AZeroPointI8 = mmiOp.getAZeroPoint();
    if (getRank(AZeroPointI8.getType()) == 1) {
      // Normalize the zeropoint tensor to tensor<dtype>.
      AZeroPointI8 = create.onnx.squeeze(
          RankedTensorType::get({}, getElementType(AZeroPointI8.getType())),
          AZeroPointI8, {zeroI64});
    }
    AZeroPointI8 = getOrCastToI8(AZeroPointI8, create, true);
    Value AZeroPointF32 = create.onnx.cast(AZeroPointI8, f32Ty);
    // TESTING: minus zeropoint in advance to cancel out the software part of
    // zdnn quantized matmul.
    // AI8 = create.onnx.sub(AI8, AZeroPointI8);
    // Value AZeroPointF32 = zero;
    Value BZeroPointI8 = mmiOp.getBZeroPoint();
    if (getRank(BZeroPointI8.getType()) == 1) {
      // Normalize the zeropoint tensor to tensor<dtype>.
      BZeroPointI8 = create.onnx.squeeze(
          RankedTensorType::get({}, getElementType(BZeroPointI8.getType())),
          BZeroPointI8, {zeroI64});
    }
    BZeroPointI8 = getOrCastToI8(BZeroPointI8, create, true);
    Value BZeroPointF32 = create.onnx.cast(BZeroPointI8, f32Ty);
    // TESTING: minus zeropoint in advance to cancel out the software part of
    // zdnn quantized matmul.
    // BI8 = create.onnx.sub(BI8, AZeroPointI8);
    // Value BZeroPointF32 = zero;
    Value YZeroPointF32 = zero;

    // Recscale in f32.
    // Set recscale of A and B to 1. In dynamic quantization the output of
    // MatMulInteger is scaled later outside the op.
    Value ARecScale = one;
    Value BRecScale = one;
    Value YRecScale = one;

    // Only pre-compute bias when B is a constant and BZeroPoint is zero.
    bool canPreComputeBias = isDenseONNXConstant(BI8) &&
                             isDenseONNXConstant(BZeroPointI8) &&
                             isConstOf(BZeroPointI8, 0.0);

    // Stickify AI8, Transform AI8 into zTensor format.
    int64_t rankA = getRank(AI8.getType());
    StringAttr aLayoutAttr =
        rewriter.getStringAttr((rankA == 2) ? LAYOUT_2D : LAYOUT_3DS);
    ZHighQuantizedStickOp qAOp =
        rewriter.create<ZHighQuantizedStickOp>(loc, AI8, ARecScale,
            AZeroPointF32, aLayoutAttr, rewriter.getStringAttr(QTYPE_INT8));

    // Stickify BI8. It is potentially folded at compile time.
    int64_t rankB = getRank(BI8.getType());
    StringAttr bLayoutAttr =
        rewriter.getStringAttr((rankB == 2) ? LAYOUT_2D : LAYOUT_3DS);
    ZHighQuantizedStickOp qBOp =
        rewriter.create<ZHighQuantizedStickOp>(loc, BI8, BRecScale,
            BZeroPointF32, bLayoutAttr, rewriter.getStringAttr(QTYPE_WEIGHTS));

    // Bias is none or precomputed.
    Value qcTilde, qcTildeRecScale, qcTildeZeroPointF32;
    if (canPreComputeBias)
      preComputeBias(create, ARecScale, AZeroPointF32, BI8, BRecScale,
          YRecScale, YZeroPointF32, qcTilde, qcTildeRecScale,
          qcTildeZeroPointF32);

    // Emit zhigh.QuantizedMatMul. Bias is none.
    // Do not dequantize, we want to keep the integer values that will be scaled
    // outside this op.
    // Do not clip the output values to i8, keep i32.
    SmallVector<Type, 3> resTypes;
    resTypes.emplace_back(UnrankedTensorType::get(f16Ty));
    resTypes.emplace_back(RankedTensorType::get({}, f32Ty));
    resTypes.emplace_back(RankedTensorType::get({}, f32Ty));
    ZHighQuantizedMatMulOp zhighQuantizedMatMulOp =
        rewriter.create<ZHighQuantizedMatMulOp>(loc, resTypes,
            qAOp.getResult(0), qAOp.getResult(1), qAOp.getResult(2),
            qBOp.getResult(0), qBOp.getResult(1), qBOp.getResult(2),
            /*Bias*/ canPreComputeBias ? qcTilde : none,
            /*BiasRecScale*/ canPreComputeBias ? qcTildeRecScale : none,
            /*BiasOffset*/ canPreComputeBias ? qcTildeZeroPointF32 : none,
            /*OutRecScale*/ YRecScale, /*OutOffset*/ YZeroPointF32,
            /*PreComputedBias*/ canPreComputeBias ? trueAttr : falseAttr,
            /*DisableClipping*/ trueAttr,
            /*DequantizeOutput*/ falseAttr);
    (void)zhighQuantizedMatMulOp.inferShapes([](Region &region) {});

    // Unstickify the matmul result that is int8-as-float.
    Value resI8F32 = rewriter.create<ZHighUnstickOp>(
        loc, zhighQuantizedMatMulOp.getResult(0));
    Value res = create.onnx.cast(resI8F32, outElemTy);

    rewriter.replaceOp(op, res);
    return success();
  }

  static mlir::LogicalResult canBeRewritten(
      PatternRewriter &rewriter, ONNXMatMulIntegerOp mmiOp) {
    if (!isSuitableForZDNN<ONNXMatMulIntegerOp>(mmiOp))
      return rewriter.notifyMatchFailure(
          mmiOp, "MatMulInteger is not suitable for zDNN");
    return success();
  }
};

// Replace by zhigh ops the following pattern:
// clang-format off
// func.func @pattern_in_bert(%X: tensor<?x?x768xf32>) : (tensor<?x?x768xf32>) -> tensor<?x?x768xf32> {
//     %y = onnx.Constant dense_resource<__elided__> : tensor<768x768xi8>
//     %y_scale = onnx.Constant dense<0.00656270096> : tensor<f32>
//     %y_zero_point = onnx.Constant dense<0> : tensor<i8>
//
//     %x, %x_scale, %x_zero_point = "onnx.DynamicQuantizeLinear"(%X) : (tensor<?x?x768xf32>) -> (tensor<?x?x768xui8>, tensor<f32>, tensor<ui8>)
//
//     %matmul = "onnx.MatMulInteger"(%x, %y, %x_zero_point, %y_zero_point) : (tensor<?x?x768xui8>, tensor<768x768xi8>, tensor<ui8>, tensor<i8>) -> tensor<?x?x768xi32>
//     %cast = "onnx.Cast"(%matmul) {saturate = 1 : si64, to = f32} : (tensor<?x?x768xi32>) -> tensor<?x?x768xf32>
//     %mul_1= "onnx.Mul"(%cast, %x_scale) : (tensor<?x?x768xf32>, tensor<f32>) -> tensor<?x?x768xf32>
//     %mul_2= "onnx.Mul"(%mul_1, %y_scale) : (tensor<?x?x768xf32>, tensor<f32>) -> tensor<?x?x768xf32>
//
//     return %mul_2: tensor<?x?x768xf32>
// }
// clang-format on
class replaceMatMulIntegerSubGraphFromMulPattern
    : public OpRewritePattern<ONNXMulOp> {
public:
  using OpRewritePattern<ONNXMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXMulOp mulOp, PatternRewriter &rewriter) const override {
    Location loc = mulOp.getLoc();
    Operation *op = mulOp.getOperation();
    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);

    // Match
    Value A, AI8, AScale, AZeroPointI8, BI8, BScale, BZeroPointI8;
    if (failed(canBeRewritten(rewriter, mulOp, A, AI8, AScale, AZeroPointI8,
            BI8, BScale, BZeroPointI8)))
      return failure();

    Type si64Ty = rewriter.getIntegerType(64, true);
    Type f16Ty = rewriter.getF16Type();
    Type f32Ty = rewriter.getF32Type();
    IntegerAttr trueAttr = rewriter.getIntegerAttr(si64Ty, -1);
    IntegerAttr falseAttr = rewriter.getIntegerAttr(si64Ty, 0);
    Value none = create.onnx.none();

    // Only pre-compute bias when BZeroPoint is zero.
    bool canPreComputeBias = isDenseONNXConstant(BI8) &&
                             isDenseONNXConstant(BZeroPointI8) &&
                             isConstOf(BZeroPointI8, 0.0);

    // Stickify A.
    int64_t rankA = getRank(A.getType());
    StringAttr aLayoutAttr =
        rewriter.getStringAttr((rankA == 2) ? LAYOUT_2D : LAYOUT_3DS);
    ZHighQuantizedStickOp qAOp;
    if (nnpaUseDynamicQuantizeLinearOnCPU) {
      Value zeroI64 = create.onnx.constantInt64({0});
      // Input A was quantized on CPU by onnx.DynamicQuantizedLinear: f32 to i8.
      if (getRank(AZeroPointI8.getType()) == 1) {
        // Normalize the zeropoint tensor to tensor<dtype>.
        AZeroPointI8 = create.onnx.squeeze(
            RankedTensorType::get({}, getElementType(AZeroPointI8.getType())),
            AZeroPointI8, {zeroI64});
      }
      AZeroPointI8 = getOrCastToI8(AZeroPointI8, create, true);
      Value AZeroPointF32 = create.onnx.cast(AZeroPointI8, f32Ty);
      Value ARecScale = create.onnx.reciprocal(AScale);
      AI8 = getOrCastToI8(AI8, create, true);
      // Stickify the quantized input A to ztensor format.
      qAOp = rewriter.create<ZHighQuantizedStickOp>(loc, AI8, ARecScale,
          AZeroPointF32, aLayoutAttr, rewriter.getStringAttr(QTYPE_INT8));
    } else {
      // Stickify input A to dlfloat16, and it will be quantized internally by
      // the NNPA quantized matmul.
      qAOp = rewriter.create<ZHighQuantizedStickOp>(loc, A, none, none,
          aLayoutAttr, rewriter.getStringAttr(QTYPE_DLFLOAT16));
    }
    Value qA = qAOp.getResult(0);
    Value ARecScale = qAOp.getResult(1);
    Value AZeroPoint = qAOp.getResult(2);

    // Stickify B. It is potentially folded at compile time.
    int64_t rankB = getRank(BI8.getType());
    StringAttr bLayoutAttr =
        rewriter.getStringAttr((rankB == 2) ? LAYOUT_2D : LAYOUT_3DS);
    Value BRecScale = create.onnx.reciprocal(BScale);
    Value BZeroPoint = create.onnx.cast(BZeroPointI8, f32Ty);
    ZHighQuantizedStickOp qBOp =
        rewriter.create<ZHighQuantizedStickOp>(loc, BI8, BRecScale, BZeroPoint,
            bLayoutAttr, rewriter.getStringAttr(QTYPE_WEIGHTS));
    Value qB = qBOp.getResult(0);

    // Output's rescale and zeropoint
    auto cst0Attr =
        DenseElementsAttr::get(RankedTensorType::get({}, f32Ty), (float)0);
    auto cst1Attr =
        DenseElementsAttr::get(RankedTensorType::get({}, f32Ty), (float)1);
    Value OutRecScale = create.onnx.constant(cst1Attr);
    Value OutZeroPoint = create.onnx.constant(cst0Attr);

    // Bias is none or precomputed.
    Value qcTilde, qcTildeRecScale, qcTildeZeroPoint;
    if (canPreComputeBias)
      preComputeBias(create, ARecScale, AZeroPoint, BI8, BRecScale, OutRecScale,
          OutZeroPoint, qcTilde, qcTildeRecScale, qcTildeZeroPoint);

    // Emit zhigh.QuantizedMatMul.
    SmallVector<Type, 3> resTypes;
    resTypes.emplace_back(UnrankedTensorType::get(f16Ty));
    resTypes.emplace_back(RankedTensorType::get({}, f32Ty));
    resTypes.emplace_back(RankedTensorType::get({}, f32Ty));
    ZHighQuantizedMatMulOp zhighQuantizedMatMulOp =
        rewriter.create<ZHighQuantizedMatMulOp>(loc, resTypes, qA, ARecScale,
            AZeroPoint, qB, BRecScale, BZeroPoint,
            /*Bias*/ canPreComputeBias ? qcTilde : none,
            /*BiasRecScale*/ canPreComputeBias ? qcTildeRecScale : none,
            /*BiasOffset*/ canPreComputeBias ? qcTildeZeroPoint : none,
            /*OutRecScale*/ OutRecScale, /*OutOffset*/ OutZeroPoint,
            /*PreComputedBias*/ canPreComputeBias ? trueAttr : falseAttr,
            /*DequantizeOutput*/ trueAttr);
    (void)zhighQuantizedMatMulOp.inferShapes([](Region &region) {});

    // Unstickify the matmul result.
    Value res = rewriter.create<ZHighUnstickOp>(
        loc, zhighQuantizedMatMulOp.getResult(0));

    rewriter.replaceOp(op, res);
    return success();
  }

  // clang-format off
  // func.func @pattern_in_bert(%A) {
  //   // A is dynamically quantized.
  //   %a, %a_scale, %a_zero_point = "onnx.DynamicQuantizeLinear"(%A)
  //
  //   // B is a constant and already quantized.
  //   %b             = onnx.Constant
  //   %b_scale       = onnx.Constant
  //   %b_zero_point  = onnx.Constant
  //
  //
  //   %matmul = "onnx.MatMulInteger"(%b, %b, %b_zero_point, %b_zero_point)
  //
  //   // Scale the output.
  //   %mm_f32     = "onnx.Cast"(%matmul) {to = f32}
  //   %mm_a_scale = "onnx.Mul"(%mm_f32, %a_scale)
  //   %mm_ab_scale = "onnx.Mul"(%mm_a_scale, %b_scale)
  //
  //   return %mm_y_scale
  // }
  // clang-format on
  static mlir::LogicalResult canBeRewritten(PatternRewriter &rewriter,
      ONNXMulOp mulOp, Value &A, Value &AI8, Value &AScale, Value &AZeroPoint,
      Value &BI8, Value &BScale, Value &BZeroPoint) {

    // Match `cast(mm_out) * a_scale * b_scale` to find two scales but we don't
    // know yet which scale is for A or B.
    Value scale1, scale2;
    ONNXCastOp castOp;
    ONNXMulOp mulScaleOp;

    Value opr1 = mulOp.getOperand(0);
    Value opr2 = mulOp.getOperand(1);

    // Match cast(mm_out) * (a_scale * b_scale)
    castOp = opr1.getDefiningOp<ONNXCastOp>();
    mulScaleOp = opr2.getDefiningOp<ONNXMulOp>();
    bool foundScales = false;
    if (castOp && mulScaleOp && isScalarTensor(opr2)) {
      Value lhs = mulScaleOp.getOperand(0);
      Value rhs = mulScaleOp.getOperand(1);
      if (isScalarTensor(lhs) && isScalarTensor(rhs)) {
        // mulScaleOp is a_scale * b_scale;
        foundScales = true;
        scale1 = lhs;
        scale2 = rhs;
      }
    }
    // Match (a_scale * b_scale) * cast(mm_out)
    if (!foundScales) {
      mulScaleOp = opr1.getDefiningOp<ONNXMulOp>();
      castOp = opr2.getDefiningOp<ONNXCastOp>();
      if (mulScaleOp && isScalarTensor(opr1) && castOp) {
        Value lhs = mulScaleOp.getOperand(0);
        Value rhs = mulScaleOp.getOperand(1);
        if (isScalarTensor(lhs) && isScalarTensor(rhs)) {
          // mulScaleOp is a_scale * b_scale;
          foundScales = true;
          scale1 = lhs;
          scale2 = rhs;
        }
      }
    }
    // Match [cast(mm_out) * a_scale] * b_scale
    if (!foundScales & isScalarTensor(opr2)) {
      scale1 = opr2;
      mulScaleOp = opr1.getDefiningOp<ONNXMulOp>();
      if (mulScaleOp) {
        Value lhs = mulScaleOp.getOperand(0);
        Value rhs = mulScaleOp.getOperand(1);
        castOp = lhs.getDefiningOp<ONNXCastOp>();
        if (castOp && isScalarTensor(rhs)) {
          // Match cast(mm_out) * a_scale
          scale2 = rhs;
          foundScales = true;
        }
        if (!foundScales) {
          // Match a_scale * cast(mm_out)
          castOp = rhs.getDefiningOp<ONNXCastOp>();
          if (isScalarTensor(lhs) && castOp) {
            scale2 = lhs;
            foundScales = true;
          }
        }
      }
      // Match b_scale * [cast(mm_out) * a_scale]
      if (!foundScales && isScalarTensor(opr1)) {
        scale1 = opr1;
        mulScaleOp = opr2.getDefiningOp<ONNXMulOp>();
        if (mulScaleOp) {
          Value lhs = mulScaleOp.getOperand(0);
          Value rhs = mulScaleOp.getOperand(1);
          castOp = lhs.getDefiningOp<ONNXCastOp>();
          if (castOp && isScalarTensor(rhs)) {
            // Match cast(mm_out) * a_scale
            scale2 = rhs;
            foundScales = true;
          }
          if (!foundScales) {
            // Match a_scale * cast(mm_out)
            castOp = rhs.getDefiningOp<ONNXCastOp>();
            if (isScalarTensor(lhs) && castOp) {
              scale2 = lhs;
              foundScales = true;
            }
          }
        }
      }
    }
    if (!foundScales)
      return rewriter.notifyMatchFailure(mulOp, "Not found scale values");

    // Identify a_scale and b_scale.
    // a_scale is from DynamicQuantizeLinear.
    if (scale1.getDefiningOp<ONNXDynamicQuantizeLinearOp>()) {
      AScale = scale1;
      BScale = scale2;
    } else if (scale2.getDefiningOp<ONNXDynamicQuantizeLinearOp>()) {
      AScale = scale2;
      BScale = scale1;
    } else {
      return rewriter.notifyMatchFailure(
          mulOp, "Could not identify a_scale and b_scale");
    }

    // Match cast.
    //   %cast = "onnx.Cast"(%matmul) {saturate = 1 : si64, to = f32}
    Type castOutputType = castOp.getOutput().getType();
    Type castInputType = castOp.getInput().getType();
    if (isRankedShapedType(castInputType) &&
        isRankedShapedType(castOutputType)) {
      if (!getElementType(castInputType).isInteger(32))
        return rewriter.notifyMatchFailure(
            mulOp, "ONNXCast is not casting from i32");
      if (!getElementType(castOutputType).isF32())
        return rewriter.notifyMatchFailure(
            mulOp, "ONNXCast is not casting to f32");
    } else {
      return rewriter.notifyMatchFailure(mulOp, "ONNXCast is unranked");
    }

    // Match matmul to get BI8 and BZeroPoint.
    ONNXMatMulIntegerOp matmulOp =
        castOp.getInput().getDefiningOp<ONNXMatMulIntegerOp>();
    if (!matmulOp)
      return rewriter.notifyMatchFailure(
          mulOp, "The input of the CastOp is not defined by MatMulIntegerOp");
    if (!isSuitableForZDNN<ONNXMatMulIntegerOp>(matmulOp))
      return rewriter.notifyMatchFailure(
          mulOp, "MatMulInteger is not suitable for zDNN");

    AI8 = matmulOp->getOperand(0);
    BI8 = matmulOp->getOperand(1);
    AZeroPoint = matmulOp->getOperand(2);
    BZeroPoint = matmulOp->getOperand(3);
    if (!isDenseONNXConstant(BI8))
      return rewriter.notifyMatchFailure(mulOp, "Quantized Y is not constant");
    if (!isDenseONNXConstant(BZeroPoint))
      return rewriter.notifyMatchFailure(mulOp, "BZeroPoint is not constant");
    if (!(getElementType(BI8.getType()).isUnsignedInteger(8) ||
            getElementType(BI8.getType()).isSignlessInteger(8)))
      return rewriter.notifyMatchFailure(
          mulOp, "Quantized Y is not signed int8");

    // Match dynamic quantize linear to get A.
    if (auto dqlOp =
            llvm::dyn_cast<ONNXDynamicQuantizeLinearOp>(AI8.getDefiningOp())) {
      if (AScale != dqlOp.getResult(1))
        return rewriter.notifyMatchFailure(mulOp, "AScale is not used");
      if (AZeroPoint != dqlOp.getResult(2))
        return rewriter.notifyMatchFailure(mulOp, "AZeroPoint is not used");
      // return A.
      A = dqlOp.getOperand();
    } else {
      return rewriter.notifyMatchFailure(
          mulOp, "Quantized A is not defined by DynamicQuantizeLinearOp");
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Fuse ZHighQuantizedMatMul and ONNXAdd
//===----------------------------------------------------------------------===//
// Rewrite this pattern:
//   (ONNXAddOp
//     $x,
//     (ZHighUnstickOp
//       (ZHighQuantizedMatMulOp:$mm_res
//         $a, $Sa, $Za,
//         $b, $Sb, $Zb,
//         (ZHighQuantizedStick $c), $Sc, $Zb,
//         $So, $Zo,
//         $preComputed, $disableClipping, $dequantized))),
//
// into this pattern where $x is added to $c:
//
//   (ZHighUnstickOp
//     (ZHighQuantizedMatMulOp
//       $a, $Sa, $Za,
//       $b, $Sb, $Zb,
//       (ZHighQuantizedStick (ONNXAddOp $x, $c)), $Sc, $Zb,
//       $So, $Zo,
//       $preComputed, $disableClipping, $dequantized)),
//
// Requirement: `preComputed` is true.

class fuseZHighQuantizedMatMulONNXAddPattern
    : public OpRewritePattern<ONNXAddOp> {
public:
  using OpRewritePattern<ONNXAddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXAddOp addOp, PatternRewriter &rewriter) const override {
    Location loc = addOp.getLoc();
    Operation *op = addOp.getOperation();
    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);

    ZHighUnstickOp unstickOp;
    ZHighQuantizedMatMulOp mmOp;
    ZHighQuantizedStickOp qstickOp;
    Value addInput;

    // match
    if (failed(canBeRewritten(
            rewriter, addOp, unstickOp, mmOp, qstickOp, addInput)))
      return failure();

    // rewrite
    Value newBias = create.onnx.add(addInput, qstickOp.getIn());
    ZHighQuantizedStickOp newQStickOp = rewriter.create<ZHighQuantizedStickOp>(
        loc, newBias, qstickOp.getInRecScale(), qstickOp.getInOffset(),
        qstickOp.getLayoutAttr(), qstickOp.getQuantizedTypeAttr());

    SmallVector<Type, 3> resTypes;
    resTypes.emplace_back(mmOp.getResult(0).getType());
    resTypes.emplace_back(mmOp.getResult(1).getType());
    resTypes.emplace_back(mmOp.getResult(2).getType());
    ZHighQuantizedMatMulOp newQMMOp = rewriter.create<ZHighQuantizedMatMulOp>(
        loc, resTypes, mmOp.getX(), mmOp.getXRecScale(), mmOp.getXOffset(),
        mmOp.getY(), mmOp.getYRecScale(), mmOp.getYOffset(),
        newQStickOp.getResult(0), newQStickOp.getResult(1),
        newQStickOp.getResult(2), mmOp.getOutRecScaleIn(),
        mmOp.getOutOffsetIn(), mmOp.getPreComputedBiasAttr(),
        mmOp.getDisableClippingAttr(), mmOp.getDequantizeOutputAttr());
    ZHighUnstickOp newUnstickOp =
        rewriter.create<ZHighUnstickOp>(loc, newQMMOp.getResult(0));

    rewriter.replaceOp(op, newUnstickOp);
    return success();
  }

  static mlir::LogicalResult canBeRewritten(PatternRewriter &rewriter,
      ONNXAddOp addOp, ZHighUnstickOp &unstickOp, ZHighQuantizedMatMulOp &mmOp,
      ZHighQuantizedStickOp &qstickOp, Value &addInput) {
    Value lhs = addOp.getOperand(0);
    Value rhs = addOp.getOperand(1);
    bool found = false;
    if (auto op1 = lhs.getDefiningOp<ZHighUnstickOp>()) {
      addInput = rhs;
      unstickOp = op1;
      Value mmOutput = unstickOp.getIn();
      if (auto op2 = mmOutput.getDefiningOp<ZHighQuantizedMatMulOp>()) {
        mmOp = op2;
        bool precomputed = (mmOp.getPreComputedBias() == -1);
        if (!precomputed)
          return rewriter.notifyMatchFailure(
              addOp, "not precomputed quantized matmul");
        Value qBias = mmOp.getB();
        if (auto op3 = qBias.getDefiningOp<ZHighQuantizedStickOp>()) {
          qstickOp = op3;
          Value bias = qstickOp.getIn();
          // Check rank.
          if (getRank(bias.getType()) != getRank(addInput.getType()))
            return rewriter.notifyMatchFailure(addOp, "rank mismatched");
          found = true;
        }
      }
    }
    if (found)
      return success();

    if (auto op1 = rhs.getDefiningOp<ZHighUnstickOp>()) {
      addInput = lhs;
      unstickOp = op1;
      Value mmOutput = unstickOp.getIn();
      if (auto op2 = mmOutput.getDefiningOp<ZHighQuantizedMatMulOp>()) {
        mmOp = op2;
        bool precomputed = (mmOp.getPreComputedBias() == -1);
        if (!precomputed)
          return rewriter.notifyMatchFailure(
              addOp, "not precomputed quantized matmul");
        Value qBias = mmOp.getB();
        if (auto op3 = qBias.getDefiningOp<ZHighQuantizedStickOp>()) {
          qstickOp = op3;
          Value bias = qstickOp.getIn();
          // Check rank.
          if (getRank(bias.getType()) != getRank(addInput.getType()))
            return rewriter.notifyMatchFailure(addOp, "rank mismatched");
          found = true;
        }
      }
    }
    if (found)
      return success();

    return rewriter.notifyMatchFailure(addOp, "unstick not found");
  }
};

class replaceONNXQLinearMatMulPattern
    : public OpRewritePattern<ONNXQLinearMatMulOp> {
public:
  using OpRewritePattern<ONNXQLinearMatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
      ONNXQLinearMatMulOp qmmOp, PatternRewriter &rewriter) const override {
    Location loc = qmmOp.getLoc();
    Operation *op = qmmOp.getOperation();
    MultiDialectBuilder<OnnxBuilder> create(rewriter, loc);

    // Match
    if (failed(canBeRewritten(rewriter, qmmOp)))
      return failure();

    Type si64Ty = rewriter.getIntegerType(64, true);
    Type f16Ty = rewriter.getF16Type();
    Type f32Ty = rewriter.getF32Type();
    IntegerAttr trueAttr = rewriter.getIntegerAttr(si64Ty, -1);
    IntegerAttr falseAttr = rewriter.getIntegerAttr(si64Ty, 0);

    Value A = qmmOp.getA();
    Value AScale = qmmOp.getAScale();
    Value AZeroPoint = qmmOp.getAZeroPoint();
    Value B = qmmOp.getB();
    Value BScale = qmmOp.getBScale();
    Value BZeroPoint = qmmOp.getBZeroPoint();
    Value Y = qmmOp.getY();
    Value YScale = qmmOp.getYScale();
    Value YZeroPoint = qmmOp.getYZeroPoint();

    // Only pre-compute bias when B is a constant and BZeroPoint is int8 zero.
    bool canPreComputeBias = false;
    if (isDenseONNXConstant(B) && isDenseONNXConstant(BZeroPoint)) {
      if (getElementType(BZeroPoint.getType()).isUnsignedInteger())
        canPreComputeBias = isConstOf(BZeroPoint, 128.0);
      else
        canPreComputeBias = isConstOf(BZeroPoint, 0.0);
    }

    // Emit some common values.
    Value none = create.onnx.none();
    Value zero = create.onnx.constantInt64({0});

    // Normalize scalar tensors to tensor<dtype>.
    if (getRank(AScale.getType()) == 1) {
      AScale = create.onnx.squeeze(
          RankedTensorType::get({}, getElementType(AScale.getType())), AScale,
          {zero});
    }
    if (getRank(AZeroPoint.getType()) == 1) {
      AZeroPoint = create.onnx.squeeze(
          RankedTensorType::get({}, getElementType(AZeroPoint.getType())),
          AZeroPoint, {zero});
    }
    if (getRank(BScale.getType()) == 1) {
      BScale = create.onnx.squeeze(
          RankedTensorType::get({}, getElementType(BScale.getType())), BScale,
          {zero});
    }
    if (getRank(BZeroPoint.getType()) == 1) {
      BZeroPoint = create.onnx.squeeze(
          RankedTensorType::get({}, getElementType(BZeroPoint.getType())),
          BZeroPoint, {zero});
    }
    if (getRank(YScale.getType()) == 1) {
      YScale = create.onnx.squeeze(
          RankedTensorType::get({}, getElementType(YScale.getType())), YScale,
          {zero});
    }
    if (getRank(YZeroPoint.getType()) == 1) {
      YZeroPoint = create.onnx.squeeze(
          RankedTensorType::get({}, getElementType(YZeroPoint.getType())),
          YZeroPoint, {zero});
    }

    // zdnn supports signed int8, convert unsigned int8 inputs to signed int8.
    Value AI8 = getOrCastToI8(A, create);
    Value BI8 = getOrCastToI8(B, create);

    Value ARecScale = create.onnx.reciprocal(AScale);
    Value AZeroPointI8 = getOrCastToI8(AZeroPoint, create);
    Value AZeroPointF32 = create.onnx.cast(AZeroPointI8, f32Ty);

    Value BRecScale = create.onnx.reciprocal(BScale);
    Value BZeroPointI8 = getOrCastToI8(BZeroPoint, create);
    Value BZeroPointF32 = create.onnx.cast(BZeroPointI8, f32Ty);

    Value YRecScale = create.onnx.reciprocal(YScale);
    Value YZeroPointI8 = getOrCastToI8(YZeroPoint, create);
    Value YZeroPointF32 = create.onnx.cast(YZeroPointI8, f32Ty);

    // Stickify AI8, Transform AI8 into zTensor format.
    int64_t rankA = getRank(AI8.getType());
    StringAttr aLayoutAttr =
        rewriter.getStringAttr((rankA == 2) ? LAYOUT_2D : LAYOUT_3DS);
    ZHighQuantizedStickOp qAOp =
        rewriter.create<ZHighQuantizedStickOp>(loc, AI8, ARecScale,
            AZeroPointF32, aLayoutAttr, rewriter.getStringAttr(QTYPE_INT8));

    // Stickify BI8. It is potentially folded at compile time.
    int64_t rankB = getRank(BI8.getType());
    StringAttr bLayoutAttr =
        rewriter.getStringAttr((rankB == 2) ? LAYOUT_2D : LAYOUT_3DS);
    ZHighQuantizedStickOp qBOp =
        rewriter.create<ZHighQuantizedStickOp>(loc, BI8, BRecScale,
            BZeroPointF32, bLayoutAttr, rewriter.getStringAttr(QTYPE_WEIGHTS));

    // Bias is none or precomputed.
    Value qcTilde, qcTildeRecScale, qcTildeZeroPointF32;
    if (canPreComputeBias)
      preComputeBias(create, ARecScale, AZeroPointF32, BI8, BRecScale,
          YRecScale, YZeroPointF32, qcTilde, qcTildeRecScale,
          qcTildeZeroPointF32);

    // Emit zhigh.QuantizedMatMul. Bias is none.
    // DisableClipping gives the same output as the onnx backend test since the
    // onnx backend test uses `astype` instead of `clipping` to cast the output
    // to i8.
    SmallVector<Type, 3> resTypes;
    resTypes.emplace_back(UnrankedTensorType::get(f16Ty));
    resTypes.emplace_back(RankedTensorType::get({}, f32Ty));
    resTypes.emplace_back(RankedTensorType::get({}, f32Ty));
    ZHighQuantizedMatMulOp zhighQuantizedMatMulOp =
        rewriter.create<ZHighQuantizedMatMulOp>(loc, resTypes,
            qAOp.getResult(0), qAOp.getResult(1), qAOp.getResult(2),
            qBOp.getResult(0), qBOp.getResult(1), qBOp.getResult(2),
            /*Bias*/ canPreComputeBias ? qcTilde : none,
            /*BiasRecScale*/ canPreComputeBias ? qcTildeRecScale : none,
            /*BiasOffset*/ canPreComputeBias ? qcTildeZeroPointF32 : none,
            /*OutRecScale*/ YRecScale, /*OutOffset*/ YZeroPointF32,
            /*PreComputedBias*/ canPreComputeBias ? trueAttr : falseAttr,
            /*DisableClipping*/ trueAttr,
            /*DequantizeOutput*/ falseAttr);
    (void)zhighQuantizedMatMulOp.inferShapes([](Region &region) {});

    // Unstickify the matmul result that is int8-as-float.
    Value resI8F32 = rewriter.create<ZHighUnstickOp>(
        loc, zhighQuantizedMatMulOp.getResult(0));
    Value res;
    Type outElemTy = getElementType(Y.getType());
    if (outElemTy.isUnsignedInteger(8)) {
      // The zdnn output is int8. Convert int8 to uint8.
      // Use int16 to avoid integer overflow.
      Type i16Ty = rewriter.getI16Type();
      Type ui16Ty = rewriter.getIntegerType(16, false);
      auto cst128Attr = DenseElementsAttr::get(
          RankedTensorType::get({}, i16Ty), static_cast<int16_t>(128));
      // clang-format off
      Value resUI16 =
        create.onnx.cast(
          create.onnx.add(create.onnx.cast(resI8F32, i16Ty),
                          create.onnx.constant(cst128Attr)),
          ui16Ty);
      // clang-format on
      res = create.onnx.cast(resUI16, outElemTy);
    } else {
      res = create.onnx.cast(resI8F32, outElemTy);
    }
    rewriter.replaceOp(op, res);
    return success();
  }

  static mlir::LogicalResult canBeRewritten(
      PatternRewriter &rewriter, ONNXQLinearMatMulOp qmmOp) {
    if (!isSuitableForZDNN<ONNXQLinearMatMulOp>(qmmOp))
      return rewriter.notifyMatchFailure(
          qmmOp, "QLinearMatMul is not suitable for zDNN");
    return success();
  }
};

struct ONNXToZHighLoweringPass
    : public PassWrapper<ONNXToZHighLoweringPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ONNXToZHighLoweringPass)

  StringRef getArgument() const override { return "convert-onnx-to-zhigh"; }

  StringRef getDescription() const override {
    return "Lower ONNX ops to ZHigh ops.";
  }

  // Make sure that we have a valid default constructor and copy
  // constructor to make sure that the options are initialized properly.
  ONNXToZHighLoweringPass() = default;
  ONNXToZHighLoweringPass(const ONNXToZHighLoweringPass &pass)
      : PassWrapper<ONNXToZHighLoweringPass, OperationPass<ModuleOp>>() {}
  void runOnOperation() final;
};
} // end anonymous namespace.

void getONNXToZHighOneOpPatterns(RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.insert<normalizeONNXGemmTransAPattern>(context);
  patterns.insert<normalizeONNXGemmTransBPattern>(context);
  patterns.insert<replaceONNXAddPattern>(context);
  patterns.insert<replaceONNXAveragePoolPattern>(context);
  patterns.insert<replaceONNXConv2DPattern>(context);
  patterns.insert<replaceONNXDivBroadcastPattern1>(context);
  patterns.insert<replaceONNXDivBroadcastPattern2>(context);
  patterns.insert<replaceONNXDivPattern>(context);
  patterns.insert<replaceONNXExpPattern>(context);
  patterns.insert<replaceONNXGRUPattern1>(context);
  patterns.insert<replaceONNXGRUPattern2>(context);
  patterns.insert<replaceONNXGRUPattern3>(context);
  patterns.insert<replaceONNXGRUPattern4>(context);
  patterns.insert<replaceONNXGeluPattern>(context);
  patterns.insert<replaceONNXGemmBias2DPattern>(context);
  patterns.insert<replaceONNXGemmBiasNoneOr1DPattern>(context);
  patterns.insert<replaceONNXGemmTransPattern>(context);
  patterns.insert<replaceONNXLSTMPattern1>(context);
  patterns.insert<replaceONNXLSTMPattern2>(context);
  patterns.insert<replaceONNXLSTMPattern3>(context);
  patterns.insert<replaceONNXLSTMPattern4>(context);
  patterns.insert<replaceONNXLeakyReluPattern>(context);
  patterns.insert<replaceONNXLogPattern>(context);
  patterns.insert<replaceONNXMatMulPattern>(context);
  patterns.insert<replaceONNXMatMulIntegerPattern>(context);
  patterns.insert<replaceONNXMaxPattern>(context);
  patterns.insert<replaceONNXMaxPoolSingleOutPattern>(context);
  patterns.insert<replaceONNXMinPattern>(context);
  patterns.insert<replaceONNXMulPattern>(context);
  patterns.insert<replaceONNXQLinearMatMulPattern>(context);
  patterns.insert<replaceONNXReduceMaxPattern>(context);
  patterns.insert<replaceONNXReduceMeanV13Pattern>(context);
  patterns.insert<replaceONNXReduceMinPattern>(context);
  patterns.insert<replaceONNXReluPattern>(context);
  patterns.insert<replaceONNXSigmoidPattern>(context);
  patterns.insert<replaceONNXSoftmax2DPattern>(context);
  patterns.insert<replaceONNXSoftmax3DPattern>(context);
  patterns.insert<replaceONNXSqrtPattern>(context);
  patterns.insert<replaceONNXSubPattern>(context);
  patterns.insert<replaceONNXSumOpPatternEnhancedRecursion>(context);
  patterns.insert<replaceONNXSumOpPatternRecursion>(context);
  patterns.insert<replaceONNXSumOpPatternSingleton>(context);
  patterns.insert<replaceONNXTanhPattern>(context);

  // Pattern for i8 dynamic quantization.
  if (isCompatibleWithNNPALevel(NNPALevel::M15) &&
      ONNXToZHighLoweringConfiguration::isDynQuant) {
    // Bump up the pattern benefit to run these before non-quantization
    // patterns.
    PatternBenefit quantPriority(QUANT_PATTERN_BENEFIT);
    if (llvm::any_of(ONNXToZHighLoweringConfiguration::Quant::opTypes,
            [](std::string s) {
              return StringRef(s).equals_insensitive("MatMul");
            })) {
      patterns.insert<replaceONNXMatMulByDynQuantI8Pattern>(
          context, quantPriority);
      patterns.insert<replaceONNXGemmByDynQuantI8Pattern>(
          context, quantPriority);
    }
  }
}

void getONNXToZHighOneOpDynamicallyLegal(
    ConversionTarget *target, const DimAnalysis *dimAnalysis) {
  addDynamicallyLegalOpFor<ONNXAddOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXSubOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXMulOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXDivOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXSumOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXMinOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXMaxOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXGeluOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXLeakyReluOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXReluOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXSqrtOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXTanhOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXSigmoidOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXLogOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXExpOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXSoftmaxOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXMaxPoolSingleOutOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXAveragePoolOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXMatMulOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXGemmOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXReduceMaxOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXReduceMeanV13Op>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXReduceMinOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXLSTMOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXGRUOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXConvOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXMatMulIntegerOp>(target, dimAnalysis);
  addDynamicallyLegalOpFor<ONNXQLinearMatMulOp>(target, dimAnalysis);
}

void getONNXToZHighMultipleOpPatterns(RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.insert<replaceONNXMatMulAddPattern1>(context);
  patterns.insert<replaceONNXMatMulAddPattern2>(context);
  patterns.insert<replaceONNXReluConvPattern>(context);
  patterns.insert<replaceONNXLogSoftmaxPattern>(context);
  patterns.insert<replaceONNXTransAMatMulPattern>(context);
  patterns.insert<replaceONNXTransBMatMulPattern>(context);
  patterns.insert<replaceONNXTransABMatMulPattern>(context);
  patterns.insert<replaceDiv1SqrtPattern>(context);
  patterns.insert<replaceReciprocalSqrtPattern>(context);
  patterns.insert<replaceMatMulIntegerSubGraphFromMulPattern>(context);
  patterns.insert<fuseZHighQuantizedMatMulONNXAddPattern>(context);

  // Pattern for i8 dynamic quantization.
  if (isCompatibleWithNNPALevel(NNPALevel::M15) &&
      (ONNXToZHighLoweringConfiguration::isDynQuant)) {
    // Bump up the pattern benefit to run these before non-quantization
    // patterns.
    PatternBenefit quantPriority(QUANT_PATTERN_BENEFIT);
    if (llvm::any_of(ONNXToZHighLoweringConfiguration::Quant::opTypes,
            [](std::string s) {
              return StringRef(s).equals_insensitive("MatMul");
            })) {
      patterns.insert<replaceONNXMatMulAddByDynQuantI8Pattern>(
          context, quantPriority);
    }
  }

  // Shape inference for newly-added operations.
  getShapeInferencePatterns(patterns);
}

void ONNXToZHighLoweringPass::runOnOperation() {
  ModuleOp module = getOperation();

  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // Enable reporting on NNPA unsupported ops when specifying
  // `--opt-report=NNPAUnsupportedOps`.
  ONNXToZHighLoweringConfiguration::reportOnNNPAUnsupportedOps =
      ONNXToZHighLoweringConfiguration::optReportNNPAUnsupportedOps;

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering.
  target.addLegalDialect<ONNXDialect, zhigh::ZHighDialect, KrnlDialect,
      func::FuncDialect, arith::ArithDialect>();

  // NOTE: if we change the order of calling combinedPatterns and single op
  // patterns, make sure to change the order in DevicePlacement.cpp also to make
  // them synced.

  // Combined ONNX ops to ZHigh lowering.
  // There are some combinations of ONNX ops that can be lowering into a single
  // ZHigh op, e.g. ONNXMatMul and ONNXAdd can be lowered to ZHighMatmul.
  // The lowering of such combinations should be done before the lowering of
  // a single ONNX Op, because the single op lowering might have conditions that
  // prohibit the combined ops lowering happened.
  RewritePatternSet combinedPatterns(&getContext());
  onnx_mlir::getONNXToZHighMultipleOpPatterns(combinedPatterns);

  // It's ok to fail.
  (void)applyPatternsAndFoldGreedily(module, std::move(combinedPatterns));

  // Run the unknown dimension analysis to help check equality of unknown
  // dimensions at compile time.
  onnx_mlir::DimAnalysis dimAnalysis(module);
  dimAnalysis.analyze();

  // Single ONNX to ZHigh operation lowering.
  RewritePatternSet patterns(&getContext());
  onnx_mlir::getONNXToZHighOneOpPatterns(patterns);

  // This is to make sure we don't want to alloc any MemRef at this high-level
  // representation.
  target.addIllegalOp<mlir::memref::AllocOp>();
  target.addIllegalOp<mlir::memref::DeallocOp>();

  // ONNX ops to ZHigh dialect under specific conditions.
  // When adding a new op, need to implement a method, i.e. isSuitableForZDNN,
  // for the op in ONNXLegalityCheck.cpp.
  getONNXToZHighOneOpDynamicallyLegal(&target, &dimAnalysis);

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> createONNXToZHighPass() {
  return std::make_unique<ONNXToZHighLoweringPass>();
}

} // namespace onnx_mlir
