/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- ONNXLegalityCheck.cpp - Check legality for ONNX ops -------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains common methods to check whether an ONNX op is suitable for
// being lowered to zDNN or not.
//
//===----------------------------------------------------------------------===//

#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXLegalityCheck.hpp"
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/NNPALimit.h"
#include "src/Conversion/ONNXToKrnl/RNN/RNNBase.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

/// A function to check whether a value's element type is valid for zAIU or not.
/// zAIU supports only F16, F32 and BFLOAT. Since MLIR does not support BFLOAT,
/// we check F16 and F32 here only.
bool isValidElementType(Value val) {
  if (val.getType().isa<NoneType>())
    return true;
  ShapedType valueType = val.getType().dyn_cast_or_null<ShapedType>();
  Type elementType = (valueType) ? valueType.getElementType() : val.getType();
  if (elementType.isa<FloatType>() &&
      (elementType.cast<FloatType>().getWidth() == 16 ||
          elementType.cast<FloatType>().getWidth() == 32))
    return true;
  return false;
}

/// A function to check whether two tensors have the same shape or not.
/// In case where they have the same rank but unknown dimensions, we cannot
/// detect whether the shapes are exactly the same or not. Hence, return false.
/// Also, check the ranks of two tensors, they must be in range of (0, 4].
bool haveSameStaticShape(Value value1, Value value2) {
  ShapedType valueType1 = value1.getType().cast<ShapedType>();
  ShapedType valueType2 = value2.getType().cast<ShapedType>();
  if (!valueType1.hasRank() || !valueType2.hasRank())
    return false;
  // Different rank, return false.
  if (valueType1.getRank() != valueType2.getRank())
    return false;
  // Rank must be in range of (0, 4].
  if (valueType1.getRank() == 0 || valueType1.getRank() > 4)
    return false;
  // Only check when both tensors have static dimensions.
  if (valueType1.hasStaticShape() && valueType2.hasStaticShape())
    return (valueType1.getShape() == valueType2.getShape());
  return false;
}

/// Common legality check for pooling ops
template <typename POOLOP, typename POOLOPAdaptor, typename POOLOPShapeHelper>
bool checkLegalityPoolOpsCommon(POOLOP op, Value Y) {
  POOLOPAdaptor operandAdaptor = POOLOPAdaptor(op);
  POOLOPShapeHelper shapeHelper(&op);
  assert(succeeded(shapeHelper.computeShape(operandAdaptor)) &&
         "Failed to scan POOLOP parameters successfully");
  Value X = op.X();
  int64_t ceilMode = op.ceil_mode();
  ShapedType inputType = X.getType().cast<ShapedType>();
  ShapedType outputType = Y.getType().cast<ShapedType>();
  ArrayRef<int64_t> shapeInput = inputType.getShape();
  ArrayRef<int64_t> shapeOutput = outputType.getShape();

  // 4D tensors(N x C x H x W) are supported as input and output.
  if (shapeInput.size() != 4 || shapeOutput.size() != 4)
    return false;

  // ceil_mode not supported.
  if (ceilMode != 0)
    return false;

  // `getStrPaddingType` returns `SAME_PADDING`, `VALID_PADDING`, or empty.
  // zDNN only support padding for `SAME_PADDING` and `VALID_PADDING`.
  // When input has unknown dimension and auto_pad is `NOTSET`, paddingType is
  // empty.
  StringRef paddingType =
      getStrPaddingType<POOLOP, POOLOPAdaptor, POOLOPShapeHelper>(op);
  if (paddingType.empty())
    return false;

  // Check "MaxPool2D/AvgPool2D Parameter Restrictions". These restrictions are
  // described in "zDNN API Reference". Input tensor N(batchNum) and C(Channel)
  // dimensions must always match the output tensor's respective dimensions.
  if (shapeInput[0] != shapeOutput[0] || shapeInput[1] != shapeOutput[1])
    return false;

  // Check if kernelShape is literal. Only static value is supported.
  if (llvm::any_of(shapeHelper.kernelShape,
          [](IndexExpr val) { return !val.isLiteral(); }))
    return false;

  // Check parameter restrictions for maxpool2d/avgpool2d for each axis only
  // when input and output are of static tensor type. When unknown dimensions
  // are included, the restrictions are not checked and error messages are
  // generated at runtime in zDNN.
  if (inputType.hasStaticShape() && outputType.hasStaticShape()) {
    int64_t inputShapeH = shapeInput[2];
    int64_t inputShapeW = shapeInput[3];
    int64_t outputShapeH = shapeOutput[2];
    int64_t outputShapeW = shapeOutput[3];
    int64_t kernelShapeH = shapeHelper.kernelShape[0].getLiteral();
    int64_t kernelShapeW = shapeHelper.kernelShape[1].getLiteral();
    int64_t stridesH = shapeHelper.strides[0];
    int64_t stridesW = shapeHelper.strides[1];
    bool checkH = meetPoolParamRestrictions(
        inputShapeH, kernelShapeH, stridesH, outputShapeH, paddingType);
    bool checkW = meetPoolParamRestrictions(
        inputShapeW, kernelShapeW, stridesW, outputShapeW, paddingType);
    if (checkH && checkW)
      return true;
    else
      return false;
  } else {
    // No check for tensors with unknown dimensions.
    return true;
  }
}

/// Get padding type using shape helper. This returns
/// `SAME_PADDING`, `VALID_PADDING`, or empty.
template <typename OP, typename OPAdaptor, typename OPShapeHelper>
StringRef getStrPaddingType(OP op) {
  OPAdaptor operandAdaptor = OPAdaptor(op);
  OPShapeHelper shapeHelper(&op);
  assert(succeeded(shapeHelper.computeShape(operandAdaptor)) &&
         "Failed to scan OP parameters successfully");

  auto autoPad = op.auto_pad();
  if (autoPad == "SAME_UPPER")
    return "SAME_PADDING";
  else if (autoPad == "SAME_LOWER")
    // zDNN not support SAME_LOWER.
    return StringRef();
  else if (autoPad == "VALID")
    return "VALID_PADDING";
  else {
    // Only support static pads at this moment being.
    if (llvm::any_of(
            shapeHelper.pads, [](IndexExpr val) { return !val.isLiteral(); }))
      return StringRef();

    // VALID_PADDING.
    if (llvm::all_of(shapeHelper.pads,
            [&](IndexExpr val) { return (val.getLiteral() == 0); }))
      return "VALID_PADDING";

    if (autoPad == "NOTSET") {
      // Pads must be set manually.
      // Check pad values according to SAME_PADDING in zDNN:
      // More information about SAME_PADDING:
      // https://www.pico.net/kb/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-tensorflow/
      // https://github.ibm.com/zosdev/AI-on-Z/wiki/SAME_PADDING
      //
      // height_out = math.ceil(height_in / strides[0])
      // width_out = math.ceil(width_in / strides[1])
      // pad_along_height
      //   = max((height_out - 1) * strides[0] + kernel_size[0] - height_in, 0)
      // pad_along_width
      //   = max((width_out - 1) * strides[1] + kernel_size[1] - width_in, 0)
      // pad_top = pad_along_height // 2
      // pad_bottom = pad_along_height - pad_top
      // pad_left = pad_along_width // 2
      // pad_right = pad_along_width - pad_left

      // Input height and width.
      MemRefBoundsIndexCapture XBounds(op.X());
      IndexExpr hi = XBounds.getDim(2);
      IndexExpr wi = XBounds.getDim(3);
      if (!hi.isLiteral() || !wi.isLiteral())
        return StringRef();
      // Output height and width.
      IndexExpr ho = shapeHelper.dimsForOutput()[2];
      IndexExpr wo = shapeHelper.dimsForOutput()[3];
      if (!ho.isLiteral() || !wo.isLiteral())
        return StringRef();

      // Check if output height and width are correct.
      if (ho.getLiteral() != hi.ceilDiv(shapeHelper.strides[0]).getLiteral())
        return StringRef();
      if (wo.getLiteral() != wi.ceilDiv(shapeHelper.strides[1]).getLiteral())
        return StringRef();

      // Compute pad values according to zDNN.
      LiteralIndexExpr zeroIE(0);
      LiteralIndexExpr twoIE(2);
      IndexExpr padH =
          (ho - 1) * shapeHelper.strides[0] + shapeHelper.kernelShape[0] - hi;
      IndexExpr padW =
          (wo - 1) * shapeHelper.strides[1] + shapeHelper.kernelShape[1] - wi;
      IndexExpr pH = IndexExpr::max(padH, zeroIE);
      IndexExpr pW = IndexExpr::max(padW, zeroIE);
      if (!pH.isLiteral() || !pW.isLiteral())
        return StringRef();

      IndexExpr pHTop = pH.floorDiv(twoIE);
      IndexExpr pHBottom = pH - pHTop;
      IndexExpr pWLeft = pW.floorDiv(twoIE);
      IndexExpr pWRight = pW - pWLeft;

      // Compare ONNX pads and zDNN pads.
      if (pHTop.getLiteral() == shapeHelper.pads[0].getLiteral() &&
          pWLeft.getLiteral() == shapeHelper.pads[1].getLiteral() &&
          pHBottom.getLiteral() == shapeHelper.pads[2].getLiteral() &&
          pWRight.getLiteral() == shapeHelper.pads[3].getLiteral())
        return "SAME_PADDING";
    }
  }

  return StringRef();
}

/// Check if input, output, kernel, strides, and paddingYype for each axis meet
/// parameter restrictions for maxpool. See "MaxPool2D Parameter Restrictions"
/// in "zDNN API Reference"
bool meetPoolParamRestrictions(int64_t inputShape, int64_t kernelShape,
    int64_t strides, int64_t outputShape, StringRef paddingType) {
  // TODO: Shape inference fails when `strides` is zero.
  // (third_party/onnx-mlir/src/Dialect/ONNX/ONNXOps.cpp:L204). So strides==0
  // case is not tested. Need to investigate how to handle this.
  if (strides == 0) {
    // Both input tensor's Height/Width dimension and the kernel_height/width
    // must match
    if (inputShape != kernelShape)
      return false;
    // inputShape and kernelShape are less than or equal to 1024.
    if (inputShape > 1024)
      return false;
    // Output tensor's height and width dimensions must be 1.
    if (outputShape != 1)
      return false;
    // padding_type must be VALID_PADDING.
    if (!paddingType.equals("VALID_PADDING"))
      return false;
  } else {
    // strides are greater than zero
    // kernel_width and kernel_height must be less than or equal to 64.
    if (kernelShape > 64)
      return false;
    if (paddingType.equals("SAME_PADDING")) {
      if (outputShape != ceil((float)inputShape / strides))
        return false;
    } else { // VALID_PADDING
      if (outputShape != ceil((float)(inputShape - kernelShape + 1) / strides))
        return false;
    }
  }
  return true;
}

/// Default legality check.
template <typename OP_TYPE>
bool isSuitableForZDNN(OP_TYPE op) {
  return false;
}

/// Check legality for ONNXAdd.
// zDNN Add, Sub, Mul, Div do not support broadcasting.
template <>
bool isSuitableForZDNN<ONNXAddOp>(ONNXAddOp op) {
  if (!isValidElementType(op.A()))
    return false;
  if (!isValidElementType(op.B()))
    return false;
  return haveSameStaticShape(op.A(), op.B());
}

/// Check legality for ONNXSub.
template <>
bool isSuitableForZDNN<ONNXSubOp>(ONNXSubOp op) {
  if (!isValidElementType(op.A()))
    return false;
  if (!isValidElementType(op.B()))
    return false;
  return haveSameStaticShape(op.A(), op.B());
}

/// Check legality for ONNXMul.
template <>
bool isSuitableForZDNN<ONNXMulOp>(ONNXMulOp op) {
  if (!isValidElementType(op.A()))
    return false;
  if (!isValidElementType(op.B()))
    return false;
  return haveSameStaticShape(op.A(), op.B());
}

/// Check legality for ONNXDiv.
template <>
bool isSuitableForZDNN<ONNXDivOp>(ONNXDivOp op) {
  if (!isValidElementType(op.A()))
    return false;
  if (!isValidElementType(op.B()))
    return false;
  return haveSameStaticShape(op.A(), op.B());
}

/// Check legality for ONNXSum.
template <>
bool isSuitableForZDNN<ONNXSumOp>(ONNXSumOp op) {
  // Do not support a single input.
  if (op.data_0().size() < 2)
    return false;
  // Check data type.
  if (!isValidElementType(op.data_0()[0]))
    return false;
  // All inputs must have the same static shape.
  for (unsigned int i = 1; i < op.data_0().size(); ++i) {
    // Check data type.
    if (!isValidElementType(op.data_0()[i]))
      return false;
    if (!haveSameStaticShape(op.data_0()[0], op.data_0()[i]))
      return false;
  }
  return true;
}

/// Check legality for ONNXMin.
/// zDNN Min/Max do not support boradcasting, and getNumOperands != 2.
template <>
bool isSuitableForZDNN<ONNXMinOp>(ONNXMinOp op) {
  int64_t opnum = op.getNumOperands();
  if (opnum != 2) {
    return false;
  }
  if (!isValidElementType(op.getOperand(0)))
    return false;
  if (!isValidElementType(op.getOperand(1)))
    return false;
  return haveSameStaticShape(op.getOperand(0), op.getOperand(1));
}

/// Check legality for ONNXMax.
/// zDNN Min/Max do not support boradcasting, and getNumOperands != 2.
template <>
bool isSuitableForZDNN<ONNXMaxOp>(ONNXMaxOp op) {
  int64_t opnum = op.getNumOperands();
  if (opnum != 2) {
    return false;
  }
  if (!isValidElementType(op.getOperand(0)))
    return false;
  if (!isValidElementType(op.getOperand(1)))
    return false;
  return haveSameStaticShape(op.getOperand(0), op.getOperand(1));
}

/// Check legality for ONNXSoftmax.
/// zDNN softmax only supports axis = 1 (or -1 when rank = 2). If axis is not
/// 1 (or -1 when rank = 2), keep ONNXSoftmax unchanged.
/// TODO: support rank != 2.
template <>
bool isSuitableForZDNN<ONNXSoftmaxOp>(ONNXSoftmaxOp op) {
  if (!isValidElementType(op.input()))
    return false;
  ShapedType inputType = op.getType().cast<ShapedType>();
  return (op.axis() == 1 || op.axis() == -1) && inputType.hasRank() &&
         (inputType.getRank() == 2);
}

/// Check legality for ONNXRelu.
template <>
bool isSuitableForZDNN<ONNXReluOp>(ONNXReluOp op) {
  if (!isValidElementType(op.X()))
    return false;
  ShapedType xType = op.X().getType().cast<ShapedType>();
  return xType.hasRank() && (xType.getRank() <= 4);
}

/// Check legality for ONNXTanh.
template <>
bool isSuitableForZDNN<ONNXTanhOp>(ONNXTanhOp op) {
  if (!isValidElementType(op.input()))
    return false;
  ShapedType inputType = op.getType().cast<ShapedType>();
  return inputType.hasRank() && (inputType.getRank() <= 4);
}

/// Check legality for ONNXSigmoid.
template <>
bool isSuitableForZDNN<ONNXSigmoidOp>(ONNXSigmoidOp op) {
  if (!isValidElementType(op.X()))
    return false;
  ShapedType xType = op.X().getType().cast<ShapedType>();
  return xType.hasRank() && (xType.getRank() <= 4);
}

/// Check legality for ONNXLog.
template <>
bool isSuitableForZDNN<ONNXLogOp>(ONNXLogOp op) {
  if (!isValidElementType(op.input()))
    return false;
  ShapedType inputType = op.input().getType().cast<ShapedType>();
  return inputType.hasRank() && (inputType.getRank() <= 4);
}

/// Check legality for ONNXExp.
template <>
bool isSuitableForZDNN<ONNXExpOp>(ONNXExpOp op) {
  if (!isValidElementType(op.input()))
    return false;
  ShapedType inputType = op.input().getType().cast<ShapedType>();
  return inputType.hasRank() && (inputType.getRank() <= 4);
}

/// Check legality for ONNXMatMul.
template <>
bool isSuitableForZDNN<ONNXMatMulOp>(ONNXMatMulOp op) {
  int64_t opnum = op.getNumOperands();
  if (opnum != 2) {
    return false;
  }
  if (!isValidElementType(op.getOperand(0)))
    return false;
  if (!isValidElementType(op.getOperand(1)))
    return false;
  ShapedType aType = op.getOperand(0).getType().cast<ShapedType>();
  ShapedType bType = op.getOperand(1).getType().cast<ShapedType>();

  // Illegal if A or B is unranked.
  if (!aType.hasRank() || !bType.hasRank())
    return false;

  auto shapeA = aType.getShape();
  auto shapeB = bType.getShape();

  // In case of Tensors with unknown dimension, check only size of matrices.
  // Actual shape is not checked. If actual shape does not meet, get error at
  // runtime.
  // TODO: Support other cases
  // (https://github.com/onnx/onnx/blob/main/docs/Operators.md#MatMul) on zDNN
  // by using broadcasting etc.
  if ((shapeA.size() == 2) && (shapeB.size() == 2)) {
    // unstacked case
    if (aType.hasStaticShape() && bType.hasStaticShape())
      return (shapeA[1] == shapeB[0]);
    else
      return true;
  } else if ((shapeA.size() == 3) && (shapeB.size() == 3)) {
    // stacked w/o bcast case
    if (aType.hasStaticShape() && bType.hasStaticShape())
      return ((shapeA[0] == shapeB[0]) && (shapeA[2] == shapeB[1]));
    else
      return true;
  } else if ((shapeA.size() == 3) && (shapeB.size() == 2)) {
    // stacked w/ bcast
    if (aType.hasStaticShape() && bType.hasStaticShape())
      return (shapeA[2] == shapeB[0]);
    else
      return true;
  }
  return false; // unsupported case
}

/// Check legality for ONNXGemm.
template <>
bool isSuitableForZDNN<ONNXGemmOp>(ONNXGemmOp op) {
  Value A = op.A();
  Value B = op.B();
  Value C = op.C();

  // Check data type.
  if (!isValidElementType(A))
    return false;
  if (!isValidElementType(B))
    return false;
  if (!isValidElementType(C))
    return false;

  ShapedType aType = A.getType().cast<ShapedType>();
  ShapedType bType = B.getType().cast<ShapedType>();
  ShapedType cType;
  ArrayRef<int64_t> aShape = aType.getShape();
  ArrayRef<int64_t> bShape = bType.getShape();
  ArrayRef<int64_t> cShape;

  bool hasC = !isNoneType(C);
  if (hasC) {
    cType = C.getType().cast<ShapedType>();
    cShape = cType.getShape();
  }

  // Element type must be f32.
  if (!aType.getElementType().isF32() || !bType.getElementType().isF32() ||
      (hasC && !cType.getElementType().isF32()))
    return false;
  // A and B's rank must be 2 and C's rank must be 1 or 2.
  if ((aShape.size() != 2) || (bShape.size() != 2) ||
      (hasC && (cShape.size() != 1) && (cShape.size() != 2)))
    return false;

  ONNXGemmOp gemmOp = llvm::cast<ONNXGemmOp>(op);
  if ((gemmOp.alpha().convertToFloat() != 1.0) ||
      (gemmOp.beta().convertToFloat() != 1.0)) {
    return false;
  }
  auto bShape1 = gemmOp.transB() ? bShape[0] : bShape[1];
  // If C's rank is 1: Only support B's second dim is the same with C's dim
  // (A(m, n) * B(n, p) + C(p))
  if (hasC && cShape.size() == 1) {
    // Cannot check broadcasting at compile time.
    if (cShape[0] == -1)
      return false;
    if (cShape[0] != bShape1)
      return false;
  }
  return true;
}

/// Check legality for ONNXReduceMean.
template <>
bool isSuitableForZDNN<ONNXReduceMeanOp>(ONNXReduceMeanOp op) {
  // Check data type.
  if (!isValidElementType(op.data()))
    return false;

  llvm::Optional<mlir::ArrayAttr> axes = op.axes();
  int64_t keepdims = op.keepdims();
  ShapedType dataType = op.data().getType().cast<ShapedType>();
  auto shapeData = dataType.getShape();

  // Check keepdims.
  if ((shapeData.size() != 4) || (keepdims == 0) || !axes)
    return false;

  // Check axes.
  mlir::ArrayAttr axesVal = axes.getValue();
  SmallVector<Attribute> axesAttrs(axesVal.begin(), axesVal.end());
  if ((axesAttrs.size() != 2) ||
      (axesAttrs[0].dyn_cast<IntegerAttr>().getInt() != 2) ||
      (axesAttrs[1].dyn_cast<IntegerAttr>().getInt() != 3)) {
    return false;
  }

  // Check dimensions.
  if ((shapeData[2] < 0) || (shapeData[3] < 0) || (shapeData[2] > 1024) ||
      (shapeData[3] > 1024))
    return false;

  return true;
}

/// Check legality for ONNXLSTM.
/// TODO: current ONNX-to-zhigh conversion does not support bi-direction
template <>
bool isSuitableForZDNN<ONNXLSTMOp>(ONNXLSTMOp op) {
  StringRef direction = op.direction();
  Value W = op.W();
  Value R = op.R();
  Value B = op.B();

  // Check direction.
  if ((direction != FORWARD) && (direction != REVERSE) &&
      (direction != BIDIRECTIONAL))
    return false;

  // Check data type.
  if (!isValidElementType(W))
    return false;
  if (!isValidElementType(R))
    return false;
  if (!isValidElementType(B))
    return false;

  int64_t hidden_size = R.getType().cast<ShapedType>().getShape()[2];
  llvm::Optional<ArrayAttr> activations = op.activations();
  // Check if direction and hidden_size in W have static dimensions.
  ArrayRef<int64_t> wShape = W.getType().cast<ShapedType>().getShape();
  if ((wShape[0] != 1 && wShape[0] != 2) || wShape[1] < 0)
    return false;
  // Check if R has static dimensions, and the direction dim is 1 or 2.
  ArrayRef<int64_t> rShape = R.getType().cast<ShapedType>().getShape();
  if (!R.getType().cast<ShapedType>().hasStaticShape() ||
      (rShape[0] != 1 && rShape[0] != 2))
    return false;
  // Check hidden_size.
  if (hidden_size > MAXIMUM_NUM_HIDDEN_SIZE_LSTM)
    return false;
  // zDNN does not support sequence_lens.
  if (!isNoneType(op.sequence_lens()))
    return false;
  // check if B, initial_h and initial_c have static dimensions if given.
  if (!isNoneType(B) && !B.getType().cast<ShapedType>().hasStaticShape())
    return false;
  // check if B's direction dim is 1 or 2.
  if (!isNoneType(B)) {
    ArrayRef<int64_t> bShape = B.getType().cast<ShapedType>().getShape();
    if (bShape[0] != 1 && bShape[0] != 2)
      return false;
  }
  // zDNN does not support P(peepholes), activation_alpha and activation_beta.
  if (!isNoneType(op.P()) || op.activation_alpha() || op.activation_beta())
    return false;
  // zDNN support the default activations (["Sigmoid", "Tanh", "Tanh"]) only.
  if ((activations && (activations.getValue().size() > 0) &&
          (activations.getValue()[0].cast<StringAttr>().getValue() !=
              "Sigmoid")) ||
      (activations && (activations.getValue().size() > 1) &&
          (activations.getValue()[1].cast<StringAttr>().getValue() !=
              "Tanh")) ||
      (activations && (activations.getValue().size() > 2) &&
          (activations.getValue()[2].cast<StringAttr>().getValue() != "Tanh")))
    return false;
  // zDNN does not supprt clip(Cell clip threshold).
  if (op.clip())
    return false;
  // zDNN does not support hidden_size not equal to the hidden size in
  // other inputs.
  if (op.hidden_size() && (op.hidden_size().getValue() != hidden_size))
    return false;
  // zDNN does not support input_forget.
  if (op.input_forget() != 0)
    return false;
  return true;
}

/// Check legality for ONNXGRU.
/// TODO: current ONNX-to-zhigh conversion does not support bi-direction
template <>
bool isSuitableForZDNN<ONNXGRUOp>(ONNXGRUOp op) {
  StringRef direction = op.direction();
  Value W = op.W();
  Value R = op.R();
  Value B = op.B();

  // Check direction.
  if ((direction != FORWARD) && (direction != REVERSE) &&
      (direction != BIDIRECTIONAL))
    return false;

  // Check data type.
  if (!isValidElementType(W))
    return false;
  if (!isValidElementType(R))
    return false;
  if (!isValidElementType(B))
    return false;

  int64_t hidden_size = R.getType().cast<ShapedType>().getShape()[2];
  llvm::Optional<ArrayAttr> activations = op.activations();
  // Check if direction and hidden_size in W have static dimensions.
  ArrayRef<int64_t> wShape = W.getType().cast<ShapedType>().getShape();
  if ((wShape[0] != 1 && wShape[0] != 2) || wShape[1] < 0)
    return false;
  // Check if R has static dimensions.
  if (!R.getType().cast<ShapedType>().hasStaticShape())
    return false;
  // Check hidden_size.
  if (hidden_size > MAXIMUM_NUM_HIDDEN_SIZE_GRU)
    return false;
  // zDNN does not support sequence_lens.
  if (!isNoneType(op.sequence_lens()))
    return false;
  // check if B and initial_h have static dimensions if given.
  if (!isNoneType(B) && !B.getType().cast<ShapedType>().hasStaticShape())
    return false;
  // check if B's direction dim is 1 or 2.
  if (!isNoneType(B)) {
    ArrayRef<int64_t> bShape = B.getType().cast<ShapedType>().getShape();
    if (bShape[0] != 1 && bShape[0] != 2)
      return false;
  }
  // zDNN does not support activation_alpha and activation_beta.
  if (op.activation_alpha() || op.activation_beta())
    return false;
  // zDNN support the default activations (["Sigmoid", "Tanh", "Tanh"]) only.
  if ((activations && (activations.getValue().size() > 0) &&
          (activations.getValue()[0].cast<StringAttr>().getValue() !=
              "Sigmoid")) ||
      (activations && (activations.getValue().size() > 1) &&
          (activations.getValue()[1].cast<StringAttr>().getValue() !=
              "Tanh")) ||
      (activations && (activations.getValue().size() > 2) &&
          (activations.getValue()[2].cast<StringAttr>().getValue() != "Tanh")))
    return false;
  // zDNN does not supprt clip(Cell clip threshold).
  if (op.clip())
    return false;
  // zDNN does not support hidden_size not equal to the hidden size in
  // other inputs.
  if (op.hidden_size() && (op.hidden_size().getValue() != hidden_size))
    return false;
  // zDNN support the "linear_before_reset==1" case only.
  if (op.linear_before_reset() != 1)
    return false;
  return true;
}

/// Check legality for ONNXMaxpool.
template <>
bool isSuitableForZDNN<ONNXMaxPoolSingleOutOp>(ONNXMaxPoolSingleOutOp op) {
  // Check data type.
  if (!isValidElementType(op.X()))
    return false;

  ONNXMaxPoolSingleOutOpAdaptor operandAdaptor =
      ONNXMaxPoolSingleOutOpAdaptor(op);
  ONNXMaxPoolSingleOutOpShapeHelper shapeHelper(&op);
  assert(succeeded(shapeHelper.computeShape(operandAdaptor)) &&
         "Failed to scan ONNXMaxPoolSingleOutOp parameters successfully");

  if (!checkLegalityPoolOpsCommon<ONNXMaxPoolSingleOutOp,
          ONNXMaxPoolSingleOutOpAdaptor, ONNXMaxPoolSingleOutOpShapeHelper>(
          op, op.o_Y()))
    return false;

  // dilations not supported. Only default one is accepted.
  if (shapeHelper.dilations[0] != 1 || shapeHelper.dilations[1] != 1)
    return false;

  return true;
}

/// Check legality for ONNXAveragePool.
template <>
bool isSuitableForZDNN<ONNXAveragePoolOp>(ONNXAveragePoolOp op) {
  // Check data type.
  if (!isValidElementType(op.X()))
    return false;

  // count_include_pad not supported.
  if (op.count_include_pad() != 0)
    return false;

  return checkLegalityPoolOpsCommon<ONNXAveragePoolOp, ONNXAveragePoolOpAdaptor,
      ONNXAveragePoolOpShapeHelper>(op, op.Y());
}

/// Check if input, output, kernel, strides, and paddingType for each axis meet
/// parameter restrictions for conv2d. See "Conv2D Parameter Restrictions"
/// in "zDNN API Reference"
static bool checkConv2DParamRestrictions(int64_t inputDim, int64_t kernelDim,
    int64_t stride, int64_t outputDim, StringRef paddingType) {
  if (stride == 0) {
    // paddingType must be VALID_PADDING.
    if (!paddingType.equals("VALID_PADDING"))
      return false;
    // inputDim must be = kernel dim.
    if (inputDim != kernelDim)
      return false;
    // inputDim and kernelDim are less than or equal to 448.
    if (inputDim > 448)
      return false;
    // outputDim must be 1.
    if (outputDim != 1)
      return false;
  } else if (stride > 0 && stride <= 13) {
    // stride is greater than zero and less than or equal to 13.
    // kernel dim must be less than or equal to 64.
    if (kernelDim > 64)
      return false;
    if (paddingType.equals("SAME_PADDING")) {
      // height_out restriction.
      if (outputDim != ceil((float)inputDim / stride))
        return false;
    } else { // VALID_PADDING
      // inputDim must be >= kernelDim.
      if (inputDim < kernelDim)
        return false;
      // height_out restriction.
      if (outputDim != ceil((float)(inputDim - kernelDim + 1) / stride))
        return false;
    }
  } else
    return false;

  return true;
}

/// Check legality for ONNXConvOp.
template <>
bool isSuitableForZDNN<ONNXConvOp>(ONNXConvOp op) {
  // Check data type.
  if (!isValidElementType(op.X()))
    return false;
  if (!isValidElementType(op.W()))
    return false;
  if (!isValidElementType(op.B()))
    return false;

  ONNXConvOpAdaptor operandAdaptor = ONNXConvOpAdaptor(op);
  ONNXConvOpShapeHelper shapeHelper(&op);
  assert(succeeded(shapeHelper.computeShape(operandAdaptor)) &&
         "Failed to scan Conv parameters successfully");

  ShapedType inputType = op.X().getType().cast<ShapedType>();
  ShapedType outputType = op.Y().getType().cast<ShapedType>();
  ArrayRef<int64_t> shapeInput = inputType.getShape();
  ArrayRef<int64_t> shapeOutput = outputType.getShape();

  // Do not support dynamic height and weight dimensions since we can not check
  // them at compile time.
  if (shapeInput[2] == -1 || shapeInput[3] == -1 || shapeOutput[2] == -1 ||
      shapeOutput[3] == -1)
    return false;

  // Do not support group.
  if (operandAdaptor.group() != 1)
    return false;

  // 4D tensors(N x C x H x W) are supported as input and output.
  if (shapeInput.size() != 4 || shapeOutput.size() != 4)
    return false;

  // Do not support non-default dilations.
  if (shapeHelper.dilations[0] != 1 || shapeHelper.dilations[1] != 1)
    return false;

  // `getStrPaddingType` returns `SAME_PADDING`, `VALID_PADDING`, or empty.
  // `zdnn_conv2d` only support padding for `SAME_PADDING` and `VALID_PADDING`.
  StringRef paddingType =
      getStrPaddingType<ONNXConvOp, ONNXConvOpAdaptor, ONNXConvOpShapeHelper>(
          op);

  if (paddingType.empty())
    return false;

  // Check if kernelShape is literal. Only static value is supported.
  if (llvm::any_of(shapeHelper.kernelShape,
          [](IndexExpr val) { return !val.isLiteral(); }))
    return false;

  int64_t inputShapeH = shapeInput[2];
  int64_t inputShapeW = shapeInput[3];
  int64_t outputShapeH = shapeOutput[2];
  int64_t outputShapeW = shapeOutput[3];
  int64_t kernelShapeH = shapeHelper.kernelShape[0].getLiteral();
  int64_t kernelShapeW = shapeHelper.kernelShape[1].getLiteral();
  int64_t stridesH = shapeHelper.strides[0];
  int64_t stridesW = shapeHelper.strides[1];

  // Check parameter restrictions for conv2d for each axis.
  bool isHOK = checkConv2DParamRestrictions(
      inputShapeH, kernelShapeH, stridesH, outputShapeH, paddingType);
  if (!isHOK)
    return false;
  bool isWOK = checkConv2DParamRestrictions(
      inputShapeW, kernelShapeW, stridesW, outputShapeW, paddingType);
  if (!isWOK)
    return false;

  return true;
}

/// Check legality for ONNXBatchNormOp.
template <>
bool isSuitableForZDNN<ONNXBatchNormalizationInferenceModeOp>(
    ONNXBatchNormalizationInferenceModeOp op) {
  ShapedType inputType = op.X().getType().cast<ShapedType>();
  ShapedType outputType = op.o_Y().getType().cast<ShapedType>();
  ArrayRef<int64_t> shapeInput = inputType.getShape();
  ArrayRef<int64_t> shapeOutput = outputType.getShape();

  // 4D tensors(N x C x H x W) are supported as input and output.
  if (shapeInput.size() != 4 || shapeOutput.size() != 4)
    return false;

  return true;
}
