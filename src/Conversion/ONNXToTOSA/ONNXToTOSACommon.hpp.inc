/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ONNXToTOSACommon.hpp - ONNX dialects to TOSA lowering --------===//
//
// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
// Copyright (c) 2023 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file contains common code shared by the functions performing the
// lowering to the TOSA dialect.
//
//===----------------------------------------------------------------------===//

// Common function for lowering reduce operations to TOSA ops.
// Modified from TensorFlow
template <typename T>
std::optional<mlir::Value> convertReduceOpCommon(
    mlir::PatternRewriter &rewriter, mlir::Operation *op,
    mlir::RankedTensorType outputType, mlir::Value inputValue,
    mlir::ElementsAttr axesElems, bool keepDims, mlir::Type reduceElementType) {
  TosaBuilder tosaBuilder(rewriter, op->getLoc());
  mlir::RankedTensorType inputType =
      mlir::dyn_cast<mlir::RankedTensorType>(inputValue.getType());
  if (!inputType)
    return std::nullopt;

  llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
  llvm::ArrayRef<int64_t> outputShape = outputType.getShape();
  auto inputRank = inputShape.size();

  if (axesElems.getNumElements() == 0) {
    // No axes means return the original tensor.
    auto identityOp = onnx_mlir::tosa::CreateOpAndInfer<mlir::tosa::IdentityOp>(
        rewriter, op->getLoc(), outputType, inputValue);
    return identityOp.getResult();
  }
  // Reduce along each axis
  llvm::SmallVector<int64_t> shapeVec(inputShape.begin(), inputShape.end());
  mlir::Value newValue = inputValue;
  for (int i = 0; i < axesElems.getNumElements(); i++) {
    int64_t axisVal = axesElems.getValues<mlir::IntegerAttr>()[i].getInt();
    if (axisVal < 0)
      axisVal += inputRank;
    auto axisAttr = rewriter.getI32IntegerAttr(axisVal);

    shapeVec[axisVal] = 1;
    mlir::RankedTensorType reduceType =
        mlir::RankedTensorType::get(shapeVec, reduceElementType);

    auto reduceOp = CreateOpAndInfer<T>(
        rewriter, op->getLoc(), reduceType, newValue, axisAttr);

    newValue = reduceOp.getResult();
  }

  // Optionally squeeze out the reduced axes.
  if (!keepDims) {
    newValue = tosaBuilder.reshape(newValue, outputShape);
  }
  return newValue;
}

// This calculates the values that need to be added to the padding in order to
// simulate the ceil mode
template <typename ShapeHelperType>
llvm::SmallVector<int64_t> getCeilConstants(llvm::ArrayRef<int64_t> inputShape,
    ONNXGenericPoolOpShapeHelper<ShapeHelperType> &shapeHelper,
    int64_t ceilMode) {
  // This avoids having more if statements when creating the padding const.
  if (ceilMode == 0)
    return llvm::SmallVector<int64_t>{0, 0};

  llvm::SmallVector<int64_t, 2> kernelShapeVec;
  IndexExpr::getLiteral(shapeHelper.kernelShape, kernelShapeVec);

  // Get stride and pad vectors.
  llvm::SmallVector<int64_t, 2> stridesVec = shapeHelper.strides;
  llvm::SmallVector<int64_t, 4> padsVec;
  IndexExpr::getLiteral(shapeHelper.pads, padsVec);

  // Check if the idiv_check for the output dimentions in
  // https://www.mlplatform.org/tosa/tosa_spec.html#_max_pool2d has no
  // remainder. If it has a remainder, we add size(stride) to the end of the
  // padding dimension to get one dimension up. Height and width need to have
  // seperate values.
  int64_t xAxis = 0;
  if ((inputShape[2] + padsVec[0] + padsVec[2] - kernelShapeVec[0]) %
      stridesVec[0])
    xAxis = stridesVec[0];

  int64_t yAxis = 0;
  if ((inputShape[3] + padsVec[1] + padsVec[3] - kernelShapeVec[1]) %
      stridesVec[1])
    yAxis = stridesVec[1];

  return llvm::SmallVector<int64_t>{xAxis, yAxis};
}

// Create an ArrayAttr of pad from \p shapeHelper using \p padIndexOrder.
// Values are calculated considering \p ceilMode.
template <typename ShapeHelperType>
mlir::ArrayAttr createOrderedPadAttrForWindowBasedOps(
    mlir::PatternRewriter &rewriter, const llvm::ArrayRef<int64_t> inputShape,
    ONNXGenericPoolOpShapeHelper<ShapeHelperType> &shapeHelper,
    const int64_t ceilMode, const llvm::ArrayRef<int64_t> padIndexOrder) {

  // When ceil mode is 1, we need to add values to the padding
  const llvm::SmallVector<int64_t, 4> ceilConstants =
      getCeilConstants<ShapeHelperType>(inputShape, shapeHelper, ceilMode);

  // Convert padding to an array
  llvm::SmallVector<int64_t, 4> pads;
  IndexExpr::getLiteral(shapeHelper.pads, pads);

  // Create the right order for the pad according to padIndexOrder
  llvm::SmallVector<int64_t, 4> padOrder;
  for (auto idx : padIndexOrder) {
    padOrder.push_back(pads[idx]);
  }

  // reorder padding according to the passed order and considering ceilMode.
  return rewriter.getI64ArrayAttr({padOrder[0], padOrder[1] + ceilConstants[0],
      padOrder[2], padOrder[3] + ceilConstants[1]});
}

inline mlir::LogicalResult getAvgPool2dAccType(mlir::PatternRewriter &rewriter,
    mlir::Value input, mlir::TypeAttr &accType) {
  auto inputTy = llvm::dyn_cast<mlir::ShapedType>(input.getType());
  if (!inputTy)
    return mlir::failure();
  auto inputETy = inputTy.getElementType();

  if (auto quantType =
          llvm::dyn_cast<mlir::quant::UniformQuantizedType>(inputETy))
    inputETy = quantType.getStorageType();

  // Tosa supports FP16 and FP32 accumulator type for FP16 input. When the time
  // FP16 is supported, the accumulator type can be selected based on trade-off
  // between performance and accuracy. Set to FP32 by default.
  accType = mlir::isa<mlir::FloatType>(inputETy)
                ? mlir::TypeAttr::get(rewriter.getF32Type())
                : mlir::TypeAttr::get(rewriter.getIntegerType(32));

  return mlir::success();
}

// Lower MaxPool and AveragePool to TOSA ops.
template <typename ONNXPoolOp, typename TOSAPoolOp>
mlir::FailureOr<mlir::Value> convertPoolOp(
    mlir::PatternRewriter &rewriter, mlir::Operation *op) {
  using OpAdaptor = typename ONNXPoolOp::Adaptor;
  mlir::Location loc = op->getLoc();
  OpAdaptor adaptor(op->getOperands(), op->getAttrDictionary());
  // Get shape.
  IndexExprBuilderForTosa createTosaIE(rewriter, loc);
  ONNXGenericPoolOpShapeHelper<ONNXPoolOp> shapeHelper(op, {}, &createTosaIE);
  shapeHelper.computeShapeAndAssertOnFailure();

  TosaBuilder tosaBuilder(rewriter, loc);

  mlir::Value input = adaptor.getX();
  auto inputType = mlir::cast<mlir::TensorType>(input.getType());
  if (inputType.getShape().size() != 4) {
    (void)rewriter.notifyMatchFailure(op, "TOSA only supports 2d pooling");
    return mlir::failure();
  }

  auto kernelShape = adaptor.getKernelShapeAttr();

  const int64_t ceilMode = adaptor.getCeilMode();

  // Construct the transposed type for the new Pool OP
  mlir::Type newResultType = mlir::RankedTensorType::get(
      llvm::SmallVector<int64_t, 4>(
          inputType.getShape().size(), mlir::ShapedType::kDynamic),
      inputType.getElementType());

  // ONNX Mlir uses NCHW as an input while TOSA expects NHWC. Insert a
  // transpose to change the format
  input = tosaBuilder.transpose(input, {0, 2, 3, 1});

  if (!IndexExpr::isLiteral(shapeHelper.pads)) {
    (void)rewriter.notifyMatchFailure(op, "could not determine pad values");
    return mlir::failure();
  }
  if (!IndexExpr::isLiteral(shapeHelper.kernelShape)) {
    (void)rewriter.notifyMatchFailure(
        op, "could not determine kernel_shape values");
    return mlir::failure();
  }

  // When ceil mode is 1, we need to add values to the padding
  const llvm::SmallVector<int64_t, 4> ceilConstants =
      getCeilConstants<ONNXPoolOp>(inputType.getShape(), shapeHelper, ceilMode);

  llvm::SmallVector<int64_t, 4> pads;
  IndexExpr::getLiteral(shapeHelper.pads, pads);

  // reorder padding values
  auto newPads = rewriter.getDenseI64ArrayAttr({pads[0],
      pads[2] + ceilConstants[0], pads[1], pads[3] + ceilConstants[1]});

  auto strides = rewriter.getDenseI64ArrayAttr(shapeHelper.strides);

  auto newKernelShape = rewriter.getDenseI64ArrayAttr(
      mlir::extractFromIntegerArrayAttr<int64_t>(kernelShape));

  static_assert(std::is_same<TOSAPoolOp, mlir::tosa::MaxPool2dOp>::value ||
                    std::is_same<TOSAPoolOp, mlir::tosa::AvgPool2dOp>::value,
      "Expected either tosa::MaxPool2dOp or tosa::AvgPool2dOp");
  if constexpr (std::is_same<TOSAPoolOp, mlir::tosa::MaxPool2dOp>::value) {
    input = tosa::CreateOpAndInfer<TOSAPoolOp>(
        rewriter, loc, newResultType, input, newKernelShape, strides, newPads)
                .getResult();
  } else if constexpr (std::is_same<TOSAPoolOp,
                           mlir::tosa::AvgPool2dOp>::value) {
    mlir::TypeAttr accType;
    if (failed(tosa::getAvgPool2dAccType(rewriter, input, accType))) {
      (void)rewriter.notifyMatchFailure(
          op, "Failed to get accumulator type for pooling");
      return mlir::failure();
    }
    input = tosa::CreateOpAndInfer<TOSAPoolOp>(rewriter, loc, newResultType,
        input, newKernelShape, strides, newPads, accType)
                .getResult();
  }

  // Revert to original shape (NCHW)
  // Construct the old result shape out of the new one
  mlir::Value transpose = tosaBuilder.transpose(input, {0, 3, 1, 2});
  return transpose;
};
