/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- ZLowToLLVM.cpp - Lowering from ZLow to LLVM ---------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file defines patterns to lower ZLow operations to LLVM dialect.
//
// Note that once a type is lowed to LLVM, it can be opaque pointer and its
// element type is lost. Thus, to get element type in LLVM, get it from the
// original operation (not via operandAdaptor), then use
// `typeConverter(elementType)` to convert it to LLVM. See `ZLowStickLowering`
// as an example.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"

#include "src/Accelerators/NNPA/Conversion/ZLowToLLVM/ZLowToLLVM.hpp"
#include "src/Accelerators/NNPA/Conversion/ZLowToLLVM/ZLowToLLVMCommon.hpp"
#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp"
#include "src/Accelerators/NNPA/Support/LayoutHelper.hpp"
#include "src/Accelerators/NNPA/Support/NNPALimit.hpp"
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "zdnn.h"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
namespace zlow {

static bool FUNC_CALL_FOR_DLF16_CONVERSION = false;
static bool SIMD_FOR_DLF16_CONVERSION = true;

zdnn_data_layouts UNDEFINED_ZDNN_LAYOUT = static_cast<zdnn_data_layouts>(255);

// Obtain a zDNN API for an elementwise ZLow operation.
template <>
API APIFor<ZLowAddOp>() {
  return API::ZDNN_ADD;
}
template <>
API APIFor<ZLowSubOp>() {
  return API::ZDNN_SUB;
}
template <>
API APIFor<ZLowMulOp>() {
  return API::ZDNN_MUL;
}
template <>
API APIFor<ZLowDivOp>() {
  return API::ZDNN_DIV;
}
template <>
API APIFor<ZLowMinOp>() {
  return API::ZDNN_MIN;
}
template <>
API APIFor<ZLowMaxOp>() {
  return API::ZDNN_MAX;
}
template <>
API APIFor<ZLowLogOp>() {
  return API::ZDNN_LOG;
}
template <>
API APIFor<ZLowExpOp>() {
  return API::ZDNN_EXP;
}
template <>
API APIFor<ZLowInvSqrtOp>() {
  return API::ZDNN_INVSQRT;
}
template <>
API APIFor<ZLowReluOp>() {
  return API::ZDNN_RELU;
}
template <>
API APIFor<ZLowGeluOp>() {
  return API::ZDNN_GELU;
}
template <>
API APIFor<ZLowTanhOp>() {
  return API::ZDNN_TANH;
}
template <>
API APIFor<ZLowSigmoidOp>() {
  return API::ZDNN_SIGMOID;
}

template <>
API APIFor<ZLowSqrtOp>() {
  return API::ZDNN_SQRT;
}

class ZLowStickLowering : public mlir::ConvertToLLVMPattern {
public:
  explicit ZLowStickLowering(MLIRContext *context, LLVMTypeConverter &lowering_,
      ApiRegistry apiRegistry)
      : ConvertToLLVMPattern(
            ZLowStickOp::getOperationName(), context, lowering_) {
    this->apiRegistry = apiRegistry;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();
    ZLowStickOp stickOp = mlir::cast<ZLowStickOp>(op);
    std::optional<int64_t> saturationOpt = stickOp.getSaturation();
    bool saturation = saturationOpt.has_value() && saturationOpt.value() != 0;

    ZLowStickOpAdaptor operandAdaptor(operands);
    // Do not get element type from adaptor since the type can be opaque.
    Type llvmElementTy = typeConverter->convertType(
        mlir::cast<MemRefType>(stickOp.getX().getType()).getElementType());

    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);

    // Get the dimensions of the original shape (the shape before stickifying)
    // used for creating a zTensor. For 'zLow.stick', the original shape is
    // obtained from the first argument.
    SmallVector<Value, 3> dims;
    getDimsFromMemRef(rewriter, loc, module, operandAdaptor.getX(), dims);

    // Get zDNN data type.
    zdnn_data_types zDNNDataType = llvmTypeToZDNNType(llvmElementTy);

    // Get zDNN data layout.
    zdnn_data_layouts zDNNDataLayout =
        convertLayoutAttrToZDNNDataLayout(dims.size(), stickOp.getLayoutAttr());

    // Create a zTensor.
    Value stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getOut());
    ZTensor zTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/zDNNDataLayout, /*originalDims=*/dims,
            /*isTransformed=*/false);

    // Ready to stickify.
    Value unstickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getX());
    if (saturation)
      callApi(rewriter, loc, module, apiRegistry,
          API::ZDNN_TRANSFORM_ZTENSOR_WITH_SATURATION,
          {toOpaquePtr(rewriter, loc, module, zTensor.val), unstickI8Ptr});
    else
      callApi(rewriter, loc, module, apiRegistry, API::ZDNN_TRANSFORM_ZTENSOR,
          {toOpaquePtr(rewriter, loc, module, zTensor.val), unstickI8Ptr});

    rewriter.eraseOp(op);
    return success();
  }

private:
  ApiRegistry apiRegistry;
};

class ZLowQuantizedStickLowering : public mlir::ConvertToLLVMPattern {
public:
  explicit ZLowQuantizedStickLowering(MLIRContext *context,
      LLVMTypeConverter &lowering_, ApiRegistry apiRegistry)
      : ConvertToLLVMPattern(
            ZLowQuantizedStickOp::getOperationName(), context, lowering_) {
    this->apiRegistry = apiRegistry;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

    ZLowQuantizedStickOp stickOp = cast<ZLowQuantizedStickOp>(op);
    ZLowQuantizedStickOpAdaptor operandAdaptor(operands);
    Value recScale = operandAdaptor.getRecScale();
    Value offset = operandAdaptor.getOffset();
    StringRef transformTypeStr = stickOp.getQType();

    // Do not get element type from adaptor since the type can be opaque.
    Type llvmElementTy = typeConverter->convertType(
        mlir::cast<MemRefType>(stickOp.getX().getType()).getElementType());
    Type llvmI64Ty = rewriter.getI64Type();
    Type llvmF32Ty = rewriter.getF32Type();

    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);

    // Get the dimensions of the original shape (the shape before stickifying)
    // used for creating a zTensor. For 'zLow.quantizedStick', the original
    // shape is obtained from the first argument.
    SmallVector<Value, 3> dims;
    getDimsFromMemRef(rewriter, loc, module, operandAdaptor.getX(), dims);

    // Get zDNN data type.
    zdnn_data_types zDNNDataType = llvmTypeToZDNNType(llvmElementTy);

    // Get zDNN data layout.
    zdnn_data_layouts zDNNDataLayout =
        convertLayoutAttrToZDNNDataLayout(dims.size(), stickOp.getLayoutAttr());

    // Get zDNN transform type.
    zdnn_quantized_transform_types transformType =
        getQuantizedTransformType(transformTypeStr);

    // Create a zTensor.
    Value stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getOut());
    Value recScaleF32 = loadFromMemRef(create.llvm, llvmF32Ty, recScale, 0);
    Value offsetF32 = loadFromMemRef(create.llvm, llvmF32Ty, offset, 0);
    ZTensor zTensor =
        zTensorHelper.getQuantizedZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/zDNNDataLayout, /*transformType=*/transformType,
            /*originalDims=*/dims, /*recScale=*/recScaleF32,
            /*offset=*/offsetF32,
            /*isTransformed=*/false);

    // Always saturate.
    Value saturationVal =
        create.llvm.constant(llvmI64Ty, static_cast<int64_t>(1));

    // Min, Max clip values.
    Value clipMIN =
        create.llvm.constant(llvmI64Ty, static_cast<int64_t>(INT8_MIN));
    Value clipMAX =
        create.llvm.constant(llvmI64Ty, static_cast<int64_t>(INT8_MAX));

    // Ready to stickify.
    Value unstickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getX());
    callApi(rewriter, loc, module, apiRegistry,
        API::ZDNN_TRANSFORM_QUANTIZED_ZTENSOR,
        {toOpaquePtr(rewriter, loc, module, zTensor.val), saturationVal,
            clipMIN, clipMAX, unstickI8Ptr});

    rewriter.eraseOp(op);
    return success();
  }

private:
  ApiRegistry apiRegistry;
};

class ZLowStickForLSTMLowering : public ConvertToLLVMPattern {
public:
  explicit ZLowStickForLSTMLowering(MLIRContext *context,
      LLVMTypeConverter &lowering_, ApiRegistry apiRegistry)
      : ConvertToLLVMPattern(
            ZLowStickForLSTMOp::getOperationName(), context, lowering_) {
    this->apiRegistry = apiRegistry;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();
    ZLowStickForLSTMOp stickForLSTMOp = mlir::cast<ZLowStickForLSTMOp>(op);

    ZLowStickForLSTMOpAdaptor operandAdaptor(operands);
    Type llvmElementTy = typeConverter->convertType(
        mlir::cast<MemRefType>(stickForLSTMOp.getFGate().getType())
            .getElementType());

    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);

    // Get the dimensions of the original shape (the shape before stickifying)
    // used for creating a zTensor. F, I, C, O gates have the same shape. Thus,
    // they share the dimensions.
    SmallVector<Value, 3> dims;
    getDimsFromMemRef(rewriter, loc, module, operandAdaptor.getFGate(), dims);

    // Get zDNN data type.
    zdnn_data_types zDNNDataType = llvmTypeToZDNNType(llvmElementTy);

    // Get zDNN data layout and concatInfo
    zdnn_data_layouts zDNNDataLayout;
    zdnn_concat_info zDNNConcatInfo;
    StringRef prevLayerStr = stickForLSTMOp.getPrevLayer();
    int64_t prevLayer = -1;
    if (prevLayerStr.equals_insensitive("none")) {
      prevLayer = PREV_LAYER_NONE;
    } else if (prevLayerStr.equals_insensitive("uni")) {
      prevLayer = PREV_LAYER_UNI;
    } else if (prevLayerStr.equals_insensitive("bidir")) {
      prevLayer = PREV_LAYER_BIDIR;
    }
    assert((prevLayer >= 0) &&
           "invalid prev_layer attribute in zlow.StickForLSTM");

    if (dims.size() == 2) {
      // for stickify input/hidden biases.
      zDNNDataLayout = ZDNN_2DS;
      zDNNConcatInfo = RNN_TYPE_LSTM | USAGE_BIASES | prevLayer;
    } else if (dims.size() == 3) {
      // for stickify input/hidden weights.
      zDNNDataLayout = ZDNN_3DS;
      zDNNConcatInfo = RNN_TYPE_LSTM | USAGE_WEIGHTS | prevLayer;
    } else {
      // Set invalid value to avoid uninitvar cppcheck warning.
      zDNNDataLayout = UNDEFINED_ZDNN_LAYOUT;
      llvm_unreachable("Unsupported layout");
    }

    // Create a zTensor.
    Value stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getOut());
    ZTensor zTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/zDNNDataLayout, /*originalDims=*/dims,
            /*isTransformed=*/false, /*isConcat=*/true,
            /*concatInfo=*/zDNNConcatInfo);

    // Ready to stickify.
    Value fGatePtr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getFGate());
    Value iGatePtr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getIGate());
    Value cGatePtr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getCGate());
    Value oGatePtr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getOGate());
    callApi(rewriter, loc, module, apiRegistry, API::ZDNN_TRANSFORM_ZTENSOR,
        {toOpaquePtr(rewriter, loc, module, zTensor.val), fGatePtr, iGatePtr,
            cGatePtr, oGatePtr});

    rewriter.eraseOp(op);
    return success();
  }

private:
  ApiRegistry apiRegistry;
};

class ZLowStickForGRULowering : public ConvertToLLVMPattern {
public:
  explicit ZLowStickForGRULowering(MLIRContext *context,
      LLVMTypeConverter &lowering_, ApiRegistry apiRegistry)
      : ConvertToLLVMPattern(
            ZLowStickForGRUOp::getOperationName(), context, lowering_) {
    this->apiRegistry = apiRegistry;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();
    ZLowStickForGRUOp stickForGRUOp = mlir::cast<ZLowStickForGRUOp>(op);

    ZLowStickForGRUOpAdaptor operandAdaptor(operands);
    Type llvmElementTy = typeConverter->convertType(
        mlir::cast<MemRefType>(stickForGRUOp.getZGate().getType())
            .getElementType());

    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);

    // Get the dimensions of the original shape (the shape before stickifying)
    // used for creating a zTensor. Z, R, H gates have the same shape. Thus,
    // they share the dimensions.
    SmallVector<Value, 3> dims;
    getDimsFromMemRef(rewriter, loc, module, operandAdaptor.getZGate(), dims);

    // Get zDNN data type.
    zdnn_data_types zDNNDataType = llvmTypeToZDNNType(llvmElementTy);

    // Get zDNN data layout.
    zdnn_data_layouts zDNNDataLayout;
    zdnn_concat_info zDNNConcatInfo;
    StringRef prevLayerStr = stickForGRUOp.getPrevLayer();
    int64_t prevLayer = -1;
    if (prevLayerStr.equals_insensitive("none")) {
      prevLayer = PREV_LAYER_NONE;
    } else if (prevLayerStr.equals_insensitive("uni")) {
      prevLayer = PREV_LAYER_UNI;
    } else if (prevLayerStr.equals_insensitive("bidir")) {
      prevLayer = PREV_LAYER_BIDIR;
    }
    assert((prevLayer >= 0) &&
           "invalid prev_layer attribute in zlow.StickForLSTM");
    if (dims.size() == 2) {
      // for stickify input/hidden biases.
      zDNNDataLayout = ZDNN_2DS;
      zDNNConcatInfo = RNN_TYPE_GRU | USAGE_BIASES | prevLayer;
    } else if (dims.size() == 3) {
      // for stickify input/hidden weights.
      zDNNDataLayout = ZDNN_3DS;
      zDNNConcatInfo = RNN_TYPE_GRU | USAGE_WEIGHTS | prevLayer;
    } else {
      // Set invalid value to avoid uninitvar cppcheck warning.
      zDNNDataLayout = UNDEFINED_ZDNN_LAYOUT;
      llvm_unreachable("Unsupported layout");
    }

    // Create a zTensor.
    Value stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getOut());
    ZTensor zTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/zDNNDataLayout, /*originalDims=*/dims,
            /*isTransformed=*/false, /*isConcat=*/true,
            /*concatInfo=*/zDNNConcatInfo);

    // Ready to stickify.
    Value zGatePtr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getZGate());
    Value rGatePtr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getRGate());
    Value hGatePtr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getHGate());
    callApi(rewriter, loc, module, apiRegistry, API::ZDNN_TRANSFORM_ZTENSOR,
        {toOpaquePtr(rewriter, loc, module, zTensor.val), zGatePtr, rGatePtr,
            hGatePtr});

    rewriter.eraseOp(op);
    return success();
  }

private:
  ApiRegistry apiRegistry;
};

class ZLowLSTMLowering : public ConvertToLLVMPattern {
public:
  explicit ZLowLSTMLowering(MLIRContext *context, LLVMTypeConverter &lowering_,
      ApiRegistry apiRegistry)
      : ConvertToLLVMPattern(
            ZLowLSTMOp::getOperationName(), context, lowering_) {
    this->apiRegistry = apiRegistry;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();
    ZLowLSTMOp lstmOp = mlir::cast<ZLowLSTMOp>(op);
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

    ZLowLSTMOpAdaptor operandAdaptor(operands);
    Type llvmElementTy = typeConverter->convertType(
        mlir::cast<MemRefType>(lstmOp.getInput().getType()).getElementType());

    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);

    // Some frequently used types and constants.
    Type llvmI64Ty = rewriter.getI64Type();
    Value oneI64 = create.llvm.constant(llvmI64Ty, static_cast<int64_t>(1));

    // Get the dimensions of the original shape (the shape before stickifying)
    // used for creating zTensors.
    std::vector<Value> dims = getDimsFromShapeMemRefBySize(
        rewriter, loc, module, operandAdaptor.getShape(), /*size=*/5);
    // direction
    Value D = dims[0];
    // timestep
    Value T = dims[1];
    // batch size
    Value B = dims[2];
    // feature size
    Value F = dims[3];
    // hidden size
    Value H = dims[4];

    StringRef prevLayerStr = lstmOp.getPrevLayer();
    int64_t prevLayer = -1;
    if (prevLayerStr.equals_insensitive("none")) {
      prevLayer = PREV_LAYER_NONE;
    } else if (prevLayerStr.equals_insensitive("uni")) {
      prevLayer = PREV_LAYER_UNI;
    } else if (prevLayerStr.equals_insensitive("bidir")) {
      prevLayer = PREV_LAYER_BIDIR;
    }
    assert((prevLayer >= 0) && "invalid prev_layer attribute in zlow.LSTM");

    // Get zDNN data type.
    zdnn_data_types zDNNDataType = llvmTypeToZDNNType(llvmElementTy);

    // Create a zTensor for input.
    Value stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getInput());
    ZTensor inputZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_3DS, /*originalDims=*/{T, B, F},
            /*isTransformed=*/true);

    // Create zTensor for h0.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getH0());
    ZTensor h0ZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_3DS, /*originalDims=*/{D, B, H},
            /*isTransformed=*/true);

    // Create zTensor for c0. Reuse descriptors from h0 because h0 and c0 have
    // the same shape.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getC0());
    ZTensor c0ZTensor = zTensorHelper.getZTensor(
        /*preTransformedDescPtr=*/h0ZTensor.preTransformedDescPtr,
        /*transformedDescPtr=*/h0ZTensor.transformedDescPtr,
        /*bufferSize=*/h0ZTensor.bufferSize,
        /*alignedBuffer=*/stickI8Ptr,
        /*isTransformed=*/true);

    // Create zTensor for input_weights.
    stickI8Ptr =
        zTensorHelper.getAlignedI8Ptr(operandAdaptor.getInputWeights());
    ZTensor inputWeightsZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_3DS, /*originalDims=*/{D, F, H},
            /*isTransformed=*/true, /*'isConcat=*/true,
            /*concatInfo=*/RNN_TYPE_LSTM | USAGE_WEIGHTS | prevLayer);

    // Create zTensor for input_bias.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getInputBias());
    ZTensor inputBiasZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_2DS, /*originalDims=*/{D, H},
            /*isTransformed=*/true, /*'isConcat=*/true,
            /*concatInfo=*/RNN_TYPE_LSTM | USAGE_BIASES | prevLayer);

    // Create zTensor for hidden_weights.
    stickI8Ptr =
        zTensorHelper.getAlignedI8Ptr(operandAdaptor.getHiddenWeights());
    ZTensor hiddenWeightsZTensor = zTensorHelper.getZTensor(stickI8Ptr,
        /*dataType=*/zDNNDataType,
        /*layout=*/ZDNN_3DS, /*originalDims=*/{D, H, H},
        /*isTransformed=*/true, /*'isConcat=*/true,
        /*concatInfo=*/RNN_TYPE_LSTM | USAGE_HIDDEN_WEIGHTS | prevLayer);

    // Create zTensor for hidden_bias.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getHiddenBias());
    ZTensor hiddenBiasZTensor = zTensorHelper.getZTensor(stickI8Ptr,
        /*dataType=*/zDNNDataType,
        /*layout=*/ZDNN_2DS, /*originalDims=*/{D, H},
        /*isTransformed=*/true, /*'isConcat=*/true,
        /*concatInfo=*/RNN_TYPE_LSTM | USAGE_HIDDEN_BIASES | prevLayer);

    // Direction input.
    Value direction;
    StringRef directionStr = lstmOp.getDirection();
    if (directionStr.equals_insensitive("forward")) {
      direction = create.llvm.constant(llvmI64Ty, static_cast<int64_t>(FWD));
    } else if (directionStr.equals_insensitive("reverse")) {
      direction = create.llvm.constant(llvmI64Ty, static_cast<int64_t>(BWD));
    } else if (directionStr.equals_insensitive("bidirectional")) {
      direction = create.llvm.constant(llvmI64Ty, static_cast<int64_t>(BIDIR));
    } else
      llvm_unreachable("Unsupported direction");

    // work_area.
    Value workArea =
        zTensorHelper.getAlignedI8Ptr(operandAdaptor.getWorkArea());

    // Create zTensor for hn_output.
    Value preTransformedDescPtr;

    if (lstmOp.getReturnAllSteps() == -1)
      // all steps.
      preTransformedDescPtr = zTensorHelper.getPreTransformedDescPtr(
          zDNNDataType, ZDNN_4DS, {T, D, B, H});
    else
      // the last step.
      preTransformedDescPtr = zTensorHelper.getPreTransformedDescPtr(
          zDNNDataType, ZDNN_4DS, {oneI64, D, B, H});
    zdnn_concat_info concatInfo =
        RNN_TYPE_LSTM | USAGE_WEIGHTS | PREV_LAYER_NONE;
    // Transformed descriptor.
    Value transformedDescPtr = zTensorHelper.getTransformedDescPtr(
        preTransformedDescPtr, /*isConcat=*/false,
        /*concatInfo=*/concatInfo);
    // Buffer size.
    Value bufferSize = zTensorHelper.getBufferSize(transformedDescPtr);
    // Buffer pointer.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getHnOutput());
    ZTensor hnOutputZTensor = zTensorHelper.getZTensor(
        /*preTransformedDescPtr=*/preTransformedDescPtr,
        /*transformedDescPtr=*/transformedDescPtr,
        /*bufferSize=*/bufferSize,
        /*alignedBuffer=*/stickI8Ptr,
        /*isTransformed=*/true);

    // Create zTensor for cf_output. Reuse descriptors from hn_output if
    // hn_output is the last step output.
    ZTensor cfOutputZTensor;
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getCfOutput());
    if (lstmOp.getReturnAllSteps() != -1)
      cfOutputZTensor = zTensorHelper.getZTensor(
          /*preTransformedDescPtr=*/hnOutputZTensor.preTransformedDescPtr,
          /*transformedDescPtr=*/hnOutputZTensor.transformedDescPtr,
          /*bufferSize=*/hnOutputZTensor.bufferSize,
          /*alignedBuffer=*/stickI8Ptr,
          /*isTransformed=*/true);
    else
      cfOutputZTensor =
          zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
              /*layout=*/ZDNN_4DS, /*originalDims=*/{oneI64, D, B, H},
              /*isTransformed=*/true);

    // Ready to call zDNN LSTM.
    callApi(rewriter, loc, module, apiRegistry, API::ZDNN_LSTM,
        {toOpaquePtr(rewriter, loc, module, inputZTensor.val),
            toOpaquePtr(rewriter, loc, module, h0ZTensor.val),
            toOpaquePtr(rewriter, loc, module, c0ZTensor.val),
            toOpaquePtr(rewriter, loc, module, inputWeightsZTensor.val),
            toOpaquePtr(rewriter, loc, module, inputBiasZTensor.val),
            toOpaquePtr(rewriter, loc, module, hiddenWeightsZTensor.val),
            toOpaquePtr(rewriter, loc, module, hiddenBiasZTensor.val),
            direction, workArea,
            toOpaquePtr(rewriter, loc, module, hnOutputZTensor.val),
            toOpaquePtr(rewriter, loc, module, cfOutputZTensor.val)});

    rewriter.eraseOp(op);
    return success();
  }

private:
  ApiRegistry apiRegistry;
};

class ZLowGRULowering : public ConvertToLLVMPattern {
public:
  explicit ZLowGRULowering(MLIRContext *context, LLVMTypeConverter &lowering_,
      ApiRegistry apiRegistry)
      : ConvertToLLVMPattern(
            ZLowGRUOp::getOperationName(), context, lowering_) {
    this->apiRegistry = apiRegistry;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();
    ZLowGRUOp gruOp = mlir::cast<ZLowGRUOp>(op);
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

    ZLowGRUOpAdaptor operandAdaptor(operands);
    Type llvmElementTy = typeConverter->convertType(
        mlir::cast<MemRefType>(gruOp.getInput().getType()).getElementType());

    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);

    // Some frequently used types and constants.
    Type llvmI64Ty = rewriter.getI64Type();
    Value oneI64 = create.llvm.constant(llvmI64Ty, static_cast<int64_t>(1));

    // Get the dimensions of the original shape (the shape before stickifying)
    // used for creating zTensors.
    std::vector<Value> dims = getDimsFromShapeMemRefBySize(
        rewriter, loc, module, operandAdaptor.getShape(), /*size=*/5);
    // direction
    Value D = dims[0];
    // timestep
    Value T = dims[1];
    // batch size
    Value B = dims[2];
    // feature size
    Value F = dims[3];
    // hidden size
    Value H = dims[4];

    // Get zDNN data type.
    zdnn_data_types zDNNDataType = llvmTypeToZDNNType(llvmElementTy);

    // Create a zTensor for input.
    Value stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getInput());
    ZTensor inputZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_3DS, /*originalDims=*/{T, B, F},
            /*isTransformed=*/true);

    // Create zTensor for h0.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getH0());
    ZTensor h0ZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_3DS, /*originalDims=*/{D, B, H},
            /*isTransformed=*/true);

    // Create zTensor for input_weights.
    stickI8Ptr =
        zTensorHelper.getAlignedI8Ptr(operandAdaptor.getInputWeights());
    ZTensor inputWeightsZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_3DS, /*originalDims=*/{D, F, H},
            /*isTransformed=*/true, /*'isConcat=*/true,
            /*concatInfo=*/RNN_TYPE_GRU | USAGE_WEIGHTS | PREV_LAYER_NONE);

    // Create zTensor for input_bias.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getInputBias());
    ZTensor inputBiasZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_2DS, /*originalDims=*/{D, H},
            /*isTransformed=*/true, /*'isConcat=*/true,
            /*concatInfo=*/RNN_TYPE_GRU | USAGE_BIASES | PREV_LAYER_NONE);

    // Create zTensor for hidden_weights.
    stickI8Ptr =
        zTensorHelper.getAlignedI8Ptr(operandAdaptor.getHiddenWeights());
    ZTensor hiddenWeightsZTensor = zTensorHelper.getZTensor(stickI8Ptr,
        /*dataType=*/zDNNDataType,
        /*layout=*/ZDNN_3DS, /*originalDims=*/{D, H, H},
        /*isTransformed=*/true, /*'isConcat=*/true,
        /*concatInfo=*/RNN_TYPE_GRU | USAGE_HIDDEN_WEIGHTS | PREV_LAYER_NONE);

    // Create zTensor for hidden_bias.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getHiddenBias());
    ZTensor hiddenBiasZTensor = zTensorHelper.getZTensor(stickI8Ptr,
        /*dataType=*/zDNNDataType,
        /*layout=*/ZDNN_2DS, /*originalDims=*/{D, H},
        /*isTransformed=*/true, /*'isConcat=*/true,
        /*concatInfo=*/RNN_TYPE_GRU | USAGE_HIDDEN_BIASES | PREV_LAYER_NONE);

    // Direction input.
    Value direction;
    StringRef directionStr = gruOp.getDirection();
    if (directionStr.equals_insensitive("forward")) {
      direction = create.llvm.constant(llvmI64Ty, static_cast<int64_t>(FWD));
    } else if (directionStr.equals_insensitive("reverse")) {
      direction = create.llvm.constant(llvmI64Ty, static_cast<int64_t>(BWD));
    } else if (directionStr.equals_insensitive("bidirectional")) {
      direction = create.llvm.constant(llvmI64Ty, static_cast<int64_t>(BIDIR));
    } else
      llvm_unreachable("Unsupported direction");

    // work_area.
    Value workArea =
        zTensorHelper.getAlignedI8Ptr(operandAdaptor.getWorkArea());

    // Create zTensor for hn_output.
    Value preTransformedDescPtr;
    if (gruOp.getReturnAllSteps() == -1)
      // all steps.
      preTransformedDescPtr = zTensorHelper.getPreTransformedDescPtr(
          zDNNDataType, ZDNN_4DS, {T, D, B, H});
    else
      // the last step.
      preTransformedDescPtr = zTensorHelper.getPreTransformedDescPtr(
          zDNNDataType, ZDNN_4DS, {oneI64, D, B, H});
    zdnn_concat_info concatInfo =
        RNN_TYPE_GRU | USAGE_WEIGHTS | PREV_LAYER_NONE;
    // Transformed descriptor.
    Value transformedDescPtr = zTensorHelper.getTransformedDescPtr(
        preTransformedDescPtr, /*isConcat=*/false,
        /*concatInfo=*/concatInfo);
    // Buffer size.
    Value bufferSize = zTensorHelper.getBufferSize(transformedDescPtr);
    // Buffer pointer.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getHnOutput());
    ZTensor hnOutputZTensor = zTensorHelper.getZTensor(
        /*preTransformedDescPtr=*/preTransformedDescPtr,
        /*transformedDescPtr=*/transformedDescPtr,
        /*bufferSize=*/bufferSize,
        /*alignedBuffer=*/stickI8Ptr,
        /*isTransformed=*/true);

    // Ready to call zDNN GRU.
    callApi(rewriter, loc, module, apiRegistry, API::ZDNN_GRU,
        {toOpaquePtr(rewriter, loc, module, inputZTensor.val),
            toOpaquePtr(rewriter, loc, module, h0ZTensor.val),
            toOpaquePtr(rewriter, loc, module, inputWeightsZTensor.val),
            toOpaquePtr(rewriter, loc, module, inputBiasZTensor.val),
            toOpaquePtr(rewriter, loc, module, hiddenWeightsZTensor.val),
            toOpaquePtr(rewriter, loc, module, hiddenBiasZTensor.val),
            direction, workArea,
            toOpaquePtr(rewriter, loc, module, hnOutputZTensor.val)});

    rewriter.eraseOp(op);
    return success();
  }

private:
  ApiRegistry apiRegistry;
};

class ZLowUnstickLowering : public ConvertToLLVMPattern {
public:
  explicit ZLowUnstickLowering(MLIRContext *context,
      LLVMTypeConverter &lowering_, ApiRegistry apiRegistry)
      : ConvertToLLVMPattern(
            ZLowUnstickOp::getOperationName(), context, lowering_) {
    this->apiRegistry = apiRegistry;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();
    ZLowUnstickOp unstickOp = mlir::cast<ZLowUnstickOp>(op);

    ZLowUnstickOpAdaptor operandAdaptor(operands);
    Type llvmElementTy = typeConverter->convertType(
        mlir::cast<MemRefType>(unstickOp.getOut().getType()).getElementType());

    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);

    // Get the dimensions of the original shape (the shape before stickifying)
    // used for creating a zTensor. For 'zLow.unstick', the original shape is
    // obtained from the second argument.
    SmallVector<Value, 3> dims;
    getDimsFromMemRef(rewriter, loc, module, operandAdaptor.getOut(), dims);

    // Get zDNN data type.
    zdnn_data_types zDNNDataType = llvmTypeToZDNNType(llvmElementTy);

    // Get zDNN data layout.
    zdnn_data_layouts zDNNDataLayout = convertLayoutAttrToZDNNDataLayout(
        dims.size(), unstickOp.getLayoutAttr());

    // Create a zTensor.
    Value stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getX());
    ZTensor zTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/zDNNDataLayout, /*originalDims=*/dims,
            /*isTransformed=*/true);

    // Ready to unstickify.
    Value unstickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getOut());
    callApi(rewriter, loc, module, apiRegistry, API::ZDNN_TRANSFORM_ORIGTENSOR,
        {toOpaquePtr(rewriter, loc, module, zTensor.val), unstickI8Ptr});

    rewriter.eraseOp(op);
    return success();
  }

private:
  ApiRegistry apiRegistry;
};

template <typename UnaryElementwiseOp>
class ZLowUnaryElementwiseOpLowering : public ConvertToLLVMPattern {
public:
  explicit ZLowUnaryElementwiseOpLowering(MLIRContext *context,
      LLVMTypeConverter &lowering_, ApiRegistry apiRegistry)
      : ConvertToLLVMPattern(
            UnaryElementwiseOp::getOperationName(), context, lowering_) {
    this->apiRegistry = apiRegistry;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();
    MLIRContext *context = rewriter.getContext();
    UnaryElementwiseOp unaryOp = mlir::cast<UnaryElementwiseOp>(op);
    typename UnaryElementwiseOp::Adaptor operandAdaptor(operands);
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

    Value input = operandAdaptor.getX();
    Value shape = operandAdaptor.getShape();
    Value output = operandAdaptor.getOut();
    Type llvmElementTy = typeConverter->convertType(
        mlir::cast<MemRefType>(op->getOperand(0).getType()).getElementType());

    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);

    // Get zDNN data type.
    zdnn_data_types zDNNDataType = llvmTypeToZDNNType(llvmElementTy);

    // Get zDNN data layout.
    zdnn_data_layouts zDNNDataLayout =
        convertLayoutAttrToZDNNDataLayout(0, unaryOp.getLayoutAttr());

    // Get the dimensions of the original shape (the shape before stickifying)
    // used for creating a zTensor.
    std::vector<Value> dims =
        getDimsFromShapeMemRef(rewriter, loc, module, shape,
            /*layout=*/zDNNDataLayout);

    // Create an input zTensor.
    Value stickI8Ptr = zTensorHelper.getAlignedI8Ptr(input);
    ZTensor inputZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/zDNNDataLayout, /*originalDims=*/dims,
            /*isTransformed=*/true);

    // Create an output zTensor.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(output);
    ZTensor outputZTensor = zTensorHelper.getZTensor(
        /*preTransformedDescPtr=*/inputZTensor.preTransformedDescPtr,
        /*transformedDescPtr=*/inputZTensor.transformedDescPtr,
        /*bufferSize=*/inputZTensor.bufferSize,
        /*alignedBuffer=*/stickI8Ptr,
        /*isTransformed=*/true);

    // Ready to call a zDNN elementwise API.
    if (APIFor<UnaryElementwiseOp>() == API::ZDNN_RELU) {
      // Insert "nullptr" as the third argument for the "clipping_value",
      // because onnx.Relu does not use the clipping value.
      Value nullpointer = create.llvm.null(krnl::getI8PointerType(context));
      callApi(rewriter, loc, module, apiRegistry, APIFor<UnaryElementwiseOp>(),
          {toOpaquePtr(rewriter, loc, module, inputZTensor.val), nullpointer,
              toOpaquePtr(rewriter, loc, module, outputZTensor.val)});
    } else if (APIFor<UnaryElementwiseOp>() == API::ZDNN_INVSQRT) {
      MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
      // Create a float for the epsilon value.
      Value epsilon = create.llvm.constant(rewriter.getF32Type(), nnpaEpsilon);
      // Pass to ZDNN.
      callApi(rewriter, loc, module, apiRegistry, APIFor<UnaryElementwiseOp>(),
          {toOpaquePtr(rewriter, loc, module, inputZTensor.val), epsilon,
              toOpaquePtr(rewriter, loc, module, outputZTensor.val)});
    } else {
      callApi(rewriter, loc, module, apiRegistry, APIFor<UnaryElementwiseOp>(),
          {toOpaquePtr(rewriter, loc, module, inputZTensor.val),
              toOpaquePtr(rewriter, loc, module, outputZTensor.val)});
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  ApiRegistry apiRegistry;
};

template <typename BinaryElementwiseOp>
class ZLowBinaryElementwiseOpLowering : public ConvertToLLVMPattern {
public:
  explicit ZLowBinaryElementwiseOpLowering(MLIRContext *context,
      LLVMTypeConverter &lowering_, ApiRegistry apiRegistry)
      : ConvertToLLVMPattern(
            BinaryElementwiseOp::getOperationName(), context, lowering_) {
    this->apiRegistry = apiRegistry;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();
    BinaryElementwiseOp binaryOp = mlir::cast<BinaryElementwiseOp>(op);
    typename BinaryElementwiseOp::Adaptor operandAdaptor(operands);

    Value input1 = operandAdaptor.getX();
    Value input2 = operandAdaptor.getY();
    Value shape = operandAdaptor.getShape();
    Value output = operandAdaptor.getOut();
    Type llvmElementTy = typeConverter->convertType(
        mlir::cast<MemRefType>(op->getOperand(0).getType()).getElementType());

    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);

    // Get zDNN data type.
    zdnn_data_types zDNNDataType = llvmTypeToZDNNType(llvmElementTy);

    // Get zDNN data layout.
    zdnn_data_layouts zDNNDataLayout =
        convertLayoutAttrToZDNNDataLayout(0, binaryOp.getLayoutAttr());

    // Get the dimensions of the original shape (the shape before stickifying)
    // used for creating a zTensor.
    std::vector<Value> dims =
        getDimsFromShapeMemRef(rewriter, loc, module, shape,
            /*layout=*/zDNNDataLayout);

    // Create the first zTensor input.
    Value stickI8Ptr = zTensorHelper.getAlignedI8Ptr(input1);
    ZTensor inputZTensor1 =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/zDNNDataLayout, /*originalDims=*/dims,
            /*isTransformed=*/true);

    // Create the second zTensor input.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(input2);
    ZTensor inputZTensor2 = zTensorHelper.getZTensor(
        /*preTransformedDescPtr=*/inputZTensor1.preTransformedDescPtr,
        /*transformedDescPtr=*/inputZTensor1.transformedDescPtr,
        /*bufferSize=*/inputZTensor1.bufferSize,
        /*alignedBuffer=*/stickI8Ptr,
        /*isTransformed=*/true);

    // Create an output zTensor.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(output);
    ZTensor outputZTensor = zTensorHelper.getZTensor(
        /*preTransformedDescPtr=*/inputZTensor1.preTransformedDescPtr,
        /*transformedDescPtr=*/inputZTensor1.transformedDescPtr,
        /*bufferSize=*/inputZTensor1.bufferSize,
        /*alignedBuffer=*/stickI8Ptr,
        /*isTransformed=*/true);

    // Ready to call a zDNN elementwise API.
    callApi(rewriter, loc, module, apiRegistry, APIFor<BinaryElementwiseOp>(),
        {toOpaquePtr(rewriter, loc, module, inputZTensor1.val),
            toOpaquePtr(rewriter, loc, module, inputZTensor2.val),
            toOpaquePtr(rewriter, loc, module, outputZTensor.val)});

    rewriter.eraseOp(op);
    return success();
  }

private:
  ApiRegistry apiRegistry;
};

class ZLowSoftmaxOpLowering : public ConvertToLLVMPattern {
public:
  explicit ZLowSoftmaxOpLowering(MLIRContext *context,
      LLVMTypeConverter &lowering_, ApiRegistry apiRegistry)
      : ConvertToLLVMPattern(
            ZLowSoftmaxOp::getOperationName(), context, lowering_) {
    this->apiRegistry = apiRegistry;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();
    ZLowSoftmaxOp softmaxOp = mlir::cast<ZLowSoftmaxOp>(op);
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

    ZLowSoftmaxOpAdaptor operandAdaptor(operands);
    Type llvmElementTy = typeConverter->convertType(
        mlir::cast<MemRefType>(softmaxOp.getX().getType()).getElementType());

    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);

    // Get zDNN data type.
    zdnn_data_types zDNNDataType = llvmTypeToZDNNType(llvmElementTy);

    // zDNN softmax uses 3DS layout.
    zdnn_data_layouts zDNNDataLayout = ZDNN_3DS;
    // Get the dimensions of the original shape (the shape before stickifying)
    // used for creating a zTensor.
    std::vector<Value> dims =
        getDimsFromShapeMemRef(rewriter, loc, module, operandAdaptor.getShape(),
            /*layout=*/zDNNDataLayout);

    // Create the input zTensor.
    Value stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getX());
    ZTensor inputZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/zDNNDataLayout, /*originalDims=*/dims,
            /*isTransformed=*/true);

    // Create activation function type.
    nnpa_softmax_act actType;
    StringRef actFuncStr = softmaxOp.getActFunc();
    if (actFuncStr.equals_insensitive("act_none"))
      actType = NNPA_SOFTMAX_NONE;
    else if (actFuncStr.equals_insensitive("act_log"))
      actType = NNPA_SOFTMAX_LOG;
    else
      llvm_unreachable("Unsupported activation function");
    Value actFunc = create.llvm.constant(
        rewriter.getI64Type(), static_cast<int64_t>(actType));

    // Create the output zTensor.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getOut());
    ZTensor outputZTensor = zTensorHelper.getZTensor(
        /*preTransformedDescPtr=*/inputZTensor.preTransformedDescPtr,
        /*transformedDescPtr=*/inputZTensor.transformedDescPtr,
        /*bufferSize=*/inputZTensor.bufferSize,
        /*alignedBuffer=*/stickI8Ptr,
        /*isTransformed=*/true);

    // work_area.
    Value workArea =
        zTensorHelper.getAlignedI8Ptr(operandAdaptor.getWorkArea());

    // Call zDNN softmax.
    callApi(rewriter, loc, module, apiRegistry, API::ZDNN_SOFTMAX,
        {
            toOpaquePtr(rewriter, loc, module, inputZTensor.val),
            workArea,
            actFunc,
            toOpaquePtr(rewriter, loc, module, outputZTensor.val),
        });

    rewriter.eraseOp(op);
    return success();
  }

private:
  ApiRegistry apiRegistry;
};

class ZLowLeakyReluLowering : public ConvertToLLVMPattern {
public:
  explicit ZLowLeakyReluLowering(MLIRContext *context,
      LLVMTypeConverter &lowering_, ApiRegistry apiRegistry)
      : ConvertToLLVMPattern(
            ZLowLeakyReluOp::getOperationName(), context, lowering_) {
    this->apiRegistry = apiRegistry;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();
    ZLowLeakyReluOp leakyreluOp = cast<ZLowLeakyReluOp>(op);
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
    MLIRContext *context = rewriter.getContext();
    typename ZLowLeakyReluOp::Adaptor operandAdaptor(operands);

    Value input = operandAdaptor.getX();
    Value shape = operandAdaptor.getShape();
    Value output = operandAdaptor.getOut();
    Type llvmElementTy = typeConverter->convertType(
        mlir::cast<MemRefType>(op->getOperand(0).getType()).getElementType());

    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);

    // Get zDNN data type.
    zdnn_data_types zDNNDataType = llvmTypeToZDNNType(llvmElementTy);

    // Get zDNN data layout.
    zdnn_data_layouts zDNNDataLayout =
        convertLayoutAttrToZDNNDataLayout(0, leakyreluOp.getLayoutAttr());

    // Get the dimensions of the original shape (the shape before stickifying)
    // used for creating a zTensor.
    std::vector<Value> dims =
        getDimsFromShapeMemRef(rewriter, loc, module, shape,
            /*layout=*/zDNNDataLayout);

    // Create an input zTensor.
    Value stickI8Ptr = zTensorHelper.getAlignedI8Ptr(input);
    ZTensor inputZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/zDNNDataLayout, /*originalDims=*/dims,
            /*isTransformed=*/true);

    // Create an output zTensor.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(output);
    ZTensor outputZTensor = zTensorHelper.getZTensor(
        /*preTransformedDescPtr=*/inputZTensor.preTransformedDescPtr,
        /*transformedDescPtr=*/inputZTensor.transformedDescPtr,
        /*bufferSize=*/inputZTensor.bufferSize,
        /*alignedBuffer=*/stickI8Ptr,
        /*isTransformed=*/true);

    // Create the clipping value as null because the zDNN LeakyRelu API does not
    // use it.
    Value clippingVal = create.llvm.null(krnl::getI8PointerType(context));

    // Create the adjustment factor value from the input alpha attribute.
    FloatAttr alphaAttr = leakyreluOp.getAlphaAttr();
    float alphaFloat = (float)alphaAttr.getValueAsDouble();
    Value adjustmentFactorVal =
        create.llvm.constant(rewriter.getF32Type(), alphaFloat);

    // Call the zDNN LeakyRelu API.
    callApi(rewriter, loc, module, apiRegistry, API::ZDNN_LEAKY_RELU,
        {toOpaquePtr(rewriter, loc, module, inputZTensor.val), clippingVal,
            adjustmentFactorVal,
            toOpaquePtr(rewriter, loc, module, outputZTensor.val)});

    rewriter.eraseOp(op);
    return success();
  }

private:
  ApiRegistry apiRegistry;
};

template <typename REDUCE_OP>
zdnn_reduce_ops getZDNNReduceOpType() {
  return REDUCE_OP_MAXIMUM;
}

template <>
zdnn_reduce_ops getZDNNReduceOpType<ZLowReduceMaxOp>() {
  return REDUCE_OP_MAXIMUM;
}

template <>
zdnn_reduce_ops getZDNNReduceOpType<ZLowReduceMinOp>() {
  return REDUCE_OP_MINIMUM;
}

template <typename REDUCE_OP>
class ZLowReduceLowering : public ConvertToLLVMPattern {
public:
  explicit ZLowReduceLowering(MLIRContext *context,
      LLVMTypeConverter &lowering_, ApiRegistry apiRegistry)
      : ConvertToLLVMPattern(
            REDUCE_OP::getOperationName(), context, lowering_) {
    this->apiRegistry = apiRegistry;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();
    REDUCE_OP reduceOp = mlir::cast<REDUCE_OP>(op);
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
    typename REDUCE_OP::Adaptor operandAdaptor(operands);

    Value data = operandAdaptor.getX();
    Value shape = operandAdaptor.getShape();
    Value output = operandAdaptor.getOut();
    Type llvmElementTy = typeConverter->convertType(
        mlir::cast<MemRefType>(op->getOperand(0).getType()).getElementType());

    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);

    // Get zDNN data type.
    zdnn_data_types zDNNDataType = llvmTypeToZDNNType(llvmElementTy);

    // Get zDNN data layout.
    zdnn_data_layouts zDNNDataLayout =
        convertLayoutAttrToZDNNDataLayout(0, reduceOp.getLayoutAttr());

    // Get the dimensions of the original shape (the shape before stickifying)
    // used for creating a zTensor.
    std::vector<Value> dims =
        getDimsFromShapeMemRef(rewriter, loc, module, shape,
            /*layout=*/zDNNDataLayout);

    Type llvmI64Ty = rewriter.getI64Type();
    Value one = create.llvm.constant(llvmI64Ty, static_cast<int64_t>(1));

    // Calculation for the output dimension
    int64_t axis = dims.size() - 1;
    SmallVector<Value, 4> outputDims;
    for (int64_t i = 0; i < axis; ++i) {
      outputDims.emplace_back(dims[i]);
    }
    outputDims.emplace_back(one);

    // Create an input zTensor.
    Value stickI8Ptr = zTensorHelper.getAlignedI8Ptr(data);
    ZTensor inputZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/zDNNDataLayout, /*originalDims=*/dims,
            /*isTransformed=*/true);

    // Create an output zTensor.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(output);
    ZTensor outputZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/zDNNDataLayout, /*originalDims=*/outputDims,
            /*isTransformed=*/true);

    // work_area.
    Value workArea =
        zTensorHelper.getAlignedI8Ptr(operandAdaptor.getWorkArea());

    // op_type
    zdnn_reduce_ops zdnnOpType = getZDNNReduceOpType<REDUCE_OP>();
    Value opType = create.llvm.constant(
        rewriter.getI64Type(), static_cast<int64_t>(zdnnOpType));

    // Call the zDNN ReduceMax/ReduceMin API.
    callApi(rewriter, loc, module, apiRegistry, API::ZDNN_REDUCE,
        {toOpaquePtr(rewriter, loc, module, inputZTensor.val), workArea, opType,
            toOpaquePtr(rewriter, loc, module, outputZTensor.val)});

    rewriter.eraseOp(op);
    return success();
  }

private:
  ApiRegistry apiRegistry;
};

class ZLowMatMulLowering : public ConvertToLLVMPattern {
public:
  explicit ZLowMatMulLowering(MLIRContext *context,
      LLVMTypeConverter &lowering_, ApiRegistry apiRegistry)
      : ConvertToLLVMPattern(
            ZLowMatMulOp::getOperationName(), context, lowering_) {
    this->apiRegistry = apiRegistry;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();
    ZLowMatMulOp matmulOp = mlir::cast<ZLowMatMulOp>(op);
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

    ZLowMatMulOpAdaptor operandAdaptor(operands);
    Type llvmElementTy = typeConverter->convertType(
        mlir::cast<MemRefType>(matmulOp.getX().getType()).getElementType());

    bool stacked = false, broadcasting1 = false, broadcasting23 = false,
         transposeA = false, transposeB = false;
    if (matmulOp.getIsStacked() == -1)
      stacked = true;
    if (matmulOp.getIsBcast1() == -1)
      broadcasting1 = true;
    else if (matmulOp.getIsBcast23() == -1)
      broadcasting23 = true;
    if (matmulOp.getTransposeA() != 0)
      transposeA = true;
    if (matmulOp.getTransposeB() != 0)
      transposeB = true;

    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);

    // Some frequently used types and constants.
    Type llvmI64Ty = rewriter.getI64Type();

    // Get the dimensions of the original shape (the shape before stickifying)
    // used for creating zTensors.
    int dimCount = 3;
    if (stacked || broadcasting1 || broadcasting23)
      dimCount = 4;
    std::vector<Value> dims = getDimsFromShapeMemRefBySize(
        rewriter, loc, module, operandAdaptor.getShape(), /*size=*/dimCount);
    // Dimensions: s, m, n, p;
    Value S, M, N, P;
    if (stacked || broadcasting23) {
      S = dims[0];
      M = dims[1];
      N = dims[2];
      P = dims[3];
    } else if (broadcasting1) {
      M = dims[0];
      N = dims[1];
      S = dims[2];
      P = dims[3];
    } else {
      M = dims[0];
      N = dims[1];
      P = dims[2];
    }

    // Get zDNN data type.
    zdnn_data_types zDNNDataType = llvmTypeToZDNNType(llvmElementTy);

    // Create zTensors.
    ZTensor xZTensor, yZTensor, biasZTensor, outputZTensor;

    // clang-format off
    // Requirements
    // Type        X                    Y                   Bias             Output
    // ----------------------------------------------------------------------------------------
    // unstacked   ZDNN_2D (m, n)       ZDNN_2D (n, p)      ZDNN_1D (p)      ZDNN_2D (m, p)
    // stacked     ZDNN_3DS (s, m, n)   ZDNN_3DS (s, n, p)  ZDNN_2DS (s, p)  ZDNN_3DS (s, m, p)
    // bcast1      ZDNN_2D (m, n)       ZDNN_3DS (s, n, p)  ZDNN_2DS (s, p)  ZDNN_3DS (s, m, p)
    // bcast23     ZDNN_3DS (s, m, n)   ZDNN_2D (n, p)      ZDNN_1D (p)      ZDNN_3DS (s, m, p)
    // clang-format on

    // X
    Value stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getX());
    if (stacked || broadcasting23) {
      if (transposeA)
        // ZDNN_3DS (s, n, m)
        xZTensor =
            zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
                /*layout=*/ZDNN_3DS, /*originalDims=*/{S, N, M},
                /*isTransformed=*/true);
      else
        // ZDNN_3DS (s, m, n)
        xZTensor =
            zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
                /*layout=*/ZDNN_3DS, /*originalDims=*/{S, M, N},
                /*isTransformed=*/true);
    } else { /* unstacked || broadcasting1 */
      if (transposeA)
        // ZDNN_2D (n, m)
        xZTensor =
            zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
                /*layout=*/ZDNN_2D, /*originalDims=*/{N, M},
                /*isTransformed=*/true);
      else
        // ZDNN_2D (m, n)
        xZTensor =
            zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
                /*layout=*/ZDNN_2D, /*originalDims=*/{M, N},
                /*isTransformed=*/true);
    }
    // Y
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getY());
    if (stacked || broadcasting1) {
      if (transposeB)
        // ZDNN_3DS (s, p, n)
        yZTensor =
            zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
                /*layout=*/ZDNN_3DS, /*originalDims=*/{S, P, N},
                /*isTransformed=*/true);
      else
        // ZDNN_3DS (s, n, p)
        yZTensor =
            zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
                /*layout=*/ZDNN_3DS, /*originalDims=*/{S, N, P},
                /*isTransformed=*/true);
    } else { /* unstacked || broadcasting23 */
      if (transposeB)
        // ZDNN_2D (p, n)
        yZTensor =
            zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
                /*layout=*/ZDNN_2D, /*originalDims=*/{P, N},
                /*isTransformed=*/true);
      else
        // ZDNN_2D (n, p)
        yZTensor =
            zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
                /*layout=*/ZDNN_2D, /*originalDims=*/{N, P},
                /*isTransformed=*/true);
    }
    // Bias
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getBias());
    if (stacked || broadcasting1)
      // ZDNN_2D (s, p)
      biasZTensor =
          zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
              /*layout=*/ZDNN_2DS, /*originalDims=*/{S, P},
              /*isTransformed=*/true);
    else
      // ZDNN_1D (p)
      biasZTensor =
          zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
              /*layout=*/ZDNN_1D, /*originalDims=*/{P},
              /*isTransformed=*/true);
    // Op_type
    Value opType;
    if (broadcasting23 || broadcasting1)
      opType = create.llvm.constant(
          llvmI64Ty, static_cast<int64_t>(NNPA_MATMUL_BCAST_OP_ADDITION));
    else
      opType = create.llvm.constant(
          llvmI64Ty, static_cast<int64_t>(NNPA_MATMUL_OP_ADDITION));
    // Transposing
    Value transposeAVal;
    if (transposeA)
      transposeAVal = create.llvm.constant(llvmI64Ty, static_cast<int64_t>(1));
    else
      transposeAVal = create.llvm.constant(llvmI64Ty, static_cast<int64_t>(0));
    Value transposeBVal;
    if (transposeB)
      transposeBVal = create.llvm.constant(llvmI64Ty, static_cast<int64_t>(1));
    else
      transposeBVal = create.llvm.constant(llvmI64Ty, static_cast<int64_t>(0));
    // Output
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getOut());
    if (stacked || broadcasting23 || broadcasting1)
      outputZTensor =
          zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
              /*layout=*/ZDNN_3DS, /*originalDims=*/{S, M, P},
              /*isTransformed=*/true);
    else
      outputZTensor =
          zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
              /*layout=*/ZDNN_2D, /*originalDims=*/{M, P},
              /*isTransformed=*/true);

    // Ready to call zDNN MatMul.
    if (transposeA || transposeB) {
      callApi(rewriter, loc, module, apiRegistry, API::ZDNN_MATMUL_TRANSPOSE_OP,
          {toOpaquePtr(rewriter, loc, module, xZTensor.val),
              toOpaquePtr(rewriter, loc, module, yZTensor.val),
              toOpaquePtr(rewriter, loc, module, biasZTensor.val),
              transposeAVal, transposeBVal, opType,
              toOpaquePtr(rewriter, loc, module, outputZTensor.val)});
    } else if (broadcasting23 || broadcasting1) {
      callApi(rewriter, loc, module, apiRegistry, API::ZDNN_MATMUL_BCAST_OP,
          {toOpaquePtr(rewriter, loc, module, xZTensor.val),
              toOpaquePtr(rewriter, loc, module, yZTensor.val),
              toOpaquePtr(rewriter, loc, module, biasZTensor.val), opType,
              toOpaquePtr(rewriter, loc, module, outputZTensor.val)});
    } else {
      callApi(rewriter, loc, module, apiRegistry, API::ZDNN_MATMUL_OP,
          {toOpaquePtr(rewriter, loc, module, xZTensor.val),
              toOpaquePtr(rewriter, loc, module, yZTensor.val),
              toOpaquePtr(rewriter, loc, module, biasZTensor.val), opType,
              toOpaquePtr(rewriter, loc, module, outputZTensor.val)});
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  ApiRegistry apiRegistry;
};

class ZLowQuantizedMatMulLowering : public ConvertToLLVMPattern {
public:
  explicit ZLowQuantizedMatMulLowering(MLIRContext *context,
      LLVMTypeConverter &lowering_, ApiRegistry apiRegistry)
      : ConvertToLLVMPattern(
            ZLowQuantizedMatMulOp::getOperationName(), context, lowering_) {
    this->apiRegistry = apiRegistry;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    MLIRContext *context = module.getContext();
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

    ZLowQuantizedMatMulOp matmulOp = cast<ZLowQuantizedMatMulOp>(op);
    ZLowQuantizedMatMulOpAdaptor operandAdaptor(operands);

    // Inputs.
    // X
    Value X = operandAdaptor.getX();
    Value XRecScale = operandAdaptor.getXRecScale();
    Value XOffset = operandAdaptor.getXOffset();
    StringRef XQType = matmulOp.getXQType();
    // Y
    Value Y = operandAdaptor.getY();
    Value YRecScale = operandAdaptor.getYRecScale();
    Value YOffset = operandAdaptor.getYOffset();
    StringRef YQType = matmulOp.getYQType();
    // Bias
    Value Bias = operandAdaptor.getBias();
    Value BiasRecScale = operandAdaptor.getBiasRecScale();
    Value BiasOffset = operandAdaptor.getBiasOffset();
    StringRef BiasQType = matmulOp.getBiasQType();
    // Out
    Value Out = operandAdaptor.getOut();
    Value OutRecScale = operandAdaptor.getOutRecScale();
    Value OutOffset = operandAdaptor.getOutOffset();
    StringRef OutQType = matmulOp.getOutQType();

    // Types.
    Type llvmXElementTy = typeConverter->convertType(
        mlir::cast<MemRefType>(matmulOp.getX().getType()).getElementType());
    Type llvmYElementTy = typeConverter->convertType(
        mlir::cast<MemRefType>(matmulOp.getY().getType()).getElementType());
    Type llvmBiasElementTy = typeConverter->convertType(
        mlir::cast<MemRefType>(matmulOp.getBias().getType()).getElementType());
    Type llvmOutElementTy = typeConverter->convertType(
        mlir::cast<MemRefType>(matmulOp.getOut().getType()).getElementType());
    Type llvmF32Ty = rewriter.getF32Type();
    Type llvmI64Ty = rewriter.getI64Type();
    Type llvmZTensorTy = getZTensorStructTy(context);
    Type llvmZTensorPtrTy = krnl::getPointerType(context, llvmZTensorTy);

    bool stacked, broadcasting;
    if (matmulOp.getIsStacked() == -1)
      stacked = true;
    else
      stacked = false;
    if (matmulOp.getIsBcast() == -1)
      broadcasting = true;
    else
      broadcasting = false;

    // Get the dimensions of the original shape (the shape before stickifying)
    // used for creating zTensors.
    int dimCount = 3;
    if (stacked || broadcasting)
      dimCount = 4;
    std::vector<Value> dims = getDimsFromShapeMemRefBySize(
        rewriter, loc, module, operandAdaptor.getShape(), /*size=*/dimCount);
    // Dimensions: s, m, n, p;
    Value S, M, N, P;
    if (stacked || broadcasting) {
      S = dims[0];
      M = dims[1];
      N = dims[2];
      P = dims[3];
    } else {
      M = dims[0];
      N = dims[1];
      P = dims[2];
    }

    // Create zTensors.
    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);
    ZTensor xZTensor, yZTensor, biasZTensor, outputZTensor;
    // X
    zdnn_data_types zDNNDataType = llvmTypeToZDNNType(llvmXElementTy);
    zdnn_quantized_transform_types zDNNQType =
        getQuantizedTransformType(XQType);
    Value recScale = loadFromMemRef(create.llvm, llvmF32Ty, XRecScale, 0);
    Value offset = loadFromMemRef(create.llvm, llvmF32Ty, XOffset, 0);
    Value stickI8Ptr = zTensorHelper.getAlignedI8Ptr(X);
    if (stacked || broadcasting)
      xZTensor = zTensorHelper.getQuantizedZTensor(stickI8Ptr,
          /*dataType=*/zDNNDataType, /*layout=*/ZDNN_3DS,
          /*transformType=*/zDNNQType,
          /*originalDims=*/{S, M, N},
          /*recScale=*/recScale, /*offset=*/offset,
          /*isTransformed=*/true);
    else
      xZTensor = zTensorHelper.getQuantizedZTensor(stickI8Ptr,
          /*dataType=*/zDNNDataType, /*layout=*/ZDNN_2D,
          /*transformType=*/zDNNQType,
          /*originalDims=*/{M, N}, /*recScale=*/recScale, /*offset=*/offset,
          /*isTransformed=*/true);
    // Y
    zDNNDataType = llvmTypeToZDNNType(llvmYElementTy);
    zDNNQType = getQuantizedTransformType(YQType);
    recScale = loadFromMemRef(create.llvm, llvmF32Ty, YRecScale, 0);
    offset = loadFromMemRef(create.llvm, llvmF32Ty, YOffset, 0);
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(Y);
    if (stacked)
      yZTensor = zTensorHelper.getQuantizedZTensor(stickI8Ptr,
          /*dataType=*/zDNNDataType, /*layout=*/ZDNN_3DS,
          /*transformType=*/zDNNQType, /*originalDims=*/{S, N, P},
          /*recScale=*/recScale, /*offset=*/offset, /*isTransformed=*/true);
    else
      yZTensor = zTensorHelper.getQuantizedZTensor(stickI8Ptr,
          /*dataType=*/zDNNDataType, /*layout=*/ZDNN_2D,
          /*transformType=*/zDNNQType, /*originalDims=*/{N, P},
          /*recScale=*/recScale, /*offset=*/offset, /*isTransformed=*/true);
    // Bias
    zDNNDataType = llvmTypeToZDNNType(llvmBiasElementTy);
    zDNNQType = getQuantizedTransformType(BiasQType);
    recScale = loadFromMemRef(create.llvm, llvmF32Ty, BiasRecScale, 0);
    offset = loadFromMemRef(create.llvm, llvmF32Ty, BiasOffset, 0);
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(Bias);
    if (stacked)
      biasZTensor = zTensorHelper.getQuantizedZTensor(stickI8Ptr,
          /*dataType=*/zDNNDataType,
          /*layout=*/ZDNN_2DS,
          /*transformType=*/zDNNQType, /*originalDims=*/{S, P},
          /*recScale=*/recScale, /*offset=*/offset, /*isTransformed=*/true);
    else
      biasZTensor = zTensorHelper.getQuantizedZTensor(stickI8Ptr,
          /*dataType=*/zDNNDataType, /*layout=*/ZDNN_1D,
          /*transformType=*/zDNNQType, /*originalDims=*/{P},
          /*recScale=*/recScale, /*offset=*/offset, /*isTransformed=*/true);

    // Op_type
    Value opType = create.llvm.constant(
        llvmI64Ty, static_cast<int64_t>(NNPA_MATMUL_OP_ADDITION));

    // Min, Max clip values.
    Value clipMIN =
        create.llvm.constant(llvmI64Ty, static_cast<int64_t>(INT8_MIN));
    Value clipMAX =
        create.llvm.constant(llvmI64Ty, static_cast<int64_t>(INT8_MAX));

    // work_area.
    Value workArea;
    if (mlir::isa<NoneType>(matmulOp.getWorkArea().getType()))
      workArea = create.llvm.null(krnl::getI8PointerType(context));
    else
      workArea = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getWorkArea());

    // Output
    zDNNDataType = llvmTypeToZDNNType(llvmOutElementTy);
    zDNNQType = getQuantizedTransformType(OutQType);
    recScale = loadFromMemRef(create.llvm, llvmF32Ty, OutRecScale, 0);
    offset = loadFromMemRef(create.llvm, llvmF32Ty, OutOffset, 0);
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(Out);
    if (stacked || broadcasting)
      outputZTensor = zTensorHelper.getQuantizedZTensor(stickI8Ptr,
          /*dataType=*/zDNNDataType,
          /*layout=*/ZDNN_3DS,
          /*transformType=*/zDNNQType,
          /*originalDims=*/{S, M, P},
          /*recScale=*/recScale, /*offset=*/offset,
          /*isTransformed=*/true);
    else
      outputZTensor = zTensorHelper.getQuantizedZTensor(stickI8Ptr,
          /*dataType=*/zDNNDataType,
          /*layout=*/ZDNN_2D,
          /*transformType=*/zDNNQType,
          /*originalDims=*/{M, P},
          /*recScale=*/recScale, /*offset=*/offset,
          /*isTransformed=*/true);

    // Ready to call zDNN MatMul.
    Value disableClipping = create.llvm.constant(
        llvmI64Ty, static_cast<int64_t>(matmulOp.getDisableClipping()));
    Value dequantizeOutput = create.llvm.constant(
        llvmI64Ty, static_cast<int64_t>(matmulOp.getDequantizeOutput()));
    Value preComputedBias = create.llvm.constant(
        llvmI64Ty, static_cast<int64_t>(matmulOp.getPreComputedBias()));
    zlow::API apiName = API::ZDNN_QUANTIZED_MATMUL_OP;
    callApi(rewriter, loc, module, apiRegistry, apiName,
        {/*input_a=*/toOpaquePtr(rewriter, loc, module, xZTensor.val),
            /*input_b=*/toOpaquePtr(rewriter, loc, module, yZTensor.val),
            /*input_c=*/toOpaquePtr(rewriter, loc, module, biasZTensor.val),
            /*op_type=*/opType,
            /*clip_min=*/clipMIN,
            /*clip_max=*/clipMAX,
            /*disable_clipping=*/disableClipping,
            /*dequantized=*/dequantizeOutput,
            /*pre_computed=*/preComputedBias,
            /*work_area=*/workArea,
            /*output=*/
            toOpaquePtr(rewriter, loc, module, outputZTensor.val)});

    // Store the output rec_scale.
    Value recScalePtr = create.llvm.getElemPtr(llvmZTensorPtrTy, llvmZTensorTy,
        outputZTensor.val, ArrayRef<LLVM::GEPArg>{0, 6});
    Value outRecScale = create.llvm.load(llvmF32Ty, recScalePtr);
    storeToMemRef(create.llvm, outRecScale, OutRecScale, 0);
    // Store the output offset.
    Value offsetPtr = create.llvm.getElemPtr(llvmZTensorPtrTy, llvmZTensorTy,
        outputZTensor.val, ArrayRef<LLVM::GEPArg>{0, 7});
    Value outOffset = create.llvm.load(llvmF32Ty, offsetPtr);
    storeToMemRef(create.llvm, outOffset, OutOffset, 0);

    rewriter.eraseOp(op);
    return success();
  }

private:
  ApiRegistry apiRegistry;
};

class ZLowConv2DLowering : public ConvertToLLVMPattern {
public:
  explicit ZLowConv2DLowering(MLIRContext *context,
      LLVMTypeConverter &lowering_, ApiRegistry apiRegistry)
      : ConvertToLLVMPattern(
            ZLowConv2DOp::getOperationName(), context, lowering_) {
    this->apiRegistry = apiRegistry;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();
    MLIRContext *context = rewriter.getContext();
    ZLowConv2DOp convOp = mlir::cast<ZLowConv2DOp>(op);
    ZLowConv2DOpAdaptor operandAdaptor(operands);
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

    Type llvmElementTy = typeConverter->convertType(
        mlir::cast<MemRefType>(convOp.getInput().getType()).getElementType());

    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);

    // Some frequently used types and constants.
    Type llvmI64Ty = rewriter.getI64Type();

    // Get the dimensions of the original shape (the shape before stickifying)
    // used for creating zTensors.
    std::vector<Value> dims = getDimsFromShapeMemRefBySize(
        rewriter, loc, module, operandAdaptor.getShape(), /*size=*/7);
    // batch size
    Value N = dims[0];
    // channel in
    Value CIn = dims[1];
    // height in
    Value HIn = dims[2];
    // width in
    Value WIn = dims[3];
    // channel out
    Value COut = dims[4];
    // height out
    Value HOut = dims[5];
    // width out
    Value WOut = dims[6];
    // kernel shape
    ArrayRef<Attribute> kernelShapeArrayAttr =
        convOp.getKernelShape().getValue();
    // kernel height
    Value KH = create.llvm.constant(llvmI64Ty,
        static_cast<int64_t>(
            mlir::cast<IntegerAttr>(kernelShapeArrayAttr[0]).getInt()));
    // kernel width
    Value KW = create.llvm.constant(llvmI64Ty,
        static_cast<int64_t>(
            mlir::cast<IntegerAttr>(kernelShapeArrayAttr[1]).getInt()));

    // Get zDNN data type.
    zdnn_data_types zDNNDataType = llvmTypeToZDNNType(llvmElementTy);

    // Create a zTensor for input.
    Value stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getInput());
    ZTensor inputZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_NHWC, /*originalDims=*/{N, HIn, WIn, CIn},
            /*isTransformed=*/true);

    // Create zTensor for input kernel.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getInputKernel());
    ZTensor kernelZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_HWCK, /*originalDims=*/{KH, KW, CIn, COut},
            /*isTransformed=*/true);

    // Create zTensor for bias.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getInputBias());
    ZTensor biasZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_1D, /*originalDims=*/{COut},
            /*isTransformed=*/true);

    // Padding type.
    Value paddingType;
    if (convOp.getPaddingType().equals_insensitive("SAME_PADDING"))
      paddingType = create.llvm.constant(
          llvmI64Ty, static_cast<int64_t>(zdnn_pool_padding::SAME_PADDING));
    else if (convOp.getPaddingType().equals_insensitive("VALID_PADDING"))
      paddingType = create.llvm.constant(
          llvmI64Ty, static_cast<int64_t>(zdnn_pool_padding::VALID_PADDING));
    else
      llvm_unreachable("Unsupported padding type");

    // Strides
    ArrayRef<Attribute> strideArrayAttr = convOp.getStrides().getValue();
    Value strideHeight = create.llvm.constant(
        llvmI64Ty, static_cast<int64_t>(
                       mlir::cast<IntegerAttr>(strideArrayAttr[0]).getInt()));
    Value strideWidth = create.llvm.constant(
        llvmI64Ty, static_cast<int64_t>(
                       mlir::cast<IntegerAttr>(strideArrayAttr[1]).getInt()));

    // Activation function.
    Value actFunc;
    if (convOp.getActFunc().equals_insensitive("ACT_NONE"))
      actFunc = create.llvm.constant(
          llvmI64Ty, static_cast<int64_t>(zdnn_conv2d_act::CONV2D_ACT_NONE));
    else if (convOp.getActFunc().equals_insensitive("ACT_RELU"))
      actFunc = create.llvm.constant(
          llvmI64Ty, static_cast<int64_t>(zdnn_conv2d_act::CONV2D_ACT_RELU));
    else
      llvm_unreachable("Unsupported activation function");

    // Create zTensor for output.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getOutput());
    ZTensor outputZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_NHWC, /*originalDims=*/{N, HOut, WOut, COut},
            /*isTransformed=*/true);

    // Prepare nullpointer for the clipping value
    Value nullpointer = create.llvm.null(krnl::getI8PointerType(context));
    // Ready to call zDNN Conv2D.
    callApi(rewriter, loc, module, apiRegistry, API::ZDNN_CONV2D,
        {toOpaquePtr(rewriter, loc, module, inputZTensor.val),
            toOpaquePtr(rewriter, loc, module, kernelZTensor.val),
            toOpaquePtr(rewriter, loc, module, biasZTensor.val), paddingType,
            strideHeight, strideWidth, actFunc, nullpointer,
            toOpaquePtr(rewriter, loc, module, outputZTensor.val)});

    rewriter.eraseOp(op);
    return success();
  }

private:
  ApiRegistry apiRegistry;
};

template <typename POOLOP>
API getPool2DAPI() {
  return API::NULL_API;
}

template <>
API getPool2DAPI<ZLowAvgPool2DOp>() {
  return API::ZDNN_AVGPOOL2D;
}

template <>
API getPool2DAPI<ZLowMaxPool2DOp>() {
  return API::ZDNN_MAXPOOL2D;
}

template <typename POOLOP>
class ZLowPool2DLowering : public ConvertToLLVMPattern {
public:
  explicit ZLowPool2DLowering(MLIRContext *context,
      LLVMTypeConverter &lowering_, ApiRegistry apiRegistry)
      : ConvertToLLVMPattern(POOLOP::getOperationName(), context, lowering_) {
    this->apiRegistry = apiRegistry;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();
    POOLOP poolOp = mlir::cast<POOLOP>(op);
    typename POOLOP::Adaptor operandAdaptor(operands);
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

    Value input = operandAdaptor.getInput();
    Value shape = operandAdaptor.getShape();
    Value output = operandAdaptor.getOutput();
    Type llvmElementTy = typeConverter->convertType(
        mlir::cast<MemRefType>(op->getOperand(0).getType()).getElementType());

    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);

    // Some frequently used types and constants.
    Type llvmI64Ty = rewriter.getI64Type();

    // Get the dimensions of the original shape (the shape before stickifying)
    // used for creating zTensors.
    std::vector<Value> dims =
        getDimsFromShapeMemRefBySize(rewriter, loc, module, shape, /*size=*/6);
    // batch size
    Value N = dims[0];
    // channel in
    Value CIn = dims[1];
    // height in
    Value HIn = dims[2];
    // width in
    Value WIn = dims[3];
    // height out
    Value HOut = dims[4];
    // width out
    Value WOut = dims[5];
    // kernel shape
    ArrayRef<Attribute> kernelShapeArrayAttr =
        poolOp.getKernelShape().getValue();
    // kernel height
    Value KH = create.llvm.constant(llvmI64Ty,
        static_cast<int64_t>(
            mlir::cast<IntegerAttr>(kernelShapeArrayAttr[0]).getInt()));
    // kernel width
    Value KW = create.llvm.constant(llvmI64Ty,
        static_cast<int64_t>(
            mlir::cast<IntegerAttr>(kernelShapeArrayAttr[1]).getInt()));

    // Get zDNN data type.
    zdnn_data_types zDNNDataType = llvmTypeToZDNNType(llvmElementTy);

    // Create a zTensor for input.
    Value stickI8Ptr = zTensorHelper.getAlignedI8Ptr(input);
    ZTensor inputZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_NHWC, /*originalDims=*/{N, HIn, WIn, CIn},
            /*isTransformed=*/true);

    // Padding type.
    Value paddingType;
    if (poolOp.getPaddingType().equals_insensitive("SAME_PADDING"))
      paddingType = create.llvm.constant(
          llvmI64Ty, static_cast<int64_t>(zdnn_pool_padding::SAME_PADDING));
    else if (poolOp.getPaddingType().equals_insensitive("VALID_PADDING"))
      paddingType = create.llvm.constant(
          llvmI64Ty, static_cast<int64_t>(zdnn_pool_padding::VALID_PADDING));
    else
      llvm_unreachable("Unsupported padding type");

    // Strides
    ArrayRef<Attribute> strideArrayAttr = poolOp.getStrides().getValue();
    Value strideHeight = create.llvm.constant(
        llvmI64Ty, static_cast<int64_t>(
                       mlir::cast<IntegerAttr>(strideArrayAttr[0]).getInt()));
    Value strideWidth = create.llvm.constant(
        llvmI64Ty, static_cast<int64_t>(
                       mlir::cast<IntegerAttr>(strideArrayAttr[1]).getInt()));

    // Create zTensor for output.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(output);
    ZTensor outputZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_NHWC, /*originalDims=*/{N, HOut, WOut, CIn},
            /*isTransformed=*/true);

    // Ready to call zDNN Pool2D.
    callApi(rewriter, loc, module, apiRegistry, getPool2DAPI<POOLOP>(),
        {toOpaquePtr(rewriter, loc, module, inputZTensor.val), paddingType, KH,
            KW, strideHeight, strideWidth,
            toOpaquePtr(rewriter, loc, module, outputZTensor.val)});

    rewriter.eraseOp(op);
    return success();
  }

private:
  ApiRegistry apiRegistry;
};

class ZLowMeanReduce2DLowering : public ConvertToLLVMPattern {
public:
  explicit ZLowMeanReduce2DLowering(MLIRContext *context,
      LLVMTypeConverter &lowering_, ApiRegistry apiRegistry)
      : ConvertToLLVMPattern(
            ZLowMeanReduce2DOp::getOperationName(), context, lowering_) {
    this->apiRegistry = apiRegistry;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();
    ZLowMeanReduce2DOp meanOp = mlir::cast<ZLowMeanReduce2DOp>(op);
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

    ZLowMeanReduce2DOpAdaptor operandAdaptor(operands);
    Type llvmElementTy = typeConverter->convertType(
        mlir::cast<MemRefType>(meanOp.getInput().getType()).getElementType());

    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);

    // Some frequently used types and constants.
    Type llvmI64Ty = rewriter.getI64Type();
    Value oneI64 = create.llvm.constant(llvmI64Ty, static_cast<int64_t>(1));

    // Get the dimensions of the original shape (the shape before stickifying)
    // used for creating zTensors.
    std::vector<Value> dims = getDimsFromShapeMemRefBySize(
        rewriter, loc, module, operandAdaptor.getShape(), /*size=*/4);
    // batch size
    Value N = dims[0];
    // height in
    Value H = dims[1];
    // width in
    Value W = dims[2];
    // channel in
    Value C = dims[3];
    // Get zDNN data type.
    zdnn_data_types zDNNDataType = llvmTypeToZDNNType(llvmElementTy);

    // Create a zTensor for input.
    Value stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getInput());
    ZTensor inputZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_NHWC, /*originalDims=*/{N, H, W, C},
            /*isTransformed=*/true);

    // Create zTensor for output.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getOutput());
    ZTensor outputZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_NHWC, /*originalDims=*/{N, oneI64, oneI64, C},
            /*isTransformed=*/true);

    // Ready to call zDNN MeanReduce2D.
    callApi(rewriter, loc, module, apiRegistry, API::ZDNN_MEANREDUCE2D,
        {toOpaquePtr(rewriter, loc, module, inputZTensor.val),
            toOpaquePtr(rewriter, loc, module, outputZTensor.val)});

    rewriter.eraseOp(op);
    return success();
  }

private:
  ApiRegistry apiRegistry;
};

class ZLowBatchNormLowering : public ConvertToLLVMPattern {
public:
  explicit ZLowBatchNormLowering(MLIRContext *context,
      LLVMTypeConverter &lowering_, ApiRegistry apiRegistry)
      : ConvertToLLVMPattern(
            ZLowBatchNormOp::getOperationName(), context, lowering_) {
    this->apiRegistry = apiRegistry;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();
    ZLowBatchNormOp batchnormOp = mlir::cast<ZLowBatchNormOp>(op);

    ZLowBatchNormOpAdaptor operandAdaptor(operands);
    Type llvmElementTy = typeConverter->convertType(
        mlir::cast<MemRefType>(batchnormOp.getInput().getType())
            .getElementType());

    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);

    // Get the dimensions of the original shape (the shape before stickifying)
    // used for creating zTensors.
    std::vector<Value> dims = getDimsFromShapeMemRefBySize(
        rewriter, loc, module, operandAdaptor.getShape(), /*size=*/4);
    // batch size
    Value N = dims[0];
    // height in
    Value H = dims[1];
    // width in
    Value W = dims[2];
    // channel in
    Value C = dims[3];

    // Get zDNN data type.
    zdnn_data_types zDNNDataType = llvmTypeToZDNNType(llvmElementTy);

    // Create a zTensor for input.
    Value stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getInput());
    ZTensor inputZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_NHWC, /*originalDims=*/{N, H, W, C},
            /*isTransformed=*/true);

    // Create a zTensor for A.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getA());
    ZTensor aZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_1D, /*originalDims=*/{C},
            /*isTransformed=*/true);

    // Create a zTensor for B.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getB());
    ZTensor bZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_1D, /*originalDims=*/{C},
            /*isTransformed=*/true);

    // Create zTensor for output.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.getOutput());
    ZTensor outputZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_NHWC, /*originalDims=*/{N, H, W, C},
            /*isTransformed=*/true);

    // Ready to call zDNN BatchNorm.
    callApi(rewriter, loc, module, apiRegistry, API::ZDNN_BATCHNORM,
        {toOpaquePtr(rewriter, loc, module, inputZTensor.val),
            toOpaquePtr(rewriter, loc, module, aZTensor.val),
            toOpaquePtr(rewriter, loc, module, bZTensor.val),
            toOpaquePtr(rewriter, loc, module, outputZTensor.val)});

    rewriter.eraseOp(op);
    return success();
  }

private:
  ApiRegistry apiRegistry;
};

class ZLowDLF16ToF32Lowering : public ConvertToLLVMPattern {
public:
  explicit ZLowDLF16ToF32Lowering(MLIRContext *context,
      LLVMTypeConverter &lowering_, ApiRegistry apiRegistry)
      : ConvertToLLVMPattern(
            ZLowConvertDLF16ToF32Op::getOperationName(), context, lowering_) {
    this->apiRegistry = apiRegistry;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();
    ZLowConvertDLF16ToF32Op::Adaptor operandAdaptor(operands);
    Value input = operandAdaptor.getInput();
    Type i16Ty = rewriter.getI16Type();
    Type i32Ty = rewriter.getI32Type();
    Type f32Ty = rewriter.getF32Type();

    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
    Value outputF32;
    Value inputI16 = create.llvm.bitcast(i16Ty, input);

    if (FUNC_CALL_FOR_DLF16_CONVERSION) {
      // This code is for the purpose of testing the correctness of the
      // generated LLVM code.
      outputF32 = callApi(
          rewriter, loc, module, apiRegistry, API::DLF16_TO_F32, {inputI16});
    } else {
      if (SIMD_FOR_DLF16_CONVERSION) {
        // a vector of 8 elements of i16 - for input
        Type vecTypeI16 = LLVM::getFixedVectorType(i16Ty, 8);
        // a vector of 4 elements of i32 - for output
        Type vecTypeI32 = LLVM::getFixedVectorType(i32Ty, 4);
        Type vecTypeF32 = LLVM::getFixedVectorType(f32Ty, 4);

        // SIMD instruction in string for z/Linux and z/OS.
        // Convert and lengthen from DLF16: VCLFN(H/L) V1,V2,M3,M4
        // M3 = 2 = FP32, M4 = 0 = DLF16
        // Note the spaces are required by the z/OS assembler.
        const char *asmStr = "       VCLFNH $0,$2,2,0       \n\t"
                             "       VCLFNL $1,$2,2,0       \n\t";
        const char *asmConstraints = "=&v,=v,v";

        // Prepare the input vector.
        // Only care about the first element.
        Value inputVecI16 = rewriter.create<LLVM::UndefOp>(loc, vecTypeI16);
        inputVecI16 = create.llvm.insertElement(inputVecI16, inputI16, 0);
        SmallVector<Value> asmVals{inputVecI16};

        // Emit SIMD instruction for conversion.
        Value outVecI32Struct =
            rewriter
                .create<LLVM::InlineAsmOp>(loc,
                    LLVM::LLVMStructType::getLiteral(rewriter.getContext(),
                        {vecTypeI32, vecTypeI32}, /*Packed=*/false),
                    /*operands=*/asmVals,
                    /*asm_string=*/asmStr,
                    /*constraints=*/asmConstraints, /*has_side_effects=*/false,
                    /*is_align_stack=*/false,
                    /*asm_dialect=*/LLVM::AsmDialectAttr(),
                    /*operand_attrs=*/ArrayAttr())
                .getResult(0);
        Value outVecI32 =
            create.llvm.extractValue(vecTypeI32, outVecI32Struct, 0);
        Value outVecF32 = create.llvm.bitcast(vecTypeF32, outVecI32);
        outputF32 = create.llvm.extractElement(f32Ty, outVecF32, 0);
      } else {
        // Generating LLVM instruction here.
        // This code is equivalent to the one generated by clang:
        // `clang -emit-llvm convert_dlf16_to_f32.cpp  -S -O3`
        // where `convert_dlf16_to_f32.cpp` can be found at
        // https://github.com/tungld/onnx-mlir-tools/blob/main/convert_dlf16_to_f32.cpp
        Value inputI32 = create.llvm.zext(i32Ty, inputI16);
        // ~DLF16_SIGN
        Value c32767 = create.llvm.constant(i32Ty, static_cast<int64_t>(32767));
        // dlf16 & ~DLF16_SIGN
        Value v19 = create.llvm.andi(inputI32, c32767);
        Value c0 = create.llvm.constant(i32Ty, static_cast<int64_t>(0));

        // Split the block right before the current op into two blocks.
        Block *currentBlock = rewriter.getInsertionBlock();
        // New block contains the terminator of the current block.
        Block *newBlock =
            currentBlock->splitBlock(rewriter.getInsertionPoint());

        // Add a block for zero case.
        Block *trueBlock = rewriter.createBlock(currentBlock->getParent(),
            std::next(Region::iterator(currentBlock)));

        // Add a block for non-zero case.
        Block *falseBlock = rewriter.createBlock(
            trueBlock->getParent(), std::next(Region::iterator(trueBlock)));

        // Add a new block that acts as a phi node.
        Block *endBlock = rewriter.createBlock(
            newBlock->getParent(), Region::iterator(newBlock), f32Ty, loc);
        rewriter.mergeBlocks(newBlock, endBlock, {});

        // Emit `if (v19 == 0) then trueBlock else falseBlock`
        rewriter.setInsertionPointToEnd(currentBlock);
        Value v19Zero = create.llvm.icmp(LLVM::ICmpPredicate::eq, v19, c0);
        create.llvm.condBr(v19Zero, trueBlock, {}, falseBlock, {});

        // Emit code for zero case.
        rewriter.setInsertionPointToEnd(trueBlock);
        Value cf0 =
            create.llvm.constant(f32Ty, static_cast<float>(0.000000e+00));
        Value cfm0 =
            create.llvm.constant(f32Ty, static_cast<float>(-0.000000e+00));
        Value c32768 = create.llvm.constant(i32Ty, static_cast<int64_t>(32768));
        Value v20 = create.llvm.andi(inputI32, c32768);
        Value v21 = create.llvm.icmp(LLVM::ICmpPredicate::eq, v20, c0);
        Value v22 = create.llvm.select(v21, cf0, cfm0);
        create.llvm.br({v22}, endBlock);

        // Emit code for non-zero case.
        rewriter.setInsertionPointToEnd(falseBlock);
        {
          Block *condBlock = rewriter.getInsertionBlock();
          Block *defaultBlock =
              condBlock->splitBlock(rewriter.getInsertionPoint());

          rewriter.setInsertionPointToEnd(condBlock);
          Value nan =
              create.llvm.constant(f32Ty, static_cast<float>(0x7FC00000));
          Value inf = create.llvm.constant(i32Ty, static_cast<int64_t>(32767));
          Value v19Inf = create.llvm.icmp(LLVM::ICmpPredicate::eq, v19, inf);
          // Emit `if (v19 == inf) then endBlock(nan) else defaultBlock`
          create.llvm.condBr(v19Inf, endBlock, {nan}, defaultBlock, {});

          // Emit code for non-infinity case.
          rewriter.setInsertionPointToEnd(defaultBlock);
          Value c14 = create.llvm.constant(i32Ty, static_cast<int64_t>(14));
          Value c16 = create.llvm.constant(i32Ty, static_cast<int64_t>(16));
          Value cm2147483648 =
              create.llvm.constant(i32Ty, static_cast<int64_t>(-2147483648));
          Value c528482304 =
              create.llvm.constant(i32Ty, static_cast<int64_t>(528482304));
          Value c805306368 =
              create.llvm.constant(i32Ty, static_cast<int64_t>(805306368));
          Value c8372224 =
              create.llvm.constant(i32Ty, static_cast<int64_t>(8372224));
          Value v23 = create.llvm.shl(inputI32, c16);
          Value v24 = create.llvm.andi(v23, cm2147483648);
          Value v25 = create.llvm.shl(inputI32, c14);
          Value v26 = create.llvm.andi(v25, c528482304);
          Value v27 = create.llvm.add(v26, c805306368);
          Value v28 = create.llvm.ori(v27, v24);
          Value v29 = create.llvm.andi(v25, c8372224);
          Value v30 = create.llvm.ori(v28, v29);
          Value v31 = create.llvm.bitcast(f32Ty, v30);
          create.llvm.br({v31}, endBlock);
        }

        rewriter.setInsertionPoint(op);
        outputF32 = endBlock->getArgument(0);
      }
    }

    rewriter.replaceOp(op, {outputF32});
    return success();
  }

private:
  ApiRegistry apiRegistry;
};

class ZLowF32ToDLF16Lowering : public ConvertToLLVMPattern {
public:
  explicit ZLowF32ToDLF16Lowering(MLIRContext *context,
      LLVMTypeConverter &lowering_, ApiRegistry apiRegistry)
      : ConvertToLLVMPattern(
            ZLowConvertF32ToDLF16Op::getOperationName(), context, lowering_) {
    this->apiRegistry = apiRegistry;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Location loc = op->getLoc();
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

    Type llvmF16Ty = rewriter.getF16Type();
    Type i16Ty = rewriter.getI16Type();
    Type f16Ty = rewriter.getF16Type();
    Type i32Ty = rewriter.getI32Type();

    ZLowConvertF32ToDLF16Op::Adaptor operandAdaptor(operands);
    Value input = operandAdaptor.getInput();
    Value outputDLF16;

    if (FUNC_CALL_FOR_DLF16_CONVERSION) {
      // This code is for the purpose of testing the correctness of the
      // generated LLVM code.
      Value outputI16 = callApi(
          rewriter, loc, module, apiRegistry, API::F32_TO_DLF16, {input});
      outputDLF16 = create.llvm.bitcast(llvmF16Ty, outputI16);
    } else {
      if (SIMD_FOR_DLF16_CONVERSION) {
        // a vector of 4 elements of i32 - for input
        Type vecTypeI32 = LLVM::getFixedVectorType(i32Ty, 4);
        // a vector of 8 elements of i16 - for output
        Type vecTypeI16 = LLVM::getFixedVectorType(i16Ty, 8);
        Type vecTypeF16 = LLVM::getFixedVectorType(f16Ty, 8);

        // SIMD instruction in string for z/Linux and z/OS.
        // Convert and round to DLF16: VCRNF V1,V2,V3,M4,M5
        // M4 = 0 = DLF16, M5 = 2 = FP32
        // Note the spaces are required by the z/OS assembler.
        const char *asmStr = "       VCRNF $0,$1,$2,0,2         \n\t";
        const char *asmConstraints = "=v,v,v";

        // Prepare two input vectors: each for left/right four elements.
        // Only care about the first element.
        Value inputI32 = create.llvm.bitcast(i32Ty, input);
        Value inputVecI32Left = rewriter.create<LLVM::UndefOp>(loc, vecTypeI32);
        inputVecI32Left =
            create.llvm.insertElement(inputVecI32Left, inputI32, 0);
        Value inputVecI32Right =
            rewriter.create<LLVM::UndefOp>(loc, vecTypeI32);
        SmallVector<Value> asmVals{inputVecI32Left, inputVecI32Right};

        // Emit SIMD instruction for conversion.
        Value outVecI16 =
            rewriter
                .create<LLVM::InlineAsmOp>(loc, vecTypeI16,
                    /*operands=*/asmVals,
                    /*asm_string=*/asmStr,
                    /*constraints=*/asmConstraints, /*has_side_effects=*/false,
                    /*is_align_stack=*/false,
                    /*asm_dialect=*/LLVM::AsmDialectAttr(),
                    /*operand_attrs=*/ArrayAttr())
                .getResult(0);
        Value outVecDLF16 = create.llvm.bitcast(vecTypeF16, outVecI16);
        outputDLF16 = create.llvm.extractElement(f16Ty, outVecDLF16, 0);
      } else {
        // Generating LLVM instruction here.
        // This code is equivalent to the one generated by clang:
        // `clang -emit-llvm convert_f32_to_dlf16.cpp  -S -O3`
        // where `convert_f32_to_dlf16.cpp` can be found at
        // https://github.com/tungld/onnx-mlir-tools/blob/main/convert_f32_to_dlf16.cpp
        Value c0 = create.llvm.constant(i32Ty, static_cast<int64_t>(0));
        Value c9 = create.llvm.constant(i32Ty, static_cast<int64_t>(9));
        Value c14 = create.llvm.constant(i32Ty, static_cast<int64_t>(14));
        Value c16 = create.llvm.constant(i32Ty, static_cast<int64_t>(16));
        Value c23 = create.llvm.constant(i32Ty, static_cast<int64_t>(23));
        Value c255 = create.llvm.constant(i32Ty, static_cast<int64_t>(255));
        Value c8192 = create.llvm.constant(i32Ty, static_cast<int64_t>(8192));
        Value c32767 = create.llvm.constant(i32Ty, static_cast<int64_t>(32767));
        Value c32768 = create.llvm.constant(i32Ty, static_cast<int64_t>(32768));
        Value c32256 = create.llvm.constant(i32Ty, static_cast<int64_t>(32256));
        Value c8388607 =
            create.llvm.constant(i32Ty, static_cast<int64_t>(8388607));
        Value c8380415 =
            create.llvm.constant(i32Ty, static_cast<int64_t>(8380415));
        Value c1342152704 =
            create.llvm.constant(i32Ty, static_cast<int64_t>(1342152704));
        Value c2147475456 =
            create.llvm.constant(i32Ty, static_cast<int64_t>(2147475456));
        Value cm1 = create.llvm.constant(i32Ty, static_cast<int64_t>(-1));
        Value cm95 = create.llvm.constant(i32Ty, static_cast<int64_t>(-95));
        Value cm96 = create.llvm.constant(i32Ty, static_cast<int64_t>(-96));
        Value inputI32 = create.llvm.bitcast(i32Ty, input);
        Value v24 = create.llvm.lshr(inputI32, c23);
        Value v25 = create.llvm.andi(v24, c255);
        Value v26 = create.llvm.andi(inputI32, c8388607);
        Value v27 = create.llvm.add(v26, c8192);
        Value v28 = create.llvm.icmp(LLVM::ICmpPredicate::ugt, v26, c8380415);
        Value v29 = create.llvm.select(v28, cm95, cm96);
        Value v30 = create.llvm.add(v29, v25);
        Value v31 = create.llvm.lshr(inputI32, c16);
        Value v32 = create.llvm.andi(v31, c32768);
        Value v33 = create.llvm.icmp(LLVM::ICmpPredicate::sgt, v30, cm1);

        // Split the block right before the current op into two blocks.
        Block *currentBlock = rewriter.getInsertionBlock();
        // New block contains the terminator of the current block.
        Block *newBlock =
            currentBlock->splitBlock(rewriter.getInsertionPoint());

        // Add a new block for the true branch of the conditional statement we
        // will add.
        Block *trueBlock = rewriter.createBlock(currentBlock->getParent(),
            std::next(Region::iterator(currentBlock)));

        // Add a new block that acts as a phi node.
        Block *endBlock = rewriter.createBlock(newBlock->getParent(),
            Region::iterator(newBlock), v32.getType(), loc);
        rewriter.mergeBlocks(newBlock, endBlock, {});

        rewriter.setInsertionPointToEnd(currentBlock);
        create.llvm.condBr(v33, trueBlock, {}, endBlock, {v32});

        rewriter.setInsertionPointToEnd(trueBlock);
        {
          Block *currentBlock = rewriter.getInsertionBlock();
          Block *thenBlock =
              currentBlock->splitBlock(rewriter.getInsertionPoint());
          Block *elseBlock = rewriter.createBlock(
              thenBlock->getParent(), std::next(Region::iterator(thenBlock)));

          rewriter.setInsertionPointToEnd(currentBlock);
          Value v34 = create.llvm.andi(inputI32, c2147475456);
          Value v35 =
              create.llvm.icmp(LLVM::ICmpPredicate::ult, v34, c1342152704);
          create.llvm.condBr(v35, thenBlock, {}, elseBlock, {});

          rewriter.setInsertionPointToEnd(thenBlock);
          Value v36 = create.llvm.shl(v30, c9);
          Value v37 = create.llvm.andi(v36, c32256);
          Value v38 = create.llvm.lshr(v27, c14);
          Value v39 = create.llvm.select(v28, c0, v38);
          Value v40 = create.llvm.ori(v39, v37);
          Value v41 = create.llvm.ori(v40, v32);
          create.llvm.br({v41}, endBlock);

          rewriter.setInsertionPointToEnd(elseBlock);
          Value v42 = create.llvm.ori(v31, c32767);
          create.llvm.br({v42}, endBlock);
        }

        rewriter.setInsertionPoint(op);
        Value outputI32 = endBlock->getArgument(0);
        Value outputI16 = create.llvm.trunc(i16Ty, outputI32);
        outputDLF16 = create.llvm.bitcast(llvmF16Ty, outputI16);
      }
    }

    rewriter.replaceOp(op, {outputDLF16});
    return success();
  }

private:
  ApiRegistry apiRegistry;
};

class ZLowDLF16ToF32VectorLowering : public ConvertToLLVMPattern {
public:
  explicit ZLowDLF16ToF32VectorLowering(MLIRContext *context,
      LLVMTypeConverter &lowering_, ApiRegistry apiRegistry)
      : ConvertToLLVMPattern(ZLowConvertDLF16ToF32VectorOp::getOperationName(),
            context, lowering_) {
    this->apiRegistry = apiRegistry;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

    // Vector types.
    Type vecTypeI16 = LLVM::getFixedVectorType(rewriter.getI16Type(), 8);
    Type vecTypeI32 = LLVM::getFixedVectorType(rewriter.getI32Type(), 4);
    Type vecTypeF32 = LLVM::getFixedVectorType(rewriter.getF32Type(), 4);

    // Use integer as container.
    ZLowConvertDLF16ToF32VectorOp::Adaptor operandAdaptor(operands);
    Value inputVecI16 =
        create.llvm.bitcast(vecTypeI16, operandAdaptor.getInput());

    // SIMD instruction in string for z/Linux and z/OS.
    // Note this .insn version of asmStr was used previously for z/Linux.
    // const char *asmStr = ".insn vrr,0xe60000000056,$0,$2,0,2,0,0 \n\t"
    //                      ".insn vrr,0xe6000000005E,$1,$2,0,2,0,0 \n\t";
    // Convert and lengthen from DLF16: VCLFN(H/L) V1,V2,M3,M4
    // M3 = 2 = FP32, M4 = 0 = DLF16
    // Note the spaces are required by the z/OS assembler.
    const char *asmStr = "       VCLFNH $0,$2,2,0       \n\t"
                         "       VCLFNL $1,$2,2,0       \n\t";
    const char *asmConstraints = "=&v,=v,v";
    SmallVector<Value> asmVals{inputVecI16};
    Value outVecI32Struct =
        rewriter
            .create<LLVM::InlineAsmOp>(loc,
                LLVM::LLVMStructType::getLiteral(rewriter.getContext(),
                    {vecTypeI32, vecTypeI32}, /*Packed=*/false),
                /*operands=*/asmVals,
                /*asm_string=*/asmStr,
                /*constraints=*/asmConstraints, /*has_side_effects=*/false,
                /*is_align_stack=*/false,
                /*asm_dialect=*/LLVM::AsmDialectAttr(),
                /*operand_attrs=*/ArrayAttr())
            .getResult(0);

    Value outVecI32H = create.llvm.extractValue(vecTypeI32, outVecI32Struct, 0);
    Value outVecI32L = create.llvm.extractValue(vecTypeI32, outVecI32Struct, 1);

    Value outVecF32H = create.llvm.bitcast(vecTypeF32, outVecI32H);
    Value outVecF32L = create.llvm.bitcast(vecTypeF32, outVecI32L);
    rewriter.replaceOp(op, {outVecF32H, outVecF32L});
    return success();
  }

private:
  ApiRegistry apiRegistry;
};

class ZLowF32ToDLF16VectorLowering : public ConvertToLLVMPattern {
public:
  explicit ZLowF32ToDLF16VectorLowering(MLIRContext *context,
      LLVMTypeConverter &lowering_, ApiRegistry apiRegistry)
      : ConvertToLLVMPattern(ZLowConvertF32ToDLF16VectorOp::getOperationName(),
            context, lowering_) {
    this->apiRegistry = apiRegistry;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

    // Vector types.
    Type vecTypeI16 = LLVM::getFixedVectorType(rewriter.getI16Type(), 8);
    Type vecTypeI32 = LLVM::getFixedVectorType(rewriter.getI32Type(), 4);
    Type vecTypeF16 = LLVM::getFixedVectorType(rewriter.getF16Type(), 8);

    // Use integer as container.
    ZLowConvertF32ToDLF16VectorOp::Adaptor operandAdaptor(operands);
    Value vecI32H = create.llvm.bitcast(vecTypeI32, operandAdaptor.getInput1());
    Value vecI32L = create.llvm.bitcast(vecTypeI32, operandAdaptor.getInput2());

    // SIMD instruction in string for z/Linux and z/OS.
    // Note this .insn version of asmStr was used previously for z/Linux.
    // asmStr = ".insn vrr,0xe60000000075,$0,$1,$2,0,2,0";
    // Convert and round to DLF16: VCRNF V1,V2,V3,M4,M5
    // M4 = 0 = DLF16, M5 = 2 = FP32
    // Note the spaces are required by the z/OS assembler.
    const char *asmStr = "       VCRNF $0,$1,$2,0,2         \n\t";
    const char *asmConstraints = "=v,v,v";
    SmallVector<Value> asmVals{vecI32H, vecI32L};

    // Emit SIMD instruction for conversion.
    Value outVecI16 =
        rewriter
            .create<LLVM::InlineAsmOp>(loc, vecTypeI16,
                /*operands=*/asmVals,
                /*asm_string=*/asmStr,
                /*constraints=*/asmConstraints, /*has_side_effects=*/false,
                /*is_align_stack=*/false,
                /*asm_dialect=*/LLVM::AsmDialectAttr(),
                /*operand_attrs=*/ArrayAttr())
            .getResult(0);

    Value outVecF16 = create.llvm.bitcast(vecTypeF16, outVecI16);
    rewriter.replaceOp(op, {outVecF16});
    return success();
  }

private:
  ApiRegistry apiRegistry;
};

void populateZLowToLLVMConversionPattern(mlir::RewritePatternSet &patterns,
    mlir::LLVMTypeConverter &typeConverter, mlir::MLIRContext *ctx) {
  ApiRegistry apiRegistry = RegisterAllApis(ctx);
  // clang-format off
  patterns.insert<
      ZLowStickLowering,
      ZLowQuantizedStickLowering,
      ZLowUnstickLowering,
      ZLowStickForLSTMLowering,
      ZLowStickForGRULowering,
      // Activation operations
      ZLowSoftmaxOpLowering,
      // RNN operations
      ZLowLSTMLowering,
      ZLowGRULowering,
      // Other operations
      ZLowMatMulLowering,
      ZLowQuantizedMatMulLowering,
      ZLowConv2DLowering,
      ZLowMeanReduce2DLowering,
      ZLowBatchNormLowering,
      ZLowLeakyReluLowering,
      // Scalar operations
      ZLowDLF16ToF32Lowering,
      ZLowF32ToDLF16Lowering,
      // Vector operations
      ZLowDLF16ToF32VectorLowering,
      ZLowF32ToDLF16VectorLowering
    >(ctx, typeConverter, apiRegistry);
  patterns.insert<
      // Elementwise operations
      ZLowBinaryElementwiseOpLowering<ZLowAddOp>,
      ZLowBinaryElementwiseOpLowering<ZLowSubOp>,
      ZLowBinaryElementwiseOpLowering<ZLowMulOp>,
      ZLowBinaryElementwiseOpLowering<ZLowDivOp>,
      ZLowBinaryElementwiseOpLowering<ZLowMinOp>,
      ZLowBinaryElementwiseOpLowering<ZLowMaxOp>,
      // Unary operations
      ZLowUnaryElementwiseOpLowering<ZLowLogOp>,
      ZLowUnaryElementwiseOpLowering<ZLowExpOp>,
      ZLowUnaryElementwiseOpLowering<ZLowInvSqrtOp>,
      // Activation operations
      ZLowUnaryElementwiseOpLowering<ZLowReluOp>,
      ZLowUnaryElementwiseOpLowering<ZLowGeluOp>,
      ZLowUnaryElementwiseOpLowering<ZLowTanhOp>,
      ZLowUnaryElementwiseOpLowering<ZLowSigmoidOp>,
      ZLowUnaryElementwiseOpLowering<ZLowSqrtOp>,
      // Other operations
      ZLowPool2DLowering<ZLowAvgPool2DOp>,
      ZLowPool2DLowering<ZLowMaxPool2DOp>,
      ZLowReduceLowering<ZLowReduceMaxOp>,
      ZLowReduceLowering<ZLowReduceMinOp>
    >(ctx, typeConverter, apiRegistry);
  // clang-format on
}

} // namespace zlow
} // namespace onnx_mlir
