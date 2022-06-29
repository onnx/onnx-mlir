/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- ZLowToLLVM.cpp - Lowering from ZLow to LLVM ---------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
//
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"

#include "src/Accelerators/NNPA/Conversion/ZLowToLLVM/ZLowToLLVM.hpp"
#include "src/Accelerators/NNPA/Conversion/ZLowToLLVM/ZLowToLLVMCommon.hpp"
#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp"
#include "src/Accelerators/NNPA/Support/LayoutHelper.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "zdnn.h"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
namespace zlow {

zdnn_data_layouts UNDEFINED_ZDNN_LAYOUT = (zdnn_data_layouts)255;

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
API APIFor<ZLowReluOp>() {
  return API::ZDNN_RELU;
}
template <>
API APIFor<ZLowTanhOp>() {
  return API::ZDNN_TANH;
}
template <>
API APIFor<ZLowSigmoidOp>() {
  return API::ZDNN_SIGMOID;
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

    ZLowStickOpAdaptor operandAdaptor(operands);
    Type llvmElementTy = operandAdaptor.X()
                             .getType()
                             .dyn_cast<LLVM::LLVMStructType>()
                             .getBody()[0]
                             .cast<LLVM::LLVMPointerType>()
                             .getElementType();

    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);

    // Get the dimensions of the original shape (the shape before stickifying)
    // used for creating a zTensor. For 'zLow.stick', the original shape is
    // obtained from the first argument.
    SmallVector<Value, 3> dims;
    getDimsFromMemRef(rewriter, loc, module, operandAdaptor.X(), dims);

    // Get zDNN data type.
    zdnn_data_types zDNNDataType = llvmTypeToZDNNType(llvmElementTy);

    // Get zDNN data layout.
    zdnn_data_layouts zDNNDataLayout = convertLayoutAttrToZDNNDataLayout(
        dims.size(), dyn_cast_or_null<ZLowStickOp>(op).layoutAttr());

    // Create a zTensor.
    Value stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.Out());
    ZTensor zTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/zDNNDataLayout, /*originalDims=*/dims,
            /*isTransformed=*/false);

    // Ready to stickify.
    Value unstickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.X());
    callApi(rewriter, loc, module, apiRegistry, API::ZDNN_TRANSFORM_ZTENSOR,
        {toOpaquePtr(rewriter, loc, module, zTensor.val), unstickI8Ptr});

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

    ZLowStickForLSTMOpAdaptor operandAdaptor(operands);
    Type llvmElementTy = operandAdaptor.f_gate()
                             .getType()
                             .dyn_cast<LLVM::LLVMStructType>()
                             .getBody()[0]
                             .cast<LLVM::LLVMPointerType>()
                             .getElementType();

    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);

    // Get the dimensions of the original shape (the shape before stickifying)
    // used for creating a zTensor. F, I, C, O gates have the same shape. Thus,
    // they share the dimensions.
    SmallVector<Value, 3> dims;
    getDimsFromMemRef(rewriter, loc, module, operandAdaptor.f_gate(), dims);

    // Get zDNN data type.
    zdnn_data_types zDNNDataType = llvmTypeToZDNNType(llvmElementTy);

    // Get zDNN data layout and concatInfo
    zdnn_data_layouts zDNNDataLayout;
    zdnn_concat_info zDNNConcatInfo;
    StringRef prevLayerStr =
        dyn_cast_or_null<ZLowStickForLSTMOp>(op).prev_layer();
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
    Value stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.out());
    ZTensor zTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/zDNNDataLayout, /*originalDims=*/dims,
            /*isTransformed=*/false, /*isConcat=*/true,
            /*concatInfo=*/zDNNConcatInfo);

    // Ready to stickify.
    Value fGatePtr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.f_gate());
    Value iGatePtr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.i_gate());
    Value cGatePtr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.c_gate());
    Value oGatePtr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.o_gate());
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

    ZLowStickForGRUOpAdaptor operandAdaptor(operands);
    Type llvmElementTy = operandAdaptor.z_gate()
                             .getType()
                             .dyn_cast<LLVM::LLVMStructType>()
                             .getBody()[0]
                             .cast<LLVM::LLVMPointerType>()
                             .getElementType();

    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);

    // Get the dimensions of the original shape (the shape before stickifying)
    // used for creating a zTensor. Z, R, H gates have the same shape. Thus,
    // they share the dimensions.
    SmallVector<Value, 3> dims;
    getDimsFromMemRef(rewriter, loc, module, operandAdaptor.z_gate(), dims);

    // Get zDNN data type.
    zdnn_data_types zDNNDataType = llvmTypeToZDNNType(llvmElementTy);

    // Get zDNN data layout.
    zdnn_data_layouts zDNNDataLayout;
    zdnn_concat_info zDNNConcatInfo;
    StringRef prevLayerStr =
        dyn_cast_or_null<ZLowStickForGRUOp>(op).prev_layer();
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
    Value stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.out());
    ZTensor zTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/zDNNDataLayout, /*originalDims=*/dims,
            /*isTransformed=*/false, /*isConcat=*/true,
            /*concatInfo=*/zDNNConcatInfo);

    // Ready to stickify.
    Value zGatePtr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.z_gate());
    Value rGatePtr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.r_gate());
    Value hGatePtr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.h_gate());
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
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

    ZLowLSTMOpAdaptor operandAdaptor(operands);
    Type llvmElementTy = operandAdaptor.input()
                             .getType()
                             .dyn_cast<LLVM::LLVMStructType>()
                             .getBody()[0]
                             .cast<LLVM::LLVMPointerType>()
                             .getElementType();

    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);

    // Some frequently used types and constants.
    Type llvmI64Ty = rewriter.getI64Type();
    Value oneI64 = create.llvm.constant(llvmI64Ty, (int64_t)1);

    // Get the dimensions of the original shape (the shape before stickifying)
    // used for creating zTensors.
    std::vector<Value> dims = getDimsFromShapeMemRefBySize(
        rewriter, loc, module, operandAdaptor.shape(), /*size=*/5);
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

    StringRef prevLayerStr = dyn_cast_or_null<ZLowLSTMOp>(op).prev_layer();
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
    Value stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.input());
    ZTensor inputZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_3DS, /*originalDims=*/{T, B, F},
            /*isTransformed=*/true);

    // Create zTensor for h0.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.h0());
    ZTensor h0ZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_3DS, /*originalDims=*/{D, B, H},
            /*isTransformed=*/true);

    // Create zTensor for c0. Reuse descriptors from h0 because h0 and c0 have
    // the same shape.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.c0());
    ZTensor c0ZTensor = zTensorHelper.getZTensor(
        /*preTransformedDescPtr=*/h0ZTensor.preTransformedDescPtr,
        /*transformedDescPtr=*/h0ZTensor.transformedDescPtr,
        /*bufferSize=*/h0ZTensor.bufferSize,
        /*alignedBuffer=*/stickI8Ptr,
        /*isTransformed=*/true);

    // Create zTensor for input_weights.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.input_weights());
    ZTensor inputWeightsZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_3DS, /*originalDims=*/{D, F, H},
            /*isTransformed=*/true, /*'isConcat=*/true,
            /*concatInfo=*/RNN_TYPE_LSTM | USAGE_WEIGHTS | prevLayer);

    // Create zTensor for input_bias.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.input_bias());
    ZTensor inputBiasZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_2DS, /*originalDims=*/{D, H},
            /*isTransformed=*/true, /*'isConcat=*/true,
            /*concatInfo=*/RNN_TYPE_LSTM | USAGE_BIASES | prevLayer);

    // Create zTensor for hidden_weights.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.hidden_weights());
    ZTensor hiddenWeightsZTensor = zTensorHelper.getZTensor(stickI8Ptr,
        /*dataType=*/zDNNDataType,
        /*layout=*/ZDNN_3DS, /*originalDims=*/{D, H, H},
        /*isTransformed=*/true, /*'isConcat=*/true,
        /*concatInfo=*/RNN_TYPE_LSTM | USAGE_HIDDEN_WEIGHTS | prevLayer);

    // Create zTensor for hidden_bias.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.hidden_bias());
    ZTensor hiddenBiasZTensor = zTensorHelper.getZTensor(stickI8Ptr,
        /*dataType=*/zDNNDataType,
        /*layout=*/ZDNN_2DS, /*originalDims=*/{D, H},
        /*isTransformed=*/true, /*'isConcat=*/true,
        /*concatInfo=*/RNN_TYPE_LSTM | USAGE_HIDDEN_BIASES | prevLayer);

    // Direction input.
    Value direction;
    StringRef directionStr = dyn_cast_or_null<ZLowLSTMOp>(op).direction();
    if (directionStr.equals_insensitive("forward")) {
      direction = create.llvm.constant(llvmI64Ty, (int64_t)FWD);
    } else if (directionStr.equals_insensitive("reverse")) {
      direction = create.llvm.constant(llvmI64Ty, (int64_t)BWD);
    } else if (directionStr.equals_insensitive("bidirectional")) {
      direction = create.llvm.constant(llvmI64Ty, (int64_t)BIDIR);
    } else
      llvm_unreachable("Unsupported direction");

    // work_area.
    Value workArea = zTensorHelper.getAlignedI8Ptr(operandAdaptor.work_area());

    // Create zTensor for hn_output.
    Value preTransformedDescPtr;

    if (dyn_cast_or_null<ZLowLSTMOp>(op).return_all_steps() == -1)
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
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.hn_output());
    ZTensor hnOutputZTensor = zTensorHelper.getZTensor(
        /*preTransformedDescPtr=*/preTransformedDescPtr,
        /*transformedDescPtr=*/transformedDescPtr,
        /*bufferSize=*/bufferSize,
        /*alignedBuffer=*/stickI8Ptr,
        /*isTransformed=*/true);

    // Create zTensor for cf_output. Reuse descriptors from hn_output if
    // hn_output is the last step output.
    ZTensor cfOutputZTensor;
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.cf_output());
    if (dyn_cast_or_null<ZLowLSTMOp>(op).return_all_steps() != -1)
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
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

    ZLowGRUOpAdaptor operandAdaptor(operands);
    Type llvmElementTy = operandAdaptor.input()
                             .getType()
                             .dyn_cast<LLVM::LLVMStructType>()
                             .getBody()[0]
                             .cast<LLVM::LLVMPointerType>()
                             .getElementType();

    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);

    // Some frequently used types and constants.
    Type llvmI64Ty = rewriter.getI64Type();
    Value oneI64 = create.llvm.constant(llvmI64Ty, (int64_t)1);

    // Get the dimensions of the original shape (the shape before stickifying)
    // used for creating zTensors.
    std::vector<Value> dims = getDimsFromShapeMemRefBySize(
        rewriter, loc, module, operandAdaptor.shape(), /*size=*/5);
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
    Value stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.input());
    ZTensor inputZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_3DS, /*originalDims=*/{T, B, F},
            /*isTransformed=*/true);

    // Create zTensor for h0.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.h0());
    ZTensor h0ZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_3DS, /*originalDims=*/{D, B, H},
            /*isTransformed=*/true);

    // Create zTensor for input_weights.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.input_weights());
    ZTensor inputWeightsZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_3DS, /*originalDims=*/{D, F, H},
            /*isTransformed=*/true, /*'isConcat=*/true,
            /*concatInfo=*/RNN_TYPE_GRU | USAGE_WEIGHTS | PREV_LAYER_NONE);

    // Create zTensor for input_bias.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.input_bias());
    ZTensor inputBiasZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_2DS, /*originalDims=*/{D, H},
            /*isTransformed=*/true, /*'isConcat=*/true,
            /*concatInfo=*/RNN_TYPE_GRU | USAGE_BIASES | PREV_LAYER_NONE);

    // Create zTensor for hidden_weights.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.hidden_weights());
    ZTensor hiddenWeightsZTensor = zTensorHelper.getZTensor(stickI8Ptr,
        /*dataType=*/zDNNDataType,
        /*layout=*/ZDNN_3DS, /*originalDims=*/{D, H, H},
        /*isTransformed=*/true, /*'isConcat=*/true,
        /*concatInfo=*/RNN_TYPE_GRU | USAGE_HIDDEN_WEIGHTS | PREV_LAYER_NONE);

    // Create zTensor for hidden_bias.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.hidden_bias());
    ZTensor hiddenBiasZTensor = zTensorHelper.getZTensor(stickI8Ptr,
        /*dataType=*/zDNNDataType,
        /*layout=*/ZDNN_2DS, /*originalDims=*/{D, H},
        /*isTransformed=*/true, /*'isConcat=*/true,
        /*concatInfo=*/RNN_TYPE_GRU | USAGE_HIDDEN_BIASES | PREV_LAYER_NONE);

    // Direction input.
    Value direction;
    StringRef directionStr = dyn_cast_or_null<ZLowGRUOp>(op).direction();
    if (directionStr.equals_insensitive("forward")) {
      direction = create.llvm.constant(llvmI64Ty, (int64_t)FWD);
    } else if (directionStr.equals_insensitive("reverse")) {
      direction = create.llvm.constant(llvmI64Ty, (int64_t)BWD);
    } else if (directionStr.equals_insensitive("bidirectional")) {
      direction = create.llvm.constant(llvmI64Ty, (int64_t)BIDIR);
    } else
      llvm_unreachable("Unsupported direction");

    // work_area.
    Value workArea = zTensorHelper.getAlignedI8Ptr(operandAdaptor.work_area());

    // Create zTensor for hn_output.
    Value preTransformedDescPtr;
    if (dyn_cast_or_null<ZLowGRUOp>(op).return_all_steps() == -1)
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
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.hn_output());
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

    ZLowUnstickOpAdaptor operandAdaptor(operands);
    Type llvmElementTy = operandAdaptor.Out()
                             .getType()
                             .dyn_cast<LLVM::LLVMStructType>()
                             .getBody()[0]
                             .cast<LLVM::LLVMPointerType>()
                             .getElementType();

    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);

    // Get the dimensions of the original shape (the shape before stickifying)
    // used for creating a zTensor. For 'zLow.unstick', the original shape is
    // obtained from the second argument.
    SmallVector<Value, 3> dims;
    getDimsFromMemRef(rewriter, loc, module, operandAdaptor.Out(), dims);

    // Get zDNN data type.
    zdnn_data_types zDNNDataType = llvmTypeToZDNNType(llvmElementTy);

    // Get zDNN data layout.
    zdnn_data_layouts zDNNDataLayout = convertLayoutAttrToZDNNDataLayout(
        dims.size(), dyn_cast_or_null<ZLowUnstickOp>(op).layoutAttr());

    // Create a zTensor.
    Value stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.X());
    ZTensor zTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/zDNNDataLayout, /*originalDims=*/dims,
            /*isTransformed=*/true);

    // Ready to unstickify.
    Value unstickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.Out());
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
    UnaryElementwiseOp unaryOp = dyn_cast_or_null<UnaryElementwiseOp>(op);
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

    Value input = operands[0];
    Value shape = operands[1];
    Value output = operands[2];
    Type llvmElementTy = input.getType()
                             .dyn_cast<LLVM::LLVMStructType>()
                             .getBody()[0]
                             .cast<LLVM::LLVMPointerType>()
                             .getElementType();

    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);

    // Get zDNN data type.
    zdnn_data_types zDNNDataType = llvmTypeToZDNNType(llvmElementTy);

    // Get zDNN data layout.
    zdnn_data_layouts zDNNDataLayout =
        convertLayoutAttrToZDNNDataLayout(0, unaryOp.layoutAttr());

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
      Value nullpointer = create.llvm.nullI8Ptr();
      callApi(rewriter, loc, module, apiRegistry, APIFor<UnaryElementwiseOp>(),
          {toOpaquePtr(rewriter, loc, module, inputZTensor.val), nullpointer,
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
    BinaryElementwiseOp binaryOp = dyn_cast_or_null<BinaryElementwiseOp>(op);

    Value input1 = operands[0];
    Value input2 = operands[1];
    Value shape = operands[2];
    Value output = operands[3];
    Type llvmElementTy = input1.getType()
                             .dyn_cast<LLVM::LLVMStructType>()
                             .getBody()[0]
                             .cast<LLVM::LLVMPointerType>()
                             .getElementType();

    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);

    // Get zDNN data type.
    zdnn_data_types zDNNDataType = llvmTypeToZDNNType(llvmElementTy);

    // Get zDNN data layout.
    zdnn_data_layouts zDNNDataLayout =
        convertLayoutAttrToZDNNDataLayout(0, binaryOp.layoutAttr());

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
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

    ZLowSoftmaxOpAdaptor operandAdaptor(operands);
    Type llvmElementTy = operandAdaptor.X()
                             .getType()
                             .dyn_cast<LLVM::LLVMStructType>()
                             .getBody()[0]
                             .cast<LLVM::LLVMPointerType>()
                             .getElementType();

    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);

    // Get zDNN data type.
    zdnn_data_types zDNNDataType = llvmTypeToZDNNType(llvmElementTy);

    // zDNN softmax uses 3DS layout.
    zdnn_data_layouts zDNNDataLayout = ZDNN_3DS;
    // Get the dimensions of the original shape (the shape before stickifying)
    // used for creating a zTensor.
    std::vector<Value> dims =
        getDimsFromShapeMemRef(rewriter, loc, module, operandAdaptor.shape(),
            /*layout=*/zDNNDataLayout);

    // Create the input zTensor.
    Value stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.X());
    ZTensor inputZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/zDNNDataLayout, /*originalDims=*/dims,
            /*isTransformed=*/true);

    // Create activation function type.
    nnpa_softmax_act actType;
    StringRef actFuncStr = llvm::dyn_cast<ZLowSoftmaxOp>(op).act_func();
    if (actFuncStr.equals_insensitive("act_none"))
      actType = NNPA_SOFTMAX_NONE;
    else if (actFuncStr.equals_insensitive("act_log"))
      actType = NNPA_SOFTMAX_LOG;
    else
      llvm_unreachable("Unsupported activation function");
    Value actFunc =
        create.llvm.constant(rewriter.getI64Type(), (int64_t)actType);

    // Create the output zTensor.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.Out());
    ZTensor outputZTensor = zTensorHelper.getZTensor(
        /*preTransformedDescPtr=*/inputZTensor.preTransformedDescPtr,
        /*transformedDescPtr=*/inputZTensor.transformedDescPtr,
        /*bufferSize=*/inputZTensor.bufferSize,
        /*alignedBuffer=*/stickI8Ptr,
        /*isTransformed=*/true);

    // work_area.
    Value workArea = zTensorHelper.getAlignedI8Ptr(operandAdaptor.work_area());

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
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

    ZLowMatMulOpAdaptor operandAdaptor(operands);
    Type llvmElementTy = operandAdaptor.X()
                             .getType()
                             .dyn_cast<LLVM::LLVMStructType>()
                             .getBody()[0]
                             .cast<LLVM::LLVMPointerType>()
                             .getElementType();

    bool stacked, broadcasting;
    if (dyn_cast_or_null<ZLowMatMulOp>(op).is_stacked() == -1)
      stacked = true;
    else
      stacked = false;
    if (dyn_cast_or_null<ZLowMatMulOp>(op).is_bcast() == -1)
      broadcasting = true;
    else
      broadcasting = false;

    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);

    // Some frequently used types and constants.
    Type llvmI64Ty = rewriter.getI64Type();

    // Get the dimensions of the original shape (the shape before stickifying)
    // used for creating zTensors.
    int dimCount = 3;
    if (stacked || broadcasting)
      dimCount = 4;
    std::vector<Value> dims = getDimsFromShapeMemRefBySize(
        rewriter, loc, module, operandAdaptor.shape(), /*size=*/dimCount);
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

    // Get zDNN data type.
    zdnn_data_types zDNNDataType = llvmTypeToZDNNType(llvmElementTy);

    // Create zTensors.
    ZTensor xZTensor, yZTensor, biasZTensor, outputZTensor;
    // X
    Value stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.X());
    if (stacked || broadcasting)
      xZTensor = zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
          /*layout=*/ZDNN_3DS, /*originalDims=*/{S, M, N},
          /*isTransformed=*/true);
    else
      xZTensor = zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
          /*layout=*/ZDNN_2D, /*originalDims=*/{M, N},
          /*isTransformed=*/true);
    // Y
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.Y());
    if (stacked)
      yZTensor = zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
          /*layout=*/ZDNN_3DS, /*originalDims=*/{S, N, P},
          /*isTransformed=*/true);
    else
      yZTensor = zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
          /*layout=*/ZDNN_2D, /*originalDims=*/{N, P},
          /*isTransformed=*/true);
    // Bias
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.Bias());
    if (stacked)
      biasZTensor =
          zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
              /*layout=*/ZDNN_2DS, /*originalDims=*/{S, P},
              /*isTransformed=*/true);
    else
      biasZTensor =
          zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
              /*layout=*/ZDNN_1D, /*originalDims=*/{P},
              /*isTransformed=*/true);
    // Op_type
    Value op_type;
    if (broadcasting)
      op_type = create.llvm.constant(
          llvmI64Ty, (int64_t)NNPA_MATMUL_BCAST_OP_ADDITION);
    else
      op_type =
          create.llvm.constant(llvmI64Ty, (int64_t)NNPA_MATMUL_OP_ADDITION);
    // Output
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.Out());
    if (stacked || broadcasting)
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
    if (broadcasting) {
      callApi(rewriter, loc, module, apiRegistry, API::ZDNN_MATMUL_BCAST_OP,
          {toOpaquePtr(rewriter, loc, module, xZTensor.val),
              toOpaquePtr(rewriter, loc, module, yZTensor.val),
              toOpaquePtr(rewriter, loc, module, biasZTensor.val), op_type,
              toOpaquePtr(rewriter, loc, module, outputZTensor.val)});
    } else {
      callApi(rewriter, loc, module, apiRegistry, API::ZDNN_MATMUL_OP,
          {toOpaquePtr(rewriter, loc, module, xZTensor.val),
              toOpaquePtr(rewriter, loc, module, yZTensor.val),
              toOpaquePtr(rewriter, loc, module, biasZTensor.val), op_type,
              toOpaquePtr(rewriter, loc, module, outputZTensor.val)});
    }

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
    ZLowConv2DOp convOp = dyn_cast_or_null<ZLowConv2DOp>(op);
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

    ZLowConv2DOpAdaptor operandAdaptor(operands);
    Type llvmElementTy = operandAdaptor.input()
                             .getType()
                             .dyn_cast<LLVM::LLVMStructType>()
                             .getBody()[0]
                             .cast<LLVM::LLVMPointerType>()
                             .getElementType();

    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);

    // Some frequently used types and constants.
    Type llvmI64Ty = rewriter.getI64Type();

    // Get the dimensions of the original shape (the shape before stickifying)
    // used for creating zTensors.
    std::vector<Value> dims = getDimsFromShapeMemRefBySize(
        rewriter, loc, module, operandAdaptor.shape(), /*size=*/7);
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
    ArrayRef<Attribute> kernelShapeArrayAttr = convOp.kernel_shape().getValue();
    // kernel height
    Value KH = create.llvm.constant(llvmI64Ty,
        (int64_t)kernelShapeArrayAttr[0].cast<IntegerAttr>().getInt());
    // kernel width
    Value KW = create.llvm.constant(llvmI64Ty,
        (int64_t)kernelShapeArrayAttr[1].cast<IntegerAttr>().getInt());

    // Get zDNN data type.
    zdnn_data_types zDNNDataType = llvmTypeToZDNNType(llvmElementTy);

    // Create a zTensor for input.
    Value stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.input());
    ZTensor inputZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_NHWC, /*originalDims=*/{N, HIn, WIn, CIn},
            /*isTransformed=*/true);

    // Create zTensor for input kernel.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.input_kernel());
    ZTensor kernelZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_HWCK, /*originalDims=*/{KH, KW, CIn, COut},
            /*isTransformed=*/true);

    // Create zTensor for bias.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.input_bias());
    ZTensor biasZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_1D, /*originalDims=*/{COut},
            /*isTransformed=*/true);

    // Padding type.
    Value paddingType;
    if (convOp.padding_type().equals_insensitive("SAME_PADDING"))
      paddingType = create.llvm.constant(
          llvmI64Ty, (int64_t)zdnn_pool_padding::SAME_PADDING);
    else if (convOp.padding_type().equals_insensitive("VALID_PADDING"))
      paddingType = create.llvm.constant(
          llvmI64Ty, (int64_t)zdnn_pool_padding::VALID_PADDING);
    else
      llvm_unreachable("Unsupported padding type");

    // Strides
    ArrayRef<Attribute> strideArrayAttr = convOp.strides().getValue();
    Value strideHeight = create.llvm.constant(
        llvmI64Ty, (int64_t)strideArrayAttr[0].cast<IntegerAttr>().getInt());
    Value strideWidth = create.llvm.constant(
        llvmI64Ty, (int64_t)strideArrayAttr[1].cast<IntegerAttr>().getInt());

    // Activation function.
    Value actFunc;
    if (convOp.act_func().equals_insensitive("ACT_NONE"))
      actFunc = create.llvm.constant(
          llvmI64Ty, (int64_t)zdnn_conv2d_act::CONV2D_ACT_NONE);
    else if (convOp.act_func().equals_insensitive("ACT_RELU"))
      actFunc = create.llvm.constant(
          llvmI64Ty, (int64_t)zdnn_conv2d_act::CONV2D_ACT_RELU);
    else
      llvm_unreachable("Unsupported activation function");

    // Create zTensor for output.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.output());
    ZTensor outputZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_NHWC, /*originalDims=*/{N, HOut, WOut, COut},
            /*isTransformed=*/true);

    // Prepare nullpointer for the clipping value
    Value nullpointer = create.llvm.nullI8Ptr();
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
    POOLOP poolOp = dyn_cast_or_null<POOLOP>(op);
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

    Value input = operands[0];
    Value shape = operands[1];
    Value output = operands[2];
    Type llvmElementTy = input.getType()
                             .dyn_cast<LLVM::LLVMStructType>()
                             .getBody()[0]
                             .cast<LLVM::LLVMPointerType>()
                             .getElementType();

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
    ArrayRef<Attribute> kernelShapeArrayAttr = poolOp.kernel_shape().getValue();
    // kernel height
    Value KH = create.llvm.constant(llvmI64Ty,
        (int64_t)kernelShapeArrayAttr[0].cast<IntegerAttr>().getInt());
    // kernel width
    Value KW = create.llvm.constant(llvmI64Ty,
        (int64_t)kernelShapeArrayAttr[1].cast<IntegerAttr>().getInt());

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
    if (poolOp.padding_type().equals_insensitive("SAME_PADDING"))
      paddingType = create.llvm.constant(
          llvmI64Ty, (int64_t)zdnn_pool_padding::SAME_PADDING);
    else if (poolOp.padding_type().equals_insensitive("VALID_PADDING"))
      paddingType = create.llvm.constant(
          llvmI64Ty, (int64_t)zdnn_pool_padding::VALID_PADDING);
    else
      llvm_unreachable("Unsupported padding type");

    // Strides
    ArrayRef<Attribute> strideArrayAttr = poolOp.strides().getValue();
    Value strideHeight = create.llvm.constant(
        llvmI64Ty, (int64_t)strideArrayAttr[0].cast<IntegerAttr>().getInt());
    Value strideWidth = create.llvm.constant(
        llvmI64Ty, (int64_t)strideArrayAttr[1].cast<IntegerAttr>().getInt());

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
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);

    ZLowMeanReduce2DOpAdaptor operandAdaptor(operands);
    Type llvmElementTy = operandAdaptor.input()
                             .getType()
                             .dyn_cast<LLVM::LLVMStructType>()
                             .getBody()[0]
                             .cast<LLVM::LLVMPointerType>()
                             .getElementType();

    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);

    // Some frequently used types and constants.
    Type llvmI64Ty = rewriter.getI64Type();
    Value oneI64 = create.llvm.constant(llvmI64Ty, (int64_t)1);

    // Get the dimensions of the original shape (the shape before stickifying)
    // used for creating zTensors.
    std::vector<Value> dims = getDimsFromShapeMemRefBySize(
        rewriter, loc, module, operandAdaptor.shape(), /*size=*/4);
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
    Value stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.input());
    ZTensor inputZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_NHWC, /*originalDims=*/{N, H, W, C},
            /*isTransformed=*/true);

    // Create zTensor for output.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.output());
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

    ZLowBatchNormOpAdaptor operandAdaptor(operands);
    Type llvmElementTy = operandAdaptor.input()
                             .getType()
                             .dyn_cast<LLVM::LLVMStructType>()
                             .getBody()[0]
                             .cast<LLVM::LLVMPointerType>()
                             .getElementType();

    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);

    // Get the dimensions of the original shape (the shape before stickifying)
    // used for creating zTensors.
    std::vector<Value> dims = getDimsFromShapeMemRefBySize(
        rewriter, loc, module, operandAdaptor.shape(), /*size=*/4);
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
    Value stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.input());
    ZTensor inputZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_NHWC, /*originalDims=*/{N, H, W, C},
            /*isTransformed=*/true);

    // Create a zTensor for A.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.A());
    ZTensor aZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_1D, /*originalDims=*/{C},
            /*isTransformed=*/true);

    // Create a zTensor for B.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.B());
    ZTensor bZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_1D, /*originalDims=*/{C},
            /*isTransformed=*/true);

    // Create zTensor for output.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.output());
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

void populateZLowToLLVMConversionPattern(mlir::RewritePatternSet &patterns,
    mlir::LLVMTypeConverter &typeConverter, mlir::MLIRContext *ctx) {
  ApiRegistry apiRegistry = RegisterAllApis(ctx);
  // clang-format off
  patterns.insert<
      ZLowStickLowering,
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
      ZLowConv2DLowering,
      ZLowMeanReduce2DLowering,
      ZLowBatchNormLowering
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
      // Activation operations
      ZLowUnaryElementwiseOpLowering<ZLowReluOp>,
      ZLowUnaryElementwiseOpLowering<ZLowTanhOp>,
      ZLowUnaryElementwiseOpLowering<ZLowSigmoidOp>,
      // Other operations
      ZLowPool2DLowering<ZLowAvgPool2DOp>,
      ZLowPool2DLowering<ZLowMaxPool2DOp>
    >(ctx, typeConverter, apiRegistry);
  // clang-format on
}

} // namespace zlow
} // namespace onnx_mlir
