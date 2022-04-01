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

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/IR/DataLayout.h"

#include "src/Accelerators/NNPA/Conversion/ZLowToLLVM/ZLowToLLVMCommon.hpp"
#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp"
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
#include "src/Accelerators/NNPA/Support/LayoutHelper.hpp"
#include "src/Conversion/KrnlToLLVM/ConvertKrnlToLLVM.hpp"
#include "src/Dialect/Krnl/KrnlTypes.hpp"
#include "third_party/zdnn-lib/zdnn/zdnn.h"

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

class ZLowStickLowering : public ConvertToLLVMPattern {
public:
  explicit ZLowStickLowering(MLIRContext *context, LLVMTypeConverter &lowering_,
      ApiRegistry apiRegistry)
      : ConvertToLLVMPattern(
            ZLowStickOp::getOperationName(), context, lowering_) {
    this->apiRegistry = apiRegistry;
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto loc = op->getLoc();

    ZLowStickOpAdaptor operandAdaptor(operands);
    auto llvmElementTy = operandAdaptor.X()
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
    auto module = op->getParentOfType<ModuleOp>();
    auto loc = op->getLoc();

    ZLowStickForLSTMOpAdaptor operandAdaptor(operands);
    auto llvmElementTy = operandAdaptor.f_gate()
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
    if (dims.size() == 2) {
      // for stickify input/hidden biases.
      zDNNDataLayout = ZDNN_2DS;
      zDNNConcatInfo = RNN_TYPE_LSTM | USAGE_BIASES | PREV_LAYER_NONE;
    } else if (dims.size() == 3) {
      // for stickify input/hidden weights.
      zDNNDataLayout = ZDNN_3DS;
      zDNNConcatInfo = RNN_TYPE_LSTM | USAGE_WEIGHTS | PREV_LAYER_NONE;
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
    auto fGatePtr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.f_gate());
    auto iGatePtr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.i_gate());
    auto cGatePtr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.c_gate());
    auto oGatePtr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.o_gate());
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
    auto module = op->getParentOfType<ModuleOp>();
    auto loc = op->getLoc();

    ZLowStickForGRUOpAdaptor operandAdaptor(operands);
    auto llvmElementTy = operandAdaptor.z_gate()
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
    if (dims.size() == 2) {
      // for stickify input/hidden biases.
      zDNNDataLayout = ZDNN_2DS;
      zDNNConcatInfo = RNN_TYPE_GRU | USAGE_BIASES | PREV_LAYER_NONE;
    } else if (dims.size() == 3) {
      // for stickify input/hidden weights.
      zDNNDataLayout = ZDNN_3DS;
      zDNNConcatInfo = RNN_TYPE_GRU | USAGE_WEIGHTS | PREV_LAYER_NONE;
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
    auto zGatePtr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.z_gate());
    auto rGatePtr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.r_gate());
    auto hGatePtr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.h_gate());
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
    auto *context = op->getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto loc = op->getLoc();

    ZLowLSTMOpAdaptor operandAdaptor(operands);
    auto llvmElementTy = operandAdaptor.input()
                             .getType()
                             .dyn_cast<LLVM::LLVMStructType>()
                             .getBody()[0]
                             .cast<LLVM::LLVMPointerType>()
                             .getElementType();

    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);

    // Some frequently used types and constants.
    auto llvmI64Ty = IntegerType::get(context, 64);
    auto oneI64 = rewriter.create<LLVM::ConstantOp>(
        loc, llvmI64Ty, rewriter.getI64IntegerAttr(1));

    // Get the dimensions of the original shape (the shape before stickifying)
    // used for creating zTensors.
    std::vector<Value> dims = getDimsFromShapeMemRefBySize(
        rewriter, loc, module, operandAdaptor.shape(), /*size=*/5);
    // direction
    auto D = dims[0];
    // timestep
    auto T = dims[1];
    // batch size
    auto B = dims[2];
    // feature size
    auto F = dims[3];
    // hidden size
    auto H = dims[4];

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
            /*concatInfo=*/RNN_TYPE_LSTM | USAGE_WEIGHTS | PREV_LAYER_NONE);

    // Create zTensor for input_bias.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.input_bias());
    ZTensor inputBiasZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_2DS, /*originalDims=*/{D, H},
            /*isTransformed=*/true, /*'isConcat=*/true,
            /*concatInfo=*/RNN_TYPE_LSTM | USAGE_BIASES | PREV_LAYER_NONE);

    // Create zTensor for hidden_weights.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.hidden_weights());
    ZTensor hiddenWeightsZTensor = zTensorHelper.getZTensor(stickI8Ptr,
        /*dataType=*/zDNNDataType,
        /*layout=*/ZDNN_3DS, /*originalDims=*/{D, H, H},
        /*isTransformed=*/true, /*'isConcat=*/true,
        /*concatInfo=*/RNN_TYPE_LSTM | USAGE_HIDDEN_WEIGHTS | PREV_LAYER_NONE);

    // Create zTensor for hidden_bias.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.hidden_bias());
    ZTensor hiddenBiasZTensor = zTensorHelper.getZTensor(stickI8Ptr,
        /*dataType=*/zDNNDataType,
        /*layout=*/ZDNN_2DS, /*originalDims=*/{D, H},
        /*isTransformed=*/true, /*'isConcat=*/true,
        /*concatInfo=*/RNN_TYPE_LSTM | USAGE_HIDDEN_BIASES | PREV_LAYER_NONE);

    // Direction input.
    Value direction;
    StringRef directionStr = dyn_cast_or_null<ZLowLSTMOp>(op).direction();
    if (directionStr.equals_insensitive("forward"))
      direction = rewriter.create<LLVM::ConstantOp>(
          loc, llvmI64Ty, rewriter.getI64IntegerAttr(FWD));
    else if (directionStr.equals_insensitive("reverse"))
      direction = rewriter.create<LLVM::ConstantOp>(
          loc, llvmI64Ty, rewriter.getI64IntegerAttr(BWD));
    else
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
        preTransformedDescPtr, /*isConcat=*/false, /*concatInfo=*/concatInfo);
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
    auto *context = op->getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto loc = op->getLoc();

    ZLowGRUOpAdaptor operandAdaptor(operands);
    auto llvmElementTy = operandAdaptor.input()
                             .getType()
                             .dyn_cast<LLVM::LLVMStructType>()
                             .getBody()[0]
                             .cast<LLVM::LLVMPointerType>()
                             .getElementType();

    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);

    // Some frequently used types and constants.
    auto llvmI64Ty = IntegerType::get(context, 64);
    auto oneI64 = rewriter.create<LLVM::ConstantOp>(
        loc, llvmI64Ty, rewriter.getI64IntegerAttr(1));

    // Get the dimensions of the original shape (the shape before stickifying)
    // used for creating zTensors.
    std::vector<Value> dims = getDimsFromShapeMemRefBySize(
        rewriter, loc, module, operandAdaptor.shape(), /*size=*/5);
    // direction
    auto D = dims[0];
    // timestep
    auto T = dims[1];
    // batch size
    auto B = dims[2];
    // feature size
    auto F = dims[3];
    // hidden size
    auto H = dims[4];

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
    if (directionStr.equals_insensitive("forward"))
      direction = rewriter.create<LLVM::ConstantOp>(
          loc, llvmI64Ty, rewriter.getI64IntegerAttr(FWD));
    else if (directionStr.equals_insensitive("reverse"))
      direction = rewriter.create<LLVM::ConstantOp>(
          loc, llvmI64Ty, rewriter.getI64IntegerAttr(BWD));
    else
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
        preTransformedDescPtr, /*isConcat=*/false, /*concatInfo=*/concatInfo);
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
    auto module = op->getParentOfType<ModuleOp>();
    auto loc = op->getLoc();

    ZLowUnstickOpAdaptor operandAdaptor(operands);
    auto llvmElementTy = operandAdaptor.Out()
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
    auto unstickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.Out());
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
    auto *context = op->getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto loc = op->getLoc();
    auto unaryOp = dyn_cast_or_null<UnaryElementwiseOp>(op);

    Value input = operands[0];
    Value shape = operands[1];
    Value output = operands[2];
    auto llvmElementTy = input.getType()
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
      auto nullpointer = rewriter.create<LLVM::NullOp>(
          loc, LLVM::LLVMPointerType::get(IntegerType::get(context, 8)));
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
    auto module = op->getParentOfType<ModuleOp>();
    auto loc = op->getLoc();
    auto binaryOp = dyn_cast_or_null<BinaryElementwiseOp>(op);

    Value input1 = operands[0];
    Value input2 = operands[1];
    Value shape = operands[2];
    Value output = operands[3];
    auto llvmElementTy = input1.getType()
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
    auto *context = op->getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto loc = op->getLoc();

    ZLowSoftmaxOpAdaptor operandAdaptor(operands);
    auto llvmElementTy = operandAdaptor.X()
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
    Value actFunc = rewriter.create<LLVM::ConstantOp>(loc,
        IntegerType::get(context, 64), rewriter.getI64IntegerAttr(actType));

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
    auto *context = op->getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto loc = op->getLoc();

    ZLowMatMulOpAdaptor operandAdaptor(operands);
    auto llvmElementTy = operandAdaptor.X()
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
    auto llvmI64Ty = IntegerType::get(context, 64);

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
      op_type = rewriter.create<LLVM::ConstantOp>(loc, llvmI64Ty,
          rewriter.getI64IntegerAttr(NNPA_MATMUL_BCAST_OP_ADDITION));
    else
      op_type = rewriter.create<LLVM::ConstantOp>(
          loc, llvmI64Ty, rewriter.getI64IntegerAttr(NNPA_MATMUL_OP_ADDITION));
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
    auto *context = op->getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto loc = op->getLoc();
    ZLowConv2DOp convOp = dyn_cast_or_null<ZLowConv2DOp>(op);

    ZLowConv2DOpAdaptor operandAdaptor(operands);
    auto llvmElementTy = operandAdaptor.input()
                             .getType()
                             .dyn_cast<LLVM::LLVMStructType>()
                             .getBody()[0]
                             .cast<LLVM::LLVMPointerType>()
                             .getElementType();

    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);

    // Some frequently used types and constants.
    auto llvmI64Ty = IntegerType::get(context, 64);

    // Get the dimensions of the original shape (the shape before stickifying)
    // used for creating zTensors.
    std::vector<Value> dims = getDimsFromShapeMemRefBySize(
        rewriter, loc, module, operandAdaptor.shape(), /*size=*/7);
    // batch size
    auto N = dims[0];
    // channel in
    auto CIn = dims[1];
    // height in
    auto HIn = dims[2];
    // width in
    auto WIn = dims[3];
    // channel out
    auto COut = dims[4];
    // height out
    auto HOut = dims[5];
    // width out
    auto WOut = dims[6];
    // kernel shape
    ArrayRef<Attribute> kernelShapeArrayAttr = convOp.kernel_shape().getValue();
    // kernel height
    Value KH = rewriter.create<LLVM::ConstantOp>(loc, llvmI64Ty,
        rewriter.getI64IntegerAttr(
            kernelShapeArrayAttr[0].cast<IntegerAttr>().getInt()));
    // kernel width
    Value KW = rewriter.create<LLVM::ConstantOp>(loc, llvmI64Ty,
        rewriter.getI64IntegerAttr(
            kernelShapeArrayAttr[1].cast<IntegerAttr>().getInt()));

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
      paddingType = rewriter.create<LLVM::ConstantOp>(loc, llvmI64Ty,
          rewriter.getI64IntegerAttr(zdnn_pool_padding::SAME_PADDING));
    else if (convOp.padding_type().equals_insensitive("VALID_PADDING"))
      paddingType = rewriter.create<LLVM::ConstantOp>(loc, llvmI64Ty,
          rewriter.getI64IntegerAttr(zdnn_pool_padding::VALID_PADDING));
    else
      llvm_unreachable("Unsupported padding type");

    // Strides
    ArrayRef<Attribute> strideArrayAttr = convOp.strides().getValue();
    Value strideHeight = rewriter.create<LLVM::ConstantOp>(loc, llvmI64Ty,
        rewriter.getI64IntegerAttr(
            strideArrayAttr[0].cast<IntegerAttr>().getInt()));
    Value strideWidth = rewriter.create<LLVM::ConstantOp>(loc, llvmI64Ty,
        rewriter.getI64IntegerAttr(
            strideArrayAttr[1].cast<IntegerAttr>().getInt()));

    // Activation function.
    Value actFunc;
    if (convOp.act_func().equals_insensitive("ACT_NONE"))
      actFunc = rewriter.create<LLVM::ConstantOp>(loc, llvmI64Ty,
          rewriter.getI64IntegerAttr(zdnn_conv2d_act::CONV2D_ACT_NONE));
    else if (convOp.act_func().equals_insensitive("ACT_RELU"))
      actFunc = rewriter.create<LLVM::ConstantOp>(loc, llvmI64Ty,
          rewriter.getI64IntegerAttr(zdnn_conv2d_act::CONV2D_ACT_RELU));
    else
      llvm_unreachable("Unsupported activation function");

    // Create zTensor for output.
    stickI8Ptr = zTensorHelper.getAlignedI8Ptr(operandAdaptor.output());
    ZTensor outputZTensor =
        zTensorHelper.getZTensor(stickI8Ptr, /*dataType=*/zDNNDataType,
            /*layout=*/ZDNN_NHWC, /*originalDims=*/{N, HOut, WOut, COut},
            /*isTransformed=*/true);

    // Prepare nullpointer for the clipping value
    auto nullpointer = rewriter.create<LLVM::NullOp>(
        loc, LLVM::LLVMPointerType::get(IntegerType::get(context, 8)));
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
    auto *context = op->getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto loc = op->getLoc();
    auto poolOp = dyn_cast_or_null<POOLOP>(op);

    Value input = operands[0];
    Value shape = operands[1];
    Value output = operands[2];
    auto llvmElementTy = input.getType()
                             .dyn_cast<LLVM::LLVMStructType>()
                             .getBody()[0]
                             .cast<LLVM::LLVMPointerType>()
                             .getElementType();

    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);

    // Some frequently used types and constants.
    auto llvmI64Ty = IntegerType::get(context, 64);

    // Get the dimensions of the original shape (the shape before stickifying)
    // used for creating zTensors.
    std::vector<Value> dims =
        getDimsFromShapeMemRefBySize(rewriter, loc, module, shape, /*size=*/6);
    // batch size
    auto N = dims[0];
    // channel in
    auto CIn = dims[1];
    // height in
    auto HIn = dims[2];
    // width in
    auto WIn = dims[3];
    // height out
    auto HOut = dims[4];
    // width out
    auto WOut = dims[5];
    // kernel shape
    ArrayRef<Attribute> kernelShapeArrayAttr = poolOp.kernel_shape().getValue();
    // kernel height
    Value KH = rewriter.create<LLVM::ConstantOp>(loc, llvmI64Ty,
        rewriter.getI64IntegerAttr(
            kernelShapeArrayAttr[0].cast<IntegerAttr>().getInt()));
    // kernel width
    Value KW = rewriter.create<LLVM::ConstantOp>(loc, llvmI64Ty,
        rewriter.getI64IntegerAttr(
            kernelShapeArrayAttr[1].cast<IntegerAttr>().getInt()));

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
      paddingType = rewriter.create<LLVM::ConstantOp>(loc, llvmI64Ty,
          rewriter.getI64IntegerAttr(zdnn_pool_padding::SAME_PADDING));
    else if (poolOp.padding_type().equals_insensitive("VALID_PADDING"))
      paddingType = rewriter.create<LLVM::ConstantOp>(loc, llvmI64Ty,
          rewriter.getI64IntegerAttr(zdnn_pool_padding::VALID_PADDING));
    else
      llvm_unreachable("Unsupported padding type");

    // Strides
    ArrayRef<Attribute> strideArrayAttr = poolOp.strides().getValue();
    Value strideHeight = rewriter.create<LLVM::ConstantOp>(loc, llvmI64Ty,
        rewriter.getI64IntegerAttr(
            strideArrayAttr[0].cast<IntegerAttr>().getInt()));
    Value strideWidth = rewriter.create<LLVM::ConstantOp>(loc, llvmI64Ty,
        rewriter.getI64IntegerAttr(
            strideArrayAttr[1].cast<IntegerAttr>().getInt()));

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
    auto *context = op->getContext();
    auto module = op->getParentOfType<ModuleOp>();
    auto loc = op->getLoc();

    ZLowMeanReduce2DOpAdaptor operandAdaptor(operands);
    auto llvmElementTy = operandAdaptor.input()
                             .getType()
                             .dyn_cast<LLVM::LLVMStructType>()
                             .getBody()[0]
                             .cast<LLVM::LLVMPointerType>()
                             .getElementType();

    ZTensorHelper zTensorHelper =
        ZTensorHelper(rewriter, loc, module, apiRegistry);

    // Some frequently used types and constants.
    auto llvmI64Ty = IntegerType::get(context, 64);
    auto oneI64 = rewriter.create<LLVM::ConstantOp>(
        loc, llvmI64Ty, rewriter.getI64IntegerAttr(1));

    // Get the dimensions of the original shape (the shape before stickifying)
    // used for creating zTensors.
    std::vector<Value> dims = getDimsFromShapeMemRefBySize(
        rewriter, loc, module, operandAdaptor.shape(), /*size=*/4);
    // batch size
    auto N = dims[0];
    // height in
    auto H = dims[1];
    // width in
    auto W = dims[2];
    // channel in
    auto C = dims[3];
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
    auto module = op->getParentOfType<ModuleOp>();
    auto loc = op->getLoc();

    ZLowBatchNormOpAdaptor operandAdaptor(operands);
    auto llvmElementTy = operandAdaptor.input()
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
    auto N = dims[0];
    // height in
    auto H = dims[1];
    // width in
    auto W = dims[2];
    // channel in
    auto C = dims[3];

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

//===----------------------------------------------------------------------===//
// ZLow dialect lowering to LLVM.
//===----------------------------------------------------------------------===//

namespace {
struct ZLowToLLVMLoweringPass
    : public PassWrapper<ZLowToLLVMLoweringPass, OperationPass<ModuleOp>> {

  StringRef getArgument() const override { return "convert-all-to-llvm"; }

  StringRef getDescription() const override {
    return "Lower all ops including zlow ops to LLVM.";
  }

  void runOnOperation() final;
};
} // end anonymous namespace

void ZLowToLLVMLoweringPass::runOnOperation() {
  ModuleOp module = getOperation();
  const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
  LowerToLLVMOptions options(
      &getContext(), dataLayoutAnalysis.getAtOrAbove(module));
  options.emitCWrappers = true;

  // Define the target for this lowering i.e. the LLVM dialect.
  ConversionTarget target(getContext());
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalOp<ModuleOp>();
  target.addLegalOp<UnrealizedConversionCastOp>();

  // Convert types to legal types for the LLVM dialect.
  LLVMTypeConverter typeConverter(&getContext(), options);

  // Determine, for each output, whether it is a constant or not.
  SmallVector<bool, 4> outputOMTensorOwnerships;
  onnx_mlir::krnl::determineOwnershipForOutputOMTensors(
      module, outputOMTensorOwnerships);

  // Record entry point names and their input/output signatures.
  // This info is used to generate global signature functions.
  SmallVector<std::string, 1> entryPointNames, inSignatures, outSignatures;
  onnx_mlir::krnl::recordEntryPointSignatures(
      module, entryPointNames, inSignatures, outSignatures);

  typeConverter.addConversion([&](MemRefType type) -> llvm::Optional<Type> {
    Type elementType = type.getElementType();
    if (!elementType.isa<krnl::StringType>())
      return llvm::None;

    elementType =
        elementType.cast<krnl::StringType>().getLLVMType(type.getContext());
    return typeConverter.convertType(
        MemRefType::get(type.getShape(), elementType));
  });

  typeConverter.addConversion([&](krnl::StringType type) -> Type {
    return typeConverter.convertType(type.getLLVMType(type.getContext()));
  });

  // We have a combination of `zlow`, `krnl`, `affine`, and `std` operations. We
  // lower in stages until all the code is in the LLVM dialect.
  RewritePatternSet patterns(&getContext());
  onnx_mlir::krnl::populateAffineAndKrnlToLLVMConversion(patterns,
      typeConverter, &getContext(), outputOMTensorOwnerships,
      /*singleEntryPoint=*/entryPointNames.size() == 1);

  // ZLow to LLVM.
  auto apiRegistry = RegisterAllApis(&getContext());
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
    >(&getContext(), typeConverter, apiRegistry);
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
    >(&getContext(), typeConverter, apiRegistry);
  // clang-format on

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  if (failed(
          applyFullConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
  }

  // Generate signature functions.
  if (entryPointNames.size() >= 1)
    onnx_mlir::krnl::genSignatureFunction(
        module, entryPointNames, inSignatures, outSignatures);
}

/// Create the pass for lowering `zlow`, `krnl`, `affine` and `std` dialects
/// to LLVM.
std::unique_ptr<mlir::Pass> createZLowToLLVMPass() {
  return std::make_unique<ZLowToLLVMLoweringPass>();
}

} // namespace zlow
} // namespace onnx_mlir
