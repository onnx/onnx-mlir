/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- MaxPoolSingleOutOpTransformToTorchPass.cpp - ONNX Op Transform
//------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a combined pass that dynamically invoke several
// transformation on ONNX ops.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

#include <fstream>
#include <iostream>
#include <set>

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/ToolOutputFile.h"

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Interface/ShapeInferenceOpInterface.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/OMOptions.hpp"

#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/StringExtras.h"

#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"

#ifdef _WIN32
#include <io.h>
#endif

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

/**
 * ONNX MaxPool operation
 *
 * MaxPool consumes an input tensor X and applies max pooling across the tensor
 *according to kernel sizes, stride sizes, and pad lengths. max pooling
 *consisting of computing the max on all values of a subset of the input tensor
 *according to the kernel size and downsampling the data into the output tensor
 *Y for further processing.
 *
 * Where is this used?
 * max pooling is applied after convolution op.
 *
 * Operands :
 * X		tensor of 16-bit/32-bit/64-bit float values or memref of any
 *type values Input data tensor from the previous operator; dimensions for image
 *case are (N x C x H x W), where N is the batch size, C is the number of
 *channels, and H and W are the height and the width of the data. For non image
 *case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is
 *		the batch size. Optionally, if dimension denotation is in effect,
 *the operation expects the input data tensor to arrive with the dimension
 *denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].
 * Output   :
 *
 * Y		tensor of 16-bit/32-bit/64-bit float values or memref of any
 *type values or none type Output data tensor from average or max pooling across
 *the input tensor. Dimensions will vary based on various kernel, stride, and
 *pad sizes. Floor value of the dimension is used differentiable
 *
 * Attributes
 * auto_pad 		string attribute DEPRECATED
 * ceiling_mode 	int (default is 0), Whether to use ceil or floor (default)
 *to compute the output shape.
 *
 * dilations 		list of ints, 64-bit integer array attribute
 * 			Dilation value along each spatial axis of filter. If not
 *present, the dilation defaults to 1 along each spatial axis.
 *
 * kernel_shape 	list of ints (required) : 64-bit integer array attribute
 *              	The size of the kernel along each axis.
 *
 * pads 		list of ints, 64-bit integer array attribute
 * storage_order        int (default is 0)
 *			The storage order of the tensor. 0 is row major, and 1 is
 *column major. strides 		list of ints 64-bit integer array
 *attribute Stride along each spatial axis
 *
 * AtenMaxPool2dOp Arguments as below
 * -------------------------------
 *
 *  AnyTorchTensorType:$self,
 *  TorchIntListType:$kernel_size,
 *  TorchIntListType:$stride,
 *  TorchIntListType:$padding,
 *  TorchIntListType:$dilation,
 *  Torch_BoolType:$ceil_mode
 *
 * Validation
 * ----------
 * ./Debug/bin/onnx-mlir --EmitONNXIR --debug
 *../../../third-party/onnx-mlir/third_party/onnx/onnx/backend/test/data/node/test_maxpool_2d_pads/model.onnx
 *
 * Limitations
 * -----------
 * The atribute values have been used in the below code are to be corrected.
 *
 */

typedef struct dim_pads {
  int dim_start;
  int dim_end;
} dim_pads;

struct ONNXMaxPoolSingleOutOpToTorchLowering : public ConversionPattern {
public:
  ONNXMaxPoolSingleOutOpToTorchLowering(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter,
            mlir::ONNXMaxPoolSingleOutOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    ONNXMaxPoolSingleOutOp op1 = llvm::dyn_cast<ONNXMaxPoolSingleOutOp>(op);
    ONNXMaxPoolSingleOutOpAdaptor adapter(op1);
    mlir::MLIRContext *context = op1.getContext();
    Location loc = op1.getLoc();

    Value x = op1.X(); // ONNX operands

    auto autopad = op1.auto_padAttr();            // ::mlir::StringAttr
    auto dilations = op1.dilationsAttr();         // ::mlir::ArrayAttr
    auto kernal_shape = op1.kernel_shapeAttr();   // ::mlir::ArrayAttr
    auto pads = op1.padsAttr();                   // ::mlir::ArrayAttr
    auto strides = op1.stridesAttr();             // ::mlir::ArrayAttr
    int64_t ceiling_mode = op1.ceil_mode();       // int64_t
    auto ceiling_mode_attr = op1.ceil_modeAttr(); // ::mlir::IntegerAttr

    auto storage_order_attr = op1.storage_orderAttr(); // ::mlir::IntegerAttr
    int64_t storage_order = op1.storage_order();       // int64_t

    // Reading the ONNX side pads values and store in the array.
    dim_pads dimArray[pads.size()];
    std::vector<Value> translatepadsList;
    auto ty = IntegerType::get(op1.getContext(), 64);
    if (pads) {
      int j = 0;
      for (unsigned int i = 0; i < pads.size(); i++) {
        dimArray[j].dim_start =
            (pads[i].dyn_cast<IntegerAttr>()).getValue().getZExtValue();
        i++;
        dimArray[j].dim_end =
            (pads[i].dyn_cast<IntegerAttr>()).getValue().getZExtValue();
        j++;
      }

      // read the onnx pad values from array(dim_start values)
      int k = 0;
      for (unsigned int i = 0; i < pads.size(); i = i + 2) {
        auto f0 = IntegerAttr::get(ty, (dimArray[k].dim_start));
        Value p0v = rewriter.create<ConstantIntOp>(loc, f0);
        translatepadsList.push_back(p0v);
        k++;
      }

      // read the onnx pad values from array(dim_end values)
      k = 0;
      for (unsigned int i = 0; i < pads.size(); i = i + 2) {
        auto f1 = IntegerAttr::get(ty, (dimArray[k].dim_end));
        Value p1v = rewriter.create<ConstantIntOp>(loc, f1);
        translatepadsList.push_back(p1v);
        k++;
      }
    }

    std::vector<Value> dilationonnxList;
    std::vector<Value> kernalshapeonnxList;
    std::vector<Value> stridesonnxList;

    // reading the dilation values.
    if (dilations) {
      for (unsigned int i = 0; i < dilations.size(); i++) {
        auto f1 = IntegerAttr::get(ty,
            (dilations[i].dyn_cast<IntegerAttr>()).getValue().getZExtValue());
        Value p1v = rewriter.create<ConstantIntOp>(loc, f1);
        dilationonnxList.push_back(p1v);
      }
    }

    // reading the kernal_shape values.
    if (kernal_shape) {
      for (unsigned int i = 0; i < kernal_shape.size(); i++) {
        auto f1 = IntegerAttr::get(ty, (kernal_shape[i].dyn_cast<IntegerAttr>())
                                           .getValue()
                                           .getZExtValue());
        Value p1v = rewriter.create<ConstantIntOp>(loc, f1);
        kernalshapeonnxList.push_back(p1v);
      }
    }

    // reading the strides values.
    if (strides) {
      for (unsigned int i = 0; i < strides.size(); i++) {
        auto f1 = IntegerAttr::get(
            ty, (strides[i].dyn_cast<IntegerAttr>()).getValue().getZExtValue());
        Value p1v = rewriter.create<ConstantIntOp>(loc, f1);
        stridesonnxList.push_back(p1v);
      }
    }

    auto one = 1;
    auto two = 2;
    auto three = 3;
    auto zero = 0;

    auto f33 = IntegerAttr::get(ty, three);
    auto f00 = IntegerAttr::get(ty, zero);
    auto f22 = IntegerAttr::get(ty, two);

    auto f3 = f33;
    auto f0 = f00;
    auto f2 = f22;

    Value f3v = rewriter.create<ConstantIntOp>(loc, f3);
    Value f0v = rewriter.create<ConstantIntOp>(loc, f0);
    Value f22v = rewriter.create<ConstantIntOp>(loc, f2);

    Value ceiling_mode_val;
    if (ceiling_mode_attr)
      ceiling_mode_val = rewriter.create<ConstantIntOp>(loc, ceiling_mode_attr);
    else
      ceiling_mode_val = f0v;

    Value stridesList = rewriter.create<PrimListConstructOp>(loc,
        Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        ValueRange{stridesonnxList});

    Value dilationList = rewriter.create<PrimListConstructOp>(loc,
        Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        ValueRange{dilationonnxList});

    Value padsList = rewriter.create<PrimListConstructOp>(loc,
        Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        ValueRange{translatepadsList});

    Value kernalShapeList = rewriter.create<PrimListConstructOp>(loc,
        Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        ValueRange{kernalshapeonnxList});

    TensorType x_tensor_type = x.getType().cast<TensorType>();
    TensorType op_tensor_type = op->getResult(0).getType().cast<TensorType>();

    auto xTy = Torch::ValueTensorType::get(
        context, x_tensor_type.getShape(), x_tensor_type.getElementType());
    auto resultTy = Torch::ValueTensorType::get(op1.getContext(),
        op_tensor_type.getShape(), op_tensor_type.getElementType());
    auto xtt = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
        loc, xTy, x);

    llvm::outs() << "\n resultTy:   "
                 << "\n"
                 << resultTy << "\n"
                 << "\n";
    llvm::outs() << "xtt torch tensor from MLIR tensor "
                 << "\n"
                 << xtt << "\n"
                 << "\n";
    llvm::outs() << "kernalShapeList:   "
                 << "\n"
                 << kernalShapeList << "\n"
                 << "\n";
    llvm::outs() << "stridesList:   "
                 << "\n"
                 << stridesList << "\n"
                 << "\n";
    llvm::outs() << "padsList "
                 << "\n"
                 << padsList << "\n"
                 << "\n";
    llvm::outs() << "dilationList:   "
                 << "\n"
                 << dilationList << "\n"
                 << "\n";
    llvm::outs() << "ceiling_mode_val:    "
                 << "\n"
                 << ceiling_mode_val << "\n"
                 << "\n";

    Value atenmaxpool2d = rewriter.create<AtenMaxPool2dOp>(loc, resultTy, xtt,
        kernalShapeList, stridesList, padsList, dilationList, ceiling_mode_val);

    llvm::outs() << "AtenMaxPool2dOp operation creation"
                 << "\n"
                 << atenmaxpool2d << "\n"
                 << "\n";

    Value result = atenmaxpool2d;

    rewriter.replaceOpWithNewOp<torch::TorchConversion::ToBuiltinTensorOp>(
        op, op->getResult(0).getType(), result);

    return success();
  }
};

void populateLoweringONNXToTorchMaxPoolSingleOutOpPattern(
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXMaxPoolSingleOutOpToTorchLowering>(typeConverter, ctx);
}
