/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- ReduceMean.cpp - ONNX Op Transform ------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a combined pass that dynamically invoke several
// transformation on ONNX ops.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/NN/CommonUtils.h"
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

 /*
  * ONNX ReduceMean operation
  *
  * “Computes the mean of the input tensor’s element along the provided axes. 
  * The resulted” “tensor has the same rank as the input if keepdims equal 1. 
  * If keepdims equal 0, then” “the resulted tensor have the reduced 
  * dimension pruned.” “” “The above behavior is similar to numpy, with the 
  * exception that numpy default keepdims to” “False instead of True.”
  *
  * Attributes:
  *   axes	::mlir::ArrayAttr	64-bit integer array attribute
  *  keepdims	::mlir::IntegerAttr	64-bit signed integer attribute
  *
  *  Operands:
  *  data	tensor of 32-bit/64-bit unsigned integer values or 
  *  		tensor of 32-bit/64-bit signless integer values or 
  *  		tensor of 16-bit/32-bit/64-bit float values or 
  *  		tensor of bfloat16 type values or memref of any type values.
  *
  *  Results:
  *  reduced	tensor of 32-bit/64-bit unsigned integer values or 
  *  		tensor of 32-bit/64-bit signless integer values or 
  *  		tensor of 16-bit/32-bit/64-bit float values or tensor 
  *  		of bfloat16 type values or memref of any type values
  */

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

struct ONNXReduceMeanOpToTorchLowering : public ConversionPattern {
public:
  ONNXReduceMeanOpToTorchLowering(
      TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter,
            mlir::ONNXReduceMeanOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    ONNXReduceMeanOp op1 = llvm::dyn_cast<ONNXReduceMeanOp>(op);
    ONNXReduceMeanOpAdaptor adapter(op1);
    mlir::MLIRContext *context = op1.getContext();
    Location loc = op1.getLoc();

    auto axes = op1.axesAttr();         // ::mlir::ArrayAttr
    auto keepDims = op1.keepdimsAttr(); // ::mlir::IntegerAttr

    Value data = op1.data(); // ONNX operands

    auto ty = IntegerType::get(op1.getContext(), 64);

    // Reading the ONNX side pads values and store in the array.
    std::vector<Value> axesList = createArrayAttribute(axes, ty, loc, rewriter);

    auto zero = 0;
    auto f00 = IntegerAttr::get(ty, zero);
    auto f0 = f00;
    Value f0v = rewriter.create<ConstantIntOp>(loc, f0);

    Value keepdimVal;
    if (keepDims)
      keepdimVal = rewriter.create<ConstantIntOp>(loc, keepDims);
    else
      keepdimVal = f0v;

    Value axesShapeList = rewriter.create<PrimListConstructOp>(loc,
        Torch::ListType::get(rewriter.getType<Torch::IntType>()),
        ValueRange{axesList});

    TensorType data_tensor_type = data.getType().cast<TensorType>();
    TensorType op_tensor_type = op->getResult(0).getType().cast<TensorType>();

    auto dataTy = Torch::ValueTensorType::get(context,
        data_tensor_type.getShape(), data_tensor_type.getElementType());
    auto dtt = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
        loc, dataTy, data);
    auto resultTy = Torch::ValueTensorType::get(op1.getContext(),
        op_tensor_type.getShape(), op_tensor_type.getElementType());


    Value atenmultensorOp =
        rewriter.create<AtenMeanOp>(loc, resultTy, dtt, keepdimVal);

    Value result = atenmultensorOp;

    rewriter.replaceOpWithNewOp<torch::TorchConversion::ToBuiltinTensorOp>(
        op, op->getResult(0).getType(), result);

    return success();
  }
};

void populateLoweringONNXToTorchReduceMeanOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXReduceMeanOpToTorchLowering>(typeConverter, ctx);
}
