/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- FlattenOp.cpp - ONNX Op Transform ------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// ===================================================================
//
// This file implements a combined pass that dynamically invoke several
// transformation on ONNX ops.
//
//===-------------------------------------------------------------===//

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
 * Flattens input by reshaping it into a one-dimensional tensor.
 * If start_dim or end_dim are passed, only dimensions starting with
 * start_dim and ending with end_dim are flattened.
 * The order of elements in input is unchanged.
 *
 * Attributes
 *    axis    	::mlir::IntegerAttr 	i64-bit signed integer attribute
 *
 * Operands:
 *    input     tensor of 8-bit/16-bit/32-bit unsigned integer values or
 *    		tensor of 64-bit unsigned integer values or
 *    		tensor of 8-bit/16-bit/32-bit/64-bit signless integer values
 *    		tensor of bfloat16 type or tensor of 16-bit float values
 *    		tensor of 32-bit float or tensor of 64-bit float values
 *    		tensor of string type values or tensor of 1-bit signless 
 *    		integer values or tensor of complex type with 32-bit float
 *    		elements values or tensor of complex type with 64-bit float
 *    		elements values or memref of any type values
 *
 * Results:
 *    output     tensor of 8-bit/16-bit/32-bit unsigned integer values or
 *              tensor of 64-bit unsigned integer values or
 *              tensor of 8-bit/16-bit/32-bit/64-bit signless integer values
 *              tensor of bfloat16 type or tensor of 16-bit float values
 *              tensor of 32-bit float or tensor of 64-bit float values
 *              tensor of string type values or tensor of 1-bit signless
 *              integer values or tensor of complex type with 32-bit float
 *              elements values or tensor of complex type with 64-bit float
 *              elements values or memref of any type values
 *
 */

class ONNXFlattenOpToTorchLowering : public ConversionPattern {
public:
  ONNXFlattenOpToTorchLowering(TypeConverter &typeConverter,
		  MLIRContext *ctx)
      : ConversionPattern( typeConverter,
		      ::mlir::ONNXFlattenOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    Location loc = op->getLoc();
    mlir::MLIRContext *context = op->getContext();
    ONNXFlattenOp op1 = llvm::dyn_cast<ONNXFlattenOp>(op);
    ONNXFlattenOpAdaptor adaptor(operands);

    Value input = op1.input();
    auto axisValue = op1.axis();       // ::mlir::IntegerAttr

    auto inputShape = input.getType().cast<ShapedType>().getShape();
    int64_t inputRank = inputShape.size();

    TensorType op_tensor_type =
	    op->getResult(0).getType().cast<TensorType>();
    auto resultTy = Torch::ValueTensorType::get(op1.getContext(),
               op_tensor_type.getShape(), op_tensor_type.getElementType());

    TensorType input_tensor_type  = input.getType().cast<TensorType>();
    auto inputTy =
	 Torch::ValueTensorType::get(context, input_tensor_type.getShape(),
                    input_tensor_type.getElementType());
    auto itt  =
	  rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(loc,
                    inputTy, input);
    auto ty = IntegerType::get(op1.getContext(), 64);
    int64_t startDim;
    int64_t endDim;
    Value atenflaten1 = itt;
    if (axisValue != 0) {
      // flatten the region upto axis-1.
      startDim = 0;
      endDim = axisValue - 1;
      auto f0 = IntegerAttr::get(ty, (startDim));
      Value p0v = rewriter.create<ConstantIntOp>(loc, f0);
      auto f1 = IntegerAttr::get(ty, (endDim));
      Value p1v = rewriter.create<ConstantIntOp>(loc, f1);
      llvm::outs() << "input Value:   "
	      << "\n" << itt << "\n" << "\n";
      llvm::outs() << "startDim1 Value:   "
	      << "\n" << p0v << "\n" << "\n";
      llvm::outs() << "endDim1 Value:   "
	      << "\n" << p1v << "\n" << "\n";
      atenflaten1 = rewriter.create<AtenFlattenUsingIntsOp>(loc,
                    resultTy, itt, p0v, p1v);
      llvm::outs() << "Aten Flatten1 Op:   "
	      << "\n" << atenflaten1 << "\n" << "\n";
    }

    // flatten the region from axis upwards.
    startDim = axisValue;
    endDim = inputRank;
    auto f2 = IntegerAttr::get(ty, (startDim));
    Value p2v = rewriter.create<ConstantIntOp>(loc, f2);
    auto f3 = IntegerAttr::get(ty, (endDim));
    Value p3v = rewriter.create<ConstantIntOp>(loc, f3);
    llvm::outs() << "startDim2 Value:   " << "\n" << p2v << "\n" << "\n";
    llvm::outs() << "endDim2 Value:   " << "\n" << p3v << "\n" << "\n";
    Value atenflaten2 = rewriter.create<AtenFlattenUsingIntsOp>(loc,
                    resultTy, atenflaten1, p2v, p3v);
    Value result = atenflaten2;
    
    llvm::outs() << "AtenFlatten Op created" << "\n" << "\n";
    llvm::outs() << "Aten Flatten Op:   "
	    << "\n" << atenflaten2 << "\n" << "\n";
    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op,
                    resultTy, result);
    return success();
  }
};

void populateLoweringONNXToTorchFlattenOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXFlattenOpToTorchLowering>(typeConverter, ctx);
}
