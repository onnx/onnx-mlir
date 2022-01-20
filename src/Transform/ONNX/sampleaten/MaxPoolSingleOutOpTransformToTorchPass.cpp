/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- MaxPoolSingleOutOpTransformToTorchPass.cpp - ONNX Op Transform ------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a combined pass that dynamically invoke several
// transformation on ONNX ops.
//
//===----------------------------------------------------------------------===//

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


#include "../../../../../llvm-project/mlir/include/mlir/Transforms/DialectConversion.h"
#include "../../../../../torch-mlir/include/torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "../../../../../torch-mlir/include/torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "../../../../../torch-mlir/include/torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "../../../../../torch-mlir/include/torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "../../../../../llvm-project/llvm/include/llvm/ADT/StringExtras.h"

#include "../../../../../torch-mlir/include/torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "../../../../../torch-mlir/include/torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"
#include "../../../../../torch-mlir/include/torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

#ifdef _WIN32
#include <io.h>
#endif

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

/**
 * ONNX MaxPool operation
 *
 * MaxPool consumes an input tensor X and applies max pooling across the tensor according to 
 * kernel sizes, stride sizes, and pad lengths. max pooling consisting of computing the max 
 * on all values of a subset of the input tensor according to the kernel size and downsampling 
 * the data into the output tensor Y for further processing. 
 * 
 * Where is this used? 
 * max pooling is applied after convolution op.
 *
 * Operands :
 * X		tensor of 16-bit/32-bit/64-bit float values or memref of any type values
 *		Input data tensor from the previous operator; dimensions for image case are 
 * 		(N x C x H x W), where N is the batch size, C is the number of channels, 
 *		and H and W are the height and the width of the data. For non image case, 
 *		the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is 
 *		the batch size. Optionally, if dimension denotation is in effect, the operation 
 *		expects the input data tensor to arrive with the dimension denotation of 
 *		[DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...].
 * Output   : 
 *     
 * Y		tensor of 16-bit/32-bit/64-bit float values or memref of any type values or none type
 *		Output data tensor from average or max pooling across the input tensor. Dimensions will 
 *		vary based on various kernel, stride, and pad sizes. Floor value of the dimension is used
 *              differentiable 
 *
 * Attributes 
 * auto_pad 		string attribute DEPRECATED
 * ceiling_mode 	int (default is 0), Whether to use ceil or floor (default) to compute the output shape.
 * 
 * dilations 		list of ints, 64-bit integer array attribute
 * 			Dilation value along each spatial axis of filter. If not present, the dilation defaults 
 * 			to 1 along each spatial axis.
 * 
 * kernel_shape 	list of ints (required) : 64-bit integer array attribute
 *              	The size of the kernel along each axis.
 * 
 * pads 		list of ints, 64-bit integer array attribute
 * storage_order        int (default is 0)	
 *			The storage order of the tensor. 0 is row major, and 1 is column major.
 * strides 		list of ints 64-bit integer array attribute
 * 			Stride along each spatial axis
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
 * ./Debug/bin/onnx-mlir --EmitONNXIR --debug ../../../third-party/onnx-mlir/third_party/onnx/onnx/backend/test/data/node/test_maxpool_2d_pads/model.onnx 
 * 
 * Limitations
 * -----------
 * The atribute values have been used in the below code are to be corrected. 
 * 
 */
namespace {

class DecomposeONNXToAtenMaxPool2DOp: public OpRewritePattern<ONNXMaxPoolSingleOutOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ONNXMaxPoolSingleOutOp op,
                                PatternRewriter &rewriter) const override {

    ONNXMaxPoolSingleOutOpAdaptor adapter = ONNXMaxPoolSingleOutOpAdaptor(op);
    mlir::MLIRContext *context =  op.getContext();
    Location loc = op.getLoc();

    Value x = op.X();				// ONNX operands

    auto autopad = op.auto_padAttr(); 		// ::mlir::StringAttr
    auto dilations = op.dilationsAttr();   	// ::mlir::ArrayAttr
    auto kernal_shape = op.kernel_shapeAttr(); 	// ::mlir::ArrayAttr
    auto pads = op.padsAttr(); 			// ::mlir::ArrayAttr
    auto strides = op.stridesAttr(); 		// ::mlir::ArrayAttr
    int64_t ceiling_mode = op.ceil_mode();         // int64_t
    auto ceiling_mode_attr = op.ceil_modeAttr();// ::mlir::IntegerAttr


    auto storage_order_attr = op.storage_orderAttr();	// ::mlir::IntegerAttr
    int64_t storage_order = op.storage_order();	// int64_t

    auto one = 1;
    auto three = 3;
    auto zero  = 0;
    
    //auto f3 = IntegerAttr::get(ceiling_mode_attr.getType(), three);
    auto f3 = IntegerAttr::get( IntegerType::get(op.getContext(), 1) , APInt(64, three));
    auto f0 = IntegerAttr::get( IntegerType::get(op.getContext(), 1) , APInt(64, zero));
    //auto f0 = IntegerAttr::get(ceiling_mode_attr.getType(), zero);

    Value f3v = rewriter.create<ConstantIntOp>(loc,f3);
    Value f0v = rewriter.create<ConstantIntOp>(loc,f0);
    Value f1v = rewriter.create<ConstantIntOp>(loc,storage_order_attr);
    Value f2v = rewriter.create<ConstantIntOp>(loc,storage_order_attr);

    
    Value ceiling_mode_val = rewriter.create<ConstantIntOp>(loc,ceiling_mode_attr);
    Value storage_order_val = rewriter.create<ConstantIntOp>(loc,storage_order_attr);
    Type tensorType = op.getType();

    Value stridesList = rewriter.create<PrimListConstructOp>(
	loc, Torch::ListType::get(rewriter.getType<Torch::IntType>()), ValueRange{f1v, f2v}); 

    Value dilationList = rewriter.create<PrimListConstructOp>(
	loc, Torch::ListType::get(rewriter.getType<Torch::IntType>()), ValueRange{f1v, f2v});

    Value padsList = rewriter.create<PrimListConstructOp>(
	loc, Torch::ListType::get(rewriter.getType<Torch::IntType>()), ValueRange{f0v, f0v, f0v, f0v});

    Value kernalShapeList = rewriter.create<PrimListConstructOp>(
	loc, Torch::ListType::get(rewriter.getType<Torch::IntType>()), ValueRange{f3v, f3v});

    auto r0 = Torch::ValueTensorType::getWithLeastStaticInformation(op.getContext());
    auto xtt = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>( op.getLoc(), r0, x);

    llvm::outs() << "xtt torch tensor from MLIR tensor " << "\n" << xtt << "\n" << "\n";

    //auto t = Torch::ValueTensorType::get(context, optionalSizesArrayRef, x1.getType());

    Value atenmaxpool2d = rewriter.create<AtenMaxPool2dOp>(loc, x.getType(), xtt, 
		kernalShapeList, stridesList, padsList, dilationList, ceiling_mode_val);

    llvm::outs() << "AtenMaxPool2dOp operation creation" << "\n" << atenmaxpool2d << "\n" << "\n";

    Value result = atenmaxpool2d; 

    llvm::outs() << "Before Writer replace Op " << "\n"; 
    
    rewriter.replaceOpWithNewOp<TensorStaticInfoCastOp>(op, op.getType(), result);

    llvm::outs() << "After Writer replace Op " << "\n"; 

    return success();
  }
};



} // namespace 


namespace { 

class  ONNXToAtenMaxPool2dOpTransformPass
    : public PassWrapper<ONNXToAtenMaxPool2dOpTransformPass, OperationPass<::mlir::FuncOp>> {
     void runOnOperation() override {
          MLIRContext *context = &getContext();
          
	  auto *dialect1 = context->getOrLoadDialect<::mlir::torch::Torch::TorchDialect>();
	  auto *dialect2 = context->getOrLoadDialect<::mlir::torch::TorchConversion::TorchConversionDialect>();

	  RewritePatternSet patterns(context);
	  ConversionTarget target(*context);
	  target.addLegalDialect<Torch::TorchDialect>();
	  target.addLegalDialect<::mlir::torch::Torch::TorchDialect>();

	  llvm::outs() << "ONNXToAtenMaxPool2dOpTransformPass Before OpTransform " << "\n"; 
	  patterns.add<DecomposeONNXToAtenMaxPool2DOp>(context);

	  //target.addIllegalOp<ONNXConvOp>();
	  llvm::outs() << "ONNXToAtenMaxPool2dOpTransformPass After OpTransform " << "\n"; 

	  
	  if (failed(applyPartialConversion(getOperation(), target,
	      std::move(patterns)))) {
	      return signalPassFailure();
	  }

	  if (onnxOpTransformReport) {
	    llvm::outs() << "ONNXToAtenConv2DOpTransformPass iterated " << 3 
			 << " times, converged "
			 << "\n";
	  }
      }
};


} // end anonymous namespace

/*!
 * Create an instrumentation pass.
 */
std::unique_ptr<Pass> mlir::createONNXToAtenMaxPool2dOpTransformPass() {
  return std::make_unique<ONNXToAtenMaxPool2dOpTransformPass>();
}
