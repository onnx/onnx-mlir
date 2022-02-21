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


#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/StringExtras.h"

#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
#include "torch-mlir/Dialect/TorchConversion/Transforms/BackendTypeConversion.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

#ifdef _WIN32
#include <io.h>
#endif

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::TorchConversion;

/**
 * ONNX Pad Operation 
 *
 * ONNX Padding modes supported as in ONNX Pad operation
 * TORCH padding is govenred by 4-tuple (left, right, top, bottom)  
 *
 * Where is this used? 
 * padding is typically used just before/after convolution operation
 *
 * Operands :
 * data		tensor of 16-bit/32-bit/64-bit float values or memref of any type values
 * pads	        tensor of 64-bit signless integer values or memref of any type values	
 * constant_value tensor of 16-bit/32-bit/64-bit float values or memref of any type values	
 *
 * Output   : 
 *     
 * Y		tensor of 16-bit/32-bit/64-bit float values or memref of any type values or none type
 *		Output data tensor from average or max pooling across the input tensor. Dimensions will 
 *		vary based on various kernel, stride, and pad sizes. Floor value of the dimension is used
 *              differentiable 
 *
 * Attributes 
 * mode 		string attribute constant 
 * 
 * AtenConstantPadNdOp is used 
 * -------------------------------
 *
 * 
 * Validation 
 * ----------
 * ./scripts/docker/build_with_docker.py --external-build --build-dir build --command "build/Ubuntu1804-Release/third-party/onnx-mlir/Release/bin/onnx-mlir --EmitONNXIR --debug /home/sachin/try10/FlexML/third-party/onnx-mlir/third_party/onnx/onnx/backend/test/data/pytorch-operator/test_operator_pad/model.onnx" 
 * 
 */
namespace {

class ApplyONNXPaddingToAtenConstantPadNdOp : public OpRewritePattern<ONNXPadOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ONNXPadOp op,
                                PatternRewriter &rewriter) const override {

    ONNXPadOpAdaptor adapter = ONNXPadOpAdaptor(op);
    mlir::MLIRContext *context =  op.getContext();
    Location loc = op.getLoc();

    Value data = op.data();				// ONNX operands
    auto pads = op.pads();				// ONNX operands
    Value const_value = op.constant_value();		// ONNX operands

    llvm::outs() << "pads type" << "\n" << pads.getType() << "\n" << "\n";
    llvm::outs() << "pads data " << "\n" << pads << "\n" << "\n";

    // mode being constant
    // auto mode = op.modeAttr(); 				// ::mlir::StringAttr

    auto one = 1;
    auto two = 2;
    auto zero = 0;
    auto three = 3;
    //auto four = 4;
    auto ty = IntegerType::get(op.getContext(), 64);

    auto f33 = IntegerAttr::get(ty, three);
    auto f11 = IntegerAttr::get(ty, one);
    auto f22 = IntegerAttr::get(ty, two);
    auto f00 = IntegerAttr::get(ty, zero);
    //auto f44 = IntegerAttr::get(ty, four);

    Value f3v = rewriter.create<ConstantIntOp>(loc,f33);
    Value f2v = rewriter.create<ConstantIntOp>(loc,f22);
    Value f1v = rewriter.create<ConstantIntOp>(loc,f11);
    Value f0v = rewriter.create<ConstantIntOp>(loc,f00);
    //Value f4v = rewriter.create<ConstantIntOp>(loc,f44);

    TensorType data_tensor_type  = data.getType().cast<TensorType>();
    TensorType const_val_tensor_type  = const_value.getType().cast<TensorType>();

    llvm::outs() << "data type " << "\n" << data_tensor_type.getElementType() << "\n" << "\n";
    llvm::outs() << "data " << "\n" << data << "\n" << "\n";

    auto dataTy = Torch::ValueTensorType::get(context, data_tensor_type.getShape(), 
								data_tensor_type.getElementType());
    auto constValTy = Torch::ValueTensorType::get(context, const_val_tensor_type.getShape(), 
								const_val_tensor_type.getElementType());

    auto dtt  = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>( loc, dataTy, data); 
    auto ctt  = rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>( loc, constValTy, const_value); 

    // ONNX : b1, e1, b2, e2, b3, e3, b4, e4
    // TORCH : b1, b2, b3, b4, e1, e2, e3, e4
 
    auto padsList1 = rewriter.create<PrimListConstructOp>(
	         	loc, Torch::ListType::get(rewriter.getType<Torch::IntType>()), 
				ValueRange{f0v, f0v, f0v, f1v, f0v, f2v, f0v, f3v} );

    for (auto p : padsList1.elements()) {
    	llvm::outs() << " padding list element: " << "\n" << p << "\n" << "\n";
    }

    TensorType op_tensor_type = op->getResult(0).getType().cast<TensorType>();
    auto resultTy = Torch::ValueTensorType::get(op.getContext(), op_tensor_type.getShape(), 
							op_tensor_type.getElementType());

    Value atenconstantpad = rewriter.create<AtenConstantPadNdOp>(loc, resultTy, dtt, padsList1, ctt);

    llvm::outs() << "AtenConstantPadNdOp operation creation" << "\n" << atenconstantpad << "\n" << "\n";

    Value result = atenconstantpad;

    llvm::outs() << "Before Writer replace Op " << atenconstantpad << "\n"; 
    
    rewriter.replaceOpWithNewOp<torch::TorchConversion::ToBuiltinTensorOp>(op, op->getResult(0).getType(), result);

    llvm::outs() << "After Writer replace Op " << "\n"; 

    return success();
  }
};



} // namespace 


namespace { 

class  ONNXToAtenConstantPadNdOpTransformPass
    : public PassWrapper<ONNXToAtenConstantPadNdOpTransformPass, OperationPass<::mlir::FuncOp>> {

     StringRef getArgument() const override { return "onnx-to-aten-constantpadnd-transform"; }


     void runOnOperation() override {
          MLIRContext *context = &getContext();
          
	  //auto *dialect1 = context->getOrLoadDialect<::mlir::torch::Torch::TorchDialect>();
	  //auto *dialect2 = context->getOrLoadDialect<::mlir::torch::TorchConversion::TorchConversionDialect>();

	  RewritePatternSet patterns(context);
	  ConversionTarget target(*context);
	  target.addLegalDialect<Torch::TorchDialect>();
	  target.addLegalDialect<::mlir::torch::Torch::TorchDialect>();

	  llvm::outs() << "ONNXToAtenConstantPadNdOpTransformPass Before OpTransform " << "\n"; 
	  patterns.add<ApplyONNXPaddingToAtenConstantPadNdOp>(context);

	  llvm::outs() << "ONNXToAtenConstantPadNdOpTransformPass After OpTransform " << "\n"; 

	  
	  if (failed(applyPartialConversion(getOperation(), target,
	      std::move(patterns)))) {
	      return signalPassFailure();
	  }

	  if (onnxOpTransformReport) {
	    llvm::outs() << "ONNXToAtenConstantPadNdOpTransformPass iterated " << 3 
			 << " times, converged "
			 << "\n";
	  }
      }
};


} // end anonymous namespace

/*!
 * Create an instrumentation pass.
 */
std::unique_ptr<Pass> mlir::createONNXToAtenConstantPadNdOpTransformPass() {
  return std::make_unique<ONNXToAtenConstantPadNdOpTransformPass>();
}
