/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- DecomposeONNXToAtenConv2DOp.cpp - ONNX Op Transform
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
 * ONNX Constant  operation
 *
 * Creates the constant tensor.
 *
 * Operands :
 *
 *
 * Validation
 * ----------
 * /scripts/docker/build_with_docker.py --external-build --build-dir build
 * --command
 * "build/Ubuntu1804-Release/third-party/onnx-mlir/Release/bin/onnx-mlir
 * --EmitONNXIR --debug
 * third-party/onnx-mlir/third_party/onnx/onnx/backend/test/data/node/test_constant/model.onnx"
 *
 * Limitations
 * -----------
 * uses literal.
 *
 */
namespace {

class DecomposeONNXToConstOp : public OpRewritePattern<ONNXConstantOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(
      ONNXConstantOp op, PatternRewriter &rewriter) const override {

    ONNXConstantOpAdaptor adapter = ONNXConstantOpAdaptor(op);
    mlir::MLIRContext *context = op.getContext();
    Location loc = op.getLoc();

    auto sparese_value_attr = op.sparse_valueAttr(); // ::mlir::Attribute
    auto value_attr = op.valueAttr();                // ::mlir::Attribute
    bool v00 = value_attr.isa<::mlir::FloatAttr>();

    auto val = adapter.value();
    ::mlir::FloatAttr va = adapter.value_floatAttr();

    auto value_flt_attr = op.value_floatAttr();          // ::mlir::FloatAttr
    auto value_flts_attr_array = op.value_floatsAttr();  // ::mlir::ArrayAttr
    auto value_int_attr = op.value_intAttr();            // ::mlir::IntegerAttr
    auto value_ints_attr_array = op.value_intsAttr();    // ::mlir::ArrayAttr
    auto value_str_attr = op.value_stringAttr();         // ::mlir::StringAttr
    auto value_strs_attr_array = op.value_stringsAttr(); // ::mlir::ArrayAttr

    //      Steps
    //	1) Extract float attributes array from ONNX and compare with the Netron
    //file, 	2) Find the shape of this array in step 1, 	3) Create the result
    //type, 	4) Create the torch tensor of shape as in 2, 	5) Create the torch op
    //and replace it.

    TensorType flt_array_tensor_type = value_attr.getType().cast<TensorType>();
    TensorType op_tensor_type = op->getResult(0).getType().cast<TensorType>();
    auto resultTy = Torch::ValueTensorType::get(op.getContext(),
        op_tensor_type.getShape(), op_tensor_type.getElementType());

    auto one = 1;
    auto three = 3;

    auto ty = IntegerType::get(op.getContext(), 32);
    auto f33 = IntegerAttr::get(ty, three);
    Value device = rewriter.create<ConstantDeviceOp>(loc, "CPU");
    Value f3v = rewriter.create<ConstantIntOp>(loc, f33);
    Value bfv = rewriter.create<ConstantBoolOp>(loc, false);
    auto xTy =
        Torch::ValueTensorType::get(context, flt_array_tensor_type.getShape(),
            flt_array_tensor_type.getElementType());

    Value literal =
        rewriter.create<Torch::ValueTensorLiteralOp>(loc, resultTy, value_attr);


    Value result = literal;

    rewriter.replaceOpWithNewOp<torch::TorchConversion::ToBuiltinTensorOp>(
        op, op->getResult(0).getType(), result);

    return success();
  }
};

} // namespace

namespace {

class ONNXToAtenConstantOpTransformPass
    : public PassWrapper<ONNXToAtenConstantOpTransformPass,
          OperationPass<::mlir::FuncOp>> {
  StringRef getArgument() const override {
    return "onnx-to-aten-constop-transform";
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();

    auto *dialect1 =
        context->getOrLoadDialect<::mlir::torch::Torch::TorchDialect>();
    auto *dialect2 = context->getOrLoadDialect<
        ::mlir::torch::TorchConversion::TorchConversionDialect>();

    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    target.addLegalDialect<Torch::TorchDialect>();
    target.addLegalDialect<::mlir::torch::Torch::TorchDialect>();
    target.addLegalDialect<
        ::mlir::torch::TorchConversion::TorchConversionDialect>();

    patterns.add<DecomposeONNXToConstOp>(context);


    if (failed(applyPartialConversion(
            getOperation(), target, std::move(patterns)))) {
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
std::unique_ptr<Pass> mlir::createONNXToAtenConstantOpTransformPass() {
  return std::make_unique<ONNXToAtenConstantOpTransformPass>();
}
