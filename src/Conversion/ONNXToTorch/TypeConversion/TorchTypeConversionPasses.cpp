/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====----- TorchTypeConversionPasses.cpp - ONNX types to Torch types conversion
// passes ---------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// ===================================================================================================
//
// This file defines additional passes for finishing the function type
// conversion as well as finalizing the type conversion to Torch types.
//
//===--------------------------------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"

using namespace mlir;
using namespace mlir::torch;
using namespace onnx_mlir;

//===----------------------------------------------------------------------===//
// FuncTorchTypeConversionPass
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
struct FuncTorchTypeConversionPass
    : public PassWrapper<FuncTorchTypeConversionPass, OperationPass<ModuleOp>> {

  StringRef getArgument() const override {
    return "convert-function-types-to-torch-types";
  }

  StringRef getDescription() const override {
    return "Convert types in function calls and definitions to torch types.";
  }

  // Make sure that we have a valid default constructor and copy
  // constructor to make sure that the options are initialized properly.
  FuncTorchTypeConversionPass() = default;
  void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();

    TypeConverter typeConverter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    typeConverter.addConversion([](Type type) { return type; });
    onnx_mlir::setupTorchTypeConversion(target, typeConverter);

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](func::CallOp op) { return typeConverter.isLegal(op); });

    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addLegalOp<ModuleOp>();

    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
             isLegalForBranchOpInterfaceTypeConversionPattern(
                 op, typeConverter) ||
             isLegalForReturnOpTypeConversionPattern(op, typeConverter);
    });

    if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace onnx_mlir

// std::unique_ptr<OperationPass<ModuleOp>>
std::unique_ptr<mlir::Pass> onnx_mlir::createFuncTorchTypeConversionPass() {
  return std::make_unique<FuncTorchTypeConversionPass>();
}

//===----------------------------------------------------------------------===//
// FinalizingTorchTypeConversionPass
//===----------------------------------------------------------------------===//

namespace onnx_mlir {
// In a finalizing conversion, we know that all of the source types have been
// converted to the destination types, so the materialization becomes an
// identity.
template <typename OpTy>
class FinalizeMaterialization : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;
  LogicalResult matchAndRewrite(OpTy op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands()[0]);
    return success();
  }
};
} // namespace onnx_mlir

template <typename OpTy>
static void setupFinalization(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  target.addIllegalOp<OpTy>();
  patterns.add<FinalizeMaterialization<OpTy>>(
      typeConverter, patterns.getContext());
}

template <typename OpTy, typename OpTy2, typename... OpTys>
static void setupFinalization(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  setupFinalization<OpTy>(target, patterns, typeConverter);
  setupFinalization<OpTy2, OpTys...>(target, patterns, typeConverter);
}

namespace onnx_mlir {
struct FinalizingTorchTypeConversionPass
    : public PassWrapper<FinalizingTorchTypeConversionPass,
          OperationPass<ModuleOp>> {

  StringRef getArgument() const override {
    return "finalize-torch-type-conversion";
  }

  StringRef getDescription() const override {
    return "Finalize the conversion from builtin types to torch types.";
  }

  // Make sure that we have a valid default constructor and copy
  // constructor to make sure that the options are initialized properly.
  FinalizingTorchTypeConversionPass() = default;

  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();

    TypeConverter typeConverter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    typeConverter.addConversion([](Type type) { return type; });
    onnx_mlir::setupTorchTypeConversion(target, typeConverter);

    // Mark materializations as illegal in this pass (since we are finalizing)
    // and add patterns that eliminate them.
    setupFinalization<UnrealizedConversionCastOp>(
        target, patterns, typeConverter);

    // If all result types are legal, and all block arguments are legal, then
    // all types in the program are legal.
    //
    // We also check that the operand types are legal to avoid creating invalid
    // IR. For example, this prevents the patterns from updating
    // the types of the operands to a return op without updating the enclosing
    // function.
    target.markUnknownOpDynamicallyLegal(
        [&](Operation *op) { return typeConverter.isLegal(op); });

    if (failed(applyFullConversion(func, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace onnx_mlir

// std::unique_ptr<OperationPass<func::FuncOp>>
std::unique_ptr<mlir::Pass>
onnx_mlir::createFinalizingTorchTypeConversionPass() {
  return std::make_unique<FinalizingTorchTypeConversionPass>();
}
