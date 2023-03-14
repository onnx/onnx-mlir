/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- EntryPoint.cpp - EntryPoint Op --------------------===//
//
// Copyright (c) 2022 Advanced Micro Devices, Inc.
//
// =============================================================================
//
// This file removes the "onnx.EntryPoint" and renames the func.func to @forward
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSACommon.hpp"
#include "src/Conversion/ONNXToTOSA/ONNXToTOSALegalizeUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

class ONNXEntryPointLoweringToTOSA
    : public OpConversionPattern<ONNXEntryPointOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  using OpAdaptor = typename ONNXEntryPointOp::Adaptor;
  // This function is from typesTransformsToTorchPass.cpp
  LogicalResult matchAndRewrite(ONNXEntryPointOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    auto functionName = op.getFunc().getRootReference().getValue();
    // Differs from origin to get module
    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto mainFuncOp = module.lookupSymbol<func::FuncOp>(functionName);
    if (mainFuncOp) {
      StringRef forwardRef = "forward";
      auto forwardAttr = StringAttr::get(module.getContext(), forwardRef);
      mainFuncOp->setAttr(llvm::StringRef("sym_name"), forwardAttr);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateLoweringONNXEntryPointOpToTOSAPattern(ConversionTarget &target,
    RewritePatternSet &patterns, TypeConverter &typeConverter,
    MLIRContext *ctx) {
  patterns.insert<ONNXEntryPointLoweringToTOSA>(ctx);
}

} // namespace onnx_mlir
