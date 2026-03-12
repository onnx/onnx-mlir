/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- AppendDecodingStrategy.cpp - Decoding Strategy ---------------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a pass that appends a decoding strategy such as greedy
// algorith, top_k, top_p, to the end of the main graph.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {

#define GEN_PASS_DEF_APPENDDECODINGSTRATEGYPASS
#include "src/Dialect/ONNX/Transforms/Passes.h.inc"
} // namespace onnx_mlir

namespace {

class AppendDecodingStrategyPass
    : public onnx_mlir::impl::AppendDecodingStrategyPassBase<
          AppendDecodingStrategyPass> {

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AppendDecodingStrategyPass)

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Find the ONNXEntryPoint operation that knows which is the main function.
    ONNXEntryPointOp entryPointOp;
    moduleOp.walk([&entryPointOp](ONNXEntryPointOp op) -> WalkResult {
      entryPointOp = op;
      return WalkResult::advance();
    });
    if (!entryPointOp)
      return;

    // Get the main function.
    SymbolRefAttr funcRefAttr =
        entryPointOp.getOperation()->getAttrOfType<SymbolRefAttr>(
            ONNXEntryPointOp::getEntryPointFuncAttrName());
    StringRef mainFuncName = funcRefAttr.getLeafReference().getValue();
    Operation *mainFuncOp = moduleOp.lookupSymbol(mainFuncName);
    if (!mainFuncOp)
      return;
    func::FuncOp mainFunc = mlir::dyn_cast<func::FuncOp>(mainFuncOp);
    if (!mainFunc)
      return;

    // Find the ONNXReturnOp.
    mlir::Operation *term = mainFunc.getBody().back().getTerminator();
    ONNXReturnOp returnOp = mlir::dyn_cast<ONNXReturnOp>(term);
    if (!returnOp)
      return;

    returnOp.dump();
  }
};

} // namespace
