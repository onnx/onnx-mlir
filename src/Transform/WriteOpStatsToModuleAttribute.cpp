/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===- WriteOpStatsToModuleAttribute.cpp - Operation statistics pass ------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a Module level pass that writes operation statistics
// to a module attribute.
//
//===----------------------------------------------------------------------===//

#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"

namespace onnx_mlir {

#define GEN_PASS_DEF_WRITEOPSTATSTOMODULEATTRIBUTEPASS
#include "src/Transform/Passes.h.inc"

} // namespace onnx_mlir

using namespace mlir;

namespace {

class WriteOpStatsToModuleAttributePass
    : public onnx_mlir::impl::WriteOpStatsToModuleAttributePassBase<
          WriteOpStatsToModuleAttributePass> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      WriteOpStatsToModuleAttributePass)

  void runOnOperation() override {
    Operation *module = getOperation();

    std::string opStatsJSON;
    llvm::raw_string_ostream opStatsStream(opStatsJSON);

    OpPassManager pm("builtin.module");
    pm.addNestedPass<func::FuncOp>(
        mlir::createPrintOpStatsPass(opStatsStream, /*printAsJSON=*/true));
    if (failed(runPipeline(pm, module)))
      return signalPassFailure();

    opStatsStream.flush();
    module->setAttr(
        "onnx-mlir.op_stats", StringAttr::get(&getContext(), opStatsJSON));
  }
};

} // namespace
