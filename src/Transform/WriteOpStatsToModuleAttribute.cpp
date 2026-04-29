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

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

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
    Operation *moduleOp = getOperation();
    // Compute the operation statistics for the currently visited operations.
    llvm::StringMap<int64_t> opCount;
    moduleOp->walk([&](Operation *op) {
      if (isa<ModuleOp, func::FuncOp, ONNXEntryPointOp>(op))
        return WalkResult::advance();
      // Construct op name for printing.
      std::string opName = op->getName().getStringRef().str();
      // Append a rank string that contains ranks of all inputs.
      // For example, a rank string ".2D.1D.unranked.0D.none" is for the
      // folowing inputs
      // - (tensor<?x?xf32>, tensor<?xf32>, tensor<*xf32>, tensor<?xf32>, none)
      for (Value v : op->getOperands()) {
        if (onnx_mlir::isNoneValue(v)) {
          opName += ".none";
          continue;
        }
        if (auto type = mlir::dyn_cast<ShapedType>(v.getType()))
          opName += "." + std::to_string(type.getRank()) + "D";
        else
          opName += ".unranked";
      }
      // Append ".scalar" if this is a scalar op since looking at rank we don't
      // know if it is scalar or not (e.g. both tensor<1xf32> and tensor<5xf32>
      // have rank of 1).
      // A scalar op is the one whose inputs and outputs are scalar tensors
      // (tensor<dtype> or tensor<1xdtype>) or none.
      bool isScalarOp = llvm::all_of(op->getOperands(), [](Value v) {
        return onnx_mlir::isNoneValue(v) || onnx_mlir::isScalarTensor(v);
      });
      isScalarOp &= llvm::all_of(op->getResults(), [](Value v) {
        return onnx_mlir::isNoneValue(v) || onnx_mlir::isScalarTensor(v);
      });
      if (isScalarOp)
        opName += ".scalar";
      // Record this operation.
      ++opCount[StringRef(opName)];
      return WalkResult::advance();
    });

    // Write to a JSON string.
    std::string opStatsJSON;
    llvm::raw_string_ostream os(opStatsJSON);
    // Sort by operation name.
    SmallVector<StringRef, 64> sorted(opCount.keys());
    llvm::sort(sorted);
    os << "{\n";
    for (unsigned i = 0, e = sorted.size(); i != e; ++i) {
      const auto &key = sorted[i];
      os << "  \"" << key << "\" : " << opCount[key];
      if (i != e - 1)
        os << ",\n";
      else
        os << "\n";
    }
    os << "}\n";
    os.flush();

    // Put the op stats into an attribute in ModuleOp.
    moduleOp->setAttr(
        "onnx-mlir.op_stats", StringAttr::get(&getContext(), opStatsJSON));
  }
};

} // namespace
