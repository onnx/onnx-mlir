/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----- ZHighDecomposeStickUnstick.cpp - ZHigh High Level Optimizer ----===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a set of simple combiners for optimizing operations in
// the ZHigh dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/OpHelper.hpp"
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
#include "src/Accelerators/NNPA/Support/LayoutHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;
using namespace onnx_mlir::zhigh;

namespace onnx_mlir {
namespace zhigh {

bool canDecomposeUnstick(Value val) {
  ZHighUnstickOp unstickOp = val.getDefiningOp<ZHighUnstickOp>();
  assert(unstickOp && "Expected zhigh.UnstickOp");

  // We traverse from the current value and match the following patterns:
  // clang-format off
  // - unstick -> data-movement onnx ops -> ... -> data-movement onnx_ops -> return
  // - unstick -> data-movement onnx ops -> ... -> data-movement onnx_ops -> stick
  // - unstick -> data-movement onnx ops -> ... -> data-movement onnx_ops -> compute onnx ops
  // clang-format on
  //
  // which forms a tree (pottentially a graph) whose root is the current value
  // and leaves are return or stick or compute ops.
  //
  // We only decompose unstick when all the leaves are stick ops or compute ops.
  //
  // The following situations are not good for decompositon:
  // - there are more than one returned values as leaf.
  // - there are more than one compute ops as leaf.
  // - all data-movement ops are just view without data copying.
  //
  // If all the leaves are stick ops, DLF16ToF32 (of Unstick) will be propagated
  // down to F32ToDLF16 (of Stick) then they cancel each other.
  //
  // If all the leaves are compute ops, DLF16ToF32 will be propagated down so
  // that data-movement ops works on dlf16 type instead of f32 type, making the
  // data-movement ops potentially faster. Do this only there is only a single
  // branch for compute ops to avoid doing DFL16ToF32 multiple times.
  //
  uint64_t numStickOps = 0;
  uint64_t numComputeOps = 0;
  uint64_t numDataOps = 0;
  uint64_t numReturns = 0;
  bool allDataOpsAreView = false;

  SmallVector<Operation *> workList;
  DenseSet<Operation *> visited;
  workList.push_back(unstickOp.getOperation());
  // llvm::outs() << "visiting \n";
  // unstickOp.dump();
  while (!workList.empty()) {
    Operation *current = workList.back();
    workList.pop_back();
    // llvm::outs() << "current\n";
    // current->dump();
    for (auto *user : current->getUsers()) {
      // llvm::outs() << "user\n";
      // user->dump();
      bool isONNXOp = user->getDialect()->getNamespace() ==
                      ONNXDialect::getDialectNamespace();
      // Continue if user is StickOp or F32ToDLF16.
      if (isa<ZHighStickOp, ZHighF32ToDLF16Op>(user)) {
        visited.insert(user);
        numStickOps++;
        continue;
      }
      // Continue if user is an ONNX Ops but not a data movement op.
      if (isONNXOp && !isDataMovementONNXOp(user)) {
        numComputeOps++;
        visited.insert(user);
        continue;
      }
      // Continue if user is returned.
      if (isa<func::ReturnOp>(user)) {
        visited.insert(user);
        numReturns++;
        continue;
      }
      // Early stop if user is not an ONNX Op.
      if (!isONNXOp)
        return false;

      assert(isDataMovementONNXOp(user) && "Must be a data movement op");
      numDataOps++;

      // Check if this is a view op.
      if (numDataOps == 1)
        allDataOpsAreView = isViewONNXOp(user);
      else
        allDataOpsAreView &= isViewONNXOp(user);

      // Put to the work list if not visited.
      if (visited.insert(user).second)
        workList.push_back(user);
    };
  }
  // All data ops are just view and leaves are only compute ops, no need to
  // decompose.
  if (allDataOpsAreView && (numStickOps == 0 && numReturns == 0) {
    // llvm::outs() << "all are view\n";
    return false;
  }
  // More than one return values. If decomposing, there would have at least two
  // DLF16ToF32, which is not good.
  if (numReturns > 1) {
    // llvm::outs() << " > 1 returns\n";
    return false;
  }
  // More than one compute ops. If decomposing, there would have at least two
  // DLF16ToF32, which is not good.
  if (numComputeOps > 1) {
    // llvm::outs() << " > 1 compute ops\n";
    return false;
  }
  return true;
}

} // namespace zhigh
} // namespace onnx_mlir

namespace {
/// Use anonymous namespace to avoid duplication symbol `populateWithGenerated`
/// among multiple tablegen-based definitions.

/// Include the patterns defined in the Declarative Rewrite framework.
#include "src/Accelerators/NNPA/Transform/ZHigh/ONNXZHighDecomposeStickUnstick.inc"
} // namespace

namespace onnx_mlir {
namespace zhigh {

//===----------------------------------------------------------------------===//
// ZHigh layout propagation Pass
//===----------------------------------------------------------------------===//

struct ZHighDecomposeStickUnstickPass
    : public PassWrapper<ZHighDecomposeStickUnstickPass,
          OperationPass<func::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ZHighDecomposeStickUnstickPass)

  StringRef getArgument() const override {
    return "zhigh-decompose-stick-unstick";
  }

  StringRef getDescription() const override {
    return "Decompose ZHighStickOp and ZHighUnstickOp and do some "
           "layout-related optimizations.";
  }

  void runOnOperation() override {
    Operation *function = getOperation();
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());

    // Get patterns from tablegen.
    populateWithGenerated(patterns);

    // Get canonicalization rules for some important operations.
    ZHighDLF16ToF32Op::getCanonicalizationPatterns(patterns, &getContext());
    ZHighF32ToDLF16Op::getCanonicalizationPatterns(patterns, &getContext());
    ONNXLayoutTransformOp::getCanonicalizationPatterns(patterns, &getContext());
    (void)applyPatternsGreedily(function, std::move(patterns));
  }
};

std::unique_ptr<Pass> createZHighDecomposeStickUnstickPass() {
  return std::make_unique<ZHighDecomposeStickUnstickPass>();
}

} // namespace zhigh
} // namespace onnx_mlir
