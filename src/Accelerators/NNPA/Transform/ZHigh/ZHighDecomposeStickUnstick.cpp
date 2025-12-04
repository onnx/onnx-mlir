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
  // When decomposing a unstick op, a DLF16ToF32 op is propagated down to
  // return, stick and compute ops.
  // - When it meets a stick op, the DLF16ToF32 in unstick and the F32ToDLF16s
  // in stick ops are removed.
  // - When it meets a return or a compute op, each return or compute op needs
  // one DLF16ToF32.
  //
  // So, the decomposition is only beneficial if the total number of removed
  // F32ToDLF16/DLF16ToF32 is strictly greater than the total number of added
  // DLF16ToF32.
  //
  uint64_t numStickOps = 0;
  uint64_t numComputeOps = 0;
  uint64_t numDataOps = 0;
  uint64_t numReturns = 0;
  bool allDataOpsAreView = false;

  SmallVector<Operation *> workList;
  DenseSet<Operation *> visited;
  workList.push_back(unstickOp.getOperation());
  while (!workList.empty()) {
    Operation *current = workList.back();
    workList.pop_back();
    for (auto *user : current->getUsers()) {
      bool isONNXOp = user->getDialect()->getNamespace() ==
                      ONNXDialect::getDialectNamespace();
      // ZHighQuantizedStickOp currently does not work with DLF16 in symmetric
      // mode. Temporarily put it into the ONNX category. Once it supports DLF16
      // with symmetric mode, it would be in the same category as ZHighStickOp.
      isONNXOp |= isa<ZHighQuantizedStickOp>(user);
      // Continue if user is StickOp or F32ToDLF16.
      if (isa<ZHighStickOp, ZHighF32ToDLF16Op>(user)) {
        visited.insert(user);
        numStickOps++;
        continue;
      }
      // Continue if user is an ONNX op but not a data movement op.
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
      // NNPA ops cannot consumes any (unstickified) output from unstick but
      // ONNX ops can, so if a consumming op here is not an ONNX op, we don't
      // know how to compute cost for it. Thus, just stop.
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

  // No data movement ops. Do nothing.
  if (numDataOps == 0)
    return false;

  // There is no benefit if all data ops are just view and there is no stick.
  if (allDataOpsAreView && (numStickOps == 0))
    return false;

  // For N stick ops, we can remove N F32ToDLF16s in N stick ops plus 1
  // DLF16ToF32 in the unstick op.
  uint64_t numRemovedOps = numStickOps + 1;
  // Each return or compute op needs one DLF16ToF32.
  uint64_t numAddedOps = numReturns + numComputeOps;
  // There is no benefit if the number of added DLF16ToF32 ops is more than or
  // equal to the number of removed DLF16ToF32/F32ToDLF16.
  if (numAddedOps > 1 && numAddedOps >= numRemovedOps)
    return false;

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
