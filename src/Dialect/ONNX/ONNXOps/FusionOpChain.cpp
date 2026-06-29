/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ FusionOpChain.cpp - ONNXFusedOp builder base -----------===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================

#include "src/Dialect/ONNX/ONNXOps/FusionOpChain.hpp"

#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "fusion-op-chain"

using namespace mlir;

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// FusionOpChain — non-virtual method implementations
//===----------------------------------------------------------------------===//

ONNXFusedOp FusionOpChain::createFusedOp(
    PatternRewriter &rewriter, Location loc, StringRef kind) {
  // Build the set of values produced by the chain ops themselves; these
  // are visible inside the body via the clone mapping and never external.
  DenseSet<Value> chainProduced;
  for (Operation *op : ops)
    for (Value result : op->getResults())
      chainProduced.insert(result);

  // Pre-scan: collect ALL external values the chain ops need.
  //   - Constant-like ops (ONNXConstantOp, ConstantLike trait) will be
  //     cloned inside the body — they do NOT become block arguments.
  //   - Everything else (non-constant tensors, e.g. dynamically-computed
  //     reshape shape vectors) becomes an additional FusedOp input.
  SmallVector<Value> fusedInputs;
  DenseSet<Value> inputSet;

  std::function<void(Value)> collectExternals = [&](Value v) {
    if (inputSet.contains(v) || chainProduced.contains(v))
      return;
    Operation *defOp = v.getDefiningOp();
    if (!defOp) {
      // Block argument (e.g. function parameter) — thread through as an input.
      inputSet.insert(v);
      fusedInputs.push_back(v);
      return;
    }
    if (defOp->hasTrait<mlir::OpTrait::ConstantLike>() ||
        mlir::isa<ONNXNoneOp, ONNXConstantOp>(defOp)) {
      for (Value operand : defOp->getOperands())
        collectExternals(operand);
    } else {
      inputSet.insert(v);
      fusedInputs.push_back(v);
    }
  };
  for (Operation *op : ops)
    for (Value operand : op->getOperands())
      collectExternals(operand);

  // Build FusedOp with the complete input list.
  SmallVector<Type, 4> outputTypes;
  for (Value v : finalResults)
    outputTypes.push_back(v.getType());
  auto fusedOp = ONNXFusedOp::create(
      rewriter, loc, outputTypes, rewriter.getStringAttr(kind), fusedInputs);

  // Build the isolated body: one block argument per fusedInput.
  SmallVector<Type, 4> argTypes;
  SmallVector<Location> argLocs;
  for (Value v : fusedInputs) {
    argTypes.push_back(v.getType());
    argLocs.push_back(v.getLoc());
  }
  Block *body = rewriter.createBlock(&fusedOp.getBody(), {}, argTypes, argLocs);
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(body);

  // Map every fusedInput to its corresponding block argument.
  IRMapping mapping;
  for (auto [v, arg] : llvm::zip(fusedInputs, body->getArguments()))
    mapping.map(v, arg);

  // Recursively clone constant-producing ops inside the body on demand.
  std::function<void(Value)> ensureInBody = [&](Value v) {
    if (mapping.contains(v) || chainProduced.contains(v))
      return;
    Operation *defOp = v.getDefiningOp();
    if (!defOp)
      return;
    assert((defOp->hasTrait<mlir::OpTrait::ConstantLike>() ||
               mlir::isa<ONNXNoneOp, ONNXConstantOp>(defOp)) &&
           "non-constant external value not collected in pre-scan");
    for (Value operand : defOp->getOperands())
      ensureInBody(operand);
    rewriter.clone(*defOp, mapping);
  };

  for (Operation *op : ops) {
    for (Value operand : op->getOperands())
      ensureInBody(operand);
    rewriter.clone(*op, mapping);
  }

  // Yield the mapped results of the last chain ops.
  SmallVector<Value> yieldVals;
  for (Value v : finalResults)
    yieldVals.push_back(mapping.lookup(v));
  ONNXYieldOp::create(rewriter, loc, ValueRange(yieldVals));

  return fusedOp;
}

ONNXFusedOp FusionOpChain::fuse(PatternRewriter &rewriter, Location loc) {
  assert(!ops.empty() && "fuse() called with empty ops list");
  rewriter.setInsertionPoint(ops.back());
  ONNXFusedOp fusedOp = create(rewriter, loc);
  replaceAndErase(rewriter, fusedOp);
  return fusedOp;
}

ONNXFusedOp FusionOpChain::create(PatternRewriter &rewriter, Location loc) {
  ONNXFusedOp fusedOp = createFusedOp(rewriter, loc, getKind());
  embedAttrs(fusedOp);
  return fusedOp;
}

void FusionOpChain::retrieveOpsAndOutputValues(ONNXFusedOp fusedOp) {
  ops.clear();
  finalResults.clear();
  Block &body = fusedOp.getBody().front();
  for (Operation &op : body) {
    if (isa<ONNXYieldOp>(&op)) {
      for (Value v : op.getOperands())
        finalResults.push_back(v);
    } else {
      ops.push_back(&op);
    }
  }
}

void FusionOpChain::replaceAndErase(
    PatternRewriter &rewriter, ONNXFusedOp fusedOp) {
  DenseMap<Value, unsigned> outputMap;
  for (auto [idx, v] : llvm::enumerate(finalResults))
    outputMap[v] = idx;

  for (int i = (int)ops.size() - 1; i >= 0; --i) {
    auto it = outputMap.find(ops[i]->getResult(0));
    if (it != outputMap.end())
      rewriter.replaceOp(ops[i], fusedOp.getOutputs()[it->second]);
    else
      rewriter.eraseOp(ops[i]);
  }
}

bool FusionOpChain::verifyAndRetrieveAttrs(ONNXFusedOp fusedOp) {
  if (!retrieveAttrs(fusedOp)) {
    LLVM_DEBUG(llvm::dbgs() << "FusionOpChain: retrieveAttrs failed for kind '"
                            << fusedOp.getKind() << "'\n");
    return false;
  }
  if (!verify()) {
    LLVM_DEBUG(llvm::dbgs() << "FusionOpChain: verify failed for kind '"
                            << fusedOp.getKind() << "'\n");
    return false;
  }
  return true;
}

} // namespace onnx_mlir
