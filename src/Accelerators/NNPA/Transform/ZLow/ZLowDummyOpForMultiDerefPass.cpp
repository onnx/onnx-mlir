/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- ZLowDummyOpForMultiDerefPass.cpp - ZLow Dummy Ops -----------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This pass introduce DummyOps for multiple dereferencing uses in a single op.
// This is a bypass to avoid calling normalize-memrefs on a single op with
// multiple dereferencing uses because normalize-memrefs does not support.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallSet.h"

#include "src/Accelerators/NNPA/Dialect/ZLow/ZLowOps.hpp"
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace zlow {

/// This pattern rewrites
/// ```mlir
///   zlow.op(%input1, %input1, $input2)
/// ```
/// into
/// ```mlir
///     %input1_1 = zlow.dummy(%input1)
///     zlow.op(%input1, %input1_1, $input2)
/// in order to avoid multiple dereferencing uses in a single ops which has not
/// yet been supported by MLIR normalize-memrefs pass. Otherwise, it fails to
/// normalize memrefs in `zlow.op`.

class ZLowDummyOpForMultiDerefPass
    : public PassWrapper<ZLowDummyOpForMultiDerefPass,
          OperationPass<mlir::ModuleOp>> {
public:
  StringRef getArgument() const override {
    return "zlow-dummyop-for-multideref";
  }

  StringRef getDescription() const override {
    return "Add ZLow DummyOps for multiple dereferencing uses in a single op";
  }

  void runOnOperation() override {
    auto walkResult =
        getOperation()->walk([&](mlir::Operation *op) -> WalkResult {
          if (op->getDialect()->getNamespace() ==
              ZLowDialect::getDialectNamespace()) {
            ValueRange operands = op->getOperands();
            llvm::SmallSet<uint64_t, 4> processed;
            for (uint64_t i = 0; i < operands.size() - 1; ++i) {
              if (processed.contains(i))
                continue;
              for (uint64_t j = i + 1; j < operands.size(); ++j) {
                if (processed.contains(j))
                  continue;
                if (operands[i].getDefiningOp() ==
                    operands[j].getDefiningOp()) {
                  OpBuilder b(op);
                  op->setOperand(
                      j, b.create<ZLowDummyOp>(op->getLoc(), operands[j]));
                  processed.insert(j);
                }
              }
            }
          }
          return WalkResult::advance();
        });
    if (walkResult.wasInterrupted())
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createZLowDummyOpForMultiDerefPass() {
  return std::make_unique<ZLowDummyOpForMultiDerefPass>();
}

} // namespace zlow
} // namespace onnx_mlir
