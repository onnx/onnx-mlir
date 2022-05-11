/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- InstrumentZHighPass.cpp - Instrumentation --------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a Function level pass that inserts instrumentation
// for ZHigh ops.
//
//===----------------------------------------------------------------------===//

#include <set>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

#include "src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Accelerators/NNPA/Pass/NNPAPasses.hpp"
#include "src/Accelerators/NNPA/Support/OMNNPAOptions.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Interface/ShapeInferenceOpInterface.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace zhigh {

/*!
 * This pass insert KrnlInstrumentOp before and after each ZHigh ops
 * TODO: There is duplicate code with InstrumentONNXPass of onnx-mlir.
 *       Need to update onnx-mlir to remove them
 */

// Strong-typed enum is NOT used because the value will be static_cast to int
enum InstrumentActions {
  InstrumentBeforeZHighOp,
  InstrumentAfterZHighOp,
  InstrumentReportTimeZHigh,
  InstrumentReportMemoryZHigh
};

// Inherited issue: no default value support for cl::bits
llvm::cl::bits<InstrumentActions> InstrumentControlBits(
    llvm::cl::desc("Specify what instrumentation actions at runtime:"),
    llvm::cl::values(
        clEnumVal(InstrumentBeforeZHighOp, "insert instrument before op"),
        clEnumVal(InstrumentAfterZHighOp, "insert instrument after op"),
        clEnumVal(
            InstrumentReportTimeZHigh, "instrument runtime reports time usage"),
        clEnumVal(InstrumentReportMemoryZHigh,
            "instrument runtime reports memory usage")),
    llvm::cl::cat(OMNNPAPassOptions));

class InstrumentZHighPass
    : public mlir::PassWrapper<InstrumentZHighPass, OperationPass<FuncOp>> {

private:
  bool allOpsAllowed;
  std::set<std::string> allowedOps;
  unsigned runtimeActions;

public:
  StringRef getArgument() const override { return "instrument-zhigh"; }

  StringRef getDescription() const override {
    return "instrument on zhigh ops.";
  }

  void init(std::string allowedOps_) {
    if (allowedOps_ == "ALL") {
      allOpsAllowed = true;
    } else {
      allOpsAllowed = false;
      std::stringstream ss(allowedOps_);
      std::istream_iterator<std::string> begin(ss);
      std::istream_iterator<std::string> end;
      allowedOps = std::set<std::string>(begin, end);
    }
    runtimeActions = InstrumentControlBits.getBits();
  };

  void runOnOperation() override {
    if (instrumentZHighOps == "" || instrumentZHighOps == "NONE")
      return;
    init(instrumentZHighOps);

    // Iterate on the operations nested in this function
    getOperation().walk([&](mlir::Operation *op) {
      if (isa<zhigh::ZHighDialect>(op->getDialect())) {
        // Skip the prefix "zhigh." of zhigh op name
        const char *opName = op->getName().getStringRef().data() + 6;
        if (!allOpsAllowed && allowedOps.find(opName) == allowedOps.end())
          return;

        Location loc = op->getLoc();
        OpBuilder opBuilder(op);
        if (InstrumentControlBits.isSet(InstrumentBeforeZHighOp)) {
          uint64_t tag = runtimeActions &
                         (~(1 << static_cast<int>(InstrumentAfterZHighOp)));
          opBuilder.create<mlir::KrnlInstrumentOp>(loc, op, tag);
        }

        // Can not insert after Op (e.g. ONNXReturnOP) with IsTerminator Trait
        if (InstrumentControlBits.isSet(InstrumentAfterZHighOp) &&
            !op->hasTrait<OpTrait::IsTerminator>()) {
          opBuilder.setInsertionPointAfter(op);
          uint64_t tag = runtimeActions &
                         (~(1 << static_cast<int>(InstrumentBeforeZHighOp)));
          opBuilder.create<mlir::KrnlInstrumentOp>(loc, op, tag);
        }
      }
    });
  }
};

/*!
 * Create an instrumentation pass.
 */
std::unique_ptr<mlir::Pass> createInstrumentZHighPass() {
  return std::make_unique<InstrumentZHighPass>();
}

} // namespace zhigh
} // namespace onnx_mlir
