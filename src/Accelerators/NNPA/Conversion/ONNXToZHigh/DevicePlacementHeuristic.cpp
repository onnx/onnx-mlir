/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- DevicePlacementHeuristic.hpp - Place ops using model  -------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains heuristics to place operations on CPU or NNPA.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/DevicePlacementHeuristic.hpp"
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXToZHighCommon.hpp"
#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/PerfModel.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

#include <cmath>
#include <functional>

#define DEBUG_TYPE "device-placement-heuristic"
#define DEBUG 2

using namespace mlir;
using namespace onnx_mlir;

namespace {

//===----------------------------------------------------------------------===//
// Support to classify ops.

bool isMappedToDevice(Operation *op) {
  StringAttr device = op->getAttrOfType<mlir::StringAttr>(DEVICE_ATTRIBUTE);
  return device && !device.getValue().empty();
}

bool isMappedToCPU(Operation *op) {
  StringAttr device = op->getAttrOfType<mlir::StringAttr>(DEVICE_ATTRIBUTE);
  return device && device.getValue().equals_insensitive(CPU_DEVICE);
}

bool isMappedToNNPA(Operation *op) {
  StringAttr device = op->getAttrOfType<mlir::StringAttr>(DEVICE_ATTRIBUTE);
  return device && device.getValue().equals_insensitive(NNPA_DEVICE);
}

// Determine if op is unsuitable because its not an ONNX op of interest, or it
// is already mapped to the CPU device.
bool isNNPAFriendlyOp(Operation *op) {
  if (op->getDialect()->getNamespace() != ONNXDialect::getDialectNamespace())
    return false;
  // These ops are NNPA unfriendly. Constants are friendly.
  if (isa<ONNXEntryPointOp, ONNXReturnOp>(op))
    return false;
  // If `device` is already set to CPU, it is NNPA unfriendly
  if (isMappedToCPU(op))
    return false;
  return true;
}

//===----------------------------------------------------------------------===//
// Support functions op assignment.

// Return true with a debug message reporting reason for success on NNPA.
inline bool fasterOnNNPA(Operation *op, bool significant = false) {
  LLVM_DEBUG({
    if (significant)
      llvm::dbgs() << "  Significantly faster ";
    else
      llvm::dbgs() << "  Faster ";
    llvm::dbgs() << "on NNPA model for op:";
    op->dump();
  });
  return true;
}

// Return false with a debug message reporting reason for failure on NNPA.
inline bool fasterOnCPU(Operation *op, bool significant = false) {
  LLVM_DEBUG({
    if (significant)
      llvm::dbgs() << "  Significantly faster ";
    else
      llvm::dbgs() << "  Faster ";
    llvm::dbgs() << "on CPU model for op:";
    op->dump();
  });
  return false;
}

inline void assignToNNPA(Operation *op, MLIRContext *context) {
  LLVM_DEBUG({
    llvm::dbgs() << "Assign to NNPA:";
    op->dump();
  });
  op->setAttr(DEVICE_ATTRIBUTE, StringAttr::get(context, NNPA_DEVICE));
}

inline void assignToCPU(Operation *op, MLIRContext *context) {
  LLVM_DEBUG({
    llvm::dbgs() << "Assign to CPU:";
    op->dump();
  });
  op->setAttr(DEVICE_ATTRIBUTE, StringAttr::get(context, CPU_DEVICE));
}

//===----------------------------------------------------------------------===//
// Support functions simple cost model analysis, based solely on one operation.

// Simply determine if operation is faster on CPU or NNPA.
bool isOpFasterOnNNPA(Operation *op, const DimAnalysis *dimAnalysis) {
  LLVM_DEBUG({
    llvm::dbgs() << "\nTest cost-benefit of CPU/NNPA for op\n";
    op->dump();
  });
  // Estimate time
  double cpuTime, nnpaTime;
  if (!estimateTimeForOpWithModel(op, dimAnalysis, cpuTime, nnpaTime)) {
    // No performance model for this operation, assume faster on NNPA;
    cpuTime = 1;
    nnpaTime = 0;
  }
  if (nnpaTime < cpuTime)
    return fasterOnNNPA(op);
  return fasterOnCPU(op);
}

//===----------------------------------------------------------------------===//
// Support functions cost/benefit operation that takes stick/unstick into
// account.

struct DevicePlacementWithStickUnstickCost {
  DevicePlacementWithStickUnstickCost() = delete;
  DevicePlacementWithStickUnstickCost(MLIRContext *context, ModuleOp module,
      const DimAnalysis *dimAnalysis, const OpSetType &cpuOps)
      : context(context), dimAnalysis(dimAnalysis), cpuOps(cpuOps) {
    characterizeOps(module);
  }

  // Data
  MLIRContext *context;
  const DimAnalysis *dimAnalysis;
  // All ops that must execute on CPU, aka not eligible to run on  NNPA. Ops
  // in this set can be marked as device=CPU.
  const OpSetType &cpuOps;
  // All ops that may execute on NNPA. Ops in this set can be marked as
  // device=CPU or NNPA.
  OpSetType nnpaCandidateOps;
  // All ops that run on CPU but do not require stick/unstick at runtime. Ops in
  // thi set can be marked as device=CPU.
  OpSetType nnpaNeutralOps;

  void characterizeOps(ModuleOp module) {
    nnpaCandidateOps.clear();
    nnpaNeutralOps.clear();
    module.walk([&](Operation *op) -> WalkResult {
      // Skip ops that are NNPA unfriendly such as ops already assigned to CPU.
      if (!isNNPAFriendlyOp(op))
        return WalkResult::advance();
      // Ops that cannot/may not go on NNPA but can operate on NNPA data "for
      // free" are included here in NNPA neutral ops.
      // I assume here (not really true) that transpose and reshape can carry
      // the stickified data.
      if (isa<ONNXConstantOp, ONNXTransposeOp, ONNXReshapeOp>(op)) {
        nnpaNeutralOps.insert(op);
        return WalkResult::advance();
      }
      // Skip ops that the compiler determined are not suitable for NNPA.
      if (cpuOps.contains(op))
        return WalkResult::advance();
      // Remaining ops can be mapped to NNPA.
      nnpaCandidateOps.insert(op);
      return WalkResult::advance();
    });
#if DEBUG >= 2
    LLVM_DEBUG({
      llvm::dbgs() << "\nCPU Ops:\n";
      for (auto op : cpuOps) {
        if (isa<ONNXConstantOp, func::FuncOp>(op))
          continue;
        llvm::dbgs() << "cpu ";
        op->dump();
      }
      llvm::dbgs() << "\nNNPA Neutral Ops:\n";
      for (auto op : nnpaNeutralOps) {
        if (isa<ONNXConstantOp, func::FuncOp>(op))
          continue;
        llvm::dbgs() << "neutral ";
        op->dump();
      }
      llvm::dbgs() << "\nNNPA Candidate Ops:\n";
      for (auto op : nnpaCandidateOps) {
        llvm::dbgs() << "candidate ";
        op->dump();
      }
    });
#endif
  }

  void classifyValueUsage(Value value, Operation *opToSkip, int64_t &cpuOpCount,
      int64_t &nnpaOpCount, int64_t &nnpaCandidateOpCount,
      int64_t &nnpaNeutralOpCount) {
    cpuOpCount = nnpaOpCount = nnpaCandidateOpCount = nnpaNeutralOpCount = 0;

    std::string msg = "";
    for (Operation *userOp : value.getUsers()) {
      // Skip op if requested.
      if (userOp == opToSkip) {
        LLVM_DEBUG(msg = " Skipped op.");
        // Test ops that are already mapped.
      } else if (isMappedToCPU(userOp))
        cpuOpCount++;
      else if (isMappedToNNPA(userOp))
        nnpaOpCount++;
      // Not mapped, test now ops that are candidate to execute on NNPA.
      else if (nnpaCandidateOps.contains(userOp))
        nnpaCandidateOpCount++;
      // Not candidate, test now ops that are neutral to NNPA.
      else if (nnpaNeutralOps.contains(userOp))
        nnpaNeutralOpCount++;
      // None of the above, will be on CPU.
      else
        cpuOpCount++;
    }
    LLVM_DEBUG({
      llvm::dbgs() << "    Use pattern for value from "
                   << value.getDefiningOp()->getName() << ": used by CPU "
                   << cpuOpCount << ", NNPA " << nnpaOpCount
                   << ", NNPA candidates " << nnpaCandidateOpCount
                   << ", neutral " << nnpaNeutralOpCount << "." << msg << "\n";
    });
  }

  // Cost benefit analysis of moving this op X to the NNPA, with respect the ops
  // that are using the results of op X. Positive cost are additional cost to
  // have op X on NNPA, negative costs are benefits to have op X on NNPA.
  double costBenefitIncurredForResults(Operation *opX) {
    assert(!isMappedToDevice(opX) && "cannot evaluate an already mapped op");
    double totalCostBenefit = 0;
    LLVM_DEBUG(llvm::dbgs() << "  Look at cost benefit for results:\n");
    for (Value resVal : opX->getResults()) {
      // Look at all the users of currRes and classify them.
      int64_t cpuOpCount, nnpaOpCount, nnpaCandidateOpCount, nnpaNeutralOpCount;
      classifyValueUsage(resVal, /*skip op*/ nullptr, cpuOpCount, nnpaOpCount,
          nnpaCandidateOpCount, nnpaNeutralOpCount);
      /*
        Case study:
        1)  Op X remains on CPU  | 2) Op X migrates to NNPA:
                   X.CPU         |          X.NNPA
                /    |    \      |      /     |      \
               /   stick? stick  | unstick unstick?   \
              /      |       \   |    /       |        \
            CPU  Candidate  NNPA |  CPU   Candidate    NNPA
                 on NNPA         |        on CPU
        placing X on NNPA:       |
            cost:                | +1 unstick if has CPU users
            benefit:             | -1 stick if has NNPA users

        TODO: If migrate X to NNPA, could attribute some benefits for having
        users that are NNPA.
      */
      double costOfUnstickOp = estimateTimeForUnstickOp(resVal);
      double costOfStickOp = estimateTimeForStickOp(resVal);
      if (cpuOpCount > 0) {
        // Moving this op to NNPA will cost one unstick as there are one or
        // more ops that must execute on CPU.
        LLVM_DEBUG(
            llvm::dbgs() << "      +1 unstick: " << costOfUnstickOp << "\n");
        totalCostBenefit += costOfUnstickOp;
      }
      if (nnpaOpCount > 0) {
        // Moving this op to NNPA will remove the need to stick this result
        LLVM_DEBUG(
            llvm::dbgs() << "      -1 stick: " << -costOfStickOp << "\n");
        totalCostBenefit -= costOfStickOp;
      }
    }
    return totalCostBenefit;
  }

  // Cost benefit analysis of moving this op X to the NNPA, with respect the ops
  // that define the inputs of op X. Positive cost are additional cost to
  // have op X on NNPA, negative costs are benefits to have op X on NNPA.
  double costBenefitIncurredForInputs(Operation *opX) {
    assert(!isMappedToDevice(opX) && "cannot evaluate an already mapped op");
    double totalCostBenefit = 0;
    LLVM_DEBUG(llvm::dbgs() << "  Look at cost benefit for inputs:\n");
    OpSetType visitedDefiningOps;
    for (Value inputVal : opX->getOperands()) {
      // Investigate the operation that defines inputVal (which is used by op)
      Operation *definingOp = inputVal.getDefiningOp();
      if (!definingOp)
        continue;
      // If we have AddOp(%3, %3), should visit cost associated with %3 input
      // only once.
      if (visitedDefiningOps.contains(definingOp)) {
        LLVM_DEBUG(llvm::dbgs() << "    has multiple use of same input\n");
        continue;
      }
      visitedDefiningOps.insert(definingOp);

      // Classify all other users of this input value.
      int64_t cpuOpCount, nnpaOpCount, nnpaCandidateOpCount, nnpaNeutralOpCount;
      classifyValueUsage(inputVal, /*skip op X that we are analyzing*/ opX,
          cpuOpCount, nnpaOpCount, nnpaCandidateOpCount, nnpaNeutralOpCount);
      /*
        Case study:
        3) Op X remains on CPU           | 4) Op X remains on CPU
                  def.CPU ----.          |        def.NNPA -----.
                /    |    \     \        |      /     |    \     \
               /   stick? stick  \       | unstick unstick? \   unstick
              /      |       \    \      |    /       |      \     \
            CPU  Candidate  NNPA  X.CPU  |  CPU  Candidate  NNPA  X.CPU
                 on NNPA                 |       on CPU

        5) Op X migrates to NNPA         | 6) Op X migrates to NNPA
                  def.CPU ----.          |        def.NNPA -----.
                /    |    \     \        |      /     |    \     \
               /   stick? stick stick    | unstick unstick? \     \
              /      |       \    \      |    /       |      \     \
            CPU  Candidate  NNPA  X.NNPA |  CPU  Candidate  NNPA  X.NNPA
                 on NNPA                 |       on CPU

        placing X on NNPA:               |
            cost: +1 stick if first NNPA |
            benefit:                     | -1 stick
      */
      double costOfStickOp = estimateTimeForStickOp(inputVal);
      if (isMappedToCPU(definingOp) ||
          !(nnpaCandidateOps.contains(definingOp) ||
              nnpaNeutralOps.contains(definingOp))) {
        // Case 5.
        if (nnpaOpCount == 0) {
          LLVM_DEBUG(llvm::dbgs() << "      def-op on cpu (case 5), +1 stick "
                                  << costOfStickOp << ".\n");
          totalCostBenefit += costOfStickOp;
        }
      }
      if (isMappedToNNPA(definingOp)) {
        // Case 6.
        LLVM_DEBUG(llvm::dbgs() << "      def-op on NNPA (case 6), -1 stick "
                                << -costOfStickOp << ".\n");
        totalCostBenefit -= costOfStickOp;
      }
    }
    return totalCostBenefit;
  }

  bool significantlyFaster(double fast, double slow, double factor) {
    // At least factor x faster.
    return factor * fast <= slow;
  }

  // Determine if op is faster on the NNPA or not. To be faster than the CPU,
  // expect the NNPA to be at least minFactor faster than CPU. Significant is
  // set if the op is significantFactor faster / slower on the device.
  bool isOpFasterOnNNPA(Operation *op, double minFactor,
      double significantCPUFactor, double significantNNPAFactor,
      bool &significant) {
    LLVM_DEBUG({
      llvm::dbgs()
          << "\nTest cost-benefit with stick/unstick of CPU/NNPA for op\n";
      op->dump();
    });
    // Estimate time
    double cpuTime, nnpaTime, nnpaTimeWithOverheads;
    if (estimateTimeForOpWithModel(op, dimAnalysis, cpuTime, nnpaTime)) {
      // Has performance model, account for stick/unstick.
      double useCostBenefit = costBenefitIncurredForResults(op);
      double inputCostBenefit = costBenefitIncurredForInputs(op);
      nnpaTimeWithOverheads = nnpaTime + useCostBenefit + inputCostBenefit;
      LLVM_DEBUG(llvm::dbgs()
                 << "  New estimated nnpa time with stick/unstick:"
                 << nnpaTimeWithOverheads << " vs cpu " << cpuTime << ".\n");
    } else {
      // No performance model for this operation, assume faster on NNPA;
      cpuTime = 10;
      nnpaTime = nnpaTimeWithOverheads = 1;
      LLVM_DEBUG(llvm::dbgs() << "    no time estimate, assume NNPA better\n.");
    }
    if (nnpaTimeWithOverheads * minFactor <= cpuTime) {
      // For significant, don't take overheads into account as it may change
      // depending on mapping.
      significant =
          significantlyFaster(nnpaTime, cpuTime, significantNNPAFactor);
      return fasterOnNNPA(op, significant);
    }
    // For significant, don't take overheads into account as it may change
    // depending on mapping.
    significant = significantlyFaster(cpuTime, nnpaTime, significantCPUFactor);
    return fasterOnCPU(op, significant);
  }

}; // DevicePlacementWithStickUnstickCost

} // namespace

//===----------------------------------------------------------------------===//
// Exported heuristics for device placement.

namespace onnx_mlir {

void PlaceAllLegalOpsOnNNPA(MLIRContext *context,
    const SmallVector<Operation *, 32> &ops, const OpSetType &cpuOps) {
  for (Operation *op : ops) {
    if (isMappedToDevice(op))
      continue;
    // Op that cannot go on NNPA.
    if (cpuOps.contains(op))
      continue;
    // Compiler determined that we want this op on the NNPA, mark as such.
    assignToNNPA(op, context);
  }
}

void PlaceBeneficialOpsOnNNPA(MLIRContext *context,
    const SmallVector<Operation *, 32> &ops, const DimAnalysis *dimAnalysis,
    const OpSetType &cpuOps) {
  for (Operation *op : ops) {
    if (isMappedToDevice(op))
      continue;
    // Op that cannot go on NNPA.
    if (cpuOps.contains(op))
      continue;
    // Now we have an operation that can work on the NNPA, check if its
    // beneficial
    if (!isOpFasterOnNNPA(op, dimAnalysis)) {
      assignToCPU(op, context);
      continue;
    }
    // Compiler determined that we want this op on the NNPA, mark as such.
    assignToNNPA(op, context);
  }
}

void PlaceBeneficialOpsOnNNPAWithStickUnstick(MLIRContext *context,
    ModuleOp module, const SmallVector<Operation *, 32> &ops,
    const DimAnalysis *dimAnalysis, const OpSetType &cpuOps, double minFactor,
    double significantCPUFactor, double significantNNPAFactor) {
  // Init model.
  DevicePlacementWithStickUnstickCost model(
      context, module, dimAnalysis, cpuOps);
  int64_t ub = 5;
  int64_t i = 0;
  while (i < ub) {
    int64_t modified = 0;
    bool first = (i == 0);
    bool last = (i == ub - 1);
    LLVM_DEBUG(llvm::dbgs() << "\n\n\nPlacement Iteration " << i << "\n\n");
    for (Operation *op : ops) {
      if (isMappedToDevice(op))
        continue;
      // Op that cannot go on NNPA.
      if (cpuOps.contains(op))
        continue;
      // Now we have an operation that can work on the NNPA, check if its
      // beneficial
      bool significant;
      if (!model.isOpFasterOnNNPA(op, minFactor, significantCPUFactor,
              significantNNPAFactor, significant)) {
        if (last || significant) {
          modified++;
          assignToCPU(op, context);
        }
        continue;
      }
      // Compiler determined that we want this op on the NNPA, mark as such.
      if (!first || significant) {
        modified++;
        assignToNNPA(op, context);
      }
    }
    if (last) {
      break;
    } else if (first) {
      LLVM_DEBUG(llvm::dbgs() << "\nFirst, go on.\n");
      ++i;
    } else if (modified) {
      LLVM_DEBUG(llvm::dbgs() << "\nHad " << modified << " changes, go on.\n");
      ++i;
    } else {
      LLVM_DEBUG(llvm::dbgs() << "\nHad no changes, skip to last iter.\n");
      i = ub - 1;
    }
  }
}

} // namespace onnx_mlir
