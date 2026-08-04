/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- AppendInstrumentedOutputsPass.cpp - Instrumentation ---------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a pass that appends selected intermediate tensors,
// matched by onnx_node_name using the same "REGEX[:inN+outN]" syntax as
// --instrument-onnx-node, as extra outputs of each entry function. Unlike
// --instrument-onnx-node (which prints matched tensors as "==SIG-REPORT=="
// text at runtime), this lets tooling built around comparing model outputs
// (e.g. utils/RunONNXModel.py's --save-ref/--verify=ref) diff them directly,
// with no log parsing required.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/ONNX/Transforms/NodeNamePattern.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;
using onnx_mlir::NodeIOEntry;
using onnx_mlir::parseNodeNamePattern;

namespace onnx_mlir {

#define GEN_PASS_DEF_APPENDINSTRUMENTEDOUTPUTSPASS
#include "src/Dialect/ONNX/Transforms/Passes.h.inc"

} // namespace onnx_mlir

namespace {

/*!
 * This pass appends selected intermediate tensors as extra results of the
 * entry function(s), so they become real outputs of the compiled model.
 */

class AppendInstrumentedOutputsPass
    : public onnx_mlir::impl::AppendInstrumentedOutputsPassBase<
          AppendInstrumentedOutputsPass> {

public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AppendInstrumentedOutputsPass)

  AppendInstrumentedOutputsPass() = default;
  AppendInstrumentedOutputsPass(const AppendInstrumentedOutputsPass &pass)
      : onnx_mlir::impl::AppendInstrumentedOutputsPassBase<
            AppendInstrumentedOutputsPass>() {
    nodeNamePattern = pass.nodeNamePattern;
  }
  AppendInstrumentedOutputsPass(const std::string nodePattern)
      : nodeNamePattern(nodePattern) {}

private:
  std::string nodeNamePattern;

public:
  void runOnOperation() override {
    std::vector<NodeIOEntry> nodeNameEntries =
        parseNodeNamePattern(nodeNamePattern);
    if (nodeNameEntries.empty())
      return;

    ModuleOp moduleOp = getOperation();
    for (ONNXEntryPointOp entryPointOp : moduleOp.getOps<ONNXEntryPointOp>())
      processEntryPoint(moduleOp, entryPointOp, nodeNameEntries);
  }

private:
  // A value to add as a new output, together with the label ("in0"/"out1")
  // it was selected under, used to build a recognizable onnx.name.
  struct NewOutput {
    Value value;
    std::string label; // e.g. "/layer1/MatMul__out0"
  };

  void processEntryPoint(ModuleOp moduleOp, ONNXEntryPointOp entryPointOp,
      const std::vector<NodeIOEntry> &nodeNameEntries) {
    SymbolRefAttr funcRefAttr =
        entryPointOp.getOperation()->getAttrOfType<SymbolRefAttr>(
            ONNXEntryPointOp::getEntryPointFuncAttrName());
    if (!funcRefAttr)
      return;
    Operation *funcOp =
        moduleOp.lookupSymbol(funcRefAttr.getLeafReference().getValue());
    func::FuncOp mainFunc = mlir::dyn_cast_or_null<func::FuncOp>(funcOp);
    if (!mainFunc || mainFunc.getBody().empty())
      return;

    Block &entryBlock = mainFunc.getBody().back();
    auto returnOp = mlir::dyn_cast<func::ReturnOp>(entryBlock.getTerminator());
    if (!returnOp)
      return;

    // Values already returned (or newly selected in this same pass), used to
    // dedup by identity: don't add the same Value as an output twice, even
    // if it's matched under two different node names (possible after CSE).
    llvm::SmallPtrSet<Value, 8> existing(
        returnOp.getOperands().begin(), returnOp.getOperands().end());

    SmallVector<NewOutput, 4> newOutputs;
    auto tryAdd = [&](Value v, StringRef nodeName, StringRef label) {
      if (mlir::isa<NoneType>(v.getType())) {
        llvm::errs() << "Warning: --instrument-onnx-node-return match \""
                     << nodeName << "\" " << label
                     << " is absent (NoneType); skipping.\n";
        return;
      }
      if (!existing.insert(v).second)
        return; // already returned or already added this round.
      newOutputs.push_back(
          {v, ("__instrumented__" + nodeName + "__" + label).str()});
    };

    // Only scan the entry function's own top-level block: ops nested inside
    // Loop/If/Scan/Fused-op bodies live in separate regions, and returning
    // their values from here would violate SSA dominance. Such matches are
    // simply never found by a top-level-only scan, by construction, rather
    // than needing an explicit dominance check.
    for (Operation &op : entryBlock) {
      StringAttr onnxNodeName = op.getAttrOfType<StringAttr>("onnx_node_name");
      if (!onnxNodeName || onnxNodeName.getValue().empty())
        continue;
      std::string name = onnxNodeName.getValue().str();
      for (const NodeIOEntry &entry : nodeNameEntries) {
        if (!std::regex_match(name, entry.nameRegex))
          continue;
        if (!entry.hasIOFilter) {
          for (auto it : llvm::enumerate(op.getOperands()))
            tryAdd(it.value(), name, "in" + std::to_string(it.index()));
          for (auto it : llvm::enumerate(op.getResults()))
            tryAdd(it.value(), name, "out" + std::to_string(it.index()));
        } else {
          OperandRange operands = op.getOperands();
          for (int64_t idx : entry.inputIdx) {
            if (idx >= 0 && (size_t)idx < operands.size())
              tryAdd(operands[idx], name, "in" + std::to_string(idx));
            else
              llvm::errs()
                  << "Warning: --instrument-onnx-node-return selector in" << idx
                  << " out of range for node \"" << name << "\" ("
                  << operands.size() << " operand(s)); ignoring.\n";
          }
          ResultRange results = op.getResults();
          for (int64_t idx : entry.outputIdx) {
            if (idx >= 0 && (size_t)idx < results.size())
              tryAdd(results[idx], name, "out" + std::to_string(idx));
            else
              llvm::errs()
                  << "Warning: --instrument-onnx-node-return selector out"
                  << idx << " out of range for node \"" << name << "\" ("
                  << results.size() << " result(s)); ignoring.\n";
          }
        }
        // A node name is matched by at most one entry.
        break;
      }
    }

    if (newOutputs.empty())
      return;

    // Append the new values to the terminator in place.
    SmallVector<Value, 4> newValues;
    for (const NewOutput &out : newOutputs)
      newValues.push_back(out.value);
    returnOp->insertOperands(returnOp.getNumOperands(), newValues);

    // Grow the function type to match.
    SmallVector<Type, 8> newResultTypes(mainFunc.getResultTypes());
    for (Value v : newValues)
      newResultTypes.push_back(v.getType());
    FunctionType newFuncType = FunctionType::get(
        mainFunc.getContext(), mainFunc.getArgumentTypes(), newResultTypes);
    mainFunc.setFunctionType(newFuncType);

    // Grow res_attrs to match: copy the existing per-result attributes
    // (getAllResultAttrs pads with empty dicts up to the *old* result count
    // if none were set), then append one new dict per new output. This must
    // run after setFunctionType and use setAllResultAttrs (which replaces
    // the whole array in one shot) rather than setResultAttrs(newIndex, ...)
    // -- the latter indexes into the *old-sized* res_attrs array at the
    // *new* index and is out-of-bounds once any res_attrs already exist.
    SmallVector<DictionaryAttr, 8> resAttrs;
    mainFunc.getAllResultAttrs(resAttrs);
    OpBuilder b(mainFunc.getContext());
    for (const NewOutput &out : newOutputs)
      resAttrs.push_back(DictionaryAttr::get(mainFunc.getContext(),
          {b.getNamedAttr("onnx.name", b.getStringAttr(out.label))}));
    mainFunc.setAllResultAttrs(ArrayRef<DictionaryAttr>(resAttrs));
  }
};

} // end anonymous namespace

/*!
 * Create the append-instrumented-outputs pass.
 */
std::unique_ptr<mlir::Pass> onnx_mlir::createAppendInstrumentedOutputsPass(
    const std::string nodePattern) {
  return std::make_unique<AppendInstrumentedOutputsPass>(nodePattern);
}
