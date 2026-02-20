#include "OnnxToMlirPasses.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "src/Compiler/DisposableGarbageCollector.hpp"
#include "src/Pass/Passes.hpp"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace mlir;
namespace onnx_mlir {

void addXmcMlirPasses(mlir::OpPassManager &pm, OnnxToMlirOptions opts) {
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createOptimizeOnnxRequantizationPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createQuantTypesPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createConvertInstanceNormToGroupNormPass());
  //  pm.addNestedPass<func::FuncOp>(onnx_mlir::createSplitGroupConvPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createRemoveDilationConv());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createTransferResizeLinearToDwConv());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createConvWithBiasPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createRemoveRedundantReshapePass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createLowerReduceToPoolPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createTransferPoolFixToDownsampleFixPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createTransferReduceMeanSumToConvPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createRemoveRedundantReluPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createStandardizeSliceOpsPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createMergeContinuousStridedSlicePass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createConvertMulToDepthwiseConv2dPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createTransferDepthwiseConv2dWithChannelMultiplierPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createRemoveUselessQLinearPoolPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createOptimizeSliceReshapeTransposeBlockPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createTransferSpaceToDepthToConv2dPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createMergeBatchnormToConvPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createEliminateReshapeAroundSlicePass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createMergeSliceConcatPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createMergeStridedSliceConcatConvPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createTransferConvSliceToConvPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createTransferOp1dToOp2dPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createTransferScaleToDwConv2dPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createConvertToChannelLastPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createConvertMatMulToXFEConvPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createONNXTransposeOptimizationPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createConstPropONNXToONNXPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createRemoveContinuousTransposeWithReshapePass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createTransferOp3dToOp2dPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createTransformReshapelikeOpToReshapePass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createRemoveSemanticallyUselessOpsPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createTransfer5dBlockTo4dPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createTransform5DTransposeTo4DPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createCombineTransposePairPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createReplaceNDimTransposePass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createTransfer5dStridedSliceTo4d());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createTransferOpShapeTo4dPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createBatchReductionToReshapeReductionPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createReplaceQDQEltwisePass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createReplaceHsigmoidAndHswishPass());
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createReplaceQDQSigmoidPass());
  pm.addNestedPass<func::FuncOp>(
      onnx_mlir::createConvertXFEConvToDepthwiseConvPass());
}

// ============================================================================
// XMC Pass Registry: a single source-of-truth list of passes and factories
// ============================================================================


/// A named pass entry: human-readable name + factory to create a fresh
/// instance.
struct XmcPassEntry {
  std::string name;
  std::function<std::unique_ptr<Pass>()> factory;
};

/// Return the ordered list of all XMC passes.
static std::vector<XmcPassEntry> getXmcPassEntries() {
  std::vector<XmcPassEntry> passes;

  auto add = [&](const char *name,
                 std::function<std::unique_ptr<Pass>()> factory) {
    passes.push_back({name, std::move(factory)});
  };

  // clang-format off
  add("optimize-onnx-requantization",                   onnx_mlir::createOptimizeOnnxRequantizationPass);
  add("quant-types",                                    onnx_mlir::createQuantTypesPass);
  add("convert-instancenorm-to-groupnorm",              onnx_mlir::createConvertInstanceNormToGroupNormPass);
  // add("split-group-conv",                            onnx_mlir::createSplitGroupConvPass);
  add("remove-dilation-conv",                           onnx_mlir::createRemoveDilationConv);
  add("transfer-resize-linear-to-dw-conv",              onnx_mlir::createTransferResizeLinearToDwConv);
  add("conv-with-bias",                                 onnx_mlir::createConvWithBiasPass);
  add("remove-redundant-reshape",                       onnx_mlir::createRemoveRedundantReshapePass);
  add("lower-reduce-to-pool",                           onnx_mlir::createLowerReduceToPoolPass);
  add("transfer-pool-fix-to-downsample-fix",            onnx_mlir::createTransferPoolFixToDownsampleFixPass);
  add("transfer-reducemeansum-to-conv",                 onnx_mlir::createTransferReduceMeanSumToConvPass);
  add("remove-redundant-relu",                          onnx_mlir::createRemoveRedundantReluPass);
  add("standardize-slice-ops",                          onnx_mlir::createStandardizeSliceOpsPass);
  add("merge-continuous-strided-slice",                  onnx_mlir::createMergeContinuousStridedSlicePass);
  add("convert-mul-to-depthwise-conv2d",                onnx_mlir::createConvertMulToDepthwiseConv2dPass);
  add("transfer-dw-conv2d-channel-multiplier",          onnx_mlir::createTransferDepthwiseConv2dWithChannelMultiplierPass);
  add("remove-useless-qlinear-pool",                    onnx_mlir::createRemoveUselessQLinearPoolPass);
  add("optimize-slice-reshape-transpose-block",         onnx_mlir::createOptimizeSliceReshapeTransposeBlockPass);
  add("transfer-space-to-depth-to-conv2d",              onnx_mlir::createTransferSpaceToDepthToConv2dPass);
  add("merge-batchnorm-to-conv",                        onnx_mlir::createMergeBatchnormToConvPass);
  add("eliminate-reshape-around-slice",                  onnx_mlir::createEliminateReshapeAroundSlicePass);
  add("merge-slice-concat",                             onnx_mlir::createMergeSliceConcatPass);
  add("merge-strided-slice-concat-conv",                onnx_mlir::createMergeStridedSliceConcatConvPass);
  add("transfer-conv-slice-to-conv",                    onnx_mlir::createTransferConvSliceToConvPass);
  add("transfer-op1d-to-op2d",                          onnx_mlir::createTransferOp1dToOp2dPass);
  add("transfer-scale-to-dw-conv2d",                    onnx_mlir::createTransferScaleToDwConv2dPass);
  add("convert-to-channel-last",                        onnx_mlir::createConvertToChannelLastPass);
  add("convert-matmul-to-xfe-conv",                     onnx_mlir::createConvertMatMulToXFEConvPass);
  add("onnx-transpose-optimization",                    onnx_mlir::createONNXTransposeOptimizationPass);
  add("constprop-onnx",                                 onnx_mlir::createConstPropONNXToONNXPass);
  add("remove-continuous-transpose-with-reshape",       onnx_mlir::createRemoveContinuousTransposeWithReshapePass);
  add("transfer-op3d-to-op2d",                          onnx_mlir::createTransferOp3dToOp2dPass);
  add("transform-reshapelike-to-reshape",               onnx_mlir::createTransformReshapelikeOpToReshapePass);
  add("remove-semantically-useless-ops",                onnx_mlir::createRemoveSemanticallyUselessOpsPass);
  add("transfer-5d-block-to-4d",                        onnx_mlir::createTransfer5dBlockTo4dPass);
  add("transform-5d-transpose-to-4d",                   onnx_mlir::createTransform5DTransposeTo4DPass);
  add("combine-transpose-pair",                         onnx_mlir::createCombineTransposePairPass);
  add("replace-ndim-transpose",                         onnx_mlir::createReplaceNDimTransposePass);
  add("transfer-5d-strided-slice-to-4d",                onnx_mlir::createTransfer5dStridedSliceTo4d);
  add("transfer-op-shape-to-4d",                        onnx_mlir::createTransferOpShapeTo4dPass);
  add("batch-reduction-to-reshape-reduction",           onnx_mlir::createBatchReductionToReshapeReductionPass);
  add("replace-adjacent-op",                            onnx_mlir::createReplaceAdjacentOpPass);
  add("replace-qdq-eltwise",                            onnx_mlir::createReplaceQDQEltwisePass);
  add("remove-pairs-and-move-down-reshape",             onnx_mlir::createRemovePairsAndMoveDownReshapePass);
  add("replace-contained-concat",                       onnx_mlir::createReplaceContainedConcatPass);
  add("optimize-sibling-concat",                        onnx_mlir::createOptimizeSiblingConcatPass);
  add("canonicalize",                                   []() -> std::unique_ptr<Pass> { return mlir::createCanonicalizerPass(); });
  add("replace-hsigmoid-hswish",                        onnx_mlir::createReplaceHsigmoidAndHswishPass);
  add("replace-qdq-sigmoid",                            onnx_mlir::createReplaceQDQSigmoidPass);
  add("convert-xfe-conv-to-depthwise-conv",             onnx_mlir::createConvertXFEConvToDepthwiseConvPass);
  // clang-format on

  return passes;
}

// ============================================================================
// MLIR dump helper
// ============================================================================

/// Dump module IR to a file named <outputDir>/xmc_pass_<index>_<name>.mlir.
static void dumpMLIR(mlir::ModuleOp module, const std::string &passName,
    size_t passIndex, const std::string &outputDir) {
  // Ensure the output directory exists.
  std::error_code mkdirEc = llvm::sys::fs::create_directories(outputDir);
  if (mkdirEc) {
    llvm::errs() << "  [XMC-DEBUG] Warning: could not create dir '" << outputDir
                 << "': " << mkdirEc.message() << "\n";
  }

  // Build the filename using llvm path utilities for cross-platform compat.
  llvm::SmallString<512> filepath(outputDir);
  std::string basename =
      "xmc_pass_" + std::to_string(passIndex) + "_" + passName + ".mlir";
  llvm::sys::path::append(filepath, basename);

  std::string filename = std::string(filepath);
  std::error_code ec;
  llvm::raw_fd_ostream file(filename, ec);
  if (!ec) {
    module.print(file);
    llvm::errs() << "  [XMC-DEBUG] Dumped IR to: " << filename << "\n";
  } else {
    llvm::errs() << "  [XMC-DEBUG] Failed to write " << filename << ": "
                 << ec.message() << "\n";
  }
}

// ============================================================================
// XMC Debug PassInstrumentation
// ============================================================================

/// PassInstrumentation that provides timing, IR change detection, and optional
/// MLIR dumping for XMC passes. This works transparently with any PassManager,
/// including those used by vaiml-lite-cli.
class XmcDebugInstrumentation : public mlir::PassInstrumentation {
  std::string outputDir;
  std::unordered_set<std::string> xmcPassNames;
  // Map from C++ pass name to human-readable CLI name for logging/filenames.
  std::unordered_map<std::string, std::string> passDisplayNames;
  size_t passIndex = 0;

  // Per-pass invocation state
  std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
  size_t hashBefore = 0;
  bool isTrackedPass = false;

  /// Compute a hash of the IR rooted at the given operation.
  static size_t computeHash(Operation *op) {
    std::string irStr;
    llvm::raw_string_ostream os(irStr);
    op->print(os);
    return std::hash<std::string>{}(irStr);
  }

  /// Walk up to the top-level ModuleOp from any operation.
  static mlir::ModuleOp getParentModule(Operation *op) {
    while (op->getParentOp())
      op = op->getParentOp();
    return mlir::dyn_cast<mlir::ModuleOp>(op);
  }

public:
  /// \param dir  Output directory for MLIR dumps (from opts.xmcOutputDir).
  ///             Callers set this via OnnxToMlirOptions before calling
  ///             addXmcMlirPasses / addONNXToMLIRPasses. Defaults to ".".
  XmcDebugInstrumentation(
      const std::string &dir, const std::vector<XmcPassEntry> &passes) {
    // Resolve "." to an absolute path for clear logging.
    llvm::SmallString<256> absDir;
    if (dir.empty() || dir == ".") {
      llvm::sys::fs::current_path(absDir);
    } else {
      absDir = dir;
      llvm::sys::fs::make_absolute(absDir);
    }
    outputDir = std::string(absDir);

    // Resolve actual pass names by creating temporary instances, because
    // pass->getName() returns the C++ class name (e.g. "OptimizeOnnx...Pass"),
    // not the CLI flag name (e.g. "optimize-onnx-requantization").
    for (auto &p : passes) {
      auto tmp = p.factory();
      std::string runtimeName = tmp->getName().str();
      xmcPassNames.insert(runtimeName);
      passDisplayNames[runtimeName] = p.name;
    }
    llvm::errs() << "[XMC-DEBUG] Instrumentation active for "
                 << xmcPassNames.size()
                 << " XMC passes, dump dir: " << outputDir << "\n";
  }

  void runBeforePass(Pass *pass, Operation *op) override {
    // Only instrument XMC passes (filter by name).
    StringRef passName = pass->getName();
    isTrackedPass = xmcPassNames.count(passName.str()) > 0;
    if (!isTrackedPass)
      return;

    startTime = std::chrono::high_resolution_clock::now();
    // Hash the module to detect IR changes.
    if (auto module = getParentModule(op))
      hashBefore = computeHash(module.getOperation());
    else
      hashBefore = computeHash(op);
  }

  void runAfterPass(Pass *pass, Operation *op) override {
    // Re-check pass name (isTrackedPass can be stale from nested passes
    // when OpToOpPassAdaptor wraps multiple nested XMC passes).
    std::string name = pass->getName().str();
    if (xmcPassNames.count(name) == 0)
      return;

    auto endTime = std::chrono::high_resolution_clock::now();
    auto durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        endTime - startTime)
                          .count();

    // Hash the module again to see if the pass modified IR.
    size_t hashAfter = 0;
    mlir::ModuleOp module = getParentModule(op);
    if (module)
      hashAfter = computeHash(module.getOperation());
    else
      hashAfter = computeHash(op);

    bool modified = (hashBefore != hashAfter);

    // Use the human-readable CLI name for logging and dump filenames.
    auto it = passDisplayNames.find(name);
    std::string displayName =
        (it != passDisplayNames.end()) ? it->second : name;

    llvm::errs() << "  [XMC-DEBUG] Pass [" << passIndex << "] '" << displayName
                 << "' completed in " << durationMs << " ms"
                 << (modified ? " ### Modified IR ###" : " ### No change ###")
                 << "\n";

    if (modified && module && !outputDir.empty())
      dumpMLIR(module, displayName, passIndex, outputDir);

    passIndex++;
  }

  void runAfterPassFailed(Pass *pass, Operation *op) override {
    // Re-check pass name (same stale-flag guard as runAfterPass).
    std::string name = pass->getName().str();
    if (xmcPassNames.count(name) == 0)
      return;

    auto it = passDisplayNames.find(name);
    std::string displayName =
        (it != passDisplayNames.end()) ? it->second : name;
    llvm::errs() << "  [XMC-DEBUG] Pass [" << passIndex << "] '" << displayName
                 << "' FAILED <<<\n";
    // Dump the IR at the point of failure for debugging.
    if (auto module = getParentModule(op)) {
      if (!outputDir.empty())
        dumpMLIR(module, displayName + "_FAILED", passIndex, outputDir);
    }
    passIndex++;
  }
};

// ============================================================================
// addXmcMlirPasses — always adds to PM, optionally adds instrumentation
// ============================================================================

void addXmcMlirPasses(mlir::PassManager &pm, OnnxToMlirOptions opts) {
  auto passes = getXmcPassEntries();

  // If debug mode is enabled, add PassInstrumentation for timing/hash/dump.
  if (opts.dumpMlirAfterEachXmcPass) {
    pm.addInstrumentation(
        std::make_unique<XmcDebugInstrumentation>(opts.xmcOutputDir, passes));
  }

  for (auto &entry : passes) {
    pm.addNestedPass<func::FuncOp>(entry.factory());
  }
}

// ============================================================================
// addONNXToMLIRPasses
// ============================================================================

void addONNXToMLIRPasses(mlir::PassManager &pm, bool targetCPU,
    bool donotScrubDisposableElementsAttr, OnnxToMlirOptions opts) {
  // This is a transition from previous static passes to full dynamic passes
  // Static passes are kept and the dynamic pass is added as IF-THEN
  // with the static iteration.
  // The reasons are
  // 1. The debug flag, --print-ir-after/befor-all, can display IR for each
  //    static pass, but the dynamic pipeline will be viewed as one. MLIR
  //    may have solution that I am not aware of yet.
  // 2. Easy to compare two approaches.
  // In future, only the dynamic pass, ONNXOpTransformPass, will be used for
  // this function.

  if (!donotScrubDisposableElementsAttr)
    pm.addInstrumentation(
        std::make_unique<DisposableGarbageCollector>(pm.getContext()));

  // Decompose first. Eliminates some unsupported ops without shape inference.
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createDecomposeONNXToONNXPass(
      /*target=*/"", opts.enableConvTransposeDecompose,
      opts.enableConvTransposeDecomposeToPhasedConv,
      opts.enableConvTranspose1dDecomposeToPhasedConv,
      opts.enableInstanceNormDecompose, opts.enableSplitToSliceDecompose));
  if (!opts.disableRecomposeOption)
    pm.addNestedPass<func::FuncOp>(
        onnx_mlir::createRecomposeONNXToONNXPass(/*target=*/""));

  if (opts.enableONNXHybridPass) {
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createONNXHybridTransformPass(
        !opts.disableRecomposeOption, opts.enableQuarkQuantizedLegalization,
        opts.enableConvTransposeDecompose,
        opts.enableConvTransposeDecomposeToPhasedConv,
        opts.enableConvTranspose1dDecomposeToPhasedConv,
        opts.enableInstanceNormDecompose, opts.enableSplitToSliceDecompose));
    // Convolution Optimization for CPU: enable when there are no accelerators.
    if (targetCPU && opts.enableConvOptPass) {
      pm.addNestedPass<func::FuncOp>(onnx_mlir::createConvOptONNXToONNXPass(
          opts.enableSimdDataLayout && !opts.disableSimdOption));
      pm.addNestedPass<func::FuncOp>(onnx_mlir::createONNXHybridTransformPass(
          !opts.disableRecomposeOption,
          /*enableQuarkQuantizedOpsLegalization=*/false,
          opts.enableConvTransposeDecompose,
          opts.enableConvTransposeDecomposeToPhasedConv,
          opts.enableConvTranspose1dDecomposeToPhasedConv,
          opts.enableInstanceNormDecompose, opts.enableSplitToSliceDecompose));
    }
    // If quark quantized legalization is enabled, do a last const prop after it
    // so that we cover any remaining Cast -> Cast patterns that weren't covered
    // by the pass.
    if (opts.enableQuarkQuantizedLegalization) {
      configureConstPropONNXToONNXPass(/*roundFPToInt=*/false,
          /*expansionBound=*/-1, /*disabledPatterns=*/{""},
          /*constantPropIsDisabled=*/false);
      pm.addNestedPass<func::FuncOp>(
          onnx_mlir::createConstPropONNXToONNXPass());
    }
  } else {
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
    // Convolution Optimization for CPU: enable when there are no accelerators.
    if (targetCPU && opts.enableConvOptPass) {
      pm.addNestedPass<func::FuncOp>(onnx_mlir::createConvOptONNXToONNXPass(
          opts.enableSimdDataLayout && !opts.disableSimdOption));
      pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
    }
    pm.addNestedPass<func::FuncOp>(
        onnx_mlir::createLegalizeQuarkQuantizedOpsPass());
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createConstPropONNXToONNXPass());
    if (opts.onnxOpTransformThreshold > 0) {
      // Dynamic iterate in ONNXOpTransformPass
      pm.addPass(onnx_mlir::createONNXOpTransformPass(
          opts.onnxOpTransformThreshold, opts.onnxOpTransformReport, targetCPU,
          opts.enableSimdDataLayout && !opts.disableSimdOption,
          opts.enableConvOptPass, !opts.disableRecomposeOption));
    } else {
      // Statically add extra passes
      for (int i = 0; i < opts.repeatOnnxTransform; i++) {
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
        pm.addNestedPass<func::FuncOp>(
            onnx_mlir::createConstPropONNXToONNXPass());
      }
    }
  }

  // Simplify shape-related ops.
  pm.addPass(onnx_mlir::createSimplifyShapeRelatedOpsPass(
      opts.enableQuarkQuantizedLegalization));

  // Canonicalizing Q-DQ related ops
  pm.addNestedPass<func::FuncOp>(onnx_mlir::createQDQCanonicalizePass(
      opts.enableRemoveBinary, opts.enableRemoveDqQAroundOp));

  // One more call to ONNX shape inference/canonicalization/... to update
  // shape if possible.
  if (opts.enableONNXHybridPass) {
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createONNXHybridTransformPass(
        !opts.disableRecomposeOption, opts.enableQuarkQuantizedLegalization,
        opts.enableConvTransposeDecompose,
        opts.enableConvTransposeDecomposeToPhasedConv,
        opts.enableConvTranspose1dDecomposeToPhasedConv,
        opts.enableInstanceNormDecompose, opts.enableSplitToSliceDecompose));
  } else {
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createShapeInferencePass());
  }

  // Replace ONNXReturnOp with func::ReturnOp.
  pm.addPass(onnx_mlir::createStandardFuncReturnPass());

  // Clean dead code.
  pm.addPass(mlir::createSymbolDCEPass());

  // Replace every DisposableElementsAttr with DenseElementsAttr.
  if (!donotScrubDisposableElementsAttr)
    pm.addPass(createScrubDisposablePass());

  // Set onnx_node_name if it is missing. Keep this pass at the end of this
  // function and just before instrumentation.
  pm.addPass(createSetONNXNodeNamePass());

  // Add instrumentation for Onnx Ops
  // Keep this pass at the end of this function.
  unsigned instrumentActions = opts.instrumentControlBits;
  if (opts.profileIR == onnx_mlir::ProfileIRs::Onnx) {
    opts.instrumentStage = onnx_mlir::InstrumentStages::Onnx;
    opts.instrumentOps = "onnx.*";
    instrumentActions |= (1 << 3) - 1;
  }
  if (opts.instrumentStage == onnx_mlir::InstrumentStages::Onnx)
    pm.addNestedPass<func::FuncOp>(
        onnx_mlir::createInstrumentPass(opts.instrumentOps, instrumentActions));
  if (opts.instrumentSignatures != "NONE" || opts.instrumentOnnxNode != "NONE")
    pm.addNestedPass<func::FuncOp>(onnx_mlir::createInstrumentONNXSignaturePass(
        opts.instrumentSignatures, opts.instrumentOnnxNode));

  // XMC passes: always added to the PM. Debug instrumentation (if enabled)
  // is attached inside addXmcMlirPasses via PassInstrumentation.
  if (opts.enableXMCPasses)
    addXmcMlirPasses(pm, opts);
}

} // namespace onnx_mlir