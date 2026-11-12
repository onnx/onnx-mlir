---
name: add-fusion-pattern
description: 'Use when adding a new op-fusion kind wrapped in an onnx.Fused (ONNXFusedOp) container and lowered as one unit, on CPU or NNPA: the FusionOpKindHelper subclass, creation and lowering adapters, registration, phase ordering, and tests.'
---

<!-- Generated from the onnx-mlir knowledge base (skills/add-fusion-pattern.md), verified against onnx-mlir@b505d64b. Do not edit here: see .agents/skills/README.md. -->

# Add an ONNXFusedOp Fusion Pattern

Walkthrough for adding a new fusion *kind*: a short chain of ops that is matched once, wrapped in an `onnx.Fused` (`ONNXFusedOp`) container, and later lowered as one unit instead of op by op. It covers the reusable mechanism — the base class you subclass, the two adapters that plug it into a rewrite pass and a conversion pass, where each piece is registered on the CPU and NNPA paths, and how to test it — independent of any particular pattern.

onnx-mlir's own `docs/optimization-onnx-lowering/AddingAFusionPattern.md` is the upstream walkthrough, and `FusionOpPatterns.md` (same directory) catalogues the kinds that exist today. This page follows the code at the pinned SHA; where it says more than the upstream doc (the `kMaxOpCount` contract, the insertion-point check, multi-result lowering, phase ordering, the CPU path, NNPA legality), the code is what was checked.

---

## Parameters

| Parameter | Description | Example shape |
|-----------|-------------|---------------|
| `{{FUSION}}` | Your `FusionOpKindHelper` subclass | `MyChainFusionHelper` |
| `{{KIND}}` | The `kind` string stored on the fused op | `"my-chain"` (CPU) / `"zhigh.my-chain"` (NNPA) |
| `{{ANCHOR_OP}}` | The op type the creation pattern matches on | `ONNXConcatOp`, `ZHighUnstickOp`, ... |
| `{{LOWERING}}` | Your `FusedOpKindLowering<{{FUSION}}>` subclass | `ONNXFusedMyChainLowering` |

---

## Step 0: Is a fused op the right tool?

A fusion kind is for a chain that has **no single standard op equivalent** but can be lowered better jointly than op by op. If the chain is just a spelled-out form of an existing ONNX op, recompose it instead (`--recompose-onnx`, see [stack/dialect-onnx.md](../../../.agents/skills/onnx-mlir-internals/references/dialect-onnx.md#--recompose-onnx)). If one op alone can be lowered better, improve its lowering ([add-onnx-op-lowering.md](../../../.agents/skills/add-onnx-op-lowering/SKILL.md)). The fused-op container differs from both: the original ops survive inside an `IsolatedFromAbove` body, so if your dedicated lowering can't handle an instance, the body is inlined back and lowered normally (Step 6). See [stack/dialect-onnx.md](../../../.agents/skills/onnx-mlir-internals/references/dialect-onnx.md#onnxfusedop-internal-fusion-container) for the op itself.

---

## Step 1: Decide where it lives — CPU or accelerator

The framework is generic, but where you add a kind depends on which lowering consumes it.

| Piece | CPU (generic) kind | NNPA kind |
|---|---|---|
| Base class, creation adapter (don't edit) | `src/Dialect/ONNX/Transforms/FusionOpHelper.{hpp,cpp}`, `FusionOpBasePattern.hpp` | same |
| Lowering adapters (don't edit) | `src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.{hpp,cpp}` | same |
| Your subclass | `src/Dialect/ONNX/Transforms/ONNXFusionOpHelper.{hpp,cpp}` (the shared home for CPU kinds) | `src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/ZHighFusionOpHelper.{hpp,cpp}` |
| Creation registration | `populateONNXFusionOpPatterns()` in `src/Dialect/ONNX/Transforms/FusionOpTransform.cpp` | `FusionOpStickUnstick::runOnOperation()` in `src/Accelerators/NNPA/Transform/ZHigh/FusionOpStickUnstick.cpp` |
| Lowering | a new file under `src/Conversion/ONNXToKrnl/` (e.g. `Tensor/`), registered from `ConvertONNXToKrnl.cpp` | a struct in `src/Accelerators/NNPA/Conversion/ZHighToZLow/ZHighToZLow.cpp`, registered in `populateZHighToZLowConversionPattern()` |
| Kind string | no prefix convention | **must** start with `zhigh.` (see Step 6) |

You only write a CPU kind once: on NNPA builds, `FusionOpStickUnstick` also calls `populateONNXFusionOpPatterns()`, and `CompilerPasses.cpp` schedules the CPU pass `--fusion-op-transform` only when `targetCPU && !disableFusedOp`. So a CPU kind is formed on both builds, but only one fusion pass ever runs. Both passes run late in their pipelines, so the fused op isn't disturbed by later graph optimizations.

---

## Step 2: Declare the subclass

```cpp
class {{FUSION}} : public onnx_mlir::FusionOpKindHelper {
public:
  static constexpr llvm::StringLiteral kKind{{{KIND}}};
  // Longest chain this kind can match, every optional op included.
  static constexpr int kMaxOpCount = /* N */;

  // Parameters found during detection and needed by the lowering.
  int64_t someAxis = -1;
  bool someFlag = false;

  bool detectIfBeneficial(const DimAnalysis *dimAnalysis, {{ANCHOR_OP}} startOp);

  llvm::StringRef getKind() const override { return kKind; }
  void embedAttrs(mlir::ONNXFusedOp fusedOp) const override;
  bool retrieveAttrs(mlir::ONNXFusedOp fusedOp) override;
  bool verify() const override;
};
```

The contract has three parts:

- **Four pure virtuals**: `getKind`, `embedAttrs`, `retrieveAttrs`, `verify`.
- **Two non-virtual members the base can't declare**: `detectIfBeneficial` (its anchor type differs per kind) and `static constexpr int kMaxOpCount`. Both are needed where you instantiate `FusedPatternForOpKind<{{ANCHOR_OP}}, {{FUSION}}>`, and leaving either out fails to compile *there*, not in the header. `kKind` is needed the same way by `FusedOpKindLowering`.
- **One optional hook**: `absorbShapeConcatOfDims()` (default `false`). Override it to return `true` when the chain consumes a small i64 shape `Concat` of constants and `onnx.Dim` ops that is defined too late in the block to be an input. The `Concat` is then cloned into the body, and the `Dim` results become inputs.

The base's `ops` and `finalResults` fields are `protected`. Fill them from your own methods. External code never touches them.

---

## Step 3: Implement `detectIfBeneficial`

This is the only place where you decide whether a fusion is legal. Follow these rules:

1. **First line, always:** `if (isInsideFusedOp(startOp)) return false;`. Fusion moves the matched ops into the body rather than deleting them, so without this guard the pattern matches its own output again and the greedy driver never converges.
2. **Fill `ops` in block (topological) order.** It is usually a straight line, but a DAG is fine (for example, two producers feeding a join op). `ops.back()` is the op whose result becomes the fused op's output.
3. **Fill `finalResults`**: one value per fused-op result, in result order.
4. **Prove every dynamic-dim equality the lowering relies on, using `DimAnalysis`** (`sameDim` / `sameShape`). Examples: no runtime broadcast, or a dim kept through a `Reshape`. A `?` on both sides is not a proof. `dimAnalysis` is never null here (`FusedPatternForOpKind` asserts it).
5. **End with a profitability check.** Return `true` only when fusing is worth it, not just when it's legal.
6. **Reject extra uses** of the chain's intermediate values. Today's kinds require a single use. The helper `uniqueUserOfOpType<T>()` in `ZHighFusionOpHelper.cpp` is an example. It is file-`static`, so to reuse it, add your kind to that file or copy it.

Returning `true` doesn't guarantee a fusion. Right after detection, `FusedPatternForOpKind::matchAndRewrite` calls the base's `computeInputsAndInsertionPoint()`. This is a read-only check that looks for one insertion point that comes after every external input and before every outside use of the yielded values. If none exists, the match is declined. That happens when a chain value used outside the chain feeds back, directly or indirectly, into one of the chain's own inputs. You don't call it yourself. Just don't assume every detected chain gets fused.

---

## Step 4: Implement `embedAttrs` / `retrieveAttrs` / `verify`

- **`embedAttrs`** writes each parameter field as a named attribute on the fused op (`fusedOp->setAttr("axis", b.getI64IntegerAttr(axis))`). It is the *only* method that writes attributes.
- **`retrieveAttrs`** reads them back with `getAttrOfType<...>` and returns `false` if a required one is missing. It is the *only* method that reads attributes. Keeping both directions in these two methods makes a kind's attribute set easy to audit.
- **`verify`** re-checks `ops`, which the base rebuilds from the body, against the retrieved fields: op count, op types, and the parameters each op should still have. It runs at lowering time **without** `DimAnalysis`. Its job is to catch a body that was changed after fusion, not to re-prove legality. It may compare static shapes only and accept `?` vs `?`, trusting the proof from Step 3. If a predicate is shared between detection and `verify`, give it a `const DimAnalysis *`: non-null from detection means precise, null from `verify` means static-only. Document the null behavior on that predicate.

A corollary: a hand-written `onnx.Fused` in a lit test never went through detection, so nothing proved its dynamic dims equal. The test input has to make them equal itself.

For a reference implementation, see `SplitOpGatherFusionHelper::embedAttrs` / `retrieveAttrs` / `verify` in `ONNXFusionOpHelper.cpp`.

---

## Step 5: Register the creation pattern

```cpp
using FusedPatternsFor{{FUSION}} = FusedPatternForOpKind<{{ANCHOR_OP}}, {{FUSION}}>;
patterns.insert<FusedPatternsFor{{FUSION}}>(context, dimAnalysis);
```

- **CPU kind:** add the `insert` to `populateONNXFusionOpPatterns()`. Both passes pick it up (Step 1).
- **NNPA kind:** add it to the main `RewritePatternSet` in `FusionOpStickUnstick::runOnOperation()`, inside the `if (!disableFusedOpOption && !disableFusedOp)` block next to the existing kinds.

**Precedence between kinds:**

- *Same anchor op type:* the pattern's benefit is `kMaxOpCount`, so the kind with the longer chain is tried first. Set it to the real maximum `ops.size()`.
- *Different anchor types, where one chain can contain the other:* benefit doesn't help, because MLIR only ranks patterns that share an anchor op. Run the longer kind's pattern in its **own, earlier** `applyPatternsGreedily` call. `FusionOpStickUnstick` already does this, running `FusedPatternsForConcatExpandStick` to a fixpoint before the main pattern set (see the comment there).

**Disable switch:** `--disable-fused-op` (`disableFusedOp`, `src/Compiler/CompilerOptions.cpp`, default `false`) turns off fused-op formation in both passes. When it's off, NNPA falls back to its hand-written composite-op patterns. If your kind needs its own escape hatch, add one, and test both settings.

---

## Step 6: Implement the lowering

```cpp
struct {{LOWERING}} : public FusedOpKindLowering<{{FUSION}}> {
  using Base = FusedOpKindLowering<{{FUSION}}>;
  // Constructor takes (TypeConverter &, MLIRContext *, <your flags>) and
  // forwards the first two to Base.

  FailureOr<SmallVector<Value>> lowerVerified(ONNXFusedOp fusedOp,
      OpAdaptor adaptor, ConversionPatternRewriter &rewriter,
      {{FUSION}} &fusion) const override {
    // fusion's parameter fields are populated; adaptor.getInputs() are the
    // converted inputs. Emit code; return one Value per fused-op result,
    // in fusedOp.getOutputs() order.
  }
};
```

The base's `matchAndRewrite` (`final`) does everything except your code generation:

1. It returns `failure()` if `fusedOp.getKind() != kKind`, so other kinds' patterns get a chance.
2. It calls `retrieveOpsAndOutputValues` and `verifyAndRetrieveAttrs`. **On failure it unfuses**: the body is inlined back, and the original ops lower individually in the same pass.
3. It calls your `lowerVerified`. If that returns `failure()`, the pattern simply fails.
4. It drops the body's internal def-use edges and calls `replaceOp`. Don't call `replaceOp` yourself.

**Registration:**

- **CPU:** put the struct in a new `src/Conversion/ONNXToKrnl/...` file. Add a `populateLowering...Pattern(...)` function, declare it in `ONNXToKrnlCommon.hpp`, call it from `populateONNXToKrnlConversionPattern()` in `ConvertONNXToKrnl.cpp`, and add the file to `src/Conversion/ONNXToKrnl/CMakeLists.txt`. `Tensor/FusedSplitOpGather.cpp` follows exactly this layout.
- **NNPA:** add a `patterns.insert<...>` in `populateZHighToZLowConversionPattern()` in `ZHighToZLow.cpp`, under the "FusedOp: one pattern per zhigh.* kind" comment. `NNPAAccelerator::conversionTargetONNXToKrnl()` marks every `onnx.Fused` whose `kind` starts with `zhigh.` as **illegal**, so it must be converted, and leaves other kinds to the normal rules. That is why NNPA kinds need the prefix.

**Fallback:** `FusedOpInlineFallback` is registered once in `ConvertONNXToKrnl.cpp` at benefit 0. A kind with no lowering yet still compiles correctly: the fallback emits the warning *"no dedicated lowering for onnx.Fused (kind='…'); inlining body as fallback"* and inlines the body. So you can land creation and lowering separately. It also means your lowering must have benefit above 0. The default `OpConversionPattern` benefit is fine.

---

## Step 7: Test

Existing tests to copy, by role:

| What | Example RUN line | Examples |
|---|---|---|
| Creation, CPU | `onnx-mlir-opt --fusion-op-transform %s -split-input-file \| FileCheck %s` | `test/mlir/onnx/onnx_fusion_op_transform.mlir` |
| Creation, NNPA | `onnx-mlir-opt -O3 --march=z16 --maccel=NNPA --fusion-op-stick-unstick %s -split-input-file \| FileCheck %s` | `test/mlir/accelerators/nnpa/transform/zhigh-fused-*.mlir` |
| Lowering, CPU | `onnx-mlir-opt --shape-inference --convert-onnx-to-krnl ... %s` | `test/mlir/conversion/onnx_to_krnl/Tensor/FusedSplitOpGather*.mlir` |
| Lowering, NNPA | `onnx-mlir-opt --march=z16 --maccel=NNPA --convert-onnx-to-krnl --canonicalize %s` | `test/mlir/accelerators/nnpa/conversion/onnx-to-krnl/*-stick*.mlir` |
| Unfuse / fallback | same as lowering, on a body that `verify()` must reject | `FusedSplitOpGatherFallback.mlir`, `concat-expand-stick-unfuse.mlir` |

At minimum, cover:

- **Detection, positive and negative:** the chain is present and profitable; present but not profitable; broken by an extra use; and anything a profitability rule or dynamic dim is supposed to reject. Include **dynamic-dim** inputs: matching dims that `DimAnalysis` can prove equal, and `?` dims it can't.
- **Lowering of a well-formed fused op.**
- **`verify()` rejection:** hand-edit a fused body (drop or change one op) and confirm it unfuses and lowers op by op. It must not crash or produce wrong code.
- **`--disable-fused-op`** (and any flag of your own) in both settings.

Before freezing output into CHECK lines, show it is numerically correct. [ground-lit-tests.md](../../../.agents/skills/ground-lit-tests/SKILL.md) is the workflow, and its minimal-RUN-line rule applies here. In particular, the end-to-end result must match what the unfused chain computes.

---

## Step 8: Document the kind

Add it to `docs/optimization-onnx-lowering/FusionOpPatterns.md`:
- a row in the Summary table (kind, anchor, short chain, inner dim, typical source) linking to a new section;
- the section itself, under §1 (ONNX/CPU) or §2 (ZHigh/NNPA), giving the chain, the idiom it targets, the anchor, the main matching conditions, and what the lowering emits;
- your tests in its Tests section.

Then run [coding-conventions.md](../../../.agents/skills/coding-conventions/SKILL.md) on the new files.

---

## Debugging a kind that doesn't fire (or fires wrongly)

- **Trace:** all framework and per-kind files use `DEBUG_TYPE "op-fusion"`. Use `--debug-only=op-fusion` in a debug build (see [debug-type-catalogue.md](../../../.agents/skills/debug-type-catalogue/SKILL.md)). Use the same channel name in your own file.
- **No fused op formed:** check, in order: the `isInsideFusedOp` guard; detection's single-use and dim checks; `computeInputsAndInsertionPoint()` declining; another kind with a different anchor consuming the ops first (Step 5); `--disable-fused-op`; and, for CPU kinds, whether `--fusion-op-transform` ran at all (it doesn't under an accelerator, where the accelerator's pass forms the CPU kinds instead).
- **Formed but lowered op by op:** look for the `FusedOpInlineFallback` warning (lowering not registered in the pass that saw it), or `verify()` rejecting the body because something changed it between fusion and lowering. `op-fusion` debug output from `verifyAndRetrieveAttrs` says which.
- **Correctness:** compile with and without `--disable-fused-op` and compare outputs ([debug-compilation.md](../../../.agents/skills/debug-compilation/SKILL.md)).
- **Dims lost through the fused-op boundary:** `DimAnalysis` traces through block arguments via `resolveThroughFusedOp()`. See [flows/dynamic-dimension-analysis.md](../../../docs/DynamicDimensionAnalysis.md).

---

## Common pitfalls

- **Missing `isInsideFusedOp` guard:** the rewrite loops forever.
- **`kMaxOpCount` set lower than the real maximum chain:** a shorter kind on the same anchor wins and the longer fusion is lost.
- **Relying on benefit to order kinds with different anchors:** it has no effect. Use a separate, earlier greedy phase.
- **`ops` out of block order:** the insertion point and the back-to-front erase both rely on it, and a still-used op can be erased.
- **Attribute access outside `embedAttrs` / `retrieveAttrs`:** breaks the auditable attribute set.
- **Proving dim equality in `verify()` instead of detection:** `verify` has no `DimAnalysis`, so "accept `?` vs `?`" there is only safe because detection already proved it.
- **NNPA kind without the `zhigh.` prefix:** it isn't forced illegal by `NNPAAccelerator`'s legality callback, so it falls under the generic ONNXToKrnl rules instead.
- **Lowering registered at benefit ≤ 0:** it can lose to `FusedOpInlineFallback`, and the kind always inlines.

---

## Verification

- **Grounded against:** `repo:onnx-mlir@b505d64` (first grounded at `5534bc7`; re-checked at this sync: none of the fusion framework, NNPA fusion, `FusedSplitOpGather` registration or fallback files changed in a way this page depends on; `FusedSplitOpGather.cpp`'s parallel loop now goes through a `noCollapse` `KrnlParallelPlan`; `FusionOpPatterns.md` gained a note on the PyTorch `unbind` form that `recompose-onnx` rewrites into the split chain)
- **Files read:** `src/Dialect/ONNX/Transforms/FusionOpHelper.hpp` (contract, `kMaxOpCount`, `absorbShapeConcatOfDims`, `computeInputsAndInsertionPoint`), `FusionOpHelper.cpp` (`DEBUG_TYPE`, absorbable shape-`Concat`), `FusionOpBasePattern.hpp` (`FusedPatternForOpKind`, benefit = `kMaxOpCount`, `DimAnalysis` assert), `FusionOpTransform.{hpp,cpp}` (`populateONNXFusionOpPatterns`, pass body), `ONNXFusionOpHelper.{hpp,cpp}` (`SplitOpGatherFusionHelper`), `src/Dialect/ONNX/Transforms/CMakeLists.txt` (`OMFusionOpHelper`, `OMONNXFusionOpHelper`, `OMFusionOpTransform`), `Passes.td` (`FusionOpTransformPass`); `src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.{hpp,cpp}` (`FusedOpKindLowering`, `FusedOpInlineFallback` benefit 0 and warning), `ConvertONNXToKrnl.cpp` (fallback and `populateLoweringONNXFusedSplitOpGatherOpPattern` registration), `Tensor/FusedSplitOpGather.cpp`, `src/Conversion/ONNXToKrnl/CMakeLists.txt`; `src/Compiler/CompilerPasses.cpp` (`targetCPU && !disableFusedOp`), `src/Compiler/CompilerOptions.cpp` (`--disable-fused-op`); `src/Accelerators/NNPA/Transform/ZHigh/FusionOpStickUnstick.cpp` (phase ordering, registration, CPU patterns merged in), `src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/ZHighFusionOpHelper.{hpp,cpp}` (`uniqueUserOfOpType`, `kKind` values), `src/Accelerators/NNPA/Conversion/ZHighToZLow/ZHighToZLow.cpp` (`populateZHighToZLowConversionPattern`), `src/Accelerators/NNPA/NNPAAccelerator.cpp` (`zhigh.` legality callback), `src/Accelerators/NNPA/Compiler/NNPACompilerUtils.cpp`; the RUN lines of the tests in Step 7; `docs/optimization-onnx-lowering/{AddingAFusionPattern,FusionOpPatterns}.md`
- **How to test:** after Steps 2–6, the creation lit test should show one `onnx.Fused` with `kind = {{KIND}}` and your attributes. The lowering test should show your code with no `onnx.Fused` left and no fallback warning. The tampered-body test should show the original ops lowered individually.
