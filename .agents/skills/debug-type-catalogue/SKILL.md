---
name: debug-type-catalogue
description: 'Use when you want -debug-only=<type> output from a specific onnx-mlir pass and need the right DEBUG_TYPE string, or want to know what a given DEBUG_TYPE traces: a lookup table from every DEBUG_TYPE in src/ to its pass and output.'
---

<!-- Generated from the onnx-mlir knowledge base (skills/debug-type-catalogue.md), verified against onnx-mlir@b505d64b. Do not edit here: see .agents/skills/README.md. -->

# DEBUG_TYPE Catalogue: Pass-Level Debug Tracing

A lookup table from every `#define DEBUG_TYPE "..."` string in onnx-mlir's `src/` to the pass(es) it belongs to and what its `LLVM_DEBUG` output actually shows. Use this when you already suspect *which* pass is misbehaving and want its own internal reasoning, rather than reading a raw `--EmitONNXIR`/`--EmitMLIR` dump line by line.

## The two-tool combo

`-debug-only=<type>` and the pass-isolation workflow in [skills/debug-compilation.md](../../../.agents/skills/debug-compilation/SKILL.md) answer two different questions, and the fastest path to a diagnosis is usually both together:

1. **Find the `DEBUG_TYPE` for the suspect pass** in the table below.
2. **Run `onnx-mlir -debug-only=<type> ...`** to get that pass's own narration of *why* it made the decision it made (which pattern matched, which legality check failed, which branch it took).
3. **Separately**, run `utils/onnx-mlir-print.sh <args> /tmp/all-passes.log` and `utils/IsolatePass.py -i /tmp/all-passes.log -p <pass-name-regex>` to pull the *actual IR* immediately before and after that same pass, per [skills/debug-compilation.md](../../../.agents/skills/debug-compilation/SKILL.md#not-sure-which-pass-introduced-the-bug-dump-every-pass-then-isolate-one).

Step 2 tells you the pass's reasoning; step 3 shows you what it actually did to the IR. Neither alone is as fast as both together — the narration tells you where to look in the (much larger) isolated IR diff.

## Two things that will trip you up

- **Granularity is inconsistent by design.** Some `DEBUG_TYPE` strings are one-to-one with a single narrow pass or even a single op's lowering pattern (`rewrite-for-zhigh`, `decompose-onnx`, `dim_analysis`, `matmul`). Others are shared, by convention, across every file in an entire conversion stage — `-debug-only=krnl_to_llvm` prints trace output from **19 different files** covering the whole KrnlToLLVM conversion, not one pass. If you expect a narrow trace and get a firehose, check the "Scope" column below before assuming something's broken.
- **Two near-identical strings are actually different files.** `dialect-builder` (hyphen, `DialectBuilder.cpp`) and `dialect_builder` (underscore, `VectorMachineSupport.cpp`) are *not* the same debug channel despite both being "dialect builder"-shaped. If you're chasing a SIMD-architecture-selection question and only turn on `dialect-builder`, you'll see nothing — that trace lives under `dialect_builder`.

## Frontend / Import

| `DEBUG_TYPE` | File(s) | Scope | What you'll see |
|---|---|---|---|
| `onnx_mlir_main` | `src/onnx-mlir.cpp` | Top-level driver | Compiler entry-point trace: argument processing, overall pipeline stage transitions |
| `frontend_dialect_transformer` | `src/Builder/FrontendDialectTransformer.cpp` | .onnx protobuf → ONNX dialect import | Per-node import trace as the protobuf graph is walked and converted to ONNX dialect ops |

## ONNX Dialect Transforms (graph-level, before any lowering)

| `DEBUG_TYPE` | File(s) | Scope / pass name | What you'll see |
|---|---|---|---|
| `decompose` | `src/Dialect/ONNX/Transforms/Decompose.cpp` | `decompose-onnx` pass — includes the CPU-target Conv→Im2Col+MatMul decomposition (`ConvToIm2ColPattern`/`shouldDecomposeConvToIm2Col`, gated by `--disable-conv-to-matmul`), reused by the NNPA path's own `rewrite-for-zhigh` pass below for Conv ops that don't qualify for `zhigh.Conv2D` | Which decompose pattern matched a given op and why (e.g. `shouldDecomposeConvToIm2Col`'s verdict) |
| `recompose` | `src/Dialect/ONNX/Transforms/Recompose.cpp` | Recompose pass — inverse of some decompose patterns, folds a decomposed op sequence back into a single higher-level op | Which recompose pattern matched a given op sequence |
| `constprop-onnx` | `src/Dialect/ONNX/Transforms/ConstProp.cpp` | `--constprop-onnx` pass | Which constant-folding DRR pattern fired for a given op and the computed result |
| `rewrite` | `src/Dialect/ONNX/ONNXOps/Canonicalize.cpp` | Per-op canonicalization/folding patterns (`getCanonicalizationPatterns()`) registered directly on ONNX ops | Which canonicalization pattern matched during greedy rewriting |
| `ONNXOps` | `src/Dialect/ONNX/ONNXOps/Quantize/DequantizeLinear.cpp` | `DequantizeLinear`-specific shape-inference/verifier trace | Shape-inference reasoning specific to `DequantizeLinear` |
| `dim_analysis` | `src/Dialect/ONNX/ONNXDimAnalysis.cpp` | `DimAnalysis` — the dynamic-dimension-equality analysis, see [flows/dynamic-dimension-analysis.md](../../../docs/DynamicDimensionAnalysis.md) | Per-tensor dimension-equivalence exploration, e.g. `"Explore using consuming operators"`, `"Added a new dim(...)"` |
| `simplify_shape_related_ops` | `src/Dialect/ONNX/Transforms/SimplifyShapeRelatedOps.cpp` | Simplifies chains of shape-computation ops (`Shape`/`Slice`/`Concat`/... on 1-D int tensors) into simpler equivalent forms | Which shape-related simplification pattern matched |
| `set-onnx-node-name` | `src/Dialect/ONNX/Transforms/SetONNXNodeName.cpp` | `set-onnx-node-name` pass — assigns stable per-node names used elsewhere (e.g. JSON device-placement config matching, see [stack/accelerators.md](../../../.agents/skills/nnpa-internals/references/accelerators.md)) | Which name was assigned to which op |
| `onnx-cse-with-node-name` | `src/Transform/ONNXCSEWithNodeName.cpp` | CSE variant that additionally considers/merges node names | Which ops were merged and how their names were reconciled |

## ONNXToKrnl Lowering

| `DEBUG_TYPE` | File(s) | Scope | What you'll see |
|---|---|---|---|
| `lowering-to-krnl` | 11 files: `ONNXToKrnlCommon.cpp`, `KrnlParallelPlan.cpp`, `Math/{Elementwise,Reduction,Softmax}.cpp`, `NN/Normalization.cpp`, `Tensor/{Transpose,Concat,NonZero}.cpp`, plus **`Accelerators/NNPA/Conversion/ZHighToZLow/ProcessStickData{,Helper}.cpp`** | Stage-wide within `--convert-onnx-to-krnl` for the listed CPU op categories — **and** the NNPA stick/unstick SIMD codegen (see [concepts/stick-unstick-simd-codegen.md](../../../.agents/skills/nnpa-internals/references/stick-unstick-simd-codegen.md)), which reuses the same string despite being a different dialect conversion (`ZHighToZLow`, not `ONNXToKrnl`). Expect trace from both if both are compiled in | Per-op lowering strategy decisions (e.g. which broadcast/SIMD/parallel code path was chosen) for whichever of the listed op categories is present in the model |
| `matmul` | `src/Conversion/ONNXToKrnl/Math/MatMul.cpp` | `ONNXMatMulOp`'s own lowering pattern within `--convert-onnx-to-krnl` | Which matmul lowering strategy (blocked loops, SIMD, parallel) was chosen for the op's shape |
| `gemm` | `src/Conversion/ONNXToKrnl/Math/Gemm.cpp` | `ONNXGemmOp`'s own lowering pattern | Same, for Gemm |
| `perfect_hash` | `src/Conversion/ONNXToKrnl/PerfectHash.cpp` | Perfect-hash-table codegen helper, consumed by `CategoryMapper`'s lowering | Hash table construction/collision-resolution trace |
| `category_mapper_onnx_to_krnl` | `src/Conversion/ONNXToKrnl/ML/CategoryMapper.cpp` | `ONNXCategoryMapperOp` lowering pattern | Category-to-index mapping codegen trace |
| `reshape_onnx_to_krnl` | `src/Conversion/ONNXToKrnl/Tensor/Reshape.cpp` | `ONNXReshapeOp` lowering pattern | Reshape-specific memref/layout handling trace |
| `space_to_depth_onnx_to_krnl` | `src/Conversion/ONNXToKrnl/Tensor/SpaceToDepth.cpp` | `ONNXSpaceToDepthOp` lowering pattern | Index-remapping trace for the space→depth rearrangement |
| `depth_to_space_onnx_to_krnl` | `src/Conversion/ONNXToKrnl/Tensor/DepthToSpace.cpp` | `ONNXDepthToSpaceOp` lowering pattern | Index-remapping trace for the depth→space rearrangement |
| `gather_nd_onnx_to_krnl` | `src/Conversion/ONNXToKrnl/Tensor/GatherND.cpp` | `ONNXGatherNDOp` lowering pattern | Index-gathering loop construction trace |
| `layout-transform` | `src/Conversion/ONNXToKrnl/Additional/LayoutTransform.cpp` | `ONNXLayoutTransformOp` lowering pattern (generic memref layout conversion, distinct from NNPA's ZTensor layouts) | Source/target layout mapping trace |
| `onnx_to_stablehlo` | `src/Conversion/ONNXToStablehlo/Tensor/Concat.cpp` | `ONNXToStablehlo` — a separate conversion target from the Krnl path entirely, lowering ONNX directly to the StableHLO dialect. Not otherwise documented elsewhere in this wiki; flagged here as a gap, not a full write-up | Concat-specific lowering trace within the StableHLO conversion |

## KrnlToAffine

| `DEBUG_TYPE` | File(s) | Scope | What you'll see |
|---|---|---|---|
| `krnl_to_affine` | 9 files: `ConvertKrnlToAffine.cpp`, `Krnl{Load,Store,Prefetch,Terminator,GetLinearOffsetIndex,Matmul,CopyToBuffer,CopyFromBuffer}.cpp` | Stage-wide `--convert-krnl-to-affine` | Per-Krnl-op interpretation trace during lowering, e.g. `"Call interpretOperation"` (`ConvertKrnlToAffine.cpp:1032`) |

## Krnl/Affine/SCF Loop Transform Utility Passes

Standalone passes that run around the KrnlToAffine boundary, each a single narrow pass with its own `DEBUG_TYPE`:

| `DEBUG_TYPE` | File | What it does | What you'll see |
|---|---|---|---|
| `krnl-parallel-clause` | `src/Transform/ProcessKrnlParallelClause.cpp` | Folds a `KrnlParallelClauseOp`'s parameters into the enclosing OpenMP `parallel` construct | Which clause was merged into which parallel region |
| `buffer-omploop-hoisting` | `src/Transform/BufferOMPLoopHoist.cpp` | Hoists loop-invariant buffer allocations out of OpenMP loop bodies | Which allocation was hoisted and to where |
| `eliminate-write-only-alloc` (pass arg also `eliminate-write-only-alloc`) | `src/Transform/EliminateWriteOnlyAlloc.cpp` | Removes locally-allocated memrefs that are only ever written, never read — arises from fused-op lowering where shape tensors are computed but never loaded | Which allocation was proven dead and removed |
| `eliminate-entry-arg-clone` (pass arg also `eliminate-entry-arg-clone`) | `src/Transform/EliminateEntryArgClone.cpp` | `func::FuncOp` pass run right after MLIR's `buildBufferDeallocationPipeline` in `addKrnlToLLVMPasses()` (and in the Linalg path's `addLinalgToAffinePasses()`). In public (entry) functions it removes each `bufferization.clone` of a returned input argument, or a view of one, whose only use is `func.return`; the runtime owns entry inputs/outputs | `"Removing <clone op>"` for each clone erased |
| `scf-parallel-private` | `src/Transform/ProcessScfParallelPrivate.cpp` | Wraps `scf.parallel` bodies in `alloca_scope` so temporary buffers aren't shared across threads | Which parallel body was wrapped |
| `print-empty-instrumentation` | `src/Transform/InstrumentCleanupPass.cpp` | Removes back-to-back before/after instrumentation op pairs that measure nothing | Which instrumentation pair was elided |

## KrnlToLLVM

| `DEBUG_TYPE` | File(s) | Scope | What you'll see |
|---|---|---|---|
| `krnl_to_llvm` | 19 files: `ConvertKrnlToLLVM.cpp` plus every `Krnl*.cpp` runtime-support op lowering (`KrnlGlobal`, `KrnlCall`, `KrnlEntryPoint`, `KrnlPrint*`, `KrnlMemcpy`, `KrnlInstrument`, `KrnlUnaryMath`, `KrnlRandomNormal`, `KrnlRoundEven`, `KrnlStrlen`, `KrnlStrncmp`, `KrnlNone`, `KrnlVectorTypeCast`, ...), **and** the separate `KrnlSeqToMemref` conversion (`KrnlSeq{Store,Alloc,Dealloc,Extract}.cpp`) | Stage-wide across both `--convert-krnl-to-llvm` and the distinct `KrnlSeqToMemref` pass — two different conversions sharing one string, the widest-scope entry in this catalogue | Per-runtime-call lowering trace, plus a `"Compilation Info: <json>"` dump of the final compile configuration (`ConvertKrnlToLLVM.cpp:892`) |

## NNPA Accelerator Path

| `DEBUG_TYPE` | File | Pass | What you'll see |
|---|---|---|---|
| `onnx-to-zhigh` | `src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXToZHigh.cpp` | `ONNXToZHigh` conversion — the offload legality/lowering pass itself | Per-op legality-check trace and the combined-pattern (e.g. MatMul+Add→ZHighMatMul) matching decisions |
| `rewrite-for-zhigh` | `.../RewriteONNXForZHigh.cpp` | `rewrite-onnx-for-zhigh` pass — decides Conv Pad/1x1-Matmul/Im2Col+Matmul decomposition, broadcast-legalization rewrites, etc. | Per-op decision trace, e.g. for Conv: `canDecomposeToIm2ColAndMatmul`, `canDecomposeToPadAndConv`, `canDecompose1x1ToAMatmul`, `suitable`, `legal`, plus the op dump |
| `device-placement` | `.../DevicePlacement.cpp` | `DevicePlacement` pass — applies JSON `nnpa_ops_config` device/quantization rewrites (see [stack/accelerators.md](../../../docs/JsonConfigFile-NNPA.md)) | Which JSON pattern matched which op, and the resulting placement/quantization override |
| `device-placement-heuristic` | `.../DevicePlacementHeuristic.cpp` | Implements `--nnpa-placement-heuristic` (`QualifyingOps`/`FasterOps`/`FasterOpsWSU`/`MuchFasterOpsWSU`) | Per-op cost comparison and the resulting keep-on-NNPA-or-fall-back-to-CPU decision |
| `zhigh-perf-model` | `.../PerfModel.cpp` | Cost model backing the `FasterOps*` heuristics above | Estimated cost computation for a given op/shape, including matmul broadcast cost rules |
| `quant-op-selection` | `.../QuantOpSelection.cpp` | `QuantOpSelection` pass — marks ops for quantization per `nnpa_ops_config`/`--nnpa-quant-op-types` | Which op was selected/excluded for quantization and why |
| `generate-config-file` | `.../GenerateConfigFile.cpp` | Backs `--save-config-file` | Which nodes/shapes were captured into the generated JSON config |
| `expand-attention-mask` | `src/Dialect/ONNX/Transforms/ExpandAttentionMask.cpp` | `expand-attention-mask` pass (`func::FuncOp`), run once at the start of NNPA's `addONNXToZHighPasses()` unless `--nnpa-disable-expand-attention-mask`; also registered in `onnx-mlir-opt`. Lives under `Dialect/ONNX/Transforms/` but exists for NNPA. When the same broadcast attention mask, or the same scalar scale of an optional `Mul`, feeds at least `minUses` (4) `MatMul → [Mul] → Add → Softmax` chains with a matching `MatMul` output shape (checked with `DimAnalysis`), it materializes that operand once, so those `Add`s/`Mul`s no longer broadcast | Nothing at `b505d64`: the file defines the string but contains no `LLVM_DEBUG` call, so `-debug-only=expand-attention-mask` is silent. To see whether a mask or scale was expanded, diff `--mlir-print-ir-before/after=expand-attention-mask` for a new `onnx.Expand` feeding the attention `Add`s/`Mul`s |
| `zhigh-to-zlow` | `src/Accelerators/NNPA/Conversion/ZHighToZLow/ZHighToZLow.cpp` | `ZHighToZLow` conversion — includes the `convertZTensorToMemRefType` affine-mapping logic documented in [concepts/nnpa-ztensor-layouts.md](../../../.agents/skills/nnpa-internals/references/nnpa-ztensor-layouts.md) | Per-ZHigh-op lowering trace, including which `DataLayout` affine map was generated |
| `zlow-stick-expansion` | `src/Accelerators/NNPA/Transform/ZLow/ZLowStickExpansion.cpp` | `ZLowStickExpansion` — a ZLow-level pass expanding stick operations for codegen | Stick expansion decisions at the ZLow level |
| `op-fusion` | 4 files: `.../ZHighOps/ZHighFusionOpHelper.cpp`, `.../Transform/ZHigh/FusionOpStickUnstick.cpp`, **and the two generic ones, `src/Dialect/ONNX/Transforms/FusionOpHelper.cpp` and `src/Dialect/ONNX/Transforms/ONNXFusionOpHelper.cpp`** | NNPA's `FusionOpStickUnstick` (see [stack/accelerators.md](../../../.agents/skills/nnpa-internals/references/accelerators.md#op-fusion-before-lowering)), the CPU-side `FusionOpTransform` pass (see [stack/dialect-onnx.md](../../../.agents/skills/onnx-mlir-internals/references/dialect-onnx.md#--fusion-op-transform--the-cpu-side-formation-pass)), and the cross-target `FusionOpKindHelper` base both are built on — shared string, not NNPA-exclusive, and the one channel that spans both the accelerator and CPU fusion paths | Which chain shape matched: the NNPA kinds (`zhigh.extended_layout_transform` / `zhigh.expand-mul-stick` / `zhigh.concat-expand-stick` / `zhigh.unstick-split-heads` / `zhigh.mul-add-stick`) or the CPU kind (`simd-split-op-gather`) |
| `NNPAAccelerator` | `src/Accelerators/NNPA/NNPAAccelerator.cpp` | Top-level `NNPAAccelerator` registration (pass pipeline assembly, `conversionTargetONNXToKrnl`) | Which passes were registered/configured and the dynamic-legality decisions for fused ops |
| `NNPACompilerOptions` | `.../Compiler/NNPACompilerOptions.cpp` | NNPA `cl::opt` flag definitions and their direct validation | Flag-value validation trace at option-parse time |
| `NNPACompilerUtils` | `.../Compiler/NNPACompilerUtils.cpp` | `configurePassesNNPA` — assembles the NNPA pass pipeline, including the M14/saturation override described in [stack/accelerators.md](../../../.agents/skills/nnpa-internals/references/accelerators.md#nnpa-placement-and-tuning-flags) | Which passes were added to the pipeline and why (e.g. the saturation-override decision) |

## Support / Infrastructure (cross-cutting, no single pipeline stage)

| `DEBUG_TYPE` | File | Scope | What you'll see |
|---|---|---|---|
| `index-expr` | `src/Dialect/Mlir/IndexExpr.cpp`, `IndexExprDetail.cpp` | `IndexExpr` — the literal/affine/value index abstraction used throughout shape inference and lowering (see [concepts/index-expr.md](../../../.agents/skills/onnx-mlir-internals/references/index-expr.md)) | Internal state transitions of an `IndexExpr` as it's simplified/combined |
| `shape-helper` | `src/Dialect/ONNX/ONNXOps/ShapeHelper.cpp` | Generic `ONNXOpShapeHelper` machinery computing op output shapes via `IndexExpr` | Shape-computation trace for whichever op's shape helper is active |
| `dialect-builder` | `src/Dialect/Mlir/DialectBuilder.cpp` | `MultiDialectBuilder`/`MathBuilder`/`MemRefBuilder`/etc. helper construction (see [concepts/dialect-builders.md](../../../.agents/skills/onnx-mlir-internals/references/dialect-builders.md)) | Builder-level operation construction trace |
| `dialect_builder` (underscore — **not** the same channel as `dialect-builder` above) | `src/Dialect/Mlir/VectorMachineSupport.cpp`, `src/Dialect/Mlir/ParallelMachineSupport.cpp` | SIMD architecture/vector-length selection, and (`ParallelMachineSupport`) the target-derived parallel-region cost-model constants — minimum trip count, work per region, region-entry cost — set once by the driver | Which SIMD arch was selected, e.g. `"use SIMD arch <name>"` (`VectorMachineSupport.cpp:63`), and which parallel tuning was picked, e.g. `"use parallel tuning <arch> for triple ... minParTripCountFloor ..."` (`ParallelMachineSupport.cpp:50`) |
| `compiler_options` | `src/Compiler/CompilerOptions.cpp` | General (non-NNPA) `cl::opt` flag definitions and cross-flag validation | Flag-value validation trace at option-parse time |

## Test Infrastructure (outside `src/`, included for completeness)

| `DEBUG_TYPE` | File | Scope | What you'll see |
|---|---|---|---|
| `category-mapper-test` | `test/modellib/CategoryMapperModel.cpp` | Test-model-generation helper (`ModelLib`) that builds and verifies a `CategoryMapper` backend test model — not a compiler pass | Result-verification trace while checking the generated test model's output, e.g. `"Result Verification:"` |

**Not included, and not a real channel:** `my_opt_name_here` appears in `docs/Testing.md` inside a fenced code example teaching the `LLVM_DEBUG`/`DEBUG_TYPE` convention itself — it's a placeholder in a tutorial snippet, not code that compiles into the binary. Grepping the repo for `DEBUG_TYPE` will surface it; it deliberately has no row here.

## Verification

- **Grounded against:** `repo:onnx-mlir@b505d64:src/` (this page's stated scope; originally grounded at `c706d974`, fully re-swept at `b505d64`) plus a full-repo sweep (excluding `third_party/`) to confirm nothing outside `src/` was missed. Full repo-wide grep for `#define DEBUG_TYPE "..."` found **49 distinct strings across 90 defining lines in `src/`**, plus 2 more outside it (`category-mapper-test` in `test/`, `my_opt_name_here` in `docs/Testing.md` — the latter excluded as a non-functional tutorial placeholder, see above). Each file's header comment and a targeted `LLVM_DEBUG` snippet were read to confirm scope and output content.
- **Files read:** every file listed in the tables above (header comment + `getArgument()`/`LLVM_DEBUG` call sites where present); full list of matching files reproducible via `grep -rhoE '#define DEBUG_TYPE "[^"]+"' src/ | sort -u` (src/-scoped) or the same against `.` excluding `third_party/` (repo-wide).
- **How to test:** pick any row, compile a small model with `onnx-mlir --march=z16 --maccel=NNPA -debug-only=<type> model.onnx 2>&1 | less` (drop `--maccel`/`--march` for non-NNPA rows) on an assert/debug build, and confirm output matching the "What you'll see" description appears.
- **Known gap:** this catalogue is a point-in-time snapshot of a fact that changes whenever a file adds/removes/renames a `DEBUG_TYPE` — re-run the grep above on Repository Sync and diff before trusting it against a much later commit. This check is now an explicit step (5a) in `schema.md`'s Repository Sync workflow, added after being missed once (see below).

## Related Pages

- [skills/debug-compilation.md](../../../.agents/skills/debug-compilation/SKILL.md) — the `-debug-only` mechanism itself, plus `onnx-mlir-print.sh`/`IsolatePass.py` for IR-level isolation
- [concepts/nnpa-ztensor-layouts.md](../../../.agents/skills/nnpa-internals/references/nnpa-ztensor-layouts.md) — the layout logic traced by `zhigh-to-zlow`
- [stack/accelerators.md](../../../.agents/skills/nnpa-internals/references/accelerators.md) — the NNPA passes traced by `onnx-to-zhigh`, `rewrite-for-zhigh`, `device-placement`, `device-placement-heuristic`, `op-fusion`
- [flows/dynamic-dimension-analysis.md](../../../docs/DynamicDimensionAnalysis.md) — the analysis traced by `dim_analysis`
