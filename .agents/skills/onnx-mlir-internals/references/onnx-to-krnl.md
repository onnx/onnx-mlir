<!-- Generated from the onnx-mlir knowledge base (interfaces/onnx-to-krnl.md), verified against onnx-mlir@b505d64b. Do not edit here: see .agents/skills/README.md. -->

# Interface: ONNX Dialect → Krnl Dialect (selected sections)

The `ConvertONNXToKrnlPass` lowers each ONNX op into Krnl loop constructs plus Affine/MemRef/Arith/Math ops. This is the largest single lowering step in onnx-mlir — every supported ONNX op must have a lowering pattern here.

## Lowerings That Expand Into Other ONNX Ops

### Attention

`onnx.Attention` is handled by `ONNXAttentionOpLowering` (`Math/Attention.cpp`, registered by `populateLoweringONNXAttentionOpPattern()`). It emits no Krnl itself: it calls `lowerONNXAttentionOp()`, which rewrites the op through `OnnxBuilder` into Reshape / Transpose / Concat / MatMul / Mul / Add / Softmax (plus Less / LessOrEqual / Sub / Where for synthesized masks), and those ops are lowered to Krnl by their own patterns in the same conversion. The decomposition lives in `src/Dialect/ONNX/Transforms/AttentionToONNXOps.{hpp,cpp}` (library `OMONNXRewrite`), a helper function, not a pass. The NNPA `AttentionForZHighPattern` in `RewriteONNXForZHigh.cpp` calls the same function on tensors before the NNPA lowering (see [stack/accelerators.md](../../../../.agents/skills/nnpa-internals/references/accelerators.md)).[^8]

`lowerONNXAttentionOp()` first rejects, with an op error, what neither path implements: `kv_num_heads` ≠ `q_num_heads`, or for 4-D inputs a K head count different from Q's (GQA/MQA); a non-zero `softcap`; `qk_matmul_output_mode` ≠ 0; and any `softmax_precision`. It then picks one of two paths:[^8]

- **Fixed-size KV cache**, chosen when `attn_mask`, `past_key` and `past_value` are all none and `nonpad_kv_seqlen` is given. K/V are the whole padded cache. The KV and query sequence lengths must be static. The path builds an additive padding mask (`Where(Less(kv_pos, nonpad_kv_seqlen[b]), 0, neg)`) and, when `is_causal` is set, adds a causal mask aligned to the end of each batch's valid prefix (`LessOrEqual(j − i, nonpad_kv_seqlen[b] − q_len)`). `present_key`/`present_value` are `none`.
- **Growing KV cache** (all other inputs). `past_key`/`past_value`, when present, are concatenated with K/V on axis 2 and returned as `present_key`/`present_value`; otherwise those outputs are `none`. A boolean `attn_mask` is converted to an additive bias with `Where(mask, 0, neg)`. A float mask must have Q's element type; other integer types are rejected. A mask whose static kv length is neither 1 nor the K/V sequence length is also rejected. When `is_causal` is set, a constant `(q_len, kv_len)` causal mask with offset = `past_key`'s sequence length is added on top. That requires static q, kv and past lengths.

Both paths share the rest of the computation:

1. A 3-D input `(B, S, hidden)` goes to 4-D by `Reshape` to `(B, S, heads, hidden/heads)` and then `Transpose [0,2,1,3]`, with `heads = q_num_heads` (default 1) for Q, K and V. The output goes back the same way in reverse. The pattern fails to match when a 3-D input's hidden size is dynamic.
2. `Y = MatMul(Softmax(MatMul(Q, Transpose(K, [0,1,3,2])) * scale + mask, axis=-1), V)`. The scale is the `scale` attribute, or ONNX's default `1/sqrt(head_size)` when it is absent. If `scale` is absent and `head_size` is dynamic, the pattern fails. The `Mul` is emitted only when the scale ≠ 1.0, as a constant of Q's element type.
3. The mask value is −1e9, or −1e4 for f16 so that summed masks stay finite. `qk_matmul_output` is always `none`.

The growing path emits the `past_key`/`past_value` concatenation, but the ONNX backend tests for that input pattern (`test_attention_*_with_past_and_present_*`) are disabled: they crash in a bug the test list calls not yet root-caused. The op's CPU limitations therefore list that pattern as unsupported. The `bf16` causal test is also disabled, because `MathBuilder::constant` has no bfloat16 case.[^8]

The compiler option `--kv-cache` (common to `onnx-mlir` and `onnx-mlir-opt`; values `Fixed`, `Grow`, default unset) overrides the operand-based choice. `Grow` forces the growing path. `Fixed` is an op error unless the operands already match the fixed pattern.[^8]

### QLinearMatMul

`ONNXQLinearMatMulOpLowering` still builds the zero-point subtraction, i32 `MatMul`, rescale by `a_scale·b_scale/y_scale` and round-to-even as ONNX ops through `OnnxBuilder`; those are lowered in turn. The output step is now a single Krnl loop over the broadcast of the i32 product and `y_zero_point`. For each element it adds the zero point (plus 128 for a `ui8` output, undoing `getOrCastToI8`'s re-centering), clips to `[0,255]` or `[-128,127]`, casts to the output type and stores. It uses `simdIterateIE` on the innermost dimension when `enableSIMD` is set and `hasManageableBroadcastForInnerDims()` holds, and a `noCollapse` parallel plan over the outer two levels when parallel is enabled for the op. `populateLoweringONNXQLinearMatMulOpPattern()` now takes `enableSIMD` and `enableParallel`.[^9]

## Parallel Region Planning and Loop Collapse

Every lowering site that may emit a parallel loop goes through `KrnlParallelPlan` (`KrnlParallelPlan.{hpp,cpp}`, included from `ONNXToKrnlCommon.hpp`). It replaced the free functions `tryCreateKrnlParallel()` and `findSuitableParallelDimension()` that used to be declared in `ONNXToKrnlCommon.hpp`. A site builds one plan, then calls one of two entry points:[^6]

- `tryCreateParallel(createKrnl, op, msg, lbs, ubs)` — decides and emits: a `krnl.collapse` when the decision is a group of levels, then the `krnl.parallel`. It returns the chosen level or `NO_PAR_FOUND` (−1). Because a collapse *substitutes* loop references, the following `krnl.iterate` must be given `plan.optimizedLoopDef()`, not the original `loopDef`; the plan's destructor asserts that a plan which collapsed was consumed.
- `findParallelDim(op, msg, lbs, ubs)` — asks only, emits nothing; always single-level. It is used by sites that parallelize through `forLoopIE`/`forLoopsIE`'s `useParallel` flag.

Plans are built three ways:

| Constructor | When to use | Collapse? |
|-------------|-------------|-----------|
| `KrnlParallelPlan(loopDef, enableCollapse, parFirst, parLast, collapseLast, cost, exclusiveDims)` | Plain nest of original loops where fusing adjacent levels is safe | Yes, when `enableCollapse` is set and the nest is not rank 0 |
| `KrnlParallelPlan::noCollapse(loopDef, parFirst, parLast, cost, exclusiveDims)` | Blocked/permuted refs, `iterArgs`, or a decision over a subset of the iterate's loops | Never |
| `KrnlParallelPlan::noLoopRefs(parFirst, parLast, cost, exclusiveDims)` | Sites with no `krnl.iterate` loop list (use with `findParallelDim`) | Never |

`KrnlParallelCost` carries the two numbers only the site knows: `minTripCountForParallel` (default 4) and `bodyCost`, the rough work per innermost iteration in work units (1 = a load+store copy, ~10 = a handful of ops, ~1000 = a transcendental or a tile). The header warns that a site whose innermost iteration hides a memcpy, an scf loop or a SIMD span must set `bodyCost`; under-declaring it changes which groups are candidates.[^6]

The decision (`decideKrnlParallel`): with no collapse claim or no run of two fusable levels (a fusable level lies in the collapse window and has a literal-zero lower bound), it is the old outermost-qualifying-level search. Otherwise a cost-model policy (`decideCollapseByCostModel`) prefers a single level whose trip count is a literal at or above the target under a statically small prefix; failing that, it grows candidate groups of adjacent fusable levels until their guaranteed trip count reaches the target while the index-recovery cost stays amortized, and ranks them. Trip counts are tracked as a product of literal factors plus a count of unknown ones, and an unknown factor is never counted toward a target. A candidate that leaves a dynamic level above the region (so the number of region entries is unknown) loses to one that absorbs it.[^6]

The cost-model constants come from `ParallelMachineSupport` (`src/Dialect/Mlir/ParallelMachineSupport.{hpp,cpp}`), a sibling of `VectorMachineSupport` set once in `configurePasses()` from `mtriple`/`march`/`mcpu`. Every target currently gets `GenericParallelMachineSupport`: `minParTripCountFloor = 0`, `minAmortWork = 16`, `maxForkCount = 2`. Each can be overridden with `--collapse-min-par-trip-count-floor`, `--collapse-min-amort-work`, `--collapse-max-fork-count` (common to `onnx-mlir` and `onnx-mlir-opt`; −1 = use the target value).[^7]

**Which sites can collapse.** `enableCollapse` is passed only to the populate functions of sites using the collapse-eligible constructor: Transpose, Gather, Concat, Slice, Expand, ScatterElements, ScatterND, and LayoutTransform (`populateLowering…OpPattern(…, enableParallel, enableCollapse)`). Every other parallel site uses `noCollapse`/`noLoopRefs`.[^6]

**Flags and defaults.** Collapse only ever happens together with parallelism.[^7]

- `onnx-mlir`: on by default under `--parallel`; turn it off with `--disable-collapse`. `addONNXToKrnlPasses()` passes `enableCollapse = enableParallel && !disableCollapse`; the NNPA accelerator's Krnl patterns get the same expression.
- `--convert-onnx-to-krnl` pass option `enable-collapse` (default `true`); the pass ANDs it with `enable-parallel`. `onnx-mlir-opt`'s registration of the pass also passes `enableCollapse = true` (with `enableParallel = false`).
- Example (from the lit test): `onnx-mlir-opt -O3 --convert-onnx-to-krnl=enable-parallel` lowers a `Slice` on `tensor<?x16x?x128xf32>` to `krnl.collapse(%l#0, %l#1)` + `krnl.parallel` over the fused reference; `"enable-parallel enable-collapse=false"` gives the single-level result.

[^8]: repo:onnx-mlir@b505d64:src/Conversion/ONNXToKrnl/Math/Attention.cpp (`ONNXAttentionOpLowering`), repo:onnx-mlir@b505d64:src/Dialect/ONNX/Transforms/AttentionToONNXOps.cpp (`lowerONNXAttentionOp`, `checkSupportedAttentionAttributes`, `lowerFixedSizeKVCacheAttention`, `lowerGrowingSizeKVCacheAttention`, `getEffectiveScale`, `negMaskValueFor`, `reshapeToMultiHead4D`), repo:onnx-mlir@b505d64:src/Dialect/ONNX/Transforms/CMakeLists.txt, repo:onnx-mlir@b505d64:src/Accelerators/NNPA/Conversion/ONNXToZHigh/RewriteONNXForZHigh.cpp (`AttentionForZHighPattern`), repo:onnx-mlir@b505d64:src/Compiler/CompilerOptions.cpp (`kv-cache`), repo:onnx-mlir@b505d64:test/mlir/conversion/onnx_to_krnl/Math/Attention_with_canonicalize.mlir, repo:onnx-mlir@b505d64:test/backend/inference_backend.py (`test_attention_*` entries and comments), repo:onnx-mlir@b505d64:docs/SupportedONNXOps-cpu.md (Attention row)
[^9]: repo:onnx-mlir@5534bc7:src/Conversion/ONNXToKrnl/Math/QLinearMatMul.cpp, repo:onnx-mlir@5534bc7:test/mlir/conversion/onnx_to_krnl/Math/QLinearMatMul_final.mlir
[^6]: repo:onnx-mlir@5534bc7:src/Conversion/ONNXToKrnl/KrnlParallelPlan.hpp (`KrnlParallelPlan`, `KrnlParallelCost`, `KrnlParallelDecision`, `NO_PAR_FOUND`), repo:onnx-mlir@5534bc7:src/Conversion/ONNXToKrnl/KrnlParallelPlan.cpp (algorithm banner, `decideKrnlParallel`, `decideCollapseByCostModel`), repo:onnx-mlir@b505d64:src/Conversion/ONNXToKrnl/ConvertONNXToKrnl.cpp (`populateONNXToKrnlConversionPattern`), repo:onnx-mlir@5534bc7:src/Conversion/ONNXToKrnl/Tensor/Slice.cpp (collapse-eligible plan), repo:onnx-mlir@5534bc7:test/mlir/conversion/onnx_to_krnl/parallel_collapse.mlir
[^7]: repo:onnx-mlir@5534bc7:src/Dialect/Mlir/ParallelMachineSupport.hpp (`ParallelTuning`, `GenericParallelMachineSupport`), repo:onnx-mlir@5534bc7:src/Dialect/Mlir/ParallelMachineSupport.cpp, repo:onnx-mlir@5534bc7:src/Compiler/CompilerOptions.cpp (`disable-collapse`, `collapse-min-par-trip-count-floor`, `collapse-min-amort-work`, `collapse-max-fork-count`), repo:onnx-mlir@5534bc7:src/Compiler/CompilerPasses.cpp (`configurePasses`, `addONNXToKrnlPasses`), repo:onnx-mlir@5534bc7:src/Conversion/ONNXToKrnl/ConvertONNXToKrnl.cpp (`enableCollapse` option), repo:onnx-mlir@5534bc7:src/Tools/onnx-mlir-opt/RegisterPasses.cpp, repo:onnx-mlir@5534bc7:src/Accelerators/NNPA/NNPAAccelerator.cpp
