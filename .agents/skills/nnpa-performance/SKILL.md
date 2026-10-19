---
name: nnpa-performance
description: 'Use when a model is slow or not offloading on IBM Z NNPA (--maccel=NNPA): confirm the accelerator is active, see what offloaded and why ops stayed on CPU, check op limits, handle dynamic shapes, quantize, and tune placement and fusion.'
---

<!-- Generated from the onnx-mlir knowledge base (skills/nnpa-performance.md), verified against onnx-mlir@b505d64b. Do not edit here: see .agents/skills/README.md. -->

# Get Performance on NNPA

Ordered walkthrough for making a model run fast on IBM's NNPA accelerator. Getting an op *onto* NNPA is the first lever and the one that fails most often, so steps 1–5 cover offload; steps 6–8 cover the throughput levers that only matter once your ops are actually on the accelerator. Every check up to step 8 is compile-time — none of them need a run on real z hardware.

---

## Step 1: Confirm the accelerator is actually active

Do this before investigating any individual op, because getting it wrong rejects the *whole model* rather than one op, and it does so silently:

```bash
onnx-mlir -O3 --march=z16 --maccel=NNPA mymodel.onnx      # NNPA generation M14
onnx-mlir -O3 --march=z17 --maccel=NNPA mymodel.onnx      # NNPA generation M15
```

- `--maccel=NNPA` selects the accelerator path; `--march` selects the *generation*. Only `z16`/`arch14` (M14) and `z17`/`arch15` (M15) map to an NNPA level.
- **`--maccel=NNPA` with `--march` omitted, or set to z14/z15, is not an error.** The level resolves to none, every op fails its level check, and the entire model compiles to CPU with no diagnostic. See [stack/compiler.md](../../../.agents/skills/onnx-mlir-internals/references/compiler.md#target-selection-which---march-values-get-simd-and-an-accelerator) for what each value gets you.
- NNPA flags exist in `--help` at all only in a build configured with `-DONNX_MLIR_ACCELERATORS=NNPA`. If `--maccel=NNPA` is rejected outright, the accelerator isn't compiled in — see [stack/accelerators.md](../../../.agents/skills/nnpa-internals/references/accelerators.md#build-time-gate).

## Step 2: See what offloaded

Every NNPA compile already prints a fallback summary — nothing to turn on:

```
[Warning] There are 2 onnx.Conv, 1 onnx.MatMul operations that run on CPU (not accelerated by NNPA). To get more information, recompile the model with the --onnx-op-stats option.
```

Two limits on what that tells you: it counts only nine *heavy* op types (`Conv`, `Gemm`, `GRU`, `LSTM`, `MatMul`, `MatMulInteger`, `QLinearMatMul`, `RNN`, `Softmax`), and it does not say why. For the full picture — ops that offloaded appear as `zhigh.*`, ops that didn't stay `onnx.*`:

```bash
onnx-mlir -O3 --march=z16 --maccel=NNPA --onnx-op-stats TXT mymodel.onnx
onnx-mlir -O3 --march=z16 --maccel=NNPA --EmitZHighIR mymodel.onnx    # writes .onnx.mlir
```

## Step 3: Ask why an op stayed on CPU

`--opt-report=NNPAUnsupportedOps` is the flag built for this question, and the one the `[Warning]` line above does not mention:

```bash
onnx-mlir -O3 --march=z16 --maccel=NNPA --opt-report=NNPAUnsupportedOps mymodel.onnx \
  | grep NNPA-UNSUPPORTEDOPS-REPORT
```

One line per rejected op, four comma-separated fields:

```
==NNPA-UNSUPPORTEDOPS-REPORT==, onnx.MaxPoolSingleOut, resnetv18_pool0_fwd, ceil_mode not supported
==NNPA-UNSUPPORTEDOPS-REPORT==, onnx.Gemm, fc_1, `alpha` (2.000000) must be 1.0
```

The third field is **the node name from your own model** (the `onnx_node_name` attribute), so it is searchable in Netron or against your graph — that is what turns "a Gemm was rejected" into "*this* Gemm was rejected." Many reasons also name the flag that relaxes them, so read the message text rather than just the category.

### Two preconditions, one of which is a trap

- **`--maccel=NNPA` is required**, and the `NNPAUnsupportedOps` value only exists in an NNPA-enabled build. Otherwise `--opt-report` offers only `Parallel` and `Simd`.
- **The report is silent below `--EmitMLIR`.** The reasons come from the ONNX-to-ZHigh lowering pass, which only runs at emission targets from `--EmitMLIR` up. `--EmitONNXIR` gives you *no output at all* — not "no rejections found," but nothing, which looks identical. The default target (`--EmitLib`), `--EmitMLIR`, `--EmitLLVMIR` and `--EmitZHighIR` all work. `--onnx-op-stats` has the same requirement and says so in its help text; this flag does not.

Not a problem, in case you go looking: device placement runs the same legality checks earlier with reporting deliberately switched off, and the lowering pass switches it back on — so a JSON device-placement config does not cost you the report, and you don't get duplicate lines. A silent report is always the emission-target trap above.

## Step 4: Check the op's documented limits before changing anything

There is a per-op table with a **Limitations** column and a **Minimum NNPA Level**, which is the authoritative answer to "can this op offload at all, and under what conditions":

- **[repos/onnx-mlir/supported-ops-nnpa.md](../../../docs/SupportedONNXOps-NNPA.md) — start here.** This wiki carries the full table, all ops and every limitation, at a pinned commit. It also explains something the generated table doesn't: a limitation reading "must be static" is a compile-time *verifiability* requirement of onnx-mlir's own legality check, not a hardware restriction — zDNN only ever sees concrete numbers at run time — which is precisely why step 5's escape hatch exists.
- [docs/SupportedONNXOps-NNPA.md](https://github.com/onnx/onnx-mlir/blob/main/docs/SupportedONNXOps-NNPA.md) and [docs/SupportedOps-NNPA-supplement.md](https://github.com/onnx/onnx-mlir/blob/main/docs/SupportedOps-NNPA-supplement.md) — upstream on `main`, generated from the source. Reach for these when you need something newer than the pin above, but note the direction of the mismatch: `main` can be *ahead* of the onnx-mlir you actually have installed, so an op listed there may not offload in your build. The table shipped in your own install (`docs/` of the release you built from) is the version that matches your compiler.

In the level column, `z16 - ^` means z16 and up; an op marked `z17` needs Telum II.

Read your op's row before reaching for a flag. `Conv`, for example, documents six conditions: `auto_pad` must be `NOTSET`, `VALID`, or `SAME_UPPER` (and with `NOTSET`, `pads` must produce valid or same-upper padding); **height and width dimensions must be static**; `group` must be 1; `dilations` must be 1; input and output must be 4D (N×C×H×W); and `kernel_shape` must be static. Several of those are export-side facts about how the model was authored, not things a compile flag can change — a grouped or dilated convolution is outside zDNN's contract, and no flag makes it eligible.

Also note the hardware envelope: NNPA computes in DLFLOAT16, so F32/F16 tensors are converted and integer or F64 tensors are ineligible; rank must be 1–4; and each generation has dimension-index and tensor-size limits (large enough for normal models, but a model that exceeds them falls back to CPU).

## Step 5: Make dynamic shapes acceptable

Dynamic dimensions are the most common *fixable* rejection. There are two different tools, and the difference matters:

**`--shapeInformation` — pin the shape.** Changes what gets compiled: the dimension becomes static, so the normal checks pass on their own merits.

```bash
onnx-mlir -O3 --march=z16 --maccel=NNPA --shapeInformation 0:1x3x224x224 mymodel.onnx
```

The syntax is `INPUT_ID:D1xD2x...xDn`, where the id is an input index from 0 (a range like `5-17`, or `-1` for all inputs) and `-1` as a dimension leaves it dynamic. Re-run step 3 afterwards — pinning one input often clears several rejections at once, because the unknown dimension was propagating downstream.

**`--nnpa-disable-shape-restriction` — keep the shape dynamic and assert it will be fine.** The shape stays dynamic in the compiled model; the compiler skips the static height/width checks and defers validation to run time, on your word that the actual values will satisfy NNPA's constraints:

```bash
onnx-mlir -O3 --march=z16 --maccel=NNPA --nnpa-disable-shape-restriction mymodel.onnx
```

This is the flag the rejection message itself points you at:

```
==NNPA-UNSUPPORTEDOPS-REPORT==, onnx.Conv, conv_1, Height and/or width have dynamic dimensions. They are not supported for the VALID padding mode. Use --nnpa-disable-shape-restriction to assume dynamic shapes will satisfy NNPA constraints at runtime.
```

It affects the dynamic height/width checks in the **Conv2D parameter restrictions** (so `Conv`) and in the **common pooling legality check** (so `MaxPool`/`AveragePool`). Use it when you know the spatial dimensions are fixed at inference even though the model declares them dynamic — a model exported with a symbolic H/W that you always call at one resolution is the standard case.

The trade: you are removing a check, not satisfying it. Nothing re-validates the assertion for you, so a run whose actual height/width violates the zDNN constraint is your problem, not the compiler's. Prefer `--shapeInformation` when the shape really is fixed — it makes the same fact explicit in the compiled model instead of implicit in a promise — and reach for `--nnpa-disable-shape-restriction` when the model must stay shape-dynamic for other reasons.

## Step 6: Quantize (z17 only)

8-bit quantization is a throughput lever on **Telum II**: it requires `-march=z17 -maccel=NNPA` and is not available on z16. Two very different situations:

**Your model is already quantized** (by ONNX Runtime or similar, so it contains 8-bit ops): **no flags needed.** The compiler selects suitable 8-bit ops for NNPA on its own, and both statically and dynamically quantized models are supported.

**Your model is float32**: the compiler can quantize during compilation, dynamically only. It is off by default, so "disabling" it means omitting the flag:

```bash
# let the compiler choose scheme and op types
onnx-mlir -O3 --march=z17 --maccel=NNPA --nnpa-quant-dynamic mymodel.onnx

# choose the scheme yourself
onnx-mlir -O3 --march=z17 --maccel=NNPA --nnpa-quant-dynamic=asymActivation,symWeight mymodel.onnx
```

Which scheme to use — the four values are `symWeight`, `asymWeight`, `symActivation`, `asymActivation`, combined comma-separated:

- **Symmetric is usually faster but less accurate; asymmetric is usually more accurate.** The recommended practice is to try combinations on your own model rather than assume.
- **`asymActivation,symWeight` is the useful starting pair** — and not only for the accuracy/speed balance: asymmetric quantization *for weights* is documented as not yet supported, even though `asymWeight` is accepted as a flag value. Passing it does not get you a working asymmetric-weight path.
- Passing `--nnpa-quant-dynamic` with no value lets the compiler pick, and that empty value wins — once it is in the list the other options are ignored.

`--nnpa-quant-op-types` selects which op types get quantized. The published doc's example shows `MatMul,Conv`, but the flag's own help text says **only `MatMul` is currently supported** — so treat MatMul as the real answer and let the compiler decide by default. Two further flags move the dynamic-quantize-linear work off the accelerator when that turns out faster: `--nnpa-cpu-dql` (the whole DQL on CPU) and `--nnpa-cpu-dql-scale` (just the scale/offset computation).

Current limitations worth knowing before you invest: per-tensor only (no per-channel), 8-bit signed integer target, no blocked quantization, and the asymmetric-weight gap above. Full detail: [docs/Quantization-NNPA.md](https://github.com/onnx/onnx-mlir/blob/main/docs/Quantization-NNPA.md), plus [stack/accelerators.md](../../../docs/Quantization-NNPA.md) for the selective `nnpa_ops_config` JSON route.

## Step 7: Tune placement and data movement

Once ops are eligible, two things decide whether offloading them actually pays.

**Placement.** The default heuristic offloads *everything* eligible, even ops that would run faster on CPU:

```bash
--nnpa-placement-heuristic=QualifyingOps      # default: all qualifying ops
--nnpa-placement-heuristic=FasterOps          # only ops faster on NNPA
--nnpa-placement-heuristic=FasterOpsWSU       # FasterOps, counting stick/unstick cost
--nnpa-placement-heuristic=MuchFasterOpsWSU   # only when significantly faster
```

The `WSU` variants price in stick/unstick, which is the point: moving a tensor into and out of NNPA's DLFLOAT16 format is not free, and a cheap op sandwiched between conversions can lose. If your model got *slower* after more ops became eligible, this is the first thing to change.

**Data movement.** Stick/unstick conversion is where offload wins get eaten. Fusion of ops with their surrounding stick/unstick is on by default; the relevant switches (`--nnpa-disable-fusion-op-stick-unstick`, `--disable-compiler-stick-unstick`, `--nnpa-disable-saturation`, `--disable-zhigh-to-onnx`) are documented with their interactions in [stack/accelerators.md](../../../.agents/skills/nnpa-internals/references/accelerators.md#nnpa-placement-and-tuning-flags) and [concepts/stick-unstick-simd-codegen.md](../../../.agents/skills/nnpa-internals/references/stick-unstick-simd-codegen.md). Treat these as experiments to measure, not defaults to change.

**Attention blocks.** Two attention rewrites are on by default, so there is nothing to enable. An `onnx.Attention` op is decomposed into MatMul/Softmax/Add (plus Mul for the scale) before offload, so each piece can go to NNPA. The decomposition does not support GQA/MQA (`kv_num_heads` ≠ `q_num_heads`), `softcap`, `qk_matmul_output_mode` or `softmax_precision`; an `onnx.Attention` using any of them fails the compile with an op error rather than falling back to CPU. An attention mask that broadcasts into at least 4 `MatMul → [Mul] → Add → Softmax` chains of the same shape is expanded once, so those Adds no longer broadcast; broadcasting Adds are a common NNPA rejection. A broadcast scalar scale shared by at least 4 such chains is expanded the same way for the Muls. To measure what the mask expansion is worth, compare against a build with `--nnpa-disable-expand-attention-mask`. Details are in [stack/accelerators.md](../../../.agents/skills/nnpa-internals/references/accelerators.md#onnx-level-rewrites-before-zhigh-lowering).

## Step 8: Measure — the only step that settles it

Everything above is a compile-time decision. Offload is not the goal; wall-clock time is, and more ops on NNPA is not automatically faster. Get per-op timings and compare:

`onnx-mlir` produces a **shared library**, not an executable, so a compiled model needs a driver to load it and call the entry point. `RunONNXModel.py` both compiles and runs, which is the shortest path:

```bash
python3 utils/RunONNXModel.py --model mymodel.onnx \
  --compile-args="-O3 --march=z16 --maccel=NNPA --profile-ir=ZHigh" > profile.txt

grep "PERF-REPORT" profile.txt | sort -t, -k5 -rn | head -20
```

`--profile-ir=ZHigh` is all you need: it instruments both `onnx.*` and `zhigh.*` ops, so one log shows offloaded and fallback ops side by side and you can see what the fallback actually costs. Use `--profile-ir-with-sig=ZHigh` instead to get each op's tensor shapes alongside the timings.

See [debug-compilation.md](../../../.agents/skills/debug-compilation/SKILL.md#step-5-profile-op-level-performance) for reading the profile and finding the slowest ops, and [flows/instrumentation-and-profiling.md](../../../docs/Instrumentation.md) for the full instrumentation surface. Verify correctness alongside performance — `utils/RunONNXModel.py --verify onnxruntime --verify-every-value` — since `--nnpa-disable-shape-restriction` and quantization both trade guarantees for speed.

## Related Pages

- [stack/accelerators.md](../../../.agents/skills/nnpa-internals/references/accelerators.md) — the NNPA path itself: dialects, placement reporter, hardware levels, quantization, tuning flags
- [repos/onnx-mlir/supported-ops-nnpa.md](../../../docs/SupportedONNXOps-NNPA.md) — which ops can offload
- [concepts/nnpa-broadcasting.md](../../../.agents/skills/nnpa-internals/references/nnpa-broadcasting.md) — which broadcast shapes each op supports
- [stack/compiler.md](../../../.agents/skills/onnx-mlir-internals/references/compiler.md#target-selection-which---march-values-get-simd-and-an-accelerator) — what each `--march` value gets you, for SIMD and for the accelerator
- [skills/debug-compilation.md](../../../.agents/skills/debug-compilation/SKILL.md) — general compilation diagnosis and op-level profiling

## Verification

- **Grounded against:** `repo:onnx-mlir@b505d64:src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXLegalityCheck.cpp`, `repo:onnx-mlir@b505d64:src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXToZHighCommon.{cpp,hpp}`, `repo:onnx-mlir@b505d64:src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXToZHigh.cpp`, `repo:onnx-mlir@b505d64:src/Accelerators/NNPA/Conversion/ONNXToZHigh/DevicePlacement.cpp`, `repo:onnx-mlir@b505d64:src/Accelerators/NNPA/Conversion/ONNXToZHigh/RewriteONNXForZHigh.cpp`, `repo:onnx-mlir@b505d64:src/Accelerators/NNPA/Compiler/NNPACompilerOptions.{cpp,hpp}`, `repo:onnx-mlir@b505d64:src/Accelerators/NNPA/Compiler/NNPACompilerUtils.cpp`, `repo:onnx-mlir@b505d64:src/Accelerators/NNPA/Support/NNPALimit.hpp`, `repo:onnx-mlir@b505d64:src/Compiler/CompilerOptions.cpp`, `repo:onnx-mlir@b505d64:src/Dialect/ONNX/ONNXOps/OpHelper.cpp`, `repo:onnx-mlir@b505d64:docs/SupportedONNXOps-NNPA.md`, `repo:onnx-mlir@b505d64:docs/Quantization-NNPA.md`, `repo:onnx-mlir@b505d64:src/Dialect/ONNX/Transforms/ExpandAttentionMask.cpp`; the Attention-blocks paragraph re-checked against `repo:onnx-mlir@b505d64` (`src/Dialect/ONNX/Transforms/ExpandAttentionMask.cpp`, `src/Dialect/ONNX/Transforms/AttentionToONNXOps.cpp`, `src/Accelerators/NNPA/Conversion/ONNXToZHigh/RewriteONNXForZHigh.cpp`)
- **Files read:**
  - `ONNXLegalityCheck.cpp` — `onnxToZHighUnsupportedReport` (the `printf` format) and the reason strings; `onnxToZHighInCompatibilityReport`; `isValidElementTypeAndRank` (F16/F32, rank 1–4); the three `nnpaDisableShapeRestriction` sites, which are `checkLegalityPoolOpsCommon` (pooling H/W) and `checkConv2DParamRestrictions` (Conv, called at the `ONNXConvOp` legality check), including the two rejection messages that name the flag
  - `NNPACompilerOptions.cpp/.hpp` — `OPTREPORT_CL_ENUM_NNPA`; `nnpaDisableShapeRestriction` desc; `nnpaQuantDynamicOpt` (the four scheme values plus the `autoQuantOpt` empty-string default that suppresses the others, `ValueOptional`, `CommaSeparated`); `nnpaQuantOpTypesOpt` ("Currently, only MatMul is supported"); `nnpaUseDynamicQuantizeLinearOnCPU{,ForScaleOffset}`; `nnpaPlacementHeuristic` (four values, `QualifyingOps` default); `nnpaDisableSaturation`; `nnpaDisableFusionOpStickUnstick`; `NNPAEmissionTargetType`
  - `NNPACompilerUtils.cpp` — `ExpandAttentionMaskPass` scheduled at the start of `addONNXToZHighPasses` unless `nnpaDisableExpandAttentionMask`; `RewriteONNXForZHigh.cpp` — `AttentionForZHighPattern` (delegating to `lowerONNXAttentionOp`) and the always-illegal `ONNXAttentionOp` legality rule; `AttentionToONNXOps.cpp` — `checkSupportedAttentionAttributes`; `ExpandAttentionMask.cpp` — `minUses = 4`, optional scalar-scale `Mul`
  - `NNPACompilerUtils.cpp` — `configureONNXToZHighLoweringPass(optReport == OptReport::NNPAUnsupportedOps, ...)`; pass order in `addPassesNNPA`: `createDevicePlacementPass` under `emissionTarget >= EmitONNXIR` but `addONNXToZHighPasses` under `>= EmitMLIR`, which is what makes the report silent at `--EmitONNXIR`
  - `DevicePlacement.cpp` / `ONNXToZHigh.cpp` — `reportOnNNPAUnsupportedOps` zeroed in the placement pass and restored in `ONNXToZHighLoweringPass::runOnOperation`
  - `NNPALimit.hpp` — `NNPALevel` M14/M15, `MAXIMUM_NUM_HIDDEN_SIZE_LSTM` 8192 / `_GRU` 10880, `NNPAGetMaxForDim`; `RewriteONNXForZHigh.cpp` — oversized MatMul dims are split, not rejected
  - `CompilerOptions.cpp` — `--opt-report` values, `--shapeInformation` syntax, `--onnx-op-stats` and its emission-target requirement; `OpHelper.cpp` — `getNodeNameInPresenceOfOpt` resolving the report's third field from `onnx_node_name`
  - `docs/SupportedONNXOps-NNPA.md` — the `Conv` row's six limitations, the level column convention, the DLFLOAT16/dimension-limit preamble; `docs/Quantization-NNPA.md` — z17 requirement, already-quantized vs float32 paths, per-tensor/i8/no-blocked/asymmetric-weight limitations, and the symmetric-faster/asymmetric-more-accurate guidance
- **How to test:** compile a model with `--maccel=NNPA --opt-report=NNPAUnsupportedOps` and confirm `==NNPA-UNSUPPORTEDOPS-REPORT==` lines appear at the default emission target and are absent at `--EmitONNXIR`; a `Conv` with a dynamic H or W should be rejected with a message naming `--nnpa-disable-shape-restriction`, and should offload either once that flag is passed or once `--shapeInformation` pins the input; `--nnpa-quant-dynamic` on a float32 model requires `--march=z17` to have any effect
