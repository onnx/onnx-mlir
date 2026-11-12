---
name: add-new-onnx-operation
description: 'Use when adding support for an ONNX operator onnx-mlir doesn''t handle yet, end to end: dialect registration, shape inference and verifier, canonicalize/decompose/recompose choices, Krnl lowering, and CPU/NNPA backend tests. Start here; it routes to the narrower add-onnx-op-to-dialect and add-onnx-op-lowering recipes.'
---

<!-- Generated from the onnx-mlir knowledge base (skills/add-new-onnx-operation.md), verified against onnx-mlir@b505d64b. Do not edit here: see .agents/skills/README.md. -->

# Add a New ONNX Operation, End to End

Walkthrough tying together dialect registration, shape inference/verification, canonicalize/decompose/recompose, `ONNXToKrnl` lowering, and backend testing for a brand-new ONNX operator. It is the map connecting two narrower existing recipes — [add-onnx-op-to-dialect.md](../../../.agents/skills/add-onnx-op-to-dialect/SKILL.md) and [add-onnx-op-lowering.md](../../../.agents/skills/add-onnx-op-lowering/SKILL.md) — and filling in what neither of them covers.

---

## 1. Register the op in the dialect

Official ONNX ops (anything defined in the `onnx` opset) are added through `utils/gen_onnx_mlir.py`, which auto-generates `src/Dialect/ONNX/ONNXOps.td.inc` — see [add-onnx-op-to-dialect.md](../../../.agents/skills/add-onnx-op-to-dialect/SKILL.md) for the `version_dict`/`OpsWithVerifier`/etc. mechanics. Never hand-edit `ONNXOps.td.inc`.

`src/Dialect/ONNX/AdditionalONNXOps.td` is **not** a catch-all for "custom ops outside the ONNX standard." Its own header comment scopes it to two things only: internal helper ops that assist processing the ONNX dialect (`NoneOp`, `YieldOp`, ...) and ops that model optional-argument/special-case variants of an *existing* standard op (e.g. `ONNXBatchNormalizationInferenceModeOp`). A genuinely new op that mirrors an ONNX opset operator still goes through `gen_onnx_mlir.py`, even if it feels bespoke.

Regeneration and rebuild:
- `gen_onnx_mlir.py` edits (new op, `version_dict` opset bump, `OpsWithVerifier`, `OpsWithCanonicalizer`, ...) have no effect until `make OMONNXOpsIncTranslation` reruns the script and refreshes `ONNXOps.td.inc` and `src/Builder/OpBuildTable.inc`.
- If the change affects the rendered op documentation, also run `make onnx-mlir-docs`.
- Upgrading an existing op to a newer opset version is the same file, same list: add the new version number to that op's entry in `version_dict`. An upgrade is not done until the op's backend tests have been rechecked — see [Extending an existing op](#extending-an-existing-op-recheck-its-backend-tests) in step 6.
- Because these generated files ripple into a lot of downstream TableGen/build state, it's common practice after editing `gen_onnx_mlir.py` to remove the `build/` directory and rebuild the compiler from scratch rather than trust an incremental build.

## 2. Implement shape inference and the verifier

Per-op `inferShapes()`/`verify()` bodies live under `src/Dialect/ONNX/ONNXOps/<Category>/<File>.cpp` — one directory per op family: `Additional`, `ControlFlow`, `Math`, `ML`, `NN`, `ObjectDetection`, `Quantize`, `RNN`, `Sequence`, `Tensor`. For example, `Math/ElementwiseUnary.cpp` implements both for the elementwise math ops. The top-level `src/Dialect/ONNX/ONNXOps.cpp` is dialect-wide boilerplate, not where new per-op logic belongs.

Shared shape-computation infrastructure sits in `ONNXOps/ShapeHelper.hpp`/`.cpp`. Look for an existing shape-helper class for a semantically similar op before writing a new one — reuse or subclass it. If you do write a custom shape-helper class, it's fine to store extra fields on it that a later lowering pattern will need; that avoids recomputing the same `IndexExpr` twice.

All shape arithmetic, in both shape inference and lowering, should go through `IndexExpr` (same data structure, same arithmetic, different scope) — see [concepts/index-expr.md](../../../.agents/skills/onnx-mlir-internals/references/index-expr.md).

A verifier is optional but strongly recommended whenever the op has constraints beyond basic type checking. Register it via `OpsWithVerifier` in `gen_onnx_mlir.py`, then rerun the regeneration step above.

Ops already in the dialect without shape inference are listed in `src/Dialect/ONNX/ONNXUnsupportedOps.hpp`, whose `UNSUPPORTED_OPS` macro stubs `inferShapes()` to a no-op; remove the op from that list when you implement it (as the `Col2Im` and `Det` additions did; `Det` also added `ONNXDetOpShapeHelper` to `ShapeHelper.hpp`, a verifier in `ONNXOps/Math/Det.cpp`, and `"Det"` to `OpsWithVerifier`). `onnx.TensorScatter` (`ONNXOps/Math/TensorScatter.cpp`) is a compact example of a verifier plus reuse of `ONNXUnaryOpShapeHelper`.

## 3. Consider a DialectBuilder helper

If the new op will be constructed programmatically in more than one place (a decompose/recompose pattern, a lowering, both), add a named method to `OnnxBuilder` (`src/Dialect/ONNX/DialectBuilder.hpp`/`.cpp`) rather than repeating the same builder calls at each call site. Existing methods (`add`, `cast`, `conv`, ...) are the pattern to follow — they wrap `createOpAndInferShapes<ONNX...Op>(...)`. See [concepts/dialect-builders.md](../../../.agents/skills/onnx-mlir-internals/references/dialect-builders.md).

## 4. Canonicalize, decompose, or recompose?

Three distinct rewrite mechanisms exist around the ONNX dialect; pick the one that matches what the op actually needs (background: [stack/dialect-onnx.md](../../../.agents/skills/onnx-mlir-internals/references/dialect-onnx.md)):

- **`--canonicalize`** (continuous peephole simplification). Register the op via `OpsWithCanonicalizer` in `gen_onnx_mlir.py` — this is what makes the generated op class declare the `getCanonicalizationPatterns` hook at all; skipping it means the pattern below is simply never called. Then implement the pattern either as a DRR rule in `Canonicalize.td`, or, when the rewrite needs logic DRR can't express, hand-written C++ in `Canonicalize.cpp` (`<Op>::getCanonicalizationPatterns`, e.g. `ONNXAddOp::getCanonicalizationPatterns`). Check how sibling ops in the same family are already canonicalized before defaulting to hand-written C++. A canonicalizer can also be the op's whole CPU support: `GlobalLpPoolPattern` rewrites `onnx.GlobalLpPool` into `Abs`/`Pow`/`ReduceSum`, so the op needs no Krnl lowering of its own.
- **`--decompose-onnx`** (runs early, before shape inference exists for anything). Rewrites one op into a composition of simpler, already-supported ops; patterns are DRR rules in `Decompose.td`/`Decompose.cpp`. If every instance of the new op can be fully decomposed this early, the op may not need its own shape inference at all — the ops it decomposes into carry their own.
- **`--recompose-onnx`** (also early, the inverse direction). Detects a pattern of existing ops in the graph and fuses them into the new op. Implemented as hand-written `mlir::OpRewritePattern` C++ classes in `Recompose.cpp` — there is no `Recompose.td`/DRR path here (the DRR include is commented out) — e.g. `RecomposeLayerNormFromMulPattern`, `RecomposeGeluFromMulPattern`. A recompose pattern that should be switchable gets a flag in `CompilerOptions.cpp` checked in `getRecomposeONNXToONNXPatterns()` (`--enable-attention-op-construct` gates one in; `--disable-split-matmul-by-slice` and `--disable-unbind-to-split` each gate one out).

Regardless of which mechanism is used, the `OMONNXOpsIncTranslation` rebuild from step 1 is what actually wires a canonicalizer registration into the generated op class; a DRR pattern change alone (`Canonicalize.td`/`Decompose.td`) is picked up by a normal build via its own TableGen target.

## 5. Lower to Krnl

Mechanics of adding an `ONNXToKrnl` conversion pattern are in [add-onnx-op-lowering.md](../../../.agents/skills/add-onnx-op-lowering/SKILL.md); onnx-mlir's own writeup of the lowering pass is `docs/LoweringCode.md`.

One rule that's easy to get wrong: an `IndexExpr` built in one `IndexExprScope` cannot simply be reused once you descend into a new, nested scope — e.g. the body of a loop just introduced. Create a fresh child scope per loop level (only at the innermost level of a perfectly-nested loop, not at every level), and regenerate what's needed from the outer scope via `DimIE`/`SymIE` rather than reusing the outer-scope object directly — an outer `Dim` becomes an inner `Symbol`. See [concepts/index-expr.md](../../../.agents/skills/onnx-mlir-internals/references/index-expr.md#nested-scopes) for the worked example (`ONNXSliceOp` bounds).

Prefer existing `DialectBuilder` methods (`create.krnl.*`, `create.math.*`, `create.onnx.*`, ...) over hand-rolled `OpBuilder` calls when creating ops. If the method needed doesn't exist yet, add it to the relevant builder (see step 3) rather than reaching around the builder layer.

## 6. Add tests

**Lit tests** — one per transformation stage actually touched: dialect import/shape-inference, canonicalize/decompose/recompose if applicable, `ONNXToKrnl` lowering. Follow the pattern of existing tests in the same directory.

**Backend (end-to-end numeric) tests** compile an ONNX-provided model and compare its output against ONNX's reference result. The test cases themselves come from ONNX, not onnx-mlir: each op has a generator file `third_party/onnx/onnx/backend/test/case/node/<op>.py` whose `expect(..., name="test_<...>")` calls define the cases, and the backend runner adds a `_cpu` suffix to each name (`name="test_softmax_axis_0"` → `test_softmax_axis_0_cpu`). onnx-mlir only chooses which of those cases to enable — in two separate lists, one for CPU and one for NNPA.

### CPU: `test/backend/inference_backend.py`

Documented in `docs/Testing.md`.

- Every test case ONNX itself provides is listed in `test/backend/all_test_names.txt`, including the `_expanded` / `_expanded_ver18` variants that run an op's ONNX function body instead of the op itself. It's regenerated by `make check-onnx-backend-case`, needed only when the vendored ONNX package is upgraded — don't hand-edit it.
- Enable specific cases by adding entries to `variables.node_test_to_enable_dict` in `test/backend/inference_backend.py`, `_cpu` suffix included (e.g. `test_add_uint8_cpu`). Add new ops in the order they appear in `third_party/onnx/onnx/backend/test/case/node`, and add each op's individual tests in the order that generator file defines them.
- Each entry's `STATIC_SHAPE` / `DYNAMIC_SHAPE` / `CONSTANT_INPUT` keys say which inputs/dimensions get exercised with static shapes, forced-dynamic shapes, or forced-constant inputs. `DYNAMIC_SHAPE: {-1: {-1}}` and `CONSTANT_INPUT: {-1}` (all inputs, all dimensions) is the default, recommended for most ops — narrow the indices only when the op can't tolerate a dynamic shape or constant-folded input on every argument.
- Directly above an op's first enabled test, a comment block records `# ==OP== <OpName>`, `# ==MIN== <opset>`, and, when there's a known gap, `# ==LIM== <description>`. These are machine-read by `utils/documentOps.py` to regenerate `docs/SupportedONNXOps-cpu.md` (the file's header comment still calls the script `utils/genSupportedOps.py`). The generated opset range runs from `==MIN==` (6 when omitted) to `*`; an `==UNSUPPORTED== <n>` line caps it at `n-1`. The `==LIM==` text becomes the Limitations column verbatim.
- A known limitation is recorded as a commented-out test line directly below the `==LIM==` comment that explains it — e.g. `# "test_add_uint8_cpu" : {...}` sits right under `Add`'s `# ==LIM== No support for short integers.`. If you know a limitation exists for the op you're adding, find the matching test name this same way and comment it out with a `==LIM==` line above the block, rather than leaving it silently enabled and failing.
- To see what a specific test actually exercises, read the matching generator file: `third_party/onnx/onnx/backend/test/case/node/<name>.py`, where `<name>` is the ONNX test name with the trailing `_cpu` (and any onnx-mlir-added type suffix) stripped — e.g. `test_add_uint8_cpu` → `add.py`.
- Run one test in isolation with `TEST_CASE_BY_USER=<test_name> cmake --build . --config Release --target check-onnx-backend`; this preserves the intermediate `.onnx`/`.so` files under `build/test/backend` for debugging.

### NNPA: `test/accelerators/NNPA/backend/CMakeLists.txt`

Documented in `docs/AccelNNPAHowToUseAndTest.md`. An op that offloads to NNPA needs its own entries here; the CPU dictionary above does not cover it.

- The enabled cases are CMake lists rather than a Python dict: `NNPA_TEST_LIST_z16` for ops supported from z16 on, and `NNPA_TEST_LIST_ARCH_15` for z17-only ops (e.g. `Gelu`). One entry per line, of the form `<test_name>,<zdnn_function>,<dimParams>`:
  - `<zdnn_function>` is the zDNN entry point that must appear in the compiled library — e.g. `test_softmax_example_cpu,zdnnx_softmax`. CMake sets `TEST_INSTRUCTION_CHECK` for these runs, so a test whose op silently fell back to CPU fails instead of passing.
  - `<dimParams>` (optional) is passed as `--dimParams` for the dynamic-shape runs and says which dims of which inputs are equal — `"0:0=a,1=b,2=c|1:0=a,1=b,2=c"` means dims 0–2 of inputs 0 and 1 match. `NO_DYNAMIC_SHAPE_TEST` instead drops the case from the dynamic targets (used throughout pooling and convolution).
- Same marker block above each op, with one addition: `# ==LEVEL== z16,z17` lists the hardware levels. It is required — `documentOps.py` leaves an op without `==LEVEL==` out of `docs/SupportedONNXOps-NNPA.md` altogether. `==MIN==` and `==LIM==` work as on CPU, and so does the commented-out-test convention.
- Run with `cmake --build . --config Release --target check-onnx-backend-nnpa`, plus `check-onnx-backend-dynamic-nnpa` and `check-onnx-backend-constant-nnpa`; the z17 list runs under `check-onnx-backend-arch15-nnpa` and `check-onnx-backend-dynamic-arch15-nnpa`. Tolerances (`TEST_ATOL=0.01`, `TEST_RTOL=0.05`) are set in the CMake file, since NNPA computes in DLFLOAT16.

### Regenerate the supported-ops docs

After any change to `==OP==` / `==MIN==` / `==LIM==` / `==LEVEL==` / `==UNSUPPORTED==` lines in either file, regenerate and commit the matching doc with the change — never hand-edit them:

```bash
make onnx_mlir_supported_ops_cpu    # docs/SupportedONNXOps-cpu.md  from inference_backend.py
make onnx_mlir_supported_ops_NNPA   # docs/SupportedONNXOps-NNPA.md from NNPA/backend/CMakeLists.txt
make onnx_mlir_supported_ops        # both
```

### Extending an existing op: recheck its backend tests

Any change that widens what an existing op supports — an opset upgrade in `version_dict`, a new attribute value, a new element type, a dynamic-shape or rank case, a lifted NNPA restriction — can make ONNX test cases pass that were previously disabled. Checking for those is part of the change, not a follow-up. Do it for the CPU list, and for the NNPA list too if the op offloads:

1. **List every case ONNX defines for the op.** Read `third_party/onnx/onnx/backend/test/case/node/<op>.py` and collect each `name="test_..."`, noting what each one varies (axis, attribute, type, rank). After an ONNX upgrade, also check `all_test_names.txt` for names that are new for the op.
2. **Compare against both lists.** For each case, check whether it is enabled, commented out under a `==LIM==`, or absent. Every commented-out or absent case that your change now covers is a candidate — enable it (for NNPA, with the zDNN function name to check).
3. **Update the `==LIM==` text.** Narrow or delete the part describing the restriction you lifted; remove the line when nothing is left. Also adjust `==MIN==` if the lowest supported opset changed, and `==LEVEL==` if the NNPA hardware levels changed. Leave cases that still fail commented out, with a reason.
4. **Regenerate the docs** (above) and commit the regenerated `docs/SupportedONNXOps-*.md` alongside the code.
5. **Run each newly enabled case** with `TEST_CASE_BY_USER=<test_name>`, through the static, dynamic and constant targets, before relying on the full suite.

**Worked example — NNPA `Softmax` at `b505d64`.** `softmax.py` defines seven cases: `example` (2-D, axis 1), `large_number` (2-D, default axis), and on a 3-D input `axis_0`, `axis_1`, `axis_2`, `negative_axis` (-1) and `default_axis` (-1). The NNPA list's `==LIM==` reads "`axis` must be the last dimension", and only `test_softmax_example_cpu,zdnnx_softmax` is enabled: `axis_0`/`axis_1`/`axis_2`/`default_axis` sit commented out under that `==LIM==`, `large_number` is commented out for an accuracy error, and `negative_axis` is not listed at all. Running step 2 against the legality check (`isSuitableForZDNN<ONNXSoftmaxOp>` in `ONNXLegalityCheck.cpp` accepts rank 2 or 3 with `axis == rank-1` or `-1`) shows three last-axis 3-D cases — `axis_2`, `negative_axis`, `default_axis` — are already within the restriction but not enabled. It also shows the `==LIM==` text omits the rank-2-or-3 condition. A change that lifted the last-axis restriction would make `axis_0` and `axis_1` candidates as well, and would replace that `==LIM==` line.

---

## Verification

- **Grounded against:** `repo:onnx-mlir@b505d64:src/Dialect/ONNX/AdditionalONNXOps.td`, `repo:onnx-mlir@b505d64:src/Dialect/ONNX/ONNXOps/` (directory layout, `Canonicalize.cpp`/`.td`), `repo:onnx-mlir@b505d64:src/Dialect/ONNX/Transforms/Decompose.cpp`, `repo:onnx-mlir@b505d64:src/Dialect/ONNX/Transforms/Recompose.cpp`, `repo:onnx-mlir@b505d64:src/Dialect/ONNX/DialectBuilder.hpp`, `repo:onnx-mlir@b505d64:utils/CMakeLists.txt`, `repo:onnx-mlir@b505d64:MLIR.cmake`, `repo:onnx-mlir@b505d64:test/backend/inference_backend.py`, `repo:onnx-mlir@b505d64:test/backend/all_test_names.txt`, `repo:onnx-mlir@b505d64:docs/Testing.md`
- **Files read:** `src/Dialect/ONNX/AdditionalONNXOps.td` (header comment), `src/Dialect/ONNX/ONNXOps/` directory listing, `src/Dialect/ONNX/ONNXOps/Math/ElementwiseUnary.cpp` (per-op `inferShapes`/`verify` placement), `src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp`, `src/Dialect/ONNX/DialectBuilder.hpp` (`OnnxBuilder` struct), `src/Dialect/ONNX/Transforms/Decompose.cpp` and `Recompose.cpp` (header comments), `utils/CMakeLists.txt` (`OMONNXOpsIncTranslation`, `OMONNXCheckVersion`, `add_onnx_mlir_supported_ops` calls), `MLIR.cmake` (`add_onnx_mlir_supported_ops` function body, `onnx-mlir-docs` target), `test/backend/inference_backend.py` (`node_test_to_enable_dict`, `==OP==`/`==MIN==`/`==LIM==` markers, the commented-out `test_add_uint8_cpu` line), `utils/documentOps.py` (marker regexes), `docs/Testing.md` (`all_test_names.txt`, `TEST_CASE_BY_USER`, `check-onnx-backend-case`)
- **Re-checked at `b505d64`** (every item above, first grounded at `cf0842f3`): `AdditionalONNXOps.td` header (two sections, `make OMONNXOpsIncTranslation`/`make onnx-mlir-docs`), the ten `ONNXOps/` subdirectories, `Math/ElementwiseUnary.cpp` (`inferShapes` and `verify` bodies), `Canonicalize.cpp`/`.td`, `Decompose.td`/`.cpp`, `Recompose.cpp` (DRR include still commented out; `RecomposeLayerNormFromMulPattern`, `RecomposeGeluFromMulPattern`; `disableUnbindToSplit` guard), `DialectBuilder.cpp` (`add`/`cast`/`conv` via `createOpAndInferShapes`), `utils/CMakeLists.txt`, `MLIR.cmake`, `test/backend/inference_backend.py` (`Add` `==LIM==` and commented-out `test_add_uint8_cpu`; header still says `utils/genSupportedOps.py`), `docs/Testing.md`, plus `ONNXUnsupportedOps.hpp` and `ONNXOps/Math/Det.cpp` for the `Det` example
- **Re-checked at:** `repo:onnx-mlir@b505d64:src/Dialect/ONNX/ONNXUnsupportedOps.hpp`, `repo:onnx-mlir@b505d64:src/Dialect/ONNX/ONNXOps/Math/TensorScatter.cpp`, `repo:onnx-mlir@b505d64:src/Dialect/ONNX/ONNXOps/Canonicalize.cpp` (`GlobalLpPoolPattern`), `repo:onnx-mlir@b505d64:src/Dialect/ONNX/Transforms/Recompose.cpp` (`getRecomposeONNXToONNXPatterns`), `repo:onnx-mlir@b505d64:src/Compiler/CompilerOptions.cpp`
- **Backend-test sections grounded against:** `repo:onnx-mlir@b505d64:test/backend/inference_backend.py` (header comment, `Softmax` block), `repo:onnx-mlir@b505d64:test/backend/all_test_names.txt` (`test_softmax_*`), `repo:onnx-mlir@b505d64:test/accelerators/NNPA/backend/CMakeLists.txt` (entry-format comment, `NNPA_TEST_LIST_z16`/`NNPA_TEST_LIST_ARCH_15`, `Softmax`/`LogSoftmax` blocks, `NO_DYNAMIC_SHAPE_TEST` filtering, `check-onnx-backend-*-nnpa` targets, `TEST_ATOL`/`TEST_RTOL`), `repo:onnx-mlir@b505d64:utils/documentOps.py` (directive list, `min_opset_default`, `==LEVEL==` requirement for NNPA, `==UNSUPPORTED==` → max), `repo:onnx-mlir@b505d64:MLIR.cmake` (`add_onnx_mlir_supported_ops`, `onnx_mlir_supported_ops[_<arch>]` targets), `repo:onnx-mlir@b505d64:utils/CMakeLists.txt`, `repo:onnx-mlir@b505d64:docs/Testing.md`, `repo:onnx-mlir@b505d64:docs/AccelNNPAHowToUseAndTest.md`, `repo:onnx-mlir@b505d64:src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXLegalityCheck.cpp` (`isSuitableForZDNN<ONNXSoftmaxOp>`), `repo:onnx@ee3ccbd2:onnx/backend/test/case/node/softmax.py` (seven `expect` names and their axes)
- **How to test:** for a real op addition, work through sections 1–6 in order; after step 1's rebuild, `onnx-mlir --EmitONNXIR test.onnx` should show the new op; after step 6, `TEST_CASE_BY_USER=<test_name> cmake --build . --config Release --target check-onnx-backend` should compile and numerically verify the enabled case(s) for that op. For an extension of an existing op, step 6's checklist should end with no case in the generator file that the change covers still disabled, and `git diff docs/SupportedONNXOps-*.md` should show the regenerated row.
