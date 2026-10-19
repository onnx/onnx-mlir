---
name: add-onnx-transform-pass
description: 'Use when creating a new onnx-mlir compiler pass (ONNX-level transform or other): declaring it in the right Passes.td with TableGen, implementing it, wiring CMake, registering it in onnx-mlir-opt, and adding it to the compiler pipeline.'
---

<!-- Generated from the onnx-mlir knowledge base (skills/add-onnx-transform-pass.md), verified against onnx-mlir@b505d64b. Do not edit here: see .agents/skills/README.md. -->

# Add a New Pass Using TableGen

Recipe for declaring a new onnx-mlir compiler pass's class via TableGen (`Passes.td` + `PassBase.td`) instead of hand-rolling a `PassWrapper` subclass. This is the pattern to follow for any new pass; most *existing* passes in this area predate the convention and hand-roll their class directly — see "Legacy pattern" below before copying one of those as a template.

---

## Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `{{PASS_NAME}}` | Pass class name (CamelCase, ends in `Pass`) | `AppendDecodingStrategyPass` |
| `{{CLI_NAME}}` | Command-line pass name (kebab-case) | `append-decoding-strategy` |
| `{{ROOT_OP}}` | MLIR op type the pass runs on | `ModuleOp` |
| `{{OWNER_DIR}}` | Directory owning the `Passes.td` for this area | `src/Dialect/ONNX/Transforms` |

---

## Step 1: Pick the owning `Passes.td`

Four pass-declaration `.td` files exist repo-wide today. Three are wired into the build by the `add_onnx_mlir_pass(NAME TD_FILE PREFIX)` CMake helper in `MLIR.cmake` (`mlir_tablegen(...-gen-pass-decls -name ${NAME})` + `add_public_tablegen_target(OM${NAME}PassIncGen)`):

- `src/Dialect/ONNX/Transforms/Passes.td` — ONNX-dialect graph-level passes (`add_onnx_mlir_pass(ONNXTransforms Passes ONNXTransforms)` in that directory's `CMakeLists.txt`)
- `src/Transform/Passes.td` — cross-dialect/general transforms (OpenMP/parallel plumbing, instrumentation, CSE) (`add_onnx_mlir_pass(Transforms Passes Transforms)`)
- `src/Transform/PassesKrnl.td` — Krnl-specific transforms, split into its own file so Krnl-dialect code can depend on it separately; today only `LowerKrnlRegionPass` (`add_onnx_mlir_pass(TransformsKrnl PassesKrnl TransformsKrnl)`, target `OMTransformsKrnlPassIncGen`)

The fourth, `src/Conversion/ONNXToLinalg/Passes.td` (the ONNX→Linalg conversion pass), calls `mlir_tablegen(Passes.h.inc -gen-pass-decls -name ONNXToLinalg)` and `add_public_tablegen_target(OMONNXToLinalgPassIncGen)` directly instead of using the helper.

If your new pass belongs to one of these areas, add a `def` there. Otherwise, create a new `Passes.td` and add a matching `add_onnx_mlir_pass(...)` call to that directory's `CMakeLists.txt`, mirroring the first three above.

---

## Step 2: Write the `def`

```tablegen
def {{PASS_NAME}} : Pass<"{{CLI_NAME}}", "{{ROOT_OP}}"> {
  let summary = "One-line summary.";
  let description = [{
    Longer description of what the pass does and why.
  }];
}
```

This is exactly the shape of the passes already following this convention in `src/Dialect/ONNX/Transforms/Passes.td`: `AppendDecodingStrategyPass`, `AppendInstrumentedOutputsPass`, `FusionOpTransformPass` (all `ModuleOp`), and `ExpandAttentionMaskPass` (`func::FuncOp`). `ExpandAttentionMaskPass` additionally sets `let constructor = "onnx_mlir::createExpandAttentionMaskPass()";` and `let dependentDialects = [...]`, declares the factory by hand in `src/Pass/Passes.hpp`, and defines it at the bottom of its `.cpp` — a default-constructible pass does not need that, but it is a working variant.

### Step 2a: Prefer a TableGen `Option` over a hand-written constructor

If the pass needs a simple per-invocation setting (a string/bool/int), give it a TableGen `Option` rather than a custom C++ constructor — it becomes a real field on the generated base class *and* a CLI flag settable through both `onnx-mlir-opt` and pass-pipeline strings, at no extra registration cost. `ConvertONNXToLinalg` (`src/Conversion/ONNXToLinalg/Passes.td`) is the grounded example:

```tablegen
let options = [
  Option<"linalgOps", "linalg-ops", "std::string", /*default=*/"",
         "...">,
  Option<"useLinalgPath", "use-linalg-path", "bool", /*default=*/"false",
         "...">
];
```

### Step 2b: Fall back to a custom constructor only when the value can't be an `Option`

Use this only when the value must be threaded in programmatically by the driver rather than exposed as a standalone `--{{CLI_NAME}}=opt` flag (e.g. it's derived from another already-parsed compiler option). `AppendInstrumentedOutputsPass` and `InstrumentPass` follow this fallback: the `def` in `Passes.td` stays option-less, and in `src/Pass/Passes.hpp`:

```cpp
#define GEN_PASS_DECL_{{PASS_NAME_UPPERCASE}}
#include "{{OWNER_DIR}}/Passes.h.inc"
// GEN_PASS_DEF method only adds default constructor only,
// we add custom constructor with an explicit ... argument.
std::unique_ptr<mlir::Pass> create{{PASS_NAME}}(/* real args */);
```

Reuse that exact comment (or a close variant) — it's the established way this codebase flags "yes, we deliberately bypassed the generated default constructor here."

---

## Step 3: Implement the class

In the pass's `.cpp`:

```cpp
namespace onnx_mlir {
#define GEN_PASS_DEF_{{PASS_NAME_UPPERCASE}}
#include "{{OWNER_DIR}}/Passes.h.inc"
} // namespace onnx_mlir

namespace {
class {{PASS_NAME}}
    : public onnx_mlir::impl::{{PASS_NAME}}Base<{{PASS_NAME}}> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID({{PASS_NAME}})

  void runOnOperation() override {
    // ...
  }
};
} // namespace
```

Any `Option`s from Step 2a show up as plain member fields on the base class (e.g. `linalgOps`, `useLinalgPath`) — don't redeclare them. If you took the Step 2b fallback, add the constructors you declared in `Passes.hpp` and store the extra state as an ordinary private member, the way `AppendInstrumentedOutputsPass` stores `nodeNamePattern`.

For a no-option, default-constructible pass, no hand-written factory is needed — the generated header already provides `create{{PASS_NAME}}()`, matching `createAppendDecodingStrategyPass()`. For the Step 2b fallback, define the factory yourself at the bottom of the `.cpp`, e.g.:

```cpp
std::unique_ptr<mlir::Pass> onnx_mlir::create{{PASS_NAME}}(/* args */) {
  return std::make_unique<{{PASS_NAME}}>(/* args */);
}
```

---

## Step 4: Wire up the build

Add the `.cpp` to (or add a new) `add_onnx_mlir_library(...)` target in that directory's `CMakeLists.txt`, and make sure it depends on the tablegen target from Step 1 so the generated header exists before the pass compiles:

```cmake
add_onnx_mlir_library(OM{{LibraryName}}
  {{PassName}}.cpp

  DEPENDS
  OM{{PREFIX}}PassIncGen

  LINK_LIBS PUBLIC
  OMONNXOps
  MLIRPass
  )
```

(`DEPENDS OMONNXTransformsPassIncGen` is what `OMAppendDecodingStrategy`, `OMInstrumentONNX`, `OMFusionOpTransform` and `OMExpandAttentionMask` use for the `ONNXTransforms` area.) `OMFusionOpHelper` (`FusionOpHelper.cpp`) also declares it although it defines no pass, which orders it after the pass-declaration TableGen step.

---

## Step 5: Register and wire into pipelines

- **`onnx-mlir-opt`**: add a `mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> { return create{{PASS_NAME}}(...); });` lambda inside `registerOMPasses()` in `src/Tools/onnx-mlir-opt/RegisterPasses.cpp`. Note that even TableGen-declared passes in `Dialect/ONNX/Transforms/Passes.td` are registered this way, one lambda per pass — onnx-mlir does not rely on the TableGen-generated bulk `registerXXXPasses()` entry point for that file (it does use it for `src/Transform/Passes.td` and `src/Conversion/ONNXToLinalg/Passes.td`, via `::registerTransformsPasses()` / `::registerONNXToLinalgPasses()`).
- **Compiler driver**: call `pm.addPass(onnx_mlir::create{{PASS_NAME}}(...))` at the right pipeline stage in `src/Compiler/CompilerPasses.cpp` — `pm.addNestedPass<func::FuncOp>(...)` for a `func::FuncOp`-rooted pass. A pass only an accelerator wants is scheduled from that accelerator's driver instead, e.g. `ExpandAttentionMaskPass` in `addONNXToZHighPasses()` (`src/Accelerators/NNPA/Compiler/NNPACompilerUtils.cpp`), gated by `--nnpa-disable-expand-attention-mask`.

---

## Step 6: Build and verify

```bash
cd build && make -j$(nproc) onnx-mlir-opt onnx-mlir
onnx-mlir-opt --help | grep {{CLI_NAME}}
```

Run it on a small `.mlir` input to confirm it's reachable and does what's expected:

```bash
onnx-mlir-opt --{{CLI_NAME}} /tmp/test.mlir
```

This confirms the pass is *wired up*, not that its output is *correct*. Before freezing that output into a FileCheck test, ground it numerically — [ground-lit-tests.md](../../../.agents/skills/ground-lit-tests/SKILL.md) is the workflow, and it also covers the two cases that matter most for a pass with a CLI option like this one: a case grounded with the option *off* over the same code, and a negative test for every restriction the pass documents (otherwise the restriction is only a comment). Its [Step 1b](../../../.agents/skills/ground-lit-tests/SKILL.md#step-1b-keep-the-run-line-minimal) also covers the RUN line itself: pass `--{{CLI_NAME}}` and as few other optimization passes as practical, so a failure is attributable to this pass rather than to something upstream of it.

---

## Legacy pattern — do not copy for new passes

Most passes in `src/Dialect/ONNX/Transforms/` predate this convention and hand-roll a `PassWrapper<{{PassName}}, OperationPass<...>>` subclass directly, with no corresponding `Passes.td` entry: `DecomposeONNXToONNXPass`, `ConvOptONNXToONNXPass`, `ShapeInferencePass`, `ONNXHybridTransformPass`, and `ReplaceOpWithItsOperandPass` all do this. They are not being retrofitted as part of this recipe — but a new pass should follow Steps 1–6 above, not imitate one of these.

---

## Verification

- **Re-checked at:** `repo:onnx-mlir@b505d64:src/Dialect/ONNX/Transforms/Passes.td` (four `def`s; `ExpandAttentionMaskPass` `constructor`/`dependentDialects`), `repo:onnx-mlir@b505d64:src/Dialect/ONNX/Transforms/CMakeLists.txt` (`DEPENDS OMONNXTransformsPassIncGen` on `OMFusionOpHelper`, `OMFusionOpTransform`, `OMInstrumentONNX`, `OMExpandAttentionMask`, `OMAppendDecodingStrategy`), `repo:onnx-mlir@b505d64:src/Pass/Passes.hpp` ("GEN_PASS_DEF method only adds default constructor only" comments), `repo:onnx-mlir@b505d64:src/Accelerators/NNPA/Compiler/NNPACompilerUtils.cpp`, `repo:onnx-mlir@b505d64:src/Tools/onnx-mlir-opt/RegisterPasses.cpp` (per-pass lambdas; `::registerTransformsPasses()`, `::registerONNXToLinalgPasses()`), `repo:onnx-mlir@b505d64:MLIR.cmake` (`add_onnx_mlir_pass`), `repo:onnx-mlir@b505d64:src/Transform/CMakeLists.txt` and `src/Transform/PassesKrnl.td` (the fourth `.td`, missing from the original grounding), `repo:onnx-mlir@b505d64:src/Conversion/ONNXToLinalg/CMakeLists.txt` and `Passes.td` (`linalgOps`/`useLinalgPath` options), and the five legacy `PassWrapper` files listed below
- **Originally grounded against** (at `aeb55f8f`): `src/Dialect/ONNX/Transforms/Passes.td`, `src/Dialect/ONNX/Transforms/AppendDecodingStrategy.cpp`, `src/Dialect/ONNX/Transforms/AppendInstrumentedOutputsPass.cpp`, `src/Conversion/ONNXToLinalg/Passes.td`, `src/Transform/Passes.td`, `src/Pass/Passes.hpp`, `src/Tools/onnx-mlir-opt/RegisterPasses.cpp`, `src/Compiler/CompilerPasses.cpp`, `MLIR.cmake` (`add_onnx_mlir_pass`), `src/Dialect/ONNX/Transforms/CMakeLists.txt`
- **Files read:** all of the above, plus a `PassWrapper`-based cross-check on `Decompose.cpp`, `ConvOpt.cpp`, `ShapeInferencePass.cpp`, `ONNXHybridTransformPass.cpp`, `ReplaceOpWithItsOperand.cpp` (confirming which existing passes are the legacy pattern, not the template)
- **How to test:** after adding a pass per Steps 1–6, `onnx-mlir-opt --help` should list `--{{CLI_NAME}}`; running it on a minimal `.mlir` file exercising the pass's `runOnOperation()` should produce the expected rewrite with no build errors from a missing `DEPENDS` on the `...PassIncGen` target
