---
name: ground-lit-tests
description: 'Use when writing or updating FileCheck lit tests for an onnx-mlir pass: prove numerically that the pass output is correct with GroundLitTest.py before freezing CHECK lines with fixLitTest.py, and keep RUN-line flags minimal.'
---

<!-- Generated from the onnx-mlir knowledge base (skills/ground-lit-tests.md), verified against onnx-mlir@b505d64b. Do not edit here: see .agents/skills/README.md. -->

# Ground a Lit Test in an Actual Run

Walkthrough for producing FileCheck tests you can actually trust when you write or change a pass: establish numerically that the pass's output is *correct*, then freeze that output into CHECK lines. This is the terminal step of [add-onnx-op-lowering.md](../../../.agents/skills/add-onnx-op-lowering/SKILL.md) and [add-onnx-transform-pass.md](../../../.agents/skills/add-onnx-transform-pass/SKILL.md), both of which stop at "run it and confirm it does what's expected."

`docs/GroundedLitTests.md` in the repo is the authority for the full detail — the scenario matrix, the minimal-diff review, the gotchas table. This page exists to sequence the tools, state which decisions come first, and connect the workflow to the rest of this wiki. Read it alongside, not instead.

---

## Why the order matters

**A FileCheck test asserts the shape of a pass's output, never its correctness.** `mlir2FileCheck.py` will faithfully turn wrong IR into a green test, and `docs/Testing.md`'s recipe of "manually check whether the output is correct" is exactly where correctness bugs slip through — eyeballing a lowered loop nest does not scale to the dozens of cases a pass change needs.

So the order is fixed, and inverting it silently destroys the test's value:

```
1. GroundLitTest.py   → prove it computes the right numbers   (correctness)
2. fixLitTest.py -r   → freeze the output as CHECK lines      (shape)
3. llvm-lit           → the suite
```

Freeze first and a green test tells you nothing: you have pinned whatever the compiler happened to emit. The only precondition is that the pass's output can be compiled and run — anything at or above ONNX-to-Krnl qualifies.

## Step 0: Pick the grounding mode

Decide once, up front; it sets the cost of every case in the file.

| Mode | Use when | Command |
|---|---|---|
| **Flag mode** (prefer) | A compiler flag switches the behavior on and off — a new optimization, an `-O3`-only path, an accelerator option, a pass option. **Almost always available for ONNX-to-Krnl work**, where the input is ONNX dialect and compiles end to end, making `-O0` vs `-O3` a ready-made oracle | `GroundLitTest.py -f my_func -r=--O0 -t=--O3 mytest.mlir` |
| **File mode** | No flag expresses the difference — e.g. a new Krnl op, where the baseline is the same loop nest written without it | `GroundLitTest.py -f my_func mytest.mlir` (baseline found by name) |

Write option values with `=` (`-r=--O0`, not `-r --O0`); argparse otherwise consumes the bare leading `--` and the run dies with `expected one argument`.

File mode's baseline is `<test>.baseline<ext>` next to the test, holding a same-named function computing the same thing through a path you already trust. It needs no `// RUN:` line and no CHECK lines — the only thing that ever reads it is `fixLitTest.py -m`, which copies the function out as text. `test/mlir/lit.cfg.py` excludes every `.baseline<ext>` file from collection, so it never shows up as an `UNRESOLVED` test.

Two guards worth knowing before you write a baseline: comments are stripped from both variants before comparison, and **two variants that are identical modulo comments are reported as a FAILURE, not a pass** — nothing was compared, so a "PASS" would be a lie. This catches the likeliest mistake, a copy-pasted baseline that was never edited.

## Step 1: Write the test function where it will ship

One `// -----`-separated segment per scenario, in the final file. Give it a real signature: concrete element types, and `{onnx.name = "..."}` on arguments and results. It has to be *isolatable and runnable*, not merely parseable — `GroundLitTest.py` isolates it via `fixLitTest.py -m` and actually executes it.

## Step 1b: Keep the RUN line minimal

Unlike the rest of this page, this section applies to **any** new lit test, not just one you are grounding — a canonicalization test or a `-verify-diagnostics` negative test never runs numerically, and still gets its RUN line composed the same way. Settle the pipeline before Step 2, since it is what you will be grounding.

Start from the shortest RUN line that exercises the thing under test and **add a flag only when you can say what breaks without it.** For an ONNX-to-Krnl test that is four flags — and this is the single most common RUN line in the suite, carried verbatim by 45 files at `b505d64`, e.g. `test/mlir/conversion/onnx_to_krnl/Math/Elementwise_with_canonicalize.mlir`:

```
// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s
```

(`-split-input-file` because scenarios are `// -----`-separated, per Step 1.) A further 32 files carry that line minus `--canonicalize` — `.../Math/Elementwise.mlir` is the plain sibling — but see the canonicalize exception below before choosing that form for a Krnl-level test.

Note what is *absent* from the most common line in a repo whose main target is IBM Z: no `--march`, no `--mtriple`, no `-O3`. The failure mode is copying a neighbouring test's RUN line wholesale — every flag in it was justified by *that* test, and nobody rechecks the copy.

**Pin a target only when the target is what's under test.** A CPU op lowering with nothing architecture-specific about it should set no `-march`/`--mtriple` at all — the output does not depend on them, the test misleadingly reads as being about Z, and it needlessly narrows where the test means anything. More generic is better. Target flags belong in an NNPA test, or a SIMD test where the vector width is the point, which is where the suite already puts them: of the 200 test files mentioning `-march` at `b505d64`, 164 are under `test/mlir/accelerators/`, and in `test/mlir/conversion/onnx_to_krnl/` 99 of 130 files pin no target. The ones that legitimately do are the accelerator tests, which pair the target with the accelerator being tested (`--march=z16 --maccel=NNPA ...`, 19 files), and the `-O3` SIMD variants, where vector width is the point:

```
// RUN: onnx-mlir-opt -O3 --mtriple=s390x-ibm-loz --march=z16 --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s
```

**For a SIMD test specifically, `-O3` and `--march` are a conjunction, and getting it wrong fails silently.** SIMD passes through two independent gates ([stack/compiler.md](../../../.agents/skills/onnx-mlir-internals/references/compiler.md#performance-relevant-flag-dependencies)): an optimization-level gate — `onnx-mlir-opt` sets `enableSIMD` from `optLevel >= 3` (`src/Tools/onnx-mlir-opt/RegisterPasses.cpp`) — and a recognized-target gate, since `setGlobalVectorMachineSupport()` installs `NoVectorMachineSupport` for any `--march` it does not recognize, and `--march` defaults to the empty string. Pass only `-O3` and you get no vectorization; pass only `--march` and you get no vectorization. Neither case errors: the pass runs, emits scalar code, and `fixLitTest.py` cheerfully freezes that into green CHECK lines, so you end up with a "SIMD test" that proves nothing. Confirm the output actually contains `vector.*` ops before generating assertions. `--mtriple` is *not* part of this — the z16 tests carry it by convention, but `test/mlir/parallel/scf_parallel_private.mlir` vectorizes with `-O3 --march=x86-64` alone.

**Use as few optimization passes as practical**, so the test exercises the new optimization as purely as possible. Every additional pass is another candidate explanation for the IR you are checking — it slows attribution when a case fails, and lets the CHECK lines drift for reasons unrelated to your change.

**The standing exception is a canonicalization pass after lowering to Krnl.** Lowering introduces a lot of needless replication — redundant index arithmetic, dead casts, repeated bounds — and `--canonicalize` clears it, so the CHECK lines describe the lowering under test rather than the noise around it. Treat it as the default at Krnl level, not an extra: 145 of the 189 RUN lines invoking `--convert-onnx-to-krnl` already follow it with `--canonicalize`. A canonicalize *before* lowering is a different flag needing its own justification — exactly one test does it, `test/mlir/conversion/onnx_to_krnl/Tensor/Transpose_neg_pattern_with_canonicalize.mlir`, because the ONNX-level rewrite is itself what that test covers.

One self-check: the suite announces extra flags in the filename, and `Elementwise` carries all three variants — `Elementwise.mlir` (no canonicalize), `Elementwise_with_canonicalize.mlir` (the common default), `Elementwise_with_canonicalize_O3.mlir` (adds `-O3` and a target because SIMD is the point). If you cannot justify the suffix you would have to add, you probably do not need the flag.

## Step 2: Ground it, and iterate here

```bash
GroundLitTest.py -f my_func [mode options] mytest.mlir
```

On failure, re-run with `-d` (side-by-side body diff; picks up `icdiff` if installed) and `-v`. The `glt_test.mlir`, `glt_baseline.mlir` and `glt_ref/` left in the working directory are *what actually ran* — inspect those rather than re-deriving. Note only a single-`-f` run keeps them; an all-functions run clears them, since only the last function's leftovers would survive.

Once it passes, move the options it needed into the file, so the knowledge outlives your shell history:

```mlir
// Defaults for every function in this file.
// GROUND-ALL: -c="-O3 -parallel"

// Defaults for the next function only.
// GROUND-THIS: -shape-info=0:10x20
func.func @my_dynamic_case(%arg0: memref<?x?xf32> {onnx.name = "x"}) -> ...
```

Precedence is **per option**, most specific winning: command line > `GROUND-THIS` > `GROUND-ALL` > built-in default. A directive line is options and nothing else — a trailing `// like this` explanation on the same line is parsed as arguments and rejected. A directive may not name a file or set `-f` (a file doesn't get to choose which of its functions is tested), and only the test file is scanned — a baseline's directives are ignored.

This is what makes the whole file runnable by one command: **without `-f`, every function is grounded in turn**, ending in a `Succeeded`/`Failed` summary. That whole-file run is the acceptance gate.

## Step 3: Only now, freeze the assertions

```bash
fixLitTest.py -p -r -f my_func mytest.mlir > tmp.mlir && mv tmp.mlir mytest.mlir
```

`-r -f <fn>` regenerates that one function unconditionally (bare `-r` only repairs functions whose test *fails*); `-p` prints the others unchanged. Never redirect into the file you are reading — write a temp and move. Then `fixLitTest.py -t mytest.mlir` self-tests each function individually, which localizes a failure far better than a whole-file FileCheck run.

See [stack/tools.md](../../../docs/Testing.md) for the full option table, including the multi-prefix rule (a function carrying only *another* prefix's CHECK lines is deliberately skipped, never injected into) and the column-0 requirement for existing CHECK lines.

## Step 4: Make the comparison have teeth

A numerical comparison only catches what the computation is sensitive to. The choices that turn a passing run into real evidence:

- **Make the result depend on the indices**, not just the values. `out[i][j] = in[i][j] * in[i][j]` cannot tell a correct index mapping from a transposed one; adding an index-derived term (`out[i][j] = in[i][j] + (i*N + j)`) makes any mis-ordering a numerical difference. This is what catches an off-by-one or a swapped dimension.
- **Use asymmetric, non-square shapes** (`10x20`, not `8x8`), so a swapped subscript is out of range rather than merely different. Avoid commutative-looking bodies for the same reason.
- **Tighten the tolerance** when the arithmetic is exact. The defaults (`rtol=0.05`, `atol=0.01`) are loose enough to hide a small systematic error.
- **Pin the inputs** with `--seed` when chasing a specific failure, and `--lower-bound`/`--upper-bound`/`--input-value` when the default random range makes the output uninteresting (all-negative input into a ReLU, say).
- **In file mode, keep the baseline differing only where the feature requires it**, and read the `-d` diff line by line to confirm — for a new pair, before generating any CHECK lines. The diff is the feature's specification and the first thing a reviewer reads. Gratuitous differences (a constant hoisted in one variant, renamed SSA values, an unneeded shape) should be brought into line; arithmetic the baseline performs *by hand to derive what the compiler is supposed to produce* is essential and legitimately makes some diffs large — say so in a comment so nobody "cleans it up." `docs/GroundedLitTests.md`'s "Read the `-d` diff" section has the full keep/fix lists, and points at the worked `krnl_to_affine_collapse.mlir` family as the example to copy (see "Committed examples" below).

## Step 5: Get breadth deliberately

Coverage comes from crossing axes, not from writing many similar cases. `docs/GroundedLitTests.md`'s scenario matrix is the full list (rank, shape knowledge, sizes, element type, broadcasting, attributes, composition with neighbouring transforms, rejections). Three parts of it are worth repeating here because they are the ones most often skipped:

- **Dynamic dimensions are mandatory, not optional.** Cover all three forms — all-dynamic, all-static, and **mixed** (a third case, not a midpoint: the generated maps combine constants and symbols, and code paths special-casing "all static" or "all dynamic" fall between them). A fully static suite can be entirely green while the pass is broken for every real model, because a static shape constant-folds away the very code you need to exercise: runtime trip counts whose computation must *dominate* every use, and affine maps carrying `symbol(%dim)` instead of a literal, which have extra validity rules. Ground at least one dynamic case at more than one shape, including a degenerate `1` (which catches loops assuming more than one iteration) and a size that is not a multiple of any tile or vector length.
- **When changing an existing pass, cover what you did *not* mean to change.** Ground at least one case with the new behavior *off*, over the same code you touched. Then find the existing tests covering that code (`grep` the pass name under `test/mlir`, or the ops your transform matches) and read their *diffs*, not just their pass/fail — a test that still passes but whose output changed is telling you something, and a case whose output changed with *no* test is the one that will bite a user.
- **Every documented restriction needs a negative test**, or the restriction is a comment. These are cheap, since nothing runs: a `-verify-diagnostics` RUN line plus `// expected-error @+1 {{distinctive fragment}}`. Match a substring so wording can improve. Prefer a real diagnostic over an `assert`, which is compiled out of release builds — a rejection that only asserts is a silent miscompile for release users.

## Step 6: Done means

- `GroundLitTest.py <file>` with no other options reports `Failed (0)` — every function grounded, each case's options living in its own directives.
- CHECK lines were generated *after* that passed.
- Dynamic dims covered in all three forms, one case at more than one shape.
- Every documented restriction has a negative test.
- `fixLitTest.py -t <file>` is clean.
- The full suite is green: `llvm-lit build/test/mlir` or `make check-onnx-lit`.
- The committed file is exactly what the tools generate. Hand-editing generated CHECK lines is a generator bug worth fixing — a hand-patched file silently stops being regenerable.

## Committed examples

Four test files in the tree are grounded with this workflow; copy the one closest to your case:

| File | Mode | Directives |
|---|---|---|
| `test/mlir/conversion/krnl_to_affine/krnl_to_affine_collapse.mlir` + `krnl_to_affine_collapse.baseline.mlir` | **File mode**: each baseline function computes the same thing with an ordinary loop nest, without `krnl.collapse` | `GROUND-ALL: -c="-O3 -parallel"`, `GROUND-THIS: -shape-info=0:10x20` on the dynamic cases |
| `test/mlir/conversion/onnx_to_krnl/parallel_collapse.mlir` | **Flag mode**: collapse on vs. off | `GROUND-ALL: -c="-O3 --parallel" -r="--disable-collapse"`, one `GROUND-THIS: --shape-info=...` per function |
| `test/mlir/onnx/onnx_recompose_split_matmul_by_slice.mlir` | **Flag mode**, ONNX-level rewrite | `GROUND-ALL: -r="--disable-split-matmul-by-slice"`, one `GROUND-THIS: --shape-info=...` per function |
| `test/mlir/onnx/onnx_recompose_unbind_split.mlir` | **Flag mode**, ONNX-level rewrite | `GROUND-ALL: -r="--disable-unbind-to-split"`; a `GROUND-THIS: --shape-info=...` only on the one function with a dynamic input |

The flag-mode files show the usual shape: `-c` carries what both runs need, and `-r` adds the one flag that turns the feature off, so the test run is the feature and the reference run is the code without it. `parallel_collapse.mlir` also has a second RUN line with a `NOCOLLAPSE` prefix, covering the same input with the feature off, as Step 5 asks.

The baseline file has no `// RUN:` line. Each of its functions keeps a `CHECK-LABEL` in its own segment, so a plain `fixLitTest.py -r` does not treat it as a new test and fill it with assertions. Two stale details in these files' header comments: they show `GroundLitTest.py -m <file>` and a `-baseline` file name, but the tool takes the file as a positional argument and looks for `<test>.baseline<ext>`. Run `GroundLitTest.py test/mlir/conversion/krnl_to_affine/krnl_to_affine_collapse.mlir`.

## When grounding fails and you don't know why

This workflow proves *whether* the output is right. Diagnosing *why* it isn't is the other skill: [debug-compilation.md](../../../.agents/skills/debug-compilation/SKILL.md) for the `--EmitONNXIR`/`onnx-mlir-opt`/pass-isolation loop, and [debug-type-catalogue.md](../../../.agents/skills/debug-type-catalogue/SKILL.md) to find your pass's `-debug-only=<DEBUG_TYPE>` channel and read its own trace. For what the output *should* look like at a given boundary, the interface pages are the reference: [onnx-to-krnl.md](../../../.agents/skills/onnx-mlir-internals/references/onnx-to-krnl.md), [krnl-to-affine.md](../../../.agents/skills/onnx-mlir-internals/references/krnl-to-affine.md).

Two build-level traps that look like pass bugs: `onnx-mlir` and `onnx-mlir-opt` are **separate targets**, so a stale binary reports an error in code you already fixed — rebuild both. And `--shape-info` (a run-time option of `RunONNXModel.py`, choosing the shape used to generate inputs) is not `--shapeInformation` (a compiler option rewriting ONNX graph inputs during shape inference, which does nothing for an already-lowered module). `docs/GroundedLitTests.md`'s gotchas table covers the rest.

---

## Verification

- **Re-checked at `b505d64`:** the Step 1b counts (all re-run, numbers below are current) and the "Committed examples" searches (one new grounded file, `test/mlir/onnx/onnx_recompose_unbind_split.mlir`, directive and RUN lines read). The tool/doc files below were not changed by the `5534bc7..b505d64` commits.
- **Grounded against:** `repo:onnx-mlir@b505d64:docs/GroundedLitTests.md`, `repo:onnx-mlir@b505d64:utils/GroundLitTest.py`, `repo:onnx-mlir@b505d64:utils/fixLitTest.py`, `repo:onnx-mlir@b505d64:test/mlir/lit.cfg.py` (still excludes `.baseline` files; now uses `lit.formats.ShTest()` with no `execute_external` argument), for "Committed examples" `repo:onnx-mlir@b505d64:test/mlir/conversion/krnl_to_affine/krnl_to_affine_collapse.mlir`/`.baseline.mlir`, `repo:onnx-mlir@b505d64:test/mlir/conversion/onnx_to_krnl/parallel_collapse.mlir`, `repo:onnx-mlir@b505d64:test/mlir/onnx/onnx_recompose_split_matmul_by_slice.mlir` (directive lines and RUN lines read), and for Step 1b `repo:onnx-mlir@b505d64:test/mlir/` in aggregate — specifically `test/mlir/conversion/onnx_to_krnl/Math/Elementwise.mlir`, `.../Math/Elementwise_with_canonicalize_O3.mlir`, `.../Tensor/Transpose_neg_pattern_with_canonicalize.mlir` (RUN lines quoted verbatim)
- **Step 1b's authority is split, deliberately.** The *rule* — minimize flags, justify each, prefer generic, few optimization passes — is maintainer convention, not something derived from code: `docs/Testing.md` and `docs/GroundedLitTests.md` say nothing about flag choice. What is grounded is that current practice already matches it. Re-verify by rerunning the searches, not by diffing a file list (counts as of `b505d64`, from a full checkout at that commit):
  ```bash
  grep -rl -e '-march' test/mlir --include='*.mlir' | wc -l                        # 200
  grep -rl -e '-march' test/mlir/accelerators --include='*.mlir' | wc -l           # 164
  find test/mlir/conversion/onnx_to_krnl -name '*.mlir' | wc -l                    # 130
  grep -rl -e '-march' test/mlir/conversion/onnx_to_krnl --include='*.mlir' | wc -l # 31 -> 99 pin no target
  grep -rh 'RUN:.*convert-onnx-to-krnl' test/mlir --include='*.mlir' | wc -l       # 189
  grep -rh 'RUN:.*convert-onnx-to-krnl.*canonicalize' test/mlir --include='*.mlir' | wc -l  # 145
  grep -rn 'RUN:.*canonicalize.*convert-onnx-to-krnl' test/mlir --include='*.mlir' # 1 (Transpose_neg_pattern)
  # Frequency ranking of the distinct Krnl RUN lines -- the top row is the form quoted above.
  grep -rh 'RUN:.*convert-onnx-to-krnl' test/mlir --include='*.mlir' | sed 's/  */ /g' | sort | uniq -c | sort -rn | head -5
  #   45  --shape-inference --convert-onnx-to-krnl --canonicalize        (no target)
  #   32  --shape-inference --convert-onnx-to-krnl                       (no target, no canonicalize)
  #   19  --march=z16 --maccel=NNPA --shape-inference ... --canonicalize  (target under test)
  #   13  --march=z16 --maccel=NNPA --convert-onnx-to-krnl --canonicalize (target under test)
  #    9  -O3 --mtriple=s390x-ibm-loz --march=z16 ... --canonicalize      (SIMD width under test)
  ```
  The frequency ranking is the load-bearing evidence for Step 1b, more than the aggregate counts: the modal RUN line in a repo whose main target is IBM Z pins no target and does run `--canonicalize`, which is the convention exactly.
- **Files read:** `docs/GroundedLitTests.md` in full. `utils/GroundLitTest.py`: the `add_argument` definitions, `DEFAULTS` (confirming `rtol=0.05`/`atol=0.01`), `GLT_TEST_FILE`/`GLT_BASELINE_FILE`/`GLT_REF_DIR` (confirming the literal `glt_test.mlir`/`glt_baseline.mlir`/`glt_ref` names), `BASELINE_TAG`/`default_baseline_path()`, `GROUND_ALL`/`GROUND_ONE` directive parsing and its documented precedence layering, `strip_comments()` and the identical-variants `FAIL` path, the `fixLitTest.py -m` invocations used to isolate both the test and the baseline function, the `shutil.rmtree(GLT_REF_DIR)` cleanup, and the `Succeeded`/`Failed` summary. `utils/fixLitTest.py`: `print_usage()`, the `-m`/`-r`/`-p`/`-t`/`-f` option branches, `emit_module_for_function()`, and the `segment_has_check_lines` multi-prefix guard. `test/mlir/lit.cfg.py`: the `BASELINE_TAG` `config.excludes` set comprehension.
- **How to test:** on a build with **both** `onnx-mlir` and `onnx-mlir-opt`, run `GroundLitTest.py test/mlir/conversion/onnx_to_krnl/parallel_collapse.mlir` (flag mode) and `GroundLitTest.py test/mlir/conversion/krnl_to_affine/krnl_to_affine_collapse.mlir` (file mode) from the onnx-mlir checkout root, and confirm each ends in a `Succeeded`/`Failed (0)` summary with the options coming only from the files' directives. To exercise the tool on a fresh file: copy one function out of `test/mlir/conversion/krnl_to_affine/krnl_to_affine_matmul.mlir` (verified to carry the `{onnx.name = "..."}`-annotated signature grounding requires) into a scratch file, then flag-mode ground it with `GroundLitTest.py -f <func> -r=--O0 -t=--O3 scratch.mlir` and confirm a pass plus a `Succeeded`/`Failed` summary. Re-run without `-f` to confirm the whole-file gate. Then verify the two guards this page relies on: pass `-d` on a single-`-f` run and confirm `glt_test.mlir`/`glt_baseline.mlir`/`glt_ref/` are left behind (they are cleared on an all-functions run), and copy the function verbatim into `scratch.baseline.mlir` to confirm file mode reports `FAIL: ... is identical (ignoring comments)` rather than passing.
- **Known gaps:**
  - This page deliberately does not reproduce `docs/GroundedLitTests.md`'s full scenario matrix or gotchas table — it cites them. If that doc is restructured, the section names referenced here ("Read the `-d` diff", "Gotchas") need re-checking on Repository Sync. The tool behaviors above are grounded in `GroundLitTest.py`/`fixLitTest.py`/`lit.cfg.py` directly and are unaffected by doc edits.
  - "Committed examples" lists the grounded files found by `grep -rl 'GROUND-ALL\|GROUND-THIS' test` and `find test -name '*.baseline*'` at `b505d64`. Rerun both searches on a later sync rather than diffing the table.

## Related Pages

- [stack/tools.md](../../../docs/GroundedLitTests.md) — `GroundLitTest.py`/`fixLitTest.py` option reference
- [skills/add-onnx-op-lowering.md](../../../.agents/skills/add-onnx-op-lowering/SKILL.md) — the recipe this workflow terminates
- [skills/add-onnx-transform-pass.md](../../../.agents/skills/add-onnx-transform-pass/SKILL.md) — likewise, for a new pass
- [skills/debug-compilation.md](../../../.agents/skills/debug-compilation/SKILL.md) — diagnosing *why* a grounding run fails
- [skills/coding-conventions.md](../../../.agents/skills/coding-conventions/SKILL.md) — what CI enforces on the code under test
