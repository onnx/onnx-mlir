<!--- SPDX-License-Identifier: Apache-2.0 -->

# Grounded lit tests: testing a pass you are changing

How to build a large body of *trustworthy* FileCheck tests for compiler work,
using `utils/GroundLitTest.py`, `utils/fixLitTest.py`, `utils/mlir2FileCheck.py`
and `llvm-lit`.

This applies to **any change to a pass**, not just a pass written from scratch --
which is the rare case. It is just as applicable, and the workflow is identical,
when you:

- write a new pass;
- add an optimization, a new op, or a new special case to an **existing** pass
  (the common case, and the one that motivated this document: `krnl.collapse` was
  added inside the existing `ConvertKrnlToAffine`);
- extend an existing lowering to shapes, ranks or types it did not handle;
- fix a miscompile, where the regression test needs to be grounded for exactly
  the same reason.

The only real precondition is that the pass's output can be compiled and run, so
that there is something to compare numbers against. Anything at or above the level
your pipeline can execute end to end qualifies -- ONNX-to-KRNL, KRNL-to-Affine,
and everything downstream.

[Testing.md](Testing.md) covers the mechanics of FileCheck tests. This document
covers the part that mechanics alone cannot give you: evidence that the output
you are freezing into assertions is *correct*.

## The one rule

**A FileCheck test asserts the shape of the output, never its correctness.**
`mlir2FileCheck.py` will faithfully turn wrong IR into a green test. Testing.md's
FileCheck recipe says "manually check whether the output is correct" -- eyeballing
a lowered loop nest is exactly where correctness bugs slip through, and it does
not scale to the dozens of cases a pass change needs to be trusted.

So the order is always:

> **Ground the behavior by execution first. Freeze the assertions second.**

`GroundLitTest.py` does the grounding: it isolates one function, compiles it two
ways, feeds both the same inputs, and compares outputs numerically. Only once
that passes have you earned the right to record CHECK lines. If you freeze first
and ground later, a green test tells you nothing -- you have pinned whatever the
compiler happened to emit.

It names the file it grounds the way `fixLitTest.py` does, as the one positional
argument:

```
GroundLitTest.py [options] <lit-test-filename>
```

## Prerequisites

Build **both** driver binaries. They are separate targets, and the two tools use
different ones:

```
make -C build -j onnx-mlir-opt onnx-mlir
```

| Tool | Binary it uses |
|---|---|
| `llvm-lit`, `fixLitTest.py`, `mlir2FileCheck.py` | `onnx-mlir-opt` (runs the RUN line) |
| `GroundLitTest.py` | `onnx-mlir` (compiles and runs a real model) |

A stale `onnx-mlir` is a genuine time sink: grounding fails with a compile error
that reflects code you already changed, which reads like a bug in your pass.
**When a grounding run fails for a reason that makes no sense, rebuild `onnx-mlir`
before debugging anything else.**

`GroundLitTest.py` also needs `ONNX_MLIR_HOME` set to the build's install root
(e.g. `build/Debug`), and a Python interpreter with the `onnx` package available,
since it drives `RunONNXModel.py`.

## Pick a grounding mode first

This choice decides how much work each new test costs, so make it once, up front.

### Flag mode -- prefer this

Compile the **same file** twice with different `onnx-mlir` options. No baseline
file to write, so each new test case is just one function:

```
GroundLitTest.py -f my_func -r=--O0 -t=--O3 mytest.mlir
```

`-r` is the reference options, `-t` the test options; `-a` adds to the reference
instead of replacing it. Use flag mode whenever the behavior you are testing can
be turned on and off with a compiler flag -- a new optimization, an `--O3`-only
path, an accelerator option, or a pass option. **For ONNX-to-KRNL work this is
almost always available**, because the test input is ONNX dialect and therefore
compiles end to end on its own; `-O0` versus `-O3` is a ready-made oracle.

Write the option values with `=` (`-r=--O0`, not `-r --O0`). Without it, argparse
consumes the leading `--` as an option of its own and the run dies with
`expected one argument`. Same for `--compile-args=--parallel`.

### File mode -- when no flag expresses the difference

Write a second `.mlir` file holding a function with the **same name** and the
**same computation**, expressed through a path you already trust:

```
GroundLitTest.py -f my_func mytest.mlir        # baseline found automatically
GroundLitTest.py -f my_func -d mytest.mlir     # plus a side-by-side body diff
```

The baseline defaults to `<test>.baseline<ext>` next to the test file, so
`mytest.baseline.mlir` is picked up without `-b`. Use file mode when the variants
are genuinely different source and no single flag switches between them -- for
example a new KRNL op, where the baseline is the same loop nest written without
it. `-d` prints the two isolated bodies side by side, which doubles as a readable
summary of exactly what your feature changes.

Keep the baseline file for as long as the feature is under active development.
Whether it also ships as a committed regression artifact is a separate call;
generally there are benefits in keeping such artifact for the long run. Either
way it can sit next to the test it grounds: the `.baseline` tag is what keeps it
out of the lit run. `test/mlir/lit.cfg.py` collects every `.mlir`, `.json` and
`.onnxtext` it finds *except* those whose name ends in `.baseline<ext>`, so a
baseline never appears as a test. It needs nothing but its functions: no `// RUN:`
line and no CHECK lines, since `fixLitTest.py -m` -- the only thing that ever
reads it -- copies the function out as text rather than running it.

The flip side is that nothing in the lit suite parses a baseline any more, so a
long-lived one is only checked when someone next grounds against it. That is the
trade for a green `check-onnx-lit`, and it is why the `-d` diff is worth
re-reading whenever you come back to an old pair.

In file mode the two variants must genuinely differ: comments are stripped from
both before comparing, and two variants that are the same MLIR modulo comments
are reported as a **FAILURE**, not a pass. Nothing was actually compared, so a
"PASS" would be a lie. This is the guard against the most likely mistake -- a
baseline copy-pasted from the test and then not edited.

### Read the `-d` diff, and make it minimal

The tool only checks that the two variants differ *somehow*. Keeping them
differing **only** where the feature requires is your job. **When adding a new
file-mode pair, run `-d` and read the diff line by line before generating any
CHECK lines.** For each differing line, ask: *is this line different because the
feature demands it?* If not, change the baseline until it is. Install `icdiff` for
a side-by-side view; `-d` picks it up automatically.

A gratuitous difference does not make the comparison wrong, so it is worth being
clear on why this earns the effort:

- **The diff is the feature's specification.** For a new op, `-d` is the clearest
  statement anywhere of what the op changes, and the first thing a reviewer reads.
  Noise buries the signal.
- **It keeps failures pointed at one thing.** Every unrelated difference is also
  being tested, so a variant that differs in four places takes four times as long
  to bisect when it breaks.
- **A large diff is a prompt to check the test's scope.** The usual cause is one
  function testing two things; splitting it gives sharper failures for the same
  coverage. A prompt, not a verdict -- see the essential list below.

Almost always gratuitous, so fix the baseline:

- a constant, `memref.dim` or `memref.alloc` declared at a different point --
  hoisted in one variant, in the loop body in the other;
- different SSA names for the same value, or a different order for independent
  operations;
- a memref shape, element type, or bound spelling the feature does not require;
- arithmetic in one variant that the other does not need.

Essential, so keep:

- the ops the feature adds or removes, and any whose operands or result counts it
  changes -- the point of the test;
- arithmetic the baseline performs **by hand to derive what the compiler is
  supposed to produce**. Whenever the test asks the compiler for a value, the
  baseline must reach that value another way or there is nothing to compare
  against. This legitimately makes some diffs large, and shrinking them would
  delete the comparison. Say so in a comment so the next reader does not "clean it
  up".

What minimal looks like -- the whole diff for one function of a new-KRNL-op
family, where the op replaces a loop reference and changes a query's result count:

```
-  krnl.iterate(%ii, %jj) with (%ii -> %i = 0 to 10, %jj -> %j = 0 to 20) {
-    %a, %b = krnl.get_induction_var_value(%ii, %jj) : (!krnl.loop, !krnl.loop) -> (index, index)
+  %ff = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
+  krnl.iterate(%ff) with (%ii -> %i = 0 to 10, %jj -> %j = 0 to 20) {
+    %a, %b = krnl.get_induction_var_value(%ff) : (!krnl.loop) -> (index, index)
```

`krnl_to_affine_collapse.mlir` and its baseline are a worked family to copy the
shape from, including one function whose diff is deliberately large for the
by-hand-derivation reason above.

For a new case that follows an existing pair's shape, a minimal diff comes for
free and re-reading it is not worth the time. Do the review when the pair is new,
or when you have reshaped an existing one.

## Let the file carry its own options

Most cases need something to run: a shape for a dynamic input, a pipeline flag, a
tighter tolerance. Put those in the file rather than in your shell history,
using directives that `GroundLitTest.py` reads out of the test file:

```mlir
// Defaults for every function in this file.
// GROUND-ALL: -c="-O3 -parallel"

// Defaults for the next function only.
// GROUND-THIS: -shape-info=0:10x20
func.func @my_dynamic_case(%arg0: memref<?x?xf32> {onnx.name = "x"}) -> ...
```

A directive line is options and nothing else: everything after the `:` is parsed,
so a trailing `// like this` explanation is read as arguments and rejected. Put
the prose on its own comment line, as above.

Both directives take this tool's own options, in either the `--flag` or
single-dash `-flag` spelling, and both may be repeated across several lines.
Precedence is **per option**, most specific winning:

```
command line  >  GROUND-THIS  >  GROUND-ALL  >  built-in default
```

so a `GROUND-THIS` that sets only `--shape-info` still inherits `GROUND-ALL`'s
`--compile-args`, and anything you type still overrides both. A directive may
neither name a file nor set `-f` -- a file does not get to name itself or choose
which of its functions is tested. Only the test file is scanned; a baseline
file's directives are ignored. A value starting with `-` needs the
`--flag=value` form.

This is what makes the whole suite runnable by one command, and it is the reason
to do it: **without `-f`, every function in the file is grounded in turn**, each
reporting as a single `-f` run would, followed by a pass/fail summary.

```
GroundLitTest.py mytest.mlir
...
=== summary ===
Options: --compile-args=-O3 -parallel (GROUND-ALL)
Succeeded (9): collapse_base, collapse_then_parallel, ...
Failed    (0): <none>
```

Treat that as the acceptance gate for the file. It also means nobody has to
rediscover, months later, that one function only means anything at
`--shape-info 0:10x20`. Note a single-function run leaves its `glt_*` files
behind for inspection, while an all-functions run clears them -- only one
function's leftovers would survive, which would be misleading.

## The loop, per scenario

1. **Write the test function in its final file**, the one that will ship. Give it
   a real signature with concrete element types and `{onnx.name = "..."}` on the
   arguments and results, because it has to be isolatable and runnable, not just
   parseable. One `// -----`-separated segment per scenario.

2. **Set up the oracle.** Flag mode: nothing to do. File mode: add the
   same-named function to `<test>.baseline<ext>`.

3. **Ground it.** Iterate here, not later:
   ```
   GroundLitTest.py -f my_func [mode options] mytest.mlir
   ```
   Once it passes, move whatever options it needed into a `// GROUND-THIS:` line
   above the function (or `// GROUND-ALL:` if the whole file needs them), so the
   knowledge lives in the file. On failure, re-run with `-d` and `-v`; the kept
   `glt_test.mlir` / `glt_baseline.mlir` / `glt_ref/` in the working directory are
   what actually ran, so inspect those rather than re-deriving.

4. **Freeze the assertions**, only now:
   ```
   fixLitTest.py -p -r -f my_func mytest.mlir > tmp.mlir && mv tmp.mlir mytest.mlir
   ```
   `-r -f <fn>` regenerates that one function unconditionally; `-p` prints the
   others unchanged. A function with no CHECK lines yet gets a full set generated;
   a function carrying only *another* prefix's CHECK lines is deliberately left
   alone. Never redirect into the file you are reading -- write a temp and move.

5. **Confirm.** `fixLitTest.py -t mytest.mlir` self-tests each function
   individually, which localizes a failure far better than a whole-file
   FileCheck. Then run it for real through lit.

6. **Repeat** for the next scenario, and run the full suite before you stop.

## Make the comparison have teeth

A numerical comparison only catches what the computation is sensitive to. These
choices are what turn a passing grounding run into real evidence:

- **Make the result depend on the indices**, not just the values. A body that
  computes `out[i][j] = in[i][j] * in[i][j]` cannot tell a correct index mapping
  from a transposed one. Adding an index-derived term
  (`out[i][j] = in[i][j] + (i*N + j)`) makes any mis-ordering a numerical
  difference. This is what catches an off-by-one or swapped-dimension bug.
- **Use asymmetric, non-square shapes** (`10x20`, not `8x8`), so a swapped
  subscript is out of range rather than merely different.
- **Avoid commutative-looking bodies** for the same reason.
- **Prefer exact arithmetic** where you can; the default tolerance is
  `rtol=0.05, atol=0.01`, which is loose enough to hide a small systematic error.
  Tighten with `--rtol`/`--atol` when the computation is exact.
- **Pin the inputs** with `--seed` when chasing a specific failure, and
  `--lower-bound`/`--upper-bound`/`--input-value` when the default random range
  makes the output uninteresting (all-negative input into a ReLU, say).
- **In file mode, keep the baseline differing only where the feature requires**,
  and read the `-d` diff to confirm it. A pair that differs in several unrelated
  places is testing several things at once, so a failure no longer points at one.
  See "Read the `-d` diff, and make it minimal".

## Getting breadth: the scenario matrix

Coverage comes from crossing a few axes deliberately rather than writing many
similar cases. For a lowering pass, walk these and pick the combinations your
pass actually distinguishes:

| Axis | Values worth covering |
|---|---|
| Rank | 1-D, 2-D, 3-D, and one higher than any special case in the code |
| Shape knowledge | fully static, fully dynamic, **mixed** static/dynamic |
| Sizes | a degenerate `1`, a size that is *not* a multiple of any tile/vector length, a large one |
| Element type | `f32` plus every other type the pass claims to handle |
| Broadcasting | none, one operand broadcast, scalar operand |
| Attributes | each attribute/option of the op, at its default and at a non-default |
| Composition | your pass together with each neighbouring transform it can legally meet (tiling, permutation, parallel, SIMD) |
| Rejections | every combination the pass declares unsupported |

### When changing an existing pass, cover what you did *not* mean to change

A new pass starts with no behavior to protect. A change to an existing one does,
and the cases at risk are the ones you were not thinking about. Two additions to
the matrix above:

- **The "off" path.** If the new behavior is conditional -- on an option, an
  optimization level, a shape property, a target -- ground at least one case with
  it *off*, over the same code your change touches. That is what shows you altered
  only what you intended to.
- **The neighbours that already worked.** Find the existing tests covering the
  code you edited (`grep` the pass name under `test/mlir`, or the ops your
  transform matches) and read their diffs, not just their pass/fail. A test that
  still passes but whose output changed is telling you something; a case whose
  output changed and has *no* test is the one that will bite a user. Where you
  find such a hole, that is the next test to write -- grounded, like the rest.

### Dynamic dimensions are mandatory, not optional

This is the axis most often skipped and the one that finds the most bugs, so treat
it as a requirement: **your change is not tested until it has been tested with dynamic
dimensions.** A suite of static-shape cases can be entirely green while the pass
is broken for every real model, because a static shape lets the compiler constant
-fold away the very code you need to exercise.

Concretely, dynamic shapes are the only way to reach:

- **Bounds that are runtime values.** Trip counts become computed products rather
  than constants, and whatever computes them must be materialized somewhere that
  *dominates* every use. A static shape folds this to a constant and the
  dominance question never arises.
- **Maps with symbols instead of literals.** Index arithmetic that reads
  `d0 floordiv 20` when static becomes `d0 floordiv symbol(%dim)`, which has
  extra validity rules -- an affine symbol must be a legal symbol at that point.
- **Mixed static/dynamic**, which is a *third* case, not a midpoint: the
  generated maps combine constants and symbols, and code paths that special-case
  "all static" or "all dynamic" fall between the two.

Requirements for these cases:

1. Cover all three: all-dynamic, all-static, and mixed.
2. Ground each with `--shape-info 0:10x20`, pinned in a `// GROUND-THIS:` line.
3. **Run at least one dynamic case at more than one shape**, including a
   degenerate `1` and a size that is not a multiple of any tile or vector length.
   One shape proves the code runs; several prove nothing was baked in for that
   shape. A `1x1` run in particular catches loops that assume more than one
   iteration.
4. Remember `--shape-info` is a *run-time* option: it chooses the shape used to
   generate inputs, leaving the compiled model genuinely dynamic. The compiler's
   `--shapeInformation` is a different thing -- it rewrites ONNX graph inputs
   during shape inference, so it does nothing for an already-lowered module and
   would defeat the point even where it applies.

## Negative tests

Every restriction a pass documents needs a test that it is actually enforced,
otherwise the restriction is a comment. These are cheap -- no grounding, since
nothing runs:

```
// RUN: onnx-mlir-opt --my-pass %s -split-input-file -verify-diagnostics
...
  // expected-error @+1 {{a distinctive fragment of the message}}
```

Match a distinctive substring, not the whole message, so wording can improve
without breaking tests. Prefer a real diagnostic over an assertion: an `assert`
is compiled out of release builds, so a rejection that only asserts is a silent
miscompile for release users. If a construct must be rejected, and the check
cannot live in a verifier because it depends on context the op cannot see, do it
in the pass and emit a proper error there.

## Definition of done

- `GroundLitTest.py <file>` with no other options grounds **every** function
  in the file and reports `Failed (0)`. The options each case needs live in its
  `GROUND-ALL`/`GROUND-THIS` directives, so this one command is the gate.
- The CHECK lines were generated *after* that passed, never before.
- For every **new** file-mode pair, the `-d` diff was read line by line and every
  remaining difference is one the feature requires. Not needed for a case that
  follows an existing pair's shape.
- Dynamic dimensions are covered in all three forms (all-dynamic, all-static,
  mixed), and at least one dynamic case was grounded at more than one shape.
- Every documented restriction has a negative test.
- `fixLitTest.py -t <file>` is clean, so each function passes on its own.
- The full suite is green, not just your file:
  `llvm-lit build/test/mlir` (or `make check-onnx-lit`).
- The committed test file is exactly what the tools generate. If you had to
  hand-edit generated CHECK lines, that is a bug in the generator worth fixing --
  a hand-patched file silently stops being regenerable.

## Gotchas

| Symptom | Cause |
|---|---|
| Grounding fails with an error mentioning code you already fixed | Stale `onnx-mlir`; `onnx-mlir-opt` is a different target. Rebuild both. |
| `expected one argument` from an option | Write `-r=--O0`, `--compile-args=--parallel`; argparse eats a bare leading `--`. |
| Dynamic input has no shape to run with | Use `--shape-info 0:10x20` (a *run-time* option of `RunONNXModel.py`). Not `--shapeInformation`, which rewrites ONNX graph inputs during shape inference and so has no effect on an already-lowered module. |
| `failed to legalize operation 'scf.parallel'` | The parallel path needs `--compile-args=--parallel`; without it nothing in the pipeline lowers `scf.parallel`. |
| A test file parses on its own but breaks under lit | Never write the split marker (`// ` followed by five dashes) inside a comment: `-split-input-file` cuts on that substring anywhere in the file, not just at line start, and splits your prose mid-sentence. |
| `lit` reports your helper `.mlir` as UNRESOLVED | Every `.mlir` under `test/mlir` is collected as a test, `.baseline<ext>` files aside. Name a file-mode baseline `<test>.baseline.mlir` and lit leaves it alone; any *other* helper file needs its own RUN line, plus one `CHECK-LABEL` inside each split segment. |
| `fixLitTest.py` ignores your existing CHECK lines | They must start at **column 0**; the prefix scanner is anchored and does not see an indented `// CHECK`. |
| `fixLitTest.py -r` leaves a function untouched | Bare `-r` only repairs functions whose test *fails*. Use `-r -f <fn>` to regenerate one unconditionally. |
| A multi-prefix file skips some functions | Intended: a function carrying only another prefix's CHECK lines is skipped rather than failed, and never has assertions injected for a prefix it was not written for. |
| A `GROUND-THIS` seems ignored | It applies to the *next function defined after it*, and only in the test file named on the command line; directives in a baseline file are not read. It also cannot name a file or set `-f`. |
| `bad "GROUND-ALL" directive ... unrecognized arguments: // ...` | Everything after the `:` is options, so a trailing `//` explanation on the same line is parsed as arguments. Move the prose to its own comment line. |
| A directive value starting with `-` is rejected | Use the `--flag=value` form (`-c="-O3 -parallel"`, `-shape-info=0:10x20`), exactly as on the command line. |
| File mode reports FAILURE saying the variants are identical | The baseline is the same MLIR as the test modulo comments, so nothing was compared. Usually a copy-paste baseline that was never edited. |
| The `-d` diff is far larger than the feature warrants | Either the baseline differs gratuitously (a constant declared in a different place, renamed values, an unneeded shape) and should be brought into line, or the function is testing two things and wants splitting. Not always a defect: a baseline deriving by hand what the compiler is asked to produce is *supposed* to be longer. |
| `glt_*.mlir` missing after a run | Only a single-`-f` run keeps them; an all-functions run clears them, since only the last function's leftovers would survive. |
| `FileCheck: undefined variable` in generated output | A definition site `mlir2FileCheck.py` does not recognize. It detects definitions by the `%x =` shape, so anything binding a name without an `=` degrades to a dangling use. Fix the generator rather than hand-binding the line. |
