---
name: debug-compilation
description: 'Use when an onnx-mlir compile fails, produces wrong results, or runs slowly: pick the right --Emit* stage, dump and isolate IR per pass (onnx-mlir-print.sh, IsolatePass.py), compare against onnxruntime, profile with instrumentation, and check NNPA offload.'
---

<!-- Generated from the onnx-mlir knowledge base (skills/debug-compilation.md), verified against onnx-mlir@b505d64b. Do not edit here: see .agents/skills/README.md. -->

# Debug an onnx-mlir Compilation

Step-by-step walkthrough for diagnosing onnx-mlir compilation failures and performance issues using the built-in emission and instrumentation tools. `docs/DebuggingNumericalError.md` is onnx-mlir's own, more official version of much of this walkthrough (comparing against a reference, comparing two compile options, pass-by-pass IR inspection, printing values at runtime, and — not otherwise covered on this page — finding memory errors and crashes) — treat it as the primary source and this page as a companion index into it.

---

## Step 1: Identify the failure mode

Run with the default target first:

```bash
onnx-mlir mymodel.onnx -o mymodel 2>&1
```

Common failure categories:
- **Import error**: malformed protobuf, unsupported opset version → fix the model or update opset support
- **Shape inference failure**: `assert` or `error: operation has unresolved type` → Step 2
- **Lowering failure**: `error: failed to legalize operation 'onnx.SomeOp'` → missing lowering
- **Compile crash**: segfault or ICE → Step 3
- **Wrong results**: compiles fine but outputs are incorrect → Step 4
- **Inputs rejected at run time**: the model returns `NULL` (errno `EINVAL`) after printing e.g. `Wrong rank for the input 1: expect 2, but got 3` or `Inconsistent dimension for symbol 'batch_size' at dimension 0 of input 1: expect 4, but got 8` → the inputs do not match the model's signature, including inputs that must agree on a shared symbolic dimension. Print the signature with `inspect-so mymodel.so` and see [stack/runtime.md](../../../docs/InputVerification.md)

---

## Step 2: Inspect the ONNX dialect IR

```bash
onnx-mlir --EmitONNXBasic mymodel.onnx 2>&1 | less  # before shape inference
onnx-mlir --EmitONNXIR mymodel.onnx 2>&1 | less     # after graph transforms
```

Look for:
- `tensor<*xf32>` — unranked tensors that shape inference failed to resolve
- `tensor<?xNxM xf32>` — check if the `?` is expected or a bug
- `onnx.Custom` — op type not in the dialect's dispatch table (not a nonexistent `onnx.UnknownOp`)
- Missing ops entirely — may be inside a subgraph that wasn't visited

To run only shape inference:
```bash
onnx-mlir-opt --shape-inference mymodel.mlir  # (first emit with --EmitONNXBasic)
```

---

## Step 3: Narrow to a single pass

Emit at each stage to find where the error occurs:

```bash
onnx-mlir --EmitONNXBasic mymodel.onnx -o /tmp/stage1  # import only
onnx-mlir --EmitONNXIR    mymodel.onnx -o /tmp/stage2  # + graph transforms
onnx-mlir --EmitMLIR      mymodel.onnx -o /tmp/stage3  # + krnl lowering
onnx-mlir --EmitLLVMIR    mymodel.onnx -o /tmp/stage4  # + llvm lowering
```

The stage that fails first contains the buggy pass. Then run individual passes with `onnx-mlir-opt`:

```bash
# Test a single pass
onnx-mlir-opt --constprop-onnx /tmp/stage1.mlir
onnx-mlir-opt --shape-inference /tmp/stage1.mlir
onnx-mlir-opt --convert-onnx-to-krnl /tmp/stage2.mlir
onnx-mlir-opt --convert-krnl-to-affine /tmp/stage3.mlir
```

### Getting a pass's internal reasoning with `LLVM_DEBUG`

If narrowing to a stage isn't enough — you know *which* pass is misbehaving but not *why* — most passes (onnx-mlir's own Krnl/ONNX transforms, and the underlying LLVM/MLIR infrastructure alike) wrap internal trace output in `LLVM_DEBUG(...)`, gated by a `#define DEBUG_TYPE "<name>"` near the top of that pass's source file.[^2] Running the compiled `onnx-mlir` (or `onnx-mlir-opt`) with `-debug-only=<name>` prints just that pass's own reasoning — usually far more targeted than reading a generic `--EmitONNXIR`/`--EmitMLIR` dump line by line:

```bash
onnx-mlir -debug-only=lowering-to-krnl mymodel.onnx 2>&1 | less
```

See [skills/debug-type-catalogue.md](../../../.agents/skills/debug-type-catalogue/SKILL.md) for the full lookup table of every `DEBUG_TYPE` string in `src/` to the pass it belongs to and what its trace output actually shows — including which strings are shared stage-wide across many files versus scoped to one narrow pass, and one pair of near-identical strings (`dialect-builder`/`dialect_builder`) that are easy to confuse. That page's "two-tool combo" section also spells out how to pair this with the dump-everything-then-isolate workflow below: the `-debug-only` trace tells you *why*, `IsolatePass.py` shows you *what changed*.

This convention isn't specific to onnx-mlir — it applies equally to `llvm-project/mlir/lib/` passes, since `DEBUG_TYPE`/`LLVM_DEBUG` is an LLVM-wide idiom.

### Not sure which pass introduced the bug: dump every pass, then isolate one

When it's unclear which of the *many* passes between two emission stages caused a regression, `utils/onnx-mlir-print.sh` runs a normal `onnx-mlir` invocation and dumps the IR after **every** pass into one log file. It takes the same arguments as `onnx-mlir`, plus a trailing output-file argument:

```bash
onnx-mlir-print.sh <normal onnx-mlir args...> /tmp/all-passes.log
```

Under the hood it adds `--mlir-print-ir-after-all --mlir-print-ir-after-change` and delegates to `utils/onnx-mlir-truncate.py`. Because of `--mlir-print-ir-after-change`, only passes that actually modified the IR appear in the log. The script's own header comment still says that flag is not added by default, but the command line adds it. `onnx-mlir-truncate.py` adds `--mlir-elide-elementsattrs-if-larger=20` and `--mlir-elide-resource-strings-if-larger=20`, writes the command line at the top of the log, and cleans each line with `clean_line()` from `utils/mlir_log_utils.py`. It does not cut lines at a fixed length. Instead it rewrites the unparseable `dense<__elided__>` marker as `dense_resource<__elided__>` and replaces any quoted string longer than 300 characters (e.g. an `llvm.mlir.global` byte dump) with a short placeholder, so individual IR listings stay parseable.

Then `utils/IsolatePass.py` extracts one pass's IR back out of that log:

```bash
IsolatePass.py -i /tmp/all-passes.log -l               # list every pass name in the log (default when -p/-n omitted)
IsolatePass.py -i /tmp/all-passes.log -p <regex>        # isolate pass(es) whose name matches regex
IsolatePass.py -i /tmp/all-passes.log -n <num>          # isolate the num'th pass
IsolatePass.py -i /tmp/all-passes.log -p <regex> -a <N> # isolate N passes after the match (N may be negative)
```

`-c`/`--comments` additionally annotates the extracted listing with per-value use-counts and scalar constant values, and `-w`/`--wrap` splits long function signatures, ops and returns across several indented lines.

Typical flow: dump everything with `onnx-mlir-print.sh`, skim with `-l` to spot where the IR looks wrong, then isolate that pass (and the one immediately before it, via `-a -1`) to diff before/after in isolation.

---

## Step 4: Verify correctness against onnxruntime

```bash
python3 utils/RunONNXModel.py \
  --model mymodel.onnx \
  --verify onnxruntime --verify-every-value \
  --rtol=1e-3 --atol=1e-5 \
  2>&1
```

If outputs differ:
- Try `--O0` to disable optimizations and see if that changes the result
- Narrow to the offending op with instrumentation (Step 5)
- Check if `--disable-simd` fixes it (SIMD vectorization bug)

---

## Step 5: Profile op-level performance

`onnx-mlir` emits a shared library, not an executable, so profiling needs a driver process to load the `.so` and call it — `RunONNXModel.py` compiles and runs in one step:

```bash
# Compile with profiling instrumentation and run
python3 utils/RunONNXModel.py --model mymodel.onnx \
  --compile-args="-O3 --profile-ir=Onnx" > profile.txt

# Find top-10 slowest ops
grep "PERF-REPORT" profile.txt | sort -t, -k5 -rn | head -10
```

`--profile-ir=<stage>` is the flag to use; it sets the whole `--instrument-stage`/`--instrument-ops`/`--Instrument*` combination, including the correct op pattern for the stage. The raw `--instrument-*` flags are deprecated for everyday use — see [flows/instrumentation-and-profiling.md](../../../docs/Instrumentation.md), which also covers running an already-compiled `.so` and sending the log to `ONNX_MLIR_INSTRUMENT_FILE` instead of stdout.

For per-op *shape* information (e.g. to determine which dimension a `Transpose` actually permutes, or spot a missing broadcast) rather than timing, use `--profile-ir-with-sig` and `utils/make-report.py` instead — see [flows/instrumentation-and-profiling.md](../../../docs/Instrumentation.md).

---

## Step 6: Check NNPA offload (if using --maccel=NNPA)

Offload is decided at compile time, so start there — no run needed. Every NNPA compile already prints a fallback summary for nine heavy op types with no flag, and `--opt-report=NNPAUnsupportedOps` gives the per-op reason:

```bash
onnx-mlir -O3 --march=z16 --maccel=NNPA --opt-report=NNPAUnsupportedOps mymodel.onnx \
  | grep -E "not accelerated by NNPA|NNPA-UNSUPPORTEDOPS-REPORT"
```

Full walkthrough of the reason categories and the fix for each: [nnpa-performance.md](../../../.agents/skills/nnpa-performance/SKILL.md).

Use the runtime route below when you need the *cost* of what fell back rather than the fact of it — it measures per-op time, which the compile-time report cannot:

```bash
python3 utils/RunONNXModel.py --model mymodel.onnx \
  --compile-args="-O3 --march=z16 --maccel=NNPA --profile-ir=ZHigh" > profile_nnpa.txt

# Count ops on NNPA vs CPU
grep "zhigh\." profile_nnpa.txt | wc -l  # NNPA ops
grep "onnx\."  profile_nnpa.txt | wc -l  # CPU fallback ops
```

`--profile-ir=ZHigh` instruments `onnx.*` and `zhigh.*` together, so one run gives both counts and both timings.

---

## Step 7: Get SIMD/parallel report

```bash
onnx-mlir -O3 --opt-report=Simd --opt-report=Parallel mymodel.onnx 2>&1 | grep -E "SIMD|parallel"
```

Without `--parallel` every op reports "parallelism is disabled". With `--parallel`, some lowerings collapse several loop levels into one parallel region by default; the report line then says which levels were collapsed. To check whether collapsing changes a result or a timing, compare against a `--disable-collapse` build, e.g. `utils/CheckONNXModel.py --model=mymodel.onnx --compile-args="-O3 --parallel" --ref-compile-args="--disable-collapse"` (the reference run adds `--disable-collapse`; the test run uses the shared args alone). `utils/make-report.py -c <compile_log> --stats=par` summarizes the report — see [flows/instrumentation-and-profiling.md](../../../docs/Instrumentation.md).

---

## Step 8: Find memory errors and crashes

If the compiled model crashes, hangs, or you suspect a leak/buffer-overrun/use-after-free, don't debug it inside a full test harness — build a small, standalone executable and run *that* under a memory tool instead:

```bash
# Static build: model compiled directly into the binary — symbols resolved
# at link time, no dlopen/dlsym indirection, easiest target for a debugger
# or sanitizer.
sh utils/build-run-onnx-lib.sh mymodel.so
<bin-dir>/run-onnx-lib

valgrind --error-exitcode=1 <bin-dir>/run-onnx-lib --shape-info 0:1x10x20 --seed 1 -n 20
```

`-n`/`--iterations` re-runs the model in one process — useful for issues that only appear after repeated calls (leaks, state not reset between calls). Full flag reference and the Electric Fence library (for pinpointing buffer overruns/use-after-free specifically) are in [stack/tools.md#build-run-onnx-libsh--runonnxlibcpp](../../../docs/DebuggingNumericalError.md) and `docs/DebuggingNumericalError.md`'s "Finding memory errors and crashes" section.

## Related: proving a pass you're writing is correct

This page is for *diagnosis* — a compile or a result is wrong and you need to find which pass did it. The complementary problem, *verification* of a pass you are actively writing or changing, is [ground-lit-tests.md](../../../.agents/skills/ground-lit-tests/SKILL.md): establish numerically that the output is right, then freeze it into FileCheck assertions. The `-debug-only=<DEBUG_TYPE>` and pass-isolation steps above are what you reach for when a grounding run fails and you need to know why.

## Verification

- **Grounded against:** `repo:onnx-mlir@b505d64:docs/Instrumentation.md`, `repo:onnx-mlir@b505d64:src/Compiler/CompilerOptions.cpp` (emission targets, `--profile-ir`, `--opt-report`, `--parallel`, `--disable-collapse`, `--disable-simd`); Step 3's dump-and-isolate section, Step 4's `--verify` choices, Step 5's `PERF-REPORT` column layout and Step 8 re-checked against `repo:onnx-mlir@b505d64` (`utils/onnx-mlir-print.sh`, `utils/onnx-mlir-truncate.py`, `utils/mlir_log_utils.py` `clean_line`, `utils/IsolatePass.py`, `utils/RunONNXModel.py`, `docs/Instrumentation.md`, `src/Runtime/OMInstrument.c`, `docs/DebuggingNumericalError.md`, `utils/build-run-onnx-lib.sh`); the run-time input-rejection category (`src/Conversion/KrnlToLLVM/KrnlEntryPoint.cpp`, `docs/InputVerification.md`, `src/Tools/inspect-so/InspectSo.cpp`) and Step 7 (`src/Compiler/CompilerOptions.cpp` `disable-collapse`, `src/Conversion/ONNXToKrnl/KrnlParallelPlan.cpp`, `utils/make-report.py`) re-checked against `repo:onnx-mlir@b505d64`
- **Files read:** `docs/Instrumentation.md`, `CompilerOptions.cpp` (emission target enum), `docs/TestingHighLevel.md`, `utils/onnx-mlir-print.sh`, `utils/onnx-mlir-truncate.py`, `utils/mlir_log_utils.py`, `utils/IsolatePass.py` (argparse definitions), `utils/RunONNXModel.py` (`--verify` choices), grep for `#define DEBUG_TYPE` across `src/`, `docs/DebuggingNumericalError.md` (full read, 2026-08-06 sync)
- **How to test:** Run any `.onnx` model through each step; `RunONNXModel.py --verify onnxruntime --verify-every-value` is the correctness gate; for Step 8, `valgrind --error-exitcode=1` against a known-good model should exit 0

[^2]: repo:onnx-mlir@b505d64:src/ (convention verified via grep for `#define DEBUG_TYPE`), also used throughout repo:llvm-project@43574226:mlir/lib/
