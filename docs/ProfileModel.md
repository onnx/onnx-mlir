# Profiling a compiled ONNX model with `profile-model.py`

`utils/profile-model.py` runs a compiled `model.so` in a tight loop,
samples the CPU at 1 ms intervals, and breaks the time down by
ONNX op, by ASM instruction, and by per-op instruction mix. It
also produces an annotated disassembly highlighting the hottest
instructions and basic blocks.

The script handles the workload C++, the timing loop, the SIGPROF
sampler, the symbolisation, and the DWARF-based op-attribution. The
only thing you provide per model is the input tensors — either via
`--shape-info` / `--input-value` / `--lower-bound` / `--upper-bound`
flags (no C++ required), or by writing an `initialize_model_input()`
C++ function yourself for cases the flags can't express.

## 1. Compile the model with `--profile-ir`

The op attribution is driven by DWARF entries the compiler emits
into `model.so` when `--profile-ir` is set. Without it, the script
will still print the per-instruction histogram but no per-op
breakdown.

CPU only:

```bash
onnx-mlir -O3 --profile-ir=Onnx \
  -shapeInformation=0:1x384 \
  roberta-base-11.onnx
```

NNPA accelerator (s390x):

```bash
onnx-mlir -O3 -march=z17 -maccel=NNPA --profile-ir=ZHigh \
  -shapeInformation=0:1x384 \
  roberta-base-11.onnx
```

Use `--profile-ir=Onnx` to see ops in their original `onnx.*` form,
or `--profile-ir=ZHigh` to see ops post-conversion (with
`zhigh.MatMul`, `zhigh.Softmax`, etc. for the NNPA-eligible ones).

The output is `roberta-base-11.so` plus, on macOS, a `.dSYM` bundle
holding the DWARF.

## 2. Provide the model's inputs

The workload needs input tensors before it can run. There are two
ways to supply them:

- **Option A — auto-generate from flags**, no C++ required. Covers
  the common case: dynamic-dimension overrides and random/constant
  fill of the input buffers.
- **Option B — write `initialize_model_input.cpp`**, for cases the
  flags can't express (data read from a file, a specific non-random
  pattern, per-element logic, etc.).

### Option A: auto-generate inputs from flags

Pass `--shape-info` / `--input-value` / `--lower-bound` /
`--upper-bound` / `--seed` directly to `profile-model.py` and it
builds `initialize_model_input()` for you, reading the model's own
input signature and filling buffers accordingly — no `-i`/`--init`
needed:

```bash
profile-model.py -m roberta-base-11.so -t 30 \
  --shape-info 0:1x384 \
  --input-value 0:min0max50264
```

| flag | purpose |
|---|---|
| `--shape-info STR` | Dimension overrides for dynamic inputs. Format: `INPUT_ID:D1xD2x...xDn, ...`. `INPUT_ID` is an integer, a range (`5-17`), or `-1` for all inputs; a dimension of `-1` keeps the signature's value. E.g. `--shape-info 0:1x180,1:1x180`. |
| `--input-value STR` | Per-input fill spec. Format: `INPUT_ID:spec1 spec2 ..., ...` where each spec is `min<n>`, `max<n>`, `val<n>` (constant fill), or `soz<n>` (sequence of `<n>` ones then zeros along the innermost dim — handy for attention masks; `soz-1` picks a random count per row). E.g. `--input-value 0:min0max30000,1:soz-1`. |
| `--lower-bound STR` | Per-type default lower bounds, used when `--input-value` doesn't set an explicit `min` for a tensor. Format: `typename:value, ...` (`bool`, `int8`, `uint8`, ..., `float32`, `float64`). E.g. `--lower-bound float32:-0.1,int64:0`. |
| `--upper-bound STR` | Same format as `--lower-bound`, for upper bounds. E.g. `--upper-bound float32:0.1,int64:30000`. |
| `--seed N` | RNG seed for reproducibility (default 42). |

Passing any one of these flags is enough to trigger auto-generation.
Anything you don't set falls back first to onnx-mlir's built-in
per-type defaults (floats `[-0.1, 0.1]`, signed ints `[-10, 10]`,
unsigned ints `[0, 10]`, bool `{false, true}`), then to whatever
shape the model's signature already specifies. These flags are
mutually exclusive with `-i`/`--init` (Option B).

### Option B: write `initialize_model_input.cpp`

Reach for this when the auto-generated fill isn't flexible enough —
e.g. you need values loaded from a file, a specific non-random
pattern, or logic that varies per element.

The script needs one C++ function:

```cpp
extern "C" OMTensorList *initialize_model_input(void);
```

It is called once before the timing loop and the returned tensor
list is reused across every inference. The buffers must outlive the
process — `static` storage is the simplest way.

Get a ready-to-edit template:

```bash
profile-model.py -h init > init-roberta.cpp
```

Adapt the shapes, dtypes, and content to your model. Example for
roberta with `-shapeInformation=0:1x384`:

```cpp
#include <cstdint>
#include <random>
#include <OnnxMlirRuntime.h>

extern "C" OMTensorList *initialize_model_input(void) {
  constexpr int64_t kBatch = 1, kSeqLen = 384, kRank = 2;
  constexpr int64_t kNumElems = kBatch * kSeqLen;
  static int64_t shape[2] = {kBatch, kSeqLen};
  static int64_t inputIds[kNumElems];

  std::mt19937_64 rng(12345);
  std::uniform_int_distribution<int64_t> vocab(0, 50264);
  for (int64_t i = 0; i < kNumElems; ++i) inputIds[i] = vocab(rng);

  OMTensor *t = omTensorCreate(inputIds, shape, kRank, ONNX_TYPE_INT64);
  OMTensor *list[1] = {t};
  return omTensorListCreate(list, 1);
}
```

The profiler does not care about the exact values, only the shape
of the workload. A fixed seed makes runs bit-identical.

### Includes and libraries

The only header your init `.cpp` needs is `<OnnxMlirRuntime.h>`,
which ships with the onnx-mlir checkout:

```
<onnx-mlir-repo>/include/OnnxMlirRuntime.h
```

It declares `OMTensor`, `OMTensorList`, `omTensorCreate`,
`omTensorListCreate`, and the `ONNX_TYPE_*` enum.

You don't need to link anything by hand — `profile-model.py`
takes care of compilation and linking for you. For reference,
the command it issues is roughly:

```bash
clang++ -std=c++17 -O2 -g \
    -I <onnx-mlir-repo>/include \
    /tmp/profile-model-workload-<tag>.cpp \
    init-roberta.cpp \
    roberta-base-11.so \
    -o /tmp/profile-model-bin-<tag>
# Linux only: -L <so-dir> -Wl,-rpath,<so-dir> -ldl
```

`model.so` is linked directly (it carries the runtime symbols —
including `OMCurrent*` markers — statically from `libcruntime.a`),
so there is no separate `-lOMRuntime` step.  On Linux,
`-ldl` is added because the in-process sampler calls `dladdr()`
to resolve `model.so`'s runtime base; harmless on macOS where
it is part of libc.

## 3. Run the profiler

With a custom init `.cpp` (Option B):

```bash
profile-model.py -m roberta-base-11.so -i init-roberta.cpp -t 30
```

Or with auto-generated inputs (Option A), no `.cpp` needed:

```bash
profile-model.py -m roberta-base-11.so -t 30 \
  --shape-info 0:1x384 --input-value 0:min0max50264
```

Common flags:

| flag | purpose |
|---|---|
| `-t SECONDS` | sampling duration (default 30) |
| `-s FILE.json` | save the raw profile to JSON for later replay |
| `-l FILE.json` | replay a saved profile (`-l` instead of `-i`/`-m`/auto-init flags) |
| `-a FILE.s` | write annotated disassembly with `<<<= X.XX%` markers |
| `--op REGEX` | restrict instruction mix to ops whose name matches |
| `--not-op REGEX` | inverse: restrict to ops NOT matching |
| `--sampler {auto,inproc,sample,perf}` | sampler backend (default `auto` → `inproc`) |
| `--debug-omip` | dump every recovered op span with PC ranges |

(See [Option A](#option-a-auto-generate-inputs-from-flags) above for
`--shape-info` / `--input-value` / `--lower-bound` / `--upper-bound` /
`--seed`.)

### Sampler choice

- **`inproc`** (default): an in-process SIGPROF / ITIMER_PROF
  sampler. Records exact leaf PC and stamps each sample with the
  ONNX op the runtime is currently in (via the `OMCurrent*`
  globals updated by `OMInstrumentPoint`). Most accurate per-op
  attribution, especially on NNPA where the heavy `nnpa` /
  `jo` instructions live in zDNN's `invoke_nnpa` outside
  `main_graph` and would otherwise miss the DWARF spans. No
  kernel privilege required, so it works in any container.

- **`sample`** (macOS only): captures full call stacks via Apple's
  `sample`. Useful if you want stack-based per-binary or
  top-frame breakdowns. Aggregates sibling PCs into call-tree
  rows, which blurs the per-PC histogram.

- **`perf`** (Linux only): full per-PC counts plus stacks, but
  needs `perf_event_open` (kernel privilege; blocked in most
  containers).

Stick with `auto` unless you specifically need stacks.

## 4. Reading the output

Sections, in print order:

- **Samples / inside model**: how many ticks fired during
  `invoke_run` (only ticks bracketed by it count).
- **Per-binary leaf samples**: which `.so` each leaf PC fell in.
  100% in `model.so` is the expected case.
- **Instruction mix inside model.so**: by ASM mnemonic. Top of
  the list points at where the CPU is spending time.
- **OMInstrumentPoint markers (DWARF): N op spans, M unpaired**:
  the count of `__omip:` DWARF subprograms recovered. M=0 means
  every begin/end marker paired up cleanly.
- **Per-ONNX-op samples (attributed N / unattributed M)**: the
  per-op breakdown. With `inproc` and a `--profile-ir` build,
  attribution is exact (M ≈ 0). With kernel-side samplers, the
  script also walks recorded call stacks to recover the calling
  op for samples whose leaf PC lands in a callee outside
  `main_graph`.
- **Top N ops — instruction mix**: per-op breakdown of which
  instructions dominate that op's runtime. Useful for spotting,
  e.g., that `MatMul` is `fmla`-bound while `Softmax` is
  `vfmax`-bound.

If you passed `--annotate FILE.s`, that file contains the
hottest functions in objdump form with `<<<= X.XX%` annotations
on every instruction that received samples.

## NNPA / s390x note

When NNPA takes an op (e.g. `MatMul` becomes `zhigh.MatMul`), the
actual arithmetic happens inside zDNN's `invoke_nnpa` helper —
which is in the `.so` but outside `main_graph`'s PC range. Without
the runtime markers, the heavy `nnpa` / `jo` PCs would not fall
inside any DWARF span and would be lost as "unattributed".

The `inproc` sampler reads `OMCurrentOpName` (set by
`OMInstrumentPoint`) at every tick, so it knows the active op
regardless of where the leaf PC actually landed. As a result the
per-op breakdown faithfully reflects time on the accelerator,
matching the runtime instrumentation report.

## Notes

- `ONNX_MLIR_NO_INSTRUMENT=1` is set automatically by the script
  while sampling: it suppresses the per-call CSV output the
  runtime would otherwise emit, but the marker globals are still
  updated, so attribution stays exact.
- Recompile the `.so` after pulling onnx-mlir if you suspect
  stale runtime; the marker globals (`OMCurrent*`) need to be
  present in `libcruntime.a`.
- The dSYM (macOS) or in-binary DWARF (Linux) is required for
  per-op attribution; passing `-O3` does not strip them.
