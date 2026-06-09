# Profiling a compiled ONNX model with `profile-model.py`

`utils/profile-model.py` runs a compiled `model.so` in a tight loop,
samples the CPU at 1 ms intervals, and breaks the time down by
ONNX op, by ASM instruction, and by per-op instruction mix. It
also produces an annotated disassembly highlighting the hottest
instructions and basic blocks.

The script handles the workload C++, the timing loop, the SIGPROF
sampler, the symbolisation, and the DWARF-based op-attribution. The
only piece you write per model is an `initialize_model_input()`
C++ function describing the input tensors.

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

## 2. Write `initialize_model_input.cpp`

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

```bash
profile-model.py -m roberta-base-11.so -i init-roberta.cpp -t 30
```

Common flags:

| flag | purpose |
|---|---|
| `-t SECONDS` | sampling duration (default 30) |
| `-s FILE.json` | save the raw profile to JSON for later replay |
| `-l FILE.json` | replay a saved profile (`-l` instead of `-i`/`-m`) |
| `-a FILE.s` | write annotated disassembly with `<<<= X.XX%` markers |
| `--op REGEX` | restrict instruction mix to ops whose name matches |
| `--not-op REGEX` | inverse: restrict to ops NOT matching |
| `--sampler {auto,inproc,sample,perf}` | sampler backend (default `auto` → `inproc`) |
| `--debug-omip` | dump every recovered op span with PC ranges |

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
