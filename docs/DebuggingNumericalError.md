<!--- SPDX-License-Identifier: Apache-2.0 -->

# Debugging Numerical Error

`onnx-mlir` ships a handful of `utils/` scripts and a small C++ driver
that cover the recurring debugging scenarios: comparing numerical
results against a reference, comparing two compiler options against
each other, inspecting the IR pass-by-pass, printing values from
generated code at runtime, and chasing crashes/memory corruption. Jump
to the section that matches your symptom.

## What problem are you trying to solve?

- My model's numeric output doesn't match the training framework /
  onnxruntime &rarr; [Comparing against a reference](#comparing-against-a-reference)
- I want to know whether a compiler option (e.g. `-O3` vs `-O0`, a new
  target flag) changes the result &rarr; [Comparing two compile options](#comparing-two-compile-options)
- I suspect a specific pass or operator produces the wrong IR and want
  to see the compiler's output at that point &rarr; [Inspecting the compiler's IR pass-by-pass](#inspecting-the-compilers-ir-pass-by-pass)
- I want to print a tensor's or scalar's value at a specific point in
  the generated code &rarr; [Printing values at runtime](#printing-values-at-runtime)
- The compiled model crashes, hangs, or I suspect memory corruption
  (buffer overrun, use-after-free, leak) &rarr; [Finding memory errors and crashes](#finding-memory-errors-and-crashes)

## Comparing against a reference

Use `utils/RunONNXModel.py` when an onnx-mlir-compiled inference
executable produces numerical results that are inconsistent with those
produced by the training framework. This python script runs the model
through onnx-mlir and a reference backend, and compares the
intermediate results produced by these two backends layer by layer.

### Prerequisite
- Set `ONNX_MLIR_HOME` environment variable to be the path to the HOME
  directory for onnx-mlir. The HOME directory for onnx-mlir refers to the
  parent folder containing the `bin`, `lib`, etc sub-folders in which ONNX-MLIR
  executables and libraries can be found.

### Reference backend
Outputs by onnx-mlir can be verified by using a reference ONNX backend or
reference inputs and outputs in protobuf.
- To verify using a reference backend, install onnxruntime by running `pip
  install onnxruntime`. To use a different testing backend, simply replace code
  importing onnxruntime to some other ONNX-compliant backend.
- To verify using reference outputs, use `--verify=ref --load-ref=data_folder`
  where `data_folder` is the path to a folder containing protobuf files for
  inputs and outputs. [This
  guideline](https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md#manipulating-tensorproto-and-numpy-array)
  is a how-to for creating protobuf files from numpy arrays.
- Reference data can also come straight from numpy: `--inputs-from-arrays`
  takes a list of numpy arrays directly, and `--load-ref-from-numpy PATH`
  points at a python script that defines `inputs` (and optionally
  `outputs`) as lists of numpy arrays.

### Usage

Run `utils/RunONNXModel.py --help` for the full, current list of
options — it changes over time, so this doc only calls out the ones
you'll reach for most:

| flag | purpose |
|---|---|
| `-m MODEL` / `--model MODEL` | Path to an ONNX model (`.onnx`, `.mlir`, or `.onnxtext`). |
| `-c COMPILE_ARGS` / `--compile-args COMPILE_ARGS` | Arguments passed directly to the `onnx-mlir` command. |
| `-C` / `--compile-only` | Only compile the input model. |
| `--shape-info SHAPE_INFO` | Shape for each dynamic input, used to generate random inputs when `--load-ref` isn't set. See [format](#--shape-info-format) below. |
| `--input-value INPUT_VALUE` | Per-input fill spec, overriding `--lower-bound`/`--upper-bound` for individual tensors. Format: `INPUT_ID:spec1 spec2 ..., ...` where each spec is `min<n>`, `max<n>`, `val<n>` (constant), or `soz<n>` (sequence-of-ones-then-zeros along the innermost dim — handy for attention masks). E.g. `--input-value=0:min-1.0max1.0,1:val0`. |
| `--lower-bound` / `--upper-bound` | Per-type default bounds for random-input generation, e.g. `--lower-bound=int64:-10,float32:-0.2` / `--upper-bound=int64:10,float32:0.2`. Supported types: `bool`, `uint8`, `int8`, `uint16`, `int16`, `uint32`, `int32`, `uint64`, `int64`, `float16`, `float32`, `float64`. |
| `--seed SEED` | RNG seed for generated inputs. |
| `--save-ref PATH` / `--load-ref PATH` | Save or load reference inputs/outputs as protobuf. |
| `--write-compile-log [PATH]` / `--write-runtime-log [PATH]` | Save compilation messages (default `compile.log`) or the runtime profile info (default `runtime.log`, requires the model compiled with `--profile-ir`). |
| `--verify {onnxruntime,ref}` | Compare onnx-mlir's output against onnxruntime, or against `--load-ref` data. Must be paired with `--verify-every-value` (compare every value within `--rtol`/`--atol`) or `--verify-with-softmax AXIS_INDEX` (compare after softmax — useful when comparing raw logits would be too strict). |
| `--verify-all-ops` | With `--verify=onnxruntime`, verify every intermediate op's output, not just the final one — this is the layer-by-layer comparison mentioned above. |
| `--rtol` / `--atol` | Relative / absolute tolerance used by `--verify-every-value`. |

The verification itself is opt-in — none of the flags above do
anything until you add `--verify`. A typical invocation comparing
every op against onnxruntime looks like:

```bash
utils/RunONNXModel.py --model model.onnx \
  --shape-info 0:1x384 \
  --verify=onnxruntime --verify-all-ops --verify-every-value
```

#### `--shape-info` format

Most ONNX models leave some input dimensions dynamic (e.g. batch size
or sequence length), which `RunONNXModel.py` needs concrete values for
in order to generate random test inputs. The format is:

```
INPUT_ID1:D1xD2x...xDn, INPUT_ID2:D1xD2x...xDn, ...
```

- `INPUT_ID` selects which model input the dimensions apply to. It can
  be a single index (`0`), an inclusive range (`5-17`, applying the
  same dimensions to inputs 5 through 17), or `-1` to target every
  input at once.
- Each `D` is a dimension size; `-1` for a given `D` leaves that
  dimension as whatever the model's own signature says (dynamic or
  fixed), instead of overriding it.
- E.g. `--shape-info 0:1x10x20,1:7x5x3` fixes input 0 to `1x10x20` and
  input 1 to `7x5x3`; `--shape-info=-1:1x384` sets every input's shape
  to `1x384` (dimensions that don't apply are ignored).

**Note:** `--shape-info` only affects the inputs `RunONNXModel.py`
generates at runtime — the compiled model itself may still treat those
dimensions as dynamic. For debugging, it's worth also fixing the
shapes at compile time, to check whether the bug is actually in the
compiler's handling of dynamic dimensions (which is harder for the
compiler to get right than the static case). Do so by passing the same
`shapeInformation` option through to `onnx-mlir` itself, e.g.
`--compile-args="-O3 -shapeInformation=0:1x384"`. With the shape baked
in at compile time, the compiler can constant-fold shape computations
and emit simpler, more static code — which also makes the IR easier to
read when [inspecting it pass-by-pass](#inspecting-the-compilers-ir-pass-by-pass).

## Comparing two compile options

Based on `RunONNXModel.py`, `utils/CheckONNXModel.py` runs a given model
twice under two distinct compile options and compares the results. This
lets you test a new option by comparing a known-safe compilation (e.g.
`-O0` or `-O3`) against a more advanced one (e.g. `-O3` or
`-O3 --march=x86-64`).

Specify the compile options with `--ref-compile-args`/`-r` and either
`--test-compile-args`/`-t` (a full replacement set of options) or
`--additional-test-compile-args`/`-a` (options added on top of `-r`), a
model with `--model`/`-m`, and, for dynamic-shape inputs,
`--shape-info`. It accepts the same input-generation flags as
`RunONNXModel.py` (`--input-value`, `--lower-bound`, `--upper-bound`,
`--seed`) plus `--cache-ref-model`/`--cache-test-model` to avoid
recompiling between runs and `--skip-ref` to skip the reference build
entirely once it's already been produced. Full, current options are
listed under `utils/CheckONNXModel.py --help`.

The comparison (tunable with `--rtol`/`--atol`) is only on the model's
final output, not a layer-by-layer breakdown — it tells you *whether*
the two compiles diverge, not *where*. Once you know they diverge,
pair this with
[inspecting the compiler's IR pass-by-pass](#inspecting-the-compilers-ir-pass-by-pass)
(run it once per set of compile args and compare the two logs) to find
which pass is responsible.

## Inspecting the compiler's IR pass-by-pass

To gain more insight in what the compiler is doing, print the output of
the compiler after each transformation pass. The flags are well
documented in the `mlir` literature; `utils/onnx-mlir-print.sh` wraps
them for you. Invoke it with all the desired `onnx-mlir` options, plus
one last argument giving the file to save the log to. The log is a
sequence of MLIR dumps, so giving it a `.mlir` extension (rather than
e.g. `.txt` or `.log`) gets you MLIR syntax highlighting when you open
it in VSCode:

```bash
utils/onnx-mlir-print.sh -O3 --profile-ir=Onnx model.onnx pass-log.mlir
```

Only passes that actually changed the IR are listed. That compiler
log may still have anywhere from 10 to 100+ passes. Use
`utils/IsolatePass.py` to isolate a given one:

- `utils/IsolatePass.py pass-log.mlir -l` lists the name of every pass.
- `utils/IsolatePass.py pass-log.mlir -p "convert-onnx-to-krnl"` prints
  the pass(es) whose name matches that regex — e.g. the ONNX-to-Krnl
  conversion. `-n 34` isolates the 34th pass listed by `-l` instead.
- `-a -1` / `-a 1` prints the pass immediately before / after the one
  selected by `-p`/`-n`. `-a` also accepts a regex, in which case it
  prints the next pass matching it.
- `-c` annotates the listing with use-counts for each def and values for
  scalar constants.

Full options are listed under `utils/IsolatePass.py --help`.

## Printing values at runtime

If you know, or suspect, that a particular ONNX operator produces an
incorrect result and want to narrow down the problem, there are two
ways to print tensor values at runtime: instrumentation flags that
need no source changes, and Krnl print operators you inject into the
compiler source for cases the flags can't reach.

### Instrumenting specific ops, no source changes needed

`onnx-mlir` can insert the printing for you at compile time, driven by
two flags:

- `--instrument-signature=REGEX` matches ops by dialect op name (e.g.
  `onnx.MatMul`, or `onnx.*` for everything) and prints each match's
  operand/result **type and shape** (no data).
- `--instrument-onnx-node=REGEX` matches ops by their unique
  `onnx_node_name` attribute instead, and additionally prints the
  **actual tensor data**. Find the node names you can match against in
  the output of `onnx-mlir --EmitONNXIR model.onnx`.

```bash
# Find the onnx_node_name of the op you suspect.
onnx-mlir --EmitONNXIR model.onnx

# Recompile with instrumentation on that node — no other source or
# compiler changes needed.
onnx-mlir -O3 --instrument-onnx-node="/encoder/layer.0/attention/MatMul" model.onnx
```

Run the resulting model as usual (e.g. via `RunONNXModel.py` or the
[`build-run-onnx-lib.sh` inference engine](#build-a-compact-c-inference-engine-for-a-given-model-using-build-run-onnx-libsh));
every matching op prints an `==SIG-REPORT==` line to stdout on each
call. Both flags accept `NONE`/`ALL`, a comma-separated
list, or `*`-glob patterns. This is a different mechanism from the
`--instrument-ops`/`--InstrumentReportTime`/`--InstrumentReportMemory`
family described in [Instrumentation.md](Instrumentation.md) — that
one reports timing and memory, not values.

### Injecting print calls into the compiler source

For anything the flags above can't express — custom formatting,
conditional printing, or printing an intermediate value that never
becomes an op's operand/result — a couple of Krnl operators let you
print (at runtime) the value of a tensor, or a value that has a
primitive data type, from inside the compiler's own lowering code.
This requires editing onnx-mlir's C++ source and rebuilding the
compiler (not just recompiling the model).

To print out the value of a tensor at a particular program point, inject the following code (where `X` is the tensor to be printed):

```code
create.krnl.printTensor("Tensor X: ", X);
```

Note: currently the content of the tensor is printed only when the tensor rank is less than four.

To print a message followed by one value, inject the following code (where `val` is the value to be printed and `valType` is its type):

```code
create.krnl.printf("inputElem: ", val, valType);
```

## Finding memory errors and crashes

If the compiled model crashes, hangs, or you suspect a memory issue
(leak, buffer overrun, use-after-free), don't debug it inside a full
test harness — build a small, standalone executable around the model
and run that under a memory tool instead.

### Build a compact C++ inference engine for a given model using `build-run-onnx-lib.sh`

`utils/build-run-onnx-lib.sh` builds a minimal C++ inference engine
around a compiled model: it loads the model, fills its inputs (random,
by default, using the same
`--shape-info`/`--input-value`/`--lower-bound`/`--upper-bound`/`--seed`
mini-language as `RunONNXModel.py`), and calls the model's entry point
in a loop. It's built from `utils/RunONNXLib.cpp`, and can build in one
of two modes — critically, it can statically link the model into the
binary, which is what makes it so much easier to debug (see below). Set
`ONNX_MLIR_HOME` (same convention as `RunONNXModel.py`, e.g.
`onnx-mlir/build/Debug`) to run the script from any directory;
otherwise it must be run from the `onnx-mlir/build` directory itself.
Either way, the binary lands under that build's `bin` subdirectory
(`Debug/bin` or `Release/bin`, depending on which build you're
pointing at) and the script prints the exact path once it's built:

```bash
# Statically linked: the model is linked directly into the executable.
sh utils/build-run-onnx-lib.sh test/backend/test_add/test_add.so
<bin-dir>/run-onnx-lib

# Dynamically loaded: model.so is passed on the command line at
# runtime, so one binary can drive any model.
sh utils/build-run-onnx-lib.sh
<bin-dir>/run-onnx-lib test/backend/test_add/test_add.so
```

The static build is the one to reach for when chasing a crash: the
result is a single self-contained binary with the model's code and
symbols resolved at link time, no `dlopen`/`dlsym` indirection to step
through, and no separate `.so` to keep track of. That makes it much
easier to point valgrind, gdb/lldb, or a sanitizer straight at it, so
prefer building statically over the dynamic default whenever you're
debugging a crash or memory issue rather than just running the model:

```bash
valgrind --error-exitcode=1 <bin-dir>/run-onnx-lib \
  --shape-info 0:1x10x20 --seed 1 -n 20
```

Useful flags for reproducing a crash reliably (see
`<bin-dir>/run-onnx-lib --help` for the full list):

| flag | purpose |
|---|---|
| `-s` / `--shape-info` | Shape overrides for dynamic input dimensions, same format as `RunONNXModel.py`. |
| `-i` / `--input-value`, `-l` / `--lower-bound`, `-u` / `--upper-bound` | Same per-input fill spec / per-type bound mini-language as `RunONNXModel.py`. |
| `-k` / `--seed` | RNG seed — pin this once you have a reproducing seed. |
| `-n` / `--iterations` | Run the model this many times in one process; useful for issues that only show up after repeated calls (e.g. leaks, or state that isn't reset between calls). |
| `-r` / `--reuse true\|false` | Whether to reuse the same input buffers across iterations (default on) or regenerate them each time. |
| `-e` / `--entry-point` | Entry point name, if not `run_main_graph` (dynamic build only). |
| `-v` / `--verbose` | Print the shape and value range of each input. |

### Memory tools (valgrind / mtrace / Electric Fence)

Run the `run-onnx-lib` executable above under whichever tool fits the
symptom:

- The [valgrind framework](https://valgrind.org/) or the
  [mtrace memory tool](https://github.com/sstefani/mtrace) trace
  memory allocation/free-related APIs and can detect issues such as
  memory leaks.
- Buffer-overrun problems are notoriously difficult to debug this way,
  because the run-time error surfaces outside of the code that caused
  it. The
  ["Electric Fence library"](https://github.com/CheggEng/electric-fence)
  helps here: it detects both overruns past a `malloc()` allocation's
  boundary and accesses to memory already `free()`d, including read
  accesses, and pinpoints the exact instruction that causes the error.

  Since the Electric Fence library is not officially supported by
  RedHat, you need to download, build and install the source code
  yourself. After installing it, link this library by adding
  `-lefence` to the link line in `utils/build-run-onnx-lib.sh` (or your
  own equivalent build command) when generating the inference
  executable, then simply run it — it will trigger a runtime error and
  stop at the place causing the memory access problem. Identify the
  exact spot with a debugger, or with the printing techniques described
  in [Printing values at runtime](#printing-values-at-runtime).
