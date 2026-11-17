<!--- SPDX-License-Identifier: Apache-2.0 -->

# Performance Testing

onnx-mlir has two complementary ways to find out where time is going in a
compiled model:

* **Instrumentation-based profiling** (this page): compile with `--profile-ir`
  (or `--profile-ir-with-sig`), run with `RunONNXModel.py`, and analyze the
  resulting text log with `utils/make-report.py`. This gives per-op wall-clock
  time (and, optionally, per-op tensor shapes) with very little setup. 
* **Sampling-based profiling** ([ProfileModel.md](ProfileModel.md)):
  `utils/profile-model.py` samples a running model's CPU at a fixed interval
  and breaks time down by op, assembly instruction, and instruction mix. Use
  it when you need instruction-level detail, not just per-op timing.

This page walks through the first workflow end to end: compile, run, collect
a log, and turn that log into a report with `make-report.py`.

## Step 1: Compile and run with `RunONNXModel.py`

Compile with `--profile-ir=<stage>` (`Onnx`, or `ZHigh` on an NNPA build) —
via `RunONNXModel.py`, that goes in `--compile-args`. Add `-w`/`-n` for
warmup/timed iteration counts and `--write-runtime-log` to capture the
instrumentation output:

```bash
utils/RunONNXModel.py --model mymodel.onnx -w 2 -n 10  \
  --compile-args="-O3 --profile-ir=Onnx" \
  --write-runtime-log run.log
```

* `-w`/`--warmup <num>`: number of untimed warmup runs (default 0).
* `-n`/`--n-iteration <num>`: number of timed runs (default 1). With more
  than one, `RunONNXModel.py` also prints min/max/mean/stdev across
  iterations, plus a quartile-trimmed variant ("Statistics 2").
* `--write-runtime-log [PATH]` (default `runtime.log`): writes the
  instrumentation log to `PATH`.

`run.log` will contain data from the warmup runs too — pass a matching `-w 2`
to `make-report.py` in the next step to exclude them.

## Step 2: Aggregate stats with `make-report.py`

```bash
utils/make-report.py -r run.log -w 2
```

With only `-r` given, `make-report.py` defaults to `--stats=perf`: one line
per op, with columns `op-name, count, average-time, cumulative-time,
percent-of-total`:

```
Statistics start (all ops).
  onnx.Add, 112, 0.0001166, 0.0130570, 8.4%
  onnx.Cast, 105, 0.0000018, 0.0001860, 0.1%
  onnx.MatMul, 72, 0.0112630, 0.8109330, 52.3%
```

Useful flags at this stage:

* `-w`/`--warmup <num>`: exclude the first `<num>` iterations from the
  aggregated stats (should match the warmup count you ran with).
* `-u`/`--unit {s,ms,us}`: report times in seconds (default), milliseconds,
  or microseconds.
* `--sort {name,num,time}`: sort rows by op name, occurrence count, or time
  (default: by time when a runtime log is given).
* `--reporting {quartile,all,sum}`: how to combine multiple timed iterations —
  average excluding the top/bottom quartile (default), average over all of
  them, or sum. `quartile` is the direct analog of `RunONNXModel.py`'s
  "Statistics 2". The `sum` option is useful for logs that comprise, for example, of the cumulative experiments of a decode process where each measurement corresponds to the processing of one additional token.
* `-m`/`--min <pct>`: only show ops using at least `<pct>`% of total exec time
  — useful to remove the statistics of a model with many low-impact ops.
* `-f`/`--focus <regexp>`: restrict the report to ops matching a regex, e.g.
  `-f onnx.MatMul`.

## Step 3: Per-shape timing with `--profile-ir-with-sig`

`--profile-ir` tells you *which op* is slow. It doesn't tell you *which
instance* of that op — the same op name can appear many times in a model with
different tensor shapes, and shape is often the reason one instance is slow
and another isn't. `--profile-ir-with-sig=<stage>` adds each instrumented op's
input/output tensor shapes to the log alongside the timing:

```bash
utils/RunONNXModel.py --model mymodel.onnx \
  --compile-args="-O3 --profile-ir-with-sig=Onnx" \
  --write-runtime-log sig.log
```

(`--profile-ir-with-sig` takes the same stage values as `--profile-ir` —
`Onnx`, or `ZHigh` on an NNPA build.)

Then report shape-signature statistics, focused on one op:

```bash
utils/make-report.py -r sig.log --stats=sig -l2 -f onnx.Transpose
```

* `--stats=sig` groups by shape signature instead of by op alone.
* `-l`/`--level <num>` controls how much of the signature is used to group
  rows:
  * `0` (default): just a count/time per op, no shape breakdown.
  * `1`: group by the first field of the signature only.
  * `2`: group by the **full** signature — every distinct combination of
    input+output shapes for that op becomes its own row, each with its own
    count and average/cumulative time. This is what you want for
    "time spent by this op for each shape instance."
  * `3`: same as `2`, plus the node name, so you can tie a row back to a
    specific instance in the IR (useful together with
    `onnx-mlir --debug-only=lowering-to-krnl` to correlate against compiler
    output).
* `-f`/`--focus <regexp>` is strongly recommended with `--stats=sig`: signature
  output for *all* ops in a model gets large fast, so focus on one op (or op
  family) at a time.

Reading the printed shapes for a single op is often enough to answer shape
questions without going back to an IR dump by hand. Here is a real report from
profiling a granite-3.1-2b decoder model on an NNPA (z17) build, adding `-w 2`
to skip the run's warmup iterations, `-u ms` for milliseconds, and `-m 0.1` to
hide ops below 0.1% of total time:

```
> make-report.py -r rz17-nnpa-fused-512.log -w 2 -u ms -m 0.1 -f onnx.Transpose -l2 --stats=sig
[...]
Statistics start all ops, 0.1%+ exec time ordered_by time, tot_time,  5.8831250
  onnx.Transpose, 81, 0.0668164, 5.4121250, 92.0%
    40, 0.1103594, 4.4143750, 81.6%, sig, 1x256x32x64xfloat, 1x32x256x64xfloat
    40, 0.0248719, 0.9948750, 18.4%, sig, 1x256x8x64xfloat, 1x8x256x64xfloat
    1, 0.0028750, 0.0028750, 0.1%, sig, 1x32x256xfloat, 1x256x32xfloat
Statistics end all ops, 0.1%+ exec time ordered_by time, tot_time,  5.8831250
```

`onnx.Transpose` alone accounts for 92% of total execution time here, and the
`-l2` breakdown shows it isn't one uniform operation: it's really three
distinct shape instances hiding behind the same op name.

* `1x256x32x64 -> 1x32x256x64` (40 calls, 81.6% of Transpose time): swaps
  dimensions 1 and 2 — a higher-dimension transpose, not the last two axes.
* `1x256x8x64 -> 1x8x256x64` (40 calls, 18.4%): the same permutation pattern,
  but on a tensor with an 8-wide instead of 32-wide second dimension — and
  its average time is about 4x lower, tracking the 4x-smaller dimension.
* `1x32x256 -> 1x256x32` (1 call, 0.1%): this one *does* permute the last two
  dimensions of a 3-D tensor.

Without `-l2`, all of this would have collapsed into a single
`onnx.Transpose, 81, ...` line — no way to tell that the vast majority of
Transpose time comes from permuting a middle dimension rather than the last
two, which matters directly on hardware with a dedicated transpose-matmul for
last-two-dim transposes (e.g. NNPA on z17). The same shape listing makes a
missing broadcast on some other op visible just as directly, since the
mismatched dimensions show up side by side in the signature.

## Other `--stats` views: `par` and `simd`

`par` and `simd` report *compile-time* decisions rather than runtime timing:
whether each op was parallelized (`par`) or vectorized (`simd`), and why not
when it wasn't. These come from a separate compiler flag, `--opt-report`, and
need a compile-time log (`-c`) in addition to a runtime log:

```bash
utils/RunONNXModel.py --model mymodel.onnx -w 2 -n 16 \
  --compile-args="-O3 --opt-report=Simd --profile-ir=Onnx" \
  --write-compile-log compile.log --write-runtime-log run.log
utils/make-report.py -c compile.log -r run.log -w 2
```

```
Statistics start (all ops).
  onnx.Add-simd, 112, 0.0130570
  onnx.Cast, 23, 0.0000650
  onnx.Gemm, 1, 0.0003570
  onnx.Gemm-simd, 72, 0.8109330
```

Ops that were successfully vectorized/parallelized are listed separately, with
a `-simd`/`-par` suffix, from instances of the same op that weren't. Combining
`-c` and `-r` like this correlates the compile-time decision with the actual
runtime cost of each case — e.g. above, `onnx.Gemm-simd` instances took far
longer in total than plain `onnx.Gemm`, but that's dominated by the fact there
are 72 of them vs. 1.

## Gathering compile-time info

Two independent ways to see what a model was actually compiled with:

* `RunONNXModel.py --write-compile-log [PATH]` (default `compile.log`) saves
  the raw onnx-mlir compiler output for that run — useful for diagnosing a
  compile failure or checking exactly which flags took effect, next to
  `--write-runtime-log` for the run itself.
* `RunONNXModel.py --print-compilation-info` prints the compiler version,
  compile options, and per-accelerator info that gets embedded into the
  compiled `.so` at compile time.

The same embedded info is also picked up automatically by `make-report.py`:
whenever it's present in the log passed via `-r`/`-c`, `make-report.py`
prints the `compiler_version`, `compile_options`, and accelerator info before
the statistics table. This means a saved `run.log` is self-describing even
without keeping a separate `compile.log` around.

## Implementation notes

A few internals that explain some of the behavior above, in case you need to
inspect a raw log directly or something doesn't line up:

* Each inference run — warmup or timed — is delimited in the log by its own
  `==START-REPORT==` marker, and `RunONNXModel.py` flushes and writes out the
  instrumentation data after every run, not just the timed ones. That's why
  `make-report.py -w <num>` needs to be told how many warmup runs to skip: it
  has no way to distinguish a warmup block from a timed one otherwise.
* Within a single run, the shape-signature text (`--profile-ir-with-sig`) is
  written unbuffered, while the timing text is written through a buffer
  that's only flushed on overflow or at the end of the run. This means the
  two kinds of lines can appear visually interleaved out of order if you
  inspect a raw log file directly — it doesn't affect `make-report.py`'s
  parsing, which re-associates timing and signature by op and node
  internally.
* Compiler/accelerator info is embedded in the log as a `==COMPILE-INFO-REPORT==`
  line whenever the model was compiled with `--profile-ir`/`--profile-ir-with-sig`;
  `make-report.py` looks for it in whatever file is passed via `-r`/`-c` and
  prints it automatically if found.

## Related pages

* [Instrumentation.md](Instrumentation.md) — the general, lower-level
  instrumentation mechanism `--profile-ir`/`--profile-ir-with-sig` are built
  on top of (background only; use `--profile-ir`/`--profile-ir-with-sig`
  directly for performance testing).
* [ProfileModel.md](ProfileModel.md) — sampling-based CPU profiling with
  `utils/profile-model.py`, for instruction-level detail.
* [Testing.md](Testing.md) — general testing guide, including
  `RunONNXModelZoo.py`.
