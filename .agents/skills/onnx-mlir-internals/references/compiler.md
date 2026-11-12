<!-- Generated from the onnx-mlir knowledge base (stack/compiler.md), verified against onnx-mlir@b505d64b. Do not edit here: see .agents/skills/README.md. -->

# onnx-mlir Compiler (selected sections)

The compiler module (`src/Compiler/`) implements the compilation driver, option parsing, and pass pipeline construction. It bridges user-facing CLI flags to the MLIR pass manager.

## Target Selection: Which `--march` Values Get SIMD and an Accelerator

One flag decides two independent things a user needs settled before compiling: which CPU vector model the compiler generates SIMD for, and which accelerator generation is available to `--maccel`. Neither failure mode is an error — an unrecognized or omitted value silently means "no SIMD," and with `--maccel=NNPA` also "nothing offloaded."

| `--march` (or `--mcpu`) value | SIMD at `-O3`? | Vector width | Accelerator available |
|---|---|---|---|
| `z14` / `arch12` | yes | 128-bit, 32 registers | none — NNPA needs z16 or later |
| `z15` / `arch13` | yes | 128-bit, 32 registers | none — NNPA needs z16 or later |
| `z16` / `arch14` | yes | 128-bit, 32 registers | `--maccel=NNPA`, hardware level M14 (Telum) |
| `z17` / `arch15` | yes | 128-bit, 32 registers | `--maccel=NNPA`, hardware level M15 (Telum II) — adds int8 quantized matmul (`--nnpa-quant-dynamic`) |
| `x86-64` | yes | 128-bit, 16 registers | none |
| `aarch64`, `arm64`, `apple-m1`…`apple-m4` | yes | 128-bit, 32 registers | none |
| `native` | depends on the host — see below | — | as per whatever value the host resolves to |
| anything else, or flag omitted | **no** | — | none |

Vector width is the full SIMD unit, so usable lanes depend on element type: 128 bits is 4 × f32, 8 × f16, 2 × f64, or 16 × int8. All four Z levels share one vector model, so moving from z14 to z17 changes what the accelerator can do, not how wide CPU vectors are.[^10]

To confirm what a given target actually produced rather than inferring it from the flag, compile with the report — SIMD decisions are per-op:

```bash
onnx-mlir -O3 --march=z16 --opt-report=Simd model.onnx
```

Three traps this table exists to close:

- **`x86-64` is the only x86 spelling recognized, and it always gets the narrower 128-bit model.** A wider AVX2 model exists in the compiler but is unreachable from the `onnx-mlir` driver, which requires a target-attribute string it never supplies.[^10] A CPU-specific value such as `--march=skylake` matches nothing at all and lands in the bottom row.
- **`--march=native` is not a safe default.** It replaces the flag's value with the host CPU's *name*, and only some names are recognized: on Apple Silicon the name is `apple-m1`…`apple-m4`, which the table's Arm row covers, but on x86 the host name is a CPU model (`skylake`, `znver3`, …) and never the literal `x86-64` — so `native` on x86 falls into the no-SIMD row.[^10] Check with `--opt-report=Simd` on the host in question before relying on it.
- **`--maccel=NNPA` at a level below z16, or with `--march` omitted entirely, is silent rather than rejected.** Every op fails the hardware-level check and stays on CPU; `--opt-report=NNPAUnsupportedOps` prints the per-op reason, including the level mismatch.[^11] NNPA additionally requires a build configured with `-DONNX_MLIR_ACCELERATORS=NNPA` and a zDNN library implementing that level — see [stack/accelerators.md](../../../../.agents/skills/nnpa-internals/references/accelerators.md).

## Performance-Relevant Flag Dependencies

Several flags only have their advertised effect when another flag — or the build itself — is set a particular way, and the compiler does not warn when a flag is silently a no-op because a dependency wasn't met. These are the ones worth checking first when "I passed a flag and nothing changed":

- **SIMD vectorization has two independent, silent gates**: an optimization-level gate (`-O3` enables SIMD; check each SIMD-related flag's own `desc()` in `CompilerOptions.cpp` rather than assuming `-O3` is the only trigger, since this can shift), and a recognized-target-machine gate. For the second: `VectorMachineSupport::setGlobalVectorMachineSupport()` in `src/Dialect/Mlir/VectorMachineSupport.cpp` maps the current `--march`/`--mcpu`/target-attr strings to a machine-support subclass — IBM Z (`ZArch12/13/14/15VectorMachineSupport`, keyed off a z/arch-number decode), x86-64 (`AVX2x86VectorMachineSupport` for skylake+avx2, else `SSE42x86VectorMachineSupport`), Arm/Apple-silicon (`NeonVectorMachineSupport`) — and falls back to `NoVectorMachineSupport` for anything unrecognized. That fallback is the "silently no SIMD" case. The per-value result of that mapping, including vector widths and which levels have an accelerator, is tabulated in [Target Selection](#target-selection-which---march-values-get-simd-and-an-accelerator) above; re-verify that table against this function on every sync, since supported targets are added over time.
- **NNPA op legality depends on the hardware level selected by `--march`/`--mcpu`**, not merely on `--maccel=NNPA` being set — see [stack/accelerators.md](../../../../.agents/skills/nnpa-internals/references/accelerators.md). The authoritative per-op answer is in `ONNXLegalityCheck.cpp`, not the auto-generated `docs/SupportedONNXOps-NNPA.md` summary (which can lag).
- **The zDNN library version linked at build time is a second, separate gate on top of the hardware-level flag.** The `--march` flag is a compiler-side *request*; whether the zDNN build actually implements that hardware level is a build-time fact tracked via the zdnn pin, the `setup_zdnn(<tag>)` call in `src/Accelerators/NNPA/CMakeLists.txt`.
- **`--parallel` depends on the build having found an OpenMP installation.** `src/CMakeLists.txt` probes for `omp.h` via `find_path(OPENMP_INCLUDE_DIR ...)` and sets `OMP_SUPPORTED` ON/OFF accordingly, with no build warning either way if it's OFF — re-check current behavior in that file (and how `OMP_SUPPORTED` is consumed downstream) before asserting what happens at runtime when `--parallel` is requested on a build without OpenMP.
- **NNPA-specific flags as a class depend on the build-time CMake accelerator gate** `-DONNX_MLIR_ACCELERATORS=NNPA`.[^4] Without it, none of the NNPA-only flags exist in `--help` at all — same *shape* of dependency as the OpenMP case above.

[^10]: repo:onnx-mlir@6e44d5a8:src/Dialect/Mlir/VectorMachineSupport.hpp (`computeArchVectorBitWidth`/`computeArchVectorRegisterNum` per subclass; `ZArch12/13/15VectorMachineSupport` are `using` aliases of `ZArch14VectorMachineSupport`, hence one model across all Z levels), repo:onnx-mlir@6e44d5a8:src/Dialect/Mlir/VectorMachineSupport.cpp (`setGlobalVectorMachineSupport`: the x86 branch requires `arch == "x86-64"` and the AVX2 branch additionally `cpu == "skylake" && attr == "avx2"`; the Arm branch's literal list and its `arch=native returns apple-mXXX` comment; `else` installs `NoVectorMachineSupport`), repo:onnx-mlir@6e44d5a8:src/Compiler/CompilerPasses.cpp (sole caller, `setGlobalVectorMachineSupport(march, mcpu, "")` — `attr` is always empty, so the AVX2 model has no reachable selector), repo:onnx-mlir@6e44d5a8:src/Compiler/CompilerOptions.cpp (`decodeZArchNum`: `arch12`/`z14` → 12 … `arch15`/`z17` → 15; `march == "native"` replaced by `llvm::sys::getHostCPUName()`). Two developer-facing oddities in the same code: the AVX2 subclass returns `258` rather than `256` for its bit width, which would trip `computeArchVectorLength`'s `simdBitSize % typeBitSize == 0` assert if it were reachable; and `VectorMachineSupport::UNSUPPORTED` is `1`, not `0`, despite the header comments saying a zero vector length means no SIMD.
[^11]: repo:onnx-mlir@6e44d5a8:src/Accelerators/NNPA/Support/NNPALimit.hpp/.cpp (`NNPALevel` enum `NONE`/`M14`/`M15`; `getNNPAFromTargetFlag` matches only `z16`/`z17`/`arch14`/`arch15`, so a z14/z15 or absent `--march` yields `NONE`, and `isCompatibleWithNNPALevel` then fails every op's level check), repo:onnx-mlir@6e44d5a8:src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXLegalityCheck.cpp (`onnxToZHighInCompatibilityReport` — emits the level-mismatch string only when `reportOnNNPAUnsupportedOps` is set, i.e. under `--opt-report=NNPAUnsupportedOps`; otherwise the fallback is silent)
[^4]: repo:onnx-mlir@f4b8e8de:docs/AccelNNPAHowToUseAndTest.md
