<!-- Generated from the onnx-mlir knowledge base (concepts/nnpa-broadcasting.md), verified against onnx-mlir@b505d64b. Do not edit here: see .agents/skills/README.md. -->

# NNPA Broadcasting Support

Which NNPA operations tolerate input operands of different shapes, and what happens when they don't match exactly. This page treats **zdnn as the primary source of truth** and **onnx-mlir's `ONNXToZHigh`/`ONNXLegalityCheck` as a secondary, current implementation of what zdnn actually supports** — the reverse of this wiki's usual onnx-mlir-first framing, deliberate here because onnx-mlir's job is to correctly *reflect* zdnn's hardware contract, not to define it. Where the two disagree or only approximately match, that is called out explicitly rather than silently describing onnx-mlir's behavior as if it were the spec.

## zdnn's Broadcasting Model (primary source)

### Elementwise binary ops: zero broadcast support

zdnn's own docs are explicit and identical, word-for-word, across every elementwise binary kernel: *"broadcasting of the input tensor(s) must be performed by the caller. As such, the input tensors must be of the same shape."*[^z1] This is not a limitation of one op — it's stated for all six:

| zdnn kernel | Broadcast support |
|---|---|
| `zdnn_add` | None — caller must pre-broadcast to equal shapes |
| `zdnn_sub` | None |
| `zdnn_mul` | None |
| `zdnn_div` | None |
| `zdnn_min` | None |
| `zdnn_max` | None |

There is no scalar-broadcast special case anywhere in zdnn's own API for these six — the hardware kernel simply requires two identically-shaped `zdnn_ztensor` inputs.[^z1]

### MatMul family: the only kernels with native broadcast

Matrix multiplication is where zdnn actually implements broadcast at the hardware/API level, via a dedicated `_bcast_op`/`_transpose_op` variant rather than a flag on the base kernel:

| zdnn kernel | `input_a` | `input_b` | `input_c` | `result` | Case name |
|---|---|---|---|---|---|
| `zdnn_matmul_op` | `ZDNN_2D (m,n)` | `ZDNN_2D (n,p)` | `ZDNN_1D (p)` | `ZDNN_2D (m,p)` | unstacked |
| `zdnn_matmul_op` | `ZDNN_3DS (s,m,n)` | `ZDNN_3DS (s,n,p)` | `ZDNN_2DS (s,p)` | `ZDNN_3DS (s,m,p)` | stacked |
| `zdnn_matmul_bcast_op` / `zdnn_matmul_transpose_op` | `ZDNN_2D (m,n)` | `ZDNN_2D (n,p)` | `ZDNN_1D (p)` | `ZDNN_2D (m,p)` | unstacked |
| `zdnn_matmul_bcast_op` / `zdnn_matmul_transpose_op` | `ZDNN_3DS (s,m,n)` | `ZDNN_3DS (s,n,p)` | `ZDNN_2DS (s,p)` | `ZDNN_3DS (s,m,p)` | stacked |
| `zdnn_matmul_bcast_op` / `zdnn_matmul_transpose_op` | `ZDNN_2D (m,n)` | `ZDNN_3DS (s,n,p)` | `ZDNN_2DS (s,p)` | `ZDNN_3DS (s,m,p)` | **bcast1** — `input_a` broadcast over `input_b`'s stack |
| `zdnn_matmul_bcast_op` / `zdnn_matmul_transpose_op` | `ZDNN_3DS (s,m,n)` | `ZDNN_2D (n,p)` | `ZDNN_1D (p)` | `ZDNN_3DS (s,m,p)` | **bcast23** — `input_b`/`input_c` broadcast over `input_a`'s stack |

`zdnn_matmul_transpose_op` supports all four of the same cases as `zdnn_matmul_bcast_op` (it is the transpose-capable superset — `zdnn_matmul_op` has no transpose parameter at all).[^z2]

Two hardware/API gates apply, both stated directly in zdnn's docs rather than inferred:
- `zdnn_matmul_transpose_op` as a whole "is not supported when `NNPA_PARMBLKFORMAT_1` is not installed and will return `ZDNN_UNAVAILABLE_FUNCTION`."[^z3]
- Within `zdnn_matmul_bcast_op`, the **bcast1** case specifically requires "the underlying hardware supports zDNN APIs 1.1.x or later at runtime" for `bcast1`'s `zdnn_data_layouts` entry and several `zdnn_matmul_bcast_ops` comparison op-types — a narrower, later-added gate than the op's own "Since 1.0.0."[^z4]

No other zdnn kernel (`zdnn_conv2d`, `zdnn_lstm`, `zdnn_gru`, `zdnn_avgpool2d`/`zdnn_maxpool2d`, `zdnn_batchnorm`, `zdnn_reduce`) documents any broadcast behavior — each takes a fixed-role set of operands with a fixed relationship between their shapes (e.g. `zdnn_conv2d`'s `input`/`kernel`/`bias` shapes are determined by the convolution parameters, not independently broadcastable), so broadcasting as a concept doesn't apply to them.

## onnx-mlir's Implementation (secondary source)

### Elementwise ops: legality checks enforce zdnn's exact-shape rule faithfully

`ONNXLegalityCheck.cpp`'s `isSuitableForZDNN<...>` specializations for `ONNXAddOp`/`ONNXSubOp`/`ONNXMulOp`/`ONNXMinOp`/`ONNXMaxOp`/`ONNXSumOp` all reject the op unless `DimAnalysis::sameShape()` confirms both operands have the same shape — this is a direct, correct implementation of zdnn's "must be of the same shape" requirement, not an approximation of it.[^m1]

`ONNXDivOp` has one apparent exception — a scalar operand is accepted even when its shape differs from the other operand, gated by `--nnpa-enable-scalar-bcast-binary` (default off) — but this is **not** onnx-mlir extending zdnn's broadcast capability. `replaceONNXDivBroadcastPattern1`/`2` (`ONNXToZHigh.td`) implement it by materializing the scalar into a full same-shape tensor via `ZHighStickifiedConstantOfShapeOp` *before* the op reaches `ZHighDivOp`/`zdnn_div` — i.e. onnx-mlir performs the "caller must pre-broadcast" step zdnn's own docs require, in software, rather than asking the hardware to do something it doesn't support.[^m2] `zdnn_div` itself still only ever sees two identically-shaped ztensors.

Attention masks are the common case where an elementwise `Add` broadcasts in practice (a `[B,1,1,S]`-style mask added to `[B,H,S,S]` scores), and onnx-mlir pre-broadcasts them in software before legality is checked. `ExpandAttentionMaskPass` runs once at the start of the NNPA ONNX-to-ZHigh pipeline and, when one mask feeds at least four `MatMul → [Mul] → Add → Softmax` chains whose MatMul results all have the same shape (checked with `DimAnalysis`), replaces the mask with a single `onnx.Expand` to that shape, so each `Add` then passes the same-shape rule above. A scalar attention scale multiplied onto at least four such MatMul results is expanded the same way, so the `Mul`s also stop broadcasting. `--nnpa-disable-expand-attention-mask` turns it off; see [stack/accelerators.md](accelerators.md#onnx-level-rewrites-before-zhigh-lowering).[^m8]

### MatMul family: legality checks mirror zdnn's four cases exactly

`isSuitableForZDNN<ONNXMatMulOp>` branches on `(shapeA.rank(), shapeB.rank())` and names its four cases "Unstacked," "Stacked w/o bcast," "Bcast23," and "Bcast1" — the same four cases, in the same shape combinations, as zdnn's `matmul-bcast-io-table` above.[^m3] The **bcast1** case additionally requires `isCompatibleWithNNPALevel(NNPALevel::M15)`, matching zdnn's later-API-level gate for that case.[^m3]

Transpose is handled separately: `replaceONNXTransAMatMulPattern`/`TransB`/`TransAB` (`ONNXToZHigh.td`) rewrite `onnx.MatMul(onnx.Transpose(...), ...)` into `ZHighMatMulOp` with `transposeA`/`transposeB` set, gated by `IsCompatibleWithNNPALevelArch15` (M15) — correctly matching zdnn's requirement that `zdnn_matmul_transpose_op` needs `NNPA_PARMBLKFORMAT_1`.[^m4] At the runtime dispatch layer, `Runtime/zdnnx/seq_ops.c`'s `call_zdnn_matmul_op()` picks `zdnn_matmul_transpose_op` whenever either transpose flag is set (regardless of whether the shapes are also a broadcast case), else `zdnn_matmul_bcast_op` if the shapes are a broadcast case, else the plain `zdnn_matmul_op`[^m5] — a direct three-way mirror of zdnn's own kernel split, not a reimplementation of the shape logic.

`ONNXGemmOp`'s legality check handles the classic "bias broadcast" case (`C(p)` added to an `(m,p)` result) by requiring `C`'s rank-1 shape to equal `B`'s second dimension — this is the same `ZDNN_1D(p)` bias slot in zdnn's "unstacked" `matmul_op`/`matmul_bcast_op` row above, not a separate broadcast mechanism. A rank-2 `C` that doesn't fit that slot is not handled by `ZHighMatMulOp`'s own bias operand at all; it falls through to a plain `ZHighAddOp` afterward, which is subject to the elementwise "no broadcast" rule above unless `RewriteONNXForZHigh.cpp`'s `ExpandAddConstantPattern` can first rewrite it into one of the layout-legal stacked/bcast shapes it recognizes (`unstacked`, `bcast23`, `stacked`, `bcast1` — reusing the same case names again).[^m6]

## Where the Implementation Only Approximates zdnn

- **bcast1's hardware gate is checked two different ways at two different times, and they are assumed — not proven — equivalent.** zdnn's own requirement for `bcast1` is a *runtime* capability query: "the underlying hardware supports zDNN APIs 1.1.x or later at runtime," which onnx-mlir's own runtime layer (`Runtime/zdnnx/zdnnx.c`) checks live via `zdnn_is_nnpa_parmblk_fmt_installed(1, NNPA_PARMBLKFORMAT_1)`.[^m7] But the *compile-time* legality check that decides whether to offload a bcast1-shaped MatMul to NNPA at all never calls that function — it instead checks the static `NNPALevel::M15` architecture enum (i.e. "was this compiled with `--march=z17`/`arch15`").[^z3][^m3] In practice these are assumed to coincide (every M15/Telum II part is expected to have `NNPA_PARMBLKFORMAT_1`), but the compile-time gate is a static assumption about a fact zdnn itself treats as a runtime-queryable hardware property — if that assumption is ever wrong for some M15 stepping, or a future M16+ level ships without it, this is the place the compiler's legality check would silently diverge from what the hardware actually supports at runtime.
- No other divergence was found while grounding this page — the elementwise "no broadcast" rule and the four-case MatMul table both match zdnn's documented behavior exactly, case name for case name.

[^z1]: repo:zdnn@7ff177c1:README.md (`zdnn_add`, `zdnn_sub`, `zdnn_mul`, `zdnn_div`, `zdnn_min`, `zdnn_max` — "Note that for zDNN use, broadcasting of the input tensor(s) must be performed by the caller...")
[^z2]: repo:zdnn@7ff177c1:README.md (`#matmul-io-table`, `#matmul-bcast-io-table`, `#matmul-transpose-io-table`)
[^z3]: repo:zdnn@7ff177c1:README.md (`zdnn_matmul_transpose_op` Programming Notes: "is not supported when `NNPA_PARMBLKFORMAT_1` is not installed")
[^z4]: repo:zdnn@7ff177c1:README.md (`zdnn_matmul_bcast_op` Requirements: "hardware supports zDNN APIs 1.1.x or later at runtime for... bcast1... `MATMUL_BCAST_OP_GREATER`...")
[^m1]: repo:onnx-mlir@b505d64:src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXLegalityCheck.cpp (`isSuitableForZDNN<ONNXAddOp/ONNXSubOp/ONNXMulOp/ONNXMinOp/ONNXMaxOp/ONNXSumOp>`)
[^m2]: repo:onnx-mlir@b505d64:src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXToZHigh.td (`replaceONNXDivBroadcastPattern1`, `replaceONNXDivBroadcastPattern2`)
[^m3]: repo:onnx-mlir@b505d64:src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXLegalityCheck.cpp (`isSuitableForZDNN<ONNXMatMulOp>`)
[^m4]: repo:onnx-mlir@b505d64:src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXToZHigh.td (`replaceONNXTransAMatMulPattern`, `replaceONNXTransBMatMulPattern`, `replaceONNXTransABMatMulPattern`, `IsCompatibleWithNNPALevelArch15`)
[^m5]: repo:onnx-mlir@b505d64:src/Accelerators/NNPA/Runtime/zdnnx/seq_ops.c (`call_zdnn_matmul_op`)
[^m6]: repo:onnx-mlir@b505d64:src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXLegalityCheck.cpp (`isSuitableForZDNN<ONNXGemmOp>`), repo:onnx-mlir@b505d64:src/Accelerators/NNPA/Conversion/ONNXToZHigh/RewriteONNXForZHigh.cpp (`ExpandAddConstantPattern`)
[^m7]: repo:onnx-mlir@b505d64:src/Accelerators/NNPA/Runtime/zdnnx/zdnnx.c (`zdnn_is_nnpa_parmblk_fmt_installed` call)
[^m8]: repo:onnx-mlir@b505d64:src/Dialect/ONNX/Transforms/ExpandAttentionMask.cpp (`detectAttentionPattern`, `patternsByMask`/`patternsByScale`, `minUses = 4`), repo:onnx-mlir@b505d64:src/Accelerators/NNPA/Compiler/NNPACompilerUtils.cpp (`addONNXToZHighPasses`)
