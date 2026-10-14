# Current ONNXFusedOp Fusion Patterns

This page lists the op chains that the compiler currently wraps in an
`onnx.Fused` op. For each fusion kind it gives the pattern matched, the
conditions under which it fires, the model idiom it targets, and what the
dedicated lowering does differently from lowering the ops one by one.

To add a new kind, or for details on the infrastructure (`ONNXFusedOp`,
`FusionOpKindHelper`, `FusedPatternForOpKind`, `FusedOpKindLowering`), see
[AddingAFusionPattern.md](AddingAFusionPattern.md).

## Summary

The six fused kinds below are described in §1 (ONNX) and §2 (ZHigh). In the
chains, `?` marks an optional op, *or* separates alternatives, and LT stands
for LayoutTransform.

| Kind | Anchor | Chain (short) | Inner dim | Typical source |
|---|---|---|---|---|
| [`simd-split-op-gather`](#simd-split-op-gather) (§1.1) | Concat | 2× (Slice → op?) → Concat | static | RoPE `rotate_half` |
| [`zhigh.extended_layout_transform`](#zhigh-extended-layout-transform) (§2.1) | LayoutTransform | LT → Reshape? → Transpose? → Reshape? → (LT *or* DLF16ToF32 → Mul?) | %64 or 32 | head reshuffles between NNPA ops |
| [`zhigh.expand-mul-stick`](#zhigh-expand-mul-stick) (§2.2) | Unsqueeze | Unsqueeze → Expand → Mul? → Reshape → Stick | %64 | GQA/MQA head repeat |
| [`zhigh.concat-expand-stick`](#zhigh-concat-expand-stick) (§2.3) | Concat | Concat → Unsqueeze → F32ToDLF16? → Expand → Mul? → Reshape → (Stick *or* LT) | %64 | KV cache + GQA head repeat |
| [`zhigh.unstick-split-heads`](#zhigh-unstick-split-heads) (§2.4) | Unstick | Unstick → Reshape → Transpose? → Split (→ Squeeze → Reshape → Stick) | D = 32 or %64 | fused QKV split into heads |
| [`zhigh.mul-add-stick`](#zhigh-mul-add-stick) (§2.5) | Add / Sub | 2× Mul → (Add *or* Sub) → Mul? → Reshape → Stick | 32 or %64 | rotary embedding before attention |

## How fusion works

Fusion happens in two steps:

- **Creation.** A pattern anchored on one op of the chain walks the
  surrounding ops. If the chain matches and fusing pays off, the pattern
  moves the ops into the body of an `onnx.Fused` op. The fused op stores the
  chain's parameters as attributes, and its `kind` attribute names the
  pattern. All dynamic-dim equalities that the lowering relies on are proven
  at this point, using `DimAnalysis`.
- **Lowering.** A dedicated lowering, chosen by `kind`, checks that the body
  still matches the stored parameters, then emits one loop nest for the
  whole chain.
  - If the body no longer matches, or no lowering exists for the kind, a
    catch-all fallback inlines the body. The ops then lower one at a time,
    so the result is still correct, just without the fused code.
  - `--disable-fused-op` turns off all fused-op creation.

Fusion runs in a single pass, late in the pipeline:

- CPU-only builds use `FusionOpTransform`, which registers the ONNX kinds.
- NNPA builds use `FusionOpStickUnstick`, which registers the ZHigh kinds and
  also merges in the ONNX kinds.

A chain diagram such as `A → B? → C` means that op `A` feeds `B`, and `B`
feeds `C`. A `?` marks an optional op. Unless stated otherwise, every
intermediate result in a chain must have exactly one use.

> **Note.** `FusionOpStickUnstick` also folds a Stick or Unstick into an
> adjacent elementwise op or LayerNorm, so that the op reads or writes the
> zTensor directly. That rewrite edits the ops in place and does not create
> an `onnx.Fused` op, so it is not covered here.

---

## 1. ONNX (CPU) kinds

Source files:
- Helpers: `src/Dialect/ONNX/Transforms/ONNXFusionOpHelper.{hpp,cpp}`
- Pattern registration: `src/Dialect/ONNX/Transforms/FusionOpTransform.cpp`
- Lowering: `src/Conversion/ONNXToKrnl/Tensor/`

<a id="simd-split-op-gather"></a>

### 1.1 `simd-split-op-gather`: RoPE `rotate_half`

```
            ┌─ Slice [0, k)  → op? ─┐
  source ───┤                       ├─→ Concat (axis = innermost)
            └─ Slice [k, D)  → op? ─┘
```

- **Idiom.** `rotate_half(x) = concat(-x[..., D/2:], x[..., :D/2])` in rotary
  position embeddings. More generally: split a tensor in two along its
  innermost dim, optionally transform each half, and concatenate the halves
  back in either order.
- **Matched when:**
  - The Concat (the anchor) has exactly two inputs, and its axis is the
    innermost dim.
  - Each input traces back to a dense Slice (step 1) of one shared source.
  - The two Slices are contiguous and together cover the whole axis.
  - The split point and the axis size are static.
- **Per-half op.** Each half may pass through at most one elementwise op.
  The op is either unary, or binary with one extra operand. That operand must
  have exactly the shape of the half and must not be the other half's Slice.
- **Lowering** (`FusedSplitOpGather.cpp`). Emits one or two SIMD loops. Each
  loop reads its half of the source and writes it at the right offset of the
  output. The two Slice results are never allocated, and neither is any
  intermediate before the Concat.

---

## 2. ZHigh (NNPA) kinds

Source files:
- Helpers: `src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/ZHighFusionOpHelper.{hpp,cpp}`
- Pattern registration: `src/Accelerators/NNPA/Transform/ZHigh/FusionOpStickUnstick.cpp`
- Lowering: `src/Accelerators/NNPA/Conversion/ZHighToZLow/ZHighToZLow.cpp`

All ZHigh kinds share three properties:

- **Kind names** start with `zhigh.`.
- **Purpose.** Each kind removes CPU work around the accelerator: converting
  between the stickified dlf16 zTensor format and F32, and reshuffling data
  with Reshape, Transpose, Expand, Split or Concat.
- **Lowering.** The fused lowering writes the final layout directly, usually
  one stick (64 values) or half stick (32 values) at a time, and never
  allocates the intermediate F32 tensors.

Two shape conditions recur below:

- **Innermost dim.** Most kinds need a static innermost dim that is a
  multiple of 64 (full sticks). Kinds marked *(32 ok)* also accept an
  innermost dim of exactly 32 (half sticks).
- **Scalar Mul.** An optional "scalar Mul" is an element-wise multiply by a
  constant F32, I32 or I64 scalar, folded into the store. A typical use is
  attention scaling. When the Mul is absent, the lowering skips the multiply.

<a id="zhigh-extended-layout-transform"></a>

### 2.1 `zhigh.extended_layout_transform`: re-layout of a zTensor *(32 ok)*

```
LayoutTransform (zTensor → CPU)
        ↓
Reshape?    (split one dim into two)
        ↓
Transpose?  (last dim stays last)
        ↓
Reshape?    (merge two dims into one)
        │
        ├── zTensor ending:  LayoutTransform (CPU → zTensor)
        │   or
        └── F32 ending:      DLF16ToF32 → scalar Mul?
```

- **Idiom.** Attention head reshuffles between two NNPA ops: the transposes
  that split heads or merge them back, done on data that is already
  stickified.
- **Anchor.** The first `LayoutTransform`.
- **Matched when:**
  - The source layout can be handled by compiler-generated stick/unstick.
  - The innermost dim is static and is a multiple of 64, or exactly 32.
- **Ending.** The chain ends in one of two ways:
  - back into a zTensor, with a new layout;
  - or as an F32 CPU tensor, with an optional scalar Mul that must not
    broadcast.
- **Lowering.** Each stick is moved from its source position to its
  destination position. When both ends are zTensors, the data stays in dlf16.
  The innermost dim is processed in tiles of 64, or 32 for half sticks.
  - Half sticks are allowed only when the innermost dim is exactly 32. Every
    source half stick then starts at offset 0 of its stick.
  - The destination may start mid-stick. For example, when 12 heads of 32
    are merged into 384, each output stick is written as two half sticks.
- **Fallback.** With `--disable-fused-op`, the same chain is rewritten into
  the composite op `zhigh.ExtendedLayoutTransform` instead.

<a id="zhigh-expand-mul-stick"></a>

### 2.2 `zhigh.expand-mul-stick`: broadcast-then-stick

```
Unsqueeze (axis P) → Expand (dim P: 1 → N) → scalar Mul? → Reshape → Stick (3D/3DS/4D)
```

- **Idiom.** Repeating KV heads for grouped-query or multi-query attention
  (GQA/MQA) before the attention MatMul.
- **Anchor.** The `Unsqueeze`.
- **Matched when:**
  - `N` is static and at least 2.
  - The Reshape collapses only dims `0..P`; the dims after `P` keep their
    sizes.
  - The innermost dim is a multiple of 64.
- **Lowering.** Loops over the input before the Unsqueeze. Each value is
  scaled and converted once, then stored to all `N` stickified locations it
  is broadcast to. The expanded tensor is never allocated.

<a id="zhigh-concat-expand-stick"></a>

### 2.3 `zhigh.concat-expand-stick`: KV-cache concat + head repeat

```
Concat (2 inputs, axis A, not innermost)
        ↓
Unsqueeze (axis P ≤ A)
        │
        ├── LT ending:     F32ToDLF16 → Expand (dim P: 1 → N) → Reshape
        │                    → LayoutTransform (CPU → zTensor, 3D/3DS/4D)
        │   or
        └── Stick ending:  Expand (dim P: 1 → N) → scalar Mul? → Reshape
                             → Stick (3D/3DS/4D)
```

- **Idiom.** In a decoder, the new keys and values are concatenated to the
  KV cache, then the KV heads are repeated for GQA. This kind extends §2.2
  with the Concat in front.
- **Anchor.** The `Concat`.
- **Matched when:** the Concat has two inputs, each with an innermost dim
  that is a multiple of 64. The Expand and Reshape conditions are the same as
  in §2.2.
- **Scalar Mul.** Allowed only in the Stick ending. In the LayoutTransform
  ending, the data is already dlf16 at that point.
- **Extra uses of the Concat result.** Unlike the rest of the chain, the
  Concat result may have other uses, typically the updated KV cache passed to
  the next step. In that case the fused op gets a second result, which is the
  Concat result itself.
- **Lowering.**
  - An outer loop runs over the dims before `A`.
  - Inside it, two tiled loop nests run back to back, one per Concat input.
    Each one fans its values out to the `N` stickified locations, as in §2.2.
  - When the Concat result has other uses, two plain copy loops also write
    the concatenated tensor.
- **Ordering.** This kind can absorb an entire §2.2 chain. It therefore runs
  in an earlier, separate round of the pass, so that §2.2 does not fuse that
  chain first.

<a id="zhigh-unstick-split-heads"></a>

### 2.4 `zhigh.unstick-split-heads`: split fused QKV into heads *(32 ok)*

```
Unstick (3D/3DS, (A,S,C))
        ↓
Reshape (A,S,N,H,D)
        ↓
Transpose?  (D stays last)
        ↓
Split (N outputs of size 1 along the N axis)
        │
        │   each of the N outputs, independently:
        │
        ├── "f32" mode:        output = the Split result (F32)
        │   or
        └── "stick-3DS" mode:  Squeeze → Reshape (A·H, S, D) → Stick (3DS)
                               output = the Stick result (zTensor)
```

- **Idiom.** A single QKV projection MatMul whose result is split into Q, K
  and V. Each of these is reshaped and transposed into heads. Here `N = 3`,
  `H` is the number of heads and `D` the head dim.
- **Anchor.** The `Unstick`.
- **Matched when:**
  - The Reshape leaves `A` and `S` unchanged and splits `C` into
    `N · H · D`, all static.
  - `D == 32` or `D % 64 == 0`.
  - `(H · D) % 64 == 0`.
- **Outputs.** There is one output per Split result, each with its own mode:
  - `f32`: the Split result, an F32 tensor. The Squeeze that usually follows
    stays outside the fused op; it lowers to a zero-copy
    `memref.reinterpret_cast`.
  - `stick-3DS`: chosen when the Split result feeds only the chain
    Squeeze → Reshape → 3DS Stick. That chain is pulled into the body. It
    requires the transpose to order the dims as `(A,H,N,S,D)` or
    `(A,N,H,S,D)`.
- **Lowering.**
  - One parallel loop nest over `(a, stick column, s)` writes all `N`
    outputs, and every read and write is a stream.
  - Each input stick is read once.
  - For an `f32` output, the stick is converted and written as two 32-value
    halves, each half to its own output row.
  - For a `stick-3DS` output, the stick is copied as dlf16 with no
    conversion: there is no dlf16 → f32 → dlf16 round trip. When `D == 32`,
    the upper half of each output stick is left uninitialized, exactly as by
    the Stick it replaces.

<a id="zhigh-mul-add-stick"></a>

### 2.5 `zhigh.mul-add-stick`: rotary embedding then stick *(32 ok)*

```
Mul (A0·A1) ─┐
             ├→ Add or Sub
Mul (B0·B1) ─┘      │
                    ↓
               scalar Mul?
                    ↓
            Reshape (rank 3)
                    ↓
            Stick (3D / 3DS)
```

- **Idiom.** Applying rotary embeddings to Q or K right before attention:
  `(x·cos ± rotate_half(x)·sin) · k`.
  - The ops that produce the four Mul operands stay outside the fused op.
    In the rotary case these are a Squeeze, a `simd-split-op-gather` fused
    op (§1.1), and the cos/sin tables, which have many uses.
  - The two kinds therefore compose: §1.1 produces `rotate_half(x)`, and this
    kind consumes it.
- **Anchor.** The `Add` or the `Sub` (one pattern registered for each).
  Matching walks back to the two Muls, then forward to the Stick.
- **Matched when:**
  - All operands are F32 and not constants.
  - Each operand broadcasts to the result shape only through static size-1
    dims (for example, cos/sin tables shared across the batch and heads).
  - The innermost dim `D` is 32 or a multiple of 64.
  - The Reshape keeps the last dim and collapses at most one run of dims.
- **Lowering.** Iterates over the result of the Add or Sub, with `D` tiled
  by full or half sticks. Each operand is read once, the expression is
  computed and converted to dlf16, and the result is stored straight into the
  stick. None of the four F32 intermediates is allocated.

## Tests

Each kind has lit tests for detection and for lowering:

- **ONNX kinds**
  - Detection: `test/mlir/onnx/onnx_fusion_op_transform.mlir`
  - Lowering: `test/mlir/conversion/onnx_to_krnl/Tensor/FusedSplitOpGather*.mlir`
- **ZHigh kinds**
  - Detection: `test/mlir/accelerators/nnpa/transform/zhigh-fused-*.mlir` and
    `zhigh-fold-op-stick-unstick-*.mlir`
  - Lowering: `test/mlir/accelerators/nnpa/conversion/onnx-to-krnl/`, which
    holds `expand-mul-reshape.mlir`, `concat-expand-stick*.mlir`,
    `unstick-split-heads.mlir`, `mul-add-stick.mlir`, and
    `onnx-on-ztensor-dlf16-extended-layout-transform*.mlir`
