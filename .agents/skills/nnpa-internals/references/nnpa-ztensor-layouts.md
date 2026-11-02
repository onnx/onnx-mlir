<!-- Generated from the onnx-mlir knowledge base (concepts/nnpa-ztensor-layouts.md), verified against onnx-mlir@b505d64b. Do not edit here: see .agents/skills/README.md. -->

# NNPA ZTensor Data Layouts

Every ZTensor (`ZTensorEncodingAttr`) carries a `DataLayout` value that determines how its logical dimensions are tiled into the physical 6D stick layout NNPA hardware operates on. This page enumerates each layout's affine mapping and its known consumers.

## The Physical Tile Shape

Regardless of logical layout, every stickified tensor is mapped down to a 6-dimensional physical shape. `convertZTensorToMemRefType` names the six affine result expressions `n, h, w, c, res32, res64` — these are just local variable names for "the 6 mapped dimensions," not a separate concept from the `(e4, e3, e2, e1)` logical-dimension notation used throughout onnx-mlir's and zdnn's own comments. `res32`/`res64` are always `e2 % 32` and `e1 % 64` (or their layout-specific equivalent) — the within-stick offsets — and the outer four (`n, h, w, c`) are always a `floorDiv`/passthrough of one of `e4, e3, e2, e1`.[^code1]

A "stick" is the innermost `32 (res32) × 64 (res64)` tile — 32 sticks of 64 DLFloat16 elements each — matching the `stickLen=64` constant used by the compiler's SIMD stick/unstick codegen; see [concepts/stick-unstick-simd-codegen.md](stick-unstick-simd-codegen.md) for how that tile is iterated in generated loops.

`e2`/`e1` blocking uses `constE2Block=32`/`constE1Block=64` for the default DLFLOAT16 element type. Two quantized variants change these: INT8 uses a 128-element `constE1Block` (128 cells per stick), and the `WEIGHTS` quantized type halves `constE2Block` to 64 (two interleaved vectors per stick, rounded up to stay even).[^code1]

## Per-Layout Affine Mappings

Each entry below gives the logical dimension assignment, the mapping comment as written in source, and the resulting `(n, h, w, c, res32, res64)` expressions. `e4/e3/e2/e1` numbering follows onnx-mlir's own convention: `e4` is the outermost logical dimension, `e1` the innermost.

### `_1D`, `_2D`, `_3D`, `_4D` — plain rank-N tensors

| Layout | Logical dims | Mapping |
|--------|---------------|---------|
| `_1D` | `(e1)` | `(e1) → (1,1,1,e1) → (1, ceil(e1/64), 1, 1, 32, 64)` — `n=0, h=floorDiv(e1,64), w=0, c=0, res32=31, res64=e1%64` |
| `_2D` | `(e2, e1)` | `(e2,e1) → (1,1,e2,e1) → (1, ceil(e1/64), 1, ceil(e2/32), 32, 64)` — `n=0, h=floorDiv(e1,64), w=0, c=floorDiv(e2,32), res32=e2%32, res64=e1%64` |
| `_3D` | `(e3, e2, e1)` | `(e3,e2,e1) → (1,e3,e2,e1) → (1, ceil(e1/64), e3, ceil(e2/32), 32, 64)` — `n=0, h=floorDiv(e1,64), w=e3, c=floorDiv(e2,32), res32=e2%32, res64=e1%64` |
| `_4D` | `(e4, e3, e2, e1)` | `(e4,e3,e2,e1) → (e4, ceil(e1/64), e3, ceil(e2/32), 32, 64)` — `n=e4, h=floorDiv(e1,64), w=e3, c=floorDiv(e2,32), res32=e2%32, res64=e1%64` |

`_1D`'s `res32` is the constant `31`, not a modulo expression — since `_1D` has no `e2` dimension to place in the 32-wide sublane, the mapping pins it to the last row rather than computing an offset.

### `_2DS`, `_3DS`, `_4DS` — "special" batched variants (LSTM/GRU/Softmax/MatMul)

The `S`-suffixed layouts add a leading `e4` batch/direction dimension that plain `_2D`/`_3D`/`_4D` don't have — zdnn's own comment calls these "special ... tensors required by LSTM/GRU" (and, for `_3DS`, also Softmax/MatMul).[^code2]

| Layout | Logical dims | Mapping |
|--------|---------------|---------|
| `_2DS` | `(e4, e1)` | `(e4,e1) → (e4,1,1,e1) → (e4, ceil(e1/64), 1, 1, 32, 64)` |
| `_3DS` | `(e4, e2, e1)` | `(e4,e2,e1) → (e4,1,e2,e1) → (e4, ceil(e1/64), 1, ceil(e2/32), 32, 64)` |
| `_4DS` | `(e4, e3, e2, e1)`, `shape[1] ∈ {1,2}` | Unidirectional (`shape[1]=1`): same as `_4D` — `h=floorDiv(e1,64)`. Bidirectional (`shape[1]=2`): `h=floorDiv(2·ceilDiv(e1,64)·64, 64)` — `e1` is first padded up to a 64-multiple, then doubled, to make room for both directions' output stacked along `h` |

`_4DS` is the output layout of `ZHighLSTMOp`/`ZHighGRUOp` (`hn_output`/`cf_output`) — `shape[1]` is the direction count (1 for unidirectional, 2 for bidirectional RNNs).

### `NHWC`, `NCHW` — 4D feature tensors

| Layout | Logical dims | Mapping |
|--------|---------------|---------|
| `NHWC` | `(e4,e3,e2,e1) = (N,H,W,C)` | `(e4, ceil(e1/64), e3, ceil(e2/32), 32, 64)` — `n=e4, h=floorDiv(e1,64), w=e3, c=floorDiv(e2,32), res32=e2%32, res64=e1%64`. Identical field-for-field to `_4D` — `NHWC` is `_4D` with its dims interpreted as N/H/W/C. |
| `NCHW` | `(e4,e3,e2,e1) = (N,C,H,W)` | Comment: `(e4, ceil(e2/64), e1, ceil(e3/32), 32, 64)` — but the branch body starts with `llvm_unreachable("Not tested yet")` before computing any of `n/h/w/c/res32/res64`.[^code1] **This mapping is written in source but not exercised** — treat the values in the branch as unverified until that `unreachable` is removed. |

`ZHighStickOp`'s own doc comment states the convention: `layout=NHWC` input must already be in `NCHW` (the CPU/ONNX-side row-major convention), and stickification both tiles the data *and* permutes N/C/H/W → N/H/W/C in the same op. This is why `stack/accelerators.md`'s canonicalization section needs a dedicated `GetAxisNCHW` helper to translate an `onnx.Dim` axis back across a `zhigh.Stick` boundary — see [Layout-Restricted Optimizations](#layout-restricted-optimizations) below.[^code3]

### `HWCK` — 4D convolution kernel

| Layout | Logical dims | Mapping |
|--------|---------------|---------|
| `HWCK` | `(e4,e3,e2,e1) = (H,W,C,K)` | Comment: `HWCK (e4,e3,e2,e1) → KHWC (ceil(e1/64), e4, e3, ceil(e2/32), 32, 64)` — `n=floorDiv(e1,64)` (K, the output-channel dim, becomes the outer tile dim instead of `e4`), `h=e4, w=e3, c=floorDiv(e2,32), res32=e2%32, res64=e1%64` |

Only `ZHighConv2DOp`'s `input_kernel` operand uses `HWCK` — a distinct tiling from every other 4D layout because the tiled "batch" dimension is the kernel's output-channel count (`K`), not a leading `e4`.

### `FICO`, `ZRH` — concatenated recurrent-gate layouts

Both pack multiple per-gate matrices (forget/input/cell/output for LSTM; update/reset/hidden for GRU) into one tensor along the last dimension, then pad *each gate's slice independently* up to a 64-multiple before tiling — so gate boundaries always land on stick boundaries.

| Layout | Concatenation | Mapping (rank-2, i.e. no `e3`) |
|--------|----------------|-------|
| `FICO` | last dim = 4 × gate size `s` (`hidden_size/4`) | `s_pad = ceil(s/64)·64`, `pad_size = s_pad − s`. `h = floorDiv(shape[0]·(e1 + pad_size·floorDiv(e1,s)), 64)`, `c = floorDiv(e2,32)`, `res32 = e2%32`, `res64 = (e1 + pad_size·floorDiv(e1,s)) % 64` |
| `ZRH` | last dim = 3 × gate size `s` (`hidden_size/3`) | Same shape of formula with `s = hidden_size/3` in place of `hidden_size/4` |

`shape[0]` is the LSTM/GRU direction-count dimension (asserted to be 1 or 2) and multiplies `h` for rank-2 inputs; for rank-3 (`e3` present) it's folded to `1` instead and `w=e3` carries the extra dimension. The `pad_size·floorDiv(e1,s)` term is what inserts padding at each of the 4 (or 3) gate-boundary points as `e1` crosses each multiple of `s`, so that gate `k`'s data starts at `k·s_pad` in the tiled layout rather than `k·s`.[^code1]

`ZHighStickForLSTMOp`/`ZHighStickForGRUOp` are the ops that *produce* these layouts — they take the individual un-concatenated F32 gate tensors as separate operands (`f_gate,i_gate,c_gate,o_gate` for LSTM in that FICO order; `z_gate,r_gate,h_gate` for GRU in ZRH order) and stick+concatenate them in one step, rather than concatenating first and sticking second.[^code4]

### `BFICO`, `BZRH` — bidirectional concatenated variants (not yet implemented)

`ZHigh.td` declares `ZTensor_BFICO`/`ZTensor_BZRH` types (rank `[2,3]`, same as `FICO`/`ZRH`) as the "variant to work with bidirectional LSTM/GRU output" per zdnn's own enum comment,[^code2] but `convertZTensorToMemRefType`'s branches for both are a bare `llvm_unreachable("Unsupported layout yet")` — **no affine mapping exists yet for these two layouts** at this commit. Any op or pattern that would produce a `BFICO`/`BZRH` ztensor cannot currently be lowered through `ZHighToZLow`.[^code1]

## Layout → NNPA Instruction / zdnn Call Mapping

zdnn's own `zdnn_data_layouts` enum (`ZDNN_1D` ... `ZDNN_BIDIR_FICO`) is the runtime-facing counterpart of onnx-mlir's `ZTensorEncodingAttr::DataLayout` — the two enumerate the same set of layouts under `ZDNN_`-prefixed names (`ZDNN_BIDIR_ZRH`/`ZDNN_BIDIR_FICO` corresponding to onnx-mlir's `BZRH`/`BFICO`).[^code2] A zdnn ztensor's layout is fixed at descriptor-creation time (`zdnn_init_pre_transformed_desc(layout, ...)` / `zdnn_generate_transformed_desc_concatenated(desc, info, ...)`), not passed to each kernel call — so "which layout does instruction X use" is really "which layout does ZHigh require of X's operands," which the `ZHigh.td` op signatures pin down precisely:[^code5]

| ZHigh op | zdnn kernel(s) called | Operand layouts |
|----------|------------------------|-------------------|
| `ZHighMatMulOp` / `ZHighQuantizedMatMulOp` | `zdnn_matmul_op`, `zdnn_matmul_bcast_op`, `zdnn_matmul_transpose_op`, `zdnn_quantized_matmul_op` | `X`,`Y`,`Out`: `_2D` or `_3DS`; bias `B`: `_1D` or `_2DS`. Which of `_2D`/`_3DS` each operand takes depends on whether the multiply is broadcasting one operand across the other's stack dimension — see [concepts/nnpa-broadcasting.md](nnpa-broadcasting.md) for the four shape cases (unstacked/stacked/bcast1/bcast23) |
| `ZHighConv2DOp` | `zdnn_conv2d` | `input`,`output`: `NHWC`; `input_kernel`: `HWCK`; `input_bias`: `_1D` |
| `ZHighMaxPool2DOp` | `zdnn_maxpool2d` | `input`,`output`: `NHWC` |
| `ZHighAvgPool2DOp` | `zdnn_avgpool2d` | `input`,`output`: `NHWC` |
| `ZHighBatchNormOp` | `zdnn_batchnorm` | `input`,`output`: `NHWC`; `a`,`b`: `_1D` |
| `ZHighLSTMOp` | `zdnn_lstm` | `input`,`h0`,`c0`: `_3DS`; `input_weights`,`input_bias`,`hidden_weights`,`hidden_bias`: `FICO`; `hn_output`,`cf_output`: `_4DS` |
| `ZHighGRUOp` | `zdnn_gru` | `input`,`h0`: `_3DS`; `input_weights`,`input_bias`,`hidden_weights`,`hidden_bias`: `ZRH`; `hn_output`: `_4DS` |
| `ZHighStickForLSTMOp` | (feeds `zdnn_lstm`'s weight/bias operands; no separate kernel call itself — it's a ZHigh-level stick+concatenate) | produces `FICO` from 4 separate F32 gate tensors |
| `ZHighStickForGRUOp` | (same, feeds `zdnn_gru`) | produces `ZRH` from 3 separate F32 gate tensors |
| `ZHighReduceMax/MinOp`, `ZHighMeanReduce2DOp`, elementwise ops (`Add`/`Sub`/`Mul`/.../`Softmax`) | corresponding `zdnn_reduce`/`zdnn_meanreduce2d`/elementwise kernels | `SameOperandsAndResultLayout`-constrained — whatever layout the input already has is preserved, no fixed layout of their own |
| `ZHighStickOp` / `ZHighUnstickOp` | (no zdnn kernel — implemented either by compiler-generated SIMD or `stickify.c`, see [concepts/stick-unstick-simd-codegen.md](stick-unstick-simd-codegen.md)) | any of `_1D`,`_2D`,`_3D`,`_4D`,`_2DS`,`_3DS`,`_4DS`,`NHWC`,`NCHW`,`HWCK` |

The zdnn side's function signatures (`zdnn_matmul_op`, `zdnn_conv2d`, `zdnn_lstm`, `zdnn_gru`, `zdnn_avgpool2d`, `zdnn_maxpool2d`, `zdnn_batchnorm`) take opaque `zdnn_ztensor *` pointers rather than a layout parameter — the layout constraint each kernel expects on its operands is documented in zdnn's own `README.md`/`zdnn.h` comments, not enforced by the C signature itself; onnx-mlir's `ZHigh.td` type constraints above are what actually pin the layout at the MLIR level before lowering ever reaches the zdnn call.[^code5]

## Layout-Restricted Optimizations

- **`DimStickNHWCRemovalPattern`** (`ZHighStick.td`, part of `ZHighStickOp`'s canonicalization patterns): folds `onnx.Dim(zhigh.Stick(%X), axis)` → `onnx.Dim(%X, axis)`, but only for the `NHWC` layout does this require an extra step — remapping `axis` from NCHW numbering to NHWC numbering via `GetAxisNCHW` before the fold applies, since `zhigh.Stick` permutes dimensions under `NHWC` but not under any other layout. The layout-agnostic sibling pattern `DimStickRemovalPattern` handles every other layout without an axis remap.[^code3]
- **`FusionOpStickUnstick`** (collapses `unstick → op → stick` chains into `ONNXFusedOp` kinds, see [stack/accelerators.md](accelerators.md#op-fusion-before-lowering)): layout-restricted. The stick/layout-transform end of a chain must target `3D`, `3DS`, or `4D` for `zhigh.extended_layout_transform`, `zhigh.expand-mul-stick`, and `zhigh.concat-expand-stick`; `zhigh.mul-add-stick` accepts only a `3D` or `3DS` final stick; and `zhigh.unstick-split-heads` requires its source unstick to be `3D` or `3DS`, with each stick-mode output produced as `3DS`. Innermost-dimension requirements are separate from layout and vary per kind (listed on the accelerators page); for example, `zhigh.extended_layout_transform` needs a static innermost dimension that is a multiple of 64 or exactly 32 (a half stick, so every source half stick starts at offset 0 of its stick), via `getExtendedLayoutTransformInnerTile()`.[^code6]
- **Compiler-generated SIMD stick/unstick** ([concepts/stick-unstick-simd-codegen.md](stick-unstick-simd-codegen.md)): the codegen there is written in terms of the generic 64-element stick tile shared by every layout above (`res64`), not in terms of any one `DataLayout` — it does not special-case `NHWC`/`FICO`/etc.

No other pass or canonicalization in `src/Accelerators/NNPA/` was found (via grep for each `DataLayout` enumerator name outside `ZHighToZLow.cpp`/`ZHigh.td`) to branch on a specific layout value. If a future optimization is added with a layout restriction, add it to this list rather than only to the pass's own page — this table is meant to be the place to check "is optimization X available for layout Y."

[^code1]: repo:onnx-mlir@b505d64:src/Accelerators/NNPA/Conversion/ZHighToZLow/ZHighToZLow.cpp (`convertZTensorToMemRefType`, lines ~262-530)
[^code2]: repo:zdnn@7ff177c1:zdnn/zdnn.h (`zdnn_data_layouts` enum)
[^code3]: repo:onnx-mlir@b505d64:src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/Stick/ZHighStick.td (`DimStickRemovalPattern`, `DimStickNHWCRemovalPattern`), repo:onnx-mlir@b505d64:src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/OpHelper.cpp (`getAxisNCHW`)
[^code4]: repo:onnx-mlir@b505d64:src/Accelerators/NNPA/Dialect/ZHigh/ZHigh.td (`ZHighStickForLSTMOp`, `ZHighStickForGRUOp`, `ZHighConv2DOp`, `ZHighMaxPool2DOp`, `ZHighAvgPool2DOp`, `ZHighMatMulOp`, `ZHighBatchNormOp`, `ZHighLSTMOp`, `ZHighGRUOp`)
[^code6]: repo:onnx-mlir@5534bc7:src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/ZHighFusionOpHelper.cpp (`detectStickTail`, `ConcatExpandStickFusionHelper::detectIfBeneficial`, `UnstickSplitHeadsFusionHelper::detectIfBeneficial`, `MulAddStickFusionHelper::detectFromJoin`), repo:onnx-mlir@5534bc7:src/Accelerators/NNPA/Dialect/ZHigh/ZHighOps/OpHelper.cpp (`getExtendedLayoutTransformInnerTile`)
[^code5]: repo:zdnn@7ff177c1:zdnn/zdnn.h (`zdnn_matmul_op`, `zdnn_conv2d`, `zdnn_lstm`, `zdnn_gru`, `zdnn_avgpool2d`, `zdnn_maxpool2d`, `zdnn_batchnorm`, `zdnn_init_pre_transformed_desc`, `zdnn_generate_transformed_desc_concatenated`)
