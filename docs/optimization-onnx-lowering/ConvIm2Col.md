<!--- SPDX-License-Identifier: Apache-2.0 -->
# Conv Lowering with Im2Col and MatMul

**Status:** Enabled for supported [`Conv`](../onnx-mlir/src/Dialect/ONNX/ONNXOps/NN/Conv.cpp) configurations.

## Overview

This document describes the common lowering pattern that rewrites an ONNX [`Conv`](../onnx-mlir/src/Dialect/ONNX/ONNXOps/NN/Conv.cpp) into:

1. [`Im2Col`](../onnx-mlir/src/Conversion/ONNXToKrnl/Additional/Im2Col.cpp): extract each convolution window into a column.
2. [`MatMul`](../onnx-mlir/src/Conversion/ONNXToKrnl/Math/MatMul.cpp): multiply reshaped weights by these columns.
3. [`Reshape`](../onnx-mlir/src/Dialect/ONNX/ONNXOps/Tensor/Reshape.cpp): restore the convolution output shape.

The key idea is that convolution can be expressed as matrix multiplication once the input patches and weights are laid out appropriately.

## Main Idea

For a 2-D convolution with input shape:

```text
X: [N, C, H, W]
W: [M, C, KH, KW]
```

the output has shape:

```text
Y: [N, M, OH, OW]
```

where `M` is the number of output channels.

Instead of directly computing:

```text
Y[n, m, oh, ow] =
  sum over c, kh, kw of
    X[n, c, ih, iw] * W[m, c, kh, kw]
```

we rewrite the computation as:

- Flatten each sliding input window into one column.
- Flatten each filter into one row.
- Multiply the filter matrix by the patch matrix.

This gives the same result, but maps the computation to a dense matrix multiplication.

## Shapes Used by the Lowering

For the simple 2-D case:

- Input patches are packed into:

```text
X_col: [N, C * KH * KW, OH * OW]
```

- Weights are reshaped into:

```text
W_row: [M, C * KH * KW]
```

Then for each batch `n`, we compute:

```text
Y_mat[n] = W_row * X_col[n]
```

with result shape:

```text
Y_mat[n]: [M, OH * OW]
```

Finally, we reshape:

```text
Y_mat: [N, M, OH * OW] -> Y: [N, M, OH, OW]
```

## Im2Col Construction

[`Im2Col`](../onnx-mlir/src/Conversion/ONNXToKrnl/Additional/Im2Col.cpp) builds one column for each output spatial position.

For output position `(oh, ow)`, the input origin is:

```text
ihBase = oh * strideH - padTop
iwBase = ow * strideW - padLeft
```

Then for each channel `c` and kernel coordinate `(kh, kw)`, the sampled input position is:

```text
ih = ihBase + kh * dilationH
iw = iwBase + kw * dilationW
```

and the corresponding row inside the column is:

```text
p = c * KH * KW + kh * KW + kw
```

while the column number is:

```text
q = oh * OW + ow
```

So the lowered operation fills:

```text
X_col[n, p, q] = X[n, c, ih, iw]
```

or zero when `(ih, iw)` is outside the input bounds.

In the simple formulation, this can be written as a direct nested loop over
batch, channel, kernel, and output coordinates. In the implementation work we
just did, the same mapping is reorganized around one large outer loop over the
output columns:

```text
linearCol in [0, N * OH * OW)
n  = linearCol / (OH * OW)
q  = linearCol % (OH * OW)
oh = q / OW
ow = q % OW
```

This layout has two advantages:

- It computes one complete output column at a time.
- It exposes a natural outer loop that can be parallelized.

For each output column, we first compute the input origin:

```text
ihBase = oh * strideH - padTop
iwBase = ow * strideW - padLeft
```

Then we split the work into two cases:

- Interior path: the full kernel window is known to be in bounds.
- Border path: some kernel points may be outside the input and require zero fill.

The interior path avoids per-element bounds checks. The border path keeps the
same logical layout, but checks each sampled `(ih, iw)` before loading the
input value.

## Worked Example

Consider:

```text
X shape = [1, 1, 4, 4]
W shape = [2, 1, 3, 3]
stride = [1, 1]
pads = [1, 1, 1, 1]
dilation = [1, 1]
```

Then:

```text
OH = 4
OW = 4
C * KH * KW = 1 * 3 * 3 = 9
OH * OW = 16
```

So:

```text
X_col: [1, 9, 16]
W_row: [2, 9]
Y_mat: [1, 2, 16]
Y:     [1, 2, 4, 4]
```

For the output position `(oh=0, ow=0)`, the receptive field starts at:

```text
ihBase = 0 * 1 - 1 = -1
iwBase = 0 * 1 - 1 = -1
```

So the 3x3 window overlaps the padded border. The first column of [`X_col`](../onnx-mlir/src/Conversion/ONNXToKrnl/Additional/Im2Col.cpp) becomes:

```text
[0, 0, 0,
 0, X[0,0,0,0], X[0,0,0,1],
 0, X[0,0,1,0], X[0,0,1,1]]
```

For an interior output position such as `(oh=1, ow=1)`, the receptive field is fully in bounds, and the corresponding column is:

```text
[X[0,0,0,0], X[0,0,0,1], X[0,0,0,2],
 X[0,0,1,0], X[0,0,1,1], X[0,0,1,2],
 X[0,0,2,0], X[0,0,2,1], X[0,0,2,2]]
```

Each filter is also flattened. For example, the first filter becomes:

```text
W_row[0] =
[W[0,0,0,0], W[0,0,0,1], W[0,0,0,2],
 W[0,0,1,0], W[0,0,1,1], W[0,0,1,2],
 W[0,0,2,0], W[0,0,2,1], W[0,0,2,2]]
```

Then:

```text
Y_mat[0, 0, q] = dot(W_row[0], X_col[0, :, q])
Y_mat[0, 1, q] = dot(W_row[1], X_col[0, :, q])
```

for each output column `q` in `[0, 16)`.

## Why This Lowering Is Useful

This pattern is useful because it converts convolution into a regular dense linear algebra operation.

Benefits include:

- Reuse of optimized [`MatMul`](../onnx-mlir/src/Conversion/ONNXToKrnl/Math/MatMul.cpp) lowering and code generation.
- Simpler loop structure after the data layout transform.
- A clear separation between:
  - patch extraction, and
  - numerical multiply-accumulate work.

The cost is the temporary [`Im2Col`](../onnx-mlir/src/Conversion/ONNXToKrnl/Additional/Im2Col.cpp) buffer, which may be large for some shapes.

## Implementation Notes

The recent lowering work follows these code generation rules:

- Keep the outer traversal as one large [`krnl.iterateIE`](../onnx-mlir/src/Conversion/ONNXToKrnl/Additional/Im2Col.cpp) loop over all output columns.
- Split interior and border handling with [`SCFBuilder::ifThenElse()`](../onnx-mlir/src/Dialect/Mlir/DialectBuilder.cpp:753).
- Use [`SCFBuilder::forLoopIE()`](../onnx-mlir/src/Dialect/Mlir/DialectBuilder.cpp:719) for the inner loop nest inside each branch.
- Keep index arithmetic in [`IndexExpr`](../onnx-mlir/src/Conversion/ONNXToKrnl/Additional/Im2Col.cpp) form.
- When an outer [`IndexExpr`](../onnx-mlir/src/Conversion/ONNXToKrnl/Additional/Im2Col.cpp) is used inside a deeper loop scope, reintroduce it with [`DimIE(...)`](../onnx-mlir/src/Conversion/ONNXToKrnl/Additional/Im2Col.cpp:299).
- Materialize conditions as MLIR values with [`MathBuilder`](../onnx-mlir/src/Conversion/ONNXToKrnl/Additional/Im2Col.cpp:267), not with C++ boolean operators on [`IndexExpr`](../onnx-mlir/src/Conversion/ONNXToKrnl/Additional/Im2Col.cpp).
- Prefer [`loadIE()`](../onnx-mlir/src/Conversion/ONNXToKrnl/Additional/Im2Col.cpp:313) and [`storeIE()`](../onnx-mlir/src/Conversion/ONNXToKrnl/Additional/Im2Col.cpp:316).

When parallel lowering is enabled, the outer loop can be marked for parallel execution using [`tryCreateKrnlParallel()`](../onnx-mlir/src/Accelerators/NNPA/Conversion/ZHighToZLow/ProcessStickData.cpp:77) on the outer loop definition before emitting the [`krnl.iterateIE`](../onnx-mlir/src/Conversion/ONNXToKrnl/Additional/Im2Col.cpp). This keeps the algorithm unchanged while exposing coarse-grain parallelism across output columns.

## Brief Note on the 1x1 Path

For a `1x1` convolution with stride `1`, dilation `1`, and no effective spatial expansion, the lowering can avoid the general [`Im2Col`](../onnx-mlir/src/Conversion/ONNXToKrnl/Additional/Im2Col.cpp) transform.

In that case, each output position uses exactly one input element per channel, so the input can be viewed directly as:

```text
X_1x1: [N, C, H * W]
```

and the weights as:

```text
W_row: [M, C]
```

Then convolution is again a matrix multiply per batch:

```text
Y_mat[n] = W_row * X_1x1[n]
```

followed by:

```text
[N, M, H * W] -> [N, M, H, W]
```

So the `1x1` case is the same general idea, but without materializing the full sliding-window expansion.

## Summary

The simple im2col-based convolution lowering is:

```text
Conv
  -> Im2Col
  -> Reshape weights
  -> MatMul
  -> Reshape result
```

For 2-D convolution, the important packed shapes are:

```text
X_col = [N, C * KH * KW, OH * OW]
W_row = [M, C * KH * KW]
Y_mat = [N, M, OH * OW]
Y     = [N, M, OH, OW]
```

This is the fundamental pattern used to express convolution as matrix multiplication.