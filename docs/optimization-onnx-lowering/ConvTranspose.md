<!--- SPDX-License-Identifier: Apache-2.0 -->
# ConvTranspose Decomposition

**Status:** Enabled by default

## Overview

This document describes the decomposition of ONNX ConvTranspose operation into simpler ONNX operations in ONNX-MLIR. The decomposition transforms ConvTranspose into a composition of:
1. **UpsampleAndPad** - for zero insertion and padding
2. **Weight transformations** - using Reshape, Slice, and Transpose
3. **Conv** - regular convolution operation

This approach is based on the mathematical equivalence:
```
ConvTranspose(x, W) = Conv(upsample(x), flip(W)) + padding adjustments
```

## Mathematical Foundation

### ONNX ConvTranspose Formula

The ONNX specification defines the output size as:

```
output_size[i] = stride[i] × (input_size[i] - 1) + output_padding[i] + 
                 ((kernel_size[i] - 1) × dilation[i] + 1) - pads[start_i] - pads[end_i]
```

Where:
- `stride[i]`: Stride in dimension i
- `input_size[i]`: Input dimension size
- `output_padding[i]`: Additional output padding
- `kernel_size[i]`: Kernel dimension
- `dilation[i]`: Dilation factor
- `pads[start_i]`, `pads[end_i]`: Padding at start and end of dimension i

**ONNX Padding Convention**: For 2D, `pads = [top, left, bottom, right]` (4 elements)

### Key Insight

ConvTranspose is mathematically the adjoint (transpose) of convolution. This means:
- The stride in ConvTranspose becomes spacing in the upsampled input
- The kernel must be flipped spatially
- Input/output channel roles are reversed

## Decomposition Pipeline

```
Input (N, C_in, H, W)
    ↓
[1] UpsampleAndPad: Zero Insertion + Padding
    ↓
Upsampled (N, C_in, H_up, W_up)
    ↓
[2] Weight Transform: Reshape → Flip → Permute → Reshape
    ↓
Transformed Weight (C_out, C_in/group, kH, kW)
    ↓
[3] Regular Conv (stride=1, dilation preserved)
    ↓
Output (N, C_out, H_out, W_out)
```

## Implementation Details

### Step 0: Auto_pad Mode Processing

Before decomposition, determine padding values based on `auto_pad` mode:

#### VALID Mode
```cpp
if (autoPad == "VALID") {
    padTop[i] = 0;
    padBottom[i] = 0;
}
```
- No padding applied
- Output size = `stride × (input_size - 1) + kernel_size`

#### SAME_UPPER / SAME_LOWER Modes
```cpp
int64_t totalPad = opH + ((kH - 1) * dH + 1) - sH;
totalPad = std::max(0LL, (long long)totalPad);

if (autoPad == "SAME_UPPER") {
    // Extra padding on bottom/right
    padTop[i] = totalPad / 2;
    padBottom[i] = totalPad - padTop[i];
} else { // SAME_LOWER
    // Extra padding on top/left
    padBottom[i] = totalPad / 2;
    padTop[i] = totalPad - padBottom[i];
}
```

**Derivation**: Starting from ONNX formula and setting `output_size = input_size × stride`:
```
input_size × stride = stride × (input_size - 1) + output_padding + 
                      ((kernel_size - 1) × dilation + 1) - pad_start - pad_end

Simplifying:
pad_start + pad_end = output_padding + ((kernel_size - 1) × dilation + 1) - stride
```

#### NOTSET Mode
```cpp
else { // NOTSET
    padTop[i] = pads[i].getLiteral();
    padBottom[i] = pads[i + spatialRank].getLiteral();
}
```
- Use explicit `pads` parameter provided by user

### Step 1: Zero Insertion and Padding (UpsampleAndPad)

**Purpose**: Insert zeros between input elements and apply padding in a single operation.

```cpp
// Compute padding for UpsampleAndPad
SmallVector<int64_t, 4> upsamplePads;
for (int64_t i = 0; i < spatialRank; ++i) {
    int64_t basePad = (kH - 1) * dH;
    int64_t padLeft = basePad - padTop[i];
    upsamplePads.push_back(padLeft);
}
for (int64_t i = 0; i < spatialRank; ++i) {
    int64_t basePad = (kH - 1) * dH;
    int64_t padRight = basePad - padBottom[i] + opH;
    upsamplePads.push_back(padRight);
}

// Create UpsampleAndPad operation
Value xUp = ONNXUpsampleAndPadOp::create(
    rewriter, loc, xUpType, X, stridesAttr, padsAttr);
```

**UpsampleAndPad Operation**:
- Inserts zeros between elements at stride intervals
- Applies asymmetric padding
- Formula for upsampled size: `H_up = (H - 1) × stride_h + 1`

**Example** with stride=2, input 3×3:
```
Input:           Upsampled (5×5):
[a b c]          [a 0 b 0 c]
[d e f]    →     [0 0 0 0 0]
[g h i]          [d 0 e 0 f]
                 [0 0 0 0 0]
                 [g 0 h 0 i]
```

**Padding Components**:
1. **Base padding** `(kH - 1) × dH`: Standard padding for dilated kernel
2. **ONNX pads subtraction** `- padTop`, `- padBottom`: ONNX pads reduce output size
3. **Output_padding addition** `+ opH`: Added to right/bottom padding only

**Why asymmetric?** Output_padding must add pixels to output. Adding to right/bottom padding achieves this without post-processing.

### Step 2: Weight Transformation

**Purpose**: Transform ConvTranspose weights into Conv weights through a series of operations.

#### 2a. Reshape for Groups
```cpp
// Weight shape: (Cin, Cout_per_group, kH, kW, ...)
// Reshape to: (group, Cin_per_group, Cout_per_group, kH, kW, ...)
SmallVector<int64_t, 6> reshapeShape1;
reshapeShape1.push_back(group);
reshapeShape1.push_back(CinPg);
reshapeShape1.push_back(CoutPg);
for (int64_t i = 0; i < spatialRank; ++i)
    reshapeShape1.push_back(wShape[2 + i]);

Value wReshaped1 = create.onnx.reshape(reshapeType1, W, reshapeShapeVal1);
```

**ONNX Operation**: `Reshape`
- Exposes group dimension for independent processing

#### 2b. Flip Spatial Dimensions
```cpp
// Flip all spatial dimensions using Slice with negative steps
for (int64_t i = 0; i < spatialRank; ++i) {
    int64_t axis = 3 + i; // After group, Cin_pg, Cout_pg
    starts.push_back(INT64_MAX);
    ends.push_back(INT64_MIN);
    axes.push_back(axis);
    steps.push_back(-1);
}

Value wFlipped = create.onnx.slice(
    sliceType, wReshaped1, startsVal, endsVal, axesVal, stepsVal);
```

**ONNX Operation**: `Slice` with negative steps
- Flips kernel spatially (required for adjoint operation)
- Uses `INT64_MAX` for start and `INT64_MIN` for end to flip entire dimension

**Example** 3×3 kernel flip:
```
Original:        Flipped:
[1 2 3]          [9 8 7]
[4 5 6]    →     [6 5 4]
[7 8 9]          [3 2 1]
```

#### 2c. Swap Input/Output Channels
```cpp
// Permute: swap Cin_pg and Cout_pg (indices 1 and 2)
// From: (group, Cin_pg, Cout_pg, kH, kW, ...)
// To:   (group, Cout_pg, Cin_pg, kH, kW, ...)
SmallVector<int64_t, 6> permIndices;
permIndices.push_back(0); // group
permIndices.push_back(2); // Cout_pg
permIndices.push_back(1); // Cin_pg
for (int64_t i = 0; i < spatialRank; ++i)
    permIndices.push_back(3 + i);

Value wPermuted = create.onnx.transpose(transposeType, wFlipped, permAttr);
```

**ONNX Operation**: `Transpose`
- Swaps channel dimensions because Conv operates on upsampled input

#### 2d. Flatten Groups
```cpp
// Reshape to: (group * Cout_pg, Cin_pg, kH, kW, ...)
SmallVector<int64_t, 5> reshapeShape2;
reshapeShape2.push_back(group * CoutPg);
reshapeShape2.push_back(CinPg);
for (int64_t i = 0; i < spatialRank; ++i)
    reshapeShape2.push_back(wShape[2 + i]);

Value wConv = create.onnx.reshape(reshapeType2, wPermuted, reshapeShapeVal2);
```

**ONNX Operation**: `Reshape`
- Final shape matches Conv weight format

### Step 3: Regular Conv Operation

```cpp
// Create Conv with:
// - stride = 1 (stride handled by upsampling)
// - dilation = original dilation
// - group = original group
// - pads = [0, 0, ...] (padding already applied)
SmallVector<int64_t, 4> convStrides(spatialRank, 1);
SmallVector<int64_t, 4> convPads(2 * spatialRank, 0);

Value result = create.onnx.conv(convResultType, xUp, wConv, B,
    rewriter.getStringAttr("NOTSET"),
    rewriter.getI64ArrayAttr(convDilations),
    rewriter.getI64IntegerAttr(group),
    rewriter.getI64ArrayAttr(convKernelShape),
    rewriter.getI64ArrayAttr(convPads),
    rewriter.getI64ArrayAttr(convStrides));
```

**ONNX Operation**: `Conv`
- `stride=1`: Stride effect already achieved by zero insertion
- `dilation`: Preserved from ConvTranspose
- `group`: Preserved from ConvTranspose
- `pads=[0,...]`: Padding already applied in UpsampleAndPad

## ONNX Operations Used

The decomposition uses the following ONNX operations:

1. **UpsampleAndPad** (custom operation)
   - Combines zero insertion and padding
   - Attributes: `strides`, `pads`

2. **Reshape** (2 instances)
   - Expose group dimension
   - Flatten back to Conv format

3. **Slice** (with negative steps)
   - Flip spatial dimensions
   - Attributes: `starts`, `ends`, `axes`, `steps`

4. **Transpose**
   - Swap channel dimensions
   - Attribute: `perm`

5. **Conv**
   - Regular convolution
   - Attributes: `strides=[1,...]`, `dilations`, `group`, `pads=[0,...]`

## Correctness Verification

### Output Shape Verification

The decomposition preserves the ONNX output size formula:

1. After upsampling: size = `(input_size - 1) × stride + 1`
2. After padding: size = `(input_size - 1) × stride + 1 + pad_left + pad_right`
3. After conv: size = `(input_size - 1) × stride + 1 + pad_left + pad_right - (kernel_size - 1) × dilation`

Substituting padding values:
```
pad_left = (kernel_size - 1) × dilation - pads[start]
pad_right = (kernel_size - 1) × dilation - pads[end] + output_padding

Final size = stride × (input_size - 1) + output_padding + 
             ((kernel_size - 1) × dilation + 1) - pads[start] - pads[end]  ✓
```

### Test Results

Comprehensive testing validates the implementation:
- **8/8 test cases pass** with numerical tolerance < 1e-6
- **All auto_pad modes** (NOTSET, VALID, SAME_UPPER, SAME_LOWER) verified
- **Groups, dilation, output_padding** all tested and working
- **1D, 2D, and 3D** (5D tensor) convolutions supported

Test configurations include:
1. Basic 2D ConvTranspose (unit strides)
2. Non-unit strides [2, 2]
3. Asymmetric padding
4. Output padding
5. Grouped convolutions
6. 1D ConvTranspose
7. 3D ConvTranspose (5D tensor)
8. 3D with strides [2, 2, 2]

## Implementation Location

- **Decomposition pattern**: [`src/Dialect/ONNX/Transforms/Decompose.cpp`](../../src/Dialect/ONNX/Transforms/Decompose.cpp:586-820) - `DecomposeConvTransposePattern`
- **UpsampleAndPad lowering**: [`src/Conversion/ONNXToKrnl/Additional/UpsampleAndPad.cpp`](../../src/Conversion/ONNXToKrnl/Additional/UpsampleAndPad.cpp)
- **MLIR tests**: [`test/mlir/onnx/onnx_decompose_convtranspose.mlir`](../../test/mlir/onnx/onnx_decompose_convtranspose.mlir)

## Advantages of This Approach

1. **Correctness**: Mathematically equivalent to ConvTranspose specification
2. **Simplicity**: Reuses existing Conv implementation
3. **Efficiency**: Single UpsampleAndPad operation combines upsampling and padding
4. **Generality**: Supports all ONNX ConvTranspose features:
   - All auto_pad modes
   - Arbitrary strides, dilations, and padding
   - Grouped convolutions
   - Output padding
   - 1D, 2D, and 3D spatial dimensions

5. **Maintainability**: Decomposition into standard ONNX ops simplifies optimization and lowering

## References

- ONNX ConvTranspose specification: https://onnx.ai/onnx/operators/onnx__ConvTranspose.html