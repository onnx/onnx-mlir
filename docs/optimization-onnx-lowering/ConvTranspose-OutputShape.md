# ConvTranspose with output_shape Parameter Support

## Overview

This document describes the implementation of `output_shape` parameter support for the ONNX ConvTranspose operation in ONNX-MLIR.

## ONNX Specification

According to the [ONNX ConvTranspose specification](https://onnx.ai/onnx/operators/onnx__ConvTranspose.html), when the `output_shape` parameter is provided:

1. The `pads` parameter is **ignored**
2. Pads are **auto-calculated** to achieve the exact output dimensions specified
3. The standard ONNX formula still applies:
   ```
   output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + 
                     ((kernel_shape[i] - 1) * dilations[i] + 1) - pads[start_i] - pads[end_i]
   ```

## Algorithm

### Pad Calculation from output_shape

When `output_shape` is provided, we solve for the total padding:

```
total_pad[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + 
               ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i]
```

### Pad Distribution

The total padding is distributed based on the `auto_pad` mode:

- **SAME_UPPER** or **NOTSET** (default): Extra padding on bottom/right
  ```
  pad_top[i] = total_pad[i] / 2
  pad_bottom[i] = total_pad[i] - pad_top[i]
  ```

- **SAME_LOWER**: Extra padding on top/left
  ```
  pad_bottom[i] = total_pad[i] / 2
  pad_top[i] = total_pad[i] - pad_bottom[i]
  ```

### Negative Padding Handling

When `total_pad[i]` is negative (meaning the desired output is larger than the default), the pads are **swapped to opposite sides**:

```cpp
if (padTop[i] < 0) {
  swap(padTop[i], padBottom[i]);
}
```

This matches ONNX Runtime behavior where negative padding adds zeros on opposite edges.

## Implementation

### Files Modified

1. **[`src/Dialect/ONNX/Transforms/Decompose.cpp`](../../src/Dialect/ONNX/Transforms/Decompose.cpp)**
   - Added logic to detect `output_shape` parameter
   - Implemented pad calculation from `output_shape`
   - Added negative padding swap logic

### Code Location

The implementation is in the `DecomposeConvTransposePattern::matchAndRewrite` method, starting around line 333.

## Examples

### Example 1: Negative Pads (output_shape = 12x12)

**Input:**
- Input shape: 1x1x5x5
- Weight shape: 1x1x3x3
- Strides: (2, 2)
- output_shape: (12, 12)

**Calculation:**
```
total_pad = 2 * (5 - 1) + 0 + 3 - 12 = 8 + 3 - 12 = -1
```

**Distribution (NOTSET):**
```
pad_top = -1 / 2 = 0 (integer division)
pad_bottom = -1 - 0 = -1
```

**After swap (pad_top < 0 is false, but pad_bottom < 0 is true):**
```
Final pads: (top=0, left=0, bottom=-1, right=-1)
```

**UpsampleAndPad parameters:**
```
pad_h_left = base_pad_h - pad_top = 2 - 0 = 2
pad_h_right = base_pad_h - pad_bottom + opH = 2 - (-1) + 0 = 3
pad_w_left = base_pad_w - pad_left = 2 - 0 = 2
pad_w_right = base_pad_w - pad_right + opW = 2 - (-1) + 0 = 3
```

**Result:** `pads = [2, 2, 3, 3]` → Output: 12x12 ✓

### Example 2: Positive Pads (output_shape = 10x10)

**Input:**
- Input shape: 1x1x5x5
- Weight shape: 1x1x3x3
- Strides: (2, 2)
- output_shape: (10, 10)

**Calculation:**
```
total_pad = 2 * (5 - 1) + 0 + 3 - 10 = 8 + 3 - 10 = 1
```

**Distribution (NOTSET):**
```
pad_top = 1 / 2 = 0
pad_bottom = 1 - 0 = 1
```

**No swap needed (all pads >= 0)**

**UpsampleAndPad parameters:**
```
pad_h_left = 2 - 0 = 2
pad_h_right = 2 - 1 + 0 = 1
pad_w_left = 2 - 0 = 2
pad_w_right = 2 - 1 + 0 = 1
```

**Result:** `pads = [2, 2, 1, 1]` → Output: 10x10 ✓

## Testing

### Test Scripts

1. **MLIR tests:** [`test/mlir/onnx/onnx_decompose_convtranspose_output_shape.mlir`](../../test/mlir/onnx/onnx_decompose_convtranspose_output_shape.mlir)
   - Tests decomposition with output_shape
   - Tests with different auto_pad modes
   - Tests with positive and negative pads

### Running Tests

```bash
# MLIR tests
cd build
./Debug/bin/onnx-mlir-opt --decompose-onnx ../test/mlir/onnx/onnx_decompose_convtranspose_output_shape.mlir -split-input-file
```

## Key Insights

1. **Negative padding semantics**: In the ONNX formula, negative pads mean we're subtracting a negative value, which adds to the output size. In our conv-based implementation, this translates to adding more padding on the opposite side.

2. **Swap logic**: The key insight is that when computed pads are negative, they need to be swapped to the opposite side to match ONNX Runtime behavior.

3. **Compatibility**: The implementation maintains backward compatibility - when `output_shape` is not provided, the original logic is used.

## References

- [ONNX ConvTranspose Specification](https://onnx.ai/onnx/operators/onnx__ConvTranspose.html)
- [ConvTranspose Decomposition Documentation](ConvTranspose.md)
