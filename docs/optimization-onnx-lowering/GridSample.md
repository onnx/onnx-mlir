<!--- SPDX-License-Identifier: Apache-2.0 -->
# GridSample Bilinear 2D Optimization

**Status:** Enabled by default (can be disabled by setting `ENABLE_GRIDSAMPLE_BILINEAR_OPT` to 0)

## Overview

This document describes the optimized implementation of GridSample for 2D bilinear interpolation in ONNX-MLIR, inspired by the ONNX Runtime implementation.

## Optimization Strategy

The optimization uses a **sampling plan** approach that separates the computation into two phases:

1. **Plan Computation Phase**: For each output position (h, w), precompute:
   - Corner indices (x0, x1, y0, y1) 
   - Bilinear weights (w11, w12, w21, w22)
   - Validity mask (4 bits indicating which corners are in bounds)

2. **Application Phase**: For each channel, apply the precomputed plan to interpolate values

This approach is beneficial because:
- The sampling plan is computed once per batch and reused across all channels
- For images with many channels (e.g., C=256), this significantly reduces redundant computation
- Memory access patterns are more cache-friendly

## Implementation Details

### Data Structures

Three arrays are allocated per batch with dimensions HxW:

1. **Indices Array** (`HxWx4xindex`): Stores corner coordinates
   - `[h][w][0]` = x0 (left x coordinate)
   - `[h][w][1]` = x1 (right x coordinate)  
   - `[h][w][2]` = y0 (top y coordinate)
   - `[h][w][3]` = y1 (bottom y coordinate)

2. **Weights Array** (`HxWx4xf32`): Stores bilinear interpolation weights
   - `[h][w][0]` = w11 (weight for top-left corner)
   - `[h][w][1]` = w12 (weight for top-right corner)
   - `[h][w][2]` = w21 (weight for bottom-left corner)
   - `[h][w][3]` = w22 (weight for bottom-right corner)

3. **Mask Array** (`HxWxi8`): Stores validity flags (4 bits)
   - Bit 0: top-left corner is valid
   - Bit 1: top-right corner is valid
   - Bit 2: bottom-left corner is valid
   - Bit 3: bottom-right corner is valid

### Padding Modes

#### Zeros Mode
- Uses the mask array to check bounds for each corner
- Out-of-bounds corners contribute 0 to the interpolation
- Requires bit testing during application phase

#### Border Mode  
- Clamps coordinates to valid range during plan computation
- All corners are guaranteed in-bounds, so mask checking is skipped
- More efficient than zeros mode

### Loop Structure

```
for n in 0..N:                    // Batch loop
  // Phase 1: Compute sampling plan
  for h in 0..H:
    for w in 0..W:
      grid_x = grid[n, h, w, 0]
      grid_y = grid[n, h, w, 1]
      
      // Denormalize and compute corners
      x = denormalize(grid_x, W_in)
      y = denormalize(grid_y, H_in)
      
      x0 = floor(x), x1 = x0 + 1
      y0 = floor(y), y1 = y0 + 1
      
      // Compute bilinear weights
      dx = x - x0, dy = y - y0
      w11 = (1-dx) * (1-dy)
      w12 = dx * (1-dy)
      w21 = (1-dx) * dy
      w22 = dx * dy
      
      // Store in plan arrays
      indices[h, w] = [x0, x1, y0, y1]
      weights[h, w] = [w11, w12, w21, w22]
      mask[h, w] = compute_validity_mask(x0, x1, y0, y1)
  
  // Phase 2: Apply plan to all channels
  parallel for c in 0..C:         // Parallelized on channel
    for h in 0..H:
      for w in 0..W:
        // Load precomputed values
        x0, x1, y0, y1 = indices[h, w]
        w11, w12, w21, w22 = weights[h, w]
        mask_bits = mask[h, w]
        
        // Interpolate using plan
        result = 0
        if (mask_bits & 1): result += input[n,c,y0,x0] * w11
        if (mask_bits & 2): result += input[n,c,y0,x1] * w12
        if (mask_bits & 4): result += input[n,c,y1,x0] * w21
        if (mask_bits & 8): result += input[n,c,y1,x1] * w22
        
        output[n, c, h, w] = result
```

### Parallelization

The implementation supports parallelization on the channel dimension during the application phase:
- Each channel can be processed independently
- Enabled with `--convert-onnx-to-krnl=enable-parallel`
- Particularly beneficial for high-channel-count inputs

## Performance Characteristics

### Memory Usage
- Additional memory: `H*W*(4*sizeof(index) + 4*sizeof(f32) + sizeof(i8))` per batch
- Typically: `H*W*(32 + 16 + 1) = 49*H*W` bytes per batch
- Example: For 224x224 output, ~2.5 MB per batch

### Computational Savings
- **Without optimization**: `N*C*H*W` coordinate computations
- **With optimization**: `N*H*W` coordinate computations
- **Speedup factor**: Approximately `C` (number of channels)
- Example: For C=256, up to 256x reduction in coordinate computation

### When Optimization Applies
The optimized path is used when:
- Spatial dimensions = 2 (2D images)
- Interpolation mode = "linear" (bilinear)
- All other modes fall back to the general implementation

## Testing

The optimization is tested through:
1. **MLIR lit tests**: Verify correct IR generation
   - `test/mlir/conversion/onnx_to_krnl/Tensor/GridSample.mlir`
   - `test/mlir/conversion/onnx_to_krnl/Tensor/GridSampleParallel.mlir`

2. **Backend tests**: Verify numerical correctness
   - `test/backend/test_gridsample.py`
   - Compares against PyTorch reference implementation

## References

- ONNX Runtime implementation: [grid_sample.cc](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/tensor/grid_sample.cc)
- ONNX GridSample operator: [GridSample-20](https://onnx.ai/onnx/operators/onnx__GridSample.html)