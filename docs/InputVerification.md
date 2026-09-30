# Input Verification

## Overview

ONNX-MLIR provides runtime input verification to ensure that input tensors passed to compiled models match the expected specifications. This feature helps catch errors early and provides clear, actionable error messages when inputs don't match the model's requirements.

## Enabling Input Verification

Input verification is controlled by the `--verifyInputTensors` compiler option and is **enabled by default**:

```bash
# Input verification is enabled by default
onnx-mlir model.onnx -o model

# Explicitly disable input verification if needed
onnx-mlir --verifyInputTensors=false model.onnx -o model
```

**Important**: Input verification runs automatically on **every model invocation** by default. The verification checks are embedded in the compiled model and execute before the model runs. To skip verification for performance-critical scenarios, you must explicitly disable it at compile time with `--verifyInputTensors=false`.

When enabled, the compiled model will perform the following checks at runtime before executing the model:

1. **Number of inputs**: Verifies the correct number of input tensors
2. **Data types**: Ensures each input has the expected data type (e.g., f32, i64)
3. **Tensor ranks**: Checks that each input has the correct number of dimensions
4. **Dimension sizes**: Validates dimension sizes, including:
   - Static dimensions must match exactly
   - Dynamic dimensions (marked as `-1`) can be any non-negative value
   - **Symbolic dimensions** must be consistent across all inputs that share the same symbol

## Symbolic Dimension Consistency

### What are Symbolic Dimensions?

Symbolic dimensions allow you to specify that multiple inputs (or dimensions within inputs) must have the same size, even if that size is not known at compile time. This is particularly useful for batch processing, sequence models, and other scenarios where dimensions are related.

For example, in a model with two inputs:
- Input 0: shape `[batch_size, seq_len]`
- Input 1: shape `[batch_size, seq_len]`

Both inputs must have the same `batch_size` and `seq_len` values at runtime.

### How It Works

1. **Compile Time**: The compiler extracts symbolic dimension names from the ONNX model's `dim_param` attributes and includes them in the model signature.

2. **Runtime**: When the model is executed:
   - The first occurrence of each symbol establishes its value
   - Subsequent occurrences are checked for consistency
   - If a mismatch is detected, a detailed error message is generated

### Example 1: All Symbolic Dimensions

Given a model with symbolic dimensions where both inputs share `batch_size` and `seq_len`:

**Correct usage**: Both inputs have matching dimensions
- Input 0: shape `[4, 128]` (batch_size=4, seq_len=128)
- Input 1: shape `[4, 128]` (batch_size=4, seq_len=128)
- Result: ✓ Success

**Incorrect usage**: Mismatched batch_size
- Input 0: shape `[4, 128]` (batch_size=4, seq_len=128)
- Input 1: shape `[8, 128]` (batch_size=8, seq_len=128)
- Result: ✗ Error!
- Error message: `Inconsistent dimension for symbol 'batch_size' at dimension 0 of input 1: expect 4, but got 8`

### Example 2: Mixed Symbolic and Static Dimensions

Given a model with mixed dimensions (e.g., a transformer model):
- Input 0: shape `[batch_size, seq_len, 768]` (768 is a fixed embedding dimension)
- Input 1: shape `[batch_size, seq_len, 768]`

**Correct usage**: Symbolic dimensions match, static dimension is correct
- Input 0: shape `[2, 512, 768]` (batch_size=2, seq_len=512, embedding=768)
- Input 1: shape `[2, 512, 768]` (batch_size=2, seq_len=512, embedding=768)
- Result: ✓ Success

**Incorrect usage**: Static dimension mismatch
- Input 0: shape `[2, 512, 768]`
- Input 1: shape `[2, 512, 1024]` (wrong embedding size)
- Result: ✗ Error!
- Error message: `Wrong size for dimension 2 of input 1: expect 768, but got 1024`

## Model Signature Format

The model signature is stored as a JSON string and includes symbolic dimension information:

**Example 1: All symbolic dimensions**
```json
[
  {
    "type": "f32",
    "dims": ["batch_size", "seq_len"],
    "name": "input0"
  },
  {
    "type": "f32",
    "dims": ["batch_size", "seq_len"],
    "name": "input1"
  }
]
```

**Example 2: Mixed symbolic and static dimensions**
```json
[
  {
    "type": "f32",
    "dims": ["batch_size", "seq_len", 768],
    "name": "input0"
  },
  {
    "type": "f32",
    "dims": ["batch_size", "seq_len", 768],
    "name": "input1"
  }
]
```

**Example 3: Dynamic dimensions without symbolic names**
```json
[
  {
    "type": "f32",
    "dims": [-1, 10],
    "name": "input0"
  }
]
```

## Error Messages

Input verification provides detailed error messages to help diagnose issues:

### Wrong Number of Inputs
```
Wrong number of input tensors: expect 2, but got 3
```

### Wrong Data Type
```
Wrong data type for the input 0: expect f32
```

### Wrong Rank
```
Wrong rank for the input 1: expect 2, but got 3
```

### Wrong Dimension Size (Static)
```
Wrong size for dimension 1 of input 0: expect 10, but got 20
```

### Wrong Dimension Size (Non-negative Check)
```
Wrong size for dimension 0 ('batch_size') of input 0: expect a non-negative value
```

### Inconsistent Symbolic Dimension
```
Inconsistent dimension for symbol 'batch_size' at dimension 0 of input 1: expect 4, but got 8
```

## Performance Considerations

### Runtime Overhead

Input verification adds a small overhead at model invocation time:
- **Number/type/rank checks**: O(number of inputs) - negligible
- **Dimension size checks**: O(total dimensions across all inputs) - typically < 1ms
- **Symbol consistency checks**: O(1) per symbolic dimension using compile-time hash maps

For most applications, this overhead is negligible compared to model execution time. However, if you're running very small models with extremely high throughput requirements, you may want to disable verification in production after thorough testing.

### Memory Usage

Symbol consistency verification uses stack-allocated arrays (no heap allocation), so memory overhead is minimal and deterministic:
- Memory usage: `8 bytes × number of unique symbols`
- Typical models: < 100 bytes

## Best Practices

1. **Enable During Development**: Always enable input verification during development and testing to catch errors early.

2. **Test with Real Data**: Verify your model with representative input shapes to ensure symbolic dimensions are correctly specified.

3. **Production Deployment**: Consider keeping verification enabled in production for critical applications, or disable it only after thorough testing if performance is critical.

4. **Error Handling**: Implement proper error handling in your application to gracefully handle verification failures and provide meaningful feedback to users.

## Implementation Details

For developers interested in the implementation:

- **Signature Generation**: `src/Conversion/ONNXToKrnl/ConvertONNXToKrnl.cpp`
  - Extracts `dim_param` attributes from ONNX model
  - Generates JSON signature with symbolic dimension names

- **Runtime Verification**: `src/Conversion/KrnlToLLVM/KrnlEntryPoint.cpp`
  - Builds compile-time symbol-to-index mapping for O(1) lookups
  - Performs verification before model execution
  - Generates detailed error messages with context

## See Also

- [Testing Documentation](Testing.md) - How to test models with input verification
- [Error Handling](ErrorHandling.md) - General error handling in ONNX-MLIR
- [ONNX Specification](https://github.com/onnx/onnx/blob/main/docs/IR.md) - ONNX IR specification including dimension parameters
