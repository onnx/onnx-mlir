# Krnl Dialect

The Krnl (Kernel) dialect is designed to represent loop nests and memory operations at a level that is suitable for generating optimized code for ONNX operations. It serves as an intermediate representation between the high-level ONNX dialect and low-level affine/standard dialects.

## Operations

### Core Operations

#### `krnl.iterate`

The `iterate` operation represents a loop nest with multiple induction variables. It is the primary way to express iteration in the Krnl dialect.

Syntax:
```mlir
krnl.iterate(%iv0, %iv1, ...) with (%arg0[%lb0, %ub0, %step0], %arg1[%lb1, %ub1, %step1], ...) {
  // loop body
}
```

This operation takes a list of induction variables and a list of iteration specifications, where each specification includes the bounds and step size for that dimension.

### Memory Operations

#### `krnl.load`

Loads a value from a memref at the given indices.

Syntax:
```mlir
%value = krnl.load %memref[%i0, %i1, ...] : memref<type>
```

#### `krnl.store`

Stores a value to a memref at the given indices.

Syntax:
```mlir
krnl.store %value, %memref[%i0, %i1, ...] : memref<type>
```

### Utility Operations

#### `krnl.get_induction_var_value`

Extracts the current value of an induction variable.

Syntax:
```mlir
%value = krnl.get_induction_var_value %iv : index
```

#### `krnl.malloc`

Allocates memory on the heap.

Syntax:
```mlir
%memref = krnl.malloc %size : index -> memref<type>
```

#### `krnl.free`

Deallocates memory previously allocated with `krnl.malloc`.

Syntax:
```mlir
krnl.free %memref : memref<type>
```

## Usage Patterns

The Krnl dialect is typically used to represent ONNX operations that involve loops and memory operations. For example, a matrix multiplication operation might be lowered to Krnl dialect as a loop nest with memory loads and stores.

## Lowering

Krnl operations are designed to be lowered to affine and standard dialect operations. The lowering process typically involves:

1. Converting `krnl.iterate` operations to affine loops
2. Converting `krnl.load`/`krnl.store` operations to affine memory operations
3. Converting `krnl.malloc`/`krnl.free` operations to standard allocation operations

## Removed Operations

The following operations have been removed from the Krnl dialect as they were identified as redundant:

- `krnl.define_loops` - Previously used to define loop handles, functionality merged into `krnl.iterate`
- `krnl.optimize_loops` - Previously used for loop optimization hints, functionality integrated into the optimization pipeline

These operations are no longer available in current versions of the dialect.