<!-- Generated from the onnx-mlir knowledge base (concepts/dialect-builders.md), verified against onnx-mlir@b505d64b. Do not edit here: see .agents/skills/README.md. -->

# Dialect Builders

onnx-mlir provides a set of builder structs that wrap MLIR's `OpBuilder` with type-dispatched, location-implicit APIs for each dialect used during lowering. They are the primary API surface for writing `ONNXToKrnl` lowerings.

## Base: DialectBuilder

All builders inherit from `DialectBuilder`, which holds the `OpBuilder` and `Location`:[^1]

```cpp
struct DialectBuilder {
  DialectBuilder(OpBuilder &b, Location loc);
  DialectBuilder(const DialectBuilder &db);  // copy constructs with same rewriter+loc

  OpBuilder &getBuilder() const;  // also exposed as getBuilderPtr()
  Location getLoc() const;
};
```

[^1]: repo:onnx-mlir@b505d64:src/Dialect/Mlir/DialectBuilder.hpp (`struct DialectBuilder`)

## MathBuilder

Type-dispatched scalar arithmetic. Automatically selects between integer and float ops:[^2]

```cpp
struct MathBuilder : DialectBuilder {
  Value add(Value lhs, Value rhs);   // arith.addi or arith.addf
  Value sub(Value lhs, Value rhs);
  Value mul(Value lhs, Value rhs);
  Value div(Value lhs, Value rhs);
  Value exp(Value val);              // math.exp
  Value select(Value cmp, Value lhs, Value rhs);
  Value sgt(Value lhs, Value rhs);   // signed greater-than
  Value slt(Value lhs, Value rhs);
  Value eq(Value lhs, Value rhs);
  Value andi(Value lhs, Value rhs);  // bitwise and
};
```

`MathBuilder::cast(destType, val)` between integer types of different widths builds `arith.extsi`/`arith.trunci` on a signless integer type of the destination width, vector-shaped when the destination is a vector. An unsigned destination type such as `ui8` is not passed to those ops directly.[^2b]

[^2]: repo:onnx-mlir@b505d64:src/Dialect/Mlir/DialectBuilder.hpp (`struct MathBuilder final : DialectBuilder`)
[^2b]: repo:onnx-mlir@5534bc7:src/Dialect/Mlir/DialectBuilder.cpp (`MathBuilder::cast`)

## ShapeBuilder

A smaller builder for computing shape-related values (e.g. `shape.dim`-style queries feeding shape reification), alongside `MathBuilder` in the same header.[^3]

[^3]: repo:onnx-mlir@b505d64:src/Dialect/Mlir/DialectBuilder.hpp (`struct ShapeBuilder final : DialectBuilder`)

## MemRefBuilder

Memory allocation and access:[^4]

```cpp
struct MemRefBuilder : DialectBuilder {
  memref::AllocOp alloc(MemRefType type);                       // static shape
  memref::AllocOp alloc(MemRefType type, ValueRange dynSymbols);
  memref::AllocOp alloc(MemRefType type, DimsExprRef dims);     // IndexExpr dims
  memref::AllocaOp alloca(MemRefType type);
  memref::DeallocOp dealloc(Value val);
  Value dim(Value val, int64_t index);
  // alignedAlloc(...) variants with the same argument forms, etc.
};
```

[^4]: repo:onnx-mlir@b505d64:src/Dialect/Mlir/DialectBuilder.hpp (`struct MemRefBuilder final : DialectBuilder`)

## KrnlBuilder

High-level Krnl loop construction. This struct lives in `src/Dialect/Krnl/DialectBuilder.hpp` (`KrnlHelper.hpp` holds lower-level helpers like `KrnlIterateOperandPack`, not `KrnlBuilder` itself).[^5]

```cpp
struct KrnlBuilder : DialectBuilder {
  // Loop definition
  ValueRange defineLoops(int64_t n);
  ValueRange block(Value loop, int64_t blockSize);   // returns {blockedLoop, innerLoop}
  void permute(ValueRange loops, ArrayRef<int64_t> map);

  // Loop collapse and parallelism
  Value collapse(ValueRange loops);                  // one loop: returned as-is, no op emitted
  ValueRange getInductionVarValue(ValueRange loops); // one index per original dim
  ValueRange getFusedInductionVarValue(ValueRange loops); // one linear index per collapsed ref
  void parallel(ValueRange loops);                   // prefer one (collapsed) ref per nest

  // Loop iteration (with lambda body)
  void iterate(ValueRange origLoops, ValueRange optLoops,
               ValueRange lbs, ValueRange ubs,
               function_ref<void(const KrnlBuilder &, ValueRange)> body);

  // Memory ops
  Value load(Value memref, ValueRange indices = {});
  void store(Value val, Value memref, ValueRange indices = {});

  // Bulk memory ops
  void memcpy(Value dest, Value src, Value numElems);
  void memcpy(Value dest, Value src, Value numElems,
              Value destOffset, Value srcOffset);    // copy a sub-range
  void memset(Value dest, Value val, bool delayed = false);
};
```

`collapse()`, `getFusedInductionVarValue()` and the advice on `parallel()` are explained in stack/dialect-krnl.md. ONNX-to-Krnl lowerings normally get collapse through `KrnlParallelPlan` rather than by calling `collapse()` themselves ([interfaces/onnx-to-krnl.md](onnx-to-krnl.md#parallel-region-planning-and-loop-collapse)).[^5b]

[^5]: repo:onnx-mlir@b505d64:src/Dialect/Krnl/DialectBuilder.hpp (`struct KrnlBuilder : public DialectBuilder`)
[^5b]: repo:onnx-mlir@5534bc7:src/Dialect/Krnl/DialectBuilder.hpp (`collapse`, `getFusedInductionVarValue`, `parallel`), repo:onnx-mlir@5534bc7:src/Dialect/Krnl/DialectBuilder.cpp

## OnnxBuilder

Creates ONNX dialect ops programmatically (used in decompose/recompose patterns, and in ONNXToKrnl lowerings that expand an op into other ONNX ops, such as `Math/QLinearMatMul.cpp` and the Attention decomposition in `src/Dialect/ONNX/Transforms/AttentionToONNXOps.cpp` that `Math/Attention.cpp` calls). The struct is `OnnxBuilder` (not `ONNXBuilder`).[^6]

```cpp
struct OnnxBuilder : DialectBuilder {
  Value add(Value lhs, Value rhs);
  Value reshape(Type outputType, Value input, Value shape);
  // ... other ONNX ops
};
```

[^6]: repo:onnx-mlir@b505d64:src/Dialect/ONNX/DialectBuilder.hpp (`struct OnnxBuilder : DialectBuilder`, `OnnxBuilder::reshape`), repo:onnx-mlir@b505d64:src/Dialect/ONNX/Transforms/AttentionToONNXOps.cpp, repo:onnx-mlir@b505d64:src/Conversion/ONNXToKrnl/Math/Attention.cpp

## SCFBuilder

Generates SCF dialect ops for non-affine control flow:[^7]

```cpp
struct SCFBuilder : DialectBuilder {
  void ifThenElse(Value cond,
      function_ref<void(const SCFBuilder &)> thenFn,
      function_ref<void(const SCFBuilder &)> elseFn);
  void forLoop(Value lb, Value ub, int64_t step,
      SCFLoopBodyFn bodyFn);                  // scf.for
  void forLoopIE(IndexExpr lb, IndexExpr ub, int64_t step, bool useParallel,
      SCFLoopBodyFn bodyFn);                  // common krnl/affine/scf loop API
  void parallelLoops(ValueRange lbs, ValueRange ubs, ValueRange steps,
      SCFLoopBodyFn bodyFn);                  // scf.parallel
};
```

[^7]: repo:onnx-mlir@b505d64:src/Dialect/Mlir/DialectBuilder.hpp (`struct SCFBuilder final : DialectBuilder`)

## VectorBuilder and LLVMBuilder

Two more builders in the same family: `VectorBuilder` (SIMD vector-dialect ops, `.vec` field) and `LLVMBuilder` (LLVM-dialect ops, `.llvm` field) — both `final : DialectBuilder` alongside the others above.[^8]

For a variadic `LLVMBuilder::call(..., isVarArg=true)`, `handleVarArgCall()` sets the call's `var_callee_type` to the callee's *declared* function type, looked up as an `LLVMFuncOp` in the parent module. MLIR at the current LLVM pin uses that type's parameter count to split fixed from variadic operands and verifies it against the declaration. When the symbol is not yet in the module, the fallback is a variadic type with no fixed parameters.[^8b]

[^8]: repo:onnx-mlir@b505d64:src/Dialect/Mlir/DialectBuilder.hpp (`struct VectorBuilder final : DialectBuilder`, `struct LLVMBuilder final : DialectBuilder`)
[^8b]: repo:onnx-mlir@5534bc7:src/Dialect/Mlir/DialectBuilder.cpp (`LLVMBuilder::handleVarArgCall`)

## MultiDialectBuilder

This is the preferred way to use builders, esp when 2 or more are used.

Composes multiple builders into one object to avoid managing separate instances:[^9]

```cpp
// Declaration
MultiDialectBuilder<KrnlBuilder, MathBuilder, MemRefBuilder> create(rewriter, loc);

// Access
create.krnl.defineLoops(2);   // KrnlBuilder
create.math.add(a, b);         // MathBuilder
create.mem.alloca(type);       // MemRefBuilder
```

Field-name mapping, verified against each builder's partial-specialization of `MultiDialectBuilder`:

| Builder | Field |
|---------|-------|
| `MathBuilder` | `.math` |
| `ShapeBuilder` | `.shape` |
| `MemRefBuilder` | `.mem` |
| `AffineBuilder` (alias for `GenericAffineBuilder<AffineLoadOp, AffineStoreOp>`) | `.affine` |
| `SCFBuilder` | `.scf` |
| `VectorBuilder` | `.vec` |
| `LLVMBuilder` | `.llvm` |
| `KrnlBuilder` | `.krnl` |
| `OnnxBuilder` | `.onnx` |

[^9]: repo:onnx-mlir@b505d64:src/Dialect/Mlir/DialectBuilder.hpp (`struct MultiDialectBuilder` and its per-builder partial specializations, e.g. `MultiDialectBuilder<MathBuilder, Ts...>` declaring `MathBuilder math;`), repo:onnx-mlir@b505d64:src/Dialect/Krnl/DialectBuilder.hpp (`MultiDialectBuilder<KrnlBuilder, Ts...>` declaring `KrnlBuilder krnl;`), repo:onnx-mlir@b505d64:src/Dialect/ONNX/DialectBuilder.hpp (`MultiDialectBuilder<OnnxBuilder, Ts...>` declaring `OnnxBuilder onnx;`)

## Example: Full ONNXToKrnl Lowering Skeleton

```cpp
MultiDialectBuilder<KrnlBuilder, MathBuilder, MemRefBuilder> create(rewriter, loc);

// Allocate output
Value output = create.mem.alloc(outputType, dynDims);

// Build 2D loop
ValueRange loops = create.krnl.defineLoops(2);
create.krnl.iterate(loops, loops, {zero, zero}, {ub0, ub1},
    [&](const KrnlBuilder &b, ValueRange idx) {
        Value lval = b.load(lhs, idx);
        Value rval = b.load(rhs, idx);
        Value res = create.math.add(lval, rval);
        b.store(res, output, idx);
    });
```
