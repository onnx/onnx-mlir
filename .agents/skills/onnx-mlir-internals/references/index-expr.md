<!-- Generated from the onnx-mlir knowledge base (concepts/index-expr.md), verified against onnx-mlir@b505d64b. Do not edit here: see .agents/skills/README.md. -->

# IndexExpr

`IndexExpr` is a unified data structure for index computations that works identically during **shape inference** (no code generation, compile-time only) and **lowering** (code generation into MLIR IR). A single function can determine output shapes during analysis and then generate the actual loop bounds and memory access indices during lowering — the same `IndexExpr` arithmetic does both.

## Core Idea

At any point in time, an `IndexExpr` holds one of three representations, in descending order of preference:

1. **Integer literal** — known at compile time; no IR generated
2. **AffineExpr** — kept as an affine expression as long as possible; enables `affine.for`, `affine.load`, etc.
3. **MLIR Value** — falls back to a generic SSA value when affine constraints are violated (e.g. dividing by a runtime symbol rather than a literal)

The system automatically demotes from affine to value when an operation forces it, so callers never manage representation manually.

## The 8 Kinds

Constructed via subclasses; the base `IndexExpr` type is used for all subsequent operations. Verified directly against `class` declarations in `IndexExpr.hpp`:[^1]

| Subclass | Alias | Kind enum | Description |
|----------|-------|-----------|-------------|
| `UndefinedIndexExpr` | — | — | Sentinel for error/unset; `isUndefined()` returns true |
| `LiteralIndexExpr` | `LitIE` | — | Compile-time integer or float constant |
| `NonAffineIndexExpr` | — | `NonAffine` | Runtime `Value` with no affine structure |
| `QuestionmarkIndexExpr` | — | `Questionmark` | Unknown at compile time; used during shape inference where code gen is not possible |
| `PredicateIndexExpr` | `PredIE` | `Predicate` | Result of a compare (1-bit int); used in `select` and `clamp` |
| `AffineIndexExpr` | — | `Affine` | Wraps an `mlir::AffineExpr` |
| `DimIndexExpr` | `DimIE` | `Dim` | Affine *dimension* variable — a value that changes within scope (tensor dim during shape inference; loop index inside a loop) |
| `SymbolIndexExpr` | `SymIE` | `Symbol` | Affine *symbol* — a runtime value treated as constant within the current scope |

If a `NonAffineIndexExpr` is constructed from a value that turns out to be a compile-time constant, it silently becomes a `LiteralIndexExpr` internally.

[^1]: repo:onnx-mlir@b505d64:src/Dialect/Mlir/IndexExpr.hpp (`class IndexExpr`, `class UndefinedIndexExpr`, `class LiteralIndexExpr`, `class NonAffineIndexExpr`, `class QuestionmarkIndexExpr`, `class PredicateIndexExpr`, `class AffineIndexExpr`, `class DimIndexExpr`, `class SymbolIndexExpr` — all declared as forward decls near the top of the file, then defined in order)

## IndexExprScope

Every `IndexExpr` belongs to a scope, which tracks the Dim and Symbol lists needed to build `AffineMap`s. The current scope is a **thread-local** singleton — no explicit passing required.

```cpp
// Shape inference — no rewriter; code gen is disabled
IndexExprScope scope(nullptr, getLoc());

// Lowering — rewriter present; code is emitted
IndexExprScope outerScope(&rewriter, loc);
```

Whether a rewriter is present is what distinguishes shape-inference from lowering mode (`isShapeInferencePass()` checks for null rewriter).[^2]

[^2]: repo:onnx-mlir@b505d64:src/Dialect/Mlir/IndexExpr.hpp (`class IndexExprScope`)

### Nested Scopes

When loop bounds are not fully affine, start a child scope for the loop body. Dims from the outer scope become Symbols in the inner scope:

```cpp
IndexExprScope outerScope(&rewriter, loc);
DimIndexExpr d1(dataBounds.getDim(2));   // dim in outer scope

{
    IndexExprScope innerScope(&rewriter, &outerScope);
    SymbolIndexExpr s1(d1);  // outer dim → inner symbol (constant within loop)
    // ... loop body computations
}  // innerScope destroyed; outer scope resumes
```

Literal and NonAffine expressions can be used across enclosing scopes. Dim/Symbol/Affine must belong to the current scope.

## Operations

All arithmetic returns a new `IndexExpr` at the best possible representation level:

```cpp
// Arithmetic
a + b,  a - b,  a * b
a.floorDiv(b),  a.ceilDiv(b)
a % b

// Comparisons — return Predicate or Literal
a == b,  a != b,  a < b,  a <= b,  a > b,  a >= b

// Boolean on predicates
p1 & p2,  p1 | p2,  !p

// Clamp: result is in [min, max)
a.clamp(min, max)

// Conditional select: cond ? trueVal : falseVal
IndexExpr::select(cond, trueVal, falseVal)
a.selectOrSelf(cond, trueVal)  // false branch returns `a` itself

// Reduction
IndexExpr::min(list),  IndexExpr::max(list)

// Float ops
a.convertToFloat(),  a.convertToIndex()
```

Operations on two `IndexExpr` values automatically select the correct representation:
- Both literals → literal result (no IR)
- Mixed literal + affine → affine result
- Any non-affine operand → Value result (IR emitted)

## IndexExprBuilder

`IndexExprBuilder` is the main entry point for creating `IndexExpr` from MLIR structures. It is a `DialectBuilder` subclass ([concepts/dialect-builders.md](dialect-builders.md)) used as `createIE` in lowering patterns.

Three families of factory methods, all verified directly against `IndexExprBuilder.hpp`:[^3]

### From Attributes
```cpp
// Single element from an integer array attribute
IndexExpr getIntFromArrayAsLiteral(ArrayAttr arr, uint64_t i);

// Whole list
void getIntFromArrayAsLiterals(ArrayAttr arr, IndexExprList &list);
```

### From Runtime Values
```cpp
// Scalar value
IndexExpr getIntAsSymbol(Value v);   // → SymbolIndexExpr or Literal
IndexExpr getIntAsDim(Value v);      // → DimIndexExpr or Literal

// Element of a 1D array value
IndexExpr getIntFromArrayAsSymbol(Value arr, uint64_t i);
IndexExpr getIntFromArrayAsDim(Value arr, uint64_t i);
```

### From Tensor/Memref Shapes
```cpp
IndexExpr getShapeAsDim(Value tensorOrMemref, uint64_t i);
IndexExpr getShapeAsSymbol(Value tensorOrMemref, uint64_t i);
IndexExpr getShapeAsLiteral(Value tensorOrMemref, uint64_t i);

// Whole shape as list
void getShapeAsDims(Value tensorOrMemref, IndexExprList &list);
```

### Subclasses

| Subclass | Used in | Behaviour for runtime values |
|----------|---------|------------------------------|
| `IndexExprBuilderForAnalysis` | Shape inference pass | Returns `QuestionmarkIndexExpr` |
| `IndexExprBuilderForKrnl` | ONNXToKrnl lowering | Emits Krnl ops to compute the value |
| `IndexExprBuilderForStablehlo` | StableHLO lowering | Emits StableHLO ops |

Locations verified: `IndexExprBuilderForAnalysis` in `src/Dialect/ONNX/DialectBuilder.hpp`, `IndexExprBuilderForKrnl` in `src/Dialect/Krnl/DialectBuilder.hpp`, `IndexExprBuilderForStablehlo` in `src/Conversion/ONNXToStablehlo/DialectBuilder.hpp` — each defined alongside that path's other `DialectBuilder` subclasses, not in `IndexExprBuilder.hpp` itself.[^4]

How a builder gets runtime values is factored out into `IndexExprValueProvider` (`src/Dialect/Mlir/IndexExprValueProvider.hpp`), an interface with three pure virtual methods: `getConst`, `getVal` and `getShapeVal`. `IndexExprBuilder` inherits from both `DialectBuilder` and `IndexExprValueProvider`, and each subclass above implements the three methods. `IndexExprBuilder.hpp` also defines `IndexExprBuilderWithProvider`, which forwards them to an external provider instead. `ReifyIndexExprValueProvider` (`src/Dialect/ONNX/ReifyIndexExprValueProvider.hpp`) is one such provider.[^5]

[^3]: repo:onnx-mlir@b505d64:src/Dialect/Mlir/IndexExprBuilder.hpp
[^4]: repo:onnx-mlir@b505d64:src/Dialect/ONNX/DialectBuilder.hpp (`struct IndexExprBuilderForAnalysis`), repo:onnx-mlir@b505d64:src/Dialect/Krnl/DialectBuilder.hpp (`struct IndexExprBuilderForKrnl`), repo:onnx-mlir@b505d64:src/Conversion/ONNXToStablehlo/DialectBuilder.hpp (`IndexExprBuilderForStablehlo`)
[^5]: repo:onnx-mlir@b505d64:src/Dialect/Mlir/IndexExprValueProvider.hpp (`struct IndexExprValueProvider`), repo:onnx-mlir@b505d64:src/Dialect/Mlir/IndexExprBuilder.hpp (`struct IndexExprBuilder : public DialectBuilder, public IndexExprValueProvider`, `struct IndexExprBuilderWithProvider`), repo:onnx-mlir@b505d64:src/Dialect/ONNX/ReifyIndexExprValueProvider.hpp

## DimsExpr

```cpp
using DimsExpr    = llvm::SmallVector<IndexExpr, 4>;
using DimsExprRef = mlir::ArrayRef<IndexExpr>;
```

`DimsExpr` is the standard container for lists of `IndexExpr` — used for output shapes, loop bounds, and memory access indices throughout `ONNXToKrnl` lowerings.

## Tile Helper Methods

`IndexExprBuilder` includes helpers for tiled/blocked loop patterns:[^3]

```cpp
IndexExpr isTileFull(IndexExpr i, IndexExpr block, IndexExpr UB);
IndexExpr partialTileSize(IndexExpr i, IndexExpr block, IndexExpr UB);
IndexExpr tileSize(IndexExpr i, IndexExpr block, IndexExpr UB);
```

## Debugging

```bash
# Enable index-expr debug output
onnx-mlir --debug-only=index-expr ...
```

Or call `debugPrint(msg)` on any `IndexExpr` or `IndexExprScope` in code.

## Worked Example: ONNXSliceOp Bounds

From `src/Dialect/ONNX/ONNXOps/Tensor/Slice.cpp` (shape helper):

```cpp
// 1. Create a scope (shape inference — no rewriter)
IndexExprScope scope(nullptr, op->getLoc());

// 2. Get inputs
SymbolIndexExpr startInput = createIE->getIntFromArrayAsSymbol(starts, i);
if (startInput.isUndefined())
    return op->emitError("...");

DimIndexExpr dimInput = createIE->getShapeAsDim(data, i);

// 3. Compute clamped start: start < 0 ? start + dim : start
IndexExpr startPos = IndexExpr::select(startInput < 0,
                                       startInput + dimInput, startInput);
// Clamp based on step sign
IndexExpr neg = startPos.clamp(0, dimInput - 1);
IndexExpr pos = startPos.clamp(0, dimInput);
IndexExpr startFinal = IndexExpr::select(stepInput < 0, neg, pos);
```

In the loop body (lowering scope), loop induction variables become `DimIndexExpr` and the computed bounds become `SymbolIndexExpr`:

```cpp
IndexExprScope childContext(outerLoopContext);
DimIndexExpr inductionIndex(inductionVal);
IndexExpr start = SymIE(shapeHelper.starts[ii]);
IndexExpr step  = SymIE(shapeHelper.steps[ii]);
loadIndices.emplace_back((step * inductionIndex) + start);
```
