<!-- Generated from the onnx-mlir knowledge base (interfaces/krnl-to-affine.md), verified against onnx-mlir@b505d64b. Do not edit here: see .agents/skills/README.md. -->

# Interface: Krnl Dialect → Affine/SCF Dialects

The `KrnlToAffine` pass lowers Krnl loop constructs to MLIR's Affine and SCF dialects, applying the blocking and permutation annotations specified during ONNXToKrnl lowering.

The pass-registration file is `ConvertKrnlToAffine.cpp`. Individual op lowerings live in their own files in the same directory: `KrnlLoad.cpp`, `KrnlStore.cpp`, `KrnlMemset.cpp`, `KrnlMatmul.cpp`, `KrnlCopyToBuffer.cpp`/`KrnlCopyFromBuffer.cpp`, `KrnlPrefetch.cpp`, `KrnlTerminator.cpp`, `KrnlGetLinearOffsetIndex.cpp`.[^0]

[^0]: repo:onnx-mlir@b505d64:src/Conversion/KrnlToAffine/ (directory listing)

## Contract

**Input**: Krnl dialect + Affine + MemRef + Arith + Math (after ONNXToKrnl)
**Output**: Affine + SCF + MemRef + Arith + Math + Vector dialects; no Krnl loop ops remain

## What KrnlToAffine Does

Every Krnl op named below is a real, verified `def Krnl*Op` entry in `src/Dialect/Krnl/Krnl.td`:[^1]

1. `krnl.define_loops` → loop descriptor SSA values are consumed, no op emitted
2. `krnl.block` → records the blocking factor for the specified loop descriptor
3. `krnl.permute` → records the desired iteration order
4. `krnl.iterate` → emits `affine.for` loops respecting the blocking + permutation; for loops that can't use affine (non-affine bounds) falls back to `scf.for`
5. `krnl.load` / `krnl.store` → `affine.load` / `affine.store` (or `memref.load`/`memref.store` for non-affine accesses)
6. `krnl.memcpy` → not converted here; it survives this stage and is lowered to an LLVM call in KrnlToLLVM (`src/Conversion/KrnlToLLVM/KrnlMemcpy.cpp`)
7. `krnl.memset` → loop + store sequence
8. `krnl.collapse` → the fused band becomes one `affine.for` (see below); `krnl.parallel` on a reference turns its `affine.for` into an `affine.parallel`, and now fails with "parallelizes a loop reference that is not an optimized loop of any krnl.iterate" when the reference was never handed to an iterate (e.g. a collapse result that was emitted but not passed on)[^4]

[^1]: repo:onnx-mlir@b505d64:src/Dialect/Krnl/Krnl.td (`def KrnlDefineLoopsOp : Op<Krnl_Dialect, "define_loops", ...>`, `KrnlBlockOp "block"`, `KrnlPermuteOp "permute"`, `KrnlIterateOp "iterate"`, `KrnlLoadOp "load"`, `KrnlStoreOp "store"`, `KrnlMemcpyOp "memcpy"`, `KrnlMemsetOp "memset"`)

## `krnl.collapse` Resolution

`lowerIterateOp()` resolves every `krnl.collapse` among an iterate's optimized loops (`resolveCollapseOps()`) at the point where the nest is still one plain `affine.for` per original dimension, perfectly nested, with the body spliced into the innermost loop — before any block/permute/unroll recipe is applied. For each collapse it:[^4]

1. checks the restrictions that the op's own verifier cannot see (no `iterArgs`; fused loops adjacent and outer-to-inner; no other user of a fused-away reference; step 1; no direct use of a fused dimension's `with`-clause induction variable outside the innermost region), each with its own diagnostic;
2. normalizes a lower bound written as a constant SSA value (`%c0 to N`) into a literal `0`, because `affine::coalesceLoops` tests the bound syntactically (`normalizeZeroLowerBound()`), and rejects anything that does not fold to 0;
3. materializes each dimension's trip count (its upper bound; `affine.min` for a multi-result bound) above the band, then calls MLIR's `affine::coalesceLoops` and erases the index-recovery `affine.apply` ops that `coalesceLoops` leaves dead at the top of the merged loop — left in place, they make the nest imperfect, which later recipes such as `krnl.permute` assert against;
4. records the trip counts in a pass-local side table keyed by the merged loop's induction variable.

`krnl.get_induction_var_value` then looks up that side table: for a collapsed reference without `fusedIndex` it emits the same floordiv/mod chain `coalesceLoops` uses (`iv_i = floordiv(iv_fused, product of inner trip counts) mod tc_i`, no modulo on the outermost), at the query site inside the body; with `fusedIndex` it substitutes the merged induction variable directly.[^4]

Two adjustments elsewhere in the pass support this: the loop-body mover now positions movable blocks relative to the first `affine.for`/`affine.parallel` in a body rather than its first op (a collapsed band leaves `affine.apply` bound computations in front of its loop), and `lowerIterateOp()` adds missing terminators to loops skipped by its "last optimized loop" early exit, which collapse makes reachable when one iterate holds two collapsed bands.[^4]

[^4]: repo:onnx-mlir@5534bc7:src/Conversion/KrnlToAffine/ConvertKrnlToAffine.cpp (`resolveCollapseOps`, `normalizeZeroLowerBound`, `emitCollapsedIndices`, `lowerGetInductionVariableValueOp`, `CollapseTripCounts`, `interpretOperation` handling of `KrnlParallelOp`), repo:onnx-mlir@5534bc7:test/mlir/conversion/krnl_to_affine/krnl_to_affine_collapse.mlir, repo:onnx-mlir@5534bc7:test/mlir/conversion/krnl_to_affine/krnl_to_affine_collapse_invalid.mlir

## AffineBuilderKrnlMem

The generic `AffineBuilder` struct is the base alias defined in `src/Dialect/Mlir/DialectBuilder.hpp` ([concepts/dialect-builders.md](dialect-builders.md)). The lowering patterns in this directory use `AffineBuilderKrnlMem`, a Krnl-specific subclass, confirmed in real usage across `KrnlMemset.cpp`, `KrnlCopyToBuffer.cpp`, `KrnlCopyFromBuffer.cpp`:[^2]

```cpp
MultiDialectBuilder<AffineBuilderKrnlMem, IndexExprBuilderForKrnl> create(rewriter, loc);
create.affine.forLoopIE(lb, ub, step,
    [&](const AffineBuilderKrnlMem &createAffine, ValueRange loopInd){ ... });
create.affine.store(val, memref, indices);
```

Loops are built with `forLoopIE`, the `IndexExpr`-based API described in [concepts/index-expr.md](index-expr.md) (e.g. the read and pad loops in `KrnlCopyToBuffer.cpp`).

[^2]: repo:onnx-mlir@b505d64:src/Conversion/KrnlToAffine/KrnlMemset.cpp, repo:onnx-mlir@b505d64:src/Conversion/KrnlToAffine/KrnlCopyToBuffer.cpp

## Loop Optimizations at O3

`-O3` is not implemented as extra MLIR passes tacked on after `KrnlToAffine` — there is no `affine-scalrep` or `affine-loop-unroll` pass anywhere in onnx-mlir. Instead, `-O3` sets `enableTiling`/`enableSIMD`/`enableFastMath` flags passed into `createLowerToKrnlPass(...)` (`src/Compiler/CompilerPasses.cpp`), which cause the *ONNXToKrnl lowering patterns themselves* to emit `krnl.block`/`krnl.permute` blocking annotations and SIMD-oriented (`vector.*`) code as they build the Krnl loops in the first place. `KrnlToAffine` then just lowers whatever blocking/vector structure those patterns already emitted — it isn't itself where the O3 decision is made.[^3]

[^3]: repo:onnx-mlir@b505d64:src/Compiler/CompilerPasses.cpp (`createLowerToKrnlPass` call site in `addONNXToKrnlPasses`, `enableTiling`/`enableSIMD`/`enableFastMath` args)
