# Adding a New ONNXFusedOp Fusion Pattern — Walkthrough

Snapshot of the fusion infrastructure as of the `Dialect/ONNX/Transforms`
reorg (`FusionOpChain` → `FusionOpKindHelper`, `FusedOpKindPattern` →
`FusedPatternForOpKind`, NNPA's `OpFusionHelper` → `ZHighFusionOpHelper`).
Supersedes `FusionOpChain.md` in this directory, which describes the
pre-reorg names and layout.

## 1. What a fusion pattern is

Some IR patterns are a short, linear chain of ops that is cheaper to
recognize once and lower as a single custom code-generation step than to
lower op-by-op. The infrastructure captures such a chain inside an
`ONNXFusedOp` container between two independent passes:

- **Creation** (a `Transform` pass, e.g. `FusionOpStickUnstick`): matches an
  anchor op, walks forward/backward through the chain, and if beneficial,
  wraps the chain in an `ONNXFusedOp`, storing the chain's structural
  parameters as named MLIR attributes.
- **Lowering** (a `Conversion` pass, e.g. `ZHighToZLow` or `ONNXToKrnl`):
  reads those attributes back, re-verifies the body wasn't altered by an
  intervening optimization, and emits optimized code in one step — instead
  of re-discovering the pattern from scratch.

Both directions share one non-accelerator base class so that only the
per-kind *content* (what the chain looks like, what parameters it has) is
written once, per kind.

## 2. Data structures

### 2.1 `ONNXFusedOp` — the container (generic ONNX op)

Defined in `src/Dialect/ONNX/AdditionalONNXOps.td` (verifier body in
`src/Dialect/ONNX/ONNXOps/Additional/FusedOp.cpp`).

| Field | Kind | Notes |
|---|---|---|
| `kind` | `StrAttr` | Identifies which fusion pattern this instance is (e.g. `"zhigh.extended_layout_transform"`). Lowering dispatches on this string. |
| `inputs` | `Variadic<AnyType>` operands | Every external tensor value the body needs, in order. Becomes the body's block arguments, same order. |
| `outputs` | `Variadic<AnyType>` results | One per value the body's `onnx.Yield` produces. |
| `body` | `SizedRegion<1>` | `IsolatedFromAbove` — nothing inside may reference a value from outside except through `inputs`/block arguments. Constant-producing ops (`ONNXConstantOp`, `ONNXNoneOp`, `ConstantLike`) are cloned inside instead of threaded as inputs. |

Traits: `Pure`, `IsolatedFromAbove`, has a verifier, and delegates shape
inference to the ops inside the body (the greedy rewriter infers each inner
op, then the `Yield` drives the FusedOp's own result types).

### 2.2 `FusionOpKindHelper` — generic builder/consumer base

`src/Dialect/ONNX/Transforms/FusionOpHelper.hpp` / `.cpp`. Never
instantiated directly — always subclassed, once per fusion kind.

**Fields** (populated by the subclass, consumed by the base):

| Field | Type | Ordering requirement |
|---|---|---|
| `ops` | `SmallVector<Operation*>` | Chain order: `ops[i]`'s output feeds `ops[i+1]`'s input; `ops.back()` is the op whose result(s) become the FusedOp's outputs (or one of several, see `finalResults`). |
| `finalResults` | `SmallVector<Value>` | One entry per FusedOp output/`Yield` operand, in the same order as `fusedOp.getOutputs()`. |

**Non-virtual template methods** (the calling sequences — do not override):

- `fuse(rewriter, loc) -> ONNXFusedOp` — creation side. Sets the insertion
  point to `ops.back()`, builds the isolated body (external inputs
  collected, constants cloned inside), calls `embedAttrs()`, then replaces
  and erases the original chain ops back-to-front.
- `retrieveOpsAndOutputValues(fusedOp)` — lowering side. Walks the body,
  repopulating `ops`/`finalResults` from a live `ONNXFusedOp`. Cannot fail.
- `verifyAndRetrieveAttrs(fusedOp) -> bool` — lowering side. Calls the
  virtual `retrieveAttrs()` then `verify()`; `false` on either failure
  (with `LLVM_DEBUG` output). Requires `ops` already populated.
- `static unFuse(rewriter, fusedOp) -> LogicalResult` — the generic
  fallback: inlines the body back into the enclosing function so the
  constituent ops lower on their own. Static so the catch-all pattern
  (§2.4) can call it without an instance.

**Protected helper for subclasses:**

- `static isInsideFusedOp(op) -> bool` — `true` when `op` is nested inside
  an `ONNXFusedOp` body already. **Mandatory first check** in every
  subclass's `detectIfBeneficial` (see §4, step 3) — fusion moves ops into
  the body rather than erasing them, so without this guard the same pattern
  re-matches its own output and the pass diverges.

**Pure-virtual subclass contract** (four methods, enforced by the compiler):

| Method | Direction | Contract |
|---|---|---|
| `getKind() const -> StringRef` | both | Returns the `kind` string constant for this pattern. |
| `embedAttrs(fusedOp) const` | creation | Writes every parameter field to a named attr. **Only** function that writes attrs. |
| `retrieveAttrs(fusedOp) -> bool` | lowering | Reads every attr back into the fields; `false` if any required attr is missing. **Only** function that reads attrs. |
| `verify() const -> bool` | lowering | Cross-checks `ops` (from `retrieveOpsAndOutputValues`) against the fields (from `retrieveAttrs`) — catches a body silently altered by another pass after fusion. |

**One additional, non-virtual contract member** (documented in
`FusionOpHelper.hpp`, not enforceable as a real virtual because its
signature varies per subclass):

```cpp
bool detectIfBeneficial(const DimAnalysis *dimAnalysis, AnchorOpType startOp);
```

`AnchorOpType` is whatever op the subclass anchors its match on
(`ONNXLayoutTransformOp`, `ONNXUnsqueezeOp`, ...). Virtual dispatch needs a
uniform signature across overrides, so this can't be declared `= 0` in the
base — instead it's enforced at compile time wherever the subclass is
plugged into `FusedPatternForOpKind<AnchorOpType, FusionT>` (§2.3):
omitting it fails to compile there, not in `FusionOpHelper.hpp`.

### 2.3 `FusedPatternForOpKind<AnchorOpType, FusionT>` — creation-side adapter

`src/Dialect/ONNX/Transforms/FusionOpBasePattern.hpp`. Generic
`OpRewritePattern<AnchorOpType>` template — one instantiation per
`(anchor op, fusion kind)` pair:

```cpp
template <typename AnchorOpType, typename FusionT>
class FusedPatternForOpKind : public mlir::OpRewritePattern<AnchorOpType> {
  DimAnalysis *dimAnalysis;
public:
  FusedPatternForOpKind(MLIRContext *context, DimAnalysis *dimAnalysis);
  LogicalResult matchAndRewrite(AnchorOpType anchorOp,
      PatternRewriter &rewriter) const override {
    FusionT fusion;
    if (!fusion.detectIfBeneficial(dimAnalysis, anchorOp))
      return failure();
    fusion.fuse(rewriter, anchorOp.getLoc());
    return success();
  }
};
```

You never subclass this — you instantiate it with your `FusionOpKindHelper`
subclass and register it in a `RewritePatternSet` (§4, step 5).

### 2.4 Lowering-side adapters (generic, non-accelerator)

`src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp` / `.cpp`.

- **`FusedOpKindLowering<FusionT>`** — `OpConversionPattern<ONNXFusedOp>`
  template. Subclass it and implement only:
  ```cpp
  FailureOr<Value> lowerVerified(ONNXFusedOp fusedOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter, FusionT &fusion) const override;
  ```
  The base's `matchAndRewrite` handles: bail if `fusedOp.getKind() !=
  FusionT::kKind` (so unrelated kinds fall through to the next pattern);
  `retrieveOpsAndOutputValues` + `verifyAndRetrieveAttrs` (falls back to
  `FusionOpKindHelper::unFuse` on failure); call your `lowerVerified`; drop
  all intra-body def-use edges; `replaceOp`. You never call `replaceOp` or
  touch body uses yourself.

- **`FusedOpInlineFallback`** — a concrete, benefit-0
  `OpConversionPattern<ONNXFusedOp>`. Register it once per conversion pass
  that may see an `ONNXFusedOp` (already done in the generic
  `ConvertONNXToKrnl.cpp`). Catches any kind with no dedicated
  `FusedOpKindLowering` subclass registered *in that pass*, emits a warning,
  and inlines the body via `FusionOpKindHelper::unFuse` so the constituent
  ops lower individually through their own patterns.

## 3. File map

| Piece | Lives in | Accelerator-specific? |
|---|---|---|
| `ONNXFusedOp` op definition | `src/Dialect/ONNX/AdditionalONNXOps.td`, `.../ONNXOps/Additional/FusedOp.cpp` | No |
| `FusionOpKindHelper` (base) | `src/Dialect/ONNX/Transforms/FusionOpHelper.{hpp,cpp}` | No |
| `FusedPatternForOpKind<A,F>` (creation adapter) | `src/Dialect/ONNX/Transforms/FusionOpBasePattern.hpp` | No |
| `FusedOpKindLowering<F>`, `FusedOpInlineFallback` (lowering adapters) | `src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.{hpp,cpp}` | No |
| Your `FusionOpKindHelper` subclass (kind + params + detect/embed/retrieve/verify) | e.g. `.../ZHigh/ZHighOps/ZHighFusionOpHelper.{hpp,cpp}` for NNPA; a new file under `Dialect/ONNX/Transforms/` if truly generic | Depends on the pattern |
| Creation-pass registration (`patterns.insert<FusedPatternForOpKind<...>>`) | The `Transform` pass that owns the anchor op's dialect, e.g. `src/Accelerators/NNPA/Transform/ZHigh/FusionOpStickUnstick.cpp` | Follows the anchor op |
| Lowering-pass registration (your `FusedOpKindLowering<F>` subclass) | The `Conversion` pass that lowers `ONNXFusedOp` for that kind, e.g. `src/Accelerators/NNPA/Conversion/ZHighToZLow/ZHighToZLow.cpp` | Follows where the kind is lowered |

**Existing worked examples**: `ExtLayoutTransformFusionHelper` and
`ExpandMulStickFusionHelper`, both in `ZHighFusionOpHelper.{hpp,cpp}`,
registered for creation in `FusionOpStickUnstick.cpp` and for lowering in
`ZHighToZLow.cpp`.

## 4. Step-by-step: capturing a new pattern

Say you're adding a new kind, `"zhigh.my_pattern"`, anchored on
`ONNXFooOp`.

1. **Pick the kind string and anchor op.** Kind strings are dialect-prefixed
   (`"zhigh."` for NNPA fusions) so the lowering pass can namespace by
   dispatch. The anchor op is whichever op in the chain is cheapest/most
   specific to `match` on (head or tail of the chain — either works, see
   `ExtLayoutTransformFusionHelper` anchored on the *first* op of its chain
   vs. `ExpandMulStickFusionHelper` also anchored on the first op).

2. **Create the subclass file** (or add to an existing one in the same
   dialect, like `ZHighFusionOpHelper.hpp`/`.cpp`):
   ```cpp
   class MyPatternFusion : public onnx_mlir::FusionOpKindHelper {
   public:
     static constexpr llvm::StringLiteral kKind{"zhigh.my_pattern"};

     // Parameter fields extracted during detection, needed during lowering.
     int64_t myAxis = -1;
     bool myFlag = false;

     bool detectIfBeneficial(
         const DimAnalysis *dimAnalysis, mlir::ONNXFooOp startOp);

     llvm::StringRef getKind() const override { return kKind; }
     void embedAttrs(mlir::ONNXFusedOp fusedOp) const override;
     bool retrieveAttrs(mlir::ONNXFusedOp fusedOp) override;
     bool verify() const override;
   };
   ```

3. **Implement `detectIfBeneficial`.** First line, always:
   ```cpp
   if (isInsideFusedOp(startOp)) return false;
   ```
   Then walk the chain from `startOp` (e.g. via a `singleUserOfType<T>`
   helper — see `ZHighFusionOpHelper.cpp` for reusable static helpers),
   validating each step and populating `ops` (chain order) and
   `finalResults` (one per eventual FusedOp output) and every parameter
   field. End with a **beneficial threshold** check — only return `true`
   when fusing is actually worth it, not just legal.

4. **Implement `embedAttrs` / `retrieveAttrs` / `verify`.** `embedAttrs`
   writes each field as a named MLIR attr; `retrieveAttrs` reads them back
   (fail if any required one is missing — guards against stale/hand-edited
   IR); `verify` re-derives the expected op count/types from the fields and
   checks `ops` still matches, emitting `LLVM_DEBUG` on mismatch.

5. **Register the creation pattern** in the `Transform` pass that owns
   `ONNXFooOp`'s matching (in `FusionOpStickUnstick.cpp`, alongside the
   existing two):
   ```cpp
   using FusedPatternsForMyPattern =
       FusedPatternForOpKind<ONNXFooOp, MyPatternFusion>;
   ...
   patterns.insert<FusedPatternsForMyPattern>(&getContext(), dimAnalysis);
   ```
   (`FusionOpStickUnstick.cpp` also gates its two existing fused patterns
   behind a `disableFusedOpOption`/`disableFusedOp` flag, falling back to a
   hand-written composite-op pattern when disabled — decide whether your
   pattern needs the same escape hatch or can always fuse.)

6. **Implement the lowering.** In the `Conversion` pass responsible for
   lowering this kind (`ZHighToZLow.cpp` for NNPA `"zhigh.*"` kinds):
   ```cpp
   struct MyPatternLowering : public FusedOpKindLowering<MyPatternFusion> {
     using Base = FusedOpKindLowering<MyPatternFusion>;
     using Base::Base;
     FailureOr<Value> lowerVerified(ONNXFusedOp fusedOp, OpAdaptor adaptor,
         ConversionPatternRewriter &rewriter,
         MyPatternFusion &fusion) const override {
       // fusion.myAxis, fusion.myFlag, adaptor.getInputs(), ... — emit code,
       // return the single Value that replaces the FusedOp's output(s).
     }
   };
   ```
   Register it in that pass's pattern list (see `ZHighToZLow.cpp` around
   `patterns.insert<ZHighToZLowFusedExtLayoutTransformLowering>(...)`).
   The generic `FusedOpInlineFallback` (benefit 0, already registered in
   `ConvertONNXToKrnl.cpp`) covers you automatically if this step is skipped
   or not yet written — the FusedOp just inlines and lowers op-by-op instead.

7. **Test.** At minimum:
   - Unit-test `detectIfBeneficial` on positive/negative IR snippets
     (chain present but not beneficial; chain broken by an extra use;
     chain present and beneficial).
   - Round-trip: run the creation pass then the lowering pass on the same
     module; confirm the final Krnl/output IR matches what direct,
     non-fused lowering of the original chain would produce.
   - Confirm `verify()` actually rejects a body you've hand-tampered with
     between creation and lowering (e.g. delete one op in the body) —
     it should fall back to `unFuse` rather than crash or mis-lower.
   - If you added a disable flag, test both settings.

## 5. Common pitfalls

- **Forgetting the `isInsideFusedOp` guard** → infinite rewrite loop, since
  matched ops are moved into the body, not erased.
- **`embedAttrs`/`retrieveAttrs` touching attrs outside those two methods**
  → breaks the "only two functions touch attrs" invariant that makes the
  attr set easy to audit for a given kind.
- **`ops` not in chain order** → `fuse()`'s insertion-point choice
  (`ops.back()`) and `replaceAndErase`'s back-to-front erase both assume
  strict chain order for dominance; violating it can erase a still-used op.
- **Registering your `FusedOpKindLowering<F>` at benefit 0 or below** →
  it can lose to `FusedOpInlineFallback` and your kind always inlines
  instead of using your optimized lowering. Use default/explicit benefit
  above 0.
- **`DimAnalysis` null** — required non-null throughout; there's no
  shape-comparison fallback path.
