---
name: add-onnx-op-lowering
description: 'Use when writing or changing the ONNXToKrnl lowering of an ONNX op that is already in the dialect: the conversion pattern class, registration in ConvertONNXToKrnl.cpp, IndexExpr shape helpers, SIMD/parallel gating, and lit tests.'
---

<!-- Generated from the onnx-mlir knowledge base (skills/add-onnx-op-lowering.md), verified against onnx-mlir@b505d64b. Do not edit here: see .agents/skills/README.md. -->

# Add an ONNXToKrnl Lowering

Recipe for adding a new ONNX op lowering to the `ONNXToKrnl` conversion pass.

---

## Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `{{OP_NAME}}` | ONNX op name (CamelCase) | `Elu` |
| `{{CATEGORY}}` | Conversion category directory | `Math` |
| `{{OP_FILE}}` | Source file for the lowering | `Elementwise.cpp` |
| `{{OPSET_VER}}` | Opset version to support | `6` |

---

## Step 1: Verify the op is in the ONNX dialect

Check that `{{OP_NAME}}` appears in `ONNXOps.td.inc`. If not, add it via `gen_onnx_mlir.py` first (see [add-onnx-op-to-dialect.md](../../../.agents/skills/add-onnx-op-to-dialect/SKILL.md)).

```bash
grep "ONNX{{OP_NAME}}Op" src/Dialect/ONNX/ONNXOps.td.inc | head -5
```

If absent, the op is not in the dialect. Add it before continuing.

---

## Step 2: Add the lowering class

Create `src/Conversion/ONNXToKrnl/{{CATEGORY}}/{{OP_FILE}}`. Current lowerings are typed `OpConversionPattern<ONNX{{OP_NAME}}Op>` with a generated adaptor, take output dims from the op's shape helper, and allocate with `MemRefBuilder::alignedAlloc`. The smallest real template is `Tensor/EyeLike.cpp`:

```cpp
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;
namespace onnx_mlir {

struct ONNX{{OP_NAME}}OpLowering : public OpConversionPattern<ONNX{{OP_NAME}}Op> {
  ONNX{{OP_NAME}}OpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNX{{OP_NAME}}Op onnxOp,
      ONNX{{OP_NAME}}OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = onnxOp.getOperation();
    Location loc = ONNXLoc<ONNX{{OP_NAME}}Op>(op);
    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MemRefBuilder,
        MathBuilder> create(rewriter, loc);

    // 1. Output type and dims (shape helper computes them as IndexExprs).
    MemRefType memRefType = mlir::cast<MemRefType>(
        typeConverter->convertType(*op->result_type_begin()));
    ONNX{{OP_NAME}}OpShapeHelper shapeHelper(op, adaptor.getOperands(), &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    DimsExpr &outputDims = shapeHelper.getOutputDims();

    // 2. Allocate output.
    Value alloc = create.mem.alignedAlloc(memRefType, outputDims);

    // 3. Loop nest over the output.
    int64_t rank = memRefType.getRank();
    ValueRange loopDef = create.krnl.defineLoops(rank);
    SmallVector<IndexExpr, 4> lbs(rank, LitIE(0));
    create.krnl.iterateIE(loopDef, loopDef, lbs, outputDims,
        [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
          MultiDialectBuilder<KrnlBuilder, MathBuilder> create(createKrnl);
          Value x = create.krnl.load(adaptor.getInput(), loopInd);  // adjust per op
          Value result = /* computation using create.math.* */;
          create.krnl.store(result, alloc, loopInd);
        });

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

void populateLoweringONNX{{OP_NAME}}OpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNX{{OP_NAME}}OpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
```

If the op has no shape helper yet, add one in `src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp` first (Col2Im, for example, uses `using ONNXCol2ImOpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNXCol2ImOp>;`).

An op can also be lowered by expanding it into other ONNX ops with `MultiDialectBuilder<OnnxBuilder>`. The conversion legalizes those ops in the same run, so no Krnl code is needed. `Math/Attention.cpp` works this way: its pattern only calls `lowerONNXAttentionOp()` in `src/Dialect/ONNX/Transforms/AttentionToONNXOps.cpp`, which takes a plain `PatternRewriter` so that the NNPA path can reuse the same expansion on tensors ([interfaces/onnx-to-krnl.md](../../../.agents/skills/onnx-mlir-internals/references/onnx-to-krnl.md#lowerings-that-expand-into-other-onnx-ops)). Put a shared expansion there too if another backend needs it, and add `OMONNXRewrite` to the `OMONNXToKrnl` link libraries, as Attention does. When FileCheck should see the expansion before it is lowered, add the emitted ops to the `emit-intermediate-ir` legal list in `ConvertONNXToKrnl.cpp`.

### Optional: SIMD and parallel

To support `enableSIMD`/`enableParallel`, add the bools to the constructor and the populate function. Gate parallelism per op with `OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(ONNX{{OP_NAME}}Op::getOperationName())`, as NonZero, Col2Im, QLinearMatMul, Det and TensorScatter do. Emit the parallel loop through a `KrnlParallelPlan`, and pass `plan.optimizedLoopDef()` (not `loopDef`) as the iterate's optimized loops:

```cpp
auto plan = KrnlParallelPlan::noCollapse(loopDef, /*parFirstInclusiveDim=*/0,
    /*parLastExclusiveDim=*/2, {.minTripCountForParallel = 16});
if (enableParallel)
  plan.tryCreateParallel(create.krnl, op, "my op outer loop", lbs, outputDims);
create.krnl.iterateIE(loopDef, plan.optimizedLoopDef(), lbs, outputDims, ...);
```

Use the collapse-eligible constructor `KrnlParallelPlan(loopDef, enableCollapse, ...)` only for a plain nest of original loops with zero lower bounds and no `iterArgs`. In that case also thread `enableCollapse` through the populate function. See [interfaces/onnx-to-krnl.md](../../../.agents/skills/onnx-mlir-internals/references/onnx-to-krnl.md#parallel-region-planning-and-loop-collapse).

---

## Step 3: Declare and register the pattern

1. Declare the populate function in `ONNXToKrnlCommon.hpp` next to its category:
   ```cpp
   void populateLoweringONNX{{OP_NAME}}OpPattern(
       mlir::RewritePatternSet &, mlir::TypeConverter &, mlir::MLIRContext *);
   ```
2. Call it from `populateONNXToKrnlConversionPattern()` in `ConvertONNXToKrnl.cpp`:
   ```cpp
   populateLoweringONNX{{OP_NAME}}OpPattern(patterns, typeConverter, ctx);
   ```
3. Add the source file to `src/Conversion/ONNXToKrnl/CMakeLists.txt` (e.g. `Tensor/EyeLike.cpp`).

---

## Step 4: Conversion target (usually nothing to do)

Do **not** add `target.addIllegalOp<ONNX{{OP_NAME}}Op>()`. The pass uses `applyPartialConversion`, and the ONNX dialect is not declared illegal (`addIllegalDialect<ONNXOpsDialect>` is commented out). Ops with a matching pattern are converted, and ops without one are left in place and fail later in the pipeline. If the op was listed in `src/Dialect/ONNX/ONNXUnsupportedOps.hpp`, remove it from there (the Col2Im and Det lowerings did).

---

## Step 5: Build and test

```bash
# Build
cd build && make -j$(nproc) onnx-mlir onnx-mlir-opt

# Test with onnx-mlir-opt
echo '
func.func @test(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "onnx.{{OP_NAME}}"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
' > /tmp/test.mlir
onnx-mlir-opt --shape-inference --convert-onnx-to-krnl /tmp/test.mlir
# With parallel (and collapse, which defaults on with it):
onnx-mlir-opt -O3 --shape-inference --convert-onnx-to-krnl=enable-parallel /tmp/test.mlir

# End-to-end test
python3 utils/RunONNXModel.py --model test_{{OP_NAME}}.onnx --verify ort
```

The `onnx-mlir-opt` invocation above confirms the lowering is *reachable*, not that it is *correct* — reading a lowered loop nest is where correctness bugs slip through. Before writing any FileCheck test for this lowering, ground it numerically: see [ground-lit-tests.md](../../../.agents/skills/ground-lit-tests/SKILL.md). Flag mode (`-r=--O0 -t=--O3`) is almost always available for ONNX-to-Krnl work, since the input is ONNX dialect and compiles end to end. Compose the test's RUN line before grounding it, and keep it minimal — for a lowering with nothing architecture-specific about it that means no `-march`/`--mtriple` at all, plus `--canonicalize` after the Krnl lowering: [ground-lit-tests.md#step-1b-keep-the-run-line-minimal](../../../.agents/skills/ground-lit-tests/SKILL.md#step-1b-keep-the-run-line-minimal).

`RunONNXModel.py` checks one model you built yourself. The ONNX backend tests for the op still need enabling in `test/backend/inference_backend.py`, with the doc regenerated — see [add-new-onnx-operation.md step 6](../../../.agents/skills/add-new-onnx-operation/SKILL.md#6-add-tests). When the lowering extends an existing op (a new attribute, type, or shape case) rather than adding one, recheck which disabled backend tests it now covers: [Extending an existing op](../../../.agents/skills/add-new-onnx-operation/SKILL.md#extending-an-existing-op-recheck-its-backend-tests).

---

## Common Patterns

### Elementwise unary op (input and output same shape)

```cpp
// x → f(x) for each element
create.krnl.iterate(loops, loops, lbs, ubs,
    [&](const KrnlBuilder &b, ValueRange idx) {
        Value x = b.load(input, idx);
        Value result = create.math.exp(x);  // or other math op
        b.store(result, output, idx);
    });
```

### Elementwise binary op with NumPy broadcasting

Use `ONNXBroadcastOpShapeHelper` (`ShapeHelper.hpp`): `computeShapeAndAssertOnFailure()` gives the output dims, and inside the loop `getAccessExprs(operand, i, outputAccess, access)` gives each input's broadcast-adjusted indices. The final output loop in `Math/QLinearMatMul.cpp` is a compact example.

### Op with attributes

```cpp
double alpha = onnxOp.getAlpha().convertToDouble();  // e.g. Celu in Elementwise.cpp
Value alphaVal = create.math.constant(elemType, alpha);
```

---

## Verification

- **Grounded against:** `repo:onnx-mlir@b505d64:src/Conversion/ONNXToKrnl/`
- **Files read:** `ConvertONNXToKrnl.cpp`, `ONNXToKrnlCommon.hpp`, `KrnlParallelPlan.hpp`, `CMakeLists.txt`, `Tensor/EyeLike.cpp`, `Tensor/Col2Im.cpp`, `Tensor/TensorScatter.cpp`, `Math/Det.cpp`, `Math/QLinearMatMul.cpp`, `Math/Attention.cpp`, `Math/Elementwise.cpp`, `src/Dialect/ONNX/Transforms/AttentionToONNXOps.{hpp,cpp}`, `src/Dialect/ONNX/ONNXUnsupportedOps.hpp`
- **Re-checked at b505d64:** the EyeLike template, the populate/registration steps (Det and TensorScatter, added in #3646 and #3657, follow them exactly, including the `CMakeLists.txt` entry and per-op parallel gating), and the Attention expansion, which moved into the shared `lowerONNXAttentionOp()` (#3673)
- **How to test:** Run `make check-onnx-mlir` or use `RunONNXModel.py --verify ort`
