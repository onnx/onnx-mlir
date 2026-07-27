# Dynamic Dimension Analysis

It is often the case where we want to know two dynamic dimensions are equal or not at compile
time. This helps with decisions on how to lower an ONNX operator (e.g. skipping runtime
broadcasting code, or deciding whether an op can be offloaded to an accelerator that does not
support broadcasting). `DimAnalysis` is the class that answers this question; `--onnx-dim-analysis`
is a companion pass for inspecting and debugging its results.

- [Motivation](#motivation)
- [The DimAnalysis Class](#the-dimanalysis-class)
  - [How dimensions get grouped](#how-dimensions-get-grouped)
  - [Querying the result](#querying-the-result)
- [Scoped Analysis for ShapeHelper](#scoped-analysis-for-shapehelper)
- [Inspecting Results with `--onnx-dim-analysis`](#inspecting-results-with---onnx-dim-analysis)
  - [Getting a `.mlir` File to Analyze](#getting-a-mlir-file-to-analyze)
  - [Reading `onnx.DimGroup`](#reading-onnxdimgroup)
  - [Worked Example](#worked-example)
  - [Debugging the Analysis Itself](#debugging-the-analysis-itself)
- [Other Notable Capabilities](#other-notable-capabilities)

## Motivation

Given an ONNXAddOp as follows:

```mlir
%0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x3x5xf32>, tensor<?x3x5xf32>) -> tensor<?x3x5xf32>
```

If we know at compile time that the first dimensions of `%arg0` and `%arg1` are the same (e.g.,
coming from the same tensor), there is no need to generate runtime code to handle broadcasting
rules.

This also helps generate code for accelerators. If an accelerator does not support broadcasting,
we can check at compile time to decide whether the ONNXAddOp will be offloaded to the accelerator
or not. See [ONNXToZHigh](../src/Accelerators/NNPA/Conversion/ONNXToZHigh/ONNXToZHigh.cpp) for how
NNPA lowering uses `DimAnalysis` this way.

## The DimAnalysis Class

We provide a helper class [DimAnalysis](../src/Dialect/ONNX/ONNXDimAnalysis.hpp) to analyze
dynamic dimensions and to check whether two dynamic dimensions are the same or not. Below is an
example of using DimAnalysis:

```C
#include "src/Dialect/ONNX/ONNXDimAnalysis.hpp"

// Run the dynamic dimension analysis to help check equality of dynamic
// dimensions at compile time.
ModuleOp moduleOp = getOperation();
onnx_mlir::DimAnalysis dimAnalysis(moduleOp);
dimAnalysis.analyze();
```

DimAnalysis is constructed for a ModuleOp so that all operations in the ModuleOp will be analyzed.
Then, actual analysis is done via calling `analyze()`.

### How dimensions get grouped

DimAnalysis works by putting every dynamic dimension in the module into a singleton set and then
iteratively discovering dynamic dimensions that must be equal, merging their sets together, until
a fixed point is reached (no set changes on an iteration). Two dynamic dimensions end up in the
same set if they are proven to be equal, whether that is because they come from the same value,
because an operator's shape-inference rules force them to be equal (e.g. the batch dimension of an
elementwise op, the non-contracting dimension of a MatMul/Gemm, the dimensions untouched by a
Reshape, ...), or because the model declares them equal via the `onnx.dim_params` attribute (see
[Other Notable Capabilities](#other-notable-capabilities)).

Sets are also allowed to be related by a constant integer offset (e.g. `dim_a + 1 == dim_b`),
which is tracked separately and lets the analysis answer offset queries such as the ones used by
padding/slicing patterns.

### Querying the result

After `analyze()` completes, the result can be queried in several ways. The dimensions being
compared are always identified by a `(tensor value, axis)` pair; a negative axis is interpreted as
counting from the innermost dimension, matching Python/NumPy convention.

```C
// Are dimensions tensor1[dimAxis1] and tensor2[dimAxis2] provably equal (static or dynamic)?
bool same = dimAnalysis.sameDim(tensor1, dimAxis1, tensor2, dimAxis2);

// Same as sameDim, but returns false if either dimension is actually static
// (i.e., only useful for comparing two *dynamic* dimensions).
bool sameDyn = dimAnalysis.sameDynDim(tensor1, dimAxis1, tensor2, dimAxis2);

// Do tensor1 and tensor2 have the exact same shape (every axis, static and dynamic)?
bool sameShape = dimAnalysis.sameShape(tensor1, tensor2);

// Do tensor1 and tensor2 agree on every *dynamic* axis? Static-vs-static mismatches
// are ignored, but a static-vs-dynamic axis pair is still considered a mismatch.
bool sameDynShape = dimAnalysis.sameDynShape(tensor1, tensor2);

// Is tensor1 broadcastable to tensor2 along the last dimension (tensor1's last dim is
// 1 or provably equal to tensor2's last dim, and the rest of tensor1's dims match)?
// Note that the check is directional: broadcastLastDim(a, b) is not the same as
// broadcastLastDim(b, a).
bool bcast = dimAnalysis.broadcastLastDim(tensor1, tensor2);

// If tensor1[dimAxis1] + offset == tensor2[dimAxis2] for some compile-time-known
// offset, return that offset; otherwise return std::nullopt.
std::optional<int64_t> offset =
    dimAnalysis.getDimOffset(tensor1, dimAxis1, tensor2, dimAxis2);

// Is dim1 + offset1 == dim2 + offset2 ?
bool sameWithOffset = dimAnalysis.sameDimWithOffset(
    tensor1, dimAxis1, offset1, tensor2, dimAxis2, offset2);
```

`sameDim`/`sameDynDim`/`getDimOffset`/`sameDimWithOffset` take the pair `(tensor, axis)` directly;
`sameShape`/`sameDynShape`/`broadcastLastDim` compare whole tensors instead of a single axis.

## Scoped Analysis for ShapeHelper

Constructing `DimAnalysis` over the whole module can be expensive if it is only needed to answer a
question about a single operation, e.g. inside a `ShapeHelper` implementation. `ScopedDimAnalysis`
is a lighter-weight variant that only traces back a limited number of steps ("upward level") from
a starting operation, instead of analyzing every operation in the module:

```C
// Only look at operations within 3 steps upward from `op`.
onnx_mlir::ScopedDimAnalysis scopedAnalysis(op, /*upwardLevel=*/3, shapeHelper);
```

See [Reshape.cpp](../src/Dialect/ONNX/ONNXOps/Tensor/Reshape.cpp) for an example of this usage
inside `ONNXReshapeOpShapeHelper`.

## Inspecting Results with `--onnx-dim-analysis`

`onnx-mlir-opt` exposes the analysis as a standalone pass, `--onnx-dim-analysis`, that is mainly
intended for testing and debugging the `DimAnalysis` class itself rather than for production
compilation. Running the pass does not change the semantics of the IR: it inserts an
`"onnx.DimGroup"` operation right after the definition of each dynamic dimension it analyzed (and
at the top of the entry block for dimensions coming from block arguments), then leaves the rest of
the IR untouched.

```
onnx-mlir-opt --onnx-dim-analysis model.mlir
```

The pass is normally run together with `--canonicalize` (as in the example below) so that dead
`onnx.Dim` computations used only to build shapes are folded away, making the remaining
`onnx.DimGroup` annotations easier to read.

### Getting a `.mlir` File to Analyze

`--onnx-dim-analysis` is a pass on `onnx-mlir-opt`, which only accepts MLIR text input — it does
not read `.onnx` protobuf files directly. To get a `.mlir` file for a real model, first run it
through the regular `onnx-mlir` driver and stop at the IR stage you care about:

```shell
# Ingest model.onnx and stop right after importing to the ONNX dialect (before any
# onnx-mlir-specific optimization/decomposition passes have run).
onnx-mlir --EmitONNXIR model.onnx
```

This writes `model.onnx.mlir` next to the input, which can then be fed to `onnx-mlir-opt`:

```shell
onnx-mlir-opt --onnx-dim-analysis --canonicalize model.onnx.mlir
```

If you are working on the NNPA accelerator and want to see how dynamic dimensions are related
*after* the ONNX-to-ZHigh conversion (i.e. on the IR that `ONNXToZHigh.cpp` itself analyzes), use
`--EmitZHighIR` with `--maccel=NNPA` instead, which stops right after that conversion and emits
`model.onnx.mlir` in the ZHigh dialect:

```shell
onnx-mlir --maccel=NNPA --EmitZHighIR model.onnx
onnx-mlir-opt --onnx-dim-analysis --canonicalize model.onnx.mlir
```

See the [NNPA dim-analysis LIT tests](../test/mlir/accelerators/nnpa/analysis/dyn-dim-analysis.mlir)
for examples of `--onnx-dim-analysis` running on such IR.

### Reading `onnx.DimGroup`

Each `onnx.DimGroup` op takes the form:

```mlir
"onnx.DimGroup"(%tensor) {axis = 0 : si64, group_id = 1 : si64} : (tensor<?x3x5xf32>) -> ()
```

- `%tensor` is the value whose dimension is being annotated.
- `axis` is the axis of `%tensor` that the annotation refers to (always non-negative here, even if
  the original query used a negative axis).
- `group_id` is the id of the equality set that this `(tensor, axis)` pair was placed into by
  `analyze()`. Two `onnx.DimGroup` ops with the same `group_id` mean the analysis proved those two
  dimensions are equal at runtime, i.e. `dimAnalysis.sameDynDim(...)` would return true for them.
  `group_id` values are just internal set identifiers with no meaning beyond identity/equality —
  do not rely on their numeric value or on the order in which they are assigned, only on whether
  two ops share the same one.

### Worked Example

For:

```mlir
func.func @test(%arg0: tensor<?x3x?xf32>) -> tensor<?x3x?xf32> {
  %0 = "onnx.Sigmoid"(%arg0) : (tensor<?x3x?xf32>) -> tensor<?x3x?xf32>
  onnx.Return %0 : tensor<?x3x?xf32>
}
```

the pass produces (group ids may differ from run to run; what matters is which ops share one):

```mlir
func.func @test(%arg0: tensor<?x3x?xf32>) -> tensor<?x3x?xf32> {
  "onnx.DimGroup"(%arg0) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x3x?xf32>) -> ()
  "onnx.DimGroup"(%arg0) {axis = 2 : si64, group_id = 1 : si64} : (tensor<?x3x?xf32>) -> ()
  %0 = "onnx.Sigmoid"(%arg0) : (tensor<?x3x?xf32>) -> tensor<?x3x?xf32>
  "onnx.DimGroup"(%0) {axis = 0 : si64, group_id = 0 : si64} : (tensor<?x3x?xf32>) -> ()
  "onnx.DimGroup"(%0) {axis = 2 : si64, group_id = 1 : si64} : (tensor<?x3x?xf32>) -> ()
  onnx.Return %0 : tensor<?x3x?xf32>
}
```

showing that `%arg0`'s dim 0 is the same as `%0`'s dim 0 (both `group_id = 0`), and likewise for
dim 2 (both `group_id = 1`), which is exactly what we expect since `Sigmoid` does not change shape.

Because the group ids are unstable, the LIT tests under
[test/mlir/onnx/onnx_dim_analysis.mlir](../test/mlir/onnx/onnx_dim_analysis.mlir) (and the NNPA
counterpart in [test/mlir/accelerators/nnpa/analysis/dyn-dim-analysis.mlir](../test/mlir/accelerators/nnpa/analysis/dyn-dim-analysis.mlir))
typically use `FileCheck`'s `[[GROUP_n_:.*]]` capture-and-reuse pattern rather than hardcoding
group id values, except where the id is known to be deterministic for a given test.

### Debugging the Analysis Itself

Since `--onnx-dim-analysis` only visualizes the *final* result, it can be more useful when
debugging DimAnalysis internals to see the fixed-point iteration and the offset relationships as
they are computed. This is available via LLVM's debug logging, e.g.:

```
onnx-mlir-opt --onnx-dim-analysis -debug-only=dim_analysis model.mlir
```

which prints each iteration's set updates, the final `dump()` of the grouping result, and
`dumpOffsetRelations()`, in addition to the transformed IR.

## Other Notable Capabilities

- **`onnx.dim_params`**: function arguments/results can carry an `onnx.dim_params` attribute
  (e.g. `onnx.dim_params = "0:M,1:N"`, imported from ONNX's symbolic `dim_param` shape info) that
  names dynamic dimensions. DimAnalysis seeds its initial sets using these names, so two arguments
  sharing a `dim_param` name are considered equal from the start, even before any operator-specific
  reasoning is applied.
- **Offset relationships**: beyond plain equality, DimAnalysis tracks relations of the form
  `dim1 + offset1 == dim2 + offset2` (e.g. as produced by padding, slicing, or `Concat`-built
  shapes), exposed via `getDimOffset` and `sameDimWithOffset`.
- **Fused ops**: `DimAnalysis` also understands `ONNXFusedOp`, propagating dimension relationships
  through its captured arguments and results.
