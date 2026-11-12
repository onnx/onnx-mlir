---
name: add-onnx-op-to-dialect
description: 'Use when an ONNX operator (or a new opset version of one) must be added to onnx-mlir''s ONNX dialect: version_dict in utils/gen_onnx_mlir.py, regenerating ONNXOps.td.inc, shape inference, and a custom verifier. Only the dialect step; lowering is add-onnx-op-lowering.'
---

<!-- Generated from the onnx-mlir knowledge base (skills/add-onnx-op-to-dialect.md), verified against onnx-mlir@b505d64b. Do not edit here: see .agents/skills/README.md. -->

# Add an ONNX Op to the Dialect

Recipe for adding a new ONNX operator to the onnx-mlir ONNX dialect using `gen_onnx_mlir.py`. This is the prerequisite step before writing a lowering.

---

## Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `{{OP_NAME}}` | ONNX op name (exact, as in the ONNX spec) | `Elu` |
| `{{OPSET_VER}}` | Opset version to support | `6` |

---

## Step 1: Verify the op exists in the ONNX package

```python
import onnx.defs
schema = onnx.defs.get_schema("{{OP_NAME}}", version={{OPSET_VER}})
print(schema.doc)
print([i.name for i in schema.inputs])
print([o.name for o in schema.outputs])
print(schema.attributes)
```

If this raises `RuntimeError`, the op or version does not exist. Check `onnx.defs.onnx_opset_version()` for the max supported version and `onnx.defs.get_all_schemas_with_history()` for the history.

Use the onnx package version `gen_onnx_mlir.py` expects: it compares `onnx.__version__` with its `current_onnx_version` (`"1.23.0"`) and quits without generating anything on a mismatch. `requirements.txt` pins the same version.

---

## Step 2: Add to version_dict in gen_onnx_mlir.py

Open `utils/gen_onnx_mlir.py` and locate `version_dict`:

```python
version_dict = {
    # ... existing ops ...
    "{{OP_NAME}}": [{{OPSET_VER}}],   # ← add this line
}
```

If the op needs special handling, also check and update:
- `OpsWithResultTypeInference` — if output element type can be inferred
- `OpsWithCanonicalizer` — if a canonicalization pattern exists
- `OpsWithVerifier` — if extra semantic constraints need checking
- `OpsWithHelpers` — if helper functions are needed
- `custom_builder_broadcast_ops_list` — if output type is broadcast of inputs

If the op's schema uses an element type `parse_type_str()` does not yet map (it maps ONNX type strings such as `float8e8m0` → `F8E8M0FNU`, `float4e2m1` → `F4E2M1FN`), add the mapping there too, or generation fails on that op.

---

## Step 3: Regenerate ONNXOps.td.inc

```bash
cd /path/to/onnx-mlir
python3 utils/gen_onnx_mlir.py
```

This overwrites `src/Dialect/ONNX/ONNXOps.td.inc`. The new op will appear as:

```tablegen
def ONNX{{OP_NAME}}Op : ONNX_Op<"{{OP_NAME}}", [...]> {
  let summary = "...";
  let arguments = (ins ...);
  let results = (outs ...);
}
```

---

## Step 4: Implement shape inference

Every generated op declares `ShapeInferenceOpInterface` and `ShapeHelperOpInterface`, and its generated `getShapeHelper()` instantiates `ONNX{{OP_NAME}}OpShapeHelper` — so the op will not compile until that type exists. Put the implementation in a per-category file, `src/Dialect/ONNX/ONNXOps/<Category>/{{OP_NAME}}.cpp` (not the top-level `ONNXOps.cpp`), and add that file to the `OMONNXOps` source list in `src/Dialect/ONNX/CMakeLists.txt`.

For an op whose output has the shape and type of its first input, alias an existing helper in `ONNXOps/ShapeHelper.hpp` and reuse the generic inference:

```cpp
// ShapeHelper.hpp, in the unary-op alias list
using ONNX{{OP_NAME}}OpShapeHelper = ONNXUnaryOpShapeHelper;

// ONNXOps/<Category>/{{OP_NAME}}.cpp
LogicalResult ONNX{{OP_NAME}}Op::inferShapes(
    std::function<void(Region &)> doShapeInference) {
  return inferShapeForUnaryOps(this->getOperation());
}
```

Otherwise declare `using ONNX{{OP_NAME}}OpShapeHelper = ONNXNonSpecificOpShapeHelper<mlir::ONNX{{OP_NAME}}Op>;` in `ShapeHelper.hpp`, define its `computeShape()` in the op's `.cpp`, and instantiate it there with `template struct ONNXNonSpecificOpShapeHelper<ONNX{{OP_NAME}}Op>;`.

If the op is listed in `src/Dialect/ONNX/ONNXUnsupportedOps.hpp` (ops whose `inferShapes()` is stubbed to a no-op), remove that line when adding real shape inference, or the stub and your definition collide.

---

## Step 5: Add a custom verifier (if needed)

If the op has constraints beyond type checking, implement `verify()`:

```cpp
LogicalResult ONNX{{OP_NAME}}Op::verify() {
  ONNX{{OP_NAME}}OpAdaptor operandAdaptor(*this);
  // Check only when shaped and ranked
  if (!hasShapeAndRank(operandAdaptor.getInput()))
    return success();
  auto type = mlir::cast<ShapedType>(operandAdaptor.getInput().getType());
  if (type.getRank() < 1)
    return onnx_mlir::Diagnostic::emitOperandHasUnexpectedRankError(
        *this->getOperation(), operandAdaptor.getInput(), type.getRank(),
        ">= 1");
  return success();
}
```

And add the op name to `OpsWithVerifier` in `gen_onnx_mlir.py` (which emits `let hasVerifier = 1;`), then regenerate. `onnx_mlir::Diagnostic` (`src/Support/Diagnostic.hpp`: `emitOperandHasUnexpectedRankError`, `emitAttributeOutOfRangeError`, `emitDimensionHasUnexpectedValueError`) gives the standard message formats.

---

## Step 6: Rebuild

```bash
cd build && make -j$(nproc) onnx-mlir
```

---

## Step 7: Test dialect import

```python
import onnx
from onnx import helper, TensorProto

# Create a minimal test model
X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])
node = helper.make_node("{{OP_NAME}}", ["X"], ["Y"])
graph = helper.make_graph([node], "test", [X], [Y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", {{OPSET_VER}})])
onnx.save(model, "/tmp/test_{{OP_NAME}}.onnx")
```

```bash
onnx-mlir --EmitONNXIR /tmp/test_{{OP_NAME}}.onnx
```

The output should contain `onnx.{{OP_NAME}}` ops, not `onnx.Custom` (what the importer emits, with a "Could not find op importer" warning, for an op type with no import handler and no ONNX function body). Also add a shape-inference lit test to `test/mlir/onnx/onnx_shape_inference.mlir` (input result type `tensor<*x...>`, `CHECK` the inferred type) and run it with `onnx-mlir-opt --shape-inference`.

Lit tests only cover import and shape inference. Enabling the op's ONNX backend tests, the `==OP==`/`==MIN==`/`==LIM==` markers, and regenerating `docs/SupportedONNXOps-*.md` are in [add-new-onnx-operation.md step 6](../../../.agents/skills/add-new-onnx-operation/SKILL.md#6-add-tests). That step also applies when this recipe is used to add a new opset version to an existing op: see [Extending an existing op](../../../.agents/skills/add-new-onnx-operation/SKILL.md#extending-an-existing-op-recheck-its-backend-tests).

---

## Worked example: `TensorScatter`

The commit adding `onnx.TensorScatter` (opset 24) touched exactly the files this recipe names:

- `utils/gen_onnx_mlir.py`: `"TensorScatter": [24]` in `version_dict`, `"TensorScatter"` in `OpsWithVerifier`, and `parse_type_str()` entries for `float8e8m0`/`float4e2m1`, which its type constraint uses
- regenerated `src/Dialect/ONNX/ONNXOps.td.inc` (`def ONNXTensorScatterOp`, `let hasVerifier = 1;`) and `src/Builder/OpBuildTable.inc` (`op_dialect_version_map_` and `import_handler_map_` entries)
- `src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp`: `using ONNXTensorScatterOpShapeHelper = ONNXUnaryOpShapeHelper;` — `present_cache` always has `past_cache`'s shape
- new `src/Dialect/ONNX/ONNXOps/Math/TensorScatter.cpp`: `verify()` (`mode`, element types, ranks, `axis` range, per-dim shape checks, `write_indices` rank/size) and `inferShapes()` via `inferShapeForUnaryOps`
- `src/Dialect/ONNX/CMakeLists.txt`: `ONNXOps/Math/TensorScatter.cpp`
- `test/mlir/onnx/onnx_shape_inference.mlir`: `test_tensorscatter_static`, `test_tensorscatter_dynamic`

---

## Verification

- **Grounded against:** `repo:onnx-mlir@b505d64:utils/gen_onnx_mlir.py`, `repo:onnx-mlir@b505d64:src/Dialect/ONNX/ONNXOps/Math/TensorScatter.cpp`, `repo:onnx-mlir@b505d64:src/Dialect/ONNX/ONNXOps/Tensor/Col2Im.cpp`, `repo:onnx-mlir@b505d64:src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp`, `repo:onnx-mlir@b505d64:src/Dialect/ONNX/ONNXUnsupportedOps.hpp`, `repo:onnx-mlir@b505d64:src/Dialect/ONNX/CMakeLists.txt`, `repo:onnx-mlir@b505d64:src/Builder/FrontendDialectTransformer.cpp` (`ImportNode`)
- **Files read:** `utils/gen_onnx_mlir.py` (`current_onnx_version` check, `version_dict`, `OpsWithVerifier`, `OpsWithResultTypeInference`, `parse_type_str`), the `TensorScatter` and `Col2Im` op additions (per-category `.cpp`, `ShapeHelper.hpp` aliases, `CMakeLists.txt`, `ONNXUnsupportedOps.hpp`), `test/mlir/onnx/onnx_shape_inference.mlir`, `docs/ImportONNXDefs.md`
- **How to test:** `onnx-mlir --EmitONNXIR test.onnx` should show the new op; shape inference should produce concrete types for static-shape inputs
