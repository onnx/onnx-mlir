---
name: onnx-mlir-internals
description: 'Background on onnx-mlir compiler internals: IndexExpr and shape helpers, the dialect builders (create.krnl, create.onnx, ...), ONNX-to-Krnl and Krnl-to-Affine lowering mechanics, loop collapse and parallel planning, recompose and ONNXFusedOp fusion, and which -march/flags change codegen. Use when you need to understand how these pieces work, not just follow a recipe.'
---

<!-- Generated from the onnx-mlir knowledge base (generated briefing), verified against onnx-mlir@b505d64b. Do not edit here: see .agents/skills/README.md. -->

# onnx-mlir Internals

Reference material that the task skills in this directory link to. Each file explains one part of the compiler; the task skills (add-onnx-op-lowering, add-new-onnx-operation, add-fusion-pattern, ground-lit-tests, ...) say when to read which.

Read only the reference you need:

- [Dialect Builders](references/dialect-builders.md): onnx-mlir provides a set of builder structs that wrap MLIR's `OpBuilder` with type-dispatched, location-implicit APIs for each dialect used during lowering.
- [IndexExpr](references/index-expr.md): `IndexExpr` is a unified data structure for index computations that works identically during **shape inference** (no code generation, compile-time only) and **lowering** (code generation into MLIR IR).
- [Interface: Krnl Dialect → Affine/SCF Dialects](references/krnl-to-affine.md): The `KrnlToAffine` pass lowers Krnl loop constructs to MLIR's Affine and SCF dialects, applying the blocking and permutation annotations specified during ONNXToKrnl lowering.
- [Interface: ONNX Dialect → Krnl Dialect](references/onnx-to-krnl.md) (selected sections): The `ConvertONNXToKrnlPass` lowers each ONNX op into Krnl loop constructs plus Affine/MemRef/Arith/Math ops.
- [onnx-mlir Compiler](references/compiler.md) (selected sections): The compiler module (`src/Compiler/`) implements the compilation driver, option parsing, and pass pipeline construction.
- [ONNX Dialect](references/dialect-onnx.md) (selected sections): The ONNX dialect (`src/Dialect/ONNX/`) is the primary representation of the neural network graph within onnx-mlir.
