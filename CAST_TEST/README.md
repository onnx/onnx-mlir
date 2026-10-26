# Transpose-Cast-Transpose canonicalization reproducer

This directory demonstrates why the existing `FuseTransposeAndCastPattern`
cannot remain enabled without fixing its result-type construction. Commit
`b6d6a0e243b7b9c2c716c8d01b87e537f53353f9` disables it. The comparison uses
these source states:

- before-equivalent source: `86a2a80f11718f72d122d153ace7bbd103446a44`
- after/source-only parent: `b6d6a0e243b7b9c2c716c8d01b87e537f53353f9`
- LLVM/MLIR: `1053047a4be7d1fece3adaf5e7597f838058c947`
- build: Release, x86-64 CPU target, `-O0`

The bug is target-independent and occurs during shared ONNX canonicalization,
before ONNX-to-Krnl, Krnl-to-Affine, and Krnl-to-LLVM lowering. These artifacts
exercise the CPU pipeline; they prove the shared source bug, not NPU integration.

## Test case

`transpose_cast_transpose.mlir` contains:

```text
tensor<1x2x3x4xf32>
  -> Transpose [0, 2, 3, 1]
  -> Cast f32 to i64
  -> Transpose [0, 2, 3, 1]
  -> tensor<1x4x2x3xi64>
```

The two permutations compose to `[0, 3, 1, 2]`, not identity. This keeps the
fused Transpose visible in the dumped invalid IR.

## Root cause and result

The old TableGen rule rewrites:

```text
Transpose(Cast(Transpose(v), to=i64))
  -> Transpose(Cast(v, to=i64, resultType=returnType(v)))
```

`returnType(v)` retains the original `f32` element type. The actual before
IR in `before.invalid-canonicalized.onnx.mlir` therefore contains:

```mlir
%0 = "onnx.Cast"(%arg0) <{saturate = 1 : si64, to = i64}>
    : (tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
```

This is invalid: an ONNX Cast with `to = i64` must produce an `i64`
element type. Verification fails with:

```text
'onnx.Cast' op element type does not match the 'to' attribute
```

Consequently, the normal verified before pipeline fails before emitting LLVM
IR. This is the expected comparison result, not a missing artifact.

After the source change, canonicalization preserves the valid
Transpose-Cast-Transpose chain. Full lowering succeeds, and both LLVM forms
contain the required conversion:

```text
llvm.fptosi ... : f32 to i64  # after.onnx.mlir
fptosi float ... to i64       # after.ll
```

## Artifacts

| File | Meaning |
| --- | --- |
| `transpose_cast_transpose.mlir` | Valid ONNX dialect input |
| `before.invalid-canonicalized.onnx.mlir` | Actual malformed before IR, emitted with verification disabled |
| `before.canonicalize.stderr.txt` | Before canonicalization verifier failure, exit 1 |
| `before.lowering.stderr.txt` | Before full-lowering failure, exit 14 |
| `after.canonicalized.onnx.mlir` | Valid chain preserved after the fix |
| `canonicalization.diff` | Focused before/after canonicalization comparison |
| `after.onnx.mlir` | LLVM dialect MLIR emitted after the fix |
| `after.ll` | Textual LLVM IR emitted after the fix |

## Commands

The build was configured with:

```bash
LLVM_BUILD=/home/eunsangson/Project_2/llvm-project/build

cmake -S . -B build-cast-test -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR="$LLVM_BUILD/lib/cmake/mlir" \
  -DLLVM_DIR="$LLVM_BUILD/lib/cmake/llvm" \
  -DCMAKE_C_COMPILER=/usr/bin/cc \
  -DCMAKE_CXX_COMPILER=/usr/bin/c++

cmake --build build-cast-test \
  --target onnx-mlir onnx-mlir-opt -j 16
```

The after artifacts were produced with:

```bash
build-cast-test/Release/bin/onnx-mlir-opt \
  --canonicalize --verify-each \
  CAST_TEST/transpose_cast_transpose.mlir \
  -o CAST_TEST/after.canonicalized.onnx.mlir

build-cast-test/Release/bin/onnx-mlir \
  --EmitLLVMIR -O0 --omit-compile-info \
  CAST_TEST/transpose_cast_transpose.mlir \
  -o CAST_TEST/after

"$LLVM_BUILD/bin/mlir-translate" \
  --mlir-to-llvmir CAST_TEST/after.onnx.mlir \
  -o CAST_TEST/after.ll
```

For the before run, HEAD remained `b6d6a0e`, but only the three deleted source
lines were restored. The two relevant source files were then byte-for-byte
equivalent to `86a2a80`, as verified against `HEAD^` before rebuilding:

```bash
git diff --exit-code HEAD^ -- \
  src/Dialect/ONNX/ONNXOps/Canonicalize.cpp \
  src/Dialect/ONNX/ONNXOps/Canonicalize.td

cmake --build build-cast-test \
  --target onnx-mlir onnx-mlir-opt -j 16

build-cast-test/Release/bin/onnx-mlir-opt \
  --canonicalize --verify-each=false \
  CAST_TEST/transpose_cast_transpose.mlir \
  -o CAST_TEST/before.invalid-canonicalized.onnx.mlir
```

The same command with `--verify-each` fails with exit 1. The full
`onnx-mlir --EmitLLVMIR` run fails with exit 14 before creating an output
file. The temporary source restoration was then removed, and the tracked
source files were verified clean against `HEAD`.

## Why both source files change

- `Canonicalize.cpp` stops registering the unsafe rewrite at runtime.
- `Canonicalize.td` removes the unsafe generated pattern definition.

Removing only the TableGen definition would leave the C++ registration
referring to a nonexistent pattern. Removing only the C++ registration would
disable the behavior but leave dead, unsafe generated code behind.
