<!--- SPDX-License-Identifier: Apache-2.0 -->

## Minimal ONNX dialect build

This document describes how to build a minimal subset of ONNX-MLIR that exposes only the ONNX dialect and the minimal helpers required to parse ONNX models and produce ONNX dialect MLIR.

### Rationale

Some projects want to import the ONNX dialect IR without bringing the full onnx-mlir stack (lowering passes, Krnl dialect, runtime). The CMake option `ONNX_MLIR_ENABLE_ONLY_ONNX_DIALECT` enables such a build.

### What gets built

When `-DONNX_MLIR_ENABLE_ONLY_ONNX_DIALECT=ON` is set, the following subdirectories are included:

- `include/`
- `src/Interface/` (ShapeInference, ResultTypeInference, HasOnnxSubgraph, ShapeHelper)
- `src/Support/`
- `src/Builder/` (for `ImportFrontendModelFile` and builders)
- `src/Dialect/Mlir/` (utility builders only; no Krnl or Compiler dependencies)
- `src/Dialect/ONNX/`

In addition, a small utility `test-onnx-to-mlir` is built to load an ONNX model and print the ONNX dialect IR.

### Configure and build

```
cmake -S . -B build -G Ninja \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
  -DONNX_MLIR_ENABLE_ONLY_ONNX_DIALECT=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --target test-onnx-to-mlir -j$(nproc)
```

Release builds are recommended to lower memory usage during compilation.

### Using the test helper

```
build/Release/bin/test-onnx-to-mlir /path/to/model.onnx > model.mlir
```

### Notes

- Krnl and Compiler modules are excluded.
- SpecializedKernel interface is excluded.
- The minimal build still requires the upstream LLVM/MLIR toolchain and ONNX protobuf.

