# Building onnx-mlir as a Standalone Binary

This document explains how to build `onnx-mlir` as a standalone binary with static linking, similar to how LLVM builds `llc` and `opt`.

## Overview

By default, `onnx-mlir` is built with shared libraries, which requires LLVM/MLIR shared libraries to be present at runtime. The standalone build mode statically links all dependencies (except system libraries) to create a self-contained binary.

## Build Modes

### Default Build (Shared Libraries)

**Characteristics:**
- Binary size: ~5-10 MB
- Requires LLVM/MLIR shared libraries at runtime
- Faster build times
- Smaller disk footprint

**Build Commands:**
```bash
# Build LLVM/MLIR with shared libraries
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DBUILD_SHARED_LIBS=ON \
  ../llvm
ninja

# Build onnx-mlir (default mode)
cmake -G Ninja \
  -DMLIR_DIR=/path/to/llvm-build/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Release \
  ..
ninja onnx-mlir
```

### Standalone Build (Static Linking)

**Characteristics:**
- Binary size: ~200-400 MB
- Self-contained (only depends on system libraries: libc, libm, libpthread, libdl, libstdc++)
- Portable across systems with the same architecture
- No need for LLVM/MLIR installation at runtime
- StableHLO support automatically disabled (Java/JNI support remains available)

**Build Commands:**
```bash
# Build LLVM/MLIR with static libraries
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLVM_BUILD_LLVM_DYLIB=OFF \
  -DLLVM_LINK_LLVM_DYLIB=OFF \
  ../llvm
ninja

# Build onnx-mlir as standalone
cmake -G Ninja \
  -DMLIR_DIR=/path/to/llvm-build/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=Release \
  -DONNX_MLIR_BUILD_STANDALONE=ON \
  ..
ninja onnx-mlir
```

## CMake Options

### ONNX_MLIR_BUILD_STANDALONE

**Type:** Boolean (ON/OFF)  
**Default:** OFF  
**Description:** Build onnx-mlir as a standalone binary with static linking

When enabled:
- Forces `BUILD_SHARED_LIBS=OFF`
- Disables `ONNX_MLIR_ENABLE_STABLEHLO` (adds significant size)
- Statically links all LLVM/MLIR libraries
- Java/JNI support remains available (uses static libraries)

**Example:**
```bash
cmake -DONNX_MLIR_BUILD_STANDALONE=ON ..
```

## Dependencies

### Runtime Dependencies

**Default Build:**
- LLVM/MLIR shared libraries (libMLIR*.so, libLLVM*.so)
- System libraries (libc, libm, libpthread, libdl, libstdc++)
- LLVM tools: `llc`, `opt` (must be in PATH or specified)

**Standalone Build:**
- System libraries only (libc, libm, libpthread, libdl, libstdc++)
- LLVM tools: `llc`, `opt` (must be in PATH or specified)

Note: Both builds require `llc` and `opt` at runtime for code generation.

## Use Cases

### When to Use Default Build
- Development and testing
- When LLVM/MLIR is already installed on target systems
- When disk space is limited
- When you need StableHLO support

### When to Use Standalone Build
- Distribution to systems without LLVM/MLIR
- Containerized deployments
- Ensuring consistent behavior across different systems
- When you want a single portable binary

## Verification

To verify your build mode, check the binary size and dependencies:

```bash
# Check binary size
ls -lh /path/to/onnx-mlir

# Check dynamic library dependencies (Linux)
ldd /path/to/onnx-mlir

# Check dynamic library dependencies (macOS)
otool -L /path/to/onnx-mlir
```

**Expected output for standalone build:**
- Binary size: 200-400 MB
- Only system libraries in dependencies (no libMLIR*, libLLVM*)

**Expected output for default build:**
- Binary size: 5-10 MB
- Many libMLIR* and libLLVM* libraries in dependencies

## Troubleshooting

### Issue: Binary still has LLVM/MLIR shared library dependencies

**Solution:** Ensure LLVM/MLIR was built with `BUILD_SHARED_LIBS=OFF`:
```bash
cd /path/to/llvm-build
cmake -L | grep BUILD_SHARED_LIBS
# Should show: BUILD_SHARED_LIBS:BOOL=OFF
```

### Issue: Build fails with undefined references

**Solution:** Clean the build directory and rebuild:
```bash
rm -rf build
mkdir build && cd build
cmake -DONNX_MLIR_BUILD_STANDALONE=ON ..
ninja onnx-mlir
```

### Issue: Binary is too large

**Solution:** This is expected for standalone builds. To reduce size:
1. Use `CMAKE_BUILD_TYPE=MinSizeRel` instead of `Release`
2. Strip debug symbols: `strip /path/to/onnx-mlir`
3. Use compression: `upx /path/to/onnx-mlir` (if available)

## Performance Considerations

Static linking may provide slight performance benefits due to:
- Better optimization opportunities (link-time optimization)
- Reduced dynamic linking overhead
- Better instruction cache locality

However, the differences are typically negligible for most workloads.