# Building onnx-mlir as a Standalone Binary

This document explains how to build `onnx-mlir` as a standalone binary with static linking, similar to how LLVM builds `llc` and `opt`. Note that just like LLVM, key system-wide libraries like `libc` remain dynamically liked.

## Overview

By default, `onnx-mlir` is built with shared libraries, which requires LLVM/MLIR shared libraries to be present at runtime. The standalone build mode statically links all dependencies (except system libraries) to create a self-contained binary.

## Build Modes

### Default Build (Shared Libraries)

**Characteristics:**
- Binary size: ~5-10 MB
- Requires LLVM/MLIR shared libraries at runtime
- Faster build times
- Smaller disk footprint

Build: see [Linux or OSX](BuildOnLinuxOSX.md).

### Standalone Build

**Characteristics:**
- Binary size: ~200-400 MB
- Self-contained (only depends on system libraries: libc, libm, libpthread, libdl, libstdc++)
- Portable across systems with the same architecture
- No need for LLVM/MLIR installation at runtime
- StableHLO support automatically disabled (Java/JNI support remains available)
- Protobuf and Abseil built from source and statically linked

**Build Commands:**

In the commands below, we build the compilers in a custom `build_standalone` directory. Users may 
decide whether to use the traditional `build` or a custom `build_standalone` directory. Both solutions work.

**Step 1: Build LLVM/MLIR with static libraries**
```bash
# Clean build directory (needed if you already have a build directory).
rm -rf build_standalone
mkdir build_standalone && cd build_standalone

cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_ENABLE_RUNTIMES="openmp" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_ENABLE_LIBEDIT=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLVM_BUILD_LLVM_DYLIB=OFF \
  -DLLVM_LINK_LLVM_DYLIB=OFF \
  -DLLVM_ENABLE_ZSTD=OFF

cmake --build . -- ${MAKEFLAGS}
cmake --build . --target check-mlir
```

**Notes:**
- LLVM defaults to static libraries, so `-DBUILD_SHARED_LIBS=OFF` is technically optional but recommended for clarity
- The critical flags are `-DLLVM_BUILD_LLVM_DYLIB=OFF` and `-DLLVM_LINK_LLVM_DYLIB=OFF` which prevent building the monolithic LLVM shared library
- `-DLLVM_ENABLE_ZSTD=OFF` disables zstd compression support to avoid a Homebrew dependency on macOS. If you need zstd, use `-DLLVM_ENABLE_ZSTD=FORCE_ON -DLLVM_USE_STATIC_ZSTD=ON` instead (requires static zstd library)

**Step 2: Build onnx-mlir as standalone**
```bash
# Clean build directory (needed if you already have a build directory).
rm -rf build_standalone
mkdir build_standalone && cd build_standalone

# Configure with standalone mode (Linux example)
MLIR_DIR=$(pwd)/llvm-project/build_standalone/lib/cmake/mlir
mkdir onnx-mlir/build_standalone && cd onnx-mlir/build_standalone
if [[ -z "$pythonLocation" ]]; then
  cmake -G "Unix Makefiles" \
        -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DMLIR_DIR=${MLIR_DIR} \
        -DONNX_MLIR_BUILD_STANDALONE=ON \
        -DCMAKE_IGNORE_PATH="/usr/local;/opt" \
        ..
else
  cmake -G "Unix Makefiles" \
        -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DPython3_ROOT_DIR=$pythonLocation \
        -DMLIR_DIR=${MLIR_DIR} \
        -DONNX_MLIR_BUILD_STANDALONE=ON \
        -DCMAKE_IGNORE_PATH="/usr/local;/opt" \
        ..
fi
cmake --build . --parallel 12
cmake --build . --target check-onnx-lit
```

**Important Notes:**
- The `-DCMAKE_IGNORE_PATH` flag prevents CMake from finding system-installed packages, forcing it to build everything from source.
- **Linux/z:** Use `-DCMAKE_IGNORE_PATH="/usr/local;/opt"` (shown above)
- **macOS:** Use `-DCMAKE_IGNORE_PATH="/opt/homebrew;/usr/local"` instead
- First build will take longer (~10-20 minutes extra) as it builds protobuf and abseil from source

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
- Faster incremental builds

### When to Use Standalone Build
- Distribution to systems without LLVM/MLIR
- Containerized deployments
- Ensuring consistent behavior across different systems
- When you want a single portable binary
- Production deployments where dependencies are a concern

## Verification

### Quick Verification

**macOS:**
```bash
# Check binary size (should be 200-400 MB for standalone)
ls -lh Debug/bin/onnx-mlir

# Check for non-system dependencies (should be minimal)
otool -L Debug/bin/onnx-mlir | grep -v "^/usr/lib\|^/System"

# Check for third-party libraries (should return nothing or only zstd)
otool -L Debug/bin/onnx-mlir | grep -E "absl|protobuf|homebrew"
```

**Linux/z:**
```bash
# Check binary size (should be 200-400 MB for standalone)
ls -lh Debug/bin/onnx-mlir

# Check for non-system dependencies (should be minimal)
ldd Debug/bin/onnx-mlir | grep -v "^/lib\|^/usr/lib\|linux-vdso\|ld-linux"

# Check for third-party libraries (should return nothing)
ldd Debug/bin/onnx-mlir | grep -E "protobuf|absl|boost"
```

### Comprehensive Verification Script

**macOS:**
```bash
#!/bin/bash
echo "=== ONNX-MLIR Standalone Verification (macOS) ==="
echo ""
echo "Binary Size:"
ls -lh Debug/bin/onnx-mlir
echo ""
echo "Non-System Dependencies:"
otool -L Debug/bin/onnx-mlir | grep -v "^/usr/lib\|^/System\|Debug/bin" || echo "None (Good!)"
echo ""
echo "Third-Party Dependencies:"
otool -L Debug/bin/onnx-mlir | grep -E "absl|protobuf|homebrew" || echo "None (Good!)"
```

**Linux/z:**
```bash
#!/bin/bash
echo "=== ONNX-MLIR Standalone Verification (Linux) ==="
echo ""
echo "Binary Size:"
ls -lh Debug/bin/onnx-mlir
echo ""
echo "Non-System Dependencies:"
ldd Debug/bin/onnx-mlir | grep -v "^/lib\|^/usr/lib\|linux-vdso\|ld-linux" || echo "None (Good!)"
echo ""
echo "Third-Party Dependencies:"
ldd Debug/bin/onnx-mlir | grep -E "protobuf|absl|boost" || echo "None (Good!)"
echo ""
echo "Symbol Count (>100k = static):"
nm Debug/bin/onnx-mlir 2>/dev/null | wc -l
```

### Expected Results

**✅ Standalone Build (Good):**
- Binary size: 200-400 MB
- Non-system dependencies: None or only zstd (from LLVM)
- Third-party libraries: None
- Symbol count: >100,000

**❌ Default Build:**
- Binary size: 5-10 MB
- Non-system dependencies: Many libMLIR*, libLLVM* libraries
- Third-party libraries: May include protobuf, abseil
- Symbol count: <10,000

## Troubleshooting

### Issue: Binary still has protobuf/abseil dependencies from Homebrew or /usr/local

**Symptoms:**
```bash
otool -L Debug/bin/onnx-mlir | grep -E "absl|protobuf|homebrew"
# Shows: /opt/homebrew/opt/abseil/lib/libabsl_*.dylib
```

**Solution:** You forgot to use `CMAKE_IGNORE_PATH`. Clean and rebuild:
```bash
rm -rf build_standalone
mkdir build_standalone && cd build_standalone
cmake -DONNX_MLIR_BUILD_STANDALONE=ON \
      -DCMAKE_IGNORE_PATH="/opt/homebrew;/usr/local" \
      ..
ninja onnx-mlir
```

### Issue: Binary still has LLVM/MLIR shared library dependencies

**Solution:** Ensure LLVM/MLIR was built with static libraries:
```bash
cd /path/to/llvm-build
cmake -L | grep BUILD_SHARED_LIBS
# Should show: BUILD_SHARED_LIBS:BOOL=OFF
```

If not, rebuild LLVM with `-DBUILD_SHARED_LIBS=OFF -DLLVM_BUILD_LLVM_DYLIB=OFF -DLLVM_LINK_LLVM_DYLIB=OFF`

### Issue: Build fails with "could not find absl" or protobuf errors

**Solution:** This is expected when using `CMAKE_IGNORE_PATH`. The build system will automatically fetch and build these from source. Just wait for the build to complete (first build takes longer).

### Issue: Binary is too large

**Solution:** This is expected for standalone builds (200-400 MB). To reduce size:
1. Use `CMAKE_BUILD_TYPE=MinSizeRel` instead of `Release`
2. Strip debug symbols: `strip /path/to/onnx-mlir`
3. Use compression: `upx /path/to/onnx-mlir` (if available)
4. Disable features you don't need (e.g., `-DONNX_MLIR_ENABLE_JAVA=OFF`)

### Issue: CMake finds wrong MLIR_DIR

**Solution:** Ensure you're pointing to the static LLVM build:
```bash
cmake -DMLIR_DIR=/path/to/static-llvm-build/lib/cmake/mlir \
      -DONNX_MLIR_BUILD_STANDALONE=ON \
      -DCMAKE_IGNORE_PATH="/opt/homebrew;/usr/local" \
      ..
```

## Performance Considerations

Static linking may provide slight performance benefits due to:
- Better optimization opportunities (link-time optimization)
- Reduced dynamic linking overhead
- Better instruction cache locality

However, the differences are typically negligible for most workloads.