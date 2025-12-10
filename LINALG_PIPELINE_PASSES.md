# Linalg Pipeline Pass Application Summary

## Overview

The Linalg pipeline transforms ONNX models into executable LLVM IR through a series of dialect conversions and optimizations. This document provides a detailed breakdown of all passes applied in the pipeline, including their locations, levels, and transformations.

## Pipeline Flow

```
ONNX IR → [Preprocessing] → Linalg → [Bufferization] → Loops → Affine/SCF → CF → LLVM IR
```

---

## Pass Application Summary Table

| Step | Pass Name | Level | Location | Role | Input → Output |
|------|-----------|-------|----------|------|-----------------|
| **Phase 1: ONNX → Linalg (`addONNXToLinalgPasses`)** |
| 1.1 | `createConvertONNXToLinalg()` | Func | `src/Conversion/ONNXToLinalg/ConvertONNXToLinalg.cpp`<br>`src/Compiler/CompilerPasses.cpp:243` | ONNX → Linalg conversion | `onnx.MatMul` → `linalg.matmul` |
| 1.2 | `ConvertONNXEntryPointToKrnlPass` | Module | `src/Compiler/CompilerPasses.cpp:251-278`<br>(uses `src/Conversion/ONNXToKrnl/ConvertONNXToKrnl.cpp`) | Entry point conversion | `onnx.EntryPoint` → `krnl.EntryPoint` |
| 1.3 | `createOneShotBufferizePass()` | Module | MLIR standard library<br>`src/Compiler/CompilerPasses.cpp:284` | Tensor → Memref | `tensor<...>` → `memref<...>` |
| 1.4 | `createCanonicalizerPass()` | Module | MLIR standard library<br>`src/Compiler/CompilerPasses.cpp:287` | Canonicalization | Optimization |
| **Phase 2: Linalg → Affine/SCF → CF (`addLinalgToAffinePasses`)** |
| 2.1 | `createConvertLinalgToLoopsPass()` | Func | MLIR standard library<br>`src/Compiler/CompilerPasses.cpp:297` | Linalg → Loops | `linalg.matmul` → `affine.for` |
| 2.2 | `createBufferLoopHoistingPass()` | Func | MLIR standard library<br>`src/Compiler/CompilerPasses.cpp:304` | Memory allocation hoisting | Move allocations outside loops |
| 2.3 | `buildBufferDeallocationPipeline()` | Func | MLIR standard library<br>`src/Compiler/CompilerPasses.cpp:306-307` | Memory deallocation | Insert `dealloc` operations |
| 2.4 | `createOptimizeAllocationLivenessPass()` | Func | MLIR standard library<br>`src/Compiler/CompilerPasses.cpp:308` | Lifetime optimization | Optimize allocation lifetimes |
| 2.5 | `createConvertBufferizationToMemRefPass()` | Func | MLIR standard library<br>`src/Compiler/CompilerPasses.cpp:309` | Bufferization → MemRef | Standardize to memref ops |
| 2.6 | `createLowerAffinePass()` | Func | MLIR standard library<br>`src/Compiler/CompilerPasses.cpp:312` | Affine → SCF | `affine.for` → `scf.for` |
| 2.7 | `createSCFToControlFlowPass()` | Func | MLIR standard library<br>`src/Compiler/CompilerPasses.cpp:313` | SCF → CF | `scf.for` → `cf.br` |
| **Phase 3: CF → LLVM (`addLinalgToLLVMPasses`)** |
| 3.1 | `createConvertKrnlToLLVMPass()` | Module | `src/Conversion/KrnlToLLVM/ConvertKrnlToLLVM.cpp`<br>`src/Compiler/CompilerPasses.cpp:334` | Krnl → LLVM + Runtime | `krnl.EntryPoint` → LLVM + Runtime functions |
| 3.2 | `createReconcileUnrealizedCastsPass()` | Module | MLIR standard library<br>`src/Compiler/CompilerPasses.cpp:339` | Cast resolution | Resolve type casts |
| 3.3 | `createCanonicalizerPass()` | Module | MLIR standard library<br>`src/Compiler/CompilerPasses.cpp:340` | Final canonicalization | Final optimization |

---

## Detailed Pass Descriptions

### Phase 1: ONNX → Linalg (`addONNXToLinalgPasses`)

**Location**: `src/Compiler/CompilerPasses.cpp:235-288`

#### 1.1 `createConvertONNXToLinalg()` - Function-level Pass

- **Implementation**: `src/Conversion/ONNXToLinalg/ConvertONNXToLinalg.cpp:33-76`
- **Declaration**: `src/Pass/Passes.hpp:136`
- **Purpose**: Converts ONNX operations to Linalg dialect operations
- **Operations**:
  - Scans function for ONNX ops (currently supports `ONNXMatMulOp`)
  - Calls `populateLoweringONNXMatMulOpToLinalgPattern()` to register conversion patterns
  - Transforms `onnx.MatMul` → `linalg.matmul`
  - Initializes output tensor: `tensor.empty` + `linalg.fill` (zeros)
- **Result**: Linalg dialect operations (`linalg.matmul`, `tensor.empty`, `linalg.fill`)

#### 1.2 `ConvertONNXEntryPointToKrnlPass` - Module-level Pass

- **Implementation**: `src/Compiler/CompilerPasses.cpp:251-278`
- **Pattern Source**: `src/Conversion/ONNXToKrnl/ConvertONNXToKrnl.cpp:35-176` (via `populateLoweringONNXEntryPointOpPattern`)
- **Purpose**: Converts `onnx.EntryPoint` to `krnl.EntryPoint` for runtime function generation
- **Critical Timing**: MUST execute BEFORE bufferization because signature generation requires tensor types, not memref types
- **Operations**:
  - Calls `populateLoweringONNXEntryPointOpPattern()` to reuse existing pattern
  - Generates JSON signature from function type (input/output types, dimensions, names)
  - Creates `krnl.EntryPoint` with signature information
- **Result**: `krnl.EntryPoint` with JSON signature for runtime function generation

#### 1.3 `createOneShotBufferizePass()` - Module-level Pass

- **Source**: MLIR standard library (`mlir/Dialect/Bufferization/Transforms/Passes.h`)
- **Usage**: `src/Compiler/CompilerPasses.cpp:284`
- **Purpose**: Converts Tensor types to Memref types (bufferization)
- **Configuration**: `bufferizeFunctionBoundaries = true` to handle function boundaries
- **Operations**:
  - Transforms `tensor<...>` → `memref<...>`
  - Converts function signatures to memref types
  - Generates memory allocation and access patterns
- **Result**: All tensor types converted to memref types

#### 1.4 `createCanonicalizerPass()` - Module-level Pass

- **Source**: MLIR standard library (`mlir/Transforms/Passes.h`)
- **Usage**: `src/Compiler/CompilerPasses.cpp:287`
- **Purpose**: Canonicalization and optimization
- **Operations**: Removes redundancies, simplifies patterns, optimizes pattern matching

---

### Phase 2: Linalg → Affine/SCF → CF (`addLinalgToAffinePasses`)

**Location**: `src/Compiler/CompilerPasses.cpp:290-314`

All passes in this phase are Function-level (`pm.nest<func::FuncOp>()`)

#### 2.1 `createConvertLinalgToLoopsPass()` - Function-level Pass

- **Source**: MLIR standard library (`mlir/Dialect/Linalg/Passes.h`)
- **Usage**: `src/Compiler/CompilerPasses.cpp:297`
- **Purpose**: Converts Linalg operations to structured loops
- **Operations**:
  - Transforms `linalg.matmul` → nested loops (`affine.for`, `scf.for`)
  - For MatMul: creates `for i, for j, for k` loops
  - Generates memory access index calculations
- **Result**: Structured loops (affine.for, scf.for)

#### 2.2 `createBufferLoopHoistingPass()` - Function-level Pass

- **Source**: MLIR standard library (`mlir/Dialect/Bufferization/Transforms/Passes.h`)
- **Usage**: `src/Compiler/CompilerPasses.cpp:304`
- **Purpose**: Hoists memory allocations outside of loops
- **Reason**: Prevents stack overflow and improves memory efficiency

#### 2.3 `buildBufferDeallocationPipeline()` - Function-level Pipeline

- **Source**: MLIR standard library (`mlir/Dialect/Bufferization/Pipelines/Passes.h`)
- **Usage**: `src/Compiler/CompilerPasses.cpp:306-307`
- **Purpose**: Generates memory deallocation code
- **Operations**:
  - Analyzes lifetime of allocated memory
  - Inserts `dealloc` at appropriate locations
  - Prevents memory leaks
- **Critical Requirement**: Requires structured loops (created in step 2.1)

#### 2.4 `createOptimizeAllocationLivenessPass()` - Function-level Pass

- **Source**: MLIR standard library (`mlir/Dialect/Bufferization/Transforms/Passes.h`)
- **Usage**: `src/Compiler/CompilerPasses.cpp:308`
- **Purpose**: Optimizes memory allocation lifetimes
- **Operations**: Moves deallocation immediately after last use

#### 2.5 `createConvertBufferizationToMemRefPass()` - Function-level Pass

- **Source**: MLIR standard library (`mlir/Dialect/Bufferization/Transforms/Passes.h`)
- **Usage**: `src/Compiler/CompilerPasses.cpp:309`
- **Purpose**: Converts Bufferization dialect ops to MemRef ops
- **Operations**: Transforms temporary bufferization dialect ops to standard memref ops

#### 2.6 `createLowerAffinePass()` - Function-level Pass

- **Source**: MLIR standard library (`mlir/Conversion/AffineToStandard/AffineToStandard.h`)
- **Usage**: `src/Compiler/CompilerPasses.cpp:312`
- **Purpose**: Lowers Affine dialect to SCF dialect
- **Operations**: Converts `affine.for` → `scf.for`

#### 2.7 `createSCFToControlFlowPass()` - Function-level Pass

- **Source**: MLIR standard library (`mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h`)
- **Usage**: `src/Compiler/CompilerPasses.cpp:313`
- **Purpose**: Converts SCF dialect to CF (Control Flow) dialect
- **Operations**:
  - Transforms `scf.for` → `cf.br` (unstructured control flow)
  - Converts structured loops to unstructured control flow
- **Result**: CF dialect (cf.br, cf.cond_br, etc.)

---

### Phase 3: CF → LLVM (`addLinalgToLLVMPasses`)

**Location**: `src/Compiler/CompilerPasses.cpp:316-341`

#### 3.1 `createConvertKrnlToLLVMPass()` - Module-level Pass

- **Implementation**: `src/Conversion/KrnlToLLVM/ConvertKrnlToLLVM.cpp:974-977`
- **Declaration**: `src/Pass/Passes.hpp:124-128`
- **Usage**: `src/Compiler/CompilerPasses.cpp:334`
- **Purpose**: Converts Krnl operations to LLVM operations and generates runtime functions
- **Operations**:
  1. **Entry point preprocessing**: Adds name postfix, removes unhandled parameter attributes
  2. **Runtime information collection**: Records input/output memref types, determines single entry point, determines ownership for output OMTensors
  3. **KrnlEntryPointOp → LLVM conversion**:
     - Generates dynamic entry point functions
     - Converts OMTensor types
     - Initializes accelerators
     - Records signatures
  4. **Runtime function generation**:
     - `omQueryEntryPoints`: Returns list of available entry points
     - `omInputSignature`: Returns input signature JSON
     - `omOutputSignature`: Returns output signature JSON
  5. **Additional features**:
     - Constants file storage (for large models)
     - C wrapper generation
     - `.lrodata` section generation
- **Result**: LLVM dialect operations + Runtime functions

#### 3.2 `createReconcileUnrealizedCastsPass()` - Module-level Pass

- **Source**: MLIR standard library (`mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h`)
- **Usage**: `src/Compiler/CompilerPasses.cpp:339`
- **Purpose**: Resolves unrealized casts
- **Operations**: Resolves type casts remaining from conversion process

#### 3.3 `createCanonicalizerPass()` - Module-level Pass

- **Source**: MLIR standard library (`mlir/Transforms/Passes.h`)
- **Usage**: `src/Compiler/CompilerPasses.cpp:340`
- **Purpose**: Final canonicalization
- **Operations**: Final optimization and cleanup of LLVM IR

---

## Key Characteristics

### 1. **Bufferization Timing**
- Entry point conversion happens **BEFORE** bufferization
- Signature generation requires tensor types (not memref types)
- Bufferization converts all tensors to memrefs after signature generation

### 2. **Structured Loop Preservation**
- Buffer management passes require structured loops
- Must execute buffer management **BEFORE** converting SCF to CF
- `createSCFToControlFlowPass()` removes structured loops, so buffer management must come first

### 3. **Gradual Dialect Lowering**
- Linalg → Loops → Affine → SCF → CF → LLVM
- Each step transforms to a lower-level dialect
- Maintains correctness at each stage

### 4. **Runtime Function Generation**
- Uses `krnl.EntryPoint` (converted from `onnx.EntryPoint`) to generate runtime functions
- Runtime functions are generated in the LLVM phase
- Ensures consistent signature generation with Krnl pipeline

### 5. **Pass Level Organization**
- **Module-level passes**: Handle cross-function operations (entry points, bufferization, runtime)
- **Function-level passes**: Operate within individual functions (loop conversion, memory management)

---

## Pass Registration

Passes are registered in:
- **Custom passes**: `src/Tools/onnx-mlir-opt/RegisterPasses.cpp`
- **MLIR standard passes**: Registered via `registerMLIRPasses()` in `RegisterPasses.cpp:170-213`
- **Linalg passes**: Registered via `registerLinalgPasses()` (MLIR standard)

---

## Entry Point in Compiler

The pipeline is orchestrated in:
- **Main entry**: `src/Compiler/CompilerPasses.cpp:addPasses()` (line 445)
- **Linalg path selection**: `src/Compiler/CompilerPasses.cpp:623-625` (when `useLinalgPath` is true)
- **Phase execution**: 
  - Phase 1: `addONNXToLinalgPasses()` (line 625)
  - Phase 2: `addLinalgToAffinePasses()` (line 634)
  - Phase 3: `addLinalgToLLVMPasses()` (line 653)

---

## Summary

The Linalg pipeline transforms ONNX models through three main phases:
1. **ONNX → Linalg**: Converts high-level ONNX operations to Linalg dialect with proper entry point handling
2. **Linalg → CF**: Gradually lowers Linalg operations through loops, affine, and SCF to unstructured control flow
3. **CF → LLVM**: Converts remaining operations to LLVM dialect and generates runtime functions

Each phase maintains correctness while progressively lowering the abstraction level, ultimately producing executable LLVM IR with complete runtime support.

