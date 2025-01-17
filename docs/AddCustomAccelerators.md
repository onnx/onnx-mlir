# A guideline on adding a new custom accelerator

In general, onnx-mlir handles custom accelerators as pluggins which can be turned on/off when building onnx-mlir and compiling a model. The handling is mainly via `cmake` and we will outline its procedure in this document.

Besides this document, [NNPA accelerator](../src/Accelerators/NNPA) can be used as an example that has been deployed in onnx-mlir.

## 1. Code folder

In onnx-mlir, all code for an accelerator should be put inside a separate folder under `src/Accelerators`. Thus, the first step to support an accelerator is to create a folder for it inside `src/Accelerators`.

The folder name will be used as the accelerator name in onnx-mlir. In particular, it is used to
1. instruct `cmake` to build the code inside the accelerator folder,
2. compile a model for the accelerator when using `onnx-mlir` command, and
3. enable passes related to the accelerator when using `onnx-mlir-opt` command.

The folder content is flexible depending on each accelerator. However, we recomment to follow the same structure as the root folder of `onnx-mlir` as much as possbile. This helps maintain the consitency across the whole project.

### 1.1 Build accelerators in onnx-mlir

To build accelerators in onnx-mlir, use the cmake variable `ONNX_MLIR_ACCELERATORS` when building onnx-mlir. `ONNX_MLIR_ACCELERATORS` accepts a semicolon-separated list of accelerator names. For example,
```bash
$ cd build
$ cmake .. -DONNX_MLIR_ACCELERATORS='accel1;accel2'
```
Note that the list should be quoted.

### 1.2 Compile a model to run with selected accelerators.

The compiler command `onnx-mlir` has an option, i.e. `--maccel`, to compile a model for selected accelerators. For each accelerator add a `--maccel=accel_name` entry. For example,

```bash
$ onnx-mlir --maccel=accel1 --maccel=accel2 model.onnx
```

Only built accelerators can be used with `--maccel`.

### 1.3 Run passes related to selected accelerators.

Passes defined by an accelerator can be run or tested via `onnx-mlir-opt` command by using option `--maccel` which is similar to `--maccel` in `onnx-mlir` (See Sec. [1.2](#1.2-compile-a-model-to-run-with-selected-accelerators)). For example, to call a pass `--optimize-data-layout` defined by accelerator `accel1`:

```bash
$ onnx-mlir-opt --maccel=accel1 --optimize-data-layout model.mlir
```

Only built accelerators can be used with `--maccel`.

## 2. Code integration

### 2.1 Macro

Each accelerator is required to define a few macros. These needs to be included in [onnx_mlir::accel::Accelerator](../src/Accelerators/Accelerator.hpp). These macros are:

1. `INSTRUMENTSTAGE_ENUM_<accel_name>`
2. `INSTRUMENTSTAGE_CL_ENUM_<accel_name>`
3. `PROFILEIR_CL_ENUM_<accel_name>`
4. `OPTREPORT_ENUM_<accel_name>`
5. `OPTREPORT_CL_ENUM_<accel_name>`

Replace `<accel_name>` with the name of the accelerator, for example if your accelerator is named `ACCEL1` use:

```C
#define INSTRUMENTSTAGE_ENUM_ACCEL1
#define INSTRUMENTSTAGE_CL_ENUM_ACCEL1
#define PROFILEIR_CL_ENUM_ACCEL1
#define OPTREPORT_ENUM_ACCEL1
#define OPTREPORT_CL_ENUM_ACCEL1
```

### 2.2 Dialects and passes

Writing code in MLIR typically involves desiging dialects and passes. So does supporting an accelerator. Thus, to integrate accelerator code into onnx-mlir is to register dialects and passes in onnx-mlir.

We provide a base class [onnx_mlir::accel::Accelerator](../src/Accelerators/Accelerator.hpp) from which users can define an inherited class and write hooks to register dialects and passes.

```C
//===--------------------------------------------------------------------===//
// Hooks for onnx-mlir driver
//===--------------------------------------------------------------------===//

/// Add the transformations necessary to support the accelerator.
virtual void addPasses(mlir::OwningOpRef<mlir::ModuleOp> &module,
    mlir::PassManager &pm,
    onnx_mlir::EmissionTargetType &emissionTarget) const = 0;

//===--------------------------------------------------------------------===//
// Hooks for onnx-mlir-opt driver
//===--------------------------------------------------------------------===//

/// Register the MLIR dialects required to support an accelerator.
virtual void registerDialects(mlir::DialectRegistry &registry) const = 0;

/// Register accelerator transformation passes to make available as
/// command line options.
virtual void registerPasses(int optLevel) const = 0;

//===--------------------------------------------------------------------===//
// Hooks for both onnx-mlir and onnx-mlir-opt drivers
//===--------------------------------------------------------------------===//

/// Configure passes for the accelerator.
virtual void configurePasses() const = 0;

//===--------------------------------------------------------------------===//
// Hooks for onnx-to-krnl pass
//===--------------------------------------------------------------------===//

/// Convert TensorType to MemRefType.
/// Acccelators may have special versions of TensorType. If not, override this
/// method and return nullptr.
virtual mlir::MemRefType convertTensorTypeToMemRefType(
    const mlir::TensorType tensorType) const = 0;

/// Define conversion target to be used with ONNXToKrnl.
virtual void conversionTargetONNXToKrnl(
    mlir::ConversionTarget &target) const = 0;

/// Define rewrite patterns to be used with ONNXToKrnl.
virtual void rewritePatternONNXToKrnl(mlir::RewritePatternSet &patterns,
    mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx) const = 0;

//===--------------------------------------------------------------------===//
// Hooks for krnl-to-llvm pass
//===--------------------------------------------------------------------===//

/// Define conversion target to be used with KrnlToLLVM.
virtual void conversionTargetKrnlToLLVM(
    mlir::ConversionTarget &target) const = 0;

/// Define rewrite patterns to be used with KrnlToLLVM.
virtual void rewritePatternKrnlToLLVM(mlir::RewritePatternSet &patterns,
    mlir::LLVMTypeConverter &typeConverter, mlir::MLIRContext *ctx) const = 0;
```

Though there are many passes in onnx-mlir, we provide hooks for two passes `onnx-to-krnl` and `krnl-to-llvm` only. The reason is that in principal they are the first and the last passes in onnx-mlir. Pass `onnx-to-krnl` is the place where we can decide which ONNX operators will be run on host (by lowering them to Krnl dialect) or on an accelerator (by lowering them to a dialect defined for the accelerator). Pass `krnl-to-llvm` is the place where we lower Krnl and accelerator operators to LLVM dialect, e.g. generate assembly code or simply call external APIs for the accelerator. There can have any dialects and passes for the accelerator between `onnx-to-krnl` and `krnl-to-llvm`.

For example, for NNPA acclerator, we define [ZHigh dialect](../src/Accelerators/NNPA/Dialect/ZHigh) to be used in `onnx-to-krnl` and [ZLow dialect](../src/Accelerators/Dialect/ZLow) to be used in `krnl-to-llvm`.

## 3. Testing

Tests for accelerators should be put inside the folder [test](../test). In particular,
- LIT tests are placed inside a newly-created folder under [mlir/accelerators](../test/mlir/accelerators)
- Other tests are place inside a newly-created folder under [accelerators](../test/accelerators)
