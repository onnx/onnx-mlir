# A guideline on adding a new custom accelerator

In general, onnx-mlir handles custom accelerators as pluggins which can be turned on/off when building onnx-mlir and compiling a model. The handling is via `cmake` and we will outline its procedure in this document.

Besides this document, [NNPA accelerator](../src/Accelerators/NNPA) can be used as an example that has been deployed in onnx-mlir.

## 1. Code folder
In onnx-mlir, all code for an accelerator should be put inside a separate folder under `src/Accelerators`. Thus, the first step to support an accelerator is to create a folder for it inside `src/Accelerators`.

The folder content is flexible depending on each accelerator. However, we recomment to follow the same structure as the root folder of `onnx-mlir` as much as possbile. This helps maintain the consitency across the whole project.

The folder name will be used as the accelerator name in onnx-mlir. In particular, it is used to
1. instruct `cmake` to build the code inside the accelerator folder,
2. compile a model for the accelerator when using `onnx-mlir` command, and
3. enable passes related to the accelerator when using `onnx-mlir-opt` command.

### 1.1 Build accelerators in onnx-mlir
To build accelerators in onnx-mlir, use the cmake variable `ONNX_MLIR_ACCELERATORS` when building onnx-mlir. `ONNX_MLIR_ACCELERATORS` accepts a comma-separated list of accelerator names. For example,
```bash
$ cd build
$ cmake .. -DONNX_MLIR_ACCELERATORS=accel1,accel2
```

### 1.2 Compile a model to run with selected accelerators.
The compiler command `onnx-mlir` has an option, i.e. `--maccel`, to compile a model for selected accelerators. `--maccel` accepts a comma-separated list of accelerator names. For example,

```bash
$ onnx-mlir --maccel=accel1,accel2 model.onnx
```

Only built accelerators can be used with `--maccel`.

### 1.3 Run passes related to selected accelerators.
Passes defined by an accelerator can be run or tested via `onnx-mlir-opt` command by using option `--maccel` which is similar to `--maccel` in `onnx-mlir` (See Sec. [1.2](#1.2-compile-a-model-to-run-with-selected-accelerators)). For example, to call a pass `--optimize-data-layout` defined by accelerator `accel1`:

```bash
$ onnx-mlir-opt --maccel=accel1 --convert-onnx-to-accel  model.mlir
```

Only built accelerators can be used with `--maccel`.

## 2. Code integration

## 3. Testing
