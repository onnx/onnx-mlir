<!--- SPDX-License-Identifier: Apache-2.0 -->

# ONNX-MLIR: Build trouble-shooting and testing ONNX_MLIR

## Trouble shooting the building of ONNX-MLIR

If you have issues during the first `onnx-mlir` build, you may need to check the cmake variables used by our build. See the last section of this page for help.

If you have used the source directory successfully for a while, you may experience difficulties to rebuild `onnx-mlir` after merging the latest changes from the `main` branch.

Below is a couple of steps you may perform. If any of them apply, it is recommended to remove the `onnx-mlir/build` subdirectory and rebuild from scratch using the `cmake` commands.

### 1) Checking the right commit of the llvm-project

If the latest `onnx-mlir` `main` branch has moved to a newer commit level of the `llvm-project`, the build process will typically experience multiple compiler failures related to LLVM and MLIR code.

Level required is found in the first code box of the [Building ONNX-MLIR](BuildOnLinuxOSX.md#mlir) page next to the `git checkout` command.

Level used in the code is found by executing a `git log` in the `llvm-project` subdirectory.

If they don't match, please update the llvm project to the required level.

### 2) Checking the right third_party support

Typically, when we update the ONNX op level, it results in new software in the `third_party/onnx` subdirectory. Failing to update that code results typically in compiler failures related to ONNX dialect code.

It is easier to simply remove the `third_party` directory and then reinstalling the code using `git submodule update --init --recursive`.

### 3) Dialect update

Sometimes a dialect update requires the entire build directory to be rebuilt. Typical errors that you may see are missing declarations, for example to `verifier` methods. The recommendation is to simply remove the `onnx-mlir/build` subdirectory and rebuild from scratch using the `cmake` commands.

### 4) Protobuf related issues

If you run into protobuf related errors during the build, check the following potential causes:

* protobuf version is too low or too new (relative to the prereq)
* libprotobuf version and python binding version mismatch
* llvm-project, onnx, and/or onnx-mlir are built against different versions of protobuf, because after updating protobuf you only rebuild one of them
* llvm-project, onnx, and/or onnx-mlir may detect different versions of python3 (so watch their cmake output) if you have multiple python versions installed
* cmake caches stuff and you should never use "make clean" when rebuilding. Instead remove everything under the build tree and start from scratch.

These and many other trickeries for setting up the build env are the reason why we recommend using the [onnxmlir/onnx-mlir-dev](https://github.com/users/onnxmlir/packages/container/onnx-mlir-dev) docker image for development.

## High level testing of ONNX-MLIR

To run the lit ONNX-MLIR tests, use the following command:

[same-as-file]: <> ({"ref": "utils/check-onnx-mlir.cmd", "skip-ref": 1})
```shell
call cmake --build . --config Release --target check-onnx-lit
```

Or simply invoke the `check-onnx-lit` target for `ninja` or `make` in the build directory.

To run the numerical ONNX-MLIR tests, use the following command:

[same-as-file]: <> ({"ref": "utils/check-onnx-numerical.cmd", "skip-ref": 1})
```shell
call cmake --build . --config Release --target check-onnx-numerical
```

Or simply invoke the `check-onnx-numerical` target for `ninja` or `make` in the build directory.

To run the doc ONNX-MLIR tests, use the following command after installing third_party ONNX shown below. Details to first install the third_party ONNX project are detailed [here](BuildONNX.md). Note that it is key to install the ONNX project's version listed in our third_party subdirectory, as ONNX-MLIR may be behind the latest version from the ONNX standard.

[same-as-file]: <> ({"ref": "utils/check-docs.cmd", "skip-ref": 1})
```shell
call cmake --build . --config Release --target check-docs
```

Or simply invoke the `check-docs` target for `ninja` or `make` in the build directory.

# Summary of LLVM and ONNX-MLIR Cmake Variables

The following CMake variables from LLVM and ONNX-MLIR can be used when compiling ONNX-MLIR.

**MLIR_DIR**:PATH
  Path to to the mlir cmake module inside an llvm-project build or install directory (e.g., c:/repos/llvm-project/build/lib/cmake/mlir).
  This is required if **MLIR_DIR** is not already set from a previous cmake invocation.

**LLVM_EXTERNAL_LIT**:PATH
  Path to the lit tool. Defaults to an empty string and LLVM will find the tool based on **MLIR_DIR** if possible.
  This is required when **MLIR_DIR** points to an install directory.

