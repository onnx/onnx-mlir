<!--- SPDX-License-Identifier: Apache-2.0 -->

# ONNX-MLIR: Testing and Specific Environment variables

## High level testing of ONNX-MLIR

To run the lit ONNX-MLIR tests, use the following command:

[same-as-file]: <> ({"ref": "utils/check-onnx-mlir.cmd", "skip-ref": 1})
```shell
call cmake --build . --config Release --target check-onnx-lit
```

To run the numerical ONNX-MLIR tests, use the following command:

[same-as-file]: <> ({"ref": "utils/check-onnx-numerical.cmd", "skip-ref": 1})
```shell
call cmake --build . --config Release --target check-onnx-numerical
```

To run the doc ONNX-MLIR tests, use the following command after installing third_party ONNX shown below. Details to first install the third_party ONNX project are detailed [here](BuildONNX.md). Note that it is key to install the ONNX project's version listed in our third_party subdirectory, as ONNX-MLIR may be behind the latest version from the ONNX standard.

[same-as-file]: <> ({"ref": "utils/check-docs.cmd", "skip-ref": 1})
```shell
call cmake --build . --config Release --target check-docs
```

## Summary of LLVM and ONNX-MLIR Environment Variables

The following CMake variables from LLVM and ONNX-MLIR can be used when compiling ONNX-MLIR.

**MLIR_DIR**:PATH
  Path to to the mlir cmake module inside an llvm-project build or install directory (e.g., c:/repos/llvm-project/build/lib/cmake/mlir).
  This is required if **MLIR_DIR** is not specified as an environment variable.

**LLVM_EXTERNAL_LIT**:PATH
  Path to the lit tool. Defaults to an empty string and LLVM will find the tool based on **MLIR_DIR** if possible.
  This is required when **MLIR_DIR** points to an install directory.

