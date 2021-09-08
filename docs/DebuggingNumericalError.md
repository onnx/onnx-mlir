<!--- SPDX-License-Identifier: Apache-2.0 -->

# Debugging Numerical Error

Use `utils/RunONNXModel.py` python script to debug numerical errors, when
onnx-mlir-compiled inference executable produces numerical results that are
inconsistent with those produced by the training framework. This python script
will run the model through onnx-mlir and a reference backend, and compare the
intermediate results produced by these two backends layer by layer.

## Prerequisite
- Set `ONNX_MLIR_HOME` environment variable to be the path to the HOME
  directory for onnx-mlir. The HOME directory for onnx-mlir refers to the
  parent folder containing the `bin`, `lib`, etc sub-folders in which ONNX-MLIR
  executables and libraries can be found.

## Reference backend
Outputs by onnx-mlir can be verified by using a reference ONNX backend or
reference inputs and outputs in protobuf.
- To verify using a reference backend, install onnxruntime by running `pip
  install onnxruntime`. To use a different testing backend, simply replace code
  importing onnxruntime to some other ONNX-compliant backend.
- To verify using reference outputs, use `--verify=ref --ref_folder=ref_folder`
  where `ref_folder` is the path to a folder containing protobuf files for
  inputs and outputs. [This
  guideline](https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md#manipulating-tensorproto-and-numpy-array)
  is a how-to for creating protobuf files from numpy arrays. 

## Usage

`utils/RunONNXModel.py` supports the following command-line options:

```bash
$ python ../utils/RunONNXModel.py  --help
usage: RunONNXModel.py [-h] [--print_input] [--print_output] [--compile_args COMPILE_ARGS] [--shape_info SHAPE_INFO] [--verify {onnxruntime,ref}]
                       [--ref_folder REF_FOLDER] [--rtol RTOL] [--atol ATOL]
                       model_path

positional arguments:
  model_path            Path to the ONNX model.

optional arguments:
  -h, --help            show this help message and exit
  --print_input         Print out inputs
  --print_output        Print out outputs
  --compile_args COMPILE_ARGS
                        Arguments passed directly to onnx-mlir command. See bin/onnx-mlir --help
  --shape_info SHAPE_INFO
                        Shape for each dynamic input, e.g. 0:1x10x20,1:7x5x3
  --verify {onnxruntime,ref}
                        Verify the output by using onnxruntime or reference inputs/outputs. By default, no verification
  --ref_folder REF_FOLDER
                        Path to the folder containing reference inputs and outputs stored in protobuf. Used when --verify=ref
  --rtol RTOL           Relative tolerance for verification
  --atol ATOL           Absolute tolerance for verification
```
