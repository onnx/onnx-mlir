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
  guideline](https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md#manipulating-tensorproto-and-numpy-array)
  is a how-to for creating protobuf files from numpy arrays.

## Usage

`utils/RunONNXModel.py` supports the following command-line options:

```bash
$ python ../utils/RunONNXModel.py  --help
usage: RunONNXModel.py [-h]
                       [--print_input]
                       [--print_output]
                       [--save_onnx PATH]
                       [--save_so PATH | --load_so PATH]
                       [--save_data PATH]
                       [--data_folder DATA_FOLDER | --shape_info SHAPE_INFO]
                       [--compile_args COMPILE_ARGS]
                       [--verify {onnxruntime,ref}]
                       [--verify_all_ops]
                       [--compile_using_input_shape]
                       [--rtol RTOL]
                       [--atol ATOL]
                       model_path

positional arguments:
  model_path            Path to the ONNX model

optional arguments:
  -h, --help            show this help message and exit
  --print_input         Print out inputs
  --print_output        Print out inference outputs produced by onnx-mlir
  --save_onnx PATH      File path to save the onnx model
  --save_so PATH        File path to save the generated shared library of the
                        model
  --load_so PATH        File path to load a generated shared library for
                        inference, and the ONNX model will not be re-compiled
  --save_data PATH      Path to a folder to save the inputs and outputs in
                        protobuf
  --data_folder DATA_FOLDER
                        Path to a folder containing inputs and outputs stored
                        in protobuf. If --verify=ref, inputs and outputs are
                        reference data for verification
  --shape_info SHAPE_INFO
                        Shape for each dynamic input of the model, e.g.
                        0:1x10x20,1:7x5x3. Used to generate random inputs for
                        the model if --data_folder is not set
  --compile_args COMPILE_ARGS
                        Arguments passed directly to onnx-mlir command. See
                        bin/onnx-mlir --help
  --verify {onnxruntime,ref}
                        Verify the output by using onnxruntime or reference
                        inputs/outputs. By default, no verification
  --verify_all_ops      Verify all operation outputs when using onnxruntime.
  --compile_using_input_shape
                        Compile the model by using the shape info getting from
                        the inputs in data folder. Must set --data_folder
  --rtol RTOL           Relative tolerance for verification
  --atol ATOL           Absolute tolerance for verification
```
