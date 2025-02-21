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
- To verify using reference outputs, use `--verify=ref --load-ref=data_folder`
  where `data_folder` is the path to a folder containing protobuf files for
  inputs and outputs. [This
  guideline](https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md#manipulating-tensorproto-and-numpy-array)
  is a how-to for creating protobuf files from numpy arrays.

## Usage

`utils/RunONNXModel.py` supports the following command-line options:

```
$ python ../utils/RunONNXModel.py  --help
usage: RunONNXModel.py [-h] [--log-to-file [LOG_TO_FILE]] [--model MODEL] [--compile-args COMPILE_ARGS] [--compile-only] [--compile-using-input-shape] [--print-input]
                       [--print-output] [--save-onnx PATH] [--verify {onnxruntime,ref}] [--verify-all-ops] [--verify-with-softmax] [--verify-every-value] [--rtol RTOL]
                       [--atol ATOL] [--save-so PATH | --load-so PATH] [--save-ref PATH] [--load-ref PATH | --shape-info SHAPE_INFO] [--lower-bound LOWER_BOUND]
                       [--upper-bound UPPER_BOUND]

optional arguments:
  -h, --help                  show this help message and exit
  --log-to-file [LOG_TO_FILE] Output compilation messages to file, default compilation.log
  --model MODEL               Path to an ONNX model (.onnx or .mlir)
  --compile-args COMPILE_ARGS Arguments passed directly to onnx-mlir command. See bin/onnx-mlir --help
  --compile-only              Only compile the input model
  --compile-using-input-shape Compile the model by using the shape info getting from the inputs in the reference folder set by --load-ref
  --print-input               Print out inputs
  --print-output              Print out inference outputs produced by onnx-mlir
  --save-onnx PATH            File path to save the onnx model. Only effective if --verify=onnxruntime
  --verify {onnxruntime,ref}  Verify the output by using onnxruntime or reference inputs/outputs. By default, no verification. When being enabled, --verify-with-softmax or --verify-every-value must be used to specify verification mode.
  --verify-all-ops            Verify all operation outputs when using onnxruntime
  --verify-with-softmax       Verify the result obtained by applying softmax to the output
  --verify-every-value        Verify every value of the output using atol and rtol
  --rtol RTOL                 Relative tolerance for verification
  --atol ATOL                 Absolute tolerance for verification
  --save-so PATH              File path to save the generated shared library of the model
  --load-so PATH              File path to load a generated shared library for inference, and the ONNX model will not be re-compiled
  --save-ref PATH             Path to a folder to save the inputs and outputs in protobuf
  --load-ref PATH             Path to a folder containing reference inputs and outputs stored in protobuf. If --verify=ref, inputs and outputs are reference data for verification
  --shape-info SHAPE_INFO     Shape for each dynamic input of the model, e.g. 0:1x10x20,1:7x5x3. Used to generate random inputs for the model if --load-ref is not set
  --lower-bound LOWER_BOUND   Lower bound values for each data type. Used inputs. E.g. --lower-bound=int64:-10,float32:-0.2,uint8:1. Supported types are bool, uint8, int8, uint16, int16, uint32, int32, uint64, int64,float16, float32, float64
  --upper-bound UPPER_BOUND   Upper bound values for each data type. Used to generate random inputs. E.g. --upper-bound=int64:10,float32:0.2,uint8:9. Supported types are bool, uint8, int8, uint16, int16, uint32, int32, uint64, int64, float16, float32, float64
```

## Helper script to compare a model under two distinct compile option.

Based on the above `utils/runONNXModel.py`, the `utils/checkONNXModel.py` allows a user to run a given model twice, under two distinct compile options, and compare its results.
This let a user simply test a new option, comparing the safe version of the compiler (e.g. `-O0` or `-O3`) with a more advanced version (e.g. `-O3` or `-O3 --march=x86-64`). Simply specify the compile options using the `--ref-compile-args` and `--test-compile-args` flags, a model using the `--model` flag, and possibly a `--shape-info` in presence of dynamic shape inputs.
Full options are listed under the `--help` flag.

## Debugging the Code Generated for an Operator.

If you know, or suspect, that a particular ONNX MLIR operator produces an incorrect result, and want to narrow down the problem, we provide a couple of useful Krnl operators that allow printing (at runtime) the value of a tensor, or a value that has a primitive data type. 

To print out the value of a tensor at a particular program point, inject the following code (where `X` is the tensor to be printed):

```code
create.krnl.printTensor("Tensor X: ", X);
```

Note: currently the content of the tensor is printed only when the tensor rank is less than four.

To print a message followed by one value, inject the following code (where `val` is the value to be printed and `valType` is its type):

```code
create.krnl.printf("inputElem: ", val, valType);
```

## Finding memory errors

If you know, or suspect, that an onnx-mlir-compiled inference executable
suffers from memory allocation related issues, the
[valgrind framework](https://valgrind.org/) or
[mtrace memory tool](https://github.com/sstefani/mtrace) can be used to facilitate debugging.
These tools trace memory
allocation/free-related APIs, and can detect memory issues, such as memory leaks.

However if the problems relating to memory access, especially buffer overrun problems, are notoriously difficult to debug because run-time errors occur outside of the point containing the problem. 
The ["Electric Fence library"](https://github.com/CheggEng/electric-fence) can be
used for debugging these problems. It helps you detect two common programming problems: software that overruns the boundaries of a malloc() memory allocation, and
software that touches a memory allocation
that has been released by free(). Unlike other memory debuggers, Electric
Fence will detect read accesses as well as writes, and it will pinpoint the
exact instruction that causes an error.

Since the Electric Fence library is not officially supported by RedHat, you
need to download, build and install the source code by yourself on yours.
After installing it, link this library by using the "-lefence" option when
generating inference executables. Then simply execute it, which will
cause a runtime error and stop at the place causing memory access problems. You can
identify the place with a debugger or debugging print functions
described in the previous section.
