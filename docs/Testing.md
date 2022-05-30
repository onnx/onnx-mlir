<!--- SPDX-License-Identifier: Apache-2.0 -->

# Testing

In onnx-mlir, there are three types of tests to ensure correctness of implementation:
1. [ONNX Backend Tests](#onnx-backend-tests)
2. [LLVM FileCheck Tests](#llvm-filecheck-tests)
3. [Numerical Tests](#numerical-tests)
4. [Use gdb](#use-gdb)
4. [ONNX Model Zoo](#onnx-model-zoo)

## ONNX Backend Tests

Backend tests are end-to-end tests for onnx-mlir based on onnx node and model tests. They are available for testing both the C/C++ .so library and the JNI .jar archive. For each C/C++ test target, adding the `-jni` suffix gives the corresponding JNI test target.
To invoke the test, use the following command:

```
cmake --build . --config Release --target check-onnx-backend[-jni]
``` 
Packages, such as third_party/onnx, needs to be installed to run the backend test. JNI test requires the jsoniter jar which is downloaded from its maven repository by default if no installed version is found on the system. If the user turns on the cmake option `ONNX_MLIR_BUILD_JSONITER` when building ONNX-MLIR, the jsoniter jar will be built locally from the source cloned from its github repository. Note that building jsoniter jar locally requires the maven build tool to be installed.

The node and model tests in onnx that will be run by check-onnx-backend is defined by variable test_to_enable in test/backend/test.py. User can test one test case by environment variable `TEST_CASE_BY_USER`. For example,
```
TEST_CASE_BY_USER=selected_test_name cmake --build . --config Release --target check-onnx-backend[-jni]
```
With `TEST_CASE_BY_USER` specified, the intermediate result, the .onnx file and .so file, are kept in build/test/backend for debugging.

When the ONNX-to-Krnl conversion of an operator is added, the corresponding backend tests for this operator should be added to test.py. The available test cases can be found in third_part/onnx/onnx/backend/test/case/node. Please note to add suffix `_cpu` to the onnx test name. 

### Tests with unknown dimensions

Testing with dynamic tensor sizes is most easily performed by using the following command, also used by our checkers. 
```
cmake --build . --config Release --target check-onnx-backend-dynamic[-jni]
``` 

The onnx node tests usually have known dimension size for input tensors. So, to test tensor with unknown dimension, the model importer (Build/FrontendONNXTransformer.cpp) provides a functionality to generate such cases. When the environment variable, `IMPORTER_FORCE_DYNAMIC`, is set, the frontend import will turn the all the dimensions (by default) of all the input tensors of the model into -1. For example,
```
IMPORTER_FORCE_DYNAMIC='-1:-1' all dimensions of all the inputs will be changed
IMPORTER_FORCE_DYNAMIC='0:-1' all dimensions of the first input will be changed
IMPORTER_FORCE_DYNAMIC='0:-1|1:0,1' all dimensions of the first input and the 1st and 2nd dimensions of the second input will be changed
```

The Backus-Naur Form (BNF) for `IMPORTER_FORCE_DYNAMIC` is as follows.
```
<ImportForceDynamicExpr> :== `'` <expr> `'`
                  <expr> ::= <inputString> | <inputString> `|` <expr>
            <inputString ::= <inputIndex> `:` <dimString>
             <dimString> ::= <dimIndex> | <dimIndex> `,` <dimString>
            <inputIndex> ::= <index>
              <dimIndex> ::= <index>
                 <index> ::= -1 | <number>
                <number> ::= <digit> | <digit><number>
                 <digit> ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
```
Value `-1` semantically represents all inputs or all dimensions, and it has the highest priority. E.g. `'0: -1, 0'` means all dimensions of the first input will be changed. Input and dimension indices start from 0.

For example, the default model for test_add_cpu is:
```
func @main_graph(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<3x4x5xf32>
```
with `IMPORTER_FORCE_DYNAMIC='-1:-1'`, the result is:
```
func @main_graph(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
```
with `IMPORTER_FORCE_DYNAMIC='0:-1'`, the result is:
```
func @main_graph(%arg0: tensor<?x?x?xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<3x4x5xf32>
```
with `IMPORTER_FORCE_DYNAMIC='0:0,2|1:1'`, the result is:
```
func @main_graph(%arg0: tensor<?x4x?xf32>, %arg1: tensor<3x?x5xf32>) -> tensor<3x4x5xf32>
```
This is a way to use existing node test for dynamic tensors. Since not all test case can pass with dynamic tensor, there is a list in test/backend/test.py, test_not_for_dynamic, to specify which test can not pass with `IMPORTER_FORCE_DYNAMIC` is defined.

### Tests with constant inputs

Because the onnx node tests accepts input tensors at runtime, the inputs are not
constants when compiling the onnx model. However, in practice, inputs can be
constants and we want to test such a situation.

Testing with constant inputs is most easily performed by using the following
command, also used by our checkers.
```
cmake --build . --config Release --target check-onnx-backend-constant[-jni]
```

To test a single onnx node, e.g. `test_add_cpu`, use two environment variables
`TEST_CONSTANT` and `IMPORTER_FORCE_CONSTANT`, e.g.:
```
TEST_CONSTANT=true IMPORTER_FORCE_CONSTANT="0" TEST_CASE_BY_USER=test_add_cpu make check-onnx-backend[-jni]
```
which turns the first input (index 0) to a constant, and thus the model now has
only one input instead of two.

The environment variable `IMPORTER_FORCE_CONSTANT` is a list of indices
separated by `,` (starting from 0, or -1 for all input indices), e.g. `0, 2, 3`
or `-1`.

### Input Signature tests

Testing input signature of an onnx models with a variety of data type by using the following command, also used by our checkers.

```
cmake --build . --config Release --target check-onnx-backend-signature
```

### Enable SIMD instructions

On supported platforms, currently s390x only, backend tests can generate SIMD instructions for the compiled models. To enable SIMD, set the TEST_MCPU environment variable, e.g.,
```
TEST_MCPU=z14 cmake --build . --config Release --target check-onnx-backend[-jni]
```

### Execution of backend tests

A tool defined in `utils/RunONNXLib.cpp` can be used to easily execute files from their `.so`
models, such as the ones generated using the
`TEST_CASE_BY_USER=selected_test_name make check-onnx-backend` command.
Models can also be preserved when built in other manners by setting the
`overridePreserveFiles` value in the `onnx-mlir/src/Compiler/CompilerUtils.cpp` file to
`KeepFilesOfType::All`, for example.

When the onnx model is older than the current version supported by onnx-mlir, 
onnx version converter can be invoked with environment variable `INVOKECONVERTER` set 
to true. For example, converter will be called for all test cases for 
`INVOKECONVERTER=true make check-onnx-backend`. 
In test.py, there is a list called `test_need_converter` for you to invoke converter on individual cases.

The tool directly scans the signature provided by the model, initializes the needed inputs with random
values, and then makes a function call into the model. The program can then be used in conjunction
with other tools, such as `gdb`, `lldb`, or `valgrind`.
To list the utility options, simply use the `-h` or `--help` flags at runtime.

We first need to compile the tool, which can be done in one of two modes.
In the first mode, the tool is compiled with a statically linked model.
This mode requires the `-D LOAD_MODEL_STATICALLY=0` option during compilation in addition to including the `.so` file.
Best is to use the `build-run-onnx-lib.sh` script in the `onnx-mlir/utils` directory to compile the tool with its model, which is passed as a parameter to the script.
To avoid library path issues, just run the tool in the home directory of the model.

``` sh
# Compile tool with model.
cd onnx-mlir/build
. ../utils/build-run-onnx-lib.sh test/backend/test_add.so
# Run tool in the directory of the model.
(cd test/backend; run-onnx-lib)
```

In the second mode, the tool is compiled without models, which will be passed at runtime.
To enable this option, simply compile the tool with the `-D LOAD_MODEL_STATICALLY=1` option.
You may use the same script as above but without arguments. The tool can then be be run from
any directories as long as you pass the `.so` model file at runtime to the tool.

``` sh
# Compile tool without a model.
cd onnx-mlir/build
. ../utils/build-run-onnx-lib.sh
# Run the tool with an argument pointing to the model.
run-onnx-lib test/backend/test_add.so
```

## LLVM FileCheck Tests

We can test the functionality of one pass by giving intermediate representation
as input and checking the output IR with LLVM FileCheck utility.
For example, we have a test case, test.mlir,  for shape inference.
```
func @test_default_transpose(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) : (tensor<5x5x1x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
```

You can run the shape inference pass  on this test case, and get the following 
output:
```
module  {
  func @test_default_transpose(%arg0: tensor<5x5x1x32xf32>) -> tensor<32x1x5x5xf32> {
    %0 = "onnx.Transpose"(%arg0) {perm = [3, 2, 1, 0]} : (tensor<5x5x1x32xf32>) -> tensor<32x1x5x5xf32>
    return %0 : tensor<32x1x5x5xf32>
  }
}
```
Manually check whether the output is correct.
If the output is correct, cover the output to what can be automatically checked
in future. Use command:
```
Debug/bin/onnx-mlir-opt --shape-inference test.mlir | python ../utils/mlir2FileCheck.py 
```
You will get the following:
```
// mlir2FileCheck.py
// CHECK-LABEL:  func @test_default_transpose
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<5x5x1x32xf32>) -> tensor<32x1x5x5xf32> {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Transpose"([[PARAM_0_]]) {perm = [3, 2, 1, 0]} : (tensor<5x5x1x32xf32>) -> tensor<32x1x5x5xf32>
// CHECK:           return [[VAR_0_]] : tensor<32x1x5x5xf32>
// CHECK:         }
```
Combine the source and the check code and add to the adequate test cases. 
All the test cases for onnx dialect are collected under test/mlir/onnx directory.
These test cases can be invoked with `make check-onnx-lit`. 
This target is an essential requirement for a build.

## Numerical Tests

Numerical tests are used to test for numerical correctness in addition to the tests provided by the ONNX package.
The goal is to provide extensive numerical value based unit tests; this is very important for ensuring that
optimization transformations are valid and correct: more corner cases will arise as we specialize for specific 
architecture parameters (like vector width). Numerical tests generates extensive amount of numerical value-based 
unit tests based on simple, naive (and extremely slow) implementation of operations being tested, used to verify 
the correctness of our operation lowering and optimization.

Numerical tests should be structured such that the following two components are independent and separate:
- Generation of test case parameters (for instance, the dimensions of convolutions N, C, H, W, kH, kW ...).
- Checking that the values produced by onnx-mlir is consistent with those produced by naive implementation.

The motivation is that there are two ways we want to generate test case parameters:
- Exhaustive generation of test case parameters. Where we want to exhaustively test the correctness of a small range
of parameters (for instance, if we would like to test and verify that 3x3 convolution is correctly implemented for
all valid padding configurations.)
- When the possible parameter space is extremely large, we can rely on RapidCheck to randomly generate test cases
that becomes increasingly large as smaller test cases succeed. And it also automatically shrinks the test cases
in the event that an error occurs. For example, the following RapidCheck test case automatically generates test
case parameters (N from between 1 and 10, C from within 1 and 20 etc...). By default rc::check will draw 100 sets of
test case parameters and invoke the value checking function `isOMConvTheSameAsNaiveImplFor`.

```cpp
  // RapidCheck test case generation.
  bool success = rc::check("convolution implementation correctness", []() {
    const auto N = *rc::gen::inRange(1, 10);
    const auto C = *rc::gen::inRange(1, 20);
    const auto H = *rc::gen::inRange(5, 20);
    const auto W = *rc::gen::inRange(5, 20);

    const auto kH = *rc::gen::inRange(1, 15);
    const auto kW = *rc::gen::inRange(1, 15);

    // We don't want an entire window of padding.
    const auto pHBegin = *rc::gen::inRange(0, kH - 1);
    const auto pHEnd = *rc::gen::inRange(0, kH - 1);
    const auto pWBegin = *rc::gen::inRange(0, kW - 1);
    const auto pWEnd = *rc::gen::inRange(0, kW - 1);

    // Make sure we have at least 1 output per dimension.
    RC_PRE((H >= kH) && (W > kW));

    RC_ASSERT(isOMConvTheSameAsNaiveImplFor(
        N, C, H, W, kH, kW, pHBegin, pHEnd, pWBegin, pWEnd));
  });
  assert(success && "error while performing RapidCheck tests");
```
  
Sometimes it is convenient to be able to see the mlir files associated with a
numerical tests. To do so, the easiest is to set the `overridePreserveFiles`
variable in `src/Compiler/CompilerUtils.cpp` to the types of files that you want to
preserve (e.g. `KeepFilesOfType::All`). Then, no matter how you compile
your model, input and output mlir files will be preserved, as well as
unoptimized and optimized bytecode files as well as a few additional binaries.

In case of failures, both RapidCheck (infrastructure used for numerical testing) and the onnx models allow a user to re-run a test with the same values. When running a test, you may get the following output.
```
Model will use the random number generator seed provided by "TEST_SEED=1440995966"
RapidCheck Matrix-Vector test case generation.
Using configuration: seed=4778673019411245358
```

By recording the seed values in the following two environment variables:
```
export RC_PARAMS="seed=4778673019411245358"
export TEST_SEED=1440995966
```
you can force, respectively, the random seeds used in RapidCheck and the random seeds used to populate the ONNX input vectors to be the same. Set only the first one (`RC_PARAMS`) and you will see the same test configurations being run but with different input values. Set both and you will see the same configuration and the same input being used for a completely identical run.

### Enable SIMD instructions

On supported platforms, currently s390x only, numerical tests can generate SIMD instructions for the compiled models. To enable SIMD, set the `TEST_ARGS` environment variable, e.g.,
```
TEST_ARGS="-mcpu=z14" CTEST_PARALLEL_LEVEL=$(nproc) cmake --build . --config Release --target check-onnx-numerical
```

### Testing of specific accelerators

Currently we provide testing for accelerator NNPA. It is described [here](docs/AccelNNPAHowToUseAndTest.md).

## Use gdb
### Get source code for ONNX model
When you compile an ONNX model, add option `--preserveMLIR`. A source code for the  model in MLIR format, named your_model_name.input.mlir,  will be created. The line information for operation will be attached and propagated all the way to binary.
When you run the compiled library in gdb, you can stop in the model and step through with respect to the ONNX operations. Here is an example for model test_add.onnx:

```
$Debug/bin/onnx-mlir --preserveMLIR test_add.onnx
$. ../utils/build-run-onnx-lib.sh
$gdb Debug/bin/run-onnx-lib
(gdb) b run_main_graph
(gdb) run ./test_add.so
(gdb) list
1	builtin.module  {
2	  builtin.func @main_graph(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<3x4x5xf32> attributes {input_names = ["x", "y"], output_names = ["sum"]} {
3	    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xf32>
4	    return %0 : tensor<3x4x5xf32>
5	  }
(gdb) b 3
Breakpoint 2 at 0x3fffdf01778: file /home/chentong/onnx-mlir/build/test_add.input.mlir, line 3.
(gdb) c
Continuing.

Breakpoint 2, main_graph () at /home/chentong/onnx-mlir/build/test_add.input.mlir:3
3	    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xf32>
(gdb) n
[Detaching after vfork from child process 2333437]
#  0) before op=     Add VMem:  6804
[Detaching after vfork from child process 2333470]
#  1) after  op=     Add VMem:  6804
4	    return %0 : tensor<3x4x5xf32>
(gdb)
```
Note that the output of instrumentation showed that the gdb step at the onnx op level correctly. You need extra flags for onnx-mlir to run on instrumentation, which is not necessary for gdb. The source file is test_add.input.mlir.
One of furtuer works is to support symbols at onnx level in gdb. It would be really useful if tensors can be printed out in gdb.

## Use LLVM debug support

The standard way to add tracing code in the LLVM and MLIR projects is to use the LLVM_DEBUG macro. Official documentation from LLVM is [here](https://llvm.org/docs/ProgrammersManual.html#the-llvm-debug-macro-and-debug-option).

To insert a single "printout" under debug control, the following template can be used.
```C++
#define DEBUG_TYPE "my_opt_name_here"
...
LLVM_DEBUG(llvm::dbgs() << "debug msg here" <<  obj_to_print << "\n");
```
To trigger the debug trace one would simply invoke the compiler with --debug-only=my_opt_name_here.

Another macro called `DEBUG_WITH_TYPE` can be used situations where a source file has maybe just one tracing message. In that case you can forgo defining `DEBUG_TYPE` and use the following instead.

```C++
DEBUG_WITH_TYPE("my_debug_msg", llvm::dbgs() << "my trace msg here\n");
```
To protect larger portion of code, this template can be used.
```C++
LLVM_DEBUG({
  for(i...) {
    llvm::dbgs() << "emit trace for a: " << a << "\n";
    compute b;  // should be side effects free
    llvm::dbgs() << "emit trace for 'b':" << b << "\n";
    ...
});
```

Some examples that uses this support in the project are in these files.

* src/Conversion/KrnlToAffine/KrnlToAffine.cpp
* src/Conversion/ONNXToKrnl/Math/Gemm/Gemm.cpp

Again, these debug statements can then be activated by adding the `--debug-only=my_opt_name_here` option to `onnx-mlir` or `onnx-mlir-opt`.

## ONNX Model Zoo

We provide a Python script, i.e. [CheckONNXModelZoo.py](test/onnx-model-zoo/CheckONNXModelZoo.py), to check inference accuracy with models in the [ONNX model zoo](https://github.com/onnx/models). The script must be invoked from the ONNX model zoo repository, not the onnx-mlir repository.

Below is a quick start for checking the `mnist-8` model:
```bash
$ git clone https://github.com/onnx/models
$ cd models
$ ln -s /onnx_mlir/test/onnx-model/test/onnx-model-zoo/CheckONNXModelZoo.py CheckONNXModelZoo.py
$ ln -s /onnx_mlir/utils/RunONNXModel.py RunONNXModel.py
$ VERBOSE=1 ONNX_MLIR_HOME=/onnx-mlir/build/Release/ python CheckONNXModelZoo.py -pull-models -m mnist-8 -compile_args="-O3"
```
