<!--- SPDX-License-Identifier: Apache-2.0 -->

# How to support to process a new ONNX Operation

The text below describe the list of actions needed to successfully process a new ONNX operation. We are adding here support for an existing operation, thus when parsing the ONNX list of operation form the ONNX repository, this operation is already present. The step below explain how to add the necessary support to process that operation and lower it to code that can be executed using the LLVM compiler backend.

In the example below, we assume that we add support for the Concat ONNX operation. All paths are relatives to the root directory of onnx-mlir repository.


## Generate the proper ONNX.td.inc

The first step is to add support so that MLIR can determine the output type and shape from its input variables and parameters. This step is called “Shape Inference.” The first step is to check if the new operation needs support for  special handling, such as support for canonical forms or special parsing tools. If that is the case, the script that automatically generates the `ONNXOps.td.inc` file will need to be updated.  The script file is named [gen_onnx_mlir.py](../utils/gen_onnx_mlir.py). Detailed list of customization is described [here](ImportONNXDefs.md#customization).

Most operations have constraints on their input and parameters. The best way to test for these are in a verifier. Locate the array below in `gen_onnx_mlir.py` and add your new operation in it.
```
OpsWithVerifier = ['AveragePool', 'Conv', 'InstanceNormalization', 'Mod']
```

The next step will be to invoke the modified `gen_onnx_mlir.py` file. For this operation, consult the help [here](ImportONNXDefs.md).

## Add verifier

You will need to add code in the `src/Dialect/ONNX/ONNXOps.cpp` when the new op was declared as using a verifier.  Best is to look at other operations to get the general pattern, by searching for [static LogicalResult verify(ONNXInstanceNormalizationOp op)](../src/Dialect/ONNX/ONNXOps.cpp), for example. Note that a verifier will execute each time that one such op is created. So you will need to ensure that it can work with tensors and MemRefs, and possibly unranked tensors. So guard each of your tests to the proper circumstances. For examples, once a tensor is ranked, you may then verify that the rank is within the approved range (if there is such a constraint); before it is ranked, do not perform this test yet.

Tips:
* Use `operandAdaptor` object to get the inputs (must use  `operandAdaptor` to get the current values of the inputs) and the `op` object to get the attributes (can use `op` because attributes are typically immutable). 
* Use `hasShapeAndRank(X)` to test if `X` input is currently shaped and ranked. If not, return success as we will get a chance later to test the operation with this info. Note that some inputs may be scalar too, in which case they may or may not be encoded as a shape type.
* You can then use MLIR call `X.getType().cast<ShapedType>()` to get a shape types, for which you can get the rank and the dimensions. At this time, we only check dimension validity for values known at runtime. Unknown dimensions are encoded as a negative number. Please only use the cast when you are sure that it will not assert, i.e. the type is indeed a `ShapedType`.
* When you find an error, report it with a friendly error message using `op->emitError(msg)`.

## Add shape inference

You will need to add code in the `src/Dialect/ONNX/ONNXOps.cpp.` Best is to look at other operations to get the general pattern. In general, the function return `true` in case of success, and `false` otherwise. User input errors (such as unsupported type, unknown option,…) should be flagged to the user using `emitError` using a friendly explanation of the error.

If your operation has a parameter ` dilations`, then uses the function ` dilations()` to provide the current value of that parameter. Parameter values can be set too; for example, in the `concat` operation, negative `axis` indicates an axis where counting goes from right to left. In this case, we can choose to normalize all values of the parameter `axis` to be from the usual left to right direction. Use `axisAttr(<<new attribute value>>)` to set the `axis` to its expected value. 

For references, all variables and parameters have getter functions, all parameters have setter functions as well.

## Invoke shape inference and test it. 

Next step is to enable onnx-mlir to invoke the new shape inference for this new operation. This is done by adding the operation name in function ` returnsDynamicShape` in file `Transform/ONNX/ShapeInferencePass.cpp.`

Once it is invoked, you will need to add literal tests in ` test/mlir/onnx/onnx_shape_inference.mlir.` Tests are executed by the `make check-onnx-lit` command in the build directory or by a `build/bin/onnx-mlir-opt` with parameter listed in the first line of the corresponding test mlir file.

## Lowering to krnl dialect

Files related to the lowering of the new operations resides in the `src/Conversion/ONNXtoKRNL` directory and subdirectories. For the `concat` operation, we added code to lower it to krnl dialect in the `src/Conversion/ONNXToKrnl/Tensor/concat.cpp` file. See other similar lowering for inspiration. We suggest to use `assert` statements for any unexpected values encountered while lowering the operation, as illegal parameter values should be caught in the shape inference phase, not successive passes such as lowering to the krnl dialect.

In that file, the `populateLoweringONNXConcatOpPattern` operation (where `Concat` would be replaced with the actual new operation) will need to be defined in ` src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp` and invoked in the ` runOnOperation` function in the ` src/Conversion/ONNXToKrnl/ConvertONNXToKrnl.cpp` file.

To compile properly, you will also need to add the new `.cpp` file in the ` src/Conversion/ONNXToKrnl/CMakeLists.txt` file.

We recommend the use of KrnlBuilder class infrastructure to make the code more readable. This class is defined in `KrnlHelper.cpp` and examples of usage are found in an increasing number of files, check for example `MatMul.cpp`, `Normalization.cpp` or `Pooling.cpp`.

## Testing using ONNX backend tests

Locate the new operation’s test in the ` third_party/onnx/onnx/backend/test/case/node` test directory. You will need to deduce from the code the names of the test files generated by the python script. A simple way to locate the file for your operation is to perform a search of the ONNX Operation (without the prefix ONNX or the suffix Op) in the `onnx-mlir/third_party` directory, picking the `py` file under the `node` subdirectory. You will then need to add these strings in the ` test/backend/test.py` file.

When adding new tests in the `test.py` file, make sure to also include the appropriate dynamic tests, by setting the proper fields. For example, the string below:
```
    "test_erf_cpu": {STATIC_SHAPE:{}, DYNAMIC_SHAPE:{-1:{-1}}, CONSTANT_INPUT:{-1}},
```
enables a new test, `test_erf_cpu`, where all inputs may accommodate a fully dynamic input. See [here](Testing.md) for more details.

Tests are executed by the `make check-onnx-backend` command in the build directory. Additionally, `make check-onnx-backend-dynamic` and `make check-onnx-backend-constant` will further test the new operations with, respectively, dynamic and constant inputs.


