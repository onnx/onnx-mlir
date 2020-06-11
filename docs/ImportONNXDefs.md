# Import ONNX specifications into ONNX-MLIR

ONNX specifications are defined under `onnx/defs` directory in the ONNX project repository. 
There is a python script onnx/defs/gen_onnx_mlir.py that automatically generate documents about operations in ONNX (docs/Operations.md). 
ONNX-MLIR modified this script to import ONNX specifications into ONNX-MLIR. 
There are two files generated for ONNX MLIR with the modified gen_onnx_mlir.py:

1. `src/Dialect/ONNX/ONNXOps.td.inc`: Operation definition for MLIR TableGen. `src/Dialect/ONNX/ONNXOps.td` includes this file.
2. `src/Builder/OpBuildTable.inc`: C++ code for ONNX-MLIR frontend to import operation nodes from ONNX model. `src/Builder/FrontendDialectTransformer.cpp` includes this file.

## How to use the script
1. Install [ONNX](https://github.com/onnx/onnx). We highly recommend that you use the one located at `third_party/onnx.`
2. Make target `OMONNXOpsIncTranslation`. For example,
```
make OMONNXOpsIncTranslation
````
Target `OMONNXOpsIncTranslation` invokes the script and places the generated files into the correct directories correspondingly.

## Consistency
For reference to the schema and semantics of an operation, please refer to [ONNX Dialect](Dialects/onnx.md). 
Even though we strive to support the latest version of ONNX specification as quickly as we can, there will inevitably be a delay between the introduction of new changes in the ONNX specification and the adoption in our codebase. 
Due to the possibility of such a delay, operator definition within the ONNX project repository may describe features and schemas that we do not yet support.

## Customization
In addition to following the ONNX specification, the script gen_onnx_mlir.py,  modified gen_onnx_mlir.py, provides some mechanism for you to customize the output. 
Several tables are defined at the beginning of the script:
1. `special_attr_defaults`: gives attribute special default value.
2. `special_op_handler`: creates special import function in frontend_dialect_transformer.cpp. Currently, a special handler is used for operations with operational arguments
3. `OpsWithShapeInference`: list of operations which have shape inference defined
4. `OpsWithCanonicalizer`: list of operations which have a canonical form
5. `OpsWithPromotableConstOperands`: list of operations which have operands that, if produced by constant operations, should be promoted to become an attribute (via attribute promotion)
6. `custom_builder_ops_list`: list of operations which need custom build methods to deduce result types

## Version of Operations
As stated previous, we try to support the latest version of ONNX operations. The version of each operation currently supported is recorded in gen_onnx_mlir.py. This mechanism provides some stability in version. To check the changes in version, run gen_onnx_mlir.py with flag "--check-version" and the changes will be reported. To move to a newer version, manually update the version dictionary in the script.
Supporting mulitple versions of one operation is not available yet.

