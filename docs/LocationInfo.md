<!--- SPDX-License-Identifier: Apache-2.0 -->

# Maintain and Use Location Info in onnx-mlir

1. [Summary](#Summary)
2. [ONNX Model](#ONNX-model)
3. [MLIR File](#MLIR-file)

## Summary
Support of Location info propagation in transformation is one of the attractive features of MLIR. onnx-mlir can takes advantage of this feature in compiler transformation, and runtime debugging. This document describes how to maintain and use the location info in onnx-mlir. In summary:
- All onnx-mlir transformations are required to propagate the location info from the source to the target
- Create location info when an ONNX model is imported. If there is `onnx_node_name` string attribute for an operation, the string is transferred to its location. Otherwise, Unknown location is used.
- MLIR adds file location (in form of filename:line:column) to nodes when reading in a MLIR file, unless the MLIR file already contains location.
- Use the flag `--preserveLocations` to turn on location info in the output.
- With the previous two combined, we can track the source of error by dumping out the MLIR file(without `--preserveLocations`) at desired stage (for example, EmitONNXIR, or EmitMLIR), and then continuing transformation by loading the dumped file. The location info will be line number for that dumped file, providing more details than just from the onnx model. 

## ONNX model
When reading an ONNX model (.onnx file), onnx-mlir tries to attach location info to the generated IR. 
Some ONNX exporter annotates every operation with an StringAttr, "onnx_node_name", with an unique string for that operation. 
The importer of onnx-mlir converts the "onnx_node_name" attribute in the ONNX file tostring location info for the operation.
If the ONNX model does not have "onnx_node_name" attribute, Unknown location is attached.

### Example with onnx_node_name

This roberta  model is downloaded from onnx model zoo. Compile it with command
`onnx-mlir roberta-base-11.onnx --preserveLocations --EmitONNXBasic`.
The location info will be displayed in the output when the flag `--preserveLocations` is used.
In the output file(roberta-base-11.onnx.mlir), two nodes and their location info are shown below.

```
    %392 = "onnx.Sub"(%390, %391) {onnx_node_name = "Sub_109"} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32> loc(#loc393)
    %393 = onnx.Constant dense<2.000000e+00> : tensor<f32> loc(#loc394)

#loc393 = loc("Sub_109")
#loc394 = loc("Constant_110")
```
The 'onnx_node_name` attribute is only for operations, not for the constant. The location info for Constant is given by the importer.

### Example without onnx_node_name
The following model, test_add.onnx, came from onnx backend test. It is not from
onnx exporter and does not have `onnx_node_name` attribute.
The output of command `onnx-mlir test_add.onnx --preserveLocations --EmitONNXBasic`:

```
#loc = loc(unknown)

...
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xf32> loc(#loc)
...
```
There is no useful location info.

## MLIR file

MLIR automatically creates location info when the intermediate file (.mlir file) is read in as long as there is no location info in that .mlir file.  
For example, though there is no location info for the test_add.onnx, we can dump the importer result and load it again. Then we can find useful location info in the output.
Commands:
```
onnx-mlir test_add.onnx --EmitONNXBasic`
onnx-mlir test_add.onnx.mlir --preserveLocations --EmitONNXBasic`
```
...
Then location info can be found in the output of test_add.onnx.onnx.mlir
```
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xf32> loc(#loc4)
    onnx.Return %0 : tensor<3x4x5xf32> loc(#loc5)
...

#loc4 = loc("test_add.onnx.mlir":3:10)
#loc5 = loc("test_add.onnx.mlir":4:5)
```
The test_add.onnx.mlir content:

```
  1 module attributes {llvm.data_layout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-apple-darwin22.3.0", "onnx-mlir.symbol-postfix" = "test_add"} {
  2   func.func @main_graph(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<3x4x5xf32> {
  3     %0 = "onnx.Add"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xf32>
  4     onnx.Return %0 : tensor<3x4x5xf32>
  5   }
  6   "onnx.EntryPoint"() {func = @main_graph} : () -> ()
  7 }
```

If you want to track the operations for krnl IR, dump the file after lowering to krnl.

