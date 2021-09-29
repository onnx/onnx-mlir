<!--- SPDX-License-Identifier: Apache-2.0 -->

# Generate Location info for operations

MLIR provides support to define and propagate location information of operations all the way to binary code. Such information can be used to track transformation along passes, or by binary tools such as gdb. 
Since onnx model is defined by protobuf file, we have first to create a source file. The source file can be dumped at different IR level, or at different pass point even for the same IR level. 
Current implementation supports only location info after the model is imported with builder.

## Location after model imported

The Location is annotated to onnx operations in Builder with estimiation of line number. `FileLineColLoc` is used with column vaule always 0. The file is generated withi the keepFile functionality after Builder. The name of this file is `*.input.mlir`.

## Turn on location info
By default, `UnKnowLoc` is used. To turn on location info after builder, command line option `--BuilderLoc` should be used.  Command line with test_add.onnx as an example is listed below.  

`Debug/bin/onnx-mlir --BuilderLoc test_add.onnx`.

 The location info annotated to operations will be line number in test_add.input.mlir.
To verify the location info, option `--preserveLocations` can be used to show location info in the source file.
With command `Debug/bin/onnx-mlir --BuilderLoc --preserveLocations test_add.onnx`, the test_add.input.mlir is as follows:

```
#loc1 = loc("test_add.input.mlir":2:0)
builtin.module  {
  builtin.func @main_graph(%arg0: tensor<3x4x5xf32> loc("test_add.input.mlir":2:0), %arg1: tensor<3x4x5xf32> loc("test_add.input.mlir":2:0)) -> tensor<3x4x5xf32> attributes {input_names = ["x", "y"], output_names = ["sum"]} {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xf32> loc(#loc2)
    return %0 : tensor<3x4x5xf32> loc(#loc3)
  } loc(#loc1)
  "onnx.EntryPoint"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [3 , 4 , 5] , \22name\22 : \22x\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [3 , 4 , 5] , \22name\22 : \22y\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [3 , 4 , 5] , \22name\22 : \22sum\22 }\0A\0A]\00"} : () -> () loc(#loc4)
} loc(#loc0)
#loc0 = loc(unknown)
#loc2 = loc("test_add.input.mlir":3:0)
#loc3 = loc("test_add.input.mlir":4:0)
#loc4 = loc("test_add.input.mlir":5:0)
```

## Use with gdb

If the model is compiled with location info turned on and instrumentation on, the source code will be shown in gdb. We can set break point, and step through at the source code level.
For example, we can use the driver built from `utils/build-run-onnx-lib.sh`.  
```
$Debug/bin/onnx-mlir --BuilderLoc
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
The output of instrumentation showed that the gdb step at the onnx op level correctly.

##Future work
1. The line number should be gained by inspecting the source code while the operations are traversed. 
2. Generate Location info at different places, such as Krnl.
3. Generate info for symbols so that tensors can be dumped in gdb. 

